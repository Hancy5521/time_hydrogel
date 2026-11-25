"""Freeze and test device-budgeted graph-transition representations."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.linalg import solve
from scipy.spatial.distance import cdist
from scipy.special import expit

import capture as src


ROOT = Path(__file__).resolve().parent
METHODS = ("legacy_ridge", "legacy_rbf", "density_rbf", "enhanced_ridge", "enhanced_rbf", "transition_projection")


def load(role):
    with np.load(ROOT / (role.upper() + ".npz"), allow_pickle=False) as f:
        return f["x"], f["counts"]


def build_input(data, window, rates, mode):
    frames = src.WINDOWS[window]
    x, count = data
    selected = x[..., frames, :]
    static = np.stack((selected.mean(axis=3), selected.std(axis=3)), axis=-1).reshape(*x.shape[:2], -1)
    if mode == "legacy":
        return static
    transitions = [src.TRANSITIONS.index(t) for t in zip(frames[:-1], frames[1:])]
    merged = count[..., transitions, :, :].sum(axis=3)
    expo, events = merged[..., 0, :], merged[..., 1, :]
    density = np.log1p(expo).reshape(*x.shape[:2], -1)
    if mode == "density":
        return np.concatenate((static, density), axis=-1)
    signal = (events - expo * rates) / np.sqrt(1. + expo * rates * (1. - rates))
    count_raw = np.log1p(events).reshape(*x.shape[:2], -1)
    return np.concatenate((static, density, count_raw, signal.reshape(*x.shape[:2], -1)), axis=-1)


def rate_reference(data, window):
    frames = src.WINDOWS[window]
    times = [src.TRANSITIONS.index(t) for t in zip(frames[:-1], frames[1:])]
    c = data[1][..., times, :, :].sum(axis=(0, 1, 3))
    return (c[..., 1, :] + .5) / (c[..., 0, :] + 1.)


def to_rows(x):
    return x.reshape(-1, x.shape[-1])


def labels(n):
    return np.tile(np.array([0., 1.]), n)


def distance(x, y):
    return cdist(x, y, "sqeuclidean") / x.shape[-1]


def make_weights(train, other, algorithm):
    width = train.shape[-1]
    n = len(train)
    x = to_rows(train)
    ref = labels(n) - .5
    if algorithm == "rbf":
        d0, d1 = distance(x, x), distance(other, x)
        for gamma in (.20, .65, 2.):
            k0, k1 = np.exp(-gamma * d0), np.exp(-gamma * d1)
            for penalty in (.3, 3., 30.):
                dual = solve(k0 + penalty * np.eye(len(x)), ref, assume_a="pos")
                yield np.clip(.5 + k1 @ dual, .001, .999), {"dual": dual, "gamma": gamma,
                                                          "penalty": penalty, "support": x}
    elif algorithm == "ridge":
        for penalty in (1., 20., 200.):
            coef = solve(x.T @ x + penalty * np.eye(width), x.T @ ref, assume_a="pos")
            yield np.clip(.5 + other @ coef, .001, .999), {"coef": coef, "penalty": penalty}
    else:
        assert algorithm == "paired"
        midpoint = train.mean(axis=1)
        delta = (train[:, 1] - train[:, 0]) / 2.
        axis_ref = midpoint.mean(axis=0)
        u, v = midpoint - axis_ref, delta - delta.mean(axis=0)
        noise = (u.T @ u + v.T @ v) / n
        for ridge in (.2, 1., 5.):
            coef = solve(noise + ridge * np.eye(width), delta.mean(axis=0), assume_a="pos")
            for scale in (.25, 1., 4.):
                yield expit(((other - axis_ref) @ coef) * scale), {"coef": coef, "midpoint": axis_ref,
                         "ridge": ridge, "scale": scale}


def freeze():
    assert not (ROOT / "TEST.npz").exists()
    reg = src.read("REGISTERED.json")
    lock = src.read("MODEL_REGISTERED.json")
    assert reg["source_sha256"] == src.binding()
    assert lock["source_sha256"] == src.digest(Path(__file__))
    data = src.read("DEVELOPMENT.json")["raw_sha256"]
    for name, digest in data.items():
        assert src.digest(ROOT / name) == digest
    train, valid = (load(role) for role in ("train", "validate"))
    candidates = {"legacy_ridge": ("legacy", "ridge"), "legacy_rbf": ("legacy", "rbf"),
                  "density_rbf": ("density", "rbf"), "enhanced_ridge": ("enhanced", "ridge"),
                  "enhanced_rbf": ("enhanced", "rbf"), "transition_projection": ("enhanced", "paired")}
    best = {}
    per_window = {}
    ref = labels(src.ROLES["validate"])
    for window in src.WINDOWS:
        rates = rate_reference(train, window)
        for name, (mode, algorithm) in candidates.items():
            tx, vx = (build_input(d, window, rates, mode) for d in (train, valid))
            center, spread = to_rows(tx).mean(axis=0), to_rows(tx).std(axis=0)
            norm = np.maximum(.1 * np.median(spread[spread > 0]), spread)
            option = min(make_weights((tx - center) / norm, (to_rows(vx) - center) / norm, algorithm),
                         key=lambda p: float(np.mean((p[0] - ref) ** 2)))
            valid_brier = float(np.mean((option[0] - ref) ** 2))
            per_window[f"{name}:{window}"] = valid_brier
            if name not in best or valid_brier < best[name]["brier"] - 1.e-12:
                best[name] = {"brier": valid_brier, "window": window, "mode": mode, "algorithm": algorithm,
                              "rates": rates, "center": center, "norm": norm, **option[1]}
    numerical, metadata = {}, {}
    for name, entry in best.items():
        metadata[name] = {}
        for key, value in entry.items():
            if isinstance(value, np.ndarray):
                numerical[f"{name}_{key}"] = value
            else:
                metadata[name][key] = value
    src.arrays("MODEL.npz", **numerical)
    src.write("MODEL_FREEZE.json", {"model": metadata, "validation_brier": per_window,
               "registration_sha256": src.digest(ROOT / "REGISTERED.json"), "development_sha256": data,
               "model_source_sha256": src.digest(Path(__file__)),
               "model_arrays_sha256": src.digest(ROOT / "MODEL.npz"), "test_existed_at_freeze": False})


def predict(mode):
    freeze_doc = src.read("MODEL_FREEZE.json")
    confirmed = src.read("CONFIRMED.json")
    assert src.digest(ROOT / "MODEL_FREEZE.json") == confirmed["model_freeze"]
    assert src.digest(ROOT / "TEST.npz") == confirmed["raw_sha256"]["TEST.npz"]
    assert src.digest(ROOT / "MODEL.npz") == freeze_doc["model_arrays_sha256"]
    assert src.digest(Path(__file__)) == freeze_doc["model_source_sha256"]
    held = load("test")
    probabilities = {"prior": np.full((src.ROLES["test"], 2), .5)}
    with np.load(ROOT / "MODEL.npz", allow_pickle=False) as m:
        for name in METHODS:
            setup = freeze_doc["model"][name]
            def get(key):
                return m[f"{name}_{key}"]
            x = build_input(held, setup["window"], get("rates"), setup["mode"])
            x = (to_rows(x) - get("center")) / get("norm")
            if setup["algorithm"] == "rbf":
                p = .5 + np.exp(-setup["gamma"] * distance(x, get("support"))) @ get("dual")
            elif setup["algorithm"] == "ridge":
                p = .5 + x @ get("coef")
            else:
                p = expit(((x - get("midpoint")) @ get("coef")) * setup["scale"])
            probabilities[name] = np.clip(p, .001, .999).reshape(-1, 2)
    src.arrays("PREDICTED.npz", **probabilities)
    src.write("PREDICTION_FREEZE.json", {"model_freeze": confirmed["model_freeze"],
               "test_input_sha256": confirmed["raw_sha256"]["TEST.npz"],
               "predictions_sha256": src.digest(ROOT / "PREDICTED.npz")})


def summary(values, draws):
    ci = np.quantile(values[draws].mean(axis=1), [.025, .975])
    return {"mean": float(values.mean()), "ci95": ci.tolist(), "card": values.tolist()}


def score():
    seal = src.read("PREDICTION_FREEZE.json")
    assert seal["predictions_sha256"] == src.digest(ROOT / "PREDICTED.npz")
    assert seal["model_freeze"] == src.digest(ROOT / "MODEL_FREEZE.json")
    assert seal["test_input_sha256"] == src.digest(ROOT / "TEST.npz")
    scores = {}
    with np.load(ROOT / "PREDICTED.npz", allow_pickle=False) as q:
        for name in ("prior", *METHODS):
            p = q[name]
            lab = np.array([0., 1.])
            ok = np.where(p == .5, .5, ((p > .5) == lab)).mean(axis=1)
            gap = p[:, 1] - p[:, 0]
            scores[name] = {"accuracy": ok, "brier": ((p - lab) ** 2).mean(axis=-1),
                            "paired_rank": np.where(gap == 0., .5, gap > 0.)}
    draws = np.random.default_rng(445112).integers(src.ROLES["test"], size=(18000, src.ROLES["test"]))
    evaluated = {name: {measure: {"value": summary(v, draws),
                                  "vs_legacy_rbf": summary(v - scores["legacy_rbf"][measure], draws),
                                  "vs_enhanced_rbf": summary(v - scores["enhanced_rbf"][measure], draws),
                                  "vs_prior": summary(v - scores["prior"][measure], draws)}
                        for measure, v in stat.items()} for name, stat in scores.items()}
    src.write("SCORED.json", {"score_source_sha256": src.digest(Path(__file__)),
               "freeze_sha256": src.digest(ROOT / "PREDICTION_FREEZE.json"),
               "selected": src.read("MODEL_FREEZE.json")["model"], "models": evaluated})


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("action", choices=("freeze", "predict", "score"))
    opt = cli.parse_args().action
    if opt == "predict":
        predict(opt)
    else:
        (freeze if opt == "freeze" else score)()

