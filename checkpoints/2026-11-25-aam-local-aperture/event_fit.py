"""Fit and freeze conditional local risk-set inference with same-input controls."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.special import logsumexp, expit

import capture as src
import event_capture as event
from fit import labels, make_weights, summary, to_rows, distance


ROOT = Path(__file__).resolve().parent
MODELS = ("prior", "snapshot_rbf", "event_ridge", "event_rbf", "conditional_single",
          "conditional_marginal")
TYPE = np.array([-.5, 1., -.5, 1.])
SENS = TYPE * np.array([0., 0., 1., 1.])


def load(role):
    with np.load(ROOT / ("EVENT_" + role.upper() + ".npz"), allow_pickle=False) as values:
        m = values["marks"]
        x = values["x"]
    return m, x


def received(marks):
    risk = marks[..., :4]
    known = marks[..., 4] >= 0
    positive = np.where(known[..., None], risk, 0.)
    selected = np.maximum(0, marks[..., 4]).astype(int)
    if not np.all(np.take_along_axis(positive, selected[..., None], axis=-1)[..., 0][known] > 0):
        raise ValueError("Selected bin must carry positive known reference weight.")
    return positive, selected, known


def conditional(marks, marginal):
    risk, selected, mask = received(marks)
    valid = risk[mask]
    observed = selected[mask]
    # Flatten to one sample (one sign of a precursor) while summing its independent evidence.
    owners = np.broadcast_to(np.arange(marks.shape[0] * 2).reshape(*marks.shape[:2], 1, 1, 1), mask.shape)
    batch = owners[mask]
    if marginal:
        nodes, quadrature = np.polynomial.legendre.leggauss(25)
        b, omega = .48 * nodes, quadrature * .5
    else:
        b, omega = np.array([0.]), np.array([1.])
    forms = []
    for facing in (-1., 1.):
        logweight = b[:, None] * TYPE + facing * src.SLOPE * SENS
        denom = valid @ np.exp(logweight).T
        contrasts = logweight[:, observed].T - np.log(denom)
        by_card = np.zeros((marks.shape[0] * 2, len(omega)))
        np.add.at(by_card, batch, contrasts)
        forms.append(logsumexp(by_card + np.log(omega), axis=1))
    return forms[1] - forms[0]


def features(marks, x, mode):
    fixed = np.stack((x.mean(axis=3), x.std(axis=3)), axis=-1).reshape(*x.shape[:2], -1)
    if mode == "snapshot":
        return fixed
    risk, selected, known = received(marks)
    count = np.eye(4)[selected] * known[..., None]
    normal = np.maximum(risk.sum(axis=-1, keepdims=True), 1.e-99)
    prob = risk / normal
    heads = [v.sum(axis=(3, 4)).reshape(*x.shape[:2], -1) for v in
             (count, prob, count - prob, np.log1p(risk))]
    statistic = ((count - prob) @ SENS).sum(axis=(3, 4))
    info = ((prob @ (SENS**2) - (prob @ SENS)**2) * known).sum(axis=(3, 4))
    return np.concatenate([fixed, *heads, statistic, info], axis=-1)


def freeze():
    if (ROOT / "EVENT_CHECK.npz").exists():
        raise RuntimeError("Cannot tune after the new witness cards were collected.")
    data = src.read("EVENT_DEVELOPMENT.json")["data_sha256"]
    for name, digest in data.items():
        assert src.digest(ROOT / name) == digest
    registered = src.read("EVENT_MODEL_REGISTERED.json")
    assert registered["source_sha256"] == src.digest(Path(__file__))
    fit, dev = (load(role) for role in ("fit", "dev"))
    val_y = labels(event.GROUPS["dev"])
    arrays, settings = {}, {}
    for name, mode, kern in (("snapshot_rbf", "snapshot", "rbf"),
                             ("event_ridge", "events", "ridge"),
                             ("event_rbf", "events", "rbf")):
        train, val = (features(*raw, mode) for raw in (fit, dev))
        center, sd = to_rows(train).mean(axis=0), to_rows(train).std(axis=0)
        normal = np.maximum(sd, .1 * np.median(sd[sd > 0.]))
        p, config = min(make_weights((train - center) / normal, (to_rows(val) - center) / normal, kern),
                        key=lambda pick: np.mean((pick[0] - val_y) ** 2))
        settings[name] = {"validation_brier": float(np.mean((p - val_y) ** 2)), "mode": mode, "kernel": kern}
        for k, v in {**config, "center": center, "normal": normal}.items():
            if isinstance(v, np.ndarray):
                arrays[f"{name}_{k}"] = v
            else:
                settings[name][k] = v
    for name, nuisance in (("conditional_single", False), ("conditional_marginal", True)):
        log_ratio = conditional(dev[0], nuisance)
        temperature = min((.35, .65, 1., 1.5, 2.), key=lambda t: np.mean((expit(t * log_ratio) - val_y)**2))
        settings[name] = {"alpha": temperature, "marginal": nuisance,
                          "validation_brier": float(np.mean((expit(temperature * log_ratio) - val_y)**2))}
    src.arrays("EVENT_MODELS.npz", **arrays)
    src.write("EVENT_MODEL_FREEZE.json", {"registration_sha256": src.digest(ROOT / "EVENT_REGISTERED.json"),
              "models_sha256": src.digest(ROOT / "EVENT_MODELS.npz"),
              "source_sha256": src.digest(Path(__file__)), "models": settings, "dev_sha256": data})


def predict():
    conf = src.read("EVENT_CONFIRMED.json")
    assert conf["freeze_sha256"] == src.digest(ROOT / "EVENT_MODEL_FREEZE.json")
    assert conf["data_sha256"]["EVENT_CHECK.npz"] == src.digest(ROOT / "EVENT_CHECK.npz")
    lock = src.read("EVENT_MODEL_FREEZE.json")
    assert lock["models_sha256"] == src.digest(ROOT / "EVENT_MODELS.npz")
    assert lock["source_sha256"] == src.digest(Path(__file__))
    held = load("check")
    n = event.GROUPS["check"]
    results = {"prior": np.full((n, 2), .5)}
    with np.load(ROOT / "EVENT_MODELS.npz", allow_pickle=False) as model:
        for name, entry in lock["models"].items():
            def get(tag):
                return model[f"{name}_{tag}"]
            if "kernel" in entry:
                x = (to_rows(features(*held, entry["mode"])) - get("center")) / get("normal")
                if entry["kernel"] == "rbf":
                    v = np.exp(-entry["gamma"] * distance(x, get("support"))) @ get("dual")
                else:
                    v = x @ get("coef")
                p = np.clip(.5 + v, .001, .999)
            else:
                p = expit(entry["alpha"] * conditional(held[0], entry["marginal"]))
            results[name] = p.reshape(n, 2)
    src.arrays("EVENT_PREDICTED.npz", **results)
    src.write("EVENT_PREDICTION_FREEZE.json", {"model_freeze_sha256": conf["freeze_sha256"],
               "data_sha256": conf["data_sha256"], "probabilities_sha256": src.digest(ROOT / "EVENT_PREDICTED.npz")})


def score():
    lock = src.read("EVENT_PREDICTION_FREEZE.json")
    assert lock["probabilities_sha256"] == src.digest(ROOT / "EVENT_PREDICTED.npz")
    stats = {}
    with np.load(ROOT / "EVENT_PREDICTED.npz", allow_pickle=False) as file:
        for name in MODELS:
            p = file[name]
            y = np.array([0., 1.])
            accuracy = np.where(p == .5, .5, (p > .5) == y).mean(axis=-1)
            rank = np.where(p[:, 1] == p[:, 0], .5, p[:, 1] > p[:, 0])
            brier = np.square(p - y).mean(axis=-1)
            stats[name] = {"accuracy": accuracy, "paired_rank": rank, "brier": brier}
    n = event.GROUPS["check"]
    draw = np.random.default_rng(1944402).integers(n, size=(20000, n))
    rows = {name: {metric: {"value": summary(v, draw),
                             "vs_prior": summary(v - stats["prior"][metric], draw),
                             "vs_snapshot": summary(v - stats["snapshot_rbf"][metric], draw),
                             "vs_event_rbf": summary(v - stats["event_rbf"][metric], draw)}
                   for metric, v in values.items()} for name, values in stats.items()}
    marks, _ = load("check")
    counts = (marks[..., 4] >= 0).sum(axis=(2, 3, 4))
    src.write("EVENT_SCORED.json", {"prediction_freeze_sha256": src.digest(ROOT / "EVENT_PREDICTION_FREEZE.json"),
              "results": rows, "captures_total": int(counts.sum()),
              "witness_groups_with_signal": int(np.count_nonzero(counts)), "counts": counts.tolist()})


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("stage", choices=("freeze", "predict", "score"))
    {"freeze": freeze, "predict": predict, "score": score}[cli.parse_args().stage]()

