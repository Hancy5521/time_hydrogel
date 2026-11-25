"""Blind comparison of reduced observation and fully calibrated local candidate risks."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy.special import expit

import capture as src
import mobile_capture as mobile
import mobile_fit as full
from fit import make_weights, labels, summary, to_rows, distance


ROOT = Path(__file__).resolve().parent
SETS = {"fit": 96, "development": 48, "held": 80}
MODELS = ("prior", "old_summary", "count_rbf", "marked_counts", "full_conditional")


def cohort(role):
    n = SETS[role]
    return (src.rng("new-minimal-" + role, "code").uniform(-.48, .48, (n, 4)),
            src.rng("new-minimal-" + role, "local addresses").integers(0, 2**63 - 1, (n, 2, 4)))


def binding():
    paths = ("minimal.py", "MINIMAL_PROTOCOL_ZH.md", "minimal_geometry.py", "minimal_kernels.py",
             "generate_minimal.py", "mobile_capture.py", "mobile_fit.py", "MOBILE_MODEL_FREEZE.json",
             "REPLICATION_RESULT.json", "REPLICATION_C_INDEPENDENT_AUDIT.json")
    return {**mobile.bindings(), **{str(ROOT / f): src.digest(ROOT / f) for f in paths}}


def register():
    fresh = np.concatenate([cohort(key)[1].ravel() for key in SETS])
    assert len(np.unique(fresh)) == len(fresh)
    for f in ROOT.glob("*.npz"):
        with np.load(f, allow_pickle=False) as p:
            for field in ("addresses", "seeds"):
                if field in p.files:
                    assert not np.intersect1d(fresh, p[field]).size
    src.write("MINIMAL_REGISTERED.json", {"binding": binding(), "cards": SETS, "scope": "calibration ablation",
              "same_sensor_high_resolution_limit": 3, "noninferiority_brier": .01})


def prep():
    mobile.setup()
    import minimal_kernels
    import mobile_kernels
    mobile_kernels.advance_mobile = minimal_kernels.advance_minimal


def acquire(stage, workers):
    assert binding() == src.read("MINIMAL_REGISTERED.json")["binding"]
    hold = stage == "held"
    if hold:
        lock = src.read("MINIMAL_MODEL_FREEZE.json")
        assert lock["model_sha256"] == src.digest(ROOT / "MINIMAL_MODELS.npz")
    hashes = {}
    with ProcessPoolExecutor(max_workers=workers, initializer=prep) as pool:
        for key in (("held",) if hold else ("fit", "development")):
            g, addr = cohort(key)
            args = ((z, env, rid, slope, False) for z, row in zip(g, addr, strict=True)
                    for slope in (-1., 1.) for env, r in zip(("A", "B"), row, strict=True) for rid in r)
            all_marks, x, lite = map(np.asarray, zip(*pool.map(mobile.evidence, args, chunksize=14), strict=True))
            shape = (len(g), 2, 2, 4)
            name = "MINIMAL_" + key.upper() + ".npz"
            src.arrays(name, full=all_marks.reshape(*shape, 3, 6), x=x.reshape(*shape, 3, 14),
                       lite=lite.reshape(*shape, 3, 6), seeds=addr)
            hashes[name] = src.digest(ROOT / name)
            print("minimal", key, int((all_marks[..., 4] >= 0).sum()), int((lite[..., 4] >= 0).sum()), flush=True)
    src.write("MINIMAL_HELD.json" if hold else "MINIMAL_DATA.json", {"hashes": hashes,
               "freeze_sha256": src.digest(ROOT / "MINIMAL_MODEL_FREEZE.json") if hold else None})


def load(key):
    with np.load(ROOT / ("MINIMAL_" + key.upper() + ".npz"), allow_pickle=False) as data:
        return tuple(data[k] for k in ("full", "x", "lite"))


def count(marks):
    y = marks[..., 4]
    pair = np.maximum(y, 0.).astype(int)
    clock_bin = (marks[..., 5] > 3.45).astype(int)
    return (np.eye(8)[4 * clock_bin + pair] * (y >= 0)[..., None]).sum(axis=(3, 4))


def x_form(x, m, mode):
    props = full.features(m, x, "snapshot")
    if mode == "snapshot":
        return props
    seen = count(m).reshape(*x.shape[:2], -1)
    return np.concatenate((props, np.log1p(seen)), axis=-1)


def marked_prob(chosen, log_freq, tau):
    values = chosen.reshape(-1, 2, 2, 4)
    ll = (values * (log_freq[1] - log_freq[0])).sum(axis=(1, 2, 3))
    return expit(ll * tau)


def freeze():
    assert not (ROOT / "MINIMAL_HELD.npz").exists()
    assert binding() == src.read("MINIMAL_REGISTERED.json")["binding"]
    hashes = src.read("MINIMAL_DATA.json")["hashes"]
    assert all(src.digest(ROOT / n) == value for n, value in hashes.items())
    raw, check = (load(role) for role in ("fit", "development"))
    dev_labels = labels(SETS["development"])
    directions = count(raw[2]).sum(axis=0).reshape(2, 2, 2, 4) + .75
    freqs = directions / directions.sum(axis=-1, keepdims=True)
    mark_log = np.log(freqs)
    scale = min((.25, .65, 1., 1.5, 2.), key=lambda s: np.mean((marked_prob(count(check[2]), mark_log, s) - dev_labels)**2))
    data, char = {"mark_log": mark_log}, {"mark_scale": scale}
    for key, mode in (("old_summary", "snapshot"), ("count_rbf", "counts")):
        tx, vx = (x_form(x, m, mode) for _, x, m in (raw, check))
        center, std = to_rows(tx).mean(axis=0), to_rows(tx).std(axis=0)
        safe = np.maximum(std, .1 * np.median(std[std > 0]))
        chosen = min(make_weights((tx - center) / safe, (to_rows(vx) - center) / safe, "rbf"),
                     key=lambda p: np.mean((p[0] - dev_labels)**2))
        char[key] = {"mode": mode, "dev_brier": float(np.mean((chosen[0] - dev_labels)**2))}
        for field, value in {**chosen[1], "center": center, "std": safe}.items():
            if isinstance(value, np.ndarray):
                data[f"{key}_{field}"] = value
            else:
                char[key][field] = value
    src.arrays("MINIMAL_MODELS.npz", **data)
    src.write("MINIMAL_MODEL_FREEZE.json", {"model_sha256": src.digest(ROOT / "MINIMAL_MODELS.npz"),
              "bindings": binding(), "settings": char})


def predict():
    info = src.read("MINIMAL_HELD.json")
    assert info["freeze_sha256"] == src.digest(ROOT / "MINIMAL_MODEL_FREEZE.json")
    assert info["hashes"]["MINIMAL_HELD.npz"] == src.digest(ROOT / "MINIMAL_HELD.npz")
    freeze = src.read("MINIMAL_MODEL_FREEZE.json")
    assert freeze["bindings"] == binding()
    assert freeze["model_sha256"] == src.digest(ROOT / "MINIMAL_MODELS.npz")
    raw, x, m = load("held")
    n = SETS["held"]
    which = freeze["settings"]
    alpha = src.read("MOBILE_MODEL_FREEZE.json")["models"]["conditional_marginal"]["alpha"]
    result = {"prior": np.full((n, 2), .5),
              "full_conditional": expit(alpha * full.conditional(raw, True)).reshape(n, 2)}
    with np.load(ROOT / "MINIMAL_MODELS.npz", allow_pickle=False) as bank:
        result["marked_counts"] = marked_prob(count(m), bank["mark_log"], which["mark_scale"]).reshape(n, 2)
        for model in ("old_summary", "count_rbf"):
            cfg = which[model]
            data = to_rows(x_form(x, m, cfg["mode"]))
            flat = (data - bank[f"{model}_center"]) / bank[f"{model}_std"]
            gram = np.exp(-cfg["gamma"] * distance(flat, bank[f"{model}_support"]))
            p = .5 + gram @ bank[f"{model}_dual"]
            result[model] = np.clip(p, .001, .999).reshape(n, 2)
    src.arrays("MINIMAL_PREDICTED.npz", **result)
    src.write("MINIMAL_PREDICTION_FREEZE.json", {"probabilities_sha256": src.digest(ROOT / "MINIMAL_PREDICTED.npz"),
               "model_freeze_sha256": info["freeze_sha256"]})


def score():
    freeze = src.read("MINIMAL_PREDICTION_FREEZE.json")
    assert freeze["probabilities_sha256"] == src.digest(ROOT / "MINIMAL_PREDICTED.npz")
    n = SETS["held"]
    bootstrap = np.random.default_rng(840062).integers(n, size=(18000, n))
    label = np.array([0., 1.])
    scores = {}
    with np.load(ROOT / "MINIMAL_PREDICTED.npz", allow_pickle=False) as packet:
        brier = {k: ((packet[k] - label)**2).mean(axis=1) for k in MODELS}
        for key in MODELS:
            p = packet[key]
            a = np.where(p == .5, .5, (p > .5) == label).mean(axis=-1)
            scores[key] = {"accuracy": summary(a, bootstrap), "brier": summary(brier[key], bootstrap),
                           "vs_full_conditional_brier": summary(brier[key]-brier["full_conditional"], bootstrap)}
    raw, _, lite = load("held")
    counts = [int((m[..., 4] >= 0).sum()) for m in (raw, lite)]
    def passes(key):
        v = scores[key]
        return v["accuracy"]["mean"] >= .55 and v["accuracy"]["ci95"][0] > .5 and (
            v["vs_full_conditional_brier"]["ci95"][1] < .010)
    src.write("MINIMAL_RESULT.json", {"frozen_probability": freeze["probabilities_sha256"],
              "models": scores, "marks_full_then_minimal": counts,
              "passes_gate": {k: passes(k) for k in ("count_rbf", "marked_counts")}})


if __name__ == "__main__":
    a = argparse.ArgumentParser(description=__doc__)
    a.add_argument("stage", choices=("register", "dev", "freeze", "held", "predict", "score"))
    a.add_argument("--workers", type=int, default=10)
    v = a.parse_args()
    if v.stage in ("dev", "held"):
        acquire(v.stage, v.workers)
    else:
        {"register": register, "freeze": freeze, "predict": predict, "score": score}[v.stage]()

