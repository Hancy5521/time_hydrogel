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
MODELS = ("prior", "old_summary", "proxy_rbf", "proxy_conditional", "full_conditional")


def cohort(role):
    n = SETS[role]
    return (src.rng("new-proxy-" + role, "code").uniform(-.48, .48, (n, 4)),
            src.rng("new-proxy-" + role, "local addresses").integers(0, 2**63 - 1, (n, 2, 4)))


def binding():
    paths = ("proxy.py", "PROXY_PROTOCOL_ZH.md", "proxy_geometry.py", "proxy_kernels.py",
             "generate_proxy_driver.py", "generate_proxy.py", "mobile_capture.py", "mobile_fit.py", "MOBILE_MODEL_FREEZE.json",
             "MINIMAL_RESULT.json", "REPLICATION_RESULT.json", "REPLICATION_C_INDEPENDENT_AUDIT.json")
    return {**mobile.bindings(), **{str(ROOT / f): src.digest(ROOT / f) for f in paths}}


def register():
    fresh = np.concatenate([cohort(key)[1].ravel() for key in SETS])
    assert len(np.unique(fresh)) == len(fresh)
    for f in ROOT.glob("*.npz"):
        with np.load(f, allow_pickle=False) as p:
            for field in ("addresses", "seeds"):
                if field in p.files:
                    assert not np.intersect1d(fresh, p[field]).size
    src.write("PROXY_REGISTERED.json", {"binding": binding(), "cards": SETS, "scope": "independent geometry proxy",
              "same_sensor_high_resolution_limit": 3, "noninferiority_brier": .01})


def prep():
    mobile.setup()
    import proxy_kernels
    import mobile_kernels
    mobile_kernels.advance_mobile = proxy_kernels.advance_proxy


def acquire(stage, workers):
    assert binding() == src.read("PROXY_REGISTERED.json")["binding"]
    hold = stage == "held"
    if hold:
        lock = src.read("PROXY_MODEL_FREEZE.json")
        assert lock["model_sha256"] == src.digest(ROOT / "PROXY_MODELS.npz")
    hashes = {}
    with ProcessPoolExecutor(max_workers=workers, initializer=prep) as pool:
        for key in (("held",) if hold else ("fit", "development")):
            g, addr = cohort(key)
            args = ((z, env, rid, slope, False) for z, row in zip(g, addr, strict=True)
                    for slope in (-1., 1.) for env, r in zip(("A", "B"), row, strict=True) for rid in r)
            all_marks, x, lite = map(np.asarray, zip(*pool.map(mobile.evidence, args, chunksize=14), strict=True))
            shape = (len(g), 2, 2, 4)
            name = "PROXY_" + key.upper() + ".npz"
            src.arrays(name, full=all_marks.reshape(*shape, 3, 6), x=x.reshape(*shape, 3, 14),
                       lite=lite.reshape(*shape, 3, 6), seeds=addr)
            hashes[name] = src.digest(ROOT / name)
            print("minimal", key, int((all_marks[..., 4] >= 0).sum()), int((lite[..., 4] >= 0).sum()), flush=True)
    src.write("PROXY_HELD.json" if hold else "PROXY_DATA.json", {"hashes": hashes,
               "freeze_sha256": src.digest(ROOT / "PROXY_MODEL_FREEZE.json") if hold else None})


def load(key):
    with np.load(ROOT / ("PROXY_" + key.upper() + ".npz"), allow_pickle=False) as data:
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
    return full.features(m, x, "events")


def freeze():
    assert not (ROOT / "PROXY_HELD.npz").exists()
    assert binding() == src.read("PROXY_REGISTERED.json")["binding"]
    hashes = src.read("PROXY_DATA.json")["hashes"]
    assert all(src.digest(ROOT / n) == value for n, value in hashes.items())
    raw, check = (load(role) for role in ("fit", "development"))
    dev_labels = labels(SETS["development"])
    lr = full.conditional(check[2], True)
    scale = min((.25, .65, 1., 1.5, 2.), key=lambda s: np.mean((expit(lr*s)-dev_labels)**2))
    data, char = {}, {"mark_scale": scale}
    for key, mode in (("old_summary", "snapshot"), ("proxy_rbf", "counts")):
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
    src.arrays("PROXY_MODELS.npz", **data)
    src.write("PROXY_MODEL_FREEZE.json", {"model_sha256": src.digest(ROOT / "PROXY_MODELS.npz"),
              "bindings": binding(), "settings": char})


def predict():
    info = src.read("PROXY_HELD.json")
    assert info["freeze_sha256"] == src.digest(ROOT / "PROXY_MODEL_FREEZE.json")
    assert info["hashes"]["PROXY_HELD.npz"] == src.digest(ROOT / "PROXY_HELD.npz")
    freeze = src.read("PROXY_MODEL_FREEZE.json")
    assert freeze["bindings"] == binding()
    assert freeze["model_sha256"] == src.digest(ROOT / "PROXY_MODELS.npz")
    raw, x, m = load("held")
    n = SETS["held"]
    which = freeze["settings"]
    alpha = src.read("MOBILE_MODEL_FREEZE.json")["models"]["conditional_marginal"]["alpha"]
    result = {"prior": np.full((n, 2), .5),
              "full_conditional": expit(alpha * full.conditional(raw, True)).reshape(n, 2)}
    with np.load(ROOT / "PROXY_MODELS.npz", allow_pickle=False) as bank:
        result["proxy_conditional"] = expit(which["mark_scale"] * full.conditional(m, True)).reshape(n, 2)
        for model in ("old_summary", "proxy_rbf"):
            cfg = which[model]
            data = to_rows(x_form(x, m, cfg["mode"]))
            flat = (data - bank[f"{model}_center"]) / bank[f"{model}_std"]
            gram = np.exp(-cfg["gamma"] * distance(flat, bank[f"{model}_support"]))
            p = .5 + gram @ bank[f"{model}_dual"]
            result[model] = np.clip(p, .001, .999).reshape(n, 2)
    src.arrays("PROXY_PREDICTED.npz", **result)
    src.write("PROXY_PREDICTION_FREEZE.json", {"probabilities_sha256": src.digest(ROOT / "PROXY_PREDICTED.npz"),
               "model_freeze_sha256": info["freeze_sha256"]})


def score():
    freeze = src.read("PROXY_PREDICTION_FREEZE.json")
    assert freeze["probabilities_sha256"] == src.digest(ROOT / "PROXY_PREDICTED.npz")
    n = SETS["held"]
    bootstrap = np.random.default_rng(840062).integers(n, size=(18000, n))
    label = np.array([0., 1.])
    scores = {}
    with np.load(ROOT / "PROXY_PREDICTED.npz", allow_pickle=False) as packet:
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
    src.write("PROXY_RESULT.json", {"frozen_probability": freeze["probabilities_sha256"],
              "models": scores, "marks_full_then_proxy": counts,
              "passes_gate": {k: passes(k) for k in ("proxy_rbf", "proxy_conditional")}})


if __name__ == "__main__":
    a = argparse.ArgumentParser(description=__doc__)
    a.add_argument("stage", choices=("register", "dev", "freeze", "held", "predict", "score"))
    a.add_argument("--workers", type=int, default=10)
    v = a.parse_args()
    if v.stage in ("dev", "held"):
        acquire(v.stage, v.workers)
    else:
        {"register": register, "freeze": freeze, "predict": predict, "score": score}[v.stage]()

