"""Recheck frozen local-event inference on entirely fresh precursor cards and process addresses."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy.spatial.distance import cdist
from scipy.special import expit

import capture as src
import mobile_capture as sensor
import mobile_fit as fixed
import forecast
from fit import summary, distance, to_rows


ROOT = Path(__file__).resolve().parent
CARDS, REPEATS = 224, 16
METHODS = ("constant_four", "equal_mixture", "fixed_sensor", "snapshot_rbf",
           "mobile_rbf", "mobile_conditional", "known_direction")


def cohorts():
    return src.rng("resolved_binary_replication", "ground_code").uniform(-.48, .48, (CARDS, 4))


def seeds(stage):
    dimensions = (CARDS, 2, 4) if stage == "witness" else (CARDS, REPEATS)
    return src.rng("resolved_binary_replication", stage).integers(0, 2**63 - 1, dimensions)


def bindings():
    assert src.digest(forecast.NATIVE_SETTINGS) == forecast.EXPECTED_NATIVE_SHA
    files = ("replication.py", "REPLICATION_PROTOCOL_ZH.md", "MOBILE_MODEL_FREEZE.json",
             "MOBILE_MODELS.npz", "mobile_capture.py", "mobile_fit.py", "mobile_kernels.py",
             "mobile_geometry.py", "local_observer.py", "forecast.py", "MOBILE_REGISTERED.json",
             "MOBILE_C_RESULT.json", "MOBILE_C_INDEPENDENT_AUDIT.json")
    return {**src.binding(), **{str(ROOT / name): src.digest(ROOT / name) for name in files}}


def register():
    current = np.concatenate([seeds(role).ravel() for role in ("witness", "forecast", "truth")])
    assert np.unique(current).size == current.size
    for arr in ROOT.glob("*.npz"):
        with np.load(arr, allow_pickle=False) as sample:
            for key in ("seeds", "addresses"):
                if key in sample.files:
                    assert not np.intersect1d(current, sample[key]).size
    src.write("REPLICATION_REGISTERED.json", {"sources_sha256": bindings(), "cards": CARDS,
              "repeat_per_C_role": REPEATS, "registered_addresses": current.size,
              "new_fit": False, "near_strong_margin": 0.00075})


def verify():
    assert src.read("REPLICATION_REGISTERED.json")["sources_sha256"] == bindings()


def observe(workers):
    verify()
    addr = seeds("witness")
    jobs = ((g, env, rid, sign, False) for g, row in zip(cohorts(), addr, strict=True)
            for sign in (-1., 1.) for env, group in zip(("A", "B"), row, strict=True) for rid in group)
    with ProcessPoolExecutor(max_workers=workers, initializer=sensor.setup) as pool:
        out = list(pool.map(sensor.evidence, jobs, chunksize=16))
    mobile, x, stationary = map(np.asarray, zip(*out, strict=True))
    shape = (CARDS, 2, 2, 4)
    src.arrays("REPLICATION_WITNESS.npz", marks=mobile.reshape(*shape, 3, 6),
              x=x.reshape(*shape, 3, 14), fixed=stationary.reshape(*shape, 3, 6), seeds=addr)
    src.write("REPLICATION_WITNESS.json", {"data_sha256": src.digest(ROOT / "REPLICATION_WITNESS.npz"),
               "witness_paths": len(out)})


def freeze():
    verify()
    witness = src.read("REPLICATION_WITNESS.json")
    assert witness["data_sha256"] == src.digest(ROOT / "REPLICATION_WITNESS.npz")
    model = src.read("MOBILE_MODEL_FREEZE.json")
    assert model["models_sha256"] == src.digest(ROOT / "MOBILE_MODELS.npz")
    with np.load(ROOT / "REPLICATION_WITNESS.npz", allow_pickle=False) as obs:
        marks, x, stationary = (obs[key] for key in ("marks", "x", "fixed"))
    alpha = model["models"]["conditional_marginal"]["alpha"]
    prob = {"mobile_conditional": expit(alpha * fixed.conditional(marks, True)).reshape(CARDS, 2),
            "fixed_sensor": expit(alpha * fixed.conditional(stationary, True)).reshape(CARDS, 2)}
    with np.load(ROOT / "MOBILE_MODELS.npz", allow_pickle=False) as weights:
        for mode, baseline in (("snapshot_rbf", "snapshot_rbf"), ("mobile_rbf", "event_rbf")):
            cfg = model["models"][baseline]
            def get(key):
                return weights[f"{baseline}_{key}"]
            v = fixed.features(marks, x, cfg["mode"])
            p = np.exp(-cfg["gamma"] * distance((to_rows(v) - get("center")) / get("normal"), get("support"))) @ get("dual")
            prob[mode] = np.clip(.5 + p, .001, .999).reshape(CARDS, 2)
    assert not (ROOT / "REPLICATION_TRUTH.npz").exists()
    src.arrays("REPLICATION_PROBABILITIES.npz", **prob)
    src.write("REPLICATION_PREDICTIONS.json", {"sha256": src.digest(ROOT / "REPLICATION_PROBABILITIES.npz"),
              "witness_sha256": witness["data_sha256"], "source_from_prior_before_hold": True})


def generate(role, workers):
    verify()
    odds_lock = src.read("REPLICATION_PREDICTIONS.json")
    assert odds_lock["sha256"] == src.digest(ROOT / "REPLICATION_PROBABILITIES.npz")
    if role == "truth":
        native_lock = src.read("REPLICATION_C_FREEZE.json")
        assert native_lock["sha256"] == src.digest(ROOT / "REPLICATION_FORECAST.npz")
        assert native_lock["probability_sha256"] == odds_lock["sha256"]
    plan = src.read(str(forecast.NATIVE_SETTINGS))
    action = (("schedule", plan["selected"][0]), ("pendant", plan["selected"][2]))
    codes = (0, -1, 1) if role == "forecast" else (-1, 1)
    addr = seeds(role)
    items = ((z, rid, mode, spec, sign) for z, rr in zip(cohorts(), addr, strict=True)
             for mode, spec in action for sign in codes for rid in rr)
    result = []
    with ProcessPoolExecutor(max_workers=workers, initializer=forecast.config) as pool:
        for i, item in enumerate(pool.map(forecast.work, items, chunksize=16), 1):
            result.append(item)
            if i % 3072 == 0:
                print("replicate", role, i, flush=True)
    shape = (CARDS, 2, len(codes), REPEATS, 6, 10)
    file = "REPLICATION_" + role.upper() + ".npz"
    src.arrays(file, path=np.asarray(result).reshape(shape), seeds=addr)
    src.write("REPLICATION_C_FREEZE.json" if role == "forecast" else "REPLICATION_C_CONFIRM.json",
              {"sha256": src.digest(ROOT / file), "probability_sha256": odds_lock["sha256"],
               "native_paths": len(result), "bank": (src.digest(ROOT / "REPLICATION_C_FREEZE.json") if role == "truth" else None)})


def score():
    verify()
    conf, lock = (src.read(name) for name in ("REPLICATION_C_CONFIRM.json", "REPLICATION_C_FREEZE.json"))
    assert conf["bank"] == src.digest(ROOT / "REPLICATION_C_FREEZE.json")
    assert conf["sha256"] == src.digest(ROOT / "REPLICATION_TRUTH.npz")
    assert lock["sha256"] == src.digest(ROOT / "REPLICATION_FORECAST.npz")
    assert lock["probability_sha256"] == src.digest(ROOT / "REPLICATION_PROBABILITIES.npz")
    with np.load(ROOT / "REPLICATION_TRUTH.npz", allow_pickle=False) as final:
        actual = forecast.feature(final["path"])
    with np.load(ROOT / "REPLICATION_FORECAST.npz", allow_pickle=False) as predicted:
        bank = forecast.feature(predicted["path"])
    with np.load(ROOT / "REPLICATION_PROBABILITIES.npz", allow_pickle=False) as learned:
        p = {name: learned[name] for name in learned.files}
    rows = {}
    stats = np.zeros((len(METHODS), CARDS, 2, 2))
    w = np.zeros((len(METHODS), CARDS, 2, 3))
    w[0, :, :, 0], w[1, :, :, 1:] = 1., .5
    w[-1, :, 0, 1], w[-1, :, 1, 2] = 1., 1.
    for name, prob in p.items():
        w[METHODS.index(name), :, :, 1] = 1. - prob
        w[METHODS.index(name), :, :, 2] = prob
    assert np.allclose(w.sum(axis=-1), 1.)
    use = ~np.eye(REPEATS, dtype=bool)
    for c in range(CARDS):
        for a in range(2):
            cells = bank[c, a]
            within = np.array([[cdist(k, l)[use].mean() for l in cells] for k in cells])
            for sign in range(2):
                rates = np.array([cdist(actual[c, a, sign], v).mean() for v in cells])
                mixture = w[:, c, sign]
                stats[:, c, a, sign] = mixture @ rates - .5 * np.sum((mixture @ within) * mixture, axis=-1)
    boot = np.random.default_rng(319684).integers(CARDS, size=(22000, CARDS))
    score_vec = stats.mean(axis=(2, 3))
    for index, name in enumerate(METHODS):
        v = score_vec[index]
        rows[name] = {"energy": summary(v, boot), "vs_four": summary(v - score_vec[0], boot),
                      "vs_same_input_rbf": summary(v - score_vec[METHODS.index("mobile_rbf")], boot)}
    control = rows["mobile_conditional"]
    src.write("REPLICATION_RESULT.json", {"frozen_bank_sha256": conf["bank"],
               "succeeded_vs_four": control["vs_four"]["ci95"][1] < 0.,
               "near_rbf": control["vs_same_input_rbf"]["ci95"][1] < 0.00075,
               "scores": rows, "blind_C_paths": conf["native_paths"]})


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("stage", choices=("register", "observe", "freeze", "forecast", "truth", "score"))
    cli.add_argument("--workers", type=int, default=12)
    options = cli.parse_args()
    if options.stage == "observe":
        observe(options.workers)
    elif options.stage in ("forecast", "truth"):
        generate(options.stage, options.workers)
    else:
        {"register": register, "freeze": freeze, "score": score}[options.stage]()

