"""Prospective fair forward comparison from frozen local-event observations."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import importlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial.distance import cdist

import capture as src
from fit import summary


ROOT = Path(__file__).resolve().parent
REPS = 16
KNOWN = ("constant_four", "equal_mixture", "known_true_direction")
NATIVE_SETTINGS = (ROOT / "provenance/prior_native_settings/final_paired_process/FROZEN.json")
EXPECTED_NATIVE_SHA = "c7cfa710c81482c9ed046ec4395d71816946c3edd0c45433b32c7da3471568a6"


def source_for(prefix):
    return importlib.import_module("event_capture" if prefix == "EVENT" else "mobile_capture")


def group(prefix):
    return source_for(prefix).group("check")


def addresses(prefix, role):
    return src.rng("forward-" + prefix, role).integers(0, 2**63 - 1, (len(group(prefix)[0]), REPS))


def names(prefix, file):
    return prefix + "_C_" + file


def config():
    src.configure()
    import reaction_adapter as sim
    import scheduled
    import pendant
    sim.EARLY_TIMES = (.65, 1.35, 2.05)
    scheduled.warm()
    pendant.warm()
    sim._clock.cache_clear()


def bindings(prefix):
    assert src.digest(NATIVE_SETTINGS) == EXPECTED_NATIVE_SHA
    files = [ROOT / name for name in ("forecast.py", "FORECAST_PROTOCOL_ZH.md",
              prefix + "_PREDICTION_FREEZE.json", prefix + "_PREDICTED.npz", prefix + "_REGISTERED.json")]
    return {**src.binding(), **{str(path): src.digest(path) for path in [NATIVE_SETTINGS, *files]}}


def register(prefix):
    g, witness = group(prefix)
    planned = np.concatenate([addresses(prefix, part).ravel() for part in ("forecast", "truth")])
    assert np.unique(planned).size == planned.size
    for path in ROOT.glob("*.npz"):
        with np.load(path, allow_pickle=False) as raw:
            for role in ("seeds", "addresses"):
                if role in raw.files:
                    assert not np.intersect1d(planned, raw[role]).size
    src.write(names(prefix, "REGISTERED.json"), {"source_sha256": bindings(prefix), "cards": len(g),
              "witness_addresses": witness.size, "novel_C_addresses": planned.size,
              "C_repeats_each_role": REPS, "scope": "forecast with known four, conditional slope unknown"})


def work(task):
    import reaction_adapter as sim

    code, seed, action, setting, profile = task
    decode = sim.decode
    def edited(g, base):
        current = decode(g, base)
        if profile:
            factors = np.asarray(current["factors"])
            factors[:, 1:] *= np.exp(profile * src.SLOPE * src.PROFILE)
            current["factors"] = factors.tolist()
        return current
    sim.decode = edited
    try:
        if action == "schedule":
            from scheduled import worker
        else:
            from pendant import worker
        return worker((code, setting, int(seed), False))["path"]
    finally:
        sim.decode = decode


def verify_freeze(prefix):
    file = names(prefix, "FORECAST_FREEZE.json")
    locked = src.read(file)
    assert locked["model_sha256"] == src.digest(ROOT / f"{prefix}_PREDICTED.npz")
    assert locked["predicted_native_bank_sha256"] == src.digest(ROOT / names(prefix, "FORECAST.npz"))
    assert locked["source_sha256"] == bindings(prefix)
    return src.digest(ROOT / file)


def simulate(prefix, role, workers):
    assert src.read(names(prefix, "REGISTERED.json"))["source_sha256"] == bindings(prefix)
    if role == "truth":
        lock = verify_freeze(prefix)
    g, _ = group(prefix)
    ids = addresses(prefix, role)
    mode = (0, -1, 1) if role == "forecast" else (-1, 1)
    old = json.loads(NATIVE_SETTINGS.read_text(encoding="utf-8"))
    arms = (("schedule", old["selected"][0]), ("pendant", old["selected"][2]))
    tasks = ((code, rid, environment, setting, sign) for code, row in zip(g, ids, strict=True)
             for environment, setting in arms for sign in mode for rid in row)
    paths = []
    with ProcessPoolExecutor(max_workers=workers, initializer=config) as pool:
        for i, path in enumerate(pool.map(work, tasks, chunksize=12), 1):
            paths.append(path)
            if i % 2048 == 0:
                print(prefix, role, i, flush=True)
    y = np.asarray(paths).reshape(len(g), len(arms), len(mode), REPS, 6, 10)
    assert np.isfinite(y).all()
    file = names(prefix, role.upper() + ".npz")
    src.arrays(file, path=y, seeds=ids)
    if role == "forecast":
        src.write(names(prefix, "FORECAST_FREEZE.json"), {"source_sha256": bindings(prefix),
                  "model_sha256": src.digest(ROOT / f"{prefix}_PREDICTED.npz"),
                  "predicted_native_bank_sha256": src.digest(ROOT / file)})
    else:
        assert lock == verify_freeze(prefix)
        src.write(names(prefix, "CONFIRMATION.json"), {"freeze_sha256": lock,
                  "truth_sha256": src.digest(ROOT / file)})


def feature(path):
    data = path[..., 2:, [3, 5]] / np.array([.3, .2])
    return data.reshape(*data.shape[:-2], -1) / np.sqrt(8.)


def score(prefix):
    lock = verify_freeze(prefix)
    confirm = src.read(names(prefix, "CONFIRMATION.json"))
    actual = ROOT / names(prefix, "TRUTH.npz")
    assert confirm["freeze_sha256"] == lock and confirm["truth_sha256"] == src.digest(actual)
    with np.load(actual, allow_pickle=False) as raw:
        y = feature(raw["path"])
    with np.load(ROOT / names(prefix, "FORECAST.npz"), allow_pickle=False) as raw:
        simulated = feature(raw["path"])
    with np.load(ROOT / f"{prefix}_PREDICTED.npz", allow_pickle=False) as p:
        probabilities = {k: p[k] for k in p.files if k != "prior"}
    n = len(y)
    names_all = (*KNOWN, *probabilities)
    weights = np.zeros((len(names_all), n, 2, 3))
    weights[0, :, :, 0] = 1.
    weights[1, :, :, 1:] = .5
    weights[2, :, 0, 1], weights[2, :, 1, 2] = 1., 1.
    for i, val in enumerate(probabilities.values(), start=len(KNOWN)):
        weights[i, :, :, 1], weights[i, :, :, 2] = 1. - val, val
    assert np.allclose(weights.sum(axis=-1), 1.)
    disjoint = ~np.eye(REPS, dtype=bool)
    results = np.zeros((len(names_all), n, 2, 2))
    for c in range(n):
        for a in range(2):
            bank = simulated[c, a]
            part = np.array([[cdist(g, h)[disjoint].mean() for h in bank] for g in bank])
            for sign in range(2):
                cross = np.array([cdist(k, y[c, a, sign]).mean() for k in bank])
                w = weights[:, c, sign]
                results[:, c, a, sign] = w @ cross - .5 * np.sum((w @ part) * w, axis=1)
    score_by_card = results.mean(axis=(2, 3))
    reps = np.random.default_rng(6051851).integers(n, size=(18000, n))
    base = score_by_card[names_all.index("event_rbf")]
    out = {k: {"energy": summary(row, reps), "vs_four": summary(row - score_by_card[0], reps),
               "vs_equal": summary(row - score_by_card[1], reps), "vs_direct": summary(row - base, reps)}
           for k, row in zip(names_all, score_by_card, strict=True)}
    src.write(names(prefix, "RESULT.json"), {"freeze_sha256": lock, "confirmed_sha256": confirm["truth_sha256"],
               "fair_scores": out, "paths_forecast": int(n * 6 * REPS), "paths_truth": int(n * 4 * REPS)})


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("prefix", choices=("EVENT", "MOBILE"))
    cli.add_argument("stage", choices=("register", "forecast", "truth", "score"))
    cli.add_argument("--workers", type=int, default=8)
    args = cli.parse_args()
    if args.stage in ("forecast", "truth"):
        simulate(args.prefix, args.stage, args.workers)
    else:
        (register if args.stage == "register" else score)(args.prefix)

