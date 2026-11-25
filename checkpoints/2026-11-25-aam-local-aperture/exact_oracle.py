"""Diagnostic: remove interpolated-grid error from the continuous privileged oracle."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy.spatial.distance import cdist

import capture as src
import continuum as exp
from fit import summary


ROOT = Path(__file__).resolve().parent
TARGET = ("CONTINUUM_C_BANK.npz", "CONTINUUM_C_TRUTH.npz", "CONTINUUM_RESULT.json",
          "CONTINUUM_C_FREEZE.json", "CONTINUUM_C_CONFIRM.json", "exact_oracle.py",
          "continuum.py", "forecast.py")


def binding():
    return {str(ROOT / name): src.digest(ROOT / name) for name in TARGET}


def run(workers):
    pinned = src.read("ORACLE_REGISTERED.json")
    assert pinned["sources_sha256"] == binding()
    codes, slopes, _ = exp.cohort("held")
    ids = exp.addresses("bank")
    with np.load(ROOT / "CONTINUUM_C_BANK.npz", allow_pickle=False) as plan:
        np.testing.assert_array_equal(ids, plan["seeds"])
    assert np.min(np.abs(slopes[..., None] - exp.GRID)) > 1.e-10
    plan = src.read(str(exp.NATIVE_SETTINGS))["selected"]
    arms = (("schedule", plan[0]), ("pendant", plan[2]))
    jobs = ((z, i, mode, setting, amp) for z, s, rr in zip(codes, slopes, ids, strict=True)
            for mode, setting in arms for amp in s for i in rr)
    rows = []
    with ProcessPoolExecutor(max_workers=workers, initializer=exp.prepare_native) as pool:
        for i, datum in enumerate(pool.map(exp.c_worker, jobs, chunksize=20), 1):
            rows.append(datum)
            if i % 2048 == 0:
                print("exact oracle", i, flush=True)
    data = np.asarray(rows).reshape(len(codes), 2, 2, exp.REPS, 6, 10)
    src.arrays("ORACLE_NATIVE.npz", path=data, seeds=ids)
    src.write("ORACLE_NATIVE_RECEIPT.json", {"raw": src.digest(ROOT / "ORACLE_NATIVE.npz"), "paths": len(rows),
               "base_registration": src.digest(ROOT / "ORACLE_REGISTERED.json")})


def score():
    locked = src.read("ORACLE_NATIVE_RECEIPT.json")
    assert locked["raw"] == src.digest(ROOT / "ORACLE_NATIVE.npz")
    assert locked["base_registration"] == src.digest(ROOT / "ORACLE_REGISTERED.json")
    assert src.read("ORACLE_REGISTERED.json")["sources_sha256"] == binding()
    arrays = []
    for input_name in ("ORACLE_NATIVE.npz", "CONTINUUM_C_TRUTH.npz"):
        with np.load(ROOT / input_name, allow_pickle=False) as current:
            arrays.append(exp.feature(current["path"]))
    oracle, truth = arrays
    card_count = len(oracle)
    previous = src.read("CONTINUUM_RESULT.json")["energy_by_forecaster"]
    center = np.array(previous["fixed_four"]["value"]["card"])
    coarse = np.array(previous["known_continuous_rule_interpolated"]["value"]["card"])
    values = np.zeros((card_count, 2, 2))
    off = ~np.eye(exp.REPS, dtype=bool)
    for c in range(card_count):
        for a in range(2):
            for g in range(2):
                x, y = oracle[c, a, g], truth[c, a, g]
                values[c, a, g] = cdist(x, y).mean() - .5 * cdist(x, x)[off].mean()
    by_card = values.mean(axis=(1, 2))
    reps = np.random.default_rng(616712).integers(card_count, size=(16000, card_count))
    src.write("ORACLE_RESULT.json", {"status": "privileged diagnostic only", "fresh_forecast_paths": locked["paths"],
               "shared_forecast_addresses": int(card_count*exp.REPS), "energy": summary(by_card, reps),
               "vs_fixed_four": summary(by_card-center, reps), "vs_interpolated_oracle": summary(by_card-coarse, reps)})


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("action", choices=("register", "run", "score"))
    cli.add_argument("--workers", type=int, default=8)
    opts = cli.parse_args()
    if opts.action == "register":
        src.write("ORACLE_REGISTERED.json", {"sources_sha256": binding(),
                   "scope": "known continuous labels and consumed truth for mechanistic headroom check only",
                   "preserve_time_and_parameters": "same true card and native C actions, no estimated-model fit",
                   "seeds": "intentional common-address control with original forecast; truth different"})
    elif opts.action == "run":
        run(opts.workers)
    else:
        score()

