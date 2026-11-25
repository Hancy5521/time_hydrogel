"""One frozen matched witness intervention to increase observable rate opportunities."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

import capture as src
import mobile_capture as mobile
import continuum as cont
from fit import summary


ROOT = Path(__file__).resolve().parent
CARDS = 128
LOW = .25
CHANGE, STOP = 2.05, 4.85
HIGH = (STOP - LOW * CHANGE) / (STOP - CHANGE)


def mothers():
    rng = src.rng("pulse-witness", "rules")
    return rng.uniform(-.48, .48, (CARDS, 4)), rng.uniform(0., 1.10, CARDS)[:, None] * [-1., 1.], (
        src.rng("pulse-witness", "paired initial states").integers(0, 2**63 - 1, (CARDS, 2, 4)))


def bindings():
    files = ("pulse_witness.py", "PULSE_PROTOCOL_ZH.md", "CONTINUUM_MODEL_FREEZE.json",
             "continuum.py", "mobile_kernels.py", "mobile_capture.py")
    return {**cont.bindings(), **{str(ROOT / name): src.digest(ROOT / name) for name in files}}


def register():
    ids = mothers()[2].ravel()
    assert np.unique(ids).size == ids.size
    for sample in ROOT.glob("*.npz"):
        with np.load(sample, allow_pickle=False) as other:
            for key in ("seeds", "addresses"):
                if key in other.files:
                    assert not np.intersect1d(ids, other[key]).size
    assert np.isclose(CHANGE * LOW + (STOP - CHANGE) * HIGH, STOP, atol=1.e-14)
    src.write("PULSE_REGISTERED.json", {"source_sha256": bindings(), "cards": CARDS,
               "known_conditions": [LOW, HIGH], "witness_messages_max": 3,
               "methods_fitted": 0, "C_outcome_access": False})


def seen(job):
    import reaction_adapter as sim
    import mobile_kernels as kern

    code, env, seed, value, pulse = job
    if not pulse:
        return cont.witness((code, env, seed, value))[0]
    original = kern.advance_mobile
    threshold = sim._tick(CHANGE)
    def adjusted(*args):
        packet, start, end = args[:3]
        if start < threshold < end:
            raise ValueError("The pulse boundary must be a native stage boundary.")
        rate = packet[3][4:6]
        before = rate.copy()
        rate *= LOW if start < threshold else HIGH
        try:
            return original(*args)
        finally:
            rate[:] = before
    kern.advance_mobile = adjusted
    try:
        return cont.witness((code, env, seed, value))[0]
    finally:
        kern.advance_mobile = original


def acquire(workers):
    assert bindings() == src.read("PULSE_REGISTERED.json")["source_sha256"]
    g, slopes, addresses = mothers()
    records = []
    jobs = ((code, env, k, h, control) for code, direction, a in zip(g, slopes, addresses, strict=True)
            for h in direction for control in (False, True)
            for env, group in zip(("A", "B"), a, strict=True) for k in group)
    with ProcessPoolExecutor(max_workers=workers, initializer=mobile.setup) as pool:
        for i, row in enumerate(pool.map(seen, jobs, chunksize=18), 1):
            records.append(row)
            if i % 1024 == 0:
                print("pulse witness", i, flush=True)
    data = np.asarray(records).reshape(CARDS, 2, 2, 2, 4, 3, 6)
    src.arrays("PULSE_RAW.npz", marks=data, seeds=addresses)
    src.write("PULSE_CONFIRMED.json", {"registration_sha256": src.digest(ROOT / "PULSE_REGISTERED.json"),
               "raw_sha256": src.digest(ROOT / "PULSE_RAW.npz"), "paths": len(data)*32})


def score():
    current = src.read("PULSE_CONFIRMED.json")
    assert current["registration_sha256"] == src.digest(ROOT / "PULSE_REGISTERED.json")
    assert current["raw_sha256"] == src.digest(ROOT / "PULSE_RAW.npz")
    assert bindings() == src.read("PULSE_REGISTERED.json")["source_sha256"]
    with np.load(ROOT / "PULSE_RAW.npz", allow_pickle=False) as file:
        marks = file["marks"]
    y = mothers()[1].ravel()
    alpha = src.read("CONTINUUM_MODEL_FREEZE.json")["models"]["mobile_conditional"]["alpha"]
    first, second = (cont.law(marks[:, :, i], alpha) for i in range(2))
    first_loss, second_loss = (cont.crps(p, y).reshape(CARDS, 2).mean(axis=1) for p in (first, second))
    captures = (marks[..., 4] >= 0).sum(axis=(3, 4, 5)).mean(axis=1)
    rng = np.random.default_rng(7437736).integers(CARDS, size=(16000, CARDS))
    src.write("PULSE_RESULT.json", {"registration_sha256": current["registration_sha256"],
               "raw_sha256": current["raw_sha256"], "frozen_alpha": alpha,
               "nominal_crps": summary(first_loss, rng), "pulse_crps": summary(second_loss, rng),
               "gain": summary(first_loss-second_loss, rng),
               "capture_nominal_to_pulse": captures.mean(axis=0).tolist(),
               "capture_gain": summary(captures[:, 1]-captures[:, 0], rng),
               "passed": bool(summary(first_loss-second_loss, rng)["ci95"][0] > 0.
                              and summary(captures[:, 1]-captures[:, 0], rng)["ci95"][0] > 0.)})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("register", "acquire", "score"))
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if args.stage == "acquire":
        acquire(args.workers)
    else:
        (register if args.stage == "register" else score)()

