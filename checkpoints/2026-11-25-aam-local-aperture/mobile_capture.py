"""Collect qualified visible event records and check observer non-interference."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

import capture as src


ROOT = Path(__file__).resolve().parent
GROUPS = {"fit": 112, "dev": 48, "check": 112}
PARAMETER = np.array([1., -.5, -.5])[:, None]
STUDY_FILES = ("MOBILE_PROTOCOL_ZH.md", "local_observer.py", "mobile_geometry.py", "generate_mobile.py",
               "mobile_kernels.py", "mobile_capture.py", "EVENT_SCORED.json")


def group(role):
    n = GROUPS[role]
    return (src.rng("mobile-gate-" + role, "four").uniform(-.48, .48, (n, 4)),
            src.rng("mobile-gate-" + role, "witnesses").integers(0, 2**63 - 1, (n, 2, 4)))


def bindings():
    return {**src.binding(), **{str(ROOT / name): src.digest(ROOT / name) for name in STUDY_FILES}}


def reference():
    from reaction_adapter import make_config
    base = make_config()
    rate = np.asarray(base["rates"])[[4, 5, 5]]
    factors = np.asarray(base["factors"])
    return factors[:, :, None] * rate[:, None, :]


def setup():
    src.configure()
    import reaction_adapter as sim
    sim.EARLY_TIMES = (2.05, 3.50, 4.85)
    sim._clock.cache_clear()


def evidence(spec):
    import reaction_adapter as sim
    from mobile_kernels import advance_mobile

    material, context, seed, sign, audit = spec
    old_step, old_decode = sim.advance_fixed, sim.decode
    rows = np.full((3, 6), -1.)
    found = np.zeros(1, dtype=np.int64)
    origin = np.random.default_rng(np.random.SeedSequence(int(seed)).spawn(5)[3]).uniform(0., 1., 3)
    ref = reference()
    fixed = np.full_like(rows, -1.)
    fixed_cursor = np.zeros_like(found)
    dither = src.rng("mobile-aperture-dither", str(seed)).uniform(size=(160, 3))

    def dec(g, spec):
        model = old_decode(g, spec)
        factors = np.asarray(model["factors"])
        factors[:, 1:] *= np.exp(sign * src.SLOPE * PARAMETER)
        model["factors"] = factors.tolist()
        return model

    def step(*args):
        return advance_mobile(*args, origin, ref, rows, found, dither, fixed, fixed_cursor)

    sim.advance_fixed, sim.decode = step, dec
    try:
        got = sim.simulate(material, context, int(seed), full=audit, journal=audit)
        if audit:
            sim.advance_fixed = old_step
            undisturbed = sim.simulate(material, context, int(seed), full=True, journal=True)
            for label in ("path", "journal", "local", "milestones"):
                np.testing.assert_array_equal(got[label], undisturbed[label], err_msg=label)
    finally:
        sim.advance_fixed, sim.decode = old_step, old_decode
    return rows, got["local"], fixed


def register():
    joined = np.concatenate([group(role)[1].ravel() for role in GROUPS])
    assert len(np.unique(joined)) == joined.size
    for role in ("FIT", "DEV", "CHECK"):
        with np.load(ROOT / ("EVENT_" + role + ".npz"), allow_pickle=False) as earlier:
            assert not np.intersect1d(joined, earlier["seeds"]).size
    for role in src.ROLES:
        assert not np.intersect1d(joined, src.cohort(role)[1]).size
    for witness in src.OLD.glob("*.npz"):
        with np.load(witness, allow_pickle=False) as data:
            if "seeds" in data:
                assert not np.intersect1d(joined, data["seeds"]).size
    src.write("MOBILE_REGISTERED.json", {"source_sha256": bindings(), "roles": GROUPS,
               "sensor_budget": 3, "unique_witness_addresses": int(joined.size),
               "scope": "synthetic moving and fixed local event cameras in the same native history"})


def collect(roles, workers):
    assert src.read("MOBILE_REGISTERED.json")["source_sha256"] == bindings()
    locked = None
    if roles == ("check",):
        selected = src.read("MOBILE_MODEL_FREEZE.json")
        assert selected["registration_sha256"] == src.digest(ROOT / "MOBILE_REGISTERED.json")
        assert selected["models_sha256"] == src.digest(ROOT / "MOBILE_MODELS.npz")
        locked = src.digest(ROOT / "MOBILE_MODEL_FREEZE.json")
    receipts = {}
    with ProcessPoolExecutor(max_workers=workers, initializer=setup) as pool:
        for role in roles:
            codes, addr = group(role)
            work = ((g, env, i, flag, n < 16 and role == "fit")
                    for n, (g, card) in enumerate(zip(codes, addr, strict=True))
                    for flag in (-1., 1.) for env, row in zip(("A", "B"), card, strict=True) for i in row)
            gathered = []
            for i, data in enumerate(pool.map(evidence, work, chunksize=12), 1):
                gathered.append(data)
                if i % 512 == 0:
                    print(role, i, flush=True)
            mark, x, fixed = (np.array(values) for values in zip(*gathered, strict=True))
            shape = (len(codes), 2, 2, 4)
            name = "MOBILE_" + role.upper() + ".npz"
            src.arrays(name, marks=mark.reshape(*shape, 3, 6), x=x.reshape(*shape, 3, 14), seeds=addr, fixed=fixed.reshape(*shape, 3, 6))
            receipts[name] = src.digest(ROOT / name)
            print(role, "certified events", int((mark[..., 4] >= 0).sum()),
                  "witness histories", len(mark), flush=True)
    src.write("MOBILE_CONFIRMED.json" if locked else "MOBILE_DEVELOPMENT.json",
              {"data_sha256": receipts, "freeze_sha256": locked,
               "native_exact_noninterference_paths": 256 if not locked else 0})


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("stage", choices=("register", "dev", "check"))
    cli.add_argument("--workers", type=int, default=8)
    args = cli.parse_args()
    if args.stage == "register":
        register()
    else:
        collect(("check",) if args.stage == "check" else ("fit", "dev"), args.workers)

