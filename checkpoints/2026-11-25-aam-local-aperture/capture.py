"""Generate masked AAm snapshot transitions without event or hidden-graph inputs."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from hashlib import file_digest, sha256
import json
import os
from pathlib import Path
import sys

import numpy as np


from public_integrity import FROZEN_RUN_ID


ROOT = Path(__file__).resolve().parent
OLD = (ROOT / "provenance/prior_address_evidence")
TIMES = (.65, 1.35, 2.05, 3.10, 3.50, 3.90, 4.30, 4.70, 4.85, 5.10)
WINDOWS = {"early": (0, 1, 2), "late": (2, 4, 8), "short_early": (3, 4, 5),
           "short_late": (5, 6, 7), "long_late": (2, 5, 9)}
TRANSITIONS = tuple(sorted({pair for window in WINDOWS.values() for pair in zip(window[:-1], window[1:])}))
ROLES = {"train": 144, "validate": 64, "test": 96}
SLOPE = 1.10
PARTICLES, AAM = 78, 64
PROFILE = np.array([1., -.5, -.5])[:, None]


def digest(path):
    from public_integrity import original_digest

    return original_digest(path)


def read(name):
    return json.loads((ROOT / name).read_text(encoding="utf-8-sig"))


def write(name, value):
    with (ROOT / name).open("x", encoding="utf-8") as handle:
        json.dump({"utc": datetime.now(timezone.utc).isoformat(), **value},
                  handle, indent=2, allow_nan=False)


def arrays(name, **payload):
    with (ROOT / name).open("xb") as handle:
        np.savez_compressed(handle, **payload)


def rng(role, stream):
    value = sha256(f"{FROZEN_RUN_ID}|masked-graph-transition|{role}|{stream}".encode()).digest()
    return np.random.default_rng(int.from_bytes(value[:16], "little"))


def cohort(role):
    n = ROLES[role]
    return (rng(role, "codes").uniform(-.48, .48, (n, 4)),
            rng(role, "monomer-addresses").integers(0, 2**63 - 1, (n, 2, 4)))


def configure():
    native = ROOT / "snapshot" / "native"
    folder = ROOT / "snapshot" / "run"
    os.environ["AAM_REACTION_WORKTREE"] = str(native)
    os.environ["NUMBA_CACHE_DIR"] = str(ROOT / "jit" / str(os.getpid()))
    sys.path[:0] = [str(folder / "pendant_timing"), str(folder / "scheduled_edit"), str(folder)]
    import reaction_adapter as sim

    sim.STEP = 32
    sim.EARLY_TIMES = TIMES
    sim._clock.cache_clear()


def binding():
    sources = [ROOT / "capture.py", ROOT / "PROTOCOL_ZH.md", OLD / "FAIR_ENERGY_DIAGNOSTIC.json",
               *(ROOT / "snapshot").rglob("*.py")]
    return {str(file): digest(file) for file in sorted(sources)}


def register():
    seeds = np.concatenate([cohort(role)[1].ravel() for role in ROLES])
    assert np.unique(seeds).size == seeds.size
    for file in OLD.glob("*.npz"):
        with np.load(file, allow_pickle=False) as other:
            if "seeds" in other:
                assert np.intersect1d(seeds, other["seeds"]).size == 0
    assert all(0 <= i < len(TIMES) for window in WINDOWS.values() for i in window)
    write("REGISTERED.json", {"source_sha256": binding(), "roles": ROLES, "sampling_times": TIMES,
            "menu": WINDOWS, "transitions": TRANSITIONS, "unique_addresses": len(seeds),
            "scope": "true visible snapshots only; no event tape or global labels",
            "labels": ["negative", "positive"]})


def reconstruct(positions, degree, edges, box, corridor, cutoff, frames):
    # Frames use a common aperture-aligned coordinate system. Negative values are masked.
    before, after = frames
    current = positions[before]
    old, new = degree[before], degree[after]
    both = (old >= 0) & (new >= 0)
    lefts = np.flatnonzero(both[:AAM] & (old[:AAM] >= 1))
    rights = np.flatnonzero(both[AAM:] & (old[AAM:] == 1)) + AAM
    if not len(lefts) or not len(rights):
        return np.zeros((2, 4), dtype=np.int16)
    i, j = (x.ravel() for x in np.meshgrid(lefts, rights, indexing="ij"))
    diff = current[j] - current[i]
    diff -= box * np.rint(diff / box)
    middle = current[i] + .5 * diff
    side = .70 * box
    keep = ((diff * diff).sum(axis=1) < cutoff**2) & (
        ((middle > corridor) & (middle < side - corridor)).all(axis=1))
    i, j, middle = i[keep], j[keep], middle[keep]
    if not len(i):
        return np.zeros((2, 4), dtype=np.int16)
    present = np.flatnonzero(old >= 0)
    d = current[present][None] - middle[:, None]
    d -= box * np.rint(d / box)
    nbr = ((d * d).sum(axis=-1) < corridor**2) & (
           (present[None] != i[:, None]) & (present[None] != j[:, None]))
    crowded = nbr.any(axis=1)
    parent = np.arange(PARTICLES)
    def root(k):
        while parent[k] != k:
            k = parent[k]
        return k
    def adj(at):
        present_edges = edges[at][edges[at, :, 0] >= 0]
        listed = np.zeros((PARTICLES, PARTICLES), dtype=np.int16)
        u, v = present_edges.T
        np.add.at(listed, (u, v), 1)
        np.add.at(listed, (v, u), 1)
        return listed, present_edges
    previous, old_edges = adj(before)
    following, _ = adj(after)
    for u, v in old_edges:
        parent[root(v)] = root(u)
    closure = np.array([root(u) == root(v) for u, v in zip(i, j, strict=True)])
    kind = crowded.astype(int) * 2 + closure
    formed = (following[i, j] > previous[i, j]) & (new[j] == 2)
    opportunities = np.bincount(kind, minlength=4)
    events = np.bincount(kind[formed], minlength=4)
    assert np.all(events <= opportunities)
    return np.stack((opportunities, events)).astype(np.int16)


def worker(entry):
    import reaction_adapter as sim

    material, env, seed, direction = entry
    decode, obs, make = sim.decode, sim._local_observation, sim.make_config
    stash = []
    def rule(g, spec):
        cfg = decode(g, spec)
        factors = np.asarray(cfg["factors"])
        factors[:, 1:] *= np.exp(direction * SLOPE * PROFILE)
        cfg["factors"] = factors.tolist()
        return cfg
    def conf():
        cfg = make()
        return cfg
    def record(state, origin, box, cutoff):
        x = np.remainder(state.points - origin, box)
        seen = state.present & np.all(x < .70 * box, axis=1)
        occupied = (state.owner.reshape(PARTICLES, 2) >= 0).sum(axis=1)
        physical = state.edges[:state.counts[1]]
        kept = physical[np.all(seen[physical], axis=1)]
        visible_edges = np.full((2 * PARTICLES, 2), -1, dtype=np.int16)
        visible_edges[:len(kept)] = kept
        stash.append((np.where(seen[:, None], x, -1.), np.where(seen, occupied, -1), visible_edges,
                      float(box), float(cutoff)))
        return obs(state, origin, box, cutoff)
    sim.decode, sim._local_observation, sim.make_config = rule, record, conf
    try:
        out = sim.simulate(material, env, int(seed))["local"]
    finally:
        sim.decode, sim._local_observation, sim.make_config = decode, obs, make
    assert len(stash) == len(TIMES)
    pts, deg, edges, box, cut = (np.asarray(k) for k in zip(*stash, strict=True))
    # The default source uses corridor radius .82; the register binds the native source.
    corridor = .82
    features = np.stack([reconstruct(pts, deg, edges, box[0], corridor, cut[0], pair) for pair in TRANSITIONS])
    return out, features, np.array([box[0], corridor, cut[0]]), (pts, deg, edges)


def data(roles, workers):
    assert read("REGISTERED.json")["source_sha256"] == binding()
    if "test" in roles:
        raise ValueError("Test aperture data are collected by a separate command.")
    collect(roles, workers)


def collect(roles, workers):
    assert read("REGISTERED.json")["source_sha256"] == binding()
    if tuple(roles) == ("test",):
        freeze = read("MODEL_FREEZE.json")
        assert freeze["registration_sha256"] == digest(ROOT / "REGISTERED.json")
        assert freeze["model_source_sha256"] == digest(ROOT / "fit.py")
        assert freeze["model_arrays_sha256"] == digest(ROOT / "MODEL.npz")
    else:
        assert tuple(roles) == ("train", "validate")
    receipts = {}
    with ProcessPoolExecutor(max_workers=workers, initializer=configure) as pool:
        for role in roles:
            g, address = cohort(role)
            work = ((z, env, rid, sign) for z, row in zip(g, address, strict=True)
                    for sign in (-1., 1.) for env, env_seed in zip(("A", "B"), row, strict=True)
                    for rid in env_seed)
            out = list(pool.map(worker, work, chunksize=12))
            shape = (len(g), 2, 2, 4)
            stat = np.array([v[0] for v in out]).reshape(*shape, len(TIMES), 14)
            pair = np.array([v[1] for v in out]).reshape(*shape, len(TRANSITIONS), 2, 4)
            geometry = np.asarray([v[2] for v in out])
            assert np.all(geometry == geometry[0])
            audit_n = 4
            audit = {name: np.stack([v[3][j] for v in out[:audit_n]])
                     for j, name in enumerate(("points", "occupied", "edges"))}
            name = role.upper() + ".npz"
            arrays(name, x=stat, counts=pair, addresses=address,
                   geometry=geometry[0], audit_x=stat.reshape(-1, len(TIMES), 14)[:audit_n],
                   **audit)
            receipts[name] = digest(ROOT / name)
            print("masked trajectories", role, len(out), "eligible", int(pair[..., 0, :].sum()),
                  "visible events", int(pair[..., 1, :].sum()), flush=True)
    write("CONFIRMED.json" if tuple(roles) == ("test",) else "DEVELOPMENT.json",
          {"raw_sha256": receipts, "model_freeze": (digest(ROOT / "MODEL_FREEZE.json")
           if tuple(roles) == ("test",) else None)})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("register", "dev", "test"))
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if args.stage == "register":
        register()
    else:
        collect(("test",) if args.stage == "test" else ("train", "validate"), args.workers)

