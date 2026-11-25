"""Verify sealed transport trajectories by an independent incidence solver and event replay."""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import sys
import json

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
from study import stamp, read, digest, dump
from audit_reaction import replay


def incidence_energy(journal, clock):
    event = [e for e in journal if e[0] <= clock and int(e[1]) in {3, 4, 5, 6, 7, 9}]
    if not event:
        return np.zeros(3)
    left, right = (np.asarray([e[k] for e in event], dtype=int)//2 for k in (2, 3))
    m = np.zeros((len(event), 78))
    m[np.arange(len(event)), left] = -1.
    m[np.arange(len(event)), right] = 1.
    p = np.asarray([e[10:13] for e in event], dtype=float)
    z = np.linalg.lstsq(m, -p, rcond=None)[0]
    return np.square(m @ z+p).sum(axis=0)


def run(expected):
    if expected != digest(ROOT/"FROZEN.json"):
        raise ValueError("A matching frozen prediction is required to audit the holdout.")
    c = read("CONFIRMED.json")
    assert c["freeze"] == expected and c["sha256"] == digest(ROOT/"CONFIRMED.npz")
    f = read("FROZEN.json")
    s = read("SCORED.json")
    assert s["frozen_sha256"] == expected and s["truth_sha256"] == c["sha256"]
    with np.load(ROOT/"CONFIRMED.npz") as d:
        trace, journal, offsets, i, feed = (d[key] for key in ("trace","journal","offsets","initiators","feed"))
    n, a, r, t, axes = trace.shape
    assert (n,a,r,t,axes) == (24,9,16,3,3)
    actual = trace[:,:,0].reshape(-1,t,axes)
    errors = []
    flags = []
    events = Counter()
    for j,(x,y) in enumerate(zip(offsets[:-1],offsets[1:])):
        raw = journal[x:y]
        replayed, _, counted = replay(raw, int(i[j]), float(feed[j]))
        events.update(counted)
        fresh = np.stack([incidence_energy(raw,t) for t in (3.5,4.85,6.)])
        errors.append(float(np.max(np.abs(fresh-actual[j]))))
        flags.append(np.array_equal(fresh.max(axis=1)>1e-13,replayed[-3:,-1]!=0.))
    assert len(errors)==c["journal_count"]==n*a and max(errors)<1e-11 and all(flags)
    k = trace.sum(axis=-1)
    early = np.logical_and(k[...,0]>=.02,k[...,2]>=.02)
    late = np.logical_and.reduce((k[...,0]==0.,k[...,1]>=.02,k[...,2]>=.02))
    checks = 0
    for goal, truth in enumerate((early, late)):
        for m,method in enumerate(f["policies"]):
            offered = [f["choices"][m][card][goal] for card in range(n)]
            hits = sum(int(truth[card,action].sum()) for card,action in enumerate(offered))
            assert hits == s["result"][f["cases"][goal]]["method"][method]["hits"]
            checks += 1
    result = {**stamp(), "raw_journals":len(errors), "checked_directions_and_times":len(errors)*t*axes,
              "max_abs_error":max(errors), "winding_flag_agreement_on_journals":all(flags),
              "target_hit_counts_recomputed":checks, "total_raw_events_replayed":sum(events.values()),
              "frozen_sha256":expected, "truth_sha256":c["sha256"],
              "audit_source_sha256":digest(Path(__file__).resolve())}
    dump("INDEPENDENT_AUDIT.json",result)
    print(json.dumps(result),flush=True)


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--freeze-sha256",required=True)
    run(p.parse_args().freeze_sha256)
