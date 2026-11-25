"""Construct a native local-observation ambiguity with no fitted components."""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np

import capture as src


ROOT = Path(__file__).resolve().parent


def legal(state, material, geo, rates):
    from aam_edge_fork import candidates

    count = len(state.active)
    capacity = count * (count + 2)
    rows = np.empty((capacity, 6), dtype=np.int64)
    d = np.empty(capacity)
    hazard = np.empty(capacity)
    live = np.flatnonzero(state.present)
    cache = np.full((len(state.present), len(state.present)), -1, dtype=np.int8)
    n = candidates(state, int(material[0]), geo[2], geo[7], geo[8], geo[9], rates,
                   cache, live, rows, d, hazard)
    keep = hazard[:n] > 0.
    return rows[:n][keep], d[:n][keep], hazard[:n][keep]


def compare(packet, clock):
    from aam_edge_fork import apply_event

    state, material, geo, rates, _ = packet
    before, r, w = legal(state, material, geo, rates)
    for q in np.flatnonzero(before[:, 0] == 8):
        vanished = (before[:, 4] >= 0) & np.isin(before[:, 1], before[q, 1:3])
        if not vanished.any():
            continue
        clone = type(state)(*(x.copy() for x in state))
        apply_event(clone, before[q], r[q], geo[2], clock, clock)
        for k in ("points", "present", "owner", "available"):
            np.testing.assert_array_equal(getattr(clone, k), getattr(state, k), err_msg=k)
        assert clone.counts[1] == state.counts[1]
        np.testing.assert_array_equal(clone.edges[:clone.counts[1]], state.edges[:state.counts[1]])
        assert clone.counts[5] == state.counts[5]
        after, _, wa = legal(clone, material, geo, rates)
        begin_mask = before[:, 4] >= 0
        end_mask = after[:, 4] >= 0
        return {
            "time": clock, "termination_site_pair": before[q, 1:3].tolist(),
            "count_active_before": int(state.active.sum()), "count_active_after": int(clone.active.sum()),
            "visible_state": "Every position, site occupancy, current edge and species identical across whole box",
            "second_site_candidates_before": int(begin_mask.sum()), "second_site_candidates_after": int(end_mask.sum()),
            "second_site_primitive_rate_before": float(w[begin_mask].sum()),
            "second_site_primitive_rate_after": float(wa[end_mask].sum()),
            "removed_relevant_candidates": int(vanished.sum()),
        }
    return None


def execute():
    import reaction_adapter as sim
    from event_capture import setup

    setup()
    assert src.read("IDENTIFIABILITY_REGISTERED.json")["source"] == src.digest(Path(__file__))
    called = sim.advance_fixed
    seeds = src.rng("nonidentifiability-native-fixture", "test-logic-only").integers(0, 2**63 - 1, 16)
    for seed in seeds:
        matches = []
        def compare_stage(*args):
            valid = compare(args[0], args[1] * sim.TICK)
            if valid is not None:
                matches.append(valid)
            return called(*args)
        sim.advance_fixed = compare_stage
        try:
            sim.simulate(np.zeros(4), "B", int(seed), full=True)
        finally:
            sim.advance_fixed = called
        if matches:
            src.write("IDENTIFIABILITY_FIXTURE.json", {"stage": matches[0], "seed": int(seed),
                      "scope": "single native counterexample, supports dependence on hidden radical state, no population estimate"})
            return
    raise RuntimeError("No witness in the preregistered fixture seeds.")


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("action", choices=("register", "run"))
    if cli.parse_args().action == "register":
        src.write("IDENTIFIABILITY_REGISTERED.json", {"source": src.digest(Path(__file__)),
                  "native": src.binding(), "role": "separate fixture; no training, scoring or empirical enlargement"})
    else:
        execute()

