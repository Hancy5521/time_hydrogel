"""Replay all 240 retained final chemistry journals from the extracted bundle."""
import json
import os
from pathlib import Path
import sys

import numpy as np


def sha(path):
    from public_integrity import original_digest

    return original_digest(path)



def main():
    bundle = Path(__file__).resolve().parent
    run = bundle / "run"
    branch = run / "final_paired_process"
    receipt = json.loads((branch/"CONFIRMED.json").read_text(encoding="utf-8-sig"))
    raw = branch/"RAW.npz"
    if sha(raw) != receipt["raw_sha256"] or sha(branch/"FROZEN.json") != receipt["freeze"]:
        raise ValueError("Final freeze or raw file does not match its confirmation receipt.")

    # Original replay implementation is unchanged. Point its worktree lookup
    # at the bundled frozen dependencies before importing it.
    os.environ["AAM_REACTION_WORKTREE"] = str(bundle/"native")
    sys.path.insert(0, str(run))
    from audit_reaction import replay

    with np.load(raw,allow_pickle=False) as data:
        history = data["path"]
        milestones = data["milestones"]
        offsets = data["offsets"]
        events = data["journals"]
        budget = data["initiators"]
        feed = data["feed"]
    arms, cards, _ = history.shape[:3]
    np.testing.assert_equal(arms*cards, 240)
    np.testing.assert_equal(len(offsets), arms*cards + 1)
    np.testing.assert_equal(offsets[[0,-1]], [0,len(events)])
    np.testing.assert_equal(len(budget), arms*cards)
    np.testing.assert_equal(len(feed), arms*cards)
    peak = np.zeros(2)
    for j, (first, end) in enumerate(zip(offsets[:-1],offsets[1:],strict=True)):
        arm, card = divmod(j,cards)
        trace, marks, _ = replay(events[first:end], int(budget[j]), float(feed[j]))
        peak = np.maximum(peak, [abs(history[arm,card,0]-trace).max(),
                                 abs(milestones[arm,card,0]-marks).max()])
    if np.max(peak) > 2.e-13:
        raise AssertionError(f"Replay differs from sealed readouts by {peak}.")
    print(f"OK: {arms*cards} retained journals, {len(events)} events; max errors {peak.tolist()}.")
    print("Replay reads sealed arrays and produces no new reaction trajectories.")


if __name__ == "__main__":
    main()
