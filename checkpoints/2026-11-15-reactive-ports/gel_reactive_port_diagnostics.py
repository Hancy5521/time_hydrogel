"""Diagnose locked reaction simulations without changing the task or policy."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.special import ndtri
from scipy.stats import qmc

from gel_reactive_port_study import evaluate, policy_choices, sampled_materials
from gel_reactive_ports import bank, first_crossings, schedules


def diagnostic(cfg, scored):
    trial = cfg["goal"]
    plans = schedules(cfg)
    source = sampled_materials(cfg["development"])
    t, train, fitted, _ = bank(source, plans, cfg, cfg["primary_feed"])
    train_flags, train_miss, _ = evaluate(t, fitted, trial, trial["deadline"])
    noise = ndtri(qmc.Sobol(d=4, scramble=True, seed=cfg["noise_seed"]).random_base2(cfg["log2_noise"]))
    groups = []
    audited = 0
    example = None
    for design, prior in zip(cfg["holdouts"], scored["groups"], strict=True):
        params = sampled_materials(design)
        alternatives = []
        benefit = None
        for dose in cfg["dose_rates"]:
            time, observations, paths, balance = bank(params, plans, cfg, dose, cells=cfg["test_cells"], dt=cfg["test_step"])
            hits = first_crossings(time, paths, trial["threshold"])
            for deadline in (trial["deadline"], cfg["final_time"]):
                flags, misses, ends = evaluate(time, paths, trial, deadline)
                low, high = trial["terminal_window"]
                every_above = (ends > high).any(axis=-1).all(axis=-1)
                every_below = (ends < low).any(axis=-1).all(axis=-1)
                terminal = ((ends >= low) & (ends <= high)).all(axis=-1)
                alternatives.append({"feed": dose, "time": deadline, "any_terminal_valid": int(terminal.any(axis=-1).sum()),
                                     "all_actions_exceed_upper": int(every_above.sum()), "all_actions_below_lower": int(every_below.sum()),
                                     "both_orders": int(flags.any(axis=1).all(axis=-1).sum()),
                                     "max_conversion_difference_at_equal_per_port_imbalance": float(np.ptp(ends, axis=1).max()),
                                     "conserved_mass_error": balance})
                if dose != cfg["primary_feed"] or deadline != trial["deadline"]:
                    continue
                if example is None:
                    valid = flags[:, :, None, 0] & flags[:, None, :, 1]
                    difference = abs(ends[:, :, None] - ends[:, None, :]).max(axis=-1)
                    state, left, right = np.argwhere(valid & (difference <= cfg["pair_tolerance"]))[0]
                    selected = (left, right)
                    here = time <= deadline + 1e-9
                    example = {"state_index": int(state), "material": params[state].tolist(), "actions": [plans[a] for a in selected],
                               "regional_curves": paths[state, np.array(selected)][:, here].tolist(), "times": time[here].tolist(),
                               "crossings": hits[state, list(selected)].tolist(), "endpoints": ends[state, list(selected)].tolist(),
                               "max_terminal_difference": float(difference[state, left, right])}
                selected = policy_choices(observations, noise, train, train_flags, train_miss, cfg)
                outcome = {}
                for method, indexes in selected.items():
                    states, draws, goals = indexes.shape
                    action_prob = np.eye(len(plans))[indexes].mean(axis=1)
                    per_state = (action_prob * flags.transpose(0, 2, 1)).sum(axis=-1).mean(axis=-1)
                    mean = per_state.mean()
                    mean_loss = (action_prob * np.maximum(misses, 0).transpose(0, 2, 1)).sum(axis=-1).mean()
                    expected = prior["policies"][method]
                    if abs(mean - expected["success"]) > 1e-13 or abs(mean_loss - expected["mean_violation"]) > 1e-13:
                        raise ArithmeticError(f"Score reconstruction differs for {design['name']}/{method}.")
                    audited += states * draws * goals
                    outcome[method] = per_state
                difference = (outcome["local_joint"] - outcome["free_only"]).reshape(-1, 2).mean(axis=1)
                benefit = {"matched_configuration_pairs": len(difference), "joint_improved_pairs": int((difference > 0).sum()),
                           "joint_worse_pairs": int((difference < 0).sum()), "joint_tied_pairs": int((difference == 0).sum()),
                           "mean_difference": float(difference.mean())}
        groups.append({"name": design["name"], "failure_breakdown": alternatives, "extra_channel_effect": benefit})
    return {"scores_reconstructed_for_actions": audited, "holdout_groups": groups, "first_eligible_case": example,
            "scope": "Frozen simulation replay and arithmetic on expected action counts, not new material evidence."}


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    for field in ("protocol", "scored", "out"):
        cli.add_argument(field, type=Path)
    args = cli.parse_args()
    cfg = json.loads(args.protocol.read_text(encoding="utf-8"))
    scored = json.loads(args.scored.read_text(encoding="utf-8"))
    with args.out.open("x", encoding="utf-8", newline="\n") as output:
        json.dump(diagnostic(cfg, scored), output, indent=2, allow_nan=False)
        output.write("\n")


if __name__ == "__main__":
    main()
