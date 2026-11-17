"""Assess ambiguities from two local observations with matched early kinetics."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.special import ndtr

from gel_reactive_port_kinetic_check import bank_for, observation
from gel_reactive_port_study import sampled_materials
from gel_reactive_port_timing import read_json, sha256
from gel_reactive_ports import schedules


def paired_limit(first, second, distance_over_sd):
    """Exact average joint success with equal prior and known twin identities."""
    has_first = first.any(axis=1)
    has_second = second.any(axis=1)
    both = has_first & has_second
    shared = (first & second).any(axis=1)
    return {
        "neither": ~(has_first | has_second),
        "one_feasible": has_first ^ has_second,
        "common_action": shared,
        "incompatible_actions": both & ~shared,
        "full_information": (has_first.astype(float) + has_second) / 2,
        "two_probe_limit": (has_first ^ has_second) * .5 + shared
                           + (both & ~shared) * ndtr(distance_over_sd[:, None] / 2),
    }


def summarize_cases(pair):
    numeric = ("full_information", "two_probe_limit")
    masks = {key: int(value.sum()) for key, value in pair.items() if key not in numeric}
    return {"case_counts": masks, **{key + "_mean": float(pair[key].mean()) for key in numeric}}


def run(protocol, cfg):
    beta, high = protocol["beta_pair"]
    materials = sampled_materials(protocol["twins"])
    stronger_inhibition = materials.copy()
    pivot = cfg["goal"]["threshold"]
    stronger_inhibition[:, 1] *= (1 + beta * pivot) / (1 + high * pivot)
    clocks = {time for layout in protocol["layouts"].values() for time in layout}
    plans = schedules(cfg)
    first = bank_for(materials, plans, cfg, clocks, beta, cfg["test_cells"], cfg["test_step"])
    second = bank_for(stronger_inhibition, plans, cfg, clocks, high, cfg["test_cells"], cfg["test_step"])
    layouts = {}
    for label, clock in protocol["layouts"].items():
        raw = np.linalg.norm(observation(first, clock) - observation(second, clock), axis=-1)
        condition = []
        for sd in protocol["noise_sd"]:
            separation = raw / sd
            pair = paired_limit(first["flags"], second["flags"], separation)
            conflict = pair["incompatible_actions"]
            close = (separation <= 1)[:, None]
            ceiling = pair["two_probe_limit"][conflict]
            condition.append({
                "noise_sd": sd,
                "median_distance_in_noise_units": float(np.median(separation)),
                "max_distance_in_noise_units": float(separation.max()),
                "close_configuration_count": int(close.sum()),
                "close_conflicting_goals": int((conflict & close).sum()),
                "conflict_only_mean_limit": float(ceiling.mean()) if ceiling.size else None,
                "conflict_only_max_limit": float(ceiling.max()) if ceiling.size else None,
                **summarize_cases(pair)
            })
        layouts[label] = condition
    return {
        "scope": protocol["scope"],
        "pair_count": len(materials), "base_mirror_groups": len(materials) // 2,
        "first_law_rate_limits": [float(materials[:, 1].min()), float(materials[:, 1].max())],
        "second_law_rate_limits": [float(stronger_inhibition[:, 1].min()), float(stronger_inhibition[:, 1].max())],
        "mass_errors": [first["mass_error"], second["mass_error"]],
        "layouts": layouts,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("protocol", type=Path)
    parser.add_argument("parent_protocol", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    protocol = read_json(args.protocol)
    sources = {Path(__file__).parent / name: digest for name, digest in protocol["sources"].items()}
    sources[args.parent_protocol] = protocol["parent_sha256"]
    if any(sha256(path) != digest for path, digest in sources.items()):
        raise ValueError("Frozen input or source hash mismatch.")
    with args.output.open("x", encoding="utf-8") as file:
        json.dump({"registered_sha256": sha256(args.protocol), **run(protocol, read_json(args.parent_protocol))},
                  file, indent=2, allow_nan=False)
        file.write("\n")


if __name__ == "__main__":
    main()
