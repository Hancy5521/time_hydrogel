"""Assess reading age at a fixed two-port actuation deadline."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.linalg import solve
from scipy.special import ndtri, softmax
from scipy.stats import qmc

from gel_reactive_ports import advance, bank, material_fields, prepare_step, schedules
from gel_reactive_port_study import best_action, evaluate, features, sampled_materials


def prefix_snapshots(parameters, acquisition_times, cfg, cells, dt):
    """Return readings of free reagent at the two fixed sites, indexed by time."""
    steps = {round(time / dt): time for time in acquisition_times}
    if any(time < 0 or time > cfg["prefix_duration"] or abs(step * dt - time) > 1e-12
           for step, time in steps.items()) or len(steps) != len(set(acquisition_times)):
        raise ValueError("Acquisition times must lie on the common prefix clock.")
    coordinates = (np.arange(cells) + .5) / cells
    diffusion, rate, capacity = material_fields(parameters, coordinates)
    operator = prepare_step(cells, diffusion, dt, cfg["prefix_feed"], np.zeros(len(parameters)))
    free = np.zeros_like(rate)
    bound = np.zeros_like(rate)
    readings = {}
    for step in range(max(steps) + 1):
        if step in steps:
            readings[steps[step]] = np.array([
                np.interp(cfg["probe_locations"], coordinates, row) for row in free
            ])
        if step < max(steps):
            free, bound = advance(free, bound, rate, capacity, operator, dt)
    return readings


def two_readings(snapshots, clocks):
    """Read each sensor at its own acquisition time."""
    return np.column_stack([snapshots[time][:, site] for site, time in enumerate(clocks)])


def posterior_choices(reference, queries, sd, flags, misses):
    score = -.5 * np.sum(((queries[..., None, :] - reference) / sd)**2, axis=-1)
    weights = softmax(score, axis=-1)
    dimensions = (*queries.shape[:-1], *flags.shape[1:])
    flags_flat = flags.reshape(len(reference), -1)
    severity = np.maximum(misses, 0).reshape(len(reference), -1)
    return best_action((weights @ flags_flat).reshape(dimensions),
                       (weights @ severity).reshape(dimensions))


def quadratic_choices(reference, queries, sd, noise, flags, misses, ridge):
    """Direct fit on the same clock-matched, noisy information and outcomes."""
    center = reference.mean(axis=0)
    scale = reference.std(axis=0) + sd
    simulated = reference[:, None] + noise[None] * sd
    inputs = features((simulated - center) / scale).reshape(len(reference) * len(noise), -1)
    design = features((queries - center) / scale)
    gram = inputs.T @ inputs + np.eye(inputs.shape[1]) * ridge * len(inputs)
    estimates = []
    for outcome in (flags, np.maximum(misses, 0)):
        labels = np.repeat(outcome.reshape(len(reference), -1), len(noise), axis=0)
        weights = solve(gram, inputs.T @ labels, assume_a="pos")
        estimates.append((design @ weights).reshape(*queries.shape[:-1], *flags.shape[1:]))
    return best_action(*estimates)


def evaluate_materials(parameters, plans, cfg, clocks, cells, dt):
    times, observed, paths, mass_error = bank(parameters, plans, cfg, cfg["primary_feed"], cells=cells, dt=dt)
    flags, misses, _ = evaluate(times, paths, cfg["goal"], cfg["goal"]["deadline"])
    snapshots = prefix_snapshots(parameters, clocks, cfg, cells, dt)
    match_error = float(np.max(np.abs(observed[:, :2] - snapshots[cfg["prefix_duration"]])))
    if match_error > 1e-12:
        raise ArithmeticError("Repeated prefix disagrees with the registered simulator.")
    return snapshots, flags, misses, {"mass_error": mass_error, "deadline_read_error": match_error}


def summarize(choices, flags):
    selected = flags[np.arange(len(flags))[:, None, None], choices, np.arange(flags.shape[-1])]
    # Mirrors are grouped with the same base configurations. Noise draws and goal
    # directions provide an average per configuration, not extra sample size.
    return selected.reshape(-1, 2, *selected.shape[1:]).mean(axis=(1, 2, 3))


def compare_cases(train, test, cfg, protocol, noise):
    train_readings, train_flags, train_misses, _ = train
    test_readings, test_flags, _, _ = test
    fixed_policy = best_action(train_flags.mean(axis=0), np.maximum(train_misses, 0).mean(axis=0))
    shape = (len(test_flags), len(noise), test_flags.shape[-1])
    fixed = summarize(np.broadcast_to(fixed_policy, shape), test_flags)
    completed = []
    for case in protocol["clocks"]:
        actual = case["acquisition"]
        nominal = case["ignored_clock"]
        reference = two_readings(train_readings, actual)
        stale_reference = two_readings(train_readings, nominal)
        observed = two_readings(test_readings, actual)
        for sd in protocol["noise_sd"]:
            query = observed[:, None] + sd * noise[None]
            policies = {
                "matched": posterior_choices(reference, query, sd, train_flags, train_misses),
                "ignored": posterior_choices(stale_reference, query, sd, train_flags, train_misses),
                "quadratic": quadratic_choices(reference, query, sd, noise, train_flags,
                                               train_misses, cfg["ridge"]),
            }
            results = {name: summarize(policy, test_flags) for name, policy in policies.items()}
            mean = {name: float(value.mean()) for name, value in results.items()}
            delta = results["matched"] - results["ignored"]
            completed.append({
                "case": case["name"], "acquisition": actual, "ignored_clock": nominal, "noise_sd": sd,
                "rates": mean, "matched_minus_fixed": float((results["matched"] - fixed).mean()),
                "matched_minus_ignored": float(delta.mean()),
                "matched_minus_quadratic": float((results["matched"] - results["quadratic"]).mean()),
                "matched_vs_ignored_mirror_pairs": {
                    "improved": int(np.sum(delta > 0)), "tied": int(np.sum(delta == 0)),
                    "worse": int(np.sum(delta < 0)),
                },
                "disagreeing_decisions": int(np.sum(policies["matched"] != policies["ignored"])),
            })
    return {"fixed_rate": float(fixed.mean()), "fixed_schedule": fixed_policy.tolist(),
            "oracle_rate": float(test_flags.any(axis=1).mean()),
            "materials": len(test_flags), "mirror_groups": len(fixed), "cases": completed}


def run(protocol, cfg):
    clocks = {time for case in protocol["clocks"] for key in ("acquisition", "ignored_clock")
              for time in case[key]}
    plans = schedules(cfg)
    dev = evaluate_materials(sampled_materials(cfg["development"]), plans, cfg, clocks,
                             cfg["cells"], cfg["step"])
    noise = ndtri(qmc.Sobol(d=2, scramble=True, seed=protocol["noise_seed"]).random_base2(
        protocol["log2_noise"]))
    holdouts = {}
    for design in protocol["holdouts"]:
        test = evaluate_materials(sampled_materials(design), plans, cfg, clocks,
                                  cfg["test_cells"], cfg["test_step"])
        holdouts[design["name"]] = {**test[-1], **compare_cases(dev, test, cfg, protocol, noise)}
    return {"scope": protocol["scope"], "development_checks": dev[-1], "holdouts": holdouts}


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("protocol", type=Path)
    parser.add_argument("prior_protocol", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    protocol = read_json(args.protocol)
    files = {Path(__file__).resolve().parent / name: digest for name, digest in protocol["sources"].items()}
    files[args.prior_protocol] = protocol["prior_sha256"]
    if any(sha256(path) != digest for path, digest in files.items()):
        raise ValueError("Frozen input or source hash mismatch.")
    result = {"registered_sha256": sha256(args.protocol), **run(protocol, read_json(args.prior_protocol))}
    with args.output.open("x", encoding="utf-8") as file:
        json.dump(result, file, indent=2, allow_nan=False)
        file.write("\n")


if __name__ == "__main__":
    main()
