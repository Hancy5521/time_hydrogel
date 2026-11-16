"""Test decision transfer under product-dependent resistance in gelation."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.fft import dct, idct
from scipy.special import ndtri
from scipy.stats import qmc

from gel_reactive_port_study import best_action, evaluate, sampled_materials, summarize as feasibility
from gel_reactive_port_timing import posterior_choices, quadratic_choices, read_json, sha256, summarize
from gel_reactive_ports import bank, local_binding, material_fields, prepare_step, schedules


def binding(free, bound, rate, capacity, span, beta, pivot):
    """Predict midpoint mobility, then use the exact conservative reaction update."""
    if beta == 0:
        return local_binding(free, bound, rate, capacity, span)
    centered_rate = rate * (1 + beta * pivot)
    _, midpoint = local_binding(free, bound, centered_rate / (1 + beta * bound), capacity, span / 2)
    return local_binding(free, bound, centered_rate / (1 + beta * midpoint), capacity, span)


def simulate(parameters, plans, cfg, clocks, beta, cells, dt):
    """Use the frozen boundary flux and vary only the local reaction law."""
    t = np.arange(round(cfg["final_time"] / dt) + 1) * dt
    switching = round(cfg["prefix_duration"] / dt)
    fed = round(cfg["feed_duration"] / dt)
    clock_steps = {round(clock / dt): clock for clock in clocks}
    if any(abs(step * dt - clock) > 1e-12 or step > switching or step < 0
           for step, clock in clock_steps.items()):
        raise ValueError("Readout clock lies outside the simulated common prefix grid.")

    n, actions = len(parameters), len(plans)
    coordinates = (np.arange(cells) + .5) / cells
    diffusion, rate, capacity = material_fields(parameters, coordinates)
    masks = [((coordinates >= lo) & (coordinates <= hi)) for lo, hi in cfg["regions"]]
    if not all(np.any(mask) for mask in masks):
        raise ValueError("An outcome region has no cells.")
    operators_prefix = prepare_step(cells, diffusion, dt, cfg["prefix_feed"], np.zeros(n))
    split = {phase: np.broadcast_to([plan[phase + "_split"] for plan in plans], (n, actions))
             for phase in ("early", "late")}
    early_length = np.array([fed * plan["early_fraction"] for plan in plans])
    if np.any(abs(early_length - np.round(early_length)) > 1e-10):
        raise ValueError("Supply split must fall on a time-grid edge.")
    feed = cfg["primary_feed"]
    operator_early = prepare_step(cells, diffusion[:, None], dt, feed, split["early"])
    operator_late = prepare_step(cells, diffusion[:, None], dt, feed, split["late"])
    operator_off = prepare_step(cells, diffusion[:, None], dt, 0., np.zeros((n, actions)))

    def step(free, bound, rate_field, site_capacity, operator):
        pivot = cfg["goal"]["threshold"]
        mid_free, mid_bound = binding(free, bound, rate_field, site_capacity, dt / 2, beta, pivot)
        multiplier, add = operator
        diffused = idct(multiplier * dct(mid_free, norm="ortho", axis=-1) + add, norm="ortho", axis=-1)
        if diffused.min() < -1e-9:
            raise ArithmeticError("Diffusion produced negative free solute.")
        return binding(np.maximum(diffused, 0), mid_bound, rate_field, site_capacity, dt / 2, beta, pivot)

    free, bound = np.zeros_like(rate), np.zeros_like(rate)
    snapshots = {}
    for index in range(switching + 1):
        if index in clock_steps:
            snapshots[clock_steps[index]] = np.array([np.interp(cfg["probe_locations"], coordinates, row)
                                                     for row in free])
        if index < switching:
            free, bound = step(free, bound, rate, capacity, operators_prefix)

    path = np.empty((n, actions, len(t) - switching, 2))
    free = np.broadcast_to(free[:, None], (n, actions, cells)).copy()
    bound = np.broadcast_to(bound[:, None], free.shape).copy()
    max_error = 0.
    for offset in range(path.shape[2]):
        path[:, :, offset] = np.stack([bound[..., mask].mean(axis=-1) for mask in masks], axis=-1)
        dose = cfg["prefix_feed"] * cfg["prefix_duration"] + min(offset, fed) * dt * feed
        max_error = max(max_error, float(np.max(np.abs((free + capacity[:, None] * bound).mean(axis=-1) - dose))))
        if offset == path.shape[2] - 1:
            break
        operator = operator_off
        if offset < fed:
            operator = (operator_early[0],
                        np.where((offset < early_length)[None, :, None], operator_early[1], operator_late[1]))
        free, bound = step(free, bound, rate[:, None], capacity[:, None], operator)
    if max_error > 1e-10 or np.any(np.diff(path, axis=2) < -1e-12) or path.min() < 0 or path.max() > 1:
        raise ArithmeticError("Reaction bounds, monotonicity or mass balance failed.")
    if np.any(path[:, :, 0] >= cfg["goal"]["threshold"]):
        raise ArithmeticError("Regional crossings during the common prefix would be left-censored.")
    return t[switching:], snapshots, path, max_error


def bank_for(parameters, plans, cfg, clocks, beta, cells, dt):
    times, reads, path, conservation = simulate(parameters, plans, cfg, clocks, beta, cells, dt)
    base_error = None
    if beta == 0:
        old_times, old_reads, old_path, _ = bank(parameters[:2], plans, cfg, cfg["primary_feed"], cells=cells, dt=dt)
        if not np.array_equal(old_times, times):
            raise ArithmeticError("The nominal time grid differs from the previous solver.")
        base_error = float(max(np.max(np.abs(old_reads[:, :2] - reads[cfg["prefix_duration"]][:2])),
                               np.max(np.abs(old_path - path[:2]))))
        if base_error > 1e-12:
            raise ArithmeticError("The nominal solver does not reproduce the frozen implementation.")
    flags, misses, terminal = evaluate(times, path, cfg["goal"], cfg["goal"]["deadline"])
    return {"reads": reads, "flags": flags, "misses": misses, "terminal": terminal,
            "mass_error": conservation, "nominal_solver_error": base_error}


def observation(source, clocks):
    return np.column_stack([source["reads"][time][:, site] for site, time in enumerate(clocks)])


def decisions(training, queries, clocks, noise, protocol, cfg):
    flags, misses = training["flags"], training["misses"]
    reference = observation(training, clocks)
    sd = protocol["noise_sd"]
    fixed = best_action(flags.mean(axis=0), np.maximum(misses, 0).mean(axis=0))
    return {
        "posterior": posterior_choices(reference, queries, sd, flags, misses),
        "quadratic": quadratic_choices(reference, queries, sd, noise, flags, misses, cfg["ridge"]),
        "fixed": np.broadcast_to(fixed, (*queries.shape[:2], len(fixed))),
    }


def compare(nominal, law_known, test, protocol, cfg, noise, passable_nominal):
    flags = test["flags"]
    oracle = flags.any(axis=1)
    groups = len(flags) // 2
    oracle_group = oracle.reshape(groups, 2, -1).mean(axis=(1, 2))
    feasible_both = oracle & passable_nominal
    rows = []
    for clock in protocol["comparisons"]:
        actual, common = clock["actual"], clock["common_time"]
        queries = observation(test, actual)[:, None] + protocol["noise_sd"] * noise[None]
        policies = {}
        for name, training in [("nominal", nominal), ("law_known", law_known)]:
            policies.update({f"{name}_{key}": value for key, value in
                             decisions(training, queries, actual, noise, protocol, cfg).items()})
            policies[f"{name}_ignore_time"] = posterior_choices(observation(training, common), queries,
                                                               protocol["noise_sd"], training["flags"],
                                                               training["misses"])
        for key, prefix, action_law in (
            ("correct_prefix_only", law_known, nominal),
            ("correct_outcomes_only", nominal, law_known),
        ):
            policies[key] = posterior_choices(observation(prefix, actual), queries,
                                              protocol["noise_sd"], action_law["flags"], action_law["misses"])
        grouped = {name: summarize(choice, flags) for name, choice in policies.items()}
        result = {}
        for name, scored in grouped.items():
            result[name] = {"pass_rate": float(scored.mean()),
                            "oracle_gap": float((oracle_group - scored).mean()),
                            "conditional_rate_where_feasible": float(scored.mean() / oracle_group.mean())
                            if oracle_group.mean() > 0 else None}
        paired = grouped["law_known_posterior"] - grouped["nominal_posterior"]
        common_mask = np.broadcast_to(feasible_both[:, None], queries.shape[:2] + (flags.shape[-1],))
        common_success = {
            name: float(np.mean(flags[np.arange(len(flags))[:, None, None], policies[name],
                                      np.arange(flags.shape[-1])][common_mask]))
            if feasible_both.any() else None
            for name in ("nominal_posterior", "law_known_posterior")
        }
        rows.append({"clock": clock["name"], "rates": result,
                     "success_given_both_laws_feasible": common_success,
                     "law_calibration_change_by_group": {key: int(value) for key, value in
                                                       zip(("better", "same", "worse"),
                                                           [(paired > 0).sum(), (paired == 0).sum(),
                                                            (paired < 0).sum()], strict=True)}})
    return {"fixed_clock_actions": flags.shape[1],
            "feasibility": feasibility(flags, test["misses"], test["terminal"], cfg["pair_tolerance"]),
            "feasibility_transition_counts": {
                "originally_attainable_lost": int(np.sum(passable_nominal & ~oracle)),
                "originally_unattainable_gained": int(np.sum(~passable_nominal & oracle)),
                "attainable_under_both_laws": int(feasible_both.sum()),
            },
            "timing": rows}


def run(protocol, cfg):
    plans = schedules(cfg)
    clocks = {time for case in protocol["comparisons"]
              for key in ("actual", "common_time") for time in case[key]}
    noise = ndtri(qmc.Sobol(d=2, scramble=True, seed=protocol["noise_seed"]).random_base2(protocol["log2_noise"]))
    development, holdout = (sampled_materials(item) for item in (cfg["development"], protocol["holdout"]))
    rows = {}
    mass_errors = {}
    for beta in protocol["beta"]:
        trained = bank_for(development, plans, cfg, clocks, beta, cfg["cells"], cfg["step"])
        tested = bank_for(holdout, plans, cfg, clocks, beta, cfg["test_cells"], cfg["test_step"])
        if beta == 0:
            nominal = trained
            passable = tested["flags"].any(axis=1)
        rows[str(beta)] = compare(nominal, trained, tested, protocol, cfg, noise, passable)
        mass_errors[str(beta)] = {"development": trained["mass_error"], "evaluation": tested["mass_error"],
                                  "development_nominal_reproduction": trained["nominal_solver_error"],
                                  "evaluation_nominal_reproduction": tested["nominal_solver_error"]}
    return {"scope": protocol["scope"], "numerical_controls": mass_errors, "kinetic_levels": rows}


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
    results = run(protocol, read_json(args.parent_protocol))
    with args.output.open("x", encoding="utf-8") as file:
        json.dump({"registered_sha256": sha256(args.protocol), **results}, file, indent=2, allow_nan=False)
        file.write("\n")


if __name__ == "__main__":
    main()
