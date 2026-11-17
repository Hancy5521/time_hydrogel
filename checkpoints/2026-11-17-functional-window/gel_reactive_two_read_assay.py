"""Select a two-read assay on development simulations and lock it before evaluation."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.fft import dct, idct
from scipy.special import ndtri
from scipy.stats import qmc

from gel_reactive_port_kinetic_check import bank_for, binding
from gel_reactive_port_study import best_action, sampled_materials
from gel_reactive_port_timing import quadratic_choices, read_json, sha256, summarize
from gel_reactive_port_twin_limit import paired_limit
from gel_reactive_port_uncertain_kinetics import conditional_outcomes
from gel_reactive_ports import material_fields, prepare_step, schedules


def prefix_fields(parameters, cfg, times, sites, beta, cells, dt):
    """Observe soluble c and bound fraction q in the common prefix."""
    ticks = {round(time / dt): time for time in times}
    if len(ticks) != len(times) or any(abs(i * dt - t) > 1e-12 or not 0 <= t <= cfg["prefix_duration"]
                                       for i, t in ticks.items()):
        raise ValueError("Assay times must lie on the common prefix grid.")
    site_coordinates = np.asarray(sites)
    grid = (np.arange(cells) + .5) / cells
    if np.any((site_coordinates < grid[0]) | (site_coordinates > grid[-1])):
        raise ValueError("An assay site lies outside the cell-center span.")
    diffusion, rate, cap = material_fields(parameters, grid)
    exact, source = prepare_step(cells, diffusion, dt, cfg["prefix_feed"], np.zeros(len(parameters)))
    f, q = np.zeros_like(rate), np.zeros_like(rate)
    values = {}
    pivot = cfg["goal"]["threshold"]
    stop = max(ticks)
    for tick in range(stop + 1):
        if tick in ticks:
            values[ticks[tick]] = np.stack([
                np.array([np.interp(site_coordinates, grid, row) for row in field])
                for field in (f, q)
            ], axis=1)
        if tick != stop:
            f, q = binding(f, q, rate, cap, dt / 2, beta, pivot)
            f = idct(exact * dct(f, norm="ortho", axis=-1) + source, norm="ortho", axis=-1)
            if f.min() < -1e-9:
                raise ArithmeticError("Negative free concentration in the shared prefix.")
            f, q = binding(np.maximum(f, 0), q, rate, cap, dt / 2, beta, pivot)
    error = np.max(np.abs((f + cap * q).mean(axis=1) - cfg["prefix_feed"] * stop * dt))
    if error > 1e-10 or q.min() < 0 or q.max() > 1:
        raise ArithmeticError("Shared prefix violates reaction bounds or mass balance.")
    return values


def bank(parameters, beta, ctx, cells, dt):
    cfg, plans, times, sites = (ctx[k] for k in ("cfg", "plans", "times", "sites"))
    output = bank_for(parameters, plans, cfg, times, beta, cells, dt)
    fields = prefix_fields(parameters, cfg, times, sites, beta, cells, dt)
    indices = [sites.index(round(site, 10)) for site in cfg["probe_locations"]]
    for time, values in fields.items():
        if not np.allclose(values[:, 0, indices], output["reads"][time], atol=1e-12, rtol=0):
            raise ArithmeticError("New interpolation does not reproduce legacy free reads.")
    return {**output, "fields": fields}


def sites_and_times(layouts):
    all_reads = [read for item in layouts.values() for read in item]
    times = sorted(set(time for _, _, time in all_reads))
    sites = sorted(set(round(x, 10) for _, site, _ in all_reads for x in (site, 1 - site)))
    return sites, times


def orientations(layout):
    reflected = [(phase, round(1 - site, 10), time) for phase, site, time in layout]
    yield layout
    if sorted(map(tuple, layout)) != sorted(reflected):
        yield reflected


def observe(item, layout, sites):
    phase_index = {"c": 0, "q": 1}
    return np.column_stack([item["fields"][time][:, phase_index[phase], sites.index(site)]
                            for phase, site, time in layout])


def noise_scale(layout, free_sd, bound_sd):
    return np.array([{"c": free_sd, "q": bound_sd}[phase] for phase, _, _ in layout])


def train_policy(library, layout, sites):
    return (
        np.stack([observe(item, layout, sites) for item in library]),
        np.stack([case["flags"] for case in library]),
        np.stack([case["misses"] for case in library]),
    )


def evaluate_layout(library, tests, layout, ctx, bound_sd):
    noise, sites, cfg = (ctx[k] for k in ("noise", "sites", "cfg"))
    output = {beta: [] for beta in tests}
    for view in orientations(layout):
        reference, flags, misses = train_policy(library, view, sites)
        sd = noise_scale(view, ctx["free_sd"], bound_sd)
        for beta, test in tests.items():
            query = observe(test, view, sites)[:, None] + sd * noise[None]
            weights, success, violation = conditional_outcomes(reference, query, sd, flags, misses)
            shape = (-1, *flags.shape[-2:])
            mixture = best_action((weights[..., None, None] * success).sum(axis=-3),
                                  (weights[..., None, None] * violation).sum(axis=-3))
            quadratic = quadratic_choices(reference.reshape(-1, 2), query, sd, noise,
                                           flags.reshape(shape), misses.reshape(shape), cfg["ridge"])
            output[beta].append({
                "posterior": float(summarize(mixture, test["flags"]).mean()),
                "quadratic": float(summarize(quadratic, test["flags"]).mean()),
            })
    return {beta: {method: float(np.mean([row[method] for row in rows])) for method in rows[0]}
            for beta, rows in output.items()}


def twin_bank(design, ctx):
    cfg = ctx["cfg"]
    first = sampled_materials(design)
    second = first.copy()
    pivot = cfg["goal"]["threshold"]
    second[:, 1] *= (1 + 2 * pivot) / (1 + 5 * pivot)
    return (bank(first, 2, ctx, cfg["test_cells"], cfg["test_step"]),
            bank(second, 5, ctx, cfg["test_cells"], cfg["test_step"]))


def conflict_limit(twins, layout, ctx, bound_sd):
    first, second = twins
    values = []
    for view in orientations(layout):
        sd = noise_scale(view, ctx["free_sd"], bound_sd)
        distance = np.linalg.norm((observe(first, view, ctx["sites"]) - observe(second, view, ctx["sites"])) / sd, axis=1)
        limits = paired_limit(first["flags"], second["flags"], distance)
        values.append(limits["two_probe_limit"][limits["incompatible_actions"]])
    if not values[0].size:
        raise ArithmeticError("This matched-rate design contains no action-conflict cases.")
    return {"conflicting_goals": int(values[0].size),
            "informed_upper_bound": float(np.mean(values)),
            "by_orientation": [float(value.mean()) for value in values]}


def mean_over(rows, beta, method):
    return float(np.mean([rows[shape][method] for shape in beta]))


def develop(protocol, ctx):
    cfg = ctx["cfg"]
    probe = sampled_materials(cfg["development"])
    library = [bank(probe, beta, ctx, cfg["cells"], cfg["step"]) for beta in protocol["hypotheses"]]
    selection = sampled_materials(protocol["selector"])
    checks = {beta: bank(selection, beta, ctx, cfg["test_cells"], cfg["test_step"])
              for beta in protocol["selector_beta"]}
    twins = twin_bank(protocol["selector_twins"], ctx)
    candidates = protocol["layouts"]
    sd = protocol["development_q_sd"]
    development = {}
    for name, layout in candidates.items():
        rate = evaluate_layout(library, checks, layout, ctx, sd)
        development[name] = {
            "action_scores": rate,
            "posterior_mean": mean_over(rate, protocol["selector_beta"], "posterior"),
            "quadratic_mean": mean_over(rate, protocol["selector_beta"], "quadratic"),
            **conflict_limit(twins, layout, ctx, sd),
        }

    def pick(metric, names=development):
        return min(names, key=lambda k: (-development[k][metric], k))

    free_only = [name for name, reads in candidates.items() if all(phase == "c" for phase, _, _ in reads)]
    with_conversion = [name for name in candidates if name not in free_only]
    selected = {"reference": protocol["reference_layout"],
                "max_conflict_separation": pick("informed_upper_bound"),
                "best_posterior_on_development": pick("posterior_mean"),
                "best_quadratic_on_development": pick("quadratic_mean"),
                "best_free_only_on_development": pick("posterior_mean", free_only),
                "best_with_conversion_on_development": pick("posterior_mean", with_conversion)}
    return library, development, selected


def verify(protocol, ctx, library, selected):
    cfg = ctx["cfg"]
    evaluation = sampled_materials(protocol["holdout"])
    tests = {beta: bank(evaluation, beta, ctx, cfg["test_cells"], cfg["test_step"])
             for beta in protocol["held_beta"]}
    twins = twin_bank(protocol["held_twins"], ctx)
    flags = np.concatenate([case["flags"] for case in library])
    positive = np.concatenate([np.maximum(case["misses"], 0) for case in library])
    fixed = best_action(flags.mean(axis=0), positive.mean(axis=0))
    scores = {}
    for label in sorted(set(selected.values())):
        layout = protocol["layouts"][label]
        scores[label] = []
        contains_bound_read = any(phase == "q" for phase, _, _ in layout)
        for sd in protocol["held_q_sd"]:
            if not contains_bound_read and scores[label]:
                scores[label].append({**scores[label][0], "noise_q_sd": sd})
                continue
            score = evaluate_layout(library, tests, layout, ctx, sd)
            scores[label].append({
                "noise_q_sd": sd,
                "action_scores": score,
                "interpolation_mean": {method: mean_over(score, protocol["main_beta"], method)
                                       for method in ("posterior", "quadratic")},
                **conflict_limit(twins, layout, ctx, sd),
            })
    return {
        "layouts": scores,
        "selected": selected,
        "observations_per_decision": 2,
        "held_mirror_groups": len(evaluation) // 2,
        "twin_mirror_groups": len(twins[0]["flags"]) // 2,
        "physical_oracle": {beta: float(case["flags"].any(axis=1).mean()) for beta, case in tests.items()},
        "fixed_action_scores": {beta: float(case["flags"][:, fixed, np.arange(len(fixed))].mean())
                                for beta, case in tests.items()},
    }


def write_new(path, data):
    with path.open("x", encoding="utf-8") as output:
        json.dump(data, output, indent=2, allow_nan=False)
        output.write("\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("protocol", type=Path)
    parser.add_argument("parent", type=Path)
    parser.add_argument("freeze", type=Path)
    parser.add_argument("score", type=Path)
    args = parser.parse_args()
    protocol = read_json(args.protocol)
    inputs = {Path(__file__).parent / name: digest for name, digest in protocol["sources"].items()}
    inputs[args.parent] = protocol["parent_sha256"]
    if any(sha256(path) != digest for path, digest in inputs.items()):
        raise ValueError("Protocol source or parent checksum mismatch.")
    if args.freeze.exists() or args.score.exists():
        raise FileExistsError("Freeze and score outputs must be new files.")
    cfg = read_json(args.parent)
    sites, times = sites_and_times(protocol["layouts"])
    draws = ndtri(qmc.Sobol(d=2, scramble=True, seed=protocol["noise_seed"]).random_base2(protocol["log2_noise"]))
    ctx = {"cfg": cfg, "plans": schedules(cfg), "sites": sites, "times": times,
           "noise": draws, "free_sd": protocol["free_sd"]}
    library, data, selected = develop(protocol, ctx)
    write_new(args.freeze, {"registered_sha256": sha256(args.protocol),
                            "development": data, "selected": selected})
    verified = verify(protocol, ctx, library, selected)
    write_new(args.score, {"registered_sha256": sha256(args.protocol),
                           "frozen_sha256": sha256(args.freeze), **verified})


if __name__ == "__main__":
    main()
