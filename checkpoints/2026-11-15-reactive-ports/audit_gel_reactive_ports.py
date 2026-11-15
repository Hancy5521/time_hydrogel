"""Check split integration against a face-flux Radau solve of the same model."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
from scipy.sparse import diags, lil_matrix

from gel_reactive_ports import bank, first_crossings, schedules


def reference(parameters, schedule, protocol, feed_rate, cells, step, threshold):
    diffusion, rate, capacity, gradient, curvature = parameters
    centers = (np.arange(cells) + 0.5) / cells
    coordinate = 2 * centers - 1
    local_rate = rate * np.exp(gradient * coordinate + curvature * (coordinate**2 - 1 / 3))
    regions = [(centers >= lo) & (centers <= hi) for lo, hi in protocol["regions"]]
    probes = protocol["probe_locations"]

    # Free reagent couples to its two neighboring cells and to the local bound
    # fraction. Binding couples only to the local free and bound variables.
    structure = lil_matrix((2 * cells + 2, 2 * cells + 2))
    bands = diags((np.ones(cells - 1), np.ones(cells), np.ones(cells - 1)), (-1, 0, 1))
    structure[:cells, :cells] = bands
    structure[:cells, cells:2 * cells] = np.eye(cells)
    structure[cells:2 * cells, :cells] = np.eye(cells)
    structure[cells:2 * cells, cells:2 * cells] = np.eye(cells)
    structure = structure.tocsc()

    def rate_of_change(_time, state, left, right):
        free, bound = state[:cells], state[cells:2 * cells]
        # Oriented positive toward the right. The right inlet is therefore
        # negative at the outer face, so both inlets increase total reagent.
        flux = np.r_[left, -diffusion * cells * np.diff(free), -right]
        reaction = local_rate * free * (1 - bound)
        return np.r_[-cells * np.diff(flux) - reaction, reaction / capacity, left, right]

    def crossing(region):
        def event(_time, state):
            return float(state[cells:2 * cells][region].mean() - threshold)

        return event

    prefix = protocol["prefix_duration"]
    feed_duration = protocol["feed_duration"]
    finish = protocol["final_time"]
    split_time = prefix + schedule["early_fraction"] * feed_duration
    feed_end = prefix + feed_duration
    phases = (
        (0., prefix, protocol["prefix_feed"], 0.),
        (prefix, split_time, feed_rate, schedule["early_split"]),
        (split_time, feed_end, feed_rate, schedule["late_split"]),
        (feed_end, finish, 0., 0.),
    )
    grid = prefix + np.arange(round((finish - prefix) / step) + 1) * step
    paths = np.empty((len(grid), len(regions)))
    assigned = np.zeros(len(grid), dtype=bool)
    first = np.full(len(regions), np.inf)
    current = np.zeros(2 * cells + 2)
    maximum_balance_error = 0.
    for i, (start, stop, amount, split) in enumerate(phases):
        left, right = amount * (1 + split) / 2, amount * (1 - split) / 2
        solution = solve_ivp(rate_of_change, (start, stop), current, method="Radau",
                             args=(left, right), jac_sparsity=structure, dense_output=True,
                             rtol=3e-10, atol=2e-12)
        if not solution.success:
            raise ArithmeticError(solution.message)
        current = solution.y[:, -1]
        total = (solution.y[:cells] + capacity * solution.y[cells:2 * cells]).mean(axis=0)
        error = np.max(np.abs(total - solution.y[-2:].sum(axis=0)))
        maximum_balance_error = max(maximum_balance_error, float(error))
        if i == 0:
            observed = np.r_[np.interp(probes, centers, current[:cells]),
                             np.interp(probes, centers, current[cells:2 * cells])]
        inside = (~assigned) & (grid >= start - 1e-13) & (grid <= stop + 1e-13)
        values = solution.sol(np.clip(grid[inside], start, stop))[cells:2 * cells] if inside.any() else None
        for j, region in enumerate(regions):
            if values is not None:
                paths[inside, j] = values[region].mean(axis=0)
            if first[j] == np.inf:
                event = crossing(region)
                if event(start, solution.y[:, 0]) >= 0:
                    first[j] = max(prefix, start)
                elif event(stop, current) >= 0:
                    time = brentq(lambda t: event(t, solution.sol(t)), start, stop, xtol=1e-12)
                    first[j] = max(prefix, time)
        assigned |= inside
    if not assigned.all():
        raise ArithmeticError("The reference phases do not span the sample grid.")
    return grid, observed, paths, first, current[-2:], maximum_balance_error


def compare(protocol):
    plans = schedules(protocol)
    asymmetric = [plan for plan in plans if plan["early_split"] > 0]
    if not asymmetric:
        raise ValueError("The protocol must have an asymmetric plan to test reflection.")
    threshold = protocol.get("goal", {}).get("threshold", 0.4)
    low, middle, high = protocol["dose_rates"]
    cases = (
        ((.065, .85, .80, 0., 0.), plans[0], low),
        ((.13, 1.25, .65, .55, .15), asymmetric[0], middle),
        ((.22, .75, .95, -.65, -.25), asymmetric[-1], high),
    )
    records = []
    for cells in (20, 40):
        step = .0125
        for parameters, plan, dose_rate in cases:
            truth = reference(parameters, plan, protocol, dose_rate, cells, step, threshold)
            ref_times, ref_obs, ref_path, ref_hit, ref_ports, ref_mass = truth
            times, observations, paths, split_mass = bank([parameters], [plan], protocol, dose_rate,
                                                            cells=cells, dt=step)
            if not np.allclose(times, ref_times, rtol=0., atol=1e-13):
                raise AssertionError("The solvers returned different output clocks.")
            split_hit = first_crossings(times, paths[0, 0], threshold)
            finite = np.isfinite(ref_hit)
            if not np.array_equal(finite, np.isfinite(split_hit)):
                raise AssertionError("Crossing censoring differs between solvers.")
            intended = .5 * (protocol["prefix_feed"] * protocol["prefix_duration"]
                             + dose_rate * protocol["feed_duration"])
            records.append({
                "cells": cells, "dt": step, "material_parameters": parameters,
                "schedule": plan, "feed_rate": dose_rate,
                "port_doses": ref_ports.tolist(), "intended_per_port": intended,
                "max_port_dose_error": float(np.max(np.abs(ref_ports - intended))),
                "reference_mass_error": ref_mass, "split_mass_error": split_mass,
                "max_probe_difference": float(np.max(np.abs(ref_obs - observations[0]))),
                "max_trace_difference": float(np.max(np.abs(ref_path - paths[0, 0]))),
                "crossing_differences": [float(abs(a - b)) if ok else None
                                         for a, b, ok in zip(ref_hit, split_hit, finite, strict=True)],
            })

    parameters, plan, dose_rate = cases[1]
    opposite = (*parameters[:3], -parameters[3], parameters[4])
    opposite_plan = {"early_fraction": plan["early_fraction"], "early_split": -plan["early_split"],
                     "late_split": -plan["late_split"]}
    _, _, path, _ = bank([parameters], [plan], protocol, dose_rate, cells=40, dt=.0125)
    _, _, opposite_path, _ = bank([opposite], [opposite_plan], protocol, dose_rate, cells=40, dt=.0125)
    truth = reference(parameters, plan, protocol, dose_rate, 40, .0125, threshold)
    reflected = reference(opposite, opposite_plan, protocol, dose_rate, 40, .0125, threshold)
    mirror_errors = {
        "split_solver": float(np.max(np.abs(path - opposite_path[..., ::-1]))),
        "radau_solver": float(np.max(np.abs(truth[2] - reflected[2][..., ::-1]))),
    }
    keys = ("max_port_dose_error", "reference_mass_error", "split_mass_error",
            "max_probe_difference", "max_trace_difference")
    maxima = {key: max(row[key] for row in records) for key in keys}
    maxima["max_crossing_difference"] = max((hit for row in records
                                              for hit in row["crossing_differences"] if hit is not None), default=0.)
    tolerances = {"max_port_dose_error": 1e-9, "reference_mass_error": 1e-9,
                  "split_mass_error": 1e-9, "max_probe_difference": 3e-4,
                  "max_trace_difference": 3e-4, "max_crossing_difference": 2e-3}
    passed = all(maxima[key] < tol for key, tol in tolerances.items()) and max(mirror_errors.values()) < 1e-9
    return {"status": "pass" if passed else "fail", "reference": "direct FV face flux, coupled Radau",
            "threshold": threshold, "path_cases": records, "maxima": maxima,
            "tolerances": tolerances, "reflection_errors": mirror_errors}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("protocol", type=Path)
    args = parser.parse_args()
    output = compare(json.loads(args.protocol.read_text(encoding="utf-8-sig")))
    print(json.dumps(output, indent=2, allow_nan=False))
    return 0 if output["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
