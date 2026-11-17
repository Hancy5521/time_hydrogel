"""Independent checks of fixed areal inventory, nominal policy and locked thickness scores."""

import argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path
from statistics import fmean
import json

import numpy as np

from audit_window_factorized_sampling import HERE, digest, fixture, read
from patch_reservoir_joint_bank import prepared_fields
from patch_thickness_catalogue_benchmark import freeze, operational_margins
from window_spatial_probe import forecast_spatial


def spatial_and_nominal(plan):
    task = read(HERE / plan["task_path"])
    case = fixture(.83, 0)
    old = forecast_spatial(case, task, plan)
    expanded = freeze(case, task, plan)
    x = old["forecasts"]["library_depth"]
    new = expanded["forecasts"]["effective_conductivity"]["original9"]
    mapping = plan["thickness"]["catalogues"]["original9"]
    assert np.array_equal(np.asarray(mapping)[np.array(x["selected"])], new["actions_full_index"])
    assert np.array_equal(old["local_readings"][1], expanded["observed_depth"])
    assert np.array_equal(old["features"], expanded["features_at_original_witness"])
    probability_error = float(np.max(abs(np.array(x["probability"]) - new["probability"])))
    assert probability_error < 2e-13

    recipe = task["mechanism"]
    known = case["known"]
    initial_areal = recipe["dose"] * known["thickness"]
    bound_fraction = 1 - np.exp(-recipe["process_time"] * recipe["process_k"])
    errors = []
    for scale in (.85, 1., 1.15):
        size = known["thickness"] * scale
        points = 48
        h = size / points
        parent = {**known, "thickness": size}
        cfg = {**recipe, "dose": initial_areal / size}
        states = np.array([[-1.28, .88], [0., 0.], [1.28, -.88]])
        prepared = prepared_fields(parent, task["actions"], cfg, states, points)
        calculated = h * prepared.sum(axis=-1)
        errors.extend(abs(calculated - bound_fraction * initial_areal).ravel().tolist())
    assert max(errors) < 1e-10

    from audit_patch_reservoir_study import extremes, normalized_clauses
    from window_risk_forecasts import elapsed
    time = elapsed(task, plan)
    wet = np.ones((1, len(time), 1, 4)) * np.array([.1, .72, .95, .15])
    weak = np.ones_like(wet) * np.array([.19, .72, .95, .06])
    multiplier = .85
    raw, conductance = operational_margins(wet.copy(), weak.copy(), time, task, multiplier)
    raw_parts = extremes((wet, weak), time, task)
    scaled = raw_parts.copy()
    for channel in (0, 1, 4):
        scaled[..., channel] /= multiplier
    empirical = normalized_clauses(scaled, task).max(axis=-1)
    assert np.array_equal(raw, normalized_clauses(raw_parts, task).max(axis=-1))
    assert np.array_equal(conductance, empirical)
    return {"nominal_probabilities_max_error": probability_error,
            "original_nominal_decisions_identical": True,
            "areal_mass_checks": len(errors), "areal_mass_max_error": max(errors),
            "fixed_area_readout_algebra_checks": empirical.size}


def rows_for_case(pred, actual, spec):
    assert pred["parent"] == actual["parent"]
    out = {}
    differences = []
    for channel, matrix in actual["margins"].items():
        assert np.isfinite(np.asarray(matrix)).all()
        out[channel] = {}
        for key, raw in pred["forecasts"][channel].items():
            idx = spec["catalogues"][key]
            values = []
            for si, targets in enumerate(matrix):
                for wi, truth in enumerate(targets):
                    i = raw["actions_full_index"][si][wi]
                    assert i in idx and len(truth) == 43
                    hits = {j for j in idx if truth[j] <= 0}
                    p = raw["probability"][si][wi]
                    brier = fmean((x - (truth[j] > 0)) ** 2 for j, x in zip(idx, p, strict=True))
                    stats = {
                        "feasible": bool(hits),
                        "success": i in hits,
                        "missed_feasible": bool(hits) and i not in hits,
                        "unreachable": not hits,
                        "positive_regret": max(truth[i], 0.) - max(min(truth), 0.),
                        "new_thickness_selected": i >= 25,
                        "all_action_brier": brier,
                    }
                    for name, result in stats.items():
                        differences.append(abs(result - actual["scores"][channel][key][name][si][wi]))
                    values.append(stats)
            out[channel][key] = values
    return out, differences


def replay(run, plan):
    paths = [run / f"{x}.json" for x in ("REGISTERED", "FROZEN", "SCORED")]
    registry, locked, scored = map(read, paths)
    hashes = {p.stem: digest(p) for p in paths}
    assert hashes["REGISTERED"] == locked["registered_sha256"] == scored["registered_sha256"]
    assert hashes["FROZEN"] == scored["frozen_sha256"]
    order = [plan["registered_utc"], locked["started_utc"], locked["completed_utc"],
             scored["started_utc"], scored["completed_utc"]]
    assert sorted(order) == order
    assert all(digest(HERE / name) == h for name, h in registry["hashes"].items())
    report = {}
    errors = []
    max_areal, max_balance = 0., 0.
    for split, preds in locked["splits"].items():
        readouts = scored["splits"][split]
        assert len(preds) == len(readouts) == plan["sizes"][split]
        assert len({p["parent"] for p in preds}) == len(preds)
        records = []
        for p, r in zip(preds, readouts, strict=True):
            max_areal = max(max_areal, p["prepared_areal_error"], r["prepared_areal_error"])
            max_balance = max(max_balance, r["inventory_closure_max"])
            cond = np.asarray(r["margins"]["effective_conductivity"])[..., :25]
            fixed = np.asarray(r["margins"]["fixed_area_conductance"])[..., :25]
            assert np.array_equal(cond, fixed)
            obs, checks = rows_for_case(p, r, plan["thickness"])
            records.append(obs)
            errors.extend(checks)
        report[split] = {}
        for channel, summary in scored["summary"][split].items():
            report[split][channel] = {}
            arms = records[0][channel]
            rng = np.random.default_rng(plan["thickness"]["uncertainty"]["seed"])
            n = len(records)
            idx = rng.integers(n, size=(plan["thickness"]["uncertainty"]["samples"], n))
            for key, vals in arms.items():
                report[split][channel][key] = {}
                for name in vals[0]:
                    v = [item[name] for r in records for item in r[channel][key]]
                    m = fmean(v)
                    errors.append(abs(m - summary["means"][key][name]))
                    if name in summary["by_request"][key]:
                        per_request = [sum(v[k::4]) for k in range(4)]
                        errors.extend(abs(i - j) for i, j in zip(
                            per_request, summary["by_request"][key][name], strict=True))
                    report[split][channel][key][name] = m
            for pair, statistics in summary["comparisons"].items():
                case, baseline = pair.split("_minus_")
                diffs = {}
                for name, target in statistics.items():
                    if name in ("feasible", "success", "missed_feasible", "positive_regret"):
                        v = np.array([
                            fmean(x[name] - y[name] for x, y in zip(r[channel][case],
                                  r[channel][baseline], strict=True))
                            for r in records
                        ])
                        calculated = (fmean(v), *np.quantile(v[idx].mean(axis=1), (.025, .975)))
                        errors.extend(abs(x - y) for x, y in zip(calculated, target, strict=True))
                        diffs[name] = calculated
                    else:
                        label = "feasible" if name.endswith("feasible") else "success"
                        reverse = name.startswith("lost_")
                        result = sum(
                            x[label] and not y[label] if not reverse else y[label] and not x[label]
                            for row in records for x, y in zip(row[channel][case],
                                row[channel][baseline], strict=True)
                        )
                        errors.append(abs(result - target))
                        diffs[name] = result
                report[split][channel][pair] = diffs
    assert max(errors) < 1e-12 and max_areal < 1e-10 and max_balance < 2e-10
    return {"matches": len(errors), "maximum_error": max(errors), "max_areal_error": max_areal,
            "max_inventory_balance_error": max_balance, "artifacts": hashes, "summary": report,
            "registered_source_hash_checks": len(plan["hashes"])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=lambda x: HERE / x)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--fixture-only", action="store_true")
    mode.add_argument("--score-only", action="store_true")
    args = parser.parse_args()
    plan = read(args.run / "REGISTERED.json")
    start = datetime.now(timezone.utc)
    result = {"started_utc": start.isoformat(),
              "started_shanghai": start.astimezone(timezone(timedelta(hours=8))).isoformat(),
              "audit_source_sha256": digest(Path(__file__))}
    if not args.score_only:
        result["physical_fixture"] = spatial_and_nominal(plan)
    if not args.fixture_only:
        result["locked_scores"] = replay(args.run, plan)
    target = HERE / "runs" / start.strftime("thickness_catalogue_audit_%Y%m%dT%H%M%S%fZ")
    target.mkdir()
    result["completed_utc"] = datetime.now(timezone.utc).isoformat()
    with (target / "RESULTS.json").open("x", encoding="utf-8") as f:
        json.dump(result, f, indent=2, allow_nan=False)
    print(target, flush=True)


if __name__ == "__main__":
    main()
