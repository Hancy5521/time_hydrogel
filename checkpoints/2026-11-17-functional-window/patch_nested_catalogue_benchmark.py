"""Test a nested manufacturing catalogue on fresh paired mother contexts."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timedelta, timezone
from itertools import product, repeat
from pathlib import Path
from time import perf_counter

import numpy as np

from gel_reactive_port_timing import read_json, sha256
from gel_reactive_two_read_assay import write_new
from patch_window_probe_benchmark import outcome
from window_risk_cases import groups
from window_spatial_probe import forecast_spatial


ROOT = Path(__file__).resolve().parent
BASE = Path("runs/patch_window_spatial_probe_20261116T142236751931Z/REGISTERED.json")
CHEMISTRY = ("wet_transfer", "near_loading")
POLICIES = ("library_depth", "library_bulk", "library_legacy")


def register():
    base = read_json(ROOT / BASE)
    task_path = Path(base["task_path"])
    task = read_json(ROOT / task_path)
    old = [tuple(a[key] for key in CHEMISTRY) for a in task["actions"]]
    expanded = list(product((.22, .36, .50, .75, 1.), (.10, .30, .50, .70, .90)))
    embedded = [expanded.index(a) for a in old]
    if embedded != sorted(embedded) or len(embedded) != 9:
        raise ValueError("Full catalogue ordering does not preserve the 9 reference actions.")
    paths = {*base["hashes"], str(BASE), str(task_path), Path(__file__).name}
    digest = {key: sha256(ROOT / key) for key in sorted(paths)}
    now = datetime.now(timezone.utc)
    run = ROOT / "runs" / now.strftime("patch_nested_catalogue_%Y%m%dT%H%M%S%fZ")
    run.mkdir()
    plan = {
        **base,
        "scientific_question": (
            "How much of whole-window failure can a nested catalogue of manufacturing settings "
            "remove, and how much of the extra capacity can the same early packet realize?"
        ),
        "superseded_protocol": str(BASE),
        "registered_utc": now.isoformat(),
        "registered_shanghai": now.astimezone(timezone(timedelta(hours=8))).isoformat(),
        "seeds": {"test": 3290017, "shift_test": 3290057},
        "sizes": {"test": 36, "shift_test": 36},
        "hashes": digest,
        "catalogue_extension": {
            "old_actions": list(task["actions"]),
            "new_actions": [dict(zip(CHEMISTRY, setting, strict=True)) for setting in expanded],
            "embedded_indices": embedded,
            "new_reference": embedded[base["fixed_action"]],
            "sampling": "Uniform midpoint interpolation of each existing consecutive control setting. "
                        "Predefined interpolation, no target look or adaptive constraints.",
            "mechanism": "Fresh mother/noise seeds. Same-law nominal and independently solved "
                         "shifted law with already scoped nonlinear hydraulic variation. "
                         "This is a new control intervention, not a new mechanism.",
            "information": "Every catalogue uses the same legacy or same two-witness 7-read packet, "
                           "the same fixed (.50,.50) witness, same posterior and same 256 latent paths "
                           "per action/bath. The larger catalogue has more action evaluations.",
            "primary": {
                "population": "shift_test",
                "policy": "library_depth",
                "capacity": "Paired mother-bootstrap difference in exact enumerated feasibility, "
                            "same 4 requests and thresholds. Report effects by each request.",
                "value": "Paired realized success, missed feasible count and positive regret relative "
                         "to the same full-catalogue oracle for both policies.",
                "interpretation": "Capacity gain and realized gain are separate. Mark practical "
                                  "improvement for a gain of at least 0.025 absolute with the "
                                  "mother-bootstrap 95% interval above zero; otherwise give "
                                  "the signed effect and its uncertainty. No claim of continuous "
                                  "global capacity or equal total trajectory count.",
            },
            "bootstrap_seed": 924627,
            "bootstrap_draws": 2000,
        },
    }
    write_new(run / "REGISTERED.json", plan)
    print(run, flush=True)


def check_registered(run):
    plan = read_json(run / "REGISTERED.json")
    for file, expected in plan["hashes"].items():
        if sha256(ROOT / file) != expected:
            raise ValueError(f"Source or input changed since registration: {file}")
    old = read_json(ROOT / plan["task_path"])
    c = plan["catalogue_extension"]
    if old["actions"] != c["old_actions"]:
        raise ValueError("Embedded action set has changed.")
    full = {**old, "actions": c["new_actions"]}
    embedded = c["embedded_indices"]
    if [full["actions"][j] for j in embedded] != old["actions"]:
        raise ValueError("Embedded action catalogue must be nested exactly.")
    changed = {**plan, "fixed_action": c["new_reference"]}
    if full["actions"][changed["fixed_action"]] != old["actions"][plan["fixed_action"]]:
        raise ValueError("The two catalogues must have the same witness preparation.")
    return plan, changed, old, full, embedded


def freeze(group, prior_task, full_task, plan, full_plan, indices):
    # Calling the already-frozen implementation preserves the exact policy and
    # avoids recomputing a choice from insufficient forecast summary statistics.
    plain = forecast_spatial(group, prior_task, plan)
    expanded = forecast_spatial(group, full_task, full_plan)
    if plain["features"] != expanded["features"] or plain["local_readings"] != expanded["local_readings"]:
        raise AssertionError("A manufacturing extension changed its observable packet.")
    forecast = {}
    maximum = 0.
    for arm in POLICIES:
        old = plain["forecasts"][arm]
        new = expanded["forecasts"][arm]
        for field in ("probability", "margin_mean", "margin_q"):
            x = np.asarray(old[field])
            y = np.asarray(new[field])[..., indices]
            maximum = max(maximum, float(np.abs(x - y).max()))
        if old["hydraulic_scale_mean"] != new["hydraulic_scale_mean"]:
            raise AssertionError("A manufacturing extension changed the hydraulic posterior.")
        mapped = np.array(indices)[np.asarray(old["selected"])]
        forecast[arm] = {
            "old_selection_full_indices": mapped.tolist(),
            "new_selection": new["selected"],
            "old_failure_probabilities": old["probability"],
            "new_failure_probabilities": new["probability"],
            "old_margin_mean": old["margin_mean"],
            "new_margin_mean": new["margin_mean"],
        }
    if maximum > 2e-12:
        raise ArithmeticError("Embedded trajectories are inconsistent between predictor calls.")
    return {
        "parent": group["parent"],
        "features": plain["features"],
        "witness": plain["local_readings"],
        "embedded_max_prediction_difference": maximum,
        "forecasts": forecast,
    }


def summarize(records, plan):
    c = plan["catalogue_extension"]
    rng = np.random.default_rng(c["bootstrap_seed"])
    sampled_groups = rng.integers(len(records), size=(c["bootstrap_draws"], len(records)))
    result = {}
    for arm in POLICIES:
        key = list(records[0]["metrics"][arm])
        matrix = np.array([[row["metrics"][arm][metric] for metric in key] for row in records])
        boot = matrix[sampled_groups].mean(axis=1)
        summary = {
            name: {"mean": float(v), "lo": float(a), "hi": float(b)}
            for name, v, a, b in zip(key, matrix.mean(axis=0),
                                    *np.quantile(boot, [.025, .975], axis=0), strict=True)
        }
        old_hit = summary["old_success"]["mean"]
        new_hit = summary["new_success"]["mean"]
        old_reach = summary["old_oracle"]["mean"]
        new_reach = summary["new_oracle"]["mean"]
        summary["oracle_recall"] = {"old": old_hit / old_reach, "new": new_hit / new_reach}
        summary["counts_by_request"] = {}
        for name in records[0]["per_request"][arm]:
            summary["counts_by_request"][name] = np.array([
                r["per_request"][arm][name] for r in records
            ]).sum(axis=0).tolist()
        result[arm] = summary
    return result


def score(predictions, truth, old_indices):
    if predictions["parent"] != truth["parent"]:
        raise ValueError("Forecast and revealed targets have different parents.")
    margins = np.asarray(truth["margins"])
    optimum_old = np.min(margins[..., old_indices], axis=-1)
    optimum = np.min(margins, axis=-1)
    old_reach = optimum_old <= 0
    reach = optimum <= 0
    newly_possible = reach & ~old_reach
    alerts = {}
    per_request = {}
    for arm, row in predictions["forecasts"].items():
        a = np.asarray(row["old_selection_full_indices"])
        b = np.asarray(row["new_selection"])
        m0 = np.take_along_axis(margins, a[..., None], axis=-1)[..., 0]
        m1 = np.take_along_axis(margins, b[..., None], axis=-1)[..., 0]
        old = m0 <= 0
        new = m1 <= 0
        old_pool_rescued = new & ~old & old_reach
        new_pool_realized = new & newly_possible
        worsened = old & ~new
        old_regret = np.maximum(m0, 0) - np.maximum(optimum, 0)
        new_regret = np.maximum(m1, 0) - np.maximum(optimum, 0)
        indicators = {
            "old_oracle": old_reach,
            "new_oracle": reach,
            "oracle_capacity_gain": newly_possible,
            "old_success": old,
            "new_success": new,
            "realized_gain": new.astype(int) - old.astype(int),
            "catalogue_floor_old": ~old_reach,
            "catalogue_floor_new": ~reach,
            "missed_feasible_old": old_reach & ~old,
            "missed_feasible_new": reach & ~new,
            "rescued_old_feasible": old_pool_rescued,
            "realized_new_feasible": new_pool_realized,
            "degraded_previously_successful": worsened,
            "old_regret_common_oracle": old_regret,
            "new_regret_common_oracle": new_regret,
            "regret_change_common_oracle": new_regret - old_regret,
            "new_action_selected": ~np.isin(b, old_indices),
        }
        accounted = old_pool_rescued.astype(int) + new_pool_realized.astype(int) - worsened.astype(int)
        if not np.array_equal(accounted, indicators["realized_gain"]):
            raise AssertionError("The success decomposition failed to close exactly.")
        alerts[arm] = {name: float(x.mean()) for name, x in indicators.items()}
        per_request[arm] = {name: x.sum(axis=0).tolist() for name, x in indicators.items()}
    return {
        "parent": truth["parent"], "margins": truth["margins"],
        "embedded_catalogue_indices": old_indices, "metrics": alerts,
        "per_request": per_request,
        "inventory_closure_max": truth["inventory_closure_max"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("register", "freeze", "score"))
    parser.add_argument("run", type=Path, nargs="?")
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    if args.phase == "register":
        if args.run:
            parser.error("Registration selects a path from actual UTC.")
        register()
        return
    if args.run is None:
        parser.error("Provide the registered run directory.")
    run = args.run.resolve()
    target = run / {"freeze": "FROZEN.json", "score": "SCORED.json"}[args.phase]
    if target.exists():
        raise FileExistsError(target)
    plan, changed, old, full, ids = check_registered(run)
    reg_hash = sha256(run / "REGISTERED.json")
    t0 = perf_counter()
    receipt = {"registered_sha256": reg_hash, "started_utc": datetime.now(timezone.utc).isoformat(),
               "splits": {}}
    if args.phase == "score":
        frozen = run / "FROZEN.json"
        stored = read_json(frozen)
        receipt["frozen_sha256"] = sha256(frozen)
        if stored["registered_sha256"] != reg_hash:
            raise ValueError("Prediction freeze did not bind the current protocol.")
        receipt["summary"] = {}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for name in plan["sizes"]:
            groups_now = groups(name, plan, old)
            if args.phase == "freeze":
                rows = list(pool.map(freeze, groups_now, repeat(old), repeat(full),
                                     repeat(plan), repeat(changed), repeat(ids)))
            else:
                labels = pool.map(outcome, groups_now, repeat(full), repeat(changed))
                rows = [score(p, y, ids)
                        for p, y in zip(stored["splits"][name], labels, strict=True)]
                receipt["summary"][name] = summarize(rows, plan)
            print(name, args.phase, len(rows), flush=True)
            receipt["splits"][name] = rows
    receipt.update(completed_utc=datetime.now(timezone.utc).isoformat(),
                   wall_seconds=perf_counter() - t0)
    write_new(target, receipt)


if __name__ == "__main__":
    main()
