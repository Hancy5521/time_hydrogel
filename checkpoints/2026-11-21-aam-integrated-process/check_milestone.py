"""Recompute frozen AAm milestone statistics or plan from three simulated snapshots."""

import argparse
import json
from math import isclose
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
EXPECTED_BOUNDS = (0.006956975421325911, 0.03290058466715163)


def verify():
    from aam_regret_noninferiority import score_noninferiority
    from run_aam_same_atlas_integration import (
        ALPHA, GOALS, INVERSE_METHODS, PATH_METRICS, PATH_METHODS, UNITS,
    )

    results = {}
    with np.load(DATA / "SCORED_ARRAYS.npz", allow_pickle=False) as arrays:
        np.testing.assert_array_equal(arrays["units"], UNITS)
        cases = (
            ("recovery", arrays["path_losses"], PATH_METHODS, PATH_METRICS, 0.01, 12),
            ("inverse_design", arrays["losses"], INVERSE_METHODS, GOALS, 0.05, 6),
        )
        for case, expected in zip(cases, EXPECTED_BOUNDS, strict=True):
            name, losses, methods, metrics, margin, count = case
            report = score_noninferiority(
                losses, methods, metrics, methods[0], methods[1:],
                primary_margin=margin, alpha=ALPHA,
            )
            upper = report["largest_endpoint_certifiable_relative_excess"]
            if (not report["primary"]["passed"] or report["family_comparisons"] != count
                    or upper is None or not isclose(upper, expected, rel_tol=1e-9, abs_tol=5e-12)):
                raise ValueError(f"{name} differs from the frozen milestone result.")
            results[name] = {
                "passed": count,
                "comparisons": count,
                "allowed_relative_loss": margin,
                "largest_simultaneous_relative_loss_upper_bound": upper,
            }
    return {"independent_simulated_mothers": len(UNITS), "verified": True, **results}


def query_plan(raw_query):
    from aam_same_atlas_phase_plan import plan

    if not raw_query.lstrip().startswith(("[", "{")):
        raw_query = Path(raw_query).read_text(encoding="utf-8-sig")
    query = np.asarray(json.loads(raw_query), dtype=float)
    if query.shape != (3, 4) or not np.isfinite(query).all():
        raise ValueError("Supply a JSON matrix with three time rows and four finite columns.")
    config = json.loads((DATA / "DEPLOYMENT_CONFIG.json").read_text(encoding="utf-8-sig"))
    with (np.load(DATA / "FULL_BANK_ORBITS.npz", allow_pickle=False) as bank,
          np.load(DATA / "CACHE_LOSSES.npz", allow_pickle=False) as cache):
        clean = bank["clean"]
        bank_ids = bank["unit"]
        route_ids = cache["units"]
        result = plan(
            query, clean, config["noise"], bank_ids, route_ids, cache["losses"],
            frozen_route_ids=route_ids,
            population_controls=config["population_controls"], fixed_controls=config["fixed_controls"],
        )
    selection = result["methods"]["ess1024_cophase"]
    return {
        "scope": "AAm/MBAA simulation. Time and risk are in the original model units.",
        "controls": {
            goal: {
                "feed_time": float(control[0]), "left_probability": float(control[1]),
                "estimated_loss": float(risk),
            }
            for goal, control, risk in zip(
                result["goals"], selection["controls"], selection["planning_risk"], strict=True
            )
        },
        "probability_summary": {
            "whole_bank_trajectories": len(bank_ids),
            "shared_positions_per_trajectory": clean.shape[1],
            "offline_cached_trajectories": len(route_ids),
            "cached_probability_mass": result["route_retained_recovery_mass"],
            "cached_effective_trajectory_count": result["route_mother_ess"],
            "whole_bank_probability_sha256": result["full_recovery_digest"],
        },
        "new_online_native_intervals": 0,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--verify", action="store_true", help="Recompute the 18 frozen comparisons.")
    mode.add_argument("--query", metavar="JSON_OR_FILE", help="Plan from a 3 by 4 JSON matrix.")
    args = parser.parse_args()
    sys.path.insert(0, str(ROOT / "source"))
    try:
        result = verify() if args.verify else query_plan(args.query)
    except (AssertionError, OSError, KeyError, TypeError, ValueError) as error:
        parser.exit(1, f"{parser.prog}: {error}\n")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
