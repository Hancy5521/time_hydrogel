"""Independent catalogue capacity/selection replay using already frozen outputs."""

import argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path
from statistics import fmean
import json

import numpy as np

from audit_window_factorized_sampling import HERE, digest, fixture, read
from patch_window_probe_benchmark import outcome


def score_parent(forecast, result, plan):
    assert forecast["parent"] == result["parent"]
    old_ids = plan["catalogue_extension"]["embedded_indices"]
    rows = []
    for arm, prediction in forecast["forecasts"].items():
        choices9 = prediction["old_selection_full_indices"]
        choices25 = prediction["new_selection"]
        arm_rows = []
        for old, new, margins in zip(choices9, choices25, result["margins"], strict=True):
            for before, after, values in zip(old, new, margins, strict=True):
                assert before in old_ids and 0 <= after < len(values)
                possible_full = {i for i, v in enumerate(values) if v <= 0}
                possible_old = possible_full.intersection(old_ids)
                prev_success = before in possible_old
                full_success = after in possible_full
                prev_regret = max(0., values[before]) - max(0., min(values))
                full_regret = max(0., values[after]) - max(0., min(values))
                arm_rows.append({
                    "old_oracle": bool(possible_old),
                    "new_oracle": bool(possible_full),
                    "oracle_capacity_gain": bool(possible_full) and not possible_old,
                    "old_success": prev_success,
                    "new_success": full_success,
                    "realized_gain": int(full_success) - int(prev_success),
                    "catalogue_floor_old": not possible_old,
                    "catalogue_floor_new": not possible_full,
                    "missed_feasible_old": bool(possible_old) and not prev_success,
                    "missed_feasible_new": bool(possible_full) and not full_success,
                    "rescued_old_feasible": full_success and not prev_success and bool(possible_old),
                    "realized_new_feasible": full_success and not possible_old,
                    "degraded_previously_successful": prev_success and not full_success,
                    "old_regret_common_oracle": prev_regret,
                    "new_regret_common_oracle": full_regret,
                    "regret_change_common_oracle": full_regret - prev_regret,
                    "new_action_selected": after not in old_ids,
                })
        errors = []
        for name in arm_rows[0]:
            vals = [r[name] for r in arm_rows]
            errors.append(abs(fmean(vals) - result["metrics"][arm][name]))
            # Two states paired with the same four requests.
            errors.extend(abs(x - y) for x, y in zip(
                (sum(vals[j::4]) for j in range(4)),
                result["per_request"][arm][name],
                strict=True,
            ))
        rows.append((arm, arm_rows, errors))
    return rows


def replay(plan, locked, measured):
    all_errors = []
    compact = {}
    for split, fore in locked["splits"].items():
        actual = measured["splits"][split]
        indices = np.random.default_rng(plan["catalogue_extension"]["bootstrap_seed"]).integers(
            len(fore), size=(plan["catalogue_extension"]["bootstrap_draws"], len(fore))
        )
        assert len(fore) == len(actual) == plan["sizes"][split]
        assert len({r["parent"] for r in actual}) == len(actual)
        samples = {}
        for pred, labels in zip(fore, actual, strict=True):
            for arm, rows, err in score_parent(pred, labels, plan):
                samples.setdefault(arm, []).append(rows)
                all_errors.extend(err)
        compact[split] = {}
        for arm, sample in samples.items():
            expected = measured["summary"][split][arm]
            compact[split][arm] = {}
            for name in sample[0][0]:
                mothers = np.array([fmean(r[name] for r in parent) for parent in sample])
                mu = fmean(mothers)
                lo, hi = np.quantile(mothers[indices].mean(axis=1), [.025, .975])
                errs = [abs(mu - expected[name]["mean"]), abs(lo - expected[name]["lo"]),
                        abs(hi - expected[name]["hi"])]
                all_errors.extend(errs)
                compact[split][arm][name] = {
                    "mean": mu, "lo": float(lo), "hi": float(hi),
                    "sum": float(sum(sum(r[name] for r in parent) for parent in sample)),
                }
    assert max(all_errors) < 1e-12
    return {"matches": len(all_errors), "maximum_discrepancy": max(all_errors), "summary": compact}


def physical_embed(plan):
    dim = plan["catalogue_extension"]
    old = read(HERE / plan["task_path"])
    full = {**old, "actions": dim["new_actions"]}
    adjusted = {**plan, "fixed_action": dim["new_reference"]}
    stubs = (fixture(.91, 0), fixture(.88, 2))
    out = {}
    for stub in stubs:
        original = outcome(stub, old, plan)
        expanded = outcome(stub, full, adjusted)
        ref = np.asarray(original["margins"])
        measured = np.asarray(expanded["margins"])[..., dim["embedded_indices"]]
        max_error = np.abs(ref - measured).max().item()
        assert max_error < 1e-12
        out[stub["parent"]] = {"max_truth_error": max_error, "margins_compared": ref.size}
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=lambda path: HERE / path)
    args = parser.parse_args()
    start = datetime.now(timezone.utc)
    names = ("REGISTERED", "FROZEN", "SCORED")
    files = tuple(args.run / f"{key}.json" for key in names)
    hashes = {key: digest(file) for key, file in zip(names, files, strict=True)}
    plan, locked, scored = map(read, files)
    assert hashes["REGISTERED"] == locked["registered_sha256"] == scored["registered_sha256"]
    assert hashes["FROZEN"] == scored["frozen_sha256"]
    ordering = (plan["registered_utc"], locked["started_utc"], locked["completed_utc"],
                scored["started_utc"], scored["completed_utc"])
    assert list(ordering) == sorted(ordering)
    for file, h in plan["hashes"].items():
        assert digest(HERE / file) == h
    witness = max(row["embedded_max_prediction_difference"]
                  for split in locked["splits"].values() for row in split)
    result = {
        "started_utc": start.isoformat(),
        "started_shanghai": start.astimezone(timezone(timedelta(hours=8))).isoformat(),
        "source_sha256": digest(Path(__file__)),
        "fixture_source_sha256": digest(HERE / "audit_window_factorized_sampling.py"),
        "held_source_hashes": hashes,
        "protocol_inputs_checked": len(plan["hashes"]),
        "embedded_frozen_prediction_error": witness,
        "physical_embed_fixtures": physical_embed(plan),
        "locked_score_replay": replay(plan, locked, scored),
    }
    target = HERE / "runs" / start.strftime("nested_catalogue_audit_%Y%m%dT%H%M%S%fZ")
    target.mkdir()
    result["completed_utc"] = datetime.now(timezone.utc).isoformat()
    with (target / "RESULTS.json").open("x", encoding="utf-8") as out:
        json.dump(result, out, indent=2, allow_nan=False)
    print(target, flush=True)


if __name__ == "__main__":
    main()

