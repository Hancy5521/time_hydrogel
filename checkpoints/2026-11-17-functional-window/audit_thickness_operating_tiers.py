"""Prespecified secondary risk tier for the same two locked thickness policies."""

import argparse
from datetime import datetime, timedelta, timezone
import json

from audit_calibrated_thickness import positive_binomial, same_replay
from audit_window_factorized_sampling import HERE, digest, read


DEPENDENCIES = (
    "audit_thickness_operating_tiers.py",
    "audit_calibrated_thickness.py",
    "audit_window_factorized_sampling.py",
)


def register(run):
    if any((run / n).exists() for n in ("POLICY_LOCK.json", "CERTIFICATE.json", "SCORED.json")):
        raise ValueError("Register the secondary risk tier before the new policy or outcomes appear.")
    now = datetime.now(timezone.utc)
    audit = HERE / "runs" / now.strftime("thickness_operating_tiers_%Y%m%dT%H%M%S%fZ")
    audit.mkdir()
    protocol = run / "REGISTERED.json"
    obj = {
        "registered_utc": now.isoformat(),
        "registered_shanghai": now.astimezone(timezone(timedelta(hours=8))).isoformat(),
        "input_run": run.relative_to(HERE).as_posix(),
        "input_registered_sha256": digest(protocol),
        "dependencies": {p: digest(HERE / p) for p in DEPENDENCIES},
        "scope": "Prespecified secondary 25% conditional risk description of the exact two "
                 "geometry catalogue policies locked from separate development mothers. "
                 "No retuning, new acceptance thresholds, alternative policy selection or "
                 "reuse of test labels. This does not replace the 10% primary analysis.",
        "primary": {
            "risk_target": .10,
            "activity_target": .10,
            "per_catalogue_delta": .025,
            "families": 2,
        },
        "comparison": {
            "risk_targets": [.10, .25],
            "activity_target": .10,
            "per_claim_delta": .0125,
            "families": 4,
            "decision": "For joint claims across either risk tier, compare all four "
                        "intersection-union p-values to .0125. Require both catalogues to pass "
                        "at the indicated risk tier to compare paired observed safe issuance "
                        "as a jointly certified operating-point result. Data-average safe "
                        "issuance comes from confirmation, not the calibration counts. "
                        "When comparison is only at the original primary tier, retain its "
                        "stated two-gate .025 threshold.",
        },
        "state_at_registration": "No development lock, calibration certificate or score file present.",
    }
    with (audit / "REGISTERED.json").open("x", encoding="utf-8") as out:
        json.dump(obj, out, indent=2, allow_nan=False)
    print(audit, flush=True)


def evaluate(audit):
    option = read(audit / "REGISTERED.json")
    run = HERE / option["input_run"]
    for path, checksum in option["dependencies"].items():
        if digest(HERE / path) != checksum:
            raise ValueError(f"Secondary risk tier source changed: {path}")
    if digest(run / "REGISTERED.json") != option["input_registered_sha256"]:
        raise ValueError("The secondary risk tier is linked to a different main protocol.")
    protocol = read(run / "REGISTERED.json")
    completed = read(run / "CERTIFICATE.json")
    if datetime.fromisoformat(completed["started_utc"]) <= datetime.fromisoformat(option["registered_utc"]):
        raise ValueError("The secondary risk tier was not fixed before calibration.")
    reference = same_replay(run, protocol)
    output = {
        "registered_sha256": digest(audit / "REGISTERED.json"),
        "actual_utc": datetime.now(timezone.utc).isoformat(),
        "source_paths": reference["artifacts"],
        "iid_assumption": "Same registered mixed-law population. Per-catalogue policy and "
                          "acceptance cutoff are frozen on the separate development role.",
        "catalogues": {},
        "paired_safe_issuance": {
            "difference": reference["paired_delta_fraction"],
            "interval": reference["paired_ci"],
        },
    }
    levels = option["comparison"]["risk_targets"]
    for name, overview in reference["gates"].items():
        row = overview["methods"][overview["primary"]]
        results = {}
        for alpha in levels:
            risk = positive_binomial(row["active_mothers"], row["unsafe_active_mothers"],
                                     alpha, "lower")
            activity = positive_binomial(row["groups"], row["active_mothers"],
                                         option["comparison"]["activity_target"], "upper")
            p = max(risk, activity)
            results[str(alpha)] = {
                "conditional_risk_p": risk,
                "activity_p": activity,
                "intersection_union_p": p,
                "passes_four_gate_adjustment": p <= option["comparison"]["per_claim_delta"],
            }
        if abs(results["0.1"]["intersection_union_p"] - row["joint_p"]) > 5e-13:
            raise ArithmeticError("The independently recomputed primary gate changed.")
        output["catalogues"][name] = {
            "frozen_primary_method": overview["primary"],
            "calibration_groups": row["groups"],
            "active_mothers": row["active_mothers"],
            "unsafe_active_mothers": row["unsafe_active_mothers"],
            "primary_pair_result": row["certified"],
            "same_policy_alternative_risk_tiers": results,
            "held_safe_issuance": reference["held_safe_issuance"][name],
        }
    output["both_pass_four_gate_adjustment"] = {
        str(level): all(r["same_policy_alternative_risk_tiers"][str(level)]["passes_four_gate_adjustment"]
                        for r in output["catalogues"].values())
        for level in levels
    }
    with (audit / "RESULTS.json").open("x", encoding="utf-8") as dst:
        json.dump(output, dst, indent=2, allow_nan=False)
    print(audit, flush=True)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("phase", choices=("register", "evaluate"))
    cli.add_argument("directory", type=lambda p: HERE / p)
    args = cli.parse_args()
    if args.phase == "register":
        register(args.directory)
    else:
        evaluate(args.directory)
