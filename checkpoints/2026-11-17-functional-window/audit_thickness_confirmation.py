"""Replay 640-mother thickness choice, and the separate 320+320 learned-score analysis."""

import argparse
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from statistics import fmean
import json

import numpy as np
from scipy.special import logit

from audit_calibrated_thickness import development_from_lock, gate
from audit_witness_noise_addresses import addresses
from audit_window_factorized_sampling import HERE, digest, read


DEFAULT_REPLICATION = HERE / "runs/patch_window_thickness_delivery_replicate_20261117T025235612415Z"
DEFAULT_GEOMETRY = HERE / "runs/patch_window_geometry_score_20261117T025740037371Z"
BASE = HERE / "runs/patch_window_thickness_independent_20261117T023031799865Z"
INITIAL = HERE / "runs/patch_window_event_policy_20261117T020807259000Z"
STRICT = HERE / "runs/patch_window_strict_independent_20261117T023031718082Z"


def bound(path, plan):
    for name, sha in plan["hashes"].items():
        assert digest(HERE / name) == sha
    return digest(path / "REGISTERED.json")


def timestamp(value):
    return datetime.fromisoformat(value)


def noise_namespaces(replication):
    all_seeds = defaultdict(list)
    peers = (INITIAL, STRICT, BASE, replication)
    for p in peers:
        plan = read(p / "REGISTERED.json")
        kept = set(plan["event_roles"])
        if p == INITIAL:
            kept = {"policy_train", "policy_development"}
        for seed, role, idx, shift, realized in addresses(plan):
            if role in kept:
                all_seeds[seed].append((p.name, role, idx, "shift_witness" if shift else "nominal_witness"))
        for role in kept:
            item = plan["event_roles"][role]
            for name in ("nominal_seed", "shift_seed", "mechanism_seed"):
                delta = 0 if name == "mechanism_seed" else 772
                all_seeds[item[name] + delta].append((p.name, role, -1, name))
            hydraulic = item["hydraulic_salt"] + plan["spatial_probe"]["shift_variation_seed"]
            for i in range(item["groups"]):
                all_seeds[hydraulic + i].append((p.name, role, i, "hydraulic"))
                if role == "hidden_bath_stress":
                    all_seeds[plan["hidden_bath_stress"]["seed"] + i].append((p.name, role, i, "hidden_bath"))
    overlap = {k: v for k, v in all_seeds.items() if len(v) > 1}
    assert not overlap, f"{len(overlap)} overlapping RNG addresses"
    return {"potential_address_count": len(all_seeds), "duplicate_address_count": len(overlap),
            "includes_witness_hydraulics_parent_and_mixture_streams": True}


def nominal_banks(forecasts, key):
    views = [p["banks"][key] for p in forecasts]
    p = np.asarray([v["probability"] for v in views])
    mean = np.asarray([v["mean"] if "mean" in v else v["margin_mean"] for v in views])
    upper = np.asarray([v["upper"] if "upper" in v else v["margin_q"][2] for v in views])
    return p, mean, upper


def lex_smallest(fields):
    shape = fields[0].shape
    result = np.empty(shape[:-1], dtype=int)
    for i in np.ndindex(result.shape):
        values = [f[i] for f in fields]
        result[i] = min(range(shape[-1]), key=lambda a: tuple(x[a] for x in values))
    return result


def take(values, actions):
    return np.take_along_axis(values, actions[..., None], axis=-1)[..., 0]


def state_summary(margins, chosen, issued):
    negative = take(margins, chosen) <= 0.
    bad = issued & ~negative
    hit = issued & negative
    active = issued.reshape(len(issued), -1).any(axis=-1)
    unsafe = bad.reshape(len(issued), -1).any(axis=-1)
    has_action = (margins <= 0.).any(axis=-1)
    summary = {
        "mothers": len(issued),
        "requests": issued.size,
        "active_mothers": int(active.sum()),
        "unsafe_mothers": int(unsafe.sum()),
        "accepted_requests": int(issued.sum()),
        "unsafe_accepted_requests": int(bad.sum()),
        "successful_requests": int(hit.sum()),
        "requests_with_feasible_action": int(has_action.sum()),
        "raw_successes": int(negative.sum()),
        "active_fraction": fmean(active),
        "unsafe_fraction": fmean(unsafe),
        "unsafe_given_active": int(unsafe.sum()) / int(active.sum()) if active.any() else None,
        "unissued_when_frozen_choice_passes": int((negative & ~issued).sum()),
    }
    return summary, hit, has_action


def compare_numbers(independent, written, errors):
    for k, v in written.items():
        if v is None:
            assert independent[k] is None
        else:
            errors.append(abs(independent[k] - v))


def differences(left, right, seed, amount):
    flattened = left.reshape(len(left), -1), right.reshape(len(right), -1)
    delta = np.array([fmean(int(b) - int(a) for a, b in zip(*row, strict=True))
                      for row in zip(*flattened, strict=True)])
    rng = np.random.default_rng(seed)
    selections = rng.integers(len(delta), size=(amount, len(delta)))
    ci = np.quantile(delta[selections].mean(axis=1), (.025, .975))
    return float(fmean(delta)), ci, delta


def labels_margins(labels, cfg, catalogue):
    full = np.asarray([r["margins"]["fixed_area_conductance"] for r in labels])
    action = cfg["thickness"]["catalogues"][catalogue]
    return full[..., action]


def replication_audit(path):
    plan = read(path / "REGISTERED.json")
    certificate = read(HERE / plan["frozen_source"] / "CERTIFICATE.json")
    original = read(HERE / plan["frozen_source"] / "POLICY_LOCK.json")
    reg = bound(path, plan)
    f, s = (read(path / name) for name in ("FROZEN.json", "SCORED.json"))
    fhash = digest(path / "FROZEN.json")
    assert reg == f["protocol_sha256"] == s["protocol_sha256"] and fhash == s["frozen_sha256"]
    assert timestamp(certificate["completed_utc"]) < timestamp(plan["registered_utc"])
    chron = [plan["registered_utc"], f["started_utc"], f["completed_utc"],
             s["started_utc"], s["completed_utc"]]
    assert list(map(timestamp, chron)) == sorted(map(timestamp, chron))
    pred, truth = f["parents"], s["labels"]
    n = plan["event_roles"]["confirmation"]["groups"]
    ids = [r["parent"] for r in pred]
    assert len(ids) == len(set(ids)) == len(truth) == n == 640
    assert ids == [r["parent"] for r in truth]
    assert ids == [f"confirmation-{i:03d}" for i in range(n)]
    summary = {}
    counts, feasible = [], []
    errors = []
    for catalogue in plan["compared_catalogues"]:
        cutoff = plan["fixed_cutoffs"][catalogue]
        original_choice = original["catalogues"][catalogue]
        assert original_choice["primary_method"] == "upper_margin"
        assert original_choice["primary_cutoff"] == cutoff
        assert certificate["catalogues"][catalogue]["calibration"]["upper_margin"]["fixed_policy"]["certified"]
        p, mean, upper = nominal_banks(pred, catalogue)
        act = lex_smallest((upper, p, mean))
        keep = take(upper, act) < cutoff
        assert np.array_equal(act, np.asarray([r["banks"][catalogue]["choice"] for r in pred]))
        assert np.array_equal(keep, np.asarray([r["banks"][catalogue]["accept"] for r in pred]))
        given = labels_margins(truth, plan, catalogue)
        item, success, reach = state_summary(given, act, keep)
        compare_numbers(item, s["results"]["catalogues"][catalogue], errors)
        summary[catalogue] = item
        counts.append(success)
        feasible.append(reach)
    boot = plan["confirmation_stats"]
    effect, interval, row_effect = differences(*counts, boot["bootstrap_seed"], boot["bootstrap_draws"])
    written = s["results"]
    assert np.array_equal(row_effect, written["mother_effects"])
    errors.extend((abs(effect - written["paired_delta"]), *abs(interval - written["paired_ci_95"])))
    added = feasible[1] & ~feasible[0]
    assert int(added.sum()) == written["newly_feasible_requests"]
    realized = int((added & counts[1]).sum())
    assert realized == written["thickness_success_on_newly_feasible"]
    assert max(errors) < 5e-13
    return {
        "registered_sha256": reg,
        "frozen_sha256": fhash,
        "scored_sha256": digest(path / "SCORED.json"),
        "exact_numeric_values_compared": len(errors),
        "max_numerical_difference": max(errors),
        "catalogues": summary,
        "registered_paired_difference": effect,
        "registered_paired_ci": interval.tolist(),
        "newly_feasible_requests": int(added.sum()),
        "newly_feasible_safe_issuances": realized,
        "freeze_seconds": f["wall_seconds"],
        "score_seconds": s["wall_seconds"],
    }


def score_matrix(plan, key, views):
    p, mean, upper = nominal_banks(views, key)
    features = [logit(p.clip(.001, .999)), mean, upper, upper - mean]
    geometry = [tuple([v["scale"], a["wet_transfer"], a["near_loading"]])
                for v in plan["thickness"]["actions_at_scale"] for a in v["actions"]]
    c = np.asarray(geometry)[plan["thickness"]["catalogues"][key]]
    return np.concatenate((np.stack(features, axis=-1),
                           np.broadcast_to(c, (*p.shape, 3))), axis=-1), upper, mean


def recalc_score(model, x, upper, mean):
    standard = (x - model["location"]) / model["scale"]
    log_odds = standard @ model["weights"] + model["intercept"]
    pick = lex_smallest((log_odds, upper, mean))
    return pick, take(log_odds, pick)


def secondary_preflight(source, target):
    protocol = read(source / "REGISTERED.json")
    fitted = read(source / "FITTED.json")
    reg = bound(source, protocol)
    assert reg == fitted["registered_sha256"]
    assert digest(target / "REGISTERED.json") == protocol["target_protocol_sha256"]
    assert (HERE / protocol["target_study"]).resolve() == target.resolve()
    start_fit = timestamp(fitted["utc"])
    finish_fs = datetime.fromtimestamp((source / "FITTED.json").stat().st_mtime, timezone.utc)
    assert timestamp(protocol["registered_utc"]) <= start_fit <= finish_fs
    return {"registered_sha256": reg, "fit_sha256": digest(source / "FITTED.json"),
            "lock_time_register_utc": protocol["registered_utc"], "fit_started": fitted["utc"],
            "fit_observed_mtime_utc": finish_fs.isoformat(),
            "replication_outcome_present_when_audited": (target / "SCORED.json").is_file()}


def secondary_audit(source, target):
    preflight = secondary_preflight(source, target)
    registry, fitted, measured = (read(source / f) for f in ("REGISTERED.json", "FITTED.json", "EVALUATED.json"))
    upstream = HERE / registry["source_study"]
    used = {s: read(upstream / name) for s, name in (("forecasts", "FROZEN.json"),
            ("train", "CERTIFICATE.json"), ("dev", "POLICY_LOCK.json"))}
    assert used["train"]["frozen_sha256"] == digest(upstream / "FROZEN.json")
    assert used["train"]["policy_sha256"] == digest(upstream / "POLICY_LOCK.json")
    pre, label = (read(target / f) for f in ("FROZEN.json", "SCORED.json"))
    assert measured["fitted_sha256"] == preflight["fit_sha256"]
    assert measured["registered_sha256"] == preflight["registered_sha256"]
    predicted_sha = digest(target / "FROZEN.json")
    assert label["frozen_sha256"] == measured["frozen_sha256"] == predicted_sha
    assert measured["target_scored_sha256"] == digest(target / "SCORED.json")
    assert timestamp(preflight["fit_observed_mtime_utc"]) < timestamp(label["started_utc"])
    assert timestamp(label["completed_utc"]) <= timestamp(measured["utc"])
    pred, truth = pre["parents"], label["labels"]
    offset = registry["calibration_mothers"]
    assert len(pred) == offset * 2 == 640
    ids = [r["parent"] for r in truth]
    assert ids == [r["parent"] for r in pred]
    assert tuple(ids[:offset]) == tuple(f"confirmation-{i:03d}" for i in range(320))
    assert tuple(ids[offset:]) == tuple(f"confirmation-{i:03d}" for i in range(320, 640))
    result, errors = {}, []
    for name, mod in fitted["models"].items():
        model = mod["coefficient"]
        train, _, _ = score_matrix(registry, name, used["forecasts"]["splits"]["gate_calibration"])
        train = train.reshape(-1, train.shape[-1])
        avg, std = np.mean(train, axis=0), np.std(train, axis=0)
        # StandardScaler returns scale=1 on structurally constant feature columns.
        const = np.all(train == train[0], axis=0)
        std[const] = 1.
        assert np.max(abs(avg - model["location"])) < 3e-11
        assert np.max(abs(std - model["scale"])) < 3e-11
        assert train.shape[-1] == 7
        assert len(used["train"]["splits"]["gate_calibration"]) == 340
        old_dev = used["forecasts"]["splits"]["policy_development"]
        assert len(old_dev) == len(used["dev"]["splits"]["policy_development"]) == 72
        shape = score_matrix(registry, name, old_dev)
        design = recalc_score(model, *shape)
        target_dev = labels_margins(used["dev"]["splits"]["policy_development"], registry, name)
        best, detail = development_from_lock({"trained_score": design}, target_dev,
                                             registry["development_criterion"])
        best_record = detail["trained_score"]
        rule_cut = None if best_record is None else -best_record[2]
        assert rule_cut == mod["primary_cutoff"]
        pick, rank = recalc_score(model, *score_matrix(registry, name, pred))
        issued = rank < rule_cut if rule_cut is not None else np.zeros_like(rank, dtype=bool)
        y = labels_margins(truth, registry, name)
        key = measured["results"][name]
        assert np.array_equal(pick, key["frozen_actions"])
        assert np.array_equal(issued, key["frozen_accepted"])
        spec = {"risk_target": .1, "activity_target": .1, "delta": .025}
        check = gate(issued[:offset], pick[:offset], y[:offset], spec)
        compare_numbers(check, {k: key["certificate"][k] for k in check}, errors)
        item, safe, _ = state_summary(y[offset:], pick[offset:], issued[offset:])
        compare_numbers(item, key["learned"], errors)
        old_action = np.asarray([r["banks"][name]["choice"] for r in pred[offset:]])
        old_kept = np.asarray([r["banks"][name]["accept"] for r in pred[offset:]])
        ref, baseline, _ = state_summary(y[offset:], old_action, old_kept)
        compare_numbers(ref, key["upper_margin_control"], errors)
        delta, ci, _ = differences(baseline, safe, registry["bootstrap_seed"], 5000)
        errors.extend((abs(delta - key["correct_recommendation_change"]), *abs(ci - key["paired_ci"])))
        result[name] = {
            "development_cutoff": rule_cut,
            "calibration_sums_and_p_values": check,
            "held_learned": item,
            "held_upper_margin": ref,
            "paired_change": delta,
            "paired_interval": ci.tolist(),
        }
    assert max(errors) < 5e-13
    return {**preflight, "exact_numeric_values_checked": len(errors), "max_numeric_difference": max(errors),
            "evaluated_sha256": digest(source / "EVALUATED.json"), "results": result,
            "calibration_and_confirmation_are_disjoint_by_ordered_mother": True}


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("phase", choices=("preflight", "replication", "secondary", "all"))
    cli.add_argument("--replication", default=DEFAULT_REPLICATION, type=lambda p: HERE / p)
    cli.add_argument("--secondary", default=DEFAULT_GEOMETRY, type=lambda p: HERE / p)
    args = cli.parse_args()
    now = datetime.now(timezone.utc)
    out = {"started_utc": now.isoformat(), "started_shanghai": now.astimezone(timezone(timedelta(hours=8))).isoformat(),
           "audit_source_sha256": digest(Path(__file__)), "random_addresses": noise_namespaces(args.replication)}
    if args.phase == "preflight":
        out["secondary_preflight"] = secondary_preflight(args.secondary, args.replication)
    if args.phase in ("replication", "all"):
        out["replication"] = replication_audit(args.replication)
    if args.phase in ("secondary", "all"):
        out["secondary"] = secondary_audit(args.secondary, args.replication)
    root = HERE / "runs" / now.strftime("thickness_delivery_audit_%Y%m%dT%H%M%S%fZ")
    root.mkdir()
    out["completed_utc"] = datetime.now(timezone.utc).isoformat()
    with (root / "RESULTS.json").open("x", encoding="utf-8") as f:
        json.dump(out, f, indent=2, allow_nan=False)
    print(root, flush=True)


if __name__ == "__main__":
    main()
