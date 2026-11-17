"""Audit mixed-mother RNG independence, packet parity and exact binomial delivery gates."""

import argparse
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from math import comb
from pathlib import Path
from statistics import fmean
import json

import numpy as np
from scipy.special import logit

from audit_window_factorized_sampling import HERE, digest, fixture, read
from calibrate_thickness_catalogue import forecast_packet
from patch_thickness_catalogue_benchmark import freeze
from patch_window_event_study import current_plan


def seed_audit(plan):
    """Check actual additive addresses, including their offsets across roles."""
    src = plan["spatial_probe"]
    observed = defaultdict(list)
    all_possible = defaultdict(list)
    roles = plan["event_roles"]
    for role, spec in roles.items():
        assigned = np.random.default_rng(spec["mechanism_seed"]).random(spec["groups"])
        use_shift = assigned < spec["shift_fraction"]
        for i, shift in enumerate(use_shift):
            for s in (False, True):
                name = "shift_test" if s else "test"
                salt = src["snapshot_noise_seed"] + spec["spatial_noise_salt"]
                address = salt + i + plan["seeds"][name]
                all_possible[address].append((role, i, s))
                if shift == s:
                    observed[address].append((role, i, s))
    collisions = {str(k): v for k, v in observed.items() if len(v) > 1}
    alternatives = {str(k): v for k, v in all_possible.items() if len(v) > 1}
    return {
        "registered_groups": sum(r["groups"] for r in roles.values()),
        "observed_unique_witness_seed_addresses": len(observed),
        "observed_duplicate_witness_seed_addresses": len(collisions),
        "possible_duplicate_addresses": len(alternatives),
        "observed_duplicates_by_role": {
            role: sum(first[0] == role for first, *_ in collisions.values()) for role in roles
        },
        "observed_collisions": collisions,
        "iid_assumption_compromised_by_duplicate_witness_noise": bool(collisions),
    }


def parity_fixture(plan):
    task = read(HERE / plan["task_path"])
    profile = fixture(.924, 2)
    subprotocol = current_plan(plan, "policy_development")
    earlier = freeze(profile, task, subprotocol)
    current = forecast_packet(profile, task, subprotocol)
    assert np.array_equal(current["observed_depth"], earlier["observed_depth"])
    assert np.array_equal(current["features"], earlier["features_at_original_witness"])
    out = {}
    for key in plan["compared_catalogues"]:
        a = current["banks"][key]
        b = earlier["forecasts"]["fixed_area_conductance"][key]
        action_ids = plan["thickness"]["catalogues"][key]
        mapped = np.asarray(action_ids)[np.asarray(a["selected"])]
        assert np.array_equal(mapped, b["actions_full_index"])
        delta = abs(np.array(a["probability"]) - b["probability"])
        assert delta.max() <= 2e-13
        out[key] = {
            "probability_scalar_checks": int(delta.size),
            "max_probability_disagreement": float(delta.max()),
            "fixed_area_legacy_action_identity": True,
        }
    assert np.array_equal(
                          np.asarray(current["banks"]["grid25"]["probability"])[...,
                          plan["thickness"]["catalogues"]["original9"]],
                          np.asarray(current["banks"]["thickness27"]["probability"])[..., :9])
    return {"independent_fixture": profile["parent"], "same_witness_and_features": True, "arm": out}


def positive_binomial(n, k, p, side):
    """Exact finite count sum, independent of SciPy CDF and survival routines."""
    indices = range(0, k + 1) if side == "lower" else range(k, n + 1)
    return float(sum(comb(n, j) * p**j * (1 - p)**(n - j) for j in indices))


def gate(expected, selected, outcomes, settings):
    e = np.asarray(expected, dtype=bool)
    y = np.asarray(outcomes)
    y = np.take_along_axis(y, selected[..., None], axis=-1)[..., 0]
    active = int(e.reshape(len(e), -1).any(axis=-1).sum())
    unsafe = int((e & (y > 0)).reshape(len(e), -1).any(axis=-1).sum())
    first = positive_binomial(active, unsafe, settings["risk_target"], "lower")
    second = positive_binomial(len(e), active, settings["activity_target"], "upper")
    delta = settings["delta"]
    return {"groups": len(e), "active_mothers": active, "unsafe_active_mothers": unsafe,
            "accepted_requests": int(e.sum()), "requests": int(e.size),
            "risk_p": first, "activity_p": second, "joint_p": max(first, second),
            "certified": max(first, second) <= delta}


def rank_from_lock(forecasts, catalogue, response):
    rows = [r["banks"][catalogue] for r in forecasts]
    p = np.array([r["probability"] for r in rows])
    mean = np.array([r["margin_mean"] for r in rows])
    q = np.array([r["margin_q"] for r in rows])
    at = np.stack((logit(np.clip(p, .001, .999)), q[:, 1], q[:, 2], q[:, 3] - q[:, 0]), axis=-1)
    linear = ((at - response["location"]) / response["scale"]) @ response["coefficient"]
    linear += response["intercept"][0]

    def minimize(fields):
        result = np.zeros(p.shape[:-1], dtype=int)
        for index in np.ndindex(result.shape):
            result[index] = min(range(p.shape[-1]), key=lambda i: tuple(f[index][i] for f in fields))
        return result

    candidates = {
        "posterior_probability": (np.array([r["selected"] for r in rows]), p),
        "upper_margin": (minimize((q[:, 2], p, mean)), q[:, 2]),
        "outcome_score": (minimize((linear, q[:, 2], mean)), linear),
    }
    return {key: (chosen, np.take_along_axis(field, chosen[..., None], axis=-1)[..., 0])
            for key, (chosen, field) in candidates.items()}


def development_from_lock(banks, margins, settings):
    output = {}
    for name, (chosen, score) in banks.items():
        selected = np.take_along_axis(margins, chosen[..., None], axis=-1)[..., 0]
        right = np.unique(score)
        good = selected <= 0
        best = None
        for upper in right:
            accept = score <= upper
            active_count = accept.reshape(len(good), -1).any(axis=-1).sum()
            if active_count < settings["min_development_mothers"]:
                continue
            unsafe = (accept & ~good).reshape(len(good), -1).any(axis=-1).sum()
            if unsafe / active_count <= settings["design_risk"]:
                safe = int((accept & good).sum())
                cut = float(np.nextafter(upper, np.inf))
                key = (safe, -int(unsafe), -cut)
                best = max(key, best) if best is not None else key
        output[name] = best
    possible = [(v[0], v[1], k, -v[2]) for k, v in output.items() if v is not None]
    return max(possible)[2] if possible else "outcome_score", output


def same_replay(run, plan):
    settings = plan["event_settings"]
    files = [run / f for f in ("REGISTERED.json", "FROZEN.json", "POLICY_LOCK.json",
                               "CERTIFICATE.json", "SCORED.json")]
    checks = {p.name: digest(p) for p in files}
    registry, forecast, dev, certified, outcome = map(read, files)
    assert all(x["protocol_sha256"] == checks["REGISTERED.json"] for x in (forecast, dev, certified, outcome))
    assert all(x["frozen_sha256"] == checks["FROZEN.json"] for x in (dev, certified, outcome))
    assert certified["policy_sha256"] == checks["POLICY_LOCK.json"]
    assert outcome["certificate_sha256"] == checks["CERTIFICATE.json"]
    assert all(digest(HERE / f) == h for f, h in registry["hashes"].items())
    phases = [plan["registered_utc"]]
    for x in (forecast, dev, certified, outcome):
        phases.extend((x["started_utc"], x["completed_utc"]))
    dt = [datetime.fromisoformat(t) for t in phases]
    assert dt == sorted(dt)
    responses = read(HERE / plan["learned_policy"])["response"]

    def role_data(role, holder, ident):
        x = forecast["splits"][role]
        rows = holder["splits"][role]
        assert len(x) == len(rows) == plan["event_roles"][role]["groups"]
        assert all(a["parent"] == b["parent"] for a, b in zip(x, rows, strict=True))
        assert len({r["parent"] for r in rows}) == len(rows)
        full = np.array([r["margins"]["fixed_area_conductance"] for r in rows])
        return rank_from_lock(x, ident, responses), full[..., plan["thickness"]["catalogues"][ident]]

    gates, issued, values = {}, {}, []
    for catalogue in plan["compared_catalogues"]:
        ranks_dev, y_dev = role_data("policy_development", dev, catalogue)
        a, checked = development_from_lock(ranks_dev, y_dev, settings)
        assert dev["catalogues"][catalogue]["primary_method"] == a
        for k, best in checked.items():
            stored = dev["catalogues"][catalogue]["variants"][k]["selected"]
            if best is None:
                assert stored is None
            else:
                assert stored["cutoff"] == -best[2] and stored["successful_requests"] == best[0]
        ranks_cal, y_cal = role_data("gate_calibration", certified, catalogue)
        ranks_test, y_test = role_data("confirmation", outcome, catalogue)
        recorded = certified["catalogues"][catalogue]["calibration"]
        g = {}
        for name, (chosen, rank) in ranks_cal.items():
            t = recorded[name]
            threshold = t["locked_cutoff"]
            accept = rank < threshold if threshold is not None else np.zeros_like(rank, dtype=bool)
            result = gate(accept, chosen, y_cal, settings)
            for key, v in result.items():
                values.append(abs(v - t["fixed_policy"][key]))
            g[name] = result
        picked, scores = ranks_test[a]
        threshold = recorded[a]["locked_cutoff"]
        accepted = scores < threshold if threshold is not None else np.zeros_like(scores, dtype=bool)
        y = np.take_along_axis(y_test, picked[..., None], axis=-1)[..., 0]
        success = (accepted & (y <= 0)).sum(axis=(1, 2))
        saved = outcome["results"]["catalogues"][catalogue]
        stored_test = saved["policies"][a]
        assert np.array_equal(picked, stored_test["choice"])
        assert np.array_equal(accepted, stored_test["accept_locked"])
        assert np.array_equal(success, saved["per_mother_success"])
        assert saved["primary"] == a
        accepted_count = int(accepted.sum())
        failures = int((accepted & (y > 0)).sum())
        bad = int((accepted & (y > 0)).reshape(len(y), -1).any(axis=1).sum())
        assert accepted_count == stored_test["locked"]["accepted_requests"]
        assert int(success.sum()) == stored_test["locked"]["successful_requests"]
        assert failures == stored_test["locked"]["unsafe_accepted_requests"]
        assert bad == stored_test["locked"]["unsafe_mothers"]
        issued[catalogue] = {"primary": a, "safe_issued_total": int(success.sum()),
                             "unsafe_issued": failures, "active": int(accepted.reshape(len(y), -1).any(1).sum()),
                             "unsafe_mothers": bad,
                             "accepted": accepted_count, "per_mother": success.tolist()}
        gates[catalogue] = {"primary": a, "methods": g}
    list_a, list_b = [np.array(issued[name]["per_mother"]) for name in plan["compared_catalogues"]]
    change = list_b - list_a
    samples = np.random.default_rng(plan["integration_bootstrap_seed"]).integers(
        len(change), size=(2500, len(change))
    )
    stat = float(fmean(change) / 8)
    bounds = np.quantile(change[samples].mean(axis=1), [.025, .975]) / 8
    record = outcome["results"]
    both = all(summary["methods"][summary["primary"]]["certified"] for summary in gates.values())
    assert both == record["both_certified"]
    values.extend([abs(stat - record["safe_issued_delta_fraction"]), *abs(bounds - record["paired_ci"])])
    assert max(values) < 5e-13
    return {"artifacts": checks, "calibration_statistics_compared": len(values),
            "max_arithmetic_error": max(values), "gates": gates, "held_safe_issuance": issued,
            "paired_delta_fraction": stat, "paired_ci": bounds.tolist(), "both_certified_numeric": both}


def main():
    cmd = argparse.ArgumentParser(description=__doc__)
    cmd.add_argument("run", type=lambda x: HERE / x)
    choices = cmd.add_mutually_exclusive_group()
    choices.add_argument("--seeds-only", action="store_true")
    choices.add_argument("--fixture-only", action="store_true")
    choices.add_argument("--score-only", action="store_true")
    args = cmd.parse_args()
    now = datetime.now(timezone.utc)
    plan_path = args.run / "REGISTERED.json"
    plan = read(plan_path)
    output = {
        "started_utc": now.isoformat(),
        "started_shanghai": now.astimezone(timezone(timedelta(hours=8))).isoformat(),
        "sources": {f.name: digest(f) for f in (Path(__file__), plan_path,
            HERE / "calibrate_thickness_catalogue.py", HERE / "patch_thickness_catalogue_benchmark.py")},
        "seed_addresses": seed_audit(plan),
    }
    if args.fixture_only or not (args.seeds_only or args.score_only):
        output["fixture"] = parity_fixture(plan)
    if args.score_only or not (args.seeds_only or args.fixture_only):
        output["replayed"] = same_replay(args.run, plan)
    target = HERE / "runs" / now.strftime("calibrated_thickness_audit_%Y%m%dT%H%M%S%fZ")
    target.mkdir()
    output["completed_utc"] = datetime.now(timezone.utc).isoformat()
    with (target / "RESULTS.json").open("x", encoding="utf-8") as path:
        json.dump(output, path, indent=2, allow_nan=False)
    print(target, flush=True)


if __name__ == "__main__":
    main()

