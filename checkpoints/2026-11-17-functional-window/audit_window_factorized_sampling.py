"""Independent fixture quadrature and optional replay of locked window scores."""

import argparse
from datetime import datetime, timedelta, timezone
from hashlib import sha256
from pathlib import Path
from statistics import fmean
from unittest.mock import patch
import json

import numpy as np

import window_factorized_sampling as candidate
import window_risk_forecasts as weighted


HERE = Path(__file__).resolve().parent
DEFAULT_RUN = HERE / "runs" / "patch_window_factorized_20261116T141618879530Z"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def centers(span, count):
    edges = np.linspace(*span, count + 1)
    return (edges[1:] + edges[:-1]) / 2


def normalize(log_probability):
    probability = np.exp(log_probability - np.max(log_probability))
    return probability / probability.sum()


def dense_release(known, reads, plan, rate_nodes, power_nodes, time_nodes):
    spans = plan["prior_spans"]
    obs = plan["probes"]
    powers = centers(spans["power"], power_nodes)
    rates = known["release"] * np.r_[
        centers(spans["rate_slow"], rate_nodes),
        centers(spans["rate_fast"], rate_nodes),
    ]
    tau = np.array(plan["tau_grid"])
    roots, weights = np.polynomial.legendre.leggauss(time_nodes)
    posterior = []
    for state in reads:
        log_joint = np.zeros((len(powers), len(rates)))
        activity_terms = {}
        for observation in state:
            activity, stop = observation.activity, observation.stop
            times = (roots + 1) * stop / 2
            water = activity * (1 - np.exp(-times[None, :] / tau[:, None]))
            integrated = np.sum(
                water[None, :, :] ** powers[:, None, None] * weights, axis=-1
            ) * stop / 2
            log_total = -rates[None, :, None] * integrated[:, None, :]
            predicted = 1 - np.exp(log_total)
            water_obs = activity * (
                1 - np.exp(-np.array(obs["hydration_times"])[None, :] / tau[:, None])
            )
            water_chi2 = np.sum(
                ((water_obs - observation.hydration) / obs["sigma_hydration"]) ** 2,
                axis=-1,
            )
            chi2 = ((predicted - observation.total) / obs["sigma_release"]) ** 2
            activity_terms.setdefault(activity, np.zeros_like(chi2))
            activity_terms[activity] += -.5 * (chi2 + water_chi2)
        for log_terms in activity_terms.values():
            shift = log_terms.max(axis=-1)
            log_joint += shift + np.log(np.exp(log_terms - shift[..., None]).mean(axis=-1))
        posterior.append(normalize(log_joint))
    return powers, rates, np.array(posterior)


def reference_half(known, task, plan, bias, shape):
    fixed = [task["actions"][plan["fixed_action"]]]
    profile = candidate.prepared_fields(
        known, fixed, task["mechanism"], np.column_stack((bias, shape)),
        plan["prediction_mesh"][0],
    )[:, 0]
    left, right = np.split(profile, 2, axis=-1)
    return left.sum(axis=-1) / (left.sum(axis=-1) + right.sum(axis=-1))


def fixture(release, variant):
    eps = np.array([
        [[.17, -.27, .43], [-.09, .45, -.19]],
        [[-.47, .81, .23], [.39, -.35, .16]],
    ])
    return {
        "parent": f"independent-fixture-{variant}",
        "i": -(variant + 1),
        "known": {
            "water_D": .103 + .013 * variant,
            "release": release,
            "thickness": .95 + .041 * variant,
        },
        "kinetic": np.array([
            [.52 * release, 2.15 - .38 * variant, .095, .71],
            [1.64 * release, 1.65 + .32 * variant, .143, .31],
        ]),
        "stock": np.array([[-.75 + .2 * variant, -.61], [.67 - .18 * variant, .51]]),
        "tau": .16 + .031 * variant,
        "lag": 1.31 + .12 * variant,
        "slow_share": .22 if variant == 2 else 0.,
        "shift": variant == 2,
        "noise": eps,
        "half_noise": np.array([.58, -.6]),
    }


def diagnostic_fixtures(task, plan):
    out = []
    for case in (fixture(.83, 0), fixture(1.01, 1), fixture(.91, 2)):
        reads, half, _ = candidate.packet(case, plan, task)
        settings = plan["factorization"]
        p, k, mass = candidate.release_weights(case["known"], reads, plan)
        p_old, k_old, old = dense_release(
            case["known"], reads, plan, settings["rate_nodes"],
            settings["power_nodes"], 96,
        )
        assert np.max(abs(p - p_old)) < 4e-15
        assert np.max(abs(k - k_old)) < 4e-15
        likelihood_error = float(np.max(np.abs(old - mass)))
        assert likelihood_error < 2e-6

        p_high, k_high, exact = dense_release(case["known"], reads, plan, 192, 192, 96)
        dead = case["known"]["release"]
        spans = plan["prior_spans"]
        k_low, k_upper = dead * np.array([spans["rate_slow"][0], spans["rate_fast"][1]])
        rel = np.array([k_upper - k_low, *[hi - lo for lo, hi in [
            spans[k] for k in ("power", "loss", "interface", "bias", "shape")
        ]]])
        bias = centers(spans["bias"], 1024)
        shape = np.zeros_like(bias)
        profile = reference_half(case["known"], task, plan, bias, shape)
        varied = reference_half(case["known"], task, plan, bias, .87 * np.cos(bias * 3))
        sep = float(np.max(np.abs(profile - varied)))
        assert sep < 2e-14
        loading = np.array([
            normalize(-.5 * ((h - profile) / plan["probes"]["sigma_half"]) ** 2) for h in half
        ])
        wanted = np.column_stack((
            exact.sum(axis=1) @ k_high,
            exact.sum(axis=2) @ p_high,
            np.repeat(np.mean(spans["loss"]), len(half)),
            np.repeat(np.mean(spans["interface"]), len(half)),
            loading @ bias,
            np.repeat(np.mean(spans["shape"]), len(half)),
        ))
        exact_mean_delta = np.column_stack((
            mass.sum(axis=1) @ k, mass.sum(axis=2) @ p
        )) - wanted[:, :2]
        reference_slow = (exact.sum(axis=1) @ (k_high < dead))
        reference_p_k = ((exact * p_high[None, :, None]).sum(axis=1) @ k_high)
        reference_p_k -= wanted[:, 0] * wanted[:, 1]
        prior_paths = weighted.prior(case["known"], plan, True)
        weights = weighted.choice_weights(case, task, plan, prior_paths)[0]["late"][0]
        conditional_paths = np.asarray([
            np.column_stack((ke, st)) for ke, st in candidate.conditional_atoms(case, task, plan)
        ])
        assert conditional_paths.shape == (len(half), plan["continuum_count"], 6)
        assert np.all(
            ((conditional_paths[..., 0] >= dead * spans["rate_slow"][0])
             & (conditional_paths[..., 0] <= dead * spans["rate_slow"][1]))
            | ((conditional_paths[..., 0] >= dead * spans["rate_fast"][0])
               & (conditional_paths[..., 0] <= dead * spans["rate_fast"][1]))
        )
        for i, name in enumerate(("power", "loss", "interface", "bias", "shape"), start=1):
            assert np.all(
                (conditional_paths[..., i] >= spans[name][0])
                & (conditional_paths[..., i] <= spans[name][1])
            )
        prior_mean = weights @ prior_paths
        posterior_mean = conditional_paths.mean(axis=1)
        conditional_scaled_error = abs((posterior_mean - wanted) / rel)
        prior_scaled_error = abs((prior_mean - wanted) / rel)
        out.append({
            "fixture": case["parent"],
            "truth_has_assay_shift": case["shift"],
            "max_same_grid_weight_error": likelihood_error,
            "max_mean_difference_registered_vs_dense_rate_power": float(
                np.max(abs(exact_mean_delta))
            ),
            "max_half_change_varying_shape": sep,
            "reference_mean": wanted.tolist(),
            "conditional_mean": posterior_mean.tolist(),
            "prior_weighted_mean": prior_mean.tolist(),
            "observed_states": len(half),
            "prior_weight_ess": (1 / np.sum(weights ** 2, axis=1)).tolist(),
            "mean_span_scaled_abs_error": {
                "conditional": float(conditional_scaled_error.mean()),
                "weighted": float(prior_scaled_error.mean()),
            },
            "reference_slow_probability": reference_slow.tolist(),
            "conditional_slow_probability": (conditional_paths[..., 0] < dead).mean(1).tolist(),
            "prior_weighted_slow_probability": (weights @ (prior_paths[:, 0] < dead)).tolist(),
            "reference_rate_power_covariance": reference_p_k.tolist(),
            "conditional_rate_power_covariance": (
                (conditional_paths[..., 0] * conditional_paths[..., 1]).mean(1)
                - posterior_mean[:, 0] * posterior_mean[:, 1]
            ).tolist(),
            "prior_weighted_rate_power_covariance": (
                (weights @ (prior_paths[:, 0] * prior_paths[:, 1]))
                - prior_mean[:, 0] * prior_mean[:, 1]
            ).tolist(),
        })
    return out


def inverse_check():
    w = np.array([[.125, 0., .125, .25, .5], [.25, .125, 0., .5, .125]])
    q = np.array([.0009, np.nextafter(.125, 0.), .125, np.nextafter(.125, 1.),
                  .25, .5, .875, np.nextafter(1., 0.)])
    expected = np.array([np.searchsorted(np.cumsum(row), q, side="left") for row in w])
    calculated = np.array([candidate.invert(np.broadcast_to(row, (len(q), len(row))), q) for row in w])
    assert np.array_equal(expected, calculated)
    return {"exact_inverse_cdf_checks": int(expected.size), "nonempty_intervals_required": True}


def forward_count(task, plan):
    case = fixture(.83, 0)
    methods = {}

    def tally(original, which, activity, kinetics, stock, cells, dt, times, inventory=False):
        methods.setdefault(which, []).append({
            "bath": float(activity),
            "candidates": len(kinetics),
            "actions": len(task["actions"]),
            "cells": cells,
            "inventory": bool(inventory),
        })
        return original(case["known"], task["actions"], task["mechanism"], activity,
                        kinetics, stock, cells, dt, times, inventory=inventory)

    def intercept(original, which):
        def wrapper(known, actions, cfg, activity, kinetics, stock, cells, dt, times,
                    inventory=False):
            return tally(original, which, activity, kinetics, stock, cells, dt, times, inventory)
        return wrapper

    with patch.object(candidate, "joint_bank", intercept(candidate.joint_bank, "conditional")):
        new = candidate.conditional_forecast(case, task, plan)
    with patch.object(weighted, "joint_bank", intercept(weighted.joint_bank, "standard")):
        old = weighted.foresee(case, task, plan, diagnose=False)["forecasts"]
    assert old["continuum"]["forward_candidates"] == plan["continuum_count"]
    c_calls = methods["conditional"]
    p_calls = [row for row in methods["standard"] if row["candidates"] == plan["continuum_count"]]
    assert len(c_calls) == len(new["selected"]) * len(p_calls) == 4
    assert all(row["candidates"] == plan["continuum_count"] for row in c_calls)
    return {
        "calls": methods,
        "unique_latent_rows_per_mother": {
            "conditional": sum(row["candidates"] for row in c_calls) // 2,
            "prior_weighted": sum(row["candidates"] for row in p_calls) // 2,
        },
        "latent_action_bath_trajectories_per_mother": {
            "conditional": sum(row["candidates"] * row["actions"] for row in c_calls),
            "prior_weighted": sum(row["candidates"] * row["actions"] for row in p_calls),
        },
        "probability_shape": list(np.asarray(new["probability"]).shape),
    }


def locked_scores(run, plan):
    reg_hash = digest(run / "REGISTERED.json")
    frozen_hash = digest(run / "FROZEN.json")
    lock, score = read(run / "FROZEN.json"), read(run / "SCORED.json")
    assert reg_hash == lock["registered_sha256"] == score["registered_sha256"]
    assert frozen_hash == score["frozen_sha256"]
    assert plan["registered_utc"] <= lock["started_utc"] <= lock["completed_utc"] <= score["started_utc"]
    metrics = ("all_action_brier", "choice_brier", "success", "positive_regret",
               "upper_coverage", "interval_90_width")
    audited = {}
    checks = []
    for split, forecast_groups in lock["splits"].items():
        out = {arm: [] for arm in forecast_groups[0]["forecasts"]}
        scored_groups = score["splits"][split]
        assert len(scored_groups) == len(forecast_groups) == plan["sizes"][split]
        assert len({row["parent"] for row in scored_groups}) == len(scored_groups)
        for group, targets in zip(forecast_groups, scored_groups, strict=True):
            assert group["parent"] == targets["parent"]
            margins = targets["margins"]
            for arm, observations in group["forecasts"].items():
                success, selected_brier, action_brier, regret, upper, width = [], [], [], [], [], []
                low_q, _, high_q, top_q = observations["margin_q"]
                for i, state in enumerate(margins):
                    for j, window in enumerate(state):
                        a = observations["selected"][i][j]
                        prediction = observations["probability"][i][j]
                        loss = window[a]
                        success.append(loss <= 0)
                        selected_brier.append((prediction[a] - (loss > 0)) ** 2)
                        regret.append(max(0, loss) - max(0, min(window)))
                        for k, target in enumerate(window):
                            action_brier.append((prediction[k] - (target > 0)) ** 2)
                            upper.append(high_q[i][j][k] >= target)
                            width.append(top_q[i][j][k] - low_q[i][j][k])
                record = [fmean(action_brier), fmean(selected_brier), fmean(success), fmean(regret),
                          float(all(upper)), fmean(width)]
                out[arm].append(record)
                checks.extend(
                    abs(x - targets["results"][arm][k]) for k, x in zip(metrics, record, strict=True)
                )
        as_arrays = {arm: np.array(rows) for arm, rows in out.items()}
        means = {arm: dict(zip(metrics, np.mean(matrix, axis=0), strict=True))
                 for arm, matrix in as_arrays.items()}
        diffs = {}
        sample = np.random.default_rng(plan["factorization"]["bootstrap_seed"]).integers(
            len(scored_groups), size=(2000, len(scored_groups))
        )
        for arm in as_arrays.keys() - {"conditional"}:
            delta = as_arrays["conditional"] - as_arrays[arm]
            sample_delta = delta[sample].mean(axis=1)
            extremes = np.quantile(sample_delta, (.025, .975), axis=0)
            summary = np.stack((delta.mean(axis=0), *extremes), axis=1)
            diffs[arm] = dict(zip(metrics, summary.tolist(), strict=True))
            for k, vals in diffs[arm].items():
                checks.extend(abs(x - y) for x, y in zip(
                    vals, score["summary"][split]["conditional_minus"][arm][k], strict=True
                ))
        for arm, summary in means.items():
            checks.extend(abs(x - score["summary"][split]["means"][arm][k]) for k, x in summary.items())
        audited[split] = {"groups": len(forecast_groups), "means": means, "conditional_minus": diffs}
    error = max(checks)
    assert error < 1e-12
    return {
        "registered_sha256": reg_hash,
        "frozen_sha256": frozen_hash,
        "scored_sha256": digest(run / "SCORED.json"),
        "scalar_matches": len(checks), "max_discrepancy": error, "splits": audited,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--audit-locked-scores", action="store_true")
    args = parser.parse_args()
    started = datetime.now(timezone.utc)
    target = HERE / "runs" / started.strftime("window_factorized_audit_%Y%m%dT%H%M%S%fZ")
    target.mkdir()
    plan = read(args.run / "REGISTERED.json")
    task = read(HERE / plan["task_path"])
    result = {
        "actual_utc": started.isoformat(),
        "asia_shanghai": started.astimezone(timezone(timedelta(hours=8))).isoformat(),
        "files": {p.name: digest(p) for p in (Path(__file__), Path(candidate.__file__),
                                               Path(weighted.__file__), args.run / "REGISTERED.json")},
        "scope": "Hand-specified synthetic fixtures only for quadrature and forward calls. "
                 "Locked-score audit only after both artifacts already exist.",
        "inverse": inverse_check(),
        "fixtures": diagnostic_fixtures(task, plan),
        "path_budget": forward_count(task, plan),
        "locked_score_replay": locked_scores(args.run, plan) if args.audit_locked_scores else None,
    }
    with (target / "RESULTS.json").open("x", encoding="utf-8") as dst:
        json.dump(result, dst, indent=2)
    print(target)
    print("Fixture means by method", [c["mean_span_scaled_abs_error"] for c in result["fixtures"]])
    print("Physical path budget", result["path_budget"]["latent_action_bath_trajectories_per_mother"])
    if result["locked_score_replay"]:
        print("Locked replay scalar matches", result["locked_score_replay"]["scalar_matches"])


if __name__ == "__main__":
    main()
