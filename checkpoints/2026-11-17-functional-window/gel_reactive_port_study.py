"""Freeze and compare control policies on independent synthetic material banks."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.linalg import solve
from scipy.special import ndtr, ndtri, softmax
from scipy.stats import qmc

from gel_reactive_ports import bank, first_crossings, schedules


def sampled_materials(design):
    sample = qmc.Sobol(d=5, scramble=True, seed=design["seed"]).random_base2(design["log2_pairs"])
    values = qmc.scale(sample, *np.asarray(design["ranges"]).T)
    reflected = values.copy()
    reflected[:, 3] *= -1
    return np.stack([values, reflected], axis=1).reshape(-1, 5)


def evaluate(times, paths, task, finish):
    final = paths[:, :, np.argmin(abs(times - finish))]
    reached = first_crossings(times, paths, task["threshold"])
    early, late = task["lead_window"]
    lower, upper = task["terminal_window"]
    valid = np.isfinite(reached).all(axis=-1)
    lag = np.subtract(reached[..., 1], reached[..., 0], out=np.zeros(reached.shape[:-1]), where=valid)
    lag = lag[..., None] * np.array([1, -1])
    deadline = np.clip((reached.max(axis=-1) - finish)[..., None], 0, 3)
    terminal = np.maximum(lower - final.min(axis=-1), final.max(axis=-1) - upper)
    term = (terminal / task["terminal_norm"])[..., None]
    misses = np.maximum(np.maximum(early - lag, lag - late), np.maximum(term, deadline))
    return misses <= 0., misses, final


def best_action(success, violation):
    # Maximize joint success probability, then minimize expected miss severity.
    return np.lexsort((np.maximum(violation, 0), -success), axis=-2)[..., 0, :]


def control_sign(obs):
    return obs[..., 2] - obs[..., 3] > 0


def features(data):
    span = data.shape[-1]
    i, j = np.triu_indices(span)
    return np.concatenate([np.ones((*data.shape[:-1], 1)), data, data[..., i] * data[..., j]], axis=-1)


def policy_choices(observed, noise, train, flags, misses, cfg):
    queries = observed[:, None] + noise[None] * cfg["noise"]
    strategies = {"fixed": np.broadcast_to(best_action(flags.mean(axis=0), misses.mean(axis=0)), (*queries.shape[:2], 2))}
    sd = np.hypot(*cfg["noise"][2:])
    sign_prob = ndtr((train[:, 2] - train[:, 3]) / sd)
    sign_prob = np.stack([1 - sign_prob, sign_prob], axis=0)
    sign_prob /= sign_prob.sum(axis=1, keepdims=True)
    counts = flags.shape[1:]
    matrix = flags.reshape(len(train), -1)
    severity = np.maximum(misses, 0).reshape(len(train), -1)
    partition = best_action((sign_prob @ matrix).reshape(2, *counts), (sign_prob @ severity).reshape(2, *counts))
    strategies["two_point_sign"] = partition[control_sign(queries).astype(int)]

    def posterior(ref, target, variance):
        scaled = ((target[..., None, :] - ref) / variance)**2
        weights = softmax(-.5 * scaled.sum(axis=-1), axis=-1)
        return best_action((weights @ matrix).reshape(*queries.shape[:2], *counts),
                           (weights @ severity).reshape(*queries.shape[:2], *counts))

    # Same snapshots and same candidate outcomes in these baselines.
    strategies["free_only"] = posterior(train[:, :2], queries[..., :2], cfg["noise"][:2])
    strategies["bound_only"] = posterior(train[:, 2:], queries[..., 2:], cfg["noise"][2:])
    s = np.asarray(cfg["noise"]) / np.sqrt(2.)
    pooled_train = train.reshape(-1, 2, 2).mean(axis=-1)
    pooled_query = queries.reshape(*queries.shape[:2], 2, 2).mean(axis=-1)
    strategies["pooled_fields"] = posterior(pooled_train, pooled_query, s[[0, 2]])
    strategies["local_joint"] = posterior(train, queries, cfg["noise"])

    scale = train.std(axis=0) + cfg["noise"]
    center = train.mean(axis=0)
    train_features = features((train[:, None, :] + noise[None] * cfg["noise"] - center) / scale).reshape(len(train) * len(noise), -1)
    query_features = features((queries - center) / scale)
    gram = train_features.T @ train_features + np.eye(train_features.shape[-1]) * cfg["ridge"] * len(train_features)
    direct = []
    for targets in (matrix.astype(float), severity):
        response = np.repeat(targets, len(noise), axis=0)
        dual = solve(gram, train_features.T @ response, assume_a="pos")
        direct.append((query_features @ dual).reshape(*queries.shape[:2], *counts))
    strategies["direct_quadratic"] = best_action(*direct)
    return strategies


def summarize(flags, misses, endpoints, tolerance):
    both = flags.any(axis=1).all(axis=-1)
    pairs = flags[:, :, None, 0] & flags[:, None, :, 1]
    differences = abs(endpoints[:, :, None, :] - endpoints[:, None, :, :]).max(axis=-1)
    close = (pairs & (differences <= tolerance)).any(axis=(1, 2))
    return {"states": len(both), "oracle_rate": float(flags.any(axis=1).mean()),
            "both_orders_joint_window_states": int(both.sum()),
            "both_orders_with_pairwise_equivalent_endpoint_states": int(close.sum()),
            "endpoints_history_span_median": float(np.median(np.ptp(endpoints, axis=1).max(axis=-1))),
            "endpoints_history_span_max": float(np.ptp(endpoints, axis=1).max())}


def run(cfg):
    plans = schedules(cfg)
    train = sampled_materials(cfg["development"])
    t, tr_o, tr_p, tr_error = bank(train, plans, cfg, cfg["primary_feed"])
    good, loss, _ = evaluate(t, tr_p, cfg["goal"], cfg["goal"]["deadline"])
    noise = ndtri(qmc.Sobol(d=4, scramble=True, seed=cfg["noise_seed"]).random_base2(cfg["log2_noise"]))
    groups = []
    for design in cfg["holdouts"]:
        params = sampled_materials(design)
        primary_error = None
        alternatives = []
        for level in cfg["dose_rates"]:
            time, truth_o, truth_p, mass = bank(params, plans, cfg, level, cells=cfg["test_cells"], dt=cfg["test_step"])
            for end in (cfg["goal"]["deadline"], cfg["final_time"]):
                flags, misses, ends = evaluate(time, truth_p, cfg["goal"], end)
                alternatives.append({"feed": level, "end": end, "mass_balance_max": mass,
                                     **summarize(flags, misses, ends, cfg["pair_tolerance"])})
                if level != cfg["primary_feed"] or end != cfg["goal"]["deadline"]:
                    continue
                choices = policy_choices(truth_o, noise, tr_o, good, loss, cfg)
                expect = flags.any(axis=1).mean()
                primary_error = {}
                for key, picks in choices.items():
                    i = np.arange(len(params))[:, None, None]
                    j = np.arange(2)[None, None, :]
                    success = flags[i, picks, j]
                    violation = np.maximum(misses[i, picks, j], 0)
                    primary_error[key] = {"success": float(success.mean()),
                                          "mean_violation": float(violation.mean()),
                                          "captured_feasible_fraction": float(success.mean() / expect) if expect else None}
        groups.append({"name": design["name"], "alternatives": alternatives, "policies": primary_error})
    return {"design": "Synthetic finite-volume reaction and transport, no wet observations",
            "development_parameters": len(train), "training_mass_balance_max": tr_error,
            "plans": plans, "groups": groups}


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("protocol", type=Path)
    cli.add_argument("result", type=Path)
    args = cli.parse_args()
    cfg = json.loads(args.protocol.read_text(encoding="utf-8"))
    for filename, wanted in cfg["sources"].items():
        if hashlib.sha256(Path(__file__).with_name(filename).read_bytes()).hexdigest() != wanted:
            raise ValueError(f"Registered source changed: {filename}")
    if args.result.exists():
        raise FileExistsError(args.result)
    result = run(cfg)
    with args.result.open("x", encoding="utf-8", newline="\n") as output:
        json.dump(result, output, indent=2, allow_nan=False)
        output.write("\n")


if __name__ == "__main__":
    main()
