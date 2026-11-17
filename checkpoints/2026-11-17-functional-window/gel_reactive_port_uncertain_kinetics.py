"""Compare process decisions that retain or collapse uncertain gelation kinetics."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.special import logsumexp, ndtri, softmax
from scipy.stats import qmc

from gel_reactive_port_kinetic_check import bank_for, observation
from gel_reactive_port_study import best_action, sampled_materials
from gel_reactive_port_timing import quadratic_choices, read_json, sha256, summarize
from gel_reactive_ports import schedules


def conditional_outcomes(reference, queries, sd, flags, misses):
    """Return the model weight and the conditional outcome means for each law."""
    log_likelihood = -.5 * np.sum(((queries[..., None, None, :] - reference) / sd)**2, axis=-1)
    log_evidence = logsumexp(log_likelihood, axis=-1)
    # Equal training counts in every law and a uniform law prior.
    law_weights = softmax(log_evidence, axis=-1)
    state_weights = np.exp(log_likelihood - log_evidence[..., None])
    shape = (*queries.shape[:-1], len(reference), *flags.shape[2:])
    expected = [
        np.einsum("...km,kmr->...kr", state_weights, labels.reshape(*reference.shape[:2], -1),
                  optimize=True).reshape(shape)
        for labels in (flags, np.maximum(misses, 0))
    ]
    return law_weights, *expected


def policies(posterior, success, misses):
    """Use the same conditional outcome table with four law-weight rules."""
    weights = posterior[..., None, None]
    k = posterior.shape[-1]
    map_weights = np.eye(k)[posterior.argmax(axis=-1)][..., None, None]
    return {
        "posterior_mixture": best_action(np.sum(weights * success, axis=-3),
                                         np.sum(weights * misses, axis=-3)),
        "map_single_law": best_action(np.sum(map_weights * success, axis=-3),
                                      np.sum(map_weights * misses, axis=-3)),
        "uniform_law_weights": best_action(success.mean(axis=-3), misses.mean(axis=-3)),
        "nominal_beta0": best_action(success[..., 0, :, :], misses[..., 0, :, :]),
    }


def law_diagnostics(posterior, hypothesis, test_beta):
    n_laws = len(hypothesis)
    modes = posterior.argmax(axis=-1)
    result = {
        "mean_posterior_mass": posterior.mean(axis=(0, 1)).tolist(),
        "mean_peak_confidence": float(posterior.max(axis=-1).mean()),
        "mode_frequency": (np.bincount(modes.ravel(), minlength=n_laws) / modes.size).tolist(),
        "posterior_mean_beta_mae": float(np.mean(np.abs(posterior @ hypothesis - test_beta))),
    }
    if test_beta in hypothesis:
        truth = hypothesis.index(test_beta)
        probability = posterior[..., truth]
        result.update({
            "top1_accuracy": float(np.mean(modes == truth)),
            "law_nll": float(-np.mean(np.log(np.maximum(probability, np.finfo(float).tiny)))),
            "law_brier_sum": float(np.mean(np.sum((posterior - np.eye(n_laws)[truth])**2, axis=-1))),
        })
    return result


def paired_counts(a, b):
    return {"improved": int((a > b).sum()), "tied": int((a == b).sum()), "worse": int((a < b).sum())}


def compare(training, holdout, clock, noise, protocol, cfg, beta):
    reference = np.stack([observation(case, clock) for case in training])
    flags, misses = (np.stack([case[name] for case in training]) for name in ("flags", "misses"))
    shape = (len(holdout["flags"]), len(noise), holdout["flags"].shape[-1])
    sd = protocol["noise_sd"]
    queries = observation(holdout, clock)[:, None] + noise[None] * sd
    posterior, success, violation = conditional_outcomes(reference, queries, sd, flags, misses)
    commands = policies(posterior, success, violation)
    pooled_flags = flags.reshape(-1, *flags.shape[-2:])
    pooled_misses = misses.reshape(*pooled_flags.shape)
    commands["pooled_quadratic"] = quadratic_choices(reference.reshape(-1, 2), queries, sd, noise,
                                                      pooled_flags, pooled_misses, cfg["ridge"])
    fixed = best_action(pooled_flags.mean(axis=0), np.maximum(pooled_misses, 0).mean(axis=0))
    commands["pooled_fixed"] = np.broadcast_to(fixed, shape)
    point = protocol["hypotheses"]
    if beta in point:
        law = point.index(beta)
        commands["known_law_reference"] = best_action(success[..., law, :, :], violation[..., law, :, :])
    groups = {name: summarize(command, holdout["flags"]) for name, command in commands.items()}
    mixture = groups["posterior_mixture"]
    comparisons = {
        name: {
            "paired_difference": float((mixture - paired).mean()),
            "groups": paired_counts(mixture, paired),
            "action_disagreement": float(np.mean(commands["posterior_mixture"] != commands[name])),
        }
        for name, paired in groups.items() if name != "posterior_mixture"
    }
    return {
        "performance": {name: float(rates.mean()) for name, rates in groups.items()},
        "mixture_vs": comparisons,
        "law_diagnostics": law_diagnostics(posterior, point, beta)
    }


def run(protocol, cfg):
    hypotheses = protocol["hypotheses"]
    if hypotheses[0] != 0 or len(set(hypotheses)) != len(hypotheses):
        raise ValueError("A beta-zero nominal law and distinct hypotheses are required.")
    clocks = {time for layout in protocol["layouts"].values() for time in layout}
    plans = schedules(cfg)
    development = sampled_materials(cfg["development"])
    evaluation = sampled_materials(protocol["holdout"])
    noise = ndtri(qmc.Sobol(d=2, scramble=True, seed=protocol["noise_seed"]).random_base2(protocol["log2_noise"]))
    train = [bank_for(development, plans, cfg, clocks, beta, cfg["cells"], cfg["step"]) for beta in hypotheses]
    report = {}
    for beta in protocol["held_beta"]:
        held = bank_for(evaluation, plans, cfg, clocks, beta, cfg["test_cells"], cfg["test_step"])
        report[str(beta)] = {
            "oracle_rate": float(held["flags"].any(axis=1).mean()),
            "mass_error": held["mass_error"],
            "layouts": {name: compare(train, held, layout, noise, protocol, cfg, beta)
                        for name, layout in protocol["layouts"].items()},
        }
    return {
        "scope": protocol["scope"],
        "materials": len(evaluation), "mirror_groups": len(evaluation) // 2, "noise_integration_points": len(noise),
        "development_rows_per_law": len(development),
        "development_mass_errors": [case["mass_error"] for case in train],
        "held": report,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("protocol", type=Path)
    parser.add_argument("parent_protocol", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    protocol = read_json(args.protocol)
    sources = {Path(__file__).parent / name: digest for name, digest in protocol["sources"].items()}
    sources[args.parent_protocol] = protocol["parent_sha256"]
    if any(sha256(path) != digest for path, digest in sources.items()):
        raise ValueError("Frozen input or source hash mismatch.")
    result = run(protocol, read_json(args.parent_protocol))
    with args.output.open("x", encoding="utf-8") as file:
        json.dump({"registered_sha256": sha256(args.protocol), **result}, file, indent=2, allow_nan=False)
        file.write("\n")


if __name__ == "__main__":
    main()
