"""Independent bootstrap SMC islands for the original three-camera path model.

Each island has an ordinary original-prior birth for every slot and makes one
original native propagation per slot and frame. Width therefore spends
simulation calls on new paths instead of retained-suffix replay. The collapsed
arm uses one common transverse phase across the whole trajectory. The raw
arm uses only the same four unshifted camera observations at each frame.
"""

from operator import index

import numpy as np
from scipy.special import logsumexp

import aam_original_tape_pgas as raw
import aam_phase_marginal_tape_pgas as phase
from run_aam_suffix_group_smoother import counts


SCHEMA = "aam-original-tape-independent-island-smc-v1"


def _systematic(probabilities, uniform):
    """Draw N ordered ancestors from one uniform over a stratum width."""
    n = len(probabilities)
    points = (np.arange(n, dtype=float) + uniform) / n
    points = np.minimum(points, np.nextafter(1., 0.))
    cdf = np.cumsum(probabilities)
    cdf[-1] = 1.
    return np.searchsorted(cdf, points, side="right")


def _raw_output(path):
    output = raw._as_output(path)
    return {**output, "clean_orbit": output["clean"][:, None, :],
            "phase_logl": output["logl"][:, None], "phase_offsets": np.zeros(1),
            "cumulative_phase_logl": np.cumsum(output["logl"])[:, None],
            "phase_weights": np.ones(1)}


def _island(cfg, y, noise, island, n, rng, calls, collapsed):
    if collapsed:
        start, leaf, extend, output = phase._start_phase, phase._leaf, phase._extend, phase._output
        increment = phase._increment
    else:
        start, leaf, extend, output = raw._start, raw._leaf, raw._extend, _raw_output
        increment = _last_raw_log_likelihood

    ordered = np.arange(n)
    uniform_base = np.full(n, -np.log(n))
    live = [start(cfg, rng, f"island_{island}_initial_{slot}") for slot in ordered]
    genealogy = ordered.copy()
    log_weights = uniform_base.copy()
    log_evidence = 0.
    stages = []
    for camera in range(3):
        incoming = raw._weights(log_weights)
        ess = float(1. / (incoming @ incoming))
        # Reserve this address even if ESS keeps identity. Thus matched raw and
        # collapsed arms use the same proposal RNG addresses after this branch.
        uniform = None if camera == 0 else float(rng.random())
        resampled = uniform is not None and ess <= n / 2
        ancestors = _systematic(incoming, uniform) if resampled else ordered
        base = uniform_base if resampled else log_weights
        children = []
        for parent_index in ancestors:
            parent = live[parent_index]
            tape = raw._table(cfg, camera, rng)
            child = leaf(cfg, y, noise, parent["packets"][-1], camera, tape, calls)
            children.append(extend(parent, tape, child, f"prior_tape_{camera + 1}"))
        genealogy = genealogy[ancestors]
        incremental = np.asarray([increment(child) for child in children])
        updated = base + incremental
        normalizer = float(logsumexp(updated))
        log_evidence += normalizer
        log_weights = updated - normalizer
        weights = raw._weights(log_weights)
        initial_mass = np.bincount(genealogy, weights=weights, minlength=n)
        stages.append({
            "camera": camera + 1, "resampled": resampled, "uniform": uniform,
            "ancestors": ancestors.copy(), "initial_slots": genealogy.copy(),
            "incoming_ess": ess, "incoming_weights": incoming, "normalized_weights": weights,
            "log_incremental_potential": incremental,
            "log_normalizer_increment": normalizer, "cumulative_log_normalizer": log_evidence,
            "output_ess": float(1. / (weights @ weights)),
            "distinct_initial_slots": int(np.count_nonzero(np.bincount(genealogy, minlength=n))),
            "weighted_initial_ess": float(1. / (initial_mass @ initial_mass)),
        })
        live = children
    terminal = tuple(output(history) for history in live)
    return {"island": island, "log_normalizer": log_evidence, "stages": stages,
            "terminal_histories": terminal, "terminal_weights": weights,
            "terminal_log_weights": log_weights, "terminal_initial_slots": genealogy}


def _last_raw_log_likelihood(path):
    return path["logl"][-1]


def forward(cfg, measured, islands, n_particles, stage_rng, *, collapsed=True):
    """Run independent islands and combine their terminal finite approximations.

    The island estimate of the normalizer for the unnormalized Gaussian
    camera potential uses the weighted mean for SIS or the arithmetic mean
    after conditionally unbiased systematic resampling. The incremental
    collapsed potential is the ratio of cumulative shared-phase integrals.
    Each normalizer estimate and each unnormalized test-function integral is
    unbiased under the ideal stochastic model. Their self-normalized finite
    mixture does not inherit that unbiasedness. The recorded log estimates
    omit the fixed Gaussian density constants and are not log-unbiased.
    """
    k, n = index(islands), index(n_particles)
    if min(k, n) < 1 or type(collapsed) is not bool:
        raise ValueError("Use positive island and particle counts and an explicit boolean collapse flag.")
    y, sigma = raw._camera_contract(cfg, measured)
    calls = counts()
    rows = tuple(_island(cfg, y, sigma, j, n, stage_rng, calls, collapsed) for j in range(k))
    expected = k * n * 3
    if calls != counts(expected):
        raise AssertionError("Every island slot must receive exactly one native propagation per frame.")
    log_z = np.asarray([row["log_normalizer"] for row in rows])
    island_weights = raw._weights(log_z)
    local_weights = np.stack([row["terminal_weights"] for row in rows])
    terminal_weights = (island_weights[:, None] * local_weights).reshape(-1)
    terminal_histories = tuple(path for row in rows for path in row["terminal_histories"])
    global_birth = np.concatenate([row["terminal_initial_slots"] + row["island"] * n for row in rows])
    initial_mass = np.bincount(global_birth, weights=terminal_weights, minlength=k*n)
    return {
        "schema": SCHEMA, "collapsed": collapsed, "n_islands": k, "particles_per_island": n,
        "calls": calls, "native_calls": expected, "islands": rows,
        "island_log_normalizers": log_z, "global_log_normalizer": float(logsumexp(log_z) - np.log(k)),
        "island_weights": island_weights, "terminal_histories": terminal_histories,
        "terminal_weights": terminal_weights, "terminal_initial_slots": global_birth,
        "weighted_initial_ess": float(1. / (initial_mass @ initial_mass)),
        "terminal_phase_weights": np.stack([path["phase_weights"] for path in terminal_histories]),
    }
