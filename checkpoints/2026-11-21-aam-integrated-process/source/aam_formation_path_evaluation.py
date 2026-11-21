"""Saved-array evaluation for complete-history annealed SMC and matched SIR.

Import is inert. Callers must freeze the complete production/teacher queue
before supplying held reference arrays. No fitting, State initialization,
propagation, model forward call, random draw or file access occurs here.
The SIR pool is scored with every native weight, without resampling or atom
compression. Its early-history score is never described as a full-future score.
"""
from __future__ import annotations

from math import fsum

import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import t


SCHEMA = "aam-whole-history-path-evaluation-v4"
METRICS = ("early24_energy", "early_winding_brier")
LAWS = ("psi", "teacher")
INFERENCE = ("annealed_smc", "matched_sir")
MAX_POOL = 3904


def _weights(value, count):
    weights = np.asarray(value, dtype=float)
    if (weights.shape != (count,) or not np.isfinite(weights).all()
            or np.any(weights < 0.) or not 0. < weights.sum() < np.inf):
        raise ValueError("All original finite nonnegative pool weights are required.")
    return weights / weights.sum()


def _binary(value, shape, name):
    array = np.asarray(value)
    if array.shape != shape or not np.isin(array, (False, True)).all():
        raise ValueError(name + " must retain all binary winding events and non-events.")
    return array.astype(float, copy=False)


def _pair_distance(atoms, weights, block_size):
    """Exact finite empirical V-statistic, bounded block memory, no RNG.

    Every diagonal (including the zero self-distance) and every off-diagonal
    contribution remains in the empirical target. Symmetry only avoids doing
    the identical off-diagonal calculation twice; this is not an ESS correction.
    """
    pieces = []
    count = len(atoms)
    for left in range(0, count, block_size):
        stop_left = min(count, left+block_size)
        x, wx = atoms[left:stop_left], weights[left:stop_left]
        for right in range(left, count, block_size):
            stop_right = min(count, right+block_size)
            y, wy = atoms[right:stop_right], weights[right:stop_right]
            contribution = float(np.einsum("i,ij,j->", wx, cdist(x, y), wy))
            pieces.append(contribution if left == right else 2.*contribution)
    return fsum(pieces)


def score_early_prefix(early, winding, weights, *, reference_early,
                       reference_winding, scale, block_size=256):
    """Score all N complete histories' three early micro8 readouts only.

    early: (N,3,8); winding: (N,3,3); weights: (N,)
    reference_early: (3,8); reference_winding: (3,3); scale: (8,)
    Scales are the unchanged training-only eight scales from the frozen parent.
    The caller supplies actual micro summaries extracted from the complete
    paths; this numeric helper does not substitute summaries for native State.
    """
    early = np.asarray(early, dtype=float)
    truth = np.asarray(reference_early, dtype=float)
    scale = np.asarray(scale, dtype=float)
    if (early.ndim != 3 or early.shape[1:] != (3, 8)
            or not 1 <= len(early) <= MAX_POOL or not np.isfinite(early).all()
            or truth.shape != (3, 8) or not np.isfinite(truth).all()
            or scale.shape != (8,) or not np.isfinite(scale).all()
            or np.any(scale <= 0.) or type(block_size) is not int or block_size < 1):
        raise ValueError("Need complete early micro8 histories, one truth and fixed positive scales.")
    count = len(early)
    weights = _weights(weights, count)
    wind = _binary(winding, (count, 3, 3), "Predictive winding")
    wind_truth = _binary(reference_winding, (3, 3), "Reference winding")
    atoms = (early / scale).reshape(count, 24)
    observation = (truth / scale).reshape(1, 24)
    first_parts = []
    for begin in range(0, count, block_size):
        end = min(count, begin+block_size)
        first_parts.append(float(weights[begin:end] @ cdist(atoms[begin:end], observation)[:, 0]))
    first = fsum(first_parts) / np.sqrt(24.)
    half_pair = .5 * _pair_distance(atoms, weights, block_size) / np.sqrt(24.)
    probability = np.tensordot(weights, wind, axes=1)
    return {"schema": SCHEMA, "early24_energy": first-half_pair,
        "early_winding_brier": float(np.mean((probability-wind_truth)**2)),
        "energy_reference_distance": first, "energy_half_pair_spread": half_pair,
        "native_atoms": count, "positive_weight_atoms": int(np.count_nonzero(weights)),
        "atom_ESS": float(1./np.square(weights).sum()),
        "maximum_distance_block_shape": (min(count, block_size), min(count, block_size)),
        "all_native_weights_retained": True,
        "scope": "early3frames24D only;not full48D or any future score"}


def paired_interval(values, *, n_mothers):
    """Independent simulated mother is the sole unit; never suffix/time/atom."""
    values = np.asarray(values, dtype=float)
    if (type(n_mothers) is not int or n_mothers not in (16, 32)
            or values.shape[0] != n_mothers or not np.isfinite(values).all()):
        raise ValueError("Use all32production or all16teacher mothers, without dropping rows.")
    mean = values.mean(axis=0)
    half = float(t.ppf(.975, n_mothers-1))*values.std(axis=0, ddof=1)/np.sqrt(n_mothers)
    return {"n_independent_simulated_mothers": n_mothers, "degrees_of_freedom": n_mothers-1,
            "mean": mean, "paired_t95": np.stack((mean-half, mean+half), axis=-1)}


def summarize_teacher_prefix_diagnostics(scores):
    """Fixed first16 mothers, two laws, two inference methods, two metrics.

    scores shape (16,2,2,2), with law=(psi,teacher),
    inference=(annealed_smc,matched_sir), metric=(early24ES,windingBrier).
    Both inference methods here are their final64 retained complete paths,
    matching the production-prefix representation. Raw SIR-pool scores belong
    to the separately labelled compression diagnostic, not a substituted cell.
    The teacher has independent random streams and learned-SMC's registered
    temperature sequence and T; this helper cannot verify their provenance.
    """
    scores = np.asarray(scores, dtype=float)
    if scores.shape != (16, 2, 2, 2) or not np.isfinite(scores).all():
        raise ValueError("Teacher comparison requires the fixed16xlaw2xmethod2xmetric2 table.")
    return {"schema": SCHEMA, "laws": LAWS, "inference": INFERENCE, "metrics": METRICS,
        "n_independent_simulated_mothers": 16, "representation": "final64completepaths",
        "mean": scores.mean(axis=0),
        "teacher_minus_psi": {name: paired_interval(scores[:, 1, method]-scores[:, 0, method], n_mothers=16)
                              for method, name in enumerate(INFERENCE)},
        "smc_minus_sir_by_law": {name: paired_interval(scores[:, law, 0]-scores[:, law, 1], n_mothers=16)
                                 for law, name in enumerate(LAWS)},
        "interaction_teacher_minus_psi_of_smc_minus_sir": paired_interval(
            (scores[:, 1, 0]-scores[:, 1, 1])-(scores[:, 0, 0]-scores[:, 0, 1]), n_mothers=16),
        "scope": "earlyprefix only;no sixaction futures/no G;teacher is not a deployable model or posterior goldstandard",
        "interpretation": "Paired descriptive model-law/inference contrasts;not unique causal attribution or a new production outcome."}


def summarize_sir_compression(scores, *, n_mothers):
    """Before/after SIR-pool compression early scores, paired within each mother.

    scores shape (n_mothers,2,2): representation=(allMweightedpool,selected64),
    metrics as METRICS. Production uses32; each separately labelled teacher law
    uses16. The same actual reference prefix must be used for both cells.
    """
    scores = np.asarray(scores, dtype=float)
    if scores.shape != (n_mothers, 2, 2) or not np.isfinite(scores).all():
        raise ValueError("Need the complete before/after early score table.")
    return {"schema": SCHEMA, "metrics": METRICS,
        "representations": ("allMweightedpool", "selected64completepaths"),
        "n_independent_simulated_mothers": n_mothers, "mean": scores.mean(axis=0),
        "after_minus_before": paired_interval(scores[:, 1]-scores[:, 0], n_mothers=n_mothers),
        "scope": "earlyhistory compression only;no claim that64future forecasts retain full pool probability mass"}
