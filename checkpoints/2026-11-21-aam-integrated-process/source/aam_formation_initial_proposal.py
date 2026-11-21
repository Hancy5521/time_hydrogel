"""Candidate initial-State proposal for the complete AAm/MBAA cycle.

Import is inert: no file access, fit, setup, RNG draw or reaction is performed.
This module does not change the frozen V1 physical law.  It defines a small
training-only guide and importance sampling *indices* of a caller-registered,
random finite bank of complete native initial packets.  The conditional bank
target is uniform over its M entries.  Its proposal correction is 1/(M*q_j),
not an assumed continuous density for a few overwritten physical parameters.

The functions below have not been accepted as a successful scientific method
merely by being implemented.  Execution requires the new version's registration
and complete-cycle evaluation.  No query's hidden State belongs in guide inputs.
"""
from __future__ import annotations

import hashlib
from time import perf_counter

import numpy as np
from scipy.special import logsumexp


SCHEMA = "aam-formation-finite-bank-initial-proposal-v1"
GUIDE_SCHEMA = "aam-initial-three-summary-ridge1-guide-v1"
BANK_SIZE = 256
PARTICLES = 64
TRAINING_MOTHERS = 64
RIDGE_PENALTY = 1.
UNIFORM_MIXTURE = .5
SUMMARY_FIELDS = ("initial_unconsumed_initiators", "t0_left_MBAA_tracer", "t0_right_MBAA_tracer")
CAMERA_NOISE = (.08, .10, .08, .10)


def _array_sha(value):
    value = np.asarray(value)
    if value.dtype.hasobject:
        raise TypeError("No object-array evidence.")
    digest = hashlib.sha256()
    digest.update(value.dtype.str.encode("ascii"))
    digest.update(str(value.shape).encode("ascii"))
    digest.update(np.ascontiguousarray(value).tobytes())
    return digest.hexdigest()


def _summaries(value, *, rows=None):
    out = np.asarray(value, dtype=np.float64)
    if (out.ndim != 2 or out.shape[1] != 3 or not len(out)
            or (rows is not None and len(out) != rows) or not np.isfinite(out).all()):
        raise ValueError("Expected finite (mothers,3) initial physical summaries.")
    return out


def _query12(value):
    out = np.asarray(value, dtype=np.float64)
    if out.shape == (3, 4):
        out = out.reshape(12)
    if out.shape != (12,) or not np.isfinite(out).all():
        raise ValueError("Only the existing finite12noisy early readings are guide query inputs.")
    return out


def initial_summary(packet):
    """Read the actual t=0 initiator count and two MBAA aperture signals.

    The frozen readout is evaluated on an independent read-only full copy.
    This function does not initialize, evolve or reconstruct a physical State.
    It deliberately does not infer/overwrite the unrecorded Beta profile.
    """
    from aam_formation_actual_rollout import _packet_copy, packet_fingerprint
    from aam_snapshot_process import observe
    if not isinstance(packet, (tuple, list)) or len(packet) != 5:
        raise ValueError("Use an actual five-part native packet.")
    state = packet[0]
    if (len(state) != 16 or state.counts.shape != (7,) or state.position.shape != (4,)
            or not np.array_equal(state.position, np.zeros(4))
            or not np.array_equal(state.counts[[0, 1, 2, 4, 5, 6]], np.zeros(6, dtype=np.int64))
            or not 6 <= int(state.counts[3]) <= 12
            or state.counts[3] != int(state.counts[3])):
        raise ValueError("Summary requires the original unreacted age-zero initial State.")
    if (np.asarray(packet[1]).shape != (4,)
            or not np.array_equal(np.asarray(packet[1])[:3], [64, 14, 8])):
        raise ValueError("This candidate uses the unchanged64AAm/14MBAA/8initial recipe.")
    before = packet_fingerprint(packet)
    supplied = _packet_copy(packet, readonly=True)
    supplied_before = packet_fingerprint(supplied)
    readings = np.asarray(observe(supplied), dtype=np.float64)
    if (readings.shape != (4,) or not np.isfinite(readings).all()
            or not np.array_equal(readings[[0, 2]], [0., 0.])):
        raise ValueError("Age-zero AAm conversion must be zero and tracer signals finite.")
    if (packet_fingerprint(packet) != before or packet_fingerprint(supplied) != supplied_before
            or any(a.flags.writeable for a in (*supplied[0], *supplied[1:]))):
        raise AssertionError("Initial summary changed a physical packet.")
    return np.array([float(state.counts[3]), readings[1], readings[3]], dtype=np.float64)


def fit_guide(training_summaries, training_observations, *, training_mother_ids, noise=CAMERA_NOISE):
    """Define one ridge1 fit using ONLY explicit64training arrays.

    Calling this function is a real fit and must be registered by its caller.
    This module never opens training/held folders and never invokes the fit on
    import.  Initial summaries are privileged *training* simulation annotation,
    not extra observations available at a held query.
    """
    z = _summaries(training_summaries, rows=TRAINING_MOTHERS)
    y = np.asarray(training_observations, dtype=np.float64)
    if y.shape == (TRAINING_MOTHERS, 3, 4):
        y = y.reshape(TRAINING_MOTHERS, 12)
    if y.shape != (TRAINING_MOTHERS, 12) or not np.isfinite(y).all():
        raise ValueError("Guide targets must be the64training mothers' original12noisy readings.")
    ids = np.asarray(training_mother_ids)
    if (ids.shape != (TRAINING_MOTHERS,) or ids.dtype.kind not in "iu"
            or np.any(ids < 0) or len(np.unique(ids)) != TRAINING_MOTHERS):
        raise ValueError("Explicit64distinct nonnegative integer training mother IDs are required.")
    noise = np.asarray(noise, dtype=np.float64)
    if noise.shape != (4,) or not np.array_equal(noise, CAMERA_NOISE):
        raise ValueError("The initial-guide candidate preserves the original four noise levels.")
    mean = z.mean(axis=0)
    scale = z.std(axis=0, ddof=0)
    scale = np.where(scale == 0., 1., scale)
    design = np.column_stack((np.ones(TRAINING_MOTHERS), (z-mean)/scale))
    penalty = np.diag([0., RIDGE_PENALTY, RIDGE_PENALTY, RIDGE_PENALTY])
    coefficients = np.linalg.solve(design.T@design+penalty, design.T@y)
    residual = y-design@coefficients
    noise12 = np.tile(noise, 3)
    variance = np.mean(np.square(residual), axis=0)+np.square(noise12)
    if (not np.isfinite(coefficients).all() or not np.isfinite(variance).all()
            or np.any(variance <= 0.)):
        raise ArithmeticError("Guide fit produced nonfinite coefficients/variance; no hidden fallback.")
    return {"schema": GUIDE_SCHEMA, "summary_fields": SUMMARY_FIELDS,
            "summary_mean": mean.copy(), "summary_scale": scale.copy(),
            "coefficients": coefficients.copy(), "variance": variance.copy(),
            "noise12": noise12.copy(), "ridge_penalty": RIDGE_PENALTY,
            "intercept_penalized": False, "training_mothers": TRAINING_MOTHERS,
            "training_mother_ids": ids.astype(np.int64, copy=True),
            "training_summary_sha256": _array_sha(z), "training_observation_sha256": _array_sha(y),
            "training_residual_mean_square": np.mean(np.square(residual), axis=0),
            "scope": "One fixed training-only proposal guide; not the physical likelihood or forwardpsi.",
            "variance_rule": "mean_training_residual_squared+registered_observation_noise_squared;not a noise decomposition."}


def guide_log_scores(guide, summaries, measured):
    """Unnormalised positive-guide log scores, with no physical readout or RNG."""
    if guide.get("schema") != GUIDE_SCHEMA:
        raise ValueError("Wrong guide schema.")
    z = _summaries(summaries)
    y = _query12(measured)
    mean, scale = (np.asarray(guide[k], dtype=np.float64) for k in ("summary_mean", "summary_scale"))
    coefficients = np.asarray(guide["coefficients"], dtype=np.float64)
    variance = np.asarray(guide["variance"], dtype=np.float64)
    if (mean.shape != (3,) or scale.shape != (3,) or coefficients.shape != (4, 12)
            or variance.shape != (12,) or any(not a.size or not np.isfinite(a).all()
                                             for a in (mean, scale, coefficients, variance))
            or np.any(scale <= 0.) or np.any(variance <= 0.)):
        raise ValueError("Malformed ridge guide arrays.")
    design = np.column_stack((np.ones(len(z)), (z-mean)/scale))
    predicted = design@coefficients
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        score = -.5*np.sum(np.square(y-predicted)/variance, axis=1)
    if not np.isfinite(score).all():
        raise ArithmeticError("Nonfinite guide score; no altered query or hidden rescue.")
    return score


def bank_probabilities(log_scores=None, *, bank_size=BANK_SIZE):
    """Uniform control, or the fixed .5-uniform/.5-guided proposal on indices.

    No actual likelihood is multiplied here.  Guide scores can use all three
    observed cameras for offline recovery; initial1/q cancels their use in the
    proposal before the true per-camera likelihood factors are applied once.
    """
    if bank_size != BANK_SIZE:
        raise ValueError("This candidate fixes the prior initial bank at256entries.")
    if log_scores is None:
        return np.full(bank_size, 1./bank_size)
    scores = np.asarray(log_scores, dtype=np.float64)
    if scores.shape != (bank_size,) or not np.isfinite(scores).all():
        raise ValueError("Expected256finite log-guide scores.")
    guided = np.exp(scores-logsumexp(scores))
    probabilities = UNIFORM_MIXTURE/bank_size+(1.-UNIFORM_MIXTURE)*guided
    probabilities /= probabilities.sum()
    if (np.any(probabilities <= 0.) or not np.isfinite(probabilities).all()
            or probabilities.min() < UNIFORM_MIXTURE/bank_size-2.e-15):
        raise ArithmeticError("Initial proposal lost its declared full-bank support.")
    return probabilities


def draw_initial_indices(probabilities, rng, *, count=PARTICLES):
    """Draw with replacement using an explicit caller-owned controller RNG.

    The returned uniforms make inverse-CDF selection independently auditable.
    Different arms may reuse the same controller uniforms as a computational
    coupling.  Native reaction/Brownian streams must use separately registered
    addresses; this function derives no physical seed from a sampled index.
    """
    q = np.asarray(probabilities, dtype=np.float64)
    if (q.shape != (BANK_SIZE,) or np.any(q <= 0.) or not np.isfinite(q).all()
            or abs(float(q.sum())-1.) > 2.e-14 or count != PARTICLES):
        raise ValueError("Use one full-support256-entry proposal and exactly64draws.")
    q = q.copy()/q.sum()
    uniforms = np.asarray(rng.random(count), dtype=np.float64)
    if uniforms.shape != (count,) or not np.isfinite(uniforms).all() or np.any((uniforms < 0.) | (uniforms >= 1.)):
        raise ValueError("Caller controller must return64uniforms in[0,1).")
    cumulative = np.cumsum(q)
    cumulative[-1] = 1.
    chosen = np.searchsorted(cumulative, uniforms, side="right").astype(np.int64)
    log_ratio = -np.log(float(BANK_SIZE))-np.log(q[chosen])
    ratio = np.exp(log_ratio)
    normalizer = float(logsumexp(log_ratio))
    normalized = np.exp(log_ratio-normalizer)
    return {"schema": SCHEMA, "bank_size": BANK_SIZE, "count": count,
            "proposal_probabilities": q.copy(), "controller_uniforms": uniforms.copy(),
            "bank_indices": chosen, "initial_log_importance_ratios": log_ratio,
            "initial_importance_ratios": ratio, "initial_normalized_weights": normalized,
            "log_mean_initial_importance_ratio": normalizer-np.log(float(count)),
            "distinct_bank_indices": int(np.unique(chosen).size),
            "scope": "Initial finite-bank index importance only; true camera likelihood has not been applied."}


def make_prior_bank(config, initialization_seeds, factors):
    """Create256complete native initial packets, but never evolve a reaction.

    This is a real setup/readout operation when called.  The caller must first
    register all256uint32 addresses and keep them separate from native-camera,
    selection-controller and mother-truth addresses.  This function does not
    derive addresses, deduplicate seeds/packets, or open any query/eval archive.
    The returned plain tree can be archived by the existing numeric_tree writer.
    """
    from aam_formation_actual_rollout import _packet_copy, packet_fingerprint
    from aam_formation_cycle_core import _physical_config
    from aam_snapshot_process import initialize
    seeds = np.asarray(initialization_seeds)
    if (seeds.shape != (BANK_SIZE,) or seeds.dtype.kind not in "iu"
            or np.any(seeds < 0) or np.any(seeds.astype(np.uint64) >= 2**32)):
        raise ValueError("Supply exactly256registered uint32-range initial seeds; no implicit seed schedule.")
    seeds = seeds.astype(np.uint32, copy=True)
    actual_config = _physical_config(config, factors)
    started = perf_counter()
    packets, summaries, fingerprints = [], [], []
    initialization_seconds, summary_seconds = 0., 0.
    for seed in seeds:
        segment = perf_counter()
        packet = initialize(actual_config, int(seed))
        initialization_seconds += perf_counter()-segment
        segment = perf_counter()
        summaries.append(initial_summary(packet))
        summary_seconds += perf_counter()-segment
        frozen = _packet_copy(packet, readonly=True)
        fingerprints.append(packet_fingerprint(frozen))
        packets.append(frozen)
    hashes = np.asarray(fingerprints, dtype="U64")
    return {"schema": SCHEMA, "bank_size": BANK_SIZE,
            "initialization_seeds": seeds, "packets": packets,
            "summaries": np.asarray(summaries), "summary_fields": SUMMARY_FIELDS,
            "packet_sha256": hashes, "forward_factors": np.asarray(factors, dtype=np.float64).copy(),
            "distinct_initialization_seeds": int(np.unique(seeds).size),
            "distinct_packet_fingerprints": int(np.unique(hashes).size),
            "deduplication_performed": False,
            "cost": {"setup_calls": BANK_SIZE, "native_interval_calls": 0,
                     "initialization_seconds": initialization_seconds, "summary_readout_seconds": summary_seconds,
                     "elapsed_seconds": perf_counter()-started},
            "scope": "Random finite prior bank of actual age-zero packets;not256evolvedprefixes or newmeasurements."}


def initial_population(bank, draw):
    """Copy complete sampled initial packets into the V1 particle-item schema.

    Each duplicate gets its own mutable State, physical parameter arrays and
    frames.  The caller's V2 filter must keep the returned initial ratios,
    source-bank identity and full ancestry, apply each actual camera likelihood
    once, and retain all final weights.  The bank itself remains unchanged.
    """
    from aam_formation_actual_rollout import _packet_copy, packet_fingerprint
    from aam_snapshot_process import clone
    if (bank.get("schema") != SCHEMA or draw.get("schema") != SCHEMA
            or bank.get("bank_size") != BANK_SIZE or draw.get("bank_size") != BANK_SIZE
            or len(bank["packets"]) != BANK_SIZE):
        raise ValueError("Use the matching256-entry actual initial bank and sampled index record.")
    seeds, hashes = np.asarray(bank["initialization_seeds"]), np.asarray(bank["packet_sha256"])
    indices = np.asarray(draw["bank_indices"])
    q = np.asarray(draw["proposal_probabilities"], dtype=np.float64)
    log_ratio = np.asarray(draw["initial_log_importance_ratios"], dtype=np.float64)
    if (seeds.shape != (BANK_SIZE,) or hashes.shape != (BANK_SIZE,) or indices.shape != (PARTICLES,)
            or indices.dtype.kind not in "iu" or np.any(indices < 0) or np.any(indices >= BANK_SIZE)
            or q.shape != (BANK_SIZE,) or np.any(q <= 0.) or not np.isfinite(q).all()
            or abs(float(q.sum())-1.) > 2.e-14 or log_ratio.shape != (PARTICLES,)
            or not np.allclose(log_ratio, -np.log(float(BANK_SIZE))-np.log(q[indices]), atol=2.e-14, rtol=0.)):
        raise ValueError("Malformed selection or initial finite-bank importance correction.")
    population = []
    for index in indices:
        source = bank["packets"][int(index)]
        before = packet_fingerprint(source)
        if before != str(hashes[int(index)]):
            raise ValueError("Original bank packet changed since initialization.")
        packet = _packet_copy(source)
        population.append({"packet": packet, "initial_packet": _packet_copy(source),
                           "frames": {0.: clone(source[0])}, "initial_seed": int(seeds[int(index)]),
                           "interval_seeds": np.full(3, -1, dtype=np.int64), "camera": -1,
                           "source_bank_index": int(index), "source_packet_sha256": before})
        if packet_fingerprint(source) != before:
            raise AssertionError("Selecting a particle changed its original bank State.")
    normalized = np.exp(log_ratio-logsumexp(log_ratio))
    return {"schema": SCHEMA, "population": population,
            "initialization_seeds": seeds[indices].astype(np.uint32, copy=True),
            "initial_bank_indices": indices.astype(np.int64, copy=True),
            "initial_packet_sha256": hashes[indices].copy(),
            "initial_log_importance_ratios": log_ratio.copy(),
            "initial_weights": normalized,
            "log_mean_initial_importance_ratio": float(logsumexp(log_ratio)-np.log(float(PARTICLES))),
            "scope": "Starting particles only;no reaction,posterior likelihood or future has been simulated."}
