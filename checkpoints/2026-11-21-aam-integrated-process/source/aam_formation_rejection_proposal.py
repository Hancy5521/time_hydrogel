"""Full-prior rejection initialisation for the formation-process cycle.

No fit, initialisation, RNG draw, readout or file access occurs on import.
The only guide learned in a prospective run is the existing three-summary
ridge1 guide.  This module prepares its fixed relative-SVD projection and uses
it to propose COMPLETE native initial packets.  It never overwrites physical
parameters or limits the physical prior to a fixed finite bank.

For accepted s0, q(ds0|y) = mu(ds0)*a(s0,y)/C_y.  Returned initial log ratios
are -log(a), up to the UNKNOWN common constant log(C_y).  They are sufficient
for normalized posterior weights, NOT absolute evidence estimation.
"""
from __future__ import annotations

from copy import deepcopy
from time import perf_counter

import numpy as np

from aam_formation_initial_proposal import initial_summary, fit_guide, GUIDE_SCHEMA


SCHEMA = "aam-formation-full-prior-rejection-initial-v1"
PREPARED_SCHEMA = "aam-initial-ridge-projected-rejection-guide-v1"
PARTICLES = 64
MAX_CANDIDATES = 8192
ACCEPTANCE_FLOOR = .05
RELATIVE_SVD_CUTOFF = 1.e-12


def _query12(value):
    value = np.asarray(value, dtype=np.float64)
    if value.shape == (3, 4):
        value = value.reshape(12)
    if value.shape != (12,) or not np.isfinite(value).all():
        raise ValueError("Use the original12finite noisy readings without clipping.")
    return value


def _guide_arrays(guide):
    mean = np.asarray(guide["summary_mean"], dtype=np.float64)
    scale = np.asarray(guide["summary_scale"], dtype=np.float64)
    coefficients = np.asarray(guide["coefficients"], dtype=np.float64)
    variance = np.asarray(guide["variance"], dtype=np.float64)
    if (mean.shape != (3,) or scale.shape != (3,) or coefficients.shape != (4, 12)
            or variance.shape != (12,) or np.any(scale <= 0.) or np.any(variance <= 0.)
            or any(not np.isfinite(a).all() for a in (mean, scale, coefficients, variance))):
        raise ValueError("Malformed three-summary/12observation ridge guide.")
    return mean, scale, coefficients, variance


def prepare_rejection_guide(guide):
    """One fixed algebraic SVD of the already-fitted guide, not another fit.

    Standardized z predicts c+Bz.  C=diag(v)^(-1/2)B uses the SAME registered
    residual-plus-noise variance as the small training guide.  Only relative
    singular values >1e-12*s_max are retained.  Rank0 has h=1 by definition.
    The complete prepared dictionary can be frozen by the caller's numeric
    tree writer before any new held mother or query prediction is generated.
    """
    if guide.get("schema") != GUIDE_SCHEMA:
        raise ValueError("Prepare only the fixed training-only ridge guide.")
    _, _, coefficients, variance = _guide_arrays(guide)
    started = perf_counter()
    weighted_B = coefficients[1:].T/np.sqrt(variance)[:, None]
    left, singular, right_T = np.linalg.svd(weighted_B, full_matrices=False)
    keep = singular > RELATIVE_SVD_CUTOFF*singular[0] if singular[0] > 0. else np.zeros_like(singular, dtype=bool)
    prepared = deepcopy(guide)
    prepared.update(schema=PREPARED_SCHEMA, source_ridge_schema=guide["schema"],
        weighted_B=weighted_B, left_basis=left[:, keep].copy(),
        singular_values=singular[keep].copy(), right_basis_T=right_T[keep].copy(),
        all_singular_values=singular.copy(), retained_singular_mask=keep.copy(),
        svd_rank=int(keep.sum()), relative_svd_cutoff=RELATIVE_SVD_CUTOFF,
        acceptance_floor=ACCEPTANCE_FLOOR,
        preparation_seconds=perf_counter()-started,
        scope="Fixed envelope-normalized proposal guide;not a second likelihood or fitted physical law.")
    return prepared


def acceptance_probs(guide, summaries, measured):
    """Return .05+.95*exp(-projected squared error/2), without altering inputs.

    The orthogonal component of b is deliberately removed using the frozen
    truncated SVD; this is not the original12D likelihood or a query-dependent
    choice of rank.  Neither measured values nor physical tracer summaries are
    clipped.  The positive .05 floor preserves the full native prior support.
    """
    if guide.get("schema") != PREPARED_SCHEMA:
        raise ValueError("Use the frozen prepared rejection guide.")
    mean, scale, coefficients, variance = _guide_arrays(guide)
    z = np.asarray(summaries, dtype=np.float64)
    if z.ndim != 2 or z.shape[1] != 3 or not len(z) or not np.isfinite(z).all():
        raise ValueError("Use explicit finite initial-state summaries(n,3).")
    y = _query12(measured)
    rank = int(guide["svd_rank"])
    U, S, VT = (np.asarray(guide[k], dtype=np.float64)
                for k in ("left_basis", "singular_values", "right_basis_T"))
    if (rank < 0 or rank > 3 or U.shape != (12, rank) or S.shape != (rank,)
            or VT.shape != (rank, 3) or np.any(S <= 0.)
            or any(not np.isfinite(a).all() for a in (U, S, VT))
            or guide["relative_svd_cutoff"] != RELATIVE_SVD_CUTOFF
            or guide["acceptance_floor"] != ACCEPTANCE_FLOOR):
        raise ValueError("Prepared fixed-rank SVD contract changed.")
    if rank == 0:
        return np.ones(len(z), dtype=np.float64)
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        standardized = (z-mean)/scale
        b = (y-coefficients[0])/np.sqrt(variance)
        difference = (U.T@b)[None, :]-(standardized@VT.T)*S
        square = np.sum(np.square(difference), axis=1)
        h = np.exp(-.5*square)
        probabilities = ACCEPTANCE_FLOOR+(1.-ACCEPTANCE_FLOOR)*h
    if (not np.isfinite(probabilities).all() or np.any(probabilities < ACCEPTANCE_FLOOR)
            or np.any(probabilities > 1.)):
        raise ArithmeticError("Invalid rejection probability;no hidden rescue or probability clipping.")
    return probabilities


def sample_initial(config, factors, measured, guide, seed_rng, accept_rng, *,
                   guided=True, count=PARTICLES, max_candidates=MAX_CANDIDATES):
    """Sample64actual initial packets from the full registered seed prior.

    Candidate seed law nu0 is iid Uniform{0,...,2**32-1}, WITH replacement;
    mu=initialize#nu0.  Separate caller-owned named streams provide seeds and
    acceptance uniforms.  Seed/packet repetitions remain; no deduplication,
    arithmetic offsets or unregistered physical randomness is introduced.

    Guided acceptance uses a(z,y); its accepted packets have q=mu*a/C_y.
    Control guided=False accepts its first64iid prior candidates with a=1,
    makes no acceptance-uniform draws and never evaluates the guide.

    All candidate seeds, summaries, a/u/decisions and fingerprints are returned,
    including rejected candidates.  Accepted packets preserve every actual
    State and physical-parameter field.  The caller must save this tree before
    propagating; an8192-attempt failure is retained, never replaced or rescued.
    This function initializes/readouts only: it never calls progress/evolve.
    """
    from aam_formation_actual_rollout import _packet_copy, packet_fingerprint
    from aam_formation_cycle_core import _physical_config
    from aam_snapshot_process import initialize
    if count != PARTICLES or max_candidates != MAX_CANDIDATES:
        raise ValueError("This prospective version fixes64accepted particles and8192maximum attempts.")
    if type(guided) is not bool:
        raise TypeError("Explicit guided=True/False required.")
    measured = _query12(measured)
    if guided and guide.get("schema") != PREPARED_SCHEMA:
        raise ValueError("Guided initialization needs the frozen SVD-prepared training guide.")
    actual_config = _physical_config(config, factors)
    started = perf_counter()
    packets, accepted_seeds, accepted_indices, log_ratios, accepted_hashes = [], [], [], [], []
    candidate_seeds, candidate_z, candidate_a, candidate_u = [], [], [], []
    decisions, valid_rows, row_hashes, row_errors = [], [], [], []
    setup_calls = summary_calls = guide_calls = seed_draws = accept_draws = 0
    setup_seconds = summary_seconds = guide_seconds = copy_seconds = 0.
    error = None
    for attempt in range(max_candidates):
        stage, proposed_seed, z, a, u, fingerprint = "seed", -1, np.full(3, np.nan), np.nan, np.nan, ""
        accepted = False
        row_error = None
        try:
            # high is exclusive.  This covers the complete uint32 seed space.
            raw_seed = seed_rng.integers(0, 2**32, dtype=np.uint32)
            seed_draws += 1
            if (np.ndim(raw_seed) != 0 or isinstance(raw_seed, (bool, np.bool_))
                    or int(raw_seed) != raw_seed or not 0 <= int(raw_seed) < 2**32):
                raise ValueError("Initial stream must supply one uniform uint32-compatible seed per candidate.")
            proposed_seed = int(raw_seed)
            stage, segment = "native_initialize", perf_counter()
            setup_calls += 1
            try:
                packet = initialize(actual_config, proposed_seed)
            finally:
                setup_seconds += perf_counter()-segment
            stage, segment = "initial_summary", perf_counter()
            summary_calls += 1
            try:
                z = initial_summary(packet)
            finally:
                summary_seconds += perf_counter()-segment
            fingerprint = packet_fingerprint(packet)
            if guided:
                stage, segment = "guide", perf_counter()
                guide_calls += 1
                try:
                    a = float(acceptance_probs(guide, z[None, :], measured)[0])
                finally:
                    guide_seconds += perf_counter()-segment
                stage = "acceptance_uniform"
                raw_u = accept_rng.random()
                accept_draws += 1
                if np.ndim(raw_u) != 0 or not np.isfinite(raw_u) or not 0. <= float(raw_u) < 1.:
                    raise ValueError("Acceptance stream must supply one uniform in[0,1).")
                u = float(raw_u)
                accepted = bool(u < a)
            else:
                a, accepted = 1., True
            if accepted:
                stage, segment = "copy_actual_packet", perf_counter()
                frozen = _packet_copy(packet, readonly=True)
                if packet_fingerprint(frozen) != fingerprint:
                    raise AssertionError("Accepted initial State changed on complete-packet copy.")
                packets.append(frozen)
                accepted_seeds.append(proposed_seed)
                accepted_indices.append(attempt)
                accepted_hashes.append(fingerprint)
                log_ratios.append(-float(np.log(a)))
                copy_seconds += perf_counter()-segment
        except Exception as exc:
            row_error = {"type": type(exc).__name__, "message": str(exc), "stage": stage}
            error = {**row_error, "candidate": attempt, "accepted_count": len(packets)}
        candidate_seeds.append(proposed_seed); candidate_z.append(z)
        candidate_a.append(a); candidate_u.append(u); decisions.append(accepted)
        valid_rows.append(row_error is None); row_hashes.append(fingerprint); row_errors.append(row_error)
        if error is not None or len(packets) == count:
            break
    valid = error is None and len(packets) == count
    if not valid and error is None:
        error = {"type": "RejectionBudgetExceeded", "stage": "candidate_limit",
                 "message": "8192attempts did not yield64accepted packets;preserve failed query,no replacement.",
                 "accepted_count": len(packets), "attempts": len(candidate_seeds)}
    seeds = np.asarray(candidate_seeds, dtype=np.int64)
    selected_seeds = np.asarray(accepted_seeds, dtype=np.uint32)
    return {"schema": SCHEMA, "status": "complete" if valid else "failed", "valid": valid,
            "guided": guided, "packets": packets, "initialization_seeds": selected_seeds,
            "initial_log_ratios": np.asarray(log_ratios, dtype=np.float64),
            "initial_weight_semantics": "1/a up to unknown C_y;posterior-normalization only,not absolute evidence.",
            "unknown_proposal_normalizer_C_y": bool(guided),
            "accepted_attempt_indices": np.asarray(accepted_indices, dtype=np.int64),
            "accepted_packet_sha256": np.asarray(accepted_hashes, dtype="U64"),
            "ledger": {"candidate_index": np.arange(len(candidate_seeds), dtype=np.int64),
                       "seed": seeds, "summaries": np.asarray(candidate_z, dtype=np.float64),
                       "acceptance_probability": np.asarray(candidate_a, dtype=np.float64),
                       "acceptance_uniform": np.asarray(candidate_u, dtype=np.float64),
                       "accepted": np.asarray(decisions, dtype=bool),
                       "valid": np.asarray(valid_rows, dtype=bool),
                       "packet_sha256": np.asarray(row_hashes, dtype="U64"), "errors": row_errors},
            "prior": "initialize pushforward of iid uniform uint32 seeds;with replacement,no deduplication",
            "distinct_candidate_seeds": int(np.unique(seeds[seeds >= 0]).size),
            "distinct_accepted_seeds": int(np.unique(selected_seeds).size),
            "distinct_accepted_packets": int(np.unique(accepted_hashes).size),
            "cost": {"setup_calls": setup_calls, "candidate_attempts": len(candidate_seeds),
                     "accepted": len(packets), "rejections": int(np.count_nonzero(np.asarray(valid_rows) & ~np.asarray(decisions))),
                     "summary_readout_calls": summary_calls, "guide_calls": guide_calls,
                     "seed_draws": seed_draws, "acceptance_uniform_draws": accept_draws,
                     "native_interval_calls": 0, "setup_seconds": setup_seconds,
                     "summary_seconds": summary_seconds, "guide_seconds": guide_seconds,
                     "accepted_copy_seconds": copy_seconds, "elapsed_seconds": perf_counter()-started},
            "error": error,
            "scope": "Complete-prior initial proposal only;no prefix or future simulated;no absolute evidence."}


def rejection_initialization(config, factors, guide, measured, *, initial_rng, acceptance_rng,
                             guided=True, count=PARTICLES, max_candidates=MAX_CANDIDATES):
    """Keyword-oriented adapter for the whole-cycle driver; same single sampler."""
    return sample_initial(config, factors, measured, guide, initial_rng, acceptance_rng,
                          guided=guided, count=count, max_candidates=max_candidates)
