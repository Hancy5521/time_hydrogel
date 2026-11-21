"""Complete-State filtering from an explicit, importance-weighted initial pool.

This is an unexecuted candidate extension of the frozen formation-cycle V1.
Only the initialization interface changes.  The original forward chemistry,
three camera likelihoods, resampling rule and complete-history inheritance are
reused.  Import performs no simulation, fitting, RNG draw or output.

Initial packets must be actual t=0 packets produced by the caller's registered
native initialization procedure.  This function checks their structure and
physical configuration without regenerating or altering them.  A seed is
provenance, not a substitute for a saved physical state or its prior law.
The caller may supply independently drawn native seed tables and its own
resampling RNG.  This is required for the V2 full-prior rejection proposal.
"""
from __future__ import annotations

from time import perf_counter

import numpy as np
from scipy.special import logsumexp

from aam_edge_fork import State
from aam_formation_actual_rollout import _packet_copy, packet_fingerprint
from aam_formation_cycle_core import (
    SCHEMA as CORE_SCHEMA, PRODUCTION_PARTICLES, PREFIX_CLOCK, RESAMPLE_FRACTION,
    _advance, _clone_particle, _cost, _physical_config, _systematic, _uint32, _update,
)
from aam_snapshot_process import clone, likelihood, observe
from aam_spatial_motion import Motion


SCHEMA = "aam-formation-explicit-initial-population-filter-v1"


def filter_population(config, measured, seed, factors, *, initial_packets,
                      initial_log_ratios, initialization_seeds, mother_id=None,
                      initial_source_ids=None, native_seeds=None,
                      resampling_rng=None, unknown_proposal_normalizer=False):
    """Filter 64 complete supplied t=0 packets with their actual IS ratios.

    ``initial_log_ratios[i]`` is the caller's *unnormalized* log target/proposal
    ratio for initial draw i, up to a common proposal constant if explicitly
    declared unknown.  For complete-prior rejection with acceptance a, use
    ``-log(a)`` and ``unknown_proposal_normalizer=True``.  Its normalized
    exponential initializes the particle weights.  The initial log mean ratio
    is retained for bookkeeping; normalizing initial weights must not silently
    discard it or turn an unknown proposal constant into absolute evidence.

    All query readings may inform the initial proposal only when the caller
    includes its correct ratio.  This function subsequently multiplies each
    original camera likelihood once; the guide is not another likelihood.
    For complete-prior rejection, the unknown acceptance normalizer C_y drops
    out of normalized posterior weights.  For a finite-bank caller, the target
    instead remains that bank's empirical prior.  Neither finite-N normalized
    weights nor a log normalizer are claimed to be unbiased posteriors.

    V2 supplies ``native_seeds`` as an actual (3,64) uint32 table drawn with
    replacement from a separately named stream, and ``resampling_rng`` from
    another named stream.  Numeric seed repeats are retained.  Their named
    stream provenance belongs in the caller's registration and receipts.
    With both omitted, legacy V1 arithmetic addresses are available explicitly
    as a compatibility mode, not as a claim of independent incremental seeds.
    No initialization, proposal-selection or native-seed draw occurs here.

    ``initial_source_ids`` optionally maps draw slots to bank members.  The
    legacy ``ancestral_diversity`` measures draw-slot ancestry; repeated bank
    indices/seeds are not independent initial states.  Use ``source_ancestor``
    and final weights to aggregate actual source-level diversity separately.

    Invalid inputs raise before native work.  A later kernel/readout failure
    returns its partial complete-State history and costs; there is no replay,
    replacement query, repaired history or alternate forward law.
    """
    started = perf_counter()
    particles = PRODUCTION_PARTICLES
    if type(unknown_proposal_normalizer) is not bool:
        raise TypeError("Declare unknown_proposal_normalizer explicitly as a boolean.")
    explicit_streams = native_seeds is not None
    if explicit_streams != (resampling_rng is not None):
        raise ValueError("Supply both the explicit native seed table and resampling RNG, or neither.")
    if unknown_proposal_normalizer and not explicit_streams:
        raise ValueError("The V2 rejection proposal requires explicit named native and resampling streams.")
    seed = _uint32(seed, 0 if explicit_streams else 16000)
    if explicit_streams:
        supplied_native_seeds = np.asarray(native_seeds)
        if supplied_native_seeds.shape != (3, particles) or supplied_native_seeds.dtype.kind not in "iu":
            raise ValueError("Supply the actual3x64uint32-compatible native seed table.")
        native_seed_table = np.asarray([_uint32(s) for s in supplied_native_seeds.reshape(-1)],
                                       dtype=np.uint32).reshape(3, particles)
        if not callable(getattr(resampling_rng, "random", None)):
            raise TypeError("The caller-supplied resampling controller must provide random().")
        rng = resampling_rng
        stream_mode = "explicit_native_seed_table_and_caller_resampling_rng;numeric_seed_repeats_retained"
    else:
        native_seed_table = np.asarray([
            seed+5000*(camera+1)+np.arange(particles) for camera in range(3)], dtype=np.uint32)
        rng = np.random.default_rng(seed+16000)
        stream_mode = "legacy_V1_arithmetic_compatibility;not_independent_seed_draws"
    measured = np.asarray(measured, dtype=np.float64)
    noise = np.asarray(config["noise"], dtype=np.float64)
    if (measured.shape != (3, 4) or not np.isfinite(measured).all()
            or noise.shape != (4,) or not np.isfinite(noise).all() or np.any(noise <= 0.)):
        raise ValueError("Use exactly the original3x4noisy camera readings and positive noise.")
    actual_config = _physical_config(config, factors)
    if len(initial_packets) != particles:
        raise ValueError("This candidate accepts exactly64complete initial packets.")
    log_ratios = np.asarray(initial_log_ratios, dtype=np.float64)
    if log_ratios.shape != (particles,) or not np.isfinite(log_ratios).all():
        raise ValueError("Every sampled initial packet requires a finite log target/proposal ratio.")
    supplied_seeds = np.asarray(initialization_seeds)
    if (supplied_seeds.shape != (particles,) or supplied_seeds.dtype.kind not in "iu"
            or any(isinstance(s, (bool, np.bool_)) for s in initialization_seeds)):
        raise ValueError("Provide exactly64uint32-compatible original initialization seed addresses.")
    initialization_seeds = np.asarray([_uint32(s) for s in supplied_seeds], dtype=np.uint32)
    if initial_source_ids is None:
        source_ids = initialization_seeds.astype(np.int64)
        source_id_kind = "initialization_seed; equal seeds grouped as the same source"
    else:
        source_ids = np.asarray(initial_source_ids)
        if (source_ids.shape != (particles,) or source_ids.dtype.kind not in "iu"
                or np.any(source_ids < 0) or np.any(source_ids > np.iinfo(np.int64).max)):
            raise ValueError("Optional source ids must be64nonnegative int64-compatible bank indices.")
        source_ids = source_ids.astype(np.int64, copy=True)
        source_id_kind = "caller-registered initial bank index"

    recipe, motion = actual_config["recipe"], Motion(**actual_config["motion"])
    expected_material = np.array([recipe["aam"], recipe["mbaa"], recipe["initial_mbaa"],
                                  recipe["feed_time"]])
    expected_geometry = np.array([motion.step, motion.diffusion, motion.box, motion.spring,
                                 motion.rest, motion.repulsion, motion.core, motion.capture,
                                 motion.inner, actual_config["corridor"]])
    expected_rates = np.asarray(actual_config["rates"])
    expected_factors = np.asarray(actual_config["factors"])
    input_fingerprints, population = [], []
    n, sites = 78, 156
    expected_present = np.arange(n) < 72
    expected_available = np.zeros(sites, dtype=bool)
    expected_available[:128:2] = True
    expected_available[128:144] = True
    for slot, supplied in enumerate(initial_packets):
        if (not isinstance(supplied, (tuple, list)) or len(supplied) != 5
                or not isinstance(supplied[0], State)):
            raise TypeError("Each initial candidate must be the actual native16-field State packet.")
        state, material, geometry, rates, forward = supplied
        arrays = (*state, material, geometry, rates, forward)
        if any(not isinstance(a, np.ndarray) or a.dtype.hasobject for a in arrays):
            raise TypeError("Native packet fields must be explicit non-object arrays.")
        dtypes = {"points": "float64", "unwrapped": "float64", "present": "bool",
                  "active": "bool", "available": "bool", "owner": "int64", "labels": "int64",
                  "edges": "int64", "lifts": "int64", "events": "float64",
                  "periodic_parent": "int64", "periodic_shift": "int64",
                  "periodic_rank": "int64", "winding": "bool", "counts": "int64",
                  "position": "float64"}
        if (any(getattr(state, key).dtype != np.dtype(dtype) for key, dtype in dtypes.items())
                or any(a.dtype != np.dtype("float64") for a in supplied[1:])):
            raise TypeError("Preserve the existing native State and physical-array dtypes exactly.")
        if (not np.array_equal(material, expected_material)
                or not np.array_equal(geometry, expected_geometry)
                or not np.array_equal(rates, expected_rates)
                or not np.array_equal(forward, expected_factors)):
            raise ValueError("Initial packet has a different material, motion, rate or forwardpsi law.")
        if (state.points.shape != (n, 3) or state.unwrapped.shape != (n, 3)
                or not np.isfinite(state.points).all() or np.any(state.points < 0.)
                or np.any(state.points >= motion.box)
                or not np.array_equal(state.points, state.unwrapped)
                or state.present.dtype.kind != "b" or not np.array_equal(state.present, expected_present)
                or state.active.shape != (sites,) or state.active.dtype.kind != "b" or state.active.any()
                or state.available.dtype.kind != "b" or not np.array_equal(state.available, expected_available)
                or state.owner.shape != (sites,) or state.owner.dtype.kind not in "iu"
                or np.any(state.owner != -1)):
            raise ValueError("Input must be an untouched native t=0 AAm/MBAA packet, not a prefix or reconstructed graph.")
        if (state.counts.shape != (7,) or state.counts.dtype.kind not in "iu"
                or np.any(state.counts[[0, 1, 2, 4, 5, 6]] != 0)
                or not 6 <= int(state.counts[3]) <= 12
                or state.position.shape != (4,) or np.any(state.position != 0.)):
            raise ValueError("Initial reaction ledger or physical clock is not at pristine t=0.")
        capacity = 64+2*14+int(state.counts[3])//2+14+1
        if (state.edges.shape != (capacity, 2) or state.lifts.shape != (capacity, 3)
                or state.events.shape != (capacity, 13)
                or np.any(state.edges != 0) or np.any(state.lifts != 0) or np.any(state.events != 0)
                or not np.array_equal(state.labels, np.arange(n))
                or not np.array_equal(state.periodic_parent, np.arange(n))
                or state.periodic_shift.shape != (n, 3) or np.any(state.periodic_shift != 0)
                or state.periodic_rank.shape != (n,) or np.any(state.periodic_rank != 0)
                or state.winding.shape != (3,) or state.winding.any()):
            raise ValueError("Initial actual graph, native capacities or periodic metadata are not pristine.")
        input_fingerprints.append(packet_fingerprint(supplied))
        packet = _packet_copy(supplied)
        population.append({"packet": packet, "initial_packet": _packet_copy(supplied),
                           "frames": {0.: clone(state)}, "initial_seed": int(initialization_seeds[slot]),
                           "interval_seeds": np.full(3, -1, dtype=np.int64), "camera": -1})

    initial_log_total = float(logsumexp(log_ratios))
    initial_log_mean_ratio = initial_log_total-float(np.log(particles))
    weights = np.exp(log_ratios-initial_log_total)
    weights /= weights.sum()
    initial_weights = weights.copy()
    cost = _cost()
    ancestry = np.full((particles, 3), -1, dtype=np.int64)
    ancestry[:, 0] = np.arange(particles)
    log_values = np.full((particles, 3), np.nan)
    ledger, error = [], None
    camera_log_normalizers = []
    log_evidence = initial_log_mean_ratio
    for camera in range(3):
        incoming = weights.copy()
        raw = np.full((particles, 4), np.nan)
        advanced = np.zeros(particles, dtype=bool)
        seeds = native_seed_table[camera].copy()
        record = {"camera": camera, "time": PREFIX_CLOCK[camera+1],
                  "weights_incoming": incoming, "native_seeds": seeds.copy(),
                  "ancestry_incoming": ancestry.copy(), "advanced": advanced,
                  "clean_observations": raw}
        ledger.append(record)
        try:
            clock = np.asarray(PREFIX_CLOCK[:camera+2])
            for slot, item in enumerate(population):
                item["interval_seeds"][camera] = int(seeds[slot])
                _advance(item["packet"], clock, int(seeds[slot]), False, cost)
                advanced[slot] = True
                item["camera"] = camera
                item["frames"][float(clock[-1])] = clone(item["packet"][0])
                raw[slot] = observe(_packet_copy(item["packet"]))
            ll = likelihood(raw, measured[camera], noise)
            log_values[:, camera] = ll
            updated = _update(incoming, ll)
            record.update(log_likelihood=ll.copy(), weights_updated=updated["weights"].copy(),
                          measured=measured[camera].copy(), noise=noise.copy(),
                          diagnostics={key: value for key, value in updated.items() if key != "weights"})
            if not updated["valid"]:
                weights = updated["weights"]
                raise ArithmeticError("All-zero weighted camera support; no floor, rescue or replacement query.")
            weights = updated["weights"]
            camera_log_normalizers.append(updated["log_normalizer"])
            log_evidence += updated["log_normalizer"]
            if camera < 2:
                ancestry[:, camera+1] = np.arange(particles)
            resampled = bool(camera < 2 and updated["ESS"] < RESAMPLE_FRACTION*particles)
            u = float(rng.random()) if resampled else None
            if u is not None and (not np.isfinite(u) or not 0. <= u < 1.):
                raise ValueError("Explicit resampling controller must produce a finite uniform in[0,1).")
            parents = _systematic(weights, u) if resampled else np.arange(particles)
            if resampled:
                population = [_clone_particle(population[int(parent)]) for parent in parents]
                ancestry = ancestry[parents].copy()
                log_values = log_values[parents].copy()
                weights = np.full(particles, 1./particles)
            record.update(resampled=resampled, systematic_uniform=u, parents=parents.copy(),
                          weights_after=weights.copy(), ancestry_after=ancestry.copy(),
                          consumed_seeds_after=np.stack([item["interval_seeds"] for item in population]))
        except Exception as exc:
            error = {"camera": camera, "type": type(exc).__name__, "message": str(exc)}
            break
    complete = error is None
    if complete and (cost["native_segment_attempted"] != 3*particles
                     or cost["native_segment_completed"] != 3*particles):
        error = {"type": "RuntimeError", "message": "Complete filtering violated exactly3Nnativeintervals."}
        complete = False
    after_fingerprints = [packet_fingerprint(packet) for packet in initial_packets]
    if input_fingerprints != after_fingerprints:
        raise RuntimeError("Caller initial packets changed; do not conceal mutation or regenerate history.")
    exported = []
    for item in population:
        frames = {time: clone(state) for time, state in item["frames"].items()}
        for state in frames.values():
            for array in state:
                array.flags.writeable = False
        exported.append({"packet": _packet_copy(item["packet"], readonly=True),
                         "initial_packet": _packet_copy(item["initial_packet"], readonly=True),
                         "frames": frames, "initial_seed": item["initial_seed"],
                         "interval_seeds": item["interval_seeds"].copy(), "camera": item["camera"]})
    diversity = {}
    for column, age in enumerate((0., .65, 1.35)):
        valid = ancestry[:, column] >= 0
        mass = np.bincount(ancestry[valid, column], weights=weights[valid], minlength=particles)
        diversity[str(age)] = {"distinct_particle_ancestors": int(len(np.unique(ancestry[valid, column]))),
                               "weighted_ancestral_ESS": float(1./(mass@mass)) if mass.sum() > 0. else None,
                               "unit": "initial draw slot; not unique bank member or independent mother"}
    source_ancestor = source_ids[ancestry[:, 0]].copy()
    result = {"schema": SCHEMA, "compatible_core_schema": CORE_SCHEMA,
            "status": "complete" if complete else "failed", "valid": complete,
            "mother_id": mother_id, "seed": seed, "particles": exported,
            "packets": [item["packet"] for item in exported], "cached_results": exported,
            "weights": weights.copy(),
            "log_weights": np.log(weights, where=weights > 0., out=np.full_like(weights, -np.inf)),
            "log_likelihoods": log_values, "initial_log_ratios": log_ratios.copy(),
            "initial_weights": initial_weights, "initial_log_mean_ratio": initial_log_mean_ratio,
            "camera_log_normalizers": np.asarray(camera_log_normalizers),
            "native_seed_table": native_seed_table.copy(), "random_stream_mode": stream_mode,
            "unknown_proposal_normalizer": unknown_proposal_normalizer,
            "initialization_seeds": initialization_seeds.copy(), "initial_source_ids": source_ids,
            "initial_source_id_kind": source_id_kind, "source_ancestor": source_ancestor,
            "input_packet_sha256": input_fingerprints,
            "caller_initial_packets_unchanged": input_fingerprints == after_fingerprints,
            "ancestry_ids": ancestry, "initial_ancestor": ancestry[:, 0].copy(),
            "ancestral_diversity": diversity, "ledger": ledger, "cost": cost, "error": error,
            "elapsed_seconds": perf_counter()-started,
            "scope": "Explicit64actualinitialpackets with caller-declared importance ratios; unchanged learnedforwardpsi and camera updates. All final native weights and actual histories retained."}
    if unknown_proposal_normalizer:
        result.update(
            relative_log_normalizer_up_to_unknown_proposal_constant=log_evidence,
            normalizer_scope=("Relative bookkeeping only:logmean(exp(initial_log_ratios))+camera"
                              "lognormalizers;unknowncommonC_y omitted alongwithGaussianconstants;"
                              "not absoluteevidence or an unbiased loglikelihood."))
    else:
        result.update(
            log_evidence_up_to_common_Gaussian_constants=log_evidence,
            normalizer_scope=("Initial empirical mean declaredISratio times sequentialcamera"
                              "normalizers;logvalue isnotanunbiasedloglikelihood. Targetdepends"
                              "on thecaller'sdeclaredinitialproposal."))
    return result
