"""V4 whole-history annealed resample--move inference on explicit uint32 seeds.

Import is inert.  Every path is an actual initialization plus three unmodified
native intervals.  No guide, fit, learned likelihood, property computation or
hidden outcome enters this module.  Candidate histories (including rejected
ones) are streamed through caller-owned archive callbacks before reuse.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from time import perf_counter

import numpy as np
from scipy.special import logsumexp

from aam_formation_actual_rollout import _packet_copy, packet_fingerprint
from aam_formation_cycle_core import PREFIX_CLOCK, _advance, _cost, _physical_config
from aam_snapshot_process import clone, initialize, likelihood, observe


SCHEMA = "aam-complete-seed-path-annealing-v4"
PATH_SCHEMA = "aam-complete-four-seed-prefix-v4"
PARTICLES = 64
MAX_STAGES = 20
CESS_FRACTION = .8
RESAMPLE_ESS = 32.
REPLAY_SEGMENTS = (3, 3, 2, 1)


def _z4(value):
    raw = np.asarray(value)
    if (raw.shape != (4,) or raw.dtype.kind not in "iu" or np.any(raw < 0)
            or np.any(raw.astype(np.uint64) >= 2**32)):
        raise ValueError("A path requires exactly four full-range uint32 seeds.")
    return raw.astype(np.uint32, copy=True)


def _query(config, measured):
    y, noise = np.asarray(measured, float), np.asarray(config["noise"], float)
    if y.shape == (12,):
        y = y.reshape(3, 4)
    if (y.shape != (3, 4) or not np.isfinite(y).all() or noise.shape != (4,)
            or not np.isfinite(noise).all() or np.any(noise <= 0.)):
        raise ValueError("Use the actual finite3x4 early readings and four fixed noise scales.")
    return y.copy(), noise.copy()


def _seed(rng):
    value = rng.integers(0, 2**32, dtype=np.uint32)
    if (np.ndim(value) != 0 or isinstance(value, (bool, np.bool_))
            or int(value) != value or not 0 <= int(value) < 2**32):
        raise ValueError("The named stream must return one uniform uint32 seed.")
    return int(value)


def _uniform(rng):
    value = rng.random()
    if np.ndim(value) != 0 or not np.isfinite(value) or not 0. <= float(value) < 1.:
        raise ValueError("The named controller must return a uniform in[0,1).")
    return float(value)


def _weights(weights, size=PARTICLES):
    w = np.asarray(weights, dtype=float)
    if (w.shape != (size,) or not np.isfinite(w).all() or np.any(w < 0.)
            or not 0. < w.sum() < np.inf):
        raise ValueError("Use all finite nonnegative particle masses, including zero atoms.")
    return w/w.sum()


def _logs(weights):
    w = _weights(weights)
    result = np.full(PARTICLES, -np.inf)
    positive = w > 0.
    result[positive] = np.log(w[positive])
    return result


def conditional_ess(weights, log_likelihoods, delta):
    """64*(sum W exp(delta*ell))^2/sum W exp(2delta*ell), not W squared."""
    ell = np.asarray(log_likelihoods, float)
    if ell.shape != (PARTICLES,) or not np.isfinite(ell).all() or not 0. <= delta <= 1.:
        raise ValueError("CESS requires64finite complete log likelihoods and delta in[0,1].")
    lw = _logs(weights)
    result = PARTICLES*np.exp(2.*logsumexp(lw+delta*ell)-logsumexp(lw+2.*delta*ell))
    # Only floating-point roundoff can exceed the analytical upper bound64.
    if not np.isfinite(result) or result <= 0. or result > PARTICLES*(1.+1.e-12):
        raise ArithmeticError("Invalid weighted CESS.")
    return float(min(result, PARTICLES))


def next_beta(beta, weights, log_likelihoods, stage):
    """Largest admissible increment by80fixed bisections; last stage reaches1."""
    if not 0. <= beta < 1. or type(stage) is not int or not 1 <= stage <= MAX_STAGES:
        raise ValueError("Use a current beta<1 and a registered stage1..20.")
    remaining = 1.-float(beta)
    endpoint_cess = conditional_ess(weights, log_likelihoods, remaining)
    if endpoint_cess >= CESS_FRACTION*PARTICLES:
        return 1., endpoint_cess, False
    if stage == MAX_STAGES:
        return 1., endpoint_cess, True
    lower, upper = 0., remaining
    for _ in range(80):
        middle = (lower+upper)/2.
        if conditional_ess(weights, log_likelihoods, middle) >= CESS_FRACTION*PARTICLES:
            lower = middle
        else:
            upper = middle
    chosen = float(beta+lower)
    if not float(beta) < chosen < 1.:
        raise ArithmeticError("Floating-point temperature cannot advance; do not invent a rescue step.")
    return chosen, conditional_ess(weights, log_likelihoods, chosen-beta), False


def update_weights(weights, log_likelihoods, delta):
    ell = np.asarray(log_likelihoods, float)
    if ell.shape != (PARTICLES,) or not np.isfinite(ell).all() or not 0. <= delta <= 1.:
        raise ValueError("Use64complete finite log likelihoods and a valid temperature increment.")
    logw = _logs(weights)+delta*ell
    normalizer = float(logsumexp(logw))
    w = np.exp(logw-normalizer)
    w /= w.sum()
    return w, normalizer, float(1./np.square(w).sum())


def systematic_indices(weights, rng):
    w = _weights(weights, len(weights))
    u = _uniform(rng)
    cdf = np.cumsum(w); cdf[-1] = 1.
    locations = np.minimum((u+np.arange(PARTICLES))/PARTICLES, np.nextafter(1., 0.))
    indices = np.searchsorted(cdf, locations, side="right")
    if indices.shape != (PARTICLES,) or np.any(indices >= len(w)):
        raise ArithmeticError("Invalid complete-support systematic ancestors.")
    return indices.astype(np.int64), u


def _fingerprints(checkpoints):
    """Exact full-packet identity plus an explicit reacted-structure field hash."""
    full, reacted = hashlib.sha256(), hashlib.sha256()
    for time, packet in zip(PREFIX_CLOCK, checkpoints, strict=True):
        full.update(str(time).encode("ascii")); full.update(packet_fingerprint(packet).encode("ascii"))
        state = packet[0]
        fields = {"present": state.present, "active": state.active, "available": state.available,
                  "owner": state.owner, "counts": state.counts,
                  "edges": state.edges[:int(state.counts[1])],
                  "lifts": state.lifts[:int(state.counts[1])], "winding": state.winding}
        reacted.update(str(time).encode("ascii"))
        for name, array in fields.items():
            array = np.asarray(array)
            reacted.update(name.encode("ascii")); reacted.update(array.dtype.str.encode("ascii"))
            reacted.update(json.dumps(list(array.shape)).encode("ascii"))
            reacted.update(np.ascontiguousarray(array).tobytes())
    return full.hexdigest(), reacted.hexdigest()


def _new_cost():
    return {**_cost(), "actual_initializations": 0, "initialization_seconds": 0.,
            "path_evaluations": 0, "copy_seconds": 0., "archive_seconds": 0.,
            "packet_copy_calls": 0, "frame_copy_calls": 0, "fingerprint_seconds": 0.,
            "archives_written": 0, "archive_bytes": 0, "seed_draws": 0,
            "acceptance_uniforms": 0, "order_permutations": 0,
            "systematic_uniforms": 0, "likelihood_evaluations": 0}


def _copy_packet(packet, cost, *, readonly=False):
    segment = perf_counter(); cost["packet_copy_calls"] += 1
    try:
        return _packet_copy(packet, readonly=readonly)
    finally:
        cost["copy_seconds"] += perf_counter()-segment


def _copy_state(state, cost):
    segment = perf_counter(); cost["frame_copy_calls"] += 1
    try:
        return clone(state)
    finally:
        cost["copy_seconds"] += perf_counter()-segment


def _merge_cost(total, fresh):
    for name, value in fresh.items():
        if name not in total:
            total[name] = 0
        total[name] += value


def generate_path(config, factors, z, measured, *, previous=None, changed_coordinate=None,
                  mother_id=None, birth_id=None):
    """Actual F_psi(z); partial refresh replays3/3/2/1 native intervals.

    A complete previous path can supply only checkpoints strictly before the
    changed increment.  All four checkpoint packets include the full State and
    physical parameter arrays.  Returned readouts use copies.  Computational
    failures remain failed paths, never zero-likelihood MH rejections.
    """
    started = perf_counter(); z = _z4(z); y, noise = _query(config, measured)
    actual = _physical_config(config, factors)
    if changed_coordinate is not None and (type(changed_coordinate) is not int or not 0 <= changed_coordinate <= 3):
        raise ValueError("A changed coordinate is one of0,1,2,3.")
    if (previous is None) != (changed_coordinate is None):
        raise ValueError("A partial refresh needs both the complete previous path and its changed coordinate.")
    if previous is not None:
        oldz = _z4(previous["z"])
        retained = np.arange(4) != changed_coordinate
        if (previous.get("valid") is not True or not np.array_equal(oldz[retained], z[retained])
                or len(previous["checkpoint_packets"]) != 4
                or not np.array_equal(previous["factors"], np.asarray(factors))):
            raise ValueError("Only the declared coordinate may change on a complete same-law parent.")
    cost = _new_cost(); cost["path_evaluations"] = 1
    checkpoints, clean, calls = [], [], []
    packet = None; error = None; phase = "initialize"; camera = -1
    try:
        if previous is None or changed_coordinate == 0:
            segment = perf_counter(); cost["actual_initializations"] += 1
            try:
                packet = initialize(actual, int(z[0]))
            finally:
                cost["initialization_seconds"] += perf_counter()-segment
            checkpoints.append(_copy_packet(packet, cost, readonly=True)); start_camera = 0
        else:
            start_camera = changed_coordinate-1
            checkpoints = [_copy_packet(value, cost, readonly=True)
                           for value in previous["checkpoint_packets"][:start_camera+1]]
            clean = [row.copy() for row in np.asarray(previous["clean"])[:start_camera]]
            calls = [deepcopy(row) for row in previous["native_calls"][:start_camera]]
            for row in calls:
                row["reused_checkpoint_interval"] = True
            packet = _copy_packet(checkpoints[-1], cost)
        for camera in range(start_camera, 3):
            phase = "native_progress"
            clock = np.asarray(config["prefix_clock"][:camera+2], float)
            call = {"camera": camera, "seed": int(z[camera+1]), "clock": clock.copy(),
                    "completed": False, "position_before": packet[0].position.copy(),
                    "reused_checkpoint_interval": False}
            calls.append(call)
            try:
                _advance(packet, clock, int(z[camera+1]), False, cost)
                call["completed"] = True
            finally:
                call["position_after"] = packet[0].position.copy()
            checkpoints.append(_copy_packet(packet, cost, readonly=True))
            phase = "clean_observation"
            read_packet = _copy_packet(packet, cost); segment = perf_counter()
            try:
                clean.append(np.asarray(observe(read_packet), float).copy())
            finally:
                cost["q_read_seconds"] += perf_counter()-segment
        phase = "complete_likelihood"
        clean_array = np.asarray(clean, float)
        if clean_array.shape != (3, 4) or not np.isfinite(clean_array).all():
            raise ArithmeticError("The actual full path did not produce three finite copied readouts.")
        camera_ll = np.asarray(likelihood(clean_array, y, noise), float)
        cost["likelihood_evaluations"] += 1
        if camera_ll.shape != (3,) or not np.isfinite(camera_ll).all():
            raise ArithmeticError("Computational nonfinite likelihood is not physical zero support.")
        segment = perf_counter()
        try:
            full_hash, reacted_hash = _fingerprints(checkpoints)
        finally:
            cost["fingerprint_seconds"] += perf_counter()-segment
    except Exception as exc:
        error = {"type": type(exc).__name__, "message": str(exc), "phase": phase, "camera": camera}
        clean_array = np.asarray(clean, float).reshape(-1, 4)
        camera_ll = np.empty(0, dtype=float); full_hash = reacted_hash = ""
    frames = {float(time): _copy_state(value[0], cost) for time, value in zip(PREFIX_CLOCK, checkpoints)}
    for state in frames.values():
        for array in state:
            array.flags.writeable = False
    return {"schema": PATH_SCHEMA, "valid": error is None,
        "status": "complete" if error is None else "failed", "mother_id": mother_id,
        "birth_id": birth_id, "parent_birth_id": None if previous is None else previous["birth_id"],
        "changed_coordinate": changed_coordinate, "z": z,
        "factors": np.asarray(factors, float).copy(), "checkpoint_packets": checkpoints,
        "initial_packet": None if not checkpoints else checkpoints[0],
        "packet": None if packet is None else _copy_packet(packet, cost, readonly=True),
        "frames": frames, "clean": clean_array, "camera_log_likelihood": camera_ll,
        "log_likelihood": None if error is not None else float(camera_ll.sum()),
        "history_fingerprint": full_hash, "reacted_structure_fingerprint": reacted_hash,
        "native_calls": calls, "cost": cost, "error": error,
        "elapsed_seconds": perf_counter()-started,
        "scope": "Full actual microscopic hypothesis; exact field hashes do not alone establish useful structural diversity."}


def _save_path(path, label, callback, cost):
    segment = perf_counter()
    ref = callback(path, label)
    cost["archive_seconds"] += perf_counter()-segment; cost["archives_written"] += 1
    if isinstance(ref, dict) and "bytes" in ref:
        cost["archive_bytes"] += int(ref["bytes"])
    return ref


def _particles(paths):
    return [{"packet": path["packet"], "initial_packet": path["initial_packet"],
             "frames": path["frames"], "initial_seed": int(path["z"][0]),
             "interval_seeds": path["z"][1:].copy(), "z": path["z"].copy(),
             "birth_id": path["birth_id"], "history_fingerprint": path["history_fingerprint"],
             "camera": 2} for path in paths]


def _export(paths, refs, weights, cost, *, valid, error=None, **extra):
    w = np.asarray(weights, float)
    return {"schema": SCHEMA, "valid": valid, "status": "complete" if valid else "failed",
        "error": error, "paths": paths, "path_refs": refs, "weights": w.copy(),
        "packets": [path["packet"] for path in paths], "particles": _particles(paths),
        "z": np.asarray([path["z"] for path in paths], dtype=np.uint32).reshape(-1, 4),
        "initialization_seeds": np.asarray([path["z"][0] for path in paths], dtype=np.uint32),
        "initial_source_ids": np.asarray([int(path["z"][0]) for path in paths], dtype=np.int64),
        "log_likelihoods": np.asarray([path["log_likelihood"] for path in paths], float),
        "cost": cost, "initial_weight_semantics": "Full iid uniform uint32 prior;no initial guide and no1/a.",
        "final_equal_weights_are_MC_approximation": True,
        "scope": "Actual latent history inference; final ESS is not an effectiveness result.", **extra}


def _schedule(value):
    if value is None:
        return None
    b = np.asarray(value, float)
    if (b.ndim != 1 or not 2 <= len(b) <= MAX_STAGES+1 or not np.isfinite(b).all()
            or b[0] != 0. or b[-1] != 1. or np.any(np.diff(b) <= 0.)):
        raise ValueError("A frozen teacher schedule includes0,allstrictlyincreasing stages,and1,atmost20stages.")
    return b.copy()


def anneal_paths(config, factors, measured, initial_rng, seed_rng, order_rng, accept_rng,
                 resample_rng, *, save_path, load_path=None, beta_schedule=None, mother_id=None):
    """N64,CESS.8N,atmost20temperatures; four single-seed MH moves per stage.

    beta_schedule=None adapts to this population.  Teacher diagnostics receive
    the frozen learned sequence, yet use their own masses for middle ESS tests.
    save_path(path,label) must exclusively archive the complete actual path and
    return a JSON-serializable reference.  No already-computed V3 posterior is
    accepted.  load_path is accepted for common driver callback symmetry only.
    """
    del load_path
    _query(config, measured); _physical_config(config, factors)
    fixed = _schedule(beta_schedule)
    if not callable(save_path):
        raise TypeError("Stream all actual accepted and rejected paths through save_path.")
    cost = _new_cost(); paths, refs = [], []
    w = np.full(PARTICLES, 1./PARTICLES); stages = []; betas = [0.]
    for slot in range(PARTICLES):
        z = np.asarray([_seed(initial_rng) for _ in range(4)], dtype=np.uint32)
        cost["seed_draws"] += 4
        path = generate_path(config, factors, z, measured, mother_id=mother_id,
                             birth_id=f"prior_{slot:02d}")
        _merge_cost(cost, path["cost"])
        ref = _save_path(path, f"prior/{slot:02d}", save_path, cost)
        if not path["valid"]:
            return _export(paths, refs, w[:len(paths)], cost, valid=False,
                           error={"phase": "prior", "slot": slot, "path": ref, "details": path["error"]},
                           stages=stages, stages_used=0, beta_schedule=np.asarray(betas), betas=np.asarray(betas))
        paths.append(path); refs.append(ref)
    for stage in range(1, MAX_STAGES+1):
        beta = betas[-1]; ell = np.asarray([path["log_likelihood"] for path in paths])
        if fixed is None:
            target, cess, forced = next_beta(beta, w, ell, stage)
        else:
            target = float(fixed[stage]); cess = conditional_ess(w, ell, target-beta); forced = False
        incoming = w.copy(); updated, logz, ess = update_weights(w, ell, target-beta)
        final = target == 1.
        resampled = bool(final or ess < RESAMPLE_ESS)
        before_refs, before_z = list(refs), np.stack([path["z"] for path in paths])
        if resampled:
            parents, uniform = systematic_indices(updated, resample_rng); cost["systematic_uniforms"] += 1
            paths = [paths[int(index)] for index in parents]
            refs = [refs[int(index)] for index in parents]
            w = np.full(PARTICLES, 1./PARTICLES)
        else:
            parents = np.arange(PARTICLES); uniform = None; w = updated.copy()
        row = {"stage": stage, "beta_before": beta, "beta": target, "delta": target-beta,
            "conditional_ESS": cess, "CESS_target": CESS_FRACTION*PARTICLES,
            "forced": forced, "temperature_selection": "adaptive" if fixed is None else "frozen_learned_schedule",
            "weights_incoming": incoming, "log_likelihoods_before": ell.copy(),
            "weights_updated": updated.copy(), "relative_log_normalizer_increment": logz,
            "ordinary_ESS_before_resampling": ess, "resampled": resampled,
            "systematic_uniform": uniform, "parents": parents.copy(),
            "path_refs_before": before_refs, "z_before": before_z,
            "weights_after_resampling": w.copy(), "moves": [], "orders": np.empty((PARTICLES, 4), dtype=np.int64)}
        for slot in range(PARTICLES):
            order = np.asarray(order_rng.permutation(4))
            cost["order_permutations"] += 1
            if order.shape != (4,) or order.dtype.kind not in "iu" or not np.array_equal(np.sort(order), np.arange(4)):
                raise ValueError("Each particle must receive its own permutation of all four coordinates.")
            row["orders"][slot] = order
            for order_position, coordinate_raw in enumerate(order):
                coordinate = int(coordinate_raw); previous = paths[slot]; previous_ref = refs[slot]
                z = previous["z"].copy(); z[coordinate] = _seed(seed_rng); cost["seed_draws"] += 1
                proposal = generate_path(config, factors, z, measured, previous=previous,
                    changed_coordinate=coordinate, mother_id=mother_id,
                    birth_id=f"stage_{stage:02d}_slot_{slot:02d}_move_{order_position}")
                _merge_cost(cost, proposal["cost"])
                move = {"slot": slot, "order_position": order_position, "coordinate": coordinate,
                    "parent_ref": previous_ref, "parent_birth_id": previous["birth_id"],
                    "parent_z": previous["z"].copy(), "proposal_z": z.copy(),
                    "log_likelihood_before": previous["log_likelihood"],
                    "log_likelihood_proposed": proposal["log_likelihood"],
                    "native_segments_required": REPLAY_SEGMENTS[coordinate],
                    "candidate_cost": deepcopy(proposal["cost"])}
                if proposal["valid"]:
                    u = _uniform(accept_rng); cost["acceptance_uniforms"] += 1
                    logalpha = min(0., target*(proposal["log_likelihood"]-previous["log_likelihood"]))
                    accepted = bool(u == 0. or np.log(u) < logalpha)
                    move.update(acceptance_uniform=u, log_acceptance_probability=logalpha, accepted=accepted,
                        seed_changed=not np.array_equal(proposal["z"], previous["z"]),
                        full_history_changed=proposal["history_fingerprint"] != previous["history_fingerprint"],
                        reacted_structure_changed=proposal["reacted_structure_fingerprint"] != previous["reacted_structure_fingerprint"])
                else:
                    move.update(acceptance_uniform=None, log_acceptance_probability=None, accepted=None,
                                error=proposal["error"])
                proposal["proposal_ledger"] = deepcopy(move)
                ref = _save_path(proposal, f"stage/{stage:02d}/slot/{slot:02d}/move/{order_position}", save_path, cost)
                move["proposal_ref"] = ref; row["moves"].append(move)
                if not proposal["valid"]:
                    stages.append(row)
                    return _export(paths, refs, w, cost, valid=False,
                        error={"phase": "MH_computational_failure_not_rejection", "stage": stage,
                               "slot": slot, "coordinate": coordinate, "path": ref, "details": proposal["error"]},
                        stages=stages, stages_used=stage-1, beta_schedule=np.asarray(betas), betas=np.asarray(betas))
                if move["accepted"]:
                    paths[slot] = proposal; refs[slot] = ref
        row.update(path_refs_after=list(refs), z_after=np.stack([path["z"] for path in paths]),
                   log_likelihoods_after=np.asarray([path["log_likelihood"] for path in paths]),
                   weights_after=w.copy())
        stages.append(row); betas.append(target)
        if final:
            break
    if betas[-1] != 1. or not np.array_equal(w, np.full(PARTICLES, 1./PARTICLES)):
        raise AssertionError("The fixed algorithm must end at beta1 with explicit equal64terminal slots.")
    expected = 192+576*len(stages)
    if cost["native_segment_attempted"] != expected or cost["native_segment_completed"] != expected:
        raise AssertionError("Whole-path native cost differs from192+576T; no free cached transitions allowed.")
    return _export(paths, refs, w, cost, valid=True, stages=stages, stages_used=len(stages),
        beta_schedule=np.asarray(betas), betas=np.asarray(betas),
        relative_log_normalizer_estimate=float(sum(row["relative_log_normalizer_increment"] for row in stages)),
        normalizer_scope="Unnormalized Gaussian camera likelihood;not an absolute material-evidence claim.",
        expected_native_prefix_cost=expected)


def sir_paths(config, factors, measured, initial_rng, resample_rng, *, stages_used,
              save_path, load_path, mother_id=None):
    """Independent whole-prior pool M64+192T,full likelihood,final64systematic.

    The complete weighted pool is archived and returned by reference.  Only the
    selected64 paths are loaded, so no pool-size collection of State lives in
    RAM.  Compressed final64 slots do not preserve the pool's exact mass.
    """
    _query(config, measured); _physical_config(config, factors)
    if type(stages_used) is not int or not 1 <= stages_used <= MAX_STAGES:
        raise ValueError("Match the actual learned T in1..20, never choose SIR size from its results.")
    if not callable(save_path) or not callable(load_path):
        raise TypeError("SIR needs complete actual path archive and checked load callbacks.")
    count = PARTICLES+192*stages_used; cost = _new_cost(); refs = []; loglikes = []
    for index in range(count):
        z = np.asarray([_seed(initial_rng) for _ in range(4)], dtype=np.uint32); cost["seed_draws"] += 4
        path = generate_path(config, factors, z, measured, mother_id=mother_id, birth_id=f"pool_{index:04d}")
        _merge_cost(cost, path["cost"])
        ref = _save_path(path, f"pool/{index:04d}", save_path, cost); refs.append(ref)
        if not path["valid"]:
            return _export([], [], np.empty(0), cost, valid=False,
                error={"phase": "SIR_pool", "index": index, "path": ref, "details": path["error"]},
                stages_used=stages_used, pool_refs=refs, log_likelihoods_pool=np.asarray(loglikes))
        loglikes.append(path["log_likelihood"])
        del path
    ell = np.asarray(loglikes); weights = np.exp(ell-logsumexp(ell)); weights /= weights.sum()
    selected, uniform = systematic_indices(weights, resample_rng); cost["systematic_uniforms"] += 1
    paths = []
    for index in selected:
        path = load_path(refs[int(index)])
        if (path.get("valid") is not True or path.get("schema") != PATH_SCHEMA
                or path["log_likelihood"] != ell[int(index)]):
            raise ValueError("A selected actual pool history changed before terminal compression.")
        paths.append(path)
    expected = 3*count
    if cost["native_segment_attempted"] != expected or cost["native_segment_completed"] != expected:
        raise AssertionError("Independent prior SIR must spend3M=192+576Tnative prefixes.")
    return _export(paths, [refs[int(i)] for i in selected], np.full(PARTICLES, 1./PARTICLES), cost,
        valid=True, stages_used=stages_used, pool_size=count, pool_refs=refs,
        pool_weights=weights, log_likelihoods_pool=ell, pool_log_likelihoods=ell.copy(),
        selected_indices=selected, systematic_uniform=uniform,
        beta_schedule=np.asarray([0., 1.]), expected_native_prefix_cost=expected,
        pool_ESS=float(1./np.square(weights).sum()),
        pool_support_retained_by_reference=True,
        compression_scope="Final64equal slots are a systematic MC approximation; full weighted pool is separately retained.")
