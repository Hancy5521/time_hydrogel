"""Explicit independent random increments on the frozen formation mechanics.

The two functions below were source-copied from frozen V1, changing only the
random-seed inputs/provenance. No physical rate, numerical clock, State update,
observation, intervention or mechanical computation is changed. Import is inert.
"""
from time import perf_counter
import numpy as np
from aam_formation_cycle_core import (
    SCHEMA, ACTIONS, COMMON_STOPS, _physical_config, _uint32, _cost, _advance,
    extract_teacher_marks, future_clock,
)
from aam_formation_actual_rollout import (
    SCHEMA as ROLLOUT_SCHEMA, _frozen_value, _packet_copy,
    packet_fingerprint, validate_rollout_inputs,
)
from aam_snapshot_process import PORTS, clone, initialize, observe, values


def _explicit_seeds(seeds):
    values = np.asarray(seeds)
    if values.shape != (3,) or values.dtype.kind not in 'iu' or np.any(values < 0) or np.any(values.astype(np.uint64) >= 2**32):
        raise ValueError('Three explicitly registered uint32 native seeds are required.')
    return values.astype(np.uint32, copy=True)


def formation_prefix_streamed(config, seed, factors, *, native_seeds, record_teacher=False, record_control=False, mother_id=None):
    """One actual nonzero-MBAA prefix, three native calls, all past States kept."""
    started = perf_counter()
    seed = _uint32(seed)
    native_seeds = _explicit_seeds(native_seeds)
    if record_control and not record_teacher:
        raise ValueError("The one integrated native control needs already annotated teacher intervals.")
    actual_config = _physical_config(config, factors)
    packet = initialize(actual_config, seed)  # unchanged physical initialization law
    initial_packet = _packet_copy(packet, readonly=True)
    frames = {0.: clone(packet[0])}
    cost, records, clean, calls = _cost(), [], [], []
    error = None
    try:
        for camera in range(3):
            clock = np.asarray(config["prefix_clock"][:camera+2], dtype=np.float64)
            call_seed = int(native_seeds[camera])  # explicit independent named-stream increment
            call = {"camera": camera, "seed": call_seed, "clock": clock.copy(),
                    "completed": False, "position_before": packet[0].position.copy()}
            calls.append(call)
            try:
                record = _advance(packet, clock, call_seed, record_teacher, cost, record_control=record_control)
                call["completed"] = True
            finally:
                call["position_after"] = packet[0].position.copy()
            if record is not None:
                records.append(record)
            frames[float(clock[-1])] = clone(packet[0])
            clean.append(np.asarray(observe(_packet_copy(packet)), dtype=np.float64))
    except Exception as exc:
        error = {"type": type(exc).__name__, "message": str(exc), "camera": len(calls)-1}
    for state in frames.values():
        for array in state:
            array.flags.writeable = False
    return {"schema": SCHEMA, "status": "complete" if error is None else "failed",
            "valid": error is None, "mother_id": mother_id, "seed": seed,
            "initial_packet": initial_packet, "packet": _packet_copy(packet, readonly=True),
            "frames": frames, "clean": np.asarray(clean, dtype=np.float64).reshape(-1, 4),
            "factors": np.asarray(factors).copy(), "cost": cost, "native_calls": calls,
            "teacher_records": records, **extract_teacher_marks(records), "error": error,
            "elapsed_seconds": perf_counter()-started,
            "scope": "Actual unquenched AAm/MBAA prefix; synthetic forward chemistry, not measured kinetics."}


def formation_rollout_streamed(actual_packet, action, config, seed, *, native_seeds, placement_seed, properties=None,
                      record_teacher=False, record_control=False, mother_id=None):
    """Continue actual MBAA chemistry on the common numerical stop grid.

    This is the same physical action/placement law as the frozen thin rollout,
    but each single future call includes both declared pulse stops up to its age.
    Exactly three native calls reach3.25,4.85,7.20; no quench, graft, prefix replay
    or extra progress call at2.30/3.00. All readouts use independent State copies.
    """
    started = perf_counter()
    native_seeds = _explicit_seeds(native_seeds)
    _physical_config(config, actual_packet[4])
    if record_control and not record_teacher:
        raise ValueError("The one integrated native control needs already annotated teacher intervals.")
    action, prefix, future = validate_rollout_inputs(actual_packet, action, config, seed,
                                                      mother_id=mother_id)
    if not any(np.array_equal(action, candidate) for candidate in ACTIONS):
        raise ValueError("Use one of the six fixed feed-time/spatial-allocation actions.")
    if properties is not None and not callable(properties):
        raise TypeError("properties is an optional callback(readonly_actual_packet).")
    original_hash = packet_fingerprint(actual_packet)
    packet = _packet_copy(actual_packet)
    packet[1][3] = action[0]
    state = packet[0]
    absent = ~state.present
    rng = np.random.default_rng(_uint32(placement_seed))
    count = int(absent.sum())
    centers = np.where(rng.random(count) < action[1], PORTS[0], PORTS[1])
    state.points[absent, 0] = (centers+rng.normal(0., .64, count)) % packet[2][2]
    state.unwrapped[absent] = state.points[absent]
    branch_start = _packet_copy(packet, readonly=True)
    snapshots, calls, records, cost = [], [], [], _cost()
    error, phase = None, "prepare"
    try:
        for interval, age in enumerate(future):
            clock = future_clock(config, float(age))
            phase = "native_progress"
            call = {"interval": interval, "seed": int(native_seeds[interval]),
                    "requested_time": float(age), "clock": clock.copy(), "completed": False,
                    "position_before": packet[0].position.copy(), "counts_before": packet[0].counts.copy()}
            calls.append(call)
            try:
                record = _advance(packet, clock, call["seed"], record_teacher, cost, record_control=record_control)
                call["completed"] = True
            finally:
                call.update(position_after=packet[0].position.copy(), counts_after=packet[0].counts.copy())
            if record is not None:
                records.append(record)
            snapshot = {"time": float(age), "packet": _packet_copy(packet, readonly=True),
                        "q": None, "properties": None}
            snapshots.append(snapshot)
            phase = "q_readout"
            read_started = perf_counter()
            try:
                q = np.asarray(values(_packet_copy(packet)), dtype=np.float64)
                if q.shape != (3,) or not np.isfinite(q).all():
                    raise ValueError("Expected finite native qL,qR,conversion.")
                q.flags.writeable = False
                snapshot["q"] = q
            finally:
                cost["q_read_seconds"] += perf_counter()-read_started
            if properties is not None:
                phase = "properties_callback"
                supplied = _packet_copy(packet, readonly=True)
                callback_hash = packet_fingerprint(supplied)
                property_started = perf_counter()
                try:
                    output = properties(supplied)
                    if (packet_fingerprint(supplied) != callback_hash
                            or any(a.flags.writeable for a in (*supplied[0], *supplied[1:]))):
                        raise ValueError("Property callback mutated its independent read-only physical input.")
                    snapshot["properties"] = _frozen_value(output)
                finally:
                    cost["property_seconds"] += perf_counter()-property_started
        phase = "complete"
    except Exception as exc:
        error = {"phase": phase, "interval": len(calls)-1,
                 "type": type(exc).__name__, "message": str(exc)}
    if packet_fingerprint(actual_packet) != original_hash:
        raise RuntimeError("Original mother history changed; no hidden repair is permitted.")
    q_rows = [snapshot["q"] for snapshot in snapshots if snapshot["q"] is not None]
    return {"schema": ROLLOUT_SCHEMA, "core_schema": SCHEMA,
            "status": "complete" if error is None else "failed", "valid": error is None,
            "mother_id": mother_id, "action": action, "seed": int(seed),
            "seed_is_compatibility_marker_not_rng": True,
            "explicit_native_seeds": native_seeds.copy(), "explicit_placement_seed": int(placement_seed),
            "prefix_clock": prefix, "future_times": future, "common_stops": np.asarray(COMMON_STOPS),
            "mother_packet_sha256": original_hash,
            "original_packet": _packet_copy(actual_packet, readonly=True),
            "branch_start_packet": branch_start, "packet": _packet_copy(packet, readonly=True),
            "snapshots": snapshots, "q": np.stack(q_rows) if q_rows else np.empty((0, 3)),
            "cost": cost, "native_calls": calls, "teacher_records": records,
            **extract_teacher_marks(records), "error": error,
            "elapsed_seconds": perf_counter()-started,
            "scope": "Actual nonzeroMBAA continuation; common numerical feed stops, learned psi not RN-reverted."}


