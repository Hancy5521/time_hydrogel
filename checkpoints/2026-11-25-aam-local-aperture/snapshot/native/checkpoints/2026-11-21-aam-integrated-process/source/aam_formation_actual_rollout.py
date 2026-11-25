"""Thin future AAm/MBAA formation rollout from an actual native packet.

No simulation, random draw, fit or file write occurs on import.  The physical
kernel and action law are those of ``aam_snapshot_process.branch``.  This layer
adds pure readouts, full State snapshots, failure/cost accounting and lossless
numeric serialization.  It never replays the observed prefix or converts an
actual history into a latent seed description.
"""
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from time import perf_counter
import ast
import hashlib
import inspect
import json

import numpy as np

from aam_edge_fork import State
from aam_snapshot_process import PORTS, clone, progress, values


SCHEMA = "aam-actual-formation-rollout-v1"
PACKET_FIELDS = ("state", "material", "geometry", "rates", "factors")
STATE_FIELDS = tuple(State._fields)


def _packet_copy(packet, *, readonly=False):
    copied = (clone(packet[0]), *(np.array(x, copy=True) for x in packet[1:]))
    if readonly:
        for array in (*copied[0], *copied[1:]):
            array.flags.writeable = False
    return copied


def packet_fingerprint(packet):
    """Hash dtype, shape and every stored byte, including spare native buffers."""
    h = hashlib.sha256()
    for name, array in zip((*STATE_FIELDS, *PACKET_FIELDS[1:]),
                           (*packet[0], *packet[1:]), strict=True):
        array = np.asarray(array)
        h.update(name.encode("ascii"))
        h.update(array.dtype.str.encode("ascii"))
        h.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
        h.update(np.ascontiguousarray(array).tobytes())
    return h.hexdigest()


def validate_rollout_inputs(actual_packet, action, config, seed, *, mother_id=None):
    """Deterministic shape/clock checks; no physical kernel or RNG is called."""
    if not isinstance(actual_packet, (tuple, list)) or len(actual_packet) != 5:
        raise ValueError("Use the actual (State, material, geometry, rates, factors) packet.")
    state, material, geometry, rates, factors = actual_packet
    if not isinstance(state, State):
        raise TypeError("Expected the existing 16-field native State, not a reconstructed graph.")
    material, geometry, rates, factors = map(np.asarray, (material, geometry, rates, factors))
    if material.shape != (4,) or geometry.shape != (10,) or rates.shape != (8, 2) or factors.shape != (3, 5):
        raise ValueError("Unexpected native material/geometry/rate/factor shapes.")
    if any(not np.isfinite(x).all() for x in (material, geometry, rates, factors)):
        raise ValueError("Native physical parameters must be finite.")
    if (np.any(rates < 0.) or np.any(factors <= 0.) or geometry[2] <= 0.
            or state.counts.shape != (7,) or state.position.shape != (4,)):
        raise ValueError("Invalid native rates, factors, box or clock ledger.")
    n = len(state.points)
    if (state.points.shape != (n, 3) or state.unwrapped.shape != (n, 3)
            or state.present.shape != (n,) or state.active.shape != (2*n,)
            or state.available.shape != (2*n,) or state.owner.shape != (2*n,)
            or int(material[0])+int(material[1]) != n):
        raise ValueError("Actual State sizes do not match the physical material.")
    if any(int(v) != v for v in material[:3]) or not 0 <= material[2] <= material[1]:
        raise ValueError("Expected integer molecule counts and valid initial MBAA count.")
    aam = int(material[0])
    if aam <= 0 or np.any(~state.present[:aam]):
        raise ValueError("Only future, not-yet-added MBAA may be repositioned.")
    action = np.asarray(action, dtype=np.float64)
    if action.shape != (2,) or not np.isfinite(action).all() or not 0. <= action[1] <= 1.:
        raise ValueError("Action must be (future pulse time, left-port probability in [0,1]).")
    prefix = np.asarray(config["prefix_clock"], dtype=np.float64)
    future = np.asarray(config["future_times"], dtype=np.float64)
    if (prefix.ndim != 1 or len(prefix) < 2 or prefix[0] != 0.
            or not np.isfinite(prefix).all() or np.any(np.diff(prefix) <= 0.)
            or future.ndim != 1 or not len(future) or not np.isfinite(future).all()
            or np.any(np.diff(future) <= 0.) or future[0] <= prefix[-1]):
        raise ValueError("Use the completed canonical prefix and strictly later future clocks.")
    if (state.counts[6] or state.present.all() or state.position[0] != prefix[-1]
            or action[0] <= state.position[0] or state.counts[2] != len(prefix)-1):
        raise ValueError("Branch only at the common completed prefix before an unconsumed pulse.")
    if (isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer))
            or seed < 0 or seed+max(51, len(future)) >= 2**32):
        raise ValueError("Base and all inherited native future seeds must fit uint32, without wrapping.")
    if (mother_id is not None and (isinstance(mother_id, (bool, np.bool_))
            or not isinstance(mother_id, (str, int, np.integer)))):
        raise TypeError("mother_id is caller-supplied integer/string grouping metadata only.")
    return action.copy(), prefix.copy(), future.copy()


def _frozen_value(value):
    """Copy callback outputs; never retain a reference to callback-owned arrays."""
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            raise TypeError("Properties must be numeric/string arrays without object pickle data.")
        copied = value.copy()
        copied.flags.writeable = False
        return copied
    if isinstance(value, np.generic):
        return _frozen_value(value.item())
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise TypeError("Property mapping keys must be strings.")
        return {key: _frozen_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return tuple(_frozen_value(item) for item in value)
    raise TypeError("Properties must contain only arrays, scalars and string-keyed containers.")


def branch_with_snapshots(actual_packet, action, config, seed, *, properties=None, mother_id=None):
    """Continue an actual prefix under the inherited future-feed action.

    ``properties`` is optional ``callback(readonly_packet)``.  It receives only
    an independent copy of the current physical packet: no query truth, target,
    seed history or future state is supplied.  Readouts occur after a completed
    native interval and never feed back to the live reaction/motion state.

    Invalid inputs raise before advancing.  A later kernel/readout/callback
    error returns ``status='failed'`` with the partial live packet, completed
    snapshots, attempted/completed native-call counts and the failing phase.
    There is no automatic replay, replacement mother or retry.
    """
    started = perf_counter()
    action, prefix, future = validate_rollout_inputs(actual_packet, action, config, seed,
                                                      mother_id=mother_id)
    if properties is not None and not callable(properties):
        raise TypeError("properties must be a callable or None.")
    original_hash = packet_fingerprint(actual_packet)
    original = _packet_copy(actual_packet, readonly=True)
    packet = _packet_copy(actual_packet)
    packet[1][3] = action[0]
    moved, _, geometry, _, _ = packet
    absent = ~moved.present
    rng = np.random.default_rng(int(seed)+51)
    count = int(absent.sum())
    centers = np.where(rng.random(count) < action[1], PORTS[0], PORTS[1])
    moved.points[absent, 0] = (centers+rng.normal(0., .64, count)) % geometry[2]
    moved.unwrapped[absent] = moved.points[absent]
    branch_start = _packet_copy(packet, readonly=True)
    clock = list(prefix)
    snapshots, calls = [], []
    cost = {"native_segment_attempted": 0, "native_segment_completed": 0,
            "native_seconds": 0., "q_read_seconds": 0., "property_seconds": 0.,
            "native_calls_per_future_boundary": 1,
            "internal_brownian_substeps_counted": False}
    error, phase, interval = None, "prepare", None
    try:
        for interval, time in enumerate(future):
            clock.append(float(time))
            phase = "native_progress"
            before_position = packet[0].position.copy()
            before_counts = packet[0].counts.copy()
            call = {"interval": interval, "seed": int(seed)+1+interval,
                    "requested_time": float(time), "clock": tuple(clock),
                    "position_before": before_position, "counts_before": before_counts,
                    "completed": False}
            calls.append(call)
            cost["native_segment_attempted"] += 1
            native_started = perf_counter()
            try:
                progress(packet, clock, int(seed)+1+interval)
                if packet[0].position[0] != time or packet[0].counts[6]:
                    raise RuntimeError("Native progress did not reach a completed requested boundary.")
                call["completed"] = True
                cost["native_segment_completed"] += 1
            finally:
                elapsed = perf_counter()-native_started
                call.update(seconds=elapsed, position_after=packet[0].position.copy(),
                            counts_after=packet[0].counts.copy())
                cost["native_seconds"] += elapsed
            snapshot = {"time": float(time), "packet": _packet_copy(packet, readonly=True),
                        "q": None, "properties": None}
            snapshots.append(snapshot)
            phase = "q_readout"
            read_started = perf_counter()
            try:
                q = np.array(values(_packet_copy(packet)), copy=True)
                if q.shape != (3,) or not np.isfinite(q).all():
                    raise RuntimeError("Inherited q readout must be three finite values.")
                q.flags.writeable = False
                snapshot["q"] = q
            finally:
                cost["q_read_seconds"] += perf_counter()-read_started
            if properties is not None:
                phase = "properties_callback"
                readonly_packet = _packet_copy(packet, readonly=True)
                callback_hash = packet_fingerprint(readonly_packet)
                property_started = perf_counter()
                try:
                    supplied = properties(readonly_packet)
                    if (packet_fingerprint(readonly_packet) != callback_hash
                            or any(a.flags.writeable for a in (*readonly_packet[0], *readonly_packet[1:]))):
                        raise RuntimeError("Property callback changed its read-only physical input.")
                    snapshot["properties"] = _frozen_value(supplied)
                finally:
                    cost["property_seconds"] += perf_counter()-property_started
        phase = "complete"
    except Exception as exc:
        error = {"phase": phase, "interval": interval,
                 "type": type(exc).__name__, "message": str(exc)}
    if packet_fingerprint(actual_packet) != original_hash:
        raise RuntimeError("The caller's original mother packet changed; stop rather than conceal it.")
    complete = error is None
    q_rows = [snapshot["q"] for snapshot in snapshots if snapshot["q"] is not None]
    return {"schema": SCHEMA, "status": "complete" if complete else "failed", "valid": complete,
            "mother_id": int(mother_id) if isinstance(mother_id, np.integer) else mother_id,
            "action": action, "seed": int(seed), "prefix_clock": prefix,
            "future_times": future, "mother_packet_sha256": original_hash,
            "original_packet": original, "branch_start_packet": branch_start,
            "packet": _packet_copy(packet, readonly=True), "snapshots": snapshots,
            "q": np.stack(q_rows) if q_rows else np.empty((0, 3)),
            "cost": cost, "native_calls": calls, "error": error,
            "elapsed_seconds": perf_counter()-started,
            "scope": ("Actual AAm/MBAA continuation, inherited feed-time/spatial placement action; "
                      "q is a microscopic graph statistic, not a calibrated physical property. "
                      "Across actions, shared suffix seeds are computational CRN, not a physical joint future.")}


def check_snapshot_contract(result, *, actual_packet=None):
    """Deterministic result/alias/accounting checks; no replay or new trajectory."""
    if result["schema"] != SCHEMA or result["status"] not in ("complete", "failed"):
        raise ValueError("Unknown rollout schema/status.")
    packets = [result[key] for key in ("original_packet", "branch_start_packet", "packet")]
    packets += [snapshot["packet"] for snapshot in result["snapshots"]]
    if actual_packet is not None:
        if packet_fingerprint(actual_packet) != result["mother_packet_sha256"]:
            raise AssertionError("Mother packet differs from its pre-rollout fingerprint.")
        arrays_actual = (*actual_packet[0], *actual_packet[1:])
        if any(np.shares_memory(x, y) for p in packets for x in (*p[0], *p[1:]) for y in arrays_actual):
            raise AssertionError("A saved packet aliases the caller's physical state.")
    for i, packet in enumerate(packets):
        arrays = (*packet[0], *packet[1:])
        if any(array.flags.writeable for array in arrays):
            raise AssertionError("Saved physical snapshots must be read-only copies.")
        for other in packets[i+1:]:
            if any(np.shares_memory(x, y) for x in arrays for y in (*other[0], *other[1:])):
                raise AssertionError("Two physical snapshots share writable storage.")
    calls = result["native_calls"]
    completed = sum(bool(call["completed"]) for call in calls)
    if (len(calls) != result["cost"]["native_segment_attempted"]
            or completed != result["cost"]["native_segment_completed"]
            or completed != len(result["snapshots"])):
        raise AssertionError("Native interval attempts/completions/snapshots disagree.")
    for i, (snapshot, call) in enumerate(zip(result["snapshots"], calls, strict=False)):
        if (not call["completed"] or snapshot["time"] != result["future_times"][i]
                or snapshot["packet"][0].position[0] != snapshot["time"]
                or snapshot["packet"][0].counts[6] or call["seed"] != result["seed"]+1+i):
            raise AssertionError("A saved future frame/clock/seed is inconsistent.")
    if result["valid"] != (result["status"] == "complete"):
        raise AssertionError("Validity/status mismatch.")
    if result["valid"] and (completed != len(result["future_times"]) or result["error"] is not None):
        raise AssertionError("A complete rollout must contain every declared native future interval.")
    return {"passed": True, "snapshots": len(result["snapshots"]),
            "native_segment_attempted": len(calls), "native_segment_completed": completed,
            "scope": "Pure saved-result/copy/accounting checks, not an independent physical replay."}


def _pack_tree(value, arrays):
    if isinstance(value, State):
        return {"__state__": {name: _pack_tree(array, arrays)
                             for name, array in zip(STATE_FIELDS, value, strict=True)}}
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            raise TypeError("No pickle/object arrays in physical rollout archives.")
        key = f"array_{len(arrays):06d}"
        arrays[key] = np.asarray(value)
        return {"__array__": key, "dtype": value.dtype.str, "shape": list(value.shape)}
    if isinstance(value, np.generic):
        return _pack_tree(value.item(), arrays)
    if isinstance(value, Mapping):
        return {"__mapping__": [[key, _pack_tree(item, arrays)] for key, item in value.items()]}
    if isinstance(value, tuple):
        return {"__tuple__": [_pack_tree(item, arrays) for item in value]}
    if isinstance(value, list):
        return [_pack_tree(item, arrays) for item in value]
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise TypeError(f"Unserializable rollout field: {type(value).__name__}")


def save_rollout_npz(result, path):
    """Exclusive, no-pickle archive of all actual State fields and partial costs.

    A different array is stored for every frame/field, so variable native event
    buffer capacities never require padding, truncation or graph reconstruction.
    Native kernels are not called; existing files are never overwritten.
    """
    check_snapshot_contract(result)
    arrays = {}
    tree = _pack_tree(result, arrays)
    metadata = json.dumps({"schema": SCHEMA, "state_fields": STATE_FIELDS, "result": tree},
                          ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    path = Path(path)
    with path.open("xb") as stream:
        np.savez_compressed(stream, metadata_json=np.asarray(metadata), **arrays)
    return {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "bytes": path.stat().st_size, "arrays": len(arrays), "status": result["status"],
            "mother_id": result["mother_id"]}


def _unpack_tree(value, arrays):
    if isinstance(value, dict):
        if "__array__" in value:
            array = arrays[value["__array__"]]
            if array.dtype.str != value["dtype"] or list(array.shape) != value["shape"]:
                raise ValueError("Serialized native array shape/dtype mismatch.")
            copied = array.copy()
            copied.flags.writeable = False
            return copied
        if "__state__" in value:
            return State(*(_unpack_tree(value["__state__"][name], arrays) for name in STATE_FIELDS))
        if "__tuple__" in value:
            return tuple(_unpack_tree(item, arrays) for item in value["__tuple__"])
        if "__mapping__" in value:
            return {key: _unpack_tree(item, arrays) for key, item in value["__mapping__"]}
        return {key: _unpack_tree(item, arrays) for key, item in value.items()}
    if isinstance(value, list):
        return [_unpack_tree(item, arrays) for item in value]
    return value


def load_rollout_npz(path, *, expected_sha256=None):
    """Read stored arrays once; does not regenerate or resume a physical state."""
    path = Path(path)
    if expected_sha256 is not None and hashlib.sha256(path.read_bytes()).hexdigest() != expected_sha256:
        raise ValueError("Rollout archive hash mismatch.")
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata_json"].item()))
        arrays = {key: archive[key] for key in archive.files if key != "metadata_json"}
    if metadata["schema"] != SCHEMA or tuple(metadata["state_fields"]) != STATE_FIELDS:
        raise ValueError("Unknown physical rollout archive schema.")
    result = _unpack_tree(metadata["result"], arrays)
    check_snapshot_contract(result)
    return result


def deterministic_source_checks():
    """AST-level contract fixture only; no import-time invocation/native call."""
    tree = ast.parse(inspect.getsource(branch_with_snapshots))
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
    called = {node.func.id for node in calls if isinstance(node.func, ast.Name)}
    if "progress" not in called or "values" not in called or "clone" in called:
        # All copying intentionally goes through the one deep packet helper.
        raise AssertionError("Unexpected thin-layer native/readout/copy call structure.")
    forbidden = {"precursor", "replay", "initialize", "assimilate", "setup"}
    if called & forbidden:
        raise AssertionError("Rollout must continue actual State, never recreate a prefix.")
    if len(STATE_FIELDS) != 16:
        raise AssertionError("The expected complete native State schema changed.")
    return {"passed": True, "state_fields": STATE_FIELDS,
            "scope": "Static AST/API fixture only; no reaction, property, data or RNG validation."}
