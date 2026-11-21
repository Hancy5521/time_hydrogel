"""Prospective V5 mu5 truth/training data and explicit query/reveal boundary.

Import performs no draw, State construction, native call, fit or output.  Only
the newly registered groups may call ``data_job``.  V1--V4 data/physics remain
unchanged; raw archives use the existing lossless numeric-tree schema.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from time import perf_counter

import numpy as np

from aam_formation_actual_rollout import _packet_copy, packet_fingerprint
from aam_formation_cycle_core import (
    SCHEMA as CORE_SCHEMA, _advance, _cost, _physical_config, extract_teacher_marks,
)
from aam_formation_cycle_data import (
    _branch_check, _numeric_load, _numeric_save, _read6, _stage, _teacher_arrays,
    _tree_load, check_actual_state, event, now, property_readout, read, sha, write,
)
from aam_formation_local_initial import (
    SCHEMA as INITIAL_SCHEMA, build_packet, clone_initial, draw_initial,
    initial_fingerprint, validate_initial,
)
from aam_formation_local_streams import (
    MASTER, NAMESPACE, future_streams, rng, seed_draws, stream_address,
)
from aam_formation_streamed_rollout import _explicit_seeds, formation_rollout_streamed
from aam_snapshot_process import clone, observe


SCHEMA = "aam-formation-explicit-initial-local-v5-data"
FREEZE_SCHEMA = "aam-formation-local-initial-cycle-v5-freeze-A"
GROUP_COUNTS = {"train": 64, "dev": 4, "held": 24}
FREEZE_NAME = "ALL_PREDICTIONS_AND_SELECTIONS_FROZEN.json"


def formation_prefix_local(config, initial, factors, *, native_seeds,
                           record_teacher=False, mother_id=None):
    """Build one actual mu5 initial State and advance three retained intervals.

    The explicit initial variables, all four checkpoint packets and three clean
    cameras are retained.  No legacy initialize/setup or fabricated u0 is used.
    Teacher annotation uses the existing single-call native annotation bridge;
    no extra control or replay is invoked here.
    """
    started = perf_counter()
    native_seeds = _explicit_seeds(native_seeds)
    actual_config = _physical_config(config, factors)
    validate_initial(actual_config, initial)
    initialization_started = perf_counter()
    packet = build_packet(actual_config, initial, factors=factors)
    initialization_seconds = perf_counter()-initialization_started
    initial_packet = _packet_copy(packet, readonly=True)
    checkpoints = [initial_packet]
    frames = {0.: clone(packet[0])}
    cost, records, clean, calls = _cost(), [], [], []
    cost.update(actual_initializations=1, initialization_seconds=initialization_seconds)
    error = None
    try:
        for camera in range(3):
            clock = np.asarray(config["prefix_clock"][:camera+2], dtype=np.float64)
            call = {"camera": camera, "seed": int(native_seeds[camera]),
                    "clock": clock.copy(), "completed": False,
                    "position_before": packet[0].position.copy()}
            calls.append(call)
            try:
                record = _advance(packet, clock, call["seed"], record_teacher, cost)
                call["completed"] = True
            finally:
                call["position_after"] = packet[0].position.copy()
            if record is not None:
                records.append(record)
            checkpoints.append(_packet_copy(packet, readonly=True))
            frames[float(clock[-1])] = clone(packet[0])
            clean.append(np.asarray(observe(_packet_copy(packet)), dtype=np.float64))
    except Exception as exc:
        error = {"type": type(exc).__name__, "message": str(exc), "camera": len(calls)-1}
    for state in frames.values():
        for array in state:
            array.flags.writeable = False
    return {"schema": CORE_SCHEMA, "data_schema": SCHEMA,
            "status": "complete" if error is None else "failed", "valid": error is None,
            "mother_id": mother_id, "initial_variables": clone_initial(initial),
            "initial_prior_schema": INITIAL_SCHEMA,
            "initial_fingerprint": initial_fingerprint(initial),
            "explicit_native_seeds": native_seeds.copy(),
            "initial_packet": initial_packet, "checkpoint_packets": checkpoints,
            "packet": _packet_copy(packet, readonly=True), "frames": frames,
            "clean": np.asarray(clean, dtype=np.float64).reshape(-1, 4),
            "factors": np.asarray(factors, dtype=np.float64).copy(),
            "cost": cost, "native_calls": calls, "teacher_records": records,
            **extract_teacher_marks(records), "error": error,
            "elapsed_seconds": perf_counter()-started,
            "scope": "Actual mu5 effective AAm/MBAA prefix; not measured kinetics or literal V4 finite-u0 law."}


def _context(run, group, mother, *, create=False, verify_registered_files=True):
    run = Path(run).resolve()
    if (group not in GROUP_COUNTS or isinstance(mother, (bool, np.bool_))
            or not isinstance(mother, (int, np.integer)) or not 0 <= mother < GROUP_COUNTS[group]):
        raise ValueError("Use a registered train/dev/held mother index.")
    mother = int(mother)
    registration = run/"REGISTERED.json"
    plan = read(registration)
    if (plan.get("namespace") != NAMESPACE or plan.get("master_seed") != MASTER
            or plan.get("training_mothers") != 64 or plan.get("development_mothers") != 4
            or plan.get("queries") != 24 or plan.get("training_suffixes") != 2
            or plan.get("reference_suffixes") != 4):
        raise ValueError("Use the complete prospective V5 mu5 registration and fixed groups.")
    if verify_registered_files:
        for category in ("sources", "inputs"):
            for path, expected in plan[category].items():
                if sha(path) != expected:
                    raise ValueError("Registered "+category+" identity changed: "+path)
    folder = run/group/f"parent_{mother:03d}"
    if create:
        folder.mkdir(parents=True, exist_ok=True)
    binding = {"schema": SCHEMA, "namespace": NAMESPACE, "master_seed": MASTER,
               "initial_prior_schema": INITIAL_SCHEMA,
               "root_registration": str(registration), "root_registration_sha256": sha(registration),
               "role": group, "mother_id": mother,
               "initial_stream_address": list(stream_address(group, mother, "truth_initial")),
               "native_stream_address": list(stream_address(group, mother, "truth_native")),
               "assay_stream_address": list(stream_address(group, mother, "truth_assay"))}
    return run, plan, folder, binding


def _info(path, binding):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": sha(path), "bytes": path.stat().st_size,
            "actual_utc": now(), "binding": binding}


def _check_info(info, folder, binding, *, check_bytes=True):
    path = Path(info["path"]).resolve()
    if (path.parent != folder.resolve() or path.stat().st_size != info["bytes"]
            or info.get("binding") != binding):
        raise ValueError("Saved payload identity/path/size differs from its registered mother.")
    if check_bytes and sha(path) != info["sha256"]:
        raise ValueError("Saved payload changed: "+str(path))
    return path


def _completed(path, folder, binding):
    receipt = read(path)
    if receipt.get("status") != "complete" or receipt.get("binding") != binding:
        raise ValueError("Incomplete or mismatched completed mother receipt.")
    # Full raw bytes are audited before adoption; compact values/query are
    # freshly verified when loaded.  Resume additionally verifies each saved raw
    # that it actually decodes through _stage/_tree_load, never regenerates it.
    for info in receipt["files"].values():
        _check_info(info, folder, binding, check_bytes=Path(info["path"]).suffix != ".npz"
                    or Path(info["path"]).name in {"QUERY.npz", "TRAIN_VALUES.npz", "EVAL_VALUES.npz", "PSI_TRAIN.npz"})
    return receipt


@lru_cache(maxsize=4)
def _frozen_identity(path_string, expected_sha, root_sha):
    path = Path(path_string)
    if sha(path) != expected_sha:
        raise ValueError("Prediction freeze changed.")
    registration = path.parent/"REGISTERED.json"
    if sha(registration) != root_sha:
        raise ValueError("Global freeze no longer has its original root registration.")
    plan = read(registration)
    freeze = read(path)
    if (freeze.get("schema") != FREEZE_SCHEMA
            or freeze.get("prediction_and_selection_frozen") is not True
            or freeze.get("queries") != 24
            or freeze.get("raw_verification_contract") != "A+final_full_raw_audit"
            or freeze.get("root_registration_sha256") != root_sha
            or freeze.get("held_futures_revealed") is not False
            or not isinstance(freeze.get("all_forecast_files"), dict)
            or not freeze["all_forecast_files"] or not isinstance(freeze.get("sources"), dict)
            or freeze["sources"] != plan["sources"]
            or not isinstance(freeze.get("fit_completed_sha256"), str)
            or sha(path.parent/"FIT_COMPLETED.json") != freeze["fit_completed_sha256"]):
        raise ValueError("Incomplete V5 prediction/selection freeze A contract.")
    return {"path": str(path.resolve()), "sha256": expected_sha}


def verify_prediction_freeze(run, supplied, binding):
    """Check immutable global identity, not reread all raw payloads per mother."""
    run = Path(run).resolve()
    if (run/"PREDICTIONS_ACTIVE.lock").exists():
        raise RuntimeError("Cannot reveal while predictions are active.")
    if not isinstance(supplied, dict) or set(supplied) != {"path", "sha256"}:
        raise TypeError("Supply the actual completed prediction freeze path and SHA.")
    path = Path(supplied["path"]).resolve()
    if path != run/FREEZE_NAME:
        raise ValueError("Reference reveal must bind this run's global prediction freeze.")
    return dict(_frozen_identity(str(path), supplied["sha256"], binding["root_registration_sha256"]))


def _query(folder, binding, group, mother, prefix, cfg):
    metadata, target = folder/"QUERY.json", folder/"QUERY.npz"
    if metadata.exists():
        info = read(metadata)
        _check_info(info, folder, binding)
        return info
    if target.exists():
        raise RuntimeError("Unclosed QUERY publication; preserve without redrawing assay noise.")
    measured = np.asarray(prefix["clean"])+rng(group, mother, "truth_assay").normal(
        0., np.asarray(cfg["noise"]), size=(3, 4))
    info = _numeric_save(target, X=measured.reshape(12), measured=measured,
                         mother_id=np.asarray(mother), role=np.asarray(group))
    info.update(actual_utc=now(), binding=binding)
    write(metadata, info)
    return info


def _prefix(plan, folder, binding, group, mother, dispatched):
    cfg = plan["config"]
    prefix, raw = _stage(folder, "PREFIX_RAW", binding,
        lambda: formation_prefix_local(cfg, draw_initial(cfg, rng(group, mother, "truth_initial")),
            np.asarray(cfg["factors"]),
            native_seeds=seed_draws(group, mother, "truth_native", size=3),
            record_teacher=group == "train", mother_id=f"{group}:{mother}"), dispatched)
    if (not prefix["valid"] or prefix["status"] != "complete"
            or prefix["clean"].shape != (3, 4)
            or len(prefix["checkpoint_packets"]) != 4
            or prefix["cost"]["native_segment_attempted"] != 3
            or prefix["cost"]["native_segment_completed"] != 3):
        raise RuntimeError("Actual mu5 prefix failed; preserve output without replacement.")
    check_actual_state(prefix["packet"][0], expected_age=2.05)
    query = _query(folder, binding, group, mother, prefix, cfg)
    receipt_path = folder/"QUERY_COMPLETE.json"
    if receipt_path.exists():
        receipt = _completed(receipt_path, folder, binding)
        if receipt["files"]["prefix_raw"] != raw or receipt["files"]["query"] != query:
            raise ValueError("Saved query completion differs from original prefix payloads.")
    else:
        receipt = {"actual_utc": now(), "status": "complete", "binding": binding,
                   "role": group+"/prefixonly", "files": {"prefix_raw": raw, "query": query},
                   "native_cost": prefix["cost"], "dispatched_cost": list(dispatched),
                   "reference_future_generated": False, "observed_channels": 12,
                   "initial_prior_schema": INITIAL_SCHEMA}
        write(receipt_path, receipt)
    return prefix, receipt


def _full_values(run, plan, folder, binding, group, mother, prefix, query_receipt,
                 freeze, dispatched):
    from aam_formation_micro_history import extract_frames, extract_future

    cfg = plan["config"]
    packet = prefix["packet"]
    origin_hash = packet_fingerprint(packet)

    def cutoff_readout():
        output = property_readout(_packet_copy(packet, readonly=True), cfg)
        cost = _cost()
        cost["property_seconds"] = output["readout_seconds"]
        return {"schema": SCHEMA, "readout": output, "cost": cost,
                "actual_cutoff_fingerprint": origin_hash, "mechanical_snapshots": 1}

    # Retain explicit invalid readouts before checking them.  A failure cannot
    # silently cause another property solve on the next dispatch.
    cutoff_raw, cutoff_info = _stage(folder, "CUTOFF_PROPERTIES_RAW", binding,
                                     cutoff_readout, dispatched)
    prop = cutoff_raw["readout"]
    if not prop["valid"]:
        raise RuntimeError("Invalid actual cutoff mechanics; not a fabricated zero.")
    cutoff = _read6(_packet_copy(packet, readonly=True), prop)
    micro_early, micro_early_winding = extract_frames(prefix["frames"])
    repeats = plan["training_suffixes"] if group == "train" else plan["reference_suffixes"]
    shape = (repeats, 6, 3)
    future_values, g, raw_g = np.empty((*shape, 6)), np.empty((*shape, 5)), np.empty((*shape, 5))
    micro_future, future_winding = np.empty((*shape, 8)), np.empty((*shape, 3), dtype=np.bool_)
    refs, costs, first_branch = {}, [], None
    for repeat in range(repeats):
        streams = future_streams(group, mother, arm=-1, slot=0, repeat=repeat)
        for action_index, action in enumerate(cfg["actions"]):
            name = f"FUTURE_a{action_index:02d}_r{repeat:02d}"
            keep_teacher = group == "train" and repeat == 0 and action_index == 0
            result, info = _stage(folder, name, binding,
                lambda a=action, st=streams, keep=keep_teacher: formation_rollout_streamed(
                    packet, a, cfg, **st, properties=lambda p: property_readout(p, cfg),
                    record_teacher=keep, mother_id=f"{group}:{mother}"), dispatched)
            row, gv, gr, valid, errors = _branch_check(result, packet, action, cfg, streams["seed"])
            if not valid.all() or not np.isfinite(row).all():
                raise RuntimeError("Invalid actual future mechanics; retained raw is not replaced: "+str(errors))
            future_values[repeat, action_index], g[repeat, action_index], raw_g[repeat, action_index] = row, gv, gr
            micro_future[repeat, action_index], future_winding[repeat, action_index] = extract_future(result)
            refs[name] = info
            costs.append(result["cost"])
            if keep_teacher:
                first_branch = result
            if packet_fingerprint(packet) != origin_hash:
                raise AssertionError("A future intervention modified its actual common microscopic past.")
    query_info = query_receipt["files"]["query"]
    query = _numeric_load(query_info["path"], query_info["sha256"])
    target = np.concatenate((np.broadcast_to(cutoff, (repeats, 6)), future_values.reshape(repeats, 108)), axis=1)
    files = {**query_receipt["files"], "cutoff_property_raw": cutoff_info, **refs}
    if group == "train":
        teacher = _teacher_arrays(prefix, first_branch)
        psi_path = folder/"PSI_TRAIN.npz"
        if psi_path.exists():
            raise RuntimeError("Unclosed PSI training publication; preserve without automatic replacement.")
        psi_info = _numeric_save(psi_path, **teacher, mother_id=np.asarray(mother), role=np.asarray(group))
        psi_info.update(actual_utc=now(), binding=binding, training_allowed=True,
                        branch_action=0, branch_repeat=0, prefix_counted_once=True)
        write(folder/"PSI_TRAIN.json", psi_info)
        files["psi"] = psi_info
    value_name = "TRAIN_VALUES" if group == "train" else "EVAL_VALUES"
    target_path = folder/(value_name+".npz")
    if target_path.exists():
        raise RuntimeError("Unclosed target publication; preserve without replacement.")
    value_info = _numeric_save(target_path, X=query["X"], measured=query["measured"],
        cutoff_values=cutoff, future_values=future_values, target114=target,
        G_tau=g, G_tau_raw=raw_g, cutoff_G_tau=prop["dimensionless_finite_G_tau"],
        cutoff_G_tau_raw=prop["raw_dimensionless_finite_G_tau"],
        micro_early=micro_early, micro_future=micro_future,
        micro_early_winding=micro_early_winding, micro_future_winding=future_winding,
        mother_id=np.asarray(mother), role=np.asarray(group),
        source_actual_cutoff_fingerprint=np.asarray(origin_hash))
    value_info.update(actual_utc=now(), binding=binding)
    write(folder/(value_name+".json"), value_info)
    files["values"] = value_info
    if sum(c["native_segment_completed"] for c in costs) != 18*repeats:
        raise AssertionError("Six interventions per suffix must use exactly18native intervals.")
    return {"actual_utc": now(), "status": "complete", "binding": binding,
            "files": files, "prediction_freeze": freeze,
            "native_cost": prefix["cost"], "reference_cost": costs,
            "dispatched_cost": list(dispatched), "cutoff_property_seconds": prop["readout_seconds"],
            "unique_native_segment_attempted": 3+sum(c["native_segment_attempted"] for c in costs),
            "unique_native_segment_completed": 3+sum(c["native_segment_completed"] for c in costs),
            "actual_prefix_count": 1, "future_branches": 6*repeats,
            "mechanical_snapshots": 1+18*repeats, "physical_measurements": 0,
            "teacher_phase_rows": None if group != "train" else [
                len(prefix["teacher_opportunity"]), len(first_branch["teacher_opportunity"])],
            "initial_prior_schema": INITIAL_SCHEMA,
            "raw_verification_contract": "A+final_full_raw_audit"}


def data_job(job):
    """Exclusive one-mother source of new mu5 data; never retries failed physics."""
    group, mother, stage = job["group"], job["mother"], job["stage"]
    if (stage not in ("prefixonly", "full", "reveal")
            or (group == "dev" and stage != "prefixonly")
            or (group == "train" and stage == "reveal")
            or (group == "held" and stage == "full")):
        raise ValueError("Allowed stages are train prefix/full, dev prefix, held prefix/reveal.")
    run, plan, folder, binding = _context(job["run"], group, mother, create=True)
    freeze = verify_prediction_freeze(run, job.get("prediction_freeze"), binding) if stage == "reveal" else None
    if (folder/"FAILED.json").exists():
        raise RuntimeError("A failed mother is retained; no automatic retry or replacement.")
    complete_name = "QUERY_COMPLETE.json" if stage == "prefixonly" else (
        "TRAIN_COMPLETE.json" if group == "train" else "EVAL_COMPLETE.json")
    completed = folder/complete_name
    if completed.exists():
        result = _completed(completed, folder, binding)
        if stage == "reveal" and result.get("prediction_freeze") != freeze:
            raise ValueError("Completed reference belongs to a different prediction freeze.")
        return result
    lock = folder/"DATA_ACTIVE.lock"
    with lock.open("x", encoding="utf-8") as stream:
        stream.write(now()+"\n")
    started, dispatched = perf_counter(), []
    try:
        event(folder, "v5_mother_stage_started", group=group, mother=int(mother), stage=stage, binding=binding)
        if stage == "reveal":
            query_receipt = _completed(folder/"QUERY_COMPLETE.json", folder, binding)
            raw = query_receipt["files"]["prefix_raw"]
            prefix = _tree_load(raw["path"], binding, raw["sha256"])
            write(folder/"REVEAL_AUTHORIZATION.json", {"actual_utc": now(), "binding": binding,
                                                      "prediction_freeze": freeze})
        else:
            prefix, query_receipt = _prefix(plan, folder, binding, group, mother, dispatched)
        if stage == "prefixonly":
            return query_receipt
        result = _full_values(run, plan, folder, binding, group, mother, prefix, query_receipt,
                              freeze, dispatched)
        result["elapsed_seconds_this_dispatch"] = perf_counter()-started
        write(completed, result)
        write(folder/"MOTHER_RECEIPT.json", result)
        event(folder, "v5_mother_stage_complete", group=group, mother=int(mother), stage=stage,
              unique_native=result["unique_native_segment_completed"],
              mechanics=result["mechanical_snapshots"])
        return result
    except BaseException as exc:
        failure = {"actual_utc": now(), "binding": binding, "stage": stage,
                   "error_type": type(exc).__name__, "error": str(exc),
                   "new_stage_costs_this_dispatch": dispatched, "no_automatic_retry": True,
                   "elapsed_seconds_this_dispatch": perf_counter()-started}
        if not (folder/"FAILED.json").exists():
            write(folder/"FAILED.json", failure)
        event(folder, "v5_mother_stage_failed", **failure)
        raise
    finally:
        lock.unlink()


def _location(receipt_or_run, mother, group):
    if isinstance(receipt_or_run, dict):
        binding = receipt_or_run["binding"]
        return Path(binding["root_registration"]).parent/binding["role"]/f"parent_{binding['mother_id']:03d}"
    path = Path(receipt_or_run).resolve()
    if mother is not None:
        if (group not in GROUP_COUNTS or isinstance(mother, (bool, np.bool_))
                or not isinstance(mother, (int, np.integer)) or not 0 <= mother < GROUP_COUNTS[group]):
            raise ValueError("Use a registered mother index without implicit numeric coercion.")
        return path/group/f"parent_{int(mother):03d}"
    return path if path.is_dir() else path.parent


def _load_context(receipt_or_run, mother, group):
    folder = _location(receipt_or_run, mother, group)
    qr = read(folder/"QUERY_COMPLETE.json")
    binding = qr["binding"]
    run, _, canonical, actual = _context(Path(binding["root_registration"]).parent,
        binding["role"], binding["mother_id"], verify_registered_files=False)
    if canonical != folder or actual != binding:
        raise ValueError("Requested case does not have its exact V5 registration binding.")
    if qr.get("status") != "complete":
        raise ValueError("No completed original query is available.")
    # Observation loaders read binding/QUERY only, not raw States or any
    # registered training inputs.  Scientific source/input identity is checked
    # by the registered dispatcher/data jobs and final audit separately.
    return run, folder, binding, qr


def load_query(receipt_or_run, mother=None, *, group="held"):
    """Observation-only loader; never opens a prefix State or target payload."""
    _, folder, binding, receipt = _load_context(receipt_or_run, mother, group)
    info = receipt["files"]["query"]
    arrays = _numeric_load(_check_info(info, folder, binding), info["sha256"])
    if (set(arrays) != {"X", "measured", "mother_id", "role"}
            or arrays["X"].shape != (12,) or arrays["measured"].shape != (3, 4)
            or not np.array_equal(arrays["X"], arrays["measured"].reshape(12))
            or not np.isfinite(arrays["X"]).all()
            or int(arrays["mother_id"]) != binding["mother_id"]
            or str(arrays["role"]) != binding["role"]):
        raise ValueError("Use only the original12noisy observations from this mother.")
    return {"X": arrays["X"], "measured": arrays["measured"], "observed": arrays["measured"],
            "mother_id": binding["mother_id"], "role": binding["role"], "query_sha256": info["sha256"]}


def load_prefix(receipt_or_run, mother=None, *, group="train"):
    """Actual-prefix loader for registered training/development/audit callers."""
    _, _, binding, receipt = _load_context(receipt_or_run, mother, group)
    info = receipt["files"]["prefix_raw"]
    return _tree_load(info["path"], binding, info["sha256"])


def _load_values(receipt_or_run, mother, group, prediction_freeze=None):
    run, folder, binding, _ = _load_context(receipt_or_run, mother, group)
    if binding["role"] != group or group not in ("train", "held"):
        raise ValueError("Training and held outcomes may not be interchanged.")
    name = "TRAIN_COMPLETE.json" if group == "train" else "EVAL_COMPLETE.json"
    receipt = _completed(folder/name, folder, binding)
    if group == "held":
        freeze = verify_prediction_freeze(run, prediction_freeze, binding)
        if freeze != receipt["prediction_freeze"]:
            raise ValueError("Held target was revealed under a different global freeze.")
    info = receipt["files"]["values"]
    arrays = _numeric_load(info["path"], info["sha256"])
    if (arrays["X"].shape != (12,) or arrays["measured"].shape != (3, 4)
            or arrays["cutoff_values"].shape != (6,)
            or arrays["future_values"].shape != ((2 if group == "train" else 4), 6, 3, 6)
            or int(arrays["mother_id"]) != binding["mother_id"] or str(arrays["role"]) != group):
        raise ValueError("Wrong completed V5 target layout or mother identity.")
    if group == "train":
        psi = receipt["files"]["psi"]
        supervision = _numeric_load(psi["path"], psi["sha256"])
        for key in ("teacher_opportunity", "teacher_selection", "teacher_event_rows", "teacher_phase"):
            arrays[key] = supervision[key]
    arrays.update(observed=arrays["measured"], receipt=receipt,
                  mother_id=binding["mother_id"], role=group)
    return arrays


def load_training(receipt_or_run, mother=None):
    return _load_values(receipt_or_run, mother, "train")


def load_eval(receipt_or_run, mother=None, *, prediction_freeze=None):
    return _load_values(receipt_or_run, mother, "held", prediction_freeze)
