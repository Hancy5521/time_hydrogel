"""AAm/MBAA formation core for one complete process-design prototype.

Import starts no simulation, training, RNG stream or output.  Existing physical
sources stay unchanged.  The learned five-parameter psi is a *forward* mark law;
there is no importance ratio that silently reverts it to the synthetic teacher.
Inference is the classical 64-particle complete-State bootstrap filter (eta=1).
"""
from __future__ import annotations

from copy import deepcopy
from time import perf_counter

import numpy as np
from scipy.special import logsumexp

from aam_contact_environment_study import BASE_RATE
from aam_formation_actual_rollout import (
    SCHEMA as ROLLOUT_SCHEMA, _frozen_value, _packet_copy,
    packet_fingerprint, validate_rollout_inputs,
)
from aam_network_competition_study import fit_mark, reference
from aam_reaction_innovations import InnovationSource, propagate_interval
from aam_snapshot_process import PORTS, clone, initialize, likelihood, observe, progress, values
from aam_spatial_motion import Motion


SCHEMA = "aam-mbaa-complete-formation-core-v1"
PRODUCTION_PARTICLES = 64
PREFIX_CLOCK = (0., .65, 1.35, 2.05)
COMMON_STOPS = (2.30, 3.00)
FUTURE_TIMES = (3.25, 4.85, 7.20)
ACTIONS = tuple((time, fraction) for time in COMMON_STOPS for fraction in (0., .5, 1.))
RESAMPLE_FRACTION = .5


def make_config():
    """Pure fixed physical configuration; namespace/data roles belong to driver."""
    teacher, scale = reference("allocation")
    if scale != 1.:
        raise ValueError("This prototype freezes the physical reaction clock at one.")
    return {
        "schema": SCHEMA,
        "recipe": {"aam": 64, "mbaa": 14, "initial_mbaa": 8, "feed_time": 20.,
                   "initiators": 9, "horizon": 20.},
        "motion": vars(Motion(step=.025)), "corridor": .82,
        "rates": deepcopy(BASE_RATE), "factors": teacher.tolist(),
        "prefix_clock": list(PREFIX_CLOCK), "common_stops": list(COMMON_STOPS),
        "future_times": list(FUTURE_TIMES), "actions": [list(action) for action in ACTIONS],
        "noise": [.08, .10, .08, .10], "particles": PRODUCTION_PARTICLES,
        "clock_factor": 1.,
        "prior": "Unchanged snapshot initialization:6..12 initiators,Beta(.65,.65) initial MBAA x imbalance.",
        "scope": "Synthetic effective AAm/MBAA chemistry; no measured rate or real time/modulus calibration.",
    }


def _uint32(seed, margin=0):
    if (isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer))
            or seed < 0 or seed+margin >= 2**32):
        raise ValueError("Use an explicitly registered uint32 stream with sufficient suffix room.")
    return int(seed)


def _physical_config(config, factors):
    """No caller configuration mutation; learned psi must be supplied explicitly."""
    factors = np.asarray(factors, dtype=np.float64)
    if factors.shape != (3, 5) or not np.isfinite(factors).all() or np.any(factors <= 0.):
        raise ValueError("Forward psi requires finite positive3x5 mark factors.")
    if config.get("clock_factor", 1.) != 1.:
        raise ValueError("No fitted or hidden change to the physical clock is allowed.")
    if (tuple(config["prefix_clock"]) != PREFIX_CLOCK
            or tuple(config["future_times"]) != FUTURE_TIMES
            or tuple(config["common_stops"]) != COMMON_STOPS):
        raise ValueError("This version uses the fixed prefix, future and common-stop clocks.")
    recipe = config["recipe"]
    if (recipe["aam"], recipe["mbaa"], recipe["initial_mbaa"]) != (64, 14, 8):
        raise ValueError("This complete prototype uses64AAm,14MBAA,8initialMBAA.")
    if config["motion"]["step"] != .025:
        raise ValueError("The fixed Brownian step is.025.")
    actual = deepcopy(config)
    actual["factors"] = factors.tolist()
    return actual


def future_clock(config, age):
    """All actions share both feed stops; one native call per requested age."""
    if age not in config["future_times"]:
        raise ValueError("Use a declared formation age.")
    stops = [time for time in (*config["common_stops"], *config["future_times"]) if time <= age]
    clock = np.asarray([*config["prefix_clock"], *sorted(set(stops))], dtype=np.float64)
    if np.any(np.diff(clock) <= 0.) or clock[-1] != age:
        raise ValueError("Invalid common numerical stopping grid.")
    return clock


def _cost():
    return {"native_segment_attempted": 0, "native_segment_completed": 0,
            "native_segment_failed": 0, "native_seconds": 0.,
            "teacher_record_calls": 0, "brownian_slabs_recorded": 0,
            "q_read_seconds": 0., "property_seconds": 0.}


def _advance(packet, clock, seed, record_teacher, cost, *, record_control=False):
    """One native interval: ordinary evolve or identity legacy annotation, never both."""
    seed = _uint32(seed)
    started = perf_counter()
    cost["native_segment_attempted"] += 1
    before_position = packet[0].position.copy()
    before_counts = packet[0].counts.copy()
    control_before = _packet_copy(packet, readonly=True) if record_control else None
    try:
        if record_teacher:
            source = InnovationSource("legacy-record", seed=seed)
            raw = propagate_interval(packet, clock, source, event_proposal=None,
                                     clock_factor=1., record_states=False)
            if raw["log_target_over_proposal"] != 0.:
                raise ValueError("Teacher annotation is exactly proposal=target, with no RN correction.")
            cost["teacher_record_calls"] += 1
            cost["brownian_slabs_recorded"] += int(raw["slabs"])
            marks = extract_teacher_marks([{"raw": raw}])
            record = {"schema": "compact-native-pendant-risk-record-v1",
                      "seed": seed, "clock": np.asarray(clock).copy(), **marks,
                      "position_before": before_position, "counts_before": before_counts,
                      "position_after": packet[0].position.copy(), "counts_after": packet[0].counts.copy(),
                      "spans": len(raw["spans"]), "censored_spans": sum(s["censored"] for s in raw["spans"]),
                      "zero_target_hazard_spans": sum(s["target_A"] == 0. for s in raw["spans"]),
                      "brownian_slabs": int(raw["slabs"]),
                      "integrated_target_intensity": raw["integrated_target_intensity"],
                      "target_primitive_increment": raw["target_primitive_increment"],
                      "log_target_over_proposal": raw["log_target_over_proposal"],
                      "scope": "Risk/actualmark only; full native events and positions live in saved actual State frames."}
            if record_control:
                record["packet_before"] = control_before
                record["packet_after"] = _packet_copy(packet, readonly=True)
        else:
            progress(packet, clock, seed)
            record = None
        if packet[0].position[0] != clock[-1] or packet[0].counts[6]:
            raise ValueError("Native interval did not finish at the declared boundary.")
        cost["native_segment_completed"] += 1
        return record
    except Exception:
        cost["native_segment_failed"] += 1
        raise
    finally:
        cost["native_seconds"] += perf_counter()-started


def extract_teacher_marks(records):
    """Event-conditioned forward-psi sufficient data, plus explicit provenance.

    A risk cell sums *primitive* pendant candidate weights before teacher factors.
    During annotation the original full ledger supplies all legal candidates and
    no-event spans. Saved compact records retain exposure/count diagnostics and
    pendant risks; full native event histories remain in actual State. Only pendant marks enter the
    fixed five-parameter conditional-mark objective; total clock remains one.
    """
    opportunity, selected, provenance = [], [], []
    for interval, record in enumerate(records):
        if record.get("schema") == "compact-native-pendant-risk-record-v1":
            risks = np.asarray(record["teacher_opportunity"])
            labels = np.asarray(record["teacher_selection"])
            rows = np.asarray(record["teacher_event_rows"]).copy()
            if risks.shape != (len(labels), 3, 5) or labels.shape != (len(rows), 2) or rows.shape[1:] != (8,):
                raise ValueError("Malformed compact native teacher record.")
            rows[:, 0] = interval
            opportunity.extend(risks)
            selected.extend(labels)
            provenance.extend(rows)
            continue
        for span_index, span in enumerate(record["raw"]["spans"]):
            chosen = int(span["chosen"])
            if chosen < 0:
                continue
            rows = np.asarray(span["candidate_rows"])
            primitive = np.asarray(span["primitive_rate"])
            row = rows[chosen]
            kind, occupancy = int(row[4]), int(row[5])
            if kind < 0:
                continue
            if span["censored"] or kind not in (0, 1, 2) or not 0 <= occupancy < 5:
                raise ValueError("Invalid actual pendant-event category or support.")
            risk = np.zeros((3, 5), dtype=np.float64)
            for candidate, rate in zip(rows, primitive, strict=True):
                if int(candidate[4]) >= 0:
                    risk[int(candidate[4]), int(candidate[5])] += rate
            if (not np.isfinite(risk).all() or np.any(risk < 0.)
                    or primitive[chosen] <= 0. or risk[kind, occupancy] <= 0.):
                raise ValueError("A selected pendant mark needs positive original candidate support.")
            opportunity.append(risk)
            selected.append((kind, occupancy))
            provenance.append((interval, span_index, span["end"], row[0], row[1], row[2],
                               span["geometry_time"], primitive[chosen]))
    return {"teacher_opportunity": np.asarray(opportunity, dtype=np.float64).reshape(-1, 3, 5),
            "teacher_selection": np.asarray(selected, dtype=np.int64).reshape(-1, 2),
            "teacher_event_rows": np.asarray(provenance, dtype=np.float64).reshape(-1, 8)}


def formation_prefix(config, seed, factors, *, record_teacher=False, record_control=False, mother_id=None):
    """One actual nonzero-MBAA prefix, three native calls, all past States kept."""
    started = perf_counter()
    seed = _uint32(seed, 20)
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
            call_seed = seed+18+camera  # same seed+17+i convention as original precursor
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


def formation_rollout(actual_packet, action, config, seed, *, properties=None,
                      record_teacher=False, record_control=False, mother_id=None):
    """Continue actual MBAA chemistry on the common numerical stop grid.

    This is the same physical action/placement law as the frozen thin rollout,
    but each single future call includes both declared pulse stops up to its age.
    Exactly three native calls reach3.25,4.85,7.20; no quench, graft, prefix replay
    or extra progress call at2.30/3.00. All readouts use independent State copies.
    """
    started = perf_counter()
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
    rng = np.random.default_rng(int(seed)+51)
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
            call = {"interval": interval, "seed": int(seed)+1+interval,
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
            "prefix_clock": prefix, "future_times": future, "common_stops": np.asarray(COMMON_STOPS),
            "mother_packet_sha256": original_hash,
            "original_packet": _packet_copy(actual_packet, readonly=True),
            "branch_start_packet": branch_start, "packet": _packet_copy(packet, readonly=True),
            "snapshots": snapshots, "q": np.stack(q_rows) if q_rows else np.empty((0, 3)),
            "cost": cost, "native_calls": calls, "teacher_records": records,
            **extract_teacher_marks(records), "error": error,
            "elapsed_seconds": perf_counter()-started,
            "scope": "Actual nonzeroMBAA continuation; common numerical feed stops, learned psi not RN-reverted."}


def fit_psi_lineage(opportunity, selection, *, training_mother_ids=None):
    """One fixed five-parameter forward fit; all-one fallback on numeric failure.

    Driver supplies only64new training mothers, common prefix once and exactly
    action0/suffix0 continuation once per mother. Event-weighted original
    conditional mark likelihood/regularizer/bounds are unchanged. No validation
    selection, true teacher factor substitution, global clock fit or RN reversal.
    """
    opportunity = np.asarray(opportunity, dtype=np.float64)
    selection = np.asarray(selection, dtype=np.int64)
    if (opportunity.ndim != 3 or opportunity.shape[1:] != (3, 5)
            or selection.shape != (len(opportunity), 2) or not np.isfinite(opportunity).all()
            or np.any(opportunity < 0.)):
        raise ValueError("Malformed forward-psi training data, not a numerical-fit fallback case.")
    mothers = None if training_mother_ids is None else list(training_mother_ids)
    if mothers is not None and (len(mothers) != 64 or len(set(mothers)) != 64):
        raise ValueError("Exactly64unique training mother IDs are required.")
    if len(selection) and (np.any(selection < 0) or np.any(selection[:, 0] >= 3)
                           or np.any(selection[:, 1] >= 5)):
        raise ValueError("Invalid pendant selection labels.")
    if len(selection) and np.any(opportunity[np.arange(len(selection)), selection[:, 0], selection[:, 1]] <= 0.):
        raise ValueError("Selected pendant marks must have original support.")
    started, error = perf_counter(), None
    try:
        if not len(selection):
            raise ArithmeticError("No observed pendant marks to fit.")
        factors, coefficients = fit_mark(opportunity, selection, "lineage")
        if (factors.shape != (3, 5) or coefficients.shape != (5,)
                or not np.isfinite(factors).all() or np.any(factors <= 0.)
                or not np.isfinite(coefficients).all()):
            raise ArithmeticError("Nonfinite or invalid fixed five-parameter fit.")
    except (RuntimeError, ArithmeticError, FloatingPointError) as exc:
        factors, coefficients = np.ones((3, 5)), np.zeros(5)
        error = {"type": type(exc).__name__, "message": str(exc)}
    return {"factors": factors, "coefficients": coefficients,
            "fallback": error is not None, "fit_status": "all_one_fallback" if error else "fit_complete",
            "error": error, "training_mother_ids": mothers, "pendant_events": len(selection),
            "clock_factor": 1., "method": "existing_fit_mark_lineage_five_parameters",
            "objective_unit": "pendant event, original conditional risk-set likelihood",
            "elapsed_seconds": perf_counter()-started,
            "scope": "Forward synthetic psi, not an observation-conditioned proposal phi or real chemistry calibration."}


def score_psi_marks(opportunity, selection, factors):
    """Pure event-risk score; no model fit or selected-case omission."""
    risk = np.asarray(opportunity, dtype=np.float64)
    selected = np.asarray(selection, dtype=np.int64)
    factors = np.asarray(factors, dtype=np.float64)
    if risk.shape != (len(selected), 3, 5) or selected.shape != (len(risk), 2) or factors.shape != (3, 5):
        raise ValueError("Unexpected risk, label or factor shape.")
    weighted = risk*factors
    total = weighted.sum(axis=(1, 2))
    if not len(risk):
        return {"events": 0, "selected_cell_log_probability": np.empty(0), "mean_nll": None}
    probability = weighted[np.arange(len(risk)), selected[:, 0], selected[:, 1]]/total
    if not np.isfinite(probability).all() or np.any(probability <= 0.):
        raise ValueError("Every actual held pendant event needs positive predicted support.")
    ll = np.log(probability)
    return {"events": len(risk), "selected_cell_log_probability": ll, "mean_nll": float(-ll.mean())}


def _clone_particle(item):
    return {"packet": _packet_copy(item["packet"]),
            "initial_packet": _packet_copy(item["initial_packet"]),
            "frames": {time: clone(state) for time, state in item["frames"].items()},
            "initial_seed": item["initial_seed"], "interval_seeds": item["interval_seeds"].copy(),
            "camera": item["camera"]}


def _update(weights, log_likelihood):
    finite = np.isfinite(log_likelihood)
    support = float(weights[finite].sum())
    if np.isnan(log_likelihood).any() or np.isposinf(log_likelihood).any():
        raise ValueError("Gaussian log likelihood may be finite or minus infinity only.")
    if support == 0.:
        return {"valid": False, "weights": np.zeros_like(weights), "ESS": 0., "max_weight": 0.,
                "old_weight_support_mass": 0., "finite_fraction": float(finite.mean()), "log_normalizer": None}
    logw = np.full_like(weights, -np.inf)
    positive = weights > 0.
    logw[positive] = np.log(weights[positive])+log_likelihood[positive]
    z = float(logsumexp(logw))
    out = np.exp(logw-z)
    out /= out.sum()
    return {"valid": True, "weights": out, "ESS": float(1./(out@out)),
            "max_weight": float(out.max()), "old_weight_support_mass": support,
            "finite_fraction": float(finite.mean()), "log_normalizer": z}


def _systematic(weights, uniform):
    cumulative = np.cumsum(weights)
    cumulative[-1] = 1.
    locations = np.minimum((uniform+np.arange(len(weights)))/len(weights), np.nextafter(1., 0.))
    return np.searchsorted(cumulative, locations, side="right")


def filter_formation(config, measured, seed, factors, *, particles=PRODUCTION_PARTICLES, mother_id=None):
    """Classical eta=1 bootstrap filtering, complete physical history retained.

    Resample only after camera0/1 if ordinary ESS<.5N. Current parameters, initial
    packet, all consumed streams and all past States follow the joint ancestor.
    Unconsumed interval streams follow the preregistered base+5000*step+slot
    addresses. Initial streams are base+20*slot; the controller is base+16000.
    No terminal resampling or equal12 compression.
    """
    started = perf_counter()
    seed = _uint32(seed)
    if particles != PRODUCTION_PARTICLES:
        raise ValueError("This registered complete prototype uses exactly64particles.")
    measured = np.asarray(measured, dtype=np.float64)
    noise = np.asarray(config["noise"], dtype=np.float64)
    if (measured.shape != (3, 4) or not np.isfinite(measured).all()
            or noise.shape != (4,) or not np.isfinite(noise).all() or np.any(noise <= 0.)):
        raise ValueError("Use only the original3x4noisy local readings and positive original noise.")
    actual_config = _physical_config(config, factors)
    _uint32(seed, 16000)
    rng = np.random.default_rng(seed+16000)  # independent named controller, not particle0 initialization RNG
    initialization_seeds = np.asarray(seed+20*np.arange(particles), dtype=np.uint32)
    population = []
    for initial_seed in initialization_seeds:
        packet = initialize(actual_config, int(initial_seed))
        population.append({"packet": packet, "initial_packet": _packet_copy(packet),
                           "frames": {0.: clone(packet[0])}, "initial_seed": int(initial_seed),
                           "interval_seeds": np.full(3, -1, dtype=np.int64), "camera": -1})
    cost = _cost()
    weights = np.full(particles, 1./particles)
    ancestry = np.full((particles, 3), -1, dtype=np.int64)
    ancestry[:, 0] = np.arange(particles)
    log_values = np.full((particles, 3), np.nan)
    ledger, error, log_evidence = [], None, 0.
    for camera in range(3):
        incoming = weights.copy()
        raw = np.full((particles, 4), np.nan)
        advanced = np.zeros(particles, dtype=bool)
        seeds = np.asarray(seed+5000*(camera+1)+np.arange(particles), dtype=np.uint32)
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
            # Identical original camera likelihood, omitting only the common
            # Gaussian normalizer (which cancels from all posterior weights).
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
            log_evidence += updated["log_normalizer"]
            if camera < 2:
                ancestry[:, camera+1] = np.arange(particles)
            resampled = bool(camera < 2 and updated["ESS"] < RESAMPLE_FRACTION*particles)
            u = float(rng.random()) if resampled else None
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
                               "weighted_ancestral_ESS": float(1./(mass@mass)) if mass.sum() > 0. else None}
    return {"schema": SCHEMA, "status": "complete" if complete else "failed", "valid": complete,
            "mother_id": mother_id, "seed": seed, "particles": exported,
            "packets": [item["packet"] for item in exported], "cached_results": exported,
            "weights": weights.copy(),
            "log_weights": np.log(weights, where=weights > 0., out=np.full_like(weights, -np.inf)),
            "log_likelihoods": log_values,
            "initialization_seeds": initialization_seeds, "ancestry_ids": ancestry,
            "initial_ancestor": ancestry[:, 0].copy(), "ancestral_diversity": diversity,
            "ledger": ledger, "cost": cost, "error": error,
            "log_evidence_up_to_common_Gaussian_constants": log_evidence,
            "elapsed_seconds": perf_counter()-started,
            "scope": "Standard64-State bootstrap,eta1; all native final weights/histories retained, learned forwardpsi unchanged."}


def check_teacher_annotation_equivalence(records):
    """Six ordinary native calls from six already saved annotated before packets.

    Root invokes only inside the full study, after first train mother raw storage.
    Pass prefix.teacher_records + action0/suffix0.teacher_records, created with
    record_control=True. No initial state or prefix is regenerated. Each actual
    before packet is independently cloned, advanced once under the original
    progress/evolve, and checked against its corresponding annotated endpoint.
    """
    if len(records) != 6 or any("packet_before" not in r or "packet_after" not in r for r in records):
        raise ValueError("Use exactly the six previously saved first-mother control records.")
    started = perf_counter()
    attempted = completed = comparisons = 0
    errors, intervals = [], []
    for interval, record in enumerate(records):
        packet = _packet_copy(record["packet_before"])
        attempted += 1
        interval_started = perf_counter()
        report = {"interval": interval, "seed": int(record["seed"]), "completed": False}
        try:
            progress(packet, record["clock"], int(record["seed"]))
            completed += 1
            report["completed"] = True
            expected = record["packet_after"]
            fields = (*packet[0]._fields, "material", "geometry", "rates", "factors")
            for name, actual, target in zip(fields, (*packet[0], *packet[1:]),
                                             (*expected[0], *expected[1:]), strict=True):
                comparisons += 1
                if (actual.dtype != target.dtype or actual.shape != target.shape
                        or actual.tobytes() != target.tobytes()):
                    errors.append({"interval": interval, "field": name})
        except Exception as exc:
            errors.append({"interval": interval, "type": type(exc).__name__, "message": str(exc)})
        report["seconds"] = perf_counter()-interval_started
        intervals.append(report)
    return {"passed": not errors, "array_comparisons": comparisons, "mismatches": errors,
            "extra_native_segment_attempted": attempted, "extra_native_segment_completed": completed,
            "intervals": intervals, "elapsed_seconds": perf_counter()-started,
            "scope": "Original evolve vs saved MBAA identity annotation, six already saved before packets, no prefix regeneration."}

def deterministic_contract_checks(config):
    """Pure fixed-grid/config check; not a simulation or numerical validation."""
    checked = _physical_config(config, np.ones((3, 5)))
    expected = ([0., .65, 1.35, 2.05, 2.3, 3., 3.25],
                [0., .65, 1.35, 2.05, 2.3, 3., 3.25, 4.85],
                [0., .65, 1.35, 2.05, 2.3, 3., 3.25, 4.85, 7.2])
    if any(not np.array_equal(future_clock(checked, age), grid)
           for age, grid in zip(FUTURE_TIMES, expected, strict=True)):
        raise AssertionError("Common numerical grid changed.")
    return {"passed": True, "future_native_calls": 3, "filter_native_calls": 3*PRODUCTION_PARTICLES,
            "psi_parameters": 5, "no_terminal_resampling": True,
            "scope": "Pure declared config/grid contract; zero native calls or fits."}
