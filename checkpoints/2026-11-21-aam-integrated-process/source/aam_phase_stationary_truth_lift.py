"""Zero-propagation stationary initializer from a frozen generating trajectory.

This module is for an oracle audit after blind predictions have been frozen.
For original base innovation U with prior mu and a finite discrete uniform
phase J, the physical innovations are tau_J(U). Ideal translation invariance
gives mu(tau_J U)=mu(U). If U* generated the original noisy cameras, then
(tau_-J(U*), J), with fresh independent uniform J, has exactly the augmented
posterior law given those cameras. Discard J to initialize collapsed PGAS.

An archived path is translated to the same canonical image gauge as
build_packet. Archived hazard/mark receipts remain source provenance from the
physical generator, rather than being presented as a new simulator replay.
This argument uses ideal torus arithmetic. Physical floating-point ties and
near-threshold branch differences can prevent byte-identical continuation.
"""

import hashlib
from fractions import Fraction
from operator import index

import numpy as np

from aam_first_interval_paired_fork import MARK_CELLS, SCHEMA as TAPE_SCHEMA
from aam_formation_actual_rollout import _packet_copy, packet_fingerprint
from aam_formation_cycle_data import check_actual_state
from aam_formation_local_initial import build_packet, clone_initial, draw_initial, validate_initial
from aam_original_tape_pgas import SCHEMA as RAW_SCHEMA, _camera_contract
from aam_phase_marginal_tape_pgas import PHASES, phase_offsets, phase_reference
from aam_snapshot_process import observe
from run_aam_mark_candidate_filter import camera_likelihood
from run_aam_suffix_group_smoother import counts, rng, tape


SCHEMA = "aam-stationary-archived-truth-shared-phase-v1"
_TAPE_FIELDS = ("brownian", "waiting", "mark")


def _tape_hash(table):
    digest = hashlib.sha256()
    for name in _TAPE_FIELDS:
        a = np.ascontiguousarray(table[name])
        digest.update(f"{name}:{a.dtype.str}:{a.shape}\n".encode("ascii"))
        digest.update(a.tobytes())
    return digest.hexdigest()


def _verify_consumption(table, receipt):
    """Check every consumed wait/mark against the independent original RNG address."""
    reported = receipt["tape"]
    if (receipt["native_calls"] != 1 or receipt["log_ratio"] != 0.
            or reported["schema"] != TAPE_SCHEMA or reported["forced_mark_consumed"]):
        raise ValueError("The archive must come from one original unforced native interval.")
    used = np.zeros(len(table["waiting"]), dtype=np.int64)
    marks = 0
    for span in receipt["spans"]:
        slab, attempt = index(span["slab"]), index(span["attempt"])
        if not 0 <= slab < len(used) or attempt != used[slab]:
            raise ValueError("Native waiting trace disagrees with its original table addresses.")
        waited = span["exponential"]
        if waited is None:
            if span["target_A"] != 0. or not span["censored"]:
                raise ValueError("Only a zero-hazard censoring can leave a wait unconsumed.")
        else:
            if not 0 <= attempt < table["waiting"].shape[1] or table["waiting"][slab, attempt] != waited:
                raise ValueError("Archived wait is not from the registered independent RNG table.")
            used[slab] += 1
        if not span["censored"]:
            if (waited is None or span["mark_tilt"] is not None
                    or int(table["mark"][slab, attempt]) / MARK_CELLS != span["uniform"]):
                raise ValueError("Archived reaction mark is not the original unforced RNG cell.")
            marks += 1
    np.testing.assert_array_equal(used, reported["waits_per_slab"])
    return int(used.sum()), marks


def _wrapped_shift(y, displacement, box):
    shifted = np.remainder(y + displacement, box)
    # Remainder may round to box for a negative input infinitesimally below
    # zero. Keep the point on the correct side of the declared half-open box.
    at_upper = shifted == box
    shifted[at_upper] = np.nextafter(box, 0.)
    return shifted, int(np.count_nonzero(at_upper))


def _canonical_packet(packet, displacement, initial_images):
    """Translate one archived packet, conjugating all stored lattice gauges."""
    new = _packet_copy(packet)
    state = new[0]
    box = float(new[2][2])
    state.points[:, 1], seam_count = _wrapped_shift(state.points[:, 1], displacement, box)
    state.unwrapped[:, 1] += displacement - box * initial_images
    if state.position[0] == 0.:
        state.unwrapped[:, 1] = state.points[:, 1]
    n_events, n_edges = map(int, state.counts[:2])
    source, target = state.edges[:n_edges].T
    image_difference = initial_images[target] - initial_images[source]
    state.lifts[:n_edges, 1] += image_difference
    active = state.events[:n_events]
    linked = (active[:, 2] >= 0.) & (active[:, 1] != 8.)
    if np.count_nonzero(linked) != n_edges:
        raise ValueError("The image correction must preserve the complete event-to-edge ledger.")
    active[linked, 11] += image_difference
    state.periodic_shift[:, 1] += initial_images - initial_images[state.periodic_parent]
    return new, seam_count


def inverse_truth_reference(cfg, measured, raw, unit, phase_index):
    """Validate an addressed generating truth and return a collapsed reference.

    The caller verifies the external archive receipt and independently samples
    the audit-only phase_index uniformly per branch. The returned standard
    reference includes a truth_lift_receipt, containing no new native calls.
    The production sweep leaves that receipt to the outer audit run to archive.
    """
    unit, j = index(unit), index(phase_index)
    if unit < 0 or not 0 <= j < PHASES:
        raise ValueError("Use a registered generating mother and a legal uniform phase index.")
    y, sigma = _camera_contract(cfg, measured)
    packets = tuple(raw["packets"])
    intervals = tuple(raw["intervals"])
    ages = cfg["prefix_clock"]
    if len(ages) != 4 or len(packets) != 4 or len(intervals) != 3 or raw["calls"] != counts(3):
        raise ValueError("Use the authentic four native generating frames and three complete receipts.")
    validate_initial(cfg, raw["initial"])
    redrawn = draw_initial(cfg, rng("data_initial", unit))
    if packet_fingerprint(build_packet(cfg, redrawn)) != packet_fingerprint(packets[0]):
        raise ValueError("The initial packet is not from the specified original role address.")
    if (raw["initial"]["I"] != redrawn["I"]
            or any(not np.array_equal(raw["initial"][f], redrawn[f]) for f in ("left", "normal_x", "points"))):
        raise ValueError("Archived initialization preimages differ from their role address.")
    box = float(packets[0][2][2])
    if any(p[2][2] != box or float(p[1][3]) <= ages[-1] for p in packets):
        raise ValueError("The original three-camera phase lift only applies to the registered no-feed prefix.")
    for age, packet in zip(ages, packets, strict=True):
        check_actual_state(packet[0], expected_age=age)
    clean = np.stack([observe(packet) for packet in packets[1:]])
    np.testing.assert_array_equal(clean, raw["clean"])
    expected_y = clean + sigma * rng("measurement", unit).standard_normal((3, 4))
    np.testing.assert_array_equal(y, expected_y)
    tables = tuple(tape(cfg, "data_tape", unit, c) for c in range(3))
    consumed = [_verify_consumption(t, r) for t, r in zip(tables, intervals, strict=True)]
    for table in tables:
        for values in table.values():
            values.flags.writeable = False
    displacement = -float(phase_offsets(packets[0])[j])
    staged = clone_initial(raw["initial"])
    images = np.floor((staged["points"][:, 1] + displacement) / box).astype(np.int64)
    staged["points"][:, 1], seam_count = _wrapped_shift(staged["points"][:, 1], displacement, box)
    validate_initial(cfg, staged)
    transformed = []
    for age, packet in zip(ages, packets, strict=True):
        if displacement == 0.:
            new = _packet_copy(packet)
        else:
            new, seams = _canonical_packet(packet, displacement, images)
            seam_count += seams
        check_actual_state(new[0], expected_age=age)
        transformed.append(_packet_copy(new, readonly=True))
    if packet_fingerprint(build_packet(cfg, staged)) != packet_fingerprint(transformed[0]):
        raise AssertionError("Inverse translation must use the canonical initial and lattice-image gauge.")
    revised_clean = np.stack([observe(packet) for packet in transformed[1:]])
    base = {"schema": RAW_SCHEMA, "initial": staged, "packets": tuple(transformed), "tapes": tables,
            "intervals": intervals, "clean": revised_clean,
            "logl": camera_likelihood(revised_clean, y, sigma)}
    ref = phase_reference(cfg, y, base)
    # Bring the reference back to the generating physical orientation as an
    # observational check independent of any native transition replay.
    np.testing.assert_allclose(ref["clean_orbit"][:, j], clean, rtol=0., atol=2e-13)
    ref["truth_lift_receipt"] = {
        "schema": SCHEMA, "unit": unit, "inverse_phase_index": j,
        "inverse_displacement": displacement, "initial_y_image_numbers": images,
        "seam_rounding_clamps": seam_count, "native_calls": 0,
        "original_tape_hashes": [_tape_hash(t) for t in tables],
        "waits_and_marks_verified_by_interval": consumed,
        "original_packet_fingerprints": [packet_fingerprint(p) for p in packets],
        "canonical_inverse_packet_fingerprints": [packet_fingerprint(p) for p in transformed],
        "original_interval_receipts_unchanged": True,
        "validity": "ideal native torus symmetry, up to numerical rounding; authenticated archive required",
    }
    return ref


def fixtures():
    """Exact finite posterior lift and a virtual molecule lattice-gauge cycle."""
    likelihood = tuple(map(Fraction, (2, 9, 4, 1)))
    total = sum(likelihood)
    for physical in range(4):
        for phase in (0, 1):
            base = physical ^ phase
            mass_from_archive = likelihood[physical] / (2 * total)
            mass_from_augmented_target = likelihood[base ^ phase] / (2 * total)
            assert mass_from_archive == mass_from_augmented_target
    from aam_edge_fork import State

    n, box = 4, 6.
    points = np.zeros((n, 3))
    points[:, 1] = [.2, 2.1, 4., 5.8]
    edges = np.array([(0, 1), (1, 2), (2, 3), (3, 0)])
    u, v = edges.T
    lifts = -np.rint((points[v]-points[u])/box).astype(np.int64)
    events = np.zeros((n+1, 13))
    events[:n, 1] = 3.
    events[:n, 2:4] = edges * 2
    events[:n, 10:13] = lifts
    events[n, 1:4] = (8., 0., 4.)
    graph_parent = np.array([0, 0, 1, 2])
    state = State(points, points.copy(), np.ones(n, bool), np.zeros(2*n, bool),
                  np.zeros(2*n, bool), np.zeros(2*n, np.int64), np.zeros(n, np.int64),
                  edges, lifts, events, graph_parent, np.zeros((n, 3), np.int64),
                  np.zeros(n, np.int64), np.array([False, True, False]),
                  np.array([n+1, n, 0, 0, 0, 0, 0]), np.array([1., 0., 0., 0.]))
    delta = -.75
    images = np.floor((points[:, 1] + delta)/box).astype(np.int64)
    translated, seams = _canonical_packet((state, np.zeros(4), np.array([0., 0., box]),
                                          np.zeros(1), np.zeros(1)), delta, images)
    new = translated[0]
    relift = -np.rint((new.unwrapped[v]-new.unwrapped[u])/box).astype(np.int64)
    np.testing.assert_array_equal(new.lifts, relift)
    np.testing.assert_array_equal(new.lifts.sum(axis=0), lifts.sum(axis=0))
    np.testing.assert_array_equal(new.events[:n, 10:13], new.lifts)
    np.testing.assert_array_equal(new.events[n], events[n])
    np.testing.assert_array_equal(new.periodic_shift[1:, 1], new.lifts[:3, 1])
    np.testing.assert_array_equal(new.winding, state.winding)
    assert seams == 0 and not np.shares_memory(new.events, events)
    return {"schema": SCHEMA, "native_calls": 0, "exact_finite_posterior_lift": True,
            "gauge_event_and_winding_fixture": True}


if __name__ == "__main__":
    print(fixtures())
