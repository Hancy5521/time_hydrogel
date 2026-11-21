"""Original-tape Particle Gibbs with a shared finite group of transverse phases.

For a native path z under the product prior, ideal law is invariant to T_s,
the simultaneous translation in y of every particle (including absent feed).
Augment z with s uniform on a finite subgroup of the periodic box. The model
evaluates observations on T_s z, so the marginal physical path still has the
original prior for every group size. Conditional inference sums out this one
phase, then samples it back once per trajectory. Floating-point native
trajectories are equivariant up to numerical roundoff, without a claim of
byte-identical replay at translated positions or matching seed addresses.
"""

from fractions import Fraction
from itertools import product
from math import prod
from operator import index

import numpy as np
from scipy.special import logsumexp

from aam_original_tape_pgas import (
    MODES, _camera_contract, _draw, _start, _table, _validate_reference as _validate_original,
    _weights,
)
from aam_snapshot_process import observe
from aam_transverse_orbit import camera_orbit, translate_packet
from run_aam_mark_candidate_filter import camera_likelihood
from run_aam_suffix_group_smoother import counts, native_step


SCHEMA = "aam-original-tape-shared-phase-pgas-v1"
PHASES = 16
_LOG_PHASES = np.log(PHASES)


def phase_offsets(packet):
    box = float(packet[2][2])
    if not np.isfinite(box) or box <= 0:
        raise ValueError("The original packet requires a finite positive periodic box.")
    return np.arange(PHASES, dtype=float) * (box / PHASES)


def _orbit(packet, y, noise):
    clean = camera_orbit(packet, phase_offsets(packet))
    clean[0] = observe(packet)
    logl = camera_likelihood(clean, y, noise)
    if clean.shape != (PHASES, 4) or logl.shape != (PHASES,) or not np.isfinite(logl).all():
        raise ValueError("Every phase needs the same four finite original camera channels.")
    return clean, logl


def _marginal(phase_logl):
    return logsumexp(phase_logl, axis=-1) - _LOG_PHASES


def _start_phase(cfg, rng, source):
    return {**_start(cfg, rng, source), "schema": SCHEMA, "clean_orbit": (),
            "phase_logl": (), "cumulative_phase_logl": (), "marginal_logl": ()}


def _leaf(cfg, y, noise, parent, camera, table, calls):
    packet, receipt = native_step(parent, cfg, camera, table, calls)
    clean, logl = _orbit(packet, y[camera], noise)
    return packet, clean, logl, receipt


def _extend(parent, table, child, source):
    packet, clean, logl, receipt = child
    total = logl if not parent["cumulative_phase_logl"] else parent["cumulative_phase_logl"][-1] + logl
    return {"schema": SCHEMA, "initial": parent["initial"],
            "packets": (*parent["packets"], packet), "tapes": (*parent["tapes"], table),
            "intervals": (*parent["intervals"], receipt), "sources": (*parent["sources"], source),
            "clean_orbit": (*parent["clean_orbit"], clean), "phase_logl": (*parent["phase_logl"], logl),
            "cumulative_phase_logl": (*parent["cumulative_phase_logl"], total),
            "marginal_logl": (*parent["marginal_logl"], float(_marginal(total)))}


_ROW_FIELDS = ("tapes", "intervals", "clean_orbit", "phase_logl", "cumulative_phase_logl", "marginal_logl")


def _prefix(ref, length):
    return {"schema": SCHEMA, "initial": ref["initial"], "packets": tuple(ref["packets"][:length + 1]),
            **{name: tuple(ref[name][:length]) for name in _ROW_FIELDS},
            "sources": ("reference_initial", *[f"reference_tape_{j+1}" for j in range(length)])}


def _output(path):
    clean = np.stack(path["clean_orbit"])
    logl = np.stack(path["phase_logl"])
    cumulative = np.stack(path["cumulative_phase_logl"])
    return {**path, "clean": clean[:, 0].copy(), "logl": logl[:, 0].copy(),
            "clean_orbit": clean, "phase_logl": logl, "cumulative_phase_logl": cumulative,
            "marginal_logl": np.asarray(path["marginal_logl"]),
            "phase_offsets": phase_offsets(path["packets"][0]),
            "phase_weights": _weights(cumulative[-1])}


def physical_packets(history, phase_index=None):
    """Return every frame at one common phase while preserving base reference tapes."""
    phase = index(history["selected_phase_index"] if phase_index is None else phase_index)
    if not 0 <= phase < PHASES:
        raise ValueError("Use a phase index in the registered finite group.")
    if phase == 0:
        return history["packets"]
    return tuple(translate_packet(packet, history["phase_offsets"][phase]) for packet in history["packets"])


def make_history(cfg, measured, rng):
    """Draw one complete original prior base trajectory, usable as a reference."""
    y, sigma = _camera_contract(cfg, measured)
    calls = counts()
    history = _start_phase(cfg, rng, "prior_initial")
    for c in range(3):
        tape = _table(cfg, c, rng)
        child = _leaf(cfg, y, sigma, history["packets"][-1], c, tape, calls)
        history = _extend(history, tape, child, f"prior_tape_{c+1}")
    if calls != counts(3):
        raise AssertionError("A fresh prior reference costs exactly three native intervals.")
    return _output(history), {"schema": SCHEMA, "mode": "prior", "calls": calls, "native_calls": 3}


def phase_reference(cfg, measured, original_ref):
    """Lift a complete existing unshifted original reference without native calls."""
    y, sigma = _camera_contract(cfg, measured)
    _validate_original(cfg, original_ref, y, sigma)
    history = {
        "schema": SCHEMA, "initial": original_ref["initial"], "packets": original_ref["packets"][:1],
        **{name: () for name in _ROW_FIELDS}, "sources": ("original_prior_reference",),
    }
    for c in range(3):
        packet = original_ref["packets"][c + 1]
        clean, logl = _orbit(packet, y[c], sigma)
        history = _extend(history, original_ref["tapes"][c],
                          (packet, clean, logl, original_ref["intervals"][c]), f"original_reference_tape_{c+1}")
    return _output(history)


def _validate_reference(cfg, ref, y, noise):
    if (ref.get("schema") != SCHEMA or len(ref["packets"]) != 4
            or len(ref["tapes"]) != 3 or len(ref["intervals"]) != 3):
        raise ValueError("The reference must be a complete original base trajectory with phase caches.")
    ages = cfg["prefix_clock"]
    if any(float(packet[0].position[0]) != age for age, packet in zip(ages, ref["packets"], strict=True)):
        raise ValueError("Base reference states must use the original camera boundaries.")
    offsets = phase_offsets(ref["packets"][0])
    if any(not np.array_equal(phase_offsets(p), offsets) for p in ref["packets"][1:]):
        raise ValueError("Every base reference frame needs the same periodic box.")
    camera_rows = [_orbit(packet, y[c], noise) for c, packet in enumerate(ref["packets"][1:])]
    clean, logl = (np.stack(rows) for rows in zip(*camera_rows, strict=True))
    np.testing.assert_array_equal(ref["phase_offsets"], offsets)
    np.testing.assert_array_equal(ref["clean_orbit"], clean)
    np.testing.assert_array_equal(ref["phase_logl"], logl)
    cumulative = np.cumsum(logl, axis=0)
    np.testing.assert_array_equal(ref["cumulative_phase_logl"], cumulative)
    np.testing.assert_allclose(ref["marginal_logl"], _marginal(cumulative), rtol=0., atol=2.e-13)
    np.testing.assert_array_equal(ref["phase_weights"], _weights(cumulative[-1]))


def _replay_suffix(cfg, y, noise, path, ref, camera, calls):
    packet = path["packets"][-1]
    total = path["cumulative_phase_logl"][-1].copy()
    first = None
    for c in range(camera, 3):
        child = _leaf(cfg, y, noise, packet, c, ref["tapes"][c], calls)
        packet = child[0]
        total += child[2]
        if first is None:
            first = child
    return first, total


def _increment(path):
    marginal = path["marginal_logl"]
    return marginal[-1] if len(marginal) == 1 else marginal[-1] - marginal[-2]


def sweep(cfg, measured, ref, n_particles, stage_rng, mode="pgas"):
    """One fixed-budget sweep with a shared phase marginalized across all frames.

    With M_t the log-mean of the cumulative phase likelihoods, the bootstrap
    weight is exp(M_t - M_{t-1}). Each retained ancestor is weighted by its
    previous incremental weight times exp(M_T(graft) - M_{t-1}(prefix)).
    The graft replays all remaining original tapes under its actual prefix.
    A selected phase is drawn only after selecting the terminal base path.
    """
    n = index(n_particles)
    if n < 2 or mode not in MODES:
        raise ValueError("Use at least two particles and a PG or full-suffix PGAS sweep.")
    y, sigma = _camera_contract(cfg, measured)
    _validate_reference(cfg, ref, y, sigma)
    rng, calls = stage_rng, counts()
    live = []
    for j in range(n - 1):
        parent = _start_phase(cfg, rng, f"initial_particle_{j}")
        tape = _table(cfg, 0, rng)
        child = _leaf(cfg, y, sigma, parent["packets"][-1], 0, tape, calls)
        live.append(_extend(parent, tape, child, "prior_tape_1"))
    live.append(_prefix(ref, 1))
    log = np.asarray([_increment(path) for path in live])
    initial_log = log.copy()
    stages = []
    for c in (1, 2):
        chosen, uniform, ordinary_probabilities = _draw(log, rng, n - 1)
        new = []
        for ancestor in chosen:
            parent = live[ancestor]
            tape = _table(cfg, c, rng)
            child = _leaf(cfg, y, sigma, parent["packets"][-1], c, tape, calls)
            new.append(_extend(parent, tape, child, f"prior_tape_{c+1}"))
        prefix_marginal = np.asarray([path["marginal_logl"][-1] for path in live])
        if mode == "pgas":
            replays = [_replay_suffix(cfg, y, sigma, path, ref, c, calls) for path in live]
            graft_phase = np.stack([total for _, total in replays])
            graft_marginal = _marginal(graft_phase)
            log_as = log + graft_marginal - prefix_marginal
            chosen_ref, ref_uniform, ref_probabilities = _draw(log_as, rng, 1)
            j = int(chosen_ref[0])
            new.append(_extend(live[j], ref["tapes"][c], replays[j][0], f"reference_tape_{c+1}"))
        else:
            graft_phase, graft_marginal, log_as, ref_uniform, ref_probabilities = (None,) * 5
            j = n - 1
            new.append(_prefix(ref, c + 1))
        log = np.asarray([_increment(path) for path in new])
        stages.append({"camera": c+1, "ordinary_ancestors": chosen, "ordinary_uniforms": uniform,
                       "ordinary_probabilities": ordinary_probabilities, "reference_ancestor": j,
                       "reference_uniform": ref_uniform, "reference_probabilities": ref_probabilities,
                       "prefix_marginal_logl": prefix_marginal, "graft_phase_logl": graft_phase,
                       "graft_marginal_logl": graft_marginal, "ancestor_log_weight": log_as,
                       "incremental_log_weights": log.copy(),
                       "ordinary_ancestor_ess": float(1. / (ordinary_probabilities @ ordinary_probabilities)),
                       "reference_ancestor_ess": (None if ref_probabilities is None
                                                  else float(1. / (ref_probabilities @ ref_probabilities))),
                       "reference_first_replay_reused": mode == "pgas"})
        live = new
    pick, selection_uniform, probabilities = _draw(log, rng, 1)
    expected = 3 * (n - 1) + (3 * n if mode == "pgas" else 0)
    if calls != counts(expected):
        raise AssertionError("The shared phases must not add any original native intervals.")
    selected = int(pick[0])
    terminal = tuple(_output(path) for path in live)
    result = terminal[selected]
    phase, phase_uniform, phase_probabilities = _draw(result["cumulative_phase_logl"][-1], rng, 1)
    return {**result, "selected_phase_index": int(phase[0])}, {
        "schema": SCHEMA, "mode": mode, "particles": n, "phase_count": PHASES, "calls": calls,
        "native_calls": expected, "initial_log_weights": initial_log, "stages": stages,
        "selected_index": selected, "selection_uniform": selection_uniform,
        "selection_probabilities": probabilities, "selected_phase_index": int(phase[0]),
        "phase_uniform": phase_uniform, "selected_phase_probabilities": phase_probabilities,
        "terminal_histories": terminal, "terminal_weights": probabilities.copy(),
        "terminal_phase_weights": np.stack([path["phase_weights"] for path in terminal]),
    }


def _toy_camera(path):
    return 1 + (sum((2*j + 1)*v for j, v in enumerate(path)) + len(path)**2) % 7


def _toy_shift(path, phase):
    return (path[0] ^ phase, *path[1:])


def _toy_marginal(path):
    if not path:
        return Fraction(1)
    return sum(Fraction(prod(_toy_camera(_toy_shift(path[:k], phase))
                             for k in range(1, len(path) + 1)), 2) for phase in (0, 1))


def _toy_independent_phase(path):
    return prod(Fraction(sum(_toy_camera(_toy_shift(path[:k], s)) for s in (0, 1)), 2)
                for k in range(1, len(path) + 1))


def _toy_norm(scores):
    total = sum(scores)
    return [score / total for score in scores]


def _toy_increment(path):
    return _toy_marginal(path) / _toy_marginal(path[:-1])


def _toy_kernel(reference, mode):
    population_prob = {((x,), reference[:1]): Fraction(1, 2) for x in (0, 1)}
    for t in (1, 2):
        updated = {}
        for population, mass in population_prob.items():
            incremental = [_toy_increment(path) for path in population]
            weights = _toy_norm(incremental)
            marginal_func = _toy_marginal if mode == "pgas" else _toy_independent_phase
            scores = [w * marginal_func(path + reference[t:]) / _toy_marginal(path)
                      for w, path in zip(weights, population, strict=True)]
            ancestor = [Fraction(0), Fraction(1)] if mode == "pg" else _toy_norm(scores)
            for i, j, bit in product(range(2), repeat=3):
                output = (population[i] + (bit,), population[j] + (reference[t],))
                updated[output] = updated.get(output, 0) + mass * weights[i] * ancestor[j] / 2
        population_prob = updated
    output = {}
    for population, mass in population_prob.items():
        weights = _toy_norm([_toy_increment(path) for path in population])
        for path, weight in zip(population, weights, strict=True):
            output[path] = output.get(path, 0) + mass * weight
    return output


def fixtures():
    """Enumerate finite-group lift, PG, PGAS, and a wrong independent-phase kernel."""
    paths = tuple(product(range(2), repeat=3))
    posterior = _toy_norm([_toy_marginal(path) for path in paths])
    restored = {}
    for path, probability in zip(paths, posterior, strict=True):
        for phase in (0, 1):
            physical = _toy_shift(path, phase)
            likelihood = prod(_toy_camera(physical[:k]) for k in (1, 2, 3))
            restored[physical] = restored.get(physical, 0) + probability * likelihood / (2 * _toy_marginal(path))
    direct = _toy_norm([Fraction(prod(_toy_camera(path[:k]) for k in (1, 2, 3))) for path in paths])
    assert [restored[p] for p in paths] == direct
    discrepancy = {}
    for mode in ("pg", "pgas", "independent_phase"):
        kernels = [_toy_kernel(p, mode) for p in paths]
        assert all(sum(kernel.values()) == 1 for kernel in kernels)
        computed = [sum(prob * kernel.get(p, 0) for prob, kernel in zip(posterior, kernels, strict=True)) for p in paths]
        discrepancy[mode] = max(abs(actual - expected) for actual, expected in zip(computed, posterior, strict=True))
    assert discrepancy["pg"] == discrepancy["pgas"] == 0
    assert discrepancy["independent_phase"] > 0
    two_frames = np.full((2, PHASES), -7.)
    two_frames[0, 0] = two_frames[1, 1] = 0.
    assert float(_marginal(two_frames.sum(axis=0))) < _marginal(two_frames).sum()
    return {"schema": SCHEMA, "native_calls": 0, "exact_finite_lift": True, "exact_pg_stationarity": True,
            "exact_pgas_stationarity": True,
            "independent_phase_negative_control_residual": str(discrepancy["independent_phase"])}


if __name__ == "__main__":
    print(fixtures())
