"""Particle Gibbs in the independent innovation representation of native paths."""

from fractions import Fraction
from itertools import product
from math import prod
from operator import index

import numpy as np

from aam_camera_mark_tilt import make_interval_table
from aam_formation_actual_rollout import _packet_copy
from aam_formation_local_initial import build_packet, draw_initial
from aam_snapshot_process import observe
from run_aam_mark_candidate_filter import camera_likelihood
from run_aam_suffix_group_smoother import counts, native_step


SCHEMA = "aam-original-independent-tape-pgas-v1"
MODES = ("pgas", "pg")


def _camera_contract(cfg, measured):
    y, noise = np.asarray(measured, dtype=float), np.asarray(cfg["noise"], dtype=float)
    if (y.shape != (3, 4) or not np.isfinite(y).all() or noise.shape != (4,)
            or not np.isfinite(noise).all() or np.any(noise <= 0.)):
        raise ValueError("Use three finite original camera rows and their original noise.")
    return y, noise


def _table(cfg, camera, rng):
    # Consecutive independent ideal innovations from the caller-owned stream.
    # Sampling a table does not cause the simulator to consume that RNG.
    return make_interval_table(cfg, camera, rng, rng, rng)


def _leaf(cfg, y, noise, parent, camera, table, calls):
    packet, receipt = native_step(parent, cfg, camera, table, calls)
    clean = np.asarray(observe(packet), dtype=float)
    logl = float(camera_likelihood(clean, y[camera], noise))
    if clean.shape != (4,) or not np.isfinite(logl):
        raise ValueError("Original native interval failed its finite likelihood contract.")
    return packet, clean, logl, receipt


def _start(cfg, rng, source):
    initial = draw_initial(cfg, rng)
    packet = _packet_copy(build_packet(cfg, initial), readonly=True)
    return {"schema": SCHEMA, "initial": initial, "packets": (packet,),
            "tapes": (), "clean": (), "logl": (), "intervals": (), "sources": (source,)}


def _extend(parent, table, child, source):
    packet, clean, logl, receipt = child
    return {"schema": SCHEMA, "initial": parent["initial"],
            "packets": (*parent["packets"], packet), "tapes": (*parent["tapes"], table),
            "clean": (*parent["clean"], clean), "logl": (*parent["logl"], logl),
            "intervals": (*parent["intervals"], receipt), "sources": (*parent["sources"], source)}


def _prefix(ref, length):
    return {"schema": SCHEMA, "initial": ref["initial"], "packets": tuple(ref["packets"][:length+1]),
            **{name: tuple(ref[name][:length]) for name in ("tapes", "clean", "logl", "intervals")},
            "sources": ("reference_initial", *[f"reference_tape_{j+1}" for j in range(length)])}


def _as_output(trail):
    return {**trail, "clean": np.stack(trail["clean"]), "logl": np.asarray(trail["logl"])}


def make_history(cfg, measured, rng):
    """Draw a complete legal prior starting trajectory, without access to truth."""
    y, sigma = _camera_contract(cfg, measured)
    calls = counts()
    history = _start(cfg, rng, "prior_initial")
    for c in range(3):
        tape = _table(cfg, c, rng)
        child = _leaf(cfg, y, sigma, history["packets"][-1], c, tape, calls)
        history = _extend(history, tape, child, f"prior_tape_{c+1}")
    return _as_output(history), {"schema": SCHEMA, "mode": "prior", "calls": calls, "native_calls": 3}


def _weights(log):
    a = np.asarray(log, dtype=float)
    if a.ndim != 1 or not np.isfinite(a).all() or not len(a):
        raise ValueError("Every frozen native importance weight needs a finite original log potential.")
    z = np.exp(a - a.max())
    return z / z.sum()


def _draw(log, rng, number):
    probabilities = _weights(log)
    uniform = np.asarray(rng.random(number), dtype=float)
    cumulative = np.cumsum(probabilities)
    cumulative[-1] = 1.
    return np.searchsorted(cumulative, uniform, side="right"), uniform, probabilities


def _replayed_suffix(cfg, y, sigma, history, ref, camera, calls):
    packet = history["packets"][-1]
    logs, first_child = [], None
    for c in range(camera, 3):
        child = _leaf(cfg, y, sigma, packet, c, ref["tapes"][c], calls)
        packet = child[0]
        if first_child is None:
            first_child = child
        logs.append(child[2])
    return first_child, logs


def _validate_reference(cfg, ref, y, noise):
    if (ref.get("schema", SCHEMA) != SCHEMA or len(ref["packets"]) != 4 or len(ref["tapes"]) != 3
            or len(ref["intervals"]) != 3 or np.asarray(ref["clean"]).shape != (3, 4)
            or np.asarray(ref["logl"]).shape != (3,)):
        raise ValueError("Provide one legitimate complete original reference tape and its native states.")
    expected_ages = cfg["prefix_clock"]
    if any(float(packet[0].position[0]) != age for age, packet in zip(expected_ages, ref["packets"], strict=True)):
        raise ValueError("Reference cameras must use the original unchanged boundaries.")
    readouts = np.stack([observe(p) for p in ref["packets"][1:]])
    np.testing.assert_array_equal(readouts, ref["clean"])
    np.testing.assert_allclose(camera_likelihood(readouts, y, noise), ref["logl"], rtol=0., atol=2.e-13)


def sweep(cfg, measured, ref, n_particles, stage_rng, mode="pgas"):
    """One multinomial PG or full-suffix PGAS sweep on the original tape law.

    The first block is (initial variables, camera 1 tape). For a later block
    c, every proposed prefix is replayed with all remaining reference tapes.
    Independence of the original tape priors makes the ancestor weight
    proportional to its current filter weight times all replayed future
    camera likelihoods. The first replay from the selected prefix is reused.
    """
    n = index(n_particles)
    if n < 2 or mode not in MODES:
        raise ValueError("Use at least two particles and an original-tape PG or PGAS sweep.")
    y, sigma = _camera_contract(cfg, measured)
    _validate_reference(cfg, ref, y, sigma)
    rng, calls = stage_rng, counts()
    live = []
    for j in range(n-1):
        parent = _start(cfg, rng, f"initial_particle_{j}")
        tape = _table(cfg, 0, rng)
        child = _leaf(cfg, y, sigma, parent["packets"][-1], 0, tape, calls)
        live.append(_extend(parent, tape, child, "prior_tape_1"))
    live.append(_prefix(ref, 1))
    log = np.array([path["logl"][-1] for path in live])
    initial_log = log.copy()
    stages = []
    for c in (1, 2):
        chosen, uniform, ordinary_probabilities = _draw(log, rng, n-1)
        new = []
        for ancestor in chosen:
            tape = _table(cfg, c, rng)
            parent = live[ancestor]
            child = _leaf(cfg, y, sigma, parent["packets"][-1], c, tape, calls)
            new.append(_extend(parent, tape, child, f"prior_tape_{c+1}"))
        if mode == "pgas":
            extensions = [_replayed_suffix(cfg, y, sigma, h, ref, c, calls) for h in live]
            replayed = np.array([tail for _, tail in extensions])
            log_as = log + replayed.sum(axis=1)
            retained, retained_u, retained_probability = _draw(log_as, rng, 1)
            j = int(retained[0])
            new.append(_extend(live[j], ref["tapes"][c], extensions[j][0], f"reference_tape_{c+1}"))
        else:
            replayed, log_as, retained_u, retained_probability = None, None, None, None
            j = n-1
            new.append(_prefix(ref, c+1))
        log = np.asarray([x["logl"][-1] for x in new])
        stages.append({"camera": c+1, "ordinary_ancestors": chosen, "ordinary_uniforms": uniform,
                       "ordinary_probabilities": ordinary_probabilities,
                       "reference_ancestor": j, "reference_uniform": retained_u,
                       "reference_probabilities": retained_probability, "replayed_future_logl": replayed,
                       "ancestor_log_weight": log_as, "incremental_log_weights": log.copy(),
                       "ordinary_ancestor_ess": float(1./(ordinary_probabilities @ ordinary_probabilities)),
                       "reference_ancestor_ess": (None if retained_probability is None
                                                  else float(1./(retained_probability @ retained_probability))),
                       "reference_first_replay_reused": mode == "pgas"})
        live = new
    pick, u, probabilities = _draw(log, rng, 1)
    expected = 3*(n-1)+(3*n if mode == "pgas" else 0)
    if calls != counts(expected):
        raise AssertionError("Original native accounting does not match fixed fair budget.")
    index_final = int(pick[0])
    terminal = tuple(_as_output(history) for history in live)
    return terminal[index_final], {"schema": SCHEMA, "mode": mode, "particles": n,
          "calls": calls, "native_calls": expected, "initial_log_weights": initial_log, "stages": stages,
          "selected_index": index_final, "selection_uniform": u, "selection_probabilities": probabilities,
          "terminal_histories": terminal, "terminal_weights": probabilities.copy()}


def _finite_potential(path):
    return 1+(sum((2*j + 1)*v for j, v in enumerate(path)) + len(path)**2) % 7


def _finite_probs(values):
    den = sum(values)
    return [Fraction(x, den) for x in values]


def _enumerated_kernel(reference, mode):
    """Independent exact enumeration on a tiny binary non-Markovian target."""
    live = {((x,), reference[:1]): Fraction(1, 2) for x in (0, 1)}
    for t in (1, 2):
        updated = {}
        for population, mass in live.items():
            potentials = [_finite_potential(x) for x in population]
            weights = _finite_probs(potentials)
            horizon = t+1 if mode == "truncated" else 3
            prefixed = [x+reference[t:] for x in population]
            scores = [p*prod(_finite_potential(x[:k]) for k in range(t+1, horizon+1))
                      for x, p in zip(prefixed, potentials, strict=True)]
            ancestor = [Fraction(0), Fraction(1)] if mode == "pg" else _finite_probs(scores)
            for i, j, b in product(range(2), repeat=3):
                outcome = (population[i]+(b,), population[j]+(reference[t],))
                updated[outcome] = updated.get(outcome, Fraction(0))+mass*weights[i]*ancestor[j]/2
        live = updated
    output = {}
    for population, mass in live.items():
        for path, p in zip(population, _finite_probs([_finite_potential(x) for x in population]), strict=True):
            output[path] = output.get(path, Fraction(0))+mass*p
    return output


def fixtures():
    """Test exact enumerable stationarity and detect truncated suffix weights."""
    paths = tuple(product(range(2), repeat=3))
    posterior = _finite_probs([prod(_finite_potential(x[:k]) for k in (1,2,3)) for x in paths])
    results = {}
    for mode in ("pg", "pgas", "truncated"):
        rows = [_enumerated_kernel(x, mode) for x in paths]
        assert all(sum(row.values()) == 1 for row in rows)
        stationary = [sum(p*row.get(x, 0) for p, row in zip(posterior, rows, strict=True)) for x in paths]
        results[mode] = max(abs(a-b) for a,b in zip(posterior, stationary, strict=True))
    assert results["pg"] == results["pgas"] == 0
    assert results["truncated"] > 0
    return {"schema": SCHEMA, "native_calls": 0, "exact_pg_stationarity": True,
            "exact_pgas_stationarity": True, "truncation_negative_control_residual": str(results["truncated"])}


if __name__ == "__main__":
    print(fixtures())
