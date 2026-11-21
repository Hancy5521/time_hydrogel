"""Same-atlas recovery and cached Bernstein planning over whole co-phased paths."""

from hashlib import sha256

import numpy as np
from scipy.special import logsumexp

from aam_bernstein_feed_design import bernstein_minimum, bernstein_weights, binary53_effective_probability
from aam_formation_cycle_methods import GOALS
from aam_multiscale_phase_law import PHASES, multiscale_laws
from run_aam_continuous_inverse import ACTIONS, TIMES


SCHEMA = "aam-full-bank-ess1024-common-atlas-legal-branch-planner-v1"
BANK_IDS = np.arange(2026145000, 2026153192, dtype=np.int64)
BANK_IDS.flags.writeable = False
ROUTE_COUNT = 384
METHODS = ("ess1024_cophase", "ess1024_uniform_phase", "old_population", "fixed")


def _ids(values, shape, name):
    ids = np.asarray(values)
    if ids.shape != shape or not np.issubdtype(ids.dtype, np.integer):
        raise ValueError(f"Use the registered integer identities for {name}.")
    return ids


def recovery_law(query, cleaned_camera, camera_noise, bank_ids):
    """Bind the original 8,192 ordered mothers and their 16 whole-path phases."""
    ids = _ids(bank_ids, BANK_IDS.shape, "the full original bank")
    if not np.array_equal(ids, BANK_IDS) or np.shape(cleaned_camera) != (len(BANK_IDS), PHASES, 3, 4):
        raise ValueError("The complete original 8,192 ordered mother bank is required.")
    trained = multiscale_laws(cleaned_camera, query, camera_noise, targets=(1024,))
    weight = trained["weights"]["ess1024"]
    return {"weights": weight, "log_weights": trained["log_weights"]["ess1024"],
            "diagnostics": trained["diagnostics"]["ess1024"],
            "digest": sha256(np.ascontiguousarray(weight, dtype="<f8").tobytes()).hexdigest()}


def _positions(route_ids, frozen_route_ids):
    shape = (ROUTE_COUNT,)
    ids = _ids(route_ids, shape, "the route cache")
    locked = _ids(frozen_route_ids, shape, "the frozen query-independent route selection")
    if (not np.array_equal(ids, locked) or len(np.unique(ids)) != ROUTE_COUNT
            or not ((ids >= BANK_IDS[0]) & (ids <= BANK_IDS[-1])).all()):
        raise ValueError("Routing rows must match the frozen distinct in-bank identities in order.")
    return np.searchsorted(BANK_IDS, ids)


def legal_action_weights(route_phase, control):
    """Mass on existing (route, shared-phase, count) paths for one chosen time."""
    atom = np.asarray(route_phase, dtype=float)
    value = np.asarray(control, dtype=float)
    if (atom.shape != (ROUTE_COUNT, PHASES) or not np.isfinite(atom).all() or np.any(atom < 0.)
            or not np.isclose(atom.sum(), 1., rtol=0., atol=2e-13)
            or value.shape != (2,) or not np.isfinite(value).all() or value[0] not in TIMES):
        raise ValueError("Need a normalized route-phase law and one action on the original time grid.")
    t = int(np.searchsorted(TIMES, value[0]))
    count = bernstein_weights(binary53_effective_probability(value[1]))
    return t, atom[:, :, None] * count


def _optimal(coefficients):
    controls, expected = [], []
    for goal in range(len(GOALS)):
        best = [bernstein_minimum(row) for row in coefficients[:, :, goal]]
        t = int(np.argmin([r["risk"] for r in best]))
        controls.append((TIMES[t], best[t]["p"]))
        expected.append(best[t]["risk"])
    return np.asarray(controls), np.asarray(expected)


def _provided(controls, name):
    selected = np.asarray(controls, dtype=float)
    if (selected.shape != (len(GOALS), 2) or not np.isfinite(selected).all()
            or not np.isin(selected[:, 0], TIMES).all()
            or not ((0. <= selected[:, 1]) & (selected[:, 1] <= 1.)).all()):
        raise ValueError(f"{name} controls must be frozen for both original goals on the registered domain.")
    if name == "fixed" and not np.any(np.all(selected[:, None, :] == ACTIONS, axis=-1), axis=-1).all():
        raise ValueError("Original fixed controls must match the six original recipe anchors.")
    return selected


def _risk(coefficients, selected):
    t = np.searchsorted(TIMES, selected[:, 0])
    p = bernstein_weights(binary53_effective_probability(selected[:, 1]))
    return np.sum(coefficients[t, :, np.arange(len(GOALS))] * p, axis=1)


def plan(query, cleaned_camera, camera_noise, bank_ids, route_ids, phase_losses, *,
         frozen_route_ids, population_controls=None, fixed_controls=None, return_recovery=False):
    """Weight each complete route before minimizing the original six-time risk law.

    Phase losses are in (mother, phase, time, count, goal) order. Conditional
    and uniform-phase planners use exactly the same mothers, mother weights,
    and native continuations. Selecting 384 fixed uniform mothers provides
    a self-normalized approximation to full-bank planning risk. Recovery
    itself always uses all original mothers and phases.
    """
    positions = _positions(route_ids, frozen_route_ids)
    loss = np.asarray(phase_losses, dtype=float)
    dimensions = (ROUTE_COUNT, PHASES, len(TIMES), 7, len(GOALS))
    if loss.shape != dimensions or not np.isfinite(loss).all() or np.any(loss < 0.):
        raise ValueError("Supply finite original per-path losses with shape (384,16,6,7,2).")
    recovery = recovery_law(query, cleaned_camera, camera_noise, bank_ids)
    log_subset = recovery["log_weights"].reshape(len(BANK_IDS), PHASES)[positions]
    log_total = float(logsumexp(log_subset))
    route_phase = np.exp(log_subset - log_total)
    route_phase /= route_phase.sum()
    route_mass = route_phase.sum(axis=1)
    uniform_phase = np.broadcast_to(route_mass[:, None] / PHASES, route_phase.shape)
    coefficients = {
        METHODS[0]: np.einsum("rs,rstkg->tkg", route_phase, loss, optimize=False),
        METHODS[1]: np.einsum("rs,rstkg->tkg", uniform_phase, loss, optimize=False),
        "route_uniform": loss.mean(axis=(0, 1)),
    }
    results = {}
    for name, coeff in coefficients.items():
        controls, expected = _optimal(coeff)
        results[name] = {"controls": controls, "planning_risk": expected}
    for name, controls in zip(METHODS[2:], (population_controls, fixed_controls), strict=True):
        if controls is not None:
            results[name] = {"controls": _provided(controls, name)}
    for info in results.values():
        info["estimated_risk_under_matched_cophase_cache"] = _risk(coefficients[METHODS[0]], info["controls"])
    output = {"schema": SCHEMA, "methods": results, "formal_methods": [m for m in METHODS if m in results],
              "goals": GOALS, "times": TIMES.copy(), "route_positions": positions,
              "route_mother_weights": route_mass, "route_phase_weights": route_phase,
              "route_uniform_phase_weights": uniform_phase, "risk_coefficients": coefficients,
              "full_recovery_digest": recovery["digest"], "full_recovery_diagnostics": recovery["diagnostics"],
              "route_log_retained_recovery_mass": log_total,
              "route_retained_recovery_mass": float(np.exp(log_total)),
              "route_mother_ess": float(1. / np.sum(np.square(route_mass)))}
    if return_recovery:
        output["recovery"] = recovery
    return output


def fixtures():
    """A synthetic legal-loss table makes two matched phase laws choose different actions."""
    phase = np.ones((len(BANK_IDS), PHASES, 3, 4)) * 10.
    phase[:, 0] = 0.
    routes = BANK_IDS[np.random.default_rng(613299).choice(len(BANK_IDS), ROUTE_COUNT, replace=False)]
    config = {"query": np.zeros((3, 4)), "cleaned_camera": phase, "camera_noise": np.ones(4),
              "bank_ids": BANK_IDS, "route_ids": routes, "frozen_route_ids": routes}
    costs = np.full((ROUTE_COUNT, PHASES, len(TIMES), 7, len(GOALS)), 10.)
    count = np.arange(7)
    costs[:, 0, 0] = .1 * count[:, None]
    costs[:, 0, -1] = 3.
    costs[:, 1:, -1] = .1 * (6 - count[:, None])
    fixed = np.asarray(ACTIONS)[[0, -1]]
    provided = np.array([[2.5625, .45], [2.9125, .31]])
    outcome = plan(**config, phase_losses=costs, population_controls=provided, fixed_controls=fixed,
                   return_recovery=True)
    opts = outcome["methods"]
    np.testing.assert_array_equal(opts[METHODS[0]]["controls"], [[TIMES[0], 0.]] * 2)
    np.testing.assert_array_equal(opts[METHODS[1]]["controls"], [[TIMES[-1], 1.]] * 2)
    np.testing.assert_array_equal(opts["old_population"]["controls"], provided)
    np.testing.assert_array_equal(opts["fixed"]["controls"], fixed)
    np.testing.assert_allclose(outcome["route_mother_weights"], 1./ROUTE_COUNT)
    np.testing.assert_allclose(outcome["route_uniform_phase_weights"].sum(axis=1), outcome["route_mother_weights"])
    np.testing.assert_allclose(outcome["route_retained_recovery_mass"], ROUTE_COUNT / len(BANK_IDS))
    assert outcome["full_recovery_digest"] == outcome["recovery"]["digest"]
    time, w = legal_action_weights(outcome["route_phase_weights"], (TIMES[1], .5))
    assert time == 1
    np.testing.assert_allclose(w.sum(), 1.)
    np.testing.assert_allclose(w.sum(axis=(0, 1)), bernstein_weights(.5), atol=2.e-13)

    table = np.ones((len(TIMES), 7, len(GOALS))) * 10.
    optimum = .37
    table[2] = (1. + count*(count-1)/30. - 2.*optimum*count/6. + optimum**2)[:, None]
    np.testing.assert_allclose(_optimal(table)[0], [[TIMES[2], optimum]] * 2, atol=2.e-13)
    try:
        plan(**{**config, "frozen_route_ids": routes[::-1]}, phase_losses=costs)
    except ValueError:
        pass
    else:
        raise AssertionError("A mismatched frozen route order was accepted.")
    return {"passed": True, "native_calls": 0, "fits": 0, "simulation_or_held_artifacts_opened": 0}


if __name__ == "__main__":
    print(fixtures())
