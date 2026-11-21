"""Pure declared-objective/readout algebra for actual formation trajectories.

No target, physical scale, threshold, confidence coefficient or fitted parameter
is chosen here.  No reaction, random draw, fit or file write occurs on import.
The leading axes are DRAW, ACTION; time/probe axes are never exchangeable
counterfactual samples.  An action mask excludes an explicitly unsupported
action from selection; it never deletes a draw or silently repairs bad data.
"""
from __future__ import annotations

import numpy as np


SCHEMA = "aam-formation-process-objective-v1"


def _finite(value, name):
    a = np.asarray(value, dtype=np.float64)
    if not np.isfinite(a).all():
        raise ValueError(name + " must be finite, including zero-weight entries.")
    return a


def _process(properties):
    p = _finite(properties, "properties")
    if p.ndim != 4 or any(n == 0 for n in p.shape):
        raise ValueError("properties must have nonempty (draw,action,age,probe_lag) axes.")
    return p


def _grid(value, shape, name, *, strictly_positive=False):
    a = _finite(value, name)
    if a.shape != shape:
        raise ValueError(name + " must explicitly match (age,probe_lag), without broadcasting.")
    invalid = np.any(a <= 0.) if strictly_positive else np.any(a < 0.)
    if invalid:
        raise ValueError(name + (" must be positive." if strictly_positive else " must be nonnegative."))
    return a


def _weights(value, draws, actions, name):
    w = _finite(value, name)
    if w.shape == (draws,):
        w = np.repeat(w[:, None], actions, axis=1)
    elif w.shape != (draws, actions):
        raise ValueError(name + " must have shape (draw,) or (draw,action).")
    if np.any(w < 0.):
        raise ValueError(name + " must be nonnegative, without probability floors.")
    totals = w.sum(axis=0)
    if not np.isfinite(totals).all() or np.any(totals <= 0.):
        raise ValueError("Every action must retain positive total " + name + ".")
    # Explicit normalization preserves every original atom and every exact zero.
    return w / totals[None, :], totals


def _action_vector(value, actions, name, *, interval=None):
    a = _finite(value, name)
    if a.ndim == 0:
        a = np.full(actions, float(a))
    elif a.shape != (actions,):
        raise ValueError(name + " must be scalar or one value per action.")
    if interval is not None and (np.any(a < interval[0]) or np.any(a > interval[1])):
        raise ValueError(name + " lies outside its declared numerical domain.")
    return a


def stable_argmin(expected_loss, action_support):
    """Smallest original action index among supported minimizers; no tolerance tie."""
    loss = _finite(expected_loss, "expected_loss")
    support = np.asarray(action_support)
    if (loss.ndim != 1 or not len(loss) or support.shape != loss.shape
            or support.dtype.kind != "b" or not support.any()):
        raise ValueError("A nonempty 1D loss and Boolean supported-action mask are required.")
    ids = np.flatnonzero(support)
    return int(ids[np.argmin(loss[ids])])


def trajectory_objective(properties, terminal_conversion, weights, *, target,
                         scales, point_weights, minimum_conversion,
                         conversion_penalty, action_support):
    """Compute E[trajectory loss], never loss(E[trajectory]).

    For each draw/action, L=sum(w_t*(G_t-target_t)^2/scales_t^2)/sum(w_t)
    + penalty*max(cmin-conversion,0)^2.  point_weights, scales, cmin and penalty
    must come from the caller's frozen registration.  The hinge is the stated
    objective; conversion is never clipped into [0,1].  Native atom weights may
    be unnormalized and are normalized once per action, without resampling,
    truncation, flooring or support filtering.  All input draws are validated,
    including atoms of weight zero and unsupported actions.
    """
    p = _process(properties)
    d, a, t, k = p.shape
    shape = (t, k)
    y = _finite(target, "target")
    if y.shape != shape:
        raise ValueError("target must explicitly match (age,probe_lag).")
    s = _grid(scales, shape, "scales", strictly_positive=True)
    g = _grid(point_weights, shape, "point_weights")
    if not np.isfinite(g.sum()) or g.sum() <= 0.:
        raise ValueError("point_weights must have positive finite total mass.")
    conversion = _finite(terminal_conversion, "terminal_conversion")
    if conversion.shape != (d, a) or np.any(conversion < 0.) or np.any(conversion > 1.):
        raise ValueError("terminal_conversion must be an actual [0,1] array (draw,action).")
    cmin = _action_vector(minimum_conversion, a, "minimum_conversion", interval=(0., 1.))
    penalty = _action_vector(conversion_penalty, a, "conversion_penalty")
    if np.any(penalty < 0.):
        raise ValueError("conversion_penalty must be nonnegative.")
    w, totals = _weights(weights, d, a, "weights")
    mismatch = np.sum(((p-y[None, None]) / s[None, None])**2
                      * (g/g.sum())[None, None], axis=(2, 3))
    conversion_loss = penalty[None, :] * np.maximum(cmin[None, :]-conversion, 0.)**2
    loss = mismatch + conversion_loss
    expected = np.sum(w*loss, axis=0)
    chosen = stable_argmin(expected, action_support)
    return {"schema": SCHEMA, "per_draw_trajectory_loss": mismatch,
            "per_draw_conversion_loss": conversion_loss, "per_draw_total_loss": loss,
            "native_normalized_weights": w, "input_weight_sum": totals,
            "expected_loss": expected, "chosen_action_index": chosen,
            "action_support": np.array(action_support, copy=True),
            "draws_retained_per_action": np.full(a, d, dtype=np.int64),
            "positive_weight_draws_per_action": np.count_nonzero(w, axis=0),
            "expectation_before_decision": True,
            "counterfactual_actions_treated_as_joint_distribution": False}


def per_action_energy(properties, reference, weights, *, scales, point_weights,
                      reference_weights, block_size=256):
    """Native weighted V-energy score separately for every action.

    reference has (reference_draw,action,age,probe_lag); a single observation
    still needs the explicit reference_draw axis.  The norm is
    sqrt(sum(normalized_point_weights*((x-y)/scales)^2)).  Thus equal point
    weights give Euclidean score/sqrt(number of trajectory entries).  Self
    pairs are retained in the weighted V-statistic; no unsupported claim of
    unbiased U scoring is made.  Arrays from distinct actions never enter the
    same distance calculation.  Reference draws may share a simulated mother;
    they are not additional independent mothers.
    """
    x = _process(properties)
    y = _process(reference)
    d, a, t, k = x.shape
    if y.shape[1:] != x.shape[1:]:
        raise ValueError("Reference action/time/lag axes must match predictions.")
    if (isinstance(block_size, (bool, np.bool_)) or not isinstance(block_size, (int, np.integer))
            or block_size <= 0):
        raise ValueError("block_size must be a positive integer.")
    s = _grid(scales, (t, k), "scales", strictly_positive=True)
    g = _grid(point_weights, (t, k), "point_weights")
    if not np.isfinite(g.sum()) or g.sum() <= 0.:
        raise ValueError("point_weights must have positive finite total mass.")
    w, totals = _weights(weights, d, a, "weights")
    v, reference_totals = _weights(reference_weights, len(y), a, "reference_weights")
    transform = np.sqrt(g/g.sum())/s
    first = np.zeros((len(y), a))
    second = np.zeros(a)
    for action in range(a):
        xx = (x[:, action]*transform).reshape(d, -1)
        yy = (y[:, action]*transform).reshape(len(y), -1)
        for begin in range(0, d, block_size):
            stop = min(d, begin+block_size)
            dist = np.linalg.norm(xx[begin:stop, None]-yy[None, :], axis=2)
            first[:, action] += w[begin:stop, action] @ dist
            for other in range(0, d, block_size):
                other_stop = min(d, other+block_size)
                pairs = np.linalg.norm(xx[begin:stop, None]-xx[None, other:other_stop], axis=2)
                second[action] += (w[begin:stop, action] @ pairs
                                   @ w[other:other_stop, action])
    by_reference = first-.5*second[None, :]
    score = np.sum(v*by_reference, axis=0)
    if not np.isfinite(score).all():
        raise ArithmeticError("Energy score became nonfinite; no replacement value supplied.")
    return {"schema": SCHEMA, "per_action_energy_score": score,
            "per_reference_action_energy": by_reference,
            "per_action_pair_distance": second,
            "native_normalized_weights": w, "input_weight_sum": totals,
            "reference_normalized_weights": v, "reference_weight_sum": reference_totals,
            "draws_retained_per_action": np.full(a, d, dtype=np.int64),
            "reference_draws_per_action": np.full(a, len(y), dtype=np.int64),
            "pair_estimator": "weighted_V_including_self_pairs",
            "counterfactual_actions_treated_as_joint_distribution": False}


def paired_mother_summary(candidate, baseline, *, mother_ids, critical_value):
    """Paired independent-mother differences, candidate minus baseline.

    Each row must already be ONE independently generated mother, after all
    within-mother goals/actions/replicates were reduced by a registered rule.
    Additional trailing axes are summarized individually, never pooled into
    extra mothers.  The registered critical value (e.g. a t or simultaneous
    coefficient) is supplied explicitly; no confidence convention is chosen.
    """
    x, y = _finite(candidate, "candidate"), _finite(baseline, "baseline")
    ids = np.asarray(mother_ids)
    if (x.shape != y.shape or x.ndim < 1 or len(x) < 2 or ids.shape != (len(x),)
            or len(np.unique(ids)) != len(ids)):
        raise ValueError("Equal arrays with >=2 unique, explicitly identified mother rows are required.")
    critical = float(critical_value)
    if not np.isfinite(critical) or critical <= 0.:
        raise ValueError("critical_value must be explicitly registered, positive and finite.")
    delta = x-y
    mean = delta.mean(axis=0)
    se = delta.std(axis=0, ddof=1)/np.sqrt(len(delta))
    return {"schema": SCHEMA, "mothers": len(delta), "mother_ids": ids.copy(),
            "candidate_mean": x.mean(axis=0), "baseline_mean": y.mean(axis=0),
            "paired_difference": delta, "difference_mean": mean,
            "difference_standard_error": se,
            "interval_lower": mean-critical*se, "interval_upper": mean+critical*se,
            "registered_critical_value": critical,
            "direction": "candidate_minus_baseline",
            "additional_axes_counted_as_independent_mothers": False}


def deterministic_objective_checks():
    """Defined only; caller may later execute once in an integrated registration.

    Fixed arrays check Jensen distinction, native weights and zero atoms,
    opposite-target choices, stable ties, action support, weighted ES against
    independent scalar loops, paired-mother algebra, explicit NaN rejection and
    all input bytes unchanged.  No RNG, physical simulator, fit or file access.
    """
    import copy
    maximum = 0.
    checks = 0
    x = np.array([[[[0.]], [[1.]]], [[[2.]], [[1.]]]])
    conversion = np.ones((2, 2))
    weights = np.array([.5, .5])
    grid = np.ones((1, 1))
    support = np.ones(2, dtype=bool)
    inputs = (x, conversion, weights, grid, support)
    before = copy.deepcopy(inputs)

    def close(value, expected):
        nonlocal maximum, checks
        actual, wanted = np.asarray(value), np.asarray(expected)
        if actual.shape != wanted.shape or not np.isfinite(actual).all():
            raise AssertionError("Deterministic objective fixture shape/finiteness mismatch.")
        error = float(np.max(np.abs(actual-wanted))) if actual.size else 0.
        maximum = max(maximum, error); checks += 1
        if error > 2.e-12:
            raise AssertionError("Deterministic objective fixture error: "+str(error))

    def objective(data=x, w=weights, target=grid, mask=support, conv=conversion,
                  cmin=0., penalty=0.):
        return trajectory_objective(data, conv, w, target=target, scales=grid,
                                    point_weights=grid, minimum_conversion=cmin,
                                    conversion_penalty=penalty, action_support=mask)

    jensen = objective()
    close(jensen["expected_loss"], [1., 0.])
    close(jensen["chosen_action_index"], 1)
    close(objective(w=np.array([3., 1.]), target=np.zeros((1, 1)))["expected_loss"], [1., 1.])
    close(objective(w=np.array([1., 0.]), target=np.zeros((1, 1)))["expected_loss"], [0., 1.])
    close(objective(target=np.zeros((1, 1)))["chosen_action_index"], 1)
    constant = np.array([[[[0.]], [[2.]]], [[[0.]], [[2.]]]])
    close(objective(data=constant, target=np.zeros((1, 1)))["chosen_action_index"], 0)
    close(objective(data=constant, target=np.full((1, 1), 2.))["chosen_action_index"], 1)
    close(objective(mask=np.array([True, False]))["chosen_action_index"], 0)
    close(stable_argmin(np.array([1., 1.]), support), 0)
    cv = np.array([[.25, .5], [.75, 1.]])
    close(objective(conv=cv, cmin=.75, penalty=2.)["per_draw_conversion_loss"],
          [[.5, .125], [0., 0.]])
    references = np.array([[[[.3]], [[1.4]]], [[[1.1]], [[.7]]]])
    rw = np.array([.25, .75])
    es = per_action_energy(x, references, weights, scales=grid,
                           point_weights=grid, reference_weights=rw, block_size=1)
    gold = []
    for action in range(2):
        first = sum(weights[i]*rw[j]*abs(x[i, action, 0, 0]-references[j, action, 0, 0])
                    for i in range(2) for j in range(2))
        pair = sum(weights[i]*weights[j]*abs(x[i, action, 0, 0]-x[j, action, 0, 0])
                   for i in range(2) for j in range(2))
        gold.append(first-.5*pair)
    close(es["per_action_energy_score"], gold)
    pm = paired_mother_summary(np.array([1., 3., 2.]), np.array([2., 2., 2.]),
                               mother_ids=np.array(["m0", "m1", "m2"]), critical_value=2.)
    close(pm["difference_mean"], 0.)
    close(pm["difference_standard_error"], 1./np.sqrt(3.))
    bad = x.copy(); bad[0, 0, 0, 0] = np.nan
    try:
        objective(data=bad, w=np.array([0., 1.]))
    except ValueError:
        checks += 1
    else:
        raise AssertionError("NaN at a zero-weight atom was silently discarded.")
    for left, right in zip(inputs, before, strict=True):
        if left.dtype != right.dtype or left.shape != right.shape or left.tobytes() != right.tobytes():
            raise AssertionError("Objective fixture changed a caller-owned array.")
    return {"passed": True, "checks": checks, "maximum_absolute_error": maximum,
            "scope": "Deterministic algebra fixtures only; no material, inference or inverse-design performance evidence.",
            "native_calls": 0, "fits": 0, "random_draws": 0}
