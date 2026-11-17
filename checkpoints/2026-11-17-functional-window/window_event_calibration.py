"""Calibrate the event that an active mother has any unsafe recommendation.

Inputs have independent mothers on axis zero and a predeclared workload on the
remaining axes. Actions and scores may adapt to the observed measurements and
other predictions within the mother. They are fixed functions of those inputs
before this calibration, which uses the realized margins of their chosen actions.
"""

from fractions import Fraction

import numpy as np
from scipy.stats import beta, binom


def _mask(values, valid):
    if not values.ndim or not values.size or not values.shape[0]:
        raise ValueError('Expected a nonempty workload with mothers on axis zero.')
    if valid is None:
        return np.ones(values.shape, dtype=bool)
    mask = np.asarray(valid)
    if mask.dtype.kind != 'b' or mask.shape != values.shape:
        raise ValueError('Expected a valid-target Boolean mask matching the workload.')
    if not mask.reshape(len(values), -1).any(axis=1).all():
        raise ValueError('Every mother must have at least one predeclared request.')
    return mask


def _scores(values, valid):
    x = np.asarray(values, dtype=float)
    mask = _mask(x, valid)
    if not np.isfinite(x[mask]).all():
        raise ValueError('Every deployed score must be finite.')
    return x, mask


def _truth(margins, mask):
    y = np.asarray(margins, dtype=float)
    if y.shape != mask.shape or not np.isfinite(y[mask]).all():
        raise ValueError('Margins must be finite and match the predeclared workload.')
    return y


def _probability(value, name):
    if not 0 < value < 1:
        raise ValueError(f'{name} must lie strictly between zero and one.')


def critical_unsafe_score(margins, scores, *, valid=None):
    """Each mother's lowest acceptance score among truly unsafe recommendations."""
    s, mask = _scores(scores, valid)
    y = _truth(margins, mask)
    return np.where(mask & (y > 0), s, np.inf).reshape(len(s), -1).min(axis=1)


def _failure_count(n, risk, tail):
    """Largest integer count with binomial lower tail at most tail, or -1."""
    candidate = binom.ppf(tail, n, risk).astype(int)
    return np.where(binom.cdf(candidate, n, risk) <= tail, candidate, candidate - 1)


def fit_marginal_event(margins, scores, *, alpha=.10, delta=None, valid=None):
    """Fit a strict threshold for the event that any recommendation is unsafe.

    Without delta, the risk bound is marginal over exchangeable calibration and
    the next mother. With delta, the bound holds with probability 1-delta over
    IID calibration for its corresponding population. Neither version conditions
    on activity. Any workload acceptance after this fit must use the same score
    and action rule as calibration.
    """
    _probability(alpha, 'alpha')
    critical = critical_unsafe_score(margins, scores, valid=valid)
    n = len(critical)
    if delta is None:
        rank = int((n + 1) * Fraction(str(alpha)))
        guarantee = 'exchangeable_marginal'
    else:
        _probability(delta, 'delta')
        rank = int(_failure_count(n, alpha, delta)) + 1
        guarantee = 'iid_high_probability_marginal'
    threshold = float(np.partition(critical, rank - 1)[rank - 1]) if rank else None
    decision = 'never' if threshold is None else 'all' if np.isposinf(threshold) else 'strict_lt'
    return {'alpha':float(alpha), 'delta':None if delta is None else float(delta), 'guarantee':guarantee,
            'rank':rank, 'groups':n, 'finite_critical_scores':int(np.isfinite(critical).sum()),
            'decision':decision, 'threshold':threshold if decision=='strict_lt' else None}


def apply_marginal_event(scores, fit, *, valid=None):
    s, mask = _scores(scores, valid)
    decision = fit['decision']
    if decision == 'never':
        return np.zeros(s.shape, dtype=bool)
    if decision == 'all':
        return mask
    if decision != 'strict_lt' or not np.isfinite(fit['threshold']):
        raise ValueError('Invalid marginal event threshold.')
    return mask & (s < fit['threshold'])


def certify_policy(margins, accepted, *, max_risk=.25, delta=.05, min_activity=.10, valid=None):
    """Exact one-sided test for a frozen policy at fixed population targets.

    Risk is P(any unsafe recommendation | at least one accepted request).
    Use min_activity=None for a risk-only test. Otherwise a passing policy also
    certifies P(at least one accepted request) > min_activity. The two
    individual confidence bounds are descriptive after policy selection
    because only the prespecified joint threshold claim is protected.
    """
    _probability(max_risk, 'max_risk')
    _probability(delta, 'delta')
    if min_activity is not None:
        _probability(min_activity, 'min_activity')
    accept = np.asarray(accepted)
    mask = _mask(accept, valid)
    if accept.dtype.kind != 'b' or np.any(accept & ~mask):
        raise ValueError('Expected a Boolean acceptance mask on valid requests.')
    y = _truth(margins, mask)
    n = len(accept)
    m = int(accept.reshape(n, -1).any(axis=1).sum())
    bad = int((accept & (y > 0)).reshape(n, -1).any(axis=1).sum())
    risk_p = float(binom.cdf(bad, m, max_risk))
    use_p = None if min_activity is None else float(binom.sf(m - 1, n, min_activity))
    joint_p = risk_p if use_p is None else max(risk_p, use_p)
    upper = 1. if bad==m else float(beta.ppf(1-delta, bad+1, m-bad))
    lower = 0. if m==0 else float(beta.ppf(delta, m, n-m+1))
    return {'groups':n, 'active_mothers':m, 'unsafe_active_mothers':bad,
            'accepted_requests':int(accept.sum()), 'requests':int(mask.sum()),
            'max_risk':float(max_risk), 'min_activity':None if min_activity is None else float(min_activity),
            'delta':float(delta), 'risk_p':risk_p, 'activity_p':use_p, 'joint_p':joint_p,
            'individual_risk_upper':upper, 'individual_activity_lower':lower,
            'certified':bool(joint_p <= delta)}


def certify_policy_sequence(candidates, *, max_risk=.25, delta=.05, min_activity=.10, valid=None):
    """Test an independently ordered sequence of (name, margins, acceptance) policies.

    Stop at the first joint test failure. Selecting among passed policies
    controls the fixed target family-wise claim at delta. Caller must ensure
    the mothers, observation workloads and their order match across policies.
    """
    items = list(candidates)
    names = [item[0] for item in items]
    if not names or len(names)!=len(set(names)):
        raise ValueError('Expected unique policy names in a nonempty fixed sequence.')
    if len({np.shape(item[2]) for item in items}) != 1:
        raise ValueError('Every policy must apply to the same workload shape.')
    passed, tests = [], []
    for name, margins, accepted in items:
        result = certify_policy(margins, accepted, max_risk=max_risk, delta=delta,
                                min_activity=min_activity, valid=valid)
        tests.append({'name':name, **result})
        if not result['certified']:
            break
        passed.append(name)
    return {'passed':passed, 'selected':passed[-1] if passed else None, 'tests':tests,
            'untested':names[len(tests):], 'delta':float(delta)}


def fit_policy_sequence(margins, scores, *, cutoffs, max_risk=.25, delta=.05, min_activity=.10, valid=None):
    """Apply fixed-sequence testing to strict cutoffs in their preregistered order."""
    s, mask = _scores(scores, valid)
    limits = [float(t) for t in cutoffs]
    if not np.isfinite(limits).all():
        raise ValueError('Every proposed cutoff must be finite.')
    policies = [(t, margins, mask & (s < t)) for t in limits]
    result = certify_policy_sequence(policies, max_risk=max_risk, delta=delta,
                                     min_activity=min_activity, valid=mask)
    return {**result, 'cutoff':result['selected'], 'comparison':'strict_lt'}


def single_policy_power(n, *, true_activity, true_risk, max_risk=.25, delta=.05, min_activity=.10):
    """Exact design power for one locked policy under hypothetical IID rates."""
    _probability(max_risk, 'max_risk')
    _probability(delta, 'delta')
    if min_activity is not None:
        _probability(min_activity, 'min_activity')
    if int(n)!=n or n<1 or not 0<=true_activity<=1 or not 0<=true_risk<=1:
        raise ValueError('Expected positive integer n and true rates in [0,1].')
    active = np.arange(1, int(n)+1)
    passes_activity = True if min_activity is None else binom.sf(active-1, n, min_activity) <= delta
    critical = _failure_count(active, max_risk, delta)
    return float(np.sum(binom.pmf(active, n, true_activity) *
                        binom.cdf(critical, active, true_risk) * passes_activity))
