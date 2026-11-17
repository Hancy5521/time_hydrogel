"""Calibrate window margins and selective decisions at the mother-group level.

Signed margin <= 0 passes a complete sampled window. Axis zero indexes independent
mother groups. Requests, counterfactual cases, times and actions within a group
may be arbitrarily dependent. All forecast rules, scales, masks and threshold
candidates must be fixed independently of the calibration labels.
"""

from dataclasses import dataclass
from fractions import Fraction

import numpy as np
from scipy.stats import beta


def _array(values):
    result = np.asarray(values, dtype=float)
    if not result.ndim or not result.shape[0] or not result.size:
        raise ValueError('Expect a nonempty array with mother groups on axis zero.')
    return result


def _matching(values, shape):
    result = np.asarray(values, dtype=float)
    if result.shape != shape:
        raise ValueError(f'Expected shape {shape}, received {result.shape}.')
    return result


def _valid(valid, shape):
    if valid is None:
        return np.ones(shape, dtype=bool)
    mask = np.asarray(valid)
    if mask.dtype.kind != 'b' or mask.shape != shape:
        raise ValueError('Mask must be Boolean and match the target shape.')
    if not mask.reshape(shape[0], -1).any(axis=1).all():
        raise ValueError('Each mother group must have at least one predeclared target.')
    return mask


def _finite(values, valid):
    if not np.isfinite(values[valid]).all():
        raise ValueError('Nonfinite values in predeclared targets or forecasts.')


def _scales(scale, shape, valid):
    values = np.broadcast_to(np.asarray(scale, dtype=float), shape)
    _finite(values, valid)
    if (values[valid] <= 0).any():
        raise ValueError('Every active scale must be strictly positive.')
    return np.where(valid, values, 1.)


def _flat(values):
    return values.reshape(values.shape[0], -1)


def take_actions(values, selected):
    """Gather frozen integer actions from the last axis of [group, ..., action]."""
    bank = _array(values)
    idx = np.asarray(selected)
    if bank.ndim < 2 or idx.dtype.kind not in 'iu' or idx.shape != bank.shape[:-1]:
        raise ValueError('Selected actions must be integer indices matching all non-action axes.')
    if (idx < 0).any() or (idx >= bank.shape[-1]).any():
        raise ValueError('Invalid action selection.')
    return np.take_along_axis(bank, idx[..., None], axis=-1)[..., 0]


@dataclass(frozen=True)
class GroupCorrection:
    """Finite-sample split-conformal correction and its original rank."""

    radius: float
    rank: int
    n_groups: int
    alpha: float

    def apply(self, upper, *, scale=1.):
        """Apply the frozen upper correction on the same target/scale definition."""
        forecast = _array(upper)
        valid = np.ones(forecast.shape, dtype=bool)
        _finite(forecast, valid)
        return forecast + self.radius*_scales(scale, forecast.shape, valid)

    def apply_interval(self, lower, upper, *, scale=1.):
        """Expand a frozen marginal band into a simultaneous group band."""
        low = _array(lower)
        high = _matching(upper, low.shape)
        if (low > high).any():
            raise ValueError('Lower endpoint exceeds upper endpoint.')
        return -self.apply(-low, scale=scale), self.apply(high, scale=scale)

    def metadata(self):
        """JSON-compatible summary, with an explicit flag for infinite radii."""
        return {'radius': self.radius if np.isfinite(self.radius) else None,
                'vacuous': bool(np.isinf(self.radius)), 'rank': self.rank,
                'n_groups': self.n_groups, 'alpha': self.alpha}


def _correction(score, valid, alpha):
    if not 0 < alpha < 1:
        raise ValueError('Alpha must be strictly between zero and one.')
    mother_scores = _flat(np.where(valid, score, -np.inf)).max(axis=1)
    if not np.isfinite(mother_scores).all():
        raise ValueError('Every mother must have a finite calibration score.')
    n = len(mother_scores)
    cutoff = (n+1)*(1-Fraction(str(alpha)))
    rank = -(-cutoff.numerator//cutoff.denominator)
    radius = np.inf if rank > n else max(0., float(np.partition(mother_scores, rank-1)[rank-1]))
    return GroupCorrection(radius, rank, n, float(alpha))


def fit_group_upper(y, upper, *, alpha=.1, scale=1., valid=None):
    """Calibrate an upper envelope for all predeclared targets in a fresh mother.

    The score is max_j (y_j - upper_j)/scale_j within one mother. A catalogue
    bank supports selection after calibration because every action is included.
    For a selected-action bank, the action policy must already be frozen.
    """
    y = _array(y)
    forecast = _matching(upper, y.shape)
    mask = _valid(valid, y.shape)
    width = _scales(scale, y.shape, mask)
    _finite(y, mask)
    _finite(forecast, mask)
    difference = (np.where(mask, y, 0.) - np.where(mask, forecast, 0.)) / width
    return _correction(difference, mask, alpha)


def fit_group_interval(y, lower, upper, *, alpha=.1, scale=1., valid=None):
    """Calibrate two-sided targets using each mother's largest endpoint error."""
    y = _array(y)
    low, high = (_matching(v, y.shape) for v in (lower, upper))
    mask = _valid(valid, y.shape)
    width = _scales(scale, y.shape, mask)
    for arr in (y, low, high):
        _finite(arr, mask)
    if (low[mask] > high[mask]).any():
        raise ValueError('Lower endpoint exceeds upper endpoint.')
    outcome, low, high = (np.where(mask, arr, 0.) for arr in (y, low, high))
    return _correction(np.maximum(outcome-high, low-outcome)/width, mask, alpha)


def _ratios(components, bootstrap, confidence, seed):
    if not 0 < confidence < 1:
        raise ValueError('Bootstrap confidence must be strictly between zero and one.')
    if int(bootstrap) != bootstrap or bootstrap < 0:
        raise ValueError('Bootstrap draws must be a nonnegative integer.')
    group_count = len(next(iter(components.values()))[0])
    idx = np.random.default_rng(seed).integers(group_count,size=(int(bootstrap),group_count)) if bootstrap else None
    result, intervals = {}, {}
    for name, (num, den) in components.items():
        n, d = np.sum(num), np.sum(den)
        result[name] = float(n/d) if d else None
        if idx is not None:
            m = np.asarray(den)[idx].sum(axis=1)
            sampled = np.asarray(num)[idx].sum(axis=1)[m > 0]/m[m > 0]
            # Undefined ratios stay undefined. Report how often bootstrap draws omit all acceptances.
            interval = np.quantile(sampled, [(1-confidence)/2,(1+confidence)/2]).tolist() if len(sampled) else None
            intervals[name] = {'interval':interval,'usable_draws':int(len(sampled))}
    if idx is not None:
        result['mother_bootstrap'] = {'draws':int(bootstrap),'confidence':confidence,'ratios':intervals}
    return result


def _band_metrics(target, miss, valid, width, reference, bootstrap, confidence, seed):
    defect = _flat(miss & valid)
    per_group = _flat(valid).sum(axis=1)
    group = np.ones(len(target),dtype=int)
    components = {
        'simultaneous_mother_coverage':(~defect.any(axis=1),group),
        'pooled_target_coverage':(per_group-defect.sum(axis=1),per_group),
    }
    widths = width[valid] if width is not None else None
    result = _ratios(components,bootstrap,confidence,seed)
    result['groups'] = len(target)
    if widths is not None:
        finite_widths = np.isfinite(widths)
        result['finite_width_fraction'] = float(finite_widths.mean())
        result['mean_width'] = float(widths.mean()) if finite_widths.all() else None
        result['mean_finite_width'] = float(widths[finite_widths].mean()) if finite_widths.any() else None
    if reference is not None:
        result['width_reference'] = reference
    return result


def evaluate_upper(y, upper, *, median=None, valid=None, bootstrap=0, confidence=.95, seed=0):
    """Report simultaneous/point coverage and efficiency above a frozen median."""
    y = _array(y)
    high = _matching(upper, y.shape)
    mask = _valid(valid, y.shape)
    _finite(y, mask)
    if np.isnan(high[mask]).any() or np.isneginf(high[mask]).any():
        raise ValueError('Upper forecasts must be finite or +infinity.')
    center = None if median is None else _matching(median, y.shape)
    if center is not None:
        _finite(center, mask)
        if (center[mask] > high[mask]).any():
            raise ValueError('Median exceeds upper forecast.')
    return _band_metrics(y,y>high,mask,None if center is None else high-center,
                         None if center is None else 'upper_minus_median',bootstrap,confidence,seed)


def evaluate_interval(y, lower, upper, *, valid=None, bootstrap=0, confidence=.95, seed=0):
    """Report simultaneous and pointwise two-sided coverage and interval width."""
    y = _array(y)
    low, high = (_matching(v,y.shape) for v in (lower,upper))
    mask = _valid(valid,y.shape)
    _finite(y,mask)
    if (np.isnan(low[mask]).any() or np.isnan(high[mask]).any() or np.isposinf(low[mask]).any()
            or np.isneginf(high[mask]).any() or (low[mask] > high[mask]).any()):
        raise ValueError('Invalid interval endpoints.')
    return _band_metrics(y,(y>high)|(y<low),mask,high-low,'upper_minus_lower',bootstrap,confidence,seed)


def _probability(probability, shape, valid):
    p = _matching(probability,shape)
    _finite(p,valid)
    if (p[valid] < -1e-12).any() or (p[valid] > 1+1e-12).any():
        raise ValueError('Failure probabilities must lie in [0,1].')
    return np.clip(p,0,1)


def evaluate_decisions(y, accepted, *, probability=None, valid=None, bootstrap=0, confidence=.95, seed=0,
                       bins=(0.,.10,.20,.40,.60,.80,1.)):
    """Keep abstention, useful accepted feasible requests and failures separate.

    Intervals are descriptive percentile bootstrap intervals clustered by mother.
    They are distinct from the finite-sample selective bound on calibration data.
    """
    y = _array(y)
    mask = _valid(valid,y.shape)
    _finite(y,mask)
    selected = np.asarray(accepted)
    if selected.shape != y.shape or selected.dtype.kind != 'b':
        raise ValueError('Acceptance mask must be Boolean and match chosen margins.')
    use = _flat(selected & mask)
    fail = _flat((y>0) & mask)
    bad = use & fail
    requests = _flat(mask).sum(axis=1)
    accept = use.sum(axis=1)
    active = accept>0
    failed = bad.sum(axis=1)
    one = np.ones(len(y),dtype=int)
    components = {
        'accepted_request_fraction':(accept,requests),
        'successful_request_fraction':(accept-failed,requests),
        'active_mother_fraction':(active,one),
        'any_unsafe_mother_fraction':(failed>0,one),
        'unsafe_fraction_among_active_mothers':(failed>0,active),
        'pooled_unsafe_fraction_among_accepted_requests':(failed,accept),
        'unrestricted_chosen_failure_fraction':(fail.sum(axis=1),requests),
    }
    reliability = None
    if probability is not None:
        p = _probability(probability,y.shape,mask)
        brier = _flat(np.where(mask,(p-(y>0))**2,0)).sum(axis=1)
        components['chosen_failure_brier'] = (brier, requests)
        edges = np.asarray(bins,dtype=float)
        if edges.ndim != 1 or len(edges)<2 or edges[0]!=0 or edges[-1]!=1 or (np.diff(edges)<=0).any():
            raise ValueError('Reliability bins must increase strictly from zero to one.')
        reliability = []
        for i,(left,right) in enumerate(zip(edges[:-1],edges[1:],strict=True)):
            present = mask & (p>=left) & ((p<=right) if i==len(edges)-2 else (p<right))
            reliability.append({'lo':float(left),'hi':float(right),'requests':int(present.sum()),
                                'mothers':int(_flat(present).any(axis=1).sum()),
                                'mean_risk':float(p[present].mean()) if present.any() else None,
                                'observed_failure':float((y[present]>0).mean()) if present.any() else None})
    result = _ratios(components,bootstrap,confidence,seed)
    result.update({'mothers':len(y),'requests':int(requests.sum()),'active_mothers':int(active.sum()),
                   'unsafe_mothers':int((failed>0).sum()),'accepted_requests':int(accept.sum()),
                   'unsafe_accepted_requests':int(failed.sum())})
    if reliability is not None:
        result['probability_reliability'] = reliability
    return result


def fit_selective_group(y, probability, *, cutoffs, max_risk, delta=.05, valid=None, min_active=1):
    """Certify a fixed family of probability thresholds by exact binomial bounds.

    A mother is active if any request is accepted, and unsafe if any such request
    fails. Conditional on the activity count, the failures are binomial under
    IID mothers for each frozen threshold. Bonferroni across all candidates gives
    confidence 1-delta on P(unsafe | active) for the highest certified cutoff.
    No active data means no certificate. If nothing passes, deployment abstains.
    """
    if not 0<max_risk<1 or not 0<delta<1:
        raise ValueError('Risk target and confidence failure probability must be in (0,1).')
    if int(min_active)!=min_active or min_active<1:
        raise ValueError('Minimum active mothers must be a positive integer.')
    threshold = np.asarray(tuple(cutoffs),dtype=float)
    if (threshold.ndim!=1 or not len(threshold) or not np.isfinite(threshold).all()
            or (threshold<0).any() or (threshold>1).any() or (np.diff(threshold)<=0).any()):
        raise ValueError('Declare a nonempty, strictly increasing grid in [0,1].')
    y = _array(y)
    mask = _valid(valid,y.shape)
    _finite(y,mask)
    p = _flat(_probability(probability,y.shape,mask))
    ok = _flat(mask)
    fail = _flat(y>0)
    tail = delta/len(threshold)
    rows = []
    for limit in threshold:
        accepts = ok & (p<=limit)
        active = int(accepts.any(axis=1).sum())
        unsafe = int((accepts & fail).any(axis=1).sum())
        bound = float(beta.isf(tail, unsafe+1, active-unsafe)) if active>unsafe else 1.
        rows.append({'cutoff':float(limit),'active_mothers':active,'unsafe_mothers':unsafe,
                     'accepted_requests':int(accepts.sum()),'upper_risk':bound,
                     'certified':bool(active>=min_active and bound<=max_risk)})
    chosen = next((row for row in reversed(rows) if row['certified']),None)
    return {'cutoff':None if chosen is None else chosen['cutoff'],'certified':chosen is not None,
            'max_risk':float(max_risk),'delta':float(delta),'per_cutoff_tail':float(tail),
            'n_groups':len(y),'min_active':int(min_active),'candidates':rows}
