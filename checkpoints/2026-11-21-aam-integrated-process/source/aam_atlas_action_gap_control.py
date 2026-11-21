"""Train-only atlas control variates for legal continuous-action risk estimation."""

from dataclasses import dataclass

import numpy as np
from scipy.stats import t

from aam_bernstein_feed_design import bernstein_weights, binary53_effective_probability


TIMES, COUNTS, GOALS = 6, 7, 2
RIDGE = .50
FOLDS = 4


def _finite(matrix, name):
    array = np.asarray(matrix, dtype=float)
    if not np.isfinite(array).all():
        raise ValueError(f'{name} must be finite.')
    return array


def _positions(value, size, *, unique=False):
    idx = np.asarray(value)
    if (idx.ndim != 1 or not len(idx) or idx.dtype.kind not in 'iu' or np.any((idx<0)|(idx>=size))
            or unique and len(np.unique(idx)) != len(idx)):
        raise ValueError('Use valid physical atlas identity positions.')
    return idx.astype(np.int64, copy=False)


def _time_probability(time_index, probability):
    time = _positions(time_index,TIMES)
    scale = _finite(probability,'Left-feed probability')
    if time.shape != scale.shape or np.any((scale<0.)|(scale>1.)):
        raise ValueError('One valid registered time and allocation probability are needed per control.')
    return time, np.stack([bernstein_weights(binary53_effective_probability(p)) for p in scale])


def route_risk(coeff, time_index, probability):
    """Evaluate the same original Bernstein route risk at fixed controls, by goal."""
    coef = _finite(coeff, 'Route coefficients')
    pause, weight = _time_probability(time_index,probability)
    if coef.shape[-3:] != (TIMES,COUNTS,GOALS) or pause.shape != (GOALS,):
        raise ValueError('One fixed reference per goal on the original 6-by-7 route table is required.')
    return np.stack([coef[...,ti,:,g] @ p for g,(ti,p) in enumerate(zip(pause,weight,strict=True))], axis=-1)


def action_gaps(coeff, anchor_time, anchor_left):
    coef = _finite(coeff,'Original route loss')
    return coef-route_risk(coef,anchor_time,anchor_left)[...,None,None,:]


def _fit(x, targets, queries):
    # Scaling must exclusively use the relevant training fold.
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale == 0.] = 1.
    centered = (x-mean)/scale
    average = targets.mean(axis=0)
    normal = centered.T @ centered / len(x) + RIDGE*np.eye(x.shape[1])
    rhs = centered.T @ (targets-average) / len(x)
    return average + ((queries-mean)/scale) @ np.linalg.solve(normal,rhs)


@dataclass(frozen=True)
class AtlasControl:
    coefficients: np.ndarray
    training_positions: np.ndarray
    anchor_time: np.ndarray
    anchor_left: np.ndarray
    training_oof: np.ndarray
    training_oof_constant: np.ndarray
    training_gaps: np.ndarray


def fit_atlas(legal_features, population_positions, population_losses, anchor_time, anchor_left):
    """Fit a fixed ridge predictor for whole action-gap tables using training mothers only."""
    x = _finite(legal_features,'Frozen low-dimensional atlas features')
    if x.ndim != 2 or not 1 <= x.shape[1] <= 16:
        raise ValueError('Use no more than sixteen externally frozen atlas features.')
    donors = _positions(population_positions,len(x),unique=True)
    raw = _finite(population_losses, 'Independent training population route losses')
    if len(donors) < 2*FOLDS or raw.shape!=(len(donors),TIMES,COUNTS,GOALS):
        raise ValueError('Population rows must map to distinct original training mothers and their complete route tables.')
    anchor_time = _positions(anchor_time,TIMES)
    anchor_left = _finite(anchor_left,'Reference allocation')
    gaps = action_gaps(raw,anchor_time,anchor_left)
    y = gaps.reshape(len(donors),-1)
    predictions = np.empty_like(y)
    constants = np.empty_like(y)
    # Grouped folds use sorted physical positions, independent of all route values.
    fold = np.argsort(np.argsort(donors,kind='stable'),kind='stable')%FOLDS
    for i in range(FOLDS):
        validation = fold == i
        train = ~validation
        predictions[validation] = _fit(x[donors[train]],y[train],x[donors[validation]])
        constants[validation] = y[train].mean(axis=0)
    surrogate = _fit(x[donors], y, x).reshape(len(x),TIMES,COUNTS,GOALS)
    return AtlasControl(surrogate,donors.copy(),anchor_time.copy(),anchor_left.copy(),
                        predictions.reshape(raw.shape),constants.reshape(raw.shape),gaps)


def estimate(control, posterior, draws, actual_losses):
    """Same-atlas unbiased gap coefficients using the same iid legal scenarios."""
    p = _finite(posterior,'Original finite atlas posterior')
    if p.shape!=(len(control.coefficients),) or np.any(p<0.) or not np.isclose(p.sum(),1.,atol=2e-13,rtol=0.):
        raise ValueError('Require an already normalized query-only posterior on the exact atlas.')
    ids = _positions(draws,len(p))
    if np.any(p[ids]<=0.):
        raise ValueError('Every sampled path must have positive original posterior mass.')
    observed = _finite(actual_losses,'Original legal scenario route losses')
    if observed.shape != (len(ids),TIMES,COUNTS,GOALS):
        raise ValueError('Each iid planning scenario requires a complete original legal route table.')
    actual_gap = action_gaps(observed,control.anchor_time,control.anchor_left)
    mean = np.einsum('i,itkg->tkg',p,control.coefficients)
    correction = actual_gap-control.coefficients[ids]
    return {'controlled':mean+correction.mean(axis=0), 'raw':actual_gap.mean(axis=0),
            'surrogate_only':mean,'residual':correction}


def _paired_interval(data):
    a = np.asarray(data, dtype=float)
    n = len(a)
    average = float(np.mean(a))
    width = float(t.isf(.025,n-1) * np.std(a,ddof=1) / np.sqrt(n))
    return {'mean':average,'t95':[average-width,average+width],'pairs':n}


def validate_training_gate(control, anchor_repeat_losses, anchor_actions, *, time_indices=(0,5)):
    """Training-only check on independent mothers and independently seeded legal suffixes.

    The two old repeats must come from every original atlas mother and contain the
    original same six discrete controls. Disjoint pairs remove unknown means without
    treating the several actions/repeats from a mother as independent.
    """
    losses = _finite(anchor_repeat_losses,'Two original training anchor suffixes per atlas mother')
    n = len(control.coefficients)
    base = _positions(anchor_actions,6)
    time, p = _time_probability(np.repeat(time_indices,3),np.tile([0.,.5,1.],2))
    if losses.shape!=(n,2,6,GOALS) or base.shape!=(GOALS,) or len(time)!=6:
        raise ValueError('Require two archived independent suffixes, same atlas and original six training anchors.')
    candidates = np.flatnonzero(~np.isin(np.arange(n),control.training_positions))
    if len(candidates)<16:
        raise ValueError('Reserve enough mothers disjoint from the population surrogate fit.')
    # With the registered 512-48 design all validation mothers form disjoint pairs.
    total = len(candidates)//2*2
    ids, spare = candidates[:total].reshape(-1,2), candidates[total:]
    pred = np.stack([np.einsum('ikg,k->ig',control.coefficients[:,ix],w) for ix,w in zip(time,p,strict=True)],axis=1)
    report = []
    for g,j in enumerate(base):
        keep = np.arange(6)!=j
        real = losses[...,g]-losses[:,:,j,g][:,:,None]
        fit = pred[...,g]-pred[:,j,g][:,None]
        delta = (real[ids[:,0]]-real[ids[:,1]])[:,:,keep]
        forecast = (fit[ids[:,0]]-fit[ids[:,1]])[:,keep]
        between = .5*np.mean(delta[:,0]*delta[:,1],axis=1)
        variance = .5*np.mean(delta**2-(delta-forecast[:,None,:])**2,axis=(1,2))
        report.append({'between_state_gap_covariance':_paired_interval(between),
                       'independent_pair_variance_reduction':_paired_interval(variance),
                       'reference_action_index':int(j)})
    oof_gain = np.mean((control.training_gaps-control.training_oof_constant)**2
                       -(control.training_gaps-control.training_oof)**2,axis=(0,1,2))
    return {'training_oof_mse_difference_by_goal':oof_gain,'validation_pairs':ids,'spare_positions':spare,
            'by_goal':report,
            'both_goals_positive':all(r[k]['t95'][0]>0. for r in report
                 for k in ('between_state_gap_covariance','independent_pair_variance_reduction'))}


def fixtures():
    """Enumerate one conditional law exactly, with additional positive and null gates."""
    sizes=(TIMES,COUNTS,GOALS)
    train_time=np.array([0,5])
    train_p=np.array([.5,.5])
    posterior=np.array([.15,.35,.5])
    toy=np.arange(3*np.prod(sizes),dtype=float).reshape(3,*sizes)/128
    toy+=np.arange(7)[None,None,:,None]*np.arange(3)[:,None,None,None]
    surrogate=action_gaps(.8*toy,train_time,train_p)
    learned=AtlasControl(surrogate,np.arange(3),train_time,train_p,surrogate,surrogate,surrogate)
    outcomes=np.stack((toy-.1*np.arange(7)[None,None,:,None],toy+.1*np.arange(7)[None,None,:,None]),axis=1)
    controlled, raw = np.zeros(sizes),np.zeros(sizes)
    variance=np.zeros(2)
    for state in range(3):
        for finish in range(2):
            v=estimate(learned,posterior,np.array([state]),outcomes[state,finish][None])
            controlled+=posterior[state]*.5*v['controlled']
            raw+=posterior[state]*.5*v['raw']
    np.testing.assert_allclose(controlled,raw,atol=3.e-14)
    for state in range(3):
        for finish in range(2):
            v=estimate(learned,posterior,np.array([state]),outcomes[state,finish][None])
            variance+=posterior[state]*.5*np.array([np.mean((v[route]-raw)**2) for route in ('raw','controlled')])
    if variance[1]>=variance[0]:
        raise AssertionError('Finite informative control did not reduce the action-gap sampling variance.')
    n=48
    x=np.linspace(-1,1,n)[:,None]
    slope=(np.indices(sizes)[1]+np.indices(sizes)[0]/3.)[None]
    exact=2+x[:,:,None,None]*slope
    donors=np.arange(0,n,3)
    control=fit_atlas(x,donors,exact[donors],train_time,train_p)
    full=np.stack([exact[:,ti,k] for ti in (0,5) for k in (0,3,6)],axis=1)
    repeated=np.stack((full,full),axis=1)
    outcome=validate_training_gate(control,repeated,[0,5])
    if not outcome['both_goals_positive']:
        raise AssertionError('Clear between-mother signal did not pass the independent training-pair gate.')
    null=validate_training_gate(control,np.full_like(repeated,2.),[0,5])
    if null['both_goals_positive']:
        raise AssertionError('No state signal passed the gate.')
    return {'exact_unbiased_gap':True,'raw_variance':float(variance[0]),'controlled_variance':float(variance[1]),
            'positive_and_null_train_only_gates':True}


if __name__ == '__main__':
    print(fixtures())
