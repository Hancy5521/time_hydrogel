"""Count-stratified risks and lawful continuous V5 feed actions."""

from itertools import combinations, permutations
from math import comb
from operator import index
from time import perf_counter

import numpy as np
from numpy.polynomial import Polynomial

from aam_formation_actual_rollout import (
    SCHEMA as RAW_SCHEMA, _packet_copy, packet_fingerprint, validate_rollout_inputs,
)
from aam_formation_cycle_core import SCHEMA as CORE_SCHEMA, _advance, _cost, _physical_config, future_clock
from aam_formation_cycle_data import property_readout
from aam_formation_streamed_rollout import _explicit_seeds
from aam_snapshot_process import PORTS, values


class UniformTieError(ValueError):
    """A single original scalar threshold cannot split tied uniform cells."""

    def __init__(self, seed, draws):
        self.placement_seed = seed
        self.uniforms = draws.copy()
        super().__init__(f'Exact-count rank routing unavailable for placement seed {seed}: {draws.tolist()!r}')


def rank_thresholds(placement_seed, count=6):
    """Return legal fixed-count placements without advancing any reaction."""
    seed, count = index(placement_seed), index(count)
    if not 0 <= seed < 2**32 or count <= 0:
        raise ValueError('The native placement draw uses a valid uint32 seed and nonempty pending stock.')
    uniforms = np.random.default_rng(seed).random(count)
    order = np.argsort(uniforms,kind='stable')
    ordered = uniforms[order]
    if np.any(np.diff(ordered)==0.):
        raise UniformTieError(seed,uniforms)
    # Native feed tests u<p. For 1≤k<n choose the upper order statistic,
    # not a midpoint which can round to the wrong side at tiny spacing.
    thresholds = np.r_[0.,ordered[1:],1.]
    masks = uniforms[None,:]<thresholds[:,None]
    np.testing.assert_array_equal(masks.sum(axis=1),np.arange(count+1))
    return {'uniforms':uniforms,'order':order,'thresholds':thresholds,'left_count_masks':masks}


def binary53_effective_probability(p):
    """Probability of U<p for a uniform integer U*2**53, with ideal role independence."""
    p = np.asarray(p,float)
    if not np.isfinite(p).all() or np.any((p<0.)|(p>1.)):
        raise ValueError('A legal left-port probability belongs to [0,1].')
    return np.ceil(p*(1<<53))/(1<<53)


def bernstein_weights(p, count=6):
    p = np.asarray(p,float)
    n = index(count)
    if n < 1 or not np.isfinite(p).all() or np.any((p<0.)|(p>1.)):
        raise ValueError('Need a valid left-port probability and pending stock count.')
    k = np.arange(n+1)
    return np.asarray([comb(n,i) for i in k]) * p[...,None]**k * (1.-p[...,None])**(n-k)


def normalized_rows(loss, weights):
    loss = np.asarray(loss,float)
    if loss.ndim==1:
        loss=loss[None,:]
    if loss.ndim!=2 or not len(loss) or loss.shape[1]<2 or not np.isfinite(loss).all():
        raise ValueError('One finite conditional loss for each exact count and common seed is required.')
    w = np.ones(len(loss)) if weights is None else np.asarray(weights,float)
    if w.shape!=(len(loss),) or not np.isfinite(w).all() or np.any(w<0.) or not w.max()>0.:
        raise ValueError('Use one nonnegative mass for each independently addressed conditional row.')
    w=w/w.max()
    return loss,w/w.sum()


def bernstein_minimum(conditional_losses,row_weights=None):
    """Global continuous-p minimum, up to numerical polynomial root precision."""
    loss, w = normalized_rows(conditional_losses,row_weights)
    coefficients = w@loss
    n=len(coefficients)-1
    basis=Polynomial((0.,1.))
    differences=np.diff(coefficients)
    deriv=sum((n*comb(n-1,i)*d*basis**i*(1.-basis)**(n-1-i) for i,d in enumerate(differences)),
              start=Polynomial((0.,))).trim()
    roots=deriv.roots() if deriv.degree()>0 else np.empty(0)
    stationary=[float(z.real) for z in roots if abs(z.imag)<=2e-7 and 0.<z.real<1.]
    candidates = np.unique([0.,*stationary,1.])
    risks = bernstein_weights(candidates,n)@coefficients
    i=int(np.argmin(risks))
    return {'p':float(candidates[i]),'risk':float(risks[i]),'candidates':candidates,'risks':risks,
            'coefficients':coefficients}


def threshold_sweep_minimum(uniforms,conditional_losses,row_weights=None):
    """Exactly minimize the shared finite-sample SAA step curve over p in [0,1]."""
    loss,w = normalized_rows(conditional_losses,row_weights)
    u=np.asarray(uniforms,float)
    if u.shape!=(len(loss),loss.shape[1]-1) or not np.isfinite(u).all() or np.any((u<0.)|(u>=1.)):
        raise ValueError('Use one original pre-placement uniform per late particle and one loss for every exact count.')
    ordered=np.sort(u,axis=1)
    if np.any(np.diff(ordered,axis=1)==0.):
        raise ValueError('A tied per-route uniform cannot provide the same seven threshold outcomes.')
    flat=ordered.ravel()
    changes=(w[:,None]*np.diff(loss,axis=1)).ravel()
    order=np.argsort(flat,kind='stable')
    sorted_u=flat[order]
    unique,positions=np.unique(sorted_u,return_index=True)
    jump=np.add.reduceat(changes[order],positions)
    path=np.r_[w@loss[:,0],w@loss[:,0]+np.cumsum(jump)]
    # Between two successive uniforms the exact objective is constant.
    candidates=np.r_[0.,np.nextafter(unique,1.)]
    i=int(np.argmin(path))
    p=float(candidates[i])
    # Recompute the selected threshold directly to avoid carrying cumulative rounding.
    count=(u<p).sum(axis=1)
    risk=float(w@loss[np.arange(len(u)),count])
    return {'p':p,'risk':risk,'candidates':candidates,'risks':path}


def continuous_feed_rollout(actual_packet,action,config,seed,*,native_seeds,placement_seed,mother_id=None):
    """Use the frozen V5 kernel for a newly generated continuous (time,left probability)."""
    started=perf_counter()
    _physical_config(config,actual_packet[4])
    action,prefix,future=validate_rollout_inputs(actual_packet,action,config,seed,mother_id=mother_id)
    if not 2.30<=action[0]<=3.00:
        raise ValueError('Generated intervention time must stay inside the original V5 time bounds.')
    placement=index(placement_seed)
    if not 0<=placement<2**32:
        raise ValueError('Use the original positive uint32 domain for placement.')
    tape=_explicit_seeds(native_seeds)
    original=packet_fingerprint(actual_packet)
    packet=_packet_copy(actual_packet)
    original_packet=_packet_copy(actual_packet,readonly=True)
    absent=~packet[0].present
    packet[1][3]=action[0]
    rng=np.random.default_rng(placement)
    count=int(absent.sum())
    if count!=6 or not np.array_equal(np.flatnonzero(absent),np.arange(72,78)):
        raise ValueError('Generated feed must retain the six original absent MBAA identities.')
    center=np.where(rng.random(count)<action[1],PORTS[0],PORTS[1])
    packet[0].points[absent,0]=(center+rng.normal(0.,.64,count))%packet[2][2]
    packet[0].unwrapped[absent]=packet[0].points[absent]
    beginning=_packet_copy(packet,readonly=True)
    snapshots,calls,cost=[],[],_cost()
    error,phase=None,'prepare'
    try:
        for i,age in enumerate(future):
            clock=future_clock(config,float(age))
            phase='native_progress'
            call={'interval':i,'seed':int(tape[i]),'requested_time':float(age),'clock':clock.copy(),
                  'completed':False,'position_before':packet[0].position.copy(),
                  'counts_before':packet[0].counts.copy()}
            calls.append(call)
            try:
                _advance(packet,clock,call['seed'],False,cost)
                call['completed']=True
            finally:
                call.update(position_after=packet[0].position.copy(),counts_after=packet[0].counts.copy())
            saved={'time':float(age),'packet':_packet_copy(packet,readonly=True),'q':None,'properties':None}
            snapshots.append(saved)
            phase='q_readout'
            start=perf_counter()
            q=np.asarray(values(_packet_copy(packet)),float)
            if q.shape!=(3,) or not np.isfinite(q).all():
                raise ValueError('The original graph readout requires qL, qR and conversion.')
            q.flags.writeable=False
            saved['q']=q
            cost['q_read_seconds']+=perf_counter()-start
            phase='properties_callback'
            start=perf_counter()
            saved['properties']=property_readout(saved['packet'],config)
            cost['property_seconds']+=perf_counter()-start
        phase='complete'
    except Exception as exc:
        error={'phase':phase,'interval':len(calls)-1,'type':type(exc).__name__,'message':str(exc)}
    if packet_fingerprint(actual_packet)!=original:
        raise RuntimeError('The original precursor changed during a generated future reaction.')
    reported=[x['q'] for x in snapshots if x['q'] is not None]
    return {'schema':RAW_SCHEMA,'core_schema':CORE_SCHEMA,'status':'complete' if error is None else 'failed',
            'valid':error is None,'mother_id':mother_id,'seed':int(seed),'action':action,
            'seed_is_compatibility_marker_not_rng':True,'explicit_native_seeds':tape,'explicit_placement_seed':placement,
            'prefix_clock':prefix,'future_times':future,'common_stops':np.asarray(config['common_stops']),
            'mother_packet_sha256':original,'original_packet':original_packet,'branch_start_packet':beginning,
            'packet':_packet_copy(packet,readonly=True),'snapshots':snapshots,'native_calls':calls,
            'q':np.stack(reported) if reported else np.empty((0,3)),'cost':cost,'error':error,
            'elapsed_seconds':perf_counter()-started}


def fixtures():
    from fractions import Fraction

    # Enumerating all rank orders independently of branch physics gives the exact
    # average native conditional loss for every k.
    n=6
    forward=lambda subset: Fraction(2+sum((j+1)**2 for j in subset)+int(0 in subset and 3 in subset),31)
    mean=[sum((forward(x[:k]) for x in permutations(range(n))),Fraction(0))/720 for k in range(n+1)]
    probabilities=[Fraction(0),Fraction(1,7),Fraction(2,11),Fraction(4,5),Fraction(1)]
    for p in probabilities:
        by_count=sum(comb(n,k)*p**k*(1-p)**(n-k)*mean[k] for k in range(n+1))
        direct=sum(p**k*(1-p)**(n-k)*forward(s) for k in range(n+1) for s in combinations(range(n),k))
        assert by_count==direct
        np.testing.assert_allclose(bernstein_weights(float(p))@np.asarray(mean,float),float(direct),atol=1e-14)
    # Degree-elevate a bowl with an interior minimum different from common anchors.
    p=.273
    c=np.array([i*(i-1)/(n*(n-1))-2*p*i/n+p*p for i in range(n+1)])
    answer=bernstein_minimum(c)
    np.testing.assert_allclose([answer['p'],answer['risk']],[p,0.],atol=5e-14)
    assert bernstein_minimum(np.arange(7))['p']==0.
    assert bernstein_minimum(np.arange(7)[::-1])['p']==1.
    rows=[rank_thresholds(i) for i in range(15,24)]
    u=np.stack([a['uniforms'] for a in rows])
    rng=np.random.default_rng(2547)
    y=rng.random((len(rows),7))
    w=rng.random(len(rows))
    w/=w.sum()
    fit=threshold_sweep_minimum(u,y,w)
    for seed,route in enumerate(rows,15):
        exact=np.random.default_rng(seed).random(6)
        np.testing.assert_array_equal(exact,route['uniforms'])
        np.testing.assert_array_equal((exact[None,:]<route['thresholds'][:,None]).sum(axis=1),range(7))
    candidate=np.sort(np.r_[0.,np.nextafter(u.ravel(),1.),1.])
    table=np.stack([y[np.arange(len(u)),(u<x).sum(axis=1)] for x in candidate])
    np.testing.assert_allclose(fit['risk'],np.min(table@w),atol=2e-14)
    close=np.array([0.,np.nextafter(.5,1.),1.])
    assert np.all(binary53_effective_probability(close)>=close)
    print('PASS finite seed count routing, exact labelled-subset Bernstein identity, off-anchor optimum and threshold SAA.')


if __name__=='__main__':
    fixtures()
