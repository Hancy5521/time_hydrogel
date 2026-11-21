"""Persistent legal suffix groups for blinded third-frame assimilation."""
from dataclasses import dataclass
from itertools import product
from math import log
from operator import index

import numpy as np
from scipy.special import logsumexp


THIRD_BETAS = (0.,1./3.,2./3.,1.)
PARTICLES = 16
SWEEPS = {1:5,4:1}
EQUAL_SUFFIX_NATIVE_BUDGET = 256


@dataclass(frozen=True,slots=True)
class SuffixGroup:
    """Exact attached payloads must remain read-only in downstream callers."""
    parent: int
    tails: tuple
    log_camera: tuple
    log_potential: float


def attach(parent,tails,log_camera):
    """Keep the actual complete group, never replace a rejected weight estimate."""
    parent=index(parent); tails=tuple(tails); ll=np.asarray(log_camera,float)
    n=len(tails)
    if parent < 0 or n not in SWEEPS or ll.shape!=(n,) or not np.isfinite(ll).all() or np.any(ll>0.):
        raise ValueError('Attach one group of 1 or 4 actual same-parent favorable camera factors.')
    return SuffixGroup(parent,tails,tuple(float(v) for v in ll),float(logsumexp(ll)-log(n)))


def prior_parent(weights,uniform):
    """Draw a parent from the fixed empirical first-two-frame law."""
    w=np.asarray(weights,float)
    if (w.ndim!=1 or not len(w) or not np.isfinite(w).all() or np.any(w<0.) or not w.sum()>0.
            or not 0.<=uniform<1.):
        raise ValueError('Need a complete nonnegative early-bank law and a uniform in [0,1).')
    cumulative=np.cumsum(w/w.sum()); cumulative[-1]=1.
    return int(np.searchsorted(cumulative,uniform,side='right'))


def increments(groups,previous_beta,next_beta):
    """Log incremental SMC weights for a fixed change of third-frame beta."""
    if not 0.<=previous_beta<=next_beta<=1.:
        raise ValueError('Use the ordered beta bridge into the predefined third-frame target.')
    return (next_beta-previous_beta)*np.asarray([g.log_potential for g in groups])


def independent_move(old,new,beta,uniform):
    """A proposed *entire group* is drawn from the fixed early/suffix prior."""
    if len(old.tails)!=len(new.tails) or not 0.<=beta<=1. or not 0.<=uniform<1.:
        raise ValueError('Keep the group size and valid beta/uniform within one chain.')
    alpha = min(0.,beta*(new.log_potential-old.log_potential))
    accepted = bool(uniform==0. or log(uniform)<alpha)
    return (new if accepted else old),accepted,alpha


def pick_tail(group,uniform):
    """At beta=1, project the retained group onto a single legal posterior tail."""
    weights=np.exp(np.asarray(group.log_camera)-logsumexp(group.log_camera))
    j=prior_parent(weights,uniform)
    return j,group.tails[j],weights


def fixture():
    early=np.array([.61,.39]); suffix=np.array([[.22,.78],[.83,.17]])
    camera=np.array([[.15,.76],[.52,.08]])
    exact=early[:,None]*suffix*camera
    exact/=exact.sum()
    found=[]
    for n in (1,4):
        groups=[]; prior=[]
        for step in range(2):
            for path in product((0,1),repeat=n):
                group=attach(step,path,np.log(camera[step,list(path)]))
                groups.append(group)
                prior.append(early[step]*np.prod(suffix[step,list(path)]))
        prior=np.asarray(prior)
        np.testing.assert_allclose(prior.sum(),1.,rtol=0.,atol=2.e-15)
        potential=np.asarray([g.log_potential for g in groups])
        maxima=[]
        for start,beta in zip(THIRD_BETAS,THIRD_BETAS[1:]):
            np.testing.assert_allclose(increments(groups,start,beta),(beta-start)*potential,rtol=0.,atol=0.)
            want=prior*np.exp(beta*potential); want/=want.sum()
            K=np.zeros((len(prior),len(prior)))
            for i,old in enumerate(groups):
                for j,new in enumerate(groups):
                    if i!=j:
                        _,_,alpha=independent_move(old,new,beta,.63)
                        K[i,j]=prior[j]*np.exp(alpha)
                K[i,i]=1.-K[i].sum()
            flux=want[:,None]*K
            maxima.append(float(np.abs(flux-flux.T).max()))
            np.testing.assert_allclose(want@K,want,rtol=0.,atol=2.e-15)
        projected=np.zeros((2,2))
        for g,w in zip(groups,want,strict=True):
            i,x,source=pick_tail(g,.5)
            if x!=g.tails[i]:
                raise AssertionError('Selected tail is not attached to retained group.')
            for outcome,p in zip(g.tails,source,strict=True):
                projected[g.parent,outcome]+=w*p
        discrepancy=float(np.abs(projected-exact).max())
        if discrepancy>2.e-15 or max(maxima)>2.e-15:
            raise AssertionError('Grouped target balance or complete projection differs from direct posterior.')
        expected=PARTICLES*n*(1+SWEEPS[n]*(len(THIRD_BETAS)-1))
        assert expected==EQUAL_SUFFIX_NATIVE_BUDGET
        found.append({'group_size':n,'states_exhausted':len(groups),'maximum_marginal_error':discrepancy,
                      'maximum_balance_error':max(maxima),'registered_suffix_cost':expected})
        old=attach(0,range(n),[-.1]*n); new=attach(1,range(n),[-5.]*n)
        separate,accepted,_=independent_move(old,new,1.,.8)
        require=separate is old and not accepted
        if not require:
            raise AssertionError('A rejection did not preserve the old augmented state identity.')
    print('PASS finite enumeration: exact expanded target, registered equal native budget and maintained suffix groups',found)


if __name__=='__main__':
    fixture()
