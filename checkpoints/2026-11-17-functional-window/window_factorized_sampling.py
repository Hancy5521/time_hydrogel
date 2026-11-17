"""Sample observation-conditioned physical parameters under a fixed path budget."""

import numpy as np
from scipy.special import softmax
from scipy.stats import qmc

from patch_reservoir_joint_bank import joint_bank, prepared_fields
from reservoir_coupon_likelihood import infer, wet_prefix
from window_risk_cases import packet
from window_risk_forecasts import choose, elapsed, margins, summary


def midpoints(interval, count):
    start, stop = interval
    return start+(stop-start)*(np.arange(count)+.5)/count


def release_weights(known, reads, plan):
    """Marginalize hydration time on a resolved rate-by-power quadrature grid."""
    settings=plan['factorization']
    spans=plan['prior_spans']
    rates=np.stack([midpoints(spans[k],settings['rate_nodes']) for k in ('rate_slow','rate_fast')])
    rates=known['release']*rates.ravel()
    powers=midpoints(spans['power'],settings['power_nodes'])
    tau=np.asarray(plan['tau_grid'])
    obs=plan['probes']
    g,k=np.meshgrid(powers,rates,indexing='ij')
    nodes,weights=np.polynomial.legendre.leggauss(24)
    wetting={}
    values={}
    for read in reads[0]:
        a,t=read.activity,read.stop
        wetting[a]=wet_prefix(np.array(obs['hydration_times'])[None,:],a,tau[:,None])
        h=wet_prefix(t*(nodes+1)/2,a,tau[:,None])
        dwell=(h[None,:,:]**g.reshape(-1,1,1))@weights
        total=-np.expm1(-k.reshape(-1,1)*t*dwell/2)
        values[a,t]=total[...,None]
    cfg={'sigma':{'hydration':obs['sigma_hydration'],'total':obs['sigma_release']}}
    p=np.array([infer(read,(wetting,values),cfg)[0].reshape(g.shape) for read in reads])
    return powers,rates,p


def invert(weights, points):
    edge=np.cumsum(weights,axis=-1)
    edge[...,-1]=1.
    return (edge < points[...,None]).sum(axis=-1)


def conditional_atoms(group,task,plan):
    """Take equal-weight QMC paths using the posterior's conditional quantiles."""
    settings=plan['factorization']
    spans=plan['prior_spans']
    count=plan['continuum_count']
    order=int(np.log2(count))
    if 2**order != count:
        raise ValueError('The path count must be a power of two.')
    reads,observed_half,_=packet(group,plan,task)
    powers,rates,kinetics_p=release_weights(group['known'],reads,plan)
    b=midpoints(spans['bias'],settings['loading_nodes'])
    reference=[task['actions'][plan['fixed_action']]]
    profile=prepared_fields(group['known'],reference,task['mechanism'],np.column_stack((b,np.zeros_like(b))),
                            plan['prediction_mesh'][0])[:,0]
    half=profile[:,:profile.shape[-1]//2].sum(axis=-1)/profile.sum(axis=-1)
    dry_p=softmax(-.5*((observed_half[:,None]-half)/plan['probes']['sigma_half'])**2,axis=-1)
    u=qmc.Sobol(6,scramble=True,seed=settings['qmc_seed']).random_base2(order)
    limits=np.array([spans[key] for key in ('loss','interface','shape')])
    loss,rho,shape=(limits[:,0]+u[:,3:]*(limits[:,1]-limits[:,0])).T
    result=[]
    for joint,loading in zip(kinetics_p,dry_p,strict=True):
        j=invert(joint.sum(axis=-1),u[:,0])
        p_k=joint[j]/joint[j].sum(axis=-1,keepdims=True)
        i=invert(p_k,u[:,1])
        bias=b[invert(loading,u[:,2])]
        result.append((np.column_stack((rates[i],powers[j],loss,rho)),np.column_stack((bias,shape))))
    return result


def conditional_forecast(group,task,plan):
    t=elapsed(task,plan)
    outputs={key:[] for key in ('selected','probability','margin_mean','margin_q')}
    cells,dt=plan['prediction_mesh']
    for kinetics,stock in conditional_atoms(group,task,plan):
        wet,weak=[joint_bank(group['known'],task['actions'],task['mechanism'],a,kinetics,stock,cells,dt,t)
                  for a in (task['bath']['wet'],task['bath']['weak'])]
        raw=margins(wet,weak,t,task)
        p=np.full((1,len(raw)),1/len(raw))
        act,risk=choose(raw,p,plan['risk_weight'])
        m,q=summary(raw,p)
        for key,value in zip(outputs,(act[0],risk[0],m[0],q[:,0]),strict=True):
            outputs[key].append(value)
    return {key:np.stack(value,axis=1 if key=='margin_q' else 0).tolist() for key,value in outputs.items()}
