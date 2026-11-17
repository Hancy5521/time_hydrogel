"""Physical ensemble forecasts and a time-marginal-preserving negative control."""

from itertools import product

import numpy as np
from scipy.stats import qmc

from audit_patch_reservoir_study import extremes, normalized_clauses
from patch_reservoir_joint_bank import joint_bank, prepared_fields
from reservoir_coupon_likelihood import infer, observation_table
from reservoir_joint_observation import stock_fractions, updated_weights
from window_risk_cases import packet


def elapsed(task,plan):
    step=plan['sampling_dt']
    return np.arange(round(task['scoring']['horizon']/step)+1)*step


def prior(known,plan,continuous):
    if not continuous:
        return np.array(list(product(np.asarray([.52,1.52])*known['release'],[1.24,1.80,2.36],
                                     [.030,.120],[0.,1.0],[-.82,.82],[-.67,.67])))
    count=plan['continuum_count']
    order=int(np.log2(count))
    if 2**order!=count:
        raise ValueError('The conditional atlas uses a power-of-two quasi-Monte Carlo grid.')
    point=qmc.Sobol(7,scramble=True,seed=99177).random_base2(order)
    ranges=plan['prior_spans']
    labels=('power','loss','interface','bias','shape')
    limits=np.array([ranges[key] for key in labels])
    slow,fast=np.asarray([ranges['rate_slow'],ranges['rate_fast']])
    low,high=np.where(point[:,0,None]<.5,slow,fast).T
    rate=(low+(high-low)*point[:,1])*known['release']
    return np.column_stack((rate,limits[:,0]+np.diff(limits,axis=1).ravel()*point[:,2:]))


def choice_weights(group,task,plan,candidates):
    obs=plan['probes']
    cells=plan['prediction_mesh'][0]
    action=[task['actions'][plan['fixed_action']]]
    depth=stock_fractions(prepared_fields(group['known'],action,task['mechanism'],candidates[:,4:],cells)[:,0])
    half_grid=depth[:,:4].sum(axis=1)
    keys=[(obs['low'],obs['last']),(obs['low'],obs['first']),(obs['high'],obs['last'])]
    cfg={'tau_grid':plan['tau_grid'],'hydration_times':obs['hydration_times'],'coupon_loss_factor':4.,
         'sigma':{'hydration':obs['sigma_hydration'],'total':obs['sigma_release'],'extra':obs['sigma_release']}}
    unique,inverse=np.unique(candidates[:,:2],axis=0,return_inverse=True)
    means=observation_table(np.column_stack((unique,np.zeros(len(unique)))),cfg,[obs['low'],obs['high']],keys)
    output={}
    for name,span in (('late',False),('spanning',True)):
        observed,dry,features=packet(group,plan,task,spanning=span)
        likelihood=np.asarray([infer(reads,means,cfg)[0][inverse] for reads in observed])
        output[name]=(updated_weights(likelihood,half_grid,dry,obs['sigma_half']),features)
    return output,depth


def summary(values,weights,quantiles=(.05,.50,.90,.95)):
    """Weighted quantiles and mean for an ensemble with any trailing shape."""
    shape=values.shape[1:]
    flat=values.reshape(len(values),-1)
    order=np.argsort(flat,axis=0,kind='stable')
    sorted_values=np.take_along_axis(flat,order,axis=0)
    cumulative=np.cumsum(weights[:,order],axis=1)
    index=np.stack([(cumulative<q).sum(axis=1).clip(0,len(values)-1) for q in quantiles])
    selected=sorted_values[index,np.arange(flat.shape[1])]
    return (weights@flat).reshape((len(weights),*shape)),selected.reshape((len(quantiles),len(weights),*shape))


def margins(wet,weak,t,task):
    return normalized_clauses(extremes((wet,weak),t,task),task).max(axis=-1)


def choose(m,weight,risk_weight):
    cost=(m>0)+risk_weight*np.maximum(m,0)
    selected=np.einsum('sh,hwa->swa',weight,cost).argmin(axis=-1)
    probability=np.einsum('sh,hwa->swa',weight,m>0)
    return selected,probability


def dependence_control(wet,weak,weights,t,task,plan,seed):
    """Each output keeps all time-specific empirical distributions exactly."""
    n=plan['dependence_count']
    uniform=np.full((1,n),1/n)
    rng=np.random.default_rng(seed)
    output={name:{'selection':[],'failure':[]} for name in ('paired','time_permuted')}
    max_marginal_error=0.
    for w in weights:
        idx=np.searchsorted(np.cumsum(w),(np.arange(n)+.5)/n).clip(0,len(w)-1)
        original=(wet[idx],weak[idx])
        permutations=np.stack([rng.permutation(n) for _ in t],axis=1)
        shuffled=tuple(np.take_along_axis(s,permutations[:,:,None,None],axis=0) for s in original)
        max_marginal_error=max(max_marginal_error,max(float(np.abs(np.sort(a,axis=0)-np.sort(b,axis=0)).max())
                                                      for a,b in zip(original,shuffled,strict=True)))
        for name,fields in zip(output,(original,shuffled),strict=True):
            action,p=choose(margins(*fields,t,task),uniform,plan['risk_weight'])
            output[name]['selection'].append(action[0].tolist())
            output[name]['failure'].append(p[0].tolist())
    return {'arms':output,'marginal_error':max_marginal_error}


def foresee(group,task,plan,diagnose=True):
    t=elapsed(task,plan)
    cells,dt=plan['prediction_mesh']
    results={}
    trace_clock=np.rint(np.array(plan['times_diagnostic'])/plan['sampling_dt']).astype(int)
    fixed=plan['fixed_action']
    source=None
    for continuous in (False,True):
        name='continuum' if continuous else 'discrete'
        candidate=prior(group['known'],plan,continuous)
        reads,depth=choice_weights(group,task,plan,candidate)
        for label,activity in task['bath'].items():
            answer=joint_bank(group['known'],task['actions'],task['mechanism'],activity,candidate[:,:4],candidate[:,4:],
                              cells,dt,t,inventory=label=='wet')
            if label=='wet':
                wet,stock=answer
            else:
                weak=answer
        raw=margins(wet,weak,t,task)
        track=np.concatenate((stock[:,trace_clock,fixed],wet[:,trace_clock,fixed,3:4]),axis=-1)
        for channel,(weights,features) in reads.items():
            if channel=='spanning' and not continuous:
                continue
            key=name if channel=='late' else 'continuum_spanning'
            act,prob=choose(raw,weights,plan['risk_weight'])
            ml,mq=summary(raw,weights)
            dm,dq=summary(depth,weights)
            pm,pq=summary(track,weights)
            results[key]={'selected':act.tolist(),'probability':prob.tolist(),'margin_mean':ml.tolist(),'margin_q':mq.tolist(),
                          'dry_mean':dm.tolist(),'dry_q':dq.tolist(),'process_mean':pm.tolist(),'process_q':pq.tolist(),
                          'profile_support':[depth.min(axis=0).tolist(),depth.max(axis=0).tolist()],
                          'ess':(1/(weights**2).sum(axis=1)).tolist(), 'forward_candidates':len(candidate)}
            if key=='continuum':
                source=features
                if diagnose:
                    results['time_control']=dependence_control(wet,weak,weights,t,task,plan,735127+group['i'])
    return {'parent':group['parent'],'features':source.tolist(),'forecasts':results}
