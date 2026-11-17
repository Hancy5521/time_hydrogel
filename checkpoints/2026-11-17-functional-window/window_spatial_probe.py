"""Equal-count local and bulk snapshots for process-dependent material design."""

import numpy as np
from scipy.special import softmax

from patch_exchange_forward import _hydraulic_path
from patch_reservoir_face_bank import _hydraulic_bank
from patch_reservoir_joint_bank import joint_bank, prepared_fields
from reservoir_coupon_likelihood import Read, infer, observation_table
from window_risk_cases import packet
from window_risk_forecasts import choose, elapsed, margins, prior, summary


def observed_zone(profile, band):
    """Average a finite sensed layer, avoiding dependence on a mesh node."""
    edges=np.linspace(0,1,profile.shape[-1]+1)
    weight=np.maximum(0,np.minimum(edges[1:],band[1])-np.maximum(edges[:-1],band[0]))
    return profile@(weight/(band[1]-band[0]))


def water_signals(profile,settings):
    return np.array([profile.mean(axis=-1),[observed_zone(h,b) for h,b in
                                          zip(profile,settings['bands'],strict=True)]])


def plausible_profiles(known, action, cfg, ticks, cells, dt):
    _,water,*_= _hydraulic_bank(known,[action],cfg,1.,cells,ticks[-1],dt)
    return water[0,2*ticks]


def observed_profiles(group,task,plan):
    s=plan['spatial_probe']
    cells,dt=plan['truth_mesh']
    ticks=np.rint(np.asarray(s['times'])/dt).astype(int)
    action=task['actions'][plan['fixed_action']]
    cfg=task['mechanism']
    known=group['known']
    if not group['shift']:
        return plausible_profiles(known,action,cfg,ticks,cells,dt)
    ex=plan['exchange']
    dx=known['thickness']/cells
    transfer=action['wet_transfer']
    cap=max(2*known['water_D']/dx**2,known['water_D']/dx**2+transfer/dx,cfg['bond_on']+cfg['bond_off'])
    n=max(1,int(np.ceil(dt*cap/.92)))
    parameters=(ex['water_dry_ratio'],ex['water_wet_ratio'],ex['water_power'])
    history,*_=_hydraulic_path(cells,ticks[-1]+1,n,dt,dx,known['water_D'],parameters,
                               (cfg['bond_on'],cfg['bond_off']),transfer,1.)
    return history[ticks]


def changed_mechanism(group,plan):
    """Change independent truth generation; expose only preregistered readouts."""
    if not group['shift']:
        return plan
    s=plan['spatial_probe']
    rng=np.random.default_rng(s['shift_variation_seed']+group['i'])
    values=rng.uniform(*np.asarray(s['unseen_hydraulics']).T)
    return {**plan,'exchange':{**plan['exchange'],'water_dry_ratio':float(values[0]),'water_power':float(values[1])}}


def conditional_release(observations, parameters, plan):
    probes=plan['probes']
    all_cfg={'hydration_times':probes['hydration_times'],'tau_grid':plan['tau_grid'],'coupon_loss_factor':4.,
             'sigma':{'hydration':probes['sigma_hydration'],'total':probes['sigma_release']}}
    conditions=[(r.activity,r.stop) for r in observations[0]]
    kinetics=np.column_stack((parameters[:,:2],np.zeros(len(parameters))))
    means=observation_table(kinetics,all_cfg,[a for a,_ in conditions],conditions)
    legacy=np.array([infer(x,means,all_cfg)[0] for x in observations])
    short=[[Read(r.activity,r.stop,r.hydration[:1],r.total,r.alternate_total,r.bath) for r in x] for x in observations]
    early={a:y[:,:1] for a,y in means[0].items()}
    updated=np.array([infer(x,(early,means[1]),all_cfg)[0] for x in short])
    return legacy,updated


def posterior(m,weights,plan):
    choose_action,risk=choose(m,weights,plan['risk_weight'])
    mean,quantiles=summary(m,weights)
    return {'selected':choose_action.tolist(),'probability':risk.tolist(),'margin_mean':mean.tolist(),
            'margin_q':quantiles.tolist()}


def forecast_spatial(group,task,plan):
    s=plan['spatial_probe']
    observations,half,features=packet(group,plan,task)
    cells,dt=plan['prediction_mesh']
    t=elapsed(task,plan)
    ticks=np.rint(np.asarray(s['times'])/dt).astype(int)
    action=task['actions'][plan['fixed_action']]
    spatial_noise=np.random.default_rng(s['snapshot_noise_seed']+group['i']+plan['seeds']['shift_test' if group['shift'] else 'test']).normal(size=(2,2))
    signal=water_signals(observed_profiles(group,task,changed_mechanism(group,plan)),s)
    observed=signal[:,None,:]+s['sigma']*spatial_noise[None]
    marg=[]
    signals=[]
    atoms=prior(group['known'],{**plan,'continuum_count':s['atoms_per_water']},True)
    hpower=atoms[:,4:]
    stock=prepared_fields(group['known'],[action],task['mechanism'],hpower,cells)[:,0]
    stock_half=stock[:,:cells//2].sum(axis=-1)/stock.sum(axis=-1)
    dry_lik=-.5*((half[:,None]-stock_half)/plan['probes']['sigma_half'])**2
    local,updated=conditional_release(observations,atoms,plan)
    for scale in s['water_grid']:
        known={**group['known'],'water_D':group['known']['water_D']*scale}
        surfaces=[joint_bank(known,task['actions'],task['mechanism'],b,atoms[:,:4],hpower,cells,dt,t)
                  for b in (task['bath']['wet'],task['bath']['weak'])]
        marg.append(margins(*surfaces,t,task))
        expected=water_signals(plausible_profiles(known,action,task['mechanism'],ticks,cells,dt),s)
        signals.append(-.5*(((observed-expected[:,None,:])/s['sigma'])**2).sum(axis=-1))
    responses=np.concatenate(marg)
    ll=np.stack(signals,axis=-1)
    n=len(s['water_grid'])
    with np.errstate(divide='ignore'):
        old_log=np.log(local)+dry_lik
        new_log=np.log(updated)+dry_lik
    weights={
        'library_legacy':softmax(np.tile(old_log,(1,n)),axis=-1),
        'library_bulk':softmax((new_log[:,None,:]+ll[0,:,:,None]).reshape(2,-1),axis=-1),
        'library_depth':softmax((new_log[:,None,:]+ll[1,:,:,None]).reshape(2,-1),axis=-1),
    }
    results={key:posterior(responses,value,plan) for key,value in weights.items()}
    for name,w in weights.items():
        results[name]['hydraulic_scale_mean']=(w.reshape(2,n,-1).sum(axis=-1)@s['water_grid']).tolist()
    return {'forecasts':results,'features':features.tolist(),'local_readings':observed.tolist()}
