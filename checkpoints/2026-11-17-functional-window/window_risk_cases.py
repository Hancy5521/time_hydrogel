"""Frozen synthetic parents and observable packets for the time-window risk study."""

import numpy as np

from patch_reservoir_joint_bank import prepared_fields
from reservoir_coupon_likelihood import Read, coupon_end, wet_prefix
from reservoir_joint_observation import stock_fractions


def specifications():
    sizes = {'train':64, 'development':16, 'probability_calibration':32,
             'risk_calibration':48, 'test':40, 'shift_calibration':48, 'shift_test':40}
    seeds = dict(zip(sizes, (2191153,2191173,2191213,2191229,2191259,2191297,2191327), strict=True))
    return {'scientific_question':'Does extending physically conservative support improve held path inference, and when can calibrated whole-window decisions survive mechanism mismatch?',
            'sizes':sizes,'seeds':seeds,'number_of_states':2,'probes':{'low':.55,'high':1.,'last':.9,'first':.48,
                         'hydration_times':[.11,.24],'sigma_hydration':.016,'sigma_release':.016,'sigma_half':.025},
            'truth_mesh':[48,.0025],'prediction_mesh':[24,.01],'sampling_dt':.04,
            'times_diagnostic':[0.,.8,1.6,2.4,3.2,4.8,6.4],'fixed_action':4,'held_bin':5,
            'risk_weight':.05,'continuum_count':256,'dependence_count':64,
            'prior_spans':{'rate_slow':[.40,.65],'rate_fast':[1.20,1.80], 'power':[1.,2.60],
                           'loss':[.010,.205],'interface':[0.,1.75],'bias':[-1.28,1.28],'shape':[-.88,.88]},
            'tau_grid':[.09,.125,.17,.225,.29,.37,.47,.59],
            'calibration':{'alpha':.10,'confidence':.95,'selective_families':[.035,.08,.15,.24],
                           'selective_target':.25,'scale_floor':.025,
                           'protocol':'Strict family-wise split-conformal over true synthetic mothers; mechanism-transfer scores descriptive. Selective bounds apply to mother level with at least one accepted choice.'},
            'rules':{'budget':'Both exposure designs have two total-release assays, two hydration reads per coupon and one protected half-inventory assay. No bath read. Properties at action choices remain hidden.',
                     'split':'Different mother and noise seeds for supervised training, development, probability calibration, risk calibration, final tests, and separately identified mechanism calibration.',
                     'controls':'Discrete physical model, continuous physical model, trained direct distribution forest and same-physics-feature distribution forest; same features and held labels for algorithm groups. Shared- and shuffled-trajectory predictions preserve per-time empirical marginals.',
                     'shift':'Held generator uses reversible and inhibited release, state-dependent diffusion and a finite bath. Fitting and native calibration use an irreversible generator. Predictions never get the shift flag or its parameters; optional fresh target risk calibration is labeled separately.'}}


def groups(split, plan, task):
    n=plan['sizes'][split]
    seed=plan['seeds'][split]
    rng=np.random.default_rng(seed+772)
    storage=rng.random((n,16))
    names=list(task['known_ranges'])
    lows,highs=np.array(list(task['known_ranges'].values())).T
    common=lows+(highs-lows)*storage[:,:3]
    shift=split.startswith('shift_')
    for i,(x,ctx) in enumerate(zip(storage,common,strict=True)):
        known=dict(zip(names,ctx,strict=True))
        rate=known['release']*np.array([.44+.16*x[3],1.30+.42*x[4]])
        power=1.20+1.18*x[5]
        loss=.025+.142*x[6]
        rho=(.10 if not shift else .31)+(1.05 if not shift else 1.12)*x[7]
        kinetic=np.column_stack((rate,[power,max(1.,power+.22*(2*x[8]-1))], [loss,loss],[rho,rho]))
        reserve=np.array([[1.12*(2*x[9]-1),1.12*(2*x[10]-1)],[.80*(2*x[11]-1),.80*(2*x[12]-1)]]).T
        yield {'parent':f'{split}-{i:03d}','i':i,'known':known,'kinetic':kinetic,'stock':reserve,
               'tau':.11+.13*x[13],'lag':1.15+.66*x[14], 'slow_share':.10+.26*x[15] if shift else 0.,
               'shift':shift,'noise':rng.normal(size=(2,2,3)), 'half_noise':rng.normal(size=2)}


def material_config(task,group,actual):
    return task['mechanism'].copy()


def _external_coupon(kinetic,activity,stop,tau,mixture):
    """Changed observation mechanism on shift: nonlinear wet-rate inhibition.

    This assay predicts mobilized stock, never exports true service parameters.
    It is deliberately only partially matched to the independent service PDE.
    """
    z,w=np.polynomial.legendre.leggauss(64)
    t=stop*(1+z)/2
    h=wet_prefix(t,activity,tau,mixture)
    # Local departure from pure hydration power, with slowly developing inhibition.
    expo=h[None,:]**kinetic[:,1,None]/(1+.45*(1-h[None,:]))
    return -np.expm1(-kinetic[:,0]*stop/2*(expo@w))


def packet(group,plan,task,spanning=False):
    obs=plan['probes']
    data=[]
    for slot,a in enumerate((obs['low'],obs['high'])):
        t=obs['first'] if slot==0 and spanning else obs['last']
        tau=group['tau']*(group['lag'] if slot==0 else 1.)
        mixed=group['slow_share']
        if group['shift']:
            fraction=_external_coupon(group['kinetic'],a,t,tau,mixed)
        else:
            fraction=coupon_end(group['kinetic'][:,:3],a,t,tau,4.,slow_share=mixed,order=64)[:,0]
        h=wet_prefix(np.asarray(obs['hydration_times']),a,tau,mixed)
        eps=group['noise'][:,slot]
        reads=[]
        for i in range(plan['number_of_states']):
            observed=float(fraction[i]+obs['sigma_release']*eps[i,2])
            reads.append(Read(a,t,h+obs['sigma_hydration']*eps[i,:2],observed,observed,0.))
        data.append(reads)
    action=[task['actions'][plan['fixed_action']]]
    stock=stock_fractions(prepared_fields(group['known'],action,material_config(task,group,True),group['stock'],48)[:,0])
    half=stock[:,:4].sum(axis=1)+obs['sigma_half']*group['half_noise']
    pairs=list(zip(*data,strict=True))
    features=np.asarray([[*group['known'].values(),*[a for r in readings for a in (*r.hydration,r.total)],d]
                         for readings,d in zip(pairs,half,strict=True)])
    return pairs,half,features
