"""Independent continuous-action inverse-design audit, with no native replay."""
import argparse
from datetime import datetime
import hashlib
from pathlib import Path

import numpy as np
from scipy.interpolate import BPoly, PPoly
from scipy.linalg import cho_factor, cho_solve
from scipy.stats import qmc,t
from threadpoolctl import threadpool_limits

from aam_edge_fork import State
from aam_formation_actual_rollout import STATE_FIELDS, packet_fingerprint
from aam_formation_cycle_data import check_actual_state, independent_q, property_readout
from aam_formation_cycle_driver import np_load,read,sha,write
from aam_formation_micro_history import extract_state
from aam_formation_path_cycle import load_tree,stamp
from aam_snapshot_process import observe


MOTHERS=48
RANGE=(2026123000,2026123048)
PENDING=6
TIMES=(3.25,4.85,7.20)
PORTS=np.array([1.45,4.45])
METHODS=('bernstein','sweep','gp','sobol','bernstein_six','population','fixed')
GOALS=('delayed_support','early_buffer_then_relax')
PLAN_TIMES=np.array([2.30,2.3875,2.5625,2.7375,2.9125,3.00])
ANCHORS=np.array([(tau,p) for tau in (2.30,3.00) for p in (0.,.5,1.)])
NAMESPACE='aam-continuous-unchanged-formation-inverse-count-law-v1'
ROLES={'prior_index':1,'sobol':2,'planning_future':3,'reference_future':4}


def need(condition,why):
    if not condition:
        raise ValueError(why)


def close(a,b,*,atol=3.e-12):
    np.testing.assert_allclose(a,b,atol=atol,rtol=3.e-12)


def same(a,b):
    np.testing.assert_array_equal(a,b)


def bound(receipt,target):
    need(Path(receipt['path']).resolve()==target and sha(target)==receipt['sha256'],
         'A registered artifact moved or its bytes changed.')
    if 'bytes' in receipt:
        need(target.stat().st_size==receipt['bytes'],'Artifact byte count changed.')
    return target


def goal_loss(curves,mech_scale):
    x,s=np.asarray(curves,float),np.asarray(mech_scale,float)
    need(x.shape[-2:]==(3,6) and np.isfinite(x).all() and s.shape==(3,) and np.all(s>0.),
         'Expected original three-age six-channel trajectories and frozen scales.')
    h=x[...,3:]
    need(np.all(h>=0.) and not np.any((h[...,0]==0.)&(h[...,2]!=0.)),
         'The physical terminal short-probe ratio is undefined.')
    ratio=np.divide(h[...,2],h[...,0],out=np.zeros_like(h[...,0]),where=h[...,0]>0.)
    q=(x[...,0]+x[...,1])/2.
    conv=np.maximum(.82-x[...,2,2],0.)**2
    network=np.stack((np.mean((q-[.2,.4,.6])**2,axis=-1),np.mean((q-[.4,.55,.6])**2,axis=-1)),axis=-1)
    delayed=np.mean((h[...,1]/s[1]-[.15,.45,.8])**2,axis=-1)
    early=.5*(np.mean((h[...,0]/s[0]-[.65,.8,.8])**2,axis=-1)+np.mean(np.maximum(ratio-.25,0.)**2,axis=-1))
    blocks=np.stack((network,np.stack((delayed,early),axis=-1),np.repeat(conv[...,None],2,axis=-1)),axis=-1)
    return blocks.mean(axis=-1),blocks


def expected_weights(p):
    """Counterpart to the closed binomial formula via repeated Bernoulli convolution."""
    need(np.isfinite(p) and 0.<=p<=1.,'Port probability outside original physical domain.')
    x=np.ones(1)
    for _ in range(PENDING):
        x=np.convolve(x,[1.-p,p])
    return x


def binary_cell_p(p):
    a,b=float(p).as_integer_ratio()
    cells=1<<53
    return ((a*cells+b-1)//b)/cells


def whole_polynomial(coefficients):
    c=np.asarray(coefficients,float)
    need(c.shape==(7,) and np.isfinite(c).all(),'Every pulse-time stratum has seven finite expected conditional counts.')
    b=BPoly(c[:,None],[0.,1.])
    power=PPoly.from_bernstein_basis(b)
    roots=power.derivative().roots(extrapolate=False)
    candidates=np.sort(np.r_[0.,roots[np.isfinite(roots)&(roots>0.)&(roots<1.)],1.])
    values=b(candidates)
    idx=int(np.argmin(values))
    return float(candidates[idx]),float(values[idx]),b


def exact_sweep(u,loss,p):
    """Brute-force every strict original inequality U<p over this small sample."""
    u,loss=np.asarray(u,float),np.asarray(loss,float)
    n=len(u)
    need(u.shape==(n,6) and loss.shape==(n,7) and np.isfinite(loss).all(),
         'Each shared native trajectory needs six original uniform cells and seven actual losses.')
    cuts=np.r_[0.,np.nextafter(np.unique(u),1.),1.]
    oracle=np.asarray([loss[np.arange(n),(u<c).sum(axis=1)].mean() for c in cuts])
    reported=float(loss[np.arange(n),(u<p).sum(axis=1)].mean())
    return reported,float(oracle.min()),cuts


def allocation_audit(values,uniforms,taus,chosen_b,chosen_s):
    """Cross-check independent global p minima for each raw fixed-time risk table."""
    x=np.asarray(values,float)
    t=np.asarray(taus,float)
    need(x.shape==(4,len(t),7,2) and len(t)==6 and np.isfinite(x).all(),
         'Exactly four shared precursor roles and six by seven lawful branches per goal.')
    for method,controls in (('bernstein',chosen_b),('sweep',chosen_s)):
        need(np.asarray(controls).shape==(2,2),'One exact time and p for each of the two frozen goals.')
        for goal,control in enumerate(controls):
            tau,p=control
            idx=np.flatnonzero(t==tau)
            need(len(idx)==1 and 0.<=p<=1.,'The count method emitted an illegal time or probability.')
            ix=int(idx[0])
            banks=x[...,goal]
            if method=='bernstein':
                curves=[whole_polynomial(bank.mean(axis=0)) for bank in np.moveaxis(banks,1,0)]
                optimum=min(row[1] for row in curves)
                achieved=float(curves[ix][2](p))
            else:
                results=[exact_sweep(uniforms,bank,p) for bank in np.moveaxis(banks,1,0)]
                optimum=min(row[1] for row in results)
                achieved=results[ix][0]
            close(achieved,optimum,atol=4.e-11)


def rbf_weights(atlas,q,setting):
    """Independent direct normalization of the locked atlas lookup."""
    width,mixture=setting
    x=(atlas['x']-q)/atlas['camera_scale']
    d=np.square(x.reshape(len(x),-1)).sum(axis=1)
    expo=-.5*d/(width**2*atlas['kernel_sq_scale'])
    estimated=np.exp(expo-expo.max()); estimated/=estimated.sum()
    return (1.-mixture)*atlas['prior']+mixture*estimated


def matern5_2(a,b,length):
    a,b,l=np.asarray(a,float),np.asarray(b,float),np.asarray(length,float)
    r=np.sqrt(5.*np.sum(((a[:,None]-b[None,:])/l)**2,axis=-1))
    return (1.+r+r*r/3.)*np.exp(-r)


def gp_mean(x,v,grid,length,jitter):
    """Fixed kernel mean, recomputed using a separate Cholesky algorithm."""
    matrix=matern5_2(x,x,length)+jitter*np.eye(len(x))
    m=v.mean(axis=0)
    return m+matern5_2(grid,x,length)@cho_solve(cho_factor(matrix),v-m)


def placements(seed,p,count=6):
    rng=np.random.default_rng(int(seed))
    uniform=rng.random(count)
    offset=rng.normal(0.,.64,count)
    mask=uniform<p
    return uniform,offset,mask


def addressed(unit, role, draw=0, child=0):
    hash_words=np.frombuffer(hashlib.sha256(NAMESPACE.encode()).digest()[:16],dtype='<u4').tolist()
    return np.random.default_rng(np.random.SeedSequence([*hash_words,unit,ROLES[role],draw,child]))


def seed_receipt(unit,j,role):
    return {'seed':addressed(unit,role,j,1).integers(0,2**32-51,dtype=np.uint32),
            'native_seeds':addressed(unit,role,j,2).integers(0,2**32,3,dtype=np.uint32),
            'placement_seed':addressed(unit,role,j,3).integers(0,2**32,dtype=np.uint32)}


def audit_packed(path,expected,controls,streams,origin,packet,physics,*,deep,expected_micro,expected_winding):
    """All branches bind raw histories; an a priori subset gets full readout recomputation."""
    with np.load(path,allow_pickle=False) as raw:
        for name,value in streams.items():
            same(raw[name],value)
        same(raw['controls'],controls)
        same(raw['values'],expected)
        same(raw['origin'],origin)
        n=len(controls)
        same(raw['completed'],np.ones((n,3),bool))
        uniform,_,_=placements(streams['placement_seed'],.5)
        groups={}
        for i,(tau,p) in enumerate(controls):
            key=(tau,*map(bool,uniform<p))
            j=groups.setdefault(key,i)
            close(expected[j],expected[i])
        same(raw['unique_physical_layouts'],len(groups))
        m,w=raw['micro'],raw['winding']
        same(m[:len(expected_micro)],expected_micro)
        same(w[:len(expected_winding)],expected_winding)
        timing=raw['native_timing']
        need(timing.shape==(n,3) and np.isfinite(timing).all() and np.all(timing>=0.),'Invalid measured cost.')
        if not deep:
            return timing.sum(axis=0)
        need(packet is not None and packet_fingerprint(packet)==origin,'Deep branch lacks the original packet.')
        params=[raw['source_'+k] for k in ('material','geometry','rates','factors')]
        for actual,expected_param in zip(params,packet[1:],strict=True):
            same(actual,expected_param)
        packed=[raw['state_'+key] for key in STATE_FIELDS]
        raw_g,cleaned,born,micro,winding=[raw[key] for key in ('raw_g','cleaned_g','born','micro','winding')]
        original=packet[0]
        absent=~original.present
        same(np.flatnonzero(absent),np.arange(72,78))
        for i,(tau,p) in enumerate(controls):
            states=[State(*(x[i,j] for x in packed)) for j in range(4)]
            for field in STATE_FIELDS:
                a,b=getattr(original,field),getattr(states[0],field)
                if field in ('points','unwrapped'):
                    a,b=a[~absent],b[~absent]
                same(a,b)
            _,offset,left=placements(streams['placement_seed'],p)
            pos=original.points[absent].copy()
            pos[:,0]=(PORTS[(~left).astype(int)]+offset)%params[1][2]
            same(pos,states[0].points[absent]);same(pos,states[0].unwrapped[absent])
            material=params[0].copy();material[3]=tau
            for j,(prior,s,age) in enumerate(zip(states[:-1],states[1:],TIMES,strict=True)):
                check_actual_state(s,expected_age=age,full_feed=True)
                for name,count in (('events',prior.counts[0]),('edges',prior.counts[1]),('lifts',prior.counts[1])):
                    same(getattr(s,name)[:count],getattr(prior,name)[:count])
                events=s.events[:int(s.counts[0])]
                feeds=events[events[:,1]==0.]
                close(feeds[:,0],np.full(6,tau))
                current=(s,material,*params[1:])
                mech=property_readout(current,physics)
                need(mech['valid'],'Packed actual mechanical snapshot is invalid.')
                close(raw_g[i,j],mech['raw_dimensionless_finite_G_tau'])
                close(cleaned[i,j],mech['dimensionless_finite_G_tau'])
                close(born[i,j],mech['dimensionless_affine_Born'])
                close(expected[i,j],np.r_[independent_q(s),cleaned[i,j,[1,2,3]]])
                reconstructed,wrap=extract_state(s)
                close(micro[i,j],reconstructed);same(winding[i,j],wrap)
        return timing.sum(axis=0)


def direct_control_reconstruction(unit):
    seed=int(addressed(unit,'sobol').integers(0,2**32,dtype=np.uint32))
    controls=qmc.Sobol(d=2,scramble=True,seed=seed).random_base2(6)[:36]
    controls[:,0]=2.30+.70*controls[:,0]
    return np.vstack((ANCHORS,controls))


def audit_controls(control,uniforms,values,controls_values,choices,losses,mech_scale,fixed,population,gp):
    s,_=goal_loss(values,mech_scale)
    d,_=goal_loss(controls_values,mech_scale)
    allocation_audit(s,uniforms,PLAN_TIMES,choices[0],choices[1])
    for g in range(2):
        for a in range(2):
            loc=np.flatnonzero(PLAN_TIMES==choices[a,g,0])[0]
            p=choices[a,g,1]
            if a==0:
                close(losses[a,g],s[:,loc,:,g].mean(axis=0)@expected_weights(p))
            else:
                close(losses[a,g],exact_sweep(uniforms,s[:,loc,:,g],p)[0])
    averages=d.mean(axis=0)
    length,nugget,grid=gp
    ticks=np.linspace(0.,1.,grid)
    domain=np.unique(np.vstack((np.asarray([(2.30+.70*x,y) for x in ticks for y in ticks]),control)),axis=0)
    scale=np.array([.70,1.]);offset=np.array([2.30,0.])
    prediction=gp_mean((control-offset)/scale,averages,(domain-offset)/scale,length,nugget)
    for a,x,obj in ((2,domain,prediction),(3,control,averages)):
        for g in range(2):
            ix=np.flatnonzero(np.all(x==choices[a,g],axis=1))
            need(len(ix)>=1,'Selected coordinate missing from its frozen search domain.')
            close(obj[ix[0],g],obj[:,g].min())
            close(losses[a,g],obj[ix[0],g])
    risks_six=np.asarray([expected_weights(p)@s[:,0 if tau==2.30 else 5,:,:].mean(axis=0)
                          for tau,p in ANCHORS])
    same(choices[4],ANCHORS[risks_six.argmin(axis=0)])
    close(losses[4],risks_six.min(axis=0))
    same(choices[5],population)
    for g,(tau,p) in enumerate(population):
        j=np.flatnonzero(PLAN_TIMES==tau)[0]
        close(losses[5,g],s[:,j,:,g].mean(axis=0)@expected_weights(binary_cell_p(p)))
    same(choices[6],fixed)
    for g in range(2):
        ix=np.flatnonzero(np.all(control==fixed[g],axis=1))[0]
        close(losses[6,g],averages[ix,g])


def trajectory_scores(strata,law,matches,refs,mech_scale,controls):
    """Independent whole-trajectory sample energy and per-component winding score."""
    result=[]
    for goal,(tau,p) in enumerate(controls):
        time_idx=np.flatnonzero(PLAN_TIMES==tau)[0]
        same(law['time_positions'][goal],time_idx)
        mass=np.tile(expected_weights(binary_cell_p(p)),4)/4.
        close(law['weights'][goal],mass.reshape(4,7))
        reference_idx=matches[goal]
        row=[]
        for label,scale in (('values',np.r_[np.ones(3),mech_scale]),('micro',1.)):
            atoms=(strata[label][:,time_idx]/scale).reshape(28,-1)
            actual=(refs[label][:,reference_idx]/scale).reshape(24,-1)
            pair=np.linalg.norm(atoms[:,None]-actual[None,:],axis=-1)
            own=np.linalg.norm(atoms[:,None]-atoms[None,:],axis=-1)
            value=(np.sum(pair*mass[:,None])/24.-.5*np.sum(own*mass[:,None]*mass[None,:]))/np.sqrt(atoms.shape[1])
            row.append(value)
        winding=strata['winding'][:,time_idx].reshape(28,3,3)
        probability=(winding*mass[:,None,None]).sum(axis=0)
        row.append(np.square(probability-refs['winding'][:,reference_idx]).mean())
        result.append(row)
    return result


def interval(values,reported=None):
    x=np.asarray(values,float)
    m=float(x.mean()); h=float(t.ppf(.975,len(x)-1)*x.std(ddof=1)/np.sqrt(len(x)))
    if reported is not None:
        close(m,reported['mean'])
        close([m-h,m+h],reported.get('t95',reported.get('pointwise_t95')))
        return
    return m,[m-h,m+h]


def fixtures():
    n=6;p=.273
    c=np.array([i*(i-1)/(n*(n-1))-2*p*i/n+p*p for i in range(n+1)])
    selected,loss,curve=whole_polynomial(c)
    close([selected,loss],[p,0.],atol=1.e-14)
    for q in (0.,.14,.51,1.):
        close(curve(q),expected_weights(q)@c)
    u=np.array([np.arange(1,7)/7.,np.linspace(.01,.83,6)])
    risk=np.array([np.arange(7)[::-1],np.arange(7)])
    a,b,cuts=exact_sweep(u,risk,.38)
    close(a,risk[np.arange(2),(u<.38).sum(axis=1)].mean())
    u0=cuts[np.argmin([exact_sweep(u,risk,q)[0] for q in cuts])]
    close(exact_sweep(u,risk,u0)[0],b)
    q=np.zeros((2,7,3,6));q[...,2]=1.
    v,_=goal_loss(q,np.ones(3))
    close(v[...,0],(np.mean(np.array([.2,.4,.6])**2)+np.mean(np.array([.15,.45,.8])**2))/3.)
    print('PASS synthetic: independent Bernstein reparameterization, strict mark sweep and untouched trajectory goals.')


def checked_time(record):
    utc=datetime.fromisoformat(record['actual_utc'])
    local=datetime.fromisoformat(record['asia_shanghai'])
    need(utc==local and utc.utcoffset().total_seconds()==0 and local.utcoffset().total_seconds()==28800,
         'Stage timestamps must have independently checkable UTC and Shanghai times.')
    return utc


def audit(out):
    out=out.resolve(strict=True)
    registration,run,acq_start,acq,predict,freeze,reveal,ref,score=(read(out/n) for n in (
        'REGISTERED.json','RUN.json','ACQUISITION_STARTED.json','ACQUIRED.json','PREDICT_STARTED.json',
        'PREDICTIONS_FROZEN.json','REVEAL_STARTED.json','REFERENCES_CLOSED.json','SCORED.json'))
    units=list(range(*RANGE))
    ent=np.frombuffer(hashlib.sha256(NAMESPACE.encode()).digest()[:16],dtype='<u4').tolist()
    need(registration['namespace']==NAMESPACE and registration['held_units']==units
         and registration['entropy']==ent and registration['roles']==ROLES
         and registration['reference_repeats']==24 and registration['planning_repeats']==4
         and registration['old_held_targets_opened'] is False and tuple(registration['methods'])==METHODS
         and tuple(registration['goals'])==GOALS and registration['rbf']==[.5,1.]
         and registration['gp']==[.22,.03,33] and registration['raw_audit_mothers']==[units[0],units[-1]]
         and registration['raw_audit_repeats']==[0,1] and registration['planning_native_per_mother']==1008,
         'Experimental constants differ from the prospective definition.')
    same(registration['times'],PLAN_TIMES)
    same(registration['box'],[[2.30,3.00],[0.,1.]])
    starts=[checked_time(r) for r in (registration,run,acq_start,acq,predict,freeze,reveal,ref,score)]
    need(all(x<=y for x,y in zip(starts,starts[1:])), 'Science phases ran out of chronological order.')
    need(run['registered_before_native'] is True and freeze['held_truth_opened'] is False,
         'Physical acquisition began before registration or held truth leaked.')
    reg_id,acq_id,freeze_id,ref_id=[sha(out/n) for n in ('REGISTERED.json','ACQUIRED.json',
                                                          'PREDICTIONS_FROZEN.json','REFERENCES_CLOSED.json')]
    same([r['registration_sha256'] for r in (run,acq_start,acq,freeze)],[reg_id]*4)
    same([r['acquisition_sha256'] for r in (predict,freeze)],[acq_id]*2)
    same([r['freeze_sha256'] for r in (reveal,ref,score)],[freeze_id]*3)
    need(score['registered_sha256']==reg_id and score['reference_sha256']==ref_id,'Score improperly bound.')
    for path,digest in registration['inputs'].items():
        need(sha(path)==digest,'Frozen input hash changed: '+path)
    for path,digest in registration['sources'].items():
        need(sha(path)==digest and sha(out/'SOURCE_SNAPSHOT'/Path(path).name)==digest,
             'Frozen runnable source hash changed: '+path)
    cfg=registration['physics']
    atlas_path=Path(registration['atlas']).resolve(strict=True)
    atlas_reg=read(atlas_path/'REGISTERED.json')
    train=read(atlas_path/'ACQUIRED.json')['groups']['train']
    train_ids=registration['train_units']
    need(len(set(train_ids))==512 and [r['unit'] for r in train]==train_ids
         and not set(units).intersection(u for subset in atlas_reg['units'].values() for u in subset),
         'Confirmation units overlap the prior atlas or training order changed.')
    need(cfg==atlas_reg['physics'],'Original physics changed between training and confirmation.')
    model=load_tree(registration['atlas_model'],registration['atlas_model_sha256'])
    historical=Path(registration['inverse_for_fixed']).resolve(strict=True)
    old_reg=read(historical/'REGISTERED.json')
    old_lib=read(historical/'LIBRARY_COMMITTED.json')
    old_audit=read(historical/'INDEPENDENT_AUDIT.json')
    need(old_reg['physics']==cfg and old_audit['passed'] is True and old_audit['scales_and_selection_checked'] is True
         and not set(units).intersection([*old_reg['held_ids'],*old_reg['train_ids']]),'Original fixed baseline provenance invalid.')
    mechanical=np.asarray(registration['normalization']['mechanical_scale'])
    same(mechanical,old_reg['normalization']['mechanical_scale'])
    old= np_load(bound(old_lib['arrays'],historical/'TRAIN_FUTURES.npz'))
    same(old['units'],old_reg['train_ids'])
    prior,_=goal_loss(old['atoms'][...,6:].reshape(512,2,6,3,6),mechanical)
    fixed=ANCHORS[prior.mean(axis=(0,1)).argmin(axis=0)]
    same(fixed,registration['fixed_controls'])
    population=np.asarray(registration['population_controls'],float)
    need(population.shape==(2,2) and np.isfinite(population).all()
         and np.isin(population[:,0],PLAN_TIMES).all() and np.all((population[:,1]>=0.)&(population[:,1]<=1.)),
         'Unconditioned continuous reference must emit one legal globally fixed control for each goal.')
    pop_folder=Path(registration['population']).resolve(strict=True)
    pop_score=read(pop_folder/'SCORED.json')
    pop_audit=read(pop_folder/'INDEPENDENT_AUDIT.json')
    for name in ('REGISTERED.json','SCORED.json','INDEPENDENT_AUDIT.json'):
        need(registration['inputs'][str(pop_folder/name)]==sha(pop_folder/name),'Population comparator was not frozen.')
    need(pop_audit['passed'] is True and pop_audit['scored_sha256']==sha(pop_folder/'SCORED.json')
         and pop_score['old_held_opened']==0 and pop_score['independent_train_mothers']==48,
         'Population comparator is missing its independent train-only audit.')
    same(pop_score['controls'],population)
    posterior=np_load(bound(freeze['posterior'],out/'POSTERIOR_SCENARIOS.npz'))
    chosen=np_load(bound(freeze['choices'],out/'CHOICES.npz'))
    same(posterior['train_units'],train_ids)
    for r in (posterior,chosen):
        same(r['held_units'],units)
    for rows in (acq['held'],freeze['plans'],ref['rows']):
        same([r['unit'] for r in rows],units)
    held_losses,held_blocks,trajectory=[],[],[]
    planning_calls=reference_calls=full_readouts=0
    plan_timing=np.zeros((2,3))
    ref_timing=np.zeros(3)
    def audit_role_rows(rawfile,expected,controls,streams,origin,packet,deep,micro,winding):
        nonlocal full_readouts
        full_readouts+=3*len(controls) if deep else 0
        return audit_packed(rawfile,expected,controls,streams,origin,packet,cfg,deep=deep,
                            expected_micro=micro,expected_winding=winding)
    for i,unit in enumerate(units):
        records=acq['held'][i]
        folder=out/'held'
        held=load_tree(bound(records['raw'],folder/f'{unit}_PREFIX.npz'))
        query=np_load(bound(records['query'],folder/f'{unit}_QUERY.npz'))['measured']
        same(records['native_calls'],3)
        need(held['calls']=={'attempted':3,'completed':3,'failed':0},'Held prefix is not complete.')
        for frame,age in zip(held['packets'],cfg['prefix_clock'],strict=True):
            check_actual_state(frame[0],expected_age=age)
        close([observe(frame) for frame in held['packets'][1:]],held['clean'])
        need(query.shape==(3,4) and np.isfinite(query).all(),'Query is not exactly three finite cameras.')
        same(posterior['query'][i],query)
        q=rbf_weights(model,query,registration['rbf'])
        close(posterior['c3_weights'][i],q)
        selected=addressed(unit,'prior_index').choice(len(train),size=4,p=q)
        same(posterior['train_positions'][i],selected)
        folder=out/'planning'/str(unit)
        pp=freeze['plans'][i]
        plan=read(bound(pp['plan'],folder/'PLANNED.json'))
        need(plan['unit']==unit and starts[4]<=checked_time(plan)<=starts[5], 'Planning receipt outside blind phase.')
        same([plan['native_calls'],pp['native_calls']],[1008,1008]);planning_calls+=1008
        strata=np_load(bound(plan['strata'],folder/'STRATA_VALUES.npz'))
        direct=np_load(bound(plan['direct'],folder/'DIRECT_VALUES.npz'))
        law=np_load(bound(plan['law'],folder/'SELECTED_PATH_LAW.npz'))
        ctrl=direct_control_reconstruction(unit)
        same(direct['controls'],ctrl);same(strata['times'],PLAN_TIMES);same(strata['train_idxs'],selected)
        need(strata['values'].shape==(4,6,7,3,6) and direct['values'].shape==(4,42,3,6),
             'Both planning arms must use exactly 42 futures per scenario.')
        choices=chosen['choices'][i]
        same(plan['choices'],choices)
        need(choices.shape==(len(METHODS),2,2) and len(plan['raw_blocks'])==4,'Invalid planned control or receipt count.')
        audit_controls(ctrl,strata['uniforms'],strata['values'],direct['values'],choices,
                       np.asarray(plan['planning_losses']),mechanical,fixed,population,registration['gp'])
        for j,chosen_atom in enumerate(selected):
            receipt=plan['raw_blocks'][j]
            old_record=train[chosen_atom]
            old_path=bound(old_record['raw'],atlas_path/'train'/f"{old_record['unit']}_RAW.npz")
            need(receipt['train_identity']=={'unit':old_record['unit'],'source_raw_sha256':sha(old_path)},
                 'Planning ancestor was not sampled from the permitted training atlas.')
            packet=load_tree(old_path)['packets'][-1]
            path=bound(receipt['raw'],folder/f'P{j}_RAW.npz')
            streams=seed_receipt(unit,j,'planning_future')
            uniform,_,_=placements(streams['placement_seed'],.5)
            threshold=np.r_[0.,np.sort(uniform)[1:],1.]
            same(strata['uniforms'][j],uniform);same(strata['thresholds'][j],threshold)
            same((uniform[None,:]<threshold[:,None]).sum(axis=1),np.arange(7))
            force=np.column_stack((np.repeat(PLAN_TIMES,7),np.tile(threshold,6)))
            controls=np.vstack((force,ctrl))
            expected=np.concatenate((strata['values'][j].reshape(42,3,6),direct['values'][j]),axis=0)
            deep=unit in registration['raw_audit_mothers'] and j in registration['raw_audit_repeats']
            audit_role_rows(path,expected,controls,streams,packet_fingerprint(packet),packet,deep,
                            strata['micro'][j].reshape(42,3,8),strata['winding'][j].reshape(42,3,3))
            with np.load(path,allow_pickle=False) as payload:
                timing=payload['native_timing']
                plan_timing[0]+=timing[:42].sum(axis=0);plan_timing[1]+=timing[42:].sum(axis=0)
        folder=out/'reference'/str(unit)
        rr=ref['rows'][i]
        result=read(bound(rr['reference'],folder/'COMPLETE.json'))
        need(result['unit']==unit and result['freeze_sha256']==freeze_id and starts[6]<=checked_time(result)<=starts[7],
             'Reference completion was not after the complete common freeze.')
        refs=np_load(bound(result['reference'],folder/'REFERENCE_VALUES.npz'))
        packet=held['packets'][-1]
        origin=packet_fingerprint(packet)
        need(origin==result['new_c3_identity'] and len(result['branches'])==24,
             'Reference omitted a frozen independent mother or trial.')
        unique=np.asarray(sorted(set(map(tuple,choices.reshape(-1,2)))))
        same(refs['controls'],unique);same(refs['choices'],choices);same(unique[refs['mapping']],choices)
        n=len(unique)*24*3
        same([result['native_calls'],rr['native_calls']],[n,n]);reference_calls+=n
        values=refs['values']
        need(values.shape==(24,len(unique),3,6),'Reference loops must use every frozen distinct action with 24 draws.')
        for j,receipt in enumerate(result['branches']):
            path=bound(receipt,folder/f'R{j}_RAW.npz')
            deep=unit in registration['raw_audit_mothers'] and j in registration['raw_audit_repeats']
            streams=seed_receipt(unit,j,'reference_future')
            ref_timing+=audit_role_rows(path,values[j],unique,streams,origin,packet,deep,refs['micro'][j],refs['winding'][j])
        l,b=goal_loss(values,mechanical)
        ix=refs['mapping']
        held_losses.append(l[:,ix,np.arange(2)].mean(axis=0))
        held_blocks.append(b[:,ix,np.arange(2),:].mean(axis=0))
        trajectory.append(trajectory_scores(strata,law,ix[0],refs,mechanical,choices[0]))
    l,b=np.asarray(held_losses),np.asarray(held_blocks)
    A=np_load(bound(score['arrays'],out/'SCORED_ARRAYS.npz'))
    same(A['held_units'],units);same(A['choices'],chosen['choices'])
    close(A['losses'],l);close(A['blocks'],b);close(score['means'],l.mean(axis=0));close(score['mean_blocks'],b.mean(axis=0))
    close(A['selected_path_predictive_scores'],trajectory)
    predictive=score['selected_path_scores']
    need(predictive['same_control_ref_only'] is True
         and predictive['names']==['functional_energy','micro8_energy','winding_brier'],
         'Selected path quality must bind its own exact chosen physical control.')
    close(predictive['means'],np.mean(trajectory,axis=0))
    anchor=np.all(chosen['choices'][...,None,:]==ANCHORS,axis=-1).any(axis=-1)
    same(score['off_anchor_counts'],(~anchor).sum(axis=0))
    gate=[]
    for j,name in enumerate(METHODS[1:],1):
        paired=l[:,j]-l[:,0]
        interval(paired.mean(axis=-1),score['same_budget_gain'][name])
        gate.append(interval(paired.mean(axis=-1))[1][0]>0.)
        for g in range(2):
            interval(paired[:,g],score['by_goal_gain'][name][g])
    need(score['primary_gate']==all(gate) and score['wet_experiments']==0,'Score overstates its prospective synthetic result.')
    same([acq['native_calls'],freeze['native_calls'],ref['native_calls']],[3*MOTHERS,planning_calls,reference_calls])
    same([score['native_acquisition'],score['native_planning'],score['native_reference'],score['independent_mothers']],
         [3*MOTHERS,planning_calls,reference_calls,MOTHERS])
    write(out/'INDEPENDENT_AUDIT.json',{**stamp(),'passed':True,'registered_sha256':reg_id,'scored_sha256':sha(out/'SCORED.json'),
          'freeze_sha256':freeze_id,'references_sha256':ref_id,'independent_mothers':MOTHERS,'deep_graph_mechanical_snapshots':full_readouts,
          'registered_sources':len(registration['sources']),'registered_inputs':len(registration['inputs']),
          'verified_native_receipts':{'acquisition':3*MOTHERS,'planning':planning_calls,'reference':reference_calls},
          'observed_process_seconds':{'count_plan':plan_timing[0],'direct_plan':plan_timing[1],'reference':ref_timing,
                                    'columns':['native','q_read','property']},'native_rollouts_invoked_by_auditor':0,
          'independently_recomputed':['old train-only best fixed','blind RBF posterior and prior atoms','every planned optimum and selected control',
                                      'strict feed subsets and frozen continuum','actual graph and tangent readouts in registered raw subset',
                                      'all paired original goal scores and uncertainty','selected whole-trajectory energy and winding scores']})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path)
    parser.add_argument('--fixtures',action='store_true')
    args=parser.parse_args()
    if args.fixtures:
        fixtures()
    elif args.out is not None:
        with threadpool_limits(limits=1):
            audit(args.out)
    else:
        parser.error('Provide --out or --fixtures.')


if __name__=='__main__':
    main()
