"""Independent no-native audit for the amortized continuous AAm/MBAA design."""

import argparse
from datetime import datetime
import hashlib
from pathlib import Path

import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import t
from threadpoolctl import threadpool_limits

from aam_formation_actual_rollout import packet_fingerprint
from aam_formation_cycle_driver import np_load, read, sha, write
from aam_formation_path_cycle import load_tree, stamp
from audit_aam_continuous_inverse import (
    audit_packed, binary_cell_p, expected_weights, goal_loss, whole_polynomial,
)


TIMES=np.array([2.30,2.3875,2.5625,2.7375,2.9125,3.00])
ORIGINAL=np.array([(tau,p) for tau in (2.30,3.00) for p in (0.,.5,1.)])
GOALS=('delayed_support','early_buffer_then_relax')
METRICS=('functional_energy','micro8_energy','winding_brier')
CHANNELS=('values','micro','winding')


def need(condition, message):
    if not condition:
        raise ValueError(message)


def exact(a,b):
    np.testing.assert_array_equal(a,b)


def close(a,b,*,atol=4.e-12):
    np.testing.assert_allclose(a,b,rtol=4.e-12,atol=atol)


def bound(entry,path):
    expected=Path(path).resolve()
    need(Path(entry['path']).resolve()==expected and expected.stat().st_size==entry['bytes']
         and sha(expected)==entry['sha256'],'Frozen artifact changed: '+str(expected))
    return expected


def time(record):
    instant=datetime.fromisoformat(record['actual_utc'])
    local=datetime.fromisoformat(record['asia_shanghai'])
    need(instant==local and instant.utcoffset().total_seconds()==0
         and local.utcoffset().total_seconds()==28800,'Time in a registered receipt is inconsistent.')
    return instant


def source_integrity(plan,snapshot):
    for filename,digest in plan['sources'].items():
        need(sha(filename)==digest and sha(snapshot/Path(filename).name)==digest,
             'Frozen source changed: '+str(filename))
    for filename,digest in plan['inputs'].items():
        need(sha(filename)==digest,'Frozen input changed: '+str(filename))


def restrict_probabilities(measured,train_x,scale,width,length):
    """Direct standardized squared distance and independent finite weights."""
    diff=(train_x[None]-measured[:,None])/scale
    sq=np.square(diff).reshape(*diff.shape[:2],-1).sum(axis=-1)
    exponent=-sq/(2.*width**2*length)
    exponent-=exponent.max(axis=1,keepdims=True)
    numerator=np.exp(exponent)
    return numerator/numerator.sum(axis=1,keepdims=True)


def risk_one(curve,control,goal):
    tau,p=control
    where=np.flatnonzero(TIMES==tau)
    need(len(where)==1 and 0.<=p<=1.,'Nonregistered time or illegal effective probability.')
    return expected_weights(binary_cell_p(p))@curve[where[0],:,goal]


def inspect_choice(curve,choice,reported,*,anchors=False):
    """Global polynomials or old anchors, both via an independent B-spline basis."""
    need(curve.shape==(6,7,2) and np.isfinite(curve).all() and np.asarray(choice).shape==(2,2),
         'Wrong six-time, seven-count, two-goal table.')
    for j,action in enumerate(choice):
        risk=risk_one(curve,action,j)
        close(risk,reported[j])
        if anchors:
            need(np.any(np.all(ORIGINAL==action,axis=1)),'A six-catalogue arm emitted an off-anchor action.')
            optimum=min(risk_one(curve,a,j) for a in ORIGINAL)
        else:
            optimum=min(whole_polynomial(coeff)[1] for coeff in curve[:,:,j])
        close(optimum,risk,atol=4.e-11)


def array_scores(bank,tau,p,w,reference,mechanical):
    """Independent multivariate finite-mixture energy plus marginal winding Brier."""
    k=np.flatnonzero(TIMES==tau)[0]
    weights=(w[:,None]*expected_weights(binary_cell_p(p))).reshape(-1)
    need(np.all(weights>=0.) and np.isclose(weights.sum(),1.,rtol=1.e-13),
         'Unnormalized microtrajectory distribution.')
    positive=np.flatnonzero(weights>0.)
    weights=weights[positive]
    result=[]
    for label,scale in (('values',np.r_[np.ones(3),mechanical]),('micro',1.)):
        x=(bank[label][:,k].reshape(-1,*bank[label].shape[-2:])[positive]/scale).reshape(len(positive),-1)
        y=(reference[label]/scale).reshape(len(reference[label]),-1)
        spread=(weights[:,None]*weights*cdist(x,x)).sum()
        shown=(weights*cdist(x,y).mean(axis=1)).sum()
        result.append((shown-.5*spread)/np.sqrt(x.shape[1]))
    winding=bank['winding'][:,k].reshape(-1,3,3)[positive]
    p_wind=(weights[:,None,None]*winding).sum(axis=0)
    result.append(np.square(reference['winding']-p_wind).mean())
    return result


def stats(x):
    x=np.asarray(x,float)
    n=len(x)
    m=x.mean(axis=0)
    h=t.ppf(.975,n-1)*x.std(axis=0,ddof=1)/np.sqrt(n)
    return m,np.stack((m-h,m+h),axis=-1)


def check_stats(x,y):
    mean,interval=stats(x)
    close(mean,y['mean'])
    close(interval,y.get('pointwise_t95',y.get('t95')))
    need(y.get('mothers',y.get('independent_mothers',len(x)))==len(x),'Uncertainty unit is not the registered mother.')


def fixtures():
    count=np.arange(7.)
    p=.273
    risks=count*(count-1.)/30.-2*p*count/6.+p*p
    curve=np.broadcast_to(risks[None,:,None],(6,7,2))
    inspect_choice(curve,[[2.3,p],[3.,p]],[0.,0.])
    inspect_choice(curve,[[2.3,.5],[2.3,.5]],[(.5-p)**2]*2,anchors=True)
    bank={name:np.zeros((2,6,7,3,n)) for name,n in zip(CHANNELS,(6,8,3),strict=True)}
    seen={name:np.ones((24,3,n)) for name,n in zip(CHANNELS,(6,8,3),strict=True)}
    close(array_scores(bank,2.3,.8,np.array([.79,.21]),seen,np.ones(3)),np.ones(3))
    print('PASS independent finite polynomial and actual path score fixtures.')


def population_bank(root,registration,train_ids):
    """Bind every population atom and reconstruct all frozen readout arrays."""
    root=Path(root).resolve(strict=True)
    lock,sheet,verify=(root/p for p in ('REGISTERED.json','SCORED.json','INDEPENDENT_AUDIT.json'))
    r,s,a=(read(p) for p in (lock,sheet,verify))
    need(a['passed'] is True and a['scored_sha256']==sha(sheet) and a['registration_sha256']==sha(lock)
         and r['physics']==registration['physics'] and s['old_held_opened']==0 and s['native_calls']==6048
         and s['independent_train_mothers']==48,'Missing lawful independent population archive.')
    exact(r['times'],TIMES)
    positions=np.asarray(r['positions'],int)
    need(len(set(positions))==48 and np.all((positions>=0)&(positions<len(train_ids))),'Population mother positions are not a 48-atom subset.')
    idx=np.asarray(train_ids)[positions]
    exact([row['unit'] for row in s['raw_mothers']],idx)
    sheet_array=np_load(bound(s['arrays'],root/'POPULATION_ARRAYS.npz'))
    bank={key:[] for key in CHANNELS}
    for i,(unit,row) in enumerate(zip(idx,s['raw_mothers'],strict=True)):
        path=bound(row['raw'],root/'raw'/f'{unit}_RAW.npz')
        uniforms=np.random.default_rng(int(row['streams']['placement_seed'])).random(6)
        exact(row['uniforms'],uniforms)
        thresholds=np.r_[0.,np.sort(uniforms)[1:],1.]
        exact(row['thresholds'],thresholds)
        exact((uniforms<thresholds[:,None]).sum(axis=1),np.arange(7))
        with np.load(path,allow_pickle=False) as raw:
            exact(raw['controls'],np.column_stack((np.repeat(TIMES,7),np.tile(thresholds,6))))
            exact(raw['completed'],np.ones((42,3),bool))
            for key,n in (('values',6),('micro',8),('winding',3)):
                bank[key].append(raw[key].reshape(6,7,3,n))
            exact(bank['values'][-1],sheet_array['arrays'][i])
    bank={key:np.asarray(val) for key,val in bank.items()}
    losses,_=goal_loss(bank['values'],registration['normalization']['mechanical_scale'])
    close(losses,sheet_array['losses'])
    return bank,losses,idx,positions


def verify_prefixes(folder,records,units,queries,physics):
    """Confirm observation-only payload binding and packet identity without using target readouts."""
    data=[]
    for i,(row,unit) in enumerate(zip(records,units,strict=True)):
        need(row['unit']==unit and row['native_calls']==3,'Held acquisition was omitted or has wrong native accounting.')
        query=np_load(bound(row['query'],folder/f'{unit}_QUERY.npz'))['measured']
        need(query.shape==(3,4) and np.isfinite(query).all(),'Prospective forecast did not use exactly three camera snapshots.')
        exact(query,queries[i])
        raw=load_tree(bound(row['raw'],folder/f'{unit}_PREFIX.npz'))
        need(raw['calls']=={'attempted':3,'completed':3,'failed':0} and len(raw['packets'])==4,
             'Malformed native query prefix.')
        exact([float(x[0].position[0]) for x in raw['packets']],physics['prefix_clock'])
        data.append(raw['packets'][-1])
    return data


NS='aam-full48-legal-library-amortized-inverse-v1'
PREVIOUS_RNG_NAMESPACE='aam-continuous-unchanged-formation-inverse-count-law-v1'
UNITS=list(range(2026125000,2026125096))
METHODS=('rbf_bernstein','uniform_bernstein','rbf_threshold','rbf_six','rbf_four','rbf_live_four','fixed')
SAME_CONTROL=('rbf_bernstein','uniform_bernstein','rbf_four','rbf_live_four')
PLANNING_SEGMENTS=504


def prior_seed_rng(unit,role,draw=0,lane=0):
    words=np.frombuffer(hashlib.sha256(PREVIOUS_RNG_NAMESPACE.encode()).digest()[:16],dtype='<u4').tolist()
    return np.random.default_rng(np.random.SeedSequence([*words,unit,role,draw,lane]))


def suffix_seeds(unit,draw,role):
    return {'seed':prior_seed_rng(unit,role,draw,1).integers(0,2**32-51,dtype=np.uint32),
            'native_seeds':prior_seed_rng(unit,role,draw,2).integers(0,2**32,3,dtype=np.uint32),
            'placement_seed':prior_seed_rng(unit,role,draw,3).integers(0,2**32,dtype=np.uint32)}


def stable_indices(w,u):
    cdf=np.cumsum(w)
    cdf[-1]=1.
    return np.searchsorted(cdf,u,side='right')


def inspect_thresholds(uniforms,routes,w,choices,losses):
    """Global exact thresholds reconstructed without the driver's incremental scan."""
    testing=np.r_[0.,np.nextafter(np.unique(uniforms),1.),1.]
    ordered=(uniforms[None,:,:]<testing[:,None,None]).sum(axis=-1)
    rows=np.arange(len(w))
    for g,(tau,p) in enumerate(choices):
        at=np.flatnonzero(TIMES==tau)
        need(len(at)==1 and 0.<=p<=1.,'Invalid threshold optimizer result.')
        masks=(uniforms<p).sum(axis=1)
        fitted=routes[rows,at[0],masks,g]@w
        close(fitted,losses[g])
        best=min(np.min(routes[rows[None,:],i,ordered,g]@w) for i in range(len(TIMES)))
        close(fitted,best,atol=4.e-11)


def prior_source(reg):
    model=Path(reg['atlas_model']).resolve(strict=True)
    folder=model.parent
    plan=read(folder/'REGISTERED.json')
    acq=read(folder/'ACQUIRED.json')
    train=acq['groups']['train']
    ids=plan['units']['train']
    exact([r['unit'] for r in train],ids)
    need(len(set(ids))==512 and plan['physics']==reg['physics'] and acq['registration_sha256']==sha(folder/'REGISTERED.json')
         and not set(UNITS).intersection(u for group in plan['units'].values() for u in group),
         'Fresh cohort overlaps training or the prior physics changed.')
    old_arrays=[Path(p) for p in reg['inputs'] if Path(p).name=='TRAIN_FUTURES.npz']
    need(len(old_arrays)==1,'The old training-only fixed control is not registered.')
    old=old_arrays[0].parent
    ireg=read(old/'REGISTERED.json')
    oldaudit=read(old/'INDEPENDENT_AUDIT.json')
    need(oldaudit['passed'] is True and oldaudit['registered_sha256']==sha(old/'REGISTERED.json')
         and oldaudit['scored_sha256']==sha(old/'SCORED.json')
         and ireg['physics']==reg['physics'] and ireg['train_ids']==ids
         and ireg['normalization']==reg['normalization'] and not set(UNITS).intersection(ireg['held_ids']),
         'Prior fixed control does not bind its original train-only source.')
    library=read(old/'LIBRARY_COMMITTED.json')
    records=np_load(bound(library['arrays'],old_arrays[0]))
    exact(records['units'],ids)
    ll,_=goal_loss(records['atoms'][...,6:].reshape(512,2,6,3,6),reg['normalization']['mechanical_scale'])
    fixed=ORIGINAL[np.argmin(ll.mean(axis=(0,1)),axis=0)]
    exact(reg['fixed_controls'],fixed)
    kernel=load_tree(model,reg['inputs'][str(model)])
    return kernel['x'],kernel,folder,train,ids,fixed


def verify_one_block(folder,name,entry,expected,controls,streams,origin,packet,cfg,deep):
    path=bound(entry,folder/name)
    elapsed=audit_packed(path,expected['values'],controls,streams,origin,packet,cfg,deep=deep,
                         expected_micro=expected['micro'],expected_winding=expected['winding'])
    return elapsed,3*len(controls)


def audit_live(out,reg,manifest,unit,picked,train_root,train,choices,risks,clock):
    """Freeze-bound common-CRN live planner with 4 original independent draws."""
    folder=out/'planning'/str(unit)
    row=read(bound(manifest['plan'],folder/'PLANNED.json'))
    tables=np_load(bound(row['tabs'],folder/'LIVE_TABLES.npz'))
    need(row['unit']==unit and row['native_calls']==manifest['native_calls']==PLANNING_SEGMENTS
         and len(row['raw_blocks'])==4 and clock[0]<=time(row)<=clock[1],
         'Live planning was outside its freeze clock or did not spend exactly 504 native segments per query.')
    exact(row['controls'],choices);exact(row['estimated_risks'],risks)
    exact(tables['train_idxs'],picked)
    cfg=reg['physics']
    timings=np.zeros(3)
    deep_readouts=0
    for draw,(slot,record) in enumerate(zip(picked,row['raw_blocks'],strict=True)):
        streams=suffix_seeds(unit,draw,3)
        source=train[slot]
        path=bound(source['raw'],train_root/'train'/f"{source['unit']}_RAW.npz")
        need(record['train_identity']=={'unit':source['unit'],'source_raw_sha256':source['raw']['sha256']},
             'Live precursor is not its independently selected old training atlas atom.')
        packet=load_tree(path)['packets'][-1]
        u=np.random.default_rng(int(streams['placement_seed'])).random(6)
        cut=np.r_[0.,np.sort(u)[1:],1.]
        exact(tables['uniforms'][draw],u)
        exact(tables['thresholds'][draw],cut)
        exact((u<cut[:,None]).sum(axis=1),np.arange(7))
        controls=np.column_stack((np.repeat(TIMES,7),np.tile(cut,6)))
        expected={name:tables[name][draw].reshape(42,3,-1) for name in CHANNELS}
        deep=unit in (UNITS[0],UNITS[-1]) and draw in (0,1)
        dt,k=verify_one_block(folder,f'P{draw}_RAW.npz',record['raw'],expected,controls,streams,
                              packet_fingerprint(packet),packet,cfg,deep)
        timings+=dt
        deep_readouts+=k if deep else 0
    return tables,timings,deep_readouts


def run_audit(out):
    out=out.resolve(strict=True)
    names=('REGISTERED.json','RUN.json','ACQUISITION_STARTED.json','ACQUIRED.json','PREDICTION_STARTED.json',
           'PREDICTIONS_FROZEN.json','REVEAL_STARTED.json','REFERENCES_CLOSED.json','SCORED.json')
    files=[out/name for name in names]
    reg,run,start,acquired,predict,frozen,reveal,refs,score=(read(p) for p in files)
    times=[time(i) for i in (reg,run,start,acquired,predict,frozen,reveal,refs,score)]
    need(all(x<=y for x,y in zip(times,times[1:])), 'Missing chronological forecast/held-reference separation.')
    need(reg['namespace']==NS and reg['units']==UNITS and reg['planning_draws']==4 and reg['reference_repeats']==24
         and reg['old_held_payloads_read']==0 and reg['new_fits']==0 and reg['rbf']==[.5,1.]
         and tuple(reg['methods'])==METHODS and tuple(reg['same_control_methods'])==SAME_CONTROL
         and tuple(reg['goals'])==GOALS and tuple(reg['forecast_metrics'])==METRICS
         and reg['main_new_planning_native']==frozen['main_native_calls']==0
         and reg['live_native_per_mother']==PLANNING_SEGMENTS and reg['live_total_native']==len(UNITS)*PLANNING_SEGMENTS
         and reg['cached_population_native_already_spent']==6048
         and predict['query_only'] is True and frozen['held_truth_opened'] is False and run['registered_before_native'] is True,
         'Wrong prospective arm, chemistry or registered acquisition scope.')
    exact(reg['times'],TIMES)
    exact(reg['box'],[[2.30,3.00],[0.,1.]])
    reg_sha,acq_sha,freeze_sha,ref_sha=(sha(files[k]) for k in (0,3,5,7))
    exact([x['registration_sha256'] for x in (run,start,acquired,frozen,score)],[reg_sha]*5)
    exact([x['acquisition_sha256'] for x in (predict,frozen)],[acq_sha]*2)
    exact([x['freeze_sha256'] for x in (reveal,refs,score)],[freeze_sha]*3)
    need(score['reference_sha256']==ref_sha,'Score points to another independent reference.')
    source_integrity(reg,out/'SOURCE_SNAPSHOT')
    x,model,training,train,ids,fixed=prior_source(reg)
    bank,risk,donors,positions=population_bank(reg['population_root'],reg,ids)
    exact(reg['donor_units'],donors);exact(reg['donor_positions'],positions)
    need(not set(UNITS).intersection(donors), 'Same true mother was reused in the archive and prospective test.')
    widths=[model['camera_scale'],reg['rbf'][0],model['kernel_sq_scale']]
    query=np_load(bound(frozen['choices'],out/'CHOICES.npz'))
    exact(query['units'],UNITS);exact(query['donors'],donors)
    true_c3=verify_prefixes(out/'held',acquired['held'],UNITS,query['query'],reg['physics'])
    exact(frozen['query_sha256'],[r['query']['sha256'] for r in acquired['held']])
    full=restrict_probabilities(query['query'],x,*widths)
    select=restrict_probabilities(query['query'],x[positions],*widths)
    close(query['weights'],select);close(query['full_atlas_weights'],full)
    u4=np.asarray([prior_seed_rng(u,1).random(4) for u in UNITS])
    exact(query['shared_uniform4'],u4)
    cached=np.asarray([stable_indices(w,z) for w,z in zip(select,u4,strict=True)])
    live=np.asarray([stable_indices(w,z) for w,z in zip(full,u4,strict=True)])
    exact(query['draw4'],cached);exact(query['live_draw4'],live)
    choices=query['choices']
    need(choices.shape==(len(UNITS),len(METHODS),2,2) and np.isfinite(choices).all()
         and len(frozen['live_plans'])==len(UNITS) and len(refs['rows'])==len(UNITS),
         'Missing physical controls or reference rows.')
    cfg=reg['physics']
    mech=reg['normalization']['mechanical_scale']
    uniform=np.full(48,1./48)
    u=np.asarray([i['uniforms'] for i in reg['donor_rows']])
    losses,blocks,path_scores=[],[],[]
    native_refs=deep_scans=0
    native_live=0
    prior_timing=np.zeros(3)
    ref_timing=np.zeros(3)
    for i,unit in enumerate(UNITS):
        c,r=choices[i],query['planning_gap_risk'][i]
        w=select[i]
        sparse=np.bincount(cached[i],minlength=48)/4.
        a=np.tensordot(w,risk,axes=1)
        b=risk.mean(axis=0)
        d=np.tensordot(sparse,risk,axes=1)
        inspect_choice(a,c[0],r[0]);inspect_choice(b,c[1],r[1])
        close(c[1],reg['population_controls'],atol=5.e-9)
        inspect_thresholds(u,risk,w,c[2],r[2])
        inspect_choice(a,c[3],r[3],anchors=True);inspect_choice(d,c[4],r[4])
        live_manifest=frozen['live_plans'][i]
        need(live_manifest['unit']==unit,'A live planning mother was swapped.')
        lb,dt,k=audit_live(out,reg,live_manifest,unit,live[i],training,train,c[5],r[5],times[4:6])
        native_live+=PLANNING_SEGMENTS
        prior_timing+=dt;deep_scans+=k
        live_l,_=goal_loss(lb['values'],mech)
        inspect_choice(live_l.mean(axis=0),c[5],r[5])
        exact(c[6],fixed);close([risk_one(a,p,g) for g,p in enumerate(fixed)],r[6])
        folder=out/'reference'/str(unit)
        ref=refs['rows'][i]
        rr=read(bound(ref['reference'],folder/'COMPLETE.json'))
        need(ref['unit']==rr['unit']==unit and rr['freeze_sha256']==freeze_sha and len(rr['branches'])==24
             and times[6]<=time(rr)<=times[7],'Actual suffixes do not follow frozen identity/chronology.')
        data=np_load(bound(rr['reference'],folder/'REFERENCE_VALUES.npz'))
        distinct=np.asarray(sorted(set(map(tuple,c.reshape(-1,2)))))
        exact(data['choices'],c);exact(data['controls'],distinct);exact(distinct[data['mapping']],c)
        required=len(distinct)*24*3
        need(ref['native_calls']==rr['native_calls']==required,'Deduplicated reference budget mismatch.')
        native_refs+=required
        origin=packet_fingerprint(true_c3[i]);exact(rr['new_c3_identity'],origin)
        for k,entry in enumerate(rr['branches']):
            v={name:data[name][k] for name in CHANNELS}
            deep=unit in (UNITS[0],UNITS[-1]) and k in (0,1)
            dt,n=verify_one_block(folder,f'R{k}_RAW.npz',entry,v,distinct,suffix_seeds(unit,k,4),origin,true_c3[i],cfg,deep)
            ref_timing+=dt;deep_scans+=n if deep else 0
        a,b=goal_loss(data['values'],mech)
        map_idx=data['mapping']
        losses.append(a[:,map_idx,np.arange(2)].mean(axis=0))
        blocks.append(b[:,map_idx,np.arange(2),:].mean(axis=0))
        rows=[]
        for goal,(tau,p) in enumerate(c[0]):
            exact(TIMES[query['time_index'][i,goal]],tau)
            counts=expected_weights(binary_cell_p(p))
            close(query['same_control_mass'][i,goal],np.array([w,uniform,sparse])[:,:,None]*counts)
            close(query['live_same_control_mass'][i,goal],np.broadcast_to(counts/4,(4,7)))
            v={name:data[name][:,map_idx[0,goal]] for name in CHANNELS}
            rows.append([array_scores(bank,tau,p,ww,v,mech) for ww in (w,uniform,sparse)]
                        +[array_scores(lb,tau,p,np.full(4,.25),v,mech)])
        path_scores.append(rows)
    l,b,e=map(np.asarray,(losses,blocks,path_scores))
    retained=np_load(bound(score['arrays'],out/'SCORED_ARRAYS.npz'))
    exact(retained['units'],UNITS);close(retained['losses'],l);close(retained['blocks'],b);close(retained['same_control_scores'],e)
    close(score['means'],l.mean(axis=0));close(score['mean_blocks'],b.mean(axis=0))
    close(score['mean_donor_ess'],np.mean(1./np.sum(select**2,axis=1)))
    close(score['same_control_forecast']['means'],e.mean(axis=0))
    need(score['same_control_forecast']['all_evaluated_at_frozen_rbf_bernstein_control'] is True,
         'Forward comparator reference condition differs between methods.')
    decisions=[]
    for j,name in enumerate(METHODS[1:],1):
        d=l[:,j]-l[:,0]
        check_stats(d.mean(axis=1),score['gain_over'][name]);decisions.append(stats(d.mean(axis=1))[1][0]>0.)
        for g in range(2):
            check_stats(d[:,g],score['by_goal_gain'][name][g])
    for j,name in enumerate(SAME_CONTROL[1:],1):
        for k,metric in enumerate(METRICS):
            check_stats((e[:,:,j,k]-e[:,:,0,k]).mean(axis=1),score['same_control_forecast']['gain_over'][name][metric])
    need(score['mothers']==len(UNITS) and score['primary_gate']==all(decisions)
         and score['new_fits']==0 and score['old_held_payloads_read']==0 and score['new_wet_measurements']==0,
         'Outcome summary misstates cohort, superiority evidence or wet work.')
    exact([score['native_acquisition'],score['native_planning'],score['native_live_comparator_planning'],
           score['native_cached_population_already_spent'],score['native_reference']],
          [3*len(UNITS),0,native_live,6048,native_refs])
    exact([acquired['native_calls'],frozen['native_calls'],refs['native_calls']],
          [3*len(UNITS),native_live,native_refs])
    write(out/'INDEPENDENT_AUDIT.json',{**stamp(),'passed':True,'registration_sha256':reg_sha,'freeze_sha256':freeze_sha,
          'scored_sha256':sha(files[-1]),'reference_sha256':ref_sha,'mothers':len(UNITS),'original_train_donors':48,
          'real_native_calls_by_auditor':0,'deep_native_mechanical_and_graph_rows':deep_scans,'wet_experiments':0,
          'native':{'prefix':3*len(UNITS),'live_planning':native_live,'amortized_planning':0,'references':native_refs,
                    'already_committed_population_bank':6048},
          'summed_process_seconds':{'live_planning':prior_timing,'reference':ref_timing,'columns':['native','q','property']},
          'independently_recomputed':['conditioning probabilities','all native and amortized selections','bounded raw path check',
                                      'same-control trajectory scores','paired original goal scores','strong fixed train-only policy']})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path)
    parser.add_argument('--fixtures',action='store_true')
    a=parser.parse_args()
    if a.fixtures:
        fixtures()
    elif a.out is not None:
        with threadpool_limits(limits=1):
            run_audit(a.out)
    else:
        parser.error('Provide --out or --fixtures.')


if __name__=='__main__':
    main()
