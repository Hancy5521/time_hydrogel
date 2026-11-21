"""Prospective V5 action headroom from complete C3 and independent futures."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from hashlib import sha256
from pathlib import Path
from shutil import copyfile

import numpy as np
from threadpoolctl import threadpool_limits

from aam_first_interval_paired_fork import future_for_action, score_future_for_action
from aam_formation_actual_rollout import packet_fingerprint
from aam_formation_cycle_core import ACTIONS, FUTURE_TIMES, PREFIX_CLOCK
from aam_formation_cycle_data import check_actual_state
from aam_formation_cycle_driver import np_load, np_save, read, save_tree, sha, write
from aam_formation_cycle_methods import GOALS, evaluate_choices, process_goal_losses, unpack_targets
from aam_formation_local_baselines import validate_fixed_normalization
from aam_formation_path_cycle import event, load_tree, stamp
from aam_temporal_inverse_bridge import validate_catalogue, weighted_decision
from package_aam_formation_local_cycle import source_closure
from run_aam_mark_candidate_filter import interval, verify
from run_aam_suffix_group_smoother import acquired_data


ROOT=Path(__file__).resolve().parent
PROTOCOL=ROOT/'AAM_PRIVILEGED_V5_ACTION_HEADROOM_ZH.md'
NS='aam-original-V5-true-C3-action-headroom-v1'
ENTROPY=[int.from_bytes(sha256(NS.encode()).digest()[i:i+4],'little') for i in range(0,16,4)]
UNITS=list(range(2026119000,2026119064))
REPEATS={'plan':32,'reference':24}
ROLES={'plan':1,'reference':2}
ROUTE={'seed':1,'native_seeds':2,'placement_seed':3}
MOTHERS=len(UNITS)


def reference(path):
    return {'path':str(path.resolve(strict=True)),'bytes':path.stat().st_size,'sha256':sha(path)}


def best_fixed(source):
    reg,lib,audit=map(read,[source/n for n in ('REGISTERED.json','LIBRARY_COMMITTED.json','INDEPENDENT_AUDIT.json')])
    if (reg['namespace']!='aam-temporal-law-v5-original-goals-fresh96-v1'
            or not audit['passed'] or audit['registered_sha256']!=sha(source/'REGISTERED.json')
            or not audit['scales_and_selection_checked'] or audit['library_mothers']!=512
            or audit['scored_sha256']!=sha(source/'SCORED.json')
            or lib['registration_sha256']!=sha(source/'REGISTERED.json')):
        raise ValueError('The original fixed action must use the independently audited train-only archive.')
    norm=validate_fixed_normalization(reg['normalization'])
    predicted_io=verify(source/'TRAIN_FUTURES.npz',lib['arrays'])
    raw=np_load(predicted_io)
    if (np.shape(raw['atoms'])!=(512,2,114)
            or not np.array_equal(raw['units'],reg['train_ids'])
            or set(UNITS).intersection(reg['held_ids']+reg['train_ids'])):
        raise ValueError('Frozen 512-mother support or fresh complete-state mother IDs changed.')
    design=weighted_decision(raw['atoms'],np.ones(512),norm)
    validate_catalogue(reg['physics'])
    return reg['physics'],norm,{'choices':design['choices'],'expected_loss':design['expected_loss'],
                               'original_library':lib['arrays'],'definition':'uniform over both suffixes of each fixed training mother'}


def register(out,source,review,workers):
    physical,norm,baseline=best_fixed(source)
    text=review.read_text(encoding='utf-8')
    if not text.strip() or '完整第三帧' not in text or '最佳固定' not in text:
        raise ValueError('Use the completed Work review asking for complete C3 and unchanged best fixed.')
    files = [*(source/n for n in ('REGISTERED.json','LIBRARY_COMMITTED.json','TRAIN_FUTURES.npz',
                                   'INDEPENDENT_AUDIT.json','SCORED.json')),review,PROTOCOL]
    closure=source_closure([Path(__file__),ROOT/'audit_aam_privileged_v5_action_headroom.py'])
    out.mkdir(exist_ok=False)
    reg=out/'REGISTERED.json'
    write(reg,{**stamp(),'namespace':NS,'entropy':ENTROPY,'route_roles':ROLES,'child_roles':ROUTE,
               'workers':workers,'physics':physical,'normalization':norm,'held_units':UNITS,
               'goals':GOALS,'actions':ACTIONS,'future_ages':FUTURE_TIMES,'planning_repeats':32,
               'planning_halves':[list(range(16)),list(range(16,32))],'reference_repeats':24,
               'fixed_baseline':baseline,'predecessor':str(source),
               'primary':'mean of two fixed-minus-combined selected losses on post-freeze references',
               'gate':'paired mother t95 lower > 0','predecessor_policy_arrays_opened':False,
               'old_held_targets_read':0,'new_fits':0,'new_wet_measurements':0,
               'budget':{'prefix_native':192,'plan_native':36864,'reference_native':27648,
                         'future_mechanical':64512,'total_native':64704},
               'inputs':{str(p):sha(p) for p in files},'sources':{str(p):sha(p) for p in closure.values()}})
    snapshot=out/'SOURCE_SNAPSHOT'
    snapshot.mkdir()
    for name,path in closure.items():
        copyfile(path,snapshot/name)
    write(out/'RUN.json',{**stamp(),'registered_before_native':True,'registration_sha256':sha(reg)})


def checked(out):
    path=out/'REGISTERED.json'
    plan=read(path)
    if (plan['namespace']!=NS or plan['entropy']!=ENTROPY or plan['held_units']!=UNITS
            or read(out/'RUN.json')['registration_sha256']!=sha(path)):
        raise ValueError('Need the fresh prospective complete-state design.')
    for name,digest in {**plan['inputs'],**plan['sources']}.items():
        if sha(name)!=digest:
            raise ValueError('Frozen input or source changed: '+name)
    return plan


def use_pool(worker,jobs,workers,out,action):
    rows=[]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for row in pool.map(worker,jobs,chunksize=1):
            rows.append(row)
            event(out,action,unit=row['unit'])
    if [r['unit'] for r in rows]!=UNITS:
        raise ValueError('The registered disjoint mother order changed.')
    return rows


def acquire_one(job):
    out,cfg,unit=job
    _,raw=acquired_data(cfg,unit)
    if raw['calls'] != {'attempted':3,'completed':3,'failed':0}:
        raise ValueError('Three original prefix intervals must complete for every fresh mother.')
    return {'unit':unit,'raw':save_tree(out/'held'/f'{unit}_PREFIX.npz',raw)}


def acquire(out,plan):
    rh=sha(out/'REGISTERED.json')
    write(out/'ACQUISITION_STARTED.json',{**stamp(),'registration_sha256':rh})
    (out/'held').mkdir()
    jobs=[(out,plan['physics'],u) for u in UNITS]
    rows=use_pool(acquire_one,jobs,plan['workers'],out,'complete_true_C3_archived')
    write(out/'ACQUIRED.json',{**stamp(),'registration_sha256':rh,'held':rows,'native_calls':192})


def future_seeds(label,unit,j):
    root=[*ENTROPY,ROLES[label],unit,j]
    streams={key:np.random.default_rng(np.random.SeedSequence([*root,code])) for key,code in ROUTE.items()}
    return {'seed':int(streams['seed'].integers(0,2**32-51,dtype=np.uint32)),
            'native_seeds':streams['native_seeds'].integers(0,2**32,3,dtype=np.uint32),
            'placement_seed':int(streams['placement_seed'].integers(0,2**32,dtype=np.uint32))}


def future_job(job):
    out,config,record,role,freeze=job
    u=record['unit']
    if role=='reference' and sha(out/'CHOICES_FROZEN.json')!=freeze:
        raise ValueError('References must start after the immutable global choice freeze.')
    raw=load_tree(verify(out/'held'/f'{u}_PREFIX.npz',record['raw']))
    packet=raw['packets'][-1]
    check_actual_state(packet[0],expected_age=PREFIX_CLOCK[-1])
    origin=packet_fingerprint(packet)
    closed=out/role/str(u)
    closed.mkdir()
    values,refs=[],[]
    with threadpool_limits(limits=1):
        try:
            for j in range(REPEATS[role]):
                result=[]
                streams=future_seeds(role,u,j)
                for a,action in enumerate(ACTIONS):
                    branch=future_for_action(packet,action,config,streams,f'{role}:{u}')
                    # The origin is already sealed in PREFIX; the final exact state
                    # is already present at the third snapshot. Keep each once.
                    archive=branch
                    if (branch['valid'] and len(branch['snapshots'])==3
                            and packet_fingerprint(branch['original_packet'])==origin
                            and packet_fingerprint(branch['packet'])==
                            packet_fingerprint(branch['snapshots'][-1]['packet'])):
                        archive={k:v for k,v in branch.items() if k not in ('original_packet','packet')}
                        archive['final_packet_sha256']=packet_fingerprint(branch['packet'])
                    receipt=save_tree(closed/f'R{j:02d}_A{a}_RAW.npz',archive)
                    refs.append({'repeat':j,'action':a,'file':receipt})
                    per_age,_,_=score_future_for_action(branch,packet,action,config,streams)
                    result.append(per_age)
                values.append(result)
        except Exception as exc:
            write(closed/'FAILED.json',{**stamp(),'role':role,'mother':u,'raw':refs,'error':repr(exc)})
            raise
    if packet_fingerprint(packet)!=origin:
        raise RuntimeError('The true C3 starting packet changed during a counterfactual.')
    miniature=np_save(closed/'FUTURES.npz',values=np.asarray(values))
    status=closed/'COMPLETE.json'
    write(status,{**stamp(),'unit':u,'role':role,'prediction_freeze_sha256':freeze,
                  'source_prefix_sha256':record['raw']['sha256'],'cutoff_fingerprint':origin,
                  'branches':refs,'values':miniature,'native_calls':18*REPEATS[role],
                  'mechanical_readouts':18*REPEATS[role]})
    return {'unit':u,'receipt':reference(status)}


def plans_from_values(future,norm):
    losses=process_goal_losses(future,norm)['loss']
    if losses.shape!=(32,6,2):
        raise ValueError('Require 32 actual matched planning actions and both original goals.')
    averages=np.stack((losses[:16].mean(axis=0),losses[16:].mean(axis=0),losses.mean(axis=0)))
    choices=np.argmin(averages,axis=1)
    return choices,averages


def plan_actions(out,plan):
    ah=sha(out/'ACQUIRED.json')
    acquired=read(out/'ACQUIRED.json')
    if acquired['registration_sha256']!=sha(out/'REGISTERED.json'):
        raise ValueError('Planning starts from exactly the registered complete cutoff states.')
    write(out/'PLANNING_STARTED.json',{**stamp(),'acquisition_sha256':ah,'reference_future_available':False})
    (out/'plan').mkdir()
    jobs=[(out,plan['physics'],r,'plan',None) for r in acquired['held']]
    rows=use_pool(future_job,jobs,plan['workers'],out,'closed_planning_native_branches')
    chosen,risks=[],[]
    for r in rows:
        folder=out/'plan'/str(r['unit'])
        route=read(verify(folder/'COMPLETE.json',r['receipt']))
        loss=np_load(verify(folder/'FUTURES.npz',route['values']))['values']
        c,v=plans_from_values(loss,plan['normalization'])
        chosen.append(c)
        risks.append(v)
    freeze=np_save(out/'CHOICES.npz',units=np.asarray(UNITS), choices=np.asarray(chosen),
                   risks=np.asarray(risks), fixed=np.asarray(plan['fixed_baseline']['choices']))
    write(out/'CHOICES_FROZEN.json',{**stamp(),'registration_sha256':sha(out/'REGISTERED.json'),
          'acquisition_sha256':ah,'plan':rows,'choices':freeze,'native_calls':36864,
          'reference_future_available':False,'all_references_unopened':True})


def verify_choices(out):
    f=read(out/'CHOICES_FROZEN.json')
    if (f['registration_sha256']!=sha(out/'REGISTERED.json') or f['acquisition_sha256']!=sha(out/'ACQUIRED.json')
            or f['reference_future_available'] or not f['all_references_unopened']):
        raise ValueError('All complete-state policy choices must be frozen before reference dispatch.')
    verify(out/'CHOICES.npz',f['choices'])
    return f


def reveal(out,plan):
    verify_choices(out)
    fh=sha(out/'CHOICES_FROZEN.json')
    write(out/'REVEAL_STARTED.json',{**stamp(),'choice_freeze_sha256':fh})
    (out/'reference').mkdir()
    jobs=[(out,plan['physics'],r,'reference',fh) for r in read(out/'ACQUIRED.json')['held']]
    rows=use_pool(future_job,jobs,plan['workers'],out,'closed_post_freeze_reference_native_branches')
    write(out/'REFERENCES_CLOSED.json',{**stamp(),'choice_freeze_sha256':fh,'rows':rows,'native_calls':27648})


def score(out,plan):
    choice=verify_choices(out)
    p=np_load(out/'CHOICES.npz')
    reference=read(out/'REFERENCES_CLOSED.json')
    fh=sha(out/'CHOICES_FROZEN.json')
    require=(reference['choice_freeze_sha256']==fh and np.array_equal(p['units'],UNITS))
    if not require:
        raise ValueError('Independent reference must bind the complete freeze and declared mother IDs.')
    norm=plan['normalization']
    fixed,predicted,truthful=[],[],[]
    for i,(r,u) in enumerate(zip(reference['rows'],UNITS,strict=True)):
        folder=out/'reference'/str(u)
        receipt=read(verify(folder/'COMPLETE.json',r['receipt']))
        if r['unit']!=u or receipt['prediction_freeze_sha256']!=fh:
            raise ValueError('A held reference came from the wrong fresh mother or choice freeze.')
        future=np_load(verify(folder/'FUTURES.npz',receipt['values']))['values']
        if future.shape!=(24,6,3,6):
            raise ValueError('Require every post-freeze reference suffix and original action.')
        f=evaluate_choices(p['fixed'],future,norm)
        d=evaluate_choices(p['choices'][i,2],future,norm)
        fixed.append(f['selected_goal_loss'])
        predicted.append(d['selected_goal_loss'])
        truthful.append(d['finite_reference_best_goal_loss'])
    fixed,predicted,truthful=map(np.asarray,(fixed,predicted,truthful))
    delta=fixed-predicted
    half_equal=p['choices'][:,0]==p['choices'][:,1]
    main=interval(delta.mean(axis=1))
    r=np_save(out/'SCORED_ARRAYS.npz',units=np.asarray(UNITS),fixed=fixed,privileged=predicted,
              finite_reference_best=truthful,half_agreement=half_equal,each_goal_gain=delta)
    write(out/'SCORED.json',{**stamp(),'registration_sha256':sha(out/'REGISTERED.json'),
          'choice_freeze_sha256':fh,'reference_sha256':sha(out/'REFERENCES_CLOSED.json'),
          'arrays':r,'independent_mothers':MOTHERS,'planning_suffixes':32,'reference_suffixes':24,
          'original_fixed_actions':p['fixed'],'mean_fixed':fixed.mean(axis=0),'mean_privileged':predicted.mean(axis=0),
          'mean_finite_reference_best_descriptive':truthful.mean(axis=0),
          'primary_gain':main,'each_goal_gain':[interval(row) for row in delta.T],
          'headroom_supported':main['pointwise_t95'][0]>0.,
          'plan_only_half_agreement':half_equal.sum(axis=0),'plan_only_both_agree':int(half_equal.all(axis=1).sum()),
          'combined_choice_counts':np.stack([np.bincount(p['choices'][:,2,k],minlength=6) for k in range(2)]),
          'native_total':192+choice['native_calls']+reference['native_calls'],'new_scoring_native_calls':0,
          'new_wet_measurements':0,'new_fits':0})


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase',choices=('register','acquire','plan','reveal','score'))
    p.add_argument('--out',required=True,type=Path)
    p.add_argument('--inverse',type=Path)
    p.add_argument('--review',type=Path)
    p.add_argument('--workers',type=int,choices=range(1,9),default=6)
    a=p.parse_args()
    out=a.out.resolve()
    with threadpool_limits(limits=1):
        if a.phase=='register':
            if a.inverse is None or a.review is None:
                p.error('Register using completed original --inverse and --review.')
            register(out,a.inverse.resolve(strict=True),a.review.resolve(strict=True),a.workers)
        else:
            {'acquire':acquire,'plan':plan_actions,'reveal':reveal,'score':score}[a.phase](out,checked(out))


if __name__=='__main__':
    main()
