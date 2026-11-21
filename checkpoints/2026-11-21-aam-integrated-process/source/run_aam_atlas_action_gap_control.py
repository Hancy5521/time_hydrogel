"""Prospective training-only variance gate for frozen legal atlas gap control."""

import argparse
from pathlib import Path
from shutil import copyfile

import numpy as np
from threadpoolctl import threadpool_limits

from aam_atlas_action_gap_control import FOLDS, RIDGE, fit_atlas, fixtures, route_risk, validate_training_gate
from aam_camera_information import FEATURE_NAMES
from aam_formation_cycle_driver import np_load, np_save, read, sha, write
from aam_formation_cycle_methods import GOALS, process_goal_losses
from aam_formation_micro_history import NAMES
from aam_formation_path_cycle import event, load_tree, stamp
from package_aam_formation_local_cycle import source_closure
from run_aam_direct_energy_trajectory_law import atlas_receipts
from run_aam_mark_candidate_filter import verify


ROOT = Path(__file__).resolve().parent
PROTOCOL = ROOT/'AAM_ATLAS_ACTION_GAP_CONTROL_ZH.md'
AUDITOR = ROOT/'audit_aam_atlas_action_gap_control.py'
NS = 'aam-legal-atlas-action-gap-variance-gate-trainonly-v1'
TIMES = (2.30,2.3875,2.5625,2.7375,2.9125,3.)
REFERENCE_GRID = tuple((k,p) for k in (0,5) for p in (0.,.5,1.))
GLOBAL = (1,2,3,7)
FEATURES = tuple(f'{block}/port{port}/{name}' for block in ('C3','C3_minus_C2')
                 for port in (0,1) for name in FEATURE_NAMES) + tuple('C3/global/'+NAMES[j] for j in GLOBAL)


def read_inputs(atlas, population, inverse):
    temporal, units, path, temporal_refs = atlas_receipts(atlas)
    p_names = ('REGISTERED.json','SCORED.json','INDEPENDENT_AUDIT.json')
    i_names = ('REGISTERED.json','LIBRARY_COMMITTED.json','INDEPENDENT_AUDIT.json')
    p_refs = tuple(population/v for v in p_names)
    i_refs = tuple(inverse/v for v in i_names)
    p, summary, pa = map(read,p_refs)
    inv, library, ia = map(read,i_refs)
    if (p['namespace']!='aam-continuous-fixed-population-train48-rank-law-v1'
            or inv['namespace']!='aam-temporal-law-v5-original-goals-fresh96-v1'
            or pa['passed'] is not True or ia['passed'] is not True
            or pa['registration_sha256']!=sha(p_refs[0]) or pa['scored_sha256']!=sha(p_refs[1])
            or ia['registered_sha256']!=sha(i_refs[0]) or ia['scored_sha256']!=sha(inverse/'SCORED.json')
            or library['registration_sha256']!=sha(i_refs[0])
            or pa['old_held_opened']!=0 or ia['old_held_target_payloads_read']!=0
            or pa['independent_train_mothers']!=48 or ia['library_mothers']!=512
            or list(units)!=inv['train_ids'] or p['physics']!=inv['physics'] or p['physics']!=temporal['physics']
            or p['normalization']!=inv['normalization'] or p['times']!=list(TIMES)
            or p['goals']!=list(GOALS) or inv['goals']!=list(GOALS)
            or Path(p['atlas']).resolve()!=atlas or Path(inv['temporal_run']).resolve()!=atlas):
        raise ValueError('Require three verified, unchanged training-only archives of the original objective.')
    positions=p['positions']
    selected=[units[i] for i in positions]
    protocol_positions=[(2*i+1)*512//96 for i in range(48)]
    if (positions!=protocol_positions or [r['unit'] for r in p['training_mothers']]!=selected
            or [r['unit'] for r in summary['raw_mothers']]!=selected
            or summary['old_held_opened']!=0 or len(set(selected))!=48):
        raise ValueError('The audited training population must map exactly to the original atlas positions.')
    pfile=verify(population/'POPULATION_ARRAYS.npz',summary['arrays'])
    ifile=verify(inverse/'TRAIN_FUTURES.npz',library['arrays'])
    data={'atlas':str(path),'population':str(pfile),'anchor_training':str(ifile)}
    files=(*temporal_refs,*p_refs,*i_refs,inverse/'SCORED.json',pfile,ifile)
    return units,positions,inv['normalization'],data,files


def register(out,atlas,population,inverse,work):
    units,positions,norm,data,files=read_inputs(atlas,population,inverse)
    if not work.read_text(encoding='utf-8-sig').strip() or not AUDITOR.is_file():
        raise ValueError('Bind the completed Work assessment and independent score checker.')
    closure=source_closure([Path(__file__),AUDITOR])
    p=out/'REGISTERED.json'
    out.mkdir(exist_ok=False)
    write(p,{**stamp(),'namespace':NS,'source_roles':data,'atlas_ids':units,'population_positions':positions,
             'features':FEATURES,'feature_count':16,'goals':GOALS,'times':TIMES,
             'reference_action_grid':REFERENCE_GRID,
             'reference_selection':'original index argmin of the 48 donor mean Bernstein losses, first index on tie',
             'ridge':RIDGE,'oof_folds':FOLDS,'normalization':norm,
             'validation_design':'remaining original train mothers, matched by ordered consecutive pairs',
             'gate':'both original goals have lower t95 bounds above zero on between-state contrast covariance and held-out-train variance reduction',
             'native_calls_authorized':0,'expected_fits':5,'old_or_new_held_payloads_authorized':0,
             'inputs':{str(f):sha(f) for f in (*files,work,PROTOCOL)},
             'sources':{str(f):sha(f) for f in closure.values()}})
    snapshot=out/'SOURCE_SNAPSHOT'
    snapshot.mkdir()
    for name,source in closure.items():
        copyfile(source,snapshot/name)
    write(out/'RUN.json',{**stamp(),'registration_sha256':sha(p),'registered_before_real_fit':True})
    event(out,'registered_training_only_control_variate_check',models=5,native_calls=0)


def checked(out):
    p=out/'REGISTERED.json'
    plan=read(p)
    if (plan['namespace']!=NS or plan['features']!=list(FEATURES)
            or plan['reference_action_grid']!=[list(u) for u in REFERENCE_GRID]
            or read(out/'RUN.json')['registration_sha256']!=sha(p)):
        raise ValueError('Use the original prospective training-only registration.')
    for path,digest in {**plan['sources'],**plan['inputs']}.items():
        if sha(path)!=digest:
            raise ValueError(f'Frozen input/source changed: {path}.')
    return plan


def legal_features(atlas):
    local=atlas['bank']['local']
    early=atlas['bank']['global_early']
    if local.shape!=(512,3,2,3) or early.shape!=(512,3,8):
        raise ValueError('The complete audited original training bank is required.')
    last=local[:,2].reshape(512,6)
    delta=(local[:,2]-local[:,1]).reshape(512,6)
    return np.column_stack((last,delta,early[:,2,GLOBAL]))


def donor_reference(losses):
    if losses.shape!=(48,6,7,2):
        raise ValueError('Exactly 48 complete donor route tables determine the anchors.')
    mean=losses.mean(axis=0)
    values=np.stack([route_risk(mean,[j,j],[p,p]) for j,p in REFERENCE_GRID])
    winners=np.argmin(values,axis=0)
    times,left=zip(*(REFERENCE_GRID[j] for j in winners),strict=True)
    return winners,times,left,values


def fit(out,plan):
    reg=sha(out/'REGISTERED.json')
    write(out/'FIT_STARTED.json',{**stamp(),'registration_sha256':reg,
          'inputs_train_only':plan['source_roles'],'old_or_new_held_payloads_read':0})
    sources=plan['source_roles']
    atlas=load_tree(sources['atlas'],plan['inputs'][sources['atlas']])
    population=np_load(sources['population'])
    try:
        chosen,times,left,reference_scores=donor_reference(population['losses'])
        selection=out/'REFERENCE_FROZEN.json'
        write(selection,{**stamp(),'registration_sha256':reg,'population_sha256':plan['inputs'][sources['population']],
              'population_positions':plan['population_positions'],'prior_anchor_suffixes_opened':False,
              'candidate_mean_losses':reference_scores,'reference_action_indices':chosen,
              'reference_time_indices':times,'reference_left_fractions':left})
        anchors=np_load(sources['anchor_training'])
        x=legal_features(atlas)
        if anchors['future'].shape!=(512,2,6,3,6) or not np.array_equal(anchors['units'],plan['atlas_ids']):
            raise ValueError('Retain the exactly aligned original two anchor suffixes.')
        control=fit_atlas(x,plan['population_positions'],population['losses'],times,left)
        actual=process_goal_losses(anchors['future'],plan['normalization'])['loss']
        gate=validate_training_gate(control,actual,chosen)
    except Exception as exc:
        write(out/'FIT_FAILED.json',{**stamp(),'registration_sha256':reg,'error':repr(exc),
                                    'native_calls':0,'old_or_new_held_payloads_read':0})
        raise
    result=np_save(out/'CONTROL_ARRAYS.npz',features=x,coefficients=control.coefficients,
                   training_positions=control.training_positions,training_gaps=control.training_gaps,
                   training_oof=control.training_oof,training_oof_constant=control.training_oof_constant,
                   anchor_losses=actual,validation_pairs=gate['validation_pairs'],
                   spare_positions=gate['spare_positions'])
    write(out/'SCORED.json',{**stamp(),'registration_sha256':reg,'arrays':result,
          'reference_frozen_sha256':sha(selection),'reference_action_indices':chosen,
          'fit_mothers':48,'disjoint_validation_mothers':len(gate['validation_pairs'])*2,
          'validation_pairs':len(gate['validation_pairs']),'by_goal':dict(zip(GOALS,gate['by_goal'],strict=True)),
          'training_oof_mse_difference_by_goal':gate['training_oof_mse_difference_by_goal'],
          'both_goals_positive':gate['both_goals_positive'],'new_native_calls':0,'new_fits':5,
          'old_or_new_held_payloads_read':0,'effect_scope':'same old training anchors only, no efficacy confirmation'})
    event(out,'training_only_variance_gate_closed',passed=gate['both_goals_positive'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=('fixtures','register','fit'))
    parser.add_argument('--out',type=Path)
    for arg in ('atlas','population','inverse','work'):
        parser.add_argument('--'+arg,type=Path)
    args=parser.parse_args()
    with threadpool_limits(limits=1):
        if args.phase=='fixtures':
            print(fixtures())
            return
        if args.out is None:
            parser.error('Provide --out for a prospective real training execution.')
        out=args.out.resolve()
        if args.phase=='register':
            source=(args.atlas,args.population,args.inverse,args.work)
            if any(p is None for p in source):
                parser.error('Provide --atlas, --population, --inverse and --work.')
            register(out,*(p.resolve(strict=True) for p in source))
        else:
            fit(out,checked(out))


if __name__=='__main__':
    main()
