"""Prospective V4 whole-history annealing,matched SIR and model-law diagnostics.

Draft source preparation is not an experiment. Import is inert. Registration
requires a real completed Work review and fixed protocol; execution is separate.
V1 physics, targets, learned psi and strong baselines remain byte-identical.
"""
from pathlib import Path
from datetime import datetime, timezone, timedelta
from time import perf_counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse
import ast
import json
import os

from aam_formation_cycle_driver import (
    sha, read, write, plain, np_save, np_load, save_tree, worker_initializer,
)


ROOT = Path(__file__).resolve().parent
BASE = Path('H:/polymer_completion_public/research')
NAMESPACE = 'aam-formation-whole-path-tempering-v4-20261119'
MASTER = 3151920000
ARMS=('guided_rejection64_psi','annealed_path64_psi','matched_sir64_psi')
BASELINE_METHODS = ('knn8', 'extra_trees_128_leaf4', 'best_fixed')
PROTOCOL = ROOT/'AAM_FORMATION_PATH_CYCLE_V4_PROTOCOL_ZH.md'
V2_AUDIT = BASE/'aam_formation_guided_cycle_v2_audit_20261119T022910548040Z/INDEPENDENT_AUDIT.json'
MICRO_PREFLIGHT=ROOT/'logs/research/20261119T014213543579Z/V3_TRAINING_MICRO_RAW_PREFLIGHT.json'


def stamp():
    t = datetime.now(timezone.utc)
    return {'actual_utc': t.isoformat(),
            'asia_shanghai': t.astimezone(timezone(timedelta(hours=8))).isoformat()}


def event(run, action, **details):
    with (Path(run)/'EVENTS.jsonl').open('a', encoding='utf8') as stream:
        stream.write(json.dumps(plain({**stamp(), 'action': action, **details}),
                                ensure_ascii=False, allow_nan=False)+'\n')


def load_tree(path, expected=None):
    """Read the root driver's existing no-pickle tree representation."""
    from aam_formation_actual_rollout import _unpack_tree
    path = Path(path)
    if expected is not None and sha(path) != expected:
        raise ValueError('Saved numeric tree changed: '+str(path))
    arrays = np_load(path)
    metadata = json.loads(str(arrays.pop('metadata_json').item()))
    return _unpack_tree(metadata, arrays)


def source_closure():
    """AST-only local dependency closure; no science module invocation."""
    queue, found = [Path(__file__).resolve()], {}
    while queue:
        path = queue.pop()
        if str(path) in found:
            continue
        found[str(path)] = sha(path)
        tree = ast.parse(path.read_text(encoding='utf-8-sig'))
        for node in ast.walk(tree):
            names = ([item.name.split('.')[0] for item in node.names]
                     if isinstance(node, ast.Import) else
                     [node.module.split('.')[0]] if isinstance(node, ast.ImportFrom)
                     and node.module else [])
            queue.extend(ROOT/(name+'.py') for name in names
                         if (ROOT/(name+'.py')).is_file() and str(ROOT/(name+'.py')) not in found)
    return found


def _input(inputs, path, expected=None):
    path = Path(path).resolve()
    digest = sha(path)
    if expected is not None and digest != expected:
        raise ValueError('Frozen parent input changed: '+str(path))
    inputs[str(path)] = digest


def stream_contract():
    """Metadata only. Numeric uint32 collisions are retained, never deduplicated."""
    from aam_formation_path_streams import contract
    result = contract()
    result.update(rejection_attempts_max=8192,
        target='NewSMC/SIR:nu(uint32)^4*exp(beta*fullcameraLL),rawpriorwithout1/a;beta1final.OnlyfrozenV2baselinehasunknowninitialCy.',
        evidence='Relative Gaussian normalizer ledgers only;no absolute material-evidence claim.')
    return result


def register(parent,v2,v3,review,protocol=PROTOCOL,*,queries=32,v3_audit=None):
    """Bind actual source/input identities before any guide fit or new mother."""
    parent,v2,v3,review,protocol=[Path(p).resolve() for p in (parent,v2,v3,review,protocol)]
    if type(queries) is not int or queries != 32:
        raise ValueError('Work freezes exactly32new independent simulated mothers for this whole-version screening.')
    if not review.is_file() or not review.read_text(encoding='utf8').strip():
        raise ValueError('A real received Work review file is required; no fabricated approval.')
    if not protocol.is_file():
        raise ValueError('The final post-review protocol must exist before registration.')
    prior = read(parent/'REGISTERED.json')
    if (prior['namespace'] != 'aam-mbaa-property-inverse-cycle-v1-20261118'
            or not read(parent/'SCORED.json')['functional_cycle_complete']
            or not read(parent/'LEARNING_INDEPENDENT_AUDIT.json')['passed']):
        raise ValueError('Use the completed and independently checked V1 parent, not an interrupted fit.')
    for path, expected in prior['sources'].items():
        if sha(path) != expected:
            raise ValueError('Frozen V1 science source changed: '+path)
    inputs = {}
    for name in ('REGISTERED.json', 'SCORED.json', 'TRAINING_BANK.npz',
                 'PSI_FINAL.npz', 'PSI_FIT.json', 'FIT_COMPLETED.json',
                 'LEARNING_INDEPENDENT_AUDIT.json'):
        _input(inputs, parent/name)
    for path in sorted((parent/'BASELINES').rglob('*')):
        if path.is_file():
            _input(inputs, path)
    if v3_audit is None:raise ValueError('Supply the actual passedV3independentaudit.')
    v3_audit=Path(v3_audit).resolve();audit=read(v3_audit)
    if not audit['passed'] or Path(audit['run']).resolve()!=v3:
        raise ValueError('V3fullrawauditmustpassbeforeV4admission.')
    _input(inputs,v3_audit)
    for source in (v2,v3):
        _input(inputs,source/'REGISTERED.json');_input(inputs,source/'SCORED.json')
    guide=read(v2/'GUIDE_FIT_COMPLETED.json')['files']['guide']
    _input(inputs,v2/'GUIDE_FIT_COMPLETED.json');_input(inputs,v2/'GUIDE_FROZEN.npz',guide['sha256'])
    micro=read(v3/'MICRO_TRAINING_BANK.json')
    _input(inputs,v3/'MICRO_TRAINING_BANK.json');_input(inputs,v3/'MICRO_TRAINING_BANK.npz',micro['sha256'])
    _input(inputs, review)
    _input(inputs, protocol)
    streams = stream_contract()
    if streams['master'] != MASTER:
        raise ValueError('The named-stream master must match the version registration.')
    t = datetime.now(timezone.utc)
    run = BASE/t.strftime('aam_formation_path_cycle_v4_%Y%m%dT%H%M%S%fZ')
    run.mkdir()
    plan = {**stamp(), 'namespace': NAMESPACE, 'master_seed': MASTER,
            'parent':str(parent),'v2_parent':str(v2),'v3_parent':str(v3),'review': str(review), 'protocol': str(protocol),
            'sources': source_closure(), 'inputs': inputs, 'config': prior['config'],
            'queries': queries, 'reference_suffixes': 4, 'training_mothers_reused': 64,
            'new_training_mothers': 0, 'new_wet_measurements': 0,
            'workers': 4, 'BLAS_threads_per_worker': 1,
            'arms': list(ARMS), 'baseline_methods': list(BASELINE_METHODS),
            'SMC_particles': 64, 'future_B': 32, 'rejection_attempt_cap': 8192,
            'future_atoms_upper_bound':96,'new_model_fits':0,'maximum_annealing_stages':20,'teacher_queries':16,
            'new_psi_fits': 0, 'new_forest_fits': 0,
            'random_stream_contract': streams,
            'initial_guide':'FrozenV2guideusedonlybybaseline;newSMC/SIRstartfromrawprior.',
            'tempering':'N64,CESStarget.8N,20stagemax,refresh4coordinatesperparticleperstage;finalresample64andmoves.',
            'matched_SIR':'M=64+192Tindependentfullpriorpaths;select64uniformfuture slots;saveallpoolmass/history.',
            'future_allocation':'V2B32/allpositive<=96;newSMC/SIRexact64slotsxonefutureperaction.',
            'microscopic_evaluation':'FrozenV3eightstatistics,early24/full48/future24/windBrier;SIRprecompressionearlyESseparate.',
            'native_interval_upper_bound':1261926,'mechanical_snapshot_upper_bound':137504,
            'deterministic_replay_extra_native_max':6,'deterministic_replay_extra_initializations':2,
            'named_streams': streams,
            'read_boundary': 'Prediction jobs read only QUERY.npz/QUERY.json for held mothers. No PREFIX_RAW/EVAL/clean truth enters predictors.',
            'target':'nu=Unif(uint32)^4;pi_beta proportional nu*exp(beta*fullcameraLL);finalbeta1;rawpriornewarmswithout1/a.',
            'CRN': 'All6actions share each atom placement/native streams;arms use distinct rolekeys. Computational coupling only.',
            'evaluation': 'Complete hidden6, six separate18Dfuture ES, both fixed goal losses and costs; no ESS-only gate or consumed16retuning.'}
    write(run/'REGISTERED.json', plan)
    write(run/'RUN.json', {**stamp(), 'status': 'registered_not_executed',
                          'registration_sha256': sha(run/'REGISTERED.json')})
    event(run, 'prospective_whole_cycle_registered', new_native_calls=0, fits=0)
    print(str(run), flush=True)
    return run


def checked(run, *, inputs=True):
    plan = read(Path(run)/'REGISTERED.json')
    if plan['namespace'] != NAMESPACE or plan['master_seed'] != MASTER:
        raise ValueError('Wrong separately frozenV4runidentity.')
    for path, expected in plan['sources'].items():
        if sha(path) != expected:
            raise ValueError('Registered source changed: '+path)
    if inputs:
        for path, expected in plan['inputs'].items():
            if sha(path) != expected:
                raise ValueError('Registered input changed: '+path)
    return plan


def _parallel(run, plan, function, jobs, label):
    records, failures = [], []
    with ProcessPoolExecutor(max_workers=plan['workers'], initializer=worker_initializer) as pool:
        pending = {pool.submit(function, job): job['query'] for job in jobs}
        for future in as_completed(pending):
            query = pending[future]
            try:
                result = future.result()
                records.append(result)
                event(run, label, query=query, status=result.get('status'))
                print(label, query, result.get('status'), flush=True)
            except BaseException as error:
                failures.append({'query': query, 'error': repr(error)})
                event(run, label+'_failed_retained', query=query, error=repr(error))
    if failures:
        raise RuntimeError('Full registered cohort contains failures; no replacement:'+json.dumps(failures))
    return records


def prepare_bindings(run,plan):
    """Zero fits:copy exact frozen initialguide and microtrainingbank identities."""
    import shutil
    for source,target in ((Path(plan['v2_parent'])/'GUIDE_FROZEN.npz',Path(run)/'INITIAL_GUIDE_FROZEN.npz'),
                          (Path(plan['v3_parent'])/'MICRO_TRAINING_BANK.npz',Path(run)/'MICRO_TRAINING_BANK.npz')):
        if target.exists():
            if sha(target)!=sha(source):raise ValueError('Frozenresourcecopychanged.')
        else:shutil.copyfile(source,target)
    if not (Path(run)/'FROZEN_BINDINGS.json').exists():
        write(Path(run)/'FROZEN_BINDINGS.json',{**stamp(),'new_model_fits':0,
            'initialguide_sha256':sha(Path(run)/'INITIAL_GUIDE_FROZEN.npz'),
            'microbank_sha256':sha(Path(run)/'MICRO_TRAINING_BANK.npz')})


def _cache_receipt(done,*,verify_raw=False):
    result=read(done)
    records=[*result['arms'].values(),*result.get('teacher',{}).values()]
    for info in result['files'].values():
        if sha(info['path'])!=info['sha256']:raise ValueError('Completedquerycachechanged.')
    for item in records:
        for info in item['files'].values():
            if sha(info['path'])!=info['sha256']:raise ValueError('Completedarmcachechanged.')
        if verify_raw:
            if 'future_raw_index' in item['files']:
                for info in read(item['files']['future_raw_index']['path'])['raw_trajectories']:
                    if sha(info['path'])!=info['sha256']:raise ValueError('Savedfuturechanged.')
            if 'path_raw_index' in item['files']:
                for info in read(item['files']['path_raw_index']['path'])['records']:
                    if sha(info['path'])!=info['sha256']:raise ValueError('Savedpathproposalchanged.')
    return result


def _replay_path(actual,cfg,factors,measured,out):
    import numpy as np
    from aam_formation_path_tempering import generate_path
    repeated=generate_path(cfg,factors,actual['z'],measured,mother_id=actual.get('mother_id'),birth_id='deterministic-replay')
    info=save_tree(out/'DETERMINISTIC_REPLAY_RAW.npz',repeated)
    if not repeated.get('valid',True):raise RuntimeError('Newpathdeterministicreplayfailed.')
    comparisons=0
    for first,second in zip(actual['checkpoint_packets'],repeated['checkpoint_packets'],strict=True):
        for x,y in zip((*first[0],*first[1:]),(*second[0],*second[1:]),strict=True):
            comparisons+=1
            if x.dtype!=y.dtype or x.shape!=y.shape or x.tobytes()!=y.tobytes():
                raise ValueError('Deterministicnewpathcheckpointmismatch.')
    for key in ('z','clean','camera_log_likelihood'):
        if not np.array_equal(actual[key],repeated[key]):raise ValueError('ReplayLL/readoutmismatch.')
    if repeated['cost']['native_segment_completed']!=3 or repeated['cost']['actual_initializations']!=1:
        raise ValueError('Replaymustcostexactly3nativeintervalsandonefreshinitialization.')
    receipt={**stamp(),'passed':True,'comparisons':comparisons,'extra_native':3,
        'extra_initializations':repeated['cost']['actual_initializations'],'raw':info,'new_independent_mothers':0}
    write(out/'DETERMINISTIC_REPLAY.json',receipt)
    return 3


def _path_callbacks(out,cfg,factors,measured,*,replay_first=False):
    raw=out/'path_raw';raw.mkdir();index=[];extra=[0]
    def save(record,label):
        info=save_tree(raw/f'path_{len(index):07d}.npz',record)
        index.append({'label':plain(label),**info})
        if replay_first and len(index)==1:
            extra[0]=_replay_path(record,cfg,factors,measured,out)
        return info
    def load(ref):return load_tree(ref['path'],ref['sha256'])
    return save,load,index,extra


def _save_pool_micro(out,post,load):
    import numpy as np
    from aam_formation_micro_history import extract_frames
    early,wind,z=[],[],[]
    for ref in post['pool_refs']:
        path=load(ref);m,w=extract_frames(path['frames'])
        early.append(m);wind.append(w);z.append(path['z'])
    return np_save(out/'SIR_POOL_MICRO.npz',early=np.asarray(early),winding=np.asarray(wind),
        weights=np.asarray(post['pool_weights']),log_likelihood=np.asarray(post['pool_log_likelihoods']),
        z=np.asarray(z),selected_indices=np.asarray(post['selected_indices']))


def prediction_job(job):
    """Full32productionandfirst16teacherprefix-only;queryinputsremain12noisyvalues."""
    import numpy as np
    from threadpoolctl import threadpool_limits
    from aam_formation_cycle_data import load_query,property_readout,_read6,_stage,_branch_check
    from aam_formation_population_filter import filter_population
    from aam_formation_cycle_methods import load_fit,select_actions
    from aam_formation_rejection_proposal import sample_initial
    from aam_formation_path_streams import rng as named_rng,seed_draws,future_streams
    from aam_formation_streamed_rollout import formation_rollout_streamed
    from aam_formation_micro_history import extract_frames,extract_future
    from aam_formation_path_tempering import anneal_paths,sir_paths
    run,query_id=Path(job['run']),int(job['query']);plan=checked(run,inputs=False)
    cfg,parent=plan['config'],Path(plan['parent'])
    out=run/'predictions'/f'query_{query_id:03d}';out.mkdir(parents=True,exist_ok=True)
    done=out/'COMPLETE.json'
    if done.exists():return _cache_receipt(done,verify_raw=True)
    if (out/'STARTED.json').exists():raise RuntimeError('Unclosedpredictionattempt;do not repeat physical work.')
    query=load_query(run/'held'/f'parent_{query_id:03d}')
    write(out/'STARTED.json',{**stamp(),'query':query_id,'query_sha256':query['query_sha256'],
        'registration_sha256':sha(run/'REGISTERED.json'),'pid':os.getpid()})
    started,files,arm_records,teacher_records=perf_counter(),{},{},{}
    active_arm=None;arm_files={};future_receipts=[];current_cost={}
    with threadpool_limits(limits=1):
      try:
        fitted=load_fit(parent/'BASELINES');psi=np_load(parent/'PSI_FINAL.npz')['factors']
        guide=load_tree(run/'INITIAL_GUIDE_FROZEN.npz');schedule=None;stages_used=None
        for arm_index,arm in enumerate(ARMS):
            active_arm=arm;arm_out=out/arm;arm_out.mkdir();arm_files={};future_receipts=[]
            arm_started=perf_counter();replay_extra=0
            if arm_index==0:
                initialization,info=_stage(arm_out,'INITIALIZATION_RAW',
                    {'namespace':NAMESPACE,'registration_sha256':sha(run/'REGISTERED.json'),
                     'query':query_id,'arm':arm,'role':'predictive_initialization_not_truth'},
                    lambda:sample_initial(cfg,psi,query['measured'],guide,
                        named_rng(query_id,'infer_initial',arm=0),named_rng(query_id,'infer_accept',arm=0),guided=True),[])
                arm_files['initialization']=info;initialization_cost=initialization['cost']
                if not initialization['valid']:raise RuntimeError('FrozenV2initializationfailed;retaincase.')
                post=filter_population(cfg,query['measured'],MASTER,psi,
                    initial_packets=initialization['packets'],initial_log_ratios=initialization['initial_log_ratios'],
                    initialization_seeds=initialization['initialization_seeds'],
                    native_seeds=seed_draws(query_id,'infer_native',arm=0,size=(3,64)),
                    resampling_rng=named_rng(query_id,'infer_resample',arm=0),
                    unknown_proposal_normalizer=True,mother_id=f'held:{query_id}')
            else:
                save,load,index,extra=_path_callbacks(arm_out,cfg,psi,query['measured'],replay_first=query_id==0 and arm_index==1)
                if arm_index==1:
                    post=anneal_paths(cfg,psi,query['measured'],
                        named_rng(query_id,'path_initial',arm=1),named_rng(query_id,'path_seed_proposal',arm=1),
                        named_rng(query_id,'path_order',arm=1),named_rng(query_id,'path_accept',arm=1),
                        named_rng(query_id,'path_resample',arm=1),save_path=save,load_path=load,mother_id=f'held:{query_id}')
                    schedule=post['beta_schedule'];stages_used=int(post['stages_used'])
                else:
                    post=sir_paths(cfg,psi,query['measured'],named_rng(query_id,'pool_initial',arm=2),
                        named_rng(query_id,'pool_resample',arm=2),stages_used=stages_used,
                        save_path=save,load_path=load,mother_id=f'held:{query_id}')
                    if post['valid']:arm_files['pool_micro']=_save_pool_micro(arm_out,post,load)
                if not post['valid']:
                    arm_files['failed_path_posterior']=save_tree(arm_out/'FAILED_PATH_POSTERIOR.npz',post)
                    write(arm_out/'PATH_RAW_INDEX.json',{**stamp(),'records':index})
                    raise RuntimeError('Pathinferencefailed:'+str(post['error']))
                replay_extra=extra[0]
                if replay_extra:
                    rp=arm_out/'DETERMINISTIC_REPLAY.json'
                    arm_files['replay_receipt']={'path':str(rp),'sha256':sha(rp),'bytes':rp.stat().st_size}
                    arm_files['replay_raw']=read(rp)['raw']
                initialization_cost={'scope':'Rawpriorfullpathinitializationsareincludedinprefixcost.'}
                write(arm_out/'PATH_RAW_INDEX.json',{**stamp(),'records':index})
                path=arm_out/'PATH_RAW_INDEX.json';arm_files['path_raw_index']={'path':str(path),'sha256':sha(path),'bytes':path.stat().st_size}
            proposal_kind=('frozenV2guidedPF','wholepath_annealedSMC','matched_fullprior_SIR')[arm_index]
            arm_files['posterior'] = save_tree(arm_out/'POSTERIOR_RAW.npz', post)
            current_cost = {'native_interval_attempted': post['cost']['native_segment_attempted'],
                'native_interval_completed': post['cost']['native_segment_completed'],
                'native_seconds': post['cost']['native_seconds'],
                'property_seconds': 0., 'mechanical_snapshots': 0}
            if not post['valid']:
                raise RuntimeError('Actual posterior failed: '+str(post['error']))
            w = np.asarray(post['weights'])
            if (w.shape != (64,) or not np.isfinite(w).all() or np.any(w < 0.)
                    or abs(float(w.sum())-1.) > 1.e-12):
                raise ValueError('Invalid native final mass.')
            allocation = np.where(w > 0., np.maximum(1, np.ceil(32*w)).astype(np.int64), 0)
            if arm_index>0:
                if not np.allclose(w,np.full(64,1./64),rtol=0.,atol=1.e-14):raise ValueError('Newpatharmsmustuse64equalfuture slots.')
                allocation=np.ones(64,dtype=np.int64)
            if allocation.sum()>96:
                raise ValueError('All-positive-support B32 bound exceeded.')
            native = int(post['cost']['native_segment_completed'])
            attempted = int(post['cost']['native_segment_attempted'])
            native_seconds = float(post['cost']['native_seconds'])
            property_seconds, mechanical = 0., 0
            hidden,early_micro,early_winding=[],[],[]
            micro_seconds=0.
            for item,packet in zip(post['particles'],post['packets'],strict=True):
                micro_started=perf_counter()
                micro,wind=extract_frames(item['frames'])
                micro_seconds+=perf_counter()-micro_started
                early_micro.append(micro);early_winding.append(wind)
                prop = property_readout(packet, cfg)
                mechanical += 1
                property_seconds += prop['readout_seconds']
                current_cost.update(property_seconds=property_seconds, mechanical_snapshots=mechanical)
                if not prop['valid']:
                    raise RuntimeError('Invalid cutoff mechanics; not replaced by zero.')
                hidden.append(_read6(packet, prop))
            hidden=np.asarray(hidden);early_micro=np.asarray(early_micro);early_winding=np.asarray(early_winding)
            raw_dir = arm_out/'future_raw'
            raw_dir.mkdir()
            rows,atom_weights,prefix_ids,atom=[],[],[],0
            micro_rows,wind_rows=[],[]
            for prefix_id, count in enumerate(allocation):
                for _ in range(int(count)):
                    streams = future_streams(query_id, arm=arm_index, atom=atom)
                    future,micro_future,wind_future=[],[],[]
                    for action_id, action in enumerate(cfg['actions']):
                        result = formation_rollout_streamed(post['packets'][prefix_id], action, cfg,
                            properties=lambda packet: property_readout(packet, cfg),
                            mother_id=f'held:{query_id}', **streams)
                        native += result['cost']['native_segment_completed']
                        attempted += result['cost']['native_segment_attempted']
                        native_seconds += result['cost']['native_seconds']
                        property_seconds += result['cost']['property_seconds']
                        mechanical += sum(s['properties'] is not None for s in result['snapshots'])
                        current_cost.update(native_interval_attempted=attempted,
                            native_interval_completed=native, native_seconds=native_seconds,
                            property_seconds=property_seconds, mechanical_snapshots=mechanical)
                        raw = save_tree(raw_dir/f'atom_{atom:03d}_action_{action_id}.npz', result)
                        future_receipts.append({'atom': atom, 'prefix': prefix_id,
                            'action': action_id, **streams, **raw})
                        if not result['valid']:
                            raise RuntimeError('Actual future failed: '+str(result['error']))
                        _branch_check(result,post['packets'][prefix_id],action,cfg,streams['seed'])
                        micro_started=perf_counter()
                        micro,wind=extract_future(result)
                        micro_seconds+=perf_counter()-micro_started
                        micro_future.append(micro);wind_future.append(wind)
                        values = []
                        for snapshot in result['snapshots']:
                            if not snapshot['properties']['valid']:
                                raise RuntimeError('Invalid future mechanics; raw retained.')
                            values.append(_read6(snapshot['packet'], snapshot['properties'],
                                                 reported_q=snapshot['q']))
                        future.append(values)
                    rows.append(np.r_[hidden[prefix_id], np.asarray(future).reshape(108)])
                    micro_rows.append(micro_future);wind_rows.append(wind_future)
                    atom_weights.append(w[prefix_id]/int(count))
                    prefix_ids.append(prefix_id)
                    atom += 1
            draws, weights = np.asarray(rows), np.asarray(atom_weights)
            if abs(float(weights.sum())-1.) > 1.e-12:
                raise ValueError('Native posterior mass changed during future allocation.')
            selected = select_actions(draws, weights, fitted['normalization'], fitted['action_support'])
            write(arm_out/'FUTURE_RAW_INDEX.json', {**stamp(), 'raw_trajectories': future_receipts})
            index = arm_out/'FUTURE_RAW_INDEX.json'
            arm_files['future_raw_index'] = {'path': str(index.resolve()), 'sha256': sha(index),
                                            'bytes': index.stat().st_size}
            arm_files['forecast'] = np_save(arm_out/'FORECAST.npz', targets114=draws,
                weights=weights, prefix_id=np.asarray(prefix_ids),
                allocated_suffix_count=allocation, native_weights=w, cutoff64=hidden,
                choices=selected['choices'], expected_loss=selected['expected_loss'])
            arm_files['micro_forecast']=np_save(arm_out/'MICRO_FORECAST.npz',
                early=early_micro[np.asarray(prefix_ids)],future=np.asarray(micro_rows),
                early_winding=early_winding[np.asarray(prefix_ids)],future_winding=np.asarray(wind_rows),
                weights=weights,prefix_id=np.asarray(prefix_ids),prefix_early64=early_micro,
                prefix_early_winding64=early_winding)
            source_ids=np.asarray(post['source_ancestor'] if arm_index==0 else post['initialization_seeds'])
            _, group = np.unique(source_ids, return_inverse=True)
            source_mass = np.bincount(group, weights=w)
            record = {**stamp(), 'status': 'complete', 'query': query_id, 'arm': arm,
                'proposal_kind': proposal_kind, 'files': arm_files,
                'choices': selected['choices'], 'native_interval_attempted': attempted,
                'native_interval_completed': native, 'native_seconds': native_seconds,
                'property_seconds': property_seconds, 'mechanical_snapshots': mechanical,
                'positive_prefixes': int((w > 0.).sum()), 'forecast_atoms': len(draws),
                'terminal_ESS': float(1./(w@w)), 'ancestral_diversity':post.get('ancestral_diversity',{'unit':'Currentu0groupsafterMH,notoriginalparticleancestorsornumberofindependentmothers'}),
                'distinct_initial_source_ancestors': int(len(source_mass)),
                'weighted_initial_source_ESS': float(1./(source_mass@source_mass)),
                'source_ancestor_unit':'V2initialsourcegenealogy' if arm_index==0 else 'Currentu0seedgroups;notunchangedpastancestors',
                'initialization_cost':initialization_cost,'micro_readout_seconds':micro_seconds,'replay_native_extra':replay_extra,
                'filter_cost':post['cost'],
                'initial_log_mean_ratio': post.get('initial_log_mean_ratio', 0.),
                'elapsed_seconds': perf_counter()-arm_started,
                'future_distribution_from_actual_states': True, 'held_future_seen': False}
            write(arm_out/'COMPLETE.json', record)
            arm_records[arm] = record
        # Fixedfirst16only;noG,nointerventionfuture,noextraqueryinformation.
        if query_id<16:
            teacher=np.asarray(cfg['factors'])
            for diagnostic_arm,method in ((3,'annealed_smc'),(4,'matched_sir')):
                folder=out/('teacher_'+method);folder.mkdir();save,load,index,extra=_path_callbacks(
                    folder,cfg,teacher,query['measured'],replay_first=query_id==0 and method=='annealed_smc')
                if method=='annealed_smc':
                    post=anneal_paths(cfg,teacher,query['measured'],
                        named_rng(query_id,'teacher_path_initial',arm=diagnostic_arm),
                        named_rng(query_id,'teacher_path_seed_proposal',arm=diagnostic_arm),
                        named_rng(query_id,'teacher_path_order',arm=diagnostic_arm),
                        named_rng(query_id,'teacher_path_accept',arm=diagnostic_arm),
                        named_rng(query_id,'teacher_path_resample',arm=diagnostic_arm),
                        save_path=save,load_path=load,beta_schedule=schedule,mother_id=f'teacher-diagnostic:{query_id}')
                else:
                    post=sir_paths(cfg,teacher,query['measured'],
                        named_rng(query_id,'teacher_pool_initial',arm=diagnostic_arm),
                        named_rng(query_id,'teacher_pool_resample',arm=diagnostic_arm),
                        stages_used=stages_used,save_path=save,load_path=load,mother_id=f'teacher-diagnostic:{query_id}')
                tfiles={'posterior':save_tree(folder/'POSTERIOR_RAW.npz',post)}
                if not post['valid']:raise RuntimeError('Teacherprefixdiagnosticfailed:'+str(post['error']))
                early,wind=zip(*(extract_frames(path['frames']) for path in post['particles']))
                tfiles['early']=np_save(folder/'EARLY_FORECAST.npz',early=np.asarray(early),winding=np.asarray(wind),weights=post['weights'])
                if method=='matched_sir':tfiles['pool_micro']=_save_pool_micro(folder,post,load)
                write(folder/'PATH_RAW_INDEX.json',{**stamp(),'records':index})
                path=folder/'PATH_RAW_INDEX.json';tfiles['path_raw_index']={'path':str(path),'sha256':sha(path),'bytes':path.stat().st_size}
                if extra[0]:
                    rp=folder/'DETERMINISTIC_REPLAY.json'
                    tfiles['replay_receipt']={'path':str(rp),'sha256':sha(rp),'bytes':rp.stat().st_size}
                    tfiles['replay_raw']=read(rp)['raw']
                teacher_records[method]={**stamp(),'status':'complete','files':tfiles,'prefix_cost':post['cost'],
                    'replay_native_extra':extra[0],'stages_used':stages_used,'beta_schedule':schedule,
                    'no_future_or_G':True,'new_independent_mothers':0}
                write(folder/'COMPLETE.json',teacher_records[method])
        complete={**stamp(),'status':'complete','query':query_id,'query_sha256':query['query_sha256'],
            'files':files,'arms':arm_records,'teacher':teacher_records,'stages_used':stages_used,
            'beta_schedule':schedule,'held_truth_or_future_seen':False,'elapsed_seconds':perf_counter()-started}
        write(done,complete);return complete
      except BaseException as error:
        write(out/'FAILED.json',{**stamp(),'error':repr(error),'active_arm':active_arm,
            'active_arm_files':arm_files,'completed_arms':arm_records,'teacher':teacher_records,
            'completed_future_raw':future_receipts,'last_cost':current_cost,'no_case_replacement_or_retry':True})
        raise


def score_micro_complete(run,plan):
    """Summary-distribution histories,not a claim that the unique graph is known."""
    import numpy as np
    from scipy.stats import t
    from aam_formation_path_cohort import load_eval
    from aam_formation_micro_history import score_micro_history
    run=Path(run);n=plan['queries']
    bank=load_tree(run/'MICRO_TRAINING_BANK.npz')
    baseline=load_tree(run/'MICRO_BASELINE_PREDICTIONS_RAW.npz')
    methods=(*ARMS,'empirical64x2','knn8x2')
    fields=('early24_energy','full48_energy','future24_energy',
            'early_winding_brier','future_winding_brier')
    scores={key:np.empty((n,len(methods))) for key in fields}
    full_action=np.empty((n,len(methods),6));future_action=full_action.copy()
    future_wind_action=full_action.copy()
    for q in range(n):
        truth=load_eval(run,q)
        for m,name in enumerate(methods):
            if m<len(ARMS):
                receipt=read(run/'predictions'/f'query_{q:03d}'/'COMPLETE.json')['arms'][name]
                info=receipt['files']['micro_forecast']
                if sha(info['path'])!=info['sha256']:raise ValueError('Microscopic forecast changed.')
                values=np_load(info['path'])
            else:
                source=baseline['empirical' if m==len(ARMS) else 'knn8']
                values={key:source[key][q] for key in
                        ('early','future','early_winding','future_winding','weights')}
            result=score_micro_history(values['early'],values['future'],
                values['early_winding'],values['future_winding'],values['weights'],
                reference_early=truth['micro_early'],reference_future=truth['micro_future'],
                reference_early_winding=truth['micro_early_winding'],
                reference_future_winding=truth['micro_future_winding'],scale=bank['scale'])
            for key in fields:scores[key][q,m]=result[key]
            full_action[q,m]=result['per_action_full48_energy']
            future_action[q,m]=result['per_action_future24_energy']
            future_wind_action[q,m]=result['per_action_future_winding_brier']
    arrays=np_save(run/'MICRO_SCORED_CASES.npz',mother=np.arange(n),methods=np.asarray(methods),
        **scores,per_action_full48_energy=full_action,
        per_action_future24_energy=future_action,
        per_action_future_winding_brier=future_wind_action,scale=bank['scale'])
    def interval(x):
        x=np.asarray(x);mean=x.mean(axis=0)
        half=t.ppf(.975,n-1)*x.std(axis=0,ddof=1)/np.sqrt(n)
        return {'n_independent_mothers':n,'mean':mean,
                'paired_t95':np.stack((mean-half,mean+half),axis=-1)}
    return {'methods':methods,'mean':{key:value.mean(axis=0) for key,value in scores.items()},
        'annealed_minus_comparators':{name:{key:interval(value[:,1]-value[:,m])
            for key,value in scores.items()} for m,name in enumerate(methods) if m!=1},
        'score_arrays':arrays,'physical8_then_train_scale':bank['scale'],
        'forest_micro':None,'forest_micro_reason':'No microscopic-history forecast is claimed for numeric forest.',
        'scope':'Weighted distributions of8structural/event summaries over time;not uniquely reconstructed fullbondgraphs.'}


def score_prefix_diagnostics(run,plan):
    import numpy as np
    from aam_formation_path_cohort import load_eval
    from aam_formation_path_evaluation import score_early_prefix,summarize_teacher_prefix_diagnostics,summarize_sir_compression
    run=Path(run);n=plan['queries'];scale=load_tree(run/'MICRO_TRAINING_BANK.npz')['scale']
    compression=np.empty((n,2,2));teacher_scores=np.empty((16,2,2,2));teacher_compression=np.empty((16,2,2))
    teacher_native=extra_native=teacher_initializations=0
    stages=[];teacher_cost=[]
    def score(values,truth,winding_key='winding'):
        if winding_key not in values:winding_key='early_winding'
        result=score_early_prefix(values['early'],values[winding_key],values['weights'],
            reference_early=truth['micro_early'],reference_winding=truth['micro_early_winding'],scale=scale)
        return np.array([result['early24_energy'],result['early_winding_brier']])
    for q in range(n):
        truth=load_eval(run,q);receipt=read(run/'predictions'/f'query_{q:03d}'/'COMPLETE.json')
        stages.append(receipt['stages_used'])
        learned=[]
        for name in ARMS[1:]:
            arm=receipt['arms'][name]
            values=np_load(arm['files']['micro_forecast']['path'])
            learned.append(score(values,truth,'early_winding'))
        sir=receipt['arms'][ARMS[2]]
        pool=np_load(sir['files']['pool_micro']['path'])
        compression[q,0]=score(pool,truth);compression[q,1]=learned[1]
        extra_native+=sum(int(a.get('replay_native_extra',0)) for a in receipt['arms'].values())
        if q<16:
            teacher_scores[q,0]=np.asarray(learned)
            for inference,name in enumerate(('annealed_smc','matched_sir')):
                d=receipt['teacher'][name];cost=d['prefix_cost']
                teacher_native+=int(cost['native_segment_completed'])
                teacher_initializations+=int(cost['actual_initializations'])
                extra_native+=int(d['replay_native_extra'])
                teacher_cost.append({'query':q,'method':name,**cost})
                values=np_load(d['files']['early']['path'])
                teacher_scores[q,1,inference]=score(values,truth)
                if name=='matched_sir':
                    pool=np_load(d['files']['pool_micro']['path'])
                    teacher_compression[q,0]=score(pool,truth)
                    teacher_compression[q,1]=teacher_scores[q,1,inference]
    if extra_native!=6:raise ValueError('Registerednewlearned/teacherreplaysmustcostexactly6nativeintervals.')
    arrays=np_save(run/'PREFIX_DIAGNOSTICS_CASES.npz',production_mother=np.arange(n),teacher_mother=np.arange(16),
        stages_used=np.asarray(stages),production_sir_compression=compression,
        teacher_cells=teacher_scores,teacher_sir_compression=teacher_compression,
        metric_names=np.asarray(['early24_energy','early_winding_brier']))
    return {'arrays':arrays,'teacher_final64':summarize_teacher_prefix_diagnostics(teacher_scores),
        'production_sir_pre_vs_post_compression':summarize_sir_compression(compression,n_mothers=32),
        'teacher_sir_pre_vs_post_compression':summarize_sir_compression(teacher_compression,n_mothers=16),
        'teacher_native_intervals':teacher_native,'teacher_actual_initializations':teacher_initializations,
        'teacher_cost_by_case':teacher_cost,'extra_deterministic_replay_native':extra_native,
        'extra_deterministic_replay_initializations':2,'stages_used':stages,
        'teacher_scope':'First16fixedmotherprefix-only;samelearnedbeta/T/budget,independentstreams;notdeployableornotposteriorgoldstandard.',
        'compression_scope':'PrepoolSIRsupportisnotusedasafinal64teachercomparisoncell;noallpoolmasspreservationclaimforfuture.'}


def score_complete(run, plan, fitted, freeze, started):
    """All consumed outcomes only after complete prediction/choice freezing."""
    import numpy as np
    from scipy.stats import t
    from aam_formation_path_cohort import load_eval
    from aam_formation_cycle_methods import score_forecast, evaluate_choices
    n = plan['queries']
    methods = (*ARMS, *BASELINE_METHODS)
    m = len(methods)
    es, hidden = np.empty((n, m)), np.empty((n, m))
    per_action = np.empty((n, m, 6))
    loss, choices = np.empty((n, m, 2)), np.empty((n, m, 2), dtype=np.int64)
    finite_best, ess = np.empty((n, 2)), np.empty((n, len(ARMS)))
    ancestor_ess=np.empty((n,len(ARMS)))
    native_cost = np.zeros((n,len(ARMS)),dtype=np.int64)
    mechanical_cost = np.zeros((n,len(ARMS)),dtype=np.int64)
    initialization_cost=[]
    filter_costs=[]
    reference_native = reference_attempted = reference_mechanics = 0
    reference_native_seconds = reference_property_seconds = 0.
    baseline = np_load(run/'BASELINE_PREDICTIONS.npz')
    for q in range(n):
        truth = load_eval(run, q)
        # Actual completed receipts, not the planned budget, determine costs.
        prefix_receipt = read(run/'held'/f'parent_{q:03d}'/'QUERY_COMPLETE.json')
        eval_receipt = read(run/'held'/f'parent_{q:03d}'/'EVAL_COMPLETE.json')
        ref_costs = [prefix_receipt['native_cost'], *eval_receipt['reference_cost']]
        reference_native += sum(int(row['native_segment_completed']) for row in ref_costs)
        reference_attempted += sum(int(row['native_segment_attempted']) for row in ref_costs)
        reference_native_seconds += sum(float(row['native_seconds']) for row in ref_costs)
        reference_property_seconds += (float(eval_receipt['cutoff_property_seconds'])
            + sum(float(row['property_seconds']) for row in eval_receipt['reference_cost']))
        # Every saved valid reference branch is independently checked by the
        # cohort helper to contain3successfully read mechanical snapshots.
        reference_mechanics += 1+3*len(eval_receipt['reference_cost'])
        query_receipt = _cache_receipt(run/'predictions'/f'query_{q:03d}'/'COMPLETE.json')
        for index, name in enumerate(methods):
            if index < len(ARMS):
                receipt = query_receipt['arms'][name]
                initialization_cost.append({'query':q,'arm':name,**receipt['initialization_cost']})
                filter_costs.append({'query':q,'arm':name,**receipt['filter_cost']})
                f = np_load(receipt['files']['forecast']['path'])
                draws, w, selected = f['targets114'], f['weights'], f['choices']
                ess[q, index] = receipt['terminal_ESS']
                ancestor_ess[q, index] = receipt['weighted_initial_source_ESS']
                native_cost[q, index] = receipt['native_interval_completed']
                mechanical_cost[q, index] = receipt['mechanical_snapshots']
            else:
                draws, w, selected = [baseline[name+'__'+key][q]
                                     for key in ('targets114', 'weights', 'choices')]
            s = score_forecast(draws, w, reference_future=truth['future_values'],
                              reference_cutoff=truth['cutoff_values'],
                              normalization=fitted['normalization'])
            c = evaluate_choices(selected, truth['future_values'], fitted['normalization'])
            es[q, index], hidden[q, index] = s['primary_future_energy'], s['hidden6_energy']
            per_action[q, index] = s['per_action_future_energy']
            loss[q, index], choices[q, index] = c['selected_goal_loss'], selected
            finite_best[q] = c['finite_reference_best_goal_loss']
    arrays = np_save(run/'SCORED_CASES.npz', mother=np.arange(n), methods=np.asarray(methods),
        future_energy=es, cutoff_energy=hidden, per_action_future_energy=per_action,
        selected_loss=loss, equal_mean_goal_loss=loss.mean(axis=-1), choices=choices, finite_reference_best_loss=finite_best,
        native_terminal_ESS=ess, initial_source_ancestor_ESS=ancestor_ess,
        native_interval_cost=native_cost, mechanical_snapshot_cost=mechanical_cost)
    def interval(values):
        values = np.asarray(values)
        mean = values.mean(axis=0)
        half = t.ppf(.975, n-1)*values.std(axis=0, ddof=1)/np.sqrt(n)
        return {'n_independent_mothers': n, 'mean': mean,
                'paired_t95': np.stack((mean-half, mean+half), axis=-1)}
    prefix_diagnostics=score_prefix_diagnostics(run,plan)
    total_native=int(native_cost.sum())+reference_native+prefix_diagnostics['teacher_native_intervals']+prefix_diagnostics['extra_deterministic_replay_native']
    total_mechanics = int(mechanical_cost.sum())+reference_mechanics
    if (total_native > plan['native_interval_upper_bound']
            or total_mechanics > plan['mechanical_snapshot_upper_bound']):
        raise ValueError('Actual full cohort exceeded its registered complete-cycle budget.')
    micro_report=score_micro_complete(run,plan)
    report = {**stamp(), 'functional_cycle_complete': True,
        'namespace': NAMESPACE, 'new_held_mothers': n, 'reused_training_mothers': 64,
        'new_psi_fits': 0, 'new_forest_fits': 0, 'new_model_fits':0,
        'methods': methods, 'mean_future_energy': es.mean(axis=0),
        'mean_cutoff_energy': hidden.mean(axis=0), 'mean_selected_goal_loss': loss.mean(axis=0),
        'mean_equal_mean_goal_loss': loss.mean(axis=(0, 2)),
        'mean_finite_reference_best_loss': finite_best.mean(axis=0),
        'annealed_minus_comparators': {name: {'future_energy': interval(es[:, 1]-es[:, k]),
            'cutoff_energy': interval(hidden[:, 1]-hidden[:, k]),
            'selected_goal_loss': interval(loss[:, 1]-loss[:, k]),
            'equal_mean_goal_loss': interval(loss[:, 1].mean(axis=-1)-loss[:, k].mean(axis=-1))}
            for k, name in enumerate(methods) if k != 1},
        'choice_counts': {name: [np.bincount(choices[:, k, goal], minlength=6) for goal in range(2)]
                          for k, name in enumerate(methods)},
        'native_terminal_ESS_explanatory_only_min_median_max': np.stack((ess.min(axis=0), np.median(ess, axis=0), ess.max(axis=0))),
        'initial_source_ancestor_ESS_min_median_max': np.stack((ancestor_ess.min(axis=0),
            np.median(ancestor_ess, axis=0), ancestor_ess.max(axis=0))),
        'cost': {'native_predictive_by_arm': native_cost.sum(axis=0),
                 'reference_native_intervals': reference_native,
                 'reference_native_attempted': reference_attempted,
                 'reference_native_seconds': reference_native_seconds,
                 'reference_property_seconds': reference_property_seconds,
                 'total_native_intervals':total_native,
                 'teacher_prefix_native':prefix_diagnostics['teacher_native_intervals'],
                 'deterministic_replay_extra_native':prefix_diagnostics['extra_deterministic_replay_native'],
                 'mechanical_predictive_by_arm': mechanical_cost.sum(axis=0),
                 'reference_mechanical_snapshots': reference_mechanics,
                 'total_mechanical_snapshots': total_mechanics,
                 'initialization_by_query_arm':initialization_cost,'filter_by_query_arm':filter_costs,
                 'guided_initialization_arms':1,'rawprior_path_arms':2,
                 'native_budget_only_not_equal_walltime': True},
        'score_arrays':arrays,'prediction_freeze':freeze,'microscopic':micro_report,'prefix_diagnostics':prefix_diagnostics,
        'timing_scope':'Callback/archive seconds can include firstsame-zreplaytime;componentsecondsareoverlapping,notsummedtowalltime.',
        'final64ESS_not_primary':True,
        'same_query_information': 'All methods only12noisy readings;guide/psi additionally use microscopic training annotation.',
        'initial_proposal_boundary':'OnlyV2baselineusesfrozenrejectionguide;newtwoarmsrawpriorwholepathposterior.Final64equalmassforfutureisexplicitMCapproximation,notpoolmasspreservation.',
        'fixed_design_predictive_score': 'Unconditioned training prior is descriptive, not a same-information conditional winner.',
        'elapsed_seconds_this_execution': perf_counter()-started,
        'scientific_goal_achieved': False,
        'next': 'Independent whole-version audit, archive and Work review;decide from full process/inverse evidence,not ESS alone.'}
    write(run/'SCORED.json', report)
    return report


def execute(run):
    import numpy as np
    from threadpoolctl import threadpool_limits
    from aam_formation_path_cohort import prefix_job, reveal_job, freeze_receipt
    from aam_formation_cycle_data import load_query
    from aam_formation_cycle_methods import load_fit, predict_methods
    from aam_formation_micro_history import micro_baseline_predictions
    run = Path(run).resolve()
    plan = checked(run)
    descriptor = os.open(run/'DRIVER_ACTIVE.lock', os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    os.write(descriptor, str(os.getpid()).encode())
    os.close(descriptor)
    started = perf_counter()
    try:
        with threadpool_limits(limits=1):
            prepare_bindings(run,plan)
            micro_bank=load_tree(run/'MICRO_TRAINING_BANK.npz')
            fitted = load_fit(Path(plan['parent'])/'BASELINES')
            if not (run/'ALL_PREDICTIONS_AND_SELECTIONS_FROZEN.json').exists():
                jobs = [{'run': str(run), 'query': q} for q in range(plan['queries'])]
                _parallel(run, plan, prefix_job, jobs, 'new_held_query_complete')
                X = np.stack([load_query(run/'held'/f'parent_{q:03d}')['X']
                              for q in range(plan['queries'])])
                if not (run/'BASELINE_PREDICTIONS.npz').exists():
                    predictions = predict_methods(fitted, X)
                    save_tree(run/'BASELINE_PREDICTIONS_RAW.npz', predictions)
                    arrays = {name+'__'+key: value[key] for name, value in predictions.items()
                              for key in ('targets114', 'weights', 'choices', 'expected_loss')}
                    np_save(run/'BASELINE_PREDICTIONS.npz',**arrays)
                if not (run/'MICRO_BASELINE_PREDICTIONS_RAW.npz').exists():
                    micro_predictions=micro_baseline_predictions(micro_bank,X)
                    save_tree(run/'MICRO_BASELINE_PREDICTIONS_RAW.npz',micro_predictions)
                prediction_job(jobs[0])
                event(run,'firstquery_replay_and_fullprediction_complete',query=0)
                _parallel(run,plan,prediction_job,jobs[1:],'all_native_arms_and_design_complete')
                files={str(run/name):sha(run/name) for name in
                    ('BASELINE_PREDICTIONS.npz','BASELINE_PREDICTIONS_RAW.npz',
                     'INITIAL_GUIDE_FROZEN.npz','MICRO_TRAINING_BANK.npz','MICRO_BASELINE_PREDICTIONS_RAW.npz','FROZEN_BINDINGS.json')}
                for q in range(plan['queries']):
                    folder=run/'predictions'/f'query_{q:03d}'
                    _cache_receipt(folder/'COMPLETE.json')
                    for path in folder.rglob('*'):
                        if path.is_file():files[str(path)]=sha(path)
                write(run/'ALL_PREDICTIONS_AND_SELECTIONS_FROZEN.json', {**stamp(),
                    'prediction_and_selection_frozen': True, 'queries': plan['queries'],
                    'all_forecast_files': files, 'sources': plan['sources'],
                    'zero_fit_bindings_sha256':sha(run/'FROZEN_BINDINGS.json'),'teacher_prefix_diagnostics':16,
                    'held_futures_revealed': False})
            freeze = freeze_receipt(run, verify_all=True)
            _parallel(run, plan, reveal_job,
                [{'run': str(run), 'query': q} for q in range(plan['queries'])],
                'new_held_reference_complete')
            if (run/'SCORED.json').exists():
                return read(run/'SCORED.json')
            report = score_complete(run, plan, fitted, freeze, started)
            event(run, 'whole_cycle_scored', functional_cycle_complete=True,
                  scientific_goal_achieved=False)
            print(json.dumps(plain(report), ensure_ascii=True, indent=2), flush=True)
            return report
    except BaseException as error:
        event(run, 'driver_interrupted_retained', error=repr(error), no_automatic_retry=True)
        raise
    finally:
        (run/'DRIVER_ACTIVE.lock').unlink()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--register', action='store_true')
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--parent', type=Path)
    parser.add_argument('--review', type=Path)
    parser.add_argument('--protocol', type=Path, default=PROTOCOL)
    parser.add_argument('--queries',type=int,default=32)
    parser.add_argument('--v3',type=Path)
    parser.add_argument('--v3-audit',type=Path,required=False)
    parser.add_argument('--v2',type=Path)
    parser.add_argument('--run', type=Path)
    args = parser.parse_args()
    if args.register and args.parent is not None and args.v2 is not None and args.v3 is not None and args.review is not None:
        register(args.parent,args.v2,args.v3,args.review,args.protocol,queries=args.queries,v3_audit=args.v3_audit)
    elif args.execute and args.run is not None:
        execute(args.run)
    else:
        parser.error('Use --register --parent COMPLETED_V1 --v2 COMPLETED_V2 --review REAL_REVIEW then --execute --run NEW_RUN.')
