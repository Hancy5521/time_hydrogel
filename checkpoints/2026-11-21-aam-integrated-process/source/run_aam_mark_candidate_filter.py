"""Prospective shared-first-two-camera native bootstrap versus mark-guided SMC."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
from pathlib import Path
import shutil
from time import perf_counter

import numpy as np
from scipy.stats import t
from threadpoolctl import threadpool_limits

from aam_camera_information import energy, history_features
from aam_camera_mark_tilt import MAX_TRIALS, SCHEMA, make_interval_table, mark_only_interval
from aam_formation_actual_rollout import _packet_copy, packet_fingerprint
from aam_formation_cycle_driver import np_load, np_save, read, save_tree, write
from aam_formation_local_initial import build_packet, draw_initial
from aam_formation_micro_history import extract_frames
from aam_formation_path_cycle import event, load_tree, stamp
from aam_snapshot_process import observe
from package_aam_formation_local_cycle import sha, source_closure


ROOT = Path(__file__).resolve().parent
NAMESPACE = 'aam-role-addressed-v1-shared-camera2-third-mark-filter'
ENTROPY = [int.from_bytes(hashlib.sha256(NAMESPACE.encode()).digest()[i:i+4],'little') for i in range(0,16,4)]
TRAIN, HELD, PARTICLES, WORKERS = 64, 24, 64, 3
GROUP = {'train':1, 'held':2, 'infer':3}
ROLE = {'initial':1, 'bd':2, 'waiting':3, 'mark':4, 'camera_noise':5, 'systematic':6, 'retry':7}
ARMS = ('ordinary','marked')
COMPARATORS = ('ordinary','knn8','empirical')


def rng(group, unit, role, camera=0, slot=0):
    return np.random.default_rng(np.random.SeedSequence(ENTROPY+[GROUP[group],unit,ROLE[role],camera,slot]))


def table(cfg, group, unit, camera, slot):
    return make_interval_table(cfg,camera,*(rng(group,unit,role,camera,slot) for role in ('bd','waiting','mark')))


def register(out, v5_reg, fixture, review):
    predecessor, eng, proof = read(v5_reg), read(fixture/'REGISTERED.json'), read(fixture/'RESULT.json')
    if (predecessor['namespace'] != 'aam-formation-explicit-initial-local-v5-20261119'
            or eng['namespace'] != 'aam-mark-tilt-engineering-fixture-v1'
            or proof['passed'] is not True or proof['new_scored_mothers'] != 0 or proof['native_intervals'] != 9
            or proof['registration_sha256'] != sha(fixture/'REGISTERED.json')
            or proof['finite_grid']['exact_proposal_and_importance_normalization'] is not True
            or eng['physics'] != predecessor['config']):
        raise ValueError('Require the original physical configuration and actual passing engineering fixture.')
    cfg = predecessor['config']
    if tuple(cfg['prefix_clock']) != (0.,.65,1.35,2.05) or tuple(cfg['noise']) != (.08,.10,.08,.10):
        raise ValueError('The three original causal camera times and error law must be frozen.')
    sources = source_closure([Path(__file__), ROOT/'audit_aam_mark_candidate_filter.py'])
    hashes = {str(path):sha(path) for path in sources.values()}
    for manifest in (predecessor,eng):
        if any(hashes[name] != digest for name,digest in manifest['sources'].items() if name in hashes):
            raise ValueError('The originally tested native/source code changed.')
    raw = Path(proof['raw']['path']).resolve(strict=True)
    if raw.parent != fixture or sha(raw) != proof['raw']['sha256']:
        raise ValueError('The actual engineering fixture raw archive is unbound.')
    inputs = (v5_reg, fixture/'REGISTERED.json', fixture/'RESULT.json', raw, review,
              ROOT/'AAM_CAMERA_MARK_TILT_PROTOCOL_ZH.md')
    out.mkdir(exist_ok=False)
    path = out/'REGISTERED.json'
    write(path,{**stamp(),'namespace':NAMESPACE,'entropy':ENTROPY,'group_codes':GROUP,'role_codes':ROLE,
         'physics':cfg,'native_law':SCHEMA,'teacher_factors':cfg['factors'],'n_training':TRAIN,'n_held':HELD,
         'particles_per_arm':PARTICLES,'workers':WORKERS,'resampling':'common systematic after cameras 0 and 1; final exact weights',
         'guide_camera_index':2,'max_guide_trials':MAX_TRIALS,'arms':ARMS,'primary_gate_comparators':COMPARATORS,
         'primary':'joint early18 normalized energy for 3 times × 2 held apertures × 3 active structural observables',
         'secondary':'joint early24 energy, eight global microhistory metrics, scales fresh train std floor 0.05',
         'auxiliary_hidden_metrics':{'hidden_t3_6':'six-dimensional third-frame hidden energy',
            'hidden_delta32_6':'six-dimensional hidden energy of each history’s third frame minus its second frame'},
         'final_ancestor_ess':'Final normalized weights grouped by the actual pre-resampling second-frame parent ID.',
         'third_execution_order':'Ordinary first for even mother+slot, marked first for odd. Addresses are unchanged.',
         'knn_coordinate_scale':'fresh training population std floored at original camera noise; stable mother-ID tie break',
         'budget_native':{'data':3*(TRAIN+HELD),'each_comparison_arm':HELD*3*PARTICLES,'actual_inference_shared':HELD*4*PARTICLES},
         'old_observations':0,'new_fits':0,'new_wet_measurements':0,
         'sources':hashes,'inputs':{str(p):sha(p) for p in inputs}})
    snap = out/'SOURCE_SNAPSHOT'; snap.mkdir()
    for name, source in sources.items():
        shutil.copyfile(source, snap/name)
    write(out/'RUN.json',{**stamp(),'registered_before_draws':True,'registration_sha256':sha(path)})
    event(out,'registered_shared_prefix_mark_filter',fresh_training=TRAIN,fresh_held=HELD)


def checked(out):
    path = out/'REGISTERED.json'
    plan = read(path)
    if (plan['namespace'] != NAMESPACE or plan['entropy'] != ENTROPY or plan['guide_camera_index'] != 2
            or sha(path) != read(out/'RUN.json')['registration_sha256']):
        raise ValueError('No exact prospective shared-prefix registration.')
    for name,digest in {**plan['inputs'],**plan['sources']}.items():
        if sha(name) != digest:
            raise ValueError('Registered source or evidence changed: '+name)
    return plan


def verify(path, receipt):
    if str(path.resolve()) != receipt['path'] or path.stat().st_size != receipt['bytes'] or sha(path) != receipt['sha256']:
        raise ValueError('Saved artifact identity changed: '+str(path))
    return path


def interval_evidence(record):
    """Save CDF and score sufficient for corrected tilted marks, omit redundant candidate buffers."""
    compact = []
    for row in record['spans']:
        item = {key:row[key] for key in ('slab','attempt','begin','end','exponential','target_A','censored')}
        if not row['censored']:
            item.update({key:row[key] for key in ('uniform','chosen','geometry_time','mark_tilt')})
            if 'actual_candidate_cumulative' in row:
                item.update(cdf=row['actual_candidate_cumulative'],score=row['guide_score'])
        compact.append(item)
    return {'spans':compact,'tape':record['tape'],'log_ratio':record['log_target_over_proposal'],
            'native_calls':record['native_interval_calls'],'seconds':record['seconds']}


def apply_interval(packet, cfg, camera, tables, counts, *, query=None, retry=None):
    record = mark_only_interval(packet,cfg['prefix_clock'][:camera+2],camera,tables,
                               measured=query,noise=cfg['noise'] if query is not None else None,
                               retry_seed=retry,record_full=query is not None,counts=counts)
    return interval_evidence(record)


def born(plan, group, identity, slot=0):
    cfg = plan['physics']
    initial = draw_initial(cfg,rng(group,identity,'initial',slot=slot))
    return build_packet(cfg,initial,factors=np.asarray(plan['teacher_factors']))


def global_history(packets, clock):
    return extract_frames({age:packet[0] for age,packet in zip(clock,packets,strict=True)})[0]


def physical_data(plan, group, unit):
    cfg = plan['physics']
    packet = born(plan,group,unit)
    packets = [_packet_copy(packet,readonly=True)]
    counts, ledger = {'attempted':0,'completed':0,'failed':0}, []
    for camera in range(3):
        r = apply_interval(packet,cfg,camera,table(cfg,group,unit,camera,0),counts)
        ledger.append(r)
        packets.append(_packet_copy(packet,readonly=True))
    y = np.stack([observe(p) for p in packets[1:]])
    noisy = y + np.asarray(cfg['noise'])*rng(group,unit,'camera_noise').standard_normal((3,4))
    return noisy, {'group':group,'unit':unit,'packets':packets,'clean':y,'intervals':ledger,'cost':counts}


def acquire(out, plan):
    write(out/'ACQUISITION_STARTED.json',{**stamp(),'registration_sha256':sha(out/'REGISTERED.json')})
    archive, features, global_rows, measurements = {}, [], [], []
    for group, size in [('train',TRAIN),('held',HELD)]:
        folder = out/group
        folder.mkdir()
        refs = []
        for unit in range(size):
            ask, raw = physical_data(plan,group,unit)
            saved = save_tree(folder/f'{unit:03d}_RAW.npz',raw)
            y = np_save(folder/f'{unit:03d}_QUERY.npz',measured=ask)
            refs.append({'unit':unit,'query':y,'raw':saved})
            if group == 'train':
                measurements.append(ask)
                features.append(history_features(raw['packets']))
                global_rows.append(global_history(raw['packets'],plan['physics']['prefix_clock']))
        archive[group] = refs
    x, unscaled = np.asarray(measurements), np.asarray(global_rows)
    train = np_save(out/'TRAIN_FEATURES.npz',noisy=x,local=np.asarray(features),global_early=unscaled,
                  xscale=np.maximum(x.std(axis=0),plan['physics']['noise']),
                  global_scale=np.maximum(unscaled.reshape(-1,8).std(axis=0),.05))
    write(out/'ACQUIRED.json',{**stamp(),'registration_sha256':sha(out/'REGISTERED.json'),
                             'groups':archive,'train_features':train,'native_calls':3*(TRAIN+HELD)})


def normalized(update):
    update = np.asarray(update,float)
    if update.shape != (PARTICLES,) or not np.isfinite(update).all():
        raise ValueError('All original camera/proposal contributions must be finite.')
    w = np.exp(update-update.max())
    return w/w.sum()


def systematic(w, u):
    c = np.cumsum(w); c[-1] = 1.
    return np.searchsorted(c,np.minimum((u+np.arange(PARTICLES))/PARTICLES,np.nextafter(1.,0.)),side='right')


def camera_likelihood(clean, y, noise):
    return -.5*np.square((clean-y)/np.asarray(noise)).sum(axis=-1)


def _population(job):
    with threadpool_limits(limits=1):
        return population(*job)


def population(out, plan, q):
    cfg, unit, start = plan['physics'], q['unit'], perf_counter()
    y = np_load(verify(out/'held'/f'{unit:03d}_QUERY.npz',q['query']))['measured']
    folder = out/'predictions'/f'{unit:03d}'
    folder.mkdir(exist_ok=False)
    counts = {'attempted':0,'completed':0,'failed':0}
    packets = [_packet_copy(born(plan,'infer',unit,slot),readonly=True) for slot in range(PARTICLES)]
    history = [(p,) for p in packets]
    cached, final, arm_order = [], {arm:{'packets':[],'ledger':[]} for arm in ARMS}, []
    try:
        for camera in (0,1):
            ledger, proposals = [], []
            for slot, past in enumerate(history):
                fresh = _packet_copy(past[-1])
                record = apply_interval(fresh,cfg,camera,table(cfg,'infer',unit,camera,slot),counts)
                ledger.append(record)
                proposals.append(_packet_copy(fresh,readonly=True))
            camera_values = np.stack([observe(p) for p in proposals])
            ll = camera_likelihood(camera_values,y[camera],cfg['noise'])
            w = normalized(ll)
            u = rng('infer',unit,'systematic',camera).random()
            parents = systematic(w,u)
            offered = [past+(state,) for past,state in zip(history,proposals,strict=True)]
            history = [offered[int(parent)] for parent in parents]
            cached.append({'camera':camera,'packets':proposals,'ledger':ledger,'clean':camera_values,'ll':ll,
                           'weights':w,'uniform':u,'parents':parents,
                           'seconds':sum(row['seconds'] for row in ledger)})
        anchor = [packet_fingerprint(p[-1]) for p in history]
        for slot, past in enumerate(history):
            tables = table(cfg,'infer',unit,2,slot)
            order = ARMS if (unit+slot)%2 == 0 else ARMS[::-1]
            arm_order.append(order)
            for arm in order:
                fresh = _packet_copy(past[-1])
                retry = int(rng('infer',unit,'retry',2,slot).integers(0,2**32,dtype=np.uint32)) if arm == 'marked' else None
                record = apply_interval(fresh,cfg,2,tables,counts,query=y[2] if arm=='marked' else None,retry=retry)
                final[arm]['packets'].append(_packet_copy(fresh,readonly=True))
                final[arm]['ledger'].append(record)
    except Exception as error:
        save_tree(folder/'PARTIAL_RAW.npz',{'initial':packets,'shared':cached,'third':final,
                                           'third_arm_order':arm_order,'actual_counts':counts})
        write(folder/'FAILED.json',{**stamp(),'unit':unit,'error':repr(error),'actual_counts':counts})
        raise
    expected = 4*PARTICLES
    if counts != {'attempted':expected,'completed':expected,'failed':0}:
        raise ArithmeticError('The actual native budget did not close.')
    evidence = save_tree(folder/'RAW.npz',{'initials':packets,'shared':cached,'third':final,'shared_cutoff_sha256':anchor,
                                           'third_arm_order':arm_order,'cost':counts})
    c2_ancestor_id = cached[1]['parents']
    arms = {}
    for arm, data in final.items():
        states = [past+(state,) for past,state in zip(history,data['packets'],strict=True)]
        clean = np.stack([observe(p) for p in data['packets']])
        ll = camera_likelihood(clean,y[2],cfg['noise'])
        correction = np.asarray([row['log_ratio'] for row in data['ledger']])
        weights = normalized(ll+correction)
        c2_ancestor_mass = np.bincount(c2_ancestor_id,weights=weights,minlength=PARTICLES)
        arms[arm] = np_save(folder/f'{arm}_WEIGHTS.npz',clean=clean,ll=ll,log_ratio=correction,weights=weights,
                    c2_ancestor_id=c2_ancestor_id,c2_ancestor_mass=c2_ancestor_mass,
                    local=np.stack([history_features(p) for p in states]),
                    global_early=np.stack([global_history(p,cfg['prefix_clock']) for p in states]))
    tilt = [span['mark_tilt'] for interval in final['marked']['ledger'] for span in interval['spans']
            if not span['censored'] and span['mark_tilt'] is not None]
    ledger_budget = {'shared_native_seconds':sum(s['seconds'] for s in cached),
        'third_native_seconds':{arm:sum(row['seconds'] for row in final[arm]['ledger']) for arm in ARMS},
        'tilted_firings':len(tilt),'bounded_trials':sum(len(q['trials']) for q in tilt),
        'native_mark_fallbacks':sum(q['fallback'] for q in tilt)}
    return {'unit':unit,'query_sha256':q['query']['sha256'],'raw':evidence,'arms':arms,'native_calls_actual':expected,
            'native_calls_equivalent_each_arm':3*PARTICLES,'ledger_budget':ledger_budget,'wall_seconds':perf_counter()-start}


def predict(out, plan):
    acq_path = out/'ACQUIRED.json'
    acq = read(acq_path)
    write(out/'PREDICTION_STARTED.json',{**stamp(),'acquisition_sha256':sha(acq_path),'observation_only':True})
    (out/'predictions').mkdir()
    queries = [{'unit':q['unit'],'query':q['query']} for q in acq['groups']['held']]
    if [q['unit'] for q in queries] != list(range(HELD)):
        raise ValueError('The registered held queries are incomplete.')
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        forecasts = list(pool.map(_population,((out,plan,q) for q in queries),chunksize=1))
    bank = np_load(verify(out/'TRAIN_FEATURES.npz',acq['train_features']))
    neighbors = []
    for q in queries:
        y = np_load(verify(out/'held'/f'{q["unit"]:03d}_QUERY.npz',q['query']))['measured']
        dist = np.square((bank['noisy']-y)/bank['xscale']).sum(axis=(1,2))
        neighbors.append(np.argsort(dist,kind='stable')[:8])
    neighbor = np_save(out/'KNN_FROZEN.npz',neighbors=np.stack(neighbors))
    write(out/'PREDICTIONS_FROZEN.json',{**stamp(),'registration_sha256':sha(out/'REGISTERED.json'),
        'acquisition_sha256':sha(acq_path),'held_targets_opened':False,'forecast':forecasts,'knn':neighbor})


def interval(values):
    vals = np.asarray(values,float)
    mean = vals.mean()
    width = t.ppf(.975,len(vals)-1)*vals.std(ddof=1)/np.sqrt(len(vals))
    return {'mean':mean,'pointwise_t95':[mean-width,mean+width],'mothers':len(vals)}


def hidden_scores(local, weights, actual):
    return {'early18':energy(local,weights,actual),
            'hidden_t3_6':energy(local[:,2],weights,actual[2]),
            'hidden_delta32_6':energy(local[:,2]-local[:,1],weights,actual[2]-actual[1])}


def score(out,plan):
    f, a = out/'PREDICTIONS_FROZEN.json', out/'ACQUIRED.json'
    frozen, acq = read(f), read(a)
    if (frozen['registration_sha256'] != sha(out/'REGISTERED.json') or frozen['acquisition_sha256'] != sha(a)
            or frozen['held_targets_opened'] is not False or len(frozen['forecast']) != HELD):
        raise ValueError('All predictions must freeze before revealing target trajectories.')
    write(out/'REVEAL_STARTED.json',{**stamp(),'freeze_sha256':sha(f)})
    bank = np_load(verify(out/'TRAIN_FEATURES.npz',acq['train_features']))
    neighbors = np_load(verify(out/'KNN_FROZEN.npz',frozen['knn']))['neighbors']
    scale = bank['global_scale']
    rows = []
    for unit,(query, forecast) in enumerate(zip(acq['groups']['held'],frozen['forecast'],strict=True)):
        if query['unit'] != unit or forecast['unit'] != unit or forecast['query_sha256'] != query['query']['sha256']:
            raise ValueError('The frozen scored mother binding changed.')
        raw = load_tree(verify(out/'held'/f'{unit:03d}_RAW.npz',query['raw']))
        actual = history_features(raw['packets'])
        g = global_history(raw['packets'],plan['physics']['prefix_clock'])/scale
        values = {}
        for arm in ARMS:
            d = np_load(verify(out/'predictions'/f'{unit:03d}'/f'{arm}_WEIGHTS.npz',forecast['arms'][arm]))
            w = d['weights']
            mass = d['c2_ancestor_mass']
            values[arm] = {**hidden_scores(d['local'],w,actual),'global24':energy(d['global_early']/scale,w,g),
                          'ess':float(1./(w@w)),'c2_ancestor_ess':float(1./(mass@mass)),
                          'median_final_ll':np.median(d['ll'])}
        for arm, ids in [('empirical',np.arange(TRAIN)),('knn8',neighbors[unit])]:
            w = np.full(len(ids),1./len(ids))
            values[arm] = {**hidden_scores(bank['local'][ids],w,actual),'global24':energy(bank['global_early'][ids]/scale,w,g)}
        rows.append(values)
    means = {key:{name:np.mean([r[key][name] for r in rows]) for name in first} for key,first in rows[0].items()}
    paired = {key:{name:interval([r['marked'][name]-r[key][name] for r in rows]) for name in means[key]} for key in COMPARATORS}
    budget = [x['ledger_budget'] for x in frozen['forecast']]
    ledger_totals = {key:sum(row[key] for row in budget) for key in ('shared_native_seconds','tilted_firings','bounded_trials','native_mark_fallbacks')}
    ledger_totals['third_native_seconds'] = {arm:sum(row['third_native_seconds'][arm] for row in budget) for arm in ARMS}
    write(out/'SUMMARY.json',{**stamp(),'registration_sha256':sha(out/'REGISTERED.json'),'freeze_sha256':sha(f),
           'rows':rows,'means':means,'marked_minus':paired,'primary_complete_gate':all(v['early18']['pointwise_t95'][1]<0. for v in paired.values()),
           'equivalent_each_arm_native_calls':HELD*3*PARTICLES,'actual_inference_native_calls':sum(x['native_calls_actual'] for x in frozen['forecast']),
           'data_native_calls':acq['native_calls'],'native_time_reporting':'per-arm third-frame per-interval seconds plus two shared interval totals; excludes I/O and score',
           'ledger_totals':ledger_totals,
           'scored_mothers':HELD,'new_wet_measurements':0,'new_fits':0})


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('phase',choices=('register','acquire','predict','score'))
    cli.add_argument('--out',type=Path,required=True)
    for key in ('v5-reg','fixture','review'):
        cli.add_argument('--'+key,type=Path)
    args = cli.parse_args()
    out = args.out.resolve()
    if args.phase == 'register':
        supplied = (args.v5_reg,args.fixture,args.review)
        if any(p is None for p in supplied):
            cli.error('Register with --v5-reg, --fixture, and --review.')
        register(out,*(p.resolve(strict=True) for p in supplied))
    else:
        with threadpool_limits(limits=1):
            try:
                {'acquire':acquire,'predict':predict,'score':score}[args.phase](out,checked(out))
            except Exception as exc:
                write(out/f'{args.phase.upper()}_FAILED.json',{**stamp(),'error':repr(exc)})
                raise


if __name__ == '__main__':
    main()
