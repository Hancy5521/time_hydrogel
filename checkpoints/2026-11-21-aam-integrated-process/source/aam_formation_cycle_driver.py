"""One prospective complete formation/property/inverse experiment.

No work on import. Every new physical stage and fit has an exclusive receipt.
The failed observation-conditioned phi is neither loaded nor relabelled.
"""
from pathlib import Path
from datetime import datetime, timezone, timedelta
from time import perf_counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse
import ast
import hashlib
import json
import os

ROOT = Path(__file__).resolve().parent
BASE = Path('H:/polymer_completion_public/research')
NAMESPACE = 'aam-mbaa-property-inverse-cycle-v1-20261118'
MASTER = 1918000000
PROTOCOL = ROOT/'AAM_FORMATION_PROPERTY_INVERSE_CYCLE_V1_PROTOCOL_ZH.md'


def now(): return datetime.now(timezone.utc).isoformat()
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p): return json.loads(Path(p).read_text(encoding='utf8'))


def plain(x):
    import numpy as np
    if isinstance(x, np.ndarray): return x.tolist()
    if isinstance(x, np.generic): return x.item()
    if isinstance(x, Path): return str(x)
    if isinstance(x, dict): return {str(k): plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)): return [plain(v) for v in x]
    return x


def write(p, value):
    with Path(p).open('x', encoding='utf8') as f:
        json.dump(plain(value), f, ensure_ascii=False, indent=2, allow_nan=False)


def event(run, action, **details):
    row = {'actual_utc': now(), 'action': action, **details}
    with (run/'EVENTS.jsonl').open('a', encoding='utf8') as f:
        f.write(json.dumps(plain(row), ensure_ascii=False, allow_nan=False)+'\n')


def np_save(p, **a):
    import numpy as np
    with Path(p).open('xb') as f: np.savez_compressed(f, **a)
    return {'path': str(Path(p).resolve()), 'sha256': sha(p), 'bytes': Path(p).stat().st_size}


def np_load(p):
    import numpy as np
    with np.load(p, allow_pickle=False) as f: return {k: f[k] for k in f.files}


def save_tree(path, value):
    from aam_formation_actual_rollout import _pack_tree
    import numpy as np
    arrays = {}
    tree = _pack_tree(value, arrays)
    arrays['metadata_json'] = np.array(json.dumps(tree, ensure_ascii=True))
    return np_save(path, **arrays)


def source_closure():
    seeds = [Path(__file__), ROOT/'aam_formation_cycle_core.py', ROOT/'aam_formation_cycle_data.py',
             ROOT/'aam_formation_cycle_methods.py', ROOT/'aam_actual_network_properties.py',
             ROOT/'aam_formation_process_objective.py']
    found = {}
    while seeds:
        p = seeds.pop()
        if str(p) in found: continue
        found[str(p)] = sha(p)
        tree = ast.parse(p.read_text(encoding='utf-8-sig'))
        for n in ast.walk(tree):
            names = ([a.name.split('.')[0] for a in n.names] if isinstance(n, ast.Import)
                     else [n.module.split('.')[0]] if isinstance(n, ast.ImportFrom) and n.module else [])
            for name in names:
                dep = ROOT/(name+'.py')
                if dep.is_file() and str(dep) not in found: seeds.append(dep)
    return found


def register():
    from aam_formation_cycle_core import make_config
    stamp = datetime.now(timezone.utc)
    run = BASE/stamp.strftime('aam_formation_property_inverse_v1_%Y%m%dT%H%M%S%fZ')
    run.mkdir()
    plan = {'actual_utc': stamp.isoformat(),
            'asia_shanghai': stamp.astimezone(timezone(timedelta(hours=8))).isoformat(),
            'namespace': NAMESPACE, 'master_seed': MASTER, 'config': make_config(),
            'sources': source_closure(), 'protocol': str(PROTOCOL), 'protocol_sha256': sha(PROTOCOL),
            'training_mothers': 64, 'held_mothers': 16, 'training_suffixes': 2, 'held_suffixes': 4,
            'workers': 4, 'BLAS_threads_per_worker': 1, 'SMC_particles': 64,
            'future_B': 32, 'future_atoms_max': 96, 'native_interval_upper_bound': 34416,
            'extra_annotation_control_max': 12, 'mechanical_snapshot_upper_bound': 32208,
            'old_histories_reused': 0, 'new_wet_measurements': 0,
            'seed_roles': {'train': 'master+100000*id', 'held_truth': 'master+10000000+100000*id',
                'assay': 'motherbase+1001', 'truth_future': 'motherbase+2000+200*repeat',
                'inference': 'master+20000000+100000*query',
                'forecast_future': 'master+30000000+100000*query+1000*atom',
                'forest_OOF_then_final': 'master+40000000+fold(0..4)'},
            'read_boundary': 'Predictors receive query X12/measured only, fitted psi and training baseline files. Held actual State/risks/cutoff/future are EVAL.',
            'fit_policy': 'One5parameterpsi;fourOOFplusonefinalfixedforest;no validation model search.',
            'completion': 'Run all functional stages regardless of performance; preserve actual correctness failures.',
            'previous_phi': 'Unique model c7c0c0ee failed; not loaded and not called passed.'}
    write(run/'REGISTERED.json', plan)
    print(str(run), flush=True)
    return run


def checked(run):
    p = read(run/'REGISTERED.json')
    for f, h in p['sources'].items():
        if sha(f) != h: raise ValueError('Registered source changed: '+f)
    if sha(p['protocol']) != p['protocol_sha256']: raise ValueError('Protocol changed.')
    return p


def data_spec(run, plan, role, index, phase, freeze=None):
    item = {'run_dir': str(run), 'config': plan['config'], 'role': role, 'mother_id': index,
            'mother_seed': MASTER+(0 if role == 'train' else 10000000)+100000*index, 'phase': phase}
    if role == 'train' and index == 0: item['validate_native_record'] = True
    if freeze is not None: item['prediction_freeze'] = freeze
    return item


def worker_initializer():
    """Task-local child runtime limits, including libraries loaded after spawn."""
    global _WORKER_LIMITER
    for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
        os.environ[name] = '1'
    import numpy
    import scipy.linalg
    from threadpoolctl import threadpool_limits
    _WORKER_LIMITER = threadpool_limits(limits=1)


def run_data(run, plan, jobs, label):
    from aam_formation_cycle_data import data_job
    records = []
    with ProcessPoolExecutor(max_workers=plan['workers'], initializer=worker_initializer) as pool:
        futures = {pool.submit(data_job, j): j['mother_id'] for j in jobs}
        for f in as_completed(futures):
            result = f.result(); records.append(result)
            event(run, label, mother=futures[f], status=result.get('status'),
                  unique_native=result.get('unique_native_segment_completed'))
            print(label, futures[f], result.get('status'), flush=True)
    return records


def prediction_job(job):
    """Only load_query crosses the held-data boundary in this function."""
    import numpy as np
    from threadpoolctl import threadpool_limits
    from aam_formation_cycle_data import load_query, property_readout, _read6
    from aam_formation_cycle_core import filter_formation, formation_rollout
    from aam_formation_cycle_methods import load_fit, select_actions
    from aam_formation_actual_rollout import save_rollout_npz
    run, q = Path(job['run']), job['query']
    plan = checked(run); cfg = plan['config']
    out = run/'predictions'/f'query_{q:03d}'; out.mkdir(parents=True, exist_ok=True)
    done = out/'COMPLETE.json'
    if done.exists():
        r = read(done)
        for info in r['files'].values():
            if sha(info['path']) != info['sha256']: raise ValueError('Prediction cache changed.')
        return r
    if (out/'STARTED.json').exists(): raise RuntimeError('Unclosed prediction attempt; do not repeat physical work.')
    query = load_query(run/'held'/f'parent_{q:03d}')
    write(out/'STARTED.json', {'actual_utc': now(), 'query_sha256': query['query_sha256'],
          'registration_sha256': sha(run/'REGISTERED.json'), 'pid': os.getpid()})
    start = perf_counter(); files = {}; native = attempted = 0; mechanical = 0
    native_seconds = property_seconds = 0.; future_receipts = []
    with threadpool_limits(limits=1):
        try:
            fitted = load_fit(run/'BASELINES')
            psi = np_load(run/'PSI_FINAL.npz')['factors']
            post = filter_formation(cfg, query['measured'], MASTER+20000000+100000*q, psi,
                                    particles=64, mother_id=f'held:{q}')
            native += post['cost']['native_segment_completed']
            attempted += post['cost']['native_segment_attempted']
            native_seconds += post['cost']['native_seconds']
            files['posterior'] = save_tree(out/'POSTERIOR_RAW.npz', post)
            if not post['valid']: raise RuntimeError('Posterior failed: '+str(post['error']))
            w = np.asarray(post['weights'])
            if w.shape != (64,) or not np.isfinite(w).all() or np.any(w < 0.) or abs(w.sum()-1.) > 1e-12:
                raise ValueError('Invalid native final weights.')
            n = np.where(w > 0., np.maximum(1, np.ceil(32*w)).astype(int), 0)
            if n.sum() > 96: raise ValueError('B32 all-support allocation exceeded bound.')
            hidden = []
            for packet in post['packets']:
                prop = property_readout(packet, cfg); mechanical += 1
                property_seconds += prop['readout_seconds']
                if not prop['valid']: raise RuntimeError('Invalid cutoff mechanics: '+str(prop['error']))
                hidden.append(_read6(packet, prop))
            hidden = np.asarray(hidden)
            rows, atom_weights, prefix_ids = [], [], []
            raw_dir = out/'future_raw'; raw_dir.mkdir()
            atom = 0
            for i, count in enumerate(n):
                for repeat in range(int(count)):
                    values = []
                    future_base = MASTER+30000000+100000*q+1000*atom
                    for ai, action in enumerate(cfg['actions']):
                        result = formation_rollout(post['packets'][i], action, cfg, future_base,
                            properties=lambda packet: property_readout(packet, cfg), mother_id=f'held:{q}')
                        native += result['cost']['native_segment_completed']
                        attempted += result['cost']['native_segment_attempted']
                        native_seconds += result['cost']['native_seconds']
                        property_seconds += result['cost']['property_seconds']
                        mechanical += len(result['snapshots'])
                        raw_receipt = save_rollout_npz(result, raw_dir/f'atom_{atom:03d}_action_{ai}.npz')
                        future_receipts.append({'atom': atom, 'prefix': i, 'action': ai,
                                                'future_base': future_base, **raw_receipt})
                        if not result['valid']: raise RuntimeError('Future failed: '+str(result['error']))
                        trajectory = []
                        for snapshot in result['snapshots']:
                            if not snapshot['properties']['valid']: raise RuntimeError('Invalid future mechanics.')
                            trajectory.append(_read6(snapshot['packet'], snapshot['properties'], reported_q=snapshot['q']))
                        values.append(trajectory)
                    rows.append(np.r_[hidden[i], np.asarray(values).reshape(108)])
                    atom_weights.append(w[i]/count); prefix_ids.append(i); atom += 1
            draws, weights = np.asarray(rows), np.asarray(atom_weights)
            if abs(weights.sum()-1.) > 1e-12: raise ValueError('All-support suffix mass changed.')
            selected = select_actions(draws, weights, fitted['normalization'], fitted['action_support'])
            write(out/'FUTURE_RAW_INDEX.json', {'actual_utc': now(), 'raw_trajectories': future_receipts})
            files['future_raw_index'] = {'path': str((out/'FUTURE_RAW_INDEX.json').resolve()),
                'sha256': sha(out/'FUTURE_RAW_INDEX.json'), 'bytes': (out/'FUTURE_RAW_INDEX.json').stat().st_size}
            files['forecast'] = np_save(out/'FORECAST.npz', targets114=draws, weights=weights,
                prefix_id=np.asarray(prefix_ids), allocated_suffix_count=n, native_weights=w,
                cutoff64=hidden, choices=selected['choices'], expected_loss=selected['expected_loss'])
            write(out/'COMPLETE.json', {'actual_utc': now(), 'status': 'complete', 'query': q,
                'query_sha256': query['query_sha256'], 'files': files, 'choices': selected['choices'],
                'native_interval_attempted': attempted, 'native_interval_completed': native,
                'native_seconds': native_seconds, 'property_seconds': property_seconds,
                'mechanical_snapshots': mechanical,
                'positive_prefixes': int((w > 0.).sum()), 'forecast_atoms': len(draws),
                'terminal_ESS': float(1./(w@w)), 'ancestral_diversity': post['ancestral_diversity'],
                'future_distribution_from_actual_states': True, 'held_future_seen': False,
                'elapsed_seconds': perf_counter()-start})
            return read(done)
        except BaseException as error:
            write(out/'FAILED.json', {'actual_utc': now(), 'error': repr(error), 'files': files,
                'native_interval_attempted': attempted, 'native_interval_completed': native,
                'native_seconds': native_seconds, 'property_seconds': property_seconds,
                'mechanical_snapshots': mechanical, 'completed_future_raw': future_receipts,
                'no_query_replacement_or_retry': True})
            raise


def execute(run):
    import numpy as np
    from scipy.stats import t
    from threadpoolctl import threadpool_limits
    from aam_formation_cycle_core import (fit_psi_lineage, score_psi_marks, deterministic_contract_checks)
    from aam_formation_cycle_data import data_job, load_training_parent, load_query, load_eval
    from aam_formation_cycle_methods import (fit_methods, save_fit, load_fit, predict_methods,
        score_forecast, evaluate_choices, deterministic_cycle_method_checks)
    from aam_actual_network_properties import deterministic_small_graph_checks
    from aam_formation_process_objective import deterministic_objective_checks
    plan = checked(run)
    lock = run/'DRIVER_ACTIVE.lock'
    try:
        descriptor = os.open(lock, os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    except FileExistsError:
        raise RuntimeError('Existing driver lock; inspect actual process, never launch duplicate.')
    os.write(descriptor, str(os.getpid()).encode()); os.close(descriptor)
    started = perf_counter()
    try:
        with threadpool_limits(limits=1):
            if not (run/'INTEGRATED_CHECKS.json').exists():
                write(run/'INTEGRATED_CHECKS.json', {'actual_utc': now(),
                    'core': deterministic_contract_checks(plan['config']),
                    'mechanics': deterministic_small_graph_checks(),
                    'objective': deterministic_objective_checks(), 'methods': deterministic_cycle_method_checks()})
            if not (run/'TRAINING_BANK.npz').exists():
                first = data_job(data_spec(run, plan, 'train', 0, 'full'))
                control = read(first['files']['native_record_control']['path'])
                if not control['passed']: raise RuntimeError('Native MBAA annotation identity failed.')
                run_data(run, plan, [data_spec(run, plan, 'train', i, 'full') for i in range(1, 64)], 'training_mother_complete')
                rows = [load_training_parent(run/'train'/f'parent_{i:03d}') for i in range(64)]
                bank = np_save(run/'TRAINING_BANK.npz', X=np.stack([r['X'] for r in rows]),
                    cutoff=np.stack([r['cutoff_values'] for r in rows]),
                    future=np.stack([r['future_values'] for r in rows]),
                    opportunity=np.concatenate([r['teacher_opportunity'] for r in rows]),
                    selection=np.concatenate([r['teacher_selection'] for r in rows]))
                write(run/'TRAINING_BANK.json', {'actual_utc': now(), **bank, 'mothers': 64})
            if not (run/'FIT_COMPLETED.json').exists():
                if (run/'FIT_STARTED.json').exists(): raise RuntimeError('Interrupted fit; do not reinitialize.')
                write(run/'FIT_STARTED.json', {'actual_utc': now(), 'training_sha256': sha(run/'TRAINING_BANK.npz'),
                    'source_hashes': plan['sources'], 'protocol_sha256': plan['protocol_sha256'],
                    'five_parameter_psi_fits': 1, 'fixed_forest_fits_including_OOF': 5})
                b = np_load(run/'TRAINING_BANK.npz')
                psi = fit_psi_lineage(b['opportunity'], b['selection'], training_mother_ids=list(range(64)))
                np_save(run/'PSI_FINAL.npz', factors=psi['factors'], coefficients=psi['coefficients'])
                write(run/'PSI_FIT.json', psi)
                fitted = fit_methods(b['X'], b['cutoff'], b['future'], mother_ids=np.arange(64),
                    random_state=MASTER+40000000, action_support=np.ones(6, dtype=bool))
                info = save_fit(run/'BASELINES', fitted)
                write(run/'FIT_COMPLETED.json', {'actual_utc': now(), 'psi_sha256': sha(run/'PSI_FINAL.npz'),
                    'baselines_frozen_sha256': sha(run/'BASELINES/FROZEN.json'), 'baseline': info})
            fitted = load_fit(run/'BASELINES')
            if not (run/'ALL_PREDICTIONS_AND_SELECTIONS_FROZEN.json').exists():
                run_data(run, plan, [data_spec(run, plan, 'held', i, 'prefixonly') for i in range(16)], 'held_query_ready')
                x = np.stack([load_query(run/'held'/f'parent_{i:03d}')['X'] for i in range(16)])
                if not (run/'BASELINE_PREDICTIONS.npz').exists():
                    predictions = predict_methods(fitted, x)
                    save_tree(run/'BASELINE_PREDICTIONS_RAW.npz', predictions)
                    arrays = {}
                    for name, value in predictions.items():
                        for key in ('targets114', 'weights', 'choices', 'expected_loss'):
                            arrays[name+'__'+key] = value[key]
                    np_save(run/'BASELINE_PREDICTIONS.npz', **arrays)
                with ProcessPoolExecutor(max_workers=plan['workers'], initializer=worker_initializer) as pool:
                    jobs = {pool.submit(prediction_job, {'run': str(run), 'query': q}): q for q in range(16)}
                    for future in as_completed(jobs):
                        result = future.result()
                        event(run, 'posterior_and_design_frozen', query=jobs[future],
                              atoms=result['forecast_atoms'], native=result['native_interval_completed'])
                        print('prediction_complete', jobs[future], result['forecast_atoms'], flush=True)
                files = {str(run/'BASELINE_PREDICTIONS.npz'): sha(run/'BASELINE_PREDICTIONS.npz')}
                for q in range(16):
                    p = run/'predictions'/f'query_{q:03d}'/'COMPLETE.json'
                    result = read(p); files[str(p)] = sha(p)
                    for item in result['files'].values(): files[item['path']] = item['sha256']
                write(run/'ALL_PREDICTIONS_AND_SELECTIONS_FROZEN.json', {'actual_utc': now(),
                    'prediction_and_selection_frozen': True, 'queries': 16, 'all_forecast_files': files,
                    'fit_completed_sha256': sha(run/'FIT_COMPLETED.json'), 'sources': plan['sources'],
                    'held_futures_revealed': False})
            freeze_path = run/'ALL_PREDICTIONS_AND_SELECTIONS_FROZEN.json'
            freeze = {'path': str(freeze_path.resolve()), 'sha256': sha(freeze_path)}
            run_data(run, plan, [data_spec(run, plan, 'held', i, 'reveal', freeze) for i in range(16)], 'held_reference_complete')
            if (run/'SCORED.json').exists(): return read(run/'SCORED.json')
            baseline = np_load(run/'BASELINE_PREDICTIONS.npz')
            methods = ['bootstrap64_psi', 'knn8', 'extra_trees_128_leaf4', 'best_fixed']
            es = np.empty((16, 4)); hidden = es.copy(); loss = np.empty((16, 4, 2))
            choice = np.empty((16, 4, 2), dtype=np.int64); reference_best = np.empty((16, 2))
            psi_mother_delta = []
            psi_factors = np_load(run/'PSI_FINAL.npz')['factors']
            for q in range(16):
                truth = load_eval(run/'held'/f'parent_{q:03d}', prediction_freeze=freeze)
                scores = score_psi_marks(truth['teacher_opportunity'], truth['teacher_selection'], psi_factors)
                distance = score_psi_marks(truth['teacher_opportunity'], truth['teacher_selection'], np.ones((3, 5)))
                psi_mother_delta.append(float(-scores['selected_cell_log_probability'].sum()+distance['selected_cell_log_probability'].sum()))
                for mi, name in enumerate(methods):
                    if mi == 0:
                        f = np_load(run/'predictions'/f'query_{q:03d}'/'FORECAST.npz')
                        atoms, w, c = f['targets114'], f['weights'], f['choices']
                    else:
                        atoms, w, c = [baseline[name+'__'+key][q] for key in ('targets114','weights','choices')]
                    scored = score_forecast(atoms, w, reference_future=truth['future_values'],
                        reference_cutoff=truth['cutoff_values'], normalization=fitted['normalization'])
                    selected = evaluate_choices(c, truth['future_values'], fitted['normalization'])
                    es[q, mi] = scored['primary_future_energy']; hidden[q, mi] = scored['hidden6_energy']
                    loss[q, mi] = selected['selected_goal_loss']; choice[q, mi] = c
                    reference_best[q] = selected['finite_reference_best_goal_loss']
            evidence = np_save(run/'SCORED_CASES.npz', mother=np.arange(16), methods=np.asarray(methods),
                future_energy=es, cutoff_energy=hidden, selected_loss=loss, choices=choice,
                finite_reference_best_loss=reference_best, psi_minus_distance_path_NLL=np.asarray(psi_mother_delta))
            def ci(a):
                a = np.asarray(a); mean = a.mean(axis=0)
                radius = t.ppf(.975, 15)*a.std(axis=0, ddof=1)/4.
                return {'n_mothers': 16, 'mean': mean, 't95': np.stack((mean-radius, mean+radius), axis=-1)}
            report = {'actual_utc': now(), 'functional_cycle_complete': True,
                'scope': 'New64train16held complete functional/development version, not wet or highconfidence superiority.',
                'methods': methods, 'mean_future_energy': es.mean(axis=0), 'mean_cutoff_energy': hidden.mean(axis=0),
                'mean_selected_goal_loss': loss.mean(axis=0), 'mean_finite_reference_best_loss': reference_best.mean(axis=0),
                'psi_minus_distance_path_NLL': ci(psi_mother_delta),
                'bootstrap_minus_baselines': {methods[m]: {'future_energy': ci(es[:, 0]-es[:, m]),
                    'goal_loss': ci(loss[:, 0]-loss[:, m])} for m in (1, 2, 3)},
                'choice_counts': {name: [np.bincount(choice[:, m, g], minlength=6) for g in range(2)]
                                  for m, name in enumerate(methods)},
                'score_arrays': evidence, 'prediction_freeze': freeze,
                'old_phi_failure_preserved': True, 'fit_fallback': read(run/'PSI_FIT.json')['fallback'],
                'elapsed_seconds_this_execution': perf_counter()-started,
                'scientific_goal_achieved': False,
                'next': 'Independent numerical review and full resources to Work; improve bottleneck indicated by entire system.'}
            write(run/'SCORED.json', report)
            print(json.dumps(plain(report), ensure_ascii=True, indent=2), flush=True)
            return report
    except BaseException as error:
        event(run, 'driver_interrupted', error=repr(error), no_automatic_retry=True)
        raise
    finally:
        lock.unlink()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--register', action='store_true'); parser.add_argument('--execute', action='store_true')
    parser.add_argument('--run', type=Path); args = parser.parse_args()
    if args.register: register()
    elif args.execute: execute(args.run)
    else: print('Explicit register then execute one complete prospective functional cycle.')
