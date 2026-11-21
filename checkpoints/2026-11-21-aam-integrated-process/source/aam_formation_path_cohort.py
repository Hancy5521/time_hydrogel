"""New held cohorts on unchanged formation physics; explicit query/reveal stages.

Import does not generate an initial state, propagate, fit or score.  The caller's
prospective registration owns seeds, sizes and source hashes. V1 stays frozen.
"""
from pathlib import Path
from time import perf_counter
import numpy as np

from aam_formation_cycle_driver import read, write, sha, np_save, np_load, now
from aam_formation_cycle_data import (
    _stage, _tree_load, property_readout, _read6, check_actual_state, _branch_check,
)
from aam_formation_streamed_rollout import formation_prefix_streamed, formation_rollout_streamed
from aam_formation_path_streams import seed_draws, rng, future_streams
from aam_formation_actual_rollout import packet_fingerprint


def context(run, query):
    run = Path(run)
    plan = read(run/'REGISTERED.json')
    if plan.get('namespace') != 'aam-formation-whole-path-tempering-v4-20261119':
        raise ValueError('Use the separately registered V4 cohort, never rewrite V1.')
    if type(query) is not int or not 0 <= query < plan['queries']:
        raise ValueError('Query identifier outside the registered whole cohort.')
    for path, expected in plan['sources'].items():
        if sha(path) != expected:
            raise ValueError('Frozen source changed: '+path)
    folder = run/'held'/f'parent_{query:03d}'
    folder.mkdir(parents=True, exist_ok=True)
    base = int(seed_draws(query, 'truth_initial'))
    binding = {'schema': 'new-formation-held-v4', 'namespace': plan['namespace'],
               'root_registration': str((run/'REGISTERED.json').resolve()),
               'root_registration_sha256': sha(run/'REGISTERED.json'),
               'role': 'held', 'mother_id': query, 'mother_seed': base}
    return plan, folder, binding, base


def prefix_job(job):
    run, query = Path(job['run']), int(job['query'])
    plan, folder, binding, seed = context(run, query)
    path = folder/'QUERY_COMPLETE.json'
    if path.exists():
        result = read(path)
        for info in result['files'].values():
            if sha(info['path']) != info['sha256']:
                raise ValueError('Completed held prefix/query changed.')
        return result
    started = perf_counter()
    dispatched = []
    cfg = plan['config']
    prefix, raw = _stage(folder, 'PREFIX_RAW', binding,
        lambda: formation_prefix_streamed(cfg, seed, np.asarray(cfg['factors']),
            native_seeds=seed_draws(query, 'truth_native', size=3),
            record_teacher=False, mother_id=f'held:{query}'), dispatched)
    if not prefix['valid']:
        raise RuntimeError('New held prefix failed; actual output retained, no replacement.')
    check_actual_state(prefix['packet'][0], expected_age=2.05)
    clean = np.asarray(prefix['clean'])
    if clean.shape != (3, 4):
        raise ValueError('Three original cameras required.')
    # The assay is saved before any predictor or reference future is called.
    query_path = folder/'QUERY.npz'
    if query_path.exists():
        info = read(folder/'QUERY.json')
        if sha(query_path) != info['sha256']:
            raise ValueError('Saved query changed; no new measurement draw.')
    else:
        measured = clean+rng(query, 'truth_assay').normal(
            0., np.asarray(cfg['noise']), size=clean.shape)
        info = np_save(query_path, X=measured.reshape(12), measured=measured,
                       mother_id=np.array(query), role=np.array('held'))
        write(folder/'QUERY.json', {'actual_utc': now(), **info, 'binding': binding})
    result = {'actual_utc': now(), 'status': 'complete', 'role': 'held/prefixonly',
              'binding': binding, 'files': {'prefix_raw': raw, 'query': info},
              'native_cost': prefix['cost'], 'dispatched_cost': dispatched,
              'reference_future_generated': False, 'elapsed_seconds': perf_counter()-started}
    write(path, result)
    return result


def freeze_receipt(run, *, verify_all=False):
    path = Path(run)/'ALL_PREDICTIONS_AND_SELECTIONS_FROZEN.json'
    frozen = read(path)
    if frozen.get('prediction_and_selection_frozen') is not True:
        raise ValueError('All native/baseline predictions and choices must be frozen first.')
    # The dispatcher can verify all forecast payloads once. Workers bind the
    # immutable all-model receipt; they do not reread every other query's large
    # raw state archives for every reference branch.
    if verify_all:
        for file, expected in frozen['all_forecast_files'].items():
            if sha(file) != expected:
                raise ValueError('Prediction/choice freeze has changed: '+file)
    return {'path': str(path.resolve()), 'sha256': sha(path)}


def reveal_job(job):
    run, query = Path(job['run']), int(job['query'])
    plan, folder, binding, seed = context(run, query)
    freeze = freeze_receipt(run)
    completed = folder/'EVAL_COMPLETE.json'
    if completed.exists():
        result = read(completed)
        if result['prediction_freeze'] != freeze:
            raise ValueError('Reference belongs to a different prediction freeze.')
        for info in result['files'].values():
            if sha(info['path']) != info['sha256']:
                raise ValueError('Completed reference changed.')
        return result
    before = read(folder/'QUERY_COMPLETE.json')
    raw_info = before['files']['prefix_raw']
    prefix = _tree_load(raw_info['path'], binding, raw_info['sha256'])
    packet = prefix['packet']
    origin_hash = packet_fingerprint(packet)
    cfg = plan['config']
    # Actual cutoff mechanics is EVAL, never a predictor input.
    prop = property_readout(packet, cfg)
    if not prop['valid']:
        raise ValueError('Invalid actual cutoff mechanics; not zero.')
    cutoff = _read6(packet, prop)
    from aam_formation_micro_history import extract_frames, extract_future
    micro_early, micro_early_winding = extract_frames(prefix['frames'])
    micro_futures, micro_windings = [], []
    values, refs, costs, dispatched = [], {}, [], []
    for repeat in range(plan['reference_suffixes']):
        row = []
        micro_row, winding_row = [], []
        for action_index, action in enumerate(cfg['actions']):
            streams = future_streams(query, suffix=repeat, truth=True)
            branch_seed = streams['seed']
            name = f'FUTURE_r{repeat:02d}_a{action_index}'
            result, info = _stage(folder, name, binding,
                lambda a=action, st=streams: formation_rollout_streamed(packet, a, cfg, **st,
                    properties=lambda p: property_readout(p, cfg), mother_id=f'held:{query}'), dispatched)
            _branch_check(result, packet, action, cfg, branch_seed)
            trajectory = []
            for snapshot in result['snapshots']:
                if not snapshot['properties']['valid']:
                    raise ValueError('Invalid reference future mechanics; raw retained.')
                trajectory.append(_read6(snapshot['packet'], snapshot['properties'],
                                          reported_q=snapshot['q']))
            row.append(trajectory)
            micro, winding = extract_future(result)
            micro_row.append(micro)
            winding_row.append(winding)
            refs[name] = info
            costs.append(result['cost'])
        values.append(row)
        micro_futures.append(micro_row)
        micro_windings.append(winding_row)
    if packet_fingerprint(packet) != origin_hash:
        raise ValueError('True actual past was modified by an intervention.')
    target = folder/'EVAL_VALUES.npz'
    if target.exists():
        raise RuntimeError('Unclosed EVAL value publication; inspect, do not replace.')
    info = np_save(target, cutoff_values=cutoff, future_values=np.asarray(values),
                   micro_early=micro_early, micro_future=np.asarray(micro_futures),
                   micro_early_winding=micro_early_winding,
                   micro_future_winding=np.asarray(micro_windings),
                   source_actual_cutoff_fingerprint=np.array(origin_hash))
    result = {'actual_utc': now(), 'status': 'complete', 'binding': binding,
              'files': {'values': info, **refs}, 'prediction_freeze': freeze,
              'cutoff_property_seconds': prop['readout_seconds'],
              'reference_cost': costs, 'dispatched_cost': dispatched,
              'new_independent_mothers': 1, 'physical_measurements': 0}
    write(completed, result)
    return result


def load_eval(run, query):
    """Only for after-freeze scoring, not imported into prediction workers."""
    run = Path(run)
    receipt = read(run/'held'/f'parent_{query:03d}'/'EVAL_COMPLETE.json')
    path = Path(receipt['prediction_freeze']['path'])
    if sha(path) != receipt['prediction_freeze']['sha256']:
        raise ValueError('Reference freeze identity changed.')
    info = receipt['files']['values']
    if sha(info['path']) != info['sha256']:
        raise ValueError('EVAL values hash changed.')
    return np_load(info['path'])
