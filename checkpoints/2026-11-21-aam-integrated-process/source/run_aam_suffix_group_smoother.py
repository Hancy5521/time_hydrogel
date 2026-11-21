"""Joint blind parent/suffix smoothing with the registered AAm role-addressed law."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
from pathlib import Path
from shutil import copyfile
from time import perf_counter

import numpy as np
from threadpoolctl import threadpool_limits

from aam_camera_information import history_features
from aam_camera_mark_tilt import SCHEMA as NATIVE_LAW, make_interval_table
from aam_formation_actual_rollout import _packet_copy, packet_fingerprint
from aam_formation_cycle_driver import np_load, np_save, read, save_tree, sha, write
from aam_formation_local_initial import build_packet, draw_initial
from aam_formation_path_cycle import event, load_tree, stamp
from aam_grouped_third_frame import (
    EQUAL_SUFFIX_NATIVE_BUDGET, PARTICLES, SWEEPS, THIRD_BETAS, attach,
)
from aam_snapshot_process import observe
from package_aam_formation_local_cycle import source_closure
from run_aam_mark_candidate_filter import (
    apply_interval, camera_likelihood, global_history, hidden_scores, interval,
    normalized as early_normalized, systematic as early_systematic, verify,
)


ROOT = Path(__file__).resolve().parent
PROTOCOL = ROOT / 'AAM_SUFFIX_GROUP_SMOOTHER_PROTOCOL_ZH.md'
NAMESPACE = 'aam-shared-early-bank-grouped-native-suffix-smoother-v1'
ENTROPY = [int.from_bytes(hashlib.sha256(NAMESPACE.encode()).digest()[i:i+4], 'little')
           for i in range(0, 16, 4)]
ROLES = {'data_initial': 1, 'data_tape': 2, 'measurement': 3, 'initial': 4,
         'early_tape': 5, 'early_resample': 6, 'suffix_tape': 7, 'parent': 8,
         'temper_resample': 9, 'mh': 10, 'one_tail_readout': 11}
TRAIN, HELD, EARLY = 64, 48, 64
LIVE, SUFFIX_COST = PARTICLES, EQUAL_SUFFIX_NATIVE_BUDGET
BETAS = THIRD_BETAS[1:]
ARMS = {'single': (1, SWEEPS[1], 0), 'grouped': (4, SWEEPS[4], 1)}
METHODS = ('grouped', 'single', 'sir256', 'knn8', 'empirical')
STUDY_METRICS = ('hidden_t3_6', 'hidden_delta32_6', 'early18',
                 'hidden_t2_6', 'global24', 'new_edge_brier')


def rng(role, unit, *address):
    return np.random.default_rng(np.random.SeedSequence([
        *ENTROPY, ROLES[role], int(unit), *[int(k) for k in address]]))


def tape(cfg, role, unit, camera, *address):
    return make_interval_table(cfg, camera, *(rng(role, unit, camera, *address, j)
                                               for j in range(3)))


def register(out, native_fixture, prior, work, workers):
    eng, verified = read(native_fixture / 'REGISTERED.json'), read(native_fixture / 'RESULT.json')
    past, past_audit = read(prior / 'SCORED.json'), read(prior / 'INDEPENDENT_AUDIT.json')
    cfg = eng['physics']
    if (not verified['passed'] or verified['registration_sha256'] != sha(native_fixture / 'REGISTERED.json')
            or not past_audit['passed'] or past_audit['scored_sha256'] != sha(prior / 'SCORED.json')
            or past['mothers'] != 32 or cfg != read(prior / 'REGISTERED.json')['physics']
            or tuple(cfg['prefix_clock']) != (0., .65, 1.35, 2.05)
            or tuple(cfg['noise']) != (.08, .10, .08, .10)):
        raise ValueError('Bind the verified original camera law and audited known-C2 reference.')
    closure = source_closure([Path(__file__), ROOT / 'audit_aam_suffix_group_smoother.py'])
    sources = {str(path): sha(path) for path in closure.values()}
    if any(sources[path] != digest for path, digest in eng['sources'].items() if path in sources):
        raise ValueError('A source from the verified native engine changed.')
    inputs = [native_fixture / 'REGISTERED.json', native_fixture / 'RESULT.json',
              prior / 'REGISTERED.json', prior / 'SCORED.json', prior / 'INDEPENDENT_AUDIT.json',
              work, PROTOCOL]
    out.mkdir(exist_ok=False)
    reg = out / 'REGISTERED.json'
    write(reg, {**stamp(), 'namespace': NAMESPACE, 'entropy': ENTROPY, 'roles': ROLES,
                'physics': cfg, 'native_law': NATIVE_LAW, 'training_mothers': TRAIN,
                'held_mothers': HELD, 'early_particles': EARLY, 'stage_particles': LIVE,
                'arms': ARMS, 'betas': BETAS, 'workers': workers, 'methods': METHODS,
                'primary': 'hidden_t3_6', 'secondary': STUDY_METRICS[1:],
                'primary_gain_rule': 'grouped better than all listed comparators; all paired t95 lower > 0',
                'design': 'unchanged original physical law, three camera noise, no action evaluations in this stage',
                'budget': {'shared_early_eaches': 2*EARLY, 'third_each': SUFFIX_COST,
                           'inference_actual': HELD*(2*EARLY+2*SUFFIX_COST),
                           'simulation_data': 3*(TRAIN+HELD)},
                'sir': 'reuses all independent K1 proposals prior to selection; charges same third budget',
                'native_arms_address_sharing': 'same two-frame bank; disjoint late suffix banks',
                'new_fits': 0, 'new_wet_measurements': 0, 'old_held_targets_read': 0,
                'inputs': {str(p): sha(p) for p in inputs}, 'sources': sources})
    snapshot = out / 'SOURCE_SNAPSHOT'
    snapshot.mkdir()
    for name, path in closure.items():
        copyfile(path, snapshot / name)
    write(out / 'RUN.json', {**stamp(), 'registration_sha256': sha(reg), 'registered_before_native': True})
    event(out, 'registered_blind_native_suffix_group_comparison', new_held_mothers=HELD)


def checked(out):
    reg = out / 'REGISTERED.json'
    plan = read(reg)
    if (plan['namespace'] != NAMESPACE or plan['entropy'] != ENTROPY
            or sha(reg) != read(out / 'RUN.json')['registration_sha256']):
        raise ValueError('Require the prospective registered study.')
    for path, digest in {**plan['sources'], **plan['inputs']}.items():
        if sha(path) != digest:
            raise ValueError(f'Registered dependency changed: {path}')
    return plan


def native_step(parent, cfg, camera, table, count):
    packet = _packet_copy(parent)
    receipt = apply_interval(packet, cfg, camera, table, count)
    if receipt['log_ratio'] != 0. or receipt['tape']['forced_mark_consumed']:
        raise ValueError('Inference may use only the original unforced simulator.')
    return _packet_copy(packet, readonly=True), receipt


def counts(number=0):
    return {'attempted': number, 'completed': number, 'failed': 0}


def distinct_edges(previous, later):
    def edges(packet):
        state = packet[0]
        n = len(state.points)
        ij = np.sort(state.edges[:int(state.counts[1])], axis=1)
        return set((ij[:, 0]*n+ij[:, 1]).tolist())
    return sorted(edges(later)-edges(previous))


def features(past, tail, cfg):
    history = (*past, tail)
    return {'local': history_features(history),
            'global_early': global_history(history, cfg['prefix_clock']),
            'new_edges': distinct_edges(past[-1], tail)}


def acquired_data(cfg, unit):
    initial = draw_initial(cfg, rng('data_initial', unit))
    past = [build_packet(cfg, initial, factors=np.asarray(cfg['factors']))]
    count, receipts = counts(), []
    for c in range(3):
        q, evidence = native_step(past[-1], cfg, c, tape(cfg, 'data_tape', unit, c), count)
        past.append(q)
        receipts.append(evidence)
    clean = np.stack([observe(q) for q in past[1:]])
    measured = clean+np.asarray(cfg['noise'])*rng('measurement', unit).standard_normal((3, 4))
    return measured, {'initial': initial, 'packets': past, 'clean': clean, 'intervals': receipts, 'calls': count}


def acquire(out, plan):
    write(out / 'ACQUISITION_STARTED.json', {**stamp(), 'registration_sha256': sha(out / 'REGISTERED.json')})
    groups, train = {}, []
    for name, total, offset in [('training', TRAIN, 0), ('held', HELD, TRAIN)]:
        folder = out / name
        folder.mkdir()
        rows = []
        for unit in range(offset, total+offset):
            y, raw = acquired_data(plan['physics'], unit)
            rows.append({'unit': unit, 'raw': save_tree(folder / f'{unit:03d}_RAW.npz', raw),
                         'query': np_save(folder / f'{unit:03d}_QUERY.npz', measured=y)})
            if name == 'training':
                train.append((y, features(raw['packets'][:-1], raw['packets'][-1], plan['physics'])))
        groups[name] = rows
    y = np.stack([item[0] for item in train])
    g = np.stack([item[1]['global_early'] for item in train])
    bank = save_tree(out / 'TRAIN_BANK.npz', {'noisy': y,
        'xscale': np.maximum(y.std(axis=0), plan['physics']['noise']),
        'global_scale': np.maximum(g.reshape(-1, 8).std(axis=0), .05),
        'features': [item[1] for item in train]})
    write(out / 'ACQUIRED.json', {**stamp(), 'registration_sha256': sha(out / 'REGISTERED.json'),
                                  'groups': groups, 'train_bank': bank, 'native_calls': 3*(TRAIN+HELD)})


def early_bank(cfg, unit, measured, count):
    initial = [draw_initial(cfg, rng('initial', unit, k)) for k in range(EARLY)]
    states = [build_packet(cfg, x, factors=np.asarray(cfg['factors'])) for x in initial]
    early = []
    histories = [(state,) for state in states]
    for c in range(2):
        proposed, receipts = zip(*(native_step(h[-1], cfg, c, tape(cfg, 'early_tape', unit, c, k), count)
                                   for k, h in enumerate(histories)), strict=True)
        clean = np.stack([observe(q) for q in proposed])
        w = early_normalized(camera_likelihood(clean, measured[c], cfg['noise']))
        histories = [h+(p,) for h, p in zip(histories, proposed, strict=True)]
        if c == 0:
            u = rng('early_resample', unit).random()
            parents = early_systematic(w, u)
            histories = [histories[int(k)] for k in parents]
        else:
            u, parents = None, None
        early.append({'packets': proposed, 'clean': clean, 'weights': w, 'uniform': u,
                      'parents': parents, 'receipts': receipts})
    return histories, w, {'initial': initial, 'packets0': states, 'frames': early}


def sample_index(weights, u):
    cumulative = np.cumsum(weights)
    cumulative[-1] = 1.
    return int(np.searchsorted(cumulative, min(u, np.nextafter(1., 0.)), side='right'))


def normalized(log):
    w = np.exp(log-log.max())
    return w/w.sum()


def arm_solver(cfg, unit, measured, histories, w, arm, count, trials):
    k, sweeps, code = ARMS[arm]
    started = perf_counter()
    native_seconds = 0.

    def propose():
        nonlocal native_seconds
        group_id = len(trials)
        u = rng('parent', unit, code, group_id).random()
        parent = sample_index(w, u)
        tails, ledgers = [], []
        for j in range(k):
            packet, evidence = native_step(histories[parent][-1], cfg, 2,
                                           tape(cfg, 'suffix_tape', unit, 2, code, group_id, j), count)
            tails.append(packet)
            ledgers.append(evidence)
            native_seconds += evidence['seconds']
        clean = np.stack([observe(t) for t in tails])
        log = camera_likelihood(clean, measured, cfg['noise'])
        group = attach(parent, tails, log)
        trials.append({'group': group_id, 'parent': parent, 'parent_uniform': u, 'tails': tails,
                       'clean': clean, 'log_like': log, 'log_mean': group.log_potential,
                       'receipts': ledgers})
        return group_id

    active = np.asarray([propose() for _ in range(LIVE)])
    stages = []
    previous = 0.
    for stage, beta in enumerate(BETAS):
        weights = normalized(np.asarray([trials[g]['log_mean'] for g in active])*(beta-previous))
        u = rng('temper_resample', unit, code, stage).random()
        parents = [sample_index(weights, (u+i)/LIVE) for i in range(LIVE)]
        initial = active.copy()
        active = active[parents]
        moves = []
        for sweep in range(sweeps):
            for slot in range(LIVE):
                now, proposal = int(active[slot]), propose()
                log_ratio = beta*(trials[proposal]['log_mean']-trials[now]['log_mean'])
                log_u = -rng('mh', unit, code, stage, sweep, slot).exponential()
                accepted = log_u < min(0., log_ratio)
                active[slot] = proposal if accepted else now
                moves.append((sweep, slot, now, proposal, log_u, accepted))
        stages.append({'beta': beta, 'initial': initial, 'weights': weights, 'uniform': u, 'parents': parents,
                       'moves': moves, 'terminal': active.copy()})
        previous = beta
    if len(trials)*k != SUFFIX_COST:
        raise ArithmeticError('Arm violated the frozen third-camera native budget.')
    return {'trials': trials, 'stages': stages, 'final_groups': active,
            'native_seconds': native_seconds, 'operator_seconds': perf_counter()-started}


def forecast_atoms(trials, group_ids, all_sir, histories, cfg, unit, arm):
    unique = {}
    chosen = []
    for slot, g in enumerate(group_ids):
        log = trials[g]['log_like']
        w = normalized(log)
        if not all_sir:
            chosen.append((int(g), sample_index(w, rng('one_tail_readout', unit, ARMS[arm][2], slot).random())))
        for j, prob in enumerate(w):
            key = (int(g), j)
            unique[key] = unique.get(key, 0.)+prob/len(group_ids)
    ids = sorted(unique)
    weights = (normalized(np.array([trials[g]['log_like'][j] for g, j in ids])) if all_sir
               else np.asarray([unique[key] for key in ids]))
    new = [features(histories[trials[g]['parent']], trials[g]['tails'][j], cfg) for g, j in ids]
    parent = [trials[g]['parent'] for g, _ in ids]
    return {'weights': weights, 'refs': ids, 'parent': parent, 'features': new, 'one_tail_refs': chosen}


def predict_one(out, cfg, query):
    with threadpool_limits(limits=1):
        unit = query['unit']
        folder = out / 'predictions' / f'{unit:03d}'
        folder.mkdir()
        y = np_load(verify(out / 'held' / f'{unit:03d}_QUERY.npz', query['query']))['measured']
        count = counts()
        histories, w, early = early_bank(cfg, unit, y, count)
        index = packet_fingerprints(histories)
        raw, predictions, unfinished, diagnostic = {}, {}, {}, {}
        order = ('single', 'grouped') if unit % 2 == 0 else ('grouped', 'single')
        try:
            for arm in order:
                unfinished[arm] = []
                data = arm_solver(cfg, unit, y[2], histories, w, arm, count, unfinished[arm])
                raw[arm] = save_tree(folder / f'{arm}_RAW.npz', data)
                predictions[arm] = forecast_atoms(data['trials'], data['final_groups'], False, histories, cfg, unit, arm)
                diagnostic[arm] = {'native_seconds': data['native_seconds'],
                    'operator_seconds': data['operator_seconds'],
                    'mh_accepted': sum(move[-1] for stage in data['stages'] for move in stage['moves']),
                    'mh_attempts': sum(len(stage['moves']) for stage in data['stages'])}
                if arm == 'single':
                    predictions['sir256'] = forecast_atoms(data['trials'], range(SUFFIX_COST), True,
                                                           histories, cfg, unit, arm)
                del unfinished[arm], data
        except Exception as error:
            save_tree(folder / 'PARTIAL_RAW.npz', {'early': early, 'unfinished': unfinished, 'counts': count})
            write(folder / 'FAILED.json', {**stamp(), 'actual_counts': count, 'error': repr(error)})
            raise
        if count != counts(2*EARLY+2*SUFFIX_COST):
            raise ArithmeticError('Native calls did not close.')
        for arm, arm_data in predictions.items():
            mass = np.bincount(arm_data['parent'], weights=arm_data['weights'], minlength=EARLY)
            diagnostic.setdefault(arm, {})['ancestor_ess'] = float(1./(mass@mass))
        return {'unit': unit, 'query_sha256': query['query']['sha256'], 'order': order, 'count': count,
                'diagnostics': diagnostic,
                'early': save_tree(folder / 'EARLY.npz', {'bank': early, 'history_fingerprints': index}),
                'raw': raw, 'forecast': save_tree(folder / 'PREDICTIONS.npz', predictions)}


def packet_fingerprints(histories):
    return [[packet_fingerprint(p) for p in h] for h in histories]


def predict(out, plan):
    acq = out / 'ACQUIRED.json'
    doc = read(acq)
    write(out / 'PREDICTION_STARTED.json', {**stamp(), 'acquisition_sha256': sha(acq), 'query_only': True})
    (out / 'predictions').mkdir()
    queries = [{'unit': item['unit'], 'query': item['query']} for item in doc['groups']['held']]
    with ProcessPoolExecutor(max_workers=plan['workers']) as pool:
        rows = list(pool.map(predict_one, [out]*HELD, [plan['physics']]*HELD, queries, chunksize=1))
    bank = load_tree(verify(out / 'TRAIN_BANK.npz', doc['train_bank']))
    ids = []
    for query in queries:
        y = np_load(verify(out / 'held' / f'{query["unit"]:03d}_QUERY.npz', query['query']))['measured']
        d = np.square((bank['noisy']-y)/bank['xscale']).sum(axis=(1, 2))
        ids.append(np.argsort(d, kind='stable')[:8])
    knn = np_save(out / 'KNN_FROZEN.npz', index=np.stack(ids))
    write(out / 'PREDICTIONS_FROZEN.json', {**stamp(), 'registration_sha256': sha(out / 'REGISTERED.json'),
          'acquisition_sha256': sha(acq), 'predictions': rows, 'knn': knn, 'opened_held_targets': False})


def scores(estimates, weights, actual, scale):
    from aam_camera_information import energy
    local = np.stack([x['local'] for x in estimates])
    global_early = np.stack([x['global_early'] for x in estimates])
    beliefs = {}
    for row, amount in zip(estimates, weights, strict=True):
        for eid in row['new_edges']:
            beliefs[eid] = beliefs.get(eid, 0.)+amount
    observed_edges = set(actual['new_edges'])
    brier = sum((beliefs.get(eid, 0.)-(eid in observed_edges))**2
                for eid in observed_edges | beliefs.keys())
    return {**hidden_scores(local, weights, actual['local']),
            'hidden_t2_6': energy(local[:, 1], weights, actual['local'][1]),
            'global24': energy(global_early/scale, weights, actual['global_early']/scale),
            'new_edge_brier': brier}


def score(out, plan):
    acq, freeze = out / 'ACQUIRED.json', out / 'PREDICTIONS_FROZEN.json'
    data, f = read(acq), read(freeze)
    if (f['opened_held_targets'] is not False or f['acquisition_sha256'] != sha(acq)
            or f['registration_sha256'] != sha(out / 'REGISTERED.json') or len(f['predictions']) != HELD):
        raise ValueError('Freeze every blind prediction before revealing raw targets.')
    write(out / 'REVEAL_STARTED.json', {**stamp(), 'freeze_sha256': sha(freeze)})
    bank = load_tree(verify(out / 'TRAIN_BANK.npz', data['train_bank']))
    ids = np_load(verify(out / 'KNN_FROZEN.npz', f['knn']))['index']
    result = []
    for i, (record, predict) in enumerate(zip(data['groups']['held'], f['predictions'], strict=True)):
        unit = i+TRAIN
        if not (unit == record['unit'] == predict['unit'] and record['query']['sha256'] == predict['query_sha256']):
            raise ValueError('Held unit identity changed.')
        p = f'{unit:03d}'
        raw = load_tree(verify(out / 'held' / f'{p}_RAW.npz', record['raw']))
        actual = features(raw['packets'][:-1], raw['packets'][-1], plan['physics'])
        atoms = load_tree(verify(out / 'predictions' / p / 'PREDICTIONS.npz', predict['forecast']))
        for key, indexes in [('knn8', ids[i]), ('empirical', np.arange(TRAIN))]:
            atoms[key] = {'weights': np.full(len(indexes), 1/len(indexes)),
                          'features': [bank['features'][j] for j in indexes]}
        result.append({key: scores(arm['features'], arm['weights'], actual, bank['global_scale'])
                       for key, arm in atoms.items()})
    means = {a: {m: np.mean([row[a][m] for row in result]) for m in STUDY_METRICS} for a in METHODS}
    paired = {a: {m: interval([row[a][m]-row['grouped'][m] for row in result])
                  for m in STUDY_METRICS} for a in METHODS[1:]}
    operator = {a: {key: sum(row['diagnostics'][a][key] for row in f['predictions'])
                    for key in f['predictions'][0]['diagnostics'][a]}
                for a in ('single', 'grouped', 'sir256')}
    for d in operator.values():
        d['mean_ancestor_ess'] = d.pop('ancestor_ess')/HELD
    write(out / 'SCORED.json', {**stamp(), 'freeze_sha256': sha(freeze), 'registration_sha256': sha(out / 'REGISTERED.json'),
          'means': means, 'rows': result, 'grouped_gain_over': paired, 'held_mothers': HELD,
          'primary_gate': all(x['hidden_t3_6']['pointwise_t95'][0] > 0. for x in paired.values()),
          'operator_diagnostics': operator,
          'actual_inference_native_calls': sum(x['count']['completed'] for x in f['predictions']),
          'native_acquisition': data['native_calls'], 'new_fits': 0, 'new_wet_measurements': 0})


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('phase', choices=('register', 'acquire', 'predict', 'score'))
    cli.add_argument('--out', required=True, type=Path)
    cli.add_argument('--workers', type=int, choices=range(1, 9), default=4)
    for option in ('native-fixture', 'prior', 'work'):
        cli.add_argument('--'+option, type=Path)
    opts = cli.parse_args()
    out = opts.out.resolve()
    if opts.phase == 'register':
        sources = (opts.native_fixture, opts.prior, opts.work)
        if any(p is None for p in sources):
            cli.error('Registration requires --native-fixture, --prior and --work.')
        register(out, *(p.resolve(strict=True) for p in sources), opts.workers)
    else:
        {'acquire': acquire, 'predict': predict, 'score': score}[opts.phase](out, checked(out))


if __name__ == '__main__':
    main()
