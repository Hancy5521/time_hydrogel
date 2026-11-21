"""Prospective, equal-native-budget comparison of wide phase SMC and phase PGAS."""

from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from shutil import copyfile
from time import perf_counter

import numpy as np
from scipy.spatial.distance import cdist
from threadpoolctl import threadpool_limits

from aam_common_phase_hidden import history_orbit
from aam_formation_actual_rollout import PACKET_FIELDS, STATE_FIELDS, packet_fingerprint
from aam_formation_cycle_driver import np_load, np_save, read, save_tree, sha, write
from aam_formation_path_cycle import event, load_tree, stamp
from aam_original_tape_pgas import make_history
from aam_phase_marginal_tape_pgas import phase_reference, sweep
from aam_phase_wide_smc import forward
from package_aam_formation_local_cycle import source_closure
from run_aam_direct_energy_trajectory_law import atlas_receipts
from run_aam_mark_candidate_filter import verify
from run_aam_original_tape_pgas import ATLAS, CHAINS, GATES, SWEEPS, WARM, paired, probabilities
from run_aam_suffix_group_smoother import acquired_data, features, global_history
from run_aam_temporal_energy_law import predict_variant, readout_inputs


ROOT = Path(__file__).resolve().parent
PROTOCOL = ROOT / 'AAM_PHASE_WIDE_SMC_PROTOCOL_ZH.md'
NAMESPACE = 'aam-original-shared-phase-width-vs-replay-v1'
PARENT_NAMESPACE = 'aam-finite-y-group-marginal-pgas-blind-v1'
UNITS = {'dev': list(range(2026140000, 2026140032)), 'held': list(range(2026140032, 2026140096))}
KEY = 148906707
WIDE = {'raw_wide': (1, 436, False), 'collapsed_wide': (1, 436, True), 'collapsed_islands': (4, 109, True)}
NATIVE = (*WIDE, 'phase_pgas16')
DIRECT = (*NATIVE, *ATLAS)
HYBRID = 'wide_rbf_mix'
METRICS = ('early18', 'hidden_delta32_6', 'hidden_t3_6', 'hidden_t2_6', 'global24')
ALPHAS = np.array([0., .1, .25, .5, .75, 1.])
SHIFT = np.arange(16, dtype=np.float64) * .375
CALLS = 1308
SWEEP_CALLS = 93
GATE_CONTROLS = ('raw_wide', 'collapsed_islands', 'phase_pgas16', *ATLAS)


def register(out, atlas, parent, review, workers):
    p, units, model, refs = atlas_receipts(atlas)
    previous = parent / 'REGISTERED.json'
    summary = parent / 'SCORED.json'
    audit = parent / 'INDEPENDENT_AUDIT.json'
    old, old_result, old_audit = (read(x) for x in (previous, summary, audit))
    new_ids = sum(UNITS.values(), [])
    if (old['namespace'] != PARENT_NAMESPACE or old['physics'] != p['physics']
            or old_audit.get('passed') is not True or old_audit.get('scored_sha256') != sha(summary)
            or old_audit.get('registered_sha256') != sha(previous) or old_result['go_to_inverse']
            or set(new_ids).intersection(old['units'] + sum(p['units'].values(), []))
            or len(set(new_ids)) != len(new_ids) or workers < 1
            or not np.array_equal(np.asarray(old['shift']), SHIFT)
            or 3 * WARM + CHAINS * SWEEPS * SWEEP_CALLS != CALLS):
        raise ValueError('Require fresh role IDs, an audited same-law comparator and the identical budget contract.')
    sources = source_closure([Path(__file__), ROOT / 'audit_aam_phase_wide_smc.py'])
    original_settings = read(atlas / 'TRAINED_AND_SELECTED.json')['selected']
    out.mkdir(exist_ok=False)
    r = out / 'REGISTERED.json'
    write(r, {**stamp(), 'namespace': NAMESPACE, 'units': UNITS, 'workers': workers,
              'physics': p['physics'], 'previous_independent_case': str(parent), 'atlas_units': units,
              'atlas_model': str(model), 'atlas_settings': original_settings, 'alpha_grid': ALPHAS,
              'hybrid': HYBRID, 'methods': DIRECT, 'wide': WIDE, 'metrics': METRICS, 'gates': GATES,
              'gate_controls': GATE_CONTROLS, 'offsets': SHIFT, 'key': KEY,
              'depth': {'warm': WARM, 'chains': CHAINS, 'sweeps': SWEEPS, 'particles': 16,
                        'calls_per_sweep': SWEEP_CALLS},
              'native_forecast_calls_per_method_mother': CALLS, 'native_truth_calls_per_mother': 3,
              'planned_all_forecast_calls': CALLS * len(NATIVE) * len(new_ids),
              'planned_all_truth_calls': 3 * len(new_ids),
              'tuning': 'DEV only, minimum arithmetic mean of early18 and hidden_delta32_6, ties smaller alpha',
              'paired_control': 'raw_wide and collapsed_wide have identical addressed RNG seeds, with outcomes-dependent paths',
              'old_held_targets_read': 0, 'new_wet_measurements': 0, 'inverse_scored': False,
              'inputs': {str(x): sha(x) for x in (*refs, previous, summary, audit, PROTOCOL, review)},
              'sources': {str(path): sha(path) for path in sources.values()}})
    snapshot = out / 'SOURCE_SNAPSHOT'
    snapshot.mkdir()
    for name, path in sources.items():
        copyfile(path, snapshot / name)
    write(out / 'RUN.json', {**stamp(), 'registration_sha256': sha(r), 'before_native': True})
    event(out, 'registered_wide_phase_smc', all_new_mothers=len(new_ids), forecast_budget=CALLS)


def lock(out):
    path = out / 'REGISTERED.json'
    p = read(path)
    if (p['namespace'] != NAMESPACE or p['units'] != UNITS or p['key'] != KEY
            or p['native_forecast_calls_per_method_mother'] != CALLS
            or sha(path) != read(out / 'RUN.json')['registration_sha256']):
        raise ValueError('Require unchanged prospective registration.')
    for name, checksum in {**p['sources'], **p['inputs']}.items():
        if sha(name) != checksum:
            raise ValueError('Frozen source or receipt changed: ' + name)
    return p


def fetch(ref):
    return verify(Path(ref['path']), ref)


def rng(unit, *address):
    return np.random.default_rng(np.random.SeedSequence([KEY, unit, *address]))


def truth(job):
    out, cfg, unit = job
    with threadpool_limits(limits=1):
        measured, raw = acquired_data(cfg, unit)
    if raw['calls'] != {'attempted': 3, 'completed': 3, 'failed': 0}:
        raise ArithmeticError('Every independent truth mother must have all three original frames.')
    return {'unit': unit, 'query': np_save(out / f'{unit}_QUERY.npz', measured=measured),
            'truth': save_tree(out / f'{unit}_TRUTH.npz', raw), 'native_calls': 3}


def phase_pgas(cfg, y, unit):
    warm = [make_history(cfg, y, rng(unit, 2, j)) for j in range(WARM)]
    initial = probabilities(np.asarray([h['logl'].sum() for h, _ in warm]))
    starts = rng(unit, 3).choice(WARM, size=CHAINS, p=initial)
    calls = sum(receipt['native_calls'] for _, receipt in warm)
    histories, weights, logs = [], [], []
    for chain, start in enumerate(starts):
        path = phase_reference(cfg, y, warm[start][0])
        for step in range(SWEEPS):
            path, receipt = sweep(cfg, y, path, 16, rng(unit, 4, chain, step), mode='pgas')
            histories.extend(receipt.pop('terminal_histories'))
            weights.append(receipt['terminal_weights'] / (CHAINS * SWEEPS))
            calls += receipt['native_calls']
            logs.append(receipt)
    ledger = {'warm_weights': initial, 'starts': starts, 'sweeps': logs, 'native_calls': calls,
              'warm_initial_ids': [packet_fingerprint(h['packets'][0]) for h, _ in warm]}
    return histories, np.concatenate(weights), np.stack([h['phase_weights'] for h in histories]), ledger


def compact_smc(ledger):
    return {**{k: v for k, v in ledger.items() if k not in ('islands', 'terminal_histories')},
            'islands': [{k: v for k, v in island.items() if k != 'terminal_histories'}
                        for island in ledger['islands']]}


def summaries(histories, probs, phase, ledger, cfg, support, cache):
    weights = np.asarray(probs, dtype=float)
    phase = np.asarray(phase, dtype=float)
    if phase.ndim != 2:
        raise ValueError('Require a conditional phase vector for each native trajectory.')
    offsets = SHIFT if phase.shape[1] == len(SHIFT) else np.zeros(1)
    if (ledger['native_calls'] != CALLS or weights.shape != (len(histories),)
            or phase.shape != (len(histories), len(offsets)) or np.any(weights < 0.) or np.any(phase < 0.)):
        raise ValueError('Invalid registered budget or original weighted path representation.')
    np.testing.assert_allclose(weights.sum(), 1., rtol=0., atol=2.e-13)
    np.testing.assert_allclose(phase.sum(axis=1), 1., rtol=0., atol=2.e-13)
    paths = []
    local, global_, cameras = [], [], []
    for h in histories:
        packets = h['packets']
        ids = tuple(packet_fingerprint(x) for x in packets)
        for name, packet in zip(ids, packets, strict=True):
            support.setdefault(name, packet)
        paths.append(ids)
        key = (ids, len(offsets))
        if key not in cache:
            cache[key] = (history_orbit(packets, offsets), global_history(packets, cfg['prefix_clock']))
        left, right = cache[key]
        local.append(left)
        global_.append(right)
        cameras.append(h['clean_orbit'])
    return {'weights': weights, 'phase': phase, 'offsets': offsets, 'local': np.stack(local),
            'global': np.stack(global_), 'clean_orbit': np.stack(cameras), 'path_ids': np.asarray(paths),
            'ledger': ledger}


def store_packet_dag(out, unit, packets):
    shapes, families, family_ids, row_ids = {}, [], [], []
    column_names = tuple('state_' + s for s in STATE_FIELDS) + tuple('packet_' + p for p in PACKET_FIELDS[1:])
    for packet in packets.values():
        row = tuple(np.asarray(x) for x in (*packet[0], *packet[1:]))
        key = tuple((x.shape, x.dtype.str) for x in row)
        if key not in shapes:
            shapes[key] = len(families)
            families.append([])
        family = shapes[key]
        family_ids.append(family)
        row_ids.append(len(families[family]))
        families[family].append(row)
    packed = {'packet_ids': np.asarray(list(packets)), 'packet_family': np.asarray(family_ids, dtype=np.int64),
              'packet_row': np.asarray(row_ids, dtype=np.int64)}
    for family, rows in enumerate(families):
        for j, name in enumerate(column_names):
            packed[f'g{family}_{name}'] = np.stack([row[j] for row in rows])
    return np_save(out / f'{unit}_PACKETS.npz', **packed)


def predict_one(job):
    out, cfg, view = job
    unit = view['unit']
    y = np_load(fetch(view['query']))['measured']
    methods, packets, cache, sampler_time, readout_time = {}, {}, {}, {}, {}
    with threadpool_limits(limits=1):
        for name, (islands, width, collapsed) in WIDE.items():
            started = perf_counter()
            address = 1 if name == 'collapsed_islands' else 0
            ledger = forward(cfg, y, islands, width, rng(unit, address), collapsed=collapsed)
            sampler_time[name] = perf_counter() - started
            started = perf_counter()
            methods[name] = summaries(ledger['terminal_histories'], ledger['terminal_weights'],
                                      ledger['terminal_phase_weights'], compact_smc(ledger), cfg, packets, cache)
            readout_time[name] = perf_counter() - started
            del ledger
        started = perf_counter()
        h, w, phase, receipt = phase_pgas(cfg, y, unit)
        sampler_time['phase_pgas16'] = perf_counter() - started
        started = perf_counter()
        methods['phase_pgas16'] = summaries(h, w, phase, receipt, cfg, packets, cache)
        readout_time['phase_pgas16'] = perf_counter() - started
    raw = store_packet_dag(out, unit, packets)
    saved = save_tree(out / f'{unit}_FORECAST.npz', {'unit': unit, 'query': y, 'methods': methods,
                      'native_call_counts': {name: m['ledger']['native_calls'] for name, m in methods.items()},
                      'seconds_sampler_including_phase_filter': sampler_time,
                      'seconds_hidden_phase_and_global_readouts': readout_time, 'native_packets': raw})
    return {'unit': unit, 'forecast': saved, 'packets': raw, 'native_calls': CALLS * len(NATIVE)}


def collected(fun, tasks, pool, out, ids, label):
    rows = []
    for record in pool.map(fun, tasks, chunksize=1):
        rows.append(record)
        event(out, label, unit=record['unit'], native_calls=record['native_calls'])
    if [row['unit'] for row in rows] != ids:
        raise ArithmeticError('A registered mother is missing or out of order.')
    return rows


def freeze_role(out, p, role):
    alpha_path = out / 'DEV_SELECTED.json'
    if role == 'held':
        chosen = read(alpha_path)
        if (chosen['held_truth_opened'] is not False or chosen['alpha'] not in ALPHAS
                or chosen['dev_forecast_sha256'] != sha(out / 'dev' / 'PREDICTIONS_FROZEN.json')):
            raise ValueError('Held inference requires frozen DEV-only mixture selection.')
    folder = out / role
    folder.mkdir(exist_ok=False)
    ids = UNITS[role]
    write(folder / 'STARTED.json', {**stamp(), 'role': role, 'units': ids,
                                    'registration_sha256': sha(out / 'REGISTERED.json')})
    with ProcessPoolExecutor(max_workers=p['workers']) as pool:
        actual = collected(truth, [(folder, p['physics'], i) for i in ids], pool, out, ids, role + '_truth_sealed')
        write(folder / 'ACQUIRED.json', {**stamp(), 'role': role, 'rows': actual, 'truth_calls': 3 * len(ids)})
        views = [{'unit': x['unit'], 'query': x['query']} for x in actual]
        prediction = collected(predict_one, [(folder, p['physics'], x) for x in views], pool, out, ids, role + '_blind_forecast')
    atlas = load_tree(p['atlas_model'])
    y = np.stack([np_load(fetch(v['query']))['measured'] for v in views])
    with threadpool_limits(limits=1):
        values = readout_inputs(atlas, y)
        weights = {m: predict_variant(m, p['atlas_settings'].get(m), atlas, values)['weights'] for m in ATLAS}
    pointer = np_save(folder / 'ATLAS_WEIGHTS.npz', query=y, **weights)
    write(folder / 'PREDICTIONS_FROZEN.json', {**stamp(), 'role': role, 'units': ids, 'rows': prediction,
          'registration_sha256': sha(out / 'REGISTERED.json'), 'acquired_sha256': sha(folder / 'ACQUIRED.json'),
          'atlas_weights': pointer, 'truth_opened': False,
          'devalpha_sha256': sha(alpha_path) if role == 'held' else None})


def combine_identical(x, w):
    """Exact numerical aggregation of coincident readouts from a legal finite law."""
    x, w = np.asarray(x, dtype=float), np.asarray(w, dtype=float)
    keep = w > 0.
    unique, order = np.unique(x[keep], axis=0, return_inverse=True)
    return unique, np.bincount(order.reshape(-1), weights=w[keep], minlength=len(unique))


def integral(x, w, y, v, block=256):
    return sum(float(w[i:i+block] @ cdist(x[i:i+block], y) @ v) for i in range(0, len(w), block))


def moments(atoms, weights, truth):
    x = atoms.reshape(len(atoms), -1)
    y = truth.reshape(1, -1)
    x, w = combine_identical(x, weights)
    root_d = np.sqrt(y.shape[1])
    return x, w, float(w @ cdist(x, y)[:, 0]) / root_d, integral(x, w, x, w) / root_d, root_d


def select_output(local, global_, w, wg, scale):
    return ((local, w), (local[:, 2] - local[:, 1], w), (local[:, 2], w), (local[:, 1], w),
            (global_ / scale, wg))


def initial_support(model):
    owner = model['path_ids'][:, 0]
    _, ids = np.unique(owner, return_inverse=True)
    mass = np.bincount(ids, weights=model['weights'])
    return [int(np.count_nonzero(mass)), float(1. / (mass @ mass))]


def score_one(job):
    cfg, f, tr, atlas, aw, scale, alpha = job
    with threadpool_limits(limits=1):
        actual = load_tree(fetch(tr['truth']))
        h = actual['packets']
        observed = features(h[:-1], h[-1], cfg)
        local = observed['local']
        targets = (local, local[2] - local[1], local[2], local[1], observed['global_early'] / scale)
        forecast = load_tree(fetch(f['forecast']))
        if forecast['unit'] != tr['unit'] or forecast['unit'] != f['unit']:
            raise ValueError('Prediction and truth must belong to the same independent mother.')
        outputs = {}
        for name, model in forecast['methods'].items():
            w = model['weights']
            mass = (w[:, None] * model['phase']).ravel()
            outputs[name] = select_output(model['local'].reshape(-1, 3, 2, 3), model['global'], mass, w, scale)
        for name, w in aw.items():
            outputs[name] = select_output(atlas['local'], atlas['global_early'], w, w, scale)
        terms, score_time = {}, {}
        for name, atoms in outputs.items():
            started = perf_counter()
            terms[name] = [moments(x, w, y) for (x, w), y in zip(atoms, targets, strict=True)]
            score_time[name] = perf_counter() - started
        direct = {name: [r[2] - .5 * r[3] for r in parts] for name, parts in terms.items()}
        mixed, started = [], perf_counter()
        for left, right in zip(terms['collapsed_wide'], terms['kernel'], strict=True):
            x, w, first_x, spread_x, d = left
            y, v, first_y, spread_y, _ = right
            cross = integral(x, w, y, v) / d
            other = 1. - alpha
            mixed.append(alpha * first_x + other * first_y
                         - .5 * (alpha**2 * spread_x + other**2 * spread_y + 2. * alpha * other * cross))
        score_time[HYBRID] = perf_counter() - started
        return {'unit': f['unit'], 'direct': direct, 'mixed_by_metric': np.stack(mixed, axis=1),
                'initial_support': {n: initial_support(m) for n, m in forecast['methods'].items()},
                'seconds_sampler_including_phase_filter': forecast['seconds_sampler_including_phase_filter'],
                'seconds_hidden_phase_and_global_readouts': forecast['seconds_hidden_phase_and_global_readouts'],
                'seconds_exact_energy': score_time}


def role_scores(out, p, role, alpha):
    folder = out / role
    r = folder / 'PREDICTIONS_FROZEN.json'
    freeze = read(r)
    if (freeze['units'] != UNITS[role] or freeze['truth_opened'] is not False
            or freeze['registration_sha256'] != sha(out / 'REGISTERED.json')
            or freeze['acquired_sha256'] != sha(folder / 'ACQUIRED.json')
            or (role == 'held' and freeze['devalpha_sha256'] != sha(out / 'DEV_SELECTED.json'))):
        raise ValueError('Require all blind frozen predictions before opening truth.')
    write(folder / 'REVEAL_STARTED.json', {**stamp(), 'role': role, 'forecast_sha256': sha(r), 'alpha': alpha})
    a = read(folder / 'ACQUIRED.json')
    w = np_load(fetch(freeze['atlas_weights']))
    train = load_tree(p['atlas_model'])
    jobs = [(p['physics'], f, t, train['bank'], {n: w[n][j] for n in ATLAS}, train['global_scale'], alpha)
            for j, (f, t) in enumerate(zip(freeze['rows'], a['rows'], strict=True))]
    with ProcessPoolExecutor(max_workers=p['workers']) as pool:
        rows = list(pool.map(score_one, jobs, chunksize=1))
    if [r['unit'] for r in rows] != UNITS[role]:
        raise ArithmeticError('Incomplete mother-level scoring.')
    return rows


def tune(out, p):
    rows = role_scores(out, p, 'dev', ALPHAS)
    losses = np.stack([row['mixed_by_metric'] for row in rows])
    objective = losses[:, :, :2].mean(axis=(0, 2))
    np.testing.assert_allclose(losses[:, 0], [r['direct']['kernel'] for r in rows], rtol=0., atol=2.e-15)
    np.testing.assert_allclose(losses[:, -1], [r['direct']['collapsed_wide'] for r in rows], rtol=0., atol=2.e-15)
    selected = int(np.argmin(objective))
    points = save_tree(out / 'DEV_SCORE_ARRAYS.npz', {'rows': rows, 'alpha_grid': ALPHAS})
    write(out / 'DEV_SELECTED.json', {**stamp(), 'dev_forecast_sha256': sha(out / 'dev' / 'PREDICTIONS_FROZEN.json'),
          'alpha': float(ALPHAS[selected]), 'selected_index': selected, 'alpha_grid': ALPHAS,
          'objective_means': objective, 'held_truth_opened': False, 'arrays': points})
    event(out, 'alpha_selected_on_dev_only', alpha=float(ALPHAS[selected]))


def score(out, p):
    choice = read(out / 'DEV_SELECTED.json')
    alpha = float(choice['alpha'])
    rows = role_scores(out, p, 'held', np.asarray([alpha]))
    losses = {name: np.asarray([r['direct'][name] for r in rows]) for name in DIRECT}
    losses[HYBRID] = np.stack([r['mixed_by_metric'][0] for r in rows])
    gains = {metric: {name: paired(losses[name][:, i] - losses[HYBRID][:, i]) for name in DIRECT}
             for i, metric in enumerate(METRICS)}
    contrasts = {candidate: {m: paired(losses['phase_pgas16'][:, i] - losses[candidate][:, i])
                             for i, m in enumerate(METRICS)} for candidate in WIDE}
    gate = all(gains[m][x][1] > 0. for m in GATES for x in GATE_CONTROLS)
    diagnostics = {n: np.asarray([r['initial_support'][n] for r in rows]) for n in NATIVE}
    archive = save_tree(out / 'HELD_SCORE_ARRAYS.npz', {'rows': rows, 'losses': losses, 'initial_support': diagnostics})
    write(out / 'SCORED.json', {**stamp(), 'forecast_sha256': sha(out / 'held' / 'PREDICTIONS_FROZEN.json'),
          'devalpha_sha256': sha(out / 'DEV_SELECTED.json'), 'alpha': alpha, 'mothers': len(rows), 'metrics': METRICS,
          'means': {n: v.mean(axis=0) for n, v in losses.items()}, 'paired_gains': gains, 'width_vs_pgas': contrasts,
          'distinct_initial_and_weighted_initial_ess_means': {n: d.mean(axis=0) for n, d in diagnostics.items()},
          'timing_seconds_means': {key: {name: np.mean([r[key][name] for r in rows]) for name in rows[0][key]}
                                   for key in ('seconds_sampler_including_phase_filter',
                                               'seconds_hidden_phase_and_global_readouts', 'seconds_exact_energy')},
          'actual_held_forecast_native_calls': len(rows) * len(NATIVE) * CALLS, 'go_to_inverse': bool(gate),
          'new_wet_measurements': 0, 'inverse_scored': False, 'arrays': archive})
    event(out, 'new_held_scored', mothers=len(rows), inverse_gate=bool(gate))


if __name__ == '__main__':
    cli = ArgumentParser(description=__doc__)
    commands = cli.add_subparsers(dest='phase', required=True)
    r = commands.add_parser('register')
    for arg in ('out', 'atlas', 'parent', 'review'):
        r.add_argument(arg, type=Path)
    r.add_argument('--workers', type=int, default=8)
    for phase in ('dev', 'tune', 'held', 'score'):
        commands.add_parser(phase).add_argument('out', type=Path)
    q = cli.parse_args()
    out = q.out.resolve()
    if q.phase == 'register':
        register(out, q.atlas.resolve(), q.parent.resolve(), q.review.resolve(), q.workers)
    elif q.phase in ('dev', 'held'):
        freeze_role(out, lock(out), q.phase)
    elif q.phase == 'tune':
        tune(out, lock(out))
    else:
        score(out, lock(out))
