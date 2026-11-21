"""Separated observations and stationary same-law phase-teacher training support."""

from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.spatial.distance import cdist
from threadpoolctl import threadpool_limits

from aam_common_phase_hidden import history_orbit
from aam_formation_actual_rollout import packet_fingerprint
from aam_formation_cycle_driver import np_load, np_save, read, save_tree, sha, write
from aam_formation_local_initial import build_packet
from aam_formation_path_cycle import event, load_tree, stamp
from aam_original_tape_pgas import SCHEMA as ORIGINAL_TAPE
from aam_phase_marginal_tape_pgas import PHASES, phase_reference, sweep
from aam_phase_stationary_truth_lift import inverse_truth_reference as archived_inverse_reference
from aam_snapshot_process import observe
from check_aam_transverse_orbit import same as check_transverse_equivariance
from run_aam_mark_candidate_filter import camera_likelihood, verify
from run_aam_phase_wide_smc import store_packet_dag
from run_aam_suffix_group_smoother import acquired_data, counts, features, global_history, native_step


ROOT = Path(__file__).resolve().parent
SPLITS = {'train': list(range(2026143000, 2026143256)), 'dev': list(range(2026143256, 2026143320)),
          'held': list(range(2026143320, 2026143448))}
METRICS = ('early18', 'hidden_delta32_6', 'hidden_t3_6')
ACQUIRE_CALLS = 3
CHAINS, SWEEPS, PARTICLES, SWEEP_CALLS = 4, 8, 16, 93
TEACHER_CALLS = CHAINS * (ACQUIRE_CALLS + SWEEPS * SWEEP_CALLS)


class InversePhaseParityError(ArithmeticError):
    def __init__(self, record):
        super().__init__('Canonical inverse-phase native replay differs from the original conjugacy.')
        self.record = record


def fetch(ref):
    return verify(Path(ref['path']), ref)


def acquire_worker(job):
    """Only original generating-law data; the two files have different access roles."""
    out, cfg, role, unit = job
    if role not in SPLITS or unit not in SPLITS[role]:
        raise ValueError('Use a mother assigned to this acquisition role in the prospective split.')
    folder = out / role
    with threadpool_limits(limits=1):
        observed, raw = acquired_data(cfg, unit)
    if raw['calls'] != counts(ACQUIRE_CALLS) or len(raw['packets']) != 4:
        save_tree(folder / f'{unit}_FAILED.npz', {'unit': unit, 'role': role, 'partial': raw})
        raise ValueError('Require all original acquisition ages and budget.')
    if np.asarray(observed).shape != (3, 4) or not np.isfinite(observed).all():
        raise ValueError('Original mother observations must be finite three-frame readings.')
    obs_file = np_save(folder / f'{unit}_OBS.npz', measured=observed)
    raw_file = save_tree(folder / f'{unit}_RAW.npz', raw)
    return {'unit': unit, 'role': role, 'obs': obs_file, 'raw': raw_file, 'native_calls': ACQUIRE_CALLS}


def inverse_truth_reference(cfg, y, raw, unit, phase):
    """Draw a base-gauge representative of the actual generating path and replay it.

    Ideal simultaneous periodic Y symmetry makes the canonical replay the
    inverse translate of the truth under every original addressed innovation.
    We preserve the physically advanced canonical packets and fail explicitly
    if finite precision changes a reaction or invalidates this conjugacy.
    """
    if type(phase) is not int or not 0 <= phase < PHASES:
        raise ValueError('Require one original finite-group index.')
    anchor = archived_inverse_reference(cfg, y, raw, unit, phase)
    proof = anchor['truth_lift_receipt']
    shift = proof['inverse_displacement']
    wraps = proof['initial_y_image_numbers']
    box = float(raw['packets'][0][2][2])
    initial = anchor['initial']
    h = [build_packet(cfg, initial, factors=np.asarray(cfg['factors']))]
    tables = anchor['tapes']
    intervals, parity = [], []
    counter = counts()
    attention = None
    try:
        parity.append(check_transverse_equivariance(raw['packets'][0], h[0], shift, box, wraps))
        for c, table in enumerate(tables):
            attention = c + 1
            packet, receipt = native_step(h[-1], cfg, c, table, counter)
            h.append(packet)
            intervals.append(receipt)
            parity.append(check_transverse_equivariance(raw['packets'][c + 1], packet, shift, box, wraps))
            expected, actual = anchor['packets'][c + 1][0], packet[0]
            for name in ('periodic_parent', 'periodic_shift'):
                np.testing.assert_array_equal(getattr(actual, name), getattr(expected, name))
            np.testing.assert_allclose(actual.unwrapped, expected.unwrapped, rtol=0., atol=2.e-11)
            prior_spans = raw['intervals'][c]['spans']
            for old, new in zip(prior_spans, receipt['spans'], strict=True):
                for key in ('slab', 'attempt', 'censored', 'exponential', 'uniform', 'chosen'):
                    if old.get(key) != new.get(key):
                        raise ValueError('The inverse replay changed a consumed native address: ' + key)
        if counter != counts(ACQUIRE_CALLS):
            raise ArithmeticError('Inverse truth reference did not replay the exact original clock.')
        clean = np.stack([observe(p) for p in h[1:]])
        ref = {'schema': ORIGINAL_TAPE, 'initial': initial, 'packets': tuple(h), 'tapes': tuple(tables),
               'intervals': tuple(intervals), 'clean': clean, 'logl': camera_likelihood(clean, y, cfg['noise'])}
        lift = phase_reference(cfg, y, ref)
        np.testing.assert_allclose(lift['clean_orbit'][:, phase], raw['clean'], rtol=0., atol=2.e-11)
    except (AssertionError, ArithmeticError, ValueError) as e:
        raise InversePhaseParityError({'unit': unit, 'phase': phase, 'inverse_shift': shift, 'failed_at': attention,
                                      'native_counter': counter, 'packet_sha256': [packet_fingerprint(p) for p in h],
                                      'original_chain_sha256': [packet_fingerprint(p) for p in raw['packets']],
                                      'native_events_each_frame': [int(p[0].counts[0]) for p in h],
                                      'original_events_each_frame': [int(p[0].counts[0]) for p in raw['packets']],
                                      'error': repr(e), 'current_lifted_packets': h}) from e
    return lift, {'phase': phase, 'inverse_shift': shift, 'native_counter': counter,
                  'max_torus_position_error': float(max(parity)), 'position_parity_by_frame': parity,
                  'phase_camera_max_abs_error': float(np.max(np.abs(lift['clean_orbit'][:, phase] - raw['clean']))),
            'canonical_packet_sha256': [packet_fingerprint(p) for p in h], 'archived_anchor_proof': proof}


def _seed(seed, unit, chain, purpose, tick=0):
    return np.random.default_rng(np.random.SeedSequence([seed, unit, chain, purpose, tick]))


def teach_one(cfg, y, raw, unit, seed):
    """Return four final-sweep Rao–Blackwell laws from independently seeded chains.

    Uniform inverse phases put each reference under the ideal marginal cavity
    posterior. All chains share the same generating truth, so their law is
    stationary per chain but the chains are not independent mother replicates.
    The final categorical particle and phase selections are integrated out.
    """
    if unit not in SPLITS['train'] or raw['calls'] != counts(ACQUIRE_CALLS):
        raise ValueError('Stationary references are permitted only on new training truth.')
    y = np.asarray(y, dtype=float)
    if y.shape != (3, 4) or not np.isfinite(y).all():
        raise ValueError('Use the immutable original noisy query for its training truth.')
    chains, support, cache = [], {}, {}
    native_calls = 0
    for j in range(CHAINS):
        phase = int(_seed(seed, unit, j, 0).integers(PHASES))
        ref, reconciliation = inverse_truth_reference(cfg, y, raw, unit, phase)
        for identity, packet in zip(reconciliation['canonical_packet_sha256'], ref['packets'], strict=True):
            support.setdefault(identity, packet)
        native_calls += reconciliation['native_counter']['completed']
        rows = []
        for k in range(SWEEPS):
            ref, ledger = sweep(cfg, y, ref, PARTICLES, _seed(seed, unit, j, 1, k), mode='pgas')
            if ledger['native_calls'] != SWEEP_CALLS or ledger['calls'] != counts(SWEEP_CALLS):
                raise ArithmeticError('Phase teacher must use the unchanged exact original PGAS16 sweep.')
            terminal = ledger.pop('terminal_histories')
            rows.append({**ledger, 'selected_path_ids': [packet_fingerprint(p) for p in ref['packets']]})
            native_calls += SWEEP_CALLS
        data = []
        for row in terminal:
            packets = row['packets']
            ids = tuple(packet_fingerprint(p) for p in packets)
            for identity, packet in zip(ids, packets, strict=True):
                support.setdefault(identity, packet)
            if ids not in cache:
                cache[ids] = (history_orbit(packets, row['phase_offsets']), global_history(packets, cfg['prefix_clock']))
            local, glob = cache[ids]
            data.append((local, glob, ids, row['clean_orbit'], row['phase_logl']))
        local, glob, ids, cameras, logl = zip(*data, strict=True)
        chains.append({'initial_reference': reconciliation, 'sweeps': rows, 'path_weights': ledger['terminal_weights'],
                       'phase_weights': ledger['terminal_phase_weights'], 'local': np.stack(local),
                       'global_early': np.stack(glob), 'path_ids': np.asarray(ids), 'clean_orbit': np.stack(cameras),
                       'phase_logl': np.stack(logl)})
    if native_calls != TEACHER_CALLS:
        raise ArithmeticError('Every training mother must receive the four preregistered teacher chains.')
    packed = {n: np.stack([row[n] for row in chains]) for n in ('path_weights', 'phase_weights', 'local', 'global_early',
                                                               'path_ids', 'clean_orbit', 'phase_logl')}
    np.testing.assert_allclose(packed['path_weights'].sum(axis=1), 1., rtol=0., atol=2.e-13)
    np.testing.assert_allclose(packed['phase_weights'].sum(axis=2), 1., rtol=0., atol=2.e-13)
    return {**packed, 'chains': [{'initial_reference': row['initial_reference'], 'sweeps': row['sweeps']} for row in chains],
            'packet_support': support, 'native_calls': native_calls, 'unit': unit}


def _coordinates(local):
    return local, local[:, 2] - local[:, 1], local[:, 2]


def linear_cross(atlas_local, target_local, probabilities):
    """Exact three normalized Euclidean linear terms of a weighted target law."""
    atlas = np.asarray(atlas_local, dtype=float)
    targets = np.asarray(target_local, dtype=float)
    w = np.asarray(probabilities, dtype=float)
    if (atlas.shape != (512, 3, 2, 3) or targets.shape != (len(w), 3, 2, 3)
            or np.any(w < 0.) or not np.isfinite(atlas).all() or not np.isfinite(targets).all()):
        raise ValueError('Teacher must compare complete joint histories to the original 512-atom atlas.')
    np.testing.assert_allclose(w.sum(), 1., rtol=0., atol=2.e-13)
    result = []
    for a, b in zip(_coordinates(atlas), _coordinates(targets), strict=True):
        x, y = a.reshape(len(a), -1), b.reshape(len(b), -1)
        result.append((cdist(x, y) @ w) / np.sqrt(x.shape[1]))
    return np.stack(result)


def teach_worker(job):
    out, cfg, row, seed, atlas_local = job
    unit = row['unit']
    folder = out / 'train'
    if (row['role'] != 'train' or unit not in SPLITS['train']
            or Path(row['obs']['path']).resolve() != (folder / f'{unit}_OBS.npz').resolve()
            or Path(row['raw']['path']).resolve() != (folder / f'{unit}_RAW.npz').resolve()):
        raise ValueError('Only the freshly registered TRAIN files may enter the stationary teacher.')
    raw, y = load_tree(fetch(row['raw'])), np_load(fetch(row['obs']))['measured']
    started = perf_counter()
    try:
        with threadpool_limits(limits=1):
            result = teach_one(cfg, y, raw, unit, seed)
            w = (result['path_weights'][..., None] * result['phase_weights'] / CHAINS).ravel()
            teacher_cross = linear_cross(atlas_local, result['local'].reshape(-1, 3, 2, 3), w)
            packets = raw['packets']
            truth = features(packets[:-1], packets[-1], cfg)['local']
            truth_cross = linear_cross(atlas_local, truth[None], np.ones(1))
    except Exception as e:
        save_tree(folder / f'{unit}_TEACHER_FAILED.npz', {'unit': unit, 'raw': row['raw'], 'obs': row['obs'],
                  'error': repr(e), 'parity_failure': getattr(e, 'record', None)})
        raise
    support = store_packet_dag(folder, unit, result.pop('packet_support'))
    teacher = save_tree(folder / f'{unit}_TEACHER.npz', {**result, 'measured': y, 'support': support,
                        'teacher_cross': teacher_cross, 'truth_cross': truth_cross, 'metrics': METRICS,
                        'elapsed_seconds': perf_counter() - started})
    return {'unit': unit, 'teacher': teacher, 'packets': support, 'native_calls': result['native_calls'],
            'teacher_cross': teacher_cross, 'truth_cross': truth_cross, 'measured': y}


def committed(out):
    registration = out / 'REGISTERED.json'
    p = read(registration)
    active = read(out / 'RUN.json')
    if (p['units'] != SPLITS or p['sources'].get(str(Path(__file__).resolve())) != sha(__file__)
            or active.get('registered_sha256') != sha(registration)):
        raise ValueError('Require the unchanged parent prospective registration before any native acquisition.')
    for path, expected in {**p['sources'], **p['inputs']}.items():
        if sha(path) != expected:
            raise ValueError('Frozen source or input changed: ' + path)
    return p


def _collect(out, worker, jobs, units, workers, label):
    finished = []
    with ProcessPoolExecutor(max_workers=workers) as p:
        for row in p.map(worker, jobs, chunksize=1):
            event(out, label, unit=row['unit'], native_calls=row['native_calls'])
            finished.append(row)
    if [r['unit'] for r in finished] != units:
        raise RuntimeError('All independently registered units must be retained without replacements.')
    return finished


def acquire(out, plan, role):
    if role == 'dev' and not (out / 'FIT_FROZEN.json').is_file():
        raise ValueError('Seal both train-only fits before acquiring development data.')
    if role == 'held':
        selection = read(out / 'DEV_SELECTED.json')
        if (selection['go_to_fresh_held'] is not True or selection['held_not_opened'] is not True
                or selection['forecast_sha256'] != sha(out / 'DEV_PREDICTIONS_FROZEN.json')):
            raise ValueError('Only the frozen development decision can authorize held acquisition.')
    folder = out / role
    folder.mkdir(exist_ok=False)
    r = sha(out / 'REGISTERED.json')
    write(folder / 'ACQUISITION_STARTED.json', {**stamp(), 'role': role, 'registration_sha256': r})
    ids = SPLITS[role]
    rows = _collect(out, acquire_worker, [(out, plan['physics'], role, u) for u in ids], ids, plan['workers'],
                    role + '_original_native_mother_sealed')
    f = out / f'{role.upper()}_ACQUIRED.json'
    write(f, {**stamp(), 'registration_sha256': r, 'role': role, 'rows': rows,
              'native_calls': ACQUIRE_CALLS * len(ids)})
    return f


def teach(out, plan):
    source = out / 'TRAIN_ACQUIRED.json'
    train = read(source)
    if (train['registration_sha256'] != sha(out / 'REGISTERED.json') or train['role'] != 'train'
            or [r['unit'] for r in train['rows']] != SPLITS['train']):
        raise ValueError('Only the sealed fresh 256 training mothers may supply reference rows.')
    atlas = load_tree(plan['atlas_model'])['bank']['local']
    write(out / 'TRAIN_TEACHER_STARTED.json', {**stamp(), 'training_acquisition_sha256': sha(source),
          'allowed_roles': ['train'], 'native_budget': TEACHER_CALLS * len(train['rows'])})
    tasks = [(out, plan['physics'], r, plan['teacher_seed'], atlas) for r in train['rows']]
    rows = _collect(out, teach_worker, tasks, SPLITS['train'], plan['workers'], 'stationary_training_teacher_frozen')
    columns = ('teacher_cross', 'truth_cross', 'measured')
    cross = np_save(out / 'TRAIN_TEACHER_CROSS.npz', unit=np.asarray(SPLITS['train']), metrics=np.asarray(METRICS),
                   **{k: np.stack([r[k] for r in rows]) for k in columns})
    f = out / 'TRAIN_TEACHER_FROZEN.json'
    write(f, {**stamp(), 'training_acquisition_sha256': sha(source), 'role': 'train', 'arrays': cross,
              'rows': [{k: v for k, v in r.items() if k not in columns} for r in rows],
              'native_calls': sum(r['native_calls'] for r in rows), 'raw_truth_as_query_input': False,
              'true_training_path_used_for_reference_only': True,
              'held_read': False, 'dev_read': False})
    return f


if __name__ == '__main__':
    cli = ArgumentParser(description=__doc__)
    sub = cli.add_subparsers(dest='phase', required=True)
    collect = sub.add_parser('acquire')
    collect.add_argument('out', type=Path)
    collect.add_argument('role', choices=tuple(SPLITS))
    sub.add_parser('teach').add_argument('out', type=Path)
    args = cli.parse_args()
    folder = args.out.resolve()
    reg = committed(folder)
    if args.phase == 'acquire':
        acquire(folder, reg, args.role)
    else:
        teach(folder, reg)
