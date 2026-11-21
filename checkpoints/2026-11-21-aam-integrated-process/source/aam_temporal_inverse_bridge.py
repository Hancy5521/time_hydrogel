"""Shared temporal-law future archive, with the unchanged planted V5 design metric."""

import hashlib
from operator import index
from pathlib import Path

import numpy as np

from aam_first_interval_paired_fork import future_for_action, score_future_for_action
from aam_formation_actual_rollout import packet_fingerprint
from aam_formation_cycle_core import ACTIONS, FUTURE_TIMES, PREFIX_CLOCK
from aam_formation_cycle_data import _read6, check_actual_state, property_readout
from aam_formation_cycle_driver import np_load, read, sha
from aam_formation_cycle_methods import (
    CHANNEL_COUNT, DIMENSION, GOALS, evaluate_choices, pack_targets, project_forest_support,
    score_forecast, select_actions,
)
from aam_formation_local_baselines import validate_fixed_normalization
from aam_formation_path_cycle import load_tree
from run_aam_mark_candidate_filter import verify


NS = 'aam-temporal-law-unchanged-V5-action-bridge-v1'
ENTROPY = [int.from_bytes(hashlib.sha256(NS.encode()).digest()[j:j+4], 'little') for j in range(0, 16, 4)]
TRAIN_REPEATS = 2
REFERENCE_REPEATS = 4
ROLES = {'library': 1, 'independent_reference': 2}
FUTURE_ROLES = {'seed': 1, 'native_seeds': 2, 'placement_seed': 3}


def v5_fixed_metric(v5):
    """Load only the historical normalization pinned by the original V5 plan."""
    v5 = Path(v5).resolve(strict=True)
    plan = read(v5 / 'REGISTERED.json')
    if plan['namespace'] != 'aam-formation-explicit-initial-local-v5-20261119':
        raise ValueError('Bind the actual V5 fixed metric registration.')
    path = Path(plan['fixed_macro_arrays']).resolve(strict=True)
    if sha(path) != plan['inputs'][str(path)]:
        raise ValueError('The actual old mechanical scales differ from frozen V5 inputs.')
    with np.load(path, allow_pickle=False) as payload:
        scales = {n.removeprefix('normalization_'): payload[n] for n in payload.files
                  if n.startswith('normalization_')}
    return validate_fixed_normalization(scales), {'v5_registration_sha256': sha(v5 / 'REGISTERED.json'),
        'normalization_path': str(path), 'normalization_sha256': sha(path)}


def training_boundary(parent):
    """Read the completed train/dev lock and its training records without loading any held raw."""
    parent = Path(parent).resolve(strict=True)
    reg_path, acq_path = parent / 'REGISTERED.json', parent / 'ACQUIRED.json'
    reg, acquired = read(reg_path), read(acq_path)
    fitted_path = parent / 'TRAINED_AND_SELECTED.json'
    fitted = read(fitted_path)
    records = acquired['groups']['train']
    train_ids = [record['unit'] for record in records]
    if (reg['namespace'] != 'aam-temporal-energy-positive-law-v1'
            or acquired['registration_sha256'] != sha(reg_path)
            or fitted['acquisition_sha256'] != sha(acq_path) or fitted['held_opened'] is not False
            or fitted['train_ids'] != train_ids or reg['units']['train'] != train_ids
            or len(set(train_ids)) != len(train_ids)):
        raise ValueError('Future support must use the frozen training mother ordering.')
    validate_catalogue(reg['physics'])
    return records, reg['physics'], {
        'temporal_registration_sha256': sha(reg_path), 'acquisition_sha256': sha(acq_path),
        'trained_and_selected_sha256': sha(fitted_path),
        'fitted': str(verify(parent / 'TRAINED_NATIVE_LAW.npz', fitted['fitted'])),
        'fitted_sha256': fitted['fitted']['sha256'], 'selected': fitted['selected'],
    }


def load_training_c3(parent, record, config):
    """Call for registered training records only. A held payload cannot resolve to this path."""
    folder = Path(parent) / 'train'
    unit = record['unit']
    archive = verify(folder / f'{unit}_RAW.npz', record['raw'])
    raw = load_tree(archive)
    if len(raw['packets']) != 4 or raw['calls'] != {'attempted': 3, 'completed': 3, 'failed': 0}:
        raise ValueError('A train atom must be a complete lawful four-packet history.')
    for packet, age in zip(raw['packets'], config['prefix_clock'], strict=True):
        check_actual_state(packet[0], expected_age=age)
    return raw['packets'][-1], {'source_raw_sha256': record['raw']['sha256'], 'unit': unit}


def validate_catalogue(config):
    if (tuple(map(tuple, config['actions'])) != ACTIONS or tuple(config['future_times']) != FUTURE_TIMES
            or tuple(config['prefix_clock']) != PREFIX_CLOCK):
        raise ValueError('Use the exact original six actions and native formation clocks.')


def future_addresses(role, unit, repeat):
    unit, repeat = index(unit), index(repeat)
    if not (0 <= unit < 2**32 and 0 <= repeat < 2**32):
        raise ValueError('Register uint32 address coordinates.')
    common = [*ENTROPY, ROLES[role], unit, repeat]
    rng = {key: np.random.Generator(np.random.PCG64(np.random.SeedSequence([*common, code])))
           for key, code in FUTURE_ROLES.items()}
    # The marker is for unchanged legacy validation; it is not a physical seed.
    return {'seed': int(rng['seed'].integers(0, 2**32 - 51, dtype=np.uint32)),
            'native_seeds': rng['native_seeds'].integers(0, 2**32, 3, dtype=np.uint32),
            'placement_seed': int(rng['placement_seed'].integers(0, 2**32, dtype=np.uint32))}


def continue_c3_atom(packet, config, *, role, unit, archive):
    """Make all six future branches before any method can attach probability."""
    validate_catalogue(config)
    repeats = {'library': TRAIN_REPEATS, 'independent_reference': REFERENCE_REPEATS}[role]
    check_actual_state(packet[0], expected_age=PREFIX_CLOCK[-1])
    original = packet_fingerprint(packet)
    cutoff = property_readout(packet, config)
    archive(role, unit, None, None, {'packet_fingerprint': original, 'properties': cutoff})
    if not cutoff['valid']:
        raise ValueError('The actual cutoff response is invalid; retain its raw result.')
    cutoff6 = _read6(packet, cutoff)
    future, future_micro, future_winding = [], [], []
    for repeat in range(repeats):
        streams = future_addresses(role, unit, repeat)
        values, micro, winding = [], [], []
        for action_index, action in enumerate(ACTIONS):
            raw = future_for_action(packet, action, config, streams, f'{role}:{unit}')
            # Store real invalid branches as well as valid branches before the numeric check.
            archive(role, unit, repeat, action_index, {'streams': streams, 'raw': raw,
                'actual_cutoff_packet_fingerprint': original})
            value, local, wind = score_future_for_action(raw, packet, action, config, streams)
            values.append(value)
            micro.append(local)
            winding.append(wind)
        future.append(values)
        future_micro.append(micro)
        future_winding.append(winding)
    if packet_fingerprint(packet) != original:
        raise RuntimeError('The common action origin changed.')
    return {'cutoff': cutoff6, 'future': np.asarray(future), 'future_micro': np.asarray(future_micro),
            'future_winding': np.asarray(future_winding), 'role': role, 'unit': unit,
            'native_intervals': 3 * len(ACTIONS) * repeats, 'mechanical_snapshots': 1 + 3 * len(ACTIONS) * repeats,
            'origin_fingerprint': original}


def shared_atoms(cutoff, future):
    atoms = pack_targets(cutoff, future)
    if atoms.shape[1] != TRAIN_REPEATS:
        raise ValueError('Each library mother has the same two legal future addresses.')
    return atoms


def weighted_decision(atoms, mother_weights, normalization):
    """Same archive and one mother probability vector across suffixes, actions and time."""
    atoms = np.asarray(atoms, float)
    p = np.asarray(mother_weights, float)
    if (p.ndim != 1 or atoms.shape != (len(p), TRAIN_REPEATS, DIMENSION)
            or not np.isfinite(p).all() or np.any(p < 0.) or p.sum() <= 0.):
        raise ValueError('Require all weighted complete library atoms in their registered order.')
    q = np.repeat(p / p.sum() / TRAIN_REPEATS, TRAIN_REPEATS)
    trajectories = atoms.reshape(-1, DIMENSION)
    decision = select_actions(trajectories, q, normalization, np.ones(len(ACTIONS), dtype=bool))
    return {'targets114': trajectories, 'weights': q, 'choices': decision['choices'],
            'expected_loss': decision['expected_loss'], 'goal_names': GOALS}


def evaluate_frozen(choices, future, normalization):
    """Scoring driver must verify the full prediction freeze before using this helper."""
    shape = (REFERENCE_REPEATS, len(ACTIONS), len(FUTURE_TIMES), CHANNEL_COUNT)
    if np.shape(future) != shape:
        raise ValueError('Require four independently addressed full native reference suffixes.')
    return evaluate_choices(choices, future, normalization)


def score_frozen_forecast(forecast, reference, normalization):
    return {**score_forecast(forecast['targets114'], forecast['weights'],
                reference_future=reference['future'], reference_cutoff=reference['cutoff'],
                normalization=normalization), **evaluate_frozen(forecast['choices'], reference['future'], normalization)}


def fit_full_library_forest(cameras, atoms, ids, normalization, seed):
    """Same V5 ExtraTrees family, with the common larger train-only library."""
    from sklearn.ensemble import ExtraTreesRegressor

    x, atoms, ids = np.asarray(cameras, float), np.asarray(atoms, float), np.asarray(ids)
    scale = normalization['target114_scale']
    if (x.shape != (len(ids), 12) or atoms.shape != (len(ids), TRAIN_REPEATS, DIMENSION)
            or ids.dtype.kind not in 'iu' or len(np.unique(ids)) != len(ids)
            or not np.isfinite(x).all() or not np.isfinite(atoms).all()):
        raise ValueError('One finite same-information query and two targets per unique training mother.')
    fold = ids % 4
    counts = np.bincount(fold.astype(np.int64), minlength=4)
    if np.any(counts == 0) or len(set(counts)) != 1 or not 0 <= index(seed) <= 2**32-5:
        raise ValueError('Register four equal mother-level folds and an admissible forest seed base.')
    center = atoms.mean(axis=1)
    oof = np.empty_like(center)
    for k in range(5):
        estimator = ExtraTreesRegressor(n_estimators=128, min_samples_leaf=4, random_state=seed+k, n_jobs=1)
        mask = fold != k
        estimator.fit(x[mask], center[mask] / scale)
        if k < 4:
            oof[~mask] = estimator.predict(x[~mask]) * scale
    return {'forest': estimator, 'residual': atoms-oof[:, None, :], 'oof_prediction': oof, 'fold': fold,
            'training_ids': ids.copy(), 'forest_seed_base': seed, 'forest_fits': 5}


def forest_decision(query, fitted, normalization):
    """Frozen same-X12 direct comparator with the original V5 numerical projection."""
    x = np.asarray(query, float).reshape(1, 12)
    center = fitted['forest'].predict(x)[0] * normalization['target114_scale']
    raw = center + fitted['residual'].reshape(-1, DIMENSION)
    paths, diagnostic = project_forest_support(raw)
    q = np.full(len(paths), 1./len(paths))
    decision = select_actions(paths, q, normalization, np.ones(len(ACTIONS), dtype=bool))
    return {'targets114': paths, 'raw_targets114': raw, 'weights': q, 'choices': decision['choices'],
            'expected_loss': decision['expected_loss'], 'projection_diagnostic': diagnostic,
            'goal_names': GOALS}


def fixtures():
    """Synthetic checks for the common actions, complete-atom weights and score algebra."""
    from unittest.mock import patch

    from aam_formation_cycle_methods import process_goal_losses, training_normalization

    rng = np.random.default_rng(200512)
    cfg = {'actions': ACTIONS, 'future_times': FUTURE_TIMES, 'prefix_clock': PREFIX_CLOCK}
    qcv = rng.uniform(.1, .95, (8, TRAIN_REPEATS, 6, 3, 3))
    qcv[..., 2] = np.sort(qcv[..., 2], axis=-1)
    motion = -np.sort(-rng.uniform(.005, .04, (8, TRAIN_REPEATS, 6, 3, 3)), axis=-1)
    future = np.concatenate([qcv, motion], axis=-1)
    cutoff = np.tile([.10, .10, .05, .03, .02, .01], (8, 1))
    p = np.arange(1, 9)
    norm = validate_fixed_normalization(training_normalization(future))
    atoms = shared_atoms(cutoff, future)
    pol = weighted_decision(atoms, p, norm)
    loss = process_goal_losses(future, norm)['loss'].mean(axis=1)
    exact = np.average(loss, axis=0, weights=p).T
    np.testing.assert_allclose(pol['expected_loss'], exact)
    np.testing.assert_array_equal(pol['choices'], exact.argmin(axis=1))
    np.testing.assert_allclose(pol['weights'].reshape(-1, 2).sum(axis=-1), p/p.sum())
    untouched = future[:2].reshape(4, 6, 3, 6)
    result = score_frozen_forecast(pol, {'future': untouched, 'cutoff': cutoff[0]}, norm)
    expected = process_goal_losses(untouched, norm)['loss'].mean(axis=0)
    np.testing.assert_allclose(result['selected_goal_loss'], expected[pol['choices'], np.arange(2)])
    raw = []

    def log_branch(role, unit, repeat, action, value):
        raw.append((role, unit, repeat, action, value))

    def fake_branch(packet, action, config, streams, mother_id):
        return {'action': action, 'streams': streams}

    def fake_score(value, packet, action, config, streams):
        return future[0, 0, ACTIONS.index(action)], np.zeros((3, 8)), np.zeros((3, 3), bool)

    here = globals()
    with patch.dict(here, {'check_actual_state': lambda *_a, **_k: None, 'packet_fingerprint': lambda _: 'origin',
                           'property_readout': lambda *_: {'valid': True}, '_read6': lambda *_: cutoff[0],
                           'future_for_action': fake_branch, 'score_future_for_action': fake_score}):
        future_bank = continue_c3_atom(('fake',), cfg, role='library', unit=0, archive=log_branch)
    assert future_bank['future'].shape == (2, 6, 3, 6) and len(raw) == 13
    assert future_bank['native_intervals'] == 36 and future_bank['mechanical_snapshots'] == 37
    for repeat in range(2):
        branches = [x for x in raw if x[2] == repeat]
        np.testing.assert_array_equal([x[3] for x in branches], np.arange(6))
        for value in branches:
            np.testing.assert_array_equal(value[-1]['streams']['native_seeds'],
                                          future_addresses('library', 0, repeat)['native_seeds'])
    assert np.any(future_addresses('library', 0, 0)['native_seeds']
                  != future_addresses('independent_reference', 0, 0)['native_seeds'])
    print('PASS synthetic: two-repeat full actions; same-origin streams, exact weighted risks, four-reference aliases.')


if __name__ == '__main__':
    fixtures()
