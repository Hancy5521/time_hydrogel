"""Fixed-depth cold replay with an identical SIR initializer and shared RNG addresses."""
from operator import index
from time import perf_counter

import numpy as np

from aam_bounded_path_replay import reject_with_remaining_bound
from aam_formation_cycle_core import _advance, _physical_config
from aam_formation_local_initial import build_packet, draw_initial, initial_fingerprint, propose_initial
from aam_formation_local_tempering import (
    PATH_SCHEMA, _cost, generate_path, mh_log_acceptance, systematic_indices, update_weights,
)
from aam_formation_path_tempering import _copy_packet, _fingerprints, _merge_cost, _query
from aam_snapshot_process import likelihood, observe


SCHEMA = 'aam-fixed-depth-budgeted-cold-replay-v1'
RNG_ADDRESS_NAMESPACE = 0x41414D42
NATIVE_COUNTS = ('native_segment_attempted', 'native_segment_completed', 'native_segment_failed')


def _rng(seed, role, *address):
    return np.random.default_rng(np.random.SeedSequence([RNG_ADDRESS_NAMESPACE, seed, role, *address]))


def _native_counts(cost):
    return {name: cost[name] for name in NATIVE_COUNTS}


def _replay(config, actual, factors, measured, noise, old, initial, seeds, change,
            ratio, uniform, bounded, birth_id, clock, row, cost):
    """Replay a proposal with costs visible even when native evaluation fails."""
    cost['path_evaluations'] += 1
    start = 0 if change is None else change
    row['phase'] = 'initialization' if change is None else 'reuse_checkpoint'
    if change is None:
        started = perf_counter()
        cost['actual_initializations'] += 1
        try:
            raw = build_packet(actual, initial, factors=factors)
        finally:
            cost['initialization_seconds'] += perf_counter() - started
        checkpoints = [_copy_packet(raw, cost, readonly=True)]
        clean = []
        reused_calls = []
    else:
        checkpoints = list(old['checkpoint_packets'][:start + 1])
        raw = _copy_packet(checkpoints[-1], cost)
        clean = list(old['clean'][:start])
        reused_calls = [dict(call, reused_checkpoint_interval=True) for call in old['native_calls'][:start]]

    calls = row['native_calls']
    for camera in range(start, 3):
        row['camera'] = camera
        if bounded:
            row['phase'] = 'camera_bound'
            observed = likelihood(np.asarray(clean, float).reshape(-1, 4), measured[:camera], noise)
            cost['likelihood_evaluations'] += 1
            reject, threshold = reject_with_remaining_bound(observed, old['log_likelihood'], 1., ratio, uniform)
            if reject:
                row.update(accepted=False, screened_at=camera, upper_log_acceptance=threshold,
                           camera_prefix_ll=observed)
                return None
        row['phase'] = 'native_progress'
        age = clock[:camera + 2]
        call = {'camera': camera, 'seed': int(seeds[camera]), 'clock': age.copy(),
                'completed': False, 'position_before': raw[0].position.copy(),
                'reused_checkpoint_interval': False}
        calls.append(call)
        try:
            _advance(raw, age, call['seed'], False, cost)
            call['completed'] = True
        finally:
            call['position_after'] = raw[0].position.copy()
        row['phase'] = 'clean_observation'
        checkpoints.append(_copy_packet(raw, cost, readonly=True))
        supplied = _copy_packet(raw, cost)
        started = perf_counter()
        try:
            clean.append(np.asarray(observe(supplied), float).copy())
        finally:
            cost['q_read_seconds'] += perf_counter() - started

    row['phase'] = 'full_likelihood'
    raw_readouts = np.asarray(clean, float)
    if raw_readouts.shape != (3, 4) or not np.isfinite(raw_readouts).all():
        raise ArithmeticError('A complete path needs three finite camera readouts.')
    camera_ll = likelihood(raw_readouts, measured, noise)
    cost['likelihood_evaluations'] += 1
    if camera_ll.shape != (3,) or not np.isfinite(camera_ll).all() or np.any(camera_ll > 0.):
        raise ArithmeticError('The complete path violated the frozen Gaussian camera law.')
    proposed_ll = float(camera_ll.sum())
    decision = mh_log_acceptance(1., old['log_likelihood'], proposed_ll, ratio)
    accepted = bool(uniform == 0. or np.log(uniform) < decision)
    row.update(accepted=accepted, proposal_camera_ll=camera_ll, log_acceptance=decision)
    if not accepted:
        return None

    row['phase'] = 'fingerprints'
    started = perf_counter()
    try:
        history, reacted = _fingerprints(checkpoints)
    finally:
        cost['fingerprint_seconds'] += perf_counter() - started
    return {'schema': PATH_SCHEMA, 'valid': True, 'status': 'complete', 'error': None,
            'mother_id': old['mother_id'], 'birth_id': birth_id, 'parent_birth_id': old['birth_id'],
            'changed_interval': change, 'initial_variables': initial,
            'initial_fingerprint': row['proposed_initial_fingerprint'],
            'informative_initial_fingerprint': initial_fingerprint(initial, informative_only=True),
            'interval_seeds': seeds, 'factors': old['factors'],
            'checkpoint_packets': checkpoints, 'initial_packet': checkpoints[0], 'packet': checkpoints[-1],
            'frames': {float(t): p[0] for t, p in zip(clock, checkpoints, strict=True)},
            'clean': raw_readouts, 'camera_log_likelihood': camera_ll, 'log_likelihood': proposed_ll,
            'history_fingerprint': history, 'reacted_structure_fingerprint': reacted,
            'native_calls': reused_calls + calls, 'cost': cost}


def sample_prefix(config, factors, measured, *, seed, mode, sweeps, native_cap,
                  prior_count=64, particles=16, mother_id=-1):
    """Produce equal-weight cutoff states after a precommitted number of cold sweeps.

    Both modes reserve a cap sufficient for every possible query. `bounded` can
    save realized calls through pathwise exact screening. More fixed sweeps are
    a separate finite-computation comparison, not a budget-stopped kernel.
    """
    if mode not in ('full', 'bounded'):
        raise ValueError('mode must be full or bounded.')
    seed, sweeps, native_cap, prior_count, particles = map(index, (seed, sweeps, native_cap, prior_count, particles))
    if seed < 0 or sweeps < 0 or prior_count <= 0 or particles <= 0:
        raise ValueError('Require a nonnegative seed and sweep count, and positive population sizes.')
    upper_bound = 3 * prior_count + 9 * particles * sweeps
    if native_cap < upper_bound:
        raise ValueError(f'The fixed-depth schedule needs a universal cap of at least {upper_bound}.')
    query, noise = _query(config, measured)
    actual = _physical_config(config, factors)
    clock = np.asarray(config['prefix_clock'], float)
    started = perf_counter()
    cost = _cost()
    bootstrap = {'prior': [], 'beta': 1.}
    moves, sweep_ledger = [], []

    def export(paths, error=None):
        return {'schema': SCHEMA, 'mode': mode, 'valid': error is None,
                'status': 'complete' if error is None else 'failed', 'error': error,
                'mother_id': mother_id, 'seed': seed, 'rng_address_namespace': RNG_ADDRESS_NAMESPACE,
                'native_cap': native_cap, 'native_upper_bound': upper_bound, 'planned_sweeps': sweeps,
                'completed_sweeps': len(sweep_ledger), 'prior_count': prior_count, 'particle_count': particles,
                'paths': paths, 'packets': [path['packet'] for path in paths],
                'weights': np.full(particles, 1. / particles) if error is None else np.empty(0),
                'bootstrap': bootstrap, 'moves': moves, 'sweep_ledger': sweep_ledger,
                'cost': cost, 'elapsed_seconds': perf_counter() - started}

    pool = []
    for slot in range(prior_count):
        tic = perf_counter()
        initial = draw_initial(config, _rng(seed, 0, slot))
        cost['prior_draw_seconds'] += perf_counter() - tic
        cost['prior_initial_draws'] += 1
        seeds = _rng(seed, 1, slot).integers(0, 2**32, 3, dtype=np.uint32)
        cost['seed_draws'] += 3
        path = generate_path(config, factors, initial, seeds, query, mother_id=mother_id, birth_id=f'prior_{slot:04d}')
        _merge_cost(cost, path['cost'])
        bootstrap['prior'].append({'slot': slot, 'initial_fingerprint': path['initial_fingerprint'],
                                   'interval_seeds': seeds, 'camera_ll': path['camera_log_likelihood'],
                                   'history_fingerprint': path['history_fingerprint'],
                                   'error': path['error'], **_native_counts(path['cost'])})
        if not path['valid']:
            return export([], {'phase': 'prior', 'slot': slot, 'error': path['error']})
        pool.append(path)
    prior_ll = [path['log_likelihood'] for path in pool]
    weights, _, ess = update_weights(np.full(prior_count, 1. / prior_count), prior_ll, 1.)
    indices, resample_u = systematic_indices(weights, _rng(seed, 2), particles)
    cost['systematic_uniforms'] += 1
    bootstrap.update(weights=weights, ess=ess, indices=indices, uniform=resample_u, **_native_counts(cost))
    paths = [pool[i] for i in indices]
    del pool

    for sweep in range(sweeps):
        sweep_started = perf_counter()
        native_before = cost['native_segment_completed']
        for slot, old in enumerate(paths):
            for coordinate in range(4):
                change = None if coordinate == 0 else coordinate - 1
                rng = _rng(seed, 3, sweep, slot, coordinate)
                initial, seeds, ratio = old['initial_variables'], old['interval_seeds'].copy(), 0.
                if change is None:
                    tic = perf_counter()
                    initial, ratio, metadata = propose_initial(config, initial, rng)
                    cost['initial_proposal_seconds'] += perf_counter() - tic
                    cost['initial_proposals'] += 1
                    self_loop = metadata['self_loop']
                else:
                    seeds[change] = rng.integers(0, 2**32, dtype=np.uint32)
                    cost['seed_draws'] += 1
                    cost['interval_proposals'] += 1
                    metadata = {'kind': 'interval_seed'}
                    self_loop = seeds[change] == old['interval_seeds'][change]
                row = {'sweep': sweep + 1, 'slot': slot, 'coordinate': coordinate, 'beta': 1.,
                       'initial_move': metadata, 'log_hastings': ratio, 'uniform': None,
                       'parent_birth_id': old['birth_id'], 'previous_log_likelihood': old['log_likelihood'],
                       'previous_history_fingerprint': old['history_fingerprint'],
                       'proposed_initial_fingerprint': initial_fingerprint(initial),
                       'proposed_interval_seeds': seeds, 'self_loop': bool(self_loop), 'screened_at': None,
                       'native_calls': [], 'camera': None, 'phase': 'self_loop'}
                moves.append(row)
                move_cost = _cost()
                move_started = perf_counter()
                try:
                    if self_loop:
                        if ratio != 0.:
                            raise ArithmeticError('An exact self-loop cannot have a Hastings correction.')
                        cost['self_loop_proposals'] += 1
                        row['accepted'] = True
                    else:
                        uniform = float(_rng(seed, 4, sweep, slot, coordinate).random())
                        row['uniform'] = uniform
                        cost['acceptance_uniforms'] += 1
                        birth_id = f's{sweep + 1:04d}_p{slot:04d}_c{coordinate}'
                        path = _replay(config, actual, factors, query, noise, old, initial, seeds,
                                       change, ratio, uniform, mode == 'bounded', birth_id, clock, row, move_cost)
                        if path is not None:
                            old = paths[slot] = path
                    cost['skipped_native_segments'] += 3 - (0 if change is None else change) - move_cost['native_segment_completed']
                    row.update(result_history_fingerprint=old['history_fingerprint'], result_birth_id=old['birth_id'])
                except Exception as exc:
                    row['error'] = {'type': type(exc).__name__, 'message': str(exc)}
                finally:
                    _merge_cost(cost, move_cost)
                    row.update(seconds=perf_counter() - move_started, **_native_counts(move_cost))
                if 'error' in row:
                    return export([], {'phase': row['phase'], 'sweep': sweep + 1, 'slot': slot,
                                       'coordinate': coordinate, 'camera': row['camera'], 'error': row['error']})
        sweep_ledger.append({'sweep': sweep + 1, 'seconds': perf_counter() - sweep_started,
                             'sweep_native_completed': cost['native_segment_completed'] - native_before,
                             'history_fingerprints': [path['history_fingerprint'] for path in paths],
                             'birth_ids': [path['birth_id'] for path in paths], **_native_counts(cost)})

    if not (cost['native_segment_attempted'] == cost['native_segment_completed'] <= upper_bound):
        raise AssertionError('Fixed-depth complete native accounting failed.')
    if cost['native_segment_completed'] != 3 * prior_count + sum(row['native_segment_completed'] for row in moves):
        raise AssertionError('The prefix native total does not reconcile with the move ledger.')
    return export(paths)
