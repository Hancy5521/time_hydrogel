"""V5 explicit-initial full-history SMC and streamed fixed-prior SIR.

Import is inert. The only changed sampler variable is the explicit mu5 initial
object; all native motion, reaction and camera calls use frozen V1--V4 code.
No seed is fabricated for that initial object. Source preparation must not call
any prior draw, constructor, propagation or RNG before prospective registration.
"""
from __future__ import annotations

from copy import deepcopy
from time import perf_counter

import numpy as np
from scipy.special import logsumexp

from aam_formation_path_tempering import (
    _query, _seed, _uniform, _fingerprints, _copy_packet, _copy_state,
    _merge_cost, _new_cost, _save_path,
)
from aam_formation_cycle_core import PREFIX_CLOCK, _advance, _physical_config
from aam_snapshot_process import observe, likelihood
from aam_formation_local_initial import (
    draw_initial, validate_initial, clone_initial, build_packet, propose_initial,
    refresh_stock_initial, initial_fingerprint,
)


SCHEMA = 'aam-explicit-initial-local-history-smc-v5'
PATH_SCHEMA = 'aam-explicit-initial-three-seed-prefix-v5'
INITIAL_KINDS = ('initiators', 'aam_position', 'mbaa_position', 'side_label', 'global')
CESS_FRACTION = .8
COLD_BETA = .8


def _cost():
    value = _new_cost()
    value.update(prior_initial_draws=0, initial_proposals=0, interval_proposals=0,
                 self_loop_proposals=0, skipped_native_segments=0,
                 prior_draw_seconds=0., initial_proposal_seconds=0.,
                 stock_refreshes=0, stock_refresh_seconds=0.,
                 stock_observation_checks=0)
    return value


def _positive_int(value, name):
    if type(value) is not int or value <= 0:
        raise ValueError(name+' must be a positive Python integer.')
    return value


def _seeds(value):
    raw = np.asarray(value)
    if (raw.shape != (3,) or raw.dtype.kind not in 'iu' or np.any(raw < 0)
            or np.any(raw.astype(np.uint64) >= 2**32)):
        raise ValueError('Exactly three independent uint32 interval seeds are required.')
    return raw.astype(np.uint32, copy=True)


def _weights(value):
    w = np.asarray(value, dtype=float)
    if (w.ndim != 1 or not len(w) or not np.isfinite(w).all()
            or np.any(w < 0.) or not 0. < float(w.sum()) < np.inf):
        raise ValueError('Retain all finite nonnegative masses, including zero weights.')
    return w/w.sum()


def _logweights(weights):
    w = _weights(weights); out = np.full(len(w), -np.inf)
    good = w > 0.; out[good] = np.log(w[good])
    return out


def conditional_ess(weights, log_likelihoods, delta):
    """N*(sum W exp(delta*ell))²/sum W exp(2delta*ell), dynamic N."""
    w, ell = _weights(weights), np.asarray(log_likelihoods, float)
    if ell.shape != w.shape or not np.isfinite(ell).all() or not 0. <= delta <= 1.:
        raise ValueError('CESS requires one finite log likelihood per particle.')
    if delta == 0.:
        return float(len(w))
    lw = _logweights(w)
    result = len(w)*np.exp(2*logsumexp(lw+delta*ell)-logsumexp(lw+2*delta*ell))
    if not np.isfinite(result) or result <= 0. or result > len(w)*(1.+1.e-12):
        raise ArithmeticError('Invalid weighted CESS.')
    return float(min(result, len(w)))


def update_weights(weights, log_likelihoods, delta):
    w, ell = _weights(weights), np.asarray(log_likelihoods, float)
    if ell.shape != w.shape or not np.isfinite(ell).all() or not 0. <= delta <= 1.:
        raise ValueError('One finite likelihood per particle and delta in[0,1] required.')
    if delta == 0.:
        return w.copy(), 0., float(1./(w@w))
    raw = _logweights(w)+delta*ell; normalizer = float(logsumexp(raw))
    updated = np.exp(raw-normalizer); updated /= updated.sum()
    return updated, normalizer, float(1./(updated@updated))


def next_beta(beta, weights, log_likelihoods, stage, working_stages):
    """Adaptive bridge; fixed working stages continue invariant moves at beta1."""
    if not 0. <= beta <= 1. or not 1 <= stage <= working_stages:
        raise ValueError('Invalid computational bridge stage.')
    if beta == 1.:
        return 1., float(len(weights)), False
    remaining = 1.-beta
    full = conditional_ess(weights, log_likelihoods, remaining)
    if full >= CESS_FRACTION*len(weights):
        return 1., full, False
    if stage == working_stages:
        return 1., full, True
    low, high = 0., remaining
    for _ in range(80):
        middle = .5*(low+high)
        if conditional_ess(weights, log_likelihoods, middle) >= CESS_FRACTION*len(weights):
            low = middle
        else:
            high = middle
    target = beta+low
    if not beta < target < 1.:
        raise ArithmeticError('Adaptive beta did not make representable progress.')
    return target, conditional_ess(weights, log_likelihoods, target-beta), False


def systematic_indices(weights, rng, count):
    """Systematic resampling with count draws, including arbitrary SIR pools."""
    _positive_int(count, 'count'); w = _weights(weights); uniform = _uniform(rng)
    cdf = np.cumsum(w); cdf[-1] = 1.
    positions = np.minimum((uniform+np.arange(count))/count, np.nextafter(1., 0.))
    parents = np.searchsorted(cdf, positions, side='right').astype(np.int64)
    if np.any(parents >= len(w)):
        raise ArithmeticError('Systematic resampling left the complete support.')
    return parents, uniform


def mh_log_acceptance(beta, previous_log_likelihood, proposed_log_likelihood,
                      log_prior_proposal_ratio=0.):
    """The collapsed label ratio is outside beta; at beta0 it still applies."""
    if (not 0. <= beta <= 1. or not np.isfinite(previous_log_likelihood)
            or not np.isfinite(proposed_log_likelihood) or not np.isfinite(log_prior_proposal_ratio)):
        raise ValueError('Finite declared likelihoods/prior-proposal ratio and beta in[0,1] required.')
    result = beta*(proposed_log_likelihood-previous_log_likelihood)+log_prior_proposal_ratio
    if not np.isfinite(result):
        raise ArithmeticError('Nonfinite MH arithmetic is a computation failure,not a physical rejection.')
    return float(min(0., result))


def _readonly_initial(initial):
    result = clone_initial(initial)
    for name in ('left', 'normal_x', 'points'):
        result[name].flags.writeable = False
    return result


def generate_path(config, factors, initial, interval_seeds, measured, *,
                  previous=None, changed_interval=None, mother_id=-1, birth_id=''):
    """Construct/replay an actual mu5 prefix; interval refresh costs3/2/1.

    Initial proposals use previous=None and reconstruct all three intervals.
    Partial replay requires identical explicit initial variables and all other
    interval seeds. Failures retain their actual partial output, never -inf.
    """
    started = perf_counter(); y, noise = _query(config, measured)
    actual = _physical_config(config, factors); validate_initial(actual, initial)
    seeds = _seeds(interval_seeds)
    if (previous is None) != (changed_interval is None):
        raise ValueError('A partial replay needs its actual parent and declared interval.')
    if changed_interval is not None and (type(changed_interval) is not int or changed_interval not in (0, 1, 2)):
        raise ValueError('changed_interval must be0,1,or2.')
    if previous is not None:
        old = _seeds(previous['interval_seeds']); keep = np.arange(3) != changed_interval
        if (previous.get('valid') is not True or len(previous['checkpoint_packets']) != 4
                or initial_fingerprint(initial) != initial_fingerprint(previous['initial_variables'])
                or not np.array_equal(seeds[keep], old[keep])
                or not np.array_equal(previous['factors'], np.asarray(factors))):
            raise ValueError('Only the declared interval seed may change on a complete same-law history.')
    cost = _cost(); cost['path_evaluations'] = 1
    checkpoints, clean, calls = [], [], []
    packet = None; error = None; phase = 'build_initial'; camera = -1
    try:
        if previous is None:
            segment = perf_counter(); cost['actual_initializations'] += 1
            try:
                packet = build_packet(actual, initial, factors=factors)
            finally:
                cost['initialization_seconds'] += perf_counter()-segment
            checkpoints.append(_copy_packet(packet, cost, readonly=True)); start = 0
        else:
            start = changed_interval
            checkpoints = [_copy_packet(p, cost, readonly=True)
                           for p in previous['checkpoint_packets'][:start+1]]
            clean = [row.copy() for row in np.asarray(previous['clean'])[:start]]
            calls = [deepcopy(row) for row in previous['native_calls'][:start]]
            for row in calls:
                row['reused_checkpoint_interval'] = True
            packet = _copy_packet(checkpoints[-1], cost)
        for camera in range(start, 3):
            phase = 'native_progress'; clock = np.asarray(config['prefix_clock'][:camera+2], float)
            call = {'camera': camera, 'seed': int(seeds[camera]), 'clock': clock.copy(),
                    'completed': False, 'position_before': packet[0].position.copy(),
                    'reused_checkpoint_interval': False}
            calls.append(call)
            try:
                _advance(packet, clock, int(seeds[camera]), False, cost)
                call['completed'] = True
            finally:
                call['position_after'] = packet[0].position.copy()
            checkpoints.append(_copy_packet(packet, cost, readonly=True))
            phase = 'clean_observation'; supplied = _copy_packet(packet, cost)
            segment = perf_counter()
            try:
                clean.append(np.asarray(observe(supplied), float).copy())
            finally:
                cost['q_read_seconds'] += perf_counter()-segment
        phase = 'full_likelihood'; clean_array = np.asarray(clean, float)
        if clean_array.shape != (3, 4) or not np.isfinite(clean_array).all():
            raise ArithmeticError('A complete path requires three finite original camera readouts.')
        camera_ll = np.asarray(likelihood(clean_array, y, noise), float)
        cost['likelihood_evaluations'] += 1
        if camera_ll.shape != (3,) or not np.isfinite(camera_ll).all():
            raise ArithmeticError('Nonfinite computation is not a zero-likelihood candidate.')
        phase = 'fingerprints'; segment = perf_counter()
        try:
            full, reacted = _fingerprints(checkpoints)
        finally:
            cost['fingerprint_seconds'] += perf_counter()-segment
    except Exception as exc:
        error = {'type': type(exc).__name__, 'message': str(exc), 'phase': phase, 'camera': camera}
        clean_array = np.asarray(clean, float).reshape(-1, 4)
        camera_ll = np.empty(0); full = reacted = ''
    frames = {float(t): _copy_state(p[0], cost) for t, p in zip(PREFIX_CLOCK, checkpoints)}
    for state in frames.values():
        for array in state:
            array.flags.writeable = False
    frozen_initial = _readonly_initial(initial)
    return {'schema': PATH_SCHEMA, 'valid': error is None, 'status': 'complete' if error is None else 'failed',
        'mother_id': mother_id, 'birth_id': birth_id,
        'parent_birth_id': None if previous is None else previous['birth_id'],
        'changed_interval': changed_interval, 'initial_variables': frozen_initial,
        'initial_fingerprint': initial_fingerprint(frozen_initial),
        'informative_initial_fingerprint': initial_fingerprint(frozen_initial, informative_only=True),
        'interval_seeds': seeds, 'factors': np.asarray(factors, float).copy(),
        'checkpoint_packets': checkpoints, 'initial_packet': None if not checkpoints else checkpoints[0],
        'packet': None if packet is None else _copy_packet(packet, cost, readonly=True),
        'frames': frames, 'clean': clean_array, 'camera_log_likelihood': camera_ll,
        'log_likelihood': None if error is not None else float(camera_ll.sum()),
        'history_fingerprint': full, 'reacted_structure_fingerprint': reacted,
        'native_calls': calls, 'cost': cost, 'error': error,
        'elapsed_seconds': perf_counter()-started,
        'scope': 'Explicit-mu5 actual history; no fabricated u0 or physical-mode interpretation of field hashes.'}


def _same_array(a, b):
    return isinstance(a, np.ndarray) and isinstance(b, np.ndarray) and a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


def _same_state(a, b):
    return a._fields == b._fields and all(_same_array(x, y) for x, y in zip(a, b, strict=True))


def _same_packet(a, b):
    return len(a) == len(b) == 5 and _same_state(a[0], b[0]) and all(_same_array(x, y) for x, y in zip(a[1:], b[1:], strict=True))


def _stock_state(state, points, old_initial_points, cost):
    """New complete State; only the six absent points/unwrapped may differ."""
    stock = np.arange(72, 78)
    if np.any(state.present[stock]):
        raise ValueError('The conditional stock refresh requires all6unfed MBAA absent at every prefix frame.')
    if not np.array_equal(state.points[stock], old_initial_points[stock]) or not np.array_equal(state.unwrapped[stock], old_initial_points[stock]):
        raise ValueError('Absent stock is no longer an untouched prior variable.')
    copied = _copy_state(state, cost)
    copied.points[stock] = points[stock]; copied.unwrapped[stock] = points[stock]
    present = np.asarray(state.present)
    for name, before, after in zip(state._fields, state, copied, strict=True):
        if name in ('points', 'unwrapped'):
            if not _same_array(before[present], after[present]):
                raise AssertionError('Stock refresh changed present material geometry.')
        elif not _same_array(before, after):
            raise AssertionError('Stock refresh changed a non-position State field: '+name)
        after.flags.writeable = False
    return copied


def _stock_packet(packet, points, old_points, cost):
    copied = _copy_packet(packet, cost)
    state = _stock_state(packet[0], points, old_points, cost)
    result = (state, *copied[1:])
    for old, new in zip(packet[1:], result[1:], strict=True):
        if not _same_array(old, new):
            raise AssertionError('Stock refresh changed physical parameter arrays.')
        new.flags.writeable = False
    return result


def refresh_path_stock(config, factors, path, measured, rng, *, parent_ref, birth_id):
    """Exact common final stock conditional update, no native call or reweight."""
    if path.get('valid') is not True or path.get('schema') != PATH_SCHEMA:
        raise ValueError('Only a complete actual V5 path may receive stock refresh.')
    if not np.array_equal(np.asarray(factors), path['factors']):
        raise ValueError('Stock refresh must keep the exact original forward law.')
    started = perf_counter(); cost = _cost(); cost['stock_refreshes'] = 1
    before_initial = initial_fingerprint(path['initial_variables'])
    before_info = initial_fingerprint(path['initial_variables'], informative_only=True)
    before_full, before_reacted = _fingerprints(path['checkpoint_packets'])
    if before_full != path['history_fingerprint'] or before_reacted != path['reacted_structure_fingerprint']:
        raise ValueError('Actual pre-refresh history identity has changed.')
    if (tuple(sorted(path['frames'])) != PREFIX_CLOCK or len(path['checkpoint_packets']) != 4
            or not _same_packet(path['initial_packet'], path['checkpoint_packets'][0])
            or not _same_packet(path['packet'], path['checkpoint_packets'][-1])
            or any(not _same_state(path['frames'][time], packet[0])
                   for time, packet in zip(PREFIX_CLOCK, path['checkpoint_packets'], strict=True))):
        raise ValueError('Initial/cutoff packets and cached frame histories must agree before stock refresh.')
    refreshed, metadata = refresh_stock_initial(config, path['initial_variables'], rng)
    validate_initial(config, refreshed)
    if initial_fingerprint(refreshed, informative_only=True) != before_info:
        raise AssertionError('Stock refresh changed informative initial variables.')
    old_points, new_points = path['initial_variables']['points'], refreshed['points']
    checkpoints = [_stock_packet(p, new_points, old_points, cost) for p in path['checkpoint_packets']]
    frames = {time: _stock_state(state, new_points, old_points, cost) for time, state in path['frames'].items()}
    initial_packet = _stock_packet(path['initial_packet'], new_points, old_points, cost)
    cutoff = _stock_packet(path['packet'], new_points, old_points, cost)
    y, noise = _query(config, measured); checked_clean = []
    for packet in checkpoints[1:]:
        supplied = _copy_packet(packet, cost); segment = perf_counter()
        try:
            checked_clean.append(np.asarray(observe(supplied), float))
            cost['stock_observation_checks'] += 1
        finally:
            cost['q_read_seconds'] += perf_counter()-segment
    clean = np.asarray(checked_clean, float)
    ll = np.asarray(likelihood(clean, y, noise), float); cost['likelihood_evaluations'] += 1
    if (not _same_array(clean, np.asarray(path['clean']))
            or not _same_array(ll, np.asarray(path['camera_log_likelihood']))
            or float(ll.sum()) != path['log_likelihood']):
        raise AssertionError('Exact stock conditional update changed original cameras or likelihood.')
    full, reacted = _fingerprints(checkpoints)
    if (reacted != before_reacted or initial_fingerprint(path['initial_variables']) != before_initial
            or _fingerprints(path['checkpoint_packets'])[0] != before_full):
        raise AssertionError('Stock update changed reacted fields or mutated its original source path.')
    if (not _same_packet(initial_packet, checkpoints[0]) or not _same_packet(cutoff, checkpoints[-1])
            or any(not _same_state(frames[time], packet[0])
                   for time, packet in zip(PREFIX_CLOCK, checkpoints, strict=True))):
        raise AssertionError('Stock refresh did not synchronize every packet and cached frame.')
    # Parent calls describe the original native prefix; stock updates are
    # separately declared, persisted and costed, never disguised as propagation.
    result = {**path, 'birth_id': birth_id, 'parent_birth_id': path['birth_id'],
        'initial_variables': _readonly_initial(refreshed), 'initial_fingerprint': initial_fingerprint(refreshed),
        'informative_initial_fingerprint': before_info, 'checkpoint_packets': checkpoints,
        'initial_packet': initial_packet, 'packet': cutoff, 'frames': frames,
        'clean': clean.copy(), 'camera_log_likelihood': ll.copy(), 'log_likelihood': path['log_likelihood'],
        'history_fingerprint': full, 'reacted_structure_fingerprint': reacted,
        'native_calls': deepcopy(path['native_calls']), 'cost': cost,
        'stock_refresh': {'source_path_ref': parent_ref, 'metadata': metadata,
            'informative_initial_unchanged': True, 'present_and_nonposition_fields_bitwise_unchanged': True,
            'original_cameras_and_likelihood_bitwise_unchanged': True, 'native_replay': False,
            'not_early_reconstruction_gain': True}, 'elapsed_seconds': perf_counter()-started}
    result['cost']['stock_refresh_seconds'] += perf_counter()-started
    return result


def _diversity(paths, weights):
    w = _weights(weights)
    values = [p['informative_initial_fingerprint'] for p in paths]
    _, group = np.unique(values, return_inverse=True)
    masses = np.bincount(group, weights=w)
    return {'distinct_informative_initial_field_identities': len(masses),
        'informative_initial_identity_mass_ESS': float(1./(masses@masses)),
        'distinct_full_history_field_fingerprints': len({p['history_fingerprint'] for p in paths}),
        'distinct_reacted_field_fingerprints': len({p['reacted_structure_fingerprint'] for p in paths}),
        'I': np.asarray([p['initial_variables']['I'] for p in paths], dtype=np.int64),
        'nL': np.asarray([np.count_nonzero(p['initial_variables']['left']) for p in paths], dtype=np.int64),
        'weights': w.copy(), 'scope': 'Before final stock refresh; field identities,not physical-mode count or independent sample ESS.'}


def _export(paths, refs, weights, cost, *, valid, error=None, **extra):
    w = np.asarray(weights, float)
    particles = [{'packet': p['packet'], 'initial_packet': p['initial_packet'], 'frames': p['frames'],
                  'initial_variables': p['initial_variables'], 'interval_seeds': p['interval_seeds'].copy(),
                  'birth_id': p['birth_id'], 'history_fingerprint': p['history_fingerprint'], 'camera': 2}
                 for p in paths]
    return {'schema': SCHEMA, 'valid': valid, 'status': 'complete' if valid else 'failed', 'error': error,
        'paths': paths, 'path_refs': refs, 'packets': [p['packet'] for p in paths], 'particles': particles,
        'weights': w.copy(), 'log_likelihoods': np.asarray([p['log_likelihood'] for p in paths], float),
        'initial_variables': [p['initial_variables'] for p in paths],
        'interval_seeds': np.asarray([p['interval_seeds'] for p in paths], dtype=np.uint32).reshape(-1, 3),
        'initial_fingerprints': np.asarray([p['initial_fingerprint'] for p in paths], dtype='U64'),
        'informative_initial_fingerprints': np.asarray([p['informative_initial_fingerprint'] for p in paths], dtype='U64'),
        'cost': cost, 'initial_weight_semantics': 'New explicit mu5 prior shared by allV5native arms;no old u0 or1/a.',
        'final_equal_weights_are_MC_approximation': True,
        'scope': 'Explicit initial variables and actual microscopic histories; no fabricated initialization seed genealogy.', **extra}


def _rngs(rngs, *, sir=False):
    needed = ('prior', 'seed', 'resample', 'stock') if sir else ('prior', 'proposal', 'seed', 'accept', 'resample', 'stock')
    if not isinstance(rngs, dict) or any(name not in rngs for name in needed):
        raise ValueError('Supply all independently named caller generators: '+','.join(needed))
    if len({id(rngs[name]) for name in needed}) != len(needed):
        raise ValueError('Caller stream roles must use distinct generator objects.')


def _schedule(fixed_betas, working_stages):
    if fixed_betas is None:
        return None
    value = np.asarray(fixed_betas, float)
    if (value.shape != (working_stages,) or not np.isfinite(value).all()
            or value[0] <= 0. or value[-1] != 1. or np.any(np.diff(value) <= 0.)):
        raise ValueError('Fixed betas must give every work stage,strictly increasing,and end at1.')
    return value


def _finish_stock(config, factors, measured, rng, paths, refs, weights, callback, cost):
    informative = _diversity(paths, weights); refreshed_paths, refreshed_refs, ledger = [], [], []
    for slot, (path, ref) in enumerate(zip(paths, refs, strict=True)):
        refreshed = refresh_path_stock(config, factors, path, measured, rng,
            parent_ref=ref, birth_id=f'final_stock_slot_{slot:03d}')
        _merge_cost(cost, refreshed['cost'])
        newref = _save_path(refreshed, f'final_stock/{slot:03d}', callback, cost)
        refreshed_paths.append(refreshed); refreshed_refs.append(newref)
        ledger.append({'slot': slot, 'source_ref': ref, 'refreshed_ref': newref,
                       'metadata': refreshed['stock_refresh']})
    return refreshed_paths, refreshed_refs, informative, ledger


def anneal_paths(config, factors, measured, rngs, *, mode, count=64,
                 working_stages=20, fixed_betas=None, mother_id=-1, archive_callback=None):
    """Fixed work: one initial proposal then interval0/1/2 per slot/stage.

    If beta reaches1 early, later stages use delta0 and invariant moves only.
    Only the last working stage unconditionally resamples N equal slots.
    Exact self-loops reuse the actual parent and spend zero native intervals.
    """
    count = _positive_int(count, 'count'); working_stages = _positive_int(working_stages, 'working_stages')
    if working_stages > 20 or mode not in ('local', 'global'):
        raise ValueError('Use local/global and no more than20registered working stages.')
    _rngs(rngs); _query(config, measured); _physical_config(config, factors)
    fixed = _schedule(fixed_betas, working_stages)
    if not callable(archive_callback):
        raise TypeError('Persist every actual prior/non-self proposal through archive_callback.')
    paths, refs, stages, cold_records = [], [], [], []
    cost = _cost(); weights = np.full(count, 1./count); betas = [0.]
    for slot in range(count):
        segment = perf_counter(); initial = draw_initial(config, rngs['prior'])
        cost['prior_initial_draws'] += 1; cost['prior_draw_seconds'] += perf_counter()-segment
        seeds = np.asarray([_seed(rngs['seed']) for _ in range(3)], dtype=np.uint32); cost['seed_draws'] += 3
        path = generate_path(config, factors, initial, seeds, measured, mother_id=mother_id,
                             birth_id=f'prior_{slot:03d}')
        _merge_cost(cost, path['cost']); ref = _save_path(path, f'prior/{slot:03d}', archive_callback, cost)
        if not path['valid']:
            return _export(paths, refs, weights[:len(paths)], cost, valid=False,
                error={'phase': 'prior', 'slot': slot, 'ref': ref, 'error': path['error']},
                stages=stages, stages_used=0, beta_schedule=np.asarray(betas), mode=mode)
        paths.append(path); refs.append(ref)
    for stage in range(1, working_stages+1):
        beta = betas[-1]; ell = np.asarray([p['log_likelihood'] for p in paths], float)
        if fixed is None:
            target, cess, forced = next_beta(beta, weights, ell, stage, working_stages)
        else:
            target = float(fixed[stage-1]); cess = conditional_ess(weights, ell, target-beta); forced = False
        delta = target-beta
        incoming = weights.copy()
        # In particular, delta0 at beta1 does NOT multiply the likelihood again.
        updated, logz, ess = update_weights(weights, ell, delta)
        final = stage == working_stages
        resampled = bool(final or ess < count/2.)
        row = {'stage': stage, 'beta_before': beta, 'beta': target, 'delta': delta,
            'conditional_ESS': cess, 'CESS_target': CESS_FRACTION*count, 'forced': forced,
            'bridge_weight_applied': delta > 0., 'invariant_cold_stage_only': beta == 1.,
            'weights_incoming': incoming, 'log_likelihoods_before': ell.copy(),
            'weights_updated': updated.copy(), 'relative_log_normalizer_increment': logz,
            'ordinary_ESS_before_resampling': ess, 'resampled': resampled,
            'path_refs_before': list(refs), 'moves': []}
        if resampled:
            parents, uniform = systematic_indices(updated, rngs['resample'], count)
            cost['systematic_uniforms'] += 1
            paths, refs = [paths[int(i)] for i in parents], [refs[int(i)] for i in parents]
            weights = np.full(count, 1./count)
        else:
            parents, uniform, weights = np.arange(count), None, updated.copy()
        row.update(parents=parents.copy(), systematic_uniform=uniform, weights_after_resampling=weights.copy())
        for slot in range(count):
            for position in range(4):
                old, oldref = paths[slot], refs[slot]
                ratio = 0.; metadata = {}; proposed_initial = old['initial_variables']
                seeds = old['interval_seeds'].copy(); coordinate = None
                if position == 0:
                    segment = perf_counter()
                    proposed_initial, ratio, metadata = propose_initial(config, old['initial_variables'], rngs['proposal'], mode=mode)
                    cost['initial_proposal_seconds'] += perf_counter()-segment; cost['initial_proposals'] += 1
                    validate_initial(config, proposed_initial)
                    kind = metadata['kind']; self_loop = bool(metadata['self_loop'])
                    if kind not in INITIAL_KINDS or not np.isfinite(ratio):
                        raise ValueError('Unknown initial move kind or nonfinite prior/proposal ratio.')
                    exact_same = initial_fingerprint(proposed_initial) == initial_fingerprint(old['initial_variables'])
                    if self_loop != exact_same or (self_loop and ratio != 0.):
                        raise ValueError('Initial self-loop metadata does not match actual variables.')
                    replay = 3
                else:
                    coordinate = position-1; kind = f'interval_{coordinate}'
                    seeds[coordinate] = _seed(rngs['seed']); cost['seed_draws'] += 1; cost['interval_proposals'] += 1
                    self_loop = np.array_equal(seeds, old['interval_seeds']); replay = 3-coordinate
                    metadata = {'kind': kind, 'self_loop': self_loop, 'interval': coordinate}
                move = {'slot': slot, 'position': position, 'kind': kind, 'changed_interval': coordinate,
                    'beta': target, 'cold': target >= COLD_BETA, 'parent_ref': oldref,
                    'parent_birth_id': old['birth_id'], 'proposal_metadata': metadata,
                    'log_prior_proposal_ratio': float(ratio), 'self_loop': self_loop,
                    'log_likelihood_before': old['log_likelihood'], 'native_segments_upper_bound': replay}
                if self_loop:
                    cost['self_loop_proposals'] += 1; cost['skipped_native_segments'] += replay
                    move.update(accepted=True, acceptance_uniform=None, log_acceptance_probability=0.,
                        log_likelihood_proposed=old['log_likelihood'], delta_log_likelihood=0.,
                        full_history_changed=False, reacted_structure_changed=False,
                        informative_initial_changed=False, proposal_ref=oldref, native_segments_completed=0)
                else:
                    candidate = generate_path(config, factors, proposed_initial, seeds, measured,
                        previous=None if position == 0 else old, changed_interval=coordinate,
                        mother_id=mother_id, birth_id=f'stage_{stage:02d}_slot_{slot:03d}_move_{position}')
                    _merge_cost(cost, candidate['cost']); move['candidate_cost'] = deepcopy(candidate['cost'])
                    move['log_likelihood_proposed'] = candidate['log_likelihood']
                    if candidate['valid']:
                        uniform_accept = _uniform(rngs['accept']); cost['acceptance_uniforms'] += 1
                        difference = candidate['log_likelihood']-old['log_likelihood']
                        logalpha = mh_log_acceptance(target, old['log_likelihood'],
                                                     candidate['log_likelihood'], float(ratio))
                        accepted = bool(uniform_accept == 0. or np.log(uniform_accept) < logalpha)
                        move.update(accepted=accepted, acceptance_uniform=uniform_accept,
                            log_acceptance_probability=logalpha, delta_log_likelihood=difference,
                            full_history_changed=candidate['history_fingerprint'] != old['history_fingerprint'],
                            reacted_structure_changed=candidate['reacted_structure_fingerprint'] != old['reacted_structure_fingerprint'],
                            informative_initial_changed=candidate['informative_initial_fingerprint'] != old['informative_initial_fingerprint'],
                            native_segments_completed=candidate['cost']['native_segment_completed'])
                    else:
                        move.update(accepted=None, acceptance_uniform=None, error=candidate['error'],
                                    log_acceptance_probability=None)
                    candidate['proposal_ledger'] = deepcopy(move)
                    ref = _save_path(candidate, f'stage/{stage:02d}/slot/{slot:03d}/move/{position}', archive_callback, cost)
                    move['proposal_ref'] = ref
                    if not candidate['valid']:
                        row['moves'].append(move); stages.append(row)
                        return _export(paths, refs, weights, cost, valid=False,
                            error={'phase': 'MH_computation_failure_not_rejection', 'ref': ref, 'error': candidate['error']},
                            stages=stages, stages_used=stage-1, beta_schedule=np.asarray(betas), mode=mode)
                    if move['accepted']:
                        paths[slot], refs[slot] = candidate, ref
                row['moves'].append(move)
                if move['cold']:
                    cold_records.append(deepcopy(move))
        row.update(path_refs_after=list(refs), weights_after=weights.copy(),
                   log_likelihoods_after=np.asarray([p['log_likelihood'] for p in paths], float))
        stages.append(row); betas.append(target)
    if betas[-1] != 1. or not np.array_equal(weights, np.full(count, 1./count)):
        raise AssertionError('The final working stage must end at beta1 with equalN slots.')
    upper = 3*count+9*count*working_stages
    if (cost['native_segment_attempted'] != upper-cost['skipped_native_segments']
            or cost['native_segment_completed'] != cost['native_segment_attempted']):
        raise AssertionError('Actual prefix calls do not reconcile with fullwork minus exactselfloops.')
    paths, refs, informative, refresh_ledger = _finish_stock(
        config, factors, measured, rngs['stock'], paths, refs, weights, archive_callback, cost)
    return _export(paths, refs, weights, cost, valid=True, stages=stages, stages_used=working_stages,
        working_stages=working_stages, mode=mode, beta_schedule=np.asarray(betas), betas=np.asarray(betas),
        cold_type_records=cold_records, pre_stock_informative_diversity=informative,
        stock_refresh_ledger=refresh_ledger, expected_native_prefix_upper_bound=upper,
        relative_log_normalizer_estimate=float(sum(r['relative_log_normalizer_increment'] for r in stages)),
        normalizer_scope='Original unnormalized Gaussian camera likelihood;beta1 cold stages addzero,not absolute material evidence.')


def sir_paths(config, factors, measured, rngs, *, pool_size=3904, count=64,
              mother_id=-1, archive_callback=None, load_path=None):
    """Stream all fixed-prior pool paths; only final selectedN stay resident."""
    pool_size = _positive_int(pool_size, 'pool_size'); count = _positive_int(count, 'count')
    _rngs(rngs, sir=True); _query(config, measured); _physical_config(config, factors)
    if not callable(archive_callback) or not callable(load_path):
        raise TypeError('SIR needs fullraw archive_callback and exactref load_path.')
    refs, ell, pool_seeds, pool_initial_hashes = [], [], [], []
    cost = _cost()
    for index in range(pool_size):
        segment = perf_counter(); initial = draw_initial(config, rngs['prior'])
        cost['prior_initial_draws'] += 1; cost['prior_draw_seconds'] += perf_counter()-segment
        seeds = np.asarray([_seed(rngs['seed']) for _ in range(3)], dtype=np.uint32); cost['seed_draws'] += 3
        path = generate_path(config, factors, initial, seeds, measured, mother_id=mother_id, birth_id=f'pool_{index:05d}')
        _merge_cost(cost, path['cost']); ref = _save_path(path, f'pool/{index:05d}', archive_callback, cost)
        refs.append(ref)
        if not path['valid']:
            return _export([], [], [], cost, valid=False,
                error={'phase': 'prior_pool', 'index': index, 'ref': ref, 'error': path['error']},
                pool_refs=refs, pool_log_likelihoods=np.asarray(ell), pool_size=pool_size)
        ell.append(path['log_likelihood']); pool_seeds.append(seeds)
        pool_initial_hashes.append(path['initial_fingerprint'])
        del path, initial
    ell = np.asarray(ell, float); pool_weights = np.exp(ell-logsumexp(ell)); pool_weights /= pool_weights.sum()
    selected, uniform = systematic_indices(pool_weights, rngs['resample'], count); cost['systematic_uniforms'] += 1
    paths = []
    for index in selected:
        path = load_path(refs[int(index)])
        if (path.get('schema') != PATH_SCHEMA or path.get('valid') is not True
                or path['log_likelihood'] != ell[int(index)]
                or path['initial_fingerprint'] != pool_initial_hashes[int(index)]
                or not np.array_equal(path['interval_seeds'], pool_seeds[int(index)])):
            raise ValueError('A selected full prior path has changed before compression.')
        paths.append(path)
    if cost['native_segment_completed'] != 3*pool_size or cost['native_segment_attempted'] != 3*pool_size:
        raise AssertionError('The full SIR prior pool must retain all3M native calls.')
    equal = np.full(count, 1./count)
    paths, selected_refs, informative, refresh_ledger = _finish_stock(
        config, factors, measured, rngs['stock'], paths, [refs[int(i)] for i in selected],
        equal, archive_callback, cost)
    return _export(paths, selected_refs, equal, cost, valid=True, pool_size=pool_size,
        pool_refs=refs, pool_weights=pool_weights, pool_log_likelihoods=ell,
        log_likelihoods_pool=ell.copy(), pool_interval_seeds=np.asarray(pool_seeds, dtype=np.uint32),
        pool_initial_fingerprints=np.asarray(pool_initial_hashes, dtype='U64'), selected_indices=selected,
        systematic_uniform=uniform, pool_ESS=float(1./(pool_weights@pool_weights)),
        pre_stock_informative_diversity=informative, stock_refresh_ledger=refresh_ledger,
        beta_schedule=np.asarray([0., 1.]), expected_native_prefix_upper_bound=3*pool_size,
        pool_support_retained_by_reference=True,
        compression_scope='FinalN equal MC slots with independentcommonstockrefresh;not exactpoolmasspreservation.')
