"""Independent algebra and tape checks for the staged AAm suffix smoother.

The numerical helpers operate on archived values. They neither import the
production smoother nor advance any stored native state.
"""

import argparse
from datetime import datetime
import hashlib
from itertools import product
from math import log
from pathlib import Path

import numpy as np
from scipy.stats import t
from threadpoolctl import threadpool_limits

from aam_formation_actual_rollout import packet_fingerprint
from aam_formation_cycle_data import check_actual_state
from aam_formation_cycle_driver import np_load, read, sha, write
from aam_formation_local_initial import draw_initial
from aam_formation_path_cycle import load_tree, stamp
from audit_aam_assay_guided_sir import _camera, _global
from audit_aam_camera_information_sandwich import independent_apertures


TEMPERATURES = (1./3., 2./3., 1.)
PARTICLES = 16
THIRD_NATIVE_BUDGET = 256


def likelihood_frames(predicted, measured, noise):
    """Recompute vector-Gaussian log factors in the declared camera coordinates."""
    p, x, s = (np.asarray(value, dtype=float) for value in (predicted, measured, noise))
    if (s.ndim != 1 or p.ndim < 1 or x.ndim < 1 or p.shape[-1] != len(s) or x.shape[-1] != len(s)
            or np.any(s <= 0.) or not all(np.isfinite(value).all() for value in (p, x, s))):
        raise ValueError('Camera values must be finite and match the registered positive noise.')
    residual = (p-x)/s
    ll = -.5*np.sum(residual*residual, axis=-1)
    if not np.isfinite(ll).all():
        raise ValueError('Camera likelihood is not finite.')
    return ll


def probabilities(logs):
    x = np.asarray(logs, float)
    if x.ndim < 1 or not x.shape[-1] or not np.isfinite(x).all():
        raise ValueError('Require at least one finite log weight per normalization.')
    relative = np.exp(x-np.max(x, axis=-1, keepdims=True))
    return relative/np.sum(relative, axis=-1, keepdims=True)


def group_log_values(tail_logs):
    """Stable log of the mean native likelihood within each prospective group."""
    x = np.asarray(tail_logs, float)
    if x.ndim != 2 or x.shape[1] not in (1, 4) or not np.isfinite(x).all() or np.any(x > 0.):
        raise ValueError('Require a finite nonpositive Gaussian log factor for every K=1 or K=4 tail.')
    m = np.max(x, axis=1)
    avg = m+np.log(np.exp(x-m[:, None]).mean(axis=1))
    if np.any(avg > 0.):
        raise ArithmeticError('A mean likelihood cannot exceed one.')
    return avg


def systematic_branches(weights, raw_u):
    """Raw U[0,1) jitter before division by the output population count."""
    w = np.asarray(weights, dtype=float)
    if w.ndim != 1 or not len(w) or np.any(w < 0.) or not np.isfinite(w).all() or not 0 <= raw_u < 1.:
        raise ValueError('Invalid systematic resampling distribution or ordinate.')
    if not np.isclose(w.sum(), 1., atol=2e-13, rtol=0.):
        raise ValueError('Systematic resampling expects normalized finite probabilities.')
    borders = np.cumsum(w)
    borders[-1] = 1.
    points = np.minimum((np.arange(len(w))+raw_u)/len(w), np.nextafter(1., 0.))
    return np.searchsorted(borders, points, side='right')


def independent_mh(previous, offered, beta, uniform):
    """Ordinary independence MH under the known finite second-camera parent law."""
    if (not np.isfinite([previous, offered, beta, uniform]).all() or not 0. < beta <= 1.
            or not 0. <= uniform < 1. or max(previous, offered) > 0.):
        raise ValueError('Invalid group likelihood, temperature or MH uniform.')
    threshold = min(0., beta*(offered-previous))
    return uniform == 0. or log(uniform) < threshold


def audit_group_tape(tail_logs, initialization, stages):
    """Replay one complete tempering arm with audit rows in sweep-major order.

    ``tail_logs`` uses chronological group birth ID. Each stage has scalar ``u``,
    ``offers``/``uniforms``/``accepted`` of shape (sweeps,16), and may retain
    ``resampled_positions`` (the previous particle *positions*) and
    ``final_ids`` (the actually retained group birth IDs). The arm cannot skip
    any native proposal draw after a rejection or amend its stored group.
    """
    logs = np.asarray(tail_logs, float)
    means = group_log_values(logs)
    k = logs.shape[1]
    if len(logs)*k != THIRD_NATIVE_BUDGET or len(stages) != len(TEMPERATURES):
        raise ValueError('Wrong fixed third-stage native budget or temperature count.')
    now = np.asarray(initialization).copy()
    np.testing.assert_array_equal(now, np.arange(PARTICLES))
    previous_beta, cursor = 0., PARTICLES
    stage_stats = []
    sweeps = 5 if k == 1 else 1
    for b, record in zip(TEMPERATURES, stages, strict=True):
        if 'beta' in record and float(record['beta']) != b:
            raise ValueError('Unexpected annealing temperature.')
        inc = probabilities((b-previous_beta)*means[now])
        order = systematic_branches(inc, record['u'])
        if 'initial_ids' in record:
            np.testing.assert_array_equal(now, record['initial_ids'])
        if 'weights' in record:
            np.testing.assert_allclose(inc, record['weights'], atol=2e-13, rtol=0.)
        if 'resampled_positions' in record:
            np.testing.assert_array_equal(order, record['resampled_positions'])
        now = now[order].copy()
        if 'resampled_ids' in record:
            np.testing.assert_array_equal(now, record['resampled_ids'])
        expected = np.arange(cursor, cursor+sweeps*PARTICLES).reshape(sweeps, PARTICLES)
        has_log = 'log_uniforms' in record
        draws = np.asarray(record['log_uniforms' if has_log else 'uniforms'])
        offers, accept = (np.asarray(record[name]) for name in ('offers', 'accepted'))
        np.testing.assert_array_equal(offers, expected)
        if draws.shape != offers.shape or accept.shape != offers.shape or accept.dtype.kind != 'b':
            raise ValueError('Every candidate needs one stored ordinary MH draw and bit.')
        actual = np.zeros_like(accept)
        for j, pos in np.ndindex(offers.shape):
            new = int(offers[j, pos])
            if 'previous_ids' in record and record['previous_ids'][j, pos] != now[pos]:
                raise ValueError('A rejected ancestor or an accepted whole group was not retained consistently.')
            if has_log:
                log_u = float(draws[j,pos])
                if not np.isfinite(log_u) or log_u > 0.:
                    raise ValueError('Require the original negative exponential MH innovation.')
                actual[j, pos] = log_u < min(0., b*(means[new]-means[now[pos]]))
            else:
                actual[j, pos] = independent_mh(means[now[pos]], means[new], b, float(draws[j, pos]))
            if actual[j, pos]:
                now[pos] = new
        np.testing.assert_array_equal(actual, accept)
        if 'final_ids' in record:
            np.testing.assert_array_equal(now, record['final_ids'])
        previous_beta, cursor = b, cursor+sweeps*PARTICLES
        stage_stats.append({'offspring_positions': order, 'incremental_ess': float(1./(inc@inc)),
                            'accepted_moves': int(np.count_nonzero(actual)), 'live_group_ids': now.copy()})
    if cursor != len(logs):
        raise ArithmeticError('The stored candidate tape and physical allocation differ.')
    return {'final_group_ids': now, 'groups_generated': cursor, 'native_third_calls': cursor*k,
            'stage_stats': stage_stats, 'flattened_tail_weights': terminal_tail_weights(logs, now)}


def terminal_tail_weights(logs, final_ids):
    """All retained group members, with equal group masses and conditional L3 weights."""
    ids, values = np.asarray(final_ids), np.asarray(logs, float)
    if (ids.shape != (PARTICLES,) or ids.dtype.kind not in 'ui'
            or np.any(ids < 0) or np.any(ids >= len(values))):
        raise ValueError('Require 16 final group instances, preserving repeated ancestry.')
    return probabilities(values[ids])/PARTICLES


def sir_tail_weights(logs):
    x = np.asarray(logs, float)
    if x.shape != (THIRD_NATIVE_BUDGET,) or np.any(x > 0.):
        raise ValueError('Use all 256 third-stage SIR candidates.')
    return probabilities(x)


def grouped_ess(parent_ids, flattened_weights, count=64):
    ids, w = np.asarray(parent_ids), np.asarray(flattened_weights, dtype=float)
    if (ids.dtype.kind not in 'ui' or ids.shape != w.shape or np.any(ids < 0) or np.any(ids >= count)
            or np.any(w < 0.) or not np.isfinite(w).all() or not np.isclose(w.sum(), 1., atol=2e-13, rtol=0.)):
        raise ValueError('Aggregate the true second-frame ancestry, not tail or slot identity.')
    mass = np.bincount(ids.ravel(), weights=w.ravel(), minlength=count)
    return {'mass': mass, 'weighted_ess': 1./(mass@mass), 'surviving_parent_count': int(np.count_nonzero(mass))}


def direct_energy(atoms, truth, weights, *, normalize=True):
    """Weighted V form with a direct pairwise distance implementation."""
    atom = np.asarray(atoms, dtype=float)
    w = np.asarray(weights, dtype=float).ravel()
    observe = np.asarray(truth, dtype=float)
    if atom.ndim < 2 or observe.shape != atom.shape[1:] or len(w) != len(atom):
        raise ValueError('Energy support and held outcome do not share endpoint dimensions.')
    if (not all(np.isfinite(x).all() for x in (atom, observe, w)) or np.any(w < 0.)
            or not np.isclose(w.sum(), 1., atol=2e-13, rtol=0.)):
        raise ValueError('Require a finite normalized empirical forecast and actual endpoint.')
    p, y = atom.reshape(len(atom), -1), observe.ravel()
    a = np.sqrt(np.sum((p-y)**2, axis=1))
    b = np.sqrt(np.sum((p[:, None]-p[None])**2, axis=2))
    value = np.sum(w*a)-.5*np.sum((w[:, None]*w[None])*b)
    return float(value/np.sqrt(len(y)) if normalize else value)


def paired_interval(delta):
    """Use independent mothers, with fixed tests made entirely within each one."""
    arr = np.asarray(delta, float)
    if arr.ndim != 1 or len(arr) < 2 or not np.isfinite(arr).all():
        raise ValueError('Need a finite contrast for each actual independent mother.')
    n = len(arr)
    effect = sum(arr)/n
    q = t.interval(.95, n-1)[1]
    error = q*np.sqrt(np.sum((arr-effect)**2)/(n*(n-1)))
    return {'mean': float(effect), 't95': [float(effect-error), float(effect+error)], 'n': n}


def finite_marginal_control():
    """Enumerate C2×K tails to check the exact endpoint target, not finite-SMC accuracy."""
    first = np.array([.3, .7])
    conditional = np.array([[.4, .1, .5], [.2, .6, .2]])
    evidence = np.array([[.2, .7, .02], [.05, .4, .9]])
    exact = first[:, None]*conditional*evidence
    partition = exact.sum()
    expected = exact/partition
    independent = {}
    for k in (1, 4):
        calculated, uncorrected = np.zeros_like(exact), np.zeros_like(exact)
        mass = 0.
        for i, heads in product(range(len(first)), product(range(3), repeat=k)):
            heads = np.asarray(heads)
            proposal = first[i]*np.prod(conditional[i, heads])
            third = evidence[i, heads]
            lifted = proposal*np.mean(third)
            mass += lifted
            np.add.at(calculated[i], heads, lifted*(third/third.sum()))
            np.add.at(uncorrected[i], heads, lifted/k)
        np.testing.assert_allclose(mass, partition, atol=2e-15, rtol=0.)
        np.testing.assert_allclose(calculated/mass, expected, atol=2e-15, rtol=0.)
        independent[str(k)] = bool(np.max(np.abs(uncorrected/mass-expected)) < 2e-15)
    if independent != {'1': True, '4': False}:
        raise ArithmeticError('Uniform tail picking should fail the K=4 marginal counterexample.')
    return {'known_normalizer': float(partition), 'both_exact_weighted_marginals': True,
            'uniform_tail_selection_pass': independent}


def _require(ok, message):
    if not ok:
        raise ValueError(message)


def _same(x, y):
    np.testing.assert_array_equal(x, y)


def _close(x, y):
    np.testing.assert_allclose(x, y, atol=2e-12, rtol=2e-12)


def _file(path, ref):
    _require(Path(ref['path']).resolve() == path and path.stat().st_size == ref['bytes']
             and sha(path) == ref['sha256'], f'Artifact identity differs: {path}')
    return path


def _rng(plan, role, unit, *address):
    return np.random.default_rng(np.random.SeedSequence([*plan['entropy'], plan['roles'][role], unit, *address]))


def _position(weights, uniform):
    cdf = np.cumsum(weights)
    cdf[-1] = 1.
    return int(np.count_nonzero(cdf <= uniform))


def _initial(plan, unit, role, native, archived, *address):
    value = draw_initial(plan['physics'], _rng(plan, role, unit, *address))
    _require(set(value) == set(archived) and value['I'] == int(native[0].counts[3]),
             'Original initial variables or initiator budget changed.')
    for key in value:
        _same(archived[key], value[key])
    _same(native[0].points, value['points'])
    _same(native[0].unwrapped, value['points'])
    _require(not np.any(native[0].counts[:3]), 'An inference start already contains future work.')


def _state(packet, cfg, age):
    state, material, geometry, rates, factors = packet
    _same(material, [cfg['recipe'][key] for key in ('aam','mbaa','initial_mbaa','feed_time')])
    _same(geometry, [*(cfg['motion'][key] for key in ('step','diffusion','box','spring','rest','repulsion','core','capture','inner')),
                     cfg['corridor']])
    _same(rates, cfg['rates'])
    _same(factors, cfg['factors'])
    return check_actual_state(state, expected_age=age)['groups']


def _interval(plan, unit, role, camera, address, before, after, ledger):
    cfg = plan['physics']
    lo, end = cfg['prefix_clock'][camera:camera+2]
    knots = [lo]
    while knots[-1] < end:
        knots.append(min(knots[-1]+cfg['motion']['step'], end))
    dimensions = (len(knots)-1, 87)
    waiting = _rng(plan, role, unit, camera, *address, 1).exponential(size=dimensions)
    mark = _rng(plan, role, unit, camera, *address, 2).integers(0, 2**53, size=dimensions, dtype=np.int64)
    s0, s1 = before[0], after[0]
    _require(s0.position[0] == lo and s1.position[0] == end and not(s0.counts[6] or s1.counts[6])
             and ledger['native_calls'] == 1 and ledger['log_ratio'] == 0.
             and not ledger['tape']['forced_mark_consumed'] and np.isfinite(ledger['seconds'])
             and ledger['seconds'] >= 0., 'Wrong native boundary, count, or unforced base law.')
    ev, ed = map(int, s0.counts[:2])
    _same(s0.events[:ev], s1.events[:ev])
    _same(s0.edges[:ed], s1.edges[:ed])
    _same(s0.lifts[:ed], s1.lifts[:ed])
    events = iter(s1.events[ev:int(s1.counts[0]), 0])
    spans, counts = iter(ledger['spans']), []
    for slab, (a, b) in enumerate(zip(knots[:-1], knots[1:], strict=True)):
        i, now = 0, a
        while True:
            recorded = next(spans)
            rate = recorded['target_A']
            _require(0 <= i < 87 and recorded['slab'] == slab and recorded['attempt'] == i
                     and recorded['begin'] == now and np.isfinite(rate) and rate >= 0.,
                     'Addressed reaction clock or primitive hazard domain changed.')
            if rate:
                _same(waiting[slab, i], recorded['exponential'])
                fire = now+waiting[slab, i]/rate
            else:
                _require(recorded['exponential'] is None, 'Zero hazard consumed a reaction-clock draw.')
                fire = np.inf
            _same(fire >= b, recorded['censored'])
            if fire >= b:
                _same(recorded['end'], b)
                counts.append(i+int(rate > 0.))
                break
            _same(recorded['uniform'], mark[slab, i]/2**53)
            _same(recorded['end'], next(events))
            _same(recorded['geometry_time'], a+.5*(b-a))
            _require(recorded['mark_tilt'] is None and int(recorded['chosen']) >= 0,
                     'An original unforced mark was replaced.')
            _close(recorded['end'], fire)
            now, i = recorded['end'], i+1
    _require(next(spans, None) is None and next(events, None) is None, 'Unused or missing original tape records.')
    _same(counts, ledger['tape']['waits_per_slab'])


def _edges(packet):
    state = packet[0]
    return {min(i, j)*len(state.points)+max(i, j) for i, j in state.edges[:int(state.counts[1])]}


def _metric_frame(packet, cfg, camera):
    groups = _state(packet, cfg, cfg['prefix_clock'][camera+1])
    return independent_apertures(packet), np.asarray(_global(packet[0], groups))


def _features(packets, cfg):
    local, global_rows = zip(*(_metric_frame(packet, cfg, i) for i, packet in enumerate(packets[1:])), strict=True)
    return {'local': np.stack(local), 'global_early': np.stack(global_rows),
            'new_edges': sorted(_edges(packets[3])-_edges(packets[2]))}


def _validate_features(recorded, recomputed):
    for key in ('local', 'global_early'):
        _close(recorded[key], recomputed[key])
    _same(recorded['new_edges'], recomputed['new_edges'])


def _true_mother(out, plan, group, record, unit):
    cfg = plan['physics']
    _require(record['unit'] == unit, 'Training/held mother ID unexpectedly changed.')
    folder = out/group
    query = np_load(_file(folder/f'{unit:03d}_QUERY.npz', record['query']))
    _require(set(query) == {'measured'} and query['measured'].shape == (3,4), 'A blind query contained truth.')
    raw = load_tree(_file(folder/f'{unit:03d}_RAW.npz', record['raw']))
    packets = raw['packets']
    _require(len(packets) == 4 and len(raw['intervals']) == 3
             and raw['calls'] == {'attempted':3,'completed':3,'failed':0}, 'Incomplete actual prefix or budget.')
    _initial(plan, unit, 'data_initial', packets[0], raw['initial'])
    _state(packets[0], cfg, 0.)
    for c, ledger in enumerate(raw['intervals']):
        _interval(plan, unit, 'data_tape', c, (), packets[c], packets[c+1], ledger)
    calibrated = np.stack([_camera(p) for p in packets[1:]])
    _close(raw['clean'], calibrated)
    _same(query['measured'], raw['clean'] + np.asarray(cfg['noise'])*_rng(plan, 'measurement', unit).standard_normal((3,4)))
    return query['measured'], _features(packets, cfg)


def _early(plan, unit, y, info):
    cfg = plan['physics']
    bank = info['bank']
    histories = [(p,) for p in bank['packets0']]
    _require(len(histories) == len(bank['initial']) == 64 and len(bank['frames']) == 2,
             'Must retain the same 64 original first-two-camera trajectories for all arms.')
    for i, p in enumerate(bank['packets0']):
        _initial(plan, unit, 'initial', p, bank['initial'][i], i)
        _state(p, cfg, 0.)
    for c, f in enumerate(bank['frames']):
        proposed = f['packets']
        for i, ((p, ledger), old) in enumerate(zip(zip(proposed, f['receipts'], strict=True), histories, strict=True)):
            _interval(plan, unit, 'early_tape', c, (i,), old[-1], p, ledger)
        clean = np.stack([_camera(p) for p in proposed])
        _close(clean, f['clean'])
        w = probabilities(likelihood_frames(f['clean'], y[c], cfg['noise']))
        _close(w, f['weights'])
        histories = [h+(p,) for h, p in zip(histories, proposed, strict=True)]
        if c == 0:
            u = _rng(plan, 'early_resample', unit).random()
            _same(u, f['uniform'])
            ids = systematic_branches(w, u)
            _same(ids, f['parents'])
            histories = [histories[i] for i in ids]
        else:
            _require(f['uniform'] is None and f['parents'] is None,
                     'Second-frame exact parental law must remain weighted before resampling.')
    _same([[packet_fingerprint(p) for p in prior] for prior in histories], info['history_fingerprints'])
    # Reuse archived production-normalized weights for exact CDF boundary replay.
    # They were independently verified above to tighter-than-score tolerance.
    return histories, np.asarray(bank['frames'][1]['weights'])


def _arm(plan, unit, y, hist, w, arm, raw):
    cfg = plan['physics']
    k, sweeps, code = plan['arms'][arm]
    trials = raw['trials']
    _require((k, sweeps) in ((1,5),(4,1)) and len(trials)*k == 256, 'Wrong allocated native third budget.')
    logs, seconds = [], []
    for g, trial in enumerate(trials):
        u = _rng(plan, 'parent', unit, code, g).random()
        parent = _position(w, u)
        _same([g, parent], [trial['group'], trial['parent']])
        _same(u, trial['parent_uniform'])
        _require(len(trial['tails']) == len(trial['receipts']) == k, 'Incomplete native suffix group.')
        for j, (packet, row) in enumerate(zip(trial['tails'], trial['receipts'], strict=True)):
            _interval(plan, unit, 'suffix_tape', 2, (code,g,j), hist[parent][-1], packet, row)
            _state(packet, cfg, cfg['prefix_clock'][-1])
            seconds.append(row['seconds'])
        _close(np.stack([_camera(p) for p in trial['tails']]), trial['clean'])
        ll = likelihood_frames(trial['clean'], y[2], cfg['noise'])
        _close(ll, trial['log_like'])
        logs.append(ll)
    loggroup = group_log_values(logs)
    _close(loggroup, [x['log_mean'] for x in trials])
    stage_rows = []
    for n, stage in enumerate(raw['stages']):
        u = _rng(plan, 'temper_resample', unit, code, n).random()
        _same(u, stage['uniform'])
        arrays = {name: np.zeros((sweeps, 16), bool if name == 'accepted' else
                    float if name == 'log_uniforms' else np.int64)
                  for name in ('previous_ids','offers','log_uniforms','accepted')}
        _require(len(stage['moves']) == sweeps*16, 'A rejuvenation chain omitted original candidates.')
        for move, (s, j) in zip(stage['moves'], np.ndindex(sweeps,16), strict=True):
            sweep, slot, prior, candidate, log_u, accepted = move
            _same([sweep,slot], [s,j])
            _same(log_u, -_rng(plan, 'mh', unit, code, n, s, j).exponential())
            for key, value in zip(arrays, (prior,candidate,log_u,accepted), strict=True):
                arrays[key][s,j] = value
        stage_rows.append({**arrays, 'u':u,'beta':stage['beta'],'weights':stage['weights'],
             'initial_ids':stage['initial'],'resampled_positions':stage['parents'],'final_ids':stage['terminal']})
    rep = audit_group_tape(logs, np.arange(16), stage_rows)
    _same(rep['final_group_ids'], raw['final_groups'])
    _close(sum(seconds), raw['native_seconds'])
    _require(np.isfinite(raw['operator_seconds']) and raw['operator_seconds'] >= raw['native_seconds'] >= 0.,
             'The operator clock excluded some registered orbit work.')
    return rep, raw['native_seconds'], raw['operator_seconds']


def _forecast(plan, unit, hist, name, pred, raw, report, cache):
    trials = raw['trials']
    chosen, expected = [], {}
    if name == 'sir256':
        expected = dict(zip(((g,0) for g in range(256)), sir_tail_weights([g['log_like'][0] for g in trials]), strict=True))
    else:
        group_ids = report['final_group_ids']
        k, _, code = plan['arms'][name]
        weights = report['flattened_tail_weights']
        for slot, g in enumerate(group_ids):
            u = _rng(plan, 'one_tail_readout', unit, code, slot).random()
            chosen.append((g, _position(probabilities(trials[g]['log_like']), u)))
            for j in range(k):
                expected[g,j] = expected.get((g,j), 0.) + weights[slot,j]
    refs = sorted(expected)
    _same(refs, pred['refs'])
    _same(chosen, pred['one_tail_refs'])
    _close([expected[key] for key in refs], pred['weights'])
    ids = [trials[g]['parent'] for g, _ in refs]
    _same(ids, pred['parent'])
    for (g,j), recorded in zip(refs, pred['features'], strict=True):
        key = ('single' if name == 'sir256' else name, g, j)
        if key not in cache:
            old = hist[trials[g]['parent']]
            tail = trials[g]['tails'][j]
            prefix_key = ('early', trials[g]['parent'])
            if prefix_key not in cache:
                cache[prefix_key] = [_metric_frame(p, plan['physics'], c) for c,p in enumerate(old[1:])]
            local, global_rows = zip(*(*cache[prefix_key], _metric_frame(tail, plan['physics'], 2)), strict=True)
            cache[key] = {'local': np.stack(local), 'global_early': np.stack(global_rows),
                          'new_edges': sorted(_edges(tail)-_edges(old[-1]))}
        _validate_features(recorded, cache[key])
    return grouped_ess(np.asarray(ids), pred['weights'])


def _scores(rows, w, target, scale):
    local = np.stack([a['local'] for a in rows])
    g = np.stack([a['global_early'] for a in rows])
    y = target['local']
    truth = set(target['new_edges'])
    all_edges = set().union(truth, *(set(a['new_edges']) for a in rows))
    edge_loss = sum((sum(p for p, a in zip(w,rows,strict=True) if label in a['new_edges'])-(label in truth))**2
                    for label in all_edges)
    return {'hidden_t3_6': direct_energy(local[:,2], y[2], w),
            'hidden_delta32_6': direct_energy(local[:,2]-local[:,1], y[2]-y[1], w),
            'early18': direct_energy(local, y, w), 'hidden_t2_6': direct_energy(local[:,1], y[1], w),
            'global24': direct_energy(g/scale, target['global_early']/scale, w), 'new_edge_brier': edge_loss}


def _phase(out):
    names = ('REGISTERED.json','RUN.json','ACQUISITION_STARTED.json','ACQUIRED.json',
             'PREDICTION_STARTED.json','PREDICTIONS_FROZEN.json','REVEAL_STARTED.json','SCORED.json')
    paths = [out/name for name in names]
    p, run, acquire_started, acquire, predict_started, frozen, reveal, score = rows = [read(x) for x in paths]
    dates = [datetime.fromisoformat(x['actual_utc']) for x in rows]
    reg_hash, acq_hash, freeze_hash = [sha(paths[k]) for k in (0,3,5)]
    ns = 'aam-shared-early-bank-grouped-native-suffix-smoother-v1'
    signature = hashlib.sha256(ns.encode()).digest()[:16]
    entropy = [int.from_bytes(signature[k:k+4], 'little') for k in range(0,16,4)]
    roles = ('data_initial','data_tape','measurement','initial','early_tape','early_resample',
             'suffix_tape','parent','temper_resample','mh','one_tail_readout')
    _require(dates == sorted(dates) and p['namespace'] == ns and p['entropy'] == entropy
             and p['roles'] == {name:k+1 for k,name in enumerate(roles)}
             and p['arms'] == {'single':[1,5,0],'grouped':[4,1,1]}
             and tuple(p['betas']) == TEMPERATURES and p['stage_particles'] == 16
             and (p['training_mothers'], p['held_mothers'], p['early_particles']) == (64,48,64)
             and run['registered_before_native'] and predict_started['query_only']
             and frozen['opened_held_targets'] is False
             and reg_hash == run['registration_sha256'] == acquire_started['registration_sha256'] == acquire['registration_sha256']
             == frozen['registration_sha256'] == score['registration_sha256']
             and acq_hash == predict_started['acquisition_sha256'] == frozen['acquisition_sha256']
             and freeze_hash == reveal['freeze_sha256'] == score['freeze_sha256'],
             'Blind role addressing or registration-to-reveal identity changed.')
    for name, expected in p['sources'].items():
        _require(sha(out/'SOURCE_SNAPSHOT'/Path(name).name) == expected, f'Frozen source differs: {name}')
    for name, expected in p['inputs'].items():
        _require(sha(name) == expected, f'Frozen input identity differs: {name}')
    _require(list(p['methods']) == ['grouped','single','sir256','knn8','empirical']
             and p['budget'] == {'shared_early_eaches':128,'third_each':256,'inference_actual':30720,'simulation_data':336}
             and score['actual_inference_native_calls'] == 30720 and acquire['native_calls'] == score['native_acquisition'] == 336,
             'Changed target/comparator or native budget.')
    return p, acquire, frozen, score


def audit(out):
    plan, acq, f, scored = _phase(out)
    partition = finite_marginal_control()
    archive = load_tree(_file(out/'TRAIN_BANK.npz', acq['train_bank']))
    _require(len(acq['groups']['training']) == len(archive['features']) == 64 and
             len(acq['groups']['held']) == len(scored['rows']) == len(f['predictions']) == 48,
             'Incomplete prospective sample or finite training bank.')
    Y, bank = [], []
    for unit, ref in enumerate(acq['groups']['training']):
        measured, features = _true_mother(out, plan, 'training', ref, unit)
        _validate_features(archive['features'][unit], features)
        Y.append(measured)
        bank.append(features)
    Y = np.stack(Y)
    scale = np.maximum(np.stack([x['global_early'] for x in bank]).reshape(-1,8).std(axis=0), .05)
    xscale = np.maximum(Y.std(axis=0), plan['physics']['noise'])
    _same(Y, archive['noisy'])
    _close(scale, archive['global_scale'])
    _close(xscale, archive['xscale'])
    neighbors = np_load(_file(out/'KNN_FROZEN.npz', f['knn']))['index']
    answer, ess_records, costs = [], [], []
    for j, (actual_ref, pred_ref) in enumerate(zip(acq['groups']['held'], f['predictions'], strict=True)):
        unit = 64+j
        _require(pred_ref['unit'] == actual_ref['unit'] == unit and
                 pred_ref['query_sha256'] == actual_ref['query']['sha256'] and
                 pred_ref['count'] == {'attempted':640,'completed':640,'failed':0} and
                 pred_ref['order'] == (['single','grouped'] if unit%2 == 0 else ['grouped','single']),
                 'Fresh held mother, query-only binding or timing order changed.')
        y, actual = _true_mother(out, plan, 'held', actual_ref, unit)
        folder = out/'predictions'/f'{unit:03d}'
        earlier = load_tree(_file(folder/'EARLY.npz', pred_ref['early']))
        hist,w = _early(plan, unit, y, earlier)
        fore = load_tree(_file(folder/'PREDICTIONS.npz', pred_ref['forecast']))
        _require(set(fore) == {'grouped','single','sir256'}, 'Frozen prediction added a different route.')
        cache, stats, cost = {}, {}, {}
        for kind in ('single','grouped'):
            raw = load_tree(_file(folder/f'{kind}_RAW.npz', pred_ref['raw'][kind]))
            verified, native, op = _arm(plan, unit, y, hist, w, kind, raw)
            stats[kind] = _forecast(plan, unit, hist, kind, fore[kind], raw, verified, cache)
            cost[kind] = {'native':native,'operator':op,'native_calls':verified['native_third_calls']}
            if kind == 'single':
                stats['sir256'] = _forecast(plan, unit, hist, 'sir256', fore['sir256'], raw, verified, cache)
        dd = np.sum(((Y-y)/xscale)**2, axis=(1,2))
        nn = np.lexsort((np.arange(64),dd))[:8]
        _same(nn, neighbors[j])
        for method,i in [('knn8',nn),('empirical',np.arange(64))]:
            fore[method] = {'features':[bank[k] for k in i], 'weights':np.full(len(i),1./len(i))}
        outcome = {name:_scores(pred['features'], pred['weights'], actual, scale) for name,pred in fore.items()}
        for key,item in outcome.items():
            _require(item.keys()==scored['rows'][j][key].keys(), 'One frozen endpoint is missing.')
            _close(list(item.values()), [scored['rows'][j][key][m] for m in item])
        answer.append(outcome)
        ess_records.append(stats)
        costs.append(cost)
    passed = []
    for route in plan['methods']:
        for metric in ('hidden_t3_6',*plan['secondary']):
            own = np.asarray([a[route][metric] for a in answer])
            _close(own.mean(), scored['means'][route][metric])
            if route != 'grouped':
                other = np.asarray([a['grouped'][metric] for a in answer])
                span = paired_interval(own-other)
                registered = scored['grouped_gain_over'][route][metric]
                _close([span['mean'],*span['t95']], [registered['mean'],*registered['pointwise_t95']])
                _require(registered['mothers'] == 48, 'Mothers are the independent unit.')
                if metric == 'hidden_t3_6':
                    passed.append(span['t95'][0] > 0.)
    _require(len(passed) == 4 and all(passed) == scored['primary_gate'], 'Primary gate changed during scoring.')
    ess = {name:{'mean':float(np.mean([row[name]['weighted_ess'] for row in ess_records])),
                 'mean_surviving_parents':float(np.mean([row[name]['surviving_parent_count'] for row in ess_records]))}
           for name in ('single','grouped','sir256')}
    time = {arm:{role:float(sum(item[arm][role] for item in costs)) for role in ('native','operator')}
            for arm in ('single','grouped')}
    native_counts = [row[arm]['native_calls'] for row in costs for arm in ('single','grouped')]
    _require(len(native_counts)==96 and all(x==256 for x in native_counts), 'Third physical budget changed.')
    write(out/'INDEPENDENT_AUDIT.json', {**stamp(), 'passed': True, 'scored_sha256': sha(out/'SCORED.json'),
          'registered_sha256': sha(out/'REGISTERED.json'), 'held_mothers':48,'training_mothers':64,
          'independent_legal_original_intervals_checked':30720+336,'new_native_calls':0,
          'table_scope':'Original wait and mark tapes, full native state validity and append-only parent ancestry checked. No integration or verification of unarchived Brownian increments.',
          'finite_exact_marginal_check':partition,'each_third_charged_native_calls':256,
          'ancestral_ess':ess,'arm_native_and_solver_times_seconds':time,
          'timing_scope':'Registered third solver encompasses native calls, cloning/readouts and resample-moves. It excludes final forecast formatting. SIR reuses the complete K1 tape; its independent wall time is unknown.',
          'primary_gate_verified':scored['primary_gate']})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        print(finite_marginal_control())
    elif args.out:
        with threadpool_limits(limits=1):
            audit(args.out.resolve(strict=True))
    else:
        parser.error('Pass --self-test or an already scored output folder.')
