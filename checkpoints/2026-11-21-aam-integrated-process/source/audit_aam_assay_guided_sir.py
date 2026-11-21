"""Independently read back the registered assay-guided prefix experiment."""
import argparse
from datetime import datetime
import hashlib
from pathlib import Path

import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import t
from threadpoolctl import threadpool_limits

from aam_formation_cycle_data import check_actual_state
from aam_formation_cycle_driver import np_load, read, write
from aam_formation_local_data import load_prefix, load_query
from aam_formation_local_initial import draw_initial, initial_fingerprint
from aam_formation_path_cycle import load_tree, stamp
from audit_aam_camera_information_sandwich import close, history, require, same
from package_aam_formation_local_cycle import sha


NAMESPACE = 'aam-guided-prior-v5-32-mothers-128-path-importance-v1'
N_MOTHERS, N_ROUTES, N_TRAIN = 32, 128, 64
MAX_BLOCKS, BLOCK_SIZE = 16, 32
GRID = 2**53
ARMS = ('prior', 'guided')
ENTROPY = [int.from_bytes(hashlib.sha256(NAMESPACE.encode()).digest()[i:i+4], 'little') for i in range(0, 16, 4)]


def _rng(mother, role, *address):
    return np.random.default_rng(np.random.SeedSequence(ENTROPY + [mother, role, *address]))


def _artifact(receipt, expected):
    require(Path(receipt['path']).resolve() == expected and expected.stat().st_size == receipt['bytes']
            and sha(expected) == receipt['sha256'], f'Frozen bytes or path changed: {expected}')
    return expected


def _chronology(out):
    files = [out/name for name in ('REGISTERED.json', 'RUN.json', 'ACQUIRED.json', 'PREDICTION_STARTED.json',
                                   'PREDICTIONS_FROZEN.json', 'SUMMARY.json')]
    plan, run, acquired, start, freeze, summary = records = [read(path) for path in files]
    dates = [datetime.fromisoformat(record['actual_utc']) for record in records]
    require(dates == sorted(dates) and run['registered_before_native_trials'] is True
            and start['observation_only'] is True and freeze['hidden_targets_seen'] is False,
            'The registered order or blinded forecast receipt changed.')
    registration, acquisition, prediction = (sha(files[i]) for i in (0, 2, 4))
    require(registration == run['registration_sha256'] == freeze['registration_sha256'] == summary['registration_sha256']
            and acquisition == start['acquisition_sha256'] == freeze['acquisition_sha256']
            and prediction == summary['freeze_sha256'], 'Registration, acquisition, forecast or score hash changed.')
    require(plan['namespace'] == NAMESPACE and plan['entropy'] == ENTROPY and plan['mothers'] == N_MOTHERS
            and len(acquired['mothers']) == len(freeze['mothers']) == len(summary['rows']) == summary['mothers'] == N_MOTHERS
            and plan['native_candidate_routes_per_arm'] == N_ROUTES and plan['guide_max_cheap_blocks'] == MAX_BLOCKS
            and plan['v5_training_only'] is True and plan['observation_views_unchanged'] is True
            and plan['new_fits'] == summary['new_fits'] == plan['wet_measurements'] == summary['new_wet_measurements'] == 0,
            'Cohorts, allocation, observation law, or experimental status changed.')
    for name, frozen in plan['sources'].items():
        require(sha(name) == sha(out/'SOURCE_SNAPSHOT'/Path(name).name) == frozen, 'A registered source changed: '+name)
    for name, frozen in plan['inputs'].items():
        require(sha(name) == frozen, 'A registered input changed: '+name)
    same(plan['teacher_factors'], plan['physics']['factors'])
    return plan, acquired, freeze, summary


def _camera(packet):
    state, material, geometry, *_ = packet
    aam, loaded = int(material[0]), int(material[2])
    rows = []
    for x in (1.45, 4.45):
        delta = (state.points[:, :2]-[x, 1.35] + geometry[2]/2.) % geometry[2] - geometry[2]/2.
        w = np.clip(1.-np.abs(delta)/1.3, 0., 1.).prod(axis=1)
        rows.extend((np.dot(w[:aam], state.owner[:2*aam:2] >= 0)/max(w[:aam].sum(), 1e-12),
                     6.*np.dot(w[aam:], state.present[aam:])/loaded))
    return rows


def _global(state, groups):
    count = np.asarray(state.owner).reshape(78, 2)
    lengths = np.bincount(count[:64,0][count[:64,0] >= 0]).astype(float)
    lengths = lengths[lengths > 0.]
    ln, lw = (lengths.mean()/64., np.dot(lengths,lengths)/lengths.sum()/64.) if len(lengths) else (0., 0.)
    present = state.present
    reacted = count.max(axis=1) >= 0
    largest = max(np.bincount(groups[reacted & present]), default=0)/78.
    cycle = (int(state.counts[1])-int(present.sum())+len(set(groups[present])))/14.
    tags = state.events[:int(state.counts[0]),1].astype(int)
    tag_counts = np.bincount(tags,minlength=10)
    pendant = np.count_nonzero(present[64:] & ((count[64:] >= 0).sum(axis=1) == 1))/14.
    return (ln, lw, largest, cycle, *(tag_counts[[5, 6, 9]]/14.), pendant)


def _snapshots(path, plan):
    cfg = plan['physics']
    packets = path['checkpoint_packets']
    require(len(packets) == 4, 'A snapshot path is incomplete.')
    global_history = []
    for age, packet in zip(cfg['prefix_clock'], packets, strict=True):
        state, material, geometry, rates, factors = packet
        graph = check_actual_state(state, expected_age=age)
        if age:
            global_history.append(_global(state, graph['groups']))
        same(material, [cfg['recipe'][key] for key in ('aam', 'mbaa', 'initial_mbaa', 'feed_time')])
        same(geometry, [*(cfg['motion'][key] for key in ('step','diffusion','box','spring','rest','repulsion','core','capture','inner')),
                       cfg['corridor']])
        same(rates, cfg['rates'])
        same(factors, plan['teacher_factors'])
    cameras = np.asarray([_camera(packet) for packet in packets[1:]])
    close(path['clean'], cameras)
    return cameras, history(packets), np.asarray(global_history)


def _route(path, initial, seeds, identity, plan):
    require(path['valid'] is True and path['status'] == 'complete' and path['error'] is None
            and path['mother_id'] == identity and path['cost']['actual_initializations'] == 1,
            'Invalid or mismatched actual route.')
    require(path['initial_fingerprint'] == initial_fingerprint(initial) == initial_fingerprint(path['initial_variables']),
            'The recorded physical initialization differs from its only registered prior draw.')
    same(path['explicit_native_seeds'], seeds)
    same(path['factors'], plan['teacher_factors'])
    same([path['cost'][key] for key in ('native_segment_attempted','native_segment_completed','native_segment_failed')], [3,3,0])
    packets = path['checkpoint_packets']
    state = packets[0][0]
    same(state.points, initial['points'])
    same(state.unwrapped, initial['points'])
    require(state.counts[3] == initial['I'] and not state.counts[0] and not state.counts[1],
            'The archived route does not start at the registered initial State.')
    cfg = plan['physics']
    for index, (row, seed) in enumerate(zip(path['native_calls'], seeds, strict=True)):
        require(row['camera'] == index and row['seed'] == int(seed) and row['completed'] is True,
                'An original native interval, role or seed is missing.')
        same(row['clock'], cfg['prefix_clock'][:index+2])
        same(row['position_before'], packets[index][0].position)
        same(row['position_after'], packets[index+1][0].position)
        previous, current = packets[index][0], packets[index+1][0]
        count_events, count_edges = map(int, previous.counts[:2])
        same(current.events[:count_events], previous.events[:count_events])
        same(current.edges[:count_edges], previous.edges[:count_edges])
        same(current.lifts[:count_edges], previous.lifts[:count_edges])
    return _snapshots(path, plan)


def _integer_threshold(initial, proxy, y):
    # Arithmetic is kept in the registered implementation's float64 order
    # before rounding to the grid. Comparing only probabilities could miss a
    # one-integer discrepancy at a cheap Bernoulli threshold.
    xy = initial['points'][:72,:2]
    dx = (xy[:,0,None] - np.array([1.45,4.45]) + 3.) % 6. - 3.
    dy = (xy[:,1] - 1.35 + 3.) % 6. - 3.
    x_corridor = np.clip(1.-np.abs(dx)/1.3, 0., 1.)
    xy_aperture = x_corridor * np.clip(1.-np.abs(dy)/1.3, 0., 1.)[:,None]
    stats = np.concatenate(([initial['I'], np.mean(initial['left'])], xy_aperture[:64].sum(axis=0),
                            xy_aperture[64:].sum(axis=0), x_corridor[64:].sum(axis=0)))
    predicted = ((stats - proxy['center'])/proxy['scale']) @ proxy['coefficients'] + proxy['mean']
    residual = (predicted-y.reshape(12))/proxy['proxy_sigma']
    log_proxy = -.5*np.square(residual).sum()/3.
    threshold = int(np.ceil(GRID*(1./8. + (7./8.)*np.exp(log_proxy))))
    require(GRID//8 <= threshold <= GRID, 'An independently rebuilt Bernoulli mass lost support.')
    return threshold


def _guide_entry(cfg, proxy, y, mother, slot, attempts):
    require(1 <= len(attempts) <= MAX_BLOCKS, 'A guide route lacks the capped cheap-block ledger.')
    draws = 0
    for block, saved in enumerate(attempts):
        stream = _rng(mother, 4, slot, block)
        accepted, rows = None, []
        for _ in range(BLOCK_SIZE):
            candidate = draw_initial(cfg, stream)
            threshold = _integer_threshold(candidate, proxy, y)
            uniform = int(stream.integers(GRID, dtype=np.int64))
            chosen = uniform < threshold
            rows.append({'threshold':threshold,'uniform53':uniform,'chosen':chosen})
            if chosen:
                accepted = candidate
                break
        success = accepted is not None
        require(success == (block == len(attempts)-1), 'First successful cheap block was skipped or absent.')
        mass = (threshold/GRID) if success else None
        expected = {'status':'success' if success else 'exhausted', 'attempts':len(rows),
                    'candidate_threshold':threshold if success else None,'candidate_mass':mass,'cheap_draws':rows}
        require(saved == expected, 'Integer acceptance evidence or denominator changed.')
        draws += len(rows)
    return accepted, mass, draws


def _weighted_energy(atoms, weights, truth):
    x, y = atoms.reshape(len(atoms), -1), truth.reshape(1,-1)
    distances = cdist(x, x)
    contribution = np.sum(weights*cdist(x,y)[:,0])
    return (contribution-.5*np.sum(np.outer(weights,weights)*distances))/np.sqrt(x.shape[-1])


def _training(plan):
    folder = Path(plan['v5'])
    frozen = load_tree(folder/'MICRO_TRAINING_BANK.npz')
    feature, global_history, observed = [], [], []
    for mother in range(N_TRAIN):
        raw = load_prefix(folder, mother, group='train')
        same(raw['factors'], plan['teacher_factors'])
        _, local, global_values = _snapshots(raw, plan)
        feature.append(local)
        global_history.append(global_values)
        observed.append(load_query(folder, mother, group='train')['measured'])
    close(global_history, frozen['early'])
    scale = np.asarray(frozen['scale'])
    xscale = np_load(folder/'BASELINES'/'FIT_ARRAYS.npz')['knn_Xscale'].reshape(3, 4)
    require(scale.shape == (8,) and np.all(np.isfinite(scale) & (scale > 0.))
            and np.all(np.isfinite(xscale) & (xscale > 0.)), 'An original fixed metric scale is not valid.')
    return np.asarray(feature), np.asarray(global_history)/scale, np.asarray(observed), scale, xscale


def audit(out):
    plan, acquired, frozen, summary = _chronology(out)
    proxy = np_load(plan['proxy_file'])
    cfg = plan['physics']
    train, train_global, train_y, scale, xscale = _training(plan)
    rows, cheap_draws, native = [], 0, dict.fromkeys(ARMS, 0)
    for mother, (held, pred) in enumerate(zip(acquired['mothers'], frozen['mothers'], strict=True)):
        folder = out/f'mother_{mother:03d}'
        require(held['mother'] == pred['mother'] == mother and held['query']['sha256'] == pred['query_sha256']
                and set(pred['arms']) == set(ARMS), 'The original acquired query is no longer the frozen input.')
        query = np_load(_artifact(held['query'], folder/'QUERY.npz'))
        require(set(query) == {'measured'} and query['measured'].shape == (3,4), 'A predictor input is not an observation-only camera.')
        y = query['measured']
        raw = load_tree(_artifact(held['truth'], folder/'SEALED_TRUTH.npz'))
        truth_seed = _rng(mother,2).integers(0,2**32,3,dtype=np.uint32)
        truth_camera, truth_feature, truth_global = _route(raw, draw_initial(cfg,_rng(mother,1)), truth_seed,
                                                           f'held:{mother}', plan)
        close(y, truth_camera + _rng(mother,3).standard_normal((3,4))*cfg['noise'])
        values = {}
        for arm in ARMS:
            info = pred['arms'][arm]
            raw = load_tree(_artifact(info['raw'], folder/f'{arm}_RAW.npz'))
            guide = load_tree(_artifact(info['guide_ledger'], folder/f'{arm}_GUIDE.npz'))
            forecast = np_load(_artifact(info['forecast'], folder/f'{arm}_FORECAST.npz'))
            require(len(raw) == len(guide) == N_ROUTES and info['native_intervals'] == 3*N_ROUTES,
                    'The actual native candidate budget was not equal and complete.')
            assembled, masses = [], []
            for slot, (z, attempts) in enumerate(zip(raw, guide, strict=True)):
                if arm == 'guided':
                    initial, mass, draws = _guide_entry(cfg, proxy, y, mother, slot, attempts)
                    cheap_draws += draws
                else:
                    require(not attempts, 'The prior comparator was guide-filtered.')
                    initial, mass = draw_initial(cfg, _rng(mother,4,slot,0)), 1.
                seeds = _rng(mother,5,slot).integers(0,2**32,3,dtype=np.uint32)
                assembled.append(_route(z, initial, seeds, f'{mother}:{arm}:{slot}', plan))
                masses.append(mass)
            cameras, feature, global_history = map(np.stack, zip(*assembled, strict=True))
            ll = -.5*np.square((cameras-y)/np.asarray(cfg['noise'])).reshape(N_ROUTES, -1).sum(axis=1)
            w = np.exp(ll - np.log(masses) - np.max(ll - np.log(masses)))
            w /= w.sum()
            for name, expected in [('features',feature),('early',global_history),('mass',masses),('ll',ll),('weights',w)]:
                close(forecast[name], expected)
            values[arm] = {'early18':_weighted_energy(feature,w,truth_feature),
                           'global24':_weighted_energy(global_history/scale,w,truth_global/scale),
                           'ess':float(1./(w@w)), 'median_ll':float(np.median(ll))}
            native[arm] += 3*len(assembled)
        distances = np.square(((train_y-y)/xscale).reshape(N_TRAIN,-1)).sum(axis=1)
        near = np.lexsort((np.arange(N_TRAIN), distances))[:8]
        for arm, ids in (('knn8',near), ('empirical',np.arange(N_TRAIN))):
            w = np.ones(len(ids))/len(ids)
            values[arm] = {'early18':_weighted_energy(train[ids],w,truth_feature),
                           'global24':_weighted_energy(train_global[ids],w,truth_global/scale)}
        for arm, value in values.items():
            for name, expected in value.items():
                close(summary['rows'][mother][arm][name], expected)
        rows.append(values)
    require(all(count == summary['inference_native_each'] == N_MOTHERS*N_ROUTES*3 for count in native.values())
            and acquired['native_intervals'] == summary['native_true'] == N_MOTHERS*3, 'Native route totals do not match receipts.')
    gates = []
    for arm, values in rows[0].items():
        for key in values:
            close(summary['means'][arm][key], np.mean([row[arm][key] for row in rows]))
            if arm == 'guided':
                continue
            diff = np.asarray([row['guided'][key]-row[arm][key] for row in rows])
            center = diff.mean()
            radius = t.ppf(.975, N_MOTHERS-1)*diff.std(ddof=1)/np.sqrt(N_MOTHERS)
            reported = summary['guided_minus'][arm][key]
            close([reported['mean'], *reported['pointwise_t95']], [center, center-radius, center+radius])
            if key == 'early18' and arm in ('prior', 'knn8', 'empirical'):
                gates.append(center+radius < 0.)
    require(len(gates) == 3 and all(gates) == summary['primary_complete_gate'], 'The primary paired decision did not reproduce.')
    write(out/'INDEPENDENT_AUDIT.json',{**stamp(), 'passed':True, 'summary_sha256':sha(out/'SUMMARY.json'),
          'registration_sha256':sha(out/'REGISTERED.json'), 'prediction_freeze_sha256':sha(out/'PREDICTIONS_FROZEN.json'),
          'source_sha256':sha(__file__), 'mothers':N_MOTHERS, 'inference_native_by_arm':native,
          'truth_native_intervals':N_MOTHERS*3, 'guided_cheap_draws_recomputed':cheap_draws,
          'complete_proposal_rng_and_raw_feature_readback':True, 'new_native_intervals':0,
          'numerical_scope':'Independent stored raw State/camera/graph, exact cheap Bernoulli and first-success routing, likelihood, weight, energy, strong frozen baselines and all paired score decisions.'})


def fixtures():
    """A cheap-grid fixture and an energy fixture. No native calls."""
    from aam_assay_guided_prior import draw_guided_initial, log_proxy, proposal_threshold, signature

    cfg = {'recipe': {'aam':64, 'mbaa':14, 'initial_mbaa':8, 'feed_time':2.5, 'initiators':9, 'horizon':3.},
           'motion': {'box':6.}}
    independent = np.random.default_rng(27185)
    proxy = {'center':independent.normal(size=8), 'scale':independent.uniform(.2,2.,size=8),
             'mean':independent.normal(size=12), 'coefficients':independent.normal(size=(8,12))*.03,
             'proxy_sigma':independent.uniform(.5,1.5,size=12)}
    y = independent.normal(size=(3,4))
    for _ in range(160):
        z = draw_initial(cfg, independent)
        require(_integer_threshold(z, proxy, y) == int(proposal_threshold(log_proxy(proxy,signature(z),y))),
                'Independent exact grid arithmetic did not match a synthetic guide threshold.')
    # Make acceptance occur at the floor and find a reproducible exhausted
    # first block followed by a success without ever constructing a State.
    distant = np.full((3,4), 500.)
    for slot in range(200):
        _, first = draw_guided_initial(cfg,distant,proxy,_rng(0,4,slot,0))
        if first['status'] == 'exhausted':
            z, second = draw_guided_initial(cfg,distant,proxy,_rng(0,4,slot,1))
            if z is not None:
                rebuild, mass, draws = _guide_entry(cfg,proxy,distant,0,slot,[first,second])
                require(initial_fingerprint(z) == initial_fingerprint(rebuild) and mass == .125
                        and draws == BLOCK_SIZE+second['attempts'], 'Independent first-success replay failed.')
                try:
                    _guide_entry(cfg,proxy,distant,0,slot,[first,second,second])
                except ValueError:
                    break
                raise ArithmeticError('A retained block after a first success was accepted.')
    else:
        raise ArithmeticError('Expected cheap synthetic exhaustion and then success.')
    close(_weighted_energy(np.asarray([[0.],[2.]]),np.asarray([.5,.5]),np.asarray([0.])), .5)
    print('PASS synthetic integer grid, exhausted-block replay, mutation rejection, and discrete energy. Zero native calls.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    mutually_exclusive = parser.add_mutually_exclusive_group(required=True)
    mutually_exclusive.add_argument('--out', type=Path)
    mutually_exclusive.add_argument('--fixtures', action='store_true')
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        fixtures() if args.fixtures else audit(args.out.resolve(strict=True))
