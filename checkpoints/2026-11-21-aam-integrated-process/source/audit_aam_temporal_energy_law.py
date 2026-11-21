"""Independent mathematical and numerical checks for the temporal energy law."""

import argparse
from datetime import datetime
from hashlib import sha256
from itertools import combinations_with_replacement, product
from pathlib import Path

import numpy as np
from scipy.stats import t
from threadpoolctl import threadpool_limits

from aam_formation_cycle_driver import np_load, read, sha, write
from aam_formation_path_cycle import load_tree, stamp
from audit_aam_assay_guided_sir import _camera
from audit_aam_suffix_group_smoother import _features, _initial, _interval, _rng, _state


BASE = 2026112100
SIZES = {'train': 512, 'dev': 128, 'held': 192}
LAW = 'aam-temporal-energy-positive-law-v1'
RNG_LAW = 'aam-shared-early-bank-grouped-native-suffix-smoother-v1'
ADDRESSES = {'data_initial': 1, 'data_tape': 2, 'measurement': 3}
PRIMARY = 'hidden_t3_6'
METRICS = (PRIMARY, 'hidden_delta32_6', 'early18', 'hidden_t2_6', 'global24', 'new_edge_brier')
MIXTURE = (.25, .5, 1.)
WIDTHS = (.5, 1., 2.)
SPLIT_RIDGE = .20
RANK = 12
BASE_GRIDS = {'temporal': tuple(product(MIXTURE, (0., .5, 1.))),
              'flat': tuple((gain, 1.) for gain in MIXTURE),
              'kernel_cme': tuple((gain, 1.) for gain in MIXTURE),
              'kernel': tuple(product(WIDTHS, MIXTURE)),
              'knn': tuple(product((8, 32, 96), MIXTURE))}
FIXED = ('knn8', 'empirical', 'likelihood')
SUPERVISED = ('temporal', 'flat', 'kernel_cme')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def close(actual, expected, atol=2e-11, rtol=2e-11):
    np.testing.assert_allclose(actual, expected, atol=atol, rtol=rtol)


def rows(value):
    value = np.asarray(value, dtype=float)
    require(value.ndim >= 2 and len(value) > 0 and np.isfinite(value).all(),
            'Expected finite, nonempty rows.')
    return value.reshape(len(value), -1)


def distances(left, right, normalize=True):
    """Direct Euclidean distances in blocks without calling the model kernel."""
    left, right = rows(left), rows(right)
    require(left.shape[1] == right.shape[1] and left.shape[1] > 0,
            'The two sets must use the same nonempty coordinate space.')
    result = np.empty((len(left), len(right)))
    for start in range(0, len(left), 32):
        difference = left[start:start + 32, None] - right[None, :]
        result[start:start + 32] = np.linalg.norm(difference, axis=2)
    if normalize:
        result /= np.sqrt(left.shape[1])
    return result


def simplex(weight, n, positive=False):
    weight = np.asarray(weight, dtype=float)
    require(weight.ndim in (1, 2) and weight.shape[-1] == n and np.isfinite(weight).all()
            and np.all(weight >= 0) and (not positive or np.all(weight > 0)),
            'Invalid finite probability masses for the ordered trajectory bank.')
    close(weight.sum(axis=-1), 1., atol=2e-12, rtol=0.)
    return weight


def full_energy(weight, support, target, pair=None, cross=None):
    """Full untruncated energy for every query using stable weighted distances."""
    support, target = rows(support), rows(target)
    weight = simplex(weight, len(support))
    require(weight.shape == (len(target), len(support)),
            'Each independent query requires its own probability vector.')
    if pair is None:
        pair = distances(support, support)
    if cross is None:
        cross = distances(target, support)
    require(pair.shape == (len(support), len(support)) and cross.shape == weight.shape,
            'Precomputed distances have incompatible dimensions.')
    return np.sum(weight * (cross - .5 * (weight @ pair)), axis=1)


def embedding(atoms, prior, offsets, average, basis, eigenvalues, coordinates, max_rank):
    """Validate weighted energy centering, retained complete spectrum and coordinates."""
    atoms = rows(atoms)
    prior = simplex(prior, len(atoms), positive=True)
    pair = distances(atoms, atoms)
    center = np.eye(len(atoms)) - np.broadcast_to(prior, pair.shape)
    gram = -.5 * center @ pair @ center.T
    close(offsets, pair @ prior)
    close(average, prior @ pair @ prior)
    close(gram, gram.T)
    close(gram @ prior, 0.)
    root = np.sqrt(prior)
    weighted = gram * root[:, None] * root[None, :]
    spectrum = np.linalg.eigvalsh(weighted)
    tolerance = 32 * np.finfo(float).eps * len(atoms) * np.max(np.abs(spectrum))
    require(spectrum[0] >= -tolerance, 'Energy Gram matrix is not positive semidefinite.')
    expected = spectrum[spectrum > tolerance][::-1][:max_rank]
    close(eigenvalues, expected)
    require(basis.shape == coordinates.shape == (len(atoms), len(expected)),
            'Invalid shape of the retained coordinate basis.')
    vectors = (basis / root[:, None]) * np.sqrt(eigenvalues)
    close(vectors.T @ vectors, np.eye(len(expected)))
    close(weighted @ vectors, vectors * eigenvalues)
    close(gram @ basis, coordinates)
    close(prior @ coordinates, 0.)
    close(coordinates.T @ (prior[:, None] * coordinates), np.diag(eigenvalues))
    return {'rank': len(expected), 'full_positive_rank': int(np.count_nonzero(spectrum > tolerance)),
            'lowest_eigenvalue': float(spectrum[0]), 'retained_variance': float(expected.sum()),
            'full_variance': float(np.maximum(spectrum, 0.).sum())}


def coordinates_at(target, support, prior, basis):
    pair = distances(support, support)
    to_atom = distances(target, support)
    prior = simplex(prior, len(support), positive=True)
    offset = pair @ prior
    centered = -.5 * (to_atom - (to_atom @ prior)[:, None] - offset + prior @ offset)
    return centered @ basis


def projection(weights, means, coordinates, prior, tau, tilts=None, residuals=None):
    """Check both primal and dual optimality for every archived posterior."""
    coordinates = rows(coordinates)
    means = rows(means)
    weights = simplex(weights, len(coordinates), positive=True)
    prior = simplex(prior, len(coordinates), positive=True)
    require(weights.shape == (len(means), len(coordinates))
            and means.shape[1] == coordinates.shape[1]
            and np.isfinite(tau) and tau > 0.,
            'Invalid posterior, forecast moment or entropy regularization.')
    moment = weights @ coordinates
    primal = (moment - means) @ coordinates.T / tau + np.log(weights / prior)
    primal -= (primal @ prior)[:, None]
    if tilts is None:
        tilts = (means - moment) / tau
    tilts = rows(tilts)
    require(tilts.shape == means.shape, 'Archived dual coordinates have a different width.')
    dual = moment - means + tau * tilts
    logits = np.log(prior) + tilts @ coordinates.T
    logits -= logits.max(axis=1)[:, None]
    expected = np.exp(logits)
    expected /= expected.sum(axis=1)[:, None]
    close(weights, expected, atol=2e-11, rtol=3e-10)
    dual_by_row = np.max(np.abs(dual), axis=1)
    if residuals is not None:
        close(residuals, dual_by_row, atol=2e-12, rtol=2e-8)
    scale = np.maximum(1., np.maximum(np.abs(means).max(axis=1), np.abs(coordinates).max()))
    require(np.all(dual_by_row <= 2e-8 * scale), 'Entropy dual stationarity does not meet tolerance.')
    primal_tolerance = 4e-8 * scale * max(1., np.abs(coordinates).max()) / tau + 2e-10
    require(np.all(np.abs(primal).max(axis=1) <= primal_tolerance),
            'Entropy primal stationarity does not meet tolerance.')
    return {'dual_max': float(np.abs(dual).max()), 'primal_max': float(np.abs(primal).max()),
            'simplex_max': float(np.abs(weights.sum(axis=1) - 1.).max()),
            'smallest_mass': float(weights.min())}


def stable_weights(log_weight):
    log_weight = rows(log_weight)
    values = np.exp(log_weight - log_weight.max(axis=1)[:, None])
    return values / values.sum(axis=1)[:, None]


def gaussian_camera(query, support, noise):
    query, support = rows(query), rows(support)
    noise = np.asarray(noise, dtype=float).ravel()
    require(len(noise) and np.all(np.isfinite(noise)) and np.all(noise > 0.)
            and query.shape[1] == support.shape[1] and query.shape[1] % len(noise) == 0,
            'Expected positive registered camera noise for each observed feature.')
    scale = np.tile(noise, query.shape[1] // len(noise))
    return stable_weights(-.5 * distances(query / scale, support / scale, normalize=False)**2)


def knn_probability(distance, neighbors, shrink=0.):
    require(0 < neighbors <= distance.shape[1] and 0. <= shrink <= 1.,
            'Invalid neighboring law shrunk towards the ordered training bank.')
    order = np.argsort(distance, axis=1, kind='stable')[:, :neighbors]
    weight = np.full(distance.shape, shrink / distance.shape[1])
    weight[np.arange(len(distance))[:, None], order] += (1 - shrink) / neighbors
    return weight


def kernel_probability(distance, width, shrink=0.):
    require(width > 0 and 0. <= shrink <= 1., 'Invalid RBF probability settings.')
    return (1 - shrink) * stable_weights(-.5 * (distance / width)**2) + shrink / distance.shape[1]


def interval(values):
    values = np.asarray(values, dtype=float)
    require(values.ndim == 1 and len(values) > 1 and np.isfinite(values).all(),
            'Pair each contrast within one novel independent mother.')
    n = len(values)
    mean = np.sum(values) / n
    term = t.isf(.025, n - 1) * np.sqrt(np.sum((values - mean)**2) / (n * (n - 1)))
    return {'mean': float(mean), 'pointwise_t95': [float(mean - term), float(mean + term)], 'mothers': n}


def solve_dev_projection(atoms, means, prior, tau):
    """Solve dev candidates with independent Newton descent on the log partition."""
    atoms, means = rows(atoms), rows(means)
    result = []
    for target in means:
        direction = np.zeros(atoms.shape[1])
        for _ in range(80):
            logits = np.log(prior) + atoms @ direction
            q = stable_weights(logits[None, :])[0]
            expectation = q @ atoms
            gradient = expectation - target + tau * direction
            if np.linalg.norm(gradient, ord=np.inf) < 5e-12:
                break
            centered = atoms - expectation
            hessian = (centered * q[:, None]).T @ centered + tau * np.eye(len(target))
            step = np.linalg.solve(hessian, gradient)
            objective = np.logaddexp.reduce(logits) - target @ direction + tau / 2 * (direction @ direction)
            for j in range(20):
                trial = direction - 2.**(-j) * step
                trial_value = (np.logaddexp.reduce(np.log(prior) + atoms @ trial) - target @ trial
                               + tau / 2 * (trial @ trial))
                if trial_value <= objective + 5e-15:
                    direction = trial
                    break
            else:
                raise ArithmeticError('Independent Newton line search failed.')
        else:
            raise ArithmeticError('Independent development projection failed to converge.')
        result.append(q)
    return np.stack(result)


def _camera_terms(x, time):
    frame = x[:, time]
    terms = [frame, np.array([frame[:, i] * frame[:, j]
                              for i, j in combinations_with_replacement(range(4), 2)]).T]
    terms.extend((frame[:, cur] * x[:, prev, k])[:, None]
                 for cur in range(4) for prev in range(time) for k in range(4))
    return np.concatenate(terms, axis=1)


def _innovation(X, xmean, xscale, blocks):
    x = (X - xmean) / xscale
    regressors = np.ones((len(x), 1))
    effects = []
    for time, block in enumerate(blocks):
        difference = (_camera_terms(x, time) - regressors @ block['residual_projection'])
        scaled = difference / block['residual_scale']
        effects.append(scaled @ block['coordinate_coefficients'])
        regressors = np.column_stack((regressors, scaled))
    return np.stack(effects, axis=1)


def check_innovations_fit(model, X, Y, noise):
    mean = X.mean(axis=0)
    scale = np.maximum(X.std(axis=0), noise)
    target_mean = Y.mean(axis=0)
    close(model['camera_mean'], mean)
    close(model['camera_scale'], scale)
    close(model['target_mean'], target_mean)
    blocks = model['blocks']
    require(len(blocks) == 3, 'Exactly three trained residual blocks must remain.')
    x, y = (X - mean) / scale, Y - target_mean
    past = np.ones((len(x), 1))
    maximum = 0.
    for time, block in enumerate(blocks):
        added = _camera_terms(x, time)
        fitted_projection = np.linalg.pinv(past, rcond=1.e-10) @ added
        close(block['residual_projection'], fitted_projection, atol=4e-10)
        residual = added - past @ fitted_projection
        sigma = np.maximum(np.sqrt(np.mean((residual - residual.mean(axis=0))**2, axis=0)), .05)
        close(block['residual_scale'], sigma)
        z = residual / sigma
        close(past.T @ z / len(past), 0.)
        coefficient = block['coordinate_coefficients']
        stationarity = z.T @ (z @ coefficient - y) / len(z) + SPLIT_RIDGE * coefficient
        close(stationarity, 0.)
        maximum = max(maximum, float(np.abs(stationarity).max()))
        past = np.column_stack((past, z))
    training = _innovation(X, mean, scale, blocks)
    replaced = np.array(X)
    replaced[:, 2] = np.roll(replaced[:, 2], 7, axis=0)
    np.testing.assert_array_equal(training[:, :2], _innovation(replaced, mean, scale, blocks)[:, :2])
    return maximum


def role_units():
    keys = list(SIZES)
    boundaries = [BASE, *np.cumsum(tuple(SIZES.values())) + BASE]
    return {key: list(range(lo, hi)) for key, lo, hi in zip(keys, boundaries[:-1], boundaries[1:], strict=True)}


def chrono(documents):
    utc = [datetime.fromisoformat(d['actual_utc']) for d in documents]
    local = [datetime.fromisoformat(d['asia_shanghai']) for d in documents]
    require(all(when.utcoffset().total_seconds() == 0. for when in utc)
            and all(when.utcoffset().total_seconds() == 8 * 3600. for when in local)
            and utc == local and all(a < b for a, b in zip(utc[:-1], utc[1:], strict=True)),
            'Actual registration, acquisition, fitting, forecast and reveal chronology failed.')


def artifact(path, receipt):
    require(Path(receipt['path']).resolve() == path.resolve()
            and path.stat().st_size == receipt['bytes'] and sha(path) == receipt['sha256'],
            f'Hash or whole original path changed: {path}')
    return path


def role_records(out, parts, role, design):
    """Verify every fresh query and state; no native trajectory is reintegrated."""
    cfg = design['physics']
    addresses = {**design, 'entropy': np.frombuffer(sha256(RNG_LAW.encode()).digest()[:16], dtype='<u4').tolist(),
                 'roles': ADDRESSES}
    observed, measured = [], []
    for record, unit in zip(parts, role_units()[role], strict=True):
        require(record['unit'] == unit, 'Split reorders or shares independently addressed mothers.')
        name = out / role / str(unit)
        query = np_load(artifact(name.with_name(name.name + '_QUERY.npz'), record['query']))
        require(set(query) == {'measured'} and query['measured'].shape == (3, 4),
                'A blind query carries more than its noisy camera channels.')
        q = query['measured']
        raw = load_tree(artifact(name.with_name(name.name + '_RAW.npz'), record['raw']))
        require(len(raw['packets']) == 4 and len(raw['intervals']) == 3
                and raw['calls'] == {'attempted': 3, 'completed': 3, 'failed': 0},
                'Fresh independent mother did not save the complete unforced event prefix.')
        packets = raw['packets']
        _initial(addresses, unit, 'data_initial', packets[0], raw['initial'])
        _state(packets[0], cfg, 0.)
        for camera, ledger in enumerate(raw['intervals']):
            _interval(addresses, unit, 'data_tape', camera, (), packets[camera], packets[camera + 1], ledger)
        clean = np.stack([_camera(p) for p in packets[1:]])
        close(raw['clean'], clean)
        expected_noise = np.asarray(cfg['noise']) * _rng(addresses, 'measurement', unit).standard_normal((3, 4))
        np.testing.assert_array_equal(q, raw['clean'] + expected_noise)
        measured.append(q)
        observed.append({**_features(packets, cfg), 'clean': clean})
    return np.stack(measured), {field: (np.stack([row[field] for row in observed])
                                      if field != 'new_edges' else [row[field] for row in observed])
                                for field in ('local', 'global_early', 'clean', 'new_edges')}


def joint_geometry(local):
    return np.column_stack(tuple(local[:, k].reshape(len(local), 6) for k in range(3))
                           + ((local[:, 2] - local[:, 1]).reshape(len(local), 6),))


def metric_spaces(data, bank, global_scale):
    x, z = data['local'], bank['local']
    return {PRIMARY: (x[:, 2], z[:, 2]),
            'hidden_delta32_6': (x[:, 2] - x[:, 1], z[:, 2] - z[:, 1]),
            'early18': (x, z),
            'hidden_t2_6': (x[:, 1], z[:, 1]),
            'global24': (data['global_early'] / global_scale, bank['global_early'] / global_scale)}


def all_metrics(probabilities, data, bank, global_scale, nsites):
    pair = {}
    for metric, (test, atoms) in metric_spaces(data, bank, global_scale).items():
        pair[metric] = (atoms, test, distances(atoms, atoms), distances(test, atoms))
    edges = []
    for target in (bank, data):
        flags = np.zeros((len(target['new_edges']), nsites**2))
        for row, ids in zip(flags, target['new_edges'], strict=True):
            require(len(set(ids)) == len(ids) and all(0 <= x < nsites**2 and x // nsites < x % nsites for x in ids),
                    'New edge targets must identify distinct unordered original graph edges.')
            row[ids] = 1.
        edges.append(flags)
    train, held = edges
    result = {}
    for method, weight in probabilities.items():
        terms = {name: full_energy(weight, atoms, truth, inside, between)
                 for name, (atoms, truth, inside, between) in pair.items()}
        terms['new_edge_brier'] = np.sum((weight @ train - held)**2, axis=1)
        result[method] = terms
    return result


def validate_receipts(out):
    files = ('REGISTERED.json', 'RUN.json', 'ACQUISITION_STARTED.json', 'ACQUIRED.json',
             'TRAINING_STARTED.json', 'TRAINED_AND_SELECTED.json', 'PREDICTIONS_FROZEN.json',
             'REVEAL_STARTED.json', 'SCORED.json')
    docs = [read(out / file) for file in files]
    chrono(docs)
    reg, run, begin_acq, acq, begin_fit, fit, forecast, reveal, scored = docs
    rh, ah, fh, ph = [sha(out / files[i]) for i in (0, 3, 5, 6)]
    require(reg['namespace'] == LAW and reg['units'] == role_units() and reg['rank'] == RANK
            and reg['ridge'] == SPLIT_RIDGE and reg['primary'] == PRIMARY
            and tuple(reg['methods']) == (*BASE_GRIDS, *FIXED) and tuple(reg['metrics']) == METRICS
            and {key: [list(row) for row in values] for key, values in BASE_GRIDS.items()} == reg['grids']
            and reg['grids'] == fit['registered_grids']
            and tuple(reg['physics']['prefix_clock']) == (0., .65, 1.35, 2.05)
            and tuple(reg['physics']['noise']) == (.08, .10, .08, .10)
            and run['registered_before_simulation'] is True
            and reg['old_held_targets_read'] == 0 and reg['new_wet_measurements'] == scored['new_wet_measurements'] == 0
            and all(d['registration_sha256'] == rh for d in (run, begin_acq, acq, scored))
            and begin_fit['acquired_sha256'] == fit['acquisition_sha256'] == forecast['acquisition_sha256'] == ah
            and forecast['fit_selection_sha256'] == fh and reveal['freeze_sha256'] == scored['freeze_sha256'] == ph
            and begin_fit['allowed'] == ['train', 'dev'] and fit['held_opened'] is False
            and forecast['held_targets_opened'] is False and fit['train_ids'] == forecast['train_atoms'] == role_units()['train']
            and fit['dev_ids'] == role_units()['dev'] and forecast['query_units'] == role_units()['held']
            and scored['held_mothers'] == SIZES['held'] and scored['selected'] == fit['selected']
            and reg['native_calls'] == acq['native_calls'] == scored['native_acquisition'] == 3 * sum(SIZES.values())
            and reg['native_inference_calls'] == scored['native_inference'] == 0
            and set(acq['groups']) == set(SIZES),
            'Prospective design, resource counts or query/target role binding differs.')
    for name, digest in reg['sources'].items():
        require(sha(out / 'SOURCE_SNAPSHOT' / Path(name).name) == digest, f'Frozen source changed: {name}')
    require(any(Path(path).name == Path(__file__).name and digest == sha(__file__)
                for path, digest in reg['sources'].items()), 'Audit must use its own prospectively registered source.')
    for name, digest in reg['inputs'].items():
        require(sha(name) == digest, f'Registered input changed: {name}')
    return reg, acq, fit, forecast, scored


def train_check(X, bank, fitted, noise):
    close(fitted['x'], X)
    close(fitted['noise'], noise)
    for name in ('local', 'global_early', 'clean'):
        close(fitted['bank'][name], bank[name])
    require(fitted['bank']['new_edges'] == bank['new_edges'], 'Training edge rows changed.')
    prior = np.full(len(X), 1. / len(X))
    basis = fitted['basis']
    atoms = joint_geometry(bank['local'])
    close(basis['atoms'], atoms)
    close(basis['prior'], prior)
    close(fitted['prior'], prior)
    result = embedding(atoms, prior, basis['prior_distances'], basis['prior_pair_distance'],
                       basis['basis'], basis['eigenvalues'], basis['training_coordinates'], RANK)
    z = basis['training_coordinates']
    tau = max(1.e-8, .5 * np.sum(z*z) / len(z))
    scale = np.maximum(np.std(X, axis=0), noise)
    close(fitted['entropy'], tau)
    close(fitted['camera_scale'], scale)
    close(fitted['global_scale'], np.maximum(np.std(bank['global_early'].reshape(-1, 8), axis=0), .05))
    dis2 = distances(X / scale, X / scale, normalize=False)**2
    denominator = max(1.e-8, float(np.median(dis2)))
    close(fitted['kernel_sq_scale'], denominator)
    kernel = np.exp(-dis2 / (2 * denominator))
    raw = z - fitted['model']['target_mean']
    equation = (kernel + SPLIT_RIDGE * np.eye(len(kernel))) @ fitted['kernel_coefficients'] - raw
    close(equation, 0.)
    result['kernel_cme_stationarity_max'] = float(np.abs(equation).max())
    result['increment_fit_stationarity_max'] = check_innovations_fit(fitted['model'], X, z, noise)
    return result


def independently_predict(fitted, query):
    meanmap = fitted['model']
    scale = fitted['camera_scale']
    dist = distances(query / scale, fitted['x'] / scale, normalize=False)
    innovation = _innovation(query, meanmap['camera_mean'], meanmap['camera_scale'], meanmap['blocks'])
    anchor = meanmap['target_mean']
    width = np.sqrt(fitted['kernel_sq_scale'])
    kernel = np.exp(-.5 * (dist / width)**2) @ fitted['kernel_coefficients']
    law = {'empirical': np.broadcast_to(fitted['prior'], dist.shape),
           'likelihood': gaussian_camera(query, fitted['bank']['clean'], fitted['noise']),
           'knn8': knn_probability(dist, 8)}

    def candidate(method, setting):
        if method in law:
            return 'weight', law[method]
        first, second = setting
        if method == 'knn':
            return 'weight', knn_probability(dist, first, 1 - second)
        if method == 'kernel':
            return 'weight', kernel_probability(dist, first * width, 1 - second)
        update = kernel if method == 'kernel_cme' else innovation[:, 0] + innovation[:, 1] + second * innovation[:, 2]
        return 'target', anchor + first * update
    return candidate, innovation


def check_selection(model, records, bank, query, dev):
    probability, _ = independently_predict(model, query)
    atoms, truth = bank['local'][:, 2], dev['local'][:, 2]
    pair, cross = distances(atoms, atoms), distances(truth, atoms)
    dev_best = {}
    for method, grid in BASE_GRIDS.items():
        values = []
        for setting in grid:
            kind, w = probability(method, setting)
            if kind == 'target':
                w = solve_dev_projection(model['basis']['training_coordinates'], w, model['prior'], model['entropy'])
            values.append(float(np.mean(full_energy(w, atoms, truth, pair, cross))))
        prior_report = records['full_dev_loss_grid'][method]
        close(prior_report, values, atol=2e-10)
        position = int(np.argmin(prior_report))
        require(list(grid[position]) == records['selected'][method]
                and values[position] <= min(values) + 2.e-10,
                f'{method} did not select its registered development minimum.')
        dev_best[method] = {'configuration': list(grid[position]), 'full_energy': values[position]}
    return dev_best


def check_forecast(model, selection, archive, query):
    require(set(archive.keys()) == {'query', 'methods', 'innovations'}
            and tuple(archive['methods']) == (*BASE_GRIDS, *FIXED),
            'Frozen predictions do not consist of exactly the registered routes.')
    np.testing.assert_array_equal(archive['query'], query)
    independently, innovation = independently_predict(model, query)
    close(archive['innovations'], innovation)
    projection_diagnostics = {}
    weights = {}
    for method, result in archive['methods'].items():
        kind, value = independently(method, selection.get(method))
        weights[method] = simplex(result['weights'], len(model['prior']))
        if kind == 'weight':
            require(set(result) == {'weights'}, 'Direct probability route carries extra forecast inputs.')
            close(result['weights'], value)
        else:
            require(set(result) == {'weights', 'targets', 'tilts', 'residual'},
                    'A learned route does not bind its estimated coordinate and exact law.')
            close(result['targets'], value)
            projection_diagnostics[method] = projection(result['weights'], result['targets'],
                                                        model['basis']['training_coordinates'], model['prior'],
                                                        model['entropy'], result['tilts'], result['residual'])
    return weights, projection_diagnostics


def verify_scores(archive, bank, held, estimates, scored, prediction, model, nsites):
    result = all_metrics(estimates, held, bank, model['global_scale'], nsites)
    successes = []
    require(tuple(result) == tuple(archive['losses']) == tuple(scored['means'])
            and tuple(scored['pair_gain_over']) == tuple(name for name in result if name != 'temporal'),
            'Summary differs from frozen comparator structure.')
    for method, loss in result.items():
        require(tuple(loss) == METRICS and tuple(archive['losses'][method]) == METRICS, 'Output endpoints differ.')
        for m, values in loss.items():
            close(values, archive['losses'][method][m], atol=5e-10)
            close(np.mean(values), scored['means'][method][m], atol=5e-10)
            if method == 'temporal':
                continue
            proposed = interval(values - result['temporal'][m])
            stored = scored['pair_gain_over'][method][m]
            close([proposed['mean'], *proposed['pointwise_t95']], [stored['mean'], *stored['pointwise_t95']])
            require(stored['mothers'] == proposed['mothers'] == SIZES['held'], 'Pairing changed independent unit.')
            if m == PRIMARY:
                successes.append(proposed['pointwise_t95'][0] > 0.)
        close(np.mean(1. / np.sum(estimates[method]**2, axis=1)), scored['mean_ess'][method])
    require(len(successes) == len(estimates) - 1 and all(successes) == scored['primary_gate'],
            'Primary pass criterion differs from its registered paired intervals.')
    basis = model['basis']
    y = coordinates_at(joint_geometry(held['local']), basis['atoms'], basis['prior'], basis['basis'])
    z = prediction['methods']['temporal']['targets']
    q = estimates['temporal'] @ basis['training_coordinates']
    raw = np.sum(2 * y * z - z*z, axis=1)
    projected = np.sum(2 * y * q - q*q, axis=1)
    close(raw, archive['raw_energy_gain'])
    close(projected, archive['projected_energy_gain'])
    return {'raw_energy_gain': interval(raw), 'projected_energy_gain': interval(projected)}


def audit(out):
    reg, acq, fit, forecast, scored = validate_receipts(out)
    model = load_tree(artifact(out / 'TRAINED_NATIVE_LAW.npz', fit['fitted']))
    prediction = load_tree(artifact(out / 'PREDICTION_ARRAYS.npz', forecast['arrays']))
    results = load_tree(artifact(out / 'SCORED_ARRAYS.npz', scored['raw']))
    X, bank = role_records(out, acq['groups']['train'], 'train', reg)
    spectral = train_check(X, bank, model, np.asarray(reg['physics']['noise']))
    Y, dev = role_records(out, acq['groups']['dev'], 'dev', reg)
    chosen = check_selection(model, fit, bank, Y, dev)
    Z, held = role_records(out, acq['groups']['held'], 'held', reg)
    weights, stationarity = check_forecast(model, fit['selected'], prediction, Z)
    nsites = int(reg['physics']['recipe']['aam']) + int(reg['physics']['recipe']['mbaa'])
    outcome = verify_scores(results, bank, held, weights, scored, prediction, model, nsites)
    return {**stamp(), 'passed': True, 'run': str(out), 'independent_mothers': SIZES,
            'registered_sha256': sha(out / 'REGISTERED.json'), 'scored_sha256': sha(out / 'SCORED.json'),
            'forecast_sha256': sha(out / 'PREDICTIONS_FROZEN.json'), 'sources_checked': len(reg['sources']),
            'birth_ledger_checks': 3 * sum(SIZES.values()), 'new_native_calls': 0,
            'primary_gate': scored['primary_gate'], 'training_math': spectral,
            'development_selection': chosen, 'forecast_stationarity': stationarity,
            'full_paired_intervals_recomputed': True, 'full_untruncated_energy_recomputed': True,
            'local_and_global_endpoints_from_original_packets': True, 'diagnostics': outcome,
            'selftest': selftest()}


def selftest():
    atoms = np.array([[0.], [2.]])
    prior = np.array([.5, .5])
    basis = np.array([[.5], [-.5]]) / np.sqrt(.5)
    coords = basis.copy()
    record = embedding(atoms, prior, [1., 1.], 1., basis, np.array([.5]), coords, 2)
    close(coordinates_at(atoms, atoms, prior, basis), coords)
    weights = np.array([[.5, .5], [1., 0.]])
    close(full_energy(weights, atoms, atoms), [.5, 2.])
    tilt = np.array([[.7], [-.2]])
    q = stable_weights(tilt @ coords.T)
    means = q @ coords + .3 * tilt
    project = projection(q, means, coords, prior, .3, tilt, np.zeros(2))
    try:
        projection(q, means + .01, coords, prior, .3, tilt)
    except (ValueError, AssertionError):
        pass
    else:
        raise AssertionError('Known nonstationary probability passed.')
    close(knn_probability(np.array([[1., 1., 2.]]), 1, .3), [[.8, .1, .1]])
    close(gaussian_camera(atoms, atoms, [2.]), stable_weights(-np.array([[0., .5], [.5, 0.]])))
    return {'passed': True, 'embedding': record, 'projection': project}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('run', type=Path, nargs='?')
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        if args.run is None:
            print(selftest())
        else:
            out = args.run.resolve(strict=True)
            result = audit(out)
            write(out / 'INDEPENDENT_AUDIT.json', result)
            print('PASS temporal law audit:', str(out))


if __name__ == '__main__':
    main()
