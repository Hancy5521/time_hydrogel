"""Train probabilities of complete native paths by the full energy score."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace
from itertools import product
from pathlib import Path
from shutil import copyfile

import numpy as np
from scipy.spatial.distance import cdist
from threadpoolctl import threadpool_limits

from aam_energy_coordinate_law import EnergyCoordinates, project_energy_mean
from aam_formation_cycle_driver import read, save_tree, sha, write
from aam_formation_path_cycle import event, load_tree, stamp
from aam_full_energy_tilt import Fit, energy_distances, fit, make_objective
from aam_nested_camera_coordinates import fit_nested_camera_map
from package_aam_formation_local_cycle import source_closure
from run_aam_mark_candidate_filter import interval, verify
from run_aam_temporal_energy_law import (
    acquire_one, all_losses, expose, geometry, load_map, measure_losses, measurements, normalize,
)


ROOT = Path(__file__).resolve().parent
PROTOCOL = ROOT / 'AAM_DIRECT_ENERGY_TRAJECTORY_LAW_PROTOCOL_ZH.md'
AUDITOR = ROOT / 'audit_aam_direct_energy_trajectory_law.py'
NAMESPACE = 'aam-full-energy-direct-path-probability-v1'
BASE = 2026114000
SIZES = {'learn': 768, 'dev': 128, 'held': 384}
COMPONENTS = {'hidden_t3_6': .50, 'early18': .25, 'hidden_delta32_6': .25}
PRIMARY = 'hidden_t3_6'
METRICS = (PRIMARY, 'hidden_delta32_6', 'early18', 'hidden_t2_6', 'global24', 'new_edge_brier')
RIDGE_FACTORS = (.01, .05, .20)
GAINS = (.25, .5, 1.)
GRIDS = {'direct': RIDGE_FACTORS, 'prior_fitted': RIDGE_FACTORS, 'kernel_cme': GAINS,
         'projected_temporal': tuple(product(GAINS, (0., .5, 1.))),
         'atlas_knn': tuple(product((8, 32, 96), GAINS)),
         'atlas_rbf': tuple(product((.5, 1., 2.), GAINS))}
FIXED = ('likelihood', 'empirical')
MATCHED = ('truncated_matched', 'first_two')
METHODS = (*GRIDS, *MATCHED, *FIXED)
RIDGE_COORDINATE = .20


def splits():
    result = {}
    base = BASE
    for group, count in SIZES.items():
        result[group] = list(range(base, base + count))
        base += count
    return result


def atlas_receipts(origin):
    """Use the independently audited training atlas without reading old held targets."""
    r = origin / 'REGISTERED.json'
    fit_doc = origin / 'TRAINED_AND_SELECTED.json'
    forecast = origin / 'PREDICTIONS_FROZEN.json'
    audit_doc = origin / 'INDEPENDENT_AUDIT.json'
    plan, trained, freeze, audit = (read(p) for p in (r, fit_doc, forecast, audit_doc))
    atoms = plan['units']['train']
    if (plan['namespace'] != 'aam-temporal-energy-positive-law-v1' or len(atoms) != 512
            or len(set(atoms)) != len(atoms) or trained['train_ids'] != atoms
            or freeze['train_atoms'] != atoms or trained['held_opened'] is not False
            or freeze['held_targets_opened'] is not False or audit['passed'] is not True
            or audit['independent_mothers']['train'] != len(atoms)
            or audit['registered_sha256'] != sha(r) or audit['forecast_sha256'] != sha(forecast)
            or freeze['fit_selection_sha256'] != sha(fit_doc)):
        raise ValueError('The fixed atlas must be the verified training-only data.')
    model = verify(origin / 'TRAINED_NATIVE_LAW.npz', trained['fitted'])
    return plan, atoms, model, (r, fit_doc, forecast, audit_doc, model)


def register(out, origin, work, workers):
    past, atoms, model, atlas_inputs = atlas_receipts(origin)
    units = splits()
    if set(atoms).intersection(i for group in units.values() for i in group):
        raise ValueError('Fresh acquisition mother addresses must exclude the fixed dictionary.')
    if not AUDITOR.is_file():
        raise ValueError('Independent numerical audit source must exist before freezing.')
    closure = source_closure([Path(__file__), AUDITOR])
    sources = {str(path): sha(path) for path in closure.values()}
    if any(sources[path] != digest for path, digest in past['sources'].items() if path in sources):
        raise ValueError('A verified existing source changed since the atlas experiment.')
    inputs = (*atlas_inputs, work, PROTOCOL)
    out.mkdir(exist_ok=False)
    plan = {**stamp(), 'namespace': NAMESPACE, 'physics': past['physics'], 'workers': workers,
            'units': units, 'atlas_units': atoms, 'atlas_model': str(model), 'atlas_rank': 12,
            'methods': METHODS, 'grids': GRIDS, 'metrics': METRICS, 'primary': PRIMARY,
            'composite_training': COMPONENTS, 'ridge_coordinate': RIDGE_COORDINATE,
            'direct_features': 'constant, first level, second-minus-first, third-minus-second; new train standardization',
            'atom_features': 'the 12 old training atlas coordinates divided by their single joint RMS',
            'ridge_scale': 'mean all-pair training atlas composite distance',
            'dev_selection': 'minimum full held-C3 energy on dev independently by family, ties earliest grid index',
            'matched_only': {name: 'fit whole ridge grid but inherit direct-selected ridge unchanged' for name in MATCHED},
            'truncated_loss': 'first 12 directions of the same composite negative-type geometry; squared Hilbert distance',
            'first_two': 'zero third camera increment covariates while keeping full loss and matching direct ridge',
            'gate': 'positive pointwise paired C3 t95 lower endpoints against every registered comparator',
            'swap_rule': 'freeze same direct probability rows exchanged within adjacent disjoint held mother pairs',
            'native_acquisition_calls': 3 * sum(SIZES.values()), 'native_inference_calls': 0,
            'old_held_targets_read': 0, 'new_wet_measurements': 0,
            'sources': sources, 'inputs': {str(path): sha(path) for path in inputs}}
    write(out / 'REGISTERED.json', plan)
    snapshot = out / 'SOURCE_SNAPSHOT'
    snapshot.mkdir()
    for name, path in closure.items():
        copyfile(path, snapshot / name)
    write(out / 'RUN.json', {**stamp(), 'registration_sha256': sha(out / 'REGISTERED.json'),
                             'registered_before_simulation': True})
    event(out, 'registered_full_score_training_with_same_native_atlas', native=plan['native_acquisition_calls'])


def checked(out):
    path = out / 'REGISTERED.json'
    plan = read(path)
    if (plan['namespace'] != NAMESPACE or plan['units'] != splits()
            or sha(path) != read(out / 'RUN.json')['registration_sha256']):
        raise ValueError('Require the original prospective experiment.')
    for filename, checksum in {**plan['sources'], **plan['inputs']}.items():
        if sha(filename) != checksum:
            raise ValueError(f'Frozen source/input changed: {filename}.')
    return plan


def acquire(out, plan):
    reg = sha(out / 'REGISTERED.json')
    write(out / 'ACQUISITION_STARTED.json', {**stamp(), 'registration_sha256': reg})
    groups = {}
    with ProcessPoolExecutor(max_workers=plan['workers']) as pool:
        for group, units in plan['units'].items():
            folder = out / group
            folder.mkdir()
            records = list(pool.map(acquire_one, [plan['physics']]*len(units), [folder]*len(units), units, chunksize=4))
            if [record['unit'] for record in records] != units:
                raise ValueError('Acquired mother addresses no longer match the registered split.')
            groups[group] = records
            event(out, 'closed_acquisition_role', group=group, mothers=len(records))
    write(out / 'ACQUIRED.json', {**stamp(), 'registration_sha256': reg, 'groups': groups,
                                  'native_calls': 3 * sum(SIZES.values())})


def causal_values(X):
    return np.column_stack((X[:, 0], X[:, 1] - X[:, 0], X[:, 2] - X[:, 1]))


def direct_design(X, mean, scale):
    return np.column_stack((np.ones(len(X)), (causal_values(X) - mean) / scale))


def composite_distances(truth, atoms):
    spaces = ((truth[:, 2], atoms[:, 2]),
              (truth.reshape(len(truth), -1), atoms.reshape(len(atoms), -1)),
              (truth[:, 2] - truth[:, 1], atoms[:, 2] - atoms[:, 1]))
    pieces = [energy_distances(a.reshape(len(a), -1), b.reshape(len(b), -1)) for a, b in spaces]
    between = sum(weight * pair[0] for weight, pair in zip(COMPONENTS.values(), pieces, strict=True))
    inside = sum(weight * pair[1] for weight, pair in zip(COMPONENTS.values(), pieces, strict=True))
    return between, inside


def truncate_composite(cross, pair, p):
    """Project the same full negative-type distance onto the first twelve axes."""
    offset = pair @ p
    origin = p @ offset
    centered = -.5 * (pair - offset[:, None] - offset[None, :] + origin)
    root = np.sqrt(p)
    weighted = root[:, None] * centered * root[None, :]
    eigenvalues, axes = np.linalg.eigh(.5 * (weighted + weighted.T))
    tol = 32 * len(p) * np.finfo(float).eps * np.abs(eigenvalues).max()
    if eigenvalues[0] < -tol or np.count_nonzero(eigenvalues > tol) < 12:
        raise ValueError('The composite score does not define twelve positive energy coordinates.')
    last = slice(-1, -13, -1)
    values = eigenvalues[last]
    projection = (root[:, None] * axes[:, last]) / np.sqrt(values)
    atlas = centered @ projection
    query = -.5 * (cross - (cross @ p)[:, None] - offset + origin) @ projection
    return cdist(query, atlas, 'sqeuclidean'), cdist(atlas, atlas, 'sqeuclidean'), {
        'eigenvalues': values, 'projection': projection, 'atlas_coordinates': atlas,
    }


def only_first_two(covariates):
    result = covariates.copy()
    result[:, -4:] = 0.
    return result


def fit_learning(X, local, atlas, learn_ids, atom_ids, noise):
    """Fit models to newly acquired learning pairs over the fixed dictionary."""
    p = atlas['prior']
    k = EnergyCoordinates(**atlas['basis'])
    atom = k.training_coordinates
    rms = np.sqrt(np.mean(np.sum(atom*atom, axis=1)))
    if atom.shape != (len(p), 12) or not np.isfinite(rms) or rms <= 0.:
        raise ValueError('The old registered atlas has lost its twelve valid directions.')
    Z = atom / rms
    photo = causal_values(X)
    mean = photo.mean(axis=0)
    floor = np.tile(noise, 3) * np.repeat([1., np.sqrt(2.), np.sqrt(2.)], 4)
    scale = np.maximum(photo.std(axis=0), floor)
    design = direct_design(X, mean, scale)
    raw_true, raw_atoms = local.reshape(len(local), -1), atlas['bank']['local'].reshape(len(p), -1)
    all_x, all_d = composite_distances(local, atlas['bank']['local'])
    trim_x, trim_d, trim_receipt = truncate_composite(all_x, all_d, p)
    multiplier = float(all_d.mean())
    base = make_objective(design, Z, raw_true, raw_atoms, p, query_mothers=learn_ids,
                          atom_mothers=atom_ids, ridge=RIDGE_FACTORS[0]*multiplier)
    complete = replace(base, target_distances=all_x, pair_distances=all_d)
    objectives = {'direct': complete, 'prior_fitted': replace(complete, query=design[:, :1]),
                  'first_two': replace(complete, query=only_first_two(design)),
                  'truncated_matched': replace(complete, target_distances=trim_x, pair_distances=trim_d)}
    trained = {}
    for name, obj in objectives.items():
        trained[name] = [asdict(fit(replace(obj, ridge=multiplier*factor))) for factor in RIDGE_FACTORS]
    targets = k.transform(geometry(local))
    y_bar = targets.mean(axis=0)
    atlas_scale = atlas['camera_scale']
    train_scale = np.maximum(X.std(axis=0), noise)
    learn_camera = (X / train_scale).reshape(len(X), -1)
    distance = cdist(learn_camera, learn_camera, 'sqeuclidean')
    kernel_scale = max(float(np.median(distance)), 1.e-8)
    kernel = np.exp(-.5 * distance / kernel_scale) + RIDGE_COORDINATE * np.eye(len(X))
    return {'direct_features': {'mean': mean, 'scale': scale, 'atlas_rms': rms, 'ridge_scale': multiplier},
            'direct_fits': trained, 'truncated_loss_geometry': trim_receipt,
            'learn_cme': {'camera': learn_camera, 'scale': train_scale,
                 'kernel_sq_scale': kernel_scale, 'coefficients': np.linalg.solve(kernel, targets - y_bar),
                 'target_mean': y_bar}, 'atlas_camera_scale': atlas_scale,
            'projected_temporal': asdict(fit_nested_camera_map(X, targets, noise, RIDGE_COORDINATE))}


def candidate_weights(model, atlas, X):
    p = atlas['prior']
    features = model['direct_features']
    base = np.broadcast_to(p, (len(X), len(p)))
    design = direct_design(X, features['mean'], features['scale'])
    atom = atlas['basis']['training_coordinates']
    standard = atom / features['atlas_rms']
    scale = model['atlas_camera_scale']
    train_camera, query_camera = ((x / scale).reshape(len(x), -1) for x in (atlas['x'], X))
    atlas_distance = cdist(query_camera, train_camera, 'sqeuclidean')
    neighbors = np.argsort(atlas_distance, axis=1, kind='stable')
    noise = atlas['noise']
    clean, query = ((x / noise).reshape(len(x), -1) for x in (atlas['bank']['clean'], X))
    log_like = -.5 * cdist(query, clean, 'sqeuclidean')
    cme = model['learn_cme']
    camera = (X / cme['scale']).reshape(len(X), -1)
    learned = np.exp(-.5 * cdist(camera, cme['camera'], 'sqeuclidean') / cme['kernel_sq_scale']) @ cme['coefficients']
    temporal = load_map(model['projected_temporal'])
    innovation = temporal.innovations(X)

    def candidate(name, setting):
        if name == 'empirical':
            return {'weights': base.copy()}
        if name == 'likelihood':
            return {'weights': normalize(log_like)}
        if name == 'atlas_knn':
            size, mixing = setting
            weights = (1. - mixing) * base
            weights[np.arange(len(X))[:, None], neighbors[:, :size]] += mixing / size
            return {'weights': weights}
        if name == 'atlas_rbf':
            bandwidth, mixing = setting
            weights = normalize(-.5 * atlas_distance / (bandwidth**2 * atlas['kernel_sq_scale']))
            return {'weights': mixing * weights + (1. - mixing) * base}
        if name in ('direct', 'prior_fitted', *MATCHED):
            record = model['direct_fits'][name][RIDGE_FACTORS.index(setting)]
            if name == 'prior_fitted':
                use = design[:, :1]
            elif name == 'first_two':
                use = only_first_two(design)
            else:
                use = design
            return {'weights': Fit(**record).predict(use, standard, p)}
        if name == 'kernel_cme':
            targets = cme['target_mean'] + setting * learned
        elif name == 'projected_temporal':
            gain, late = setting
            targets = temporal.target_mean + gain * (innovation[:, :2].sum(axis=1) + late * innovation[:, 2])
        else:
            raise ValueError('Unexpected forecast family.')
        result = [project_energy_mean(atom, row, p, atlas['entropy']) for row in targets]
        return {'weights': np.stack([r.weights for r in result]), 'targets': targets,
                'tilts': np.stack([r.tilt for r in result]),
                'residual': np.array([r.optimality_residual for r in result])}
    return candidate


def read_atlas(plan):
    path = Path(plan['atlas_model'])
    return load_tree(path, plan['inputs'][str(path)])


def train_select(out, plan):
    acquired_path = out / 'ACQUIRED.json'
    acquired = read(acquired_path)['groups']
    write(out / 'TRAINING_STARTED.json', {**stamp(), 'acquisition_sha256': sha(acquired_path),
                                           'allowed_query_and_truth': ['learn', 'dev']})
    atlas = read_atlas(plan)
    X = measurements(out, 'learn', acquired['learn'])
    local = expose(out, 'learn', acquired['learn'], plan['physics'])['local']
    train = fit_learning(X, local, atlas, plan['units']['learn'], plan['atlas_units'], atlas['noise'])
    fitted = save_tree(out / 'DIRECT_TRAINED.npz', train)
    y = measurements(out, 'dev', acquired['dev'])
    dev = expose(out, 'dev', acquired['dev'], plan['physics'])['local']
    cross, pair = energy_distances(dev[:, 2].reshape(len(dev), -1),
                                   atlas['bank']['local'][:, 2].reshape(len(atlas['prior']), -1))
    candidate = candidate_weights(train, atlas, y)
    def dev_losses(name, settings):
        return [float(np.mean(measure_losses(candidate(name, s)['weights'], cross, pair))) for s in settings]

    values = {name: dev_losses(name, settings) for name, settings in GRIDS.items()}
    matched = {name: dev_losses(name, RIDGE_FACTORS) for name in MATCHED}
    selected = {name: GRIDS[name][int(np.argmin(scores))] for name, scores in values.items()}
    selected.update({name: selected['direct'] for name in MATCHED})
    write(out / 'TRAINED_AND_SELECTED.json', {**stamp(), 'acquisition_sha256': sha(acquired_path),
          'model': fitted, 'atlas_sha256': plan['inputs'][plan['atlas_model']], 'learn_ids': plan['units']['learn'],
          'dev_ids': plan['units']['dev'], 'full_dev_loss_grid': values, 'matched_dev_scores': matched, 'selected': selected,
          'held_targets_read': False})


def swap_rows(number):
    if number % 2:
        raise ValueError('A paired label-swap requires an even count of independent mothers.')
    return (np.arange(number).reshape(-1, 2)[:, ::-1]).ravel()


def predict(out, plan):
    acquired_path = out / 'ACQUIRED.json'
    selected_path = out / 'TRAINED_AND_SELECTED.json'
    acquired, selected = read(acquired_path), read(selected_path)
    if selected['acquisition_sha256'] != sha(acquired_path) or selected['held_targets_read'] is not False:
        raise ValueError('Require new learning and development freeze before blind prediction.')
    atlas = read_atlas(plan)
    trained = load_tree(verify(out / 'DIRECT_TRAINED.npz', selected['model']))
    query_only = [{'unit': row['unit'], 'query': row['query']} for row in acquired['groups']['held']]
    y = measurements(out, 'held', query_only)
    candidate = candidate_weights(trained, atlas, y)
    methods = {name: candidate(name, selected['selected'].get(name)) for name in METHODS}
    arrays = save_tree(out / 'PREDICTION_ARRAYS.npz', {'query': y, 'methods': methods, 'swap_index': swap_rows(len(y))})
    write(out / 'PREDICTIONS_FROZEN.json', {**stamp(), 'acquisition_sha256': sha(acquired_path),
          'training_selection_sha256': sha(selected_path), 'atlas_units': plan['atlas_units'],
          'atlas_sha256': plan['inputs'][plan['atlas_model']], 'query_units': plan['units']['held'],
          'arrays': arrays, 'held_targets_read': False})


def score(out, plan):
    acquired_path = out / 'ACQUIRED.json'
    selected_path = out / 'TRAINED_AND_SELECTED.json'
    freeze = out / 'PREDICTIONS_FROZEN.json'
    acquired, lock = read(acquired_path), read(freeze)
    if (lock['acquisition_sha256'] != sha(acquired_path) or lock['training_selection_sha256'] != sha(selected_path)
            or lock['held_targets_read'] is not False or lock['query_units'] != plan['units']['held']):
        raise ValueError('The blind held queries lack their actual frozen predictions.')
    write(out / 'REVEAL_STARTED.json', {**stamp(), 'forecast_sha256': sha(freeze)})
    pred = load_tree(verify(out / 'PREDICTION_ARRAYS.npz', lock['arrays']))
    atlas = read_atlas(plan)
    truth = expose(out, 'held', acquired['groups']['held'], plan['physics'])
    recipe = plan['physics']['recipe']
    nbead = int(recipe['aam']) + int(recipe['mbaa'])

    def scores(weight):
        return all_losses(weight, atlas['bank'], truth, atlas['global_scale'], nbead)

    losses = {name: scores(pred['methods'][name]['weights']) for name in METHODS}
    swaps = pred['swap_index']
    if not np.array_equal(swaps, swap_rows(len(plan['units']['held']))):
        raise ValueError('Held swap changes the registered independent pair units.')
    swapped = scores(pred['methods']['direct']['weights'][swaps])
    comparisons = {name: {metric: interval(losses[name][metric] - losses['direct'][metric])
                           for metric in METRICS} for name in METHODS[1:]}
    paired_swap = {metric: (swapped[metric] - losses['direct'][metric]).reshape(-1, 2).mean(axis=1)
                   for metric in METRICS}
    data = save_tree(out / 'SCORED_ARRAYS.npz', {'losses': losses, 'direct_swapped_losses': swapped,
                     'direct_swap_pair_gains': paired_swap})
    write(out / 'SCORED.json', {**stamp(), 'registration_sha256': sha(out / 'REGISTERED.json'),
          'forecast_sha256': sha(freeze), 'scored_arrays': data, 'held_mothers': len(swaps),
          'means': {name: {key: float(np.mean(val)) for key, val in values.items()} for name, values in losses.items()},
          'gain_over': comparisons, 'primary_gate': all(c[PRIMARY]['pointwise_t95'][0] > 0 for c in comparisons.values()),
          'matched_vs_swapped_gain': {name: interval(values) for name, values in paired_swap.items()},
          'swap_independent_pairs': len(swaps)//2, 'selected': read(selected_path)['selected'],
          'mean_ess': {name: float(np.mean(1 / np.sum(p['weights']**2, axis=1))) for name, p in pred['methods'].items()},
          'native_acquisition': acquired['native_calls'], 'native_inference': 0, 'new_wet_measurements': 0})


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('phase', choices=('register', 'acquire', 'train-select', 'predict', 'score'))
    cli.add_argument('--out', required=True, type=Path)
    cli.add_argument('--atlas', type=Path)
    cli.add_argument('--work', type=Path)
    cli.add_argument('--workers', type=int, choices=range(1, 9), default=6)
    args = cli.parse_args()
    out = args.out.resolve()
    with threadpool_limits(limits=1):
        if args.phase == 'register':
            if args.atlas is None or args.work is None:
                cli.error('Provide --atlas and --work before prospective registration.')
            register(out, args.atlas.resolve(strict=True), args.work.resolve(strict=True), args.workers)
        else:
            {'acquire': acquire, 'train-select': train_select, 'predict': predict, 'score': score}[args.phase](out, checked(out))


if __name__ == '__main__':
    main()
