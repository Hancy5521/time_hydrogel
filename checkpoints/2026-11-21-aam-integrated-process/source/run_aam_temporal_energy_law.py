"""Train and blind-score temporal probability updates on complete native AAm paths."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from itertools import product
from pathlib import Path
from shutil import copyfile

import numpy as np
from scipy.spatial.distance import cdist
from threadpoolctl import threadpool_limits

from aam_energy_coordinate_law import fit_energy_coordinates, project_energy_mean
from aam_formation_cycle_driver import np_load, np_save, read, save_tree, sha, write
from aam_formation_path_cycle import event, load_tree, stamp
from aam_nested_camera_coordinates import NestedCameraMap, TimeBlock, fit_nested_camera_map
from package_aam_formation_local_cycle import source_closure
from run_aam_mark_candidate_filter import interval, verify
from run_aam_suffix_group_smoother import acquired_data, features


ROOT = Path(__file__).resolve().parent
PROTOCOL = ROOT / 'AAM_TEMPORAL_ENERGY_LAW_PROTOCOL_ZH.md'
NAMESPACE = 'aam-temporal-energy-positive-law-v1'
BASE = 2026112100
SIZES = {'train':512, 'dev':128, 'held':192}
RANK, RIDGE = 12, .20
K = (8, 32, 96)
MIX = (.25, .5, 1.)
LATE = (0., .5, 1.)
BANDWIDTH = (.5, 1., 2.)
METHODS = ('temporal', 'flat', 'kernel_cme', 'kernel', 'knn', 'knn8', 'empirical', 'likelihood')
METRICS = ('hidden_t3_6', 'hidden_delta32_6', 'early18', 'hidden_t2_6', 'global24', 'new_edge_brier')
GRIDS = {'temporal': tuple(product(MIX, LATE)), 'flat': tuple((g, 1.) for g in MIX),
         'kernel_cme': tuple((g, 1.) for g in MIX),
         'kernel': tuple(product(BANDWIDTH, MIX)), 'knn': tuple(product(K, MIX))}


def units():
    start = BASE
    result = {}
    for group, size in SIZES.items():
        result[group] = list(range(start, start+size))
        start += size
    return result


def register(out, rb, work, workers):
    audit, plan = read(rb / 'INDEPENDENT_AUDIT.json'), read(rb / 'REGISTERED.json')
    if (not audit['passed'] or audit['scored_sha256'] != sha(rb / 'SCORED.json')
            or audit['independent_mothers'] != 96):
        raise ValueError('Start from the independently audited new observation evidence.')
    src = source_closure([Path(__file__), ROOT / 'audit_aam_temporal_energy_law.py'])
    sources = {str(path): sha(path) for path in src.values()}
    if any(sources[path] != value for path, value in plan['sources'].items() if path in sources):
        raise ValueError('A verified original native source changed.')
    inputs = [rb / 'REGISTERED.json', rb / 'SCORED.json', rb / 'INDEPENDENT_AUDIT.json', work,
              PROTOCOL, ROOT / 'AAM_ENERGY_COORDINATE_LAW_MATH_ZH.md',
              ROOT / 'AAM_TEMPORAL_ENERGY_LAW_DESIGN_ZH.md']
    out.mkdir(exist_ok=False)
    reg = out / 'REGISTERED.json'
    write(reg, {**stamp(), 'namespace': NAMESPACE, 'physics': plan['physics'], 'units': units(), 'workers': workers,
                'rank': RANK, 'ridge': RIDGE, 'grids': GRIDS, 'methods': METHODS, 'metrics': METRICS,
                'primary': METRICS[0], 'gate': 'all paired 95% lower endpoints against listed controls > 0',
                'compact_embedding': 'joint three hidden port windows plus true adjacent C3-C2 difference',
                'fit': 'train only; rank and ridge fixed, exact same path dictionary across all methods',
                'selection': 'dev only: each registered family minimum full hidden C3 ES, tie earliest',
                'native_calls': 3*sum(SIZES.values()), 'native_inference_calls': 0,
                'old_held_targets_read': 0, 'new_wet_measurements': 0,
                'inputs': {str(path): sha(path) for path in inputs}, 'sources': sources})
    source = out / 'SOURCE_SNAPSHOT'
    source.mkdir()
    for name, path in src.items():
        copyfile(path, source / name)
    write(out / 'RUN.json', {**stamp(), 'registered_before_simulation': True, 'registration_sha256': sha(reg)})
    event(out, 'registered_temporal_law_and_unchanged_native_acquisition', native_calls=3*sum(SIZES.values()))


def checked(out):
    reg = out / 'REGISTERED.json'
    plan = read(reg)
    if (plan['namespace'] != NAMESPACE or plan['units'] != units()
            or sha(reg) != read(out / 'RUN.json')['registration_sha256']):
        raise ValueError('Require the actual prospective registration.')
    for name, expected in {**plan['sources'], **plan['inputs']}.items():
        if sha(name) != expected:
            raise ValueError(f'Registered source changed: {name}')
    return plan


def acquire_one(cfg, folder, unit):
    with threadpool_limits(limits=1):
        y, raw = acquired_data(cfg, unit)
        return {'unit': unit, 'query': np_save(folder / f'{unit}_QUERY.npz', measured=y),
                'raw': save_tree(folder / f'{unit}_RAW.npz', raw)}


def acquire(out, plan):
    write(out / 'ACQUISITION_STARTED.json', {**stamp(), 'registration_sha256': sha(out / 'REGISTERED.json')})
    groups = {}
    with ProcessPoolExecutor(max_workers=plan['workers']) as executor:
        for name, ids in plan['units'].items():
            folder = out / name
            folder.mkdir()
            records = list(executor.map(acquire_one, [plan['physics']]*len(ids), [folder]*len(ids), ids, chunksize=4))
            if [r['unit'] for r in records] != ids:
                raise ValueError('Actual acquired mother identities do not match registration.')
            groups[name] = records
            event(out, 'group_acquisition_closed', split=name, mothers=len(ids))
    write(out / 'ACQUIRED.json', {**stamp(), 'registration_sha256': sha(out / 'REGISTERED.json'), 'groups': groups,
                                  'native_calls': 3*sum(SIZES.values())})


def measurements(out, name, records):
    return np.stack([np_load(verify(out / name / f'{r["unit"]}_QUERY.npz', r['query']))['measured']
                     for r in records])


def expose(out, name, records, cfg):
    rows = []
    for r in records:
        raw = load_tree(verify(out / name / f'{r["unit"]}_RAW.npz', r['raw']))
        if raw['calls'] != {'attempted':3, 'completed':3, 'failed':0}:
            raise ValueError('Original physics has an incomplete mother.')
        rows.append({**features(raw['packets'][:-1], raw['packets'][-1], cfg), 'clean': raw['clean']})
    return {'local': np.stack([q['local'] for q in rows]), 'global_early': np.stack([q['global_early'] for q in rows]),
            'clean': np.stack([q['clean'] for q in rows]), 'new_edges': [q['new_edges'] for q in rows]}


def geometry(local):
    a = np.concatenate((local, (local[:,2]-local[:,1])[:,None]), axis=1)
    return a.reshape(len(local), -1)


def load_map(data):
    return NestedCameraMap(data['camera_mean'], data['camera_scale'], data['target_mean'],
                           tuple(TimeBlock(**block) for block in data['blocks']))


def fit_train(x, bank, noise):
    p = np.full(len(x), 1./len(x))
    basis = fit_energy_coordinates(geometry(bank['local']), p, RANK)
    coordinates = basis.training_coordinates
    model = fit_nested_camera_map(x, coordinates, noise, RIDGE)
    entropy = max(1.e-8, .5*np.mean(np.sum(coordinates**2, axis=1)))
    norm = np.maximum(x.std(axis=0), noise)
    a = (x/norm).reshape(len(x), -1)
    distance = cdist(a, a, 'sqeuclidean')
    scale = max(float(np.median(distance)), 1.e-8)
    kernel = np.exp(-.5*distance/scale)
    kernel.flat[::len(kernel)+1] += RIDGE
    coefficients = np.linalg.solve(kernel, coordinates-model.target_mean)
    return {'x': x, 'bank': bank, 'prior': p, 'noise': np.asarray(noise),
            'basis': asdict(basis), 'model': asdict(model), 'camera_scale': norm,
            'kernel_sq_scale': scale, 'kernel_coefficients':coefficients, 'entropy': entropy,
            'global_scale': np.maximum(bank['global_early'].reshape(-1, 8).std(axis=0),.05)}


def normalize(logs):
    a = np.exp(logs-logs.max(axis=1, keepdims=True))
    return a/a.sum(axis=1, keepdims=True)


def readout_inputs(model, y):
    norm = model['camera_scale']
    law = load_map(model['model'])
    n = len(y)
    d = cdist((y/norm).reshape(n,-1), (model['x']/norm).reshape(len(model['x']),-1), 'sqeuclidean')
    dist_clean = cdist((y/model['noise']).reshape(n,-1),
                       (model['bank']['clean']/model['noise']).reshape(len(model['x']),-1), 'sqeuclidean')
    return {'dist': d, 'dist_clean': dist_clean, 'neighbors': np.argsort(d, axis=1, kind='stable'),
            'innovations':law.innovations(y), 'center':law.target_mean}


def predict_variant(name, setting, model, inputs):
    n, p = len(inputs['dist']), model['prior']
    if name == 'empirical':
        return {'weights':np.broadcast_to(p, (n,len(p))).copy()}
    if name == 'likelihood':
        return {'weights':normalize(-.5*inputs['dist_clean'])}
    if name in ('knn','knn8'):
        k, mixture = setting if setting else (8, 1.)
        weights = np.broadcast_to((1.-mixture)*p, (n,len(p))).copy()
        weights[np.arange(n)[:,None], inputs['neighbors'][:,:k]] += mixture/k
        return {'weights':weights}
    if name == 'kernel':
        bandwidth, mixture = setting
        w = normalize(-.5*inputs['dist']/(bandwidth**2*model['kernel_sq_scale']))
        return {'weights':mixture*w+(1.-mixture)*p}
    g, last = setting
    if name == 'kernel_cme':
        update = np.exp(-.5*inputs['dist']/model['kernel_sq_scale'])@model['kernel_coefficients']
    else:
        update = inputs['innovations'][:,:2].sum(axis=1)+last*inputs['innovations'][:,2]
    target = inputs['center']+g*update
    phi = model['basis']['training_coordinates']
    result = [project_energy_mean(phi, row, p, model['entropy']) for row in target]
    return {'weights':np.stack([r.weights for r in result]), 'targets':target,
            'tilts':np.stack([r.tilt for r in result]), 'residual':np.asarray([r.optimality_residual for r in result])}


def distance_cube(a, b):
    x, y = a.reshape(len(a), -1), b.reshape(len(b), -1)
    return cdist(x, y)/np.sqrt(x.shape[1])


def measure_losses(weights, between, inside):
    return np.sum(weights*between, axis=1)-.5*np.sum((weights@inside)*weights,axis=1)


def train_select(out, plan):
    acquired_path = out / 'ACQUIRED.json'
    a = read(acquired_path)['groups']
    write(out / 'TRAINING_STARTED.json', {**stamp(), 'acquired_sha256':sha(acquired_path), 'allowed':['train','dev']})
    bank = expose(out, 'train', a['train'], plan['physics'])
    x = measurements(out, 'train', a['train'])
    model = fit_train(x, bank, plan['physics']['noise'])
    fitted = save_tree(out / 'TRAINED_NATIVE_LAW.npz', model)
    dev = expose(out, 'dev', a['dev'], plan['physics'])
    y = measurements(out, 'dev', a['dev'])
    inputs = readout_inputs(model, y)
    test, atoms = dev['local'][:,2], bank['local'][:,2]
    between, inside = distance_cube(test, atoms), distance_cube(atoms, atoms)
    selected, values = {}, {}
    for method, grid in GRIDS.items():
        values[method] = [float(measure_losses(predict_variant(method, config, model, inputs)['weights'],
                                                between, inside).mean()) for config in grid]
        selected[method] = grid[int(np.argmin(values[method]))]
    write(out / 'TRAINED_AND_SELECTED.json', {**stamp(), 'acquisition_sha256':sha(acquired_path), 'fitted':fitted,
          'train_ids': plan['units']['train'], 'dev_ids': plan['units']['dev'], 'held_opened':False,
          'registered_grids': GRIDS, 'selected': selected, 'full_dev_loss_grid': values})


def predict(out, plan):
    acquired_path, selected_path = out / 'ACQUIRED.json', out / 'TRAINED_AND_SELECTED.json'
    a, lock = read(acquired_path)['groups'], read(selected_path)
    if lock['acquisition_sha256'] != sha(acquired_path) or lock['held_opened']:
        raise ValueError('Only development selection can authorize blind forecasts.')
    model = load_tree(verify(out / 'TRAINED_NATIVE_LAW.npz', lock['fitted']))
    query = [{'unit':r['unit'], 'query':r['query']} for r in a['held']]
    y = measurements(out, 'held', query)
    inputs = readout_inputs(model, y)
    result = {name:predict_variant(name, lock['selected'].get(name), model, inputs) for name in METHODS}
    arrays = save_tree(out / 'PREDICTION_ARRAYS.npz', {'query': y, 'methods': result, 'innovations': inputs['innovations']})
    write(out / 'PREDICTIONS_FROZEN.json', {**stamp(), 'fit_selection_sha256':sha(selected_path),
          'acquisition_sha256':sha(acquired_path), 'arrays': arrays, 'query_units': plan['units']['held'],
          'train_atoms': plan['units']['train'], 'held_targets_opened': False})


def all_losses(w, atoms, actual, scale, bead_count):
    ac, tr = actual['local'], atoms['local']
    point = {'hidden_t3_6': (ac[:,2], tr[:,2]),
             'hidden_delta32_6': (ac[:,2]-ac[:,1], tr[:,2]-tr[:,1]),
             'early18':(ac,tr), 'hidden_t2_6':(ac[:,1],tr[:,1]),
             'global24':(actual['global_early']/scale, atoms['global_early']/scale)}
    losses = {key:measure_losses(w,distance_cube(test, prior),distance_cube(prior, prior))
              for key,(test,prior) in point.items()}
    n = bead_count**2
    y, x = np.zeros((len(ac),n),bool), np.zeros((len(tr),n),bool)
    for a,i in zip(x,atoms['new_edges'],strict=True):
        a[i] = True
    for a,i in zip(y,actual['new_edges'],strict=True):
        a[i] = True
    losses['new_edge_brier'] = np.square(w@x-y).sum(axis=1)
    return losses


def score(out, plan):
    freeze, acquired, fit_doc = (out / name for name in ('PREDICTIONS_FROZEN.json','ACQUIRED.json','TRAINED_AND_SELECTED.json'))
    f, a, fitted = read(freeze), read(acquired), read(fit_doc)
    if (f['acquisition_sha256'] != sha(acquired) or f['fit_selection_sha256'] != sha(fit_doc)
            or f['held_targets_opened'] is not False):
        raise ValueError('Require original blind forecast before revealing hidden histories.')
    write(out / 'REVEAL_STARTED.json', {**stamp(), 'freeze_sha256':sha(freeze)})
    model = load_tree(verify(out / 'TRAINED_NATIVE_LAW.npz', fitted['fitted']))
    pred = load_tree(verify(out / 'PREDICTION_ARRAYS.npz', f['arrays']))['methods']
    actual = expose(out, 'held', a['groups']['held'], plan['physics'])
    recipe = plan['physics']['recipe']
    bead_count = int(recipe['aam'])+int(recipe['mbaa'])
    losses = {name:all_losses(row['weights'],model['bank'],actual,model['global_scale'],bead_count)
              for name,row in pred.items()}
    paired = {name:{m:interval(losses[name][m]-losses['temporal'][m]) for m in METRICS}
              for name in METHODS[1:]}
    basis = model['basis']
    z = geometry(actual['local'])
    cross = distance_cube(z,basis['atoms'])
    center = -.5*(cross-(cross@basis['prior'])[:,None]-basis['prior_distances']+basis['prior_pair_distance'])
    target = center@basis['basis']
    estimate = pred['temporal']['targets']
    projected = pred['temporal']['weights']@basis['training_coordinates']
    scored_arrays = save_tree(out / 'SCORED_ARRAYS.npz', {'losses':losses,
        'raw_energy_gain':np.square(target).sum(axis=1)-np.square(target-estimate).sum(axis=1),
        'projected_energy_gain':np.square(target).sum(axis=1)-np.square(target-projected).sum(axis=1)})
    write(out / 'SCORED.json', {**stamp(), 'registration_sha256':sha(out / 'REGISTERED.json'),
          'freeze_sha256':sha(freeze), 'raw':scored_arrays, 'held_mothers': len(z),
          'means':{name:{m:np.mean(result[m]) for m in METRICS} for name,result in losses.items()},
          'pair_gain_over':paired, 'primary_gate':all(x[METRICS[0]]['pointwise_t95'][0]>0. for x in paired.values()),
          'selected':fitted['selected'], 'native_acquisition':a['native_calls'], 'native_inference':0,
          'mean_ess':{n:float(np.mean(1./np.square(d['weights']).sum(axis=1))) for n,d in pred.items()},
          'new_wet_measurements':0})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('register','acquire','train-select','predict','score'))
    parser.add_argument('--out', required=True,type=Path)
    parser.add_argument('--rb',type=Path)
    parser.add_argument('--work',type=Path)
    parser.add_argument('--workers', type=int, choices=range(1,9),default=6)
    args = parser.parse_args()
    out = args.out.resolve()
    with threadpool_limits(limits=1):
        if args.phase == 'register':
            if args.rb is None or args.work is None:
                parser.error('Provide --rb and --work at registration.')
            register(out,args.rb.resolve(strict=True),args.work.resolve(strict=True),args.workers)
        else:
            {'acquire':acquire,'train-select':train_select,'predict':predict,'score':score}[args.phase](out,checked(out))


if __name__ == '__main__':
    main()
