"""Check zero-law coupling and independently recompute frozen process study scores."""

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


ARMS = ('constant_physics', 'learned_physics', 'direct_trees')
PATH_CHANNELS = ('hydration', 'near_modulus', 'far_modulus', 'conductance')
ZERO_LAW_FIXTURE_TOLERANCE = 3e-4


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(path.read_bytes())


def export(path, body):
    report = {'audit_utc': datetime.now(timezone.utc).isoformat(), 'audit_code_sha256': sha(Path(__file__)), **body}
    with path.open('x', encoding='utf8') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    print(path.as_posix(), flush=True)


def read_context(run):
    protocol = run/'REGISTERED.json'
    plan = read(protocol)
    for name, expected in plan['hashes'].items():
        if sha(Path(name)) != expected:
            raise ValueError(f'Registered input changed: {name}.')
    task = read(Path(plan['task_path']))
    request = task['requests'][plan['tight']['request']]
    clock = np.arange(round(request['work_end']/plan['sampling_dt'])+1) * plan['sampling_dt']
    return protocol, plan, task, request, clock


def fixture(args):
    from patch_learned_water_ion import joint_trace
    from patch_reservoir_joint_bank import joint_bank

    protocol, plan, task, _request, clock = read_context(args.run)
    parent = {'thickness': 1., 'water_D': .14, 'release': .92}
    # Nonmonotone film order and repeated films check action-to-hydraulics indexing.
    actions = [task['actions'][i] for i in (6, 1, 8, 3)]
    kinetics = np.array([[.29, .75, .025, 0.], [1.05, 1.22, .15, .90]])
    shapes = np.array([[.12, .40], [-.22, -.35]])
    law = np.zeros(5)
    entries = []
    error = np.zeros(len(PATH_CHANNELS))
    max_inventory_error = 0.
    for mesh_name in ('prediction_mesh', 'truth_mesh'):
        cells, step = plan[mesh_name]
        for thickness in (plan['coupon']['witness_scale'], 1., 1.35):
            physical = {**parent, 'thickness': thickness}
            coefficients = {**task['mechanism'], 'dose': task['mechanism']['dose']/thickness}
            for wet, bath in (*task['bath'].items(), ('dry_edge', 0.)):
                prior = joint_bank(physical, actions, coefficients, bath, kinetics, shapes, cells, step, clock)
                coupled, profiles, inventory_error = joint_trace(physical, actions, coefficients, bath, kinetics,
                                                                shapes, cells, step, clock, law)
                difference = np.max(np.abs(coupled-prior), axis=(0, 1, 2))
                if not np.allclose(coupled[0, ..., 0], profiles.mean(axis=-1).T, atol=1e-14, rtol=0.):
                    raise ValueError('Hydration averages do not index the same action and time.')
                if not np.allclose(profiles[0], profiles[2], atol=1e-14, rtol=0.):
                    raise ValueError('Identical wetting films gave different hydraulic fields.')
                if wet == 'dry_edge' and (profiles.any() or not np.allclose(coupled[..., 3], coefficients['dry_conductivity'])):
                    raise ValueError('Dry reference changed hydration or conductance.')
                entries.append({'mesh': mesh_name, 'thickness': thickness, 'boundary': wet,
                                'path_max_absolute_error': dict(zip(PATH_CHANNELS, difference.tolist(), strict=True)),
                                'inventory_error': float(inventory_error)})
                max_inventory_error = max(max_inventory_error, float(inventory_error))
                error = np.maximum(error, difference)
    if error.max() > ZERO_LAW_FIXTURE_TOLERANCE:
        raise ArithmeticError(f'Zero-law fixture exceeded the agreement scale: {error}.')
    export(args.output, {'protocol_sha256': sha(protocol), 'passed': True,
                         'absolute_agreement_tolerance': ZERO_LAW_FIXTURE_TOLERANCE,
                         'max_error_by_channel': dict(zip(PATH_CHANNELS, error.tolist(), strict=True)),
                         'max_ion_inventory_error': max_inventory_error, 'fixtures': entries,
                         'scope': 'Zero hydration-law control uses a different hydraulic integrator from the older constant-D implementation. Checks both registered grids, with no outcomes from the held set.'})


def margin_from_paths(paths, clock, task, request):
    # The clauses are intentionally expanded here instead of calling the production extrema routine.
    stop = clock <= request['install_end'] + 1e-10
    operate = (clock >= request['work_start'] - 1e-10) & (clock <= request['work_end'] + 1e-10)
    through_end = clock <= request['work_end'] + 1e-10
    wet = paths[:, :, 0]
    weak = paths[:, :, 1]
    before = wet[:, :, stop]
    working = wet[:, :, operate]
    safety = weak[:, :, through_end]
    target, scale = task['thresholds'], task['scales']
    blocks = (
        (before[..., 3].max(axis=2) - target['off']) / scale['off'],
        (target['on'] - working[..., 3].min(axis=2)) / scale['on'],
        (working[..., 1].max(axis=2) - target['soft']) / scale['soft'],
        (target['support'] - before[..., 2].min(axis=2)) / scale['support'],
        (safety[..., 3].max(axis=2) - target['weak_off']) / scale['weak_off'],
    )
    return np.maximum.reduce(blocks)


def same_or_float(observed, expected, label):
    if not np.allclose(observed, expected, rtol=3e-13, atol=3e-15):
        raise ValueError(f'{label} mismatches ({observed} versus {expected}).')


def check_group(key, mask, frozen, truth, plan, recorded, future):
    y = truth['margin'][mask]
    observed_paths = truth['paths'][mask]
    groups, states, actions = y.shape
    if groups != recorded['mothers'] or int((y.min(axis=-1) <= 0).sum()) != recorded['feasible_requests']:
        raise ValueError(f'{key} mother or feasibility count differs.')
    successes = {}
    for arm in ARMS:
        choice = frozen[f'{arm}_choice'][mask]
        if (choice < 0).any() or (choice >= actions).any():
            raise ValueError(f'{key}/{arm} has an invalid action index.')
        posterior = frozen[f'{arm}_probability'][mask]
        if posterior.min() < -1e-12 or posterior.max() > 1. + 1e-12:
            raise ValueError(f'{key}/{arm} has an invalid probability.')
        if arm != 'direct_trees':
            cost = posterior + plan['risk_weight']*frozen[f'{arm}_margin'][mask]
            if not np.array_equal(choice, cost.argmin(axis=-1)):
                raise ValueError(f'{key}/{arm} frozen action differs from the stated physical ranking.')
        picked = y[np.arange(groups)[:, None], np.arange(states)[None, :], choice]
        safe = picked <= 0
        successes[arm] = safe.mean(axis=1)
        computed = {'brier': float(np.mean((posterior-(y>0))**2)), 'ungated_successes': int(safe.sum()),
                    'mean_positive_regret': float(np.mean(np.maximum(picked,0.)-np.maximum(y.min(axis=-1),0.)))}
        if arm != 'direct_trees':
            diff = np.abs(frozen[f'{arm}_paths'][mask]-observed_paths)
            future_diff = np.compress(future, diff, axis=3)
            computed['wet_future_modulus_mae'] = float(future_diff[:, :, 0, :, :, 1:3].mean())
            for i, environment in enumerate(('wet', 'weak')):
                computed[environment+'_future_conductance_mae'] = float(future_diff[:, :, i, :, :, 3].mean())
        for name, value in computed.items():
            same_or_float(value, recorded['methods'][arm][name], f'{key}/{arm}/{name}')
    rng = np.random.default_rng(plan['process_transfer']['bootstrap_seed'])
    draws = rng.integers(groups, size=(5000, groups))
    for control, claimed in recorded['learned_choice_gain'].items():
        effect = successes['learned_physics'] - successes[control]
        same_or_float(effect.mean(), claimed['difference'], f'{key}/difference/{control}')
        same_or_float(np.quantile(effect[draws].mean(axis=1), [.025, .975]), claimed['ci95'], f'{key}/paired interval/{control}')


def score(args):
    protocol, plan, task, request, clock = read_context(args.run)
    f, t = (args.run/filename for filename in ('FROZEN.npz', 'TRUTH.npz'))
    receipt = read(args.run/'FORECAST_RECEIPT.json')
    scored = read(args.run/'SCORED.json')
    if receipt['protocol_sha256'] != sha(protocol) or scored['protocol_sha256'] != sha(protocol):
        raise ValueError('The forecast or score points to a different registration.')
    if receipt['frozen_sha256'] != sha(f) or scored['freeze_sha256'] != sha(f):
        raise ValueError('The released outcome points to a different forecast.')
    if not plan['registered_utc'] < receipt['completed_utc'] < scored['completed_utc']:
        raise ValueError('Registration, freeze and release are not ordered.')
    with np.load(f, allow_pickle=False) as saved_forecast, np.load(t, allow_pickle=False) as saved_truth:
        frozen, truth = dict(saved_forecast), dict(saved_truth)
    if not np.array_equal(frozen['mother'], truth['mother']) or len(set(truth['mother'])) != len(truth['mother']):
        raise ValueError('Mother pairing differs, or a mother is repeated.')
    if len(truth['mother']) != plan['event_roles']['confirmation']['groups']:
        raise ValueError('The number of mothers differs from the registration.')
    recovered_margin = margin_from_paths(truth['paths'], clock, task, request)
    same_or_float(recovered_margin, truth['margin'], 'full true margins')
    future = clock > max(plan['spatial_probe']['times']) + 1e-10
    membership = np.asarray(truth['exchange'], dtype=bool)
    for key, mask in (('all', np.ones_like(membership)), ('nominal', ~membership), ('exchange', membership)):
        check_group(key, mask, frozen, truth, plan, scored['results'][key], future)
    for arm in ARMS[:2]:
        same_or_float(np.max(frozen[f'{arm}_conservation']), scored['ion_closure_max'][arm], f'{arm}/conservation')
    export(args.output, {'passed': True, 'protocol_sha256': sha(protocol), 'forecast_sha256': sha(f), 'truth_sha256': sha(t),
                         'score_sha256': sha(args.run/'SCORED.json'), 'mothers': len(membership),
                         'clock_window_after_final_observation': [float(clock[future][0]), float(clock[future][-1])],
                         'independent_metrics': ['all signed output margins', 'physical rankings', 'Brier score', 'ungated successes',
                                                 'mean positive regret', 'future path losses', 'mother paired interval', 'upstream hashes'],
                         'scope': 'Independent recomputation from saved prediction and truth arrays. The tree model is not refitted and solver integrations are not rerun.'})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('fixture','score'))
    parser.add_argument('run', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    (fixture if args.action == 'fixture' else score)(args)


if __name__ == '__main__':
    main()
