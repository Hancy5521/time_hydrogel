"""Test matched-budget spatial and kinetic observations in a joint inverse problem."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from itertools import product, repeat
from pathlib import Path

import numpy as np
from scipy.stats import qmc

from audit_patch_reservoir_study import extremes, normalized_clauses
from gel_reactive_port_timing import read_json, sha256
from gel_reactive_two_read_assay import write_new
from patch_reservoir_joint_bank import joint_bank, prepared_fields
from reservoir_joint_observation import predictive_summary, readout_likelihoods, stock_fractions, stock_mean, updated_weights


def protocol():
    nominal = {'power': [1.26, 2.26], 'loss': [.029, .133], 'interface': [.02, .92],
               'bias': [.57, 1.10], 'shape': [.48, .82], 'precursor_diffusion_factor': [1., 1.],
               'tau_high': [.11, .22], 'tau_ratio': [1.15, 1.75], 'slow_share': [0., 0.], 'rate_jitter': .08}
    shifted = {**nominal, 'power': [1.18, 2.17], 'loss': [.052, .175], 'interface': [.52, 1.55],
               'bias': [.45, .98], 'precursor_diffusion_factor': [1.13, 1.70], 'slow_share': [.09, .31],
               'tau_ratio': [1.30, 1.95], 'rate_jitter': .13}
    return {'evidence': 'Synthetic model study. No wet measurements or empirical prior calibration.',
            'development': {'shared_and_lagged_coupons': {**nominal, 'seed': 12690187, 'order': 3}},
            'evaluation': {'joint_uncertainty': {**nominal, 'seed': 18249011, 'order': 4},
                           'interface_and_preparation_shift': {**shifted, 'seed': 18249193, 'order': 4}},
            'kinetic_grid': {'factors': [.52, 1.52], 'power': [1.24, 1.80, 2.36], 'loss': [.030, .120],
                             'interface': [0., 1.0], 'bias': [-.82, .82], 'shape': [-.67, .67]},
            'activities': {'low': .55, 'high': 1.}, 'deadline': .90, 'early_grid': [.32, .48, .64, .80],
            'tau_grid': [.09, .125, .17, .225, .29, .37, .47, .59], 'hydration_times': [.11, .24],
            'sigma': {'hydration': .016, 'total': .016, 'extra': .012, 'dry_inventory': .025},
            'coupon_loss_factor': 4., 'risk_weight': .05,
            'meshes': {'inference': [24, .010], 'development': [48, .0025], 'evaluation': [96, .00125], 'secondary': [48, .0025]},
            'service_sample_step': .04, 'process_times': [0., .8, 1.6, 2.4, 3.2, 4.8, 6.4],
            'fixed_diagnostic_action': 4, 'unseen_dry_bin': 5,
            'factorial': ['timed_total', 'timed_half', 'dual_total', 'dual_half'],
            'diagnostic': 'Dual plus fixed last-eighth slice at the same count. The measured slice is not counted as an unseen bin.',
            'controls': 'Each arm uses two sacrificed kinetic coupons, two early hydration reads on each, and one dry stock analysis on a separate companion coupon. Equal deadline and noise within each channel comparison. No service bath read during deployment.',
            'development_rule': 'Choose one same-activity early time by summed successes across the total and half arms, break ties by pooled regret then earlier time. Freeze before evaluation.',
            'test_grouping': 'Each synthetic parent has eight paired states in rate × bias × within-half order. All observations have identical shared noise across within-half twins. Four requests per state. Report parents, not independent counts per state.',
            'decision_estimands': 'Report both simple effects conditional on the other sensor and difference-in-differences. A positive simple effect does not imply a positive interaction.',
            'negative_controls': 'Half integrals and total, and both dynamic total-release assays, are blind to within-half redistribution by construction. Interface and bath loss stay independent nuisance parameters in the likelihood.',
            'process_check': 'Conditional mean and 90% equal-tail interval on three inventory trajectories and conductance of a fixed unused full-thickness diagnostic choice, plus eight dry bins with one specified unseen bin.'}


def sample_groups(task, cfg, spec):
    design = qmc.Sobol(13, scramble=True, seed=spec['seed']).random_base2(spec['order'])
    names = list(task['known_ranges'])
    material = qmc.scale(design[:, :3], *np.array(list(task['known_ranges'].values())).T)
    noise = np.random.default_rng(spec['seed'] + 51)
    keys = ('power','loss','interface','bias','shape','precursor_diffusion_factor','tau_high','tau_ratio')
    for group, (known, x) in enumerate(zip(material, design, strict=True)):
        values = {key: float(np.interp(u, [0,1], spec[key])) for key, u in zip(keys, x[3:11], strict=True)}
        values['slow_share'] = float(np.interp(x[11], [0,1], spec['slow_share']))
        # Companion read noise is paired across counterfactual methods and shape twins only.
        yield {'group': group, 'known': dict(zip(names, known, strict=True)), **values,
               'rate_factor': 1+spec['rate_jitter']*(2*x[12]-1),
               'noise': np.repeat(noise.normal(size=(4, 2, 3)), 2, axis=0),
               'dry_noise': np.repeat(noise.normal(size=4), 2)}


def parameters(group, cfg, actual):
    grid = cfg['kinetic_grid']
    rates = group['known']['release']*np.asarray(grid['factors'])
    if actual:
        rates *= group['rate_factor']
        return np.array([(k, group['power'], group['loss'], group['interface'], b, v)
                         for k, b, v in product(rates, (-group['bias'],group['bias']), (-group['shape'],group['shape']))])
    return np.array(list(product(rates, grid['power'], grid['loss'], grid['interface'], grid['bias'], grid['shape'])))


def times(task, cfg):
    dt = cfg['service_sample_step']
    return np.arange(round(task['scoring']['horizon']/dt)+1)*dt


def process_input(task, group, actual):
    mechanism = task['mechanism'].copy()
    if actual:
        mechanism['process_D'] *= group['precursor_diffusion_factor']
    return mechanism


def build_service(group, params, task, cfg, mesh, actual, diagnostics=True):
    cells, step = cfg['meshes'][mesh]
    elapsed = times(task, cfg)
    materials = process_input(task, group, actual)
    mechanical, stock = params[:, :4], params[:, 4:]
    known = group['known']
    wet = joint_bank(known, task['actions'], materials, task['bath']['wet'], mechanical, stock, cells, step, elapsed,
                     inventory=diagnostics)
    if diagnostics:
        wet, inventory = wet
    weak = joint_bank(known, task['actions'], materials, task['bath']['weak'], mechanical, stock, cells, step, elapsed)
    margin = normalized_clauses(extremes((wet, weak), elapsed, task), task).max(axis=-1)
    if not diagnostics:
        return margin, None
    action = cfg['fixed_diagnostic_action']
    index = np.rint(np.asarray(cfg['process_times'])/cfg['service_sample_step']).astype(int)
    curve = np.concatenate((inventory[:, index, action], wet[:, index, action, 3:4]), axis=-1)
    depth = stock_fractions(prepared_fields(known, [task['actions'][action]], materials, stock, cells)[:, 0])
    return margin, {'process': curve, 'dry': depth, 'G': wet[..., 3]}


def dry_probe(group, params, task, cfg, actual, cells):
    materials = process_input(task, group, actual)
    probe = [task['actions'][cfg['fixed_diagnostic_action']]]
    return stock_fractions(prepared_fields(group['known'], probe, materials, params[:, 4:], cells)[:, 0])


def designs(cfg, earliest):
    high, low = (cfg['activities'][key] for key in ('high','low'))
    deadline = cfg['deadline']
    late = (high, deadline)
    contrast = ((low,deadline), late)
    if earliest is None:
        return {f'timed_{t:.2f}': ((high,t), late) for t in cfg['early_grid']}
    return {'timed': ((high,earliest),late), 'dual': contrast}


def arm_set(protocols, earliest):
    if earliest is None:
        return [(name, channel) for name in protocols for channel in ('total','half')]
    return [(name, channel) for name in protocols for channel in ('total','half')] + [('dual','back_slice')]


def freeze_group(group, task, cfg, earliest):
    expected = parameters(group, cfg, False)
    truth = parameters(group, cfg, True)
    margin, diagnostic = build_service(group, expected, task, cfg, 'inference', False, diagnostics=earliest is not None)
    expected_stock = dry_probe(group, expected, task, cfg, False, cfg['meshes']['inference'][0])
    actual_stock = dry_probe(group, truth, task, cfg, True, cfg['meshes']['evaluation'][0])
    arms = {}
    rr = np.arange(len(task['requests']))
    positive = np.maximum(margin, 0)
    cost = (margin > 0) + cfg['risk_weight']*positive
    schedules = designs(cfg, earliest)
    available = readout_likelihoods(group, truth, expected, cfg, schedules)
    for name, channel in arm_set(schedules, earliest):
        raw = stock_mean(actual_stock, channel) + cfg['sigma']['dry_inventory']*group['dry_noise']
        kinetics, reads = available[name]
        weight = updated_weights(kinetics, stock_mean(expected_stock, channel), raw, cfg['sigma']['dry_inventory'])
        selected = np.einsum('sh,hwa->swa', weight, cost).argmin(axis=-1)
        risk = np.array([p@(margin[:, rr, a] > 0) for p, a in zip(weight, selected, strict=True)])
        record = {'selected': selected.tolist()}
        if diagnostic is not None:
            record.update(fail_probability=risk.tolist(), posterior=weight.tolist(),
                          channels={'stock': channel,'dry_value': raw.tolist(),
                                    'kinetics': [[{'a': q.activity,'stop': q.stop,'hydration': q.hydration.tolist(),'total': q.total}
                                                  for q in r] for r in reads]},
                          summary={'profile': predictive_summary(weight, diagnostic['dry']),
                                   'process': predictive_summary(weight, diagnostic['process'])})
        arms[f'{name}_{channel}'] = record
    return {'group':group['group'], 'known':group['known'], 'arms':arms}


def blind_shape_limit(ok):
    pairs = ok.reshape(-1, 2, *ok.shape[1:])
    return int(pairs.sum(axis=1).max(axis=-1).sum())


def score_group(group, frozen, task, cfg, evaluation):
    if group['group'] != frozen['group'] or group['known'] != frozen['known']:
        raise ValueError('Frozen batch context changed before objective evaluation.')
    truth = parameters(group, cfg, True)
    mesh = 'evaluation' if evaluation else 'development'
    margins, actual = build_service(group, truth, task, cfg, mesh, True, diagnostics=evaluation)
    fine = margins <= 0
    if evaluation:
        low, _ = build_service(group, truth, task, cfg, 'secondary', True, diagnostics=False)
    else:
        low = margins
    outcome = {}
    for policy, record in frozen['arms'].items():
        idx = np.array(record['selected'])[..., None]
        selected = np.take_along_axis(margins, idx, axis=-1)[..., 0]
        changed = np.take_along_axis((low <= 0) != fine, idx, axis=-1).sum()
        row = {'count': int((selected <= 0).sum()), 'success': (selected <= 0).tolist(),
               'regret': float(np.mean(np.maximum(selected,0)-np.maximum(margins.min(axis=-1),0))),
               'mesh_changes': int(changed)}
        if evaluation:
            row['brier'] = float(np.mean((np.asarray(record['fail_probability'])-(selected > 0))**2))
            for key, truth_value in (('profile',actual['dry']),('process',actual['process'])):
                summary = record['summary'][key]
                error = np.abs(np.asarray(summary['mean'])-truth_value)
                coverage = (np.asarray(summary['lo'])-1e-10 <= truth_value) & (truth_value <= np.asarray(summary['hi'])+1e-10)
                row[key+'_mae'] = error.mean(axis=0).tolist()
                row[key+'_coverage'] = coverage.mean(axis=0).tolist()
                row[key+'_interval_width'] = (np.asarray(summary['hi'])-np.asarray(summary['lo'])).mean(axis=0).tolist()
        outcome[policy] = row
    result = {'group': group['group'], 'arms': outcome}
    if evaluation:
        v = actual['dry'].reshape(-1, 2, 8)
        g = actual['G'].reshape(-1, 2, *actual['G'].shape[1:])
        result.update(oracle=int(fine.any(axis=-1).sum()), shape_blind_oracle=blind_shape_limit(fine),
                      catalogue_mesh_changes=int(((low <= 0) != fine).sum()), margins=margins.tolist(), secondary_margins=low.tolist(),
                      kinetics=truth.tolist(), truth={'dry':actual['dry'].tolist(),'process':actual['process'].tolist()},
                      twins={'max_total_difference':float(np.abs(v[:,0].sum(axis=-1)-v[:,1].sum(axis=-1)).max()),
                             'max_half_difference':float(np.abs(v[:,0,:4].sum(axis=-1)-v[:,1,:4].sum(axis=-1)).max()),
                             'max_conductance_difference_fixed_candidate':float(np.abs(g[:,0,:,cfg['fixed_diagnostic_action']]-g[:,1,:,cfg['fixed_diagnostic_action']]).max()),
                             'max_within_half_bin_difference':float(np.abs(v[:,0]-v[:,1]).max())})
    return result


def development_lock(rows, cfg):
    table = []
    for t in cfg['early_grid']:
        pairs = [group['arms'][f'timed_{t:.2f}_{c}'] for g in rows.values() for group in g for c in ('total','half')]
        table.append({'early':t,'sum_count':sum(a['count'] for a in pairs),'regret':float(np.mean([a['regret'] for a in pairs]))})
    best = min(table, key=lambda a: (-a['sum_count'], a['regret'], a['early']))
    return {'chosen':best['early'],'candidates':table}


def read_sealed(path, digest):
    data = read_json(path)
    if data['registered_sha256'] != digest:
        raise ValueError('Registration is inconsistent with the receipt.')
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('development','freeze','score'))
    parser.add_argument('run', type=Path)
    parser.add_argument('--workers', type=int, default=6)
    args = parser.parse_args()
    record = args.run/'REGISTERED.json'
    digest = sha256(record)
    cfg = read_json(record)
    for path, expected in {**cfg['sources'], **cfg['inputs']}.items():
        if sha256(Path(path)) != expected:
            raise ValueError(f'Changed registered input: {path}')
    task = read_json(Path(cfg['task_path']))
    paths = {'development': args.run/'DEVELOPMENT.json', 'freeze': args.run/'FROZEN.json', 'score': args.run/'SCORED.json'}
    if paths[args.phase].exists():
        raise FileExistsError(paths[args.phase])
    earliest = None if args.phase == 'development' else read_sealed(paths['development'],digest)['lock']['chosen']
    saved = read_sealed(paths['freeze'],digest) if args.phase == 'score' else None
    if saved is not None and saved['development_sha256'] != sha256(paths['development']):
        raise ValueError('The registered timing lock changed after freezing.')
    groups = cfg['development'] if args.phase == 'development' else cfg['evaluation']
    started = datetime.now(timezone.utc).isoformat()
    result = {}
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for name, spec in groups.items():
            cases = list(sample_groups(task,cfg,spec))
            predictions = (list(executor.map(freeze_group,cases,repeat(task),repeat(cfg),repeat(earliest)))
                           if saved is None else saved['cohorts'][name])
            if args.phase == 'freeze':
                result[name] = predictions
            else:
                result[name] = list(executor.map(score_group,cases,predictions,repeat(task),repeat(cfg),repeat(saved is not None)))
            print(name, len(result[name]), args.phase, flush=True)
    payload = {'registered_sha256':digest,'started_utc':started,'completed_utc':datetime.now(timezone.utc).isoformat(),'cohorts':result}
    if args.phase == 'development':
        payload['lock'] = development_lock(result,cfg)
    else:
        payload.update(early_time=earliest,development_sha256=sha256(paths['development']))
        if saved is not None:
            payload['frozen_sha256'] = sha256(paths['freeze'])
    write_new(paths[args.phase],payload)


if __name__ == '__main__':
    main()
