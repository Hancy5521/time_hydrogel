"""Freeze and score a dimensionless, autonomous release design experiment."""

import argparse
from itertools import product
from pathlib import Path

import numpy as np
from scipy.special import ndtri, softmax
from scipy.stats import qmc

from gel_reactive_port_timing import read_json, sha256
from gel_reactive_two_read_assay import write_new
from patch_reservoir_service import simulate, traces


def study_definition():
    return {
        'purpose': 'Does a prepared ion reservoir give autonomous design freedom and value to an early electrical read?',
        'unit_and_origin': 'dimensionless service time; zero is contact with a constant bath after rinse and drying',
        'evidence_kind': 'synthetic parameter and model-shift challenge; no experimental time or units inferred',
        'mechanism': {'process_D': .035, 'process_k': 1.6, 'process_time': 1.2, 'dose': 1.1,
                      'mobile_D': .025, 'bond_off': .85, 'bond_on': .32,
                      'dry_conductivity': .022, 'water_background': .026, 'ion_conductivity': 1.7},
        'actions': [dict(wet_transfer=wet, near_loading=share)
                    for wet, share in product((.22, .50, 1.00), (.10, .50, .90))],
        'requests': [dict(install_end=left, work_start=right, work_end=right+1.6)
                     for left, right in product((1., 1.8), (3.2, 4.8))],
        'thresholds': {'off': .063, 'on': .18, 'soft': .79, 'support': .91, 'weak_off': .085},
        'scales': {'off': .063, 'on': .18, 'soft': .25, 'support': .25, 'weak_off': .085},
        'bath': {'wet': 1., 'weak': .32},
        'coupon': {'time': 1.2, 'thickness_scale': .55, 'wet_transfer': .66, 'near_loading': .5,
                   'observation_sd': [.015, .009]},
        'nominal': {'cells': 20, 'dt': .020, 'power': 2., 'loss': .025},
        'scoring': {'cells': 32, 'dt': .005, 'sample_dt': .04, 'horizon': 6.4},
        'cohorts': {'matched_kinetics': {'seed': 113221, 'order': 4, 'power': 2., 'loss': (.012, .045)},
                    'faster_wet_release_and_leaching': {'seed': 928769, 'order': 4, 'power': 1.35,
                                                        'loss': (.045, .085)}},
        'known_ranges': {'water_D': [.078, .155], 'release': [.62, 1.18], 'thickness': [.82, 1.14]},
        'hidden_factors': [.52, 1.52],
        'state_shift': .10,
        'risk': 'Expected failure plus 0.05 times worst positive normalized threshold margin. Equal prior.',
        'decisive_gap_for_order': .08,
        'order_mid_action': 4,
        'claims_before_scoring': ['Same stock for every loading arrangement and a fixed wet boundary.',
                                  'A monotone common time warp preserves strict event order. Test both crossing orders at the same fixed action and thresholds.',
                                  'Use the extrema on every sampled install/work time. Conductivity can decrease.',
                                  'Water-only oracle ceiling conditions on known batch context but no hidden release state.',
                                  'No fit, target-dependent selection of requests, or repeat evaluation with modified thresholds.'],
    }


def cohorts(cfg, name):
    setting = cfg['cohorts'][name]
    points = qmc.Sobol(9, scramble=True, seed=setting['seed']).random_base2(setting['order'])
    keys = list(cfg['known_ranges'])
    material = qmc.scale(points[:, :len(keys)], *np.array(list(cfg['known_ranges'].values())).T)
    for index, (values, raw) in enumerate(zip(material, points, strict=True)):
        yield {'group': index, 'known': dict(zip(keys, values, strict=True)),
               'loss': np.interp(raw[3], [0, 1], setting['loss']),
               'release_multiplier': 1 + cfg['state_shift']*(2*raw[4]-1),
               'noise': ndtri(raw[5:9]).reshape(2, 2)}


def state(known, factor, multiplier):
    return {**known, 'release': known['release'] * factor * multiplier}


def times_for(cfg, refinement=1):
    evaluation = cfg['scoring']
    step = evaluation['sample_dt']/refinement
    return np.arange(round(evaluation['horizon']/step) + 1)*step


def plan_traces(parent, actions, cfg, numerical, bath, times):
    fields = simulate(parent, actions, cfg['mechanism'], bath, numerical['power'], numerical['loss'],
                      numerical['cells'], numerical['dt'], times)
    return traces(fields, cfg['mechanism'])


def coupon(parent, cfg, numerical):
    spec = cfg['coupon']
    small = {**parent, 'thickness': parent['thickness']*spec['thickness_scale']}
    result = plan_traces(small, [spec], cfg, {**numerical, 'dt': numerical['dt']/2}, cfg['bath']['wet'],
                         [0., spec['time']])
    return result[-1, 0, [0, 3]]


def clauses(wet, weak, times, cfg):
    h = cfg['thresholds']
    s = cfg['scales']
    values = []
    for goal in cfg['requests']:
        installed = times <= goal['install_end'] + 1e-12
        during = (times >= goal['work_start'] - 1e-12) & (times <= goal['work_end'] + 1e-12)
        until_end = times <= goal['work_end'] + 1e-12
        values.append(np.stack(((wet[installed, :, 3].max(axis=0)-h['off'])/s['off'],
                                (h['on']-wet[during, :, 3].min(axis=0))/s['on'],
                                (wet[during, :, 1].max(axis=0)-h['soft'])/s['soft'],
                                (h['support']-wet[installed, :, 2].min(axis=0))/s['support'],
                                (weak[until_end, :, 3].max(axis=0)-h['weak_off'])/s['weak_off']), axis=-1))
    return np.stack(values)


def banks(group, cfg, numerical, actual):
    times = times_for(cfg)
    wet, weak = [], []
    multiplier = group['release_multiplier'] if actual else 1.
    for factor in cfg['hidden_factors']:
        material = state(group['known'], factor, multiplier)
        wet.append(plan_traces(material, cfg['actions'], cfg, numerical, cfg['bath']['wet'], times))
        weak.append(plan_traces(material, cfg['actions'], cfg, numerical, cfg['bath']['weak'], times))
    return np.array(wet), np.array(weak)


def evaluate_clauses(wet, weak, times, cfg):
    return np.stack([clauses(a, b, times, cfg) for a, b in zip(wet, weak, strict=True)])


def evaluation_numerics(cfg, name, group):
    return {**cfg['scoring'], 'power': cfg['cohorts'][name]['power'], 'loss': group['loss']}


def uniform_actions(cfg):
    return np.array([i for i, action in enumerate(cfg['actions']) if action['near_loading'] == .5])


def choose(weights, cost):
    return np.einsum('qs,swa->qwa', weights, cost).argmin(axis=-1)


def freeze(cfg):
    times = times_for(cfg)
    output = {}
    for name in cfg['cohorts']:
        rows = []
        for group in cohorts(cfg, name):
            actual = evaluation_numerics(cfg, name, group)
            expected = [coupon(state(group['known'], f, 1.), cfg, cfg['nominal']) for f in cfg['hidden_factors']]
            measured = [coupon(state(group['known'], f, group['release_multiplier']), cfg, actual)
                        for f in cfg['hidden_factors']]
            sd = np.array(cfg['coupon']['observation_sd'])
            observed = np.asarray(measured) + group['noise']*sd
            residual = (observed[:, None, :] - np.asarray(expected)[None, :, :])/sd
            probability = softmax(-.5*residual**2, axis=1)
            wet, weak = banks(group, cfg, cfg['nominal'], False)
            margin = evaluate_clauses(wet, weak, times, cfg).max(axis=-1)
            cost = (margin > 0) + .05*np.maximum(margin, 0)
            uniform = uniform_actions(cfg)
            rows.append({'group': group['group'], 'known': group['known'], 'observed_coupon': observed.tolist(),
                         'expected_coupon': np.array(expected).tolist(), 'posterior_probabilities': probability.tolist(),
                         'choices': {'water_full_state': choose(probability[..., 0], cost).tolist(),
                                     'electrical_full_state': choose(probability[..., 1], cost).tolist(),
                                     'electrical_uniform_only': uniform[choose(probability[..., 1], cost[..., uniform])].tolist()}})
        output[name] = rows
        print(name, len(rows), 'frozen paired batch contexts', flush=True)
    return output


def crossing(times, values, threshold):
    reached = np.flatnonzero(values >= threshold)
    if not len(reached):
        return None
    k = reached[0]
    if k == 0:
        return float(times[k])
    fraction = (threshold-values[k-1])/(values[k]-values[k-1])
    return float(times[k-1] + (times[k]-times[k-1])*fraction)


def order_witness(wet, times, cfg):
    traces_by_state = wet[:, :, cfg['order_mid_action'], :]
    limits = cfg['thresholds']
    soft_time = crossing(times, -traces_by_state[0, :, 1], -limits['soft'])
    on = [crossing(times, row[:, 3], limits['on']) for row in traces_by_state]
    gap = cfg['decisive_gap_for_order']
    reversed_order = bool(soft_time is not None and None not in on and on[1]+gap < soft_time < on[0]-gap)
    return {'soft': soft_time, 'on': on, 'opposite_order': reversed_order,
            'equal_hydration_and_mechanics_difference': float(np.abs(traces_by_state[0, :, :3] - traces_by_state[1, :, :3]).max()),
            'max_conductivity_separation': float(np.abs(traces_by_state[0, :, 3] - traces_by_state[1, :, 3]).max())}


def score(cfg, frozen):
    times = times_for(cfg)
    output, raw = {}, {}
    for name, inputs in frozen.items():
        output[name], matrices = [], []
        for row, group in zip(inputs, cohorts(cfg, name), strict=True):
            if row['group'] != group['group'] or row['known'] != group['known']:
                raise ValueError('Unexpected group membership after prediction freeze.')
            wet, weak = banks(group, cfg, evaluation_numerics(cfg, name, group), True)
            all_clauses = evaluate_clauses(wet, weak, times, cfg)
            margin = all_clauses.max(axis=-1)
            passed = margin <= 0
            best = np.maximum(margin.min(axis=-1), 0)
            individual = {}
            for policy, actions in row['choices'].items():
                chosen = np.take_along_axis(margin, np.array(actions)[..., None], axis=-1)[..., 0]
                individual[policy] = {'selected': actions, 'pass': (chosen <= 0).tolist(), 'margin': chosen.tolist(),
                                      'regret': (np.maximum(chosen, 0)-best).tolist()}
            ceilings = passed.any(axis=-1).sum(axis=0)
            shared = passed.sum(axis=0).max(axis=-1)
            output[name].append({'group': group['group'], 'fully_informed_count': ceilings.tolist(),
                                 'shared_count': shared.tolist(), 'choices_scored': individual,
                                 'uniform_fully_informed_count': passed[..., uniform_actions(cfg)].any(axis=-1).sum(axis=0).tolist(),
                                 'event_order': order_witness(wet, times, cfg), 'margins': margin.tolist()})
            matrices.append(np.stack((wet, weak)))
        raw[name] = np.array(matrices)
        print(name, len(output[name]), 'groups scored', flush=True)
    return output, raw


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('freeze', 'score'))
    parser.add_argument('run', type=Path)
    args = parser.parse_args()
    plan = args.run/'REGISTERED.json'
    cfg = read_json(plan)
    for source, digest in cfg['sources'].items():
        if sha256(Path(__file__).with_name(source)) != digest:
            raise ValueError(f'Changed source after registration: {source}')
    frozen = args.run/'FROZEN.json'
    if args.phase == 'freeze':
        if frozen.exists():
            raise FileExistsError(frozen)
        write_new(frozen, {'registered_sha256': sha256(plan), 'predictions': freeze(cfg)})
        return
    receipt = read_json(frozen)
    if receipt['registered_sha256'] != sha256(plan):
        raise ValueError('Registration changed after freeze.')
    target = args.run/'SCORED.json'
    trajectories = args.run/'TRAJECTORIES.npz'
    if target.exists() or trajectories.exists():
        raise FileExistsError('Scoring outputs already exist.')
    rows, raw = score(cfg, receipt['predictions'])
    with trajectories.open('xb') as file:
        np.savez_compressed(file, time=times_for(cfg), **raw)
    write_new(target, {'registered_sha256': sha256(plan), 'frozen_sha256': sha256(frozen),
                      'trajectories_sha256': sha256(trajectories), 'cohorts': rows})


if __name__ == '__main__':
    main()
