"""Audit the frozen autonomous reservoir study and refine its boundary cases."""

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
from scipy.stats import qmc

from audit_patch_reservoir_service import independent_curve
from patch_reservoir_service import simulate


CLAUSE_NAMES = ('installation_leak', 'work_conductance', 'work_softness',
                'installation_support', 'weak_environment_leak')
CHANNELS = ('hydration', 'near_stiffness', 'far_stiffness', 'conductivity')


def digest(path):
    with path.open('rb') as source:
        from hashlib import file_digest
        return file_digest(source, sha256).hexdigest()


def read_receipts(run):
    paths = {name: run / name for name in ('REGISTERED.json', 'FROZEN.json', 'SCORED.json', 'TRAJECTORIES.npz')}
    hashes = {name: digest(path) for name, path in paths.items()}
    cfg, frozen, scored = (json.loads(paths[name].read_text(encoding='utf-8-sig'))
                           for name in ('REGISTERED.json', 'FROZEN.json', 'SCORED.json'))
    if (frozen['registered_sha256'] != hashes['REGISTERED.json']
            or scored['registered_sha256'] != hashes['REGISTERED.json']
            or scored['frozen_sha256'] != hashes['FROZEN.json']
            or scored['trajectories_sha256'] != hashes['TRAJECTORIES.npz']):
        raise ValueError('Registered, frozen and scored input digests do not agree.')
    for source, expected in cfg['sources'].items():
        if digest(Path(__file__).with_name(source)) != expected:
            raise ValueError(f'Registered source differs: {source}')
    with np.load(paths['TRAJECTORIES.npz'], allow_pickle=False) as data:
        curves = {name: data[name] for name in data.files}
    return cfg, frozen['predictions'], scored['cohorts'], curves, hashes


def extremes(curves, times, cfg):
    """Return raw extrema [state, request, action, clause] on closed windows."""
    wet, weak = curves
    windows = []
    for goal in cfg['requests']:
        start = np.searchsorted(times, goal['work_start'] - 1e-10)
        end = np.searchsorted(times, goal['work_end'] + 1e-10)
        install = np.searchsorted(times, goal['install_end'] + 1e-10)
        if not np.allclose([times[install - 1], times[start], times[end - 1]],
                           [goal['install_end'], goal['work_start'], goal['work_end']], rtol=0, atol=1e-10):
            raise ValueError(f'Missing time-window boundary: {goal}')
        windows.append(np.stack([np.amax(wet[:, :install, :, 3], axis=1),
                                 np.amin(wet[:, start:end, :, 3], axis=1),
                                 np.amax(wet[:, start:end, :, 1], axis=1),
                                 np.amin(wet[:, :install, :, 2], axis=1),
                                 np.amax(weak[:, :end, :, 3], axis=1)], axis=-1))
    return np.stack(windows, axis=1)


def normalized_clauses(raw, cfg):
    keys = ('off', 'on', 'soft', 'support', 'weak_off')
    target = np.array([cfg['thresholds'][key] for key in keys])
    scale = np.array([cfg['scales'][key] for key in keys])
    return (raw - target) * np.array([1, -1, 1, -1, 1]) / scale


def first_crossing(times, values, level):
    if values[0] >= level:
        return float(times[0])
    edges = np.flatnonzero((values[:-1] < level) & (values[1:] >= level))
    if not edges.size:
        return None
    k = edges[0]
    return float(np.interp(level, values[k:k + 2], times[k:k + 2]))


def events(wet, times, cfg):
    two = wet[:, :, cfg['order_mid_action']]
    limit = cfg['thresholds']
    soft = first_crossing(times, -two[0, :, 1], -limit['soft'])
    on = [first_crossing(times, trace[:, 3], limit['on']) for trace in two]
    gaps = [float(t - soft) if t is not None and soft is not None else None for t in on]
    decisive = cfg['decisive_gap_for_order']
    opposite = None not in gaps and gaps[0] > decisive and gaps[-1] < -decisive
    return {'soft': soft, 'on': on, 'signed_on_minus_soft': gaps, 'opposite_order': opposite,
            'equal_hydration_and_mechanics_difference': float(np.max(np.abs(two[0, :, :3] - two[1, :, :3]))),
            'max_conductivity_separation': float(np.max(np.abs(two[0, :, 3] - two[1, :, 3])))}


def assert_close(a, b, description):
    if not np.allclose(a, b, rtol=0, atol=2e-12):
        raise ArithmeticError(f'Disagreement in {description}')


def check_scores(curves, times, cfg, freeze, score):
    raw = extremes(curves, times, cfg)
    clauses = normalized_clauses(raw, cfg)
    margin = clauses.max(axis=-1)
    passed = margin <= 0
    individual = passed.any(axis=-1).sum(axis=0)
    shared = passed.sum(axis=0).max(axis=-1)
    uniform_ids = [i for i, action in enumerate(cfg['actions']) if action['near_loading'] == .5]
    uniform = passed[..., uniform_ids].any(axis=-1).sum(axis=0)
    assert_close(margin, score['margins'], 'catalogue margins')
    assert_close(individual, score['fully_informed_count'], 'full-information ceiling')
    assert_close(shared, score['shared_count'], 'water-only ceiling')
    assert_close(uniform, score['uniform_fully_informed_count'], 'uniform-loading ceiling')

    differences = (np.asarray(freeze['observed_coupon'])[:, None, :]
                   - np.asarray(freeze['expected_coupon'])[None, :, :]) / cfg['coupon']['observation_sd']
    log_density = -differences**2 / 2
    weight = np.exp(log_density - log_density.max(axis=1, keepdims=True))
    posterior = weight / weight.sum(axis=1, keepdims=True)
    assert_close(posterior, freeze['posterior_probabilities'], 'scalar-readout probabilities')
    assert_close(posterior[0, :, 0], posterior[1, :, 0], 'water-only hidden-state information')
    policies = {}
    for policy, selected in freeze['choices'].items():
        actual = score['choices_scored'][policy]
        assert_close(selected, actual['selected'], f'{policy} action IDs')
        if policy == 'water_full_state':
            assert_close(selected[0], selected[1], 'common water-only decision')
        if policy == 'electrical_uniform_only' and not np.isin(selected, uniform_ids).all():
            raise ArithmeticError('Uniform-only policy used a layered loading action.')
        chosen = np.array([[margin[state, j, a] for j, a in enumerate(row)]
                           for state, row in enumerate(selected)])
        regret = np.maximum(chosen, 0.) - np.maximum(margin.min(axis=-1), 0.)
        assert_close(chosen, actual['margin'], f'{policy} scored margin')
        assert_close(chosen <= 0, actual['pass'], f'{policy} pass labels')
        assert_close(regret, actual['regret'], f'{policy} positive regret')
        policies[policy] = {'successes': int((chosen <= 0).sum()), 'positive_regret_sum': float(regret.sum())}

    observed_events = events(curves[0], times, cfg)
    for name, value in score['event_order'].items():
        candidate = observed_events[name]
        if name == 'on':
            for measured, expected in zip(candidate, value, strict=True):
                if measured is None or expected is None:
                    if measured is not expected:
                        raise ArithmeticError('Crossing presence differs.')
                else:
                    assert_close(measured, expected, 'threshold crossing time')
        elif candidate is None or value is None:
            if candidate is not value:
                raise ArithmeticError('Crossing presence differs.')
        else:
            assert_close(candidate, value, name)
    return {'fully_informed_count': int(individual.sum()), 'water_ceiling': int(shared.sum()),
            'uniform_ceiling': int(uniform.sum()), 'policies': policies, 'events': observed_events}, raw, margin


def material_settings(cfg, cohort, rows):
    spec = cfg['cohorts'][cohort]
    design = qmc.Sobol(9, scramble=True, seed=spec['seed']).random_base2(spec['order'])
    names = list(cfg['known_ranges'])
    bounds = np.array([cfg['known_ranges'][name] for name in names])
    for index, (u, row) in enumerate(zip(design, rows, strict=True)):
        if row['group'] != index:
            raise ValueError('A frozen group is out of order.')
        known = bounds[:, 0] + np.diff(bounds, axis=1)[:, 0] * u[:len(names)]
        assert_close(known, [row['known'][name] for name in names], 'known model inputs')
        loss = spec['loss'][0] + np.diff(spec['loss'])[0]*u[3]
        multiplier = 1 + cfg['state_shift']*(2*u[4] - 1)
        models = [{**row['known'], 'release': row['known']['release']*factor*multiplier}
                  for factor in cfg['hidden_factors']]
        yield models, spec['power'], loss


def project_fields(fields, mechanism):
    h, b, c = (fields[key] for key in ('h', 'b', 'c'))
    cells = h.shape[-1]
    stiffness = (0.27 + 0.93*b) * (1 - 0.27*h)
    sigma = mechanism['dry_conductivity'] + h*(mechanism['water_background'] + mechanism['ion_conductivity']*c)
    return np.stack((h.mean(axis=-1), stiffness[..., :cells//2].mean(axis=-1),
                     stiffness[..., cells//2:].mean(axis=-1), cells/np.sum(1/sigma, axis=-1)), axis=-1)


def physical(field, thickness, spec):
    expected = thickness*spec['dose']*(1 - np.exp(-spec['process_k']*spec['process_time']))
    current = thickness*(field['r'] + field['c']).mean(axis=-1) + field['lost']
    return {'ion_budget_error': float(np.max(np.abs(current - expected))),
            'minimum_concentration': float(min(field['r'].min(), field['c'].min())),
            'largest_water_decrease': float(max(0, -np.diff(field['h'], axis=0).min())),
            'largest_bond_increase': float(max(0, np.diff(field['b'], axis=0).max())),
            'largest_reservoir_increase': float(max(0, np.diff(field['r'], axis=0).max())),
            'largest_loss_decrease': float(max(0, -np.diff(field['lost'], axis=0).min()))}


def audit_same_mesh(curves, models, power, loss, times, cfg):
    ids = [i for i, a in enumerate(cfg['actions']) if a['wet_transfer'] == .50 and a['near_loading'] in (.10, .90)]
    if len(ids) != 2:
        raise ValueError('Missing fixed middle-transfer extreme loading pair.')
    cells, dt = (cfg['scoring'][key] for key in ('cells', 'dt'))
    spec = cfg['mechanism']
    log = []
    for env, bath in enumerate((cfg['bath']['wet'], cfg['bath']['weak'])):
        for state, parent in enumerate(models):
            actions = [cfg['actions'][i] for i in ids]
            main = simulate(parent, actions, spec, bath, power, loss, cells, dt, times)
            for col, (index, action) in enumerate(zip(ids, actions, strict=True)):
                reference = independent_curve(parent, action, spec, times, bath, cells, power, loss)
                production = {key: value[:, col] for key, value in main.items()}
                line = project_fields(reference, spec)
                assert_close(curves[env, state, :, index], project_fields(production, spec), 'frozen trace regeneration')
                trace_error = np.max(np.abs(line - curves[env, state, :, index]), axis=0)
                log.append({'environment': ('wet', 'weak')[env], 'state': state, 'action': index,
                            'channel_max_abs_error': dict(zip(CHANNELS, map(float, trace_error), strict=True)),
                            'field_max_abs_error': {key: float(np.max(np.abs(value - production[key])))
                                                    for key, value in reference.items()},
                            'radau_physical': physical(reference, parent['thickness'], spec),
                            'primary_physical': physical(production, parent['thickness'], spec)})
    return log


def refine(models, power, loss, cfg, fine_times):
    cells = 2*cfg['scoring']['cells']
    dt = cfg['scoring']['dt']/2
    spec = cfg['mechanism']
    return np.asarray([[project_fields(simulate(parent, cfg['actions'], spec, bath, power, loss,
                                               cells, dt, fine_times), spec) for parent in models]
                       for bath in (cfg['bath']['wet'], cfg['bath']['weak'])])


def audit(run):
    cfg, frozen, scored, curves, hashes = read_receipts(run)
    started = datetime.now(timezone.utc).isoformat()
    times = curves['time']
    fine_times = np.linspace(0., times[-1], 2*len(times) - 1)
    assert_close(fine_times[::2], times, 'half grid nests the frozen clocks')
    details = {}
    for cohort in cfg['cohorts']:
        cohort_log = {'score_readback': [], 'refinement': [], 'same_mesh_radau': []}
        matched = material_settings(cfg, cohort, frozen[cohort])
        for index, (stored, prior, scored_row, conditioning) in enumerate(
                zip(curves[cohort], frozen[cohort], scored[cohort], matched, strict=True)):
            if scored_row['group'] != index:
                raise ValueError('A scored group is out of order.')
            reports, old_raw, old_margin = check_scores(stored, times, cfg, prior, scored_row)
            cohort_log['score_readback'].append(reports)
            if index < 2:
                print(cohort, index, 'independent Radau', flush=True)
                checks = audit_same_mesh(stored, *conditioning, times, cfg)
                cohort_log['same_mesh_radau'].append({'group': index, 'comparisons': checks})
            near = float(np.min(np.abs(old_margin)))
            if index >= 2 and near >= .01:
                continue
            print(cohort, index, '64-cell refinement', flush=True)
            refined = refine(*conditioning, cfg, fine_times)
            refined_raw = extremes(refined, fine_times, cfg)
            refined_shared_grid = extremes(refined[:, :, ::2], times, cfg)
            new_margin = normalized_clauses(refined_raw, cfg).max(axis=-1)
            changed = np.argwhere((old_margin <= 0) != (new_margin <= 0))
            refined_events = events(refined[0], fine_times, cfg)
            flipped = [{'state': int(state), 'request': int(window), 'action': int(action),
                        'frozen_margin': float(old_margin[state, window, action]),
                        'refined_margin': float(new_margin[state, window, action])}
                       for state, window, action in changed]
            selection_flips = {p: sum((old_margin[s, w, a] <= 0) != (new_margin[s, w, a] <= 0)
                                      for s, row in enumerate(actions) for w, a in enumerate(row))
                               for p, actions in prior['choices'].items()}
            cohort_log['refinement'].append({'group': index, 'minimum_frozen_absolute_margin': near,
                'max_trace_change_by_channel': dict(zip(CHANNELS, map(float, np.max(np.abs(stored-refined[:, :, ::2]), axis=(0, 1, 2, 3))), strict=True)),
                'max_raw_extrema_change_by_clause': dict(zip(CLAUSE_NAMES, map(float, np.max(np.abs(old_raw-refined_raw), axis=(0, 1, 2))), strict=True)),
                'max_new_sampling_extrema_change_by_clause': dict(zip(CLAUSE_NAMES, map(float, np.max(np.abs(refined_shared_grid-refined_raw), axis=(0, 1, 2))), strict=True)),
                'max_normalized_margin_change': float(np.max(np.abs(old_margin-new_margin))),
                'label_flips': flipped, 'selected_policy_label_flips': {k: int(v) for k,v in selection_flips.items()},
                'event_order': refined_events})
        details[cohort] = cohort_log

    comparisons = [check for cohort in details.values() for batch in cohort['same_mesh_radau']
                   for check in batch['comparisons']]
    trace_error = {c: max(row['channel_max_abs_error'][c] for row in comparisons) for c in CHANNELS}
    return {'started_utc': started, 'completed_utc': datetime.now(timezone.utc).isoformat(),
            'source_hashes': {Path(__file__).name: digest(Path(__file__)), **cfg['sources']},
            'input_hashes': hashes, 'same_mesh_trace_limit': 3e-5,
            'same_mesh_trace_pass': max(trace_error.values()) <= 3e-5, 'same_mesh_trace_max_errors': trace_error,
            'notes': 'Closed windows on sampled grids. Refined labels supplement the frozen primary labels.',
            'cohorts': details}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    args = parser.parse_args()
    output = args.run/'AUDIT.json'
    if output.exists():
        raise FileExistsError(output)
    result = audit(args.run)
    with output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print('Same-mesh Radau max trace errors:', result['same_mesh_trace_max_errors'], flush=True)


if __name__ == '__main__':
    main()
