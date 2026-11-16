"""Independently read back the frozen joint reservoir study without its simulator."""

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


TOL = 3e-12
FACTORIAL_CONTRASTS = {
    'dual_at_total': {'dual_total': 1, 'timed_total': -1},
    'dual_at_half': {'dual_half': 1, 'timed_half': -1},
    'half_at_timed': {'timed_half': 1, 'timed_total': -1},
    'half_at_dual': {'dual_half': 1, 'dual_total': -1},
    'joint_vs_baseline': {'dual_half': 1, 'timed_total': -1},
    'interaction': {'dual_half': 1, 'timed_half': -1, 'dual_total': -1, 'timed_total': 1},
    'slice_vs_half': {'dual_back_slice': 1, 'dual_half': -1},
}


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def close(actual, expected, label):
    if not np.allclose(actual, expected, rtol=0, atol=TOL):
        raise ArithmeticError(f'Disagreement in {label}.')


def load_receipts(run):
    filenames = ('REGISTERED.json', 'DEVELOPMENT.json', 'FROZEN.json', 'SCORED.json')
    index = {name: digest(run/name) for name in filenames}
    registration, development, frozen, scored = [read(run/name) for name in filenames]
    for stage in (development, frozen, scored):
        if stage['registered_sha256'] != index['REGISTERED.json']:
            raise ValueError('A study receipt references a different registration.')
    for stage in (frozen, scored):
        if stage['development_sha256'] != index['DEVELOPMENT.json']:
            raise ValueError('A confirmation receipt references a different timing lock.')
    if scored['frozen_sha256'] != index['FROZEN.json']:
        raise ValueError('The prediction receipt differs from the one evaluated.')
    chronology = [registration['registered_utc']]
    chronology.extend(item[key] for item in (development, frozen, scored) for key in ('started_utc', 'completed_utc'))
    stamps = [datetime.fromisoformat(item) for item in chronology]
    if stamps != sorted(stamps):
        raise ValueError('Registration, development, freezing and evaluation are out of order.')
    all_seeds = [spec['seed'] for split in (registration['development'], registration['evaluation'])
                 for spec in split.values()]
    if len(set(all_seeds)) != len(all_seeds):
        raise ValueError('Development and evaluation cohorts reuse a seed.')
    for cohort, spec in registration['evaluation'].items():
        expected_count = 1 << spec['order']
        for stage in (frozen, scored):
            if len(stage['cohorts'][cohort]) != expected_count:
                raise ValueError('Incorrect number of registered evaluation parents.')
    for relative, expected in {**registration['sources'], **registration['inputs']}.items():
        if digest(Path(relative)) != expected:
            raise ValueError(f'A registered source or input checksum changed: {relative}')
    return registration, development, frozen, scored, index


def verify_development(reg, dev, frozen, scored):
    results = []
    for time in reg['early_grid']:
        rows = [item['arms'][f'timed_{time:.2f}_{probe}'] for batch in dev['cohorts'].values()
                for item in batch for probe in ('half', 'total')]
        for row in rows:
            if sum(map(sum, row['success'])) != row['count']:
                raise ArithmeticError('Invalid count in timing development receipt.')
        results.append((time, sum(r['count'] for r in rows), sum(r['regret'] for r in rows)/len(rows)))
    chosen = min(results, key=lambda row: (-row[1], row[2], row[0]))[0]
    decision = dev['lock']
    close(results, [(row['early'], row['sum_count'], row['regret']) for row in decision['candidates']],
          'timing candidate scores')
    close([decision['chosen'], frozen['early_time'], scored['early_time']], chosen, 'frozen early timing')
    return chosen, results


def shape_pair_delta(values):
    a = np.asarray(values)
    return float(np.abs(a[::2]-a[1::2]).max())


def verify_summary(record, scored, truth, key):
    mean = np.asarray(record['mean'])
    low = np.asarray(record['lo'])
    high = np.asarray(record['hi'])
    if (mean.shape != truth.shape or low.shape != truth.shape or high.shape != truth.shape or
            not np.isfinite(mean).all() or not np.isfinite(low).all() or not np.isfinite(high).all() or
            (low > high).any() or (low < -TOL).any()):
        raise ArithmeticError(f'Invalid {key} summary.')
    if key == 'profile':
        close(mean.sum(axis=-1), 1, 'posterior mean stock normalization')
    if key == 'process':
        close(mean[..., :3].sum(axis=-1), 1, 'posterior mean service mass conservation')
        if (np.diff(mean[:, :, 0], axis=-1) > TOL).any() or (np.diff(mean[:, :, 2], axis=-1) < -TOL).any():
            raise ArithmeticError('Nonphysical monotonicity in predicted retained/exported inventory.')
    error = np.mean(np.abs(mean-truth), axis=0)
    fraction = ((low-1e-10 <= truth) & (truth <= high+1e-10)).mean(axis=0)
    width = (high-low).mean(axis=0)
    close(error, scored[key+'_mae'], key+' MAE')
    close(fraction, scored[key+'_coverage'], key+' coverage')
    close(width, scored[key+'_interval_width'], key+' interval width')
    return {'mae': error.tolist(), 'coverage': fraction.tolist(), 'width': width.tolist()}


def verify_arm(name, forecast, outcome, fine, coarse, raw, sizes, reg, early_time):
    indices = np.asarray(forecast['selected'])
    expected_shape = fine.shape[:2]
    if indices.shape != expected_shape or indices.min() < 0 or indices.max() >= fine.shape[-1]:
        raise ArithmeticError('Invalid selection indices.')
    actual = np.array([[fine[state, window, index] for window, index in enumerate(row)]
                       for state, row in enumerate(indices)])
    secondary = np.array([[coarse[state, window, index] for window, index in enumerate(row)]
                          for state, row in enumerate(indices)])
    regret = np.mean(np.maximum(actual, 0) - np.maximum(fine.min(axis=-1), 0))
    probability = np.asarray(forecast['fail_probability'])
    if probability.shape != expected_shape or probability.min() < -TOL or probability.max() > 1+TOL:
        raise ArithmeticError('Invalid failure-probability summary.')
    brier = np.mean((probability-(actual > 0))**2)
    close(actual <= 0, outcome['success'], name+' pass labels')
    close([(actual <= 0).sum(), regret, ((actual <= 0) != (secondary <= 0)).sum(), brier],
          [outcome['count'], outcome['regret'], outcome['mesh_changes'], outcome['brier']], name+' score')
    dry_sensor = np.asarray(forecast['channels']['dry_value'])
    kinetics = forecast['channels']['kinetics']
    schedule = [(reg['activities']['high'], early_time), (reg['activities']['high'], reg['deadline'])]
    if name.startswith('dual'):
        schedule[0] = (reg['activities']['low'], reg['deadline'])
    for pair in kinetics:
        close([(readout['a'], readout['stop']) for readout in pair], schedule, name+' readout schedule')
        if any(len(readout['hydration']) != len(reg['hydration_times']) for readout in pair):
            raise ArithmeticError('Unexpected hydration read count.')
    if kinetics[::2] != kinetics[1::2]:
        raise ArithmeticError('Kinetic observations differ between within-half twins.')
    tensor = np.asarray(forecast['posterior']).reshape(len(indices), *sizes)
    close(tensor.reshape(len(indices), -1).sum(axis=-1), 1, 'joint posterior normalization')
    if tensor.min() < 0:
        raise ArithmeticError('Negative posterior mass.')
    marginals = {}
    for axis, parameter in ((3, 'loss'), (4, 'interface'), (6, 'within_half')):
        keep = tensor.sum(axis=tuple(i for i in range(tensor.ndim) if i not in (0, axis)))
        marginals[parameter] = keep
        if parameter != 'within_half' or name != 'dual_back_slice':
            close(keep, 1/keep.shape[-1], name+' prior invariance for '+parameter)
    report = {'count': int((actual <= 0).sum()), 'pass': (actual <= 0).tolist(),
              'brier': float(brier), 'regret': float(regret),
              'mesh_changes': int(((actual <= 0) != (secondary <= 0)).sum()),
              'dry_twin_difference': shape_pair_delta(dry_sensor),
              'posterior_twin_difference': shape_pair_delta(tensor),
              'shape_probability_shift': float(np.abs(marginals['within_half']-.5).mean()),
              'shape_twin_choice_differences': int((indices[::2] != indices[1::2]).sum()),
              'max_loss_prior_difference': float(np.abs(marginals['loss']-.5).max()),
              'max_interface_prior_difference': float(np.abs(marginals['interface']-.5).max())}
    if name != 'dual_back_slice':
        close([report['dry_twin_difference'], report['posterior_twin_difference']], 0,
              name+' indistinguishable twin observations and posterior')
    for key, targets in (('profile', raw['dry']), ('process', raw['process'])):
        report[key] = verify_summary(forecast['summary'][key], outcome, np.asarray(targets), key)
    return report


def verify_group(pred, score, reg, early_time):
    if pred['group'] != score['group']:
        raise ArithmeticError('Forecast and score are assigned to different parents.')
    a, b = (np.asarray(score[key]) for key in ('margins', 'secondary_margins'))
    passed = a <= 0
    oracle = int(sum(any(choices) for state in passed for choices in state))
    shape_blind = int(sum(max(x.astype(int)+y) for pair in zip(passed[::2], passed[1::2], strict=True)
                          for x, y in zip(*pair, strict=True)))
    wholly_blind = int(passed.sum(axis=0).max(axis=-1).sum())
    close([oracle, shape_blind, ((a <= 0) != (b <= 0)).sum()],
          [score['oracle'], score['shape_blind_oracle'], score['catalogue_mesh_changes']], 'oracle caps and grid labels')
    # Dry bins are a disjoint partition. Process channels are remaining, mobile, bath, conductance.
    stock = np.asarray(score['truth']['dry'])
    history = np.asarray(score['truth']['process'])
    close(stock.sum(axis=-1), 1, 'realized stock normalization')
    close(history[:, :, :3].sum(axis=-1), 1, 'realized service inventory closure')
    if ((history < -TOL).any() or (stock < -TOL).any() or (np.diff(history[:, :, 0], axis=-1) > TOL).any()
            or (np.diff(history[:, :, 2], axis=-1) < -TOL).any()):
        raise ArithmeticError('Nonphysical realized stock trajectory.')
    close([shape_pair_delta(stock.sum(axis=-1)), shape_pair_delta(stock[:, :4].sum(axis=-1)), shape_pair_delta(stock)],
          [score['twins']['max_total_difference'], score['twins']['max_half_difference'],
           score['twins']['max_within_half_bin_difference']], 'within-half control')
    if shape_pair_delta(history[:, :, 3]) > score['twins']['max_conductance_difference_fixed_candidate'] + TOL:
        raise ArithmeticError('Reported all-time twin conductance gap is below a sampled gap.')
    sizes = [len(reg['kinetic_grid'][key]) for key in ('factors', 'power', 'loss', 'interface', 'bias', 'shape')]
    metrics = {name: verify_arm(name, row, score['arms'][name], a, b, score['truth'], sizes, reg, early_time)
               for name, row in pred['arms'].items()}
    truth_kinetics = np.array(score['kinetics'])
    close(truth_kinetics[::2, :5], truth_kinetics[1::2, :5], 'latent rate, bias and transport paired across shape')
    close(truth_kinetics[::2, 5], -truth_kinetics[1::2, 5], 'paired opposite within-half modes')
    for name, row in pred['arms'].items():
        mass = np.asarray(row['posterior']).reshape(len(a), -1, sizes[-1]).sum(axis=1)
        closest_shape = np.abs(truth_kinetics[:, 5, None]-reg['kinetic_grid']['shape']).argmin(axis=1)
        metrics[name]['mean_posterior_mass_on_correct_shape'] = float(mass[np.arange(len(a)), closest_shape].mean())
    return {'group': score['group'], 'oracle': oracle, 'shape_blind': shape_blind, 'all_blind': wholly_blind,
            'catalogue_mesh_delta': score['catalogue_mesh_changes'], 'arms': metrics,
            'twins': score['twins']}


def summarize(groups, reg):
    arms = {}
    for name in groups[0]['arms']:
        rows = [g['arms'][name] for g in groups]
        arms[name] = {key: float(np.mean([r[key] for r in rows])) for key in (
            'regret', 'brier', 'shape_probability_shift', 'mean_posterior_mass_on_correct_shape')}
        for key in ('count', 'shape_twin_choice_differences', 'mesh_changes'):
            arms[name][key] = sum(r[key] for r in rows)
        for key in ('dry_twin_difference', 'posterior_twin_difference', 'max_loss_prior_difference',
                    'max_interface_prior_difference'):
            arms[name][key] = max(r[key] for r in rows)
        for key in ('profile', 'process'):
            arms[name][key] = {stat: np.mean([r[key][stat] for r in rows], axis=0).tolist()
                               for stat in ('mae', 'coverage', 'width')}
        index = reg['unseen_dry_bin']
        arms[name]['unseen_bin'] = {stat: arms[name]['profile'][stat][index] for stat in ('mae', 'coverage', 'width')}
    contrasts = {}
    for contrast, weights in FACTORIAL_CONTRASTS.items():
        delta = np.asarray([sum(x*g['arms'][arm]['count'] for arm, x in weights.items()) for g in groups])
        contrasts[contrast] = {'sum': int(delta.sum()), 'median': float(np.median(delta)),
                               'parents_win_tie_loss': [int((delta > 0).sum()), int((delta == 0).sum()),
                                                        int((delta < 0).sum())]}
        if len(weights) == 2:
            better = next(key for key, coefficient in weights.items() if coefficient == 1)
            reference = next(key for key, coefficient in weights.items() if coefficient == -1)
            a, b = [np.array([g['arms'][name]['pass'] for g in groups], dtype=bool) for name in (better, reference)]
            contrasts[contrast]['recovered_lost_requests'] = [int((a & ~b).sum()), int((~a & b).sum())]
    return {'parents': len(groups), 'oracle': sum(g['oracle'] for g in groups),
            'shape_blind_oracle': sum(g['shape_blind'] for g in groups),
            'all_blind_oracle': sum(g['all_blind'] for g in groups),
            'catalogue_mesh_changes': sum(g['catalogue_mesh_delta'] for g in groups),
            'arms': arms, 'mother_level_contrasts': contrasts,
            'total_equality_max': max(g['twins']['max_total_difference'] for g in groups),
            'half_equality_max': max(g['twins']['max_half_difference'] for g in groups),
            'max_spatial_difference': max(g['twins']['max_within_half_bin_difference'] for g in groups),
            'max_conductivity_difference': max(g['twins']['max_conductance_difference_fixed_candidate'] for g in groups)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    args = parser.parse_args()
    started = datetime.now(timezone.utc).isoformat()
    reg, dev, frozen, scored, index = load_receipts(args.run)
    chosen, grid = verify_development(reg, dev, frozen, scored)
    cohorts = {}
    for name in frozen['cohorts']:
        rows = [verify_group(x, y, reg, chosen) for x, y in zip(frozen['cohorts'][name], scored['cohorts'][name], strict=True)]
        cohorts[name] = {'summary': summarize(rows, reg), 'groups': rows}
        print(name, {key: values['count'] for key, values in cohorts[name]['summary']['arms'].items()}, flush=True)
    with (args.run/'INDEPENDENT_SCORE_AUDIT.json').open('x', encoding='utf-8') as file:
        json.dump({'started_utc': started, 'finished_utc': datetime.now(timezone.utc).isoformat(), 'inputs': index,
                   'audit_sha256': digest(Path(__file__)), 'chosen_timing': chosen, 'timing_candidates': grid,
                   'limits': 'Scores and summary calibration independently recomputed from frozen choices, intervals and stored margins. Does not re-solve the PDE or recompute predictive intervals.',
                   'cohorts': cohorts}, file, indent=2, allow_nan=False)


if __name__ == '__main__':
    main()
