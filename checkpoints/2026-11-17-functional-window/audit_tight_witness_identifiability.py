"""Quantify which paired hidden material states actually require distinguishing."""

import argparse
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.special import ndtr

from gel_reactive_port_timing import read_json, sha256
from gel_reactive_two_read_assay import write_new
from patch_exchange_forward import exchange_bank
from patch_reservoir_joint_bank import joint_bank
from patch_thickness_catalogue_benchmark import geometry
from tight_window_phase_study import parents
from tight_window_study import open_rows
from window_spatial_probe import changed_mechanism


def coupon_clock(group, task, local, scale):
    witness = task['actions'][local['fixed_action']]
    known, cfg = geometry(group['known'], task, {'scale':scale}, 1.)
    cells, period = local['truth_mesh']
    alternative = changed_mechanism(group, local)
    bank, keywords = (exchange_bank, {'outcome_cfg':alternative['exchange']}) if group['shift'] else (joint_bank, {})
    trace = bank(known, [witness], cfg, 1., group['kinetic'], group['stock'], cells, period,
                 np.r_[0., local['spatial_probe']['times']], **keywords)
    return np.log(trace[:, 1:, 0, 3] / known['thickness'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('output', type=Path)
    a = parser.parse_args()
    protocol, scored = [a.run / name for name in ('REGISTERED.json', 'DEVELOPMENT_SCORED.json')]
    plan = read_json(protocol)
    task = read_json(Path(plan['task_path']))
    records = open_rows(scored, protocol)['rows']
    groups, local = parents(plan, 'development', task)
    if [r['mother'] for r in records] != [g['parent'] for g in groups]:
        raise ValueError('Saved outcomes and regenerated witness groups differ.')
    y = np.asarray([r['clauses'] for r in records])
    feasible = y.max(axis=-1) <= 0.
    any_state = feasible.any(axis=-1)
    shared = (feasible[:, 0] & feasible[:, 1]).any(axis=-1)
    needs_distinction = any_state.any(axis=-1) & ~shared
    counts = {'mothers': len(groups), 'neither_state_has_feasible_candidate': int((~any_state.any(axis=-1)).sum()),
              'only_one_state_has_feasible_candidate': int((any_state.sum(axis=-1) == 1).sum()),
              'both_feasible_but_requires_different_action': int((any_state.all(axis=-1) & ~shared).sum()),
              'one_action_can_serve_both': int(shared.sum())}
    signals = {}
    sigma = plan['tight']['sensors']['log_conductance']
    for scale in (1., .65):
        values = np.asarray([coupon_clock(g, task, local, scale) for g in groups])
        d = np.abs(values[:, 0] - values[:, 1]) / sigma
        signals[str(scale)] = {'early_late_dimensionless_times': local['spatial_probe']['times'],
            'mean_bayes_identification_error_given_relevant_pair': ndtr(-d[needs_distinction] / 2).mean(axis=0).tolist(),
            'median_expected_separation_sigma_given_relevant_pair': np.median(d[needs_distinction], axis=0).tolist()}
    result = {'utc': datetime.now(timezone.utc).isoformat(), 'source_sha256': sha256(Path(__file__)),
              'registration_sha256': sha256(protocol), 'scored_sha256': sha256(scored), 'counts': counts,
              'kinetics_only_water_measurement': 'exactly same water field in both paired states, so balanced error .5',
              'gaussian_assay': {'sigma_log_conductance': sigma, 'signals': signals},
              'scope': 'Retrospective optimal binary discrimination with known true simulated means; not a deployed error rate. '
                       'A .65-thickness witness is a previously untested sensor geometry sensitivity, without material validation.'}
    write_new(a.output, result)
    print(result)


if __name__ == '__main__':
    main()
