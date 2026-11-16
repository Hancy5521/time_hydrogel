"""Locate frozen depth interval misses that no reweighting of the prior can repair."""

import argparse
from datetime import datetime, timezone
from itertools import product
from pathlib import Path

import numpy as np

from gel_reactive_port_timing import read_json, sha256
from gel_reactive_two_read_assay import write_new
from patch_reservoir_joint_bank import prepared_fields
from reservoir_joint_observation import stock_fractions


def diagnose(plan, task, frozen, scores):
    action = plan['fixed_diagnostic_action']
    shape = np.array(list(product(plan['kinetic_grid']['bias'], plan['kinetic_grid']['shape'])))
    bins = plan['unseen_dry_bin']
    output = {}
    for name, observed in scores.items():
        accum = {key: np.zeros(8, dtype=int) for key in ('tested','outside_full_support','posterior_miss_inside_support','covered')}
        for forecast, truth in zip(frozen[name],observed,strict=True):
            if forecast['group'] != truth['group']:
                raise ValueError('A posterior no longer matches its scored parent.')
            priors = prepared_fields(forecast['known'], [task['actions'][action]], task['mechanism'], shape,
                                     plan['meshes']['inference'][0])[:, 0]
            prior_bins = stock_fractions(priors)
            y = np.asarray(truth['truth']['dry'])
            posterior = forecast['arms']['dual_half']['summary']['profile']
            lo, hi = (np.array(posterior[q]) for q in ('lo','hi'))
            slack = 1e-10
            outside = (y < prior_bins.min(axis=0)-slack) | (y > prior_bins.max(axis=0)+slack)
            hit = (y >= lo-slack) & (y <= hi+slack)
            if np.any(outside & hit):
                raise ArithmeticError(f'The finite-candidate posterior exceeds its prior support for action {action}.')
            accum['tested'] += len(y)
            accum['outside_full_support'] += outside.sum(axis=0)
            accum['posterior_miss_inside_support'] += (~outside & ~hit).sum(axis=0)
            accum['covered'] += hit.sum(axis=0)
        output[name] = {'per_bin': {key: v.tolist() for key,v in accum.items()},
                        'overall': {k:int(v.sum()) for k,v in accum.items()},
                        'unseen_bin': {k:int(v[bins]) for k,v in accum.items()}}
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder', type=Path)
    args = parser.parse_args()
    manifest = args.folder/'REGISTERED.json'
    description = read_json(manifest)
    for file, digest in description['inputs'].items():
        if sha256(Path(file)) != digest:
            raise ValueError(f'Modified diagnostic ingredient {file}.')
    run = Path(description['run'])
    plan = read_json(run/'REGISTERED.json')
    task = read_json(Path(plan['task_path']))
    findings = diagnose(plan,task,read_json(run/'FROZEN.json')['cohorts'],read_json(run/'SCORED.json')['cohorts'])
    data = {'registered_sha256':sha256(manifest),'time_utc':datetime.now(timezone.utc).isoformat(),
            'evidence':'Descriptive post-result score diagnosis on the same frozen groups. No procedure updated.',
            'cohorts':findings}
    write_new(args.folder/'RESULTS.json',data)
    for name,result in findings.items():
        print(name, result['overall'], 'held bin', result['unseen_bin'])


if __name__ == '__main__':
    main()
