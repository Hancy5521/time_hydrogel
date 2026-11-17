"""Descriptive audit of frozen development actions, without changing a threshold."""

import argparse
from pathlib import Path

import numpy as np

from gel_reactive_port_timing import read_json, sha256
from gel_reactive_two_read_assay import write_new


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('run', type=Path)
    p.add_argument('output', type=Path)
    a = p.parse_args()
    protocol, frozen, scored = [a.run / n for n in ('REGISTERED.json','DEVELOPMENT_FROZEN.json','DEVELOPMENT_SCORED.json')]
    cfg, predicted, realized = [read_json(x) for x in (protocol, frozen, scored)]
    x, y = predicted['rows'], realized['rows']
    if [r['mother'] for r in x] != [r['mother'] for r in y] or realized['frozen_sha256'] != sha256(frozen):
        raise ValueError('Mismatched frozen development receipts.')
    margins = np.asarray([r['clauses'] for r in y]).max(axis=-1)
    outcomes, minima = {}, {}
    active_floor = cfg['tight']['development_criterion']['min_development_mothers']
    for panel in realized['results']:
        minima[panel] = {}
        for rank in x[0]['forecasts'][panel]['banks']:
            bank = [r['forecasts'][panel]['banks'][rank] for r in x]
            a_idx = np.array([b['choice'] for b in bank])
            s = np.array([b['score'] for b in bank])
            correct = np.take_along_axis(margins,a_idx[..., None],axis=-1)[..., 0] <= 0.
            if rank == 'posterior_violation':
                outcomes[panel] = correct
            candidates = []
            for t in np.unique(s):
                accepted = s <= t
                active = int(accepted.any(axis=1).sum())
                if active >= active_floor:
                    failed = int((accepted & ~correct).any(axis=1).sum())
                    candidates.append((failed/active,active,failed,int((accepted&correct).sum())))
            risk, active, failed, successful = min(candidates)
            minima[panel][rank] = {'minimum_observed_risk':risk,'active_mothers':active,
                                   'failed_mothers':failed,'passed_requests':successful}
    paired = {}
    rng = np.random.default_rng(cfg['tight']['bootstrap_seed'])
    sample = rng.integers(len(margins), size=(5000,len(margins)))
    for comparison in ('thin_depth','standard_depth'):
        effect = (outcomes['thin_electric'].astype(float) - outcomes[comparison]).mean(axis=1)
        paired[comparison] = {'fraction':float(effect.mean()),'ci95':np.quantile(effect[sample].mean(axis=1),[.025,.975]).tolist()}
    result = {'protocol_sha256':sha256(protocol),'frozen_sha256':sha256(frozen),'scored_sha256':sha256(scored),
              'feasible_requests':int((margins.min(axis=-1)<=0).sum()),'minima':minima,'ungated_rank_paired':paired,
              'scope':'Exploratory on the development mothers. No policy or risk claim.'}
    write_new(a.output,result)


if __name__ == '__main__':
    main()
