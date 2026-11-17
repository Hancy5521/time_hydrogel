"""Analytic paired-state diagnostic with known simulated feasible sets and witness means."""

import argparse
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.special import logit, ndtr

from audit_tight_witness_identifiability import coupon_clock
from gel_reactive_port_timing import read_json, sha256
from gel_reactive_two_read_assay import write_new
from tight_window_phase_study import parents
from tight_window_study import open_rows


def paired_frontier(feasible, separation, cutoff):
    """Exact expectation of this reference policy over independent within-mother assay noise."""
    meets = feasible.any(axis=-1)
    common = (feasible[:, 0] & feasible[:, 1]).any(axis=-1)
    threshold = np.divide(logit(cutoff), separation, out=np.full_like(separation, np.inf), where=separation > 0.)
    right = ndtr(.5*separation-threshold)
    wrong = ndtr(-.5*separation-threshold)
    true_issue = np.where(common[:, None], 1., right[:, None] * meets)
    false_issue = np.where(common[:, None], 0., wrong[:, None] * meets[:, ::-1])
    p_active = 1. - np.prod(1.-true_issue-false_issue, axis=1)
    p_fail = 1. - np.prod(1.-false_issue, axis=1)
    active, failed = float(p_active.mean()), float(p_fail.mean())
    return {'posterior_cutoff':cutoff, 'expected_activity':active, 'expected_unsafe_active':failed,
            'expected_conditional_unsafe':failed/active if active else None,
            'expected_correct_requests':float(true_issue.mean())}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('run', type=Path)
    p.add_argument('output', type=Path)
    a = p.parse_args()
    protocol, scored = [a.run / name for name in ('REGISTERED.json','DEVELOPMENT_SCORED.json')]
    plan = read_json(protocol)
    labels = open_rows(scored,protocol)['rows']
    task = read_json(Path(plan['task_path']))
    groups, local = parents(plan,'development',task)
    if [x['mother'] for x in labels] != [g['parent'] for g in groups]:
        raise ValueError('The outcome groups differ from the witness cohort.')
    f = np.asarray([r['clauses'] for r in labels]).max(axis=-1)<=0.
    sigma = plan['tight']['sensors']['log_conductance']
    cutoffs = np.r_[np.linspace(.55,.95,17),.975,.99,.999]
    data = {}
    for name, scale in (('standard_hydration',None),('standard_electrical',1.),('thin_electrical',.65)):
        if scale is None:
            d = np.zeros(len(groups))
        else:
            clock = np.array([coupon_clock(g,task,local,scale) for g in groups])
            d = np.abs(clock[:,0,-1]-clock[:,1,-1])/sigma
        curve = [paired_frontier(f,d,float(t)) for t in cutoffs]
        eligible = [row for row in curve if row['expected_activity']>=.10 and row['expected_conditional_unsafe']<=.10]
        data[name] = {'curves':curve,'best_demonstration_at_10_percent':max(eligible,key=lambda r:r['expected_correct_requests'],default=None)}
    result = {'generated_at_utc':datetime.now(timezone.utc).isoformat(),'protocol_sha256':sha256(protocol),
              'score_sha256':sha256(scored),'source_sha256':sha256(Path(__file__)),'result':data,
              'scope':('Descriptive two-hypothesis policy assumes the true pair of means and the true feasible sets are known '
                       'for each mother. Only the label within that pair is hidden. Uses one sensor alone and analytic expected '
                       'error over known synthetic groups. Not an estimated or calibrated operating policy.')}
    write_new(a.output,result)
    for key,val in data.items():
        print(key,val['best_demonstration_at_10_percent'])


if __name__ == '__main__':
    main()
