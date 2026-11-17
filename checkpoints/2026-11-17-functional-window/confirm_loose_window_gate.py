"""Audit the unchanged 25%-risk policy on the repaired independent cohorts."""

import argparse
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from gel_reactive_port_timing import read_json, sha256
from gel_reactive_two_read_assay import write_new
from patch_window_event_study import apply_threshold
from window_event_calibration import certify_policy
from window_event_policy import outcomes, slate
from window_group_calibration import take_actions


def analyze(predictions, labels, response, ranking, threshold):
    if [r['parent'] for r in predictions]!=[r['parent'] for r in labels]:
        raise ValueError('Outcome, forecast and mother order disagree.')
    given=slate(predictions,response)[ranking]
    y=np.asarray([r['margins'] for r in labels])
    accepted=apply_threshold(given,threshold)
    actual=take_actions(y,given['chosen'])
    return y,given,accepted,actual


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('run',type=Path)
    arg=p.parse_args()
    run=arg.run
    registr=run/'REGISTERED.json'
    protocol=read_json(registr)
    for name,digest in protocol['hashes'].items():
        if sha256(Path(name))!=digest:
            raise ValueError(f'Changed input or source {name}.')
    previous=read_json(Path(protocol['prior_study'])/'POLICY_LOCK.json')
    f,c,s=[read_json(run/n) for n in ('FROZEN.json','CERTIFICATE.json','SCORED.json')]
    if any(r['protocol_sha256']!=sha256(registr) for r in (f,c,s)) or s['certificate_sha256']!=sha256(run/'CERTIFICATE.json'):
        raise ValueError('Need the linked independent confirmation receipts.')
    main=previous['development']['primary_method']
    cutoff=previous['development']['primary_cutoff']
    recipe=protocol['replay_loose_policy']
    labels=c['splits']['gate_calibration']
    y,policy,accepted,realized=analyze(f['splits']['gate_calibration'],labels,previous['response'],main,cutoff)
    test=certify_policy(realized,accepted,max_risk=recipe['target'],delta=recipe['delta'],min_activity=recipe['activity'])
    strata={}
    for label in ('irreversible','exchange'):
        idx=np.array([a['engine']==label for a in labels])
        strata[label]=certify_policy(realized[idx],accepted[idx],max_risk=recipe['target'],delta=recipe['delta'],min_activity=recipe['activity'])
    h,q,keep,v=analyze(f['splits']['confirmation'],s['splits']['confirmation'],previous['response'],main,cutoff)
    cert={'protocol_sha256':sha256(registr),'outcomes_sha256':sha256(run/'SCORED.json'),
          'completed_utc':datetime.now(timezone.utc).isoformat(),'model':main,'threshold':cutoff,
          'exact_certificate':test,'all_prevalences':{'certified':all(a['certified'] for a in strata.values()),'strata':strata},
          'calibration_descriptive':outcomes(y,policy['chosen'],accepted),'confirmation':outcomes(h,q['chosen'],keep)}
    write_new(run/'LOOSE_CONFIRMED.json',cert)


if __name__=='__main__':
    main()
