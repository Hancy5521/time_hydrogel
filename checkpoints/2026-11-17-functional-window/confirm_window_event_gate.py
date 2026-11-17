"""Confirm a stricter gate without retraining, then test a new hidden ion law."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from itertools import repeat
from pathlib import Path
from time import perf_counter

import numpy as np

from calibrate_window_spatial_probe import forecast
from gel_reactive_port_timing import read_json, sha256
from gel_reactive_two_read_assay import write_new
from patch_window_event_study import current_plan, measured, mixtures, open_receipt, policy_calibration, report
from patch_window_probe_benchmark import outcome
from window_event_calibration import certify_policy
from window_event_policy import development_choice, slate
from window_group_calibration import take_actions


def hidden_bath(group,task,plan):
    stress=plan['hidden_bath_stress']
    rng=np.random.default_rng(stress['seed']+group['i'])
    rates={k:float(rng.uniform(*bounds)) for k,bounds in stress['ranges'].items()}
    settings={**plan,'exchange':{**plan['exchange'],**rates}}
    return outcome(group,task,settings)


def each_mechanism(x, y, cal, primary, plan):
    """Can one frozen rule cover all prevalences of the two specified laws?"""
    s=plan['event_settings']
    learned=slate(x,cal['response'])[primary]
    accepted=learned['score']<cal['policy']['primary_cutoff']
    actual=take_actions(y,learned['chosen'])
    result={}
    for law in ('irreversible','exchange'):
        ids=np.array([r['engine']==law for r in cal['labels']])
        result[law]=certify_policy(actual[ids],accepted[ids],max_risk=s['risk_target'],delta=s['delta'],min_activity=s['activity_target'])
    return {'laws':result,'certified':all(r['certified'] for r in result.values()),
            'joint_p':max(r['joint_p'] for r in result.values())}


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('phase',choices=('forecast','calibrate','score'))
    cli.add_argument('run',type=Path)
    cli.add_argument('--workers',type=int,default=12)
    args=cli.parse_args()
    protocol=args.run/'REGISTERED.json'
    plan=read_json(protocol)
    for name,expected in plan['hashes'].items():
        if sha256(Path(name))!=expected:
            raise ValueError(f'Changed input or source {name}.')
    task=read_json(Path(plan['task_path']))
    target=args.run/{'forecast':'FROZEN.json','calibrate':'CERTIFICATE.json','score':'SCORED.json'}[args.phase]
    if target.exists():
        raise FileExistsError(target)
    root=Path(plan['prior_study'])
    lock=read_json(root/'POLICY_LOCK.json')
    base=read_json(root/'FROZEN.json')
    if lock['frozen_sha256']!=sha256(root/'FROZEN.json'):
        raise ValueError('Prior development package is not linked.')
    old_y=np.asarray([row['margins'] for row in lock['splits']['policy_development']])
    response=lock['response']
    dev=development_choice(old_y,slate(base['splits']['policy_development'],response),plan['event_settings'])
    primary=dev['primary_method']
    if primary is None:
        raise ValueError('The original development study cannot support a less risky active cutoff.')
    receipt={'protocol_sha256':sha256(protocol),'started_utc':datetime.now(timezone.utc).isoformat(),
             'policy':dev,'splits':{}}
    start=perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        if args.phase=='forecast':
            for role in plan['event_roles']:
                design=current_plan(plan,role)
                r=list(pool.map(forecast,mixtures(design,role,task),repeat(task),repeat(design)))
                receipt['splits'][role]=r
                print(args.phase,role,len(r),flush=True)
        else:
            frozen,receipt['frozen_sha256']=open_receipt(args.run,'FROZEN.json',receipt['protocol_sha256'])
            if frozen['policy']!=dev:
                raise ValueError('Risk-aligned decisions changed after freeze.')
            if args.phase=='calibrate':
                role='gate_calibration'
                x=frozen['splits'][role]
                rows,y=measured(pool,plan,role,task,x)
                receipt['splits'][role]=rows
                receipt['results']=policy_calibration(slate(x,response),y,dev,plan)
                receipt['all_prevalences']=each_mechanism(x,y,{'response':response,'policy':dev,'labels':rows},primary,plan)
                print(args.phase,role,len(rows),flush=True)
            else:
                fitted,receipt['certificate_sha256']=open_receipt(args.run,'CERTIFICATE.json',receipt['protocol_sha256'])
                if fitted['frozen_sha256']!=receipt['frozen_sha256']:
                    raise ValueError('Calibration does not match locked predictions.')
                receipt['reports']={}
                for role in ('confirmation','hidden_bath_stress'):
                    x=frozen['splits'][role]
                    clean,y=measured(pool,plan,role,task,x)
                    receipt['splits'][role]=clean
                    forecasts=slate(x,response)
                    receipt['reports'][role]=report(forecasts,y,fitted['results'],primary)
                    if role=='hidden_bath_stress':
                        design=current_plan(plan,role)
                        variant=list(pool.map(hidden_bath,mixtures(design,role,task),repeat(task),repeat(design)))
                        value=np.asarray([r['margins'] for r in variant])
                        receipt['splits']['changed_bath']=variant
                        receipt['reports']['changed_bath']=report(forecasts,value,fitted['results'],primary)
                    print(args.phase,role,len(clean),flush=True)
    receipt.update(completed_utc=datetime.now(timezone.utc).isoformat(),wall_seconds=perf_counter()-start)
    write_new(target,receipt)


if __name__=='__main__':
    main()
