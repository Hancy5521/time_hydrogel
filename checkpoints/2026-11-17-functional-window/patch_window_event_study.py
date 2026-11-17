"""Check risk and utility on independent mothers with unobserved mixture membership."""

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
from patch_window_probe_benchmark import outcome
from window_event_calibration import apply_marginal_event, certify_policy, fit_marginal_event
from window_event_policy import development_choice, fit_response, outcomes, slate
from window_risk_cases import groups


PHASES={'forecast':'FROZEN.json','lock':'POLICY_LOCK.json','calibrate':'CERTIFICATE.json','score':'SCORED.json'}


def current_plan(plan, role):
    spec=plan['event_roles'][role]
    witness=plan['spatial_probe'].copy()
    witness['snapshot_noise_seed']+=spec['spatial_noise_salt']
    witness['shift_variation_seed']+=spec['hydraulic_salt']
    return {**plan,'spatial_probe':witness}


def mixtures(plan, role, task):
    spec=plan['event_roles'][role]
    ordinary, shifted=role,'shift_'+role
    size=spec['groups']
    local={**plan,'sizes':{ordinary:size,shifted:size},'seeds':{ordinary:spec['nominal_seed'],shifted:spec['shift_seed']}}
    shifted_draw=np.random.default_rng(spec['mechanism_seed']).random(size)<spec['shift_fraction']
    for a,b,use in zip(groups(ordinary,local,task),groups(shifted,local,task),shifted_draw,strict=True):
        selected=b if use else a
        yield {**selected,'parent':a['parent']}


def measured(pool, plan, role, task, inputs):
    design=current_plan(plan,role)
    cases=list(mixtures(design,role,task))
    rows=list(pool.map(outcome,cases,repeat(task),repeat(design)))
    if [x['parent'] for x in rows]!=[x['parent'] for x in inputs]:
        raise ValueError('Truth identities differ from the frozen predictions.')
    return rows,np.array([x['margins'] for x in rows])


def apply_threshold(policy,limit):
    s=policy['score']
    return s<limit if limit is not None else np.zeros_like(s,dtype=bool)


def policy_calibration(policies, y, dev, plan):
    result={}
    settings=plan['event_settings']
    from window_group_calibration import take_actions
    for name, policy in policies.items():
        selected=dev['variants'][name]['selected']
        threshold=selected['cutoff'] if selected is not None else None
        realized=take_actions(y,policy['chosen'])
        accept=apply_threshold(policy,threshold)
        result[name]={'locked_cutoff':threshold,
                      'fixed_policy':certify_policy(realized,accept,max_risk=settings['risk_target'],
                                                    delta=settings['delta'],min_activity=settings['activity_target']),
                      'marginal_event':fit_marginal_event(realized,policy['score'],alpha=settings['marginal_target']),
                      'locked_calibration_outcomes':outcomes(y,policy['chosen'],accept)}
    return result


def report(policies,y,fitted,primary):
    metrics={}
    for name,p in policies.items():
        rule=fitted[name]
        locked=apply_threshold(p,rule['locked_cutoff'])
        event=apply_marginal_event(p['score'],rule['marginal_event'])
        metrics[name]={'locked':outcomes(y,p['chosen'],locked),'marginal':outcomes(y,p['chosen'],event),
                       'choice':p['chosen'].tolist(),'score':p['score'].tolist(),
                       'accept_locked':locked.tolist(),'accept_marginal':event.tolist(),
                       'primary':name==primary}
    return metrics


def open_receipt(run,name,digest):
    f=run/name
    data=read_json(f)
    if data['protocol_sha256']!=digest:
        raise ValueError(f'{name} belongs to a different protocol.')
    return data,sha256(f)


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('phase',choices=PHASES)
    cli.add_argument('run',type=Path)
    cli.add_argument('--workers',type=int,default=10)
    args=cli.parse_args()
    path=args.run/'REGISTERED.json'
    plan=read_json(path)
    digest=sha256(path)
    for name,expected in plan['hashes'].items():
        if sha256(Path(name))!=expected:
            raise ValueError(f'Changed input or source {name}.')
    target=args.run/PHASES[args.phase]
    if target.exists():
        raise FileExistsError(target)
    task=read_json(Path(plan['task_path']))
    receipt={'protocol_sha256':digest,'started_utc':datetime.now(timezone.utc).isoformat(),'splits':{}}
    start=perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        if args.phase=='forecast':
            for role in plan['event_roles']:
                waves=current_plan(plan,role)
                inputs=list(pool.map(forecast,mixtures(waves,role,task),repeat(task),repeat(waves)))
                receipt['splits'][role]=inputs
                print(args.phase,role,len(inputs),flush=True)
        else:
            f,receipt['frozen_sha256']=open_receipt(args.run,PHASES['forecast'],digest)
            def truth_for(role):
                predictions=f['splits'][role]
                rows, values=measured(pool,plan,role,task,predictions)
                receipt['splits'][role]=rows
                print(args.phase,role,len(rows),flush=True)
                return predictions,values

            if args.phase=='lock':
                train,y=truth_for('policy_train')
                fit=fit_response(train,y,plan['event_settings']['logistic_c'])
                develop,truth=truth_for('policy_development')
                receipt.update(response=fit,development=development_choice(truth,slate(develop,fit),plan['event_settings']))
            else:
                locked,receipt['policy_sha256']=open_receipt(args.run,PHASES['lock'],digest)
                if locked['frozen_sha256']!=receipt['frozen_sha256']:
                    raise ValueError('The policy points to a different forecast freeze.')
                if args.phase=='calibrate':
                    x,y=truth_for('gate_calibration')
                    receipt['results']=policy_calibration(slate(x,locked['response']),y,locked['development'],plan)
                else:
                    fitted,receipt['certificate_sha256']=open_receipt(args.run,PHASES['calibrate'],digest)
                    if fitted['policy_sha256']!=receipt['policy_sha256'] or fitted['frozen_sha256']!=receipt['frozen_sha256']:
                        raise ValueError('Inconsistent calibration ancestry.')
                    receipt['reports']={}
                    for role in ('confirmation','prevalence_shift'):
                        x,y=truth_for(role)
                        receipt['reports'][role]=report(slate(x,locked['response']),y,fitted['results'],
                                                        locked['development']['primary_method'])
    receipt.update(completed_utc=datetime.now(timezone.utc).isoformat(),wall_seconds=perf_counter()-start)
    write_new(target,receipt)


if __name__=='__main__':
    main()
