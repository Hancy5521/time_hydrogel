"""Freeze unchanged sensor policies, calibrate on new mothers, and check non-vacuity."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from itertools import repeat
from pathlib import Path
from time import perf_counter

import numpy as np

from gel_reactive_port_timing import read_json, sha256
from gel_reactive_two_read_assay import write_new
from patch_window_probe_benchmark import outcome
from window_group_calibration import evaluate_decisions, evaluate_upper, fit_group_upper, take_actions
from window_risk_cases import groups
from window_spatial_probe import forecast_spatial


def forecast(group,task,plan):
    return {'parent':group['parent'],**forecast_spatial(group,task,plan)}


def role_plan(plan,role):
    witness=plan['spatial_probe'].copy()
    salt=plan['fresh_calibration']['noise_salts'][role]
    witness['snapshot_noise_seed']+=salt
    witness['shift_variation_seed']+=salt
    return {**plan,'spatial_probe':witness}


def readouts(predictions,model):
    rows=[row['forecasts'][model] for row in predictions]
    q=np.asarray([row['margin_q'] for row in rows])
    return q[:,2],q[:,1],np.asarray([row['selected'] for row in rows]),np.asarray([row['probability'] for row in rows])


def labels(pool,predictions,role,task,plan):
    values=list(pool.map(outcome,groups(role,plan,task),repeat(task),repeat(plan)))
    if [x['parent'] for x in values]!=[x['parent'] for x in predictions]:
        raise ValueError('Scored parents changed from the locked predictions.')
    return values


def calibrate(y,prediction,alpha):
    output={}
    for method in prediction[0]['forecasts']:
        u,_,a,_=readouts(prediction,method)
        catalogue=fit_group_upper(y,u,alpha=alpha)
        selected=fit_group_upper(take_actions(y,a),take_actions(u,a),alpha=alpha)
        output[method]={'catalogue':catalogue.metadata(),'selected':selected.metadata()}
    return output


def evaluate(y,prediction,calibration,raw_cfg):
    result={}
    opts={'bootstrap':raw_cfg['bootstrap'],'seed':raw_cfg['bootstrap_seed']}
    for method,stored in calibration.items():
        upper,center,idx,p=readouts(prediction,method)
        chosen=take_actions(y,idx)
        chosen_p=take_actions(p,idx)
        result[method]={'raw':evaluate_decisions(chosen,np.ones_like(chosen,dtype=bool),probability=chosen_p,**opts)}
        band=upper+stored['catalogue']['radius']
        approved=band<=0.
        preference=np.lexsort((np.broadcast_to(np.arange(y.shape[-1]),y.shape),center,p,~approved),axis=-1)[...,0]
        result[method]['catalogue']={
            'coverage':evaluate_upper(y,band,median=center,**opts),
            'decision':evaluate_decisions(take_actions(y,preference),take_actions(approved.astype(float),preference)>0,
                                          probability=take_actions(p,preference),**opts)}
        selected_upper=take_actions(upper,idx)+stored['selected']['radius']
        result[method]['selected']={
            'coverage':evaluate_upper(chosen,selected_upper,median=take_actions(center,idx),**opts),
            'decision':evaluate_decisions(chosen,selected_upper<=0.,probability=chosen_p,**opts)}
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=('freeze','calibrate','score'))
    parser.add_argument('run',type=Path)
    parser.add_argument('--workers',type=int,default=8)
    cli=parser.parse_args()
    protocol=cli.run/'REGISTERED.json'
    plan=read_json(protocol)
    for name,expected in plan['hashes'].items():
        if sha256(Path(name))!=expected:
            raise ValueError(f'Registered file changed: {name}')
    task=read_json(Path(plan['task_path']))
    digest=sha256(protocol)
    filenames={'freeze':'FROZEN.json','calibrate':'CALIBRATION.json','score':'SCORED.json'}
    destination=cli.run/filenames[cli.phase]
    if destination.exists():
        raise FileExistsError(destination)
    receipt={'registered_sha256':digest,'started_utc':datetime.now(timezone.utc).isoformat(),'splits':{}}
    start=perf_counter()
    pre=plan['fresh_calibration']
    if cli.phase!='freeze':
        frozen_path=cli.run/'FROZEN.json'
        frozen=read_json(frozen_path)
        receipt['frozen_sha256']=sha256(frozen_path)
        if frozen['registered_sha256']!=digest:
            raise ValueError('Protocol changed after forecast.')
    if cli.phase=='score':
        cp=cli.run/'CALIBRATION.json'
        recovery=read_json(cp)
        receipt['calibration_sha256']=sha256(cp)
        if recovery['frozen_sha256']!=receipt['frozen_sha256'] or recovery['registered_sha256']!=digest:
            raise ValueError('A protocol or forecast changed after calibration.')
    with ProcessPoolExecutor(max_workers=cli.workers) as pool:
        if cli.phase=='freeze':
            for role in plan['sizes']:
                settings=role_plan(plan,role)
                receipt['splits'][role]=list(pool.map(forecast,groups(role,settings,task),repeat(task),repeat(settings)))
                print(cli.phase,role,len(receipt['splits'][role]),flush=True)
        else:
            for test,cal in pre['roles'].items():
                role=cal if cli.phase=='calibrate' else test
                preds=frozen['splits'][role]
                rows=labels(pool,preds,role,task,role_plan(plan,role))
                y=np.asarray([x['margins'] for x in rows])
                result=calibrate(y,preds,plan['calibration']['alpha']) if cli.phase=='calibrate' else evaluate(y,preds,recovery['splits'][cal]['results'],pre)
                receipt['splits'][role]={'values':rows,'results':result}
                print(cli.phase,role,len(rows),flush=True)
    receipt.update(completed_utc=datetime.now(timezone.utc).isoformat(),wall_seconds=perf_counter()-start)
    write_new(destination,receipt)


if __name__=='__main__':
    main()
