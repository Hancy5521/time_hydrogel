"""Register and freeze full-window predictions before the matched and shifted outcomes."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from itertools import repeat
from pathlib import Path
from time import perf_counter

import numpy as np

from gel_reactive_port_timing import read_json, sha256
from gel_reactive_two_read_assay import write_new
from patch_exchange_forward import exchange_bank
from patch_reservoir_joint_bank import joint_bank, prepared_fields
from reservoir_joint_observation import stock_fractions
from window_risk_baselines import OutcomeForest, development_lock
from window_risk_cases import groups
from window_risk_forecasts import elapsed, foresee, margins


def truth(group,task,plan):
    start=perf_counter()
    t=elapsed(task,plan)
    query=np.rint(np.asarray(plan['times_diagnostic'])/plan['sampling_dt']).astype(int)
    action=plan['fixed_action']
    cells,dt=plan['truth_mesh']
    arguments=(group['known'],task['actions'],task['mechanism'])
    tail=(group['kinetic'],group['stock'],cells,dt,t)
    environment=task['bath']
    if group['shift']:
        wet,stock=exchange_bank(*arguments,environment['wet'],*tail,inventory=True,outcome_cfg=plan['exchange'])
        weak=exchange_bank(*arguments,environment['weak'],*tail,outcome_cfg=plan['exchange'])
    else:
        wet,stock=joint_bank(*arguments,environment['wet'],*tail,inventory=True)
        weak=joint_bank(*arguments,environment['weak'],*tail)
    raw=margins(wet,weak,t,task)
    dry=stock_fractions(prepared_fields(group['known'],[task['actions'][action]],task['mechanism'],group['stock'],cells)[:,0])
    path=np.concatenate((stock[:,query,action],wet[:,query,action,3:4]),axis=-1)
    return {'parent':group['parent'],'margins':raw.tolist(),'dry':dry.tolist(),'process':path.tolist(),
            'inventory_closure_max':float(np.abs(stock.sum(axis=-1)-1).max()),
            'engine':'exchange' if group['shift'] else 'irreversible','seconds':perf_counter()-start}


def forecast(group,task,plan,depth_check):
    start=perf_counter()
    row=foresee(group,task,plan,diagnose=depth_check)
    row['seconds']=perf_counter()-start
    return row


def prepare(group,task,plan):
    return {'input':forecast(group,task,plan,False),'output':truth(group,task,plan)}


def get(path,plan_hash):
    result=read_json(path)
    if result['registered_sha256']!=plan_hash:
        raise ValueError('Different protocol in prerequisite '+str(path))
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=('train','development','freeze','score'))
    parser.add_argument('run',type=Path)
    parser.add_argument('--workers',type=int,default=6)
    cli=parser.parse_args()
    registered=cli.run/'REGISTERED.json'
    plan=read_json(registered)
    digest=sha256(registered)
    for name,expected in {**plan['sources'],**plan['inputs']}.items():
        if sha256(Path(name))!=expected:
            raise ValueError(f'Changed benchmark file: {name}')
    parent=read_json(Path(plan['task_path']))
    train=cli.run/'TRAIN.json'
    dev=cli.run/'DEVELOPMENT.json'
    frozen=cli.run/'FROZEN.json'
    target={'train':train,'development':dev,'freeze':frozen,'score':cli.run/'SCORED.json'}[cli.phase]
    if target.exists():
        raise FileExistsError(target)
    stamp=datetime.now(timezone.utc).isoformat()
    start=perf_counter()
    receipt={'registered_sha256':digest,'started_utc':stamp}
    with ProcessPoolExecutor(max_workers=cli.workers) as pool:
        if cli.phase in ('train','development'):
            cases=list(groups(cli.phase,plan,parent))
            receipt['rows']=list(pool.map(prepare,cases,repeat(parent),repeat(plan)))
            if cli.phase=='development':
                source=get(train,digest)['rows']
                receipt['forest_lock']=development_lock(source,receipt['rows'],plan)
                receipt['train_sha256']=sha256(train)
            print(cli.phase,len(receipt['rows']),flush=True)
        else:
            source=get(dev,digest)
            if source['train_sha256']!=sha256(train):
                raise ValueError('Frozen supervised training source changed.')
            receipt['development_sha256']=sha256(dev)
            receipt['splits']={}
            if cli.phase=='freeze':
                training=get(train,digest)['rows']
                models={m:OutcomeForest(training,m,v['leaf'],plan['forest']['seed'])
                        for m,v in source['forest_lock'].items()}
            else:
                predictions=get(frozen,digest)
                if predictions['development_sha256']!=sha256(dev):
                    raise ValueError('Development model choices changed after freeze.')
                receipt['frozen_sha256']=sha256(frozen)
            for split in plan['sizes']:
                if split in ('train','development'):
                    continue
                cases=list(groups(split,plan,parent))
                if cli.phase=='freeze':
                    assess=split in ('test','shift_test')
                    rows=list(pool.map(forecast,cases,repeat(parent),repeat(plan),repeat(assess)))
                    for row in rows:
                        before=perf_counter()
                        for model,fit in models.items():
                            row['forecasts'][model]=fit.predict(row,plan)
                        row['learned_online_seconds']=perf_counter()-before
                else:
                    held=[row['parent'] for row in predictions['splits'][split]]
                    if held != [row['parent'] for row in cases]:
                        raise ValueError('The committed prediction groups changed before scoring.')
                    rows=list(pool.map(truth,cases,repeat(parent),repeat(plan)))
                receipt['splits'][split]=rows
                print(split,cli.phase,len(rows),flush=True)
    receipt.update(completed_utc=datetime.now(timezone.utc).isoformat(),wall_seconds=perf_counter()-start)
    write_new(target,receipt)


if __name__=='__main__':
    main()
