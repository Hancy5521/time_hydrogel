"""Compare the two 256-path integrators on new parent groups."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from itertools import repeat
from pathlib import Path
from time import perf_counter

import numpy as np

from gel_reactive_port_timing import read_json, sha256
from gel_reactive_two_read_assay import write_new
from patch_window_risk_benchmark import truth
from window_factorized_sampling import conditional_forecast
from window_risk_baselines import OutcomeForest
from window_risk_cases import groups
from window_risk_forecasts import foresee


def forecast(group, task, plan):
    start=perf_counter()
    standard=foresee(group,task,plan,diagnose=False)
    standard['seconds']=perf_counter()-start
    start=perf_counter()
    standard['forecasts']['conditional']=conditional_forecast(group,task,plan)
    standard['conditional_seconds']=perf_counter()-start
    del standard['forecasts']['continuum_spanning']
    return standard


def score_one(frozen, observed):
    if frozen['parent']!=observed['parent']:
        raise ValueError('Different frozen and scored mother identities.')
    y=np.asarray(observed['margins'])
    optimum=y.min(axis=-1)
    result={}
    for key,outputs in frozen['forecasts'].items():
        indices=np.asarray(outputs['selected'])[...,None]
        margin=np.take_along_axis(y,indices,axis=-1)[...,0]
        p=np.asarray(outputs['probability'])
        risk=np.take_along_axis(p,indices,axis=-1)[...,0]
        result[key]={'all_action_brier':float(((p-(y>0))**2).mean()),
                     'choice_brier':float(((risk-(margin>0))**2).mean()),
                     'success':float((margin<=0).mean()),
                     'positive_regret':float((np.maximum(margin,0)-np.maximum(optimum,0)).mean()),
                     'upper_coverage':float(np.all(np.asarray(outputs['margin_q'])[2]>=y)),
                     'interval_90_width':float(np.diff(np.asarray(outputs['margin_q'])[[0,3]],axis=0).mean())}
    return {'parent':observed['parent'],'margins':observed['margins'],'oracle_success':float((optimum<=0).mean()),
            'inventory_closure_max':observed['inventory_closure_max'],'results':result}


def summarize(rows,plan):
    keys=list(rows[0]['results'])
    metrics=list(rows[0]['results'][keys[0]])
    data={key:np.array([[x['results'][key][metric] for metric in metrics] for x in rows]) for key in keys}
    rng=np.random.default_rng(plan['factorization']['bootstrap_seed'])
    idx=rng.integers(len(rows),size=(2000,len(rows)))
    effects={}
    for key in keys:
        if key=='conditional':
            continue
        diff=data['conditional']-data[key]
        ci=np.quantile(diff[idx].mean(axis=1),[.025,.975],axis=0)
        effects[key]={metric:[float(mean),float(lo),float(hi)]
                      for metric,mean,lo,hi in zip(metrics,diff.mean(axis=0),*ci,strict=True)}
    return {'means':{key:dict(zip(metrics,arr.mean(axis=0).tolist(),strict=True)) for key,arr in data.items()},
            'conditional_minus':effects,'oracle_success':float(np.mean([x['oracle_success'] for x in rows]))}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=('freeze','score'))
    parser.add_argument('destination',type=Path)
    parser.add_argument('--workers',type=int,default=6)
    cli=parser.parse_args()
    registry=cli.destination/'REGISTERED.json'
    plan=read_json(registry)
    for name,expected in plan['hashes'].items():
        if sha256(Path(name))!=expected:
            raise ValueError(f'Source or input changed after registration: {name}')
    task=read_json(Path(plan['task_path']))
    original=Path(plan['locked_training'])
    target=cli.destination/('FROZEN.json' if cli.phase=='freeze' else 'SCORED.json')
    if target.exists():
        raise FileExistsError(target)
    cfg={'registered_sha256':sha256(registry),'started_utc':datetime.now(timezone.utc).isoformat(),'splits':{}}
    t0=perf_counter()
    if cli.phase=='freeze':
        train=read_json(original/'TRAIN.json')['rows']
        locks=read_json(original/'DEVELOPMENT.json')['forest_lock']
        forest={key:OutcomeForest(train,key,value['leaf'],plan['forest']['seed']) for key,value in locks.items()}
    else:
        frozen=cli.destination/'FROZEN.json'
        cfg['frozen_sha256']=sha256(frozen)
        lock=read_json(frozen)
        if lock['registered_sha256']!=cfg['registered_sha256']:
            raise ValueError('Unexpected frozen protocol digest.')
        cfg['summary']={}
    with ProcessPoolExecutor(max_workers=cli.workers) as pool:
        for split in ('test','shift_test'):
            if cli.phase=='freeze':
                rows=list(pool.map(forecast,groups(split,plan,task),repeat(task),repeat(plan)))
                for row in rows:
                    for name,model in forest.items():
                        row['forecasts'][name]=model.predict(row,plan)
            else:
                outcomes=pool.map(truth,groups(split,plan,task),repeat(task),repeat(plan))
                rows=[score_one(pred,obs) for pred,obs in zip(lock['splits'][split],outcomes,strict=True)]
                cfg['summary'][split]=summarize(rows,plan)
            cfg['splits'][split]=rows
            print(split,cli.phase,len(rows),flush=True)
    cfg.update(completed_utc=datetime.now(timezone.utc).isoformat(),wall_seconds=perf_counter()-t0)
    write_new(target,cfg)


if __name__=='__main__':
    main()
