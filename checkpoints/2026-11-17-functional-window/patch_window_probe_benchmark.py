"""Freeze an equal-count local/bulk sensor experiment and equal-mother-budget sampling check."""

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
from patch_window_sampling_benchmark import score_one
from window_factorized_sampling import conditional_forecast
from window_risk_baselines import OutcomeForest
from window_risk_cases import groups
from window_risk_forecasts import foresee
from window_spatial_probe import changed_mechanism, forecast_spatial


def evaluate_forecasts(group,task,plan):
    start=perf_counter()
    old=foresee(group,task,plan,diagnose=False)
    original=perf_counter()-start
    start=perf_counter()
    result=forecast_spatial(group,task,plan)
    local=perf_counter()-start
    start=perf_counter()
    divided={**plan,'continuum_count':plan['spatial_probe']['path_per_observation']}
    conditional=conditional_forecast(group,task,divided)
    result['timing']={'shared_prior_all_outputs':original,'hydraulic_library':local,'conditional_128x2':perf_counter()-start}
    result['forecasts'].update(continuum=old['forecasts']['continuum'],conditional=conditional)
    result['parent']=group['parent']
    return result


def outcome(group,task,plan):
    return truth(group,task,changed_mechanism(group,plan))


def evaluation(rows,plan):
    keys=list(rows[0]['results'])
    metrics=list(rows[0]['results'][keys[0]])
    values={key:np.array([[x['results'][key][metric] for metric in metrics] for x in rows]) for key in keys}
    rng=np.random.default_rng(plan['factorization']['bootstrap_seed'])
    idx=rng.integers(len(rows),size=(2000,len(rows)))
    pairs={}
    comparisons=[('library_bulk','library_legacy'),('library_depth','library_bulk'),('library_depth','library_legacy'),
                 ('library_depth','forest_direct'),('library_bulk','forest_direct'),('conditional','continuum')]
    for reference,compared in comparisons:
        diff=values[reference]-values[compared]
        ci=np.quantile(diff[idx].mean(axis=1),[.025,.975],axis=0)
        pairs[f'{reference}_minus_{compared}']={metric:[float(mean),float(lo),float(hi)]
                                               for metric,mean,lo,hi in zip(metrics,diff.mean(axis=0),*ci,strict=True)}
    return {'means':{key:dict(zip(metrics,b.mean(axis=0).tolist(),strict=True)) for key,b in values.items()},
            'paired_difference':pairs,'oracle_success':float(np.mean([x['oracle_success'] for x in rows]))}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=('freeze','score'))
    parser.add_argument('destination',type=Path)
    parser.add_argument('--workers',type=int,default=8)
    cli=parser.parse_args()
    registry=cli.destination/'REGISTERED.json'
    plan=read_json(registry)
    for key,value in plan['hashes'].items():
        if sha256(Path(key))!=value:
            raise ValueError(f'Digest mismatch for {key}')
    destination=cli.destination/('FROZEN.json' if cli.phase=='freeze' else 'SCORED.json')
    if destination.exists():
        raise FileExistsError(destination)
    digest=sha256(registry)
    receipt={'registered_sha256':digest,'started_utc':datetime.now(timezone.utc).isoformat(),'splits':{}}
    task=read_json(Path(plan['task_path']))
    start=perf_counter()
    if cli.phase=='freeze':
        base=Path(plan['locked_training'])
        source=read_json(base/'TRAIN.json')['rows']
        locks=read_json(base/'DEVELOPMENT.json')['forest_lock']
        forests={m:OutcomeForest(source,m,l['leaf'],plan['forest']['seed']) for m,l in locks.items()}
    else:
        frozen=cli.destination/'FROZEN.json'
        stored=read_json(frozen)
        if stored['registered_sha256']!=digest:
            raise ValueError('Protocol changed after prediction freeze.')
        receipt['frozen_sha256']=sha256(frozen)
        receipt['summary']={}
    with ProcessPoolExecutor(max_workers=cli.workers) as pool:
        for name in ('test','shift_test'):
            cases=groups(name,plan,task)
            if cli.phase=='freeze':
                rows=list(pool.map(evaluate_forecasts,cases,repeat(task),repeat(plan)))
                for row in rows:
                    for key,model in forests.items():
                        row['forecasts'][key]=model.predict(row,plan)
            else:
                answers=pool.map(outcome,cases,repeat(task),repeat(plan))
                rows=[score_one(pred,true) for pred,true in zip(stored['splits'][name],answers,strict=True)]
                receipt['summary'][name]=evaluation(rows,plan)
            print(name,cli.phase,len(rows),flush=True)
            receipt['splits'][name]=rows
    receipt.update(completed_utc=datetime.now(timezone.utc).isoformat(),wall_seconds=perf_counter()-start)
    write_new(destination,receipt)


if __name__=='__main__':
    main()
