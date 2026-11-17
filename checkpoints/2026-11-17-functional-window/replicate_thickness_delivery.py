"""Independent larger confirmation of two already frozen catalogue policies."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from itertools import repeat
from pathlib import Path
from time import perf_counter

import numpy as np

from calibrate_thickness_catalogue import forecast_packet
from gel_reactive_port_timing import read_json, sha256
from gel_reactive_two_read_assay import write_new
from patch_thickness_catalogue_benchmark import reveal
from patch_window_event_study import current_plan, mixtures, open_receipt
from window_event_policy import outcomes
from window_group_calibration import take_actions


def anchored_prediction(group,task,plan):
    base=forecast_packet(group,task,plan)
    banks={}
    for name,bank in base['banks'].items():
        upper=np.asarray(bank['margin_q'])[2]
        action=np.lexsort((np.asarray(bank['margin_mean']),np.asarray(bank['probability']),upper),axis=-1)[...,0]
        accepted=np.take_along_axis(upper,action[...,None],axis=-1)[...,0]<plan['fixed_cutoffs'][name]
        banks[name]={'upper':upper.tolist(),'mean':bank['margin_mean'],'probability':bank['probability'],
                     'choice':action.tolist(),'accept':accepted.tolist()}
    return {'parent':group['parent'],'banks':banks,'observed_depth':base['observed_depth']}


def report(predictions,labels,plan):
    if [row['parent'] for row in predictions]!=[row['parent'] for row in labels]:
        raise ValueError('The new outcomes do not match the forecast freeze.')
    actual=np.asarray([r['margins']['fixed_area_conductance'] for r in labels])
    successful=[]
    action_reached=[]
    result={}
    for cat in plan['compared_catalogues']:
        raw=actual[...,plan['thickness']['catalogues'][cat]]
        pred=[r['banks'][cat] for r in predictions]
        choose=np.asarray([p['choice'] for p in pred],dtype=int)
        keep=np.asarray([p['accept'] for p in pred],dtype=bool)
        risk=take_actions(raw,choose)>0
        success=(~risk)&keep
        successful.append(success)
        reached=raw.min(axis=-1)<=0
        action_reached.append(reached)
        result[cat]=outcomes(raw,choose,keep)|{'unissued_when_frozen_choice_passes':int((~risk&~keep).sum())}
    old,new=successful
    better=action_reached[1]&~action_reached[0]
    diff=(new.astype(float)-old).mean(axis=(1,2))
    idx=np.random.default_rng(plan['confirmation_stats']['bootstrap_seed']).integers(len(diff),size=(5000,len(diff)))
    return {'catalogues':result,'paired_delta':float(diff.mean()),
            'paired_ci_95':np.quantile(diff[idx].mean(axis=1),[.025,.975]).tolist(),
            'mother_effects':diff.tolist(),'newly_feasible_requests':int(better.sum()),
            'thickness_success_on_newly_feasible':int((new&better).sum())}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase',choices=('forecast','score'))
    p.add_argument('run',type=Path)
    p.add_argument('--workers',type=int,default=15)
    a=p.parse_args()
    path=a.run/'REGISTERED.json'
    plan=read_json(path)
    for name,expected in plan['hashes'].items():
        if sha256(Path(name))!=expected:
            raise ValueError(f'Registration source changed: {name}.')
    destination=a.run/('FROZEN.json' if a.phase=='forecast' else 'SCORED.json')
    if destination.exists():
        raise FileExistsError(destination)
    task=read_json(Path(plan['task_path']))
    role='confirmation'
    settings=current_plan(plan,role)
    draws=mixtures(settings,role,task)
    result={'protocol_sha256':sha256(path),'started_utc':datetime.now(timezone.utc).isoformat()}
    start=perf_counter()
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        if a.phase=='forecast':
            result['parents']=[]
            for k,row in enumerate(pool.map(anchored_prediction,draws,repeat(task),repeat(settings)),start=1):
                result['parents'].append(row)
                if k%64==0:
                    print('froze',k,'forecasts in memory',flush=True)
        else:
            lock,result['frozen_sha256']=open_receipt(a.run,'FROZEN.json',result['protocol_sha256'])
            result['labels']=list(pool.map(reveal,draws,repeat(task),repeat(settings)))
            result['results']=report(lock['parents'],result['labels'],plan)
    result.update(completed_utc=datetime.now(timezone.utc).isoformat(),wall_seconds=perf_counter()-start)
    write_new(destination,result)


if __name__=='__main__':
    main()
