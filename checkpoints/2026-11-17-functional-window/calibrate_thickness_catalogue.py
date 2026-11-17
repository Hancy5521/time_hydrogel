"""Test whether thickness-enabled designs improve risk-constrained delivery."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from itertools import repeat
from pathlib import Path
from time import perf_counter

import numpy as np

from gel_reactive_port_timing import read_json, sha256
from gel_reactive_two_read_assay import write_new
from patch_reservoir_joint_bank import joint_bank
from patch_thickness_catalogue_benchmark import geometry, operational_margins, prediction_weights, reveal, wet_reads
from patch_window_event_study import current_plan, mixtures, open_receipt, policy_calibration, report
from window_event_policy import METHOD, development_choice, slate
from window_risk_forecasts import choose, elapsed, prior, summary


def forecast_packet(group, task, plan):
    s=plan['spatial_probe']
    atoms=prior(group['known'],{**plan,'continuum_count':s['atoms_per_water']},True)
    obs,dry,depth,features=wet_reads(group,task,plan)
    weights=prediction_weights(group,task,plan,atoms,obs,dry,depth)
    cells,dt=plan['prediction_mesh']
    t=elapsed(task,plan)
    records=[]
    for water_scale in s['water_grid']:
        block=[]
        for variant in plan['thickness']['actions_at_scale']:
            known,cfg=geometry(group['known'],task,variant,water_scale)
            wet,weak=[joint_bank(known,variant['actions'],cfg,bath,atoms[:,:4],atoms[:,4:],cells,dt,t)
                      for bath in (task['bath']['wet'],task['bath']['weak'])]
            block.append(operational_margins(wet,weak,t,task,variant['scale'])[1])
        records.append(np.concatenate(block,axis=-1))
    raw=np.concatenate(records,axis=0)
    banks={}
    for name in plan['compared_catalogues']:
        index=plan['thickness']['catalogues'][name]
        state=raw[...,index]
        action,p=choose(state,weights,plan['risk_weight'])
        mean,q=summary(state,weights)
        banks[name]={'probability':p.tolist(),'selected':action.tolist(),'margin_mean':mean.tolist(),'margin_q':q.tolist()}
    return {'parent':group['parent'],'observed_depth':depth.tolist(),'features':features.tolist(),'banks':banks}


def ranking(rows, catalog, response):
    return slate([{'forecasts':{METHOD:r['banks'][catalog]}} for r in rows],response)


def test_results(rows, y, cal, plan, response):
    results={}
    trials=[]
    passed=[]
    for catalog in plan['compared_catalogues']:
        ids=plan['thickness']['catalogues'][catalog]
        raw=y[...,ids]
        stage=cal['catalogues'][catalog]
        r=report(ranking(rows,catalog,response),raw,stage['calibration'],stage['development']['primary_method'])
        eligible=stage['calibration'][stage['development']['primary_method']]['fixed_policy']['certified']
        selected=r[stage['development']['primary_method']]
        selected['certified']=eligible
        passed.append(eligible)
        chosen=np.array(selected['choice'])
        accepted=np.array(selected['accept_locked'])
        outcome=np.take_along_axis(raw,chosen[...,None],axis=-1)[...,0]
        counts=(accepted&(outcome<=0)).sum(axis=(1,2))
        trials.append(counts)
        results[catalog]={'policies':r,'primary':stage['development']['primary_method'],'per_mother_success':counts.tolist()}
    diff=trials[1]-trials[0]
    r=np.random.default_rng(plan['integration_bootstrap_seed']).integers(len(diff),size=(2500,len(diff)))
    factor=y.shape[1]*y.shape[2]
    return {'catalogues':results,'safe_issued_delta_fraction':float(np.mean(diff)/factor),
            'paired_ci':(np.quantile(diff[r].mean(axis=1),[.025,.975])/factor).tolist(),
            'both_certified':all(passed)}


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('phase',choices=('forecast','lock','calibrate','score'))
    cli.add_argument('run',type=Path)
    cli.add_argument('--workers',type=int,default=12)
    a=cli.parse_args()
    address=a.run/'REGISTERED.json'
    p=read_json(address)
    for name,digest in p['hashes'].items():
        if sha256(Path(name))!=digest:
            raise ValueError(f'Changed input or source {name}.')
    task=read_json(Path(p['task_path']))
    response=read_json(Path(p['learned_policy']))['response']
    dst=a.run/{'forecast':'FROZEN.json','lock':'POLICY_LOCK.json','calibrate':'CERTIFICATE.json','score':'SCORED.json'}[a.phase]
    if dst.exists():
        raise FileExistsError(dst)
    cfg={'protocol_sha256':sha256(address),'started_utc':datetime.now(timezone.utc).isoformat(),'splits':{}}
    start=perf_counter()
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        if a.phase=='forecast':
            for role in p['event_roles']:
                plan=current_plan(p,role)
                cfg['splits'][role]=list(pool.map(forecast_packet,mixtures(plan,role,task),repeat(task),repeat(plan)))
                print(a.phase,role,len(cfg['splits'][role]),flush=True)
        else:
            previous,cfg['frozen_sha256']=open_receipt(a.run,'FROZEN.json',cfg['protocol_sha256'])
            role={'lock':'policy_development','calibrate':'gate_calibration','score':'confirmation'}[a.phase]
            plan=current_plan(p,role)
            pred=previous['splits'][role]
            data=list(pool.map(reveal,mixtures(plan,role,task),repeat(task),repeat(plan)))
            if [r['parent'] for r in data]!=[r['parent'] for r in pred]:
                raise ValueError('Truth and prediction mother groups changed.')
            cfg['splits'][role]=data
            values=np.array([r['margins']['fixed_area_conductance'] for r in data])
            if a.phase=='score':
                certified,cfg['certificate_sha256']=open_receipt(a.run,'CERTIFICATE.json',cfg['protocol_sha256'])
                if certified['frozen_sha256']!=cfg['frozen_sha256']:
                    raise ValueError('The certificate points to another forecast package.')
                cfg['results']=test_results(pred,values,certified,plan,response)
            else:
                cfg['catalogues']={}
                if a.phase=='calibrate':
                    locked,cfg['policy_sha256']=open_receipt(a.run,'POLICY_LOCK.json',cfg['protocol_sha256'])
                    if locked['frozen_sha256']!=cfg['frozen_sha256']:
                        raise ValueError('The locked policy uses different predictions.')
                for name in p['compared_catalogues']:
                    y=values[...,plan['thickness']['catalogues'][name]]
                    policies=ranking(pred,name,response)
                    if a.phase=='lock':
                        dev=development_choice(y,policies,plan['event_settings'])
                        if dev['primary_method'] is None:
                            dev['primary_method']='outcome_score'
                        cfg['catalogues'][name]=dev
                    else:
                        dev=locked['catalogues'][name]
                        cfg['catalogues'][name]={'development':dev,'calibration':policy_calibration(policies,y,dev,plan)}
            print(a.phase,role,len(data),flush=True)
    cfg.update(completed_utc=datetime.now(timezone.utc).isoformat(),wall_seconds=perf_counter()-start)
    write_new(dst,cfg)


if __name__=='__main__':
    main()
