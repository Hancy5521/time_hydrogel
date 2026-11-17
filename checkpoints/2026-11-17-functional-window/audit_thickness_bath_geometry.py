"""Check the geometry lever under fixed absolute bath volume and a separate dose rule."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from itertools import repeat
from pathlib import Path

import numpy as np

from gel_reactive_port_timing import read_json, sha256
from gel_reactive_two_read_assay import write_new
from patch_exchange_forward import exchange_bank
from patch_thickness_catalogue_benchmark import geometry, operational_margins
from window_risk_cases import groups
from window_risk_forecasts import elapsed
from window_spatial_probe import changed_mechanism


def compare(group, task, plan, frozen, scored):
    if len({group['parent'],frozen['parent'],scored['parent']})!=1:
        raise ValueError('Different physical groups in the paired sensitivity.')
    outcomes={'original_areal_scaled_bath':np.asarray(scored['margins']['fixed_area_conductance'])}
    t=elapsed(task,plan)
    cells,dt=plan['truth_mesh']
    settings=changed_mechanism(group,plan)['exchange']
    for mode,fix_areal in [('areal_fixed_bath',True),('volumetric_fixed_bath',False)]:
        matrices=[]
        for variant in plan['thickness']['actions_at_scale']:
            scale=variant['scale']
            known,cfg=geometry(group['known'],task,variant,1.)
            law={**settings,'bath_volume_ratio':settings['bath_volume_ratio']/scale}
            if not fix_areal:
                cfg={**cfg,'dose':task['mechanism']['dose']}
            wet,weak=[exchange_bank(known,variant['actions'],cfg,a,group['kinetic'],group['stock'],cells,dt,t,outcome_cfg=law)
                      for a in (task['bath']['wet'],task['bath']['weak'])]
            matrices.append(operational_margins(wet,weak,t,task,scale)[1])
        outcomes[mode]=np.concatenate(matrices,axis=-1)
    result={}
    for mode,m in outcomes.items():
        result[mode]={}
        for cat in ('grid25','thickness27'):
            mask=plan['thickness']['catalogues'][cat]
            action=np.asarray(frozen['forecasts']['fixed_area_conductance'][cat]['actions_full_index'])[...,None]
            picked=np.take_along_axis(m,action,axis=-1)[...,0]
            result[mode][cat]={'feasible':int((m[...,mask]<=0).any(axis=-1).sum()),
                               'selected_pass':int((picked<=0).sum())}
    return {'parent':group['parent'],'requests':outcomes['areal_fixed_bath'].shape[0]*outcomes['areal_fixed_bath'].shape[1],
            'results':result}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('audit',type=Path)
    p.add_argument('--workers',type=int,default=5)
    a=p.parse_args()
    registration=a.audit/'REGISTERED.json'
    test=read_json(registration)
    for name,digest in test['hashes'].items():
        if sha256(Path(name))!=digest:
            raise ValueError(f'Post-score sensitivity input changed: {name}.')
    parent=Path(test['source'])
    plan=read_json(parent/'REGISTERED.json')
    task=read_json(Path(plan['task_path']))
    f,s=[read_json(parent/name)['splits']['shift_test'] for name in ('FROZEN.json','SCORED.json')]
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        rows=list(pool.map(compare,groups('shift_test',plan,task),repeat(task),repeat(plan),f,s))
    aggregate={mode:{cat:{k:sum(r['results'][mode][cat][k] for r in rows) for k in stats}
                     for cat,stats in d.items()} for mode,d in rows[0]['results'].items()}
    write_new(a.audit/'RESULTS.json',{'registered_sha256':sha256(registration),'completed_utc':datetime.now(timezone.utc).isoformat(),
                                     'requests':sum(r['requests'] for r in rows),'rows':rows,'summary':aggregate})


if __name__=='__main__':
    main()
