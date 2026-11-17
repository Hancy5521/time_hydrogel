"""Re-evaluate frozen thickness actions on a denser service-time grid."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from itertools import repeat
from pathlib import Path

import numpy as np

from gel_reactive_port_timing import read_json, sha256
from gel_reactive_two_read_assay import write_new
from patch_thickness_catalogue_benchmark import reveal
from window_risk_cases import groups


def compare(earlier, predicted, dense, plan):
    if len({r['parent'] for r in (earlier,predicted,dense)})!=1:
        raise ValueError('Time-resolution audit must use identical frozen mothers.')
    outcomes={}
    for name, old in earlier['margins'].items():
        coarse=np.asarray(old)
        fine=np.asarray(dense['margins'][name])
        # A max across a larger set cannot become smaller on the same forward trajectory.
        if np.min(fine-coarse)<-1e-10:
            raise AssertionError('The dense grid is not a deterministic extension of the coarse grid.')
        changed=(fine>0)!=(coarse>0)
        refs={}
        for cat, item in predicted['forecasts'][name].items():
            selected=np.asarray(item['actions_full_index'])[...,None]
            ids=plan['thickness']['catalogues'][cat]
            refs[cat]={'choice_flips':int(np.take_along_axis(changed,selected,axis=-1).sum()),
                       'oracle_feasibility_flips':int(((fine[...,ids]<=0).any(axis=-1)!=(coarse[...,ids]<=0).any(axis=-1)).sum())}
        outcomes[name]={'catalogue_flips':int(changed.sum()),'largest_worsening':float(np.max(fine-coarse)),'arms':refs}
    return {'mother':dense['parent'],'comparisons':outcomes}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('audit',type=Path)
    parser.add_argument('--workers',type=int,default=6)
    a=parser.parse_args()
    registered=a.audit/'REGISTERED.json'
    spec=read_json(registered)
    for name,expected in spec['hashes'].items():
        if sha256(Path(name))!=expected:
            raise ValueError(f'Changed frozen input {name}.')
    origin=Path(spec['benchmark'])
    plan=read_json(origin/'REGISTERED.json')
    task=read_json(Path(plan['task_path']))
    fresh={**plan,'sampling_dt':spec['sampling_dt']}
    original=read_json(origin/'SCORED.json')['splits']
    forecast=read_json(origin/'FROZEN.json')['splits']
    output={}
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        for role, rows in original.items():
            dense=pool.map(reveal,groups(role,plan,task),repeat(task),repeat(fresh))
            output[role]=[compare(r,p,d,plan) for r,p,d in zip(rows,forecast[role],dense,strict=True)]
    write_new(a.audit/'RESULTS.json',{'registered_sha256':sha256(registered),'completed_utc':datetime.now(timezone.utc).isoformat(),
                                     'role_results':output})


if __name__=='__main__':
    main()
