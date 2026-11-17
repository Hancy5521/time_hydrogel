from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import subprocess
import sys

import pytest

from validate_window_confirmation import COMMON, check_choices, combine, prepare, requirements, score


def contract():
    start=datetime(2026,1,5,tzinfo=timezone.utc)

    def when(x):
        return (start+timedelta(seconds=x)).isoformat()

    def reading(value,specimen,second):
        return {'value':value,'specimen_id':specimen,'unit':'fraction',
                'acquired_utc':when(second),'available_utc':when(second+1)}

    protocol={'study_kind':'synthetic_fixture','methods':{'global':{'view':'bulk','artifact_sha256':'1'*64},
              'local':{'view':'depth','artifact_sha256':'2'*64}},'calibration_sha256':'3'*64,'calibration_mothers':['development'],
              'candidate_ids':['A','B'],'times_s':[0,5,10,15,20],'time_tolerance_s':.5,'witness_times_s':[2,4],
              'limits':{'off_siemens':.05,'on_siemens':.20,'soft_pa':140.,'support_pa':250.,'weak_siemens':.06}}
    request={'id':'request','install_end_s':5,'work_s':[10,20]}
    observations=[{'mother':'held','cutoff_utc':when(8),'release_utc':when(11),'witness_exposure_utc':when(0),
                  'context':{'thickness_m':.002,'metadata_sha256':'4'*64},
                  'witness_specimen_id':'witness','common':{key:reading(.2,f'short_{key}',2) for key in COMMON},
                  'witness':{'bulk':[reading(.1,'witness',2),reading(.4,'witness',4)],
                             'depth':[reading(.05,'witness',2),reading(.15,'witness',4)]},
                  'requests':[request]}]
    choices={'held':{'request':{'global':'A','local':'A'}}}
    values={'install_off':.03,'install_support':350.,'work_on':.25,'work_soft':120.,'weak_off':.03}
    rows=[]
    for clause,environment,channel,stamp,_,_ in requirements(request,protocol):
        val=values[clause]
        rows.append({'mother':'held','candidate':'A','environment':environment,'channel':channel,
                     'specimen_id':environment,'exposure_utc':when(12),'acquired_utc':when(12+stamp),
                     'planned_s':stamp,'low':val,'high':val,'unit':'S' if channel=='conductance' else 'Pa'})
    return protocol,observations,choices,rows,start+timedelta(seconds=9)


def test_views_are_emitted_separately():
    p,x,c,rows,when=contract()
    method,cases,owners=prepare(p,x)
    check_choices(p,x,c,cases)
    assert method['global'][0]['witness'][0]['value']==.1
    assert method['local'][0]['witness'][0]['value']==.05
    assert 'witness_exposure_utc' not in method['local'][0]
    assert score(p,x,c,rows,when,owners)['method_summary']['local']['success_fraction_equal_mother_bounds']==[1.,1.]


def test_midwindow_crossing_fails_despite_successful_endpoints():
    p,x,c,rows,when=contract()
    _,_,owners=prepare(p,x)
    target=next(r for r in rows if r['environment']=='wet' and r['channel']=='conductance' and r['planned_s']==15)
    target.update(low=.17,high=.19)
    report=score(p,x,c,rows,when,owners)
    assert report['method_summary']['local']['maternal_failure_count_bounds']==[1,1]
    missing=[r for r in rows if r is not target]
    report=score(p,x,c,missing,when,owners)
    assert report['method_summary']['local']['maternal_failure_count_bounds']==[0,1]
    assert report['requests']['held']['request']['local']['status']=='unresolved'


def test_late_witness_and_specimen_splicing_rejected():
    p,x,c,rows,when=contract()
    invalid=deepcopy(x)
    invalid[0]['witness']['depth'][0]['available_utc']='2026-01-05T00:00:10+00:00'
    with pytest.raises(ValueError,match='deadline'):
        prepare(p,invalid)
    _,_,owners=prepare(p,x)
    rows[1]['specimen_id']='different_wet'
    with pytest.raises(ValueError,match='splice'):
        score(p,x,c,rows,when,owners)


def test_different_mothers_do_not_share_companions():
    p,x,c,rows,when=contract()
    copied=deepcopy(x[0]); copied['mother']='other'
    with pytest.raises(ValueError,match='one original mother'):
        prepare(p,x+[copied])


def test_cli_freeze_replays_archived_software_fixture(tmp_path):
    p,x,c,rows,_=contract()
    for name,data in zip(('protocol','readings','choices','outcomes'),(p,x,c,rows),strict=True):
        (tmp_path/f'{name}.json').write_text(json.dumps(data),encoding='utf8')
    command=[sys.executable,'-B',str(Path(__file__).with_name('validate_window_confirmation.py'))]
    paths=[str(tmp_path/f'{name}.json') for name in ('protocol','readings','choices','frozen','outcomes','report')]
    subprocess.run([*command,'freeze',*paths[:4]],check=True)
    subprocess.run([*command,'score',*paths],check=True)
    result=json.loads(Path(paths[-1]).read_bytes())
    assert result['method_summary']['global']['success_fraction_equal_mother_bounds']==[1.,1.]
    second={**result,'requests':{'another mother':result['requests']['held']}}
    assert combine([result,second])['mothers']==2
    with pytest.raises(ValueError,match='more than one'):
        combine([result,result])
