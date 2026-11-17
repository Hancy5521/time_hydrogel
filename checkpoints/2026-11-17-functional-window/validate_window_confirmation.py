"""Freeze paired bulk/depth inputs and score installed specimens at every registered time."""

import argparse
import math
from datetime import datetime, timezone
from pathlib import Path

from validate_hydrogel_pilot import SHA, assess, number, ordered_unique, read_packet, sha256, utc, write_new


COMMON=('low_hydration','high_hydration','low_release','high_release','dry_near_inventory')
VIEWS=('bulk','depth')
OUTCOMES={'conductance':'S','front_modulus':'Pa','back_modulus':'Pa'}


def requirements(request, protocol):
    install=number(request['install_end_s'],'Install end')
    first,last=map(lambda t:number(t,'Work window time'),request['work_s'])
    schedule=protocol['times_s']
    if not 0<install<=first<last or any(t not in schedule for t in (0,install,first,last)):
        raise ValueError('Include both registered windows and all their endpoints.')
    q=protocol['limits']
    return [*(("install_off","wet","conductance",t,"max",q['off_siemens']) for t in schedule if t<=install),
            *(("install_support","wet","back_modulus",t,"min",q['support_pa']) for t in schedule if t<=install),
            *(("work_on","wet","conductance",t,"min",q['on_siemens']) for t in schedule if first<=t<=last),
            *(("work_soft","wet","front_modulus",t,"max",q['soft_pa']) for t in schedule if first<=t<=last),
            *(("weak_off","weak","conductance",t,"max",q['weak_siemens']) for t in schedule if t<=last)]


def check_observation(item, cutoff, witness_utc=None, intended=None, tolerance=None):
    if item['unit']!='fraction' or not item['specimen_id']:
        raise ValueError('A readout needs its calibrated unit and source specimen.')
    if item['value'] is None:
        if not item.get('missing_reason'):
            raise ValueError('Missing observations need a recorded cause.')
        return
    number(item['value'],'Calibrated fraction')
    acquired,available=utc(item['acquired_utc']),utc(item['available_utc'])
    if not acquired<=available<=cutoff:
        raise ValueError('A present readout must be available by the design deadline.')
    if witness_utc is not None:
        elapsed=(acquired-witness_utc).total_seconds()
        if abs(elapsed-intended)>tolerance:
            raise ValueError('Witness readings do not match the registered time budget.')


def prepare(protocol, packets):
    if protocol['study_kind'] not in ('physical_pilot','synthetic_fixture'):
        raise ValueError('Declare a physical pilot or a synthetic software fixture.')
    methods=protocol['methods']
    if set(x['view'] for x in methods.values())!=set(VIEWS):
        raise ValueError('Declare at least one method for each of the two witness readings.')
    if any(not SHA.fullmatch(item['artifact_sha256']) for item in methods.values()) or not SHA.fullmatch(protocol['calibration_sha256']):
        raise ValueError('Register the calibration and software digests.')
    grid=ordered_unique(protocol['times_s'],'Outcome times')
    if grid!=sorted(grid) or grid[0]!=0 or any(number(x,'Time')<0 for x in grid):
        raise ValueError('Outcome times start at zero and increase.')
    tol=number(protocol['time_tolerance_s'],'Clock tolerance')
    if tol<0 or any(number(v,'Functional limit')<0 for v in protocol['limits'].values()):
        raise ValueError('Use nonnegative physical units and tolerance.')
    candidates=set(ordered_unique(protocol['candidate_ids'],'Candidate IDs'))
    if len(protocol['witness_times_s'])!=2 or not 0<protocol['witness_times_s'][0]<protocol['witness_times_s'][1]:
        raise ValueError('Register the two equal-budget early times.')
    mother_ids=set(ordered_unique([r['mother'] for r in packets],'Mother IDs'))
    if mother_ids.intersection(protocol['calibration_mothers']):
        raise ValueError('Do not reuse calibration mothers in confirmation.')
    prepared={name:[] for name in methods}
    owner={}
    for p in packets:
        cutoff,release,witness=map(utc,(p['cutoff_utc'],p['release_utc'],p['witness_exposure_utc']))
        if not witness<cutoff<release or set(p['common'])!=set(COMMON) or set(p['witness'])!=set(VIEWS):
            raise ValueError('Register one complete paired input packet for each mother.')
        context=p['context']
        if number(context['thickness_m'],'Initial thickness')<=0 or not SHA.fullmatch(context['metadata_sha256']):
            raise ValueError('Include measured geometry and the associated preparation and boundary metadata hash.')
        witness_id=p['witness_specimen_id']
        identifiers={x['specimen_id'] for x in p['common'].values()}|{witness_id}
        if not witness_id or any(owner.setdefault(s,p['mother'])!=p['mother'] for s in identifiers):
            raise ValueError('A companion specimen belongs to one original mother.')
        for item in p['common'].values():
            check_observation(item,cutoff)
        for view in VIEWS:
            rows=p['witness'][view]
            for item,t in zip(rows,protocol['witness_times_s'],strict=True):
                if item['specimen_id']!=witness_id:
                    raise ValueError('The information comparison uses the same witness geometry and specimen.')
                check_observation(item,cutoff,witness,t,tol)
        ordered_unique([r['id'] for r in p['requests']],'Request IDs within a mother')
        for req in p['requests']:
            requirements(req,protocol)
        for method,item in methods.items():
            prepared[method].append({k:p[k] for k in ('mother','cutoff_utc','context','common','requests')}
                                    | {'witness':p['witness'][item['view']],'witness_view':item['view']})
    return prepared,candidates,owner


def check_choices(protocol, packets, choices, candidates):
    keys={p['mother']:{r['id'] for r in p['requests']} for p in packets}
    if set(choices)!=set(keys):
        raise ValueError('Freeze decisions for all registered mothers.')
    for mother,requests in choices.items():
        if set(requests)!=keys[mother]:
            raise ValueError('Freeze decisions for every registered request.')
        for design in requests.values():
            if set(design)!=set(protocol['methods']) or any(a is not None and a not in candidates for a in design.values()):
                raise ValueError('Every method chooses a registered recipe or records no decision.')


def read_panel(protocol, packets, choices, observations, freeze_at, companions):
    onsets={p['mother']:utc(p['release_utc']) for p in packets}
    allowed={(m,a) for m,requests in choices.items() for choices_one in requests.values() for a in choices_one.values() if a is not None}
    index={}
    specimens={}
    for row in observations:
        m,a,env=row['mother'],row['candidate'],row['environment']
        spot=(m,a,env)
        if (m,a) not in allowed or env not in ('wet','weak') or row['specimen_id'] in companions:
            raise ValueError('Outcomes require a registered independent main specimen.')
        run=utc(row['exposure_utc'])
        identity=(row['specimen_id'],run)
        impossible_start=run<onsets[m] or (protocol['study_kind']=='physical_pilot' and run<=freeze_at)
        if not identity[0] or specimens.setdefault(identity[0],spot)!=spot or impossible_start:
            raise ValueError('Every main sample must have an exclusive candidate, environment and post-freeze start.')
        saved=index.setdefault(spot,{'identity':identity,'stamps':{}})
        if saved['identity']!=identity:
            raise ValueError('Do not splice main-specimen trajectories across the same panel.')
        channel,stamp=row['channel'],row['planned_s']
        key=(env,channel,stamp)
        if row['unit']!=OUTCOMES.get(channel) or stamp not in protocol['times_s'] or key in saved['stamps']:
            raise ValueError('Use registered units and one reading per channel and time.')
        if row['low'] is None or row['high'] is None:
            if row['low'] is not None or row['high'] is not None or not row.get('missing_reason'):
                raise ValueError('Unavailable outcomes need both null bounds and a cause.')
            elapsed=stamp
        else:
            if number(row['low'],'Lower physical measurement')>number(row['high'],'Upper physical measurement'):
                raise ValueError('Measurement bounds must be ordered.')
            acquired=utc(row['acquired_utc'])
            elapsed=(acquired-run).total_seconds()
            if protocol['study_kind']=='physical_pilot' and (acquired>datetime.now(timezone.utc) or (elapsed<0 and stamp>0) or acquired<=freeze_at):
                raise ValueError('Measurement timestamps contradict the physical release.')
        saved['stamps'][key]={**row,'actual_s':elapsed}
    return index


def score(protocol, packets, choices, measurements, freeze_at, companions):
    panel=read_panel(protocol,packets,choices,measurements,freeze_at,companions)
    detail={}
    for p in packets:
        mother=p['mother']
        per_mother={}
        for req in p['requests']:
            per_request={}
            registered=requirements(req,protocol)
            selected=choices[mother][req['id']]
            evidence={}
            for action in set(selected.values())-{None}:
                raw={k:v for env in ('wet','weak') for k,v in panel.get((mother,action,env),{'stamps':{}})['stamps'].items()}
                evidence[action]=assess(registered,raw,protocol['time_tolerance_s'])
            for method,action in selected.items():
                status,clauses=evidence.get(action,('no_decision',[]))
                per_request[method]={'candidate':action,'status':status,'clauses':clauses}
            per_mother[req['id']]=per_request
        detail[mother]=per_mother
    return summarize(protocol['study_kind'],protocol['methods'],detail)


def summarize(kind,methods,detail):
    summary={}
    for m in methods:
        status=[[r[m]['status'] for r in v.values()] for v in detail.values()]
        active=sum(any(x!='no_decision' for x in s) for s in status)
        lower=sum('fail' in s for s in status)
        upper=sum('fail' in s or 'unresolved' in s for s in status)
        pass_rates=[[s.count('pass')/len(s),(s.count('pass')+s.count('unresolved'))/len(s)] for s in status]
        summary[m]={'active_mothers':active,'maternal_failure_count_bounds':[lower,upper],
                    'selected_failure_fraction_bounds':[lower/active,upper/active] if active else None,
                    'success_fraction_equal_mother_bounds':[math.fsum(x)/len(status) for x in zip(*pass_rates,strict=True)]}
    return {'study_kind':kind,'mothers':len(detail),'method_summary':summary,'requests':detail,
            'scope':'All registered observed times, distinct wet and weak intact panels.'}


def combine(reports):
    first=reports[0]
    methods=first['method_summary']
    detail={}
    for r in reports:
        if (r['protocol_sha256']!=first['protocol_sha256'] or r['study_kind']!=first['study_kind']
                or set(r['method_summary'])!=set(methods)):
            raise ValueError('Combine only confirmation blocks from the identical frozen protocol.')
        if set(detail).intersection(r['requests']):
            raise ValueError('A mother appears in more than one scoring block.')
        detail.update(r['requests'])
    return {**summarize(first['study_kind'],methods,detail),'protocol_sha256':first['protocol_sha256']}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    cmds=parser.add_subparsers(dest='phase',required=True)
    layout={'inputs':('protocol','readings','output_dir'), 'freeze':('protocol','readings','choices','frozen'),
            'score':('protocol','readings','choices','frozen','outcomes','report')}
    for name,fields in layout.items():
        cmd=cmds.add_parser(name)
        for f in fields:
            cmd.add_argument(f,type=Path)
    merge=cmds.add_parser('combine')
    merge.add_argument('report',type=Path)
    merge.add_argument('scored_blocks',nargs='+',type=Path)
    a=parser.parse_args()
    if a.phase=='combine':
        received=[read_packet(p) for p in a.scored_blocks]
        data=combine([p[0] for p in received])
        write_new(a.report,{**data,'block_sha256':[p[1] for p in received]})
        return
    protocol,p_hash=read_packet(a.protocol)
    data,d_hash=read_packet(a.readings)
    isolated,candidates,companions=prepare(protocol,data)
    if a.phase=='inputs':
        a.output_dir.mkdir(exist_ok=False)
        for i,(name,packets) in enumerate(isolated.items(),start=1):
            write_new(a.output_dir/f'{i:02d}.json',{'method':name,'protocol_sha256':p_hash,'readings_sha256':d_hash,'inputs':packets})
        return
    chosen,c_hash=read_packet(a.choices)
    check_choices(protocol,data,chosen,candidates)
    inputs={'protocol':p_hash,'readings':d_hash,'choices':c_hash,'source':sha256(Path(__file__)),
            'base_source':sha256(Path(__file__).with_name('validate_hydrogel_pilot.py'))}
    if a.phase=='freeze':
        now=datetime.now(timezone.utc)
        if protocol['study_kind']=='physical_pilot' and any(not utc(x['cutoff_utc'])<=now<utc(x['release_utc']) for x in data):
            raise ValueError('Freeze after readout deadlines and before any candidate exposure.')
        write_new(a.frozen,{'frozen_utc':now.isoformat(),'inputs_sha256':inputs})
    else:
        committed,f_hash=read_packet(a.frozen)
        if committed['inputs_sha256']!=inputs:
            raise ValueError('Frozen data, choice, protocol or source has changed.')
        outcomes,o_hash=read_packet(a.outcomes)
        report=score(protocol,data,chosen,outcomes,utc(committed['frozen_utc']),companions)
        write_new(a.report,{**report,'protocol_sha256':p_hash,'frozen_sha256':f_hash,'outcome_sha256':o_hash})


if __name__=='__main__':
    main()
