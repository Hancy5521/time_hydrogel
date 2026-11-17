"""Test whether amplifying the early electrical witness improves tight-window delivery."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timedelta, timezone
from itertools import repeat
from pathlib import Path

import numpy as np

from check_window_random_streams import preflight
from gel_reactive_port_timing import read_json, sha256
from gel_reactive_two_read_assay import write_new
from patch_thickness_catalogue_benchmark import geometry
from tight_window_phase_model import design, fit, predictive_margins, rank, witness_inputs
from tight_window_phase_study import parents
from tight_window_study import RANKS, ROLE_INDEX, arrays, open_rows
from tight_window_witness import reveal
from window_event_calibration import certify_policy
from window_event_policy import development_choice, outcomes
from window_group_calibration import take_actions


PANELS = ('standard_depth', 'thin_depth', 'thin_electric')


def register(sources):
    now = datetime.now(timezone.utc)
    plan = read_json(sources[1]/'REGISTERED.json')
    previous = [read_json(p/'REGISTERED.json') for p in sources]
    plan.update(registered_utc=now.isoformat(), registered_shanghai=now.astimezone(timezone(timedelta(hours=8))).isoformat())
    plan['scientific_question'] = ('Does a physically more sensitive, thinner witness make the tight '
                                   'window decidable early at the same observation count and delivery risk?')
    panels = dict(development=160, calibration=360, confirmation=192)
    for k, (role, size) in enumerate(panels.items()):
        plan['event_roles'][role].update(groups=size, nominal_seed=937780221 + 63001*k, shift_seed=937780763 + 63001*k,
            mechanism_seed=937781129 + 63001*k, spatial_noise_salt=6520211 + 18481*k, hydraulic_salt=9209934 + 126153*k)
    plan['tight'].update(witness_entropy=5414401081)
    plan['coupon'] = {'training_source_runs': [p.as_posix() for p in sources], 'witness_scale': .65,
                      'one_witness_each': 'Use one witness per arm and the same times .4 and .8. Two hydration '
                        'scalars at standard or .65 thickness, or first local water and later log G at .65 thickness. '
                        'Keep measured common five, candidate catalogue and thresholds. Hold the sensor coupon areal '
                        'stock fixed under thickness change, and same wet-transfer condition. Noise assumptions equal '
                        'for each type, .016 hydration and .08 on log G. Time/count only, no equivalence in lab cost.',
                      'inputs': 'Observed process context and seven scalar observations only. No full PDE posterior '
                        'is supplied to any arm. Same ExtraTrees signed clause regression and paired out-of-fold '
                        'residual draws across all arms. Both consumed development groups enter training only.',
                      'primary': 'Thin electrical versus equally thin double-depth witness. Standard double-depth is a '
                        'secondary fixed control. Use the same prospective development gate per arm, no rule if none '
                        'locks. 10% conditional risk and activity targets at .05/3 per eligible fixed arm on 360 '
                        'fresh mothers. All locked arms scored on 192 fresh mothers regardless of certificate. '
                        'Primary gain large enough if >= .025 and 95% paired interval above zero and both certify. '
                        'If only one arm certifies, report it as a single policy result and conditional request type.',
                      'evidence': 'Same specified synthetic population. The thinner sensor is distinct from the final specimen designs.'}
    files = {*plan['hashes'], Path(__file__).name,
             *(str(p/name) for p in sources for name in ('REGISTERED.json','DEVELOPMENT_SCORED.json')),
             (sources[0]/'PAIR_IDENTIFIABILITY_AUDIT.json').as_posix()}
    # Existing registrations use forward-slash paths, including on Windows.
    plan['hashes'] = {Path(name).as_posix(): sha256(Path(name)) for name in sorted(files)}
    current = preflight(plan)
    for old in previous:
        if current.keys() & preflight(old).keys():
            raise ValueError('Generator addresses overlap an earlier role.')
    run = Path('runs')/now.strftime('patch_tight_thin_coupon_%Y%m%dT%H%M%S%fZ')
    run.mkdir()
    write_new(run/'REGISTERED.json', plan)
    write_new(run/'STREAM_CHECK.json', {'protocol_sha256':sha256(run/'REGISTERED.json'), 'independent_scalar_addresses':len(current),
              'witness_addresses_by_role':panels,'fresh_witness_entropy':plan['tight']['witness_entropy']})
    print(run.as_posix(),flush=True)


def inputs(group, task, plan):
    standard = witness_inputs(group, task, plan)
    scale = plan['coupon']['witness_scale']
    known, corrected = geometry(group['known'], task, {'scale':scale}, 1.)
    # The thin witness changes only two sensor channels. The five separate assays are the standard ones.
    thin = witness_inputs({**group,'known':known}, {**task,'mechanism':corrected}, plan)
    x = np.asarray(standard['depth_mid'])
    views = {'standard_depth':x, 'thin_depth':np.column_stack((x[:,:-2],np.asarray(thin['depth_mid'])[:,-2:])),
             'thin_electric':np.column_stack((x[:,:-2],np.asarray(thin['log_conductance'])[:,-2:]))}
    return {'mother':group['parent'], 'raw_inputs':{k:v.tolist() for k,v in views.items()}}


def training(plan, task):
    data, labels = [], []
    for location in plan['coupon']['training_source_runs']:
        path = Path(location)
        old = {**read_json(path/'REGISTERED.json'), 'coupon':plan['coupon']}
        groups, local = parents(old,'development',task)
        rows = open_rows(path/'DEVELOPMENT_SCORED.json',path/'REGISTERED.json')['rows']
        if [g['parent'] for g in groups] != [r['mother'] for r in rows]:
            raise ValueError('Training identities changed when computing the alternative witness.')
        data.extend(inputs(g,task,local) for g in groups)
        labels.extend(r['clauses'] for r in rows)
    return data,np.asarray(labels)


def predictions(rows, target, task, plan):
    data, y = training(plan,task)
    models = {}
    for panel in PANELS:
        def single_view(items):
            return [{'raw_inputs':{'depth_mid':r['raw_inputs'][panel]}} for r in items]

        kind = 'direct_depth'
        x = design(single_view(data),task,plan,kind)
        new = design(single_view(rows),task,plan,kind)
        model,residuals = fit(x,y,kind,plan['phase']['fit_seed'])
        response = predictive_margins(new,model,residuals)
        banks = rank(response)
        for i,row in enumerate(target):
            row['forecasts'][panel] = {'banks':{r:{k:a[i].tolist() for k,a in f.items()} for r,f in banks.items()}}
        models[panel] = {'training_mothers':len(x),'mae':np.abs(residuals).mean(axis=(0,1)).tolist()}
    return models


def develop(rows, y, plan):
    result = {}
    for panel in PANELS:
        slates = {r:arrays(rows,panel,r) for r in RANKS}
        entry = development_choice(y,slates,plan['tight']['development_criterion'])
        entry['ungated'] = {r:outcomes(y,p['chosen'],np.ones(p['score'].shape,dtype=bool)) for r,p in slates.items()}
        result[panel] = entry
    return result


def test(locked,rows,y,role,seed):
    result, masks = {}, {}
    for panel,rule in locked.items():
        if rule['primary_method'] is None:
            continue
        p = arrays(rows,panel,rule['primary_method'])
        accept = p['score'] < rule['primary_cutoff']
        measured = take_actions(y,p['chosen'])
        masks[panel] = accept & (measured<=0.)
        result[panel] = {'outcomes':outcomes(y,p['chosen'],accept),'choices':p['chosen'].tolist(),'accepted':accept.tolist()}
        if role=='calibration':
            result[panel]['gate'] = certify_policy(measured,accept,max_risk=.10,min_activity=.10,delta=.05/len(PANELS))
    paired = ('thin_electric','thin_depth')
    if role=='confirmation' and all(p in masks for p in paired):
        effect = (masks[paired[0]].astype(float)-masks[paired[1]]).mean(axis=1)
        boot = np.random.default_rng(seed).integers(len(effect),size=(5000,len(effect)))
        result['primary_delta'] = {'estimate':float(effect.mean()),'ci95':np.quantile(effect[boot].mean(axis=1),[.025,.975]).tolist()}
    return result


def run_phase(args):
    protocol = args.run/'REGISTERED.json'
    plan = read_json(protocol)
    for name,h in plan['hashes'].items():
        if sha256(Path(name))!=h:
            raise ValueError(f'Changed registered input or source {name}.')
    stage = args.role.upper()
    saved = args.run/f'{stage}_FROZEN.json'
    target = saved if args.action=='forecast' else args.run/f'{stage}_SCORED.json'
    if target.exists():
        raise FileExistsError(target)
    receipt = {'protocol_sha256':sha256(protocol),'started_utc':datetime.now(timezone.utc).isoformat()}
    if args.role!='development':
        lockpath = args.run/'DEVELOPMENT_SCORED.json'
        locked = open_rows(lockpath,protocol)['results']
        receipt['development_sha256'] = sha256(lockpath)
        if not any(x['primary_method'] for x in locked.values()):
            raise ValueError('None of the recorded arms found a development rule.')
    if args.role=='confirmation':
        calpath = args.run/'CALIBRATION_SCORED.json'
        open_rows(calpath,protocol)
        receipt['calibration_sha256'] = sha256(calpath)
    task = read_json(Path(plan['task_path']))
    groups, local = parents(plan,args.role,task)
    if args.action=='score':
        x = open_rows(saved,protocol)['rows']
        receipt['frozen_sha256'] = sha256(saved)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        fn = inputs if args.action=='forecast' else reveal
        y = list(pool.map(fn,groups,repeat(task),repeat(local)))
    if args.action=='forecast':
        x = [{'mother':r['mother'],'forecasts':{}} for r in y]
        receipt['training_diagnostics'] = predictions(y,x,task,plan)
        receipt['rows'] = x
    else:
        if [r['mother'] for r in x]!=[r['mother'] for r in y]:
            raise ValueError('Forecast and outcome identities disagree.')
        receipt['rows'] = y
        m = np.asarray([r['clauses'] for r in y]).max(axis=-1)
        receipt['results'] = develop(x,m,plan) if args.role=='development' else test(locked,x,m,args.role,plan['tight']['bootstrap_seed'])
    receipt['completed_utc'] = datetime.now(timezone.utc).isoformat()
    write_new(target,receipt)
    print(target.as_posix(),flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='action',required=True)
    add = sub.add_parser('register')
    add.add_argument('source',type=Path,nargs=2)
    for phase in ('forecast','score'):
        a = sub.add_parser(phase)
        a.add_argument('run',type=Path)
        a.add_argument('role',choices=tuple(ROLE_INDEX))
        a.add_argument('--workers',type=int,default=15)
    a = p.parse_args()
    if a.action=='register':
        register(a.source)
    else:
        run_phase(a)


if __name__=='__main__':
    main()
