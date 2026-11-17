"""Test whether a learned hydration law changes tight-window electrical design."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timedelta, timezone
from itertools import repeat
from pathlib import Path

import numpy as np
from scipy.special import softmax

from audit_patch_reservoir_study import extremes, normalized_clauses
from check_window_random_streams import preflight
from gel_reactive_port_timing import read_json, sha256
from gel_reactive_two_read_assay import write_new
from patch_exchange_forward import exchange_bank
from patch_learned_water_ion import joint_trace
from patch_reservoir_joint_bank import joint_bank, prepared_fields
from patch_thickness_catalogue_benchmark import geometry
from tight_window_coupon_study import inputs, training
from tight_window_phase_model import design, fit, predictive_margins, rank
from tight_window_phase_study import parents
from tight_window_witness import catalogue, scoped_task
from window_risk_cases import packet
from window_risk_forecasts import elapsed, prior
from window_spatial_probe import changed_mechanism, conditional_release, observed_zone


COUPON = Path('runs/patch_tight_thin_coupon_20261117T035923687266Z/REGISTERED.json')
HYDROL = Path('runs/hydraulic_process_transfer_20261117T043746009391Z')
ARMS = ('constant_physics','learned_physics','direct_trees')


def register():
    plan = read_json(COUPON)
    from_config = read_json(HYDROL/'REGISTERED.json')
    now = datetime.now(timezone.utc)
    plan.update(registered_utc=now.isoformat(), registered_shanghai=now.astimezone(timezone(timedelta(hours=8))).isoformat())
    plan['scientific_question'] = 'Does improved hydration physics help electrical trajectories and tight-window design?'
    plan['event_roles'] = {'confirmation':{'groups':72,'shift_fraction':.50,'nominal_seed':1173545309,
        'shift_seed':1173546991,'mechanism_seed':1173548347,'spatial_noise_salt':22706500,'hydraulic_salt':17960560}}
    plan['tight']['witness_entropy'] = 9025511161
    plan['process_transfer'] = {
        'predicted_hydration': 'Use the central curve already fitted from independent dense simulated hydration training profiles. '
                                'Only change D(h) between the same-number, same-prior conservative two physics arms.',
        'direct_model': 'Matched early inputs, existing ExtraTrees and joint residuals learned from the prior 224 training mothers. '
                        'It outputs signed window margins. The learned closure used separate dense simulated hydration labels.',
        'geometry': 'Same .65 fixed-areal-inventory witness and 27 product actions, separate forward histories on each geometry.',
        'comparison': 'Forecast before revealing 72 new mothers, no hyperparameter choice or new gate. Compare mechanical and '
                      'electrical full-process paths for both physics arms, and ungated decisions and Brier against all three.',
        'scope': 'Unmeasured ionic closure unchanged. Simulator family does not confer real patch evidence or certify risk.',
        'atoms_per_water': 32, 'bootstrap_seed': 12545213,
        'hydraulic_freeze':(HYDROL/'FROZEN.npz').as_posix()}
    files = {*plan['hashes'], COUPON.as_posix(), 'patch_process_transfer_study.py','patch_learned_water_ion.py',
             *(str(Path(p)) for p in from_config['source']), *((HYDROL/name).as_posix() for name in ('REGISTERED.json','FROZEN.npz'))}
    plan['hashes'] = {Path(f).as_posix():sha256(Path(f)) for f in sorted(files)}
    addrs = preflight(plan)
    for earlier in (COUPON,Path('runs/patch_tight_factorial_20261117T043535980478Z/REGISTERED.json')):
        if addrs.keys() & preflight(read_json(earlier)).keys():
            raise ValueError('Overlapping simulator addresses.')
    run = Path('runs')/now.strftime('patch_process_transfer_%Y%m%dT%H%M%S%fZ')
    run.mkdir()
    write_new(run/'REGISTERED.json',plan)
    write_new(run/'STREAM_CHECK.json',{'protocol_sha256':sha256(run/'REGISTERED.json'), 'distinct_scalar_addresses':len(addrs),
                                      'witness_entropy':plan['tight']['witness_entropy'],'witness_role':3})
    print(run.as_posix(),flush=True)


def physical_case(group, task, plan, learned):
    wet_context = inputs(group,task,plan)
    task = scoped_task(task,plan)
    raw = np.asarray(wet_context['raw_inputs']['thin_electric'])
    observed = raw[:,-2:]
    shared,dry,_ = packet(group,plan,task)
    nominal_witness = task['actions'][plan['fixed_action']]
    atoms = prior(group['known'],{**plan,'continuum_count':plan['process_transfer']['atoms_per_water']},True)
    cells,step = plan['prediction_mesh']
    ref_stock = prepared_fields(group['known'],[nominal_witness],task['mechanism'],atoms[:,4:],cells)[:,0]
    stock = ref_stock[:,:cells//2].sum(axis=1)/ref_stock.sum(axis=1)
    _, taste = conditional_release(shared,atoms,plan)
    with np.errstate(divide='ignore'):
        initial = np.log(taste)-.5*((dry[:,None]-stock)/plan['probes']['sigma_half'])**2
    output = {'mother':group['parent'],'direct':{'raw_inputs':{'depth_mid':raw.tolist()}}}
    t = elapsed(task,plan)
    snapshots = np.r_[0.,plan['spatial_probe']['times']]
    for name,law in zip(ARMS[:2],(np.zeros_like(learned),learned),strict=True):
        response, evidence, conservation = [], [], 0.
        for scale in plan['spatial_probe']['water_grid']:
            context = {**group['known'],'water_D':group['known']['water_D']*scale}
            thin,cfg = geometry(context,task,{'scale':plan['coupon']['witness_scale']},1.)
            short, water, error = joint_trace(thin,[nominal_witness],cfg,1.,atoms[:,:4],atoms[:,4:],cells,step,snapshots,law)
            conservation = max(conservation,error)
            x = observed_zone(water[0,1],plan['spatial_probe']['bands'][0])
            g = np.log(short[:,-1,0,3]/thin['thickness'])
            e = -.5*((observed[:,0,None]-x)/plan['tight']['first_sigma'])**2
            e = e-.5*((observed[:,1,None]-g)/plan['tight']['sensors']['log_conductance'])**2
            evidence.append(initial+e)
            blocks = []
            for variant in catalogue(task,plan):
                parent,settings = geometry(context,task,variant,1.)
                def trace(activity):
                    return joint_trace(parent,variant['actions'],settings,activity,atoms[:,:4],atoms[:,4:],cells,step,t,law)
                wet, weak = trace(task['bath']['wet']),trace(task['bath']['weak'])
                conservation = max(conservation,wet[2],weak[2])
                block = np.stack((wet[0],weak[0]),axis=1)
                block[...,3] /= variant['scale']
                blocks.append(block)
            response.append(np.concatenate(blocks,axis=3))
        scenes = np.concatenate(response)
        weights = softmax(np.concatenate(evidence,axis=1),axis=1)
        clauses = normalized_clauses(extremes((scenes[:,0],scenes[:,1]),t,task),task)[:,0,:]
        margin = clauses.max(axis=-1)
        output[name] = {'paths': np.einsum('sa,aetjn->setjn',weights,scenes),
                        'probability':weights@(margin>0.),'margin':weights@np.maximum(margin,0.),
                        'conservation':conservation}
    return output


def realized_case(group, task, plan):
    task = scoped_task(task,plan)
    t = elapsed(task,plan)
    settings = changed_mechanism(group,plan)
    ref = exchange_bank if group['shift'] else joint_bank
    kw = {'outcome_cfg':settings['exchange']} if group['shift'] else {}
    cells,step = plan['truth_mesh']
    traces = []
    for v in catalogue(task,plan):
        known,cfg = geometry(group['known'],task,v,1.)
        values = np.stack([ref(known,v['actions'],cfg,task['bath'][kind],group['kinetic'],group['stock'],cells,step,t,**kw)
                           for kind in ('wet','weak')],axis=1)
        values[...,3] /= v['scale']
        traces.append(values)
    all_paths = np.concatenate(traces,axis=3)
    margins = normalized_clauses(extremes((all_paths[:,0],all_paths[:,1]),t,task),task)[:,0,:].max(axis=-1)
    return group['parent'],group['shift'],all_paths,margins


def mats(records,task,plan):
    seen,y = training(plan,task)
    current = [r['direct'] for r in records]
    def mapped(rows):
        return [{'raw_inputs':{'depth_mid':r['raw_inputs']['thin_electric']}} for r in rows]
    x = design(mapped(seen),task,plan,'direct_depth')
    z = design(current,task,plan,'direct_depth')
    model,residual = fit(x,y,'direct_depth',plan['phase']['fit_seed'])
    pred = predictive_margins(z,model,residual)
    return {'probability':pred[...,2],'choice':rank(pred)['posterior_violation']['choice']}


def freeze(records,task,plan,run):
    output = {'mother':np.array([r['mother'] for r in records])}
    for name in ARMS[:2]:
        for item in ('paths','probability','margin','conservation'):
            output[name+'_'+item] = np.array([r[name][item] for r in records])
        p, m = (output[name+'_'+item] for item in ('probability','margin'))
        output[name+'_choice'] = np.argmin(p+plan['risk_weight']*m,axis=-1)
    for k,v in mats(records,task,plan).items():
        output['direct_trees_'+k] = v
    with (run/'FROZEN.npz').open('xb') as f:
        np.savez_compressed(f,**output)


def score(records,run,plan):
    fixed = np.load(run/'FROZEN.npz')
    names, shifted, paths, margins = [np.asarray(v) for v in zip(*records,strict=True)]
    if not np.array_equal(names,fixed['mother']):
        raise ValueError('Revealed mothers do not match the forecast freeze.')
    report, by_mother = {}, {}
    for set_name, group in (('all',np.ones(len(names),dtype=bool)),('nominal',~shifted),('exchange',shifted)):
        results = {}
        rng = np.random.default_rng(plan['process_transfer']['bootstrap_seed'])
        n = int(group.sum())
        bootstrap = rng.integers(n,size=(5000,n))
        subset = margins[group]
        for arm in ARMS:
            p = fixed[arm+'_probability'][group]
            choice = fixed[arm+'_choice'][group]
            margin = np.take_along_axis(subset,choice[...,None],axis=-1)[...,0]
            correct = (margin<=0.)
            by_mother[arm] = correct.mean(axis=1)
            results[arm] = {'brier':float(np.square(p-(subset>0)).mean()), 'ungated_successes':int(correct.sum()),
                            'mean_positive_regret':float((np.maximum(margin,0.)-np.maximum(subset.min(axis=-1),0.)).mean())}
            if arm!=ARMS[-1]:
                error = abs(fixed[arm+'_paths'][group]-paths[group])
                for i,typ in enumerate(('wet','weak')):
                    results[arm][typ+'_future_conductance_mae'] = float(error[:,:,i,21:,:,3].mean())
                results[arm]['wet_future_modulus_mae'] = float(error[:,:,0,21:,:,1:3].mean())
        deltas = {}
        for control in ('constant_physics','direct_trees'):
            d = by_mother['learned_physics']-by_mother[control]
            deltas[control] = {'difference':float(d.mean()),'ci95':np.quantile(d[bootstrap].mean(axis=1),[.025,.975]).tolist()}
        report[set_name] = {'mothers':n,'feasible_requests':int((subset.min(axis=-1)<=0.).sum()),'methods':results,
                            'learned_choice_gain':deltas}
    with (run/'TRUTH.npz').open('xb') as f:
        np.savez_compressed(f,mother=names,exchange=shifted,paths=paths,margin=margins)
    write_new(run/'SCORED.json',{'protocol_sha256':sha256(run/'REGISTERED.json'),'freeze_sha256':sha256(run/'FROZEN.npz'),
                 'ion_closure_max':{arm:float(fixed[arm+'_conservation'].max()) for arm in ARMS[:2]},
                 'completed_utc':datetime.now(timezone.utc).isoformat(),'results':report})


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('phase',choices=('register','forecast','score'))
    cli.add_argument('run',nargs='?',type=Path)
    cli.add_argument('--workers',type=int,default=15)
    a = cli.parse_args()
    if a.phase=='register':
        register()
        return
    address = a.run/'REGISTERED.json'
    plan = read_json(address)
    for filename,h in plan['hashes'].items():
        if sha256(Path(filename))!=h:
            raise ValueError(f'Changed source {filename}.')
    task = read_json(Path(plan['task_path']))
    groups,local = parents(plan,'confirmation',task)
    freezing = a.phase=='forecast'
    if freezing:
        curves = np.load(plan['process_transfer']['hydraulic_freeze'])['learned_curves']
    else:
        f = read_json(a.run/'FORECAST_RECEIPT.json')
        if f['frozen_sha256']!=sha256(a.run/'FROZEN.npz') or f['protocol_sha256']!=sha256(address):
            raise ValueError('Predictions were not frozen for the registered study.')
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        jobs = pool.map(physical_case,groups,repeat(task),repeat(local),repeat(curves[1])) if freezing else pool.map(realized_case,groups,repeat(task),repeat(local))
        values = []
        for item in jobs:
            values.append(item)
            if len(values)%12==0:
                print(a.phase,len(values),flush=True)
    if freezing:
        freeze(values,task,plan,a.run)
        write_new(a.run/'FORECAST_RECEIPT.json',{'protocol_sha256':sha256(address),'frozen_sha256':sha256(a.run/'FROZEN.npz'),
                    'completed_utc':datetime.now(timezone.utc).isoformat()})
    else:
        score(values,a.run,plan)


if __name__=='__main__':
    main()
