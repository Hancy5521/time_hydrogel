"""Evaluate mechanism evidence and fixed pooling for tight-window trajectories."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timedelta, timezone
from itertools import repeat
from pathlib import Path

import numpy as np
from scipy.special import logsumexp, softmax
from sklearn.metrics import roc_auc_score

from check_window_random_streams import preflight
from gel_reactive_port_timing import read_json, sha256
from gel_reactive_two_read_assay import write_new
from patch_learned_water_ion import joint_trace
from patch_process_transfer_study import ARMS, freeze, physical_case, realized_case
from patch_reservoir_joint_bank import prepared_fields
from patch_thickness_catalogue_benchmark import geometry
from tight_window_phase_study import parents
from window_risk_cases import packet
from window_risk_forecasts import prior
from window_spatial_probe import conditional_release, observed_zone


OLD = Path('runs/patch_process_transfer_20261117T044639365780Z')
EXTRA = ('equal_mixture','evidence_mixture')


def register():
    plan = read_json(OLD/'REGISTERED.json')
    now = datetime.now(timezone.utc)
    plan.update(registered_utc=now.isoformat(), registered_shanghai=now.astimezone(timezone(timedelta(hours=8))).isoformat())
    plan['scientific_question'] = 'Does retaining and weighting both physically permissible mechanisms transfer between regimes?'
    plan['event_roles'] = {'confirmation':{'groups':128,'shift_fraction':.5,'nominal_seed':1340044181,
        'shift_seed':1340044699,'mechanism_seed':1340045813,'spatial_noise_salt':23025391,'hydraulic_salt':18473220}}
    plan['tight']['witness_entropy'] = 9414485742
    plan['mechanism_mixture'] = {
        'basis': 'Mechanism effect reversal was observed in the earlier 72. New cohort disjoint, same functioning catalogue and inputs.',
        'weights': 'Prior half each. Compare equal weights with exact discrete marginal witness-likelihood weighting. '
                   'All 4×32 states within each law have same prior support as the physical comparator. '
                   'Common assays weight the paired chemistry subspace first. A different law predicts its own two thin witness reads. '
                   'No per-case or per-mother truth label is used, and observations from the second state are not reused.',
        'baseline': 'Hold constant and learned single-law models and direct trees. Evaluate pooled and by true generator offline, '
                    'ungated design only. Neither earlier score nor held outcomes are used to set blending or thresholds.',
        'primary': 'Unweighted confirmation by whole-mother paired utility and Brier of evidence mixture against equal mixture and original controls.',
        'definition': 'Likelihood comparison is about compatibility with the two simulated predictor families, not calibrated identification of the true ion mechanism.',
        'bootstrap_seed': 5633279}
    previous = preflight(read_json(OLD/'REGISTERED.json'))
    recent = preflight(plan)
    if previous.keys() & recent.keys():
        raise ValueError('Duplicate generator addresses.')
    files = {*plan['hashes'], 'patch_process_mixture_study.py', (OLD/'REGISTERED.json').as_posix(),(OLD/'SCORED.json').as_posix()}
    plan['hashes'] = {f:sha256(Path(f)) for f in sorted(files)}
    run = Path('runs')/now.strftime('patch_process_mixture_%Y%m%dT%H%M%S%fZ')
    run.mkdir()
    write_new(run/'REGISTERED.json',plan)
    write_new(run/'STREAM_CHECK.json',{'registration_sha256':sha256(run/'REGISTERED.json'),'new_address_count':len(recent),
                                      'disjoint_from_primary':len(previous),'witness_entropy':plan['tight']['witness_entropy']})
    print(run.as_posix(),flush=True)


def branch_evidence(group,task,plan,readings,learned):
    shared,dry,_ = packet(group,plan,task)
    nominal_action = task['actions'][plan['fixed_action']]
    atoms = prior(group['known'],{**plan,'continuum_count':plan['process_transfer']['atoms_per_water']},True)
    _, likes = conditional_release(shared,atoms,plan)
    cells, dt = plan['prediction_mesh']
    prepared = prepared_fields(group['known'],[nominal_action],task['mechanism'],atoms[:,4:],cells)[:,0]
    stock_half = prepared[:,:cells//2].sum(axis=1)/prepared.sum(axis=1)
    with np.errstate(divide='ignore'):
        base = np.log(likes)-.5*((dry[:,None]-stock_half)/plan['probes']['sigma_half'])**2
    times = np.r_[0.,plan['spatial_probe']['times']]
    sensor_sigma = np.array([plan['tight']['first_sigma'],plan['tight']['sensors']['log_conductance']])
    loglik = []
    for curve in (np.zeros_like(learned),learned):
        family = []
        for factor in plan['spatial_probe']['water_grid']:
            witness,cfg = geometry(group['known'],task,{'scale':plan['coupon']['witness_scale']},factor)
            series,h,_ = joint_trace(witness,[nominal_action],cfg,1.,atoms[:,:4],atoms[:,4:],cells,dt,times,curve)
            probe = observed_zone(h[0,1],plan['spatial_probe']['bands'][0])
            electrical = np.log(series[:,-1,0,3]/witness['thickness'])
            predicted = np.column_stack((np.full(len(atoms),probe),electrical))
            ll = base-.5*np.square((readings[:,None]-predicted)/sensor_sigma).sum(axis=-1)
            family.append(ll)
        loglik.append(logsumexp(np.concatenate(family,axis=1),axis=1))
    return softmax(np.stack(loglik,axis=1),axis=1)[:,1]


def simulate(group,task,plan,curve):
    record = physical_case(group,task,plan,curve)
    reads = np.asarray(record['direct']['raw_inputs']['depth_mid'])[:,-2:]
    record['evidence'] = branch_evidence(group,task,plan,reads,curve)
    return record


def summarize(plan,run,records):
    fixed = np.load(run/'FROZEN.npz')
    names, truth, paths, margin = [np.asarray(x) for x in zip(*records,strict=True)]
    if not np.array_equal(names,fixed['mother']):
        raise ValueError('Revealed labels do not match frozen prediction identities.')
    p = {k:fixed[k+'_probability'] for k in ARMS}
    chosen = {k:fixed[k+'_choice'] for k in ARMS}
    correct = {}
    prob_law = np.load(run/'FAMILY_WEIGHTS.npz')['learned_family_weight']
    if not np.isfinite(prob_law).all() or prob_law.shape!=margin.shape[:2]:
        raise ValueError('Inconsistent frozen mechanism weights.')
    for name,alpha in zip(EXTRA,(np.full_like(prob_law,.5),prob_law),strict=True):
        weight = alpha[...,None]
        p[name] = (1.-weight)*p[ARMS[0]]+weight*p[ARMS[1]]
        m = (1.-weight)*fixed[ARMS[0]+'_margin']+weight*fixed[ARMS[1]+'_margin']
        chosen[name] = np.argmin(p[name]+plan['risk_weight']*m,axis=-1)
    results = {}
    for name in p:
        value = np.take_along_axis(margin,chosen[name][...,None],axis=2)[...,0]
        correct[name] = (value<=0.).mean(axis=1)
    for name,part in (('all',np.ones_like(truth,dtype=bool)),('nominal',~truth),('exchange',truth)):
        stats = {}
        n = int(part.sum())
        delta = np.random.default_rng(plan['mechanism_mixture']['bootstrap_seed']).integers(n,size=(5000,n))
        for method,v in correct.items():
            stats[method] = {'successes':int(2*v[part].sum()),'brier':float(np.square(p[method][part]-(margin[part]>0.)).mean())}
        diff = {m:(correct['evidence_mixture']-right)[part] for m,right in correct.items() if m!='evidence_mixture'}
        comparison = {m:{'difference':float(d.mean()),'ci95':np.quantile(d[delta].mean(axis=1),[.025,.975]).tolist()}
                      for m,d in diff.items()}
        results[name] = {'mothers':n,'requests_with_feasible_action':int((margin[part].min(axis=-1)<=0.).sum()),
                         'methods':stats,'evidence_mixture_differences':comparison,'mean_learned_law_weight':float(prob_law[part].mean())}
    with (run/'TRUTH.npz').open('xb') as f:
        np.savez_compressed(f,mother=names,exchange=truth,margin=margin)
    write_new(run/'SCORED.json',{'completed_utc':datetime.now(timezone.utc).isoformat(), 'results':results,
              'protocol_sha256':sha256(run/'REGISTERED.json'),'frozen_sha256':sha256(run/'FROZEN.npz'),
              'family_sha256':sha256(run/'FAMILY_WEIGHTS.npz'),
              'mechanism_auc_audit_only':float(roc_auc_score(truth,prob_law.mean(axis=1)))})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase',choices=('register','forecast','score'))
    p.add_argument('run',nargs='?',type=Path)
    p.add_argument('--workers',type=int,default=15)
    a = p.parse_args()
    if a.phase=='register':
        register()
        return
    protocol = a.run/'REGISTERED.json'
    plan = read_json(protocol)
    for path,h in plan['hashes'].items():
        if sha256(Path(path))!=h:
            raise ValueError(f'Changed registered source {path}.')
    task = read_json(Path(plan['task_path']))
    groups,local = parents(plan,'confirmation',task)
    forecast = a.phase=='forecast'
    if forecast:
        law = np.load(plan['process_transfer']['hydraulic_freeze'])['learned_curves'][1]
    else:
        f = read_json(a.run/'FORECAST_RECEIPT.json')
        if any(sha256(a.run/name)!=value for name,value in f['freeze_hashes'].items()):
            raise ValueError('Scoring requires a valid prediction freeze.')
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        jobs = pool.map(simulate,groups,repeat(task),repeat(local),repeat(law)) if forecast else pool.map(realized_case,groups,repeat(task),repeat(local))
        records = []
        for record in jobs:
            records.append(record)
            if len(records)%16==0:
                print(a.phase,len(records),flush=True)
    if forecast:
        freeze(records,task,plan,a.run)
        with (a.run/'FAMILY_WEIGHTS.npz').open('xb') as f:
            np.savez_compressed(f,learned_family_weight=np.array([r['evidence'] for r in records]))
        write_new(a.run/'FORECAST_RECEIPT.json',{'completed_utc':datetime.now(timezone.utc).isoformat(),
                 'protocol_sha256':sha256(protocol),'freeze_hashes':{name:sha256(a.run/name) for name in ('FROZEN.npz','FAMILY_WEIGHTS.npz')}})
    else:
        summarize(plan,a.run,records)


if __name__=='__main__':
    main()
