"""Post-score independent audit of the two locked targets on repaired RNG streams."""

import argparse
from datetime import datetime
from pathlib import Path

from audit_strict_window_gate import audit_certificate, run_audit
from audit_window_event_policy import compare, counters, digest, flags, inputs, policy, read, stamp, subset, write


def address_sets(specification):
    # The interval form independently reconstructs reserved addresses from the driver.
    witness,categories=set(),[]
    view=specification['spatial_probe']
    for role,setup in specification['event_roles'].items():
        n=setup['groups']
        starts=[view['snapshot_noise_seed']+setup['spatial_noise_salt']+x
                for x in (specification['seeds']['test'],specification['seeds']['shift_test'])]
        for start in starts:
            witness.update(range(start,start+n))
            categories.append(set(range(start,start+n)))
        h=view['shift_variation_seed']+setup['hydraulic_salt']
        categories.extend((set(range(h,h+n)),{setup['nominal_seed']+772},
                           {setup['shift_seed']+772},{setup['mechanism_seed']}))
    stress=specification['hidden_bath_stress']['seed']
    count=specification['event_roles']['hidden_bath_stress']['groups']
    categories.append(set(range(stress,stress+count)))
    total=set().union(*categories)
    if sum(map(len,categories))!=len(total):
        raise ValueError('The reserved stream addresses overlap.')
    return witness,total


def audit_streams(run, witness_receipt, plan, hashes):
    proof=read(witness_receipt)
    w,t=address_sets(plan)
    preflight=read(run/'STREAM_CHECK.json')
    compare(preflight['protocol_sha256'],hashes['REGISTERED.json'])
    compare(preflight['source_sha256'],plan['hashes']['check_window_random_streams.py'])
    compare(preflight['distinct_addresses'],len(t))
    started=read(run/'FROZEN.json')['started_utc']
    if datetime.fromisoformat(preflight['checked_utc'])>datetime.fromisoformat(started):
        raise ValueError('The RNG preflight postdates the freeze start.')
    witness=proof['protocols'][run.name]
    compare(witness['registered_sha256'],hashes['REGISTERED.json'])
    compare(witness['possible']['address_total'],len(w))
    compare(witness['actual']['address_total'],sum(x['groups'] for x in plan['event_roles'].values()))
    for scope in ('actual','possible'):
        parsed=witness[scope]
        compare(parsed['distinct'],parsed['address_total'])
        compare(parsed['reused_excess'],0)
        if any(run.name in d['groups'] for d in proof['joint_'+scope]['collision_groups']):
            raise AssertionError('A cross-protocol witness address overlaps the repaired run.')
    old=proof['protocols'][Path(plan['prior_study']).name]['actual']
    for collision in old['collision_groups']:
        if {'policy_train','policy_development'}.intersection(collision['groups']):
            raise AssertionError('Original development used a reused witness noise array.')
    return {'preflight_sha256':digest(run/'STREAM_CHECK.json'),'witness_receipt_sha256':digest(witness_receipt),
            'possible_witness_addresses':len(w),'full_reserved_addresses':len(t),
            'actual_witness_addresses':witness['actual']['distinct'],'older_training_and_development_unshared':True}


def audit_loose(run, hashes, plan):
    receipt=read(run/'LOOSE_CONFIRMED.json')
    compare(receipt['protocol_sha256'],hashes['REGISTERED.json'])
    compare(receipt['outcomes_sha256'],hashes['SCORED.json'])
    lock=read(Path(plan['prior_study'])/'POLICY_LOCK.json')
    f,c,s=(read(run/n) for n in ('FROZEN.json','CERTIFICATE.json','SCORED.json'))
    primary=lock['development']['primary_method']
    cutoff=lock['development']['primary_cutoff']
    compare(receipt['model'],primary)
    compare(receipt['threshold'],cutoff)
    if datetime.fromisoformat(receipt['completed_utc'])<datetime.fromisoformat(s['completed_utc']):
        raise ValueError('Loose-rule receipt predates its confirmation outcome receipt.')
    spec=plan['replay_loose_policy']
    settings={'risk_target':spec['target'],'activity_target':spec['activity'],'delta':spec['delta']}
    rows,stats={},{}
    for role,source in (('gate_calibration',c),('confirmation',s)):
        frame=inputs(f['splits'][role],source['splits'][role])
        chosen=policy(frame,primary,lock['response'])
        accepted=flags(chosen,cutoff)
        summary,extra=counters(chosen,accepted)
        compare(summary,receipt['calibration_descriptive' if role=='gate_calibration' else 'confirmation'])
        rows[role]=(frame,chosen,accepted)
        stats[role]={**summary,**extra}
    test=audit_certificate(stats['gate_calibration'],settings,receipt['exact_certificate'])
    layers={}
    for regime,ref in receipt['all_prevalences']['strata'].items():
        summary=subset(*rows['gate_calibration'],regime)
        layers[regime]=audit_certificate(summary,settings,ref)
    prevalence_p=max(x['joint_p'] for x in layers.values())
    compare(prevalence_p<=settings['delta'],receipt['all_prevalences']['certified'])
    return {'primary':primary,'cutoff':cutoff,'test':test,'confirmation':stats['confirmation'],
            'all_prevalences':{'certified':receipt['all_prevalences']['certified'],'joint_p':prevalence_p,'strata':layers},
            'input_sha256':digest(run/'LOOSE_CONFIRMED.json')}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run',type=Path)
    parser.add_argument('witness_receipt',type=Path)
    opts=parser.parse_args()
    run=opts.run
    start=stamp()
    plan_file=run/'REGISTERED.json'
    strict,hashes=run_audit(run,{'upstream_protocol_sha256':digest(plan_file)})
    plan=read(plan_file)
    streams=audit_streams(run,opts.witness_receipt,plan,hashes)
    loose=audit_loose(run,hashes,plan)
    source=Path(__file__)
    write(run/'REPAIRED_INDEPENDENT_AUDIT.json',{'started_utc':start,'finished_utc':stamp(),
        'timing':'Post-score audit. Scientific criteria and multiplicity split were registered before predictions and outcomes.',
        'inputs':hashes,'sources':{f.name:digest(f) for f in (source,source.with_name('audit_strict_window_gate.py'),
                                                            source.with_name('audit_window_event_policy.py'))},
        'streams':streams,'strict':strict,'reused_loose':loose,
        'scope':'Standard-library arithmetic on fixed actions and exact binomial tails; no model refit or adaptive threshold.'})
    print('Passed exact strict and loose rule audit and disjoint witness preflight.')


if __name__=='__main__':
    main()
