"""Independently audit stricter window-risk certification and a matched bath stress."""

import argparse
import hashlib
import json
import math
from datetime import datetime
from itertools import groupby
from pathlib import Path

from audit_window_event_policy import (
    MODELS, REGIMES, compare, counters, cumulative, digest, exact, flags,
    inputs, marginal, policy, read, stamp, subset, write,
)


def inverse_binomial(counts,delta):
    n,m,k=(counts[key] for key in ('mothers','active_mothers','unsafe_mothers'))
    lower,upper=0.,1.
    if m:
        lo,hi=0.,1.
        for _ in range(64):
            p=(lo+hi)/2
            if cumulative(m,n,n,p)<delta:
                lo=p
            else:
                hi=p
        lower=(lo+hi)/2
    if k<m:
        lo,hi=0.,1.
        for _ in range(64):
            p=(lo+hi)/2
            if cumulative(0,k,m,p)>delta:
                lo=p
            else:
                hi=p
        upper=(lo+hi)/2
    return {'individual_risk_upper':upper,'individual_activity_lower':lower}


def reconstruct_development(frame,coefficients,settings):
    """Sweep tied scores, incrementing active and unsafe mothers independently."""
    output,champions={},[]
    for model in MODELS:
        chosen=policy(frame,model,coefficients)
        fixed,_=counters(chosen,flags(chosen,None))
        ranked=sorted((score,i,row['truth'][a]>0) for i,mother in enumerate(chosen)
                      for case in mother for row,a,score in case)
        active,unsafe=set(),set()
        issued,fail,usable=0,0,0
        candidate=[]
        for value,requests in groupby(ranked,key=lambda x:x[0]):
            for _,i,bad in requests:
                active.add(i)
                if bad:
                    unsafe.add(i)
                issued+=1
                fail+=bad
            a,b=len(active),len(unsafe)
            if a<settings['min_development_mothers']:
                continue
            usable+=1
            if b/a<=settings['design_risk']:
                n=len(frame)
                candidate.append({**fixed,'active_mothers':a,'unsafe_mothers':b,'accepted_requests':issued,
                                  'unsafe_accepted_requests':fail,'successful_requests':issued-fail,
                                  'active_fraction':a/n,'unsafe_fraction':b/n,'unsafe_given_active':b/a,
                                  'cutoff':math.nextafter(value,math.inf)})
        best=max(candidate,key=lambda r:(r['successful_requests'],-r['unsafe_mothers'],-r['cutoff']),default=None)
        output[model]={'thresholds_with_required_groups':usable,'thresholds_below_design_risk':len(candidate),
                       'selected':best}
        if best is not None:
            champions.append((model,best))
    winner=max(champions,key=lambda z:(z[1]['successful_requests'],-z[1]['unsafe_mothers'],z[0]),default=None)
    return {'variants':output,'primary_method':winner[0] if winner else None,
            'primary_cutoff':winner[1]['cutoff'] if winner else None}


def audit_certificate(summary,settings,reference):
    result={**exact(summary,settings),**inverse_binomial(summary,settings['delta'])}
    for key,value in result.items():
        compare(value,reference[key])
    for key,origin in (('groups','mothers'),('active_mothers','active_mothers'),
                       ('unsafe_active_mothers','unsafe_mothers'),('accepted_requests','accepted_requests'),
                       ('requests','requests')):
        compare(summary[origin],reference[key])
    return {**summary,**result}


def record(frame,chosen,masks):
    stats,utility=counters(chosen,masks)
    return {**stats,**utility,'strata':{e:subset(frame,chosen,masks,e) for e in REGIMES}}


def paired_changes(original,changed,accepted):
    requests,mothers={'became_unsafe':0,'became_safe':0},{'became_unsafe':0,'became_safe':0}
    for group_a,group_b,choices in zip(original,changed,accepted,strict=True):
        family_a,family_b=False,False
        for case_a,case_b,mask in zip(group_a,group_b,choices,strict=True):
            for (a,i,s),(b,j,t),issue in zip(case_a,case_b,mask,strict=True):
                compare((i,s),(j,t))
                a_bad,b_bad=issue and a['truth'][i]>0,issue and b['truth'][j]>0
                requests['became_unsafe']+=b_bad and not a_bad
                requests['became_safe']+=a_bad and not b_bad
                family_a|=a_bad
                family_b|=b_bad
        mothers['became_unsafe']+=family_b and not family_a
        mothers['became_safe']+=family_a and not family_b
    return {'accepted_request_flips':requests,'active_mother_flips':mothers}


def load_run(run,protocol_hash):
    names=('REGISTERED.json','FROZEN.json','CERTIFICATE.json','SCORED.json')
    files={n:digest(run/n) for n in names}
    compare(files['REGISTERED.json'],protocol_hash)
    plan,frozen,cert,score=(read(run/n) for n in names)
    for stage in (frozen,cert,score):
        compare(stage['protocol_sha256'],protocol_hash)
        compare(stage['policy'],frozen['policy'])
    for stage in (cert,score):
        compare(stage['frozen_sha256'],files['FROZEN.json'])
    compare(score['certificate_sha256'],files['CERTIFICATE.json'])
    starts=(frozen['started_utc'],cert['started_utc'],score['started_utc'])
    for earlier,later in zip((plan['registered_utc'],frozen['completed_utc'],cert['completed_utc']),starts,strict=True):
        if datetime.fromisoformat(earlier)>datetime.fromisoformat(later):
            raise AssertionError('Freeze, calibration and score are out of chronological order.')
    root=Path(plan['prior_study'])
    for name in ('FROZEN.json','POLICY_LOCK.json'):
        path=root/name
        key=path.as_posix()
        files[key]=digest(path)
        compare(files[key],plan['hashes'][key])
    original,lock=(read(root/n) for n in ('FROZEN.json','POLICY_LOCK.json'))
    compare(lock['frozen_sha256'],files[(root/'FROZEN.json').as_posix()])
    return plan,frozen,cert,score,original,lock,files


def run_audit(run,registration):
    plan,frozen,cert,score,original,lock,files=load_run(run,registration['upstream_protocol_sha256'])
    settings=plan['event_settings']
    dev=inputs(original['splits']['policy_development'],lock['splits']['policy_development'])
    coefficients=lock['response']
    compare(reconstruct_development(dev,coefficients,settings),frozen['policy'])
    decoded={**cert['splits'],**score['splits']}
    frames={}
    seed_keys=('nominal_seed','shift_seed','mechanism_seed','spatial_noise_salt','hydraulic_salt')
    prior_plan=read(Path(plan['prior_study'])/'REGISTERED.json')
    for key in seed_keys:
        before={settings[key] for settings in prior_plan['event_roles'].values()}
        now=[settings[key] for settings in plan['event_roles'].values()]
        if len(set(now))!=len(now) or before.intersection(now):
            raise ValueError(f'Mother generation stream reused: {key}')
    def visible(row):
        return tuple(tuple(features) for features in row['features'])
    fingerprints={visible(row) for split in original['splits'].values() for row in split}
    seen=set()
    for role,cfg in plan['event_roles'].items():
        frames[role]=inputs(frozen['splits'][role],decoded[role])
        compare(len(frames[role]),cfg['groups'])
        ids=[m['parent'] for m in frames[role]]
        if len(ids)!=len(set(ids)) or seen.intersection(ids):
            raise ValueError('Mother identifiers overlap within the strict run.')
        seen.update(ids)
        for row in frozen['splits'][role]:
            feature=visible(row)
            if feature in fingerprints:
                raise ValueError('An observational feature payload is shared across independent roles or runs.')
            fingerprints.add(feature)
    frames['changed_bath']=inputs(frozen['splits']['hidden_bath_stress'],decoded['changed_bath'])
    for label in ('hidden_bath_stress','changed_bath'):
        for row in frames[label]:
            compare(row['engine'],'exchange')
    closure={role:max(float(r['inventory_closure_max']) for r in rows) for role,rows in decoded.items()}
    if any(not 0.<=v<1e-8 for v in closure.values()):
        raise ArithmeticError('An outcome receipt violates the conservative solver tolerance.')
    primary=frozen['policy']['primary_method']
    cal=frames['gate_calibration']
    prediction_roles=('confirmation','hidden_bath_stress','changed_bath')
    result={'primary':primary,'same_response_coefficients_as_original':True,'variants':{},
            'all_prevalences':None,'stable_prediction_digest_by_policy':{},
            'maximum_inventory_closure_residual':closure}
    for model in MODELS:
        fit=cert['results'][model]
        dev_choice=frozen['policy']['variants'][model]['selected']
        cutoff=dev_choice['cutoff'] if dev_choice else None
        compare(cutoff,fit['locked_cutoff'])
        chosen=policy(cal,model,coefficients)
        accept=flags(chosen,cutoff)
        stats,_=counters(chosen,accept)
        compare(stats,fit['locked_calibration_outcomes'])
        exact_check=audit_certificate(stats,settings,fit['fixed_policy'])
        rankfit=marginal(chosen,settings['marginal_target'])
        compare(rankfit,fit['marginal_event'])
        if model==primary:
            actual=cert['all_prevalences']
            laws={k:audit_certificate(subset(cal,chosen,accept,k),settings,actual['laws'][k]) for k in REGIMES}
            p=max(d['joint_p'] for d in laws.values())
            compare(p,actual['joint_p'])
            compare(p<=settings['delta'],actual['certified'])
            result['all_prevalences']={'laws':laws,'joint_p':p,'certified':actual['certified']}
        held={}
        for role in prediction_roles:
            frame=frames[role]
            choices=policy(frame,model,coefficients)
            report=score['reports'][role][model]
            compare([[[a for _,a,_ in c] for c in g] for g in choices],report['choice'])
            compare([[[s for _,_,s in c] for c in g] for g in choices],report['score'])
            a=flags(choices,cutoff)
            b=flags(choices,rankfit['threshold'],rankfit['decision'])
            compare(a,report['accept_locked'])
            compare(b,report['accept_marginal'])
            for name,mask in (('locked',a),('marginal',b)):
                compare(counters(choices,mask)[0],report[name])
            held[role]={'locked':record(frame,choices,a),'marginal':record(frame,choices,b)}
        keys=('choice','score','accept_locked','accept_marginal')
        nominal=score['reports']['hidden_bath_stress'][model]
        stressed=score['reports']['changed_bath'][model]
        same_forecasts={k:nominal[k] for k in keys}
        if same_forecasts!={k:stressed[k] for k in keys}:
            raise AssertionError('A hidden bath change altered the shared frozen prediction.')
        result['stable_prediction_digest_by_policy'][model]=hashlib.sha256(
            json.dumps(same_forecasts,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        before,after=(policy(frames[role],model,coefficients) for role in prediction_roles[1:])
        flips={name:paired_changes(before,after,nominal['accept_'+name]) for name in ('locked','marginal')}
        result['variants'][model]={'locked_cutoff':cutoff,'calibration':exact_check,'marginal':rankfit,
                                   'observed_test':held,'counterfactual_flips':flips}
    return result,files


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run',type=Path)
    parser.add_argument('--seal-source',action='store_true')
    args=parser.parse_args()
    run=args.run
    re=run/'STRICT_AUDIT_REGISTERED.json'
    source=run/'STRICT_AUDIT_SOURCE.json'
    hashes={entry.name:digest(entry) for entry in (Path(__file__),Path(__file__).with_name('audit_window_event_policy.py'),re)}
    if args.seal_source:
        write(source,{'utc':stamp(),'hashes':hashes,'receipt_presence':{n:(run/n).is_file()
             for n in ('FROZEN.json','CERTIFICATE.json','SCORED.json')}})
        return
    compare(hashes,read(source)['hashes'])
    start=stamp()
    outcome,files=run_audit(run,read(re))
    write(run/'STRICT_AUDIT.json',{'started_utc':start,'finished_utc':stamp(),'study_hashes':files,
          'audit_hashes':hashes,'passed':True,'results':outcome})
    print('Passed independent stricter-risk and identical-input stress audit.')


if __name__=='__main__':
    main()
