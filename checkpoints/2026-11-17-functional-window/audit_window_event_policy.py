"""Audit event gates, subgroup tests, and equal-input margin controls independently."""

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path


ROLES = ('gate_calibration','confirmation','prevalence_shift')
MODELS = ('posterior_probability','upper_margin','outcome_score')
REGIMES = ('irreversible','exchange')


def stamp():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_bytes())


def write(path,data):
    with path.open('x',encoding='utf-8') as out:
        json.dump(data,out,indent=2,allow_nan=False)
        out.write('\n')


def compare(a,b):
    if isinstance(a,dict) and isinstance(b,dict):
        if a.keys()!=b.keys():
            raise AssertionError('Different audit keys.')
        for key in a:
            compare(a[key],b[key])
    elif isinstance(a,(list,tuple)) and isinstance(b,(list,tuple)):
        for value,target in zip(a,b,strict=True):
            compare(value,target)
    elif isinstance(a,float) and isinstance(b,(int,float)) and math.isclose(a,b,abs_tol=1e-10,rel_tol=1e-9):
        return
    elif a!=b:
        raise AssertionError(f'Independent evaluation mismatch: {a} != {b}')


def inputs(predictions,decoded):
    frame=[]
    for p,y in zip(predictions,decoded,strict=True):
        if p['parent']!=y['parent'] or y['engine'] not in REGIMES:
            raise ValueError('Outcomes do not align with the frozen evaluation identities.')
        f=p['forecasts']['library_depth']
        per_case=[]
        for truth,choice,prob,quantiles,mean in zip(y['margins'],f['selected'],f['probability'],
                                                  zip(*f['margin_q'],strict=True),f['margin_mean'],strict=True):
            per_case.append([{'truth':yy,'chosen':a,'prob':pp,'q':qq,'mean':mu}
                             for yy,a,pp,qq,mu in zip(truth,choice,prob,zip(*quantiles,strict=True),mean,strict=True)])
        frame.append({'engine':y['engine'],'parent':p['parent'],'cases':per_case})
    return frame


def ranking(row,rule,coef):
    p,mean=row['prob'],row['mean']
    median,upper=row['q'][1:3]
    if rule=='posterior_probability':
        index=row['chosen']
        return index,p[index]
    if rule=='upper_margin':
        index=min(range(len(p)),key=lambda j:(upper[j],p[j],mean[j],j))
        return index,upper[index]
    score=[]
    for j,prob in enumerate(p):
        pclip=min(.999,max(.001,prob))
        features=(math.log(pclip/(1-pclip)),median[j],upper[j],row['q'][3][j]-row['q'][0][j])
        score.append(coef['intercept'][0]+math.fsum(
            w*(x-center)/scale for w,x,center,scale in zip(coef['coefficient'],features,
                                                          coef['location'],coef['scale'],strict=True)))
    index=min(range(len(p)),key=lambda j:(score[j],upper[j],mean[j],j))
    return index,score[index]


def policy(frame,rule,coef):
    return [[[(row,*ranking(row,rule,coef)) for row in case] for case in mother['cases']] for mother in frame]


def flags(chosen,cutoff,decision='strict_lt'):
    def accept(score):
        return decision=='all' or decision=='strict_lt' and cutoff is not None and score<cutoff
    return [[[accept(score) for _,_,score in case] for case in group] for group in chosen]


def counters(chosen,mask):
    names=('requests','active','unsafe','accepted','unsafe_requests','feasible','raw_success')
    counts=dict.fromkeys(names,0)
    regret=0.
    for group,selection in zip(chosen,mask,strict=True):
        active=False
        bad=False
        for case,actions in zip(group,selection,strict=True):
            for (row,action,_),accepted in zip(case,actions,strict=True):
                y=row['truth'][action]
                oracle=min(row['truth'])
                active |= accepted
                bad |= accepted and y>0
                counts['requests']+=1
                counts['accepted']+=accepted
                counts['unsafe_requests']+=accepted and y>0
                counts['feasible']+=oracle<=0
                counts['raw_success']+=y<=0
                regret+=max(0.,y)-max(0.,oracle)
        counts['active']+=active
        counts['unsafe']+=bad
    n=len(chosen)
    a,b=counts['active'],counts['unsafe']
    ok=counts['accepted']-counts['unsafe_requests']
    outputs={'mothers':n,'requests':counts['requests'],'active_mothers':a,'unsafe_mothers':b,
             'accepted_requests':counts['accepted'],'unsafe_accepted_requests':counts['unsafe_requests'],
             'successful_requests':ok,'requests_with_feasible_action':counts['feasible'],'raw_successes':counts['raw_success'],
             'active_fraction':a/n if n else None,'unsafe_fraction':b/n if n else None,
             'unsafe_given_active':b/a if a else None}
    return outputs,{'positive_regret_all_frozen_actions':regret/counts['requests'] if n else None,
                    'successful_issued_per_request':ok/counts['requests'] if n else None}


def cumulative(lo,hi,n,p):
    return math.fsum(math.comb(n,i)*p**i*(1-p)**(n-i) for i in range(lo,hi+1))


def exact(count,settings):
    m,k,n=(count[key] for key in ('active_mothers','unsafe_mothers','mothers'))
    r,a=(settings[k] for k in ('risk_target','activity_target'))
    pr,pa=cumulative(0,k,m,r),cumulative(m,n,n,a)
    p=max(pr,pa)
    return {'risk_p':pr,'activity_p':pa,'joint_p':p,'certified':p<=settings['delta']}


def marginal(chosen,alpha):
    extremes=[min((score for case in group for row,a,score in case if row['truth'][a]>0),default=math.inf)
              for group in chosen]
    n=len(chosen)
    rank=int((n+1)*Fraction(str(alpha)))
    threshold=sorted(extremes)[rank-1] if rank else None
    decision='never' if threshold is None else 'all' if math.isinf(threshold) else 'strict_lt'
    return {'alpha':float(alpha),'delta':None,'guarantee':'exchangeable_marginal','rank':rank,'groups':n,
            'finite_critical_scores':sum(math.isfinite(v) for v in extremes),'decision':decision,
            'threshold':threshold if decision=='strict_lt' else None}


def residual_radius(frame,catalogue,alpha):
    n=len(frame)
    k=math.ceil((n+1)*(1-Fraction(str(alpha))))
    if k>n:
        raise ValueError('The audit plan expects a finite conformal rank.')
    maxima=[]
    for group in frame:
        maxima.append(max(y-q for case in group['cases'] for row in case
                          for a,(y,q) in enumerate(zip(row['truth'],row['q'][2],strict=True))
                          if catalogue or a==row['chosen']))
    return {'rank':k,'radius':max(0.,sorted(maxima)[k-1])}


def envelope_policy(frame,fit,catalogue):
    radius=fit['radius']
    def selected(row):
        if not catalogue:
            a=row['chosen']
            return row,a,row['q'][2][a]+radius
        eligible=[a for a,q in enumerate(row['q'][2]) if q+radius<=0]
        a=min(eligible,key=lambda i:(row['prob'][i],row['q'][1][i],i)) if eligible else row['chosen']
        return row,a,row['q'][2][a]+radius
    chosen=[[[selected(row) for row in case] for case in group['cases']] for group in frame]
    accepted=[[[score<=0 for _,_,score in case] for case in group] for group in chosen]
    covered=sum(all(y<=q+radius for case in g['cases'] for row in case
                    for a,(y,q) in enumerate(zip(row['truth'],row['q'][2],strict=True)) if catalogue or a==row['chosen'])
                for g in frame)
    return chosen,accepted,covered/len(frame)


def subset(frame,chosen,mask,engine):
    triples=[(g,c,a) for g,c,a in zip(frame,chosen,mask,strict=True) if g['engine']==engine]
    return counters([c for _,c,_ in triples],[a for _,_,a in triples])[0]


def audit(run,registration):
    paths=('REGISTERED.json','FROZEN.json','POLICY_LOCK.json','CERTIFICATE.json','SCORED.json')
    h={name:digest(run/name) for name in paths}
    if registration['upstream_protocol_sha256']!=h['REGISTERED.json']:
        raise ValueError('Upstream registration was modified.')
    plan,forecast,policy_lock,cert,scored=(read(run/name) for name in paths)
    for stage in (forecast,policy_lock,cert,scored):
        compare(stage['protocol_sha256'],h['REGISTERED.json'])
    for stage in (policy_lock,cert,scored):
        compare(stage['frozen_sha256'],h['FROZEN.json'])
    for stage in (cert,scored):
        compare(stage['policy_sha256'],h['POLICY_LOCK.json'])
    compare(scored['certificate_sha256'],h['CERTIFICATE.json'])
    for previous,later in zip((plan,forecast,policy_lock,cert),(forecast,policy_lock,cert,scored),strict=True):
        finished=previous.get('completed_utc',previous.get('registered_utc'))
        if datetime.fromisoformat(finished)>datetime.fromisoformat(later['started_utc']):
            raise ValueError('Research receipts were not executed in lock-before-use order.')
    outputs={**policy_lock['splits'],**cert['splits'],**scored['splits']}
    mothers=[]
    for role,rows in forecast['splits'].items():
        parents=[row['parent'] for row in rows]
        compare(parents,[r['parent'] for r in outputs[role]])
        compare(len(parents),plan['event_roles'][role]['groups'])
        mothers.extend(parents)
    if len(mothers)!=len(set(mothers)):
        raise ValueError('Mother labels overlap across roles.')
    frames={role:inputs(forecast['splits'][role],outputs[role]) for role in ROLES}
    settings=plan['event_settings']
    dev=policy_lock['development']
    cal=frames['gate_calibration']
    coef=policy_lock['response']
    margins={name:residual_radius(cal,name=='catalogue',settings['marginal_target'])
             for name in ('selected','catalogue')}
    champion=dev['primary_method']
    outcomes={'same_input_margins':margins,'models':{},'champion':champion}
    for model in MODELS:
        selected=dev['variants'][model]['selected']
        cutoff=selected['cutoff'] if selected else None
        c=cert['results'][model]
        compare(cutoff,c['locked_cutoff'])
        chosen=policy(cal,model,coef)
        accept=flags(chosen,cutoff)
        empirical,_=counters(chosen,accept)
        compare(empirical,c['locked_calibration_outcomes'])
        test=exact(empirical,settings)
        compare(test,{k:c['fixed_policy'][k] for k in test})
        rankfit=marginal(chosen,settings['marginal_target'])
        compare(rankfit,c['marginal_event'])
        model_results={'calibration':{**empirical,**test},'marginal':rankfit,'held':{}}
        if model==champion:
            strata={engine:subset(cal,chosen,accept,engine) for engine in REGIMES}
            tests={engine:{**value,**exact(value,settings)} for engine,value in strata.items()}
            all_p=max(row['joint_p'] for row in tests.values())
            outcomes['supplementary_all_mixture_weights']={'joint_p':all_p,'certified':all_p<=settings['delta'],
                                                            'same_locked_gate':True,'calibration_strata':tests,
                                                            'scope':'Fixed nominal/exchange laws only, arbitrary mixture weights if both pass.'}
        for role in ROLES[1:]:
            frame=frames[role]
            pc=policy(frame,model,coef)
            report=scored['reports'][role][model]
            compare([[[i for _,i,_ in case] for case in g] for g in pc],report['choice'])
            compare([[[s for _,_,s in case] for case in g] for g in pc],report['score'])
            policies={'locked':flags(pc,cutoff),'marginal':flags(pc,rankfit['threshold'],rankfit['decision'])}
            result={}
            for name,mask in policies.items():
                compare(mask,report['accept_'+name])
                counts,extra=counters(pc,mask)
                compare(counts,report[name])
                result[name]={**counts,**extra,'strata':{e:subset(frame,pc,mask,e) for e in REGIMES}}
            model_results['held'][role]=result
        outcomes['models'][model]=model_results
    for name,fit in margins.items():
        fit['held']={}
        for role in ROLES[1:]:
            frame=frames[role]
            pc,mask,coverage=envelope_policy(frame,fit,name=='catalogue')
            counts,extra=counters(pc,mask)
            fit['held'][role]={**counts,**extra,'simultaneous_coverage':coverage,
                               'strata':{e:subset(frame,pc,mask,e) for e in REGIMES}}
    return h,outcomes


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('run',type=Path)
    cli.add_argument('--seal-source',action='store_true')
    args=cli.parse_args()
    run=args.run
    registration=run/'EVENT_AUDIT_REGISTERED.json'
    source_lock=run/'EVENT_AUDIT_SOURCE.json'
    current={'source_sha256':digest(Path(__file__)),'audit_plan_sha256':digest(registration)}
    if args.seal_source:
        write(source_lock,{'utc':stamp(),**current,'receipt_presence':{n:(run/n).is_file()
              for n in ('FROZEN.json','POLICY_LOCK.json','CERTIFICATE.json','SCORED.json')}})
        return
    lock=read(source_lock)
    compare(current,{name:lock[name] for name in current})
    start=stamp()
    hashes,results=audit(run,read(registration))
    write(run/'EVENT_AUDIT.json',{'started_utc':start,'finished_utc':stamp(),'hashes':hashes,**current,
             'scope':'Standalone standard-library rank, policy accounting and binomial test audit. No model imports.',
             'results':results})
    print('Independent event-gate and same-input margin audit finished.')


if __name__=='__main__':
    main()
