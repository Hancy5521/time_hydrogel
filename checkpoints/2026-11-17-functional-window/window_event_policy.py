"""Frozen slate policies for whole-window outcomes under a common observation budget."""

import numpy as np
from scipy.special import logit
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from window_group_calibration import take_actions


METHOD='library_depth'


def predictors(rows):
    per_mother=[r['forecasts'][METHOD] for r in rows]
    p=np.array([x['probability'] for x in per_mother])
    q=np.array([x['margin_q'] for x in per_mother])
    mean=np.array([x['margin_mean'] for x in per_mother])
    features=np.stack((logit(np.clip(p,.001,.999)),q[:,1],q[:,2],q[:,3]-q[:,0]),axis=-1)
    return p,q,mean,features


def fit_response(rows, margins, c):
    """Estimate violation risk only on the separate supervised mother groups."""
    _,_,_,x=predictors(rows)
    model=make_pipeline(StandardScaler(),LogisticRegression(C=c,max_iter=1500))
    model.fit(x.reshape(-1,x.shape[-1]),(margins>0).ravel())
    scale,reg=model.steps[0][1],model.steps[1][1]
    return {'location':scale.mean_.tolist(),'scale':scale.scale_.tolist(),
            'intercept':reg.intercept_.tolist(),'coefficient':reg.coef_[0].tolist()}


def slate(rows,response):
    """Uncalibrated risk rankings and candidate selections, with no mechanism flag."""
    p,q,mean,x=predictors(rows)
    base=np.array([r['forecasts'][METHOD]['selected'] for r in rows],dtype=int)
    tail=np.lexsort((mean,p,q[:,2]),axis=-1)[...,0]
    z=((x-response['location'])/response['scale'])@response['coefficient']+response['intercept'][0]
    learned=np.lexsort((mean,q[:,2],z),axis=-1)[...,0]
    return {key:{'chosen':index,'score':take_actions(risk,index)} for key,index,risk in (
        ('posterior_probability',base,p),('upper_margin',tail,q[:,2]),('outcome_score',learned,z))}


def outcomes(margins,action,accepted):
    """Independent mother units for activity and any unsafe issued recommendation."""
    choice=take_actions(margins,action)
    fail=choice>0.
    covers=margins.min(axis=-1)<=0.
    active=accepted.reshape(len(choice),-1).any(axis=-1)
    unsafe=(accepted&fail).reshape(len(choice),-1).any(axis=-1)
    return {'mothers':len(choice),'requests':choice.size,'active_mothers':int(active.sum()),
            'unsafe_mothers':int(unsafe.sum()),'accepted_requests':int(accepted.sum()),
            'unsafe_accepted_requests':int((accepted&fail).sum()),
            'successful_requests':int((accepted&~fail).sum()),'requests_with_feasible_action':int(covers.sum()),
            'raw_successes':int((~fail).sum()),
            'active_fraction':float(active.mean()),'unsafe_fraction':float(unsafe.mean()),
            'unsafe_given_active':float(unsafe.sum()/active.sum()) if active.any() else None}


def development_choice(margins, slates, criterion):
    """Lock a score and cutoff on development labels before formal calibration."""
    plan=[]
    candidates={}
    for key,policy in slates.items():
        s=policy['score']
        thresholds=np.unique(s)
        thresholds=np.nextafter(thresholds,np.inf)
        rows=[]
        for t in thresholds:
            y=outcomes(margins,policy['chosen'],s<t)
            if y['active_mothers']>=criterion['min_development_mothers']:
                rows.append({**y,'cutoff':float(t)})
        eligible=[r for r in rows if r['unsafe_given_active']<=criterion['design_risk']]
        best=max(eligible,key=lambda r:(r['successful_requests'],-r['unsafe_mothers'],-r['cutoff']),default=None)
        candidates[key]={'thresholds_with_required_groups':len(rows),'thresholds_below_design_risk':len(eligible),
                         'selected':best}
        if best is not None:
            plan.append((key,best))
    champion=max(plan,key=lambda kv:(kv[1]['successful_requests'],-kv[1]['unsafe_mothers'],kv[0]),default=None)
    return {'variants':candidates,'primary_method':champion[0] if champion is not None else None,
            'primary_cutoff':champion[1]['cutoff'] if champion is not None else None}
