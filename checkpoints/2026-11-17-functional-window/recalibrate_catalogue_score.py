"""Test a catalogue-specific risk score on a prepartitioned replication cohort."""

import argparse
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.special import logit
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from gel_reactive_port_timing import read_json, sha256
from gel_reactive_two_read_assay import write_new
from window_event_calibration import certify_policy
from window_event_policy import development_choice, outcomes
from window_group_calibration import take_actions


def engineering_features(plan,name,bank):
    p=np.array([r['banks'][name]['probability'] for r in bank])
    mean=np.array([r['banks'][name].get('mean',r['banks'][name].get('margin_mean')) for r in bank])
    upper=np.array([r['banks'][name]['upper'] if 'upper' in r['banks'][name] else r['banks'][name]['margin_q'][2] for r in bank])
    spec=plan['thickness']
    geometry=np.asarray([(v['scale'],a['wet_transfer'],a['near_loading']) for v in spec['actions_at_scale'] for a in v['actions']])
    cols=np.broadcast_to(geometry[spec['catalogues'][name]],(*p.shape,3))
    x=np.concatenate((np.stack((logit(np.clip(p,.001,.999)),mean,upper,upper-mean),axis=-1),cols),axis=-1)
    return x,upper,mean


def array(plan,labels,name):
    return np.asarray([x['margins']['fixed_area_conductance'] for x in labels])[...,plan['thickness']['catalogues'][name]]


def rank(plan,name,bank,coeff):
    x,upper,mean=engineering_features(plan,name,bank)
    score=((x-coeff['location'])/coeff['scale'])@coeff['weights']+coeff['intercept']
    chosen=np.lexsort((mean,upper,score),axis=-1)[...,0]
    return {'chosen':chosen,'score':take_actions(score,chosen)}


def fit(plan,previous):
    design,train,forecasts=[read_json(previous/x) for x in ('POLICY_LOCK.json','CERTIFICATE.json','FROZEN.json')]
    if train['policy_sha256']!=sha256(previous/'POLICY_LOCK.json') or train['frozen_sha256']!=sha256(previous/'FROZEN.json'):
        raise ValueError('The fully observed development data changed.')
    training=forecasts['splits']['gate_calibration']
    dev=forecasts['splits']['policy_development']
    results={}
    for name in plan['compared_catalogues']:
        y=array(plan,train['splits']['gate_calibration'],name)
        x,_,_=engineering_features(plan,name,training)
        model=make_pipeline(StandardScaler(),LogisticRegression(C=plan['logistic_c'],max_iter=1500))
        model.fit(x.reshape(-1,x.shape[-1]),(y>0).ravel())
        standard,logic=model.steps[0][1],model.steps[1][1]
        coeff={'location':standard.mean_.tolist(),'scale':standard.scale_.tolist(),'weights':logic.coef_[0].tolist(),'intercept':float(logic.intercept_[0])}
        truth=array(plan,design['splits']['policy_development'],name)
        picked=development_choice(truth,{'trained_score':rank(plan,name,dev,coeff)},plan['development_criterion'])
        results[name]={'coefficient':coeff,'primary_cutoff':picked['primary_cutoff'],'development':picked['variants']}
    return results


def compare(plan,fit,replicates,labels):
    stop=plan['calibration_mothers']
    if [x['parent'] for x in replicates]!=[x['parent'] for x in labels]:
        raise ValueError('Locked forecasts and outcomes are not aligned.')
    result={}
    for name in plan['compared_catalogues']:
        y=array(plan,labels,name)
        policy=rank(plan,name,replicates,fit[name]['coefficient'])
        cut=fit[name]['primary_cutoff']
        accepted=policy['score']<cut if cut is not None else np.zeros_like(policy['score'],dtype=bool)
        raw=take_actions(y,policy['chosen'])
        old_action=np.asarray([x['banks'][name]['choice'] for x in replicates],dtype=int)
        old_accept=np.asarray([x['banks'][name]['accept'] for x in replicates],dtype=bool)
        old_risk=take_actions(y,old_action)>0.
        selected_effect=((accepted[stop:]&(raw[stop:]<=0.)).astype(float)-(old_accept[stop:]&~old_risk[stop:])).mean(axis=(1,2))
        draws=np.random.default_rng(plan['bootstrap_seed']).integers(len(selected_effect),size=(5000,len(selected_effect)))
        result[name]={'certificate':certify_policy(raw[:stop],accepted[:stop],max_risk=.10,min_activity=.10,delta=.025),
                      'learned':outcomes(y[stop:],policy['chosen'][stop:],accepted[stop:]),
                      'upper_margin_control':outcomes(y[stop:],old_action[stop:],old_accept[stop:]),
                      'correct_recommendation_change':float(selected_effect.mean()),
                      'paired_ci':np.quantile(selected_effect[draws].mean(axis=1),[.025,.975]).tolist(),
                      'frozen_accepted':accepted.tolist(),'frozen_actions':policy['chosen'].tolist()}
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase',choices=('fit','evaluate'))
    p.add_argument('run',type=Path)
    a=p.parse_args()
    protocol=a.run/'REGISTERED.json'
    plan=read_json(protocol)
    for f,digest in plan['hashes'].items():
        if sha256(Path(f))!=digest:
            raise ValueError(f'Changed source {f}.')
    receipt={'registered_sha256':sha256(protocol),'utc':datetime.now(timezone.utc).isoformat()}
    if a.phase=='fit':
        receipt['models']=fit(plan,Path(plan['source_study']))
        write_new(a.run/'FITTED.json',receipt)
    else:
        f=a.run/'FITTED.json'
        fit_info=read_json(f)
        if fit_info['registered_sha256']!=receipt['registered_sha256']:
            raise ValueError('Model lock belongs to different registration.')
        upstream=Path(plan['target_study'])
        f1,f2=[upstream/name for name in ('FROZEN.json','SCORED.json')]
        forecasts,y=read_json(f1),read_json(f2)
        if y['frozen_sha256']!=sha256(f1) or y['protocol_sha256']!=plan['target_protocol_sha256']:
            raise ValueError('Different frozen replication under evaluation.')
        receipt.update(fitted_sha256=sha256(f),frozen_sha256=sha256(f1),target_scored_sha256=sha256(f2),
                       results=compare(plan,fit_info['models'],forecasts['parents'],y['labels']))
        write_new(a.run/'EVALUATED.json',receipt)


if __name__=='__main__':
    main()
