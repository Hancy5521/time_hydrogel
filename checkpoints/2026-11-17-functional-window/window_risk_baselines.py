"""Outcome-supervised joint distribution forest on the registered shared inputs."""

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor

from window_risk_forecasts import choose, summary


def features(row, model):
    observed=np.asarray(row['features'])
    if model=='forest_direct':
        return observed
    if model!='forest_physics':
        raise ValueError(f'Unexpected forest feature setting: {model}')
    equivalent=np.asarray(row['forecasts']['continuum']['margin_q'])[1]
    return np.column_stack((observed,equivalent.reshape(len(observed),-1)))


class OutcomeForest:
    """A leaf mixture over complete training cases, never independent marginals."""

    def __init__(self,train,model,leaf,seed):
        self.model=model
        train_x=np.concatenate([features(row['input'],model) for row in train])
        self.margin=np.concatenate([np.asarray(row['output']['margins']) for row in train])
        self.dry=np.concatenate([np.asarray(row['output']['dry']) for row in train])
        self.track=np.concatenate([np.asarray(row['output']['process']) for row in train])
        targets=self.margin.reshape(len(train_x),-1)
        self.forest=ExtraTreesRegressor(n_estimators=96,min_samples_leaf=leaf,random_state=seed,
                                       max_features=1.,n_jobs=1)
        self.forest.fit(train_x,(targets-targets.mean(axis=0))/np.maximum(targets.std(axis=0),.025))
        self.training_leaf=self.forest.apply(train_x)

    def predict(self,row,plan):
        paths=self.forest.apply(features(row,self.model))
        same=paths[:,None,:]==self.training_leaf[None,:,:]
        weights=(same/same.sum(axis=1,keepdims=True)).mean(axis=-1)
        weights/=weights.sum(axis=1,keepdims=True)
        selected,probability=choose(self.margin,weights,plan['risk_weight'])
        mm,mq=summary(self.margin,weights)
        dm,dq=summary(self.dry,weights)
        pm,pq=summary(self.track,weights)
        return {'selected':selected.tolist(),'probability':probability.tolist(),'margin_mean':mm.tolist(),
                'margin_q':mq.tolist(),'dry_mean':dm.tolist(),'dry_q':dq.tolist(),'process_mean':pm.tolist(),
                'process_q':pq.tolist(),'profile_support':[self.dry.min(axis=0).tolist(),self.dry.max(axis=0).tolist()],
                'ess':(1/(weights**2).sum(axis=1)).tolist(),'forward_candidates':len(self.margin)}


def development_lock(train,dev,plan):
    result={}
    for model in ('forest_direct','forest_physics'):
        rows=[]
        for leaf in plan['forest']['min_leaf']:
            fitter=OutcomeForest(train,model,leaf,plan['forest']['seed'])
            fail=[]
            regret=[]
            brier=[]
            for row in dev:
                pred=fitter.predict(row['input'],plan)
                chosen=np.asarray(pred['selected'])[...,None]
                risk=np.asarray(pred['probability'])
                truth=np.asarray(row['output']['margins'])
                realized=np.take_along_axis(truth,chosen,axis=-1)[...,0]
                fail.extend((realized>0).ravel())
                regret.extend((np.maximum(realized,0)-np.maximum(truth.min(axis=-1),0)).ravel())
                brier.extend((risk-(truth>0)) .ravel()**2)
            rows.append({'leaf':leaf,'failure':float(np.mean(fail)),'regret':float(np.mean(regret)),
                         'all_catalogue_brier':float(np.mean(brier))})
        best=min(rows,key=lambda x:(x['failure'],x['regret'],x['all_catalogue_brier'],x['leaf']))
        result[model]={'leaf':best['leaf'],'development':rows}
    return result
