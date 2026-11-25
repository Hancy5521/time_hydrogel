"""Continuous single-coordinate transfer with frozen observation and C predictions."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone, timedelta
from hashlib import file_digest, sha256
from itertools import product
from pathlib import Path
import json
import sys

import numpy as np
from scipy.spatial.distance import cdist

ROOT = Path(__file__).resolve().parent
WITNESS = ROOT.parent/"topology_witness"
sys.path.insert(0,str(WITNESS))
from witness import predict
from witness_adapter import simulate, sources, EARLY_TIMES, TIMES, LOCAL_NAMES

N_TRAIN,N_TEST,REPS,N_C = 100,56,4,24
VIEWS = {"early":(0,1,2),"later":(2,3,4)}
MODELS = ("early_ridge","early_rbf","later_ridge","later_rbf","zero")
NAMESPACE = ROOT.parent.name+"|continuous-bridge-v1"


def st():
    t=datetime.now(timezone.utc)
    return {"utc":t.isoformat(),"shanghai":t.astimezone(timezone(timedelta(hours=8))).isoformat()}


def hashed(path):
    with Path(path).open("rb") as f:
        return file_digest(f,"sha256").hexdigest()


def rng(role):
    raw = sha256((NAMESPACE+"|"+role).encode()).digest()
    return np.random.default_rng(int.from_bytes(raw[:16],"little"))


def get(name):
    return json.loads((ROOT/name).read_text(encoding="utf-8-sig"))


def put(name,result):
    with (ROOT/name).open("x",encoding="utf-8") as f:
        json.dump(result,f,indent=2,allow_nan=False)


def save(name,**payload):
    with (ROOT/name).open("xb") as f:
        np.savez_compressed(f,**payload)


def gen(n,role):
    values=rng(role+" material").uniform(-1,1,(n,4))*[.48,.48,.48,.70]
    seed=rng(role+" A/B addresses").integers(0,2**63-1,(n,2,REPS))
    return values,seed


def register():
    deps=sources()
    for p in (ROOT/"bridge_transfer.py",ROOT/"PROTOCOL_ZH.md",WITNESS/"witness.py"):
        deps[str(p)]=hashed(p)
    put("REGISTERED.json",{**st(),"dependencies":deps,"dev_cards":N_TRAIN,"held_cards":N_TEST,
                           "ab_observations":REPS,"c_ensembles_each":N_C,"c_known":["initiation","growth","pendant"],
                           "ab_local_times":EARLY_TIMES,"ab_local_features":LOCAL_NAMES,"plans":VIEWS,
                           "c_path_times":TIMES,"models":MODELS,"seed_namespace":NAMESPACE})


def guard():
    for file,digest in get("REGISTERED.json")["dependencies"].items():
        assert hashed(file)==digest,file


def work(job):
    g,env,seed = job
    return simulate(g,env,int(seed),full=env=="C")


def many(tasks,workers,mode):
    result=[]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for j,p in enumerate(pool.map(work,tasks,chunksize=24),1):
            result.append(p)
            if j%640==0:
                print(mode,j,len(tasks),flush=True)
    return result


def pool_obs(x,select):
    view=x[...,list(select),:]
    return np.c_[view.mean(2).reshape(len(x),-1),view.std(2).reshape(len(x),-1)]


def collect(workers):
    guard()
    dev,dev_seed=gen(N_TRAIN,"development")
    held,held_seed=gen(N_TEST,"unseen")
    cases=np.r_[dev,held]
    address=np.concatenate((dev_seed,held_seed),axis=0)
    tasks=[(g,e,s) for g,a in zip(cases,address) for e,ss in zip(("A","B"),a) for s in ss]
    result=many(tasks,workers,"collect")
    local=np.array([r["local"] for r in result]).reshape(len(cases),2,REPS,5,len(LOCAL_NAMES))
    save("DEV.npz",X=local[:N_TRAIN],G=dev,seeds=dev_seed)
    save("QUERY.npz",X=local[N_TRAIN:],known=held[:,:3],seeds=held_seed)
    save("ESCROW.npz",bridge=held[:,3])
    put("ACQUISITION.json",{**st(),"files":{name:hashed(ROOT/name) for name in ("DEV.npz","QUERY.npz","ESCROW.npz")}})


def fit(train,y,query,kind):
    folds=np.array_split(rng("development cv partitions").permutation(len(train)),4)
    grid=[(z,) for z in (4.,16.,64.,256.)] if kind=="ridge" else list(product((.60,1.20,2.40),(.015,.15,1.5)))
    errors=[]
    for hp in grid:
        loss=np.empty(len(y))
        for held in folds:
            keep=np.setdiff1d(np.arange(len(y)),held)
            loss[held]=(predict(train[keep],y[keep],train[held],kind,hp)-y[held])**2
        errors.append(float(loss.mean()))
    best=int(np.argmin(errors))
    return predict(train,y,query,kind,grid[best]),{"params":grid[best],"search":list(zip(grid,errors))}


def job_for_c(codes,seeds):
    return [(g,"C",seed) for g,row in zip(codes,seeds) for seed in row]


def path_arrays(r,outer):
    return {key:np.asarray([x[key] for x in r]).reshape(*outer,*np.asarray(r[0][key]).shape)
            for key in ("path","milestones")}


def freeze(workers):
    guard()
    acquire=get("ACQUISITION.json")
    for src in ("DEV.npz","QUERY.npz"):
        assert acquire["files"][src]==hashed(ROOT/src)
    with np.load(ROOT/"DEV.npz") as d,np.load(ROOT/"QUERY.npz") as q:
        known=q["known"]
        prediction,tuning=[],{}
        for view,frame in VIEWS.items():
            for kind in ("ridge","rbf"):
                p,t=fit(pool_obs(d["X"],frame),d["G"][:,3],pool_obs(q["X"],frame),kind)
                prediction.append(p)
                tuning[view+"_"+kind]=t
    codes=np.asarray(prediction+[np.zeros(N_TEST)])
    address=rng("C predictive addresses").integers(0,2**63-1,(N_TEST,N_C))
    tasks=[task for values in codes for task in job_for_c(np.c_[known,values],address)]
    r=many(tasks,workers,"forecast C")
    save("PREDICTED.npz",bridge=codes,seeds=address,**path_arrays(r,(len(MODELS),N_TEST,N_C)))
    put("FROZEN.json",{**st(),"predicted_sha256":hashed(ROOT/"PREDICTED.npz"),
                       "acquisition_sha256":hashed(ROOT/"ACQUISITION.json"),"registered_sha256":hashed(ROOT/"REGISTERED.json"),
                       "models":MODELS,"tuning":tuning})
    print("bridge freeze",hashed(ROOT/"FROZEN.json"),flush=True)


def conf_gate(expected):
    guard()
    assert hashed(ROOT/"FROZEN.json")==expected
    freeze=get("FROZEN.json")
    for name,key in (("PREDICTED.npz","predicted_sha256"),("ACQUISITION.json","acquisition_sha256"),("REGISTERED.json","registered_sha256")):
        assert hashed(ROOT/name)==freeze[key]


def confirm(workers,expected):
    conf_gate(expected)
    assert get("ACQUISITION.json")["files"]["ESCROW.npz"]==hashed(ROOT/"ESCROW.npz")
    with np.load(ROOT/"ESCROW.npz") as e,np.load(ROOT/"QUERY.npz") as q:
        code=np.c_[q["known"],e["bridge"]]
    address=rng("C independent truth addresses").integers(0,2**63-1,(N_TEST,N_C))
    r=many(job_for_c(code,address),workers,"true C")
    save("CONFIRMED.npz",code=code,seeds=address,**path_arrays(r,(N_TEST,N_C)))
    put("CONFIRMED.json",{**st(),"frozen_sha256":expected,"truth_sha256":hashed(ROOT/"CONFIRMED.npz")})


def energy(f,y):
    vector=lambda a:(a[...,3:6,[5,2]]/[.18,.10]).reshape(len(a),6)
    x,z=vector(f),vector(y)
    return (cdist(x,z).mean()-.5*cdist(x,x).mean())/np.sqrt(6.)


def paired(d):
    n=len(d)
    choice=rng("bootstrap by card").integers(0,n,(8192,n))
    return {"gain":float(d.mean()),"percentile95":np.quantile(d[choice].mean(axis=1),(.025,.975)).tolist(),
            "card_wins":int((d>0).sum()),"cards":n}


def read_score(expected):
    conf_gate(expected)
    confirmation=get("CONFIRMED.json")
    assert hashed(ROOT/"CONFIRMED.npz")==confirmation["truth_sha256"] and confirmation["frozen_sha256"]==expected
    with np.load(ROOT/"PREDICTED.npz") as p,np.load(ROOT/"CONFIRMED.npz") as y:
        code_loss=np.square(p["bridge"]-y["code"][:,3])
        forecast, outcome=p["path"],y["path"]
        d=np.array([[energy(a,b) for a,b in zip(x,outcome)] for x in forecast])
        bridge=np.abs(forecast[...,5,5].mean(axis=2)-outcome[...,5,5].mean(axis=1))
        largest=np.abs(forecast[...,5,3].mean(axis=2)-outcome[...,5,3].mean(axis=1))
        def milestone(z):
            r=z[...,0]>0
            return {"reached":r.mean(axis=(0,1)).tolist(),
                    "largest_on_reach":[float(np.mean(z[...,k,3][r[...,k]])) if r[...,k].any() else None for k in range(2)]}
        mil={m:milestone(v) for m,v in zip(MODELS,p["milestones"])}
        mil["truth"]=milestone(y["milestones"])
    metrics={"code_mse":code_loss,"c_joint":d,"c_final_bridge":bridge,"c_final_largest":largest}
    choices={key:{m:float(v.mean()) for m,v in zip(MODELS,value)} for key,value in metrics.items()}
    comp={}
    for key,value in metrics.items():
        rates={}
        for j,learner in enumerate(("ridge","rbf")):
            rates[learner]={"later_vs_early":paired(value[j]-value[j+2]),
                            "later_vs_zero":paired(value[-1]-value[j+2])}
        comp[key]=rates
    put("SCORED.json",{**st(),"frozen_sha256":expected,"truth_sha256":confirmation["truth_sha256"],
                       "metrics":choices,"paired_gain":comp,"milestones":mil,
                       "per_card":{k:v.tolist() for k,v in metrics.items()}})
    print(choices,comp,flush=True)


if __name__=="__main__":
    cli=argparse.ArgumentParser()
    cli.add_argument("stage",choices=("register","collect","freeze","confirm","score"))
    cli.add_argument("--workers",type=int,default=5)
    cli.add_argument("--freeze-sha256")
    options=cli.parse_args()
    if options.stage=="register":
        register()
    elif options.stage=="collect":
        collect(options.workers)
    elif options.stage=="freeze":
        freeze(options.workers)
    elif options.stage=="confirm":
        confirm(options.workers,options.freeze_sha256)
    else:
        read_score(options.freeze_sha256)
