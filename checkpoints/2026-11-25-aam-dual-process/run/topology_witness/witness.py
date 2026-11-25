"""Controlled A/B observations move the same measurement budget along the same paths."""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone, timedelta
from hashlib import file_digest, sha256
from itertools import product
import json
from pathlib import Path
import numpy as np
from scipy.linalg import solve
from scipy.spatial.distance import cdist

ROOT = Path(__file__).resolve().parent
TRIAL = ROOT.parent.name + "|witness-v1"
N_TRAIN, N_TEST, REPEATS = 80, 48, 4
SIGNS = (-.70,.70)
SCHEDULES = {"early":(0,1,2),"later":(2,3,4)}


def rng(role):
    address = int.from_bytes(sha256((TRIAL + "|" + role).encode()).digest()[:16],"little")
    return np.random.default_rng(address)


def stamp():
    now = datetime.now(timezone.utc)
    return {"utc":now.isoformat(), "shanghai":now.astimezone(timezone(timedelta(hours=8))).isoformat()}


def sha(path):
    with Path(path).open("rb") as stream:
        return file_digest(stream,"sha256").hexdigest()


def write(name, value):
    with (ROOT/name).open("x", encoding="utf-8") as stream:
        json.dump(value,stream,indent=2,allow_nan=False)


def get(name):
    return json.loads((ROOT/name).read_text(encoding="utf-8-sig"))


def save(name,**arrays):
    with (ROOT/name).open("xb") as stream:
        np.savez_compressed(stream,**arrays)


def register():
    from witness_adapter import sources, EARLY_TIMES, ENVIRONMENTS, FIELD_SIDE
    deps = sources()
    for name in ("witness.py","PROTOCOL_ZH.md"):
        deps[str(ROOT/name)] = sha(ROOT/name)
    write("REGISTERED.json", {**stamp(),"sources":deps,"seed_namespace":TRIAL,
        "development_pairs":N_TRAIN,"held_pairs":N_TEST,"observations_per_environment":REPEATS,
        "binary_bridge_codes":SIGNS,"frames":EARLY_TIMES,"schedules":SCHEDULES,
        "environment":ENVIRONMENTS,"window_fraction_by_side":FIELD_SIDE,
        "private_target":"HELD_LABELS.npz may be read only after freeze verification"})


def check_sources():
    for name,digest in get("REGISTERED.json")["sources"].items():
        if sha(name)!=digest:
            raise ValueError(f"Changed registered source: {name}")


def cards(n, role):
    generator = rng(role)
    nuisance = generator.uniform(-.48,.48,(n,3))
    order = (generator.integers(0,2,n)*2-1)
    opposite = .70 * np.c_[order,-order]
    code = np.concatenate((np.broadcast_to(nuisance[:,None],(n,2,3)),opposite[...,None]),axis=-1)
    seeds = rng(role+" fresh paths").integers(0,2**63-1,(n,2,REPEATS))
    return code, seeds


def worker(job):
    from witness_adapter import simulate
    code,env,seed = job
    return simulate(code,env,int(seed),full=False)["local"]


def collect(workers):
    check_sources()
    groups = {"DEV":cards(N_TRAIN,"development"), "HELD":cards(N_TEST,"blind")}
    all_seeds = np.r_[groups["DEV"][1].ravel(),groups["HELD"][1].ravel()]
    assert len(np.unique(all_seeds)) == len(all_seeds)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for label,(code,seeds) in groups.items():
            jobs = [(g,env,j) for d,ss in zip(code,seeds) for g in d
                    for env,s in zip(("A","B"),ss) for j in s]
            trace = []
            for idx, x in enumerate(pool.map(worker,jobs,chunksize=16),1):
                trace.append(x)
                if idx%256 == 0:
                    print(label,idx,len(jobs),flush=True)
            local = np.asarray(trace).reshape(len(code),2,2,REPEATS,5,-1)
            if not np.isfinite(local).all():
                raise ValueError("Nonfinite observation")
            save(label+".npz",local=local,seeds=seeds,
                 **({"codes":code} if label=="DEV" else {}))
            if label=="HELD":
                save("HELD_LABELS.npz",codes=code)
    names=("DEV.npz","HELD.npz","HELD_LABELS.npz")
    write("ACQUISITION.json",{**stamp(),"files":{name:sha(ROOT/name) for name in names}})


def pooled(local, frames):
    x = local[...,list(frames),:]
    a,b=x.mean(3),x.std(3)
    return np.concatenate((a.reshape(*a.shape[:2],-1),b.reshape(*b.shape[:2],-1)),axis=2)


def predict(train, y, query, mode, params):
    center = train.mean(0)
    scale = np.maximum(train.std(0),1.e-8)
    z,q = (train-center)/scale,(query-center)/scale
    if mode=="ridge":
        regularizer, = params
        kp,kq = z@z.T,q@z.T
    else:
        width,regularizer=params
        a = 2.*width*width*z.shape[1]
        kp=np.exp(-cdist(z,z,"sqeuclidean")/a)
        kq=np.exp(-cdist(q,z,"sqeuclidean")/a)
    kp.flat[::len(z)+1]+=regularizer
    mean=y.mean()
    return np.clip(mean+kq@solve(kp,y-mean,assume_a="pos"),-.70,.70)


def grouped_fit(x,y,held,mode):
    grid = [(r,) for r in (4.,16.,64.,256.)] if mode=="ridge" else list(product((.60,1.2,2.4),(.015,.15,1.5)))
    assignments = np.array_split(rng("inner concentration folds").permutation(len(x)),4)
    xx=x.reshape(-1,x.shape[-1])
    yy=y.flatten()
    ranked=[]
    for hp in grid:
        err=np.empty_like(y)
        for omit in assignments:
            take=np.setdiff1d(np.arange(len(x)),omit)
            for test in (take,omit):
                if len(test)==0:
                    raise RuntimeError("Empty group fold")
            guess = predict(x[take].reshape(-1,x.shape[-1]),y[take].flatten(),
                            x[omit].reshape(-1,x.shape[-1]), mode,hp)
            err[omit]=np.square(guess.reshape(len(omit),2)-y[omit])
        ranked.append(float(err.mean()))
    best=int(np.argmin(ranked))
    val=predict(xx,yy,held.reshape(-1,x.shape[-1]),mode,grid[best]).reshape(len(held),2)
    return val, list(zip(grid,ranked)),grid[best]


def freeze():
    check_sources()
    acquisition=get("ACQUISITION.json")
    for name in ("DEV.npz","HELD.npz"):
        assert sha(ROOT/name)==acquisition["files"][name]
    with np.load(ROOT/"DEV.npz") as d, np.load(ROOT/"HELD.npz") as h:
        train, y, held=d["local"],d["codes"][...,3],h["local"]
    predictions, tuning = {},{}
    for label,frames in SCHEDULES.items():
        for mode in ("ridge","rbf"):
            val,cv,best=grouped_fit(pooled(train,frames),y,pooled(held,frames),mode)
            name=f"{label}_{mode}"
            predictions[name]=val.tolist()
            tuning[name]={"cv":cv,"selected":best}
    write("FROZEN.json",{**stamp(),"acquisition_sha256":sha(ROOT/"ACQUISITION.json"),
                         "registration_sha256":sha(ROOT/"REGISTERED.json"),
                         "predictions":predictions,"tuning":tuning})
    print("witness freeze",sha(ROOT/"FROZEN.json"),flush=True)


def score(expected):
    check_sources()
    if sha(ROOT/"FROZEN.json") != expected:
        raise ValueError("Missing expected immutable freeze SHA")
    frozen=get("FROZEN.json")
    assert frozen["acquisition_sha256"]==sha(ROOT/"ACQUISITION.json")
    assert frozen["registration_sha256"]==sha(ROOT/"REGISTERED.json")
    assert get("ACQUISITION.json")["files"]["HELD_LABELS.npz"] == sha(ROOT/"HELD_LABELS.npz")
    with np.load(ROOT/"HELD_LABELS.npz") as truth:
        y = truth["codes"][...,3]
    sel = rng("paired score bootstrap").integers(0,len(y),(8192,len(y)))
    summary, errors = {},{}
    for key,p in frozen["predictions"].items():
        z=np.array(p)
        loss = np.square(z-y).mean(axis=1)
        delta = (z*np.sign(y)).sum(axis=1)
        summary[key]={"estimate":float(loss.mean()),"brier":float(loss.mean()/1.4**2),
                      "pair_positive_order":float(np.mean((delta>0)+.5*(delta==0)))}
        errors[key]=loss
    compare={}
    for mode in ("ridge","rbf"):
        for ref, base in (("early",errors["early_"+mode]),("nominal",np.square(y).mean(axis=1))):
            gain=base-errors["later_"+mode]
            compare[mode+"_later_vs_"+ref]={"gain":float(gain.mean()),
                                           "percentile95":np.quantile(gain[sel].mean(axis=1),[.025,.975]).tolist(),
                                           "wins":int((gain>0).sum()),"pairs":len(gain)}
    with np.load(ROOT/"HELD.npz") as held:
        local=held["local"]
    signal={}
    for feature,index in (("redundancy",9),("double_mbaa",4)):
        signal[feature]={"zero_fraction_by_frame":(local[...,index]==0).mean(axis=(0,1,2,3)).tolist()}
    write("SCORED.json",{**stamp(),"frozen_sha256":expected,"scores":summary,"paired_gains":compare,
                        "local_graph_signal":signal,"squared_error_by_pair":{k:v.tolist() for k,v in errors.items()}})
    print(summary,compare,signal,flush=True)


if __name__=="__main__":
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("stage",choices=("register","collect","freeze","score"))
    cli.add_argument("--workers",type=int,default=5)
    cli.add_argument("--freeze-sha256")
    opts=cli.parse_args()
    if opts.stage=="register":
        register()
    elif opts.stage=="collect":
        collect(opts.workers)
    elif opts.stage=="freeze":
        freeze()
    else:
        score(opts.freeze_sha256)
