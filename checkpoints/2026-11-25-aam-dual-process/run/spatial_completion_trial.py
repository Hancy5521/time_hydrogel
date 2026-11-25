"""Test high-dimensional spatial state inference explicitly in the masked AAm field."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy.linalg import solve
from scipy.spatial.distance import cdist

import loop as parent


OUT = parent.ROOT / "spatial_completion"
TRAIN, TEST = 640, 192
BOUNDS = np.array([.35,.7,.85])
OBSERVABLE = np.array([x*16+y*4+z for x in range(2) for y in range(2) for z in range(2)])
HIDDEN = np.setdiff1d(np.arange(64), OBSERVABLE)
COUNTS = np.array([.35,.35,.15,.15])
VOLUME = np.einsum("i,j,k->ijk", COUNTS, COUNTS, COUNTS).reshape(-1)
SHAPE = (4,3,64)


def sample(job):
    import reaction_adapter as native

    code, address, phase = job
    acc = []
    original = native._local_observation
    early = native.EARLY_TIMES

    def read_field(state, origin, box, cutoff):
        present = state.present
        ids = np.flatnonzero(present)
        coordinates = np.remainder((state.points[ids] - origin)/box, 1.)
        field = np.arange(len(ids))
        if phase == "prefix":
            # No aggregate counts are calculated on the hidden portion.
            field = np.flatnonzero((coordinates < .7).all(axis=1))
        ids, coordinates = ids[field], coordinates[field]
        cells = np.searchsorted(BOUNDS, coordinates, side="right")
        index = 16*cells[:,0]+4*cells[:,1]+cells[:,2]
        usage = (state.owner[2*ids]>=0).astype(float)+(state.owner[2*ids+1]>=0)
        arr=np.stack([np.bincount(index,weights=weight,minlength=64)
                      for weight in (ids<64, usage*(ids<64),usage*(ids>=64))])
        acc.append(arr)
        return np.zeros(len(native.LOCAL_NAMES))

    try:
        native._local_observation = read_field
        if phase == "prefix":
            native.simulate(code,"C",int(address),full=False)
            return np.asarray(acc)[:3,:,OBSERVABLE]
        native.EARLY_TIMES = native.TIMES
        native._clock.cache_clear()
        full = native.simulate(code,"C",int(address),full=True)["path"]
        out = np.asarray(acc)
    finally:
        native._local_observation = original
        native.EARLY_TIMES = early
        native._clock.cache_clear()
    np.testing.assert_allclose((out[:,1:,:].sum(axis=(1,2)))/92., full[:,0], atol=1e-12,rtol=0.)
    return out


def collect(g, addr, phase):
    with ProcessPoolExecutor(max_workers=4) as pool:
        jobs=[(x,s,phase) for x,s in zip(g,addr)]
        return np.stack(list(pool.map(sample,jobs,chunksize=10)))


def dump(name, record):
    import json
    with (OUT/name).open("x", encoding="utf-8") as f:
        json.dump(record, f, indent=2,allow_nan=False)


def arrays(name, **data):
    with (OUT/name).open("xb") as f:
        np.savez_compressed(f, **data)


def features(codes, field=None):
    if field is None:
        return np.asarray(codes)/.28
    local = np.asarray(field).reshape(len(field),-1)
    return np.c_[codes/.28, local/np.sqrt(local.shape[1])]


def projection(field, observed):
    field=np.maximum(field,0.).copy()
    # Only all-AAm counts are exactly conserved; reacted components remain nonnegative.
    field[:,0,0,OBSERVABLE]=observed[:,2,0,:]
    want=64-observed[:,2,0,:].sum(axis=1)
    missing=field[:,0,0,HIDDEN]
    missing_sum=missing.sum(axis=1)
    adopted=np.where(missing_sum[:,None]>0,missing/np.maximum(missing_sum[:,None],1e-12),
                         VOLUME[HIDDEN]/VOLUME[HIDDEN].sum())
    field[:,0,0,HIDDEN]=adopted*want[:,None]
    count=field[:,1:,0,:]
    norm=count.sum(axis=-1,keepdims=True)
    field[:,1:,0,:]= 64*np.where(norm>0,count/np.maximum(norm,1e-12),VOLUME)
    field[:,0,1,OBSERVABLE]=observed[:,2,1,:]
    field[:,0,2,OBSERVABLE]=observed[:,2,2,:]
    field[:,:,1,:]=np.minimum(field[:,:,1,:],field[:,:,0,:])
    return field


def predict(training, queries, output, method, hyper):
    sigma, penalty = hyper
    base=output.mean(axis=0)
    centered=(output-base).reshape(len(output),-1)
    if method=="multiscale":
        # Extract shared local-timing response modes; only those modes can alter the base field.
        _, _, basis=np.linalg.svd(centered,full_matrices=False)
        basis=basis[:24]
        centered=centered@basis.T
    kernel = np.exp(-cdist(training,training,"sqeuclidean")/(2*sigma**2))
    kernel.flat[::len(kernel)+1]+=penalty
    cross = np.exp(-cdist(queries,training,"sqeuclidean")/(2*sigma**2))
    update=cross@solve(kernel,centered,assume_a="pos")
    if method=="multiscale":
        update=update@basis
    return base+update.reshape(len(queries),*SHAPE)


def err(prediction, truth):
    return {
        "hidden_at_cutoff":abs(prediction[:,0,:,HIDDEN]-truth[:,0,:,HIDDEN]).mean(axis=(1,2)),
        "future_hidden":abs(prediction[:,1:,:][:,:,:,HIDDEN]-truth[:,1:,:][:,:,:,HIDDEN]).mean(axis=(1,2,3)),
        "future_all":abs(prediction[:,1:]-truth[:,1:]).mean(axis=(1,2,3))
    }


def experiment():
    OUT.mkdir(exist_ok=False)
    import reaction_adapter as sim
    tr_g=parent.stable_rng("space development code").uniform(-.48,.48,(TRAIN,4))
    q_g=parent.stable_rng("space held code").uniform(-.48,.48,(TEST,4))
    tr_addr=parent.seeds("space development addresses",TRAIN)
    q_addr=parent.seeds("space held addresses",TEST)
    assert not np.intersect1d(tr_addr,q_addr).size
    dump("REGISTERED.json",{**parent.stamps(),"script":parent.digest(__file__),
        "source":sim.sources(),"training":TRAIN,"confirmation":TEST,"codes":"observable true synthetic g offered to all methods",
        "paths":"New independent C reactions; held prefix re-extracted after label freeze to verify equality",
        "volumes":COUNTS.tolist(),"observed_cells":OBSERVABLE.tolist(),"label_access":"two phases exact prefix then full after prediction",
        "primary":"hidden per-cell MAE current, followed by the future (3.5,4.85,6)",
        "controls":"spatial volume-based, code KRR, same-info direct KRR",
        "candidate":"same joint KRR kernel rank-24 response correction",
        "fit_hyper_grid":[[2.0,.03],[5.0,.03],[5.0,.3]],
        "wet_experiments":0})
    train=collect(tr_g,tr_addr,"complete")
    q_x=collect(q_g,q_addr,"prefix")
    tr_x=train[:,:3,:,OBSERVABLE]
    targets=train[:,2:]
    tr_b=features(tr_g,tr_x)
    q_b=features(q_g,q_x)
    tr_c=features(tr_g)
    q_c=features(q_g)
    grid=((2.,.03),(5.,.03),(5.,.30))
    index=np.arange(TRAIN)
    inx,valid=index[:480],index[480:]
    def fit(method, tr, q):
        losses=[]
        for hyper in grid:
            v=predict(tr[inx],tr[valid],targets[inx],method,hyper)
            s=err(projection(v,tr_x[valid]),targets[valid])
            losses.append(float(s["hidden_at_cutoff"].mean()+s["future_hidden"].mean()))
        choice=int(np.argmin(losses))
        return projection(predict(tr,q,targets,method,grid[choice]),q_x), loss_pair(grid[choice],losses)
    def loss_pair(a,b):
        return {"params":a,"validation_loss":b}
    means=targets.sum(axis=-1).mean(axis=0)[None,:,:,None]*VOLUME[None,None,None,:]
    climate=projection(np.repeat(means,TEST,axis=0),q_x)
    candidates={"volume":climate}
    cv={}
    for m,x,y in (("code_only",tr_c,q_c),("direct",tr_b,q_b),("multiscale",tr_b,q_b)):
        candidates[m],cv[m]=fit(m,x,y)
    arrays("PREDICTED.npz",**candidates,query_prefix=q_x,query_code=q_g,train_seeds=tr_addr,test_seeds=q_addr)
    dump("FROZEN.json",{**parent.stamps(),"sha256":parent.digest(OUT/"PREDICTED.npz"),"cv":cv})
    assert parent.digest(OUT/"PREDICTED.npz") == __import__("json").loads((OUT/"FROZEN.json").read_text())["sha256"]
    revealed=collect(q_g,q_addr,"complete")
    np.testing.assert_array_equal(q_x,revealed[:,:3,:,OBSERVABLE])
    arrays("CONFIRMED.npz",observed= revealed[:,2:])
    bootstrap=parent.stable_rng("space paired bootstrap").integers(0,TEST,(5000,TEST))
    losses={k:err(v,revealed[:,2:]) for k,v in candidates.items()}
    result={}
    for method, fields in losses.items():
        result[method]={}
        for k,v in fields.items():
            df=v-losses["code_only"][k]
            result[method][k]={"mae":float(v.mean()),"delta_vs_code_only":float(df.mean()),
                                "CI95":np.quantile(df[bootstrap].mean(axis=1),[.025,.975]).tolist()}
    dump("SCORED.json",{**parent.stamps(),"cases": TEST,"forecast_sha256":parent.digest(OUT/"FROZEN.json"),
        "truth_sha256":parent.digest(OUT/"CONFIRMED.npz"),"result":result,
        "coherence_checks":"observed cell values preserved exact at cutoff, total AAm count 64 in every frame",
        "unit":"per-cell particle/site count; model-space in simulation"})
    print(result,flush=True)


if __name__=="__main__":
    experiment()
