"""Paired grid refinement for a frozen process design, with shared Brownian increments."""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np

import loop as parent


OUT = parent.ROOT / "resolution"
N = 128
STEPS = (32, 16, 4)
CHANNELS = (0, 2, 3, 5, 9)
MARGINAL_TOL = np.array([.025,.025,.05,.05,.075])


def job(task):
    import reaction_adapter as native
    g, seed, deltas = task
    end = round(6/native.TICK)
    sim = np.random.default_rng(np.random.SeedSequence(int(seed)).spawn(5)[1])
    motion = np.empty((end+1,native.N_TOTAL,3))
    motion[0] = 0.
    motion[1:] = sim.standard_normal(motion[1:].shape)*np.sqrt(native.TICK)
    np.cumsum(motion,axis=0,out=motion)
    original = native._brownian_at_nodes
    step_old = native.STEP
    values = []
    try:
        native._brownian_at_nodes = lambda *_:motion
        for step in STEPS:
            native.STEP = step
            native._clock.cache_clear()
            values.append([native.simulate(g+age,"C",seed,full=True)["path"] for age in deltas])
    finally:
        native.STEP = step_old
        native._brownian_at_nodes = original
        native._clock.cache_clear()
    return np.asarray(values)


def emit(name, data):
    import json
    with (OUT/name).open("x",encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False,indent=2,allow_nan=False)


def experiment():
    src=parent.ROOT/"POOLED_NOMINAL.json"
    frozen = parent.load("FROZEN.json")
    choice = parent.load("POOLED_NOMINAL.json")
    assert choice["freeze"]==parent.digest(parent.ROOT/"FROZEN.json")
    name=("no_edit","pooled_early","pooled_delayed")
    menu,_=parent.actions()
    deltas=menu[[0,*choice["actions"]]]
    g=parent.stable_rng("grid refinement new cards").uniform(-.48,.48,(N,4))
    ids=parent.seeds("grid refinement new addresses",N)
    bootstrap=parent.stable_rng("grid refinement bootstrap").integers(0,N,(6000,N))
    OUT.mkdir(exist_ok=False)
    emit("REGISTERED.json",{**parent.stamps(),"source":parent.digest(__file__),
         "dependencies":parent.model_versions(),"resolution_tick_counts":STEPS,
         "marginal_tolerance_for_terminal_x_lw_cluster_bridge_wrap":MARGINAL_TOL.tolist(),
         "freeze":choice["freeze"],"selected_arm_source":parent.digest(src),
         "code_and_seeds":"separate role labels from main experiment",
         "phenotype_goal":"original two normative process targets unchanged",
         "input_codes":g.tolist(),"input_seeds":ids.tolist(),"model_arms":name,
         "thresholds":frozen["targets"]})
    with ProcessPoolExecutor(max_workers=4) as pool:
        paths=np.asarray(list(pool.map(job,[(gk,int(i),deltas) for gk,i in zip(g,ids,strict=True)],chunksize=3)))
    with (OUT/"PATHS.npz").open("xb") as f:
        np.savez_compressed(f, path=paths, codes=g, seeds=ids, edits=deltas)
    assert paths.shape==(N,len(STEPS),len(deltas),6,10)
    def stat(x):
        return {"mean":np.asarray(x.mean(axis=0)).tolist(),
                "interval_95":np.quantile(x[bootstrap].mean(axis=1),[.025,.975],axis=0).tolist()}
    repeat=[]
    for step in (0,1):
        record={"relative_step":STEPS[step],"reference":STEPS[2],"arms":{}}
        for i, arm in enumerate(name):
            diff=paths[:,step,i,5,CHANNELS]-paths[:,2,i,5,CHANNELS]
            s=stat(diff)
            interval=np.asarray(s["interval_95"])
            s["by_channel_within_tolerance"]=((interval.min(axis=0)>-MARGINAL_TOL)&
                                                (interval.max(axis=0)<MARGINAL_TOL)).tolist()
            record["arms"][arm]=s
        repeat.append(record)
    decisions={}
    for i,goal in enumerate(frozen["targets"]):
        success=parent.margins(paths,np.asarray(goal))>=-1e-12
        decisions[frozen["objective_names"][i]]={
            str(step):{"control":float(success[:,index,0].mean()),"edit":float(success[:,index,i+1].mean()),
                       "edit_vs_control":stat(success[:,index,i+1].astype(float)-success[:,index,0])}
            for index,step in enumerate(STEPS)}
    emit("SCORED.json",{**parent.stamps(),"paired_cards":N,
         "data_sha256":parent.digest(OUT/"PATHS.npz"),
         "physical_parameters":"identical from zero bonds; shared dense Brownian tape, initial state and event uniforms",
         "result":repeat,"inverse_hit":decisions})
    print(decisions,flush=True)


if __name__=="__main__":
    experiment()
