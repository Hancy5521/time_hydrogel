"""Verify both inherited native process action policies on the same new precursors."""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
import os
import sys
import numpy as np

ROOT=Path(__file__).resolve().parent
PARENT=ROOT.parent
sys.path[:0]=[str(PARENT/"pendant_timing"),str(PARENT/"scheduled_edit"),str(PARENT)]
import scheduled as s
import pendant as p

N_CARDS,N_REPS=48,16
STEP=16
ARMS=("frozen_early","static_early","frozen_delayed","static_delayed","zero")
IDS=("schedule","schedule","pendant","pendant","schedule")


def rng(role):
    data=sha256((PARENT.name+"/final-paired-halved-step/"+role).encode()).digest()
    return np.random.default_rng(int.from_bytes(data[:16],"little"))


def deps():
    return s.source_binding() | p.sources() | {str(file):s.digest(file) for file in (
        ROOT/"pair.py",ROOT/"PROTOCOL_ZH.md",PARENT/"scheduled_edit/FROZEN.json",
        PARENT/"pendant_timing/FROZEN.json",PARENT/"pendant_timing/SCORED.json",
        PARENT/"scheduled_edit/SCORED.json",
    )}


def warm(use_private):
    if use_private:
        box=Path(os.environ["AAM_PAIR_CACHE_ROOT"])/str(os.getpid())
        box.mkdir(parents=True,exist_ok=True)
        os.environ["NUMBA_CACHE_DIR"]=str(box)
    import reaction_adapter as sim
    sim.STEP=STEP
    sim._clock.cache_clear()
    s.warm()
    p.warm()


def register():
    import reaction_adapter as sim
    p_reg=p.checked()
    s_reg=s.checked()
    p_free_file=PARENT/"pendant_timing/FROZEN.json"
    s_free_file=PARENT/"scheduled_edit/FROZEN.json"
    p_free,s_free=s.read(p_free_file),s.read(s_free_file)
    last=s.read(PARENT/"pendant_timing/SCORED.json")
    assert last["practical_gate_passed"] and last["frozen_sha256"]==s.digest(p_free_file)
    s_score=s.read(PARENT/"scheduled_edit/SCORED.json")
    assert s_score["frozen_sha256"]==s.digest(s_free_file)
    both=s.read(PARENT/"FROZEN.json")["targets"]
    assert both==s_reg["targets"] and both[1]==p_reg["target"]
    early=s_free["decisions"]["early_network_with_chain"]
    delayed=s_free["decisions"]["delayed_network_then_catch_up"]
    selected=[s_reg["catalog"][early["scheduled"]],s_reg["catalog"][early["static"]],
              p_free["selected_gate"],p_reg["gate_candidates"][1],s_reg["catalog"][0]]
    assert delayed["static"]==1 and selected[3]["delta_init"]==s_reg["catalog"][1]["edit"][0]
    assert selected[0]["edit"]==selected[1]["edit"]
    warm(False)
    _, core=sim._clock("C",True)
    for arm,kind in zip(selected,IDS,strict=True):
        local=arm["context"] if kind=="schedule" else (arm["label"] if arm["switch"] is not None else "C")
        stage,check=sim._clock(local,True)
        np.testing.assert_array_equal(check,core)
        if kind=="pendant" and arm["switch"] is not None:
            assert sim._tick(arm["switch"]) in [x[0] for x in stage]
    s.write(ROOT/"FROZEN.json",{"source_sha256":deps(),"size":[N_CARDS,N_REPS],
            "arms":ARMS,"mode":IDS,"selected":selected,"targets":both,
            "step_ticks":STEP,"nominal_step":STEP*sim.TICK,
            "clock_nodes":len(core),"source_experiment":"new no-code-contact paired confirm only"})


def verify(sha):
    file=ROOT/"FROZEN.json"
    if s.digest(file)!=sha:
        raise ValueError("Must provide verified freeze hash")
    lock=s.read(file)
    if lock["source_sha256"]!=deps():
        raise ValueError("Frozen study sources differ")
    return lock


def work(task):
    kind,arm,code,address,audit=task
    pack=(code,arm,address,audit)
    return (s.worker if kind=="schedule" else p.worker)(pack)


def legacy_seeds():
    entries=[
        ("DEVELOPMENT.npz",["seeds"]),("QUERY_LOCAL.npz",["seeds"]),("ZERO_C_ANCHOR.npz",["seeds"]),
        ("PREDICTED.npz",["forward_seeds","planning_seeds"]),("CONFIRMED.npz",["seeds"]),
        ("scheduled_edit/PLANNED.npz",["seeds"]),("scheduled_edit/CONFIRMED.npz",["seeds"]),
        ("pendant_timing/PLAN.npz",["seeds"]),("pendant_timing/CONFIRMED.npz",["seeds"]),
        ("topology_witness/DEV.npz",["seeds"]),("topology_witness/HELD.npz",["seeds"]),
        ("continuous_bridge_transfer/DEV.npz",["seeds"]),("continuous_bridge_transfer/QUERY.npz",["seeds"]),
        ("continuous_bridge_transfer/PREDICTED.npz",["seeds"]),("continuous_bridge_transfer/CONFIRMED.npz",["seeds"]),
    ]
    address=[]
    for name,fields in entries:
        with np.load(PARENT/name,allow_pickle=False) as x:
            address.extend(x[k].ravel() for k in fields)
    return np.concatenate(address)


def run(sha, workers):
    lock=verify(sha)
    code=rng("mother cards").uniform(-.48,.48,(N_CARDS,4))
    addr=rng("new reaction addresses").integers(0,2**63-1,(N_CARDS,N_REPS))
    all_seed=np.r_[legacy_seeds(),addr.ravel()]
    assert np.unique(all_seed).size==all_seed.size
    task=((mode,arm,g,seed,j==0) for mode,arm in zip(lock["mode"],lock["selected"],strict=True)
          for g,group in zip(code,addr,strict=True) for j,seed in enumerate(group))
    output=[]
    with ProcessPoolExecutor(max_workers=workers,initializer=warm,initargs=(True,)) as executor:
        for i,result in enumerate(executor.map(work,task,chunksize=16),1):
            output.append(result)
            if i%480==0:
                print("paired native",i,flush=True)
    verify(sha)
    dims=(len(ARMS),N_CARDS,N_REPS)
    journals=[case for case in output if "journal" in case]
    lengths=[len(j["journal"]) for j in journals]
    target=ROOT/"RAW.npz"
    with target.open("xb") as out:
        np.savez_compressed(out,code=code,seeds=addr,
                   path=np.asarray([k["path"] for k in output]).reshape(*dims,6,10),
                   milestones=np.asarray([k["milestones"] for k in output]).reshape(*dims,2,5),
                   journals=np.concatenate([k["journal"] for k in journals]),
                   offsets=np.r_[0,np.cumsum(lengths)],
                   initiators=np.array([k["initiators"] for k in journals]),
                   feed=np.array([k["feed_time"] for k in journals]))
    s.write(ROOT/"CONFIRMED.json",{"freeze":sha,"raw_sha256":s.digest(target),"native_paths":len(output),
            "new_seed_addresses":int(addr.size),"all_seed_addresses":int(all_seed.size)})


def score(sha):
    from audit_reaction import replay
    lock=verify(sha)
    confirmation=s.read(ROOT/"CONFIRMED.json")
    assert confirmation["freeze"]==sha and confirmation["raw_sha256"]==s.digest(ROOT/"RAW.npz")
    with np.load(ROOT/"RAW.npz",allow_pickle=False) as pack:
        raw,ms=pack["path"],pack["milestones"]
        offset,events,budget,feed=(pack[k] for k in ("offsets","journals","initiators","feed"))
    assert len(offset)==N_CARDS*len(ARMS)+1
    err=[0.,0.]
    for j,(a,b) in enumerate(zip(offset[:-1],offset[1:],strict=True)):
        mode,card=divmod(j,N_CARDS)
        states,mile,_=replay(events[a:b],int(budget[j]),float(feed[j]))
        err[0]=max(err[0],float(np.abs(raw[mode,card,0]-states).max()))
        err[1]=max(err[1],float(np.abs(ms[mode,card,0]-mile).max()))
    assert max(err)<=2.e-13
    early,late=[s.slack(raw,np.asarray(box))>=-1.e-12 for box in lock["targets"]]
    table={"early":np.stack([early[0],early[1],early[4]]),
           "delayed":np.stack([late[2],late[3],late[4]])}
    table["both_matched"]=table["early"]&table["delayed"]
    assert not table["both_matched"][-1].any()
    rows={name:value.mean(axis=-1) for name,value in table.items()}
    rows["product_under_independent_addresses"]=rows["early"]*rows["delayed"]
    choices=rng("card confidence measurement").integers(0,N_CARDS,(10000,N_CARDS))
    def ci(value):
        return [float(q) for q in np.quantile(value[choices].mean(axis=1),[.025,.975])]
    stats={}
    for metric,value in rows.items():
        entry={name:{"estimate":float(r.mean()),"ci95":ci(r)} for name,r in
               zip(("frozen","static","zero"),value,strict=True)}
        if metric in table:
            for j,name in enumerate(("frozen","static","zero")):
                entry[name]["hits"]=int(table[metric][j].sum())
                entry[name]["trials"]=N_CARDS*N_REPS
        for j,tag in ((1,"static"),(2,"zero")):
            d=value[0]-value[j]
            entry["frozen_minus_"+tag]={"estimate":float(d.mean()),"ci95":ci(d)}
        stats[metric]=entry
    delta=(raw[0,...,3,3]-raw[2,...,3,3]).mean(axis=1)
    s.write(ROOT/"SCORED.json",{"freeze":sha,"raw_sha256":confirmation["raw_sha256"],
        "results":stats,"middle_largest_early_minus_delayed":{"estimate":float(delta.mean()),"ci95":ci(delta)},
        "replayed_journals":int(len(offset)-1),"replayed_events":int(len(events)),"audit_max_errors":err})
    print(stats,flush=True)


if __name__=="__main__":
    cli=argparse.ArgumentParser()
    cli.add_argument("stage",choices=("register","confirm","score"))
    cli.add_argument("--workers",type=int,default=5)
    cli.add_argument("--freeze-sha256")
    z=cli.parse_args()
    if z.stage=="register":
        register()
    elif z.stage=="confirm":
        run(z.freeze_sha256,z.workers)
    else:
        score(z.freeze_sha256)
