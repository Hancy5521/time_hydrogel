"""Controlled pendant activation timing with a fixed native monomer-to-network solver."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from hashlib import file_digest,sha256
from pathlib import Path
import json
import sys

import numpy as np


ROOT = Path(__file__).resolve().parent
PARENT = ROOT.parent
sys.path.insert(0,str(PARENT/"scheduled_edit"))
sys.path.insert(0,str(PARENT))
from scheduled import digest,read,write,slack,PARENT_LOCK

PLAN, HOLD = (16,8), (32,24)


def vectors():
    nominal = [
        {"label":"zero","switch":None,"early":1.,"late":1.,"delta_init":0.},
        {"label":"init_minus","switch":None,"early":1.,"late":1.,"delta_init":-.48},
    ]
    gates = [{
        "label":f"pendant_{early:g}_until_{switch:g}",
        "switch":switch,"early":early,"late":(6. - early*switch)/(6.-switch),"delta_init":0.,
    } for switch in (2.85,3.50) for early in (0.,.25)]
    return nominal+gates


def arrays(destination,/,**fields):
    with destination.open("xb") as stream:
        np.savez_compressed(stream,**fields)


def rng(role):
    key = f"{PARENT.name}/separate-pendant-activity-v1/{role}"
    return np.random.default_rng(int.from_bytes(sha256(key.encode()).digest()[:16],"little"))


def cohort(seed_role,shape):
    m,n = shape
    seed = rng(seed_role+" Brownian and marked reaction addresses").integers(0,2**63-1,(m,n))
    if np.unique(seed).size!=seed.size:
        raise ArithmeticError("Seed role duplicates.")
    return rng(seed_role+" four-coordinate materials").uniform(-.48,.48,(m,4)),seed


def warm():
    import reaction_adapter as sim
    base = sim.ENVIRONMENTS["C"]
    for entry in vectors()[2:]:
        clock = entry["switch"]
        pulses = tuple(sorted((*base["pulses"],(clock,1.,base["pulses"][-1][2]))))
        sim.ENVIRONMENTS[entry["label"]] = {
            "feed":base["feed"],"diffusion":base["diffusion"],"pulses":pulses,
        }


def worker(item):
    import reaction_adapter as sim
    code,gate,seed,journal = item
    code = code.copy()
    code[0] += gate["delta_init"]
    switch = gate["switch"]
    if switch is None:
        return sim.simulate(code,"C",int(seed),full=True,journal=journal)
    original = sim.advance_fixed
    cut = sim._tick(switch)

    def controlled(packet,start,end,*innovations):
        if start<cut<end:
            raise ArithmeticError("Native stage crossed a prescribed rate switch.")
        rates = packet[3]
        pending = rates[4:6].copy()
        try:
            rates[4:6] *= gate["early"] if start<cut else gate["late"]
            return original(packet,start,end,*innovations)
        finally:
            rates[4:6] = pending

    sim.advance_fixed = controlled
    try:
        return sim.simulate(code,gate["label"],int(seed),full=True,journal=journal)
    finally:
        sim.advance_fixed = original


def sources():
    import reaction_adapter as sim
    import audit_reaction as raw
    return sim.sources() | raw.sources() | {
        str(p):digest(p) for p in (Path(__file__),ROOT/"PROTOCOL_ZH.md",PARENT/"FROZEN.json",
                                    PARENT/"scheduled_edit"/"scheduled.py",
                                    PARENT/"scheduled_edit"/"SCORED.json")
    }


def register():
    import reaction_adapter as sim
    warm()
    _, native = sim._clock("C",True)
    for v in vectors()[2:]:
        bridge, side = sim._clock(v["label"],True)
        np.testing.assert_array_equal(side,native)
        if sim._tick(v["switch"]) not in [x[0] for x in bridge]:
            raise ValueError("A pendant switch is not explicitly presented to the native solver.")
        check = v["early"]*v["switch"] + v["late"]*(6.-v["switch"])
        if not np.isclose(check,6.,rtol=0.,atol=1e-12):
            raise ValueError("Gate changed integral of the imposed time signal.")
    if digest(PARENT/"FROZEN.json") != PARENT_LOCK:
        raise ValueError("Inherited target lock invalid.")
    write(ROOT/"REGISTERED.json",{
        "source_sha256":sources(),
        "parent_freeze":PARENT_LOCK,"gate_candidates":vectors(),
        "target":read(PARENT/"FROZEN.json")["targets"][1],
        "design_role_sizes":list(PLAN),"confirm_role_sizes":list(HOLD),
        "test":"Comparison of joint delayed catch-up hit, fixed target thresholds and blinded run policy.",
        "practical_threshold":.05,"comparison_clusters":"independent code cards","wet":0,
    })


def checked():
    reg=read(ROOT/"REGISTERED.json")
    if reg["source_sha256"]!=sources():
        raise ValueError("A registered native or protocol dependency changed.")
    return reg


def run(gates,selected,g,seeds,journals,workers):
    tasks=((code,gates[q],addr,journals and k==0) for q in selected
           for code,row in zip(g,seeds,strict=True) for k,addr in enumerate(row))
    values=[]
    with ProcessPoolExecutor(max_workers=workers,initializer=warm) as pool:
        for i,val in enumerate(pool.map(worker,tasks,chunksize=14),1):
            values.append(val)
            if i % 768 == 0:
                print(f"pendant native paths {i}",flush=True)
    return values


def plan(workers):
    reg=checked()
    gates = reg["gate_candidates"]
    g,seeds=cohort("development",PLAN)
    result=run(gates,range(len(gates)),g,seeds,False,workers)
    path=np.asarray([p["path"] for p in result]).reshape(len(gates),*PLAN,6,10)
    margin=slack(path,np.asarray(reg["target"]))
    hit=(margin>=-1e-12).mean(axis=(1,2))
    weakness=margin.mean(axis=(1,2))
    choice=min(range(2,len(gates)),key=lambda j:(-hit[j],-weakness[j],j))
    arrays(ROOT/"PLAN.npz",path=path,code=g,seeds=seeds)
    write(ROOT/"FROZEN.json",{
        "registration_sha256":digest(ROOT/"REGISTERED.json"),
        "planning_sha256":digest(ROOT/"PLAN.npz"),"selected":choice,
        "selected_gate":gates[choice],"candidate_hits":hit.tolist(),
        "candidate_mean_margin":weakness.tolist(),
    })
    print("FROZEN",digest(ROOT/"FROZEN.json"),gates[choice],hit.tolist(),flush=True)


def seal(sha):
    if sha!=digest(ROOT/"FROZEN.json"):
        raise ValueError("Supply the immutable pre-reveal FROZEN SHA.")
    reg=checked()
    frozen=read(ROOT/"FROZEN.json")
    if (frozen["registration_sha256"]!=digest(ROOT/"REGISTERED.json")
            or frozen["planning_sha256"]!=digest(ROOT/"PLAN.npz")):
        raise ArithmeticError("Planning inputs or model changed.")
    return reg,frozen


def confirm(sha,workers):
    if (ROOT/"CONFIRMED.npz").exists():
        raise FileExistsError("Confirmation already exists.")
    reg,frozen=seal(sha)
    g,seeds=cohort("confirmation",HOLD)
    selected=[0,1,frozen["selected"]]
    result=run(reg["gate_candidates"],selected,g,seeds,True,workers)
    dim=(len(selected),*HOLD)
    paths=np.asarray([p["path"] for p in result]).reshape(*dim,6,10)
    milestones=np.asarray([p["milestones"] for p in result]).reshape(*dim,2,5)
    tapes=[p["journal"] for p in result if "journal" in p]
    arrays(ROOT/"CONFIRMED.npz",path=paths,milestones=milestones,seeds=seeds,g=g,selected=np.array(selected),
           raw=np.concatenate(tapes),offsets=np.r_[0,np.cumsum([len(t) for t in tapes])])
    write(ROOT/"CONFIRMED.json",{
        "frozen_sha256":sha,"npz_sha256":digest(ROOT/"CONFIRMED.npz"),"saved":len(tapes),
        "n_restarts":int(np.prod(dim)),
    })


def seed_check():
    roles=[
        (ROOT/"PLAN.npz",("seeds",)),(ROOT/"CONFIRMED.npz",("seeds",)),
        *((PARENT/name, keys) for name,keys in (
        ("DEVELOPMENT.npz",("seeds",)),("QUERY_LOCAL.npz",("seeds",)),
        ("ZERO_C_ANCHOR.npz",("seeds",)),("PREDICTED.npz",("forward_seeds","planning_seeds")),
        ("CONFIRMED.npz",("seeds",)),("scheduled_edit/PLANNED.npz",("seeds",)),
        ("scheduled_edit/CONFIRMED.npz",("seeds",)))),
    ]
    seen=set(range(8914112,8914152)); totals={}
    for source,fields in roles:
        with np.load(source,allow_pickle=False) as z:
            for field in fields:
                address=set(z[field].ravel().tolist())
                if address & seen or len(address)!=z[field].size:
                    raise ArithmeticError("Shared address between distinct random roles.")
                totals[f"{source.relative_to(PARENT)}:{field}"]=len(address)
                seen.update(address)
    return totals


def score(sha):
    from audit_reaction import replay
    reg,frozen=seal(sha)
    conf=read(ROOT/"CONFIRMED.json")
    if conf["npz_sha256"]!=digest(ROOT/"CONFIRMED.npz") or conf["frozen_sha256"]!=sha:
        raise ArithmeticError("Confirmation archive or freeze does not match receipt.")
    with np.load(ROOT/"CONFIRMED.npz",allow_pickle=False) as z:
        paths,milestones,raw,offsets,seeds,selected=[z[n] for n in (
            "path","milestones","raw","offsets","seeds","selected")]
    assert list(selected)==[0,1,frozen["selected"]] and len(offsets)==conf["saved"]+1
    audit=[0.,0.]
    for ii,(a,b) in enumerate(zip(offsets[:-1],offsets[1:],strict=True)):
        option,card=divmod(ii,HOLD[0])
        child=np.random.SeedSequence(int(seeds[card,0])).spawn(5)[0]
        budget=int(np.random.default_rng(child).integers(6,13))
        ref,land,_=replay(raw[a:b],budget,1.95)
        audit[0]=max(audit[0],float(np.abs(ref-paths[option,card,0]).max()))
        audit[1]=max(audit[1],float(np.abs(land-milestones[option,card,0]).max()))
    if max(audit)>2e-13:
        raise ArithmeticError("Native event replay differs.")
    target=np.asarray(reg["target"])
    win=slack(paths,target)>=-1e-12
    # Direct independent per-field test matches replayed composite at registered tolerances.
    fields=np.stack([paths[...,2,2],paths[...,3,3],paths[...,5,0],paths[...,5,3]],axis=-1)
    sliced=np.all((fields>=target[0]-1.5e-13)&(fields<=target[1]+1.5e-13),axis=-1)
    np.testing.assert_array_equal(sliced,win)
    hits=win.mean(axis=-1)
    boot=rng("resampling mothers").integers(HOLD[0],size=(10000,HOLD[0]))

    def summary(value):
        return {"estimate":float(value.mean()),"paired_ci95":np.quantile(value[boot].mean(axis=1),[.025,.975]).tolist()}

    compare={name:summary(hits[2]-hits[base]) for base,name in enumerate(("vs_zero","vs_native_init_minus"))}
    scores={}
    for i,k in enumerate(selected):
        v=reg["gate_candidates"][k]
        scores[v["label"]]={
            "n_pass":int(win[i].sum()),"n":int(win[i].size),**summary(hits[i]),
            "criterion_pass":(((fields[i]>=target[0]-1.5e-13)&(fields[i]<=target[1]+1.5e-13)).mean(axis=(0,1))).tolist(),
            "criterion_mean":fields[i].mean(axis=(0,1)).tolist(),
            "milestones_reached":milestones[i,:,:,0:2].mean(axis=(0,1)).tolist(),
        }
    practical=bool(compare["vs_native_init_minus"]["estimate"]>=.05 and
                   min(v["paired_ci95"][0] for v in compare.values())>0.)
    write(ROOT/"SCORED.json",{
        "frozen_sha256":sha,"confirmation_sha256":conf["npz_sha256"],
        "scores":scores,"paired_contrasts":compare,"practical_gate_passed":practical,
        "raw_labeled_paths":conf["n_restarts"],"independent_seed_addresses":seed_check(),
        "audit_saved_journals":conf["saved"],"audit_replayed_events":len(raw),
        "audit_max_errors":audit,"wet":0,
    })
    print({"confirmed":conf["n_restarts"],"scores":{k:v["n_pass"] for k,v in scores.items()},
           "contrasts":compare,"practical":practical,"audit_counts":len(raw)},flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage",choices=("register","plan","confirm","score"))
    p.add_argument("--workers",type=int,default=8)
    p.add_argument("--freeze-sha256")
    x=p.parse_args()
    if x.stage=="register": register()
    elif x.stage=="plan": plan(x.workers)
    elif x.stage=="confirm": confirm(x.freeze_sha256,x.workers)
    else: score(x.freeze_sha256)
