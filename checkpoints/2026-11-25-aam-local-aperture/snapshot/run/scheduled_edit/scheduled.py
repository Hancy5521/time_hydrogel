"""Pre-register, select and confirm formation-process schedules on new simulated mothers."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime,timedelta,timezone
from hashlib import file_digest,sha256
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parent
PARENT = ROOT.parent
sys.path.insert(0,str(PARENT))
PARENT_LOCK = "07f8eee193484148dec344bcecf6aba3f8f0a1bb1479c4e4c320650134b7132d"
PLAN_MOTHERS, PLAN_REPS, HOLD_MOTHERS, HOLD_REPS = 16, 8, 32, 24
STEP = .48
TARGET_COLUMNS = ((2,2), (3,3), (5,0), (5,3))
GOALS = ("early_network_with_chain", "delayed_network_then_catch_up")
BOOT = 10000


def digest(path):
    with path.open("rb") as stream:
        return file_digest(stream,"sha256").hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write(path, value):
    now = datetime.now(timezone.utc)
    with path.open("x",encoding="utf-8") as stream:
        json.dump({"utc":now.isoformat(),"asia_shanghai":now.astimezone(
            timezone(timedelta(hours=8))).isoformat(),**value},stream,indent=2,allow_nan=False)


def arrays(path, **data):
    with path.open("xb") as stream:
        np.savez_compressed(stream,**data)


def rng(role):
    key = f"{PARENT.name}/scheduled-edit-v1/{role}"
    return np.random.default_rng(int.from_bytes(sha256(key.encode()).digest()[:16],"little"))


def paired_data(role, cards, reps):
    g = rng(role+" material").uniform(-.48,.48,(cards,4))
    address = rng(role+" paths").integers(0,2**63-1,(cards,reps))
    if np.unique(address).size != address.size:
        raise ArithmeticError("Independent stochastic stream unexpectedly repeated.")
    return g,address


def controls():
    # All changes happen on the same .025 grid used by original C.
    late_boost = (6. - 3.5*.7) / (6. - 3.5)
    catch_up = (6. - 1.35*1.15 - (3.5-1.35)*.35) / (6. - 3.5)
    schedule = {
        "C": (1.95, ((0.,1.),)),
        "early_feed": (.65, ((0.,1.),)),
        "late_feed": (3.55, ((0.,1.),)),
        "late_feed_damped": (3.55, ((0.,.7),(3.5,late_boost))),
        "late_feed_pause": (4.15, ((0.,1.15),(1.35,.35),(3.5,catch_up))),
    }
    output = {}
    for label,(feed,pulse) in schedule.items():
        starts = [t for t,_ in pulse] + [6.]
        total = sum((b-a)*mult for (a,mult),b in zip(pulse,starts[1:],strict=True))
        if not np.isclose(total,6.,rtol=0.,atol=1e-12):
            raise ArithmeticError("A schedule changed integrated nominal initiation.")
        checkpoint = sorted({1.775,*[t for t,_ in pulse]})
        output[label] = {
            "feed":feed, "diffusion":.19,
            "pulses":tuple((t,[v for s,v in pulse if s<=t][-1],1. if t<1.775 else .34/.19)
                           for t in checkpoint),
        }
    return output


def candidates():
    rows = [("zero", [0.]*4)]
    for j,name in enumerate(("initiation","growth","pendant","bridge")):
        for sign,stem in ((-1,"minus"),(1,"plus")):
            change = [0.]*4
            change[j] = sign*STEP
            rows.append((f"{name}_{stem}",change))
    for a in (-1,1):
        for b in (-1,1):
            names = ("minus","plus")
            rows.append((f"both_{names[a>0]}_{names[b>0]}",[a*STEP,b*STEP,0.,0.]))
    stat = [{"label":label,"context":"C","edit":edit} for label,edit in rows]
    combo = np.array([0,4,12])
    return stat + [
        {"label":f"{context}__{stat[i]['label']}","context":context,"edit":stat[i]["edit"]}
        for context in controls() if context!="C" for i in combo
    ]


def source_binding():
    import reaction_adapter as sim
    import audit_reaction
    result = sim.sources() | audit_reaction.sources()
    return result | {str(p):digest(p) for p in (
        Path(__file__).resolve(),ROOT/"PROTOCOL_ZH.md",PARENT/"FROZEN.json",PARENT/"REGISTERED.json",
    )}


def assert_grid():
    import reaction_adapter as sim
    old = sim.ENVIRONMENTS["C"]
    new = controls()
    np.testing.assert_allclose(new["C"]["feed"],old["feed"],rtol=0.,atol=0.)
    _, native = sim._clock("C",True)
    for context, recipe in new.items():
        if context != "C":
            sim.ENVIRONMENTS[context] = recipe
        _, nodes = sim._clock(context,True)
        np.testing.assert_array_equal(nodes,native,err_msg=context)


def register():
    file = PARENT/"FROZEN.json"
    if digest(file) != PARENT_LOCK:
        raise ValueError("Original target thresholds failed the archive hash check.")
    parent = read(file)
    assert_grid()
    write(ROOT/"REGISTERED.json",{
        "sources_sha256":source_binding(),
        "parent_freeze_sha256":PARENT_LOCK,"targets":parent["targets"],
        "objectives":list(GOALS),"catalog":candidates(),"schedules":controls(),
        "design_population":"four independent uniform rule coords [-.48,.48]",
        "planning":[PLAN_MOTHERS,PLAN_REPS],"confirmation":[HOLD_MOTHERS,HOLD_REPS],
        "mode_range":"13 static arms; 12 scheduled arms",
        "CRN":"Exact same native Brownian sampling grid, initial draw, waits and mark-uniform addresses across actions.",
        "source_of_confirmation":"new simulation; 0 real material measurements",
    })


def checked():
    r = read(ROOT/"REGISTERED.json")
    if r["sources_sha256"] != source_binding():
        raise ValueError("Registered source or original locks changed.")
    return r


def warm():
    import reaction_adapter as sim
    for context,schedule in controls().items():
        if context!="C":
            sim.ENVIRONMENTS[context] = schedule


def worker(job):
    import reaction_adapter as sim
    code, candidate, seed, trace = job
    edited = np.clip(code+candidate["edit"],-sim.BOUND,sim.BOUND)
    return sim.simulate(edited,candidate["context"],int(seed),full=True,journal=trace)


def native(catalog, code, seed, selected, trace, workers):
    jobs = ((value,catalog[i],address,trace and k==0)
            for i in selected for value,row in zip(code,seed,strict=True)
            for k,address in enumerate(row))
    out = []
    with ProcessPoolExecutor(max_workers=workers,initializer=warm) as pool:
        for j,row in enumerate(pool.map(worker,jobs,chunksize=12),1):
            if j%768==0:
                print("native paths",j,flush=True)
            out.append(row)
    return out


def slack(path, target):
    values = np.stack([path[..., i,j] for i,j in TARGET_COLUMNS],axis=-1)
    low,high = target
    left = np.where(low>0,(values-low)/.15,np.inf)
    right = np.where(high<1,(high-values)/.15,np.inf)
    return np.minimum(left,right).min(axis=-1)


def pick(margins, ids):
    hit = (margins>=-1e-12).mean(axis=(1,2))
    mean = margins.mean(axis=(1,2))
    order = np.lexsort((np.asarray(ids),-mean[ids],-hit[ids]))
    return int(np.asarray(ids)[order[0]]),hit,mean


def plan(workers):
    reg = checked()
    if (ROOT/"PLANNED.npz").exists() or (ROOT/"FROZEN.json").exists():
        raise FileExistsError("Planning already completed.")
    catalog = reg["catalog"]
    g,seeds = paired_data("development",PLAN_MOTHERS,PLAN_REPS)
    ids = np.arange(len(catalog))
    result = native(catalog,g,seeds,ids,False,workers)
    path = np.asarray([r["path"] for r in result]).reshape(len(ids),PLAN_MOTHERS,PLAN_REPS,6,10)
    arrays(ROOT/"PLANNED.npz", code=g,seeds=seeds,paths=path)
    decisions = {}
    for goal,bound in zip(GOALS,reg["targets"],strict=True):
        m = slack(path,np.asarray(bound))
        static,hit,mean = pick(m,np.arange(13))
        scheduled,*_ = pick(m,np.arange(13,25))
        decisions[goal] = {
            "static":static,"scheduled":scheduled,
            "planning_hits":hit.tolist(),"planning_mean_slack":mean.tolist(),
        }
    write(ROOT/"FROZEN.json",{
        "register_sha256":digest(ROOT/"REGISTERED.json"),
        "predicted_sha256":digest(ROOT/"PLANNED.npz"),"decisions":decisions,
        "unread_confirm":True,
    })
    print("FROZEN",digest(ROOT/"FROZEN.json"),{
        name:(catalog[d["static"]]["label"],catalog[d["scheduled"]]["label"])
        for name,d in decisions.items()},flush=True)


def validation_ids(catalog,decisions):
    rows = {0}
    match = lambda context,edit: next(i for i,x in enumerate(catalog)
                                       if x["context"]==context and x["edit"]==edit)
    for v in decisions.values():
        s = catalog[v["scheduled"]]
        rows.update((v["static"],v["scheduled"],match("C",s["edit"]),match(s["context"],[0.]*4)))
    return sorted(rows)


def confirm(workers,sha):
    f=ROOT/"FROZEN.json"
    if sha != digest(f):
        raise ValueError("Need the immutable supplied forecast hash before opening confirmation paths.")
    reg = checked()
    locked = read(f)
    if (locked["register_sha256"]!=digest(ROOT/"REGISTERED.json")
            or locked["predicted_sha256"]!=digest(ROOT/"PLANNED.npz")):
        raise ValueError("Planning archives changed before verification.")
    if (ROOT/"CONFIRMED.npz").exists():
        raise FileExistsError("This confirm can never be re-run.")
    g,seeds = paired_data("confirm",HOLD_MOTHERS,HOLD_REPS)
    with np.load(ROOT/"PLANNED.npz",allow_pickle=False) as declared:
        if np.intersect1d(seeds,declared["seeds"]).size:
            raise ValueError("Disjoint seed roles were violated.")
    catalog = reg["catalog"]
    ids = validation_ids(catalog,locked["decisions"])
    result = native(catalog,g,seeds,ids,True,workers)
    dims = (len(ids),HOLD_MOTHERS,HOLD_REPS)
    path = np.asarray([r["path"] for r in result]).reshape(*dims,6,10)
    milestones = np.asarray([r["milestones"] for r in result]).reshape(*dims,2,5)
    trace = [r for r in result if "journal" in r]
    sizes = [len(r["journal"]) for r in trace]
    arrays(ROOT/"CONFIRMED.npz", selected_ids=np.asarray(ids),path=path,milestones=milestones,
           true_code=g,seeds=seeds,journal=np.concatenate([r["journal"] for r in trace]),
           offsets=np.r_[0,np.cumsum(sizes)])
    write(ROOT/"CONFIRMED.json",{
        "frozen_sha256":sha,"confirmed_sha256":digest(ROOT/"CONFIRMED.npz"),
        "simulations":int(np.prod(dims)),"saved_journals":len(trace),
    })
    print(f"Confirmed {np.prod(dims)} new paths",flush=True)


def score(sha):
    from audit_reaction import replay
    if digest(ROOT/"FROZEN.json")!=sha:
        raise ValueError("Incorrect freeze hash for scoring.")
    reg = checked()
    locked,confirm = [read(ROOT/f) for f in ("FROZEN.json","CONFIRMED.json")]
    if confirm["frozen_sha256"]!=sha or confirm["confirmed_sha256"]!=digest(ROOT/"CONFIRMED.npz"):
        raise ValueError("Frozen inputs or outcomes differ.")
    catalog = reg["catalog"]
    actual = np.load(ROOT/"CONFIRMED.npz",allow_pickle=False)
    paths,milestones,seeds,ids = [actual[key] for key in ("path","milestones","seeds","selected_ids")]
    boot = rng("group bootstrap").integers(HOLD_MOTHERS,size=(BOOT,HOLD_MOTHERS))
    audit_error = [0.,0.]
    total_events = 0
    for j,(start,end) in enumerate(zip(actual["offsets"][:-1],actual["offsets"][1:],strict=True)):
        choice,card = divmod(j,HOLD_MOTHERS)
        radical = int(np.random.default_rng(np.random.SeedSequence(
            int(seeds[card,0])).spawn(5)[0]).integers(6,13))
        feed = reg["schedules"][catalog[ids[choice]]["context"]]["feed"]
        p,m,_ = replay(actual["journal"][start:end],radical,feed)
        e = float(np.abs(p-paths[choice,card,0]).max()),float(np.abs(m-milestones[choice,card,0]).max())
        audit_error = list(map(max,audit_error,e))
        total_events += end-start
    if max(audit_error)>2.e-13:
        raise ArithmeticError("Saved native event graph did not replay.")
    audited = {
        "n_journals":confirm["saved_journals"],"n_events":int(total_events),
        "max_errors":[float(v) for v in audit_error],"new_native_paths_from_audit":0,
    }

    def estimate(x):
        return {"estimate":float(x.mean()),"ci95":np.quantile(x[boot].mean(axis=1),[.025,.975]).tolist()}

    ranked = {}
    for name,target in zip(GOALS,reg["targets"],strict=True):
        d = locked["decisions"][name]
        win = (slack(paths,np.asarray(target))>=-1e-12)
        hits = win.mean(axis=2)
        table = {int(idx):row for idx,row in zip(ids,hits,strict=True)}
        found = catalog[d["scheduled"]]
        static_same = next(i for i,x in enumerate(catalog) if x["edit"]==found["edit"] and x["context"]=="C")
        schedule_noedit = next(i for i,x in enumerate(catalog)
                               if x["edit"]==[0.]*4 and x["context"]==found["context"])
        contrasts = {
            "scheduled_minus_best_static":(d["scheduled"],d["static"]),
            "scheduled_minus_zero":(d["scheduled"],0),
            "static_minus_zero":(d["static"],0),
            "scheduled_minus_same_g_at_native_u":(d["scheduled"],static_same),
            "scheduled_minus_same_u_at_zero_g":(d["scheduled"],schedule_noedit),
        }
        ranked[name] = {
            "chosen":{v:catalog[d[v]] for v in ("static","scheduled")},
            "hits":{catalog[idx]["label"]:{**estimate(row),"n_success":int(w.sum()),"n_paths":int(w.size)}
                    for idx,row,w in zip(ids,hits,win,strict=True)},
            "contrasts":{metric:{"labels":[catalog[i]["label"],catalog[j]["label"]],
                                 **estimate(table[i]-table[j])}
                         for metric,(i,j) in contrasts.items()},
        }
    write(ROOT/"SCORED.json",{
        "frozen_sha256":sha,"true_outcomes_sha256":confirm["confirmed_sha256"],
        "source_sha256":digest(Path(__file__)), "audit":audited, "results":ranked,
        "scope":"Population process controllability. Candidate lookup did not use individual observed snapshots.",
    })
    print(json.dumps({g:{"chosen":r["chosen"],
                         "effect":r["contrasts"]["scheduled_minus_best_static"]}
                      for g,r in ranked.items()},indent=2),flush=True)


if __name__=="__main__":
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("stage",choices=("register","plan","confirm","score"))
    cli.add_argument("--workers",type=int,default=8)
    cli.add_argument("--freeze-sha256")
    cfg=cli.parse_args()
    if cfg.stage=="register": register()
    elif cfg.stage=="plan": plan(cfg.workers)
    elif cfg.stage=="confirm": confirm(cfg.workers,cfg.freeze_sha256)
    else: score(cfg.freeze_sha256)
