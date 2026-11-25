"""Register, acquire, freeze and independently restart the AAm gene-design loop."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timedelta, timezone
from hashlib import file_digest, sha256
import json
from pathlib import Path
import numpy as np


ROOT = Path(__file__).resolve().parent
RUN_ID = ROOT.name.rsplit("_", 1)[-1]
N_DEV, N_HELD, REPS, N_ANCHORS = 128, 36, 4, 72
N_PLANNING, N_VERIFY = 8, 16
N_FORECAST = 24
BOUND = 1.10
ENV = ("A", "B", "C")
STEP = .48
METHODS = ("ridge", "rbf", "shared_response")
INDICES = ((2, 2), (3, 3), (5, 0), (5, 3))
OBJECTIVES = ("early_network_with_chain", "delayed_network_then_catch_up")


def actions():
    z = np.zeros(4)
    atlas = [z]
    labels = ["no_edit"]
    for j, name in enumerate(("initiation", "propagation", "pendant", "bridge")):
        for sign in (-1, 1):
            row = z.copy()
            row[j] = STEP * sign
            atlas.append(row)
            labels.append(f"{name}_{'up' if sign > 0 else 'down'}")
    for a in (-1, 1):
        for b in (-1, 1):
            atlas.append(STEP * np.array([a,b,0.,0.]))
            labels.append(f"init_growth_{'up' if a > 0 else 'down'}_{'up' if b > 0 else 'down'}")
    return np.asarray(atlas), labels


def stamps():
    t = datetime.now(timezone.utc)
    return {"utc": t.isoformat(), "asia_shanghai": t.astimezone(
        timezone(timedelta(hours=8))).isoformat()}


def digest(path):
    with Path(path).open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def load(name):
    return json.loads((ROOT/name).read_text(encoding="utf-8-sig"))


def write(name, value):
    with (ROOT/name).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def write_arrays(name, **data):
    with (ROOT/name).open("xb") as stream:
        np.savez_compressed(stream, **data)


def stable_rng(role):
    stamp = sha256(f"{RUN_ID}|integrated-AAm|{role}".encode()).digest()
    return np.random.default_rng(int.from_bytes(stamp[:16],"little"))


def seeds(role, shape):
    ids = stable_rng(role).integers(0, 2**63-1, size=shape)
    assert len(np.unique(ids)) == ids.size
    return ids


def target_features(path):
    return np.stack([path[..., i,j] for i,j in INDICES], axis=-1)


def bounds(path):
    v = target_features(path)
    q = lambda col, x: float(np.quantile(v[:,col],x,method="linear"))
    return np.array([
        [[q(0,.25),q(1,.65),q(2,.40),0.], [1.,1.,1.,1.]],
        [[q(0,.25),0.,0.,q(3,.40)], [1.,q(1,.35),1.,1.]],
    ])


def margins(values, box):
    current = target_features(values)
    # Min distance to operative boundary; bounds 0/1 only restrict physical support.
    low, high = box
    low_diff = np.where(low>0, (current - low)/.15, np.inf)
    high_diff = np.where(high<1, (high - current)/.15, np.inf)
    result = np.minimum(low_diff, high_diff).min(axis=-1)
    if not np.isfinite(result).all():
        raise ValueError("Every design task requires at least one operative bound.")
    return result


def model_versions(short=False):
    import reaction_adapter as sim
    dependencies = sim.sources()
    names = ["loop.py","PROTOCOL_ZH.md","reaction_adapter.py"]
    if not short:
        names.extend(("code_encoder.py","DESIGN_PARAMETERS_ZH.md"))
    dependencies.update({str(ROOT/name):digest(ROOT/name) for name in names})
    return dependencies


def register():
    from reaction_adapter import PATH_NAMES, TIMES as PATH_TIMES, LOCAL_NAMES, EARLY_TIMES as OBSERVATION_TIMES
    g, labels = actions()
    write("REGISTERED.json",{
        **stamps(),"run_id":RUN_ID, "source_sha256":model_versions(),
        "division": {"development_cards":N_DEV,"confirmation_cards":N_HELD,
                     "observations_per_context":REPS,"unseen_context_zero_anchors":N_ANCHORS,
                     "decoded_path_draws":N_FORECAST,"planning_draws":N_PLANNING,
                     "validation_draws_per_action_card":N_VERIFY},
        "observed_contexts":ENV[:2],"new_context":ENV[2],
        "static_code_prior":"independent uniform in [-.48,.48]",
        "local_names":list(LOCAL_NAMES), "local_times":list(OBSERVATION_TIMES),
        "global_names":list(PATH_NAMES), "global_times":list(PATH_TIMES),
        "actions":g.tolist(),"action_labels":labels,
        "primary_forward":"per card ensemble energy, global times 2.05,3.5,4.85,6 and channels conversion/Lw/largest/bridge/wrap; fixed historical scale",
        "primary_inverse":"per-card common-seed mean joint hit; report both objectives separately",
        "anchor_quantile":"q.25 Lw(t2); fast +M(t3) q.65 +X(t5) q.40; delayed -M(t3) q.35 +M(t5) q.40; strict equality allowed",
        "raw_stage":"single run from original monomer configuration for every learned or true edited code",
        "sampling_gate":"Encoder and choices frozen before either unedited or edited C confirmation; crystal ball only read in scoring",
        "training_env_only":"A/B early local features, known code labels. C only 0 code reference independent of withheld cards.",
        "new_wet_measurements":0,
    })


def check_versions():
    record=load("REGISTERED.json")
    for path,value in record["source_sha256"].items():
        if digest(path) != value:
            raise RuntimeError(f"Source hash mismatch: {path}")
    return record


def worker(job):
    from reaction_adapter import simulate
    code, context, seed, full, journal = job
    return simulate(np.asarray(code),context,int(seed),full=full,journal=journal)


def run_jobs(jobs, workers, title, batch=12):
    print(f"{title}: starting {len(jobs)} trajectories", flush=True)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        result = []
        for i,val in enumerate(pool.map(worker,jobs,chunksize=batch), start=1):
            result.append(val)
            if i%512==0:
                print(f"{title}: {i}/{len(jobs)}",flush=True)
    print(f"{title}: finished {len(result)}",flush=True)
    return result


def early_jobs(g, addr):
    return [(code, context, seed, False, False)
            for code, m in zip(g,addr) for context, row in zip(ENV[:2],m) for seed in row]


def collect(workers):
    check_versions()
    train_code = stable_rng("development codes").uniform(-.48,.48,(N_DEV,4))
    query_code = stable_rng("blind codes").uniform(-.48,.48,(N_HELD,4))
    train_seeds = seeds("development early trajectories", (N_DEV,2,REPS))
    query_seeds = seeds("held early trajectories", (N_HELD,2,REPS))
    anchor_seeds = seeds("registered zero code anchors",N_ANCHORS)
    zero = np.zeros(4)
    job = early_jobs(train_code,train_seeds) + early_jobs(query_code,query_seeds)
    job += [(zero,ENV[2],rid,True,False) for rid in anchor_seeds]
    result = run_jobs(job,workers,"masked A/B and C-zero anchors")
    a, b = train_seeds.size, train_seeds.size + query_seeds.size
    train_X, query_X = [np.asarray([r["local"] for r in block]).reshape(count,2,REPS,-1)
                        for block,count in ((result[:a],N_DEV),(result[a:b],N_HELD))]
    anchors = np.asarray([r["path"] for r in result[b:]])
    assert np.isfinite(train_X).all() and np.isfinite(query_X).all() and np.isfinite(anchors).all()
    write_arrays("DEVELOPMENT.npz",X=train_X,G=train_code,seeds=train_seeds)
    write_arrays("QUERY_LOCAL.npz",X=query_X,seeds=query_seeds)
    write_arrays("CODE_ESCROW.npz",G=query_code)
    write_arrays("ZERO_C_ANCHOR.npz",path=anchors,seeds=anchor_seeds)
    write("ACQUISITION.json",{**stamps(),
        "inputs":{name:digest(ROOT/name) for name in ("DEVELOPMENT.npz","QUERY_LOCAL.npz","CODE_ESCROW.npz","ZERO_C_ANCHOR.npz")},
        "query_truth_access_allowed":"confirm only","boxes":bounds(anchors).tolist()})


def rank(path, goal):
    slack = margins(path,goal)
    hit=(slack >= -1.e-12).mean(axis=-1)
    mean=slack.mean(axis=-1)
    all_choices=[int(np.lexsort((np.arange(len(z)), -q, -z))[0]) for z,q in zip(hit,mean)]
    return np.array(all_choices),hit,mean


def freeze(workers):
    from code_encoder import encode
    check_versions()
    receipts=load("ACQUISITION.json")
    for name in ("DEVELOPMENT.npz","QUERY_LOCAL.npz","ZERO_C_ANCHOR.npz"):
        assert receipts["inputs"][name]==digest(ROOT/name)
    with np.load(ROOT/"DEVELOPMENT.npz") as d:
        X, G = d["X"], d["G"]
    with np.load(ROOT/"QUERY_LOCAL.npz") as d:
        Q = d["X"]
    available = encode(X,G,Q,n_draws=N_FORECAST,seed=int(stable_rng("posterior draws").integers(0,2**31-1)))
    draws=np.asarray([available[method] for method in METHODS])
    assert draws.shape==(len(METHODS),N_HELD,N_FORECAST,4)
    assert np.isfinite(draws).all() and abs(draws).max()<=.48000001
    forward = np.concatenate((draws,np.zeros((1,N_HELD,N_FORECAST,4))),axis=0)
    edit, labels = actions()
    forward_ids = seeds("unseen context forecast", (N_HELD,N_FORECAST))
    planning_ids = seeds("inverse planning address", (N_HELD,N_PLANNING))
    # Code approximation has been fitted, but trajectories still use the common physical decoder.
    forward_jobs = [(code,ENV[2],sid,True,False) for branches in forward
                    for sample, row in zip(branches,forward_ids) for code,sid in zip(sample,row)]
    codes = np.clip(forward[:,:,None,:N_PLANNING,:] + edit[None,None,:,None,:],-BOUND,BOUND)
    planning_jobs = [(code,ENV[2],sid,True,False) for m in codes for query,row in zip(m,planning_ids)
                     for action in query for code,sid in zip(action,row)]
    result_f = run_jobs(forward_jobs,workers,"frozen C forecast")
    result_r = run_jobs(planning_jobs,workers,"blind edit ranks")
    shapes_f=(len(forward),N_HELD,N_FORECAST)
    shapes_r=(len(forward),N_HELD,len(edit),N_PLANNING)
    pack=lambda values, name, shape: np.asarray([r[name] for r in values]).reshape(*shape,*np.asarray(values[0][name]).shape)
    fpath, fmilestone = (pack(result_f,name,shapes_f) for name in ("path","milestones"))
    rpath = pack(result_r,"path",shapes_r)
    targets=np.asarray(receipts["boxes"])
    ranks=np.empty((len(forward),len(targets),N_HELD),dtype=int)
    probability=np.empty((*ranks.shape,len(edit)))
    for meth in range(len(forward)):
        for i, goal in enumerate(targets):
            ranks[meth,i], probability[meth,i], _ = rank(rpath[meth],goal)
    write_arrays("PREDICTED.npz",code=draws,forward=fpath,forward_milestones=fmilestone,
                 planning=rpath,choice=ranks,probability=probability,forward_seeds=forward_ids,
                 planning_seeds=planning_ids)
    write("FROZEN.json",{**stamps(),"acquisition_sha256":digest(ROOT/"ACQUISITION.json"),
          "registration_sha256":digest(ROOT/"REGISTERED.json"),
          "predicted_sha256":digest(ROOT/"PREDICTED.npz"),
          "forecasters":[*METHODS,"nominal"],"objective_names":list(OBJECTIVES),"actions":labels,
          "targets":targets.tolist(),"selected_actions":ranks.tolist(),
          "confirm_protocol":"New seed group for whole menu. True code never loaded during rank computation."})
    print("FROZEN SHA",digest(ROOT/"FROZEN.json"),flush=True)


def seal_future(workers, expected_freeze):
    record=check_versions()
    if digest(ROOT/"FROZEN.json") != expected_freeze:
        raise RuntimeError("Expected immutable freeze hash before loading escrow.")
    frozen=load("FROZEN.json")
    assert frozen["registration_sha256"] == digest(ROOT/"REGISTERED.json")
    assert frozen["predicted_sha256"] == digest(ROOT/"PREDICTED.npz")
    assert frozen["acquisition_sha256"] == digest(ROOT/"ACQUISITION.json")
    assert load("ACQUISITION.json")["inputs"]["CODE_ESCROW.npz"] == digest(ROOT/"CODE_ESCROW.npz")
    with np.load(ROOT/"CODE_ESCROW.npz") as e:
        actual=e["G"]
    edit,_=actions()
    ids=seeds("independent C reruns of all actions",(N_HELD,N_VERIFY))
    choice = {i:set(np.asarray(frozen["selected_actions"])[...,i].ravel().tolist())|{0}
              for i in range(N_HELD)}
    task = []
    audit_order = []
    for i, (g, addr) in enumerate(zip(actual,ids)):
        for j, delta in enumerate(edit):
            for k, seed in enumerate(addr):
                journal = bool(k==0 and j in choice[i])
                task.append((np.clip(g+delta,-BOUND,BOUND),ENV[2],seed,True,journal))
                if journal:
                    audit_order.append((i,j,k))
    result=run_jobs(task,workers,"independent C full-menu paired restarts")
    frozen_after=digest(ROOT/"FROZEN.json")
    assert frozen_after==expected_freeze
    shape=(N_HELD,len(edit),N_VERIFY)
    path=np.asarray([x["path"] for x in result]).reshape(*shape,6,10)
    milestone=np.asarray([x["milestones"] for x in result]).reshape(*shape,2,-1)
    journals=[result[(i*len(edit)+j)*N_VERIFY+k]["journal"] for i,j,k in audit_order]
    sizes=np.array([len(j) for j in journals])
    write_arrays("CONFIRMED.npz",path=path,milestones=milestone,seeds=ids,code=actual,
                 journals=np.concatenate(journals,axis=0),journal_keys=np.array(audit_order),
                 journal_offsets=np.r_[0,np.cumsum(sizes)])
    write("CONFIRMED.json",{**stamps(),"verified_prediction_sha256":frozen_after,
          "sha256":digest(ROOT/"CONFIRMED.npz"),"journal_trace_count":len(journals),"new_wet_measurements":0})


if __name__=="__main__":
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("stage",choices=("register","collect","freeze","confirm"))
    cli.add_argument("--workers",type=int,default=8)
    cli.add_argument("--freeze-sha256")
    opts=cli.parse_args()
    if opts.stage=="register":
        register()
    elif opts.stage=="collect":
        collect(opts.workers)
    elif opts.stage=="freeze":
        freeze(opts.workers)
    else:
        seal_future(opts.workers,opts.freeze_sha256)
