"""Train and blind-test a legal atlas residual, with or without stationary teachers."""
from argparse import ArgumentParser
from pathlib import Path
from shutil import copyfile

import numpy as np
from scipy.spatial.distance import cdist
from threadpoolctl import threadpool_limits

from aam_camera_information import history_features
from aam_formation_cycle_driver import np_load, np_save, read,save_tree,sha,write
from aam_formation_path_cycle import event,load_tree,stamp
from aam_stationary_distillation_net import build_geometry,cross_distances,fit,predict
from aam_stationary_teacher_data import SPLITS,TEACHER_CALLS
from package_aam_formation_local_cycle import source_closure
from run_aam_direct_energy_trajectory_law import atlas_receipts
from run_aam_mark_candidate_filter import verify
from run_aam_original_tape_pgas import ATLAS,paired
from run_aam_temporal_energy_law import predict_variant,readout_inputs


ROOT = Path(__file__).resolve().parent
PROTOCOL = ROOT/"AAM_STATIONARY_DISTILLATION_PROTOCOL_ZH.md"
AUDIT = ROOT/"audit_aam_stationary_distillation.py"
NS = "aam-physical-stationary-teacher-gru-residual-v1"
LEARNERS = ("single_truth","physical_teacher")
NAMES = (*ATLAS,*LEARNERS)
METRICS = ("early18","hidden_delta32_6")
TRAIN_SEED = 47265915
TEACHER_SEED = 935087251
EPOCHS = (8,24,64)
SHRINK = (0.,.5,1.)
GRID = tuple((e,s) for s in SHRINK for e in EPOCHS)


def register(out,atlas,bound,review,workers):
    ancestor, training, bank, refs = atlas_receipts(atlas)
    b0,b1,b2 = [bound/n for n in ("REGISTERED.json","SCORED.json","INDEPENDENT_AUDIT.json")]
    source,s,cert = (read(f) for f in (b0,b1,b2))
    ids = sum(SPLITS.values(),[])
    if (cert["passed"] is not True or cert["scored_sha256"]!=sha(b1)
        or source["physics"]!=ancestor["physics"]
        or set(ids).intersection(sum(ancestor["units"].values(),[])+source["units"])
        or len(set(ids))!=448):
        raise ValueError("Need same-law stationary audit and distinct new train/dev/held mothers.")
    codes = source_closure([Path(__file__),AUDIT])
    out.mkdir(exist_ok=False)
    snap = out/"SOURCE_SNAPSHOT"
    snap.mkdir()
    for n,f in codes.items():
        copyfile(f,snap/n)
    state = {"namespace":NS,"units":SPLITS,"physics":ancestor["physics"],"workers":workers,
             "atlas_model":str(bank),"atlas_units":training,
             "atlas_settings":read(atlas/"TRAINED_AND_SELECTED.json")["selected"],
             "epochs":EPOCHS,"shrink":SHRINK,"grid":GRID,"train_seed":TRAIN_SEED,"teacher_seed":TEACHER_SEED,
             "names":NAMES,"metrics":METRICS,"sources":{str(p):sha(p) for p in codes.values()},
             "inputs":{str(x):sha(x) for x in (*refs,b0,b1,b2,review,PROTOCOL)},
             "learner_fit_roles":["train"],"dev_selection_only":True,"old_holdout_reuse":False,
             "new_teacher_native_calls":len(SPLITS["train"])*TEACHER_CALLS,
             "forecast_native_calls":0,"new_wet_measurements":0,
             "dev_eligibility":"nonzero trained teacher residual, better in both DEV metrics than RBF, better average than single-label net",
             "held_gate":"paired lower nominal 95% bound positive vs all raw strong and matched single arm on both metrics"}
    write(out/"REGISTERED.json",{**stamp(),**state})
    write(out/"RUN.json",{**stamp(),"registered_sha256":sha(out/"REGISTERED.json"),"before_native_or_fit":True})


def lock(out):
    reg=out/"REGISTERED.json"
    p=read(reg)
    if p["namespace"]!=NS or p["units"]!=SPLITS or sha(reg)!=read(out/"RUN.json")["registered_sha256"]:
        raise ValueError("Require prospective unique mother mapping.")
    for q,h in {**p["sources"],**p["inputs"]}.items():
        if sha(q)!=h:
            raise ValueError("Frozen source or input changed: "+q)
    return p


def check(ref):
    return verify(Path(ref["path"]),ref)


def freeze_models(out,p):
    a=out/"TRAIN_TEACHER_FROZEN.json"
    tr=read(a)
    archive = np_load(check(tr["arrays"]))
    if (archive["unit"].tolist()!=SPLITS["train"] or
        archive["teacher_cross"].shape != (len(SPLITS["train"]),3,512) or
        archive["truth_cross"].shape!=archive["teacher_cross"].shape):
        raise ValueError("Use only all complete registered training mothers.")
    write(out/"FIT_STARTED.json",{**stamp(),"teacher_freeze_sha256":sha(a),
                           "roles":["train"],"dev_and_held_labels_opened":False})
    bank = load_tree(p["atlas_model"])
    Y = archive["measured"]
    local = bank["bank"]["local"]
    geometry = build_geometry(local,rank=16)
    with threadpool_limits(1):
        inputs = readout_inputs(bank,Y)
        baseline = predict_variant("kernel",p["atlas_settings"]["kernel"],bank,inputs)["weights"]
        options = {"geometry":geometry}
        for model,key in zip(LEARNERS,("truth_cross","teacher_cross"),strict=True):
            options[model] = fit(Y,baseline,archive[key][:,:2].mean(axis=1),geometry,bank["camera_scale"],
                                 seed=TRAIN_SEED,epochs=EPOCHS)
    snap=save_tree(out/"TRAINED_MODELS.npz",options)
    write(out/"FIT_FROZEN.json",{**stamp(),"model":snap,"teacher_freeze_sha256":sha(a),
          "roles":["train"],"dev_and_held_labels_opened":False,"fit_input_shapes":list(Y.shape)})


def split_view(out,role):
    name=out/f"{role.upper()}_ACQUIRED.json"
    a=read(name)
    if a["role"]!=role or [r["unit"] for r in a["rows"]]!=SPLITS[role]:
        raise ValueError("Incorrect original role membership.")
    y = np.stack([np_load(check(r["obs"]))["measured"] for r in a["rows"]])
    return name,a,y


def grams(atoms):
    x = atoms.reshape(len(atoms),-1)
    return cdist(x,x)/np.sqrt(x.shape[-1])


def atom_blocks(local):
    return local,local[:,2]-local[:,1]


def energy(w,train,target):
    a = train.reshape(len(train),-1)
    y = target.reshape(len(target),-1)
    D = grams(train)
    cross = cdist(y,a)/np.sqrt(a.shape[-1])
    return np.sum(w*cross,axis=-1)-.5*np.sum((w@D)*w,axis=-1)


def losses(bank,target,w):
    d=atom_blocks(bank)
    y=atom_blocks(target)
    return np.column_stack([energy(w,x,z) for x,z in zip(d,y,strict=True)])


def all_inputs(out,p,role):
    k,a,y = split_view(out,role)
    archive=p["atlas_model"]
    bank=load_tree(archive)
    m=read(out/"FIT_FROZEN.json")
    models=load_tree(check(m["model"]))
    info=readout_inputs(bank,y)
    weights = {n:predict_variant(n,p["atlas_settings"].get(n),bank,info)["weights"] for n in ATLAS}
    return k,a,y,bank,models,weights


def reveal(a):
    return np.stack([history_features(load_tree(check(q["raw"]))["packets"]) for q in a["rows"]])


def dev(out,p):
    k,a,y,bank,models,w = all_inputs(out,p,"dev")
    with threadpool_limits(1):
        all_weights = {m:np.stack([predict(models[m],models["geometry"],y,w["kernel"],epoch=e,shrink=s)
                      for e,s in GRID]) for m in LEARNERS}
    payload=save_tree(out/"DEV_FORECASTS.npz",{"unit":SPLITS["dev"],"weights":w,"grid":all_weights})
    f=out/"DEV_PREDICTIONS_FROZEN.json"
    write(f,{**stamp(),"archives":payload,"acquired_sha256":sha(k),"dev_labels_opened":False,
                      "fit_sha256":sha(out/"FIT_FROZEN.json")})
    local=reveal(a)
    with threadpool_limits(1):
        points = {m:np.stack([losses(bank["bank"]["local"],local,row) for row in ws]) for m,ws in all_weights.items()}
        chosen = {m:int(np.argmin(v.mean(axis=(1,2)))) for m,v in points.items()}
        baselines={m:losses(bank["bank"]["local"],local,ws) for m,ws in w.items()}
    single,teacher=(points[m][chosen[m]].mean(axis=0) for m in LEARNERS)
    keep = (GRID[chosen["physical_teacher"]][1]>0. and
            np.all(teacher < baselines["kernel"].mean(axis=0)) and teacher.mean()<single.mean())
    source=save_tree(out/"DEV_SCORE_ARRAYS.npz",{"unit":SPLITS["dev"],"grid_losses":points,
             "baseline_losses":baselines})
    decision=out/"DEV_SELECTED.json"
    write(decision,{**stamp(),"forecast_sha256":sha(f),"held_not_opened":True,"dev_grid":GRID,
         "chosen_index":chosen,"chosen":{m:GRID[i] for m,i in chosen.items()},"go_to_fresh_held":bool(keep),
         "teacher_mean":teacher,"single_mean":single,"ref_means":{m:v.mean(0) for m,v in baselines.items()},
         "arrays":source})
    event(out,"tuned_once_on_dev",fresh_held_admitted=bool(keep))


def held(out,p):
    d=read(out/"DEV_SELECTED.json")
    if d["go_to_fresh_held"] is not True:
        raise ValueError("DEV did not authorize confirmation.")
    k,a,y,bank,models,w=all_inputs(out,p,"held")
    for m,(e,s) in d["chosen"].items():
        w[m]=predict(models[m],models["geometry"],y,w["kernel"],epoch=e,shrink=s)
    rr=np_save(out/"HELD_WEIGHTS.npz",query=y,**w)
    f=out/"PREDICTIONS_FROZEN.json"
    write(f,{**stamp(),"weight_ref":rr,"acquired_sha256":sha(k),"dev_selection_sha256":sha(out/"DEV_SELECTED.json"),
                      "held_units":SPLITS["held"],"held_truth_opened":False,"native_prediction":0})
    with threadpool_limits(1):
        local=reveal(a)
        z={m:losses(bank["bank"]["local"],local,x) for m,x in w.items()}
    g={m:[paired(u) for u in (x-z["physical_teacher"]).T] for m,x in z.items() if m!="physical_teacher"}
    passes=all(a[1]>0. for row in g.values() for a in row)
    rr=save_tree(out/"HELD_SCORES.npz",{"unit":SPLITS["held"],"losses":z})
    write(out/"SCORED.json",{**stamp(),"held_forecast_sha256":sha(f),"dev_selected_sha256":sha(out/"DEV_SELECTED.json"),
          "held_mothers":len(SPLITS["held"]),"held_opened":True,"means":{m:v.mean(0) for m,v in z.items()},
          "metrics":METRICS,"paired_gains":g,"go_to_original_inverse":bool(passes),"arrays":rr,
          "study_conclusion":"real blind original simulation predictions; real-material supervision absent"})


def stop(out,p):
    dev=out/"DEV_SELECTED.json"
    d=read(dev)
    if d["go_to_fresh_held"] or (out/"HELD_ACQUIRED.json").exists():
        raise ValueError("Development closure would conceal allowed or actual fresh confirmation.")
    write(out/"SCORED.json",{**stamp(),"held_opened":False,"held_mothers":0,"metrics":METRICS,
          "dev_selected_sha256":sha(dev),"go_to_original_inverse":False,
          "conclusion":"did not pass prospective train-only proof-of-benefit; held addresses not consumed",
          "dev_result":d})
    event(out,"development_branch_closed",held_consumed=0)


if __name__=="__main__":
    cli=ArgumentParser(description=__doc__)
    q=cli.add_subparsers(dest="action",required=True)
    r=q.add_parser("register")
    for v in ("out","atlas","bound","review"):
        r.add_argument(v,type=Path)
    r.add_argument("--workers",type=int,default=8)
    actions={"fit":freeze_models,"dev":dev,"held":held,"stop":stop}
    for label in actions:
        q.add_parser(label).add_argument("out",type=Path)
    a=cli.parse_args()
    if a.action=="register":
        register(a.out.resolve(),a.atlas,a.bound,a.review,a.workers)
    else:
        out=a.out.resolve(strict=True)
        actions[a.action](out,lock(out))
