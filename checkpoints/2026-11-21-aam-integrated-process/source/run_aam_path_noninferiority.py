"""Prospective matched-support non-inferiority of the original three-frame recovery."""

from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from shutil import copyfile

import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import norm
from threadpoolctl import threadpool_limits

from aam_camera_information import history_features
from aam_formation_cycle_driver import np_load, np_save, read, save_tree, sha, write
from aam_formation_path_cycle import event, load_tree, stamp
from aam_large_phase_atlas import energy_scores, pair_uniforms, probabilities
from aam_multiscale_phase_law import multiscale_laws
from aam_regret_noninferiority import score_noninferiority
from package_aam_formation_local_cycle import source_closure
from run_aam_direct_energy_trajectory_law import atlas_receipts
from run_aam_large_phase_atlas import acquire_one, digest
from run_aam_stationary_distillation import check
from run_aam_temporal_energy_law import predict_variant, readout_inputs


ROOT=Path(__file__).resolve().parent
NS="aam-frozen-ess1024-path-noninferiority-fresh2304-v1"
PROTOCOL=ROOT/"AAM_STRONG_BASELINE_NONINFERIORITY_PROTOCOL_ZH.md"
AUDITOR=ROOT/"audit_aam_path_noninferiority.py"
UNITS=list(range(2026158000,2026160304))
ANCHOR_OFFSETS=(0,329,657,986,1316,1645,1974,2303)
METRICS=("early18","hidden_delta32_6")
METHODS=("ess1024","phase_rbf","raw_rbf","kernel","kernel_cme","knn","empirical")
PRIMARY=.01
SENSITIVITY=(0.,.005,.02)
SEED=817223451
DRAWS=32768
WORK={}


def register(out,old_root,bank_root,development,review,workers):
    old, old_units, model, previous = atlas_receipts(old_root)
    bank_reg,bank_freeze,bank_audit,bank_score=(bank_root/n for n in (
        "REGISTERED.json","BANK_FROZEN.json","INDEPENDENT_AUDIT.json","SCORED.json"))
    dev_reg,dev_score,dev_audit,dev_arrays=(development/n for n in (
        "REGISTERED.json","SCORED.json","INDEPENDENT_AUDIT.json","DEV_SCORE_ARRAYS.npz"))
    b, da, a, p=map(read,(bank_reg,dev_audit,bank_audit,dev_reg))
    stats=read(dev_score)
    if (a["passed"] is not True or a["scored_sha256"]!=sha(bank_score)
            or da["passed"] is not True or da["scored_sha256"]!=sha(dev_score)
            or p["physics"]!=b["physics"] or b["physics"]!=old["physics"]
            or stats["held_opened"] or stats["go_to_original_inverse"] or not read(review)["review_received"]):
        raise ValueError("Need audited old legal bank and development without any reopened confirmation.")
    excluded=set(sum(old["units"].values(),[])+sum(b["units"].values(),[])+sum(p["units"].values(),[]))
    if excluded.intersection(UNITS):
        raise ValueError("No new mother can coincide with old training, developing, or reserved confirmation.")
    bank=check(read(bank_freeze)["arrays"])
    arr=read(dev_score)["dev_result"]["arrays"]
    if sha(dev_arrays)!=arr["sha256"]:
        raise ValueError("Development planning array drift.")
    loss=load_tree(dev_arrays)["losses"]
    sensitivity={}
    z=norm.ppf(1-.05/(len(METRICS)*(len(METHODS)-1)))+norm.ppf(.9)
    for method in METHODS[1:]:
        ref=loss[method]
        s=(loss[METHODS[0]]-1.01*ref).std(axis=0,ddof=1)
        sensitivity[method]=np.ceil(np.square(z*s/(.005*ref.mean(axis=0)))).astype(int)
    if max(max(n) for n in sensitivity.values())>len(UNITS):
        raise ValueError("The one-endpoint conservative planning target requires a larger independent cohort.")
    files=(*previous,bank_reg,bank_freeze,bank_audit,bank_score,dev_reg,dev_score,dev_audit,dev_arrays,
           bank,review,PROTOCOL)
    sources=source_closure([Path(__file__),AUDITOR])
    out.mkdir(exist_ok=False)
    cache=out/"SOURCE_SNAPSHOT"
    cache.mkdir()
    for name,path in sources.items():
        copyfile(path,cache/name)
    reg=out/"REGISTERED.json"
    write(reg,{**stamp(),"namespace":NS,"physics":b["physics"],"units":UNITS,"old_model":str(model),
          "old_settings":b["atlas_settings"],"old_atlas_units":old_units,"bank_arrays":str(bank),
          "bank_units":b["units"]["bank"],"methods":METHODS,"metrics":METRICS,"primary_margin":PRIMARY,
          "margins":(0.,.005,PRIMARY,.02),"pair_seed":SEED,"pair_draws":DRAWS,"anchor_offsets":ANCHOR_OFFSETS,
          "workers":workers,"inputs":{str(f):sha(f) for f in files},
          "sources":{str(f):sha(f) for f in sources.values()},"development_planning_n_at_true_0p5pct":sensitivity,
          "new_native_forecast":0,"new_fits":0,"wet_measurements":0,
          "family":"one-sided nominal simultaneous 95% over 6 controls by 2 original outcomes"})
    write(out/"RUN.json",{**stamp(),"registered_sha256":sha(reg),"before_native_or_fit":True})
    event(out,"fixed_new_cohort_registered_before_native",mothers=len(UNITS),locked_margin=PRIMARY)


def locked(out):
    reg=out/"REGISTERED.json"
    p=read(reg)
    if p["namespace"]!=NS or p["units"]!=UNITS or read(out/"RUN.json")["registered_sha256"]!=sha(reg):
        raise ValueError("Original cohort and registration required.")
    for q,d in {**p["inputs"],**p["sources"]}.items():
        if sha(q)!=d:
            raise ValueError("Source or input changed: "+q)
    return p


def acquire(out,p):
    folder=out/"held"
    folder.mkdir(exist_ok=False)
    start=out/"ACQUISITION_STARTED.json"
    write(start,{**stamp(),"registered_sha256":sha(out/"REGISTERED.json")})
    jobs=[(folder,p["physics"],"held",u) for u in UNITS]
    with ProcessPoolExecutor(p["workers"]) as pool:
        rows=list(pool.map(acquire_one,jobs,chunksize=4))
    write(out/"ACQUIRED.json",{**stamp(),"rows":rows,"started_sha256":sha(start),"native_calls":3*len(rows)})
    event(out,"complete_original_cameras_and_sealed_truth",mothers=len(rows),native_calls=3*len(rows))


def rows(out):
    f=out/"ACQUIRED.json"
    records=read(f)["rows"]
    if [r["unit"] for r in records]!=UNITS or any(r["native_calls"]!=3 for r in records):
        raise ValueError("New original mother list mismatched.")
    return f,records


def coords(z):
    return (z.reshape(-1,18)/np.sqrt(18), (z[:,2]-z[:,1]).reshape(-1,6)/np.sqrt(6))


def setup(bank,model,settings):
    WORK.update(bank=np_load(bank),old=load_tree(model),settings=settings)
    x=coords(WORK["old"]["bank"]["local"])
    WORK["old_metric"]=[(z,cdist(z,z)) for z in x]


def old_score(y,p):
    return np.array([p@cdist(z,q)[...,0]-.5*p@d@p
                     for q,(z,d) in zip(coords(y[None]),WORK["old_metric"],strict=True)])


def weights(q):
    b,o,s=WORK["bank"],WORK["old"],WORK["settings"]
    law=probabilities(b["clean"],b["measured"],q,o["noise"],o["camera_scale"],o["kernel_sq_scale"],s["kernel"])
    own=multiscale_laws(b["clean"],q,o["noise"],targets=(1024,))
    info=readout_inputs(o,q[None])
    measure={m:predict_variant(m,s.get(m),o,info)["weights"][0] for m in METHODS[3:]}
    return {METHODS[0]:own["weights"][METHODS[0]],"phase_rbf":law["weights"]["phase_rbf"],
            "raw_rbf":law["weights"]["raw_rbf"],**measure}


def forecast_one(task):
    i,r=task
    q=np_load(check(r["obs"]))["measured"]
    with threadpool_limits(1):
        all_weights=weights(q)
    record={"unit":r["unit"],"obs_sha256":r["obs"]["sha256"],
            "digests":{m:digest(v) for m,v in all_weights.items()}}
    return record,all_weights if i in ANCHOR_OFFSETS else None


def args(p):
    return (p["bank_arrays"],p["old_model"],p["old_settings"])


def forecast(out,p):
    f,records=rows(out)
    with ProcessPoolExecutor(p["workers"],initializer=setup,initargs=args(p)) as pool:
        z=list(pool.map(forecast_one,enumerate(records),chunksize=4))
    d=out/"FORECAST_DESCRIPTORS.json"
    write(d,{**stamp(),"rows":[r for r,_ in z]})
    anchor=save_tree(out/"AUDIT_WEIGHTS.npz",{str(r["unit"]):w for r,w in z if w is not None})
    write(out/"PREDICTIONS_FROZEN.json",{**stamp(),"descriptor_sha256":sha(d),"acquisition_sha256":sha(f),
        "audit_weights":anchor,"held_truth_opened":False,"forecast_native":0})
    event(out,"forecasts_bound_before_hidden_reader",mothers=len(records))


def score_one(task):
    r,d=task
    u=r["unit"]
    q=np_load(check(r["obs"]))["measured"]
    with threadpool_limits(1):
        w=weights(q)
        if u!=d["unit"] or r["obs"]["sha256"]!=d["obs_sha256"] or {m:digest(x) for m,x in w.items()}!=d["digests"]:
            raise ValueError("Read target only after confirming original forecast bytes.")
        y=history_features(load_tree(check(r["raw"]))["packets"])
        uniforms=[pair_uniforms(np.random.default_rng(np.random.SeedSequence([SEED,u,i])),DRAWS) for i in range(2)]
        proper=energy_scores(WORK["bank"]["local"],{m:w[m] for m in METHODS[:3]},y,*uniforms)["methods"]
        bank_scores=[np.stack([proper[m]["energy_main"][:2],proper[m]["energy_repeat"][:2]]) for m in METHODS[:3]]
        base_scores=[np.tile(old_score(y,w[m]),(2,1)) for m in METHODS[3:]]
    score=np.stack([*bank_scores,*base_scores],axis=1)
    return u,score


def score(out,p):
    f,r=rows(out)
    freeze=out/"PREDICTIONS_FROZEN.json"
    digest_file=out/"FORECAST_DESCRIPTORS.json"
    earlier=read(freeze)
    if (earlier["acquisition_sha256"]!=sha(f) or earlier["descriptor_sha256"]!=sha(digest_file)
            or earlier["held_truth_opened"]):
        raise ValueError("Freeze mismatch before opening outcomes.")
    with ProcessPoolExecutor(p["workers"],initializer=setup,initargs=args(p)) as pool:
        all_scores=list(pool.map(score_one,zip(r,read(digest_file)["rows"],strict=True),chunksize=4))
    ids,repeats=zip(*all_scores,strict=True)
    if list(ids)!=UNITS:
        raise ValueError("Score population changed.")
    repeats=np.stack(repeats)
    losses=repeats.mean(axis=1)
    data=np_save(out/"SCORED_ARRAYS.npz",unit=ids,losses=losses,replica=repeats)
    result=score_noninferiority(losses,METHODS,METRICS,METHODS[0],METHODS[1:],
                                primary_margin=PRIMARY,sensitivity_margins=SENSITIVITY)
    write(out/"SCORED.json",{**stamp(),"freeze_sha256":sha(freeze),"arrays":data,"new_mothers":len(ids),
        "noninferiority":result,"passed":result["primary"]["passed"],"methods":METHODS,"metrics":METRICS,
        "replica_score_means":repeats.mean(axis=0),"native_calls_generation":3*len(ids),
        "new_wet_measurements":0,"new_fits":0})
    event(out,"fresh_method_level_noninferiority_scored",mothers=len(ids),passed=result["primary"]["passed"])


if __name__=="__main__":
    q=ArgumentParser(description=__doc__)
    sp=q.add_subparsers(dest="action",required=True)
    a=sp.add_parser("register")
    a.add_argument("base",type=Path)
    for name in ("old_root","bank_root","development","review"):
        a.add_argument(name,type=Path)
    a.add_argument("--workers",type=int,default=8)
    for name in ("acquire","forecast","score"):
        sp.add_parser(name).add_argument("out",type=Path)
    a=q.parse_args()
    if a.action=="register":
        out=a.base.resolve()/datetime.now(timezone.utc).strftime("aam_path_noninferiority_%Y%m%dT%H%M%S%fZ")
        register(out,a.old_root,a.bank_root,a.development,a.review,a.workers)
        print(out,flush=True)
    else:
        out=a.out.resolve(strict=True)
        {"acquire":acquire,"forecast":forecast,"score":score}[a.action](out,locked(out))
