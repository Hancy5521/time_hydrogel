"""A prospective shared trajectory law for the original recovery and inverse tasks."""
from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from shutil import copyfile

import numpy as np
from threadpoolctl import threadpool_limits

from aam_formation_cycle_driver import np_load, np_save, read, sha, write, worker_initializer
from aam_formation_cycle_methods import GOALS
from aam_formation_path_cycle import event, stamp
from aam_regret_noninferiority import score_noninferiority
from aam_same_atlas_phase_bank import bank_input_boundary, cache_one
from aam_same_atlas_phase_plan import plan as plan_query
from package_aam_formation_local_cycle import source_closure
from run_aam_atlas_action_gap_control import read_inputs
from run_aam_continuous_inverse import ACTIONS, REFERENCE_REPEATS, TIMES, acquire_one, original_losses, reference_one
from run_aam_large_phase_atlas import digest
from run_aam_mark_candidate_filter import verify
from run_aam_path_noninferiority import (
    METHODS as PATH_METHODS, METRICS as PATH_METRICS, setup as path_setup, score_one as path_score,
    weights as path_weights, WORK as PATH_WORK,
)
from run_aam_privileged_v5_action_headroom import best_fixed, reference

ROOT = Path(__file__).resolve().parent
PROTOCOL = ROOT/"AAM_SAME_ATLAS_INTEGRATED_PROTOCOL_ZH.md"
AUDITOR = ROOT/"audit_aam_same_atlas_integration.py"
NS = "aam-original-recovery-inverse-same-atlas-ess1024-20261121"
UNITS = list(range(2026163000,2026163512))
ANCHORS = (0, 73, 146, 219, 292, 365, 438, 511)
INVERSE_METHODS = ("ess1024_cophase","ess1024_uniform_phase","old_population","fixed")
ALPHA = .025
FORECAST = {}


def register(out, bank, old_path_plan, atlas, inverse, population, review, workers):
    b = bank_input_boundary(bank)
    old = read(old_path_plan)
    _, _, norm, _, files = read_inputs(atlas,population,inverse)
    cfg, fixed_norm, fixed = best_fixed(inverse)
    if (b["physics"] != old["physics"] or b["physics"] != cfg or norm["mechanical_scale"] !=
            list(fixed_norm["mechanical_scale"]) or not read(review)["review_received"]):
        raise ValueError("Bind the unchanged registered physics, normalization and Work assessment.")
    bank_ids = old["bank_units"]
    route_ids = [r["unit"] for r in b["selected_records"]]
    if (b["bank_arrays"]["path"] != old["bank_arrays"] or
        [bank_ids[i] for i in b["selected_rows"]] != route_ids or
        set(UNITS).intersection(bank_ids+old["units"]+old["old_atlas_units"])):
        raise ValueError("Shared bank identity or fresh cohort does not match.")
    old_inputs = [Path(old["old_model"]),Path(old["bank_arrays"])]
    inputs = {**b["inputs"], **{str(p):sha(p) for p in (*files,old_path_plan,review,PROTOCOL,*old_inputs)}}
    if any(inputs[str(p)]!=old["inputs"][str(p)] for p in old_inputs):
        raise ValueError("Prior recovery input digest changed.")
    sources = source_closure([Path(__file__),AUDITOR])
    hashes = {str(p):sha(p) for p in sources.values()}
    if any(hashes[p]!=h for p,h in old["sources"].items() if p in hashes):
        raise ValueError("Shared source changed since original recovery.")
    out.mkdir(exist_ok=False)
    reg = out/"REGISTERED.json"
    write(reg,{**stamp(),"namespace":NS,"units":UNITS,"physics":cfg,"normalization":norm,"bank":b,
          "bank_ids":bank_ids,"route_ids":route_ids,"old_plan":old,"times":TIMES,"goals":GOALS,
          "path_metrics":PATH_METRICS,"path_methods":PATH_METHODS,"inverse_methods":INVERSE_METHODS,
          "fixed_controls":np.asarray(ACTIONS)[fixed["choices"]],
          "population_controls":read(population/"SCORED.json")["controls"],"alpha":ALPHA,
          "primary_path_margin":.01,"primary_inverse_margin":.05,"anchor_positions":ANCHORS,
          "workers":workers,"reference_repeats":REFERENCE_REPEATS,"sources":hashes,"inputs":inputs,
          "simulation_reference_bound":3*len(UNITS)*len(INVERSE_METHODS)*len(GOALS)*REFERENCE_REPEATS,
          "legacy_outcome_payloads_opened":0,"new_fits":0,"wet_measurements":0})
    snapshot = out/"SOURCE_SNAPSHOT"
    snapshot.mkdir()
    for name,p in sources.items():
        copyfile(p,snapshot/name)
    write(out/"RUN.json",{**stamp(),"registration_sha256":sha(reg),"before_native":True})
    event(out,"frozen_integrated_registration_before_native",mothers=len(UNITS),cached_mothers=len(route_ids))


def locked(out):
    reg = out/"REGISTERED.json"
    p = read(reg)
    if (p["units"] != UNITS or p["namespace"] != NS or p["alpha"] != ALPHA or
            read(out/"RUN.json")["registration_sha256"] != sha(reg)):
        raise ValueError("Require original prospective registration.")
    for f,h in {**p["inputs"],**p["sources"]}.items():
        if sha(f) != h:
            raise ValueError("Frozen dependency changed: "+f)
    return p


def pool(out, p, task, jobs, label, ids, init=worker_initializer, initargs=()):
    rows=[]
    try:
        with ProcessPoolExecutor(p["workers"],initializer=init,initargs=initargs) as executor:
            for row in executor.map(task,jobs):
                rows.append(row)
    except Exception as error:
        write(out/f"{label}_FAILED.json",{**stamp(),"ordered_rows":len(rows),"error":repr(error)})
        raise
    actual = [r["unit"] if isinstance(r,dict) else r[0]["unit"] if isinstance(r[0],dict) else r[0] for r in rows]
    if actual != ids:
        raise ValueError("All original identities and order are required for "+label)
    event(out,label,mothers=len(rows))
    return rows


def cache(out,p):
    folder=out/"cache"
    folder.mkdir()
    start=out/"CACHE_STARTED.json"
    write(start,{**stamp(),"registration_sha256":sha(out/"REGISTERED.json")})
    jobs=[(folder,p["bank"],r,p["normalization"]) for r in p["bank"]["selected_records"]]
    rows=pool(out,p,cache_one,jobs,"complete_same_bank_original_routes",p["route_ids"])
    losses=np.stack([np_load(verify(Path(r["compact"]["path"]),r["compact"]))["losses"] for r in rows])
    if losses.shape!=(384,16,6,7,2) or sum(r["native_calls"] for r in rows)!=384*126:
        raise ValueError("The physically complete six-time and seven-count route cache is required.")
    arrays=np_save(out/"CACHE_LOSSES.npz",units=p["route_ids"],losses=losses)
    write(out/"CACHE_CLOSED.json",{**stamp(),"started_sha256":sha(start),"rows":rows,"arrays":arrays,
          "native_calls":sum(r["native_calls"] for r in rows)})


def acquire(out,p):
    closed=out/"CACHE_CLOSED.json"
    if not closed.is_file():
        raise ValueError("Close the offline same-bank cache before any validation acquisition.")
    folder=out/"held"
    folder.mkdir()
    jobs=[(folder,p["physics"],u) for u in UNITS]
    rows=pool(out,p,acquire_one,jobs,"acquired_new_original_prefix",UNITS)
    write(out/"ACQUIRED.json",{**stamp(),"registration_sha256":sha(out/"REGISTERED.json"),
          "cache_sha256":sha(closed),"held":rows,"native_calls":sum(r["native_calls"] for r in rows)})


def forecast_setup(p, cache_file):
    old=p["old_plan"]
    path_setup(old["bank_arrays"],old["old_model"],old["old_settings"])
    c=np_load(cache_file)
    np.testing.assert_array_equal(c["units"],p["route_ids"])
    FORECAST.update(plan=p,losses=c["losses"])


def forecast_one(job):
    i,r=job
    p=FORECAST["plan"]
    q=np_load(verify(Path(r["query"]["path"]),r["query"]))["measured"]
    with threadpool_limits(1):
        w=path_weights(q)
        result=plan_query(q,PATH_WORK["bank"]["clean"],PATH_WORK["old"]["noise"],p["bank_ids"],p["route_ids"],
             FORECAST["losses"],frozen_route_ids=p["route_ids"],population_controls=p["population_controls"],
             fixed_controls=p["fixed_controls"])
    digests={m:digest(x) for m,x in w.items()}
    if result["full_recovery_digest"]!=digests["ess1024"]:
        raise ValueError("Planning did not use the same complete ESS1024 distribution as recovery.")
    controls=np.stack([result["methods"][m]["controls"] for m in INVERSE_METHODS])
    info={"unit":r["unit"],"obs_sha256":r["query"]["sha256"],"digests":digests,
          "route_retained_mass":result["route_retained_recovery_mass"],
          "route_log_retained_mass":result["route_log_retained_recovery_mass"],
          "route_mother_ess":result["route_mother_ess"]}
    return info,q,controls,w["ess1024"].reshape(8192,16) if i in ANCHORS else None


def predict(out,p):
    closed=out/"CACHE_CLOSED.json"
    a=out/"ACQUIRED.json"
    record=read(a)
    if record["cache_sha256"]!=sha(closed) or [r["unit"] for r in record["held"]]!=UNITS:
        raise ValueError("Original cache and every new query are required.")
    cache_file=verify(out/"CACHE_LOSSES.npz",read(closed)["arrays"])
    rows=pool(out,p,forecast_one,enumerate(record["held"]),"completed_joint_original_forecasts",UNITS,
              forecast_setup,(p,cache_file))
    desc=out/"FORECAST_DESCRIPTORS.json"
    write(desc,{**stamp(),"rows":[d for d,_,_,_ in rows]})
    choices=np_save(out/"CHOICES.npz",units=UNITS,query=np.stack([q for _,q,_,_ in rows]),
                    choices=np.stack([c for _,_,c,_ in rows]))
    anchors=np_save(out/"ANCHORS.npz",units=[UNITS[i] for i in ANCHORS],
                    full_joint=np.stack([w for _,_,_,w in rows if w is not None]))
    write(out/"PREDICTIONS_FROZEN.json",{**stamp(),"registration_sha256":sha(out/"REGISTERED.json"),
          "acquisition_sha256":sha(a),"cache_sha256":sha(closed),"choices":choices,
          "descriptors":reference(desc),"anchors":anchors,"held_truth_opened":False,"native_calls":0})


def reveal(out,p):
    f,a=out/"PREDICTIONS_FROZEN.json",out/"ACQUIRED.json"
    lock=read(f)
    if lock["acquisition_sha256"]!=sha(a) or lock["held_truth_opened"]:
        raise ValueError("Require complete query-only predictions before native references.")
    choices=np_load(verify(out/"CHOICES.npz",lock["choices"]))["choices"]
    (out/"reference").mkdir()
    freeze=sha(f)
    write(out/"REVEAL_STARTED.json",{**stamp(),"freeze_sha256":freeze})
    jobs=[(out,r,p["physics"],c,freeze) for r,c in zip(read(a)["held"],choices,strict=True)]
    rows=pool(out,p,reference_one,jobs,"native_references_for_joint_actions",UNITS)
    calls=sum(r["native_calls"] for r in rows)
    if calls>p["simulation_reference_bound"]:
        raise ValueError("Exceeded original reference bounds.")
    write(out/"REFERENCES_CLOSED.json",{**stamp(),"freeze_sha256":freeze,"rows":rows,"native_calls":calls})


def score(out,p):
    f=out/"PREDICTIONS_FROZEN.json"
    rr=out/"REFERENCES_CLOSED.json"
    lock,closed=read(f),read(rr)
    if closed["freeze_sha256"]!=sha(f):
        raise ValueError("Original freeze must bind reference generation.")
    rows=read(out/"ACQUIRED.json")["held"]
    desc=read(verify(out/"FORECAST_DESCRIPTORS.json",lock["descriptors"]))["rows"]
    optical=[{"unit":r["unit"],"obs":r["query"],"raw":r["raw"]} for r in rows]
    old=p["old_plan"]
    scores=pool(out,p,path_score,zip(optical,desc,strict=True),"scored_exact_original_recovery",UNITS,
                path_setup,(old["bank_arrays"],old["old_model"],old["old_settings"]))
    replica=np.stack([s for _,s in scores])
    path_losses=replica.mean(axis=1)
    choices=np_load(verify(out/"CHOICES.npz",lock["choices"]))["choices"]
    losses,blocks=[],[]
    if [r["unit"] for r in closed["rows"]]!=UNITS:
        raise ValueError("All independent mother-level references are required.")
    for reference_row,control in zip(closed["rows"],choices,strict=True):
        folder=out/"reference"/str(reference_row["unit"])
        rec=read(verify(folder/"COMPLETE.json",reference_row["reference"]))
        data=np_load(verify(folder/"REFERENCE_VALUES.npz",rec["reference"]))
        j=data["mapping"]
        np.testing.assert_array_equal(data["controls"][j],control)
        y,b=original_losses(data["values"],p["normalization"])
        losses.append(y[:,j,np.arange(2)].mean(axis=0))
        blocks.append(b[:,j,np.arange(2)].mean(axis=0))
    auto=score_noninferiority(path_losses,PATH_METHODS,PATH_METRICS,PATH_METHODS[0],PATH_METHODS[1:],
                             primary_margin=.01,sensitivity_margins=(0.,.005,.02),alpha=ALPHA)
    inverse=score_noninferiority(losses,INVERSE_METHODS,GOALS,INVERSE_METHODS[0],INVERSE_METHODS[1:],
                                primary_margin=.05,sensitivity_margins=(0.,.01,.02,.03),alpha=ALPHA)
    saved=np_save(out/"SCORED_ARRAYS.npz",units=UNITS,losses=losses,blocks=blocks,replica=replica,
                  path_losses=path_losses,choices=choices)
    diagnostics={k:np.quantile([d[k] for d in desc],[0.,.1,.5,.9,1.])
                 for k in ("route_retained_mass","route_mother_ess")}
    write(out/"SCORED.json",{**stamp(),"registration_sha256":sha(out/"REGISTERED.json"),
          "freeze_sha256":sha(f),"reference_sha256":sha(rr),"arrays":saved,"path_ni":auto,"inverse_ni":inverse,
          "primary_gate":auto["primary"]["passed"] and inverse["primary"]["passed"],
          "mothers":len(UNITS),"cache_native":read(out/"CACHE_CLOSED.json")["native_calls"],"native_planning":0,
          "route_diagnostics_at_0_10_50_90_100_percentiles":diagnostics,
          "native_reference":closed["native_calls"],"native_acquisition":3*len(UNITS),"new_wet_measurements":0})
    event(out,"both_original_study_families_scored")


if __name__=="__main__":
    cli=ArgumentParser(description=__doc__)
    phases={"cache":cache,"acquire":acquire,"predict":predict,"reveal":reveal,"score":score}
    cli.add_argument("phase",choices=("register","run",*phases))
    cli.add_argument("out",type=Path)
    for field in ("bank","old_path_plan","atlas","inverse","population","review"):
        cli.add_argument("--"+field.replace("_","-"),type=Path)
    cli.add_argument("--workers",type=int,default=8)
    o=cli.parse_args()
    if o.phase=="register":
        fields=(o.bank,o.old_path_plan,o.atlas,o.inverse,o.population,o.review)
        if any(p is None for p in fields) or not 1<=o.workers:
            cli.error("Register the exact six input paths and a positive worker count.")
        register(o.out.resolve(),*(p.resolve(strict=True) for p in fields),o.workers)
    else:
        out=o.out.resolve(strict=True)
        p=locked(out)
        for name in phases if o.phase=="run" else (o.phase,):
            phases[name](out,p)
