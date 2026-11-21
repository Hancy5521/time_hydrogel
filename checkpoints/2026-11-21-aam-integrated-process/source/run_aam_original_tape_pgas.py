"""Freeze and score blind native AAm path forecasts with equal-work PG and PGAS."""
from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from shutil import copyfile
from time import perf_counter

import numpy as np
from scipy.stats import t
from threadpoolctl import threadpool_limits

from aam_formation_actual_rollout import packet_fingerprint
from aam_formation_cycle_driver import np_load, np_save, read, save_tree, sha, write
from aam_formation_path_cycle import event, load_tree, stamp
from aam_original_tape_pgas import fixtures, make_history, sweep
from audit_aam_original_tape_pgas_finite import run_fixture as finite_fixture
from package_aam_formation_local_cycle import source_closure
from run_aam_direct_energy_trajectory_law import atlas_receipts
from run_aam_mark_candidate_filter import verify
from run_aam_suffix_group_smoother import acquired_data, features
from run_aam_temporal_energy_law import all_losses, predict_variant, readout_inputs


ROOT = Path(__file__).resolve().parent
CLOSURE = ROOT / "audit_aam_original_tape_pgas.py"
PROTOCOL = ROOT / "AAM_ORIGINAL_TAPE_PGAS_PROTOCOL_ZH.md"
NS = "aam-native-full-suffix-pgas-blind-v1"
UNITS = list(range(2026136000, 2026136048))
KEY = 3631209858
WARM = 64
CHAINS = 4
SWEEPS = 3
ITER_CALLS = 93
WORK = WARM*3+2*CHAINS*SWEEPS*ITER_CALLS
METHODS = {"pgas":16,"pg":32}
ATLAS = ("kernel_cme","kernel","knn","empirical")
CONTROLS = ("pg","warm",*ATLAS)
GATES = ("early18", "hidden_delta32_6")


def register(out, origin, review):
    base, training, model, refs = atlas_receipts(origin)
    if set(UNITS).intersection(sum(base["units"].values(), [])):
        raise ValueError("Require new independent mother addresses.")
    sources = source_closure([Path(__file__), CLOSURE])
    out.mkdir(exist_ok=False)
    snap = out/"SOURCE_SNAPSHOT"
    snap.mkdir()
    for k, path in sources.items():
        copyfile(path, snap/k)
    settings = read(origin/"TRAINED_AND_SELECTED.json")["selected"]
    d = {**stamp(),"namespace":NS,"units":UNITS,"physics":base["physics"],"training_units":training,
         "atlas_model":str(model),"atlas_settings":settings,"atlas_methods":ATLAS,"workers":8,
         "method_particles":METHODS,"gates":GATES,"controls":CONTROLS,"fixed_roles_seed":KEY,
         "warm":WARM,"chains":CHAINS,"sweeps":SWEEPS,"calls_per_sweep":ITER_CALLS,
         "total_forecast_native":WORK*len(UNITS),"total_acquisition_native":3*len(UNITS),
         "truth_read_at_prediction":False,"old_held_reused":False,"new_wet":0,
         "inputs":{str(p):sha(p) for p in (*refs,review,PROTOCOL)},
         "sources":{str(p):sha(p) for p in sources.values()}}
    write(out/"REGISTERED.json",d)
    write(out/"RUN.json",{**stamp(),"registered_sha256":sha(out/"REGISTERED.json"),
                          "before_native_calls":True})
    event(out, "frozen_equal_native_cost", work_per_mother=WORK, actual_training=0)
    return d


def locked(out):
    reg = out/"REGISTERED.json"
    p = read(reg)
    if p["namespace"] != NS or p["units"] != UNITS or read(out/"RUN.json")["registered_sha256"] != sha(reg):
        raise ValueError("Require actual prospective registration.")
    for path,digest in {**p["inputs"], **p["sources"]}.items():
        if sha(path) != digest:
            raise ValueError("Frozen input changed: "+path)
    return p


def get(ref):
    return verify(Path(ref["path"]), ref)


def rng(unit, *address):
    return np.random.default_rng(np.random.SeedSequence([KEY, unit, *address]))


def gather(job):
    folder,cfg,unit = job
    with threadpool_limits(limits=1):
        y,raw = acquired_data(cfg,unit)
    return {"unit":unit, "query":np_save(folder/f"{unit}_QUERY.npz", measured=y),
            "outcome":save_tree(folder/f"{unit}_OUTCOME.npz",raw),"native_calls":3}


def lens(h, cfg):
    packets = h["packets"]
    return {**features(packets[:-1],packets[-1],cfg),
            "initial_id":packet_fingerprint(packets[0]),
            "path_id":[packet_fingerprint(p) for p in packets],
            "logl":h["logl"]}


def probabilities(logs):
    w = np.exp(logs-np.max(logs))
    return w/w.sum()


def blind(job):
    folder,cfg,view = job
    u = view["unit"]
    y = np_load(get(view["query"]))["measured"]
    start_time = perf_counter()
    with threadpool_limits(limits=1):
        bank = []
        calls = 0
        for j in range(WARM):
            h, receipt = make_history(cfg, y, rng(u,0,j))
            bank.append(h)
            calls += receipt["native_calls"]
        warm = [lens(h,cfg) for h in bank]
        w = probabilities(np.array([h["logl"].sum() for h in bank]))
        starts = rng(u,1).choice(WARM,size=CHAINS,p=w)
        methods = {}
        for method_index,(mode,n) in enumerate(METHODS.items(),start=2):
            result = {"mode":mode,"particles":n,"population":[],"ledger":[],"anchors":[]}
            for chain,index in enumerate(starts):
                h = bank[index]
                for step in range(SWEEPS):
                    h, ledger = sweep(cfg,y,h,n,rng(u,method_index,chain,step),mode=mode)
                    terminal = ledger.pop("terminal_histories")
                    result["population"].append([lens(q,cfg) for q in terminal])
                    result["ledger"].append(ledger)
                    calls += ledger["native_calls"]
                    if chain == 0 and step == 0:
                        result["anchors"] = [bank[index],h]
            methods[mode] = result
    if calls != WORK:
        raise ValueError("Unequal or incomplete original-native forecasting work.")
    raw = save_tree(folder/f"{u}_FORECAST.npz",
                    {"unit":u,"query":y,"starts":starts,"warm":warm,"warm_weights":w,
                     "methods":methods,"actual_native_work":calls})
    return {"unit":u, "forecast":raw,"native_calls":calls,"wall_seconds":perf_counter()-start_time}


def archive_each(fun, jobs, pool, out, action):
    records = []
    for row in pool.map(fun,jobs,chunksize=1):
        records.append(row)
        event(out,action,unit=row["unit"],native_calls=row["native_calls"])
    if [r["unit"] for r in records] != UNITS:
        raise ValueError("All preregistered mothers must be retained.")
    return records


def freeze(out, p):
    write(out/"EXACT_TARGET_FIXTURES.json",{**stamp(),"core":fixtures(),"independent":finite_fixture()})
    with ProcessPoolExecutor(max_workers=p["workers"]) as pool:
        a = archive_each(gather, [(out,p["physics"],u) for u in UNITS], pool,out,"truth_sealed")
        write(out/"ACQUIRED.json",{**stamp(),"rows":a,"native_calls":3*len(a)})
        view = [{"unit":r["unit"],"query":r["query"]} for r in a]
        f = archive_each(blind, [(out,p["physics"],r) for r in view],pool,out,"raw_predicted")
    y = np.stack([np_load(get(r["query"]))["measured"] for r in view])
    model = load_tree(p["atlas_model"])
    inputs = readout_inputs(model,y)
    with threadpool_limits(limits=1):
        w = {m:predict_variant(m,p["atlas_settings"].get(m),model,inputs)["weights"] for m in ATLAS}
    atlas = np_save(out/"ATLAS_WEIGHTS.npz", query=y, **w)
    write(out/"PREDICTIONS_FROZEN.json",{**stamp(),"forecast_rows":f,"units":UNITS,
          "registered_sha256":sha(out/"REGISTERED.json"),"atlas":atlas,
          "acquired_sha256":sha(out/"ACQUIRED.json"),"truth_opened":False})
    event(out,"all_predictions_frozen",native_calls=sum(r["native_calls"] for r in f))


def bank(rows):
    return {"local":np.stack([r["local"] for r in rows]),
            "global_early":np.stack([r["global_early"] for r in rows]),
            "new_edges":[r["new_edges"] for r in rows]}


def single_weights(obj):
    q = obj["population"]
    ledgers = obj["ledger"]
    if len(q) != CHAINS*SWEEPS or len(ledgers) != len(q):
        raise ValueError("Missing conditional SMC sweep.")
    return np.concatenate([z["terminal_weights"] for z in ledgers])/len(q), bank(sum(q,[]))


def paired(data):
    mean = float(np.mean(data))
    width = float(t.ppf(.975,len(data)-1)*np.std(data,ddof=1)/np.sqrt(len(data)))
    return [mean,mean-width,mean+width]


def score(out,p):
    preds = out/"PREDICTIONS_FROZEN.json"
    f = read(preds)
    a = read(out/"ACQUIRED.json")
    if (f["acquired_sha256"] != sha(out/"ACQUIRED.json") or f["units"] != UNITS
        or f["truth_opened"] is not False or f["registered_sha256"] != sha(out/"REGISTERED.json")):
        raise ValueError("Cannot score before verified freeze.")
    write(out/"SCORE_STARTED.json",{**stamp(),"predictions_sha256":sha(preds)})
    actual = []
    for row in a["rows"]:
        raw = load_tree(get(row["outcome"]))
        if raw["calls"] != {"attempted":3,"completed":3,"failed":0}:
            raise ValueError("Truth replay incomplete.")
        q = raw["packets"]
        actual.append(features(q[:-1],q[-1],p["physics"]))
    target = bank(actual)
    model = load_tree(p["atlas_model"])
    ww = np_load(get(f["atlas"]))
    scale = model["global_scale"]
    recipe = p["physics"]["recipe"]
    beads = int(recipe["aam"]) + int(recipe["mbaa"])
    losses = {m:all_losses(ww[m],model["bank"],target,scale,beads) for m in ATLAS}
    trial = []
    diag = {m:[] for m in METHODS}
    for record,x in zip(f["forecast_rows"],actual,strict=True):
        rec = load_tree(get(record["forecast"]))
        arms = {"warm":(rec["warm_weights"],bank(rec["warm"]))}
        for m,rs in rec["methods"].items():
            arms[m] = single_weights(rs)
            ids = [r["initial_id"] for s in rs["population"] for r in s]
            ledg = rs["ledger"]
            diag[m].append([len(set(ids)), len(set(tuple(s[b["selected_index"]]["path_id"])
                                                   for s,b in zip(rs["population"],ledg,strict=True))),
                            float(np.mean([1/np.square(b["terminal_weights"]).sum() for b in ledg])),
                            sum(e["reference_ancestor"] != rs["particles"]-1 for b in ledg for e in b["stages"])])
        trial.append({m:all_losses(w[None,:],candidate,bank([x]),scale,beads)
                      for m,(w,candidate) in arms.items()})
    for m in ("warm",*METHODS):
        losses[m] = {k:np.asarray([q[m][k][0] for q in trial]) for k in trial[0][m]}
    comparisons = {metric:{base:paired(losses[base][metric]-losses["pgas"][metric])
                           for base in CONTROLS} for metric in losses["pgas"]}
    passed = all(comparisons[k][base][1] > 0 for k in GATES for base in CONTROLS)
    ptr = save_tree(out/"SCORE_ARRAYS.npz", {"losses":losses,"diagnostics":diag,"units":UNITS})
    write(out/"SCORED.json",{**stamp(),"predictions_sha256":sha(preds),"new_simulated_mothers":len(UNITS),
            "primary":"early18","gate_metrics":GATES,"gate_controls":CONTROLS,"go_to_inverse":bool(passed),
            "means":{m:{k:float(v.mean()) for k,v in a.items()} for m,a in losses.items()},
            "paired_gains_over_controls":comparisons,"arrays":ptr,
            "mean_diagnostics":{m:np.mean(v,axis=0).tolist() for m,v in diag.items()},
            "diagnostic_columns":["distinct_terminal_initials","distinct_selected_paths","mean_terminal_ess",
                                  "retained_ancestor_changes_per_mother"],
            "forecast_native_calls":sum(r["native_calls"] for r in f["forecast_rows"]),
            "full_inverse_opened":False,"new_wet_measurements":0})
    event(out,"blind_truth_open_and_score_closed", native_calls=0)


if __name__ == "__main__":
    cli = ArgumentParser(description=__doc__)
    cb = cli.add_subparsers(dest="action", required=True)
    q = cb.add_parser("register")
    q.add_argument("out",type=Path)
    q.add_argument("atlas",type=Path)
    q.add_argument("review",type=Path)
    for name in ("predict","score"):
        cb.add_parser(name).add_argument("out",type=Path)
    args = cli.parse_args()
    if args.action == "register":
        register(args.out,args.atlas,args.review)
    elif args.action == "predict":
        freeze(args.out,locked(args.out))
    else:
        score(args.out,locked(args.out))
