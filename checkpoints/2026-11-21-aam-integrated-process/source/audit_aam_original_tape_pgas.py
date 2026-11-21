"""Read back frozen panel weights and native costs without re-running a blind sweep."""

from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.spatial.distance import cdist
from scipy.special import logsumexp
from threadpoolctl import threadpool_limits

from aam_formation_actual_rollout import packet_fingerprint
from aam_formation_cycle_driver import np_load, read, sha, write
from aam_formation_local_initial import build_packet, draw_initial, validate_initial
from aam_formation_path_cycle import load_tree, stamp
from audit_aam_assay_guided_sir import _camera
from audit_aam_camera_gaussian_tape_proposal import check_packet
from audit_aam_endpoint_tape_move import (
    ATTEMPTS, CELLS, artifact, close, generating_state, intervals, intervals95, require, rng,
)
from audit_aam_suffix_group_smoother import _features
from audit_aam_temporal_energy_law import independently_predict, solve_dev_projection


SCHEMA = "aam-original-independent-tape-pgas-v1"
NS = "aam-native-full-suffix-pgas-blind-v1"
UNITS = tuple(range(2026136000, 2026136048))
KEY, WARM, CHAINS, SWEEPS, CALLS = 3631209858, 64, 4, 3, 93
METHODS = {"pgas":16,"pg":32}
ATLAS = ("kernel_cme","kernel","knn","empirical")
CONTROLS = ("pg","warm",*ATLAS)
GATES = ("early18", "hidden_delta32_6")
METRICS = ("hidden_t3_6","hidden_delta32_6","early18","hidden_t2_6","global24","new_edge_brier")
WORK = WARM*3+2*CHAINS*SWEEPS*CALLS


def normalized(log):
    values = np.asarray(log)
    require(values.ndim==1 and np.isfinite(values).all() and len(values)>0,
            "Original Gibbs weights must be finite one-dimensional likelihoods.")
    return np.exp(values-logsumexp(values))


def categorical(probabilities, uniforms):
    cumulative = np.cumsum(probabilities)
    cumulative[-1] = 1.
    values = np.asarray(uniforms)
    require(np.isfinite(values).all() and np.all((0<=values)&(values<1)),
            "Categorical Gibbs draws need saved original uniforms.")
    return np.sum(cumulative<=values[...,None],axis=-1)


def log_camera(physical, inferred, observed):
    return -.5*np.sum(np.square((np.asarray(inferred)-observed)/physical["noise"]),axis=-1)


def sweep_ledger(saved, mode, particles, reference):
    n = particles
    initial = np.asarray(saved["initial_log_weights"])
    require(saved["schema"]==SCHEMA and saved["mode"]==mode and saved["particles"]==n
            and initial.shape==(n,) and len(saved["stages"])==2, "Wrong conditional-particle scope.")
    close(initial[-1],reference[0])
    previous = initial
    for camera, row in enumerate(saved["stages"],start=1):
        log = np.asarray(row["incremental_log_weights"])
        require(row["camera"]==camera+1 and log.shape==(n,)
                and np.asarray(row["ordinary_uniforms"]).shape==(n-1,)
                and np.asarray(row["ordinary_ancestors"]).shape==(n-1,),
                "Ordinary multinomial resampling lost a particle or camera.")
        w = normalized(previous)
        close(row["ordinary_probabilities"],w)
        np.testing.assert_array_equal(row["ordinary_ancestors"],categorical(w,row["ordinary_uniforms"]))
        close(row["ordinary_ancestor_ess"],1./(w @ w))
        held = row["reference_ancestor"]
        require(0 <= held < n, "Invalid retained ancestor.")
        if mode=="pgas":
            suffix = np.asarray(row["replayed_future_logl"])
            require(suffix.shape==(n,3-camera) and np.isfinite(suffix).all()
                    and np.all(suffix<=2.e-13) and row["reference_first_replay_reused"] is True,
                    "Complete likelihood from each original future tape must be used.")
            full = previous+suffix.sum(axis=-1)
            close(row["ancestor_log_weight"],full)
            bridge = normalized(full)
            close(row["reference_probabilities"],bridge)
            close(row["reference_ancestor_ess"],1./(bridge @ bridge))
            np.testing.assert_array_equal(categorical(bridge,row["reference_uniform"]),[held])
            close(log[-1],suffix[held,0])
        else:
            require(held==n-1 and row["reference_first_replay_reused"] is False and
                    all(row[k] is None for k in ("replayed_future_logl","ancestor_log_weight",
                                                "reference_uniform","reference_probabilities","reference_ancestor_ess")),
                    "Plain PG must keep the reference genealogy without a guided ancestor step.")
            close(log[-1],reference[camera])
        previous = log
    terminal = normalized(previous)
    close(saved["selection_probabilities"],terminal)
    close(saved["terminal_weights"],terminal)
    choice = saved["selected_index"]
    np.testing.assert_array_equal(categorical(terminal,saved["selection_uniform"]),[choice])
    expected = 3*(n-1)+(3*n if mode=="pgas" else 0)
    require(saved["native_calls"]==expected
            and saved["calls"]=={"attempted":expected,"completed":expected,"failed":0},
            "Original PG costs include all suffix replays and no duplicate selected first interval.")
    return terminal, choice, expected


def anchor_clock(cfg, camera, parent, child, tape, ledger):
    calendar = intervals(cfg,camera)
    st0, st1 = parent[0],child[0]
    require(st0.position[0]==calendar[0] and st1.position[0]==calendar[-1]
            and ledger["native_calls"]==1 and ledger["log_ratio"]==0.
            and not ledger["tape"]["forced_mark_consumed"],"Archived anchor used a different native law.")
    wait, mark, normals = tape["waiting"],tape["mark"],tape["brownian"]
    shape = len(calendar)-1,ATTEMPTS
    require(wait.shape==shape and mark.shape==shape and np.all(np.isfinite(wait)) and np.all(wait>=0.)
            and np.all((mark>=0)&(mark<CELLS))
            and np.all(np.isfinite(normals)) and normals.shape==(shape[0],2,len(st0.points),3),
            "Archived anchor lost a role-addressed original innovation.")
    count_ev, count_edge = st0.counts[:2]
    np.testing.assert_array_equal(st1.events[:count_ev],st0.events[:count_ev])
    np.testing.assert_array_equal(st1.edges[:count_edge],st0.edges[:count_edge])
    np.testing.assert_array_equal(st1.lifts[:count_edge],st0.lifts[:count_edge])
    events = iter(st1.events[count_ev:st1.counts[0],0])
    rows = iter(ledger["spans"])
    used = []
    for slab,(left,right) in enumerate(zip(calendar[:-1],calendar[1:],strict=True)):
        now = left
        for attempt in range(ATTEMPTS):
            span = next(rows)
            rate = span["target_A"]
            require((span["slab"],span["attempt"],span["begin"])==(slab,attempt,now)
                    and np.isfinite(rate) and rate>=0.,"Stored anchor omitted or changed a native candidate span.")
            if rate:
                require(span["exponential"]==wait[slab,attempt],"Selected waiting innovation changed.")
                arrival = now+wait[slab,attempt]/rate
            else:
                require(span["exponential"] is None,"Zero hazard consumed a wait.")
                arrival = np.inf
            require((arrival>=right)==span["censored"],"Wrong original stopping event.")
            if span["censored"]:
                require(span["end"]==right,"Frozen clock crossed a camera.")
                used.append(attempt+int(rate>0.))
                break
            require(span["mark_tilt"] is None and span["chosen"]>=0
                    and span["uniform"]==mark[slab,attempt]/CELLS and span["end"]==next(events),
                    "Anchor trajectory changed its original mark or event time.")
            close(span["end"],arrival)
            now = span["end"]
        else:
            raise ValueError("A saved span exceeded the available innovations.")
    require(next(rows,None) is None and next(events,None) is None,"Anchor excluded archived native history.")
    np.testing.assert_array_equal(used,ledger["tape"]["waits_per_slab"])


def anchor_history(history, cfg, query):
    raw = history["packets"]
    require(history["schema"]==SCHEMA and len(raw)==4
            and len(history["intervals"])==len(history["tapes"])==3,
            "A claimed legal anchor must contain all original tapes and packets.")
    validate_initial(cfg,history["initial"])
    np.testing.assert_array_equal(raw[0][0].points,history["initial"]["points"])
    require(int(raw[0][0].counts[3])==history["initial"]["I"],"Wrong selected initial random stock.")
    for k,packet in enumerate(raw):
        check_packet(packet,cfg,cfg["prefix_clock"][k])
    for k,(table,ledger) in enumerate(zip(history["tapes"],history["intervals"],strict=True)):
        anchor_clock(cfg,k,raw[k],raw[k+1],table,ledger)
    clean = np.stack([_camera(packet) for packet in raw[1:]])
    close(clean,history["clean"])
    close(log_camera(cfg,clean,query),history["logl"])
    return _features(raw,cfg)


def energy_score(atoms, weight, outcome):
    r = np.asarray(atoms).reshape(len(weight),-1)
    y = np.asarray(outcome).reshape(1,-1)
    d = cdist(r,r)
    cross = cdist(r,y)[:,0]
    return float((cross @ weight-.5*weight @ d @ weight)/np.sqrt(r.shape[1]))


def distribution(rec, weight):
    close(np.sum(weight),1.)
    require(np.asarray(weight).shape==(len(rec),) and np.isfinite(weight).all() and np.min(weight)>=0.,
            "Frozen law lacks a complete normalized probability vector.")
    return {key:[x[key] for x in rec] for key in ("local","global_early","new_edges")},np.asarray(weight)


def loss(bank, weights, truth, scale, n):
    local, hidden = np.asarray(bank["local"]),np.asarray(truth["local"])
    result = {
        "hidden_t3_6":energy_score(local[:,2],weights,hidden[2]),
        "hidden_delta32_6":energy_score(local[:,2]-local[:,1],weights,hidden[2]-hidden[1]),
        "early18":energy_score(local,weights,hidden),
        "hidden_t2_6":energy_score(local[:,1],weights,hidden[1]),
        "global24":energy_score(np.asarray(bank["global_early"])/scale,weights,np.asarray(truth["global_early"])/scale),
    }
    edges = np.zeros(n*n)
    for w,ids in zip(weights,bank["new_edges"],strict=True):
        require(len(set(ids))==len(ids) and all(0<=k<n*n and k//n<k%n for k in ids),
                "Edge Brier support must contain actual unique undirected pairs.")
        edges[np.asarray(ids,dtype=int)] += w
    edges[np.asarray(truth["new_edges"],dtype=int)] -= 1.
    return {**result,"new_edge_brier":float(edges @ edges)}


def validity(row):
    path = row["path_id"]
    require(len(path)==4 and all(isinstance(x,str) and len(x)==64 for x in path)
            and path[0]==row["initial_id"] and np.asarray(row["logl"]).shape==(3,)
            and np.isfinite(row["logl"]).all() and np.max(row["logl"])<=2.e-13
            and np.asarray(row["local"]).shape==(3,2,3) and np.asarray(row["global_early"]).shape==(3,8)
            and np.isfinite(row["local"]).all() and np.isfinite(row["global_early"]).all()
            and np.min(row["local"])>=0. and np.max(row["local"])<=1.+2.e-14,
            "A frozen hidden trajectory member is incomplete or unphysical.")


def compare_anchor(history, row, cfg, measured):
    for key,value in anchor_history(history,cfg,measured).items():
        close(row[key],value)
    close(row["logl"],history["logl"])
    require(row["path_id"]==[packet_fingerprint(q) for q in history["packets"]],
            "Archived full path does not match the claimed selected native lineage.")


def verify_warm_tape(history, cfg, unit, draw):
    random = rng(KEY,unit,0,draw)
    orig = draw_initial(cfg,random)
    for key,want in orig.items():
        np.testing.assert_array_equal(history["initial"][key],want)
    for k,tape in enumerate(history["tapes"]):
        shape = (len(intervals(cfg,k))-1,ATTEMPTS)
        np.testing.assert_array_equal(tape["brownian"],random.standard_normal((shape[0],2,len(orig["points"]),3)))
        np.testing.assert_array_equal(tape["waiting"],random.exponential(size=shape))
        np.testing.assert_array_equal(tape["mark"],random.integers(CELLS,size=shape,dtype=np.int64))


def terminal_law(pop, ledger, ref, mode):
    two, three = ledger["stages"]
    ancestor2 = [*two["ordinary_ancestors"],two["reference_ancestor"]]
    ancestor3 = [*three["ordinary_ancestors"],three["reference_ancestor"]]
    n = len(pop)
    seen = ({},{})
    for j,p in enumerate(pop):
        validity(p)
        idx2 = ancestor3[j]
        idx1 = ancestor2[idx2]
        for memory,k,size in zip(seen,(idx1,idx2),(2,3),strict=True):
            require(memory.setdefault(k,p["path_id"][:size])==p["path_id"][:size],
                    "Terminal particles copied inconsistent ancestors.")
        if idx1==n-1:
            require(p["path_id"][:2]==ref["path_id"][:2],"Retained first prefix has changed.")
        if mode=="pg" and idx2==n-1:
            require(p["path_id"][:3]==ref["path_id"][:3],"PG lost its unchanged middle reference prefix.")
        close(p["logl"], [ledger["initial_log_weights"][idx1],
                         two["incremental_log_weights"][idx2],
                         three["incremental_log_weights"][j]])
    if mode=="pg":
        require(pop[-1]["path_id"]==ref["path_id"],"PG did not retain the complete reference genealogy.")


def mother_forecast(actual, cfg, measured, meta, row):
    require(actual["unit"]==meta["unit"] and actual["actual_native_work"]==meta["native_calls"]==WORK
            and np.isfinite(meta["wall_seconds"]) and meta["wall_seconds"]>0.
            and set(actual)=={"unit","query","starts","warm","warm_weights","methods","actual_native_work"},
            "Forecast contained undeclared query information or incorrect total work.")
    np.testing.assert_array_equal(actual["query"],measured)
    unit = actual["unit"]
    warm = actual["warm"]
    require(len(warm)==WARM,"An incomplete warm bank was retained.")
    for j,p in enumerate(warm):
        validity(p)
        expected_initial = packet_fingerprint(build_packet(cfg,draw_initial(cfg,rng(KEY,unit,0,j))))
        require(p["initial_id"]==expected_initial and p["initial_id"]!=row["true_initial_id"],
                "A warm history starts from an undeclared random address or an outcome.")
    weight = normalized(np.sum([w["logl"] for w in warm],axis=1))
    close(weight,actual["warm_weights"])
    starts = actual["starts"]
    np.testing.assert_array_equal(starts,rng(KEY,unit,1).choice(WARM,size=CHAINS,p=weight))
    distributions = {"warm":distribution(warm,weight)}
    diagnostics = {}
    require(actual["methods"].keys()==METHODS.keys(),"Changed the equal-cost comparison arms.")
    summed = WARM*3
    for mode,n in METHODS.items():
        saved = actual["methods"][mode]
        pop, ledgers = saved["population"],saved["ledger"]
        require(saved["mode"]==mode and saved["particles"]==n
                and len(pop)==len(ledgers)==CHAINS*SWEEPS and len(saved["anchors"])==2,
                "A conditional Gibbs arm has the wrong particles, sweeps, or raw anchors.")
        vectors, paths, retained_changes = [],[],0
        for i,(samples,ledger) in enumerate(zip(pop,ledgers,strict=True)):
            if i%SWEEPS==0:
                ref = warm[int(starts[i//SWEEPS])]
            require(len(samples)==n,"Missing terminal particles.")
            terminal, choice, calls = sweep_ledger(ledger,mode,n,ref["logl"])
            terminal_law(samples,ledger,ref,mode)
            summed += calls
            vectors.append(terminal)
            retained_changes += sum(r["reference_ancestor"]!=n-1 for r in ledger["stages"])
            selected = samples[choice]
            if i==0:
                first,second = saved["anchors"]
                compare_anchor(first,ref,cfg,measured)
                verify_warm_tape(first,cfg,unit,int(starts[0]))
                compare_anchor(second,selected,cfg,measured)
                r2,r3 = ledger["stages"]
                ancestors3 = [*r3["ordinary_ancestors"],r3["reference_ancestor"]]
                for camera,belongs in enumerate((False,ancestors3[choice]==n-1,choice==n-1)):
                    if belongs:
                        for key,tape in first["tapes"][camera].items():
                            np.testing.assert_array_equal(second["tapes"][camera][key],tape)
            require(all(q["initial_id"]!=row["true_initial_id"] for q in samples),
                    "Conditional particle Gibbs used the hidden true initial.")
            paths.append(tuple(selected["path_id"]))
            ref = selected
        flattened = sum(pop,[])
        distributions[mode] = distribution(flattened,np.concatenate(vectors)/len(pop))
        diagnostics[mode] = [len({r["initial_id"] for r in flattened}),len(set(paths)),
                             np.mean([1./(w @ w) for w in vectors]),retained_changes]
    require(summed==WORK,"Frozen warm and suffix calls do not reproduce actual native accounting.")
    return distributions,diagnostics


def audit_atlas(out, p, frozen, y):
    source = Path(p["atlas_model"])
    origin = source.parent
    record = read(origin/"REGISTERED.json")
    selections = read(origin/"TRAINED_AND_SELECTED.json")
    independent = read(origin/"INDEPENDENT_AUDIT.json")
    old_freeze = origin/"PREDICTIONS_FROZEN.json"
    require(record["physics"]==p["physics"] and p["training_units"]==record["units"]["train"]==selections["train_ids"]
            and len(set(p["training_units"]))==512 and p["atlas_settings"]==selections["selected"]
            and not set(UNITS).intersection(sum(record["units"].values(),[]))
            and independent["passed"] is True and independent["registered_sha256"]==sha(origin/"REGISTERED.json")
            and independent["forecast_sha256"]==sha(old_freeze),
            "The main strong baseline is not the previously audited 512-path development-frozen library.")
    model = load_tree(source)
    validate,_ = independently_predict(model,y)
    scored = np_load(artifact(frozen["atlas"],out/"ATLAS_WEIGHTS.npz"))
    require(set(scored)=={"query",*ATLAS},"Strong baseline changed after new queries.")
    np.testing.assert_array_equal(scored["query"],y)
    for key in ATLAS:
        kind, value = validate(key,p["atlas_settings"].get(key))
        if kind=="target":
            value = solve_dev_projection(model["basis"]["training_coordinates"],value,model["prior"],model["entropy"])
        close(scored[key],value,atol=2.e-10)
        close(scored[key].sum(axis=1),np.ones(len(y)))
        require(np.isfinite(scored[key]).all() and np.min(scored[key])>=0.,"Invalid atlas posterior probabilities.")
    return model,scored


def audit(out):
    names = ("REGISTERED","RUN","EXACT_TARGET_FIXTURES","ACQUIRED","PREDICTIONS_FROZEN","SCORE_STARTED","SCORED")
    files = {name:out/f"{name}.json" for name in names}
    d = {name:read(path) for name,path in files.items()}
    times = [datetime.fromisoformat(x["actual_utc"]) for x in d.values()]
    require(all(x<z for x,z in zip(times[:-1],times[1:],strict=True)) and
            all(x==datetime.fromisoformat(y["asia_shanghai"]) for x,y in zip(times,d.values(),strict=True)),
            "Prospective registration, prediction freeze, or score chronology was violated.")
    p,run,a,f,s = (d[k] for k in ("REGISTERED","RUN","ACQUIRED","PREDICTIONS_FROZEN","SCORED"))
    reg, pred = sha(files["REGISTERED"]),sha(files["PREDICTIONS_FROZEN"])
    require(p["namespace"]==NS and tuple(p["units"])==tuple(f["units"])==UNITS
            and p["method_particles"]==METHODS and tuple(p["atlas_methods"])==ATLAS
            and tuple(p["gates"])==tuple(s["gate_metrics"])==GATES
            and tuple(p["controls"])==tuple(s["gate_controls"])==CONTROLS
            and p["fixed_roles_seed"]==KEY and (p["warm"],p["chains"],p["sweeps"],p["calls_per_sweep"])==(WARM,CHAINS,SWEEPS,CALLS)
            and p["new_wet"]==s["new_wet_measurements"]==0 and s["full_inverse_opened"] is False
            and s["primary"]=="early18" and s["new_simulated_mothers"]==len(UNITS)
            and p["old_held_reused"] is p["truth_read_at_prediction"] is f["truth_opened"] is False
            and run["before_native_calls"] is True
            and reg==run["registered_sha256"]==f["registered_sha256"]
            and pred==d["SCORE_STARTED"]["predictions_sha256"]==s["predictions_sha256"]
            and f["acquired_sha256"]==sha(files["ACQUIRED"])
            and a["native_calls"]==p["total_acquisition_native"]==3*len(UNITS)
            and p["total_forecast_native"]==s["forecast_native_calls"]==WORK*len(UNITS),
            "Frozen blind split, score target, or equal-cost contract changed.")
    for name,digest in p["inputs"].items():
        require(sha(name)==digest,"A preregistered input changed: "+name)
    for name,digest in p["sources"].items():
        require(sha(out/"SOURCE_SNAPSHOT"/Path(name).name)==digest,"A preregistered source changed: "+name)
    fixture = d["EXACT_TARGET_FIXTURES"]["independent"]
    require(fixture["passed"] is True and fixture["native_calls"]==0 and fixture["visible_state_is_not_markov"] is True
            and fixture["results"]["full_suffix_pgas"]["preserves_exact_target"] is True
            and fixture["results"]["conditional_pg"]["preserves_exact_target"] is True
            and fixture["results"]["one_step_ancestor"]["stationarity_tv"]>0.,
            "The independent exact stationarity and one-step negative control did not run.")
    rows, forecast = a["rows"],f["forecast_rows"]
    require(tuple(r["unit"] for r in rows)==tuple(r["unit"] for r in forecast)==UNITS,
            "Failed or missing mothers cannot be replaced.")
    q = []
    for row in rows:
        data = np_load(artifact(row["query"],out/f'{row["unit"]}_QUERY.npz'))
        require(set(data)=={"measured"} and data["measured"].shape==(3,4) and np.isfinite(data["measured"]).all(),
                "Blind view must be original three cameras only.")
        q.append(data["measured"])
    atlas, w = audit_atlas(out,p,f,np.stack(q))
    cfg, scale = p["physics"],atlas["global_scale"]
    n = int(cfg["beads"])
    losses = {name:{metric:[] for metric in METRICS} for name in (*METHODS,"warm",*ATLAS)}
    diagnostics = {name:[] for name in METHODS}
    for i,(a_row,f_row,query) in enumerate(zip(rows,forecast,q,strict=True)):
        unit = a_row["unit"]
        raw = load_tree(artifact(a_row["outcome"],out/f"{unit}_OUTCOME.npz"))
        require(a_row["native_calls"]==3,"Truth is a single independently addressed path.")
        generating_state(cfg,unit,query,raw)
        truth = _features(raw["packets"],cfg)
        full = load_tree(artifact(f_row["forecast"],out/f"{unit}_FORECAST.npz"))
        distributions, batch = mother_forecast(full,cfg,query,f_row,
                                              {"true_initial_id":packet_fingerprint(raw["packets"][0])})
        distributions.update({name:(atlas["bank"],w[name][i]) for name in ATLAS})
        for name,(bank,weight) in distributions.items():
            for metric,value in loss(bank,weight,truth,scale,n).items():
                losses[name][metric].append(value)
        for name,v in batch.items():
            diagnostics[name].append(v)
    score_arrays = load_tree(artifact(s["arrays"],out/"SCORE_ARRAYS.npz"))
    np.testing.assert_array_equal(score_arrays["units"],UNITS)
    require(score_arrays["losses"].keys()==losses.keys() and score_arrays["diagnostics"].keys()==diagnostics.keys()
            and s["diagnostic_columns"]==["distinct_terminal_initials","distinct_selected_paths","mean_terminal_ess",
                                          "retained_ancestor_changes_per_mother"], "Score dropped a recorded arm or diagnostic.")
    for name,r in losses.items():
        for metric,vals in r.items():
            close(score_arrays["losses"][name][metric],vals)
            close(s["means"][name][metric],np.mean(vals))
    for name,values in diagnostics.items():
        close(score_arrays["diagnostics"][name],values)
        close(s["mean_diagnostics"][name],np.mean(values,axis=0))
    passed = True
    for metric in METRICS:
        for base in CONTROLS:
            differences = np.asarray(losses[base][metric])-losses["pgas"][metric]
            paired = intervals95(differences)
            close(s["paired_gains_over_controls"][metric][base],paired)
            if metric in GATES:
                passed &= bool(paired[1]>0.)
    require(s["go_to_inverse"] is passed,"Registered paired strong-baseline gate did not reproduce.")
    return {**stamp(),"passed":True,"registered_sha256":reg,"forecast_sha256":pred,
            "scored_sha256":sha(files["SCORED"]),"mother_units":len(UNITS),"atlas_mothers":512,
            "native_forecast_calls_audited":WORK*len(UNITS),"acquisition_calls_audited":3*len(UNITS),
            "sweeps_audited":2*CHAINS*SWEEPS*len(UNITS),"raw_history_anchors_audited":4*len(UNITS),
            "new_native_calls":0,"new_fit":0,"new_wet_measurements":0,"gate_for_inverse":passed,
            "same_warm_ancestry_and_costs_verified":True,"camera_only_forecast_view_verified":True,
            "scope":"Blind simulator component only. Exact archived resampling, native legality anchors and paired proper scores. No native reintegration."}


def fixtures():
    a = np.array([-.1,-2.])
    p = normalized(a)
    np.testing.assert_array_equal(categorical(p,[0.,p[0],.99]),[0,1,1])
    close(energy_score([[0.],[4.]],np.array([.25,.75]),[0.]),2.25)
    return "PASS exact-cost audit primitives on independent pure fixtures"


if __name__=="__main__":
    cli = ArgumentParser(description=__doc__)
    action = cli.add_mutually_exclusive_group(required=True)
    action.add_argument("--fixtures",action="store_true")
    action.add_argument("--out",type=Path)
    arg = cli.parse_args()
    with threadpool_limits(limits=1):
        if arg.fixtures:
            print(fixtures())
        else:
            out = arg.out.resolve(strict=True)
            write(out/"INDEPENDENT_AUDIT.json",audit(out))
            print("PASS independent blind PGAS audit",out)
