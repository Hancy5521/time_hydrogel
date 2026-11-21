"""Independent readback for a shared finite translation of original native tapes."""

from argparse import ArgumentParser
from datetime import datetime
from fractions import Fraction
from functools import lru_cache
from itertools import product
import json
from math import prod
from pathlib import Path

import numpy as np
from scipy.spatial.distance import cdist
from scipy.special import logsumexp
from threadpoolctl import threadpool_limits

from aam_formation_actual_rollout import packet_fingerprint
from aam_formation_cycle_driver import np_load, read, sha, write
from aam_formation_local_initial import build_packet, draw_initial
from aam_formation_path_cycle import load_tree, stamp
from audit_aam_assay_guided_sir import _camera
from audit_aam_camera_information_sandwich import independent_apertures
from audit_aam_endpoint_tape_move import (
    artifact, close, generating_state, intervals95, require, rng,
)
from audit_aam_original_tape_pgas import (
    SCHEMA as BASE_SCHEMA,
    ATLAS, CHAINS, GATES, SWEEPS, WARM,
    anchor_history,
    audit_atlas,
    categorical,
    normalized,
    sweep_ledger,
)
from audit_aam_original_tape_pgas_finite import BITS, HISTORIES, HALF, likelihood
from audit_aam_suffix_group_smoother import _features


SCHEMA = "aam-original-tape-shared-phase-pgas-v1"
PHASES = 16
STUDY = "aam-finite-y-group-marginal-pgas-blind-v1"
UNITS = tuple(range(2026137000,2026137048))
KEY = 143508222
MODES = {"pg":32,"pgas":16,"marginal":16}
CONTROLS = ("pg","pgas","post_phase",*ATLAS)
METRICS = ("early18","hidden_delta32_6","hidden_t3_6","hidden_t2_6","global24")
WORK = 3*WARM+len(MODES)*CHAINS*SWEEPS*93


def shifted_observables(packet, phases, readout):
    """Calculate through explicit copied wrap and the independent scalar aperture."""
    state, *params = packet
    box = params[1][2]
    values = []
    for offset in phases:
        points = state.points.copy()
        points[:,1] = ((points[:,1]+offset) % box)
        virtual = (state._replace(points=points), *params)
        values.append(readout(virtual))
    return np.asarray(values)


def phase_log_masses(logs):
    """Prefix cumulative phase scores and marginal likelihoods with one origin."""
    q = np.asarray(logs)
    require(q.shape[-2:]==(3,PHASES) and np.isfinite(q).all()
            and np.max(q)<=2.e-13,"A valid fixed Gaussian product needs all three frames, all phases.")
    cumulative = np.add.accumulate(q,axis=-2)
    marginal = logsumexp(cumulative,axis=-1)-np.log(PHASES)
    return cumulative,marginal


def anchor_phase(history,cfg,query):
    """Independently reconstruct every camera and hidden location at stored anchors."""
    require(history["schema"]==SCHEMA,"Anchor omits the joint recurring latent phase.")
    readouts = anchor_history({**history,"schema":BASE_SCHEMA},cfg,query)
    phases = np.arange(PHASES)*cfg["motion"]["box"]/PHASES
    np.testing.assert_array_equal(history["phase_offsets"],phases)
    packets = history["packets"][1:]
    clean = np.stack([shifted_observables(p,phases,_camera) for p in packets])
    close(history["clean_orbit"],clean)
    logs = -.5*np.square((clean-np.asarray(query)[:,None,:])/cfg["noise"]).sum(axis=-1)
    cum,marg = phase_log_masses(logs)
    close(history["phase_logl"],logs)
    close(history["cumulative_phase_logl"],cum)
    close(history["marginal_logl"],marg)
    w = normalized(cum[-1])
    close(history["phase_weights"],w)
    grid = np.stack([shifted_observables(p,phases,independent_apertures) for p in packets],axis=1)
    close(grid[0],readouts["local"])
    return {**readouts,"local_orbit":grid},logs,marg,w


def phase_sweep(saved,mode,particles,ref_marginal,terminal_phase_logl):
    n = particles
    prev = np.asarray(saved["initial_log_weights"])
    selected_marginal = np.asarray(ref_marginal)
    require(saved["schema"]==SCHEMA and saved["mode"]==mode and saved["particles"]==n
            and saved["phase_count"]==PHASES and prev.shape==(n,)
            and selected_marginal.shape==(3,) and len(saved["stages"])==2,
            "A different conditional particle or phase study was supplied.")
    close(prev[-1],selected_marginal[0])
    prefix = prev.copy()
    ancestor_maps = []
    for camera,r in enumerate(saved["stages"],start=1):
        inc = np.asarray(r["incremental_log_weights"])
        require(r["camera"]==camera+1 and inc.shape==(n,)
                and np.asarray(r["ordinary_uniforms"]).shape==(n-1,)
                and np.asarray(r["ordinary_ancestors"]).shape==(n-1,),
                "Fixed-size native phase update missing a stage or member.")
        ordinary = normalized(prev)
        close(r["ordinary_probabilities"],ordinary)
        np.testing.assert_array_equal(r["ordinary_ancestors"],categorical(ordinary,r["ordinary_uniforms"]))
        close(r["ordinary_ancestor_ess"],1./(ordinary @ ordinary))
        close(r["prefix_marginal_logl"],prefix)
        j = int(r["reference_ancestor"])
        require(0<=j<n,"Retained prefix lies outside the registered population.")
        if mode=="pgas":
            graft = np.asarray(r["graft_phase_logl"])
            require(graft.shape==(n,PHASES) and np.isfinite(graft).all()
                    and np.max(graft)<=2.e-13 and r["reference_first_replay_reused"] is True,
                    "Every ancestor needs full replay including the one common phase.")
            graft_marginal = logsumexp(graft,axis=-1)-np.log(PHASES)
            close(r["graft_marginal_logl"],graft_marginal)
            ancestor = prev+graft_marginal-prefix
            close(r["ancestor_log_weight"],ancestor)
            p = normalized(ancestor)
            close(r["reference_probabilities"],p)
            close(r["reference_ancestor_ess"],1./(p @ p))
            np.testing.assert_array_equal(categorical(p,r["reference_uniform"]),[j])
            if camera==2:
                close(inc[-1],graft_marginal[j]-prefix[j])
        else:
            require(j==n-1 and r["reference_first_replay_reused"] is False and
                    all(r[k] is None for k in ("ancestor_log_weight","graft_phase_logl","graft_marginal_logl",
                                              "reference_uniform","reference_probabilities","reference_ancestor_ess")),
                    "Plain phase-marginal PG retained random replacement ancestors.")
            close(inc[-1],selected_marginal[camera]-selected_marginal[camera-1])
        gather = np.r_[r["ordinary_ancestors"],j]
        ancestor_maps.append(gather)
        prefix = prefix[gather]+inc
        prev = inc
    terminal = normalized(prev)
    close(saved["selection_probabilities"],terminal)
    close(saved["terminal_weights"],terminal)
    choice = int(saved["selected_index"])
    np.testing.assert_array_equal(categorical(terminal,saved["selection_uniform"]),[choice])
    cumulative, marginal = phase_log_masses(terminal_phase_logl)
    at2 = ancestor_maps[1]
    at1 = ancestor_maps[0][at2]
    expected_marginal = np.column_stack((saved["initial_log_weights"][at1],
                        saved["stages"][0]["incremental_log_weights"][at2]+saved["initial_log_weights"][at1],prefix))
    close(marginal,expected_marginal)
    if mode=="pgas":
        selected_parent = int(saved["stages"][-1]["reference_ancestor"])
        close(cumulative[-1,-1],saved["stages"][-1]["graft_phase_logl"][selected_parent])
    phase = np.stack([normalized(x) for x in cumulative[:,-1]])
    close(saved["terminal_phase_weights"],phase)
    close(saved["selected_phase_probabilities"],phase[choice])
    np.testing.assert_array_equal(categorical(phase[choice],saved["phase_uniform"]),[saved["selected_phase_index"]])
    work = 3*(n-1)+(3*n if mode=="pgas" else 0)
    require(saved["native_calls"]==work and saved["calls"]=={"attempted":work,"completed":work,"failed":0},
            "The phase estimator changed the native rollout budget.")
    return terminal,phase,choice,ancestor_maps,work


def phase_term(path, origin):
    lifted = (path[0]^origin,*path[1:])
    return prod(likelihood(lifted[:k]) for k in range(2,len(lifted)+1))


@lru_cache(maxsize=None)
def shared(path):
    return HALF*sum(phase_term(path,s) for s in BITS)


@lru_cache(maxsize=None)
def fresh_at_each_window(path):
    return prod(HALF*sum(likelihood((path[0]^s,*path[1:k])) for s in BITS) for k in range(2,len(path)+1))


def rational_normalize(weights):
    tot = sum(weights)
    return tuple(w/tot for w in weights)


def rational_increment(path):
    return shared(path)/shared(path[:-1])


def rational_transition(reference, mode):
    population = {((a,b),reference[:2]):Fraction(1,4) for a,b in product(BITS,repeat=2)}
    for cut in (2,3):
        following = {}
        for pair,mass in population.items():
            weight = rational_normalize([rational_increment(p) for p in pair])
            if mode=="pg":
                conditional = (Fraction(0),Fraction(1))
            else:
                end = cut+1 if mode=="one_window" else len(reference)
                func = fresh_at_each_window if mode=="different_phase_each_window" else shared
                ratios = [func(p+reference[cut:end])/shared(p) for p in pair]
                conditional = rational_normalize([a*b for a,b in zip(weight,ratios,strict=True)])
            for i,j,b in product(BITS,repeat=3):
                q = pair[i]+(b,),pair[j]+(reference[cut],)
                following[q] = following.get(q,0)+mass*weight[i]*conditional[j]*HALF
        population = following
    row = dict.fromkeys(HISTORIES,Fraction(0))
    for pair,mass in population.items():
        w = rational_normalize([rational_increment(p) for p in pair])
        for member,v in zip(pair,w,strict=True):
            row[member] += mass*v
    return row


def fixtures():
    physical = rational_normalize([phase_term(p,0) for p in HISTORIES])
    corrected = rational_normalize([shared(p) for p in HISTORIES])
    failed = rational_normalize([fresh_at_each_window(p) for p in HISTORIES])
    lifted = dict.fromkeys(HISTORIES,Fraction(0))
    for p,w in zip(HISTORIES,corrected,strict=True):
        posterior_phase = rational_normalize([phase_term(p,s) for s in BITS])
        for s,w_s in enumerate(posterior_phase):
            lifted[(p[0]^s,*p[1:])] += w*w_s
    assert tuple(lifted.values()) == physical
    wrong_target = HALF*sum(abs(a-b) for a,b in zip(corrected,failed,strict=True))
    assert wrong_target>0
    out = {}
    for mode in ("pg","pgas","one_window","different_phase_each_window"):
        kernel = [rational_transition(p,mode) for p in HISTORIES]
        assert all(sum(r.values())==1 for r in kernel)
        pushed = [sum(w*r[p] for w,r in zip(corrected,kernel,strict=True)) for p in HISTORIES]
        tv = HALF*sum(abs(a-b) for a,b in zip(corrected,pushed,strict=True))
        assert (tv==0) == (mode in ("pg","pgas"))
        out[mode] = float(tv)
    wrong_product = np.full((3,PHASES),-10.)
    wrong_product[0,0] = wrong_product[1,1] = wrong_product[2,2] = 0.
    _,shared_gaussian = phase_log_masses(wrong_product)
    distinct = (logsumexp(wrong_product,axis=1)-np.log(PHASES)).sum()
    assert shared_gaussian[-1] < distinct
    return {"passed":True,"native_calls":0,"finite_children":16,"joint_lift_exact":True,
            "wrong_target_tv":float(wrong_target),"stationarity_tv":out}


def fixture():
    return fixtures()


def genealogy(ids,log,n,mode,start_id,ref_id=None):
    two,three = log["stages"]
    a2 = np.r_[two["ordinary_ancestors"],two["reference_ancestor"]]
    a3 = np.r_[three["ordinary_ancestors"],three["reference_ancestor"]]
    root_index = a2[a3]
    memories = ({},{})
    for row,at1,at2 in zip(ids,root_index,a3,strict=True):
        require(len(row)==4 and all(isinstance(q,str) and len(q)==64 for q in row),
                "One of four original ordered snapshots lacks a native digest.")
        for known,j,k in zip(memories,(at1,at2),(2,3),strict=True):
            require(known.setdefault(j,row[:k])==row[:k],"Native prefixes were grafted after scoring.")
        if at1==n-1:
            require(row[0]==start_id and (ref_id is None or row[:2]==ref_id[:2]),
                    "The observed ancestor did not retain its true initial/C1 prefix.")
        if mode=="pg" and at2==n-1 and ref_id is not None:
            require(row[:3]==ref_id[:3],"Plain PG did not retain its full reference prefix.")
    if mode=="pg":
        require(ids[-1][0]==start_id and (ref_id is None or ids[-1]==ref_id),
                "Plain PG changed its frozen reference.")
    expected_log = np.column_stack((log["initial_log_weights"][root_index],two["incremental_log_weights"][a3],
                                    three["incremental_log_weights"]))
    return expected_log


def camera_phase(noise,query,orbit):
    val = np.asarray(orbit)
    require(val.ndim==4 and val.shape[1:]==(3,PHASES,4) and np.isfinite(val).all(),
            "Original four camera channels must be read at every phase and time.")
    return -.5*np.square((val-query[None,:,None,:])/noise).sum(axis=-1)


def check_anchor(h, row, candidate, cfg, query, mode):
    if mode=="marginal":
        d,_,_,_ = anchor_phase(h,cfg,query)
    elif mode=="pgas":
        d = anchor_history(h,cfg,query)
    else:
        d = anchor_history(h,cfg,query)
    ids = [packet_fingerprint(p) for p in h["packets"]]
    require(list(row["path_ids"][candidate])==ids,"Raw native anchor differs from its forecast member.")
    if mode=="pg":
        close(row["local"][candidate],d["local"])
    else:
        phases = np.arange(PHASES)*cfg["motion"]["box"]/PHASES
        libs = h["packets"][1:]
        orbit = np.stack([shifted_observables(p,phases,_camera) for p in libs])
        orbit_hidden = (d["local_orbit"] if mode=="marginal" else
                        np.stack([shifted_observables(p,phases,independent_apertures) for p in libs],axis=1))
        close(row["local"][candidate],orbit_hidden)
        close(row["clean_orbit"][candidate],orbit)
    close(row["global"][candidate],d["global_early"])
    return d


def checks_forecast(rec,cfg,*,pg_mode,true_initial_id=None):
    """Audit one method's science ledger without invoking the native simulator."""
    n = MODES[pg_mode]
    method = rec["methods"][pg_mode]
    query = np.asarray(rec["query"])
    population = method["population"]
    require(len(population)==CHAINS*SWEEPS and len(method["first"])==2,
            "Some sweeps or raw anchors did not survive the frozen forecast.")
    output, reference_first_window, mass = [],[],{}
    total = 0
    for i,row in enumerate(population):
        log = row["ledger"]
        start_of_chain = i%SWEEPS==0
        if start_of_chain:
            slot = int(rec["starts"][i//SWEEPS])
            initial_id = rec["warm_ids"][slot]
            prior_id = None
            if pg_mode=="pg":
                ref = np.array([log["initial_log_weights"][-1],
                                *(l["incremental_log_weights"][-1] for l in log["stages"])])
            elif pg_mode=="pgas":
                ref = np.r_[log["initial_log_weights"][-1],log["stages"][0]["replayed_future_logl"][-1]]
            else:
                # PGAS only conditions on M1 of the reference in its filter;
                # its subsequent grafts are independently audited below.
                ref = np.array([log["initial_log_weights"][-1],np.nan,np.nan])
            reference_first_window.append(ref.copy())
        ids = row["path_ids"]
        glob,local = np.asarray(row["global"]),np.asarray(row["local"])
        qshape = (n,3,2,3) if pg_mode=="pg" else (n,PHASES,3,2,3)
        require(len(ids)==n and local.shape==qshape and glob.shape==(n,3,8)
                and np.isfinite(glob).all() and np.isfinite(local).all()
                and np.min(local)>=0. and np.max(local)<=1.+2.e-14,
                "Population omitted members or left native hidden graph feature support.")
        if true_initial_id is not None:
            require(all(x[0]!=true_initial_id for x in ids),"Blind forecasts saw a privileged original initial history.")
        expected_log = genealogy(ids,log,n,pg_mode,initial_id,prior_id)
        if pg_mode=="pg":
            require(row["phase"] is None and row["clean_orbit"] is None,
                    "Plain equal-cost particle Gibbs unexpectedly receives extra phase data.")
            w,choice,cost = sweep_ledger(log,"pg",n,ref)
            next_ref = expected_log[choice].copy()
            phase = None
        else:
            orbit = camera_phase(np.asarray(cfg["noise"]),query,row["clean_orbit"])
            cumulative,marginal = phase_log_masses(orbit)
            phase = np.stack([normalized(x) for x in cumulative[:,-1]])
            close(row["phase"],phase)
            if pg_mode=="pgas":
                close(orbit[:,:,0],expected_log)
                w,choice,cost = sweep_ledger(log,"pgas",n,ref)
                next_ref = expected_log[choice].copy()
            else:
                w,phase_output,choice,_,cost = phase_sweep(log,"pgas",n,ref,orbit)
                close(phase,phase_output)
                next_ref = marginal[choice].copy()
        if i==0:
            h0, hs = method["first"]
            check_anchor(h0,row,0,cfg,query,pg_mode)
            check_anchor(hs,row,choice,cfg,query,pg_mode)
        total += cost
        w = w/len(population)
        for path,v in zip(ids,w,strict=True):
            mass[path[0]] = mass.get(path[0],0.)+v
        output.append((local,glob,w,phase))
        prior_id = ids[choice]
        initial_id = prior_id[0]
        ref = next_ref
    require(total==method["native_calls"]==CHAINS*SWEEPS*93,
            "Marginalization charged a different production budget.")
    return output,np.array([len(mass),1./sum(v*v for v in mass.values())]),np.stack(reference_first_window)


def proper(atoms,weight,target):
    flat = np.asarray(atoms).reshape(len(weight),-1)
    # Duplicate forecasts are the same atoms in the scoring law.
    locations, index = np.unique(flat,axis=0,return_inverse=True)
    weight = np.bincount(index,weights=weight,minlength=len(locations))
    a = cdist(locations,np.asarray(target).reshape(1,-1))[:,0]
    b = cdist(locations,locations)
    return float((weight @ a-.5*weight @ b @ weight)/np.sqrt(flat.shape[1]))


def proper_vector(loc,glob,w,wg,obs,scale):
    truth = obs["local"]
    return np.array([proper(loc,w,truth),proper(loc[:,2]-loc[:,1],w,truth[2]-truth[1]),
           proper(loc[:,2],w,truth[2]),proper(loc[:,1],w,truth[1]),
           proper(glob/scale,wg,obs["global_early"]/scale)])


def accumulate(law,mode):
    a,b,c,d = [],[],[],[]
    for local,glob,w,phase in law:
        b.append(glob)
        d.append(w)
        if mode in ("post_phase","marginal"):
            a.append(local.reshape(-1,3,2,3))
            c.append((w[:,None]*phase).ravel())
        else:
            a.append(local if mode=="pg" else local[:,0])
            c.append(w)
    loc,glob,w,wg = (np.concatenate(v) for v in (a,b,c,d))
    close((w.sum(),wg.sum()),[1.,1.])
    return loc,glob,w,wg


def audit(out):
    names = ("REGISTERED","RUN","INDEPENDENT_FRACTION_FIXTURE","ACQUIRED",
             "PREDICTIONS_FROZEN","SCORE_STARTED","SCORED")
    files = {name:out/f"{name}.json" for name in names}
    d = {name:read(path) for name,path in files.items()}
    times = [datetime.fromisoformat(v["actual_utc"]) for v in d.values()]
    require(all(a<b for a,b in zip(times[:-1],times[1:],strict=True)) and
            all(a==datetime.fromisoformat(v["asia_shanghai"]) for a,v in zip(times,d.values(),strict=True)),
            "The simulator prediction and reveal were not actually separated in time.")
    p,r,a,f,s = (d[k] for k in ("REGISTERED","RUN","ACQUIRED","PREDICTIONS_FROZEN","SCORED"))
    reg, pred = sha(files["REGISTERED"]),sha(files["PREDICTIONS_FROZEN"])
    require(p["namespace"]==STUDY and tuple(p["units"])==tuple(a["units"])==tuple(f["units"])==UNITS
            and tuple(p["methods"])==tuple(MODES) and tuple(p["particles"])==tuple(MODES.values())
            and tuple(p["metrics"])==tuple(s["metrics"])==METRICS and tuple(p["gates"])==GATES
            and tuple(p["controls"])==CONTROLS and p["fixed_seed"]==KEY
            and (p["chains"],p["sweeps"])==(CHAINS,SWEEPS)
            and r["before_native"] is True and reg==r["registration_sha256"]==f["registered_sha256"]
            and f["truth_opened"] is False and f["acquired_sha256"]==sha(files["ACQUIRED"])
            and pred==d["SCORE_STARTED"]["forecast_sha256"]==s["forecast_sha256"]
            and p["truth_calls"]==a["native_calls"]==3*len(UNITS)
            and p["per_mother_forecast_calls"]==WORK and p["forecast_calls"]==s["forecast_native_calls"]==WORK*len(UNITS)
            and s["mothers"]==len(UNITS) and s["new_wet_measurements"]==0 and s["inverse_scored"] is False,
            "Frozen cohort, paired gates, scientific role, or equal-cost contract changed.")
    close(p["shift"],np.arange(PHASES)*.375)
    require(d["INDEPENDENT_FRACTION_FIXTURE"]["independent"]==fixture(),
            "Exact common-phase law fixture did not reproduce.")
    for name,digest in p["inputs"].items():
        require(sha(name)==digest,"A preregistered input changed: "+name)
    for name,digest in p["sources"].items():
        require(sha(out/"SOURCE_SNAPSHOT"/Path(name).name)==digest,"Registered source changed: "+name)
    scores = [Path(name) for name in p["inputs"] if Path(name).name=="SCORED.json"]
    require(len(scores)==1 and read(scores[0])["go_to_inverse"] is False,
            "Expected fixed parent audit showing the unresolved mixing problem.")
    previous = scores[0].parent
    old = read(previous/"REGISTERED.json")
    require(not set(old["units"]).intersection(UNITS)
            and read(previous/"INDEPENDENT_AUDIT.json")["scored_sha256"]==sha(scores[0]),
            "Independent study repeats the consumed parent outcomes.")
    truth_rows, forecasts = a["rows"],f["rows"]
    require(tuple(x["unit"] for x in truth_rows)==tuple(x["unit"] for x in forecasts)==UNITS,
            "Some registered mother receipts were altered or dropped.")
    y = []
    for row in truth_rows:
        dat = np_load(artifact(row["query"],out/f'{row["unit"]}_QUERY.npz'))
        require(set(dat)=={"measured"} and dat["measured"].shape==(3,4)
                and np.isfinite(dat["measured"]).all(),"Forecast view contains more than the three original cameras.")
        y.append(dat["measured"])
    atlas,weights = audit_atlas(out,{**p,"training_units":p["atlas_train"]},f,np.stack(y))
    cfg,scale = p["physics"],atlas["global_scale"]
    losses = {key:[] for key in (*MODES,"post_phase",*ATLAS)}
    diagnostics = {mode:[] for mode in MODES}
    for i,(row,for_row,query) in enumerate(zip(truth_rows,forecasts,y,strict=True)):
        u = row["unit"]
        outcome = load_tree(artifact(row["truth"],out/f"{u}_TRUTH.npz"))
        require(row["native_calls"]==3,"Observed simulator evidence must be one new mother's three frames.")
        generating_state(cfg,u,query,outcome)
        actual = _features(outcome["packets"],cfg)
        true_initial = packet_fingerprint(outcome["packets"][0])
        rec = load_tree(artifact(for_row["forecast"],out/f"{u}_FORECAST.npz"))
        require(rec["unit"]==u and rec["native_calls"]==for_row["native_calls"]==WORK and rec["methods"].keys()==MODES.keys(),
                "Unexpected forecast roles or unreported total original native work.")
        np.testing.assert_array_equal(rec["query"],query)
        warm_ids = [packet_fingerprint(build_packet(cfg,draw_initial(cfg,rng(KEY,u,0,j)))) for j in range(WARM)]
        require(rec["warm_ids"]==warm_ids and true_initial not in warm_ids
                and np.asarray(rec["warm_weights"]).shape==(WARM,)
                and np.min(rec["warm_weights"])>=0.,"Warm starts changed or used a hidden truth source.")
        close(np.sum(rec["warm_weights"]),1.)
        np.testing.assert_array_equal(rec["starts"],rng(KEY,u,1).choice(WARM,size=CHAINS,p=rec["warm_weights"]))
        methods = {}
        references = {}
        for mode in MODES:
            methods[mode],diag,references[mode] = checks_forecast(rec,cfg,pg_mode=mode,true_initial_id=true_initial)
            diagnostics[mode].append(diag)
        close(references["pg"],references["pgas"])
        methods["post_phase"] = methods["pgas"]
        for mode,v in methods.items():
            losses[mode].append(proper_vector(*accumulate(v,mode),actual,scale))
        for key in ATLAS:
            w = weights[key][i]
            losses[key].append(proper_vector(atlas["bank"]["local"],atlas["bank"]["global_early"],w,w,actual,scale))
    saved = load_tree(artifact(s["arrays"],out/"SCORE_ARRAYS.npz"))
    np.testing.assert_array_equal(saved["units"],UNITS)
    require(saved["losses"].keys()==losses.keys() and saved["diagnostic"].keys()==diagnostics.keys()
            and s["diagnostic_columns"]==["distinct_terminal_initials","weighted_initial_ess"],
            "Unreported method or lost distinct-initial diagnostic.")
    for mode,v in losses.items():
        close(v,saved["losses"][mode])
        close(np.mean(v,axis=0),s["means"][mode])
    for mode,v in diagnostics.items():
        close(v,saved["diagnostic"][mode])
        close(np.mean(v,axis=0),s["diagnostics"][mode])
    passed = True
    for i,key in enumerate(METRICS):
        for mode in CONTROLS:
            difference = np.asarray(losses[mode])[:,i]-np.asarray(losses["marginal"])[:,i]
            paired = intervals95(difference)
            close(paired,s["paired_gains"][key][mode])
            if key in GATES:
                passed &= bool(paired[1]>0.)
    require(s["go_to_inverse"] is passed,"Shared-phase direction changed its registered positive gates after outcome opening.")
    return {**stamp(),"passed":True,"registered_sha256":reg,"scored_sha256":sha(files["SCORED"]),
            "predictions_sha256":pred,"mother_units":len(UNITS),"native_calls_audited":WORK*len(UNITS),
            "sweeps_audited":len(UNITS)*CHAINS*SWEEPS*len(MODES),"new_native_calls":0,"new_wet_measurements":0,
            "trajectory_anchors":len(UNITS)*6,"complete_phase_likelihood_readback":True,
            "foundation":"Same phase kept across all three frames; no independent native re-integration.",
            "limits":"Only raw archived anchors provide independent pixel/graph readback. No new fit, material observations or inverse selection.",
            "go_to_inverse":passed}


if __name__=="__main__":
    parser = ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--fixtures",action="store_true")
    action.add_argument("--out",type=Path)
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        if args.fixtures:
            print(json.dumps(fixtures(),indent=2))
        else:
            out = args.out.resolve(strict=True)
            write(out/"INDEPENDENT_AUDIT.json",audit(out))
            print("PASS independent common phase audit",out)
