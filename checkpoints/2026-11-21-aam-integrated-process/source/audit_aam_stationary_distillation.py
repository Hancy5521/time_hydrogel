"""Independent frozen teacher and GRU readback without native rollout or Torch refitting."""

from argparse import ArgumentParser
from datetime import datetime
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
from scipy.linalg import eigvalsh
from scipy.spatial.distance import cdist
from scipy.special import expit,logsumexp
from threadpoolctl import threadpool_limits

from aam_formation_actual_rollout import packet_fingerprint
from aam_formation_cycle_driver import np_load, read, sha, write
from aam_formation_path_cycle import load_tree, stamp
from audit_aam_assay_guided_sir import _camera
from audit_aam_endpoint_tape_move import artifact, close, generating_state, intervals95, require, rng
from audit_aam_original_tape_pgas import ATLAS
from audit_aam_phase_marginal import phase_log_masses,phase_sweep,shifted_observables
from audit_aam_phase_stationary_bound import inverse_gauge,hidden_orbit,packet_dag,phase_stages,table_hash
from audit_aam_temporal_energy_law import independently_predict,solve_dev_projection


NS = "aam-physical-stationary-teacher-gru-residual-v1"
SPLITS = {"train":list(range(2026143000,2026143256)),"dev":list(range(2026143256,2026143320)),
          "held":list(range(2026143320,2026143448))}
MODELS = ("single_truth","physical_teacher")
NAMES = (*ATLAS,*MODELS)
METRICS = ("early18","hidden_delta32_6")
ALL_METRICS = (*METRICS,"hidden_t3_6")
EPOCHS,SHRINK = (8,24,64),(0.,.5,1.)
GRID = [[epoch,s] for s in SHRINK for epoch in EPOCHS]
TRAIN_SEED,TEACHER_SEED = 47265915,935087251
CHAINS,SWEEPS,PARTICLES,PHASES,TEACHER_CALLS = 4,8,16,16,2988
SCHEMA = "aam-stationary-distillation-energy-gru24-v1"
GEOMETRY = "aam-equal-early18-and-delta32-negative-type-spectral-v1"


def metric_blocks(local):
    x = np.asarray(local)
    return (x,x[:,2]-x[:,1],x[:,2])


def independent_cross(atlas,target,w):
    """The three separate linear distance terms on raw aperture values."""
    a = np.asarray(atlas); v = np.asarray(target); mass = np.asarray(w)
    require(a.shape==(512,3,2,3) and v.shape==(len(mass),3,2,3) and
            np.isfinite(v).all() and np.isfinite(mass).all() and np.min(mass)>=0.,
            "The teacher does not terminate on an original normalized path law.")
    close(mass.sum(),1.)
    result = []
    for x,y in zip(metric_blocks(a),metric_blocks(v),strict=True):
        first,second = x.reshape(len(x),-1),y.reshape(len(y),-1)
        result.append((cdist(first,second) @ mass)/np.sqrt(first.shape[1]))
    return np.stack(result)


def lens(packets,cfg,cache):
    camera,local = [],[]
    phases = np.arange(PHASES)*(cfg["motion"]["box"]/PHASES)
    for i,packet in enumerate(packets[1:],start=1):
        require(packet[0].position[0]==cfg["prefix_clock"][i],"Saved target frame uses the wrong time aperture.")
        fp = packet_fingerprint(packet)
        if fp not in cache:
            cache[fp] = (shifted_observables(packet,phases,_camera),hidden_orbit(packet,cfg))
        y,z = cache[fp]
        camera.append(y)
        local.append(z)
    return np.stack(camera),np.stack(local,axis=1)


def preserved_packets(packets):
    require(len(packets)==4,"Every teacher path must have the original three time intervals.")
    for old,new in zip(packets[:-1],packets[1:],strict=True):
        a,b = old[0],new[0]
        evt,edge = a.counts[:2]
        for name,k in (("events",evt),("edges",edge),("lifts",edge)):
            np.testing.assert_array_equal(getattr(b,name)[:k],getattr(a,name)[:k])


def recovered_start(ledger,branch,unit,raw,tables,query,cfg,packets,cache,used):
    init = ledger["initial_reference"]
    truth = raw["packets"]
    phase = int(rng(TEACHER_SEED,unit,branch,0,0).integers(PHASES))
    offset = -phase*(cfg["motion"]["box"]/PHASES)
    require(init["phase"]==phase and init["inverse_shift"]==offset
            and init["native_counter"]=={"attempted":3,"completed":3,"failed":0}
            and len(init["canonical_packet_sha256"])==4 and len(init["position_parity_by_frame"])==4,
            "Teacher did not independently initialize from the physical truth with a uniform inverse phase.")
    reference = [packets[x] for x in init["canonical_packet_sha256"]]
    used.update(init["canonical_packet_sha256"])
    preserved_packets(reference)
    box = cfg["motion"]["box"]
    initial_images = np.floor((truth[0][0].points[:,1]+offset)/box).astype(int)
    errors,formal = [],[]
    for original,actual in zip(truth,reference,strict=True):
        ideal,_ = inverse_gauge(original,offset,initial_images)
        formal.append(packet_fingerprint(ideal))
        a,b = ideal[0],actual[0]
        differ = a.points-b.points
        differ -= box*np.rint(differ/box)
        er = float(np.max(np.abs(differ)))
        require(er<=2.e-11,"Inverse phase independent replay left the original shifted physical trajectory.")
        for key in ("present","active","available","owner","edges","counts","winding","periodic_parent","periodic_shift"):
            np.testing.assert_array_equal(getattr(b,key),getattr(a,key))
        np.testing.assert_array_equal(b.lifts[:b.counts[1]],a.lifts[:a.counts[1]])
        close(b.events[:b.counts[0],:10],a.events[:a.counts[0],:10],atol=2.e-11)
        close(b.unwrapped,a.unwrapped,atol=2.e-11)
        errors.append(er)
    proof = init["archived_anchor_proof"]
    require(proof["unit"]==unit and proof["inverse_phase_index"]==phase and proof["inverse_displacement"]==offset
            and proof["native_calls"]==0 and proof["original_interval_receipts_unchanged"] is True
            and proof["original_packet_fingerprints"]==[packet_fingerprint(g) for g in truth]
            and proof["canonical_inverse_packet_fingerprints"]==formal
            and proof["original_tape_hashes"]==[table_hash(g) for g in tables],
            "Teacher anchor is not the pre-existing legal physical lift of this raw training outcome.")
    np.testing.assert_array_equal(initial_images,proof["initial_y_image_numbers"])
    close(init["position_parity_by_frame"],errors)
    close(init["max_torus_position_error"],max(errors))
    y,_ = lens(reference,cfg,cache)
    expected = np.stack([_camera(p) for p in truth[1:]])
    close(y[:,phase],expected,atol=2.e-11)
    close(init["phase_camera_max_abs_error"],np.max(np.abs(y[:,phase]-expected)))
    ll = -.5*np.square((y-query[:,None,:])/cfg["noise"]).sum(axis=-1)
    _, m = phase_log_masses(ll)
    return m[0]


def teacher_case(out,ticket,training,cfg,atlas):
    u = ticket["unit"]
    folder = out/"train"
    require(training["unit"]==u and ticket["native_calls"]==TEACHER_CALLS,
            "Train teacher received a nontraining mother or partial work.")
    observed = np_load(artifact(training["obs"],folder/f"{u}_OBS.npz"))
    require(set(observed)=={"measured"} and observed["measured"].shape==(3,4),"Wrong original camera payload.")
    query = observed["measured"]
    raw = load_tree(artifact(training["raw"],folder/f"{u}_RAW.npz"))
    _,tables = generating_state(cfg,u,query,raw)
    target = raw["packets"]
    record = load_tree(artifact(ticket["teacher"],folder/f"{u}_TEACHER.npz"))
    require(record["unit"]==u and record["support"]==ticket["packets"]
            and tuple(record["metrics"])==ALL_METRICS and record["native_calls"]==TEACHER_CALLS,
            "Missing training-only teacher receipt.")
    np.testing.assert_array_equal(record["measured"],query)
    packets = packet_dag(artifact(ticket["packets"],folder/f"{u}_PACKETS.npz"),cfg)
    cache,used = {},set()
    w = np.asarray(record["path_weights"]); wp = np.asarray(record["phase_weights"])
    close(w.sum(axis=1),np.ones(CHAINS))
    close(wp.sum(axis=-1),np.ones((CHAINS,PARTICLES)))
    key = np.asarray(record["path_ids"])
    require(w.shape==(CHAINS,PARTICLES) and wp.shape==(CHAINS,PARTICLES,PHASES)
            and key.shape==(CHAINS,PARTICLES,4) and len(record["chains"])==CHAINS,
            "One equal-weight independently addressed chain lost its complete terminal laws.")
    target_values = []
    for j,route in enumerate(record["chains"]):
        ref = recovered_start(route,j,u,raw,tables,query,cfg,packets,cache,used)
        values,camera = [],[]
        require(len(route["sweeps"])==SWEEPS,"Training failed or truncated a full set of exact sweeps.")
        for k,step in enumerate(route["sweeps"]):
            if k<SWEEPS-1:
                ref,*_ = phase_stages(step,ref)
        for path in key[j]:
            used.update(path.tolist())
            history = [packets[x] for x in path]
            preserved_packets(history)
            y,local = lens(history,cfg,cache)
            camera.append(y)
            values.append(local)
        clean = np.stack(camera)
        lp = -.5*np.square((clean-query[None,:,None,:])/cfg["noise"]).sum(axis=-1)
        close(clean,record["clean_orbit"][j])
        close(lp,record["phase_logl"][j])
        close(values,record["local"][j])
        t,phi,pick,_,_ = phase_sweep(route["sweeps"][-1],"pgas",PARTICLES,[ref,np.nan,np.nan],lp)
        np.testing.assert_array_equal(route["sweeps"][-1]["selected_path_ids"],key[j,pick])
        close(w[j],t)
        close(wp[j],phi)
        target_values.extend(values)
    require(used==set(packets),"Some serialized native states have no immutable teacher provenance.")
    atoms = np.stack(target_values).reshape(-1,3,2,3)
    teacher = independent_cross(atlas,atoms,(w[:,:,None]*wp/CHAINS).ravel())
    truth = lens(target,cfg,cache)[1][0]
    single = independent_cross(atlas,truth[None],np.ones(1))
    close(record["teacher_cross"],teacher)
    close(record["truth_cross"],single)
    return query,teacher,single


def geometry_check(geometry,local):
    blocks = metric_blocks(local)
    d = 0.
    for block in blocks[:2]:
        x = block.reshape(len(block),-1)
        d += .5*cdist(x,x)/np.sqrt(x.shape[1])
    n = len(local)
    x = np.asarray(geometry["codes"])
    require(geometry["schema"]==GEOMETRY and tuple(geometry["metrics"])==METRICS
            and geometry["rank"]==16 and x.shape==(n,16),"Wrong learned residual coordinates.")
    close(d,geometry["distance"])
    H = np.eye(n)-np.ones((n,n))/n
    gram = -.5*H@d@H
    eig = eigvalsh(gram)
    keep = eig[-16:][::-1]
    close(geometry["eigenvalues"],keep)
    close(x.T@x,np.diag(keep))
    close(gram@x,x*keep)
    close(x.sum(axis=0),np.zeros(16),atol=2.e-10)
    close(geometry["smallest_full_eigenvalue"],eig[0])
    close(geometry["retained_gram_trace_fraction"],keep.sum()/np.maximum(eig,0.).sum())
    h = sha256()
    for a in (geometry["distance"],geometry["codes"]):
        a = np.asarray(a,dtype="<f8",order="C")
        h.update(str(a.shape).encode("ascii"))
        h.update(a.tobytes())
    require(h.hexdigest()==geometry["sha256"],"Frozen spectral tilt is not bound to all legal paths.")


def learned_check(model,geometry,scale):
    seed = int(np.random.SeedSequence(TRAIN_SEED).spawn(2)[1].generate_state(1,dtype=np.uint64)[0])&((1<<63)-1)
    trace = np.asarray(model["training_trace"])
    require(model["schema"]==SCHEMA and model["geometry_sha256"]==geometry["sha256"]
            and model["rank"]==16 and model["hidden_size"]==24 and model["training_mothers"]==len(SPLITS["train"])
            and model["seed"]==TRAIN_SEED and model["model_seed"]==seed and tuple(model["epochs"])==EPOCHS
            and set(model["states"])==set(map(str,EPOCHS)) and model["initial_residual"]=="zero_head"
            and model["batch_size"]==32 and model["learning_rate"]==.003 and model["weight_decay"]==.0001
            and model["tilt_penalty"]==.001 and trace.shape==(64,2) and np.isfinite(trace).all()
            and np.min(trace[:,1])>=0.,
            "The only learner differences should be original and teacher losses at training queries.")
    np.testing.assert_array_equal(model["camera_scale"],scale)


def predict_numpy(model,geometry,query,weights,epoch,shrinking):
    """PyTorch GRU r,z,n gate ordering and output law using only saved arrays."""
    base = np.asarray(weights)
    if shrinking==0.:
        return base
    d = model["states"][str(epoch)]
    x = query/model["camera_scale"]
    h = np.zeros((len(x),24))
    for k in range(3):
        u = x[:,k,:] @ d["gru.weight_ih_l0"].T+d["gru.bias_ih_l0"]
        s = h @ d["gru.weight_hh_l0"].T+d["gru.bias_hh_l0"]
        reset = expit(u[:,:24]+s[:,:24])
        residual = expit(u[:,24:48]+s[:,24:48])
        candidate = np.tanh(u[:,48:]+reset*s[:,48:])
        h = (1.-residual)*candidate+residual*h
    tilt = h @ d["output.weight"].T+d["output.bias"]
    allowed = np.full(base.shape,-np.inf)
    np.log(base,where=base>0.,out=allowed)
    log = allowed+tilt@geometry["codes"].T
    q = np.exp(log-logsumexp(log,axis=1,keepdims=True))
    return (1.-shrinking)*base+shrinking*q


def independent_losses(weights,atlas,raw):
    result = []
    for a,b in zip(metric_blocks(atlas)[:2],metric_blocks(raw)[:2],strict=True):
        x,y = a.reshape(len(a),-1),b.reshape(len(b),-1)
        D,C = cdist(x,x),cdist(y,x)
        result.append((np.einsum("ij,ij->i",C,weights)-.5*np.einsum("ij,ij->i",weights@D,weights))/np.sqrt(x.shape[1]))
    return np.column_stack(result)


def clock(doc):
    t = datetime.fromisoformat(doc["actual_utc"])
    require(t==datetime.fromisoformat(doc["asia_shanghai"]),"UTC and display timezone disagree.")
    return t


def acquisition(out,role,reg):
    info = out/f"{role.upper()}_ACQUIRED.json"
    started = out/role/"ACQUISITION_STARTED.json"
    s,a = read(started),read(info)
    ids = SPLITS[role]
    require(clock(s)<clock(a) and s["role"]==a["role"]==role and s["registration_sha256"]==a["registration_sha256"]==reg
            and [d["unit"] for d in a["rows"]]==ids and a["native_calls"]==3*len(ids)
            and all(q["role"]==role and q["native_calls"]==3 for q in a["rows"]),
            "A simulation mother migrated role, was dropped or failed its original acquisition.")
    return info,a,clock(s)


def data_readback(out,role,cfg,a):
    cameras,truth = [],[]
    folder = out/role
    for row in a["rows"]:
        unit = row["unit"]
        view = np_load(artifact(row["obs"],folder/f"{unit}_OBS.npz"))
        require(set(view)=={"measured"} and view["measured"].shape==(3,4),
                "Forecast input contains hidden C3 or another feature view.")
        y = view["measured"]
        raw = load_tree(artifact(row["raw"],folder/f"{unit}_RAW.npz"))
        generating_state(cfg,unit,y,raw)
        _,h = lens(raw["packets"],cfg,{})
        cameras.append(y)
        truth.append(h[0])
    return np.stack(cameras),np.stack(truth)


def baseline_laws(model,p,query):
    fn,_ = independently_predict(model,query)
    t = {}
    for n in ATLAS:
        kind,u = fn(n,p["atlas_settings"].get(n))
        if kind=="target":
            u = solve_dev_projection(model["basis"]["training_coordinates"],u,model["prior"],model["entropy"])
        t[n] = u
    return t


def compare_weights(expected,received):
    require(set(expected)==set(received),"One of the frozen strong baselines is missing.")
    for k,v in expected.items():
        x = np.asarray(received[k])
        require(x.shape==v.shape and np.all(np.isfinite(x)) and np.min(x)>=0.,
                "Forecast is not a valid probability law on original atlas.")
        close(v,x,atol=2.e-10)
        close(x.sum(axis=1),np.ones(len(x)))


def frozen_atlas(p):
    path = Path(p["atlas_model"])
    origin = path.parent
    r,f,a = map(lambda n:read(origin/n),("REGISTERED.json","TRAINED_AND_SELECTED.json","INDEPENDENT_AUDIT.json"))
    require(p["atlas_units"]==r["units"]["train"]==f["train_ids"] and len(set(p["atlas_units"]))==512
            and p["atlas_settings"]==f["selected"] and f["held_opened"] is False
            and a["passed"] is True and a["registered_sha256"]==sha(origin/"REGISTERED.json")
            and a["forecast_sha256"]==sha(origin/"PREDICTIONS_FROZEN.json")
            and p["physics"]==r["physics"],"The strong base library differs from the original validated same-information laws.")
    r0 = [Path(a) for a in p["inputs"] if Path(a).name=="REGISTERED.json"
          and read(a).get("namespace")=="aam-common-phase-native-stationary-risk-bound-v1"]
    require(len(r0)==1,"Missing archived same-law teacher ancestry.")
    origin = r0[0].parent
    q,s,cert = map(lambda n:read(origin/n),("REGISTERED.json","SCORED.json","INDEPENDENT_AUDIT.json"))
    require(cert["passed"] is True and cert["registration_sha256"]==sha(r0[0])
            and cert["scored_sha256"]==sha(origin/"SCORED.json")
            and q["physics"]==p["physics"] and
            not set(sum(SPLITS.values(),[])).intersection(sum(r["units"].values(),[])+q["units"]),
            "Previously scored mothers are excluded from the new train, dev and held sets.")
    return load_tree(path,p["inputs"][str(path)])


def dev_readback(out,p,model,bank,reg,fit_time,fit_hash):
    info,role,started = acquisition(out,"dev",reg)
    freeze_path = out/"DEV_PREDICTIONS_FROZEN.json"
    f,d = map(read,(freeze_path,out/"DEV_SELECTED.json"))
    require(clock(role)<clock(f)<clock(d) and fit_time<clock(f)
            and f["acquired_sha256"]==sha(info) and f["fit_sha256"]==fit_hash
            and f["dev_labels_opened"] is False and d["held_not_opened"] is True
            and d["forecast_sha256"]==sha(freeze_path) and d["dev_grid"]==GRID,
            "Development grid changes or targets entered weights before freezing.")
    y,raw = data_readback(out,"dev",p["physics"],role)
    expected = baseline_laws(bank,p,y)
    observed = load_tree(artifact(f["archives"],out/"DEV_FORECASTS.npz"))
    require(observed["unit"]==SPLITS["dev"] and set(observed["grid"])==set(MODELS),"DEV role omitted a frozen checkpoint.")
    compare_weights(expected,observed["weights"])
    local = bank["bank"]["local"]
    points = {}
    for m in MODELS:
        weights = observed["grid"][m]
        require(weights.shape==(len(GRID),len(y),512),"Frozen DEV selection grid lost an arm.")
        expected = [predict_numpy(model[m],model["geometry"],y,observed["weights"]["kernel"],e,s) for e,s in GRID]
        close(weights,expected,atol=2.e-10)
        close(weights.sum(axis=2),np.ones((len(GRID),len(y))))
        points[m] = np.stack([independent_losses(x,local,raw) for x in weights])
    baselines = {m:independent_losses(w,local,raw) for m,w in observed["weights"].items()}
    chosen = {k:int(np.argmin(v.mean(axis=(1,2)))) for k,v in points.items()}
    require(d["chosen_index"]==chosen and d["chosen"]=={m:GRID[j] for m,j in chosen.items()},
            "The frozen DEV winner is not the preregistered mean-energy minimizer.")
    single,teacher = (points[m][chosen[m]].mean(axis=0) for m in MODELS)
    close(d["teacher_mean"],teacher)
    close(d["single_mean"],single)
    stored = load_tree(artifact(d["arrays"],out/"DEV_SCORE_ARRAYS.npz"))
    require(stored["unit"]==SPLITS["dev"],"DEV comparison lost its independent mother identity.")
    for m,v in points.items():
        close(stored["grid_losses"][m],v)
    for m,v in baselines.items():
        close(stored["baseline_losses"][m],v)
        close(d["ref_means"][m],v.mean(axis=0))
    go = (d["chosen"]["physical_teacher"][1]>0. and
          np.all(teacher<baselines["kernel"].mean(axis=0)) and teacher.mean()<single.mean())
    require(d["go_to_fresh_held"] is bool(go),"The preregistered development rule did not independently reproduce.")
    return d


def held_readback(out,p,bank,model,dev,score,reg):
    paths,role,started = acquisition(out,"held",reg)
    freeze_path = out/"PREDICTIONS_FROZEN.json"
    f = read(freeze_path)
    require(clock(dev)<started<clock(role)<clock(f)<clock(score)
            and f["acquired_sha256"]==sha(paths) and f["held_truth_opened"] is False
            and f["native_prediction"]==0 and score["held_opened"] is True
            and sha(out/"DEV_SELECTED.json")==f["dev_selection_sha256"]==score["dev_selected_sha256"]
            and f["held_units"]==SPLITS["held"] and score["held_forecast_sha256"]==sha(freeze_path)
            and score["held_mothers"]==len(SPLITS["held"]),"Held data entered model selection or forecast closure.")
    y,truth = data_readback(out,"held",p["physics"],role)
    observed = np_load(artifact(f["weight_ref"],out/"HELD_WEIGHTS.npz"))
    np.testing.assert_array_equal(y,observed.pop("query"))
    expected = baseline_laws(bank,p,y)
    for m,e in dev["chosen"].items():
        expected[m] = predict_numpy(model[m],model["geometry"],y,observed["kernel"],*e)
    compare_weights(expected,observed)
    loss = {m:independent_losses(w,bank["bank"]["local"],truth) for m,w in observed.items()}
    saved = load_tree(artifact(score["arrays"],out/"HELD_SCORES.npz"))
    require(saved["unit"]==SPLITS["held"] and set(loss)==set(saved["losses"])==set(NAMES),
            "Wrong independent held assessment or omitted comparators.")
    gate = True
    for m,a in loss.items():
        close(saved["losses"][m],a)
        close(score["means"][m],a.mean(axis=0))
        if m!="physical_teacher":
            t = intervals95(a-loss["physical_teacher"])
            close(score["paired_gains"][m],t)
            gate &= bool(np.all(t[:,1]>0.))
    require(score["go_to_original_inverse"] is gate,"Held gate disagrees with matched strong original comparators.")
    return gate


def audit(out):
    reg_path = out/"REGISTERED.json"
    reg = sha(reg_path)
    p,r,score = map(read,(reg_path,out/"RUN.json",out/"SCORED.json"))
    require(p["namespace"]==NS and p["units"]==SPLITS
            and p["epochs"]==list(EPOCHS) and p["shrink"]==list(SHRINK) and p["grid"]==GRID
            and p["train_seed"]==TRAIN_SEED and p["teacher_seed"]==TEACHER_SEED
            and p["learner_fit_roles"]==["train"] and p["dev_selection_only"] is True
            and p["old_holdout_reuse"] is False and p["forecast_native_calls"]==p["new_wet_measurements"]==0
            and r["before_native_or_fit"] is True and r["registered_sha256"]==reg and clock(p)<clock(r)
            and tuple(p["metrics"])==tuple(score["metrics"])==METRICS and tuple(p["names"])==NAMES
            and p["new_teacher_native_calls"]==TEACHER_CALLS*len(SPLITS["train"]),
            "Changed original prospective scope, cohort, readout or training contract.")
    for path,digest in p["inputs"].items():
        require(sha(path)==digest,"Bound frozen original input was overwritten.")
    for path,digest in p["sources"].items():
        require(sha(out/"SOURCE_SNAPSHOT"/Path(path).name)==digest,"Original preregistered code changed.")
    a,train,started = acquisition(out,"train",reg)
    t0,t1,s0,s1 = [out/f"{k}.json" for k in ("TRAIN_TEACHER_STARTED","TRAIN_TEACHER_FROZEN","FIT_STARTED","FIT_FROZEN")]
    b,c,e,f = map(read,(t0,t1,s0,s1))
    require(clock(r)<started<clock(train)<clock(b)<clock(c)<clock(e)<clock(f)
            and sha(a)==b["training_acquisition_sha256"]==c["training_acquisition_sha256"]
            and sha(t1)==e["teacher_freeze_sha256"]==f["teacher_freeze_sha256"]
            and b["allowed_roles"]==e["roles"]==f["roles"]==["train"] and c["role"]=="train"
            and c["true_training_path_used_for_reference_only"] is True
            and c["raw_truth_as_query_input"] is c["held_read"] is c["dev_read"] is False
            and e["dev_and_held_labels_opened"] is f["dev_and_held_labels_opened"] is False
            and p["new_teacher_native_calls"]==b["native_budget"]==c["native_calls"]
            and f["fit_input_shapes"]==[len(SPLITS["train"]),3,4]
            and [v["unit"] for v in c["rows"]]==SPLITS["train"],
            "Only independent train truths may build the original native stationary teacher.")
    atlas = frozen_atlas(p)
    local = atlas["bank"]["local"]
    calc = [teacher_case(out,ticket,row,p["physics"],local)
            for ticket,row in zip(c["rows"],train["rows"],strict=True)]
    query,teacher,truth = (np.stack(rows) for rows in zip(*calc,strict=True))
    archive = np_load(artifact(c["arrays"],out/"TRAIN_TEACHER_CROSS.npz"))
    np.testing.assert_array_equal(archive["unit"],SPLITS["train"])
    np.testing.assert_array_equal(archive["metrics"],ALL_METRICS)
    np.testing.assert_array_equal(archive["measured"],query)
    close(archive["teacher_cross"],teacher)
    close(archive["truth_cross"],truth)
    model = load_tree(artifact(f["model"],out/"TRAINED_MODELS.npz"))
    require(set(model)=={"geometry",*MODELS},"A matched deep learner comparison arm is missing.")
    geometry_check(model["geometry"],local)
    for m in MODELS:
        learned_check(model[m],model["geometry"],atlas["camera_scale"])
    dev = dev_readback(out,p,model,atlas,reg,clock(f),sha(s1))
    require(score["dev_selected_sha256"]==sha(out/"DEV_SELECTED.json") and clock(dev)<clock(score),
            "The actual selection decision is not bound to the prospective score.")
    if dev["go_to_fresh_held"]:
        gate = held_readback(out,p,atlas,model,dev,score,reg)
        held = len(SPLITS["held"])
    else:
        held,gate = 0,False
        require(score["held_opened"] is False and score["held_mothers"]==0 and score["go_to_original_inverse"] is False
                and score["dev_result"]==dev and all(not (out/j).exists() for j in
                ("held","HELD_ACQUIRED.json","HELD_WEIGHTS.npz","PREDICTIONS_FROZEN.json","HELD_SCORES.npz")),
                "Formal DEV failure must retain dormant, untouched held mother slots.")
    return {**stamp(),"passed":True,"registered_sha256":reg,"scored_sha256":sha(out/"SCORED.json"),
            "teacher_freeze_sha256":sha(t1),"fit_sha256":sha(s1),"split_mothers":{"train":len(SPLITS["train"]),
            "dev":len(SPLITS["dev"]),"held_assessed":held},"new_native_calls":0,"new_fits":0,"new_wet_measurements":0,
            "teacher_native_calls_audited":p["new_teacher_native_calls"],"original_phase_laws_and_teacher_distances":True,
            "numpy_gru_forecasts_recalculated":True,"go_to_inverse":gate,
            "scope":"Frozen DEV or subsequent truly blind original-simulator scoring; no independent refit or raw simulator reintegration."}


def fixtures():
    h = np.zeros((3,2,3)); A = np.stack([h,h+1.])
    w = np.array([[.25,.75]])
    close(independent_losses(w,A,h[None]),[[.5625,0.]])
    model = {"camera_scale":np.ones((3,4)),"states":{"8":{
             "gru.weight_ih_l0":np.zeros((72,4)),"gru.bias_ih_l0":np.zeros(72),
             "gru.weight_hh_l0":np.zeros((72,24)),"gru.bias_hh_l0":np.zeros(72),
             "output.weight":np.zeros((16,24)),"output.bias":np.zeros(16)}}}
    q = predict_numpy(model,{"codes":np.zeros((2,16))},np.zeros((1,3,4)),w,8,.5)
    close(q,w)
    return {"passed":True,"native_calls":0,"new_fit":0,"original_proper_energy":True,"numpy_zero_residual_gru":True}


if __name__=="__main__":
    desc = ArgumentParser(description=__doc__)
    g = desc.add_mutually_exclusive_group(required=True)
    g.add_argument("--fixtures",action="store_true")
    g.add_argument("--out",type=Path)
    args = desc.parse_args()
    with threadpool_limits(limits=1):
        if args.fixtures:
            print(json.dumps(fixtures(),indent=2))
        else:
            out = args.out.resolve(strict=True)
            write(out/"INDEPENDENT_AUDIT.json",audit(out))
            print("PASS independent stationary teacher and frozen GRU",out)
