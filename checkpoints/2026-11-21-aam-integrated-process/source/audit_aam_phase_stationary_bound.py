"""Independent readback of quarantined shared-truth Bayes-risk diagnostics."""

from argparse import ArgumentParser
from datetime import datetime
from fractions import Fraction
from hashlib import sha256
from itertools import combinations
import json
from pathlib import Path

import numpy as np
from scipy.spatial.distance import cdist
from scipy.special import logsumexp
from threadpoolctl import threadpool_limits

from aam_edge_fork import State
from aam_formation_actual_rollout import PACKET_FIELDS, STATE_FIELDS, packet_fingerprint
from aam_formation_cycle_driver import np_load, read, sha, write
from aam_formation_local_initial import draw_initial
from aam_formation_path_cycle import load_tree, stamp
from audit_aam_assay_guided_sir import _camera
from audit_aam_camera_gaussian_tape_proposal import check_packet
from audit_aam_endpoint_tape_move import (
    ATTEMPTS,CELLS,artifact,close,generating_state,intervals,intervals95,require,rng,
)
from audit_aam_original_tape_pgas import categorical,normalized
from audit_aam_phase_marginal import PHASES,phase_log_masses,phase_sweep,shifted_observables
from audit_aam_temporal_energy_law import independently_predict


NS = "aam-common-phase-native-stationary-risk-bound-v1"
WIDE = "aam-original-shared-phase-width-vs-replay-v1"
LIFT = "aam-stationary-archived-truth-shared-phase-v1"
UNITS = tuple(range(2026142000,2026142048))
LAGS = (4,12,32)
BRANCHES,PARTICLES,NATIVE = 4,16,93
CALLS,KEY = BRANCHES*LAGS[-1]*NATIVE,0x4279344D
METRICS = ("early18","hidden_delta32_6","hidden_t3_6")


def table_hash(tape):
    h = sha256()
    for key in ("brownian","waiting","mark"):
        array = np.asarray(tape[key])
        h.update(f"{key}:{array.dtype.str}:{array.shape}\n".encode("ascii"))
        h.update(array.tobytes(order="C"))
    return h.hexdigest()


def inverse_gauge(packet,shift,images):
    value = [np.array(a,copy=True) for a in packet[0]]
    st = State(*value)
    box = packet[2][2]
    wrapped = (st.points[:,1]+shift)%box
    seam = wrapped==box
    wrapped[seam] = np.nextafter(box,0.)
    st.points[:,1] = wrapped
    st.unwrapped[:,1] += shift-box*images
    if st.position[0]==0.:
        st.unwrapped[:,1] = wrapped
    taken = slice(int(st.counts[1]))
    edge = st.edges[taken]
    image_changes = images[edge[:,1]]-images[edge[:,0]]
    st.lifts[taken,1] += image_changes
    events = st.events[:st.counts[0]]
    mask = (events[:,2]>=0.)&(events[:,1]!=8.)
    events[mask,11] += image_changes
    st.periodic_shift[:,1] += images-images[st.periodic_parent]
    return (st,*packet[1:]),int(seam.sum())


def camera_orbit(packets,cfg):
    shifts = np.arange(PHASES)*(cfg["motion"]["box"]/PHASES)
    return np.stack([shifted_observables(p,shifts,_camera) for p in packets[1:]])


def history_lift(branch,unit,raw,tables,cfg,query):
    report = branch["lift"]
    b = branch["branch"]
    p = int(rng(KEY,unit,b,0,0).integers(PHASES))
    offset = -p*(cfg["motion"]["box"]/PHASES)
    archives = raw["packets"]
    original_y = archives[0][0].points[:,1]
    box = cfg["motion"]["box"]
    images = np.floor((original_y+offset)/box).astype(np.int64)
    _,seams = inverse_gauge(archives[0],offset,images)
    repaired = []
    if p:
        for snap in archives:
            current,new_seams = inverse_gauge(snap,offset,images)
            repaired.append(current)
            seams += new_seams
    else:
        repaired = archives
    require(report["schema"]==LIFT and report["unit"]==unit and report["inverse_phase_index"]==p
            and report["inverse_displacement"]==offset and report["native_calls"]==0
            and report["original_interval_receipts_unchanged"] is True
            and report["seam_rounding_clamps"]==seams
            and report["original_packet_fingerprints"]==[packet_fingerprint(x) for x in archives]
            and report["canonical_inverse_packet_fingerprints"]==[packet_fingerprint(x) for x in repaired]
            and report["original_tape_hashes"]==[table_hash(x) for x in tables],
            "The stationary lift should start at one of 16 independently drawn inverse phases of the physical parent.")
    np.testing.assert_array_equal(report["initial_y_image_numbers"],images)
    for j,snap in enumerate(repaired):
        check_packet(snap,cfg,cfg["prefix_clock"][j])
    used = [(int(sum(r["tape"]["waits_per_slab"])),sum(not x["censored"] for x in r["spans"]))
            for r in raw["intervals"]]
    np.testing.assert_array_equal(report["waits_and_marks_verified_by_interval"],used)
    camera = camera_orbit(repaired,cfg)
    close(camera[:,p],raw["clean"])
    phase_log = -.5*np.square((camera-query[:,None,:])/cfg["noise"]).sum(axis=-1)
    _,marginal = phase_log_masses(phase_log)
    return marginal,report["canonical_inverse_packet_fingerprints"]


def packet_dag(path,cfg):
    z = np_load(path)
    ids,families,rows = (z[k] for k in ("packet_ids","packet_family","packet_row"))
    require(len(ids)==len(families)==len(rows)==len(set(ids)), "Raw native endpoint DAG lost a stable packet key.")
    columns = tuple("state_"+k for k in STATE_FIELDS)+tuple("packet_"+k for k in PACKET_FIELDS[1:])
    data = {}
    for ident,g,i in zip(ids,families,rows,strict=True):
        values = tuple(z[f"g{g}_{k}"][i] for k in columns)
        p = (State(*values[:len(STATE_FIELDS)]),*values[len(STATE_FIELDS):])
        require(packet_fingerprint(p)==ident,"Native endpoint packet bytes do not match their fingerprint.")
        check_packet(p,cfg,p[0].position[0])
        data[ident] = p
    return data


def hidden_orbit(packet,cfg):
    """Compute component sizes and lattice-cycle inconsistency once, then all apertures."""
    st,recipe,geom,*_ = packet
    n,a = len(st.points),int(recipe[0])
    adjacency = [[] for _ in range(n)]
    for (i,j),vector in zip(st.edges[:st.counts[1]],st.lifts[:st.counts[1]],strict=True):
        adjacency[i].append((j,vector))
        adjacency[j].append((i,-vector))
    assigned = np.full(n,-1,dtype=int)
    image = np.zeros((n,3),dtype=int)
    winding = np.zeros(n,bool)
    for i in range(n):
        if assigned[i]>=0:
            continue
        empty = [i]
        assigned[i] = i
        while empty:
            j = empty.pop()
            for k,vector in adjacency[j]:
                guess = image[j]+vector
                if assigned[k]<0:
                    assigned[k],image[k] = i,guess
                    empty.append(k)
                else:
                    winding[i] |= bool(np.any(image[k]!=guess))
    owner = st.owner.reshape(n,2)>=0
    assigned_am = owner[:a,0]&st.present[:a]
    sizes = np.bincount(assigned[:a][assigned_am],minlength=n)[assigned[:a]]
    in_finite = assigned_am & ~winding[assigned[:a]]
    pendant = (owner[a:].sum(axis=1)==1)&(st.available.reshape(n,2)[a:].sum(axis=1)==1)&st.present[a:]
    box = geom[2]
    offsets = np.arange(PHASES)*(box/PHASES)
    xy = st.points[None,:,:2].copy()
    x,y = xy[...,0],xy[...,1]
    dy = (((y+offsets[:,None])%box)-4.35+box/2.)%box-box/2.
    result = []
    for center in (1.45,4.45):
        dx = (x-center+box/2.)%box-box/2.
        w = np.maximum(1.-abs(dx)/1.3,0.)*np.maximum(1.-abs(dy)/1.3,0.)
        denom = w @ (st.present*np.r_[np.ones(a),np.full(n-a,2.17)])
        finite = w[:,:a]*in_finite
        counts = a*finite.sum(axis=1)
        result.append(np.column_stack((
            np.divide(w[:,:a]@assigned_am,denom,out=np.zeros(PHASES),where=denom>0.),
            np.divide(finite@sizes,counts,out=np.zeros(PHASES),where=counts>0.),
            np.divide(2.17*(w[:,a:]@pendant),denom,out=np.zeros(PHASES),where=denom>0.),
        )))
    return np.stack(result,axis=1)


def phase_stages(record,reference):
    prev = np.asarray(record["initial_log_weights"])
    require(record["mode"]=="pgas" and record["particles"]==PARTICLES and record["phase_count"]==PHASES
            and prev.shape==(PARTICLES,) and len(record["stages"])==2,
            "Original phase-stationary chain must keep the frozen P=16 sweep.")
    if reference is not None:
        close(prev[-1],reference)
    prefix = prev.copy()
    for i,stage in enumerate(record["stages"],start=1):
        x = np.asarray(stage["incremental_log_weights"])
        g = np.asarray(stage["graft_phase_logl"])
        require(x.shape==(PARTICLES,) and g.shape==(PARTICLES,PHASES)
                and stage["camera"]==i+1 and stage["reference_first_replay_reused"] is True
                and np.isfinite(x).all() and np.isfinite(g).all() and np.max(g)<=2.e-13,
               "Omitted any complete common-phase graft likelihood or path.")
        close(stage["prefix_marginal_logl"],prefix)
        log_mass = logsumexp(g,axis=-1)-np.log(PHASES)
        close(stage["graft_marginal_logl"],log_mass)
        close(stage["ancestor_log_weight"],prev+log_mass-prefix)
        normal = normalized(prev)
        a_weight = normalized(prev+log_mass-prefix)
        for key,prob,acid in (("ordinary",normal,stage["ordinary_ancestors"]),
                            ("reference",a_weight,[stage["reference_ancestor"]])):
            close(stage[key+"_probabilities"],prob)
            close(stage[key+"_ancestor_ess"],1./(prob@prob))
            np.testing.assert_array_equal(categorical(prob,stage[key+("_uniforms" if key=="ordinary" else "_uniform")]),acid)
        j = stage["reference_ancestor"]
        if i==2:
            close(x[-1],log_mass[j]-prefix[j])
        prefix = prefix[np.r_[stage["ordinary_ancestors"],j]]+x
        prev = x
    terminal = normalized(prev)
    close(record["terminal_weights"],terminal)
    close(record["selection_probabilities"],terminal)
    choice = record["selected_index"]
    np.testing.assert_array_equal(categorical(terminal,record["selection_uniform"]),[choice])
    phase_book = np.asarray(record["terminal_phase_weights"])
    require(phase_book.shape==(PARTICLES,PHASES) and np.min(phase_book)>=0.,
            "Final branch missing full one-phase conditional weights.")
    close(phase_book.sum(axis=1),np.ones(PARTICLES))
    close(record["selected_phase_probabilities"],phase_book[choice])
    np.testing.assert_array_equal(categorical(phase_book[choice],record["phase_uniform"]),[record["selected_phase_index"]])
    require(record["native_calls"]==NATIVE and record["calls"]=={"attempted":NATIVE,"completed":NATIVE,"failed":0},
            "Native chain omitted attempted counts or reused costs.")
    one,two = record["stages"]
    row2 = np.r_[two["ordinary_ancestors"],two["reference_ancestor"]][choice]
    row1 = np.r_[one["ordinary_ancestors"],one["reference_ancestor"]][row2]
    return float(record["initial_log_weights"][row1]),int(row1),int(row2)


def address_uniforms(cfg,unit,branch,index,record):
    """Replay only the RNG distributions, never a Brownian or chemical update."""
    random = rng(KEY,unit,branch,1,index)
    blocks = [len(intervals(cfg,camera))-1 for camera in range(3)]
    n = cfg["recipe"]["aam"]+cfg["recipe"]["mbaa"]
    def consume(c):
        random.standard_normal(2*blocks[c]*n*3)
        count = blocks[c]*ATTEMPTS
        random.exponential(size=count)
        random.integers(CELLS,size=count,dtype=np.int64)
    for _ in range(PARTICLES-1):
        draw_initial(cfg,random)
        consume(0)
    for k,stage in enumerate(record["stages"],start=1):
        np.testing.assert_array_equal(stage["ordinary_uniforms"],random.random(PARTICLES-1))
        for _ in range(PARTICLES-1):
            consume(k)
        np.testing.assert_array_equal(stage["reference_uniform"],random.random(1))
    np.testing.assert_array_equal(record["selection_uniform"],random.random(1))
    np.testing.assert_array_equal(record["phase_uniform"],random.random(1))


def cross_laws(atoms,w,other,v):
    a = np.asarray(atoms).reshape(len(w),-1)
    b = np.asarray(other).reshape(len(v),-1)
    return (w @ cdist(a,b) @ v)/np.sqrt(a.shape[1])


def extract(local):
    a = np.asarray(local)
    return (a,a[...,2,:,:]-a[...,1,:,:],a[...,2,:,:])


def cross_vectors(a,w,b,v):
    return np.array([cross_laws(p,w,q,v) for p,q in zip(extract(a),extract(b),strict=True)])


def endpoint(packed,cfg,query,ledger,initial_marginal,readout,used):
    ids = np.asarray(packed["path_ids"])
    require(ids.shape==(PARTICLES,4),"Endpoint did not retain all native particles and ordered frames.")
    cameras,hidden = [],[]
    for row in ids:
        path = []
        for idx,ident in enumerate(row):
            packet = readout["packet"][ident]
            require(packet[0].position[0]==cfg["prefix_clock"][idx],"Endpoint packet at wrong native camera age.")
            if path:
                front,tail = path[-1][0],packet[0]
                events,edges = map(int,front.counts[:2])
                np.testing.assert_array_equal(tail.events[:events],front.events[:events])
                np.testing.assert_array_equal(tail.edges[:edges],front.edges[:edges])
                np.testing.assert_array_equal(tail.lifts[:edges],front.lifts[:edges])
            path.append(packet)
            used.add(ident)
            if idx and ident not in readout["local"]:
                phases = np.arange(PHASES)*(cfg["motion"]["box"]/PHASES)
                readout["local"][ident] = hidden_orbit(packet,cfg)
                readout["camera"][ident] = shifted_observables(packet,phases,_camera)
        cameras.append(np.stack([readout["camera"][h] for h in row[1:]]))
        hidden.append(np.stack([readout["local"][h] for h in row[1:]],axis=1))
    clean,local = np.stack(cameras),np.stack(hidden)
    close(packed["clean_orbit"],clean)
    close(packed["local"],local)
    log_phase = -.5*np.square((clean-query[None,:,None,:])/cfg["noise"]).sum(axis=-1)
    w,phase,j,_,_=phase_sweep(ledger,"pgas",PARTICLES,[initial_marginal,np.nan,np.nan],log_phase)
    close(packed["terminal_weights"],w)
    close(packed["phase_weights"],phase)
    require(tuple(ids[j])==tuple(ledger["selected_path_ids"]),"Selected terminal is absent from frozen DAG.")
    w = (w[:,None]*phase).ravel()
    close(w.sum(),1.)
    return local.reshape(-1,3,2,3),w


def audit_mother(data,raw,query,work,p,cfg):
    unit = data["unit"]
    require(unit==work["unit"] and data["native_calls"]==work["native_calls"]==CALLS
            and np.isfinite(data["seconds"]) and data["seconds"]>0.,
            "A stationary mother omitted charged sweeps or raw receipt.")
    _,tables = generating_state(cfg,unit,query,raw)
    dag = data["packet_dag"]
    require(dag==work["packet_dag"],"Packet DAG provenance was replaced at scoring.")
    folder = Path(work["oracle"]["path"]).parent
    packets = packet_dag(artifact(dag,folder/f"{unit}_PACKETS.npz"),cfg)
    cache = {"packet":packets,"camera":{},"local":{}}
    zero = data["phase_only"]
    phases = np.arange(PHASES)*(cfg["motion"]["box"]/PHASES)
    original_orbit = camera_orbit(raw["packets"],cfg)
    only_local = np.stack([hidden_orbit(a,cfg) for a in raw["packets"][1:]],axis=1)
    only_phase = normalized(-.5*np.square((original_orbit-query[:,None,:])/cfg["noise"]).sum(axis=(0,2)))
    close(zero["offsets"],phases)
    close(zero["clean_orbit"],original_orbit)
    close(zero["phase_weights"],only_phase)
    close(zero["local"],only_local)
    lows = {"0":.5*cross_vectors(only_local,only_phase,only_local,only_phase)}
    branches = data["branches"]
    require([b["branch"] for b in branches]==list(range(BRANCHES)),"Any changed branch invalidates conditional independence.")
    sets = {str(k):[] for k in LAGS}
    used = set()
    for branch in branches:
        b = branch["branch"]
        ref,ref_ids = history_lift(branch,unit,raw,tables,cfg,query)
        first = ref[0]
        ledger = branch["steps"]
        require(set(branch["endpoints"])==set(sets) and len(ledger)==LAGS[-1],
                "Missing preregistered short-sweep or endpoint data.")
        for k,step in enumerate(ledger,start=1):
            ids = tuple(step["selected_path_ids"])
            require(step["step"]==k and len(ids)==4 and all(len(x)==64 for x in ids),
                    "A short chain changed ordered native time keys.")
            address_uniforms(cfg,unit,b,k,step)
            revised,first_ancestor,_ = phase_stages(step,first)
            if first_ancestor==PARTICLES-1:
                require(ids[:2]==tuple(ref_ids)[:2],"The selected ancestral prefix changed its held base law.")
            if k in LAGS:
                sets[str(k)].append(endpoint(branch["endpoints"][str(k)],cfg,query,step,first,cache,used))
            first,ref_ids = revised,ids
    require(used==set(packets),"Some alleged endpoint packets lack a frozen lineage.")
    pairs = {}
    for k,path_laws in sets.items():
        pair = [cross_vectors(x,w,y,v) for (x,w),(y,v) in combinations(path_laws,2)]
        require(len(pair)==6,"Only pairs of distinct independent branches belong in the lower risk integral.")
        lows[k] = .5*np.mean(pair,axis=0)
        pairs[k] = np.stack(pair)
    return only_local[0],lows,pairs


def validate_parent(p,queries): 
    path = Path(p["atlas_model"])
    atlas = path.parent
    locks = [atlas/f for f in ("REGISTERED.json","TRAINED_AND_SELECTED.json","PREDICTIONS_FROZEN.json","INDEPENDENT_AUDIT.json")]
    original,selection,forecast,receipt = map(read,locks)
    require(receipt["passed"] is True and receipt["registered_sha256"]==sha(locks[0])
            and receipt["forecast_sha256"]==sha(locks[2])
            and original["units"]["train"]==selection["train_ids"]==p["atlas_mothers"]
            and len(set(p["atlas_mothers"]))==512 and original["physics"]==p["physics"]
            and p["fixed_kernel"]==selection["selected"]["kernel"]
            and forecast["held_targets_opened"] is False
            and not set(UNITS).intersection(sum(original["units"].values(),[])),
            "The blind control must be the old audited, fixed same-information kernel on a disjoint cohort.")
    candidates = [Path(name) for name in p["inputs"] if Path(name).name=="REGISTERED.json"]
    wide = [d.parent for d in candidates if read(d).get("namespace")==WIDE]
    require(len(wide)==1,"Missing prior independently closed widest inference comparison.")
    wide = wide[0]
    p0, score, a = map(lambda key:read(wide/key), ("REGISTERED.json","SCORED.json","INDEPENDENT_AUDIT.json"))
    require(a["passed"] is True and a["registration_sha256"]==sha(wide/"REGISTERED.json")
            and a["scored_sha256"]==sha(wide/"SCORED.json") and score["go_to_inverse"] is False
            and not set(UNITS).intersection(sum(p0["units"].values(),[])),
            "The accuracy study was not closed or its consumed cases were recycled.")
    model = load_tree(path,p["inputs"][str(path)])
    predict,_ = independently_predict(model,queries)
    laws = {}
    for key in ("kernel","empirical"):
        kind,val = predict(key,p["fixed_kernel"] if key=="kernel" else None)
        require(kind=="weight","Expected frozen solely query-based kernel/reference laws.")
        laws[key] = val
    return model["bank"]["local"],laws


def audit(out):
    names = ("REGISTERED.json","RUN.json","held/STARTED.json","ACQUIRED.json",
             "PREDICTIONS_FROZEN.json","oracle/STARTED.json","ORACLE_COMPLETE.json","SCORE_STARTED.json","SCORED.json")
    files = [out/name for name in names]
    p,r,hs,a,f,os,c,started,s = docs = list(map(read,files))
    times = [datetime.fromisoformat(row["actual_utc"]) for row in docs]
    require(all(lo<hi for lo,hi in zip(times[:-1],times[1:],strict=True))
            and all(t==datetime.fromisoformat(row["asia_shanghai"]) for t,row in zip(times,docs,strict=True)),
            "Blind and stationary truth stages did not have separate, chronological receipts.")
    reg,freeze,oracle = (sha(files[k]) for k in (0,4,6))
    require(p["namespace"]==NS and f["truth_opened"] is False and tuple(f["units"])==tuple(p["units"])==UNITS
            and tuple(p["lags"])==LAGS and tuple(p["metrics"])==tuple(s["metrics"])==METRICS
            and p["audit_key"]==KEY and p["addresses"]=={"uniform_inverse_phase":0,"pgas_sweep":1}
            and (p["branches"],p["particles"],p["sweep_calls"],p["phases"])==(BRANCHES,PARTICLES,NATIVE,PHASES)
            and p["truth_calls"]==a["native_calls"]==s["truth_native_calls"]==3*len(UNITS)
            and p["mother_oracle_calls"]==CALLS and p["oracle_calls"]==c["native_calls"]==s["mothers_native_oracle"]==CALLS*len(UNITS)
            and s["independent_mothers"]==len(UNITS) and p["practical_fraction_of_frozen_kernel"]==.05
            and p["forecast_calls"]==f["native_calls"]==s["frozen_forecast_native_calls"]==0
            and p["new_fits"]==s["new_fits"]==p["new_wet_measurements"]==s["new_wet_measurements"]==0
            and p["inverse_scored"] is s["inverse_scored"] is False
            and reg==r["registration_sha256"]==hs["registration_sha256"]==a["registration_sha256"]
                    ==f["registration_sha256"]==s["registration_sha256"]
            and f["acquired_sha256"]==sha(files[3]) and os["first_truth_reveal"] is True
            and freeze==os["freeze_sha256"]==c["freeze_sha256"]
            and oracle==started["oracle_receipt_sha256"]==s["oracle_complete_sha256"]
            and r["registered_before_native"] is True,"A hidden truth was used as a forecast or the fixed diagnostic scope changed.")
    for path,digest in p["inputs"].items():
        require(sha(path)==digest,"A locked independent research receipt was overwritten.")
    for path,digest in p["sources"].items():
        require(sha(out/"SOURCE_SNAPSHOT"/Path(path).name)==digest,"A bound audit source changed.")
    train_rows, oracle_rows = a["mothers"],c["mothers"]
    require(tuple(x["unit"] for x in train_rows)==tuple(x["unit"] for x in oracle_rows)==UNITS
            and [j["query"]["sha256"] for j in train_rows]==f["queries"],"Outcome/readout bindings changed.")
    cfg = p["physics"]
    q = []
    for row in train_rows:
        fn = out/"held"/f'{row["unit"]}_QUERY.npz'
        y = np_load(artifact(row["query"],fn))
        require(set(y)=={"measured"} and y["measured"].shape==(3,4) and np.isfinite(y["measured"]).all(),
                "Blind forecast view contains data beyond the genuine three noisy camera rows.")
        q.append(y["measured"])
    atlas,weights = validate_parent(p,np.stack(q))
    f_laws = np_load(artifact(f["weights"],out/"BLIND_WEIGHTS.npz"))
    require(set(f_laws)=={"query","kernel","empirical"},"Hidden inputs entered a freeze law.")
    np.testing.assert_array_equal(q,f_laws["query"])
    for k,w in weights.items():
        close(f_laws[k],w)
    charged = load_tree(artifact(s["arrays"],out/"SCORE_ARRAYS.npz"))
    results = {"kernel":[],"empirical":[]}
    lower_by_lag = {str(k):[] for k in (0,*LAGS)}
    for i,(row,oracle_row,query,archived) in enumerate(zip(train_rows,oracle_rows,q,charged["rows"],strict=True)):
        unit = row["unit"]
        require(archived["unit"]==unit and row["native_calls"]==3
                and oracle_row["native_calls"]==archived["native_calls"]==CALLS,
                "Expected one immutable acquisition and every independent branch from that mother.")
        raw = load_tree(artifact(row["truth"],out/"held"/f"{unit}_TRUTH.npz"))
        data = load_tree(artifact(oracle_row["oracle"],out/"oracle"/f"{unit}_ORACLE.npz"))
        require(data["truth_sha256"]==row["truth"]["sha256"] and data["query_sha256"]==row["query"]["sha256"],
                "A mathematical oracle did not start from the genuine sealed held generator.")
        for folder,item,key in (("held",row,"query"),("oracle",oracle_row,"oracle")):
            ticket = read(out/folder/f"{unit}_RECEIPT.json")
            require(ticket[key]==item[key] and ticket["unit"]==unit and ticket["native_calls"]==item["native_calls"],
                    "Independent mother artifact lost original actual closure receipt.")
        truth,lows,pairs = audit_mother(data,raw,query,oracle_row,p,cfg)
        close(archived["seconds"],data["seconds"])
        for k,val in lows.items():
            close(archived["lower_bounds"][k],val)
            lower_by_lag[k].append(val)
        for k,val in pairs.items():
            close(archived["independent_pair_distances"][k],val)
        for mode,assigned in weights.items():
            w = assigned[i]
            actual_risk = cross_vectors(atlas,w,truth[None],np.ones(1))-.5*cross_vectors(atlas,w,atlas,w)
            close(archived["risks"][mode],actual_risk)
            results[mode].append(actual_risk)
    for k,x in lower_by_lag.items():
        close(charged["lower"][k],x)
    for k,x in results.items():
        close(charged["risk"][k],x)
    practical = []
    for k,name in enumerate(METRICS):
        risk = np.asarray(results["kernel"])[:,k]
        lower = np.asarray(lower_by_lag["32"])[:,k]
        report = s["summary"][name]
        close(report["frozen_kernel_risk"],intervals95(risk))
        close(report["empirical_risk"],intervals95(np.asarray(results["empirical"])[:,k]))
        for lag,value in lower_by_lag.items():
            close(report["risk_lower_bound_by_lag"][lag],intervals95(np.asarray(value)[:,k]))
        close(report["kernel_minus_risk_lower_bound"],intervals95(risk-lower))
        close(report["paired_lag32_minus_lag12"],intervals95(lower-np.asarray(lower_by_lag["12"])[:,k]))
        gap = intervals95(.95*risk-lower)
        close(report["practical_margin_upper_gap"],gap)
        requires = bool(gap[2]<0.)
        require(report["fractional_improvement_ceiling_below_margin"] is requires,
                "Finite diagnostic cannot alter the registered practical uncertainty rule.")
        practical.append(requires)
    return {**stamp(),"passed":True,"registration_sha256":reg,"scored_sha256":sha(files[-1]),
            "oracle_complete_sha256":oracle,"forecast_sha256":freeze,"new_native_calls":0,
            "new_fits":0,"new_wet_measurements":0,"mother_units":len(UNITS),
            "audited_role_addressed_sweeps":len(UNITS)*BRANCHES*LAGS[-1],
            "audited_oracle_native_calls":p["oracle_calls"],"same_truth_only_conditional_branch_start":True,
            "raw_dag_and_frozen_phase_weights_recomputed":True,
            "finite_law":fixtures(),"nominal_headroom_below_five_percent":dict(zip(METRICS,practical,strict=True)),
            "scope":"Oracle diagnostics of unchanged simulated path geometry, not a posterior fitted from new material or blind inverse performance."}


def fixtures():
    one,two = Fraction(2,5),Fraction(3,5)
    posterior = (one,two)
    bayes_risk = Fraction(2)*one*two
    values = {}
    for lag in (0,1,2,4):
        remain = Fraction(1,2)**lag
        transition = [[remain*int(i==j)+(1-remain)*posterior[j] for j in (0,1)] for i in (0,1)]
        assert [sum(posterior[i]*transition[i][j] for i in (0,1)) for j in (0,1)]==list(posterior)
        estimate = 2*sum(w*r[0]*r[1] for w,r in zip(posterior,transition,strict=True))
        assert estimate==bayes_risk*(1-remain**2) and estimate<=bayes_risk
        values[lag] = str(estimate)
    known = np.zeros((1,3,2,3))
    alternative = known+1.
    close(cross_vectors(known,np.ones(1),alternative,np.ones(1)),[1.,0.,1.])
    return {"passed":True,"native_calls":0,"stationarity_fraction_risk":str(bayes_risk),
            "lower_lags":values,"finite_law_covariance_decomposition":True}


if __name__=="__main__":
    parser = ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--fixtures",action="store_true")
    mode.add_argument("--out",type=Path)
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        if args.fixtures:
            print(json.dumps(fixtures(),indent=2))
        else:
            out = args.out.resolve(strict=True)
            write(out/"INDEPENDENT_AUDIT.json",audit(out))
            print("PASS independent stationary phase bound",out)
