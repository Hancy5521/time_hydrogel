"""Independent frozen readback of a full same-law legal phase-bank comparison."""

from argparse import ArgumentParser
from hashlib import sha256
from pathlib import Path

import numpy as np
from scipy.spatial.distance import cdist
from scipy.special import logsumexp
from threadpoolctl import threadpool_limits

from aam_formation_cycle_driver import np_load,read,sha,write
from aam_formation_path_cycle import load_tree,stamp
from audit_aam_endpoint_tape_move import artifact,close,generating_state,intervals95,require,rng
from audit_aam_stationary_distillation import baseline_laws,clock,lens
from audit_aam_original_tape_pgas import ATLAS


PHASES = 16
N = 8192
DRAWS = 32768
METRICS = ("early18","hidden_delta32_6")
NS = "aam-large-shared-phase-original-bank-v1"
UNITS = {"bank":list(range(2026145000,2026153192)),"held":list(range(2026154000,2026154384))}
BANK_ANCHORS = (0,85,170,255)
HELD_ANCHORS = (0,1,54,108,162,216,270,383)
NEW = ("phase_gaussian","phase_rbf","raw_gaussian","raw_rbf","phase_uniform")
NESTED = {"phase_gaussian_first512":512,"phase_gaussian_first2048":2048}
GATE_METHODS = (*NEW,*ATLAS)
METHODS = (*GATE_METHODS,*NESTED)
PAIRED_SEED = 592090181


def law(log):
    a = np.asarray(log,dtype=float)
    require(a.ndim>=1 and a.size>0 and np.isfinite(a).all(),"The original Gaussian/RBF support must be finite.")
    w = np.exp(a-logsumexp(a))
    close(w.sum(),1.)
    return w


def transform(local):
    a = np.asarray(local)
    require(a.shape[-3:]==(3,2,3) and np.isfinite(a).all(),
            "Legal support uses the original three frames and hidden views.")
    return a.reshape(-1,18), (a[...,2,:,:]-a[...,1,:,:]).reshape(-1,6)


def selection(w,u):
    p = np.asarray(w,dtype=float).ravel()
    u = np.asarray(u,dtype=float)
    require(np.isfinite(p).all() and np.all(p>=0.) and np.all((u>=0.)&(u<1.)),
            "Monte Carlo integrals require a normalized legal law and Uniform[0,1) draws.")
    close(p.sum(),1.)
    positive = np.flatnonzero(p>0.)
    c = np.minimum(np.cumsum(p[positive]),1.)
    c[-1] = 1.
    return positive[np.searchsorted(c,u,side="right")]


def component_distance(a,b):
    return np.linalg.norm(a-b,axis=-1)/np.sqrt(a.shape[-1])


def independent_energy(local,w,truth,pair_uniforms,*,keep_samples=False):
    """Full weighted first integral, unbiased independent-index pair integral."""
    support = transform(local)
    target = transform(truth)
    probability = np.asarray(w).ravel()
    require(len(probability)==len(support[0]) and len(target[0])==1,
            "One independent physical truth should meet one untouched forecast law.")
    i,j = (selection(probability,pair_uniforms[:,k]) for k in (0,1))
    first_terms = [probability @ (cdist(x,y)[:,0]/np.sqrt(x.shape[1]))
                   for x,y in zip(support,target,strict=True)]
    second_terms = [component_distance(x[i],x[j]) for x in support]
    values = np.asarray(first_terms)-.5*np.asarray([a.mean() for a in second_terms])
    error = np.asarray([.5*a.std(ddof=1)/np.sqrt(len(a)) for a in second_terms])
    return (values,error,np.stack(second_terms,axis=1)) if keep_samples else (values,error)


def ess(w):
    a = np.asarray(w,dtype=float)
    c = a.reshape(len(a),-1)
    close(a.sum(),1.)
    require(np.isfinite(a).all() and np.all(a>=0.),"Kish effective number uses actual law.")
    return np.array([1/np.square(c.sum(axis=1)).sum(),1/np.square(a).sum()])


def source_base(trio,y,scale):
    """Original independent product over the same four noisy channels and three times."""
    x = (np.asarray(trio)-np.asarray(y))/scale
    require(x.shape[-2:]==(3,4) and np.isfinite(x).all(),"A camera is not the original four-channel payload.")
    return .5*np.sum(np.square(x),axis=(-1,-2))


def checked_path(reference,folder):
    p = Path(reference["path"]).resolve(strict=True)
    require(p.parent==folder.resolve(strict=True),"Artifact escaped its assigned data role.")
    return artifact(reference,p)


def bank_readback(out,cfg,receipt):
    blocks = receipt["rows"]
    folder = out/"bank"
    require(len(blocks)==N//32 and [int(u) for row in blocks for u in row["units"]]==UNITS["bank"],
            "Native archive bank must use every registered independent full trajectory.")
    result = []
    for i,block in enumerate(blocks):
        path0,path1 = (checked_path(block[k],folder) for k in ("raw","arrays"))
        observed = np_load(path1)
        ids = block["units"]
        require(set(observed)=={"unit","clean","local","measured"} and len(ids)==32
                and observed["unit"].tolist()==ids and observed["clean"].shape==(32,PHASES,3,4)
                and observed["local"].shape==(32,PHASES,3,2,3) and observed["measured"].shape==(32,3,4)
                and all(np.isfinite(observed[k]).all() for k in ("clean","local","measured"))
                and np.all((observed["local"]>=0.)&(observed["local"]<=1.+2.e-14)),
                "Frame, phase or physical graph support differs from the registered shared likelihood.")
        if i in BANK_ANCHORS:
            raw = load_tree(path0)
            require([d["unit"] for d in raw["rows"]]==ids,"Raw complete paths misplaced in archive order.")
            for j,record in enumerate(raw["rows"]):
                y = record["measured"]
                np.testing.assert_array_equal(y,observed["measured"][j])
                generating_state(cfg,record["unit"],y,record["raw"])
                cameras,hidden = lens(record["raw"]["packets"],cfg,{})
                close(np.swapaxes(cameras,0,1),observed["clean"][j])
                close(observed["local"][j],hidden)
        result.append(observed)
    return {k:np.concatenate([row[k] for row in result]) for k in ("clean","local","measured")}


def hash_weights(w):
    x = np.ascontiguousarray(w,dtype="<f8")
    return sha256(x.tobytes()).hexdigest()


def native_weights(bank,query,noise,scale,width_squared,settings):
    b = np.asarray(bank["clean"])
    prior_obs = np.asarray(bank["measured"])
    factor,mixture = settings
    d = source_base(b,query,np.asarray(noise))
    shifted_noisy = prior_obs[:,None] + (b-b[:,0:1])
    radial = source_base(shifted_noisy,query,scale)/(factor**2*width_squared)
    gp = law(-d)
    kp = mixture*law(-radial)+(1.-mixture)/radial.size
    g0,r0 = np.zeros_like(gp),np.zeros_like(kp)
    g0[:,0] = law(-d[:,0])
    r0[:,0] = mixture*law(-radial[:,0])+(1.-mixture)/len(radial)
    expected = {k:v.ravel() for k,v in (
        ("phase_gaussian",gp),("phase_rbf",kp),("raw_gaussian",g0),("raw_rbf",r0),
        ("phase_uniform",np.full_like(gp,1./gp.size)))}
    for name,n in NESTED.items():
        q = np.zeros_like(gp)
        q[:n] = law(-d[:n])
        expected[name] = q.ravel()
    return expected


def check_diagnostic(w,stats):
    q = np.asarray(w).reshape(N,PHASES)
    whole = q.sum(axis=1)
    close(q.sum(),1.)
    atom,mother = 1/np.square(q).sum(),1/np.square(whole).sum()
    for key,want in (("atom_ess",atom),("mother_ess",mother),("largest_atom_mass",q.max()),
                     ("largest_mother_mass",whole.max()),("phase_marginal",q.sum(axis=0))):
        close(stats[key],want)
    require(stats["positive_atoms"]==np.count_nonzero(q)
            and stats["positive_mothers"]==np.count_nonzero(whole),
            "Reported support counted phase rotations as independent mothers.")


def exact_neighbor_score(local,w,truth):
    ans = []
    for x,y in zip(transform(local),transform(truth),strict=True):
        d = cdist(x,y)[:,0]
        e = cdist(x,x)
        ans.append((w@d-.5*w@e@w)/np.sqrt(x.shape[1]))
    return np.asarray(ans)


def score_contrasts(main,replica):
    require(set(main)==set(replica)==set(METHODS),"Only every prespecified method can support the final gate.")
    teacher = main["phase_gaussian"]
    expected,replicated = {},{}
    for m,a in main.items():
        v = replica[m]
        require(a.shape==(len(UNITS["held"]),2) and v.shape==(len(a),2,2)
                and np.isfinite(v).all() and np.isfinite(a).all(),
                "Monte Carlo sampling error or independent mothers were lost.")
        close(a,v.mean(axis=1))
        if m!="phase_gaussian":
            expected[m] = intervals95(a-teacher)
            replicated[m] = (v-replica["phase_gaussian"]).mean(axis=0)
    pass_all = bool(all(np.all(expected[k][:,1]>0.) and np.all(replicated[k]>0.) for k in GATE_METHODS[1:]))
    return expected,replicated,pass_all


def score_anchor(local,actual,methods,where,replicas,mc,gain_mc):
    u = UNITS["held"][where]
    output, toss = {},{}
    for m,w in methods.items():
        atoms = local[m]
        if m in ATLAS:
            score = exact_neighbor_score(atoms,np.asarray(w).ravel(),actual[None])
            for k in (0,1):
                close(replicas[m][where,k],score)
                close(mc[m][where,k],np.zeros(2))
        else:
            scores,errors,saved = [],[],[]
            for k in (0,1):
                rows = rng(PAIRED_SEED,u,k).random((DRAWS,2))
                v,se,d = independent_energy(atoms,w,actual[None],rows,keep_samples=True)
                close(replicas[m][where,k],v)
                close(mc[m][where,k],se)
                scores.append(v)
                errors.append(se)
                saved.append(d)
            output[m] = [scores,errors]
            toss[m] = np.stack(saved)
    for m,v in gain_mc.items():
        if m in ATLAS:
            expected = mc["phase_gaussian"][where]
        else:
            witness = toss[m]-toss["phase_gaussian"]
            expected = .5*witness.std(axis=1,ddof=1)/np.sqrt(DRAWS)
        close(v[where],expected)
    return output


def read_original_atlas(p):
    bank = Path(p["atlas_model"])
    root = bank.parent
    original,selected,independent = (read(root/name) for name in
              ("REGISTERED.json","TRAINED_AND_SELECTED.json","INDEPENDENT_AUDIT.json"))
    require(original["physics"]==p["physics"] and original["units"]["train"]==selected["train_ids"]==p["atlas_units"]
            and len(set(p["atlas_units"]))==512 and independent["passed"] is True
            and independent["registered_sha256"]==sha(root/"REGISTERED.json")
            and independent["forecast_sha256"]==sha(root/"PREDICTIONS_FROZEN.json")
            and not set(sum(UNITS.values(),[])).intersection(sum(original["units"].values(),[]))
            and p["atlas_settings"]==selected["selected"],"Original baselines must still use the same frozen old-law library.")
    return load_tree(bank,p["inputs"][str(bank)])


def read_three_camera(out,cfg,acquired):
    rows = acquired["rows"]
    require([r["unit"] for r in rows]==UNITS["held"] and all(r["native_calls"]==3 for r in rows),
            "Some registered independent held mothers were lost or replaced.")
    query,targets,truth = [],[],{}
    folder = out/"held"
    for i,row in enumerate(rows):
        obs = np_load(checked_path(row["obs"],folder))
        raw = checked_path(row["raw"],folder)
        require(set(obs)=={"measured"} and obs["measured"].shape==(3,4) and np.isfinite(obs["measured"]).all(),
                "Only original three-camera float64 views belong to forecast descriptors.")
        y = obs["measured"]
        query.append(y)
        targets.append(row["obs"]["sha256"])
        if i in HELD_ANCHORS:
            state = load_tree(raw)
            generating_state(cfg,row["unit"],y,state)
            _,local = lens(state["packets"],cfg,{})
            truth[i] = local[0]
    return np.stack(query),targets,truth


def check_anchors(bank,old,p,query,truth,rows,frozen,archive):
    phases = bank["local"]
    content = {k:(old["bank"]["local"] if k in ATLAS else phases) for k in METHODS}
    names = set(content)
    certs = {}
    old_weights = baseline_laws(old,p,query[list(HELD_ANCHORS)])
    require(set(frozen)=={str(UNITS["held"][i]) for i in HELD_ANCHORS},
            "The fixed independent raw readback subset changed after forecasts.")
    for j,i in enumerate(HELD_ANCHORS):
        r = rows[i]
        u = r["unit"]
        expected = native_weights(bank,query[i],p["physics"]["noise"],old["camera_scale"],old["kernel_sq_scale"],
                                  p["atlas_settings"]["kernel"])
        expected.update({k:w[j] for k,w in old_weights.items()})
        loaded = frozen[str(u)]
        require(set(loaded)==set(r["weights_digests"])==names and set(r["diagnostics"])>=set(NEW),
                "Frozen weight anchors fail to bind complete probability/phase hashes.")
        for name,w0 in expected.items():
            w1 = np.asarray(loaded[name])
            require(w1.shape==w0.shape and np.isfinite(w1).all() and np.min(w1)>=0.
                    and r["weights_digests"][name]==hash_weights(w1),
                    "Actual stored float64 probability digest changed.")
            close(w1,w0,atol=2.e-10)
            close(w1.sum(),1.)
            if name in NEW:
                check_diagnostic(w1,r["diagnostics"][name])
            if name in NESTED:
                check_diagnostic(w1,r["nested_diagnostics"][name])
                np.testing.assert_array_equal(w1.reshape(N,PHASES)[NESTED[name]:],0.)
        plain = native_weights(bank,query[i],p["physics"]["noise"],old["camera_scale"],old["kernel_sq_scale"],
                               (p["atlas_settings"]["kernel"][0],1.))
        for k in ("phase_rbf","raw_rbf"):
            check_diagnostic(plain[k],r["diagnostics"][k+"_unmixed"])
        certs[str(u)] = score_anchor(content,truth[i],loaded,i,archive["replica"],archive["mcse"],archive["gain_mcse"])
    return certs


def audit(out):
    names = ("REGISTERED","RUN","BANK_ACQUISITION_STARTED","BANK_ACQUIRED","BANK_FROZEN",
             "HELD_ACQUISITION_STARTED","HELD_ACQUIRED","FORECAST_DESCRIPTORS","PREDICTIONS_FROZEN","SCORED")
    files = {name:out/f"{name}.json" for name in names}
    d = {name:read(path) for name,path in files.items()}
    ticks = list(map(clock,d.values()))
    require(all(a<b for a,b in zip(ticks[:-1],ticks[1:],strict=True)),
            "Registered bank, blind query and frozen score stages did not run in their declared order.")
    p,run,b0,b1,b2,h0,h1,desc,f,s = d.values()
    r,b,h,forecast = (sha(files[name]) for name in ("REGISTERED","BANK_FROZEN","HELD_ACQUIRED","PREDICTIONS_FROZEN"))
    require(p["namespace"]==NS and p["units"]==UNITS and p["block"]==32 and p["phases"]==PHASES
            and tuple(p["methods"])==tuple(desc["ordered_method_names"])==METHODS
            and tuple(p["gate_methods"])==GATE_METHODS and p["nested_sizes"]==NESTED
            and tuple(p["metrics"])==tuple(s["metrics"])==METRICS and p["pair_samples"]==DRAWS and p["pair_seed"]==PAIRED_SEED
            and tuple(p["audit_held_indices"])==HELD_ANCHORS
            and p["new_fits"]==p["new_wet_measurements"]==s["new_measurements"]==0
            and p["original_native_bank"]==s["native_calls_bank"]==b1["native_calls"]==N*3
            and p["original_native_held"]==s["native_calls_held"]==h1["native_calls"]==len(UNITS["held"])*3
            and s["held_mothers"]==len(UNITS["held"]) and b2["bank_mothers"]==s["bank_mothers"]==N
            and r==run["registered_sha256"]==b0["registered_sha256"] and run["before_native_or_fit"] is True
            and b0["role"]==b1["role"]=="bank" and h0["role"]==h1["role"]=="held"
            and b1["started_sha256"]==sha(files["BANK_ACQUISITION_STARTED"])
            and h1["started_sha256"]==sha(files["HELD_ACQUISITION_STARTED"])
            and p["forecast_native"]==f["forecast_native"]==0 and b==f["bank_sha256"]==h0["bank_sha256"]
            and b2["acquired_sha256"]==sha(files["BANK_ACQUIRED"]) and f["held_acquired_sha256"]==h
            and f["descriptor_sha256"]==sha(files["FORECAST_DESCRIPTORS"])
            and f["held_truth_opened"] is False and f["deterministic_probability_reference"] is True
            and s["held_forecast_sha256"]==forecast,"Frozen legal support and original prospective controls changed.")
    for path,digest in p["inputs"].items():
        require(sha(path)==digest,"Previous frozen provenance/hash no longer matches.")
    for path,digest in p["sources"].items():
        require(sha(out/"SOURCE_SNAPSHOT"/Path(path).name)==digest,"Frozen source dependency changed.")
    parent = [Path(path) for path in p["inputs"] if Path(path).name=="REGISTERED.json"
              and read(path).get("namespace")=="aam-physical-teacher-convex-energy-gradient-v1"]
    require(len(parent)==1,"No single auditable frozen parent comparison accompanies this run.")
    limit = read(parent[0])
    pre = parent[0].parent
    cert = read(pre/"INDEPENDENT_AUDIT.json")
    require(cert["passed"] is True and cert["registered_sha256"]==sha(parent[0])
            and cert["scored_sha256"]==sha(pre/"SCORED.json") and not read(pre/"SCORED.json")["go_to_original_inverse"]
            and not set(sum(UNITS.values(),[])).intersection(sum(limit["units"].values(),[])+limit["train_units"]),
            "Same-law hierarchy used a consumed prior continuation family or identity.")
    old = read_original_atlas(p)
    bank = bank_readback(out,p["physics"],b1)
    combined = np_load(artifact(b2["arrays"],out/"FULL_BANK_ORBITS.npz"))
    require(set(combined)=={"unit","clean","local","measured"} and combined["unit"].tolist()==UNITS["bank"]
            and all(row["native_calls"]==96 for row in b1["rows"]),
            "Complete bank does not reproduce all legal mother identities or original work.")
    for k,a in bank.items():
        np.testing.assert_array_equal(combined[k],a)
    y,hashes,truth = read_three_camera(out,p["physics"],h1)
    rows = desc["rows"]
    require([r["unit"] for r in rows]==UNITS["held"] and [r["observation_sha256"] for r in rows]==hashes,
            "Frozen forecast descriptors used a noncamera held observation or changed ordering.")
    for row in rows:
        require(set(row)=={"unit","observation_sha256","weights_digests","diagnostics","nested_diagnostics"}
                and set(row["weights_digests"])==set(METHODS)
                and set(row["diagnostics"])=={*NEW,"phase_rbf_unmixed","raw_rbf_unmixed"}
                and set(row["nested_diagnostics"])==set(NESTED)
                and all(len(v)==64 and all(c in "0123456789abcdef" for c in v) for v in row["weights_digests"].values()),
                "All frozen method weights and camera-only access must be traceable before scoring.")
        for name,stats in {**row["diagnostics"],**row["nested_diagnostics"]}.items():
            capacity = NESTED.get(name,N)
            require(1.-2.e-12<=stats["mother_ess"]<=capacity+2.e-10
                    and stats["mother_ess"]<=stats["atom_ess"]+2.e-10 and stats["atom_ess"]<=capacity*PHASES+2.e-9
                    and 0.<=stats["largest_atom_mass"]<=stats["largest_mother_mass"]<=1.+2.e-13
                    and 1<=stats["positive_mothers"]<=capacity and 1<=stats["positive_atoms"]<=capacity*PHASES,
                    "Parent/phase support identities cannot be sampled as additional mothers.")
            close(np.sum(stats["phase_marginal"]),1.)
            if name.startswith("raw_"):
                close(stats["atom_ess"],stats["mother_ess"])
                np.testing.assert_array_equal(stats["phase_marginal"][1:],np.zeros(PHASES-1))
    frozen = load_tree(artifact(f["audit_weights"],out/"AUDIT_WEIGHTS.npz"))
    archive = load_tree(artifact(s["arrays"],out/"SCORE_ARRAYS.npz"))
    require(archive["unit"]==UNITS["held"] and archive["mcse"].keys()==archive["losses"].keys()
            and set(archive["gain_mcse"])==set(METHODS)-{"phase_gaussian"}
            and archive["gain_mcse"].keys()==s["aggregate_gain_mcse_by_replica"].keys(),
            "Mother-level score archive has a missing original comparator or unit.")
    gains,mcs,gate = score_contrasts(archive["losses"],archive["replica"])
    for k,v in archive["losses"].items():
        close(s["means"][k],v.mean(axis=0))
        se = np.asarray(archive["mcse"][k])
        require(se.shape==(len(UNITS["held"]),2,2) and np.isfinite(se).all() and np.min(se)>=0.,
                "New Monte Carlo accuracy receipts are inconsistent.")
        if k in ATLAS:
            np.testing.assert_array_equal(se,np.zeros_like(se))
            np.testing.assert_array_equal(archive["replica"][k][:,0],archive["replica"][k][:,1])
        if k!="phase_gaussian":
            close(s["paired_gains"][k],gains[k])
            close(s["mc_replicate_gains"][k],mcs[k])
            g = np.asarray(archive["gain_mcse"][k])
            require(g.shape==se.shape and np.isfinite(g).all() and np.min(g)>=0.,
                    "Paired Monte Carlo uncertainty dropped a replicate or cohort.")
            close(s["aggregate_gain_mcse_by_replica"][k],np.sqrt(np.square(g).sum(axis=0))/len(g))
    require(s["go_to_original_inverse"] is gate,"The original comparative gate did not independently reproduce.")
    check_anchors(bank,old,p,y,truth,rows,frozen,archive)
    return {**stamp(),"passed":True,"registration_sha256":r,"scored_sha256":sha(files["SCORED"]),
            "forecast_sha256":forecast,"bank_mothers":N,"held_mothers":len(UNITS["held"]),
            "all_archive_hashes_verified":True,"raw_bank_anchors":32*len(BANK_ANCHORS),
            "raw_bank_blocks_checked":list(BANK_ANCHORS),"raw_held_anchors":len(HELD_ANCHORS),
            "raw_held_indices":list(HELD_ANCHORS),"pair_repeats_checked_per_raw_held":2,
            "pair_draws_per_repeat":DRAWS,"paired_score_and_gate_recomputed_all_mothers":True,
            "nested_mother_support_not_primary":NESTED,"paired_monte_carlo_accuracy_rechecked":True,
            "new_native_calls":0,"new_fits":0,"new_wet_measurements":0,"go_to_original_inverse":gate,
            "scope":"All immutable artifact hashes and scores; raw bank and held physics, weights and pair seeds independently decoded for fixed anchor subset."}


def fixtures():
    tail = np.zeros((3,2,3)); complex_ = tail+1.
    x = np.stack([tail,complex_])
    p = np.array([.25,.75])
    r = np.array([[.10,.10],[.10,.50],[.50,.10],[.50,.50]])
    close(independent_energy(x,p,tail,r)[0],[.5,0.])
    close(ess(p[:,None]),[1.6,1.6])
    weighted_orbit = np.array([[.125,.125],[.375,.375]])
    close(ess(weighted_orbit),[1.6,3.2])
    return {"passed":True,"native_calls":0,"new_fits":0,"same_physical_energy_geometry":True,
            "normalized_parent_orbit_probabilities":True,"pair_integral_fixture":True}


if __name__=="__main__":
    cli = ArgumentParser(description=__doc__)
    args = cli.add_mutually_exclusive_group(required=True)
    args.add_argument("--fixtures",action="store_true")
    args.add_argument("--out",type=Path)
    a = cli.parse_args()
    with threadpool_limits(limits=1):
        if a.fixtures:
            print(fixtures())
        else:
            out = a.out.resolve(strict=True)
            write(out/"INDEPENDENT_AUDIT.json",audit(out))
            print("PASS independent full-phase atlas readback",out)
