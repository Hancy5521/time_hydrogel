"""Read-only audit of the conditional three-camera Brownian-tape benchmark."""

from argparse import ArgumentParser
from datetime import datetime
from hashlib import sha256
from pathlib import Path

import numpy as np
from scipy.stats import t
from threadpoolctl import threadpool_limits

from aam_formation_actual_rollout import packet_fingerprint
from aam_formation_cycle_data import check_actual_state
from aam_formation_cycle_driver import np_load, read, sha, write
from aam_formation_local_initial import draw_initial
from aam_formation_path_cycle import load_tree, stamp
from audit_aam_assay_guided_sir import _camera
from audit_aam_camera_information_sandwich import independent_apertures


NAMESPACE = "aam-gaussian-tape-endpoint-movement-v1"
TAPE_SCHEMA = "aam-original-no-feed-endpoint-preserving-pcn-v1"
UNITS = tuple(range(2026134000, 2026134048))
METHODS = ("isotropic", "refresh", "endpoint")
STEPS = 24
KEY = 427160737
RHO = .90
CELLS = 1 << 53
ATTEMPTS = 87
SOURCE = b"aam-shared-early-bank-grouped-native-suffix-smoother-v1"
ROOT = [int.from_bytes(sha256(SOURCE).digest()[i:i+4], "little") for i in (0, 4, 8, 12)]


def require(test, msg):
    if not test:
        raise ValueError(msg)


def close(actual, expected, atol=2.e-12):
    np.testing.assert_allclose(actual, expected, rtol=2.e-12, atol=atol)


def artifact(spec, path):
    require(Path(spec["path"]).resolve() == path.resolve()
            and path.stat().st_size == spec["bytes"] and sha(path) == spec["sha256"],
            f"Artifact bytes or source path changed: {path}")
    return path


def rng(*address):
    return np.random.default_rng(np.random.SeedSequence(address))


def intervals(cfg, stage):
    left, right = cfg["prefix_clock"][stage:stage+2]
    boundaries = [left]
    while boundaries[-1] < right:
        here = boundaries[-1]
        future = min(here + cfg["motion"]["step"], right)
        require(future > here, "The original half-step has nonpositive duration.")
        boundaries.append(future)
    return np.asarray(boundaries)


def roles(cfg, unit, stage):
    slabs = len(intervals(cfg, stage))-1
    n = cfg["recipe"]["aam"]+cfg["recipe"]["mbaa"]
    return {
        "brownian": rng(*ROOT, 2, unit, stage, 0).standard_normal((slabs, 2, n, 3)),
        "waiting": rng(*ROOT, 2, unit, stage, 1).exponential(size=(slabs, ATTEMPTS)),
        "mark": rng(*ROOT, 2, unit, stage, 2).integers(CELLS, size=(slabs, ATTEMPTS), dtype=np.int64),
    }


def half_direction(cfg, stage):
    v = np.sqrt(np.repeat(np.diff(intervals(cfg, stage))/2., 2))
    return v/np.linalg.norm(v)


def candidate(cfg, unit, step, stage, table, proof, mode):
    d = half_direction(cfg, stage)
    previous = table["brownian"]
    x = previous.reshape(len(d), -1)
    noise = rng(KEY, unit, step, 4)
    if mode == "refresh":
        require(proof == {"fresh_brownian_only": True}, "Wrong tagged independently refreshed stream.")
        return {**table, "brownian": noise.standard_normal(previous.shape)}
    direction = np.outer(d, d)
    ordinary_rho = 1.-(1.-RHO)*(len(d)-1)/len(d)
    z = noise.standard_normal(x.shape)
    rho = RHO if mode == "endpoint" else ordinary_rho
    transform = np.eye(len(d))-direction if mode == "endpoint" else np.eye(len(d))
    amplitude = np.sqrt(1.-rho*rho)
    # Independently form the full conditional Gaussian affine map rather than
    # calling the implementation's temporal projection arithmetic.
    updated = x + transform @ ((rho-1.)*x+amplitude*z)
    delta = updated-x
    net = float(np.max(abs(d@delta)))
    protected = x.shape[1] if mode == "endpoint" else 0
    require(proof["schema"] == TAPE_SCHEMA and proof["mode"] == mode and proof["rho"] == RHO
            and proof["temporal_coordinates"] == len(d) and proof["native_slabs"] == len(d)//2
            and proof["tape_dimensions"] == x.size and proof["protected_dimension"] == protected
            and proof["wait_and_mark_unchanged"] is True,
            "Proposal left the frozen original unit-tape geometry.")
    close(proof["effective_rho"], rho)
    close(proof["prior_expected_squared_jump"], 2.*(1.-rho)*np.trace(transform)*x.shape[1])
    close(proof["jump_squared"], np.sum(delta**2))
    close(proof["max_abs_net_standardized_change"], net)
    require(mode != "endpoint" or net < 2.e-13, "Free endpoint ceased to be preserved.")
    return {**table, "brownian": updated.reshape(previous.shape)}


def camera_matrix(packets, cfg):
    require(len(packets) == 4, "Three real readouts and the predecessor are required.")
    for packet, age in zip(packets, cfg["prefix_clock"], strict=True):
        state, recipe, geom, rates, factors = packet
        expected = [cfg["recipe"][k] for k in ("aam","mbaa","initial_mbaa","feed_time")]
        np.testing.assert_array_equal(recipe, expected)
        expected_geometry = [cfg["motion"][k] for k in
                             ("step","diffusion","box","spring","rest","repulsion","core","capture","inner")]
        np.testing.assert_array_equal(geom, [*expected_geometry,cfg["corridor"]])
        np.testing.assert_array_equal(rates, cfg["rates"])
        np.testing.assert_array_equal(factors, cfg["factors"])
        check_actual_state(state, expected_age=age)
    for a, b in zip(packets[:-1], packets[1:], strict=True):
        old, later = a[0], b[0]
        events, edges = old.counts[:2]
        np.testing.assert_array_equal(later.events[:events], old.events[:events])
        np.testing.assert_array_equal(later.edges[:edges], old.edges[:edges])
        np.testing.assert_array_equal(later.lifts[:edges], old.lifts[:edges])
    return np.stack([_camera(p) for p in packets[1:]])


def graph_features(packets):
    return np.stack([independent_apertures(packet) for packet in packets[1:]])


def hashes(packets):
    return [packet_fingerprint(p) for p in packets]


def gaussian_log(clean, query, cfg):
    clean, noise = np.asarray(clean), np.asarray(cfg["noise"])
    require(clean.shape == (3, 4) and np.isfinite(clean).all(), "Wrong camera values or support.")
    return float(-.5*np.square((clean-query)/noise).sum())


def generating_state(cfg, unit, query, reference):
    initialization = draw_initial(cfg, rng(*ROOT, 1, unit))
    require(set(initialization) == set(reference["initial"]), "Initial prior keys changed.")
    for name, value in initialization.items():
        np.testing.assert_array_equal(reference["initial"][name], value)
    packets = reference["packets"]
    np.testing.assert_array_equal(packets[0][0].points, initialization["points"])
    require(int(packets[0][0].counts[3]) == initialization["I"], "Actual initial initiator stock differs.")
    clean = camera_matrix(packets, cfg)
    close(clean, reference["clean"])
    noise = rng(*ROOT, 3, unit).standard_normal((3, 4))*cfg["noise"]
    np.testing.assert_array_equal(reference["clean"]+noise, query)
    require(reference["calls"] == {"attempted":3,"completed":3,"failed":0},
            "Exactly one initial acquisition per camera is required.")
    tables = [roles(cfg, unit, stage) for stage in range(3)]
    for stage, (tr, table) in enumerate(zip(reference["intervals"],tables,strict=True)):
        require(tr["native_calls"] == 1 and not tr["tape"]["forced_mark_consumed"]
                and tr["log_ratio"] == 0., "Original event clock was tilted or forced.")
        start, end = packets[stage][0].counts[0], packets[stage+1][0].counts[0]
        event_times = iter(packets[stage+1][0].events[start:end,0])
        for row in tr["spans"]:
            address = row["slab"],row["attempt"]
            require(all(0 <= k < n for k,n in zip(address,table["mark"].shape,strict=True)),
                    "A native waiting or mark address left the original tape.")
            if row["exponential"] is not None:
                require(row["exponential"] == table["waiting"][address], "Wrong original waiting exponential.")
            if not row["censored"]:
                require(row["uniform"] == table["mark"][address]/CELLS and row["end"] == next(event_times),
                        "Original accepted reaction lost its role-addressed mark or clock.")
        require(next(event_times,None) is None, "An actual initial covalent event is absent from the native ledger.")
    return clean, tables


def canonical_edges(packet):
    s = packet[0]
    return {tuple(sorted(map(int, edge))) for edge in s.edges[:s.counts[1]]}


def audit_chain(cfg, unit, query, reference, tables, record, mode):
    clean = np.array(reference["clean"], copy=True)
    prefix = hashes(reference["packets"])
    native = accepted = 0
    caches = {v["step"]:v for v in record["stored"]}
    require(set(caches)=={0, STEPS-1} and len(record["stored"])==2
            and len(record["attempts"])==STEPS, "Missing addressed physical-path anchors or tape attempts.")
    for step, entry in enumerate(record["attempts"]):
        stage = int(rng(KEY,unit,step,3).integers(3))
        u = float(rng(KEY,unit,step,5).random())
        require(step == entry["step"] and stage == entry["stage"] and u == entry["u"]
                and prefix == entry["old"] and len(entry["proposed"])==4
                and entry["old"][:stage+1]==entry["proposed"][:stage+1],
                "Old lineage or unchanged native predecessor no longer matches its registered move.")
        np.testing.assert_array_equal(clean,entry["from_camera"])
        proposal_camera = entry["candidate_camera"]
        np.testing.assert_array_equal(clean[:stage],proposal_camera[:stage])
        updated = candidate(cfg, unit, step, stage, tables[stage], entry["proof"], mode)
        old_log = gaussian_log(clean, query, cfg)
        new_log = gaussian_log(proposal_camera, query, cfg)
        close((entry["log_before"],entry["log_candidate"]),(old_log,new_log))
        ok = bool(u == 0. or np.log(u)<min(0.,new_log-old_log))
        require(entry["accepted"] is ok, "The complete camera-product MH decision does not reproduce.")
        if step in caches:
            kept = caches[step]
            require(hashes(kept["before"]) == prefix and hashes(kept["proposed"]) == entry["proposed"],
                    "A raw endpoint snapshot cannot reproduce its move's frozen fingerprint.")
            close(camera_matrix(kept["before"],cfg), clean)
            close(camera_matrix(kept["proposed"],cfg), proposal_camera)
            for key in ("waiting","mark"):
                np.testing.assert_array_equal(kept["before_stage_table"][key],tables[stage][key])
                np.testing.assert_array_equal(kept["after_stage_table"][key],tables[stage][key])
            close(kept["before_stage_table"]["brownian"], tables[stage]["brownian"])
            close(kept["after_stage_table"]["brownian"], updated["brownian"])
        native += 3-stage
        if ok:
            prefix, clean, tables[stage] = entry["proposed"], proposal_camera, updated
            accepted += 1
    final = record["final"]
    require(hashes(final)==prefix, "Terminal trajectory is not the final accepted native fingerprint.")
    close(camera_matrix(final,cfg),clean)
    decoded = graph_features(final)
    close(record["final_hidden"], decoded)
    delta = decoded-graph_features(reference["packets"])
    close(record["hidden_current_diff"],delta)
    edges = len(canonical_edges(reference["packets"][-1]) ^ canonical_edges(final[-1]))
    require(edges==record["end_edges_symmetric_diff"], "Terminal legal edges do not reproduce movement.")
    rms = .5*np.sqrt(np.mean(np.square(delta)))
    final_rms = .5*np.sqrt(np.mean(np.square(delta[-1])))
    change_rms = .5*np.sqrt(np.mean(np.square(delta[-1]-delta[-2])))
    return (rms,final_rms,change_rms,edges),accepted/STEPS,native


def intervals95(values):
    mean = values.mean(axis=0)
    n = len(values)
    half = t.isf(.025,n-1)*np.sqrt(np.sum((values-mean)**2,axis=0)/(n*(n-1)))
    return np.stack((mean,mean-half,mean+half),axis=-1)


def audit(out):
    names = ("REGISTERED.json","RUN.json","EXECUTION_STARTED.json","CLOSED.json","SCORED.json")
    files = [out/name for name in names]
    p, run, start, closed, score = docs = list(map(read,files))
    times = [datetime.fromisoformat(x["actual_utc"]) for x in docs]
    require(all(a<b for a,b in zip(times[:-1],times[1:],strict=True))
            and all(tt==datetime.fromisoformat(x["asia_shanghai"]) for x,tt in zip(docs,times,strict=True)),
            "Prospective/real-time execution chronology changed.")
    reg_hash = sha(files[0])
    require(p["namespace"]==NAMESPACE and tuple(p["units"])==UNITS and tuple(p["methods"])==METHODS
            and p["n_steps"]==STEPS and p["seed"]==KEY and p["proposal_rho"]==RHO
            and p["first_mother_role"]=="stationary diagnostic only; fixed realized initial and reaction tapes"
            and score["scope"]=="stationary conditional on true initial and reaction tapes, posterior mixing only"
            and p["max_native"]==len(UNITS)*(3+len(METHODS)*STEPS*3)
            and p["new_fit"]==p["old_held_opened"]==p["new_wet_measurements"]==0
            and score["new_blind_efficacy_scored"]==score["new_wet_measurements"]==0
            and run["before_new_initials_and_outcomes"] is True
            and run["registration_sha256"]==start["registration_sha256"]==reg_hash
            and score["closed_sha256"]==sha(files[3]),
            "Frozen role, limits, or target identity changed.")
    for path,digest in p["inputs"].items():
        require(sha(path)==digest, f"Original source/dataset receipt digest changed: {path}")
    for path,digest in p["sources"].items():
        require(sha(out/"SOURCE_SNAPSHOT"/Path(path).name)==digest, f"Frozen source differs: {path}")
    require(tuple(r["unit"] for r in closed["units"])==UNITS, "Missing or duplicated preregistered mothers.")
    movement, acceptance, native = [], [], []
    cfg = p["physics"]
    for item in closed["units"]:
        unit = item["unit"]
        folder = out/"study"/str(unit)
        target = load_tree(artifact(item["query"],folder/"TRUE_INITIAL_STATIONARY_QUERY.npz"))
        reference, query = target["reference"], target["measured"]
        _, tables = generating_state(cfg,unit,query,reference)
        measure_row, accept_row, costs_row = [], [], []
        for mode in METHODS:
            r = item["results"][mode]
            archive = load_tree(artifact(r["path"],folder/f"{mode}.npz"))
            measured, rate, calls = audit_chain(cfg,unit,query,reference,list(tables),archive,mode)
            require(r["cost"]=={"attempted":calls,"completed":calls,"failed":0}
                    and np.isfinite(r["seconds"]) and r["seconds"]>0., "Native route omitted undo or work.")
            measure_row.append(measured)
            accept_row.append(rate)
            costs_row.append(calls)
        movement.append(measure_row)
        acceptance.append(accept_row)
        native.append(costs_row)
    movement, acceptance, native = map(np.asarray,(movement,acceptance,native))
    arr = np_load(artifact(score["arrays"],out/"SCORED_ARRAYS.npz"))
    np.testing.assert_array_equal(arr["units"],UNITS)
    close(arr["movement"],movement)
    close(arr["acceptance"],acceptance)
    np.testing.assert_array_equal(arr["native"],native)
    close(score["means"],movement.mean(axis=0))
    close(score["moves_mean"],acceptance.mean(axis=0))
    np.testing.assert_array_equal(score["native_replay_by_method"],native.sum(axis=0))
    require(score["native_starts"]==3*len(UNITS) and score["native_starts"]+int(native.sum())<=p["max_native"],
            "Native actual total exceeds the bound.")
    gate = True
    for j,mode in enumerate(METHODS[:-1]):
        expected = intervals95(movement[:,-1]-movement[:,j])
        close(score["per_mother_endpoint_minus"][mode],expected)
        gate &= bool(expected[0,1]>0.)
    require(score["gate_for_blind_recovery"] is gate, "Primary paired mother gate changed.")
    return {**stamp(),"passed":True,"registration_sha256":reg_hash,"closed_sha256":sha(files[3]),
            "scored_sha256":sha(files[4]),"new_native_calls":0,"new_fits":0,
            "independent_mothers":len(UNITS),"complete_camera_product_acceptance_checks":len(UNITS)*len(METHODS)*STEPS,
            "tape_proposal_ledger_reconstructed":True,"raw_first_and_last_proposals_and_all_terminals_checked":True,
            "three_camera_hidden_movement_paired_gate_reproduced":True,"gate_for_blind_recovery":gate,
            "truth_initialized_conditional_diagnostic_not_a_blind_forecast":True,
            "scope":"Legal stored graphs and original addressed randomness. No independent re-integration of native history or blind efficacy."}


def fixtures():
    d = np.array([.3,.4,.5])
    d /= np.linalg.norm(d)
    q = np.eye(3)-np.outer(d,d)
    cov = np.eye(3)-(1.-RHO)*q
    close(cov@cov.T+(1.-RHO**2)*q,np.eye(3))
    close(cov@d,d)
    close(2.*np.trace(np.eye(3)-cov),2.*(1.-RHO)*(len(d)-1))
    close(intervals95(np.ones((4,2))),np.ones((2,3)))
    return "PASS exact Gaussian invariance and mother-paired interval fixtures"


if __name__ == "__main__":
    cli = ArgumentParser(description=__doc__)
    cli.add_argument("--out",type=Path)
    cli.add_argument("--fixtures",action="store_true")
    args = cli.parse_args()
    with threadpool_limits(limits=1):
        if args.fixtures:
            print(fixtures())
        else:
            out = args.out.resolve(strict=True)
            write(out/"INDEPENDENT_AUDIT.json",audit(out))
            print("PASS independent Gaussian tape audit",out)
