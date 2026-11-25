"""Prospective inverse design of a periodic-network transport proxy."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timedelta, timezone
from hashlib import file_digest, sha256
from pathlib import Path
import json
import sys

import numpy as np
from scipy.special import softmax

ROOT = Path(__file__).resolve().parent
CORE = ROOT.parent
sys.path.insert(0, str(CORE))

from network_transport import conductance_history

TAG = "20261125T052605964017Z/prospective-graph-unit-transport/v1"
TIMES = (3.5, 4.85, 6.)
N_PLAN, N_VERIFY = 12, 24
R_PLAN, R_VERIFY, R_OBS = 10, 16, 4
CODE_LIMIT, BANDWIDTH = .48, .275
POSTERIOR_DRAWS = 48
THRESHOLD = .02
CASES = ("early", "late")
POLICIES = ("shared_response", "rbf", "pooled_fixed", "zero_edit")
ACTIONS = (
    ("C_zero", "C", (0., 0., 0., 0.)),
    ("initiate_up", "C", (.44, 0., 0., 0.)),
    ("initiate_down", "C", (-.44, 0., 0., 0.)),
    ("grow_up", "C", (0., .44, 0., 0.)),
    ("grow_down", "C", (0., -.44, 0., 0.)),
    ("bridge_up", "C", (0., 0., 0., .55)),
    ("bridge_down", "C", (0., 0., 0., -.55)),
    ("start_and_feed_early", "C_proxy_early", (0., 0., 0., 0.)),
    ("start_and_feed_late", "C_proxy_late", (0., 0., 0., 0.)),
)
CONTEXTS = {
    "C_proxy_early": {
        "feed": .95, "diffusion": .19,
        "pulses": ((0., 1.45, 1.), (1.775, .55, .34/.19), (3.55, 1., .34/.19)),
    },
    "C_proxy_late": {
        "feed": 2.95, "diffusion": .19,
        "pulses": ((0., .55, 1.), (1.775, 1.45, .34/.19), (3.55, 1., .34/.19)),
    },
}


def stamp():
    now = datetime.now(timezone.utc)
    return {"utc": now.isoformat(), "asia_shanghai": now.astimezone(timezone(timedelta(hours=8))).isoformat()}


def digest(path):
    with path.open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def dump(name, data):
    with (ROOT/name).open("x", encoding="utf-8") as stream:
        json.dump(data, stream, indent=2, allow_nan=False)


def read(name):
    return json.loads((ROOT/name).read_text(encoding="utf-8-sig"))


def array_file(name, **values):
    with (ROOT/name).open("xb") as stream:
        np.savez_compressed(stream, **values)


def rng(role):
    bits = sha256(f"{TAG}/{role}".encode()).digest()
    return np.random.default_rng(np.random.SeedSequence(np.frombuffer(bits, dtype="<u4").tolist()))


def addresses(role, shape):
    return rng(role).integers(0, 2**63, size=shape, dtype=np.uint64)


def register():
    import reaction_adapter as native

    sources = native.sources()
    for f in (Path(__file__), ROOT/"PROTOCOL_ZH.md", CORE/"network_transport.py",
              CORE/"NETWORK_TRANSPORT_FIXTURES.json", CORE/"code_encoder.py", CORE/"DEVELOPMENT.npz",
              ROOT/"CAMERA_SHAPE_RECOVERY.json", ROOT/"REGISTERED_INITIAL.json", ROOT/"audit.py"):
        sources[str(f.resolve())] = digest(f)
    dump("REGISTERED.json", {
        **stamp(), "namespace": TAG, "sources": sources, "actions": ACTIONS, "contexts": CONTEXTS,
        "threshold": THRESHOLD, "target_times": TIMES, "parent_code_fit_roles": "development only",
        "card_address_roles": ["planning codes", "query code escrow", "planning initial seeds",
                               "query A/B snapshots", "verification initial seeds", "encoder"],
        "counts": [N_PLAN, R_PLAN, N_VERIFY, R_VERIFY, R_OBS, POSTERIOR_DRAWS],
    })


def verify_registration():
    registration = read("REGISTERED.json")
    for file, expected in registration["sources"].items():
        if digest(Path(file)) != expected:
            raise ValueError(f"Prepared source changed: {file}")


def run_job(job):
    import reaction_adapter

    g, seed, edit, env, save_journal = job
    if edit is None:
        observed = reaction_adapter.simulate(g, env, int(seed))["local"]
        return observed
    _, environment, delta = ACTIONS[edit]
    if environment in CONTEXTS:
        reaction_adapter.ENVIRONMENTS[environment] = CONTEXTS[environment]
    code = np.asarray(g) + np.asarray(delta)
    traced = reaction_adapter.simulate(code, environment, int(seed), full=True, journal=True)
    history = conductance_history(traced["journal"], TIMES)
    if (np.diff(history, axis=0) < -2.e-12).any():
        raise ArithmeticError("An accumulating bond graph lost structural conduction.")
    if not np.array_equal(history.any(axis=1), traced["path"][-3:, -1] != 0):
        raise ArithmeticError("Transport and the native winding flags disagree.")
    return (history, traced["journal"] if save_journal else None,
            traced["initiators"] if save_journal else None, traced["feed_time"] if save_journal else None)


def simulate(jobs, workers, title):
    results = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for index, row in enumerate(pool.map(run_job, jobs, chunksize=12), 1):
            results.append(row)
            if index % 288 == 0 or index == len(jobs):
                print(title, f"{index}/{len(jobs)}", flush=True)
    return results


def cases(trace):
    k = trace.sum(axis=-1)
    return np.stack(((k[..., 0] >= THRESHOLD) & (k[..., 2] >= THRESHOLD),
                     (k[..., 0] == 0.) & (k[..., 1] >= THRESHOLD) & (k[..., 2] >= THRESHOLD)), axis=-1)


def scheduled(g, seed, hold):
    return [(code, s, action, None, hold and k == 0)
            for code, addresses_by_code in zip(g,seed)
            for action in range(len(ACTIONS))
            for k,s in enumerate(addresses_by_code)]


def acquire(workers):
    verify_registration()
    planner = rng("planning codes").uniform(-CODE_LIMIT, CODE_LIMIT, size=(N_PLAN, 4))
    queries = rng("query code escrow").uniform(-CODE_LIMIT, CODE_LIMIT, size=(N_VERIFY, 4))
    assert len(np.unique(np.r_[planner,queries], axis=0)) == N_PLAN + N_VERIFY
    ids = addresses("planning initial seeds", (N_PLAN,R_PLAN))
    prefix_ids = addresses("query A/B snapshots", (N_VERIFY,2,R_OBS))
    planning = simulate(scheduled(planner,ids,False),workers,"planning from monomers")
    previews = simulate([(g, s, None, env, False)
                        for g, all_seeds in zip(queries,prefix_ids)
                        for env, replicate in zip(("A","B"),all_seeds) for s in replicate],
                       workers,"new A/B camera observations")
    trace = np.asarray([r[0] for r in planning]).reshape(N_PLAN,len(ACTIONS),R_PLAN,3,3)
    obs = np.asarray(previews).reshape(N_VERIFY,2,R_OBS,*previews[0].shape)
    array_file("PLANNING.npz", G=planner, seeds=ids, trace=trace)
    array_file("QUERY_CAMERA.npz", X=obs, seeds=prefix_ids)
    array_file("CODE_ESCROW.npz", G=queries)
    filenames = ("PLANNING.npz","QUERY_CAMERA.npz","CODE_ESCROW.npz")
    dump("ACQUIRED.json",{**stamp(),"files":{name:digest(ROOT/name) for name in filenames}})


def freeze():
    from code_encoder import encode

    verify_registration()
    files = read("ACQUIRED.json")["files"]
    for role in ("PLANNING.npz","QUERY_CAMERA.npz"):
        assert digest(ROOT/role) == files[role]
    with np.load(ROOT/"PLANNING.npz") as f:
        g, plan = f["G"], f["trace"]
    with np.load(ROOT/"QUERY_CAMERA.npz") as f:
        x = f["X"]
        x = x.reshape((*x.shape[:3], -1))
    with np.load(CORE/"DEVELOPMENT.npz") as f:
        posterior = encode(f["X"],f["G"],x,n_draws=POSTERIOR_DRAWS,seed=int(rng("encoder").integers(2**31)))
    success = cases(plan).mean(axis=2)
    marginal = success.mean(axis=0)
    probabilities, policies, weights = [], [], []
    for method in POLICIES:
        if method in posterior:
            dist = np.square(posterior[method][:,:,None,:]-g[None,None,:,:]).sum(axis=-1)
            weight = softmax(-.5*dist/BANDWIDTH**2,axis=2).mean(axis=1)
            p = np.einsum("qc,cag->qag",weight,success)
            choose = np.argmax(p,axis=1)
        elif method == "pooled_fixed":
            weight = np.full((len(x),len(g)),1./len(g))
            p = np.broadcast_to(marginal, (len(x),*marginal.shape))
            choose = np.argmax(p,axis=1)
        else:
            weight = np.full((len(x),len(g)),1./len(g))
            p = np.broadcast_to(marginal, (len(x),*marginal.shape))
            choose = np.zeros((len(x),len(CASES)),dtype=int)
        probabilities.append(p)
        policies.append(choose)
        weights.append(weight)
    array_file("RANKS.npz", probability=probabilities, choice=policies, weights=weights)
    dump("FROZEN.json",{**stamp(), "register":digest(ROOT/"REGISTERED.json"),
          "acquisition":digest(ROOT/"ACQUIRED.json"), "ranking_sha256":digest(ROOT/"RANKS.npz"),
          "policies":POLICIES, "cases":CASES, "choices":np.asarray(policies).tolist(),
          "planning_hit":cases(plan).sum(axis=(0,2)).tolist(), "planning_total":N_PLAN*R_PLAN})
    print("FROZEN",digest(ROOT/"FROZEN.json"),flush=True)


def confirm(workers, expected):
    verify_registration()
    if expected != digest(ROOT/"FROZEN.json"):
        raise ValueError("Exact prediction freeze required before revealing verification codes.")
    locked = read("FROZEN.json")
    assert locked["register"] == digest(ROOT/"REGISTERED.json")
    assert locked["acquisition"] == digest(ROOT/"ACQUIRED.json")
    assert locked["ranking_sha256"] == digest(ROOT/"RANKS.npz")
    assert read("ACQUIRED.json")["files"]["CODE_ESCROW.npz"] == digest(ROOT/"CODE_ESCROW.npz")
    with np.load(ROOT/"CODE_ESCROW.npz") as f:
        codes = f["G"]
    with np.load(ROOT/"PLANNING.npz") as f:
        previous = f["seeds"].ravel()
    with np.load(ROOT/"QUERY_CAMERA.npz") as f:
        previous2 = f["seeds"].ravel()
    seeds = addresses("verification initial seeds",(N_VERIFY,R_VERIFY))
    assert len(np.unique(np.r_[previous,previous2,seeds.ravel()])) == previous.size+previous2.size+seeds.size
    result = simulate(scheduled(codes,seeds,True),workers,"sealed independent restarts")
    assert digest(ROOT/"FROZEN.json") == expected
    history = np.asarray([r[0] for r in result]).reshape(N_VERIFY,len(ACTIONS),R_VERIFY,3,3)
    journals = [r for r in result if r[1] is not None]
    array_file("CONFIRMED.npz",G=codes,trace=history,seeds=seeds,
               journal=np.concatenate([r[1] for r in journals]),
               offsets=np.r_[0,np.cumsum([len(r[1]) for r in journals])],
               initiators=np.array([r[2] for r in journals]),
               feed=np.array([r[3] for r in journals]))
    dump("CONFIRMED.json", {**stamp(),"freeze":expected,"sha256":digest(ROOT/"CONFIRMED.npz"),
                            "independent_seeds":int(seeds.size),"journal_count":len(journals),"wet_samples":0})


def difference(a,b,boot):
    delta = a-b
    return {"mean":float(delta.mean()), "interval_95":np.quantile(delta[boot].mean(1),(.025,.975)).tolist()}


def score():
    verify_registration()
    facts = read("CONFIRMED.json")
    locked = read("FROZEN.json")
    assert facts["freeze"] == digest(ROOT/"FROZEN.json")
    assert facts["sha256"] == digest(ROOT/"CONFIRMED.npz")
    with np.load(ROOT/"CONFIRMED.npz") as f:
        outcomes = cases(f["trace"])
    choices = np.asarray(locked["choices"])
    boot = rng("uncertainty bootstrap").integers(N_VERIFY,size=(10000,N_VERIFY))
    report = {}
    for goal,name in enumerate(CASES):
        all_hit = outcomes[:,:,:,goal]
        selected = all_hit[np.arange(N_VERIFY)[None,:],choices[:,:,goal]]
        group_prob = selected.mean(axis=2)
        best = all_hit.mean(axis=-1).max(axis=1)
        report[name] = {
            "verified_card_success_feasibility":int((best>0).sum()),
            "finite_menu_oracle_ceiling":float(best.mean()),
            "confirmed_menu_hits":all_hit.sum(axis=(0,2)).tolist(),
            "total_per_method":N_VERIFY*R_VERIFY,
            "method":{
                name:{
                    "hits":int(hit.sum()), "probability":float(p.mean()),
                    "delta_vs_pooled":difference(p,group_prob[2],boot),
                    "delta_vs_zero":difference(p,group_prob[3],boot),
                    "unique_edits":np.unique(choices[i,:,goal]).tolist(),
                } for i,(name,hit,p) in enumerate(zip(POLICIES,selected,group_prob))
            },
        }
    dump("SCORED.json",{**stamp(),"frozen_sha256":facts["freeze"],"truth_sha256":facts["sha256"],"result":report})
    print(json.dumps(report),flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage",choices=("register","acquire","freeze","confirm","score"))
    p.add_argument("--workers",type=int,default=5)
    p.add_argument("--freeze-sha256")
    a = p.parse_args()
    if a.stage == "register":
        register()
    elif a.stage == "acquire":
        acquire(a.workers)
    elif a.stage == "freeze":
        freeze()
    elif a.stage == "confirm":
        confirm(a.workers,a.freeze_sha256)
    else:
        score()
