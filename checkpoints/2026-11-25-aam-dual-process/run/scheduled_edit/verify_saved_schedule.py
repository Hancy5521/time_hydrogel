"""Independent scores and seed role verification for a frozen intervention."""

from datetime import datetime, timedelta, timezone
from hashlib import file_digest
from pathlib import Path
import json
import sys

import numpy as np


ROOT = Path(__file__).resolve().parent


def digest(path):
    with path.open("rb") as stream:
        return file_digest(stream,"sha256").hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def targeted(series):
    return series[...,2,2],series[...,3,3],series[...,5,0],series[...,5,3]


def goal_hit(series, bounds):
    columns = targeted(series)
    return np.logical_and.reduce([
        (feature-lo >= -1.5e-13) & (feature-hi <= 1.5e-13)
        for feature,lo,hi in zip(columns,*bounds,strict=True)
    ])


def outcome(root, sha):
    freeze = root/"FROZEN.json"
    if digest(freeze) != sha:
        raise ValueError("Requires valid pre-reveal lock.")
    registration, frozen, confirmed, scored = [read(root/name) for name in (
        "REGISTERED.json","FROZEN.json","CONFIRMED.json","SCORED.json")]
    if frozen["register_sha256"] != digest(root/"REGISTERED.json") or scored["frozen_sha256"] != sha:
        raise ArithmeticError("Inconsistent source links.")
    if datetime.fromisoformat(frozen["utc"])>=datetime.fromisoformat(confirmed["utc"]):
        raise ArithmeticError("Prediction did not precede result.")
    roles = {
        "scheduled_edit/PLANNED":(root/"PLANNED.npz","seeds"),
        "scheduled_edit/CONFIRMED":(root/"CONFIRMED.npz","seeds"),
        "parent/DEVELOPMENT":(root.parent/"DEVELOPMENT.npz","seeds"),
        "parent/QUERY_LOCAL":(root.parent/"QUERY_LOCAL.npz","seeds"),
        "parent/ZERO_C_ANCHOR":(root.parent/"ZERO_C_ANCHOR.npz","seeds"),
        "parent/PREDICTED_forward":(root.parent/"PREDICTED.npz","forward_seeds"),
        "parent/PREDICTED_planning":(root.parent/"PREDICTED.npz","planning_seeds"),
        "parent/CONFIRMED":(root.parent/"CONFIRMED.npz","seeds"),
    }
    addresses = set(range(8914112,8914152))
    sizes = {}
    for role, (file,key) in roles.items():
        with np.load(file,allow_pickle=False) as z:
            seed_set = set(z[key].ravel().tolist())
            if not addresses.isdisjoint(seed_set) or len(seed_set)!=z[key].size:
                raise ArithmeticError(f"Random role overlap: {role}")
        sizes[role] = len(seed_set)
        addresses.update(seed_set)
    parent = read(root.parent/"FROZEN.json")
    if registration["targets"] != parent["targets"] or registration["parent_freeze_sha256"] != digest(root.parent/"FROZEN.json"):
        raise ArithmeticError("Parent objectives changed.")
    if (frozen["predicted_sha256"] != digest(root/"PLANNED.npz")
            or confirmed["confirmed_sha256"] != digest(root/"CONFIRMED.npz")
            or scored["true_outcomes_sha256"] != confirmed["confirmed_sha256"]):
        raise ArithmeticError("Scored values refer to modified data.")
    results = {}
    with np.load(root/"CONFIRMED.npz", allow_pickle=False) as z, np.load(root/"PLANNED.npz",allow_pickle=False) as p:
        idx = z["selected_ids"]
        for goal,target in zip(registration["objectives"], registration["targets"],strict=True):
            rate = goal_hit(p["paths"],target).mean(axis=(1,2))
            verify = frozen["decisions"][goal]
            assert np.array_equal(rate,np.asarray(verify["planning_hits"]))
            for kind,lo,hi in (("static",0,13),("scheduled",13,25)):
                assert rate[verify[kind]] == rate[lo:hi].max()
            v = goal_hit(z["path"],target)
            sums = v.sum(axis=(1,2))
            chosen = {}
            for j,c in enumerate(idx):
                name=registration["catalog"][c]["label"]
                assert sums[j]==scored["results"][goal]["hits"][name]["n_success"]
                chosen[name] = {
                    "joint_hit":int(sums[j]),
                    "constraint_hits": [float(((x>=lo-1e-13)&(x<=hi+1e-13)).mean())
                                        for x,lo,hi in zip(targeted(z["path"][j]),*target,strict=True)],
                    "target_feature_means": [float(x.mean()) for x in targeted(z["path"][j])],
                }
            results[goal]=chosen
    t=datetime.now(timezone.utc)
    return {
        "utc":t.isoformat(),"asia_shanghai":t.astimezone(timezone(timedelta(hours=8))).isoformat(),
        "script_sha256":digest(Path(__file__)),
        "all_scores_exact":True,"freeze_sha256":sha,
        "score_sha256":digest(root/"SCORED.json"),"seed_roles_addresses":sizes,"recomputed":results,
    }


if __name__=="__main__":
    result=outcome(ROOT,sys.argv[1])
    with (ROOT/"INDEPENDENT_SCORES.json").open("x",encoding="utf8") as f:
        json.dump(result,f,indent=2,allow_nan=False)
    print("Independent decision counts and disjoint address audit passed.",flush=True)
