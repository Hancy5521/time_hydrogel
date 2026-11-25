"""Audit event-marker behavior and selected margin counts from a held saved archive."""

from datetime import datetime,timedelta,timezone
from hashlib import file_digest
from pathlib import Path
import json
import sys

import numpy as np


ROOT = Path(__file__).resolve().parent


def sha(file):
    with file.open("rb") as stream:
        return file_digest(stream,"sha256").hexdigest()


def read(file):
    return json.loads(file.read_text(encoding="utf-8-sig"))


def evaluate(sha_expected):
    if sha_expected!=sha(ROOT/"FROZEN.json"):
        raise ArithmeticError("Held outcomes require the exact prospective freeze.")
    freeze,confirmation,score,registration=[read(ROOT/name) for name in (
        "FROZEN.json","CONFIRMED.json","SCORED.json","REGISTERED.json")]
    assert all(v["frozen_sha256"]==sha_expected for v in (confirmation,score))
    assert confirmation["npz_sha256"]==sha(ROOT/"CONFIRMED.npz")
    assert freeze["registration_sha256"]==sha(ROOT/"REGISTERED.json")
    assert sha(ROOT/"PLAN.npz")==freeze["planning_sha256"]
    gate = freeze["selected_gate"]
    with np.load(ROOT/"CONFIRMED.npz",allow_pickle=False) as states:
        path, raw, offsets = (states[key] for key in ("path","raw","offsets"))
    with np.load(ROOT/"PLAN.npz",allow_pickle=False) as stage:
        plan=stage["path"]
    low,high = registration["target"]
    low,high=np.array(low),np.array(high)

    def constraints(data):
        vec = np.stack((data[...,2,2],data[...,3,3],data[...,5,0],data[...,5,3]),axis=-1)
        return (vec>=low-1.5e-13) & (vec<=high+1.5e-13)

    dev=constraints(plan).all(axis=-1).mean(axis=(1,2))
    np.testing.assert_array_equal(dev,freeze["candidate_hits"])
    assert int(freeze["selected"])==int(np.argmax(dev[2:])+2)
    hyp=constraints(path)
    hits=hyp.all(axis=-1).sum(axis=(1,2))
    expected=[score["scores"][k]["n_pass"] for k in ("zero","init_minus",gate["label"])]
    np.testing.assert_array_equal(hits,expected)
    by_arm={}
    m=path.shape[1]
    for candidate,arm in enumerate(("zero","init_minus",gate["label"])):
        early=0; late=0
        for i in range(candidate*m,(candidate+1)*m):
            journal=raw[offsets[i]:offsets[i+1]]
            hit_times=journal[np.isin(journal[:,1],[5,6,9]),0]
            early+=int((hit_times<gate["switch"]).sum())
            late+=int((hit_times>=gate["switch"]).sum())
        by_arm[arm]={"joint_hits":int(hits[candidate]),"first_replicate_early_pendant_events":early,
                     "first_replicate_late_pendant_events":late,
                     "constraint_passes":hyp[candidate].sum(axis=(0,1)).tolist()}
    if gate["early"]==0. and by_arm[gate["label"]]["first_replicate_early_pendant_events"]:
        raise ArithmeticError("Native event logs violate zero pre-switch reaction gate.")
    now=datetime.now(timezone.utc)
    return {"utc":now.isoformat(),"asia_shanghai":now.astimezone(timezone(timedelta(hours=8))).isoformat(),
            "frozen_sha256":sha_expected,"scored_sha256":sha(ROOT/"SCORED.json"),
            "script_sha256":sha(Path(__file__)),"result":"pass","by_arm":by_arm}


if __name__=="__main__":
    result=evaluate(sys.argv[1])
    with (ROOT/"INDEPENDENT_SCORES.json").open("x",encoding="utf8") as out:
        json.dump(result,out,indent=2,allow_nan=False)
    print(result["by_arm"],flush=True)
