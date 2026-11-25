"""Independent transfer score and seed ledger checks. No simulator imports."""
import argparse
import json
import hashlib
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

FOLDER = Path(__file__).resolve().parent


def sha(file):
    return hashlib.sha256(Path(file).read_bytes()).hexdigest()


def get(file):
    return json.loads((FOLDER/file).read_text(encoding="utf-8-sig"))


def assert_close(x,y):
    np.testing.assert_allclose(x,y,rtol=0.,atol=1.e-13)


def energy(learned, outcome):
    def view(track):
        mass=track[:,3:6,5]/.18
        chain=track[:,3:6,2]/.10
        return np.column_stack((mass,chain))
    f,y = view(learned),view(outcome)
    dist = lambda a,b: np.sqrt(((a[:,None]-b[None])**2).sum(axis=-1)/6)
    return dist(f,y).mean()-.5*dist(f,f).mean()


def examine(expected):
    assert sha(FOLDER/"FROZEN.json")==expected
    f,r,a,c,score=(get(key+".json") for key in ("FROZEN","REGISTERED","ACQUISITION","CONFIRMED","SCORED"))
    assert f["registered_sha256"]==sha(FOLDER/"REGISTERED.json")
    assert f["acquisition_sha256"]==sha(FOLDER/"ACQUISITION.json")
    assert f["predicted_sha256"]==sha(FOLDER/"PREDICTED.npz")
    assert all(sha(source)==digest for source,digest in r["dependencies"].items())
    assert all(sha(FOLDER/file)==digest for file,digest in a["files"].items())
    assert sha(FOLDER/"CONFIRMED.npz")==c["truth_sha256"]==score["truth_sha256"]
    assert c["frozen_sha256"]==score["frozen_sha256"]==expected
    assert datetime.fromisoformat(f["utc"])<datetime.fromisoformat(c["utc"])
    model = f["models"]
    parent=FOLDER.parent
    previous=[]
    for name,keys in [
        ("DEVELOPMENT.npz",["seeds"]),("QUERY_LOCAL.npz",["seeds"]),("ZERO_C_ANCHOR.npz",["seeds"]),
        ("PREDICTED.npz",["forward_seeds","planning_seeds"]),("CONFIRMED.npz",["seeds"]),
        ("topology_witness/DEV.npz",["seeds"]),("topology_witness/HELD.npz",["seeds"])]:
        with np.load(parent/name,allow_pickle=False) as z:
            previous.extend(z[key].ravel() for key in keys)
    read = lambda name:np.load(FOLDER/(name+".npz"),allow_pickle=False)
    d,q,p,c = (read(key) for key in ("DEV","QUERY","PREDICTED","CONFIRMED"))
    independent = np.concatenate([*(z["seeds"].ravel() for z in (d,q,p,c)),*previous])
    assert len(np.unique(independent))==len(independent)
    assert q.files==["X","known","seeds"]
    assert_close(q["known"],c["code"][:,:3])
    pred,real,guessed=p["path"],c["path"],p["bridge"]
    assert pred.shape==(5,56,24,6,10) and real.shape==(56,24,6,10)
    assert np.all(np.diff(real[...,0],axis=-1)>=0)
    assert np.all(guessed[-1]==0)
    assert np.all(np.abs(guessed)<=.70)
    mse=(guessed-c["code"][:,3])**2
    joint=np.asarray([[energy(x,y) for x,y in zip(a,real)] for a in pred])
    metrics={"code_mse":mse,"c_joint":joint}
    for key,column in (("c_final_bridge",5),("c_final_largest",3)):
        metrics[key]=np.abs(pred[...,5,column].mean(axis=-1)-real[...,5,column].mean(axis=-1))
    seed=int.from_bytes(hashlib.sha256((r["seed_namespace"]+"|bootstrap by card").encode()).digest()[:16],"little")
    groups=np.random.default_rng(seed).integers(0,56,(8192,56))
    for key,err in metrics.items():
        assert_close(err,score["per_card"][key])
        assert_close(err.mean(axis=1),[score["metrics"][key][m] for m in model])
        for i,m in enumerate(("ridge","rbf")):
            for ref,j in (("early",i),("zero",4)):
                gain=err[j]-err[i+2]
                receipt=score["paired_gain"][key][m]["later_vs_"+ref]
                assert_close(gain.mean(),receipt["gain"])
                assert_close(np.quantile(gain[groups].mean(axis=1),[.025,.975]),receipt["percentile95"])
    return {"utc":datetime.now(timezone.utc).isoformat(),"frozen_sha256":expected,
            "scored_sha256":sha(FOLDER/"SCORED.json"),"source_sha256":sha(__file__),
            "distinct_address_count":len(independent),"scores_recomputed_exact_to":1.e-13,"test_cards":56,
            "paired_intervals_recomputed":16}


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--freeze-sha256",required=True)
    args=parser.parse_args()
    result=examine(args.freeze_sha256)
    with (FOLDER/"INDEPENDENT_AUDIT.json").open("x",encoding="utf-8") as out:
        json.dump(result,out,indent=2,allow_nan=False)
    print(result)
