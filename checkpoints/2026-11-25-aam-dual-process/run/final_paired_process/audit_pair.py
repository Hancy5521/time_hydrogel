"""Independent score readback for frozen dual-process confirmation."""
import json
import hashlib
from datetime import datetime, timezone
from pathlib import Path
import argparse
import numpy as np

HERE=Path(__file__).resolve().parent


def sha(file):
    with Path(file).open("rb") as f:
        return hashlib.file_digest(f,"sha256").hexdigest()


def get(file):
    return json.loads((HERE/file).read_text(encoding="utf-8-sig"))


def audit(expected):
    assert sha(HERE/"FROZEN.json")==expected
    conf,score,freeze=(get(f+".json") for f in ("CONFIRMED","SCORED","FROZEN"))
    assert datetime.fromisoformat(freeze["utc"]) < datetime.fromisoformat(conf["utc"])
    assert score["freeze"]==conf["freeze"]==expected
    assert score["raw_sha256"]==conf["raw_sha256"]==sha(HERE/"RAW.npz")
    for file,value in freeze["source_sha256"].items():
        assert sha(file)==value,file
    m,n=freeze["size"]
    with np.load(HERE/"RAW.npz",allow_pickle=False) as x:
        raw=x["path"]
        assert freeze["nominal_step"]==.0125
        assert x["seeds"].size==np.unique(x["seeds"]).size==m*n
        assert np.asarray(x["initiators"]).reshape(5,m).std(axis=0).max()==0
    assert raw.shape==(5,m,n,6,10) and np.all(np.diff(raw[...,0],axis=-1)>=0.)
    fields=[raw[...,2,2],raw[...,3,3],raw[...,5,0],raw[...,5,3]]
    def hit(lower,upper):
        return np.logical_and.reduce([
            (x>=a-1.5e-13)&(x<=b+1.5e-13) for x,a,b in zip(fields,lower,upper)])
    early,late=[hit(*bounds) for bounds in freeze["targets"]]
    early=early[[0,1,4]]
    late=late[[2,3,4]]
    states={"early":early,"delayed":late,"both_matched":early&late}
    assert not states["both_matched"][2].any()
    roles=("frozen","static","zero")
    card=hashlib.sha256((HERE.parent.name+"/final-paired-halved-step/card confidence measurement").encode()).digest()[:16]
    draws=np.random.default_rng(int.from_bytes(card,"little")).integers(0,m,(10000,m))
    intervals=0
    metric={key:value.mean(axis=-1) for key,value in states.items()}
    metric["product_under_independent_addresses"]=metric["early"]*metric["delayed"]
    for label,matrix in metric.items():
        out=score["results"][label]
        summaries={tag:v for tag,v in zip(roles,matrix,strict=True)}
        summaries["frozen_minus_static"]=matrix[0]-matrix[1]
        summaries["frozen_minus_zero"]=matrix[0]-matrix[2]
        for name,z in summaries.items():
            np.testing.assert_allclose(out[name]["estimate"],z.mean(),rtol=0.,atol=1.e-14)
            np.testing.assert_allclose(out[name]["ci95"],np.quantile(z[draws].mean(axis=1),[.025,.975]),rtol=0.,atol=1.e-14)
            intervals+=1
        if label in states:
            assert [out[k]["hits"] for k in roles]==[int(r.sum()) for r in states[label]]
    actual=(raw[0,...,3,3]-raw[2,...,3,3]).mean(1)
    np.testing.assert_allclose(score["middle_largest_early_minus_delayed"]["estimate"],actual.mean(),rtol=0.,atol=1.e-14)
    np.testing.assert_allclose(score["middle_largest_early_minus_delayed"]["ci95"],
                               np.quantile(actual[draws].mean(1),(.025,.975)),rtol=0.,atol=1.e-14)
    return {"utc":datetime.now(timezone.utc).isoformat(),"freeze":expected,"source_sha256":sha(__file__),
            "score_sha256":sha(HERE/"SCORED.json"),"recomputed_card_groups":m,"checked_intervals":intervals+1,
            "common_radical_budget":True,"quantitative_discrepancy_at_most":1.e-14}


if __name__=="__main__":
    cli=argparse.ArgumentParser()
    cli.add_argument("--freeze-sha256",required=True)
    opts=cli.parse_args()
    result=audit(opts.freeze_sha256)
    with (HERE/"INDEPENDENT_AUDIT.json").open("x",encoding="utf-8") as out:
        json.dump(result,out,indent=2,allow_nan=False)
    print(result)
