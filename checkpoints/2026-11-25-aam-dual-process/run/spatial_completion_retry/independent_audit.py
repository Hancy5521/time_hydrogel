"""Read back frozen spatial predictions and verify the published numerical invariants."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from hashlib import file_digest, sha256
from itertools import product
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
STUDY = ROOT.parent / "spatial_completion_retry.py"
METHODS = ("volume", "code_only", "direct", "multiscale")
METRICS = {"hidden_at_cutoff": ([0], True), "future_hidden": ([1, 2, 3], True),
           "future_all": ([1, 2, 3], False)}


def fingerprint(path):
    with path.open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def doc(name):
    return json.loads((ROOT/name).read_text(encoding="utf-8-sig"))


def addresses(role, count=None):
    run = ROOT.parent.name.rsplit("_",1)[-1]
    h = sha256(f"{run}|integrated-AAm|{role}".encode()).digest()
    prng = np.random.default_rng(int.from_bytes(h[:16],"little"))
    if count is None:
        return prng
    return prng.integers(0,2**63-1,size=count)


def run():
    before = doc("REGISTERED.json")
    frozen = doc("FROZEN.json")
    result = doc("SCORED.json")
    assert before["script"] == fingerprint(STUDY)
    for file,digest in before["source"].items():
        assert fingerprint(Path(file)) == digest
    pred_file = ROOT/"PREDICTED.npz"
    confirmed_file = ROOT/"CONFIRMED.npz"
    assert frozen["sha256"] == fingerprint(pred_file)
    assert result["forecast_sha256"] == fingerprint(ROOT/"FROZEN.json")
    assert result["truth_sha256"] == fingerprint(confirmed_file)
    utc = lambda d:datetime.fromisoformat(d["utc"])
    assert utc(before)<utc(frozen)<utc(result)

    with np.load(pred_file) as z:
        pred = {m:z[m].copy() for m in METHODS}
        prefix = z["query_prefix"]
        train_ids, held_ids = z["train_seeds"], z["test_seeds"]
        query_g = z["query_code"]
    with np.load(confirmed_file) as z:
        actual = z["observed"]
    n,ts,channels,cells = actual.shape
    assert (n,ts,channels,cells)==(192,4,3,64) and result["cases"]==n
    assert query_g.shape==(n,4) and (abs(query_g)<=.48).all()
    assert prefix.shape==(n,3,3,8)
    assert np.array_equal(train_ids,addresses("space development addresses",640))
    assert np.array_equal(held_ids,addresses("space held addresses",192))
    expected_code=addresses("space held code").uniform(-.48,.48,(192,4))
    assert np.array_equal(query_g,expected_code)
    assert len(np.unique(np.r_[train_ids,held_ids])) == 832
    assert all(z.shape==actual.shape and np.isfinite(z).all() for z in pred.values())

    mask = np.zeros(64,dtype=bool)
    for x,y,z in product(range(2),repeat=3):
        mask[16*x+4*y+z] = True
    observable = np.flatnonzero(mask)
    assert observable.tolist() == before["observed_cells"]
    obs_actual = actual[:,0,:,:][:,:,observable]
    pre_actual_diff = float(np.abs(obs_actual-prefix[:,2]).max())
    assert not pre_actual_diff
    assert ((actual>=0.) & (actual==np.rint(actual))).all()

    per_case, checks, data_diffs = {}, {}, []
    for m, y in {**pred,"raw_truth":actual}.items():
        observed = float(np.abs(y[:,0,:,:][:,:,observable]-prefix[:,2]).max())
        count_defect = float(np.abs(y[:,:,0,:].sum(axis=-1)-64).max())
        assert observed==0 and count_defect<2e-12 and (y>=0).all(), m
        assert np.max(y[:,:,1,:]-y[:,:,0,:]) < 1e-13
        ordered = y[:,:,1:,:].sum(axis=-1)
        checks[m] = {
            "max_visible_cutoff_mismatch":observed, "max_total_aam_count_error":count_defect,
            "max_total_reacted_mbaa_sites":float(ordered[:,:,1].max()),
            "aam_conversion_series_decreases":int((np.diff(ordered[:,:,0],axis=1)<-1e-10).sum()),
            "mbaa_conversion_series_decreases":int((np.diff(ordered[:,:,1],axis=1)<-1e-10).sum()),
        }
        if m not in pred:
            continue
        bag = np.abs(y-actual)
        channels_metrics = {}
        for k,(visits,use_hidden) in METRICS.items():
            allowed = np.flatnonzero(~mask) if use_hidden else np.arange(64)
            error = bag[:,visits,:,:][:,:,:,allowed]
            channel_case = error.sum(axis=(1,3))/(len(visits)*len(allowed))
            channels_metrics[k]=channel_case.mean(axis=1)
        per_case[m]=channels_metrics

    index=addresses("space paired bootstrap").integers(0,n,(5000,n))
    compare = {}
    for method, series in per_case.items():
        table = {}
        for metric,error in series.items():
            paired=error-per_case["code_only"][metric]
            report=result["result"][method][metric]
            confidence=np.quantile(paired[index].mean(axis=1),(.025,.975))
            numbers=np.r_[error.mean(),paired.mean(),confidence]
            recorded=np.r_[report["mae"],report["delta_vs_code_only"],report["CI95"]]
            data_diffs.append(float(np.max(np.abs(numbers-recorded))))
            vs_direct=error-per_case["direct"][metric]
            table[metric]={
                "mae":float(error.mean()),
                "delta_vs_code_only":float(paired.mean()),
                "ci95_vs_code_only":confidence.tolist(),
                "delta_vs_direct":float(vs_direct.mean()),
                "ci95_vs_direct":np.quantile(vs_direct[index].mean(axis=1),(.025,.975)).tolist(),
                "rows_better_than_code_only":int((paired<0).sum()),
                "rows_worse_than_code_only":int((paired>0).sum()),
            }
        compare[method]=table
    assert max(data_diffs)<2e-12
    t=datetime.now(timezone.utc)
    output={"utc":t.isoformat(),"asia_shanghai":t.astimezone(timezone(timedelta(hours=8))).isoformat(),
            "status":"PASS", "input_source_hashes_verified":len(before["source"]),
            "sha_prediction":fingerprint(pred_file),"sha_confirmed":fingerprint(confirmed_file),
            "sha_frozen":fingerprint(ROOT/"FROZEN.json"),"score_source_sha256":before["script"],
            "audit_source_sha256":fingerprint(Path(__file__)),
            "audited_cases":n,"additional_simulations":0,"per_cell_masks":{"observed":observable.tolist(),
            "observed_geometric_volume_share":.7**3,"unobserved_cells":56},
            "numeric_max_abs_discrepancy":max(data_diffs),"invariants":checks,
            "recomputed":compare, "per_case":{m:{k:v.tolist() for k,v in c.items()} for m,c in per_case.items()}}
    with (ROOT/"INDEPENDENT_AUDIT.json").open("x",encoding="utf-8") as stream:
        json.dump(output,stream,indent=2,allow_nan=False)
    print(json.dumps({k:v for k,v in output.items() if k!="per_case"},indent=2))


if __name__ == "__main__":
    run()
