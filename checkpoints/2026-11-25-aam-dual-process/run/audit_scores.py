"""Independent arithmetic audit of frozen AAm forecasts, choices and sealed replays.

This script never opens the held-code escrow or imports the forecasting code.
Caller must provide the digest recorded before the independent reruns.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import file_digest, sha256
import json
from pathlib import Path

import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import t


HERE = Path(__file__).resolve().parent
RESPONSE_COLUMNS = (0, 2, 3, 5, 9)
SCALE = np.array([.15, .10, .18, .18, .45])


def receipt(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def sha(path):
    with path.open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def paired_stats(delta):
    delta = np.asarray(delta, dtype=float)
    n = len(delta)
    estimate = delta.mean()
    half_width = t.ppf(.975, n-1) * delta.std(ddof=1) / np.sqrt(n)
    return {"mean": float(estimate), "paired_t95": [float(estimate-half_width), float(estimate+half_width)],
            "strictly_positive_cards": int((delta > 1.e-12).sum()), "cards": n}


def target_readout(path):
    return np.concatenate(
        [path[...,2:3,2], path[...,3:4,3], path[...,5:6,0], path[...,5:6,3]], axis=-1)


def calculate_anchor_bounds(anchors):
    t0, t1, t2, t3 = target_readout(anchors).T
    get = lambda x, p: np.quantile(x,p,method="linear")
    return np.array([[[get(t0,.25),get(t1,.65),get(t2,.4),0.], [1.,1.,1.,1.]],
                     [[get(t0,.25),0.,0.,get(t3,.4)], [1.,get(t1,.35),1.,1.]]])


def evaluate_candidates(path, lo, hi):
    values = target_readout(path)
    assert values.shape[-1] == len(lo) == len(hi)
    constraints = [(values[..., j]-left)/.15 for j,left in enumerate(lo) if left>0]
    constraints += [(right-values[..., j])/.15 for j,right in enumerate(hi) if right<1]
    margin = np.min(constraints, axis=0)
    hit = (margin >= -1.e-12).mean(axis=-1)
    return hit, margin.mean(axis=-1)


def pick(estimate, tie_margin):
    return max(range(len(estimate)), key=lambda j:(estimate[j],tie_margin[j],-j))


def future_energy(ensemble, truth):
    flatten = lambda a: (a[..., 2:6, :][..., list(RESPONSE_COLUMNS)]/SCALE).reshape(len(a),-1)
    f, y = flatten(ensemble), flatten(truth)
    crps = cdist(f,y).mean() - .5*cdist(f,f).mean()
    return float(crps / np.sqrt(f.shape[1]))


def audit(root, known_hash):
    if sha(root/"FROZEN.json") != known_hash:
        raise ValueError("Forward and inverse freeze does not match caller's prerelease hash")
    freeze = receipt(root/"FROZEN.json")
    for artifact,key in (("REGISTERED.json","registration_sha256"),("ACQUISITION.json","acquisition_sha256"),
                         ("PREDICTED.npz","predicted_sha256")):
        if sha(root/artifact) != freeze[key]:
            raise ValueError(f"Changed frozen artifact: {artifact}")
    registration = receipt(root/"REGISTERED.json")
    acquisition = receipt(root/"ACQUISITION.json")
    confirmation = receipt(root/"CONFIRMED.json")
    assert confirmation["verified_prediction_sha256"] == known_hash
    assert sha(root/"CONFIRMED.npz") == confirmation["sha256"]
    pool = receipt(root/"POOLED_NOMINAL.json")
    assert pool["freeze"] == known_hash
    assert pool["source"] == sha(root/"score_loop.py")
    assert datetime.fromisoformat(freeze["utc"]) <= datetime.fromisoformat(pool["utc"])
    assert datetime.fromisoformat(pool["utc"]) < datetime.fromisoformat(confirmation["utc"])
    scorer = receipt(root/"SCORED.json")
    assert scorer["freeze"] == known_hash and scorer["truth_sha256"] == confirmation["sha256"]
    assert scorer["score_source_sha256"] == sha(root/"score_loop.py")
    for path, digest in registration["source_sha256"].items():
        assert sha(Path(path)) == digest, path
    for name in ("DEVELOPMENT.npz","QUERY_LOCAL.npz","ZERO_C_ANCHOR.npz"):
        assert sha(root/name) == acquisition["inputs"][name]
    assert datetime.fromisoformat(freeze["utc"]) < datetime.fromisoformat(confirmation["utc"])

    predicted = np.load(root/"PREDICTED.npz", allow_pickle=False)
    confirmed = np.load(root/"CONFIRMED.npz", allow_pickle=False)
    dev = np.load(root/"DEVELOPMENT.npz", allow_pickle=False)
    early = np.load(root/"QUERY_LOCAL.npz", allow_pickle=False)
    anchor = np.load(root/"ZERO_C_ANCHOR.npz", allow_pickle=False)
    method = freeze["forecasters"]
    objective = freeze["objective_names"]
    assert len(set(method)) == len(method) and len(objective) == 2
    n = len(confirmed["code"])
    assert len(dev["G"]) == len(dev["X"]) == 128 and len(early["X"]) == n == 36
    assert set(early.files) == {"X","seeds"}
    assert set(predicted.files).issuperset({"forward","planning","choice","probability",
                                          "forward_seeds","planning_seeds","code"})
    assert predicted["forward"].shape == (len(method),n,24,6,10)
    assert predicted["planning"].shape == (len(method),n,13,8,6,10)
    assert predicted["choice"].shape == (len(method),2,n)
    assert confirmed["path"].shape == (n,13,16,6,10)
    assert np.allclose(np.asarray(freeze["targets"]),calculate_anchor_bounds(anchor["path"]),
                       rtol=0,atol=1.e-14)
    assert abs(confirmed["code"]).max()<=.48 and abs(predicted["code"]).max()<=.48

    seed_sets = {
        "dev_ab":dev["seeds"], "held_ab":early["seeds"], "c_zero_anchors":anchor["seeds"],
        "c_forecast":predicted["forward_seeds"], "c_planning":predicted["planning_seeds"],
        "c_validation":confirmed["seeds"],
    }
    for seeds in seed_sets.values():
        assert len(np.unique(seeds)) == seeds.size
    combined = np.hstack([s.ravel() for s in seed_sets.values()])
    assert len(np.unique(combined)) == combined.size, "A seed was reused between stages"

    forecast = predicted["forward"]
    actual = confirmed["path"]
    risk = np.asarray([[future_energy(a,b) for a,b in zip(runs,actual[:,0])]
                       for runs in forecast])
    nominal = method.index("nominal")
    key = f"{registration['run_id']}|integrated-AAm|score card bootstrap".encode()
    seed = int.from_bytes(sha256(key).digest()[:16],"little")
    cards_boot = np.random.default_rng(seed).integers(0,n,(5000,n))

    def recalibrate(delta):
        return np.quantile(np.asarray(delta)[cards_boot].mean(1),(.025,.975))

    def equal(a,b):
        np.testing.assert_allclose(a,b,rtol=1.e-12,atol=1.e-12)

    for i, name in enumerate(method):
        profile = scorer["new_environment_energy"][name]
        equal(profile["per_card"],risk[i])
        equal(profile["energy"],risk[i].mean())
        equal(profile["delta_vs_nominal"]["ci_95"],recalibrate(risk[i]-risk[nominal]))
        equal(profile["delta_vs_rbf"]["ci_95"],recalibrate(risk[i]-risk[method.index("rbf")]))
    forward = {name:{"energy":float(risk[i].mean()),
                      "better_than_nominal":paired_stats(risk[nominal]-risk[i])}
               for i,name in enumerate(method)}
    card_code = predicted["code"].mean(axis=-2)-confirmed["code"][None]
    encoded = method[:len(card_code)]
    code = {name:{"rmse":np.sqrt(np.square(card_code[i]).mean(axis=0)).tolist(),
                  "coverage80_actual_fraction":np.mean(
                      (np.quantile(predicted["code"][i],.1,axis=1)<=confirmed["code"])
                      &(np.quantile(predicted["code"][i],.9,axis=1)>=confirmed["code"]),axis=0
                  ).tolist()} for i,name in enumerate(encoded)}

    for name,scores in code.items():
        equal(scores["rmse"],scorer["effective_code"][name]["rmse"])
        equal(scores["coverage80_actual_fraction"],scorer["effective_code"][name]["central_80_coverage"])

    projected = predicted["planning"]
    option = np.array([0]*n)
    outcome = {}
    computed_choice = []
    computed_prob = []
    for k,(lo,hi) in enumerate(freeze["targets"]):
        prediction, tiebreak = evaluate_candidates(projected,lo,hi)
        independent_choice = np.array([[pick(a,b) for a,b in zip(m,sl)]
                                       for m,sl in zip(prediction,tiebreak)])
        computed_choice.append(independent_choice)
        computed_prob.append(prediction)
        achieved, realized_tiebreak = evaluate_candidates(actual,lo,hi)
        verified_option = np.array([pick(a,b) for a,b in zip(achieved,realized_tiebreak)])
        row = np.arange(n)
        realized = achieved[row[None],independent_choice]
        pooled_data = np.stack([projected[nominal,j] for j in range(n)],axis=1).reshape(13,-1,6,10)
        pop_prob, pop_margin = evaluate_candidates(pooled_data,lo,hi)
        pop_choice = pick(pop_prob,pop_margin)
        assert pool["actions"][k] == pop_choice
        outcome_real = np.vstack([realized,achieved[:,0],achieved[:,pop_choice]])
        noedit, pooled_nom = len(method), len(method)+1
        outcome_names = [*method,"no_edit","pooled_nominal"]
        source = scorer["inverse"][objective[k]]
        equal(source["oracle_full_menu_hit"],achieved.max(axis=1).mean())
        assert source["feasible_cards"] == int((achieved.max(axis=1)>0).sum())
        for m,name in enumerate(outcome_names):
            pred_score = source["methods"][name]
            equal(pred_score["hit_rate"],outcome_real[m].mean())
            equal(pred_score["per_card_hit"],outcome_real[m])
            equal(pred_score["mean_regret_to_finite_oracle"],(achieved.max(1)-outcome_real[m]).mean())
            equal(pred_score["delta_vs_pooled_nominal"]["ci_95"],recalibrate(outcome_real[m]-outcome_real[pooled_nom]))
            equal(pred_score["delta_vs_no_edit"]["ci_95"],recalibrate(outcome_real[m]-outcome_real[noedit]))
            equal(pred_score["delta_vs_rbf"]["ci_95"],recalibrate(outcome_real[m]-outcome_real[method.index("rbf")]))
        detail = {
            "no_edit":float(achieved[:,0].mean()),
            "posthoc_menu_upper":float(achieved[row,verified_option].mean()),
            "pooled_nominal_hit":float(achieved[:,pop_choice].mean()),
            "pooled_nominal_choice":pop_choice,
            "n_cases_no_success_in_menu":int((achieved.max(axis=1)==0).sum()),
            "method":{name:{"joint_hit":float(realized[m].mean()),
                            "gain_vs_no_edit":paired_stats(realized[m]-achieved[:,0]),
                            "gain_vs_nominal":paired_stats(realized[m]-realized[nominal]),
                            "choices":independent_choice[m].tolist()}
                      for m,name in enumerate(method)}
        }
        outcome[objective[k]] = detail
    decision = np.stack(computed_choice,axis=1)
    assert np.array_equal(decision, np.asarray(freeze["selected_actions"]))
    assert np.array_equal(decision, predicted["choice"])
    assert np.allclose(np.stack(computed_prob,axis=1), predicted["probability"],rtol=0,atol=1.e-14)
    expected_journals = set()
    for i, per_card in enumerate(decision.transpose(2,0,1)):
        for j in np.unique(np.r_[option[i],per_card.ravel()]):
            expected_journals.add((i,int(j),0))
    received_journals = {tuple(k) for k in confirmed["journal_keys"]}
    assert len(received_journals) == len(confirmed["journal_keys"])
    assert expected_journals == received_journals
    offset = confirmed["journal_offsets"]
    assert len(offset) == len(expected_journals)+1
    assert offset[0] == 0 and offset[-1] == len(confirmed["journals"])
    assert (np.diff(offset)>=0).all()

    return {"audit_utc":datetime.now(timezone.utc).isoformat(),
            "frozen_sha256":known_hash, "predicted_sha256":freeze["predicted_sha256"],
            "confirmed_sha256":confirmation["sha256"],
            "audit_source_sha256":sha(Path(__file__)),
            "verified_choice_count":decision.size, "verified_seed_counts":{k:v.size for k,v in seed_sets.items()},
            "verified_parent_score_sha256":sha(root/"SCORED.json"),
            "expected_journal_count":len(expected_journals), "forward":forward, "code":code, "inverse":outcome,
            "energy_by_card":risk.tolist()}


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--run", type=Path, default=HERE)
    cli.add_argument("--freeze-sha256", required=True)
    cli.add_argument("--out", type=Path)
    args = cli.parse_args()
    checked = audit(args.run,args.freeze_sha256)
    with (args.out or args.run/"INDEPENDENT_SCORE_AUDIT.json").open("x",encoding="utf-8") as report:
        json.dump(checked,report,indent=2,allow_nan=False)
    print(json.dumps({k:checked[k] for k in ("verified_choice_count","expected_journal_count",
                                            "verified_seed_counts","forward")},indent=2))
