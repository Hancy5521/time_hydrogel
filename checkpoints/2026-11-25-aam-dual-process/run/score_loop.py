"""Score precision on internal effective parameters and closed process objectives."""
from __future__ import annotations

import argparse
import numpy as np
from scipy.spatial.distance import cdist

from loop import ROOT, actions, digest, load, margins, rank, stable_rng, stamps, write


SCALE = np.array([.15,.10,.18,.18,.45])
CHANNEL = [0,2,3,5,9]
METHODS = ("ridge", "rbf", "shared_response", "nominal")


def pooled_freeze():
    f = load("FROZEN.json")
    artifact = ROOT/"PREDICTED.npz"
    assert f["predicted_sha256"] == digest(artifact)
    with np.load(artifact) as npz:
        nominal = npz["planning"][-1].transpose(1,0,2,3,4)
    x=nominal.reshape(1,nominal.shape[0],-1,6,10)
    choices=[int(rank(x,np.asarray(goal))[0][0]) for goal in f["targets"]]
    write("POOLED_NOMINAL.json",{**stamps(),"freeze":digest(ROOT/"FROZEN.json"),
                                "source":digest(ROOT/"score_loop.py"),"actions":choices})


def energy(x, actual):
    x, actual = (m[:,2:,CHANNEL].reshape(len(m),-1)/np.tile(SCALE,4) for m in (x,actual))
    return float((cdist(x,actual).mean()-.5*cdist(x,x).mean())/np.sqrt(x.shape[1]))


def summary_difference(left,right, bootstrap):
    diff = np.asarray(left) - np.asarray(right)
    means=diff[bootstrap].mean(axis=1)
    return {"estimate":float(diff.mean()),"ci_95":np.quantile(means,[.025,.975]).tolist()}


def analysis():
    f=load("FROZEN.json")
    c=load("CONFIRMED.json")
    pool=load("POOLED_NOMINAL.json")
    assert digest(ROOT/"FROZEN.json")==c["verified_prediction_sha256"]==pool["freeze"]
    assert pool["source"]==digest(ROOT/"score_loop.py")
    assert c["sha256"]==digest(ROOT/"CONFIRMED.npz") and f["predicted_sha256"]==digest(ROOT/"PREDICTED.npz")
    with np.load(ROOT/"CONFIRMED.npz") as real:
        path, code, mil = real["path"],real["code"],real["milestones"]
    with np.load(ROOT/"PREDICTED.npz") as model:
        forward, prediction, raw_choices = model["forward"],model["code"],model["choice"]
    assert np.isfinite(path).all() and np.all(np.diff(path[...,0],axis=-1)>=-1e-12)
    assert np.array_equal(raw_choices,f["selected_actions"])
    card_count, act_count, n = path.shape[:3]
    samples=stable_rng("score card bootstrap").integers(0,card_count,(5000,card_count))
    code_error={m:{"rmse":np.sqrt(((prediction[k].mean(axis=1)-code)**2).mean(axis=0)).tolist(),
                   "prior_rmse":np.sqrt((code**2).mean(axis=0)).tolist(),
                   "central_80_coverage":np.mean((code>=np.quantile(prediction[k],.1,axis=1))&
                                                 (code<=np.quantile(prediction[k],.9,axis=1)),axis=0).tolist()}
                 for k,m in enumerate(METHODS[:-1])}
    e={name:np.asarray([energy(y,t) for y,t in zip(arr,path[:,0])]) for name,arr in zip(METHODS,forward)}
    active={
        name:{"energy":float(err.mean()),"per_card":err.tolist(),
              "delta_vs_nominal":summary_difference(err,e["nominal"],samples),
              "delta_vs_rbf":summary_difference(err,e["rbf"],samples)}
        for name,err in e.items()
    }
    whole_choice=np.concatenate((raw_choices,np.zeros((1,2,card_count),dtype=int),
        np.repeat(np.asarray(pool["actions"])[None,:,None],card_count,axis=2)),axis=0)
    names=(*METHODS,"no_edit","pooled_nominal")
    output={}
    for goal, window in enumerate(f["targets"]):
        slack=margins(path,np.asarray(window))
        success=slack>=-1e-12
        rates=success.mean(axis=-1)
        best=rates.max(axis=1)
        selected=success[np.arange(card_count)[None,:],whole_choice[:,goal]]
        rates_chosen=selected.mean(axis=2)
        output[f["objective_names"][goal]]={
            "feasible_cards":int(np.count_nonzero(best>0)),
            "oracle_full_menu_hit":float(best.mean()),
            "bound":window,
            "methods":{
                name:{"hit":int(selected[i].sum()),"runs":int(selected[i].size),
                      "hit_rate":float(rates_chosen[i].mean()),"per_card_hit":rates_chosen[i].tolist(),
                      "distinct_edits":np.unique(whole_choice[i,goal]).tolist(),
                      "delta_vs_no_edit":summary_difference(rates_chosen[i],rates_chosen[len(names)-2],samples),
                      "delta_vs_pooled_nominal":summary_difference(rates_chosen[i],rates_chosen[len(names)-1],samples),
                      "delta_vs_rbf":summary_difference(rates_chosen[i],rates_chosen[1],samples),
                      "mean_regret_to_finite_oracle":float((best-rates_chosen[i]).mean())}
                for i,name in enumerate(names)
            }
        }
    delta_action, labels=actions()
    # Isolated bridge intervention, every actual g and innovation shared,
    # using the raw, independently restarted full card/menu truth.
    bridge_up, bridge_down=labels.index("bridge_up"),labels.index("bridge_down")
    paired=path[:,bridge_up]-path[:,bridge_down]
    isolated={"final_bridge_fraction_delta":summary_difference(paired[:,-1,5] if paired.ndim==3
                          else paired[:,:,-1,5].mean(axis=1),np.zeros(card_count),samples),
              "final_conversion_delta":summary_difference(paired[:,:,-1,0].mean(axis=1),np.zeros(card_count),samples),
              "final_largest_delta":summary_difference(paired[:,:,-1,3].mean(axis=1),np.zeros(card_count),samples),
              "final_wrap_delta":summary_difference(paired[:,:,-1,9].mean(axis=1),np.zeros(card_count),samples)}
    reached_both=(mil[:,bridge_up,:,:,0]>0)&(mil[:,bridge_down,:,:,0]>0)
    milestone_diff=mil[:,bridge_up]-mil[:,bridge_down]
    for lev,level_name in enumerate(("60", "76")):
        w=reached_both[...,lev]
        total=w.sum()
        isolated[f"equal_conversion_{level_name}"]={
            "paired_both_observed":int(total),"paired_catalogue":int(w.size),
            "largest_delta_conditional":float(milestone_diff[...,lev,3][w].mean()) if total else None,
            "winding_delta_conditional":float(milestone_diff[...,lev,4][w].mean()) if total else None,
        }
    result={**stamps(),"freeze":c["verified_prediction_sha256"],"truth_sha256":c["sha256"],
            "score_source_sha256":digest(ROOT/"score_loop.py"),
            "effective_code":code_error,"new_environment_energy":active,"inverse":output,
            "bridge_intervention_isolated":isolated,
            "micro":{
                "n_fresh_process_paths":int(path.shape[0]*path.shape[1]*path.shape[2]),
                "n_blind_cards":card_count,"true_nonzero_converter_paths":int((path[:,:,:,-1,0]>0).sum()),
                "true_nonzero_bridge_paths":int((path[:,:,:,-1,5]>0).sum()),
                "true_winding_at_final":int((path[:,:,:,-1,9]>0).sum()),
                "mean_winding_unedited":path[:,0,:,:,9].mean(axis=(0,1)).tolist(),
                "reaches_by_unedited_card":mil[:,0,:,:,0].mean(axis=(0,1)).tolist(),
            }}
    write("SCORED.json",result)
    print("effective codes",code_error,flush=True)
    print("energies",{m:r["energy"] for m,r in active.items()},flush=True)
    print("inverse",{goal:{m:card["hit"] for m,card in out["methods"].items()} for goal,out in output.items()},flush=True)
    print("micro",result["micro"],flush=True)


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage",choices=("pooled-freeze","score"))
    args=parser.parse_args()
    (pooled_freeze if args.stage=="pooled-freeze" else analysis)()
