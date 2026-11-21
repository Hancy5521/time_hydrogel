"""Read back the learned first-camera tape proposal without native replay."""

from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.linalg import lstsq, solve_triangular
from scipy.spatial.distance import cdist
from scipy.special import logsumexp
from threadpoolctl import threadpool_limits

from aam_formation_actual_rollout import packet_fingerprint
from aam_formation_cycle_data import check_actual_state
from aam_formation_cycle_driver import np_load, read, sha, write
from aam_formation_path_cycle import load_tree, stamp
from audit_aam_assay_guided_sir import _camera
from audit_aam_camera_information_sandwich import independent_apertures
from audit_aam_endpoint_tape_move import (
    ATTEMPTS, CELLS, ROOT, artifact, close, half_direction, intervals95, require, rng,
)


NS = "aam-camera-gaussian-first-tape-conditional-proposal-v1"
SCHEMA = "aam-first-camera-gaussian-brownian-projection-v1"
UNITS = tuple(range(2026135000, 2026135064))
METHODS = ("prior", "own", "crossed")
KEY = 189370086
DRAWS = 64
PRIOR = .25


def camera_axes(packet):
    """Direct piecewise gradient and Gram-Schmidt in flattened space."""
    state, recipe, geom, *_ = packet
    first, end = int(recipe[0]), int(recipe[0]+recipe[2])
    vectors = np.zeros((2, len(state.points), 3))
    for i, pos in enumerate((1.45, 4.45)):
        dx, dy = ((state.points[first:end, :2] - [pos, 1.35] + geom[2]/2.) % geom[2] - geom[2]/2.).T
        in_window = (abs(dx) < 1.3) & (abs(dy) < 1.3)
        vectors[i, first:end, 0] = np.where(in_window, -np.sign(dx)*(1.-abs(dy)/1.3), 0.)
        vectors[i, first:end, 1] = np.where(in_window, -np.sign(dy)*(1.-abs(dx)/1.3), 0.)
    basis = vectors.reshape(2, -1)
    basis[0] /= max(np.linalg.norm(basis[0]), 1.e-12)
    basis[1] -= (basis[0] @ basis[1])*basis[0]
    basis[1] /= max(np.linalg.norm(basis[1]), 1.e-12)
    active = np.flatnonzero(np.linalg.norm(basis, axis=1) > .5)
    return active, basis[active]


def check_packet(packet, cfg, age):
    st, recipe, geom, rates, factors = packet
    check_actual_state(st, expected_age=age)
    np.testing.assert_array_equal(recipe, [cfg["recipe"][k] for k in ("aam","mbaa","initial_mbaa","feed_time")])
    geometry = [cfg["motion"][k] for k in ("step","diffusion","box","spring","rest","repulsion","core","capture","inner")]
    np.testing.assert_array_equal(geom, [*geometry,cfg["corridor"]])
    np.testing.assert_array_equal(rates,cfg["rates"])
    np.testing.assert_array_equal(factors,cfg["factors"])


def source_table(cfg, unit, address=(), full=False):
    direction = half_direction(cfg,0)
    n = cfg["recipe"]["aam"]+cfg["recipe"]["mbaa"]
    address = (*ROOT,2,unit,0,*address)
    slabs = len(direction)//2
    tape = {"brownian":rng(*address,0).standard_normal((slabs,2,n,3))}
    if full:
        shape = (slabs, ATTEMPTS)
        tape.update(waiting=rng(*address,1).exponential(size=shape),
                    mark=rng(*address,2).integers(CELLS,size=shape,dtype=np.int64))
    return direction, tape


def model_close(archive, residual, coefficients, noise):
    require(archive["schema"]==SCHEMA and archive["training_mothers"]==len(residual)
            and archive["ridge"]==8. and archive["prior_weight"]==PRIOR
            and archive["covariance_prior_weight"]==.25, "Frozen learner target or shrinkage changed.")
    np.testing.assert_array_equal(archive["noise"],noise)
    codes = np.isfinite(coefficients) @ [1,2]
    require(set(archive["groups"])=={"1","2","3"}, "Wrong rank groups.")
    for key, observed in archive["groups"].items():
        code = int(key)
        mask = codes == code
        n = int(mask.sum())
        require(observed["rows"]==n and observed["trained"] is (n>=24), "Wrong active rank counts.")
        if n<24:
            continue
        a = [i for i in range(2) if code & (1<<i)]
        x = residual[mask] / noise
        y = coefficients[mask][:,a]
        xbar, ybar = x.mean(axis=0), y.mean(axis=0)
        xd, yd = x-xbar, y-ybar
        # Augment least squares with exact camera noise and ridge rows.
        left = np.vstack((xd,np.sqrt(n+8.)*np.eye(4)))
        right = np.vstack((yd,np.zeros((4,len(a)))))
        beta = lstsq(left,right)[0]
        err = np.vstack((yd-xd @ beta,np.sqrt(n)*beta))
        cov = .75*(err.T @ err)/n + .25*np.eye(len(a))
        factor = np.asarray(observed["factor"])
        require(np.array_equal(factor,np.tril(factor)) and np.all(np.diag(factor)>0.),
                "Conditional factor is not a positive Cholesky.")
        for got,want in ((observed["predictor_mean"],xbar),(observed["target_mean"],ybar),
                         (observed["beta"],beta),(factor@factor.T,cov)):
            close(got,want)


def refit(out, registration, frozen):
    origin = Path(registration["source_root"])
    atlas = read(origin/"REGISTERED.json")
    qualified = read(origin/"INDEPENDENT_AUDIT.json")
    acquired = read(origin/"ACQUIRED.json")
    ids = registration["training_units"]
    require(len(ids)==len(set(ids))==512 and not set(UNITS).intersection(sum(atlas["units"].values(),[]))
            and ids==atlas["units"]["train"] and atlas["physics"]==registration["physics"]
            and acquired["groups"]["train"]==registration["training_rows"]
            and qualified["passed"] is True and qualified["registered_sha256"]==sha(origin/"REGISTERED.json")
            and frozen["held_read"]==0 and frozen["n_original_mothers"]==512,
            "Training did not use precisely audited atlas training mothers.")
    y, z = [], []
    cfg = registration["physics"]
    for unit, row in zip(ids,registration["training_rows"],strict=True):
        require(unit==row["unit"],"Shifted an atlas mother.")
        raw = load_tree(artifact(row["raw"],origin/"train"/f"{unit}_RAW.npz"))
        before, after = raw["packets"][:2]
        check_packet(before,cfg,0.)
        check_packet(after,cfg,cfg["prefix_clock"][1])
        close(raw["clean"][0],_camera(after))
        y.append(np.asarray(_camera(after))-np.asarray(_camera(before)))
        active, basis = camera_axes(before)
        v, tape = source_table(cfg,unit)
        values = np.full(2,np.nan)
        values[active] = basis @ (v @ tape["brownian"].reshape(len(v),-1))
        z.append(values)
    coords = np_load(artifact(frozen["coords"],out/"TRAIN_COORDINATES.npz"))
    np.testing.assert_array_equal(coords["unit"],ids)
    close(coords["coeff"],z)
    close(coords["residual"],y)
    model = load_tree(artifact(frozen["model"],out/"MODEL.npz"))
    model_close(model,np.asarray(y),np.asarray(z),np.asarray(cfg["noise"]))
    return model


def proposal(index, z, model, own_residual, prompt_residual, unit, draw, active):
    """Compute coefficients and weight independently in the active subspace."""
    code = sum(1<<int(k) for k in active)
    note = {"schema":SCHEMA,"rank":len(active),"active_axis_code":code,
            "uses_observation":False,"mixture_component":"prior",
            "orthogonal_residual_changed":False,"log_prior_minus_proposal":0.,
            "original_prior_coefficients":z,"proposed_coefficients":z}
    if index==0 or not code or not model["groups"][str(code)]["trained"]:
        return note
    group = model["groups"][str(code)]
    prompt = own_residual if index==1 else prompt_residual
    mean = group["target_mean"]+(prompt/model["noise"]-group["predictor_mean"]) @ group["beta"]
    random = rng(KEY,unit,draw,2)
    prior_branch = float(random.random())<PRIOR
    L = group["factor"]
    value = z if prior_branch else mean+L @ random.standard_normal(len(z))
    q = solve_triangular(L,value-mean,lower=True)
    constant = -.5*len(z)*np.log(2.*np.pi)
    base = constant-.5*(value @ value)
    learned = constant-.5*(q @ q)-np.log(np.diag(L)).sum()
    mix = logsumexp([np.log(PRIOR)+base,np.log(1.-PRIOR)+learned])
    note.update(uses_observation=True,mixture_component="prior" if prior_branch else "guided",
                proposed_coefficients=value,log_prior_minus_proposal=base-mix,
                prior_log_density=base,guided_log_density=learned,proposal_log_density=mix,conditional_mean=mean)
    return note


def check_anchor(note, expected, tables, v, basis, record, cfg, unit):
    require(note.keys()==expected.keys(),"Learned proposal saved a different density receipt.")
    for k,want in expected.items():
        if isinstance(want,(np.ndarray,float)):
            close(note[k],want)
        else:
            require(note[k]==want,"Role-addressed method or mixture flag changed: "+k)
    tape = record["addressed_table"]
    for k in ("waiting","mark"):
        np.testing.assert_array_equal(tape[k],tables[k])
    source = tables["brownian"].reshape(len(v),-1)
    value = note["proposed_coefficients"]
    initial = note["original_prior_coefficients"]
    expected = source+np.outer(v,(value-initial) @ basis)
    close(tape["brownian"].reshape(expected.shape),expected)
    receipt = record["ref"]
    require(receipt["native_calls"]==1 and receipt["log_ratio"]==0. and not receipt["tape"]["forced_mark_consumed"],
            f"Mother {unit} applied a proposal-specific reaction law.")
    for row in receipt["spans"]:
        address = row["slab"],row["attempt"]
        if row["exponential"] is not None:
            require(row["exponential"]==tables["waiting"][address],"Archived original waiting draw changed.")
        if not row["censored"]:
            require(row["uniform"]==tables["mark"][address]/CELLS and row["mark_tilt"] is None,
                    "Archived original categorical mark changed.")
    check_packet(record["packet"],cfg,cfg["prefix_clock"][1])


def proper_score(hidden, weight, truth):
    flat = np.asarray(hidden).reshape(len(weight),-1)
    xx = cdist(flat,flat)
    xy = cdist(flat,np.asarray(truth).reshape(1,-1))[:,0]
    return float((xy @ weight-.5*weight @ xx @ weight)/np.sqrt(flat.shape[1]))


def score_case(out, cfg, model, row, forecasts, observed, crossed):
    unit = row["unit"]
    require(forecasts["unit"]==unit and row["native_calls"]==3 and forecasts["native_calls"]==
            {"attempted":3*DRAWS,"completed":3*DRAWS,"failed":0}
            and np.isfinite(forecasts["seconds"]) and forecasts["seconds"]>0., "Wrong accounting or paired unit.")
    held, folder = out/"held",out/"forecast"/str(unit)
    initial = load_tree(artifact(row["initial"],held/f"{unit}_INITIAL.npz"))
    target = load_tree(artifact(row["outcome"],held/f"{unit}_OUTCOME.npz"))
    check_packet(initial,cfg,0.)
    require(packet_fingerprint(initial)==forecasts["owner"]==packet_fingerprint(target["packets"][0]),
            "Inference did not use the same frozen initial as all comparators.")
    close(initial[0].points,target["initial"]["points"])
    truth = target["packets"][1]
    check_packet(truth,cfg,cfg["prefix_clock"][1])
    clean = target["clean"][0]
    close(clean,_camera(truth))
    noisy = clean + np.asarray(cfg["noise"])*rng(*ROOT,3,unit).standard_normal((3,4))[0]
    np.testing.assert_array_equal(observed,noisy)
    a = np_load(artifact(forecasts["predictions"],folder/"PREDICTIONS.npz"))
    raw = load_tree(artifact(forecasts["raw"],folder/"READBACK.npz"))
    anchor = {(note["method"],note["j"]):note for note in raw}
    require(len(raw)==6 and set(anchor)=={(m,j) for m in METHODS for j in (0,DRAWS-1)},
            "Raw first/last trajectory snapshots missing.")
    signals, hidden, correction = a["cameras"],a["hidden"],a["corrections"]
    require(signals.shape==(3,DRAWS,4) and hidden.shape==(3,DRAWS,2,3) and correction.shape==(3,DRAWS)
            and np.isfinite(signals).all() and np.isfinite(hidden).all() and np.isfinite(correction).all()
            and np.min(hidden)>=0. and np.max(hidden)<=1.+2.e-14 and np.max(correction)<=-np.log(PRIOR)+1.e-12,
            "Prediction tensors left physical support or correction bound.")
    initial_signal = np.asarray(_camera(initial))
    own_residual, swapped_residual = observed-initial_signal,crossed-initial_signal
    active, basis = camera_axes(initial)
    for draw in range(DRAWS):
        v, tables = source_table(cfg,unit,(909,draw),draw in (0,DRAWS-1))
        x = tables["brownian"].reshape(len(v),-1)
        z = basis @ (v @ x)
        for index, name in enumerate(METHODS):
            q = proposal(index,z,model,own_residual,swapped_residual,unit,draw,active)
            close(correction[index,draw],q["log_prior_minus_proposal"])
            if draw in (0,DRAWS-1):
                record = anchor[name,draw]
                check_anchor(record["proposal"],q,tables,v,basis,record,cfg,unit)
                close(signals[index,draw],_camera(record["packet"]))
                close(hidden[index,draw],independent_apertures(record["packet"]))
    # All proposals use the genuine camera likelihood once, including swapped prompt.
    ll = -.5*np.sum(((signals-observed)/cfg["noise"])**2,axis=-1)
    log_weight = ll+correction
    weights = np.exp(log_weight-logsumexp(log_weight,axis=1,keepdims=True))
    close(weights.sum(axis=1),np.ones(3))
    losses = [proper_score(samples,w,independent_apertures(truth)) for samples,w in zip(hidden,weights,strict=True)]
    return losses,1./np.square(weights).sum(axis=1),ll.mean(axis=1)


def audit(out):
    names = ("REGISTERED","RUN","FIT_STARTED","FIT_FROZEN","ACQUIRE_STARTED","ACQUIRED",
             "PREDICTIONS_FROZEN","REVEAL_STARTED","SCORED")
    files = {name:out/f"{name}.json" for name in names}
    d = {name:read(path) for name,path in files.items()}
    times = [datetime.fromisoformat(doc["actual_utc"]) for doc in d.values()]
    require(all(a<b for a,b in zip(times[:-1],times[1:],strict=True))
            and all(a==datetime.fromisoformat(doc["asia_shanghai"]) for a,doc in zip(times,d.values(),strict=True)),
            "Frozen fit/prediction and outcome-opening chronology changed.")
    p, s = d["REGISTERED"],d["SCORED"]
    reg, fit, fit_end, acq, pred = (sha(files[k]) for k in
                                  ("REGISTERED","FIT_FROZEN","FIT_STARTED","ACQUIRED","PREDICTIONS_FROZEN"))
    f = d["PREDICTIONS_FROZEN"]
    require(p["namespace"]==NS and tuple(p["units"])==UNITS and tuple(p["methods"])==tuple(s["methods"])==METHODS
            and p["seed"]==KEY and p["draws_per_mother_and_method"]==DRAWS and p["prior_mixture"]==PRIOR
            and p["native_fresh_truth"]==3*len(UNITS) and p["native_fresh_forecast"]==3*DRAWS*len(UNITS)
            and p["old_held_opened"]==p["new_wet"]==s["new_wet_measurements"]==0
            and d["RUN"]["before_fit_and_simulation"] is True and s["full_blind_inverse_scored"] is False
            and d["FIT_STARTED"]["held_labels_read"]==0 and d["RUN"]["registration_sha256"]==
            d["FIT_STARTED"]["registration_sha256"]==s["registration_sha256"]==reg
            and d["ACQUIRE_STARTED"]["fitted_sha256"]==f["fit_sha256"]==fit
            and f["held_targets_opened"] is False and f["acquired_sha256"]==acq
            and d["REVEAL_STARTED"]["frozen_sha256"]==s["forecast_sha256"]==pred,
            "The verified fit, no-lookahead freeze, or registered marginal scope changed.")
    for path,digest in p["inputs"].items():
        require(sha(path)==digest,"Registered input changed: "+path)
    for path,digest in p["sources"].items():
        require(sha(out/"SOURCE_SNAPSHOT"/Path(path).name)==digest,"Registered source changed: "+path)
    model = refit(out,p,d["FIT_FROZEN"])
    actual = d["ACQUIRED"]["rows"]
    require(tuple(x["unit"] for x in actual)==tuple(x["unit"] for x in f["rows"])==UNITS,
            "Day-one forecast lost a preregistered mother.")
    y = [np_load(artifact(r["query"],out/"held"/f'{r["unit"]}_QUERY.npz'))["observed"] for r in actual]
    results = [score_case(out,p["physics"],model,a,b,y[i],y[i^1])
               for i,(a,b) in enumerate(zip(actual,f["rows"],strict=True))]
    losses, ess, ll = (np.asarray(a) for a in zip(*results,strict=True))
    summary = np_load(artifact(s["data"],out/"SCORED_ARRAYS.npz"))
    np.testing.assert_array_equal(summary["units"],UNITS)
    for got,want in ((summary["losses"],losses),(summary["ess"],ess),(summary["mean_raw_likelihood"],ll),
                     (s["mean_hidden_es"],losses.mean(axis=0)),(s["mean_ess"],ess.mean(axis=0))):
        close(got,want)
    prior_gain = intervals95(np.column_stack((losses[:,0]-losses[:,1],ess[:,1]-ess[:,0])))
    swapped = np.column_stack((losses[:,2]-losses[:,1],ess[:,1]-ess[:,2]))
    swap_gain = intervals95(swapped.reshape(32,2,2).mean(axis=1))
    close(s["paired_own_gain_over_original_prior"],prior_gain)
    close(s["paired_own_gain_over_swapped_prompt"],swap_gain)
    go = bool(np.min(prior_gain[:,1])>0.)
    require(s["go_to_blind_three_camera"] is go,"Registered strong-baseline gate changed.")
    return {**stamp(),"passed":True,"registration_sha256":reg,"scored_sha256":sha(files["SCORED"]),
            "prediction_sha256":pred,"fit_started_sha256":fit_end,"independent_training_mothers":512,
            "independent_assessment_mothers":len(UNITS),"independent_swapped_pairs":32,
            "all_corrections_recomputed":len(UNITS)*len(METHODS)*DRAWS,
            "raw_anchor_predictions_checked":6*len(UNITS),"new_native_calls":0,
            "proper_energy_and_ess_recalculated":True,"gate_for_new_blind_recovery":go,
            "scope":"Known initial component test. Audited source ordering, trajectories at saved anchors, and weighted outputs; no simulator reintegration."}


def fixtures():
    w = np.array([.25,.75])
    z = np.array([[0.],[4.]])
    close(proper_score(z,w,np.array([0.])),2.25)
    close(proper_score(np.ones((2,6)),w,np.zeros(6)),1.)
    active = np.array([0])
    model = {"groups":{"1":{"trained":True,"factor":np.eye(1),"target_mean":np.zeros(1),
                             "predictor_mean":np.zeros(4),"beta":np.zeros((4,1))}},"noise":np.ones(4)}
    b = proposal(1,np.array([.3]),model,np.zeros(4),np.ones(4),1,0,active)
    close(b["log_prior_minus_proposal"],0.)
    return "PASS independently computed mixture and energy-score fixtures"


if __name__=="__main__":
    cli = ArgumentParser(description=__doc__)
    route = cli.add_mutually_exclusive_group(required=True)
    route.add_argument("--fixtures",action="store_true")
    route.add_argument("--out",type=Path)
    args = cli.parse_args()
    with threadpool_limits(limits=1):
        if args.fixtures:
            print(fixtures())
        else:
            out = args.out.resolve(strict=True)
            write(out/"INDEPENDENT_AUDIT.json",audit(out))
            print("PASS independent learned-tape audit",out)
