"""Frozen tests of reaction clock, topology choice and inverse reaction histories."""

import argparse
from itertools import product
from concurrent.futures import ProcessPoolExecutor
from datetime import timedelta, timezone
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist, pdist

from aam_contact_environment_study import BASE_RATE, checksum, now, read_json, save, write_json
from aam_mbaa_kinetics import Recipe
from aam_network_events import CAP, EVENT_COLUMNS, PENDANT_TAGS, simulate
from aam_network_path_metrics import METRICS, path_metrics
from aam_spatial_motion import Motion


FAMILIES = ("null", "clock_only", "allocation")
METHODS = ("distance", "shared", "lineage")
SOURCE = ("aam_network_competition_study.py", "aam_network_events.py", "aam_network_path_metrics.py",
          "aam_contact_environment_study.py", "aam_contact_environment.py", "aam_mbaa_kinetics.py",
          "aam_spatial_motion.py")
LEVELS = np.arange(CAP+1)
PENALTY = .15
TOPOLOGY = (2, 3, 4, 5, 6)


def register():
    start = now()
    run = Path("runs")/start.strftime("aam_network_competition_%Y%m%dT%H%M%S%fZ")
    run.mkdir()
    common = dict(aam=70, mbaa=12, initial_mbaa=12, feed_time=0., initiators=9, horizon=14.)
    development = {"batch": common, "pulse": {**common, "initial_mbaa": 4, "feed_time": 2.}}
    held = {"lean_late": {**common, "aam": 68, "mbaa": 10, "initial_mbaa": 3, "feed_time": 3.45},
            "rich_batch": {**common, "aam": 75, "mbaa": 14, "initial_mbaa": 14}}
    inverse = {f"feed_{first}_{str(pulse).replace('.', 'p')}":
               {**common, "aam": 72, "initial_mbaa": first, "feed_time": pulse}
               for first, pulse in product((2, 6, 10), (1.25, 2.85, 4.55))}
    inverse["all_at_once"] = {**common, "aam": 72}
    target = {"early_time": 2.0, "dpw_lower": 8., "min_terminal_conversion": .82,
              "winding_conversion_interval": [.56, .88]}
    plan = {
        "utc": start.isoformat(), "asia_shanghai": start.astimezone(timezone(timedelta(hours=8))).isoformat(),
        "source_hashes": {path: checksum(path) for path in SOURCE},
        "question": "Can holding total pendant hazard fixed while learning event allocation improve independently "
                    "held equal-progress topology and the selection of a feed program for a microscopic target?",
        "materials_scope": "Dimensionless mobile synthetic AAm/MBAA analogue; no same-site divinyl loops, atomistic "
                           "orientation, experimentally fixed rate or wet validation.",
        "simulation": {"motion": vars(Motion()), "rate": BASE_RATE, "capture_context_radius": .82},
        "event_columns": EVENT_COLUMNS, "metrics": METRICS,
        "identity": "Kinetic chain assigned on initiation; propagated to the attacked vinyl; no ID merge on "
                    "combination. A pendant site's first reacted sibling supplies its historic chain ID. "
                    "Winding uses persistent integer bond-image shifts at birth.",
        "reference": "Shared known primitive distance hazard. Null uses raw weights. Clock-only multiplies all "
                     "stochastic weights by 0.52. Allocation redistributes only within the pendant hazard by "
                     "smooth saturating class-dependent occupancy, keeping its sum unchanged for every state. "
                     "All stages can use the same declared primitive clock; no extra calibration data.",
        "models": "All learn one global clock factor from event count over integrated primitive hazard. Conditional "
                  "pendant fits share two lineage intercepts and additionally use no, a common, or three distinct "
                  "binary occupancy multipliers. Positive-control teacher varies smoothly with occupancy count.",
        "primary": "Held conditional three-class pendant log score, then equal-progress topology energy, with "
                   "both full-feed reach and conditional CRPS reported. Group by independent simulated paths. "
                   "Inverse target is an additional blinded candidate ranking; report it even if gate fails.",
        "reference_parameters": {kind: {"factor": reference(kind)[0].tolist(), "clock": reference(kind)[1]}
                                 for kind in FAMILIES},
        "clock": np.arange(0., 14.01, .25).tolist(), "levels": [.56, .67, .78, .88],
        "dev": development, "held": held, "inverse": inverse, "inverse_target": target,
        "streams": {"dev": [130115001, 140], "forecast": [230125101, 144], "reference": [330136021, 180],
                    "inverse_forecast": [430146031, 160], "inverse_reference": [530166021, 200]},
    }
    write_json(run/"REGISTERED.json", plan)
    print(run, flush=True)
    return run


def reference(kind):
    if kind != "allocation":
        return np.ones((3, CAP+1)), .52 if kind == "clock_only" else 1.
    raw = LEVELS/(.8+LEVELS)
    contrast = np.array([-2.35, 2.05, .65])[:, None]*raw
    return np.exp(contrast), 1.


def extract_risk(events, risk):
    tag = events[:, 1].astype(int)
    which = np.isin(tag, PENDANT_TAGS)
    channel = np.searchsorted(np.asarray(PENDANT_TAGS), tag[which])
    level = events[which, 7].astype(int)
    if np.any(risk[which][np.arange(len(level)), channel, level] <= 0.):
        raise ValueError("Selected pendant cell must appear in the risk set.")
    return risk[which], np.column_stack((channel, level))


def path_job(job):
    recipe, motion, rate, factors, scale, seed, clock, levels, corridor, example = job
    events, risk, hazard, integrated = simulate(recipe, motion, rate, factors, scale, seed, clock, corridor)
    timed, matched, hit = path_metrics(recipe, events, clock, levels, with_first_winding=True)
    opportunity, selection = extract_risk(events, risk)
    row = dict(timed=timed, matched=matched, first_winding=hit, n_reactions=int(np.count_nonzero(events[:, 1])),
               exposure=integrated, opportunity=opportunity, selection=selection)
    raw = dict(events=events, risk_at_event=risk, primitive_hazard=hazard, clock_exposure=integrated, seed=seed)
    return row, raw if example else None


def collect(pool, plan, stage, family, name, index, setup, factors, scale, examples=False):
    base, count = plan["streams"][stage]
    first = base+100000*FAMILIES.index(family)+1000*index
    seeds = range(first, first+count)
    recipe = Recipe(**setup)
    settings = plan["simulation"]
    inputs = ((recipe, Motion(**settings["motion"]), np.asarray(settings["rate"]), factors, scale,
               seed, np.asarray(plan["clock"]), plan["levels"], settings["capture_context_radius"],
               examples and seed == first) for seed in seeds)
    rows = list(pool.map(path_job, inputs, chunksize=4))
    samples = [row for row, _ in rows]
    result = {name: np.stack([row[name] for row in samples]) for name in samples[0]
              if name not in ("opportunity", "selection")}
    sizes = np.asarray([len(row["selection"]) for row in samples])
    result.update(opportunity=np.concatenate([row["opportunity"] for row in samples]),
                  selection=np.concatenate([row["selection"] for row in samples]),
                  group=np.repeat(np.arange(count), sizes), seeds=np.asarray(seeds))
    print(stage, family, name, flush=True)
    return result, rows[0][1]


def bind(arrays, name, result):
    arrays.update({f"{name}_{k}": value for k, value in result.items()})


def collect_reference(run, pool, plan, stage, groups, families):
    arrays = {}
    receipt = {}
    for family in families:
        factors, scale = reference(family)
        for index, (name, setup) in enumerate(plan[groups].items()):
            result, example = collect(pool, plan, stage, family, name, index, setup, factors, scale, True)
            bind(arrays, f"{family}_{name}", result)
            path = run/f"{stage}_{family}_{name}_EXAMPLE.npz"
            save(path, **example)
            receipt[path.name] = checksum(path)
    return arrays, receipt


def develop(run, pool, plan):
    t = now().isoformat()
    state, examples = collect_reference(run, pool, plan, "dev", "dev", FAMILIES)
    output = run/"DEVELOP.npz"
    save(output, **state)
    write_json(run/"DEVELOP_RECEIPT.json", {"started_utc": t, "completed_utc": now().isoformat(),
                 "development_sha256": checksum(output), "registration_sha256": checksum(run/"REGISTERED.json"),
                 "examples": examples})


def design(method):
    features = np.zeros((3, CAP+1, 2+(0 if method == "distance" else 1 if method == "shared" else 3)))
    features[1, :, 0] = 1.
    features[2, :, 1] = 1.
    if method == "shared":
        features[:, 1:, 2] = 1.
    elif method == "lineage":
        for i in range(3):
            features[i, 1:, 2+i] = 1.
    return features.reshape(3*(CAP+1), -1)


def fit_mark(opportunity, selection, method):
    matrix = design(method)
    exposure = opportunity.reshape((-1, len(matrix)))
    selected_features = matrix[selection[:, 0]*(CAP+1)+selection[:, 1]].sum(axis=0)

    def loss(theta):
        eta = matrix @ theta
        scaled = exposure*np.exp(eta)
        norm = scaled.sum(axis=1)
        probabilities = scaled/norm[:, None]
        nll = np.log(norm).sum()-selected_features @ theta
        gradient = probabilities.sum(axis=0) @ matrix-selected_features
        return nll+.5*PENALTY*(theta@theta), gradient+PENALTY*theta

    result = minimize(loss, np.zeros(matrix.shape[1]), jac=True, method="L-BFGS-B",
                      bounds=[(-5., 5.)]*matrix.shape[1], options={"gtol": 1.e-6, "ftol": 1.e-13})
    if not result.success:
        raise RuntimeError(f"Mark fit stopped: {result.message}")
    return np.exp(matrix @ result.x).reshape((3, CAP+1)), result.x


def fits(dev, plan):
    output = {}
    for family in FAMILIES:
        names = [f"{family}_{recipe}" for recipe in plan["dev"]]
        union = lambda field: np.concatenate([dev[f"{name}_{field}"] for name in names])
        opportunity, selection = union("opportunity"), union("selection")
        scale = float(union("n_reactions").sum()/union("exposure").sum())
        for method in METHODS:
            factor, coef = fit_mark(opportunity, selection, method)
            output[family, method] = dict(factor=factor, scale=scale, coef=coef)
    return output


def success(clock, timed, first, target):
    early_at = int(np.searchsorted(clock, target["early_time"]))
    lo, hi = target["winding_conversion_interval"]
    return ((timed[:, early_at, 1] >= target["dpw_lower"]) & (timed[:, early_at, 6] == 0.)
            & (first >= lo) & (first <= hi) & (timed[:, -1, 0] >= target["min_terminal_conversion"]))


def forecast(run, pool, plan):
    receipt = read_json(run/"DEVELOP_RECEIPT.json")
    if (checksum(run/"DEVELOP.npz") != receipt["development_sha256"]
            or checksum(run/"REGISTERED.json") != receipt["registration_sha256"]):
        raise ValueError("Development input changed.")
    write_json(run/"TRAINING_STARTED.json", {"utc": now().isoformat(), "input_sha256": receipt["development_sha256"],
               "register_sha256": receipt["registration_sha256"], "source_hashes": plan["source_hashes"]})
    arrays = {}
    with np.load(run/"DEVELOP.npz") as data:
        options = fits(data, plan)
    ranking = {method: {} for method in METHODS}
    for (family, method), fitted in options.items():
        result_prefix = f"{family}_{method}"
        for key, value in fitted.items():
            arrays[f"{result_prefix}_{key}"] = np.asarray(value)
        if family != "allocation":
            continue  # On these families all models are evaluated on held exact risk sets rather than resampled.
        for stage, field in (("forecast", "held"), ("inverse_forecast", "inverse")):
            for index, (name, setup) in enumerate(plan[field].items()):
                result, _ = collect(pool, plan, stage, family, f"{method}:{name}", index, setup,
                                    fitted["factor"], fitted["scale"])
                bind(arrays, f"{stage}_{method}_{name}", result)
                if field == "inverse":
                    ranking[method][name] = float(success(plan["clock"], result["timed"],
                                                         result["first_winding"], plan["inverse_target"]).mean())
    candidates = list(plan["inverse"])
    selected = {method: max(candidates, key=lambda name: ranking[method][name]) for method in METHODS}
    save(run/"FROZEN.npz", **arrays)
    write_json(run/"FORECAST_RECEIPT.json", {"utc": now().isoformat(), "forecast_sha256": checksum(run/"FROZEN.npz"),
               "training_sha256": checksum(run/"TRAINING_STARTED.json"), "predicted_inverse_probability": ranking,
               "frozen_choice": selected, "fixed_choice": "all_at_once", "ties": "first in registered catalogue"})


def reveal(run, pool, plan):
    freeze = read_json(run/"FORECAST_RECEIPT.json")
    if (checksum(run/"FROZEN.npz") != freeze["forecast_sha256"]
            or checksum(run/"TRAINING_STARTED.json") != freeze["training_sha256"]):
        raise ValueError("Frozen prediction or training receipt changed.")
    t = now().isoformat()
    held, examples = collect_reference(run, pool, plan, "reference", "held", FAMILIES)
    inverse, inverse_examples = collect_reference(run, pool, plan, "inverse_reference", "inverse", ("allocation",))
    save(run/"REFERENCE.npz", **held)
    save(run/"INVERSE_REFERENCE.npz", **inverse)
    write_json(run/"REFERENCE_RECEIPT.json", {"started_utc": t, "finished_utc": now().isoformat(),
               "forecast_receipt_sha256": checksum(run/"FORECAST_RECEIPT.json"),
               "reference_sha256": checksum(run/"REFERENCE.npz"),
               "inverse_sha256": checksum(run/"INVERSE_REFERENCE.npz"), "examples": examples | inverse_examples})


def interval(x):
    x = np.asarray(x)
    mean = x.mean(axis=0)
    width = 1.96*x.std(axis=0, ddof=1)/np.sqrt(len(x))
    return np.stack((mean, mean-width, mean+width), axis=-1).tolist()


def conditioning(opportunity, selection, factors):
    weighted = opportunity*factors
    norm = weighted.sum(axis=(1, 2))
    probability = weighted.sum(axis=2)/norm[:, None]
    labels = selection[:, 0]
    log_mark = np.log(probability[np.arange(len(labels)), labels])
    log_cell = np.log(weighted[np.arange(len(labels)), labels, selection[:, 1]]/norm)
    brier = ((probability-np.eye(3)[labels])**2).sum(axis=1)
    return log_mark, log_cell, brier


def topology_score(draw, truth, mbaa):
    reached = np.isfinite(draw[:, 0])
    observed = np.isfinite(truth[:, 0])

    def state(values, complete):
        readings = np.where(complete[:, None], values[:, TOPOLOGY], 0.)
        readings[:, 1:4] /= mbaa
        return np.column_stack((complete.astype(float), readings))

    predicted = state(draw, reached)
    observed_states = state(truth, observed)
    return cdist(observed_states, predicted).mean(axis=1)-pdist(predicted).sum()/len(predicted)**2


def matched_crps(draw, truth, mbaa):
    complete = draw[np.isfinite(draw[:, 0])][:, TOPOLOGY].copy()
    observed = truth[np.isfinite(truth[:, 0])][:, TOPOLOGY].copy()
    if not len(complete) or not len(observed):
        return None
    complete[:, 1:4] /= mbaa
    observed[:, 1:4] /= mbaa
    deviation = np.abs(observed[:, None]-complete[None]).mean(axis=(0, 1))
    spread = np.abs(complete[:, None]-complete[None, :]).mean(axis=(0, 1))
    return (deviation-.5*spread).tolist()


def score_group(plan, reference_data, forecast_data, family, name):
    group = f"{family}_{name}"
    retrieve = lambda k: reference_data[f"{group}_{k}"]
    opportunity, selection, path = (retrieve(field) for field in ("opportunity", "selection", "group"))
    count, exposure = retrieve("n_reactions"), retrieve("exposure")
    metrics = {}
    scores = {}
    for method in METHODS:
        get = lambda k: forecast_data[f"{family}_{method}_{k}"]
        clock = float(get("scale"))
        l_class, l_cell, brier = conditioning(opportunity, selection, get("factor"))
        output = {"clock_gain_over_neutral": interval(count*np.log(clock)-(clock-1.)*exposure),
                  "clock_factor": clock, "coefficients": get("coef").tolist(),
                  "mean_log_mark_per_event": float(l_class.mean()), "mean_log_cell_per_event": float(l_cell.mean()),
                  "mean_brier_per_event": float(brier.mean())}
        scores[method] = {"log_mark": np.bincount(path, weights=l_class, minlength=len(count)),
                          "log_cell": np.bincount(path, weights=l_cell, minlength=len(count)),
                          "negative_brier": np.bincount(path, weights=-brier, minlength=len(count))}
        if family == "allocation":
            truth = retrieve("matched")
            raw = forecast_data[f"forecast_{method}_{name}_matched"]
            mbaa = plan["held"][name]["mbaa"]
            losses = np.column_stack([topology_score(p, q, mbaa)
                                       for p, q in zip(raw.transpose(1, 0, 2), truth.transpose(1, 0, 2), strict=True)])
            scores[method]["negative_topology_energy"] = -losses
            output["post_feed_reach_counts"] = np.isfinite(raw[:, :, 0]).sum(axis=0).tolist()
            output["matched_topology_energy"] = losses.mean(axis=0).tolist()
            output["conditional_matched_crps"] = [matched_crps(p, q, mbaa) for p, q in
                zip(raw.transpose(1, 0, 2), truth.transpose(1, 0, 2), strict=True)]
            timed = forecast_data[f"forecast_{method}_{name}_timed"]
            output["timed_conversion_mae"] = float(np.abs(timed[:, :, 0].mean(axis=0)
                                                      -retrieve("timed")[:, :, 0].mean(axis=0)).mean())
            output["winding_forecast_probability"] = float(timed[:, -1, 6].mean())
            output["posterior_event_mass_means_at_final_time"] = timed[:, -1, 2:7].mean(axis=0).tolist()
        metrics[method] = output
    changes = {comparison: {key: interval(scores["lineage"][key]-scores[comparison][key])
                            for key in scores["lineage"]} for comparison in METHODS[:-1]}
    return {"paths": len(count), "pendant_selected_counts": np.bincount(selection[:, 0], minlength=3).tolist(),
            "true_post_feed_reach": np.isfinite(retrieve("matched")[:, :, 0]).sum(axis=0).tolist(),
            "topology_metric_order": [METRICS[i] for i in TOPOLOGY],
            "true_winding_probability": float(retrieve("timed")[:, -1, 6].mean()),
            "true_event_mass_means_at_final_time": retrieve("timed")[:, -1, 2:7].mean(axis=0).tolist(),
            "methods": metrics, "lineage_improvement": changes}


def inverse_score(plan, reference_data, freeze):
    desired = plan["inverse_target"]
    reached = {}
    components = {}
    for name in plan["inverse"]:
        key = f"allocation_{name}"
        timed, first = (reference_data[f"{key}_{k}"] for k in ("timed", "first_winding"))
        reached[name] = success(plan["clock"], timed, first, desired)
        lo, hi = desired["winding_conversion_interval"]
        early = plan["clock"].index(desired["early_time"])
        components[name] = {"ever_winding": int(np.isfinite(first).sum()),
                             "winding_in_target": int(((first >= lo) & (first <= hi)).sum()),
                             "early_DPw_pass": int((timed[:, early, 1] >= desired["dpw_lower"]).sum())}
    empirical = {name: float(value.mean()) for name, value in reached.items()}
    best = max(empirical, key=empirical.get)
    selected = {"fixed": freeze["fixed_choice"], **freeze["frozen_choice"]}
    return {"success_probabilities": empirical, "oracle_candidate": best, "oracle_empirical": empirical[best],
            "distinct_selections": len(set(selected.values())),
            "per_recipe_n": len(next(iter(reached.values()))), "details": components,
            "selections": {method: {"name": name, "success": int(reached[name].sum()),
                                    "probability": empirical[name], "oracle_regret": empirical[best]-empirical[name]}
                           for method, name in selected.items()}}


def score(run, plan):
    receipt = read_json(run/"REFERENCE_RECEIPT.json")
    freeze = read_json(run/"FORECAST_RECEIPT.json")
    if (receipt["forecast_receipt_sha256"] != checksum(run/"FORECAST_RECEIPT.json")
            or freeze["forecast_sha256"] != checksum(run/"FROZEN.npz")
            or receipt["reference_sha256"] != checksum(run/"REFERENCE.npz")
            or receipt["inverse_sha256"] != checksum(run/"INVERSE_REFERENCE.npz")):
        raise ValueError("Frozen forecast or reference hashes changed.")
    with (np.load(run/"FROZEN.npz") as forecast_data, np.load(run/"REFERENCE.npz") as reference_data,
          np.load(run/"INVERSE_REFERENCE.npz") as inverse_data):
        scores = {family: {name: score_group(plan, reference_data, forecast_data, family, name)
                           for name in plan["held"]} for family in FAMILIES}
        inverse = inverse_score(plan, inverse_data, freeze)
    write_json(run/"SCORED.json", {"completed_utc": now().isoformat(), "reference_receipt_sha256":
               checksum(run/"REFERENCE_RECEIPT.json"), "interval": "Paired normal MC 95% across new reference "
               "paths with frozen predictive ensembles; descriptive, no multiple-comparison correction.",
               "scores": scores, "inverse": inverse})
    print(run/"SCORED.json", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("register", "develop", "forecast", "reference", "score", "all"))
    p.add_argument("run", type=Path, nargs="?")
    p.add_argument("--workers", type=int, default=6)
    args = p.parse_args()
    if args.action == "register":
        register()
        return
    run = register() if args.action == "all" else args.run
    plan = read_json(run/"REGISTERED.json")
    if any(checksum(name) != saved for name, saved in plan["source_hashes"].items()):
        raise ValueError("Source mismatch.")
    if args.action == "score":
        score(run, plan)
        return
    operations = {"develop": develop, "forecast": forecast, "reference": reveal}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for name in operations if args.action == "all" else (args.action,):
            operations[name](run, pool, plan)
    if args.action == "all":
        score(run, plan)


if __name__ == "__main__":
    main()

