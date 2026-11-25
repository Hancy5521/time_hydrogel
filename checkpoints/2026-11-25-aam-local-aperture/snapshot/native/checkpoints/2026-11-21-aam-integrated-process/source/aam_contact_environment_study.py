"""Fit and check a local occupancy correction in held mobile reaction histories."""

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timedelta, timezone
from hashlib import file_digest
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar

from aam_contact_environment import simulate_environment
from aam_mbaa_kinetics import METRICS, Recipe, checkpoints
from aam_spatial_motion import Motion, simulate_mobile


SOURCES = ("aam_contact_environment_study.py", "aam_contact_environment.py", "aam_spatial_motion.py",
           "aam_mbaa_kinetics.py")
REFERENCES = ("occupancy", "null")
METHODS = ("distance", "distance_and_occupancy")
COUNT_CAP = 4
PROPAGATION = slice(2, 6)
METRIC_IDS = (0, 2, 4, 5, 6)
BASE_RATE = [[.008, .008], [.009, .009], [8., 1.3], [10., 1.5], [5., .7], [.32, 2.6],
             [.3, .06], [.09, .02]]


def now():
    return datetime.now(timezone.utc)


def checksum(path):
    with open(path, "rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def write_json(path, payload):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def save(path, **arrays):
    with path.open("xb") as stream:
        np.savez_compressed(stream, **arrays)


def register():
    start = now()
    run = Path("runs")/start.strftime("aam_approach_occupancy_%Y%m%dT%H%M%S%fZ")
    run.mkdir()
    common = dict(aam=70, mbaa=6, initial_mbaa=6, feed_time=0., initiators=8, horizon=12.)
    development = {"batch": common, "pulse": {**common, "initial_mbaa": 3, "feed_time": 2.}}
    held = {"low_batch": {**common, "aam": 65, "mbaa": 4, "initial_mbaa": 4},
            "high_late": {**common, "aam": 74, "mbaa": 9, "initial_mbaa": 1, "feed_time": 3.6}}
    plan = {
        "utc": start.isoformat(),
        "asia_shanghai": start.astimezone(timezone(timedelta(hours=8))).isoformat(),
        "sources": {path: checksum(path) for path in SOURCES},
        "question": "Does one shared binary approach-occupancy correction improve reaction event timing and marks "
                    "and the later joint chain/network predictions at held compositions and feed histories?",
        "motion": vars(Motion()), "corridor_radius": .82, "count_cap": COUNT_CAP, "base_rate": BASE_RATE,
        "occupancy": "Count present third bead centers inside a periodic sphere about the approach midpoint, "
                     "excluding both endpoint molecules. Geometry fixed within each reaction substep.",
        "reference_rule": "Rate multiplier exp(-2.20*n/(.65+n)) in channels 3–6, with cap 4. "
                          "Null reference uses no multiplier. Units are dimensionless and input is synthetic.",
        "law": "Model A estimates a rate for each reaction channel/distance band. Model B adds one shared "
               "log-rate multiplier across growth/bridge/cycle when at least one third bead is present. "
               "The fit uses exact event counts and integrated rate opportunities.",
        "dev": development, "held": held, "clock": np.linspace(0., 12., 49).tolist(),
        "conversion": [.55, .70, .80], "metrics": METRICS,
        "streams": {"dev": [121009001, 96], "forecast": [221019201, 128], "reference": [321029400, 160]},
        "evaluation": "Held complete-path reaction log likelihood ratio; primary paired difference per path by "
                      "reference and recipe with independent trajectories. Forward CRPS and trajectory mean MAE "
                      "for conversion, kinetic chain DPw, largest component, bridges and cycles. Split streams "
                      "by recipe and reference. Compare reach fraction as well as largest component at 80% "
                      "post-feed conversion; the latter conditions on reaching. No model changes after freeze.",
        "event_columns": ["time", "tag", "source_site", "target_site_or_feed_molecule", "distance",
                          "contact_band", "geometry_time", "approach_count_capped"],
    }
    write_json(run/"REGISTERED.json", plan)
    print(run, flush=True)
    return run


def reference_rates(plan, kind):
    base = np.asarray(plan["base_rate"], dtype=float)
    levels = np.arange(COUNT_CAP+1)
    rates = np.repeat(base[..., None], len(levels), axis=-1)
    if kind == "occupancy":
        rates[PROPAGATION] *= np.exp(-2.2*levels/(.65+levels))
    return rates


def fit(count, exposure, add_occupancy):
    if count.shape != (8, 2, COUNT_CAP+1) or exposure.shape != count.shape:
        raise ValueError("Unexpected training risk array shape.")
    event_count = count.sum(axis=-1)
    opportunity = exposure.sum(axis=-1)
    if np.any(event_count[opportunity > 0] <= 0):
        raise ValueError("An observed channel/band has no fitted events.")
    empty = exposure[PROPAGATION, :, 0]
    crowded = exposure[PROPAGATION, :, 1:].sum(axis=-1)
    crowded_count = count[PROPAGATION, :, 1:].sum()

    def negative_profile(log_multiplier):
        weighted = empty+np.exp(log_multiplier)*crowded
        return np.sum(event_count[PROPAGATION]*np.log(weighted))-crowded_count*log_multiplier

    slope = (minimize_scalar(negative_profile, method="bounded", bounds=(-4., 4.)).x
             if add_occupancy else 0.)
    factors = np.ones_like(exposure)
    factors[PROPAGATION, :, 1:] = np.exp(slope)
    denominator = (exposure*factors).sum(axis=-1)
    intercept = np.divide(event_count, denominator, out=np.zeros_like(denominator), where=denominator > 0.)
    intercept[:2, 1] = intercept[:2, 0]
    return intercept[..., None]*factors, float(slope)


def simulate_job(job):
    recipe, motion, corridor, rate, clock, conversion, seed, record = job
    result = simulate_environment(recipe, motion, corridor, rate, seed, clock)
    events, positions, count, exposure, _, _, _, _, _ = result
    timed, matched, hit = checkpoints(recipe, events[:, :4].copy(), clock, conversion)
    if record:
        _, _, _, _, coordinates, images, present, force, energy = result
        raw = dict(events=events, event_position=positions, coordinates=coordinates, unwrapped=images,
                   present=present, force=force, energy=energy, count=count, exposure=exposure, seed=seed)
    else:
        raw = None
    return count, exposure, timed, matched, hit, raw


def collect(pool, plan, stage, rate, kind, index, recipe, record):
    start, draws = plan["streams"][stage]
    start += REFERENCES.index(kind)*10000+index*1000
    settings = Motion(**plan["motion"])
    jobs = ((Recipe(**recipe), settings, plan["corridor_radius"], rate, np.asarray(plan["clock"]),
             plan["conversion"], start+i, record and i == 0) for i in range(draws))
    rows = list(pool.map(simulate_job, jobs, chunksize=4))
    names = ("count", "exposure", "timed", "matched", "hit")
    return {name: np.stack([r[j] for r in rows]) for j, name in enumerate(names)}, rows[0][-1]


def update_group(storage, prefix, values):
    storage.update({f"{prefix}_{key}": value for key, value in values.items()})


def control(plan):
    recipe = Recipe(**plan["dev"]["pulse"])
    motion = Motion(**plan["motion"])
    rate = reference_rates(plan, "null")
    clock = np.asarray(plan["clock"])
    mobile = simulate_mobile(recipe, motion, rate[..., 0], 14995, clock)
    extended = simulate_environment(recipe, motion, plan["corridor_radius"], rate, 14995, clock)
    np.testing.assert_array_equal(mobile[0], extended[0][:, :7])
    np.testing.assert_array_equal(mobile[1], extended[1])
    np.testing.assert_array_equal(mobile[2], extended[2].sum(axis=-1))
    np.testing.assert_allclose(mobile[3], extended[3].sum(axis=-1), rtol=3e-14, atol=1e-14)
    for original, newer in zip(mobile[4:], extended[4:], strict=True):
        np.testing.assert_array_equal(original, newer)
    return {"identical_event_count": len(mobile[0]), "identical_frames": len(clock),
            "null_reduces_to_frozen_mobile_solver": True}


def develop(run, pool, plan):
    start = now().isoformat()
    arrays = {}
    for kind in REFERENCES:
        for index, (name, recipe) in enumerate(plan["dev"].items()):
            group, _ = collect(pool, plan, "dev", reference_rates(plan, kind), kind, index, recipe, False)
            update_group(arrays, f"{kind}_{name}", group)
            print("development", kind, name, flush=True)
    path = run/"DEVELOP.npz"
    save(path, **arrays)
    write_json(run/"DEVELOP_RECEIPT.json", {"started_utc": start, "finished_utc": now().isoformat(),
               "registered_sha256": checksum(run/"REGISTERED.json"), "arrays_sha256": checksum(path)})


def forecast(run, pool, plan):
    path = run/"DEVELOP.npz"
    dev = read_json(run/"DEVELOP_RECEIPT.json")
    reg = checksum(run/"REGISTERED.json")
    if checksum(path) != dev["arrays_sha256"] or reg != dev["registered_sha256"]:
        raise ValueError("Development receipt mismatch.")
    write_json(run/"TRAINING_STARTED.json", {"utc": now().isoformat(), "input_sha256": checksum(path),
               "registry_sha256": reg, "sources": plan["sources"]})
    arrays = {}
    with np.load(path) as development:
        for kind in REFERENCES:
            count, exposure = (sum(development[f"{kind}_{name}_{feature}"].sum(axis=0) for name in plan["dev"])
                               for feature in ("count", "exposure"))
            for method in METHODS:
                rate, slope = fit(count, exposure, method == METHODS[1])
                prefix = f"{kind}_{method}"
                arrays[f"{prefix}_rate"] = rate
                arrays[f"{prefix}_slope"] = np.asarray(slope)
                for index, (name, recipe) in enumerate(plan["held"].items()):
                    group, _ = collect(pool, plan, "forecast", rate, kind, index, recipe, False)
                    update_group(arrays, f"{prefix}_{name}", group)
                    print("frozen forecast", kind, method, name, flush=True)
    path = run/"FROZEN.npz"
    save(path, **arrays)
    write_json(run/"FORECAST_RECEIPT.json", {"utc": now().isoformat(), "frozen_sha256": checksum(path),
               "training_receipt_sha256": checksum(run/"TRAINING_STARTED.json"), "registered_sha256": reg})


def reveal(run, pool, plan):
    receipt = run/"FORECAST_RECEIPT.json"
    info = read_json(receipt)
    if (info["frozen_sha256"] != checksum(run/"FROZEN.npz")
            or info["registered_sha256"] != checksum(run/"REGISTERED.json")
            or info["training_receipt_sha256"] != checksum(run/"TRAINING_STARTED.json")):
        raise ValueError("Check frozen forecast before independent reference is generated.")
    start = now().isoformat()
    arrays = {}
    example_checks = {}
    for kind in REFERENCES:
        for index, (name, recipe) in enumerate(plan["held"].items()):
            prefix = f"{kind}_{name}"
            group, sample = collect(pool, plan, "reference", reference_rates(plan, kind), kind, index, recipe, True)
            update_group(arrays, prefix, group)
            path = run/f"{prefix}_TRAJECTORY.npz"
            save(path, **sample)
            example_checks[path.name] = checksum(path)
            print("reference", kind, name, flush=True)
    path = run/"REFERENCE.npz"
    save(path, **arrays)
    write_json(run/"REFERENCE_RECEIPT.json", {"started_utc": start, "finished_utc": now().isoformat(),
               "reference_sha256": checksum(path), "forecast_receipt_sha256": checksum(receipt),
               "example_sha256": example_checks})


def crps(draw, observed):
    table = np.sort(draw, axis=0)
    n = len(table)
    weights = (2*np.arange(n)-n+1).reshape((n,)+(1,)*(draw.ndim-1))
    dispersion = (table*weights).sum(axis=0)/n**2
    return (np.abs(observed[:, None]-draw[None]).mean(axis=1)-dispersion).mean(axis=1)


def mean_interval(values):
    values = np.asarray(values)
    error = 1.96*values.std(axis=0, ddof=1)/np.sqrt(len(values))
    center = values.mean(axis=0)
    return np.stack((center, center-error, center+error), axis=-1).tolist()


def score(run, plan):
    receipt = read_json(run/"REFERENCE_RECEIPT.json")
    if (receipt["reference_sha256"] != checksum(run/"REFERENCE.npz")
            or receipt["forecast_receipt_sha256"] != checksum(run/"FORECAST_RECEIPT.json")
            or read_json(run/"FORECAST_RECEIPT.json")["frozen_sha256"] != checksum(run/"FROZEN.npz")):
        raise ValueError("Reference or forecast receipt mismatch.")
    output = {"metric_order": [METRICS[i] for i in METRIC_IDS], "interval_columns": ["mean", "low", "high"],
              "intervals": "Normal 95% MC intervals across new independent reference paths, conditional on forecasts.",
              "sha256": {name: checksum(run/name) for name in
                         ("REGISTERED.json", "DEVELOP.npz", "FROZEN.npz", "REFERENCE.npz")}, "result": {}}
    with np.load(run/"FROZEN.npz") as forecast_values, np.load(run/"REFERENCE.npz") as reference:
        for kind in REFERENCES:
            estimates = [forecast_values[f"{kind}_{method}_rate"] for method in METHODS]
            slopes = [float(forecast_values[f"{kind}_{method}_slope"]) for method in METHODS]
            output["result"][kind] = {"fitted_multiplier": float(np.exp(slopes[1])), "groups": {}}
            for name in plan["held"]:
                prefix = f"{kind}_{name}"
                count = reference[f"{prefix}_count"]
                exposure = reference[f"{prefix}_exposure"]
                ratio = np.log(estimates[1]/estimates[0])
                ll = count*ratio-exposure*(estimates[1]-estimates[0])
                timed = reference[f"{prefix}_timed"][:, :, METRIC_IDS]
                matched = reference[f"{prefix}_matched"][:, -1, 4]
                report = {"paths": len(count),
                          "reactive_events": int(count.sum()),
                          "held_log_likelihood_gain": mean_interval(ll.sum(axis=(1, 2, 3))),
                          "gain_by_reaction_channel": mean_interval(ll.sum(axis=(2, 3))),
                          "occupied_growth_at_risk_fraction": float(exposure[:, PROPAGATION, :, 1:].sum()
                              /exposure[:, PROPAGATION].sum()),
                          "truth_reach_80": int(np.isfinite(matched).sum()),
                          "truth_largest_mass_at_80": float(np.nanmean(matched)),
                          "topology_scores": {}}
                losses = []
                for method in METHODS:
                    base = f"{kind}_{method}_{name}"
                    draw = forecast_values[f"{base}_timed"][:, :, METRIC_IDS]
                    prediction = forecast_values[f"{base}_matched"][:, -1, 4]
                    error = crps(draw, timed)
                    losses.append(error)
                    report["topology_scores"][method] = {
                        "forecast_paths": len(draw), "crps": error.mean(axis=0).tolist(),
                        "trace_mean_mae": np.abs(draw.mean(axis=0)-timed.mean(axis=0)).mean(axis=0).tolist(),
                        "reached_80": int(np.isfinite(prediction).sum()),
                        "largest_mass_at_80": float(np.nanmean(prediction))}
                report["crps_distance_minus_occupancy"] = mean_interval(losses[0]-losses[1])
                output["result"][kind]["groups"][name] = report
    write_json(run/"SCORED.json", output)
    print(run/"SCORED.json", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("register", "develop", "forecast", "reference", "score", "all"))
    parser.add_argument("run", type=Path, nargs="?")
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    run = register() if args.action in ("register", "all") else args.run
    if args.action == "register":
        return
    plan = read_json(run/"REGISTERED.json")
    if any(checksum(name) != expected for name, expected in plan["sources"].items()):
        raise ValueError("Registered source mismatch.")
    if args.action == "score":
        score(run, plan)
        return
    operations = {"develop": develop, "forecast": forecast, "reference": reveal}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for stage in (operations if args.action == "all" else (args.action,)):
            if stage == "develop":
                write_json(run/"EXACT_CONTROL.json", control(plan))
            operations[stage](run, pool, plan)
    if args.action == "all":
        score(run, plan)


if __name__ == "__main__":
    main()

