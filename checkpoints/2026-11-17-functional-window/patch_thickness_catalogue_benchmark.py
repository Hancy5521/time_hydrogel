"""Preregister a thickness lever under fixed areal dry ion inventory and same early packet."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timedelta, timezone
from itertools import product, repeat
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.special import softmax

from gel_reactive_port_timing import read_json, sha256
from gel_reactive_two_read_assay import write_new
from patch_exchange_forward import exchange_bank
from patch_reservoir_joint_bank import joint_bank, prepared_fields
from window_risk_cases import groups, packet
from window_risk_forecasts import choose, elapsed, margins, prior
from window_spatial_probe import (
    changed_mechanism, conditional_release, observed_profiles, plausible_profiles,
    water_signals,
)


ROOT = Path(__file__).resolve().parent
BASE = Path("runs/patch_window_spatial_probe_20261116T142236751931Z/REGISTERED.json")
GRID_REF = Path("runs/patch_nested_catalogue_20261117T020738570901Z/REGISTERED.json")
READOUTS = ("effective_conductivity", "fixed_area_conductance")


def register():
    plan = read_json(ROOT / BASE)
    grid = read_json(ROOT / GRID_REF)["catalogue_extension"]
    requested = {plan["task_path"], str(BASE), str(GRID_REF), Path(__file__).name, *plan["hashes"]}
    hashes = {name: sha256(ROOT / name) for name in sorted(requested)}
    now = datetime.now(timezone.utc)
    run = ROOT / "runs" / now.strftime("patch_thickness_catalogue_%Y%m%dT%H%M%S%fZ")
    run.mkdir()
    plan.update(
        scientific_question="Does changing diffusion distance at fixed areal ion inventory expand "
                            "whole-window action feasibility and realize a meaningful gain under "
                            "shifted process dynamics?",
        registered_utc=now.isoformat(),
        registered_shanghai=now.astimezone(timezone(timedelta(hours=8))).isoformat(),
        hashes=hashes,
        sizes={"test": 32, "shift_test": 32},
        seeds={"test": 9514123, "shift_test": 9514139},
        thickness={
            "sources": [str(BASE), str(GRID_REF)],
            "physical_context": "A preformed plane patch of fixed in-plane area. Vary thickness "
                                "at fixed protected mass of ions per area by adjusting volumetric "
                                "dose inversely, and retain the original swelling, elasticity and "
                                "ionic equations. The alternative thickness settings are assumed "
                                "manufacturable at fixed material chemistry for this synthetic test.",
            "actions_at_scale": [
                {"scale": 1., "actions": grid["new_actions"]},
                {"scale": .85, "actions": grid["old_actions"]},
                {"scale": 1.15, "actions": grid["old_actions"]},
            ],
            "catalogues": {
                "original9": grid["embedded_indices"],
                "grid25": list(range(25)),
                "thickness27": [*grid["embedded_indices"], *range(25, 43)],
            },
            "prediction": "Original single seven-read depth packet and fixed nominal witness for "
                          "every action. Posterior on common uncertain kinetics, nominal water "
                          "coefficient and preparation perturbations computed once. Original "
                          "candidate likelihood and 256 latent atoms per action/bath. No extra "
                          "reads or environment-conditioned training. Enumerate all actions "
                          "with four original time requests, no new time or response targets.",
            "primary": {
                "split": "shift_test",
                "readout": "effective_conductivity",
                "comparison": "thickness27 versus grid25",
                "criterion": "Capacity and realized success are evaluated separately. A "
                             "practical capacity or success gain requires an absolute change "
                             "of at least .025 with a paired mother bootstrap 95% interval "
                             "above zero. Adoption of the process policy also requires "
                             "nonincreasing average common-catalogue positive regret.",
            },
            "sensitivity": "For constant area and through-thickness conduction, compare specimen "
                           "conductance scaled to the nominal geometry, G*L0/A = sigma / scale, "
                           "using the same trajectories and original thresholds. Average "
                           "layer moduli remain unchanged. This is a secondary electrical "
                           "readout, separate from contact performance or patch force.",
            "uncertainty": {"samples": 2000, "seed": 463157},
            "limits": "No composition-cost matching beyond same areal bound ion mass, no "
                      "unmeasured plate support or wet-contact model, and no empirical "
                      "calibration of the thickness multipliers.",
        },
    )
    write_new(run / "REGISTERED.json", plan)
    print(run, flush=True)


def geometry(parent, task, variant, water_scale):
    scale = variant["scale"]
    known = {
        **parent, "water_D": parent["water_D"] * water_scale,
        "thickness": parent["thickness"] * scale,
    }
    cfg = {**task["mechanism"], "dose": task["mechanism"]["dose"] / scale}
    return known, cfg


def prepared_areal_error(group, task, plan, variants, modes, cells):
    expected = (task["mechanism"]["dose"] * group["known"]["thickness"]
                * -np.expm1(-task["mechanism"]["process_time"] * task["mechanism"]["process_k"]))
    error = 0.
    for entry in variants:
        known, cfg = geometry(group["known"], task, entry, 1.)
        bound = prepared_fields(known, entry["actions"], cfg, modes, cells)
        areal = bound.mean(axis=-1) * known["thickness"]
        error = max(error, float(np.max(abs(areal - expected))))
    if error > 1e-10:
        raise ArithmeticError("Candidate thickness changed the prepared mass per nominal area.")
    return error


def wet_reads(group, task, plan):
    s = plan["spatial_probe"]
    obs, dry, features = packet(group, plan, task)
    seed_name = "shift_test" if group["shift"] else "test"
    noise = np.random.default_rng(
        s["snapshot_noise_seed"] + group["i"] + plan["seeds"][seed_name]
    ).normal(size=(2, 2))
    outputs = water_signals(observed_profiles(group, task, changed_mechanism(group, plan)), s)
    depth = outputs[1] + s["sigma"] * noise
    return obs, dry, depth, features


def prediction_weights(group, task, plan, atom, obs, dry, depth):
    s = plan["spatial_probe"]
    cells, step = plan["prediction_mesh"]
    time = np.array(s["times"])
    ticks = np.rint(time / step).astype(int)
    action = task["actions"][plan["fixed_action"]]
    reference_stock = prepared_fields(group["known"], [action], task["mechanism"], atom[:, 4:], cells)
    reference_stock = reference_stock[:, 0]
    half = reference_stock[:, :cells // 2].sum(axis=-1) / reference_stock.sum(axis=-1)
    _, early_lik = conditional_release(obs, atom, plan)
    with np.errstate(divide="ignore"):
        baseline = np.log(early_lik) - .5 * ((dry[:, None] - half) / plan["probes"]["sigma_half"])**2
    weights = []
    for s_d in s["water_grid"]:
        known = {**group["known"], "water_D": group["known"]["water_D"] * s_d}
        ref = water_signals(plausible_profiles(known, action, task["mechanism"], ticks, cells, step), s)
        distance = -.5 * (((depth - ref[1]) / s["sigma"]) ** 2).sum(axis=-1)
        weights.append(baseline + distance[:, None])
    return softmax(np.concatenate(weights, axis=1), axis=1)


def operational_margins(wet, weak, times, task, scale):
    original = margins(wet, weak, times, task)
    if scale == 1.:
        return original, original
    for field in (wet, weak):
        field[..., 3] /= scale
    return original, margins(wet, weak, times, task)


def freeze(group, task, plan):
    s = plan["spatial_probe"]
    spec = plan["thickness"]
    atoms = prior(group["known"], {**plan, "continuum_count": s["atoms_per_water"]}, True)
    obs, dry, depth, features = wet_reads(group, task, plan)
    weights = prediction_weights(group, task, plan, atoms, obs, dry, depth)
    cells, dt = plan["prediction_mesh"]
    t = elapsed(task, plan)
    area_error = prepared_areal_error(group, task, plan, spec["actions_at_scale"], atoms[:, 4:], cells)
    records = [[], []]
    for water_scale in s["water_grid"]:
        variants = [[], []]
        for variant in spec["actions_at_scale"]:
            known, cfg = geometry(group["known"], task, variant, water_scale)
            wet, weak = (joint_bank(
                known, variant["actions"], cfg, a, atoms[:, :4], atoms[:, 4:], cells, dt, t
            ) for a in (task["bath"]["wet"], task["bath"]["weak"]))
            expected = operational_margins(wet, weak, t, task, variant["scale"])
            for collected, block in zip(variants, expected, strict=True):
                collected.append(block)
        for collection, by_variant in zip(records, variants, strict=True):
            collection.append(np.concatenate(by_variant, axis=-1))
    outputs = {}
    for key, values in zip(READOUTS, records, strict=True):
        raw = np.concatenate(values, axis=0)
        signal = {}
        for name, id_list in spec["catalogues"].items():
            action, probability = choose(raw[..., id_list], weights, plan["risk_weight"])
            signal[name] = {"actions_full_index": np.array(id_list)[action].tolist(),
                            "probability": probability.tolist()}
        outputs[key] = signal
    return {
        "parent": group["parent"], "features_at_original_witness": features.tolist(),
        "observed_depth": depth.tolist(), "prepared_areal_error": area_error, "forecasts": outputs,
    }


def reveal(group, task, plan):
    truth_plan = changed_mechanism(group, plan)
    variants = plan["thickness"]["actions_at_scale"]
    cells, dt = plan["truth_mesh"]
    t = elapsed(task, plan)
    truth_cells = [[], []]
    closure = 0.
    for variant in variants:
        known, cfg = geometry(group["known"], task, variant, 1.)
        args = (known, variant["actions"], cfg)
        tail = (group["kinetic"], group["stock"], cells, dt, t)
        kwargs = {"outcome_cfg": truth_plan["exchange"]} if group["shift"] else {}
        bank = exchange_bank if group["shift"] else joint_bank
        wet, stock = bank(*args, task["bath"]["wet"], *tail, inventory=True, **kwargs)
        weak = bank(*args, task["bath"]["weak"], *tail, **kwargs)
        closure = max(closure, float(np.max(np.abs(stock.sum(axis=-1) - 1))))
        values = operational_margins(wet, weak, t, task, variant["scale"])
        for result, scalar in zip(truth_cells, values, strict=True):
            result.append(scalar)
    return {
        "parent": group["parent"], "inventory_closure_max": closure,
        "prepared_areal_error": prepared_areal_error(group, task, plan, variants, group["stock"], cells),
        "margins": {name: np.concatenate(each, axis=-1).tolist()
                    for name, each in zip(READOUTS, truth_cells, strict=True)},
    }


def score(outputs, measured, plan):
    if outputs["parent"] != measured["parent"]:
        raise ValueError("Parent identity differs across forecast and outcome.")
    values = {}
    for readout, y in measured["margins"].items():
        truth = np.asarray(y)
        full_oracle = np.maximum(truth.min(axis=-1), 0.)
        values[readout] = {}
        for c, pred in outputs["forecasts"][readout].items():
            ids = plan["thickness"]["catalogues"][c]
            chosen = np.asarray(pred["actions_full_index"])
            if not np.isin(chosen, ids).all():
                raise ValueError("Selected action was not in the registered catalogue.")
            selected = np.take_along_axis(truth, chosen[..., None], axis=-1)[..., 0]
            reached = (truth[..., ids] <= 0).any(axis=-1)
            success = selected <= 0
            item = {
                "feasible": reached,
                "success": success,
                "missed_feasible": reached & ~success,
                "unreachable": ~reached,
                "positive_regret": np.maximum(selected, 0.) - full_oracle,
                "new_thickness_selected": chosen >= 25,
                "all_action_brier": ((np.asarray(pred["probability"]) - (truth[..., ids] > 0)) ** 2).mean(axis=-1),
            }
            values[readout][c] = {name: v.tolist() for name, v in item.items()}
    return {**measured, "scores": values}


def summarize(records, plan):
    cfg = plan["thickness"]["uncertainty"]
    n = len(records)
    sample = np.random.default_rng(cfg["seed"]).integers(n, size=(cfg["samples"], n))
    out = {}
    for r in READOUTS:
        by_catalogue = records[0]["scores"][r]
        out[r] = {"means": {}, "by_request": {}, "comparisons": {}}
        a = {
            c: {s: np.asarray([record["scores"][r][c][s] for record in records])
                for s in methods}
            for c, methods in by_catalogue.items()
        }
        metrics = list(a["original9"])
        for catalogue, fields in a.items():
            out[r]["means"][catalogue] = {m: float(a[catalogue][m].mean()) for m in metrics}
            out[r]["by_request"][catalogue] = {
                m: fields[m].sum(axis=(0, 1)).tolist() for m in metrics if fields[m].dtype == bool
            }
        for policy, reference in (("thickness27", "grid25"), ("thickness27", "original9"), ("grid25", "original9")):
            differences = {}
            for metric in ("feasible", "success", "missed_feasible", "positive_regret"):
                diff = a[policy][metric].astype(float) - a[reference][metric]
                d = diff.mean(axis=(1, 2))
                differences[metric] = [float(d.mean()), *np.quantile(d[sample].mean(axis=1), [.025, .975]).tolist()]
            feasible = a[policy]["feasible"]
            support = a[reference]["feasible"]
            differences["new_feasible"] = int((feasible & ~support).sum())
            differences["lost_feasible"] = int((support & ~feasible).sum())
            gain = a[policy]["success"]
            former = a[reference]["success"]
            differences["gained_success"] = int((gain & ~former).sum())
            differences["lost_success"] = int((former & ~gain).sum())
            out[r]["comparisons"][f"{policy}_minus_{reference}"] = differences
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("register", "freeze", "score"))
    parser.add_argument("run", type=Path, nargs="?")
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    if args.phase == "register":
        if args.run is not None:
            parser.error("Registration chooses its path from actual UTC.")
        register()
        return
    if args.run is None:
        parser.error("Provide the registered run path.")
    run = args.run.resolve()
    manifest = run / "REGISTERED.json"
    plan = read_json(manifest)
    for name, expected in plan["hashes"].items():
        if sha256(ROOT / name) != expected:
            raise ValueError(f"Registered input or code changed: {name}")
    task = read_json(ROOT / plan["task_path"])
    base = plan["thickness"]["actions_at_scale"][0]["actions"]
    indices = plan["thickness"]["catalogues"]["original9"]
    if [base[i] for i in indices] != task["actions"]:
        raise ValueError("The original nine choices must be identical in both catalogue designs.")
    target = run / ("FROZEN.json" if args.phase == "freeze" else "SCORED.json")
    if target.exists():
        raise FileExistsError(target)
    out = {"registered_sha256": sha256(manifest), "started_utc": datetime.now(timezone.utc).isoformat(),
           "splits": {}}
    start = perf_counter()
    if args.phase == "score":
        frozen = run / "FROZEN.json"
        old = read_json(frozen)
        if old["registered_sha256"] != out["registered_sha256"]:
            raise ValueError("The stored prediction was not frozen on the registered protocol.")
        out.update(frozen_sha256=sha256(frozen), summary={})
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for name in plan["sizes"]:
            group = groups(name, plan, task)
            if args.phase == "freeze":
                rows = list(pool.map(freeze, group, repeat(task), repeat(plan)))
            else:
                answers = pool.map(reveal, group, repeat(task), repeat(plan))
                rows = [score(p, y, plan) for p, y in zip(old["splits"][name], answers, strict=True)]
                out["summary"][name] = summarize(rows, plan)
            out["splits"][name] = rows
            print(name, args.phase, len(rows), flush=True)
    out.update(completed_utc=datetime.now(timezone.utc).isoformat(), wall_seconds=perf_counter() - start)
    write_new(target, out)


if __name__ == "__main__":
    main()
