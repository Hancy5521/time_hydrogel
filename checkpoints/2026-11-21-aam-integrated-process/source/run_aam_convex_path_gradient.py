"""Frozen conditional-gradient experiment with two new original-camera splits."""

from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from shutil import copyfile

import numpy as np
from threadpoolctl import threadpool_limits

from aam_convex_path_gradient import fit, predict
from aam_formation_cycle_driver import np_load, np_save, read, save_tree, sha, write
from aam_formation_path_cycle import event, load_tree, stamp
from package_aam_formation_local_cycle import source_closure
from run_aam_direct_energy_trajectory_law import atlas_receipts
from run_aam_original_tape_pgas import ATLAS, paired
from run_aam_stationary_distillation import check, losses, reveal
from run_aam_suffix_group_smoother import acquired_data, counts
from run_aam_temporal_energy_law import predict_variant, readout_inputs


ROOT = Path(__file__).resolve().parent
PROTOCOL = ROOT / "AAM_CONVEX_PATH_GRADIENT_PROTOCOL_ZH.md"
AUDIT = ROOT / "audit_aam_convex_path_gradient.py"
NS = "aam-physical-teacher-convex-energy-gradient-v1"
UNITS = {"dev": list(range(2026144000, 2026144128)), "held": list(range(2026144128, 2026144512))}
METRICS = ("early18", "hidden_delta32_6")
GRID = (0., .3, 1., 3.)
ARMS = ("physical_teacher", "single_truth", "constant_teacher")
DIRECT = {"archived_direct_single": ("single_truth", 64), "archived_direct_teacher": ("physical_teacher", 24)}
REFS = (*ATLAS, *DIRECT)
RIDGE = .20


def register(out, atlas, teacher, gradient, review, workers):
    ancestor, atlas_units, atlas_model, inputs = atlas_receipts(atlas)
    tr_reg, tr_fit, tr_frozen, tr_score, tr_audit = (
        teacher / name for name in ("REGISTERED.json", "FIT_FROZEN.json", "TRAIN_TEACHER_FROZEN.json",
                                    "SCORED.json", "INDEPENDENT_AUDIT.json"))
    tr_plan, tr_audit_receipt, tr_split = read(tr_reg), read(tr_audit), read(tr_frozen)
    frozen = read(tr_fit)
    teaching_cross, direct_models = check(tr_split["arrays"]), check(frozen["model"])
    train_units = tr_plan["units"]["train"]
    parent_units = set(atlas_units) | set(sum(tr_plan["units"].values(), [])) | set(sum(ancestor["units"].values(), []))
    if (tr_audit_receipt["passed"] is not True or tr_audit_receipt["scored_sha256"] != sha(tr_score)
            or tr_plan["physics"] != ancestor["physics"] or Path(tr_plan["atlas_model"]) != Path(atlas_model)
            or len(train_units) != 256 or parent_units.intersection(sum(UNITS.values(), []))
            or frozen["teacher_freeze_sha256"] != sha(tr_frozen)
            or not read(review).get("review_received")):
        raise ValueError("Expected an independent parent audit and disjoint original simulated mothers.")
    sources = source_closure([Path(__file__), AUDIT])
    files = (*inputs, tr_reg, tr_fit, tr_frozen, tr_score, tr_audit, teaching_cross, direct_models, gradient,
             review, PROTOCOL)
    out.mkdir(exist_ok=False)
    snap = out / "SOURCE_SNAPSHOT"
    snap.mkdir()
    for name, path in sources.items():
        copyfile(path, snap / name)
    plan = {
        "namespace": NS, "physics": ancestor["physics"], "units": UNITS, "workers": workers,
        "grid": GRID, "metrics": METRICS, "names": (*REFS, *ARMS), "rank": 16, "krr_ridge": RIDGE,
        "mass_floor_relative_to_kernel": .10, "atlas_model": str(atlas_model), "atlas_units": atlas_units,
        "atlas_settings": read(atlas / "TRAINED_AND_SELECTED.json")["selected"], "direct_epochs": DIRECT,
        "train_units": train_units, "teacher_cross": str(teaching_cross), "direct_models": str(direct_models),
        "fit_train_only": True, "prior_dev_and_held_labels_read": False,
        "dev_gate": "physical index > 0, both means strictly below RBF, and mean strictly below both matched controls",
        "held_gate": "paired two-sided 95% lower > 0 against every comparator on both original metrics",
        "new_teacher_native_calls": 0, "native_prediction_calls": 0, "new_wet_measurements": 0,
        "sources": {str(p): sha(p) for p in sources.values()}, "inputs": {str(p): sha(p) for p in files},
    }
    reg = out / "REGISTERED.json"
    write(reg, {**stamp(), **plan})
    write(out / "RUN.json", {**stamp(), "registered_sha256": sha(reg), "before_native_or_fit": True})
    event(out, "registered", dev=len(UNITS["dev"]), held=len(UNITS["held"]), frozen_teacher_mothers=len(train_units))


def lock(out):
    receipt = out / "REGISTERED.json"
    plan = read(receipt)
    if (plan["namespace"] != NS or plan["units"] != UNITS or
            sha(receipt) != read(out / "RUN.json")["registered_sha256"]):
        raise ValueError("Require original prospective register.")
    for location, digest in {**plan["sources"], **plan["inputs"]}.items():
        if sha(location) != digest:
            raise ValueError("Frozen dependency changed: " + location)
    return plan


def frozen_baselines(plan, bank, original, y):
    from aam_stationary_distillation_net import predict as gru_predict

    info = readout_inputs(bank, y)
    raw = {n: predict_variant(n, plan["atlas_settings"].get(n), bank, info)["weights"] for n in ATLAS}
    for key, (arm, epoch) in DIRECT.items():
        raw[key] = gru_predict(original[arm], original["geometry"], y, raw["kernel"], epoch=epoch, shrink=1.)
    return raw


def fit_models(out, plan):
    cross = np_load(plan["teacher_cross"])
    if cross["unit"].tolist() != plan["train_units"] or cross["teacher_cross"].shape != (256, 3, 512):
        raise ValueError("Use only the full 256 registered teaching mothers.")
    start = out / "FIT_STARTED.json"
    write(start, {**stamp(), "teacher_cross_sha256": sha(plan["teacher_cross"]), "fit_roles": ["train"],
                  "dev_and_held_labels_opened": False})
    old = load_tree(plan["direct_models"])
    geom = old["geometry"]
    bank = load_tree(plan["atlas_model"])
    y = cross["measured"]
    with threadpool_limits(1):
        info = readout_inputs(bank, y)
        base = predict_variant("kernel", plan["atlas_settings"]["kernel"], bank, info)["weights"]
        models = {arm: fit(y, base, cross[key][:, :2].mean(axis=1), geom["distance"], geom["codes"],
                           bank["camera_scale"], ridge=RIDGE)
                  for arm, key in (("physical_teacher", "teacher_cross"), ("single_truth", "truth_cross"))}
    model = save_tree(out / "TRAINED_MODELS.npz", models)
    write(out / "FIT_FROZEN.json", {**stamp(), "model": model, "start_sha256": sha(start), "fit_roles": ["train"],
                                   "dev_and_held_labels_opened": False})
    event(out, "train_only_fit_frozen", model_sha256=model["sha256"])


def acquire_one(job):
    dest, cfg, role, unit = job
    with threadpool_limits(1):
        measured, raw = acquired_data(cfg, unit)
    if raw["calls"] != counts(3) or measured.shape != (3, 4) or len(raw["packets"]) != 4:
        save_tree(dest / f"{unit}_FAILED.npz", raw)
        raise ValueError("Require original four native packet states and three snapshots.")
    obs = np_save(dest / f"{unit}_OBS.npz", measured=measured)
    raw_ref = save_tree(dest / f"{unit}_RAW.npz", raw)
    return {"unit": unit, "role": role, "obs": obs, "raw": raw_ref, "native_calls": 3}


def acquire(out, plan, role):
    if not (out / "FIT_FROZEN.json").is_file():
        raise ValueError("Training must be frozen before new original acquisitions.")
    if role == "held" and read(out / "DEV_SELECTED.json")["go_to_fresh_held"] is not True:
        raise ValueError("DEV did not admit independent confirmation.")
    dest = out / role
    dest.mkdir(exist_ok=False)
    acquire_receipt = out / f"{role.upper()}_ACQUISITION_STARTED.json"
    write(acquire_receipt, {**stamp(), "role": role, "fit_sha256": sha(out / "FIT_FROZEN.json"),
                            "dependent_prediction_stage": False})
    jobs = [(dest, plan["physics"], role, unit) for unit in UNITS[role]]
    with ProcessPoolExecutor(max_workers=plan["workers"]) as pool:
        rows = list(pool.map(acquire_one, jobs))
    write(out / f"{role.upper()}_ACQUIRED.json", {**stamp(), "role": role, "rows": rows, "native_calls": 3 * len(rows),
                                                "started_sha256": sha(acquire_receipt),
                                                "registration_sha256": sha(out / "REGISTERED.json")})
    event(out, "original_observations_and_truth_separated", role=role, mothers=len(rows), native_calls=3 * len(rows))


def loaded(out, plan, role):
    path = out / f"{role.upper()}_ACQUIRED.json"
    archive = read(path)
    if archive["role"] != role or [r["unit"] for r in archive["rows"]] != UNITS[role]:
        raise ValueError("Original mother assignment changed.")
    y = np.stack([np_load(check(r["obs"]))["measured"] for r in archive["rows"]])
    bank, old = load_tree(plan["atlas_model"]), load_tree(plan["direct_models"])
    trained = load_tree(check(read(out / "FIT_FROZEN.json")["model"]))
    with threadpool_limits(1):
        raw = frozen_baselines(plan, bank, old, y)
    return path, archive, y, bank["bank"]["local"], old["geometry"], trained, raw


def correction(arm, multiplier, trained, y, base, geom):
    if not multiplier:
        return base, {"identity": True}
    key = "physical_teacher" if arm == "constant_teacher" else arm
    mode = "constant" if arm == "constant_teacher" else "kernel"
    result = predict(trained[key], y, base, geom["distance"], geom["codes"], mode=mode, hessian_ridge=multiplier)
    return result["weights"], {n: x for n, x in result.items() if n != "weights"}


def dev(out, plan):
    f, a, y, atoms, geom, trained, w = loaded(out, plan, "dev")
    dev_grid, opt = {}, {}
    with threadpool_limits(1):
        for name in ARMS:
            rows = [correction(name, reg, trained, y, w["kernel"], geom) for reg in GRID]
            dev_grid[name] = np.stack([x[0] for x in rows])
            opt[name] = [x[1] for x in rows]
    archive = save_tree(out / "DEV_FORECASTS.npz", {"unit": UNITS["dev"], "weights": w, "grid": dev_grid,
                                                    "optimizer": opt})
    frozen = out / "DEV_PREDICTIONS_FROZEN.json"
    write(frozen, {**stamp(), "archives": archive, "acquired_sha256": sha(f), "dev_labels_opened": False,
                   "fit_sha256": sha(out / "FIT_FROZEN.json")})
    hidden = reveal(a)
    with threadpool_limits(1):
        points = {m: np.stack([losses(atoms, hidden, row) for row in v]) for m, v in dev_grid.items()}
        chosen = {m: int(np.argmin(p.mean(axis=(1, 2)))) for m, p in points.items()}
        refs = {m: losses(atoms, hidden, p) for m, p in w.items()}
    means = {m: v[chosen[m]].mean(axis=0) for m, v in points.items()}
    reference = {m: v.mean(axis=0) for m, v in refs.items()}
    teacher = means["physical_teacher"]
    proceed = bool(chosen["physical_teacher"] > 0 and np.all(teacher < reference["kernel"])
                   and all(teacher.mean() < means[m].mean() for m in ("single_truth", "constant_teacher")))
    cards = save_tree(out / "DEV_SCORE_ARRAYS.npz", {"unit": UNITS["dev"], "grid_losses": points,
                                                    "baseline_losses": refs})
    selected = out / "DEV_SELECTED.json"
    write(selected, {**stamp(), "forecast_sha256": sha(frozen), "held_not_opened": True,
                     "dev_grid": GRID, "chosen_index": chosen, "chosen": {m: GRID[i] for m, i in chosen.items()},
                     "go_to_fresh_held": proceed, "means": means, "ref_means": reference, "arrays": cards})
    event(out, "fresh_development_evaluated", confirmed_admission=proceed, means=means, ref_means=reference)


def held(out, plan):
    selected = out / "DEV_SELECTED.json"
    d = read(selected)
    if d["go_to_fresh_held"] is not True:
        raise ValueError("DEV did not admit independent confirmation.")
    f, a, y, atoms, geom, models, w = loaded(out, plan, "held")
    with threadpool_limits(1):
        opt = {}
        for name in ARMS:
            w[name], opt[name] = correction(name, d["chosen"][name], models, y, w["kernel"], geom)
    ref = np_save(out / "HELD_WEIGHTS.npz", query=y, **w)
    opt_ref = save_tree(out / "HELD_OPTIMIZER.npz", opt)
    frozen = out / "PREDICTIONS_FROZEN.json"
    write(frozen, {**stamp(), "weight_ref": ref, "optimizer": opt_ref, "acquired_sha256": sha(f),
                   "dev_selection_sha256": sha(selected), "held_units": UNITS["held"], "held_truth_opened": False,
                   "native_prediction": 0})
    hidden = reveal(a)
    with threadpool_limits(1):
        all_scores = {m: losses(atoms, hidden, p) for m, p in w.items()}
    comparisons = {m: [paired(col) for col in (a - all_scores["physical_teacher"]).T]
                   for m, a in all_scores.items() if m != "physical_teacher"}
    score_array = save_tree(out / "HELD_SCORES.npz", {"unit": UNITS["held"], "losses": all_scores})
    go = bool(all(row[1] > 0. for group in comparisons.values() for row in group))
    write(out / "SCORED.json", {**stamp(), "held_forecast_sha256": sha(frozen), "dev_selected_sha256": sha(selected),
                               "held_opened": True, "held_mothers": len(UNITS["held"]), "metrics": METRICS,
                               "means": {m: a.mean(axis=0) for m, a in all_scores.items()},
                               "paired_gains": comparisons, "go_to_original_inverse": go, "arrays": score_array})
    event(out, "blind_confirmation_scored", original_inverse_admitted=go)


def stop(out, plan):
    selected = out / "DEV_SELECTED.json"
    d = read(selected)
    if d["go_to_fresh_held"] or (out / "held").exists():
        raise ValueError("Do not use a development closure when independent confirmation was admitted or started.")
    write(out / "SCORED.json", {**stamp(), "held_opened": False, "held_mothers": 0, "metrics": METRICS,
                               "dev_selected_sha256": sha(selected), "go_to_original_inverse": False,
                               "conclusion": "Prospective DEV gate was not met. No confirmation was acquired.",
                               "dev_result": d})
    event(out, "development_branch_closed", held_consumed=0)


if __name__ == "__main__":
    cli = ArgumentParser(description=__doc__)
    sub = cli.add_subparsers(dest="action", required=True)
    first = sub.add_parser("register")
    for arg in ("out", "atlas", "teacher", "gradient", "review"):
        first.add_argument(arg, type=Path)
    first.add_argument("--workers", type=int, default=8)
    for action in ("fit", "acquire-dev", "dev", "acquire-held", "held", "stop"):
        sub.add_parser(action).add_argument("out", type=Path)
    args = cli.parse_args()
    if args.action == "register":
        register(args.out.resolve(), args.atlas, args.teacher, args.gradient, args.review, args.workers)
    else:
        out = args.out.resolve(strict=True)
        plan = lock(out)
        if args.action.startswith("acquire-"):
            acquire(out, plan, args.action.split("-")[1])
        else:
            {"fit": fit_models, "dev": dev, "held": held, "stop": stop}[args.action](out, plan)
