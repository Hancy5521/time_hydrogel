"""Frozen matched-support original AAm phase-bank comparison."""

from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor
from hashlib import sha256
from pathlib import Path
from shutil import copyfile

import numpy as np
from threadpoolctl import threadpool_limits

from aam_common_phase_hidden import history_orbit
from aam_formation_cycle_driver import np_load, np_save, read, save_tree, sha, write
from aam_formation_path_cycle import event, load_tree, stamp
from aam_large_phase_atlas import energy_scores, pair_uniforms, probabilities
from aam_phase_marginal_tape_pgas import phase_offsets
from aam_transverse_orbit import camera_orbit
from aam_camera_information import history_features
from package_aam_formation_local_cycle import source_closure
from run_aam_convex_path_gradient import acquire_one
from run_aam_direct_energy_trajectory_law import atlas_receipts
from run_aam_original_tape_pgas import ATLAS, paired
from run_aam_stationary_distillation import check, losses
from run_aam_suffix_group_smoother import acquired_data, counts
from run_aam_temporal_energy_law import predict_variant, readout_inputs


HERE = Path(__file__).resolve().parent
NS = "aam-large-shared-phase-original-bank-v1"
UNITS = {"bank": list(range(2026145000, 2026153192)), "held": list(range(2026154000, 2026154384))}
BLOCK = 32
PHASE_METHODS = ("phase_gaussian", "phase_rbf", "raw_gaussian", "raw_rbf", "phase_uniform")
NESTED = {"phase_gaussian_first512": 512, "phase_gaussian_first2048": 2048}
GATE_METHODS = (*PHASE_METHODS, *ATLAS)
METHODS = (*GATE_METHODS, *NESTED)
METRICS = ("early18", "hidden_delta32_6")
DRAWS = 32768
SEED = 592090181
ANCHORS = (0, 1, 54, 108, 162, 216, 270, 383)
PROTOCOL = HERE / "AAM_LARGE_PHASE_ATLAS_PROTOCOL_ZH.md"
AUDIT = HERE / "audit_aam_large_phase_atlas.py"
WORK = {}


def register(out, atlas, previous, review, workers):
    ancestor, train, model, old_inputs = atlas_receipts(atlas)
    references = [previous / n for n in ("REGISTERED.json", "SCORED.json", "INDEPENDENT_AUDIT.json")]
    prior, scores, audit = map(read, references)
    prior_ids = set(sum(ancestor["units"].values(), []) + sum(prior["units"].values(), []) + prior["train_units"])
    new = sum(UNITS.values(), [])
    if (len(set(new)) != len(new) or prior_ids.intersection(new) or prior["physics"] != ancestor["physics"]
            or audit["passed"] is not True or audit["scored_sha256"] != sha(references[1])
            or scores["go_to_original_inverse"] or not read(review).get("review_received")):
        raise ValueError("Require original audited generating law and disjoint mother addresses.")
    sources = source_closure([Path(__file__), AUDIT])
    out.mkdir(exist_ok=False)
    snap = out / "SOURCE_SNAPSHOT"
    snap.mkdir()
    for key, p in sources.items():
        copyfile(p, snap / key)
    reg = out / "REGISTERED.json"
    write(reg, {**stamp(), "namespace": NS, "physics": ancestor["physics"], "units": UNITS, "block": BLOCK,
                "phases": 16, "methods": METHODS, "gate_methods": GATE_METHODS, "nested_sizes": NESTED,
                "metrics": METRICS, "workers": workers,
                "pair_samples": DRAWS, "pair_seed": SEED, "audit_held_indices": ANCHORS, "atlas_model": str(model),
                "atlas_settings": read(atlas / "TRAINED_AND_SELECTED.json")["selected"], "atlas_units": train,
                "inputs": {str(f): sha(f) for f in (*old_inputs, *references, review, PROTOCOL)},
                "sources": {str(f): sha(f) for f in sources.values()}, "new_fits": 0, "new_wet_measurements": 0,
                "original_native_bank": len(UNITS["bank"]) * 3, "original_native_held": len(UNITS["held"]) * 3,
                "forecast_native": 0, "gate": "both metrics vs all controls t95 lower > 0 and both MC mean signs > 0"})
    write(out / "RUN.json", {**stamp(), "registered_sha256": sha(reg), "before_native_or_fit": True})
    event(out, "prospective_bank_comparison_registered", bank_mothers=len(UNITS["bank"]), held=len(UNITS["held"]))


def lock(out):
    reg = out / "REGISTERED.json"
    p = read(reg)
    if p["namespace"] != NS or p["units"] != UNITS or read(out / "RUN.json")["registered_sha256"] != sha(reg):
        raise ValueError("Falsely assigned original splits.")
    for f, h in {**p["sources"], **p["inputs"]}.items():
        if sha(f) != h:
            raise ValueError("Registered input changed: " + f)
    return p


def acquire_block(job):
    folder, cfg, units = job
    rows, clean, measured, inputs = [], [], [], []
    with threadpool_limits(1):
        for unit in units:
            y, raw = acquired_data(cfg, unit)
            if raw["calls"] != counts(3) or len(raw["packets"]) != 4:
                raise ValueError("Incomplete original native trajectory.")
            offsets = phase_offsets(raw["packets"][0])
            camera = np.stack([camera_orbit(a, offsets) for a in raw["packets"][1:]], axis=1)
            hide = history_orbit(raw["packets"], offsets)
            np.testing.assert_allclose(raw["clean"], camera[0], atol=2.e-13, rtol=0.)
            np.testing.assert_allclose(history_features(raw["packets"]), hide[0], atol=2.e-13, rtol=0.)
            rows.append({"unit": unit, "measured": y, "raw": raw})
            clean.append(camera)
            measured.append(y)
            inputs.append(hide)
    raw = save_tree(folder / f"{units[0]}_RAW.npz", {"rows": rows})
    arrays = np_save(folder / f"{units[0]}_ORBIT.npz", unit=units, clean=clean, measured=measured, local=inputs)
    return {"units": units, "raw": raw, "arrays": arrays, "native_calls": 3 * len(units)}


def bank_acquire(out, p):
    folder = out / "bank"
    folder.mkdir(exist_ok=False)
    started = out / "BANK_ACQUISITION_STARTED.json"
    write(started, {**stamp(), "registered_sha256": sha(out / "REGISTERED.json"), "role": "bank"})
    bank = UNITS["bank"]
    tasks = [(folder, p["physics"], bank[i:i+BLOCK]) for i in range(0, len(bank), BLOCK)]
    with ProcessPoolExecutor(max_workers=p["workers"]) as pool:
        rows = list(pool.map(acquire_block, tasks))
    receipt = out / "BANK_ACQUIRED.json"
    write(receipt, {**stamp(), "rows": rows, "role": "bank", "native_calls": 3 * len(bank),
                    "started_sha256": sha(started)})
    tables = [np_load(check(a["arrays"])) for a in rows]
    joined = {n: np.concatenate([t[n] for t in tables]) for n in tables[0]}
    if joined["unit"].tolist() != bank:
        raise ValueError("Fixed bank order changed.")
    ref = np_save(out / "FULL_BANK_ORBITS.npz", **joined)
    write(out / "BANK_FROZEN.json", {**stamp(), "acquired_sha256": sha(receipt), "arrays": ref, "bank_mothers": len(bank)})
    event(out, "whole_path_bank_frozen", bank_mothers=len(bank), native_calls=3 * len(bank))


def held_acquire(out, p):
    if not (out / "BANK_FROZEN.json").is_file():
        raise ValueError("Freeze complete bank before new confirmation acquisition.")
    dest = out / "held"
    dest.mkdir(exist_ok=False)
    start = out / "HELD_ACQUISITION_STARTED.json"
    write(start, {**stamp(), "bank_sha256": sha(out / "BANK_FROZEN.json"), "role": "held"})
    with ProcessPoolExecutor(max_workers=p["workers"]) as pool:
        rows = list(pool.map(acquire_one, [(dest, p["physics"], "held", i) for i in UNITS["held"]]))
    write(out / "HELD_ACQUIRED.json", {**stamp(), "rows": rows, "role": "held", "native_calls": 3 * len(rows),
                                      "started_sha256": sha(start)})
    event(out, "confirmation_camera_and_truth_saved_separately", mothers=len(rows), native_calls=3 * len(rows))


def prepare_worker(bank_path, atlas_path, settings):
    WORK.update(new=np_load(bank_path), old=load_tree(atlas_path), settings=settings)


def digest(p):
    return sha256(np.ascontiguousarray(p, dtype="<f8").tobytes()).hexdigest()


def forecast_stats(weights):
    by_phase = weights.reshape(-1, 16)
    parent = by_phase.sum(axis=1)
    return {"atom_ess": 1. / np.square(weights).sum(), "mother_ess": 1. / np.square(parent).sum(),
            "largest_atom_mass": weights.max(), "largest_mother_mass": parent.max(),
            "positive_atoms": np.count_nonzero(weights), "positive_mothers": np.count_nonzero(parent),
            "phase_marginal": by_phase.sum(axis=0)}


def predict_one(y):
    new, old, s = WORK["new"], WORK["old"], WORK["settings"]
    predicted = probabilities(new["clean"], new["measured"], y, old["noise"], old["camera_scale"],
                              old["kernel_sq_scale"], s["kernel"])
    info = readout_inputs(old, y[None])
    historical = {m: predict_variant(m, s.get(m), old, info)["weights"][0] for m in ATLAS}
    nested = {}
    full = predicted["weights"]["phase_gaussian"]
    for name, count in NESTED.items():
        cut = count * 16
        x = np.zeros_like(full)
        x[:cut] = full[:cut] / full[:cut].sum()
        nested[name] = x
    return {**predicted["weights"], **historical, **nested}, predicted["diagnostics"], {
        n: forecast_stats(x) for n, x in nested.items()}


def prediction_job(job):
    i, row = job
    y = np_load(check(row["obs"]))["measured"]
    with threadpool_limits(1):
        w, diagnostics, nested = predict_one(y)
    record = {"unit": row["unit"], "observation_sha256": row["obs"]["sha256"],
              "weights_digests": {m: digest(q) for m, q in w.items()}, "diagnostics": diagnostics,
              "nested_diagnostics": nested}
    return record, w if i in ANCHORS else None


def inputs(out, p):
    f = out / "HELD_ACQUIRED.json"
    a = read(f)
    if a["role"] != "held" or [r["unit"] for r in a["rows"]] != UNITS["held"]:
        raise ValueError("Require every new original confirmation mother.")
    bank_path = check(read(out / "BANK_FROZEN.json")["arrays"])
    init = (bank_path, p["atlas_model"], p["atlas_settings"])
    return f, a["rows"], init


def forecast(out, p):
    f, rows, init = inputs(out, p)
    with ProcessPoolExecutor(max_workers=p["workers"], initializer=prepare_worker, initargs=init) as pool:
        results = list(pool.map(prediction_job, enumerate(rows), chunksize=4))
    cards = out / "FORECAST_DESCRIPTORS.json"
    write(cards, {**stamp(), "rows": [a for a, _ in results], "ordered_method_names": METHODS})
    weights = save_tree(out / "AUDIT_WEIGHTS.npz", {str(a["unit"]): w for a, w in results if w is not None})
    write(out / "PREDICTIONS_FROZEN.json", {**stamp(), "descriptor_sha256": sha(cards), "audit_weights": weights,
              "held_acquired_sha256": sha(f), "bank_sha256": sha(out / "BANK_FROZEN.json"),
              "held_truth_opened": False, "deterministic_probability_reference": True, "forecast_native": 0})
    event(out, "all_reference_support_probabilities_frozen", held=384, prediction_native=0)


def scoring_job(job):
    row, frozen = job
    y = np_load(check(row["obs"]))["measured"]
    unit = row["unit"]
    with threadpool_limits(1):
        w, _, _ = predict_one(y)
        if {m: digest(q) for m, q in w.items()} != frozen["weights_digests"] or frozen["unit"] != unit:
            raise ValueError("Score only an exactly frozen normalized prediction.")
        raw = load_tree(check(row["raw"]))
        target = history_features(raw["packets"])
        u0, u1 = [pair_uniforms(np.random.default_rng(np.random.SeedSequence([SEED, unit, rep])), draws=DRAWS)
                  for rep in range(2)]
        result = energy_scores(WORK["new"]["local"], {m: w[m] for m in (*PHASE_METHODS, *NESTED)}, target, u0, u1)
        chosen = result["methods"]
        first = {m: np.stack((v["energy_main"][:2], v["energy_repeat"][:2])) for m, v in chosen.items()}
        se = {m: np.stack((v["energy_monte_carlo_se"][:2], v["energy_repeat_monte_carlo_se"][:2]))
              for m, v in chosen.items()}
        paired_mcse = {}
        for m in (*PHASE_METHODS[1:], *NESTED):
            d = result["contrasts"]["phase_gaussian_minus_" + m]
            paired_mcse[m] = np.stack((d["energy_monte_carlo_se"][:2], d["energy_repeat_monte_carlo_se"][:2]))
        for m in ATLAS:
            point = losses(WORK["old"]["bank"]["local"], target[None], w[m][None])[0]
            first[m] = np.stack((point, point))
            se[m] = np.zeros((2, 2))
            paired_mcse[m] = se["phase_gaussian"]
    return unit, first, se, paired_mcse


def score(out, p):
    f, rows, init = inputs(out, p)
    forecast_file = out / "PREDICTIONS_FROZEN.json"
    fr = read(forecast_file)
    cards = out / "FORECAST_DESCRIPTORS.json"
    if (fr["descriptor_sha256"] != sha(cards) or fr["held_acquired_sha256"] != sha(f)
            or fr["bank_sha256"] != sha(out / "BANK_FROZEN.json") or fr["held_truth_opened"]):
        raise ValueError("Independent outcomes must follow intact proof of freezing.")
    with ProcessPoolExecutor(max_workers=p["workers"], initializer=prepare_worker, initargs=init) as pool:
        answers = list(pool.map(scoring_job, zip(rows, read(cards)["rows"], strict=True), chunksize=4))
    units, by_row, std, delta_std = list(zip(*answers, strict=True))
    replicates = {m: np.stack([a[m] for a in by_row]) for m in METHODS}
    mcse = {m: np.stack([a[m] for a in std]) for m in METHODS}
    gain_mcse = {m: np.stack([a[m] for a in delta_std]) for m in METHODS if m != "phase_gaussian"}
    primary = {m: arr.mean(axis=1) for m, arr in replicates.items()}
    ref = "phase_gaussian"
    gains = {m: [paired(col) for col in (v - primary[ref]).T] for m, v in primary.items() if m != ref}
    mc_gains = {m: (v - replicates[ref]).mean(axis=0) for m, v in replicates.items() if m != ref}
    go = bool(all(row[1] > 0. for m in GATE_METHODS if m != ref for row in gains[m]) and
              all(np.all(mc_gains[m] > 0.) for m in GATE_METHODS if m != ref))
    archive = save_tree(out / "SCORE_ARRAYS.npz", {"unit": list(units), "losses": primary, "replica": replicates,
                                                   "mcse": mcse, "gain_mcse": gain_mcse})
    write(out / "SCORED.json", {**stamp(), "held_forecast_sha256": sha(forecast_file), "held_mothers": len(units),
              "metrics": METRICS, "means": {m: v.mean(axis=0) for m, v in primary.items()}, "paired_gains": gains,
              "mc_replicate_gains": mc_gains, "go_to_original_inverse": go, "arrays": archive,
              "aggregate_gain_mcse_by_replica": {m: np.sqrt(np.square(a).sum(axis=0)) / len(a)
                                                    for m, a in gain_mcse.items()},
              "bank_mothers": len(UNITS["bank"]), "native_calls_bank": len(UNITS["bank"])*3,
              "native_calls_held": len(UNITS["held"])*3, "new_measurements": 0})
    event(out, "new_blind_mothers_scored", original_inverse_admitted=go, means={m: s.mean(0) for m, s in primary.items()})


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    first = sub.add_parser("register")
    for k in ("out", "atlas", "previous", "review"):
        first.add_argument(k, type=Path)
    first.add_argument("--workers", type=int, default=8)
    actions = {"bank": bank_acquire, "held": held_acquire, "predict": forecast, "score": score}
    for k in actions:
        sub.add_parser(k).add_argument("out", type=Path)
    a = parser.parse_args()
    if a.action == "register":
        register(a.out.resolve(), a.atlas, a.previous, a.review, a.workers)
    else:
        run = a.out.resolve(strict=True)
        actions[a.action](run, lock(run))
