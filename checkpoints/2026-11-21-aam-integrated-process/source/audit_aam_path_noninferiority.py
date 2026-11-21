"""Independent numerical readback of a frozen ESS1024 trajectory comparison."""

from argparse import ArgumentParser
from pathlib import Path

import numpy as np
from scipy.optimize import brentq
from scipy.special import logsumexp
from scipy.stats import t
from threadpoolctl import threadpool_limits

from aam_formation_cycle_driver import np_load, read, sha, write
from aam_formation_path_cycle import load_tree, stamp
from audit_aam_endpoint_tape_move import artifact, close, generating_state, require, rng
from audit_aam_large_phase_atlas import exact_neighbor_score, hash_weights, independent_energy, native_weights
from audit_aam_stationary_distillation import baseline_laws, clock, lens


NAMESPACE = "aam-frozen-ess1024-path-noninferiority-fresh2304-v1"
UNITS = list(range(2026158000, 2026160304))
BIG_METHODS = ("ess1024", "phase_rbf", "raw_rbf")
OLD_METHODS = ("kernel", "kernel_cme", "knn", "empirical")
METHODS = (*BIG_METHODS, *OLD_METHODS)
METRICS = ("early18", "hidden_delta32_6")
ANCHORS = (0, 329, 657, 986, 1316, 1645, 1974, 2303)
MARGINS = (0., .005, .01, .02)
PRIMARY = .01
ALPHA = .05
PAIR_SEED = 817223451
PAIR_DRAWS = 32768


def source_check(out, plan):
    for filename, fingerprint in plan["sources"].items():
        snapshot = out / "SOURCE_SNAPSHOT" / Path(filename).name
        require(sha(snapshot) == fingerprint, f"Frozen source differs: {snapshot}")
    inputs = {Path(filename).resolve(strict=True): fp for filename, fp in plan["inputs"].items()}
    for path, fingerprint in inputs.items():
        require(sha(path) == fingerprint, f"Registered input differs: {path}")

    bank_file, old_file = (Path(plan[k]).resolve(strict=True) for k in ("bank_arrays", "old_model"))
    bank_root, old_root = bank_file.parent, old_file.parent
    bank_reg, old_reg = (read(root / "REGISTERED.json") for root in (bank_root, old_root))
    old_selection = read(old_root / "TRAINED_AND_SELECTED.json")
    bank_receipt = read(bank_root / "BANK_FROZEN.json")
    require(bank_file in inputs and old_file in inputs
            and artifact(bank_receipt["arrays"], bank_file) == bank_file
            and old_selection["selected"] == plan["old_settings"]
            and old_selection["train_ids"] == old_reg["units"]["train"] == plan["old_atlas_units"]
            and old_reg["physics"] == bank_reg["physics"] == plan["physics"],
            "The original same-information bank, physical law or tuning differs.")
    for root in (old_root, bank_root):
        audit, score = root / "INDEPENDENT_AUDIT.json", root / "SCORED.json"
        certificate = read(audit)
        require(audit in inputs and (root / "REGISTERED.json") in inputs
                and certificate["passed"] is True
                and certificate["scored_sha256"] == sha(score),
                f"Archive has no bound independent readback: {root}")

    for path in inputs:
        if path.name == "REGISTERED.json":
            units = read(path).get("units", [])
            groups = units.values() if isinstance(units, dict) else (units,)
            used = {int(unit) for group in groups for unit in group if isinstance(unit, int)}
            require(not used.intersection(UNITS), f"Previously registered mother used again: {path}")
    bank = np_load(bank_file)
    old = load_tree(old_file, inputs[old_file])
    require(bank["unit"].tolist() == bank_reg["units"]["bank"] == plan["bank_units"]
            and bank["local"].shape == (8192, 16, 3, 2, 3)
            and bank["clean"].shape == (8192, 16, 3, 4)
            and bank["measured"].shape == (8192, 3, 4)
            and old["bank"]["local"].shape == (512, 3, 2, 3),
            "Full legal support or original 512-mother comparison omitted.")
    return bank, old


def independent_ess_law(bank, old, observed):
    fit = (np.asarray(bank["clean"]) - observed) / old["noise"]
    log_point = -.5 * np.square(fit).sum(axis=(2, 3))
    log_mother = logsumexp(log_point, axis=1)
    log_mother -= log_mother.max()

    def residual(beta):
        mass = np.exp(beta * log_mother)
        return mass.sum() ** 2 / np.square(mass).sum() - 1024.

    temperature = 1. if residual(1.) >= 0. else brentq(residual, 0., 1., xtol=8.e-16)
    mass = np.exp(temperature * log_mother)
    mass /= mass.sum()
    conditional_phase = np.exp(log_point - logsumexp(log_point, axis=1, keepdims=True))
    weight = (mass[:, None] * conditional_phase).ravel()
    close(weight.sum(), 1.)
    require(1. / np.square(mass).sum() >= 1024. - 2.e-8, "Mother ESS misses the frozen floor.")
    return weight, temperature


def digest_valid(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def read_acquisition(out, plan, rows, descriptors):
    require([row["unit"] for row in rows] == UNITS == [row["unit"] for row in descriptors],
            "Independent mother identities or order changed.")
    raw_targets = {}
    paths = set()
    for i, (row, pred) in enumerate(zip(rows, descriptors, strict=True)):
        unit = row["unit"]
        require(row["role"] == "held" and row["native_calls"] == 3 and pred["obs_sha256"] == row["obs"]["sha256"]
                and set(pred["digests"]) == set(METHODS)
                and all(digest_valid(fp) for fp in pred["digests"].values()),
                f"Incomplete independent acquisition or blinded prediction: {unit}")
        files = {}
        for key in ("obs", "raw"):
            path = Path(row[key]["path"]).resolve(strict=True)
            require(path.is_relative_to(out) and path.name == f"{unit}_{key.upper()}.npz" and path not in paths,
                    "An acquisition was reused or stored outside the run.")
            paths.add(path)
            files[key] = artifact(row[key], path)
        observed = np_load(files["obs"])
        require(set(observed) == {"measured"} and observed["measured"].shape == (3, 4)
                and np.isfinite(observed["measured"]).all(), "Forecast has noncamera inputs.")
        if i in ANCHORS:
            query = observed["measured"]
            raw = load_tree(files["raw"])
            generating_state(plan["physics"], unit, query, raw)
            raw_targets[i] = query, lens(raw["packets"], plan["physics"], {})[1][0]
    return raw_targets


def audit_anchor(bank, old, plan, target, weights, digest, scored, unit):
    query, truth = target
    require(set(weights) == set(METHODS), "Anchor weights must bind every frozen method.")
    expect = native_weights(bank, query, old["noise"], old["camera_scale"], old["kernel_sq_scale"],
                            plan["old_settings"]["kernel"])
    expect["ess1024"], temperature = independent_ess_law(bank, old, query)
    expect.update({key: value[0] for key, value in
                   baseline_laws(old, {"atlas_settings": plan["old_settings"]}, query[None]).items()})
    pairs = [rng(PAIR_SEED, unit, k).random((PAIR_DRAWS, 2)) for k in range(2)]
    require(not np.array_equal(*pairs), "Monte Carlo pair streams are repeated.")
    for j, name in enumerate(METHODS):
        w = np.asarray(weights[name])
        require(w.shape == expect[name].shape and w.dtype == np.dtype("float64")
                and np.isfinite(w).all() and np.min(w) >= 0. and hash_weights(w) == digest[name],
                f"Frozen probability bytes do not match: {unit}, {name}.")
        close(w.sum(), 1.)
        close(w, expect[name], atol=2.e-10)
        if name in BIG_METHODS:
            for k, pair in enumerate(pairs):
                actual = independent_energy(bank["local"], w, truth[None], pair)[0]
                close(scored[k, j], actual)
        else:
            value = exact_neighbor_score(old["bank"]["local"], w, truth[None])
            close(scored[:, j], np.stack([value, value]))
    return {"ess_exponent": temperature, "all_seven_weight_digests_verified": True,
            "independent_replicate_scores_verified": True}


def independently_bound(delta, critical):
    mean = delta.sum(axis=0) / len(delta)
    sd = np.sqrt(np.square(delta - mean).sum(axis=0) / (len(delta) - 1))
    return mean, sd, mean + critical * sd / np.sqrt(len(delta))


def compare_statistics(losses, report):
    references = METHODS[1:]
    count = len(references) * len(METRICS)
    require(report["candidate"] == METHODS[0] and tuple(report["references"]) == references
            and tuple(report["metrics"]) == METRICS and report["mothers"] == len(UNITS)
            and report["family_comparisons"] == count and report["alpha"] == ALPHA,
            "Reported comparison family differs from the prospective 6×2 endpoints.")
    critical = t.ppf(1. - ALPHA / count, len(losses) - 1)
    pointwise = t.ppf(1. - ALPHA / 2., len(losses) - 1)
    close(report["critical_one_sided_bonferroni_t"], critical)
    cases = {row["margin"]: row for row in (report["primary"], *report["sensitivity"])}
    require(report["primary"]["margin"] == PRIMARY and set(cases) == set(MARGINS)
            and len(report["sensitivity"]) == len(MARGINS) - 1,
            "Registered primary relative tolerance or transparent sensitivity grid changed.")
    delta = losses[:, :1] - losses[:, 1:]
    column_mean, column_sd, upper = independently_bound(delta, critical)
    ratio_bounds = []
    for j, reference in enumerate(references):
        for k, metric in enumerate(METRICS):
            record = report["comparisons"][reference][metric]
            ref = losses[:, j + 1, k]
            target = losses[:, 0, k]
            mean, sd = column_mean[j, k], column_sd[j, k]
            half = pointwise * sd / np.sqrt(len(losses))
            for field, expected in (("candidate_mean", target.mean()), ("reference_mean", ref.mean()),
                                    ("raw_candidate_minus_reference", mean), ("raw_paired_sd", sd),
                                    ("raw_simultaneous_upper", upper[j, k]),
                                    ("raw_pointwise_two_sided_interval", [mean - half, mean + half])):
                close(record[field], expected)
            require(record["identical_scores"] is bool(np.array_equal(target, ref)), "Equality flag differs.")
            inversion = record["smallest_certifiable_relative_excess"]
            positive_denominator = independently_bound(-ref, critical)[2] < 0.
            require((inversion is not None) == positive_denominator, "Inverting relative tolerance requires a positive denominator.")
            if positive_denominator:
                close(independently_bound(delta[:, j, k] - inversion * ref, critical)[2], 0.)
            ratio_bounds.append(inversion)
    maximum = report["largest_endpoint_certifiable_relative_excess"]
    if None in ratio_bounds:
        require(maximum is None, "A missing endpoint inversion cannot certify an aggregate margin.")
    else:
        close(max(ratio_bounds), maximum)

    for margin, case in cases.items():
        shifted = delta - margin * losses[:, 1:]
        mu, sd, bound = independently_bound(shifted, critical)
        for j, reference in enumerate(references):
            for k, metric in enumerate(METRICS):
                actual = case["comparisons"][reference][metric]
                close(actual["shifted_mean"], mu[j, k])
                close(actual["shifted_paired_sd"], sd[j, k])
                close(actual["simultaneous_upper"], bound[j, k])
                require(actual["passed"] is bool(bound[j, k] < 0.), "Individual endpoint decision differs.")
        require(case["passed"] is bool(np.all(bound < 0.)), "Conjunctive decision differs.")
    return report["primary"]["passed"], critical


def audit(out):
    names = ("REGISTERED", "ACQUIRED", "FORECAST_DESCRIPTORS", "PREDICTIONS_FROZEN", "SCORED")
    paths = {name: out / f"{name}.json" for name in names}
    plan, acquisition, descriptors, frozen, scored = (read(paths[name]) for name in names)
    start = read(out / "RUN.json")
    acquisition_start = out / "ACQUISITION_STARTED.json"
    began = read(acquisition_start)
    timestamps = list(map(clock, [plan, start, began, acquisition, descriptors, frozen, scored]))
    registered = sha(paths["REGISTERED"])
    require(all(left <= right for left, right in zip(timestamps, timestamps[1:]))
            and timestamps[0] < timestamps[-1] and start["registered_sha256"] == registered == began["registered_sha256"]
            and acquisition["started_sha256"] == sha(acquisition_start)
            and start["before_native_or_fit"] is True and plan["namespace"] == NAMESPACE
            and plan["units"] == UNITS and tuple(plan["methods"]) == METHODS
            and tuple(plan["metrics"]) == METRICS and plan["primary_margin"] == PRIMARY
            and tuple(plan["margins"]) == MARGINS and tuple(plan["anchor_offsets"]) == ANCHORS
            and plan["pair_seed"] == PAIR_SEED and plan["pair_draws"] == PAIR_DRAWS
            and frozen["held_truth_opened"] is False and frozen["acquisition_sha256"] == sha(paths["ACQUIRED"])
            and frozen["descriptor_sha256"] == sha(paths["FORECAST_DESCRIPTORS"])
            and scored["freeze_sha256"] == sha(paths["PREDICTIONS_FROZEN"])
            and tuple(scored["methods"]) == METHODS and tuple(scored["metrics"]) == METRICS
            and scored["new_mothers"] == len(UNITS)
            and acquisition["native_calls"] == scored["native_calls_generation"] == 3 * len(UNITS)
            and plan["new_fits"] == scored["new_fits"] == plan["wet_measurements"] == scored["new_wet_measurements"] == 0
            and plan["new_native_forecast"] == frozen["forecast_native"] == 0,
            "Prospective membership, chronology, analysis family or frozen forecast differs.")
    bank, old = source_check(out, plan)
    targets = read_acquisition(out, plan, acquisition["rows"], descriptors["rows"])
    anchor_weights = load_tree(artifact(frozen["audit_weights"], out / "AUDIT_WEIGHTS.npz"))
    require(set(anchor_weights) == {str(UNITS[i]) for i in ANCHORS}, "Fixed anchor subset changed.")
    results = np_load(artifact(scored["arrays"], out / "SCORED_ARRAYS.npz"))
    losses, replicas = results["losses"], results["replica"]
    require(set(results) == {"unit", "losses", "replica"} and results["unit"].tolist() == UNITS
            and losses.shape == (len(UNITS), len(METHODS), len(METRICS))
            and replicas.shape == (len(UNITS), 2, len(METHODS), len(METRICS))
            and np.isfinite(losses).all() and np.isfinite(replicas).all(), "Incomplete independent mother-level scores.")
    close(losses, replicas.mean(axis=1))
    close(scored["replica_score_means"], replicas.mean(axis=0))
    np.testing.assert_array_equal(replicas[:, 0, len(BIG_METHODS):], replicas[:, 1, len(BIG_METHODS):])
    primary, critical = compare_statistics(losses, scored["noninferiority"])
    require(scored["passed"] is primary, "Final primary decision differs from the audit.")
    certificates = {}
    for i, target in targets.items():
        unit = UNITS[i]
        certificates[str(unit)] = audit_anchor(bank, old, plan, target, anchor_weights[str(unit)],
                                               descriptors["rows"][i]["digests"], replicas[i], unit)
    repeat_delta = replicas[:, :, :1] - (1. + PRIMARY) * replicas[:, :, 1:]
    repeat_upper = independently_bound(repeat_delta, critical)[2]
    return {**stamp(), "passed": True, "registration_sha256": registered,
            "scored_sha256": sha(paths["SCORED"]), "forecast_sha256": scored["freeze_sha256"],
            "independent_mothers": len(UNITS), "raw_anchor_indices": ANCHORS, "anchor_receipts": certificates,
            "new_native_calls": 0, "new_fits": 0, "new_wet_measurements": 0,
            "trajectory_primary_noninferiority": primary,
            "each_mc_stream_upper_at_primary": repeat_upper,
            "all_immutable_archive_and_source_hashes_verified": True,
            "all_mother_paired_statistics_independently_recomputed": True}


def fixtures():
    bank = {"clean": np.zeros((1024, 16, 3, 4))}
    weight, beta = independent_ess_law(bank, {"noise": np.ones(4)}, np.zeros((3, 4)))
    np.testing.assert_array_equal(weight, np.full(16384, 1. / 16384.))
    require(beta == 1., "Uniform mother floor should not temper a legal law.")
    shifted = np.array([[1., -1.], [4., 2.], [7., 5.]])
    mean, sd, upper = independently_bound(shifted, 2.)
    close(mean, [4., 2.])
    close(sd, [3., 3.])
    close(upper, [4. + np.sqrt(12.), 2. + np.sqrt(12.)])
    return {"passed": True, "new_native_calls": 0, "new_fits": 0}


if __name__ == "__main__":
    cli = ArgumentParser(description=__doc__)
    group = cli.add_mutually_exclusive_group(required=True)
    group.add_argument("--out", type=Path)
    group.add_argument("--fixtures", action="store_true")
    args = cli.parse_args()
    with threadpool_limits(limits=1):
        if args.fixtures:
            print(fixtures())
        else:
            directory = args.out.resolve(strict=True)
            write(directory / "INDEPENDENT_AUDIT.json", audit(directory))
            print("PASS independent trajectory noninferiority readback", directory)
