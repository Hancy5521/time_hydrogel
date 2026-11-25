"""Check frozen AAm results offline. This script reads arrays and runs no simulations."""
from __future__ import annotations

import argparse
from datetime import datetime
from hashlib import sha256
import json
from pathlib import Path

import numpy as np


MAIN_FREEZE = "07f8eee193484148dec344bcecf6aba3f8f0a1bb1479c4e4c320650134b7132d"
PAIRED_FREEZE = "c7cfa710c81482c9ed046ec4395d71816946c3edd0c45433b32c7da3471568a6"
ARMS = ("frozen_early", "static_early", "frozen_delayed", "static_delayed", "zero")
BASELINE = ("frozen", "static", "zero")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(file):
    from public_integrity import original_digest

    return original_digest(file)



def read(file):
    return json.loads(file.read_text(encoding="utf-8-sig"))


def bound_file(file, expected):
    require(digest(file) == expected, f"SHA256 mismatch for {file.name}")
    return file


def near(actual, registered, label):
    try:
        np.testing.assert_allclose(actual, registered, rtol=0.0, atol=1.0e-12)
    except AssertionError as exc:
        raise ValueError(f"Numerical check failed for {label}") from exc


def times_before(first, last):
    return datetime.fromisoformat(first["utc"]) < datetime.fromisoformat(last["utc"])


def readout(path):
    return np.stack((path[..., 2, 2], path[..., 3, 3], path[..., 5, 0], path[..., 5, 3]), axis=-1)


def threshold_box(anchors):
    measured = readout(anchors)
    l = np.quantile(measured[:, 0], 0.25)
    early = [[l, np.quantile(measured[:, 1], 0.65), np.quantile(measured[:, 2], 0.40), 0.0],
             [1.0, 1.0, 1.0, 1.0]]
    late = [[l, 0.0, 0.0, np.quantile(measured[:, 3], 0.40)],
            [1.0, np.quantile(measured[:, 1], 0.35), 1.0, 1.0]]
    return np.asarray([early, late])


def margin_and_hits(path, bounds):
    values = readout(path)
    low, high = np.asarray(bounds)
    active_lower = low > 0.0
    active_upper = high < 1.0
    margins = np.concatenate(((values - low)[..., active_lower],
                              (high - values)[..., active_upper]), axis=-1).min(axis=-1) / 0.15
    return margins, margins >= -1.0e-12


def rank_actions(planning, bounds):
    margins, hits = margin_and_hits(planning, bounds)
    probabilities = hits.mean(axis=-1)
    tie_margin = margins.mean(axis=-1)
    options = np.broadcast_to(np.arange(probabilities.shape[-1]), probabilities.shape)
    choice = np.lexsort((options, -tie_margin, -probabilities), axis=-1)[..., 0]
    return choice, probabilities


def seed_rows(*roles):
    seeds = np.concatenate([np.asarray(role).reshape(-1) for role in roles])
    require(len(seeds) == len(np.unique(seeds)), "An address repeats across checked seed roles")
    return seeds


def make_bootstrap(key, cards, draws):
    seed = int.from_bytes(sha256(key.encode("utf-8")).digest()[:16], "little")
    return np.random.default_rng(seed).integers(0, cards, (draws, cards))


def paired_interval(values, draw):
    return np.quantile(values[draw].mean(axis=-1), (0.025, 0.975))


def energy_score(prediction, truth):
    cols = np.array([0, 2, 3, 5, 9])
    scale = np.array([0.15, 0.10, 0.18, 0.18, 0.45])

    def flatten(group):
        return (np.take(group[..., 2:6, :], cols, axis=-1) / scale).reshape(len(group), 20)

    x, y = flatten(prediction), flatten(truth)
    cross = np.linalg.norm(x[:, None, :] - y[None, :, :], axis=-1).mean()
    self_distance = np.linalg.norm(x[:, None, :] - x[None, :, :], axis=-1).mean()
    return (cross - 0.5 * self_distance) / np.sqrt(20)


def audit_main(run):
    freeze = read(bound_file(run / "FROZEN.json", MAIN_FREEZE))
    reg = read(bound_file(run / "REGISTERED.json", freeze["registration_sha256"]))
    acquisition = read(bound_file(run / "ACQUISITION.json", freeze["acquisition_sha256"]))
    confirm = read(run / "CONFIRMED.json")
    pool = read(run / "POOLED_NOMINAL.json")
    score = read(run / "SCORED.json")
    audit_score = read(run / "INDEPENDENT_SCORE_AUDIT.json")
    audit_raw = read(run / "AUDIT_RAW.json")
    require(times_before(freeze, confirm) and times_before(pool, confirm), "Main freeze/confirmation order")
    require(score["freeze"] == pool["freeze"] == confirm["verified_prediction_sha256"] == MAIN_FREEZE,
            "Main freeze link")
    require(score["truth_sha256"] == confirm["sha256"] == audit_raw["confirmed_sha256"],
            "Main confirmation hash link")
    require(audit_score["frozen_sha256"] == MAIN_FREEZE and
            audit_score["verified_parent_score_sha256"] == digest(run / "SCORED.json"), "Main score audit link")
    bound_file(run / "score_loop.py", pool["source"])
    bound_file(run / "PREDICTED.npz", freeze["predicted_sha256"])
    bound_file(run / "CONFIRMED.npz", confirm["sha256"])
    for name in ("DEVELOPMENT.npz", "QUERY_LOCAL.npz", "ZERO_C_ANCHOR.npz"):
        bound_file(run / name, acquisition["inputs"][name])

    with np.load(run / "DEVELOPMENT.npz", allow_pickle=False) as dev, \
         np.load(run / "QUERY_LOCAL.npz", allow_pickle=False) as query, \
         np.load(run / "ZERO_C_ANCHOR.npz", allow_pickle=False) as anchor, \
         np.load(run / "PREDICTED.npz", allow_pickle=False) as prediction, \
         np.load(run / "CONFIRMED.npz", allow_pickle=False) as truth:
        roles = seed_rows(dev["seeds"], query["seeds"], anchor["seeds"], prediction["forward_seeds"],
                          prediction["planning_seeds"], truth["seeds"])
        require(len(roles) == 3112, "Main seed-role count")
        reference_targets = threshold_box(anchor["path"])
        near(reference_targets, freeze["targets"], "thresholds from anchors")
        planning, observed, forward = prediction["planning"], truth["path"], prediction["forward"]
        require(observed.shape == (36, 13, 16, 6, 10), "Main native trajectory shape")
        require(planning.shape == (4, 36, 13, 8, 6, 10), "Main planning shape")
        require(len(truth["journal_keys"]) == confirm["journal_trace_count"] == 229 and
                len(truth["journals"]) == audit_raw["event_count"] == 19866, "Main journal sizes")
        require(len(dev["X"]) == 128 and len(query["X"]) == len(truth["code"]) == 36,
                "Main independent card counts")
        require(set(query.files) == {"X", "seeds"}, "Unregistered query input field")
        require(np.isfinite(observed).all() and np.all(np.diff(observed[..., 0], axis=-1) >= -1e-12),
                "Conversion must be finite and nondecreasing")
        counts = {
            "n_fresh_process_paths": int(np.prod(observed.shape[:3])),
            "true_nonzero_converter_paths": int(np.count_nonzero(observed[..., -1, 0])),
            "true_nonzero_bridge_paths": int(np.count_nonzero(observed[..., -1, 5])),
            "true_winding_at_final": int(np.count_nonzero(observed[..., -1, 9])),
        }
        for name, value in counts.items():
            require(value == score["micro"][name], f"Unexpected count for {name}")
        methods = freeze["forecasters"]
        nominal = methods.index("nominal")
        inverse = {}
        bootstrap = make_bootstrap(f"{reg['run_id']}|integrated-AAm|score card bootstrap", 36, 5000)
        for k, (goal, box) in enumerate(zip(freeze["objective_names"], reference_targets, strict=True)):
            choice, probability = rank_actions(planning, box)
            require(np.array_equal(choice, prediction["choice"][:, k]) and
                    np.array_equal(choice, np.asarray(freeze["selected_actions"])[:, k]), "Main frozen choices")
            near(probability, prediction["probability"][:, k], "Main frozen planning probabilities")
            common = planning[nominal].transpose(1, 0, 2, 3, 4).reshape(13, -1, 6, 10)
            pooled_choice, _ = rank_actions(common, box)
            require(int(pooled_choice) == pool["actions"][k], "Main pooled nominal choice")
            _, hit = margin_and_hits(observed, box)
            choices = np.vstack((choice, np.zeros((1, 36), dtype=int), np.full((1, 36), pooled_choice)))
            achieved = hit[np.arange(36)[None, :], choices]
            goal_summary = score["inverse"][goal]
            near(hit.mean(axis=-1).max(axis=1).mean(), goal_summary["oracle_full_menu_hit"], "finite menu oracle")
            rates = achieved.mean(axis=-1)
            inverse[goal] = {}
            for j, name in enumerate([*methods, "no_edit", "pooled_nominal"]):
                expected = goal_summary["methods"][name]
                hits = int(achieved[j].sum())
                require(expected["hit"] == hits and expected["runs"] == achieved[j].size, "Main inverse hits")
                near(expected["hit_rate"], rates[j].mean(), "Main inverse rate")
                near(expected["delta_vs_no_edit"]["ci_95"], paired_interval(rates[j] - rates[-2], bootstrap),
                     "Main edit interval")
                inverse[goal][name] = hits

        observed_zero = observed[:, 0]
        energies = {}
        for i, name in enumerate(methods):
            values = np.array([energy_score(x, y) for x, y in zip(forward[i], observed_zero, strict=True)])
            near(values, score["new_environment_energy"][name]["per_card"], "Main energy scores by card")
            near(values.mean(), score["new_environment_energy"][name]["energy"], "Main energy mean")
            energies[name] = float(values.mean())

    return reg["run_id"], roles, {"paths": counts, "inverse_hits": inverse, "forecast_energy": energies}


def audit_paired(run, original_id, main_seeds):
    folder = run / "final_paired_process"
    freeze = read(bound_file(folder / "FROZEN.json", PAIRED_FREEZE))
    confirm, score = (read(folder / f"{name}.json") for name in ("CONFIRMED", "SCORED"))
    audit = read(folder / "INDEPENDENT_AUDIT.json")
    require(times_before(freeze, confirm) and times_before(confirm, score), "Dual-goal timestamps")
    require(confirm["freeze"] == score["freeze"] == audit["freeze"] == PAIRED_FREEZE, "Dual-goal freeze binding")
    require(audit["score_sha256"] == digest(folder / "SCORED.json") and
            confirm["raw_sha256"] == score["raw_sha256"], "Dual-goal output hash binding")
    require(tuple(freeze["arms"]) == ARMS and freeze["nominal_step"] == 0.0125, "Dual-goal comparison definitions")
    require(freeze["targets"] == read(run / "FROZEN.json")["targets"], "Shared design targets")

    cards, repeats = freeze["size"]
    with np.load(bound_file(folder / "RAW.npz", confirm["raw_sha256"]), allow_pickle=False) as raw:
        paths, seeds, initial = raw["path"], raw["seeds"], raw["initiators"]
        require(paths.shape == (5, 48, 16, 6, 10) and (cards, repeats) == (48, 16), "Dual-goal path shape")
        require(confirm["native_paths"] == np.prod(paths.shape[:3]), "Dual-goal native path count")
        require(np.isfinite(paths).all() and np.all(np.diff(paths[..., 0], axis=-1) >= -1e-12),
                "Dual-goal conversion check")
        seed_rows(main_seeds, seeds)
        require(initial.size == 240 and np.all(initial.reshape(5, 48) == initial[:48]),
                "Raw logged initiator budget mismatch")
        require(len(raw["journals"]) == 21243 and len(raw["offsets"]) == 241, "Dual-goal journal counts")

    early = margin_and_hits(paths, freeze["targets"][0])[1][[0, 1, 4]]
    late = margin_and_hits(paths, freeze["targets"][1])[1][[2, 3, 4]]
    matched = early & late
    outcomes = {"early": early, "delayed": late, "both_matched": matched}
    values = {name: panel.mean(axis=-1) for name, panel in outcomes.items()}
    values["product_under_independent_addresses"] = values["early"] * values["delayed"]
    draw = make_bootstrap(f"aam_integrated_gene_loop_{original_id}/final-paired-halved-step/card confidence measurement",
                          cards, 10000)

    counts = {}
    for goal, panel in values.items():
        reported = score["results"][goal]
        comparisons = dict(zip(BASELINE, panel, strict=True))
        comparisons["frozen_minus_static"] = panel[0] - panel[1]
        comparisons["frozen_minus_zero"] = panel[0] - panel[2]
        for label, card_stats in comparisons.items():
            near(card_stats.mean(), reported[label]["estimate"], f"{goal} {label}")
            near(paired_interval(card_stats, draw), reported[label]["ci95"], f"{goal} {label} interval")
        if goal in outcomes:
            counts[goal] = {}
            for label, data in zip(BASELINE, outcomes[goal], strict=True):
                total = int(data.sum())
                require(reported[label]["hits"] == total and reported[label]["trials"] == cards * repeats,
                        "Dual-goal hit totals")
                counts[goal][label] = total

    expected_primary = ((445, 365), (267, 61), (176, 34))
    require(tuple((r["frozen"], r["static"]) for r in counts.values()) == expected_primary, "Headline counts changed")
    effect = (paths[0, ..., 3, 3] - paths[2, ..., 3, 3]).mean(axis=-1)
    near(effect.mean(), score["middle_largest_early_minus_delayed"]["estimate"], "Midpoint component contrast")
    near(paired_interval(effect, draw), score["middle_largest_early_minus_delayed"]["ci95"],
         "Midpoint contrast interval")
    off_diagonal = (early.sum(axis=-1) * late.sum(axis=-1) - matched.sum(axis=-1)) / (repeats * (repeats - 1))
    diagnostic = read(folder / "INDEPENDENT_ADDRESS_DIAGNOSTIC.json")
    for name, value in zip(BASELINE, off_diagonal.mean(axis=-1), strict=True):
        near(value, diagnostic["out"][name]["value"], "Off-diagonal independent address rate")
    return {"cards": cards, "paired_addresses": cards * repeats, "hits": counts,
            "joint_gain_interval": score["results"]["both_matched"]["frozen_minus_static"]["ci95"]}


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--run-dir", type=Path, default=Path(__file__).resolve().parent / "run")
    cli.add_argument("--json", action="store_true", help="Print a machine-readable report to stdout.")
    args = cli.parse_args()
    original_id, seeds, main_result = audit_main(args.run_dir)
    paired_result = audit_paired(args.run_dir, original_id, seeds)
    if args.json:
        print(json.dumps({"status": "pass", "main": main_result, "paired": paired_result},
                         indent=2, allow_nan=False))
    else:
        print("PASS: main and paired freeze links, seed addresses, and numerical readback.")
        print(f"Main confirmation paths: {main_result['paths']['n_fresh_process_paths']}")
        print("Fresh paired confirmation:")
        for goal, row in paired_result["hits"].items():
            print(f"  {goal}: {row['frozen']}/{paired_result['paired_addresses']} "
                  f"vs static {row['static']}/{paired_result['paired_addresses']}")


if __name__ == "__main__":
    main()
