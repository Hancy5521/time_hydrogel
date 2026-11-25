"""Independently replay the already frozen replication probabilities and fair scores."""
import argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path
import json

import numpy as np

from independent_assay_forecast_audit import energies, paths
from independent_continuum_witness_audit import generator
from independent_fixed_event_audit import check, compact_features, compare, digest, independent_conditional, metrics, read


ROOT = Path(__file__).resolve().parent
CARDS = 224
MODES = ("constant_four", "equal_mixture", "fixed_sensor", "snapshot_rbf",
         "mobile_rbf", "mobile_conditional", "known_direction")


def shared_checks():
    reg, witness, freeze = map(read, ("REPLICATION_REGISTERED.json", "REPLICATION_WITNESS.json", "REPLICATION_PREDICTIONS.json"))
    times = [datetime.fromisoformat(d["utc"]) for d in (reg, witness, freeze)]
    assert times[0] < times[1] < times[2]
    d = lambda filename: digest(ROOT / filename)
    for path, h in reg["sources_sha256"].items():
        assert digest(path) == h, path
    assert reg["cards"] == CARDS and reg["repeat_per_C_role"] == 16
    assert not reg["new_fit"] and freeze["source_from_prior_before_hold"]
    assert reg["near_strong_margin"] == .00075
    assert d("REPLICATION_WITNESS.npz") == witness["data_sha256"] == freeze["witness_sha256"]
    assert d("REPLICATION_PROBABILITIES.npz") == freeze["sha256"]
    expect = {}
    for role, shape in (("witness", (CARDS, 2, 4)), ("forecast", (CARDS, 16)), ("truth", (CARDS, 16))):
        expect[role] = generator("resolved_binary_replication", role).integers(0, 2**63-1, shape)
    group = set(np.concatenate([v.ravel() for v in expect.values()]))
    assert len(group) == reg["registered_addresses"] == 8960
    old = ROOT / next(Path(path).parent for path in reg["sources_sha256"] if Path(path).name == "FAIR_ENERGY_DIAGNOSTIC.json")
    old_check = [*(ROOT.glob("*.npz")), *(old.glob("*.npz"))]
    for candidate in old_check:
        if candidate.name.startswith("REPLICATION_"):
            continue
        with np.load(candidate, allow_pickle=False) as source:
            for k in ("addresses", "seeds"):
                if k in source:
                    assert not group.intersection(source[k].ravel()), candidate
    return times, expect, reg


def witness_audit():
    times, ids, reg = shared_checks()
    old = read("MOBILE_MODEL_FREEZE.json")
    assert digest(ROOT / "MOBILE_MODELS.npz") == old["models_sha256"]
    assert datetime.fromisoformat(old["utc"]) < times[0]
    with np.load(ROOT / "REPLICATION_WITNESS.npz", allow_pickle=False) as raw:
        marks, x, fixed, seeds = (raw[k] for k in ("marks", "x", "fixed", "seeds"))
    np.testing.assert_array_equal(seeds, ids["witness"])
    assert marks.shape == fixed.shape == (CARDS, 2, 2, 4, 3, 6)
    assert x.shape == (CARDS, 2, 2, 4, 3, 14)
    assert read("REPLICATION_WITNESS.json")["witness_paths"] == CARDS * 16
    configure = old["models"]["conditional_marginal"]
    calc = {"mobile_conditional": independent_conditional(marks, configure),
            "fixed_sensor": independent_conditional(fixed, configure)}
    with np.load(ROOT / "MOBILE_MODELS.npz", allow_pickle=False) as saved:
        for name, original in (("snapshot_rbf", "snapshot_rbf"), ("mobile_rbf", "event_rbf")):
            frozen = old["models"][original]
            get = lambda tag: saved[f"{original}_{tag}"]
            v = compact_features(x, marks, frozen["mode"]).reshape(CARDS * 2, -1)
            standardized = (v - get("center")) / get("normal")
            sq = np.stack([((get("support") - one)**2).mean(axis=1) for one in standardized])
            calc[name] = np.clip(.5 + np.exp(-frozen["gamma"] * sq) @ get("dual"), .001, .999)
    errors = max(float(np.max(np.abs(v))) for v in calc.values())
    assert .5 <= errors <= 1.
    local_stats, max_diff = {}, 0.
    with np.load(ROOT / "REPLICATION_PROBABILITIES.npz", allow_pickle=False) as out:
        assert set(calc) == set(out.files)
        for name, marks_probabilities in calc.items():
            saved = out[name]
            check(marks_probabilities, saved.ravel())
            max_diff = max(max_diff, float(np.max(np.abs(marks_probabilities - saved.ravel()))))
            local_stats[name] = {k: float(v.mean()) for k, v in metrics(saved).items()}
    return {"status": "passed", "input_addresses": int(seeds.size),
            "all_planned_nonoverlap_addresses": reg["registered_addresses"],
            "forecaster_probabilities_verified": int(CARDS * 2 * len(calc)),
            "maximum_recalculation_error": max_diff,
            "mobile_marks": int((marks[..., 4] >= 0).sum()),
            "fixed_marks": int((fixed[..., 4] >= 0).sum()),
            "declared_chronology_utc": [t.isoformat() for t in times],
            "last_model": digest(ROOT / "MOBILE_MODEL_FREEZE.json"),
            "witness_probabilities": digest(ROOT / "REPLICATION_PROBABILITIES.npz"),
            "witness_auxiliary_summary": local_stats,
            "scope": "Use only existing frozen parameters. Witness classification is ancillary to the pre-registered C endpoint."}


def C_audit():
    times, seeds, reg = shared_checks()
    prior = read("REPLICATION_WITNESS_INDEPENDENT_AUDIT.json")
    assert prior["status"] == "passed" and prior["witness_probabilities"] == digest(ROOT / "REPLICATION_PROBABILITIES.npz")
    frozen, observe, scored = map(read, ("REPLICATION_C_FREEZE.json", "REPLICATION_C_CONFIRM.json", "REPLICATION_RESULT.json"))
    times.extend(datetime.fromisoformat(item["utc"]) for item in (frozen, observe, scored))
    assert all(a < b for a, b in zip(times[:-1], times[1:]))
    assert digest(ROOT / "REPLICATION_C_FREEZE.json") == observe["bank"] == scored["frozen_bank_sha256"]
    assert digest(ROOT / "REPLICATION_FORECAST.npz") == frozen["sha256"]
    assert digest(ROOT / "REPLICATION_TRUTH.npz") == observe["sha256"]
    assert digest(ROOT / "REPLICATION_PROBABILITIES.npz") == frozen["probability_sha256"] == observe["probability_sha256"]
    bank, bank_seeds = paths("REPLICATION_FORECAST.npz")
    target, target_seeds = paths("REPLICATION_TRUTH.npz")
    np.testing.assert_array_equal(bank_seeds, seeds["forecast"])
    np.testing.assert_array_equal(target_seeds, seeds["truth"])
    assert bank.shape == (CARDS, 2, 3, 16, 8) and target.shape == (CARDS, 2, 2, 16, 8)
    assert np.all(np.isfinite(bank)) and np.all(np.isfinite(target))
    assert int(np.prod(target.shape[:4])) == scored["blind_C_paths"] == observe["native_paths"]
    assert int(np.prod(bank.shape[:4])) == frozen["native_paths"]
    with np.load(ROOT / "REPLICATION_PROBABILITIES.npz", allow_pickle=False) as probabilities:
        q = {k: probabilities[k] for k in probabilities.files}
    values = energies(bank, target, q)
    values["known_direction"] = values.pop("known_true_direction")
    assert set(values) == set(MODES) == set(scored["scores"])
    rng = np.random.default_rng(319684).integers(CARDS, size=(22000, CARDS))
    max_difference = 0.
    for name, per_card in values.items():
        entry = scored["scores"][name]
        numerics = (("energy", per_card), ("vs_four", per_card - values["constant_four"]),
                    ("vs_same_input_rbf", per_card - values["mobile_rbf"]))
        for tag, profile in numerics:
            compare(entry[tag], profile, rng)
            max_difference = max(max_difference, float(np.max(np.abs(profile - entry[tag]["card"]))))
    primary = scored["scores"]["mobile_conditional"]
    win = primary["vs_four"]["ci95"][1] < 0.
    close_baseline = primary["vs_same_input_rbf"]["ci95"][1] < reg["near_strong_margin"]
    assert scored["succeeded_vs_four"] == win and scored["near_rbf"] == close_baseline
    return {"status": "passed", "registration_address_count": reg["registered_addresses"],
            "verified_C_forecast_paths": frozen["native_paths"],
            "verified_C_independent_truth_paths": observe["native_paths"],
            "physical_process_estimands_checked": len(values) * CARDS,
            "max_energy_card_error": max_difference,
            "declared_chronology_utc": [t.isoformat() for t in times],
            "archived_result_sha256": digest(ROOT / "REPLICATION_RESULT.json"),
            "primary_vs_four": {k: primary["vs_four"][k] for k in ("mean", "ci95")},
            "primary_vs_direct": {k: primary["vs_same_input_rbf"][k] for k in ("mean", "ci95")},
            "superiority_established": win, "predeclared_noninferiority_established": close_baseline,
            "qualification": "Strong binary occupancy perturbations and known four coordinates. No continuous or real-material extrapolation, no new design choice."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("WITNESS", "C"))
    name = parser.parse_args().stage
    data = (witness_audit if name == "WITNESS" else C_audit)()
    now = datetime.now(timezone.utc)
    data.update(source_sha256=digest(__file__), utc=now.isoformat(),
                asia_shanghai=now.astimezone(timezone(timedelta(hours=8))).isoformat())
    with (ROOT / f"REPLICATION_{name}_INDEPENDENT_AUDIT.json").open("x", encoding="utf-8") as out:
        json.dump(data, out, indent=2)
        out.write("\n")
    print(data)
