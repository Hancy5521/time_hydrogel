"""Independent checks of the marked-only sensor and privileged exact oracle."""
import argparse
from datetime import datetime, timedelta, timezone
from itertools import permutations
import json
from pathlib import Path

import numpy as np

from independent_continuum_witness_audit import GRID, generator, numeric_truth
from independent_fixed_event_audit import (
    check, compact_features, compare, digest, independent_conditional, metrics, read,
)


ROOT = Path(__file__).resolve().parent
MODELS = ("prior", "old_summary", "count_rbf", "marked_counts", "full_conditional")


def rows_from_marks(data, short):
    marks = data["lite"]
    result = np.zeros((*marks.shape[:3], 8), dtype=int)
    for triple in np.ndindex(marks.shape[:3]):
        for hit in marks[triple].reshape(-1, 6):
            if hit[4] >= 0:
                key = int(hit[4]) + (4 if hit[5] > 3.45 else 0)
                result[triple][key] += 1
    if short:
        return result
    features = compact_features(data["x"], marks, "snapshot")
    return np.concatenate((features, np.log1p(result).reshape(*features.shape[:2], -1)), axis=-1)


def check_marks(raw, lite):
    assert raw.shape == lite.shape
    present_lite = lite[..., 4] >= 0.
    assert np.all(lite[..., :4][present_lite] == 1.)
    assert np.all(lite[~present_lite] == -1.)
    present_raw = raw[..., 4] >= 0.
    assert np.all(raw[~present_raw] == -1.)
    overlap = 0
    for left, right in zip(raw.reshape(-1, 3, 6), lite.reshape(-1, 3, 6), strict=True):
        groups = []
        for tape in (left, right):
            values = tape[tape[:, 4] >= 0.]
            assert np.isin(values[:, 4], (0, 1, 2, 3)).all()
            assert np.all((2.05 <= values[:, 5]) & (values[:, 5] <= 4.85))
            assert np.all(np.diff(values[:, 5]) > 0.)
            groups.append({x[-1]: x[-2] for x in values})
        common = set(groups[0]) & set(groups[1])
        assert all(groups[0][k] == groups[1][k] for k in common)
        overlap += len(common)
    return {"full_marks": int(present_raw.sum()), "lite_marks": int(present_lite.sum()),
            "both_collected_marks": overlap}


def load_minimal():
    reg, dev, freeze, hold, p, result = map(read, (
        "MINIMAL_REGISTERED.json", "MINIMAL_DATA.json", "MINIMAL_MODEL_FREEZE.json",
        "MINIMAL_HELD.json", "MINIMAL_PREDICTION_FREEZE.json", "MINIMAL_RESULT.json"))
    times = [datetime.fromisoformat(x["utc"]) for x in (reg, dev, freeze, hold, p, result)]
    assert all(a < b for a, b in zip(times[:-1], times[1:]))
    assert freeze["bindings"] == reg["binding"]
    d = lambda f: digest(ROOT / f)
    for path, sha in reg["binding"].items():
        assert digest(path) == sha
    assert d("MINIMAL_MODELS.npz") == freeze["model_sha256"]
    assert d("MINIMAL_MODEL_FREEZE.json") == hold["freeze_sha256"] == p["model_freeze_sha256"]
    assert d("MINIMAL_PREDICTED.npz") == p["probabilities_sha256"] == result["frozen_probability"]
    assert reg["cards"] == {"fit": 96, "development": 48, "held": 80}
    assert reg["noninferiority_brier"] == .010
    cubes, mark_counts, all_addresses = {}, {}, set()
    for role, n in reg["cards"].items():
        f = f"MINIMAL_{role.upper()}.npz"
        assert d(f) == (hold if role == "held" else dev)["hashes"][f]
        with np.load(ROOT / f, allow_pickle=False) as available:
            record = {k: available[k] for k in available.files}
        assert record["x"].shape == (n, 2, 2, 4, 3, 14)
        assert record["full"].shape == record["lite"].shape == (n, 2, 2, 4, 3, 6)
        cues = generator("new-minimal-" + role, "local addresses").integers(0, 2**63-1, (n, 2, 4))
        np.testing.assert_array_equal(record["seeds"], cues)
        assert len(set(cues.ravel())) == cues.size and not all_addresses.intersection(cues.ravel())
        all_addresses.update(cues.ravel())
        cubes[role] = record
        mark_counts[role] = check_marks(record["full"], record["lite"])
    assert len(all_addresses) == 1792
    # Numeric inputs do not contain per-candidate weights in the masked lite channel.
    return freeze, result, times, cubes, mark_counts


def probability_from_counts(x, mark_log, settings):
    c = rows_from_marks(x, True).reshape(-1, 2, 2, 4)
    ratio = (c * (mark_log[1] - mark_log[0])).sum(axis=(1, 2, 3))
    return 1. / (1. + np.exp(-settings["mark_scale"] * ratio))


def probability_from_frozen_rbf(data, settings, archive, name):
    rbf_model = settings[name]
    def get(key):
        return archive[f"{name}_{key}"]
    view = (compact_features(data["x"], data["lite"], "snapshot") if rbf_model["mode"] == "snapshot"
            else rows_from_marks(data, False))
    n = len(view)
    rows = (view.reshape(n * 2, -1) - get("center")) / get("std")
    support = get("support")
    m = np.stack([((r-support)**2).mean(axis=1) for r in rows])
    return np.clip(.5 + np.exp(-rbf_model["gamma"] * m) @ get("dual"), .001, .999)


def minimal():
    freeze, result, times, records, tally = load_minimal()
    train, dev, held = (records[k] for k in ("fit", "development", "held"))
    stat = {}
    with np.load(ROOT / "MINIMAL_MODELS.npz", allow_pickle=False) as bank, np.load(
            ROOT / "MINIMAL_PREDICTED.npz", allow_pickle=False) as issued:
        model = freeze["settings"]
        c = rows_from_marks(train, True).sum(axis=0).reshape(2, 2, 2, 4) + .75
        log = np.log(c / c.sum(axis=-1, keepdims=True))
        check(log, bank["mark_log"])
        dev_target = np.tile([0, 1], len(dev["x"]))
        read_settings = {"mark_scale": model["mark_scale"]}
        scores = {tau: np.mean((probability_from_counts(dev, log, {"mark_scale": tau}) - dev_target)**2)
                  for tau in (.25, .65, 1., 1.5, 2.)}
        raw_temperature = min(scores, key=lambda t: scores[t])
        assert read_settings["mark_scale"] == raw_temperature
        conditional_source = read("MOBILE_MODEL_FREEZE.json")["models"]["conditional_marginal"]
        rates = {"prior": np.full(160, .5), "marked_counts": probability_from_counts(held, log, read_settings),
                 "full_conditional": independent_conditional(held["full"], conditional_source)}
        for name in ("old_summary", "count_rbf"):
            z = probability_from_frozen_rbf(dev, model, bank, name)
            check(np.mean((z - dev_target)**2), model[name]["dev_brier"])
            rates[name] = probability_from_frozen_rbf(held, model, bank, name)
        for name in MODELS:
            assert issued[name].shape == (80, 2)
            check(rates[name], issued[name].ravel())
            stat[name] = metrics(issued[name])
    draws = np.random.default_rng(840062).integers(80, size=(18000, 80))
    for model in MODELS:
        for k, values in (("accuracy", stat[model]["accuracy"]), ("brier", stat[model]["brier"]),
                          ("vs_full_conditional_brier", stat[model]["brier"] - stat["full_conditional"]["brier"])):
            compare(result["models"][model][k], values, draws)
    calculated_gates = {}
    for a in ("count_rbf", "marked_counts"):
        sensitivity = result["models"][a]
        calculated_gates[a] = (sensitivity["accuracy"]["mean"] >= .55
                               and sensitivity["accuracy"]["ci95"][0] > .5
                               and sensitivity["vs_full_conditional_brier"]["ci95"][1] < .010)
    assert result["passes_gate"] == calculated_gates
    assert result["marks_full_then_minimal"] == [tally["held"]["full_marks"], tally["held"]["lite_marks"]]
    assert not list(ROOT.glob("MINIMAL_C_*.npz"))
    geometry = (ROOT / "minimal_geometry.py").read_text(encoding="utf-8-sig")
    assert "saved[slot, :4] = 1." in geometry and "def minimal(state, source, target," in geometry
    selection = (ROOT / "minimal_kernels.py").read_text(encoding="utf-8-sig")
    assert "minimal(state, rows[selected, 1] // 2, rows[selected, 2] // 2" in selection
    return {"status": "passed", "phase": "minimal", "timestamps_utc": [t.isoformat() for t in times],
            "verified_hold_probabilities": 800, "markers": tally, "gate": calculated_gates,
            "source_limitation": "Lite channels store constant placeholders, certified realized class/time and static summaries only. Trigger and local certificate still use idealized event type, boundary stubs and coverage."}


def paths(filename):
    with np.load(ROOT / filename, allow_pickle=False) as archived:
        value, seeds = archived["path"], archived["seeds"]
    space = np.stack([value[..., time, k] / (s * np.sqrt(8.))
                      for time in (2, 3, 4, 5) for k, s in ((3, .3), (5, .2))], axis=-1)
    return space, seeds


def exact_oracle():
    reg, conf, result, prior = map(read, ("ORACLE_REGISTERED.json", "ORACLE_NATIVE_RECEIPT.json", "ORACLE_RESULT.json",
                                        "CONTINUUM_RESULT.json"))
    times = [datetime.fromisoformat(x["utc"]) for x in (prior, reg, conf, result)]
    assert all(a < b for a, b in zip(times[:-1], times[1:]))
    for path, lock in reg["sources_sha256"].items():
        assert digest(path) == lock
    assert digest(ROOT / "ORACLE_NATIVE.npz") == conf["raw"]
    assert digest(ROOT / "ORACLE_REGISTERED.json") == conf["base_registration"]
    source, ids = paths("ORACLE_NATIVE.npz")
    measured, separate = paths("CONTINUUM_C_TRUTH.npz")
    with np.load(ROOT / "CONTINUUM_C_BANK.npz", allow_pickle=False) as earlier:
        np.testing.assert_array_equal(ids, earlier["seeds"])
    assert len(set(ids.ravel()).union(separate.ravel())) == ids.size + separate.size
    assert result["shared_forecast_addresses"] == ids.size == 1536
    assert source.shape == measured.shape == (96, 2, 2, 16, 8)
    assert conf["paths"] == result["fresh_forecast_paths"] == 6144
    _, true_eta = numeric_truth("held", 96)
    min_separation = float(np.min(np.abs(true_eta[..., None] - GRID)))
    assert min_separation > 1e-10
    pairs = np.array(list(permutations(range(16), 2))).T
    values = []
    for bank_card, truth_card in zip(source, measured, strict=True):
        household = []
        for bmode, omode in zip(bank_card, truth_card, strict=True):
            for bgroup, ogroup in zip(bmode, omode, strict=True):
                first = np.linalg.norm(bgroup[:, None, :] - ogroup[None, :, :], axis=-1).mean()
                correction = np.linalg.norm(bgroup[pairs[0]] - bgroup[pairs[1]], axis=-1).mean()
                household.append(first - correction / 2.)
        values.append(np.mean(household))
    v = np.asarray(values)
    draws = np.random.default_rng(616712).integers(96, size=(16000, 96))
    compare(result["energy"], v, draws)
    for local, parent in (("vs_fixed_four", "fixed_four"), ("vs_interpolated_oracle", "known_continuous_rule_interpolated")):
        compare(result[local], v - prior["energy_by_forecaster"][parent]["value"]["card"], draws)
    assert result["status"] == "privileged diagnostic only"
    return {"status": "passed", "phase": "oracle", "paired_forecast_addresses_intentionally_reused": ids.size,
            "blind_truth_addresses_disjoint": separate.size, "card_metrics_checked": 96,
            "off_grid_minimum_distance": min_separation,
            "timeline_utc": [t.isoformat() for t in times],
            "definitive_gain_over_four": result["vs_fixed_four"]["ci95"][1] < 0.,
            "source_limitation": "Known eta, same prior 96 cards and already-read truth. The post-score native oracle does not constitute a new confirmation."}


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("kind", choices=("MINIMAL", "ORACLE"))
    kind = cli.parse_args().kind
    written = (minimal if kind == "MINIMAL" else exact_oracle)()
    now = datetime.now(timezone.utc)
    written.update(source_sha256=digest(__file__), utc=now.isoformat(),
                   asia_shanghai=now.astimezone(timezone(timedelta(hours=8))).isoformat(),
                   result_sha256=digest(ROOT / f"{kind}_RESULT.json"))
    with (ROOT / f"{kind}_INDEPENDENT_AUDIT.json").open("x", encoding="utf-8") as handle:
        json.dump(written, handle, indent=2)
        handle.write("\n")
    print(written)
