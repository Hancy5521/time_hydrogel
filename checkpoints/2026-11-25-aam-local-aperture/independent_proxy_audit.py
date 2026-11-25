"""Independent readback of geometry-only risk and its fit-only calibration."""
import argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path
import json

import numpy as np

from independent_continuum_witness_audit import generator
from independent_fixed_event_audit import (
    check, compare, compact_features, digest, independent_conditional, metrics, read,
)


ROOT = Path(__file__).resolve().parent


def conditional(raw, alpha):
    return independent_conditional(raw, {"marginal": True, "alpha": alpha})


def kernel(x, m, name, parameters, archive, shape=None):
    cfg = parameters[name]
    meta = lambda k: archive[f"{name}_{k}"]
    whole = compact_features(x, m, "snapshot" if cfg["mode"] == "snapshot" else "events")
    r = (whole.reshape(-1, whole.shape[-1]) - meta("center")) / meta("std")
    reference = meta("support")
    d = np.stack([np.square(reference - v).mean(axis=1) for v in r])
    return np.clip(.5 + np.exp(-cfg["gamma"] * d) @ meta("dual"), .001, .999).reshape(-1, 2)


def verify_dates(*docs):
    times = [datetime.fromisoformat(x["utc"]) for x in docs]
    assert all(a < b for a, b in zip(times[:-1], times[1:]))
    return [t.isoformat() for t in times]


def stacked(name):
    with np.load(ROOT / f"{name}.npz", allow_pickle=False) as d:
        return {key: d[key] for key in d.files}


def reveal(namespace, n, name, d):
    assert digest(ROOT / f"{name}.npz") == d
    arr = stacked(name)
    seed = arr["seeds"]
    assert seed.shape == (n, 2, 4)
    np.testing.assert_array_equal(seed, generator(*namespace).integers(0, 2**63 - 1, size=seed.shape))
    assert len(set(seed.ravel())) == seed.size
    return arr


def details(truth, alternative):
    assert truth.shape == alternative.shape
    sets = [t[t[..., 4] >= 0] for t in (truth, alternative)]
    for records, arr in zip(sets, (truth, alternative), strict=True):
        assert np.all(arr[arr[..., 4] < 0] == -1.)
        assert (records[:, :4] >= 0.).all()
        assert np.isin(records[:, 4], (0, 1, 2, 3)).all()
        assert np.all((2.05 <= records[:, 5]) & (records[:, 5] <= 4.85))
        assert (records[np.arange(len(records)), records[:, 4].astype(int)] > 0).all()
    overlap = 0
    for a, b in zip(truth.reshape(-1, 3, 6), alternative.reshape(-1, 3, 6), strict=True):
        old, new = ({(row[5]): row[4] for row in s if row[4] >= 0} for s in (a, b))
        same = old.keys() & new.keys()
        assert all(old[time] == new[time] for time in same)
        overlap += len(same)
    return [len(sets[0]), len(sets[1])], overlap


def storage_scope():
    geom = (ROOT / "proxy_geometry.py").read_text(encoding="utf-8-sig")
    for leak in ("state.active", "state.labels", "primitive[", "rows[", "native_kind"):
        assert leak not in geom
    assert "state.owner[2*i] >= 0" in geom and "degree[j] != 1" in geom
    assert ".5 * (reference[1, q, band] + reference[2, q, band])" in geom
    hook = (ROOT / "proxy_kernels.py").read_text(encoding="utf-8-sig")
    assert "proxy(state, rows[selected, 1] // 2, rows[selected, 2] // 2," in hook
    return "Geometry candidates use visible occupied-site counts, positions, certified closure/separation and nominal rates; selected pendant trigger and exact boundary stubs are idealized. Calibration has privileged teacher exposure only on fit trajectories."


def compare_models(checkpoint, predictions, n, rivals, draw):
    expected = stacked(checkpoint + "_PREDICTED")
    assert set(predictions) == set(expected)
    for name, p in predictions.items():
        assert p.shape == (n, 2)
        check(p, expected[name])
    return {key: metrics(p) for key, p in expected.items()}


def passes(score, name_vs):
    return (score["accuracy"]["mean"] >= .55 and score["accuracy"]["ci95"][0] > .5
            and score[name_vs]["ci95"][1] < .010)


def paired_summaries(score, values, ref, draws, key):
    for m, record in score.items():
        for label, data in (("accuracy", values[m]["accuracy"]), ("brier", values[m]["brier"]),
                            (key, values[m]["brier"] - values[ref]["brier"])):
            compare(record[label], data, draws)


def proxy():
    reg, dev, frozen, held, p, scored = map(read, (
        "PROXY_REGISTERED.json", "PROXY_DATA.json", "PROXY_MODEL_FREEZE.json", "PROXY_HELD.json",
        "PROXY_PREDICTION_FREEZE.json", "PROXY_RESULT.json"))
    times = verify_dates(reg, dev, frozen, held, p, scored)
    d = lambda x: digest(ROOT / x)
    assert reg["binding"] == frozen["bindings"]
    for path, sha in reg["binding"].items():
        assert digest(path) == sha
    assert d("PROXY_MODELS.npz") == frozen["model_sha256"]
    assert d("PROXY_MODEL_FREEZE.json") == held["freeze_sha256"] == p["model_freeze_sha256"]
    assert d("PROXY_PREDICTED.npz") == p["probabilities_sha256"] == scored["frozen_probability"]
    data, ids, counts = {}, set(), {}
    for role, n in reg["cards"].items():
        archive = f"PROXY_{role.upper()}"
        data[role] = reveal(("new-proxy-" + role, "local addresses"), n,
                            archive, (held if role == "held" else dev)["hashes"][archive + ".npz"])
        arr = data[role]
        assert arr["lite"].shape == (n, 2, 2, 4, 3, 6)
        s = set(arr["seeds"].ravel())
        assert not ids.intersection(s)
        ids.update(s)
        counts[role] = details(arr["full"], arr["lite"])
    assert scored["marks_full_then_proxy"] == counts["held"][0]
    ref, level = data["held"], data["development"]
    truth = np.tile([0, 1], len(level["x"]))
    calibration = frozen["settings"]
    result = {"prior": np.full((80, 2), .5),
              "full_conditional": conditional(ref["full"], read("MOBILE_MODEL_FREEZE.json")["models"]["conditional_marginal"]["alpha"]).reshape(-1, 2),
              "proxy_conditional": conditional(ref["lite"], calibration["mark_scale"]).reshape(-1, 2)}
    scores = {s: ((conditional(level["lite"], s) - truth)**2).mean() for s in (.25, .65, 1., 1.5, 2.)}
    assert min(scores, key=scores.get) == calibration["mark_scale"]
    with np.load(ROOT / "PROXY_MODELS.npz", allow_pickle=False) as bank:
        for key in ("old_summary", "proxy_rbf"):
            result[key] = kernel(ref["x"], ref["lite"], key, calibration, bank)
            current = kernel(level["x"], level["lite"], key, calibration, bank).ravel()
            check(np.mean((current - truth)**2), calibration[key]["dev_brier"])
    draws = np.random.default_rng(840062).integers(80, size=(18000, 80))
    values = compare_models("PROXY", result, 80, (), draws)
    paired_summaries(scored["models"], values, "full_conditional", draws, "vs_full_conditional_brier")
    gates = {name: passes(scored["models"][name], "vs_full_conditional_brier")
             for name in ("proxy_rbf", "proxy_conditional")}
    assert gates == scored["passes_gate"]
    return {"status": "passed", "phase": "proxy", "timestamps_utc": times,
            "fresh_addresses_verified": len(ids), "captured_marks_and_matched_overlap": counts,
            "probabilities_verified": 800, "gate": gates, "scope": storage_scope()}


def matched_risks(fit):
    exact, geom = fit["full"], fit["lite"]
    xs, ys, groups = [], [], []
    for location in np.ndindex(exact.shape[:4]):
        old = {row[5]: row for row in exact[location] if row[4] >= 0}
        for proposed in geom[location]:
            if proposed[4] < 0 or proposed[5] not in old:
                continue
            target = old[proposed[5]]
            assert target[4] == proposed[4]
            if np.any((proposed[:4] <= 0) & (target[:4] > 0)):
                continue
            xs.append(proposed[:4])
            ys.append(target[:4])
            groups.append(2 * location[2] + int(proposed[5] > 3.45))
    return map(np.asarray, (xs, ys, groups))


def correction(marks, beta):
    output = np.copy(marks)
    for sub in np.ndindex(marks.shape[:4]):
        for idx, row in enumerate(marks[sub]):
            choice = 0 if len(beta) == 1 else 2 * sub[2] + int(row[5] > 3.45)
            output[(*sub, idx, slice(0, 4))] *= np.exp(beta[choice])
    return output


def calibrated():
    r, freeze, truth, scored = map(read, ("CAL_PROXY_REGISTERED.json", "CAL_PROXY_FROZEN.json",
                                        "CAL_PROXY_TRUTH_RECEIPT.json", "CAL_PROXY_RESULT.json"))
    times = verify_dates(r, freeze, truth, scored)
    d = lambda x: digest(ROOT / x)
    for name, sha in r["binding"].items():
        assert digest(name) == sha
    assert not r["uses_revealed_held_data_for_selection"]
    assert truth["freeze_sha256"] == scored["frozen_sha256"] == d("CAL_PROXY_FROZEN.json")
    assert freeze["source_hash"] == d("calibrate_proxy.py")
    fit, dev = (stacked("PROXY_" + kind) for kind in ("FIT", "DEVELOPMENT"))
    n = r["fresh_held_cards"]
    assert n == 112
    h = reveal(("calibrated-proxy-fresh", "independent witnesses"), n, "CAL_PROXY_RAW", truth["data_sha256"])
    known = h["seeds"].ravel()
    for candidate in ROOT.glob("*.npz"):
        if candidate.name == "CAL_PROXY_RAW.npz":
            continue
        with np.load(candidate, allow_pickle=False) as file:
            for role in ("seeds", "addresses"):
                if role in file:
                    assert not set(known).intersection(file[role].ravel())
    beta = np.asarray(freeze["bias"])
    x, y, group = matched_risks(fit)
    assert len(x) == freeze["paired_training_records"]
    if freeze["groups"] == 1:
        group = np.zeros_like(group)
    fit_norm = x / x.sum(axis=1, keepdims=True)
    exact = y / y.sum(axis=1, keepdims=True)
    w = fit_norm * np.exp(beta[group])
    derivative = np.zeros(beta.shape)
    np.add.at(derivative, group, w / w.sum(axis=1, keepdims=True) - exact)
    gradient_residual = derivative / len(x) + freeze["penalty"] * beta
    assert np.max(np.abs(gradient_residual)) < 8e-5

    assert freeze["groups"] in (1, 4) and freeze["alpha"] in (.25, .65, 1., 1.5)
    dev_labels = np.tile([0, 1], len(dev["x"]))
    p = conditional(correction(dev["lite"], beta), freeze["alpha"])
    check(((p - dev_labels)**2).mean(), freeze["validation_brier"])
    old = read("PROXY_MODEL_FREEZE.json")["settings"]
    full = read("MOBILE_MODEL_FREEZE.json")["models"]["conditional_marginal"]["alpha"]
    fitted = {"prior": np.full((n, 2), .5),
              "full": conditional(h["exact"], full).reshape(n, 2),
              "raw": conditional(h["marks"], old["mark_scale"]).reshape(n, 2),
              "geometry_calibrated": conditional(correction(h["marks"], beta), freeze["alpha"]).reshape(n, 2)}
    with np.load(ROOT / "PROXY_MODELS.npz", allow_pickle=False) as calibration:
        fitted["proxy_rbf"] = kernel(h["x"], h["marks"], "proxy_rbf", old, calibration)
    draws = np.random.default_rng(4011691).integers(n, size=(18000, n))
    values = compare_models("CAL_PROXY", fitted, n, (), draws)
    paired_summaries(scored["methods"], values, "full", draws, "vs_full")
    gate = passes(scored["methods"]["geometry_calibrated"], "vs_full")
    assert scored["allow_new_C"] == gate
    assert scored["held_marks_full_and_proxy"] == details(h["exact"], h["marks"])[0]
    return {"status": "passed", "phase": "calibrated proxy", "timestamps_utc": times,
            "fresh_addresses_verified": int(known.size), "fit_overlap_with_matching_supported_risks": len(x),
            "gradient_residual_max_at_frozen_fit": float(np.max(np.abs(gradient_residual))),
            "verified_held_probabilities": int(n * 2 * len(fitted)), "gate": gate, "scope": storage_scope()}


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("phase", choices=("PROXY", "CAL_PROXY"))
    args = cli.parse_args()
    record = (proxy if args.phase == "PROXY" else calibrated)()
    now = datetime.now(timezone.utc)
    record.update(source_sha256=digest(__file__), result_sha256=digest(ROOT / f"{args.phase}_RESULT.json"),
                  utc=now.isoformat(), asia_shanghai=now.astimezone(timezone(timedelta(hours=8))).isoformat())
    with (ROOT / f"{args.phase}_INDEPENDENT_AUDIT.json").open("x", encoding="utf-8") as f:
        json.dump(record, f, indent=2)
        f.write("\n")
    print(record)
