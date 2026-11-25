"""Independent audit of the continuous occupancy-rule witness laws."""
from datetime import datetime, timedelta, timezone
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from independent_fixed_event_audit import B, M, check, compact_features, digest, read


from public_integrity import FROZEN_RUN_ID


ROOT = Path(__file__).resolve().parent
GRID = np.linspace(-1.10, 1.10, 7)
PRIOR = np.array([.5, 1, 1, 1, 1, 1, .5]) / 6
NAMES = ("prior", "static_conditional", "mobile_conditional", "mobile_knn", "mobile_rbf", "snapshot_rbf")


def generator(role, stream):
    text = f"{FROZEN_RUN_ID}|masked-graph-transition|{role}|{stream}".encode()
    return np.random.default_rng(int.from_bytes(sha256(text).digest()[:16], "little"))


def numeric_truth(role, n):
    rng = generator("continuum-" + role, "rule")
    nuisance = rng.uniform(-.48, .48, (n, 4))
    magnitude = rng.uniform(0., 1.10, n)
    slope = np.c_[-magnitude, magnitude]
    return nuisance, slope


def nearby_grid(slope):
    flat = np.asarray(slope).ravel()
    mass = np.zeros((len(flat), 7))
    for row, value in enumerate(flat):
        if value <= GRID[0]:
            mass[row, 0] = 1.
        elif value >= GRID[-1]:
            mass[row, -1] = 1.
        else:
            j = int(np.searchsorted(GRID, value, side="right"))
            mass[row, j] = (value - GRID[j-1]) / (GRID[j] - GRID[j-1])
            mass[row, j-1] = 1. - mass[row, j]
    check(mass @ GRID, flat)
    return mass.reshape(*np.asarray(slope).shape, len(GRID))


def audit_sources():
    reg, dev, model, held, pred = map(read, ("CONTINUUM_REGISTERED.json", "CONTINUUM_DEVELOPMENT.json",
                                         "CONTINUUM_MODEL_FREEZE.json", "CONTINUUM_HELD.json",
                                         "CONTINUUM_PREDICTION_FREEZE.json"))
    times = [datetime.fromisoformat(a["utc"]) for a in (reg, dev, model, held, pred)]
    assert all(a < b for a, b in zip(times[:-1], times[1:]))
    for path, sha in reg["source_sha256"].items():
        assert digest(path) == sha
    assert reg["grid"] == GRID.tolist()
    assert digest(ROOT / "continuum.py") == model["source_sha256"]
    assert digest(ROOT / "CONTINUUM_MODELS.npz") == model["models_sha256"]
    assert digest(ROOT / "CONTINUUM_MODEL_FREEZE.json") == held["model_freeze_sha256"] == pred["model_freeze"]
    assert digest(ROOT / "CONTINUUM_PREDICTED.npz") == pred["prediction_sha256"]
    assert model["development"] == dev["files_sha256"]
    assert pred["held_witness_sha256"] == held["files_sha256"]
    assert not model["held_future_existed"] and reg["new_wet_measurements"] == 0
    role_inputs = {}
    trial_seeds, readable = set(), {}
    for role, n in reg["cohorts"].items():
        file = f"CONTINUUM_{role.upper()}.npz"
        binding = pred["held_witness_sha256"] if role == "held" else model["development"]
        assert digest(ROOT / file) == binding[file]
        with np.load(ROOT / file, allow_pickle=False) as box:
            role_inputs[role] = {key: box[key] for key in ("x", "marks", "fixed", "seeds")}
        d = role_inputs[role]
        assert d["x"].shape == (n, 2, 2, 4, 3, 14)
        assert d["marks"].shape == d["fixed"].shape == (n, 2, 2, 4, 3, 6)
        ids = d["seeds"]
        expected = generator("continuum-" + role, "acquire").integers(0, 2**63-1, (n, 2, 4))
        np.testing.assert_array_equal(ids, expected)
        assert len(set(ids.ravel())) == ids.size and not trial_seeds.intersection(ids.ravel())
        trial_seeds.update(ids.ravel())
        readable[role] = {}
        for view in ("marks", "fixed"):
            records = d[view].reshape(-1, 6)
            has = records[:, 4] >= 0
            rows = records[has]
            assert np.all(records[~has] == -1.)
            assert ((rows[:, 5] >= 2.05) & (rows[:, 5] <= 4.85)).all()
            assert np.isin(rows[:, 4], [0, 1, 2, 3]).all()
            assert (rows[:, :4] >= 0.).all()
            assert (rows[np.arange(len(rows)), rows[:, 4].astype(int)] > 0.).all()
            readable[role][view] = int(has.sum())
    assert len(trial_seeds) == 2048
    for file in ("TRAIN", "VALIDATE", "TEST", "EVENT_FIT", "EVENT_DEV", "EVENT_CHECK",
                 "MOBILE_FIT", "MOBILE_DEV", "MOBILE_CHECK", "EVENT_C_FORECAST", "EVENT_C_TRUTH",
                 "MOBILE_C_FORECAST", "MOBILE_C_TRUTH"):
        with np.load(ROOT / f"{file}.npz", allow_pickle=False) as earlier:
            key = "seeds" if "seeds" in earlier else "addresses"
            assert not trial_seeds.intersection(earlier[key].ravel())
    return reg, model, role_inputs, times, readable


def reconstruct_conditional(marks, alpha):
    nodes, weights = np.polynomial.legendre.leggauss(25)
    derived = []
    lp = np.exp(.48 * nodes[:, None, None] * B + GRID[None, :, None] * M)
    for raw in marks.reshape(-1, 24, 6):
        chosen = raw[raw[:, 4] >= 0]
        evidence = np.ones((len(nodes), len(GRID)))
        for row in chosen:
            all_weights = lp * row[:4]
            evidence *= all_weights[..., int(row[4])] / all_weights.sum(axis=-1)
        mass = (.5 * (weights @ evidence))**alpha * PRIOR
        row = mass / sum(mass)
        if not len(chosen):
            check(row, PRIOR)
        derived.append(row)
    return np.asarray(derived)


def cdf_score(probs, values):
    # Integrate piecewise constant squared CDF error.  Unlike the putative
    # algorithm, this arithmetic never forms the pairwise energy kernel.
    rows = []
    for weights, y in zip(probs, np.asarray(values).ravel(), strict=True):
        bp = sorted(set([*GRID, float(y)]))
        cumulative = np.cumsum(weights)
        running = 0.
        for a, b in zip(bp[:-1], bp[1:]):
            t = (a + b) / 2.
            idx = np.searchsorted(GRID, t, side="right") - 1
            cdf = cumulative[idx] if idx >= 0 else 0.
            running += (b-a) * (cdf - float(y <= t))**2
        rows.append(running)
    return np.asarray(rows)


def direct_from_frozen(method, spec, data, stored, train, slope):
    get = lambda label: stored[f"{method}_{label}"]
    describe = lambda d: compact_features(d["x"], d["marks"], spec["mode"]).reshape(2 * len(d["x"]), -1)
    ref = describe(train)
    center, sigma = ref.mean(axis=0), ref.std(axis=0)
    norm = np.maximum(sigma, .1 * np.median(sigma[sigma > 0.]))
    check(center, get("center"))
    check(norm, get("scale"))
    support = (ref - center) / norm
    check(support, get("support"))
    soft = nearby_grid(slope).reshape(-1, 7)
    check(soft, get("mass"))
    q = (describe(data) - center) / norm
    distances = np.stack([((support - x)**2).mean(axis=1) for x in q])
    alphas = np.exp(-spec["gamma"] * distances)
    if spec["variant"] == "knn":
        masses = alphas @ soft
    else:
        masses = np.maximum(1e-7, PRIOR + alphas @ get("dual"))
    return masses / masses.sum(axis=1, keepdims=True)


def round_derive(proba, values, draws):
    rows = cdf_score(proba.reshape(-1, 7), values).reshape(len(values), 2).mean(axis=1)
    return rows, float(np.sqrt(np.mean((proba @ GRID - values)**2))), (
        np.quantile(np.take(rows, draws).mean(axis=1), (.025, .975)).tolist())


def calculate():
    reg, model, details, times, count = audit_sources()
    slopes = {role: numeric_truth(role, n)[1] for role, n in reg["cohorts"].items()}
    ref, raw, held = (details[key] for key in ("fit", "dev", "held"))
    modes = {}
    max_error = 0.
    draw = np.random.default_rng(616712).integers(96, size=(16000, 96))
    with np.load(ROOT / "CONTINUUM_PREDICTED.npz", allow_pickle=False) as fore, np.load(
            ROOT / "CONTINUUM_MODELS.npz", allow_pickle=False) as store:
        for method in NAMES:
            actual = fore[method]
            n = len(actual)
            assert ((0. <= actual) & (actual <= 1.)).all()
            check(actual.sum(axis=-1), 1.)
            if method == "prior":
                check(actual, np.broadcast_to(PRIOR, actual.shape))
            else:
                config = model["models"][method]
                if "alpha" in config:
                    which = "marks" if config["index"] == 0 else "fixed"
                    final, valid = (reconstruct_conditional(q[which], config["alpha"]) for q in (held, raw))
                else:
                    final, valid = (direct_from_frozen(method, config, group, store, ref, slopes["fit"])
                                    for group in (held, raw))
                check(cdf_score(valid, slopes["dev"]).mean(), config["dev_crps"])
                check(final.reshape(n, 2, 7), actual)
                max_error = max(max_error, float(np.max(np.abs(final.reshape(n, 2, 7) - actual))))
            losses, rmse, ci95 = round_derive(actual, slopes["held"], draw)
            modes[method] = {"crps": float(losses.mean()), "crps_95": ci95, "rmse": rmse,
                             "independent_card_losses": losses.tolist()}
    for name in modes:
        loss = np.asarray(modes[name]["independent_card_losses"])
        for reference in ("prior", "mobile_rbf"):
            delta = loss - modes[reference]["independent_card_losses"]
            modes[name][f"vs_{reference}"] = {"mean": float(delta.mean()),
                                             "ci95": np.quantile(np.take(delta, draw).mean(axis=1), (.025, .975)).tolist()}
    return {
        "status": "passed",
        "source_sha256": digest(__file__),
        "prediction_sha256": digest(ROOT / "CONTINUUM_PREDICTED.npz"),
        "chronological_utc": [t.isoformat() for t in times],
        "address_count": 2048,
        "held_probability_cells": 96 * 2 * 7 * len(NAMES),
        "max_reproduction_error": max_error,
        "event_counts": count,
        "methods": modes,
        "qualification": "Continuous targets recreated from pre-registered RNG. Regression and likelihood predictions reuse frozen settings. Law applies the occupied-only profile eta, while CRPS treats the seven-grid forecasts as point-mass distributions."
    }


if __name__ == "__main__":
    result = calculate()
    now = datetime.now(timezone.utc)
    result["utc"] = now.isoformat()
    result["asia_shanghai"] = now.astimezone(timezone(timedelta(hours=8))).isoformat()
    with (ROOT / "CONTINUUM_WITNESS_INDEPENDENT_AUDIT.json").open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")
    for key, item in result["methods"].items():
        print(key, item["crps"], item["rmse"], item["vs_prior"], item["vs_mobile_rbf"])
