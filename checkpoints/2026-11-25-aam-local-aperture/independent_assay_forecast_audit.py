"""Recompute independent-address fair energies for the fixed/moving local-event forecasts."""
import argparse
from datetime import datetime, timedelta, timezone
from hashlib import file_digest, sha256
from itertools import permutations
import json
from pathlib import Path

import numpy as np


from public_integrity import FROZEN_RUN_ID


ROOT = Path(__file__).resolve().parent
BASE = ("constant_four", "equal_mixture", "known_true_direction")


def read(filename):
    return json.loads((ROOT / filename).read_text(encoding="utf-8-sig"))


def digest(path):
    from public_integrity import original_digest

    return original_digest(path)


def paths(file):
    with np.load(ROOT / file, allow_pickle=False) as packed:
        y, ids = packed["path"], packed["seeds"]
    series = np.stack([y[..., t, j] / scale for t in (2, 3, 4, 5)
                       for j, scale in ((3, .3), (5, .2))], axis=-1)
    return series / np.sqrt(8.), ids


def address_checks(prefix, forecast, truth):
    for phase, archive in (("forecast", forecast), ("truth", truth)):
        salt = f"{FROZEN_RUN_ID}|masked-graph-transition|forward-{prefix}|{phase}".encode()
        key = int.from_bytes(sha256(salt).digest()[:16], "little")
        independently = np.random.default_rng(key).integers(0, 2**63 - 1, (112, 16))
        np.testing.assert_array_equal(archive, independently)
    all_ids = set(forecast.ravel()).union(truth.ravel())
    assert len(all_ids) == forecast.size + truth.size == 3584
    skip = {f"{prefix}_C_FORECAST.npz", f"{prefix}_C_TRUTH.npz"}
    seen = []
    for candidate in ROOT.glob("*.npz"):
        if candidate.name in skip:
            continue
        with np.load(candidate, allow_pickle=False) as data:
            for key in ("seeds", "addresses"):
                if key in data:
                    assert not all_ids.intersection(data[key].ravel())
                    seen.append(candidate.name)
    return len(all_ids), sorted(set(seen))


def admission(prefix):
    ck = lambda name: f"{prefix}_C_{name}"
    reg, lock, conf, score = (read(ck(name)) for name in
                              ("REGISTERED.json", "FORECAST_FREEZE.json", "CONFIRMATION.json", "RESULT.json"))
    pred = read(f"{prefix}_PREDICTION_FREEZE.json")
    ts = [datetime.fromisoformat(x["utc"]) for x in (pred, reg, lock, conf, score)]
    assert all(a < b for a, b in zip(ts[:-1], ts[1:]))
    d = lambda f: digest(ROOT / f)
    assert d(ck("FORECAST_FREEZE.json")) == conf["freeze_sha256"] == score["freeze_sha256"]
    assert d(ck("FORECAST.npz")) == lock["predicted_native_bank_sha256"]
    assert d(ck("TRUTH.npz")) == conf["truth_sha256"] == score["confirmed_sha256"]
    assert d(f"{prefix}_PREDICTED.npz") == lock["model_sha256"] == pred["probabilities_sha256"]
    assert reg["source_sha256"] == lock["source_sha256"]
    for f, sha in lock["source_sha256"].items():
        assert digest(f) == sha
    bank, predict_ids = paths(ck("FORECAST.npz"))
    test, measure_ids = paths(ck("TRUTH.npz"))
    assert bank.shape == (112, 2, 3, 16, 8) and test.shape == (112, 2, 2, 16, 8)
    assert int(np.prod(bank.shape[:4])) == score["paths_forecast"]
    assert int(np.prod(test.shape[:4])) == score["paths_truth"]
    used, compared = address_checks(prefix, predict_ids, measure_ids)
    assert used == reg["novel_C_addresses"]
    with np.load(ROOT / f"{prefix}_PREDICTED.npz", allow_pickle=False) as m:
        probs = {k: m[k] for k in m.files if k != "prior"}
    return bank, test, probs, score, ts, used, compared


def energies(bank, test, probs):
    names = BASE + tuple(probs)
    values = {name: np.zeros(len(bank)) for name in names}
    i, j = np.array(list(permutations(range(16), 2))).T
    for card_id, (forecasts, ground) in enumerate(zip(bank, test, strict=True)):
        for children, actual in zip(forecasts, ground, strict=True):
            # Ordered distinct physical-randomness addresses are enumerated,
            # including between-profile distances with mismatched addresses.
            distance = np.linalg.norm(children[:, i, None, :] -
                                      np.transpose(children[:, j, :], (1, 0, 2))[None, :, :, :],
                                      axis=-1).mean(axis=1)
            for sign, real in enumerate(actual):
                means = np.linalg.norm(children[:, :, None, :] - real[None, None, :, :],
                                       axis=-1).mean(axis=(1, 2))
                weights = {"constant_four": [1., 0., 0.], "equal_mixture": [0., .5, .5],
                           "known_true_direction": [0., float(sign == 0), float(sign == 1)]}
                weights.update({name: [0., 1.-q[card_id, sign], q[card_id, sign]] for name, q in probs.items()})
                for name, weight in weights.items():
                    np.testing.assert_allclose(sum(weight), 1., rtol=0., atol=1e-15)
                    values[name][card_id] += (np.dot(weight, means) - .5 * np.sum(distance * np.outer(weight, weight))) / 4.
    return values


def compute(values, draws):
    return {"mean": float(values.mean()), "ci95": np.quantile(np.take(values, draws).mean(axis=1),
                                                              (.025, .975)).tolist()}


def audit(prefix):
    bank, test, probabilities, arch, times, used, compared = admission(prefix)
    values = energies(bank, test, probabilities)
    draws = np.random.default_rng(6051851).integers(112, size=(18000, 112))
    max_value_difference = 0.
    for name, v in values.items():
        rows = arch["fair_scores"][name]
        for key, observed in (("energy", v), ("vs_four", v - values["constant_four"]),
                              ("vs_equal", v - values["equal_mixture"]),
                              ("vs_direct", v - values["event_rbf"])):
            observed_score = compute(observed, draws)
            np.testing.assert_allclose(observed, rows[key]["card"], rtol=0., atol=7e-14)
            for endpoint in observed_score:
                np.testing.assert_allclose(observed_score[endpoint], rows[key][endpoint], rtol=0., atol=7e-14)
            max_value_difference = max(max_value_difference, float(np.max(np.abs(observed - rows[key]["card"]))))
    subchecks = {}
    if prefix == "MOBILE":
        for method in ("conditional_marginal", "event_rbf", "event_ridge"):
            subchecks[method] = compute(values[method] - values["co_located_fixed"], draws)
    return {"status": "passed", "prefix": prefix, "study_cards": 112, "unique_C_addresses": used,
            "disjoint_against_saved_datasets": compared,
            "source_sha256": digest(__file__),
            "result_sha256": digest(ROOT / f"{prefix}_C_RESULT.json"),
            "chronological_utc": [t.isoformat() for t in times],
            "max_energy_difference": max_value_difference, "derived_card_energy_count": 112 * len(values),
            "methods_vs_co_located_fixed": subchecks,
            "scope": "Known four truth coordinates, held recipe profiles with new independent simulated C tape. Energy omits all same-address pairs consistently. No extra fit, wet samples, or process selection."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prefix", choices=("EVENT", "MOBILE"))
    args = parser.parse_args()
    result = audit(args.prefix)
    now = datetime.now(timezone.utc)
    result["utc"] = now.isoformat()
    result["asia_shanghai"] = now.astimezone(timezone(timedelta(hours=8))).isoformat()
    with (ROOT / f"{args.prefix}_C_INDEPENDENT_AUDIT.json").open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")
    print(result)
