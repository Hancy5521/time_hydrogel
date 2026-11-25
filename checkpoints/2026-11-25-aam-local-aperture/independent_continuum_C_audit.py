"""Independent address-enumerated energy for continuous native process rollouts."""
from datetime import datetime, timedelta, timezone
from itertools import permutations
import json
from pathlib import Path

import numpy as np

from independent_continuum_witness_audit import (
    GRID, NAMES, generator, nearby_grid, numeric_truth,
)
from independent_fixed_event_audit import check, compare, digest, read


ROOT = Path(__file__).resolve().parent


def feature(arch):
    vectors = [arch[..., t, j] / (np.sqrt(8) * s)
               for t in (2, 3, 4, 5) for j, s in ((3, .3), (5, .2))]
    return np.stack(vectors, axis=-1)


def check_lock():
    reg, p, frozen, confirm, scored = map(read, ("CONTINUUM_REGISTERED.json", "CONTINUUM_PREDICTION_FREEZE.json",
                                               "CONTINUUM_C_FREEZE.json", "CONTINUUM_C_CONFIRM.json",
                                               "CONTINUUM_RESULT.json"))
    dates = [datetime.fromisoformat(x["utc"]) for x in (reg, p, frozen, confirm, scored)]
    assert all(l < r for l, r in zip(dates[:-1], dates[1:]))
    assert digest(ROOT / "CONTINUUM_PREDICTED.npz") == frozen["models_sha256"] == p["prediction_sha256"]
    assert digest(ROOT / "CONTINUUM_REGISTERED.json") == frozen["registered"]
    assert digest(ROOT / "CONTINUUM_C_BANK.npz") == frozen["bank_sha256"]
    assert digest(ROOT / "CONTINUUM_C_TRUTH.npz") == confirm["truth_sha256"]
    assert digest(ROOT / "CONTINUUM_C_FREEZE.json") == confirm["c_freeze_sha256"] == scored["C_freeze"]
    for name, sha in reg["source_sha256"].items():
        assert digest(name) == sha
    return reg, dates, scored


def check_addresses(reg, scored):
    measurements, all_ids = [], set()
    for role, expected, count_field in (("bank", 7, "C_forecast_paths"), ("truth", 2, "C_truth_paths")):
        with np.load(ROOT / f"CONTINUUM_C_{role.upper()}.npz", allow_pickle=False) as data:
            raw, ids = data["path"], data["seeds"]
        assert raw.shape == (96, 2, expected, 16, 6, 10)
        assert scored[count_field] == int(np.prod(raw.shape[:4]))
        wanted = generator("continuum-C", role).integers(0, 2**63 - 1, (96, 16))
        np.testing.assert_array_equal(ids, wanted)
        assert np.isfinite(raw).all() and not all_ids.intersection(ids.ravel())
        all_ids.update(ids.ravel())
        measurements.append(feature(raw))
    assert len(all_ids) == reg["C_addresses"] == 3072
    other = []
    # Use the pre-oracle comparison set recorded by the original audit.
    # The later privileged oracle deliberately reuses the same bank seeds.
    frozen_names = read("CONTINUUM_C_INDEPENDENT_AUDIT.json")["disjoint_existing_seed_files"]
    for filename in sorted(set(frozen_names)):
        path = ROOT / filename
        with np.load(path, allow_pickle=False) as data:
            for role in ("seeds", "addresses"):
                if role in data:
                    assert not all_ids.intersection(data[role].ravel()), path
                    other.append(path.name)
    return measurements, len(all_ids), other


def recon(bank, truth, proba):
    targets = list(proba)
    i, j = np.array(list(permutations(range(16), 2))).T
    points = {k: np.zeros(len(bank)) for k in targets}
    for card, (library, observed) in enumerate(zip(bank, truth, strict=True)):
        for atoms, outcomes in zip(library, observed, strict=True):
            # Explicit ordered independent-address pairs under the kernel-bank
            # law, using the same address omission across all component pairs.
            distance = np.linalg.norm(atoms[:, i, None, :] -
                                      np.transpose(atoms[:, j, :], (1, 0, 2))[None, :, :, :],
                                      axis=-1).mean(axis=1)
            for direction, sample in enumerate(outcomes):
                difference = np.linalg.norm(atoms[:, :, None, :] - sample[None, None, :, :],
                                            axis=-1).mean(axis=(1, 2))
                for key, weight in proba.items():
                    mass = weight[card, direction]
                    check(mass.sum(), 1.)
                    points[key][card] += (mass @ difference - .5 * np.sum(distance * np.outer(mass, mass))) / 4.
    return points, len(i)


def summarize(values, draws):
    return {"mean": float(values.mean()), "ci95": np.quantile(np.take(values, draws).mean(axis=1),
                                                              [.025, .975]).tolist()}


def audit():
    reg, dates, scored = check_lock()
    (banks, truth), n_addresses, older = check_addresses(reg, scored)
    slopes = numeric_truth("held", 96)[1]
    null = np.broadcast_to(np.eye(7)[3], (96, 2, 7))
    proba = {"fixed_four": null}
    with np.load(ROOT / "CONTINUUM_PREDICTED.npz", allow_pickle=False) as file:
        proba.update({name: file[name] for name in NAMES})
    proba["known_continuous_rule_interpolated"] = nearby_grid(slopes)
    points, samples = recon(banks, truth, proba)
    draws = np.random.default_rng(616712).integers(96, size=(16000, 96))
    maxdiff = 0.
    for key in points:
        expect = scored["energy_by_forecaster"][key]
        values = points[key]
        for role, num in (("value", values), ("vs_four", values - points["fixed_four"]),
                          ("vs_direct_rbf", values - points["mobile_rbf"])):
            origin = expect[role]
            compare(origin, num, draws)
            maxdiff = max(maxdiff, float(np.max(np.abs(num - origin["card"]))))
    pre = read("CONTINUUM_WITNESS_INDEPENDENT_AUDIT.json")
    for key, methods in pre["methods"].items():
        source = scored["score_by_rule"][key]
        check(methods["crps"], source["crps"]["mean"])
        check(methods["independent_card_losses"], source["crps"]["card"])
        check(methods["crps_95"], source["crps"]["ci95"])
        check(methods["rmse"], source["rmse"])
        for local, native in (("vs_prior", "crps_vs_prior"), ("vs_mobile_rbf", "crps_vs_rbf")):
            check(methods[local]["mean"], source[native]["mean"])
            check(methods[local]["ci95"], source[native]["ci95"])
    return {"status": "passed",
            "independent_source_sha256": digest(__file__),
            "result_sha256": digest(ROOT / "CONTINUUM_RESULT.json"),
            "witness_audit_sha256": digest(ROOT / "CONTINUUM_WITNESS_INDEPENDENT_AUDIT.json"),
            "bound_utc": [t.isoformat() for t in dates],
            "new_seed_addresses": n_addresses,
            "disjoint_existing_seed_files": sorted(older),
            "ordered_distinct_address_pairs_per_component": samples,
            "card_level_energy_estimates": 96 * len(points),
            "max_independent_estimate_difference": maxdiff,
            "mobile_conditional_minus_fixed_camera": summarize(points["mobile_conditional"] - points["static_conditional"], draws),
            "mobile_conditional_minus_fixed_four": summarize(points["mobile_conditional"] - points["fixed_four"], draws),
            "mobile_conditional_minus_same_data_rbf": summarize(points["mobile_conditional"] - points["mobile_rbf"], draws),
            "qualification": [
                "The known-rule control uses true continuous eta to interpolate two grid bank laws, not a direct native run at its exact value.",
                "Forecast peers receive the true four baseline coordinates, and data are simulated.",
                "The ordered address correction compares independent draws from physical laws per process; it does not judge cross-process coupling or policy value."
            ]}


if __name__ == "__main__":
    computed = audit()
    now = datetime.now(timezone.utc)
    computed["utc"] = now.isoformat()
    computed["asia_shanghai"] = now.astimezone(timezone(timedelta(hours=8))).isoformat()
    with (ROOT / "CONTINUUM_C_INDEPENDENT_AUDIT.json").open("x", encoding="utf-8") as handle:
        json.dump(computed, handle, indent=2)
        handle.write("\n")
    print(computed)
