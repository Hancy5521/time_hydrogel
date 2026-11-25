"""Independent readback of fixed and moving views in the same histories."""
from datetime import datetime, timedelta, timezone
from hashlib import sha256
from pathlib import Path
import json

import numpy as np

from independent_fixed_event_audit import (
    check, compare, digest, independent_conditional, metrics, read, replicate_model,
)


from public_integrity import FROZEN_RUN_ID


ROOT = Path(__file__).resolve().parent
NAMES = ("prior", "snapshot_rbf", "event_ridge", "event_rbf", "conditional_single",
         "conditional_marginal", "co_located_fixed")


def chronological_bindings():
    reg, dev, code, freeze, confirm, pred, score = map(read, (
        "MOBILE_REGISTERED.json", "MOBILE_DEVELOPMENT.json", "MOBILE_MODEL_REGISTERED.json",
        "MOBILE_MODEL_FREEZE.json", "MOBILE_CONFIRMED.json", "MOBILE_PREDICTION_FREEZE.json", "MOBILE_SCORED.json"))
    times = [datetime.fromisoformat(x["utc"]) for x in (code, reg, dev, freeze, confirm, pred, score)]
    assert all(a < b for a, b in zip(times[:-1], times[1:]))
    d = lambda f: digest(ROOT / f)
    assert d("MOBILE_REGISTERED.json") == freeze["registration_sha256"]
    assert d("mobile_fit.py") == code["source_sha256"] == freeze["source_sha256"]
    assert d("MOBILE_MODELS.npz") == freeze["models_sha256"]
    assert d("MOBILE_MODEL_FREEZE.json") == confirm["freeze_sha256"] == pred["model_freeze_sha256"]
    assert d("MOBILE_PREDICTION_FREEZE.json") == score["prediction_freeze_sha256"]
    assert d("MOBILE_PREDICTED.npz") == pred["probabilities_sha256"]
    assert dev["data_sha256"] == freeze["dev_sha256"]
    assert confirm["data_sha256"] == pred["data_sha256"]
    assert dev["native_exact_noninterference_paths"] == 256
    assert list(score["results"]) == list(NAMES)
    for src, sha in reg["source_sha256"].items():
        assert digest(src) == sha
    src = (ROOT / "mobile_predict_resume.py").read_text(encoding="utf-8-sig")
    statements = [line.strip() for line in src.splitlines() if line.strip()]
    assert "mobile_fit.fixed = record[\"fixed\"]" in statements
    assert statements[-1] == "mobile_fit.predict()"
    return reg, freeze, score, {**dev["data_sha256"], **confirm["data_sha256"]}, times


def role_rng(role):
    text = f"{FROZEN_RUN_ID}|masked-graph-transition|mobile-gate-{role}|witnesses".encode()
    return np.random.default_rng(int.from_bytes(sha256(text).digest()[:16], "little"))


def complete(tape):
    observations = []
    last = -1.
    absent = False
    for row in tape:
        *risk, observed, time = row
        if observed < 0:
            absent = True
            assert np.all(row == -1.)
            continue
        assert not absent and 2.05 <= time <= 4.85 and time > last
        assert observed in (0., 1., 2., 3.) and np.min(risk) >= 0. and risk[int(observed)] > 0.
        assert sum((risk[0] + risk[1] > 0., risk[2] > 0., risk[3] > 0.)) >= 2
        observations.append((float(time), int(observed)))
        last = time
    return dict(observations)


def role_checks(reg, binding):
    used, splits, fixed, results = set(), {}, {}, {}
    for role, cards in reg["roles"].items():
        name = f"MOBILE_{role.upper()}.npz"
        assert digest(ROOT / name) == binding[name]
        with np.load(ROOT / name, allow_pickle=False) as raw:
            x, y, fixed[role], seed = (raw[k] for k in ("x", "marks", "fixed", "seeds"))
        assert x.shape == (cards, 2, 2, 4, 3, 14) and y.shape == (cards, 2, 2, 4, 3, 6)
        assert fixed[role].shape == y.shape
        assert seed.shape == (cards, 2, 4)
        reconstructed = role_rng(role).integers(0, 2**63 - 1, size=seed.shape)
        np.testing.assert_array_equal(reconstructed, seed)
        role_ids = set(seed.ravel())
        assert len(role_ids) == seed.size and not used.intersection(role_ids)
        used.update(role_ids)
        overlap = 0
        for fixed_case, moving_case in zip(fixed[role].reshape(-1, 3, 6),
                                            y.reshape(-1, 3, 6), strict=True):
            old, new = complete(fixed_case), complete(moving_case)
            shared = old.keys() & new.keys()
            for when in shared:
                assert old[when] == new[when]
            overlap += len(shared)
        splits[role] = x, y
        by_case = (y[..., 4] >= 0).sum(axis=(2, 3, 4))
        results[role] = {"fixed_captures": int((fixed[role][..., 4] >= 0).sum()),
                         "mobile_captures": int(by_case.sum()),
                         "mobile_witnesses_with_capture": int(np.count_nonzero(by_case)),
                         "co_observed_events_agree": overlap}
    assert len(used) == reg["unique_witness_addresses"]
    parent = ROOT / next(Path(path).parent for path in reg["source_sha256"] if Path(path).name == "FAIR_ENERGY_DIAGNOSTIC.json")
    prior = (*[ROOT / f"{name}.npz" for name in ("TRAIN", "VALIDATE", "TEST", "EVENT_FIT", "EVENT_DEV", "EVENT_CHECK")],
             *parent.glob("*.npz"))
    for path in prior:
        with np.load(path, allow_pickle=False) as old:
            for key in ("addresses", "seeds"):
                if key in old:
                    assert not used.intersection(old[key].ravel())
    return splits, fixed, results


def geometry_fixture():
    # Reconstruct the support of origin density across a periodic edge,
    # in physical coordinates. Both candidates give the same density if accepted.
    box, r = 6., .82
    size = .7 * box
    coord = np.array([[5.90, 1.25, 5.88], [.12, 1.27, 5.94]])
    delta = coord[1] - coord[0]
    delta -= box * np.round(delta / box)
    mid = coord[0] + .5 * delta
    chosen_uniform = np.array([.4, .1, .7])
    shift = r + (size - 2. * r) * chosen_uniform
    origin = (mid - shift) % box
    eligible = []
    for neighbor in (mid, mid + .1, mid + 3.):
        offset = (neighbor - origin) % box
        eligible.append(bool(((r <= offset) & (offset <= size - r)).all()))
    assert eligible == [True, True, False]
    check((mid - origin) % box, shift)
    # The kernel uses one preallocated dither per event innovation index.
    # Compare the observation-hook neighborhood, which is never used for inference input.
    kernel = (ROOT / "mobile_kernels.py").read_text(encoding="utf-8-sig")
    assert kernel.count("box, corridor, dither[index]") == 1
    assert kernel.find("box, corridor, dither[index]") < kernel.index("apply_event(state")
    return {"periodic_fixture_eligibility": eligible,
            "density_per_admitted_origin": float((size - 2.*r) ** -3)}


def audit():
    reg, freeze, score, binding, times = chronological_bindings()
    split, fixed, counts = role_checks(reg, binding)
    test_count = (split["check"][1][..., 4] >= 0).sum(axis=(2, 3, 4))
    np.testing.assert_array_equal(test_count, score["counts"])
    assert int(test_count.sum()) == score["captures_total"]
    assert int(np.count_nonzero(test_count)) == score["witness_groups_with_signal"]
    assert int((fixed["check"][..., 4] >= 0).sum()) == score["fixed_captures"]
    truths, maxdiff = np.tile([0, 1], reg["roles"]["dev"]), 0.
    scores = {}
    with np.load(ROOT / "MOBILE_MODELS.npz", allow_pickle=False) as model, np.load(
            ROOT / "MOBILE_PREDICTED.npz", allow_pickle=False) as pred:
        assert (pred["prior"] == .5).all()
        for method in NAMES:
            q = pred[method]
            if method == "co_located_fixed":
                recomputed = independent_conditional(fixed["check"], freeze["models"]["conditional_marginal"])
            elif method != "prior":
                settings = freeze["models"][method]
                recomputed, dev = replicate_model(method, settings, model, split)
                check(np.mean((dev - truths)**2), settings["validation_brier"])
            else:
                recomputed = np.full(q.size, .5)
            check(recomputed, q.ravel())
            maxdiff = max(maxdiff, float(np.max(np.abs(recomputed - q.ravel()))))
            scores[method] = metrics(q)
    draws = np.random.default_rng(1944402).integers(112, size=(20000, 112))
    for method, item in scores.items():
        for outcome, values in item.items():
            row = score["results"][method][outcome]
            compare(row["value"], values, draws)
            for key, ref in (("vs_prior", "prior"), ("vs_snapshot", "snapshot_rbf"), ("vs_event_rbf", "event_rbf")):
                compare(row[key], values - scores[ref][outcome], draws)
    contrast = {}
    for outcome in scores["co_located_fixed"]:
        delta = scores["conditional_marginal"][outcome] - scores["co_located_fixed"][outcome]
        contrast[outcome] = {"mean": float(np.mean(delta)),
                             "interval": np.quantile(np.take(delta, draws).mean(axis=1), (.025, .975)).tolist()}
    return {"status": "passed",
            "score_sha256": digest(ROOT / "MOBILE_SCORED.json"),
            "resume_wrapper_sha256": digest(ROOT / "mobile_predict_resume.py"),
            "independent_source_sha256": digest(__file__),
            "helper_sha256": digest(ROOT / "independent_fixed_event_audit.py"),
            "matched_acquisition": counts, "same_decoder_mobile_minus_fixed": contrast,
            "validated_probability_count": 7 * 112 * 2,
            "max_probability_error": maxdiff, "geometry": geometry_fixture(),
            "bound_times_utc": [t.isoformat() for t in times],
            "scope": "Acquisition uses coarse full-box trigger, registered conditional dither and eligibility, and nominal underlying reaction lineage/availability weights. No exact data likelihood for skipped/unobserved marks is claimed."}


if __name__ == "__main__":
    receipt = audit()
    now = datetime.now(timezone.utc)
    receipt["utc"] = now.isoformat()
    receipt["asia_shanghai"] = now.astimezone(timezone(timedelta(hours=8))).isoformat()
    with (ROOT / "MOBILE_INDEPENDENT_AUDIT.json").open("x", encoding="utf-8") as handle:
        json.dump(receipt, handle, indent=2)
        handle.write("\n")
    print(receipt)
