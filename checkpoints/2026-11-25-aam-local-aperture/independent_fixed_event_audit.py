"""Independent readback of fixed-aperture conditional mark inference."""
from datetime import datetime, timedelta, timezone
from hashlib import file_digest, sha256
import json
from pathlib import Path

import numpy as np


from public_integrity import FROZEN_RUN_ID


ROOT = Path(__file__).resolve().parent
NAMES = ("prior", "snapshot_rbf", "event_ridge", "event_rbf",
         "conditional_single", "conditional_marginal")
B = np.array([-.5, 1., -.5, 1.])
M = np.array([0., 0., -.5, 1.])


def read(name):
    return json.loads((ROOT / name).read_text(encoding="utf-8-sig"))


def digest(path):
    from public_integrity import original_digest

    return original_digest(path)


def check(x, y):
    np.testing.assert_allclose(x, y, rtol=0., atol=7.e-13)


def timestamps_and_inputs():
    reg, model, dev, freeze, confirm, predicted, score = map(read, (
        "EVENT_REGISTERED.json", "EVENT_MODEL_REGISTERED.json", "EVENT_DEVELOPMENT.json",
        "EVENT_MODEL_FREEZE.json", "EVENT_CONFIRMED.json", "EVENT_PREDICTION_FREEZE.json",
        "EVENT_SCORED.json"))
    times = sorted({d["utc"] for d in (reg, model, dev, freeze, confirm, predicted, score)})
    utc = lambda d: datetime.fromisoformat(d["utc"])
    assert utc(reg) < utc(dev) < utc(freeze) < utc(confirm) < utc(predicted) < utc(score)
    assert utc(reg) < utc(model) < utc(freeze)
    assert digest(ROOT / "EVENT_REGISTERED.json") == freeze["registration_sha256"]
    assert digest(ROOT / "event_fit.py") == freeze["source_sha256"] == model["source_sha256"]
    assert digest(ROOT / "EVENT_MODELS.npz") == freeze["models_sha256"]
    assert digest(ROOT / "EVENT_MODEL_FREEZE.json") == confirm["freeze_sha256"] == predicted["model_freeze_sha256"]
    assert digest(ROOT / "EVENT_PREDICTION_FREEZE.json") == score["prediction_freeze_sha256"]
    assert digest(ROOT / "EVENT_PREDICTED.npz") == predicted["probabilities_sha256"]
    assert dev["data_sha256"] == freeze["dev_sha256"]
    assert confirm["data_sha256"] == predicted["data_sha256"]
    assert dev["native_exact_noninterference_paths"] == 256
    for path, checksum in reg["source_sha256"].items():
        assert digest(path) == checksum
    return reg, freeze, score, {**dev["data_sha256"], **confirm["data_sha256"]}, times


def load_roles(reg, bindings):
    observed, all_ids, recap = {}, set(), {}
    for role, cards in reg["roles"].items():
        name = f"EVENT_{role.upper()}.npz"
        assert digest(ROOT / name) == bindings[name]
        with np.load(ROOT / name, allow_pickle=False) as record:
            x, mark, ids = (record[n] for n in ("x", "marks", "seeds"))
        assert x.shape == (cards, 2, 2, 4, 3, 14) and mark.shape == (cards, 2, 2, 4, 3, 6)
        assert ids.shape == (cards, 2, 4)
        text = f"{FROZEN_RUN_ID}|masked-graph-transition|event-gate-{role}|witnesses".encode()
        seed = int.from_bytes(sha256(text).digest()[:16], "little")
        np.testing.assert_array_equal(ids, np.random.default_rng(seed).integers(0, 2**63 - 1, size=ids.shape))
        assert len(set(ids.ravel())) == ids.size and not all_ids.intersection(ids.ravel())
        all_ids.update(ids.ravel())
        flat = mark.reshape(-1, 3, 6)
        for tape in flat:
            admitted = tape[:, 4] >= 0
            assert np.all(tape[~admitted] == -1.)
            assert list(admitted) == sorted(admitted, reverse=True)
            assert np.all(np.diff(tape[admitted, 5]) >= 0.)
            for *risk, choice, timestamp in tape[admitted]:
                assert 2.05 <= timestamp <= 4.85 and choice in (0, 1, 2, 3)
                assert min(risk) >= 0. and risk[int(choice)] > 0.
                assert sum((risk[0] + risk[1] > 0., risk[2] > 0., risk[3] > 0.)) >= 2
        observed[role] = x, mark
        count = (mark[..., 4] >= 0).sum(axis=(2, 3, 4))
        recap[role] = {"captures": int(count.sum()), "cards_and_sign_with_captures": int(np.count_nonzero(count)),
                       "labels_with_captures": count.sum(axis=0).tolist()}
    verify_old = ROOT / next(Path(path).parent for path in reg["source_sha256"] if Path(path).name == "FAIR_ENERGY_DIAGNOSTIC.json")
    for file in (*[ROOT / f"{role}.npz" for role in ("TRAIN", "VALIDATE", "TEST")], *verify_old.glob("*.npz")):
        with np.load(file, allow_pickle=False) as old:
            for key in ("addresses", "seeds"):
                if key in old:
                    assert not all_ids.intersection(old[key].ravel())
    assert len(all_ids) == reg["unique_witness_addresses"]
    return observed, recap


def compact_features(x, marks, mode):
    space = np.empty((*x.shape[:2], 168))
    means = x.mean(axis=3).reshape(*x.shape[:2], -1)
    devs = x.std(axis=3).reshape(*x.shape[:2], -1)
    space[..., 0::2], space[..., 1::2] = means, devs
    if mode == "snapshot":
        return space
    field = np.zeros((*x.shape[:2], 36))
    for card, branches in enumerate(marks):
        for arm, contexts in enumerate(branches):
            contexts_heads, sums, infos = [], [], []
            for env in contexts:
                choice, baseline, logs = (np.zeros(4) for _ in range(3))
                s, v = 0., 0.
                for row in env.reshape(-1, 6):
                    if row[4] < 0:
                        continue
                    exposure = row[:4]
                    prob = exposure / sum(exposure)
                    k = int(row[4])
                    choice[k] += 1.
                    baseline += prob
                    logs += np.log1p(exposure)
                    m = np.dot(M, prob)
                    s += M[k] - m
                    v += np.dot(M * M, prob) - m**2
                contexts_heads.append((choice, baseline, choice - baseline, logs))
                sums.append(s)
                infos.append(v)
            field[card, arm] = [*(item for j in range(4) for head in contexts_heads for item in head[j]), *sums, *infos]
    return np.concatenate((space, field), axis=-1)


def independent_conditional(marks, settings):
    nodes, weights = np.polynomial.legendre.leggauss(25)
    integral = (list(zip(.48 * nodes, .5 * weights)) if settings["marginal"] else [(0., 1.)])
    outputs = []
    for bundle in marks.reshape(-1, 24, 6):
        selected_rows = [(row[:4], int(row[4])) for row in bundle if row[4] >= 0]
        if not selected_rows:
            outputs.append(.5)
            continue
        evidence = [0., 0.]
        for i, slope in enumerate((-1.1, 1.1)):
            for nuisance, weight in integral:
                local_factor = np.exp(nuisance * B + slope * M)
                weights_each = [
                    float((risk * local_factor)[observed] / sum(risk * local_factor))
                    for risk, observed in selected_rows]
                evidence[i] += weight * np.prod(weights_each)
        # Independent product of mark probabilities includes each chosen reference
        # weight explicitly. Those selected factors cancel in the log ratio.
        a, b = np.asarray(evidence) ** settings["alpha"]
        outputs.append(float(b / (a + b)))
    return np.array(outputs)


def replicate_model(name, meta, archive, splits):
    train, valid, test = splits["fit"], splits["dev"], splits["check"]
    if "kernel" not in meta:
        return independent_conditional(test[1], meta), independent_conditional(valid[1], meta)
    get = lambda field: archive[f"{name}_{field}"]
    f = lambda group: compact_features(*group, meta["mode"]).reshape(2 * len(group[0]), -1)
    raw = f(train)
    center, sd = raw.mean(axis=0), raw.std(axis=0)
    normal = np.maximum(sd, .1 * np.median(sd[sd > 0.]))
    check(center, get("center"))
    check(normal, get("normal"))

    def probabilities(data):
        q = (f(data) - center) / normal
        if meta["kernel"] == "ridge":
            signal = q @ get("coef")
        else:
            support = get("support")
            check((raw - center) / normal, support)
            squared = np.stack([np.square(support - row).mean(axis=1) for row in q])
            signal = np.exp(-meta["gamma"] * squared) @ get("dual")
        return np.clip(.5 + signal, .001, .999)
    return probabilities(test), probabilities(valid)


def metrics(pred):
    a, r, e = [], [], []
    for neg, pos in pred:
        a.append(.5 * (1. if neg < .5 else .5 if neg == .5 else 0.)
                 + .5 * (1. if pos > .5 else .5 if pos == .5 else 0.))
        r.append(1. if pos > neg else .5 if pos == neg else 0.)
        e.append((neg**2 + (1. - pos)**2) / 2.)
    return {key: np.array(val) for key, val in zip(("accuracy", "paired_rank", "brier"), (a, r, e))}


def compare(record, value, indices):
    check(record["card"], value)
    check(record["mean"], value.mean())
    check(record["ci95"], np.quantile(np.take(value, indices).mean(axis=1), (.025, .975)))


def audit():
    reg, freeze, score, binding, times = timestamps_and_inputs()
    roles, counts = load_roles(reg, binding)
    truth = np.tile([0, 1], reg["roles"]["dev"])
    check_count = (roles["check"][1][..., 4] >= 0).sum(axis=(2, 3, 4))
    np.testing.assert_array_equal(check_count, score["counts"])
    assert int(check_count.sum()) == score["captures_total"]
    assert int(np.count_nonzero(check_count)) == score["witness_groups_with_signal"]
    max_error, accum = 0., {}
    with np.load(ROOT / "EVENT_MODELS.npz", allow_pickle=False) as model, np.load(
            ROOT / "EVENT_PREDICTED.npz", allow_pickle=False) as held:
        assert (held["prior"] == .5).all()
        for name in NAMES:
            if name != "prior":
                p, val = replicate_model(name, freeze["models"][name], model, roles)
                check(np.mean((val - truth)**2), freeze["models"][name]["validation_brier"])
                check(p.reshape(112, 2), held[name])
                max_error = max(max_error, float(np.max(np.abs(p - held[name].ravel()))))
                if name.startswith("conditional"):
                    assert np.all(held[name][check_count == 0] == .5)
            accum[name] = metrics(held[name])
    draws = np.random.default_rng(1944402).integers(112, size=(20000, 112))
    for method, current in accum.items():
        for metric, values in current.items():
            row = score["results"][method][metric]
            compare(row["value"], values, draws)
            for key, ref in (("vs_prior", "prior"), ("vs_snapshot", "snapshot_rbf"),
                             ("vs_event_rbf", "event_rbf")):
                compare(row[key], values - accum[ref][metric], draws)
    return {"status": "passed", "score_sha256": digest(ROOT / "EVENT_SCORED.json"),
            "audit_source_sha256": digest(__file__), "chronological_utc": times,
            "counts": counts, "probabilities_checked": len(NAMES) * 224,
            "max_independent_probability_difference": max_error,
            "source_limit": "Certified closure/bridge uses crossing bond stubs; opportunity/weight construction additionally assumes available reactive pair and nominal native lineage subtype calibration; raw marks are aggregated risk/bin records, not hidden topology recovery.",
            "estimand": "The mark likelihood given the current certified risk set is conditional; the product omits capture and risk-history likelihood."}


if __name__ == "__main__":
    out = audit()
    now = datetime.now(timezone.utc)
    out["utc"] = now.isoformat()
    out["asia_shanghai"] = now.astimezone(timezone(timedelta(hours=8))).isoformat()
    with (ROOT / "EVENT_INDEPENDENT_AUDIT.json").open("x", encoding="utf-8") as handle:
        json.dump(out, handle, indent=2)
        handle.write("\n")
    print(out)
