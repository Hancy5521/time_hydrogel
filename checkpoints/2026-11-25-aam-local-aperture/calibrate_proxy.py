"""Learn a low-dimensional geometry-to-eligible-risk calibration on development cards."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logsumexp

import capture as src
import proxy
import mobile_capture as mobile
import mobile_fit as core
from fit import labels, summary, to_rows, distance


ROOT = Path(__file__).resolve().parent
CARDS = 112


def cohort():
    return (src.rng("calibrated-proxy-fresh", "codes").uniform(-.48, .48, (CARDS, 4)),
            src.rng("calibrated-proxy-fresh", "independent witnesses").integers(0, 2**63 - 1, (CARDS, 2, 4)))


def bindings():
    files = ("calibrate_proxy.py", "CAL_PROXY_PROTOCOL_ZH.md", "PROXY_RESULT.json", "PROXY_FIT.npz",
             "PROXY_DEVELOPMENT.npz", "PROXY_MODELS.npz", "PROXY_MODEL_FREEZE.json", "mobile_fit.py")
    return {**proxy.binding(), **{str(ROOT / name): src.digest(ROOT / name) for name in files}}


def register():
    ids = cohort()[1].ravel()
    assert np.unique(ids).size == len(ids)
    for f in ROOT.glob("*.npz"):
        with np.load(f, allow_pickle=False) as old:
            for key in ("seeds", "addresses"):
                if key in old.files:
                    assert not np.intersect1d(ids, old[key]).size
    src.write("CAL_PROXY_REGISTERED.json", {"binding": bindings(), "fresh_held_cards": CARDS,
               "model_grid": "global or 4 strata [(A,B) × (before,after3.45)]; L2=.002,.02,.2; alpha .25,.65,1,1.5.",
               "uses_revealed_held_data_for_selection": False})


def pairs(reference, visible):
    x, y, groups = [], [], []
    for (c, sign, env, rep), stay in np.ndenumerate(visible[..., 0, 0]):
        measured = visible[c, sign, env, rep]
        truth = reference[c, sign, env, rep]
        for record in measured:
            if record[4] < 0.:
                continue
            match = truth[(truth[:, 4] >= 0.) & (truth[:, 5] == record[5])]
            if len(match) == 0:
                continue
            assert len(match) == 1 and int(record[4]) == int(match[0, 4])
            if np.any((record[:4] <= 0.) & (match[0, :4] > 0.)):
                continue
            x.append(record[:4])
            y.append(match[0, :4])
            groups.append(2 * env + int(record[5] > 3.45))
    return np.asarray(x), np.asarray(y), np.asarray(groups)


def fit_reference(x, y, groups, n, penalty):
    p = np.maximum(x / x.sum(axis=-1, keepdims=True), 1.e-100)
    t = y / y.sum(axis=-1, keepdims=True)
    logq = np.log(p)
    def objective(v):
        z = v.reshape(n, 4)
        part = logq + z[groups]
        logged = part - logsumexp(part, axis=-1, keepdims=True)
        value = -(t * logged).sum() / len(p) + penalty * np.dot(v, v) * .5
        grad = np.zeros_like(z)
        np.add.at(grad, groups, np.exp(logged) - t)
        return value, grad.ravel() / len(p) + penalty * v
    ans = minimize(objective, np.zeros(n*4), jac=True, method="L-BFGS-B")
    assert ans.success, ans.message
    return ans.x.reshape(n, 4)


def transform(marks, bias):
    y = marks.copy()
    if len(bias) == 1:
        z = bias[0]
    else:
        env = np.arange(2).reshape(1, 1, 2, 1, 1)
        z = bias[(marks[..., 5] > 3.45).astype(int) + 2*env]
    y[..., :4] *= np.exp(z)
    return y


def freeze():
    assert src.read("CAL_PROXY_REGISTERED.json")["binding"] == bindings()
    assert not (ROOT / "CAL_PROXY_RAW.npz").exists()
    ref, _, vis = proxy.load("fit")
    x, y, group = pairs(ref, vis)
    assert len(x) > 80
    val_data = proxy.load("development")[2]
    target = labels(proxy.SETS["development"])
    options = []
    for groups in (1, 4):
        identity = group if groups == 4 else np.zeros_like(group)
        for penalty in (.002, .02, .2):
            b = fit_reference(x, y, identity, groups, penalty)
            log = core.conditional(transform(val_data, b), True)
            for alpha in (.25, .65, 1., 1.5):
                loss = float(np.mean((expit(alpha * log)-target)**2))
                options.append((loss, groups, penalty, alpha, b.tolist()))
    loss, groups, penalty, alpha, b = min(options, key=lambda x: x[0])
    src.write("CAL_PROXY_FROZEN.json", {"source_hash": src.digest(Path(__file__)),
              "paired_training_records": len(x), "validation_brier": loss, "groups": groups,
              "penalty": penalty, "alpha": alpha, "bias": b})


def sample(workers):
    lock = src.read("CAL_PROXY_FROZEN.json")
    assert lock["source_hash"] == src.digest(Path(__file__))
    g, a = cohort()
    cases = ((z, ctx, i, s, False) for z, rr in zip(g, a, strict=True) for s in (-1., 1.)
             for ctx, env in zip(("A", "B"), rr, strict=True) for i in env)
    with ProcessPoolExecutor(max_workers=workers, initializer=proxy.prep) as pool:
        results = list(pool.map(mobile.evidence, cases, chunksize=16))
    one, x, alt = map(np.asarray, zip(*results, strict=True))
    shapes = (CARDS, 2, 2, 4)
    src.arrays("CAL_PROXY_RAW.npz", exact=one.reshape(*shapes, 3, 6), marks=alt.reshape(*shapes, 3, 6),
               x=x.reshape(*shapes, 3, 14), seeds=a)
    src.write("CAL_PROXY_TRUTH_RECEIPT.json", {"data_sha256": src.digest(ROOT / "CAL_PROXY_RAW.npz"),
               "freeze_sha256": src.digest(ROOT / "CAL_PROXY_FROZEN.json")})


def score():
    pin = src.read("CAL_PROXY_TRUTH_RECEIPT.json")
    assert pin["data_sha256"] == src.digest(ROOT / "CAL_PROXY_RAW.npz")
    assert pin["freeze_sha256"] == src.digest(ROOT / "CAL_PROXY_FROZEN.json")
    lock = src.read("CAL_PROXY_FROZEN.json")
    with np.load(ROOT / "CAL_PROXY_RAW.npz", allow_pickle=False) as raw:
        exact, x, m = (raw[key] for key in ("exact", "x", "marks"))
    field = src.read("PROXY_MODEL_FREEZE.json")["settings"]
    alpha = src.read("MOBILE_MODEL_FREEZE.json")["models"]["conditional_marginal"]["alpha"]
    p = {"prior": np.full(CARDS*2, .5),
         "full": expit(alpha * core.conditional(exact, True)),
         "raw": expit(field["mark_scale"] * core.conditional(m, True)),
         "geometry_calibrated": expit(lock["alpha"] * core.conditional(transform(m, np.asarray(lock["bias"])), True))}
    with np.load(ROOT / "PROXY_MODELS.npz", allow_pickle=False) as weights:
        target = "proxy_rbf"
        t = field[target]
        def get(k):
            return weights[f"{target}_{k}"]
        feat = to_rows(core.features(m, x, "events"))
        query = (feat - get("center")) / get("std")
        pred = np.exp(-t["gamma"] * distance(query, get("support"))) @ get("dual")
        p[target] = np.clip(.5 + pred, .001, .999)
    src.arrays("CAL_PROXY_PREDICTED.npz", **{k: v.reshape(CARDS, 2) for k, v in p.items()})
    v = labels(CARDS)
    draws = np.random.default_rng(4011691).integers(CARDS, size=(18000, CARDS))
    scores = {}
    ref = ((p["full"] - v)**2).reshape(CARDS, 2).mean(axis=1)
    for model, prob in p.items():
        loss = ((prob-v)**2).reshape(CARDS, 2).mean(axis=1)
        acc = np.where(prob == .5, .5, (prob > .5) == v).reshape(CARDS, 2).mean(axis=1)
        scores[model] = {"brier": summary(loss, draws), "vs_full": summary(loss-ref, draws),
                         "accuracy": summary(acc, draws)}
    candidate = scores["geometry_calibrated"]
    promote = candidate["accuracy"]["mean"] >= .55 and candidate["accuracy"]["ci95"][0] > .50 and candidate["vs_full"]["ci95"][1] < .010
    src.write("CAL_PROXY_RESULT.json", {"frozen_sha256": pin["freeze_sha256"], "allow_new_C": bool(promote),
              "methods": scores, "held_marks_full_and_proxy": [int((z[..., 4] >= 0).sum()) for z in (exact, m)]})


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("stage", choices=("register", "freeze", "sample", "score"))
    cli.add_argument("--workers", type=int, default=10)
    v = cli.parse_args()
    if v.stage == "sample":
        sample(v.workers)
    else:
        {"register": register, "freeze": freeze, "score": score}[v.stage]()

