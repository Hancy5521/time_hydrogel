"""Continuous local rule inference and fair native process forecast."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy.linalg import solve
from scipy.special import logsumexp
from scipy.spatial.distance import cdist

import capture as src
import mobile_capture as mobile
from mobile_fit import features, received, TYPE, SENS
from fit import summary
from forecast import config as prepare_native, NATIVE_SETTINGS, feature


ROOT = Path(__file__).resolve().parent
ROLES = {"fit": 112, "dev": 48, "held": 96}
REPS = 16
GRID = np.linspace(-1.10, 1.10, 7)
PRIOR = np.array([.5, 1., 1., 1., 1., 1., .5]) / 6.
GRID_DISTANCE = np.abs(GRID[:, None] - GRID[None])
METHODS = ("prior", "static_conditional", "mobile_conditional", "mobile_knn", "mobile_rbf", "snapshot_rbf")
PHYSICAL = ("fixed_four", *METHODS)


def cohort(role):
    n = ROLES[role]
    rng = src.rng("continuum-" + role, "rule")
    return rng.uniform(-.48, .48, (n, 4)), rng.uniform(0., 1.10, n)[:, None] * [-1, 1], (
        src.rng("continuum-" + role, "acquire").integers(0, 2**63 - 1, (n, 2, 4)))


def addresses(role):
    return src.rng("continuum-C", role).integers(0, 2**63 - 1, (ROLES["held"], REPS))


def bindings():
    expected = src.read("MOBILE_REGISTERED.json")["source_sha256"]
    for path, recorded in expected.items():
        assert src.digest(Path(path)) == recorded
    paths = ("continuum.py", "CONTINUUM_PROTOCOL_ZH.md", "MOBILE_SCORED.json",
             "MOBILE_PREDICTION_FREEZE.json", "MOBILE_PREDICTED.npz",
             "fit.py", "mobile_fit.py", "forecast.py")
    return {**expected, **{str(ROOT / path): src.digest(ROOT / path) for path in paths}}


def register():
    future = np.concatenate([addresses(role).ravel() for role in ("bank", "truth")])
    all_seeds = np.concatenate([future, *(cohort(role)[2].ravel() for role in ROLES)])
    assert np.unique(all_seeds).size == all_seeds.size
    for name in ("TRAIN", "VALIDATE", "TEST", "EVENT_FIT", "EVENT_DEV", "EVENT_CHECK",
                 "MOBILE_FIT", "MOBILE_DEV", "MOBILE_CHECK"):
        with np.load(ROOT / (name + ".npz"), allow_pickle=False) as bundle:
            key = "seeds" if "seeds" in bundle.files else "addresses"
            assert not np.intersect1d(all_seeds, bundle[key]).size
    src.write("CONTINUUM_REGISTERED.json", {"source_sha256": bindings(), "cohorts": ROLES,
              "time_role": "held witnesses after model freeze, held C before score",
              "slope_truth": "continuous paired uniform magnitudes", "grid": GRID.tolist(),
              "C_addresses": future.size, "new_wet_measurements": 0})


def verify():
    assert src.read("CONTINUUM_REGISTERED.json")["source_sha256"] == bindings()


def witness(task):
    g, ctx, seed, change = task
    return mobile.evidence((g, ctx, seed, change / src.SLOPE, False))


def acquire(at, workers):
    verify()
    names = ("fit", "dev") if at == "dev" else ("held",)
    lock = src.digest(ROOT / "CONTINUUM_MODEL_FREEZE.json") if at == "held" else None
    hashes = {}
    with ProcessPoolExecutor(max_workers=workers, initializer=mobile.setup) as pool:
        for key in names:
            g, eta, ids = cohort(key)
            work = ((z, env, iid, slope) for z, ss, group in zip(g, eta, ids, strict=True)
                    for slope in ss for env, row in zip(("A", "B"), group, strict=True) for iid in row)
            out = list(pool.map(witness, work, chunksize=16))
            mark, x, fixed = [np.array(x) for x in zip(*out, strict=True)]
            shape = (len(g), 2, 2, 4)
            name = f"CONTINUUM_{key.upper()}.npz"
            src.arrays(name, marks=mark.reshape(*shape, 3, 6), x=x.reshape(*shape, 3, 14),
                       fixed=fixed.reshape(*shape, 3, 6), seeds=ids)
            hashes[name] = src.digest(ROOT / name)
            print("continuum", key, "moves", int((mark[..., 4] >= 0).sum()), flush=True)
    src.write("CONTINUUM_DEVELOPMENT.json" if at == "dev" else "CONTINUUM_HELD.json",
               {"files_sha256": hashes, "model_freeze_sha256": lock})


def soft_truth(y):
    step = GRID[1] - GRID[0]
    w = np.maximum(0., 1. - np.abs(y[..., None] - GRID) / step)
    return w / w.sum(axis=-1, keepdims=True)


def law(mark, alpha):
    risk, chosen, flag = received(mark)
    batch = np.broadcast_to(np.arange(mark.shape[0] * 2).reshape(*mark.shape[:2], 1, 1, 1), flag.shape)[flag]
    good = risk[flag]
    chosen = chosen[flag]
    loc, weights = np.polynomial.legendre.leggauss(25)
    log_f = .48 * loc[:, None, None] * TYPE + GRID[None, :, None] * SENS
    flat = log_f.reshape(-1, 4)
    values = flat[:, chosen].T - np.log(good @ np.exp(flat).T)
    accumulated = np.zeros((mark.shape[0] * 2, flat.shape[0]))
    np.add.at(accumulated, batch, values)
    integrated = logsumexp(accumulated.reshape(-1, len(loc), len(GRID))
                          + np.log(.5 * weights)[None, :, None], axis=1)
    z = alpha * integrated + np.log(PRIOR)
    return np.exp(z - logsumexp(z, axis=1, keepdims=True))


def crps(p, y):
    linear = (p * np.abs(y[:, None] - GRID)).sum(axis=1)
    return linear - .5 * np.sum((p @ GRID_DISTANCE) * p, axis=-1)


def read_x(role):
    with np.load(ROOT / f"CONTINUUM_{role.upper()}.npz", allow_pickle=False) as b:
        return tuple(b[name] for name in ("marks", "x", "fixed"))


def normalized(x, center, spread):
    return (x.reshape(-1, x.shape[-1]) - center) / spread


def direct(x, center, spread, entry, fit_x, y, dual=None):
    rows = normalized(x, center, spread)
    dist = cdist(rows, fit_x, metric="sqeuclidean") / rows.shape[1]
    gram = np.exp(-entry["gamma"] * dist)
    if entry["variant"] == "knn":
        mass = gram @ y
    else:
        mass = np.maximum(1.e-7, PRIOR + gram @ dual)
    return mass / mass.sum(axis=1, keepdims=True)


def freeze():
    verify()
    assert not (ROOT / "CONTINUUM_HELD.npz").exists()
    file = src.read("CONTINUUM_DEVELOPMENT.json")["files_sha256"]
    for name, digest in file.items():
        assert src.digest(ROOT / name) == digest
    raw, dev = (read_x(role) for role in ("fit", "dev"))
    t = cohort("fit")[1].ravel()
    y = cohort("dev")[1].ravel()
    mass = soft_truth(t)
    numerical, config = {}, {}
    for name, index in (("static_conditional", 2), ("mobile_conditional", 0)):
        alpha = min((.35, .65, 1., 1.5, 2.), key=lambda a: crps(law(dev[index], a), y).mean())
        config[name] = {"alpha": alpha, "index": index, "dev_crps": float(crps(law(dev[index], alpha), y).mean())}
    for name, mode, variant in (("mobile_knn", "events", "knn"), ("mobile_rbf", "events", "rbf"),
                                ("snapshot_rbf", "snapshot", "rbf")):
        train, val = (features(*b[:2], mode) for b in (raw, dev))
        cen, sd = train.mean(axis=(0, 1)), train.std(axis=(0, 1))
        norm = np.maximum(sd, .1 * np.median(sd[sd > 0]))
        fit_x = normalized(train, cen, norm)
        d = cdist(fit_x, fit_x, "sqeuclidean") / fit_x.shape[-1]
        best = None
        for gamma in (.20, .65, 2., 6.):
            k = np.exp(-gamma * d)
            for penalty in ((0.,) if variant == "knn" else (.3, 3., 30.)):
                dual = None if variant == "knn" else solve(k + penalty * np.eye(len(k)), mass - PRIOR, assume_a="pos")
                setting = {"gamma": gamma, "variant": variant, "mode": mode, "ridge": penalty}
                prob = direct(val, cen, norm, setting, fit_x, mass, dual)
                error = float(crps(prob, y).mean())
                if best is None or error < best[0]:
                    best = error, setting, dual
        error, setup, answer = best
        config[name] = {"dev_crps": error, **setup}
        for k, v in {"center": cen, "scale": norm, "support": fit_x, "mass": mass, "dual": answer}.items():
            if v is not None:
                numerical[f"{name}_{k}"] = v
    src.arrays("CONTINUUM_MODELS.npz", **numerical)
    src.write("CONTINUUM_MODEL_FREEZE.json", {"models": config,
               "development": file, "models_sha256": src.digest(ROOT / "CONTINUUM_MODELS.npz"),
               "held_future_existed": False, "source_sha256": src.digest(Path(__file__))})


def witness_forecast():
    verify()
    lock = src.read("CONTINUUM_MODEL_FREEZE.json")
    hold = src.read("CONTINUUM_HELD.json")
    assert lock["models_sha256"] == src.digest(ROOT / "CONTINUUM_MODELS.npz")
    assert hold["model_freeze_sha256"] == src.digest(ROOT / "CONTINUUM_MODEL_FREEZE.json")
    assert hold["files_sha256"]["CONTINUUM_HELD.npz"] == src.digest(ROOT / "CONTINUUM_HELD.npz")
    data = read_x("held")
    n = ROLES["held"]
    probs = {"prior": np.broadcast_to(PRIOR, (n, 2, len(GRID)))}
    with np.load(ROOT / "CONTINUUM_MODELS.npz", allow_pickle=False) as fit:
        for key, m in lock["models"].items():
            if "alpha" in m:
                p = law(data[m["index"]], m["alpha"])
            else:
                def read(field):
                    return fit[f"{key}_{field}"]
                x = features(*data[:2], m["mode"])
                p = direct(x, read("center"), read("scale"), m, read("support"), read("mass"),
                           read("dual") if m["variant"] == "rbf" else None)
            probs[key] = p.reshape(n, 2, -1)
    assert all(np.isfinite(p).all() and np.allclose(p.sum(axis=-1), 1.) for p in probs.values())
    src.arrays("CONTINUUM_PREDICTED.npz", **probs)
    src.write("CONTINUUM_PREDICTION_FREEZE.json", {"prediction_sha256": src.digest(ROOT / "CONTINUUM_PREDICTED.npz"),
              "model_freeze": hold["model_freeze_sha256"], "held_witness_sha256": hold["files_sha256"]})


def c_worker(item):
    from forecast import work
    g, rid, mode, hardware, eta = item
    return work((g, rid, mode, hardware, eta / src.SLOPE))


def rollout(role, workers):
    verify()
    p = src.read("CONTINUUM_PREDICTION_FREEZE.json")
    assert p["prediction_sha256"] == src.digest(ROOT / "CONTINUUM_PREDICTED.npz")
    name = "CONTINUUM_C_" + role.upper() + ".npz"
    if role == "truth":
        freeze = src.read("CONTINUUM_C_FREEZE.json")
        assert freeze["bank_sha256"] == src.digest(ROOT / "CONTINUUM_C_BANK.npz")
        assert freeze["models_sha256"] == p["prediction_sha256"]
    g, slopes, _ = cohort("held")
    ids = addresses(role)
    planned = src.read(str(NATIVE_SETTINGS))["selected"]
    actions = (("schedule", planned[0]), ("pendant", planned[2]))
    values = np.tile(GRID, (len(g), 1)) if role == "bank" else slopes
    jobs = ((z, i, mode, setting, slope) for z, s, rr in zip(g, values, ids, strict=True)
            for mode, setting in actions for slope in s for i in rr)
    bank = []
    with ProcessPoolExecutor(max_workers=workers, initializer=prepare_native) as pool:
        for i, row in enumerate(pool.map(c_worker, jobs, chunksize=18), 1):
            bank.append(row)
            if i % 2048 == 0:
                print("continuum C", role, i, flush=True)
    path = np.asarray(bank).reshape(len(g), 2, values.shape[1], REPS, 6, 10)
    src.arrays(name, path=path, seeds=ids)
    if role == "bank":
        src.write("CONTINUUM_C_FREEZE.json", {"bank_sha256": src.digest(ROOT / name),
                  "models_sha256": p["prediction_sha256"], "registered": src.digest(ROOT / "CONTINUUM_REGISTERED.json")})
    else:
        src.write("CONTINUUM_C_CONFIRM.json", {"truth_sha256": src.digest(ROOT / name),
                  "c_freeze_sha256": src.digest(ROOT / "CONTINUUM_C_FREEZE.json")})


def score():
    verify()
    lock = src.read("CONTINUUM_C_FREEZE.json")
    conf = src.read("CONTINUUM_C_CONFIRM.json")
    assert lock["models_sha256"] == src.digest(ROOT / "CONTINUUM_PREDICTED.npz")
    assert lock["bank_sha256"] == src.digest(ROOT / "CONTINUUM_C_BANK.npz")
    assert conf["truth_sha256"] == src.digest(ROOT / "CONTINUUM_C_TRUTH.npz")
    assert conf["c_freeze_sha256"] == src.digest(ROOT / "CONTINUUM_C_FREEZE.json")
    with np.load(ROOT / "CONTINUUM_PREDICTED.npz", allow_pickle=False) as m:
        prob = np.stack([m[k] for k in METHODS])
    y = cohort("held")[1]
    n = len(y)
    reps = np.random.default_rng(616712).integers(n, size=(16000, n))
    means = prob @ GRID
    pscore = np.stack([crps(p.reshape(n*2, -1), y.ravel()).reshape(n, 2).mean(axis=1) for p in prob])
    axis = {key: {"rmse": float(np.sqrt(np.mean((means[i] - y)**2))),
                  "crps": summary(pscore[i], reps),
                  "crps_vs_prior": summary(pscore[i] - pscore[0], reps),
                  "crps_vs_rbf": summary(pscore[i] - pscore[METHODS.index("mobile_rbf")], reps)}
            for i, key in enumerate(METHODS)}
    with np.load(ROOT / "CONTINUUM_C_BANK.npz", allow_pickle=False) as raw:
        bank = feature(raw["path"])
    with np.load(ROOT / "CONTINUUM_C_TRUTH.npz", allow_pickle=False) as raw:
        truth = feature(raw["path"])
    baseline = np.broadcast_to(np.eye(len(GRID))[len(GRID)//2], (1, n, 2, len(GRID)))
    weight = np.concatenate((baseline, prob, soft_truth(y)[None]), axis=0)
    keys = (*PHYSICAL, "known_continuous_rule_interpolated")
    result = np.zeros((len(weight), n, 2, 2))
    offdiag = ~np.eye(REPS, dtype=bool)
    for card in range(n):
        for process in range(2):
            b = bank[card, process]
            inner = np.array([[cdist(a, v)[offdiag].mean() for v in b] for a in b])
            for direction in range(2):
                by = np.array([cdist(truth[card, process, direction], a).mean() for a in b])
                w = weight[:, card, direction]
                result[:, card, process, direction] = w @ by - .5 * np.sum((w @ inner) * w, axis=1)
    energy = result.mean(axis=(2, 3))
    paths = {key: {"value": summary(energy[i], reps),
                  "vs_four": summary(energy[i] - energy[0], reps),
                  "vs_direct_rbf": summary(energy[i] - energy[keys.index("mobile_rbf")], reps)}
             for i, key in enumerate(keys)}
    src.write("CONTINUUM_RESULT.json", {"C_freeze": conf["c_freeze_sha256"],
              "score_by_rule": axis, "energy_by_forecaster": paths,
              "C_forecast_paths": int(n*2*len(GRID)*REPS), "C_truth_paths": int(n*4*REPS)})


if __name__ == "__main__":
    cl = argparse.ArgumentParser(description=__doc__)
    cl.add_argument("stage", choices=("register", "dev", "freeze", "held", "predict", "bank", "truth", "score"))
    cl.add_argument("--workers", type=int, default=8)
    options = cl.parse_args()
    if options.stage in ("dev", "held"):
        acquire(options.stage, options.workers)
    elif options.stage in ("bank", "truth"):
        rollout(options.stage, options.workers)
    else:
        {"register": register, "freeze": freeze, "predict": witness_forecast, "score": score}[options.stage]()

