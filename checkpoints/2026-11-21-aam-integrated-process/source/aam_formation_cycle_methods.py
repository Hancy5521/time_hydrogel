"""Fixed formation-cycle goals, same-information baselines and native scoring.

SOURCE ONLY until the integrating registration authorizes an actual execution.
Imports never fit, simulate, generate random numbers or write files.  The fixed
Work goals below are not selected using held outcomes.  The full 114-vector is
cutoff six FIRST, followed by action-major/time-major/channel-major futures.
Channels are qL,qR,conversion,G(.1),G(.5),G(2).  A shared cutoff is predicted,
not supplied as an extra query observation: EVERY baseline sees X12 only.

One normalization rule: for EACH mechanical lag, RMS of terminal-age G over
the available training mothers, all six actions and all suffixes.  OOF fits use
their own training-fold RMS; final scoring uses the full-training RMS.  A truly
zero RMS uses numerical unit 1 and is explicitly flagged degenerate; this
completes the pipeline without claiming that normalized target is meaningful.
No epsilon/floor or held-label normalization is used.

Forest: four mother-level OOF fits plus one final fit, each 128 ExtraTrees,
min_samples_leaf=4, no hyperparameter/epoch selection.  Each target is a mother
mean; EVERY individual suffix minus that mother's OOF prediction is retained
as a JOINT 114 residual.  Final mean plus all training residuals is the native
predictive support.  It is a numerical baseline, not a legal microscopic graph.

Its sole registered physical-support projection (never chosen against raw):
q values clip [0,1]; common cutoff conversion clips [0,1] and remains fixed;
each action's three future conversions undergo equal-weight increasing PAVA,
then restrict to [common cutoff,1]; each age's three G values undergo equal-
weight decreasing PAVA, then nonnegative projection.  Raw atoms/weights and
violation/displacement diagnostics are retained; no atom is deleted/reweighted.
There is no monotonicity imposed across formation ages on G or q.
"""
from __future__ import annotations

import numpy as np


SCHEMA = "aam-formation-cycle-methods-v1"
CHANNELS = ("q_left", "q_right", "conversion", "G_tau_0p1", "G_tau_0p5", "G_tau_2")
GOALS = ("delayed_support", "early_buffer_then_relax")
METHODS = ("best_fixed", "knn8", "extra_trees_128_leaf4")
ACTIONS, AGES, CHANNEL_COUNT, DIMENSION = 6, 3, 6, 114
DELAY_G = np.array((.15, .45, .80))
DELAY_Q = np.array((.20, .40, .60))
BUFFER_G = np.array((.65, .80, .80))
BUFFER_Q = np.array((.40, .55, .60))
RETENTION_LIMIT, MIN_CONVERSION = .25, .82


def _finite(value, name):
    a = np.asarray(value, dtype=np.float64)
    if not np.isfinite(a).all():
        raise ValueError(name+" contains an invalid value; mothers/atoms are not dropped.")
    return a


def _future(value):
    a = _finite(value, "future")
    if a.ndim < 4 or a.shape[-3:] != (ACTIONS, AGES, CHANNEL_COUNT):
        raise ValueError("Future layout must end in (action6,age3,channel6).")
    return a


def pack_targets(cutoff, future):
    """N mothers, R suffixes -> N,R,114, shared cutoff repeated per suffix."""
    f, c = _future(future), _finite(cutoff, "cutoff")
    if f.ndim != 5 or c.shape != (len(f), CHANNEL_COUNT) or f.shape[1] == 0:
        raise ValueError("Training cutoff(N,6) and future(N,R,6,3,6) are required.")
    return np.concatenate((np.repeat(c[:, None], f.shape[1], axis=1),
                           f.reshape(len(f), f.shape[1], -1)), axis=-1)


def unpack_targets(targets):
    a = _finite(targets, "targets")
    if a.ndim < 2 or a.shape[-1] != DIMENSION:
        raise ValueError("Targets must have an explicit atom axis and final dimension 114.")
    return a[..., :6], a[..., 6:].reshape(*a.shape[:-1], ACTIONS, AGES, CHANNEL_COUNT)


def training_normalization(future):
    """Same declared rule on any training subset; no held reference access."""
    f = _future(future)
    if f.ndim != 5 or not len(f) or not f.shape[1]:
        raise ValueError("Normalization needs nonempty training mothers and suffixes.")
    rms = np.sqrt(np.mean(np.square(f[:, :, :, -1, 3:6]), axis=(0, 1, 2)))
    if not np.isfinite(rms).all():
        raise ArithmeticError("Mechanical normalization overflowed.")
    degenerate = rms == 0.
    scale = rms.copy(); scale[degenerate] = 1.
    channels = np.r_[np.ones(3), scale]
    return {"schema": SCHEMA, "mechanical_rms": rms, "mechanical_scale": scale,
            "degenerate_mechanical_lags": degenerate,
            "normalized_target_meaningful_by_lag": ~degenerate,
            "channel_scale": channels,
            "target114_scale": np.r_[channels, np.tile(channels, ACTIONS*AGES)],
            "rule": "training_terminal_age_RMS_over_mothers_actions_suffixes;exact_zero_uses_unit1_with_flag",
            "training_mothers": len(f), "suffixes_per_mother": f.shape[1]}


def _pava(values, increasing=True):
    """Equal-weight L2 isotonic projection, deterministic original order."""
    x = np.asarray(values, dtype=np.float64)
    sign = 1. if increasing else -1.
    means, counts = [], []
    for value in sign*x:
        means.append(float(value)); counts.append(1)
        while len(means) >= 2 and means[-2] > means[-1]:
            total = counts[-2]+counts[-1]
            mean = (means[-2]*counts[-2]+means[-1]*counts[-1])/total
            means[-2:] = [mean]; counts[-2:] = [total]
    return sign*np.repeat(np.asarray(means), counts)


def project_forest_support(raw_targets):
    """Fixed numerical support projection; preserve all raw atoms externally."""
    raw = _finite(raw_targets, "raw_targets")
    unpack_targets(raw)
    out = raw.copy().reshape(-1, DIMENSION)
    raw_flat = raw.reshape(-1, DIMENSION)
    for row in out:
        c = row[:6]
        f = row[6:].reshape(ACTIONS, AGES, CHANNEL_COUNT)
        c[:2] = np.clip(c[:2], 0., 1.)
        c[2] = np.clip(c[2], 0., 1.)
        c[3:] = np.maximum(_pava(c[3:], increasing=False), 0.)
        for action in range(ACTIONS):
            f[action, :, :2] = np.clip(f[action, :, :2], 0., 1.)
            f[action, :, 2] = np.clip(_pava(f[action, :, 2]), c[2], 1.)
            for age in range(AGES):
                f[action, age, 3:] = np.maximum(_pava(f[action, age, 3:], increasing=False), 0.)
    difference = out-raw_flat
    changed = np.any(difference != 0., axis=1)
    raw_c = raw_flat[:, :6]
    raw_f = raw_flat[:, 6:].reshape(-1, ACTIONS, AGES, CHANNEL_COUNT)
    qcv = np.concatenate((raw_c[:, :3].reshape(-1), raw_f[..., :3].reshape(-1)))
    g = np.concatenate((raw_c[:, 3:].reshape(-1, 3), raw_f[..., 3:].reshape(-1, 3)))
    conversion_path = np.concatenate((np.repeat(raw_c[:, None, 2:3], ACTIONS, axis=1), raw_f[..., 2]), axis=2)
    return out.reshape(raw.shape), {
        "rule": "fixed_common_cutoff_bounded_future_PAVA_and_per_age_nonnegative_decreasing_G_PAVA",
        "raw_atoms": len(out), "retained_atoms": len(out), "changed_atoms": int(changed.sum()),
        "raw_q_or_conversion_outside_unit_fraction": float(np.mean((qcv < 0.) | (qcv > 1.))),
        "raw_negative_G_fraction": float(np.mean(g < 0.)),
        "raw_G_tau_order_violation_fraction": float(np.mean(np.any(np.diff(g, axis=1) > 0., axis=1))),
        "raw_conversion_history_decrease_fraction": float(np.mean(np.any(np.diff(conversion_path, axis=2) < 0., axis=2))),
        "root_mean_square_raw_displacement": float(np.sqrt(np.mean(difference**2))),
        "maximum_absolute_raw_displacement": float(np.max(np.abs(difference))),
        "weights_changed": False, "atoms_dropped": 0,
        "legal_microscopic_graph_claimed": False}


def process_goal_losses(future, normalization):
    """Every suffix/action receives both complete losses, before expectation.

    Three equal blocks: network trajectory MSE, mechanical trajectory penalty,
    terminal squared conversion deficit.  Buffer's mechanical block is the
    equal average of G(.1) trajectory MSE and squared retention-ratio hinge.
    The valid 0/0 zero-response limit is defined as ratio=0; the positive G(.1)
    target still penalizes an entirely stress-free sample.  Positive G(2)/0,
    negative G, or nonfinite values are invalid, never floored or discarded.
    """
    f = _future(future)
    scale = _finite(normalization["mechanical_scale"], "mechanical_scale")
    if scale.shape != (3,) or np.any(scale <= 0.):
        raise ValueError("Frozen positive three-lag normalization is required.")
    g = f[..., 3:6]
    if np.any(g < 0.):
        raise ValueError("Goal scoring needs actual nonnegative G or the fixed forest projection.")
    first, late = g[..., 0], g[..., 2]
    zero = first == 0.
    if np.any(zero & (late != 0.)):
        raise ValueError("A nonzero late response with zero short response has no valid retention ratio.")
    ratio = np.zeros_like(first)
    np.divide(late, first, out=ratio, where=~zero)
    q = .5*(f[..., 0]+f[..., 1])
    network = np.stack((np.mean((q-DELAY_Q)**2, axis=-1),
                        np.mean((q-BUFFER_Q)**2, axis=-1)), axis=-1)
    delayed = np.mean((g[..., 1]/scale[1]-DELAY_G)**2, axis=-1)
    buffered = .5*(np.mean((first/scale[0]-BUFFER_G)**2, axis=-1)
                    + np.mean(np.maximum(ratio-RETENTION_LIMIT, 0.)**2, axis=-1))
    mechanical = np.stack((delayed, buffered), axis=-1)
    deficit = np.maximum(MIN_CONVERSION-f[..., -1, 2], 0.)**2
    conversion = np.repeat(deficit[..., None], 2, axis=-1)
    blocks = np.stack((network, mechanical, conversion), axis=-1)
    return {"loss": blocks.mean(axis=-1), "blocks": blocks,
            "block_names": ("network", "mechanical", "terminal_conversion_deficit"),
            "goal_names": GOALS, "zero_over_zero_ratio_count": int(np.count_nonzero(zero)),
            "normalized_mechanical_lags_meaningful": np.asarray(normalization["normalized_target_meaningful_by_lag"]).copy()}


def select_actions(target_draws, weights, normalization, action_support):
    """All native atoms -> expected goal loss -> stable original action index."""
    from aam_formation_process_objective import stable_argmin
    _, future = unpack_targets(target_draws)
    if future.ndim != 4 or not len(future):
        raise ValueError("Use one query's complete native predictive atoms (draw,114).")
    w = _finite(weights, "weights")
    if w.shape != (len(future),) or np.any(w < 0.) or w.sum() <= 0.:
        raise ValueError("Every native draw needs its nonnegative, nonzero-total atom weight.")
    w = w/w.sum()
    values = process_goal_losses(future, normalization)
    expected = np.einsum("d,dag->ga", w, values["loss"])
    choices = np.array([stable_argmin(row, action_support) for row in expected], dtype=np.int64)
    return {"choices": choices, "expected_loss": expected,
            "per_draw_action_goal_loss": values["loss"], "weights": w,
            "all_atoms_retained": len(w), "goal_names": GOALS}


def fit_methods(X12, cutoff, future, *, mother_ids, random_state, action_support):
    """Exactly five necessary fixed forest fits: four OOF and one final.

    No held inputs or outcomes accepted by this API.  X contains no privileged
    cutoff property.  Fold membership is integer training-mother ID modulo
    four, saved with its explicit unique mother IDs.  All 128 mother/suffix
    residuals remain joint and are later projected by the one fixed rule.
    """
    from sklearn.ensemble import ExtraTreesRegressor
    X = _finite(X12, "X12")
    f = _future(future)
    ids = np.asarray(mother_ids)
    if (X.shape != (64, 12) or f.shape != (64, 2, ACTIONS, AGES, CHANNEL_COUNT)
            or ids.shape != (64,) or ids.dtype.kind not in "iu" or len(np.unique(ids)) != 64):
        raise ValueError("This fixed version requires 64 unique mothers, X12 and two suffixes.")
    if (isinstance(random_state, (bool, np.bool_)) or not isinstance(random_state, (int, np.integer))
            or not 0 <= random_state <= 2**32-5):
        raise ValueError("Register an integer random_state allowing four subsequent fold seeds.")
    target = pack_targets(cutoff, f)
    normalization = training_normalization(f)
    target_mean = target.mean(axis=1)
    fold = ids % 4
    if not np.array_equal(np.bincount(fold.astype(np.int64), minlength=4), np.full(4, 16)):
        raise ValueError("The registered ID-modulo-four split must have 16 mothers in every fold.")
    oof = np.empty_like(target_mean)
    fold_normalizations, oof_forests = [], []
    for k in range(4):
        train = fold != k
        fold_norm = training_normalization(f[train])
        fold_normalizations.append(fold_norm)
        model = ExtraTreesRegressor(n_estimators=128, min_samples_leaf=4,
                                    random_state=int(random_state)+k, n_jobs=1)
        model.fit(X[train], target_mean[train]/fold_norm["target114_scale"])
        oof[~train] = model.predict(X[~train])*fold_norm["target114_scale"]
        oof_forests.append(model)
    residual = target-oof[:, None, :]
    model = ExtraTreesRegressor(n_estimators=128, min_samples_leaf=4,
                                random_state=int(random_state)+4, n_jobs=1)
    model.fit(X, target_mean/normalization["target114_scale"])
    mean = X.mean(axis=0); std = X.std(axis=0)
    constant = std == 0.
    std = std.copy(); std[constant] = 1.
    fixed = select_actions(target.reshape(-1, DIMENSION), np.ones(128), normalization, action_support)
    return {"schema": SCHEMA, "X12": X.copy(), "training_ids": ids.copy(),
            "training_targets": target.copy(), "training_mother_means": target_mean,
            "normalization": normalization, "fold_normalizations": fold_normalizations,
            "fold": fold, "oof_prediction": oof, "joint_suffix_residual": residual,
            "knn_Xmean": mean, "knn_Xscale": std, "constant_X_features": constant,
            "final_forest": model, "oof_forests": oof_forests,
            "forest_seed_base": int(random_state),
            "forest_seeds_oof_then_final": np.arange(int(random_state), int(random_state)+5, dtype=np.int64),
            "forest_fits": 5,
            "fixed_choices": fixed["choices"], "fixed_expected_loss": fixed["expected_loss"],
            "action_support": np.array(action_support, copy=True), "target_layout": "cutoff6_then_action_time_channel108",
            "held_outcomes_accepted": False, "hyperparameter_searches": 0}


def predict_methods(fitted, X12_query):
    """Predict/freeze from X12 only, before any held cutoff/future is read."""
    X = _finite(X12_query, "X12_query")
    if X.ndim != 2 or X.shape[1] != 12 or not len(X):
        raise ValueError("Queries must have nonempty shape (query,12).")
    training = fitted["training_targets"]
    residual = fitted["joint_suffix_residual"].reshape(-1, DIMENSION)
    trainx = (fitted["X12"]-fitted["knn_Xmean"])/fitted["knn_Xscale"]
    queryx = (X-fitted["knn_Xmean"])/fitted["knn_Xscale"]
    distance = np.sum((queryx[:, None]-trainx[None])**2, axis=2)
    neighbors = np.argsort(distance, axis=1, kind="stable")[:, :8]
    knn = training[neighbors].reshape(len(X), 16, DIMENSION)
    mean = fitted["final_forest"].predict(X)*fitted["normalization"]["target114_scale"]
    forest_raw = mean[:, None]+residual[None]
    forest, projection = project_forest_support(forest_raw)
    prior = np.repeat(training.reshape(1, 128, DIMENSION), len(X), axis=0)
    output = {}
    for method, atoms in zip(METHODS, (prior, knn, forest), strict=True):
        weight = np.full((len(X), atoms.shape[1]), 1./atoms.shape[1])
        selections = [select_actions(p, w, fitted["normalization"], fitted["action_support"])
                      for p, w in zip(atoms, weight, strict=True)]
        choices = np.stack([v["choices"] for v in selections])
        if method == "best_fixed":
            choices[:] = fitted["fixed_choices"]
        output[method] = {"targets114": atoms.copy(), "weights": weight,
                          "choices": choices,
                          "expected_loss": np.stack([v["expected_loss"] for v in selections]),
                          "native_atoms_per_query": atoms.shape[1],
                          "forecast_role": ("unconditioned_training_prior_for_fixed_design_descriptive_only"
                                            if method == "best_fixed" else "same_X12_conditional_baseline")}
    output["knn8"]["neighbor_parent_indices"] = neighbors
    output["extra_trees_128_leaf4"].update(raw_targets114=forest_raw, raw_mean114=mean,
                                           projection_diagnostic=projection,
                                           residual_parent_index=np.repeat(np.arange(64), 2),
                                           residual_suffix_index=np.tile(np.arange(2), 64))
    return output


def score_forecast(target_draws, weights, *, reference_future, reference_cutoff,
                   normalization):
    """Mean of six separate 18D normalized energy scores, NOT a joint 108D ES."""
    from aam_formation_process_objective import per_action_energy
    cutoff, future = unpack_targets(target_draws)
    ref = _future(reference_future)
    if future.ndim != 4 or ref.ndim != 4:
        raise ValueError("One-query forecast draws and independent reference suffixes are required.")
    channel = _finite(normalization["channel_scale"], "channel_scale")
    result = per_action_energy(future, ref, weights,
                               scales=np.repeat(channel[None], AGES, axis=0),
                               point_weights=np.ones((AGES, CHANNEL_COUNT)),
                               reference_weights=np.ones(len(ref)))
    c = _finite(reference_cutoff, "reference_cutoff")
    if c.shape != (6,):
        raise ValueError("One actual cutoff6 diagnostic is required.")
    current = per_action_energy(cutoff[:, None, None], c[None, None, None], weights,
                                scales=channel[None], point_weights=np.ones((1, 6)),
                                reference_weights=np.ones(1))
    return {"primary_future_energy": float(result["per_action_energy_score"].mean()),
            "per_action_future_energy": result["per_action_energy_score"],
            "cutoff_energy_diagnostic": float(current["per_action_energy_score"][0]),
            "per_action_18d_energy": result["per_action_energy_score"].copy(),
            "average_action_18d_energy": float(result["per_action_energy_score"].mean()),
            "hidden6_energy": float(current["per_action_energy_score"][0]),
            "native_draws": len(future), "reference_suffixes": len(ref),
            "counterfactual_actions_scored_jointly": False}


def evaluate_choices(choices, reference_future, normalization):
    """All held suffixes retained; finite-reference hindsight is descriptive."""
    c = np.asarray(choices)
    f = _future(reference_future)
    if c.shape != (2,) or c.dtype.kind not in "iu" or np.any(c < 0) or np.any(c >= ACTIONS) or f.ndim != 4:
        raise ValueError("Two valid chosen indices and all reference suffixes are required.")
    value = process_goal_losses(f, normalization)
    mean = value["loss"].mean(axis=0).T
    chosen = mean[np.arange(2), c]
    return {"selected_goal_loss": chosen, "all_action_goal_loss": mean,
            "finite_reference_best_goal_loss": mean.min(axis=1),
            "finite_reference_hindsight_gap": chosen-mean.min(axis=1),
            "selected_suffix_goal_loss": np.stack([value["loss"][:, c[g], g] for g in range(2)], axis=1),
            "finite_reference_is_exact_conditional_oracle": False,
            "reference_suffixes_retained": len(f)}


def targets_from_training(Y114):
    """Driver adapter: derive the one final normalization from training Y only."""
    y = _finite(Y114, "training Y114")
    if y.shape != (64, 2, DIMENSION):
        raise ValueError("This version requires Y114[64,2,114].")
    cutoff, future = unpack_targets(y)
    if not np.array_equal(cutoff[:, 0], cutoff[:, 1]):
        raise ValueError("The two suffixes must share the same actual cutoff6.")
    return training_normalization(future)


def save_baselines(fitted, output):
    """Save all five models, raw residuals, means and frozen metadata exclusively.

    The integrating driver must register its inputs/sources BEFORE fitting.
    joblib is an internal trusted-local model archive, never an untrusted-input
    interchange format.  The independent NPZ contains no pickled objects.
    """
    from pathlib import Path
    from datetime import datetime, timezone
    import hashlib
    import json
    import joblib
    folder = Path(output)
    folder.mkdir(parents=True, exist_ok=False)
    model_path, array_path = folder/"FITTED_MODELS.joblib", folder/"FIT_ARRAYS.npz"
    joblib.dump(fitted, model_path, compress=3)
    names = ("X12", "training_ids", "training_targets", "training_mother_means", "fold",
             "oof_prediction", "joint_suffix_residual", "knn_Xmean", "knn_Xscale",
             "constant_X_features", "fixed_choices", "fixed_expected_loss", "action_support",
             "forest_seeds_oof_then_final")
    arrays = {name: np.asarray(fitted[name]) for name in names}
    arrays.update({"normalization_"+key: value for key, value in fitted["normalization"].items()
                   if isinstance(value, np.ndarray)})
    for index, fold in enumerate(fitted["fold_normalizations"]):
        arrays.update({f"fold{index}_normalization_"+key: value for key, value in fold.items()
                       if isinstance(value, np.ndarray)})
    with array_path.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    info = {"schema": SCHEMA, "actual_utc": datetime.now(timezone.utc).isoformat(),
            "models_file": model_path.name,
            "models_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
            "arrays_file": array_path.name,
            "arrays_sha256": hashlib.sha256(array_path.read_bytes()).hexdigest(),
            "forest_fits": 5, "hyperparameter_searches": 0,
            "forest_seeds_oof_then_final": fitted["forest_seeds_oof_then_final"].tolist(),
            "training_mothers": 64, "training_suffixes_per_mother": 2,
            "fold_rule": "integer_training_mother_id_modulo_four",
            "methods": METHODS, "goal_names": GOALS, "channels": CHANNELS,
            "target114_layout": "cutoff6_FIRST_then_action6_time3_channel6",
            "goal_targets": {"delayed_G05": DELAY_G.tolist(), "delayed_qmean": DELAY_Q.tolist(),
                             "buffer_G01": BUFFER_G.tolist(), "buffer_qmean": BUFFER_Q.tolist(),
                             "retention_upper_hinge": RETENTION_LIMIT,
                             "terminal_conversion_lower_hinge": MIN_CONVERSION,
                             "three_blocks_equal_weight": True},
            "mechanical_rms": fitted["normalization"]["mechanical_rms"].tolist(),
            "mechanical_scale": fitted["normalization"]["mechanical_scale"].tolist(),
            "degenerate_mechanical_lags": fitted["normalization"]["degenerate_mechanical_lags"].tolist(),
            "raw_joint_suffix_residual_atoms": 128,
            "held_inputs_or_outcomes_used": False,
            "projection": "fixed_common_cutoff_bounded_future_PAVA_and_per_age_nonnegative_decreasing_G_PAVA",
            "raw_vs_projected_model_selection": False}
    with (folder/"FROZEN.json").open("x", encoding="utf8") as stream:
        json.dump(info, stream, ensure_ascii=False, indent=2, allow_nan=False)
    return info


def load_baselines(output):
    """Load a trusted self-generated model only after its frozen SHA check."""
    from pathlib import Path
    import hashlib
    import json
    import joblib
    folder = Path(output)
    info = json.loads((folder/"FROZEN.json").read_text(encoding="utf8"))
    if info["schema"] != SCHEMA:
        raise ValueError("Unknown frozen baseline schema.")
    for key in ("models", "arrays"):
        path = folder/info[key+"_file"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != info[key+"_sha256"]:
            raise ValueError("Frozen baseline archive SHA mismatch: "+key)
    return joblib.load(folder/info["models_file"])


def save_fit(path, fitted):
    """Root-driver spelling; argument order is path, then fitted state."""
    return save_baselines(fitted, path)


def load_fit(path):
    """Root-driver spelling; trusted-local archive with frozen SHA verification."""
    return load_baselines(path)


def fit_baselines(X12, Y114, output, seedbase, *, mother_ids=None, action_support=None):
    """Preferred driver API, X[64,12], Y[64,2,114], registered seed base.

    seedbase=1958000000 in the root's declared namespace: OOF seeds base+0..3,
    final base+4.  The caller registers this number; the function never chooses
    or searches a random seed.  output=None explicitly skips persistence.
    """
    y = _finite(Y114, "training Y114")
    targets_from_training(y)
    cutoff, future = unpack_targets(y)
    ids = np.arange(64, dtype=np.int64) if mother_ids is None else mother_ids
    support = np.ones(ACTIONS, dtype=bool) if action_support is None else action_support
    fitted = fit_methods(X12, cutoff[:, 0], future, mother_ids=ids,
                         random_state=seedbase, action_support=support)
    if output is not None:
        save_baselines(fitted, output)
    return fitted


def predict_baselines(Xquery, fitted):
    """Preferred driver API; one X12 vector or batch.  Never receives held Y.

    For one vector, each method has draws[D,114], weights[D], choices[2],
    expected_loss[2,6].  A batch retains the leading query axis.
    Forest additionally preserves raw_draws and all residual donor identities.
    """
    x = _finite(Xquery, "Xquery")
    single = x.ndim == 1
    if single:
        x = x[None]
    result = predict_methods(fitted, x)
    query_fields = ("targets114", "weights", "choices", "expected_loss", "raw_targets114",
                    "raw_mean114", "neighbor_parent_indices")
    for values in result.values():
        if single:
            for key in query_fields:
                if key in values:
                    values[key] = values[key][0]
        values["draws"] = values["targets114"]
        if "raw_targets114" in values:
            values["raw_draws"] = values["raw_targets114"]
    return result


def expected_goals(draws, weights, scales, *, action_support=None):
    """Native and baseline common goal API; returns loss[action6,goal2]."""
    support = np.ones(ACTIONS, dtype=bool) if action_support is None else action_support
    value = select_actions(draws, weights, scales, support)
    return {"loss": value["expected_loss"].T, "choices": value["choices"],
            "per_draw_action_goal_loss": value["per_draw_action_goal_loss"],
            "weights": value["weights"], "all_atoms_retained": value["all_atoms_retained"]}


def deterministic_cycle_method_checks():
    """Definition only; no forest fit/RNG/native.  Later integrated registration.

    Tests layout, RMS and all-zero flag, unchanged common cutoff, PAVA analytic
    examples, all-atom support/weight preservation, zero-response ratio, opposite
    goals selecting distinct fixed candidates, score-action separation and input
    nonmutation.  Actual sklearn OOF fitting is NOT claimed by this fixture.
    """
    checks = 0
    maximum = 0.
    def close(a, b):
        nonlocal checks, maximum
        x, y = np.asarray(a), np.asarray(b)
        if x.shape != y.shape or not np.isfinite(x).all():
            raise AssertionError("Cycle fixture shape/finiteness failure.")
        e = float(np.max(np.abs(x-y))) if x.size else 0.
        checks += 1; maximum = max(maximum, e)
        if e > 3.e-12:
            raise AssertionError("Cycle fixture algebra error: "+str(e))
    close(_pava([3., 1., 2.]), [2., 2., 2.])
    close(_pava([1., 3., 2.], increasing=False), [2., 2., 2.])
    f = np.zeros((2, 2, ACTIONS, AGES, CHANNEL_COUNT))
    f[..., 2] = .9; f[..., 3:6] = np.array([2., 1., .5])
    c = np.tile(np.array([.1, .2, .3, 2., 1., .5]), (2, 1))
    p = pack_targets(c, f); p_before = p.tobytes()
    cc, ff = unpack_targets(p)
    close(cc, np.repeat(c[:, None], 2, axis=1)); close(ff, f)
    norm = training_normalization(f); close(norm["mechanical_rms"], [2., 1., .5])
    zero = training_normalization(np.zeros_like(f)); close(zero["mechanical_scale"], [1., 1., 1.])
    if not zero["degenerate_mechanical_lags"].all():
        raise AssertionError("Zero-scale degeneracy was concealed.")
    malformed = p.reshape(-1, DIMENSION).copy()
    malformed[:, 2] = .7
    malformed[:, 3:6] = [1., 3., -1.]
    mf = malformed[:, 6:].reshape(-1, ACTIONS, AGES, CHANNEL_COUNT)
    mf[..., 2] = [.9, .1, .8]
    fixed, diag = project_forest_support(malformed)
    fc, fp = unpack_targets(fixed)
    close(fc[:, 2], np.full(len(fc), .7))
    close(fp[..., 2], np.broadcast_to([.7, .7, .8], fp[..., 2].shape))
    close(fc[:, 3:6], np.tile([2., 2., 0.], (len(fc), 1)))
    if diag["atoms_dropped"] or diag["weights_changed"] or len(fixed) != len(malformed):
        raise AssertionError("Fixed numerical support projection lost native support.")
    zeros = process_goal_losses(np.zeros((1, ACTIONS, AGES, CHANNEL_COUNT)), zero)
    if zeros["zero_over_zero_ratio_count"] != ACTIONS*AGES:
        raise AssertionError("True zero response was mistaken for invalid mechanics.")
    # Two literal target-like processes: no fit and no synthetic sampled data.
    pathways = np.zeros((1, ACTIONS, AGES, CHANNEL_COUNT))
    pathways[..., 2] = 1.
    pathways[0, 0, :, :2] = DELAY_Q[:, None]
    pathways[0, 0, :, 3] = DELAY_G
    pathways[0, 0, :, 4] = DELAY_G
    pathways[0, 1, :, :2] = BUFFER_Q[:, None]
    pathways[0, 1, :, 3] = BUFFER_G
    pathways[0, 1, :, 4] = .2*BUFFER_G
    pathways[0, 1, :, 5] = .1*BUFFER_G
    targets = np.concatenate((np.zeros((1, 6)), pathways.reshape(1, -1)), axis=1)
    unit = {"mechanical_scale": np.ones(3), "normalized_target_meaningful_by_lag": np.ones(3, bool)}
    selection = select_actions(targets, np.ones(1), unit, np.array([True, True, False, False, False, False]))
    close(selection["choices"], [0, 1])
    if p.tobytes() != p_before:
        raise AssertionError("Cycle methods mutated original training targets.")
    return {"passed": True, "checks": checks, "maximum_absolute_error": maximum,
            "scope": "Deterministic algebra/API only; no forest-fit, simulation or performance evidence.",
            "native_calls": 0, "fits": 0, "random_draws": 0}
