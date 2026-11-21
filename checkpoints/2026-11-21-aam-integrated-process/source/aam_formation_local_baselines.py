"""V5 new-mu5 training with an externally frozen V4 metric normalization.

Import and input preparation do not fit, initialize particles, simulate, draw
randomness or write files. Only fit_methods_fixed_normalization fits models:
four original mother-level OOF forests and one final forest, after the caller's
actual registration. No V1--V4 source is changed or monkey-patched.

The caller must bind fixed_normalization to the exact historical metric inputs
before fitting. Validation here checks internal numeric consistency, not the
external file identity; ROOT's registration and fixed-scale receipt own that.
"""
from __future__ import annotations

from copy import deepcopy

import numpy as np

from aam_formation_cycle_methods import (
    SCHEMA, ACTIONS, AGES, CHANNEL_COUNT, DIMENSION,
    _finite, _future, pack_targets, select_actions,
)


FIXED_METRIC_NAMESPACE = "aam-formation-whole-path-tempering-v4-20261119"
TRAINING_PRIOR = "mu5"


def validate_fixed_normalization(fixed_normalization):
    """Copy/check the frozen metric without estimating a scale from new data.

    Required arrays are the original normalization fields persisted by
    save_baselines. An exact-zero historical RMS uses unit one and its explicit
    flag, as in the frozen method. Additional caller provenance is preserved.
    """
    if not isinstance(fixed_normalization, dict):
        raise TypeError("Pass the externally bound fixed normalization dict.")
    required = (
        "mechanical_rms", "mechanical_scale", "degenerate_mechanical_lags",
        "normalized_target_meaningful_by_lag", "channel_scale", "target114_scale",
    )
    missing = [name for name in required if name not in fixed_normalization]
    if missing:
        raise ValueError("Incomplete fixed normalization: " + ",".join(missing))
    result = deepcopy(fixed_normalization)
    if "schema" in result and result["schema"] != SCHEMA:
        raise ValueError("Fixed metric has another methods schema.")
    result["schema"] = SCHEMA
    for name, shape in (
        ("mechanical_rms", (3,)), ("mechanical_scale", (3,)),
        ("channel_scale", (6,)), ("target114_scale", (DIMENSION,)),
    ):
        value = _finite(result[name], "fixed " + name)
        if value.shape != shape:
            raise ValueError("Unexpected fixed normalization shape: " + name)
        result[name] = value.copy()
    rms, scale = result["mechanical_rms"], result["mechanical_scale"]
    if np.any(rms < 0.) or np.any(scale <= 0.):
        raise ValueError("Fixed historical RMS must be nonnegative and scale positive.")
    for name in ("degenerate_mechanical_lags", "normalized_target_meaningful_by_lag"):
        value = np.asarray(result[name])
        if value.shape != (3,) or value.dtype.kind != "b":
            raise ValueError("Fixed normalization validity flags must be bool[3].")
        result[name] = value.copy()
    degenerate = rms == 0.
    expected_scale = np.where(degenerate, 1., rms)
    if (not np.array_equal(result["degenerate_mechanical_lags"], degenerate)
            or not np.array_equal(result["normalized_target_meaningful_by_lag"], ~degenerate)
            or not np.array_equal(scale, expected_scale)):
        raise ValueError("Historical RMS/scale/zero flags disagree.")
    channel = np.r_[np.ones(3), scale]
    target = np.r_[channel, np.tile(channel, ACTIONS*AGES)]
    if (not np.array_equal(result["channel_scale"], channel)
            or not np.array_equal(result["target114_scale"], target)):
        raise ValueError("Fixed channel and target scales must use the original114 layout.")
    return result


def prepare_training_inputs(X12, cutoff, future, *, mother_ids, random_state,
                            action_support, fixed_normalization):
    """Pure array preparation, exposed for registered fixed-algebra checks.

    No model object is constructed and no RNG is instantiated. This preserves
    the old 64-mother/two-suffix/four-fold training layout while replacing every
    scale estimate by a pre-existing externally frozen metric.
    """
    X, f = _finite(X12, "X12"), _future(future)
    ids = np.asarray(mother_ids)
    if (X.shape != (64, 12) or f.shape != (64, 2, ACTIONS, AGES, CHANNEL_COUNT)
            or ids.shape != (64,) or ids.dtype.kind not in "iu"
            or len(np.unique(ids)) != 64):
        raise ValueError("V5 requires64unique new-mu5 mothers,X12 and two suffixes.")
    if (isinstance(random_state, (bool, np.bool_))
            or not isinstance(random_state, (int, np.integer))
            or not 0 <= random_state <= 2**32-5):
        raise ValueError("Register one integer forest seed base allowing five seeds.")
    support = np.asarray(action_support)
    if support.shape != (ACTIONS,) or support.dtype.kind != "b" or not support.any():
        raise ValueError("Use the registered nonempty bool[6] action support.")
    normalization = validate_fixed_normalization(fixed_normalization)
    target = pack_targets(cutoff, f)
    target_mean = _finite(target.mean(axis=1), "training mother means")
    fold = ids % 4
    if not np.array_equal(np.bincount(fold.astype(np.int64), minlength=4), np.full(4, 16)):
        raise ValueError("The original ID-modulo-four split needs16mothers in every fold.")
    mean, std = _finite(X.mean(axis=0), "KNN mean"), _finite(X.std(axis=0), "KNN scale")
    constant = std == 0.
    std = std.copy()
    std[constant] = 1.
    return {
        "X12": X.copy(), "future": f.copy(), "training_ids": ids.copy(),
        "training_targets": target.copy(), "training_mother_means": target_mean,
        "normalization": normalization, "fold": fold.copy(),
        "knn_Xmean": mean.copy(), "knn_Xscale": std,
        "constant_X_features": constant.copy(), "action_support": support.copy(),
        "forest_seed_base": int(random_state),
    }


def fit_methods_fixed_normalization(X12, cutoff, future, *, mother_ids,
                                    random_state, action_support,
                                    fixed_normalization):
    """Fit exactly four OOF forests and one final forest using one fixed metric.

    This is the only fitting entry point. It must not be invoked before the V5
    physical/training registration. Trees, leaf size, OOF split, joint residuals
    and KNN construction are the original fixed procedures; no hyperparameter
    or seed search and no post-fit normalization replacement occur.
    """
    prepared = prepare_training_inputs(
        X12, cutoff, future, mother_ids=mother_ids, random_state=random_state,
        action_support=action_support, fixed_normalization=fixed_normalization,
    )
    # Lazy import keeps source/input-preparation checks free of model creation.
    from sklearn.ensemble import ExtraTreesRegressor

    X = prepared["X12"]
    target, target_mean = prepared["training_targets"], prepared["training_mother_means"]
    normalization, fold = prepared["normalization"], prepared["fold"]
    oof = np.empty_like(target_mean)
    fold_normalizations, oof_forests = [], []
    for k in range(4):
        train = fold != k
        fold_norm = deepcopy(normalization)
        fold_normalizations.append(fold_norm)
        model = ExtraTreesRegressor(n_estimators=128, min_samples_leaf=4,
                                    random_state=int(random_state)+k, n_jobs=1)
        model.fit(X[train], target_mean[train]/fold_norm["target114_scale"])
        predicted = _finite(model.predict(X[~train]), "OOF normalized prediction")
        if predicted.shape != (16, DIMENSION):
            raise ValueError("OOF model returned a different target layout.")
        oof[~train] = predicted*fold_norm["target114_scale"]
        oof_forests.append(model)
    residual = _finite(target-oof[:, None, :], "joint OOF suffix residual")
    model = ExtraTreesRegressor(n_estimators=128, min_samples_leaf=4,
                                random_state=int(random_state)+4, n_jobs=1)
    model.fit(X, target_mean/normalization["target114_scale"])
    fixed = select_actions(target.reshape(-1, DIMENSION), np.ones(128),
                           normalization, prepared["action_support"])
    return {
        "schema": SCHEMA, "X12": X.copy(),
        "training_ids": prepared["training_ids"].copy(),
        "training_targets": target.copy(), "training_mother_means": target_mean.copy(),
        "normalization": normalization, "fold_normalizations": fold_normalizations,
        "fold": fold.copy(), "oof_prediction": oof,
        "joint_suffix_residual": residual.copy(),
        "knn_Xmean": prepared["knn_Xmean"].copy(),
        "knn_Xscale": prepared["knn_Xscale"].copy(),
        "constant_X_features": prepared["constant_X_features"].copy(),
        "final_forest": model, "oof_forests": oof_forests,
        "forest_seed_base": int(random_state),
        "forest_seeds_oof_then_final": np.arange(int(random_state), int(random_state)+5,
                                                dtype=np.int64),
        "forest_fits": 5, "fixed_choices": fixed["choices"],
        "fixed_expected_loss": fixed["expected_loss"],
        "action_support": prepared["action_support"].copy(),
        "target_layout": "cutoff6_then_action_time_channel108",
        "held_outcomes_accepted": False, "hyperparameter_searches": 0,
        "mu5": True, "training_prior": TRAINING_PRIOR,
        "fixed_metric_namespace": FIXED_METRIC_NAMESPACE,
        "fixed_normalization_for_all_five_fits": True,
        "normalization_estimated_from_new_training": False,
        "knn_coordinates_estimated_from_new_training": True,
        "fixed_metric_identity_scope": "Caller registration binds exact historical metric inputs.",
    }
