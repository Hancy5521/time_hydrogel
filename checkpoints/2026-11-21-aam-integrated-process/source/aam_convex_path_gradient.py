"""Gaussian gradient regression followed by a positive affine atlas QP."""

import hashlib

import numpy as np
from scipy.optimize import LinearConstraint, minimize, nnls


SCHEMA = "aam-conditioned-energy-gradient-affine-qp-v1"
KERNEL_RIDGE = .20
HESSIAN_RIDGES = (.3, 1., 3.)
MASS_FLOOR = .1
TRACE_FLOOR = 1e-8
_ROOM = 1. - MASS_FLOOR
_EPS = np.finfo(np.float64).eps
_ACTIVE_TOL = 5e-7
_KKT_TOL = 2e-7


def _geometry(distance, codes):
    d, z = (np.asarray(x, dtype=np.float64) for x in (distance, codes))
    if (z.ndim != 2 or min(z.shape) < 1 or d.shape != (len(z), len(z))
            or not np.isfinite(z).all() or not np.isfinite(d).all()
            or np.any(d < 0.) or not np.array_equal(d, d.T) or np.any(d.diagonal() != 0.)):
        raise ValueError("Require the original ordered atlas codes and symmetric normalized path distances.")
    return d, z


def _fingerprint(distance, codes):
    h = hashlib.sha256()
    for value in (distance, codes):
        a = np.ascontiguousarray(value, dtype="<f8")
        h.update(str(a.shape).encode("ascii"))
        h.update(a.tobytes())
    return h.hexdigest()


def _probabilities(weights, size):
    p = np.asarray(weights, dtype=np.float64)
    if (p.ndim != 2 or p.shape[1] != size or not len(p) or not np.isfinite(p).all()
            or np.any(p < 0.) or not np.allclose(p.sum(axis=1), 1., atol=2e-13, rtol=0.)):
        raise ValueError("Require normalized frozen RBF probabilities in unchanged atlas order.")
    return p


def _cross(cross, shape):
    a = np.asarray(cross, dtype=np.float64)
    if a.shape != shape or not np.isfinite(a).all() or np.any(a < 0.):
        raise ValueError("Provide one complete nonnegative distance row per independent training mother.")
    return a


def _features(cameras, scale):
    y, s = np.asarray(cameras, dtype=np.float64), np.asarray(scale, dtype=np.float64)
    if y.shape == (3, 4):
        y = y[None, ...]
    if (y.ndim != 3 or y.shape[1:] != (3, 4) or not len(y) or not np.isfinite(y).all()
            or s.shape not in ((4,), (3, 4)) or not np.isfinite(s).all() or np.any(s <= 0.)):
        raise ValueError("Use exactly three original four-channel cameras and frozen positive atlas scales.")
    x = y / s
    mapped = np.concatenate((x[:, :1], np.diff(x, axis=1)), axis=1).reshape(len(y), 12)
    if not np.isfinite(mapped).all():
        raise ValueError("The original camera levels and lag differences must remain finite.")
    return mapped


def _squared_distances(x, y):
    delta = x[:, None] - y[None, :]
    return np.einsum("ijk,ijk->ij", delta, delta, optimize=False)


def _gradients(p, a, distance, codes):
    weighted_residual = p * (a - p @ distance)
    return weighted_residual @ codes - weighted_residual.sum(axis=1)[:, None] * (p @ codes)


def gradient_targets(base_weights, cross, distance, codes):
    """Actual gradient of the full energy with respect to affine coefficients."""
    d, z = _geometry(distance, codes)
    p = _probabilities(base_weights, len(z))
    return _gradients(p, _cross(cross, p.shape), d, z)


def fit(cameras, base_weights, cross, distance, codes, camera_scale, *, ridge=KERNEL_RIDGE):
    """Fit exact multioutput Gaussian KRR to one gradient per training mother."""
    d, z = _geometry(distance, codes)
    p = _probabilities(base_weights, len(z))
    x = _features(cameras, camera_scale)
    ridge = float(ridge)
    if len(x) != len(p) or not np.isfinite(ridge) or ridge <= 0.:
        raise ValueError("Require aligned independent training rows and a positive frozen KRR ridge.")
    targets = _gradients(p, _cross(cross, p.shape), d, z)
    center = targets.mean(axis=0)
    squared = _squared_distances(x, x)
    positive = squared[squared > 0.]
    bandwidth2 = float(np.median(positive)) if len(positive) else 1.
    gram = np.exp(-.5 * squared / bandwidth2)
    gram.flat[::len(gram) + 1] += ridge
    weights = np.linalg.solve(gram, targets - center)
    if not np.isfinite(weights).all():
        raise ArithmeticError("The registered KRR fit must produce finite gradient coefficients.")
    return {"schema": SCHEMA, "geometry_sha256": _fingerprint(d, z),
            "camera_scale": np.array(camera_scale, copy=True, dtype=np.float64),
            "feature_rule": "divide snapshot channels by frozen scales; then concatenate Y1, delta21, delta32",
            "features": x, "target_gradients": targets, "mean_gradient": center, "coefficients": weights,
            "bandwidth_squared": bandwidth2, "kernel_ridge": ridge, "training_mothers": len(x)}


def energy(weights, cross, distance):
    """Original two-metric averaged energy, using complete legal atlas support."""
    d = np.asarray(distance, dtype=np.float64)
    if d.ndim != 2 or len(d) != d.shape[1]:
        raise ValueError("Require a square complete atlas distance.")
    p = _probabilities(weights, len(d))
    a = _cross(cross, p.shape)
    return np.sum(p * (a - .5 * (p @ d)), axis=1)


def _objective(matrix, gradient, direction):
    return float(gradient @ direction + .5 * direction @ matrix @ direction)


def _ray(constraint, direction):
    loss = max(0., -float((constraint @ direction).min()))
    bound = min(1., _ROOM / loss) if loss else 1.
    # Maintain a small inward rounding margin when the lower bound is active.
    factor = bound if loss < _ROOM * (1. - 16. * _EPS) else bound * (1. - 64. * _EPS)
    return factor * direction, factor


def _certificate(matrix, gradient, constraint, direction):
    """Scaled primal/dual residual and gap using independently fitted KKT multipliers."""
    slack = constraint @ direction + _ROOM
    active = slack <= _ACTIVE_TOL
    dr = matrix @ direction + gradient
    dual = np.zeros(len(constraint))
    if active.any():
        try:
            dual[active], _ = nnls(constraint[active].T, dr)
        except RuntimeError:
            pass
    balance = gradient - constraint.T @ dual
    residual = float(np.linalg.norm(matrix @ direction + balance, ord=np.inf))
    comp = float(np.max(np.abs(slack * dual)))
    primal = _objective(matrix, gradient, direction)
    dual_value = -.5 * float(balance @ np.linalg.solve(matrix, balance)) - _ROOM * float(dual.sum())
    return {"minimum_slack": float(slack.min()), "stationarity": residual,
            "complementarity": comp, "duality_gap": float(primal - dual_value),
            "active_count": int(active.sum()), "dual_positive_count": int(np.count_nonzero(dual))}


def _fit_qp(matrix, gradient, constraint):
    """Solve the low-dimensional convex QP to a numerical KKT certificate."""
    target = -np.linalg.solve(matrix, gradient)
    ray, fraction = _ray(constraint, target)
    scale = max(float(np.linalg.norm(matrix, ord=np.inf)), float(np.linalg.norm(gradient, ord=np.inf)))
    q, g = matrix / scale, gradient / scale
    if fraction == 1.:
        certificate = _certificate(q, g, constraint, target)
        return target, ray, fraction, "unconstrained", "original ridge solution feasible", certificate, scale
    lower = LinearConstraint(constraint, -_ROOM, np.inf)
    result = minimize(lambda x: _objective(q, g, x), ray, jac=lambda x: q @ x + g,
                      method="SLSQP", constraints=[lower], options={"ftol": 1e-12, "maxiter": 192})
    if np.isfinite(result.x).all():
        candidate, safeguard = _ray(constraint, result.x)
        cert = _certificate(q, g, constraint, candidate)
        gap = cert["duality_gap"]
        if (safeguard >= 1. - _ACTIVE_TOL and cert["minimum_slack"] >= -32.*_EPS
                and cert["stationarity"] <= _KKT_TOL and cert["complementarity"] <= _KKT_TOL
                and abs(gap) <= _KKT_TOL
                and _objective(q, g, candidate) <= _objective(q, g, ray) + _KKT_TOL):
            return candidate, ray, fraction, "certified_qp", str(result.message), cert, scale
    cert_ray = _certificate(q, g, constraint, ray)
    return ray, ray, fraction, "ray_fallback", str(result.message), cert_ray, scale


def _correction(p, g, distance, codes, relative_ridge):
    a = codes - p @ codes
    support = p > 0.
    j = p[:, None] * a
    hat = -(j.T @ (distance @ j))
    hat = .5 * (hat + hat.T)
    trace_scale = float(np.trace(hat) / len(g))
    absolute_ridge = relative_ridge * max(trace_scale, TRACE_FLOOR)
    matrix = hat + absolute_ridge * np.eye(len(g))
    direction, ray, fraction, status, msg, certificate, scale = _fit_qp(matrix, g, a[support])
    updated = p + j @ direction
    updated /= updated.sum()
    floor = float((updated[support] / p[support]).min())
    if floor < MASS_FLOOR - 32.*_EPS or not np.isfinite(updated).all() or np.any(updated[~support] != 0.):
        raise ArithmeticError("The affine direction failed to preserve the original mass floor.")
    return {"weights": updated, "directions": direction, "curvature_mean": trace_scale,
            "absolute_ridge": max(0., absolute_ridge), "ray_multipliers": fraction,
            "baseline_mass_fraction": floor, "unpenalized_predicted_energy_delta": _objective(hat, g, direction),
            "penalized_objective_delta": _objective(matrix, g, direction),
            "ray_objective_delta": _objective(matrix, g, ray), "objective_scale": scale,
            "solver_status": status, "solver_message": msg, **certificate}


def predict(model, cameras, base_weights, distance, codes, *, mode="kernel", hessian_ridge=1.):
    """Correct the legal atlas law using KRR or its matched constant-gradient control."""
    d, z = _geometry(distance, codes)
    p = _probabilities(base_weights, len(z))
    x = _features(cameras, model["camera_scale"])
    rate = float(hessian_ridge)
    if (model.get("schema") != SCHEMA or model.get("geometry_sha256") != _fingerprint(d, z)
            or len(x) != len(p) or mode not in ("kernel", "constant") or not np.isfinite(rate) or rate <= 0.):
        raise ValueError("Use the frozen fit, aligned queries, and a positive preregistered relative ridge.")
    gradients = np.broadcast_to(model["mean_gradient"], (len(p), z.shape[1])).copy()
    if mode == "kernel":
        kernel = np.exp(-.5 * _squared_distances(x, model["features"]) / model["bandwidth_squared"])
        gradients += kernel @ model["coefficients"]
    if not np.isfinite(gradients).all():
        raise ArithmeticError("Predicted target gradients must be finite.")
    rows = [_correction(prior, grad, d, z, rate) for prior, grad in zip(p, gradients, strict=True)]
    data = {name: np.asarray([row[name] for row in rows]) for name in rows[0]}
    return {"mode": mode, "relative_ridge": rate, "predicted_gradients": gradients,
            "certified_qp": data["solver_status"] == "certified_qp", **data}


def fixtures():
    """Pure finite checks of ES gradient, active QP, and supported probability."""
    coord = np.array([-2., -1., 0., 1., 3.])
    d = np.abs(coord[:, None] - coord[None, :])
    z = np.column_stack((coord, coord**2))
    p = np.array([[.2, .1, .4, .25, .05]])
    a = np.array([[.2, .4, .3, .3, .7]])
    j = p[0, :, None] * (z - p[0] @ z)
    du = np.array([.21, -.41])
    e = 2.e-6
    deriv = (energy(p + e*(j @ du), a, d) - energy(p - e*(j @ du), a, d)) / (2.*e)
    g = gradient_targets(p, a, d, z)
    np.testing.assert_allclose(deriv, g @ du, rtol=0., atol=5e-11)
    face = np.array([[1., 0.], [0., 1.], [1., 1.]])
    interior, _, _, inside_status, _, interior_cert, _ = _fit_qp(np.eye(2), np.array([.1, .2]), face)
    np.testing.assert_allclose(interior, [-.1, -.2], rtol=0., atol=3e-16)
    assert inside_status == "unconstrained" and interior_cert["stationarity"] == 0.
    got, ray, ratio, status, _, certificate, _ = _fit_qp(np.eye(2), np.array([2., .3]), face)
    # The optimum activates both the first bound and the sum constraint.
    np.testing.assert_allclose(got, [-.9, 0.], rtol=0., atol=3e-7)
    assert status == "certified_qp" and ratio < 1. and certificate["stationarity"] < _KKT_TOL
    assert _objective(np.eye(2), np.array([2., .3]), got) < _objective(np.eye(2), np.array([2., .3]), ray)
    row = _correction(p[0], 25.*g[0], d, z, .3)
    assert row["baseline_mass_fraction"] >= MASS_FLOOR - 32.*_EPS and row["penalized_objective_delta"] <= 0.
    assert abs(row["weights"].sum() - 1.) < 2e-15
    point_mass = np.array([0., 0., 1., 0., 0.])
    degenerate = _correction(point_mass, np.zeros(2), d, z, .3)
    assert degenerate["absolute_ridge"] == .3 * TRACE_FLOOR
    np.testing.assert_array_equal(degenerate["weights"], point_mass)
    return {"schema": SCHEMA, "native_calls": 0, "exact_formula_gradient": True, "active_qp": True,
            "positive_affine_update": True}


if __name__ == "__main__":
    print(fixtures())
