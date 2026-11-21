"""Energy-coordinate regression and joint legal-path reweighting, without simulation."""

from dataclasses import dataclass
from operator import index

import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
from scipy.special import logsumexp


def _rows(a, name):
    a = np.asarray(a, dtype=float)
    if a.ndim != 2 or not len(a) or not np.isfinite(a).all():
        raise ValueError(f'{name} must be a finite nonempty matrix.')
    return a


def _prior(value, n):
    a = np.asarray(value, dtype=float)
    if a.shape != (n,) or not np.isfinite(a).all() or np.any(a <= 0.):
        raise ValueError('Give one strictly positive prior weight for each admissible atom.')
    # Scale before summing to handle weights with a large common magnitude.
    scaled = a / a.max()
    p = scaled / scaled.sum()
    if np.any(p == 0.):
        raise ValueError('Normalized prior falls outside the representable positive range.')
    return p


@dataclass(frozen=True)
class EnergyCoordinates:
    atoms: np.ndarray
    prior: np.ndarray
    prior_distances: np.ndarray
    prior_pair_distance: float
    basis: np.ndarray
    eigenvalues: np.ndarray
    training_coordinates: np.ndarray

    def transform(self, values):
        """Embed new rows in the frozen training span by centered cross distances."""
        values = _rows(values, 'trajectory targets')
        if values.shape[1] != self.atoms.shape[1]:
            raise ValueError('New targets must use the registered joint trajectory dimensions.')
        cross = cdist(values, self.atoms) / np.sqrt(self.atoms.shape[1])
        return _center_cross(cross, self.prior, self.prior_distances, self.prior_pair_distance) @ self.basis


def _center_cross(cross, p, prior_distance, pair_distance):
    return -.5 * (cross - (cross @ p)[:, None] - prior_distance[None, :] + pair_distance)


def fit_energy_coordinates(atoms, prior, max_rank):
    """Weighted eigensystem for the Euclidean distance of negative type."""
    z = _rows(atoms, 'trajectory atoms').copy()
    max_rank = index(max_rank)
    if z.shape[1] == 0 or max_rank < 0:
        raise ValueError('Use a nonempty target dimension and a nonnegative rank cap.')
    p = _prior(prior, len(z))
    distances = cdist(z, z) / np.sqrt(z.shape[1])
    prior_distance = distances @ p
    pair_distance = float(p @ prior_distance)
    gram = _center_cross(distances, p, prior_distance, pair_distance)
    gram = .5 * (gram + gram.T)
    root_weight = np.sqrt(p)
    weighted = root_weight[:, None] * gram * root_weight[None, :]
    values, axes = np.linalg.eigh(weighted)
    tolerance = 32 * np.finfo(float).eps * len(z) * np.abs(values).max()
    if values[0] < -tolerance:
        raise ArithmeticError('The centered Euclidean energy kernel is not numerically positive semidefinite.')
    keep = np.flatnonzero(values > tolerance)[::-1][:max_rank]
    # K * sqrt(p) * v / sqrt(eigenvalue) is the same map on new rows.
    basis = (root_weight[:, None] * axes[:, keep]) / np.sqrt(values[keep])
    coordinates = gram @ basis
    retained = values[keep]
    for value in (z, p, prior_distance, basis, retained, coordinates):
        value.flags.writeable = False
    return EnergyCoordinates(z, p, prior_distance, pair_distance, basis, retained, coordinates)


@dataclass(frozen=True)
class CoordinateRegressor:
    xcenter: np.ndarray
    xscale: np.ndarray
    intercept: np.ndarray
    coefficients: np.ndarray

    def predict(self, observations):
        y = _rows(observations, 'observations')
        if y.shape[1] != len(self.xcenter):
            raise ValueError('New observations must use the training camera coordinates.')
        return self.intercept + ((y - self.xcenter) / self.xscale) @ self.coefficients


def fit_coordinate_ridge(observations, coordinates, ridge):
    """Fit a regularized conditional mean with observed training rows only."""
    x = _rows(observations, 'training observations')
    z = _rows(coordinates, 'training coordinates')
    ridge = float(ridge)
    if len(x) != len(z) or x.shape[1] == 0 or not np.isfinite(ridge) or ridge <= 0.:
        raise ValueError('Use paired observed training rows and a fixed positive ridge.')
    xcenter, xscale, intercept = x.mean(axis=0), x.std(axis=0), z.mean(axis=0)
    xscale[xscale == 0.] = 1.
    design = (x - xcenter) / xscale
    a = design.T @ design / len(x) + ridge * np.eye(x.shape[1])
    target = design.T @ (z - intercept) / len(x)
    return CoordinateRegressor(xcenter, xscale, intercept, np.linalg.solve(a, target))


@dataclass(frozen=True)
class PathProjection:
    weights: np.ndarray
    log_weights: np.ndarray
    moment: np.ndarray
    tilt: np.ndarray
    optimality_residual: float
    iterations: int


def project_energy_mean(coordinates, target, prior, regularization):
    """Minimize half squared moment residual plus regularization times KL(q||p)."""
    z = _rows(coordinates, 'legal atom coordinates')
    p = _prior(prior, len(z))
    m = np.asarray(target, dtype=float)
    strength = float(regularization)
    if (m.shape != (z.shape[1],) or not np.isfinite(m).all()
            or not np.isfinite(strength) or strength <= 0.):
        raise ValueError('Use a finite target coordinate and positive entropy regularization.')
    log_prior = np.log(p)
    if len(m) == 0:
        return PathProjection(p, log_prior, m.copy(), m.copy(), 0., 0)

    # Centering both features and target leaves q unchanged and reduces dual offset.
    center = p @ z
    features, wanted = z - center, m - center
    ridge = strength * np.eye(len(m))

    def tilt(theta):
        unnormalized = log_prior + features @ theta
        normalizer = logsumexp(unnormalized)
        log_weights = unnormalized - normalizer
        return normalizer, log_weights, np.exp(log_weights)

    def objective(theta):
        normalizer, _, q = tilt(theta)
        value = normalizer - wanted @ theta + .5 * strength * (theta @ theta)
        gradient = q @ features - wanted + strength * theta
        return value, gradient

    def hessian(theta):
        _, _, q = tilt(theta)
        deviations = features - q @ features
        return deviations.T @ (q[:, None] * deviations) + ridge

    answer = minimize(objective, np.zeros(len(m)), jac=True, hess=hessian, method='trust-exact',
                      options={'gtol': 2e-12, 'maxiter': 250})
    residual = np.linalg.norm(objective(answer.x)[1], ord=np.inf)
    scale = max(1., np.max(np.abs(wanted)), np.max(np.abs(features)))
    if not np.isfinite(residual) or residual > 2e-8 * scale:
        raise ArithmeticError(f'Entropy projection failed its dual optimality check: {answer.message}')
    _, log_weights, weights = tilt(answer.x)
    if np.any(weights == 0.):
        raise FloatingPointError('A positive path weight underflowed. The finite support cannot be represented.')
    return PathProjection(weights, log_weights, weights @ z, answer.x, float(residual), answer.nit)


def fixtures():
    z = np.array([[0., 0.], [1., 0.], [0., 2.], [1., 0.], [3., 1.]])
    p = np.array([.12, .20, .16, .32, .20])
    kernel = fit_energy_coordinates(z, p, 9)
    coordinates = kernel.training_coordinates
    np.testing.assert_allclose(kernel.transform(z), coordinates, atol=3e-14)
    np.testing.assert_allclose(p @ coordinates, 0., atol=3e-14)
    np.testing.assert_allclose(cdist(coordinates, coordinates, 'sqeuclidean'),
                               cdist(z, z) / np.sqrt(z.shape[1]), atol=4e-14)
    empirical = np.array([.61, .06, .03, .22, .08])
    training_distances = cdist(z, z) / np.sqrt(z.shape[1])
    score = empirical @ training_distances - .5 * (empirical @ training_distances @ empirical)
    np.testing.assert_allclose(score, np.square(empirical @ coordinates - coordinates).sum(axis=1), atol=6e-14)
    # Orthogonal residual outside the training span shifts every forecast score equally.
    outside = np.array([[.25, .8], [5., -1.]])
    external = kernel.transform(outside)
    for row, location in zip(external, outside, strict=True):
        expected = (p - empirical) @ (cdist(z, location[None])[:, 0] / np.sqrt(z.shape[1]))
        expected -= .5 * (p @ training_distances @ p - empirical @ training_distances @ empirical)
        observed = np.sum((p @ coordinates - row)**2) - np.sum((empirical @ coordinates - row)**2)
        np.testing.assert_allclose(observed, expected, atol=4e-14)

    neutral = project_energy_mean(coordinates, p @ coordinates, p, .11)
    np.testing.assert_allclose(neutral.weights, p, atol=3e-14)
    m = .77 * coordinates[0] - .45 * coordinates[2] + .6 * coordinates[4]
    strength = .11
    projected = project_energy_mean(coordinates, m, p, strength)
    q = projected.weights
    np.testing.assert_allclose(q.sum(), 1., atol=2e-14)
    assert np.all(q > 0.)
    np.testing.assert_allclose(q[1] / q[3], p[1] / p[3], atol=2e-14)
    kkt = coordinates @ (q @ coordinates - m) + strength * (np.log(q / p) + 1.)
    np.testing.assert_allclose(kkt, kkt.mean(), atol=4e-8)
    np.testing.assert_allclose(projected.moment - m + strength * projected.tilt, 0., atol=4e-8)

    def primal(w):
        offset = w @ coordinates - m
        kl = np.log(w / p)
        return .5 * (offset @ offset) + strength * (w @ kl), coordinates @ offset + strength * (kl + 1.)

    optimum = minimize(primal, p, jac=True, method='SLSQP', bounds=[(1e-14, 1.)] * len(p),
                       constraints={'type': 'eq', 'fun': lambda w: w.sum()-1.,
                                    'jac': lambda w: np.ones_like(w)}, options={'ftol': 1e-13})
    if not optimum.success:
        raise AssertionError(optimum.message)
    np.testing.assert_allclose(q, optimum.x, atol=2e-7)
    np.testing.assert_allclose(primal(q)[0], optimum.fun, atol=1e-13)

    duplicate = fit_energy_coordinates(np.ones((5, 2)), p, 5)
    assert duplicate.training_coordinates.shape == (5, 0)
    new = duplicate.transform(np.ones((2, 2)))
    np.testing.assert_allclose(project_energy_mean(duplicate.training_coordinates, new[0], p, strength).weights, p)
    capped = fit_energy_coordinates(z, p, 1)
    assert capped.training_coordinates.shape == (len(z), 1) and capped.eigenvalues[0] > 0.
    x = np.column_stack((np.ones(5), np.arange(5), np.arange(5)**2))
    regression = fit_coordinate_ridge(x, coordinates, .25)
    np.testing.assert_allclose(regression.predict(x).mean(axis=0), coordinates.mean(axis=0), atol=3e-14)
    assert regression.predict(np.array([[1, 10, 50]])).shape == (1, coordinates.shape[1])
    empty = fit_coordinate_ridge(x, duplicate.training_coordinates, .25)
    assert empty.predict(x).shape == (5, 0)
    print('PASS energy law fixtures: geometry, out-of-sample score difference, simplex optimum, KKT, support and zero-rank cases.')


if __name__ == '__main__':
    fixtures()
