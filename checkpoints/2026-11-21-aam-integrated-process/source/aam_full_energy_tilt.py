"""Direct proper-score fitting on complete path atoms, without simulation."""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
from scipy.special import softmax


def rows(values):
    value = np.asarray(values, dtype=float)
    if value.ndim != 2 or min(value.shape) == 0 or not np.isfinite(value).all():
        raise ValueError('Use a finite nonempty two-dimensional feature matrix.')
    return value


def parent_mask(query_mothers, atom_mothers):
    """Exclude every copy of the query mother, including symmetry siblings."""
    query, bank = np.asarray(query_mothers), np.asarray(atom_mothers)
    if query.ndim != 1 or bank.ndim != 1:
        raise ValueError('Mother identities must be one-dimensional.')
    mask = query[:, None] != bank[None, :]
    if not np.all(np.any(mask, axis=1)):
        raise ValueError('Leave-one-mother-out removed the complete training support.')
    return mask


def equal_mother_weights(mothers):
    """Give each independent mother unit weight, split among its query rows."""
    keys = np.asarray(mothers)
    if keys.ndim != 1 or not len(keys):
        raise ValueError('Provide one independent mother label for each query row.')
    unique, which, counts = np.unique(keys, return_inverse=True, return_counts=True)
    return 1. / (len(unique) * counts[which])


def energy_distances(physical_targets, physical_atoms):
    """Use the complete given endpoint and Euclidean rather than squared distances."""
    target, atoms = rows(physical_targets), rows(physical_atoms)
    scale = np.sqrt(atoms.shape[1])
    return cdist(target, atoms) / scale, cdist(atoms, atoms) / scale


def clipped_rows(values, radius):
    value = rows(values)
    if not np.isfinite(radius) or radius <= 0.:
        raise ValueError('The bound must be strictly positive.')
    factor = np.maximum(1., np.linalg.norm(value, axis=1) / radius)
    return value / factor[:, None]


def _probabilities(X, Z, base, parameter, link):
    effect = (X @ parameter) @ Z.T
    if link == 'exponential':
        log_base = np.full_like(base, -np.inf)
        np.log(base, out=log_base, where=base > 0.)
        q = softmax(log_base + effect, axis=1)
    else:
        change = effect - np.sum(effect * base, axis=1)[:, None]
        q = base * (1. + change)
    if np.any(q[base > 0.] <= 0.) or not np.isfinite(q).all():
        raise FloatingPointError('Allowed path probability lost positivity or representability.')
    return q


def _base_mass(prior, nqueries, mask):
    if not np.isfinite(prior).all() or np.any(prior <= 0.):
        raise ValueError('Each candidate path needs a positive common prior mass.')
    shape = (nqueries, len(prior))
    legal = np.ones(shape, dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    if legal.shape != shape or not np.all(np.any(legal, axis=1)):
        raise ValueError('Forecast eligibility mask has the wrong support.')
    masked = legal * (prior / prior.max())
    base = masked / masked.sum(axis=1)[:, None]
    if np.any(base[legal] <= 0.):
        raise ValueError('Eligible atom prior underflowed.')
    return base


@dataclass(frozen=True)
class DirectEnergyObjective:
    query: np.ndarray
    atoms: np.ndarray
    target_distances: np.ndarray
    pair_distances: np.ndarray
    base: np.ndarray
    unit_mass: np.ndarray
    ridge: float
    link: str
    radius: float

    @property
    def shape(self):
        return self.query.shape[1], self.atoms.shape[1]

    def evaluate(self, parameters):
        """Mean full energy and exact derivative through normalized path probabilities."""
        w = np.asarray(parameters, dtype=float).reshape(self.shape)
        q = _probabilities(self.query, self.atoms, self.base, w, self.link)
        inner = q @ self.pair_distances
        loss = np.sum(q * (self.target_distances - .5 * inner), axis=1)
        derivative = self.target_distances - inner
        mass = q if self.link == 'exponential' else self.base
        centered = derivative - np.sum(mass * derivative, axis=1)[:, None]
        logits = self.unit_mass[:, None] * mass * centered
        gradient = (self.query.T @ logits) @ self.atoms + self.ridge * w
        value = self.unit_mass @ loss + .5 * self.ridge * np.sum(w*w)
        return float(value), gradient.ravel()


def make_objective(X, Z, targets, physical_atoms, prior, *, query_mothers, atom_mothers,
                   ridge, link='exponential', floor=.05):
    """Construct a train-only objective with exact exclusion of the target mother."""
    X, Z = rows(X), rows(Z)
    if link not in ('exponential', 'affine') or not np.isfinite(ridge) or ridge <= 0.:
        raise ValueError('Use a registered link and strictly positive ridge.')
    if not np.isfinite(floor) or not 0. < floor < 1.:
        raise ValueError('The bounded affine probability floor must lie in (0,1).')
    query_mothers, atom_mothers = np.asarray(query_mothers), np.asarray(atom_mothers)
    prior = np.asarray(prior, dtype=float)
    if query_mothers.shape != (len(X),) or atom_mothers.shape != (len(Z),) or prior.shape != (len(Z),):
        raise ValueError('One identity and prior mass are needed per row as appropriate.')
    if link == 'affine' and (np.max(np.linalg.norm(X, axis=1)) > 1. + 2e-15
                             or np.max(np.linalg.norm(Z, axis=1)) > .5 + 2e-15):
        raise ValueError('Affine positivity requires query norm <= 1 and atom norm <= 1/2.')
    cross, pair = energy_distances(targets, physical_atoms)
    if cross.shape != (len(X), len(Z)):
        raise ValueError('The full endpoint rows and corresponding predictors must agree.')
    base = _base_mass(prior, len(X), parent_mask(query_mothers, atom_mothers))
    return DirectEnergyObjective(X, Z, cross, pair, base, equal_mother_weights(query_mothers),
                                 ridge, link, 1. - floor)


@dataclass(frozen=True)
class Fit:
    parameters: np.ndarray
    objective: float
    stationarity: float
    iterations: int
    link: str
    radius: float

    def predict(self, X, Z, prior, eligibility=None):
        X, Z, prior = rows(X), rows(Z), np.asarray(prior, dtype=float)
        if self.parameters.shape != (X.shape[1], Z.shape[1]) or prior.shape != (len(Z),):
            raise ValueError('Forecast features, parameter shape or candidate prior do not match.')
        if self.link == 'affine':
            if np.linalg.norm(self.parameters) > self.radius + 2e-14:
                raise ValueError('Affine coefficients exceed the positivity radius.')
            if np.max(np.linalg.norm(X, axis=1)) > 1. + 2e-15 or np.max(np.linalg.norm(Z, axis=1)) > .5 + 2e-15:
                raise ValueError('Affine positivity requires bounded forecast features.')
        base = _base_mass(prior, len(X), eligibility)
        return _probabilities(X, Z, base, self.parameters, self.link)


def _ball(direction, radius):
    return direction / max(1., np.linalg.norm(direction) / radius)


def _affine_minimum(target):
    """Global convex minimum to numerical projected first-order tolerance."""
    w = np.zeros(np.prod(target.shape))
    value, grad = target.evaluate(w)
    step = 1.
    for passes in range(4000):
        stationarity = np.linalg.norm(w - _ball(w - grad, target.radius))
        if stationarity <= 2.e-8:
            return w, value, stationarity, passes
        step = min(2 * step, 1.e4)
        for _ in range(30):
            trial = _ball(w - step * grad, target.radius)
            next_value, next_grad = target.evaluate(trial)
            change = trial - w
            bound = value + grad @ change + np.sum(change*change) / (2 * step)
            if next_value <= bound + 1.e-15:
                break
            step *= .5
        else:
            raise ArithmeticError('The affine energy optimizer failed its quadratic descent check.')
        w, value, grad = trial, next_value, next_grad
    raise ArithmeticError('Affine projected first-order tolerance was not reached.')


def fit(objective):
    if objective.link == 'affine':
        parameter, value, defect, iterations = _affine_minimum(objective)
    else:
        fit_result = minimize(objective.evaluate, np.zeros(np.prod(objective.shape)), jac=True,
                              method='L-BFGS-B', options={'maxiter': 400, 'gtol': 2.e-9, 'ftol': 1.e-13})
        parameter, iterations = fit_result.x, fit_result.nit
        value, gradient = objective.evaluate(parameter)
        defect = np.linalg.norm(gradient, ord=np.inf)
        if defect > 2.e-7:
            raise ArithmeticError(f'Exponential energy optimizer did not reach stationarity: {fit_result.message}')
    return Fit(parameter.reshape(objective.shape), value, float(defect), iterations,
               objective.link, objective.radius)


def synthetic_checks():
    """Algebra and finite honest controls with no materials data or native calls."""
    rng = np.random.default_rng(20261121)
    group = np.repeat(np.arange(10), 2)
    X = clipped_rows(rng.normal(size=(20, 3)), 1.)
    Z = clipped_rows(rng.normal(size=(20, 2)), .5)
    labels = rng.normal(size=(20, 5))
    prior = rng.uniform(.1, 2., size=20)
    w, direction = rng.normal(size=6) * .1, rng.normal(size=6)
    epsilon = 1.e-5
    records = {}
    for link in ('exponential', 'affine'):
        objective = make_objective(X, Z, labels, labels, prior,
                                   query_mothers=group, atom_mothers=group, link=link, ridge=.02)
        value, deriv = objective.evaluate(w)
        finite = (objective.evaluate(w + epsilon * direction)[0]
                  - objective.evaluate(w - epsilon * direction)[0]) / (2 * epsilon)
        np.testing.assert_allclose(finite, deriv @ direction, atol=3.e-10)
        model = fit(objective)
        q = model.predict(X, Z, prior, parent_mask(group, group))
        np.testing.assert_array_equal(q[group[:, None] == group], 0.)
        np.testing.assert_allclose(q.sum(axis=1), 1., atol=2.e-14)
        if link == 'affine':
            midpoint = objective.evaluate(w / 2.)[0]
            if midpoint > (value + objective.evaluate(w * 0.)[0]) / 2. + 2.e-14:
                raise AssertionError('Convex affine score failed a Jensen check.')
            if np.min(q[objective.base > 0] / objective.base[objective.base > 0]) < 1. - model.radius - 1.e-14:
                raise AssertionError('The bounded affine positivity theorem was violated.')
        records[link] = {'directional_derivative_error': float(abs(finite - deriv @ direction)),
                         'stationarity': model.stationarity}

    X = np.repeat([-1., 1.], 60)[:, None]
    mothers = np.arange(len(X))
    for treatment, count in [('informative', 45), ('uninformative', 30)]:
        targets = np.r_[np.zeros(count), np.ones(60 - count),
                        np.zeros(60 - count), np.ones(count)][:, None]
        Z = targets - .5
        prior = np.ones(len(X))
        truth = np.array((1. - count / 60, count / 60))
        for link in ('exponential', 'affine'):
            objective = make_objective(X, Z, targets, targets, prior,
                                       query_mothers=mothers, atom_mothers=mothers, ridge=.0001, link=link)
            model = fit(objective)
            query = np.array([[-1.], [1.]])
            probability = model.predict(query, Z, prior) @ targets[:, 0]
            estimate = np.mean((.5 - truth)**2 - (probability - truth)**2)
            if treatment == 'informative' and link == 'affine':
                # With q(z)=p(z)(1+x wz) this deliberately exceeds the bound on w.
                # The analytic constrained optimum saturates at |w| = 0.95.
                np.testing.assert_allclose(probability, [.2625, .7375], atol=3.e-8)
            else:
                np.testing.assert_allclose(probability, truth, atol=.012)
            if treatment == 'informative' and estimate < .06:
                raise AssertionError('The honest finite oracle failed to learn strong conditional signal.')
            records[link][treatment] = {'forecast_mass': probability.tolist(), 'expected_energy_gain': float(estimate)}
    return records


if __name__ == '__main__':
    from threadpoolctl import threadpool_limits

    with threadpool_limits(limits=1):
        print(synthetic_checks())
