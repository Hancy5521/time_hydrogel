"""Infer one bounded material code from two populations of independent early snapshots."""

from itertools import product
import numpy as np
from scipy.linalg import solve, solve_triangular
from scipy.spatial.distance import cdist
from scipy.special import softmax
from scipy.stats import qmc


LIMIT = 0.48
COORDINATES = ("initiation", "growth", "pendant", "bridge")


def _observations(x):
    x = np.asarray(x, dtype=float)
    if x.ndim != 4 or x.shape[1] != 2 or min(x.shape) == 0 or not np.isfinite(x).all():
        raise ValueError("Expected finite observation tensor [card, A/B, replicate, feature]")
    if x.shape[2] < 2:
        raise ValueError("At least two independent replicates per environment are required")
    return np.column_stack([x.mean(2).reshape(len(x), -1), x.std(2).reshape(len(x), -1)])


def _scaling(x):
    scale = x.std(axis=0)
    keep = scale > 1.e-9
    return x.mean(axis=0), np.where(keep, scale, 1.), keep


def _standardize(x, scaling):
    center, scale, keep = scaling
    return (x[:, keep] - center[keep]) / scale[keep]


def _posterior_draws(mean, factor, n_draws, rng):
    """Draw from a bounded correlated Gaussian without clamping any accepted draw."""
    mean = np.clip(mean, -LIMIT, LIMIT)
    remaining = np.ones((len(mean), n_draws), dtype=bool)
    result = np.empty((len(mean), n_draws, 4))
    for _ in range(256):
        card, draw = np.nonzero(remaining)
        if not len(card):
            return result
        proposal = mean[card] + rng.normal(size=(len(card), 4)) @ factor.T
        ok = (np.abs(proposal) <= LIMIT).all(axis=1)
        result[card[ok], draw[ok]] = proposal[ok]
        remaining[card[ok], draw[ok]] = False
    raise ArithmeticError("Bounded Gaussian posterior draw did not converge")


def _fit_inverse(x, g, kind, hp):
    scaling = _scaling(x)
    z = _standardize(x, scaling)
    response_center = g.mean(axis=0)
    response = g - response_center
    if kind == "ridge":
        kernel = z @ z.T
        width = None
        ridge, = hp
    else:
        width, ridge = hp
        kernel = np.exp(-cdist(z, z, "sqeuclidean") / (2 * width**2 * max(z.shape[1], 1)))
    kernel.flat[::len(z) + 1] += ridge
    weights = solve(kernel, response, assume_a="pos")
    return scaling, z, response_center, weights, width


def _inverse_mean(fit, query):
    scaling, z, center, weights, width = fit
    query = _standardize(query, scaling)
    if width is None:
        kernel = query @ z.T
    else:
        kernel = np.exp(-cdist(query, z, "sqeuclidean") / (2 * width**2 * max(z.shape[1], 1)))
    return center + kernel @ weights


def _terms(code, degree):
    u = code / LIMIT
    if degree == 1:
        return u
    i, j = np.triu_indices(4, k=1)
    return np.c_[u, (u**2 - 1./3.) / 0.30, u[:, i] * u[:, j] / 0.33]


def _fit_response(x, g, degree, shrinkage):
    scaling = _scaling(x)
    z = _standardize(x, scaling)
    phi = _terms(g, degree)
    center_phi = phi.mean(axis=0)
    phi = phi-center_phi
    center_z = z.mean(axis=0)
    regularizer = np.full(phi.shape[1], max(len(g)*.025, 1.))
    regularizer[4:] *= 12.
    beta = solve(phi.T @ phi + np.diag(regularizer), phi.T @ (z-center_z), assume_a="pos")
    residual = z-center_z-phi @ beta
    covariance = residual.T @ residual / max(len(g)-phi.shape[1]-1, 1)
    covariance *= 1-shrinkage
    covariance.flat[::len(covariance)+1] += (
        shrinkage * (residual**2).mean(axis=0) + .05
    )
    factor = np.linalg.cholesky(covariance)
    whitened = solve_triangular(factor, beta.T, lower=True)
    # Code-invariant directions cancel from a Gaussian likelihood ratio.
    vectors, singular, _ = np.linalg.svd(whitened, full_matrices=False)
    rank = int((singular > 1.e-10).sum())
    axes = vectors[:, :rank]
    projected = (axes.T @ whitened).T
    return scaling, center_phi, center_z, factor, axes, projected, degree


def _distance(fit, x, support):
    scaling, center_phi, center_z, factor, axes, projected, degree = fit
    z = _standardize(x, scaling)-center_z
    latent = solve_triangular(factor, z.T, lower=True).T @ axes
    signal = (_terms(support, degree)-center_phi) @ projected
    return cdist(latent, signal, "sqeuclidean")


def _draw_discrete(mass, grid, n_draws, rng):
    # Stratify requested draws with a random offset in each bin. Randomizing
    # sample order keeps common downstream simulation seeds exchangeable.
    cdf = np.cumsum(mass, axis=1)
    cdf[:, -1] = 1.
    result = np.empty((len(mass), n_draws, 4))
    for i, weights in enumerate(cdf):
        threshold = (np.arange(n_draws)+rng.random(n_draws)) / n_draws
        idx = np.searchsorted(weights, threshold, side="right")
        result[i] = grid[idx[rng.permutation(n_draws)]]
    return result


def _energy(samples, independent_samples, truth):
    first = np.linalg.norm(samples-truth[:, None], axis=-1).mean()
    second = np.linalg.norm(samples-independent_samples, axis=-1).mean()
    return float(first - 0.5*second)


def _folds(n, seed):
    order = np.random.default_rng(seed).permutation(n)
    return [(np.setdiff1d(np.arange(n), test), test) for test in np.array_split(order, 4)]


def fit_encoder(X_train, G_train, seed=80123):
    """Fit baselines and a common-code likelihood using development cards alone."""
    x = _observations(X_train)
    g = np.asarray(G_train, dtype=float)
    if len(x) < 16 or g.shape != (len(x), 4) or not np.isfinite(g).all() or (np.abs(g)>LIMIT).any():
        raise ValueError("At least 16 matched development cards with four bounded codes required")
    if not _scaling(x)[2].any():
        raise ValueError("No varying development observation")
    folds = _folds(len(x), seed)
    inverse_grid = {
        "ridge": [(a,) for a in (4., 16., 64., 256.)],
        "rbf": list(product((.60, 1.20, 2.40), (.015, .15, 1.5))),
    }
    result = {"inverse": {}, "metadata": {"code_order": COORDINATES, "bound": LIMIT}}
    for kind, params in inverse_grid.items():
        choices = []
        for hp in params:
            means = np.empty_like(g)
            for train, test in folds:
                means[test] = _inverse_mean(_fit_inverse(x[train], g[train], kind, hp), x[test])
            loss = float(np.square(np.clip(means, -LIMIT, LIMIT)-g).mean())
            choices.append((loss, hp, means))
        mse, hp, means = min(choices, key=lambda model: model[0])
        residual = g - np.clip(means, -LIMIT, LIMIT)
        covariance = np.cov(residual.T)
        covariance = .85*covariance+.15*np.diag(np.diag(covariance))+.015**2*np.eye(4)
        result["inverse"][kind] = (_fit_inverse(x, g, kind, hp), np.linalg.cholesky(covariance))
        result["metadata"][kind] = {
            "parameters": hp, "cv_mean_squared": mse, "oof_residual_covariance": covariance.tolist(),
        }

    # Equally weighted quasi-random quadrature represents the declared uniform
    # material prior. Its candidate points and seeds never use validation cards.
    grid = (2*qmc.Sobol(4, scramble=True, seed=seed+2).random_base2(12)-1.) * LIMIT
    dev_choices = []
    for degree, shrinkage in product((1, 2), (.35, .80)):
        fold_distances = []
        for train, test in folds:
            fit = _fit_response(x[train], g[train], degree, shrinkage)
            fold_distances.append((_distance(fit, x[test], grid), test))
        for temperature in (.75, 1.5, 3.):
            scores = []
            for dist, test in fold_distances:
                weight = softmax(-dist/(2*temperature), axis=1)
                a = _draw_discrete(weight, grid, 128, np.random.default_rng(seed+11))
                b = _draw_discrete(weight, grid, 128, np.random.default_rng(seed+12))
                scores.append(_energy(a, b, g[test]))
            dev_choices.append((float(np.mean(scores)),degree,shrinkage,temperature))
    loss, degree, shrinkage, temperature = min(dev_choices)
    response = _fit_response(x, g, degree, shrinkage)
    result["shared_response"] = (response, grid, temperature)
    result["metadata"]["shared_response"] = {
        "degree": degree, "covariance_shrinkage": shrinkage, "temperature": temperature,
        "projected_rank": response[4].shape[1], "cv_code_energy": loss, "support_count": len(grid),
        "selection_scores": dev_choices,
    }
    return result


def predict_codes(fitted, X_query, n_draws=32, seed=80186):
    """Use only query A/B prefixes to generate out-of-card code draws."""
    if not isinstance(n_draws, (int, np.integer)) or n_draws < 1:
        raise ValueError("n_draws must be a positive integer")
    x = _observations(X_query)
    result = {}
    for kind, (fit, factor) in fitted["inverse"].items():
        result[kind] = _posterior_draws(_inverse_mean(fit, x), factor, n_draws,
                                        np.random.default_rng(seed))
    fit, grid, temperature = fitted["shared_response"]
    weights = softmax(-_distance(fit, x, grid)/(2*temperature), axis=1)
    result["shared_response"] = _draw_discrete(weights, grid, n_draws, np.random.default_rng(seed))
    return result


def encode(X_train, G_train, X_query, n_draws=32, seed=80123):
    fitted = fit_encoder(X_train, G_train, seed=seed)
    return predict_codes(fitted, X_query, n_draws=n_draws, seed=seed+1)
