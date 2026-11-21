"""Train-only time-ordered innovations for conditional energy coordinates."""

from dataclasses import dataclass

import numpy as np


def camera_rows(value):
    x = np.asarray(value, dtype=float)
    if x.ndim != 3 or any(size == 0 for size in x.shape) or not np.isfinite(x).all():
        raise ValueError('Expected a nonempty batch of finite, ordered camera channels.')
    return x


def new_features(standardized, time):
    """Append terms that first become observable at this camera time."""
    current = standardized[:, time, :]
    i, j = np.triu_indices(current.shape[1])
    products = current[:, i]*current[:, j]
    past = standardized[:, :time, :].reshape(len(current), -1)
    crosses = (current[:, :, None]*past[:, None, :]).reshape(len(current), -1)
    return np.column_stack((current, products, crosses))


@dataclass(frozen=True)
class TimeBlock:
    residual_projection: np.ndarray
    residual_scale: np.ndarray
    coordinate_coefficients: np.ndarray


@dataclass(frozen=True)
class NestedCameraMap:
    camera_mean: np.ndarray
    camera_scale: np.ndarray
    target_mean: np.ndarray
    blocks: tuple

    def innovations(self, measured):
        measured = camera_rows(measured)
        if measured.shape[1:] != self.camera_mean.shape:
            raise ValueError('Forecasts must use the registered camera times and channels.')
        x = (measured-self.camera_mean)/self.camera_scale
        parent = np.ones((len(x), 1))
        pieces = []
        for time, block in enumerate(self.blocks):
            added = new_features(x, time)
            residual = (added-parent@block.residual_projection)/block.residual_scale
            pieces.append(residual@block.coordinate_coefficients)
            parent = np.column_stack((parent, residual))
        return np.stack(pieces, axis=1)


def fit_nested_camera_map(measured, coordinates, noise, ridge):
    """Fit disjoint innovations on training snapshots and whole-history targets."""
    x = camera_rows(measured)
    y = np.asarray(coordinates, dtype=float)
    noise = np.asarray(noise, dtype=float)
    if (y.ndim != 2 or len(y) != len(x) or not np.isfinite(y).all()
            or noise.shape not in ((x.shape[2],), x.shape[1:])
            or not np.isfinite(noise).all() or np.any(noise <= 0.) or not np.isfinite(ridge) or ridge <= 0.):
        raise ValueError('Training requires matching coordinates, noise and positive ridge.')
    x_mean, x_scale = x.mean(axis=0), np.maximum(x.std(axis=0), noise)
    standardized = (x-x_mean)/x_scale
    y_mean = y.mean(axis=0)
    centered = y-y_mean
    parent, blocks = np.ones((len(x), 1)), []
    for time in range(x.shape[1]):
        added = new_features(standardized, time)
        projection = np.linalg.lstsq(parent, added, rcond=1.e-10)[0]
        residual = added-parent@projection
        residual_scale = np.maximum(residual.std(axis=0), .05)
        residual /= residual_scale
        gram = residual.T@residual/len(x)
        gram.flat[::len(gram)+1] += ridge
        rhs = residual.T@centered/len(x)
        coef = np.linalg.solve(gram, rhs)
        blocks.append(TimeBlock(projection, residual_scale, coef))
        parent = np.column_stack((parent, residual))
    return NestedCameraMap(x_mean, x_scale, y_mean, tuple(blocks))


def fixtures():
    rng = np.random.default_rng(2032)
    x = rng.normal(size=(512, 3, 4))
    z = np.column_stack([x[:, 0, 1]+2*x[:, 1, 2], x[:, 1, 0]*x[:, 2, 3]])
    model = fit_nested_camera_map(x, z, np.ones(4)*.1, 1.e-8)
    learned = model.innovations(x)
    np.testing.assert_allclose(learned.mean(axis=0), 0., atol=1.e-12)
    np.testing.assert_allclose(learned.sum(axis=1)+model.target_mean, z, atol=1.e-5)
    changed = x.copy()
    changed[:, 2] = rng.normal(size=(len(x), 4))
    np.testing.assert_array_equal(learned[:, :2], model.innovations(changed)[:, :2])
    null = fit_nested_camera_map(x, np.repeat(z.mean(axis=0)[None, :], len(x), axis=0), np.ones(4)*.1, 1.e-3)
    np.testing.assert_allclose(null.innovations(x), 0., atol=1.e-12)
    other = fit_nested_camera_map(x[:, :2, :3], z, np.ones((2, 3))*.1, .20)
    assert other.innovations(x[:, :2, :3]).shape == (len(x),2,2)
    print('PASS nested temporal innovations, strict causality and constant target.')


if __name__ == '__main__':
    fixtures()
