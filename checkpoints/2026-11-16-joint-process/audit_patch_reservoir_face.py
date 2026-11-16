"""Independent face-based Robin audit for the autonomous reservoir model."""

import numpy as np

from audit_patch_reservoir_service import (
    _integrate, _local_sparsity, _net_face_flow, independent_curve,
)


def independent_face_curve(parent, action, cfg, times, bath, cells, power, loss):
    """Solve corrected wetting and loss boundary fluxes directly at the wet face."""
    times = np.asarray(times, dtype=float)
    if (times.ndim != 1 or not len(times) or not np.isfinite(times).all()
            or times[0] < 0 or np.any(np.diff(times) < 0)):
        raise ValueError('Expected a finite, nondecreasing, nonnegative time grid.')
    dry = independent_curve(parent, action, cfg, [0.0], bath, cells, power, loss)
    initial = np.column_stack([dry[key][0] for key in ('h', 'b', 'r', 'c')])
    dx = parent['thickness'] / cells
    d_water = parent['water_D']
    water_exchange = action['wet_transfer']
    water_half_cell = 2 * d_water / dx

    def rhs(_, flat):
        h, b, r, c = flat[:-1].reshape(cells, 4).T
        wet = np.clip(h, 0, 1)
        water_surface = (water_half_cell*h[0] + water_exchange*bath) / (water_half_cell + water_exchange)
        dh = _net_face_flow(h, d_water, dx)
        dh[0] += water_exchange * (bath - water_surface) / dx
        db = cfg['bond_on'] * (1 - wet) * (1 - b) - cfg['bond_off'] * wet * b
        released = parent['release'] * wet**power * r

        diffusivity = cfg['mobile_D'] * (0.10 + 0.90 * wet)
        face_diffusivity = 2 / (1 / diffusivity[:-1] + 1 / diffusivity[1:])
        dc = _net_face_flow(c, face_diffusivity, dx) + released
        ion_exchange = loss * water_surface
        ion_half_cell = 2 * diffusivity[0] / dx
        mobile_surface = ion_half_cell * c[0] / (ion_half_cell + ion_exchange)
        escaped = ion_exchange * mobile_surface
        dc[0] -= escaped / dx
        return np.r_[np.column_stack((dh, db, -released, dc)).ravel(), escaped]

    solution = _integrate(rhs, times, np.r_[initial.ravel(), 0.0],
                          _local_sparsity(cells, 4, boundary_integral=True))
    fields = solution[:, :-1].reshape(len(times), cells, 4)
    result = dict(zip(('h', 'b', 'r', 'c'), fields.transpose(2, 0, 1), strict=True))
    result['lost'] = solution[:, -1]
    return result
