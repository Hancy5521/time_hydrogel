"""Independent method-of-lines audit for preparation and wet service of an ion reservoir."""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.sparse import lil_matrix


def _net_face_flow(field, face_diffusivity, dx):
    """Convert pairwise cell-to-cell fluxes to conservative volumetric rates."""
    across = face_diffusivity * (field[:-1] - field[1:]) / dx
    return (np.r_[0.0, across] - np.r_[across, 0.0]) / dx


def _integrate(rhs, times, start, sparsity=None):
    if times[-1] == 0:
        return np.repeat(start[None, :], len(times), axis=0)
    unique_times, indices = np.unique(times, return_inverse=True)
    solution = solve_ivp(
        rhs, (0.0, unique_times[-1]), start, t_eval=unique_times,
        method='Radau', rtol=2e-10, atol=2e-12, jac_sparsity=sparsity,
    )
    if not solution.success:
        raise ArithmeticError(solution.message)
    return solution.y.T[indices]


def _local_sparsity(cells, fields, boundary_integral=False):
    size = cells * fields + int(boundary_integral)
    pattern = lil_matrix((size, size), dtype=bool)
    for cell in range(cells):
        rows = slice(cell * fields, (cell + 1) * fields)
        columns = slice(max(0, cell - 1) * fields, min(cells, cell + 2) * fields)
        pattern[rows, columns] = True
    if boundary_integral:
        pattern[-1, :fields] = True
    return pattern.tocsc()


def independent_curve(parent, action, cfg, times, bath, cells, power, loss):
    """Recompute h, b, r, c and cumulative external loss using a separate solver.

    Spatially constant geometry and closed boundaries are assumed during preparation.
    For service, only the near face exchanges water and loses mobile ions to the bath.
    The return value has four [time, cell] fields and a [time] area-integrated loss.
    """
    times = np.asarray(times, dtype=float)
    if (times.ndim != 1 or not len(times) or not np.isfinite(times).all()
            or times[0] < 0 or np.any(np.diff(times) < 0)):
        raise ValueError('Expected a nonempty, finite, nondecreasing, nonnegative time grid.')
    if cells < 2 or cells % 2 or parent['thickness'] <= 0:
        raise ValueError('Expected positive thickness and an even mesh of at least two cells.')
    if not 0 <= bath <= 1 or not 0 <= action['near_loading'] <= 1 or power <= 0:
        raise ValueError('Expected bounded bath and loading fractions and positive release power.')

    dx = parent['thickness'] / cells
    near = action['near_loading']
    precursor = 2 * cfg['dose'] * np.repeat([near, 1 - near], cells // 2)
    preparation_start = np.column_stack((precursor, np.zeros(cells))).ravel()

    def preparation_rhs(_, flat):
        precursor_state, _ = flat.reshape(cells, 2).T
        trapped = cfg['process_k'] * precursor_state
        mobile = _net_face_flow(precursor_state, cfg['process_D'], dx)
        return np.column_stack((mobile - trapped, trapped)).ravel()

    formed = _integrate(preparation_rhs, [cfg['process_time']], preparation_start,
                        _local_sparsity(cells, 2))[-1].reshape(cells, 2)[:, 1]
    fields = np.column_stack((np.zeros(cells), np.ones(cells), formed, np.zeros(cells)))

    def service_rhs(_, flat):
        water, bonds, reservoir, mobile = flat[:-1].reshape(cells, 4).T
        wet = np.clip(water, 0, 1)
        water_rate = _net_face_flow(water, parent['water_D'], dx)
        water_rate[0] += action['wet_transfer'] * (bath - water[0]) / dx
        bond_rate = cfg['bond_on'] * (1 - wet) * (1 - bonds) - cfg['bond_off'] * wet * bonds
        released = parent['release'] * wet**power * reservoir
        diffusivity = 0.10 + 0.90 * wet
        face_diffusivity = 2 * cfg['mobile_D'] / (1 / diffusivity[:-1] + 1 / diffusivity[1:])
        mobile_rate = _net_face_flow(mobile, face_diffusivity, dx) + released
        loss_rate = loss * wet[0] * mobile[0]
        mobile_rate[0] -= loss_rate / dx
        return np.r_[np.column_stack((water_rate, bond_rate, -released, mobile_rate)).ravel(),
                     loss_rate]

    solution = _integrate(service_rhs, times, np.r_[fields.ravel(), 0.0],
                          _local_sparsity(cells, 4, boundary_integral=True))
    trajectories = solution[:, :-1].reshape(len(times), cells, 4)
    return {'h': trajectories[..., 0], 'b': trajectories[..., 1],
            'r': trajectories[..., 2], 'c': trajectories[..., 3], 'lost': solution[:, -1]}
