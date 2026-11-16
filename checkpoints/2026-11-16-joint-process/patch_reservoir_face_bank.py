"""Batch corrected-face trajectories conditional on a common prepared patch."""

import numpy as np
from numba import njit

from patch_reservoir_service import bound_after_preparation, water_frames


def _hydraulic_bank(parent, actions, cfg, bath, cells, steps, step):
    """Precompute the corrected water and ion flux coefficients once per coating."""
    dx = parent['thickness'] / cells
    transfers = np.array([a['wet_transfer'] for a in actions])
    resistance = transfers * dx / (2 * parent['water_D'])
    effective, coating = np.unique(transfers / (1 + resistance), return_inverse=True)
    half_cell = effective * dx / (2 * parent['water_D'])
    times = np.arange(2 * steps + 1) * step / 2
    water = np.ascontiguousarray(water_frames(parent, effective, bath, times, cells).transpose(1, 0, 2))
    # h_surface = h_center + J_water*dx/(2*D_water), where J_water = K_eff*(a-h_center).
    surface = water[..., 0] + half_cell[:, None] * (bath - water[..., 0])
    mobility = cfg['mobile_D'] * (.10 + .90 * water)
    ion_faces = 2 / (1 / mobility[..., :-1] + 1 / mobility[..., 1:]) / dx**2
    ion_resistance = dx / (2 * mobility[..., 0])
    return coating, water, surface, ion_resistance, ion_faces


@njit(cache=True)
def _mechanical_trace(water, indices, step, bond_on, bond_off):
    n_coatings, _, cells = water.shape
    output = np.empty((len(indices), n_coatings, 3))
    for a in range(n_coatings):
        bonds = np.ones(cells)
        sample = 0
        for tick in range(indices[-1] + 1):
            if tick == indices[sample]:
                h = water[a, 2 * tick]
                stiffness = (.27 + .93 * bonds) * (1 - .27 * h)
                output[sample, a, 0] = h.mean()
                output[sample, a, 1] = stiffness[:cells // 2].mean()
                output[sample, a, 2] = stiffness[cells // 2:].mean()
                sample += 1
            if tick < indices[-1]:
                h_mid = water[a, 2 * tick + 1]
                on = bond_on * (1 - h_mid)
                off = bond_off * h_mid
                equilibrium = on / (on + off)
                bonds = equilibrium + (bonds - equilibrium) * np.exp(-step * (on + off))
    return output


@njit(cache=True, inline='always')
def _rates(reservoir, mobile, reaction, faces, loss_coefficient, dx, dr, dc):
    for cell in range(len(reservoir)):
        released = reaction[cell] * reservoir[cell]
        dr[cell] = -released
        dc[cell] = released
    for cell in range(len(reservoir) - 1):
        flux = faces[cell] * (mobile[cell + 1] - mobile[cell])
        dc[cell] += flux
        dc[cell + 1] -= flux
    leaving = loss_coefficient * mobile[0]
    dc[0] -= leaving / dx
    return leaving


@njit(cache=True)
def _ion_trace(powered_water, water, surface, resistance, faces, stock, coating,
               indices, parameters, dx, step, conductivity):
    cells = stock.shape[-1]
    output = np.empty((len(parameters), len(indices), len(stock)))
    budget_error = 0.
    minimum_stock = 0.
    dry, baseline, ion = conductivity
    for hypothesis, (rate, loss) in enumerate(parameters):
        for action, channel in enumerate(coating):
            bound = stock[action].copy()
            mobile = np.zeros(cells)
            moved_bound = np.empty(cells)
            moved_mobile = np.empty(cells)
            dr = np.empty((4, cells))
            dc = np.empty((4, cells))
            exits = np.empty(4)
            initial_budget = bound.sum() * dx
            escaped = 0.
            sample = 0
            for tick in range(indices[-1] + 1):
                if tick == indices[sample]:
                    h = water[channel, 2 * tick]
                    sigma = dry + baseline * h + ion * h * mobile
                    output[hypothesis, sample, action] = 1 / (1 / sigma).mean()
                    sample += 1
                budget_error = max(budget_error, abs((bound.sum() + mobile.sum()) * dx + escaped - initial_budget))
                minimum_stock = min(minimum_stock, bound.min(), mobile.min(), escaped)
                if tick == indices[-1]:
                    break
                for stage in range(4):
                    offset = 2 * tick + (0 if stage == 0 else 2 if stage == 3 else 1)
                    if stage == 0:
                        stage_bound, stage_mobile = bound, mobile
                    else:
                        fraction = 1. if stage == 3 else .5
                        for cell in range(cells):
                            moved_bound[cell] = bound[cell] + fraction * step * dr[stage - 1, cell]
                            moved_mobile[cell] = mobile[cell] + fraction * step * dc[stage - 1, cell]
                        stage_bound, stage_mobile = moved_bound, moved_mobile
                    loss_at_surface = loss * surface[channel, offset]
                    loss_coefficient = loss_at_surface / (1 + loss_at_surface * resistance[channel, offset])
                    exits[stage] = _rates(stage_bound, stage_mobile, rate * powered_water[channel, offset],
                                          faces[channel, offset], loss_coefficient, dx, dr[stage], dc[stage])
                for cell in range(cells):
                    bound[cell] += step * (dr[0, cell] + 2 * dr[1, cell] + 2 * dr[2, cell] + dr[3, cell]) / 6
                    mobile[cell] += step * (dc[0, cell] + 2 * dc[1, cell] + 2 * dc[2, cell] + dc[3, cell]) / 6
                escaped += step * (exits[0] + 2 * exits[1] + 2 * exits[2] + exits[3]) / 6
    return output, budget_error, minimum_stock


def bank(parent, actions, cfg, bath, hypotheses, cells, step, times):
    """Return [hypothesis, time, action, channel] traces for absolute k, gamma and loss.

    Channels contain mean water, near-half stiffness, far-half stiffness and
    harmonic conductivity. Time integration uses the corrected face Robin
    model, fourth-order Runge–Kutta ion updates and midpoint bond relaxation.
    """
    hypotheses = np.asarray(hypotheses, dtype=float)
    times = np.asarray(times, dtype=float)
    if (hypotheses.ndim != 2 or hypotheses.shape[1:] != (3,) or not len(hypotheses)
            or not np.isfinite(hypotheses).all() or (hypotheses[:, :2] <= 0).any() or (hypotheses[:, 2] < 0).any()):
        raise ValueError('Hypotheses require positive rates and powers and nonnegative loss.')
    if (cells < 8 or cells % 2 or not np.isfinite(step) or step <= 0 or times.ndim != 1 or not len(times)
            or not np.isfinite(times).all() or not actions or not 0 <= bath <= 1):
        raise ValueError('Use an even aligned mesh, finite positive step, output times and candidate actions.')
    indices = np.rint(times / step).astype(np.int64)
    if indices[0] or np.any(np.diff(indices) <= 0) or not np.allclose(times, indices * step, rtol=0, atol=1e-12):
        raise ValueError('Output times must strictly increase from zero on the integration grid.')
    dx = parent['thickness'] / cells
    if step * max(4 * cfg['mobile_D'] / dx**2 + hypotheses[:, 2].max() / dx, hypotheses[:, 0].max()) > 2.4:
        raise ValueError('Reduce the time step for mobile-ion diffusion or fast release.')

    coating, water, surface, resistance, faces = _hydraulic_bank(parent, actions, cfg, bath, cells, indices[-1], step)
    stock = bound_after_preparation([a['near_loading'] for a in actions], cfg, parent['thickness'], cells)
    expected = cfg['dose'] * -np.expm1(-cfg['process_time'] * cfg['process_k'])
    if np.abs(stock.mean(axis=-1) - expected).max() > 2e-11:
        raise ArithmeticError('Prepared stock violates the closed mold inventory.')
    output = np.empty((len(hypotheses), len(times), len(actions), 4))
    output[..., :3] = _mechanical_trace(water, indices, step, cfg['bond_on'], cfg['bond_off'])[:, coating]
    conductivity = (cfg['dry_conductivity'], cfg['water_background'], cfg['ion_conductivity'])
    for power in np.unique(hypotheses[:, 1]):
        selected = hypotheses[:, 1] == power
        params = hypotheses[selected][:, [0, 2]].copy(order='C')
        values, error, minimum = _ion_trace(water**power, water, surface, resistance, faces, stock, coating,
                                           indices, params, dx, step, conductivity)
        if not np.isfinite(values).all() or error > 2e-11 * max(1., parent['thickness']) or minimum < -1e-10:
            raise ArithmeticError('Ion evolution violated positivity or the conservative mass budget.')
        output[selected, ..., 3] = values
    return output
