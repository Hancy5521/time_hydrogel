"""Corrected-face service trajectories for unknown preparation and ion transport."""

import numpy as np
from numba import njit
from scipy.special import expit, logit

from patch_reservoir_face_bank import _hydraulic_bank, _mechanical_trace, _rates
from patch_reservoir_service import bound_after_preparation


def prepared_fields(parent, actions, cfg, stock_modes, cells):
    """Return prepared stock over hypotheses, actions and depth cells.

    Each hypothesis specifies a near-side logit loading bias b and a coefficient
    v for a cosine redistribution within each prepared half. The redistribution
    preserves the two half-layer integrals separately, as well as total stock.
    """
    modes = np.asarray(stock_modes, dtype=float)
    if (modes.ndim != 2 or modes.shape[1] != 2 or not len(modes)
            or not np.isfinite(modes).all() or (np.abs(modes[:, 1]) >= 1).any()):
        raise ValueError('Stock modes require finite loading logits and |v| < 1.')
    if cells < 8 or cells % 2 or not actions:
        raise ValueError('Use an even mesh aligned with the two halves and nonempty actions.')
    command = np.array([action['near_loading'] for action in actions])
    if not np.isfinite(command).all() or ((command < 0) | (command > 1)).any():
        raise ValueError('Loading commands must be finite shares in [0, 1].')
    # Preserve the exact old input in the no-bias case.
    share = np.where(modes[:, :1] == 0, command, expit(logit(command) + modes[:, :1]))
    nominal = bound_after_preparation(share.ravel(), cfg, parent['thickness'], cells)
    nominal = nominal.reshape(len(modes), len(actions), 2, cells // 2)
    basis = np.cos(4*np.pi*(np.arange(cells) + .5)/cells).reshape(2, cells // 2)
    redistributed = nominal*(1 + modes[:, 1, None, None, None]*basis)
    mass = nominal.sum(axis=-1, keepdims=True)
    mass_raw = redistributed.sum(axis=-1, keepdims=True)
    scale = np.divide(mass, mass_raw, out=np.ones_like(mass), where=mass_raw > 0)
    stock = (redistributed*scale).reshape(len(modes), len(actions), cells)
    expected = cfg['dose'] * -np.expm1(-cfg['process_time'] * cfg['process_k'])
    if (not np.isfinite(stock).all() or stock.min() < -1e-12 or expected <= 0
            or np.abs(stock.mean(axis=-1) - expected).max() > 2e-11):
        raise ArithmeticError('Prepared field violates the closed-mold inventory.')
    return np.maximum(stock, 0.)


@njit(cache=True)
def _joint_ion_trace(powered_water, water, surface, resistance, faces, stocks, coating,
                     indices, parameters, dx, step, conductivity, store_inventory):
    n_hypotheses, n_actions, cells = stocks.shape
    output = np.empty((n_hypotheses, len(indices), n_actions))
    inventory = np.empty((n_hypotheses, len(indices) if store_inventory else 0, n_actions, 3))
    budget_error = 0.
    minimum_stock = 0.
    dry, baseline, ion = conductivity
    for hypothesis, (rate, loss) in enumerate(parameters):
        for action, channel in enumerate(coating):
            bound = stocks[hypothesis, action].copy()
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
                    h = water[channel, 2*tick]
                    sigma = dry + baseline*h + ion*h*mobile
                    output[hypothesis, sample, action] = 1 / (1/sigma).mean()
                    if store_inventory:
                        inventory[hypothesis, sample, action, 0] = bound.sum()*dx / initial_budget
                        inventory[hypothesis, sample, action, 1] = mobile.sum()*dx / initial_budget
                        inventory[hypothesis, sample, action, 2] = escaped / initial_budget
                    sample += 1
                budget_error = max(budget_error, abs((bound.sum() + mobile.sum())*dx + escaped - initial_budget))
                minimum_stock = min(minimum_stock, bound.min(), mobile.min(), escaped)
                if tick == indices[-1]:
                    break
                for stage in range(4):
                    offset = 2*tick + (0 if stage == 0 else 2 if stage == 3 else 1)
                    if stage == 0:
                        stage_bound, stage_mobile = bound, mobile
                    else:
                        fraction = 1. if stage == 3 else .5
                        for cell in range(cells):
                            moved_bound[cell] = bound[cell] + fraction*step*dr[stage - 1, cell]
                            moved_mobile[cell] = mobile[cell] + fraction*step*dc[stage - 1, cell]
                        stage_bound, stage_mobile = moved_bound, moved_mobile
                    loss_at_surface = loss*surface[channel, offset]
                    loss_coefficient = loss_at_surface/(1 + loss_at_surface*resistance[channel, offset])
                    exits[stage] = _rates(stage_bound, stage_mobile, rate*powered_water[channel, offset],
                                          faces[channel, offset], loss_coefficient, dx, dr[stage], dc[stage])
                for cell in range(cells):
                    bound[cell] += step*(dr[0, cell] + 2*dr[1, cell] + 2*dr[2, cell] + dr[3, cell])/6
                    mobile[cell] += step*(dc[0, cell] + 2*dc[1, cell] + 2*dc[2, cell] + dc[3, cell])/6
                escaped += step*(exits[0] + 2*exits[1] + 2*exits[2] + exits[3])/6
    return output, inventory, budget_error, minimum_stock


def joint_bank(parent, actions, cfg, bath, kinetics, stock_modes, cells, step, times, inventory=False):
    """Return [hypothesis,time,action,channel] traces and optional stock fractions.

    Kinetics columns are absolute k, gamma, Robin loss, and added dimensionless
    ion interface resistance rho. The added face resistance has units and
    value rho*thickness/mobile_D independent of the numerical grid. Water and
    mechanical models do not depend on the ion or preparation hypotheses.
    """
    kinetics = np.asarray(kinetics, dtype=float)
    times = np.asarray(times, dtype=float)
    if (kinetics.ndim != 2 or kinetics.shape[1] != 4 or not len(kinetics)
            or not np.isfinite(kinetics).all() or (kinetics[:, :2] <= 0).any()
            or (kinetics[:, 2:] < 0).any() or len(kinetics) != len(stock_modes)):
        raise ValueError('Specify positive k and gamma, nonnegative loss and rho, and paired stock modes.')
    if (cells < 8 or cells % 2 or not actions or not np.isfinite(step) or step <= 0 or times.ndim != 1
            or not len(times) or not np.isfinite(times).all() or not 0 <= bath <= 1):
        raise ValueError('Use an even aligned mesh, valid environment, actions and on-grid output times.')
    indices = np.rint(times/step).astype(np.int64)
    if indices[0] or (np.diff(indices) <= 0).any() or not np.allclose(times, indices*step, rtol=0, atol=1e-12):
        raise ValueError('Output times must strictly increase from zero on the integration grid.')
    thickness = parent['thickness']
    dx = thickness/cells
    if step*max(4*cfg['mobile_D']/dx**2 + kinetics[:, 2].max()/dx, kinetics[:, 0].max()) > 2.4:
        raise ValueError('Reduce the time step for mobile-ion diffusion or fast release.')

    coating, water, surface, resistance, faces = _hydraulic_bank(parent, actions, cfg, bath, cells, indices[-1], step)
    stock = prepared_fields(parent, actions, cfg, stock_modes, cells)
    output = np.empty((len(kinetics), len(times), len(actions), 4))
    output[..., :3] = _mechanical_trace(water, indices, step, cfg['bond_on'], cfg['bond_off'])[:, coating]
    fractions = np.empty(output.shape[:-1] + (3,)) if inventory else None
    conductivity = (cfg['dry_conductivity'], cfg['water_background'], cfg['ion_conductivity'])
    midplane = cells//2 - 1
    for power in np.unique(kinetics[:, 1]):
        powered = water**power
        for rho in np.unique(kinetics[kinetics[:, 1] == power, 3]):
            selected = (kinetics[:, 1] == power) & (kinetics[:, 3] == rho)
            interface = faces
            if rho > 0:
                interface = faces.copy()
                # The stored face coefficients equal D_face/dx^2.
                added = rho*thickness*dx/cfg['mobile_D']
                interface[..., midplane] /= 1 + added*interface[..., midplane]
            values, mass, error, minimum = _joint_ion_trace(powered, water, surface, resistance, interface,
                stock[selected], coating, indices, kinetics[selected][:, [0, 2]].copy(order='C'),
                dx, step, conductivity, inventory)
            if not np.isfinite(values).all() or error > 2e-11*max(1., thickness) or minimum < -1e-10:
                raise ArithmeticError('Joint trajectories violated positivity or mass conservation.')
            output[selected, ..., 3] = values
            if inventory:
                fractions[selected] = mass
    return (output, fractions) if inventory else output
