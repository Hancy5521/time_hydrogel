"""Conservative fabrication-to-service model for an internally loaded patch."""

import numpy as np
from scipy.fft import dct, idct
from scipy.linalg import eigh_tridiagonal


def bound_after_preparation(shares, cfg, thickness, cells):
    """Trap a precursor in a closed mold, then rinse away its free fraction."""
    shares = np.asarray(shares, dtype=float)
    half = np.repeat(np.column_stack((shares, 1 - shares)), cells // 2, axis=1)
    initial = 2 * cfg['dose'] * half
    modes = -4 * (np.sin(np.pi * np.arange(cells) / (2 * cells)) * cells / thickness) ** 2
    rate = cfg['process_k'] - cfg['process_D'] * modes
    trapped = -cfg['process_k'] * np.expm1(-cfg['process_time'] * rate) / rate
    return idct(trapped * dct(initial, norm='ortho', axis=-1), norm='ortho', axis=-1)


def water_frames(parent, transfer, bath, times, cells):
    """Solve one-sided linear wetting for all prescribed Robin coefficients."""
    dx = parent['thickness'] / cells
    rate = parent['water_D'] / dx**2
    face = np.full(cells - 1, rate)
    diagonal = -np.r_[0., face] - np.r_[face, 0.]
    output = []
    for coefficient in np.unique(transfer):
        wet = diagonal.copy()
        wet[0] -= coefficient / dx
        values, vectors = eigh_tridiagonal(wet, face)
        decays = np.exp(np.outer(times, values))
        u = bath * (1 - (decays * vectors.sum(axis=0)) @ vectors.T)
        output.append(u.clip(0, bath))
    return np.stack(output, axis=1)


def simulate(parent, actions, cfg, bath, power, loss, cells, step, times):
    """Return time × action × cell fields and per-area ion loss for a fixed bath."""
    if cells < 8 or cells % 2:
        raise ValueError('Use an even mesh aligned with the loaded halves.')
    times = np.asarray(times)
    index = np.rint(times / step).astype(int)
    if (not len(times) or times[0] != 0 or np.any(np.diff(times) <= 0)
            or not np.allclose(times, index * step, rtol=0, atol=1e-12)):
        raise ValueError('Times must increase from zero on the integration grid.')
    dx = parent['thickness'] / cells
    if step * (4 * cfg['mobile_D'] / dx**2 + loss / dx) > 2.4:
        raise ValueError('Explicit ion step is too large for the registered mesh.')
    transfers = [a['wet_transfer'] for a in actions]
    _, channel = np.unique(transfers, return_inverse=True)
    hydration = water_frames(parent, transfers, bath, np.arange(2*index[-1] + 1)*step/2, cells)
    reservoir = bound_after_preparation([a['near_loading'] for a in actions], cfg, parent['thickness'], cells)
    ions = np.zeros_like(reservoir)
    bonds = np.ones_like(reservoir)
    lost = np.zeros(len(actions))
    record = {key: [] for key in ('h', 'b', 'r', 'c', 'lost')}

    def rate(r, c, h, release_rate):
        mobilized = release_rate * r
        mobile_diffusivity = cfg['mobile_D'] * (.10 + .90*h)
        faces = 2 / (1/mobile_diffusivity[:, :-1] + 1/mobile_diffusivity[:, 1:])
        flux = faces * np.diff(c, axis=-1) / dx**2
        loss_rate = loss * h[:, 0] * c[:, 0]
        free = mobilized.copy()
        free[:, :-1] += flux
        free[:, 1:] -= flux
        free[:, 0] -= loss_rate/dx
        return -mobilized, free, loss_rate

    stops = set(index)
    for tick in range(index[-1] + 1):
        if tick in stops:
            for key, value in zip(record, (hydration[2*tick, channel], bonds, reservoir, ions, lost), strict=True):
                record[key].append(value.copy())
        if tick == index[-1]:
            break
        lo, mid, hi = hydration[2*tick:2*tick + 3, channel]
        exposure = parent['release'] * np.array([lo, mid, hi])**power
        r1, c1, l1 = rate(reservoir, ions, lo, exposure[0])
        r2, c2, l2 = rate(reservoir + step*r1/2, ions + step*c1/2, mid, exposure[1])
        r3, c3, l3 = rate(reservoir + step*r2/2, ions + step*c2/2, mid, exposure[1])
        r4, c4, l4 = rate(reservoir + step*r3, ions + step*c3, hi, exposure[2])
        reservoir += step*(r1 + 2*r2 + 2*r3 + r4)/6
        ions += step*(c1 + 2*c2 + 2*c3 + c4)/6
        lost += step*(l1 + 2*l2 + 2*l3 + l4)/6
        on = cfg['bond_on'] * (1-mid)
        off = cfg['bond_off'] * mid
        eq = on/(on+off)
        bonds = eq + (bonds-eq) * np.exp(-step*(on+off))

    fields = {key: np.stack(value) for key, value in record.items()}
    expected = cfg['dose'] * -np.expm1(-cfg['process_time'] * cfg['process_k'])
    balance = (fields['r']+fields['c']).mean(axis=-1) + fields['lost']/parent['thickness']
    if (max(np.abs(balance - expected).flat) > 2e-11 or min(fields['c'].min(), fields['r'].min()) < -1e-10
            or not all(np.isfinite(value).all() for value in fields.values())):
        raise ArithmeticError('Unphysical concentration or nonconservative ion budget.')
    return fields


def traces(fields, cfg):
    h, b, c = (fields[key] for key in ('h', 'b', 'c'))
    stiffness = (.27 + .93*b) * (1-.27*h)
    sigma = cfg['dry_conductivity'] + cfg['water_background']*h + cfg['ion_conductivity']*h*c
    conductivity = 1 / (1/sigma).mean(axis=-1)
    half = stiffness.shape[-1]//2
    return np.stack((h.mean(axis=-1), stiffness[..., :half].mean(axis=-1),
                     stiffness[..., half:].mean(axis=-1), conductivity), axis=-1)
