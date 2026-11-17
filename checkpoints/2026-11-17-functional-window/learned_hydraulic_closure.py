"""Learn a positive water diffusivity curve from finite-volume mass balances."""

import numpy as np
from numba import njit
from scipy.optimize import least_squares


KNOTS = np.array([0., .25, .50, .75, 1.])


def fit_face_law(records):
    """Each training record gives dense synthetic water fractions, dx, dt, and nominal D."""
    rows = []
    for water, dx, dt, reference in records:
        centered = water[1:-1]
        backward_mass = np.cumsum(water[:, ::-1], axis=1)[:, ::-1] * dx
        flux = (backward_mass[2:, 1:] - backward_mass[:-2, 1:]) / (2 * dt)
        gradient = (centered[:, :-1] - centered[:, 1:]) / dx
        good = (gradient > .025) & (flux > 0.)
        left, right = centered[:, :-1][good], centered[:, 1:][good]
        observed = np.log(flux[good] / (gradient[good] * reference))
        rows.append((left, right, observed, np.full(len(left), 1. / np.sqrt(len(left)))))
    h0, h1, y, weight = [np.concatenate(x) for x in zip(*rows, strict=True)]

    def residual(curve):
        d0 = np.exp(np.interp(h0, KNOTS, curve))
        d1 = np.exp(np.interp(h1, KNOTS, curve))
        effective = np.log(2. / (1./d0 + 1./d1))
        return np.r_[weight*(effective-y), .12*np.diff(curve, 2)]

    result = least_squares(residual, np.linspace(-.6, 0., len(KNOTS)), bounds=(-2.2, .8))
    if not result.success or not np.isfinite(result.x).all():
        raise ArithmeticError('The bounded face-flux fit did not converge.')
    return result.x


@njit(cache=True, inline='always')
def _face_rates(h, b, rh, rb, film, bath, bonds, dx, nominal, logcurve):
    d_left = nominal * np.exp(np.interp(h[0], KNOTS, logcurve))
    wet_face = film / (1. + film * dx / (2. * d_left))
    rh[0] = wet_face * (bath - h[0]) / dx
    for j in range(len(h)-1):
        d_right = nominal * np.exp(np.interp(h[j+1], KNOTS, logcurve))
        inward = 2. * (h[j+1]-h[j]) / ((1./d_left + 1./d_right)*dx**2)
        rh[j] += inward
        rh[j+1] = -inward
        d_left = d_right
    on, off = bonds
    for j in range(len(h)):
        rb[j] = on*(1.-h[j])*(1.-b[j]) - off*h[j]*b[j]


@njit(cache=True)
def law_path(cells, sample_count, substeps, sample_dt, dx, nominal, logcurve, film, bath, bonds):
    """Conservative two-stage stepping with symmetric interior face fluxes."""
    h = np.zeros(cells)
    b = np.ones(cells)
    rh, rb, hp, bp = (np.empty(cells), np.empty(cells), np.empty(cells), np.empty(cells))
    values = np.empty((sample_count, cells+2))
    dt = sample_dt/substeps
    for t in range(sample_count):
        values[t,:cells] = h
        modulus = (.27 + .93*b)*(1.-.27*h)
        values[t,cells] = modulus[:cells//2].mean()
        values[t,cells+1] = modulus[cells//2:].mean()
        if t == sample_count-1:
            break
        for _ in range(substeps):
            _face_rates(h,b,rh,rb,film,bath,bonds,dx,nominal,logcurve)
            for i in range(cells):
                hp[i], bp[i] = h[i]+dt*rh[i], b[i]+dt*rb[i]
            _face_rates(hp,bp,rh,rb,film,bath,bonds,dx,nominal,logcurve)
            for i in range(cells):
                h[i] = .5*(h[i]+hp[i]+dt*rh[i])
                b[i] = .5*(b[i]+bp[i]+dt*rb[i])
    return values


def advance(cells, count, interval, thickness, nominal, curve, film, bath, bonds):
    dx = thickness/cells
    fastest = nominal * np.exp(np.max(curve))
    bound = max(2.*fastest/dx**2, fastest/dx**2+film/dx, sum(bonds))
    steps = max(1,int(np.ceil(interval*bound/.8)))
    out = law_path(cells,count,steps,interval,dx,nominal,np.ascontiguousarray(curve),film,bath,bonds)
    if not np.isfinite(out).all() or out[:, :cells].min() < -1.e-11 or out[:, :cells].max() > bath+1.e-11:
        raise ArithmeticError('The learned diffusion path left the physical water domain.')
    return out
