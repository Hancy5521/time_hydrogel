"""Hydration-aware, conservative observation model for companion ion coupons."""

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from scipy.special import logsumexp, softmax


@lru_cache(maxsize=8)
def _rule(order):
    nodes, weights = np.polynomial.legendre.leggauss(order)
    return (1 + nodes)/2, weights/2


def wet_prefix(time, activity, tau, slow_share=0.):
    """Single exponential in the inference model; optional slow uptake in stress data."""
    fast = -np.expm1(-time/tau)
    slow = -np.expm1(-time/(2.8*tau))
    return activity*((1-slow_share)*fast + slow_share*slow)


def coupon_end(kinetics, activity, stop, tau, loss_factor, slow_share=0., order=24):
    """Return cumulative mobilized, bath, mobile and bound fractions, for unit stock.

    kinetics is [k, gamma, ell]. The exact integrating-factor solution is evaluated
    by nested Gaussian quadrature. Bath inventory is integrated directly to avoid
    subtractive cancellation at small ell.
    """
    rate, power, leach = np.moveaxis(np.asarray(kinetics, dtype=float), -1, 0)
    nodes, weights = _rule(order)
    tau = np.asarray(tau)
    h = lambda t: wet_prefix(t, activity, tau[..., None, None], slow_share)
    intermediate = stop*nodes
    history = h(intermediate[:, None]*nodes[None, :])
    inner = intermediate*(history**power[..., None, None] @ weights)
    water = wet_prefix(intermediate, activity, tau[..., None], slow_share)
    total = -np.expm1(-rate*stop*(water**power[..., None] @ weights))
    source = rate[..., None]*water**power[..., None]*np.exp(-rate[..., None]*inner)
    remain = (stop-intermediate)
    hydration_after = remain*(h(intermediate[:, None] + remain[:, None]*nodes[None, :]) @ weights)
    leached = -np.expm1(-loss_factor*leach[..., None]*hydration_after)
    bath = stop*((source*leached) @ weights)
    return np.stack((total, bath, total-bath, 1-total), axis=-1)


@dataclass(frozen=True)
class Read:
    """One destructive coupon and its non-destructive hydration measurements."""

    activity: float
    stop: float
    hydration: np.ndarray
    total: float
    alternate_total: float
    bath: float


def observation_table(hypotheses, cfg, activities, times):
    """Calculate means on kinetic and hydration grids without seeing any reads."""
    tau = np.asarray(cfg['tau_grid'])
    wet_times = np.array(cfg['hydration_times'])
    hydration = {a: wet_prefix(wet_times[None, :], a, tau[:, None]) for a in activities}
    values = {(a, t): coupon_end(hypotheses[:, None, :], a, t, tau[None, :], cfg['coupon_loss_factor'])
              for a, t in times}
    return hydration, values


def infer(reads, means, cfg, shared_prefix=False, extra='none'):
    """Marginalize each activity's wet-up time jointly across its coupons.

    An alternative shared-prefix model forces all activities to share one tau.
    The extra read, if present, belongs to the second coupon.
    """
    wetting, values = means
    error = cfg['sigma']
    components = {}
    for index, obs in enumerate(reads):
        modeled = values[obs.activity, obs.stop]
        water = wetting[obs.activity]
        log_water = -.5*np.sum(((obs.hydration-water)/error['hydration'])**2, axis=-1)
        joint = log_water[None, :]-.5*((obs.total-modeled[..., 0])/error['total'])**2
        if index == 1 and extra != 'none':
            actual, channel = (obs.bath, 1) if extra == 'bath' else (obs.alternate_total, 0)
            joint = joint-.5*((actual-modeled[..., channel])/error['extra'])**2
        key = 0 if shared_prefix else obs.activity
        components[key] = components.get(key, 0.) + joint
    per_activity = {a: softmax(logp, axis=1) for a, logp in components.items()}
    likelihood = sum(logsumexp(logp, axis=1)-np.log(logp.shape[1]) for logp in components.values())
    return softmax(likelihood), per_activity
