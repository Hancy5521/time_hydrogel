"""Observation and posterior summaries for the joint release/profile experiment."""

import numpy as np
from scipy.special import softmax

from reservoir_coupon_likelihood import Read, coupon_end, infer, observation_table, wet_prefix


def stock_fractions(fields):
    """Reduce dry stock to eight fixed depth fractions that sum to one."""
    cells = fields.shape[-1]
    if cells % 8:
        raise ValueError('The spatial validation grid requires eight aligned bins.')
    weights = fields.reshape(*fields.shape[:-1], 8, cells//8).sum(axis=-1)
    return weights/weights.sum(axis=-1, keepdims=True)


def stock_mean(bins, channel):
    if channel == 'total':
        return np.ones(bins.shape[:-1])
    if channel == 'half':
        return bins[..., :4].sum(axis=-1)
    if channel == 'back_slice':
        return bins[..., -1]
    raise ValueError(f'Unexpected dry stock channel {channel}')


def coupon_truth(group, truth, cfg, condition, slot):
    activity, stop = condition
    low, high = (cfg['activities'][n] for n in ('low', 'high'))
    shift = (high-activity)/(high-low)
    tau = group['tau_high']*group['tau_ratio']**shift
    end = coupon_end(truth[:, :3], activity, stop, tau, cfg['coupon_loss_factor'], group['slow_share'], order=64)
    signals = wet_prefix(np.array(cfg['hydration_times']), activity, tau, group['slow_share'])
    noise = group['noise'][:, slot, :]
    wet = signals+cfg['sigma']['hydration']*noise[:, 0:2]
    total = end[:, 0]+cfg['sigma']['total']*noise[:, 2]
    return [Read(activity, stop, h, float(f), float(f), 0.) for h, f in zip(wet, total, strict=True)]


def readout_likelihoods(group, truth, hypothesis, cfg, designs):
    """Construct equal-budget pre-deadline kinetics observations for each design."""
    conditions = sorted({condition for pair in designs.values() for condition in pair})
    activities = sorted({a for a, _ in conditions})
    # k and gamma determine total mobilization; use only unique kinetic templates.
    unique, inverse = np.unique(hypothesis[:, :2], axis=0, return_inverse=True)
    kinetics = np.column_stack((unique, np.zeros(len(unique))))
    means = observation_table(kinetics, cfg, activities, conditions)
    observations = {(key, slot): coupon_truth(group, truth, cfg, key, slot)
                    for key in conditions for slot in range(2)}
    out = {}
    for name, pair in designs.items():
        samples = list(zip(observations[pair[0], 0], observations[pair[1], 1], strict=True))
        weights = np.array([infer(sample, means, cfg)[0][inverse] for sample in samples])
        out[name] = (weights, samples)
    return out


def updated_weights(kinetic, spatial, observed, sigma):
    width = (observed[:, None]-spatial[None, :])/sigma
    with np.errstate(divide='ignore'):
        return softmax(np.log(kinetic)-.5*width**2, axis=1)


def predictive_summary(weights, latent):
    """Return posterior mean and finite-candidate equal-tail intervals."""
    values = np.asarray(latent)
    flat = values.reshape(len(values), -1)
    order = np.argsort(flat, axis=0, kind='stable')
    ordered = np.take_along_axis(flat, order, axis=0)
    cdf = np.cumsum(weights[:, order], axis=1)
    lo = (cdf < .05).sum(axis=1).clip(0, len(values)-1)
    hi = (cdf < .95).sum(axis=1).clip(0, len(values)-1)
    cols = np.arange(flat.shape[1])
    shape = (len(weights), *values.shape[1:])
    return {'mean': (weights@flat).reshape(shape).tolist(),
            'lo': ordered[lo, cols].reshape(shape).tolist(), 'hi': ordered[hi, cols].reshape(shape).tolist()}
