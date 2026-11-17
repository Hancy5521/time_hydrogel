"""Equal-count witness measurements for an install-to-work transition."""

import numpy as np
from scipy.special import ndtr, softmax, xlogy

from audit_patch_reservoir_study import extremes, normalized_clauses
from patch_exchange_forward import exchange_bank
from patch_reservoir_joint_bank import joint_bank, prepared_fields
from patch_thickness_catalogue_benchmark import geometry
from window_risk_cases import packet
from window_risk_forecasts import prior, summary
from window_spatial_probe import (
    changed_mechanism, conditional_release, observed_profiles, observed_zone, plausible_profiles,
)


def scoped_task(task, plan):
    goal = task['requests'][plan['tight']['request']]
    return {**task, 'requests': [goal], 'scoring': {**task['scoring'], 'horizon': goal['work_end']}}


def catalogue(task, plan):
    sections = []
    selected = plan['thickness']['catalogues']['thickness27']
    offset = 0
    for variant in plan['thickness']['actions_at_scale']:
        available = variant['actions']
        members = [a for i, a in enumerate(available, offset) if i in selected]
        offset += len(available)
        if members:
            sections.append({'scale': variant['scale'], 'actions': members})
    return sections


def response(known, task, plan, atoms, *, generating=False, shift=False):
    """Return five signed window clauses for every state and allowed action."""
    cells, period = plan['truth_mesh' if generating else 'prediction_mesh']
    dt = plan['sampling_dt']
    time = np.arange(round(task['scoring']['horizon'] / dt) + 1) * dt
    blocks = []
    bank, kwargs = (exchange_bank, {'outcome_cfg': plan['exchange']}) if shift else (joint_bank, {})
    for variant in catalogue(task, plan):
        parent, cfg = geometry(known, task, variant, 1.)
        traces = [bank(parent, variant['actions'], cfg, task['bath'][level], atoms[:, :4], atoms[:, 4:],
                       cells, period, time, **kwargs) for level in ('wet', 'weak')]
        for trace in traces:
            trace[..., 3] /= variant['scale']
        blocks.append(normalized_clauses(extremes(traces, time, task), task)[:, 0])
    return np.concatenate(blocks, axis=1)


def features(known, action, task, plan, atoms, *, generating=False, group=None):
    """Fixed first hydration and possible second readings, with no service outcome."""
    cells, step = plan['truth_mesh' if generating else 'prediction_mesh']
    settings = plan['tight']
    t = np.asarray(plan['spatial_probe']['times'])
    if t[-1] >= task['requests'][0]['install_end'] or t[-1] > plan['probes']['last']:
        raise ValueError('Witness acquisition exceeds the registered information deadline.')
    shift = generating and group['shift']
    bank, kwargs = (exchange_bank, {'outcome_cfg': plan['exchange']}) if shift else (joint_bank, {})
    wet = bank(known, [action], task['mechanism'], 1., atoms[:, :4], atoms[:, 4:], cells, step,
               np.r_[0., t], **kwargs)
    if shift:
        profiles = observed_profiles(group, task, plan)
    else:
        ticks = np.rint(t / step).astype(int)
        profiles = plausible_profiles(known, action, task['mechanism'], ticks, cells, step)
    first = observed_zone(profiles[0], plan['spatial_probe']['bands'][0])
    def signal(name):
        if name == 'log_conductance':
            return np.log(wet[:, -1, 0, 3] / known['thickness'])
        return np.full(len(atoms), observed_zone(profiles[1], settings['bands'][name]))
    second = np.column_stack([signal(name) for name in settings['sensors']])
    return first, second


def experiment_inputs(group, task, plan):
    spec = plan['tight']
    observe, dry, _ = packet(group, plan, task)
    witness = task['actions'][plan['fixed_action']]
    atom = prior(group['known'], {**plan, 'continuum_count': plan['spatial_probe']['atoms_per_water']}, True)
    stock = prepared_fields(group['known'], [witness], task['mechanism'], atom[:, 4:], plan['prediction_mesh'][0])[:, 0]
    fraction = stock[:, :stock.shape[-1] // 2].sum(axis=1) / stock.sum(axis=1)
    _, diffuse = conditional_release(observe, atom, plan)
    with np.errstate(divide='ignore'):
        initial = np.log(diffuse) - .5 * ((dry[:, None] - fraction) / plan['probes']['sigma_half']) ** 2
    truth_atoms = np.column_stack((group['kinetic'], group['stock']))
    actual_first, actual_second = features(group['known'], witness, task, changed_mechanism(group, plan),
                                         truth_atoms, generating=True, group=group)
    generator = np.random.default_rng(np.random.SeedSequence([spec['witness_entropy'], plan['tight_role'], group['i']]))
    noise = generator.normal(size=(len(dry), 2))
    read_sigma = np.asarray(list(spec['sensors'].values()))
    measured_first = actual_first + spec['first_sigma'] * noise[:, 0]
    measured_second = actual_second + noise[:, 1, None] * read_sigma
    preweight, model_second, model_response = [], [], []
    for scale in plan['spatial_probe']['water_grid']:
        known = {**group['known'], 'water_D': group['known']['water_D'] * scale}
        first, second = features(known, witness, task, plan, atom)
        preweight.append(initial - .5 * ((measured_first[:, None] - first) / spec['first_sigma']) ** 2)
        model_second.append(second)
        model_response.append(response(known, task, plan, atom))
    return (softmax(np.concatenate(preweight, axis=1), axis=1), measured_second,
            np.concatenate(model_second), np.concatenate(model_response))


def prospective_value(weights, sites, sigmas, loss):
    """Compare Bayes window loss and latent-state information before the reading."""
    centers = weights @ sites
    moment = weights @ sites ** 2
    scale = np.sqrt(np.maximum(moment - centers ** 2, 0.) + sigmas ** 2)
    boundaries = centers[..., None] + scale[..., None] * np.array([-2., -1.25, -.75, -.25, .25, .75, 1.25, 2.])
    cdf = ndtr((boundaries[:, None, :, :] - sites[None, :, :, None]) / sigmas[None, None, :, None])
    chance = np.diff(cdf, axis=-1, prepend=0., append=1.)
    joint = weights[:, :, None, None] * chance
    expected_loss = np.einsum('shvb,ha->svba', joint, loss).min(axis=-1).sum(axis=-1)
    mixture = joint.sum(axis=1)
    information = (xlogy(chance, chance).sum(axis=-1) * weights[:, :, None]).sum(axis=1)
    information -= xlogy(mixture, mixture).sum(axis=-1)
    return expected_loss, information


def prediction(weights, clauses, rank_weight):
    margin = clauses.max(axis=-1)
    mean, quantiles = summary(margin, weights, (.50, .90))
    zero = margin <= 0.
    risk = weights @ (~zero).astype(float)
    chance = weights @ (clauses > 0.).reshape(len(clauses), -1)
    score = risk + rank_weight * (weights @ np.maximum(margin, 0.))
    banks = {}
    for name, ranking in (('upper_margin', quantiles[1]), ('posterior_violation', score)):
        choice = np.lexsort((mean, risk, ranking), axis=-1)[:, 0]
        banks[name] = {'choice': choice.tolist(), 'score': np.take_along_axis(ranking, choice[:, None], 1)[:, 0].tolist()}
    exists = weights @ np.any(zero, axis=1).astype(float)
    return {'banks': banks, 'violation_probability': risk.tolist(), 'mean': mean.tolist(),
            'median': quantiles[0].tolist(), 'upper': quantiles[1].tolist(),
            'posterior_any_action': exists.tolist(), 'posterior_best_fixed_action': (1. - risk.min(axis=1)).tolist(),
            'clause_violation_probability': chance.reshape(*risk.shape, clauses.shape[-1]).tolist()}


def forecast(group, task, plan):
    task = scoped_task(task, plan)
    pre, observed, model, clauses = experiment_inputs(group, task, plan)
    names = list(plan['tight']['sensors'])
    sigmas = np.asarray(list(plan['tight']['sensors'].values()))
    margin = clauses.max(axis=-1)
    rank_weight = plan['risk_weight']
    loss = (margin > 0.) + rank_weight * np.maximum(margin, 0.)
    value, information = prospective_value(pre, model, sigmas, loss)
    with np.errstate(divide='ignore'):
        logweight = np.log(pre[:, :, None]) - .5 * ((observed[:, None] - model) / sigmas) ** 2
    all_weights = softmax(logweight, axis=1)
    fixed = {name: prediction(all_weights[..., i], clauses, rank_weight) for i, name in enumerate(names)}
    chosen = {'decision_directed': value.argmin(axis=1), 'state_information': information.argmax(axis=1)}
    for key, ids in chosen.items():
        # Pick the modality from the first read and predicted outcomes alone.
        selected = np.take_along_axis(all_weights, ids[:, None, None], axis=2)[..., 0]
        fixed[key] = prediction(selected, clauses, rank_weight)
    return {'mother': group['parent'], 'forecasts': fixed, 'chosen_sensor_indices': {k: v.tolist() for k, v in chosen.items()},
            'expected_bayes_loss': value.tolist(), 'expected_state_information': information.tolist()}


def reveal(group, task, plan):
    truth = changed_mechanism(group, plan)
    atoms = np.column_stack((group['kinetic'], group['stock']))
    values = response(group['known'], scoped_task(task, plan), truth, atoms, generating=True, shift=group['shift'])
    return {'mother': group['parent'], 'exchange_mechanism': group['shift'], 'clauses': values.tolist()}
