"""Learn phase-level process corrections and preserve paired residuals."""

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from patch_thickness_catalogue_benchmark import geometry
from tight_window_witness import catalogue, features, scoped_task
from window_risk_cases import packet
from window_spatial_probe import changed_mechanism


SENSORS = ('depth_mid', 'log_conductance')
METHODS = {'direct_depth': ('depth_mid', False), 'phase_depth': ('depth_mid', True),
           'direct_electric': ('log_conductance', False), 'phase_electric': ('log_conductance', True)}


def witness_inputs(group, task, plan):
    """Extract exactly the same seven available scalars, even for direct fitting."""
    task = scoped_task(task, plan)
    spec = plan['tight']
    reads, dry, _ = packet(group, plan, task)
    five = np.array([[*[v for r in pair for v in (r.hydration[0], r.total)], stock]
                     for pair, stock in zip(reads, dry, strict=True)])
    atoms = np.column_stack((group['kinetic'], group['stock']))
    action = task['actions'][plan['fixed_action']]
    first, second = features(group['known'], action, task, changed_mechanism(group, plan), atoms,
                             generating=True, group=group)
    key = [spec['witness_entropy'], plan['tight_role'], group['i']]
    noise = np.random.default_rng(np.random.SeedSequence(key)).normal(size=(len(dry), 2))
    times = first + spec['first_sigma'] * noise[:, 0]
    x = np.column_stack((np.broadcast_to(list(group['known'].values()), (len(dry), len(group['known']))), five, times))
    obs = {name: np.column_stack((x, second[:, tuple(spec['sensors']).index(name)] + spec['sensors'][name] * noise[:, 1]))
           for name in SENSORS}
    return {name: a.tolist() for name, a in obs.items()}


def design(rows, task, plan, name):
    signal, phase = METHODS[name]
    sensors = np.asarray([r['raw_inputs'][signal] for r in rows])
    action_features = np.array([(v['scale'], a['wet_transfer'], a['near_loading'])
                                for v in catalogue(task, plan) for a in v['actions']])
    n, states = sensors.shape[:2]
    raw = np.concatenate((np.broadcast_to(sensors[..., None, :], (n, states, len(action_features), sensors.shape[-1])),
                          np.broadcast_to(action_features, (n, states, *action_features.shape))), axis=-1)
    if phase:
        records = [r['forecasts'][signal] for r in rows]
        physics = np.stack([np.asarray([v[k] for v in records]) for k in ('mean', 'median', 'upper')], axis=-1)
        phase_risk = np.asarray([v['clause_violation_probability'] for v in records])
        return np.concatenate((raw, physics, phase_risk), axis=-1)
    return raw


def build(seed, phase):
    if phase:
        return make_pipeline(StandardScaler(), Ridge(alpha=28.))
    return ExtraTreesRegressor(n_estimators=192, min_samples_leaf=12, random_state=seed, n_jobs=4)


def fit(x, y, name, seed):
    """Out-of-fold residuals are vectors, grouped by entire original mothers."""
    def flat(a):
        return a.reshape(-1, a.shape[-1])

    phase = METHODS[name][1]
    residuals = np.empty_like(y)
    for train, held in KFold(4, shuffle=True, random_state=seed).split(x):
        model = build(seed, phase).fit(flat(x[train]), flat(y[train]))
        residuals[held] = y[held] - model.predict(flat(x[held])).reshape(y[held].shape)
    model = build(seed, phase).fit(flat(x), flat(y))
    # [state/mother draw, candidate, clause]. Keep co-dependence across all clauses and actions.
    return model, residuals.reshape(-1, *residuals.shape[-2:])


def predictive_margins(x, model, residuals, *, chunk=16):
    """Evaluate the full phase intersection on the same residual donor draw."""
    group_shape = x.shape[:-1]
    mean = model.predict(x.reshape(-1, x.shape[-1])).reshape(*group_shape, residuals.shape[-1])
    records = []
    for start in range(0, len(mean), chunk):
        paths = mean[start:start+chunk, :, None] + residuals[None, None]
        margin = paths.max(axis=-1)
        q90 = np.quantile(margin, .90, axis=2, method='higher')
        failure = (margin > 0).mean(axis=2)
        expected = np.maximum(margin, 0.).mean(axis=2)
        records.append(np.stack((q90, failure + .05*expected, failure), axis=-1))
    return np.concatenate(records)


def rank(values):
    banks = {}
    q90, penalized, probability = np.moveaxis(values, -1, 0)
    for name, risk in (('upper_margin', q90), ('posterior_violation', penalized)):
        action = np.lexsort((q90, probability, risk), axis=-1)[..., 0]
        banks[name] = {'choice': action, 'score': np.take_along_axis(risk, action[..., None], axis=-1)[..., 0]}
    return banks
