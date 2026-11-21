"""Early hidden-aperture chemistry/graph readouts and a stationary oracle kernel."""
from time import perf_counter

import numpy as np

from aam_budgeted_cold_replay import _replay
from aam_formation_cycle_core import _physical_config
from aam_formation_local_initial import initial_fingerprint, propose_initial
from aam_formation_local_tempering import _cost
from aam_formation_path_tempering import _fingerprints, _merge_cost, _query
from aam_snapshot_process import PORTS, WINDOW_Y, likelihood, weight


FEATURE_NAMES = ('incorporated_AAm_local_mass_fraction', 'nonwinding_component_AAm_length',
                 'available_pendant_MBAA_local_mass_fraction')
LAGS = (16, 64, 256)
NATIVE_KEYS = ('native_segment_attempted', 'native_segment_completed', 'native_segment_failed')


def component_features(aam, incorporated, edges, lifts):
    """For each AAm: incorporated count in its component and component winding."""
    adjacent = [[] for _ in range(len(incorporated))]
    for (i, j), lift in zip(edges, lifts, strict=True):
        adjacent[i].append((j, lift))
        adjacent[j].append((i, -lift))
    gauge = np.zeros((len(adjacent), 3), dtype=np.int64)
    seen = np.zeros(len(adjacent), dtype=bool)
    count = np.zeros(len(adjacent), dtype=np.int64)
    winding = np.zeros(len(adjacent), dtype=bool)
    for start in range(len(adjacent)):
        if seen[start]:
            continue
        current, inconsistent = [start], False
        seen[start] = True
        for i in current:
            for j, lift in adjacent[i]:
                candidate = gauge[i] + lift
                if seen[j]:
                    inconsistent |= bool(np.any(gauge[j] != candidate))
                else:
                    gauge[j] = candidate
                    seen[j] = True
                    current.append(j)
        nodes = np.asarray(current, dtype=np.int64)
        mass = int(np.count_nonzero(incorporated[nodes[nodes < aam]]))
        count[nodes], winding[nodes] = mass, inconsistent
    return count[:aam], winding[:aam]


def aperture_features(packet):
    """Two fixed hidden ports, each with three bounded pre-gel readouts.

    Finite-component AAm length is a covalent graph proxy. It averages total
    component AAm counts over locally visible incorporated AAm in nonwinding
    components, weighted by their aperture membership, and divides by planned
    AAm count. It does not assert experimental solubility or chain mass.
    """
    state, material, geometry, *_ = packet
    aam = int(material[0])
    owners = np.asarray(state.owner).reshape(-1, 2)
    available = np.asarray(state.available).reshape(-1, 2)
    present = np.asarray(state.present, bool)
    used = owners >= 0
    incorporated = used.any(axis=1) & present
    pendant = (used[aam:].sum(axis=1) == 1) & (available[aam:].sum(axis=1) == 1) & present[aam:]
    covalent_count = int(state.counts[1])
    size, winding = component_features(aam, incorporated, state.edges[:covalent_count], state.lifts[:covalent_count])
    finite = incorporated[:aam] & ~winding
    mass = present * np.where(np.arange(len(present)) < aam, 1., 2.17)
    rows = []
    for center in PORTS:
        w = weight(state.points, center, WINDOW_Y[1], float(geometry[2]))
        visible_mass = float(w @ mass)
        finite_weight = w[:aam] * finite
        visible_finite = float(finite_weight.sum())
        local_aam = float(w[:aam] @ incorporated[:aam])
        local_pendant = float(w[aam:] @ pendant) * 2.17
        rows.append((local_aam / visible_mass if visible_mass else 0.,
                     float(finite_weight @ size) / (aam * visible_finite) if visible_finite else 0.,
                     local_pendant / visible_mass if visible_mass else 0.))
    result = np.asarray(rows)
    if not np.isfinite(result).all() or np.any(result < 0.) or np.any(result > 1. + 2.e-14):
        raise ArithmeticError('An aperture feature violated the registered physical range.')
    return result


def history_features(packets):
    if len(packets) != 4:
        raise ValueError('Use a complete original three-camera trajectory plus its initial packet.')
    return np.stack([aperture_features(packet) for packet in packets[1:]])


def distance(first, second):
    return np.linalg.norm(np.asarray(first) - np.asarray(second)) / np.sqrt(np.size(first))


def energy(atoms, weights, target):
    x = np.asarray(atoms).reshape(len(atoms), -1)
    y = np.asarray(target).reshape(-1)
    return float((weights @ np.linalg.norm(x-y, axis=1)
                  - .5 * weights @ np.linalg.norm(x[:, None] - x[None, :], axis=2) @ weights) / np.sqrt(len(y)))


def stationarity_chain(config, teacher, truth, measured, rng_at):
    """Auditor-only oracle. A 50% identity plus a reversible random coordinate.

    The supplied generative truth is a stationary start conditional on its
    original noisy camera measurement. Its use is never a forecast.
    """
    query, noise = _query(config, measured)
    actual = _physical_config(config, teacher)
    clock = np.asarray(config['prefix_clock'], float)
    checkpoints = truth['checkpoint_packets']
    hist, reaction = _fingerprints(checkpoints)
    camera_ll = likelihood(truth['clean'], query, noise)
    if (not truth['valid'] or not np.array_equal(truth['factors'], teacher)
            or not np.isfinite(camera_ll).all() or np.any(camera_ll > 0.)):
        raise ValueError('Use the complete generating state and identical same-law observation model.')
    path = {'checkpoint_packets': checkpoints, 'native_calls': truth['native_calls'], 'clean': truth['clean'],
            'camera_log_likelihood': camera_ll, 'log_likelihood': float(camera_ll.sum()),
            'history_fingerprint': hist, 'reacted_structure_fingerprint': reaction,
            'birth_id': 'generating_truth', 'mother_id': truth['mother_id'], 'factors': teacher,
            'interval_seeds': truth['explicit_native_seeds'], 'initial_variables': truth['initial_variables']}
    rows, saved, total = [], {}, _cost()
    started = perf_counter()
    for step in range(1, max(LAGS)+1):
        rng = rng_at(step, 0)
        coin = float(rng.random())
        coordinate = int(rng.integers(4))
        row = {'step':step, 'coin':coin, 'coordinate':coordinate, 'uniform':None, 'screened_at':None,
               'old_history_fingerprint':path['history_fingerprint'], 'old_log_likelihood':path['log_likelihood'],
               'old_birth':path['birth_id'], 'accepted':False, 'native_calls':[], 'error':None}
        rows.append(row)
        cost = _cost()
        try:
            if coin >= .5:
                rng = rng_at(step, 1)
                initial = path['initial_variables']
                seeds = path['interval_seeds'].copy()
                ratio, change = 0., None if coordinate == 0 else coordinate-1
                if change is None:
                    initial, ratio, metadata = propose_initial(config, initial, rng)
                    self_loop = metadata['self_loop']
                else:
                    seeds[change] = rng.integers(0, 2**32, dtype=np.uint32)
                    self_loop = seeds[change] == path['interval_seeds'][change]
                row.update(proposed_initial_fingerprint=initial_fingerprint(initial), proposed_seeds=seeds,
                           self_loop=bool(self_loop), log_hastings=ratio)
                if self_loop:
                    if ratio != 0.:
                        raise ArithmeticError('An exact identity proposal cannot have a Hastings correction.')
                    row['accepted'] = True
                else:
                    uniform = float(rng_at(step, 2).random())
                    row['uniform'] = uniform
                    candidate = _replay(config, actual, teacher, query, noise, path, initial, seeds, change,
                                        ratio, uniform, True, f'oracle_{step:03d}', clock, row, cost)
                    if candidate is not None:
                        path = candidate
            row.update(final_history_fingerprint=path['history_fingerprint'], final_birth=path['birth_id'])
        except Exception as exc:
            row['error'] = {'type':type(exc).__name__, 'message':str(exc)}
        finally:
            _merge_cost(total, cost)
            row.update({key:cost[key] for key in NATIVE_KEYS})
        if row['error']:
            break
        if step in LAGS:
            saved[step] = {k:path[k] for k in ('checkpoint_packets','camera_log_likelihood','history_fingerprint',
                           'birth_id','clean','interval_seeds','initial_variables')}
    return {'valid':len(saved) == len(LAGS), 'initial_history_fingerprint':hist,
            'cost':total, 'moves':rows, 'lag_paths':saved, 'seconds':perf_counter()-started}
