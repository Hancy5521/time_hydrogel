"""Prospective continuous inverse design within the registered AAm/MBAA law."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from hashlib import sha256
from pathlib import Path
from shutil import copyfile

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.spatial.distance import cdist
from scipy.stats import qmc
from threadpoolctl import threadpool_limits

from aam_bernstein_feed_design import (
    bernstein_minimum, bernstein_weights, binary53_effective_probability,
    continuous_feed_rollout, rank_thresholds, threshold_sweep_minimum,
)
from aam_edge_fork import State
from aam_first_interval_paired_fork import score_future_for_action
from aam_formation_actual_rollout import packet_fingerprint
from aam_formation_cycle_core import ACTIONS
from aam_formation_cycle_driver import np_load, np_save, read, save_tree, sha, write
from aam_formation_cycle_methods import GOALS, process_goal_losses
from aam_formation_path_cycle import event, load_tree, stamp
from aam_temporal_inverse_bridge import load_training_c3, training_boundary
from package_aam_formation_local_cycle import source_closure
from run_aam_direct_energy_trajectory_law import atlas_receipts
from run_aam_mark_candidate_filter import interval, verify
from run_aam_privileged_v5_action_headroom import best_fixed, reference
from run_aam_suffix_group_smoother import acquired_data
from run_aam_temporal_energy_law import normalize


ROOT = Path(__file__).resolve().parent
PROTOCOL = ROOT/'AAM_CONTINUOUS_INVERSE_PROTOCOL_ZH.md'
AUDITOR = ROOT/'audit_aam_continuous_inverse.py'
NS = 'aam-continuous-unchanged-formation-inverse-count-law-v1'
ENTROPY = list(np.frombuffer(sha256(NS.encode()).digest()[:16], dtype='<u4').astype(int))
UNITS = list(range(2026123000, 2026123048))
METHODS = ('bernstein', 'sweep', 'gp', 'sobol', 'bernstein_six', 'population', 'fixed')
REPEATS = 4
REFERENCE_REPEATS = 24
TIMES = np.array([2.30, 2.3875, 2.5625, 2.7375, 2.9125, 3.00])
ROLES = {'prior_index': 1, 'sobol': 2, 'planning_future': 3, 'reference_future': 4}
GP_LENGTH, GP_NUGGET, GP_GRID = .22, .03, 33
NATIVE_PLAN_PER_MOTHER = 2*6*7*REPEATS*3


def rng(unit, role, block=0, child=0):
    return np.random.default_rng(np.random.SeedSequence([*ENTROPY, unit, ROLES[role], block, child]))


def future_seeds(unit, j, role):
    return {
        'seed': int(rng(unit, role, j, 1).integers(0, 2**32-51, dtype=np.uint32)),
        'native_seeds': rng(unit, role, j, 2).integers(0, 2**32, 3, dtype=np.uint32),
        'placement_seed': int(rng(unit, role, j, 3).integers(0, 2**32, dtype=np.uint32)),
    }


def direct_controls(unit):
    scramble = int(rng(unit, 'sobol').integers(0, 2**32, dtype=np.uint32))
    new = qmc.Sobol(d=2, scramble=True, seed=scramble).random_base2(6)[:36]
    new[:, 0] = 2.30 + .70*new[:, 0]
    return np.vstack([ACTIONS, new])


def q_from_atlas(x, model):
    scale = model['camera_scale']
    dist = cdist((x/scale).reshape(len(x), -1), (model['x']/scale).reshape(len(model['x']), -1),
                 'sqeuclidean')
    return normalize(-.5*dist/(.5**2*model['kernel_sq_scale']))


def original_losses(values, norm):
    # The historical score validates a six-action input. Repeat one path to
    # reuse the unchanged single-future loss and its original validity checks.
    future = np.asarray(values)
    expanded = np.broadcast_to(future[..., None, :, :], (*future.shape[:-2], 6, 3, 6))
    if expanded.ndim < 4:
        expanded = expanded[None]
    result = process_goal_losses(expanded, norm)
    return result['loss'][..., 0, :], result['blocks'][..., 0, :, :]


def gp_domain(controls):
    affine = lambda c: (c - np.array([2.30, 0.])) / np.array([.70, 1.])
    ticks = np.linspace(0., 1., GP_GRID)
    mesh = np.stack(np.meshgrid(2.30+.70*ticks, ticks, indexing='ij'), axis=-1).reshape(-1, 2)
    mesh = np.unique(np.vstack([mesh, controls]), axis=0)
    scaled = affine(controls)

    def matern(x, y):
        r = np.sqrt(5.)*cdist(x, y)/GP_LENGTH
        return (1.+r+r*r/3.)*np.exp(-r)

    matrix = matern(scaled, scaled)
    matrix.flat[::len(matrix)+1] += GP_NUGGET
    return mesh, cho_factor(matrix), matern(affine(mesh), scaled)


def choose(control, uniforms, strata_values, direct_values, norm, fixed, population):
    strata, _ = original_losses(strata_values, norm)  # draw, time, count, goal
    direct, _ = original_losses(direct_values, norm)  # draw, action, goal
    result = np.empty((len(METHODS), len(GOALS), 2))
    estimated = np.zeros((len(METHODS), len(GOALS)))
    fitted_cells = {'bernstein': [], 'sweep': []}
    for j, goal in enumerate(GOALS):
        for arm, operation in enumerate((bernstein_minimum, threshold_sweep_minimum)):
            fits = []
            for ti in range(len(TIMES)):
                y = strata[:, ti, :, j]
                fits.append(operation(y) if arm == 0 else operation(uniforms, y))
            winner = min(range(len(TIMES)), key=lambda i: fits[i]['risk'])
            fitted = fits[winner]
            result[arm, j] = TIMES[winner], fitted['p']
            estimated[arm, j] = fitted['risk']
            fitted_cells[METHODS[arm]].append(fits)
    average = direct.mean(axis=0)
    gp_x, factor, cross = gp_domain(control)
    location = average.mean(axis=0)
    gp = location + cross @ cho_solve(factor, average-location)
    for arm, scores, domain in ((2, gp, gp_x), (3, average, control)):
        best = np.argmin(scores, axis=0)
        result[arm] = domain[best]
        estimated[arm] = scores[best, np.arange(len(GOALS))]
    mean_strata = strata.mean(axis=0)
    anchored = np.stack([bernstein_weights(binary53_effective_probability(p)) @
                         mean_strata[np.flatnonzero(TIMES == t)[0]] for t,p in ACTIONS])
    best = np.argmin(anchored, axis=0)
    result[4] = np.asarray(ACTIONS)[best]
    estimated[4] = anchored[best, np.arange(len(GOALS))]
    result[5] = population
    for j, (t,p) in enumerate(population):
        ix = int(np.flatnonzero(TIMES == t)[0])
        estimated[5,j] = bernstein_weights(binary53_effective_probability(p)) @ mean_strata[ix,:,j]
    result[-1] = fixed
    for j in range(len(GOALS)):
        idx = int(np.flatnonzero(np.all(control == fixed[j], axis=1))[0])
        estimated[-1, j] = average[idx, j]
    return result, estimated, fitted_cells


def register(out, atlas, inverse, population, review, workers):
    atlas_plan, ids, model, atlas_inputs = atlas_receipts(atlas)
    records, config, origin = training_boundary(atlas)
    old, norm, fixed = best_fixed(inverse)
    if (config != old or config != atlas_plan['physics'] or [r['unit'] for r in records] != ids
            or set(UNITS).intersection(i for group in atlas_plan['units'].values() for i in group)):
        raise ValueError('Require the original physics, goal scales and a new independent group.')
    if not review.read_text(encoding='utf-8').strip():
        raise ValueError('Bind the actual Work review or continuous scope discussion.')
    ps, pa = population/'SCORED.json', population/'INDEPENDENT_AUDIT.json'
    population_score, population_audit = read(ps), read(pa)
    if (population_audit['passed'] is not True
            or population_audit['scored_sha256'] != sha(ps)
            or population_score['old_held_opened'] != 0
            or population_score['independent_train_mothers'] != 48):
        raise ValueError('Require the independently audited training-only population continuous control.')
    population_controls = np.asarray(population_score['controls'])
    if (population_controls.shape != (2,2) or not np.isin(population_controls[:,0], TIMES).all()
            or not np.isfinite(population_controls).all() or np.any(population_controls[:,1]<0.)
            or np.any(population_controls[:,1]>1.)):
        raise ValueError('Population comparator must be independently fixed in the same time/probability box.')
    files = [*atlas_inputs, atlas/'ACQUIRED.json', inverse/'REGISTERED.json',
             inverse/'LIBRARY_COMMITTED.json', inverse/'INDEPENDENT_AUDIT.json',
             inverse/'SCORED.json', inverse/'TRAIN_FUTURES.npz', review, PROTOCOL,
             ROOT/'AAM_BERNSTEIN_FEED_MATH_ZH.md', population/'REGISTERED.json', ps, pa]
    closure = source_closure([Path(__file__), AUDITOR])
    out.mkdir(exist_ok=False)
    reg = out/'REGISTERED.json'
    write(reg, {**stamp(), 'namespace': NS, 'entropy': ENTROPY, 'roles': ROLES, 'workers': workers,
                'physics': config, 'goals': GOALS, 'normalization': norm, 'held_units': UNITS,
                'atlas': str(atlas), 'atlas_model': str(model), 'atlas_model_sha256': sha(model),
                'train_units': ids, 'training_boundary': origin, 'inverse_for_fixed': str(inverse),
                'fixed_controls': np.asarray(ACTIONS)[fixed['choices']],
                'population': str(population), 'population_controls': population_controls,
                'planning_repeats': REPEATS, 'reference_repeats': REFERENCE_REPEATS,
                'times': TIMES, 'methods': METHODS, 'box': [[2.30, 3.00], [0., 1.]],
                'gp': [GP_LENGTH, GP_NUGGET, GP_GRID], 'rbf': [.5, 1.],
                'raw_audit_mothers': [UNITS[0], UNITS[-1]], 'raw_audit_repeats': [0, 1],
                'planning_native_per_mother': NATIVE_PLAN_PER_MOTHER,
                'max_reference_native_per_mother': 3*len(METHODS)*len(GOALS)*REFERENCE_REPEATS,
                'primary': 'paired 95% lower for each comparator >0 in average of two untouched goals',
                'old_held_targets_opened': False, 'sources': {str(p):sha(p) for p in closure.values()},
                'inputs': {str(p):sha(p) for p in files}})
    snapshot = out/'SOURCE_SNAPSHOT'
    snapshot.mkdir()
    for name, path in closure.items():
        copyfile(path, snapshot/name)
    write(out/'RUN.json', {**stamp(), 'registered_before_native': True, 'registration_sha256': sha(reg)})


def checked(out):
    path = out/'REGISTERED.json'
    plan = read(path)
    if (plan['namespace'] != NS or plan['held_units'] != UNITS or plan['times'] != list(TIMES)
            or plan['methods'] != list(METHODS) or read(out/'RUN.json')['registration_sha256'] != sha(path)):
        raise ValueError('Require the exact registered continuous prospective protocol.')
    for path, digest in {**plan['inputs'], **plan['sources']}.items():
        if sha(path) != digest:
            raise ValueError(f'Input or source changed: {path}')
    return plan


def pool(out, plan, fn, jobs, label):
    rows = []
    with ProcessPoolExecutor(max_workers=plan['workers']) as executor:
        for row in executor.map(fn, jobs, chunksize=1):
            rows.append(row)
            event(out, label, unit=row['unit'], native_calls=row['native_calls'])
    if [r['unit'] for r in rows] != UNITS:
        raise RuntimeError('Complete prospective cohort required, without replacements.')
    return rows


def acquire_one(job):
    folder, physics, unit = job
    with threadpool_limits(limits=1):
        y, raw = acquired_data(physics, unit)
    if raw['calls'] != {'attempted': 3, 'completed': 3, 'failed': 0}:
        save_tree(folder/f'{unit}_FAILED.npz', raw)
        raise RuntimeError('Prospective prefix failed, raw is retained.')
    return {'unit': unit, 'query': np_save(folder/f'{unit}_QUERY.npz', measured=y),
            'raw': save_tree(folder/f'{unit}_PREFIX.npz', raw), 'native_calls': 3}


def acquire(out, plan):
    folder = out/'held'
    folder.mkdir()
    write(out/'ACQUISITION_STARTED.json', {**stamp(), 'registration_sha256':sha(out/'REGISTERED.json')})
    jobs = [(folder, plan['physics'], unit) for unit in UNITS]
    rows = pool(out, plan, acquire_one, jobs, 'sealed_complete_fresh_held_prefix')
    write(out/'ACQUIRED.json', {**stamp(), 'registration_sha256':sha(out/'REGISTERED.json'), 'held':rows,
                              'native_calls':3*len(rows)})


def path_block(folder, filename, packet, controls, cfg, streams, unit_label):
    beginnings, values, micro, winding, g_raw, g_clean, born, calls, seconds = ([] for _ in range(9))
    origin = packet_fingerprint(packet)
    for k, action in enumerate(controls):
        branch = continuous_feed_rollout(packet, action, cfg, **streams, mother_id=unit_label)
        try:
            v, m, w = score_future_for_action(branch, packet, action, cfg, streams)
            count = branch['cost']['native_segment_completed']
            if count != 3 or branch['mother_packet_sha256'] != origin:
                raise RuntimeError('Invalid native call count or altered microphysical ancestor.')
        except Exception:
            save_tree(folder/f'{filename.stem}_A{k:02d}_FAILED.npz', branch)
            raise
        snapshots = branch['snapshots']
        beginnings.append([branch['branch_start_packet'][0], *[stage['packet'][0] for stage in snapshots]])
        values.append(v)
        micro.append(m)
        winding.append(w)
        g_raw.append([stage['properties']['raw_dimensionless_finite_G_tau'] for stage in snapshots])
        g_clean.append([stage['properties']['dimensionless_finite_G_tau'] for stage in snapshots])
        born.append([stage['properties']['dimensionless_affine_Born'] for stage in snapshots])
        calls.append([call['completed'] for call in branch['native_calls']])
        seconds.append([branch['cost'][name] for name in ('native_seconds','q_read_seconds','property_seconds')])
    matrix = {f'state_{name}':np.asarray([[state[i] for state in branch] for branch in beginnings])
              for i, name in enumerate(State._fields)}
    uniforms = np.random.default_rng(streams['placement_seed']).random(6)
    realizations = np.unique(np.column_stack((controls[:,0],uniforms[None,:]<controls[:,1,None])),axis=0)
    archive = np_save(folder/filename, **matrix, controls=controls, values=np.asarray(values),
                      micro=np.asarray(micro), winding=np.asarray(winding), completed=np.asarray(calls),
                      unique_physical_layouts=np.array(len(realizations)),
                      native_timing=np.asarray(seconds), raw_g=np.asarray(g_raw), cleaned_g=np.asarray(g_clean),
                      born=np.asarray(born), origin=np.array(origin), **{name:np.asarray(v) for name,v in streams.items()},
                      source_material=packet[1], source_geometry=packet[2], source_rates=packet[3],
                      source_factors=packet[4])
    return np.asarray(values), archive


def plan_one(job):
    folder, cfg, norm, atlas, train, unit, selected, fixed, population = job
    folder.mkdir()
    control = direct_controls(unit)
    uniforms, thresholds, strata, direct, blocks, micro, winding = ([] for _ in range(7))
    try:
        with threadpool_limits(limits=1):
            for j,index in enumerate(selected):
                packet, origin = load_training_c3(atlas, train[index], cfg)
                streams = future_seeds(unit, j, 'planning_future')
                grid = rank_thresholds(streams['placement_seed'])
                uniforms.append(grid['uniforms'])
                pp = grid['thresholds']
                thresholds.append(pp)
                forced = np.column_stack([np.repeat(TIMES, 7), np.tile(pp, len(TIMES))])
                controls = np.vstack([forced, control])
                values, artifact = path_block(folder, Path(f'P{j}_RAW.npz'), packet, controls, cfg,
                                              streams, f'planning:{unit}:{j}')
                blocks.append({'raw': artifact, 'train_identity': origin})
                with np.load(artifact['path'], allow_pickle=False) as payload:
                    micro.append(payload['micro'][:42].reshape(6,7,3,8))
                    winding.append(payload['winding'][:42].reshape(6,7,3,3))
                strata.append(values[:42].reshape(len(TIMES), 7, 3, 6))
                direct.append(values[42:])
            choices, risks, cells = choose(control, np.asarray(uniforms), np.asarray(strata),
                                           np.asarray(direct), norm, fixed, population)
    except Exception as error:
        write(folder/'FAILED.json', {**stamp(), 'unit':unit, 'completed_raw_blocks':blocks,
                                    'error':repr(error), 'uniform_failure':getattr(error,'uniforms',None)})
        raise
    a = np_save(folder/'STRATA_VALUES.npz', times=TIMES, uniforms=np.asarray(uniforms),
                thresholds=np.asarray(thresholds), values=np.asarray(strata), train_idxs=np.asarray(selected),
                micro=np.asarray(micro), winding=np.asarray(winding))
    b = np_save(folder/'DIRECT_VALUES.npz', controls=control, values=np.asarray(direct))
    forecast_times = np.searchsorted(TIMES, choices[0,:,0])
    forecast_law = np.broadcast_to(bernstein_weights(binary53_effective_probability(choices[0,:,1]))[:,None,:]
                                   /REPEATS, (len(GOALS), REPEATS, 7))
    law = np_save(folder/'SELECTED_PATH_LAW.npz', weights=forecast_law, time_positions=forecast_times)
    f = folder/'PLANNED.json'
    write(f, {**stamp(), 'unit':unit, 'choices':choices, 'planning_losses':risks, 'cells':cells,
              'strata':a, 'direct':b, 'law':law, 'raw_blocks':blocks, 'native_calls':NATIVE_PLAN_PER_MOTHER})
    return {'unit':unit, 'plan':reference(f), 'native_calls':NATIVE_PLAN_PER_MOTHER}


def predict(out, plan):
    acq = out/'ACQUIRED.json'
    records = read(acq)['held']
    folder = out/'planning'
    folder.mkdir()
    write(out/'PREDICT_STARTED.json', {**stamp(), 'acquisition_sha256':sha(acq)})
    x = np.stack([np_load(verify(out/'held'/f"{r['unit']}_QUERY.npz",r['query']))['measured'] for r in records])
    atlas = Path(plan['atlas'])
    train, cfg, boundary = training_boundary(atlas)
    if cfg != plan['physics'] or boundary != plan['training_boundary']:
        raise ValueError('Training-only physical atlas provenance changed.')
    model = load_tree(plan['atlas_model'], plan['atlas_model_sha256'])
    weights = q_from_atlas(x, model)
    picked = np.stack([rng(unit,'prior_index').choice(len(train), size=REPEATS, p=weight)
                       for unit, weight in zip(UNITS, weights, strict=True)])
    selected = np_save(out/'POSTERIOR_SCENARIOS.npz', query=x, c3_weights=weights,
                       train_positions=picked, held_units=np.asarray(UNITS), train_units=np.asarray(plan['train_units']))
    jobs = [(folder/str(u), cfg, plan['normalization'], atlas, train, u, picked[j],
             np.asarray(plan['fixed_controls']), np.asarray(plan['population_controls'])) for j,u in enumerate(UNITS)]
    rows = pool(out, plan, plan_one, jobs, 'continuous_blind_plan_complete')
    choices = np.stack([read(r['plan']['path'])['choices'] for r in rows])
    decisions = np_save(out/'CHOICES.npz', choices=choices, held_units=np.asarray(UNITS))
    write(out/'PREDICTIONS_FROZEN.json', {**stamp(), 'registration_sha256':sha(out/'REGISTERED.json'),
          'acquisition_sha256':sha(acq), 'held_truth_opened':False, 'posterior':selected,
          'plans':rows, 'choices':decisions, 'native_calls':sum(r['native_calls'] for r in rows)})


def reference_one(job):
    out, record, cfg, choices, freeze_sha = job
    if sha(out/'PREDICTIONS_FROZEN.json') != freeze_sha:
        raise RuntimeError('Actual reference requires the complete prediction freeze.')
    unit = record['unit']
    raw = load_tree(verify(out/'held'/f'{unit}_PREFIX.npz',record['raw']))
    packet = raw['packets'][-1]
    controls, mapping = np.unique(choices.reshape(-1, 2), axis=0, return_inverse=True)
    mapping = mapping.reshape(choices.shape[:-1])
    folder = out/'reference'/str(unit)
    folder.mkdir()
    branches, arrays, micro, winding = [], [], [], []
    try:
        with threadpool_limits(limits=1):
            for j in range(REFERENCE_REPEATS):
                streams = future_seeds(unit, j, 'reference_future')
                values, receipt = path_block(folder, Path(f'R{j}_RAW.npz'), packet, controls, cfg,
                                             streams, f'reference:{unit}:{j}')
                arrays.append(values)
                with np.load(receipt['path'], allow_pickle=False) as payload:
                    micro.append(payload['micro'])
                    winding.append(payload['winding'])
                branches.append(receipt)
    except Exception as error:
        write(folder/'FAILED.json', {**stamp(), 'unit':unit, 'completed_raw_blocks':branches, 'error':repr(error)})
        raise
    predictions = np_save(folder/'REFERENCE_VALUES.npz', controls=controls, choices=choices,
                          mapping=mapping, values=np.asarray(arrays),
                          micro=np.asarray(micro), winding=np.asarray(winding))
    f = folder/'COMPLETE.json'
    native_calls = len(controls)*REFERENCE_REPEATS*3
    write(f, {**stamp(), 'unit':unit, 'freeze_sha256':freeze_sha, 'reference':predictions,
              'branches':branches, 'native_calls':native_calls, 'new_c3_identity':packet_fingerprint(packet)})
    return {'unit':unit, 'reference':reference(f), 'native_calls':native_calls}


def reveal(out, plan):
    f = out/'PREDICTIONS_FROZEN.json'
    freeze = read(f)
    acquisition = out/'ACQUIRED.json'
    if (freeze['registration_sha256'] != sha(out/'REGISTERED.json')
            or freeze['acquisition_sha256'] != sha(acquisition)
            or freeze['held_truth_opened'] is not False
            or [r['unit'] for r in freeze['plans']] != UNITS):
        raise ValueError('Unacquired truth or partial choices cannot start actual reference.')
    choices = np_load(verify(out/'CHOICES.npz',freeze['choices']))['choices']
    folder = out/'reference'
    folder.mkdir()
    write(out/'REVEAL_STARTED.json', {**stamp(), 'freeze_sha256':sha(f)})
    jobs = [(out,r,plan['physics'],c,sha(f)) for r,c in zip(read(acquisition)['held'], choices, strict=True)]
    rows = pool(out, plan, reference_one, jobs, 'independent_selected_control_native_references')
    write(out/'REFERENCES_CLOSED.json', {**stamp(), 'freeze_sha256':sha(f),'rows':rows,
                                        'native_calls':sum(r['native_calls'] for r in rows)})


def selected_forecast_scores(strata, law, matches, ref, norm):
    result = []
    scales = {'values': np.r_[np.ones(3),norm['mechanical_scale']], 'micro': 1.}
    for g,(w,ti) in enumerate(zip(law['weights'], law['time_positions'], strict=True)):
        stats = []
        for label, scale in scales.items():
            atoms = (strata[label][:,ti]/scale).reshape(REPEATS*7,-1)
            truth = (ref[label][:,matches[g]]/scale).reshape(REFERENCE_REPEATS,-1)
            mass = w.ravel()
            es = mass @ cdist(atoms,truth).mean(axis=1) - .5*mass @ cdist(atoms,atoms) @ mass
            stats.append(es/np.sqrt(atoms.shape[1]))
        chance = np.einsum('rk,rkth->th',w,strata['winding'][:,ti])
        stats.append(np.mean((chance-ref['winding'][:,matches[g]])**2))
        result.append(stats)
    return result


def score(out, plan):
    frozen = out/'PREDICTIONS_FROZEN.json'
    refs = read(out/'REFERENCES_CLOSED.json')
    if refs['freeze_sha256'] != sha(frozen) or [r['unit'] for r in refs['rows']] != UNITS:
        raise ValueError('Require independent references for all original choices.')
    blocks, losses, energy = [], [], []
    choices = np_load(verify(out/'CHOICES.npz',read(frozen)['choices']))['choices']
    for r, c, forecast in zip(refs['rows'], choices, read(frozen)['plans'], strict=True):
        folder = out/'reference'/str(r['unit'])
        record = read(verify(folder/'COMPLETE.json',r['reference']))
        raw = np_load(verify(folder/'REFERENCE_VALUES.npz',record['reference']))
        if not np.array_equal(c, raw['controls'][raw['mapping']]):
            raise RuntimeError('Independent reference does not match the frozen continuous controls.')
        l, b = original_losses(raw['values'], plan['normalization'])
        idx = raw['mapping']
        losses.append(l[:,idx,np.arange(len(GOALS))].mean(axis=0))
        blocks.append(b[:,idx,np.arange(len(GOALS)),:].mean(axis=0))
        p = out/'planning'/str(r['unit'])
        planned = read(verify(p/'PLANNED.json',forecast['plan']))
        strata = np_load(verify(p/'STRATA_VALUES.npz',planned['strata']))
        law = np_load(verify(p/'SELECTED_PATH_LAW.npz',planned['law']))
        energy.append(selected_forecast_scores(strata,law,idx[0],raw,plan['normalization']))
    l, b = np.asarray(losses), np.asarray(blocks)
    advantage = {name:interval((l[:,j]-l[:,0]).mean(axis=-1)) for j,name in enumerate(METHODS) if j}
    target_advantage = {name:[interval(l[:,j,k]-l[:,0,k]) for k in range(len(GOALS))]
                        for j,name in enumerate(METHODS) if j}
    payload = np_save(out/'SCORED_ARRAYS.npz', losses=l, blocks=b, choices=choices,
                      selected_path_predictive_scores=np.asarray(energy),held_units=np.asarray(UNITS))
    write(out/'SCORED.json', {**stamp(), 'registered_sha256':sha(out/'REGISTERED.json'),
          'freeze_sha256':sha(frozen), 'reference_sha256':sha(out/'REFERENCES_CLOSED.json'),'arrays':payload,
          'independent_mothers':len(UNITS), 'methods':METHODS, 'goals':GOALS, 'means':l.mean(axis=0),
          'mean_blocks':b.mean(axis=0), 'same_budget_gain':advantage,'by_goal_gain':target_advantage,
          'selected_path_scores':{'names':['functional_energy','micro8_energy','winding_brier'],
                                  'means':np.mean(energy,axis=0),'same_control_ref_only':True},
          'primary_gate':all(v['pointwise_t95'][0]>0 for v in advantage.values()),
          'off_anchor_counts':np.sum(~np.any(np.all(choices[...,None,:]==np.asarray(ACTIONS),axis=-1),axis=-1),axis=0),
          'native_acquisition':3*len(UNITS), 'native_planning':read(frozen)['native_calls'],
          'native_reference':refs['native_calls'], 'wet_experiments':0})


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('phase', choices=('register','acquire','predict','reveal','score'))
    cli.add_argument('--out', type=Path, required=True)
    cli.add_argument('--atlas', type=Path)
    cli.add_argument('--inverse', type=Path)
    cli.add_argument('--population', type=Path)
    cli.add_argument('--review', type=Path)
    cli.add_argument('--workers', type=int, choices=range(1,9), default=6)
    arg = cli.parse_args()
    if arg.phase == 'register':
        if None in (arg.atlas, arg.inverse, arg.population, arg.review):
            cli.error('register requires --atlas, --inverse, --population and --review.')
        register(arg.out.resolve(),arg.atlas.resolve(strict=True),arg.inverse.resolve(strict=True),
                 arg.population.resolve(strict=True),arg.review.resolve(strict=True),arg.workers)
    else:
        out = arg.out.resolve(strict=True)
        {'acquire':acquire,'predict':predict,'reveal':reveal,'score':score}[arg.phase](out,checked(out))


if __name__ == '__main__':
    main()
