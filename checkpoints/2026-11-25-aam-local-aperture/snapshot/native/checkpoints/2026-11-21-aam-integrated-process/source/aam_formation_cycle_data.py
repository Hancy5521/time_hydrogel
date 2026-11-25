"""Prospective AAm/MBAA formation reference data and read-only physical checks.

Source only on import. The root driver owns registration, namespace admission,
pool launch and prediction freezing. A mother has one actual prefix; every
future starts from a clone of that saved State. Held futures require an explicit
all-predictions-and-selections freeze receipt. No fitting occurs in this module.
"""
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
import hashlib
import json
import os


ROOT = Path(__file__).resolve().parent
SCHEMA = 'aam-formation-cycle-reference-data-v1'
NAMESPACE = 'aam-mbaa-property-inverse-cycle-v1-20261118'
MASTER_SEED = 1918000000
PROBE_TAU = (0., .1, .5, 2., 10.)
CHANNELS = ('q_left', 'q_right', 'nominal_vinyl_conversion', 'G_tau_0.1', 'G_tau_0.5', 'G_tau_2')
PREFIX_CLOCK = (0., .65, 1.35, 2.05)
FUTURE_TIMES = (3.25, 4.85, 7.20)
ACTIONS = tuple((t, f) for t in (2.30, 3.00) for f in (0., .5, 1.))
STATE_FIELDS = ('points', 'unwrapped', 'present', 'active', 'available', 'owner',
                'labels', 'edges', 'lifts', 'events', 'periodic_parent',
                'periodic_shift', 'periodic_rank', 'winding', 'counts', 'position')
ARRAY_SCHEMA = 'aam-formation-cycle-lossless-numeric-tree-v1'


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, value):
    with Path(path).open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def event(folder, action, **details):
    with (Path(folder)/'EVENTS.jsonl').open('a', encoding='utf-8') as f:
        f.write(json.dumps({'actual_utc': now(), 'action': action, **details},
                           ensure_ascii=False, allow_nan=False)+'\n')


def data_contract():
    """Metadata only; caller must register and execute a complete version."""
    return {'schema': SCHEMA, 'namespace': NAMESPACE, 'master_seed': MASTER_SEED,
            'roles': {'train': 64, 'held': 16}, 'suffixes': {'train': 2, 'held': 4},
            'modes': ['train/full', 'held/prefixonly', 'held/reveal'],
            'mother_seed': 'train=master+100000*id;held=master+10000000+100000*id',
            'prefix': 'Oneactualprefix;initialize motherbase;progressbase+18/19/20.',
            'assay_seed': 'motherbase+1001;Gaussian noise[.08,.10,.08,.10] at3cameras.',
            'future_seed': 'motherbase+2000+200*repeat;shared across6actions as computationalCRN.',
            'future_native': 'base+1/2/3;placementbase+51;commonclock includes2.30and3.00.',
            'prefix_clock': list(PREFIX_CLOCK), 'future_times': list(FUTURE_TIMES),
            'probe_tau': list(PROBE_TAU), 'actions': [list(a) for a in ACTIONS],
            'future_values': 'repeat,action6,time3,channel6', 'channels': list(CHANNELS),
            'target114': 'cutoff6 FIRST then action/time/channel108;sharedcutoffrepeated inrepeat axis only.',
            'query': 'Only noisy measured(3,4) and X(12);nevertrueState/factors/clean/risk/cutoff/future.',
            'teacher_training': 'Oneprefix plus action0repeat0 pertrainmother;not6duplicated prefixes.',
            'teacher_held': 'EVALonly;prefix plus action0repeat0,unavailabletoprediction.',
            'reference_native_calls_max': 3696,
            'kernel_equivalence_extra': 'Firsttrainmother optional6originalevolve calls fromstoredbeforepackets;notnewmothers.',
            'resume': 'Completedmother verified/skipped;completedprefix/branchesreadfromraw;attemptwithoutcompletearchive stops,neverreruns.',
            'mechanics_invalid': 'KeepNaNwithvalidfalseanderror;neverreplacewithzero orsilentlydropmother.',
            'zero_mechanics': 'Finitezeroresponse/plateau isvalid underdeclaredtangentmodel.',
            'no_import_execution': True, 'fits': 0}


def _numeric_save(path, **arrays):
    import numpy as np
    if any(np.asarray(a).dtype.hasobject for a in arrays.values()):
        raise TypeError('No pickle/object arrays in numerical evidence.')
    with Path(path).open('xb') as f:
        np.savez_compressed(f, **arrays)
    return {'path': str(Path(path).resolve()), 'sha256': sha(path), 'bytes': Path(path).stat().st_size}


def _numeric_load(path, expected=None):
    import numpy as np
    if expected is not None and sha(path) != expected:
        raise ValueError('Numerical evidence hash changed: '+str(path))
    with np.load(path, allow_pickle=False) as f:
        return {k: f[k].copy() for k in f.files}


def _tree_save(path, value, binding):
    """Exclusive full State/ragged ledger storage, no native call or padding."""
    import numpy as np
    arrays = {}
    def encode(x):
        if isinstance(x, np.ndarray):
            if x.dtype.hasobject:
                raise TypeError('No object arrays in physical archive.')
            key = f'a{len(arrays):07d}'
            arrays[key] = x
            return {'array': key, 'dtype': x.dtype.str, 'shape': list(x.shape)}
        if isinstance(x, tuple) and getattr(x, '_fields', None) == STATE_FIELDS:
            return {'State': [encode(a) for a in x]}
        if isinstance(x, dict):
            return {'mapping': [[encode(k), encode(v)] for k, v in x.items()]}
        if isinstance(x, tuple):
            return {'tuple': [encode(v) for v in x]}
        if isinstance(x, list):
            return [encode(v) for v in x]
        if isinstance(x, np.generic):
            return encode(x.item())
        if isinstance(x, float) and not np.isfinite(x):
            return encode(np.array(x, dtype=np.float64))
        if x is None or isinstance(x, (str, bool, int, float)):
            return x
        raise TypeError('Unserializable physical evidence: '+str(type(x)))
    meta = {'schema': ARRAY_SCHEMA, 'binding': binding, 'tree': encode(value)}
    arrays['metadata_json'] = np.array(json.dumps(meta, ensure_ascii=False, separators=(',', ':'), allow_nan=False))
    result = _numeric_save(path, **arrays)
    result.update(actual_utc=now(), schema=ARRAY_SCHEMA, binding=binding, arrays=len(arrays))
    return result


def _tree_load(path, binding, expected=None):
    from aam_edge_fork import State
    raw = _numeric_load(path, expected)
    meta = json.loads(str(raw.pop('metadata_json').item()))
    if meta['schema'] != ARRAY_SCHEMA or meta['binding'] != binding:
        raise ValueError('Cached physical state is from another registered mother.')
    def decode(x):
        if isinstance(x, dict):
            if 'array' in x:
                a = raw[x['array']]
                if a.dtype.str != x['dtype'] or list(a.shape) != x['shape']:
                    raise ValueError('Stored State array schema changed.')
                return a
            if 'State' in x:
                return State(*(decode(v) for v in x['State']))
            if 'mapping' in x:
                return {decode(k): decode(v) for k, v in x['mapping']}
            if 'tuple' in x:
                return tuple(decode(v) for v in x['tuple'])
        if isinstance(x, list):
            return [decode(v) for v in x]
        return x
    return decode(meta['tree'])


def _fingerprint_packet(packet):
    import numpy as np
    h = hashlib.sha256()
    names = (*STATE_FIELDS, 'material', 'geometry', 'rates', 'factors')
    for name, value in zip(names, (*packet[0], *packet[1:]), strict=True):
        a = np.asarray(value)
        h.update(name.encode()); h.update(a.dtype.str.encode()); h.update(str(a.shape).encode())
        h.update(a.tobytes(order='C'))
    return h.hexdigest()


def _root_groups(parent):
    import numpy as np
    p = np.asarray(parent)
    out = np.empty(len(p), np.int64)
    for i in range(len(p)):
        j = i
        for _ in range(len(p)+1):
            if j < 0 or j >= len(p):
                raise ValueError('Union-find parent outside actual particle range.')
            k = int(p[j])
            if k == j:
                out[i] = j; break
            j = k
        else:
            raise ValueError('Cyclic actual union-find structure.')
    return out


def check_actual_state(state, *, expected_age=None, full_feed=False):
    """Independent readonly State conservation/BFS/event checks, no physics."""
    import numpy as np
    if tuple(getattr(state, '_fields', ())) != STATE_FIELDS or len(state.points) != 78:
        raise ValueError('Expected the complete78molecule/16field actual AAm/MBAA State.')
    if state.counts.shape != (7,) or state.position.shape != (4,) or state.counts[6]:
        raise ValueError('Read only a completed actual Brownian/chemical boundary.')
    if expected_age is not None and float(state.position[0]) != float(expected_age):
        raise ValueError('Actual snapshot age differs from the registered age.')
    if not (np.isfinite(state.points).all() and np.isfinite(state.unwrapped).all()
            and np.isfinite(state.position).all()) or np.any(state.points < 0.) or np.any(state.points >= 6.):
        raise ValueError('Actual coordinates/clock invalid.')
    if state.owner.shape != (156,) or state.active.shape != (156,) or state.available.shape != (156,):
        raise ValueError('Actual vinyl-site arrays changed shape.')
    if np.any(~state.present[:64]) or (full_feed and not state.present.all()):
        raise ValueError('Required monomers/feed molecules are absent.')
    legal = np.r_[np.tile([True, False], 64), np.ones(28, bool)]
    arrived = np.repeat(state.present, 2) & legal
    used = state.owner >= 0
    if (np.any(used & ~arrived) or np.any(state.active & ~used)
            or not np.array_equal(state.available, arrived & ~used)
            or int(used.sum()) != int(state.counts[5])):
        raise ValueError('Vinyl availability/consumption/active-site conservation failed.')
    ev, ed = map(int, state.counts[:2])
    if not 0 <= ev <= len(state.events) or not 0 <= ed <= len(state.edges):
        raise ValueError('Actual event/edge count outside native buffers.')
    events = state.events[:ev]
    if not np.isfinite(events).all() or np.any(np.diff(events[:, 0]) < -1.e-12) or np.any(events[:, 0] > state.position[0]+1.e-12):
        raise ValueError('Invalid actual event chronology.')
    tags = events[:, 1].astype(np.int64)
    if not np.array_equal(tags, events[:, 1]) or not np.isin(tags, np.arange(10)).all():
        raise ValueError('Unregistered reaction event type.')
    conversion_events = np.isin(tags, (1, 2, 3, 4, 5, 6, 9))
    if int(conversion_events.sum()) != int(state.counts[5]):
        raise ValueError('Reacted-vinyl ledger does not match physical events.')
    initiation_events = np.isin(tags, (1, 2))
    if int(initiation_events.sum()) != int(state.counts[4]) or np.any(state.owner >= state.counts[4]):
        raise ValueError('Kinetic lineage ownership/initialization ledger failed.')
    if not 6 <= int(state.counts[3]+state.counts[4]) <= 12:
        raise ValueError('Unchanged initial radical budget must remain6..12.')
    edge_events = events[(events[:, 2] >= 0) & (tags != 8)]
    expected_edges = (edge_events[:, 2:4]/2).astype(np.int64)
    edges = state.edges[:ed]
    if (len(edge_events) != ed or not np.array_equal(expected_edges, edges)
            or not np.array_equal(edge_events[:, 10:13], state.lifts[:ed])):
        raise ValueError('Actual covalent edges/lifts do not match event chronology.')
    if ed and (np.any(edges < 0) or np.any(edges >= 78) or not state.present[edges].all()
               or np.any(edges[:, 0] == edges[:, 1])):
        raise ValueError('Illegal physical bond endpoint.')
    neighbors = [[] for _ in range(78)]
    for a, b in edges:
        neighbors[int(a)].append(int(b)); neighbors[int(b)].append(int(a))
    groups = np.full(78, -1, np.int64)
    for start in range(78):
        if groups[start] >= 0:
            continue
        groups[start] = start; queue = [start]
        for i in queue:
            for j in neighbors[i]:
                if groups[j] < 0:
                    groups[j] = start; queue.append(j)
    for name in ('labels', 'periodic_parent'):
        native = _root_groups(getattr(state, name))
        if not np.array_equal(groups[:, None] == groups, native[:, None] == native):
            raise ValueError('Stored connectivity differs from independent valid-edge BFS: '+name)
    feeds = events[tags == 0]
    expected_fed = int(state.present[64:].sum())-8
    if len(feeds) != expected_fed or (len(feeds) and (not np.array_equal(feeds[:, 3], np.arange(72, 78)) or not state.present.all())):
        raise ValueError('Delayed six-MBAA feed ledger failed.')
    return {'passed': True, 'groups': groups, 'used_vinyl': int(used.sum()),
            'actual_events': ev, 'actual_edges': ed, 'fed_MBAA': int(state.present[64:].sum()),
            'initial_radical_budget': int(state.counts[3]+state.counts[4]),
            'conversion': float(used.sum()/92.), 'full_feed': bool(state.present.all())}


def independent_q(state):
    """Same declared microscopic statistic, independent BFS and no label edits."""
    import numpy as np
    check = check_actual_state(state)
    groups = check['groups']
    lineage_count = np.zeros(78, np.int64)
    for g in np.unique(groups):
        nodes = np.flatnonzero(groups == g)
        owners = state.owner.reshape(78, 2)[nodes]
        lineage_count[g] = len(np.unique(owners[owners >= 0]))
    mass = np.where(np.arange(78) < 64, 1., 2.17)
    support = state.present*mass
    reacted = (state.owner.reshape(78, 2) >= 0).any(axis=1)
    regional = mass*reacted*(lineage_count[groups] >= 3)
    out = []
    for x in (1.45, 4.45):
        delta = state.points[:, :2]-[x, 4.35]
        delta -= 6.*np.rint(delta/6.)
        w = np.maximum(0., 1.-np.abs(delta[:, 0])/1.3)*np.maximum(0., 1.-np.abs(delta[:, 1])/1.3)
        out.append(float(w@regional/max(float(w@support), 1.e-12)))
    return np.array([*out, check['conversion']], dtype=np.float64)


def property_readout(packet, config):
    """Return explicit invalid vs true zero without changing actual chemistry."""
    import numpy as np
    from aam_actual_network_properties import properties_from_actual_snapshot, model_contract
    before = _fingerprint_packet(packet)
    check_actual_state(packet[0])
    started = perf_counter()
    raw_g, tolerance = None, None
    rounding_rule = ('tol=1e-10*max(1,abs(Born));onlynegativein[-tol,0)becomes0;'
                     'largernegative/monotonicviolationinvalid;G.1=0andlaterpositive<=tolbecomes0;'
                     'largerlaterpositiveinvalid;rawvaluesretained;nopositivefloor.')
    try:
        result = properties_from_actual_snapshot(packet[0], config, PROBE_TAU)
        raw_g = np.array(result['dimensionless_finite_G_tau'], dtype=np.float64, copy=True)
        born = float(result['dimensionless_affine_Born'])
        tolerance = 1.e-10*max(1., abs(born))
        if (raw_g.shape != (5,) or not np.isfinite(raw_g).all() or not np.isfinite(born)
                or np.any(raw_g < -tolerance) or np.any(np.diff(raw_g) > tolerance)):
            raise ValueError('Tangent response violates its registered nonnegative/monotone roundoff bound.')
        cleaned = np.maximum(raw_g, 0.)
        zero_consistency = 0
        if cleaned[1] == 0. and np.any(cleaned[2:] > 0.):
            if np.any(cleaned[2:] > tolerance):
                raise ValueError('Positive long-lag response with zero G(.1) exceeds numerical tolerance.')
            zero_consistency = int(np.count_nonzero(cleaned[2:] > 0.))
            cleaned[2:] = 0.
        difference = cleaned-raw_g
        result.update(valid=True, error=None, dimensionless_finite_G_tau=cleaned,
              raw_dimensionless_finite_G_tau=raw_g,
              roundoff={'rule': rounding_rule, 'tolerance': tolerance,
                        'changed_entries': int(np.count_nonzero(difference)),
                        'maximum_absolute_adjustment': float(np.max(np.abs(difference))),
                        'zero_denominator_consistency_entries': zero_consistency,
                        'maximum_raw_monotonic_violation': float(max(0., np.diff(raw_g).max()))})
    except Exception as error:
        result = {'valid': False, 'error': {'type': type(error).__name__, 'message': str(error)},
                  'formation_age': float(packet[0].position[0]), 'probe_tau': np.array(PROBE_TAU),
                  'dimensionless_finite_G_tau': np.full(5, np.nan),
                  'raw_dimensionless_finite_G_tau': np.full(5, np.nan) if raw_g is None else raw_g,
                  'dimensionless_affine_Born': float('nan'),
                  'dimensionless_static_tangent_plateau': float('nan'), 'model': model_contract(),
                  'roundoff': {'rule': rounding_rule, 'tolerance': tolerance,
                               'changed_entries': 0, 'maximum_absolute_adjustment': 0.,
                               'invalid': True}}
    if _fingerprint_packet(packet) != before:
        raise AssertionError('Mechanical readout changed an actual State/packet.')
    result['readout_seconds'] = perf_counter()-started
    return result


def _read6(packet, prop, *, reported_q=None):
    import numpy as np
    before = _fingerprint_packet(packet)
    q = independent_q(packet[0])
    if reported_q is not None:
        if np.asarray(reported_q).shape != (3,) or np.max(np.abs(q-reported_q)) > 2.e-11:
            raise ValueError('Native microscopic q/conversion differs from independent actual-graph readout.')
    if _fingerprint_packet(packet) != before:
        raise AssertionError('Readonly BFS property/Q check altered an actual State.')
    g = np.asarray(prop['dimensionless_finite_G_tau'])
    if g.shape != (5,) or not np.array_equal(np.asarray(prop['probe_tau']), PROBE_TAU):
        raise ValueError('Wrong frozen probe-time readout.')
    if prop['valid']:
        tolerance = float(prop['roundoff']['tolerance'])
        if not np.isfinite(g).all() or g.min() < 0. or np.any(np.diff(g) > tolerance):
            raise ValueError('A valid tangent response must be finite/nonnegative/nonincreasing.')
    elif not np.isnan(g).all():
        raise ValueError('Invalid mechanics requires explicitNaN, never fabricatedzero.')
    return np.r_[q, g[[1, 2, 3]]]


def _verify_reveal(freeze):
    if not isinstance(freeze, dict) or set(freeze) != {'path', 'sha256'}:
        raise ValueError('Held reveal requires an explicit prediction_freeze={path,sha256}.')
    path = Path(freeze['path'])
    if path.name != 'ALL_PREDICTIONS_AND_SELECTIONS_FROZEN.json' or sha(path) != freeze['sha256']:
        raise ValueError('Wrong/changed held prediction and selection freeze receipt.')
    frozen = read(path)
    if frozen.get('prediction_and_selection_frozen') is not True:
        raise ValueError('All predictions/selections must be frozen before held future generation.')
    return {'path': str(path.resolve()), 'sha256': freeze['sha256']}


def _job(job):
    import numpy as np
    run = Path(job['run_dir']).resolve()
    plan = read(run/'REGISTERED.json')
    if plan.get('namespace') != NAMESPACE or plan.get('master_seed') != MASTER_SEED:
        raise ValueError('This is not the prospective fixed formation-cycle registration.')
    if plan['sources'].get(str(Path(__file__).resolve())) != sha(__file__):
        raise ValueError('Data source hash is not the root-registered source.')
    cfg = job['config']
    if cfg != plan['config']:
        raise ValueError('Worker configuration differs from the frozen root registration.')
    role, mother = job['role'], job['mother_id']
    if role not in ('train', 'held') or type(mother) is not int or not 0 <= mother < (64 if role == 'train' else 16):
        raise ValueError('Use exactly64train or16held mother identifiers.')
    phase = job.get('phase', 'full' if role == 'train' else 'prefixonly')
    if (role, phase) not in (('train', 'full'), ('held', 'prefixonly'), ('held', 'reveal')):
        raise ValueError('Invalid data role/phase combination.')
    if job.get('validate_native_record', False) and (role, mother) != ('train', 0):
        raise ValueError('Only firsttrainmother may receive the six extra native-record controls.')
    expected_seed = MASTER_SEED+(0 if role == 'train' else 10000000)+100000*mother
    if type(job['mother_seed']) is not int or job['mother_seed'] != expected_seed:
        raise ValueError('Only the root-declared mother seed formula is admitted.')
    if (tuple(cfg['prefix_clock']) != PREFIX_CLOCK or tuple(cfg['future_times']) != FUTURE_TIMES
            or tuple(cfg['common_stops']) != (2.3, 3.) or not np.array_equal(cfg['actions'], ACTIONS)
            or cfg['motion']['step'] != .025
            or tuple(cfg['noise']) != (.08, .10, .08, .10)
            or (cfg['recipe']['aam'], cfg['recipe']['mbaa'], cfg['recipe']['initial_mbaa']) != (64, 14, 8)):
        raise ValueError('Fixed chemistry, actions, clocks or observation law changed.')
    freeze = _verify_reveal(job.get('prediction_freeze')) if phase == 'reveal' else None
    folder = run/role/f'parent_{mother:03d}'
    folder.mkdir(parents=True, exist_ok=True)
    binding = {'schema': SCHEMA, 'root_registration': str(run/'REGISTERED.json'),
               'root_registration_sha256': sha(run/'REGISTERED.json'), 'namespace': NAMESPACE,
               'role': role, 'mother_id': mother, 'mother_seed': expected_seed}
    return folder, binding, cfg, phase, freeze


def _stage(folder, name, binding, callback, dispatched):
    path = folder/(name+'.npz')
    receipt = folder/(name+'_RECEIPT.json')
    attempt = folder/(name+'_ATTEMPTED.json')
    if path.exists():
        info = read(receipt) if receipt.exists() else None
        result = _tree_load(path, binding, None if info is None else info['sha256'])
        if info is None:
            info = {'path': str(path.resolve()), 'sha256': sha(path), 'bytes': path.stat().st_size,
                    'actual_utc': now(), 'binding': binding,
                    'recovered_from_complete_numeric_archive_without_physics': True}
            write(receipt, info)
        event(folder, 'loaded_existing_stage_no_replay', stage=name, sha256=info['sha256'])
        return result, info
    if attempt.exists():
        raise RuntimeError('Unresolved interrupted/failed physical attempt; preserve and do not rerun: '+name)
    write(attempt, {'actual_utc': now(), 'binding': binding, 'stage': name,
                    'no_automatic_retry': True})
    event(folder, 'physical_stage_attempted', stage=name)
    result = callback()
    # Persist actual physical output before interpreting readouts or correctness.
    info = _tree_save(path, result, binding)
    write(receipt, info)
    dispatched.append(result['cost'])
    event(folder, 'physical_stage_saved', stage=name, sha256=info['sha256'], cost=result['cost'])
    return result, info


def _teacher_arrays(prefix, branch):
    import numpy as np
    opportunities, selections, rows = [], [], []
    for phase, result in enumerate((prefix, branch)):
        risk = np.asarray(result['teacher_opportunity'])
        selected = np.asarray(result['teacher_selection'])
        provenance = np.asarray(result['teacher_event_rows'])
        n = len(risk)
        if risk.shape != (n, 3, 5) or selected.shape != (n, 2) or provenance.shape != (n, 8):
            raise ValueError('Core teacher risk/selection/provenance schema mismatch.')
        if not np.isfinite(risk).all() or np.any(risk < 0.) or selected.dtype.kind not in 'iu':
            raise ValueError('Teacher original primitive risks must be finite/nonnegative.')
        if n and (np.any(selected[:, 0] < 0) or np.any(selected[:, 0] >= 3)
                  or np.any(selected[:, 1] < 0) or np.any(selected[:, 1] >= 5)
                  or np.any(risk[np.arange(n), selected[:, 0], selected[:, 1]] <= 0.)):
            raise ValueError('Teacher selected pendant mark lacks legal primitive support.')
        # Cross-check the supervision against the actual retained event ledger,
        # without calling the teacher extractor or any physical kernel again.
        start = 0 if phase == 0 else int(result['branch_start_packet'][0].counts[0])
        state = result['packet'][0]
        actual = state.events[start:int(state.counts[0])]
        actual = actual[np.isin(actual[:, 1], (5., 6., 9.))]
        if len(actual) != n:
            raise ValueError('Teacher pendant supervision omitted/duplicated actual events.')
        if n:
            if (not np.array_equal(provenance[:, 2], actual[:, 0])
                    or not np.array_equal(provenance[:, 3:6], actual[:, 1:4])
                    or not np.array_equal(provenance[:, 6], actual[:, 6])
                    or not np.array_equal(selected[:, 1], actual[:, 7])):
                raise ValueError('Risk supervision differs from actual pendant event identity/time/occupancy.')
            kinds = np.array([{5: 0, 6: 1, 9: 2}[int(tag)] for tag in actual[:, 1]])
            if (not np.array_equal(kinds, selected[:, 0]) or not np.isfinite(provenance).all()
                    or np.any(provenance[:, 7] <= 0.)
                    or np.any(risk[np.arange(n), selected[:, 0], selected[:, 1]]+1.e-12 < provenance[:, 7])):
                raise ValueError('Actual event kind/selected primitive disagrees with original risk support.')
        opportunities.append(risk); selections.append(selected)
        rows.append(np.column_stack((np.full(n, phase), provenance)))
    return {'teacher_opportunity': np.concatenate(opportunities),
            'teacher_selection': np.concatenate(selections),
            'teacher_event_rows': np.concatenate(rows),
            'teacher_phase': np.r_[np.zeros(len(opportunities[0]), np.int64),
                                   np.ones(len(opportunities[1]), np.int64)]}


def _branch_check(result, prefix_packet, action, config, seed):
    import numpy as np
    if result['status'] != 'complete' or not result['valid']:
        raise RuntimeError('Actual future failed; raw retained, no replacement: '+str(result['error']))
    if result['cost']['native_segment_attempted'] != 3 or result['cost']['native_segment_completed'] != 3:
        raise ValueError('Future requires exactly3nativeprogress calls, not extra action-dependent splitting.')
    if len(result['snapshots']) != 3 or tuple(result['future_times']) != FUTURE_TIMES:
        raise ValueError('Actual future snapshot schedule differs from registration.')
    for call, age in zip(result['native_calls'], FUTURE_TIMES, strict=True):
        if (not call['completed'] or float(call['requested_time']) != age
                or not np.isin((2.3, 3.), call['clock']).all()):
            raise ValueError('Every action must use the SAME two interior numerical stops.')
    parent = prefix_packet[0]
    moved = result['branch_start_packet'][0]
    for name in STATE_FIELDS:
        a, b = getattr(parent, name), getattr(moved, name)
        if name in ('points', 'unwrapped'):
            a, b = a[parent.present], b[parent.present]
        if a.dtype != b.dtype or a.shape != b.shape or a.tobytes() != b.tobytes():
            raise ValueError('Action changed already-existing actual history: '+name)
    for i in (2, 3, 4):
        a, b = prefix_packet[i], result['branch_start_packet'][i]
        if a.dtype != b.dtype or a.shape != b.shape or a.tobytes() != b.tobytes():
            raise ValueError('Future action changed native geometry/rates/teacher factors.')
    if result['branch_start_packet'][1][3] != action[0] or seed != result['seed']:
        raise ValueError('Future material clock or registered provenance changed.')
    values, responses, raw_responses, valid, errors = [], [], [], [], []
    for snapshot, age in zip(result['snapshots'], FUTURE_TIMES, strict=True):
        packet = snapshot['packet']
        check_actual_state(packet[0], expected_age=age, full_feed=True)
        prop = snapshot['properties']
        values.append(_read6(packet, prop, reported_q=snapshot['q']))
        responses.append(prop['dimensionless_finite_G_tau'])
        raw_responses.append(prop['raw_dimensionless_finite_G_tau'])
        valid.append(bool(prop['valid'])); errors.append(prop['error'])
    return np.stack(values), np.stack(responses), np.stack(raw_responses), np.array(valid, bool), errors


def data_job(job):
    """Root ProcessPool job. Writes one exclusive mother; no fits/hidden inference.

    Required job keys: config, role, mother_id, mother_seed, run_dir. Held mode
    is selected by phase=prefixonly/reveal; reveal needs prediction_freeze.
    Optional validate_native_record=True is allowed only for train mother0.
    """
    import numpy as np
    from aam_formation_cycle_core import formation_prefix, formation_rollout
    from aam_network_competition_study import reference
    folder, binding, config, phase, freeze = _job(job)
    final_path = folder/'MOTHER_RECEIPT.json'
    if final_path.exists():
        result = read(final_path)
        if result['binding'] != binding:
            raise ValueError('Completed mother is from another registered run.')
        for row in result['files'].values():
            if sha(row['path']) != row['sha256']:
                raise ValueError('Completed mother evidence changed.')
        return {**result, 'dispatch_existing_complete_no_science': True}
    lock = folder/'ACTIVE.lock'
    with lock.open('x', encoding='utf-8') as f:
        json.dump({'actual_utc': now(), 'pid': os.getpid(), 'binding': binding}, f)
    started = perf_counter()
    dispatched, physical_stages, files = [], [], {}
    role, mother, base = binding['role'], binding['mother_id'], binding['mother_seed']
    try:
        if (folder/'FAILED.json').exists():
            raise RuntimeError('A failed mother record exists; no automatic retry of that dispatch.')
        teacher_factors, clock = reference('allocation')
        if clock != 1.:
            raise ValueError('Synthetic teacher clock must remain exactly1.')
        prefix, info = _stage(folder, 'PREFIX_RAW', binding,
              lambda: formation_prefix(config, base, teacher_factors, record_teacher=True,
                                       record_control=bool(job.get('validate_native_record', False)),
                                       mother_id=f'{role}:{mother}'), dispatched)
        files['prefix_raw'] = info; physical_stages.append(prefix['cost'])
        if prefix['status'] != 'complete' or not prefix['valid']:
            raise RuntimeError('Prefix failed; actual partial state preserved, do not rerun.')
        if (prefix['cost']['native_segment_attempted'] != 3 or prefix['cost']['native_segment_completed'] != 3
                or tuple(prefix['frames']) != PREFIX_CLOCK or np.asarray(prefix['clean']).shape != (3, 4)):
            raise ValueError('Actual prefix must contain exactly3calls/4frames/12cleanchannels.')
        for age, state in prefix['frames'].items():
            check_actual_state(state, expected_age=age)
        prefix_hash = _fingerprint_packet(prefix['packet'])
        past_hashes = {str(t): _fingerprint_packet((s, *prefix['packet'][1:])) for t, s in prefix['frames'].items()}
        query_path = folder/'QUERY.npz'
        if not query_path.exists():
            attempt = folder/'ASSAY_ATTEMPTED.json'
            if attempt.exists():
                raise RuntimeError('Unresolved assay draw attempt; never redraw an interrupted measurement.')
            write(attempt, {'actual_utc': now(), 'binding': binding, 'assay_seed': base+1001})
            noise = np.random.default_rng(base+1001).normal(size=(3, 4))*np.asarray(config['noise'])
            measured = prefix['clean']+noise
            files['query'] = _numeric_save(query_path, X=measured.reshape(12), measured=measured,
                                          mother_id=np.array(mother), role=np.array(role))
            write(folder/'QUERY.json', {**files['query'], 'binding': binding,
                  'assay_seed': base+1001, 'actual_utc': now(), 'fields': ['X', 'measured', 'mother_id', 'role'],
                  'truth_or_mechanisms_included': False})
        else:
            qi = read(folder/'QUERY.json') if (folder/'QUERY.json').exists() else None
            if qi is None:
                # The complete saved assay is authoritative; no second RNG call.
                saved = _numeric_load(query_path)
                if set(saved) != {'X', 'measured', 'mother_id', 'role'} or int(saved['mother_id']) != mother or str(saved['role']) != role:
                    raise ValueError('Interrupted saved assay has wrong identity/schema.')
                qi = {'path': str(query_path.resolve()), 'sha256': sha(query_path),
                      'bytes': query_path.stat().st_size, 'binding': binding,
                      'actual_utc': now(), 'assay_seed': base+1001,
                      'recovered_saved_assay_without_redraw': True, 'truth_or_mechanisms_included': False}
                write(folder/'QUERY.json', qi)
            if qi['binding'] != binding or sha(query_path) != qi['sha256']:
                raise ValueError('Original query changed.')
            files['query'] = qi
        cutoff_path = folder/'CUTOFF_EVAL.npz'
        if not cutoff_path.exists():
            prop = property_readout(prefix['packet'], config)
            cutoff = _read6(prefix['packet'], prop)
            files['cutoff_eval'] = _numeric_save(cutoff_path, cutoff_values=cutoff,
                 cutoff_G_tau=np.asarray(prop['dimensionless_finite_G_tau']), mechanics_valid=np.array(prop['valid']),
                 cutoff_G_tau_raw=np.asarray(prop['raw_dimensionless_finite_G_tau']),
                 probe_tau=np.array(PROBE_TAU), formation_age=np.array(2.05), clean=prefix['clean'])
            write(folder/'CUTOFF_EVAL.json', {**files['cutoff_eval'], 'binding': binding,
                  'actual_utc': now(), 'properties_valid': bool(prop['valid']), 'error': prop['error'],
                  'property_model': prop['model'], 'roundoff': prop['roundoff'], 'role': 'evaluation_only_not_query'})
        else:
            metadata = folder/'CUTOFF_EVAL.json'
            if metadata.exists():
                files['cutoff_eval'] = read(metadata)
            else:
                from aam_actual_network_properties import model_contract
                saved = _numeric_load(cutoff_path)
                files['cutoff_eval'] = {'path': str(cutoff_path.resolve()), 'sha256': sha(cutoff_path),
                      'bytes': cutoff_path.stat().st_size, 'binding': binding, 'actual_utc': now(),
                      'properties_valid': bool(saved['mechanics_valid']),
                      'error': 'Original metadata interrupted; saved numerical validity retained.',
                      'property_model': model_contract(), 'role': 'evaluation_only_not_query',
                      'recovered_complete_numeric_archive_without_new_property_solve': True}
                write(metadata, files['cutoff_eval'])
        cutoff_data = _numeric_load(cutoff_path, files['cutoff_eval']['sha256'])
        if _fingerprint_packet(prefix['packet']) != prefix_hash:
            raise AssertionError('Query/property readout changed actual unquenched cutoff.')
        ready = {'schema': SCHEMA, 'status': 'prefix_ready', 'binding': binding, 'actual_utc': now(),
                 'files': files, 'prefix_actual_packet_sha256': prefix_hash,
                 'unique_native_segment_attempted': 3, 'unique_native_segment_completed': 3,
                 'new_native_segment_attempted_this_dispatch': sum(c['native_segment_attempted'] for c in dispatched),
                 'new_native_segment_completed_this_dispatch': sum(c['native_segment_completed'] for c in dispatched),
                 'held_futures_not_yet_generated': role == 'held', 'teacher_risk_role': 'EVALonly' if role == 'held' else 'train'}
        if not (folder/'PREFIX_READY.json').exists():
            write(folder/'PREFIX_READY.json', ready)
        if phase == 'prefixonly':
            return ready
        if role == 'held':
            _verify_reveal(freeze)
            if not (folder/'REVEAL_AUTHORIZATION.json').exists():
                write(folder/'REVEAL_AUTHORIZATION.json', {'actual_utc': now(), 'binding': binding,
                                                          'prediction_freeze': freeze})
        repeats = 2 if role == 'train' else 4
        future_values = np.full((repeats, 6, 3, 6), np.nan)
        full_g = np.full((repeats, 6, 3, 5), np.nan)
        raw_g = np.full((repeats, 6, 3, 5), np.nan)
        valid = np.zeros((repeats, 6, 3), bool)
        errors, first_branch = [], None
        for repeat in range(repeats):
            future_seed = base+2000+200*repeat
            for action_index, action in enumerate(ACTIONS):
                keep_teacher = action_index == 0 and repeat == 0
                name = f'FUTURE_a{action_index:02d}_r{repeat:02d}'
                result, info = _stage(folder, name, binding,
                    lambda action=action, future_seed=future_seed, keep_teacher=keep_teacher:
                    formation_rollout(prefix['packet'], action, config, future_seed,
                         properties=lambda p: property_readout(p, config),
                         record_control=bool(job.get('validate_native_record', False)) and keep_teacher,
                         record_teacher=keep_teacher, mother_id=f'{role}:{mother}'), dispatched)
                files[name] = info; physical_stages.append(result['cost'])
                row, g, original_g, ok, err = _branch_check(result, prefix['packet'], action, config, future_seed)
                future_values[repeat, action_index] = row; full_g[repeat, action_index] = g
                raw_g[repeat, action_index] = original_g
                valid[repeat, action_index] = ok
                errors.append({'repeat': repeat, 'action': action_index, 'errors': err})
                if keep_teacher:
                    first_branch = result
                if (_fingerprint_packet(prefix['packet']) != prefix_hash
                        or any(_fingerprint_packet((s, *prefix['packet'][1:])) != past_hashes[str(t)]
                               for t, s in prefix['frames'].items())):
                    raise AssertionError('A future action changed its common observed microscopic past.')
                del result
        teacher = _teacher_arrays(prefix, first_branch)
        psi_name = 'PSI_TRAIN' if role == 'train' else 'PSI_EVAL'
        psi_path = folder/(psi_name+'.npz')
        if not psi_path.exists():
            files['psi'] = _numeric_save(psi_path, **teacher, mother_id=np.array(mother), role=np.array(role))
            write(folder/(psi_name+'.json'), {**files['psi'], 'binding': binding,
                 'actual_utc': now(), 'training_allowed': role == 'train', 'branch_action': 0, 'branch_repeat': 0,
                 'prefix_counted_once': True, 'event_row_columns': ['phase0prefix1future', 'interval', 'span',
                   'time', 'tag', 'source_site', 'target_site', 'geometry_time', 'selected_primitive']})
        else:
            metadata = folder/(psi_name+'.json')
            if metadata.exists():
                files['psi'] = read(metadata)
            else:
                saved = _numeric_load(psi_path)
                if int(saved['mother_id']) != mother or str(saved['role']) != role:
                    raise ValueError('Interrupted saved risk arrays have another mother identity.')
                files['psi'] = {'path': str(psi_path.resolve()), 'sha256': sha(psi_path),
                     'bytes': psi_path.stat().st_size, 'binding': binding, 'actual_utc': now(),
                     'training_allowed': role == 'train', 'branch_action': 0, 'branch_repeat': 0,
                     'prefix_counted_once': True, 'recovered_complete_numeric_archive_without_physics': True}
                write(metadata, files['psi'])
        query = _numeric_load(query_path, files['query']['sha256'])
        target = np.concatenate((np.broadcast_to(cutoff_data['cutoff_values'], (repeats, 6)),
                                 future_values.reshape(repeats, 108)), axis=1)
        values_path = folder/'VALUES.npz'
        if not values_path.exists():
            files['values'] = _numeric_save(values_path, X=query['X'], measured=query['measured'],
                future_values=future_values, cutoff_values=cutoff_data['cutoff_values'], target114=target,
                G_tau=full_g, G_tau_raw=raw_g, cutoff_G_tau=cutoff_data['cutoff_G_tau'],
                cutoff_G_tau_raw=cutoff_data['cutoff_G_tau_raw'], mechanics_valid=valid,
                cutoff_mechanics_valid=cutoff_data['mechanics_valid'], probe_tau=np.array(PROBE_TAU),
                actions=np.asarray(ACTIONS), future_times=np.asarray(FUTURE_TIMES), channels=np.asarray(CHANNELS),
                mother_id=np.array(mother), role=np.array(role))
            write(folder/'VALUES.json', {**files['values'], 'binding': binding,
                 'actual_utc': now(), 'layout': data_contract()['target114'], 'mechanics_errors': errors})
        else:
            metadata = folder/'VALUES.json'
            if metadata.exists():
                files['values'] = read(metadata)
            else:
                saved = _numeric_load(values_path)
                if int(saved['mother_id']) != mother or str(saved['role']) != role:
                    raise ValueError('Interrupted saved target arrays have another mother identity.')
                files['values'] = {'path': str(values_path.resolve()), 'sha256': sha(values_path),
                     'bytes': values_path.stat().st_size, 'binding': binding, 'actual_utc': now(),
                     'layout': data_contract()['target114'], 'mechanics_errors': errors,
                     'recovered_complete_numeric_archive_without_physics': True}
                write(metadata, files['values'])
        control = None
        if job.get('validate_native_record', False):
            if (role, mother) != ('train', 0):
                raise ValueError('Native-record bridge fixture belongs only to firsttrainmother.')
            from aam_formation_cycle_core import check_teacher_annotation_equivalence
            control_path = folder/'NATIVE_RECORD_CONTROL.json'
            if control_path.exists():
                control = read(control_path)
            else:
                attempt = folder/'NATIVE_RECORD_CONTROL_ATTEMPTED.json'
                if attempt.exists():
                    raise RuntimeError('Unresolved native-record control; no automatic repeat.')
                write(attempt, {'actual_utc': now(), 'binding': binding, 'extra_native_maximum': 6})
                control = check_teacher_annotation_equivalence([*prefix['teacher_records'], *first_branch['teacher_records']])
                write(control_path, {'actual_utc': now(), 'binding': binding, **control})
            files['native_record_control'] = {'path': str(control_path.resolve()), 'sha256': sha(control_path),
                                              'bytes': control_path.stat().st_size}
            if (control.get('passed') is not True or control.get('extra_native_segment_attempted') != 6
                    or control.get('extra_native_segment_completed') != 6):
                raise AssertionError('Firstmother originalnative-vs-teacherannotation control failed; preserve raw and stop.')
        unique_attempts = sum(c['native_segment_attempted'] for c in physical_stages)
        unique_completed = sum(c['native_segment_completed'] for c in physical_stages)
        if unique_attempts != 3+18*repeats or unique_completed != unique_attempts:
            raise ValueError('Fixed oneprefix/sixactions/reference-repeat physical cost failed.')
        report = {'schema': SCHEMA, 'status': 'complete', 'binding': binding, 'actual_utc': now(),
                  'files': files, 'root_registration_sha256': binding['root_registration_sha256'],
                  'unique_native_segment_attempted': unique_attempts, 'unique_native_segment_completed': unique_completed,
                  'new_native_segment_attempted_this_dispatch': sum(c['native_segment_attempted'] for c in dispatched),
                  'new_native_segment_completed_this_dispatch': sum(c['native_segment_completed'] for c in dispatched),
                  'kernel_equivalence_extra': control,
                  'actual_prefix_count': 1, 'future_branches': 6*repeats,
                  'mechanical_snapshots': 1+18*repeats,
                  'valid_mechanical_snapshots': int(bool(cutoff_data['mechanics_valid']))+int(valid.sum()),
                  'invalid_mechanical_snapshots': int(not bool(cutoff_data['mechanics_valid']))+int((~valid).sum()),
                  'functional_values_finite': bool(np.isfinite(target).all()),
                  'prediction_freeze': freeze, 'teacher_phase_rows': [len(prefix['teacher_opportunity']), len(first_branch['teacher_opportunity'])],
                  'elapsed_seconds_this_dispatch': perf_counter()-started,
                  'raw_state_scope': 'Actual16fields/fullbuffers andriskrecords retained;no quench/replay/postgraft.',
                  'units_scope': 'q/conversionmicroscopic;Gstressfreefrozentangentmodelunits,not measuredmodulus.'}
        write(final_path, report)
        event(folder, 'mother_complete', unique_native=unique_completed,
              new_native=report['new_native_segment_completed_this_dispatch'], mechanics_invalid=report['invalid_mechanical_snapshots'])
        return report
    except BaseException as error:
        failure = {'actual_utc': now(), 'binding': binding, 'phase': phase, 'error_type': type(error).__name__,
                   'error': str(error), 'files': files, 'completed_stage_costs': physical_stages,
                   'new_stage_costs_this_dispatch': dispatched,
                   'elapsed_seconds_this_dispatch': perf_counter()-started, 'no_automatic_retry': True}
        if not (folder/'FAILED.json').exists():
            write(folder/'FAILED.json', failure)
        event(folder, 'mother_failed', error_type=type(error).__name__, error=str(error))
        raise
    finally:
        lock.unlink()


def _folder(receipt_or_dir):
    if isinstance(receipt_or_dir, dict):
        binding = receipt_or_dir['binding']
        root = Path(binding['root_registration']).parent
        return root/binding['role']/f"parent_{binding['mother_id']:03d}"
    path = Path(receipt_or_dir)
    return path if path.is_dir() else path.parent


def load_query(receipt_or_dir):
    """Strict observation-only loader: never opens any physical/EVAL archive."""
    folder = _folder(receipt_or_dir)
    receipt = read(folder/'QUERY.json')
    data = _numeric_load(folder/'QUERY.npz', receipt['sha256'])
    if set(data) != {'X', 'measured', 'mother_id', 'role'} or data['X'].shape != (12,) or data['measured'].shape != (3, 4):
        raise ValueError('Only the original12noisy observation channels are query inputs.')
    return {'X': data['X'], 'measured': data['measured'], 'mother_id': int(data['mother_id']),
            'role': str(data['role']), 'query_sha256': receipt['sha256']}


def _values(receipt_or_dir):
    folder = _folder(receipt_or_dir)
    receipt = read(folder/'MOTHER_RECEIPT.json')
    if receipt['status'] != 'complete':
        raise ValueError('Incomplete mother has no complete trajectory target.')
    arrays = _numeric_load(folder/'VALUES.npz', receipt['files']['values']['sha256'])
    psi = _numeric_load(receipt['files']['psi']['path'], receipt['files']['psi']['sha256'])
    for key in ('teacher_opportunity', 'teacher_selection', 'teacher_event_rows', 'teacher_phase'):
        arrays[key] = psi[key]
    arrays['receipt'] = receipt
    arrays['mother_id'] = int(arrays['mother_id']); arrays['role'] = str(arrays['role'])
    return arrays


def load_training_parent(receipt_or_dir):
    folder = _folder(receipt_or_dir)
    if read(folder/'MOTHER_RECEIPT.json')['binding']['role'] != 'train':
        raise ValueError('Held actual states/risks/futures cannot become training data.')
    return _values(folder)


def load_eval(receipt_or_dir, *, prediction_freeze=None):
    folder = _folder(receipt_or_dir)
    receipt = read(folder/'MOTHER_RECEIPT.json')
    if receipt['binding']['role'] == 'held':
        supplied = _verify_reveal(prediction_freeze)
        if supplied != receipt['prediction_freeze']:
            raise ValueError('Held reveal belongs to another prediction freeze.')
    return _values(folder)
