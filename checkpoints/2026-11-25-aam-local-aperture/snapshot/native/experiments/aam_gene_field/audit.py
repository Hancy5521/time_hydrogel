"""Independently audit fixed AAm rule codes and replay fresh reaction journals."""

from argparse import ArgumentParser
from collections import Counter
from dataclasses import asdict, FrozenInstanceError
from datetime import datetime, timedelta, timezone
from hashlib import file_digest, sha256
from pathlib import Path
import json

import numpy as np


TIMES = (.65, 1.35, 2.05, 3.25, 4.85, 7.20)
LEVELS = (60, 76)
AAM, MBAA, INITIAL_MBAA = 64, 14, 8
MOLECULES = AAM + MBAA
VINYL_SITES = AAM + 2 * MBAA
FEED_TIME = 2.30
CAPTURE, INNER, TIME_STEP = 1.42, .96, .025
EPS = 2e-13
PATH_SCALES = np.array([.20, .06, .10, .18, .22, .22, .12, .12, .25, .30])
STRUCTURE_COLUMNS = np.array([2, 3, 4, 5, 6, 7])
FORECAST_METHODS = ('nominal', 'ridge', 'kernel', 'neural', 'neural_center')


def digest(path):
    with path.open('rb') as stream:
        return file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def timestamp(receipt):
    utc = datetime.fromisoformat(receipt['actual_utc'])
    local = datetime.fromisoformat(receipt['asia_shanghai'])
    if utc.utcoffset() != timedelta(0) or utc != local:
        raise ValueError('Receipt lacks a consistent UTC and local time')
    return utc


def write(path, record):
    utc = datetime.now(timezone.utc)
    payload = json.dumps({
        'actual_utc': utc.isoformat(),
        'asia_shanghai': utc.astimezone(timezone(timedelta(hours=8))).isoformat(),
        **record,
    }, indent=2, allow_nan=False) + '\n'
    with path.open('x', encoding='utf-8') as stream:
        stream.write(payload)


def require(statement, message):
    if not statement:
        raise ArithmeticError(message)


class JournalGraph:
    """Distinct site owners and molecule-node connectivity from committed events."""

    def __init__(self, initiator_budget):
        self.owner = [-1] * (2 * MOLECULES)
        self.active = [False] * (2 * MOLECULES)
        self.parent = list(range(MOLECULES))
        self.adjacency = [[] for _ in range(MOLECULES)]
        self.present = AAM + INITIAL_MBAA
        self.births = 0
        self.used = 0
        self.edges = 0
        self.counts = Counter()
        self.winding = False
        self.budget = initiator_budget

    def root(self, node):
        while self.parent[node] != node:
            node = self.parent[node]
        return node

    def usable_site(self, site):
        return 0 <= site < 2 * self.present and (site >= 2 * AAM or site % 2 == 0)

    def record_edge(self, start, stop, lift):
        one, two = self.root(start), self.root(stop)
        if one != two:
            self.parent[two] = one
        elif not self.winding:
            # A previously connected component with zero winding has consistent
            # integer offsets along all its paths. The new closing edge is tested
            # against one path using adjacency, independent of native weighted UF.
            search = [(start, (0, 0, 0))]
            seen = {start}
            found = False
            while search:
                node, offset = search.pop()
                if node == stop:
                    self.winding = offset != lift
                    found = True
                    break
                for neighbor, step in self.adjacency[node]:
                    if neighbor not in seen:
                        seen.add(neighbor)
                        search.append((neighbor, tuple(a+b for a,b in zip(offset,step,strict=True))))
            require(found, 'Event edge connected topology differs from explicit path reachability')
        self.adjacency[start].append((stop, lift))
        self.adjacency[stop].append((start, tuple(-i for i in lift)))
        self.edges += 1

    def accept(self, row):
        age, raw_tag, raw_start, raw_stop, distance, band, midpoint, occupants, before_start, before_stop, *_ = row
        tag, start, stop = map(int, (raw_tag, raw_start, raw_stop))
        lift = tuple(int(v) for v in row[10:13])
        self.counts[tag] += 1
        if tag == 0:
            require(start == -1 and stop == self.present < MOLECULES and age == FEED_TIME,
                    'Unexpected feed, duplicate molecule, or wrong feed clock')
            require((distance, band, midpoint, occupants, before_start, before_stop, *lift)
                    == (-1., -1., FEED_TIME, -1., -1., -1., 0, 0, 0), 'Corrupt feed event')
            self.present += 1
            return

        require(self.usable_site(stop), 'Reaction target is absent or chemically invalid')
        require(abs(age-midpoint) <= TIME_STEP/2+EPS, 'Reaction occurred away from its Brownian slab')
        if tag in (1, 2):
            require(start == -1 and self.owner[stop ^ 1] < 0
                    and tag == 1 + int(stop >= 2 * AAM), 'Illegal initiating event')
            require((distance, band, before_start, *lift) == (-1., 0., -1., 0, 0, 0),
                    'Initiation cannot record a covalent link to a prior molecule')
            require(self.births < self.budget, 'Initiation exhausted the immutable starting radical budget')
            lineage = self.births
            self.births += 1
        else:
            require(self.usable_site(start) and self.active[start] and start != stop,
                    'Reaction source was not an independent active tip')
            require(tag == 8 or start // 2 != stop // 2, 'Illegal same-molecule covalent bond')
            require(0. <= distance < CAPTURE and band == int(distance >= INNER),
                    'Event outside the registered native capture bands')
            lineage = self.owner[start]
            require(before_start == lineage, 'Recorded source lineage is inconsistent')
            self.active[start] = False

        pendant = tag in (5, 6, 9)
        partner = self.owner[stop ^ 1] if pendant else self.owner[stop]
        require(before_stop == partner, 'Recorded prior target lineage is inconsistent')
        require(0 <= occupants <= (4 if pendant else 0), 'Invalid registered occupancy bucket')
        if tag in (7, 8):
            require(self.active[stop], 'Termination needs two still-active vinyl sites')
            self.active[stop] = False
        else:
            require(self.owner[stop] < 0, 'Vinyl site reacted more than once')
            if pendant:
                require(stop >= 2 * AAM and partner >= 0, 'Pendant tag requires a once-attached MBAA')
                different_component = self.root(start//2) != self.root(stop//2)
                require(not different_component or lineage != partner, 'A lineage spans disconnected chemical components')
                expect = 5 if different_component else (6 if lineage == partner else 9)
                require(tag == expect, 'Pendant event tag disagrees with pre-event graph and lineages')
            elif tag in (3, 4):
                require(tag == 3 + int(stop >= 2 * AAM) and self.owner[stop ^ 1] < 0,
                        'Ordinary propagation tag disagrees with its target functionality')
            self.owner[stop] = lineage
            self.active[stop] = True
            self.used += 1
        if start >= 0 and tag != 8:
            self.record_edge(start//2, stop//2, lift)
        else:
            require(lift == (0, 0, 0), 'A non-bond event cannot add a directed lattice lift')

    def summary(self):
        lengths = Counter(self.owner[0:2*AAM:2])
        lengths.pop(-1, None)
        number = sum(lengths.values())
        ln = number / len(lengths) / AAM if lengths else 0.
        lw = sum(k*k for k in lengths.values()) / number / AAM if number else 0.
        roots = [self.root(i) for i in range(self.present)]
        reacted = Counter(roots[i] for i in range(self.present) if max(self.owner[2*i:2*i+2]) >= 0)
        largest = max(reacted.values(), default=0) / MOLECULES
        cycles = self.edges - self.present + len(set(roots))
        pendant = sum(sum(v >= 0 for v in self.owner[2*i:2*i+2]) == 1 for i in range(AAM, self.present))
        require(cycles >= 0 and self.used <= VINYL_SITES, 'Graph or site inventory conservation failure')
        return np.array((self.used/VINYL_SITES, ln, lw, largest, cycles/MBAA,
                        self.counts[5]/MBAA, self.counts[6]/MBAA, self.counts[9]/MBAA,
                        pendant/MBAA, float(self.winding)))


def replay(journal, initiator_budget=12):
    """Replay native event-only features; returned times are right-continuous."""
    require(journal.ndim == 2 and journal.shape[1] == 13 and np.isfinite(journal).all(),
            'Expected a finite N x 13 native reaction journal')
    require(np.all(np.diff(journal[:, 0]) >= 0)
            and np.all((journal[:, 0] >= 0) & (journal[:, 0] <= TIMES[-1])), 'Misordered or future event')
    integer_columns = journal[:, [1, 2, 3, 5, 7, 8, 9, 10, 11, 12]]
    require(np.array_equal(integer_columns, np.rint(integer_columns))
            and np.isin(journal[:, 1], np.arange(10)).all(), 'Undeclared or fractional event descriptor')
    graph = JournalGraph(initiator_budget)
    levels = np.full((len(LEVELS), 3), np.nan)
    cursor = 0
    output = []
    for time in TIMES:
        while cursor < len(journal) and journal[cursor, 0] <= time:
            previous = graph.used
            graph.accept(journal[cursor])
            if graph.used != previous and graph.used in LEVELS:
                current = graph.summary()
                levels[LEVELS.index(graph.used)] = (journal[cursor, 0], current[2], current[3])
            cursor += 1
        output.append(graph.summary())
    require(cursor == len(journal) and graph.present == MOLECULES, 'Unconsumed or missing feed ledger')
    require(sum(graph.active) == graph.births - 2*(graph.counts[7]+graph.counts[8]),
            'Free radical conservation disagrees with initiation and termination events')
    return np.stack(output), levels, dict(graph.counts)


def addressed_rng(namespace, part):
    payload = sha256(f'{namespace}/{part}'.encode()).digest()
    words = np.frombuffer(payload, dtype='<u4').tolist()
    return np.random.default_rng(np.random.SeedSequence(words))


def radical_budget(seed):
    # Original simulate() allocates the initial molecule prior to spawn child 1.
    prior = np.random.SeedSequence(int(seed)).spawn(5)[1]
    return int(np.random.default_rng(prior).integers(6, 13))


def qualify():
    """Check decoder immutability and event replay on synthetic engineering fixtures."""
    import field

    config = read(field.ORIGIN)['physics']
    before = json.dumps(config, sort_keys=True)
    env = {k: asdict(value) for k,value in field.ENV.items()}
    nominal_rates, nominal_factors = field.Rule(0., 0.).decode(config)
    np.testing.assert_array_equal(nominal_rates, config['rates'])
    np.testing.assert_array_equal(nominal_factors, config['factors'])
    qualification_codes = ((0., 0.), (-.7, 0.), (.7, 0.), (0., -.9), (0., .9))
    untouched = np.array([0, 1, 3, 4, 5, 6, 7])
    max_delta = 0.
    laws = 0
    for labelled in qualification_codes:
        code = np.asarray(labelled)
        rule = field.Rule(*code)
        try:
            rule.growth = 0.3
        except FrozenInstanceError:
            pass
        else:
            raise ArithmeticError('The decoded material rule must not admit in-simulation mutation')
        rates, factors = rule.decode(config)
        activity = np.arange(5)/(0.8+np.arange(5))
        expected = nominal_factors * np.exp(code[1] * np.array((1., -.5, -.5))[:, None] * activity)
        np.testing.assert_array_equal(rates[untouched], nominal_rates[untouched])
        np.testing.assert_allclose(rates[2], nominal_rates[2]*np.exp(code[0]), atol=1e-15, rtol=1e-15)
        np.testing.assert_allclose(factors, expected, atol=1e-15, rtol=1e-15)
        np.testing.assert_array_equal(factors[:, 0], nominal_factors[:, 0])
        code.setflags(write=False)
        for name in env:
            trajectory, landmark, journal = field.simulate(code, name, 872633501, config, journal=True)
            actual, marks, _ = replay(journal, radical_budget(872633501))
            np.testing.assert_allclose(actual, trajectory, atol=EPS, rtol=0.)
            np.testing.assert_allclose(marks, landmark, atol=EPS, rtol=0., equal_nan=True)
            max_delta = max(max_delta, float(np.max(np.abs(actual-trajectory))))
            laws += 1
    require(before == json.dumps(config, sort_keys=True) and env == {k: asdict(v) for k,v in field.ENV.items()},
            'Base config or controlled environment mutated during material formation')
    scratch_rates, scratch_factors = field.Rule(0., 0.).decode(config)
    scratch_rates[0, 0] = -1.
    scratch_factors[0, 0] = -1.
    fresh_rates, fresh_factors = field.Rule(0., 0.).decode(config)
    np.testing.assert_array_equal(fresh_rates, nominal_rates)
    np.testing.assert_array_equal(fresh_factors, nominal_factors)
    return {'engineering_fixture_paths': laws, 'maximum_native_replay_difference': max_delta,
            'decoder_changes_only_growth_row_and_pendant_allocation': True, 'immutable_rule_config_environment': True}


def bound_file(receipt, directory):
    file = Path(receipt['path']).resolve(strict=True)
    if file.parent != directory or digest(file) != receipt['sha256']:
        raise ValueError('Frozen file did not match its local path or SHA256')
    if 'bytes' in receipt and file.stat().st_size != receipt['bytes']:
        raise ValueError('Frozen file size differs from receipt')
    return file


def distances(a, b):
    return np.sqrt(np.square(a[:, None] - b[None, :]).sum(axis=-1))


def ensemble_energy(ensemble, reference):
    prediction = ensemble.reshape((*ensemble.shape[:2], -1))
    outcome = reference.reshape((*reference.shape[:2], -1))
    result = []
    for p, y in zip(prediction, outcome, strict=True):
        expected = distances(y, p).mean()
        spread = distances(p, p).mean()
        result.append((expected-.5*spread)/np.sqrt(p.shape[1]))
    return np.asarray(result)


def landmark_features(rows):
    reached = np.isfinite(rows[..., 0])
    values = np.where(reached[..., None], rows, 0.) / [2.5, .10, .20]
    return np.concatenate((reached[..., None], values), axis=-1)


def audit_forecast(out, predictions, truth, primary, registered):
    forecast_freeze = read(out/'C_FORECAST_FROZEN.json')
    posterior_freeze = read(out/'RULE_PREDICTIONS_FROZEN.json')
    learning = read(out/'LEARNING_REGISTERED.json')
    oracle_freeze = read(out/'ORACLE_C_FROZEN.json')
    require(forecast_freeze['no_held_trajectory_seen'] and posterior_freeze['no_held_rules_seen']
            and not posterior_freeze['future_environment_seen'] and oracle_freeze['only_after_forecast'],
            'Prediction provenance does not separate the held rule or trajectory')
    for name, h in learning['source_hashes'].items():
        require(digest(Path(name)) == h, f'Learning code or input changed: {name}')
    require(posterior_freeze['registration_sha256'] == digest(out/'LEARNING_REGISTERED.json')
            and forecast_freeze['posterior_freeze_sha256'] == digest(out/'RULE_PREDICTIONS_FROZEN.json'),
            'Unbound rule predictor')
    require(tuple(learning['methods']) == FORECAST_METHODS, 'Unknown comparison family')
    np.testing.assert_array_equal(learning['forecast_scales'], PATH_SCALES)
    require(timestamp(learning) < timestamp(posterior_freeze) < timestamp(forecast_freeze),
            'Posterior forecast preceded its declared sources')
    require(timestamp(forecast_freeze) < timestamp(oracle_freeze) < timestamp(primary),
            'Privileged rule evaluation must follow blind forecast and precede scoring')
    rule_file = bound_file(posterior_freeze['prediction'], out)
    oracle_file = bound_file(oracle_freeze['oracle'], out)
    score_file = bound_file(primary['motherscores'], out)
    seeds = addressed_rng(registered['namespace'], 'forecast_seed').integers(
        0, 2**63-1, (registered['test_mats'], registered['predict_draws']), dtype=np.int64)
    ms = (*FORECAST_METHODS, 'known_rule')
    scores, errors = {}, []
    with (np.load(predictions, allow_pickle=False) as forecast, np.load(oracle_file, allow_pickle=False) as oracle,
          np.load(rule_file, allow_pickle=False) as rules, np.load(truth, allow_pickle=False) as target,
          np.load(score_file, allow_pickle=False) as scored):
        np.testing.assert_array_equal(forecast['methods'], FORECAST_METHODS)
        np.testing.assert_array_equal(forecast['seeds'], seeds)
        np.testing.assert_array_equal(oracle['seeds'], seeds)
        require(not np.intersect1d(target['seeds'], seeds).size, 'Forecast and truth tapes overlap')
        y, ym, g = (target[k] for k in ('trajectory', 'landmark', 'codes'))
        path, lm = (np.concatenate((forecast[k], oracle[k][None]), axis=0) for k in ('trajectory', 'landmark'))
        np.testing.assert_array_equal(rules['nominal'], np.zeros_like(rules['nominal']))
        np.testing.assert_array_equal(rules['neural_center'], np.broadcast_to(
            rules['neural'].mean(axis=1, keepdims=True), rules['neural'].shape))
        for code in FORECAST_METHODS:
            require(rules[code].shape == (*seeds.shape, 2) and np.isfinite(rules[code]).all()
                    and np.max(np.abs(rules[code])) <= 1.10, 'Rule forecast outside the decoder support')
        values = {}
        for name, future, landmarks in zip(ms, path, lm, strict=True):
            replays = {
                'trajectory': ensemble_energy(future/PATH_SCALES, y/PATH_SCALES),
                'structure': ensemble_energy(future[..., STRUCTURE_COLUMNS]/PATH_SCALES[STRUCTURE_COLUMNS],
                                            y[..., STRUCTURE_COLUMNS]/PATH_SCALES[STRUCTURE_COLUMNS]),
                'landmark': ensemble_energy(landmark_features(landmarks), landmark_features(ym)),
            }
            for k, score in replays.items():
                errors.append(float(np.max(np.abs(score-scored[f'{name}/{k}']))))
            values[name] = replays
            aggregate = {f'{k}_energy': float(v.mean()) for k, v in replays.items()}
            rule_draws = rules[name] if name != 'known_rule' else g[:, None, :]
            aggregate['rule_rmse'] = np.sqrt(np.square(rule_draws.mean(axis=1)-g).mean(axis=0)).tolist()
            aggregate['rule_energy'] = float(ensemble_energy(rule_draws, g[:, None, :]).mean())
            r = np.isfinite(ym[..., 0]).mean(axis=1)
            q = np.isfinite(landmarks[..., 0]).mean(axis=1)
            aggregate['reach_Brier_60_76'] = ((q-r)**2+r*(1-r)).mean(axis=0).tolist()
            aggregate['mean_trajectory_error'] = np.abs(future.mean(axis=1)-y.mean(axis=1)).mean(axis=0).tolist()
            low, high = np.quantile(future, [.05, .95], axis=1)
            aggregate['marginal_90_predictive_coverage'] = ((y>=low[:, None])&(y<=high[:, None])).mean(axis=(0,1,2)).tolist()
            for measure, datum in aggregate.items():
                errors.append(float(np.max(np.abs(np.asarray(datum)-primary['scores'][name][measure]))))
            scores[name] = aggregate
        draws = np.random.default_rng(822751).integers(0, len(g), (10000, len(g)), dtype=np.int64)
        for name, receipt in primary['scores'].items():
            for control, targets in receipt['against'].items():
                for kind, old in targets.items():
                    diff = values[name][kind] - values[control][kind]
                    ci = np.quantile(diff[draws].mean(axis=1), [.025, .975])
                    errors.append(abs(diff.mean()-old['difference']))
                    errors.extend(np.abs(ci-old['paired_95']).tolist())
    require(max(errors) < 2e-12, f'Direct norm score or interval differs from primary by {max(errors)}')
    return {'maximum_difference': float(max(errors)), 'numeric_comparisons': len(errors),
            'metrics': scores, 'sources_sha256': {str(p): digest(p) for p in
                (rule_file, oracle_file, score_file, out/'LEARNING_REGISTERED.json',
                 out/'RULE_PREDICTIONS_FROZEN.json', out/'ORACLE_C_FROZEN.json')}}


def audit(out, prediction_receipt, score_receipt):
    # Do not even read the C-truth receipt until the designated public
    # prediction and primary score receipts have been checked.
    pred, score = read(out/prediction_receipt), read(out/score_receipt)
    locked_descriptor = next((pred[k] for k in ('predictions', 'prediction', 'data', 'forecast')
                              if k in pred and isinstance(pred[k], dict)), None)
    if locked_descriptor is None:
        raise ValueError('Prediction receipt must bind the exact frozen array')
    prediction = bound_file(locked_descriptor, out)
    require(timestamp(pred) < timestamp(score), 'Primary score predates its prediction freeze')
    require(not pred.get('truth_read', pred.get('held_truth_read', False)), 'Prediction was not frozen blind')
    registered = read(out/'REGISTERED.json')
    source_hashes = registered['sources']
    for name, expected in source_hashes.items():
        require(digest(Path(name)) == expected, f'Registered source changed: {name}')
    truth_receipt = read(out/'C_TRUTH_FROZEN.json')
    require(timestamp(pred) < timestamp(truth_receipt) < timestamp(score), 'Truth acquisition occurred outside designated freeze order')
    require(truth_receipt['forecast_freeze_sha256'] == digest(out/prediction_receipt), 'Truth does not bind the blind forecast')
    truth = bound_file(truth_receipt['truth'], out)
    require(truth.name == 'TRUTH_C.npz', 'Unexpected or historical C input')
    source, params = out/'REGISTERED.json', registered['physics']
    require(tuple(params['recipe'][key] for key in ('aam', 'mbaa', 'initial_mbaa')) == (AAM, MBAA, INITIAL_MBAA),
            'Registration does not use this native reactive-site ledger')
    require(tuple(params['motion'][key] for key in ('capture', 'inner', 'step')) == (CAPTURE, INNER, TIME_STEP),
            'Registration changed the original capture geometry')
    moments = ('trajectory', 'landmark')
    maximum = {name: 0. for name in moments}
    counts = Counter()
    with np.load(truth, allow_pickle=False) as archive:
        codes, trajectories, landmarks, journals, sizes = (archive[k] for k in
            ('codes', 'trajectory', 'landmark', 'journals', 'event_count'))
        seeds = archive['seeds'] if 'seeds' in archive.files else None
        n, repeats = registered['test_mats'], registered['truth_draws']
        expected_codes = addressed_rng(registered['namespace'], 'held_rules').uniform(-.85,.85,(n,2))
        np.testing.assert_array_equal(codes, expected_codes)
        require(trajectories.shape == (n,repeats,len(TIMES),10) and landmarks.shape == (n,repeats,len(LEVELS),3)
                and journals.ndim == 4 and journals.shape[:2] == (n,repeats) and journals.shape[-1] == 13
                and sizes.shape == (n,repeats) and np.issubdtype(sizes.dtype, np.integer)
                and np.all((sizes > 0)&(sizes <= journals.shape[-2])), 'Unexpected prospective C truth shape')
        require(seeds is None or seeds.shape == (n,repeats), 'Unaligned seed table')
        if seeds is not None:
            desired = addressed_rng(registered['namespace'], 'truth_seed').integers(0, 2**63-1, (n,repeats), dtype=np.int64)
            np.testing.assert_array_equal(seeds, desired)
        for index in np.ndindex(n,repeats):
            total = sizes[index]
            budget = radical_budget(seeds[index]) if seeds is not None else 12
            timed, levels, event_tags = replay(journals[index][:total], budget)
            expected_t, expected_levels = trajectories[index], landmarks[index]
            np.testing.assert_allclose(timed, expected_t, atol=EPS, rtol=0.)
            np.testing.assert_allclose(levels, expected_levels, atol=EPS, rtol=0., equal_nan=True)
            maximum['trajectory'] = max(maximum['trajectory'], float(np.max(np.abs(timed-expected_t))))
            finite = np.isfinite(levels)
            maximum['landmark'] = max(maximum['landmark'], float(np.max(np.abs(levels[finite]-expected_levels[finite]), initial=0.)))
            counts.update(event_tags)
    output_audit = audit_forecast(out, prediction, truth, score, registered)
    return {'mothers': n, 'truth_draws_per_mother': repeats, 'events_replayed': sum(counts.values()),
            'by_native_tag': counts, 'maximum_absolute_difference': maximum,
            'each_tape_radical_budget_replayed': seeds is not None,
            'independent_forecast_scoring': output_audit,
            'input_sha256': {str(p): digest(p) for p in (Path(__file__).resolve(), source, prediction, truth,
                out/prediction_receipt, out/score_receipt, out/'C_TRUTH_FROZEN.json')}, 'old_held_roles_read': 0, 'wet': 0}


if __name__ == '__main__':
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('fixtures', 'audit'))
    parser.add_argument('out', type=Path, nargs='?')
    parser.add_argument('--prediction-receipt', default='C_FORECAST_FROZEN.json')
    parser.add_argument('--score-receipt', default='SCORED.json')
    args = parser.parse_args()
    if args.stage == 'fixtures':
        result = qualify()
        if args.out:
            write(args.out, result)
    else:
        if args.out is None:
            parser.error('audit requires the prospectively registered experiment directory')
        root = args.out.resolve(strict=True)
        result = audit(root, args.prediction_receipt, args.score_receipt)
        write(root/'INDEPENDENT_C_FORMATION_AUDIT.json', result)
    print(result)
