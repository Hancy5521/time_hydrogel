"""Independently audit the tight-window receipt graph, choices, and group-level results."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy.optimize import brentq


ROOT = Path(__file__).resolve().parent
RUN = ROOT / 'runs/patch_tight_window_20261117T034850305493Z'
PEER = ROOT / 'runs/patch_window_thickness_delivery_replicate_20261117T025235612415Z'
ROLES = ('development', 'calibration', 'confirmation')
RANKS = ('upper_margin', 'posterior_violation')
BASELINE = 'depth_mid'
CLAUSES = ('off', 'on', 'soft', 'support', 'weak_off')


def read(path):
    return json.loads(path.read_bytes())


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def clock(stamp):
    return datetime.fromisoformat(stamp)


def stream_specs(plan, *, geometry_scan):
    before = plan['spatial_probe']['snapshot_noise_seed']
    variation = plan['spatial_probe']['shift_variation_seed']
    for slot, (role, cohort) in enumerate(plan['event_roles'].items(), start=1):
        prefix = (plan['registered_utc'], role)
        yield cohort['mechanism_seed'], (*prefix, 'mixture')
        yield cohort['nominal_seed'] + 772, (*prefix, 'nominal parent')
        yield cohort['shift_seed'] + 772, (*prefix, 'shift parent')
        for i in range(cohort['groups']):
            yield variation + cohort['hydraulic_salt'] + i, (*prefix, 'declared shift hydraulics', i)
            if not geometry_scan:
                for branch in ('test', 'shift_test'):
                    yield before + cohort['spatial_noise_salt'] + plan['seeds'][branch] + i, (*prefix, branch, i)
            else:
                yield [plan['tight']['witness_entropy'], slot, i], (*prefix, 'new matched witness', i)


def independent_streams(plan, peer_plan):
    registry = {}
    counts = {}
    for label, src, new in (('replication_640', peer_plan, False), ('tight_window', plan, True)):
        routes = list(stream_specs(src, geometry_scan=new))
        counts[label] = len(routes)
        for entropy, route in routes:
            state = np.random.PCG64(np.random.SeedSequence(entropy)).state['state']
            key = (state['state'], state['inc'])
            assert key not in registry, (route, registry.get(key))
            registry[key] = route
    return {'fresh_and_previous_generator_initial_states': counts, 'duplicates': 0,
            'continues_exact_pcg64_initial_state_check': True}


def replay_stream_preflight(plan, run):
    receipt = read(run / 'STREAM_CHECK.json')
    assert receipt['protocol_sha256'] == sha(run / 'REGISTERED.json')
    assert clock(plan['registered_utc']) <= clock(receipt['checked_utc'])
    assert list(plan['event_roles']) == list(ROLES)
    assert receipt['distinct_witness_tuples'] == sum(v['groups'] for v in plan['event_roles'].values())
    assert receipt['witness_entropy'] == plan['tight']['witness_entropy']
    peer = read(PEER / 'REGISTERED.json')
    return independent_streams(plan, peer)


def prepare(run):
    plan = read(run / 'REGISTERED.json')
    for name, expected in plan['hashes'].items():
        assert sha(ROOT / name) == expected, name
    task = read(ROOT / plan['task_path'])
    tight = plan['tight']
    goal = task['requests'][tight['request']]
    times = [goal[k] for k in ('install_end', 'work_start', 'work_end')]
    assert all(math.isclose(a, b, abs_tol=1e-13) for a, b in zip(times, (1.8, 3.2, 4.8), strict=True))
    assert plan['spatial_probe']['times'] == [.4, .8]
    assert min(times) > max(plan['spatial_probe']['times'])
    assert plan['number_of_states'] == 2 and tight['catalogue'] == 'thickness27'
    assert len(plan['thickness']['catalogues']['thickness27']) == 27
    assert all(r['shift_fraction'] == .5 for r in plan['event_roles'].values())
    return plan, {'signed_clauses': list(CLAUSES), 'requested_times': times,
                  'sensor_times': plan['spatial_probe']['times'],
                  'registered_hashes_verified': len(plan['hashes']),
                  'random_streams': replay_stream_preflight(plan, run)}


def selected_values(values, choices):
    ii, jj = np.indices(choices.shape)
    return values[ii, jj, choices]


def summarize(margins, choices, mask):
    good = selected_values(margins, choices) <= 0
    reached = (margins <= 0).any(axis=-1)
    active = mask.any(axis=1)
    bad = (~good & mask).any(axis=1)
    m, k, n = int(active.sum()), int(bad.sum()), len(margins)
    result = {'mothers': n, 'requests': good.size, 'active_mothers': m, 'unsafe_mothers': k,
              'accepted_requests': int(mask.sum()), 'unsafe_accepted_requests': int((mask & ~good).sum()),
              'successful_requests': int((mask & good).sum()), 'requests_with_feasible_action': int(reached.sum()),
              'raw_successes': int(good.sum()), 'active_fraction': m / n, 'unsafe_fraction': k / n,
              'unsafe_given_active': k / m if m else None}
    return result, mask & good


def rank_arrays(rows, method, rank):
    policy = [p['forecasts'][method]['banks'][rank] for p in rows]
    return np.asarray([x['choice'] for x in policy]), np.asarray([x['score'] for x in policy])


def independently_choose(margins, choices, scores, min_use, max_risk):
    admissible = []
    n_with_use = 0
    for score in sorted(set(scores.ravel())):
        cutoff = math.nextafter(float(score), math.inf)
        values, _ = summarize(margins, choices, scores < cutoff)
        if values['active_mothers'] < min_use:
            continue
        n_with_use += 1
        if values['unsafe_given_active'] <= max_risk:
            admissible.append({**values, 'cutoff': cutoff})
    selected = max(admissible, key=lambda d: (d['successful_requests'], -d['unsafe_mothers'], -d['cutoff']), default=None)
    return {'thresholds_with_required_groups': n_with_use, 'thresholds_below_design_risk': len(admissible), 'selected': selected}


def development(rows, margins, plan):
    limits = plan['tight']['development_criterion']
    rules = {}
    for name in rows[0]['forecasts']:
        variants, diagnostic = {}, {}
        ranking_candidates = []
        for rank in RANKS:
            choices, scores = rank_arrays(rows, name, rank)
            decision = independently_choose(margins, choices, scores, limits['min_development_mothers'], limits['design_risk'])
            variants[rank] = decision
            diagnostic[rank], _ = summarize(margins, choices, np.ones(scores.shape, dtype=bool))
            final = decision['selected']
            if final:
                ranking_candidates.append((final['successful_requests'], -final['unsafe_mothers'], rank))
        best = max(ranking_candidates)[2] if ranking_candidates else None
        rules[name] = {'variants': variants, 'primary_method': best,
                       'primary_cutoff': variants[best]['selected']['cutoff'] if best else None, 'ungated': diagnostic}
    competitors = []
    for name, rule in rules.items():
        if name != BASELINE and rule['primary_method']:
            accepted = rule['variants'][rule['primary_method']]['selected']
            competitors.append((accepted['successful_requests'], -accepted['unsafe_mothers'], name))
    return {'rules': rules, 'baseline': BASELINE if rules[BASELINE]['primary_method'] else None,
            'alternative': max(competitors)[2] if competitors else None}


def binomial_mass(n, p, lower, upper):
    return math.fsum(math.comb(n, i) * p**i * (1 - p)**(n - i) for i in range(lower, upper + 1))


def independent_gate(outcome):
    n, m, k = (outcome[x] for x in ('mothers', 'active_mothers', 'unsafe_mothers'))
    delta, risk, activity = .025, .1, .1
    risk_p = binomial_mass(m, risk, 0, k)
    activity_p = binomial_mass(n, activity, m, n)
    upper = 1. if m == k else brentq(lambda q: binomial_mass(m, q, 0, k) - delta, 0., 1., xtol=5e-15)
    lower = 0. if m == 0 else brentq(lambda q: binomial_mass(n, q, m, n) - delta, 0., 1., xtol=5e-15)
    return {'groups': n, 'active_mothers': m, 'unsafe_active_mothers': k,
            'accepted_requests': outcome['accepted_requests'], 'requests': outcome['requests'],
            'max_risk': risk, 'min_activity': activity, 'delta': delta, 'risk_p': risk_p, 'activity_p': activity_p,
            'joint_p': max(risk_p, activity_p), 'individual_risk_upper': upper, 'individual_activity_lower': lower,
            'certified': max(risk_p, activity_p) <= delta}


def compare_locked(rows, margins, plan, locked, phase):
    comparisons, hit = {}, {}
    for slot in ('baseline', 'alternative'):
        method = locked[slot]
        if method is None:
            continue
        rule = locked['rules'][method]
        rank, cutoff = rule['primary_method'], rule['primary_cutoff']
        actions, scores = rank_arrays(rows, method, rank)
        accept = scores < cutoff
        outcomes, hit[slot] = summarize(margins, actions, accept)
        record = {'method': method, 'rank': rank, 'cutoff': cutoff, 'outcomes': outcomes,
                  'choices': actions.tolist(), 'accepted': accept.tolist()}
        if phase == 'calibration':
            record['gate'] = independent_gate(outcomes)
        comparisons[slot] = record
    if phase == 'confirmation' and len(hit) == 2:
        reference, modified = hit['baseline'], hit['alternative']
        paired = np.asarray([math.fsum(int(b) - int(a) for a, b in zip(x, y, strict=True)) / len(x)
                             for x, y in zip(reference, modified, strict=True)])
        rng = np.random.default_rng(plan['tight']['bootstrap_seed'])
        resample = rng.integers(len(paired), size=(5000, len(paired)))
        interval = np.quantile(paired[resample].mean(axis=1), [.025, .975]).tolist()
        comparisons['paired'] = {'correctly_delivered_fraction_delta': math.fsum(paired) / len(paired),
                                 'ci95': interval, 'mothers': len(paired)}
    return comparisons


def match(value, saved, path='result'):
    if isinstance(value, dict):
        assert set(value) == set(saved), path
        return sum(match(v, saved[k], path + '.' + k) for k, v in value.items())
    if isinstance(value, (tuple, list)):
        assert len(value) == len(saved), path
        return sum(match(a, b, f'{path}[{i}]') for i, (a, b) in enumerate(zip(value, saved, strict=True)))
    if isinstance(value, float):
        assert math.isclose(value, saved, rel_tol=5e-11, abs_tol=4e-15), (path, value, saved)
    else:
        assert value == saved, (path, value, saved)
    return 1


def verify_row(row, plan):
    sensors = list(plan['tight']['sensors'])
    views = row['forecasts']
    assert set(views) == {*sensors, 'decision_directed', 'state_information'}
    weight = plan['risk_weight']
    checks = 0
    for method, values in views.items():
        for s, (p, m, q) in enumerate(zip(values['violation_probability'], values['mean'], values['upper'], strict=True)):
            assert len(p) == len(m) == len(q) == 27
            assert all(math.isfinite(x) for series in (p, m, q) for x in series)
            upper = min(range(len(p)), key=lambda i: (q[i], p[i], m[i]))
            bank = values['banks']['upper_margin']
            assert upper == bank['choice'][s] and bank['score'][s] == q[upper]
            alt = values['banks']['posterior_violation']
            i = alt['choice'][s]
            assert alt['score'][s] + 1e-12 >= p[i] + weight * max(0., m[i])
            assert abs(values['posterior_best_fixed_action'][s] - (1. - min(p))) < 1e-12
            assert -1e-12 <= values['posterior_any_action'][s] - values['posterior_best_fixed_action'][s] <= 1. + 1e-12
            if method not in sensors:
                field = 'expected_bayes_loss' if method == 'decision_directed' else 'expected_state_information'
                chooser = min if method == 'decision_directed' else max
                expect = chooser(range(len(sensors)), key=row[field][s].__getitem__)
                assert expect == row['chosen_sensor_indices'][method][s]
                same = views[sensors[expect]]
                for name in set(values) - {'banks'}:
                    assert np.allclose(values[name][s], same[name][s], rtol=1e-13, atol=1e-14)
                for r in RANKS:
                    assert values['banks'][r]['choice'][s] == same['banks'][r]['choice'][s]
                    assert math.isclose(values['banks'][r]['score'][s], same['banks'][r]['score'][s],
                                        rel_tol=1e-13, abs_tol=1e-14)
            checks += 1
    return checks


def diagnose(rows, clauses):
    y = clauses.max(axis=-1)
    exists = y.min(axis=-1) <= 0.
    views = [p['forecasts'][BASELINE] for p in rows]
    v = np.asarray([r['posterior_best_fixed_action'] for r in views])
    u = np.asarray([r['posterior_any_action'] for r in views])
    choice, _ = rank_arrays(rows, BASELINE, 'upper_margin')
    margins = selected_values(y, choice)
    limiting = {k: 0 for k in CLAUSES}
    for i, j in zip(*np.where(margins > 0), strict=True):
        x = clauses[i, j, choice[i, j]]
        limiting[CLAUSES[max(range(len(x)), key=x.__getitem__)]] += 1
    return {'feasible_requests': int(exists.sum()),
            'old_predicted_best_pass_median_given_feasible': float(np.median(v[exists])),
            'old_predicted_any_minus_best_median_given_feasible': float(np.median((u - v)[exists])),
            'old_predicted_best_below_half_given_actual_feasible': int((exists & (v < .5)).sum()),
            'limiting_true_clause_of_old_wrong_choices': limiting}


def run_audit(run, through):
    plan, overview = prepare(run)
    protocol = sha(run / 'REGISTERED.json')
    previous_clock = clock(read(run / 'STREAM_CHECK.json')['checked_utc'])
    upstream = {}
    checks = {}
    roles = {'preflight': (), 'development': ROLES[:1], 'calibration': ROLES[:2], 'all': ROLES}[through]
    for role in roles:
        forecast_path = run / f'{role.upper()}_FROZEN.json'
        score_path = run / f'{role.upper()}_SCORED.json'
        frozen, scored = read(forecast_path), read(score_path)
        names = (('forecast', frozen), ('score', scored))
        for stage, record in names:
            assert record['protocol_sha256'] == protocol and (record['role'], record['phase']) == (role, stage)
            begin, end = (clock(record[x]) for x in ('started_utc', 'completed_utc'))
            assert previous_clock <= begin <= end
            previous_clock = end
            if role != 'development':
                assert record['development_sha256'] == upstream['development']
            if role == 'confirmation':
                assert record['calibration_sha256'] == upstream['calibration']
        assert scored['frozen_sha256'] == sha(forecast_path)
        upstream[role] = sha(score_path)
        rows, labels = frozen['rows'], scored['rows']
        spec = plan['event_roles'][role]
        ids = [f'{role}-{i:03d}' for i in range(spec['groups'])]
        assert [p['mother'] for p in rows] == [p['mother'] for p in labels] == ids
        mechanism = np.random.default_rng(spec['mechanism_seed']).random(len(ids)) < spec['shift_fraction']
        assert mechanism.tolist() == [p['exchange_mechanism'] for p in labels]
        clauses = np.asarray([x['clauses'] for x in labels])
        assert clauses.shape == (len(ids), 2, 27, 5) and np.isfinite(clauses).all()
        margins = clauses.max(axis=-1)
        forecast_checks = sum(verify_row(r, plan) for r in rows)
        if role == 'development':
            locked = development(rows, margins, plan)
            final = locked
        else:
            final = compare_locked(rows, margins, plan, locked, role)
        fields = match(final, scored['results'])
        physical = diagnose(rows, clauses)
        fields += match(physical, scored['mechanism_diagnostics'])
        compact = {k: v for k, v in final.items() if k != 'rules'}
        for value in compact.values():
            if isinstance(value, dict):
                value.pop('accepted', None)
                value.pop('choices', None)
        checks[role] = {'outcomes': compact, 'diagnostics': physical, 'typed_fields_recomputed': fields,
                        'forecast_method_states_replayed': forecast_checks, 'shifted_mothers': int(mechanism.sum()),
                        'forecast_sha256': sha(forecast_path), 'score_sha256': upstream[role]}
    return {'registered_sha256': protocol, 'audit_scope': through, 'preflight': overview, 'checks': checks,
            'scope_limit': 'The archived full-action array for positive-margin expectation is absent. Its '
                           'posterior_violation choice and score are supplied by the freeze, and their threshold '
                           'selection, ordinal lower bound, and realized outcomes are independently checked. '
                           'Raw PVOI quadrature and simulator equations are outside this replay.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('through', choices=('preflight', 'development', 'calibration', 'all'))
    parser.add_argument('--run', type=Path, default=RUN)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    started = datetime.now(timezone.utc)
    report = run_audit(args.run, args.through)
    completed = datetime.now(timezone.utc)
    report.update(status='PASS', started_utc=started.isoformat(), completed_utc=completed.isoformat(), source_sha256=sha(Path(__file__)))
    if args.output:
        with args.output.open('x', encoding='utf-8') as file:
            json.dump(report, file, indent=2, allow_nan=False)
            file.write('\n')
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
