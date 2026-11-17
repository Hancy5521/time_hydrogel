"""Audit supervised tight-window experiments without fitting or production imports."""

import argparse
import ast
from datetime import datetime, timezone
import json
import math
from pathlib import Path

import numpy as np
from scipy.optimize import brentq

from audit_tight_window_20261117 import (
    PEER, RANKS, ROLES, ROOT, RUN, binomial_mass, clock, independently_choose,
    match, rank_arrays, read, sha, stream_specs, summarize,
)


PHASE = ROOT / 'runs/patch_tight_phase_20261117T035501438587Z'
COUPON = ROOT / 'runs/patch_tight_thin_coupon_20261117T035923687266Z'
PANELS = ('standard_depth', 'thin_depth', 'thin_electric')
METHODS = ('direct_depth', 'phase_depth', 'direct_electric', 'phase_electric')


def function_ast(file, name):
    parsed = ast.parse((ROOT / file).read_text(encoding='utf-8'))
    return next(node for node in parsed.body if isinstance(node, ast.FunctionDef) and node.name == name)


def calls(node):
    return {a.func.id for a in ast.walk(node) if isinstance(a, ast.Call) and isinstance(a.func, ast.Name)}


def strings(node):
    return {a.value for a in ast.walk(node) if isinstance(a, ast.Constant) and isinstance(a.value, str)}


def branch_ast(coupon_mode):
    study = 'tight_window_coupon_study.py' if coupon_mode else 'tight_window_phase_study.py'
    train = function_ast(study, 'training' if coupon_mode else 'bundle_training')
    assert {s for s in strings(train) if 'SCORED.json' in s} == {'DEVELOPMENT_SCORED.json'}
    assert f"'{ 'development' }'" in ast.unparse(train)
    prediction = function_ast(study, 'predictions')
    assert 'fit' in calls(prediction) and 'reveal' not in calls(prediction)
    driver = ast.unparse(function_ast(study, 'run_phase'))
    expected = "fn = inputs if args.action == 'forecast' else reveal" if coupon_mode else "fn = generate if args.action == 'forecast' else reveal"
    assert expected in driver
    extraction = function_ast('tight_window_phase_model.py', 'witness_inputs')
    values = strings(extraction)
    assert {'known', 'tight', 'witness_entropy', 'first_sigma', 'sensors'} <= values
    assert not {'clauses', 'exchange_mechanism', 'work_end'} & values
    signals = ast.unparse(function_ast('tight_window_witness.py', 'features'))
    assert "plan['spatial_probe']['times']" in signals and 'np.r_[0.0, t]' in signals
    assert "t[-1] >= task['requests'][0]['install_end']" in signals
    if coupon_mode:
        input_step = ast.unparse(function_ast(study, 'inputs'))
        assert "x[:, :-2]" in input_step and "['log_conductance'])[:, -2:]" in input_step
    return {'training_source_role': 'development', 'forecast_generator_excludes_reveal': True,
            'simulated_outcome_fields_absent_from_witness_output': True,
            'two_early_witness_times_before_install_end': True,
            'scope': 'Source tree pattern check and receipt replay. Model fitting and witness PDE are not rerun.'}


def pool_streams(run):
    pool, counts = {}, {}
    for path in (PEER, RUN, PHASE, COUPON):
        if path != run and path == COUPON:
            continue
        source = read(path / 'REGISTERED.json')
        rng = list(stream_specs(source, geometry_scan=path != PEER))
        counts[path.name] = len(rng)
        for e, label in rng:
            p = np.random.PCG64(np.random.SeedSequence(e)).state['state']
            key = (p['state'], p['inc'])
            assert key not in pool, (pool.get(key), label)
            pool[key] = label
    return {'full_prospective_addresses_with_realized_roles_subset': counts, 'distinct_initial_states': len(pool)}


def dev_locks(rows, y, plan, names):
    criteria = plan['tight']['development_criterion']
    rules = {}
    for name in names:
        stats, finalists, no_cutoff = {}, [], {}
        for rank in RANKS:
            choice, score = rank_arrays(rows, name, rank)
            values = independently_choose(y, choice, score, criteria['min_development_mothers'], criteria['design_risk'])
            stats[rank] = values
            no_cutoff[rank], _ = summarize(y, choice, np.ones(score.shape, dtype=bool))
            if values['selected']:
                win = values['selected']
                finalists.append((win['successful_requests'], -win['unsafe_mothers'], rank))
        pick = max(finalists)[2] if finalists else None
        rules[name] = {'variants': stats, 'primary_method': pick, 'primary_cutoff': stats[pick]['selected']['cutoff'] if pick else None,
                       'ungated': no_cutoff}
    return rules


def selected_panel(rules, suffix):
    stable_picks = []
    for name, config in rules.items():
        rank = config['primary_method']
        if name.endswith(suffix) and rank is not None:
            metrics = config['variants'][rank]['selected']
            stable_picks.append((metrics['successful_requests'], -metrics['unsafe_mothers'], name))
    return max(stable_picks)[-1] if stable_picks else None


def exact_gate(outcomes, delta):
    risk = activity = .10
    n, m, err = (outcomes[k] for k in ('mothers', 'active_mothers', 'unsafe_mothers'))
    rp = binomial_mass(m, risk, 0, err)
    ap = binomial_mass(n, activity, m, n)
    upper = 1. if err == m else brentq(lambda p: binomial_mass(m, p, 0, err) - delta, 0., 1., xtol=5e-15)
    lower = 0. if not m else brentq(lambda p: binomial_mass(n, p, m, n) - delta, 0., 1., xtol=5e-15)
    return {'groups': n, 'active_mothers': m, 'unsafe_active_mothers': err,
            'accepted_requests': outcomes['accepted_requests'], 'requests': outcomes['requests'], 'max_risk': risk,
            'min_activity': activity, 'delta': delta, 'risk_p': rp, 'activity_p': ap, 'joint_p': max(rp, ap),
            'individual_risk_upper': upper, 'individual_activity_lower': lower, 'certified': max(rp, ap) <= delta}


def evaluate(rows, y, role, selected, rules, seed, coupon_mode):
    output, accepted_good = {}, {}
    for key, method in selected.items():
        rank = rules[method]['primary_method']
        cutoff = rules[method]['primary_cutoff']
        choice, score = rank_arrays(rows, method, rank)
        keep = score < cutoff
        outcomes, accepted_good[key] = summarize(y, choice, keep)
        outcome = {'outcomes': outcomes, 'choices': choice.tolist(), 'accepted': keep.tolist()}
        if not coupon_mode:
            outcome.update(method=method, rank=rank, cutoff=cutoff)
        if role == 'calibration':
            outcome['gate'] = exact_gate(outcomes, .05 / 3 if coupon_mode else .025)
        output[key] = outcome
    old, new = ('thin_depth', 'thin_electric') if coupon_mode else ('baseline', 'alternative')
    if role == 'confirmation' and set((old, new)) <= set(accepted_good):
        old, new = accepted_good[old], accepted_good[new]
        change = np.asarray([math.fsum(int(b) - int(a) for a, b in zip(a_row, b_row, strict=True)) / len(a_row)
                             for a_row, b_row in zip(old, new, strict=True)])
        rng = np.random.default_rng(seed)
        boot = rng.integers(len(change), size=(5000, len(change)))
        bounds = np.quantile(change[boot].mean(axis=1), [.025, .975]).tolist()
        estimate = math.fsum(change) / len(change)
        if coupon_mode:
            output['primary_delta'] = {'estimate': estimate, 'ci95': bounds}
        else:
            output['paired'] = {'correctly_delivered_fraction_delta': estimate, 'ci95': bounds, 'mothers': len(change)}
    return output


def verify_predictions(rows, names, coupon_mode):
    for row in rows:
        assert set(row) == {'mother', 'forecasts'} and set(names) == set(row['forecasts'])
        for name in names:
            view = row['forecasts'][name]
            for rank in RANKS:
                chosen, risk = (view['banks'][rank][k] for k in ('choice', 'score'))
                assert len(chosen) == len(risk) == 2 and all(0 <= c < 27 for c in chosen)
                assert all(math.isfinite(s) for s in risk)
                if not coupon_mode:
                    for i, (q, p) in enumerate(zip(view['predicted_q90'], view['predicted_p_failure'], strict=True)):
                        assert len(q) == len(p) == 27 and all(0. <= value <= 1. for value in p)
                        if rank == 'upper_margin':
                            minimum = min(range(len(q)), key=lambda j: (q[j], p[j]))
                            assert minimum == chosen[i] and risk[i] == q[minimum]
                        else:
                            assert risk[i] >= p[chosen[i]]


def audit(run, through):
    path = run / 'REGISTERED.json'
    plan = read(path)
    coupon_mode = 'coupon' in plan
    names = PANELS if coupon_mode else METHODS
    if coupon_mode:
        training = plan['coupon']['training_source_runs']
        assert {(ROOT / name).resolve() for name in training} == {RUN, PHASE}
        assert plan['coupon']['witness_scale'] == .65
    else:
        training = [plan['phase']['training_run']]
        assert (ROOT / training[0]).resolve() == RUN
    for name, value in plan['hashes'].items():
        assert sha(ROOT / name) == value, name
    binding = sha(path)
    assert read(run / 'STREAM_CHECK.json')['protocol_sha256'] == binding
    last_clock = clock(plan['registered_utc'])
    training_mothers = 0
    for name in training:
        saved = ROOT / name / 'DEVELOPMENT_SCORED.json'
        labels = read(saved)
        assert (ROOT / name / 'REGISTERED.json').is_file()
        assert sha(ROOT / name / 'REGISTERED.json') == labels['protocol_sha256']
        assert sha(saved) == plan['hashes'][saved.relative_to(ROOT).as_posix()]
        assert labels['frozen_sha256'] == sha(ROOT / name / 'DEVELOPMENT_FROZEN.json')
        assert clock(labels['completed_utc']) < last_clock
        training_mothers += len(labels['rows'])
    assert training_mothers == (224 if coupon_mode else 112)
    assert plan['spatial_probe']['times'] == [.4, .8]
    assert len(plan['thickness']['catalogues']['thickness27']) == 27
    assert plan['number_of_states'] == 2 and plan['tight']['request'] == 2
    guard = {'training_mothers': training_mothers, 'checked_source_patterns': branch_ast(coupon_mode),
             'rng': pool_streams(run), 'hashed_dependencies': len(plan['hashes'])}
    completed = {}
    scopes = {'preflight': (), 'development': ROLES[:1], 'calibration': ROLES[:2], 'all': ROLES}
    for role in scopes[through]:
        if role != 'development' and not locked:
            assert not any((run / f'{r.upper()}_{s}.json').exists() for r in ROLES[1:] for s in ('FROZEN', 'SCORED'))
            break
        forecast, scored = (run / f'{role.upper()}_{ext}.json' for ext in ('FROZEN', 'SCORED'))
        lock, label = read(forecast), read(scored)
        for receipt in (lock, label):
            assert receipt['protocol_sha256'] == binding
            start, end = (clock(receipt[k]) for k in ('started_utc', 'completed_utc'))
            assert last_clock <= start <= end
            last_clock = end
            if role != 'development':
                assert receipt['development_sha256'] == completed['development']['score_sha256']
            if role == 'confirmation':
                assert receipt['calibration_sha256'] == completed['calibration']['score_sha256']
        assert label['frozen_sha256'] == sha(forecast)
        rows, raw = lock['rows'], label['rows']
        enrolled = plan['event_roles'][role]
        order = [f'{role}-{i:03d}' for i in range(enrolled['groups'])]
        assert [r['mother'] for r in rows] == [r['mother'] for r in raw] == order
        flags = np.random.default_rng(enrolled['mechanism_seed']).random(len(order)) < enrolled['shift_fraction']
        assert flags.tolist() == [r['exchange_mechanism'] for r in raw]
        values = np.asarray([r['clauses'] for r in raw])
        assert values.shape == (len(order), 2, 27, 5) and np.isfinite(values).all()
        y = values.max(axis=-1)
        for item in lock['training_diagnostics'].values():
            assert item.get('training_mothers', item.get('n_training_mothers')) == training_mothers
            if not coupon_mode:
                assert item['residual_draws'] == 2 * training_mothers
        verify_predictions(rows, names, coupon_mode)
        if role == 'development':
            rules = dev_locks(rows, y, plan, names)
            locked = {a: a for a, x in rules.items() if x['primary_method']} if coupon_mode else {
                key: method for key, method in zip(('baseline', 'alternative'),
                    (selected_panel(rules, 'depth'), selected_panel(rules, 'electric')), strict=True) if method}
            result = rules if coupon_mode else {'rules': rules, 'baseline': locked.get('baseline'), 'alternative': locked.get('alternative')}
            overview = {'locked_methods': locked, 'methods': {a: {r: {
                'ungated_successes': z['ungated'][r]['successful_requests'],
                'considered_cutoffs': z['variants'][r]['thresholds_with_required_groups'],
                'admissible_cutoffs': z['variants'][r]['thresholds_below_design_risk'],
            } for r in RANKS} for a, z in rules.items()}}
        else:
            result = evaluate(rows, y, role, locked, rules, plan['tight']['bootstrap_seed'], coupon_mode)
            overview = {k: {a: b for a, b in v.items() if a not in ('choices', 'accepted')}
                        for k, v in result.items()}
        checked = match(result, label['results'])
        completed[role] = {'overview': overview, 'verified_fields': checked,
                           'mothers': len(order), 'shifted': int(flags.sum()),
                           'available_requests': int((y.min(axis=-1) <= 0.).sum()),
                           'forecast_sha256': sha(forecast), 'score_sha256': sha(scored)}
    return {'mode': 'thin_coupon' if coupon_mode else 'phase_correction', 'through': through, 'protocol_sha256': binding,
            'eligibility': guard, 'roles': completed, 'status': 'PASS',
            'scope_limit': 'Frozen predictions and clauses are audited. Positive-margin risk arrays and full coupon action rankings are not saved, '
                           'so this replay validates the chosen ranks and threshold selection against frozen inputs without refitting.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('through', choices=('preflight', 'development', 'calibration', 'all'))
    parser.add_argument('--output', type=Path)
    a = parser.parse_args()
    stamp = datetime.now(timezone.utc)
    result = audit(a.run.resolve(), a.through)
    result.update(started_utc=stamp.isoformat(), completed_utc=datetime.now(timezone.utc).isoformat(),
                  source_sha256=sha(Path(__file__)))
    if a.output:
        with a.output.open('x', encoding='utf-8') as stream:
            json.dump(result, stream, ensure_ascii=False, allow_nan=False, indent=2)
            stream.write('\n')
    print(json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2))


if __name__ == '__main__':
    main()
