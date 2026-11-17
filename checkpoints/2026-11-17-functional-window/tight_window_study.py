"""Run a fresh tight-window study with separate development and gated validation."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timedelta, timezone
from itertools import repeat
from pathlib import Path

import numpy as np

from check_window_random_streams import preflight
from gel_reactive_port_timing import read_json, sha256
from gel_reactive_two_read_assay import write_new
from patch_window_event_study import current_plan, mixtures
from tight_window_witness import forecast, reveal
from window_event_calibration import certify_policy
from window_event_policy import development_choice, outcomes
from window_group_calibration import take_actions


BASE = Path('runs/patch_window_thickness_independent_20261117T023031799865Z/REGISTERED.json')
OLD = 'depth_mid'
ROLE_INDEX = {'development': 1, 'calibration': 2, 'confirmation': 3}
RANKS = ('upper_margin', 'posterior_violation')
PHASE = ('forecast', 'score')


def register():
    cfg = read_json(BASE)
    now = datetime.now(timezone.utc)
    cfg.update(registered_utc=now.isoformat(), registered_shanghai=now.astimezone(timezone(timedelta(hours=8))).isoformat())
    cfg['scientific_question'] = ('Can a different second scalar witness before the same deadline resolve '
                                  'an unserved install-to-work process window at the same mother-level risk target?')
    panels = dict(development=112, calibration=320, confirmation=160)
    cfg['event_roles'] = {role: dict(groups=n, shift_fraction=.5, nominal_seed=731400011 + 41099 * i,
        shift_seed=731400533 + 41099 * i, mechanism_seed=731401121 + 41099 * i,
        spatial_noise_salt=1234001 + 28673 * i, hydraulic_salt=7819201 + 99119 * i)
        for i, (role, n) in enumerate(panels.items())}
    cfg['tight'] = {
        'request': 2, 'catalogue': 'thickness27', 'witness_entropy': 4439796182,
        'first_sigma': .016, 'sensors': {'depth_mid': .016, 'depth_front': .016, 'depth_back': .016,
                                       'bulk_hydration': .016, 'log_conductance': .08},
        'bands': {'depth_mid': [.25, .5], 'depth_front': [0., .25], 'depth_back': [.5, .75],
                  'bulk_hydration': [0., 1.]},
        'observations': ('Five shared observations, then mid-depth water at 0.4 and one scalar on the same '
                         'standard witness at 0.8. Adaptive variants choose the second modality using the first '
                         'and the atlas alone. Common standard normal noise on alternatives for paired precision. '
                         'Same scalar count and deadline, yet assumed per-instrument noise and no equal cost claim.'),
        'selection': ('On 112 new development mothers, choose a rank and strict cutoff at no more than .05 '
                      'observed selective risk with 15 active mothers. Choose one alternative only by correct '
                      'delivery, then fewer unsafe mothers, then deterministic name. Apply identical selection '
                      'of rank and cutoff to the original two depth observations. If no method qualifies, retain '
                      'the development negative. Do not tune on the latter roles.'),
        'development_criterion': {'design_risk': .05, 'min_development_mothers': 15},
        'picks': {'comparison': 'Best alternate versus retargeted original two-depth packet. Original global policy was silent for this role.'},
        'confirmation': ('Frozen selectors and policies, same scenario mixture, no new mechanism claim. Test each '
                         'of two locked rules on the 320 independent calibration mothers using .10 risk and activity '
                         'targets at .025 each. Independently estimate held paired correct-delivery and 95% CI over '
                         '160 mothers. Same-risk improvement requires both gates, at least .025 point gain and CI '
                         'above zero. A lone passing rule supports only its own risk claim. Keep all failures.'),
        'frame': 'One operating request per material state, two states per mother, no cross-use or original multi-request risk guarantee.',
        'bootstrap_seed': 67921463,
    }
    dependencies = {*cfg['hashes'], BASE.as_posix(), 'tight_window_study.py', 'tight_window_witness.py',
                    'check_window_random_streams.py', 'audit_patch_reservoir_study.py'}
    cfg['hashes'] = {name: sha256(Path(name)) for name in sorted(dependencies)}
    keys = preflight(cfg)
    run = Path('runs') / now.strftime('patch_tight_window_%Y%m%dT%H%M%S%fZ')
    run.mkdir()
    write_new(run / 'REGISTERED.json', cfg)
    write_new(run / 'STREAM_CHECK.json', {
        'protocol_sha256': sha256(run / 'REGISTERED.json'), 'checked_utc': datetime.now(timezone.utc).isoformat(),
        'scalar_addresses_with_unused_original_depth_names': len(keys),
        'distinct_witness_tuples': sum(panels.values()), 'witness_entropy': cfg['tight']['witness_entropy'],
        'witness_key': ['fixed_entropy', 'positive_role_index', 'zero_based_mother_index'],
    })
    print(run.as_posix(), flush=True)


def bind(path, protocol, **extras):
    return {'protocol_sha256': sha256(protocol), 'started_utc': datetime.now(timezone.utc).isoformat(), **extras}


def open_rows(path, protocol):
    data = read_json(path)
    if data['protocol_sha256'] != sha256(protocol):
        raise ValueError(f'Wrong registration for {path}.')
    return data


def arrays(rows, method, rank):
    values = [r['forecasts'][method]['banks'][rank] for r in rows]
    return {'chosen': np.array([r['choice'] for r in values]), 'score': np.array([r['score'] for r in values])}


def develop(plan, rows, y):
    target = plan['tight']['development_criterion']
    result = {}
    for method in rows[0]['forecasts']:
        slates = {rank: arrays(rows, method, rank) for rank in RANKS}
        chosen = development_choice(y, slates, target)
        raw = {rank: outcomes(y, item['chosen'], np.ones(item['score'].shape, dtype=bool))
               for rank, item in slates.items()}
        result[method] = {**chosen, 'ungated': raw}
    baseline = result[OLD]['primary_method']
    others = []
    for method, policy in result.items():
        if method != OLD and policy['primary_method'] is not None:
            p = policy['variants'][policy['primary_method']]['selected']
            others.append((p['successful_requests'], -p['unsafe_mothers'], method))
    champion = max(others)[-1] if others else None
    return {'rules': result, 'baseline': OLD if baseline is not None else None, 'alternative': champion}


def evaluate(locked, rows, y, plan, role):
    results = {}
    accepted_pass = []
    for mode in ('baseline', 'alternative'):
        method = locked[mode]
        if method is None:
            continue
        entry = locked['rules'][method]
        policy = arrays(rows, method, entry['primary_method'])
        accepted = policy['score'] < entry['primary_cutoff']
        real = take_actions(y, policy['chosen'])
        result = {'method': method, 'rank': entry['primary_method'], 'cutoff': entry['primary_cutoff'],
                  'outcomes': outcomes(y, policy['chosen'], accepted),
                  'choices': policy['chosen'].tolist(), 'accepted': accepted.tolist()}
        if role == 'calibration':
            result['gate'] = certify_policy(real, accepted, max_risk=.10, min_activity=.10, delta=.025)
        results[mode] = result
        accepted_pass.append(accepted & (real <= 0.))
    if role == 'confirmation' and len(accepted_pass) == 2:
        delta = (accepted_pass[1].astype(float) - accepted_pass[0]).mean(axis=1)
        bootstrap = np.random.default_rng(plan['tight']['bootstrap_seed']).integers(len(delta), size=(5000, len(delta)))
        results['paired'] = {'correctly_delivered_fraction_delta': float(delta.mean()),
                             'ci95': np.quantile(delta[bootstrap].mean(axis=1), [.025, .975]).tolist(),
                             'mothers': len(delta)}
    return results


def diagnose(rows, clauses):
    """Decision ambiguity and the limiting physical clause on unchanged depth."""
    items = [r['forecasts'][OLD] for r in rows]
    p_any = np.asarray([x['posterior_any_action'] for x in items])
    p_act = np.asarray([x['posterior_best_fixed_action'] for x in items])
    true_feasible = (clauses.max(axis=-1).min(axis=-1) <= 0.)
    choice = arrays(rows, OLD, 'upper_margin')['chosen']
    physical = np.take_along_axis(clauses, choice[..., None, None], axis=-2).squeeze(-2)
    bad = physical.max(axis=-1) > 0.
    counts = np.bincount(physical.argmax(axis=-1)[bad], minlength=5)
    return {'feasible_requests': int(true_feasible.sum()),
            'old_predicted_best_pass_median_given_feasible': float(np.median(p_act[true_feasible])),
            'old_predicted_any_minus_best_median_given_feasible': float(np.median((p_any-p_act)[true_feasible])),
            'old_predicted_best_below_half_given_actual_feasible': int((true_feasible & (p_act<.5)).sum()),
            'limiting_true_clause_of_old_wrong_choices': dict(zip(('off','on','soft','support','weak_off'), counts.tolist(), strict=True))}


def run_phase(args):
    protocol = args.run / 'REGISTERED.json'
    plan = read_json(protocol)
    for name, h in plan['hashes'].items():
        if sha256(Path(name)) != h:
            raise ValueError(f'Changed input or source {name}.')
    role, phase = args.role, args.action
    if role != 'development':
        devfile = args.run / 'DEVELOPMENT_SCORED.json'
        dev = open_rows(devfile, protocol)['results']
        if dev['baseline'] is None and dev['alternative'] is None:
            raise ValueError('No rule existed when development was locked.')
    if role == 'confirmation':
        calfile = args.run / 'CALIBRATION_SCORED.json'
        cal = open_rows(calfile, protocol)
    name = args.run / f'{role.upper()}_{"FROZEN" if phase == "forecast" else "SCORED"}.json'
    if name.exists():
        raise FileExistsError(name)
    task = read_json(Path(plan['task_path']))
    local = {**current_plan(plan, role), 'tight_role': ROLE_INDEX[role]}
    mothers = list(mixtures(local, role, task))
    inputs = bind(name, protocol, phase=phase, role=role)
    if role != 'development':
        inputs['development_sha256'] = sha256(devfile)
    if phase == 'score':
        frozen_path = args.run / f'{role.upper()}_FROZEN.json'
        frozen = open_rows(frozen_path, protocol)
        inputs['frozen_sha256'] = sha256(frozen_path)
    if role == 'confirmation':
        inputs['calibration_sha256'] = sha256(calfile)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        fn = forecast if phase == 'forecast' else reveal
        rows = []
        for row in pool.map(fn, mothers, repeat(task), repeat(local)):
            rows.append(row)
            if len(rows) % 32 == 0:
                print(phase, role, len(rows), flush=True)
    inputs['rows'] = rows
    if phase == 'score':
        x = frozen['rows']
        if [a['mother'] for a in x] != [a['mother'] for a in rows]:
            raise ValueError('Labels do not match frozen prediction identities.')
        clauses = np.asarray([a['clauses'] for a in rows])
        y = clauses.max(axis=-1)
        inputs['results'] = develop(plan, x, y) if role == 'development' else evaluate(dev, x, y, plan, role)
        inputs['mechanism_diagnostics'] = diagnose(x, clauses)
    inputs['completed_utc'] = datetime.now(timezone.utc).isoformat()
    write_new(name, inputs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('register', *PHASE))
    parser.add_argument('run', nargs='?', type=Path)
    parser.add_argument('role', nargs='?', choices=tuple(ROLE_INDEX))
    parser.add_argument('--workers', type=int, default=12)
    args = parser.parse_args()
    if args.action == 'register':
        register()
    elif args.run is None or args.role is None:
        parser.error('Forecast and score require a run directory and a role.')
    else:
        run_phase(args)


if __name__ == '__main__':
    main()
