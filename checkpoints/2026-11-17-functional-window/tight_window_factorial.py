"""Preregister and replay a paired witness-thickness by sensing-channel study."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timedelta, timezone
from itertools import repeat
from pathlib import Path

import numpy as np

from check_window_random_streams import preflight
from gel_reactive_port_timing import read_json, sha256
from gel_reactive_two_read_assay import write_new
from patch_thickness_catalogue_benchmark import geometry
from tight_window_phase_model import design, fit, predictive_margins, rank, witness_inputs
from tight_window_phase_study import parents
from tight_window_study import RANKS, ROLE_INDEX, arrays, open_rows
from tight_window_witness import reveal
from window_event_calibration import certify_policy
from window_event_policy import development_choice, outcomes
from window_group_calibration import take_actions


SOURCE = Path('runs/patch_tight_thin_coupon_20261117T035923687266Z')
REPLICATION = Path('runs/patch_window_thickness_delivery_replicate_20261117T025235612415Z/REGISTERED.json')
PANELS = ('standard_depth', 'standard_electric', 'thin_depth', 'thin_electric')
CONTRASTS = {
    'electric_thickness_effect': (0, -1, 0, 1),
    'depth_thickness_effect': (-1, 0, 1, 0),
    'standard_electric_effect': (-1, 1, 0, 0),
    'thin_electric_effect': (0, 0, -1, 1),
    'interaction': (1, -1, -1, 1),
}


def register():
    now = datetime.now(timezone.utc)
    plan = read_json(SOURCE/'REGISTERED.json')
    settings = dict(development=160, calibration=360, confirmation=480)
    for i, (role, size) in enumerate(settings.items()):
        plan['event_roles'][role].update(
            groups=size,
            nominal_seed=1138004161 + 85001*i,
            shift_seed=1138005437 + 85001*i,
            mechanism_seed=1138006829 + 85001*i,
            spatial_noise_salt=15190041 + 12413*i,
            hydraulic_salt=15620721 + 42101*i,
        )
    plan.update(registered_utc=now.isoformat(),
                registered_shanghai=now.astimezone(timezone(timedelta(hours=8))).isoformat(),
                scientific_question='Which witness thickness and modality effects transfer to the same 27 finished patch candidates under the tight simulated operating window?')
    plan['tight']['witness_entropy'] = 6194527117
    plan['factorial'] = {
        'panels': PANELS,
        'contrasts': CONTRASTS,
        'primary': 'Thin versus standard thickness with electrical second reading. Also report both modality effects, depth-only thickness effect and thickness-by-modality interaction.',
        'primary_endpoint': 'Correct full-window action without gating; fixed posterior-violation ranking for all panels. Confirmation is retained regardless of development gate for this independently prespecified endpoint.',
        'utility': 'Paired, equally weighted mother bootstrap. Report point changes with marginal 95% intervals and Bonferroni 99% intervals across the five planned contrasts.',
        'same_information_budget': 'Five shared readings and one 0.4 depth reading plus one 0.8 reading on the same early witness. Within each thickness the first witness reading is identical for both modalities. Across thickness its physical value is allowed to change. Same clocks, outer bath condition and areal ion load. Same candidate patch geometry and training labels and same ranking for the ungated comparison.',
        'training': 'Only the 224 original development mothers of the inherited coupon plan, same fit class, four mother folds, hyperparameters, response clauses and paired residual draws across all arms. No observations or outcomes from the earlier 160-mother thin study are used for training.',
        'risk_gate': 'Same <=.05 development conditional risk and >=15 active mothers. Independently test each locked gate for .10 risk and .10 activity, with significance .05/4 across four possible panels. Report gated outcomes in the ungated confirmation cohort only for independently certified rules.',
        'bootstrap_seed': 1949826,
        'bootstrap_draws': 20000,
    }
    reads = [SOURCE/'REGISTERED.json', REPLICATION, *(Path(d)/'REGISTERED.json' for d in plan['coupon']['training_source_runs'])]
    registered = [read_json(p) for p in reads]
    keys = preflight(plan)
    for before in registered:
        if keys.keys() & preflight(before).keys():
            raise ValueError('A newly registered random seed overlaps an earlier study.')
        if before.get('tight', {}).get('witness_entropy') == plan['tight']['witness_entropy']:
            raise ValueError('A newly registered witness generator overlaps an earlier study.')
    dependencies = {*plan['hashes'], Path(__file__).name, *(p.as_posix() for p in reads)}
    plan['hashes'] = {Path(p).as_posix(): sha256(Path(p)) for p in sorted(dependencies)}
    run = Path('runs')/now.strftime('patch_tight_factorial_%Y%m%dT%H%M%S%fZ')
    run.mkdir()
    write_new(run/'REGISTERED.json', plan)
    write_new(run/'STREAM_CHECK.json', {
        'protocol_sha256': sha256(run/'REGISTERED.json'),
        'independent_scalar_addresses': len(keys),
        'reference_files': [p.as_posix() for p in reads],
        'witness_addresses_by_role': settings,
        'witness_entropy': plan['tight']['witness_entropy'],
    })
    print(run.as_posix(), flush=True)


def paired_inputs(group, task, plan):
    """Preserve the identical five standard readings when replacing a sensor coupon."""
    normal = witness_inputs(group, task, plan)
    known, amended = geometry(group['known'], task, {'scale': plan['coupon']['witness_scale']}, 1.)
    thin = witness_inputs({**group, 'known': known}, {**task, 'mechanism': amended}, plan)
    unchanged = np.asarray(normal['depth_mid'])[:, :-2]
    result = {}
    for prefix, source in (('standard', normal), ('thin', thin)):
        for name, kind in (('depth', 'depth_mid'), ('electric', 'log_conductance')):
            result[f'{prefix}_{name}'] = np.column_stack((unchanged, np.asarray(source[kind])[:, -2:])).tolist()
    return {'mother': group['parent'], 'raw_inputs': result}


def training(plan, task):
    x, y = [], []
    for source in map(Path, plan['coupon']['training_source_runs']):
        protocol = source/'REGISTERED.json'
        earlier = {**read_json(protocol), 'coupon': plan['coupon']}
        records = open_rows(source/'DEVELOPMENT_SCORED.json', protocol)['rows']
        group, local = parents(earlier, 'development', task)
        if [g['parent'] for g in group] != [r['mother'] for r in records]:
            raise ValueError('Training outcomes do not match the earlier observations.')
        x.extend(paired_inputs(g, task, local) for g in group)
        y.extend(r['clauses'] for r in records)
    return x, np.asarray(y)


def measurement_matrix(rows, panel, task, plan):
    shared_model_arm = [{'raw_inputs': {'depth_mid': row['raw_inputs'][panel]}} for row in rows]
    return design(shared_model_arm, task, plan, 'direct_depth')


def policies(rows, task, plan):
    x, y = training(plan, task)
    result = [{'mother': r['mother'], 'forecasts': {}} for r in rows]
    models = {}
    for panel in PANELS:
        train = measurement_matrix(x, panel, task, plan)
        new = measurement_matrix(rows, panel, task, plan)
        model, residuals = fit(train, y, 'direct_depth', plan['phase']['fit_seed'])
        risk = predictive_margins(new, model, residuals)
        banks = rank(risk)
        for i, row in enumerate(result):
            row['forecasts'][panel] = {
                'banks': {k: {key: value[i].tolist() for key, value in bank.items()} for k, bank in banks.items()},
                'action_score_columns': risk[i].tolist(),
            }
        models[panel] = {
            'training_mothers': len(train),
            'mean_absolute_oof_signed_clause_error': np.abs(residuals).mean(axis=(0, 1)).tolist(),
        }
        print(f'Frozen model {panel}', flush=True)
    return result, models


def lock(rows, margins, plan):
    return {panel: development_choice(margins, {r: arrays(rows, panel, r) for r in RANKS},
                                       plan['tight']['development_criterion']) for panel in PANELS}


def gated(rows, margins, locked, *, risk_check):
    result = {}
    for panel in PANELS:
        spec = locked.get(panel)
        if not spec or spec['primary_method'] is None:
            continue
        bank = arrays(rows, panel, spec['primary_method'])
        accepted = bank['score'] < spec['primary_cutoff']
        policy = outcomes(margins, bank['chosen'], accepted)
        result[panel] = {'outcomes': policy, 'accepted': accepted.tolist(), 'chosen': bank['chosen'].tolist()}
        if risk_check:
            selected = take_actions(margins, bank['chosen'])
            result[panel]['gate'] = certify_policy(selected, accepted, max_risk=.10, min_activity=.10, delta=.05/len(PANELS))
    return result


def paired_contrasts(success, membership, plan):
    result = {}
    effects = np.asarray(list(CONTRASTS.values())) @ success.mean(axis=-1)
    config = plan['factorial']
    rng = np.random.default_rng(config['bootstrap_seed'])
    draw = rng.integers(effects.shape[-1], size=(config['bootstrap_draws'], effects.shape[-1]))
    for name, row in zip(CONTRASTS, effects, strict=True):
        sampled = row[draw].mean(axis=1)
        result[name] = {
            'difference': float(row.mean()),
            'marginal_ci95': np.quantile(sampled, [.025, .975]).tolist(),
            'familywise_bonferroni_ci': np.quantile(sampled, [.005, .995]).tolist(),
            'mothers_improved_tied_worsened': [int((row>0).sum()), int((row==0).sum()), int((row<0).sum())],
            'nominal_exchange_effect_descriptive': [float(row[membership==v].mean()) if np.any(membership==v) else None for v in (False, True)],
        }
    return result


def ungated(rows, realized, plan):
    clauses = np.asarray([r['clauses'] for r in realized])
    margins = clauses.max(axis=-1)
    possible = margins.min(axis=-1) <= 0.
    detail, successes = {}, []
    for panel in PANELS:
        bank = arrays(rows, panel, 'posterior_violation')
        picked = take_actions(margins, bank['chosen'])
        safe = picked <= 0.
        selected_clauses = np.take_along_axis(clauses, bank['chosen'][..., None, None], axis=-2).squeeze(-2)
        limiting = np.bincount(np.argmax(selected_clauses, axis=-1)[possible & ~safe], minlength=5)
        detail[panel] = {
            'requests': picked.size,
            'successful_requests': int(safe.sum()),
            'feasible_but_missed': int((possible & ~safe).sum()),
            'limiting_clause_on_feasible_misses': dict(zip(('off', 'on', 'soft', 'support', 'weak_off'), limiting.tolist(), strict=True)),
        }
        successes.append(safe)
    membership = np.asarray([r['exchange_mechanism'] for r in realized])
    return {
        'mothers': len(rows),
        'exchange_mothers': int(membership.sum()),
        'requests_with_at_least_one_feasible_action': int(possible.sum()),
        'ungated_arms': detail,
        'paired_contrasts': paired_contrasts(np.array(successes, dtype=float), membership, plan),
    }


def run_stage(args):
    protocol = args.run/'REGISTERED.json'
    config = read_json(protocol)
    for name, expected in config['hashes'].items():
        if sha256(Path(name)) != expected:
            raise ValueError(f'Changed registered source or input: {name}.')
    role = args.role
    saved = args.run/f'{role.upper()}_FROZEN.json'
    target = saved if args.action == 'forecast' else args.run/f'{role.upper()}_SCORED.json'
    if target.exists():
        raise FileExistsError(target)
    receipt = {'protocol_sha256': sha256(protocol), 'started_utc': datetime.now(timezone.utc).isoformat()}
    locked = {}
    passed = {}
    if role != 'development':
        development = args.run/'DEVELOPMENT_SCORED.json'
        locked = open_rows(development, protocol)['results']['development_locks']
        receipt['development_sha256'] = sha256(development)
        if any(entry['primary_method'] is not None for entry in locked.values()):
            if role == 'confirmation':
                calibration = args.run/'CALIBRATION_SCORED.json'
                result = open_rows(calibration, protocol)['results']['calibration']
                receipt['calibration_sha256'] = sha256(calibration)
                passed = {p: locked[p] for p, evidence in result.items() if evidence['gate']['certified']}
        elif role == 'calibration':
            raise ValueError('No development locks. The calibration role remains unused.')
    task = read_json(Path(config['task_path']))
    groups, local = parents(config, role, task)
    if args.action == 'score':
        x = open_rows(saved, protocol)['rows']
        receipt['frozen_sha256'] = sha256(saved)
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        fn = paired_inputs if args.action == 'forecast' else reveal
        y = list(executor.map(fn, groups, repeat(task), repeat(local), chunksize=4))
    if args.action == 'forecast':
        receipt['rows'], receipt['training_diagnostics'] = policies(y, task, config)
    else:
        if [r['mother'] for r in x] != [r['mother'] for r in y]:
            raise ValueError('Forecast and target mother order disagree.')
        receipt['rows'] = y
        margins = np.asarray([r['clauses'] for r in y]).max(axis=-1)
        if role == 'calibration':
            receipt['results'] = {'calibration': gated(x, margins, locked, risk_check=True)}
        else:
            result = ungated(x, y, config)
            if role == 'development':
                result['development_locks'] = lock(x, margins, config)
            else:
                result['certified_decisions'] = gated(x, margins, passed, risk_check=False)
            receipt['results'] = result
    receipt['completed_utc'] = datetime.now(timezone.utc).isoformat()
    write_new(target, receipt)
    print(target.as_posix(), flush=True)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('action', choices=('register', 'forecast', 'score'))
    cli.add_argument('run', nargs='?', type=Path)
    cli.add_argument('role', nargs='?', choices=tuple(ROLE_INDEX))
    cli.add_argument('--workers', type=int, default=12)
    args = cli.parse_args()
    if args.action == 'register':
        register()
    elif args.run is not None and args.role is not None:
        run_stage(args)
    else:
        cli.error('Forecast and score require a run and a role.')


if __name__ == '__main__':
    main()
