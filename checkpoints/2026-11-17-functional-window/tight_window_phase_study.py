"""Evaluate paired phase corrections using independently held simulator groups."""

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
from tight_window_phase_model import METHODS, design, fit, predictive_margins, rank, witness_inputs
from tight_window_study import RANKS, ROLE_INDEX, arrays, evaluate, open_rows
from tight_window_witness import forecast, reveal
from window_event_policy import development_choice, outcomes


def register(source):
    files = [source / name for name in ('REGISTERED.json', 'DEVELOPMENT_FROZEN.json', 'DEVELOPMENT_SCORED.json')]
    plan = read_json(files[0])
    now = datetime.now(timezone.utc)
    plan.update(registered_utc=now.isoformat(), registered_shanghai=now.astimezone(timezone(timedelta(hours=8))).isoformat())
    plan['scientific_question'] = ('Does phase-aware forecast correction enable useful tight-window '
                                   'decisions, and does a short conductivity witness add value under the same supervision?')
    panel = dict(development=112, calibration=320, confirmation=160)
    for k, (role, n) in enumerate(panel.items()):
        plan['event_roles'][role].update(groups=n, nominal_seed=804510011 + 43001*k, shift_seed=804510699 + 43001*k,
            mechanism_seed=804511131 + 43001*k, spatial_noise_salt=4300041 + 18253*k, hydraulic_salt=7392990 + 92251*k)
    plan['tight']['witness_entropy'] += 372881
    plan['phase'] = {'training_run': source.as_posix(), 'fit_seed': 192813, 'crossfit_mothers': 4,
                     'constraints': ('Use only the original 112 developmental mothers as supervised training. '
                         'All new tuning and gates use fresh roles. Two baseline methods use depth, two new methods use electrical. '
                         'Choose a single supervised rule per sensor via the same development criterion. Compare both at the '
                         'same risk/activity targets and at least 2.5 point prespecified paired utility gain. No rule when none locks.'),
                     'models': ('Direct same-input ExtraTrees predicting a vector of the five signed clauses, and '
                         'standardized ridge using the same measurements plus whole and clause-specific physical summaries. '
                         'Out-of-fold residual vectors are kept joined across clauses and actions. No true mechanism input.'),
                     'scope': 'Paired phase-forecast correction, never a new material measurement or full state calibration.'}
    dependencies = {*plan['hashes'], Path(__file__).name, 'tight_window_phase_model.py', *(p.as_posix() for p in files)}
    plan['hashes'] = {name: sha256(Path(name)) for name in sorted(dependencies)}
    keys = preflight(plan)
    past = preflight(read_json(files[0]))
    if keys.keys() & past.keys():
        raise ValueError('Fresh generator addresses collide with the training cohort.')
    run = Path('runs') / now.strftime('patch_tight_phase_%Y%m%dT%H%M%S%fZ')
    run.mkdir()
    write_new(run / 'REGISTERED.json', plan)
    write_new(run / 'STREAM_CHECK.json', {'protocol_sha256': sha256(run/'REGISTERED.json'),
             'new_distinct_scalar_addresses': len(keys), 'disjoint_from_scalar_training_addresses': len(past),
             'distinct_new_witness_tuples': sum(panel.values()), 'checked_utc': datetime.now(timezone.utc).isoformat()})
    print(run.as_posix(), flush=True)


def generate(group, task, plan):
    frame = forecast(group, task, plan)
    frame['raw_inputs'] = witness_inputs(group, task, plan)
    frame['forecasts'] = {k:frame['forecasts'][k] for k in frame['raw_inputs']}
    return frame


def parents(plan, role, task):
    local = {**current_plan(plan, role), 'tight_role': ROLE_INDEX[role]}
    return list(mixtures(local, role, task)), local


def bundle_training(plan, task):
    source = Path(plan['phase']['training_run'])
    training = read_json(source/'REGISTERED.json')
    x = open_rows(source/'DEVELOPMENT_FROZEN.json', source/'REGISTERED.json')['rows']
    y = open_rows(source/'DEVELOPMENT_SCORED.json', source/'REGISTERED.json')['rows']
    groups, settings = parents(training, 'development', task)
    if [g['parent'] for g in groups] != [r['mother'] for r in x] or [r['mother'] for r in x] != [r['mother'] for r in y]:
        raise ValueError('Original training identities changed during raw measurement reconstruction.')
    for row, group in zip(x, groups, strict=True):
        row['raw_inputs'] = witness_inputs(group, task, settings)
    return x, np.asarray([r['clauses'] for r in y])


def predictions(train, y, new, task, plan):
    data = [{'mother':row['mother'], 'forecasts':{}} for row in new]
    statistics = {}
    for method in METHODS:
        x = design(train, task, plan, method)
        target = design(new, task, plan, method)
        model, residuals = fit(x, y, method, plan['phase']['fit_seed'])
        response = predictive_margins(target, model, residuals)
        banks = rank(response)
        for i, row in enumerate(data):
            row['forecasts'][method] = {'banks':{r:{k:a[i].tolist() for k,a in f.items()} for r, f in banks.items()},
                                        'predicted_q90': response[i, ..., 0].tolist(),
                                        'predicted_p_failure': response[i, ..., 2].tolist()}
        statistics[method] = {'n_training_mothers':len(x), 'residual_draws':len(residuals),
                              'crossfit_mae_by_signed_clause': np.abs(residuals).mean(axis=(0,1)).tolist()}
    return data, statistics


def develop(plan, rows, y):
    results = {}
    for method in METHODS:
        slates = {r: arrays(rows, method, r) for r in RANKS}
        item = development_choice(y, slates, plan['tight']['development_criterion'])
        item['ungated'] = {r:outcomes(y,p['chosen'],np.ones(p['score'].shape,dtype=bool)) for r,p in slates.items()}
        results[method] = item
    def select(suffix):
        eligible = []
        for name, entry in results.items():
            if name.endswith(suffix) and entry['primary_method'] is not None:
                q = entry['variants'][entry['primary_method']]['selected']
                eligible.append((q['successful_requests'], -q['unsafe_mothers'], name))
        return max(eligible)[-1] if eligible else None
    return {'rules': results, 'baseline': select('depth'), 'alternative': select('electric')}


def run_phase(args):
    protocol = args.run/'REGISTERED.json'
    plan = read_json(protocol)
    for name,h in plan['hashes'].items():
        if sha256(Path(name)) != h:
            raise ValueError(f'Changed input or source {name}.')
    stage = args.role.upper()
    target = args.run / f'{stage}_{"FROZEN" if args.action == "forecast" else "SCORED"}.json'
    if target.exists():
        raise FileExistsError(target)
    receipt = {'protocol_sha256':sha256(protocol), 'started_utc':datetime.now(timezone.utc).isoformat()}
    if args.role != 'development':
        lock = args.run/'DEVELOPMENT_SCORED.json'
        dev = open_rows(lock, protocol)['results']
        if not any(dev[k] for k in ('baseline','alternative')):
            raise ValueError('No development rule to calibrate.')
        receipt['development_sha256'] = sha256(lock)
    if args.role == 'confirmation':
        cal = args.run/'CALIBRATION_SCORED.json'
        open_rows(cal, protocol)
        receipt['calibration_sha256'] = sha256(cal)
    task = read_json(Path(plan['task_path']))
    groups, settings = parents(plan, args.role, task)
    fn = generate if args.action == 'forecast' else reveal
    if args.action == 'score':
        saved = args.run/f'{stage}_FROZEN.json'
        frozen = open_rows(saved,protocol)['rows']
        receipt['frozen_sha256'] = sha256(saved)
    result = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for item in pool.map(fn, groups, repeat(task), repeat(settings)):
            result.append(item)
            if len(result)%32 == 0:
                print(args.action, args.role, len(result), flush=True)
    if args.action == 'forecast':
        train, labels = bundle_training(plan, task)
        receipt['rows'],receipt['training_diagnostics'] = predictions(train, labels, result, task, plan)
    else:
        if [r['mother'] for r in result] != [r['mother'] for r in frozen]:
            raise ValueError('Forecast and revealed outcomes are misaligned.')
        margins = np.asarray([r['clauses'] for r in result]).max(axis=-1)
        receipt['rows'] = result
        receipt['results'] = develop(plan,frozen,margins) if args.role == 'development' else evaluate(dev,frozen,margins,plan,args.role)
    receipt['completed_utc'] = datetime.now(timezone.utc).isoformat()
    write_new(target,receipt)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=('register','forecast','score'))
    p.add_argument('run',type=Path)
    p.add_argument('role',nargs='?',choices=tuple(ROLE_INDEX))
    p.add_argument('--workers',type=int,default=15)
    a = p.parse_args()
    if a.action == 'register':
        register(a.run)
    elif a.role is None:
        p.error('Enter a role for forecasts and scores.')
    else:
        run_phase(a)


if __name__ == '__main__':
    main()
