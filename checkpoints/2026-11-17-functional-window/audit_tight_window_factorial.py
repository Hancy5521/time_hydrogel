"""Reconstruct the frozen factorial results without importing the fitting pipeline."""

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


ARMS = ('standard_depth', 'standard_electric', 'thin_depth', 'thin_electric')
STEPS = ('development', 'calibration', 'confirmation')
RANKS = ('upper_margin', 'posterior_violation')


def digest(file):
    return hashlib.sha256(file.read_bytes()).hexdigest()


def read(file):
    return json.loads(file.read_bytes())


def validate(condition, message):
    if not condition:
        raise ValueError(message)


def date(t):
    return datetime.fromisoformat(t)


def check_phases(directory, protocol):
    realized = {}
    for role in STEPS:
        forecast = directory / f'{role.upper()}_FROZEN.json'
        score = directory / f'{role.upper()}_SCORED.json'
        if role == 'calibration' and not forecast.exists() and not score.exists():
            continue
        f, s = read(forecast), read(score)
        validate(f.get('protocol_sha256') == s.get('protocol_sha256') == digest(protocol), f'{role} protocol changed.')
        validate(s['frozen_sha256'] == digest(forecast), f'{role} labels target a different forecast.')
        validate(date(f['completed_utc']) < date(s['started_utc']), f'{role} labels preceded freezing.')
        validate([r['mother'] for r in f['rows']] == [r['mother'] for r in s['rows']], f'{role} group order differs.')
        validate(len({r['mother'] for r in f['rows']}) == len(f['rows']), f'{role} has repeated mother identifiers.')
        if role != 'development':
            development = directory/'DEVELOPMENT_SCORED.json'
            validate(f['development_sha256'] == s['development_sha256'] == digest(development), f'{role} lock changed.')
        if 'calibration_sha256' in s:
            reference = directory/'CALIBRATION_SCORED.json'
            validate(f['calibration_sha256'] == s['calibration_sha256'] == digest(reference), 'Calibration changed.')
        realized[role] = (f, s)
    return realized


def action_replay(f, s):
    success = {}
    plans = {}
    for arm in ARMS:
        cases = []
        by_rank = {rank: [] for rank in RANKS}
        for measured, predicted in zip(s['rows'], f['rows'], strict=True):
            repaired = []
            for clauses, estimates, i in zip(measured['clauses'], predicted['forecasts'][arm]['action_score_columns'], (0, 1), strict=True):
                for rank, column in zip(RANKS, (0, 1), strict=True):
                    action = min(range(len(estimates)), key=lambda k: (estimates[k][column], estimates[k][2], estimates[k][0], k))
                    stored = predicted['forecasts'][arm]['banks'][rank]
                    validate(stored['choice'][i] == action and stored['score'][i] == estimates[action][column], 'Action ranking mismatch.')
                    passes = max(clauses[action]) <= 0
                    by_rank[rank].append((estimates[action][column], passes))
                    if rank == 'posterior_violation':
                        repaired.append(passes)
            cases.append(repaired)
        success[arm] = np.asarray(cases, dtype=int)
        plans[arm] = {rank: (np.asarray([v[0] for v in values]).reshape(-1, 2), np.asarray([v[1] for v in values]).reshape(-1, 2)) for rank, values in by_rank.items()}
    return success, plans


def gates(directory, phase, plan, dev, fit):
    criterion = plan['tight']['development_criterion']
    locked = dev['results']['development_locks']
    for arm in ARMS:
        selected = {}
        for rank, (values, passes) in fit[arm].items():
            contenders = []
            enough = 0
            for threshold in np.unique(values):
                accept = values <= threshold
                used = int(accept.any(axis=1).sum())
                if used >= criterion['min_development_mothers']:
                    enough += 1
                    bad = int((accept & ~passes).any(axis=1).sum())
                    if bad/used <= criterion['design_risk']:
                        contenders.append((int((accept & passes).sum()), -bad, -float(np.nextafter(threshold, np.inf))))
            registration = locked[arm]['variants'][rank]
            validate(registration['thresholds_with_required_groups'] == enough, 'Development threshold coverage mismatch.')
            validate(registration['thresholds_below_design_risk'] == len(contenders), 'Development eligibility mismatch.')
            if contenders:
                best = max(contenders)
                chosen = registration['selected']
                validate((chosen['successful_requests'], -chosen['unsafe_mothers'], -chosen['cutoff']) == best, 'Development winner mismatch.')
                selected[rank] = best
            else:
                validate(registration['selected'] is None, 'Ineligible development gate locked.')
        best_rank = max(selected, key=lambda rank: (*selected[rank][:2], rank)) if selected else None
        validate(locked[arm]['primary_method'] == best_rank, 'Primary rank differs from registered tie-break.')

    any_locked = any(x['primary_method'] is not None for x in locked.values())
    validate(any_locked == ('calibration' in phase), 'Independent calibration role used inconsistently.')
    if any_locked:
        forecast, realized = phase['calibration']
        _, plans = action_replay(forecast, realized)
        for arm, rule in locked.items():
            if rule['primary_method'] is None:
                continue
            values, success = plans[arm][rule['primary_method']]
            accept = values < rule['primary_cutoff']
            total = len(accept)
            active = int(accept.any(axis=1).sum())
            wrong = int((accept & ~success).any(axis=1).sum())
            risk_p = sum(math.comb(active, k)*.1**k*.9**(active-k) for k in range(wrong+1))
            use_p = sum(math.comb(total, k)*.1**k*.9**(total-k) for k in range(active, total+1))
            claimed = realized['results']['calibration'][arm]['gate']
            validate(claimed['certified'] == (max(risk_p, use_p) <= .05/len(ARMS)), 'Independent risk certification mismatch.')
    return [arm for arm, rule in locked.items() if rule['primary_method'] is not None]


def effects(orders, f, s, plan):
    received = s['results']
    all_scores = np.asarray([orders[p].mean(axis=1) for p in ARMS])
    for p in ARMS:
        validate(received['ungated_arms'][p]['successful_requests'] == int(orders[p].sum()), 'Wrong number of successful requests.')
    rng = np.random.default_rng(plan['factorial']['bootstrap_seed'])
    draw = rng.integers(len(f['rows']), size=(plan['factorial']['bootstrap_draws'], len(f['rows'])))
    for name, vector in plan['factorial']['contrasts'].items():
        per_mother = sum(k*v for k, v in zip(vector, all_scores, strict=True))
        boot = per_mother[draw].mean(axis=1)
        computed = [float(per_mother.mean()), *np.quantile(boot, [.025,.975]), *np.quantile(boot,[.005,.995])]
        claimed = received['paired_contrasts'][name]
        expected = [claimed['difference'], *claimed['marginal_ci95'], *claimed['familywise_bonferroni_ci']]
        validate(np.array_equal(computed, expected), f'Contrast mismatch: {name}.')


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('directory', type=Path)
    args = cli.parse_args()
    protocol = args.directory/'REGISTERED.json'
    plan = read(protocol)
    for relative, expected in plan['hashes'].items():
        validate(digest(Path(relative)) == expected, f'Registered input changed: {relative}.')
    stages = check_phases(args.directory, protocol)
    f, s = stages['development']
    d, fit = action_replay(f, s)
    effects(d, f, s, plan)
    locked = gates(args.directory, stages, plan, s, fit)
    f, s = stages['confirmation']
    d, _ = action_replay(f, s)
    effects(d, f, s, plan)
    result = {'ran_utc': datetime.now(timezone.utc).isoformat(), 'protocol_sha256': digest(protocol),
              'source_sha256': digest(Path(__file__)), 'audited_roles': list(stages),
              'passed': True, 'independent_checks': ['ancestry', 'phase timing', 'all action rankings', 'all arm counts', 'all paired contrast intervals', 'gates'],
              'development_locked_panels': locked, 'confirmation_sha256': digest(args.directory/'CONFIRMATION_SCORED.json'),
              'scope': 'Recomputed scores and ranking from frozen arrays, without replaying model fitting or PDE generation.'}
    output = args.directory/'FACTORIAL_AUDIT.json'
    with output.open('x', encoding='utf8') as handle:
        json.dump(result, handle, ensure_ascii=False, allow_nan=False, indent=2)
    print(output.as_posix())


if __name__ == '__main__':
    main()
