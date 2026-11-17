"""Freeze same-batch design decisions and score prespecified physical readings."""

import argparse
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


OBSERVATION_UNITS = {
    'bare_hydration': 'fraction', 'bare_stiffness': 'Pa', 'bare_conductance': 'S',
    'initial_thickness': 'm', 'sealed_mass_mid': 'kg/m2', 'sealed_mass_end': 'kg/m2',
}
OUTCOME_UNITS = {'front_stiffness': 'Pa', 'back_stiffness': 'Pa', 'bulk_conductance': 'S'}
SHA = re.compile(r'[0-9a-f]{64}')
BOUNDS = {'pass': (1, 1), 'fail': (0, 0), 'unresolved': (0, 1)}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def number(value, label):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f'{label} requires a finite number.')
    return value


def utc(value):
    instant = datetime.fromisoformat(value)
    if instant.tzinfo is None or instant.utcoffset() is None:
        raise ValueError('Timestamps need explicit offsets.')
    return instant.astimezone(timezone.utc)


def ordered_unique(values, label):
    if not values or len(set(values)) != len(values):
        raise ValueError(f'{label} must be nonempty and unique.')
    return values


def validate_contract(protocol, requests, choices):
    if protocol['study_kind'] not in ('physical_pilot', 'synthetic_fixture'):
        raise ValueError('Declare physical_pilot or synthetic_fixture explicitly.')
    methods = protocol['methods']
    if protocol['primary_method'] not in methods or len(methods) < 2:
        raise ValueError('Require a primary method and at least one frozen comparator.')
    digests = [protocol['calibration_sha256'], *methods.values()]
    if not all(isinstance(value, str) and SHA.fullmatch(value) for value in digests):
        raise ValueError('Record the SHA-256 of calibration and each method artifact.')
    candidates = set(ordered_unique(protocol['candidate_ids'], 'Candidate IDs'))
    ordered_unique(protocol['replicate_ids'], 'Replicate IDs')
    past = set(ordered_unique(protocol['calibration_batches'], 'Calibration batches'))
    horizon = number(protocol['horizon_s'], 'Horizon')
    jitter = number(protocol['time_tolerance_s'], 'Timing tolerance')
    if horizon <= 0 or jitter < 0:
        raise ValueError('Require a positive horizon and nonnegative clock tolerance.')
    for label in ('support_times_s', 'damp_times_s', 'wet_watch_times_s'):
        points = ordered_unique(protocol[label], label)
        if points != sorted(points) or any(not 0 <= number(x, label) <= horizon for x in points):
            raise ValueError('Monitoring stamps must be sorted within the horizon.')
    if protocol['damp_times_s'][-1] != horizon:
        raise ValueError('Include the horizon in the weak-wet monitoring schedule.')
    if any(number(protocol['thresholds'][k], k) < 0 for k in (
        'dry_min_pa', 'soft_pa', 'window_g_min_s', 'terminal_e_max_pa', 'terminal_g_min_s', 'back_min_pa', 'damp_min_pa',
    )):
        raise ValueError('Thresholds must use nonnegative physical units.')

    case_ids = ordered_unique([row['case_id'] for row in requests], 'Case IDs')
    if set(choices) != set(case_ids):
        raise ValueError('Every registered request needs one entry for each method.')
    now = datetime.now(timezone.utc)
    first_exposure = {}
    for row in requests:
        if row['batch_id'] in past:
            raise ValueError('Held mother batches must be absent from calibration batches.')
        low, high = [number(x, 'Request time') for x in row['window_s']]
        if not 0 < low <= high <= horizon:
            raise ValueError('Demand windows must lie within the declared horizon.')
        cutoff = utc(row['cutoff_utc'])
        earliest = utc(row['earliest_exposure_utc'])
        if earliest < cutoff or first_exposure.setdefault(row['batch_id'], earliest) != earliest:
            raise ValueError('Every candidate exposure follows its cutoff and uses the common batch release bound.')
        if cutoff > now:
            raise ValueError('Cannot report a future decision cutoff as already frozen.')
        if set(row['snapshots']) != set(OBSERVATION_UNITS):
            raise ValueError('Every request needs the same six named snapshot fields.')
        for name, item in row['snapshots'].items():
            if item['unit'] != OBSERVATION_UNITS[name]:
                raise ValueError(f'Incorrect physical unit for {name}.')
            if item['available_utc'] is not None:
                available = utc(item['available_utc'])
            else:
                available = None
            if item['value'] is not None:
                value = number(item['value'], name)
                if available is None or available > cutoff or (name == 'initial_thickness' and value <= 0):
                    raise ValueError('Only on-time calibrated observations may be given to methods.')
            elif not item.get('missing_reason'):
                raise ValueError('Record a reason for a missing or late snapshot.')
        design = choices[row['case_id']]
        if set(design) != set(methods) or any(value is not None and value not in candidates for value in design.values()):
            raise ValueError('Every method must choose a registered candidate or record no decision.')


def requirements(protocol, window):
    """Each tuple is (clause, environment, channel, second, direction, limit)."""
    low, high = window
    t = protocol['thresholds']
    end = protocol['horizon_s']
    direct = [
        ('dry_front', 'wet', 'front_stiffness', 0, 'min', t['dry_min_pa']),
        ('dry_back', 'wet', 'back_stiffness', 0, 'min', t['dry_min_pa']),
        ('wet_at_window_start', 'wet', 'front_stiffness', low, 'min', t['soft_pa']),
        ('wet_soft_by_deadline', 'wet', 'front_stiffness', high, 'max', t['soft_pa']),
        ('electrical_by_deadline', 'wet', 'bulk_conductance', high, 'min', t['window_g_min_s']),
        ('wet_terminal_softness', 'wet', 'front_stiffness', end, 'max', t['terminal_e_max_pa']),
        ('wet_terminal_conductance', 'wet', 'bulk_conductance', end, 'min', t['terminal_g_min_s']),
    ]
    return [*direct,
            *(('no_early_softening', 'wet', 'front_stiffness', x, 'min', t['soft_pa'])
              for x in protocol['wet_watch_times_s'] if x < low),
            *(('back_support', 'wet', 'back_stiffness', x, 'min', t['back_min_pa'])
              for x in protocol['support_times_s']),
            *(('weak_wet_support', 'damp', 'front_stiffness', x, 'min', t['damp_min_pa'])
              for x in protocol['damp_times_s'])]


def index_measurements(protocol, requests, choices, outcomes, frozen_utc=None):
    eligible = {(r['batch_id'], c) for r in requests for c in choices[r['case_id']].values() if c is not None}
    measurements = defaultdict(dict)
    owners = {}
    exposures = {}
    starts = {row['batch_id']: utc(row['earliest_exposure_utc']) for row in requests}
    now = datetime.now(timezone.utc)
    for row in outcomes:
        batch, candidate = row['batch_id'], row['candidate_id']
        if (batch, candidate) not in eligible or row['replicate_id'] not in protocol['replicate_ids']:
            raise ValueError('Reveal only the preselected candidate-replicate outcomes.')
        phase, channel = row['environment'], row['channel']
        if phase not in ('wet', 'damp') or row['unit'] != OUTCOME_UNITS[channel]:
            raise ValueError('Outcome environment or unit mismatch.')
        key = (batch, candidate, row['replicate_id'], phase)
        specimen = row['specimen_id']
        if not specimen or owners.setdefault(specimen, key) != key:
            raise ValueError('A specimen belongs to one batch, candidate, replicate panel and environment.')
        planned = number(row['planned_s'], 'Planned stamp')
        if not 0 <= planned <= protocol['horizon_s']:
            raise ValueError('Planned stamps must lie inside the observation horizon.')
        if row['low'] is None or row['high'] is None:
            if row['low'] is not None or row['high'] is not None or not row.get('missing_reason'):
                raise ValueError('Both bounds and a missing reason determine availability.')
        else:
            actual = number(row['actual_s'], 'Actual stamp')
            if actual < 0:
                raise ValueError('Actual outcomes need a nonnegative stamp.')
            if number(row['low'], 'Outcome lower bound') > number(row['high'], 'Outcome upper bound'):
                raise ValueError('Outcome intervals must be ordered.')
            if protocol['study_kind'] == 'physical_pilot':
                exposure, acquired = utc(row['exposure_utc']), utc(row['actual_utc'])
                elapsed = (acquired - exposure).total_seconds()
                clock_residual = abs(elapsed - actual)
                if (frozen_utc is None or exposure < starts[batch] or not frozen_utc < exposure <= now
                        or not frozen_utc < acquired <= now or exposures.setdefault(specimen, exposure) != exposure
                        or clock_residual > protocol['time_tolerance_s'] or (planned > 0 and elapsed < 0)):
                    raise ValueError('Physical outcome and specimen UTC must agree with the frozen release and relative clock.')
                row = {**row, 'scoring_elapsed_s': elapsed}
        step = (phase, channel, planned)
        block = measurements[key[:3]]
        if step in block:
            raise ValueError('Duplicate planned outcome stamp. Supply the calibrated uncertainty interval once.')
        block[step] = row
    return measurements


def assess(required, readings, clock_tolerance):
    clauses = []
    for clause, environment, channel, time, direction, threshold in required:
        row = readings.get((environment, channel, time))
        status = 'unresolved'
        reason = None
        if row is None:
            reason = 'stamp_not_recorded'
        elif row['low'] is None:
            reason = row['missing_reason']
        elif abs(row.get('scoring_elapsed_s', row['actual_s']) - time) > clock_tolerance:
            reason = 'outside_time_tolerance'
        else:
            lo, hi = row['low'], row['high']
            if direction == 'min':
                status = 'pass' if lo >= threshold else 'fail' if hi < threshold else 'unresolved'
            else:
                status = 'pass' if hi <= threshold else 'fail' if lo > threshold else 'unresolved'
            if status == 'unresolved':
                reason = 'measurement_interval_crosses_threshold'
        clauses.append({'clause': clause, 'environment': environment, 'time_s': time, 'status': status, 'reason': reason})
    states = {row['status'] for row in clauses}
    result = 'fail' if 'fail' in states else 'unresolved' if 'unresolved' in states else 'pass'
    return result, clauses


def score(protocol, requests, choices, outcomes, frozen_utc=None):
    measurements = index_measurements(protocol, requests, choices, outcomes, frozen_utc)
    methods = tuple(protocol['methods'])
    primary = protocol['primary_method']
    comparators = [name for name in methods if name != primary]
    batch_comparisons = defaultdict(lambda: defaultdict(list))
    batch_levels = defaultdict(lambda: defaultdict(list))
    status_count = {method: Counter() for method in methods}
    cases = []
    for row in requests:
        case, batch = row['case_id'], row['batch_id']
        required = requirements(protocol, row['window_s'])
        actions = choices[case]
        by_replicate = []
        for replicate in protocol['replicate_ids']:
            measured = {}
            for action in set(actions.values()) - {None}:
                status, clauses = assess(required, measurements.get((batch, action, replicate), {}), protocol['time_tolerance_s'])
                measured[action] = {'candidate_status': status, 'clauses': clauses}
            results = {name: measured[action]['candidate_status'] if action is not None else 'no_decision'
                       for name, action in actions.items()}
            for name, status in results.items():
                status_count[name][status] += 1
                batch_levels[batch][name].append(BOUNDS.get(status, (0, 0)))
            p = BOUNDS.get(results[primary], (0, 0))
            for other in comparators:
                o = BOUNDS.get(results[other], (0, 0))
                pair = (0, 0) if actions[primary] == actions[other] else (p[0] - o[1], p[1] - o[0])
                batch_comparisons[batch][other].append(pair)
            by_replicate.append({'replicate_id': replicate, 'status_by_method': results, 'candidate_clauses': measured})
        cases.append({'case_id': case, 'batch_id': batch, 'replicates': by_replicate})

    def pair_mean(pairs):
        return [math.fsum(values) / len(pairs) for values in zip(*pairs, strict=True)]

    by_batch = {batch: {name: pair_mean(pairs) for name, pairs in group.items()}
                for batch, group in batch_comparisons.items()}
    macro = {name: pair_mean([group[name] for group in by_batch.values()]) for name in comparators}
    batches = {batch: {name: pair_mean(points) for name, points in group.items()}
               for batch, group in batch_levels.items()}
    overall = {name: pair_mean([group[name] for group in batches.values()]) for name in methods}
    return {'study_kind': protocol['study_kind'], 'claim_scope': 'prespecified sampled times with assigned clock tolerance',
            'mother_batches': len(by_batch), 'total_requests': len(requests), 'method_status_counts': status_count,
            'success_bounds_equal_batch': overall,
            'primary_minus_comparator_bounds_by_batch': by_batch,
            'primary_minus_comparator_equal_batch_bounds': macro, 'cases': cases}


def read_packet(path):
    encoded = path.read_bytes()
    return json.loads(encoded), hashlib.sha256(encoded).hexdigest()


def write_new(path, payload):
    with path.open('x', encoding='utf-8', newline='\n') as file:
        json.dump(payload, file, indent=2, ensure_ascii=False, allow_nan=False, sort_keys=True)
        file.write('\n')


def main():
    args = argparse.ArgumentParser(description=__doc__)
    mode = args.add_subparsers(dest='mode', required=True)
    for name, fields in {
        'freeze': ('protocol', 'requests', 'choices', 'frozen'),
        'score': ('protocol', 'requests', 'choices', 'frozen', 'outcomes', 'report'),
    }.items():
        branch = mode.add_parser(name)
        for field in fields:
            branch.add_argument(field, type=Path)
    job = args.parse_args()
    packets = {name: read_packet(getattr(job, name)) for name in ('protocol', 'requests', 'choices')}
    signatures = {name: value[1] for name, value in packets.items()}
    signatures['source'] = sha256(Path(__file__))
    protocol, requests, choices = (item[0] for item in packets.values())
    validate_contract(protocol, requests, choices)
    if job.mode == 'freeze':
        now = datetime.now(timezone.utc)
        if protocol['study_kind'] == 'physical_pilot' and any(utc(row['earliest_exposure_utc']) <= now for row in requests):
            raise ValueError('Physical choices must be frozen before starting candidate outcome exposures.')
        write_new(job.frozen, {'input_sha256': signatures, 'frozen_utc': now.isoformat(), 'study_kind': protocol['study_kind']})
    else:
        frozen, freeze_hash = read_packet(job.frozen)
        frozen_at = utc(frozen['frozen_utc'])
        if frozen['input_sha256'] != signatures or frozen['study_kind'] != protocol['study_kind']:
            raise ValueError('Frozen choices, input observations, protocol or scoring source changed.')
        if protocol['study_kind'] == 'physical_pilot' and any(
                not utc(row['cutoff_utc']) <= frozen_at < utc(row['earliest_exposure_utc']) for row in requests):
            raise ValueError('Physical freeze must lie between the latest input cutoff and outcome exposure.')
        actual, outcomes_hash = read_packet(job.outcomes)
        report = score(protocol, requests, choices, actual, frozen_at)
        write_new(job.report, {'frozen_sha256': freeze_hash, 'outcome_sha256': outcomes_hash, **report})


if __name__ == '__main__':
    main()
