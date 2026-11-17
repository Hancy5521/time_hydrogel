"""Freeze and score a wet/weak patch study with whole-fixture mechanical endpoints."""

import argparse
from datetime import datetime, timezone
from pathlib import Path

from validate_hydrogel_pilot import SHA, assess, number, ordered_unique, read_packet, sha256, utc, write_new
from validate_window_confirmation import COMMON, VIEWS, check_choices, check_observation, combine, summarize


ENDPOINTS = {
    'installation': {'holding_load': ('N', 'min'), 'holding_deflection': ('m', 'max')},
    'working': {'working_compliance': ('m/N', 'min'), 'conformance_load': ('N', 'max')},
}
ELECTRICAL = {'off_siemens', 'on_siemens', 'weak_siemens'}
CONTRACT = 'patch_fixture_v1'


def witness_units(protocol):
    return protocol.get('witness_units', {name: ['fraction', 'fraction'] for name in VIEWS})


def validate_protocol(p):
    if p['study_kind'] not in ('physical_pilot', 'synthetic_fixture') or p['endpoint_contract'] != CONTRACT:
        raise ValueError('Declare a physical pilot or synthetic fixture for this endpoint contract.')
    methods = p['methods']
    views = witness_units(p)
    if (len(views) != 2 or {m['view'] for m in methods.values()} != set(views)
            or any(len(u) != 2 or u[0] != 'fraction' or u[1] not in ('fraction', 'S') for u in views.values())):
        raise ValueError('Register two witness views, each with a hydration fraction then a fraction or conductance in S.')
    digests = (p['setup_sha256'], p['calibration_sha256'], *(m['artifact_sha256'] for m in methods.values()))
    if not all(isinstance(v, str) and SHA.fullmatch(v) for v in digests):
        raise ValueError('Register fixture setup, calibration and method digests.')
    times = ordered_unique([number(t, 'Sample time') for t in p['times_s']], 'Sample times')
    early = [number(t, 'Witness time') for t in p['witness_times_s']]
    if times[0] != 0 or times != sorted(times) or len(early) != 2 or not 0 < early[0] < early[1]:
        raise ValueError('Use increasing sample times starting at zero and two positive witness times.')
    if number(p['time_tolerance_s'], 'Clock tolerance') < 0:
        raise ValueError('A clock tolerance must be nonnegative.')
    if set(p['limits']) != ELECTRICAL or set(p['fixture_metrics']) != set(ENDPOINTS):
        raise ValueError('Use electrical limits and two fixture metrics with no surrogate Pa limits.')
    fields = list(p['limits'].values())
    for phase, options in ENDPOINTS.items():
        entry = p['fixture_metrics'][phase]
        channel = entry.get('channel')
        if channel not in options or set(entry) != {'channel', 'limit'}:
            raise ValueError(f'Select a supported fixture channel and limit for {phase}.')
        fields.append(entry['limit'])
    if any(number(v, 'Functional limit') < 0 for v in fields):
        raise ValueError('Calibrated endpoint limits must be nonnegative.')
    return set(ordered_unique(p['candidate_ids'], 'Candidate IDs'))


def requirements(request, p):
    install = number(request['install_end_s'], 'Installation end')
    begin, end = (number(t, 'Work window time') for t in request['work_s'])
    times = p['times_s']
    if not 0 < install <= begin < end or any(t not in times for t in (install, begin, end)):
        raise ValueError('Include both windows and all of their endpoints.')
    q = p['limits']
    rules = [('install_off', 'wet', 'conductance', 'max', q['off_siemens'], 0, install),
             ('work_on', 'wet', 'conductance', 'min', q['on_siemens'], begin, end),
             ('weak_off', 'weak', 'conductance', 'max', q['weak_siemens'], 0, end)]
    for phase, (first, last) in zip(ENDPOINTS, ((0, install), (begin, end)), strict=True):
        item = p['fixture_metrics'][phase]
        channel = item['channel']
        _, side = ENDPOINTS[phase][channel]
        rules.append((phase, 'wet', channel, side, item['limit'], first, last))
    return [(label, env, channel, t, side, limit) for label, env, channel, side, limit, lo, hi in rules
            for t in times if lo <= t <= hi]


def check_witness(item, unit, cutoff, onset, elapsed, tolerance):
    if unit == 'fraction':
        check_observation(item, cutoff, onset, elapsed, tolerance)
        return
    if item['unit'] != unit or not item['specimen_id']:
        raise ValueError('Witness conductance must have its registered unit and source specimen.')
    if item['value'] is None:
        if not item.get('missing_reason'):
            raise ValueError('Missing observations need a recorded cause.')
        return
    if number(item['value'], 'Measured conductance') <= 0:
        raise ValueError('Use positive conductance or declare a failed assay.')
    acquired, available = utc(item['acquired_utc']), utc(item['available_utc'])
    if not acquired <= available <= cutoff or abs((acquired-onset).total_seconds()-elapsed) > tolerance:
        raise ValueError('Witness conductance was not ready within the registered time budget.')


def prepare(p, packets):
    candidates = validate_protocol(p)
    views = witness_units(p)
    mothers = ordered_unique([row['mother'] for row in packets], 'Mother IDs')
    if any(not m for m in mothers) or set(mothers).intersection(p['calibration_mothers']):
        raise ValueError('Confirmation mothers must be named and separate from calibration.')
    owners = {}
    for row in packets:
        witness, cutoff, release = map(utc, (row['witness_exposure_utc'], row['cutoff_utc'], row['release_utc']))
        if not witness < cutoff < release or set(row['common']) != set(COMMON) or set(row['witness']) != set(views):
            raise ValueError('Register complete paired early inputs and their common deadline.')
        context = row['context']
        if number(context['thickness_m'], 'Measured thickness') <= 0 or not SHA.fullmatch(context['metadata_sha256']):
            raise ValueError('Bind the measured geometry and preparation metadata.')
        witness_id = row['witness_specimen_id']
        specimen_ids = {v['specimen_id'] for v in row['common'].values()} | {witness_id}
        if any(not sid or owners.setdefault(sid, row['mother']) != row['mother'] for sid in specimen_ids):
            raise ValueError('Companion specimen identities are exclusive to one mother.')
        for item in row['common'].values():
            check_observation(item, cutoff)
        for view, items in row['witness'].items():
            for item, t, unit in zip(items, p['witness_times_s'], views[view], strict=True):
                if item['specimen_id'] != witness_id:
                    raise ValueError('Both views must refer to the same witness specimen.')
                check_witness(item, unit, cutoff, witness, t, p['time_tolerance_s'])
        ordered_unique([q['id'] for q in row['requests']], 'Request IDs')
        for q in row['requests']:
            requirements(q, p)
    isolated = {name: [{k: row[k] for k in ('mother', 'cutoff_utc', 'context', 'common', 'requests')}
                       | {'witness': row['witness'][m['view']], 'witness_view': m['view']}
                       for row in packets] for name, m in p['methods'].items()}
    return isolated, candidates, owners


def measurement_units(p):
    units = {'conductance': 'S'}
    for phase, item in p['fixture_metrics'].items():
        channel = item['channel']
        units[channel] = ENDPOINTS[phase][channel][0]
    return units


def read_panel(p, packets, choices, readings, frozen_at, companions):
    units = measurement_units(p)
    earliest = {m['mother']: utc(m['release_utc']) for m in packets}
    expected = {}
    for m in packets:
        for q in m['requests']:
            keys = {(env, channel, t) for _, env, channel, t, _, _ in requirements(q, p)}
            for name in set(choices[m['mother']][q['id']].values()) - {None}:
                expected.setdefault((m['mother'], name), set()).update(keys)
    samples, collection, index = {}, {}, {}
    now = datetime.now(timezone.utc)
    for row in readings:
        mother, candidate, env = row['mother'], row['candidate'], row['environment']
        key = (env, row['channel'], row['planned_s'])
        if key not in expected.get((mother, candidate), ()) or units[key[1]] != row['unit']:
            raise ValueError('Outcome is outside the registered fixture channels, units or sample times.')
        specimen, onset = row['specimen_id'], utc(row['exposure_utc'])
        panel_key = (mother, candidate, env)
        if not specimen or specimen in companions or samples.setdefault(specimen, panel_key) != panel_key:
            raise ValueError('A main specimen requires an exclusive candidate and environment.')
        if collection.setdefault(panel_key, (specimen, onset)) != (specimen, onset):
            raise ValueError('Do not splice main-specimen trajectories.')
        physical = p['study_kind'] == 'physical_pilot'
        if onset < earliest[mother] or (physical and onset <= frozen_at):
            raise ValueError('Main-specimen exposure precedes its registered release or freeze.')
        lo, hi = row['low'], row['high']
        if lo is None or hi is None:
            if lo is not None or hi is not None or not row.get('missing_reason'):
                raise ValueError('Missing bounds must be paired and explained.')
            elapsed = key[2]
        else:
            if not 0 <= number(lo, 'Lower measurement bound') <= number(hi, 'Upper measurement bound'):
                raise ValueError('Calibrated fixture bounds must be nonnegative and ordered.')
            acquired = utc(row['acquired_utc'])
            elapsed = (acquired - onset).total_seconds()
            if physical and (acquired > now or acquired <= frozen_at or (elapsed < 0 and key[2] > 0)):
                raise ValueError('Measurement timestamps contradict the physical release.')
        recorded = index.setdefault((mother, candidate), {})
        if key in recorded:
            raise ValueError('A specimen can have one reading per registered channel and time.')
        recorded[key] = {**row, 'actual_s': elapsed}
    return index


def score(p, packets, choices, readings, frozen_at, companions):
    measurements = read_panel(p, packets, choices, readings, frozen_at, companions)
    units = measurement_units(p)
    detail = {}
    for batch in packets:
        mother = batch['mother']
        detail[mother] = {}
        for request in batch['requests']:
            design = choices[mother][request['id']]
            rules = requirements(request, p)
            outcomes = {None: ('no_decision', [])}
            for candidate in set(design.values()) - {None}:
                status, clauses = assess(rules, measurements.get((mother, candidate), {}), p['time_tolerance_s'])
                for clause, (_, _, channel, _, side, limit) in zip(clauses, rules, strict=True):
                    clause.update(channel=channel, unit=units[channel], direction=side, limit=limit)
                outcomes[candidate] = status, clauses
            detail[mother][request['id']] = {m: {'candidate': c, 'status': outcomes[c][0], 'clauses': outcomes[c][1]}
                                               for m, c in design.items()}
    return {'endpoint_contract': CONTRACT, **summarize(p['study_kind'], p['methods'], detail)}


def check_freeze(protocol, data, when):
    if protocol['study_kind'] == 'physical_pilot' and any(
            not utc(row['cutoff_utc']) <= when < utc(row['release_utc']) for row in data):
        raise ValueError('Physical choices must be frozen after the cutoffs and before the release times.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='phase', required=True)
    layout = {'inputs': ('protocol', 'readings', 'output_dir'), 'freeze': ('protocol', 'readings', 'choices', 'frozen'),
              'score': ('protocol', 'readings', 'choices', 'frozen', 'outcomes', 'report')}
    for name, fields in layout.items():
        entry = sub.add_parser(name)
        for field in fields:
            entry.add_argument(field, type=Path)
    merger = sub.add_parser('combine')
    merger.add_argument('report', type=Path)
    merger.add_argument('blocks', nargs='+', type=Path)
    args = parser.parse_args()
    if args.phase == 'combine':
        received = [read_packet(path) for path in args.blocks]
        if any(report.get('endpoint_contract') != CONTRACT for report, _ in received):
            raise ValueError('Combine only whole-fixture outcome blocks.')
        report = combine([item[0] for item in received])
        write_new(args.report, {**report, 'endpoint_contract': CONTRACT, 'block_sha256': [h for _, h in received]})
        return
    p, p_hash = read_packet(args.protocol)
    data, data_hash = read_packet(args.readings)
    isolated, candidates, companions = prepare(p, data)
    hashes = {'protocol': p_hash, 'readings': data_hash}
    if args.phase == 'inputs':
        args.output_dir.mkdir(exist_ok=False)
        for i, (name, packets) in enumerate(isolated.items(), start=1):
            write_new(args.output_dir / f'{i:02d}.json', {'method': name, 'input_sha256': hashes, 'inputs': packets})
        return
    choices, hashes['choices'] = read_packet(args.choices)
    check_choices(p, data, choices, candidates)
    hashes.update({name: sha256(Path(__file__).with_name(name)) for name in (
        Path(__file__).name, 'validate_window_confirmation.py', 'validate_hydrogel_pilot.py')})
    if args.phase == 'freeze':
        now = datetime.now(timezone.utc)
        check_freeze(p, data, now)
        write_new(args.frozen, {'endpoint_contract': CONTRACT, 'frozen_utc': now.isoformat(), 'input_sha256': hashes})
        return
    frozen, frozen_hash = read_packet(args.frozen)
    if frozen.get('endpoint_contract') != CONTRACT or frozen['input_sha256'] != hashes:
        raise ValueError('Frozen data, choices, protocol or scoring source changed.')
    when = utc(frozen['frozen_utc'])
    check_freeze(p, data, when)
    outcome, outcome_hash = read_packet(args.outcomes)
    report = score(p, data, choices, outcome, when, companions)
    write_new(args.report, {**report, 'protocol_sha256': p_hash, 'frozen_sha256': frozen_hash, 'outcome_sha256': outcome_hash})


if __name__ == '__main__':
    main()
