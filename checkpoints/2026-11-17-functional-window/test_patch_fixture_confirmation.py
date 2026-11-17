from datetime import timedelta
import json
from pathlib import Path
import subprocess
import sys

import pytest

from test_window_confirmation import contract
from validate_patch_fixture_confirmation import CONTRACT, check_freeze, measurement_units, prepare, requirements, score


def fixture_contract(install='holding_deflection', work='working_compliance'):
    p, readings, choices, old, freeze_at = contract()
    endpoint = {'holding_load': (.2, .25), 'holding_deflection': (.002, .001),
                'working_compliance': (.01, .015), 'conformance_load': (.15, .1)}
    p['limits'] = {k: p['limits'][k] for k in ('off_siemens', 'on_siemens', 'weak_siemens')}
    p.update(endpoint_contract=CONTRACT, setup_sha256='5' * 64,
             fixture_metrics={phase: {'channel': channel, 'limit': endpoint[channel][0]}
                              for phase, channel in (('installation', install), ('working', work))})
    units = measurement_units(p)
    by_clock = {(row['environment'], row['planned_s']): row for row in old}
    rows = []
    for _label, env, channel, stamp, _side, _limit in requirements(readings[0]['requests'][0], p):
        original = by_clock[(env, stamp)]
        value = (.25 if stamp >= 10 and env == 'wet' else .03) if channel == 'conductance' else endpoint[channel][1]
        rows.append({**original, 'channel': channel, 'unit': units[channel], 'low': value, 'high': value})
    return p, readings, choices, rows, freeze_at


def decision(data, outcome):
    p, readings, choices, _, when = data
    _, _, owners = prepare(p, readings)
    return score(p, readings, choices, outcome, when, owners)['requests']['held']['request']['local']


@pytest.mark.parametrize('install, work', [('holding_deflection', 'working_compliance'), ('holding_load', 'conformance_load')])
def test_actual_units_and_lower_upper_acceptance(install, work):
    data = fixture_contract(install, work)
    result = decision(data, data[3])
    assert result['status'] == 'pass'
    assert {c['channel'] for c in result['clauses']} == {'conductance', install, work}
    assert not any(c['unit'] == 'Pa' for c in result['clauses'])


def test_mechanical_middle_failure_and_missing_and_late():
    data = fixture_contract()
    rows = data[3]
    middle = next(r for r in rows if r['channel'] == 'working_compliance' and r['planned_s'] == 15)
    middle.update(low=.005, high=.006)
    result = decision(data, rows)
    assert result['status'] == 'fail'
    assert [c['time_s'] for c in result['clauses'] if c['status'] == 'fail'] == [15]
    assert decision(data, [r for r in rows if r is not middle])['status'] == 'unresolved'
    middle.update(low=.009, high=.011)
    assert decision(data, rows)['status'] == 'unresolved'
    middle.update(low=.015, high=.015, acquired_utc='2026-01-05T00:00:30+00:00')
    assert decision(data, rows)['status'] == 'unresolved'


def test_rejects_pa_and_out_of_contract_outcomes():
    data = fixture_contract()
    rows = data[3]
    metric = next(r for r in rows if r['channel'] == 'holding_deflection')
    metric['unit'] = 'Pa'
    with pytest.raises(ValueError, match='channels, units'):
        decision(data, rows)
    p = data[0]
    p['limits']['support_pa'] = 200
    with pytest.raises(ValueError, match='no surrogate Pa'):
        prepare(p, data[1])


def test_typed_witness_keeps_early_electrical_read_isolated():
    p, readings, _, _, _ = fixture_contract()
    p['witness_units'] = {'depth': ['fraction', 'fraction'], 'depth_electric': ['fraction', 'S']}
    name = next(k for k, v in p['methods'].items() if v['view'] == 'bulk')
    p['methods'][name]['view'] = 'depth_electric'
    signals = readings[0]['witness']
    del signals['bulk']
    depth = signals['depth']
    signals['depth_electric'] = [depth[0], {**depth[1], 'unit': 'S', 'value': 0.003}]
    packet, _, _ = prepare(p, readings)
    assert packet[name][0]['witness'][1]['unit'] == 'S'
    assert packet['local'][0]['witness'][1]['unit'] == 'fraction'
    signals['depth_electric'][1]['unit'] = 'mS'
    with pytest.raises(ValueError, match='registered unit'):
        prepare(p, readings)
    signals['depth_electric'][1].update(unit='S', available_utc='2026-01-05T00:00:30+00:00')
    with pytest.raises(ValueError, match='registered time budget'):
        prepare(p, readings)


def test_unshared_specimen_freeze_and_exposure():
    data = fixture_contract()
    p, inputs, _, rows, when = data
    p['study_kind'] = 'physical_pilot'
    check_freeze(p, inputs, when)
    with pytest.raises(ValueError, match='after the cutoffs'):
        check_freeze(p, inputs, when + timedelta(seconds=2))
    target = next(r for r in rows if r['channel'] == 'working_compliance')
    target['specimen_id'] = 'another wet main'
    with pytest.raises(ValueError, match='splice'):
        decision(data, rows)
    target['specimen_id'] = next(r for r in rows if r['environment'] == 'weak')['specimen_id']
    with pytest.raises(ValueError, match='exclusive'):
        decision(data, rows)
    wet = [r for r in rows if r['environment'] == 'wet']
    for row in wet:
        row['specimen_id'] = 'wet'
        row['exposure_utc'] = when.isoformat()
    with pytest.raises(ValueError, match='registered release'):
        decision(data, rows)


def test_fixture_cli_hashes_and_witness_isolation(tmp_path):
    p, readings, choices, outcomes, _ = fixture_contract()
    names = ('protocol', 'readings', 'choices', 'frozen', 'outcomes', 'report')
    paths = {name: tmp_path / f'{name}.json' for name in names}
    for name, data in zip(('protocol', 'readings', 'choices', 'outcomes'), (p, readings, choices, outcomes), strict=True):
        paths[name].write_text(json.dumps(data), encoding='utf8')
    command = [sys.executable, '-B', str(Path(__file__).with_name('validate_patch_fixture_confirmation.py'))]
    out = tmp_path / 'inputs'
    subprocess.run([*command, 'inputs', paths['protocol'], paths['readings'], out], check=True)
    first, second = (json.loads(file.read_bytes())['inputs'][0] for file in sorted(out.glob('*.json')))
    assert first['witness_view'] == 'bulk' and first['witness'][0]['value'] == .1
    assert second['witness_view'] == 'depth' and second['witness'][0]['value'] == .05
    subprocess.run([*command, 'freeze', *(paths[name] for name in names[:4])], check=True)
    subprocess.run([*command, 'score', *(paths[name] for name in names)], check=True)
    result = json.loads(paths['report'].read_bytes())
    assert result['endpoint_contract'] == CONTRACT
    assert result['method_summary']['local']['success_fraction_equal_mother_bounds'] == [1., 1.]
    p['fixture_metrics']['working']['limit'] *= 2
    paths['protocol'].write_text(json.dumps(p), encoding='utf8')
    invalid = subprocess.run([*command, 'score', *(paths[name] for name in names)], capture_output=True)
    assert invalid.returncode and b'Frozen data' in invalid.stderr
