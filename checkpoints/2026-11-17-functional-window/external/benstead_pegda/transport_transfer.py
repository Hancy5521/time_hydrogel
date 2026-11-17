"""Retrospective leave-acquisition-date-out transfer for public PEGDA H-cells."""

import argparse
import csv
import hashlib
import io
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from scipy.optimize import minimize_scalar


HERE = Path(__file__).resolve().parent
ARCHIVE = HERE / 'Membrane Diffusion Testing.zip'
RANGES = {('25', 'RB'), ('25', 'MB'), ('40', 'AO')}
POWERS = {'no_thickness': 0, 'finite_reservoir': 1, 'squared_length': 2}
PREFIX = 18.


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(name, data):
    with (HERE / name).open('x', encoding='utf8') as output:
        json.dump(data, output, indent=2, allow_nan=False)


def read_csv(z, filename):
    return list(csv.DictReader(io.StringIO(z.read(filename).decode('utf-8-sig'))))


def dataset():
    samples = {}
    with ZipFile(ARCHIVE) as z:
        records = read_csv(z, 'Diffusion Testing/ADT_Diffusion_Test_Records.csv')
        key_line = {(r['Date'], r['Sample'], r['PEGDA (wt%)'], r['Dye']): r for r in records}
        for filename in z.namelist():
            if not filename.endswith('_metadata.csv') or '_manual_' in filename:
                continue
            run = filename.rsplit('/', 1)[1]
            date = run[:8]
            by_label = defaultdict(list)
            for r in read_csv(z, filename):
                by_label[r['Sample']].append(r)
            for name, rows in by_label.items():
                parts = name.split('_')
                if len(parts) != 4 or parts[0] != 'PES' or not parts[1].startswith('PEG'):
                    continue
                kind = parts[1][3:], parts[2]
                if kind not in RANGES:
                    continue
                sample_label = parts[3]
                reference = key_line.get((date, sample_label, *kind))
                if reference is None:
                    continue
                sides = defaultdict(dict)
                for row in rows:
                    hcell, chamber = row['Hcell'].rsplit('_', 1)
                    clock = float(row['Time'])
                    if clock in sides[chamber]:
                        raise ValueError(f'Ambiguous sample timing in {name}, {date}.')
                    sides[chamber][clock] = float(row['C_original'])
                cells = {r['Hcell'].rsplit('_', 1)[0] for r in rows}
                thicknesses = {float(r['Membrane_cm']) for r in rows}
                if (len(cells) != 1 or set(sides) != {'C1', 'C2'} or sides['C1'].keys() != sides['C2'].keys()
                        or len(thicknesses) != 1 or thicknesses != {float(reference['Membrane (um)'])}):
                    raise ValueError(f'Ambiguous sample, chamber or thickness in {name}, {date}.')
                t = np.array(sorted(sides['C1']))
                concentrations = np.array([[sides[c][clock] for c in ('C1', 'C2')] for clock in t])
                area = np.pi * float(reference['Radius (cm)']) ** 2
                volume = float(reference['Total Vol (mL)'])
                if t[0] != 0 or len(t[(t>0)&(t<=PREFIX)]) < 2 or not np.any(t>PREFIX):
                    continue
                identity = (date, name, next(iter(cells)))
                samples[identity] = {'date':date, 'source_file':run, 'name':name, 'cell':identity[2], 'kind':kind,
                    'paper_key':reference['Key'], 't':t, 'c':concentrations, 'l':next(iter(thicknesses)), 'area_volume':area/volume}
    return list(samples.values())


def register():
    write('TRANSPORT_TRANSFER_PROTOCOL.json', {
        'registered_at_utc':datetime.now(timezone.utc).isoformat(), 'source_sha256':digest(ARCHIVE),
        'forecast_script_sha256':digest(Path(__file__)), 'published_doi':'10.1039/D5DD00523J', 'data_doi':'10.5281/zenodo.17238337',
        'scope': ('Retrospective through-membrane transport only. These are separate H-cells, with acquisition date as the '
                  'maximum confirmed grouping and supplier batch unknown. Observed thickness timing in individual public rows '
                  'is ambiguous and the article reports using post-experiment thickness for diffusivity, so this is conditional '
                  'on retrospectively available geometry. No electrical, mount, local field, or causal thickness conclusion.'),
        'inclusion': ('Automated PEG25/RB, PEG25/MB and PEG40/AO, the only exact groups with matching paired clocks across '
                      'multiple dates and no manual run. Require exact shared t=0, two nonzero source points through 18h, '
                      'one held tail and a unique paired chamber record linked to the test table. Keep P25G00 filename '
                      'when source trace and table agree; report a sensitivity excluding that filename.'),
        'fields': ('Use C_original paired concentrations in uM, volume and radius. Thickness labels conflict with the article; '
                   'use only ratios of equal-source numbers, never an inferred absolute diffusivity. Published P and D '
                   'come from full traces and are excluded from all inputs and fits.'),
        'model': ('Fit nonnegative log-rate by least squares on normalized donor-minus-receiver at t <= 18 h in each '
                  'other-date source specimen. For target forecast use only t=0 concentrations, declared thickness, volume '
                  'and aperture. Predict paired chamber concentrations using a conserved mean and exponential decay. '
                  'Project each source rate by target-to-source A/V and by (L_source/L_target)^p for p=0,1,2; use median. '
                  'Test absolute errors on donor and receiver time points t>18h, divide by target initial concentration gap.'),
        'comparisons': ('Equal-weight acquisition dates, separately report both channels, held targets whose thickness was '
                        'absent in remaining dates, and transfer only from thinner other-date cells. No test instance weights, '
                        'post hoc model selection, or deployment claims. Observe target post-18h data only in scoring.'),
    })


def source_rate(s):
    use = s['t'] <= PREFIX
    c = s['c'][use]
    d = c[:, 0] - c[:, 1]
    if d[0] <= 0:
        raise ValueError('Nonpositive initial chamber contrast.')
    error = lambda k: np.square(np.exp(-k*s['t'][use]) - d / d[0]).mean()
    return float(minimize_scalar(error, bounds=(0., 2.), method='bounded').x)


def forecast(samples, mode):
    rows = []
    train = {s['paper_key']:source_rate(s) for s in samples}
    for v in samples:
        others = [s for s in samples if s['kind'] == v['kind'] and s['date'] != v['date']]
        if mode == 'new_thickness' and any(s['l'] == v['l'] for s in others):
            continue
        if mode == 'thinner_sources':
            others = [s for s in others if s['l'] < v['l']]
        if not others:
            continue
        after = v['t'] > PREFIX
        initial = v['c'][0]
        magnitude = initial[0] - initial[1]
        by_power = {}
        for name, power in POWERS.items():
            rates = [train[s['paper_key']] * v['area_volume'] / s['area_volume'] * (s['l']/v['l'])**power for s in others]
            effect = .5*magnitude*np.exp(-float(np.median(rates))*v['t'][after])
            by_power[name] = np.column_stack((initial.mean()+effect, initial.mean()-effect)).tolist()
        rows.append({'date':v['date'], 'name':v['name'], 'paper_key':v['paper_key'], 'cell':v['cell'],
                     'source_keys': sorted(s['paper_key'] for s in others), 'scale':float(magnitude),
                     't':v['t'][after].tolist(), 'forecasts':by_power})
    return rows


def load_protocol():
    p = HERE / 'TRANSPORT_TRANSFER_PROTOCOL.json'
    a = json.loads(p.read_bytes())
    if a['source_sha256'] != digest(ARCHIVE) or a['forecast_script_sha256'] != digest(Path(__file__)):
        raise ValueError('Source changed after registration.')
    return digest(p)


def generate():
    hash_id = load_protocol()
    samples = dataset()
    output = {kind:forecast(samples,kind) for kind in ('all', 'new_thickness', 'thinner_sources')}
    output['unambiguous_filename'] = forecast([s for s in samples if 'P25G00' not in s['source_file']], 'all')
    write('FROZEN_TRANSPORT_FORECASTS.json', {'generated_at_utc':datetime.now(timezone.utc).isoformat(),
                                            'protocol_sha256':hash_id, 'roles':output})


def score():
    p = HERE / 'FROZEN_TRANSPORT_FORECASTS.json'
    frozen = json.loads(p.read_bytes())
    if frozen['protocol_sha256'] != load_protocol():
        raise ValueError('Forecast comes from a different protocol.')
    samples = {s['paper_key']:s for s in dataset()}
    output = {}
    for role, cases in frozen['roles'].items():
        daily = defaultdict(list)
        for case in cases:
            s = samples[case['paper_key']]
            assert (case['date'],case['name'],case['cell']) == (s['date'],s['name'],s['cell'])
            assert case['t'] == s['t'][s['t']>PREFIX].tolist()
            assert all(samples[k]['date'] != s['date'] for k in case['source_keys'])
            y = s['c'][s['t']>PREFIX]
            errors = [np.mean(np.abs(np.asarray(c)-y) / case['scale'], axis=0).tolist() for c in case['forecasts'].values()]
            daily[s['date']].append(errors)
        by_date = {d:np.mean(v,axis=0).tolist() for d,v in daily.items()}
        equal_date = np.mean(list(by_date.values()),axis=0)
        output[role] = {'samples':len(cases), 'acquisition_dates':len(by_date), 'channels': ['donor','receiver'],
                        'methods':list(POWERS), 'equal_date_errors':equal_date.tolist(), 'by_date_errors':by_date}
    write('TRANSPORT_TRANSFER_RESULTS.json', {'scored_at_utc':datetime.now(timezone.utc).isoformat(),
                                            'forecast_sha256':digest(p), 'results':output})
    for role,result in output.items():
        print(role,result['samples'],result['acquisition_dates'],result['equal_date_errors'])


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase', choices=('register','forecast','score'))
    a = p.parse_args()
    {'register':register,'forecast':generate,'score':score}[a.phase]()
