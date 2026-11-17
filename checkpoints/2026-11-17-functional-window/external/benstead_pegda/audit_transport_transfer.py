"""Recompute saved validation scores from the raw Zenodo CSVs without the predictor."""

import csv
import hashlib
import io
import json
from collections import defaultdict
from math import isclose
from pathlib import Path
from statistics import mean
from zipfile import ZipFile


HERE = Path(__file__).resolve().parent


def main():
    source = HERE/'Membrane Diffusion Testing.zip'
    fpath = HERE/'FROZEN_TRANSPORT_FORECASTS.json'
    forward = json.loads(fpath.read_bytes())
    actual = json.loads((HERE/'TRANSPORT_TRANSFER_RESULTS.json').read_bytes())
    assert hashlib.sha256(fpath.read_bytes()).hexdigest() == actual['forecast_sha256']
    assert hashlib.md5(source.read_bytes()).hexdigest() == '92c9c740602a1f274a2e4550ca98409d'
    values = {}
    wanted = {(r['date'], r['name'], r['cell']) for cases in forward['roles'].values() for r in cases}
    with ZipFile(source) as archive:
        record = archive.read('Diffusion Testing/ADT_Diffusion_Test_Records.csv').decode('utf-8-sig')
        samples = {r['Key']:r for r in csv.DictReader(io.StringIO(record))}
        for n in archive.namelist():
            if not n.endswith('_metadata.csv') or '_manual_' in n:
                continue
            day = n.rsplit('/',1)[1][:8]
            rows = csv.DictReader(io.StringIO(archive.read(n).decode('utf-8-sig')))
            for r in rows:
                if (day, r['Sample'], r['Hcell'].rsplit('_', 1)[0]) not in wanted:
                    continue
                key = day, r['Sample'], r['Hcell'], float(r['Time'])
                assert key not in values
                values[key] = float(r['C_original'])
    total = 0
    for role, forecasts in forward['roles'].items():
        output = actual['results'][role]
        assert output['samples'] == len(forecasts)
        errors = defaultdict(list)
        for r in forecasts:
            date, label, cell = r['date'], r['name'], r['cell']
            assert label.endswith(samples[r['paper_key']]['Sample']) and date == samples[r['paper_key']]['Date']
            assert all(date != samples[k]['Date'] for k in r['source_keys'])
            for chamber in ('C1','C2'):
                assert (date,label,f'{cell}_{chamber}',0.) in values
                assert all(t > 18 for t in r['t'])
            assert isclose(r['scale'], values[(date,label,f'{cell}_C1',0.)] - values[(date,label,f'{cell}_C2',0.)])
            for method, predicted in r['forecasts'].items():
                for i, chamber in enumerate(('C1','C2')):
                    e = mean(abs(p[i] - values[(date,label,f'{cell}_{chamber}',t)])/r['scale']
                             for p,t in zip(predicted,r['t'],strict=True))
                    errors[(date,method,i)].append(e)
        for j, method in enumerate(output['methods']):
            for i in (0,1):
                by_day = {date:mean(e) for (date,m,c), e in errors.items() if (m,c) == (method,i)}
                assert len(by_day) == output['acquisition_dates']
                for date, exact in by_day.items():
                    assert isclose(exact,output['by_date_errors'][date][j][i],rel_tol=1e-12)
                assert isclose(mean(by_day.values()),output['equal_date_errors'][j][i],rel_tol=1e-12)
                total += len(by_day)+1
    print(f'Replayed {total} source-linked scores from the raw dual-chamber CSV files.')


if __name__ == '__main__':
    main()
