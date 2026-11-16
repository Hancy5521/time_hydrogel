"""Recover measured truth from OOXML and verify frozen forecasts and solver."""

import argparse
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from zipfile import ZipFile
import json
import math
import xml.etree.ElementTree as ET

import numpy as np
from scipy.integrate import solve_ivp


NS = '{http://schemas.openxmlformats.org/spreadsheetml/2006/main}'
ROOT = Path(__file__).resolve().parent


def sha(path):
    return sha256(path.read_bytes()).hexdigest()


def worksheet(data, index):
    with ZipFile(data) as archive:
        xml = ET.fromstring(archive.read(f'xl/worksheets/sheet{index}.xml'))
    return {c.attrib['r']: float(v.text) for c in xml.iter(NS+'c') if c.attrib.get('t', 'n') == 'n'
            and (v := c.find(NS+'v')) is not None and v.text is not None}


def rows(cells):
    return np.array([[cells[f'{col}{r}'] for col in 'DEFGHIJKLMNOPQRST'] for r in range(4,29)])


def source(time_s, x_um, d0):
    return (1000.*math.exp(-x_um**2/(4*d0*time_s))/math.sqrt(4*math.pi*d0*time_s)
            if time_s > 0 else 0.)


def reference_profile(d, x, cfg):
    dz = cfg['pde_dz_um']
    n = int(round(cfg['length_um']/dz))
    d0 = cfg['source_diffusivity_um2_s']
    h = cfg['solver_step_h']*3600
    end = cfg['held_hour']*3600
    source_times = np.linspace(0,end,round(end/h)+1)
    samples = [source(t, x, d0) for t in source_times]

    def integrate(use_interpolation):
        def feed(t):
            return float(np.interp(t, source_times, samples)) if use_interpolation else source(t, x, d0)

        def derivative(t,state):
            values = np.r_[0.,state,feed(t)]
            return d/dz**2*(values[:-2]-2*values[1:-1]+values[2:])

        ivp = solve_ivp(derivative,(0,end),np.zeros(n-1),method='Radau',rtol=1e-9,atol=5e-12,t_eval=[end],max_step=360.)
        if not ivp.success:
            raise ArithmeticError(ivp.message)
        return np.r_[0.,ivp.y[:,0],feed(end)][np.rint(np.asarray(cfg['z_um'])/dz).astype(int)]

    return integrate(True), integrate(False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run',type=Path)
    run = parser.parse_args().run
    files = ['REGISTERED.json','FROZEN.json','SCORED.json']
    reg,frozen,scored = [json.loads((run/name).read_text(encoding='utf-8')) for name in files]
    book = ROOT/'Source_Data_Fig2.xlsx'
    assert reg['audit_sha256'] == sha(Path(__file__)) and reg['book_sha256'] == sha(book)
    assert frozen['registered_sha256'] == scored['registered_sha256'] == sha(run/files[0])
    assert scored['frozen_sha256'] == sha(run/files[1])
    axis = 'DEFGHIJKLMNOPQRST'
    for i in (1,2,3):
        c = worksheet(book,i)
        assert np.allclose([c[f'{k}3'] for k in axis],reg['x_mm'])
        assert np.allclose([c[f'C{r}'] for r in range(4,29)],reg['z_um'])
    past = rows(worksheet(book,2))
    held = rows(worksheet(book,3))
    per = {}
    for name, prediction in frozen['predictions'].items():
        grid = np.array(prediction)
        agreement = sum(float(np.sign(q-r) == np.sign(y-r)) for q,y,r in zip(grid.flat,held.flat,past.flat,strict=True) if y!=r)
        assert agreement/int((held!=past).sum()) == scored['scores'][name]['direction_agreement_for_nonzero_changes']
        errors = np.array([sum(abs(a-b) for a,b in zip(grid[:,j],held[:,j],strict=True))/25 for j in range(17)])
        assert np.allclose(errors,scored['scores'][name]['per_location'],rtol=0,atol=1e-14)
        per[name] = float(errors.mean())
    assert np.array_equal(np.array(frozen['predictions']['persistence_12h']),past)
    d = frozen['fit']['dynamic_diffusivity_um2_s']
    amp, off = frozen['fit']['dynamic']
    profile = frozen['predictions']['vertical_fick_pulse']
    errors, continuous = [],[]
    for j in (0,8,16):
        pulse = reference_profile(d,1000*reg['x_mm'][j],reg)
        errors.append(float(np.max(abs(amp*pulse[0]+off-np.array(profile)[:,j]))))
        continuous.append(float(np.max(abs(amp*pulse[1]+off-np.array(profile)[:,j]))))
    if max(errors)>2e-4:
        raise ArithmeticError(f'Independent solver disagreement: {errors}')
    record = {'audited_utc':datetime.now(timezone.utc).isoformat(),'hashes':{n:sha(run/n) for n in files},
              'workbook_sha256':sha(book),'code_sha256':sha(Path(__file__)),'mean_profile_MAE':per,
              'radau_linear_feed_max_error':max(errors),'radau_continuous_feed_max_error':max(continuous),
              'test_locations_mm':[reg['x_mm'][j] for j in (0,8,16)],
              'limitation':'Same article aggregate profile audit, with no per-specimen identifiers or bath inventory.'}
    with (run/'INDEPENDENT_AUDIT.json').open('x',encoding='utf-8',newline='\n') as out:
        json.dump(record,out,indent=2,ensure_ascii=False);out.write('\n')
    print(record['mean_profile_MAE'],record['radau_linear_feed_max_error'])


if __name__ == '__main__':
    main()
