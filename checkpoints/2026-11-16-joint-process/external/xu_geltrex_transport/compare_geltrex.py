"""Fit two early measured Geltrex fields; evaluate a frozen 24-hour forecast."""

import argparse
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
import json

import numpy as np
from openpyxl import load_workbook
from scipy.fft import dst, idst
from scipy.optimize import nnls


ROOT = Path(__file__).resolve().parent
BOOK = ROOT / 'Source_Data_Fig2.xlsx'
RECEIPT = ROOT / 'SOURCE_RECEIPT.json'
SHEETS = {7: 'Fig2a_7-hours', 12: 'Fig2b_12-hours', 24: 'Fig2c_24-hours'}


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def write_new(path, value):
    with path.open('x', encoding='utf-8', newline='\n') as out:
        json.dump(value, out, ensure_ascii=False, allow_nan=False, indent=2)
        out.write('\n')


def source_amplitude(time_h, x_mm, d0):
    """The analytical horizontal concentration of a unit Gaussian pulse."""
    sec = np.asarray(time_h)[:, None] * 3600.
    den = 4 * d0 * np.maximum(sec, 1.)
    profile = 1000. * np.exp(-(1000 * np.asarray(x_mm))**2 / den) / np.sqrt(np.pi * den)
    return np.where(sec > 0, profile, 0.)


def cube(workbook, times):
    fields = []
    for t in times:
        sheet = workbook[SHEETS[t]]
        if sheet['B2'].value != 'MEAN':
            raise ValueError('Unrecognized workbook layout.')
        fields.append(list(sheet.iter_rows(min_row=4, max_row=28, min_col=4, max_col=20, values_only=True)))
    result = np.asarray(fields, dtype=float)
    if not np.isfinite(result).all() or result.shape != (len(times), 25, 17):
        raise ValueError('Missing or unexpected experimental means.')
    return result


def dynamic_profile(x, z, targets, diffusivity, cfg):
    # Difference operator between the DRAIN (known zero) and SOURCE boundaries.
    length, dz, step_h, d0 = (cfg[k] for k in ('length_um', 'pde_dz_um', 'solver_step_h', 'source_diffusivity_um2_s'))
    mesh = int(round(length/dz))
    last = int(round(max(targets)/step_h))
    source = source_amplitude(np.arange(last + 1)*step_h, x, d0)
    delta = 3600*step_h
    decay_rate = 4 * diffusivity/dz**2 * np.sin(.5*np.pi*np.arange(1, mesh)/mesh)**2
    edge = np.zeros(mesh-1)
    edge[-1] = diffusivity/dz**2
    forcing = dst(edge, type=1, norm='ortho')
    survive = np.exp(-delta * decay_rate)
    integral = -np.expm1(-delta * decay_rate)/decay_rate
    ramp = (delta - integral)/(delta*decay_rate)
    state = np.zeros((mesh - 1, len(x)))
    output = {}
    indices = {int(round(hour/step_h)): hour for hour in targets}
    z_idx = np.rint(np.asarray(z)/dz).astype(int)
    if not np.allclose(z_idx*dz, z):
        raise ValueError('Observed depth is not on this grid.')
    for tick in range(last + 1):
        if tick in indices:
            profile = np.vstack((np.zeros(len(x)), idst(state, type=1, norm='ortho', axis=0), source[tick]))
            output[indices[tick]] = profile[z_idx]
        if tick < last:
            effect = integral[:, None]*source[tick] + ramp[:, None]*(source[tick+1]-source[tick])
            state = survive[:, None]*state + forcing[:, None]*effect
    return np.stack([output[hour] for hour in targets])


def freeze(cfg, workbook):
    x = cfg['x_mm']
    z = np.array(cfg['z_um'])
    observed = cube(workbook, cfg['training_hours'])
    targets = cfg['training_hours']+[cfg['held_hour']]
    source = source_amplitude(targets, x, cfg['source_diffusivity_um2_s'])
    steady = source[:, None, :]*(z[None, :, None]/cfg['length_um'])

    def fit(model):
        train = model[:-1].ravel()
        values = observed.ravel()
        design = np.column_stack((train, np.ones(len(train))))
        parameters, error = nnls(design, values)
        return float(error), parameters.tolist()

    steady_error, steady_params = fit(steady)
    candidates = []
    for diffusivity in np.geomspace(*cfg['fit_diffusivity_grid']):
        profile = dynamic_profile(x, z, targets, diffusivity, cfg)
        error, params = fit(profile)
        candidates.append((error, float(diffusivity), params, profile[-1]))
    optimum = min(candidates, key=lambda row: row[0])
    predictions = {
        'persistence_12h': observed[-1],
        'same_profile_source_pulse': observed[-1]*(source[-1]/source[-2])[None, :],
        'quasisteady_pulse': steady_params[0]*steady[-1]+steady_params[1],
        'vertical_fick_pulse': optimum[2][0]*optimum[-1]+optimum[2][1],
    }
    return {'predictions': {a: v.tolist() for a, v in predictions.items()},
            'fit': {'dynamic_diffusivity_um2_s': optimum[1], 'parameter_order': ['pulse_amplitude', 'uniform_offset'],
                    'dynamic': optimum[2], 'quasisteady': steady_params, 'dynamic_early_sse': optimum[0]**2,
                    'quasisteady_early_sse': steady_error**2}, 'preregistered_depth_strata': cfg['depth_strata_um']}


def score(cfg, frozen, workbook):
    reference, truth = cube(workbook, [cfg['training_hours'][-1], cfg['held_hour']])
    z = np.asarray(cfg['z_um'])
    values = {}
    for name, output in frozen['predictions'].items():
        error = np.abs(truth-np.asarray(output))
        per_location = error.mean(axis=0)
        values[name] = {'profile_mae': float(per_location.mean()), 'profile_median_mae': float(np.median(per_location)),
                        'per_location': per_location.tolist(),
                        'depth_strata_mae': [float(np.mean(error[(lo<=z)&(z<=hi)])) for lo, hi in cfg['depth_strata_um']],
                        'direction_agreement_for_nonzero_changes': float(np.mean((np.sign(truth-reference)[truth!=reference]
                            == np.sign(np.asarray(output)-reference)[truth!=reference])))}
    null = np.asarray(values['persistence_12h']['per_location'])
    for row in values.values():
        shift = np.asarray(row['per_location'])-null
        row['betters_ties_worse_than_persistence'] = [int((shift < -1e-12).sum()), int((abs(shift) <= 1e-12).sum()),
                                                     int((shift > 1e-12).sum())]
    return values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('freeze','score'))
    parser.add_argument('run', type=Path)
    args = parser.parse_args()
    reg = args.run/'REGISTERED.json'
    cfg = read(reg)
    content = {'code_sha256': digest(Path(__file__)), 'book_sha256': digest(BOOK), 'receipt_sha256': digest(RECEIPT)}
    if any(cfg[k] != v for k, v in content.items()):
        raise ValueError('Changed registered source data or code.')
    workbook = load_workbook(BOOK, read_only=True, data_only=True)
    if args.stage == 'freeze':
        payload = freeze(cfg, workbook)
    else:
        finished = read(args.run/'FROZEN.json')
        if finished['registered_sha256'] != digest(reg):
            raise ValueError('Changed registration after freeze.')
        payload = {'scores': score(cfg, finished, workbook), 'frozen_sha256': digest(args.run/'FROZEN.json')}
    payload.update(registered_sha256=digest(reg), utc=datetime.now(timezone.utc).isoformat())
    write_new(args.run/('FROZEN.json' if args.stage == 'freeze' else 'SCORED.json'), payload)
    if args.stage == 'score':
        print({k: v['profile_mae'] for k,v in payload['scores'].items()})


if __name__ == '__main__':
    main()
