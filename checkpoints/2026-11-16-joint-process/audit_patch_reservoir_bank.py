"""Compare the accelerated corrected-face bank against independent Radau traces."""

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from audit_patch_reservoir_face import independent_face_curve
from patch_reservoir_face_bank import bank


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def projected(fields, mechanism):
    hydration = fields['h']
    stiffness = (.27 + .93 * fields['b']) * (1 - .27 * hydration)
    half = hydration.shape[1]//2
    sigma = (mechanism['dry_conductivity'] + hydration*mechanism['water_background']
             + mechanism['ion_conductivity']*hydration*fields['c'])
    return np.column_stack((hydration.mean(axis=1), stiffness[:, :half].mean(axis=1),
                            stiffness[:, half:].mean(axis=1), 1/(1/sigma).mean(axis=1)))


def audit(folder):
    utc_started = datetime.now(timezone.utc).isoformat()
    registration, frozen, scored = (folder/f'{name}.json' for name in ('REGISTERED', 'FROZEN', 'SCORED'))
    cfg, predictions, scores = map(read, (registration, frozen, scored))
    if scores['registered_sha256'] != digest(registration) or scores['frozen_sha256'] != digest(frozen):
        raise ValueError('Score and freeze receipts disagree.')
    task_path = Path(cfg['task'])
    task = read(task_path)
    m = task['mechanism']
    end = task['scoring']['horizon']
    times = np.linspace(0., end, round(end/cfg['output_step'])+1)
    fixtures = (('different_prefix', 0, 0), ('transfer_and_shape_shift', 1, 8))
    rows = []
    for cohort, state, action_number in fixtures:
        result = scores['cohorts'][cohort][0]
        frozen_mother = predictions['cohorts'][cohort][0]
        if result['group'] != 0 or frozen_mother['group'] != 0:
            raise ValueError('The fixed fixture mother was not preserved.')
        context = frozen_mother['known']
        action = task['actions'][action_number]
        parameters = np.asarray(result['kinetics'])[state]
        rate, power, loss = parameters
        for mesh in ('secondary', 'evaluation'):
            size, dt = (cfg['grids'][mesh][key] for key in ('cells', 'step'))
            for bath_name in ('wet', 'weak'):
                bath = task['bath'][bath_name]
                production = bank(context, [action], m, bath, parameters[None, :], size, dt, times)
                independent = independent_face_curve({**context, 'release': rate}, action, m, times, bath,
                                                     size, power, loss)
                reference = projected(independent, m)
                closed = ((independent['r'] + independent['c']).mean(axis=1)*context['thickness']
                          + independent['lost'])
                errors = np.abs(reference - production[0, :, 0]).max(axis=0)
                rows.append({'cohort': cohort, 'mother': 0, 'state': state, 'action': action_number,
                             'bath': bath_name, 'cells': size, 'step': dt, 'time_end': end,
                             'bank_dimensions': list(production.shape), 'independent_field_dimensions':
                             {key: list(value.shape) for key, value in independent.items()},
                             'projected_dimensions': list(reference.shape), 'recorded_margin_dimensions':
                             list(np.asarray(result['margins']).shape),
                             'trace_max_error_by_channel': errors.tolist(),
                             'independent_stock_closure': float(np.abs(closed-closed[0]).max())})
    max_trace = max(max(row['trace_max_error_by_channel']) for row in rows)
    max_closure = max(row['independent_stock_closure'] for row in rows)
    if max_trace > 3e-5 or max_closure > 1e-10:
        raise ArithmeticError('Independent full-trajectory fixture failed its numerical limits.')
    sources = [Path(__file__), Path('audit_patch_reservoir_face.py'), Path('audit_patch_reservoir_service.py'),
               Path('patch_reservoir_face_bank.py'), Path('patch_reservoir_service.py')]
    return {'started_utc': utc_started, 'completed_utc': datetime.now(timezone.utc).isoformat(),
            'fixture_rule': 'Mother zero in each cohort, fixed state/action pairs (0, 0), (1, 8). Both baths and both meshes.',
            'input_sha256': {str(p.as_posix()): digest(p) for p in (registration, frozen, scored, task_path)},
            'source_sha256': {p.name: digest(p) for p in sources},
            'trace_channel_order': ['mean hydration', 'near-half stiffness', 'far-half stiffness', 'conductivity'],
            'maximum_trace_error': max_trace, 'maximum_independent_inventory_defect': max_closure,
            'fixtures': rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    folder = parser.parse_args().run
    output = folder/'BANK_INDEPENDENT.json'
    if output.exists():
        raise FileExistsError(output)
    result = audit(folder)
    with output.open('x', encoding='utf-8') as f:
        json.dump(result, f, indent=2, allow_nan=False)
        f.write('\n')
    print(f"Checked {len(result['fixtures'])} fixtures. Max trace error: {result['maximum_trace_error']:.3g}.")


if __name__ == '__main__':
    main()
