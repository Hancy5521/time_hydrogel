"""Recompute fixed joint-stock fixtures with an independent Radau solver."""

import argparse
from datetime import datetime, timezone
from pathlib import Path
import json

import numpy as np
from scipy.stats import qmc

from audit_patch_reservoir_bank import digest, projected, read
from audit_patch_reservoir_service import _integrate, _local_sparsity, _net_face_flow, independent_curve
from patch_reservoir_joint_bank import joint_bank, prepared_fields


def audit_dry(parent, action, cfg, cells, bias, shape):
    """Independently integrate the preparation and redistribute the dry load."""
    near = action['near_loading']
    biased_near = near*np.exp(bias)/(1 - near + near*np.exp(bias))
    command = {**action, 'near_loading': biased_near}
    initial = independent_curve(parent, command, cfg, [0], 0., cells, 1., 0.)['r'][0]
    half = cells//2
    x = (np.arange(cells) + .5)/cells
    raw = initial*(1 + shape*np.cos(4*np.pi*x))
    remapped = raw.copy()
    for interval in (slice(None, half), slice(half, None)):
        remapped[interval] *= initial[interval].sum()/raw[interval].sum()
    return remapped, initial


def independent_joint_curve(parent, action, cfg, times, bath, cells, parameters):
    """Integrate corrected-face service equations with explicit face resistances."""
    rate, power, leach, rho, bias, shape = parameters
    stock, unshaped = audit_dry(parent, action, cfg, cells, bias, shape)
    dx = parent['thickness']/cells
    water_half = 2*parent['water_D']/dx
    transfer = action['wet_transfer']

    def rhs(_, flat):
        h, b, r, c = flat[:-1].reshape(cells, 4).T
        wet = np.clip(h, 0, 1)
        surface = (water_half*h[0] + transfer*bath)/(water_half + transfer)
        dh = _net_face_flow(h, parent['water_D'], dx)
        dh[0] += transfer*(bath-surface)/dx
        db = cfg['bond_on']*(1-wet)*(1-b) - cfg['bond_off']*wet*b
        released = rate*wet**power*r
        mobility = cfg['mobile_D']*(.1 + .9*wet)
        face_resistance = .5*dx/mobility[:-1] + .5*dx/mobility[1:]
        face_resistance[cells//2-1] += rho*parent['thickness']/cfg['mobile_D']
        dc = _net_face_flow(c, dx/face_resistance, dx) + released
        loss_resistance = dx/(2*mobility[0])
        surface_rate = leach*surface
        escaped = surface_rate*c[0]/(1+surface_rate*loss_resistance)
        dc[0] -= escaped/dx
        return np.r_[np.column_stack((dh, db, -released, dc)).ravel(), escaped]

    starting = np.column_stack((np.zeros(cells), np.ones(cells), stock, np.zeros(cells)))
    result = _integrate(rhs, times, np.r_[starting.ravel(), 0.],
                        _local_sparsity(cells, 4, boundary_integral=True))
    samples = result[:, :-1].reshape(len(times), cells, 4)
    fields = dict(zip(('h','b','r','c'), samples.transpose(2,0,1), strict=True))
    inventory = np.column_stack((dx*fields['r'].sum(axis=1), dx*fields['c'].sum(axis=1), result[:, -1]))
    inventory /= dx*stock.sum()
    bins = stock.reshape(8, cells//8).sum(axis=1)/stock.sum()
    undo = np.max(np.abs(stock.reshape(2, cells//2).sum(axis=1)-unshaped.reshape(2, cells//2).sum(axis=1)))
    return projected(fields, cfg), inventory, stock, bins, float(undo)


def fixture(case, context, task, cfg, bath_name, mesh, shape_pair, materials):
    cells, step = cfg['meshes'][mesh]
    sample = cfg['service_sample_step']
    elapsed = np.arange(round(task['scoring']['horizon']/sample)+1)*sample
    action_index = cfg['fixed_diagnostic_action']
    action = task['actions'][action_index]
    bath = task['bath'][bath_name]
    traces, mass = joint_bank(context, [action], materials, bath, shape_pair[:, :4], shape_pair[:, 4:],
                              cells, step, elapsed, inventory=True)
    stock = prepared_fields(context, [action], materials, shape_pair[:, 4:], cells)[:, 0]
    checkpoints = np.rint(np.asarray(cfg['process_times'])/sample).astype(int)
    records = []
    for state, vector in enumerate(shape_pair):
        curve, inventory, prepared, bins, half_error = independent_joint_curve(
            context, action, materials, elapsed, bath, cells, vector)
        row = {'state':state, 'absolute_trace_error':np.abs(traces[state, :, 0]-curve).max(axis=0).tolist(),
               'inventory_error':np.abs(mass[state, :, 0]-inventory).max(axis=0).tolist(),
               'dry_field_max_error':float(np.abs(stock[state]-prepared).max()),
               'dry_half_residual':half_error, 'inventory_closure':float(np.abs(inventory.sum(axis=1)-1).max())}
        if mesh == 'evaluation' and bath_name == 'wet':
            physical = np.column_stack((inventory[checkpoints], curve[checkpoints, -1]))
            row['scored_dry_bin_error'] = float(np.abs(np.asarray(case['truth']['dry'][state])-bins).max())
            row['scored_process_error'] = np.abs(np.asarray(case['truth']['process'][state])-physical).max(axis=0).tolist()
        records.append(row)
    return {'mesh':mesh, 'cells':cells, 'step':step, 'bath':bath_name, 'action':action_index,
            'sample_count':len(elapsed), 'shape_twin_conductance_difference':float(np.abs(traces[0,: ,0,-1]-traces[1,:,0,-1]).max()),
            'states':records}


def fixed_context(cohort, case, frozen, task, cfg):
    spec = cfg['evaluation'][cohort]
    point = qmc.Sobol(13, scramble=True, seed=spec['seed']).random_base2(0)[0]
    reconstructed = {name: lo+(hi-lo)*p for (name, (lo, hi)), p in zip(task['known_ranges'].items(), point[:3], strict=True)}
    context = frozen['known']
    if case['group'] != 0 or frozen['group'] != 0 or context.keys() != reconstructed.keys():
        raise ValueError('The fixed fixture parent was not preserved.')
    if not np.allclose(list(context.values()), list(reconstructed.values()), rtol=0, atol=1e-13):
        raise ValueError('Archived and reconstructed parent context disagree.')
    kinetic_stock = np.asarray(case['kinetics'])[:2]
    expected = np.array([np.interp(point[i], [0,1], spec[key]) for i, key in enumerate(
        ('power','loss','interface','bias','shape'), 3)])
    rate = context['release']*cfg['kinetic_grid']['factors'][0]*(1+spec['rate_jitter']*(2*point[12]-1))
    power, loss, rho, bias, shape = expected
    expected_pair = [[rate,power,loss,rho,-bias,-shape], [rate,power,loss,rho,-bias,shape]]
    if not np.allclose(kinetic_stock, expected_pair, rtol=0, atol=1e-13):
        raise ValueError('Archived and reconstructed within-half twins disagree.')
    process_D = np.interp(point[8], [0,1], spec['precursor_diffusion_factor'])*task['mechanism']['process_D']
    return context, kinetic_stock, {**task['mechanism'], 'process_D':float(process_D)}


def audit(run):
    started = datetime.now(timezone.utc).isoformat()
    registration, frozen, scored = (run/f'{name}.json' for name in ('REGISTERED','FROZEN','SCORED'))
    cfg, locked, outcomes = map(read, (registration, frozen, scored))
    if outcomes['registered_sha256'] != digest(registration) or outcomes['frozen_sha256'] != digest(frozen):
        raise ValueError('Score and freeze hashes disagree.')
    task_path = Path(cfg['task_path'])
    task = read(task_path)
    if cfg['inputs'][task_path.as_posix()] != digest(task_path):
        raise ValueError('Previous service task changed.')
    for name in ('patch_reservoir_joint_bank.py','patch_reservoir_face_bank.py','patch_reservoir_service.py'):
        if cfg['sources'][name] != digest(Path(name)):
            raise ValueError(f'Frozen production source changed: {name}')

    cohorts, counts = {}, []
    for key in cfg['evaluation']:
        case = outcomes['cohorts'][key][0]
        context, pair, material = fixed_context(key, case, locked['cohorts'][key][0], task, cfg)
        trials = [fixture(case, context, task, cfg, bath, grid, pair, material)
                  for bath in ('wet','weak') for grid in ('secondary','evaluation')]
        cohorts[key] = {'group':0,'kinetics':pair.tolist(),'prepared_diffusivity':material['process_D'],'fixtures':trials}
        counts.extend(state for trial in trials for state in trial['states'])

    largest = {'trace':max(max(x['absolute_trace_error']) for x in counts),
               'inventory':max(max(x['inventory_error']) for x in counts),
               'dry_field':max(x['dry_field_max_error'] for x in counts),
               'inventory_closure':max(x['inventory_closure'] for x in counts),
               'half_residual':max(x['dry_half_residual'] for x in counts),
               'scored_dry_bins':max(x['scored_dry_bin_error'] for x in counts if 'scored_dry_bin_error' in x),
               'scored_process':max(max(x['scored_process_error']) for x in counts if 'scored_process_error' in x)}
    limits = {'trace':3e-5,'inventory':3e-7,'dry_field':1e-8,'inventory_closure':1e-10,
              'half_residual':1e-10,'scored_dry_bins':1e-8,'scored_process':3e-5}
    source_names = (Path(__file__).name, 'audit_patch_reservoir_bank.py','audit_patch_reservoir_service.py',
                    'audit_patch_reservoir_face.py','patch_reservoir_joint_bank.py','patch_reservoir_face_bank.py',
                    'patch_reservoir_service.py')
    return {'started_utc':started, 'completed_utc':datetime.now(timezone.utc).isoformat(),
            'fixture_rule':'First parent, first rate and bias, both within-half twins, central catalogue action, both baths and 48/96 grids.',
            'input_sha256':{p.as_posix():digest(p) for p in (registration,frozen,scored,task_path)},
            'source_sha256':{name:digest(Path(name)) for name in source_names},
            'trace_order':['water','near_stiffness','far_stiffness','conductance'],
            'inventory_order':['bound','mobile','escaped'], 'max_errors':largest, 'limits':limits,
            'passed':all(largest[key] <= value for key, value in limits.items()), 'cohorts':cohorts}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    run = parser.parse_args().run
    output = run/'BANK_INDEPENDENT.json'
    if output.exists():
        raise FileExistsError(output)
    result = audit(run)
    with output.open('x', encoding='utf-8') as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write('\n')
    print(result['max_errors'])
    if not result['passed']:
        raise ArithmeticError('Independent joint Radau fixture exceeded a declared tolerance.')


if __name__ == '__main__':
    main()
