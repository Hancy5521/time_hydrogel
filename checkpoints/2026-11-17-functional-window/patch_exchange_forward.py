"""Independent synthetic service mechanism for hydrated ion exchange.

The stored dry formulation and measured output definitions match the previous test.
Water diffuses at a hydration-dependent rate, released stock may rebind, free ions
alter their own mobility, and the finite bath is an evolving inventory that can
exchange in either direction. This is a mechanism stress model without empirical
parameter calibration. It shares preparation, but no service integration code,
with the irreversible constant-water-diffusivity benchmark.
"""

import numpy as np
from numba import njit

from patch_reservoir_joint_bank import prepared_fields


DEFAULT_EXCHANGE = {
    'water_dry_ratio': .47,
    'water_wet_ratio': 1.0,
    'water_power': 1.45,
    'ion_dry_ratio': .13,
    'ion_water_power': 1.32,
    'mobility_half': 1.4,
    'release_half': 1.1,
    'rebind': .23,
    'rebind_water_power': 1.25,
    'extra_capacity': .30,
    'bath_volume_ratio': .62,
    'bath_partition': .80,
}


def exchange_configuration(outcome_cfg):
    result = {**DEFAULT_EXCHANGE, **(outcome_cfg or {})}
    if set(result) != set(DEFAULT_EXCHANGE):
        absent = set(result) - set(DEFAULT_EXCHANGE)
        raise ValueError(f'Unrecognized exchange fields {sorted(absent)}.')
    if not all(np.isfinite(value) for value in result.values()):
        raise ValueError('Expected finite exchange mechanism fields.')
    positive = set(result) - {'rebind', 'extra_capacity', 'bath_partition'}
    if min(result[key] for key in positive) <= 0 or min(result[key] for key in set(result)-positive) < 0:
        raise ValueError('Expected positive rates, scales and powers with nonnegative capacities and binding.')
    if result['ion_dry_ratio'] > 1:
        raise ValueError('The dry ion mobility ratio cannot exceed the wet reference.')
    return result


@njit(cache=True, inline='always')
def _water_rates(h, b, deriv_h, deriv_b, dx, dh_ref, ratio_dry, ratio_wet, power, on, off, wet_transfer, bath):
    d_left = dh_ref*(ratio_dry + (ratio_wet-ratio_dry)*h[0]**power)
    face_exchange = wet_transfer/(1 + wet_transfer*dx/(2*d_left))
    deriv_h[0] = face_exchange*(bath-h[0])/dx
    for cell in range(len(h)-1):
        d_right = dh_ref*(ratio_dry+(ratio_wet-ratio_dry)*h[cell+1]**power)
        inward = 2*(h[cell+1]-h[cell]) / ((1/d_left+1/d_right)*dx**2)
        deriv_h[cell] += inward
        deriv_h[cell+1] = -inward
        d_left = d_right
    for cell in range(len(h)):
        deriv_b[cell] = on*(1-h[cell])*(1-b[cell])-off*h[cell]*b[cell]


@njit(cache=True)
def _hydraulic_path(cells, h_frames, h_cycles, h_period, dx, dh_ref, mechanism, bonds, wet_transfer, bath):
    history = np.empty((h_frames, cells))
    modulus = np.empty((h_frames, 2))
    surface = np.empty(h_frames)
    h = np.zeros(cells)
    b = np.ones(cells)
    h1, h2, b1, b2 = (np.empty(cells), np.empty(cells), np.empty(cells), np.empty(cells))
    dh, db = np.empty(cells), np.empty(cells)
    dh_dry, dh_wet, water_power = mechanism
    on, off = bonds
    dt = h_period/h_cycles
    for sample in range(h_frames):
        history[sample] = h
        stiffness = (.27+.93*b)*(1-.27*h)
        modulus[sample, 0] = stiffness[:cells//2].mean()
        modulus[sample, 1] = stiffness[cells//2:].mean()
        d0 = dh_ref*(dh_dry+(dh_wet-dh_dry)*h[0]**water_power)
        film = wet_transfer*dx/(2*d0)
        surface[sample] = (h[0]+film*bath)/(1+film)
        if sample == h_frames-1:
            break
        for _ in range(h_cycles):
            _water_rates(h,b,dh,db,dx,dh_ref,dh_dry,dh_wet,water_power,on,off,wet_transfer,bath)
            for cell in range(cells):
                h1[cell], b1[cell] = h[cell]+dt*dh[cell], b[cell]+dt*db[cell]
            _water_rates(h1,b1,dh,db,dx,dh_ref,dh_dry,dh_wet,water_power,on,off,wet_transfer,bath)
            for cell in range(cells):
                h2[cell] = .25*(3*h[cell]+h1[cell]+dt*dh[cell])
                b2[cell] = .25*(3*b[cell]+b1[cell]+dt*db[cell])
            _water_rates(h2,b2,dh,db,dx,dh_ref,dh_dry,dh_wet,water_power,on,off,wet_transfer,bath)
            for cell in range(cells):
                h[cell] = (h[cell]+2*(h2[cell]+dt*dh[cell]))/3
                b[cell] = (b[cell]+2*(b2[cell]+dt*db[cell]))/3
    return history, modulus, surface


@njit(cache=True, inline='always')
def _ion_rates(r,c,mass,hydrated_rate,water_mobility,binding_rate,surface,capacity,dx,bath_volume,rate,loss,
               interface,exchange,dr,dc,diffusion):
    mobility_half, release_half, partition = exchange
    for cell in range(len(c)):
        diffusion[cell] = water_mobility[cell]/(1+c[cell]/mobility_half)
        released = rate*hydrated_rate[cell]*r[cell]/(1+c[cell]/release_half)
        free_capacity = max(0., capacity[cell]-r[cell])
        reabsorbed = binding_rate[cell]*c[cell]*free_capacity/(free_capacity+c[cell]+1e-15)
        dc[cell] = released-reabsorbed
        dr[cell] = -dc[cell]
    for face in range(len(c)-1):
        resistance = .5*dx*(1/diffusion[face]+1/diffusion[face+1])
        if face == len(c)//2-1:
            resistance += interface
        inward = (c[face+1]-c[face])/resistance/dx
        dc[face] += inward
        dc[face+1] -= inward
    exchange_rate = loss*surface
    escaped = exchange_rate*(c[0]-partition*mass/bath_volume)/(1+exchange_rate*dx/(2*diffusion[0]))
    dc[0] -= escaped/dx
    return escaped


@njit(cache=True)
def _ion_path(water, hydrated_rate, water_mobility, binding_rate, surface, stocks, kinetics, ticks, dt, dx,
              mobile_ref, bath_volume, extra_capacity, conductivity, exchange, return_inventory):
    time_count = len(ticks)
    cases, cells = stocks.shape
    trace = np.empty((cases, time_count))
    fractions = np.empty((cases, time_count if return_inventory else 0, 3))
    conservation = 0.
    minimum = 0.
    maximum_capacity_violation = 0.
    dry, background, ion = conductivity
    for case, (rate,loss,rho) in enumerate(kinetics):
        r = stocks[case].copy()
        c = np.zeros(cells)
        inventory0 = dx*r.sum()
        capacity = r+extra_capacity*r.mean()
        interface = rho*dx*cells/mobile_ref
        mass = 0.
        dr, dc, diffusion = (np.empty(cells), np.empty(cells), np.empty(cells))
        r1,r2,c1,c2 = (np.empty(cells),np.empty(cells),np.empty(cells),np.empty(cells))
        sample = 0
        for tick in range(ticks[-1]+1):
            h_i = 2*tick
            rsum, csum = 0., 0.
            for cell in range(cells):
                rsum += r[cell]
                csum += c[cell]
                minimum = min(minimum,r[cell],c[cell])
                maximum_capacity_violation = max(maximum_capacity_violation,r[cell]-capacity[cell])
            minimum = min(minimum,mass)
            conservation = max(conservation, abs(dx*(rsum+csum)+mass-inventory0)/inventory0)
            if tick == ticks[sample]:
                h = water[h_i]
                sigma = dry+background*h+ion*h*c
                trace[case,sample] = 1/(1/sigma).mean()
                if return_inventory:
                    fractions[case,sample,0] = dx*rsum/inventory0
                    fractions[case,sample,1] = dx*csum/inventory0
                    fractions[case,sample,2] = mass/inventory0
                sample += 1
            if tick == ticks[-1]:
                break
            j0 = _ion_rates(r,c,mass,hydrated_rate[h_i],water_mobility[h_i],binding_rate[h_i],surface[h_i],
                            capacity,dx,bath_volume,rate,loss,interface,exchange,dr,dc,diffusion)
            for cell in range(cells):
                r1[cell], c1[cell] = r[cell]+dt*dr[cell], c[cell]+dt*dc[cell]
            m1 = mass+dt*j0
            last = h_i+2
            j1 = _ion_rates(r1,c1,m1,hydrated_rate[last],water_mobility[last],binding_rate[last],surface[last],
                            capacity,dx,bath_volume,rate,loss,interface,exchange,dr,dc,diffusion)
            for cell in range(cells):
                r2[cell] = .25*(3*r[cell]+r1[cell]+dt*dr[cell])
                c2[cell] = .25*(3*c[cell]+c1[cell]+dt*dc[cell])
            m2 = .25*(3*mass+m1+dt*j1)
            mid = h_i+1
            j2 = _ion_rates(r2,c2,m2,hydrated_rate[mid],water_mobility[mid],binding_rate[mid],surface[mid],
                            capacity,dx,bath_volume,rate,loss,interface,exchange,dr,dc,diffusion)
            for cell in range(cells):
                r[cell] = (r[cell]+2*(r2[cell]+dt*dr[cell]))/3
                c[cell] = (c[cell]+2*(c2[cell]+dt*dc[cell]))/3
            mass = (mass+2*(m2+dt*j2))/3
    return trace,fractions,conservation,minimum,maximum_capacity_violation


def exchange_bank(parent, actions, cfg, bath, kinetic, stock, cells, step, times, *, outcome_cfg=None, inventory=False):
    """Compute [state, time, action, (mean h, two modulus averages, conduction)].

    ``kinetic`` contains k, hydration exponent, bath transfer and interface
    resistance. ``stock`` contains loading bias and within-half shape. Optional
    inventory is ordered (bound, mobile, in bath), all per initial total stock.
    Bath stock is an instantaneous inventory rather than a cumulative escape.
    The times are aligned to ``step``; the flows use CFL-informed subcycles.
    """
    ex = exchange_configuration(outcome_cfg)
    times = np.asarray(times,dtype=float)
    rates = np.asarray(kinetic,dtype=float)
    if (not np.isfinite(step) or step<=0 or not 0<=bath<=1 or cells<8 or cells%2 or not actions
            or times.ndim!=1 or not len(times) or not np.isfinite(times).all()):
        raise ValueError('Expected bounded environment, nonempty actions, valid time step and even aligned mesh.')
    if (rates.ndim!=2 or rates.shape[1]!=4 or not len(rates)
            or not np.isfinite(rates).all() or (rates[:,:2]<=0).any() or (rates[:,2:]<0).any()):
        raise ValueError('Provide positive release rates/powers with nonnegative transfers/resistances.')
    steps = np.rint(times/step).astype(np.int64)
    if steps[0]!=0 or (np.diff(steps)<=0).any() or not np.allclose(times,steps*step,atol=1e-12,rtol=0):
        raise ValueError('Output times must start at zero and strictly increase on integration grid.')
    wettability = np.array([act['wet_transfer'] for act in actions],dtype=float)
    dc_ref, dh_ref, thick = cfg['mobile_D'], parent['water_D'], parent['thickness']
    if min(dc_ref,dh_ref,thick)<=0 or not np.isfinite(wettability).all() or wettability.min()<0:
        raise ValueError('Diffusivity and thickness must be positive, transfer nonnegative.')
    initial = prepared_fields(parent,actions,cfg,stock,cells)
    if len(initial)!=len(rates):
        raise ValueError('Pair one initial stock pattern with each kinetic state.')
    dx = thick/cells
    bath_volume = ex['bath_volume_ratio']*thick
    outflow = rates[:,2].max()*bath
    cfl_ion = max(2*dc_ref/dx**2+outflow/dx+ex['rebind'],rates[:,0].max(),outflow*ex['bath_partition']/bath_volume)
    ion_cycles = max(1,int(np.ceil(step*cfl_ion/.92)))
    ion_period = step/ion_cycles
    dh_max = dh_ref*max(ex['water_dry_ratio'],ex['water_wet_ratio'])
    cfl_water = max(2*dh_max/dx**2,dh_max/dx**2+wettability.max()/dx,cfg['bond_on']+cfg['bond_off'])
    h_cycles = max(1,int(np.ceil(.5*ion_period*cfl_water/.92)))
    ticks = steps*ion_cycles
    trace = np.empty((len(rates),len(times),len(actions),4))
    fractions = np.empty(trace.shape[:-1]+(3,)) if inventory else None
    ion = tuple(float(ex[key]) for key in ('mobility_half','release_half','bath_partition'))
    water = tuple(float(ex[key]) for key in ('water_dry_ratio','water_wet_ratio','water_power'))
    bonds = float(cfg['bond_on']),float(cfg['bond_off'])
    conductivity = tuple(float(cfg[key]) for key in ('dry_conductivity','water_background','ion_conductivity'))
    for transfer in np.unique(wettability):
        h,e,surface = _hydraulic_path(cells,2*ticks[-1]+1,h_cycles,ion_period/2,dx,dh_ref,water,bonds,transfer,bath)
        if not np.isfinite(h).all() or h.min()<-1e-11 or h.max()>bath+1e-10:
            raise ArithmeticError('Nonlinear hydration integration violated its maximum principle.')
        ratio = ex['ion_dry_ratio']
        mobility = dc_ref*(ratio+(1-ratio)*h**ex['ion_water_power'])
        binding_rate = ex['rebind']*(1-h)**ex['rebind_water_power']
        action_indices = np.flatnonzero(wettability==transfer)
        hydration = h[2*ticks].mean(axis=-1)[None]
        for power in np.unique(rates[:,1]):
            indices = rates[:,1]==power
            hydrated_rate = h**power
            kinetic_subset = rates[indices][:,[0,2,3]].copy(order='C')
            for action_index in action_indices:
                sigma,budget,error,minimum,overfill = _ion_path(h,hydrated_rate,mobility,binding_rate,surface,
                    initial[indices,action_index],kinetic_subset,ticks,ion_period,dx,dc_ref,bath_volume,
                    ex['extra_capacity'],conductivity,ion,inventory)
                if (not np.isfinite(sigma).all() or error>2e-10 or minimum<-1e-11 or overfill>1e-10):
                    raise ArithmeticError('Ion exchange violated site capacity, positivity or stock conservation.')
                trace[indices,:,action_index,0] = hydration
                trace[indices,:,action_index,1:3] = e[2*ticks][None]
                trace[indices,:,action_index,3] = sigma
                if inventory:
                    fractions[indices,:,action_index] = budget
    return (trace,fractions) if inventory else trace
