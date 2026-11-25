"""Forward fixed-grid formation with an injectable spatial force evaluator.

This path is separate from the archived step-sensitivity experiment. It consumes
explicit Brownian increments and reaction innovations; it does not infer rates
or material parameters.
"""

import numpy as np
from numba import njit

from formation_step import TICK, add_feed, apply_event, candidates, factor_weights, CAP
from aam_spatial_motion import _compact_edges


@njit(cache=True)
def move_with_force(state, start, end, brownian, mobility, geometry, force_fn):
    ids = np.flatnonzero(state.present)
    edges = _compact_edges(state.edges[:state.counts[1]], state.present)
    box, spring, rest, repulsion, core = geometry[2:7]
    forces, _ = force_fn(state.points[state.present], edges, box, spring, rest, repulsion, core)
    dt = TICK * (end-start)
    for row, particle in enumerate(ids):
        for axis in range(3):
            noise = brownian[end, particle, axis]-brownian[start, particle, axis]
            dx = dt*mobility[particle]*forces[row, axis]+np.sqrt(2.*mobility[particle])*noise
            state.points[particle, axis] = (state.points[particle, axis]+dx)%box
            state.unwrapped[particle, axis] += dx


@njit(cache=True)
def advance_fixed(packet, start, end, brownian, expo, marks, index, remaining, step, force_fn):
    """Advance a completed packet and return the residual unit-rate clock."""
    state, material, geometry, rates, factors = packet
    aam, initially_present = int(material[0]), int(material[0]+material[2])
    feed = round(material[3]/TICK)
    if step <= 0 or step%2 or start < 0 or start%2 or end%2 or end < start or end >= len(brownian):
        raise ValueError('Elapsed interval and positive step must lie on an even lattice.')
    if state.position[0] != TICK*start:
        raise ValueError('Packet must be at the requested completed start boundary.')
    if material[3] <= TICK*end and (feed%2 or abs(feed*TICK-material[3]) > 1.e-12):
        raise ValueError('Any feed during the interval must lie on an even lattice.')
    box, capture, inner, corridor = geometry[2], geometry[7], geometry[8], geometry[9]
    mobility = np.full(len(state.points), geometry[1])
    mobility[aam:] *= .8
    sites = 2*len(mobility)
    capacity = sites*sites+2*sites
    rows = np.empty((capacity, 6), np.int64)
    distances, primitive, weights = np.empty(capacity), np.empty(capacity), np.empty(capacity)
    risk = np.empty((3, CAP+1))
    cumulative = np.empty(capacity)
    at = start
    while at < end:
        if at == feed and not state.present.all():
            add_feed(state, initially_present, material[3])
        stop = min(at+step, end)
        if at < feed < stop:
            stop = feed
        half = (at+stop)//2
        move_with_force(state, at, half, brownian, mobility, geometry, force_fn)
        live = np.flatnonzero(state.present)
        cache = np.full((len(mobility), len(mobility)), -1, np.int8)
        local, terminal = TICK*at, TICK*stop
        while local < terminal:
            count = candidates(state, aam, box, capture, inner, corridor, rates, cache,
                               live, rows, distances, primitive)
            hazard = primitive[:count].sum()
            exposure = hazard*(terminal-local)
            if hazard <= 0. or remaining >= exposure:
                remaining = max(0., remaining-exposure)
                state.position[3] += exposure
                break
            local += remaining/hazard
            state.position[3] += remaining
            factor_weights(primitive, rows[:, 4], rows[:, 5], count, factors, weights, risk)
            total = 0.
            for i in range(count):
                total += weights[i]
                cumulative[i] = total
            if total <= 0. or index+1 >= len(expo):
                raise ValueError('Invalid mark normalizer or insufficient reaction innovations.')
            selected = np.searchsorted(cumulative[:count], marks[index]*total, side='right')
            if selected >= count or weights[selected] <= 0.:
                raise ValueError('No legal positive reaction mark was selected.')
            apply_event(state, rows[selected], distances[selected], box, local, TICK*half)
            index += 1
            remaining = expo[index]
        move_with_force(state, half, stop, brownian, mobility, geometry, force_fn)
        at = stop
    if end == feed and not state.present.all():
        add_feed(state, initially_present, material[3])
    state.position[:3] = TICK*end
    return index, remaining
