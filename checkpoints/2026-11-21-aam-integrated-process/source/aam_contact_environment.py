"""Condition mobile reaction hazards on occupancy of the approach midpoint.

Uses the frozen mobility, periodic forces, and vinyl event rules from
aam_spatial_motion. This version records a new risk-set dimension. The
midpoint count does not encode atom orientation, solvent, or chemistry.
"""
import numpy as np
from numba import njit

from aam_mbaa_kinetics import Recipe, _join, _root
from aam_spatial_motion import Motion, _compact_edges, _distance, _scalar_distance, brownian_step, spatial_forces


@njit(cache=True)
def corridor_occupancy(points, cache, present, i, j, box, radius, cap):
    cached = cache[i, j]
    if cached >= 0:
        return cached
    delta, _ = _distance(points[i], points[j], box)
    midpoint = points[i]+.5*np.asarray(delta)
    occupants = 0
    for k in present:
        if k == i or k == j:
            continue
        offset, _ = _distance(midpoint, points[k], box)
        if offset[0]**2+offset[1]**2+offset[2]**2 < radius**2:
            occupants += 1
            if occupants == cap:
                break
    cache[i, j] = cache[j, i] = occupants
    return occupants


@njit(cache=True)
def _environment_path(aam, mbaa, initial, pulse, initiators, horizon, clock, points, dt,
                 mobility, box, spring, rest, repulsion, core, radius, inner, corridor, rate, seed):
    np.random.seed(seed)
    n = len(points)
    all_sites = 2*n
    consumed = np.zeros(n, np.int64)
    active = np.zeros(all_sites, np.bool_)
    available = np.zeros(all_sites, np.bool_)
    available[:2*aam:2] = True
    available[2*aam:2*(aam+initial)] = True
    present = np.arange(n) < aam+initial
    labels = np.arange(n)
    max_events = aam+2*mbaa+initiators//2+mbaa+1
    events = np.empty((max_events, 8))
    edges = np.empty((max_events, 2), np.int64)
    event_positions = np.empty((max_events, 2, 3))
    frames = np.empty((len(clock), n, 3))
    frame_images = np.empty_like(frames)
    fed_frames = np.empty((len(clock), n), np.bool_)
    force_frames = np.empty_like(frames)
    energies = np.empty(len(clock))
    counts = np.zeros((8, 2, rate.shape[2]), np.int64)
    exposure = np.zeros((8, 2, rate.shape[2]))
    channels = np.empty(all_sites*all_sites+2*all_sites, np.int64)
    ranks = np.empty_like(channels)
    crowding = np.empty_like(channels)
    source = np.empty_like(channels)
    target = np.empty_like(channels)
    distances = np.empty(len(channels))
    weights = np.empty(len(channels))
    tr = 0.
    ev = ed = frame = 0
    fed = initial == mbaa
    unwrapped = points.copy()
    while frame < len(clock):
        if not fed and tr == pulse:
            for molecule in range(aam+initial, n):
                present[molecule] = True
                available[2*molecule:2*molecule+2] = True
                events[ev] = np.array((tr, 0., -1., float(molecule), -1., -1., tr, -1.))
                event_positions[ev, 0] = np.nan
                event_positions[ev, 1] = points[molecule]
                ev += 1
            fed = True
        forces, energy = spatial_forces(points[present], _compact_edges(edges[:ed], present),
                                         box, spring, rest, repulsion, core)
        if tr == clock[frame]:
            frames[frame] = points
            frame_images[frame] = unwrapped
            fed_frames[frame] = present
            force_frames[frame] = 0.
            force_frames[frame, present] = forces
            energies[frame] = energy
            frame += 1
        if tr == horizon:
            break
        endpoint = min(tr+dt, clock[frame])
        if not fed:
            endpoint = min(endpoint, pulse)
        half = .5*(endpoint-tr)
        present_points, displacement = brownian_step(
            points[present], forces, half, mobility[present], np.random.standard_normal(forces.shape), box)
        points[present] = present_points
        unwrapped[present] += displacement
        occupancy = np.full((n, n), -1, np.int8)
        at_midpoint = np.flatnonzero(present)
        local = tr
        while local < endpoint:
            choices = 0
            opportunity = np.zeros((8, 2, rate.shape[2]))
            ra = np.flatnonzero(active)
            vacant = np.flatnonzero(available)
            for v in vacant:
                vh = v//2
                if initiators > 0 and consumed[vh] == 0:
                    ch = int(vh >= aam)
                    source[choices], target[choices] = -1, v
                    channels[choices], ranks[choices], distances[choices] = ch, 0, -1.
                    crowding[choices] = 0
                    weights[choices] = initiators*rate[ch, 0, 0]
                    opportunity[ch, 0, 0] += initiators
                    choices += 1
            for u in ra:
                uh = u//2
                for v in vacant:
                    vh = v//2
                    if uh == vh:
                        continue
                    r = _scalar_distance(points, uh, vh, box)
                    if r >= radius:
                        continue
                    ch = 2 if vh < aam else 3
                    if consumed[vh] == 1:
                        ch = 5 if _root(labels, uh) == _root(labels, vh) else 4
                    band = int(r >= inner)
                    source[choices], target[choices], channels[choices] = u, v, ch
                    ranks[choices], distances[choices] = band, r
                    level = corridor_occupancy(points, occupancy, at_midpoint, uh, vh, box, corridor, rate.shape[2]-1)
                    crowding[choices] = level
                    opportunity[ch, band, level] += 1.
                    weights[choices] = rate[ch, band, level]
                    choices += 1
            for i in range(len(ra)-1):
                u = ra[i]
                for v in ra[i+1:]:
                    r = _scalar_distance(points, u//2, v//2, box)
                    if r >= radius:
                        continue
                    band = int(r >= inner)
                    level = corridor_occupancy(points, occupancy, at_midpoint, u//2, v//2, box, corridor, rate.shape[2]-1)
                    for ch in range(6, 8):
                        if ch == 6 and u//2 == v//2:
                            continue
                        source[choices], target[choices], channels[choices] = u, v, ch
                        ranks[choices], distances[choices] = band, r
                        crowding[choices] = level
                        opportunity[ch, band, level] += 1.
                        weights[choices] = rate[ch, band, level]
                        choices += 1
            cumulative = np.cumsum(weights[:choices])
            hazard = cumulative[-1] if choices else 0.
            occurred = local + np.random.exponential()/hazard if hazard else np.inf
            end = min(occurred, endpoint)
            exposure += (end-local)*opportunity
            local = end
            if occurred >= endpoint:
                break
            candidate = np.searchsorted(cumulative, np.random.random()*hazard, side="right")
            ch, band = channels[candidate], ranks[candidate]
            u, v = source[candidate], target[candidate]
            distance = distances[candidate]
            level = crowding[candidate]
            events[ev] = np.array((occurred, float(ch+1), float(u), float(v), distance, float(band), tr+half, float(level)))
            if u >= 0:
                event_positions[ev, 0] = points[u//2]
            else:
                event_positions[ev, 0] = np.nan
            event_positions[ev, 1] = points[v//2]
            ev += 1
            counts[ch, band, level] += 1
            if u < 0:
                initiators -= 1
            else:
                active[u] = False
                if ch < 7:
                    edges[ed, 0], edges[ed, 1] = u//2, v//2
                    ed += 1
                    _join(labels, u//2, v//2)
            if ch < 6:
                available[v] = False
                consumed[v//2] += 1
                active[v] = True
            else:
                active[v] = False
        forces, _ = spatial_forces(points[present], _compact_edges(edges[:ed], present), box,
                                   spring, rest, repulsion, core)
        present_points, displacement = brownian_step(
            points[present], forces, half, mobility[present], np.random.standard_normal(forces.shape), box)
        points[present] = present_points
        unwrapped[present] += displacement
        tr = endpoint
    return events[:ev], event_positions[:ev], counts, exposure, frames, frame_images, fed_frames, force_frames, energies


def simulate_environment(recipe: Recipe, motion: Motion, corridor, rates, seed, clock, points=None):
    """Run a mobile path with rates indexed by reaction, band, and occupancy."""
    clock = np.asarray(clock, dtype=float)
    rates = np.asarray(rates, dtype=float)
    n = recipe.aam+recipe.mbaa
    if not np.isfinite(corridor) or not 0. < corridor < .5*motion.box:
        raise ValueError("Invalid approach radius.")
    if (clock.ndim != 1 or len(clock) < 2 or clock[0] != 0. or clock[-1] != recipe.horizon
            or not np.isfinite(clock).all() or np.any(np.diff(clock) <= 0.)):
        raise ValueError("Clock must increase from zero to the registered horizon.")
    if rates.ndim != 3 or rates.shape[:2] != (8, 2) or not 2 <= rates.shape[2] <= 128 or not np.isfinite(rates).all() or np.any(rates < 0.):
        raise ValueError("Require finite nonnegative rates per channel, contact band and occupancy.")
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or not 0 <= seed < 2**32:
        raise ValueError("Expected a uint32 stream identifier.")
    if points is None:
        location = np.random.default_rng(np.random.SeedSequence([int(seed), 237263])).uniform(0., motion.box, (n, 3))
    else:
        location = np.asarray(points, dtype=float).copy()
        if location.shape != (n, 3) or not np.isfinite(location).all():
            raise ValueError("Invalid molecular positions.")
        location %= motion.box
    diffusion = np.full(n, motion.diffusion)
    diffusion[recipe.aam:] *= .8
    return _environment_path(recipe.aam, recipe.mbaa, recipe.initial_mbaa, recipe.feed_time, recipe.initiators,
                        recipe.horizon, clock, location, motion.step, diffusion, motion.box, motion.spring,
                        motion.rest, motion.repulsion, motion.core, motion.capture, motion.inner, corridor, rates, int(seed))
