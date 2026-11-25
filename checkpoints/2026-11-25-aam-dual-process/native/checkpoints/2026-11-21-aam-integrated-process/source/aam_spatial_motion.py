"""Mobile AAm/MBAA-inspired beads and explicit vinyl reaction events.

Model units use kBT=1. The contact kinetics are integrated exactly conditional
on each frozen half-step geometry; Brownian integration and splitting are
discretized. This model has implicit solvent and prescribed effective forces.
"""

from dataclasses import dataclass

import numpy as np
from numba import njit

from aam_mbaa_kinetics import Recipe, _join, _root


@dataclass(frozen=True)
class Motion:
    box: float = 6.
    step: float = .025
    diffusion: float = .26
    spring: float = 9.
    rest: float = .67
    repulsion: float = 16.
    core: float = .55
    capture: float = 1.42
    inner: float = .96

    def __post_init__(self):
        values = np.array(tuple(vars(self).values()), dtype=float)
        if (not np.isfinite(values).all() or np.any(values < 0.)
                or not (0. < self.inner < self.capture < .5*self.box)
                or self.step <= 0. or not 0. < self.core < .5*self.box):
            raise ValueError("Invalid Brownian solver scales.")


@njit(cache=True, inline="always")
def _distance(a, b, box):
    x = b[0]-a[0]
    y = b[1]-a[1]
    z = b[2]-a[2]
    x -= box*np.rint(x/box)
    y -= box*np.rint(y/box)
    z -= box*np.rint(z/box)
    return (x, y, z), np.sqrt(x*x+y*y+z*z)


@njit(cache=True, inline="always")
def _scalar_distance(points, i, j, box):
    squared = 0.
    for k in range(3):
        displacement = points[j, k]-points[i, k]
        displacement -= box*np.rint(displacement/box)
        squared += displacement**2
    return np.sqrt(squared)


@njit(cache=True)
def spatial_forces(points, edges, box, spring, rest, repulsion, core):
    forces = np.zeros_like(points)
    energy = 0.
    for i in range(len(points)-1):
        for j in range(i+1, len(points)):
            delta, r = _distance(points[i], points[j], box)
            if r < core:
                energy += .5*repulsion*(r-core)**2
                if r > 0.:
                    scale = repulsion*(r-core)/r
                    for k in range(3):
                        force = scale*delta[k]
                        forces[i, k] += force
                        forces[j, k] -= force
    for i, j in edges:
        delta, r = _distance(points[i], points[j], box)
        energy += .5*spring*(r-rest)**2
        if r > 0.:
            scale = spring*(r-rest)/r
            for k in range(3):
                force = scale*delta[k]
                forces[i, k] += force
                forces[j, k] -= force
    return forces, energy


@njit(cache=True)
def brownian_step(points, forces, dt, diffusion, noise, box):
    displacement = dt*diffusion.reshape((-1, 1))*forces + np.sqrt(2.*dt*diffusion).reshape((-1, 1))*noise
    return (points+displacement) % box, displacement


@njit(cache=True)
def _mobile_path(aam, mbaa, initial, pulse, initiators, horizon, clock, points, dt,
                 mobility, box, spring, rest, repulsion, core, radius, inner, rate, seed):
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
    events = np.empty((max_events, 7))
    edges = np.empty((max_events, 2), np.int64)
    event_positions = np.empty((max_events, 2, 3))
    frames = np.empty((len(clock), n, 3))
    frame_images = np.empty_like(frames)
    fed_frames = np.empty((len(clock), n), np.bool_)
    force_frames = np.empty_like(frames)
    energies = np.empty(len(clock))
    counts = np.zeros((8, 2), np.int64)
    exposure = np.zeros((8, 2))
    channels = np.empty(all_sites*all_sites+2*all_sites, np.int64)
    ranks = np.empty_like(channels)
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
                events[ev] = np.array((tr, 0., -1., float(molecule), -1., -1., tr))
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
        local = tr
        while local < endpoint:
            choices = 0
            opportunity = np.zeros((8, 2))
            ra = np.flatnonzero(active)
            vacant = np.flatnonzero(available)
            for v in vacant:
                vh = v//2
                if initiators > 0 and consumed[vh] == 0:
                    ch = int(vh >= aam)
                    source[choices], target[choices] = -1, v
                    channels[choices], ranks[choices], distances[choices] = ch, 0, -1.
                    weights[choices] = initiators*rate[ch, 0]
                    opportunity[ch, 0] += initiators
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
                    opportunity[ch, band] += 1.
                    weights[choices] = rate[ch, band]
                    choices += 1
            for i in range(len(ra)-1):
                u = ra[i]
                for v in ra[i+1:]:
                    r = _scalar_distance(points, u//2, v//2, box)
                    if r >= radius:
                        continue
                    band = int(r >= inner)
                    for ch in range(6, 8):
                        if ch == 6 and u//2 == v//2:
                            continue
                        source[choices], target[choices], channels[choices] = u, v, ch
                        ranks[choices], distances[choices] = band, r
                        opportunity[ch, band] += 1.
                        weights[choices] = rate[ch, band]
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
            events[ev] = np.array((occurred, float(ch+1), float(u), float(v), distance, float(band), tr+half))
            if u >= 0:
                event_positions[ev, 0] = points[u//2]
            else:
                event_positions[ev, 0] = np.nan
            event_positions[ev, 1] = points[v//2]
            ev += 1
            counts[ch, band] += 1
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


@njit(cache=True)
def _compact_edges(edges, present):
    compact_id = np.cumsum(present.astype(np.int64))-1
    result = np.empty_like(edges)
    for i in range(len(edges)):
        result[i, 0] = compact_id[edges[i, 0]]
        result[i, 1] = compact_id[edges[i, 1]]
    return result


def simulate_mobile(recipe: Recipe, motion: Motion, rates, seed, clock, points=None):
    clock = np.asarray(clock, dtype=float)
    rates = np.asarray(rates, dtype=float)
    n = recipe.aam+recipe.mbaa
    if (clock.ndim != 1 or len(clock) < 2 or clock[0] != 0. or clock[-1] != recipe.horizon
            or not np.isfinite(clock).all() or np.any(np.diff(clock) <= 0.)):
        raise ValueError("Clock must increase from zero to the registered horizon.")
    if rates.shape != (8, 2) or not np.isfinite(rates).all() or np.any(rates < 0.):
        raise ValueError("Require finite nonnegative rates per channel and contact band.")
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
    return _mobile_path(recipe.aam, recipe.mbaa, recipe.initial_mbaa, recipe.feed_time, recipe.initiators,
                        recipe.horizon, clock, location, motion.step, diffusion, motion.box, motion.spring,
                        motion.rest, motion.repulsion, motion.core, motion.capture, motion.inner, rates, int(seed))


def learn_contact_rates(count, exposure):
    """Fit contact-band rates and a baseline that pools over contact distances."""
    if np.any(count.sum(axis=-1) <= 0) or np.any(exposure.sum(axis=-1) <= 0):
        raise ValueError("Every reaction type requires training support.")
    pooled = count.sum(axis=-1)/exposure.sum(axis=-1)
    informed = np.zeros_like(exposure)
    np.divide(count, exposure, out=informed, where=exposure > 0.)
    informed[:2, 1] = informed[:2, 0]
    return np.array((informed, np.broadcast_to(pooled[:, None], informed.shape)))
