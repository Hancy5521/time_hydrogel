"""Separate the reaction clock from pendant-bond choices in mobile hydrogel growth."""

import numpy as np
from numba import njit

from aam_contact_environment import corridor_occupancy
from aam_mbaa_kinetics import Recipe, _join, _root
from aam_spatial_motion import Motion, _compact_edges, _scalar_distance, brownian_step, spatial_forces


PENDANT_TAGS = (5, 6, 9)
CAP = 4
EVENT_COLUMNS = ("time", "tag", "source_site", "target_site_or_feed_molecule", "distance", "band",
                 "geometry_time", "occupancy", "source_kinetic_chain", "partner_kinetic_chain",
                 "bond_image_x", "bond_image_y", "bond_image_z")


@njit(cache=True)
def pendant_kind(labels, owner, u, v):
    if _root(labels, u//2) != _root(labels, v//2):
        return 0
    return 1 if owner[u] == owner[v ^ 1] else 2


@njit(cache=True)
def factor_weights(base, kind, context, count, factors, scaled, risk):
    """Change pendant choice without changing the global or pendant event rate."""
    risk[:, :] = 0.
    raw_pendant = 0.
    selected_pendant = 0.
    for i in range(count):
        mark = kind[i]
        weight = base[i]
        scaled[i] = weight
        if mark >= 0:
            local = context[i]
            risk[mark, local] += weight
            raw_pendant += weight
            scaled[i] *= factors[mark, local]
            selected_pendant += scaled[i]
    if raw_pendant > 0.:
        correction = raw_pendant/selected_pendant
        for i in range(count):
            if kind[i] >= 0:
                scaled[i] *= correction


@njit(cache=True)
def _simulate(aam, mbaa, initial, pulse, initiators, horizon, clock, points, dt, mobility, box, spring, rest,
              repulsion, core, radius, inner, corridor, raw_rate, factors, clock_factor, seed):
    np.random.seed(seed)
    n = len(points)
    size = 2*n
    active = np.zeros(size, np.bool_)
    available = np.zeros(size, np.bool_)
    available[:2*aam:2] = True
    available[2*aam:2*(aam+initial)] = True
    owner = np.full(size, -1, np.int64)
    present = np.arange(n) < aam+initial
    labels = np.arange(n)
    images = points.copy()
    max_events = aam+2*mbaa+initiators//2+mbaa+1
    events = np.empty((max_events, 13))
    edges = np.empty((max_events, 2), np.int64)
    event_hazard = np.empty(max_events)
    risk_at_event = np.zeros((max_events, 3, CAP+1))
    max_choices = size*size+2*size
    choices = np.empty((max_choices, 6), np.int64)
    distance = np.empty(max_choices)
    primitive = np.empty(max_choices)
    weighted = np.empty(max_choices)
    risk = np.empty((3, CAP+1))
    tr = 0.
    ev = ed = frame = chain_id = 0
    integrated_base = 0.
    fed = initial == mbaa
    while tr < horizon:
        if not fed and tr == pulse:
            for molecule in range(aam+initial, n):
                present[molecule] = True
                available[2*molecule:2*molecule+2] = True
                events[ev] = np.array((tr, 0., -1., float(molecule), -1., -1., tr, -1., -1., -1., 0., 0., 0.))
                event_hazard[ev] = 0.
                ev += 1
            fed = True
        if tr == clock[frame]:
            frame += 1
        forces, _ = spatial_forces(points[present], _compact_edges(edges[:ed], present),
                                   box, spring, rest, repulsion, core)
        endpoint = min(tr+dt, clock[frame])
        if not fed:
            endpoint = min(endpoint, pulse)
        half = .5*(endpoint-tr)
        moved, disp = brownian_step(points[present], forces, half, mobility[present],
                                    np.random.standard_normal(forces.shape), box)
        points[present] = moved
        images[present] += disp
        cache = np.full((n, n), -1, np.int8)
        neighbors = np.flatnonzero(present)
        local = tr
        while local < endpoint:
            count = 0
            radicals = np.flatnonzero(active)
            vacant = np.flatnonzero(available)
            for v in vacant:
                vh = v//2
                if initiators > 0 and owner[v ^ 1] < 0:
                    ch = int(vh >= aam)
                    choices[count] = np.array((ch+1, -1, v, 0, -1, 0))
                    distance[count] = -1.
                    primitive[count] = initiators*raw_rate[ch, 0]
                    count += 1
            for u in radicals:
                uh = u//2
                for v in vacant:
                    vh = v//2
                    if uh == vh:
                        continue
                    r = _scalar_distance(points, uh, vh, box)
                    if r >= radius:
                        continue
                    band = int(r >= inner)
                    kind = -1
                    occupancy = 0
                    tag = 3 if vh < aam else 4
                    index = tag-1
                    if owner[v ^ 1] >= 0:
                        kind = pendant_kind(labels, owner, u, v)
                        tag = PENDANT_TAGS[kind]
                        index = 4 if kind == 0 else 5
                        occupancy = corridor_occupancy(points, cache, neighbors, uh, vh, box, corridor, CAP)
                    choices[count] = np.array((tag, u, v, band, kind, occupancy))
                    distance[count] = r
                    primitive[count] = raw_rate[index, band]
                    count += 1
            for i in range(len(radicals)-1):
                u = radicals[i]
                for v in radicals[i+1:]:
                    r = _scalar_distance(points, u//2, v//2, box)
                    if r >= radius:
                        continue
                    band = int(r >= inner)
                    for ch in range(6, 8):
                        if ch == 6 and u//2 == v//2:
                            continue
                        choices[count] = np.array((ch+1, u, v, band, -1, 0))
                        distance[count] = r
                        primitive[count] = raw_rate[ch, band]
                        count += 1
            # Keep the same total exactly even when floating summation of normalized marks differs.
            hazard = np.sum(primitive[:count])
            factor_weights(primitive, choices[:, 4], choices[:, 5], count, factors, weighted, risk)
            cumulative = np.cumsum(weighted[:count])
            occurred = local+np.random.exponential()/(hazard*clock_factor) if hazard > 0. else np.inf
            end = min(occurred, endpoint)
            integrated_base += (end-local)*hazard
            local = end
            if occurred >= endpoint:
                break
            candidate = np.searchsorted(cumulative, np.random.random()*cumulative[-1], side="right")
            tag, u, v, band, kind, occupancy = choices[candidate]
            u_chain = owner[u] if u >= 0 else -1
            other = owner[v ^ 1] if kind >= 0 else owner[v]
            image = np.zeros(3)
            if u < 0:
                initiators -= 1
                new_owner = chain_id
                chain_id += 1
            else:
                new_owner = u_chain
                active[u] = False
                if tag != 8:
                    image = -np.rint((images[v//2]-images[u//2])/box)
                    edges[ed, 0], edges[ed, 1] = u//2, v//2
                    ed += 1
                    _join(labels, u//2, v//2)
            if tag != 7 and tag != 8:
                owner[v] = new_owner
                available[v] = False
                active[v] = True
            else:
                active[v] = False
            events[ev] = np.array((occurred, float(tag), float(u), float(v), distance[candidate],
                                   float(band), tr+half, float(occupancy), float(u_chain), float(other),
                                   image[0], image[1], image[2]))
            risk_at_event[ev] = risk
            event_hazard[ev] = hazard
            ev += 1
        forces, _ = spatial_forces(points[present], _compact_edges(edges[:ed], present),
                                   box, spring, rest, repulsion, core)
        moved, disp = brownian_step(points[present], forces, half, mobility[present],
                                    np.random.standard_normal(forces.shape), box)
        points[present] = moved
        images[present] += disp
        tr = endpoint
    # Preserve a right-continuous feed at the horizon, as in the previous solver.
    if not fed and tr == pulse:
        for molecule in range(aam+initial, n):
            events[ev] = np.array((tr, 0., -1., float(molecule), -1., -1., tr, -1., -1., -1., 0., 0., 0.))
            event_hazard[ev] = 0.
            ev += 1
    return events[:ev], risk_at_event[:ev], event_hazard[:ev], integrated_base


def simulate(recipe: Recipe, motion: Motion, raw_rate, factors, clock_factor, seed, clock, corridor=.82):
    """Return reaction history, exact pendant opportunities at events, and clock exposure."""
    clock = np.asarray(clock, dtype=float)
    raw_rate, factors = np.asarray(raw_rate, dtype=float), np.asarray(factors, dtype=float)
    if (clock.ndim != 1 or len(clock) < 2 or clock[0] != 0. or clock[-1] != recipe.horizon
            or not np.isfinite(clock).all() or np.any(np.diff(clock) <= 0.)):
        raise ValueError("Use a finite increasing output clock and a strictly timed batch or pulse.")
    if (raw_rate.shape != (8, 2) or factors.shape != (3, CAP+1) or not np.isfinite(raw_rate).all()
            or np.any(raw_rate < 0.) or not np.isfinite(factors).all() or not (0. < factors.min() <= factors.max() < 1.e6)
            or not np.isfinite(clock_factor) or clock_factor <= 0. or not 0. < corridor < .5*motion.box):
        raise ValueError("Require valid clock and candidate weights.")
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or not 0 <= seed < 2**32:
        raise ValueError("Expected a uint32 stream identifier.")
    n = recipe.aam+recipe.mbaa
    locations = np.random.default_rng(np.random.SeedSequence([int(seed), 237263])).uniform(0., motion.box, (n, 3))
    diffusion = np.full(n, motion.diffusion)
    diffusion[recipe.aam:] *= .8
    return _simulate(recipe.aam, recipe.mbaa, recipe.initial_mbaa, recipe.feed_time, recipe.initiators, recipe.horizon,
                     clock, locations, motion.step, diffusion, motion.box, motion.spring, motion.rest, motion.repulsion,
                     motion.core, motion.capture, motion.inner, corridor, raw_rate, factors, clock_factor, int(seed))

