"""Capture and branch at one matched radical/pendant event in the frozen reaction model."""

from typing import NamedTuple

import numpy as np
from numba import njit

from aam_contact_environment import corridor_occupancy
from aam_mbaa_kinetics import _join, _root
from aam_network_events import CAP, PENDANT_TAGS, factor_weights, pendant_kind
from aam_network_path_metrics import _add_edge, _component
from aam_spatial_motion import _compact_edges, _scalar_distance, brownian_step, spatial_forces


class State(NamedTuple):
    points: np.ndarray
    unwrapped: np.ndarray
    present: np.ndarray
    active: np.ndarray
    available: np.ndarray
    owner: np.ndarray
    labels: np.ndarray
    edges: np.ndarray
    lifts: np.ndarray
    events: np.ndarray
    periodic_parent: np.ndarray
    periodic_shift: np.ndarray
    periodic_rank: np.ndarray
    winding: np.ndarray
    # event count, edge count, output frame, initiators, lineage counter, reacted sites, midstep flag
    counts: np.ndarray
    # Brownian interval start, endpoint, chemical time, previous integrated primitive hazard
    position: np.ndarray


@njit(cache=True)
def candidates(state, aam, box, capture, inner, corridor, raw_rate, cache, present, rows, distance, primitive):
    number = 0
    active = np.flatnonzero(state.active)
    vacant = np.flatnonzero(state.available)
    initiators = state.counts[3]
    for v in vacant:
        host = v//2
        if initiators > 0 and state.owner[v ^ 1] < 0:
            tag = 1+int(host >= aam)
            rows[number] = np.array((tag, -1, v, 0, -1, 0))
            distance[number] = -1.
            primitive[number] = initiators*raw_rate[tag-1, 0]
            number += 1
    for u in active:
        for v in vacant:
            i, j = u//2, v//2
            if i == j:
                continue
            r = _scalar_distance(state.points, i, j, box)
            if r >= capture:
                continue
            band = int(r >= inner)
            kind, occupancy = -1, 0
            tag = 3 if j < aam else 4
            rate_row = tag-1
            if state.owner[v ^ 1] >= 0:
                kind = pendant_kind(state.labels, state.owner, u, v)
                tag = PENDANT_TAGS[kind]
                rate_row = 4 if kind == 0 else 5
                occupancy = corridor_occupancy(state.points, cache, present, i, j, box, corridor, CAP)
            rows[number] = np.array((tag, u, v, band, kind, occupancy))
            distance[number], primitive[number] = r, raw_rate[rate_row, band]
            number += 1
    for i in range(len(active)-1):
        u = active[i]
        for v in active[i+1:]:
            r = _scalar_distance(state.points, u//2, v//2, box)
            if r >= capture:
                continue
            band = int(r >= inner)
            for rate_row in range(6, 8):
                if rate_row == 6 and u//2 == v//2:
                    continue
                rows[number] = np.array((rate_row+1, u, v, band, -1, 0))
                distance[number], primitive[number] = r, raw_rate[rate_row, band]
                number += 1
    return number


@njit(cache=True)
def edge_lift(state, i, j, box):
    return -np.rint((state.unwrapped[j]-state.unwrapped[i])/box).astype(np.int64)


@njit(cache=True)
def apply_event(state, row, distance, box, time, midpoint):
    tag, u, v, band, kind, occupied = row
    source_lineage = state.owner[u] if u >= 0 else -1
    target_lineage = state.owner[v ^ 1] if kind >= 0 else state.owner[v]
    shift = np.zeros(3, np.int64)
    ev, ed = state.counts[:2]
    if u < 0:
        state.counts[3] -= 1
        lineage = state.counts[4]
        state.counts[4] += 1
    else:
        lineage = source_lineage
        state.active[u] = False
        if tag != 8:
            i, j = u//2, v//2
            shift = edge_lift(state, i, j, box)
            state.edges[ed, 0], state.edges[ed, 1] = i, j
            state.lifts[ed] = shift
            _join(state.labels, i, j)
            _add_edge(state.periodic_parent, state.periodic_shift, state.periodic_rank, state.winding, i, j, shift)
            state.counts[1] += 1
    if tag != 7 and tag != 8:
        state.owner[v] = lineage
        state.available[v] = False
        state.active[v] = True
        state.counts[5] += 1
    else:
        state.active[v] = False
    state.events[ev] = np.array((time, float(tag), float(u), float(v), distance, float(band),
                                 midpoint, float(occupied), float(source_lineage), float(target_lineage),
                                 float(shift[0]), float(shift[1]), float(shift[2])))
    state.counts[0] += 1


@njit(cache=True)
def add_feed(state, start, time):
    for j in range(start, len(state.points)):
        state.present[j] = True
        state.available[2*j:2*j+2] = True
        state.events[state.counts[0]] = np.array((time, 0., -1., float(j), -1., -1., time,
                                                 -1., -1., -1., 0., 0., 0.))
        state.counts[0] += 1


@njit(cache=True)
def move(state, half, mobility, box, spring, rest, repulsion, core):
    present = state.present
    forces, _ = spatial_forces(state.points[present], _compact_edges(state.edges[:state.counts[1]], present),
                               box, spring, rest, repulsion, core)
    positions, displacement = brownian_step(state.points[present], forces, half, mobility[present],
                                            np.random.standard_normal(forces.shape), box)
    state.points[present] = positions
    state.unwrapped[present] += displacement


@njit(cache=True)
def graph_hops(state, source):
    distance = np.full(len(state.points), -1, np.int64)
    queue = np.empty(len(state.points), np.int64)
    queue[0] = source
    distance[source] = 0
    read, end = 0, 1
    while read < end:
        u = queue[read]
        read += 1
        for a, b in state.edges[:state.counts[1]]:
            v = b if a == u else (a if b == u else -1)
            if v >= 0 and distance[v] < 0:
                distance[v] = distance[u]+1
                queue[end] = v
                end += 1
    return distance


@njit(cache=True)
def matched_pair(state, rows, distances, count, chosen, box, mode):
    tag, u, _, band, _, occupied = rows[chosen]
    if mode == 1 and tag != 5:
        return -1, -1, -1, -1
    if mode == 2 and tag != 6 and tag != 9:
        return -1, -1, -1, -1
    if mode == 1:
        sizes = np.zeros(len(state.points), np.int64)
        for i in range(len(sizes)):
            sizes[_root(state.labels, i)] += 1
    else:
        hops = graph_hops(state, u//2)
        _, source_lift = _component(state.periodic_parent, state.periodic_shift, u//2)
    low = high = -1
    minimum, maximum = 1000000, -1
    for k in range(count):
        ch, u2, v, band2, _, occupancy = rows[k]
        if ch != tag or u2 != u or band2 != band or occupancy != occupied:
            continue
        if mode == 1:
            value = sizes[_root(state.labels, v//2)]
        else:
            _, target_lift = _component(state.periodic_parent, state.periodic_shift, v//2)
            # A winding completed by the first forced edge would trivially determine the endpoint.
            if np.any(edge_lift(state, u//2, v//2, box)+source_lift-target_lift != 0):
                continue
            value = hops[v//2]+1
        if value < minimum:
            low, minimum = k, value
        if value > maximum:
            high, maximum = k, value
    difference = 2 if mode == 1 else 3
    if maximum-minimum < difference:
        return -1, -1, -1, -1
    return low, high, minimum, maximum


@njit(cache=True)
def evolve(state, clock, material, geometry, rates, factors, clock_factor, seed, mode):
    np.random.seed(seed)
    aam, mbaa, initial = int(material[0]), int(material[1]), int(material[2])
    pulse = material[3]
    dt, mobile, box, spring, rest, repulsion, core, capture, inner, corridor = geometry
    mobility = np.full(len(state.points), mobile)
    mobility[aam:] *= .8
    width = 2*len(state.points)
    length = width*width+2*width
    rows = np.empty((length, 6), np.int64)
    distance = np.empty(length)
    primitive = np.empty(length)
    weights = np.empty(length)
    risk = np.empty((3, CAP+1))
    found = np.zeros((2, 6), np.int64)
    diagnostics = np.full(10, np.nan)
    horizon = clock[-1]
    while state.position[0] < horizon:
        if state.counts[6] == 0:
            tr = state.position[0]
            fed = state.present.all()
            if not fed and tr == pulse:
                add_feed(state, aam+initial, tr)
                fed = True
            if tr == clock[state.counts[2]]:
                state.counts[2] += 1
            endpoint = min(tr+dt, clock[state.counts[2]])
            if not fed:
                endpoint = min(endpoint, pulse)
            state.position[1] = endpoint
            state.position[2] = tr
            state.counts[6] = 1
            move(state, .5*(endpoint-tr), mobility, box, spring, rest, repulsion, core)
        tr, endpoint = state.position[:2]
        half = .5*(endpoint-tr)
        cache = np.full((len(state.points), len(state.points)), -1, np.int8)
        present = np.flatnonzero(state.present)
        while state.position[2] < endpoint:
            n = candidates(state, aam, box, capture, inner, corridor, rates, cache, present, rows, distance, primitive)
            hazard = np.sum(primitive[:n])
            factor_weights(primitive, rows[:, 4], rows[:, 5], n, factors, weights, risk)
            cumulative = np.cumsum(weights[:n])
            local = state.position[2]
            reacted = local+np.random.exponential()/(hazard*clock_factor) if hazard > 0. else np.inf
            end = min(reacted, endpoint)
            state.position[3] += (end-local)*hazard
            state.position[2] = end
            if reacted >= endpoint:
                break
            chosen = np.searchsorted(cumulative, np.random.random()*cumulative[-1], side="right")
            conversion = state.counts[5]/(aam+2*mbaa)
            if mode and .15 <= conversion <= .64 and not state.winding.any():
                lo, hi, a, b = matched_pair(state, rows, distance, n, chosen, box, mode)
                if lo >= 0:
                    found[0], found[1] = rows[lo], rows[hi]
                    diagnostics[:] = np.array((reacted, tr+half, hazard*clock_factor, n,
                                               distance[lo], distance[hi], a, b, weights[lo], weights[hi]))
                    return found, diagnostics
            apply_event(state, rows[chosen], distance[chosen], box, reacted, tr+half)
        move(state, half, mobility, box, spring, rest, repulsion, core)
        state.position[0] = endpoint
        state.counts[6] = 0
    if not state.present.all() and state.position[0] == pulse:
        add_feed(state, aam+initial, pulse)
    return found, diagnostics


def setup(recipe, motion, seed, corridor):
    points = np.random.default_rng(np.random.SeedSequence([int(seed), 237263])).uniform(
        0., motion.box, (recipe.aam+recipe.mbaa, 3))
    n = len(points)
    slots = recipe.aam+2*recipe.mbaa+recipe.initiators//2+recipe.mbaa+1
    present = np.arange(n) < recipe.aam+recipe.initial_mbaa
    available = np.zeros(2*n, bool)
    available[:2*recipe.aam:2] = True
    available[2*recipe.aam:2*(recipe.aam+recipe.initial_mbaa)] = True
    state = State(points, points.copy(), present, np.zeros(2*n, bool), available,
                  np.full(2*n, -1, np.int64), np.arange(n), np.zeros((slots, 2), np.int64),
                  np.zeros((slots, 3), np.int64), np.zeros((slots, 13)),
                  np.arange(n), np.zeros((n, 3), np.int64), np.zeros(n, np.int64),
                  np.zeros(3, bool), np.asarray([0, 0, 0, recipe.initiators, 0, 0, 0], np.int64), np.zeros(4))
    material = np.array((recipe.aam, recipe.mbaa, recipe.initial_mbaa, recipe.feed_time))
    settings = np.array((motion.step, motion.diffusion, motion.box, motion.spring, motion.rest,
                         motion.repulsion, motion.core, motion.capture, motion.inner, corridor))
    return state, material, settings


def continue_from(state, row, distance, seed, material, settings, clock, rates, factors, scale):
    branch = State(*(field.copy() for field in state))
    start, end, time = branch.position[:3]
    apply_event(branch, row, distance, settings[2], time, .5*(start+end))
    evolve(branch, clock, material, settings, rates, factors, scale, seed, 0)
    return branch

