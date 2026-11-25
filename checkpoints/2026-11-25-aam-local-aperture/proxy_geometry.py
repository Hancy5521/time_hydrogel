"""Read only certified geometry and local candidate weights at qualified reactions."""
from __future__ import annotations

import numpy as np
from numba import njit


@njit(cache=True)
def root(parent, i):
    while parent[i] != i:
        i = parent[i]
    return i


@njit(cache=True)
def inside_corridor(x, i, j, box, radius):
    for ax in range(3):
        step = x[j, ax] - x[i, ax]
        step -= box * np.rint(step / box)
        center = x[i, ax] + .5 * step
        if not radius < center < .70 * box - radius:
            return False
    return True


@njit(cache=True)
def crowd(x, i, j, seen, box, radius):
    mid = np.empty(3)
    for axis in range(3):
        d = x[j, axis] - x[i, axis]
        d -= box * np.rint(d / box)
        mid[axis] = x[i, axis] + .5 * d
    neighbors = 0
    for q in range(len(seen)):
        if not seen[q] or q == i or q == j:
            continue
        d2 = 0.
        for axis in range(3):
            d = x[q, axis] - mid[axis]
            d -= box * np.rint(d / box)
            d2 += d * d
        if d2 < radius * radius:
            neighbors += 1
    return min(neighbors, 4)



@njit(cache=True)
def proxy(state, source, target, box, radius, capture, inner, origin, reference, saved, count, time):
    if count[0] >= len(saved):
        return
    x = np.empty_like(state.points)
    seen = state.present.copy()
    for node in range(len(seen)):
        for axis in range(3):
            x[node, axis] = (state.points[node, axis] - origin[axis] * box) % box
            if x[node, axis] >= .70 * box:
                seen[node] = False
    if not (seen[source] and seen[target] and inside_corridor(x, source, target, box, radius)):
        return
    parent = np.arange(len(seen))
    for u, v in state.edges[:state.counts[1]]:
        if seen[u] and seen[v]:
            parent[root(parent, v)] = root(parent, u)
    first, second = root(parent, source), root(parent, target)
    open_a, open_b = False, False
    for u, v in state.edges[:state.counts[1]]:
        if seen[v] and not seen[u]:
            u, v = v, u
        if seen[u] and not seen[v]:
            k = root(parent, u)
            open_a = open_a or k == first
            open_b = open_b or k == second
    bridge = first != second
    if bridge and open_a and open_b:
        return
    q_observed = crowd(x, source, target, seen, box, radius)
    observed_group = 2 * int(q_observed > 0) + int(bridge)
    component = np.empty(len(seen), dtype=np.int64)
    boundary = np.zeros(len(seen), dtype=np.bool_)
    for k in range(len(seen)):
        component[k] = root(parent, k)
    for u, v in state.edges[:state.counts[1]]:
        if seen[u] and not seen[v]:
            boundary[component[u]] = True
        elif seen[v] and not seen[u]:
            boundary[component[v]] = True
    degree = np.zeros(len(seen), dtype=np.int64)
    for i in range(len(seen)):
        if seen[i]:
            degree[i] = int(state.owner[2*i] >= 0) + int(state.owner[2*i + 1] >= 0)
    risk = np.zeros(4)
    for i in range(len(seen)):
        if not seen[i] or degree[i] < 1:
            continue
        for j in range(64, len(seen)):
            if not seen[j] or i == j or degree[j] != 1:
                continue
            x2 = 0.
            for ax in range(3):
                df = x[j, ax] - x[i, ax]
                df -= box * np.rint(df / box)
                x2 += df * df
            if x2 >= capture * capture or not inside_corridor(x, i, j, box, radius):
                continue
            left, right = component[i], component[j]
            is_bridge = left != right
            if is_bridge and boundary[left] and boundary[right]:
                continue
            q = crowd(x, i, j, seen, box, radius)
            group = 2 * int(q > 0) + int(is_bridge)
            band = int(x2 >= inner * inner)
            if is_bridge:
                risk[group] += reference[0, q, band]
            else:
                risk[group] += .5 * (reference[1, q, band] + reference[2, q, band])
    if risk[observed_group] <= 0.:
        return
    if int(risk[0] + risk[1] > 0.) + int(risk[2] > 0.) + int(risk[3] > 0.) < 2:
        return
    slot = count[0]
    saved[slot, :4] = risk
    saved[slot, 4] = observed_group
    saved[slot, 5] = time
    count[0] += 1
