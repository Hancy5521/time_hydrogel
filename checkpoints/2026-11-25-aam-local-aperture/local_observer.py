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
def observe(state, rows, primitive, chosen, count, box, radius, origin, reference, saved, cursor, time):
    if time < 2.05 or time > 4.85 or cursor[0] >= len(saved) or rows[chosen, 4] < 0:
        return
    n = len(state.present)
    x = np.empty((n, 3))
    seen = state.present.copy()
    for i in range(n):
        for ax in range(3):
            x[i, ax] = (state.points[i, ax] - origin[ax] * box) % box
            if x[i, ax] >= .70 * box:
                seen[i] = False
    i, j = rows[chosen, 1] // 2, rows[chosen, 2] // 2
    if not (seen[i] and seen[j] and inside_corridor(x, i, j, box, radius)):
        return
    parent = np.arange(n)
    for u, v in state.edges[:state.counts[1]]:
        if seen[u] and seen[v]:
            parent[root(parent, v)] = root(parent, u)
    incomplete = np.zeros(n, dtype=np.bool_)
    for u, v in state.edges[:state.counts[1]]:
        if seen[u] and not seen[v]:
            incomplete[root(parent, u)] = True
        if seen[v] and not seen[u]:
            incomplete[root(parent, v)] = True
    risk = np.zeros(4)
    selected_group = -1
    for k in range(count):
        native_kind = rows[k, 4]
        if native_kind < 0 or primitive[k] <= 0.:
            continue
        i, j = rows[k, 1] // 2, rows[k, 2] // 2
        if not (seen[i] and seen[j] and inside_corridor(x, i, j, box, radius)):
            continue
        a, b = root(parent, i), root(parent, j)
        if a == b:
            bridge = False
        elif not incomplete[a] or not incomplete[b]:
            bridge = True
        else:
            continue
        local_count = crowd(x, i, j, seen, box, radius)
        occupied = local_count > 0
        # Only the certified class and visible occupied ball enter the slope grouping.
        # This reference assay assumes calibrated candidate-lineage weights.
        if bridge != (native_kind == 0) or local_count != rows[k, 5]:
            raise AssertionError("Local certificate is inconsistent with hidden ground truth.")
        group = 2 * int(occupied) + int(bridge)
        risk[group] += reference[native_kind, local_count, rows[k, 3]]
        if k == chosen:
            selected_group = group
    if selected_group < 0:
        return
    nonzero_risk = int(risk[0] + risk[1] > 0.) + int(risk[2] > 0.) + int(risk[3] > 0.)
    if nonzero_risk < 2:
        return
    pointer = cursor[0]
    saved[pointer, :4] = risk
    saved[pointer, 4] = selected_group
    saved[pointer, 5] = time
    cursor[0] += 1

