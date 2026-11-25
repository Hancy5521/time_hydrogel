"""Exact periodic short-range forces for the pre-gene Brownian solver."""

import numpy as np
from numba import njit

# formation_step locates the immutable physical kernels.
from formation_step import spatial_forces
from aam_spatial_motion import _distance


@njit(cache=True)
def cell_forces(points, edges, box, spring, rest, repulsion, core):
    """Evaluate only cells that can contain a repelling particle.

    Positions are in [0, box). Cell width is at least the repulsive cutoff.
    Sorting candidate indices preserves the reference summation order.
    Springs always follow the explicit bond list, regardless of distance.
    """
    side = min(32, int(box/core))
    if side < 4:
        return spatial_forces(points, edges, box, spring, rest, repulsion, core)
    heads = np.full((side, side, side), -1, dtype=np.int64)
    links = np.empty(len(points), dtype=np.int64)
    cells = np.empty((len(points), 3), dtype=np.int64)
    width = box/side
    for i in range(len(points)):
        for axis in range(3):
            cells[i, axis] = min(side-1, int(points[i, axis]/width))
        x, y, z = cells[i]
        links[i] = heads[x, y, z]
        heads[x, y, z] = i

    forces = np.zeros_like(points)
    energy = 0.
    nearby = np.empty(len(points), dtype=np.int64)
    for i in range(len(points)-1):
        x, y, z = cells[i]
        count = 0
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    j = heads[(x+dx)%side, (y+dy)%side, (z+dz)%side]
                    while j >= 0:
                        if j > i:
                            nearby[count] = j
                            count += 1
                        j = links[j]
        for k in range(1, count):
            particle = nearby[k]
            slot = k
            while slot > 0 and nearby[slot-1] > particle:
                nearby[slot] = nearby[slot-1]
                slot -= 1
            nearby[slot] = particle
        for j in nearby[:count]:
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
