"""Conditionally uniform continuous relocation of an event-centered aperture."""
from numba import njit
import numpy as np


@njit(cache=True)
def mobile_origin(points, i, j, box, radius, dither):
    origin = np.empty(3)
    for axis in range(3):
        delta = points[j, axis] - points[i, axis]
        delta -= box * np.rint(delta / box)
        middle = points[i, axis] + .5 * delta
        offset = radius + dither[axis] * (.70 * box - 2. * radius)
        origin[axis] = ((middle - offset) / box) % 1.
    return origin

