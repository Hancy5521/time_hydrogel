"""Structural transport through the native periodic molecular bond graph."""

from __future__ import annotations

import numpy as np
from scipy.linalg import cho_factor, cho_solve


# Site identifiers are two consecutive slots per one coarse-grained molecule.
_BOND_TAGS = (3, 4, 5, 6, 7, 9)
_TIMES = (3.5, 4.85, 6.0)


def periodic_conductance(edges, shifts):
    """Return the unit-bond, unit-cell-drop conductance along x, y and z.

    An edge is oriented u→v with signed integer image shift k on v.
    For each axis this minimizes sum_e (phi_v - phi_u + k_e)**2.
    Parallel edges are allowed, including different image shifts. For
    nonwinding directions the output is exactly zero.
    """
    edges = np.asarray(edges)
    shifts = np.asarray(shifts)
    if edges.ndim != 2 or edges.shape[1] != 2 or shifts.shape != (len(edges), 3):
        raise ValueError("Expected a pair and three integer shifts per edge.")
    if not (np.isrealobj(edges) and np.isrealobj(shifts) and
            np.isfinite(edges).all() and np.isfinite(shifts).all()):
        raise ValueError("Edges and image shifts must be finite and real.")
    if (np.any(edges != np.rint(edges)) or np.any(shifts != np.rint(shifts))
            or np.any(edges < 0) or np.any(edges[:, 0] == edges[:, 1])):
        raise ValueError("Expected distinct nonnegative vertices and integer shifts.")
    if not len(edges):
        return np.zeros(3)

    vertices, endpoints = np.unique(edges.astype(np.int64).ravel(), return_inverse=True)
    shifts = shifts.astype(np.int64)
    endpoints = endpoints.reshape(-1, 2)
    start, stop = endpoints.T
    links = [[] for _ in vertices]
    for edge_id, (u, v) in enumerate(endpoints):
        links[u].append((v, edge_id, -1))
        links[v].append((u, edge_id, +1))

    # Use an exact integer gauge along a spanning tree of each component.
    # The remaining increments are zero on the trees and equal signed cycle
    # winding on the other edges. This avoids round-off false conduction.
    seen = np.zeros(len(vertices), dtype=bool)
    is_root = np.zeros_like(seen)
    gauge = np.zeros((len(vertices), 3), dtype=np.int64)
    for root in range(len(vertices)):
        if seen[root]:
            continue
        seen[root] = is_root[root] = True
        component = [root]
        for u in component:
            for v, edge_id, sign in links[u]:
                if not seen[v]:
                    seen[v] = True
                    gauge[v] = gauge[u] + sign * shifts[edge_id]
                    component.append(v)

    period = shifts + gauge[stop] - gauge[start]
    channels = np.any(period != 0, axis=0)
    output = np.zeros(3)
    if not channels.any():
        return output

    # Drop a root potential from each component so the Laplacian is SPD.
    # Compute the residual energy, rather than subtracting near-equal energies.
    fixed = period[:, channels].astype(float)
    laplacian = np.zeros((len(vertices), len(vertices)))
    rhs = np.zeros((len(vertices), fixed.shape[1]))
    for u, v in ((start, stop), (stop, start)):
        np.add.at(laplacian, (u, u), 1.)
        np.add.at(laplacian, (u, v), -1.)
    np.add.at(rhs, start, fixed)
    np.add.at(rhs, stop, -fixed)
    free = np.flatnonzero(~is_root)
    factor = cho_factor(laplacian[np.ix_(free, free)], lower=True, check_finite=False)
    potential = np.zeros_like(rhs)
    potential[free] = cho_solve(factor, rhs[free], check_finite=False)
    residual = fixed + potential[stop] - potential[start]
    output[channels] = np.square(residual).sum(axis=0)
    return output


def conductance_history(journal, times=_TIMES):
    """Compute right-continuous conductance from the native N×13 event rows."""
    journal = np.asarray(journal, dtype=float)
    times = np.asarray(times, dtype=float)
    if journal.ndim != 2 or journal.shape[1] != 13 or not np.isfinite(journal).all():
        raise ValueError("Expected a finite native N×13 event journal.")
    if times.ndim != 1 or not np.isfinite(times).all() or np.any(times < 0):
        raise ValueError("Inspection times must be finite nonnegative scalars.")
    clock, code = journal[:, :2].T
    discrete = journal[:, [1, 2, 3, 10, 11, 12]]
    if (np.any(np.diff(clock) < 0) or np.any(clock < 0)
            or np.any(discrete != np.rint(discrete))
            or not np.isin(code, np.arange(10)).all()):
        raise ValueError("Event timestamps or discrete fields violate the native schema.")

    is_bond = np.isin(code, _BOND_TAGS)
    sites = journal[is_bond, 2:4].astype(np.int64)
    if (sites < 0).any() or np.any(journal[~is_bond, 10:13] != 0):
        raise ValueError("Bond endpoints or non-bond shifts violate the native schema.")
    edges = sites // 2
    shifts = journal[is_bond, 10:13].astype(np.int64)
    limits = np.searchsorted(clock[is_bond], times, side="right")
    result = np.zeros((len(limits), 3))
    for count in np.unique(limits):
        result[limits == count] = periodic_conductance(edges[:count], shifts[:count])
    return result


def _check_fixtures():
    # A boundary-crossing tree can gauge its shifts to zero.
    e = np.array([[0, 1], [1, 2], [2, 3], [3, 0]])
    k = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0], [1, 0, 0]])
    np.testing.assert_array_equal(periodic_conductance(e[1:], k[1:]), np.zeros(3))
    # A once-wrapping m-edge ring has effective conductance 1/m.
    target = np.array([1/4, 0., 0.])
    np.testing.assert_allclose(periodic_conductance(e, k), target, atol=2e-14)
    offsets = np.array([[2, 1, 0], [-3, 4, 2], [1, -1, 0], [7, 3, 1]])
    np.testing.assert_allclose(
        periodic_conductance(e, k + offsets[e[:, 1]] - offsets[e[:, 0]]), target, atol=2e-14,
    )
    np.testing.assert_allclose(periodic_conductance(e[:, ::-1], -k), target, atol=2e-14)
    np.testing.assert_allclose(periodic_conductance(e + 90, k), target, atol=2e-14)
    # A parallel edge may produce stronger conduction. A disconnected
    # wrapping ring with the same orientation adds its conductance.
    added = periodic_conductance(np.r_[e, e[:1]], np.r_[k, k[:1]])
    assert (added >= target).all() and added[0] > target[0]
    np.testing.assert_allclose(periodic_conductance(np.r_[e, e+4], np.r_[k, k]), 2*target)
    ring_xyz = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0], [1, -2, 3]])
    np.testing.assert_allclose(periodic_conductance(e, ring_xyz), np.array([1., 4., 9.])/4.)
    # The journal counts only real bond events and includes bonds at exact read time.
    journal = np.zeros((6, 13))
    journal[:, 0] = (1., 2., 3., 4., 4.5, 5.)
    journal[:, 1] = (1, 3, 3, 3, 8, 7)
    journal[0, 2] = -1
    journal[1:4, 2:4] = 2*e[:3]
    journal[4, 2:4] = (0, 1)
    journal[5, 2:4] = 2*e[3]
    journal[5, 10:13] = k[3]
    got = conductance_history(journal, times=(0., 4.5, 5., 6.))
    np.testing.assert_allclose(got, np.r_[np.zeros((2, 3)), [target, target]], atol=2e-14)
    np.testing.assert_array_equal(conductance_history(np.zeros((0, 13))), np.zeros((3, 3)))
    print("Passed periodic tree, analytic rings, orientation, gauge, translation, "
          "edge addition, disconnected additivity and right-continuous journal fixtures.")


if __name__ == "__main__":
    _check_fixtures()
