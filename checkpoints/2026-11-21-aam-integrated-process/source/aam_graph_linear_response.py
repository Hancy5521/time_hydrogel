"""Dimensionless central-force tangent response at a frozen Brownian graph."""
from __future__ import annotations

import numpy as np
from scipy.linalg import eigh


def minimum_image(points, edges, box):
    vectors = points[edges[:, 1]] - points[edges[:, 0]]
    return vectors - box * np.rint(vectors / box)


def rigidity_matrix(points, edges, box):
    n = len(points)
    d = minimum_image(points, edges, box)
    lengths = np.linalg.norm(d, axis=1)
    if np.any(lengths < 1.e-10):
        raise ValueError("Coincident bonded vertices lack an unambiguous tangent")
    unit = d / lengths[:, None]
    b = np.zeros((len(edges), 3*n))
    for row, ((i, j), axis) in enumerate(zip(edges, unit, strict=True)):
        b[row, 3*i:3*i+3] = -axis
        b[row, 3*j:3*j+3] = axis
    return b, d, lengths


def prestressed_hessian(points, edges, box, spring, rest, repulsion, core):
    """Exact Hessian of the simulator's current minimum-image pair energy."""
    n = len(points)
    h = np.zeros((3*n, 3*n))
    def contribute(i, j, strength, target):
        r = points[j] - points[i]
        r -= box * np.rint(r / box)
        length = np.linalg.norm(r)
        if length < 1.e-10:
            raise ValueError("Coincident pair cannot define a radial Hessian")
        axis = r / length
        geometric = strength * (1 - target / length)
        element = (strength-geometric) * np.outer(axis, axis) + geometric * np.eye(3)
        ii, jj = slice(3*i, 3*i+3), slice(3*j, 3*j+3)
        h[ii, ii] += element
        h[jj, jj] += element
        h[ii, jj] -= element
        h[jj, ii] -= element
    for i, j in edges:
        contribute(int(i), int(j), spring, rest)
    for i in range(n-1):
        for j in range(i+1, n):
            d = points[j]-points[i]
            d -= box*np.rint(d/box)
            if d @ d < core**2:
                contribute(i, j, repulsion, core)
    return h


def graph_laplacian(n, edges, k):
    a = np.zeros((n, n))
    for i, j in edges:
        if i == j:
            raise ValueError("Self-edge does not define pair spring")
        a[i, i] += k
        a[j, j] += k
        a[i, j] -= k
        a[j, i] -= k
    return a


def step_response(points, edges, box, spring, mobility, response_times, local_weights=None,
                  relative_tolerance=1.e-9):
    """Freeze each bond's rest length to its present length, then apply xy shear."""
    b, d, lengths = rigidity_matrix(points, edges, box)
    h = spring * b.T @ b
    lam, modes = eigh(h, check_finite=True, driver="evr")
    tolerance = relative_tolerance * max(1., lam[-1])
    if lam.min() < -tolerance:
        raise ValueError("Stress-free central tangent is not positive semidefinite")
    positive = lam > tolerance
    lam, modes = lam[positive], modes[:, positive]
    affine = d[:, 0] * d[:, 1] / lengths
    drive = spring * (b.T @ affine)
    projection = modes.T @ drive
    if np.linalg.norm(drive - modes @ projection) > 1.e-7 * max(1., np.linalg.norm(drive)):
        raise ValueError("Affine drive outside central tangent range")
    amplitude = projection**2 / lam
    born = float(spring * affine @ affine)
    removed = float(amplitude.sum())
    plateau = born - removed
    if plateau < -1.e-8*max(1., born):
        raise ValueError("Unphysical negative relaxed central stiffness")
    plateau = max(0., plateau)
    t = np.asarray(response_times)
    terms = 1 - np.exp(-mobility * t[:, None] * lam[None, :])
    displacement = -(terms * (projection / lam)) @ modes.T
    # Endpoint nonaffinity corrects bond extension under a unit xy shear.
    extension = affine[None, :] + displacement @ b.T
    edge_response = spring * extension * affine[None, :]
    total = edge_response.sum(axis=1) / box**3
    if np.any(total < -1.e-8) or np.any(np.diff(total) > 1.e-8):
        raise ValueError("Central step response must be nonnegative and decreasing")
    expected = (born - terms @ amplitude) / box**3
    np.testing.assert_allclose(total, expected, atol=1.e-10, rtol=1.e-10)
    if local_weights is None:
        local = np.empty((len(t), 0))
    else:
        local = edge_response @ local_weights
    scalar = np.linalg.eigvalsh(graph_laplacian(len(points), edges, spring))
    scalar = scalar[scalar > relative_tolerance * max(1., scalar.max())]
    # This is a separate, Gaussian-vector, thermal excess stress correlator.
    rouse = np.exp(-2 * mobility * t[:, None] * scalar).sum(axis=1) / box**3
    return {
        "global_stress_free_tangent_response": total,
        "local_weighted_tangent_response": local,
        "stress_free_tangent_plateau": plateau / box**3,
        "affine_born": born / box**3,
        "central_positive_modes": len(lam),
        "central_zero_modes": 3*len(points)-len(lam),
        "central_smallest_positive_rate": mobility*float(lam.min()) if len(lam) else np.nan,
        "rouse_excess_response": rouse,
        "gaussian_positive_modes": len(scalar),
        "graph_components": len(points)-len(scalar),
        "edge_cyclomatic_number": len(edges)-len(scalar),
    }
