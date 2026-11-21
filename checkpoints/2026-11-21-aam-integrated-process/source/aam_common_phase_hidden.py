"""Original hidden-aperture readouts at common transverse phase offsets."""

import numpy as np

from aam_camera_information import component_features
from aam_snapshot_process import PORTS, WINDOW_Y, weight


def hidden_orbit(packet, offsets):
    """Read the two hidden ports at each offset without changing the covalent graph."""
    shifts = np.asarray(offsets, dtype=float)
    if shifts.ndim != 1 or shifts.size == 0 or not np.isfinite(shifts).all():
        raise ValueError('Provide a nonempty vector of finite transverse offsets.')

    state, material, geometry, *_ = packet
    aam = int(material[0])
    box = float(geometry[2])
    points = np.asarray(state.points)
    positions = np.broadcast_to(points[:, :2], (len(shifts), len(points), 2)).copy()
    positions[..., 1] = (positions[..., 1] + shifts[:, None]) % box
    xy = positions.reshape(-1, 2)
    weights = np.stack([weight(xy, center, WINDOW_Y[1], box).reshape(len(shifts), len(points))
                        for center in PORTS], axis=1)

    present = np.asarray(state.present, dtype=bool)
    used = np.asarray(state.owner).reshape(-1, 2) >= 0
    available = np.asarray(state.available).reshape(-1, 2)
    incorporated = used.any(axis=1) & present
    pendant = (used[aam:].sum(axis=1) == 1) & (available[aam:].sum(axis=1) == 1) & present[aam:]
    bonds = int(state.counts[1])
    size, winding = component_features(aam, incorporated, state.edges[:bonds], state.lifts[:bonds])
    mass = present * np.where(np.arange(len(present)) < aam, 1., 2.17)
    local_mass = weights @ mass
    aam_weights = weights[..., :aam]
    finite_weights = aam_weights * (incorporated[:aam] & ~winding)
    numerator = np.stack((aam_weights @ incorporated[:aam], finite_weights @ size,
                          2.17 * (weights[..., aam:] @ pendant)), axis=-1)
    denominator = np.stack((local_mass, aam * finite_weights.sum(axis=-1), local_mass), axis=-1)
    result = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator != 0.)
    if not np.isfinite(result).all() or np.any(result < 0.) or np.any(result > 1. + 2.e-14):
        raise ArithmeticError('An aperture feature violated the registered physical range.')
    return result


def history_orbit(packets, offsets):
    """Return offset × camera time × hidden port × feature on the same trajectory."""
    if len(packets) != 4:
        raise ValueError('Use a complete original three-camera trajectory plus its initial packet.')
    return np.stack([hidden_orbit(packet, offsets) for packet in packets[1:]], axis=1)
