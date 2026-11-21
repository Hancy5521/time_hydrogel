"""Marginalize the unobserved transverse origin of a lawful formation trajectory."""
import numpy as np
from scipy.special import logsumexp

from aam_formation_actual_rollout import _packet_copy
from aam_formation_cycle_data import _root_groups
from aam_snapshot_process import PORTS, WINDOW_Y


def translate_packet(packet, offset):
    """One common torus translation, retaining the original lifted graph gauge."""
    shifted = _packet_copy(packet)
    state, _, geometry, *_ = shifted
    box = float(geometry[2])
    state.unwrapped[:, 1] += offset
    state.points[:, 1] = (state.points[:, 1] + offset) % box
    return shifted


def _weight_matrix(points, shifts, box, view):
    x = (points[None, :, 0] - np.asarray(PORTS)[:, None] + .5*box) % box - .5*box
    y = (points[None, :, 1] + shifts[:, None] - view + .5*box) % box - .5*box
    return np.maximum(1. - np.abs(y[:, None, :])/1.3, 0.) * np.maximum(1. - np.abs(x[None, :, :])/1.3, 0.)


def camera_orbit(packet, offsets):
    """Calculate the original four assay channels at every common displacement."""
    state, material, geometry, *_ = packet
    aam, initial = int(material[0]), int(material[2])
    w = _weight_matrix(state.points,np.asarray(offsets),geometry[2],WINDOW_Y[0])
    m = state.owner[:2*aam:2]>=0
    q = (w[..., :aam] @ m) / np.maximum(w[..., :aam].sum(axis=-1),1e-12)
    inventory = 6.*(w[..., aam:] @ state.present[aam:])/initial
    return np.stack((q,inventory),axis=-1).reshape(-1,4)


def hidden_q_orbit(packet, offsets):
    """Calculate the original hidden spatial graph output after common translation."""
    state, material, geometry, *_ = packet
    aam,mbaa=int(material[0]),int(material[1])
    present=state.present
    root=_root_groups(state.labels)
    owners=state.owner.reshape(-1,2)
    live=(owners>=0).any(axis=1)
    tally={k:set() for k in np.unique(root)}
    for i,k in enumerate(root):
        tally[k].update(int(v) for v in owners[i] if v>=0)
    molecular_mass=np.where(np.arange(len(present))<aam,1.,2.17)
    connected=np.asarray([len(tally[i])>=3 for i in root])&live
    w=_weight_matrix(state.points,np.asarray(offsets),geometry[2],WINDOW_Y[1])
    local=(w@(connected*molecular_mass))/np.maximum(w@(present*molecular_mass),1e-12)
    conversion=np.full((len(offsets),1),state.counts[5]/(aam+2*mbaa))
    return np.concatenate((local,conversion),axis=1)


def posterior_from_orbits(clean, measured, noise):
    """Return joint probabilities over legal history index and transverse origin."""
    camera=np.asarray(clean,dtype=float)
    target=np.asarray(measured,dtype=float)
    sigma=np.asarray(noise,dtype=float)
    if (camera.ndim!=4 or camera.shape[-2:]!=target.shape or sigma.shape!=(target.shape[-1],)
            or not np.isfinite(camera).all() or not np.isfinite(target).all()
            or np.any(sigma<=0) or not np.isfinite(sigma).all()):
        raise ValueError('Require parent × origin × time × channel with a valid diagonal likelihood.')
    distance=(camera-target)/sigma
    log_lik=-.5*np.sum(distance**2,axis=(-1,-2))
    normalize=logsumexp(log_lik)
    joint=np.exp(log_lik-normalize)
    joint/=joint.sum()
    return joint
