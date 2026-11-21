"""Reject a formation-path proposal once its full MH threshold is unreachable."""
from time import perf_counter

import numpy as np

from aam_formation_cycle_core import _advance, _physical_config
from aam_formation_local_initial import build_packet, initial_fingerprint, validate_initial
from aam_formation_local_tempering import _seeds, mh_log_acceptance
from aam_formation_path_tempering import _copy_packet, _new_cost, _query
from aam_snapshot_process import likelihood, observe


def reject_with_remaining_bound(observed, old_total, beta, log_hastings, uniform, upper_remaining=None):
    """Test the ordinary MH uniform against a nonnegative-suffix constraint."""
    observed = np.asarray(observed, float)
    k = len(observed)
    upper = np.zeros(3) if upper_remaining is None else np.asarray(upper_remaining, float)
    if (observed.shape != (k,) or not 0 <= k < 3 or upper.shape != (3,)
            or not np.isfinite(observed).all() or np.any(observed > 0.)
            or not np.isfinite(upper).all() or np.any(upper > 0.)
            or not np.isfinite(old_total) or not np.isfinite(log_hastings)
            or not 0. <= beta <= 1. or not 0. <= uniform < 1.):
        raise ValueError('Require three bounded Gaussian camera factors and one ordinary MH draw.')
    # Use the same three-position floating-point reduction as a full V5 replay.
    # Each missing entry is an admissible upper bound. Addition, subtraction
    # of the same total, multiplication by nonnegative beta, and min are monotone.
    upper_total = np.r_[observed, upper[k:]].sum()
    possible = mh_log_acceptance(beta, old_total, upper_total, log_hastings)
    return bool(uniform > 0. and np.log(uniform) >= possible), float(possible)


def replay(config, factors, initial, seeds, measured, old, change, beta, ratio, uniform, *, bounded):
    """Continue only the changed prefix suffix, with exact common-uniform decisions."""
    start_clock = perf_counter()
    y, noise = _query(config, measured)
    actual = _physical_config(config, factors)
    validate_initial(actual, initial)
    seeds = _seeds(seeds)
    start = 0 if change is None else change
    if start not in (0,1,2) or change is not None and (
            initial_fingerprint(initial)!=initial_fingerprint(old['initial'])
            or not np.array_equal(np.delete(seeds,change),np.delete(old['seeds'],change))):
        raise ValueError('Only the declared original interval may differ in partial replay.')
    cost = _new_cost()
    if change is None:
        raw = build_packet(actual, initial, factors=factors)
        checkpoints=[_copy_packet(raw,cost,readonly=True)]
        clean=[]
    else:
        checkpoints=list(old['checkpoints'][:start+1])
        raw=_copy_packet(checkpoints[-1],cost)
        clean=list(old['clean'][:start])
    old_total=float(old['camera_ll'].sum())
    for camera in range(start,3):
        if bounded:
            observed=likelihood(np.asarray(clean,float).reshape(-1,4),y[:camera],noise)
            reject,threshold=reject_with_remaining_bound(observed,old_total,beta,ratio,uniform)
            if reject:
                return None, {'accepted':False,'screened_at':camera,'upper_log_acceptance':threshold,
                              'camera_prefix_ll':observed,'native_calls':camera-start,
                              'seconds':perf_counter()-start_clock}
        age=np.asarray(config['prefix_clock'][:camera+2],float)
        _advance(raw,age,int(seeds[camera]),False,cost)
        clean.append(observe(_copy_packet(raw,cost)))
        checkpoints.append(_copy_packet(raw,cost,readonly=True))
    measured_path=np.asarray(clean,float)
    camera_ll=likelihood(measured_path,y,noise)
    if not np.isfinite(camera_ll).all() or np.any(camera_ll > 0.):
        raise ValueError('Camera law exceeded its registered range or failed numerically.')
    decision=mh_log_acceptance(beta,old_total,float(camera_ll.sum()),ratio)
    accepted=bool(uniform==0. or np.log(uniform)<decision)
    path={'initial':initial,'seeds':seeds,'clean':measured_path,'camera_ll':camera_ll,'checkpoints':checkpoints}
    return path,{'accepted':accepted,'screened_at':None,'log_acceptance':decision,'native_calls':3-start,
                 'seconds':perf_counter()-start_clock}
