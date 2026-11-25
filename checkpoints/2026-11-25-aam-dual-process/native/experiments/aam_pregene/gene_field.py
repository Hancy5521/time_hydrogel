"""Decode a small material code into the shared native AAm/MBAA event field.

The code specifies effective response parameters in the simulator. The caller
supplies the sample state and environment or feed history independently, so a
single code can be used unchanged across histories. All coordinates are log
effects relative to the same base field. A coordinate is not a measured rate
constant, nor does decoding establish that a change is physically realizable.
"""

from copy import deepcopy

import numpy as np

if __package__:
    from .formation_step import CAP, make_config
else:
    from formation_step import CAP, make_config


COORDINATES = ("initiation", "propagation", "pendant_activity", "bridge_cycle_bias")
BOUND = 1.10
ZERO = (0., 0., 0., 0.)


def decode(code=ZERO, base=None):
    """Return an independent native configuration for ``build_packet``.

    Initiation scales both AAm and MBAA initiation. Propagation scales ordinary
    AAm addition, while the first MBAA attachment stays at the base rate.
    Pendant activity scales both primitive second-site crosslink rates.

    A positive bridge bias multiplies factors by exp(g) for cross-component
    bridges and by exp(-g/2) for both cycle types. The native ``factor_weights``
    normalizes these factors over available pendant reactions at each state.
    Hence this edit changes their relative weights, with a bridge-to-cycle
    odds multiplier exp(3*g/2), while preserving the total instantaneous hazard
    at that state. The realized trajectories can subsequently differ.
    """
    if not np.isrealobj(code):
        raise ValueError("The material code must contain real coordinates.")
    values = np.asarray(code, dtype=np.float64)
    if (values.shape != (len(COORDINATES),) or not np.isfinite(values).all()
            or np.any(np.abs(values) > BOUND)):
        raise ValueError(f"Expected four finite material coordinates in [-{BOUND}, {BOUND}].")

    config = make_config() if base is None else deepcopy(base)
    initiation, propagation, pendant, bias = values
    if initiation or propagation or pendant:
        rates = np.array(config["rates"], dtype=np.float64, copy=True)
        if rates.shape != (8, 2) or not np.isfinite(rates).all() or np.any(rates < 0.):
            raise ValueError("The base field requires finite nonnegative 8x2 primitive rates.")
        with np.errstate(over="ignore"):
            rates[:2, 0] *= np.exp(initiation)
            rates[2, :] *= np.exp(propagation)
            rates[4:6, :] *= np.exp(pendant)
        if not np.isfinite(rates).all():
            raise ValueError("The edited primitive rates exceed the native numerical range.")
        config["rates"] = rates.tolist()

    if bias:
        factors = np.array(config["factors"], dtype=np.float64, copy=True)
        if (factors.shape != (3, CAP + 1) or not np.isfinite(factors).all()
                or np.any(factors <= 0.)):
            raise ValueError("The base field requires finite positive pendant factors.")
        with np.errstate(over="ignore"):
            factors *= np.exp(bias * np.array([1., -.5, -.5]))[:, None]
        if not np.isfinite(factors).all() or np.any(factors <= 0.):
            raise ValueError("The edited pendant factors exceed the native numerical range.")
        config["factors"] = factors.tolist()

    return config
