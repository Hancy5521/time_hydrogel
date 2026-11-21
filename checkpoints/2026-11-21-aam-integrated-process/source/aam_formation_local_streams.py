"""V5 prospective named streams. Import never draws or simulates."""
from __future__ import annotations

import numpy as np

MASTER = 3151930000
NAMESPACE = 'aam-formation-explicit-initial-local-v5-20261119'
NAMESPACE_CODE = 501119
GROUPS = {'train': 1, 'dev': 2, 'held': 3}
ROLES = {
    'truth_initial': 101, 'truth_native': 102, 'truth_assay': 103,
    'prior': 201, 'proposal': 202, 'seed': 203, 'accept': 204,
    'resample': 205, 'stock': 206,
    'future_native': 301, 'future_placement': 302, 'future_marker': 303,
    'model_forest': 401,
}


def stream_address(group, mother, role, *, arm=-1, slot=0, repeat=0):
    if group not in GROUPS or role not in ROLES:
        raise ValueError('Unknown prospective V5 group/role.')
    for value, name, minimum in ((mother, 'mother', 0), (arm, 'arm', -1),
                                 (slot, 'slot', 0), (repeat, 'repeat', 0)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
            raise ValueError(name+' must be an integer.')
        if not minimum <= int(value) < 2**32-1:
            raise ValueError(name+' outside registered uint32 address domain.')
    return (MASTER, NAMESPACE_CODE, GROUPS[group], int(mother), ROLES[role],
            int(arm)+1, int(slot), int(repeat))


def rng(group, mother, role, *, arm=-1, slot=0, repeat=0):
    return np.random.default_rng(np.random.SeedSequence(
        stream_address(group, mother, role, arm=arm, slot=slot, repeat=repeat)))


def seed_draws(group, mother, role, *, arm=-1, slot=0, repeat=0, size=None):
    generator = rng(group, mother, role, arm=arm, slot=slot, repeat=repeat)
    values = generator.integers(0, 2**32, size=size, dtype=np.uint32)
    return int(values) if size is None else values


def sampler_rngs(group, mother, arm):
    if int(arm) not in (0, 1, 2):
        raise ValueError('V5 production/dev arms use0global,1local,2SIR.')
    return {role: rng(group, mother, role, arm=arm)
            for role in ('prior', 'proposal', 'seed', 'accept', 'resample', 'stock')}


def future_streams(group, mother, *, arm=-1, slot=0, repeat=0):
    """No action index: six interventions share explicit increments/placement CRN.

    arm=-1 denotes the actual teacher training/reference continuation. Native
    predictive arms0..2 use their own explicit slot streams. The marker is
    provenance compatibility only; the streamed kernel does not use it as RNG.
    """
    return {
        # Legacy validator checks marker+51 even though it is never physical RNG.
        'seed': int(rng(group, mother, 'future_marker', arm=arm,
                        slot=slot, repeat=repeat).integers(0, 2**32-51, dtype=np.uint32)),
        'native_seeds': seed_draws(group, mother, 'future_native', arm=arm,
                                   slot=slot, repeat=repeat, size=3),
        'placement_seed': seed_draws(group, mother, 'future_placement', arm=arm,
                                     slot=slot, repeat=repeat),
    }
