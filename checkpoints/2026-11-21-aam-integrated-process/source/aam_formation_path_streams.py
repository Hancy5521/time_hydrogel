"""Registered V4 role streams: iid uint32 seed draws, with replacement.

No RNG is instantiated at import. Equal drawn uint32 values are allowed and are
never discarded: named generators separate *roles*, not guaranteed seed values.
"""
import numpy as np

MASTER = 3151920000
SCHEMA = 'aam-complete-prior-independent-increment-streams-v4'
ROLES = {
    'truth_initial': 101, 'truth_native': 102, 'truth_assay': 103,
    'truth_future_native': 104, 'truth_future_placement': 105, 'truth_future_marker': 106,
    'infer_initial': 201, 'infer_accept': 202, 'infer_native': 203, 'infer_resample': 204,
    'forecast_native': 301, 'forecast_placement': 302, 'forecast_marker': 303,
    'path_initial': 401, 'path_resample': 402, 'path_order': 403,
    'path_seed_proposal': 404, 'path_accept': 405,
    'pool_initial': 451, 'pool_resample': 452,
    'teacher_path_initial': 501, 'teacher_path_resample': 502, 'teacher_path_order': 503,
    'teacher_path_seed_proposal': 504, 'teacher_path_accept': 505,
    'teacher_pool_initial': 551, 'teacher_pool_resample': 552,
}


def rng(query, role, *, arm=0, suffix=0, atom=0):
    if role not in ROLES:
        raise ValueError('Use a declared random role.')
    key = (query, arm, suffix, atom)
    if any(type(k) is not int or k < 0 for k in key):
        raise ValueError('Stream indices must be nonnegative Python integers.')
    return np.random.default_rng(np.random.SeedSequence(
        entropy=MASTER, spawn_key=(ROLES[role], *key)))


def seed_draws(query, role, *, size=None, arm=0, suffix=0, atom=0, marker=False):
    # A marker only satisfies legacy numeric provenance validation; it does not
    # initialize a physical generator in streamed_rollout.
    high = 2**32-52 if marker else 2**32
    return rng(query, role, arm=arm, suffix=suffix, atom=atom).integers(
        0, high, size=size, dtype=np.uint32)


def future_streams(query, *, arm=0, suffix=0, atom=0, truth=False):
    stem = 'truth_future_' if truth else 'forecast_'
    return {'native_seeds': seed_draws(query, stem+'native', size=3, arm=arm,
                                       suffix=suffix, atom=atom),
            'placement_seed': int(seed_draws(query, stem+'placement', arm=arm,
                                            suffix=suffix, atom=atom)),
            'seed': int(seed_draws(query, stem+'marker', arm=arm, suffix=suffix,
                                  atom=atom, marker=True))}


def contract():
    return {'schema': SCHEMA, 'master': MASTER, 'roles': ROLES.copy(),
            'generator': 'NumPy PCG64,SeedSequence(entropy=master,spawn_key=(role,query,arm,suffix,atom))',
            'initial_seed_law': 'Uniform over every uint32 value,with replacement;mu=initialize_pushforward(nu0)',
            'native_seed_law': 'Independent role-generator uint32 draws per interval/slot;no deduplication',
            'assay_and_acceptance': 'Distinct named controller streams,not physical seed arithmetic',
            'future_coupling': 'Each atom repeats its explicit placement/native streams over six actions only as CRN',
            'V1_relation': 'Same effective chemistry/observation/response,independent-increment sampling law;not bitwise V1 arithmetic-seed jointlaw'}
