"""Read-only actual-network to dimensionless frozen-geometry shear response.

This is a thin adapter to an existing, explicitly stress-free tangent model.
It does not simulate formation, infer a state, fit, or generate random numbers.
Imports perform no scientific work. Deterministic small-graph checks are defined
below for a later authorized integrated execution, never called automatically.
"""


SCHEMA = 'aam-actual-network-properties-stress-free-tangent-v1'
SOLVER_REFERENCE_SHA256 = '118e05279d5046c0809a3d22a3d5aed35e350fb84259c70131d1783ec22f6621'
RELATIVE_TOLERANCE = 1.e-9


def model_contract():
    """Return semantics only; no file, array, simulator or RNG access."""
    return {
        'model_id': SCHEMA,
        'solver': 'aam_graph_linear_response.step_response',
        'solver_reference_sha256': SOLVER_REFERENCE_SHA256,
        'solver_hash_verified_by_this_function': False,
        'hash_contract': 'The integrating registration must freeze/verify the solver source; this adapter does not hash it per snapshot.',
        'energy': 'U(u,gamma)=0.5*sum_e k*(n_e dot(u_j-u_i)+gamma*n_ex*d_ey)^2',
        'reference': 'Each existing edge is stress-free at its CURRENT snapshot length; common k is cfg.motion.spring.',
        'edge_law': 'Central axial tangents, including all parallel valid edges with their original multiplicity.',
        'geometry': 'Snapshot coordinates fixed during the mechanical probe; no chemical reaction or Brownian resampling.',
        'periodic_images': 'Current wrapped points and minimum-image edge directions; stored formation lifts are not used by this tangent model.',
        'boundary': 'Unit affine xy engineering shear plus unconstrained periodic nonaffine vertex relaxation.',
        'dynamics': 'du/dtau=-mu*(H*u+Xi*gamma); H=k*B.T*B; mu=cfg.motion.diffusion in the existing kBT=1 model.',
        'response': 'G(tau)=[C_A-sum_p (Xi dot v_p)^2*(1-exp(-mu*lambda_p*tau))/lambda_p]/box^3',
        'formation_and_probe_clocks': 'formation_age is the chemical snapshot age; probe_tau is a separate mechanical relaxation lag.',
        'thermal_scope': 'No thermal-entropic elastic contribution or solvent viscosity; deterministic quadratic tangent only.',
        'original_potential_scope': 'Not the original finite-rest/prestressed/soft-repulsion potential; its stability is not evaluated here.',
        'units': {
            'length': 'Existing simulation length unit; box is not assigned a real metre scale.',
            'energy': 'Existing dimensionless model energy; kBT=1 convention only.',
            'G_Born_plateau': 'Model energy per model volume; reported dimensionlessly, not Pa.',
            'time': 'Model probe lag, not calibrated physical seconds; distinct from formation age.',
        },
        'not_claimed': ['measured hydrogel modulus', 'true rheometric gel point',
                        'ionic/electronic conductivity', 'hydrodynamics', 'bending/entanglement',
                        'long-time elasticity from a positive short-time response'],
        'connectivity_q_used_as_property': False,
        'relative_eigenvalue_tolerance': RELATIVE_TOLERANCE,
    }


def _inputs(state, cfg, probe_tau):
    import numpy as np
    points = np.asarray(state.points)
    present = np.asarray(state.present)
    edges = np.asarray(state.edges)
    counts = np.asarray(state.counts)
    position = np.asarray(state.position)
    if (points.ndim != 2 or points.shape[1] != 3 or points.dtype.kind != 'f'
            or not np.isfinite(points).all() or present.shape != (len(points),)
            or present.dtype.kind != 'b'):
        raise ValueError('A native finite floating points(N,3) and Boolean present(N) snapshot is required.')
    if (edges.ndim != 2 or edges.shape[1] != 2 or edges.dtype.kind not in 'iu'
            or counts.ndim != 1 or len(counts) < 7 or counts.dtype.kind not in 'iu'
            or position.shape != (4,) or position.dtype.kind != 'f'
            or not np.isfinite(position).all()):
        raise ValueError('Native edge buffer, integer counts and four-field clock are required.')
    if int(counts[6]) != 0:
        raise ValueError('Only a completed actual snapshot, not an in-slab half-step State, may be probed.')
    age = float(position[0])
    if age < 0.:
        raise ValueError('Formation age must be finite and nonnegative.')
    motion = cfg['motion']
    box, spring, mobility = (float(motion[k]) for k in ('box', 'spring', 'diffusion'))
    if (not np.isfinite((box, spring, mobility)).all() or box <= 0.
            or spring <= 0. or mobility < 0.):
        raise ValueError('Positive box/spring and nonnegative frozen mobility are required.')
    if np.any(points < 0.) or np.any(points >= box):
        raise ValueError('Use actual wrapped State.points, not unwrapped or a reconstructed initial cloud.')
    edge_count = int(counts[1])
    if edge_count < 0 or edge_count > len(edges):
        raise ValueError('Native valid-edge count is outside its existing buffer.')
    valid = edges[:edge_count]
    if edge_count:
        if np.any(valid < 0) or np.any(valid >= len(points)) or np.any(valid[:, 0] == valid[:, 1]):
            raise ValueError('Each valid edge must join two distinct native nodes.')
        if not np.all(present[valid]):
            raise ValueError('A valid edge to an absent node cannot be silently discarded.')
    tau = np.atleast_1d(np.array(probe_tau, dtype=np.float64, copy=True))
    if (tau.ndim != 1 or not len(tau) or not np.isfinite(tau).all()
            or np.any(tau < 0.) or np.any(np.diff(tau) < 0.)):
        raise ValueError('Probe lag must be a scalar or a finite, nonnegative, nondecreasing vector.')
    ids = np.flatnonzero(present)
    remap = np.full(len(points), -1, dtype=np.int64)
    remap[ids] = np.arange(len(ids), dtype=np.int64)
    # Advanced indexing and explicit copies ensure ownership, even for n=0/E=0.
    local_points = np.array(points[ids], dtype=np.float64, copy=True)
    local_edges = np.array(remap[valid], dtype=np.int64, copy=True).reshape(-1, 2)
    parallel = edge_count-len(np.unique(np.sort(local_edges, axis=1), axis=0)) if edge_count else 0
    return local_points, local_edges, ids.copy(), tau, age, box, spring, mobility, parallel


def properties_from_actual_snapshot(state, cfg, probe_tau):
    """Read actual present nodes/valid edges, never replay or mutate the State.

    Returns arrays for a scalar lag too (length one). The entire original box
    volume is retained after selecting present vertices. An empty graph has
    zero response under this declared spring-only model. No value is inferred
    from connected fraction, no floor is applied to create a useful property.
    """
    import numpy as np
    from aam_graph_linear_response import step_response
    points, edges, ids, tau, age, box, spring, mobility, parallel = _inputs(state, cfg, probe_tau)
    if len(ids):
        solved = step_response(points, edges, box, spring, mobility, tau,
                               relative_tolerance=RELATIVE_TOLERANCE)
        response = np.array(solved['global_stress_free_tangent_response'], copy=True)
        born = float(solved['affine_born'])
        plateau = float(solved['stress_free_tangent_plateau'])
        positive = int(solved['central_positive_modes'])
        zero = int(solved['central_zero_modes'])
        rate = float(solved['central_smallest_positive_rate'])
        smallest_rate = rate if np.isfinite(rate) else None
    else:
        # Empty sum of the explicitly declared energy, not a replacement law.
        response = np.zeros(len(tau), dtype=np.float64)
        born = plateau = 0.
        positive = zero = 0
        smallest_rate = None
    if not np.isfinite(response).all() or not np.isfinite((born, plateau)).all():
        raise ArithmeticError('Mechanical solver returned a nonfinite primary property.')
    return {
        'schema': SCHEMA,
        'formation_age': age,
        'probe_tau': tau.copy(),
        'dimensionless_finite_G_tau': response,
        'dimensionless_affine_Born': born,
        'dimensionless_static_tangent_plateau': plateau,
        'volume': box**3,
        'present_original_node_ids': ids,
        'present_nodes': len(ids),
        'valid_edges': len(edges),
        'parallel_edges_beyond_unique_pairs': int(parallel),
        'central_positive_modes': positive,
        'central_zero_modes': zero,
        'smallest_positive_relaxation_rate': smallest_rate,
        'no_present_nodes': len(ids) == 0,
        'no_valid_edges': len(edges) == 0,
        'original_prestress_stability': 'not_evaluated',
        'static_plateau_is_not_inferred_from_short_lags': True,
        'model': model_contract(),
    }


def deterministic_small_graph_checks():
    """Defined only; later root may run once in an integrated version contract.

    No RNG/simulator/fit. Includes no-edge and no-present-node limits, analytic
    two-node/parallel-spring response, periodic translation and relabeling,
    monotone/nonnegative finite relaxation and every fixture State array's bytes.
    """
    from collections import namedtuple
    import copy
    import numpy as np
    fields = ('points', 'unwrapped', 'present', 'active', 'available', 'owner',
              'labels', 'edges', 'lifts', 'events', 'periodic_parent',
              'periodic_shift', 'periodic_rank', 'winding', 'counts', 'position')
    FixtureState = namedtuple('FixtureState', fields)
    cfg = {'motion': {'box': 6., 'spring': 9., 'diffusion': .26}}
    tau = np.array([0., .1, .5, 2.])
    maximum = 0.
    calls = 0

    def fixture(points, edges, present=None):
        p = np.array(points, dtype=np.float64).reshape(-1, 3)
        n = len(p)
        e = np.array(edges, dtype=np.int64).reshape(-1, 2)
        # One unused invalid buffer row verifies counts[1], not full capacity.
        buffer = np.vstack((e, np.full((1, 2), -1, dtype=np.int64)))
        counts = np.zeros(7, dtype=np.int64); counts[1] = len(e)
        position = np.array([3.15, 3.15, 3.15, 0.], dtype=np.float64)
        return FixtureState(p, p.copy(), np.ones(n, dtype=bool) if present is None else np.array(present, dtype=bool),
                            np.zeros(2*n, bool), np.zeros(2*n, bool), np.full(2*n, -1, np.int64),
                            np.arange(n, dtype=np.int64), buffer, np.zeros((len(buffer), 3), np.int64),
                            np.zeros((max(1, len(e)), 13)), np.arange(n, dtype=np.int64),
                            np.zeros((n, 3), np.int64), np.zeros(n, np.int64),
                            np.zeros((n, 3), bool), counts, position)

    def check_close(a, b, name, tol=2.e-11):
        nonlocal maximum
        x, y = np.asarray(a), np.asarray(b)
        if x.shape != y.shape or not np.isfinite(x).all() or not np.isfinite(y).all():
            raise AssertionError('Nonfinite/schema test failure: '+name)
        err = float(np.max(np.abs(x-y))) if x.size else 0.
        maximum = max(maximum, err)
        if err > tol:
            raise AssertionError(name+': absolute error '+str(err))

    def probe(state, lags=tau):
        nonlocal calls
        before = copy.deepcopy(state)
        config_before = copy.deepcopy(cfg)
        lags_before = np.array(lags, copy=True)
        answer = properties_from_actual_snapshot(state, cfg, lags)
        calls += 1
        for name in fields:
            a, b = getattr(state, name), getattr(before, name)
            if a.shape != b.shape or a.dtype != b.dtype or a.tobytes() != b.tobytes():
                raise AssertionError('Fixture State mutated: '+name)
        if cfg != config_before:
            raise AssertionError('Configuration mutated.')
        check_close(np.array(lags), lags_before, 'input lag unchanged', tol=0.)
        for a in state:
            for b in (answer['probe_tau'], answer['dimensionless_finite_G_tau'], answer['present_original_node_ids']):
                if np.shares_memory(a, b):
                    raise AssertionError('Output aliases State input.')
        response = answer['dimensionless_finite_G_tau']
        if np.min(response) < -1.e-10 or np.any(np.diff(response) > 1.e-10):
            raise AssertionError('Declared tangent relaxation must be nonnegative/nonincreasing.')
        if answer['dimensionless_static_tangent_plateau'] > answer['dimensionless_affine_Born']+1.e-10:
            raise AssertionError('Relaxed plateau cannot exceed Born term.')
        return answer

    pair_points = [[1., 1., 1.], [1.6, 1.8, 1.]]
    empty = probe(fixture(pair_points, []))
    check_close(empty['dimensionless_finite_G_tau'], np.zeros(len(tau)), 'empty graph')
    vacuum = probe(fixture(pair_points, [], present=[False, False]))
    check_close(vacuum['dimensionless_finite_G_tau'], np.zeros(len(tau)), 'no present nodes')
    if vacuum['present_nodes'] or vacuum['valid_edges']:
        raise AssertionError('Absent fixture nodes entered the energy sum.')
    one = probe(fixture(pair_points, [[0, 1]]))
    two = probe(fixture(pair_points, [[0, 1], [0, 1]]))
    d = np.array(pair_points[1])-pair_points[0]
    affine = d[0]*d[1]/np.linalg.norm(d)
    born = cfg['motion']['spring']*affine**2/6.**3
    rate = 2.*cfg['motion']['spring']*cfg['motion']['diffusion']
    check_close(one['dimensionless_finite_G_tau'], born*np.exp(-rate*tau), 'single axial spring analytic')
    check_close(two['dimensionless_finite_G_tau'], 2.*born*np.exp(-2.*rate*tau), 'parallel springs analytic')
    check_close(np.array(two['dimensionless_affine_Born']), np.array(2.*one['dimensionless_affine_Born']), 'parallel Born')
    if two['parallel_edges_beyond_unique_pairs'] != 1:
        raise AssertionError('Parallel physical edges were collapsed.')
    points = np.array([[.2, .3, .5], [1., .9, .8], [1.8, .4, 1.1],
                       [5.8, 1.2, .7], [2.5, 3.1, 1.4]])
    edges = np.array([[0, 1], [1, 2], [2, 3], [3, 0], [0, 2], [0, 1]])
    present = np.array([True, True, True, True, False])
    base_state = fixture(points, edges, present)
    original = probe(base_state)
    translated = probe(fixture((points+[2.1, 1.4, .8]) % 6., edges, present))
    permutation = np.array([3, 0, 4, 2, 1]); inverse = np.argsort(permutation)
    permuted = probe(fixture(points[permutation], inverse[edges], present[permutation]))
    for name in ('dimensionless_finite_G_tau', 'dimensionless_affine_Born', 'dimensionless_static_tangent_plateau'):
        check_close(np.asarray(original[name]), np.asarray(translated[name]), 'periodic translation/'+name)
        check_close(np.asarray(original[name]), np.asarray(permuted[name]), 'node permutation/'+name)
    scalar = probe(base_state, .5)
    check_close(scalar['dimensionless_finite_G_tau'], original['dimensionless_finite_G_tau'][2:3], 'scalar lag API')
    bad = fixture(points, [[0, 4]], present)
    try:
        properties_from_actual_snapshot(bad, cfg, tau)
    except ValueError:
        absent_edge_rejected = True
    else:
        raise AssertionError('A valid edge to an absent node was silently dropped.')
    return {'passed': True, 'scope': 'Deterministic implementation fixtures only, not material evidence.',
            'probe_calls': calls, 'maximum_absolute_error': maximum,
            'checks': ['empty graph', 'no present nodes', 'parallel spring multiplicity and exact relaxation',
                       'periodic translation', 'node permutation', 'nonnegative monotone relaxation',
                       'all16 State fields unchanged', 'unused edge buffer ignored', 'scalar lag'],
            'absent_endpoint_edge_rejected': absent_edge_rejected,
            'simulator_calls': 0, 'random_draws': 0, 'fit_steps': 0,
            'result_status': 'Tests execute only when this function is explicitly called by an authorized integration.'}
