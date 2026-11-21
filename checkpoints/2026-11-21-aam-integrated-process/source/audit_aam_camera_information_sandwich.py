"""Read-only audit of hidden-aperture risk bounds and their frozen provenance."""
import argparse
from datetime import datetime
import hashlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import t

from aam_formation_actual_rollout import packet_fingerprint
from aam_formation_cycle_data import check_actual_state
from aam_formation_cycle_driver import np_load, read, sha, write
from aam_formation_local_data import load_prefix, load_query
from aam_formation_local_initial import draw_initial, initial_fingerprint, propose_initial
from aam_formation_path_cycle import load_tree, stamp
from aam_snapshot_process import observe


LAGS = (16,64,256)
FEATURES = ('incorporated_AAm_local_mass_fraction','nonwinding_component_AAm_length',
            'available_pendant_MBAA_local_mass_fraction')
ARMS = ('training_knn8','training_empirical','training_simulator_likelihood')
COUNTS = ('native_segment_attempted','native_segment_completed','native_segment_failed')


def require(value, message):
    if not value:
        raise ValueError(message)


def same(actual, expected):
    np.testing.assert_array_equal(actual,expected)


def close(actual, expected):
    np.testing.assert_allclose(actual,expected,rtol=2e-11,atol=2e-13)


def file_ref(row, path):
    require(Path(row['path']).resolve() == path and sha(path) == row['sha256'], 'Frozen artifact bytes or destination differ.')
    return path


def independent_apertures(packet):
    """Connectivity via closure, winding via rank of independent lattice equations."""
    s, material, geometry, *_ = packet
    n, a = len(s.points),int(material[0])
    edges = s.edges[:s.counts[1]]
    lifts = s.lifts[:s.counts[1]]
    linked = np.eye(n,dtype=bool)
    if len(edges):
        u,v = edges.T
        linked[u,v] = linked[v,u] = True
    for k in range(n):
        linked |= linked[:,k,None] & linked[k,None,:]
    mass = np.r_[np.ones(a),np.full(n-a,2.17)] * s.present
    owner = s.owner.reshape(n,2) >= 0
    aa = owner[:a,0] & s.present[:a]
    available_pendant = (owner[a:,:].sum(axis=1) == 1) & (s.available.reshape(n,2)[a:,:].sum(axis=1) == 1)
    finite = aa.copy()
    for group in np.unique(np.argmax(linked,axis=1)):
        bound = linked[group]
        idx = np.flatnonzero(bound[edges[:,0]])
        if not len(idx):
            continue
        equation = np.zeros((len(idx),n))
        equation[np.arange(len(idx)),edges[idx,0]] = -1.
        equation[np.arange(len(idx)),edges[idx,1]] = 1.
        if np.linalg.matrix_rank(equation) != np.linalg.matrix_rank(np.c_[equation,lifts[idx]]):
            finite[bound[:a]] = False
    sizes = linked[:a,:a] @ aa.astype(int)
    out = []
    box = geometry[2]
    for x in (1.45,4.45):
        delta = (s.points[:,:2] - [x,4.35] + box/2.) % box - box/2.
        w = np.maximum(0.,1.-np.abs(delta)/1.30).prod(axis=1)
        wa, wp = w[:a],w[a:]
        total = np.dot(w,mass)
        fw = wa * finite
        out.append((np.dot(wa,aa)/total if total else 0.,
                    np.dot(fw,sizes)/(a*fw.sum()) if fw.sum() else 0.,
                    np.dot(wp,available_pendant*mass[a:])/total if total else 0.))
    return np.asarray(out)


def history(packets):
    return np.stack([independent_apertures(p) for p in packets[1:]])


def fingerprints(packets,clock):
    h = hashlib.sha256()
    for age,packet in zip(clock,packets,strict=True):
        h.update(str(float(age)).encode('ascii'))
        h.update(packet_fingerprint(packet).encode('ascii'))
    return h.hexdigest()


def interval(values, expect, n):
    mu = np.mean(values)
    half = t.ppf(.975,n-1)*np.std(values,ddof=1)/np.sqrt(n)
    close([mu,mu-half,mu+half],[expect['mean'],*expect['t95']])
    require(n == expect['independent_mothers'], 'Repeated lags or views were counted as independent mothers.')


def camera_values(clean,y,noise):
    return -.5 * np.square((clean-y)/np.asarray(noise)).sum(axis=-1)


def oracle_route(raw, truth, plan, y, rng):
    cfg = plan['physics']
    clock = cfg['prefix_clock']
    initial = truth['initial_variables']
    seeds = truth['explicit_native_seeds']
    old = camera_values(truth['clean'],y,cfg['noise'])
    state_hash = fingerprints(truth['checkpoint_packets'],clock)
    birth = 'generating_truth'
    require(raw['valid'] and raw['initial_history_fingerprint'] == state_hash and len(raw['moves']) == LAGS[-1]
            and tuple(raw['lag_paths']) == LAGS, 'Oracle did not retain all fixed lags with the generating stationary start.')
    native = 0
    for step,row in enumerate(raw['moves'],1):
        g = rng(4,step,0)
        coin,coord = float(g.random()),int(g.integers(4))
        require(row['step'] == step and row['coin'] == coin and row['coordinate'] == coord and row['error'] is None
                and row['old_history_fingerprint'] == state_hash and row['old_birth'] == birth,
                'Oracle scan order, laziness, path routing, or attached seed changed.')
        close(row['old_log_likelihood'],old.sum())
        n_calls, accepted = 0,False
        if coin >= .5:
            g = rng(4,step,1)
            start = max(coord-1,0)
            candidate, set_seeds, ratio = initial,seeds.copy(),0.
            if coord == 0:
                candidate,ratio,_ = propose_initial(cfg,initial,g)
            else:
                set_seeds[start] = g.integers(0,2**32,dtype=np.uint32)
            same(row['proposed_seeds'],set_seeds)
            candidate_hash = initial_fingerprint(candidate)
            self_loop = candidate_hash == initial_fingerprint(initial) and np.array_equal(seeds,set_seeds)
            require(row['proposed_initial_fingerprint'] == candidate_hash and row['self_loop'] == self_loop
                    and row['log_hastings'] == ratio, 'Wrong complete MH coordinate proposal.')
            if self_loop:
                accepted = True
                require(row['uniform'] is None and row['screened_at'] is None and ratio == 0., 'An exact move identity consumed an acceptance step.')
            else:
                u = float(rng(4,step,2).random())
                require(u == row['uniform'], 'Acceptance stream differs from precommitment.')
                screen = row['screened_at']
                if screen is None:
                    n_calls = 3-start
                    ll = np.asarray(row['proposal_camera_ll'])
                    require(ll.shape == (3,) and np.isfinite(ll).all() and np.all(ll <= 0.), 'Wrong full camera contributions.')
                    close(ll[:start],old[:start])
                    threshold = min(0.,float(ll.sum())-float(old.sum())+ratio)
                    close(threshold,row['log_acceptance'])
                    accepted = u == 0. or np.log(u) < threshold
                    if accepted:
                        old,initial,seeds = ll,candidate,set_seeds
                        state_hash,birth = row['final_history_fingerprint'],f'oracle_{step:03d}'
                else:
                    prefix = np.asarray(row['camera_prefix_ll'])
                    require(start <= screen < 3 and len(prefix) == screen and np.all(prefix <= 0.), 'Invalid exact MH screening point.')
                    close(prefix[:start],old[:start])
                    bound = np.r_[prefix,np.zeros(3-screen)]
                    threshold = min(0.,float(bound.sum())-float(old.sum())+ratio)
                    close(threshold,row['upper_log_acceptance'])
                    require(u > 0. and np.log(u) >= threshold, 'Camera scores could not prove early rejection.')
                    n_calls = screen-start
            for camera,call in enumerate(row['native_calls'],start):
                require(call['camera'] == camera and call['seed'] == int(set_seeds[camera]) and call['completed']
                        and call['position_after'][0] == clock[camera+1], 'Unclosed or foreign native interval.')
                same(call['clock'],clock[:camera+2])
        else:
            require(row['uniform'] is None and row['screened_at'] is None, 'Lazy step consumed replay randomness.')
        require(row['accepted'] == accepted and row['final_history_fingerprint'] == state_hash and row['final_birth'] == birth
                and len(row['native_calls']) == n_calls and tuple(row[key] for key in COUNTS) == (n_calls,n_calls,0),
                'Oracle transition or native count does not close.')
        native += n_calls
        if step in LAGS:
            z = raw['lag_paths'][step]
            p = z['checkpoint_packets']
            require(z['history_fingerprint'] == fingerprints(p,clock) == state_hash and z['birth_id'] == birth
                    and initial_fingerprint(z['initial_variables']) == initial_fingerprint(initial), 'Saved stationary state differs from followed accepted history.')
            same(z['interval_seeds'],seeds)
            for time,packet in zip(clock,p,strict=True):
                check_actual_state(packet[0],expected_age=time)
            clean = np.stack([observe(x) for x in p[1:]])
            same(z['clean'],clean)
            close(camera_values(clean,y,cfg['noise']),old)
            close(z['camera_log_likelihood'],old)
    require(tuple(raw['cost'][k] for k in COUNTS) == (native,native,0), 'Oracle aggregate native count differs.')
    return native


def audit(out):
    r,a,f,o,s = (out/name for name in ('REGISTERED.json','ACQUIRED.json','ALL_FORECASTS_FROZEN.json','ORACLE_COMPLETED.json','SCORED.json'))
    reg,acquired,freeze,oracle,score = (read(x) for x in (r,a,f,o,s))
    hashes = {x:sha(x) for x in (r,a,f,o,s)}
    run,started = read(out/'RUN.json'),read(out/'ORACLE_STARTED.json')
    require(run['registered_before_new_physics'] and reg['same_law'] and reg['reference_chain_not_used_for_prediction']
            and not freeze['held_targets_accessed'] and not freeze['oracle_started'] and started['auditor_only']
            and hashes[r] == run['registration_sha256'] == acquired['registration_sha256'] == freeze['registration_sha256'] == score['registration_sha256']
            and hashes[a] == freeze['acquisition_sha256'] and hashes[f] == started['forecast_freeze_sha256'] == oracle['forecast_freeze_sha256'] == score['freeze_sha256']
            and hashes[o] == score['oracle_sha256'], 'Phase hashes do not preserve native registration and held prediction isolation.')
    at = lambda row:datetime.fromisoformat(row['actual_utc'])
    require(at(reg) <= at(run) <= at(acquired) <= at(freeze) <= at(started) <= at(oracle) <= at(score), 'Chronological freeze order changed.')
    for path,h in reg['inputs'].items():
        require(sha(path) == h, 'Frozen source training bytes changed.')
    for path,h in reg['sources'].items():
        require(sha(out/'SOURCE_SNAPSHOT'/Path(path).name) == h, 'Frozen source snapshot changed.')
    check_files = [(freeze['training_bank'],'TRAIN_SUPPORT.npz'),(freeze['blind_predictions'],'BLIND_PREDICTIONS.npz'),(score['arrays'],'SCORED_ARRAYS.npz')]
    training,forecasts,scored = [np_load(file_ref(row,out/name)) for row,name in check_files]
    teacher = np.asarray(reg['teacher_factors'])
    same(teacher,reg['physics']['factors'])
    expected_entropy = np.frombuffer(hashlib.sha256(reg['namespace'].encode()).digest()[:16],dtype='<u4').tolist()
    same(reg['entropy'],expected_entropy)
    v5 = Path(reg['v5'])
    for j in range(reg['training_mothers']):
        z = load_prefix(v5,j)
        same(z['factors'],teacher)
        close(history(z['checkpoint_packets']),training['feature'][j])
        same(z['clean'],training['clean'][j])
        same(load_query(v5,j,group='train')['measured'],training['noisy'][j])
    same(training['xscale'],np_load(v5/'BASELINES'/'FIT_ARRAYS.npz')['knn_Xscale'].reshape(3,4))
    n = reg['new_mother_blocks']
    require(n == 64 == score['mothers'] == len(acquired['mothers']) == len(oracle['mothers'])
            and tuple(reg['fixed_lags']) == LAGS and tuple(freeze['witness_order']) == ARMS
            and reg['effect_margin_fraction_of_prior_risk'] == .05 and score['new_fits'] == score['new_wet_measurements'] == 0,
            'The fixed cohort, lag, witness order or decision margin changed.')
    total = 0
    for m,(p,q) in enumerate(zip(acquired['mothers'],oracle['mothers'],strict=True)):
        rng = lambda purpose,*address:np.random.default_rng(np.random.SeedSequence([*expected_entropy,purpose,m,*address]))
        folder = out/'MOTHERS'/f'mother_{m:03d}'
        require(m == p['mother'] == q['mother'] and q['valid'] and q == read(folder/'ORACLE_RECEIPT.json')
                and q['sealed_prefix_sha256'] == p['sealed']['sha256'] and q['forecast_freeze_sha256'] == hashes[f], 'A separate mother or oracle receipt was omitted.')
        packets = load_tree(file_ref(p['sealed'],folder/'SEALED_PREFIXES.npz'))
        y = np_load(file_ref(p['query'],folder/'QUERY.npz'))['measured']
        same(y,forecasts['queries'][m])
        same(y,packets['truth']['clean'] + rng(3).standard_normal((3,4))*reg['physics']['noise'])
        for purpose,name in ((1,'truth'),(2,'prior_partner')):
            g = rng(purpose)
            z = packets[name]
            expected_initial = initial_fingerprint(draw_initial(reg['physics'],g))
            same(z['explicit_native_seeds'],g.integers(0,2**32,3,dtype=np.uint32))
            same(z['factors'],teacher)
            require(z['valid'] and z['mother_id'] == f'{name}:{m}' and z['initial_fingerprint'] == initial_fingerprint(z['initial_variables']) == expected_initial
                    and tuple(z['cost'][k] for k in COUNTS) == (3,3,0), 'Prior, exact law or trajectory identity changed.')
            for age,w in zip(reg['physics']['prefix_clock'],z['checkpoint_packets'],strict=True):
                check_actual_state(w[0],expected_age=age)
            same(z['clean'],np.stack([observe(w) for w in z['checkpoint_packets'][1:]]))
        near = np.argsort(np.sum(((training['noisy']-y)/training['xscale'])**2,axis=(1,2)),kind='stable')[:8]
        same(near,forecasts['neighbors'][m])
        mask = np.zeros(len(training['noisy'])); mask[near] = 1./8
        ll = camera_values(training['clean'],y,reg['physics']['noise']).sum(axis=1)
        w = np.exp(ll-ll.max()); w /= w.sum()
        close(np.stack((mask,np.full(len(mask),1./len(mask)),w)),forecasts['weights'][m])
        z = load_tree(file_ref(q['oracle'],folder/'ORACLE_RAW.npz'))
        native = oracle_route(z,packets['truth'],reg,y,rng)
        require(native == scored['oracle_native'][m] and tuple(q['actual_native'][key] for key in COUNTS) == (native,native,0), 'Sealed native count differs from actual route.')
        total += native
        close(scored['truth'][m],history(packets['truth']['checkpoint_packets']))
        close(scored['prior_partner'][m],history(packets['prior_partner']['checkpoint_packets']))
        close(scored['stationary_lags'][m],[history(z['lag_paths'][k]['checkpoint_packets']) for k in LAGS])
    require(total == score['oracle_native_intervals'] and score['prior_native_intervals'] == n*6,'Wrong new native counts.')
    same(score['any_nonzero_truth_features'],np.count_nonzero(scored['truth'],axis=(0,1,2)))
    require(score['completely_zero_truth_mothers'] == np.count_nonzero(~np.any(scored['truth'],axis=(1,2,3))), 'Target activity counts changed.')
    for j,name in [(None,'joint18'),*enumerate(FEATURES)]:
        def part(array, shape):
            return (array if j is None else array[...,j]).reshape(*shape,-1)
        z,b,lag = (part(scored[key],shape) for key,shape in [('truth',(n,)),('prior_partner',(n,)),('stationary_lags',(n,3))])
        train = part(training['feature'],(len(training['feature']),))
        dim = np.sqrt(z.shape[1])
        e0 = np.linalg.norm(z-b,axis=1)/(2*dim)
        ek = np.linalg.norm(z[:,None]-lag,axis=2)/(2*dim)
        w = forecasts['weights']
        estimate = (np.einsum('mkt,mt->mk',w,cdist(z,train)) - .5*np.einsum('mki,ij,mkj->mk',w,cdist(train,train),w))/dim
        values = {'independent_prior_risk':e0,'oracle_lag_risk':ek,**{arm:estimate[:,k] for k,arm in enumerate(ARMS)}}
        for key,value in values.items():
            close(scored[name+'__'+key],value)
        result = score['metrics'][name]
        interval(e0,result['prior_risk'],n)
        for k,ell in enumerate(LAGS):
            interval(ek[:,k],result['oracle_lower_on_bayes_risk'][str(ell)],n)
        for k,arm in enumerate(ARMS):
            interval(estimate[:,k],result['witness_risk'][arm],n)
        comparisons = {'camera_skill_lower':e0-estimate[:,0], 'camera_skill_upper':e0-ek[:,-1],
                       'camera_lower_minus_margin':.95*e0-estimate[:,0], 'camera_upper_minus_margin':.95*e0-ek[:,-1],
                       'finite_likelihood_bank_gap_vs_blind_knn':estimate[:,2]-estimate[:,0]}
        for key,value in comparisons.items():
            interval(value,result[key],n)
        low = result['camera_lower_minus_margin']['t95'][0] > 0.
        high = result['camera_upper_minus_margin']['t95'][1] < 0.
        gap = result['finite_likelihood_bank_gap_vs_blind_knn']['t95'][0] > 0.
        require((result['practically_useful_camera_signal'],result['practically_limited_camera_information'],
                 result['finite_support_likelihood_gap_with_camera_signal']) == (low,high,low and gap), 'The decision interpretation changed.')
    write(out/'INDEPENDENT_AUDIT.json',{**stamp(),'passed':True,'scored_sha256':hashes[s],
          'registration_sha256':hashes[r],'forecast_freeze_sha256':hashes[f],'new_native_calls_or_fits':0,
          'independent_mother_blocks':n,'fixed_steps_checked':n*LAGS[-1],'oracle_native_calls_audited':total,
          'scored_from_raw_and_rank_based_component_proxy':True})


def fixtures():
    from aam_camera_information import aperture_features,component_features
    positions = np.array([[1.45,4.35,0.],[1.45,4.35,0.],[4.45,4.35,0.],
                          [1.45,4.35,0.],[4.45,4.35,0.],[1.45,4.35,0.]])
    owner = np.array([0,-1,0,-1,-1,-1,0,-1,1,1,2,-1])
    s = SimpleNamespace(points=positions,owner=owner,present=np.array([1,1,1,1,1,0],bool),available=owner<0,
        counts=np.array([0,2]),edges=np.array([[0,1],[0,3],[0,1]],int),lifts=np.array([[0,0,0],[0,0,0],[1,0,0]],int))
    packet = (s,[3],[0.,0.,6.])
    expected = [[2./4.17,2./3.,2.17/4.17],[0.,0.,0.]]
    close(aperture_features(packet),expected)
    close(independent_apertures(packet),expected)
    s.counts[1] = 3
    expected[0][1] = 0.
    close(aperture_features(packet),expected)
    close(independent_apertures(packet),expected)
    s.points[:,0] = (s.points[:,0]+3.) % 6.
    close(aperture_features(packet),expected[::-1])
    close(independent_apertures(packet),expected[::-1])
    from aam_camera_information import energy
    x = np.array([[0.],[1.]])
    close(energy(x,np.array([.5,.5]),np.array([1.])),.25)
    p = np.array([[.8,.2],[.2,.8]])
    distinct = np.abs(x-x.T)
    e = [.25*np.sum(np.linalg.matrix_power(p,k)*distinct) for k in (0,1,2,4)]
    require(e == sorted(e) and e[-1] < .25, 'Stationary binary score bound implementation failed.')
    print('PASS fixture-only: physical ratios, absent material, winding cycle, port translation, weighted ES and finite-lag bound.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--out',type=Path)
    group.add_argument('--fixtures',action='store_true')
    args = parser.parse_args()
    fixtures() if args.fixtures else audit(args.out.resolve(strict=True))
