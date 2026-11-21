"""Next-camera mark proposal for the explicit role-addressed BD/SSA law."""
from time import perf_counter

import numpy as np

from aam_first_interval_paired_fork import FirstIntervalSource, MARK_CELLS, MAX_ATTEMPTS_PER_SLAB
from aam_formation_actual_rollout import _packet_copy
from aam_formation_cycle_core import _physical_config
from aam_formation_local_initial import build_packet
from aam_reaction_innovations import (_advance_dwell, _begin_slab, _candidate_law, _finish_slab,
                                     _mark_index, _move_with_noise, _slab_values, _waiting_time, apply_event)
from aam_reaction_mark_involution import _first_at_least
from aam_network_events import CAP
from aam_snapshot_process import PORTS, WINDOW_Y, likelihood, observe, weight


SCHEMA = 'aam-role-addressed-three-camera-mark-tilt-v1'
MAX_TRIALS = 32
GUIDE_FLOOR = .125
GUIDE_STRENGTH = .25


def mark_cell_counts(cumulative, cells=MARK_CELLS):
    """Count exact original floating CDF cells, including duplicate boundaries."""
    c = np.asarray(cumulative, float)
    if (type(cells) is not int or not 1 <= cells <= MARK_CELLS or cells & (cells-1)
            or c.ndim != 1 or not len(c) or not np.isfinite(c).all()
            or c[0] < 0. or c[-1] <= 0. or np.any(np.diff(c) < 0.)):
        raise ValueError('Use an original positive monotone finite CDF and binary mark grid.')
    total = float(c[-1])
    edges = np.ceil((c/total)*cells).astype(np.int64)
    # Candidate brackets are checked using the exact original multiply order.
    # Correct the common one-cell error together, preserving exact fallback.
    for _ in range(2):
        lo = (edges > 0) & (((edges-1)/cells)*total >= c)
        hi = (edges < cells) & ((edges/cells)*total < c)
        edges += hi.astype(np.int64)-lo.astype(np.int64)
    bad = ((edges > 0) & (((edges-1)/cells)*total >= c)) | ((edges < cells) & ((edges/cells)*total < c))
    for i in np.flatnonzero(bad):
        edges[i] = _first_at_least(float(c[i]),total,cells)
    if edges[-1] != cells or np.any(np.diff(edges) < 0):
        raise ArithmeticError('No complete mark partition with the actual simulator floating product.')
    return np.diff(np.r_[0,edges])


def finite_tilt(cell_counts, log_scores, *, cells=MARK_CELLS, max_trials=MAX_TRIALS):
    """Marginal q/p after capped guided native draws plus one native fallback."""
    m, g = np.asarray(cell_counts,np.int64),np.asarray(log_scores,float)
    if (m.ndim != 1 or m.shape != g.shape or not np.isfinite(g).all() or np.any(m < 0)
            or int(m.sum()) != cells or not 1 <= max_trials <= MAX_TRIALS):
        raise ValueError('Need original grid counts and one fixed finite score per mark.')
    positive = m > 0
    log_guide = np.minimum(g-g[positive].max(),0.)
    thresholds = np.ceil(cells*(GUIDE_FLOOR+(1.-GUIDE_FLOOR)*np.exp(log_guide))).astype(np.int64)
    b = thresholds/cells
    average = min(1.,float((m/cells)@b))
    failed = (1.-average)**max_trials
    # The denominator C cancels in p/q. Unsupported rows stay unsupported.
    ratio = (1.-failed)*b/average + failed
    return thresholds, ratio, average, failed


def candidate_camera_scores(packet, rows, observed, sigma):
    """Instantaneous camera conversion change at the legal target molecule."""
    state,material,geometry,*_ = packet
    rows = np.asarray(rows)
    aam = int(material[0])
    spatial = np.stack([weight(state.points,x,WINDOW_Y[0],geometry[2])[:aam] for x in PORTS])
    denominator = np.maximum(spatial.sum(axis=1),1.e-12)
    valid = np.isin(rows[:,0],(1,3)) & (rows[:,2]//2 < aam)
    change = np.zeros((len(rows),2))
    if np.any(valid):
        v = rows[valid,2]
        if np.any(v < 0) or not state.available[v].all() or (v % 2).any():
            raise ValueError('An AAm conversion proposal must consume the original unused AAm site.')
        change[valid] = (spatial[:,v//2]/denominator[:,None]).T
    now = (spatial@(state.owner[:2*aam:2] >= 0))/denominator
    residual = np.asarray(observed,float)[::2]-now
    noise = np.asarray(sigma,float)[::2]
    return GUIDE_STRENGTH*np.sum((residual*change-.5*change**2)/(noise**2),axis=1)


def make_interval_table(config, camera, brownian, waiting, mark):
    """Preserve the canonical truncated slab layout and fixed role streams."""
    clock = config['prefix_clock']
    if camera not in (0,1,2) or not clock[-1] < config['recipe']['feed_time']:
        raise ValueError('The role-addressed adapter covers the three original no-feed prefix cameras.')
    now,end = clock[camera:camera+2]
    step = float(config['motion']['step'])
    if not np.isfinite(step) or step <= 0.:
        raise ValueError('Require the original positive Brownian step.')
    slabs = 0
    while now < end:
        now = min(now+step,end)
        slabs += 1
    particles = config['recipe']['aam']+config['recipe']['mbaa']
    return {'brownian':brownian.standard_normal((slabs,2,particles,3)),
            'waiting':waiting.exponential(size=(slabs,MAX_ATTEMPTS_PER_SLAB)),
            'mark':mark.integers(0,MARK_CELLS,size=(slabs,MAX_ATTEMPTS_PER_SLAB),dtype=np.int64)}


def _select(cumulative, scores, base_mark, retry, max_trials):
    counts = mark_cell_counts(cumulative)
    thresholds,ratio,mean,failed = finite_tilt(counts,scores,max_trials=max_trials)
    trials = []
    cell = base_mark
    for attempt in range(max_trials):
        if attempt:
            cell = int(retry.integers(MARK_CELLS,dtype=np.int64))
        k = int(_mark_index(cumulative,cell/MARK_CELLS))
        u = int(retry.integers(MARK_CELLS,dtype=np.int64))
        success = bool(u < thresholds[k])
        trials.append({'cell':cell,'row':k,'uniform53':u,'guided_accepted':success})
        if success:
            break
    else:
        cell = int(retry.integers(MARK_CELLS,dtype=np.int64))
        k = int(_mark_index(cumulative,cell/MARK_CELLS))
    result = {'trials':trials,'fallback':not success,'selected_cell':cell,'selected_row':k,
              'target_cells':int(counts[k]),'selected_threshold':int(thresholds[k]),
              'average_guide':mean,'failure_mass':failed,'q_over_p':float(ratio[k]),
              'log_p_over_q':float(-np.log(ratio[k]))}
    return k,result,counts,thresholds


def mark_only_interval(packet, clock, camera, table, *, measured=None, noise=None, retry_seed=None,
                       max_trials=MAX_TRIALS, record_full=False, counts=None):
    """Advance original BD/SSA with an explicitly separate mark-choice law."""
    state,material,geometry,rates,factors = packet
    clock = np.asarray(clock,float)
    if (camera not in (0,1,2) or len(clock) != camera+2 or state.counts[6]
            or state.position[0] != clock[camera] or state.counts[2] != camera or material[3] <= clock[-1]):
        raise ValueError('Begin at the correct unchanged canonical no-feed camera boundary.')
    guided = measured is not None
    if guided:
        y,sigma = np.asarray(measured,float),np.asarray(noise,float)
        if (y.shape != (4,) or sigma.shape != (4,) or not np.isfinite(y).all()
                or not np.isfinite(sigma).all() or np.any(sigma <= 0.) or retry_seed is None
                or not 1 <= max_trials <= MAX_TRIALS):
            raise ValueError('Use an original noisy camera row, noise and separate addressed retry entropy.')
    elif retry_seed is not None:
        raise ValueError('Unmodified controls consume no proposal randomness.')
    if counts is None:
        counts = {'attempted':0,'completed':0,'failed':0}
    counts['attempted'] += 1
    try:
        result = _advance_interval(packet,clock,camera,table,measured,noise,retry_seed,max_trials,record_full)
    except Exception:
        counts['failed'] += 1
        raise
    counts['completed'] += 1
    return result


def _advance_interval(packet, clock, camera, table, measured, noise, retry_seed, max_trials, record_full):
    state,material,geometry,rates,factors = packet
    present_count = len(state.points)
    width = 2*present_count
    capacity = width*width+2*width
    rows = np.empty((capacity,6),np.int64)
    distances, primitive, weights = (np.empty(capacity) for _ in range(3))
    risks = np.empty((3,CAP+1))
    mobility = np.full(present_count,geometry[1])
    mobility[int(material[0]):] *= .8
    source = FirstIntervalSource(table)
    slab, ledger, log_ratio = 0, [], 0.
    started = perf_counter()
    while state.position[0] < clock[-1]:
        if not _begin_slab(state,clock,material,geometry):
            raise ValueError('An original Brownian slab must open at a completed boundary.')
        start,end,half = _slab_values(state)
        present = np.flatnonzero(state.present)
        _move_with_noise(state,half,mobility,geometry,source.normal(present,slab,0,start,end))
        cache = np.full((present_count,present_count),-1,np.int8)
        while state.position[2] < end:
            n,A,cumulative = _candidate_law(state,material,geometry,rates,factors,cache,present,
                                             rows,distances,primitive,weights,risks)
            local = float(state.position[2])
            if not np.isfinite(A) or A < 0.:
                raise ValueError('Original primitive hazard changed domain.')
            attempt = source.attempts[slab]
            wait = source.exponential(slab,local,end) if A > 0. else None
            reacted = _waiting_time(local,wait,A,1.) if wait is not None else np.inf
            endpoint,_ = _advance_dwell(state,reacted,end,A)
            receipt = {'slab':slab,'attempt':attempt,'begin':local,'end':endpoint,'exponential':wait,
                       'target_A':A,'censored':bool(reacted >= end)}
            ledger.append(receipt)
            if receipt['censored']:
                break
            if n == 0 or cumulative[-1] <= 0.:
                raise ValueError('A realized wait requires original positive mark support.')
            u = source.uniform(slab,local,reacted)
            k = int(_mark_index(cumulative,u))
            detail,counts,limits,score = None,None,None,None
            if measured is not None:
                score = candidate_camera_scores(packet,rows[:n],measured,noise)
                if np.any(score != score[0]):
                    retry = np.random.default_rng(np.random.SeedSequence([retry_seed,camera,slab,attempt]))
                    k,detail,counts,limits = _select(cumulative,score,int(u*MARK_CELLS),retry,max_trials)
                    log_ratio += detail['log_p_over_q']
            if k >= n or weights[k] <= 0.:
                raise ValueError('Guide may choose only a native valid positive event.')
            receipt.update(uniform=u,chosen=k,geometry_time=start+half,mark_tilt=detail)
            if record_full:
                receipt.update(candidate_rows=rows[:n].copy(),candidate_distance=distances[:n].copy(),
                               native_mark_weight=weights[:n].copy(),actual_candidate_cumulative=cumulative.copy(),
                               guide_score=score,target_cells_all=counts,guide_thresholds_all=limits)
            apply_event(state,rows[k],distances[k],geometry[2],reacted,start+half)
        _finish_slab(state,half,mobility,geometry,source.normal(np.flatnonzero(state.present),slab,1,start,end))
        slab += 1
    if slab != len(table['brownian']) or state.position[0] != clock[-1] or state.counts[6]:
        raise ArithmeticError('The complete adapter did not close at the original camera boundary.')
    return {'packet':packet,'spans':ledger,'tape':source.finish(),'native_interval_calls':1,
            'log_target_over_proposal':float(log_ratio),'seconds':perf_counter()-started,
            'original_waiting_and_brownian':True,'schema':SCHEMA}


def forward_prefix(config, initial, factors, tables, *, measured=None, guided=False, retry_seeds=None,
                   max_trials=MAX_TRIALS, record_full=False):
    """Complete legal prefix; exact correction is raw log camera density plus log p/q."""
    if len(tables) != 3 or guided != (retry_seeds is not None) or (guided and len(retry_seeds) != 3):
        raise ValueError('Use exactly three original interval tables and separate tilted retry provenance.')
    if measured is None and guided:
        raise ValueError('The guide must condition only on the supplied noisy camera query.')
    actual = _physical_config(config,factors)
    y = None if measured is None else np.asarray(measured,float)
    sigma = np.asarray(config['noise'],float)
    if y is not None and (y.shape != (3,4) or not np.isfinite(y).all()):
        raise ValueError('Expected three finite original camera readouts.')
    packet = build_packet(actual,initial,factors=factors)
    previous = [_packet_copy(packet,readonly=True)]
    clean,receipts = [],[]
    counts = {'attempted':0,'completed':0,'failed':0}
    error = None
    for camera in range(3):
        try:
            row = mark_only_interval(packet,config['prefix_clock'][:camera+2],camera,tables[camera],
               measured=y[camera] if guided else None,noise=sigma if guided else None,
               retry_seed=retry_seeds[camera] if guided else None,max_trials=max_trials,
               record_full=record_full,counts=counts)
            receipts.append({k:v for k,v in row.items() if k!='packet'})
            previous.append(_packet_copy(packet,readonly=True))
            clean.append(observe(previous[-1]))
        except Exception as exc:
            error = {'camera':camera,'type':type(exc).__name__,'message':str(exc)}
            break
    clean = np.asarray(clean,float).reshape(-1,4)
    path_log = float(sum(r['log_target_over_proposal'] for r in receipts))
    log_camera = None if y is None or error is not None else float(likelihood(clean,y,sigma).sum())
    return {'schema':SCHEMA,'valid':error is None,'error':error,'guided':guided,
            'checkpoint_packets':previous,'clean':clean,'intervals':receipts,'cost':counts,
            'log_target_over_proposal':path_log,'log_camera':log_camera,
            'log_weight':None if log_camera is None else log_camera+path_log}
