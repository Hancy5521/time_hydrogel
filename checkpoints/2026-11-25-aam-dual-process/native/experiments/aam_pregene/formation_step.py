"""Pre-gene AAm/MBAA solver refinement with paired, explicit innovations."""

import argparse
import hashlib
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timedelta, timezone
from multiprocessing import get_context
from pathlib import Path
from time import process_time
from uuid import uuid4

import numpy as np
from numba import njit
from threadpoolctl import threadpool_limits


CORE = Path(__file__).resolve().parents[2] / 'checkpoints/2026-11-21-aam-integrated-process/source'
sys.path.insert(0, str(CORE))
from aam_edge_fork import add_feed, apply_event, candidates
from aam_formation_cycle_core import make_config
from aam_formation_local_initial import build_packet, draw_initial
from aam_formation_micro_history import extract_state
from aam_network_events import CAP, factor_weights, pendant_kind
from aam_spatial_motion import _compact_edges, _scalar_distance, spatial_forces


TICK = .025 / 32
TIMES = np.array([.65, 1.35, 2.05, 3.25, 4.85, 7.20])
AGES = np.rint(TIMES / TICK).astype(np.int64)
FEED = 2.30
FEED_TICK = round(FEED / TICK)
MODES = ('fixed_025', 'fixed_0125', 'fixed_00625', 'contact_adaptive', 'reference_003125', 'finer_0015625')
STEPS = (32, 16, 8, 0, 4, 2)
MOTHERS, FINER = 256, 64
RISK_TOL = .085
BOOT = 6000
NAMES = ('conversion', 'Ln', 'Lw', 'largest', 'cycle_rank', 'bridge', 'same_chain_ring',
         'different_chain_same_component', 'pendant_remaining', 'winding')
SCALE = np.array([.15, .07, .10, .18, .20, .18, .18, .18, .23, .45])


@njit(cache=True)
def pair_switch_cost(r, capture, inner, variance, inner_rate, outer_rate):
    score = 0.
    for border, jump in ((inner, abs(inner_rate-outer_rate)), (capture, outer_rate)):
        gap = r-border
        if gap*gap < 18.*variance:
            score += jump*np.exp(-.5*gap*gap/variance)
    return score


@njit(cache=True)
def contact_step(state, aam, geometry, rates, span):
    """Choose on the current state, before requesting any future noise."""
    active = np.flatnonzero(state.active)
    if active.size == 0:
        return span
    vacant = np.flatnonzero(state.available)
    box, capture, inner = geometry[2], geometry[7], geometry[8]
    # For a time midpoint dt/2 the 1-D relative variance is (D_i+D_j)*dt.
    # 2D is a common bounding scale because MBAA's mobility is .8D.
    while span > 4:
        dt = TICK*span
        variance = 2.*geometry[1]*dt
        if variance <= 0.:
            return span
        mass = 0.
        for u in active:
            for v in vacant:
                i,j=u//2,v//2
                if i==j:
                    continue
                r = _scalar_distance(state.points,i,j,box)
                row = 2 if j<aam else 3
                if state.owner[v^1] >= 0:
                    row = 4 if pendant_kind(state.labels,state.owner,u,v)==0 else 5
                mass += pair_switch_cost(r,capture,inner,variance,rates[row,0],rates[row,1])
        for k in range(len(active)-1):
            for l in range(k+1,len(active)):
                i,j=active[k]//2,active[l]//2
                if i==j:
                    continue
                r = _scalar_distance(state.points,i,j,box)
                mass += pair_switch_cost(r,capture,inner,variance,rates[6,0]+rates[7,0],rates[6,1]+rates[7,1])
        if dt*mass <= RISK_TOL:
            return span
        span = span//2
        span -= span%2
    return max(2,span)


@njit(cache=True)
def move(state, a, b, brownian, mobility, geometry):
    ids = np.flatnonzero(state.present)
    box, spring, rest, repulsion, core = geometry[2:7]
    bonds = _compact_edges(state.edges[:state.counts[1]],state.present)
    forces,_ = spatial_forces(state.points[state.present],bonds,box,spring,rest,repulsion,core)
    dt = TICK*(b-a)
    for j,i in enumerate(ids):
        for k in range(3):
            dx = dt*mobility[i]*forces[j,k] + np.sqrt(2.*mobility[i])*(brownian[b,i,k]-brownian[a,i,k])
            state.points[i,k] = (state.points[i,k]+dx)%box
            state.unwrapped[i,k] += dx


@njit(cache=True)
def evolve_to(packet, start, end, brownian, expo, uniform, event_index, remaining, fixed):
    state, material, geometry, rates, factors = packet
    aam, initial = int(material[0]), int(material[2])
    if start%2 or end%2:
        raise ValueError('Use even boundaries to represent the midpoint exactly.')
    box, capture, inner, corridor = geometry[2],geometry[7],geometry[8],geometry[9]
    mobility = np.full(len(state.points), geometry[1]); mobility[aam:] *= .8
    width = 2*len(state.points)
    size = width*width + 2*width
    rows = np.empty((size,6),np.int64)
    distances, primitive, weights = np.empty(size), np.empty(size), np.empty(size)
    risk = np.empty((3,CAP+1))
    cumulative = np.empty(size)
    at, slabs, queries = start, 0, 0
    while at < end:
        if at==FEED_TICK and not state.present.all():
            add_feed(state,aam+initial,FEED)
        span = min(end-at,32 if fixed==0 else fixed)
        if at < FEED_TICK < at+span:
            span = FEED_TICK-at
        if fixed==0:
            span = contact_step(state,aam,geometry,rates,span)
        half = at + span//2
        stop = at+span
        move(state,at,half,brownian,mobility,geometry)
        cache = np.full((len(state.points),len(state.points)),-1,np.int8)
        present = np.flatnonzero(state.present)
        local, terminal = TICK*at,TICK*stop
        while local<terminal:
            count = candidates(state,aam,box,capture,inner,corridor,rates,cache,present,rows,distances,primitive)
            queries += 1
            hazard = primitive[:count].sum()
            exposure = hazard*(terminal-local)
            if hazard<=0. or remaining>=exposure:
                remaining = max(0.,remaining-exposure)
                state.position[3] += exposure
                break
            local += remaining/hazard
            state.position[3] += remaining
            factor_weights(primitive,rows[:,4],rows[:,5],count,factors,weights,risk)
            total = 0.
            for i in range(count):
                total += weights[i]
                cumulative[i] = total
            if total<=0. or event_index+1>=len(expo):
                raise ValueError('No positive mark law or finite innovation budget exceeded.')
            mark = np.searchsorted(cumulative[:count],uniform[event_index]*total,side='right')
            if mark>=count or weights[mark]<=0.:
                raise ValueError('Waiting event without legal positive mark.')
            apply_event(state,rows[mark],distances[mark],box,local,TICK*half)
            event_index += 1
            remaining = expo[event_index]
        move(state,half,stop,brownian,mobility,geometry)
        at = stop
        slabs += 1
    if end==FEED_TICK and not state.present.all():
        add_feed(state,aam+initial,FEED)
    state.position[:3] = TICK*end
    return event_index, remaining, slabs, queries


def independent_winding(edges, shifts, size):
    neighbors = [[] for _ in range(size)]
    for (i, j), shift in zip(edges, shifts, strict=True):
        neighbors[i].append((j, shift))
        neighbors[j].append((i, -shift))
    offsets = {}
    winding = np.zeros(3, dtype=bool)
    for root in range(size):
        if root in offsets:
            continue
        offsets[root] = np.zeros(3, dtype=np.int64)
        queue = [root]
        for i in queue:
            for j, shift in neighbors[i]:
                proposal = offsets[i] + shift
                if j in offsets:
                    winding |= offsets[j] != proposal
                else:
                    offsets[j] = proposal
                    queue.append(j)
    return winding


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()


def bindings():
    deps = ['aam_edge_fork.py','aam_formation_cycle_core.py','aam_formation_local_initial.py',
            'aam_formation_micro_history.py','aam_formation_cycle_data.py','aam_network_events.py',
            'aam_network_path_metrics.py','aam_spatial_motion.py','aam_mbaa_kinetics.py',
            'aam_contact_environment_study.py','aam_contact_environment.py','aam_network_competition_study.py',
            'aam_snapshot_process.py','aam_formation_actual_rollout.py','aam_reaction_innovations.py']
    return {str(p.resolve()):sha(p) for p in (Path(__file__),*(CORE/n for n in deps))}


def stamp():
    utc=datetime.now(timezone.utc)
    return {'utc':utc.isoformat(),'asia_shanghai':utc.astimezone(timezone(timedelta(hours=8))).isoformat()}


def write(path,value):
    with path.open('x',encoding='utf8') as f:
        json.dump(value,f,indent=2,allow_nan=False)


def read(path):
    return json.loads(path.read_text(encoding='utf8'))


def register(out):
    cfg=make_config()
    cfg['recipe']['feed_time']=FEED
    out.mkdir(exist_ok=False)
    write(out/'REGISTERED.json',{**stamp(),'uuid':str(uuid4()),'sources':bindings(),'config':cfg,
        'modes':MODES,'fixed_steps_ticks':STEPS,'tick':TICK,'times':TIMES.tolist(),'features':NAMES,
        'scale':SCALE.tolist(),'risk_tolerance':RISK_TOL,'mothers':MOTHERS,'finer_first':FINER,
        'primary_topology': ['largest', 'winding'],
        'scope':'A fixed rate and factor table, no gene algorithm. Explicit IID Brownian/reaction streams, common per mother. Shared time coupling changes old uint32-to-path mapping. Numerics only, no experimental calibration.',
        'decision':'Preserve every candidate. Default should change only if an alternative improves path error against the .003125 benchmark, shows aligned mean readouts, and its cost is transparent. Finer .0015625 sensitivity on prespecified first 64.',
        'inference':'Paired bootstrap at mother level. Report pathwise errors, ensemble mean error, timed effort and finer-reference comparison separately.'})


def locked(out):
    r=read(out/'REGISTERED.json')
    if r['sources']!=bindings():
        raise ValueError('Frozen source binding changed.')
    return r


def get_sample(task):
    idx, seed, cfg=task
    start,g,r,o=np.random.SeedSequence(int(seed)).spawn(4)
    initial=draw_initial(cfg,np.random.default_rng(start))
    rng=np.random.default_rng(g)
    dw=np.empty((int(AGES[-1])+1,78,3))
    dw[0]=0.
    dw[1:]=rng.standard_normal(dw[1:].shape)*np.sqrt(TICK)
    np.cumsum(dw,axis=0,out=dw)
    rng=np.random.default_rng(r)
    expo=rng.exponential(size=160)
    uniform=rng.random(160)
    limit=len(STEPS) if idx<FINER else len(STEPS)-1
    values=np.full((len(STEPS),len(TIMES),len(NAMES)),np.nan)
    aggregate=np.full((len(STEPS),3),np.nan)
    for j in np.random.default_rng(o).permutation(limit):
        p=build_packet(cfg,initial)
        begin,index,res=0,0,expo[0]
        elapsed, counts, evals=0.,0,0
        for i,age in enumerate(AGES):
            started=process_time()
            index,res,slabs,queries=evolve_to(p,begin,int(age),dw,expo,uniform,index,res,STEPS[j])
            elapsed+=process_time()-started
            counts+=slabs
            evals+=queries
            np.testing.assert_allclose(p[0].position[0],TIMES[i],rtol=0.,atol=1e-14)
            np.testing.assert_allclose(p[0].position[3],expo[:index].sum()+expo[index]-res,rtol=1e-12,atol=1e-12)
            micro,w=extract_state(p[0])
            n_edges=p[0].counts[1]
            np.testing.assert_array_equal(w,independent_winding(p[0].edges[:n_edges],p[0].lifts[:n_edges],78))
            values[j,i]=np.r_[p[0].counts[5]/92.,micro,w.any()]
            begin=int(age)
        aggregate[j]=[elapsed,counts,evals]
    return values,aggregate


def init_worker():
    cfg=make_config()
    seed=np.random.default_rng(21)
    packet=build_packet(cfg,draw_initial(cfg,seed))
    zero=np.zeros((9,78,3))
    evolve_to(packet,0,8,zero,np.ones(160),np.full(160,.5),0,1.,4)


def acquire(out,workers):
    reg=locked(out)
    if (out/'MOTHERS.npz').exists():
        raise FileExistsError('The independent cohort is already acquired.')
    seed=int.from_bytes(hashlib.sha256(reg['uuid'].encode()).digest()[:16],'little')
    ids=np.random.default_rng(seed).integers(0,2**63-1,MOTHERS)
    if len(np.unique(ids))!=MOTHERS:
        raise ValueError('Duplicate independent mother address.')
    with threadpool_limits(1), ProcessPoolExecutor(max_workers=workers,mp_context=get_context('spawn'),initializer=init_worker) as pool:
        records=[]
        for n,entry in enumerate(pool.map(get_sample,((j,s,reg['config']) for j,s in enumerate(ids)),chunksize=2),1):
            records.append(entry)
            if n%64==0:
                print(f'{n}/{MOTHERS} verified mothers',flush=True)
    v,t=(np.stack(x) for x in zip(*records,strict=True))
    f=out/'MOTHERS.npz'
    np.savez_compressed(f,ids=ids,values=v,counts_and_cpu=t)
    write(out/'FROZEN.json',{**stamp(),'sha256':sha(f),'register_sha256':sha(out/'REGISTERED.json'),
                             'native_conservation_checked_boundaries':int(np.isfinite(v).all(axis=-1).sum())})


def stats(x,indices):
    return {'mean':float(x.mean()),'ci':np.quantile(x[indices].mean(axis=1),[.025,.975]).tolist()}


def score(out):
    locked(out)
    freeze=read(out/'FROZEN.json')
    f=out/'MOTHERS.npz'
    if freeze['sha256']!=sha(f) or freeze['register_sha256']!=sha(out/'REGISTERED.json'):
        raise ValueError('Outcome bindings changed.')
    with np.load(f,allow_pickle=False) as n:
        v,t=n['values'],n['counts_and_cpu']
    loss=(np.abs(v[:,:4]-v[:,4:5])/SCALE).mean(axis=(2,3))
    rng=np.random.default_rng(72428)
    draw=rng.integers(len(loss),size=(BOOT,len(loss)))
    fine=(np.abs(v[:FINER,:5]-v[:FINER,5:6])/SCALE).mean(axis=(2,3))
    topology = {}
    for name in ('largest', 'winding'):
        j=NAMES.index(name)
        difference=v[:,:,: ,j]-v[:,4:5,:,j]
        errors=np.abs(difference).mean(axis=2)
        topology[name]={m:{'pathwise_delta_vs_coarse':stats(errors[:,i]-errors[:,0],draw),
                           'mean_signed_bias':stats(difference[:,i].mean(axis=1),draw)}
                        for i,m in enumerate(MODES[:4])}
    rec={**stamp(),'frozen_hash':freeze['sha256'],'n':len(v),'n_finer':FINER,'features':NAMES,'wet':0,
          'mae_mean_by_cohort_feature_and_time_vs_reference':np.abs(v[:,:5].mean(axis=0)-v[:,4].mean(axis=0)).tolist(),
          'terminal_mean_by_method':v[:,:5,-1].mean(axis=0).tolist(),
          'time_paired_abs_error':{m:stats(loss[:,i],draw) for i,m in enumerate(MODES[:4])},
          'paired_error_delta_vs_coarse':{m:stats(loss[:,i]-loss[:,0],draw) for i,m in enumerate(MODES[1:4],1)},
          'adaptive_delta_vs_half':stats(loss[:,3]-loss[:,1],draw),
          'topology':topology,
          'cost_cpu_slabs_candidate_calls':dict(zip(MODES,np.nanmean(t,axis=0).tolist(),strict=True)),
          'finer_subset_against_finer':dict(zip(MODES[:5],fine.mean(axis=0).tolist(),strict=True)),
          'numerical_audit':freeze['native_conservation_checked_boundaries']}
    write(out/'SCORED.json',rec)


if __name__=='__main__':
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('stage',choices=('register','acquire','score'))
    cli.add_argument('out',type=Path)
    cli.add_argument('--workers',type=int,default=14)
    a=cli.parse_args()
    if a.stage=='register': register(a.out.resolve())
    elif a.stage=='acquire': acquire(a.out.resolve(),a.workers)
    else: score(a.out.resolve())
