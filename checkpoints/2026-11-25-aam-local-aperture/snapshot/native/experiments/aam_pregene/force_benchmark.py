"""Frozen bit-parity and paired end-to-end timing for exact cell pruning."""

import argparse
import hashlib
from pathlib import Path
from time import perf_counter, process_time
from uuid import uuid4

import numpy as np
from threadpoolctl import threadpool_limits

from advance_fixed import advance_fixed
from force_cells import cell_forces, spatial_forces
from formation_step import (
    AGES, FEED, TICK, bindings, build_packet, draw_initial, evolve_to,
    extract_state, independent_winding, make_config, read, sha, stamp, write,
)


MOTHERS = 64
STEPS = (32, 4)
METHODS = (spatial_forces, cell_forces)


def source_hashes():
    return bindings() | {str(p):sha(p) for p in
                         (Path(__file__).resolve(), Path(__file__).with_name('advance_fixed.py').resolve(),
                          Path(__file__).with_name('force_cells.py').resolve())}


def register(out):
    config = make_config()
    config['recipe']['feed_time'] = FEED
    out.mkdir(exist_ok=False)
    write(out/'REGISTERED.json', {**stamp(), 'uuid':str(uuid4()), 'sources':source_hashes(),
          'mothers':MOTHERS, 'steps':STEPS, 'tick':TICK, 'config':config,
          'claim':'A drop-in force change only. Every State field and reaction clock must have bit parity '
                  'on fresh paired rollouts. CPU and wall compare only forward advances, after JIT.',
          'adoption':'All parity must pass; upper 95% bound of median paired wall ratio below .85 at both '
                     'fixed grids. No chemical calibration or improved error is claimed.'})


def warm(config):
    initial = draw_initial(config, np.random.default_rng(2132145))
    brownian = np.zeros((33,78,3))
    waits = np.ones(160)
    marks = np.full(160,.4)
    reference = build_packet(config,initial)
    expected = evolve_to(reference,0,32,brownian,waits,marks,0,waits[0],4)[:2]
    for fn in METHODS:
        packet = build_packet(config,initial)
        result = advance_fixed(packet,0,32,brownian,waits,marks,0,waits[0],4,fn)
        if result != expected or any(a.tobytes()!=b.tobytes() for a,b in zip(reference[0],packet[0],strict=True)):
            raise ArithmeticError('Fixed solver parity failed against the archived sensitivity solver.')


def histories(seed, config):
    rng = np.random.default_rng(seed)
    initial = draw_initial(config,rng)
    brownian = np.empty((int(AGES[-1])+1,78,3))
    brownian[0] = 0.
    brownian[1:] = rng.standard_normal(brownian[1:].shape)*np.sqrt(TICK)
    np.cumsum(brownian,axis=0,out=brownian)
    waits, marks = rng.exponential(size=160), rng.random(160)
    results = np.empty((len(STEPS),len(METHODS),2))
    hashes = np.empty((len(STEPS),len(METHODS),len(AGES)),dtype='S64')
    for j in rng.permutation(len(STEPS)):
        for k in rng.permutation(len(METHODS)):
            packet = build_packet(config,initial)
            state = packet[0]
            begin,index,remainder = 0,0,waits[0]
            elapsed = np.zeros(2)
            for i,age in enumerate(AGES):
                cpu,wall = process_time(),perf_counter()
                index,remainder = advance_fixed(packet,begin,int(age),brownian,waits,marks,index,remainder,
                                                 STEPS[j],METHODS[k])
                elapsed += (process_time()-cpu,perf_counter()-wall)
                _, wind = extract_state(state)
                edges = state.counts[1]
                np.testing.assert_array_equal(wind,independent_winding(state.edges[:edges],state.lifts[:edges],78))
                h = hashlib.sha256()
                for field in state:
                    h.update(field.tobytes())
                h.update(np.int64(index).tobytes())
                h.update(np.float64(remainder).tobytes())
                hashes[j,k,i] = h.hexdigest()
                begin = int(age)
            results[j,k] = elapsed
    return results, hashes


def acquire(out, reg):
    if (out/'PAIRED.npz').exists():
        raise FileExistsError('This fresh cohort has already been acquired.')
    cohort = int.from_bytes(hashlib.sha256(reg['uuid'].encode()).digest()[:16],'little')
    seeds = np.random.default_rng(cohort).integers(0,2**63-1,MOTHERS)
    if len(np.unique(seeds)) != MOTHERS:
        raise ValueError('Duplicate mother stream.')
    with threadpool_limits(1):
        warm(reg['config'])
        entries = []
        for seed in seeds:
            entries.append(histories(seed,reg['config']))
            if len(entries)%16==0:
                print(f'{len(entries)}/{MOTHERS} timed mothers',flush=True)
    elapsed, pairs = (np.stack(x) for x in zip(*entries,strict=True))
    file = out/'PAIRED.npz'
    np.savez_compressed(file,seeds=seeds,elapsed=elapsed,state_hashes=pairs)
    write(out/'FROZEN.json',{**stamp(),'sha256':sha(file),'registered_sha256':sha(out/'REGISTERED.json')})


def score(out):
    frozen = read(out/'FROZEN.json')
    file = out/'PAIRED.npz'
    if sha(file) != frozen['sha256'] or sha(out/'REGISTERED.json') != frozen['registered_sha256']:
        raise ValueError('Invalid frozen timing binding.')
    with np.load(file,allow_pickle=False) as data:
        elapsed, pairs = data['elapsed'], data['state_hashes']
    parity = pairs[:,:,0] == pairs[:,:,1]
    wall_ratio = elapsed[:,:,1,1]/elapsed[:,:,0,1]
    idx = np.random.default_rng(171282).integers(len(pairs),size=(6000,len(pairs)))
    intervals = np.quantile(np.median(wall_ratio[idx],axis=1),[.025,.975],axis=0)
    write(out/'SCORED.json',{**stamp(),'paired_sha256':frozen['sha256'],'steps':STEPS,'step_dt':[i*TICK for i in STEPS],
          'matched_boundaries':int(parity.sum()),'compared_boundaries':int(parity.size),
          'mean_cpu_and_wall':elapsed.mean(axis=0).tolist(),
          'median_wall_ratio':np.median(wall_ratio,axis=0).tolist(),'median_wall_ratio_ci':intervals.T.tolist(),
          'adopt':bool(parity.all() and np.all(intervals[1]<.85))})


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('stage',choices=('register','acquire','score'))
    cli.add_argument('out',type=Path)
    args = cli.parse_args()
    if args.stage == 'register':
        register(args.out)
    else:
        reg = read(args.out/'REGISTERED.json')
        if reg['sources'] != source_hashes():
            raise ValueError('Frozen source binding changed.')
        if args.stage == 'acquire':
            acquire(args.out,reg)
        else:
            score(args.out)
