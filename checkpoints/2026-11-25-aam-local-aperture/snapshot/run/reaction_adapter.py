"""Frozen native AAm field, local observation and prospective path readouts."""

from __future__ import annotations

from functools import lru_cache
from hashlib import file_digest
import os
from pathlib import Path
import sys

from numba import njit
import numpy as np


DEFAULT_PROJECT = Path(__file__).resolve().parents[1] / "native"
PROJECT = Path(os.environ.get("AAM_REACTION_WORKTREE", str(DEFAULT_PROJECT))).resolve()
PREG = PROJECT / "experiments/aam_pregene"
if not (PREG / "advance_fixed.py").is_file():
    raise ImportError(f"AAM_REACTION_WORKTREE does not name the native worktree: {PROJECT}")
sys.path.insert(0, str(PREG))

from advance_fixed import advance_fixed
from force_benchmark import source_hashes
from force_cells import cell_forces
from formation_step import NAMES as PATH_NAMES
from formation_step import TICK, build_packet, draw_initial, extract_state, make_config
from gene_field import BOUND, COORDINATES, decode


EARLY_TIMES = (.65, 1.35, 2.05)
TIMES = (*EARLY_TIMES, 3.50, 4.85, 6.)
LEVELS = (60, 76)
MILESTONE_NAMES = ("reached", "time_or_6", "Lw", "largest", "winding")
LOCAL_NAMES = (
    "visible_aam", "visible_mbaa", "aam_conversion", "mba_site_conversion",
    "mba_double", "induced_bond_density", "induced_largest", "induced_lw",
    "isolated_particle_share", "internal_redundancy", "near_pairs",
    "x_patch_mass", "y_patch_mass", "z_patch_mass",
)
STEP = 32
FIELD_SIDE = .70
N_AAM, N_MBAA, N_TOTAL = 64, 14, 78
SITES = N_AAM + 2*N_MBAA
TAPE_LENGTH = 160

# Each pulse begins at the indicated time; unit values leave the native value unchanged.
# Initiation and mobility are external process inputs, independent of the fixed material code.
ENVIRONMENTS = {
    "A": {"feed": 2.85, "diffusion": .245, "pulses": ((0., 1.45, 1.), (.65, 1., 1.), (1.35, .65, 1.))},
    "B": {"feed": .55, "diffusion": .31, "pulses": ((0., .65, 1.), (.65, 1., 1.), (1.35, 1.45, 1.))},
    "C": {"feed": 1.95, "diffusion": .19, "pulses": ((0., 1., 1.), (1.775, 1., .34/.19))},
}


def _tick(time):
    value = round(time / TICK)
    if value % 2 or abs(value*TICK - time) > 1.e-12:
        raise ValueError("Every measurement, mobility switch and feed uses an even lattice boundary.")
    return value


@lru_cache(maxsize=None)
def _clock(environment, full):
    env = ENVIRONMENTS[environment]
    times = TIMES if full else EARLY_TIMES
    scheduled = {_tick(value): i for i, value in enumerate(times)}
    pulses = {_tick(value): (initiation, mobility)
              for value, initiation, mobility in env["pulses"] if value < times[-1]}
    stops = sorted({0, *scheduled, *pulses})
    feed = _tick(env["feed"])
    stages, knots = [], [0]
    initiated, mobile = 1., 1.
    for begin, end in zip(stops[:-1], stops[1:]):
        if begin in pulses:
            initiated, mobile = pulses[begin]
        stages.append((begin, end, initiated, mobile, scheduled.get(end)))
        tick = begin
        while tick < end:
            next_tick = min(tick+STEP, end)
            if tick < feed < next_tick:
                next_tick = feed
            knots.extend(((tick+next_tick)//2, next_tick))
            tick = next_tick
    return stages, np.array(knots, dtype=np.int64)


def _brownian_at_nodes(knots, generator):
    # The native routine only reads the registered slab boundary and midpoint nodes.
    # Independent Gaussian increments at these nodes have the same joint law as on the dense lattice.
    motion = np.empty((int(knots[-1])+1, N_TOTAL, 3))
    motion[0] = 0.
    jumps = generator.standard_normal(size=(len(knots)-1, N_TOTAL, 3))
    jumps *= np.sqrt(TICK*np.diff(knots))[:, None, None]
    motion[knots[1:]] = np.cumsum(jumps, axis=0)
    return motion


@njit(cache=False)
def _root(parent, node):
    while parent[node] != node:
        node = parent[node]
    return node


@njit(cache=False)
def _inside_labels(edges, visible):
    parent = np.arange(len(visible))
    edge_count = 0
    for a, b in edges:
        if visible[a] and visible[b]:
            parent[_root(parent, b)] = _root(parent, a)
            edge_count += 1
    return np.array([_root(parent, i) for i in range(len(visible))]), edge_count


def _local_observation(state, origin, box, cutoff):
    clearance = np.remainder(state.points - origin, box)
    side = FIELD_SIDE * box
    visible = state.present & np.all(clearance < side, axis=1)
    ids = np.flatnonzero(visible)
    if not len(ids):
        return np.zeros(len(LOCAL_NAMES))
    labels, bonds = _inside_labels(state.edges[:state.counts[1]], visible)
    sizes = np.bincount(labels[visible], minlength=N_TOTAL)
    usage = (state.owner.reshape(N_TOTAL, 2) >= 0).sum(axis=1)
    va, vb = int(visible[:N_AAM].sum()), int(visible[N_AAM:].sum())
    shift = state.points[ids][:, None] - state.points[ids][None, :]
    shift -= box * np.rint(shift / box)
    near = int(np.triu(np.square(shift).sum(axis=-1) < cutoff*cutoff, 1).sum())
    faces = []
    for axis in range(3):
        low = np.unique(labels[visible & (clearance[:, axis] < .25*side)])
        high = np.unique(labels[visible & (clearance[:, axis] > .75*side)])
        faces.append(float(sizes[np.intersect1d(low, high, assume_unique=True)].sum()) / len(ids))
    return np.array([
        va/N_AAM, vb/N_MBAA, usage[:N_AAM][visible[:N_AAM]].sum()/max(va, 1),
        usage[N_AAM:][visible[N_AAM:]].sum()/(2*max(vb, 1)),
        (usage[N_AAM:][visible[N_AAM:]] == 2).sum()/max(vb,1),
        bonds/len(ids), sizes.max()/len(ids), np.square(sizes).sum()/len(ids)**2,
        (sizes == 1).sum()/len(ids), (bonds-len(ids)+np.count_nonzero(sizes))/len(ids),
        near/len(ids), *faces,
    ])


@njit(cache=False)
def _find_lift(parent, moves, node):
    lift = np.zeros(3, np.int64)
    while parent[node] != node:
        lift += moves[node]
        node = parent[node]
    return node, lift


@njit(cache=False)
def _milestones(events):
    result = np.zeros((len(LEVELS), len(MILESTONE_NAMES)))
    result[:, 1] = TIMES[-1]
    owners = np.full(2*N_TOTAL, -1, dtype=np.int64)
    parent = np.arange(N_TOTAL)
    moves = np.zeros((N_TOTAL, 3), dtype=np.int64)
    next_owner, used, level = 0, 0, 0
    winding = False
    for row in events:
        tag, start, stop = int(row[1]), int(row[2]), int(row[3])
        if tag == 0:
            continue
        if start >= 0:
            lineage = owners[start]
        else:
            lineage = next_owner
            next_owner += 1
        if start >= 0 and tag != 8:
            a, da = _find_lift(parent, moves, start//2)
            b, db = _find_lift(parent, moves, stop//2)
            dz = row[10:13].astype(np.int64) + da - db
            if a == b:
                if np.any(dz):
                    winding = True
            else:
                parent[b] = a
                moves[b] = dz
        if tag not in (7, 8):
            if lineage < 0 or owners[stop] >= 0:
                raise ValueError("Duplicate or unsupported site consumption in native journal.")
            owners[stop] = lineage
            used += 1
            if level < len(LEVELS) and used == LEVELS[level]:
                lengths = np.zeros(N_TOTAL, dtype=np.int64)
                masses = np.zeros(N_TOTAL, dtype=np.int64)
                for i in range(N_TOTAL):
                    if owners[2*i] >= 0 or owners[2*i+1] >= 0:
                        masses[_root(parent, i)] += 1
                    if i < N_AAM and owners[2*i] >= 0:
                        lengths[owners[2*i]] += 1
                result[level, 0] = 1.
                result[level, 1] = row[0]
                result[level, 2] = (lengths*lengths).sum()/max(lengths.sum(),1)/N_AAM
                result[level, 3] = masses.max()/N_TOTAL
                result[level, 4] = winding
                level += 1
    return result, used


def sources():
    hashes = source_hashes()
    for path in (Path(__file__).resolve(), PREG/"gene_field.py"):
        with path.open("rb") as stream:
            hashes[str(path)] = file_digest(stream, "sha256").hexdigest()
    return hashes


def simulate(code, environment, seed, *, full=False, journal=False):
    """Advance a fresh native state. No full field is returned for an early-only call."""
    if environment not in ENVIRONMENTS:
        raise ValueError(f"Unknown environment {environment!r}")
    if journal and not full:
        raise ValueError("Only a full-label call can release the event history.")
    g = np.asarray(code, dtype=float)
    if g.shape != (len(COORDINATES),) or not np.isfinite(g).all() or np.any(np.abs(g) > BOUND):
        raise ValueError("Code must have four valid coordinates in its preregistered domain.")
    setting = ENVIRONMENTS[environment]
    config = make_config()
    config["recipe"]["feed_time"] = setting["feed"]
    config["motion"]["diffusion"] = setting["diffusion"]
    config = decode(g, config)
    streams = np.random.SeedSequence(int(seed)).spawn(5)
    initial = draw_initial(config, np.random.default_rng(streams[0]))
    packet = build_packet(config, initial)
    state, material, geometry, rates, _ = packet
    if state.counts[1] != 0 or np.any(state.owner >= 0) or state.counts[5] != 0:
        raise ValueError("Expected unreacted monomers, with no pre-formed chains.")
    base_init = rates[:2, 0].copy()
    base_diffusion = geometry[1]
    origin = np.random.default_rng(streams[3]).uniform(0., geometry[2], 3)
    stages, knots = _clock(environment, bool(full))
    brownian = _brownian_at_nodes(knots, np.random.default_rng(streams[1]))
    wait = np.random.default_rng(streams[2]).exponential(size=TAPE_LENGTH)
    marks = np.random.default_rng(streams[4]).random(TAPE_LENGTH)
    count, residual = 0, wait[0]
    local, path = [], []
    for begin, end, initiation, mobility, frame in stages:
        rates[:2, 0] = initiation*base_init
        geometry[1] = mobility*base_diffusion
        count, residual = advance_fixed(packet, begin, end, brownian, wait, marks,
                                        count, residual, STEP, cell_forces)
        if frame is not None:
            if frame < len(EARLY_TIMES):
                local.append(_local_observation(state, origin, geometry[2], geometry[7]))
            if full:
                micro, winding = extract_state(state)
                path.append(np.r_[state.counts[5]/SITES, micro, winding.any()])
    exposure = wait[:count].sum() + wait[count] - residual
    if not np.isclose(exposure, state.position[3], rtol=0., atol=2.e-10):
        raise ArithmeticError("Reaction innovation clock disagreed with native exposure.")
    result = {"local": np.stack(local)}
    if full:
        events = state.events[:state.counts[0]]
        milestones, cardinality = _milestones(events)
        if cardinality != state.counts[5]:
            raise ArithmeticError("Replayed vinyl cardinality disagreed with the native state.")
        result.update(path=np.stack(path), milestones=milestones)
        if journal:
            result.update(journal=events.copy(), initiators=int(initial["I"]), feed_time=float(material[3]))
    return result
