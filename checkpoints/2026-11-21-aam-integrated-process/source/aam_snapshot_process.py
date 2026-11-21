"""Sparse local measurements and lawful counterfactual feed on one hydrogel microstate."""

from dataclasses import replace

import numpy as np
from numba import njit
from scipy.special import logsumexp

from aam_edge_fork import State, evolve, setup
from aam_mbaa_kinetics import Recipe, _root
from aam_spatial_motion import Motion


WINDOW_Y = (1.35, 4.35)
PORTS = (1.45, 4.45)


@njit(cache=True)
def weight(points, center_x, center_y, box):
    """Separable compact aperture with no overlap between viewed and hidden slabs."""
    h = 1.30
    dx = points[:, 0]-center_x
    dy = points[:, 1]-center_y
    dx -= box*np.rint(dx/box)
    dy -= box*np.rint(dy/box)
    return np.maximum(0., 1.-np.abs(dx)/h)*np.maximum(0., 1.-np.abs(dy)/h)


@njit(cache=True)
def read_apertures(state, aam, initial_mbaa, box):
    out = np.empty(4)
    for side in range(2):
        w = weight(state.points, PORTS[side], WINDOW_Y[0], box)
        conversion = w[:aam] @ (state.owner[:2*aam:2] >= 0).astype(np.float64)
        out[2*side] = conversion/max(w[:aam].sum(), 1.e-12)
        out[2*side+1] = 6.*(w[aam:] @ state.present[aam:].astype(np.float64))/initial_mbaa
    return out


@njit(cache=True)
def graph_readout(state, aam, mbaa, box):
    """Mass in reacted components with at least three kinetic lineages, in held apertures."""
    n = len(state.points)
    root = np.array([_root(state.labels, i) for i in range(n)])
    count = np.zeros(n, np.int64)
    seen = np.zeros((n, n), np.bool_)
    active_mass = np.zeros(n)
    for i in range(n):
        for slot in range(2):
            lineage = state.owner[2*i+slot]
            if lineage >= 0:
                active_mass[i] = 1. if i < aam else 2.17
                if not seen[root[i], lineage]:
                    seen[root[i], lineage] = True
                    count[root[i]] += 1
    support = np.where(state.present, np.where(np.arange(n) < aam, 1., 2.17), 0.)
    output = np.empty(3)
    regional = np.where(count[root] >= 3, active_mass, 0.)
    for i in range(2):
        w = weight(state.points, PORTS[i], WINDOW_Y[1], box)
        output[i] = (w @ regional)/max(w @ support, 1.e-12)
    output[2] = state.counts[5]/(aam+2*mbaa)
    return output


def clone(state):
    return State(*(item.copy() for item in state))


def initialize(config, seed):
    rng = np.random.default_rng(seed)
    recipe = Recipe(**config["recipe"])
    motion = Motion(**config["motion"])
    initiators = int(rng.integers(6, 13))
    imbalance = rng.beta(.65, .65)
    recipe = replace(recipe, initiators=initiators)
    state, material, geometry = setup(recipe, motion, seed, config["corridor"])
    n = recipe.initial_mbaa
    start = recipe.aam
    centers = np.where(rng.random(n) < imbalance, PORTS[0], PORTS[1])
    state.points[start:start+n, 0] = (centers+rng.normal(0., .62, n)) % motion.box
    state.unwrapped[:] = state.points
    rates = np.asarray(config["rates"])
    factors = np.asarray(config["factors"])
    return state, material, geometry, rates, factors


def progress(packet, clock, seed):
    state, material, geometry, rates, factors = packet
    evolve(state, np.asarray(clock), material, geometry, rates, factors, 1., int(seed), 0)
    return state


def observe(packet):
    state, material, geometry, _, _ = packet
    return read_apertures(state, int(material[0]), int(material[2]), geometry[2])


def values(packet):
    state, material, geometry, _, _ = packet
    return graph_readout(state, int(material[0]), int(material[1]), geometry[2])


def precursor(config, seed):
    packet = initialize(config, seed)
    raw = []
    for i in range(1, len(config["prefix_clock"])):
        progress(packet, config["prefix_clock"][:i+1], seed+17+i)
        raw.append(observe(packet))
    return packet, np.stack(raw)


def branch(packet, action, config, seed):
    state, material, geometry, rates, factors = packet
    if state.counts[6] or state.present.all() or action[0] <= state.position[0]:
        raise ValueError("Only branch at the common completed prefix before any candidate pulse.")
    material = material.copy()
    material[3] = action[0]
    moved = clone(state)
    absent = ~moved.present
    rng = np.random.default_rng(seed+51)
    n = int(absent.sum())
    centers = np.where(rng.random(n) < action[1], PORTS[0], PORTS[1])
    moved.points[absent, 0] = (centers+rng.normal(0., .64, n)) % geometry[2]
    moved.unwrapped[absent] = moved.points[absent]
    branch_packet = moved, material, geometry, rates, factors
    time = list(config["prefix_clock"])
    trajectory = []
    for i, t in enumerate(config["future_times"]):
        time.append(t)
        progress(branch_packet, time, seed+1+i)
        trajectory.append(values(branch_packet))
    return np.stack(trajectory)


def normalize(log_weights):
    return np.exp(log_weights-logsumexp(log_weights))


def systematic(weights, count, rng):
    cdf = np.cumsum(weights)
    cdf[-1] = 1.
    return np.searchsorted(cdf, (rng.random()+np.arange(count))/count, side="right")


def likelihood(raw, measured, noise):
    return -.5*np.square((raw-measured)/noise).sum(axis=1)


def assimilate(config, measured, seed, sequential):
    """Complete-state bootstrap filter or equally budgeted nonresampled importance sampling."""
    count = config["particles"]
    noise = np.asarray(config["noise"])
    rng = np.random.default_rng(seed)
    particles = [initialize(config, seed+20*i) for i in range(count)]
    log_weights = np.zeros(count)
    ancestors = np.arange(count)
    history = []
    for step, time in enumerate(config["prefix_clock"][1:], 1):
        observations = []
        for slot, particle in enumerate(particles):
            progress(particle, config["prefix_clock"][:step+1], seed+5000*step+slot)
            observations.append(observe(particle))
        log_weights += likelihood(np.stack(observations), measured[step-1], noise)
        weights = normalize(log_weights)
        ess = 1./np.square(weights).sum()
        history.append((ess, len(np.unique(ancestors))))
        if sequential and step < len(config["prefix_clock"])-1 and ess < .6*count:
            chosen = systematic(weights, count, rng)
            particles = [(clone(particles[i][0]), *particles[i][1:]) for i in chosen]
            ancestors = ancestors[chosen]
            log_weights.fill(0.)
    selected = systematic(weights, config["posterior_draws"], rng)
    return [particles[k] for k in selected], np.asarray(history), len(np.unique(ancestors[selected]))


def posterior_predictions(config, measured, seed, method):
    packets, stats, unique = assimilate(config, measured, seed, method == "smc")
    hidden = np.stack([values(p) for p in packets])
    future = []
    for i, particle in enumerate(packets):
        future.append([branch(particle, a, config, seed+20000+100*i) for a in config["actions"]])
    return hidden, np.asarray(future), stats, unique

