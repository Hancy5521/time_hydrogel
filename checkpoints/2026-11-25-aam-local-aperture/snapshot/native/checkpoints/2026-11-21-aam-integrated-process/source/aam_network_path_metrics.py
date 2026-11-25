"""Replay distinct kinetic lineages and periodic covalent network connectivity."""

import numpy as np
from numba import njit

from aam_mbaa_kinetics import MASS_AAM, MASS_MBAA


METRICS = (
    "conversion", "lineage_dpw", "largest_mass_fraction", "bridge_count",
    "intrachain_count", "within_component_interchain_count", "any_winding",
    "winding_x", "winding_y", "winding_z",
)


@njit(cache=True, inline="always")
def _component(parent, increments, molecule):
    offset = np.zeros(3, dtype=np.int64)
    while parent[molecule] != molecule:
        offset += increments[molecule]
        molecule = parent[molecule]
    return molecule, offset


@njit(cache=True)
def _add_edge(parent, increments, rank, winding, u, v, shift):
    """Use the stored directed lattice shift as the offset(v) - offset(u) constraint."""
    root_u, offset_u = _component(parent, increments, u)
    root_v, offset_v = _component(parent, increments, v)
    offset = shift + offset_u - offset_v
    if root_u == root_v:
        for axis in range(3):
            winding[axis] |= offset[axis] != 0
    elif rank[root_u] >= rank[root_v]:
        parent[root_v] = root_u
        increments[root_v] = offset
        if rank[root_u] == rank[root_v]:
            rank[root_u] += 1
    else:
        parent[root_u] = root_v
        increments[root_u] = -offset


@njit(cache=True)
def _summary(parent, masses, spent, vinyl, squared_length, n_aam, n_mbaa, counts, winding):
    masses[:] = 0.
    largest = 0.
    for molecule in range(len(parent)):
        if spent[molecule] > 0:
            root = molecule
            while parent[root] != root:
                root = parent[root]
            masses[root] += MASS_AAM if molecule < n_aam else MASS_MBAA
            largest = max(largest, masses[root])
    total_spent = spent.sum()
    length = squared_length/total_spent if total_spent else 0.
    charge_mass = n_aam*MASS_AAM + n_mbaa*MASS_MBAA
    return np.array((total_spent/vinyl, length, largest/charge_mass,
                     float(counts[0]), float(counts[1]), float(counts[2]), 1.*winding.any(),
                     1.*winding[0], 1.*winding[1], 1.*winding[2]))


@njit(cache=True)
def _replay(n_aam, n_mbaa, initial_mbaa, events, times, query_counts):
    n = n_aam+n_mbaa
    vinyl = n_aam+2*n_mbaa
    parent = np.arange(n)
    increments = np.zeros((n, 3), dtype=np.int64)
    rank = np.zeros(n, dtype=np.int64)
    winding = np.zeros(3, dtype=np.bool_)
    owner = np.full(2*n, -1)
    active = np.zeros(2*n, dtype=np.bool_)
    present = np.arange(n) < n_aam+initial_mbaa
    chain_size = np.zeros(2*n, dtype=np.int64)
    molecule_uses = np.zeros(n, dtype=np.int64)
    masses = np.zeros(n)
    counts = np.zeros(3, dtype=np.int64)
    fed = initial_mbaa
    chains = total_spent = squared_length = 0
    row = level = 0
    first_winding = np.nan
    timed = np.empty((len(times), 10))
    matched = np.full((len(query_counts), 10), np.nan)
    for i in range(len(events)+1):
        final = i == len(events)
        instant = np.inf if final else events[i, 0]
        while row < len(times) and times[row] < instant:
            timed[row] = _summary(parent, masses, molecule_uses, vinyl, squared_length,
                                  n_aam, n_mbaa, counts, winding)
            row += 1
        if final:
            break
        tag, u, v = int(events[i, 1]), int(events[i, 2]), int(events[i, 3])
        if tag == 0:
            if present[v] or v < n_aam:
                raise ValueError("A fed molecule must be a previously absent MBAA.")
            present[v] = True
            fed += 1
            continue
        vh = v//2
        if not present[vh] or (vh < n_aam and v % 2):
            raise ValueError("The target site must be chemically present.")
        if tag >= 3:
            uh = u//2
            if not active[u] or (tag != 8 and uh == vh):
                raise ValueError("An edge source must be active and on a different molecule.")
            kinetic_chain = owner[u]
            active[u] = False
        else:
            kinetic_chain = chains
            chains += 1

        consumed = tag <= 6 or tag == 9
        if consumed:
            if owner[v] >= 0:
                raise ValueError("A vinyl site can react only once.")
            pendant = owner[v ^ 1] >= 0
            if pendant:
                if tag < 3:
                    raise ValueError("Initiation requires an unreacted molecule.")
                ru, _ = _component(parent, increments, uh)
                rv, _ = _component(parent, increments, vh)
                same_lineage = kinetic_chain == owner[v ^ 1]
                same_component = ru == rv
                if same_lineage and not same_component:
                    raise ValueError("A kinetic lineage cannot span disconnected components.")
                classification = 6 if same_lineage else (9 if same_component else 5)
                if tag != classification:
                    raise ValueError("Pendant event mark disagrees with the kinetic lineage/graph replay.")
                counts[0] += tag == 5
                counts[1] += tag == 6
                counts[2] += tag == 9
            else:
                expected = (1 if tag <= 2 else 3) + int(vh >= n_aam)
                if tag != expected:
                    raise ValueError("Non-pendant event mark disagrees with the selected site.")
            owner[v] = kinetic_chain
            active[v] = True
            molecule_uses[vh] += 1
            total_spent += 1
            squared_length += 2*chain_size[kinetic_chain]+1
            chain_size[kinetic_chain] += 1
        else:
            if not active[v] or u == v:
                raise ValueError("Termination requires two different active sites.")
            active[v] = False

        if tag >= 3 and tag != 8:
            shift = events[i, 10:13].astype(np.int64)
            _add_edge(parent, increments, rank, winding, uh, vh, shift)
            if np.isnan(first_winding) and winding.any():
                first_winding = total_spent/vinyl
        if consumed and level < len(query_counts) and total_spent == query_counts[level]:
            if fed == n_mbaa:
                matched[level] = _summary(parent, masses, molecule_uses, vinyl, squared_length,
                                          n_aam, n_mbaa, counts, winding)
            level += 1
    return timed, matched, first_winding


def path_metrics(recipe, events, times, levels, *, with_first_winding=False):
    """Measure right-continuous paths and first passage at post-feed conversion counts.

    Lineage DPw is weighted by consumed vinyl sites. The largest component uses
    full planned molecule mass as its denominator. Winding is cumulative and
    topological. A late feed cannot retrospectively validate an earlier crossing.
    """
    events, times, levels = (np.asarray(x, dtype=float) for x in (events, times, levels))
    if (events.ndim != 2 or events.shape[1] != 13 or times.ndim != 1 or levels.ndim != 1
            or not np.isfinite(events).all() or not np.isfinite(times).all() or not np.isfinite(levels).all()
            or np.any(np.diff(times) <= 0.) or np.any(np.diff(levels) <= 0.)
            or np.any((times < 0.) | (times > recipe.horizon)) or np.any((levels <= 0.) | (levels > 1.))):
        raise ValueError("Expected a finite event table and increasing time and conversion checkpoints.")
    tag, source, target = events[:, 1:4].T
    n = recipe.aam+recipe.mbaa
    feed = tag == 0
    if (np.any(np.diff(events[:, 0]) < 0.) or np.any((events[:, 0] < 0.) | (events[:, 0] > recipe.horizon))
            or np.any((tag < 0.) | (tag > 9.)) or np.any((target < 0.) | (target >= np.where(feed, n, 2*n)))
            or np.any((tag <= 2) & (source != -1.)) or np.any((tag >= 3) & ((source < 0.) | (source >= 2*n)))
            or np.any(events[:, 1:4] != np.rint(events[:, 1:4]))
            or np.any(events[:, 10:13] != np.rint(events[:, 10:13]))):
        raise ValueError("Event clocks, marks, site indices, or lattice shifts are invalid.")
    counts = np.ceil(levels*(recipe.aam+2*recipe.mbaa)-1.e-12).astype(np.int64)
    if np.any(np.diff(counts) <= 0) or (len(counts) and counts[0] < 1):
        raise ValueError("Conversion queries must use distinct positive vinyl counts.")
    timed, matched, first_winding = _replay(recipe.aam, recipe.mbaa, recipe.initial_mbaa, events,
                                           times, counts)
    return (timed, matched, first_winding) if with_first_winding else (timed, matched)
