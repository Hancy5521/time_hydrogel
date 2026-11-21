"""Well-mixed two-vinyl network kinetics with a replayable formation clock.

All coefficients use the model clock and unit simulation volume. Molecular
connectivity and kinetic-chain ownership are separate representations.
"""

from dataclasses import dataclass

import numpy as np
from numba import njit
from scipy.optimize import brentq


TAGS = ("feed", "init_aam", "init_mbaa", "grow_aam", "grow_first_mbaa", "bridge",
        "cycle", "terminate_combination", "terminate_disproportionation")
MASS_AAM, MASS_MBAA = 71.08, 154.17


@dataclass(frozen=True)
class Recipe:
    aam: int
    mbaa: int
    initial_mbaa: int
    feed_time: float
    initiators: int
    horizon: float

    def __post_init__(self):
        for key in ("aam", "mbaa", "initial_mbaa", "initiators"):
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
                raise ValueError(f"{key} must be an integer.")
        if not (self.aam > 0 and 0 <= self.initial_mbaa <= self.mbaa and self.initiators >= 0
                and np.isfinite([self.feed_time, self.horizon]).all()
                and 0. <= self.feed_time <= self.horizon and self.horizon > 0.):
            raise ValueError("Invalid precursor counts, pulse or clock.")


@njit(cache=True, inline="always")
def _root(labels, i):
    while labels[i] != i:
        labels[i] = labels[labels[i]]
        i = labels[i]
    return i


@njit(cache=True, inline="always")
def _join(labels, a, b):
    labels[_root(labels, a)] = _root(labels, b)


@njit(cache=True, inline="always")
def _consume(site, remaining, reacted):
    # remaining stores individual vinyl bits. Each molecule enters on first use.
    remaining[site] = False
    reacted[site//2] += 1


@njit(cache=True)
def _simulate(aam, mbaa, initial, pulse, initiators, horizon, rate, seed):
    np.random.seed(seed)
    number = aam + mbaa
    total = aam + 2*mbaa
    sites = np.arange(2*number)
    mol = sites//2
    present = np.zeros(2*number, dtype=np.bool_)
    present[:2*aam:2] = True
    present[2*aam:2*(aam+initial)] = True
    active = np.full(2*number, False)
    reacted = np.zeros(number, dtype=np.int64)
    labels = np.arange(number)
    counts = np.zeros((8, total+1), dtype=np.int64)
    exposure = np.zeros((8, total+1))
    # At most total vinyl reactions, initiators//2 terminations and MBAA additions.
    events = np.zeros((total+initiators//2+mbaa+1, 4))
    n_events = 0
    m = 0
    time = 0.
    fed = initial == mbaa
    base = np.empty(8)
    for event_count in range(len(events)):
        ra = np.flatnonzero(active)
        a = sites[present & (mol < aam)]
        fresh = sites[present & (mol >= aam) & (reacted[mol] == 0)]
        pendant = sites[present & (mol >= aam) & (reacted[mol] == 1)]
        nr, na, nf = len(ra), len(a), len(fresh)
        inter = intra = 0
        for u in ra:
            left = _root(labels, u//2)
            for v in pendant:
                if u//2 == v//2:
                    continue
                if left == _root(labels, v//2):
                    intra += 1
                else:
                    inter += 1
        disprop_pairs = nr*(nr-1)//2
        comb_pairs = disprop_pairs
        for i in range(nr-1):
            if ra[i]//2 == ra[i+1]//2:
                comb_pairs -= 1
        # Competition for the finite initiation budget includes only free molecules.
        base[0], base[1] = initiators*na, initiators*nf
        base[2], base[3] = nr*na, nr*nf
        base[4], base[5] = inter, intra
        base[6], base[7] = comb_pairs, disprop_pairs
        totals = base * rate[:, m]
        cumulative = np.cumsum(totals)
        hazard = cumulative[-1]
        next_event = time + np.random.exponential()/hazard if hazard > 0. else np.inf
        limit = horizon if fed else pulse
        endpoint = min(next_event, limit)
        exposure[:, m] += (endpoint-time)*base
        time = endpoint
        if next_event >= limit:
            if fed:
                break
            boundary = aam+initial
            present[2*boundary:] = True
            for v in range(boundary, number):
                events[n_events, 0], events[n_events, 1] = time, 0
                events[n_events, 2], events[n_events, 3] = -1, v
                n_events += 1
            fed = True
            if time == horizon:
                break
            continue
        channel = np.searchsorted(cumulative, np.random.random()*hazard, side="right")
        u, v = -1, -1
        rank = np.random.randint(int(base[channel])) if channel >= 2 else 0
        if channel == 0 or channel == 1:
            v = a[np.random.randint(na)] if channel == 0 else fresh[np.random.randint(nf)]
            initiators -= 1
        elif channel == 2:
            u, v = ra[rank//na], a[rank % na]
        elif channel == 3:
            u, v = ra[rank//nf], fresh[rank % nf]
        elif channel == 4 or channel == 5:
            for source in ra:
                left = _root(labels, source//2)
                for target in pendant:
                    if (source//2 == target//2
                            or (_root(labels, target//2) == left) != (channel == 5)):
                        continue
                    if rank == 0:
                        u, v = source, target
                        break
                    rank -= 1
                if u >= 0:
                    break
        else:
            for i, source in enumerate(ra):
                for target in ra[i+1:]:
                    if channel == 6 and source//2 == target//2:
                        continue
                    if rank == 0:
                        u, v = source, target
                        break
                    rank -= 1
                if u >= 0:
                    break
        if v < 0:
            raise ArithmeticError("The selected channel has no reaction pair.")
        events[n_events, 0], events[n_events, 1] = time, channel+1
        events[n_events, 2], events[n_events, 3] = u, v
        n_events += 1
        counts[channel, m] += 1
        if u >= 0:
            active[u] = False
            if channel < 7:
                _join(labels, u//2, v//2)
        if channel < 6:
            _consume(v, present, reacted)
            active[v] = True
            m += 1
        else:
            active[v] = False
    return events[:n_events], counts, exposure


def rate_table(coefficients, vinyl):
    """Log-linear kinetic coefficient by total-charge-normalized conversion."""
    values = np.asarray(coefficients, dtype=float)
    if (values.shape != (8, 2) or not np.isfinite(values).all() or np.any(values[:, 0] < 0)
            or np.any(np.abs(values[:, 1]) > 20.)):
        raise ValueError("Eight nonnegative base rates and bounded finite slopes are required.")
    return values[:, :1]*np.exp(values[:, 1:]*np.arange(vinyl+1)/vinyl)


def simulate(recipe, coefficients, seed):
    if not isinstance(seed, (int, np.integer)) or not 0 <= seed < 2**32:
        raise ValueError("The reaction stream seed must be uint32.")
    vinyl = recipe.aam+2*recipe.mbaa
    return _simulate(recipe.aam, recipe.mbaa, recipe.initial_mbaa, recipe.feed_time,
                     recipe.initiators, recipe.horizon, rate_table(coefficients, vinyl), seed)


def fit_channels(counts, exposure, conversion, *, slopes=False):
    """Poisson event likelihood with right-censored time included as exposure."""
    counts, exposure, conversion = (np.asarray(x, dtype=float) for x in (counts, exposure, conversion))
    if counts.shape != exposure.shape or counts.shape != (8, len(conversion)):
        raise ValueError("Reaction sufficient statistics and conversion grid do not agree.")
    fit = np.zeros((8, 2), dtype=float)
    for j, (events, time_at_risk) in enumerate(zip(counts, exposure, strict=True)):
        n = events.sum()
        if n == 0 or time_at_risk.sum() <= 0.:
            raise ValueError("Every fitted event channel must have support.")
        slope = 0.
        if slopes and j in (4, 5):
            mean_conversion = events @ conversion/n

            def gap(b):
                tilt = time_at_risk*np.exp(b*conversion)
                return tilt @ conversion/tilt.sum() - mean_conversion

            if gap(-8.) >= 0.:
                slope = -8.
            elif gap(8.) <= 0.:
                slope = 8.
            else:
                slope = brentq(gap, -8., 8.)
        fit[j] = n/(time_at_risk @ np.exp(slope*conversion)), slope
    return fit


@njit(cache=True)
def _path_checkpoints(events, aam, mbaa, query_time, query_count, horizon, feed_time, initial_mbaa):
    number = aam+mbaa
    vinyl = aam+2*mbaa
    labels = np.arange(number)
    chain = np.full(2*number, -1)
    size = np.zeros(2*number, dtype=np.int64)
    initial_used = np.full(number, False)
    formed_mass = np.zeros(number)
    n_chain = spent = bridges = loops = 0
    feed_count = initial_mbaa
    next_time = next_count = 0
    timed = np.full((len(query_time), 9), np.nan)
    converted = np.full((len(query_count), 9), np.nan)
    crossing_time = np.nan
    for r in range(len(events)+1):
        if r == len(events):
            time, tag, u, v = horizon, -1, -1, -1
        else:
            time, tag_f, u_f, v_f = events[r]
            tag, u, v = int(tag_f), int(u_f), int(v_f)
        while next_time < len(query_time) and (query_time[next_time] < time or r == len(events)):
            masses = np.zeros(number)
            for i in range(number):
                masses[_root(labels, i)] += formed_mass[i]
            timed[next_time] = _summary(spent, vinyl, size[:n_chain], masses, bridges, loops,
                                        aam, mbaa, feed_count)
            next_time += 1
        if tag < 0:
            break
        if tag == 0:
            feed_count += 1
            continue
        if tag <= 6:
            spent += 1
            i = v//2
            if not initial_used[i]:
                formed_mass[i] = MASS_AAM if i < aam else MASS_MBAA
                initial_used[i] = True
            if tag <= 2:
                chain[v] = n_chain
                n_chain += 1
            else:
                chain[v] = chain[u]
                _join(labels, u//2, i)
            size[chain[v]] += 1
            bridges += tag == 5
            loops += tag == 6
        elif tag == 7:
            _join(labels, u//2, v//2)
        is_query = (next_count < len(query_count) and spent == query_count[next_count]
                    and tag <= 6 and (feed_count == mbaa))
        if is_query or np.isnan(crossing_time):
            masses = np.zeros(number)
            for i in range(number):
                masses[_root(labels, i)] += formed_mass[i]
            if np.isnan(crossing_time) and masses.max() >= .5*(MASS_AAM*aam+MASS_MBAA*mbaa):
                crossing_time = time
            if is_query:
                converted[next_count] = _summary(spent, vinyl, size[:n_chain], masses, bridges, loops,
                                                  aam, mbaa, feed_count)
        # A conversion level crossed before completing the feed stays unobserved.
        if next_count < len(query_count) and spent == query_count[next_count] and tag <= 6:
            next_count += 1
    return timed, converted, crossing_time


@njit(cache=True, inline="always")
def _summary(spent, vinyl, chain_lengths, masses, bridges, loops, aam, mbaa, feed_count):
    largest = masses.max()
    mass_total = masses.sum()
    threshold = .5*(MASS_AAM*aam+MASS_MBAA*mbaa)
    soluble = masses[masses < threshold]
    dp_n = chain_lengths.mean() if len(chain_lengths) > 0 else np.nan
    dp_w = (chain_lengths**2).sum()/spent if spent else np.nan
    soluble_mw = (soluble**2).sum()/soluble.sum() if soluble.sum() > 0 else np.nan
    return np.array((spent/vinyl, dp_n, dp_w,
                     largest/mass_total if mass_total > 0 else 0.,
                     largest/(MASS_AAM*aam+MASS_MBAA*mbaa), float(bridges), float(loops),
                     soluble_mw, float(feed_count)))


METRICS = ("vinyl_conversion", "kinetic_chain_dp_n", "kinetic_chain_dp_w", "largest_reacted_mass_fraction",
           "largest_final_charge_mass_fraction", "bridge_count", "cycle_count",
           "soluble_component_weight_average_mass", "mbaa_fed")


def checkpoints(recipe, events, times, conversion):
    times, conversion = np.asarray(times, dtype=float), np.asarray(conversion, dtype=float)
    if (times.ndim != 1 or conversion.ndim != 1 or not np.isfinite(times).all()
            or not np.isfinite(conversion).all() or np.any(np.diff(times) <= 0.) or np.any(np.diff(conversion) <= 0.)
            or np.any((times < 0.) | (times > recipe.horizon)) or np.any((conversion <= 0.) | (conversion > 1.))):
        raise ValueError("Expected increasing physical queries inside the model horizon.")
    counts = np.ceil(conversion*(recipe.aam+2*recipe.mbaa)-1.e-12).astype(np.int64)
    if np.any(np.diff(counts) <= 0):
        raise ValueError("Conversion queries must use distinct reacted vinyl counts.")
    return _path_checkpoints(events, recipe.aam, recipe.mbaa, times, counts, recipe.horizon,
                             recipe.feed_time, recipe.initial_mbaa)
