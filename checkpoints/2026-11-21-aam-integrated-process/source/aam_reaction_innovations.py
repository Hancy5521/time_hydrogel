"""Source-only explicit-innovation adapter for the frozen split BD/SSA kernel.

No import-time simulation or fitting. The initial bridge is legacy-record versus
tape-replay with proposal=target. It is NOT an equality-of-measures proof between
the old uniform-uint32 seed target and a newly declared IID innovation target.
The old scientific modules are imported unchanged.
"""
from dataclasses import dataclass, field
from typing import Callable, Optional
import numpy as np
from numba import njit

from aam_edge_fork import State, add_feed, apply_event, candidates
from aam_network_events import CAP, factor_weights
from aam_snapshot_process import clone
from aam_spatial_motion import _compact_edges, brownian_step, spatial_forces


SCHEMA = 'aam-native-explicit-innovations-v1-source-only'
LEGACY_LAW = 'legacy-uint32-numba-rng-fixture-not-iid-target'
IID_LAW = 'declared-iid-normal-exponential-uniform-innovations-v1'


@njit(cache=True)
def _legacy_seed(seed):
    np.random.seed(seed)


@njit(cache=True)
def _legacy_normals(number):
    return np.random.standard_normal((number, 3))


@njit(cache=True)
def _legacy_exponential():
    return np.random.exponential()


@njit(cache=True)
def _legacy_uniform():
    return np.random.random()


@njit(cache=True)
def _begin_slab(state, clock, material, geometry):
    """The old outer-loop arithmetic, with the random half-move split out."""
    if state.counts[6] != 0:
        return False
    aam, initial = int(material[0]), int(material[2])
    pulse = material[3]
    tr = state.position[0]
    fed = state.present.all()
    if not fed and tr == pulse:
        add_feed(state, aam + initial, tr)
        fed = True
    if tr == clock[state.counts[2]]:
        state.counts[2] += 1
    endpoint = min(tr + geometry[0], clock[state.counts[2]])
    if not fed:
        endpoint = min(endpoint, pulse)
    state.position[1] = endpoint
    state.position[2] = tr
    state.counts[6] = 1
    return True


@njit(cache=True)
def _slab_values(state):
    tr, endpoint = state.position[:2]
    return tr, endpoint, .5 * (endpoint - tr)


@njit(cache=True)
def _move_with_noise(state, half, mobility, geometry, noise):
    """Identical old move expressions, except its N(0,I) array is explicit."""
    box, spring, rest, repulsion, core = geometry[2:7]
    present = state.present
    forces, _ = spatial_forces(
        state.points[present],
        _compact_edges(state.edges[:state.counts[1]], present),
        box, spring, rest, repulsion, core)
    positions, displacement = brownian_step(
        state.points[present], forces, half, mobility[present], noise, box)
    state.points[present] = positions
    state.unwrapped[present] += displacement


@njit(cache=True)
def _finish_slab(state, half, mobility, geometry, noise):
    _move_with_noise(state, half, mobility, geometry, noise)
    state.position[0] = state.position[1]
    state.counts[6] = 0


@njit(cache=True)
def _candidate_law(state, material, geometry, rates, factors, cache, present,
                   rows, distances, primitive, weights, risk):
    """Keep Numba's original sum/cumsum rather than Python pairwise sums."""
    n = candidates(state, int(material[0]), geometry[2], geometry[7],
                   geometry[8], geometry[9], rates, cache, present,
                   rows, distances, primitive)
    hazard = np.sum(primitive[:n])
    factor_weights(primitive, rows[:, 4], rows[:, 5], n,
                   factors, weights, risk)
    cumulative = np.cumsum(weights[:n])
    return n, hazard, cumulative


@njit(cache=True)
def _proposal_law(hazard, clock_factor, weights, cumulative, multipliers):
    """Identity uses the original hazard and cumulative array unchanged."""
    target_total = hazard * clock_factor
    identity = np.all(multipliers == 1.)
    if identity:
        return target_total, cumulative.copy(), weights.copy(), True
    proposed = weights * multipliers
    proposal_cumulative = np.cumsum(proposed)
    if len(cumulative) == 0 or cumulative[-1] == 0.:
        return 0., proposal_cumulative, proposed, False
    proposal_total = target_total * (proposal_cumulative[-1] / cumulative[-1])
    return proposal_total, proposal_cumulative, proposed, False


@njit(cache=True)
def _waiting_time(local, exponential, hazard, clock_factor):
    return local + exponential / (hazard * clock_factor)


@njit(cache=True)
def _proposal_waiting_time(local, exponential, proposal_total):
    return local + exponential / proposal_total


@njit(cache=True)
def _mark_index(cumulative, uniform):
    return np.searchsorted(cumulative, uniform * cumulative[-1], side='right')


@njit(cache=True)
def _advance_dwell(state, reacted, endpoint, primitive_hazard):
    local = state.position[2]
    end = min(reacted, endpoint)
    duration = end - local
    # This remains the TARGET primitive integral, not proposal_total * duration.
    state.position[3] += duration * primitive_hazard
    state.position[2] = end
    return end, duration


@dataclass
class InnovationSource:
    """One native interval's raw innovations; no independent stream per draw type.

    legacy-record uses the same Numba thread-local RNG as old evolve. Tape replay
    calls no RNG. New-IID requires an explicit law declaration and authorization;
    its generator seed is implementation provenance, not a latent feature.
    """
    mode: str
    seed: Optional[int] = None
    tape: Optional[dict] = None
    declared_law: Optional[str] = None
    allow_new_iid: bool = False
    _entries: list = field(default_factory=list, init=False, repr=False)
    _cursor: int = field(default=0, init=False, repr=False)
    _rng: object = field(default=None, init=False, repr=False)

    def __post_init__(self):
        if self.mode == 'legacy-record':
            if not isinstance(self.seed, (int, np.integer)) or isinstance(self.seed, bool) or not 0 <= self.seed < 2**32:
                raise ValueError('Legacy bridge requires its original uint32 seed.')
            self.declared_law = LEGACY_LAW
        elif self.mode == 'tape-replay':
            if self.tape is None or self.tape.get('schema') != SCHEMA:
                raise ValueError('An explicit matching-schema tape is required.')
            self.declared_law = self.tape['declared_law']
            self.seed = self.tape.get('seed_provenance')
        elif self.mode == 'new-iid':
            if not self.allow_new_iid or self.declared_law != IID_LAW:
                raise ValueError('New-IID is a separately registered target, never an implicit legacy replacement.')
            self._rng = np.random.default_rng(self.seed)
        else:
            raise ValueError('Use legacy-record, tape-replay or explicitly declared new-iid.')

    def _read(self, kind, metadata):
        if self._cursor >= len(self.tape['entries']):
            raise ValueError('Innovation tape exhausted; no automatic RNG extension.')
        entry = self.tape['entries'][self._cursor]
        if entry['kind'] != kind or entry['metadata'] != metadata:
            raise ValueError('Innovation type/order/clock mismatch.')
        self._cursor += 1
        return entry['value']

    def _keep(self, kind, metadata, value):
        stored = value.copy() if isinstance(value, np.ndarray) else float(value)
        self._entries.append({'kind': kind, 'metadata': metadata.copy(), 'value': stored})
        return stored.copy() if isinstance(stored, np.ndarray) else stored

    def normal(self, present_ids, slab, half, begin, end):
        ids = np.asarray(present_ids, dtype=np.int64)
        metadata = {'present_ids': ids.tolist(), 'slab': int(slab), 'half': int(half),
                    'begin': float(begin), 'end': float(end)}
        if self.mode == 'tape-replay':
            value = np.asarray(self._read('normal', metadata), dtype=float).copy()
        elif self.mode == 'legacy-record':
            value = self._keep('normal', metadata, _legacy_normals(len(ids)))
        else:
            value = self._keep('normal', metadata, self._rng.standard_normal((len(ids), 3)))
        if value.shape != (len(ids), 3) or not np.isfinite(value).all():
            raise ValueError('Invalid explicit Brownian innovation.')
        return value

    def exponential(self, slab, local, endpoint):
        metadata = {'slab': int(slab), 'local': float(local), 'endpoint': float(endpoint)}
        if self.mode == 'tape-replay':
            value = float(self._read('exponential', metadata))
        elif self.mode == 'legacy-record':
            value = self._keep('exponential', metadata, _legacy_exponential())
        else:
            value = self._keep('exponential', metadata, self._rng.exponential())
        if not np.isfinite(value) or value < 0.:
            raise ValueError('Invalid explicit Exp(1) innovation.')
        return value

    def uniform(self, slab, local, reacted):
        metadata = {'slab': int(slab), 'local': float(local), 'reacted': float(reacted)}
        if self.mode == 'tape-replay':
            value = float(self._read('mark', metadata))
        elif self.mode == 'legacy-record':
            value = self._keep('mark', metadata, _legacy_uniform())
        else:
            value = self._keep('mark', metadata, self._rng.random())
        if not np.isfinite(value) or not 0. <= value < 1.:
            raise ValueError('Invalid explicit mark innovation.')
        return value

    def finish(self):
        if self.mode == 'tape-replay':
            if self._cursor != len(self.tape['entries']):
                raise ValueError('Unconsumed tape suffix; no silent omission.')
            return self.tape
        return {'schema': SCHEMA, 'declared_law': self.declared_law,
                'seed_provenance': None if self.seed is None else int(self.seed),
                'entries': self._entries}


def _multipliers(proposal, state, rows, distances, primitive, weights, context):
    n = len(rows)
    if proposal is None:
        return np.ones(n)
    if callable(proposal):
        # A callback may use current candidates and observations, never mutate live
        # state, peek future innovations, sample randomness or change legal support.
        arrays = [a.copy() for a in (rows, distances, primitive, weights)]
        for value in arrays:
            value.flags.writeable = False
        eta = np.asarray(proposal(clone(state), *arrays, context.copy()), dtype=float)
    else:
        table = np.asarray(proposal, dtype=float)
        if table.ndim != 1 or len(table) < 10:
            raise ValueError('Fixed multiplier table must include original tags0..9.')
        eta = table[rows[:, 0]]
    if eta.shape != (n,) or not np.isfinite(eta).all() or np.any(eta <= 0.):
        raise ValueError('Every legal candidate requires a finite strictly-positive multiplier.')
    return eta.copy()


def propagate_interval(packet, canonical_clock, innovation_source, event_proposal=None,
                       *, clock_factor=1., record_states=False):
    """Advance one complete interval with explicit BD and event innovations.

    The packet is exclusively owned and mutated, as in old progress. A finite
    fixed multiplier callback/table is a proposal only, never a physical action.
    Nonidentity proposals are forbidden on legacy-law tapes because their RN
    formula is for the separately declared IID path law. No main driver exists.
    """
    state, material, geometry, rates, factors = packet
    clock = np.asarray(canonical_clock, dtype=float)
    if not isinstance(state, State) or clock.ndim != 1 or len(clock) < 2 or not np.isfinite(clock).all() or np.any(np.diff(clock) <= 0.):
        raise ValueError('Require the original State and increasing native clock.')
    if state.counts[6] != 0 or state.position[0] >= clock[-1] or not np.any(clock == state.position[0]):
        raise ValueError('Begin at a completed original native clock boundary.')
    if not np.isfinite(clock_factor) or clock_factor <= 0.:
        raise ValueError('Positive finite original clock factor required.')
    # Seed at the interval entry, not at source-object construction: creating
    # several source descriptors must not silently overwrite another one's RNG.
    if innovation_source.mode == 'legacy-record':
        _legacy_seed(int(innovation_source.seed))
    width = 2 * len(state.points)
    length = width * width + 2 * width
    rows = np.empty((length, 6), np.int64)
    distances = np.empty(length)
    primitive = np.empty(length)
    weights = np.empty(length)
    risk = np.empty((3, CAP + 1))
    mobility = np.full(len(state.points), geometry[1])
    mobility[int(material[0]):] *= .8
    spans, brownian = [], []
    total_log_ratio = 0.
    cumulative_target = 0.
    cumulative_proposal = 0.
    start_integrated_primitive = float(state.position[3])
    slab = 0
    while state.position[0] < clock[-1]:
        opened = _begin_slab(state, clock, material, geometry)
        if not opened:
            raise ValueError('This bridge does not start from or silently repair a mid-slab packet.')
        tr, endpoint, half = _slab_values(state)
        ids = np.flatnonzero(state.present)
        noise = innovation_source.normal(ids, slab, 0, tr, endpoint)
        _move_with_noise(state, half, mobility, geometry, noise)
        brownian.append({'slab': slab, 'half': 0, 'begin': tr, 'end': endpoint,
                         'dt': half, 'present_ids': ids.copy(), 'normal': noise.copy()})
        cache = np.full((len(state.points), len(state.points)), -1, np.int8)
        present = np.flatnonzero(state.present)
        while state.position[2] < endpoint:
            n, hazard, cumulative = _candidate_law(
                state, material, geometry, rates, factors, cache, present,
                rows, distances, primitive, weights, risk)
            local = float(state.position[2])
            eta = _multipliers(event_proposal, state, rows[:n], distances[:n],
                               primitive[:n], weights[:n],
                               {'slab': slab, 'time': local, 'endpoint': endpoint,
                                'target_primitive_total': hazard})
            proposal_total, proposal_cumulative, proposed, identity = _proposal_law(
                hazard, clock_factor, weights[:n], cumulative, eta)
            if innovation_source.declared_law == LEGACY_LAW and not identity:
                raise ValueError('Legacy finite-seed bridge is restricted to proposal=target.')
            target_total = float(hazard * clock_factor)
            if not np.isfinite(target_total) or not np.isfinite(proposal_total) or target_total < 0. or proposal_total < 0.:
                raise ValueError('Invalid target/proposal total intensity.')
            if (target_total > 0.) != (proposal_total > 0.):
                raise ValueError('Proposal support must equal original legal-event support.')
            exponential = None
            if target_total > 0.:
                if n == 0 or cumulative[-1] <= 0. or proposal_cumulative[-1] <= 0.:
                    raise ValueError('Positive total with no positive legal mark law.')
                exponential = innovation_source.exponential(slab, local, endpoint)
                reacted = (_waiting_time(local, exponential, hazard, clock_factor) if identity
                           else _proposal_waiting_time(local, exponential, proposal_total))
            else:
                reacted = np.inf
            before = clone(state) if record_states else None
            end, duration = _advance_dwell(state, reacted, endpoint, hazard)
            censored = bool(reacted >= endpoint)
            if target_total > 0.:
                target_probability = weights[:n].copy() / cumulative[-1]
                proposal_probability = proposed.copy() / proposal_cumulative[-1]
            else:
                target_probability = np.zeros(n)
                proposal_probability = np.zeros(n)
            chosen, uniform = -1, None
            target_chosen = proposal_chosen = 0.
            jump_log_ratio = 0.
            if not censored:
                uniform = innovation_source.uniform(slab, local, reacted)
                chosen = int(_mark_index(proposal_cumulative, uniform))
                if chosen >= n or weights[chosen] <= 0. or proposed[chosen] <= 0.:
                    raise ValueError('Selected event must be an original positive-rate legal candidate.')
                target_chosen = float(target_probability[chosen])
                proposal_chosen = float(proposal_probability[chosen])
                if not identity:
                    jump_log_ratio = float(np.log(target_total) + np.log(target_chosen)
                                           - np.log(proposal_total) - np.log(proposal_chosen))
            survival_log_ratio = float((proposal_total - target_total) * duration)
            log_ratio = jump_log_ratio + survival_log_ratio
            total_log_ratio += log_ratio
            cumulative_target += target_total * duration
            cumulative_proposal += proposal_total * duration
            spans.append({'slab': slab, 'begin': local, 'end': float(end), 'duration': float(duration),
                'geometry_time': float(tr + half), 'candidate_rows': rows[:n].copy(),
                'candidate_distance': distances[:n].copy(), 'primitive_rate': primitive[:n].copy(),
                'native_mark_weight': weights[:n].copy(), 'target_A': target_total,
                'proposal_A': float(proposal_total), 'target_probability': target_probability,
                'proposal_probability': proposal_probability, 'multipliers': eta,
                'chosen': chosen, 'target_chosen_probability': target_chosen,
                'proposal_chosen_probability': proposal_chosen, 'exponential': exponential,
                'uniform': uniform, 'censored': censored,
                'boundary': 'BD-slab-or-fixed-clock' if censored else 'reaction-event',
                'jump_log_target_over_proposal': jump_log_ratio,
                'survival_log_target_over_proposal': survival_log_ratio,
                'log_target_over_proposal': log_ratio, 'current_state': before})
            if censored:
                break
            apply_event(state, rows[chosen], distances[chosen], geometry[2], reacted, tr + half)
        ids = np.flatnonzero(state.present)
        noise = innovation_source.normal(ids, slab, 1, tr, endpoint)
        _finish_slab(state, half, mobility, geometry, noise)
        brownian.append({'slab': slab, 'half': 1, 'begin': tr, 'end': endpoint,
                         'dt': half, 'present_ids': ids.copy(), 'normal': noise.copy()})
        slab += 1
    if not state.present.all() and state.position[0] == material[3]:
        add_feed(state, int(material[0]) + int(material[2]), material[3])
    tape = innovation_source.finish()
    return {'packet': packet, 'tape': tape, 'spans': spans, 'brownian': brownian,
            'log_target_over_proposal': float(total_log_ratio),
            'integrated_target_intensity': float(cumulative_target),
            'integrated_proposal_intensity': float(cumulative_proposal),
            'target_primitive_increment': float(state.position[3] - start_integrated_primitive),
            'native_interval_calls': 1, 'slabs': slab,
            'scope': 'Explicit-innovation bridge source. No efficacy or material validation implied.'}
