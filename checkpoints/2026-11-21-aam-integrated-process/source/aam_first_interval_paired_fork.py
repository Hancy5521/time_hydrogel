"""One early AAm/MBAA bridge–loop intervention using shared addressed inputs."""
from dataclasses import dataclass

import numpy as np
from scipy.special import ndtr

from aam_edge_fork import apply_event
from aam_formation_actual_rollout import packet_fingerprint
from aam_formation_cycle_core import _advance, _cost
from aam_formation_cycle_data import _branch_check, check_actual_state, property_readout
from aam_formation_cycle_methods import process_goal_losses
from aam_formation_local_initial import build_packet
from aam_formation_micro_history import extract_frames, extract_future
from aam_formation_streamed_rollout import formation_rollout_streamed
from aam_network_events import CAP
from aam_reaction_innovations import (
    _advance_dwell, _begin_slab, _candidate_law, _finish_slab, _mark_index,
    _move_with_noise, _slab_values, _waiting_time, propagate_interval,
)
from aam_reaction_mark_involution import _first_at_least
from aam_snapshot_process import clone, observe


MARK_CELLS = 1 << 53
MAX_ATTEMPTS_PER_SLAB = 87
SCHEMA = 'aam-first-interval-role-addressed-bridge-loop-fork-v1'


def first_interval_inputs(config, brownian, waiting, mark):
    """Three caller-owned generators, independent from later intervals and actions."""
    dt = config['motion']['step']
    horizon = config['prefix_clock'][1]
    if config['prefix_clock'][0] != 0. or not 0. < dt <= horizon < config['recipe']['feed_time']:
        raise ValueError('The addressed table only covers the first no-feed interval.')
    start, slabs = 0., 0
    while start < horizon:
        start = min(start+dt, horizon)
        slabs += 1
    molecules = config['recipe']['aam']+config['recipe']['mbaa']
    return {'brownian': brownian.standard_normal((slabs, 2, molecules, 3)),
            'waiting': waiting.exponential(size=(slabs, MAX_ATTEMPTS_PER_SLAB)),
            'mark': mark.integers(0, MARK_CELLS, size=(slabs, MAX_ATTEMPTS_PER_SLAB), dtype=np.int64)}


@dataclass(frozen=True)
class ForcedMark:
    slab: int
    attempt: int
    cell: int


class FirstIntervalSource:
    """Each attempt uses a fixed pair of waiting and mark coordinates."""
    mode = 'role-addressed-first-interval'
    declared_law = SCHEMA

    def __init__(self, tables, force=None):
        self.tables = tables
        self.attempts = [0]*len(tables['waiting'])
        self.force = force
        self.forced = False
        self.last_wait = None

    def normal(self, present_ids, slab, half, begin, end):
        return self.tables['brownian'][slab, half, present_ids]

    def exponential(self, slab, local, endpoint):
        attempt = self.attempts[slab]
        if attempt >= MAX_ATTEMPTS_PER_SLAB:
            raise ValueError('Original first-interval event bound exceeded.')
        self.attempts[slab] += 1
        self.last_wait = (slab, attempt, local)
        return float(self.tables['waiting'][slab, attempt])

    def uniform(self, slab, local, reacted):
        if self.last_wait is None:
            raise ValueError('A mark requires a preceding positive-hazard wait.')
        waiting_slab, attempt, waiting_time = self.last_wait
        if waiting_slab != slab or waiting_time != local:
            raise ValueError('A mark must follow the same positive-hazard wait address.')
        self.last_wait = None
        if self.force is not None and (slab, attempt) == (self.force.slab, self.force.attempt):
            if not 0 <= self.force.cell < MARK_CELLS:
                raise ValueError('Forced mark cell is outside the common grid.')
            self.forced = True
            return self.force.cell/MARK_CELLS
        return int(self.tables['mark'][slab, attempt])/MARK_CELLS

    def finish(self):
        if self.force is not None and not self.forced:
            raise ValueError('The intervention address was not reached at the declared wait.')
        return {'schema': SCHEMA, 'waits_per_slab': self.attempts, 'forced_mark_consumed': self.forced}


def _mark_cell_and_count(cumulative, index):
    total = float(cumulative[-1])
    left = float(cumulative[index-1]) if index else 0.
    start = _first_at_least(left, total, MARK_CELLS)
    end = _first_at_least(float(cumulative[index]), total, MARK_CELLS)
    if start == end:
        return None, 0
    cell = (start+end-1)//2
    if int(np.searchsorted(cumulative, (cell/MARK_CELLS)*total, side='right')) != index:
        raise ArithmeticError('Forced grid cell must select the declared original candidate.')
    return cell, end-start


def _capture_first_interval(packet, clock, source, *, start_index=0):
    """Original BD/SSA helpers, capturing the exact CDF at each realized wait."""
    state, material, geometry, rates, factors = packet
    if (not 0 <= start_index < len(clock)-1 or state.position[0] != clock[start_index]
            or state.counts[2] != start_index or state.counts[6] or material[3] <= clock[-1]):
        raise ValueError('Start the trace at the declared completed no-feed camera boundary.')
    molecule_count = len(state.points)
    width = 2*molecule_count
    capacity = width*width+2*width
    rows = np.empty((capacity, 6), np.int64)
    distances, primitive, weights = (np.empty(capacity) for _ in range(3))
    risks = np.empty((3, CAP+1))
    mobility = np.full(molecule_count, geometry[1])
    mobility[int(material[0]):] *= .8
    spans = []
    slab = 0
    while state.position[0] < clock[-1]:
        if not _begin_slab(state, clock, material, geometry):
            raise ValueError('A new slab must start at the original physical boundary.')
        start, end, half = _slab_values(state)
        present = np.flatnonzero(state.present)
        noise = source.normal(present, slab, 0, start, end)
        _move_with_noise(state, half, mobility, geometry, noise)
        cache = np.full((molecule_count, molecule_count), -1, np.int8)
        while state.position[2] < end:
            count, hazard, cumulative = _candidate_law(state, material, geometry, rates, factors, cache,
                present, rows, distances, primitive, weights, risks)
            local = float(state.position[2])
            if not np.isfinite(hazard) or hazard < 0.:
                raise ValueError('The original total hazard must remain finite and nonnegative.')
            attempt = source.attempts[slab]
            wait = source.exponential(slab, local, end) if hazard > 0. else None
            reacted = _waiting_time(local, wait, hazard, 1.) if wait is not None else np.inf
            endpoint, _ = _advance_dwell(state, reacted, end, hazard)
            if reacted >= end:
                spans.append({'slab': slab, 'attempt': attempt, 'begin': local, 'end': endpoint,
                              'exponential': wait, 'censored': True})
                break
            if not count or cumulative[-1] <= 0.:
                raise ValueError('A realized event must have positive native mark support.')
            fingerprint = packet_fingerprint(packet)
            forced_here = source.force is not None and (slab, attempt) == (source.force.slab, source.force.attempt)
            pre_state = clone(state) if forced_here else None
            mark = source.uniform(slab, local, reacted)
            chosen = int(_mark_index(cumulative, mark))
            if chosen >= count or weights[chosen] <= 0.:
                raise ValueError('The original integer mark must select a legal positive candidate.')
            spans.append({'slab': slab, 'attempt': attempt, 'begin': local, 'end': endpoint,
                          'exponential': wait, 'censored': False, 'target_A': hazard,
                          'geometry_time': start+half, 'candidate_rows': rows[:count].copy(),
                          'candidate_distance': distances[:count].copy(), 'native_mark_weight': weights[:count].copy(),
                          'actual_candidate_cumulative': cumulative.copy(), 'pre_event_packet_sha256': fingerprint,
                          'forced_pre_event_state': pre_state, 'uniform': mark, 'chosen': chosen})
            apply_event(state, rows[chosen], distances[chosen], geometry[2], reacted, start+half)
        noise = source.normal(np.flatnonzero(state.present), slab, 1, start, end)
        _finish_slab(state, half, mobility, geometry, noise)
        slab += 1
    return {'packet': packet, 'spans': spans, 'tape': source.finish(), 'native_interval_calls': 1}


def first_matched_opportunity(spans):
    """Earliest realized wait with a same-radical bridge and loop in its risk set."""
    for span_index, span in enumerate(spans):
        if span['censored']:
            continue
        rows, distances, weights = (span[key] for key in ('candidate_rows', 'candidate_distance', 'native_mark_weight'))
        cumulative = span['actual_candidate_cumulative']
        kinds = rows[:, 4]
        eligible = (weights > 0.) & np.isin(kinds, (0, 1))
        bridge = np.flatnonzero(eligible & (kinds == 0))
        loop = np.flatnonzero(eligible & (kinds == 1))
        pairs = [(abs(float(distances[a]-distances[b])), int(a), int(b))
                 for a in bridge for b in loop if rows[a, 1] == rows[b, 1]]
        for _, a, b in sorted(pairs):
            left, n_left = _mark_cell_and_count(cumulative, a)
            right, n_right = _mark_cell_and_count(cumulative, b)
            if n_left and n_right:
                return {'slab': span['slab'], 'attempt': span['attempt'], 'span_index': span_index,
                        'time': span['end'], 'candidate_indices': (a, b), 'cells': (left, right),
                        'candidate_cell_counts': (n_left, n_right),
                        'candidate_rows': rows[[a, b]].copy(), 'distances': distances[[a, b]].copy(),
                        'candidate_weights': weights[[a, b]].copy(), 'original_span': span}
    return None


def _first_interval(config, initial, factors, tables, mark=None):
    packet = build_packet(config, initial, factors=factors)
    clock = np.asarray(config['prefix_clock'][:2], dtype=float)
    result = _capture_first_interval(packet, clock, FirstIntervalSource(tables, mark))
    if result['packet'][0].position[0] != clock[-1]:
        raise ValueError('Use the unmodified first-interval physics and end at the first snapshot.')
    return result


def check_first_interval_trace(config, initial, factors, tables, captured):
    """One prospective parity replay of the new read-only trace vs the original."""
    packet = build_packet(config, initial, factors=factors)
    clock = np.asarray(config['prefix_clock'][:2], dtype=float)
    original = propagate_interval(packet, clock, FirstIntervalSource(tables), record_states=False)
    if packet_fingerprint(original['packet']) != packet_fingerprint(captured['packet']):
        raise ValueError('The candidate-metadata adapter changed original first-interval physics.')
    for old, new in zip(original['spans'], captured['spans'], strict=True):
        for key in ('slab', 'begin', 'end', 'censored', 'exponential'):
            if old[key] != new[key]:
                raise ValueError('Original vs captured stochastic span differs: '+key)
        if not new['censored']:
            for key in ('uniform', 'chosen', 'geometry_time', 'target_A'):
                if old[key] != new[key]:
                    raise ValueError('Original actual mark trace differs: '+key)
            for key in ('candidate_rows', 'candidate_distance', 'native_mark_weight'):
                np.testing.assert_array_equal(old[key], new[key])
    return {'passed': True, 'original_replay': original, 'additional_intervals': 1}


def paired_first_interval(config, initial, factors, tables, *, verify_trace=False):
    """One search and up to two forks; optional one original-engine parity check."""
    original = _first_interval(config, initial, factors, tables)
    parity = check_first_interval_trace(config, initial, factors, tables, original) if verify_trace else None
    intervals = 1+int(parity is not None)
    opportunity = first_matched_opportunity(original['spans'])
    if opportunity is None:
        return {'opportunity': None, 'search': original, 'original_trace_parity': parity, 'interval_calls': intervals}
    source_span = opportunity['original_span']
    index = opportunity['span_index']
    selected = []
    for cell, candidate in zip(opportunity['cells'], opportunity['candidate_indices'], strict=True):
        mark = ForcedMark(opportunity['slab'], opportunity['attempt'], cell)
        result = _first_interval(config, initial, factors, tables, mark)
        span = result['spans'][index]
        if (span['chosen'] != candidate or span['end'] != source_span['end']
                or span['begin'] != source_span['begin'] or span['target_A'] != source_span['target_A']
                or span['exponential'] != source_span['exponential']
                or span['geometry_time'] != source_span['geometry_time']
                or span['pre_event_packet_sha256'] != source_span['pre_event_packet_sha256']
                or not np.array_equal(span['candidate_rows'], source_span['candidate_rows'])
                or not np.array_equal(span['candidate_distance'], source_span['candidate_distance'])
                or not np.array_equal(span['actual_candidate_cumulative'], source_span['actual_candidate_cumulative'])):
            raise ValueError('The two interventions must share the actual pre-event physical context.')
        selected.append(result)
    return {'opportunity': opportunity, 'search': original, 'bridge': selected[0], 'loop': selected[1],
            'first_snapshot': [observe(result['packet']) for result in selected],
            'original_trace_parity': parity, 'interval_calls': intervals+2}


def finish_to_cutoff(first_snapshot_packet, config, later_seeds):
    """Preserve the owned original first snapshot and reach the common cutoff."""
    state = clone(first_snapshot_packet[0])
    packet = (state, *(value.copy() for value in first_snapshot_packet[1:]))
    grid = config['prefix_clock']
    seeds = np.asarray(later_seeds)
    if (seeds.shape != (2,) or seeds.dtype.kind not in 'iu'
            or np.any(seeds < 0) or np.any(seeds.astype(np.uint64) >= 2**32)):
        raise ValueError('Require the two explicit uint32 seeds after the first snapshot.')
    if state.position[0] != grid[1]:
        raise ValueError('A complete first snapshot must seed both continuation branches.')
    frames, clean, cost = {grid[1]: clone(state)}, [observe(packet)], _cost()
    route = []
    for index in (2, 3):
        input_identity = packet_fingerprint(packet)
        _advance(packet, np.asarray(grid[:index+1]), int(seeds[index-2]), False, cost)
        frames[grid[index]] = clone(state)
        clean.append(observe(packet))
        route.append({'clock': tuple(grid[:index+1]), 'seed': int(seeds[index-2]),
                      'input_packet_sha256': input_identity, 'output_packet_sha256': packet_fingerprint(packet)})
    check_actual_state(state, expected_age=2.05)
    micro, wrapping = extract_frames(frames)
    return {'packet': packet, 'frames': frames, 'clean': np.asarray(clean),
            'micro': micro, 'winding': wrapping, 'route': route, 'cost': cost}


def future_for_action(packet, action, config, streams, mother_id):
    raw = formation_rollout_streamed(packet, action, config, **streams,
                                     properties=lambda state: property_readout(state, config), mother_id=mother_id)
    return raw


def score_future_for_action(raw, packet, action, config, stream):
    values, _, _, valid, errors = _branch_check(raw, packet, action, config, stream['seed'])
    if not valid.all() or not np.isfinite(values).all():
        raise RuntimeError('Actual future or mechanical reference failed: '+str(errors))
    micro, winding = extract_future(raw)
    return values, micro, winding


def paired_decision_diagnostic(clean, futures, config, normalization):
    """One independent mother; four paired suffixes, two selection/evaluation splits."""
    clean, futures = np.asarray(clean, float), np.asarray(futures, float)
    if clean.shape != (2, 3, 4) or futures.shape != (2, 4, 6, 3, 6) or not np.isfinite(clean).all():
        raise ValueError('Require both complete histories and all fixed suffixes/actions.')
    separation = float(np.linalg.norm((clean[0]-clean[1])/np.asarray(config['noise'])))
    pair_error = float(ndtr(-.5*separation))
    # Compute per-suffix losses before taking either means or selecting actions.
    losses = np.stack([process_goal_losses(values, normalization)['loss'] for values in futures])
    held_out_swap = np.empty((2, 2, 2))
    held_out_shared = np.empty((2, 2, 2))
    separate_loss = np.empty((2, 2, 2))
    common_loss = np.empty((2, 2, 2))
    selected = np.empty((2, 2, 2), dtype=np.int64)
    shared = np.empty((2, 2), dtype=np.int64)
    for split, (train, test) in enumerate(((slice(0, 2), slice(2, 4)), (slice(2, 4), slice(0, 2)))):
        choices = np.argmin(losses[:, train].mean(axis=1), axis=1)
        selected[split] = choices
        shared[split] = np.argmin(losses[:, train].mean(axis=(0, 1)), axis=0)
        for history in (0, 1):
            for goal in (0, 1):
                own, swapped = choices[history, goal], choices[1-history, goal]
                separate_loss[split, history, goal] = losses[history, test, own, goal].mean()
                common_loss[split, history, goal] = losses[history, test, shared[split, goal], goal].mean()
                held_out_swap[split, history, goal] = np.mean(losses[history, test, swapped, goal]
                                                               - losses[history, test, own, goal])
                held_out_shared[split, history, goal] = np.mean(losses[history, test, shared[split, goal], goal]
                                                                 - losses[history, test, own, goal])
    full_mean = losses.mean(axis=1)
    finite_regret = full_mean-full_mean.min(axis=1, keepdims=True)
    finite_separation = np.min(finite_regret.sum(axis=0), axis=0)
    return {'standardized_snapshot_distance': separation, 'two_history_oracle_error': pair_error,
            'per_suffix_action_goal_loss': losses, 'crossfit_choices': selected,
            'crossfit_shared_choices': shared,
            'heldout_swap_regret': held_out_swap, 'heldout_swap_regret_mean': held_out_swap.mean(axis=(0, 1)),
            'heldout_common_action_regret': held_out_shared,
            'heldout_common_action_regret_mean': held_out_shared.mean(axis=(0, 1)),
            'holdout_separate_action_loss': separate_loss, 'holdout_common_action_loss': common_loss,
            'holdout_gain_over_five_percent_target': (.95*common_loss-separate_loss).mean(axis=(0, 1)),
            'finite_reference_minimax_sum_gap': finite_separation,
            'finite_reference_ambiguity_product': finite_separation*pair_error}
