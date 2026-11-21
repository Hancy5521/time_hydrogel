"""Prior-reversible exchange for two legal reaction kinds at a fixed mark address."""
from bisect import bisect_right
from dataclasses import dataclass
from math import isfinite


def _first_at_least(cumulative, total, cells):
    """Find the integer boundary using the simulator's floating mark product."""
    lower, upper = 0, cells
    while lower < upper:
        middle = (lower + upper)//2
        if (middle/cells)*total < cumulative:
            lower = middle+1
        else:
            upper = middle
    return lower


@dataclass(frozen=True)
class _KindCoordinates:
    starts: tuple
    rank_ends: tuple

    @property
    def mass(self):
        return self.rank_ends[-1] if self.rank_ends else 0

    def mark(self, rank):
        index = bisect_right(self.rank_ends, rank)
        previous = self.rank_ends[index-1] if index else 0
        return self.starts[index]+rank-previous


class ReactionMarkExchange:
    """Exchange equal numbers of marks between two possibly interleaved kinds.

    The model state and candidate CDF just before this addressed reaction are
    common to the forward and reverse move. This component never changes a
    physical state, simulates a suffix, or draws an address or phase.
    """

    def __init__(self, cumulative, kinds, first_kind, second_kind, *, bits):
        if type(bits) is not int or not 1 <= bits <= 53:
            raise ValueError('Use a binary mark grid of 1 through 53 bits.')
        self.cells = 1 << bits
        cumulative = tuple(map(float, cumulative))
        kinds = tuple(kinds)
        if (not cumulative or len(cumulative) != len(kinds) or first_kind == second_kind
                or any(not isfinite(x) or x < 0. for x in cumulative)
                or any(x > y for x, y in zip(cumulative, cumulative[1:])) or cumulative[-1] <= 0.):
            raise ValueError('Require the original finite, increasing candidate CDF and distinct kinds.')
        total = cumulative[-1]
        self._ends = tuple(_first_at_least(value, total, self.cells) for value in cumulative)
        if self._ends[-1] != self.cells:
            raise ValueError('The top finite mark rounds out of the original candidate support.')

        starts, rank_ends = [[], []], [[], []]
        self._rows = []
        begin = 0
        for kind, end in zip(kinds, self._ends, strict=True):
            group = 0 if kind == first_kind else 1 if kind == second_kind else None
            rank = 0
            if group is not None and end > begin:
                rank = rank_ends[group][-1] if rank_ends[group] else 0
                starts[group].append(begin)
                rank_ends[group].append(rank+end-begin)
            self._rows.append((group, begin, rank))
            begin = end
        self._groups = tuple(_KindCoordinates(tuple(a), tuple(b)) for a, b in zip(starts, rank_ends, strict=True))

    @property
    def masses(self):
        return tuple(group.mass for group in self._groups)

    def candidate(self, mark):
        if type(mark) is not int or not 0 <= mark < self.cells:
            raise ValueError('Mark must be an integer in the declared binary grid.')
        return bisect_right(self._ends, mark)

    def swap(self, mark, first_phase, second_phase):
        row = self.candidate(mark)
        phases = (first_phase, second_phase)
        masses = self.masses
        if any(type(phase) is not int or not 0 <= phase < max(1, mass)
               for phase, mass in zip(phases, masses, strict=True)):
            raise ValueError('Phases must belong to the two fixed group rank circles.')
        group, begin, prior_rank = self._rows[row]
        paired_mass = min(masses)
        if group is None or not paired_mass:
            return mark
        rank = (prior_rank + mark-begin - phases[group]) % masses[group]
        if rank >= paired_mass:
            return mark
        other = 1-group
        return self._groups[other].mark((phases[other]+rank) % masses[other])
