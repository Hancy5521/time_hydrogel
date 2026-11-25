"""Read back native chemistry without calling any trajectory generator."""

from __future__ import annotations

from hashlib import file_digest
from importlib.util import module_from_spec, spec_from_file_location
import os
from pathlib import Path

import numpy as np


DEFAULT_PROJECT = Path(__file__).resolve().parents[1] / "native"
PROJECT = Path(os.environ.get("AAM_REACTION_WORKTREE", str(DEFAULT_PROJECT))).resolve()
AUDIT_SOURCE = PROJECT / "experiments/aam_gene_field/audit.py"
_spec = spec_from_file_location("aam_frozen_gene_audit", AUDIT_SOURCE)
old = module_from_spec(_spec)
_spec.loader.exec_module(old)

READ_TIMES = (.65, 1.35, 2.05, 3.50, 4.85, 6.)
LEVELS = (60, 76)


class ReplayGraph(old.JournalGraph):
    def __init__(self, budget, feed_time):
        super().__init__(budget)
        self.feed_time = feed_time

    def accept(self, row):
        if row[1] != 0:
            return super().accept(row)
        age, _, start, stop, distance, band, midpoint, occupants, before_start, before_stop, *lift = row
        old.require(
            start == -1 and stop == self.present < old.MOLECULES and age == self.feed_time,
            "Unexpected feed, duplicate nominee or wrong observation clock.",
        )
        old.require(
            (distance, band, midpoint, occupants, before_start, before_stop, *lift)
            == (-1., -1., self.feed_time, -1., -1., -1., 0., 0., 0.),
            "Invalid feed ledger fields.",
        )
        self.counts[0] += 1
        self.present += 1


def sources():
    result = {}
    for path in (Path(__file__).resolve(), AUDIT_SOURCE):
        with path.open("rb") as f:
            result[str(path)] = file_digest(f, "sha256").hexdigest()
    return result


def replay(journal, initiators, feed_time):
    """Return full path, exact equal-conversion milestones, and event counts."""
    events = np.asarray(journal, dtype=float)
    old.require(len(events.shape) == 2 and events.shape[1] == 13 and np.isfinite(events).all(),
                "Expected a finite native journal with 13 columns.")
    old.require(6 <= initiators <= 12 and int(initiators) == initiators and 0. < feed_time < READ_TIMES[-1],
                "Invalid registered radical budget or feed time.")
    t = events[:, 0]
    old.require(np.all(np.diff(t) >= 0) and np.all((0. <= t) & (t <= READ_TIMES[-1])),
                "Journal time outside prospective observation horizon or decreasing.")
    discrete = events[:, [1, 2, 3, 5, 7, 8, 9, 10, 11, 12]]
    old.require(np.array_equal(discrete, np.rint(discrete)) and
                np.isin(events[:, 1], np.arange(10)).all(), "Unknown or fractional event label.")
    g = ReplayGraph(initiators, feed_time)
    levels = np.zeros((len(LEVELS), 5))
    levels[:, 1] = READ_TIMES[-1]
    cursor = 0
    path = []
    for time in READ_TIMES:
        while cursor < len(events) and t[cursor] <= time:
            prior = g.used
            g.accept(events[cursor])
            if prior != g.used and g.used in LEVELS:
                m = g.summary()
                levels[LEVELS.index(g.used)] = (1., t[cursor], m[2], m[3], m[-1])
            cursor += 1
        path.append(g.summary())
    old.require(cursor == len(events) and g.present == old.MOLECULES
                and g.counts[0] == old.MBAA - old.INITIAL_MBAA, "Event timing or feed inventory disagrees.")
    old.require(sum(g.active) == g.births - 2*(g.counts[7]+g.counts[8]),
                "Active chain-end balance violates initiation and termination.")
    return np.stack(path), levels, dict(g.counts)


def verify(native):
    """Compare independent linked-graph summary with saved native readouts."""
    actual, lm, counts = replay(native["journal"], native["initiators"], native["feed_time"])
    np.testing.assert_allclose(actual, native["path"], rtol=0., atol=2.e-13)
    np.testing.assert_allclose(lm, native["milestones"], rtol=0., atol=2.e-13)
    return counts
