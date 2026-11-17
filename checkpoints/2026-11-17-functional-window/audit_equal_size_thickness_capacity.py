"""Bound feasibility over every equally sized subcatalogue of frozen geometric designs."""

import argparse
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

import numpy as np

from gel_reactive_port_timing import read_json, sha256
from gel_reactive_two_read_assay import write_new


def capacity_bounds(rows, catalogue, margin):
    old = catalogue['grid25']
    geometry = catalogue['thickness27']
    removals = list(combinations(geometry, len(geometry) - len(old)))
    values = np.asarray([row['margins'][margin] for row in rows])
    feasible = (values <= 0).reshape(-1, values.shape[-1])
    base = int(feasible[:, old].any(axis=1).sum())
    full = int(feasible[:, geometry].any(axis=1).sum())
    counts = feasible[:, geometry].sum(axis=1)
    reduced = {}
    for dropped in removals:
        keep = [i for i in geometry if i not in dropped]
        brute = int(feasible[:, keep].any(axis=1).sum())
        fast = int((counts > feasible[:, dropped].sum(axis=1)).sum())
        if brute != fast:
            raise ArithmeticError(f'Independent counts disagree for {dropped}.')
        reduced[dropped] = brute
    worst = min(reduced, key=reduced.get)
    best = max(reduced, key=reduced.get)
    return {'mothers': len(rows), 'requests': len(feasible), 'same_size_choices': len(old),
            'pairs_removed': len(removals), 'grid_capacity': base, 'full_geometry_capacity': full,
            'min_equal_size_capacity': reduced[worst], 'max_equal_size_capacity': reduced[best],
            'worst_removed_ids': worst, 'best_removed_ids': best}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('run', type=Path)
    p.add_argument('output', type=Path)
    args = p.parse_args()
    protocol = args.run / 'REGISTERED.json'
    scored = args.run / 'SCORED.json'
    plan, data = read_json(protocol), read_json(scored)
    if data['registered_sha256'] != sha256(protocol):
        raise ValueError('The saved labels belong to a different protocol.')
    cases = plan['thickness']['catalogues']
    results = {split: capacity_bounds(data['splits'][split], cases, 'fixed_area_conductance')
               for split in ('test', 'shift_test')}
    write_new(args.output, {'utc': datetime.now(timezone.utc).isoformat(), 'protocol_sha256': sha256(protocol),
                           'scored_sha256': sha256(scored), 'source_sha256': sha256(Path(__file__)),
                           'scope': 'After-the-fact capacity sensitivity only. No policies changed or assessed.',
                           'results': results})
    print(results)


if __name__ == '__main__':
    main()
