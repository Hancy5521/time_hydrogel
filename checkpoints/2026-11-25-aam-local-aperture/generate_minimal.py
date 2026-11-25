"""Make a certified chosen-mark-only observer, sharing the native clock."""

from pathlib import Path

import capture as src


ROOT = Path(__file__).resolve().parent
prefix = (ROOT / "local_observer.py").read_text(encoding="utf-8")
headers = prefix[:prefix.index("@njit(cache=True)\ndef observe")]
new = '''
@njit(cache=True)
def minimal(state, source, target, box, radius, origin, saved, count, time):
    if count[0] >= len(saved):
        return
    x = np.empty_like(state.points)
    seen = state.present.copy()
    for node in range(len(seen)):
        for axis in range(3):
            x[node, axis] = (state.points[node, axis] - origin[axis] * box) % box
            if x[node, axis] >= .70 * box:
                seen[node] = False
    if not (seen[source] and seen[target] and inside_corridor(x, source, target, box, radius)):
        return
    parent = np.arange(len(seen))
    for u, v in state.edges[:state.counts[1]]:
        if seen[u] and seen[v]:
            parent[root(parent, v)] = root(parent, u)
    first, second = root(parent, source), root(parent, target)
    open_a, open_b = False, False
    for u, v in state.edges[:state.counts[1]]:
        if seen[v] and not seen[u]:
            u, v = v, u
        if seen[u] and not seen[v]:
            k = root(parent, u)
            open_a = open_a or k == first
            open_b = open_b or k == second
    bridge = first != second
    if bridge and open_a and open_b:
        return
    q = crowd(x, source, target, seen, box, radius)
    slot = count[0]
    saved[slot, :4] = 1.
    saved[slot, 4] = 2 * int(q > 0) + int(bridge)
    saved[slot, 5] = time
    count[0] += 1
'''
with (ROOT / "minimal_geometry.py").open("x", encoding="utf-8") as f:
    f.write(headers + new)
before = (ROOT / "mobile_kernels.py").read_text(encoding="utf-8")
single = "            observe(state, rows, primitive, selected, count, box, corridor, origin, reference, fixed, fixed_cursor, local)"
assert before.count(single) == 1
after = before.replace(single, "").replace("from mobile_geometry import mobile_origin",
               "from mobile_geometry import mobile_origin\nfrom minimal_geometry import minimal")
after = after.replace("def advance_mobile(", "def advance_minimal(").replace(
 "and cursor[0] < len(saved):", "and (cursor[0] < len(saved) or fixed_cursor[0] < len(fixed)):")
observe = "                observe(state, rows, primitive, selected, count, box, corridor, moving, reference, saved, cursor, local)"
assert after.count(observe) == 1
after = after.replace(observe, observe + "\n                minimal(state, rows[selected, 1] // 2, rows[selected, 2] // 2, box, corridor, moving, fixed, fixed_cursor, local)")
with (ROOT / "minimal_kernels.py").open("x", encoding="utf-8") as f:
    f.write(after)

