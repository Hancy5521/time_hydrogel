"""Write a localized observer that approximates potential contacts from visible geometry."""
from pathlib import Path

D = Path(__file__).resolve().parent

def out(name, text):
    with (D / name).open("x", encoding="utf-8") as f:
        f.write(text)

base = (D / "minimal_geometry.py").read_text(encoding="utf-8")
base = base.replace("def minimal(state, source, target, box, radius, origin, saved, count, time):",
                "def proxy(state, source, target, box, radius, capture, inner, origin, reference, saved, count, time):")
part = '''    q = crowd(x, source, target, seen, box, radius)
    slot = count[0]
    saved[slot, :4] = 1.
    saved[slot, 4] = 2 * int(q > 0) + int(bridge)
    saved[slot, 5] = time
    count[0] += 1'''
assert base.count(part) == 1
new = '''    q_observed = crowd(x, source, target, seen, box, radius)
    observed_group = 2 * int(q_observed > 0) + int(bridge)
    component = np.empty(len(seen), dtype=np.int64)
    boundary = np.zeros(len(seen), dtype=np.bool_)
    for k in range(len(seen)):
        component[k] = root(parent, k)
    for u, v in state.edges[:state.counts[1]]:
        if seen[u] and not seen[v]:
            boundary[component[u]] = True
        elif seen[v] and not seen[u]:
            boundary[component[v]] = True
    degree = np.zeros(len(seen), dtype=np.int64)
    for i in range(len(seen)):
        if seen[i]:
            degree[i] = int(state.owner[2*i] >= 0) + int(state.owner[2*i + 1] >= 0)
    risk = np.zeros(4)
    for i in range(len(seen)):
        if not seen[i] or degree[i] < 1:
            continue
        for j in range(64, len(seen)):
            if not seen[j] or i == j or degree[j] != 1:
                continue
            x2 = 0.
            for ax in range(3):
                df = x[j, ax] - x[i, ax]
                df -= box * np.rint(df / box)
                x2 += df * df
            if x2 >= capture * capture or not inside_corridor(x, i, j, box, radius):
                continue
            left, right = component[i], component[j]
            is_bridge = left != right
            if is_bridge and boundary[left] and boundary[right]:
                continue
            q = crowd(x, i, j, seen, box, radius)
            group = 2 * int(q > 0) + int(is_bridge)
            band = int(x2 >= inner * inner)
            if is_bridge:
                risk[group] += reference[0, q, band]
            else:
                risk[group] += .5 * (reference[1, q, band] + reference[2, q, band])
    if risk[observed_group] <= 0.:
        return
    if int(risk[0] + risk[1] > 0.) + int(risk[2] > 0.) + int(risk[3] > 0.) < 2:
        return
    slot = count[0]
    saved[slot, :4] = risk
    saved[slot, 4] = observed_group
    saved[slot, 5] = time
    count[0] += 1'''
out("proxy_geometry.py", base.replace(part, new))
step = (D / "minimal_kernels.py").read_text(encoding="utf-8")
step = step.replace("def advance_minimal(", "def advance_proxy(")
step = step.replace("from minimal_geometry import minimal", "from proxy_geometry import proxy")
step = step.replace("minimal(state, rows[selected, 1] // 2, rows[selected, 2] // 2, box, corridor, moving, fixed, fixed_cursor, local)",
           "proxy(state, rows[selected, 1] // 2, rows[selected, 2] // 2, box, corridor, capture, inner, moving, reference, fixed, fixed_cursor, local)")
out("proxy_kernels.py", step)

