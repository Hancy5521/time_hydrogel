"""Write the mobile aperture implementation as a narrow, reproducible observer variant."""
from pathlib import Path

import capture as src


ROOT = Path(__file__).resolve().parent


def make(name, text):
    with (ROOT / name).open("x", encoding="utf-8") as out:
        out.write(text)


native = (ROOT / "observed_kernels.py").read_text(encoding="utf-8")
signal = "            observe(state, rows, primitive, selected, count, box, corridor, origin, reference, saved, cursor, local)"
assert native.count(signal) == 1
extra = '''
            observe(state, rows, primitive, selected, count, box, corridor, origin, reference, fixed, fixed_cursor, local)
            if 2.05 <= local <= 4.85 and rows[selected, 4] >= 0 and cursor[0] < len(saved):
                moving = mobile_origin(state.points, rows[selected, 1] // 2, rows[selected, 2] // 2,
                                       box, corridor, dither[index])
                observe(state, rows, primitive, selected, count, box, corridor, moving, reference, saved, cursor, local)
'''
native = native.replace(signal, extra.rstrip()).replace(
    "from local_observer import observe", "from local_observer import observe\nfrom mobile_geometry import mobile_origin")
native = native.replace("def advance_observed(", "def advance_mobile(").replace(
    "origin, reference, saved, cursor):", "origin, reference, saved, cursor, dither, fixed, fixed_cursor):")
make("mobile_kernels.py", native)

cap = (ROOT / "event_capture.py").read_text(encoding="utf-8")
cap = cap.replace('"EVENT_', '"MOBILE_').replace('"event-gate-', '"mobile-gate-').replace(
    '"local_observer.py", "generate_observer.py",\n               "observed_kernels.py", "event_capture.py", "SCORED.json"',
    '"local_observer.py", "mobile_geometry.py", "generate_mobile.py",\n               "mobile_kernels.py", "mobile_capture.py", "EVENT_SCORED.json"')
cap = cap.replace('"MOBILE_PROTOCOL_ZH.md"', '"MOBILE_PROTOCOL_ZH.md"')
cap = cap.replace("from observed_kernels import advance_observed", "from mobile_kernels import advance_mobile")
cap = cap.replace("    ref = reference()", """    ref = reference()
    fixed = np.full_like(rows, -1.)
    fixed_cursor = np.zeros_like(found)
    dither = src.rng("mobile-aperture-dither", str(seed)).uniform(size=(160, 3))""")
cap = cap.replace("return advance_observed(*args, origin, ref, rows, found)",
                  "return advance_mobile(*args, origin, ref, rows, found, dither, fixed, fixed_cursor)")
cap = cap.replace('return rows, got["local"]', 'return rows, got["local"], fixed')
cap = cap.replace("for role in src.ROLES:", """for role in ("FIT", "DEV", "CHECK"):
        with np.load(ROOT / ("EVENT_" + role + ".npz"), allow_pickle=False) as earlier:
            assert not np.intersect1d(joined, earlier["seeds"]).size
    for role in src.ROLES:""")
cap = cap.replace("mark, x = (np.array(values)", "mark, x, fixed = (np.array(values)")
cap = cap.replace("x=x.reshape(*shape, 3, 14), seeds=addr)", "x=x.reshape(*shape, 3, 14), seeds=addr, fixed=fixed.reshape(*shape, 3, 6))")
cap = cap.replace('"synthetic ideal local event trigger with a certified candidate set"',
                  '"synthetic moving and fixed local event cameras in the same native history"')
make("mobile_capture.py", cap)

fit = (ROOT / "event_fit.py").read_text(encoding="utf-8")
fit = fit.replace("import event_capture as event", "import mobile_capture as event").replace('"EVENT_', '"MOBILE_')
fit = fit.replace('"conditional_marginal")', '"conditional_marginal", "co_located_fixed")', 1)
fit = fit.replace('assert src.digest(Path(__file__)) == lock["source_sha256"]\n    held = load("check")',
                  '''assert src.digest(Path(__file__)) == lock["source_sha256"]
    held = load("check")
    with np.load(ROOT / "MOBILE_CHECK.npz", allow_pickle=False) as payload:
        fixed = payload["fixed"]''')
fit = fit.replace('results[name] = p.reshape(n, 2)', '''results[name] = p.reshape(n, 2)
    p_fixed = expit(lock["models"]["conditional_marginal"]["alpha"] * conditional(fixed, True))
    results["co_located_fixed"] = p_fixed.reshape(n, 2)''')
fit = fit.replace('marks, _ = load("check")', '''marks, _ = load("check")
    with np.load(ROOT / "MOBILE_CHECK.npz", allow_pickle=False) as f:
        fixed_events = int((f["fixed"][..., 4] >= 0).sum())''')
fit = fit.replace('"results": rows, "captures_total":', '"results": rows, "fixed_captures": fixed_events, "captures_total":')
make("mobile_fit.py", fit)

