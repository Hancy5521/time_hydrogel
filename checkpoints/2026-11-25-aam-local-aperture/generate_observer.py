"""Produce and bind the native observation-only hook without editing native source."""
from pathlib import Path

import capture as src


ROOT = Path(__file__).resolve().parent
source = ROOT / "snapshot/native/experiments/aam_pregene/advance_fixed.py"
before = source.read_text(encoding="utf-8")
signature = "def advance_fixed(packet, start, end, brownian, expo, marks, index, remaining, step, force_fn):"
enter = "            apply_event(state, rows[selected], distances[selected], box, local, TICK*half)"
assert before.count(signature) == before.count(enter) == 1
after = before.replace("from aam_spatial_motion import _compact_edges",
                       "from aam_spatial_motion import _compact_edges\nfrom local_observer import observe")
after = after.replace(signature, "def advance_observed(packet, start, end, brownian, expo, marks, index, remaining, step, force_fn, origin, reference, saved, cursor):")
after = after.replace(enter, "            observe(state, rows, primitive, selected, count, box, corridor, origin, reference, saved, cursor, local)\n" + enter)
with (ROOT / "observed_kernels.py").open("x", encoding="utf-8") as stream:
    stream.write(after)

