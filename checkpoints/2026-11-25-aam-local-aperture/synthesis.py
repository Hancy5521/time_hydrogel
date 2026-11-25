"""Transparent two-cohort summary for the identically frozen binary predictor."""
import numpy as np

import capture as src

a_path, b_path = "MOBILE_C_RESULT.json", "REPLICATION_RESULT.json"
a = src.read(a_path)["fair_scores"]
b = src.read(b_path)["scores"]
endpoints = (("against_fixed_four", "conditional_marginal", "constant_four", "mobile_conditional", "constant_four"),
             ("against_same_input_RBF", "conditional_marginal", "event_rbf", "mobile_conditional", "mobile_rbf"),
             ("against_coobserved_fixed_camera", "conditional_marginal", "co_located_fixed", "mobile_conditional", "fixed_sensor"))
r = np.random.default_rng(4827093)
index_a = r.integers(112, size=(30000, 112))
index_b = r.integers(224, size=(30000, 224))
outputs = {}
for title, left_a, right_a, left_b, right_b in endpoints:
    x = np.asarray(a[left_a]["energy"]["card"]) - np.asarray(a[right_a]["energy"]["card"])
    y = np.asarray(b[left_b]["energy"]["card"]) - np.asarray(b[right_b]["energy"]["card"])
    assert x.shape == (112,) and y.shape == (224,)
    plan = (x[index_a].sum(axis=-1) + y[index_b].sum(axis=-1)) / (len(x) + len(y))
    outputs[title] = {"cohort_A": float(x.mean()), "cohort_B": float(y.mean()), 
                      "pooled_delta": float((x.sum() + y.sum()) / (len(x) + len(y))),
                      "ci95_stratified_bootstrap": np.quantile(plan, [.025, .975]).tolist()}
src.write("BINARY_EVIDENCE_SYNTHESIS.json", {"mode": "post-score fixed-method stratified two-study summary",
          "input_hashes": {q: src.digest(src.ROOT / q) for q in (a_path, b_path)},
          "fixed_cameras": "coobserved per cohort; same decoder as mobile",
          "cohort_card_counts": [112, 224], "effects": outputs,
          "restrictions": "not a substitute for primary confirmation; no new fit, data or action choices"})

