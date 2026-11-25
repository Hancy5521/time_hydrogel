"""Independently reconstruct the frozen witness-clock pulse result."""
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path

import numpy as np

from independent_continuum_witness_audit import cdf_score, generator, reconstruct_conditional
from independent_fixed_event_audit import check, compare, digest, read


ROOT = Path(__file__).resolve().parent


def audit():
    reg, conf, score = map(read, ("PULSE_REGISTERED.json", "PULSE_CONFIRMED.json", "PULSE_RESULT.json"))
    d = lambda name: digest(ROOT / name)
    times = [datetime.fromisoformat(v["utc"]) for v in (reg, conf, score)]
    assert times[0] < times[1] < times[2]
    assert d("PULSE_REGISTERED.json") == conf["registration_sha256"] == score["registration_sha256"]
    assert d("PULSE_RAW.npz") == conf["raw_sha256"] == score["raw_sha256"]
    for name, checksum in reg["source_sha256"].items():
        assert digest(name) == checksum
    assert (reg["methods_fitted"], reg["C_outcome_access"], reg["witness_messages_max"]) == (0, False, 3)
    low, high = reg["known_conditions"]
    assert low == .25
    check(low*2.05 + high*(4.85-2.05), 4.85)
    assert score["frozen_alpha"] == read("CONTINUUM_MODEL_FREEZE.json")["models"]["mobile_conditional"]["alpha"]
    with np.load(ROOT / "PULSE_RAW.npz", allow_pickle=False) as pack:
        marks, addresses = pack["marks"], pack["seeds"]
    assert marks.shape == (128, 2, 2, 2, 4, 3, 6)
    assert addresses.shape == (128, 2, 4)
    assert len(set(addresses.ravel())) == addresses.size == 1024
    assert conf["paths"] == int(np.prod(marks.shape[:5]))
    ids = generator("pulse-witness", "paired initial states").integers(0, 2**63-1, addresses.shape)
    np.testing.assert_array_equal(ids, addresses)
    older = {"TRAIN", "VALIDATE", "TEST", "EVENT_FIT", "EVENT_DEV", "EVENT_CHECK", "MOBILE_FIT", "MOBILE_DEV",
             "MOBILE_CHECK", "EVENT_C_FORECAST", "EVENT_C_TRUTH", "MOBILE_C_FORECAST", "MOBILE_C_TRUTH",
             "CONTINUUM_FIT", "CONTINUUM_DEV", "CONTINUUM_HELD"}
    for title in older:
        with np.load(ROOT / f"{title}.npz", allow_pickle=False) as previous:
            key = "addresses" if title in ("TRAIN", "VALIDATE", "TEST") else "seeds"
            assert not set(ids.ravel()).intersection(previous[key].ravel())
    rng = generator("pulse-witness", "rules")
    rng.uniform(-.48, .48, (128, 4))
    size = rng.uniform(0., 1.1, 128)
    target = np.c_[-size, size]
    losses = []
    captures = []
    for q in (0, 1):
        group = marks[:, :, q]
        p = reconstruct_conditional(group, score["frozen_alpha"])
        loss = cdf_score(p, target).reshape(128, 2)
        losses.append(loss.mean(axis=1))
        sign_tally = (group[..., 4] >= 0).reshape(128, 2, 24).sum(axis=-1)
        captures.append(sign_tally.mean(axis=-1))
    nominal, pulse = losses
    original_seen, new_seen = captures
    draws = np.random.default_rng(7437736).integers(128, size=(16000, 128))
    for name, values in (("nominal_crps", nominal), ("pulse_crps", pulse),
                         ("gain", nominal - pulse), ("capture_gain", new_seen - original_seen)):
        compare(score[name], values, draws)
    check([original_seen.mean(), new_seen.mean()], score["capture_nominal_to_pulse"])
    adopted = score["gain"]["ci95"][0] > 0. and score["capture_gain"]["ci95"][0] > 0.
    assert score["passed"] is adopted is False
    assert not list(ROOT.glob("PULSE_C_*.npz"))
    return {"status": "passed", "paired_held_addresses": addresses.size,
            "trajectories": conf["paths"],
            "nominal_dose_check": low*2.05 + high*(4.85-2.05),
            "source_sha256": digest(__file__), "result_sha256": d("PULSE_RESULT.json"),
            "chronological_utc": [t.isoformat() for t in times],
            "frozen_law_integrated_cdf_scores_reproduced": 128*2*2,
            "candidate_gate": adopted, "candidate_C_paths": 0,
            "interpretation": "The paired pulse increases visible capture but the CRPS gain interval crosses zero. Equal integrated input multiplier does not imply equal realized reaction count."}


if __name__ == "__main__":
    receipt = audit()
    now = datetime.now(timezone.utc)
    receipt["utc"] = now.isoformat()
    receipt["asia_shanghai"] = now.astimezone(timezone(timedelta(hours=8))).isoformat()
    with (ROOT / "PULSE_INDEPENDENT_AUDIT.json").open("x", encoding="utf-8") as o:
        json.dump(receipt, o, indent=2)
        o.write("\n")
    print(receipt)
