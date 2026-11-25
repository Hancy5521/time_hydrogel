"""Independent replay and held-output audit of masked transition snapshots."""
from collections import Counter
from datetime import datetime, timedelta, timezone
from hashlib import file_digest, sha256
from itertools import combinations, product
import json
from pathlib import Path

import numpy as np


from public_integrity import FROZEN_RUN_ID


ROOT = Path(__file__).resolve().parent
METHODS = ("prior", "legacy_ridge", "legacy_rbf", "density_rbf",
           "enhanced_ridge", "enhanced_rbf", "transition_projection")


def read(name):
    return json.loads((ROOT / name).read_text(encoding="utf-8-sig"))


def digest(path):
    from public_integrity import original_digest

    return original_digest(path)


def check(x, y):
    np.testing.assert_allclose(x, y, rtol=0., atol=6.e-13)


def component_members(vertices, edges):
    neighbors = {int(u): set() for u in vertices}
    for a, b in edges:
        neighbors[a].add(b)
        neighbors[b].add(a)
    membership = {}
    for node in neighbors:
        if node in membership:
            continue
        visited, stack = set(), [node]
        while stack:
            point = stack.pop()
            if point in visited:
                continue
            visited.add(point)
            stack.extend(neighbors[point] - visited)
        group = frozenset(visited)
        membership.update({u: group for u in group})
    return membership


def decode_frame(positions, occupied, raw_edges, box):
    observed = {i for i, n in enumerate(occupied) if n >= 0}
    for i in range(78):
        if i in observed:
            assert 0 <= occupied[i] <= (1 if i < 64 else 2)
            assert (positions[i] >= 0.).all() and (positions[i] < .70 * box).all()
        else:
            assert occupied[i] == -1 and (positions[i] == -1.).all()
    raw = []
    for a, b in raw_edges:
        if a == -1:
            assert b == -1
            continue
        assert a in observed and b in observed
        raw.append(tuple(sorted((a, b))))
    edges = Counter(raw)
    member = component_members(observed, edges)
    return observed, edges, member


def minimum_image(vector, box):
    return vector - box * np.round(vector / box)


def legacy_features(positions, occupied, frame, box, capture):
    visible, edges, member = frame
    if not visible:
        return np.zeros(14)
    groups = set(member.values())
    aa = [u for u in visible if u < 64]
    mb = [u for u in visible if u >= 64]
    total = len(visible)
    sizes = [len(s) for s in groups]
    num_edges = sum(edges.values())
    close_pairs = 0
    for a, b in combinations(visible, 2):
        close_pairs += np.linalg.norm(minimum_image(positions[b] - positions[a], box)) < capture
    patch = []
    for ax in (0, 1, 2):
        clear = .70 * box
        connected = [len(s) for s in groups if
                     any(positions[u, ax] < .25 * clear for u in s) and
                     any(positions[u, ax] > .75 * clear for u in s)]
        patch.append(sum(connected) / total)
    return np.array([len(aa) / 64., len(mb) / 14.,
                     sum(occupied[u] for u in aa) / max(1, len(aa)),
                     sum(occupied[u] for u in mb) / (2. * max(1, len(mb))),
                     sum(occupied[u] == 2 for u in mb) / max(1, len(mb)),
                     num_edges / total, max(sizes) / total, sum(v**2 for v in sizes) / total**2,
                     sizes.count(1) / total, (num_edges - total + len(sizes)) / total,
                     close_pairs / total, *patch])


def transition_counts(positions, occupied, past, future, pair, bounds):
    box, radius, capture = bounds
    past_ids, past_edges, past_member = past
    future_ids, future_edges, _ = future
    degree, later = occupied[pair[0]], occupied[pair[1]]
    xy = positions[pair[0]]
    eligible = past_ids & future_ids
    aa = [u for u in eligible if u < 64 and degree[u] >= 1]
    mb = [u for u in eligible if u >= 64 and degree[u] == 1]
    counts = np.zeros((2, 4), dtype=int)
    for a, b in product(aa, mb):
        displacement = minimum_image(xy[b] - xy[a], box)
        if np.linalg.norm(displacement) >= capture:
            continue
        midpoint = xy[a] + displacement / 2.
        if min(midpoint) <= radius or max(midpoint) >= .70 * box - radius:
            continue
        crowded = any(np.linalg.norm(minimum_image(xy[u] - midpoint, box)) < radius
                      for u in past_ids - {a, b})
        connected = b in past_member[a]
        index = 2 * int(crowded) + int(connected)
        counts[0, index] += 1
        if later[b] == 2 and future_edges[(a, b)] > past_edges[(a, b)]:
            counts[1, index] += 1
    return counts


def raw_checks(dataset, transition_pairs):
    geom = dataset["geometry"]
    total_opportunities, total_events = (dataset["counts"][..., j, :].sum() for j in (0, 1))
    assert (dataset["counts"][..., 1, :] <= dataset["counts"][..., 0, :]).all()
    assert np.issubdtype(dataset["counts"].dtype, np.integer)
    sequences, times = dataset["audit_x"].shape[:2]
    assert (sequences, times) == (4, 10)
    x_flat = dataset["x"].reshape(-1, times, 14)
    c_flat = dataset["counts"].reshape(-1, len(transition_pairs), 2, 4)
    for i, (xyz, occupied, hidden_filtered) in enumerate(
            zip(dataset["points"], dataset["occupied"], dataset["edges"], strict=True)):
        frames = [decode_frame(z, occ, el, geom[0])
                  for z, occ, el in zip(xyz, occupied, hidden_filtered, strict=True)]
        for ti in range(times):
            table = legacy_features(xyz[ti], occupied[ti], frames[ti], geom[0], geom[2])
            check(table, dataset["audit_x"][i, ti])
            check(table, x_flat[i, ti])
        for j, (a, b) in enumerate(transition_pairs):
            new = transition_counts(xyz, occupied, frames[a], frames[b], (a, b), geom)
            np.testing.assert_array_equal(new, c_flat[i, j])
    return {"full_role_geometric_opportunities": int(total_opportunities),
            "full_role_detected_link_transitions": int(total_events),
            "raw_illustrative_sequences": sequences,
            "raw_frame_summaries_derived": sequences * times,
            "raw_transition_tables_derived": sequences * len(transition_pairs)}


def role_and_hash_checks():
    names = ("REGISTERED.json", "DEVELOPMENT.json", "MODEL_REGISTERED.json", "MODEL_FREEZE.json",
             "CONFIRMED.json", "PREDICTION_FREEZE.json", "SCORED.json")
    r, dev, model, freeze, confirm, lock, score = map(read, names)
    times = [datetime.fromisoformat(record["utc"]) for record in (r, dev, model, freeze, confirm, lock, score)]
    assert all(a < b for a, b in zip(times[:-1], times[1:]))
    for path, d in r["source_sha256"].items():
        assert digest(path) == d, path
    assert digest(ROOT / "REGISTERED.json") == freeze["registration_sha256"]
    assert digest(ROOT / "fit.py") == score["score_source_sha256"] == freeze["model_source_sha256"] == model["source_sha256"]
    assert digest(ROOT / "MODEL.npz") == freeze["model_arrays_sha256"]
    assert digest(ROOT / "MODEL_FREEZE.json") == lock["model_freeze"] == confirm["model_freeze"]
    assert digest(ROOT / "PREDICTION_FREEZE.json") == score["freeze_sha256"]
    assert digest(ROOT / "PREDICTED.npz") == lock["predictions_sha256"]
    assert digest(ROOT / "TEST.npz") == lock["test_input_sha256"]
    assert not freeze["test_existed_at_freeze"]
    assert freeze["development_sha256"] == dev["raw_sha256"] and score["selected"] == freeze["model"]
    data, addresses = {}, set()
    for role, size in r["roles"].items():
        filename = role.upper() + ".npz"
        assert digest(ROOT / filename) == (confirm if role == "test" else dev)["raw_sha256"][filename]
        with np.load(ROOT / filename, allow_pickle=False) as packed:
            vals = {key: packed[key] for key in packed.files}
        assert vals["x"].shape == (size, 2, 2, 4, 10, 14)
        assert vals["counts"].shape[:4] == (size, 2, 2, 4)
        assert vals["addresses"].shape == (size, 2, 4)
        namespace = f"{FROZEN_RUN_ID}|masked-graph-transition|{role}|monomer-addresses"
        seed = int.from_bytes(sha256(namespace.encode()).digest()[:16], "little")
        expected = np.random.default_rng(seed).integers(0, 2**63 - 1, (size, 2, 4))
        np.testing.assert_array_equal(vals["addresses"], expected)
        used = set(expected.ravel())
        assert len(used) == expected.size and not addresses.intersection(used)
        addresses.update(used)
        data[role] = vals
    assert len(addresses) == r["unique_addresses"] == 2432
    parent = ROOT / next(Path(path).parent for path in r["source_sha256"] if Path(path).name == "FAIR_ENERGY_DIAGNOSTIC.json")
    for candidate in parent.glob("*.npz"):
        with np.load(candidate, allow_pickle=False) as arr:
            if "seeds" in arr:
                assert not addresses.intersection(arr["seeds"].ravel())
    return r, freeze, score, data, times


def count_by_window(data, bounds, indices):
    selected = np.stack([data["counts"][..., i, :, :] for i in indices], axis=4)
    return selected.sum(axis=3)


def feature_rows(data, frames, indices, rate, kind):
    shape = data["x"].shape
    grouped = np.stack([data["x"][..., t, :] for t in frames], axis=4)
    means, sd = grouped.mean(axis=3), grouped.std(axis=3)
    view = np.empty((*shape[:2], 168))
    view[:, :, 0::2] = means.reshape(*shape[:2], -1)
    view[:, :, 1::2] = sd.reshape(*shape[:2], -1)
    pieces = [view]
    if kind != "legacy":
        count = count_by_window(data, frames, indices)
        expo, events = count[..., 0, :], count[..., 1, :]
        pieces.append(np.log1p(expo).reshape(*shape[:2], -1))
        if kind == "enhanced":
            pieces.append(np.log1p(events).reshape(*shape[:2], -1))
            signed = (events - rate * expo) / np.sqrt(1. + expo * rate * (1. - rate))
            pieces.append(signed.reshape(*shape[:2], -1))
    return np.concatenate(pieces, axis=-1).reshape(2 * shape[0], -1)


def predict(x, cfg, get):
    if cfg["algorithm"] == "ridge":
        v = .5 + x @ get("coef")
    elif cfg["algorithm"] == "rbf":
        support = get("support")
        squared = np.stack([((support - row)**2).mean(axis=1) for row in x])
        v = .5 + np.exp(-cfg["gamma"] * squared) @ get("dual")
    else:
        logits = ((x - get("midpoint")) @ get("coef")) * cfg["scale"]
        v = 1. / (1. + np.exp(-logits))
    return np.clip(v, .001, .999)


def confidence(values, boot, record):
    row = record["card"]
    check(values, row)
    check(np.mean(values), record["mean"])
    check(np.quantile(np.take(values, boot).mean(axis=1), (.025, .975)), record["ci95"])


def audit():
    reg, freeze, score, groups, times = role_and_hash_checks()
    pairs = list(map(tuple, reg["transitions"]))
    checked = {role: raw_checks(g, pairs) for role, g in groups.items()}
    held_stats, max_diff = {}, 0.
    with np.load(ROOT / "MODEL.npz", allow_pickle=False) as arrays, np.load(
            ROOT / "PREDICTED.npz", allow_pickle=False) as issued:
        assert (issued["prior"] == .5).all()
        for name in METHODS:
            p = issued[name]
            if name != "prior":
                cfg = freeze["model"][name]
                get = lambda key: arrays[f"{name}_{key}"]
                window = cfg["window"]
                countframes = reg["menu"][window]
                ix = [pairs.index(x) for x in zip(countframes[:-1], countframes[1:], strict=True)]
                train_count = count_by_window(groups["train"], countframes, ix).sum(axis=(0, 1))
                inferred = (train_count[..., 1, :] + .5) / (train_count[..., 0, :] + 1.)
                check(inferred, get("rates"))
                raw = {key: feature_rows(vals, countframes, ix, inferred, cfg["mode"]) for key, vals in groups.items()}
                mean, sd = raw["train"].mean(axis=0), raw["train"].std(axis=0)
                check(mean, get("center"))
                check(np.maximum(sd, .1 * np.median(sd[sd > 0.])), get("norm"))
                data = {role: (raw[role] - get("center")) / get("norm") for role in raw}
                if cfg["algorithm"] == "rbf":
                    check(data["train"], get("support"))
                q = predict(data["test"], cfg, get)
                val = predict(data["validate"], cfg, get)
                check(np.mean((val - np.tile([0, 1], len(val) // 2))**2), cfg["brier"])
                check(cfg["brier"], min(v for k, v in freeze["validation_brier"].items()
                                        if k.split(":")[0] == name))
                check(q.reshape(96, 2), p)
                check(predict(data["test"][[0, 151, 83]], cfg, get), q[[0, 151, 83]])
                max_diff = max(max_diff, float(np.max(np.abs(q - p.ravel()))))
            scores = []
            for low, high in p:
                hits = (float(low < .5) + float(high > .5)
                        + .5 * (float(low == .5) + float(high == .5))) / 2.
                brier = (low**2 + (1. - high)**2) / 2.
                rank = .5 if low == high else float(high > low)
                scores.append([hits, brier, rank])
            held_stats[name] = np.array(scores)
    boot = np.random.default_rng(445112).integers(96, size=(18000, 96))
    for method, s in held_stats.items():
        for j, label in enumerate(("accuracy", "brier", "paired_rank")):
            reference = score["models"][method][label]
            confidence(s[:, j], boot, reference["value"])
            for control in ("legacy_rbf", "enhanced_rbf", "prior"):
                confidence(s[:, j] - held_stats[control][:, j], boot, reference[f"vs_{control}"])
    return {
        "status": "passed",
        "score_sha256": digest(ROOT / "SCORED.json"),
        "audit_source_sha256": digest(__file__),
        "chronological_utc": [t.isoformat() for t in times],
        "witness_addresses": reg["unique_addresses"],
        "independently_reconstructed_held_probability_count": 7 * 96 * 2,
        "max_probability_error": max_diff,
        "scored_card_metrics_crosschecked": 7 * 96 * 3,
        "model_scope": "Three chosen frames of ten generated frames; fields read from simulated masked apertures",
        "observation_accounting": checked,
        "audit_scope": ["Retained illustrative raw windows are the first four trajectories per role, not all raw frames.",
                        "Visible endpoint site occupation, species, and stable identifiers are instrument assumptions.",
                        "Local closure comes from the induced snapshot graph; nonconnected in aperture does not imply global bridge.",
                        "Held class is set by simulation branch order; row predictions do not use branch order or mate observation."],
    }


if __name__ == "__main__":
    output = audit()
    now = datetime.now(timezone.utc)
    output["utc"] = now.isoformat()
    output["asia_shanghai"] = now.astimezone(timezone(timedelta(hours=8))).isoformat()
    with (ROOT / "INDEPENDENT_AUDIT.json").open("x", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
        f.write("\n")
    print(output)
