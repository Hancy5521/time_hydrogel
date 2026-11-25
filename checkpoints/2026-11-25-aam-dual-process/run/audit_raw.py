"""Verify complete saved confirmation tapes, with no simulation or model fitting."""

from argparse import ArgumentParser
from collections import Counter
from datetime import datetime, timedelta, timezone
from hashlib import file_digest
from pathlib import Path
import json

import numpy as np

from audit_reaction import replay, sources


ROOT = Path(__file__).resolve().parent
REGISTERED_FEED = 1.95
INDEPENDENT_SEED_RANGES = (
    ("DEVELOPMENT.npz", "seeds"),
    ("QUERY_LOCAL.npz", "seeds"),
    ("ZERO_C_ANCHOR.npz", "seeds"),
    ("PREDICTED.npz", "forward_seeds"),
    ("PREDICTED.npz", "planning_seeds"),
    ("CONFIRMED.npz", "seeds"),
)
ENGINEERING_SEEDS = set(range(8914112, 8914152))


def sha(path):
    with path.open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def seed_audit(directory):
    totals = {}
    seen = ENGINEERING_SEEDS.copy()
    for archive, column in INDEPENDENT_SEED_RANGES:
        with np.load(directory/archive, allow_pickle=False) as saved:
            values = saved[column].ravel()
            addresses = set(map(int, values))
        if addresses & seen or len(addresses) != len(values):
            raise ArithmeticError(f"Reused independent random address in {archive}, {column}.")
        totals[f"{archive}:{column}"] = len(addresses)
        seen.update(addresses)
    return totals


def expected_journal_keys(frozen):
    selections = np.asarray(frozen["selected_actions"], dtype=int)
    return sorted((card, action, 0)
                  for card in range(selections.shape[-1])
                  for action in set(selections[..., card].ravel()) | {0})


def check(root, frozen_sha):
    freeze_path = root/"FROZEN.json"
    if sha(freeze_path) != frozen_sha:
        raise ValueError("Before reading held histories, supply the correct locked FROZEN digest.")
    frozen = read(freeze_path)
    confirmed = read(root/"CONFIRMED.json")
    if (confirmed["verified_prediction_sha256"] != frozen_sha
            or confirmed["sha256"] != sha(root/"CONFIRMED.npz")
            or frozen["registration_sha256"] != sha(root/"REGISTERED.json")
            or frozen["predicted_sha256"] != sha(root/"PREDICTED.npz")
            or frozen["acquisition_sha256"] != sha(root/"ACQUISITION.json")
            or datetime.fromisoformat(confirmed["utc"]) <= datetime.fromisoformat(frozen["utc"])):
        raise ArithmeticError("Registered access order or source/forecast label link differs.")
    registration = read(root/"REGISTERED.json")
    if registration["new_context"] != "C":
        raise ValueError("Audit clock is registered for the specific held C environment.")
    if any(sha(Path(p)) != h for p, h in registration["source_sha256"].items()):
        raise ArithmeticError("A registered source changed after the run.")
    acquired = read(root/"ACQUISITION.json")
    if any(sha(root/name) != h for name,h in acquired["inputs"].items()):
        raise ArithmeticError("A registered early input or escrow changed after the run.")
    known_keys = np.array(expected_journal_keys(frozen))
    max_snapshot_diff, max_milestone_diff = 0., 0.
    meta = Counter()
    ever_reached, wound = 0, 0
    with np.load(root/"CONFIRMED.npz", allow_pickle=False) as values:
        paths, times, seeds, raw, keys, offsets = (values[key] for key in (
            "path", "milestones", "seeds", "journals", "journal_keys", "journal_offsets",
        ))
        if (keys.shape != known_keys.shape or not np.array_equal(keys, known_keys)
                or keys.shape[0] != confirmed["journal_trace_count"]
                or len(offsets) != len(keys)+1 or offsets[0] != 0 or offsets[-1] != len(raw)
                or np.any(np.diff(offsets) <= 0)):
            raise ArithmeticError("Journal keys or offsets do not cover the promised selections.")
        i = registration["division"]
        shape = (i["confirmation_cards"], len(frozen["actions"]), i["validation_draws_per_action_card"])
        if paths.shape != (*shape, 6, 10) or times.shape != (*shape, 2, 5) or seeds.shape != shape[::2]:
            raise ArithmeticError("Simulation archive differs from the registered dimensions.")
        for (card, variant, repeat), a, b in zip(keys, offsets[:-1], offsets[1:], strict=True):
            address = int(seeds[card, repeat])
            # Independent implementation of the only initialization random variable
            # needed to replay legal reaction marks. Code never alters the budget draw.
            child = np.random.SeedSequence(address).spawn(5)[0]
            initial_radicals = int(np.random.default_rng(child).integers(6, 13))
            path, milestones, tags = replay(raw[a:b], initial_radicals, REGISTERED_FEED)
            d1 = np.max(np.abs(path-paths[card,variant,repeat]))
            d2 = np.max(np.abs(milestones-times[card,variant,repeat]))
            if not np.isfinite(d1+d2) or max(d1,d2) > 2.e-13:
                raise ArithmeticError(f"Independent exact replay disagrees for {(card,variant,repeat)}: {d1} {d2}")
            max_snapshot_diff, max_milestone_diff = max(max_snapshot_diff,d1), max(max_milestone_diff,d2)
            meta.update(tags)
            ever_reached += int(milestones[0,0])
            wound += int(path[-1,-1])
        with np.load(root/"CODE_ESCROW.npz", allow_pickle=False) as hidden:
            if not np.array_equal(values["code"], hidden["G"]):
                raise ArithmeticError("Saved reference code differs from the locked escrow.")
    utc = datetime.now(timezone.utc)
    return {
        "utc":utc.isoformat(),
        "asia_shanghai":utc.astimezone(timezone(timedelta(hours=8))).isoformat(),
        "frozen_sha256":frozen_sha,
        "confirmed_sha256":confirmed["sha256"],
        "based_on_saved_events_no_new_simulations":True,
        "native_paths_total":int(np.prod(shape)),
        "saved_journals_checked":len(keys),
        "checked_target_snapshots":6*len(keys),
        "checked_milestones":2*len(keys),
        "event_count":len(raw),
        "event_tags":{str(key):value for key,value in sorted(meta.items())},
        "saved_journal_reaches_60_within_horizon":ever_reached,
        "saved_journal_winds_by_end":wound,
        "max_abs_snapshot_difference":float(max_snapshot_diff),
        "max_abs_milestone_difference":float(max_milestone_diff),
        "independent_seed_role_totals":seed_audit(root),
        "audit_sources":{str(Path(__file__).resolve()):sha(Path(__file__)), **sources()},
        "scope":"All saved journals of first-repeat selected conditions and unedited controls. "
                "Replays event topology and inventories, not the underlying real-world chemistry.",
    }


if __name__ == "__main__":
    arg = ArgumentParser(description=__doc__)
    arg.add_argument("--root", type=Path, default=ROOT)
    arg.add_argument("--freeze-sha256", required=True)
    options = arg.parse_args()
    output = check(options.root, options.freeze_sha256)
    with (options.root/"AUDIT_RAW.json").open("x", encoding="utf-8") as destination:
        json.dump(output, destination, indent=2, allow_nan=False)
    print({k:output[k] for k in ("saved_journals_checked", "event_count", "max_abs_snapshot_difference",
                                  "max_abs_milestone_difference")}, flush=True)
