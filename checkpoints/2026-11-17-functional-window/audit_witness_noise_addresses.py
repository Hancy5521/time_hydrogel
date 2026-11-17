"""Enumerate exact witness noise address overlap within and across experiment roles."""

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from hashlib import sha256
from pathlib import Path
import json

import numpy as np


ROOT = Path(__file__).resolve().parent
CODE = (
    Path(__file__),
    ROOT / "calibrate_thickness_catalogue.py",
    ROOT / "patch_thickness_catalogue_benchmark.py",
    ROOT / "window_spatial_probe.py",
    ROOT / "calibrate_window_spatial_probe.py",
    ROOT / "patch_window_event_study.py",
)


def addresses(plan):
    root_seed = plan["spatial_probe"]["snapshot_noise_seed"]
    rng_keys = plan["seeds"]
    if "event_roles" in plan:
        for role, rec in plan["event_roles"].items():
            n = rec["groups"]
            activity_generator = np.random.default_rng(rec["mechanism_seed"])
            flags = activity_generator.random(n) < rec["shift_fraction"]
            salt = rec["spatial_noise_salt"]
            for i, shift in enumerate(flags):
                # Check needed and possible addresses, so changing an otherwise
                # arbitrary Bernoulli assignment cannot introduce a collision.
                for value in (False, True):
                    logical = "shift_test" if value else "test"
                    yield root_seed + salt + rng_keys[logical] + i, role, i, value, bool(shift) == value
    else:
        salts = plan.get("fresh_calibration", {}).get("noise_salts", {})
        for role, n in plan["sizes"].items():
            shifted = role.startswith("shift_")
            logical = "shift_test" if shifted else "test"
            for i in range(n):
                yield root_seed + salts.get(role, 0) + rng_keys[logical] + i, role, i, shifted, True


def duplicates(items, record_names=()):
    groups = defaultdict(list)
    for key, *record in items:
        groups[key].append(record)
    shared = {str(key): v for key, v in sorted(groups.items()) if len(v) > 1}
    overlaps = Counter(tuple(dict.fromkeys(row[0] for row in v)) for v in shared.values())
    return {
        "address_total": len(items),
        "distinct": len(groups),
        "reused_excess": sum(len(v) - 1 for v in shared.values()),
        "collision_groups": [{"groups": list(k), "n": v} for k, v in sorted(overlaps.items())],
        "seed_to_reused_groups": shared,
    }


def audit(paths):
    now = datetime.now(timezone.utc)
    output = {
        "started_utc": now.isoformat(),
        "started_shanghai": now.astimezone(timezone(timedelta(hours=8))).isoformat(),
        "sources_sha256": {f.name: sha256(f.read_bytes()).hexdigest() for f in CODE},
        "method": "Ephemeral logical seed addresses only. No model outcomes, fit, re-generation "
                  "of physical sample fields, or editing experiment sources.",
    }
    actual_all, possible_all = [], []
    output["protocols"] = {}
    for path in paths:
        manifest = path / "REGISTERED.json"
        content = manifest.read_bytes()
        spec = json.loads(content)
        possible = list(addresses(spec))
        actual = [row for row in possible if row[-1]]
        possible = [row[:-1] for row in possible]
        actual = [row[:-1] for row in actual]
        output["protocols"][path.name] = {
            "registered_sha256": sha256(content).hexdigest(),
            "actual": duplicates(actual),
            "possible": duplicates(possible),
        }
        actual_all.extend((seed, path.name, *row) for seed, *row in actual)
        possible_all.extend((seed, path.name, *row) for seed, *row in possible)
    output["joint_actual"] = duplicates(actual_all)
    output["joint_possible"] = duplicates(possible_all)
    output["completed_utc"] = datetime.now(timezone.utc).isoformat()
    target = ROOT / "runs" / now.strftime("witness_seed_audit_%Y%m%dT%H%M%S%fZ")
    target.mkdir()
    with (target / "RESULTS.json").open("x", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(target)
    for name, row in output["protocols"].items():
        values = row["actual"]
        print(name, values["address_total"], values["distinct"], values["collision_groups"])
    print("cross protocol repeats:", sum(e["n"] for e in output["joint_actual"]["collision_groups"]
                                         if len(e["groups"]) > 1))
    return output


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("runs", nargs="+", type=lambda p: ROOT / p)
    audit(cli.parse_args().runs)
