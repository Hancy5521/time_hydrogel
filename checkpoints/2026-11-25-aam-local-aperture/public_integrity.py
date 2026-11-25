"""Verify portable file bytes and resolve historical checksum bindings."""

from hashlib import file_digest
import json
from pathlib import Path


BUNDLE_ROOT = Path(__file__).resolve().parent
with (BUNDLE_ROOT / "PUBLIC_TRANSFORMS.json").open(encoding="utf-8") as stream:
    _record = json.load(stream)
FROZEN_RUN_ID = _record["frozen_run_id"]
TRANSFORMS = {row["path"]: row for row in _record["transforms"]}


def original_digest(file):
    """Look up the receipt hash only after checking any redacted file."""
    path = BUNDLE_ROOT / Path(file)
    key = path.resolve().relative_to(BUNDLE_ROOT).as_posix()
    with path.open("rb") as stream:
        sha = file_digest(stream, "sha256").hexdigest()
    entry = TRANSFORMS.get(key)
    if entry is None:
        return sha
    if sha != entry["published_sha256"]:
        raise ValueError(f"Published SHA256 mismatch for {key}")
    return entry["original_sha256"]
