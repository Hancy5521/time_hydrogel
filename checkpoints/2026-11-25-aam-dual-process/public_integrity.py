"""Bridge portable file hashes to the original historical freeze receipts."""
from hashlib import file_digest
import json
from pathlib import Path


BUNDLE_ROOT = Path(__file__).resolve().parent
with (BUNDLE_ROOT / "PUBLIC_TRANSFORMS.json").open(encoding="utf-8") as stream:
    TRANSFORMS = {entry["path"]: entry for entry in json.load(stream)["transforms"]}


def original_digest(file):
    path = Path(file).resolve()
    with path.open("rb") as stream:
        digest = file_digest(stream, "sha256").hexdigest()
    key = path.relative_to(BUNDLE_ROOT).as_posix()
    entry = TRANSFORMS.get(key)
    if entry is None:
        return digest
    if digest != entry["public_sha256"]:
        raise ValueError(f"Public SHA256 mismatch for {key}")
    return entry["original_sha256"]
