"""Check the names and SHA-256s of the frozen public checkpoint payload."""

from hashlib import file_digest
from pathlib import Path, PurePosixPath
import re


def verify(root):
    names = set()
    seen = set()
    for line in (root / "MANIFEST.sha256").read_text(encoding="utf-8").splitlines():
        match = re.fullmatch(r"([0-9a-f]{64})  ([^\\]+)", line)
        if not match:
            raise ValueError(f"Invalid manifest line: {line!r}")
        expected, name = match.groups()
        relative = PurePosixPath(name)
        if relative.is_absolute() or ".." in relative.parts or ":" in name or relative.as_posix() != name:
            raise ValueError(f"Invalid member path: {name}")
        if name.casefold() in seen:
            raise ValueError(f"Duplicate member path: {name}")
        path = root.joinpath(*relative.parts)
        if not path.is_file() or path.is_symlink() or not path.resolve().is_relative_to(root):
            raise ValueError(f"Missing or non-contained member: {name}")
        with path.open("rb") as stream:
            actual = file_digest(stream, "sha256").hexdigest()
        if actual != expected:
            raise ValueError(f"SHA256 mismatch: {name}")
        names.add(name)
        seen.add(name.casefold())
    if not names:
        raise ValueError("Empty manifest")
    print(f"OK: {len(names)} payload hashes match.")
    return names


if __name__ == "__main__":
    verify(Path(__file__).resolve().parent)
