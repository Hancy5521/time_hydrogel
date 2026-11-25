"""Verify the unpacked review package without third-party dependencies."""
from hashlib import file_digest
from pathlib import Path, PurePosixPath
import re


def digest(path):
    with path.open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def verify(root):
    files = set()
    for line in (root/"MANIFEST.sha256").read_text(encoding="utf-8").splitlines():
        match = re.fullmatch(r"([0-9a-f]{64})  ([^\\]+)", line)
        if not match:
            raise ValueError(f"Invalid manifest line: {line!r}")
        expected, name = match.groups()
        relative = PurePosixPath(name)
        if relative.is_absolute() or ".." in relative.parts or ":" in name or relative.as_posix() != name:
            raise ValueError(f"Invalid member path: {name}")
        if name.casefold() in files:
            raise ValueError(f"Duplicate member path: {name}")
        path = root.joinpath(*relative.parts)
        if not path.is_file() or path.is_symlink() or not path.resolve().is_relative_to(root):
            raise ValueError(f"Missing or non-contained file: {name}")
        if digest(path) != expected:
            raise ValueError(f"SHA256 discrepancy: {name}")
        files.add(name.casefold())
    if not files:
        raise ValueError("Empty manifest")
    print(f"OK: {len(files)} payload hashes match.")
    return len(files)


if __name__ == "__main__":
    verify(Path(__file__).resolve().parent)
