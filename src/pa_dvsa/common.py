"""Shared filesystem and serialization helpers."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def project_root_from(path: str | Path | None = None) -> Path:
    """Return the project root as an absolute path."""

    return Path(path or ".").expanduser().resolve()


def sha256_file(path: str | Path) -> str:
    """Return the SHA-256 digest for a file."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    """Write stable, human-readable JSON."""

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_text(path: str | Path) -> str:
    """Read UTF-8 text."""

    return Path(path).read_text(encoding="utf-8")
