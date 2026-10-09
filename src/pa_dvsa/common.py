"""Filesystem and serialization helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def project_root_from(path: str | Path | None = None) -> Path:
    return Path(path or ".").expanduser().resolve()


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_text(path: str | Path) -> str:
    return Path(path).read_text(encoding="utf-8")
