"""CSV and JSON writers."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


def _cell(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, (bool, np.bool_)):
        return "true" if bool(value) else "false"
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        v = float(value)
        if math.isnan(v):
            return "nan"
        if math.isinf(v):
            return "inf" if v > 0 else "-inf"
        return repr(v) if abs(v) >= 1e-4 or v == 0.0 else f"{v:.6e}"
    if isinstance(value, (list, tuple, set, frozenset)):
        return ";".join(str(_cell(v)) for v in value)
    return str(value)


def write_csv(path: Path, rows: Iterable[Mapping[str, Any]], columns: Sequence[str] | None = None) -> Path:
    rows = list(rows)
    if columns is None:
        cols: list[str] = []
        for row in rows:
            for key in row:
                if key not in cols:
                    cols.append(key)
        columns = cols
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(list(columns))
        for row in rows:
            writer.writerow([_cell(row.get(c)) for c in columns])
    return path


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        v = float(value)
        return v if math.isfinite(v) else None
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (set, frozenset, tuple)):
        return list(value)
    if isinstance(value, Path):
        return str(value)
    return str(value)


def _clean(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Mapping):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    return value


def write_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_clean(payload), indent=1, default=_json_default))
    return path


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text())
