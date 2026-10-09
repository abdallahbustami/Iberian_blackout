"""Readers and aggregations of the stored result files used by the figures."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
from pathlib import Path
from typing import Any

import pandas as pd

SYSTEMS = ("kundur", "ieee39", "npcc", "gbnetwork")
FAMILIES = (
    "load_disconnection",
    "fixed_pf_ramp",
    "plant_trip",
    "generator_trip",
    "generation_redispatch",
    "shunt_reactor_disconnection",
    "capacitor_switching",
    "line_energization",
)

REPO_ROOT = Path(__file__).resolve().parents[3]
RESULTS = REPO_ROOT / "results" / "revised"

CASCADE_DEFINITIONS = ("unrefined", "refined")
DEFAULT_CASCADE = os.environ.get("PA_DVSA_CASCADE", "unrefined")

PAIR_OUTCOMES = (
    "TP", "FP", "pickup_only_trip", "pickup_only_clear", "caught_risky", "risky_clear",
    "unsafe_FN", "TN", "data_limited_trip", "data_limited_clear", "tds_inconclusive", "late_trip",
)


def _bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.lower().map({"true": True, "false": False, "1": True, "0": False})


def read_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists() or path.stat().st_size <= 2:
        return None
    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return None
    return df if len(df.columns) else None


def read_json(path: Path) -> Any:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


@dataclass
class SystemResults:
    slug: str
    root: Path
    _cache: dict[str, Any] = field(default_factory=dict)
    sources: dict[str, str] = field(default_factory=dict)

    def csv(self, name: str) -> pd.DataFrame | None:
        if name not in self._cache:
            self._cache[name] = read_csv(self.root / name)
        return self._cache[name]

    def json(self, name: str) -> Any:
        key = "json:" + name
        if key not in self._cache:
            self._cache[key] = read_json(self.root / name)
        return self._cache[key]

    def table(self, name: str) -> pd.DataFrame | None:
        if name.endswith(".csv"):
            return self.csv(name)
        obj = self.json(name)
        if isinstance(obj, list) and obj and isinstance(obj[0], dict):
            df = pd.DataFrame(obj)
            for c in df.columns:
                if df[c].map(lambda v: isinstance(v, list)).any():
                    df[c] = df[c].map(lambda v: ";".join(map(str, v)) if isinstance(v, list) else v)
            return df
        return None

    @property
    def exists(self) -> bool:
        return (self.root / "screen_events.csv").exists()

    def default_cascade(self) -> str:
        meta = self.json("screen_meta.json") or {}
        for key in ("cascade_definition", "primary_cascade", "cascade_primary"):
            val = meta.get(key) or (meta.get("settings") or {}).get(key)
            if isinstance(val, str) and val in CASCADE_DEFINITIONS:
                return val
        val = self.csv("tds_validation.csv")
        if val is not None and "regime" in val.columns and (val["regime"] == "nominal_refined").any():
            return "unrefined"
        return "refined"

    def cascade_table(self, name: str, definition: str) -> pd.DataFrame | None:
        if definition not in CASCADE_DEFINITIONS:
            raise ValueError(f"cascade definition must be one of {CASCADE_DEFINITIONS}")
        stem, ext = name.rsplit(".", 1)
        cands: list[tuple[str, list[str]]] = [(name, [f"nominal_{definition}"])]
        if self.default_cascade() == definition:
            cands.append((name, ["nominal"]))
        for e in [ext] + (["json"] if ext == "csv" else []):
            cands.append((f"{stem}_{definition}.{e}", [f"nominal_{definition}", "nominal"]))
        for fname, regimes in cands:
            df = self.table(fname)
            if df is None or "regime" not in df.columns:
                continue
            for reg in regimes:
                sub = df[df["regime"] == reg]
                if len(sub):
                    self.sources[f"{name}:{definition}"] = f"{self.root.name}/{fname} [regime={reg}]"
                    return sub.copy()
        return None

    def ablations(self) -> pd.DataFrame | None:
        for name in ("ablations/ablations.csv", "ablations.csv"):
            df = self.csv(name)
            if df is not None and len(df):
                self.sources["ablations"] = name
                return df
        return None

    def fig_data(self, name: str, definition: str) -> pd.DataFrame | None:
        stem, ext = name.rsplit(".", 1)
        alt = self.csv(f"fig_data/{stem}_{definition}.{ext}")
        if alt is not None:
            self.sources[f"fig_data/{name}"] = f"fig_data/{stem}_{definition}.{ext}"
            return alt
        df = self.csv(f"fig_data/{name}")
        if df is not None:
            self.sources[f"fig_data/{name}"] = f"fig_data/{name}"
        return df


class Results:
    def __init__(self, root: Path | None = None, *, subdir: str = "", cascade: str | None = None) -> None:
        self.root = Path(root) if root is not None else RESULTS
        self.subdir = subdir
        self.cascade = cascade or DEFAULT_CASCADE
        if self.cascade not in CASCADE_DEFINITIONS:
            raise ValueError(f"cascade must be one of {CASCADE_DEFINITIONS}")
        self.systems = {s: SystemResults(s, self.root / s / subdir if subdir else self.root / s)
                        for s in SYSTEMS}
        self.missing: list[str] = []

    @property
    def other_cascade(self) -> str:
        return [c for c in CASCADE_DEFINITIONS if c != self.cascade][0]

    def present(self) -> list[str]:
        return [s for s, r in self.systems.items() if r.exists]

    def sys(self, slug: str) -> SystemResults:
        return self.systems[slug]

    @property
    def replica(self) -> Path:
        return self.root / "activsg2000"

    def sources(self) -> dict[str, dict[str, str]]:
        return {s: dict(r.sources) for s, r in self.systems.items() if r.sources}


def meta(res: Results, slug: str) -> dict[str, Any]:
    return res.sys(slug).json("screen_meta.json") or {}


def designated_seed(res: Results, slug: str) -> str | None:
    m = meta(res, slug)
    return m.get("designated_seed")
