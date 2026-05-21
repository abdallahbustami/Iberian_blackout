"""Artifact loaders for the paper case-study figure pipeline."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd


CASE_SPECS: tuple[tuple[str, str, str], ...] = (
    ("kundur", "kundur", "Kundur 2-area"),
    ("ieee39", "ieee39_load_shedding", "IEEE-39"),
    ("npcc", "npcc_full_load_shedding", "NPCC"),
    ("gbnetwork", "gbnetwork_load_shedding", "GBnetwork"),
)


class MissingArtifactError(RuntimeError):
    """Raised when strict generation cannot proceed from real artifacts."""


@dataclass
class CaseArtifacts:
    slug: str
    directory_name: str
    display_name: str
    case_dir: Path
    paper_dir: Path
    top_json_path: Path | None = None
    top_json: dict[str, Any] = field(default_factory=dict)
    case_config: dict[str, Any] = field(default_factory=dict)
    prediction_study: dict[str, Any] = field(default_factory=dict)
    mitigation_json: dict[str, Any] = field(default_factory=dict)
    robust_screen: dict[str, Any] = field(default_factory=dict)
    scalability_metrics: list[dict[str, Any]] = field(default_factory=list)
    k_pickup: pd.DataFrame | None = None
    k_trip: pd.DataFrame | None = None
    prediction_overlay: pd.DataFrame | None = None
    case_calibration: pd.DataFrame | None = None
    screen_vs_tds: pd.DataFrame | None = None
    uncertainty: pd.DataFrame | None = None
    runtime: pd.DataFrame | None = None
    mitigation_table: pd.DataFrame | None = None
    ablations: pd.DataFrame | None = None
    tds_trace: pd.DataFrame | None = None

    @property
    def summary(self) -> dict[str, Any]:
        return dict(self.top_json.get("summary") or {})

    @property
    def metrics(self) -> dict[str, Any]:
        metrics = self.top_json.get("metrics") or self.scalability_metrics or []
        if isinstance(metrics, list) and metrics:
            return dict(metrics[0])
        return {}

    @property
    def protected_asset_ids(self) -> list[str]:
        if self.prediction_overlay is not None and "asset_id" in self.prediction_overlay:
            return [str(item) for item in self.prediction_overlay["asset_id"].tolist()]
        collectors = self.case_config.get("collectors") or []
        return [str(item.get("asset_id")) for item in collectors if item.get("asset_id")]

    @property
    def event_ids(self) -> list[str]:
        if self.k_pickup is None:
            return []
        return [str(col) for col in self.k_pickup.columns if col != "protected_asset_id"]


@dataclass
class ReplicationArtifacts:
    root: Path
    events: list[dict[str, Any]] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)
    config: dict[str, Any] = field(default_factory=dict)
    collector_traces: pd.DataFrame | None = None
    system_traces: pd.DataFrame | None = None


@dataclass
class ArtifactBundle:
    cases: dict[str, CaseArtifacts]
    replication: ReplicationArtifacts | None
    input_hashes: dict[str, str] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


def relpath(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(path.resolve())


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _record(bundle: ArtifactBundle, path: Path) -> None:
    if path.exists():
        bundle.input_hashes[relpath(path)] = sha256_file(path)


def _read_json(bundle: ArtifactBundle, path: Path, *, required: bool) -> dict[str, Any]:
    if not path.exists():
        if required:
            raise MissingArtifactError(f"Missing required JSON artifact: {relpath(path)}")
        bundle.warnings.append(f"missing optional JSON artifact: {relpath(path)}")
        return {}
    _record(bundle, path)
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise MissingArtifactError(f"Expected object JSON artifact: {relpath(path)}")
    return data


def _read_json_list(bundle: ArtifactBundle, path: Path, *, required: bool) -> list[dict[str, Any]]:
    if not path.exists():
        if required:
            raise MissingArtifactError(f"Missing required JSON artifact: {relpath(path)}")
        bundle.warnings.append(f"missing optional JSON artifact: {relpath(path)}")
        return []
    _record(bundle, path)
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if isinstance(data, list):
        return [dict(item) for item in data if isinstance(item, dict)]
    raise MissingArtifactError(f"Expected list JSON artifact: {relpath(path)}")


def _read_csv(bundle: ArtifactBundle, path: Path, *, required: bool) -> pd.DataFrame | None:
    if not path.exists():
        if required:
            raise MissingArtifactError(f"Missing required CSV artifact: {relpath(path)}")
        bundle.warnings.append(f"missing optional CSV artifact: {relpath(path)}")
        return None
    _record(bundle, path)
    return pd.read_csv(path)


def _find_top_json(case_dir: Path) -> Path | None:
    candidates = [
        path
        for path in sorted(case_dir.glob("*.json"))
        if "manifest" not in path.name and path.name != "case_config.json"
    ]
    return candidates[0] if candidates else None


def _load_case(
    bundle: ArtifactBundle,
    data_root: Path,
    slug: str,
    directory_name: str,
    display_name: str,
    *,
    strict: bool,
) -> CaseArtifacts:
    case_dir = data_root / directory_name
    paper_dir = case_dir / "paper_outputs"
    if not case_dir.exists():
        raise MissingArtifactError(f"Missing case directory: {relpath(case_dir)}")
    top_json_path = _find_top_json(case_dir)
    top_json = _read_json(bundle, top_json_path, required=strict) if top_json_path else {}
    ca = CaseArtifacts(
        slug=slug,
        directory_name=directory_name,
        display_name=display_name,
        case_dir=case_dir,
        paper_dir=paper_dir,
        top_json_path=top_json_path,
        top_json=top_json,
    )
    ca.case_config = _read_json(bundle, paper_dir / "case_config.json", required=strict)
    ca.prediction_study = _read_json(bundle, paper_dir / "prediction_study.json", required=strict)
    ca.mitigation_json = _read_json(bundle, paper_dir / "mitigation.json", required=strict)
    ca.robust_screen = _read_json(bundle, paper_dir / "robust_screen.json", required=False)
    ca.scalability_metrics = _read_json_list(
        bundle, paper_dir / "scalability_metrics.json", required=strict
    )
    ca.k_pickup = _read_csv(bundle, paper_dir / "k_pickup.csv", required=strict)
    ca.k_trip = _read_csv(bundle, paper_dir / "k_trip.csv", required=False)
    ca.prediction_overlay = _read_csv(
        bundle, paper_dir / "prediction_overlay.csv", required=strict
    )
    ca.case_calibration = _read_csv(
        bundle, paper_dir / "table_case_calibration.csv", required=strict
    )
    ca.screen_vs_tds = _read_csv(
        bundle, paper_dir / "table_screen_vs_tds_validation.csv", required=strict
    )
    ca.uncertainty = _read_csv(bundle, paper_dir / "table_uncertainty.csv", required=strict)
    ca.runtime = _read_csv(bundle, paper_dir / "table_large_system_runtime.csv", required=False)
    ca.mitigation_table = _read_csv(
        bundle, paper_dir / "table_mitigation.csv", required=strict
    )
    ca.ablations = _read_csv(bundle, paper_dir / "table_ablations.csv", required=False)
    ca.tds_trace = _read_csv(bundle, case_dir / "tds_trace.csv", required=False)
    return ca


def _load_replication(
    bundle: ArtifactBundle,
    replication_root: Path,
    *,
    strict: bool,
) -> ReplicationArtifacts | None:
    if not replication_root.exists():
        if strict:
            raise MissingArtifactError(
                f"Missing replication directory: {relpath(replication_root)}"
            )
        bundle.warnings.append(f"missing replication directory: {relpath(replication_root)}")
        return None
    rep = ReplicationArtifacts(root=replication_root)
    rep.events = _read_json_list(bundle, replication_root / "events.json", required=strict)
    rep.summary = _read_json(bundle, replication_root / "summary.json", required=strict)
    rep.config = _read_json(bundle, replication_root / "case_config.json", required=strict)
    rep.collector_traces = _read_csv(
        bundle, replication_root / "collector_traces.csv", required=strict
    )
    rep.system_traces = _read_csv(
        bundle, replication_root / "system_traces.csv", required=False
    )
    return rep


def load_artifacts(
    *,
    data_root: Path,
    replication_root: Path,
    strict: bool,
) -> ArtifactBundle:
    bundle = ArtifactBundle(cases={}, replication=None)
    for slug, directory_name, display_name in CASE_SPECS:
        bundle.cases[slug] = _load_case(
            bundle,
            data_root,
            slug,
            directory_name,
            display_name,
            strict=strict,
        )
    bundle.replication = _load_replication(bundle, replication_root, strict=strict)
    return bundle


def load_replication_bundle(
    *,
    replication_root: Path,
    strict: bool,
) -> ArtifactBundle:
    """Load only the replication artifacts needed by paper figure 1.

    This path is used when the normalized case-study dataset has already been
    regenerated directly by ``scripts/run_paper_case_sweep.py``. In that mode,
    figures 2--8 read from CSV files under ``results/paper_case_studies`` and
    do not need the older per-case artifact directories.
    """

    bundle = ArtifactBundle(cases={}, replication=None)
    bundle.replication = _load_replication(bundle, replication_root, strict=strict)
    return bundle
