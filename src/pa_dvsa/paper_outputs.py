"""Reproducible paper-output generation.

This module writes stable CSV/JSON artifacts, LaTeX-ready generated tables, and
figures under ``LaTeX/figures/generated``.  It deliberately does not edit
``LaTeX/root.tex``; generated outputs become paper inputs only after the
computational results are stable.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, is_dataclass, asdict
from enum import Enum
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .benchmark_studies import (
    BenchmarkStudyConfig,
    CausalAblation,
    PredictionStudyResult,
    ScalabilityMetrics,
)
from .common import project_root_from, sha256_file
from .data_model import AssessmentStatus, MitigationResult
from .mitigation import MitigationLPDetails
from .robust_screening import RobustScreenResult


class PaperOutputError(RuntimeError):
    """Raised when paper output generation fails."""


@dataclass(frozen=True, slots=True)
class PaperOutputPaths:
    """Filesystem layout for generated paper artifacts."""

    project_root: Path
    artifact_dir: Path
    figure_dir: Path
    table_dir: Path

    @classmethod
    def from_project_root(
        cls,
        project_root: str | Path = ".",
        *,
        artifact_dir: str | Path = "results/paper_outputs",
        figure_dir: str | Path = "LaTeX/figures/generated",
        table_dir: str | Path = "LaTeX/tables/generated",
    ) -> "PaperOutputPaths":
        root = project_root_from(project_root)
        return cls(
            project_root=root,
            artifact_dir=_resolve_under_root(root, artifact_dir),
            figure_dir=_resolve_under_root(root, figure_dir),
            table_dir=_resolve_under_root(root, table_dir),
        )

    def ensure(self) -> None:
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        self.figure_dir.mkdir(parents=True, exist_ok=True)
        self.table_dir.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True, slots=True)
class PaperArtifact:
    """One generated output with a reproducibility digest."""

    artifact_id: str
    kind: str
    path: str
    sha256: str
    row_count: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_id": self.artifact_id,
            "kind": self.kind,
            "path": self.path,
            "sha256": self.sha256,
            "row_count": self.row_count,
        }


@dataclass(frozen=True, slots=True)
class PaperOutputManifest:
    """Manifest for generated paper artifacts."""

    output_id: str
    artifacts: tuple[PaperArtifact, ...]
    root_tex_updated: bool = False
    notes: str = "Generated artifacts are reproducible inputs; root.tex is not edited automatically."

    def to_dict(self) -> dict[str, Any]:
        return {
            "output_id": self.output_id,
            "root_tex_updated": self.root_tex_updated,
            "notes": self.notes,
            "artifacts": [item.to_dict() for item in self.artifacts],
        }


@dataclass(frozen=True, slots=True)
class TableSpec:
    """LaTeX table source and its matching CSV rows."""

    table_id: str
    caption: str
    label: str
    columns: tuple[tuple[str, str], ...]
    rows: tuple[Mapping[str, Any], ...]


def generate_paper_outputs(
    *,
    output_id: str = "paper_outputs",
    config: BenchmarkStudyConfig | None = None,
    prediction: PredictionStudyResult | None = None,
    ablations: Sequence[CausalAblation | Mapping[str, Any]] = (),
    robust: RobustScreenResult | None = None,
    mitigation: MitigationLPDetails | MitigationResult | None = None,
    scalability: Sequence[ScalabilityMetrics] = (),
    paths: PaperOutputPaths | None = None,
    generate_figures: bool = True,
) -> PaperOutputManifest:
    """Generate stable CSV/JSON artifacts, tables, figures, and a manifest."""

    out_paths = paths or PaperOutputPaths.from_project_root(".")
    out_paths.ensure()
    artifacts: list[PaperArtifact] = []

    payloads: list[tuple[str, Any]] = [
        ("case_config", config),
        ("prediction_study", prediction),
        ("robust_screen", robust),
        ("mitigation", mitigation),
    ]
    for name, payload in payloads:
        if payload is not None:
            artifacts.append(write_json_artifact(out_paths, name, payload))

    if ablations:
        artifacts.append(write_json_artifact(out_paths, "ablations", tuple(ablations)))
    if scalability:
        artifacts.append(write_json_artifact(out_paths, "scalability_metrics", tuple(scalability)))

    if prediction is not None:
        artifacts.extend(write_prediction_artifacts(out_paths, prediction))

    table_specs = build_table_specs(
        config=config,
        prediction=prediction,
        ablations=ablations,
        robust=robust,
        mitigation=mitigation,
        scalability=scalability,
    )
    for spec in table_specs:
        artifacts.append(write_csv_artifact(out_paths, f"table_{spec.table_id}", spec.rows))
        artifacts.append(write_latex_table(out_paths, spec))

    if generate_figures:
        if prediction is not None:
            artifacts.append(plot_k_heatmap(out_paths, prediction, matrix="pickup"))
            artifacts.append(plot_prediction_overlay(out_paths, prediction))
        if scalability:
            artifacts.append(plot_runtime_bars(out_paths, scalability))

    manifest = PaperOutputManifest(
        output_id=output_id,
        artifacts=tuple(sorted(artifacts, key=lambda item: item.artifact_id)),
    )
    manifest_path = out_paths.artifact_dir / f"{output_id}_manifest.json"
    manifest_path.write_text(_stable_json(manifest.to_dict()), encoding="utf-8")
    return PaperOutputManifest(
        output_id=output_id,
        artifacts=tuple(sorted(
            (*manifest.artifacts, _artifact(manifest_path, output_id + "_manifest", "json")),
            key=lambda item: item.artifact_id,
        )),
    )


def write_prediction_artifacts(
    paths: PaperOutputPaths,
    prediction: PredictionStudyResult,
) -> tuple[PaperArtifact, ...]:
    """Write prediction overlay and K-matrix CSVs."""

    overlay_rows = [item.to_dict() for item in prediction.overlay]
    k_pk_rows = _matrix_rows(prediction.k_pk, prediction.protected_asset_ids, prediction.event_ids)
    k_tr_rows = _matrix_rows(prediction.k_tr, prediction.protected_asset_ids, prediction.event_ids)
    return (
        write_csv_artifact(paths, "prediction_overlay", overlay_rows),
        write_csv_artifact(paths, "k_pickup", k_pk_rows),
        write_csv_artifact(paths, "k_trip", k_tr_rows),
    )


def build_table_specs(
    *,
    config: BenchmarkStudyConfig | None = None,
    prediction: PredictionStudyResult | None = None,
    ablations: Sequence[CausalAblation | Mapping[str, Any]] = (),
    robust: RobustScreenResult | None = None,
    mitigation: MitigationLPDetails | MitigationResult | None = None,
    scalability: Sequence[ScalabilityMetrics] = (),
) -> tuple[TableSpec, ...]:
    """Build all generated table specs, using explicit missing-result rows if needed."""

    return (
        _case_calibration_table(config),
        _validation_table(prediction),
        _ablation_table(ablations),
        _uncertainty_table(robust),
        _mitigation_table(mitigation),
        _runtime_table(scalability),
    )


def write_json_artifact(
    paths: PaperOutputPaths,
    artifact_id: str,
    payload: Any,
) -> PaperArtifact:
    path = paths.artifact_dir / f"{artifact_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_stable_json(_to_serializable(payload)), encoding="utf-8")
    return _artifact(path, artifact_id, "json")


def write_csv_artifact(
    paths: PaperOutputPaths,
    artifact_id: str,
    rows: Sequence[Mapping[str, Any]],
) -> PaperArtifact:
    path = paths.artifact_dir / f"{artifact_id}.csv"
    normalized = [_normalize_row(row) for row in rows]
    fieldnames = _fieldnames(normalized)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in normalized:
            writer.writerow({key: row.get(key, "") for key in fieldnames})
    return _artifact(path, artifact_id, "csv", row_count=len(normalized))


def write_latex_table(paths: PaperOutputPaths, spec: TableSpec) -> PaperArtifact:
    path = paths.table_dir / f"{spec.table_id}.tex"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_latex_table_source(spec), encoding="utf-8")
    return _artifact(path, f"table_{spec.table_id}_tex", "latex_table", row_count=len(spec.rows))


def plot_k_heatmap(
    paths: PaperOutputPaths,
    prediction: PredictionStudyResult,
    *,
    matrix: str = "pickup",
) -> PaperArtifact:
    """Write a K-matrix heatmap into ``LaTeX/figures/generated``."""

    plt = _pyplot()
    data = np.asarray(prediction.k_pk if matrix == "pickup" else prediction.k_tr, dtype=float)
    fig, ax = plt.subplots(figsize=(5.8, 3.8))
    im = ax.imshow(data, cmap="magma", aspect="auto", interpolation="nearest")
    ax.set_xticks(range(len(prediction.event_ids)))
    ax.set_xticklabels(prediction.event_ids, rotation=45, ha="right")
    ax.set_yticks(range(len(prediction.protected_asset_ids)))
    ax.set_yticklabels(prediction.protected_asset_ids)
    ax.set_xlabel("Event channel")
    ax.set_ylabel("Protected asset")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Normalized erosion")
    fig.tight_layout()
    path = paths.figure_dir / f"{prediction.study_id}_k_{matrix}.pdf"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return _artifact(path, f"figure_{prediction.study_id}_k_{matrix}", "figure")


def plot_prediction_overlay(
    paths: PaperOutputPaths,
    prediction: PredictionStudyResult,
) -> PaperArtifact:
    """Write predicted layer vs nonlinear trip-time overlay figure."""

    plt = _pyplot()
    assets = [item.asset_id for item in prediction.overlay]
    y = np.arange(len(assets))
    layers = [
        np.nan if item.predicted_layer is None else float(item.predicted_layer)
        for item in prediction.overlay
    ]
    trip_times = [
        np.nan if item.simulated_trip_time_s is None else float(item.simulated_trip_time_s)
        for item in prediction.overlay
    ]
    fig, ax1 = plt.subplots(figsize=(6.2, max(2.6, 0.35 * len(assets) + 1.2)))
    ax1.scatter(layers, y, marker="s", label="Predicted layer", color="#1f77b4")
    ax1.set_xlabel("Predicted cascade layer")
    ax1.set_yticks(y)
    ax1.set_yticklabels(assets)
    ax1.invert_yaxis()
    ax2 = ax1.twiny()
    ax2.scatter(trip_times, y, marker="o", label="Nonlinear trip time", color="#d62728")
    ax2.set_xlabel("Nonlinear trip time (s)")
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc="best")
    fig.tight_layout()
    path = paths.figure_dir / f"{prediction.study_id}_prediction_overlay.pdf"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return _artifact(path, f"figure_{prediction.study_id}_prediction_overlay", "figure")


def plot_runtime_bars(
    paths: PaperOutputPaths,
    metrics: Sequence[ScalabilityMetrics],
) -> PaperArtifact:
    """Write large-system runtime bar chart."""

    plt = _pyplot()
    rows = tuple(metrics)
    labels = [item.case_id for item in rows]
    runtimes = [item.runtime_s for item in rows]
    fig, ax = plt.subplots(figsize=(6.0, 3.2))
    ax.bar(labels, runtimes, color="#4c78a8")
    ax.set_ylabel("Runtime (s)")
    ax.set_xlabel("Benchmark")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    path = paths.figure_dir / "large_system_runtime.pdf"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return _artifact(path, "figure_large_system_runtime", "figure")


def _case_calibration_table(config: BenchmarkStudyConfig | None) -> TableSpec:
    if config is None or not config.collectors:
        rows = (_missing_result_row("case calibration not generated"),)
        columns = (("item", "Item"), ("status", "Status"))
    else:
        rows = tuple(
            {
                "asset": item.asset_id,
                "tx_bus": item.transmission_bus,
                "collector_bus": item.collector_bus,
                "tap": _fmt(item.tap_ratio, 3),
                "v_trip_pu": _fmt(item.threshold_pu, 3),
                "dwell_s": _fmt(item.dwell_time_s, 3),
                "q_abs_mvar": _fmt(item.q_absorption_mvar, 1),
            }
            for item in config.collectors
        )
        columns = (
            ("asset", "Asset"),
            ("tx_bus", "Tx bus"),
            ("collector_bus", "Collector"),
            ("tap", "Tap"),
            ("v_trip_pu", "V trip pu"),
            ("dwell_s", "Dwell (s)"),
            ("q_abs_mvar", "Q abs MVAr"),
        )
    return TableSpec(
        "case_calibration",
        "Case calibration for protected collector-side assets.",
        "tab:generated_case_calibration",
        columns,
        rows,
    )


def _validation_table(prediction: PredictionStudyResult | None) -> TableSpec:
    if prediction is None:
        rows = (_missing_result_row("screen-vs-TDS validation not generated"),)
        columns = (("item", "Item"), ("status", "Status"))
    else:
        rows = tuple(
            {
                "asset": item.asset_id,
                "layer": _none_dash(item.predicted_layer),
                "trip_s": _none_dash(item.simulated_trip_time_s),
                "max_v": _none_dash(item.max_voltage_pu),
                "fp": "yes" if item.false_positive else "no",
                "fn": "yes" if item.false_negative else "no",
                "data_limited": "yes" if item.data_limited else "no",
            }
            for item in prediction.overlay
        )
        columns = (
            ("asset", "Asset"),
            ("layer", "Layer"),
            ("trip_s", "Trip time (s)"),
            ("max_v", "Max z pu"),
            ("fp", "FP"),
            ("fn", "FN"),
            ("data_limited", "Data limited"),
        )
    return TableSpec(
        "screen_vs_tds_validation",
        "Screen prediction compared with nonlinear TDS validation.",
        "tab:generated_screen_vs_tds_validation",
        columns,
        rows,
    )


def _ablation_table(ablations: Sequence[CausalAblation | Mapping[str, Any]]) -> TableSpec:
    if not ablations:
        rows = (_missing_result_row("ablation results not generated"),)
        columns = (("item", "Item"), ("status", "Status"))
    else:
        rows = tuple(_ablation_row(item) for item in ablations)
        columns = (
            ("ablation_id", "Ablation"),
            ("description", "Description"),
            ("feature_overrides", "Feature overrides"),
            ("parameter_overrides", "Parameter overrides"),
            ("status", "Status"),
        )
    return TableSpec(
        "ablations",
        "Causal ablation study definitions and result status.",
        "tab:generated_ablations",
        columns,
        rows,
    )


def _uncertainty_table(robust: RobustScreenResult | None) -> TableSpec:
    if robust is None:
        rows = (_missing_result_row("uncertainty screen not generated"),)
        columns = (("item", "Item"), ("status", "Status"))
    else:
        rows = tuple(
            {
                "asset": asset,
                "r_bar": _none_dash(float(robust.r_bar[index])),
                "margin_lower": _none_dash(float(robust.margin_lower_pu[index])),
                "violating": "yes" if asset in robust.violating_assets else "no",
                "data_limited": "yes" if asset in robust.data_limited_assets else "no",
            }
            for index, asset in enumerate(robust.protected_asset_ids)
        )
        columns = (
            ("asset", "Asset"),
            ("r_bar", "r bar"),
            ("margin_lower", "Lower margin"),
            ("violating", "Violating"),
            ("data_limited", "Data limited"),
        )
    return TableSpec(
        "uncertainty",
        "Robust uncertainty screen summary.",
        "tab:generated_uncertainty",
        columns,
        rows,
    )


def _mitigation_table(mitigation: MitigationLPDetails | MitigationResult | None) -> TableSpec:
    if mitigation is None:
        rows = (_missing_result_row("mitigation results not generated"),)
        columns = (("item", "Item"), ("status", "Status"))
    else:
        result = mitigation.mitigation_result if isinstance(mitigation, MitigationLPDetails) else mitigation
        rows = tuple(
            {
                "control": item.control_id,
                "alpha": _fmt(item.alpha, 4),
                "mvar": _none_dash(item.magnitude_mvar),
                "saturated": "yes" if item.saturated else "no",
                "cost": _none_dash(item.cost_contribution),
            }
            for item in result.selections
        )
        if not rows:
            rows = ({"control": "none", "alpha": "0", "mvar": "-", "saturated": "no", "cost": "-"},)
        columns = (
            ("control", "Control"),
            ("alpha", "alpha"),
            ("mvar", "MVAr"),
            ("saturated", "Saturated"),
            ("cost", "Cost"),
        )
    return TableSpec(
        "mitigation",
        "Mitigation LP/QP selected controls.",
        "tab:generated_mitigation",
        columns,
        rows,
    )


def _runtime_table(metrics: Sequence[ScalabilityMetrics]) -> TableSpec:
    if not metrics:
        rows = (_missing_result_row("large-system runtime results not generated"),)
        columns = (("item", "Item"), ("status", "Status"))
    else:
        rows = tuple(
            {
                "case": item.case_id,
                "runtime_s": _fmt(item.runtime_s, 3),
                "sparse_solves": item.sparse_solve_count,
                "channels": item.selected_channels,
                "memory_mb": _none_dash(item.memory_peak_mb),
                "status": item.status.value,
            }
            for item in metrics
        )
        columns = (
            ("case", "Case"),
            ("runtime_s", "Runtime (s)"),
            ("sparse_solves", "Sparse solves"),
            ("channels", "Channels"),
            ("memory_mb", "Peak MB"),
            ("status", "Status"),
        )
    return TableSpec(
        "large_system_runtime",
        "Large-system screening runtime and scalability metrics.",
        "tab:generated_large_system_runtime",
        columns,
        rows,
    )


def _latex_table_source(spec: TableSpec) -> str:
    align = "l" * len(spec.columns)
    headers = " & ".join(_latex_escape(label) for _, label in spec.columns)
    lines = [
        "% Auto-generated by pa_dvsa.paper_outputs. Do not edit manually.",
        "\\begin{table}[t]",
        "\\centering",
        f"\\caption{{{_latex_escape(spec.caption)}}}",
        f"\\label{{{spec.label}}}",
        f"\\begin{{tabular}}{{{align}}}",
        "\\toprule",
        headers + r" \\",
        "\\midrule",
    ]
    for row in spec.rows:
        lines.append(" & ".join(_latex_escape(row.get(key, "")) for key, _ in spec.columns) + r" \\")
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}", ""])
    return "\n".join(lines)


def _stable_json(payload: Any) -> str:
    return json.dumps(_to_serializable(payload), indent=2, sort_keys=True) + "\n"


def _to_serializable(value: Any) -> Any:
    if value is None or isinstance(value, str | int | bool):
        return value
    if isinstance(value, float):
        if math.isfinite(value):
            return value
        return None
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, np.ndarray):
        return _to_serializable(value.tolist())
    if hasattr(value, "to_dict"):
        return _to_serializable(value.to_dict())
    if is_dataclass(value):
        return _to_serializable(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _to_serializable(value[key]) for key in sorted(value, key=str)}
    if isinstance(value, Sequence) and not isinstance(value, bytes | bytearray):
        return [_to_serializable(item) for item in value]
    return str(value)


def _normalize_row(row: Mapping[str, Any]) -> dict[str, str]:
    return {str(key): _cell(value) for key, value in row.items()}


def _fieldnames(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    names: list[str] = []
    for row in rows:
        for key in row:
            if key not in names:
                names.append(str(key))
    return names or ["status"]


def _matrix_rows(
    matrix: Matrix,
    row_ids: Sequence[str],
    column_ids: Sequence[str],
) -> tuple[dict[str, Any], ...]:
    rows = []
    for row_id, values in zip(row_ids, matrix):
        row = {"protected_asset_id": row_id}
        row.update({column_id: float(value) for column_id, value in zip(column_ids, values)})
        rows.append(row)
    return tuple(rows)


def _ablation_row(item: CausalAblation | Mapping[str, Any]) -> dict[str, Any]:
    data = item.to_dict() if hasattr(item, "to_dict") else dict(item)
    return {
        "ablation_id": data.get("ablation_id", data.get("id", "unknown")),
        "description": data.get("description", ""),
        "feature_overrides": _compact_dict(data.get("feature_overrides", {})),
        "parameter_overrides": _compact_dict(data.get("parameter_overrides", {})),
        "status": data.get("status", AssessmentStatus.NOT_EVALUATED.value),
    }


def _missing_result_row(message: str) -> dict[str, str]:
    return {"item": message, "status": AssessmentStatus.NOT_EVALUATED.value}


def _compact_dict(value: Any) -> str:
    data = _to_serializable(value)
    if not data:
        return "-"
    if isinstance(data, Mapping):
        return ", ".join(f"{key}={data[key]}" for key in sorted(data, key=str))
    return str(data)


def _cell(value: Any) -> str:
    serial = _to_serializable(value)
    if serial is None:
        return ""
    if isinstance(serial, float):
        return _fmt(serial, 6)
    if isinstance(serial, list | dict):
        return json.dumps(serial, sort_keys=True)
    return str(serial)


def _none_dash(value: Any) -> str:
    if value is None:
        return "-"
    try:
        number = float(value)
    except Exception:
        return str(value)
    if not math.isfinite(number):
        return "-"
    return _fmt(number, 3)


def _fmt(value: float, digits: int) -> str:
    number = float(value)
    if not math.isfinite(number):
        return "-"
    text = f"{number:.{digits}f}"
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text or "0"


def _latex_escape(value: Any) -> str:
    text = _cell(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    return "".join(replacements.get(char, char) for char in text)


def _artifact(
    path: Path,
    artifact_id: str,
    kind: str,
    *,
    row_count: int | None = None,
) -> PaperArtifact:
    return PaperArtifact(
        artifact_id=artifact_id,
        kind=kind,
        path=str(path),
        sha256=sha256_file(path),
        row_count=row_count,
    )


def _resolve_under_root(root: Path, value: str | Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = root / path
    return path.resolve()


def _pyplot():
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    return plt
