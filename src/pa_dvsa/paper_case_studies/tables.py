"""Booktabs tables for the paper case-study section."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from .derived import DatasetPaths
from .loaders import ArtifactBundle, MissingArtifactError
from .style import tex_escape


TABLE_STEMS = (
    "tab01_benchmark_systems",
    "tab02_validation_summary",
    "tab03_mitigation_uncertainty_ablation",
)


@dataclass
class TableResult:
    path: Path
    sha256: str


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _fmt(value: object, digits: int = 2) -> str:
    if value is None:
        return "--"
    try:
        f = float(value)
    except (TypeError, ValueError):
        text = str(value)
        return "--" if text.lower() in {"nan", "none", ""} else tex_escape(text)
    if not np.isfinite(f):
        return "--"
    if abs(f - round(f)) < 1e-9:
        return str(int(round(f)))
    return f"{f:.{digits}f}"


def _write_booktabs(
    df: pd.DataFrame,
    path: Path,
    *,
    caption: str,
    label: str,
    column_format: str | None = None,
) -> TableResult:
    path.parent.mkdir(parents=True, exist_ok=True)
    if column_format is None:
        column_format = "l" + "c" * (len(df.columns) - 1)
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        rf"\begin{{tabular}}{{{column_format}}}",
        r"\toprule",
        " & ".join(tex_escape(col) for col in df.columns) + r" \\",
        r"\midrule",
    ]
    for _, row in df.iterrows():
        lines.append(" & ".join(str(row[col]) for col in df.columns) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")
    return TableResult(path=path, sha256=_sha(path))


def _write_csv(df: pd.DataFrame, path: Path) -> TableResult:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return TableResult(path=path, sha256=_sha(path))


def make_tab01_benchmark_systems(
    paths: DatasetPaths,
    bundle: ArtifactBundle,
    *,
    out_table: Path,
) -> list[TableResult]:
    systems = pd.read_csv(paths.systems)
    systems = systems[systems["case"].isin(["kundur", "ieee39", "npcc", "gbnetwork"])]
    rows = []
    for _, row in systems.iterrows():
        rows.append(
            {
                "benchmark": tex_escape(row["benchmark"]),
                "size": f"{_fmt(row['buses'], 0)} / {_fmt(row['lines'], 0)}",
                "DAE": f"{_fmt(row['dae_states'], 0)} / {_fmt(row['dae_algebraic'], 0)}",
                "protected": _fmt(row["protected_assets"], 0),
                "hidden": _fmt(row["hidden_outputs"], 0),
                "controls": _fmt(row["control_devices"], 0),
                "event families": tex_escape(row.get("event_families", "")),
                "events": _fmt(row["candidate_events"], 0),
                "TDS": "yes" if bool(row["tds_initialized"]) else "no",
            }
        )
    df = pd.DataFrame(rows)
    stem = "tab01_benchmark_systems"
    return [
        _write_csv(df, out_table / f"{stem}.csv"),
        _write_booktabs(
            df,
            out_table / f"{stem}.tex",
            caption=(
                "Benchmark systems and scenario libraries used in the case studies. "
                "The size column reports buses/branches and the DAE column reports states/algebraic variables."
            ),
            label="tab:benchmark_systems",
            column_format="lcccccccc",
        ),
    ]


def make_tab02_validation_summary(
    paths: DatasetPaths,
    bundle: ArtifactBundle,
    *,
    out_table: Path,
) -> list[TableResult]:
    val = pd.read_csv(paths.tds_validation)
    scenarios = pd.read_csv(paths.scenario_library)
    triage = pd.read_csv(paths.scaling_summary)
    runtime = pd.read_csv(paths.systems)
    rows = []
    for (case, fam), group in val.groupby(["case", "scenario_family"]):
        seed_group = group.groupby("seed_event").agg(
            actual_any=("actual_trip", "any"),
            predicted_any=("predicted_trip", "any"),
            flagged_any=("event_screen_flagged", "any") if "event_screen_flagged" in group else ("predicted_trip", "any"),
            data_limited_any=("data_limited", "any"),
        )
        dangerous = int(seed_group["actual_any"].sum())
        fn = int(((seed_group["actual_any"]) & (~seed_group["flagged_any"]) & (~seed_group["data_limited_any"])).sum())
        fp = int(((seed_group["flagged_any"]) & (~seed_group["actual_any"]) & (~seed_group["data_limited_any"])).sum())
        tp = int(((group["predicted_trip"]) & (group["actual_trip"])).sum())
        union = int(((group["predicted_trip"]) | (group["actual_trip"])).sum())
        overlap = tp / union if union else 1.0
        sub_triage = triage[triage["case"] == case]
        top1 = float(sub_triage.iloc[0]["capture_rate"]) if not sub_triage.empty else np.nan
        topk = float(sub_triage["capture_rate"].max()) if not sub_triage.empty else np.nan
        rt = runtime[runtime["case"] == case]
        screen_rt = rt["screen_runtime_s"].iloc[0] if not rt.empty else np.nan
        total_scenarios = scenarios[scenarios["case"] == case]["event_id"].nunique()
        rows.append(
            {
                "benchmark": tex_escape(group["benchmark"].iloc[0]),
                "family": tex_escape(fam),
                "screened": _fmt(total_scenarios, 0),
                "TDS": _fmt(group["seed_event"].nunique(), 0),
                "dangerous": _fmt(dangerous, 0),
                "top-1": _fmt(top1, 2),
                "top-k": _fmt(topk, 2),
                "FN": _fmt(fn, 0),
                "FP": _fmt(fp, 0),
                "overlap": _fmt(overlap, 2),
                "screen s": _fmt(screen_rt, 2),
                "nonlinear s": "--",
            }
        )
    df = pd.DataFrame(rows)
    stem = "tab02_validation_summary"
    return [
        _write_csv(df, out_table / f"{stem}.csv"),
        _write_booktabs(
            df,
            out_table / f"{stem}.tex",
            caption=(
                "Screen-versus-nonlinear validation summary. False negatives are reported explicitly; "
                "nonlinear runtime is left blank when no calibrated timing artifact exists."
            ),
            label="tab:validation_summary",
            column_format="llcccccccccc",
        ),
    ]


def make_tab03_mitigation_uncertainty_ablation(
    paths: DatasetPaths,
    bundle: ArtifactBundle,
    *,
    out_table: Path,
) -> list[TableResult]:
    abl = pd.read_csv(paths.ablation_summary)
    rows = []
    for _, row in abl.iterrows():
        rows.append(
            {
                "case": tex_escape(row["row"]),
                "worst $K$": _fmt(row["worst_k"], 2),
                "pred. size": _fmt(row["predicted_cascade_size"], 0),
                "actual size": _fmt(row["actual_cascade_size"], 0),
                "class": tex_escape(row["classification"]),
                "MVAr": _fmt(row["minimum_mvar_mitigation"], 1),
                "slack": _fmt(row["slack"], 2),
                "takeaway": tex_escape(row["engineering_takeaway"]),
            }
        )
    df = pd.DataFrame(rows)
    stem = "tab03_mitigation_uncertainty_ablation"
    return [
        _write_csv(df, out_table / f"{stem}.csv"),
        _write_booktabs(
            df,
            out_table / f"{stem}.tex",
            caption=(
                "Ablation, uncertainty, and mitigation summary. Screen-level rows are derived from the stored "
                "screen artifacts; unsupported model rows are labeled rather than evaluated."
            ),
            label="tab:ablation_uncertainty_mitigation",
            column_format="lccccclp{0.31\\linewidth}",
        ),
    ]


TABLE_BUILDERS: dict[str, Callable[..., list[TableResult]]] = {
    "tab01_benchmark_systems": make_tab01_benchmark_systems,
    "tab02_validation_summary": make_tab02_validation_summary,
    "tab03_mitigation_uncertainty_ablation": make_tab03_mitigation_uncertainty_ablation,
}
