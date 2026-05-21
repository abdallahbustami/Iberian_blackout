"""Additional replication-subsection figures.

These figures are intentionally separate from the main eight case-study figures.
They compare the physical ACTIVSg2000 mechanism replica against the real
variant runs produced by :mod:`pa_dvsa.replication.academic_replica`.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

from pa_dvsa.paper_case_studies.loaders import MissingArtifactError
from pa_dvsa.paper_case_studies.style import (
    FAMILY_COLOR,
    PALETTE,
    SaveResult,
    configure_style,
    save_figure,
    sha256_file,
    soften_axes,
    tex_escape,
)


VARIANT_ORDER: tuple[str, ...] = (
    "baseline",
    "no_pre_voltage_actions",
    "voltage_mode_res",
    "preserved_q_absorption",
    "no_post_cascade_relays",
    "strong_defense",
)

VARIANT_LABELS: dict[str, str] = {
    "baseline": "baseline",
    "no_pre_voltage_actions": "no pre-voltage\nactions",
    "voltage_mode_res": "voltage-mode\nRES",
    "preserved_q_absorption": "preserved\nQ absorption",
    "no_post_cascade_relays": "collector trips\nonly",
    "strong_defense": "stronger\ndefence",
}

VARIANT_SHORT_LABELS: dict[str, str] = {
    "baseline": "baseline",
    "no_pre_voltage_actions": "no OA",
    "voltage_mode_res": "V-mode RES",
    "preserved_q_absorption": "preserved Q",
    "no_post_cascade_relays": "trips only",
    "strong_defense": "strong defence",
}

VARIANT_COLORS: dict[str, str] = {
    "baseline": PALETTE["trip"],
    "no_pre_voltage_actions": PALETTE["safe"],
    "voltage_mode_res": PALETTE["control"],
    "preserved_q_absorption": PALETTE["data"],
    "no_post_cascade_relays": PALETTE["risky"],
    "strong_defense": PALETTE["charcoal"],
}

EVENT_MARKERS: dict[str, tuple[str, str]] = {
    "first_trip_time_s": ("o", "collector trip"),
    "first_generator_protection_time_s": ("D", "gen. protection"),
    "first_morocco_ac_trip_time_s": ("^", "Morocco AC"),
    "first_france_ac_separation_time_s": ("v", "France AC"),
    "first_hvdc_block_time_s": ("s", "HVDC block"),
    "blackout_time_s": ("X", "blackout"),
}


@dataclass
class VariantRun:
    name: str
    root: Path
    summary: dict
    events: list[dict]
    collectors: pd.DataFrame
    system: pd.DataFrame

    @property
    def label(self) -> str:
        return VARIANT_LABELS.get(self.name, self.name.replace("_", " "))

    @property
    def color(self) -> str:
        return VARIANT_COLORS.get(self.name, PALETTE["charcoal"])


@dataclass
class ReplicationFigureBundle:
    variants: dict[str, VariantRun]
    input_hashes: dict[str, str]


def _relpath(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(path.resolve())


def _read_json(path: Path) -> object:
    if not path.exists():
        raise MissingArtifactError(f"Missing required replication artifact: {_relpath(path)}")
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise MissingArtifactError(f"Missing required replication artifact: {_relpath(path)}")
    return pd.read_csv(path)


def _variant_root(replication_root: Path, variants_root: Path, variant: str) -> Path:
    if variant == "baseline":
        return replication_root
    return variants_root / variant


def load_variant(replication_root: Path, variants_root: Path, variant: str) -> VariantRun:
    root = _variant_root(replication_root, variants_root, variant)
    summary_obj = _read_json(root / "summary.json")
    events_obj = _read_json(root / "events.json")
    if not isinstance(summary_obj, dict):
        raise MissingArtifactError(f"Expected object JSON: {_relpath(root / 'summary.json')}")
    if not isinstance(events_obj, list):
        raise MissingArtifactError(f"Expected list JSON: {_relpath(root / 'events.json')}")
    collectors = _read_csv(root / "collector_traces.csv")
    system = _read_csv(root / "system_traces.csv")
    required_collectors = {"time_s"}
    required_system = {"time_s", "cumulative_lost_q_absorption_mvar"}
    missing_collectors = required_collectors.difference(collectors.columns)
    missing_system = required_system.difference(system.columns)
    if missing_collectors:
        raise MissingArtifactError(
            f"Missing columns in {_relpath(root / 'collector_traces.csv')}: {sorted(missing_collectors)}"
        )
    if missing_system:
        raise MissingArtifactError(
            f"Missing columns in {_relpath(root / 'system_traces.csv')}: {sorted(missing_system)}"
        )
    return VariantRun(
        name=variant,
        root=root,
        summary=summary_obj,
        events=[dict(item) for item in events_obj if isinstance(item, dict)],
        collectors=collectors,
        system=system,
    )


def load_bundle(replication_root: Path, variants_root: Path) -> ReplicationFigureBundle:
    variants = {
        variant: load_variant(replication_root, variants_root, variant)
        for variant in VARIANT_ORDER
    }
    input_hashes: dict[str, str] = {}
    for run in variants.values():
        for name in ("summary.json", "events.json", "collector_traces.csv", "system_traces.csv"):
            path = run.root / name
            input_hashes[_relpath(path)] = sha256_file(path)
    return ReplicationFigureBundle(variants=variants, input_hashes=input_hashes)


def _collector_names(df: pd.DataFrame) -> list[str]:
    names: list[str] = []
    for column in df.columns:
        if not column.endswith("_voltage_pu"):
            continue
        if "_transmission_" in column or "_raw" in column:
            continue
        name = column[: -len("_voltage_pu")]
        if f"{name}_threshold_pu" in df.columns:
            names.append(name)
    return names


def _max_utilization(run: VariantRun) -> pd.Series:
    names = _collector_names(run.collectors)
    if not names:
        raise MissingArtifactError(
            f"No collector voltage/threshold columns found in {_relpath(run.root / 'collector_traces.csv')}"
        )
    ratios = []
    for name in names:
        v = pd.to_numeric(run.collectors[f"{name}_voltage_pu"], errors="coerce")
        thr = pd.to_numeric(run.collectors[f"{name}_threshold_pu"], errors="coerce")
        ratios.append(v / thr)
    return pd.concat(ratios, axis=1).max(axis=1)


def _summary_row(run: VariantRun) -> dict[str, object]:
    util = _max_utilization(run)
    total_q = float(run.summary.get("total_lost_q_absorption_mvar") or 0.0)
    total_p = float(run.summary.get("total_lost_p_mw") or 0.0)
    blackout_time = run.summary.get("blackout_time_s")
    return {
        "variant": run.name,
        "label": run.label.replace("\n", " "),
        "peak_utilization": float(util.max()),
        "collector_trips": int(run.summary.get("collector_trips") or 0),
        "first_trip_time_s": run.summary.get("first_trip_time_s"),
        "blackout_time_s": blackout_time,
        "blackout_detected": blackout_time is not None,
        "total_lost_p_mw": total_p,
        "total_lost_q_absorption_mvar": total_q,
        "final_time_s": float(run.summary.get("final_time_s") or run.collectors["time_s"].max()),
    }


def write_derived_tables(bundle: ReplicationFigureBundle, work_dir: Path) -> dict[str, Path]:
    work_dir.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame([_summary_row(bundle.variants[v]) for v in VARIANT_ORDER])
    summary_path = work_dir / "replication_variant_summary.csv"
    summary.to_csv(summary_path, index=False)

    rows: list[dict[str, object]] = []
    for variant, run in bundle.variants.items():
        for event in run.events:
            q = float(event.get("lost_q_absorption_mvar") or 0.0)
            p = float(event.get("lost_p_mw") or 0.0)
            if not q and not p:
                continue
            rows.append(
                {
                    "variant": variant,
                    "time_s": float(event.get("time_s") or math.nan),
                    "code": str(event.get("code") or ""),
                    "category": str(event.get("category") or ""),
                    "lost_p_mw": p,
                    "lost_q_absorption_mvar": q,
                    "description": str(event.get("description") or ""),
                }
            )
    events_path = work_dir / "replication_mvar_events.csv"
    pd.DataFrame(rows).to_csv(events_path, index=False)
    return {"summary": summary_path, "mvar_events": events_path}


def _max_utilization_raw(run: VariantRun) -> pd.Series:
    names: list[str] = []
    for column in run.collectors.columns:
        if column.endswith("_voltage_pu_raw"):
            name = column[: -len("_voltage_pu_raw")]
            if f"{name}_threshold_pu" in run.collectors.columns:
                names.append(name)
    if not names:
        return _max_utilization(run)
    ratios = []
    for name in names:
        v = pd.to_numeric(run.collectors[f"{name}_voltage_pu_raw"], errors="coerce")
        thr = pd.to_numeric(run.collectors[f"{name}_threshold_pu"], errors="coerce")
        ratios.append(v / thr)
    return pd.concat(ratios, axis=1).max(axis=1)


def _value_at(time: np.ndarray, values: np.ndarray, t_query: float) -> float:
    idx = int(np.searchsorted(time, t_query, side="right") - 1)
    idx = max(0, min(idx, len(time) - 1))
    return float(values[idx])


def _operator_waterfall(run: VariantRun) -> pd.DataFrame:
    time = pd.to_numeric(run.collectors["time_s"], errors="coerce").to_numpy(dtype=float)
    util = _max_utilization_raw(run).to_numpy(dtype=float)
    operator_by_code: dict[str, list[float]] = {}
    for event in run.events:
        if event.get("category") != "operator_action":
            continue
        try:
            operator_by_code.setdefault(str(event.get("code", "OA")), []).append(float(event["time_s"]))
        except (KeyError, TypeError, ValueError):
            continue
    first_trip = _event_time(run.summary, "first_trip_time_s")
    if first_trip is None:
        first_trip = float(time[-1])
    checkpoints: list[tuple[str, float]] = [("initial", float(time[0]))]
    if operator_by_code.get("OA1"):
        checkpoints.append(("OA1\nmeshing", max(operator_by_code["OA1"]) + 0.30))
    if operator_by_code.get("OA3"):
        checkpoints.append(("OA3\nshunts", max(operator_by_code["OA3"]) + 0.25))
    if operator_by_code.get("OA4"):
        checkpoints.append(("OA4\nHVDC ref.", max(operator_by_code["OA4"]) + 0.25))
    # The export/fixed-PF action is ramped, so evaluate its cumulative effect
    # just before the first relay trip rather than at the ramp start.
    if operator_by_code.get("OA2"):
        checkpoints.append(("OA2\nfixed-PF ramp", max(float(time[0]), first_trip - 0.05)))
    values = [(label, t, _value_at(time, util, t)) for label, t in checkpoints]
    rows: list[dict[str, object]] = []
    previous = values[0][2]
    for i, (label, t, value) in enumerate(values):
        rows.append(
            {
                "stage": label,
                "time_s": t,
                "ratio": value,
                "delta": value - previous if i else 0.0,
                "kind": "initial" if i == 0 else "operator_action",
            }
        )
        previous = value
    rows.append(
        {
            "stage": "pre-trip\nstate",
            "time_s": float(first_trip),
            "ratio": values[-1][2],
            "delta": 0.0,
            "kind": "final",
        }
    )
    return pd.DataFrame(rows)


def _action_phase_data(case_dataset_root: Path) -> pd.DataFrame:
    assets_path = case_dataset_root / "assets.csv"
    k_path = case_dataset_root / "k_matrix.csv"
    if not assets_path.exists() or not k_path.exists():
        raise MissingArtifactError(
            "Figure 9 needs cross-benchmark assets.csv and k_matrix.csv from "
            f"{case_dataset_root}"
        )
    assets = pd.read_csv(assets_path)
    km = pd.read_csv(k_path)
    required_assets = {"case", "asset", "margin_pu"}
    required_k = {"case", "asset", "event_id", "event_family", "k_pickup"}
    missing_assets = required_assets.difference(assets.columns)
    missing_k = required_k.difference(km.columns)
    if missing_assets:
        raise MissingArtifactError(f"{assets_path} missing columns: {sorted(missing_assets)}")
    if missing_k:
        raise MissingArtifactError(f"{k_path} missing columns: {sorted(missing_k)}")
    merged = km.merge(assets[["case", "asset", "margin_pu"]], on=["case", "asset"], how="left")
    merged["margin_pu"] = pd.to_numeric(merged["margin_pu"], errors="coerce")
    merged["k_pickup"] = pd.to_numeric(merged["k_pickup"], errors="coerce")
    merged = merged.dropna(subset=["margin_pu", "k_pickup"])
    merged = merged[(merged["margin_pu"] > 0.0) & (merged["k_pickup"] >= 0.0)].copy()
    if merged.empty:
        raise MissingArtifactError("Figure 9 action-impact phase data is empty after filtering")
    return merged


def _event_time(summary: dict, key: str) -> float | None:
    value = summary.get(key)
    if value is None:
        return None
    try:
        value_f = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(value_f):
        return None
    return value_f


def _baseline_reference_events(run: VariantRun) -> list[tuple[float, str, str]]:
    refs: list[tuple[float, str, str]] = []
    # Keep only the most useful labels. Dense OA markers are summarized as a band.
    key_map = (
        ("first_trip_time_s", "E3", "collector trip"),
        ("first_generator_protection_time_s", "GEN", "generator protection"),
        ("first_morocco_ac_trip_time_s", "MA", "Morocco AC"),
        ("first_france_ac_separation_time_s", "FR", "France AC"),
        ("first_hvdc_block_time_s", "HVDC", "HVDC block"),
        ("blackout_time_s", "BO", "blackout"),
    )
    for key, code, label in key_map:
        t = _event_time(run.summary, key)
        if t is not None:
            refs.append((t, code, label))
    return refs


def _plot_event_ribbon(ax: plt.Axes, baseline: VariantRun, *, y: float, height: float) -> None:
    """Draw a compact baseline event ribbon in data-x / axes-y coordinates."""

    trans = ax.get_xaxis_transform()
    # Operator-action span.
    operator_times = [
        float(e.get("time_s"))
        for e in baseline.events
        if e.get("category") == "operator_action" and e.get("time_s") is not None
    ]
    if operator_times:
        x0, x1 = min(operator_times), max(operator_times)
        ax.add_patch(
            plt.Rectangle(
                (x0, y),
                x1 - x0,
                height,
                transform=trans,
                facecolor=PALETTE["light_blue"],
                edgecolor=PALETTE["control"],
                linewidth=0.65,
                alpha=0.90,
                clip_on=False,
            )
        )
        ax.text(
            (x0 + x1) / 2,
            y + height / 2,
            "OA",
            transform=trans,
            ha="center",
            va="center",
            fontsize=6.8,
            color=PALETTE["control"],
        )
    for t, code, _ in _baseline_reference_events(baseline):
        color = PALETTE["trip"] if code in {"E3", "GEN", "BO"} else PALETTE["charcoal"]
        ax.plot([t, t], [y, y + height * 1.9], transform=trans, color=color, lw=0.75, clip_on=False)
        ax.text(
            t,
            y - 0.035,
            tex_escape(code),
            transform=trans,
            rotation=90,
            ha="center",
            va="top",
            fontsize=6.4,
            color=color,
            clip_on=False,
        )


def make_fig09_operator_condition_voltage(
    bundle: ReplicationFigureBundle,
    out_dir: Path,
    *,
    case_dataset_root: Path | None = None,
    formats: Iterable[str] = ("pdf",),
    outline_text: bool = True,
) -> list[SaveResult]:
    """Operator-action voltage waterfall and cross-benchmark impact map."""

    configure_style(prefer_usetex=True)
    if case_dataset_root is None:
        case_dataset_root = Path("results/paper_case_studies/dataset")
    phase = _action_phase_data(case_dataset_root)
    baseline = bundle.variants["baseline"]
    waterfall = _operator_waterfall(baseline)

    fig = plt.figure(figsize=(3.45, 4.95))
    gs = fig.add_gridspec(
        2,
        1,
        height_ratios=[1.02, 0.95],
        left=0.20,
        right=0.975,
        top=0.935,
        bottom=0.130,
        hspace=0.48,
    )
    ax_w = fig.add_subplot(gs[0, 0])
    ax_p = fig.add_subplot(gs[1, 0])

    stages = waterfall["stage"].astype(str).tolist()
    ratios = waterfall["ratio"].to_numpy(dtype=float)
    deltas = waterfall["delta"].to_numpy(dtype=float)
    x = np.arange(len(stages))
    floor = max(0.0, float(np.nanmin(ratios)) - 0.006)
    y_max = max(1.010, float(np.nanmax(ratios)) + 0.010)
    prev = ratios[0]
    for i, (stage, ratio, delta, kind) in enumerate(
        zip(stages, ratios, deltas, waterfall["kind"].astype(str), strict=True)
    ):
        if kind in {"initial", "final"}:
            bottom = floor
            height = ratio - floor
            color = PALETTE["charcoal"] if kind == "initial" else PALETTE["trip"]
            edge = PALETTE["charcoal"]
        else:
            bottom = min(prev, ratio)
            height = abs(ratio - prev)
            color = PALETTE["trip"] if ratio >= 1.0 else PALETTE["risky"]
            edge = "white"
        ax_w.bar(
            i,
            height,
            bottom=bottom,
            width=0.66,
            color=color,
            edgecolor=edge,
            linewidth=0.8,
            alpha=0.92,
            zorder=3,
        )
        ax_w.text(
            i,
            ratio + 0.0017,
            f"{ratio:.3f}" if kind in {"initial", "final"} else f"{delta * 100:+.2f} pp",
            ha="center",
            va="bottom",
            fontsize=6.6,
            color=PALETTE["charcoal"] if kind != "final" else PALETTE["trip"],
            rotation=90 if kind not in {"initial", "final"} else 0,
        )
        if i > 0:
            ax_w.plot([i - 0.33, i + 0.33], [prev, prev], color=PALETTE["grid"], lw=0.75, zorder=2)
        prev = ratio
    ax_w.plot(x, ratios, color=PALETTE["charcoal"], lw=0.85, marker="o", ms=3.0, mfc="white", zorder=4)
    ax_w.axhline(1.0, color=PALETTE["trip"], lw=1.0, ls=(0, (5, 2.4)))
    ax_w.axhspan(1.0, y_max, color=PALETTE["light_red"], alpha=0.38, zorder=0)
    short_stages = ["initial", "OA1", "OA3", "OA4", "OA2", "relay\narmed"]
    ax_w.set_xticks(x, short_stages, rotation=0)
    ax_w.set_ylabel(r"max protected-voltage ratio $z_i/V_i^{\rm trip}$")
    ax_w.set_ylim(floor, y_max)
    ax_w.text(
        len(stages) - 0.25,
        1.001,
        "relay pickup",
        ha="right",
        va="bottom",
        fontsize=6.8,
        color=PALETTE["trip"],
    )
    ax_w.text(
        0.00,
        1.04,
        r"\textbf{a}" if plt.rcParams.get("text.usetex") else "a",
        transform=ax_w.transAxes,
        ha="left",
        va="bottom",
        fontsize=10.2,
        color=PALETTE["charcoal"],
    )
    soften_axes(ax_w, grid=True)

    system_order = ["kundur", "ieee39", "npcc", "gbnetwork"]
    family_order = [
        "load/pump disconnection",
        "fixed-PF RES ramp",
        "plant/generator trip",
        "export reduction",
        "shunt/reactor action",
    ]
    family_labels = {
        "load/pump disconnection": "load/pump",
        "fixed-PF RES ramp": "fixed-PF RES",
        "plant/generator trip": "plant/gen.",
        "export reduction": "export",
        "shunt/reactor action": "shunt/reactor",
    }
    phase = phase[phase["case"].isin(system_order) & phase["event_family"].isin(family_order)].copy()
    per_event = (
        phase.groupby(["case", "event_family", "event_id"], as_index=False)["k_pickup"]
        .max()
        .rename(columns={"k_pickup": "event_max_k"})
    )
    per_event["k_plot"] = per_event["event_max_k"].clip(lower=0.01)
    y_positions = np.arange(len(family_order))
    max_x = max(120.0, float(per_event["k_plot"].max()) * 1.25)
    for y_i, family in enumerate(family_order):
        vals = per_event.loc[per_event["event_family"] == family, "k_plot"].to_numpy(dtype=float)
        if vals.size:
            q10, q50, q90 = np.quantile(vals, [0.10, 0.50, 0.90])
            vmin, vmax = float(vals.min()), float(vals.max())
            dangerous = int((vals >= 1.0).sum())
            total = int(vals.size)
            ax_p.plot([vmin, vmax], [y_i, y_i], color=PALETTE["grid"], lw=1.0, zorder=1)
            ax_p.plot([q10, q90], [y_i, y_i], color=PALETTE["background"], lw=8.5, solid_capstyle="round", zorder=2)
            ax_p.scatter(
                [q50],
                [y_i],
                marker="D",
                s=35,
                facecolor=PALETTE["charcoal"],
                edgecolor="white",
                linewidth=0.7,
                zorder=4,
            )
            ax_p.text(
                0.985,
                y_i,
                f"{dangerous}/{total}",
                transform=ax_p.get_yaxis_transform(),
                ha="right",
                va="center",
                fontsize=6.5,
                color=PALETTE["trip"] if dangerous else PALETTE["charcoal"],
            )
    ax_p.axvspan(1.0, max_x, color=PALETTE["light_red"], alpha=0.30, zorder=0)
    ax_p.axvline(1.0, color=PALETTE["trip"], lw=1.0, ls=(0, (5, 2.4)), zorder=3)
    ax_p.set_xscale("log")
    ax_p.set_xlim(0.01, max_x)
    ax_p.set_ylim(len(family_order) - 0.52, -0.52)
    ax_p.set_yticks(y_positions, [family_labels[f] for f in family_order])
    ax_p.set_xlabel(r"action impact distribution, $\max_i K_{ij}^{\rm pk}$")
    ax_p.text(
        1.03,
        -0.43,
        r"$K=1$ boundary",
        ha="left",
        va="bottom",
        fontsize=6.8,
        color=PALETTE["trip"],
    )
    ax_p.text(
        0.015,
        -0.43,
        "lower impact",
        ha="left",
        va="bottom",
        fontsize=6.8,
        color=PALETTE["charcoal"],
    )
    legend_handles = [
        Line2D([0], [0], color=PALETTE["background"], lw=6.0, solid_capstyle="round", label="10--90\\% range"),
        Line2D([0], [0], marker="D", color="none", markerfacecolor=PALETTE["charcoal"], markeredgecolor="white", markersize=5.0, label="median"),
        Line2D([0], [0], color=PALETTE["grid"], lw=1.0, label="min--max"),
    ]
    ax_p.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.50, 1.18),
        ncol=3,
        frameon=False,
        fontsize=6.2,
        handletextpad=0.30,
        columnspacing=0.55,
    )
    ax_p.text(
        0.00,
        1.04,
        r"\textbf{b}" if plt.rcParams.get("text.usetex") else "b",
        transform=ax_p.transAxes,
        ha="left",
        va="bottom",
        fontsize=10.2,
        color=PALETTE["charcoal"],
    )
    soften_axes(ax_p, grid=True, ygrid=False)
    return save_figure(fig, "fig09_replication_operator_conditions", out_dir, formats, outline_text=outline_text)


def _cumulative_q(run: VariantRun) -> tuple[np.ndarray, np.ndarray]:
    t = pd.to_numeric(run.system["time_s"], errors="coerce").to_numpy()
    q = pd.to_numeric(run.system["cumulative_lost_q_absorption_mvar"], errors="coerce").to_numpy()
    return t, q


def make_fig10_mvar_absorption(
    bundle: ReplicationFigureBundle,
    out_dir: Path,
    *,
    formats: Iterable[str] = ("pdf",),
    outline_text: bool = True,
) -> list[SaveResult]:
    """Reactive absorption loss and overvoltage outcome."""

    configure_style(prefer_usetex=True)
    fig = plt.figure(figsize=(7.0, 4.05))
    gs = fig.add_gridspec(
        1,
        2,
        width_ratios=[1.15, 0.85],
        left=0.08,
        right=0.985,
        top=0.88,
        bottom=0.25,
        wspace=0.28,
    )
    ax_q = fig.add_subplot(gs[0, 0])
    ax_s = fig.add_subplot(gs[0, 1])
    baseline = bundle.variants["baseline"]

    for variant in VARIANT_ORDER:
        run = bundle.variants[variant]
        t, q = _cumulative_q(run)
        lw = 2.15 if variant == "baseline" else 1.55
        ax_q.step(t, q, where="post", color=run.color, lw=lw, alpha=0.93, label=tex_escape(run.label.replace("\n", " ")))

    _plot_event_ribbon(ax_q, baseline, y=-0.18, height=0.060)
    ax_q.set_xlim(0, 24.2)
    ymax = max(float(bundle.variants[v].system["cumulative_lost_q_absorption_mvar"].max()) for v in VARIANT_ORDER)
    ax_q.set_ylim(0, max(100.0, ymax * 1.20))
    ax_q.set_ylabel(r"cumulative lost reactive absorption (MVAr)")
    ax_q.set_xlabel("time after first report window (s)")
    ax_q.legend(loc="upper left", frameon=False, fontsize=6.9, ncol=1)
    ax_q.text(
        0.00,
        1.04,
        r"\textbf{a}" if plt.rcParams.get("text.usetex") else "a",
        transform=ax_q.transAxes,
        ha="left",
        va="bottom",
        fontsize=10.2,
        color=PALETTE["charcoal"],
    )
    soften_axes(ax_q, grid=True)

    rows = pd.DataFrame([_summary_row(bundle.variants[v]) for v in VARIANT_ORDER])
    for _, row in rows.iterrows():
        variant = str(row["variant"])
        run = bundle.variants[variant]
        trips = int(row["collector_trips"])
        blackout = bool(row["blackout_detected"])
        size = 48 + trips * 18
        marker = "X" if blackout else "o"
        ax_s.scatter(
            float(row["peak_utilization"]),
            float(row["total_lost_q_absorption_mvar"]),
            s=size,
            marker=marker,
            facecolor=run.color,
            edgecolor=PALETTE["charcoal"] if blackout else "white",
            linewidth=1.0,
            alpha=0.94,
            zorder=3,
        )
        offsets = {
            "baseline": (-0.0028, -28.0, "right"),
            "strong_defense": (-0.0030, 36.0, "right"),
            "no_post_cascade_relays": (0.0023, 0.0, "left"),
            "voltage_mode_res": (0.0022, -18.0, "left"),
            "preserved_q_absorption": (0.0022, 18.0, "left"),
            "no_pre_voltage_actions": (0.0018, 16.0, "left"),
        }
        dx, dy, ha = offsets.get(variant, (0.0018, 0.0, "left"))
        ax_s.text(
            float(row["peak_utilization"]) + dx,
            float(row["total_lost_q_absorption_mvar"]) + dy,
            tex_escape(VARIANT_SHORT_LABELS.get(variant, run.label.replace("\n", " "))),
            ha=ha,
            va="center",
            fontsize=6.8,
            color=PALETTE["charcoal"],
        )
    ax_s.axvspan(1.0, 1.06, color=PALETTE["light_red"], alpha=0.42, zorder=0)
    ax_s.axvline(1.0, color=PALETTE["trip"], lw=1.0, ls=(0, (5, 2.4)), zorder=2)
    ax_s.text(
        1.001,
        ax_s.get_ylim()[1] * 0.96,
        r"$z_i/V_i^{\rm trip}=1$",
        ha="left",
        va="top",
        fontsize=6.9,
        color=PALETTE["trip"],
    )
    ax_s.set_xlim(0.988, max(1.060, float(rows["peak_utilization"].max()) + 0.012))
    ax_s.set_ylim(-25, max(120.0, float(rows["total_lost_q_absorption_mvar"].max()) * 1.16))
    ax_s.set_xlabel(r"peak protected-voltage ratio")
    ax_s.set_ylabel(r"total lost MVAr absorption")
    ax_s.text(
        0.00,
        1.04,
        r"\textbf{b}" if plt.rcParams.get("text.usetex") else "b",
        transform=ax_s.transAxes,
        ha="left",
        va="bottom",
        fontsize=10.2,
        color=PALETTE["charcoal"],
    )
    legend_handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=PALETTE["background"], markeredgecolor=PALETTE["charcoal"], markersize=6.5, label="no blackout"),
        Line2D([0], [0], marker="X", color="none", markerfacecolor=PALETTE["trip"], markeredgecolor=PALETTE["charcoal"], markersize=7.0, label="blackout endpoint"),
    ]
    ax_s.legend(handles=legend_handles, loc="lower right", frameon=False, fontsize=6.8)
    soften_axes(ax_s, grid=True)
    return save_figure(fig, "fig10_replication_mvar_absorption", out_dir, formats, outline_text=outline_text)


def make_all(
    *,
    replication_root: Path,
    variants_root: Path,
    out_dir: Path,
    work_dir: Path,
    case_dataset_root: Path = Path("results/paper_case_studies/dataset"),
    formats: Iterable[str] = ("pdf",),
    outline_text: bool = True,
    only: str = "all",
) -> tuple[list[SaveResult], dict[str, object]]:
    bundle = load_bundle(replication_root, variants_root)
    derived_paths = write_derived_tables(bundle, work_dir)
    results: list[SaveResult] = []
    if only in {"all", "fig09_replication_operator_conditions"}:
        results.extend(
            make_fig09_operator_condition_voltage(
                bundle,
                out_dir,
                case_dataset_root=case_dataset_root,
                formats=formats,
                outline_text=outline_text,
            )
        )
    if only in {"all", "fig10_replication_mvar_absorption"}:
        results.extend(
            make_fig10_mvar_absorption(
                bundle,
                out_dir,
                formats=formats,
                outline_text=outline_text,
            )
        )
    if only not in {"all", "fig09_replication_operator_conditions", "fig10_replication_mvar_absorption"}:
        raise ValueError(f"unknown artifact requested by --only: {only}")

    input_hashes = dict(bundle.input_hashes)
    for name in ("assets.csv", "k_matrix.csv"):
        path = case_dataset_root / name
        if path.exists():
            input_hashes[_relpath(path)] = sha256_file(path)

    manifest = {
        "inputs": input_hashes,
        "derived": {_relpath(path): sha256_file(path) for path in derived_paths.values()},
        "outputs": {_relpath(item.path): item.sha256 for item in results},
    }
    return results, manifest
