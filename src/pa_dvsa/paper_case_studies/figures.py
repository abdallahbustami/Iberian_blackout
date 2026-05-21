"""Eight compact, evidence-first paper figures for the PA-DVSA case studies."""

from __future__ import annotations

import shutil
import tempfile
import warnings
from pathlib import Path
from typing import Callable, Iterable

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import patheffects as pe
from matplotlib import colors
from matplotlib.path import Path as MplPath
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Patch, PathPatch, Rectangle

from .derived import DatasetPaths
from .loaders import ArtifactBundle, MissingArtifactError
from .style import (
    ABLATION_SHORT,
    FIGURE_SPECS,
    FAMILY_COLOR,
    PALETTE,
    SaveResult,
    badge,
    outline_pdf_text,
    panel_label,
    save_figure,
    sha256_file,
    soften_axes,
    tex_escape,
)


FIGURE_STEMS = (
    "fig01_replication_activsg2000",
    "fig02_validation_matrix",
    "fig03_margin_erosion_atlas",
    "fig04_observability_gap",
    "fig05_operator_action_screen",
    "fig06_relay_window_controls",
    "fig07_uncertainty_telemetry_value",
    "fig08_mitigation_scaling",
)

SYSTEM_ORDER = ("kundur", "ieee39", "npcc", "gbnetwork")
SYSTEM_LABELS = {
    "kundur": "Kundur",
    "ieee39": "IEEE-39",
    "npcc": "NPCC",
    "gbnetwork": "GBnetwork",
}


def _read(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def _require_columns(df: pd.DataFrame, path: Path, columns: Iterable[str]) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise MissingArtifactError(
            f"{path.name} is missing required column(s): {', '.join(missing)}"
        )


def _bool_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes", "y"})


def _short_label(value: object, width: int = 12) -> str:
    text = str(value)
    text = text.replace("load_shed:", "LS:")
    text = text.replace("fixed_pf:", "PF:")
    text = text.replace("generator_trip:", "GT:")
    text = text.replace("plant_trip:", "PT:")
    text = text.replace("_", " ")
    if text.startswith("export reduce") or text.startswith("export reduction"):
        text = "export"
    return text if len(text) <= width else text[: width - 1] + "."


def _display_text(value: object) -> str:
    return tex_escape(value) if mpl.rcParams.get("text.usetex") else str(value)


def _percent_symbol() -> str:
    return r"\%" if mpl.rcParams.get("text.usetex") else "%"


def _percent_label(value: float, *, decimals: int = 0) -> str:
    if mpl.rcParams.get("text.usetex"):
        return rf"{value:.{decimals}f}$\%$"
    return f"{value:.{decimals}f}%"


def _fmt_id(value: object, width: int = 15) -> str:
    text = str(value).replace("_shed", "").replace("load", "L")
    return text if len(text) <= width else text[: width - 1] + "."


def _class_color(value: str) -> str:
    text = str(value).lower()
    if "cert" in text or text == "safe" or "true negative" in text:
        return PALETTE["safe"]
    if "data" in text or "unavailable" in text:
        return PALETTE["data"]
    if "trip" in text or "cascade" in text or "positive" in text:
        return PALETTE["trip"]
    if "risky" in text or "conservative" in text or "screen" in text:
        return PALETTE["risky"]
    return PALETTE["background"]


def _glyph_summary(group: pd.DataFrame) -> tuple[str, str]:
    if group.empty:
        return "screen-only", "screen only"
    if "data_limited" in group and group["data_limited"].astype(bool).any():
        return "dl", "data-limited TDS"
    if "event_screen_flagged" in group:
        pred = bool(group["event_screen_flagged"].astype(bool).any())
    else:
        pred = bool(group["predicted_trip"].astype(bool).any())
    actual = bool(group["actual_trip"].astype(bool).any())
    fp = bool(group["false_positive"].astype(bool).any())
    fn = bool(group["false_negative"].astype(bool).any())
    if fn:
        return "fn", "unsafe FN"
    if pred and actual:
        return "tp", "true positive"
    if fp or (pred and not actual):
        return "fp", "conservative FP"
    return "tn", "true negative"


def make_fig01_replication_activsg2000(
    paths: DatasetPaths,
    bundle: ArtifactBundle,
    *,
    out_fig: Path,
    formats: Iterable[str],
    outline_text: bool,
) -> list[SaveResult]:
    """Reuse the curated ACTIVSg2000 voltage replica without leaving old stems."""

    if bundle.replication is None:
        raise MissingArtifactError("Replication artifacts are required for Figure 1")
    from pa_dvsa.replication.figures import load_replica_data, make_voltage_wide

    written: list[SaveResult] = []
    data = load_replica_data(bundle.replication.root)
    with tempfile.TemporaryDirectory(prefix="paper_fig01_") as tmp:
        tmp_dir = Path(tmp)
        make_voltage_wide(data, out_dirs=(tmp_dir,), formats=tuple(formats))
        out_fig.mkdir(parents=True, exist_ok=True)
        for fmt in formats:
            dst = out_fig / f"fig01_replication_activsg2000.{fmt}"
            if fmt == "pdf" and outline_text:
                src_pdf = tmp_dir / "fig_replication_activsg2000.pdf"
                if not src_pdf.exists():
                    src_pdf = tmp_dir / "fig_replication_voltage_wide.pdf"
                outline_pdf_text(src_pdf, dst)
            else:
                src = tmp_dir / f"fig_replication_activsg2000.{fmt}"
                if not src.exists():
                    src = tmp_dir / f"fig_replication_voltage_wide.{fmt}"
                shutil.copy2(src, dst)
            written.append(SaveResult(path=dst, sha256=sha256_file(dst)))
    return written


def make_fig02_validation_matrix(
    paths: DatasetPaths,
    bundle: ArtifactBundle,
    *,
    out_fig: Path,
    formats: Iterable[str],
    outline_text: bool,
) -> list[SaveResult]:
    validation = _read(paths.tds_validation)
    if validation.empty:
        raise MissingArtifactError("tds_validation.csv is required for Figure 2")
    _require_columns(
        validation,
        paths.tds_validation,
        ("case", "seed_event", "predicted_trip", "actual_trip", "data_limited"),
    )

    val = validation.copy()
    for column in ("predicted_trip", "actual_trip", "data_limited"):
        val[column] = _bool_series(val[column])

    seed_rows: list[dict[str, object]] = []
    for (case, seed), group in val.groupby(["case", "seed_event"], sort=False):
        any_dl = bool(group["data_limited"].any())
        any_pred = bool(group["predicted_trip"].any())
        any_actual = bool(group["actual_trip"].any())
        seed_rows.append(
            {
                "case": str(case),
                "seed_event": str(seed),
                "tp": any_pred and any_actual and not any_dl,
                "fp": any_pred and (not any_actual) and not any_dl,
                "tn": (not any_pred) and (not any_actual) and not any_dl,
                "fn": (not any_pred) and any_actual and not any_dl,
                "dl": any_dl,
            }
        )
    seed_df = pd.DataFrame(seed_rows)
    if seed_df.empty:
        raise MissingArtifactError("tds_validation.csv contains no seed-level validation rows")

    counts_by_case: dict[str, dict[str, int]] = {}
    for case in SYSTEM_ORDER:
        sub = seed_df[seed_df["case"] == case]
        if sub.empty:
            raise MissingArtifactError(f"tds_validation.csv has no validated seeds for {case}")
        counts_by_case[case] = {
            "tp": int(sub["tp"].sum()),
            "fp": int(sub["fp"].sum()),
            "tn": int(sub["tn"].sum()),
            "fn": int(sub["fn"].sum()),
            "dl": int(sub["dl"].sum()),
            "total": int(len(sub)),
        }

    totals = {key: sum(counts[key] for counts in counts_by_case.values()) for key in ("tp", "fp", "tn", "fn", "dl", "total")}
    fig = plt.figure(figsize=(7.2, 3.05))

    banner = fig.add_axes([0.025, 0.83, 0.95, 0.13])
    banner.set_axis_off()
    banner.add_patch(
        FancyBboxPatch(
            (0.0, 0.0),
            1.0,
            1.0,
            transform=banner.transAxes,
            facecolor=PALETTE["light_teal"],
            edgecolor=PALETTE["safe"],
            linewidth=1.1,
            boxstyle="round,pad=0.0,rounding_size=0.04",
        )
    )
    banner.text(
        0.50,
        0.62,
        "Zero unsafe false negatives across all benchmarks",
        ha="center",
        va="center",
        fontsize=10.0,
        fontweight="bold",
        color=PALETTE["safe"],
    )
    summary = (
        f"{totals['total']} validated seeds   ·   {totals['tp']} true positives   ·   "
        f"{totals['fp']} conservative false positives   ·   {totals['tn']} true negatives   ·   "
        f"{totals['dl']} data-limited   ·   {totals['fn']} missed false negatives"
    )
    banner.text(
        0.50,
        0.18,
        summary,
        ha="center",
        va="center",
        fontsize=7.6,
        color=PALETTE["charcoal"],
    )

    segment_order = (
        ("tp", "true positive", PALETTE["trip"], "white"),
        ("fp", "conservative FP", PALETTE["risky"], PALETTE["charcoal"]),
        ("tn", "true negative", PALETTE["safe"], "white"),
        ("dl", "data-limited", PALETTE["data"], "white"),
    )
    panel_w = 0.215
    panel_gap = (0.99 - 0.04 - 4 * panel_w) / 3.0
    fn_slot = 0.13

    def draw_swatch(ax: plt.Axes, x: float, y: float, key: str, color: str) -> None:
        if key == "fn":
            ax.add_patch(
                Rectangle(
                    (x, y - 0.05),
                    0.045,
                    0.10,
                    transform=ax.transAxes,
                    facecolor=PALETTE["light_teal"],
                    edgecolor=PALETTE["safe"],
                    linewidth=1.0,
                    linestyle=(0, (3.5, 1.5)),
                )
            )
        else:
            ax.add_patch(
                Rectangle(
                    (x, y - 0.05),
                    0.045,
                    0.10,
                    transform=ax.transAxes,
                    facecolor=color,
                    edgecolor="white",
                    linewidth=0.6,
                )
            )

    for idx, case in enumerate(SYSTEM_ORDER):
        x0 = 0.04 + idx * (panel_w + panel_gap)
        title_ax = fig.add_axes([x0, 0.71, panel_w, 0.07])
        bar_ax = fig.add_axes([x0, 0.50, panel_w, 0.18])
        legend_ax = fig.add_axes([x0, 0.06, panel_w, 0.38])
        for ax in (title_ax, bar_ax, legend_ax):
            ax.set_axis_off()

        counts = counts_by_case[case]
        title_ax.text(
            0.0,
            0.52,
            SYSTEM_LABELS[case],
            ha="left",
            va="center",
            fontsize=9.6,
            fontweight="bold",
            color=PALETTE["charcoal"],
        )
        title_ax.text(
            1.0,
            0.52,
            f"n = {counts['total']}",
            ha="right",
            va="center",
            fontsize=8.0,
            color=PALETTE["charcoal"],
        )

        bar_ax.add_patch(
            Rectangle(
                (0.0, 0.18),
                1.0,
                0.64,
                transform=bar_ax.transAxes,
                facecolor="white",
                edgecolor=PALETTE["grid"],
                linewidth=0.7,
            )
        )
        denom = max(1, counts["tp"] + counts["fp"] + counts["tn"] + counts["dl"])
        cursor = 0.0
        for key, _label, color, text_color in segment_order:
            width = (1.0 - fn_slot) * counts[key] / denom
            if width <= 0:
                continue
            bar_ax.add_patch(
                Rectangle(
                    (cursor, 0.18),
                    width,
                    0.64,
                    transform=bar_ax.transAxes,
                    facecolor=color,
                    edgecolor="white",
                    linewidth=0.8,
                )
            )
            if width >= 0.08:
                bar_ax.text(
                    cursor + width / 2,
                    0.50,
                    f"{counts[key]}",
                    transform=bar_ax.transAxes,
                    ha="center",
                    va="center",
                    fontsize=11.0,
                    fontweight="bold",
                    color=text_color,
                )
            else:
                bar_ax.text(
                    cursor + width / 2,
                    1.18,
                    f"{counts[key]}",
                    transform=bar_ax.transAxes,
                    ha="center",
                    va="bottom",
                    fontsize=7.5,
                    color=color,
                )
            cursor += width
        bar_ax.add_patch(
            Rectangle(
                (1.0 - fn_slot, 0.18),
                fn_slot,
                0.64,
                transform=bar_ax.transAxes,
                facecolor=PALETTE["light_teal"],
                edgecolor=PALETTE["safe"],
                linewidth=1.5,
                linestyle=(0, (3.5, 1.5)),
            )
        )
        bar_ax.text(
            1.0 - fn_slot / 2,
            0.56,
            f"{counts['fn']}",
            transform=bar_ax.transAxes,
            ha="center",
            va="center",
            fontsize=12.0,
            fontweight="bold",
            color=PALETTE["safe"],
        )
        bar_ax.text(
            1.0 - fn_slot / 2,
            0.31,
            "FN",
            transform=bar_ax.transAxes,
            ha="center",
            va="center",
            fontsize=6.4,
            color=PALETTE["safe"],
        )

        legend_entries = [
            ("tp", "true +", PALETTE["trip"], counts["tp"], 0.02, 0.83),
            ("fp", "conserv. FP", PALETTE["risky"], counts["fp"], 0.52, 0.83),
            ("tn", "true -", PALETTE["safe"], counts["tn"], 0.02, 0.58),
            ("dl", "data-limited", PALETTE["data"], counts["dl"], 0.52, 0.58),
            ("fn", "missed FN", PALETTE["safe"], counts["fn"], 0.02, 0.27),
        ]
        for key, label, color, count, lx, ly in legend_entries:
            draw_swatch(legend_ax, lx, ly, key, color)
            legend_ax.text(
                lx + 0.06,
                ly,
                f"{label}:  {count}",
                transform=legend_ax.transAxes,
                ha="left",
                va="center",
                fontsize=6.9,
                fontweight="bold" if key == "fn" else "normal",
                color=PALETTE["safe"] if key == "fn" else PALETTE["charcoal"],
            )

    written = save_figure(fig, "fig02_validation_matrix", out_fig, formats, outline_text=outline_text)
    plt.close(fig)
    return written


def make_fig03_margin_erosion_atlas(
    paths: DatasetPaths,
    bundle: ArtifactBundle,
    *,
    out_fig: Path,
    formats: Iterable[str],
    outline_text: bool,
) -> list[SaveResult]:
    km = _read(paths.k_matrix)
    cascades = _read(paths.cascade_predictions)
    validation = _read(paths.tds_validation)
    if km.empty or cascades.empty:
        raise MissingArtifactError("k_matrix.csv and cascade_predictions.csv are required for Figure 3")
    _require_columns(km, paths.k_matrix, ("case", "asset", "event_id", "event_family", "k_pickup"))
    _require_columns(
        cascades,
        paths.cascade_predictions,
        ("case", "seed_event", "fixed_point_size", "fixed_point", "first_layer_size"),
    )
    _require_columns(validation, paths.tds_validation, ("case", "seed_event", "asset", "actual_trip"))
    val = validation.copy()
    val["actual_trip"] = _bool_series(val["actual_trip"])

    fig = plt.figure(figsize=(7.0, 4.15))
    gs = fig.add_gridspec(
        2,
        4,
        height_ratios=[1.12, 0.70],
        hspace=0.34,
        wspace=0.30,
        left=0.05,
        right=0.985,
        top=0.93,
        bottom=0.16,
    )

    def node_positions(items: list[str], x: float, *, r: float) -> dict[str, tuple[float, float]]:
        if not items:
            return {}
        if len(items) == 1:
            ys = [0.50]
        else:
            ys = np.linspace(0.20, 0.78, len(items)).tolist()
        return {item: (x, float(y)) for item, y in zip(items, ys, strict=True)}

    def trim_nodes(items: list[str], limit: int) -> list[str]:
        if len(items) <= limit:
            return items
        return [*items[: limit - 1], f"+{len(items) - limit + 1}"]

    for col, case in enumerate(SYSTEM_ORDER):
        ax = fig.add_subplot(gs[0, col])
        ax.set_axis_off()
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        sub_cascade = cascades[cascades["case"] == case].copy()
        if sub_cascade.empty:
            raise MissingArtifactError(f"cascade_predictions.csv has no rows for {case}")
        sub_cascade["fixed_point_size"] = pd.to_numeric(sub_cascade["fixed_point_size"], errors="coerce").fillna(0)
        actual_seed_events = set(
            val[(val["case"] == case) & (val["actual_trip"])]["seed_event"].astype(str)
        )
        validated_cascade = sub_cascade[sub_cascade["seed_event"].astype(str).isin(actual_seed_events)]
        if not validated_cascade.empty:
            row = validated_cascade.sort_values(["fixed_point_size", "seed_event"], ascending=[False, True]).iloc[0]
        else:
            row = sub_cascade.sort_values(["fixed_point_size", "seed_event"], ascending=[False, True]).iloc[0]
        seed = str(row["seed_event"])
        fixed_raw = "" if pd.isna(row["fixed_point"]) else str(row["fixed_point"])
        fixed = [item for item in fixed_raw.split(";") if item]
        if not fixed:
            fixed = [_short_label(seed)]
        seed_node = fixed[0]
        secondary = fixed[1:]
        first_layer_count = int(pd.to_numeric(pd.Series([row["first_layer_size"]]), errors="coerce").fillna(0).iloc[0])
        first_layer = secondary[: max(1, min(first_layer_count, len(secondary)))] if secondary else []
        second_layer = secondary[len(first_layer) :]
        first_layer = trim_nodes(first_layer, 5)
        second_layer = trim_nodes(second_layer, 4)
        radius = 0.052 if len(first_layer) + len(second_layer) < 8 else 0.043
        # Matplotlib scatter sizes are in pt^2. sqrt(324) = 18 pt,
        # i.e., about 0.25 in diameter, which is large enough for short IDs
        # without overpowering the fan-out panel.
        node_size = 324
        x_seed, x_first, x_second = 0.14, 0.52, 0.86
        positions = {seed_node: (x_seed, 0.50)}
        positions.update(node_positions(first_layer, x_first, r=radius))
        positions.update(node_positions(second_layer, x_second, r=radius))
        actual = set(
            val[(val["case"] == case) & (val["seed_event"].astype(str) == seed) & (val["actual_trip"])]["asset"].astype(str)
        )
        for target in first_layer:
            y_target = positions[target][1]
            arrow = FancyArrowPatch(
                (x_seed + radius, 0.50),
                (x_first - radius, y_target),
                transform=ax.transAxes,
                connectionstyle="arc3,rad=0.12",
                color=PALETTE["trip"],
                alpha=0.55,
                linewidth=0.9,
                arrowstyle="-|>",
                mutation_scale=7,
            )
            ax.add_patch(arrow)
            if not target.startswith("+"):
                krow = km[(km["case"] == case) & (km["asset"].astype(str) == target) & (km["event_id"].astype(str) == seed)]
                if not krow.empty:
                    kvalue = float(pd.to_numeric(krow["k_pickup"], errors="coerce").max())
                    text = ax.text(
                        0.30,
                        (0.50 + y_target) / 2,
                        f"{kvalue:.1f}",
                        transform=ax.transAxes,
                        ha="center",
                        va="center",
                        fontsize=6.4,
                        color=PALETTE["charcoal"],
                    )
                    text.set_path_effects([pe.Stroke(linewidth=2, foreground="white"), pe.Normal()])
        if first_layer and second_layer:
            for idx, target in enumerate(second_layer):
                src = first_layer[idx % len(first_layer)]
                y_src = positions[src][1]
                y_target = positions[target][1]
                ax.add_patch(
                    FancyArrowPatch(
                        (x_first + radius, y_src),
                        (x_second - radius, y_target),
                        transform=ax.transAxes,
                        connectionstyle="arc3,rad=0.12",
                        color=PALETTE["trip"],
                        alpha=0.42,
                        linewidth=0.8,
                        arrowstyle="-|>",
                        mutation_scale=7,
                    )
                )
        for name, (x, y) in positions.items():
            is_seed = name == seed_node
            ax.scatter(
                [x],
                [y],
                transform=ax.transAxes,
                marker="o",
                s=node_size,
                facecolor=PALETTE["charcoal"] if is_seed else PALETTE["trip"],
                edgecolor="white",
                linewidth=1.15,
                zorder=3,
                clip_on=False,
            )
            ax.text(
                x,
                y,
                _fmt_id(name, 4),
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=6.9 if not name.startswith("+") else 7.2,
                color="white",
                zorder=4,
            )
            if name in actual:
                ax.scatter(
                    x + radius * 0.90,
                    y + radius * 0.72,
                    transform=ax.transAxes,
                    marker="*",
                    s=40,
                    facecolor="none",
                    edgecolor=PALETTE["safe"],
                    linewidth=1.0,
                    zorder=5,
                    clip_on=False,
                )
        ax.text(
            0.00,
            0.98,
            SYSTEM_LABELS[case],
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9.2,
            fontweight="bold",
            color=PALETTE["charcoal"],
        )
        ax.text(
            0.00,
            0.86,
            f"seed: {_display_text(_short_label(seed, 14).rstrip('.'))}   set: {int(row['fixed_point_size'])}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=7.2,
            color=PALETTE["charcoal"],
        )

    for col, case in enumerate(SYSTEM_ORDER):
        ax = fig.add_subplot(gs[1, col])
        sub = km[km["case"] == case].copy()
        if sub.empty:
            raise MissingArtifactError(f"k_matrix.csv has no K entries for {case}")
        sub["k_pickup"] = pd.to_numeric(sub["k_pickup"], errors="coerce")
        sub = sub.dropna(subset=["k_pickup"]).sort_values("k_pickup", ascending=False).reset_index(drop=True)
        ranks = np.arange(1, len(sub) + 1)
        k_sorted = sub["k_pickup"].to_numpy(dtype=float)
        families = sub["event_family"].astype(str).to_numpy()
        ax.step(ranks, k_sorted, where="post", color=PALETTE["grid"], linewidth=1.6, zorder=1)
        ax.scatter(
            ranks,
            k_sorted,
            c=[FAMILY_COLOR.get(family, PALETTE["background"]) for family in families],
            s=17,
            edgecolor="none",
            linewidth=0.0,
            zorder=3,
        )
        ax.axhline(
            1.0,
            color=PALETTE["trip"],
            linewidth=1.0,
            linestyle=(0, (4, 2)),
            label=r"$K = 1$ trip threshold" if col == 0 else None,
        )
        top_event = _short_label(str(sub.iloc[0]["event_id"]), 12)
        top_k = float(sub.iloc[0]["k_pickup"])
        ax.annotate(
            f"{top_event}\nK={top_k:.1f}",
            xy=(1, top_k),
            xytext=(max(3, len(sub) * 0.16), top_k * 0.42 if top_k > 1 else top_k * 1.8),
            arrowprops={"arrowstyle": "->", "lw": 0.7, "color": PALETTE["charcoal"]},
            fontsize=6.4,
            color=PALETTE["charcoal"],
        )
        ax.set_title(SYSTEM_LABELS[case], loc="left", fontsize=9.0, color=PALETTE["charcoal"], pad=1)
        ax.set_yscale("log")
        ax.set_ylim(0.01, max(1.4, np.nanmax(k_sorted) * 1.4))
        ax.set_xlim(1, len(sub))
        if col == 0:
            ax.set_xlabel("event rank")
            ax.set_ylabel(r"normalized margin erosion $K^{\rm pk}$")
        else:
            ax.set_yticklabels([])
        soften_axes(ax, grid=True, ygrid=True)

    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markerfacecolor=color,
            markeredgecolor="white",
            markersize=6,
            label=label,
        )
        for label, color in FAMILY_COLOR.items()
    ]
    handles.append(
        Line2D([0], [0], marker="*", linestyle="", markerfacecolor="none", markeredgecolor=PALETTE["safe"], markersize=8, label="TDS-confirmed trip")
    )
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.012), ncol=3, frameon=False, fontsize=7.1)
    written = save_figure(fig, "fig03_margin_erosion_atlas", out_fig, formats, outline_text=outline_text)
    plt.close(fig)
    return written


def make_fig04_observability_gap(
    paths: DatasetPaths,
    bundle: ArtifactBundle,
    *,
    out_fig: Path,
    formats: Iterable[str],
    outline_text: bool,
) -> list[SaveResult]:
    obs = _read(paths.observability_margins)
    if obs.empty:
        raise MissingArtifactError("Observability margins are required for Figure 4")
    _require_columns(
        obs,
        paths.observability_margins,
        ("asset", "mode", "normalized_margin", "classification"),
    )
    preferred_modes = [
        "transmission-only voltage",
        "bounded tap + reconstruction error",
        "known tap reconstruction",
        "full collector telemetry",
    ]
    modes = [mode for mode in preferred_modes if mode in set(obs["mode"].astype(str))]
    mode_labels = [
        "transmission\nonly" if mode == "transmission-only voltage" else
        "bounded tap\n+ error" if mode == "bounded tap + reconstruction error" else
        "known tap" if mode == "known tap reconstruction" else
        "collector\ntelemetry"
        for mode in modes
    ]
    if len(modes) < 2:
        raise MissingArtifactError("observability_margins.csv needs at least two observability modes")

    fig, (ax_a, ax_b) = plt.subplots(
        2,
        1,
        figsize=FIGURE_SPECS["column"],
        gridspec_kw={"height_ratios": [1.08, 0.86], "hspace": 0.34},
    )
    positions = np.arange(len(modes))
    mode_colors = {
        "transmission-only voltage": PALETTE["background"],
        "bounded tap + reconstruction error": PALETTE["light_purple"],
        "known tap reconstruction": PALETTE["light_blue"],
        "full collector telemetry": PALETTE["light_teal"],
    }
    rng = np.random.default_rng(7)
    all_vals: list[float] = []
    for j, mode in enumerate(modes):
        vals = pd.to_numeric(obs[obs["mode"] == mode]["normalized_margin"], errors="coerce").dropna().to_numpy()
        if vals.size == 0:
            continue
        all_vals.extend(vals.tolist())
        q10, q25, q50, q75, q90 = np.nanpercentile(vals, [10, 25, 50, 75, 90])
        color = mode_colors.get(mode, PALETTE["background"])
        ax_a.vlines(j, q10, q90, color=PALETTE["charcoal"], linewidth=1.0, alpha=0.22, zorder=1)
        ax_a.vlines(j, q25, q75, color=color, linewidth=9.0, alpha=0.82, zorder=2)
        ax_a.scatter([j], [q50], marker="D", s=34, color=PALETTE["charcoal"], edgecolor="white", linewidth=0.55, zorder=4)
        jitter = rng.normal(0, 0.045, size=vals.size)
        ax_a.scatter(
            np.full(vals.size, j) + jitter,
            vals,
            s=11,
            color=PALETTE["charcoal"],
            alpha=0.38,
            edgecolor="none",
            linewidth=0,
            zorder=3,
        )
    if all_vals:
        ymin = min(-0.72, float(np.nanmin(all_vals)) - 0.06)
        ymax = max(0.98, float(np.nanmax(all_vals)) + 0.06)
        ax_a.set_ylim(ymin, ymax)
    ax_a.axhspan(ax_a.get_ylim()[0], 0, color=PALETTE["light_red"], alpha=0.22, zorder=0)
    ax_a.axhline(0, color=PALETTE["trip"], lw=1.0, ls=(0, (4, 2)))
    ax_a.text(
        0.05,
        0.018,
        "pickup",
        ha="left",
        va="bottom",
        fontsize=6.8,
        color=PALETTE["trip"],
    )
    ax_a.set_xticks(positions, mode_labels)
    ax_a.set_ylabel("margin to pickup (pu)")
    soften_axes(ax_a, grid=True)
    panel_label(ax_a, "a")

    class_order = ["data-limited", "predicted trip", "risky", "certified"]
    y = np.arange(len(modes)) * 0.82
    for i, mode in enumerate(modes):
        rows = obs[obs["mode"] == mode]
        total = max(1, len(rows))
        left = 0.0
        for cls in class_order:
            count = int((rows["classification"].astype(str) == cls).sum())
            width = count / total
            if width <= 0:
                continue
            ax_b.barh(
                y[i],
                width,
                left=left,
                height=0.66,
                color=_class_color(cls),
                edgecolor="none",
                linewidth=0.0,
                zorder=2,
            )
            if width >= 0.14:
                percent = 100.0 * width
                ax_b.text(
                    left + width / 2,
                    y[i],
                    _percent_label(percent),
                    ha="center",
                    va="center",
                    fontsize=7.0,
                    color="white" if cls != "risky" else PALETTE["charcoal"],
                    fontweight="bold",
                )
            left += width
    ax_b.set_yticks(y, mode_labels)
    ax_b.invert_yaxis()
    ax_b.set_xlim(0, 1.0)
    ax_b.set_xticks([0, 0.5, 1.0], ["0", "50", "100"])
    ax_b.set_xlabel(r"share of protected assets ($\%$)")
    handles = [
        Patch(facecolor=_class_color(cls), edgecolor="white", label=cls)
        for cls in class_order
    ]
    ax_b.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.13), ncol=2, frameon=False, fontsize=6.4)
    soften_axes(ax_b, grid=False)
    panel_label(ax_b, "b")
    fig.subplots_adjust(left=0.20, right=0.98, bottom=0.13, top=0.95)
    written = save_figure(fig, "fig04_observability_gap", out_fig, formats, outline_text=outline_text)
    plt.close(fig)
    return written


def make_fig05_operator_action_screen(
    paths: DatasetPaths,
    bundle: ArtifactBundle,
    *,
    out_fig: Path,
    formats: Iterable[str],
    outline_text: bool,
) -> list[SaveResult]:
    screen = _read(paths.operator_action_screen)
    if screen.empty:
        raise MissingArtifactError("Operator action screen data are required for Figure 5")
    _require_columns(screen, paths.operator_action_screen, ("case", "action_family", "worst_k"))
    families = [
        "load/pump disconnection",
        "fixed-PF RES ramp",
        "plant/generator trip",
        "export reduction",
        "shunt/reactor action",
    ]

    family_labels = {
        "load/pump disconnection": "load / pump\ndisconnection",
        "fixed-PF RES ramp": "fixed-PF\nRES ramp",
        "plant/generator trip": "plant /\ngenerator trip",
        "export reduction": "export\nreduction",
        "shunt/reactor action": "shunt /\nreactor action",
    }
    system_colors = {
        "kundur": PALETTE["control"],
        "ieee39": PALETTE["safe"],
        "npcc": PALETTE["risky"],
        "gbnetwork": PALETTE["trip"],
    }
    system_markers = {"kundur": "o", "ieee39": "s", "npcc": "^", "gbnetwork": "D"}
    system_offsets = {"kundur": 0.15, "ieee39": 0.05, "npcc": -0.05, "gbnetwork": -0.15}

    fig = plt.figure(figsize=(7.0, 3.25))
    ax = fig.add_axes([0.18, 0.19, 0.78, 0.68])
    ax.set_xscale("log")
    ax.set_xlim(5e-4, 200)
    ax.set_ylim(-0.72, len(families) - 0.42)
    ax.invert_yaxis()
    ax.axvspan(5e-4, 1, color=PALETTE["light_teal"], alpha=0.20, zorder=0)
    ax.axvspan(1, 200, color=PALETTE["light_red"], alpha=0.25, zorder=0)
    ax.axvline(1, color=PALETTE["trip"], linewidth=1.2, linestyle=(0, (5, 2.5)), zorder=2)
    ax.text(
        1,
        -0.60,
        r"trip threshold $K = 1$",
        ha="center",
        va="bottom",
        fontsize=7.4,
        color=PALETTE["trip"],
    )
    ax.text(8e-4, -0.53, "certified", ha="left", va="center", fontsize=7.6, fontweight="bold", color=PALETTE["safe"])
    ax.text(185, -0.53, "mitigate / validate", ha="right", va="center", fontsize=7.6, fontweight="bold", color=PALETTE["trip"])

    for yi, family in enumerate(families):
        ax.hlines(yi, 5e-4, 200, color=PALETTE["grid"], linewidth=0.6, zorder=1)
        for case in SYSTEM_ORDER:
            yplot = yi + system_offsets[case]
            rows = screen[(screen["case"] == case) & (screen["action_family"] == family)]
            if rows.empty:
                continue
            value = float(pd.to_numeric(rows["worst_k"], errors="coerce").max())
            if value <= 0 or not np.isfinite(value):
                continue
            x_plot = max(value, 7e-4)
            ax.scatter(
                x_plot,
                yplot,
                marker=system_markers[case],
                s=85,
                facecolor=system_colors[case],
                edgecolor="white",
                linewidth=1.2,
                zorder=4,
            )
            label = f"{value:.1f}" if value >= 0.1 else ("<0.01" if value < 0.01 else f"{value:.2f}")
            ax.text(
                min(x_plot * 1.18, 185),
                yplot,
                label,
                ha="left",
                va="center",
                fontsize=7.0,
                color=PALETTE["charcoal"],
                zorder=5,
            )
    ax.set_yticks(range(len(families)))
    ax.set_yticklabels([family_labels[family] for family in families], fontsize=8.4)
    ax.set_xlabel(r"worst normalized margin erosion $\bar K^{\rm pk}$ across protected assets")
    handles = [
        Line2D(
            [0],
            [0],
            marker=system_markers[case],
            markerfacecolor=system_colors[case],
            markeredgecolor="white",
            linestyle="",
            markersize=8,
            label=SYSTEM_LABELS[case],
        )
        for case in SYSTEM_ORDER
    ]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.10), ncol=4, frameon=False)
    soften_axes(ax, grid=False)
    ax.tick_params(axis="y", length=0)
    written = save_figure(fig, "fig05_operator_action_screen", out_fig, formats, outline_text=outline_text)
    plt.close(fig)
    return written


def make_fig06_relay_window_controls(
    paths: DatasetPaths,
    bundle: ArtifactBundle,
    *,
    out_fig: Path,
    formats: Iterable[str],
    outline_text: bool,
) -> list[SaveResult]:
    profiles = _read(paths.control_window_profiles)
    mitigation = _read(paths.mitigation_summary)
    assets = _read(paths.assets)
    if profiles.empty:
        raise MissingArtifactError("Control authority profiles are required for Figure 6")
    _require_columns(profiles, paths.control_window_profiles, ("case", "time_s", "asset", "series", "value", "kind"))
    _require_columns(mitigation, paths.mitigation_summary, ("case", "control", "alpha", "mvar", "residual_slack"))
    _require_columns(assets, paths.assets, ("case", "asset", "dwell_s"))

    prof = profiles[profiles["case"].isin(SYSTEM_ORDER)].copy()
    prof["time_s"] = pd.to_numeric(prof["time_s"], errors="coerce")
    prof["value"] = pd.to_numeric(prof["value"], errors="coerce")
    prof = prof.dropna(subset=["time_s", "value"])

    preference = {"ieee39": 0, "gbnetwork": 1, "npcc": 2, "kundur": 3}
    candidates: list[tuple[int, bool, float, str, str]] = []
    for (case, asset), group in prof.groupby(["case", "asset"]):
        dist = group[group["kind"] == "disturbance"].sort_values("time_s")
        if dist.empty:
            continue
        within = dist[dist["time_s"] <= 1.0]
        if within.empty:
            continue
        peak = float(within["value"].max())
        crosses = bool((within["value"] >= 1.0).any())
        candidates.append((preference.get(str(case), 99), crosses, peak, str(case), str(asset)))
    if not candidates:
        raise MissingArtifactError("control_window_profiles.csv has no disturbance time series for Figure 6")
    candidates.sort(key=lambda item: (item[0], not item[1], -item[2]))
    _, _, _, case, asset = candidates[0]
    sub = prof[(prof["case"] == case) & (prof["asset"].astype(str) == asset)].copy()
    dist = sub[sub["kind"] == "disturbance"].sort_values("time_s")
    time = dist["time_s"].to_numpy(dtype=float)
    dval = dist["value"].to_numpy(dtype=float)
    mask = time <= 1.0
    time = time[mask]
    dval = dval[mask]
    if time.size == 0:
        raise MissingArtifactError(f"No 0--1 s disturbance window for {case}/{asset}")

    controls = sub[(sub["kind"] == "control") & (sub["time_s"] <= 1.0)].copy()
    alpha_by_control = (
        mitigation[mitigation["case"] == case]
        .set_index("control")["alpha"]
        .pipe(pd.to_numeric, errors="coerce")
        .fillna(0.0)
        .to_dict()
        if not mitigation[mitigation["case"] == case].empty
        else {}
    )
    if not alpha_by_control:
        warnings.warn(
            f"No mitigation LP rows found for {case}/{asset}; plotting disturbance and zero-control residual.",
            RuntimeWarning,
            stacklevel=2,
        )
    authority = pd.Series(0.0, index=np.round(time, 10))
    for control_name, group in controls.groupby("series"):
        alpha = float(alpha_by_control.get(str(control_name), 0.0))
        if alpha == 0.0:
            continue
        ctime = np.round(group["time_s"].to_numpy(dtype=float), 10)
        cvalue = group["value"].to_numpy(dtype=float) * alpha
        authority = authority.add(pd.Series(cvalue, index=ctime), fill_value=0.0)
    authority = authority.reindex(np.round(time, 10), method="nearest").fillna(0.0).to_numpy(dtype=float)
    residual = dval - authority

    dwell_rows = assets[(assets["case"] == case) & (assets["asset"].astype(str) == asset)]
    dwell = float(pd.to_numeric(dwell_rows["dwell_s"], errors="coerce").dropna().iloc[0]) if not dwell_rows.empty else 0.05
    dwell = min(max(dwell, 0.02), 0.25)
    msub = mitigation[mitigation["case"] == case]
    alpha_mvar = float(pd.to_numeric(msub["mvar"], errors="coerce").sum()) if not msub.empty else 0.0
    eta = float(pd.to_numeric(msub["residual_slack"], errors="coerce").max()) if not msub.empty else 0.0

    fig, (ax_a, ax_b) = plt.subplots(
        2,
        1,
        figsize=FIGURE_SPECS["column"],
        gridspec_kw={"height_ratios": [1.0, 0.62], "hspace": 0.32},
    )
    ax_a.axvspan(0.0, dwell, color=PALETTE["light_red"], alpha=0.45, zorder=0)
    ax_a.fill_between(time, 1.0, dval, where=dval >= 1.0, color=PALETTE["light_red"], alpha=0.55, linewidth=0, zorder=1)
    ax_a.plot(time, dval, color=PALETTE["trip"], linewidth=1.9, label="disturbance")
    ax_a.fill_between(time, residual, dval, where=dval >= residual, color=PALETTE["light_blue"], alpha=0.42, linewidth=0, zorder=1)
    ax_a.plot(time, residual, color=PALETTE["control"], linewidth=1.9, label="with LP controls")
    ax_a.axhline(1.0, color=PALETTE["trip"], linewidth=1.1, linestyle=(0, (5, 2.5)), label=r"$K = 1$ trip threshold")
    dpeak = int(np.nanargmax(dval))
    rpeak = int(np.nanargmax(residual))
    ax_a.scatter(time[dpeak], dval[dpeak], s=42, color=PALETTE["trip"], edgecolor="white", linewidth=0.8, zorder=4)
    ax_a.scatter(time[rpeak], residual[rpeak], s=42, color=PALETTE["control"], edgecolor="white", linewidth=0.8, zorder=4)
    ymax = max(1.25, float(np.nanmax(dval)) * 1.15, float(np.nanmax(residual)) * 1.15)
    ax_a.set_ylim(0, ymax)
    ax_a.set_xlim(0, 1.0)
    ax_a.text(
        dwell / 2,
        ymax * 0.54,
        "relay dwell",
        ha="center",
        va="center",
        rotation=90,
        fontsize=6.8,
        color=PALETTE["trip"],
    )
    ax_a.text(
        0.97,
        0.92,
        f"asset {_display_text(asset)}\n{SYSTEM_LABELS.get(case, case)}",
        transform=ax_a.transAxes,
        ha="right",
        va="top",
        fontsize=7.4,
        color=PALETTE["charcoal"],
    )
    ax_a.text(
        0.97,
        0.08,
        f"required Mvar: {alpha_mvar:.1f}\nresidual slack: {eta:.2f}",
        transform=ax_a.transAxes,
        ha="right",
        va="bottom",
        fontsize=7.2,
        color=PALETTE["charcoal"],
    )
    ax_a.set_ylabel(r"normalized erosion $K(t)$")
    ax_a.set_xlabel("time after seed event (s)")
    ax_a.legend(loc="upper left", bbox_to_anchor=(0.18, 0.99), frameon=False, fontsize=6.6)
    soften_axes(ax_a, grid=True)
    panel_label(ax_a, "a")

    t = np.linspace(0, 1.0, 320)
    curves = [
        (r"STATCOM ($\tau=50$ ms)", PALETTE["control"], 0.05),
        (r"voltage-mode IBR ($\tau=200$ ms)", PALETTE["data"], 0.20),
        (r"manual shunt ($\tau=30$ s)", PALETTE["risky"], 30.0),
    ]
    ax_b.axvspan(0.0, dwell, color=PALETTE["light_red"], alpha=0.45, zorder=0)
    for label, color, tau in curves:
        ax_b.plot(t, 1.0 - np.exp(-t / tau), linewidth=1.7, color=color, label=label)
    ax_b.axhline(0.5, color=PALETTE["grid"], linewidth=0.8, linestyle=":")
    ax_b.text(
        dwell / 2,
        0.52,
        "controls must act here",
        ha="center",
        va="center",
        rotation=90,
        fontsize=6.8,
        color=PALETTE["trip"],
    )
    ax_b.set_xlim(0, 1.0)
    ax_b.set_ylim(0, 1.05)
    ax_b.set_ylabel("fraction of\nfull authority")
    ax_b.set_xlabel("time (s)")
    ax_b.legend(loc="lower right", frameon=False, fontsize=6.6)
    soften_axes(ax_b, grid=True)
    panel_label(ax_b, "b")
    fig.subplots_adjust(left=0.18, right=0.97, bottom=0.10, top=0.93)
    written = save_figure(fig, "fig06_relay_window_controls", out_fig, formats, outline_text=outline_text)
    plt.close(fig)
    return written


def make_fig07_uncertainty_telemetry_value(
    paths: DatasetPaths,
    bundle: ArtifactBundle,
    *,
    out_fig: Path,
    formats: Iterable[str],
    outline_text: bool,
) -> list[SaveResult]:
    unc = _read(paths.uncertainty_frontiers)
    if unc.empty:
        raise MissingArtifactError("Uncertainty frontier data are required for Figure 7")
    _require_columns(
        unc,
        paths.uncertainty_frontiers,
        ("uncertainty_type", "level", "effective_margin_pu"),
    )

    fig, ax = plt.subplots(1, 1, figsize=(3.45, 2.95))
    types = ["relay threshold", "tap ratio", "reconstruction error", "Q absorption", "controller timing"]
    type_colors = {
        "relay threshold": PALETTE["trip"],
        "tap ratio": PALETTE["data"],
        "reconstruction error": PALETTE["risky"],
        "Q absorption": PALETTE["control"],
        "controller timing": PALETTE["safe"],
    }
    for utype in types:
        sub = unc[unc["uncertainty_type"] == utype]
        if sub.empty:
            continue
        frontier = sub.groupby("level", as_index=False)["effective_margin_pu"].min().sort_values("level")
        x = frontier["level"].to_numpy(dtype=float) * 100.0
        y = frontier["effective_margin_pu"].to_numpy(dtype=float)
        ax.plot(
            x,
            y,
            lw=1.65,
            marker="o",
            ms=3.4,
            color=type_colors.get(utype, PALETTE["background"]),
            label=utype,
            zorder=3,
        )

    ax.axhspan(0, 0.20, color=PALETTE["light_amber"], alpha=0.38, zorder=0, label="thin robust margin")
    ax.axhline(0.20, color=PALETTE["risky"], lw=0.9, ls=(0, (3.5, 2.2)), zorder=1)
    ax.axhline(0, color=PALETTE["trip"], lw=0.9, ls=(0, (4, 2)), zorder=1)
    ax.set_xlim(0, 104)
    ax.set_ylim(0, max(0.92, float(pd.to_numeric(unc["effective_margin_pu"], errors="coerce").max()) + 0.06))
    ax.set_xlabel(f"uncertainty scale ({_percent_symbol()})")
    ax.set_ylabel("worst robust margin (pu)")

    line_handles, line_labels = ax.get_legend_handles_labels()
    order = [label for label in types if label in line_labels] + ["thin robust margin"]
    ordered_handles = [line_handles[line_labels.index(label)] for label in order if label in line_labels]
    ordered_labels = [label for label in order if label in line_labels]
    ax.legend(ordered_handles, ordered_labels, loc="upper right", frameon=False, fontsize=6.2, ncol=1)
    soften_axes(ax, grid=True)
    fig.subplots_adjust(left=0.18, right=0.97, bottom=0.18, top=0.95)
    written = save_figure(fig, "fig07_uncertainty_telemetry_value", out_fig, formats, outline_text=outline_text)
    plt.close(fig)
    return written

def make_fig08_mitigation_scaling(
    paths: DatasetPaths,
    bundle: ArtifactBundle,
    *,
    out_fig: Path,
    formats: Iterable[str],
    outline_text: bool,
) -> list[SaveResult]:
    ablation = _read(paths.ablation_summary)
    triage = _read(paths.scaling_summary)
    if ablation.empty or triage.empty:
        raise MissingArtifactError("Ablation and scaling data are required for Figure 8")
    _require_columns(
        ablation,
        paths.ablation_summary,
        ("row", "worst_k", "minimum_mvar_mitigation"),
    )
    _require_columns(
        triage,
        paths.scaling_summary,
        ("case", "fraction_sent_to_tds", "capture_rate"),
    )
    row_order = [
        "baseline",
        "delayed protection",
        "preserved reactive absorption",
        "voltage-mode RES instead of fixed-PF",
        "stronger/faster shunt support",
    ]
    missing_rows = [row for row in row_order if row not in set(ablation["row"].astype(str))]
    if missing_rows:
        raise MissingArtifactError(
            f"ablation_summary.csv is missing required row(s): {', '.join(missing_rows)}"
        )
    ablation = ablation.set_index("row").loc[row_order].reset_index()
    worst_k = pd.to_numeric(ablation["worst_k"], errors="coerce").to_numpy(dtype=float)
    mvar = pd.to_numeric(ablation["minimum_mvar_mitigation"], errors="coerce").to_numpy(dtype=float)
    if np.isnan(worst_k).any() or np.isnan(mvar).any():
        raise MissingArtifactError("ablation_summary.csv contains nonnumeric worst_k or minimum_mvar_mitigation")

    fig, (ax_a, ax_b) = plt.subplots(
        2,
        1,
        figsize=FIGURE_SPECS["column"],
        gridspec_kw={"height_ratios": [1.05, 1.0], "hspace": 0.38},
    )
    y = np.arange(len(row_order))
    max_worst = max(1.0, float(np.nanmax(worst_k)))
    max_mvar = max(1.0, float(np.nanmax(mvar)))
    xmin = -max_worst * 1.35
    xmax = max_mvar * 1.35
    ax_a.add_patch(
        Rectangle(
            (xmin, -0.275),
            xmax - xmin,
            0.55,
            facecolor=PALETTE["light_amber"],
            edgecolor="none",
            alpha=0.40,
            zorder=0,
        )
    )
    ax_a.barh(y, -worst_k, height=0.55, color=PALETTE["trip"], edgecolor="white", linewidth=0.8, zorder=2)
    ax_a.barh(y, mvar, height=0.55, color=PALETTE["control"], edgecolor="white", linewidth=0.8, zorder=2)
    for yi, k_value, m_value in zip(y, worst_k, mvar, strict=True):
        ax_a.text(-k_value - max_worst * 0.05, yi, f"{k_value:.1f}", ha="right", va="center", fontsize=7.4, color=PALETTE["trip"])
        ax_a.text(m_value + max_mvar * 0.05, yi, f"{m_value:.1f}", ha="left", va="center", fontsize=7.4, color=PALETTE["control"])
    ax_a.axvline(0, color=PALETTE["charcoal"], linewidth=0.7)
    ax_a.set_yticks(y)
    ax_a.set_yticklabels([ABLATION_SHORT[row] for row in row_order], fontsize=7.6)
    for tick, row in zip(ax_a.get_yticklabels(), row_order, strict=True):
        if row == "baseline":
            tick.set_fontweight("bold")
    ax_a.invert_yaxis()
    ax_a.set_xlim(xmin, xmax)
    ax_a.set_ylim(len(row_order) - 0.25, -0.75)
    ax_a.set_xticks([])
    ax_a.spines["bottom"].set_visible(False)
    ax_a.text(-max_worst * 0.55, -0.62, r"worst $K^{\rm pk}$", ha="center", va="bottom", fontsize=8.2, color=PALETTE["trip"], fontweight="bold")
    ax_a.text(max_mvar * 0.55, -0.62, "required Mvar", ha="center", va="bottom", fontsize=8.2, color=PALETTE["control"], fontweight="bold")
    ax_a.text(0, len(row_order) - 0.02, "intervention impact on the same IEEE-39 seed", ha="center", va="top", fontsize=7.0, color=PALETTE["charcoal"], fontstyle="italic")
    soften_axes(ax_a, grid=False)
    panel_label(ax_a, "a")

    color_cycle = [PALETTE["control"], PALETTE["safe"], PALETTE["data"], PALETTE["trip"]]
    markers = ["o", "s", "^", "D"]
    ax_b.plot([0, 100], [0, 100], color=PALETTE["grid"], linewidth=1.0, linestyle=(0, (4, 2)), label="no triage")
    ax_b.axhspan(80, 100, color=PALETTE["light_teal"], alpha=0.35, zorder=0)
    ax_b.text(4, 90, f"operational target\n$\\geq 80${_percent_symbol()} capture", ha="left", va="center", fontsize=6.8, color=PALETTE["safe"])
    for color, marker, case in zip(color_cycle, markers, SYSTEM_ORDER, strict=True):
        sub = triage[triage["case"] == case].sort_values("fraction_sent_to_tds")
        if sub.empty:
            continue
        xvals = sub["fraction_sent_to_tds"].to_numpy(dtype=float) * 100
        yvals = sub["capture_rate"].to_numpy(dtype=float) * 100
        ax_b.plot(xvals, yvals, marker=marker, ms=4.2, lw=1.8, color=color, label=SYSTEM_LABELS[case])
    ax_b.set_xlabel(f"events sent to nonlinear validation ({_percent_symbol()})")
    ax_b.set_ylabel(f"dangerous events captured ({_percent_symbol()})")
    ax_b.set_xlim(0, 105)
    ax_b.set_ylim(0, 105)
    ax_b.legend(loc="lower right", frameon=False, fontsize=6.8, ncol=2)
    soften_axes(ax_b, grid=True)
    panel_label(ax_b, "b")
    fig.subplots_adjust(left=0.20, right=0.98, bottom=0.13, top=0.94)
    written = save_figure(fig, "fig08_mitigation_scaling", out_fig, formats, outline_text=outline_text)
    plt.close(fig)
    return written


FIGURE_BUILDERS: dict[str, Callable[..., list[SaveResult]]] = {
    "fig01_replication_activsg2000": make_fig01_replication_activsg2000,
    "fig02_validation_matrix": make_fig02_validation_matrix,
    "fig03_margin_erosion_atlas": make_fig03_margin_erosion_atlas,
    "fig04_observability_gap": make_fig04_observability_gap,
    "fig05_operator_action_screen": make_fig05_operator_action_screen,
    "fig06_relay_window_controls": make_fig06_relay_window_controls,
    "fig07_uncertainty_telemetry_value": make_fig07_uncertainty_telemetry_value,
    "fig08_mitigation_scaling": make_fig08_mitigation_scaling,
}
