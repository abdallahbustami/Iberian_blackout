"""Paper-quality figures for the ACTIVSg2000 academic replica."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
import shutil
from typing import Any

import numpy as np


@dataclass(frozen=True, slots=True)
class ReplicaData:
    data_dir: Path
    collector_rows: list[dict[str, str]]
    system_rows: list[dict[str, str]]
    events: list[dict[str, Any]]
    config: dict[str, Any]
    summary: dict[str, Any]
    collector_names: tuple[str, ...]
    collector_cluster: dict[str, str]
    thresholds: dict[str, float]
    base_kv: dict[str, float]
    transmission_base_kv: dict[str, float]


def _to_float(value: Any) -> float:
    if value is None or value == "":
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _parse_raw_bus_base_kv(path: Path) -> dict[int, float]:
    bases: dict[int, float] = {}
    if not path.exists():
        return bases
    for line in path.read_text(errors="ignore").splitlines()[3:]:
        if line.strip().startswith("0") and "END OF BUS DATA" in line.upper():
            break
        parts = [part.strip().strip("'") for part in line.split(",")]
        try:
            bus = int(parts[0])
            bases[bus] = float(parts[2])
        except (IndexError, TypeError, ValueError):
            continue
    return bases


def _transmission_base_kv(config: dict[str, Any], data_dir: Path) -> dict[str, float]:
    case_id = str(config.get("case_id", ""))
    raw_candidates = [
        data_dir / "ACTIVSg2000.RAW",
        Path("data") / case_id / "ACTIVSg2000.RAW",
        data_dir.parent.parent / "data" / case_id / "ACTIVSg2000.RAW",
    ]
    bus_bases: dict[int, float] = {}
    for raw_path in raw_candidates:
        bus_bases = _parse_raw_bus_base_kv(raw_path)
        if bus_bases:
            break

    bus_by_name: dict[str, int] = {}
    for item in config.get("collector_specs", []):
        try:
            bus_by_name[str(item["name"])] = int(item.get("transmission_bus", 0))
        except (TypeError, ValueError):
            continue
    for item in config.get("collector_runtime", []):
        try:
            bus_by_name[str(item["name"])] = int(item.get("transmission_bus", 0))
        except (TypeError, ValueError):
            continue

    base_by_name: dict[str, float] = {}
    for name, bus in bus_by_name.items():
        base = bus_bases.get(bus)
        if base is not None and np.isfinite(base) and base > 0:
            base_by_name[name] = float(base)
    return base_by_name


def load_replica_data(data_dir: Path) -> ReplicaData:
    collector_rows = _read_csv(data_dir / "collector_traces.csv")
    system_rows = _read_csv(data_dir / "system_traces.csv")
    events = json.loads((data_dir / "events.json").read_text(encoding="utf-8"))
    config = json.loads((data_dir / "case_config.json").read_text(encoding="utf-8"))
    summary = json.loads((data_dir / "summary.json").read_text(encoding="utf-8"))

    if not collector_rows:
        raise ValueError(f"empty collector traces in {data_dir}")
    header = collector_rows[0].keys()
    collector_names = tuple(
        name[: -len("_voltage_pu")]
        for name in header
        if name.endswith("_voltage_pu") and not name.endswith("_raw")
    )
    collector_cluster = {
        str(item["name"]): str(item.get("report_cluster", "collector"))
        for item in config.get("collector_specs", [])
    }
    thresholds = {
        name: _to_float(collector_rows[0].get(f"{name}_threshold_pu"))
        for name in collector_names
    }
    base_kv = {
        str(item["name"]): _to_float(item.get("base_kv", 1.0))
        for item in config.get("collector_specs", [])
    }
    transmission_base_kv = _transmission_base_kv(config, data_dir)
    return ReplicaData(
        data_dir=data_dir,
        collector_rows=collector_rows,
        system_rows=system_rows,
        events=events,
        config=config,
        summary=summary,
        collector_names=collector_names,
        collector_cluster=collector_cluster,
        thresholds=thresholds,
        base_kv=base_kv,
        transmission_base_kv=transmission_base_kv,
    )


def _col(rows: list[dict[str, str]], name: str) -> np.ndarray:
    return np.asarray([_to_float(row.get(name)) for row in rows], dtype=float)


def _cluster_key(cluster: str) -> str:
    if "Granada" in cluster:
        return "E3"
    if "Badajoz" in cluster:
        return "E4"
    if "multi" in cluster:
        return "E5"
    return cluster


def _event_times(data: ReplicaData) -> dict[str, float]:
    by_category: dict[str, list[float]] = defaultdict(list)
    for event in data.events:
        by_category[str(event.get("category", ""))].append(float(event["time_s"]))

    cluster_times: dict[str, list[float]] = defaultdict(list)
    for event in data.events:
        if event.get("category") != "protection_trip":
            continue
        name = str(event.get("code", "")).replace("AA_", "")
        cluster_times[_cluster_key(data.collector_cluster.get(name, ""))].append(
            float(event["time_s"])
        )

    times: dict[str, float] = {}
    if by_category.get("operator_action"):
        times["OA"] = float(np.median(by_category["operator_action"]))
    for key in ("E3", "E4", "E5"):
        if cluster_times.get(key):
            times[key] = float(np.median(cluster_times[key]))
    category_labels = {
        "generator_trip": "Gen",
        "defense_action": "UFLS",
        "morocco_ac_trip": "Morocco",
        "out_of_step_trip": "France",
        "hvdc_block": "HVDC",
        "system_blackout_declared": "Blackout",
    }
    for category, label in category_labels.items():
        if by_category.get(category):
            times[label] = min(by_category[category])
    return times


def _apply_style() -> None:
    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 9.0,
            "axes.labelsize": 10.0,
            "xtick.labelsize": 8.6,
            "ytick.labelsize": 8.6,
            "legend.fontsize": 8.0,
            "axes.linewidth": 0.75,
            "xtick.major.width": 0.75,
            "ytick.major.width": 0.75,
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "path",
            "figure.dpi": 160,
        }
    )


def _save(fig: Any, stem: str, out_dirs: tuple[Path, ...], formats: tuple[str, ...]) -> list[Path]:
    written: list[Path] = []
    for out_dir in out_dirs:
        out_dir.mkdir(parents=True, exist_ok=True)
        for fmt in formats:
            path = out_dir / f"{stem}.{fmt}"
            kwargs: dict[str, Any] = {"bbox_inches": "tight", "pad_inches": 0.035}
            if fmt == "png":
                kwargs["dpi"] = 600
            fig.savefig(path, **kwargs)
            written.append(path)
    return written


def _x_limits(data: ReplicaData) -> tuple[float, float]:
    blackout = data.summary.get("blackout_time_s")
    if blackout is not None:
        return 0.0, float(blackout) + 0.75
    times = _col(data.system_rows, "time_s")
    return 0.0, float(np.nanmax(times))


def _draw_event_markers(
    ax: Any,
    data: ReplicaData,
    *,
    y_text: float,
    label_rotation: int = 0,
    compact: bool = False,
    labels: bool = True,
) -> None:
    event_times = _event_times(data)
    colors = {
        "OA": "#D4935C",
        "E3": "#B85C64",
        "E4": "#B85C64",
        "E5": "#B85C64",
        "Gen": "#6B7280",
        "UFLS": "#6B7280",
        "Morocco": "#3F6F8C",
        "France": "#3F6F8C",
        "HVDC": "#795A99",
        "Blackout": "#20242A",
    }
    label_text = (
        {
            "OA": "OA",
            "E3": "E3",
            "E4": "E4",
            "E5": "E5",
            "Gen": "gen.",
            "UFLS": "UFLS",
            "Morocco": "MA",
            "France": "FR",
            "HVDC": "HVDC",
            "Blackout": "BO",
        }
        if compact
        else {
            "OA": "operator actions",
            "E3": "E3 Granada",
            "E4": "E4 Badajoz",
            "E5": "E5 multi-site",
            "Gen": "gen. prot.",
            "UFLS": "UFLS/UVLS",
            "Morocco": "Morocco AC",
            "France": "France AC",
            "HVDC": "HVDC",
            "Blackout": "blackout",
        }
    )
    order = (
        "OA",
        "E3",
        "E4",
        "E5",
        "Gen",
        "UFLS",
        "Morocco",
        "France",
        "HVDC",
        "Blackout",
    )
    for idx, key in enumerate(order):
        if key not in event_times:
            continue
        x = event_times[key]
        color = colors[key]
        ax.axvline(x, color=color, lw=0.72, alpha=0.34, zorder=0)
        if not labels:
            continue
        dy = 0.0 if idx % 2 == 0 else -0.018
        ax.text(
            x,
            y_text + dy,
            label_text[key],
            ha="center",
            va="top",
            rotation=label_rotation,
            fontsize=5.8 if compact else 6.4,
            color=color,
            clip_on=False,
        )


def _shade_blackout(ax: Any, data: ReplicaData) -> None:
    blackout = data.summary.get("blackout_time_s")
    if blackout is None:
        return
    _, x_right = _x_limits(data)
    ax.axvspan(float(blackout), x_right, color="#2F3136", alpha=0.075, lw=0, zorder=-2)


def make_voltage_wide(
    data: ReplicaData,
    *,
    out_dirs: tuple[Path, ...],
    formats: tuple[str, ...],
) -> list[Path]:
    """Wide text-width per-unit voltage figure with separate upstream/collector panels."""
    _apply_style()
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    t = _col(data.collector_rows, "time_s")
    fig, (ax_trans, ax_coll, ax_evt) = plt.subplots(
        3,
        1,
        figsize=(7.15, 4.62),
        sharex=True,
        gridspec_kw={"height_ratios": [0.80, 0.98, 0.34], "hspace": 0.045},
    )
    for ax in (ax_trans, ax_coll, ax_evt):
        ax.set_facecolor("#FCFBF8")

    cluster_colors = {
        "E3": "#4E79A7",
        "E4": "#C17B58",
        "E5": "#7C6BB0",
        "collector": "#6A9F89",
    }
    highlight = {"Granada355", "Badajoz582", "Badajoz145", "Seville550", "Cuenca530"}
    visible_names = {
        str(item.get("name", ""))
        for item in data.config.get("collector_specs", [])
        if bool(item.get("visible_trace", False))
    }
    tripped_names = {
        str(event.get("code", "")).replace("AA_", "")
        for event in data.events
        if event.get("category") == "protection_trip"
    }
    representative_names = tuple(
        name
        for name in data.collector_names
        if name in visible_names or name in tripped_names or name in highlight
    )

    trans_all: list[np.ndarray] = []
    trans_thresholds: list[float] = []
    for name in representative_names:
        vtr = _col(data.collector_rows, f"{name}_transmission_voltage_pu")
        finite_vtr = vtr[np.isfinite(vtr)]
        if (
            finite_vtr.size == 0
            or float(np.nanmax(finite_vtr)) < 0.75
            or float(finite_vtr[0]) < 0.75
        ):
            continue
        trans_all.append(vtr)
        lw = 1.35 if name in highlight else 0.82
        alpha = 0.70 if name in highlight else 0.24
        ax_trans.plot(t, vtr, color="#8E9AA3", lw=lw, alpha=alpha, zorder=2)
        threshold_pu = data.thresholds.get(name, float("nan"))
        if np.isfinite(threshold_pu):
            trans_thresholds.append(float(threshold_pu))

    collector_all: list[np.ndarray] = []
    plotted_collectors: list[str] = []
    first_crossings: list[tuple[float, float]] = []
    for name in representative_names:
        cluster = _cluster_key(data.collector_cluster.get(name, "collector"))
        color = cluster_colors.get(cluster, cluster_colors["collector"])
        threshold = data.thresholds.get(name, float("nan"))
        v = _col(data.collector_rows, f"{name}_voltage_pu_raw")
        if (
            not np.isfinite(v).any()
            or (np.isfinite(threshold) and float(np.nanmax(v)) < 0.5 * threshold)
            or (not np.isfinite(threshold) and float(np.nanmax(v)) <= 0.75)
        ):
            continue
        collector_all.append(v)
        plotted_collectors.append(name)
        lw = 1.55 if name in highlight else 0.95
        alpha = 0.96 if name in highlight else 0.48
        ax_coll.plot(t, v, color=color, lw=lw, alpha=alpha, zorder=3)

        if np.isfinite(threshold):
            crossed = np.flatnonzero(np.isfinite(v) & (v >= threshold))
            if crossed.size:
                first = int(crossed[0])
                first_crossings.append((float(t[first]), float(v[first])))
                ax_coll.plot(
                    t[first],
                    v[first],
                    marker="o",
                    ms=4.2,
                    mfc="white",
                    mec="#2F3136",
                    mew=0.85,
                    color="#2F3136",
                    zorder=5,
                )

    thresholds = np.asarray(
        [
            data.thresholds[name]
            for name in plotted_collectors
            if np.isfinite(data.thresholds.get(name, float("nan")))
        ],
        dtype=float,
    )
    pickup_band_color = "#B85C64"
    pickup_fill_alpha = 0.070
    pickup_line_style = (0, (3.0, 2.2))
    if thresholds.size:
        pickup_min = float(np.nanmin(thresholds))
        pickup_max = float(np.nanmax(thresholds))
        ax_coll.axhspan(
            pickup_min,
            pickup_max,
            color=pickup_band_color,
            alpha=pickup_fill_alpha,
            lw=0,
            zorder=-1,
        )
        for y in (pickup_min, pickup_max):
            ax_coll.axhline(
                y,
                color=pickup_band_color,
                lw=0.72,
                ls=pickup_line_style,
                alpha=0.82,
                zorder=1,
            )
    trans_threshold_arr = np.asarray(trans_thresholds, dtype=float)
    if trans_threshold_arr.size:
        trans_pickup_min = float(np.nanmin(trans_threshold_arr))
        trans_pickup_max = float(np.nanmax(trans_threshold_arr))
        ax_trans.axhspan(
            trans_pickup_min,
            trans_pickup_max,
            color=pickup_band_color,
            alpha=pickup_fill_alpha * 0.82,
            lw=0,
            zorder=-1,
        )
        for y in (trans_pickup_min, trans_pickup_max):
            ax_trans.axhline(
                y,
                color=pickup_band_color,
                lw=0.72,
                ls=pickup_line_style,
                alpha=0.82,
                zorder=1,
            )

    _shade_blackout(ax_trans, data)
    _shade_blackout(ax_coll, data)
    event_colors = {
        "OA": "#D4935C",
        "E3": "#B85C64",
        "E4": "#B85C64",
        "E5": "#B85C64",
        "Gen": "#6B7280",
        "UFLS": "#6B7280",
        "Morocco": "#3F6F8C",
        "France": "#3F6F8C",
        "HVDC": "#795A99",
        "Blackout": "#20242A",
    }
    for axis in (ax_trans, ax_coll):
        axis.set_xlim(*_x_limits(data))
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.tick_params(axis="x", labelbottom=False)
    if trans_all:
        limit_values = [
            arr[np.isfinite(arr)]
            for arr in trans_all
            if np.isfinite(arr).any()
        ]
        if trans_threshold_arr.size:
            limit_values.append(trans_threshold_arr[np.isfinite(trans_threshold_arr)])
        trans_for_limits = np.concatenate(limit_values)
        y_lo = float(np.nanpercentile(trans_for_limits, 0.7))
        y_hi = float(np.nanpercentile(trans_for_limits, 99.3))
        pad = max(0.012, 0.055 * (y_hi - y_lo))
        ax_trans.set_ylim(y_lo - pad, y_hi + pad)
    collector_limit_values = [
        arr[np.isfinite(arr)]
        for arr in collector_all
        if np.isfinite(arr).any()
    ]
    if thresholds.size:
        collector_limit_values.append(thresholds[np.isfinite(thresholds)])
    if collector_limit_values:
        coll_for_limits = np.concatenate(collector_limit_values)
        coll_lo = float(np.nanpercentile(coll_for_limits, 0.7))
        coll_hi = float(np.nanpercentile(coll_for_limits, 99.3))
        coll_pad = max(0.014, 0.075 * (coll_hi - coll_lo))
        ax_coll.set_ylim(coll_lo - coll_pad, coll_hi + coll_pad)
    if first_crossings:
        tx, vy = min(first_crossings, key=lambda item: item[0])
        y0, y1 = ax_coll.get_ylim()
        ax_coll.annotate(
            "first threshold\ncrossing",
            xy=(tx, vy),
            xytext=(min(tx + 1.55, _x_limits(data)[1] - 2.5), min(y1 - 0.006, vy + 0.16 * (y1 - y0))),
            ha="left",
            va="center",
            fontsize=5.8,
            color="#2F3136",
            arrowprops={
                "arrowstyle": "->",
                "lw": 0.65,
                "color": "#2F3136",
                "shrinkA": 2,
                "shrinkB": 2,
            },
            zorder=6,
        )
    ax_trans.set_ylabel("Upstream\nvoltage (pu)")
    ax_coll.set_ylabel("Collector\nvoltage (pu)")

    ax_evt.axhline(0.0, color="#BFB8AF", lw=1.0, zorder=0)
    ribbon_styles = {
        "operator_action": ("v", "#D4935C"),
        "protection_trip": ("|", "#B85C64"),
        "generator_trip": ("D", "#6B7280"),
        "defense_action": ("s", "#6B7280"),
        "morocco_ac_trip": ("D", "#3F6F8C"),
        "out_of_step_trip": ("D", "#3F6F8C"),
        "hvdc_block": ("D", "#795A99"),
        "system_blackout_declared": ("X", "#20242A"),
        "island_blackout_declared": ("X", "#20242A"),
    }
    labeled_events: list[tuple[dict[str, Any], str, float]] = []
    oa_totals: dict[str, int] = defaultdict(int)
    for event in data.events:
        if event.get("category") == "operator_action":
            oa_totals[str(event.get("code", "OA"))] += 1
    repeated_oa_counts: dict[str, int] = defaultdict(int)
    for event in data.events:
        category = str(event.get("category", ""))
        if category == "island_blackout_declared":
            continue
        style = ribbon_styles.get(category)
        if style is None:
            continue
        marker, color = style
        x = float(event["time_s"])
        for axis in (ax_trans, ax_coll):
            axis.axvline(x, color=color, lw=0.55, alpha=0.13, zorder=0)
        size = 4.6
        mew = 1.0
        fill = color
        if category == "protection_trip":
            size = 8.0
            mew = 1.15
            fill = "none"
        elif category == "operator_action":
            size = 5.2
        if category == "system_blackout_declared":
            size = 5.6
        ax_evt.plot(
            x,
            0.0,
            marker=marker,
            ms=size,
            mew=mew,
            color=color,
            mfc=fill,
            mec=color if marker in {"|", "v"} else "white",
            clip_on=False,
            zorder=3,
        )
        label = ""
        if category == "operator_action":
            code = str(event.get("code", "OA"))
            repeated_oa_counts[code] += 1
            suffix = chr(ord("a") + repeated_oa_counts[code] - 1)
            label = f"{code}{suffix}" if oa_totals[code] > 1 else code
        elif category == "protection_trip":
            name = str(event.get("code", "")).replace("AA_", "")
            label = _cluster_key(data.collector_cluster.get(name, ""))
        elif category == "generator_trip":
            label = "GEN"
        elif category == "defense_action":
            label = str(event.get("code", "UFLS"))
        elif category == "morocco_ac_trip":
            label = "MA"
        elif category == "out_of_step_trip":
            label = "FR"
        elif category == "hvdc_block":
            label = "HVDC"
        elif category == "system_blackout_declared":
            label = "BO"
        if label:
            labeled_events.append((event, label, x))

    label_y = 0.18
    min_label_gap = 0.31
    last_label_x = -1e9
    for event, label, x in sorted(labeled_events, key=lambda item: item[2]):
        category = str(event.get("category", ""))
        color = ribbon_styles.get(category, ("", "#6B7280"))[1]
        display_x = max(x, last_label_x + min_label_gap)
        last_label_x = display_x
        if abs(display_x - x) > 0.025:
            ax_evt.plot(
                [x, display_x],
                [0.035, label_y - 0.015],
                color=color,
                lw=0.35,
                alpha=0.42,
                clip_on=False,
                zorder=2,
            )
        ax_evt.text(
            display_x,
            label_y,
            label,
            ha="center",
            va="bottom",
            rotation=90,
            fontsize=5.0,
            color=color,
            clip_on=False,
            zorder=4,
        )
    ax_evt.set_ylim(-0.26, 1.05)
    ax_evt.set_yticks([])
    ax_evt.set_xlabel("Time (s)", labelpad=3)
    ax_evt.tick_params(axis="x", length=3, pad=1)
    for name, spine in ax_evt.spines.items():
        spine.set_visible(name == "bottom")
    ax_evt.spines["bottom"].set_color("#C9C2BA")

    legend_handles = [
        Line2D([0], [0], color="#8E9AA3", lw=1.4, alpha=0.75, label="upstream 500-kV-area traces"),
        Line2D([0], [0], color="#8E9AA3", lw=1.0, alpha=0.42, label="other protected collectors"),
        Line2D([0], [0], color=cluster_colors["E3"], lw=1.7, label="E3 Granada collector"),
        Line2D([0], [0], color=cluster_colors["E4"], lw=1.7, label="E4 Badajoz collectors"),
        Line2D([0], [0], color=cluster_colors["E5"], lw=1.7, label="E5 multi-site collectors"),
        Patch(facecolor=pickup_band_color, edgecolor="none", alpha=pickup_fill_alpha * 1.8, label="relay pickup range"),
        Line2D([0], [0], color=pickup_band_color, lw=0.9, ls=pickup_line_style, label="pickup range limits"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="white", markeredgecolor="#2F3136", markersize=4.8, label="first threshold crossing"),
    ]
    ax_trans.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.50, 1.35),
        ncol=4,
        frameon=False,
        handlelength=1.45,
        columnspacing=0.65,
        borderaxespad=0.0,
        fontsize=6.2,
    )

    fig.subplots_adjust(left=0.087, right=0.992, bottom=0.120, top=0.832)
    paths = _save(fig, "fig_replication_voltage_wide", out_dirs, formats)
    alias_paths: list[Path] = []
    for path in paths:
        alias = path.with_name(path.name.replace("fig_replication_voltage_wide", "fig_replication_activsg2000"))
        shutil.copy2(path, alias)
        alias_paths.append(alias)
    plt.close(fig)
    return paths + alias_paths


def make_mechanism_compact(
    data: ReplicaData,
    *,
    out_dirs: tuple[Path, ...],
    formats: tuple[str, ...],
) -> list[Path]:
    """Two-panel compact mechanism figure."""
    _apply_style()
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    t = _col(data.collector_rows, "time_s")
    st = _col(data.system_rows, "time_s")

    fig, (ax_v, ax_loss) = plt.subplots(
        2,
        1,
        figsize=(3.55, 4.55),
        sharex=False,
        gridspec_kw={"height_ratios": [1.08, 0.92]},
    )
    for ax in (ax_v, ax_loss):
        ax.set_facecolor("#FCFBF8")
        _shade_blackout(ax, data)
        ax.grid(axis="y", color="#E8E0D7", lw=0.65)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    cluster_colors = {
        "E3": "#4E79A7",
        "E4": "#C17B58",
        "E5": "#7C6BB0",
        "collector": "#6A9F89",
    }
    trans_norm_stack: list[np.ndarray] = []
    for name in data.collector_names:
        threshold = data.thresholds.get(name, float("nan"))
        if not np.isfinite(threshold) or threshold <= 0:
            continue
        trans_norm_stack.append(_col(data.collector_rows, f"{name}_transmission_voltage_pu") / threshold)
    if trans_norm_stack:
        arr = np.vstack(trans_norm_stack)
        ax_v.fill_between(
            t,
            np.nanpercentile(arr, 10, axis=0),
            np.nanpercentile(arr, 90, axis=0),
            color="#AEB5BA",
            alpha=0.25,
            lw=0,
            label="upstream transmission band",
        )
        ax_v.plot(t, np.nanmedian(arr, axis=0), color="#8D969C", lw=1.0, ls=(0, (4, 2)))

    for name in data.collector_names:
        threshold = data.thresholds.get(name, float("nan"))
        if not np.isfinite(threshold) or threshold <= 0:
            continue
        cluster = _cluster_key(data.collector_cluster.get(name, "collector"))
        color = cluster_colors.get(cluster, cluster_colors["collector"])
        v = _col(data.collector_rows, f"{name}_voltage_pu") / threshold
        highlight = name in {"Granada355", "Badajoz582", "Badajoz145", "Seville550", "Cuenca530"}
        ax_v.plot(v * 0 + t, v, color=color, lw=1.35 if highlight else 0.82, alpha=0.90 if highlight else 0.42)
    ax_v.axhline(1.0, color="#B85C64", lw=1.0, ls=(0, (5, 2.6)))
    ax_v.text(
        0.012,
        1.006,
        "relay pickup",
        ha="left",
        va="bottom",
        transform=ax_v.get_yaxis_transform(),
        fontsize=7.4,
        color="#9C4F58",
    )
    _draw_event_markers(ax_v, data, y_text=1.145, label_rotation=90, compact=True)
    ax_v.set_xlim(*_x_limits(data))
    ax_v.set_ylim(0.89, 1.16)
    ax_v.set_ylabel("Voltage / pickup")

    lost_p_gw = _col(data.system_rows, "cumulative_lost_p_mw") / 1000.0
    lost_q_gvar = _col(data.system_rows, "cumulative_lost_q_absorption_mvar") / 1000.0
    ax_loss.step(st, lost_p_gw, where="post", color="#B85C64", lw=1.45, label="tripped generation")
    ax_loss.set_ylabel("Disconnected\nactive power (GW)")
    ax_loss.set_xlim(*_x_limits(data))
    ax_loss.set_ylim(-0.15, max(1.0, float(np.nanmax(lost_p_gw)) * 1.08))
    ax_q = ax_loss.twinx()
    ax_q.step(st, lost_q_gvar, where="post", color="#4F7F73", lw=1.45, label="removed absorption")
    ax_q.set_ylabel("Removed\nabsorption (GVAr)", color="#4F7F73")
    ax_q.tick_params(axis="y", colors="#4F7F73")
    ax_q.spines["top"].set_visible(False)
    ax_q.set_ylim(-0.03, max(0.25, float(np.nanmax(lost_q_gvar)) * 1.20))
    _draw_event_markers(
        ax_loss,
        data,
        y_text=ax_loss.get_ylim()[1] * 0.98,
        label_rotation=90,
        compact=True,
        labels=False,
    )
    ax_loss.set_xlabel("Time (s)")

    handles = [
        Line2D([0], [0], color="#AEB5BA", lw=5, alpha=0.4, label="upstream band"),
        Line2D([0], [0], color=cluster_colors["E3"], lw=1.5, label="E3"),
        Line2D([0], [0], color=cluster_colors["E4"], lw=1.5, label="E4"),
        Line2D([0], [0], color=cluster_colors["E5"], lw=1.5, label="E5"),
    ]
    ax_v.legend(
        handles=handles,
        loc="lower right",
        ncol=4,
        frameon=False,
        handlelength=1.1,
        columnspacing=0.55,
        fontsize=7.1,
        borderaxespad=0.1,
    )
    h1, l1 = ax_loss.get_legend_handles_labels()
    h2, l2 = ax_q.get_legend_handles_labels()
    ax_loss.legend(
        h1 + h2,
        l1 + l2,
        loc="upper left",
        frameon=False,
        fontsize=7.2,
        handlelength=1.5,
        borderaxespad=0.2,
    )

    fig.subplots_adjust(left=0.165, right=0.84, top=0.92, bottom=0.105, hspace=0.22)
    paths = _save(fig, "fig_replication_mechanism_compact", out_dirs, formats)
    plt.close(fig)
    return paths


def make_all(
    *,
    data_dir: Path,
    fig_dir: Path,
    formats: tuple[str, ...],
) -> list[Path]:
    data = load_replica_data(data_dir)
    out_dirs = (fig_dir, data_dir)
    written: list[Path] = []
    written.extend(make_voltage_wide(data, out_dirs=out_dirs, formats=formats))
    written.extend(make_mechanism_compact(data, out_dirs=out_dirs, formats=formats))
    return written


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("results/activsg2000_iberian_replication"))
    parser.add_argument("--fig-dir", type=Path, default=Path("LaTeX/figures/generated"))
    parser.add_argument("--formats", default="pdf")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    formats = tuple(item.strip().lower() for item in args.formats.split(",") if item.strip())
    written = make_all(data_dir=args.data_dir, fig_dir=args.fig_dir, formats=formats)
    for path in written:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
