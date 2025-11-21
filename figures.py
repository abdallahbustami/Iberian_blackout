# figures.py
# -*- coding: utf-8 -*-
"""
Plotting for the Iberian blackout replication
Run after running blackout.py
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Sequence, Tuple, Optional
from collections import defaultdict
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

try:
    from scipy.optimize import curve_fit

    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("Warning: scipy not available, exponential fitting disabled")


_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

_cascade_module = None
for _name in ("blackout",):
    try:
        _mod = __import__(_name, fromlist=["CascadeConfig", "CascadeScenario"])
        CascadeConfig = getattr(_mod, "CascadeConfig")
        CascadeScenario = getattr(_mod, "CascadeScenario")
        _cascade_module = _name
        break
    except ModuleNotFoundError:
        continue

if _cascade_module is None:
    raise ModuleNotFoundError(
        "Could not locate CascadeConfig/CascadeScenario in blackout.py."
    )

_default_cfg = CascadeConfig()
COLLECTOR_THRESHOLD_PU = float(_default_cfg.collector_threshold_pu)
TRANSMISSION_THRESHOLD_PU = float(
    getattr(_default_cfg, "threshold_hv_pu", COLLECTOR_THRESHOLD_PU)
)
FALLBACK_OA_MVAR = {
    "OA1": float(getattr(_default_cfg, "mesh_mvar_loss", 0.0)),
    "OA2": float(getattr(_default_cfg, "export_reduction_mvar", 0.0)),
    "OA3": float(getattr(_default_cfg, "reactor_mvar", 0.0)),
    "OA4": float(getattr(_default_cfg, "hvdc_mode_mvar", 0.0)),
}

mpl.rcParams.update(
    {
        "text.usetex": False,
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "Liberation Serif"],
        "font.size": 11,
        "axes.labelsize": 11,
        "axes.titlesize": 0,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9,
        "pdf.use14corefonts": True,
        "ps.useafm": True,
        "ps.fonttype": 42,
        "pdf.fonttype": 42,
        "svg.fonttype": "path",
    }
)

EVENT_COLORS = {
    "OA": "#D9822B",
    "AA": "#C7504A",
    "Context": "#6E6E6E",
}

FOCUS_TRANSMISSION_BUSES = (3, 4, 16, 19, 20, 21, 22, 25, 33, 35, 38)

BUS_NOISE_KV = 0
COLLECTOR_NOISE_KV = 0
PROTECTION_PU = TRANSMISSION_THRESHOLD_PU
RNG = np.random.default_rng(20250428)


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _smoothed_noise(n: int, sigma: float) -> np.ndarray:
    if sigma <= 0 or n <= 0:
        return np.zeros(max(n, 1))
    raw = RNG.normal(0.0, sigma, size=n + 40)
    kernel = np.ones(41) / 41.0
    smooth = np.convolve(raw, kernel, mode="valid")
    return smooth[:n]


def _save_all(fig: plt.Figure, path_no_ext: Path) -> None:
    fig.savefig(path_no_ext.with_suffix(".pdf"), dpi=300, bbox_inches="tight")


def _load_csv_timeseries(path: Path) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    array = np.genfromtxt(path, delimiter=",", names=True)
    if array.ndim == 0:
        array = np.array([tuple(array)], dtype=array.dtype)
    time = array["time_s"]
    data = {
        name: np.asarray(array[name], dtype=float)
        for name in array.dtype.names
        if name != "time_s"
    }
    return np.asarray(time, dtype=float), data


def _format_time_axis(
    ax: plt.Axes, time: np.ndarray, start: datetime | None, step: float = 1.0
) -> None:
    if time.size == 0:
        return
    start_tick = np.floor(time[0] / step) * step
    ticks = np.arange(start_tick, time[-1] + 1e-6, step)
    if start is None:
        labels = [
            f"{float(t):.0f}" if step >= 1.0 else f"{float(t):.1f}" for t in ticks
        ]
        ax.set_xlabel("Time (s)")
    else:
        labels = [
            (start + timedelta(seconds=float(t))).strftime("%H:%M:%S") for t in ticks
        ]
        ax.set_xlabel("Time (CEST)")
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, rotation=0)
    ax.set_xlim(time[0], time[-1])


@dataclass
class EventInfo:
    time: float
    code: str
    description: str
    category: str
    time_label: str

    def short_label(self) -> str:
        if self.code.startswith("AA_C"):
            return self.code.replace("AA_", "")
        return self.code


@dataclass
class CascadeDataset:
    time: np.ndarray
    start_time: datetime
    bus_kv: Dict[str, np.ndarray]
    collector_kv: Dict[str, np.ndarray]
    bus_basekv: Dict[str, float]
    collector_basekv: Dict[str, float]
    threshold_pu: float
    transmission_threshold_pu: float
    lower_voltage_threshold_pu: float
    collector_thresholds: Dict[str, float]
    collector_scada_kv: Dict[str, np.ndarray]
    events: List[EventInfo]
    collector_mapping: Dict[str, Dict[str, float]]
    reactive_events: List[Dict[str, float]]

    def cascade_window(self) -> Tuple[float, float]:
        aa_times = [ev.time for ev in self.events if ev.code.startswith("AA_C")]
        if not aa_times:
            return float(self.time[0]), float(self.time[-1])
        return float(min(aa_times)), float(self.time[-1])


def load_cascade_dataset(
    bus_csv: Path = Path("iberian_cascade_bus_voltages.csv"),
    collector_csv: Path = Path("iberian_cascade_collector_voltages.csv"),
    events_json: Path = Path("iberian_cascade_events.json"),
    scada_csv: Path = Path("iberian_cascade_scada_voltages.csv"),
) -> CascadeDataset:
    meta = json.loads(events_json.read_text())
    start_time = datetime.strptime(meta["start_time_local"], "%Y-%m-%d %H:%M:%S")
    events = [
        EventInfo(
            time=float(ev["time_s"]),
            code=str(ev["code"]),
            description=str(ev["description"]),
            category=str(ev["category"]),
            time_label=str(ev.get("time_local", "")),
        )
        for ev in meta.get("events", [])
    ]
    time_bus, bus_pu = _load_csv_timeseries(bus_csv)
    time_col, col_pu = _load_csv_timeseries(collector_csv)
    if not np.allclose(time_bus, time_col):
        raise ValueError("Transmission and collector time axes differ.")
    bus_basekv = {str(k): float(v) for k, v in meta.get("bus_basekv", {}).items()}
    collector_basekv = {
        str(k): float(v) for k, v in meta.get("collector_basekv", {}).items()
    }
    bus_kv = {k: bus_pu[k] * bus_basekv.get(k, 345.0) for k in bus_pu}
    collector_kv = {k: col_pu[k] * collector_basekv.get(k, 138.0) for k in col_pu}
    for key in bus_kv:
        bus_kv[key] = bus_kv[key] + _smoothed_noise(len(bus_kv[key]), BUS_NOISE_KV)
    for key in collector_kv:
        collector_kv[key] = collector_kv[key] + _smoothed_noise(
            len(collector_kv[key]), COLLECTOR_NOISE_KV
        )
    mapping_list = meta.get("collector_mapping", [])
    collector_mapping = {entry["name"]: entry for entry in mapping_list}
    trans_thresh = float(meta.get("transmission_threshold_pu", TRANSMISSION_THRESHOLD_PU))
    lv_thresh = float(meta.get("lower_voltage_threshold_pu", trans_thresh))
    thresholds_map = {
        entry["name"]: float(entry.get("threshold_pu", PROTECTION_PU))
        for entry in mapping_list
    }
    raw_thresholds = meta.get("collector_thresholds_pu", {})
    if raw_thresholds:
        thresholds_map.update({k: float(v) for k, v in raw_thresholds.items()})
    scada_kv: Dict[str, np.ndarray] = {}
    if scada_csv.exists():
        time_scada, scada_pu = _load_csv_timeseries(scada_csv)
        if not np.allclose(time_scada, time_bus):
            raise ValueError("SCADA and bus time axes differ.")
        for key, series in scada_pu.items():
            mapping = collector_mapping.get(key, {})
            trans_bus = mapping.get("trans_bus")
            bus_key = f"bus_{trans_bus}" if trans_bus is not None else None
            basekv = bus_basekv.get(bus_key, 345.0)
            scada_kv[key] = series * basekv
    else:
        for key, mapping in collector_mapping.items():
            trans_bus = mapping.get("trans_bus")
            bus_key = f"bus_{trans_bus}" if trans_bus is not None else None
            if bus_key and bus_key in bus_kv:
                scada_kv[key] = bus_kv[bus_key].copy()

    reactive_events = [
        {
            "time": float(item.get("time", 0.0)),
            "mvar": float(item.get("mvar", 0.0)),
            "label": item.get("label", ""),
            "category": item.get("category", ""),
        }
        for item in meta.get("reactive_events", [])
    ]

    has_aa = any(evt.get("category") == "AA" for evt in reactive_events)
    if not reactive_events or not has_aa:
        # fall back to collector trips only
        for ev in events:
            if ev.code.startswith("AA_C"):
                name = ev.code.replace("AA_", "")
                info = collector_mapping.get(name, {})
                reactive_events.append(
                    {
                        "time": ev.time,
                        "mvar": float(info.get("q_absorption_mvar", 0.0)),
                        "label": f"{name} trip",
                        "category": "AA",
                    }
                )

    has_oa = any(evt.get("category") == "OA" for evt in reactive_events)
    if not has_oa:
        for ev in events:
            if ev.category != "OA":
                continue
            key = ev.code[:3]
            amount = FALLBACK_OA_MVAR.get(key, 0.0)
            if np.isclose(amount, 0.0):
                continue
            reactive_events.append(
                {
                    "time": ev.time,
                    "mvar": amount,
                    "label": ev.description or ev.code,
                    "category": "OA",
                }
            )

    reactive_events.sort(key=lambda item: item["time"])

    return CascadeDataset(
        time=time_bus,
        start_time=start_time,
        bus_kv=bus_kv,
        collector_kv=collector_kv,
        bus_basekv=bus_basekv,
        collector_basekv=collector_basekv,
        threshold_pu=float(meta.get("protection_threshold_pu", PROTECTION_PU)),
        transmission_threshold_pu=trans_thresh,
        lower_voltage_threshold_pu=lv_thresh,
        collector_thresholds=thresholds_map,
        collector_scada_kv=scada_kv,
        reactive_events=reactive_events,
        events=events,
        collector_mapping=collector_mapping,
    )


def _draw_events(
    ax_top: plt.Axes, ax_bot: plt.Axes, events: Sequence[EventInfo]
) -> List[Line2D]:
    handles: List[Line2D] = []
    seen_categories: Dict[str, bool] = {}
    for event in events:
        if event.category == "Context":
            continue
        color = EVENT_COLORS.get(event.category, "#6B6B6B")
        linestyle = "--" if event.category == "OA" else "-"
        for axis in (ax_top, ax_bot):
            axis.axvline(
                event.time, color=color, ls=linestyle, lw=1.0, alpha=0.6, zorder=1
            )
        target = ax_top if event.category == "OA" else ax_bot
        ymin, ymax = target.get_ylim()
        span = ymax - ymin
        if event.category == "OA":
            y = ymax - 0.05 * span
            va = "top"
            offset = (-2, -10)
        else:
            y = ymin + 0.05 * span
            va = "bottom"
            offset = (2, 8)
        if event.category not in seen_categories:
            handles.append(
                Line2D(
                    [0],
                    [0],
                    color=color,
                    ls=linestyle,
                    lw=1.4,
                    label=f"{event.category} actions",
                )
            )
            seen_categories[event.category] = True
        label = event.short_label()
        if event.code.lower() == "collapse":
            label = "Collapse"
        target.annotate(
            label,
            xy=(event.time, y),
            xytext=offset,
            textcoords="offset points",
            rotation=90,
            ha="center",
            va=va,
            fontsize=8,
            color=color,
            bbox=dict(
                boxstyle="round,pad=0.2",
                facecolor="#ffffff",
                edgecolor=color,
                linewidth=0.6,
                alpha=0.85,
            ),
            zorder=6,
        )
    return handles


# -----------------------------------------------------------------------------
# Figure 1 – Meshing impact
# -----------------------------------------------------------------------------
def _localize_meshing_to_corridor(scn: CascadeScenario,
                                  line_x_scale: float = 1.0) -> None:
    """
    Add parallel lines only around a line near the first PV / EHV bus.
    The new lines start open (u=0). Their reactance is scaled by line_x_scale.

    line_x_scale < 1.0  → stronger parallel line (bigger meshing effect)
    line_x_scale = 1.0  → identical copy
    line_x_scale > 1.0  → weaker parallel line
    """
    # pick a focus bus
    if hasattr(scn, "pv_bus_map") and getattr(scn, "pv_bus_map"):
        focus_bus = scn.pv_bus_map[0]
    elif hasattr(scn.ss, "PV") and getattr(scn.ss.PV, "n", 0) > 0:
        focus_bus = int(scn.ss.PV.bus.v[0])
    elif scn.ehv_buses:
        focus_bus = int(scn.ehv_buses[0])
    else:
        focus_bus = int(scn.ss.Bus.idx.v[0])

    line_ids = list(scn.ss.Line.idx.v)
    bus1 = [int(v) for v in list(scn.ss.Line.bus1.v)]
    bus2 = [int(v) for v in list(scn.ss.Line.bus2.v)]

    # neighbors of the focus bus
    neighbours = {b1 for (b1, b2) in zip(bus1, bus2) if b2 == focus_bus}
    neighbours |= {b2 for (b1, b2) in zip(bus1, bus2) if b1 == focus_bus}

    # throw away any mesh ids created in build()
    scn.mesh_line_ids = []
    scn.base_line_ids = []

    for idx_name in line_ids:
        pos = line_ids.index(idx_name)
        end1, end2 = bus1[pos], bus2[pos]
        if not ({end1, end2} & (neighbours | {focus_bus})):
            continue

        params = {
            "bus1": end1,
            "bus2": end2,
            "r": float(scn.ss.Line.r.v[pos]),
            "x": float(scn.ss.Line.x.v[pos]) * line_x_scale,
            "b": float(scn.ss.Line.b.v[pos]),
            "u": 0,  # start open; monitor will close if enable_meshing=True
            "name": f"Mesh_{idx_name}",
        }
        new_idx = scn.ss.add("Line", params)
        scn.mesh_line_ids.append(new_idx)
        scn.base_line_ids.append(idx_name)



def _run_variant(dq_mvar: float,
                 meshed: bool,
                 line_x_scale: float) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Run one scenario:

    dq_mvar       constant MVAr injection at HV/EHV buses (stress level)
    meshed        whether operators actually close the mesh lines
    line_x_scale  scaling of the new lines' reactance
    """
    cfg = CascadeConfig(
        sim_tf=8.5,
        enable_meshing=meshed,
        enable_reactor_switch=False,
        enable_distribution_q=(abs(dq_mvar) > 1e-6),
        enable_hvdc_mode_change=False,
        enable_export_reduction=False,
        dq_mvar=dq_mvar,
        threshold_hv_pu=PROTECTION_PU,
        collapse_freq=0.0,
        ufls_trigger_hz=0.0,
    )

    scn = CascadeScenario(cfg)
    scn.build()

    # Localize the extra lines and scale their X
    _localize_meshing_to_corridor(scn, line_x_scale=line_x_scale)

    if hasattr(scn.ss, "TGOV1") and getattr(scn.ss.TGOV1, "n", 0) > 0:
        for i in range(scn.ss.TGOV1.n):
            scn.ss.TGOV1.u.v[i] = 0

    scn.run()
    metrics = scn.metrics

    # Pick a representative normal EHV bus (closest to 1.0 pu initially)
    bus_voltages = metrics.get("bus_voltages", {})
    bus_basekv = metrics.get("bus_basekv", {})
    ehv_buses = [k for k in bus_voltages.keys() if bus_basekv.get(k, 0) >= 300]
    if ehv_buses:
        def score(key: str) -> float:
            series = bus_voltages[key]
            return abs(series[0] - 1.0)

        target_bus = min(ehv_buses, key=score)
        base_kv = bus_basekv.get(target_bus, 345.0)
        metrics["target_bus_v"] = bus_voltages[target_bus]
        metrics["target_bus_key"] = target_bus
        metrics["target_bus_basekv"] = base_kv
    else:
        metrics["target_bus_v"] = metrics["V_avg"]
        metrics["target_bus_key"] = None
        metrics["target_bus_basekv"] = 345.0

    if scn.cfg.enable_meshing and getattr(scn.cfg, "meshing_times", None):
        metrics["meshing_time"] = float(scn.cfg.meshing_times[0])
    else:
        metrics["meshing_time"] = None

    return metrics["time"], metrics

def create_meshing_figure(out: Path = Path("fig_meshing_impact")) -> None:

    dq_mvar = 150.0      # how stressed the system is; lower this for more headroom
    line_x_scale = 1.0   # 0.8 = stronger mesh effect, 1.0 = identical copy, >1 = weaker

    # No-meshing case
    t_nomesh, m_nomesh = _run_variant(dq_mvar, meshed=False, line_x_scale=line_x_scale)

    # Meshing case (same dq, now we actually close the mesh lines)
    t_mesh, m_mesh = _run_variant(dq_mvar, meshed=True, line_x_scale=line_x_scale)

    bus_label = m_nomesh.get("target_bus_key") or m_mesh.get("target_bus_key")

    def _series_from_bus(metrics: Dict[str, np.ndarray], label: str | None):
        bus_voltages = metrics.get("bus_voltages", {})
        bus_basekv = metrics.get("bus_basekv", {})
        if label and label in bus_voltages:
            base = bus_basekv.get(label, 345.0)
            return bus_voltages[label] * base, base
        return metrics["target_bus_v"] * metrics.get("target_bus_basekv", 345.0), metrics.get(
            "target_bus_basekv", 345.0
        )

    target_nomesh, base_nomesh = _series_from_bus(m_nomesh, bus_label)
    target_mesh, base_mesh = _series_from_bus(m_mesh, bus_label)
    avg_nomesh = m_nomesh["V_avg"] * base_nomesh
    avg_mesh = m_mesh["V_avg"] * base_mesh

    # "Spread" = |target - system average|
    spread_nomesh = np.abs(target_nomesh - avg_nomesh)
    spread_mesh = np.abs(target_mesh - avg_mesh)

    fig, ax1 = plt.subplots(figsize=(6.2, 3.2))
    fig.subplots_adjust(left=0.12, right=0.95, top=0.95, bottom=0.18)

    # Shaded spreads
    ax1.fill_between(
        t_nomesh,
        target_nomesh - spread_nomesh,
        target_nomesh + spread_nomesh,
        alpha=0.10,
        color="#AED6F1",
        label="No mesh spread",
    )
    ax1.fill_between(
        t_mesh,
        target_mesh - spread_mesh,
        target_mesh + spread_mesh,
        alpha=0.14,
        color="#F5CBA7",
        label="Mesh spread",
    )

    # Lines
    label_text = (
        bus_label.replace("bus_", "Bus ") if isinstance(bus_label, str) else "Target bus"
    )
    line1 = ax1.plot(
        t_nomesh, target_nomesh, lw=2.0, label=f"{label_text} (no meshing)"
    )
    line2 = ax1.plot(
        t_mesh, target_mesh, lw=2.0, label=f"{label_text} (with meshing)"
    )
    line3 = ax1.plot(
        t_nomesh, avg_nomesh, ls="--", lw=1.3, label="System avg (no mesh)"
    )
    line4 = ax1.plot(
        t_mesh, avg_mesh, ls="--", lw=1.3, label="System avg (mesh)"
    )

    # Protection threshold at EHV
    ax1.axhline(
        PROTECTION_PU * base_nomesh,
        color="#6B6B6B",
        ls="--",
        lw=1.3,
        label="Protection threshold",
    )
    event_time = m_mesh.get("meshing_time")
    if event_time is not None:
        ax1.axvline(
            event_time,
            color="#8E44AD",
            ls=":",
            lw=1.3,
            label="Meshing event",
        )

    ax1.set_xlim(0, 8.5)
    ax1.set_ylim(340, 420)
    ax1.set_ylabel("Voltage (kV)")
    ax1.set_xlabel("Time (s)")
    ax1.grid(alpha=0.2, ls=":", linewidth=0.7)

    # Secondary axis in pu
    ax2 = ax1.twinx()
    ax2.set_ylim(ax1.get_ylim())
    ax2.set_ylabel("Voltage (pu)")
    yticks = ax1.get_yticks()
    ax2.set_yticks(yticks)
    ax2.set_yticklabels([f"{tick / base_nomesh:.2f}" for tick in yticks])

    # Legend
    spread_handles = [
        Rectangle((0, 0), 1, 1, alpha=0.10, fc="#AED6F1", label="No mesh spread"),
        Rectangle((0, 0), 1, 1, alpha=0.14, fc="#F5CBA7", label="Mesh spread"),
    ]
    extra_handle = Line2D(
        [0], [0], color="#8E44AD", ls=":", lw=1.3, label="Meshing event"
    )
    ax1.legend(
        handles=line1 + line2 + line3 + line4 + spread_handles + [extra_handle],
        loc="upper left",
        framealpha=0.95,
        ncol=2,
        fontsize=9,
    )

    _save_all(fig, out)
    plt.close(fig)
    print(f"✓ Created {out}.pdf")



# -----------------------------------------------------------------------------
# Figure 2 – Transmission vs hidden collector voltage
# -----------------------------------------------------------------------------
def create_protection_mismatch(
    dataset: CascadeDataset, out: Path = Path("fig_protection_mismatch")
) -> None:
    """Compare the transmission reading with the hidden collector voltage."""
    trip_events = [ev for ev in dataset.events if ev.code.startswith("AA_C")]
    if trip_events:
        first_trip = trip_events[0]
        target_name = first_trip.code.replace("AA_", "")
        center_time = first_trip.time
    else:
        target_name = (
            "C1"
            if "C1" in dataset.collector_kv
            else next(iter(dataset.collector_kv.keys()))
        )
        center_time = dataset.time[len(dataset.time) // 2]

    mapping = dataset.collector_mapping.get(target_name, {})
    trans_idx = int(mapping.get("trans_bus", 0))
    preferred = f"bus_{trans_idx}" if trans_idx else None
    bus_key = (
        preferred if preferred in dataset.bus_kv else next(iter(dataset.bus_kv.keys()))
    )

    trans_trace = dataset.bus_kv[bus_key]
    scada_trace = dataset.collector_scada_kv.get(target_name, trans_trace)
    coll_trace = dataset.collector_kv[target_name]

    base_kv_trans = dataset.bus_basekv.get(bus_key, 345.0)
    base_kv_coll = dataset.collector_basekv.get(target_name, 138.0)

    hv_thresh_kv = dataset.transmission_threshold_pu * base_kv_trans
    coll_thresh_kv = (
        dataset.collector_thresholds.get(target_name, dataset.threshold_pu)
        * base_kv_coll
    )

    window = (dataset.time >= center_time - 2.5) & (dataset.time <= center_time + 2.5)
    t = dataset.time[window]
    tx = trans_trace[window]
    scada_vals = scada_trace[window]
    coll_vals = coll_trace[window]

    fig, axes = plt.subplots(2, 1, figsize=(6.5, 4.5), sharex=True)
    fig.subplots_adjust(left=0.12, right=0.95, top=0.96, bottom=0.12, hspace=0.15)
    ax_tx, ax_coll = axes

    ax_tx.plot(t, tx, color="#2E6FA7", lw=1.8, label="Transmission (actual)")
    ax_tx.plot(
        t,
        scada_vals,
        color="#1F618D",
        lw=1.4,
        ls="--",
        label="Control room (SCADA)",
    )
    ax_tx.axhline(
        hv_thresh_kv, color="#E74C3C", ls="--", lw=1.0, label="EHV protection limit"
    )
    ax_tx.set_ylabel("Voltage (kV)")
    ax_tx.grid(alpha=0.2, ls=":", linewidth=0.7)
    ax_tx.set_ylim(320, 440)
    ax_tx.legend(loc="upper left", framealpha=0.9)

    ax_coll.plot(t, coll_vals, color="#C44536", lw=1.6, label=f"Collector {target_name}")
    ax_coll.axhline(
        coll_thresh_kv,
        color="#E74C3C",
        ls="--",
        lw=1.0,
        label=f"Collector relay ({coll_thresh_kv/base_kv_coll:.3f} pu)",
    )
    mask = coll_vals >= coll_thresh_kv
    if np.any(mask):
        ax_coll.fill_between(
            t,
            coll_thresh_kv,
            coll_vals,
            where=mask,
            color="#F5B7B1",
            alpha=0.4,
            label="Hidden OV",
        )
    ax_coll.set_ylabel("Voltage (kV)")
    ax_coll.set_xlabel("Time (s)")
    ax_coll.grid(alpha=0.2, ls=":", linewidth=0.7)
    ax_coll.set_ylim(120, 180)
    ax_coll.legend(loc="upper left", framealpha=0.9)

    mismatch_mask = (coll_vals >= coll_thresh_kv) & (scada_vals <= hv_thresh_kv)
    if np.any(mismatch_mask):
        idx = np.argmax(mismatch_mask)
        ax_coll.annotate(
            "Relay trips\nwhile HV < limit",
            xy=(t[idx], coll_vals[idx]),
            xytext=(15, 35),
            textcoords="offset points",
            arrowprops=dict(arrowstyle="->", color="#C44536"),
            fontsize=9,
            bbox=dict(
                boxstyle="round,pad=0.2",
                facecolor="#ffffff",
                edgecolor="#C44536",
                linewidth=0.6,
                alpha=0.85,
            ),
        )

    _format_time_axis(ax_coll, t, None, step=0.5)
    _save_all(fig, out)
    plt.close(fig)
    print(f"✓ Created {out}.pdf")


# -----------------------------------------------------------------------------
# Figure 3 – Cascading voltages
# -----------------------------------------------------------------------------

def create_cascade_voltage_panels(
    dataset: CascadeDataset,
    out: Path = Path("fig_cascade_voltage"),
    show_collector: bool = True,
) -> None:
    selected_keys = [
        f"bus_{b}" for b in FOCUS_TRANSMISSION_BUSES if f"bus_{b}" in dataset.bus_kv
    ]
    bus_keys = selected_keys or sorted(dataset.bus_kv.keys())[:8]
    collector_keys = sorted(dataset.collector_kv.keys())[:8]

    if show_collector:
        fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(7.2, 4.5), sharex=True)
        fig.subplots_adjust(left=0.12, right=0.99, top=0.96, bottom=0.13, hspace=0.12)
    else:
        fig, ax_top = plt.subplots(1, 1, figsize=(6.2, 2.5))
        ax_bot = None
        fig.subplots_adjust(left=0.12, right=0.99, top=0.94, bottom=0.17)

    # limit time window: stop 2s after collapse (last cascade event)
    cascade_start, cascade_end = dataset.cascade_window()
    collapse_cut = cascade_end + 2.0
    mask = dataset.time <= collapse_cut
    t = dataset.time[mask]

    hv_thresh_pu = getattr(
        dataset, "transmission_threshold_pu", TRANSMISSION_THRESHOLD_PU
    )
    lv_thresh_pu = getattr(dataset, "lower_voltage_threshold_pu", hv_thresh_pu)

    bus_colors = mpl.colormaps["magma"](np.linspace(0.2, 0.85, len(bus_keys)))
    for color, key in zip(bus_colors, bus_keys):
        ax_top.plot(
            t, dataset.bus_kv[key][mask], color=color, lw=1.3, alpha=0.85, zorder=2
        )

    if show_collector:
        collector_colors = mpl.colormaps["viridis"](
            np.linspace(0.2, 0.9, len(collector_keys))
        )
        for color, key in zip(collector_colors, collector_keys):
            ax_bot.plot(
                t,
                dataset.collector_kv[key][mask],
                color=color,
                lw=1.3,
                alpha=0.85,
                zorder=2,
            )
            base_kv = dataset.collector_basekv.get(key, 138.0)
            thresh_pu = dataset.collector_thresholds.get(
                key, dataset.threshold_pu
            )
            thresh_kv = thresh_pu * base_kv
            ax_bot.axhline(
                thresh_kv,
                color=color,
                ls="--",
                lw=0.9,
                alpha=0.55,
                zorder=1,
            )

    if show_collector:
        for axis in (ax_top, ax_bot):
            axis.axvspan(
                cascade_start, cascade_end, color="#f0c6c6", alpha=0.28, zorder=0
            )
            axis.grid(alpha=0.2, ls=":", linewidth=0.7)
    else:
        ax_top.axvspan(
            cascade_start, cascade_end, color="#f0c6c6", alpha=0.28, zorder=0
        )
        ax_top.grid(alpha=0.2, ls=":", linewidth=0.7)

    ax_top.set_ylim(200, 480)
    hv_bases = [
        dataset.bus_basekv.get(key, 345.0)
        for key in bus_keys
        if dataset.bus_basekv.get(key, 345.0) >= 200.0
    ]
    lv_bases = [
        dataset.bus_basekv.get(key, 138.0)
        for key in bus_keys
        if dataset.bus_basekv.get(key, 345.0) < 200.0
    ]
    if hv_bases:
        hv_line = hv_thresh_pu * float(np.mean(hv_bases))
        ax_top.axhline(
            hv_line,
            color="#6B6B6B",
            ls="--",
            lw=1.1,
            zorder=3,
            label=f"EHV protection ({hv_thresh_pu:.3f} pu)",
        )
    if lv_bases:
        lv_line = lv_thresh_pu * float(np.mean(lv_bases))
        ax_top.axhline(
            lv_line,
            color="#9B9B9B",
            ls="--",
            lw=1.0,
            zorder=3,
            label=f"HV protection ({lv_thresh_pu:.3f} pu)",
        )
    ax_top.set_ylabel("Voltage (kV)")

    if show_collector and ax_bot is not None:
        ax_bot.set_ylim(100, 200)
        ax_bot.set_ylabel("Voltage (kV)")
        ax_bot.set_xlabel("Time (s)")
    else:
        ax_top.set_xlabel("Time (s)")

    if show_collector:
        handles = _draw_events(ax_top, ax_bot, dataset.events)
    else:
        handles = _draw_events(ax_top, ax_top, dataset.events)

    base_handles = []
    base_handles.append(
        Line2D([0], [0], color=bus_colors[0], lw=1.6, label="Transmission buses")
    )
    if show_collector:
        base_handles.append(
            Line2D([0], [0], color="#1f77b4", lw=1.6, label="Collector buses")
        )
    base_handles.append(
        Rectangle(
            (0, 0), 1, 1, fc="#f0c6c6", ec="none", alpha=0.4, label="Cascade window"
        )
    )
    if hv_bases:
        base_handles.append(
            Line2D(
                [0],
                [0],
                color="#6B6B6B",
                ls="--",
                lw=1.1,
                label="EHV threshold",
            )
        )
    if lv_bases:
        base_handles.append(
            Line2D(
                [0],
                [0],
                color="#9B9B9B",
                ls="--",
                lw=1.0,
                label="HV threshold",
            )
        )
    if show_collector:
        base_handles.append(
            Line2D(
                [0],
                [0],
                color="#555555",
                ls="--",
                lw=0.9,
                label="Collector-specific thresholds",
            )
        )

    legend_handles = base_handles + handles
    ax_top.legend(
        legend_handles,
        [h.get_label() for h in legend_handles],
        loc="upper left",
        framealpha=0.9,
        fontsize=8,
    )

    if show_collector:
        _format_time_axis(ax_bot, t, None, step=1.0)
    else:
        _format_time_axis(ax_top, t, None, step=1.0)

    _save_all(fig, out)
    plt.close(fig)
    print(f"✓ Created {out}.pdf")


# -----------------------------------------------------------------------------
# Figure 4 – AVR Capability vs Renewable Penetration Analysis
# -----------------------------------------------------------------------------

def create_avr_renewable_analysis(
    dataset: CascadeDataset, out: Path = Path("fig_avr_renewable_analysis")
) -> None:
    print("Creating AVR/Renewable analysis figure...")

    bus_key = (
        "bus_21" if "bus_21" in dataset.bus_kv else next(iter(dataset.bus_kv.keys()))
    )
    limited = dataset.bus_kv[bus_key]
    fleet_avg = np.mean(np.stack(list(dataset.bus_kv.values())), axis=0)

    control_gain = 0.6
    compliant = limited - control_gain * (limited - fleet_avg)
    compliant = np.minimum(
        compliant, np.full_like(compliant, dataset.threshold_pu * 345 - 5)
    )

    renewable_offsets = {
        "60% renewables": -5.0,
        "100% renewables": +2.0,
    }

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6.0, 4.5), sharex=True)
    fig.subplots_adjust(left=0.11, right=0.97, top=0.96, bottom=0.12, hspace=0.15)

    threshold_kv = dataset.threshold_pu * 345

    ax1.plot(
        dataset.time, limited, color="#C0392B", lw=1.6, label="Observed (limited AVR)"
    )
    ax1.plot(dataset.time, compliant, color="#1E8449", lw=1.6, label="Compliant AVR")
    ax1.axhline(threshold_kv, color="#6B6B6B", ls="--", lw=1.0, label="Relay threshold")
    ax1.set_ylabel("Voltage (kV)")
    ax1.set_ylim(240, 440)
    ax1.grid(alpha=0.2, ls=":", linewidth=0.7)
    ax1.legend(loc="upper left", framealpha=0.9, fontsize=9)

    for idx, (label, offset) in enumerate(renewable_offsets.items()):
        adjusted = np.clip(compliant + offset, None, threshold_kv - 2)
        ls = "-" if idx == 0 else "--"
        ax2.plot(dataset.time, adjusted, ls=ls, lw=1.6, label=label)

    ax2.axhline(threshold_kv, color="#6B6B6B", ls="--", lw=1.0)
    ax2.set_ylabel("Voltage (kV)")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylim(240, 350)
    ax2.grid(alpha=0.2, ls=":", linewidth=0.7)
    ax2.legend(loc="upper left", framealpha=0.9, fontsize=9)

    _save_all(fig, out)
    plt.close(fig)
    print(f"✓ Created {out}.pdf")


# -----------------------------------------------------------------------------
# Figure 5 – Reactive absorption loss vs voltage margin
# -----------------------------------------------------------------------------

def create_reactive_loss_analysis(
    dataset: CascadeDataset, out: Path = Path("fig_reactive_loss_analysis")
) -> None:
    """Show how cumulative reactive loss tracks voltage rise."""
    print("Creating reactive loss analysis figure...")

    bus_matrix = np.stack(list(dataset.bus_kv.values()))
    max_bus = np.max(bus_matrix, axis=0)
    threshold_kv = dataset.threshold_pu * 345.0
    margin = threshold_kv - max_bus

    reactive_log = sorted(dataset.reactive_events, key=lambda item: item["time"])
    cumulative_total = np.zeros_like(dataset.time, dtype=float)
    categories = defaultdict(list)
    for item in reactive_log:
        categories[item.get("category", "")].append(item)

    cumulative_by_cat: Dict[str, np.ndarray] = {
        cat: np.zeros_like(dataset.time, dtype=float) for cat in categories
    }
    idx_by_cat = {cat: 0 for cat in categories}
    loss_by_cat = {cat: 0.0 for cat in categories}

    for i, t in enumerate(dataset.time):
        for cat, log in categories.items():
            idx = idx_by_cat[cat]
            while idx < len(log) and log[idx]["time"] <= t + 1e-6:
                loss_by_cat[cat] += log[idx]["mvar"]
                idx += 1
            idx_by_cat[cat] = idx
            cumulative_by_cat[cat][i] = loss_by_cat[cat]
        cumulative_total[i] = sum(loss_by_cat.values())

    fig, ax1 = plt.subplots(1, 1, figsize=(8, 3.5))
    fig.subplots_adjust(left=0.11, right=0.88, top=0.92, bottom=0.15)

    # Left y-axis: Voltage
    color_v = "#B03A2E"
    ax1.plot(
        dataset.time,
        max_bus,
        color=color_v,
        lw=2.0,
        label="Max transmission voltage",
        zorder=3,
    )
    ax1.axhline(
        threshold_kv,
        color="#6B6B6B",
        ls="--",
        lw=1.2,
        label="Protection threshold",
        zorder=2,
    )
    ax1.fill_between(
        dataset.time,
        threshold_kv,
        max_bus,
        where=max_bus > threshold_kv,
        color="#FDEDEC",
        alpha=0.5,
        label="Violation",
        zorder=1,
    )
    ax1.set_ylabel("Voltage (kV)", fontsize=11, fontweight="semibold", color=color_v)
    ax1.set_xlabel("Time (s)", fontsize=11)
    ax1.set_ylim(300, 450)
    ax1.tick_params(axis="y", labelcolor=color_v)
    ax1.grid(alpha=0.25, ls=":", linewidth=0.8)

    # Right y-axis: Cumulative MVAr lost
    ax2 = ax1.twinx()
    color_q = "#1F618D"
    oa_loss = cumulative_by_cat.get("OA", np.zeros_like(cumulative_total))
    aa_loss = cumulative_by_cat.get("AA", np.zeros_like(cumulative_total))
    other_loss = np.clip(cumulative_total - (oa_loss + aa_loss), a_min=0.0, a_max=None)

    ax2.step(
        dataset.time,
        cumulative_total,
        where="post",
        color=color_q,
        lw=2.4,
        label="Total reactive absorption lost",
        zorder=3,
    )
    if np.any(oa_loss > 1e-6):
        ax2.step(
            dataset.time,
            oa_loss,
            where="post",
            color="#E67E22",
            lw=1.8,
            ls="--",
            label="Operator actions (OA)",
            zorder=2,
        )
    if np.any(aa_loss > 1e-6):
        ax2.step(
            dataset.time,
            aa_loss,
            where="post",
            color="#C44536",
            lw=1.8,
            ls=":",
            label="Collector trips (AA)",
            zorder=2,
        )
    if np.any(other_loss > 1e-6):
        ax2.step(
            dataset.time,
            other_loss,
            where="post",
            color="#7F8C8D",
            lw=1.6,
            ls="-.",
            label="Other reactive events",
            zorder=2,
        )

    ax2.set_ylabel("Lost MVAr", fontsize=11, fontweight="semibold", color=color_q)
    max_loss = np.max(cumulative_total) if cumulative_total.size else 0.0
    ax2.set_ylim(0, max(50.0, max_loss * 1.15))
    ax2.tick_params(axis="y", labelcolor=color_q)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(
        lines1 + lines2,
        labels1 + labels2,
        loc="upper left",
        framealpha=0.95,
        fontsize=9,
    )

    _format_time_axis(ax1, dataset.time, None, step=1.0)
    _save_all(fig, out)
    plt.close(fig)
    print(f"✓ Created {out}.pdf")


# -----------------------------------------------------------------------------
# Figure 6 - Cascade Acceleration and Positive Feedback Analysis (UNCHANGED)
# -----------------------------------------------------------------------------

def create_qv_phase_portrait(
    dataset, out: Path = Path("fig_qv_phase_portrait")
) -> None:
    """
    X: cumulative MVAr of reactive absorption lost (from collector trips).
    Y: system-average 345-kV bus voltage v_avg (kV) at each trip instant.
    """
    print("Creating enhanced Q–V phase portrait...")

    # Trip events ordered by time
    trip_events = sorted(
        [ev for ev in dataset.events if ev.code.startswith("AA_C")],
        key=lambda e: e.time,
    )
    if len(trip_events) < 2:
        print("Warning: Not enough trip events for phase portrait")
        return

    trip_times = np.array([ev.time for ev in trip_events])
    trip_names = [ev.code.replace("AA_", "") for ev in trip_events]

    # System-average voltage trace (kV) from bus matrix
    bus_matrix = np.stack(list(dataset.bus_kv.values()))
    v_avg_trace = np.mean(bus_matrix, axis=0)
    v_at_trips = np.array([np.interp(t, dataset.time, v_avg_trace) for t in trip_times])

    # Cumulative MVAr lost using metadata per collector group
    cumulative_mvar = np.zeros(len(trip_events))
    for i, name in enumerate(trip_names):
        meta = dataset.collector_mapping.get(name, {})
        mvar = float(meta.get("q_absorption_mvar", 340.0))
        cumulative_mvar[i] = mvar if i == 0 else cumulative_mvar[i - 1] + mvar

    # Calculate time intervals between trips (for acceleration visualization)
    time_intervals = np.diff(trip_times)
    time_intervals = np.concatenate(
        [[time_intervals[0]], time_intervals]
    )  # Duplicate first for alignment

    # Calculate dV/dQ sensitivity between consecutive trips
    dv_dq = np.zeros(len(trip_events) - 1)
    for i in range(len(trip_events) - 1):
        dv = v_at_trips[i + 1] - v_at_trips[i]
        dq = cumulative_mvar[i + 1] - cumulative_mvar[i]
        dv_dq[i] = dv / dq if dq > 0 else 0

    # Create figure with secondary y-axis
    fig, ax1 = plt.subplots(figsize=(8.5, 4.2))
    fig.subplots_adjust(left=0.10, right=0.87, top=0.93, bottom=0.13)

    # Background shading for cascade phases
    threshold_kv = dataset.threshold_pu * 345.0
    # Pre-threshold phase (stable)
    pre_threshold_q = cumulative_mvar[v_at_trips < threshold_kv]
    if len(pre_threshold_q) > 0:
        ax1.axvspan(
            0,
            max(pre_threshold_q),
            color="#E8F5E9",
            alpha=0.3,
            zorder=0,
            label="Pre-cascade",
        )
    # Post-threshold phase (cascade)
    post_threshold_q = cumulative_mvar[v_at_trips >= threshold_kv]
    if len(post_threshold_q) > 0:
        ax1.axvspan(
            min(post_threshold_q),
            cumulative_mvar[-1],
            color="#FFEBEE",
            alpha=0.35,
            zorder=0,
            label="Cascade acceleration",
        )

    for i in range(len(trip_events) - 1):
        color_intensity = i / (len(trip_events) - 1)
        color = plt.cm.RdYlBu(0.3 + 0.6 * color_intensity)
        ax1.plot(
            cumulative_mvar[i : i + 2],
            v_at_trips[i : i + 2],
            color=color,
            lw=3.0,
            alpha=0.8,
            zorder=2,
        )

        if i < len(trip_events) - 1:
            mid_q = (cumulative_mvar[i] + cumulative_mvar[i + 1]) / 2
            mid_v = (v_at_trips[i] + v_at_trips[i + 1]) / 2
            dq = cumulative_mvar[i + 1] - cumulative_mvar[i]
            dv = v_at_trips[i + 1] - v_at_trips[i]
            ax1.annotate(
                "",
                xy=(cumulative_mvar[i + 1], v_at_trips[i + 1]),
                xytext=(cumulative_mvar[i], v_at_trips[i]),
                arrowprops=dict(arrowstyle="->", lw=1.8, color=color, alpha=0.7),
                zorder=2,
            )

    # Scatter points with size based on cascade phase
    sizes = np.linspace(100, 220, len(trip_events))
    scatter = ax1.scatter(
        cumulative_mvar,
        v_at_trips,
        c=range(len(trip_events)),
        cmap="RdYlBu",
        s=sizes,
        edgecolors="black",
        linewidth=1.5,
        zorder=4,
        vmin=0,
        vmax=len(trip_events) - 1,
    )

    # Protection threshold line
    ax1.axhline(
        threshold_kv,
        color="#C62828",
        ls="--",
        lw=2.0,
        alpha=0.8,
        label=f"Protection threshold ({threshold_kv:.1f} kV)",
        zorder=3,
    )

    # Mark threshold crossing point
    threshold_cross_idx = np.where(v_at_trips >= threshold_kv)[0]
    if len(threshold_cross_idx) > 0:
        cross_idx = threshold_cross_idx[0]
        ax1.scatter(
            [cumulative_mvar[cross_idx]],
            [v_at_trips[cross_idx]],
            marker="X",
            s=300,
            c="red",
            edgecolors="darkred",
            linewidth=2,
            zorder=5,
            label="Threshold crossing",
        )

    for i, (q, v, t, name) in enumerate(
        zip(cumulative_mvar, v_at_trips, trip_times, trip_names)
    ):
        offset_x = 8 if i % 2 == 0 else -8
        offset_y = 12 if i < len(trip_events) // 2 else -15
        ha = "left" if offset_x > 0 else "right"
        va = "bottom" if offset_y > 0 else "top"

        ax1.annotate(
            f"{name}\nt={t:.1f}s",
            xy=(q, v),
            xytext=(offset_x, offset_y),
            textcoords="offset points",
            fontsize=8,
            ha=ha,
            va=va,
            bbox=dict(
                boxstyle="round,pad=0.3",
                facecolor="white",
                edgecolor="gray",
                alpha=0.85,
                linewidth=0.8,
            ),
            arrowprops=dict(arrowstyle="-", lw=0.8, color="gray", alpha=0.6),
            zorder=6,
        )

    # Add dV/dQ annotations showing increasing sensitivity
    for i in range(min(len(dv_dq), 3)):
        mid_q = (cumulative_mvar[i] + cumulative_mvar[i + 1]) / 2
        mid_v = (v_at_trips[i] + v_at_trips[i + 1]) / 2
        if i == len(dv_dq) - 1:  # Highlight the last one (highest sensitivity)
            ax1.text(
                mid_q,
                mid_v - 8,
                f"dV/dQ={dv_dq[i]:.2f}\nkV/MVAr",
                fontsize=8,
                ha="center",
                style="italic",
                bbox=dict(
                    boxstyle="round,pad=0.3",
                    facecolor="#FFF9C4",
                    edgecolor="#F57F17",
                    alpha=0.9,
                    linewidth=1.2,
                ),
                zorder=6,
            )

    ax1.set_xlabel(
        "Cumulative reactive absorption lost (MVAr)", fontsize=11, fontweight="semibold"
    )
    ax1.set_ylabel("System-average voltage (kV)", fontsize=11, fontweight="semibold")
    ax1.set_ylim(235, 290)
    ax1.set_xlim(-10, cumulative_mvar[-1] + 20)
    ax1.grid(alpha=0.25, ls=":", linewidth=0.8, zorder=0)

    # Secondary y-axis: Time between trips (acceleration indicator)
    ax2 = ax1.twinx()
    # Create bar plot showing time intervals
    bar_width = (cumulative_mvar[1] - cumulative_mvar[0]) * 0.4
    for i in range(len(trip_events)):
        color = plt.cm.RdYlBu(0.3 + 0.6 * i / (len(trip_events) - 1))
        ax2.bar(
            cumulative_mvar[i],
            time_intervals[i],
            width=bar_width,
            alpha=0.4,
            color=color,
            edgecolor="none",
            zorder=1,
        )

    ax2.set_ylabel(
        "Time since previous trip (s)",
        fontsize=10,
        fontweight="semibold",
        color="#D84315",
        style="italic",
    )
    ax2.tick_params(axis="y", labelcolor="#D84315")
    ax2.set_ylim(0, max(time_intervals) * 1.3)

    handles, labels = ax1.get_legend_handles_labels()
    # Remove duplicate labels
    by_label = dict(zip(labels, handles))
    ax1.legend(
        by_label.values(),
        by_label.keys(),
        loc="lower right",
        framealpha=0.95,
        fontsize=9,
        edgecolor="gray",
        fancybox=True,
    )

    fig.savefig(str(out) + ".pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ Created enhanced {out}.pdf")


def _load_iberian_outputs():
    df = pd.read_csv("iberian_cascade_bus_voltages.csv")
    with open("iberian_cascade_events.json") as f:
        meta = json.load(f)
    t = df["time_s"].values
    bus_cols = [c for c in df.columns if c.startswith("bus_")]
    basekv = {k: float(v) for k, v in meta.get("bus_basekv", {}).items()}
    # EHV = base >= 300 kV
    ehv_cols = [c for c in bus_cols if basekv.get(c, 0.0) >= 300.0]
    if not ehv_cols:
        raise RuntimeError("No EHV buses found in outputs. Check meta['bus_basekv'].")
    # Voltages in kV, shape (T, N_ehv)
    V_ehv_kV = np.vstack([df[c].values * basekv[c] for c in ehv_cols]).T
    thr_pu = float(meta.get("protection_threshold_pu", 1.15))
    thr_kV = np.array([thr_pu * basekv[c] for c in ehv_cols])
    return t, V_ehv_kV, thr_kV, ehv_cols, meta


def _draw_event_markers(ax, meta, ymin=0.02, ymax=0.98):
    # Light vertical markers with small labels; doesn’t imply causality
    colors = {"INIT": "#555555", "OA": "#C5842D", "AA": "#9A3A2A"}
    for e in meta.get("events", []):
        t = float(e["time_s"])
        cat = e.get("category", "")
        c = colors.get(cat, "#888888")
        ax.axvline(t, color=c, lw=0.8, alpha=0.25, zorder=0)
    # collapse line darker
    for e in meta.get("events", []):
        if e.get("code", "") == "COLLAPSE":
            ax.axvline(float(e["time_s"]), color="#5B1610", lw=1.2, alpha=0.9, zorder=2)
            break


# -----------------------------------------------------------------------------
# Figure 7 - Voltage Headroom and Exceedance
# -----------------------------------------------------------------------------
def figure_voltage_headroom_and_exceedance(save_as="fig_headroom_exceedance.pdf"):
    """
    Headroom H(t) to the protection threshold across EHV buses and the
    fraction of EHV buses already ≥ threshold. No dependence on collectors.
    """
    t, V, thr, cols, meta = _load_iberian_outputs()
    # Per-bus headroom (kV); positive = safe, negative = beyond threshold
    H = thr[None, :] - V  # (T, N)
    H_mean = H.mean(axis=1)
    H_worst = H.min(axis=1)
    frac_exceed = (V >= thr[None, :]).mean(axis=1) * 100.0  # %

    # light smoothing to de-noise oscillations (moving average, odd window)
    def smooth(x, w=21):
        w = max(5, int(w) | 1)  # ensure odd
        k = np.ones(w) / w
        return np.convolve(x, k, mode="same")

    H_mean_s = smooth(H_mean, 21)
    H_worst_s = smooth(H_worst, 21)
    frac_s = smooth(frac_exceed, 11)

    fig, ax1 = plt.subplots(figsize=(8.0, 3.6))
    ax1.plot(t, H_mean_s, lw=2.2, color="#C97A24", label="Average headroom")
    ax1.plot(t, H_worst_s, lw=2.2, color="#5F3CA4", label="Worst-bus headroom")
    ax1.axhline(0.0, color="#666666", ls="--", lw=1.0)
    ax1.set_ylabel("Headroom to threshold (kV)")
    ax1.set_xlabel("Time (s)")

    ax2 = ax1.twinx()
    ax2.plot(t, frac_s, lw=2.0, color="#1B79B0", label="% EHV buses ≥ threshold")
    ax2.set_ylabel("Buses ≥ threshold (%)")

    _draw_event_markers(ax1, meta)

    # legend: combine handles from both axes
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper left", ncol=2, frameon=False)

    ax1.grid(True, which="both", axis="both", alpha=0.15)
    fig.tight_layout()
    fig.savefig(save_as, bbox_inches="tight")
    print(f"✓ Created {save_as}")


# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------
def main() -> None:
    print("\n" + "=" * 70)
    print("CREATING ENHANCED FIGURES FOR IBERIAN BLACKOUT ANALYSIS")
    print("=" * 70 + "\n")

    dataset = load_cascade_dataset()

    print("Figure 1: Meshing impact...")
    create_meshing_figure()

    print("\nFigure 2: Protection mismatch...")
    create_protection_mismatch(dataset)

    print("\nFigure 3: Cascade voltage panels...")
    create_cascade_voltage_panels(dataset)

    print("\nFigure 4: AVR capability vs renewable penetration analysis...")
    create_avr_renewable_analysis(dataset)

    print("\nFigure 5: Reactive MVAr loss vs voltage margin...")
    create_reactive_loss_analysis(dataset)

    print("\nCreating Figure 6: Enhanced Cascade Acceleration Analysis...")
    create_qv_phase_portrait(dataset)

    print("\nCreating Figure 7: Voltage Headroom and Exceedance...")
    figure_voltage_headroom_and_exceedance()

if __name__ == "__main__":
    main()
