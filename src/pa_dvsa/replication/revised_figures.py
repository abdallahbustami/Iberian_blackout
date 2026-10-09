"""Diagnostic figures of the base-state runs.

    python -m pa_dvsa.replication.revised_figures
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
INK = "#0b0b0b"
INK2 = "#52514e"
CONTEXT = "#bdbcb5"
GRID = "#e6e5e0"
SURFACE = "#ffffff"
PICKUP = "#52514e"

VARIANT_LABELS = {
    "baseline": "baseline",
    "no_pre_cascade_actions": "no pre-cascade actions",
    "no_ov_protection": "no plant OV protection",
    "delayed_protection": "OV1 delay 2.0 s",
    "preserved_absorption": "absorption preserved at trip",
    "voltage_mode_res": "volt-var RES",
    "margin_0p25pct": "initial margin 0.0025 pu",
    "margin_0p5pct": "initial margin 0.005 pu",
    "margin_2pct": "initial margin 0.02 pu",
    "pf_0p98": "PF 0.98 (13 blocks)",
    "fixed_tap_0p95": "fixed tap 0.95",
    "margin_0p25pct_preserved": "margin 0.0025 pu + absorption preserved",
}


def _style() -> None:
    import matplotlib as mpl

    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 8.5,
        "axes.labelsize": 8.5,
        "axes.titlesize": 8.5,
        "xtick.labelsize": 7.8,
        "ytick.labelsize": 7.8,
        "legend.fontsize": 7.2,
        "axes.edgecolor": INK2,
        "axes.labelcolor": INK,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "axes.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "grid.linestyle": "-",
        "lines.linewidth": 1.4,
        "lines.solid_capstyle": "round",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.facecolor": SURFACE,
        "figure.facecolor": SURFACE,
    })


def _read_csv(path: Path) -> dict[str, np.ndarray]:
    with path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    out: dict[str, np.ndarray] = {}
    if not rows:
        return out
    for key in rows[0]:
        vals = []
        for r in rows:
            v = r.get(key, "")
            try:
                vals.append(float(v) if v not in ("", None) else math.nan)
            except ValueError:
                vals.append(math.nan)
        out[key] = np.asarray(vals)
    return out


def _load_run(d: Path) -> dict[str, Any]:
    return {
        "dir": d,
        "col": _read_csv(d / "collector_traces.csv"),
        "sys": _read_csv(d / "system_traces.csv"),
        "events": json.loads((d / "events.json").read_text(encoding="utf-8")),
        "summary": json.loads((d / "summary.json").read_text(encoding="utf-8")),
        "design": json.loads((d / "case_design.json").read_text(encoding="utf-8")),
    }


def _save(fig: Any, out_dir: Path, stem: str, formats: Iterable[str]) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for fmt in formats:
        p = out_dir / f"{stem}.{fmt}"
        fig.savefig(p, bbox_inches="tight", pad_inches=0.03, dpi=300 if fmt == "png" else None)
        paths.append(p)
    return paths


def _block_names(run: dict[str, Any]) -> list[str]:
    return [b["name"] for b in run["design"]["blocks"]]


def _mark_actions(ax: Any, run: dict[str, Any], *, label: bool) -> None:
    actions = run["design"].get("pre_cascade_actions", [])
    by_time: dict[float, list[str]] = {}
    for a in actions:
        by_time.setdefault(round(float(a["start_s"]), 6), []).append(a["code"])
        if float(a["duration_s"]) > 0:
            ax.axvspan(a["start_s"], a["start_s"] + a["duration_s"], color=GRID, alpha=0.45, lw=0, zorder=0)
        else:
            ax.axvline(a["start_s"], color=CONTEXT, lw=0.8, zorder=0)
    if label:
        y0, y1 = ax.get_ylim()
        for t0, codes in by_time.items():
            dur = max(float(a["duration_s"]) for a in actions if round(float(a["start_s"]), 6) == t0)
            text = "+".join(codes) + (" ramps" if dur > 0 else "")
            ax.text(t0 + 0.25, y0 + 0.03 * (y1 - y0), text, rotation=90, va="bottom", ha="left", fontsize=6.2,
                    color=INK2, zorder=5)


def fig_timeseries(run: dict[str, Any], out_dir: Path, formats: Iterable[str], stem: str = "fig_replica_timeseries") -> list[Path]:
    import matplotlib.pyplot as plt

    _style()
    col, sys_ = run["col"], run["sys"]
    t = col["time_s"]
    ts = sys_["time_s"]
    names = _block_names(run)
    trips = [e for e in run["events"] if e["category"] == "plant_trip"]
    trip_order = [e["code"].replace("TRIP_", "") for e in trips]
    color_of = {n: SERIES[k % len(SERIES)] for k, n in enumerate(trip_order)}
    stages = {s["name"]: s for s in run["design"]["plant_protection"]}
    ov1 = stages.get("OV1")

    fig, axes = plt.subplots(4, 1, figsize=(7.0, 6.6), sharex=True,
                             gridspec_kw={"height_ratios": [1.35, 1.0, 0.8, 0.8], "hspace": 0.12})
    ax_c, ax_h, ax_f, ax_l = axes
    for n in names:
        if n in color_of:
            continue
        ax_c.plot(t, col[f"{n}_v_pu"], color=CONTEXT, lw=0.9, zorder=1)
        ax_h.plot(t, col[f"{n}_hv_v_pu"], color=CONTEXT, lw=0.9, zorder=1)
    for n in trip_order:
        c = color_of[n]
        ax_c.plot(t, col[f"{n}_v_pu"], color=c, lw=1.6, zorder=3, label=f"{n} (tripped)")
        ax_h.plot(t, col[f"{n}_hv_v_pu"], color=c, lw=1.4, zorder=3)
    for e in trips:
        n = e["code"].replace("TRIP_", "")
        ax_c.plot(e["time_s"], e["measured"]["v_pu"], "o", ms=5, color=color_of[n], mec=SURFACE, mew=1.2, zorder=4)
    if ov1:
        ax_c.axhline(ov1["pickup"], color=PICKUP, lw=0.9, ls=(0, (4, 3)), zorder=2)
        ax_c.text(t[-1], ov1["pickup"], f"OV1 pickup {ov1['pickup']:.2f} pu, {ov1['delay_s']:.1f} s ", va="bottom",
                  ha="right", fontsize=6.6, color=INK2)
    ax_c.plot([], [], color=CONTEXT, lw=0.9, label="other protected plants")
    ax_c.set_ylabel("plant-side V (pu)")
    ax_c.legend(loc="lower right", ncol=2, frameon=False)
    ax_h.set_ylabel("upstream HV V (pu)")

    f = sys_["affected_island_f_eq_hz"]
    ax_f.plot(ts, f, color=SERIES[0], lw=1.4)
    ax_f.set_ylabel("area f (Hz, 50-Hz eq.)")
    fr = np.nanmax(np.abs(f - 50.0)) if np.isfinite(f).any() else 0.05
    ax_f.set_ylim(50 - max(0.05, 1.3 * fr), 50 + max(0.05, 1.3 * fr))

    ax_l.plot(ts, sys_["cum_lost_p_mw"], color=SERIES[0], lw=1.4, label="generation lost (MW)")
    ax_l.plot(ts, sys_["cum_lost_q_absorption_mvar"], color=SERIES[1], lw=1.4, ls=(0, (5, 2)),
              label="reactive absorption lost (MVAr)")
    ax_l.set_ylabel("cumulative loss")
    ax_l.set_xlabel("time (s)")
    ax_l.legend(loc="lower right", frameon=False)
    top = max(50.0, float(np.nanmax(sys_["cum_lost_p_mw"])) * 1.15)
    ax_l.set_ylim(-0.03 * top, top)
    for k, ax in enumerate(axes):
        _mark_actions(ax, run, label=(k == 0))
        for e in trips:
            ax.axvline(e["time_s"], color=color_of[e["code"].replace("TRIP_", "")], lw=0.6, alpha=0.5, zorder=0)
    ax_c.set_xlim(float(t[0]), float(t[-1]))
    s = run["summary"]
    vname = str(s["variant"]).replace("sensitivity:", "")
    ax_c.set_title(
        f"{VARIANT_LABELS.get(vname, vname)}: {s['block_trips']} plant trip(s), "
        f"{s['generator_trips']} generator trip(s), UFLS stages {s['ufls_stages_operated']}, "
        f"separation {'yes' if s['separation'] else 'no'}, collapse {'yes' if s['affected_area_collapse'] else 'no'}",
        loc="left", color=INK, fontsize=8.0)
    paths = _save(fig, out_dir, stem, formats)
    plt.close(fig)
    return paths


def fig_condition_buildup(buildup: list[dict[str, Any]], out_dir: Path, formats: Iterable[str],
                          baseline: dict[str, Any] | None = None) -> list[Path]:
    import matplotlib.pyplot as plt

    _style()
    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    x = np.arange(len(buildup))
    for k, row in enumerate(buildup):
        vals = np.asarray(list(row["collector_v_pu"].values()))
        ax.plot(np.full(vals.size, k) + np.linspace(-0.18, 0.18, vals.size), vals, "o", ms=2.6, color=CONTEXT, zorder=1)
    ymax = [r["max_collector_v_pu"] for r in buildup]
    ax.plot(x, ymax, "-o", color=SERIES[0], ms=4.5, mec=SURFACE, mew=1.0, lw=1.6, zorder=3,
            label="power flow (armed state), most exposed plant")
    if baseline is not None:
        st = baseline["sys"]
        ts, vmax = st["time_s"], st["collector_vmax_online_pu"]
        acts = baseline["design"].get("pre_cascade_actions", [])
        ends = []
        for k, a in enumerate(acts):
            nxt = [b["start_s"] for b in acts if b["start_s"] > a["start_s"] + 1e-9]
            end = a["start_s"] + a["duration_s"] if a["duration_s"] > 0 else (min(nxt) if nxt else a["start_s"] + 1.0)
            ends.append(end - 0.05)
        t_pts = [acts[0]["start_s"] - 0.05] + ends if acts else [0.0]
        vals = [float(np.interp(tp, ts, vmax)) for tp in t_pts]
        ax.plot(x[: len(vals)], vals, "--s", color=SERIES[1], ms=4, mec=SURFACE, mew=0.8, lw=1.4, zorder=3,
                label="TDS baseline, end of each action window")
        trips = [e for e in baseline["events"] if e["category"] == "plant_trip"]
        if trips:
            t1 = trips[0]["time_s"] - 0.03
            v1 = float(np.interp(t1, ts, vmax))
            ax.annotate(f"TDS {v1:.4f} before the first trip (t={trips[0]['time_s']:.1f} s)",
                        (x[-1], v1), textcoords="offset points", xytext=(-4, 6), ha="right", fontsize=6.2, color=INK2)
    ax.axhline(1.10, color=PICKUP, lw=0.9, ls=(0, (4, 3)))
    ax.text(x[0] - 0.3, 1.10, "OV1 pickup", va="bottom", fontsize=6.6, color=INK2)
    ax.set_xticks(x, ["base"] + [r["condition"].split(" ")[1] for r in buildup[1:]])
    ax.set_ylabel("plant-side V (pu)")
    ax.set_xlabel("cumulative pre-cascade condition")
    ax.plot([], [], "o", ms=2.6, color=CONTEXT, label="all 16 plants")
    ax.legend(loc="lower right", frameon=False)
    for k, v in enumerate(ymax):
        ax.annotate(f"{v:.4f}", (k, v), textcoords="offset points", xytext=(0, -11), ha="center", fontsize=6.2, color=INK2)
    paths = _save(fig, out_dir, "fig_replica_condition_buildup", formats)
    plt.close(fig)
    return paths


def fig_outcomes(rows: list[dict[str, Any]], out_dir: Path, formats: Iterable[str]) -> list[Path]:
    import matplotlib.pyplot as plt

    _style()
    labels = [VARIANT_LABELS.get(r["name"], r["name"]) for r in rows]
    y = np.arange(len(rows))[::-1]
    metrics = [
        ("block_trips", "plant trips (of 16)", "{:.0f}"),
        ("lost_block_p_mw", "MW lost", "{:.0f}"),
        ("lost_block_q_absorption_mvar", "MVAr absorption lost", "{:.0f}"),
        ("max_collector_v_pu", "max plant-side V (pu)", "{:.3f}"),
    ]
    fig, axes = plt.subplots(1, len(metrics), figsize=(7.1, 0.28 * len(rows) + 0.9), sharey=True,
                             gridspec_kw={"wspace": 0.12})
    for ax, (key, title, fmt) in zip(axes, metrics):
        vals = np.asarray([float(r[key]) if r[key] not in ("", None) else math.nan for r in rows])
        if key == "max_collector_v_pu":
            ax.axvline(1.10, color=PICKUP, lw=0.9, ls=(0, (4, 3)))
            ax.plot(vals, y, "o", ms=5, color=SERIES[0], mec=SURFACE, mew=1.0, zorder=3)
            lo = float(np.nanmin(vals)) - 0.004
            ax.set_xlim(lo, max(1.105, float(np.nanmax(vals)) + 0.004))
        else:
            ax.barh(y, vals, height=0.55, color=SERIES[0], edgecolor=SURFACE, linewidth=0.8)
            ax.set_xlim(0, max(1.0, float(np.nanmax(vals))) * 1.25)
        for yy, v in zip(y, vals):
            if np.isfinite(v):
                ax.annotate(fmt.format(v), (v, yy), textcoords="offset points", xytext=(5, 0), va="center",
                            ha="left", fontsize=6.4, color=INK2)
        ax.set_title(title, loc="left")
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(y, labels)
    n_var = sum(1 for r in rows if r["kind"] == "variant")
    if 0 < n_var < len(rows):
        for ax in axes:
            ax.axhline(y[n_var - 1] - 0.5, color=INK2, lw=0.6)
    paths = _save(fig, out_dir, "fig_replica_variant_outcomes", formats)
    plt.close(fig)
    return paths


def fig_variant_envelopes(runs: dict[str, dict[str, Any]], out_dir: Path, formats: Iterable[str]) -> list[Path]:
    import matplotlib.pyplot as plt

    _style()
    fig, ax = plt.subplots(figsize=(7.0, 2.6))
    styles = ["-", (0, (5, 2)), (0, (1.5, 1.5)), (0, (6, 2, 1.5, 2)), (0, (3, 1)), (0, (8, 2))]
    for k, (name, run) in enumerate(runs.items()):
        s = run["sys"]
        ax.plot(s["time_s"], s["collector_vmax_online_pu"], color=SERIES[k % len(SERIES)], ls=styles[k % len(styles)],
                lw=1.4, label=f"{VARIANT_LABELS.get(name, name)} ({run['summary']['block_trips']} trips)")
    ax.axhline(1.10, color=PICKUP, lw=0.9, ls=(0, (4, 3)))
    ax.set_ylabel("max online plant-side V (pu)")
    ax.set_xlabel("time (s)")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=3, frameon=False)
    paths = _save(fig, out_dir, "fig_replica_variant_envelopes", formats)
    plt.close(fig)
    return paths


def make_all(root: Path, out_dir: Path, *, formats: Iterable[str] = ("pdf", "png")) -> list[Path]:
    formats = tuple(formats)
    written: list[Path] = []
    runs: dict[str, dict[str, Any]] = {}
    for name in ("baseline", "no_pre_cascade_actions", "no_ov_protection", "delayed_protection",
                 "preserved_absorption", "voltage_mode_res"):
        d = root / name
        if (d / "summary.json").exists():
            runs[name] = _load_run(d)
    if "baseline" in runs:
        written += fig_timeseries(runs["baseline"], out_dir, formats)
    for name, run in runs.items():
        if name != "baseline":
            written += fig_timeseries(run, out_dir, formats, stem=f"fig_replica_timeseries_{name}")
    sens_root = root / "sensitivity"
    if sens_root.exists():
        for d in sorted(sens_root.iterdir()):
            if (d / "summary.json").exists():
                written += fig_timeseries(_load_run(d), out_dir, formats, stem=f"fig_replica_timeseries_sens_{d.name}")
    if runs:
        written += fig_variant_envelopes(runs, out_dir, formats)
    summary_path = root / "variants_summary.json"
    if summary_path.exists():
        rows = json.loads(summary_path.read_text(encoding="utf-8"))
        written += fig_outcomes(rows, out_dir, formats)
    buildup_path = root / "condition_buildup.json"
    if buildup_path.exists():
        written += fig_condition_buildup(json.loads(buildup_path.read_text(encoding="utf-8")), out_dir, formats,
                                         baseline=runs.get("baseline"))
    return written


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[3] / "results" / "revised" / "activsg2000")
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--formats", default="pdf,png")
    a = p.parse_args(argv)
    fmts = tuple(x.strip() for x in a.formats.split(",") if x.strip())
    for path in make_all(a.root, a.out or (a.root / "figures"), formats=fmts):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
