"""Variants of the base state with fewer synchronous units online and smaller initial margins.

    python -m pa_dvsa.replication.commitment_study --jobs 5
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Sequence


from .activsg2000_engine import DEFAULT_OUT_ROOT, ReplicaConfig, _jsonable, _write_csv, run_replica

LEVELS: dict[str, float | None] = {"stock": None, "C2.0": 2.0, "C1.5": 1.5}
MARGINS: tuple[float, ...] = (0.0025, 0.005, 0.01, 0.02)
FAST_MARGIN = 0.01
OUT = DEFAULT_OUT_ROOT / "commitment_sweep"


def level_config(level: str, margin: float, **overrides: Any) -> ReplicaConfig:
    cfg = ReplicaConfig(commitment_c=LEVELS[level], dispatch_rule="must_run", tap_margin_pu=float(margin),
                        variant=f"commitment:{level}:m{margin:g}")
    return replace(cfg, **overrides) if overrides else cfg


def _mdir(margin: float) -> str:
    return f"m{margin:g}".replace(".", "p")


def task_list(out: Path = OUT) -> list[tuple[str, str, float, str, dict[str, Any]]]:
    tasks = []
    for level in LEVELS:
        for margin in MARGINS:
            d = out / level / _mdir(margin)
            tasks.append((str(d / "base"), level, margin, "base", {}))
            tasks.append((str(d / "cf"), level, margin, "cf", {"preserved_absorption": True}))
            tasks.append((str(d / "np"), level, margin, "np", {"plant_ov_enabled": False, "shadow_plant_ov": True}))
        d = out / level / _mdir(FAST_MARGIN)
        tasks.append((str(d / "fast_base"), level, FAST_MARGIN, "fast_base", {"plant_ov_mode": "fast"}))
        tasks.append((str(d / "fast_cf"), level, FAST_MARGIN, "fast_cf",
                      {"plant_ov_mode": "fast", "preserved_absorption": True}))
        tasks.append((str(out / level / "control"), level, FAST_MARGIN, "control", {"pre_cascade_enabled": False}))
    return tasks


def _run_task(task: tuple[str, str, float, str, dict[str, Any]]) -> dict[str, Any]:
    out_dir, level, margin, kind, overrides = task
    t0 = time.time()
    cfg = level_config(level, margin, **overrides)
    cfg = replace(cfg, variant=f"{cfg.variant}:{kind}")
    run_replica(cfg, out_dir=Path(out_dir), quiet=True)
    return {"out_dir": out_dir, "level": level, "margin": margin, "kind": kind, "wall_s": time.time() - t0}


def _load(d: Path) -> dict[str, Any]:
    return json.loads((d / "summary.json").read_text(encoding="utf-8"))


def _neighbour(d: Path) -> list[dict[str, Any]]:
    import csv

    from .activsg2000_engine import neighbour_rise_from_traces

    with (d / "collector_traces.csv").open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    design = json.loads((d / "case_design.json").read_text(encoding="utf-8"))
    events = json.loads((d / "events.json").read_text(encoding="utf-8"))
    return neighbour_rise_from_traces(rows, [b["name"] for b in design["blocks"]], events)


def classify(base: dict[str, Any], cf: dict[str, Any], npr: dict[str, Any], mode: str) -> list[dict[str, Any]]:
    cf_trip = {e["block"]: e["t"] for e in cf["block_trip_sequence"]}
    out = []
    for e in base["block_trip_sequence"]:
        blk = e["block"]
        t_np = npr["per_block"][blk].get(f"shadow_{mode}_operate_s")
        if t_np is not None:
            cls = "action-driven"
        elif blk in cf_trip:
            cls = "propagated (MW loss)"
        else:
            cls = "propagated (absorption loss)"
        out.append({"block": blk, "t": e["t"], "class": cls, "t_no_trip_trajectory": t_np,
                    "advanced_by_s": (t_np - e["t"]) if t_np is not None else None,
                    "t_counterfactual": cf_trip.get(blk), "v_pu": e["v_pu"], "p_mw": e["p_mw"],
                    "lost_q_absorption_mvar": e["lost_q_absorption_mvar"]})
    return out


def _trip_label(c: dict[str, Any]) -> str:
    extra = ""
    if c["class"] == "action-driven" and c["advanced_by_s"] is not None and c["advanced_by_s"] > 0.05:
        extra = f"; {c['t_no_trip_trajectory']:.2f} s without earlier trips"
    return f"{c['t']:.2f} {c['block']} [{c['class']}{extra}]"


def summarize_study(out: Path = OUT) -> list[dict[str, Any]]:
    rows = []
    for level in LEVELS:
        ctrl = _load(out / level / "control")
        for margin in MARGINS:
            d = out / level / _mdir(margin)
            npr = _load(d / "np")
            relay_sets = [("standard", "base", "cf")]
            if abs(margin - FAST_MARGIN) < 1e-12:
                relay_sets.append(("fast", "fast_base", "fast_cf"))
            for mode, bk, ck in relay_sets:
                base, cf = _load(d / bk), _load(d / ck)
                cls = classify(base, cf, npr, mode)
                nb = _neighbour(d / bk)
                cm = base["commitment"]
                rows.append({
                    "level": level, "commitment_c": LEVELS[level], "margin_pu": margin, "relay": mode,
                    "area_online_units": cm["area_online_sync_units"],
                    "area_online_mva": round(cm["area_online_sync_mva"], 1),
                    "area_units_with_exciter": cm["area_online_units_with_exciter"],
                    "gsu_tap": base["gsu_tap"], "initial_margin_pu": base["initial_margin_min_pu"],
                    "tds_ok": base["tds_ok"],
                    "plant_trips": base["block_trips"],
                    "action_driven": sum(1 for c in cls if c["class"] == "action-driven"),
                    "propagated_mw_loss": sum(1 for c in cls if c["class"] == "propagated (MW loss)"),
                    "propagated_absorption_loss": sum(1 for c in cls if c["class"] == "propagated (absorption loss)"),
                    "trip_sequence": "; ".join(_trip_label(c) for c in cls) or "-",
                    "cf_plant_trips": cf["block_trips"],
                    "np_shadow_operations": sum(1 for b in npr["per_block"].values()
                                                if b.get(f"shadow_{mode}_operate_s") is not None),
                    "lost_p_mw": round(base["lost_block_p_mw"], 1),
                    "lost_q_absorption_mvar": round(base["lost_block_q_absorption_mvar"], 1),
                    "generator_trips": base["generator_trips"],
                    "generator_trip_triggers": ",".join(base["generator_trip_triggers"]),
                    "other_plant_trips": base["other_plant_trips"],
                    "ufls_stages": base["ufls_stages_operated"], "separation": base["separation"],
                    "collapse": base["affected_area_collapse"],
                    "max_plant_v_pu": base["max_collector_v_pu"], "max_zone_hv_v_pu": base["max_zone_hv_v_pu"],
                    "max_area_v_pu": base["max_area_v_pu"], "max_gen_terminal_v_pu": base["max_gen_terminal_v_pu"],
                    "f_eq_min_hz": base["affected_f_eq_hz_min"], "max_coi_angle_change_deg": base["max_coi_angle_change_deg"],
                    "max_neighbour_dv_settle_pu": max((x["max_dv_settle_pu"] for x in nb if x["max_dv_settle_pu"] is not None), default=None),
                    "max_neighbour_dv_peak_pu": max((x["max_dv_peak_pu"] for x in nb if x["max_dv_peak_pu"] is not None), default=None),
                    "neighbour_rise_per_trip": "; ".join(
                        f"{x['trip']}: +{x['max_dv_settle_pu']:.4f}/{x['max_dv_peak_pu']:.4f} ({x['max_dv_peak_block']})"
                        for x in nb if x["max_dv_settle_pu"] is not None) or "-",
                    "control_run_max_dev_pu": max(v for v in ctrl["control_flatness"].values() if v is not None),
                    "control_run_events": len(json.loads((out / level / "control" / "events.json").read_text())),
                    "classification": cls,
                })
    return rows


def write_tables(rows: list[dict[str, Any]], out: Path = OUT) -> None:
    flat = [{k: v for k, v in r.items() if k != "classification"} for r in rows]
    _write_csv(out / "commitment_margin_summary.csv", flat)
    (out / "commitment_margin_summary.json").write_text(json.dumps(_jsonable(rows), indent=2) + "\n", encoding="utf-8")
    lines = ["| level | online units / MVA / with exciter | margin | relay | trips | action / MW-prop. / abs.-prop. | "
             "lost MW / MVAr | gen trips | UFLS | sep. | collapse | max V plant | max V gen term. | f_min Hz-eq | "
             "max neighbour dV settle/peak |", "|" + "---|" * 15]
    for r in rows:
        lines.append(
            f"| {r['level']} | {r['area_online_units']} / {r['area_online_mva']:.0f} / {r['area_units_with_exciter']} | "
            f"{r['margin_pu']:g} | {r['relay']} | {r['plant_trips']} | {r['action_driven']} / {r['propagated_mw_loss']} / "
            f"{r['propagated_absorption_loss']} | {r['lost_p_mw']:.0f} / {r['lost_q_absorption_mvar']:.0f} | "
            f"{r['generator_trips']} | {r['ufls_stages']} | {'yes' if r['separation'] else 'no'} | "
            f"{'yes' if r['collapse'] else 'no'} | {r['max_plant_v_pu']:.4f} | {r['max_gen_terminal_v_pu']:.4f} | "
            f"{r['f_eq_min_hz']:.3f} | "
            + (f"{r['max_neighbour_dv_settle_pu']:.4f} / {r['max_neighbour_dv_peak_pu']:.4f} |"
               if r['max_neighbour_dv_settle_pu'] is not None else "- |"))
    lines.append("")
    lines.append("Trip sequences (t [s], block, class):")
    lines.append("")
    for r in rows:
        lines.append(f"* {r['level']}, margin {r['margin_pu']:g}, {r['relay']}: {r['trip_sequence']}")
    (out / "commitment_margin_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_figure(rows: list[dict[str, Any]], out: Path = OUT, formats: Sequence[str] = ("pdf", "png")) -> list[Path]:
    import matplotlib.pyplot as plt

    from .revised_figures import INK2, SERIES, SURFACE, _save, _style

    _style()
    fig, (ax_n, ax_p) = plt.subplots(1, 2, figsize=(7.0, 3.1), gridspec_kw={"wspace": 0.28})
    styles = {"stock": ("-", "o"), "C2.0": ((0, (5, 2)), "s"), "C1.5": ((0, (1.5, 1.5)), "^")}
    for k, level in enumerate(LEVELS):
        rr = sorted((r for r in rows if r["level"] == level and r["relay"] == "standard"), key=lambda r: r["margin_pu"])
        x = [r["margin_pu"] for r in rr]
        ls, mk = styles[level]
        c = SERIES[k]
        lab = f"{level} ({rr[0]['area_online_units']} units, {rr[0]['area_online_mva'] / 1000:.1f} GVA)"
        ax_n.plot(x, [r["plant_trips"] for r in rr], ls=ls, marker=mk, color=c, ms=5, mec=SURFACE, mew=0.8, label=lab)
        ax_p.plot(x, [r["lost_p_mw"] for r in rr], ls=ls, marker=mk, color=c, ms=5, mec=SURFACE, mew=0.8, label=lab)
        ax_n.plot(x, [r["propagated_absorption_loss"] + r["propagated_mw_loss"] for r in rr], ls=ls, marker=mk,
                  color=c, ms=3.5, lw=0.8, alpha=0.55, mfc="none")
        fast = [r for r in rows if r["level"] == level and r["relay"] == "fast"]
        for r in fast:
            ax_n.plot([r["margin_pu"]], [r["plant_trips"]], marker="*", ms=9, color=c, mec=SURFACE, mew=0.6, ls="none")
            ax_p.plot([r["margin_pu"]], [r["lost_p_mw"]], marker="*", ms=9, color=c, mec=SURFACE, mew=0.6, ls="none")
    ax_n.plot([], [], marker="*", ms=9, color=INK2, ls="none", label="fast relay (1.10 pu, 50 ms), margin 0.01")
    ax_n.plot([], [], marker="o", ms=3.5, color=INK2, lw=0.8, alpha=0.55, mfc="none", label="of which propagated by earlier trips")
    for ax in (ax_n, ax_p):
        ax.set_xscale("log")
        ax.set_xticks(list(MARGINS), [f"{m:g}" for m in MARGINS])
        ax.minorticks_off()
        ax.set_xlabel("initial margin of the most exposed plant (pu)")
    ax_n.set_ylabel("plant trips (of 16)")
    ax_p.set_ylabel("generation lost to plant trips (MW)")
    ax_n.set_ylim(bottom=-0.3)
    ax_p.set_ylim(bottom=-30)
    fig.subplots_adjust(bottom=0.26)
    fig.legend(*ax_n.get_legend_handles_labels(), loc="lower center", ncol=3, frameon=False, fontsize=6.4,
               bbox_to_anchor=(0.5, 0.0))
    paths = _save(fig, out, "fig_commitment_margin_trips", formats)
    plt.close(fig)
    return paths


ARMED_CASES: tuple[tuple[str, float], ...] = (("stock", 0.01), ("stock", 0.02), ("C2.0", 0.01), ("C1.5", 0.01), ("C1.5", 0.02))


def _armed_task(case: tuple[str, float]) -> dict[str, Any]:
    from .activsg2000_engine import _pos, build_armed_replica, line_series_admittance, max_abs_dxdt

    level, margin = case
    t0 = time.time()
    cfg = level_config(level, margin)
    try:
        armed = build_armed_replica(cfg, method="dynamic_equilibrium", settle_s=40.0)
    except Exception as exc:
        return {"level": level, "margin_pu": margin, "ok": False, "error": repr(exc), "wall_s": time.time() - t0}
    md = armed.metadata
    ss = armed.system
    ss.TDS.fg_update(ss.exist.pflow_tds)
    f_max, f_state = max_abs_dxdt(armed.rs)
    f_all, f_all_state = max_abs_dxdt(armed.rs, include_out_of_service=True)
    pf = build_armed_replica(cfg, method="powerflow")
    oa1 = {}
    for lid in armed.rs.mesh_line_ids:
        p = _pos(ss.Line, lid)
        y = line_series_admittance(ss, lid)
        oa1[lid] = {"u": float(ss.Line.u.v[p]), "ghk": float(ss.Line.ghk.v[p]), "bhk": float(ss.Line.bhk.v[p]),
                    "series_admittance_ok": bool(float(ss.Line.u.v[p]) == 1.0 and abs(float(ss.Line.ghk.v[p]) - y.real) < 1e-9
                                                 and abs(float(ss.Line.bhk.v[p]) - y.imag) < 1e-9)}
    return {
        "level": level, "ok": True, "margin_pu": margin, "wall_s": time.time() - t0, "oa1_circuits": oa1,
        "polish": md["polish"], "stage1": md["continuation"], "agc_reset": md["agc_reset"],
        "max_abs_dxdt_recheck": f_max, "worst_state": f_state,
        "max_abs_dxdt_incl_out_of_service": f_all, "worst_state_incl_out_of_service": f_all_state,
        "plateau_max_abs_dv_pu": md["plateau_max_abs_dv_pu"],
        "equilibrium_omega_range_pu": md["equilibrium_omega_range_pu"],
        "plateau_omega_range_pu": md["plateau_omega_range_pu"],
        "plateau_max_abs_dxdt": md["plateau_max_abs_dxdt"], "plateau_max_abs_dxdt_state": md["plateau_max_abs_dxdt_state"],
        "protected_v_equilibrium_pu": {b.asset_id: b.z0_pu for b in armed.protected_blocks},
        "protected_v_plateau_pu": md["plateau_collector_v_pu"],
        "protected_v_powerflow_armed_pu": {b.asset_id: b.z0_pu for b in pf.protected_blocks},
        "max_protected_v_equilibrium_pu": max(b.z0_pu for b in armed.protected_blocks),
        "max_protected_v_powerflow_armed_pu": max(b.z0_pu for b in pf.protected_blocks),
        "blocks_at_or_above_ov1_pickup": sorted(b.asset_id for b in armed.protected_blocks if b.z0_pu >= 1.10),
        "n_events": len(armed.events),
    }


def armed_checks(out: Path = OUT, *, jobs: int = 3,
                 cases: Sequence[tuple[str, float]] = ARMED_CASES) -> list[dict[str, Any]]:
    cases = list(cases)
    if jobs > 1:
        import multiprocessing as mp

        with mp.get_context("spawn").Pool(processes=min(jobs, len(cases))) as pool:
            res = pool.map(_armed_task, cases)
    else:
        res = [_armed_task(c) for c in cases]
    (out / "armed_states.json").write_text(json.dumps(_jsonable(res), indent=2) + "\n", encoding="utf-8")
    return res


def run(out: Path = OUT, *, jobs: int = 8, skip_existing: bool = True) -> list[dict[str, Any]]:
    out.mkdir(parents=True, exist_ok=True)
    tasks = [t for t in task_list(out) if not (skip_existing and (Path(t[0]) / "summary.json").exists())]
    if tasks:
        if jobs > 1:
            import multiprocessing as mp

            with mp.get_context("spawn").Pool(processes=min(jobs, len(tasks))) as pool:
                for r in pool.imap_unordered(_run_task, tasks):
                    print(f"done {r['level']} m={r['margin']:g} {r['kind']} ({r['wall_s']:.0f} s)", flush=True)
        else:
            for t in tasks:
                _run_task(t)
    rows = summarize_study(out)
    write_tables(rows, out)
    make_figure(rows, out)
    return rows


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, default=OUT)
    p.add_argument("--jobs", type=int, default=8)
    p.add_argument("--rerun", action="store_true", help="re-run tasks whose outputs exist")
    p.add_argument("--armed", action="store_true", help="only build/verify the dynamic-consistent armed states")
    a = p.parse_args(argv)
    if a.armed:
        for r in armed_checks(a.out, jobs=min(a.jobs, 4)):
            print({k: r.get(k) for k in ("level", "margin_pu", "ok", "max_abs_dxdt_recheck", "plateau_max_abs_dv_pu",
                                          "max_protected_v_equilibrium_pu", "error")})
        return 0
    rows = run(a.out, jobs=a.jobs, skip_existing=not a.rerun)
    for r in rows:
        print(f"{r['level']:6s} m={r['margin_pu']:<7g} {r['relay']:8s} trips={r['plant_trips']:2d} "
              f"(action {r['action_driven']}, MW-prop {r['propagated_mw_loss']}, abs-prop {r['propagated_absorption_loss']}) "
              f"lost={r['lost_p_mw']:.0f} MW gen={r['generator_trips']} ufls={r['ufls_stages']} sep={r['separation']} "
              f"collapse={r['collapse']} maxV={r['max_plant_v_pu']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
