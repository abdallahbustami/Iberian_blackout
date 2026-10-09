#!/usr/bin/env python3
"""Base state of the 2000-bus system: the main run, its variants, the forced-islanding check and the build-up of
the operating condition, written to results/revised/activsg2000/.

    python -m pa_dvsa.replication.academic_replica --jobs 6
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Sequence

from .activsg2000_engine import (
    DEFAULT_OUT_ROOT,
    VARIANTS,
    ReplicaConfig,
    _jsonable,
    _write_csv,
    build_replica_system,
    run_replica,
    variant_config,
)

TIME_ZERO_CEST = "2025-04-28T12:32:50+02:00"


@dataclass(frozen=True, slots=True)
class ReportAnchor:
    code: str
    report_time_cest: str
    model_role: str
    active_power_mw: float | None
    reactive_absorption_mvar: float | None
    description: str
    source: str


REPORT_ANCHORS: tuple[ReportAnchor, ...] = (
    ReportAnchor("pre_1200_1232_voltage_control", "09:00-12:32", "precondition", None, None,
                 "Line coupling (meshing), shunt-reactor switching and HVDC set-point changes for voltage control.",
                 "ENTSO-E factual report, Sec. 2.1.2 / Table 2-2."),
    ReportAnchor("E1", "12:32:00-12:32:57", "pre_trip_drift", 317.3, None,
                 "Net-load increase or embedded-generation loss before the first major trip (not modeled).",
                 "ENTSO-E final report, Ch. 1."),
    ReportAnchor("E2", "12:32:00-12:32:57", "fixed_pf_res_change", 208.0, 33.3,
                 "RES operating-point change at fixed power factor (modeled as OA3, 208 MW).",
                 "ENTSO-E final report, Ch. 1."),
    ReportAnchor("E3_Granada", "12:32:57.220", "collector_overvoltage_trip", 355.0, 165.0,
                 "Granada 400/220 kV evacuation transformer trips on 220 kV-side overvoltage; 400 kV side ~417.9 kV.",
                 "ENTSO-E final report p.14."),
    ReportAnchor("E4a_Badajoz", "12:33:16.443", "collector_overvoltage_trip", 582.0, 165.0,
                 "Badajoz evacuation trip (event 4a).", "ENTSO-E final report, event chronology."),
    ReportAnchor("E4b_Badajoz", "12:33:16.820", "collector_overvoltage_trip", 145.0, 37.0,
                 "Second nearby Badajoz trip (event 4b).", "ENTSO-E final report, event chronology."),
    ReportAnchor("E5_multi_site", "12:33:17-12:33:18.020", "multi_site_generation_trip_cluster", 928.0, None,
                 "Wind/PV trips across several provinces (the replica's 13 E5 rows sum to 1824 MW; inherited table).",
                 "ENTSO-E final and factual reports, Ch. 1."),
    ReportAnchor("Morocco_AC", "12:33:20.473", "underfrequency_ac_trip", None, None,
                 "AC interconnection to Morocco trips on underfrequency (no analogue in the replica).", "ENTSO-E reports."),
    ReportAnchor("France_AC", "12:33:21.535", "loss_of_synchronism_ac_trip", None, None,
                 "France-Spain AC lines trip on loss of synchronism (replica: out-of-step relay on the complete cut set).",
                 "ENTSO-E reports."),
    ReportAnchor("HVDC", "12:33:23.960", "hvdc_block", None, None,
                 "Santa Llogaia-Baixas HVDC separation (ACTIVSg2000 has no HVDC; not modeled).", "ENTSO-E reports."),
    ReportAnchor("blackout", "near 12:33:27", "blackout_endpoint", None, None,
                 "Iberian systems de-energize (replica: per-island collapse criteria only).", "ENTSO-E reports."),
)

VARIANT_ORDER: tuple[str, ...] = (
    "baseline",
    "no_pre_cascade_actions",
    "no_ov_protection",
    "delayed_protection",
    "preserved_absorption",
    "voltage_mode_res",
)

SENSITIVITY_CASES: dict[str, dict[str, Any]] = {
    "margin_0p25pct": {"tap_margin_pu": 0.0025},
    "margin_0p5pct": {"tap_margin_pu": 0.005},
    "margin_2pct": {"tap_margin_pu": 0.02},
    "pf_0p98": {"fixed_pf": 0.98},
    "fixed_tap_0p95": {"tap_rule": "fixed", "tap_fixed": 0.95},
    "margin_0p25pct_preserved": {"tap_margin_pu": 0.0025, "preserved_absorption": True},
}


def build_academic_replica_config(*, variant: str = "baseline", tf_s: float | None = None,
                                  tstep_s: float | None = None) -> ReplicaConfig:
    overrides: dict[str, Any] = {}
    if tf_s is not None:
        overrides["tf_s"] = float(tf_s)
    if tstep_s is not None:
        overrides["tstep_s"] = float(tstep_s)
    return variant_config(variant, **overrides)


def condition_buildup(cfg: ReplicaConfig | None = None, *, quiet: bool = True) -> list[dict[str, Any]]:
    cfg = cfg or ReplicaConfig()
    actions = list(cfg.pre_cascade_actions)
    rows = []
    for k in range(len(actions) + 1):
        sub = replace(cfg, pre_cascade_actions=tuple(actions[:k]), pre_cascade_enabled=k > 0)
        rs = build_replica_system(sub, armed=k > 0, quiet=quiet, init_tds=False)
        z = {b.spec.name: b.z0_pu for b in rs.blocks}
        hv = {b.spec.name: b.hv_v0_pu for b in rs.blocks}
        worst = max(z, key=z.get)
        rows.append({
            "step": k,
            "condition": "base operating point" if k == 0 else f"+ {actions[k - 1].code} ({actions[k - 1].kind})",
            "max_collector_v_pu": z[worst],
            "max_collector_block": worst,
            "max_ratio_to_ov1_pickup": z[worst] / 1.10,
            "blocks_at_or_above_pickup": int(sum(1 for v in z.values() if v >= 1.10)),
            "collector_v_pu": z,
            "hv_v_pu": hv,
            "gsu_tap": rs.tap,
        })
    return rows


def _run_one(args: tuple[str, str, dict[str, Any], str, bool]) -> dict[str, Any]:
    kind, name, overrides, out_dir, quiet = args
    t0 = time.time()
    if kind == "variant":
        cfg = variant_config(name, **overrides)
    else:
        cfg = variant_config("baseline", **overrides)
        cfg = replace(cfg, variant=f"sensitivity:{name}")
    res = run_replica(cfg, out_dir=Path(out_dir), quiet=quiet)
    s = res.summary
    return {
        "kind": kind, "name": name, "out_dir": out_dir, "wall_s": time.time() - t0,
        "overrides": overrides, "summary": s,
    }


def _summary_row(item: dict[str, Any]) -> dict[str, Any]:
    s = item["summary"]
    seq = s.get("block_trip_sequence") or []
    return {
        "kind": item["kind"],
        "name": item["name"],
        "tds_ok": s["tds_ok"],
        "final_time_s": s["final_time_s"],
        "gsu_tap": s["gsu_tap"],
        "initial_min_margin_pu": s["initial_margin_min_pu"],
        "block_trips": s["block_trips"],
        "first_trip_s": seq[0]["t"] if seq else None,
        "last_trip_s": seq[-1]["t"] if seq else None,
        "tripped_blocks": ";".join(e["block"] for e in seq),
        "lost_block_p_mw": s["lost_block_p_mw"],
        "lost_block_q_absorption_mvar": s["lost_block_q_absorption_mvar"],
        "other_plant_trips": s["other_plant_trips"],
        "generator_trips": s["generator_trips"],
        "ufls_stages": s["ufls_stages_operated"],
        "ufls_shed_mw": s["ufls_shed_mw"],
        "separation": s["separation"],
        "affected_area_collapse": s["affected_area_collapse"],
        "max_collector_v_pu": s["max_collector_v_pu"],
        "max_zone_hv_v_pu": s["max_zone_hv_v_pu"],
        "max_system_v_pu": s["max_system_v_pu"],
        "min_system_v_pu": s["min_system_v_pu"],
        "f_eq_min_hz": s["affected_f_eq_hz_min"],
        "f_eq_max_hz": s["affected_f_eq_hz_max"],
        "max_coi_angle_change_deg": s["max_coi_angle_change_deg"],
        "static_pv_max_q_exceedance_mvar": s.get("static_pv_q_limit_exceedance", {}).get("max_exceedance_mvar"),
        "wall_s": item["wall_s"],
    }


def _extra_task(name: str) -> tuple[str, Any]:
    from .activsg2000_engine import _islanding_case, area_net_export_mw, small_island_check

    base = ReplicaConfig()
    if name in {"A", "B"}:
        balance = 0.0
        if name == "B":
            probe = build_replica_system(replace(base, pre_cascade_enabled=False), armed=False, init_tds=False)
            balance = area_net_export_mw(probe)
        return name, _islanding_case(base, t_open_s=1.0, tf_s=6.0, quiet=True, balance_mw=balance)
    if name == "C":
        return name, small_island_check(base, quiet=True)
    if name == "buildup":
        return name, condition_buildup(quiet=True)
    if name == "inventory":
        from .inventory import build_inventory

        return name, build_inventory()
    raise ValueError(name)


def run_extras(out_root: Path, *, jobs: int = 1) -> dict[str, Any]:
    names = ["A", "B", "C", "buildup", "inventory"]
    if jobs > 1:
        import multiprocessing as mp

        with mp.get_context("spawn").Pool(processes=min(jobs, len(names))) as pool:
            got = dict(pool.map(_extra_task, names))
    else:
        got = dict(_extra_task(n) for n in names)
    probe = build_replica_system(replace(ReplicaConfig(), pre_cascade_enabled=False), armed=False, init_tds=False)
    from .activsg2000_engine import area_net_export_mw

    island = {
        "area_net_export_mw_base": area_net_export_mw(probe),
        "A_base_export": got["A"],
        "B_balanced": got["B"],
        "C_single_machine": got["C"],
        "andes_can_integrate_an_island": bool(got["C"].get("tds_ok")),
        "affected_area_separation_integrable": bool(got["A"]["andes_handles_island"] and got["B"]["andes_handles_island"]),
    }
    (out_root / "islanding_check.json").write_text(json.dumps(_jsonable(island), indent=2) + "\n", encoding="utf-8")
    (out_root / "condition_buildup.json").write_text(json.dumps(_jsonable(got["buildup"]), indent=2) + "\n",
                                                     encoding="utf-8")
    (out_root / "inventory.json").write_text(json.dumps(got["inventory"], indent=2) + "\n", encoding="utf-8")
    (out_root / "report_anchors.json").write_text(
        json.dumps([asdict(a) for a in REPORT_ANCHORS], indent=2) + "\n", encoding="utf-8")
    return {"islanding_check": island}


def _git_state() -> dict[str, Any]:
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, check=False).stdout
        return {"head": head, "dirty_files": len([l for l in dirty.splitlines() if l.strip()])}
    except Exception:
        return {}


def run_study(*, out_root: Path = DEFAULT_OUT_ROOT, variants: Sequence[str] = VARIANT_ORDER,
              sensitivity: Sequence[str] = tuple(SENSITIVITY_CASES), jobs: int = 1, plot: bool = True,
              formats: tuple[str, ...] = ("pdf", "png"), quiet: bool = True, tf_s: float | None = None,
              extras: bool = True, runs_only_existing: bool = False) -> dict[str, Any]:
    out_root.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    overrides: dict[str, Any] = {} if tf_s is None else {"tf_s": float(tf_s)}
    tasks: list[tuple[str, str, dict[str, Any], str, bool]] = []
    if runs_only_existing:
        variants, sensitivity = (), ()
    for v in variants:
        tasks.append(("variant", v, dict(overrides), str(out_root / v), quiet))
    for name in sensitivity:
        tasks.append(("sensitivity", name, {**overrides, **SENSITIVITY_CASES[name]},
                      str(out_root / "sensitivity" / name), quiet))
    if jobs > 1 and len(tasks) > 1:
        import multiprocessing as mp

        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=min(jobs, len(tasks))) as pool:
            results = pool.map(_run_one, tasks)
    else:
        results = [_run_one(t) for t in tasks]
    if runs_only_existing:
        rows = json.loads((out_root / "variants_summary.json").read_text(encoding="utf-8"))
        results = [{"name": r["name"], "out_dir": str(out_root / r["name"]) if r["kind"] == "variant"
                    else str(out_root / "sensitivity" / r["name"])} for r in rows]
    else:
        rows = [_summary_row(r) for r in results]
        _write_csv(out_root / "variants_summary.csv", rows)
        (out_root / "variants_summary.json").write_text(json.dumps(_jsonable(rows), indent=2) + "\n",
                                                         encoding="utf-8")

    extra: dict[str, Any] = {}
    if extras:
        extra = run_extras(out_root, jobs=jobs)
    figures: list[str] = []
    if plot:
        from .revised_figures import make_all

        figures = [str(p) for p in make_all(out_root, out_root / "figures", formats=formats)]
    manifest = {
        "script": "python -m pa_dvsa.replication.academic_replica",
        "mode": "physics",
        "claim": "ACTIVSg2000-backed mechanism replica (not a forensic Iberian equivalent); all protection triggers "
                 "are measured quantities of the ANDES solution; no count/min-time/previous-stage gates.",
        "time_zero_cest_reference": TIME_ZERO_CEST,
        "variants": {r["name"]: VARIANTS.get(r["name"], "") for r in rows if r["kind"] == "variant"},
        "sensitivity_cases": {r["name"]: SENSITIVITY_CASES.get(r["name"]) for r in rows if r["kind"] == "sensitivity"},
        "outputs": {r["name"]: r["out_dir"] for r in results},
        "figures": figures,
        "git": _git_state(),
        "wall_time_s": time.time() - t0,
        "python": sys.version.split()[0],
    }
    (out_root / "manifest.json").write_text(json.dumps(_jsonable(manifest), indent=2) + "\n", encoding="utf-8")
    return {"rows": rows, "extra": extra, "manifest": manifest}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT_ROOT)
    p.add_argument("--variant", action="append", choices=VARIANT_ORDER, help="run only these variants (repeatable)")
    p.add_argument("--no-sensitivity", action="store_true")
    p.add_argument("--no-extras", action="store_true", help="skip islanding check, build-up and inventory")
    p.add_argument("--extras-only", action="store_true",
                   help="do not re-run the variants; recompute extras/figures/manifest from existing run outputs")
    p.add_argument("--jobs", type=int, default=1)
    p.add_argument("--tf", type=float, default=None)
    p.add_argument("--plot", action="store_true")
    p.add_argument("--formats", default="pdf,png")
    p.add_argument("--verbose-andes", action="store_true")
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    formats = tuple(x.strip().lower() for x in args.formats.split(",") if x.strip())
    result = run_study(
        out_root=args.out,
        variants=tuple(args.variant) if args.variant else VARIANT_ORDER,
        sensitivity=() if args.no_sensitivity else tuple(SENSITIVITY_CASES),
        jobs=max(1, int(args.jobs)),
        plot=bool(args.plot),
        formats=formats,
        quiet=not args.verbose_andes,
        tf_s=args.tf,
        extras=not args.no_extras,
        runs_only_existing=bool(args.extras_only),
    )
    for row in result["rows"]:
        print(f"{row['kind']:11s} {row['name']:24s} trips={row['block_trips']:2d} lostP={row['lost_block_p_mw']:7.1f} MW "
              f"lostQ={row['lost_block_q_absorption_mvar']:6.1f} MVAr maxVcol={row['max_collector_v_pu']:.4f} "
              f"gen={row['generator_trips']} ufls={row['ufls_stages']} sep={row['separation']} "
              f"collapse={row['affected_area_collapse']} tds_ok={row['tds_ok']}")
    print(f"outputs: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
