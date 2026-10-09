"""Simulations of the stressed state of the 2000-bus system: the main run and the single changes of Table 3,
written to results/revised/iberian_sequence/<label>/.

    python scripts/run_iberian_sequence.py --jobs 3
    python scripts/run_iberian_sequence.py --only final_main
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import time
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "revised" / "iberian_sequence"

FINAL_BASE = dict(tie_scale=15.0, voltage_control="q_control", q_control_tc_s=10.0, ehv_surplus_mvar=1600.0, uel_tc_s=1.0,
                  load_model="es", embedded_pv_share=0.15, tap_margin_pu=0.01,
                  actions=dict(res_reduction_mw=708.0, net_load_mw=0.0))
FINAL_CHANGES: dict[str, dict] = {
    "main": {},
    "no_drivers": dict(actions=dict(res_reduction_mw=0.0, net_load_mw=0.0)),
    "no_ov": dict(plant_ov_enabled=False, shadow_plant_ov=True, granada_instantaneous=False),
    "voltvar": dict(voltage_mode_res=True),
    "preserved": dict(preserved_absorption=True),
    "no_surplus": dict(ehv_surplus_mvar=0.0),
    "zload": dict(load_model="z"),
    "no_dg": dict(embedded_pv_share=0.0),
    "avr": dict(voltage_control="avr"),
    "fast": dict(plant_ov_mode="fast"),
    "k10": dict(tie_scale=10.0),
    "k12": dict(tie_scale=12.0),
    "granada": dict(block_tap_factor=(("Granada355", 0.9723),), tap_margin_pu=0.005),
    "m02": dict(tap_margin_pu=0.02),
    "m03": dict(tap_margin_pu=0.03),
    "m04": dict(tap_margin_pu=0.04),
}
STUDIES: dict[str, dict] = {f"final_{lab}": {**FINAL_BASE, **chg} for lab, chg in FINAL_CHANGES.items()}

KEYS = ["tds_ok", "final_time_s", "stopped_reason", "gsu_tap", "block_trips", "lost_block_p_mw", "generator_trips",
        "ufls_stages_operated", "ufls_shed_mw", "separation", "separation_time_s", "affected_area_collapse",
        "blackout_emerged", "max_collector_v_pu", "area_net_export_t0_mw"]


def run(label: str) -> tuple[str, dict]:
    sys.path.insert(0, str(ROOT / "src"))
    from pa_dvsa.replication import activsg2000_engine as E
    from pa_dvsa.replication.iberian_sequence import sequence_config

    t0 = time.time()
    kw = dict(STUDIES[label])
    out = OUT / label
    out.mkdir(parents=True, exist_ok=True)
    res: dict = {"label": label, "kwargs": kw}
    try:
        cfg, record = sequence_config(variant=f"iberian_sequence:{label}", **kw)
        res["record"] = record
        dis = list(E.detect_nonequilibrium_exciters(cfg))
        cfg = replace(cfg, tap_override=E.solve_uniform_tap(cfg, disabled_exciters=dis))
        rs = E.build_replica_system(cfg, quiet=True, disabled_exciters=dis)
        mon = E.PhysicsMonitor(rs)
        ss = rs.system
        ss.TDS.callpert = mon
        with E.QuietAndes(True):
            ok = bool(ss.TDS.run(no_summary=True))
        final_t = float(ss.dae.t)
        if not ok and mon.stopped_reason is None:
            mon._event(E.EventRecord(round(final_t, 6), "TDS_STOP", "tds_nonconvergence",
                                     "ANDES TDS failed to converge; the run is inconclusive beyond this time.", "numerical"))
        summ = E.summarize(cfg, rs, mon, tds_ok=ok, final_time_s=final_t, wall_s=time.time() - t0, build_s=0.0)
        summ["event_jump_solves"] = {"count": int(mon.jump_solves), "failures": list(mon.jump_failures)}
        E.write_outputs(E.RunResult(cfg, rs, mon, ok, final_t, time.time() - t0, summ, out), out)
        res.update({k: summ.get(k) for k in KEYS})
        res["timeline"] = [(round(e.time_s, 3), e.category, e.code) for e in mon.events]
    except Exception as exc:
        import traceback
        res["error"] = f"{type(exc).__name__}: {exc}"
        res["traceback"] = traceback.format_exc()[-2000:]
    res["wall_s"] = time.time() - t0
    (out / "study.json").write_text(json.dumps(res, indent=1, default=str))
    return label, res


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="*", choices=sorted(STUDIES), metavar="LABEL", help="run only these studies")
    ap.add_argument("--jobs", type=int, default=3, help="parallel ANDES processes")
    args = ap.parse_args(argv)
    labels = args.only or list(STUDIES)
    OUT.mkdir(parents=True, exist_ok=True)
    with mp.get_context("spawn").Pool(processes=min(args.jobs, len(labels))) as pool:
        results = dict(pool.map(run, labels))
    path = OUT / "summary.json"
    old = json.loads(path.read_text()) if path.exists() else {}
    old.update(results)
    path.write_text(json.dumps(old, indent=1, default=str))
    for lab, r in results.items():
        if "error" in r:
            print(f"{lab}: ERROR {r['error']}")
            continue
        print(f"{lab}: plant trips {r['block_trips']} ({r['lost_block_p_mw']:.0f} MW) gen {r['generator_trips']} "
              f"ufls {r['ufls_stages_operated']} sep {r['separation']} ({r['separation_time_s']}) blackout {r['blackout_emerged']} "
              f"| t_end {r['final_time_s']:.1f} {r['stopped_reason'] or ''} | {r['wall_s']:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
