#!/usr/bin/env python3
"""Compares the sparse Krylov engine with the dense engine on one base-state variant: responses, bounds and runtime.

    OPENBLAS_NUM_THREADS=1 python scripts/verify_replica_krylov_engine.py --slug stock_m0p02
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from pa_dvsa.finite_window import trip_excursion_upper
from pa_dvsa.replication import replica_screen as R

OUT = ROOT / "results" / "revised" / "activsg2000" / "screen"


def run(case: R.ArmedCase, options: dict) -> dict:
    fac = R.SnapshotFactory(case)
    ss = fac()
    t0 = time.perf_counter()
    eng = R.build_engine(ss, case.blocks, **options)
    _ = eng.pre_mode
    out = {"pre_mode_s": time.perf_counter() - t0, "events": {}}
    for ev in list(case.events) + list(case.extra_events):
        t = time.perf_counter()
        r = eng.event_response(ev)
        el = time.perf_counter() - t
        out["events"][ev.event_id] = {
            "kind": ev.kind, "wall_s": el, "values": r.values_pu,
            "peak": np.array([e.peak_upper for e in r.envelopes]),
            "trip_fast": np.array([trip_excursion_upper(e, horizon_s=1.0, dwell_s=0.05) for e in r.envelopes]),
            "trip_ov1": np.array([trip_excursion_upper(e, horizon_s=2.0, dwell_s=1.0) for e in r.envelopes]),
            "metadata": {k: v for k, v in dict(r.metadata).items() if k.startswith(("krylov", "propagation", "log_norm"))},
        }
        print(f"  {ev.event_id:45s} {el:6.2f} s", flush=True)
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--slug", default="stock_m0p02")
    a = p.parse_args(argv)
    case = R.load_case(OUT / "cache" / f"armed_{a.slug}.pkl")
    print(f"[{a.slug}] Krylov engine", flush=True)
    kry = run(case, {})
    print(f"[{a.slug}] dense engine", flush=True)
    den = run(case, dict(R.DENSE_ENGINE_OPTIONS))
    rows = []
    for eid, k in kry["events"].items():
        d = den["events"][eid]
        rows.append({
            "event_id": eid, "kind": k["kind"], "krylov_s": k["wall_s"], "dense_s": d["wall_s"],
            "max_abs_response_diff_pu": float(np.max(np.abs(k["values"] - d["values"]))),
            "max_abs_peak_bound_diff_pu": float(np.max(np.abs(k["peak"] - d["peak"]))),
            "max_abs_trip_bound_diff_pu_fast": float(np.max(np.abs(k["trip_fast"] - d["trip_fast"]))),
            "max_abs_trip_bound_diff_pu_ov1": float(np.max(np.abs(k["trip_ov1"] - d["trip_ov1"]))),
            "max_peak_bound_pu": float(np.max(d["peak"])),
            **k["metadata"],
        })
    sw = [r for r in rows if r["kind"] == "switch"]
    pa = [r for r in rows if r["kind"] == "param"]
    summary = {
        "slug": a.slug, "blas_threads": os.environ.get("OPENBLAS_NUM_THREADS"), "events_compared": len(rows),
        "max_abs_response_diff_pu": max(r["max_abs_response_diff_pu"] for r in rows),
        "max_abs_peak_bound_diff_pu": max(r["max_abs_peak_bound_diff_pu"] for r in rows),
        "max_abs_trip_bound_diff_pu": max(max(r["max_abs_trip_bound_diff_pu_fast"], r["max_abs_trip_bound_diff_pu_ov1"]) for r in rows),
        "krylov_fallbacks": sum(1 for r in rows if r.get("propagation") == "dense_fallback"),
        "krylov_m_max": max((r.get("krylov_m_max", 0) for r in rows), default=0),
        "krylov_est_output_abs_pu_max": max((r.get("krylov_est_output_abs_pu", 0.0) for r in rows), default=0.0),
        "pre_mode_krylov_s": kry["pre_mode_s"], "pre_mode_dense_s": den["pre_mode_s"],
        "events_total_krylov_s": sum(r["krylov_s"] for r in rows), "events_total_dense_s": sum(r["dense_s"] for r in rows),
        "median_switch_krylov_s": float(np.median([r["krylov_s"] for r in sw])) if sw else None,
        "median_switch_dense_s": float(np.median([r["dense_s"] for r in sw])) if sw else None,
        "median_param_krylov_s": float(np.median([r["krylov_s"] for r in pa])) if pa else None,
        "median_param_dense_s": float(np.median([r["dense_s"] for r in pa])) if pa else None,
        "rows": rows,
    }
    (OUT / f"krylov_vs_dense_{a.slug}.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
