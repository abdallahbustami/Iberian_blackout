#!/usr/bin/env python3
"""Benchmark studies on the four benchmark systems, written to results/revised/<system>/ and
results/revised/summary/.

    python scripts/run_revised_studies.py --jobs 6
    python scripts/run_revised_studies.py --horizon 2 --jobs 6
    python scripts/run_revised_studies.py --summary-only
"""

from __future__ import annotations

import argparse
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from pa_dvsa.revised_studies import config as C
from pa_dvsa.revised_studies.io import write_json


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--systems", default="kundur,ieee39,npcc,gbnetwork")
    ap.add_argument("--studies", default=",".join(C.STUDIES))
    ap.add_argument("--jobs", type=int, default=4, help="TDS worker processes (<= 6)")
    ap.add_argument("--reuse-tds", action="store_true", help="reuse the S3 TDS runs saved in tds_runs.json/tds_traces.npz")
    ap.add_argument("--no-variants", action="store_true", help="S7: skip the rebuilt variants (fast-shunt/telemetry rows only)")
    ap.add_argument("--summary-only", action="store_true")
    ap.add_argument("--report", action="store_true",
                    help="rebuild the pooled tables and write the report tables to results/revised/summary/report_tables.md")
    ap.add_argument("--fault-dir", default=None, help="directory for worker faulthandler logs (debugging)")
    ap.add_argument("--rescore", action="store_true",
                    help="reuse every saved TDS run (run only missing jobs), keep the committed S1/S5/S9 files and "
                         "rewrite only the cascade-definition-dependent outputs")
    ap.add_argument("--horizon", type=float, default=1.0,
                    help="assessment horizon T in s (screen, scoring window, LP); TDS runs to t_s + 1.5 T")
    ap.add_argument("--update-observability", action="store_true",
                    help="recompute only S5 (observability) and S6 (uncertainty) and rewrite only their files "
                         "(no TDS; every other output, including the saved TDS runs and runtimes, is kept), "
                         "then rebuild the pooled summary")
    ap.add_argument("--out-subdir", default=None,
                    help="write to results/revised/<system>/<subdir>/ (default: none for T = 1, horizon_<T>s otherwise)")
    args = ap.parse_args(argv)
    systems = [s.strip() for s in args.systems.split(",") if s.strip()]
    C.set_horizon(args.horizon)
    subdir = args.out_subdir if args.out_subdir is not None else ("" if args.horizon == 1.0 else f"horizon_{args.horizon:g}s")
    studies = [s.strip().upper() for s in args.studies.split(",") if s.strip()]
    bad = [s for s in studies if s not in C.STUDIES]
    if bad:
        ap.error(f"unknown studies {bad}; S13 (replica screen) is not part of this script")
    from pa_dvsa.revised_studies.summary import build_summary

    if args.summary_only or args.report:
        build_summary(C.RESULTS_ROOT, systems, subdir=subdir)
        print("summary written to", C.RESULTS_ROOT / "summary")
        if args.report:
            from pa_dvsa.revised_studies.summary import write_report_tables

            write_report_tables(C.RESULTS_ROOT / "summary" / "report_tables.md")
            print("tables written to", C.RESULTS_ROOT / "summary" / "report_tables.md")
        return 0
    from pa_dvsa.revised_studies.pipeline import SystemStudy
    from pa_dvsa.revised_studies.tds_jobs import TDSPool

    if args.update_observability:
        t1 = time.perf_counter()
        for slug in systems:
            SystemStudy(slug, studies=["S5", "S6"], pool=None, subdir=subdir,
                        log=lambda m: print(f"[{time.perf_counter() - t1:8.1f} s] {m}", flush=True)).update_observability()
        build_summary(C.RESULTS_ROOT, [s for s in C.SYSTEMS], subdir=subdir)
        return 0

    t0 = time.perf_counter()
    status: dict[str, str] = {}

    def log(msg: str) -> None:
        print(f"[{time.perf_counter() - t0:8.1f} s] {msg}", flush=True)

    pool = TDSPool(args.jobs, fault_dir=args.fault_dir)
    try:
        for slug in systems:
            try:
                study = SystemStudy(slug, studies=studies, pool=pool, log=log, reuse_tds=args.reuse_tds,
                                    ablation_variants=not args.no_variants, subdir=subdir,
                                    rescore=args.rescore)
                study.run()
                status[slug] = "ok"
            except Exception as exc:
                status[slug] = f"failed: {type(exc).__name__}: {exc}"
                log(f"[{slug}] FAILED: {exc}")
                traceback.print_exc()
                write_json(C.RESULTS_ROOT / slug / subdir / "run_error.json",
                           {"system": slug, "error": status[slug], "traceback": traceback.format_exc()})
    finally:
        pool.close()
    if set(studies) == {"S7"}:
        from pa_dvsa.revised_studies.summary import build_ablation_summary

        build_ablation_summary(C.RESULTS_ROOT, [s for s in C.SYSTEMS], subdir=subdir)
    else:
        build_summary(C.RESULTS_ROOT, [s for s in C.SYSTEMS], subdir=subdir)
    log(f"status: {status}")
    return 0 if all(v == "ok" for v in status.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
