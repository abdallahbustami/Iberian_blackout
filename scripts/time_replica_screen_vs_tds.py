#!/usr/bin/env python3
"""Single-core runtime of the screening against one time-domain simulation of each event, for the base state.

    OPENBLAS_NUM_THREADS=1 python scripts/time_replica_screen_vs_tds.py --slugs stock_m0p02,C1p5_m0p02
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np

import run_replica_screen as RS
from pa_dvsa.replication import replica_screen as R

OUT = ROOT / "results" / "revised" / "activsg2000" / "screen"


def tds_timing(slug: str, relay: str = "fast") -> list[dict]:
    path = OUT / f"timing_tds_{slug}_{relay}.json"
    if path.exists():
        return json.loads(path.read_text())
    jobs = [j for j in RS.make_jobs(OUT, [slug], [relay], accuracy_slugs=[]) if j["kind"] == "validation"]
    rows = []
    for j in jobs:
        res = R.tds_job(j)
        stored = OUT / "tds_runs" / f"{j['job_id']}.pkl"
        same = None
        if stored.exists():
            with open(stored, "rb") as fh:
                old = pickle.load(fh)
            same = sorted(map(str, old.get("tripped_within_window") or [])) == sorted(
                map(str, res.get("tripped_within_window") or []))
        rows.append({"seed": j["seed_id"], "wall_s": float(res["job_wall_s"]), "build_s": res.get("build_s"),
                     "sim_s": res.get("sim_s"), "status": res.get("status"), "same_trips_as_validation": same})
        print(f"  {j['seed_id']:45s} {res['job_wall_s']:6.1f} s  same trips {same}", flush=True)
    path.write_text(json.dumps(rows, indent=1) + "\n", encoding="utf-8")
    return rows


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--slugs", default="stock_m0p02,C1p5_m0p02")
    a = p.parse_args(argv)
    out = {"blas_threads": os.environ.get("OPENBLAS_NUM_THREADS"), "relay": "fast", "states": []}
    for slug in [s.strip() for s in a.slugs.split(",") if s.strip()]:
        meta = json.loads((OUT / slug / "screen_meta.json").read_text())["timing_s"]
        screen_s = float(meta["factory_build_s"] + meta["pre_mode_s"] + meta["events_total_s"])
        rows = tds_timing(slug)
        w = [r["wall_s"] for r in rows]
        out["states"].append({
            "slug": slug, "screen_factory_s": meta["factory_build_s"], "screen_pre_mode_s": meta["pre_mode_s"],
            "screen_events_s": meta["events_total_s"], "screen_total_s": screen_s, "tds_runs": len(w),
            "tds_total_s": float(sum(w)), "tds_median_s": float(np.median(w)),
            "tds_over_screen": float(sum(w)) / screen_s,
            "tds_trips_equal_validation": all(r["same_trips_as_validation"] for r in rows),
        })
        print(json.dumps(out["states"][-1]), flush=True)
    (OUT / "timing_single_core.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
