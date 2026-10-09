#!/usr/bin/env python3
"""Screening of the stressed state before the output reduction and its validation against time-domain simulation,
with relays that reset at pickup and at 0.98 of pickup (results/revised/activsg2000/screen_sequence/).

    python scripts/run_sequence_screen.py --stages build,screen,tds,tds_dropout,report --jobs 6
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import run_replica_screen as RS
from run_iberian_sequence import FINAL_BASE
from pa_dvsa.replication import replica_screen as R

OUT = ROOT / "results" / "revised" / "activsg2000" / "screen_sequence"


def slug() -> str:
    return R.slug_of(f"sequence_k{FINAL_BASE['tie_scale']:g}", FINAL_BASE["tap_margin_pu"])


def build(out: Path) -> str:
    t0 = time.perf_counter()
    case, ss, ver = R.build_sequence_case(FINAL_BASE, reduction_mw=FINAL_BASE["actions"]["res_reduction_mw"], log=RS.log)
    path = RS.case_path(out, case.slug)
    if path.exists() and R.load_case(path).fingerprint != case.fingerprint:
        raise RuntimeError(f"{case.slug}: rebuilt state differs from the stored one (screen and TDS would not match)")
    reuse = RS.verify_reuse_tds(case, ss)
    R.save_case(case, path)
    info = RS.armed_state_info(case, ver, reuse, {})
    info["sequence_record"] = case.metadata.get("sequence_record")
    R.write_json(out / case.slug / "armed_state.json", info)
    z = sorted(((b.z0_pu, b.asset_id) for b in case.blocks), reverse=True)[:3]
    RS.log(f"[{case.slug}] built in {time.perf_counter() - t0:.0f} s: tap {case.tap:.5f}, {len(case.blocks)} plants, "
           f"{len(case.events)} events, highest plant voltages {z}")
    return case.slug


DROPOUT_TAG = "ov1d"


def dropout_jobs(out: Path, slugs: list[str], ratio: float) -> list[dict]:
    jobs = []
    for j in RS.make_jobs(out, slugs, ["ov1"], accuracy_slugs=[]):
        if j["kind"] != "validation":
            continue
        jobs.append({**j, "job_id": j["job_id"].replace("__ov1__", f"__{DROPOUT_TAG}__"), "dropout_ratio": float(ratio)})
    return jobs


def score_dropout(out: Path, slug: str, ratio: float) -> None:
    import pickle

    runs = RS.load_runs(out)
    case = R.load_case(RS.case_path(out, slug))
    with open(out / slug / "screen_store.pkl", "rb") as fh:
        store = pickle.load(fh)
    relay = R.RELAYS["ov1"]
    tds = {ev.event_id: runs[k] for ev in case.events
           if (k := RS._safe(f"{slug}__{DROPOUT_TAG}__{ev.event_id}")) in runs}
    if not tds:
        return
    maps = R.window_maps(store, case, relay)
    preds = R.predictions(store, case, relay, maps, extra_seeds=[])
    vr, sr, ag = R.score_validation(store, case, relay, {k: v for k, v in preds.items() if k in tds}, tds,
                                    no_margin=case.no_margin_assets(relay.pickup_pu))
    for rows in (vr, sr, ag):
        for r in rows:
            r["relay"] = f"ov1_dropout{ratio:g}"
    R.write_csv(out / slug / "tds_validation_ov1_dropout.csv", vr)
    R.write_csv(out / slug / "tds_seeds_ov1_dropout.csv", sr)
    R.write_csv(out / slug / "validation_summary_ov1_dropout.csv", ag)
    RS.log(f"[{slug}] OV1 with dropout {ratio:g}: {len(tds)} seeds scored")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stages", default="build,screen,tds,tds_dropout,report")
    p.add_argument("--relays", default="fast,ov1")
    p.add_argument("--jobs", type=int, default=6)
    p.add_argument("--blas-threads", type=int, default=1)
    p.add_argument("--out", type=Path, default=OUT)
    p.add_argument("--rebuild", action="store_true")
    p.add_argument("--dropout", type=float, default=0.98)
    a = p.parse_args(argv)
    out = a.out
    out.mkdir(parents=True, exist_ok=True)
    RS.LOG_FILE = out / "run.log"
    s = slug()
    stages = [x.strip() for x in a.stages.split(",") if x.strip()]
    RS.log(f"sequence screen: stages={stages} slug={s} relays={a.relays}")
    os.environ["OPENBLAS_NUM_THREADS"] = os.environ["OMP_NUM_THREADS"] = "1"
    if "build" in stages and (a.rebuild or not RS.case_path(out, s).exists()):
        built = build(out)
        assert built == s, (built, s)
    if "screen" in stages:
        os.environ["OPENBLAS_NUM_THREADS"] = os.environ["OMP_NUM_THREADS"] = str(a.blas_threads)
        with ProcessPoolExecutor(max_workers=1, mp_context=mp.get_context("spawn")) as pool:
            r = pool.submit(RS.screen_task, (s, str(out), False)).result()
            RS.log(f"screen {s}: ok={r['ok']} {r['wall_s']:.0f} s {r.get('error', '')}")
            if not r["ok"]:
                RS.log(r.get("tb", ""))
    if "tds" in stages:
        os.environ["OPENBLAS_NUM_THREADS"] = os.environ["OMP_NUM_THREADS"] = "1"
        relays = [r.strip() for r in a.relays.split(",") if r.strip()]
        RS.run_jobs(out, RS.make_jobs(out, [s], relays, accuracy_slugs=[]), max(1, a.jobs))
    if "tds_dropout" in stages:
        os.environ["OPENBLAS_NUM_THREADS"] = os.environ["OMP_NUM_THREADS"] = "1"
        RS.run_jobs(out, dropout_jobs(out, [s], a.dropout), max(1, a.jobs))
    if "report" in stages:
        RS.report(out, [s])
        score_dropout(out, s, a.dropout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
