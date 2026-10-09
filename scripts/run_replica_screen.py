"""Screening of the base state of the 2000-bus system and its validation against time-domain simulation, written
to results/revised/activsg2000/screen/. Stages: build (armed states), screen, tds, report.

    python scripts/run_replica_screen.py --stages build,screen --jobs 2
    python scripts/run_replica_screen.py --stages tds --relays fast --jobs 4
    python scripts/run_replica_screen.py --stages tds --margins 0.02 --relays ov1 --jobs 4
    python scripts/run_replica_screen.py --stages report
"""

from __future__ import annotations

import argparse
import math
import multiprocessing as mp
import os
import pickle
import re
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from pa_dvsa.replication import replica_screen as R

LOG_FILE: Path | None = None


def log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    if LOG_FILE is not None:
        with open(LOG_FILE, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")


def _safe(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.+-]", "_", s)


def case_path(out: Path, slug: str) -> Path:
    return out / "cache" / f"armed_{slug}.pkl"


def build_task(args: tuple[str, float, str]) -> dict[str, Any]:
    level, margin, out_s = args
    out = Path(out_s)
    slug = R.slug_of(level, margin)
    t0 = time.perf_counter()
    try:
        case, ss, ver = R.build_armed_case(level, margin)
        reuse = verify_reuse_tds(case, ss)
        R.save_case(case, case_path(out, slug))
        oa1 = oa1_finding(case, ss)
        info = armed_state_info(case, ver, reuse, oa1)
        R.write_json(out / slug / "armed_state.json", info)
        return {"slug": slug, "ok": True, "wall_s": time.perf_counter() - t0}
    except Exception as exc:
        return {"slug": slug, "ok": False, "error": f"{type(exc).__name__}: {exc}", "tb": traceback.format_exc(),
                "wall_s": time.perf_counter() - t0}


def verify_reuse_tds(case: R.ArmedCase, armed_ss: Any) -> dict[str, Any]:
    from pa_dvsa.tds_runner import run_protection_tds

    relay = R.RELAYS["fast"]
    blocked = set(case.no_margin_assets(relay.pickup_pu))
    pickup = {b.asset_id: (math.inf if b.asset_id in blocked else relay.pickup_pu) for b in case.blocks}
    seed = case.event("plant_trip:Badajoz582")
    kw = dict(relays_enabled=True, pickup_override=pickup, dwell_override=relay.dwell_s,
              trip_events={e.event_id: e for e in case.events}, t_seed=R.T_SEED_S, window_s=relay.horizon_s,
              tstep=R.TDS_TSTEP_S, tol=R.TDS_TOL, record_traces=True)
    R.reset_tds_for_reuse(armed_ss)
    with R.QuietAndes(True):
        ra = run_protection_tds(lambda: armed_ss, seed, case.blocks, **kw)
        rb = run_protection_tds(R.SnapshotFactory(case), seed, case.blocks, **kw)
    va, vb = ra.traces["v_collector"], rb.traces["v_collector"]
    same_shape = va.shape == vb.shape
    return {
        "seed": seed.event_id, "status": [ra.status, rb.status], "trip_order": [list(ra.trip_order), list(rb.trip_order)],
        "n_samples": [int(va.shape[0]), int(vb.shape[0])],
        "max_abs_trace_diff_pu": float(np.max(np.abs(va - vb))) if same_shape else None,
        "bitwise_identical": bool(same_shape and np.array_equal(va, vb)),
        "runtime_s": [ra.runtime_s, rb.runtime_s],
    }


def oa1_finding(case: R.ArmedCase, ss: Any) -> dict[str, Any]:
    from pa_dvsa.event_response import solve_frozen_algebraic
    from pa_dvsa.events import apply_mutations, read_base_values

    ss2 = R.SnapshotFactory(case)()
    mesh = case.metadata.get("mesh_line_ids", [])
    rows = []
    for lid in mesh:
        p = R._pos(ss2.Line, lid)
        rows.append({"line": lid, "u": float(ss2.Line.u.v[p]), "ghk": float(ss2.Line.ghk.v[p]), "bhk": float(ss2.Line.bhk.v[p]),
                     "b_charging": float(ss2.Line.b.v[p])})
    muts = R.oa1_series_closure_mutations(ss2, mesh)
    out: dict[str, Any] = {"oa1_lines": rows, "series_admittance_missing": bool(muts)}
    if muts:
        addr = [int(ss2.Bus.v.a[R._pos(ss2.Bus, b.collector_bus)]) for b in case.blocks]
        v0 = np.array([ss2.dae.y[a] for a in addr])
        base = read_base_values(ss2, muts)
        apply_mutations(ss2, muts, base, 1.0)
        it, res = solve_frozen_algebraic(ss2)
        v1 = np.array([ss2.dae.y[a] for a in addr])
        dv = (v1 - v0) * 1e3
        out.update({"instantaneous_dv_collectors_mpu": {b.asset_id: float(d) for b, d in zip(case.blocks, dv)},
                    "newton_iterations": it, "residual": res,
                    "note": "frozen-state algebraic jump when the OA1 circuits get their series admittance "
                            "(not applied: the armed state is kept as built by the replica)"})
    return out


def armed_state_info(case: R.ArmedCase, ver: dict[str, Any], reuse: dict[str, Any], oa1: dict[str, Any]) -> dict[str, Any]:
    md = case.metadata
    return {
        "slug": case.slug, "level": case.level, "margin_pu": case.margin, "tap": case.tap,
        "method": "dynamic_equilibrium", "settle_s": R.SETTLE_S, "build_s": case.build_s,
        "disabled_exciters": case.disabled_exciters,
        "polish": {k: v for k, v in md.get("polish", {}).items() if k != "residual_history"},
        "plateau_max_abs_dv_pu": md.get("plateau_max_abs_dv_pu"),
        "equilibrium_omega_range_pu": md.get("equilibrium_omega_range_pu"),
        "assets": [{"asset_id": b.asset_id, "collector_bus": str(b.collector_bus), "hv_bus": str(b.hv_bus),
                    "z0_pu": b.z0_pu, "p_mw": b.p_mw, "q_mvar": b.q_mvar,
                    **{f"margin_{k}_pu": r.pickup_pu - b.z0_pu for k, r in R.RELAYS.items()}}
                   for b in case.blocks],
        "no_margin_assets": {k: case.no_margin_assets(r.pickup_pu) for k, r in R.RELAYS.items()},
        "n_events": len(case.events), "families": dict(sorted(__import__("collections").Counter(e.family for e in case.events).items())),
        "extra_events": [e.event_id for e in case.extra_events],
        "library_corrections": case.library_corrections,
        "fingerprint": case.fingerprint,
        "factory_verification": ver,
        "reuse_vs_factory_tds": reuse,
        "oa1_finding": oa1,
    }


def screen_task(args: tuple[str, str, bool]) -> dict[str, Any]:
    slug, out_s, jump_checks = args
    out = Path(out_s)
    t0 = time.perf_counter()
    try:
        case = R.load_case(case_path(out, slug))
        fac = R.SnapshotFactory(case)
        store, engine = R.run_screen(case, fac, log=log)
        with open(out / slug / "screen_store.pkl", "wb") as fh:
            pickle.dump(store, fh, protocol=pickle.HIGHEST_PROTOCOL)
        checks: list[dict[str, Any]] = []
        if jump_checks:
            t1 = time.perf_counter()
            for ev in case.events + case.extra_events:
                if ev.kind != "switch" or ev.event_id not in store.responses:
                    continue
                c = R.jump_branch_check(engine, ev, fac, store.responses[ev.event_id])
                checks.append(c)
                log(f"[{slug}]   jump {ev.event_id}: direct-vs-native {c.get('max_abs_dy_direct_vs_native', float('nan')):.2e}, "
                    f"direct-vs-cont {c.get('max_abs_dy_direct_vs_continuation', float('nan')):.2e}, "
                    f"dz20 err {c.get('dz20_screen_vs_native_max_err_mpu', float('nan')):.3f} mpu")
            store.timing_s["jump_checks_s"] = time.perf_counter() - t1
            gen = next((n for n in case.library_corrections if n["correction"].startswith("controllers ue")), None)
            if gen is not None:
                from pa_dvsa.events import EventSpec, Mutation

                ev0 = case.event(gen["event_id"])
                old = EventSpec(ev0.event_id + "#u_only", ev0.family, "switch",
                                tuple(Mutation(m, d, a, v) for m, d, a, v in gen["old"]), ev0.profile)
                try:
                    r0 = engine.event_response(old)
                    store.mode_diagnostics["uncorrected_" + ev0.event_id] = {
                        "max_real_eigenvalue": r0.diagnostics.max_real_eigenvalue,
                        "stability": r0.diagnostics.stability, "n_states": r0.diagnostics.n_states,
                        "max_abs_response_diff_vs_corrected_mpu": float(np.max(np.abs(r0.values_pu - store.responses[ev0.event_id].values_pu))) * 1e3,
                    }
                except Exception as exc:
                    store.mode_diagnostics["uncorrected_" + ev0.event_id] = {"error": f"{type(exc).__name__}: {exc}"}
            with open(out / slug / "screen_store.pkl", "wb") as fh:
                pickle.dump(store, fh, protocol=pickle.HIGHEST_PROTOCOL)
        R.write_csv(out / slug / "jump_checks.csv", [{k: (v if not isinstance(v, dict) else str(v)) for k, v in c.items()} for c in checks])
        write_screen_tables(case, store, out)
        return {"slug": slug, "ok": True, "wall_s": time.perf_counter() - t0}
    except Exception as exc:
        return {"slug": slug, "ok": False, "error": f"{type(exc).__name__}: {exc}", "tb": traceback.format_exc(),
                "wall_s": time.perf_counter() - t0}


def jumps_task(args: tuple[str, str]) -> dict[str, Any]:
    slug, out_s = args
    out = Path(out_s)
    t0 = time.perf_counter()
    try:
        case = R.load_case(case_path(out, slug))
        with open(out / slug / "screen_store.pkl", "rb") as fh:
            store: R.ScreenStore = pickle.load(fh)
        fac = R.SnapshotFactory(case)
        engine = R.build_engine(fac(), case.blocks)
        checks = []
        for ev in case.events + case.extra_events:
            if ev.kind != "switch" or ev.event_id not in store.responses:
                continue
            c = R.jump_branch_check(engine, ev, fac, store.responses[ev.event_id])
            checks.append(c)
        R.write_csv(out / slug / "jump_checks.csv", [{k: (v if not isinstance(v, dict) else str(v)) for k, v in c.items()} for c in checks])
        return {"slug": slug, "ok": True, "wall_s": time.perf_counter() - t0}
    except Exception as exc:
        return {"slug": slug, "ok": False, "error": f"{type(exc).__name__}: {exc}", "tb": traceback.format_exc(),
                "wall_s": time.perf_counter() - t0}


def write_screen_tables(case: R.ArmedCase, store: R.ScreenStore, out: Path) -> None:
    d = out / store.slug
    modes = []
    for mid, m in store.mode_diagnostics.items():
        modes.append({"slug": store.slug, "mode_id": mid, **{k: v for k, v in m.items() if k not in ("warnings", "top_eigenvalues", "timing_s")},
                      "warnings": ";".join(m.get("warnings", [])) if isinstance(m.get("warnings"), list) else "",
                      "top_eigenvalues": ";".join(f"{a:.4g}{b:+.4g}j" for a, b in m.get("top_eigenvalues", [])),
                      **{f"t_{k}_s": v for k, v in (m.get("timing_s") or {}).items()}})
    R.write_csv(d / "engine_modes.csv", modes)
    rest = [{"slug": store.slug, "event_id": e, **{f"max_abs_d{k}": v for k, v in errs.items()},
             "restored_within_1e-9": all(v <= 1e-9 for v in errs.values())} for e, errs in store.restoration.items()]
    R.write_csv(d / "engine_restoration.csv", rest)
    k_rows, ev_rows, a_rows, s_rows = [], [], [], []
    for rname, relay in R.RELAYS.items():
        maps = R.window_maps(store, case, relay)
        extra = [e.event_id for e in case.extra_events if e.event_id in store.responses]
        preds = R.predictions(store, case, relay, maps, extra_seeds=extra)
        k_rows += R.screen_k_rows(store, relay, maps)
        ev_rows += R.screen_event_rows(store, case, relay, maps, preds)
        ar, sr = R.prediction_rows(store, relay, preds, R.no_margin_of(maps))
        a_rows += ar
        s_rows += sr
    R.write_csv(d / "screen_k.csv", k_rows)
    R.write_csv(d / "screen_events.csv", ev_rows)
    R.write_csv(d / "cascade_predictions.csv", a_rows)
    R.write_csv(d / "cascade_seeds.csv", s_rows)
    R.write_json(d / "screen_meta.json", {
        "slug": store.slug, "relays": {k: r.to_dict() for k, r in R.RELAYS.items()}, "epsilon": R.EPSILON,
        "engine": store.engine_info, "timing_s": store.timing_s, "failed_events": store.failed,
        "pre_mode": store.mode_diagnostics.get("pre_event"),
        "uncorrected_generator_trip": {k: v for k, v in store.mode_diagnostics.items() if k.startswith("uncorrected_")},
    })


ACCURACY_EVENTS = ("plant_trip:Badajoz582", "plant_trip:Cuenca530", "fixed_pf_ramp:Badajoz582")


def accuracy_event_ids(case: R.ArmedCase) -> list[str]:
    ids = list(ACCURACY_EVENTS)
    for fam in ("generator_trip", "load_disconnection", "line_energization", "capacitor_switching", "generation_redispatch"):
        e = next((e for e in case.events if e.family == fam), None)
        if e is not None:
            ids.append(e.event_id)
    return [i for i in ids if any(e.event_id == i for e in case.events)]


def make_jobs(out: Path, slugs: Sequence[str], relays: Sequence[str], *, accuracy_slugs: Sequence[str]) -> list[dict[str, Any]]:
    jobs = []
    for slug in slugs:
        p = case_path(out, slug)
        case = R.load_case(p)
        base = {"slug": slug, "case_path": str(p), "fingerprint": case.fingerprint}
        for rname in relays:
            relay = R.RELAYS[rname]
            blocked = case.no_margin_assets(relay.pickup_pu)
            for ev in case.events:
                jobs.append({**base, "job_id": _safe(f"{slug}__{rname}__{ev.event_id}"), "kind": "validation", "relay": rname,
                             "seed_id": ev.event_id, "blocked_assets": blocked})
            jobs.append({**base, "job_id": _safe(f"{slug}__{rname}__no_event"), "kind": "no_event", "relay": rname,
                         "seed_id": None, "blocked_assets": blocked})
            if blocked:
                jobs.append({**base, "job_id": _safe(f"{slug}__{rname}__no_event_unblocked"), "kind": "no_event_unblocked",
                             "relay": rname, "seed_id": None, "blocked_assets": []})
                for ex in case.extra_events:
                    jobs.append({**base, "job_id": _safe(f"{slug}__{rname}__{ex.event_id}"), "kind": "base_state_seed",
                                 "relay": rname, "seed_id": ex.event_id, "blocked_assets": []})
        if slug in accuracy_slugs:
            for eid in accuracy_event_ids(case):
                jobs.append({**base, "job_id": _safe(f"{slug}__accuracy__{eid}"), "kind": "accuracy", "relay": "ov1",
                             "seed_id": eid, "relays_enabled": False, "blocked_assets": [], "tstep": 0.001, "tol": 1e-8,
                             "window_s": 2.0, "t_end": R.T_SEED_S + 2.0})
    return jobs


def run_jobs(out: Path, jobs: Sequence[dict[str, Any]], n_jobs: int) -> None:
    d = out / "tds_runs"
    d.mkdir(parents=True, exist_ok=True)
    todo = [j for j in jobs if not (d / f"{j['job_id']}.pkl").exists()]
    log(f"TDS: {len(jobs)} jobs, {len(todo)} to run with {n_jobs} workers")
    if not todo:
        return
    t0 = time.perf_counter()
    done = 0
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=n_jobs, mp_context=ctx) as pool:
        futs = {pool.submit(R.tds_job, j): j for j in todo}
        for f in as_completed(futs):
            j = futs[f]
            try:
                r = f.result()
            except Exception as exc:
                r = {"job_id": j["job_id"], "status": "job_error", "converged": False, "error": f"worker failure: {exc}", "job": j}
            with open(d / f"{j['job_id']}.pkl", "wb") as fh:
                pickle.dump(r, fh, protocol=pickle.HIGHEST_PROTOCOL)
            done += 1
            if r.get("error") or done % 10 == 0 or done == len(todo):
                log(f"  TDS {done}/{len(todo)} ({time.perf_counter() - t0:.0f} s): {j['job_id']} {r.get('status')} "
                    f"trips={r.get('trip_order')} {r.get('error') or ''}")


def load_runs(out: Path) -> dict[str, dict[str, Any]]:
    runs = {}
    for p in sorted((out / "tds_runs").glob("*.pkl")):
        with open(p, "rb") as fh:
            runs[p.stem] = pickle.load(fh)
    return runs


def report(out: Path, slugs: Sequence[str]) -> None:
    runs = load_runs(out)
    all_ev, all_val_rows, all_seed_rows, all_agg, all_acc, all_extra, runtime = [], [], [], [], [], [], []
    all_cons: list[dict[str, Any]] = []
    for slug in slugs:
        d = out / slug
        if not (d / "screen_store.pkl").exists():
            log(f"report: no screen for {slug}")
            continue
        case = R.load_case(case_path(out, slug))
        with open(d / "screen_store.pkl", "rb") as fh:
            store: R.ScreenStore = pickle.load(fh)
        write_screen_tables(case, store, out)
        val_rows, seed_rows, agg_rows, extra_rows, cons_rows = [], [], [], [], []
        tds_rt = {}
        for rname, relay in R.RELAYS.items():
            maps = R.window_maps(store, case, relay)
            extra = [e.event_id for e in case.extra_events if e.event_id in store.responses]
            preds = R.predictions(store, case, relay, maps, extra_seeds=extra)
            all_ev += R.screen_event_rows(store, case, relay, maps, preds)
            tds = {}
            for ev in case.events:
                r = runs.get(_safe(f"{slug}__{rname}__{ev.event_id}"))
                if r is not None:
                    tds[ev.event_id] = r
            if not tds:
                continue
            blocked = case.no_margin_assets(relay.pickup_pu)
            vr, sr, ag = R.score_validation(store, case, relay, {k: v for k, v in preds.items() if k in tds}, tds,
                                            no_margin=blocked)
            val_rows += vr
            seed_rows += sr
            agg_rows += ag
            cons_rows += R.conservatism_rows(store, relay, maps, tds)
            if preds and all(hasattr(p, "refined") for p in preds.values()):
                _, sr_u, ag_u = R.score_validation(store, case, relay, {k: v.refined() for k, v in preds.items() if k in tds},
                                                   tds, no_margin=blocked, prediction="refined")
                seed_rows += sr_u
                agg_rows += ag_u
            tds_rt[rname] = [float(r.get("job_wall_s", float("nan"))) for r in tds.values()]
            for kind in ("no_event", "no_event_unblocked"):
                r = runs.get(_safe(f"{slug}__{rname}__{kind}"))
                if r is not None:
                    extra_rows.append(extra_run_row(slug, rname, kind, r, None))
            for ex in case.extra_events:
                r = runs.get(_safe(f"{slug}__{rname}__{ex.event_id}"))
                if r is not None:
                    extra_rows.append(extra_run_row(slug, rname, "base_state_seed", r, preds.get(ex.event_id)))
        acc = []
        for eid in accuracy_event_ids(case):
            r = runs.get(_safe(f"{slug}__accuracy__{eid}"))
            if r is not None and r.get("status") == "ok" and eid in store.responses:
                acc += R.accuracy_rows(store, r, eid)
        R.write_csv(d / "tds_validation.csv", val_rows)
        R.write_csv(d / "tds_seeds.csv", seed_rows)
        R.write_csv(d / "validation_summary.csv", agg_rows)
        R.write_csv(d / "accuracy.csv", acc)
        R.write_csv(d / "tds_extra_runs.csv", extra_rows)
        R.write_csv(d / "conservatism.csv", cons_rows)
        all_cons += cons_rows
        all_val_rows += val_rows
        all_seed_rows += seed_rows
        all_agg += agg_rows
        all_acc += acc
        all_extra += extra_rows
        sw = [store.event_runtime_s[e] for e in store.event_ids if not e.startswith("base_state")]
        rt = {"slug": slug, "armed_build_s": case.build_s, "factory_build_s": store.timing_s.get("factory_build_s"),
              "engine_baseline_s": store.timing_s.get("engine_baseline_s"), "pre_mode_s": store.timing_s.get("pre_mode_s"),
              "screen_events_total_s": float(sum(sw)), "screen_event_median_s": float(np.median(sw)) if sw else float("nan"),
              "n_events": len(sw), "jump_checks_s": store.timing_s.get("jump_checks_s")}
        for rname, v in tds_rt.items():
            rt[f"tds_{rname}_n"] = len(v)
            rt[f"tds_{rname}_total_s"] = float(np.nansum(v))
            rt[f"tds_{rname}_median_s"] = float(np.nanmedian(v)) if v else float("nan")
            rt[f"tds_over_screen_{rname}"] = float(np.nansum(v)) / float(sum(sw)) if sw else float("nan")
        runtime.append(rt)
        R.write_json(d / "runtime.json", rt)
    R.write_csv(out / "summary_screen_events.csv", all_ev)
    R.write_csv(out / "summary_validation.csv", all_agg)
    R.write_csv(out / "summary_tds_seeds.csv", all_seed_rows)
    R.write_csv(out / "summary_accuracy.csv", all_acc)
    R.write_csv(out / "summary_extra_runs.csv", all_extra)
    R.write_csv(out / "summary_runtime.csv", runtime)
    cons_sum = R.conservatism_summary(all_cons)
    R.write_csv(out / "summary_conservatism.csv", cons_sum)
    fam = R.fig9_family_rows(all_ev)
    R.write_csv(out / "fig_data" / "fig9_bottom_k_by_family.csv", fam)
    R.write_csv(out / "fig_data" / "fig9_bottom_k_events.csv",
                [{k: r[k] for k in ("slug", "relay", "event_id", "family", "max_k_pk_nonself", "argmax_asset",
                                    "max_k_tr_nonself", "verdict", "verdict_excl_no_margin", "predicted_trips")} for r in all_ev])
    write_tables_md(out, all_ev, fam, all_agg, all_seed_rows, all_acc, all_extra, runtime, slugs, cons_sum)
    log(f"report written for {len(runtime)} armed states")


def _f(v: Any, fmt: str = "{:.3f}") -> str:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return str(v) if v not in (None, "") else "-"
    if math.isnan(x):
        return "nan"
    return fmt.format(x)


def write_tables_md(out: Path, ev_rows, fam_rows, agg_rows, seed_rows, acc_rows, extra_rows, runtime, slugs, cons_rows=()) -> None:
    L: list[str] = ["# Replica screen: generated tables", "", "Generated by `scripts/run_replica_screen.py --stages report`.", ""]
    L += ["## Armed states", "", "| slug | tap | max z0 (asset) | min margin fast/OV1 (pu) | no-margin assets (fast) | factory arrays / residuals identical | reuse vs factory TDS |",
          "|---|---|---|---|---|---|---|"]
    for slug in slugs:
        p = out / slug / "armed_state.json"
        if not p.exists():
            continue
        info = __import__("json").loads(p.read_text())
        a = max(info["assets"], key=lambda r: r["z0_pu"])
        mm = min(r["margin_fast_pu"] for r in info["assets"])
        fv, rv = info["factory_verification"], info["reuse_vs_factory_tds"]
        L.append(f"| {slug} | {info['tap']:.5f} | {a['z0_pu']:.4f} ({a['asset_id']}) | {mm:+.4f} | "
                 f"{', '.join(info['no_margin_assets']['fast']) or '-'} | {fv['arrays_identical']} / {fv['residuals_bitwise_identical']} | "
                 f"{'bitwise identical' if rv['bitwise_identical'] else _f(rv['max_abs_trace_diff_pu'], '{:.1e}')} |")
    import csv as _csv
    L += ["", "## Engine checks", "",
          "| slug | pre-mode states full -> output-reachable | pre-mode max Re eig (full) | post-modes: max of max Re eig (full) / unstable flags | cond1(G_y) range | log-norm mu (1/s) | restoration max abs err | jump: direct vs continuation (max abs dy) | jump: direct vs ANDES-native step (max abs dy, in-service) | vout of exciters staying in service: moved at jump (max, pu) / all vout direct vs native | screen jump vs native (mpu) | screen dz(20 ms) vs native (mpu) |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for slug in slugs:
        d = out / slug
        if not (d / "engine_modes.csv").exists():
            continue
        modes = list(_csv.DictReader(open(d / "engine_modes.csv")))
        rest = list(_csv.DictReader(open(d / "engine_restoration.csv")))
        jc = list(_csv.DictReader(open(d / "jump_checks.csv"))) if (d / "jump_checks.csv").exists() else []
        pre = next(m for m in modes if m["mode_id"] == "pre_event")
        post = [m for m in modes if m["mode_id"].startswith("post_")]

        def fl(x):
            try:
                return float(x)
            except (TypeError, ValueError):
                return float("nan")

        pm = max(fl(m["max_real_eigenvalue_full"]) for m in post) if post else float("nan")
        nun = sum(1 for m in post if str(m.get("unstable_flag_full")).lower() == "true")
        conds = [fl(m["gy_condition_estimate"]) for m in [pre] + post]
        rmax = max(max(fl(v) for k, v in r.items() if k.startswith("max_abs_d")) for r in rest) if rest else float("nan")

        def jmax(key):
            vals = [fl(r.get(key)) for r in jc if r.get(key) not in (None, "")]
            vals = [v for v in vals if not math.isnan(v)]
            return max(vals) if vals else float("nan")

        L.append(f"| {slug} | {pre['n_states_full']} -> {pre['n_states_output_reachable']} | {fl(pre['max_real_eigenvalue_full']):.1e} | "
                 f"{pm:.1e} / {nun} of {len(post)} | {min(conds):.2e} - {max(conds):.2e} | {fl(pre['log_norm_mu']):.1f} | {rmax:.1e} | "
                 f"{jmax('max_abs_dy_direct_vs_continuation'):.1e} | {jmax('max_abs_dy_direct_vs_native'):.1e} | "
                 f"{jmax('max_abs_dvout_enabled_exciters_at_jump'):.1e} / {jmax('max_abs_dvout_direct_vs_native'):.1e} | "
                 f"{jmax('jump_screen_vs_native_max_err_mpu'):.1e} | {jmax('dz20_screen_vs_native_max_err_mpu'):.3f} |")
    for relay in ("fast", "ov1", "ov2"):
        rows = [r for r in fam_rows if r["relay"] == relay]
        if not rows:
            continue
        fams = sorted({r["family"] for r in rows})
        L += ["", f"## K^pk by family, relay {relay} (worst non-self K^pk / events with K^pk >= 1 / events)", "",
              "| family | " + " | ".join(slugs) + " |", "|---|" + "---|" * len(slugs)]
        for fam in fams:
            cells = []
            for slug in slugs:
                r = next((x for x in rows if x["slug"] == slug and x["family"] == fam), None)
                cells.append("-" if r is None else f"{_f(r['worst_k_pk_nonself'])} / {r['n_events_k_pk_ge1']} / {r['n_events']}")
            L.append(f"| {fam} | " + " | ".join(cells) + " |")
    L += ["", "## S2 verdicts per relay (events; verdict ignoring the no-margin assets, which are data_limited in every seed)", "",
          "| slug | relay | trip_predicted | pickup_predicted | risky | data_limited | certified_safe | predicted trip sets |",
          "|---|---|---|---|---|---|---|---|"]
    for slug in slugs:
        for relay in ("fast", "ov1", "ov2"):
            rr = [r for r in ev_rows if r["slug"] == slug and r["relay"] == relay and not str(r["family"]).startswith("base_state")]
            if not rr:
                continue
            c = __import__("collections").Counter(r["verdict_excl_no_margin"] for r in rr)
            sets = sorted({f"{r['event_id']}->{r['predicted_trips']}" for r in rr if r["predicted_trips"]})
            L.append(f"| {slug} | {relay} | {c.get('trip_predicted', 0)} | {c.get('pickup_predicted', 0)} | {c.get('risky', 0)} | "
                     f"{c.get('data_limited', 0)} | {c.get('certified_safe', 0)} | {'; '.join(sets) or '-'} |")
    L += ["", "## S3 validation (per slug x relay, family ALL)", "",
          "| slug | relay | seeds (inconcl.) | dangerous | pred. dangerous | TP | FP | caught_risky | risky_clear | unsafe_FN | TN | data_limited clear/trip | pickup_only trip/clear | late | Jaccard (non-empty / all) | timing err median (s) |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in agg_rows:
        if r["family"] != "ALL":
            continue
        L.append(f"| {r['slug']} | {r['relay']}{'' if r.get('prediction', 'paper') == 'paper' else ' (refined)'} | {r['n_seeds']} ({r['n_seeds_inconclusive']}) | {r['n_dangerous']} | {r['n_predicted_dangerous']} | "
                 f"{r['TP']} | {r['FP']} | {r['caught_risky']} | {r['risky_clear']} | {r['unsafe_FN']} | {r['TN']} | "
                 f"{r['data_limited_clear']}/{r['data_limited_trip']} | {r['pickup_only_trip']}/{r['pickup_only_clear']} | {r['late_trip']} | "
                 f"{_f(r['jaccard_mean_nonempty'])} / {_f(r['jaccard_mean_all'])} | {_f(r['timing_error_median_s'])} |")
    L += ["", "## S3 per family", "", "| slug | relay | family | seeds | dangerous | pred. dangerous | TP | FP | caught_risky | unsafe_FN | TN | late |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in agg_rows:
        if r["family"] == "ALL" or r.get("prediction", "paper") != "paper":
            continue
        L.append(f"| {r['slug']} | {r['relay']} | {r['family']} | {r['n_seeds']} | {r['n_dangerous']} | {r['n_predicted_dangerous']} | {r['TP']} | {r['FP']} | "
                 f"{r['caught_risky']} | {r['unsafe_FN']} | {r['TN']} | {r['late_trip']} |")
    danger = [s for s in seed_rows if s.get("prediction", "paper") == "paper"] 
    danger = [s for s in danger if s.get("dangerous") or s.get("predicted_dangerous") or s.get("n_caught_risky") or s.get("n_unsafe_FN")]
    L += ["", "## Seeds that are dangerous in TDS or flagged dangerous by the screen", "",
          "| slug | relay | seed | predicted set | TDS set (window) | late trips | Jaccard | pred. first (t) | TDS first (t) | max K^pk non-self |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for s in danger:
        L.append(f"| {s['slug']} | {s['relay']} | {s['seed_id']} | {';'.join(s['predicted_set']) or '-'} | {';'.join(s['tds_set']) or '-'} | "
                 f"{s.get('late_trips') or '-'} | {_f(s['jaccard'], '{:.2f}')} | {s['pred_first_asset'] or '-'} ({_f(s['pred_first_time_s'])}) | "
                 f"{s['tds_first_asset'] or '-'} ({_f(s['tds_first_time_s'])}) | {_f(s['max_k_pk_nonself'])} |")
    L += ["", "## Conservatism: rigorous screen bound M^pk vs TDS window maximum (seeds without any TDS trip, non-self assets)", "",
          "| slug | relay | pairs | bound < TDS | min / median / max (bound - TDS) (mpu) | pairs with TDS rise >= 1 mpu | bound/TDS median / p90 / min |",
          "|---|---|---|---|---|---|---|"]
    for r in cons_rows:
        if r["family"] != "ALL":
            continue
        L.append(f"| {r['slug']} | {r['relay']} | {r['n_pairs']} | {r['n_bound_below_tds']} | {_f(r['min_bound_minus_tds_mpu'], '{:.2f}')} / "
                 f"{_f(r['median_bound_minus_tds_mpu'], '{:.2f}')} / {_f(r['max_bound_minus_tds_mpu'], '{:.2f}')} | {r['n_pairs_rise_ge_1mpu']} | "
                 f"{_f(r['median_bound_over_tds'], '{:.2f}')} / {_f(r['p90_bound_over_tds'], '{:.2f}')} / {_f(r['min_bound_over_tds'], '{:.2f}')} |")
    L += ["", "Per family (all slugs of the table above):", "", "| slug | relay | family | pairs | bound < TDS | min (bound - TDS) (mpu) | bound/TDS median |", "|---|---|---|---|---|---|---|"]
    for r in cons_rows:
        if r["family"] == "ALL":
            continue
        L.append(f"| {r['slug']} | {r['relay']} | {r['family']} | {r['n_pairs']} | {r['n_bound_below_tds']} | {_f(r['min_bound_minus_tds_mpu'], '{:.4f}')} | "
                 f"{_f(r['median_bound_over_tds'], '{:.2f}')} |")
    L += ["", "## Extra runs (no-event, base-state trips)", "", "| slug | relay | kind | status | trips (abs s) | trips before t_s | screen prediction |", "|---|---|---|---|---|---|---|"]
    for r in extra_rows:
        L.append(f"| {r['slug']} | {r['relay']} | {r['kind']} | {r['status']} | {r['trips_abs_s'] or '-'} | {r['trips_before_t_seed'] or '-'} | "
                 f"{r['screen_verdict'] or '-'} {r['screen_predicted_trips']} |")
    if acc_rows:
        L += ["", "## Engine vs relays-off TDS (max over collectors)", "",
              "| slug | event | max abs jump (mpu) | jump err | max abs dz(20 ms) err | err [0,0.3] | err [0,1] | err [0,2] | window max screen / TDS / bound (largest-response collector) |",
              "|---|---|---|---|---|---|---|---|---|"]
        keys = sorted({(r["slug"], r["event_id"]) for r in acc_rows})
        for slug, eid in keys:
            rr = [r for r in acc_rows if r["slug"] == slug and r["event_id"] == eid]
            big = max(rr, key=lambda r: abs(r["max_dz_0_1_tds_mpu"]))
            L.append(f"| {slug} | {eid} | {max(abs(r['jump_screen_mpu']) for r in rr):.2f} | "
                     f"{max(abs(r['jump_screen_mpu'] - r['jump_tds_mpu']) for r in rr):.3f} | {max(abs(r['dz20_screen_mpu'] - r['dz20_tds_mpu']) for r in rr):.3f} | "
                     f"{max(r['err_0_0p3_mpu'] for r in rr):.3f} | {max(r['err_0_1_mpu'] for r in rr):.3f} | {max(r['err_0_2_mpu'] for r in rr):.3f} | "
                     f"{big['max_dz_0_1_screen_mpu']:.2f} / {big['max_dz_0_1_tds_mpu']:.2f} / {big['bound_0_1_mpu']:.2f} ({big['asset_id']}) |")
    L += ["", "## Runtime", "", "| slug | armed build (s) | factory build (s) | pre-mode (s) | screen 53 events (s) | median per event (s) | TDS fast: runs / total / median (s) | TDS ov1: runs / total / median (s) |",
          "|---|---|---|---|---|---|---|---|"]
    for r in runtime:
        L.append(f"| {r['slug']} | {_f(r['armed_build_s'], '{:.0f}')} | {_f(r['factory_build_s'], '{:.1f}')} | {_f(r['pre_mode_s'], '{:.1f}')} | "
                 f"{_f(r['screen_events_total_s'], '{:.0f}')} | {_f(r['screen_event_median_s'], '{:.1f}')} | "
                 f"{r.get('tds_fast_n', 0)} / {_f(r.get('tds_fast_total_s'), '{:.0f}')} / {_f(r.get('tds_fast_median_s'), '{:.1f}')} | "
                 f"{r.get('tds_ov1_n', 0)} / {_f(r.get('tds_ov1_total_s'), '{:.0f}')} / {_f(r.get('tds_ov1_median_s'), '{:.1f}')} |")
    (out / "tables.md").write_text("\n".join(L) + "\n", encoding="utf-8")


def extra_run_row(slug: str, relay: str, kind: str, r: dict[str, Any], pred: Any) -> dict[str, Any]:
    ts = float(r.get("t_seed_s", R.T_SEED_S))
    pre = [(a["asset_id"], a["trip_s"]) for a in r.get("assets", []) if a.get("trip_s") is not None and a["trip_s"] < ts - 1e-9]
    return {"slug": slug, "relay": relay, "kind": kind, "seed_id": r.get("seed_id"), "status": r.get("status"),
            "err_msg": r.get("err_msg", ""), "blocked_assets": ";".join(r.get("blocked_assets", [])),
            "trip_order": ";".join(r.get("trip_order", [])),
            "trips_abs_s": ";".join(f"{a}@{r2:.3f}" for a, r2 in [(a["asset_id"], a["trip_s"]) for a in r.get("assets", []) if a.get("trip_s") is not None]),
            "trips_before_t_seed": ";".join(f"{a}@{t:.3f}" for a, t in pre),
            "pre_seed_max_dv_pu": r.get("pre_seed_max_dv_pu"),
            "max_v_window": ";".join(f"{a['asset_id']}={a['max_v_window_pu']:.5f}" for a in r.get("assets", []) if a.get("max_v_window_pu") is not None),
            "screen_predicted_trips": ";".join(pred.predicted_trips) if pred is not None else "",
            "screen_verdict": pred.verdict if pred is not None else "",
            "screen_max_k_pk_nonself": pred.max_k_pk_nonself if pred is not None else "",
            "runtime_s": r.get("job_wall_s")}


def main(argv: Sequence[str] | None = None) -> int:
    global LOG_FILE
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, default=R.OUT)
    p.add_argument("--stages", default="build,screen,tds,report")
    p.add_argument("--levels", default="stock,C1.5")
    p.add_argument("--margins", default="0.02,0.01")
    p.add_argument("--relays", default="fast", help="relay settings validated by TDS (comma list of fast,ov1,ov2)")
    p.add_argument("--accuracy", default="stock_m0p02,C1p5_m0p02", help="slugs with relays-off accuracy runs")
    p.add_argument("--jobs", type=int, default=2, help="worker processes (build/screen <= 3, TDS <= 4)")
    p.add_argument("--blas-threads", type=int, default=3, help="BLAS threads per screen worker")
    p.add_argument("--no-jump-checks", action="store_true")
    p.add_argument("--rebuild", action="store_true")
    a = p.parse_args(argv)
    out: Path = a.out
    out.mkdir(parents=True, exist_ok=True)
    LOG_FILE = out / "run.log"
    stages = [s.strip() for s in a.stages.split(",") if s.strip()]
    pairs = [(lv.strip(), float(m)) for lv in a.levels.split(",") for m in a.margins.split(",")]
    slugs = [R.slug_of(lv, m) for lv, m in pairs]
    log(f"stages={stages} slugs={slugs} relays={a.relays} jobs={a.jobs}")
    if "build" in stages:
        os.environ["OPENBLAS_NUM_THREADS"] = os.environ["OMP_NUM_THREADS"] = "1"
        tasks = [(lv, m, str(out)) for lv, m in pairs if a.rebuild or not case_path(out, R.slug_of(lv, m)).exists()]
        n = max(1, min(a.jobs, 3, len(tasks) or 1))
        if tasks:
            with ProcessPoolExecutor(max_workers=n, mp_context=mp.get_context("spawn")) as pool:
                for f in as_completed([pool.submit(build_task, t) for t in tasks]):
                    r = f.result()
                    log(f"build {r['slug']}: ok={r['ok']} {r['wall_s']:.0f} s {r.get('error', '')}")
                    if not r["ok"]:
                        log(r.get("tb", ""))
    if "screen" in stages:
        os.environ["OPENBLAS_NUM_THREADS"] = os.environ["OMP_NUM_THREADS"] = str(a.blas_threads)
        tasks = [(s, str(out), not a.no_jump_checks) for s in slugs if case_path(out, s).exists()]
        n = max(1, min(a.jobs, 3, len(tasks) or 1))
        with ProcessPoolExecutor(max_workers=n, mp_context=mp.get_context("spawn")) as pool:
            for f in as_completed([pool.submit(screen_task, t) for t in tasks]):
                r = f.result()
                log(f"screen {r['slug']}: ok={r['ok']} {r['wall_s']:.0f} s {r.get('error', '')}")
                if not r["ok"]:
                    log(r.get("tb", ""))
    if "jumps" in stages:
        os.environ["OPENBLAS_NUM_THREADS"] = os.environ["OMP_NUM_THREADS"] = "1"
        tasks = [(s, str(out)) for s in slugs if (out / s / "screen_store.pkl").exists()]
        n = max(1, min(a.jobs, 4, len(tasks) or 1))
        with ProcessPoolExecutor(max_workers=n, mp_context=mp.get_context("spawn")) as pool:
            for f in as_completed([pool.submit(jumps_task, t) for t in tasks]):
                r = f.result()
                log(f"jumps {r['slug']}: ok={r['ok']} {r['wall_s']:.0f} s {r.get('error', '')}")
                if not r["ok"]:
                    log(r.get("tb", ""))
    if "tds" in stages:
        os.environ["OPENBLAS_NUM_THREADS"] = os.environ["OMP_NUM_THREADS"] = "1"
        relays = [r.strip() for r in a.relays.split(",") if r.strip()]
        jobs = make_jobs(out, [s for s in slugs if case_path(out, s).exists()], relays,
                         accuracy_slugs=[s.strip() for s in a.accuracy.split(",")])
        run_jobs(out, jobs, max(1, min(a.jobs, 4)))
    if "report" in stages:
        report(out, slugs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
