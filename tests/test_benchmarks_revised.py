"""Tests of the benchmark systems; also a validation driver of every event and control in time-domain simulation.

    python -m tests.test_benchmarks_revised --validate --workers 8
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any, Mapping, Sequence
import unittest

import numpy as np

try:
    import pytest
except ImportError:
    pytest = None

from pa_dvsa import benchmarks as bm
from pa_dvsa.events import Mutation, apply_mutations, profile_value, read_base_values

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUN_SLOW = os.environ.get("PA_DVSA_SLOW", "") not in ("", "0")
T_SEED = 0.5
T_AFTER = 1.5
TSTEP = 0.005
DV_EVENT_MIN = 1e-4
KNOWN_TRANSIENT_FRAGILITY: dict[str, dict[str, Any]] = {
}


def _slow(func):
    if pytest is not None:
        func = pytest.mark.slow(func)
    return unittest.skipUnless(RUN_SLOW, "set PA_DVSA_SLOW=1 for full-library TDS checks")(func)


def run_tds(case: bm.BenchmarkCase, mutations: Sequence[Mutation], profile: Any, *, alpha: float = 1.0,
            t_seed: float = T_SEED, t_after: float = T_AFTER, tstep: float = TSTEP,
            watch_buses: Sequence[Any] = ()) -> dict[str, Any]:
    started = time.perf_counter()
    tf = t_seed + t_after
    ss = case.factory(tds_config={"tf": tf, "tstep": tstep, "criteria": 0})
    muts = list(mutations)
    base = read_base_values(ss, muts) if muts else ()
    has_u = any(m.attr == "u" for m in muts)
    state = {"switched": False, "last_frac": 0.0}

    def callback(t: float, system: Any) -> None:
        if not muts:
            return
        rel = float(t) - t_seed
        frac = alpha * profile_value(profile, rel) if rel >= -1e-12 else 0.0
        if frac == state["last_frac"] and (not has_u or state["switched"] == (frac >= 0.5)):
            return
        apply_mutations(system, muts, base, frac)
        if has_u and state["switched"] != (frac >= 0.5):
            state["switched"] = frac >= 0.5
            system.TDS.custom_event = True
        state["last_frac"] = frac

    ss.TDS.callpert = callback
    with bm._quiet():
        ok = bool(ss.TDS.run(no_summary=True))
    t_end = float(ss.dae.t)
    ts = ss.dae.ts
    t = np.asarray(ts.t, dtype=float)
    y = np.asarray(ts.y, dtype=float)
    x = np.asarray(ts.x, dtype=float)
    va = np.asarray(ss.Bus.v.a, dtype=int)
    pre = t < t_seed - 1e-9
    k_ref = int(np.flatnonzero(pre)[-1]) if np.any(pre) else 0
    v = y[:, va]
    v_ref = v[k_ref]
    post = ~pre
    dv = v[post] - v_ref if np.any(post) else np.zeros((1, len(va)))
    pre_drift = float(np.max(np.abs(v[pre] - v[0]))) if np.any(pre) else 0.0
    bus_ids = [str(b) for b in ss.Bus.idx.v]
    col_pos = [bus_ids.index(str(b.collector_bus)) for b in case.blocks]
    omega_addr, delta_addr = [], []
    for model in ss.groups["SynGen"].models.values():
        if model.n:
            live = np.asarray(model.u.v, dtype=float) != 0.0
            omega_addr += [int(a) for a, ok in zip(model.omega.a, live) if ok]
            delta_addr += [int(a) for a, ok in zip(model.delta.a, live) if ok]
    omega = x[:, omega_addr] if omega_addr else np.ones((len(t), 1))
    delta = x[:, delta_addr] if delta_addr else np.zeros((len(t), 1))
    spread = np.max(delta, axis=1) - np.min(delta, axis=1)
    out = {
        "tds_ok": ok,
        "reached_tf": bool(t_end >= tf - 1e-6),
        "tds_message": str(getattr(ss.TDS, "err_msg", "") or ""),
        "t_end": t_end,
        "n_steps": int(len(t)),
        "pre_event_max_drift_pu": pre_drift,
        "max_abs_dv_any_bus_pu": float(np.max(np.abs(dv))),
        "argmax_bus": bus_ids[int(np.argmax(np.max(np.abs(dv), axis=0)))],
        "collector_max_dv_pu": [float(np.max(dv[:, p])) for p in col_pos],
        "collector_min_dv_pu": [float(np.min(dv[:, p])) for p in col_pos],
        "collector_final_dv_pu": [float(dv[-1, p]) for p in col_pos],
        "collector_max_v_pu": [float(np.max(v[post][:, p])) if np.any(post) else float(v_ref[p]) for p in col_pos],
        "max_abs_omega_dev": float(np.max(np.abs(omega - 1.0))),
        "angle_spread_init_deg": float(np.degrees(spread[0])),
        "angle_spread_max_deg": float(np.degrees(np.max(spread))),
        "angle_spread_final_deg": float(np.degrees(spread[-1])),
        "runtime_s": time.perf_counter() - started,
    }
    out["loss_of_synchronism"] = bool(out["angle_spread_max_deg"] - out["angle_spread_init_deg"] > 180.0)
    t_post = t[post] if np.any(post) else np.array([t_seed])
    early = t_post <= t_seed + 0.5 + 1e-9
    for b in watch_buses:
        p = bus_ids.index(str(b))
        out.setdefault("watch", {})[str(b)] = {"final_dv_pu": float(dv[-1, p]), "min_dv_pu": float(np.min(dv[:, p])),
                                                "max_dv_pu": float(np.max(dv[:, p])),
                                                "min_dv_first_0p5s_pu": float(np.min(dv[early, p]))}
    return out


_CASE_CACHE: dict[tuple[str, str], bm.BenchmarkCase] = {}


def _case(slug: str, variant: str = "baseline") -> bm.BenchmarkCase:
    key = (slug, variant)
    if key not in _CASE_CACHE:
        _CASE_CACHE[key] = bm.build_benchmark(slug, variant=variant)
    return _CASE_CACHE[key]


def _control_bus(case: bm.BenchmarkCase, control: Any) -> Any:
    meta = dict(control.metadata)
    return meta.get("bus")


def _job(args: tuple[str, ...]) -> dict[str, Any]:
    slug, kind, item_id = args[:3]
    variant = args[3] if len(args) > 3 else "baseline"
    case = _case(slug, variant)
    try:
        if kind == "noevent":
            res = run_tds(case, (), None, t_after=3.0 - T_SEED)
            res.update({"kind": kind, "id": "no_event_3s"})
            return res
        if kind == "event":
            ev = case.event(item_id)
            res = run_tds(case, ev.mutations, ev.profile)
            res.update({"kind": kind, "id": ev.event_id, "family": ev.family, "variant": variant,
                        "self": [i for i, b in enumerate(case.blocks) if b.asset_id in ev.disconnects_assets]})
            return res
        ctl = case.control(item_id)
        bus = _control_bus(case, ctl)
        res = run_tds(case, ctl.mutations, ctl.envelope, alpha=1.0, watch_buses=[bus] if bus is not None else [])
        res.update({"kind": kind, "id": ctl.control_id, "family": ctl.control_class, "bus": bus})
        return res
    except Exception as exc:
        return {"kind": kind, "id": item_id, "error": repr(exc), "tds_ok": False, "reached_tf": False}


def _others(run: Mapping[str, Any], key: str) -> list[float]:
    vals = list(run.get(key, [math.nan]))
    skip = set(run.get("self", []))
    kept = [v for i, v in enumerate(vals) if i not in skip]
    return kept or [math.nan]


def _core_job(args: tuple[str, str]) -> dict[str, Any]:
    slug, event_id = args
    case = _case(slug)
    ev = case.event(event_id)
    try:
        from pa_dvsa.tds_runner import run_protection_tds

        started = time.perf_counter()
        with bm._quiet():
            res = run_protection_tds(lambda: case.factory(tds_config={"criteria": 0}), ev, case.blocks,
                                     relays_enabled=False, t_seed=T_SEED, window_s=T_AFTER, t_end=T_SEED + T_AFTER,
                                     tstep=TSTEP)
        vmax = [res.asset(b.asset_id).max_v_window_pu for b in case.blocks]
        return {"id": event_id, "status": res.status, "converged": bool(res.converged),
                "collector_max_v_pu": [float(v) for v in vmax], "runtime_s": time.perf_counter() - started}
    except Exception as exc:
        return {"id": event_id, "status": "error", "converged": False, "error": repr(exc)}


def crosscheck_core_runner(slug: str, ev_res: Sequence[Mapping[str, Any]], workers: int = 1) -> dict[str, Any]:
    case = _case(slug)
    first = {}
    for ev in case.events:
        first.setdefault(ev.family, ev.event_id)
    jobs = [(slug, eid) for eid in first.values()]
    if workers > 1:
        with ProcessPoolExecutor(max_workers=min(workers, len(jobs))) as pool:
            core = list(pool.map(_core_job, jobs))
    else:
        core = [_core_job(j) for j in jobs]
    mine = {r["id"]: r for r in ev_res}
    rows = []
    for fam, (eid, c) in zip(first, zip(first.values(), core)):
        m = mine.get(eid, {})
        mv = np.asarray(_others(m, "collector_max_v_pu"), dtype=float)
        skip = set(m.get("self", []))
        cv = np.asarray([v for i, v in enumerate(c.get("collector_max_v_pu", [])) if i not in skip], dtype=float)
        diff = float(np.max(np.abs(mv - cv))) if mv.size and cv.size == mv.size else math.nan
        rows.append({"family": fam, "event": eid, "core_status": c.get("status"), "core_error": c.get("error"),
                     "local_ok": bool(m.get("tds_ok")) and bool(m.get("reached_tf")),
                     "max_abs_diff_collector_max_v_pu": diff})
    return {"rows": rows}


def ramp_screen_check(slug: str) -> dict[str, Any]:
    from pa_dvsa.event_response import EventResponseEngine
    from pa_dvsa.events import EventSpec
    from pa_dvsa.tds_runner import run_protection_tds

    out: dict[str, Any] = {}
    for variant in ("baseline", "voltage_mode_res"):
        case = _case(slug, variant)
        blk = max(case.blocks, key=lambda b: abs(b.q_mvar))
        lib = case.event(f"fixed_pf_ramp:{blk.asset_id}")
        base = read_base_values(case.system, lib.mutations)
        scale = 0.01 / (1.0 - bm.FIXED_PF_RAMP_FRACTION)
        muts = tuple(Mutation(m.model, m.device_idx, m.attr, bv + scale * (m.value - bv)) for m, bv in zip(lib.mutations, base))
        ev = EventSpec(f"fixed_pf_ramp_1pct:{blk.asset_id}", "fixed_pf_ramp", "param", muts, lib.profile)
        outs = list(case.protected_output_addresses)
        try:
            with bm._quiet():
                eng = EventResponseEngine(case.factory(tds_config={"tstep": 0.002}), outs)
                resp = eng.event_response(ev)
                addr = {b.asset_id: a for b, a in zip(case.blocks, outs)}
                tds = run_protection_tds(lambda: case.factory(tds_config={"tstep": 0.002}), ev, case.blocks,
                                         relays_enabled=False, t_seed=T_SEED, t_end=T_SEED + 1.0, tstep=0.002, tol=1e-8,
                                         record_traces=True, trace_addresses=addr)
            t = tds.traces["t"]
            x = tds.traces["extra"]
            k0 = int(np.searchsorted(t, T_SEED - 1e-9))
            post = t > T_SEED + 1e-9
            tp = np.concatenate(([0.0], t[post] - T_SEED))
            xp = np.vstack((x[k0 + 1], x[post])) - x[k0]
            dz = np.stack([np.interp(eng.times_s, tp, xp[:, j]) for j in range(x.shape[1])], axis=1)
            err = resp.values_pu - dz
            k = [b.asset_id for b in case.blocks].index(blk.asset_id)
            m03 = eng.times_s <= 0.3 + 1e-9
            out[variant] = {"block": blk.asset_id, "screen_own_dv_1s_mpu": 1e3 * float(resp.values_pu[-1, k]),
                            "tds_own_dv_1s_mpu": 1e3 * float(dz[-1, k]),
                            "max_abs_err_0_0p3_mpu": 1e3 * float(np.max(np.abs(err[m03]))),
                            "max_abs_err_0_1_mpu": 1e3 * float(np.max(np.abs(err))),
                            "max_real_eig": float(resp.diagnostics.max_real_eigenvalue)}
        except Exception as exc:
            out[variant] = {"error": repr(exc)}
    return out


def validate_variants(slug: str, workers: int = 1) -> dict[str, Any]:
    base = _case(slug)
    va = np.asarray(base.system.Bus.v.a)
    y0 = np.asarray(base.system.dae.y)
    qmax_block = max(base.blocks, key=lambda b: abs(b.q_mvar))
    rows = {}
    jobs = []
    trip_ids = {}
    for variant in bm.VARIANTS:
        case = _case(slug, variant)
        y = np.asarray(case.system.dae.y)
        same_scenario = case.design["scenario"]["name"] == base.design["scenario"]["name"]
        own_qmax = qmax_block if same_scenario else max(case.blocks, key=lambda b: abs(b.q_mvar))
        trip_ids[variant] = own_qmax.trip_event_id
        rows[variant] = {
            "scenario": case.design["scenario"]["name"],
            "max_abs_dv_vs_baseline_pu": float(np.max(np.abs(y[va] - y0[va]))) if same_scenario else None,
            "relays_disabled": bool(case.design["variant"]["relays_disabled"]),
            "dwell_s": case.blocks[0].dwell_s,
            "plant_control": case.design["variant"]["plant_control"],
        }
        jobs.append((slug, "event", own_qmax.trip_event_id, variant))
    if workers > 1:
        with ProcessPoolExecutor(max_workers=min(workers, len(jobs))) as pool:
            results = list(pool.map(_job, jobs))
    else:
        results = [_job(j) for j in jobs]
    for variant, res in zip(bm.VARIANTS, results):
        others = _others(res, "collector_max_dv_pu")
        rows[variant].update({
            "trip_event": trip_ids[variant],
            "tds_ok": bool(res.get("tds_ok")) and bool(res.get("reached_tf")),
            "neighbour_max_rise_pu": max(others),
            "neighbour_mean_final_dv_pu": float(np.nanmean(_others(res, "collector_final_dv_pu"))),
            "n_neighbours_over_pickup": sum(v >= bm.RELAY_PICKUP_PU for v in _others(res, "collector_max_v_pu")),
            "max_abs_dv_any_bus_pu": res.get("max_abs_dv_any_bus_pu"),
        })
    return rows


def _stock_reference(slug: str) -> dict[str, Any]:
    case = _case(slug, "stock_load")
    st = bm.static_validation(case)
    ne = run_tds(case, (), None, t_after=3.0 - T_SEED)
    return {"build_runtime_s": case.build_runtime_s, "static": st,
            "no_event_max_abs_dv_pu": ne["max_abs_dv_any_bus_pu"], "no_event_max_abs_omega_dev": ne["max_abs_omega_dev"],
            "voltage_profile": case.design["voltage_profile"], "dispatch": case.design["dispatch"],
            "slack": case.design["slack"], "blocks": case.design["blocks"]}


def validate_system(slug: str, workers: int = 1) -> dict[str, Any]:
    t0 = time.perf_counter()
    try:
        case = _case(slug)
    except bm.ScenarioInfeasibleError as exc:
        return bm._jsonable({"slug": slug, "feasible": False, "error": str(exc), "details": exc.details,
                             "stock_load_reference": _stock_reference(slug),
                             "validation_runtime_s": time.perf_counter() - t0})
    build_s = case.build_runtime_s
    static = bm.static_validation(case)
    jobs = [(slug, "noevent", "")] + [(slug, "event", e.event_id) for e in case.events] + \
           [(slug, "control", c.control_id) for c in case.controls]
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(_job, jobs))
    else:
        results = [_job(j) for j in jobs]
    noevent = results[0]
    ev_res = [r for r in results if r["kind"] == "event"]
    ctl_res = [r for r in results if r["kind"] == "control"]
    qmax_block = max(case.blocks, key=lambda b: abs(b.q_mvar))
    trip = next(r for r in ev_res if r["id"] == qmax_block.trip_event_id)
    interaction = {
        "tripped_block": qmax_block.asset_id,
        "tripped_q_mvar": qmax_block.q_mvar,
        "rows": [
            {"asset_id": b.asset_id, "cluster": b.metadata["cluster"], "margin_pu": bm.RELAY_PICKUP_PU - b.z0_pu,
             "max_dv_pu": trip.get("collector_max_dv_pu", [math.nan] * len(case.blocks))[i],
             "final_dv_pu": trip.get("collector_final_dv_pu", [math.nan] * len(case.blocks))[i],
             "max_v_pu": trip.get("collector_max_v_pu", [math.nan] * len(case.blocks))[i]}
            for i, b in enumerate(case.blocks)
        ],
    }
    fam = {}
    for f in bm.EVENT_FAMILIES:
        rows = [r for r in ev_res if r.get("family") == f]
        if not rows:
            fam[f] = {"n": 0}
            continue
        fam[f] = {
            "n": len(rows),
            "n_converged": sum(bool(r.get("tds_ok")) and bool(r.get("reached_tf")) for r in rows),
            "n_dv_gt_1e-4": sum(r.get("max_abs_dv_any_bus_pu", 0.0) > DV_EVENT_MIN for r in rows),
            "max_abs_dv_any_bus_pu": max(r.get("max_abs_dv_any_bus_pu", math.nan) for r in rows),
            "min_abs_dv_any_bus_pu": min(r.get("max_abs_dv_any_bus_pu", math.nan) for r in rows),
            "max_collector_rise_pu": max(max(_others(r, "collector_max_dv_pu")) for r in rows),
            "max_collector_drop_pu": min(min(_others(r, "collector_min_dv_pu")) for r in rows),
            "n_collector_over_pickup": sum(max(_others(r, "collector_max_v_pu")) >= bm.RELAY_PICKUP_PU for r in rows),
            "max_angle_spread_increase_deg": max(r.get("angle_spread_max_deg", math.nan)
                                                 - r.get("angle_spread_init_deg", math.nan) for r in rows),
            "max_abs_omega_dev": max(r.get("max_abs_omega_dev", math.nan) for r in rows),
            "n_loss_of_synchronism": sum(bool(r.get("loss_of_synchronism")) for r in rows),
            "n_collectors_over_pickup_total": sum(sum(v >= bm.RELAY_PICKUP_PU for v in _others(r, "collector_max_v_pu"))
                                                  for r in rows),
            "failures": [{"id": r["id"], "error": r.get("error"), "t_end": r.get("t_end"),
                          "message": r.get("tds_message", "")}
                         for r in rows if not (r.get("tds_ok") and r.get("reached_tf"))],
        }
    ctl = {}
    for cc in bm.CONTROL_CLASSES:
        rows = [r for r in ctl_res if r.get("family") == cc]
        if not rows:
            ctl[cc] = {"n": 0}
            continue
        finals = [r.get("watch", {}).get(str(r.get("bus")), {}).get("final_dv_pu", math.nan) for r in rows]
        early = [r.get("watch", {}).get(str(r.get("bus")), {}).get("min_dv_first_0p5s_pu", math.nan) for r in rows]
        ctl[cc] = {
            "n": len(rows),
            "n_converged": sum(bool(r.get("tds_ok")) and bool(r.get("reached_tf")) for r in rows),
            "n_bus_voltage_dropped": sum(f < -DV_EVENT_MIN for f in finals),
            "n_bus_voltage_dropped_first_0p5s": sum(f < -DV_EVENT_MIN for f in early),
            "early_min_dv_pu_min": min(early), "early_min_dv_pu_max": max(early),
            "n_loss_of_synchronism": sum(bool(r.get("loss_of_synchronism")) for r in rows),
            "final_dv_at_bus_pu_min": min(finals),
            "final_dv_at_bus_pu_max": max(finals),
            "max_abs_dv_any_bus_pu": max(r.get("max_abs_dv_any_bus_pu", math.nan) for r in rows),
        }
    variants = validate_variants(slug, workers=workers)
    cross = crosscheck_core_runner(slug, ev_res, workers=workers)
    ramp_check = ramp_screen_check(slug)
    out = {
        "ramp_screen_check": ramp_check,
        "slug": slug,
        "feasible": True,
        "stock_load_reference": _stock_reference(slug),
        "variants": variants,
        "core_runner_crosscheck": cross,
        "build_runtime_s": build_s,
        "static": static,
        "no_event": noevent,
        "families": fam,
        "controls": ctl,
        "interaction": interaction,
        "events": ev_res,
        "control_runs": ctl_res,
        "design_summary": {
            "device_counts": case.design["device_counts"],
            "slack": case.design["slack"],
            "dispatch": case.design["dispatch"],
            "operating_point": case.design["operating_point"],
            "voltage_profile": case.design["voltage_profile"],
            "scenario": case.design["scenario"],
            "uel": case.design["uel"],
            "blocks": case.design["blocks"],
            "placement": {k: v for k, v in case.design["placement"].items() if k not in ("candidates", "excluded")},
            "n_excluded_candidates": len(case.design["placement"]["excluded"]),
            "redispatch": case.design["redispatch"],
            "control_notes": case.design["control_notes"],
            "event_family_counts": case.design["event_family_counts"],
            "control_class_counts": case.design["control_class_counts"],
            "generator_trip_machines": case.design["generator_trip_machines"],
        },
        "validation_runtime_s": time.perf_counter() - t0,
    }
    return bm._jsonable(out)


def _fmt(v: Any, nd: int = 4) -> str:
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return "NaN"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def _feasibility_md(results: Mapping[str, Mapping[str, Any]]) -> list[str]:
    lines = ["### Scenario: RES share per system (baseline = light-load, high-voltage scenario)\n",
             "| System | RES share used (requested) | reason | search (share: power-flow feasible) | "
             "stock_load reference: no-event 3 s max dV (pu) |",
             "|---|---|---|---|---|"]
    for s in bm.BENCHMARK_SLUGS:
        if s not in results:
            continue
        r = results[s]
        ref = r.get("stock_load_reference") or {}
        ne = ref.get("no_event_max_abs_dv_pu")
        if not r.get("feasible", True):
            lines.append(f"| {s} | **none** | {r['error']} | - | {_fmt(ne, 6)} |")
            continue
        sc = r["design_summary"]["scenario"]
        search = ", ".join(f"{x['res_share']:.2f}: {'yes' if x['feasible'] else 'no'}" for x in sc.get("res_share_search", []))
        lines.append(f"| {s} | {sc['res_share']:.2f} ({sc.get('res_share_requested', sc['res_share']):.2f}) | "
                     f"{sc.get('res_share_reason', '')} | {search} | {_fmt(ne, 6) if ne is not None else 'n/a'} |")
    return lines


def _profile_md(results: Mapping[str, Mapping[str, Any]], slugs: Sequence[str]) -> list[str]:
    lines = ["\n### Light-load operating point: dispatch and voltage profile\n",
             "| System | load stock -> scaled (MW) | RES (MW) (share) | dispatch k | units at must-run (governor basis) | "
             "slack P sched / PF (MW) | slack Q (Mvar) [min,max] | all-bus V (pu) | HV (>=200 kV) V min/mean/max | "
             "block HV V (pu) | machines absorbing Q | machine Q total (Mvar) | Q/Sn range (rated) | "
             "buses > 1.10 pu (>= 200 kV) |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for s in slugs:
        d = results[s]["design_summary"]
        dp, vp, sl = d["dispatch"], d["voltage_profile"], d["slack"]
        bhv = list(vp["block_hv_v_pu"].values())
        flag_q = "" if sl["q_within_limits"] else " **outside**"
        lines.append(
            f"| {s} | {dp['load_stock_mw']:.0f} -> {dp['load_p_mw']:.0f} | {dp['res_p_total_mw']:.0f} "
            f"({d['scenario']['res_share']:.2f}) | "
            f"{dp['dispatch_scale']:.4f} | {dp.get('n_at_must_run', 'n/a')} of {dp.get('n_units', 'n/a')} "
            f"({dp.get('n_must_run_governor_basis', 'n/a')}) | {dp['slack_scheduled_mw']:.1f} / {dp['slack_pf_mw']:.1f} | "
            f"{sl['q_mvar']:.1f} [{sl['qmin_mvar']:.0f},{sl['qmax_mvar']:.0f}]{flag_q} | "
            f"{vp['bus_v_min_pu']:.3f}-{vp['bus_v_max_pu']:.3f} | {vp['hv_v_min_pu']:.3f} / {vp['hv_v_mean_pu']:.3f} / "
            f"{vp['hv_v_max_pu']:.3f} | {min(bhv):.3f}-{max(bhv):.3f} | {vp['n_machines_absorbing_q']} of "
            f"{vp['n_machines']} | {vp['machine_q_total_mvar']:.0f} | "
            f"{_fmt(vp['machine_q_over_sn_min'], 3)} to {_fmt(vp['machine_q_over_sn_max'], 3)} ({vp['n_machines_rated']}) | "
            f"{vp.get('n_bus_over_1p10', 'n/a')} ({vp.get('n_hv_bus_over_1p10', 'n/a')}) |")
    return lines


def _uel_md(results: Mapping[str, Mapping[str, Any]], slugs: Sequence[str]) -> list[str]:
    lines = ["\n### Condenser under-excitation limit and low-Q machine screen\n",
             "| System | UEL applied | condensers | at limit (fixed Q) | below limit without UEL | max violation (Mvar) | "
             "spurious switches | max voltage rise from UEL (pu) | machines with Q < -0.5 Sn (EMF / vf pu) |",
             "|---|---|---|---|---|---|---|---|---|"]
    for s in slugs:
        d = results[s]["design_summary"]
        u = d.get("uel") or {"applied": False}
        low = d["voltage_profile"].get("machines_q_below_minus_half_sn", [])
        low_txt = "; ".join(f"{m['idx']} ({m['model']}, Sn {m['sn_mva']:.0f}{' default' if m['sn_is_default_100'] else ''}, "
                            f"Q {m['q_mvar']:.0f}, vf {m['vf_pu']:.2f})" for m in low) or "none"
        if u.get("applied"):
            ref = u.get("reference_without_uel", {})
            lines.append(f"| {s} | yes | {u['n_condensers']} | {u['n_at_limit']} | {u['n_below_limit_without_uel']} | "
                         f"{u['max_violation_mvar']:.3f} | {u['n_spurious_conversions']} | "
                         f"{ref.get('max_voltage_rise_pu', math.nan):.4f} | {low_txt} |")
        else:
            lines.append(f"| {s} | no | - | - | - | - | - | - | {low_txt} |")
    return lines


def render_markdown(results: Mapping[str, Mapping[str, Any]]) -> str:
    lines: list[str] = _feasibility_md(results)
    slugs = [s for s in bm.BENCHMARK_SLUGS if s in results and results[s].get("feasible", True)]
    lines += _profile_md(results, slugs)
    lines += _uel_md(results, slugs)
    lines.append("\n### Model and device counts (augmented system; base case in parentheses)\n")
    keys = [("buses", "Buses"), ("lines", "Lines"), ("transformers", "Transformers"), ("machines", "Sync. machines"),
            ("exciters", "Exciters"), ("governors", "Governors"), ("pss", "PSS"), ("loads", "Loads (PQ)"),
            ("shunts", "Shunts"), ("pvd1", "PVD1 plants"), ("freq_measurement", "BusFreq")]
    lines.append("| Quantity | " + " | ".join(slugs) + " |")
    lines.append("|---|" + "---|" * len(slugs))
    for key, label in keys:
        cells = []
        for s in slugs:
            aug = results[s]["design_summary"]["device_counts"]["augmented"]
            b = results[s]["design_summary"]["device_counts"]["base_case"]
            ins = aug.get(f"{key}_in_service")
            ins_txt = f", {ins} in service" if ins is not None and ins != aug.get(key) else ""
            cells.append(f"{aug.get(key, 0)} ({b.get(key, 0)}){ins_txt}")
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    for s in slugs:
        models = results[s]["design_summary"]["device_counts"]["augmented"]
        lines.append(f"\n{s}: machines {models['machines_models']}, exciters {models['exciters_models']}, "
                     f"governors {models['governors_models']}, PSS {models['pss_models']}.")
    lines.append("\n### Operating point, dispatch and slack\n")
    lines.append("| System | Load (MW) | RES P (MW) | Dispatch scale | Slack P (MW) [min,max] | Slack Q (Mvar) [min,max] | "
                 "V range (pu) | PF conv. | TDS init | build (s) |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for s in slugs:
        d = results[s]["design_summary"]
        sl = d["slack"]
        op = d["operating_point"]
        flag_p = "" if sl["p_within_limits"] else " **outside**"
        flag_q = "" if sl["q_within_limits"] else " **outside**"
        lines.append(
            f"| {s} | {d['dispatch']['load_p_mw']:.0f} | {d['dispatch']['res_p_total_mw']:.0f} | "
            f"{d['dispatch']['dispatch_scale']:.4f} | {sl['p_mw']:.1f} [{sl['pmin_mw']:.0f},{sl['pmax_mw']:.0f}]{flag_p} | "
            f"{sl['q_mvar']:.1f} [{sl['qmin_mvar']:.0f},{sl['qmax_mvar']:.0f}]{flag_q} | "
            f"{op['v_min_pu']:.3f}-{op['v_max_non_collector_pu']:.3f} (collectors up to {op['v_max_pu']:.3f}) | "
            f"{_fmt(op['pflow_converged'])} | {_fmt(op['tds_init_test_ok'])} | {results[s]['build_runtime_s']:.2f} |")
    lines.append("\n### Plant blocks\n")
    for s in slugs:
        lines.append(f"\n**{s}** (hubs: {', '.join(str(h['seed_bus']) for h in results[s]['design_summary']['placement']['cluster_seeds'])}; "
                     f"{results[s]['design_summary']['n_excluded_candidates']} HV buses excluded by SCR/island)\n")
        lines.append("| Block | Cluster | HV bus (kV) | P (MW) | Q (Mvar) | SCR | tap | z0 design | z0 op | margin 1.10-z0 | "
                     "V_HV | drop z0-V_HV/tap | bound |Z||S|/z0 |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for r in results[s]["design_summary"]["blocks"]:
            star = "*" if r["tap_at_limit"] else ""
            lines.append(
                f"| {r['asset_id']} | {r['cluster']} | {r['hv_bus']} ({r['hv_vn_kv']:.0f}) | {r['p_op_mw']:.1f} | "
                f"{r['q_op_mvar']:.1f} | {r['scr']:.1f} | {r['tap']:.4f}{star} | {r['z0_design_pu']:.4f} | "
                f"{r['z0_op_pu']:.4f} | {r['margin_pu']:.4f} | {r['v_hv_op_pu']:.4f} | {r['drop_op_pu']:+.4f} | "
                f"{r['drop_bound_op_pu']:.4f} |")
    lines.append("\n`*` tap at a limit of the +/-10 % range: the design z0 is not reachable at the scheduled Q, z0 op is "
                 "the power-flow collector voltage (flagged, block not moved).\n")
    lines.append("### Static checks\n")
    lines.append("| System | taps in range | taps at limit | max rel. Q error | max z0 error, unclipped (pu) | "
                 "aug. alg. Jacobian | decoupled vars | rank / conditioning | no-event 3 s: max dV (pu) | max abs(omega-1) |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for s in slugs:
        st = results[s]["static"]
        ne = results[s]["no_event"]
        if st["rank_method"] == "dense":
            rk = f"rank {st['augmented_rank']}/{st['augmented_size']} (dense); coupled cond2 {st['coupled_cond_2']:.2e}"
        else:
            rk = f"sparse LU ok={_fmt(st.get('lu_success'))}; coupled cond1 est {st.get('cond_1_estimate', math.nan):.2e}"
        lines.append(
            f"| {s} | {_fmt(st['taps_within_range'])} | {st['n_taps_at_limit']} | {st['max_q_rel_error']:.1e} | "
            f"{st['max_z0_error_pu']:.1e} | {st['augmented_size']} ({st['n_zero_tf_states']} zero-Tf states) | "
            f"{st['n_decoupled']} | full rank: {_fmt(st['full_rank'])}; {rk}; CORE check: "
            f"{_fmt(st.get('core_regularity_ok'))} (cond est {st.get('core_regularity', {}).get('condition_estimate') or math.nan:.2e}) | "
            f"{ne.get('max_abs_dv_any_bus_pu', math.nan):.2e} | {ne.get('max_abs_omega_dev', math.nan):.2e} |")
    lines.append("\n### Event library TDS check (relays off; seed 0.5 s, 1.5 s after)\n")
    lines.append("| System | Family | n | converged | moved a bus > 1e-4 pu | min/max over events of max abs dV (pu) | "
                 "max collector rise (pu) | max collector drop (pu) | events with a collector >= 1.10 | "
                 "collector pickups (sum) | max angle-spread increase (deg) | max abs(omega-1) | "
                 "runs with loss of synchronism |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for s in slugs:
        for f, row in results[s]["families"].items():
            if not row.get("n"):
                lines.append(f"| {s} | {f} | 0 | - | - | - | - | - | - | - | - | - | - |")
                continue
            lines.append(
                f"| {s} | {f} | {row['n']} | {row['n_converged']} | {row['n_dv_gt_1e-4']} | "
                f"{row['min_abs_dv_any_bus_pu']:.4f} / {row['max_abs_dv_any_bus_pu']:.4f} | "
                f"{row['max_collector_rise_pu']:+.4f} | {row['max_collector_drop_pu']:+.4f} | "
                f"{row['n_collector_over_pickup']} | {row['n_collectors_over_pickup_total']} | "
                f"{row.get('max_angle_spread_increase_deg', math.nan):.1f} | {row.get('max_abs_omega_dev', math.nan):.4f} | "
                f"{row.get('n_loss_of_synchronism', 'n/a')} |")
    fails = [(s, f, x) for s in slugs for f, row in results[s]["families"].items() for x in row.get("failures", [])]
    if fails:
        lines.append("\nNon-converged / early-stopped event runs:\n")
        for s, f, x in fails:
            lines.append(f"* {s} {x['id']}: stopped at t = {_fmt(x.get('t_end'), 3)} s"
                         + (f" ({x['message'].strip()})" if x.get("message") else "")
                         + (f", error {x['error']}" if x.get("error") else ""))
    lines.append("\n### Control library TDS check (alpha = 1)\n")
    lines.append("| System | Class | n | converged | bus voltage dropped within 0.5 s | min dV at bus in 0.5 s (pu) "
                 "min / max | bus voltage lower at t_s+1.5 s | final dV at bus (pu) min / max | runs with loss of "
                 "synchronism | max abs dV any bus |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for s in slugs:
        for c, row in results[s]["controls"].items():
            if not row.get("n"):
                note = results[s]["design_summary"]["control_notes"].get(c, "none in library")
                lines.append(f"| {s} | {c} | 0 | - | - | - | - | - | - | {note} |")
                continue
            lines.append(
                f"| {s} | {c} | {row['n']} | {row['n_converged']} | {row.get('n_bus_voltage_dropped_first_0p5s', 'n/a')} | "
                f"{row.get('early_min_dv_pu_min', math.nan):+.4f} / {row.get('early_min_dv_pu_max', math.nan):+.4f} | "
                f"{row['n_bus_voltage_dropped']} | "
                f"{row['final_dv_at_bus_pu_min']:+.4f} / {row['final_dv_at_bus_pu_max']:+.4f} | "
                f"{row.get('n_loss_of_synchronism', 'n/a')} | {row['max_abs_dv_any_bus_pu']:.4f} |")
    lines.append("\nCollector columns exclude the asset disconnected by the event (for plant trips the tripped "
                 "plant's own collector floats to V_HV/tap and is a self-term).\n")
    lines.append("\n### Variants: operating point and response to the largest-abs(Q) plant trip (relays off)\n")
    lines.append("| System | Variant | max abs dV vs baseline at init (pu) | dwell (s) | relays | plant control | "
                 "neighbour max rise (pu) | neighbours reaching 1.10 | TDS ok |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for s in slugs:
        for v, row in results[s].get("variants", {}).items():
            dvb = row.get("max_abs_dv_vs_baseline_pu")
            dvb_txt = f"{dvb:.1e}" if dvb is not None else f"n/a ({row.get('scenario')} scenario)"
            lines.append(f"| {s} | {v} | {dvb_txt} | {row['dwell_s']:.2f} | "
                         f"{'off' if row['relays_disabled'] else 'on'} | {row['plant_control']} | "
                         f"{row['neighbour_max_rise_pu']:+.4f} | {row['n_neighbours_over_pickup']} | {_fmt(row['tds_ok'])} |")
    lines.append("\n### Cross-check with the CORE runner `tds_runner.run_protection_tds` (relays off, same step)\n")
    lines.append("First event of each family; max over non-disconnected collectors of the difference in the "
                 "maximum collector voltage over [t_s, t_s + 1.5 s] between the local runner and the CORE runner.\n")
    lines.append("| System | Family | event | CORE status | local ok | max abs diff (pu) |")
    lines.append("|---|---|---|---|---|---|")
    for s in slugs:
        for r in results[s].get("core_runner_crosscheck", {}).get("rows", []):
            lines.append(f"| {s} | {r['family']} | {r['event']} | {r['core_status']} | {_fmt(r['local_ok'])} | "
                         f"{_fmt(r['max_abs_diff_collector_max_v_pu'], 5)} |")
    lines.append("\n### 1 % fixed-PF ramp of the largest-abs(Q) block: linear screen (EventResponseEngine) vs TDS\n")
    lines.append("Own-collector dV at t_s + 1 s and max abs error over all collector outputs (CORE tolerance for the "
                 "30 % ramp: 0.6 / 2.2 mpu over [0, 0.3] / [0, 1] s).\n")
    lines.append("| System | Variant | block | screen dV (mpu) | TDS dV (mpu) | max err [0,0.3] s (mpu) | max err [0,1] s (mpu) |")
    lines.append("|---|---|---|---|---|---|---|")
    for s in slugs:
        for v, row in results[s].get("ramp_screen_check", {}).items():
            if "error" in row:
                lines.append(f"| {s} | {v} | - | error: {row['error']} | | | |")
                continue
            lines.append(f"| {s} | {v} | {row['block']} | {row['screen_own_dv_1s_mpu']:+.4f} | {row['tds_own_dv_1s_mpu']:+.4f} | "
                         f"{row['max_abs_err_0_0p3_mpu']:.4f} | {row['max_abs_err_0_1_mpu']:.4f} |")
    lines.append("\n### Plant-trip interaction (trip of the block with the largest abs(Q))\n")
    for s in slugs:
        it = results[s]["interaction"]
        lines.append(f"\n**{s}**: trip of {it['tripped_block']} ({it['tripped_q_mvar']:.1f} Mvar)\n")
        lines.append("| Block | Cluster | margin (pu) | max dV collector (pu) | dV at t_s+1.5 s (pu) | max V (pu) |")
        lines.append("|---|---|---|---|---|---|")
        for r in it["rows"]:
            tag = " (tripped)" if r["asset_id"] == it["tripped_block"] else ""
            lines.append(f"| {r['asset_id']}{tag} | {r['cluster']} | {r['margin_pu']:.4f} | {r['max_dv_pu']:+.4f} | "
                         f"{r['final_dv_pu']:+.4f} | {r['max_v_pu']:.4f} |")
    lines.append("\n### Validation runtimes\n")
    lines.append("| System | build (s) | TDS runs | total TDS wall time (s) | mean per TDS (s) |")
    lines.append("|---|---|---|---|---|")
    for s in slugs:
        runs = results[s].get("events", []) + results[s].get("control_runs", [])
        tt = [r.get("runtime_s", math.nan) for r in runs]
        lines.append(f"| {s} | {results[s]['build_runtime_s']:.2f} | {len(runs)} | {np.nansum(tt):.1f} | "
                     f"{np.nanmean(tt):.2f} |")
    return "\n".join(lines) + "\n"


def _write_markdown(md: str) -> Path:
    path = PROJECT_ROOT / "results" / "revised" / "benchmarks_validation.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    begin, end = "<!-- BEGIN AUTO-GENERATED VALIDATION -->", "<!-- END AUTO-GENERATED VALIDATION -->"
    text = path.read_text(encoding="utf-8") if path.exists() else f"# Benchmarks\n\n{begin}\n{end}\n"
    if begin not in text:
        text += f"\n{begin}\n{end}\n"
    head, rest = text.split(begin, 1)
    _, tail = rest.split(end, 1)
    path.write_text(head + begin + "\n" + md + end + tail, encoding="utf-8")
    return path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Revised benchmark validation")
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--systems", default=",".join(bm.BENCHMARK_SLUGS))
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--no-markdown", action="store_true")
    parser.add_argument("--render", action="store_true", help="only re-render Sec. 7 from existing JSON results")
    args = parser.parse_args(argv)
    if not (args.validate or args.render):
        parser.print_help()
        return 0
    results = {}
    out_root = PROJECT_ROOT / "results" / "revised"
    for slug in ([] if args.render else [s.strip() for s in args.systems.split(",") if s.strip()]):
        t0 = time.perf_counter()
        res = validate_system(slug, workers=args.workers)
        (out_root / slug).mkdir(parents=True, exist_ok=True)
        (out_root / slug / "benchmark_validation.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
        if res.get("feasible", True):
            bm.write_design(_case(slug))
            stale = out_root / slug / "baseline_infeasible.json"
            if stale.exists():
                stale.unlink()
        else:
            stale = out_root / slug / "case_design.json"
            if stale.exists():
                stale.unlink()
            (out_root / slug / "baseline_infeasible.json").write_text(
                json.dumps({"error": res["error"], "details": res["details"]}, indent=1), encoding="utf-8")
        bm.write_design(_case(slug, "stock_load"))
        results[slug] = res
        print(f"{slug}: validated in {time.perf_counter() - t0:.1f} s", flush=True)
    for slug in bm.BENCHMARK_SLUGS:
        if slug not in results:
            f = out_root / slug / "benchmark_validation.json"
            if f.exists():
                results[slug] = json.loads(f.read_text(encoding="utf-8"))
    md = render_markdown(results)
    if not args.no_markdown:
        print(f"wrote {_write_markdown(md)}")
    else:
        print(md)
    return 0


class BenchmarkDesignTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cases = {slug: _case(slug) for slug in ("ieee39", "npcc")}

    def test_counts_and_families(self) -> None:
        for slug, case in self.cases.items():
            spec = bm.SYSTEMS[slug]
            self.assertEqual(len(case.blocks), spec.n_blocks)
            self.assertEqual(len(case.protected_output_addresses), spec.n_blocks)
            self.assertEqual(len(case.hidden_output_data), spec.n_blocks)
            fams = {e.family for e in case.events}
            expected = set(bm.EVENT_FAMILIES)
            if case.design["redispatch"]["n_pairs"] == 0:
                expected.discard("generation_redispatch")
                self.assertEqual(case.design["redispatch"]["n_down_with_headroom"], 0)
            self.assertEqual(fams, expected)
            self.assertEqual(len(case.events_by_family("plant_trip")), spec.n_blocks)
            self.assertEqual(len(case.events_by_family("fixed_pf_ramp")), spec.n_blocks)
            self.assertEqual(len(case.events_by_family("load_disconnection")), spec.n_blocks // 2)
            self.assertEqual(len(case.events_by_family("shunt_reactor_disconnection")), spec.n_blocks // 4)
            self.assertEqual(len(case.events_by_family("capacitor_switching")), 2)
            self.assertEqual(len(case.events_by_family("line_energization")), 2)
            ids = [e.event_id for e in case.events]
            self.assertEqual(len(ids), len(set(ids)))
            classes = {c.control_class for c in case.controls}
            self.assertTrue(classes <= set(bm.CONTROL_CLASSES))

    def test_operating_point(self) -> None:
        for slug, case in self.cases.items():
            sv = bm.static_validation(case)
            self.assertTrue(sv["pflow_converged"], slug)
            self.assertTrue(sv["tds_init_test_ok"], slug)
            self.assertLess(sv["max_abs_fg_at_init"], 1e-8, slug)
            self.assertTrue(sv["taps_within_range"], slug)
            self.assertTrue(sv["q_within_1pct"], slug)
            self.assertLess(sv["max_z0_error_pu"], 1e-8, slug)
            self.assertTrue(sv["full_rank"], slug)
            lo, hi = case.design["scenario"]["tap_range"]
            rows = {r["asset_id"]: r for r in case.design["blocks"]}
            for blk in case.blocks:
                self.assertAlmostEqual(blk.q_mvar / blk.p_mw, bm.KAPPA_Q_OVER_P, delta=1e-2 * abs(bm.KAPPA_Q_OVER_P))
                self.assertGreater(bm.RELAY_PICKUP_PU - blk.z0_pu, 0.0)
                self.assertLessEqual(blk.z0_pu, bm.Z0_RANGE_PU[1] + 1e-9)
                if rows[blk.asset_id]["tap_at_limit"]:
                    self.assertTrue(abs(blk.tap - lo) < 1e-9 or abs(blk.tap - hi) < 1e-9)
                else:
                    self.assertGreaterEqual(blk.z0_pu, bm.Z0_RANGE_PU[0] - 1e-9)

    def test_block_sizing(self) -> None:
        for slug, case in self.cases.items():
            d = case.design["dispatch"]
            total = sum(b.p_mw for b in case.blocks)
            self.assertAlmostEqual(total, case.design["scenario"]["res_share"] * d["load_p_mw"], delta=1e-3 * total)
            mean = total / len(case.blocks)
            for b in case.blocks:
                self.assertLessEqual(abs(b.p_mw / mean - 1.0), bm.BLOCK_P_SPREAD + 1e-9)
                self.assertAlmostEqual(b.sn_mva, bm.PLANT_SN_OVER_P * b.p_mw, delta=1e-6 * b.sn_mva)
                x_expected = bm.GSU_X_DEVICE_PU * case.design["system_mva"] / b.sn_mva
                self.assertAlmostEqual(b.gsu_x_pu_sysbase, x_expected, delta=1e-12)
            buses = [str(b.hv_bus) for b in case.blocks]
            self.assertEqual(len(buses), len(set(buses)))

    def test_hidden_output_bound(self) -> None:
        for case in self.cases.values():
            ss = case.system
            for h in case.hidden_output_data:
                self.assertEqual(float(ss.dae.y[h.collector_v_address]), h.z0_pu)
                drop = float(ss.dae.y[h.collector_v_address]) - float(ss.dae.y[h.hv_v_address]) / h.tap
                self.assertLessEqual(abs(drop), h.drop_bound_op_pu + 1e-12)
                self.assertLessEqual(h.drop_bound_op_pu, h.drop_bound_ilim_pu)
                self.assertLess(drop, 0.0)

    def test_added_devices_do_not_change_operating_point(self) -> None:
        case = self.cases["ieee39"]
        ss = case.system
        for c in case.design["added_devices"]["capacitors_out_of_service"]:
            self.assertEqual(float(ss.Shunt.u.v[bm._pos(ss.Shunt, c["idx"])]), 0.0)
        for ln in case.design["added_devices"]["parallel_circuits_out_of_service"]:
            self.assertEqual(float(ss.Line.u.v[bm._pos(ss.Line, ln["idx"])]), 0.0)
        for grp in ("statcom_shunts_b0", "fast_reactor_shunts_b0", "manual_reactor_shunts_b0"):
            for c in case.design["added_devices"][grp]:
                self.assertEqual(float(ss.Shunt.b.v[bm._pos(ss.Shunt, c["idx"])]), 0.0)
        for r in case.design["added_devices"]["shunt_reactors_in_service"]:
            p = bm._pos(ss.Shunt, r["idx"])
            self.assertEqual(float(ss.Shunt.u.v[p]), 1.0)
            self.assertLess(float(ss.Shunt.b.v[p]), 0.0)

    def test_factory_is_reproducible(self) -> None:
        case = self.cases["ieee39"]
        ss2 = case.factory()
        np.testing.assert_array_equal(np.asarray(ss2.dae.x), np.asarray(case.system.dae.x))
        np.testing.assert_array_equal(np.asarray(ss2.dae.y), np.asarray(case.system.dae.y))
        self.assertIsNot(ss2, case.system)

    def test_generator_trip_includes_controllers(self) -> None:
        case = self.cases["ieee39"]
        for ev in case.events_by_family("generator_trip"):
            models = {m.model for m in ev.mutations}
            self.assertIn("GENROU", models)
            self.assertIn("IEEEX1", models)
            self.assertIn("TGOV1N", models)
            self.assertIn("IEEEST", models)
            self.assertFalse(ev.metadata.get("slack_linked", False))
        slack_gen = str(case.system.Slack.idx.v[0])
        tripped = set(case.design["generator_trip_machines"])
        for m in case.design["machines"]:
            if str(m["gen"]) == slack_gen:
                self.assertNotIn(m["idx"], tripped)

    def test_line_energization_clones_series_admittance(self) -> None:
        case = self.cases["ieee39"]
        for ev in case.events_by_family("line_energization"):
            ss = case.factory()
            base = read_base_values(ss, ev.mutations)
            apply_mutations(ss, ev.mutations, base, 1.0)
            idx = ev.mutations[0].device_idx
            twin = ev.metadata["parallel_to"]
            p, q = bm._pos(ss.Line, idx), bm._pos(ss.Line, twin)
            self.assertEqual(float(ss.Line.u.v[p]), 1.0)
            for attr in ("ghk", "bhk", "gh", "bh", "gk", "bk", "itap", "itap2"):
                self.assertAlmostEqual(float(getattr(ss.Line, attr).v[p]), float(getattr(ss.Line, attr).v[q]),
                                       delta=1e-9, msg=attr)

    def test_generator_trip_disables_effective_status(self) -> None:
        case = self.cases["ieee39"]
        for ev in case.events_by_family("generator_trip"):
            attrs = {(m.model, m.attr) for m in ev.mutations}
            self.assertIn(("IEEEX1", "ue"), attrs)
            self.assertIn(("TGOV1N", "ue"), attrs)

    def test_load_events_signed(self) -> None:
        for case in list(self.cases.values()) + [_case("kundur", "stock_load")]:
            for ev in case.events_by_family("load_disconnection"):
                self.assertEqual(ev.metadata["negative_q"], ev.metadata["q_mvar"] < 0)
                ss = case.system
                p = bm._pos(ss.PQ, ev.metadata["load"])
                self.assertAlmostEqual(ev.metadata["q_mvar"], float(ss.PQ.q0.v[p]) * case.design["system_mva"],
                                       delta=1e-6)

    def test_light_load_scenario(self) -> None:
        for slug, case in self.cases.items():
            ss = case.system
            raw = bm._load_raw(bm.SYSTEMS[slug])
            stock_p = {str(i): float(v) for i, v in zip(raw.PQ.idx.v, raw.PQ.p0.v)}
            stock_q = {str(i): float(v) for i, v in zip(raw.PQ.idx.v, raw.PQ.q0.v)}
            stock_v = {str(i): float(v) for i, v in zip(raw.PV.idx.v, raw.PV.v0.v)}
            for i in range(ss.PQ.n):
                key = str(ss.PQ.idx.v[i])
                self.assertAlmostEqual(float(ss.PQ.p0.v[i]), 0.6 * stock_p[key], delta=1e-12)
                self.assertAlmostEqual(float(ss.PQ.q0.v[i]), 0.6 * stock_q[key], delta=1e-12)
            for i in range(ss.PV.n):
                key = str(ss.PV.idx.v[i])
                if key in stock_v:
                    self.assertAlmostEqual(float(ss.PV.v0.v[i]), stock_v[key], delta=1e-12)
            dp = case.design["dispatch"]
            k = dp["dispatch_scale"]
            self.assertTrue(dp["feasible"])
            for u in dp["units"]:
                if u["p0_pu"] <= 0.0:
                    self.assertAlmostEqual(u["p_new_pu"], u["p0_pu"], delta=1e-9)
                    continue
                self.assertGreaterEqual(u["p_new_pu"], u["must_run_pu"] - 1e-6)
                self.assertAlmostEqual(u["p_new_pu"], max(k * u["p0_pu"], u["must_run_pu"]), delta=1e-6)
                self.assertGreaterEqual(u["must_run_pu"], min(u["p0_pu"], bm.MUST_RUN_FRACTION * u["capability_pu"]) - 1e-6)
            self.assertAlmostEqual(dp["slack_pf_mw"], dp["slack_scheduled_mw"], delta=1e-4)
            for grp in ("SynGen",):
                for model in ss.groups[grp].models.values():
                    if model.n:
                        self.assertTrue(np.all(np.asarray(model.u.v) == 1.0))
            self.assertEqual(case.design["operating_point"]["controllers_at_limit"], [])
            lo, hi = case.design["scenario"]["tap_range"]
            self.assertEqual((lo, hi), (0.90, 1.10))

    def test_redispatch_pairs_have_headroom(self) -> None:
        for case in list(self.cases.values()) + [_case("kundur", "stock_load")]:
            units = {str(u["idx"]): u for u in case.design["dispatch"].get("units", [])}
            sb = case.design["system_mva"]
            mach = {str(m["idx"]): m for m in case.design["machines"]}
            for ev in case.events_by_family("generation_redispatch"):
                meta = ev.metadata
                dp = meta["dp_mw"] / sb
                for key, sign in (("down_machine", -1.0), ("up_machine", 1.0)):
                    m = mach[str(meta[key])]
                    p = meta["down_p_mw" if sign < 0 else "up_p_mw"] / sb
                    lim = meta["governor_limits"].get(str(meta[key]), {})
                    u = units.get(str(m["gen"]))
                    if sign < 0:
                        lo = max(lim.get("vmin_pu", 0.0), (u["must_run_pu"] / sb) if u else 0.0)
                        self.assertGreaterEqual(p - dp, lo - 1e-9, ev.event_id)
                    elif lim:
                        self.assertLessEqual(p + dp, lim["vmax_pu"] + 1e-9, ev.event_id)

    def test_design_json_roundtrip(self) -> None:
        case = self.cases["ieee39"]
        text = json.dumps(case.design)
        self.assertIn("blocks", json.loads(text))


class BenchmarkScenarioTests(unittest.TestCase):
    def test_res_share_capped_by_must_run(self) -> None:
        for slug in ("kundur", "gbnetwork"):
            case = _case(slug)
            sc = case.design["scenario"]
            share = sc["res_share"]
            self.assertTrue(sc["res_share_capped"], slug)
            self.assertLess(share, sc["res_share_requested"])
            self.assertAlmostEqual(share * 100.0, round(share * 100.0), delta=1e-9)
            self.assertTrue(sc["res_share_reason"].startswith("capped by must-run:"))
            search = {round(r["res_share"], 2): r["feasible"] for r in sc["res_share_search"]}
            self.assertTrue(search[round(share, 2)])
            self.assertFalse(search[round(share + 0.01, 2)])
            self.assertFalse(search[round(sc["res_share_requested"], 2)])
            dp = case.design["dispatch"]
            for u in dp["units"]:
                if u["p0_pu"] > 0.0:
                    self.assertGreaterEqual(u["p_new_pu"], u["must_run_pu"] - 1e-6, (slug, u["idx"]))
            slack = next(u for u in dp["units"] if u["kind"] == "Slack")
            self.assertGreaterEqual(dp["slack_pf_mw"], slack["must_run_pu"] - 1e-6)
            total = sum(b.p_mw for b in case.blocks)
            self.assertAlmostEqual(total, share * dp["load_p_mw"], delta=1e-3 * total)
        for slug in ("ieee39", "npcc"):
            sc = _case(slug).design["scenario"]
            self.assertEqual(sc["res_share"], 0.50)
            self.assertFalse(sc["res_share_capped"])

    def test_gb_condenser_uel(self) -> None:
        case = _case("gbnetwork")
        uel = case.design["uel"]
        self.assertTrue(uel["applied"])
        self.assertGreater(uel["n_at_limit"], 0)
        self.assertEqual(uel["n_spurious_conversions"], 0)
        ss = case.system
        sb = case.design["system_mva"]
        for r in uel["devices"]:
            q = float(ss.PV.q.v[bm._pos(ss.PV, r["pv"])]) * sb
            self.assertGreaterEqual(q, r["q_limit_mvar"] - 1e-3, r["pv"])
            self.assertAlmostEqual(r["q_limit_mvar"], bm.UEL_Q_OVER_SN * r["sn_mva"], delta=1e-9)
            if r["fixed_q_at_limit"]:
                self.assertGreaterEqual(r["v_bus_pu"], r["v0_setpoint_pu"] - 1e-6)
        for slug in ("kundur", "ieee39", "npcc"):
            self.assertFalse(_case(slug).design["uel"]["applied"])
        self.assertFalse(_case("gbnetwork", "stock_load").design["uel"]["applied"])

    def test_gb_rating_correction(self) -> None:
        for variant in ("baseline", "stock_load"):
            case = _case("gbnetwork", variant)
            rc = case.design["preparation"]["rating_corrections"]
            self.assertGreater(len(rc), 0)
            ss = case.system
            for r in rc:
                self.assertLess(r["sn_old_mva"], r["pmax_mw"])
                self.assertAlmostEqual(r["sn_new_mva"], r["pmax_mw"] / bm.RATING_CORRECTION_PF, delta=1e-6)
                p = bm._pos(getattr(ss, r["model"]), r["idx"])
                self.assertAlmostEqual(float(getattr(ss, r["model"]).Sn.v[p]), r["sn_new_mva"], delta=1e-6)
            sb = case.design["system_mva"]
            pmax = {str(i): float(v) * sb for i, v in zip(ss.PV.idx.v, ss.PV.pmax.v)}
            pmax.update({str(i): float(v) * sb for i, v in zip(ss.Slack.idx.v, ss.Slack.pmax.v)})
            for i in range(ss.GENCLS.n):
                pm = pmax.get(str(ss.GENCLS.gen.v[i]))
                if pm is not None and pm < bm.PMAX_ABSENT_PU * sb:
                    self.assertGreaterEqual(float(ss.GENCLS.Sn.v[i]), pm - 1e-6)
            self.assertEqual(case.design["voltage_profile"]["n_internal_angle_over_90_deg"], 0)
        for slug in ("kundur", "ieee39", "npcc"):
            self.assertEqual(_case(slug).design["preparation"]["rating_corrections"], [])

    def test_genuinely_infeasible_scenario_raises(self) -> None:
        import dataclasses

        spec = bm.SYSTEMS["kundur"]
        base = bm._base_analysis(spec)
        tiny = dataclasses.replace(bm.SCENARIOS["light_load"], load_scale=0.05)
        with self.assertRaises(bm.ScenarioInfeasibleError):
            bm._solve_with_res_cap(spec, base, tiny, "kundur")

    def test_stock_load_reference(self) -> None:
        case = _case("kundur", "stock_load")
        self.assertEqual(case.design["scenario"]["name"], "stock_load")
        self.assertEqual(tuple(case.design["scenario"]["tap_range"]), (0.85, 1.15))
        total = sum(b.p_mw for b in case.blocks)
        self.assertAlmostEqual(total, 0.35 * case.design["dispatch"]["load_p_mw"], delta=1e-3 * total)
        self.assertTrue(bm.static_validation(case)["taps_within_range"])


class BenchmarkVariantTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.base = _case("ieee39")
        cls.variants = {v: bm.build_benchmark("ieee39", variant=v) for v in bm.VARIANTS
                        if v not in ("baseline", "stock_load")}

    def test_operating_point_unchanged(self) -> None:
        y0 = np.asarray(self.base.system.dae.y)
        va = np.asarray(self.base.system.Bus.v.a)
        for name, case in self.variants.items():
            y = np.asarray(case.system.dae.y)
            np.testing.assert_allclose(y[va], y0[va], atol=1e-9, err_msg=name)

    def test_variant_specifics(self) -> None:
        self.assertTrue(all(b.dwell_s == bm.RELAY_DWELL_DELAYED_S for b in self.variants["delayed_protection"].blocks))
        self.assertTrue(all(b.metadata["relays_disabled"] for b in self.variants["no_ov_protection"].blocks))
        self.assertFalse(self.variants["no_ov_protection"].relays_enabled)
        pa = self.variants["preserved_absorption"]
        for ev in pa.events_by_family("plant_trip"):
            self.assertEqual({m.model for m in ev.mutations}, {"PVD1", "Shunt"})
        vm = self.variants["voltage_mode_res"]
        ss = vm.system
        base_ss = self.base.system
        self.assertTrue(ss.TDS.test_ok)
        self.assertLess(float(np.max(np.abs(ss.dae.fg))), 1e-9)
        d = bm.VOLTAGE_MODE_BREAKPOINT_OFFSET_PU
        for i, blk in enumerate(vm.blocks):
            p = bm._pos(ss.PVD1, blk.pvd1_idx)
            v1, qmn, dqdv, vqu, qref0, qdrp, qsum = (float(getattr(ss.PVD1, n).v[p])
                                                     for n in ("v1", "qmn", "dqdv", "Vqu", "qref0", "Qdrp", "Qsum"))
            self.assertGreater(dqdv, 0.0)
            self.assertEqual(float(ss.PVD1.VQ2.zi[p]), 1.0)
            self.assertAlmostEqual(v1, blk.z0_pu - d, delta=1e-12)
            self.assertAlmostEqual(qdrp, -dqdv * d, delta=1e-12)
            q_base = float(base_ss.PVD1.qref0.v[bm._pos(base_ss.PVD1, blk.pvd1_idx)])
            self.assertAlmostEqual(qsum, q_base, delta=1e-9)
            self.assertAlmostEqual(qref0, q_base + dqdv * d, delta=1e-9)
            self.assertAlmostEqual(blk.q_mvar, q_base * vm.design["system_mva"], delta=1e-6)
            self.assertAlmostEqual(dqdv * (v1 - vqu), qmn, delta=1e-9)
            self.assertAlmostEqual(vqu - v1, bm.VOLTAGE_MODE_SAT_DV_PU, delta=1e-9)
            self.assertEqual(float(ss.PVD1.VQ1.zi[p]), 0.0)
            self.assertEqual(float(ss.PVD1.qmx.v[p]), 0.0)
        np.testing.assert_allclose(np.asarray(ss.dae.x), np.asarray(base_ss.dae.x), atol=1e-9)

    def test_preserved_absorption_reactor_matches_q0(self) -> None:
        pa = self.variants["preserved_absorption"]
        ss = pa.system
        for blk in pa.blocks:
            p = bm._pos(ss.Shunt, blk.metadata["collector_reactor"])
            b = float(ss.Shunt.b.v[p])
            self.assertEqual(float(ss.Shunt.u.v[p]), 0.0)
            self.assertAlmostEqual(b * blk.z0_pu**2 * pa.design["system_mva"], blk.q_mvar, delta=1e-3 * abs(blk.q_mvar))


class BenchmarkTDSSpotTests(unittest.TestCase):
    def test_no_event_is_steady_ieee39(self) -> None:
        case = _case("ieee39")
        res = run_tds(case, (), None, t_after=2.5)
        self.assertTrue(res["tds_ok"] and res["reached_tf"])
        self.assertLess(res["max_abs_dv_any_bus_pu"], 1e-4)
        self.assertLess(res["max_abs_omega_dev"], 1e-5)

    def test_plant_trip_and_ramp_move_voltages_ieee39(self) -> None:
        case = _case("ieee39")
        for family in ("plant_trip", "fixed_pf_ramp"):
            ev = case.events_by_family(family)[0]
            res = run_tds(case, ev.mutations, ev.profile)
            self.assertTrue(res["tds_ok"] and res["reached_tf"], family)
            self.assertGreater(res["max_abs_dv_any_bus_pu"], DV_EVENT_MIN, family)
            self.assertGreater(max(res["collector_max_dv_pu"]), 0.0, family)

    def test_statcom_lowers_its_bus_voltage_ieee39(self) -> None:
        case = _case("ieee39")
        ctl = [c for c in case.controls if c.control_class == "statcom"][0]
        res = run_tds(case, ctl.mutations, ctl.envelope, watch_buses=[ctl.metadata["bus"]])
        self.assertTrue(res["tds_ok"] and res["reached_tf"])
        self.assertLess(res["watch"][str(ctl.metadata["bus"])]["final_dv_pu"], -DV_EVENT_MIN)


class BenchmarkLibrarySlowTests(unittest.TestCase):
    @_slow
    def test_full_libraries(self) -> None:
        for slug in bm.BENCHMARK_SLUGS:
            res = validate_system(slug, workers=int(os.environ.get("PA_DVSA_WORKERS", "1")))
            ref = res["stock_load_reference"]
            self.assertLess(ref["no_event_max_abs_dv_pu"], 1e-4, slug)
            self.assertTrue(res["feasible"], slug)
            self.assertTrue(res["static"]["pflow_converged"], slug)
            self.assertLess(res["no_event"]["max_abs_dv_any_bus_pu"], 1e-4, slug)
            for r in res["events"]:
                self.assertGreater(r.get("max_abs_dv_any_bus_pu", 0.0), DV_EVENT_MIN, f"{slug} {r['id']}")
            for c, row in res["controls"].items():
                if row.get("n") and c != "manual_reactor":
                    self.assertEqual(row["n_bus_voltage_dropped_first_0p5s"], row["n"], f"{slug} {c}")
                    self.assertEqual(row["n_bus_voltage_dropped"], row["n"], f"{slug} {c}")
                    self.assertEqual(row.get("n_loss_of_synchronism", 0), 0, f"{slug} {c}")
            slipping = [r for r in res["events"] if r.get("loss_of_synchronism")]
            known = KNOWN_TRANSIENT_FRAGILITY.get(slug)
            if known:
                self.assertTrue(0 < len(slipping) <= known["max_slipping_runs"],
                                f"{slug}: {len(slipping)} slipping runs -- documented finding changed, update docs")
                self.assertTrue({r["family"] for r in slipping} <= known["families"], slug)
            else:
                self.assertEqual(len(slipping), 0, slug)

    @_slow
    def test_all_variants_build(self) -> None:
        for slug in bm.BENCHMARK_SLUGS:
            for variant in bm.VARIANTS:
                case = bm.build_benchmark(slug, variant=variant)
                self.assertEqual(case.design["variant"]["name"], variant)
                if variant != "stock_load":
                    self.assertEqual(case.design["scenario"]["res_share"],
                                     _case(slug).design["scenario"]["res_share"])


if __name__ == "__main__":
    sys.exit(main())
