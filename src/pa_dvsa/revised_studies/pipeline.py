"""Runs every benchmark study for one system and writes its results."""

from __future__ import annotations

from collections import Counter
import math
import statistics
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from . import config as C
from . import mitigation_study as M
from . import observability as O
from . import scoring as SC
from . import triage as TR
from .io import read_json, write_csv, write_json
from .screen import ScreenResult, SeedPrediction, designated_seed, peak_time, predict_all, predict_seed, run_screen
from .tds_jobs import TDSPool, case_fingerprint, make_job, quiet, register_case, tds_job


def _fin(v: Any) -> float:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return float("nan")
    return f


def _median(vals: Sequence[float]) -> float:
    vals = [v for v in vals if v is not None and math.isfinite(v)]
    return float(statistics.median(vals)) if vals else float("nan")


def dependencies(requested: Sequence[str]) -> set[str]:
    req = set(requested)
    run = set(req) | {"S1", "S2"}
    if req & {"S8", "S10", "S11", "S12"}:
        run.add("S3")
    if "S12" in req:
        run.add("S4")
    return run


class SystemStudy:
    def __init__(
        self,
        slug: str,
        *,
        studies: Sequence[str],
        pool: TDSPool,
        out_root: Path = C.RESULTS_ROOT,
        log: Callable[[str], None] = print,
        reuse_tds: bool = False,
        ablation_variants: bool = True,
        subdir: str = "",
        rescore: bool = False,
    ) -> None:
        self.slug = slug
        self.requested = tuple(studies)
        self.run_set = dependencies(studies)
        self.pool = pool
        self.out = Path(out_root) / slug / subdir if subdir else Path(out_root) / slug
        self.fig = self.out / "fig_data"
        self.log = log
        self.reuse_tds = reuse_tds
        self.rescore = rescore
        self.ablation_variants = ablation_variants
        self.timing: dict[str, Any] = {}
        self.sanity: dict[str, Any] = {}
        self.anomalies: list[str] = []

    def build(self) -> None:
        from ..benchmarks import build_benchmark

        (self.out / "run_error.json").unlink(missing_ok=True)

        t0 = time.perf_counter()
        with quiet():
            self.case = build_benchmark(self.slug)
        self.timing["build_wall_s"] = time.perf_counter() - t0
        self.timing["build_runtime_s_reported"] = float(self.case.build_runtime_s)
        register_case(self.case)
        self.fp = case_fingerprint(self.case)
        t0 = time.perf_counter()
        with quiet():
            ss = self.case.factory()
        self.timing["tds_init_fresh_system_s"] = time.perf_counter() - t0
        del ss
        self.events = {e.event_id: e for e in self.case.events}

    def s1_s2(self) -> None:
        self.log(f"[{self.slug}] S1 screen: {len(self.case.events)} events, {len(self.case.controls)} controls")
        with quiet():
            self.scr: ScreenResult = run_screen(self.case)
        self.timing.update(self.scr.timing_s)
        if self.scr.failed_events:
            self.anomalies.append(f"screen failed for events: {self.scr.failed_events}")
        t0 = time.perf_counter()
        self.preds: dict[str, SeedPrediction] = predict_all(self.scr)
        self.timing["cascade_total_s"] = time.perf_counter() - t0
        self.timing["cascade_per_seed_s"] = [p.runtime_s for p in self.preds.values()]
        self.designated = designated_seed(self.preds)
        models, theta, _ = O.regime_setup(self.scr, self.case, "telemetry")
        self.telemetry_bounds = O.bounds_for(self.scr, models, theta)
        self.preds_telemetry = {
            e: predict_seed(self.telemetry_bounds, e, responses=self.scr.response_list, amap=self.scr.amap,
                            relays_enabled=self.scr.relays_enabled)
            for e in self.scr.event_order if e in self.scr.responses
        }
        if self.rescore:
            self._restore_committed_timings()
        n_trip = sum(1 for p in self.preds.values() if p.predicted_trips)
        self.log(f"[{self.slug}] S2: {n_trip} seeds with predicted secondary trips; designated seed {self.designated}")

    def _restore_committed_timings(self) -> None:
        ev = {r["event_id"]: r for r in _read_rows(self.out / "screen_events.csv")}
        ct = {r["control_id"]: r for r in _read_rows(self.out / "screen_controls.csv")}
        for e in list(self.scr.event_runtime_s):
            if e in ev and ev[e].get("runtime_s") not in (None, ""):
                self.scr.event_runtime_s[e] = float(ev[e]["runtime_s"])
        for c in list(self.scr.control_runtime_s):
            if c in ct and ct[c].get("runtime_s") not in (None, ""):
                self.scr.control_runtime_s[c] = float(ct[c]["runtime_s"])

    def s4_lps(self) -> None:
        dseeds = [e for e, p in self.preds.items() if p.predicted_trips]
        self.mit_seeds = dseeds
        self.plot_seed = self.designated if self.designated in dseeds else (dseeds[0] if dseeds else None)
        self.lps: dict[tuple[str, str], dict[str, Any]] = {}
        self.unavail: dict[str, dict[str, str]] = {}
        lp_times = []
        for e in dseeds:
            seed = self.events[e]
            self.unavail[e] = M.unavailable_controls(self.case, seed)
            for restr in C.LP_RESTRICTIONS:
                rec = M.solve_lp(self.case, self.scr.responses[e], self.scr.control_responses,
                                 self.scr.models_direct, restriction=restr, unavailable=self.unavail[e])
                lp_times.append(rec["lp_runtime_s"])
                self.lps[(e, restr)] = rec
        self.timing["lp_per_solve_s"] = lp_times
        self.log(f"[{self.slug}] S4: {len(dseeds)} designated seeds, {len(self.lps)} LPs")

    def tds_jobs_baseline(self) -> list[dict[str, Any]]:
        jobs = []
        if "S3" in self.run_set and not self.reuse_tds:
            for e in self.case.events:
                jobs.append(make_job(f"S3:{e.event_id}", self.case, e.event_id, kind="S3", fingerprint=self.fp,
                                     all_bus_dv=True))
        if "S4" in self.run_set:
            for (e, restr), rec in self.lps.items():
                if restr != "all" and e != self.plot_seed:
                    continue
                ctrls = M.applied_controls(rec.get("alphas", {}))
                if not ctrls:
                    continue
                jobs.append(make_job(f"S4:{e}:{restr}", self.case, e, kind="S4", controls=ctrls, fingerprint=self.fp))
        self.s4_job_ids = [j["job_id"] for j in jobs if j["kind"] == "S4"]
        return jobs

    def s5(self) -> None:
        self.obs_rows: list[dict[str, Any]] = []
        self.obs_seed_rows: list[dict[str, Any]] = []
        self.obs_summary: list[dict[str, Any]] = []
        self.regime_bounds: dict[str, Any] = {}
        self.regime_setups: dict[str, Any] = {}
        for regime in O.REGIMES + O.SENSITIVITY_REGIMES:
            b, models, theta, info = O.regime_bounds(self.scr, self.case, regime)
            self.regime_bounds[regime] = b
            self.regime_setups[regime] = (models, theta, info)
            rows = O.asset_rows(self.scr, b, info, regime)
            for r in rows:
                r["system"] = self.slug
                r["sensitivity_only"] = regime in O.SENSITIVITY_REGIMES
                for k in ("drop_lower_pu", "drop_upper_pu", "drop_op_pu", "drop_bound_op_pu", "drop_bound_ilim_pu", "upper_rule",
                          "p_hv_pu", "q_hv_pu", "z_reconstructed_pu", "gsu_z_rel", "tap_band_rel"):
                    r[k] = info[r["asset_id"]].get(k)
            self.obs_rows.extend(rows)
            verdicts = O.seed_verdicts(self.scr, b)
            n_cert = 0
            for e, v in verdicts.items():
                n_cert += int(v["certified_no_secondary"])
                self.obs_seed_rows.append({"system": self.slug, "regime": regime, "seed_id": e,
                                           "family": self.events[e].family, **v})
            cls = Counter(r["asset_class"] for r in rows)
            n = len(rows)
            h = np.array([r["h_lower_pu"] for r in rows], dtype=float)
            self.obs_summary.append({
                "system": self.slug, "regime": regime, "sensitivity_only": regime in O.SENSITIVITY_REGIMES,
                "n_assets": n,
                **{f"share_{k}": (cls.get(k, 0) / n if n else float("nan")) for k in ("data_limited", "pickup_predicted", "risky", "certified")},
                **{f"n_{k}": cls.get(k, 0) for k in ("data_limited", "pickup_predicted", "risky", "certified")},
                "h_lower_min_pu": float(h.min()), "h_lower_median_pu": float(np.median(h)), "h_lower_max_pu": float(h.max()),
                "n_seeds": len(verdicts), "n_seeds_certified_no_secondary": n_cert,
            })
        self.log(f"[{self.slug}] S5 done: " + ", ".join(f"{s['regime']}: DL {s['n_data_limited']}/{s['n_assets']}" for s in self.obs_summary))

    def s6(self) -> None:
        self.unc_rows: list[dict[str, Any]] = []
        self.unc_lp_rows: list[dict[str, Any]] = []
        order = [e for e in self.scr.event_order if e in self.scr.responses]
        nominal_scores = {e: (self.preds[e].max_k_pk_nonself, self.preds[e].max_k_tr_nonself) for e in order}
        lp_seeds = self._s6_lp_seeds()
        scaled_cache: dict[float, dict[str, Any]] = {}
        for cls in O.UNCERTAINTY_CLASSES:
            for lvl in C.UNCERTAINTY_LEVELS:
                models, theta, width = O.uncertainty_theta(self.case, self.scr, cls, lvl)
                b = O.uncertainty_bounds(self.scr, self.case, models, theta)
                met = O.robust_margin_metrics(b)
                verdicts = O.seed_verdicts(self.scr, b)
                n_cert = sum(1 for v in verdicts.values() if v["certified_no_secondary"])
                sc = O.seed_scores(b, order)
                rk = O.rank_compare(nominal_scores, sc, order)
                self.unc_rows.append({
                    "system": self.slug, "class": cls, "level": lvl, "width": width, **met,
                    "n_seeds": len(verdicts), "n_seeds_certified_no_secondary": n_cert,
                    "n_seeds_pickup_predicted": sum(1 for v in verdicts.values() if v["n_pickup_predicted"] > 0),
                    "n_seeds_with_data_limited": sum(1 for v in verdicts.values() if v["n_data_limited"] > 0),
                    "kendall_tau": rk["kendall_tau"], "top5_overlap": rk["top5_overlap"], "n_ranked": rk["n_ranked"],
                    "top5_nominal": rk["top5_nominal"], "top5_level": rk["top5_level"],
                })
                if cls == "controller_timing":
                    if lvl not in scaled_cache:
                        ctrls = O.scaled_controls(self.case, 1.0 + lvl)
                        with quiet():
                            resp = {c.control_id: self.scr.engine.control_response(c) for c in ctrls}
                        scaled_cache[lvl] = {"controls": ctrls, "responses": resp}
                    ctrl_resp = scaled_cache[lvl]["responses"]
                    ctrl_specs = scaled_cache[lvl]["controls"]
                else:
                    ctrl_resp = self.scr.control_responses
                    ctrl_specs = list(self.case.controls)
                for e in lp_seeds:
                    rec = M.solve_lp(self.case, self.scr.responses[e], ctrl_resp, models, restriction="all",
                                     unavailable=self.unavail.get(e) or M.unavailable_controls(self.case, self.events[e]),
                                     uncertainty=theta, controls=ctrl_specs)
                    mv = M.mvar_by_class(self.case, rec.get("alphas", {}))
                    self.unc_lp_rows.append({
                        "system": self.slug, "class": cls, "level": lvl, "seed_id": e, "status": rec["status"],
                        "feasible": rec["feasible"], "eta": rec["eta"],
                        "mvar_total_rated": sum(v for v in mv.values() if math.isfinite(v)),
                        **{f"mvar_{k}": mv.get(k, 0.0) for k in ("statcom", "plant_q_command", "fast_reactor", "manual_reactor")},
                        "avr_alpha_sum": sum(a for cid, a in rec.get("alphas", {}).items() if cid.startswith("avr_vref")),
                        "n_data_limited_excluded": len(rec.get("data_limited_assets", [])),
                        "lp_runtime_s": rec["lp_runtime_s"],
                    })
        self.log(f"[{self.slug}] S6 done ({len(self.unc_rows)} class-levels, {len(self.unc_lp_rows)} LPs)")

    def _s6_lp_seeds(self, cap: int = 12) -> list[str]:
        if not getattr(self, "mit_seeds", None):
            return []
        ranked = sorted(self.mit_seeds, key=lambda e: (not bool(self.lps.get((e, "all"), {}).get("feasible")),
                                                        -len(self.preds[e].fixed_point), -_nz(self.preds[e].max_k_pk_nonself),
                                                        self.scr.event_order.index(e)))
        return ranked[:cap]

    def s7_prepare(self) -> list[dict[str, Any]]:
        self.abl: dict[str, dict[str, Any]] = {}
        jobs: list[dict[str, Any]] = []
        d = self.designated
        if d is None:
            return jobs
        if self.ablation_variants:
            for variant in ("no_ov_protection", "delayed_protection", "preserved_absorption", "voltage_mode_res"):
                from ..benchmarks import build_benchmark

                t0 = time.perf_counter()
                try:
                    with quiet():
                        case_v = build_benchmark(self.slug, variant=variant)
                    register_case(case_v)
                    with quiet():
                        scr_v = run_screen(case_v)
                    preds_v = predict_all(scr_v)
                except Exception as exc:
                    self.abl[variant] = {"error": f"{type(exc).__name__}: {exc}"}
                    self.anomalies.append(f"S7 variant {variant} failed: {exc}")
                    continue
                pv = preds_v.get(d)
                unav = M.unavailable_controls(case_v, case_v.event(d))
                lp = M.solve_lp(case_v, scr_v.responses[d], scr_v.control_responses, scr_v.models_direct,
                                restriction="all", unavailable=unav)
                self.abl[variant] = {"case": case_v, "scr": scr_v, "pred": pv, "preds": preds_v, "lp": lp,
                                     "wall_s": time.perf_counter() - t0}
                jobs.append(make_job(f"S7:{variant}:{d}", case_v, d, kind="S7", fingerprint=case_fingerprint(case_v)))
                ctrls = M.applied_controls(lp.get("alphas", {}))
                if ctrls and case_v.relays_enabled:
                    jobs.append(make_job(f"S7mit:{variant}:{d}", case_v, d, kind="S7mit", controls=ctrls,
                                         fingerprint=case_fingerprint(case_v)))
                self.log(f"[{self.slug}] S7 {variant}: screened in {self.abl[variant]['wall_s']:.1f} s")
        if "S3" not in self.run_set:
            jobs.append(make_job(f"S3:{d}", self.case, d, kind="S7base", fingerprint=self.fp))
        unav = self.unavail.get(d) or M.unavailable_controls(self.case, self.events[d])
        lp_nofast = self.lps.get((d, "all_but_fast_reactor")) or M.solve_lp(
            self.case, self.scr.responses[d], self.scr.control_responses, self.scr.models_direct,
            restriction="all_but_fast_reactor", unavailable=unav)
        lp_all = self.lps.get((d, "all")) or M.solve_lp(
            self.case, self.scr.responses[d], self.scr.control_responses, self.scr.models_direct,
            restriction="all", unavailable=unav)
        self.abl["fast_automatic_shunt"] = {"lp_nofast": lp_nofast, "lp_all": lp_all}
        s4_ids = set(getattr(self, "s4_job_ids", ()))
        for tag, restr, lp in (("nofast", "all_but_fast_reactor", lp_nofast), ("all", "all", lp_all)):
            ctrls = M.applied_controls(lp.get("alphas", {}))
            if ctrls and f"S4:{d}:{restr}" not in s4_ids:
                jobs.append(make_job(f"S7fast:{tag}:{d}", self.case, d, kind="S7fast", controls=ctrls, fingerprint=self.fp))
        models, theta, _ = O.regime_setup(self.scr, self.case, "transmission_only")
        b = self.regime_bounds.get("transmission_only") if hasattr(self, "regime_bounds") else None
        b = b or O.bounds_for(self.scr, models, theta)
        pm = predict_seed(b, d, responses=self.scr.response_list, amap=self.scr.amap, relays_enabled=True)
        lp_m = M.solve_lp(self.case, self.scr.responses[d], self.scr.control_responses, models, restriction="all",
                          unavailable=unav, uncertainty=theta)
        self.abl["missing_collector_telemetry"] = {"pred": pm, "bounds": b, "lp": lp_m}
        return jobs

    def s3_score(self) -> None:
        self.val_rows: list[dict[str, Any]] = []
        self.seed_rows: list[dict[str, Any]] = []
        regimes = (("nominal", self.preds), ("telemetry_theta", self.preds_telemetry),
                   ("nominal_refined", {e: p.refined() for e, p in self.preds.items()}))
        self.summaries: dict[str, dict[str, dict[str, Any]]] = {r: {} for r, _ in regimes}
        for regime, preds in regimes:
            for e in self.scr.event_order:
                ev = self.events[e]
                r = self.tds.get(f"S3:{e}")
                if r is None:
                    continue
                ok = r.get("status") == "ok" and bool(r.get("converged"))
                if e not in preds:
                    self.anomalies.append(f"no screen prediction for {e} (response failed); not scored")
                    continue
                p = preds[e]
                tds_assets = {a["asset_id"]: a for a in r.get("assets", [])}
                rows, summ = SC.score_seed(seed_id=e, family=ev.family, self_assets=ev.disconnects_assets,
                                           screen=p.classes, tds_assets=tds_assets, tds_ok=ok, window_s=C.T_WINDOW_S)
                for row in rows:
                    row.update({"system": self.slug, "regime": regime, "tds_status": r.get("status")})
                self.val_rows.extend(rows)
                summ.update({"system": self.slug, "regime": regime, "tds_status": r.get("status"),
                             "tds_err_msg": r.get("err_msg") or r.get("error"),
                             "tds_runtime_s": r.get("runtime_s"), "tds_build_s": r.get("build_s"), "tds_sim_s": r.get("sim_s"),
                             "tds_trip_order": r.get("trip_order"), "tds_pre_seed_max_dv_pu": r.get("pre_seed_max_dv_pu"),
                             "max_abs_dv_all_buses_pu": r.get("max_abs_dv_all_buses_pu"),
                             "max_k_pk_nonself": p.max_k_pk_nonself, "max_k_tr_nonself": p.max_k_tr_nonself,
                             "screen_verdict": p.verdict, "screen_runtime_s": self.scr.event_runtime_s.get(e),
                             "cascade_runtime_s": p.runtime_s,
                             "late_trips": [a["asset_id"] for a in r.get("assets", []) if a.get("trip_rel_s") is not None
                                            and a["trip_rel_s"] > C.T_WINDOW_S + SC.WINDOW_TOL_S],
                             "tds_pickups_in_window": [a["asset_id"] for a in r.get("assets", []) if a.get("active", True)
                                                       and a.get("first_pickup_rel_s") is not None
                                                       and a["first_pickup_rel_s"] <= C.T_WINDOW_S + SC.WINDOW_TOL_S
                                                       and a["asset_id"] not in ev.disconnects_assets]})
                self.seed_rows.append(summ)
                self.summaries[regime][e] = summ
        self.val_summary: list[dict[str, Any]] = []
        for regime, _ in regimes:
            summs = list(self.summaries[regime].values())
            fams = sorted({s["family"] for s in summs}, key=lambda f: [ev.family for ev in self.case.events].index(f))
            for fam in fams:
                grp = [s for s in summs if s["family"] == fam]
                agg = SC.aggregate(grp, group={"system": self.slug, "regime": regime, "family": fam})
                agg.update(self._runtime_group([s["seed_id"] for s in grp]))
                self.val_summary.append(agg)
            agg = SC.aggregate(summs, group={"system": self.slug, "regime": regime, "family": "ALL"})
            agg.update(self._runtime_group([s["seed_id"] for s in summs]))
            self.val_summary.append(agg)
        allrow = [a for a in self.val_summary if a["family"] == "ALL" and a["regime"] == "nominal"][0]
        self.log(f"[{self.slug}] S3: seeds {allrow['n_seeds']}, dangerous {allrow['n_dangerous']}, TP {allrow['TP']}, "
                 f"FP {allrow['FP']}, caught_risky {allrow['caught_risky']}, unsafe_FN {allrow['unsafe_FN']}, "
                 f"inconclusive {allrow['n_seeds_inconclusive']}")

    def _runtime_group(self, seeds: Sequence[str]) -> dict[str, Any]:
        scr_t = [self.scr.event_runtime_s[e] for e in seeds if e in self.scr.event_runtime_s]
        tds_t = [_fin(self.tds.get(f"S3:{e}", {}).get("runtime_s")) for e in seeds]
        casc = [self.preds[e].runtime_s for e in seeds if e in self.preds]
        return {
            "screen_event_s_median": _median(scr_t), "screen_event_s_total": float(sum(scr_t)),
            "cascade_s_total": float(sum(casc)),
            "tds_s_median": _median(tds_t), "tds_s_total": float(sum(v for v in tds_t if math.isfinite(v))),
        }

    def s4_rows(self) -> None:
        self.mit_rows, self.mit_alpha_rows, self.mit_tds_rows = [], [], []
        classes = ("statcom", "plant_q_command", "fast_reactor", "manual_reactor")
        for (e, restr), rec in self.lps.items():
            p = self.preds[e]
            deadline = p.first_secondary_time_s if p.first_secondary_time_s is not None else C.T_WINDOW_S
            mv = M.mvar_by_class(self.case, rec.get("alphas", {}))
            alphas = rec.get("alphas", {})
            allowed = C.LP_RESTRICTIONS[restr]
            cls_list = list(allowed) if allowed else list(dict.fromkeys(c.control_class for c in self.case.controls))
            row = {
                "system": self.slug, "seed_id": e, "family": self.events[e].family, "is_plot_seed": e == self.plot_seed,
                "restriction": restr, "status": rec["status"], "feasible": rec["feasible"], "eta": rec["eta"],
                "objective": rec["objective"], "solver_status": rec.get("solver_status"),
                "n_controls_considered": rec["n_controls_considered"],
                "controls_excluded_unavailable": rec["controls_excluded_unavailable"],
                **{f"mvar_{k}": mv.get(k, 0.0) for k in classes},
                "mvar_total_rated": sum(v for v in mv.values() if math.isfinite(v)),
                "avr_alpha_sum": sum(a for cid, a in alphas.items() if cid.startswith("avr_vref")),
                "avr_dvref_pu_sum": M.avr_dvref(self.case, alphas),
                "n_binding": len(rec.get("binding", [])),
                "binding_assets": sorted({a for a, _ in rec.get("binding", [])}),
                "binding_times_s": sorted({round(t, 6) for _, t in rec.get("binding", [])})[:20],
                "residual_max": rec.get("residual_max"),
                "data_limited_assets": rec.get("data_limited_assets"), "excluded_assets": rec.get("excluded_assets"),
                "ineffective_controls": rec.get("ineffective_controls"),
                "lp_runtime_s": rec["lp_runtime_s"],
                "predicted_trips": p.predicted_trips, "deadline_s": deadline,
                "available_mvar": sum(_nz(M.available_mvar(self.case, k, self.unavail[e])) for k in cls_list if k != "avr_vref"),
                "deliverable_mvar_by_deadline": sum(_nz(M.deliverable_mvar(self.case, k, self.unavail[e], deadline)) for k in cls_list if k != "avr_vref"),
                "deliverable_mvar_by_T": sum(_nz(M.deliverable_mvar(self.case, k, self.unavail[e], C.T_WINDOW_S)) for k in cls_list if k != "avr_vref"),
            }
            self.mit_rows.append(row)
            for c in self.case.controls:
                if allowed is not None and c.control_class not in allowed:
                    continue
                a = alphas.get(c.control_id)
                self.mit_alpha_rows.append({
                    "system": self.slug, "seed_id": e, "restriction": restr, "control_id": c.control_id,
                    "control_class": c.control_class, "available": c.control_id not in self.unavail[e],
                    "unavailable_reason": self.unavail[e].get(c.control_id),
                    "alpha": a, "alpha_max": c.alpha_max,
                    "mvar": (a * c.mvar_per_unit) if (a is not None and c.mvar_per_unit is not None) else None,
                    "mvar_per_unit": c.mvar_per_unit,
                    "saturated": (a is not None and a >= c.alpha_max - 1e-7),
                })
            r = self.tds.get(f"S4:{e}:{restr}")
            if r is None:
                if restr == "all" or e == self.plot_seed:
                    self.mit_tds_rows.append({"system": self.slug, "seed_id": e, "restriction": restr,
                                              "tds_status": "not_run", "reason": "LP selected no control" if not M.applied_controls(alphas) else "not requested",
                                              "lp_feasible": rec["feasible"], "lp_eta": rec["eta"]})
                continue
            self.mit_tds_rows.append(self._mit_tds_row(e, restr, rec, r))

    def _mit_tds_row(self, e: str, restr: str, rec: Mapping[str, Any], r: Mapping[str, Any]) -> dict[str, Any]:
        ev = self.events[e]
        ok = r.get("status") == "ok" and bool(r.get("converged"))
        trips = [a for a in r.get("tripped_within_window", []) if a not in ev.disconnects_assets]
        late = [a["asset_id"] for a in r.get("assets", []) if a.get("trip_rel_s") is not None and a["trip_rel_s"] > C.T_WINDOW_S + SC.WINDOW_TOL_S]
        pick = [a["asset_id"] for a in r.get("assets", []) if a.get("active", True) and a.get("first_pickup_rel_s") is not None
                and a["first_pickup_rel_s"] <= C.T_WINDOW_S + SC.WINDOW_TOL_S and a["asset_id"] not in ev.disconnects_assets]
        worst = _max_erosion_before_trip(r, self.case.blocks, ev.disconnects_assets)
        s3 = self.tds.get(f"S3:{e}", {})
        unmit = [a for a in s3.get("tripped_within_window", []) if a not in ev.disconnects_assets]
        return {
            "system": self.slug, "seed_id": e, "family": ev.family, "restriction": restr, "tds_status": r.get("status"),
            "tds_err": r.get("err_msg") or r.get("error"),
            "n_controls_applied": len(r.get("metadata", {}).get("controls", [])),
            "controls_applied": [f"{c}={a:.4f}" for c, a in r.get("metadata", {}).get("controls", [])],
            "lp_feasible": rec["feasible"], "lp_eta": rec["eta"],
            "trips_in_window": trips, "n_trips_in_window": len(trips), "late_trips": late,
            "pickups_in_window": pick, "max_normalized_erosion": worst[0], "max_erosion_asset": worst[1],
            "verified_no_trip": bool(ok and not trips),
            "verified_no_pickup": bool(ok and not pick),
            "no_trip_to_tds_end": bool(ok and not trips and not late),
            "unmitigated_trips_in_window": unmit, "n_unmitigated_trips": len(unmit),
            "tds_runtime_s": r.get("runtime_s"),
        }

    def s7_rows(self) -> None:
        self.abl_rows = []
        d = self.designated
        if d is None:
            return

        def lp_fields(lp: Mapping[str, Any], case: Any) -> dict[str, Any]:
            mv = M.mvar_by_class(case, lp.get("alphas", {}))
            return {"lp_status": lp.get("status"), "lp_feasible": lp.get("feasible"), "lp_eta": lp.get("eta"),
                    "lp_mvar_total_rated": sum(v for v in mv.values() if math.isfinite(v)),
                    "lp_mvar_by_class": [f"{k}={v:.2f}" for k, v in sorted(mv.items()) if math.isfinite(v) and v > 1e-9],
                    "lp_avr_alpha_sum": sum(a for cid, a in lp.get("alphas", {}).items() if cid.startswith("avr_vref")),
                    "lp_n_data_limited_excluded": len(lp.get("data_limited_assets", []))}

        def tds_fields(r: Mapping[str, Any] | None, ev: Any, prefix: str = "tds") -> dict[str, Any]:
            if r is None:
                return {f"{prefix}_status": "not_run"}
            trips = [a for a in r.get("tripped_within_window", []) if a not in ev.disconnects_assets]
            would = [a["asset_id"] for a in r.get("assets", []) if a.get("would_trip_s") is not None
                     and a["would_trip_s"] - C.T_SEED_S <= C.T_WINDOW_S + SC.WINDOW_TOL_S
                     and a["asset_id"] not in ev.disconnects_assets]
            return {f"{prefix}_status": r.get("status"), f"{prefix}_trips_in_window_count": len(trips),
                    f"{prefix}_trips_in_window": trips, f"{prefix}_would_trip_in_window": would,
                    f"{prefix}_relays_enabled": r.get("relays_enabled")}

        def pred_fields(p: SeedPrediction | None) -> dict[str, Any]:
            if p is None:
                return {"verdict": "not_available"}
            cnt = Counter(c["asset_class"] for c in p.classes.values())
            return {"worst_k_pk_nonself": p.max_k_pk_nonself, "worst_k_tr_nonself": p.max_k_tr_nonself,
                    "worst_asset": p.argmax_asset, "fixed_point_size": len(p.fixed_point),
                    "predicted_trip_set_size": len(p.predicted_trips), "predicted_trips": p.predicted_trips,
                    "verdict": p.verdict, "n_data_limited": cnt.get("data_limited", 0),
                    "first_secondary": p.first_secondary, "first_secondary_time_s": p.first_secondary_time_s}

        ev = self.events[d]
        base = {"system": self.slug, "designated_seed": d, "family": ev.family}
        lib = lambda preds: {"library_worst_k_pk_nonself": max((_nz(p.max_k_pk_nonself) for p in preds.values()), default=float("nan")),
                             "library_n_seeds_trip_predicted": sum(1 for p in preds.values() if p.predicted_trips)}
        self.abl_rows.append({**base, "row": "baseline", "variant": "baseline", "dwell_s": self.case.blocks[0].dwell_s,
                              **pred_fields(self.preds[d]), **tds_fields(self.tds.get(f"S3:{d}"), ev),
                              **lp_fields(self.lps.get((d, "all")) or self.abl["fast_automatic_shunt"]["lp_all"], self.case),
                              **tds_fields(self.tds.get(f"S4:{d}:all") or self.tds.get(f"S7fast:all:{d}"), ev, "mitigated_tds"),
                              **lib(self.preds), "notes": ""})
        for variant in ("no_ov_protection", "delayed_protection", "preserved_absorption", "voltage_mode_res"):
            a = self.abl.get(variant)
            if a is None:
                continue
            if "error" in a:
                self.abl_rows.append({**base, "row": variant, "variant": variant, "verdict": "error", "notes": a["error"]})
                continue
            cv = a["case"]
            self.abl_rows.append({**base, "row": variant, "variant": variant, "dwell_s": cv.blocks[0].dwell_s,
                                  **pred_fields(a["pred"]), **tds_fields(self.tds.get(f"S7:{variant}:{d}"), cv.event(d)),
                                  **lp_fields(a["lp"], cv),
                                  **tds_fields(self.tds.get(f"S7mit:{variant}:{d}"), cv.event(d), "mitigated_tds"),
                                  **lib(a["preds"]),
                                  "notes": "relays disabled: no trip possible (screen K reported; would-trip from TDS)" if not cv.relays_enabled else ""})
        fs = self.abl["fast_automatic_shunt"]
        self.abl_rows.append({**base, "row": "fast_automatic_shunt", "variant": "baseline (LP without fast_reactor)",
                              "dwell_s": self.case.blocks[0].dwell_s,
                              **pred_fields(self.preds[d]), **tds_fields(self.tds.get(f"S3:{d}"), ev),
                              **lp_fields(fs["lp_nofast"], self.case),
                              **tds_fields(self.tds.get(f"S7fast:nofast:{d}") or self.tds.get(f"S4:{d}:all_but_fast_reactor"), ev, "mitigated_tds"),
                              **lib(self.preds),
                              "lp_with_fast_feasible": fs["lp_all"].get("feasible"), "lp_with_fast_eta": fs["lp_all"].get("eta"),
                              "lp_with_fast_mvar_total_rated": lp_fields(fs["lp_all"], self.case)["lp_mvar_total_rated"],
                              "notes": "LP with fast_reactor controls removed; lp_with_fast_* = all controls"})
        mt = self.abl["missing_collector_telemetry"]
        self.abl_rows.append({**base, "row": "missing_collector_telemetry", "variant": "baseline (transmission-only regime)",
                              "dwell_s": self.case.blocks[0].dwell_s,
                              **pred_fields(mt["pred"]), **tds_fields(self.tds.get(f"S3:{d}"), ev),
                              **lp_fields(mt["lp"], self.case), **{"mitigated_tds_status": "not_run"},
                              **lib(self.preds),
                              "notes": "robust screen with n in [0.90,1.10], e in [-0.02,0.02]; TDS is the physical baseline"})

    def s8(self) -> None:
        order = [e for e in self.scr.event_order if e in self.preds]
        scores = {e: (self.preds[e].max_k_pk_nonself, self.preds[e].max_k_tr_nonself) for e in order}
        labels = {e: self.summaries["nominal"].get(e, {}).get("dangerous") for e in order}
        ranked = TR.rank_seeds(scores, order)
        curve = TR.capture_curve(ranked, labels)
        pos = {e: k for k, e in enumerate(ranked, start=1)}
        self.triage_rows = []
        for row in curve:
            e = row["seed_id"]
            self.triage_rows.append({
                "system": self.slug, **row,
                "family": self.events[e].family if e else None,
                "score_k_pk": scores[e][0] if e else None, "score_k_tr": scores[e][1] if e else None,
                "library_rank": pos.get(e) if e else None,
            })
        excluded = [e for e in order if labels.get(e) is None]
        self.triage_summary = {
            "system": self.slug, "n_seeds": len(order), "n_conclusive": len(order) - len(excluded),
            "n_dangerous": sum(1 for v in labels.values() if v), "roc_auc": TR.roc_auc(scores, labels),
            "fraction_sent_to_capture_all": TR.fraction_to_capture_all(curve),
            "excluded_inconclusive": excluded, "scores": {e: list(scores[e]) for e in order},
            "labels": labels,
        }

    def sanity_checks(self) -> None:
        s: dict[str, Any] = {}
        saved = self.out / "sanity.json"
        if self.rescore and saved.exists() and read_json(saved).get("a_rerun"):
            s["a_rerun"] = {**read_json(saved)["a_rerun"], "note": "carried over from the original run (rescore)"}
        dang = [] if "a_rerun" in s else [e for e, v in self.summaries["nominal"].items() if v.get("dangerous")]
        rng = np.random.default_rng([C.SANITY_SEED, C.SYSTEMS.index(self.slug) if self.slug in C.SYSTEMS else 99])
        pick = sorted(rng.choice(dang, size=min(3, len(dang)), replace=False).tolist()) if dang else []
        reruns = []
        for e in pick:
            ref = self.tds[f"S3:{e}"]
            for tag, step in (("identical", C.TDS_TSTEP_S), ("finer_step", C.TDS_TSTEP_CHECK_S)):
                r = tds_job(make_job(f"sanity:{tag}:{e}", self.case, e, kind="sanity", fingerprint=self.fp, tstep=step))
                t_ref = {a["asset_id"]: a.get("trip_rel_s") for a in ref.get("assets", [])}
                t_new = {a["asset_id"]: a.get("trip_rel_s") for a in r.get("assets", [])}
                diffs = [abs(t_ref[a] - t_new[a]) for a in t_ref if t_ref[a] is not None and t_new.get(a) is not None]
                same_set_window = set(ref.get("tripped_within_window", [])) == set(r.get("tripped_within_window", []))
                reruns.append({"seed_id": e, "rerun": tag, "tstep": step, "status": r.get("status"),
                               "trip_order_ref": ref.get("trip_order"), "trip_order_rerun": r.get("trip_order"),
                               "same_trip_set_in_window": same_set_window,
                               "identical_trip_order": list(ref.get("trip_order", [])) == list(r.get("trip_order", [])),
                               "max_abs_trip_time_diff_s": max(diffs) if diffs else 0.0,
                               "max_v_diff_pu": max((abs(_fin(a1.get("max_v_window_pu")) - _fin(a2.get("max_v_window_pu")))
                                                     for a1, a2 in zip(ref.get("assets", []), r.get("assets", []))), default=float("nan")),
                               "runtime_s": r.get("runtime_s")})
        s["a_rerun"] = s.get("a_rerun") or {"seeds": pick, "runs": reruns,
                        "all_identical": all(x["identical_trip_order"] and x["max_abs_trip_time_diff_s"] == 0.0 for x in reruns if x["rerun"] == "identical"),
                        "finer_step_same_sets": all(x["same_trip_set_in_window"] for x in reruns if x["rerun"] == "finer_step")}
        noop = []
        dv = {}
        for e in self.scr.event_order:
            r = self.tds.get(f"S3:{e}")
            if r is None:
                continue
            v = _fin(r.get("max_abs_dv_all_buses_pu"))
            dv[e] = v
            if not (v > C.NOOP_DV_PU):
                noop.append(e)
        s["b_no_noop"] = {"threshold_pu": C.NOOP_DV_PU, "n_events": len(dv), "noop_events": noop,
                          "min_max_abs_dv_pu": min(dv.values()) if dv else float("nan"),
                          "min_event": min(dv, key=dv.get) if dv else None, "passed": not noop}
        viol = []
        maps = self.scr.maps
        for j, e in enumerate(maps.event_ids):
            dis = set(self.events[e].disconnects_assets)
            for i, a in enumerate(maps.asset_ids):
                is_self = maps.classes[i, j] == "self"
                if is_self != (a in dis):
                    viol.append(f"maps {a}:{e} self={is_self} disconnects={a in dis}")
                if is_self and (maps.k_pickup[i, j] != 0.0 or maps.k_trip[i, j] != 0.0):
                    viol.append(f"maps {a}:{e} self-term K != 0")
        for e, p in self.preds.items():
            bad = set(p.classes) & set(self.events[e].disconnects_assets)
            if bad:
                viol.append(f"cascade {e} classifies self assets {sorted(bad)}")
        for row in self.val_rows:
            if row["asset_id"] in self.events[row["seed_id"]].disconnects_assets:
                viol.append(f"scoring row for self asset {row['asset_id']} of {row['seed_id']}")
        for e in self.scr.event_order:
            r = self.tds.get(f"S3:{e}")
            if r is None:
                continue
            for a in r.get("assets", []):
                if a["asset_id"] in self.events[e].disconnects_assets:
                    if a.get("active") or a.get("trip_rel_s") is not None and a.get("tripped_by") == "relay":
                        viol.append(f"TDS self asset {a['asset_id']} of {e} active/tripped by relay")
        for (e, restr), rec in getattr(self, "lps", {}).items():
            if not set(self.events[e].disconnects_assets) <= set(rec.get("excluded_assets", []) or []) and rec.get("status") == "solved":
                viol.append(f"LP {e}/{restr} did not exclude the self assets")
        s["c_self_terms"] = {"violations": viol, "passed": not viol,
                             "n_self_entries": int(np.sum(maps.classes == "self"))}
        dmax, dtr, npad_neg, ntol = 0.0, 0.0, 0, 0
        not_cert = []
        for j, e in enumerate(maps.event_ids):
            resp = self.scr.responses.get(e)
            if resp is None:
                not_cert.append(e)
                continue
            for i, a in enumerate(maps.asset_ids):
                if maps.classes[i, j] == "self":
                    continue
                if maps.classes[i, j] != "certified":
                    not_cert.append(f"{a}:{e}")
                    continue
                env = resp.envelopes[self.scr.out_idx[a]]
                mp = env.fine_times_s <= C.T_WINDOW_S + 1e-12
                mi = env.fine_times_s[1:] <= C.T_WINDOW_S + 1e-12
                pts = np.concatenate((env.fine_values[mp], env.fine_left_values[mp]))
                peak = max(float(np.max(pts)), float(np.max(env.fine_upper[mi])))
                k_env = max(peak, 0.0) / maps.margins_pu[i]
                dmax = max(dmax, abs(k_env - maps.k_pickup[i, j]))
                if peak < float(np.max(pts)) - 1e-15:
                    npad_neg += 1
                if not maps.tolerance_met[i, j]:
                    ntol += 1
                if maps.k_trip[i, j] > maps.k_pickup[i, j] + 1e-12:
                    dtr = max(dtr, maps.k_trip[i, j] - maps.k_pickup[i, j])
        s["d_rigorous_K"] = {"bound": (maps.metadata or {}).get("bound"), "max_abs_diff_K_vs_envelope": dmax,
                             "n_entries_padding_negative": npad_neg, "n_entries_padding_above_tolerance": ntol,
                             "max_k_trip_minus_k_pickup": dtr, "non_certified_entries": not_cert,
                             "passed": dmax <= 1e-12 and npad_neg == 0 and not not_cert}
        s["e_mitigation_spot_checks"] = self._mitigation_spot_checks()
        bad_t = []
        n_t = 0
        for e, p in self.preds.items():
            for a, c in p.classes.items():
                t = c.get("predicted_time_s")
                if c.get("predicted_trip") and t is not None:
                    n_t += 1
                    if not (0.0 <= t <= C.T_WINDOW_S + 1e-12):
                        bad_t.append(f"{e}:{a}:{t}")
        no_time = [f"{e}:{a}" for e, p in self.preds.items() for a, c in p.classes.items() if c.get("predicted_trip") and c.get("predicted_time_s") is None]
        s["f_predicted_times_in_window"] = {"n_predicted_trips_with_time": n_t, "outside_window": bad_t,
                                            "predicted_trips_without_time_estimate": no_time, "passed": not bad_t}
        self.sanity = s

    def _mitigation_spot_checks(self) -> dict[str, Any]:
        cands = [e for e in getattr(self, "mit_seeds", []) if f"S4:{e}:all" in self.tds]
        if self.plot_seed in cands:
            cands = [self.plot_seed] + [e for e in cands if e != self.plot_seed]
        chosen = cands[:2]
        out = []
        plot_dir = self.out / "sanity"
        for e in chosen:
            r_m = self.tds[f"S4:{e}:all"]
            r_0 = self.tds.get(f"S3:{e}")
            ev = self.events[e]
            blk = {b.asset_id: b for b in self.case.blocks}
            v_m = np.asarray(r_m.get("trace_v_collector", []), dtype=float)
            ids = r_m.get("trace_asset_ids", [])
            t_m = np.asarray(r_m.get("trace_t", []), dtype=float)
            trip_t = {a["asset_id"]: a.get("trip_s") for a in r_m.get("assets", [])}
            maxv = {}
            for k, a in enumerate(ids):
                if a in ev.disconnects_assets or not v_m.size:
                    continue
                end = C.T_SEED_S + C.T_WINDOW_S if trip_t.get(a) is None else min(C.T_SEED_S + C.T_WINDOW_S, trip_t[a])
                m = (t_m >= C.T_SEED_S - 1e-12) & (t_m <= end + 1e-9)
                maxv[a] = float(np.max(v_m[m, k]))
            over = {a: v for a, v in maxv.items() if v >= blk[a].pickup_pu}
            path = plot_dir / f"mitigation_check_{_safe(e)}.png"
            try:
                _plot_traces(path, e, r_0, r_m, blk, ev.disconnects_assets)
                plotted = str(path.relative_to(C.REPO_ROOT))
            except Exception as exc:
                plotted = f"plot failed: {exc}"
            out.append({"seed_id": e, "tds_status": r_m.get("status"), "controls": r_m.get("metadata", {}).get("controls"),
                        "late_trips_mitigated": [a["asset_id"] for a in r_m.get("assets", []) if a.get("trip_rel_s") is not None
                                                 and a["trip_rel_s"] > C.T_WINDOW_S + SC.WINDOW_TOL_S],
                        "max_v_window_mitigated": maxv, "assets_at_or_above_pickup_mitigated": over,
                        "trips_mitigated": r_m.get("tripped_within_window"),
                        "trips_unmitigated": (r_0 or {}).get("tripped_within_window"), "plot": plotted})
        return {"checks": out}

    def s9(self) -> None:
        ev_rt = self.scr.event_runtime_s
        kinds = {e.event_id: e.kind for e in self.case.events}
        s3 = [self.tds[k] for k in self.tds if k.startswith("S3:")]
        tds_rt = [_fin(r.get("runtime_s")) for r in s3]
        tds_sim = [_fin(r.get("sim_s")) for r in s3]
        tds_build = [_fin(r.get("build_s")) for r in s3]
        casc = self.timing.get("cascade_per_seed_s", [])
        lp = self.timing.get("lp_per_solve_s", [])
        screen_total = (self.timing["engine_baseline_s"] + self.timing["pre_mode_reduction_s"] + sum(ev_rt.values())
                        + self.timing["window_maps_s"] + float(sum(casc)))
        tds_total = float(sum(v for v in tds_rt if math.isfinite(v)))
        pm = self.scr.engine.pre_mode.diagnostics().to_dict()
        self.runtime = {
            "system": self.slug,
            "n_events": len(self.case.events), "n_controls": len(self.case.controls), "n_assets": len(self.case.blocks),
            "build_load_s": self.timing["build_wall_s"], "build_runtime_s_reported_by_benchmarks": self.timing["build_runtime_s_reported"],
            "tds_init_fresh_system_s": self.timing["tds_init_fresh_system_s"],
            "engine_baseline_s": self.timing["engine_baseline_s"],
            "pre_mode_reduction_s": self.timing["pre_mode_reduction_s"],
            "pre_mode_diagnostics": pm,
            "screen_per_event_s_median": _median(list(ev_rt.values())),
            "screen_per_event_s_total": float(sum(ev_rt.values())),
            "screen_per_event_s_median_switch": _median([v for e, v in ev_rt.items() if kinds[e] == "switch"]),
            "screen_per_event_s_median_param": _median([v for e, v in ev_rt.items() if kinds[e] == "param"]),
            "screen_controls_s_total": self.timing["controls_total_s"],
            "screen_per_control_s_median": _median(list(self.scr.control_runtime_s.values())),
            "window_maps_s": self.timing["window_maps_s"],
            "cascade_s_total": float(sum(casc)), "cascade_s_median": _median(casc),
            "lp_s_median": _median(lp), "lp_s_total": float(sum(lp)), "n_lp": len(lp),
            "tds_per_event_s_median": _median(tds_rt), "tds_per_event_s_total": tds_total,
            "tds_per_event_sim_s_median": _median(tds_sim), "tds_per_event_build_s_median": _median(tds_build),
            "tds_pool_elapsed_s": self.timing.get("tds_pool_elapsed_s"), "tds_n_jobs": self.pool.n_jobs,
            "n_tds_runs_total": len(self.tds),
            "screen_all_events_s": screen_total, "tds_all_events_s": tds_total,
            "tds_over_screen_ratio": tds_total / screen_total if screen_total > 0 else float("nan"),
            "notes": ("screen, cascade and LP timed serially in the main process before the TDS pool started; "
                      "TDS per-run wall time measured inside the worker (includes the fresh-system build), "
                      f"{self.pool.n_jobs} workers in parallel on a shared host; screen_all_events_s = engine baseline + "
                      "pre-mode reduction + all event responses + window maps + cascade of all seeds (controls excluded); "
                      "tds_all_events_s = sum of per-run wall times"),
        }

    def figdata(self) -> None:
        self.fig.mkdir(parents=True, exist_ok=True)
        manifest: dict[str, Any] = {}
        maps = self.scr.maps
        rows = []
        for e in self.scr.event_order:
            p = self.preds.get(e)
            s = self.summaries.get("nominal", {}).get(e, {}) if hasattr(self, "summaries") else {}
            rows.append({"event_id": e, "family": self.events[e].family,
                         "max_k_pk_nonself": p.max_k_pk_nonself if p else float("nan"),
                         "max_k_tr_nonself": p.max_k_tr_nonself if p else float("nan"),
                         "argmax_asset": p.argmax_asset if p else None,
                         "tds_dangerous": s.get("dangerous"), "n_tds_trips_in_window": len(s.get("tds_set", []) or []),
                         "n_predicted_trips": len(p.predicted_trips) if p else None})
        rows.sort(key=lambda r: -_nz(r["max_k_pk_nonself"]))
        for k, r in enumerate(rows, start=1):
            r["rank"] = k
        write_csv(self.fig / "fig3_event_kmax_sorted.csv", rows,
                  ["rank", "event_id", "family", "max_k_pk_nonself", "max_k_tr_nonself", "argmax_asset",
                   "tds_dangerous", "n_tds_trips_in_window", "n_predicted_trips"])
        manifest["fig3_event_kmax_sorted.csv"] = "S10: every event, sorted by max non-self K^pk (nominal telemetry screen)"
        d = self.designated
        if d is not None:
            p = self.preds[d]
            s3 = self.tds.get(f"S3:{d}", {})
            t_trip = {a["asset_id"]: a.get("trip_rel_s") for a in s3.get("assets", [])}
            j_of = {e: maps.column(e) for e in maps.event_ids}
            a_idx = {a: i for i, a in enumerate(maps.asset_ids)}
            edges = []
            for a, c in p.classes.items():
                if c["asset_class"] != "pickup_predicted":
                    continue
                srcs = c.get("sources") or [d]
                for src in srcs:
                    j = j_of[src]
                    src_asset = next((b for b, ev in self.scr.amap.items() if ev == src), None)
                    edges.append({"seed_id": d, "source_event": src, "source_asset": src_asset,
                                  "source_layer": 0 if src == d else p.classes.get(src_asset, {}).get("layer"),
                                  "target_asset": a, "target_layer": c["layer"],
                                  "k_pk": float(maps.k_pickup[a_idx[a], j]), "k_tr": float(maps.k_trip[a_idx[a], j]),
                                  "target_predicted_trip": c["predicted_trip"], "target_predicted_time_s": c["predicted_time_s"],
                                  "target_tds_trip_rel_s": t_trip.get(a),
                                  "target_tds_trip_in_window": (t_trip.get(a) is not None and t_trip[a] <= C.T_WINDOW_S + SC.WINDOW_TOL_S)})
            write_csv(self.fig / "fig3_cascade_edges.csv", edges)
            nodes = []
            for a in maps.asset_ids:
                c = p.classes.get(a)
                tr = t_trip.get(a)
                nodes.append({"seed_id": d, "asset_id": a, "is_self": a in self.events[d].disconnects_assets,
                              "asset_class": c["asset_class"] if c else "self", "layer": c["layer"] if c else None,
                              "score": c["score"] if c else None,
                              "predicted_trip": c["predicted_trip"] if c else False,
                              "predicted_time_s": c["predicted_time_s"] if c else None,
                              "k_pk_seed": float(maps.k_pickup[a_idx[a], j_of[d]]),
                              "tds_trip_rel_s": tr, "tds_trip_in_window": tr is not None and tr <= C.T_WINDOW_S + SC.WINDOW_TOL_S,
                              "tds_first_pickup_rel_s": next((x.get("first_pickup_rel_s") for x in s3.get("assets", []) if x["asset_id"] == a), None)})
            write_csv(self.fig / "fig3_cascade_nodes.csv", nodes)
            manifest["fig3_cascade_edges.csv"] = f"S10: designated seed {d}: K edges source event -> pickup asset (non-self)"
            manifest["fig3_cascade_nodes.csv"] = f"S10: designated seed {d}: every asset with screen class/layer/time and TDS trip time"
        fam_rows = []
        for fam in dict.fromkeys(e.family for e in self.case.events):
            evs = [e.event_id for e in self.case.events if e.family == fam]
            ks = [(self.preds[e].max_k_pk_nonself, e) for e in evs if e in self.preds]
            worst = max(ks, key=lambda kv: _nz(kv[0])) if ks else (float("nan"), None)
            npairs = 0
            for e in evs:
                if e not in self.scr.responses:
                    continue
                j = maps.column(e)
                npairs += int(np.sum((maps.classes[:, j] == "certified") & (maps.k_pickup[:, j] >= 1.0)))
            summ = self.summaries.get("nominal", {}) if hasattr(self, "summaries") else {}
            fam_rows.append({"family": fam, "n_events": len(evs), "worst_k_pk_nonself": worst[0], "worst_event": worst[1],
                             "n_events_kmax_ge1": sum(1 for k, _ in ks if _nz(k) >= 1.0), "n_pairs_k_pk_ge1": npairs,
                             "n_tds_dangerous": sum(1 for e in evs if summ.get(e, {}).get("dangerous")),
                             "n_tds_conclusive": sum(1 for e in evs if summ.get(e, {}).get("tds_conclusive"))})
        write_csv(self.fig / "fig5_family_summary.csv", fam_rows)
        manifest["fig5_family_summary.csv"] = "S11: per family worst non-self K^pk, counts >= 1, TDS-dangerous seeds"
        ps = self.plot_seed
        if ps is not None and (ps, "all") in self.lps:
            self._fig6(ps, manifest)
        write_json(self.fig / "fig_data_manifest.json", {"system": self.slug, "files": manifest})

    def _fig6(self, ps: str, manifest: dict[str, Any]) -> None:
        maps = self.scr.maps
        resp = self.scr.responses[ps]
        lp = self.lps[(ps, "all")]
        alphas = lp.get("alphas", {})
        ev = self.events[ps]
        h = {m.asset_id: m.nominal_margin_pu for m in self.scr.models_direct}
        cand = []
        for a in maps.asset_ids:
            if a in ev.disconnects_assets:
                continue
            k = self.scr.out_idx[a]
            cand.append((float(np.max(resp.values_pu[:, k])) / h[a], a))
        _, asset = max(cand)
        k = self.scr.out_idx[asset]
        t = resp.times_s
        seed_v = resp.values_pu[:, k]
        ctrl_v = np.zeros_like(seed_v)
        for cid, a in alphas.items():
            if a > 0:
                ctrl_v += a * self.scr.control_responses[cid].values_pu[:, k]
        det = lp.get("lp_details")
        up_seed = resp.envelopes[k].upper
        resid = np.asarray(det.residual_erosion, dtype=float) if det is not None else None
        a_pos = list(det.protected_asset_ids).index(asset) if det is not None else None
        rows = []
        for l in range(t.size):
            rows.append({"seed_id": ps, "asset_id": asset, "t_s": float(t[l]), "margin_pu": h[asset],
                         "erosion_sampled_no_mitigation": float(seed_v[l] / h[asset]),
                         "erosion_sampled_with_mitigation": float((seed_v[l] + ctrl_v[l]) / h[asset]),
                         "interval_end_s": float(t[l + 1]) if l + 1 < t.size else None,
                         "erosion_upper_no_mitigation": float(max(up_seed[l], 0.0) / h[asset]) if l < up_seed.size else None,
                         "erosion_upper_with_mitigation_lp_residual": float(resid[l, a_pos]) if (resid is not None and l < resid.shape[0]) else None})
        write_csv(self.fig / "fig6_erosion_screen.csv", rows)
        manifest["fig6_erosion_screen.csv"] = (f"S12: seed {ps}, most affected non-seed asset {asset}: normalized erosion "
                                               "dz/h sampled and rigorous per-interval upper bound (LP residual), without/with LP")
        tr_rows = []
        blk = {b.asset_id: b for b in self.case.blocks}
        for run, key in (("no_mitigation", f"S3:{ps}"), ("with_mitigation", f"S4:{ps}:all")):
            r = self.tds.get(key)
            if not r:
                continue
            tt = np.asarray(r.get("trace_t", []), dtype=float)
            vv = np.asarray(r.get("trace_v_collector", []), dtype=float)
            ids = r.get("trace_asset_ids", [])
            for kk, a in enumerate(ids):
                for ti, vi in zip(tt, vv[:, kk]):
                    tr_rows.append({"run": run, "seed_id": ps, "asset_id": a, "is_most_affected": a == asset,
                                    "is_self": a in ev.disconnects_assets, "t_rel_s": float(ti - C.T_SEED_S),
                                    "v_pu": float(vi), "erosion": float((vi - blk[a].z0_pu) / (blk[a].pickup_pu - blk[a].z0_pu)),
                                    "tds_trip_rel_s": next((x.get("trip_rel_s") for x in r.get("assets", []) if x["asset_id"] == a), None)})
        write_csv(self.fig / "fig6_tds_traces.csv", tr_rows)
        manifest["fig6_tds_traces.csv"] = (f"S12: TDS collector voltages for seed {ps} without mitigation (S3 run) and with the "
                                           "all-controls LP actions (S4 verification); t_rel_s relative to the seed; t_rel_s=0 "
                                           "appears twice (pre-event and post-jump solution)")
        env_rows = []
        seen = set()
        for c in self.case.controls:
            if c.control_class in seen:
                continue
            seen.add(c.control_class)
            for ts, vs in c.envelope.samples:
                env_rows.append({"control_class": c.control_class, "control_id": c.control_id, "t_s": ts, "envelope": vs,
                                 "delay_s": c.metadata.get("delay_s"), "tau_s": c.metadata.get("tau_s", 0.0),
                                 "description": c.envelope.description})
        write_csv(self.fig / "fig6_control_envelopes.csv", env_rows)
        manifest["fig6_control_envelopes.csv"] = "S12: ControlSpec.envelope samples (normalized) of one control per class, as used by screen and TDS"
        al_rows = [{"seed_id": ps, "control_id": cid, "control_class": next(c.control_class for c in self.case.controls if c.control_id == cid),
                    "alpha": a} for cid, a in sorted(alphas.items()) if a > C.LP_ALPHA_TOL]
        write_csv(self.fig / "fig6_lp_actions.csv", al_rows, ["seed_id", "control_id", "control_class", "alpha"])
        manifest["fig6_lp_actions.csv"] = "S12: LP (all controls) actions applied in the mitigated TDS"

    def write_outputs(self) -> None:
        o = self.out
        o.mkdir(parents=True, exist_ok=True)
        maps = self.scr.maps
        if set(self.requested) == {"S7"}:
            write_csv(o / "ablations.csv", getattr(self, "abl_rows", []))
            write_json(o / "ablations_meta.json", {
                "system": self.slug, "designated_seed": self.designated,
                "designated_rule": "largest predicted fixed point of the baseline (paper cascade: pickups propagate); "
                                   "ties: largest non-self max K^pk",
                "cascade_definition": "primary (core screen_seed, Eq. cascade_map)",
                "baseline_fingerprint": self.fp,
                "variants": {v: {"fingerprint": case_fingerprint(a["case"]), "design_variant": a["case"].design.get("variant")}
                             for v, a in getattr(self, "abl", {}).items() if isinstance(a, dict) and "case" in a},
                "settings": {"T_window_s": C.T_WINDOW_S, "tds_t_end_s": C.TDS_T_END_S, "epsilon": C.EPSILON},
                "anomalies": self.anomalies,
            })
            self._write_tds_raw()
            return
        if not self.rescore:
            self._write_s1(o, maps)
        self._write_rest(o, maps)

    def _write_s1(self, o: Path, maps: Any) -> None:
        k_rows = []
        for j, e in enumerate(maps.event_ids):
            resp = self.scr.responses.get(e)
            for i, a in enumerate(maps.asset_ids):
                env = resp.envelopes[self.scr.out_idx[a]] if resp is not None else None
                k_rows.append({"system": self.slug, "event_id": e, "family": self.events[e].family, "asset_id": a,
                               "channel_class": maps.classes[i, j], "is_self": maps.classes[i, j] == "self",
                               "margin_pu": maps.margins_pu[i], "k_pk": maps.k_pickup[i, j], "k_tr": maps.k_trip[i, j],
                               "pickup_excursion_pu": maps.pickup_excursion_pu[i, j], "trip_excursion_pu": maps.trip_excursion_pu[i, j],
                               "sampled_peak_pu": maps.sampled_peak_pu[i, j], "padding_pu": maps.padding_pu[i, j],
                               "tolerance_met": bool(maps.tolerance_met[i, j]),
                               "peak_time_s": peak_time(env, C.T_WINDOW_S) if env is not None else None,
                               "jump_pu": float(resp.jump_pu[self.scr.out_idx[a]]) if resp is not None else None,
                               "refined_intervals": env.refined_intervals if env is not None else None})
        write_csv(o / "screen_k.csv", k_rows)
        ev_rows = []
        for e in self.case.events:
            resp = self.scr.responses.get(e.event_id)
            p = self.preds.get(e.event_id)
            j = maps.column(e.event_id)
            ncert = (maps.classes[:, j] == "certified")
            row = {"system": self.slug, "event_id": e.event_id, "family": e.family, "kind": e.kind,
                   "status": "ok" if resp is not None else "failed", "failure_reason": self.scr.failed_events.get(e.event_id),
                   "disconnects_assets": e.disconnects_assets, "n_mutations": len(e.mutations),
                   "max_k_pk_nonself": p.max_k_pk_nonself if p else float("nan"), "argmax_asset": p.argmax_asset if p else None,
                   "max_k_tr_nonself": p.max_k_tr_nonself if p else float("nan"),
                   "n_assets_k_pk_ge1": int(np.sum(ncert & (maps.k_pickup[:, j] >= 1.0))),
                   "n_assets_k_pk_ge_1meps": int(np.sum(ncert & (maps.k_pickup[:, j] >= 1.0 - C.EPSILON))),
                   "runtime_s": self.scr.event_runtime_s.get(e.event_id), "description": e.description}
            if resp is not None:
                tm = dict(resp.timing_s)
                dg = resp.diagnostics.to_dict()
                row.update({"t_jump_newton_s": tm.get("jump_newton"), "t_post_mode_s": tm.get("post_mode"),
                            "t_residual_s": tm.get("residual_and_restore"), "t_propagate_s": tm.get("propagate"),
                            "t_envelopes_s": tm.get("envelopes"), "newton_iterations": resp.newton_iterations,
                            "newton_residual": resp.newton_residual, "mode_id": dg["mode_id"], "n_states": dg["n_states"],
                            "n_algebraic": dg["n_algebraic"], "max_real_eig": dg["max_real_eigenvalue"],
                            "unstable_flag": dg["unstable_flag"], "stability": dg["stability"],
                            "gy_condition": dg["gy_condition_estimate"], "log_norm_mu": dg["log_norm_mu"],
                            "removed_algebraic": dg["removed_algebraic"], "mode_warnings": dg["warnings"]})
            ev_rows.append(row)
        write_csv(o / "screen_events.csv", ev_rows)
        c_rows = []
        for c in self.case.controls:
            r = self.scr.control_responses.get(c.control_id)
            red = [(float(r.envelopes[self.scr.out_idx[a]].sampled_min), a) for a in maps.asset_ids] if r is not None else []
            mn = min(red) if red else (float("nan"), None)
            c_rows.append({"system": self.slug, "control_id": c.control_id, "control_class": c.control_class,
                           "mvar_per_unit": c.mvar_per_unit, "alpha_max": c.alpha_max, "cost": c.cost,
                           "delay_s": c.metadata.get("delay_s"), "tau_s": c.metadata.get("tau_s", 0.0),
                           "envelope": c.envelope.description, "status": "ok" if r is not None else "failed",
                           "failure_reason": self.scr.failed_controls.get(c.control_id),
                           "min_collector_dz_pu": mn[0], "argmin_asset": mn[1],
                           "runtime_s": self.scr.control_runtime_s.get(c.control_id), "description": c.description})
        write_csv(o / "screen_controls.csv", c_rows)

    def _write_rest(self, o: Path, maps: Any) -> None:
        cp_rows, cs_rows = [], []
        for regime, preds in (("nominal", self.preds), ("telemetry_theta", self.preds_telemetry),
                              ("nominal_refined", {e: p.refined() for e, p in self.preds.items()})):
            for e, p in preds.items():
                for a, c in p.classes.items():
                    cp_rows.append({"system": self.slug, "regime": regime, "seed_id": e, "family": self.events[e].family,
                                    "asset_id": a, **{k: c[k] for k in ("asset_class", "score", "layer", "reason", "trip_check",
                                                                        "trip_possible", "trip_excursion_upper_pu",
                                                                        "combined_pickup_excursion_pu", "check_margin_pu",
                                                                        "sources", "predicted_trip", "predicted_time_s")}})
                cnt = Counter(c["asset_class"] for c in p.classes.values())
                cs_rows.append({"system": self.slug, "regime": regime, "seed_id": e, "family": self.events[e].family,
                                "verdict": p.verdict, "max_k_pk_nonself": p.max_k_pk_nonself, "argmax_asset": p.argmax_asset,
                                "max_k_tr_nonself": p.max_k_tr_nonself, "excluded_assets": p.excluded_assets,
                                "fixed_point_size": len(p.fixed_point), "fixed_point": p.fixed_point,
                                "predicted_trip_size": len(p.predicted_trips), "predicted_trips": p.predicted_trips,
                                "refinement_iterations": p.refinement_iterations, "refinement_changed": p.refinement_changed,
                                "refinement_note": p.refinement_note, "n_layers": p.n_layers,
                                "first_secondary": p.first_secondary, "first_secondary_time_s": p.first_secondary_time_s,
                                **{f"n_{k}": cnt.get(k, 0) for k in ("pickup_predicted", "risky", "certified", "data_limited")},
                                "is_designated": e == self.designated, "runtime_s": p.runtime_s})
        write_csv(o / "cascade_predictions.csv", cp_rows)
        write_csv(o / "cascade_seeds.csv", cs_rows)
        if hasattr(self, "val_rows"):
            write_csv(o / "tds_validation.csv", self.val_rows,
                      ["system", "regime", "seed_id", "family", "asset_id", "asset_class", "screen_label", "score", "layer",
                       "predicted_trip", "predicted_time_s", "tds_status", "tds_label", "tds_trip_rel_s",
                       "tds_first_pickup_rel_s", "tds_max_v_window_pu", "tds_max_v_after_pu", "outcome"])
            write_csv(o / "tds_seeds.csv", self.seed_rows)
            write_json(o / "tds_seeds.json", self.seed_rows)
            write_csv(o / "validation_summary.csv", self.val_summary)
        if hasattr(self, "mit_rows"):
            write_csv(o / "mitigation.csv", self.mit_rows)
            write_csv(o / "mitigation_alphas.csv", self.mit_alpha_rows)
            write_csv(o / "mitigation_tds.csv", self.mit_tds_rows)
        if hasattr(self, "obs_rows") and not self.rescore:
            write_csv(o / "observability.csv", self.obs_rows)
            write_csv(o / "observability_seeds.csv", self.obs_seed_rows)
            write_csv(o / "observability_summary.csv", self.obs_summary)
        if hasattr(self, "unc_rows"):
            write_csv(o / "uncertainty.csv", self.unc_rows)
            write_csv(o / "uncertainty_lp.csv", self.unc_lp_rows)
        if hasattr(self, "abl_rows"):
            write_csv(o / "ablations.csv", self.abl_rows)
        if hasattr(self, "triage_rows"):
            write_csv(o / "triage.csv", self.triage_rows)
            write_json(o / "triage_summary.json", self.triage_summary)
        if hasattr(self, "runtime") and not self.rescore:
            write_json(o / "runtime.json", self.runtime)
        write_json(o / "sanity.json", {"system": self.slug, **self.sanity, "anomalies": self.anomalies})
        write_json(o / "screen_meta.json", {
            "system": self.slug, "variant": self.case.variant, "fingerprint": self.fp,
            "settings": {"T_window_s": C.T_WINDOW_S, "dt_s": C.DT_S, "epsilon": C.EPSILON, "t_seed_s": C.T_SEED_S,
                         "tds_t_end_s": C.TDS_T_END_S, "tds_tstep_s": C.TDS_TSTEP_S, "eps_meas_pu": C.EPS_MEAS_PU},
            "assets": [{"asset_id": b.asset_id, "collector_bus": b.collector_bus, "hv_bus": b.hv_bus, "z0_pu": b.z0_pu,
                        "margin_pu": b.pickup_pu - b.z0_pu, "tap": b.tap, "pickup_pu": b.pickup_pu, "dwell_s": b.dwell_s}
                       for b in self.case.blocks],
            "outputs": list(self.scr.output_ids), "pre_mode": self.scr.engine.pre_mode.diagnostics().to_dict(),
            "engine_warnings": list(self.scr.engine.warnings), "designated_seed": self.designated,
            "plot_seed": getattr(self, "plot_seed", None), "mitigation_seeds": getattr(self, "mit_seeds", []),
            "failed_events": self.scr.failed_events, "failed_controls": self.scr.failed_controls,
            "requested_studies": list(self.requested), "studies_run": sorted(self.run_set),
            "scenario": self.case.design.get("scenario"),
        })
        if not self.rescore or len(self.tds) > getattr(self, "_n_saved_runs", 0):
            self._write_tds_raw()

    def _write_tds_raw(self) -> None:
        light = {k: {kk: vv for kk, vv in v.items() if not kk.startswith("trace_")} for k, v in self.tds.items()}
        write_json(self.out / "tds_runs.json", light)
        arrays = {}
        for k, v in self.tds.items():
            if v.get("trace_t"):
                key = _safe(k)
                arrays[f"{key}__t"] = np.asarray(v["trace_t"], dtype=float)
                arrays[f"{key}__v"] = np.asarray(v["trace_v_collector"], dtype=float)
        ids = [b.asset_id for b in self.case.blocks]
        np.savez_compressed(self.out / "tds_traces.npz", asset_ids=np.asarray(ids), keys=np.asarray(list(self.tds)), **arrays)

    def _load_tds_raw(self, all_runs: bool = False) -> dict[str, dict[str, Any]]:
        light = read_json(self.out / "tds_runs.json")
        with np.load(self.out / "tds_traces.npz", allow_pickle=False) as z:
            for k, v in light.items():
                key = _safe(k)
                if f"{key}__t" in z.files:
                    v["trace_t"] = z[f"{key}__t"].tolist()
                    v["trace_v_collector"] = z[f"{key}__v"].tolist()
                    v["trace_asset_ids"] = [str(a) for a in z["asset_ids"]]
        return light if all_runs else {k: v for k, v in light.items() if k.startswith("S3:")}

    def run(self) -> dict[str, Any]:
        t_all = time.perf_counter()
        self.build()
        self.s1_s2()
        self.lps = {}
        self.mit_seeds = []
        self.plot_seed = None
        self.unavail = {}
        if "S4" in self.run_set or "S6" in self.run_set or "S7" in self.run_set:
            self.s4_lps()
        jobs = self.tds_jobs_baseline()
        self.tds: dict[str, dict[str, Any]] = {}
        if self.rescore:
            self.tds.update(self._load_tds_raw(all_runs=True))
            self._n_saved_runs = len(self.tds)
            jobs = [j for j in jobs if j["job_id"] not in self.tds]
            self.log(f"[{self.slug}] rescore: reusing {len(self.tds)} saved TDS runs; {len(jobs)} missing jobs to run")
        elif self.reuse_tds and "S3" in self.run_set:
            self.tds.update(self._load_tds_raw())
            self.log(f"[{self.slug}] reusing {len(self.tds)} S3 TDS runs from tds_runs.json")
        t_pool = time.perf_counter()
        handles = self.pool.submit_all(jobs) if jobs else []
        self.log(f"[{self.slug}] submitted {len(jobs)} TDS jobs")
        if "S5" in self.run_set or "S7" in self.run_set:
            self.s5()
        if "S6" in self.run_set:
            self.s6()
        s7_handles = []
        if "S7" in self.run_set:
            s7_jobs = self.s7_prepare()
            s7_handles = self.pool.submit_all(s7_jobs) if s7_jobs else []
            self.log(f"[{self.slug}] submitted {len(s7_jobs)} S7 TDS jobs")
        self.tds.update(self.pool.gather(list(handles) + list(s7_handles), log=self.log))
        self.timing["tds_pool_elapsed_s"] = time.perf_counter() - t_pool
        errs = {k: v.get("error") for k, v in self.tds.items() if v.get("error")}
        if errs:
            self.anomalies.append(f"TDS job errors: {errs}")
        if "S3" in self.run_set:
            self.s3_score()
        if "S4" in self.run_set:
            self.s4_rows()
        if "S7" in self.run_set:
            self.s7_rows()
        if "S8" in self.run_set:
            self.s8()
        if "S3" in self.run_set:
            self.sanity_checks()
        self.s9()
        if {"S10", "S11", "S12"} & self.run_set:
            self.figdata()
        self.write_outputs()
        self.timing["total_wall_s"] = time.perf_counter() - t_all
        self.log(f"[{self.slug}] done in {self.timing['total_wall_s']:.1f} s")
        return self.summary_payload()

    def update_observability(self) -> dict[str, Any]:
        t_all = time.perf_counter()
        self.build()
        self.s1_s2()
        self.s4_lps()
        self.s5()
        self.s6()
        o = self.out
        write_csv(o / "observability.csv", self.obs_rows)
        write_csv(o / "observability_seeds.csv", self.obs_seed_rows)
        write_csv(o / "observability_summary.csv", self.obs_summary)
        write_csv(o / "uncertainty.csv", self.unc_rows)
        write_csv(o / "uncertainty_lp.csv", self.unc_lp_rows)
        self.log(f"[{self.slug}] S5/S6 update done in {time.perf_counter() - t_all:.1f} s")
        return {"system": self.slug, "observability_summary": self.obs_summary}

    def summary_payload(self) -> dict[str, Any]:
        return {
            "system": self.slug,
            "validation_summary": getattr(self, "val_summary", []),
            "mitigation": getattr(self, "mit_rows", []),
            "mitigation_tds": getattr(self, "mit_tds_rows", []),
            "observability_summary": getattr(self, "obs_summary", []),
            "observability_rows": getattr(self, "obs_rows", []),
            "uncertainty": getattr(self, "unc_rows", []),
            "uncertainty_lp": getattr(self, "unc_lp_rows", []),
            "ablations": getattr(self, "abl_rows", []),
            "triage_summary": getattr(self, "triage_summary", {}),
            "runtime": getattr(self, "runtime", {}),
            "sanity": self.sanity,
            "anomalies": self.anomalies,
            "designated_seed": self.designated,
            "plot_seed": getattr(self, "plot_seed", None),
            "wall_s": self.timing.get("total_wall_s"),
        }


def _read_rows(path: Path) -> list[dict[str, str]]:
    import csv

    if not Path(path).exists():
        return []
    with Path(path).open() as fh:
        return list(csv.DictReader(fh))


def _max_erosion_before_trip(r: Mapping[str, Any], blocks: Sequence[Any], selfs: Sequence[str]) -> tuple[float, str | None]:
    t = np.asarray(r.get("trace_t", []), dtype=float)
    v = np.asarray(r.get("trace_v_collector", []), dtype=float)
    ids = r.get("trace_asset_ids", [])
    if not t.size:
        return float("nan"), None
    blk = {b.asset_id: b for b in blocks}
    trip = {a["asset_id"]: a.get("trip_s") for a in r.get("assets", [])}
    best = (-math.inf, None)
    for k, a in enumerate(ids):
        if a in selfs:
            continue
        end = C.T_SEED_S + C.T_WINDOW_S
        if trip.get(a) is not None:
            end = min(end, float(trip[a]))
        m = (t >= C.T_SEED_S - 1e-12) & (t <= end + 1e-9)
        if not np.any(m):
            continue
        e = (float(np.max(v[m, k])) - blk[a].z0_pu) / (blk[a].pickup_pu - blk[a].z0_pu)
        if e > best[0]:
            best = (e, a)
    return best if best[1] is not None else (float("nan"), None)


def _nz(v: Any) -> float:
    f = _fin(v)
    return f if math.isfinite(f) else -math.inf


def _safe(key: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in key)


def _plot_traces(path: Path, seed: str, r0: Mapping[str, Any] | None, rm: Mapping[str, Any], blocks: Mapping[str, Any],
                 selfs: Sequence[str]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for ax, r, title in ((axes[0], r0, "no mitigation (S3)"), (axes[1], rm, "LP actions applied (S4)")):
        if not r:
            continue
        t = np.asarray(r["trace_t"], dtype=float) - C.T_SEED_S
        v = np.asarray(r["trace_v_collector"], dtype=float)
        for k, a in enumerate(r["trace_asset_ids"]):
            ax.plot(t, v[:, k], lw=1.0, ls="--" if a in selfs else "-", label=a)
        ax.axhline(blocks[r["trace_asset_ids"][0]].pickup_pu, color="k", lw=0.8)
        ax.axvline(C.T_WINDOW_S, color="grey", lw=0.8, ls=":")
        ax.set_xlim(-0.05, C.TDS_T_END_S - C.T_SEED_S)
        ax.set_title(f"{seed}: {title}\ntrips in window: {r.get('tripped_within_window')}", fontsize=8)
        ax.set_xlabel("t - t_s (s)")
    axes[0].set_ylabel("collector V (pu)")
    axes[1].legend(fontsize=6, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
