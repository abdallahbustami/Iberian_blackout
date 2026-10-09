"""Pooled tables over the four benchmark systems."""

from __future__ import annotations

import csv
import math
import statistics
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from . import config as C
from . import scoring as SC
from . import triage as TR
from .io import read_json, write_csv, write_json

HEADLINE_COLS = [
    "system", "family", "n_seeds", "n_dangerous", "TP", "FP", "TP_L1", "FP_L1", "TP_L2p", "FP_L2p", "pickup_only_trip", "pickup_only_clear", "caught_risky",
    "risky_clear", "unsafe_FN", "TN", "data_limited_trip", "data_limited_clear", "n_seeds_inconclusive", "tds_inconclusive",
    "late_trip", "jaccard_mean_nonempty", "n_both_empty", "jaccard_mean_all", "n_timing", "timing_error_median_s",
    "timing_abs_error_median_s", "timing_abs_error_max_s", "first_asset_match_rate", "seed_TP", "seed_FP", "seed_FN",
    "seed_TN", "n_seed_unsafe_FN", "n_seeds_certified_no_secondary", "screen_event_s_median", "screen_event_s_total",
    "cascade_s_total", "tds_s_median", "tds_s_total",
]


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open() as fh:
        return list(csv.DictReader(fh))


def _systems_present(root: Path, systems: Sequence[str], subdir: str = "") -> list[str]:
    return [s for s in systems if (root / s / subdir / "tds_seeds.json").exists()
            or (root / s / subdir / "cascade_seeds.csv").exists()]


def _num(v: Any) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def build_summary(root: Path = C.RESULTS_ROOT, systems: Sequence[str] = C.SYSTEMS, subdir: str = "") -> dict[str, Any]:
    out = root / "summary" / subdir if subdir else root / "summary"
    out.mkdir(parents=True, exist_ok=True)
    present = _systems_present(root, systems, subdir)
    sysdir = {s: (root / s / subdir if subdir else root / s) for s in present}
    payload: dict[str, Any] = {"systems": present}
    for regime in ("nominal", "telemetry_theta", "nominal_refined"):
        rows = []
        pooled_seeds: list[dict[str, Any]] = []
        for s in present:
            p = sysdir[s] / "tds_seeds.json"
            if not p.exists():
                continue
            seeds = [x for x in read_json(p) if x["regime"] == regime]
            pooled_seeds.extend(seeds)
            fams = list(dict.fromkeys(x["family"] for x in seeds))
            for fam in fams + ["ALL"]:
                grp = [x for x in seeds if fam == "ALL" or x["family"] == fam]
                agg = SC.aggregate(grp, group={"system": s, "family": fam})
                agg.update(_rt(grp))
                rows.append(agg)
        fams = list(dict.fromkeys(x["family"] for x in pooled_seeds))
        for fam in fams + ["ALL"]:
            grp = [x for x in pooled_seeds if fam == "ALL" or x["family"] == fam]
            if grp:
                agg = SC.aggregate(grp, group={"system": "POOLED", "family": fam})
                agg.update(_rt(grp))
                rows.append(agg)
        name = {"nominal": "headline_validation.csv", "telemetry_theta": "validation_telemetry_theta.csv",
                "nominal_refined": "validation_refined.csv"}[regime]
        write_csv(out / name, rows, HEADLINE_COLS)
        payload[f"validation_{regime}"] = rows
    obs = []
    pooled: dict[str, dict[str, float]] = {}
    for s in present:
        for r in _read_csv(sysdir[s] / "observability_summary.csv"):
            obs.append(r)
            acc = pooled.setdefault(r["regime"], {"n_assets": 0, "n_data_limited": 0, "n_pickup_predicted": 0, "n_risky": 0,
                                                  "n_certified": 0, "n_seeds": 0, "n_seeds_certified_no_secondary": 0})
            for k in acc:
                acc[k] += int(float(r[k]))
    for regime, acc in pooled.items():
        n = acc["n_assets"]
        obs.append({"system": "POOLED", "regime": regime, **acc,
                    **{f"share_{k}": acc[f"n_{k}"] / n if n else float("nan") for k in ("data_limited", "pickup_predicted", "risky", "certified")}})
    write_csv(out / "observability_pooled.csv", obs)
    payload["observability"] = obs
    tri = []
    all_scores: dict[str, tuple[float, float]] = {}
    all_labels: dict[str, Any] = {}
    for s in present:
        p = sysdir[s] / "triage_summary.json"
        if not p.exists():
            continue
        t = read_json(p)
        tri.append({k: t[k] for k in ("system", "n_seeds", "n_conclusive", "n_dangerous", "roc_auc", "fraction_sent_to_capture_all")})
        for e, sc in t["scores"].items():
            all_scores[f"{s}:{e}"] = (_num(sc[0]), _num(sc[1]))
            all_labels[f"{s}:{e}"] = t["labels"].get(e)
    if all_scores:
        order = list(all_scores)
        curve = TR.capture_curve(TR.rank_seeds(all_scores, order), all_labels)
        tri.append({"system": "POOLED", "n_seeds": len(order), "n_conclusive": sum(1 for v in all_labels.values() if v is not None),
                    "n_dangerous": sum(1 for v in all_labels.values() if v), "roc_auc": TR.roc_auc(all_scores, all_labels),
                    "fraction_sent_to_capture_all": TR.fraction_to_capture_all(curve)})
        write_csv(out / "triage_pooled_curve.csv", curve)
    write_csv(out / "triage_pooled.csv", tri)
    payload["triage"] = tri
    rt = []
    for s in present:
        p = sysdir[s] / "runtime.json"
        if p.exists():
            r = read_json(p)
            rt.append({k: v for k, v in r.items() if not isinstance(v, (dict, list))})
    write_csv(out / "runtime_summary.csv", rt)
    payload["runtime"] = rt
    for name in ("mitigation", "mitigation_tds", "ablations", "uncertainty", "uncertainty_lp", "observability"):
        rows = []
        for s in present:
            rows.extend(_read_csv(sysdir[s] / f"{name}.csv"))
        write_csv(out / f"{name}_all.csv", rows)
    san = {}
    for s in present:
        p = sysdir[s] / "sanity.json"
        if p.exists():
            d = read_json(p)
            san[s] = {k: (v.get("passed") if isinstance(v, dict) and "passed" in v else None) for k, v in d.items()
                      if k not in ("system", "anomalies")}
            san[s]["a_rerun_identical"] = (d.get("a_rerun") or {}).get("all_identical")
            san[s]["a_rerun_finer_step_same_sets"] = (d.get("a_rerun") or {}).get("finer_step_same_sets")
            san[s]["anomalies"] = d.get("anomalies", [])
    write_json(out / "sanity_summary.json", san)
    payload["sanity"] = san
    write_json(out / "summary.json", payload)
    return payload


def build_ablation_summary(root: Path = C.RESULTS_ROOT, systems: Sequence[str] = C.SYSTEMS,
                           subdir: str = "ablations") -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for s in systems:
        rows.extend(_read_csv(root / s / subdir / "ablations.csv"))
    if rows:
        write_csv(root / "summary" / subdir / "ablations_all.csv", rows)
    return rows


def _rt(seeds: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    def vals(key: str) -> list[float]:
        return [v for v in (_num(x.get(key)) for x in seeds) if math.isfinite(v)]

    scr, tds, casc = vals("screen_runtime_s"), vals("tds_runtime_s"), vals("cascade_runtime_s")
    return {"screen_event_s_median": statistics.median(scr) if scr else float("nan"), "screen_event_s_total": sum(scr),
            "tds_s_median": statistics.median(tds) if tds else float("nan"), "tds_s_total": sum(tds),
            "cascade_s_total": sum(casc)}


def _fmt(v: Any, nd: int = 3) -> str:
    if v is None or v == "":
        return "-"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, str):
        try:
            f = float(v)
        except ValueError:
            return v
        v = f
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        if math.isnan(v):
            return "n/a"
        if v.is_integer() and abs(v) < 1e6:
            return str(int(v))
        return f"{v:.{nd}f}"
    if isinstance(v, (list, tuple)):
        return ", ".join(str(x) for x in v) if v else "-"
    return str(v)


def md_table(rows: Iterable[Mapping[str, Any]], cols: Sequence[tuple[str, str]], nd: int = 3) -> str:
    rows = list(rows)
    head = "| " + " | ".join(h for _, h in cols) + " |"
    sep = "|" + "|".join("---" for _ in cols) + "|"
    body = ["| " + " | ".join(_fmt(r.get(k), nd) for k, _ in cols) + " |" for r in rows]
    return "\n".join([head, sep, *body])


def _pick(rows: Sequence[Mapping[str, Any]], **eq: Any) -> list[Mapping[str, Any]]:
    return [r for r in rows if all(str(r.get(k)) == str(v) for k, v in eq.items())]


def render_report_tables(root: Path = C.RESULTS_ROOT, systems: Sequence[str] = C.SYSTEMS) -> str:
    out = root / "summary"
    parts: list[str] = []
    head = _read_csv(out / "headline_validation.csv")
    for r in head:
        r["inc"] = r.get("n_seeds_inconclusive")
        r["po"] = f"{r.get('pickup_only_trip')}/{r.get('pickup_only_clear')}"
        r["dl"] = f"{r.get('data_limited_trip')}/{r.get('data_limited_clear')}"
        r["rc"] = r.get("risky_clear")
    parts.append("### T1. Headline validation (S3, nominal telemetry screen vs TDS with relays; paper cascade)\n")
    parts.append("Asset-level counts over (seed, non-self asset) pairs inside the 1 s window; `po` = pickup-only "
                 "(pickup predicted, trip not confirmed) with TDS trip / no trip; `DL` = data-limited trip/clear; "
                 "`inc` = inconclusive TDS seeds; J = mean Jaccard over seeds with a non-empty predicted or TDS set "
                 "(`both empty` seeds counted separately); |dt| = abs. error of the first secondary trip time "
                 "(median / max, s); runtimes are medians per event (screen: serial main process; TDS: worker wall "
                 "time incl. fresh-system build).\n")
    parts.append(md_table(head, [
        ("system", "system"), ("family", "family"), ("n_seeds", "seeds"), ("n_dangerous", "dangerous"),
        ("TP", "TP"), ("FP", "FP"), ("caught_risky", "caught_risky"), ("rc", "risky_clear"), ("po", "po"),
        ("unsafe_FN", "unsafe_FN"), ("TN", "TN"), ("dl", "DL"), ("inc", "inc"), ("late_trip", "late"),
        ("jaccard_mean_nonempty", "J"), ("n_both_empty", "both empty"), ("timing_abs_error_median_s", "|dt| med"),
        ("timing_abs_error_max_s", "|dt| max"), ("screen_event_s_median", "screen s"), ("tds_s_median", "TDS s"),
    ]))
    allrows = [r for r in head if r["family"] == "ALL"]
    parts.append("\n### T2. Seed-level confusion and layer split (S3, nominal)\n")
    parts.append("seed_TP/FP/FN/TN: predicted trip set non-empty vs >= 1 non-seed TDS trip in the window. "
                 "TP_L1/FP_L1: predicted trips of the first cascade layer (driven by the seed alone); "
                 "TP_L2p/FP_L2p: layers >= 2 (pickups propagate; combined waveforms superposed without time shifts). "
                 "seed unsafe FN: every non-self asset certified but TDS trips.\n")
    parts.append(md_table(allrows, [
        ("system", "system"), ("n_seeds", "seeds"), ("seed_TP", "seed_TP"), ("seed_FP", "seed_FP"), ("seed_FN", "seed_FN"),
        ("seed_TN", "seed_TN"), ("TP_L1", "TP_L1"), ("FP_L1", "FP_L1"), ("TP_L2p", "TP_L2p"), ("FP_L2p", "FP_L2p"),
        ("n_seeds_certified_no_secondary", "certified no-secondary"), ("n_seed_unsafe_FN", "seed unsafe FN"),
        ("first_asset_match_rate", "first-asset match"), ("n_timing", "n timing"),
        ("screen_event_s_total", "screen total s"), ("tds_s_total", "TDS total s"),
    ]))
    tel = [r for r in _read_csv(out / "validation_telemetry_theta.csv") if r["family"] == "ALL"]
    for r in tel:
        r["po"] = f"{r.get('pickup_only_trip')}/{r.get('pickup_only_clear')}"
    parts.append("\n### T3. Validation under the telemetry uncertainty set (robust screen, e = +/-0.002 pu)\n")
    parts.append(md_table(tel, [
        ("system", "system"), ("n_seeds", "seeds"), ("n_dangerous", "dangerous"), ("TP", "TP"), ("FP", "FP"),
        ("caught_risky", "caught_risky"), ("risky_clear", "risky_clear"), ("po", "po"), ("unsafe_FN", "unsafe_FN"),
        ("TN", "TN"), ("seed_FN", "seed_FN"), ("n_seed_unsafe_FN", "seed unsafe FN"),
        ("jaccard_mean_nonempty", "J"),
    ]))
    mit = _read_csv(out / "mitigation_all.csv")
    mtds = _read_csv(out / "mitigation_tds_all.csv")
    tds_by = {(r["system"], r["seed_id"], r["restriction"]): r for r in mtds}
    rows = []
    for r in _pick(mit, restriction="all"):
        t = tds_by.get((r["system"], r["seed_id"], "all"), {})
        rows.append({**r, "tds_unmit": t.get("n_unmitigated_trips"), "tds_mit": t.get("n_trips_in_window"),
                     "tds_status": t.get("tds_status"), "ero": t.get("max_normalized_erosion"),
                     "verified": t.get("verified_no_trip"),
                     "late": len([x for x in (t.get("late_trips") or "").split(";") if x])})
    parts.append("\n### T4. Mitigation LP (all controls) and TDS verification (S4)\n")
    parts.append("One row per designated seed (non-empty predicted trip set).  MVAr = alpha x rating; AVR in sum of "
                 "alpha (no MVAr rating).  TDS: non-seed relay trips in the window without / with the LP actions "
                 "applied with their envelopes; `max erosion` = max (V - z0)/h of the mitigated run (each asset up to "
                 "its own trip).\n")
    parts.append(md_table(rows, [
        ("system", "system"), ("seed_id", "seed"), ("feasible", "LP feasible"), ("eta", "eta"),
        ("mvar_statcom", "STATCOM"), ("mvar_plant_q_command", "plant Q"), ("mvar_fast_reactor", "fast reactor"),
        ("mvar_manual_reactor", "manual"), ("avr_alpha_sum", "AVR sum alpha"), ("n_binding", "binding"),
        ("tds_unmit", "TDS trips unmitigated"), ("tds_mit", "TDS trips mitigated"), ("ero", "max erosion"),
        ("verified", "verified no trip in window"), ("late", "late trips (after T, mitigated run)"),
    ]))
    rows = []
    for r in mit:
        if str(r.get("is_plot_seed")).lower() != "true":
            continue
        t = tds_by.get((r["system"], r["seed_id"], r["restriction"]), {})
        rows.append({**r, "tds_mit": t.get("n_trips_in_window", "-"), "tds_unmit": t.get("n_unmitigated_trips", "-")})
    parts.append("\n### T5. Useful vs available MVAr (S4, plot seed per system; LP restricted to one class)\n")
    parts.append("available = rating of the class (controls not switched off by the seed); deliverable = rating x "
                 "envelope at the predicted first secondary trip time (`deadline`) and at T = 1 s; LP = MVAr "
                 "chosen by the restricted LP.  `avr_vref` has no MVAr rating (0 shown; its action is the sum of "
                 "alpha, each alpha = 1 being a -0.03 pu set-point step); GB has no AVR controls (classical machines).\n")
    parts.append(md_table(rows, [
        ("system", "system"), ("seed_id", "seed"), ("restriction", "LP restriction"), ("available_mvar", "available MVAr"),
        ("deadline_s", "deadline s"), ("deliverable_mvar_by_deadline", "deliverable by deadline"),
        ("deliverable_mvar_by_T", "deliverable by T"), ("mvar_total_rated", "LP MVAr"),
        ("avr_alpha_sum", "AVR sum alpha"), ("eta", "eta"), ("feasible", "feasible"), ("tds_mit", "TDS trips mitigated"),
    ]))
    obs = _read_csv(out / "observability_pooled.csv")
    parts.append("\n### T6. Observability regimes (S5)\n")
    parts.append("Asset classes over the non-self event library under each regime's Theta (robust_window_bounds); "
                 "`certified seeds` = seeds whose every non-self asset is certified.  `known_tap_symmetric` is a "
                 "sensitivity regime (symmetric operating-point GSU drop bound), not one of the four regimes.\n")
    parts.append(md_table(obs, [
        ("system", "system"), ("regime", "regime"), ("n_assets", "assets"), ("share_data_limited", "share DL"),
        ("share_pickup_predicted", "share pickup"), ("share_risky", "share risky"), ("share_certified", "share certified"),
        ("h_lower_min_pu", "h_lower min"), ("h_lower_median_pu", "h_lower median"), ("h_lower_max_pu", "h_lower max"),
        ("n_seeds_certified_no_secondary", "certified seeds"), ("n_seeds", "seeds"),
    ]))
    unc = _read_csv(out / "uncertainty_all.csv")
    ulp = _read_csv(out / "uncertainty_lp_all.csv")
    parts.append("\n### T7. Uncertainty sensitivity (S6)\n")
    parts.append("Per class and level l: seeds certified no-secondary / assets data-limited / min h_lower (pu) / "
                 "Kendall tau of the seed ranking vs nominal / top-5 overlap.\n")
    for s in dict.fromkeys(r["system"] for r in unc):
        rows = []
        for cls in dict.fromkeys(r["class"] for r in unc if r["system"] == s):
            row = {"class": cls}
            for r in _pick(unc, system=s, **{"class": cls}):
                row[r["level"]] = (f"{_fmt(r['n_seeds_certified_no_secondary'])} / {_fmt(r['n_assets_data_limited'])} / "
                                   f"{_fmt(r['min_h_lower_pu'], 4)} / {_fmt(r['kendall_tau'], 2)} / {_fmt(r['top5_overlap'], 1)}")
            rows.append(row)
        levels = [str(l) for l in C.UNCERTAINTY_LEVELS]
        parts.append(f"\n**{s}**\n")
        parts.append(md_table(rows, [("class", "class")] + [(l, f"l = {l}") for l in levels]))
        lp_rows = []
        for cls in dict.fromkeys(r["class"] for r in ulp if r["system"] == s):
            seeds = list(dict.fromkeys(r["seed_id"] for r in ulp if r["system"] == s and r["class"] == cls))
            for sd in seeds[:1]:
                row = {"class": cls, "seed": sd}
                for r in _pick(ulp, system=s, seed_id=sd, **{"class": cls}):
                    row[r["level"]] = f"{_fmt(r['mvar_total_rated'], 1)} / {_fmt(r['eta'], 3)}"
                lp_rows.append(row)
        if lp_rows:
            parts.append(f"\nLP of the first S6 mitigation seed (MVAr rated / eta) vs level, {s}:\n")
            parts.append(md_table(lp_rows, [("class", "class"), ("seed", "seed")] + [(l, f"l = {l}") for l in levels]))
    abl_new = out / "ablations" / "ablations_all.csv"
    abl = _read_csv(abl_new if abl_new.exists() else out / "ablations_all.csv")
    for r in abl:
        r["n_would"] = len([a for a in (r.get("tds_would_trip_in_window") or "").split(";") if a])
    parts.append("\n### T8. Ablations (S7; designated seed fixed per system)\n")
    parts.append("`relay condition met` = assets whose collector voltage stayed >= pickup for the dwell inside the window "
                 "(equals the trips when relays are enabled; with relays disabled it shows what protection would have done). "
                 "Lists of assets are in `ablations.csv`.\n")
    parts.append(md_table(abl, [
        ("system", "system"), ("designated_seed", "seed"), ("row", "row"), ("worst_k_pk_nonself", "worst K^pk"),
        ("worst_k_tr_nonself", "worst K^tr"), ("predicted_trip_set_size", "pred. trips"), ("verdict", "verdict"),
        ("tds_trips_in_window_count", "TDS trips"), ("n_would", "relay condition met"),
        ("lp_status", "LP status"), ("lp_mvar_total_rated", "LP MVAr"), ("lp_eta", "LP eta"),
        ("mitigated_tds_trips_in_window_count", "TDS trips with LP"), ("lp_with_fast_eta", "eta with fast"),
    ]))
    tri = _read_csv(out / "triage_pooled.csv")
    parts.append("\n### T9. Triage (S8)\n")
    parts.append(md_table(tri, [
        ("system", "system"), ("n_seeds", "seeds"), ("n_conclusive", "conclusive"), ("n_dangerous", "dangerous"),
        ("roc_auc", "ROC AUC"), ("fraction_sent_to_capture_all", "fraction to TDS to capture all"),
    ]))
    rt = _read_csv(out / "runtime_summary.csv")
    parts.append("\n### T10. Runtime (S9, seconds)\n")
    parts.append(md_table(rt, [
        ("system", "system"), ("n_events", "events"), ("build_load_s", "build/load"),
        ("tds_init_fresh_system_s", "TDS init"), ("pre_mode_reduction_s", "pre-mode red."),
        ("screen_per_event_s_median", "screen/event med"), ("screen_per_event_s_total", "screen events total"),
        ("window_maps_s", "window maps"), ("cascade_s_total", "cascade total"), ("lp_s_median", "LP med"),
        ("tds_per_event_s_median", "TDS/event med"), ("tds_per_event_s_total", "TDS events total"),
        ("screen_all_events_s", "screen all"), ("tds_all_events_s", "TDS all"), ("tds_over_screen_ratio", "TDS/screen"),
    ]))
    san = read_json(out / "sanity_summary.json") if (out / "sanity_summary.json").exists() else {}
    rows = [{"system": s, **{k: v for k, v in d.items() if k != "anomalies"}, "n_anomalies": len(d.get("anomalies", []))}
            for s, d in san.items()]
    parts.append("\n### T11. Sanity checks\n")
    parts.append(md_table(rows, [
        ("system", "system"), ("a_rerun_identical", "(a) identical re-run"),
        ("a_rerun_finer_step_same_sets", "(a) 1 ms step same trip sets"), ("b_no_noop", "(b) no no-op event"),
        ("c_self_terms", "(c) self-terms excluded"), ("d_rigorous_K", "(d) K from rigorous envelope"),
        ("f_predicted_times_in_window", "(f) predicted times in window"), ("n_anomalies", "anomalies"),
    ]))
    refd = [r for r in _read_csv(out / "validation_refined.csv") if r["family"] == "ALL"]
    if refd:
        parts.append("\n### T3b. SECONDARY: confirmed-trip propagation (less conservative, not the paper's method)\n")
        parts.append("Only confirmed trips propagate the cascade (`screen_seed` re-run with the asset-to-trip-event map "
                     "restricted to confirmed trips).  Not conservative against model error: at T = 2 s it has 2 unsafe "
                     "FN (Sec. 8, T12).  Nominal telemetry, T = 1 s.\n")
        parts.append(md_table(refd, [
            ("system", "system"), ("n_seeds", "seeds"), ("n_dangerous", "dangerous"), ("TP", "TP"), ("FP", "FP"),
            ("pickup_only_trip", "po trip"), ("pickup_only_clear", "po clear"), ("unsafe_FN", "unsafe_FN"), ("TN", "TN"),
            ("seed_FP", "seed_FP"), ("seed_FN", "seed_FN"), ("jaccard_mean_nonempty", "J"),
        ]))
    parts.append(render_horizon_tables(root))
    return "\n".join(parts) + "\n"


def write_report_tables(report: Path, root: Path = C.RESULTS_ROOT) -> None:
    begin, end = "<!-- BEGIN AUTO-GENERATED TABLES -->", "<!-- END AUTO-GENERATED TABLES -->"
    text = report.read_text() if report.exists() else f"{begin}\n{end}\n"
    if begin not in text or end not in text:
        text += f"\n{begin}\n{end}\n"
    pre, rest = text.split(begin, 1)
    _, post = rest.split(end, 1)
    report.write_text(pre + begin + "\n" + render_report_tables(root) + end + post)


def render_horizon_tables(root: Path = C.RESULTS_ROOT, subdir: str = "horizon_2s") -> str:
    alt = root / "summary" / subdir
    if not (alt / "headline_validation.csv").exists():
        return ""
    horizon = subdir.replace("horizon_", "").replace("s", " s")
    sets = [("1 s", root / "summary", ""), (horizon, alt, subdir)]
    rows = []
    for label, sdir, sub in sets:
        head = {r["system"]: r for r in _read_csv(sdir / "headline_validation.csv") if r["family"] == "ALL"}
        refd = {r["system"]: r for r in _read_csv(sdir / "validation_refined.csv") if r["family"] == "ALL"}
        rt = {r["system"]: r for r in _read_csv(sdir / "runtime_summary.csv")}
        for s in C.SYSTEMS + ("POOLED",):
            if s not in head:
                continue
            r = dict(head[s])
            r["T"] = label
            r["unsafe_FN_refined"] = refd.get(s, {}).get("unsafe_FN")
            r["FP_refined"] = refd.get(s, {}).get("FP")
            if s != "POOLED":
                mt = [x for x in _read_csv((root / s / sub if sub else root / s) / "mitigation_tds.csv")
                      if x.get("restriction") == "all" and x.get("tds_status") == "ok"]
                feas = [x for x in mt if str(x.get("lp_feasible")).lower() == "true"]
                r["n_designated"] = len(mt)
                r["n_lp_feasible"] = len(feas)
                r["n_feasible_verified"] = sum(1 for x in feas if str(x.get("verified_no_trip")).lower() == "true")
                r["n_feasible_no_trip_to_end"] = sum(1 for x in feas if str(x.get("no_trip_to_tds_end")).lower() == "true")
                r["n_infeasible"] = len(mt) - len(feas)
                r["screen_all_s"] = rt.get(s, {}).get("screen_all_events_s")
                r["tds_all_s"] = rt.get(s, {}).get("tds_all_events_s")
            rows.append(r)
    rows.sort(key=lambda r: ((C.SYSTEMS + ("POOLED",)).index(r["system"]), r["T"] != "1 s"))
    parts = [f"\n### T12. Horizon sensitivity: T = 1 s vs T = {horizon} (S1-S4)\n",
             f"Same systems, libraries, dwell (50 ms) and settings; only the screen horizon, scoring window and LP "
             f"horizon change (TDS to t_s + 1.5 T).  Primary: the paper's cascade (pickups propagate).  "
             f"`uFN (refined)` / `FP (refined)`: the same counts for the SECONDARY confirmed-trip propagation (not "
             f"the paper's method).  Mitigation: "
             f"designated seeds, all-controls LPs feasible, feasible LPs with no relay trip in the window "
             f"(verified) and with no trip until the end of the TDS (t_s + 1.5 T).\n",
             md_table(rows, [
                 ("system", "system"), ("T", "T"), ("n_seeds", "seeds"), ("n_dangerous", "dangerous"), ("TP", "TP"),
                 ("FP", "FP"), ("TP_L1", "TP_L1"), ("FP_L1", "FP_L1"), ("unsafe_FN", "unsafe_FN"),
                 ("unsafe_FN_refined", "uFN (refined)"), ("FP_refined", "FP (refined)"), ("seed_FN", "seed_FN"),
                 ("n_seed_unsafe_FN", "seed unsafe FN"), ("jaccard_mean_nonempty", "J"),
                 ("timing_abs_error_max_s", "|dt| max"), ("n_designated", "designated"), ("n_lp_feasible", "LP feasible"),
                 ("n_feasible_verified", "verified in window"), ("n_feasible_no_trip_to_end", "no trip to TDS end"),
                 ("screen_all_s", "screen all s"), ("tds_all_s", "TDS all s"),
             ])]
    t1 = [x for s in C.SYSTEMS for x in _read_csv(root / s / "mitigation_tds.csv")
          if x.get("restriction") == "all" and str(x.get("verified_no_trip")).lower() == "true"
          and str(x.get("no_trip_to_tds_end")).lower() == "false"]
    rows = []
    for x in t1:
        s, e = x["system"], x["seed_id"]
        seeds2 = {r["seed_id"]: r for r in read_json(root / s / subdir / "tds_seeds.json") if r["regime"] == "nominal"} \
            if (root / s / subdir / "tds_seeds.json").exists() else {}
        m2 = {(r["seed_id"], r["restriction"]): r for r in _read_csv(root / s / subdir / "mitigation.csv")}
        t2 = {(r["seed_id"], r["restriction"]): r for r in _read_csv(root / s / subdir / "mitigation_tds.csv")}
        sd = seeds2.get(e, {})
        lp = m2.get((e, "all"), {})
        tv = t2.get((e, "all"), {})
        rows.append({"system": s, "seed_id": e, "late_T1": len([a for a in (x.get("late_trips") or "").split(";") if a]),
                     "dangerous_T2": sd.get("dangerous"), "pred_T2": len(sd.get("predicted_set") or []),
                     "tds_T2": len(sd.get("tds_set") or []), "lp_feasible_T2": lp.get("feasible", "not designated"),
                     "eta_T2": lp.get("eta"), "mvar_T2": lp.get("mvar_total_rated"), "mvar_T1": None,
                     "trips_mit_T2": tv.get("n_trips_in_window"),
                     "late_mit_T2": len([a for a in (tv.get("late_trips") or "").split(";") if a]) if tv else None,
                     "verified_T2": tv.get("verified_no_trip")})
    m1 = {(r["system"], r["seed_id"]): r for s in C.SYSTEMS for r in _read_csv(root / s / "mitigation.csv") if r["restriction"] == "all"}
    for r in rows:
        r["mvar_T1"] = m1.get((r["system"], r["seed_id"]), {}).get("mvar_total_rated")
    if rows:
        parts += [f"\n### T13. Mitigations verified inside the 1 s window that cascade later: the same seeds at T = {horizon}\n",
                  "`late_T1`: relay trips in (t_s + 1 s, t_s + 1.5 s] of the T = 1 s mitigated run.  At the longer horizon: "
                  "dangerous seed, predicted / TDS trip-set sizes, all-controls LP (feasible, eta, MVAr), and the mitigated "
                  "TDS (trips inside the window and after it).\n",
                  md_table(rows, [
                      ("system", "system"), ("seed_id", "seed"), ("mvar_T1", "LP MVAr T=1"), ("late_T1", "late trips T=1"),
                      ("dangerous_T2", "dangerous"), ("pred_T2", "pred. trips"), ("tds_T2", "TDS trips"),
                      ("lp_feasible_T2", "LP feasible"), ("eta_T2", "eta"), ("mvar_T2", "LP MVAr"),
                      ("trips_mit_T2", "mitigated: trips in window"), ("late_mit_T2", "mitigated: late trips"),
                      ("verified_T2", "verified"),
                  ])]
    return "\n".join(parts) + "\n"
