"""Scoring of the screening against time-domain simulation: plant classes, dangerous events, true and false
positives, missed trips and late trips.
"""

from __future__ import annotations

import math
from statistics import median
from typing import Any, Iterable, Mapping, Sequence

SCREEN_LABELS = ("trip", "pickup_only", "risky", "certified", "data_limited")
TDS_LABELS = ("trip", "no_trip", "late_trip", "inconclusive")
OUTCOMES = (
    "TP",
    "FP",
    "pickup_only_trip",
    "pickup_only_clear",
    "caught_risky",
    "risky_clear",
    "unsafe_FN",
    "TN",
    "data_limited_trip",
    "data_limited_clear",
    "late_trip",
    "tds_inconclusive",
)
SCORED_OUTCOMES = OUTCOMES[:10]
WINDOW_TOL_S = 1e-9


class ScoringError(ValueError):
    pass


def screen_label(asset_class: str, predicted_trip: bool) -> str:
    if asset_class == "pickup_predicted":
        return "trip" if predicted_trip else "pickup_only"
    if asset_class in ("risky", "certified", "data_limited"):
        if predicted_trip:
            raise ScoringError(f"a {asset_class} asset cannot be a predicted trip")
        return asset_class
    raise ScoringError(f"unknown asset class {asset_class!r}")


def tds_label(tds_ok: bool, trip_rel_s: float | None, window_s: float) -> str:
    if not tds_ok:
        return "inconclusive"
    if trip_rel_s is None or (isinstance(trip_rel_s, float) and math.isnan(trip_rel_s)):
        return "no_trip"
    if float(trip_rel_s) <= float(window_s) + WINDOW_TOL_S:
        return "trip"
    return "late_trip"


_OUTCOME_TABLE = {
    ("trip", "trip"): "TP",
    ("trip", "no_trip"): "FP",
    ("pickup_only", "trip"): "pickup_only_trip",
    ("pickup_only", "no_trip"): "pickup_only_clear",
    ("risky", "trip"): "caught_risky",
    ("risky", "no_trip"): "risky_clear",
    ("certified", "trip"): "unsafe_FN",
    ("certified", "no_trip"): "TN",
    ("data_limited", "trip"): "data_limited_trip",
    ("data_limited", "no_trip"): "data_limited_clear",
}


def outcome(screen_lbl: str, tds_lbl: str) -> str:
    if screen_lbl not in SCREEN_LABELS:
        raise ScoringError(f"unknown screen label {screen_lbl!r}")
    if tds_lbl not in TDS_LABELS:
        raise ScoringError(f"unknown TDS label {tds_lbl!r}")
    if tds_lbl == "inconclusive":
        return "tds_inconclusive"
    if tds_lbl == "late_trip":
        return "late_trip"
    return _OUTCOME_TABLE[(screen_lbl, tds_lbl)]


def jaccard(predicted: Iterable[str], actual: Iterable[str]) -> tuple[float, bool]:
    p, a = set(predicted), set(actual)
    if not p and not a:
        return 1.0, True
    return len(p & a) / len(p | a), False


def _first(times: Mapping[str, float | None], order: Mapping[str, int] | None = None) -> tuple[str | None, float | None]:
    if order and times:
        lowest = min(order.get(a, 0) for a in times)
        times = {a: t for a, t in times.items() if order.get(a, 0) == lowest}
    known = [(t, order.get(a, 0) if order else 0, a) for a, t in times.items() if t is not None and math.isfinite(t)]
    if known:
        t, _, a = min(known)
        return a, float(t)
    if times:
        a = min(times, key=lambda k: ((order or {}).get(k, 0), k))
        return a, None
    return None, None


def score_seed(
    *,
    seed_id: str,
    family: str,
    self_assets: Sequence[str],
    screen: Mapping[str, Mapping[str, Any]],
    tds_assets: Mapping[str, Mapping[str, Any]],
    tds_ok: bool,
    window_s: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    selfs = set(self_assets)
    bad = selfs & set(screen)
    if bad:
        raise ScoringError(f"self-terms present in the screen classes of {seed_id}: {sorted(bad)}")
    rows: list[dict[str, Any]] = []
    counts = {k: 0 for k in OUTCOMES}
    by_layer = {"TP_L1": 0, "FP_L1": 0, "TP_L2p": 0, "FP_L2p": 0}
    pred_set: list[str] = []
    tds_set: list[str] = []
    pred_times: dict[str, float | None] = {}
    pred_layer: dict[str, int] = {}
    tds_times: dict[str, float | None] = {}
    for asset in sorted(screen):
        sc = screen[asset]
        slbl = screen_label(str(sc["asset_class"]), bool(sc.get("predicted_trip", False)))
        rec = tds_assets.get(asset, {})
        trip_rel = rec.get("trip_rel_s")
        tlbl = tds_label(tds_ok, trip_rel, window_s)
        out = outcome(slbl, tlbl)
        counts[out] += 1
        if out in ("TP", "FP"):
            by_layer[f"{out}_{'L1' if int(sc.get('layer') or 1) <= 1 else 'L2p'}"] += 1
        if slbl == "trip":
            pred_set.append(asset)
            pt = sc.get("predicted_time_s")
            pred_times[asset] = None if pt is None else float(pt)
            pred_layer[asset] = int(sc.get("layer") or 0)
        if tlbl == "trip":
            tds_set.append(asset)
            tds_times[asset] = float(trip_rel)
        rows.append(
            {
                "seed_id": seed_id,
                "family": family,
                "asset_id": asset,
                "asset_class": sc["asset_class"],
                "screen_label": slbl,
                "score": sc.get("score"),
                "layer": sc.get("layer"),
                "predicted_trip": slbl == "trip",
                "predicted_time_s": sc.get("predicted_time_s"),
                "tds_label": tlbl,
                "tds_trip_rel_s": trip_rel,
                "tds_first_pickup_rel_s": rec.get("first_pickup_rel_s"),
                "tds_max_v_window_pu": rec.get("max_v_window_pu"),
                "tds_max_v_after_pu": rec.get("max_v_after_pu"),
                "outcome": out,
            }
        )
    conclusive = bool(tds_ok)
    if conclusive:
        jac, both_empty = jaccard(pred_set, tds_set)
    else:
        jac, both_empty = float("nan"), False
    p_first, p_time = _first(pred_times, pred_layer)
    t_first, t_time = _first(tds_times)
    timing_err = (
        p_time - t_time if (conclusive and p_time is not None and t_time is not None) else None
    )
    all_certified = bool(screen) and all(
        screen_label(str(v["asset_class"]), bool(v.get("predicted_trip", False))) == "certified" for v in screen.values()
    )
    if not screen:
        all_certified = True
    summary = {
        "seed_id": seed_id,
        "family": family,
        "tds_conclusive": conclusive,
        "n_assets_scored": sum(counts[k] for k in SCORED_OUTCOMES),
        "n_self_excluded": len(selfs),
        "dangerous": (len(tds_set) > 0) if conclusive else None,
        "predicted_dangerous": len(pred_set) > 0,
        "flagged": any(
            screen_label(str(v["asset_class"]), bool(v.get("predicted_trip", False))) != "certified" for v in screen.values()
        ),
        "certified_no_secondary": all_certified,
        "seed_unsafe_FN": bool(conclusive and all_certified and len(tds_set) > 0),
        "predicted_set": sorted(pred_set),
        "tds_set": sorted(tds_set),
        "jaccard": jac,
        "both_empty": both_empty,
        "pred_first_asset": p_first,
        "pred_first_time_s": p_time,
        "tds_first_asset": t_first,
        "tds_first_time_s": t_time,
        "first_asset_match": (p_first == t_first) if (conclusive and p_first and t_first) else None,
        "timing_error_s": timing_err,
        **{f"n_{k}": v for k, v in counts.items()},
        **{f"n_{k}": v for k, v in by_layer.items()},
    }
    return rows, summary


def aggregate(summaries: Sequence[Mapping[str, Any]], *, group: Mapping[str, Any]) -> dict[str, Any]:
    conc = [s for s in summaries if s["tds_conclusive"]]
    out: dict[str, Any] = dict(group)
    out["n_seeds"] = len(summaries)
    out["n_seeds_inconclusive"] = len(summaries) - len(conc)
    out["n_dangerous"] = sum(1 for s in conc if s["dangerous"])
    out["n_predicted_dangerous"] = sum(1 for s in summaries if s["predicted_dangerous"])
    out["seed_TP"] = sum(1 for s in conc if s["dangerous"] and s["predicted_dangerous"])
    out["seed_FP"] = sum(1 for s in conc if (not s["dangerous"]) and s["predicted_dangerous"])
    out["seed_FN"] = sum(1 for s in conc if s["dangerous"] and not s["predicted_dangerous"])
    out["seed_TN"] = sum(1 for s in conc if (not s["dangerous"]) and not s["predicted_dangerous"])
    out["n_seeds_certified_no_secondary"] = sum(1 for s in summaries if s["certified_no_secondary"])
    out["n_seed_unsafe_FN"] = sum(1 for s in conc if s["seed_unsafe_FN"])
    for k in OUTCOMES + ("TP_L1", "FP_L1", "TP_L2p", "FP_L2p"):
        out[k] = sum(int(s.get(f"n_{k}", 0)) for s in summaries)
    jac_all = [s["jaccard"] for s in conc]
    jac_nonempty = [s["jaccard"] for s in conc if not s["both_empty"]]
    out["n_both_empty"] = sum(1 for s in conc if s["both_empty"])
    out["jaccard_mean_nonempty"] = (sum(jac_nonempty) / len(jac_nonempty)) if jac_nonempty else float("nan")
    out["jaccard_mean_all"] = (sum(jac_all) / len(jac_all)) if jac_all else float("nan")
    errs = [float(s["timing_error_s"]) for s in conc if s["timing_error_s"] is not None]
    out["n_timing"] = len(errs)
    out["timing_error_median_s"] = median(errs) if errs else float("nan")
    out["timing_abs_error_median_s"] = median(abs(e) for e in errs) if errs else float("nan")
    out["timing_abs_error_max_s"] = max(abs(e) for e in errs) if errs else float("nan")
    matches = [s["first_asset_match"] for s in conc if s["first_asset_match"] is not None]
    out["first_asset_match_rate"] = (sum(1 for m in matches if m) / len(matches)) if matches else float("nan")
    return out
