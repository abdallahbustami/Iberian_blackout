"""Ranking of the events for simulation: capture curve and ROC AUC."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence


def rank_seeds(scores: Mapping[str, tuple[float, float]], order: Sequence[str]) -> list[str]:
    pos = {e: k for k, e in enumerate(order)}

    def key(e: str) -> tuple[float, float, int]:
        pk, tr = scores[e]
        pk = pk if math.isfinite(pk) else -math.inf
        tr = tr if math.isfinite(tr) else -math.inf
        return (-pk, -tr, pos[e])

    return sorted(order, key=key)


def capture_curve(ranked: Sequence[str], dangerous: Mapping[str, bool | None]) -> list[dict[str, Any]]:
    lab = [(e, dangerous[e]) for e in ranked if dangerous.get(e) is not None]
    n = len(lab)
    nd = sum(1 for _, d in lab if d)
    rows = [{"rank": 0, "seed_id": None, "dangerous": None, "fraction_sent": 0.0,
             "fraction_captured": 0.0 if nd else float("nan")}]
    cap = 0
    for k, (e, d) in enumerate(lab, start=1):
        cap += 1 if d else 0
        rows.append({"rank": k, "seed_id": e, "dangerous": bool(d), "fraction_sent": k / n,
                     "fraction_captured": (cap / nd) if nd else float("nan")})
    return rows


def roc_auc(scores: Mapping[str, tuple[float, float]], dangerous: Mapping[str, bool | None]) -> float:
    def val(e: str) -> tuple[float, float]:
        pk, tr = scores[e]
        return (pk if math.isfinite(pk) else -math.inf, tr if math.isfinite(tr) else -math.inf)

    pos = [val(e) for e, d in dangerous.items() if d is True]
    neg = [val(e) for e, d in dangerous.items() if d is False]
    if not pos or not neg:
        return float("nan")
    tot = 0.0
    for p in pos:
        for q in neg:
            tot += 1.0 if p > q else (0.5 if p == q else 0.0)
    return tot / (len(pos) * len(neg))


def fraction_to_capture_all(curve: Sequence[Mapping[str, Any]]) -> float:
    for row in curve:
        fc = row["fraction_captured"]
        if isinstance(fc, float) and math.isfinite(fc) and fc >= 1.0 - 1e-12:
            return float(row["fraction_sent"])
    return float("nan")
