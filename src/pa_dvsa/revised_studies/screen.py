"""Screening of a benchmark event library and the predicted cascade of each event."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import time
from typing import Any, Mapping, Sequence

import numpy as np

from ..cascade_certificate import screen_seed
from ..event_response import EventResponse, EventResponseEngine, OutputSelector
from ..finite_window import RigorousWindowMaps, assemble_window_maps
from ..protected_outputs import ProtectedOutputModel, output_models_from_blocks
from . import config as C


@dataclass
class ScreenResult:
    slug: str
    variant: str
    engine: EventResponseEngine
    asset_ids: tuple[str, ...]
    output_ids: tuple[str, ...]
    responses: dict[str, EventResponse]
    failed_events: dict[str, str]
    control_responses: dict[str, EventResponse]
    failed_controls: dict[str, str]
    maps: RigorousWindowMaps
    models_direct: tuple[ProtectedOutputModel, ...]
    models_reconstructed: tuple[ProtectedOutputModel, ...]
    amap: dict[str, str]
    event_order: tuple[str, ...]
    event_runtime_s: dict[str, float]
    control_runtime_s: dict[str, float]
    timing_s: dict[str, float]
    relays_enabled: bool

    @property
    def out_idx(self) -> dict[str, int]:
        return {m.asset_id: m.output_index for m in self.models_direct}

    @property
    def response_list(self) -> list[EventResponse]:
        return [self.responses[e] for e in self.event_order if e in self.responses]

    def nonself_max(self, event_id: str, which: str = "pk") -> tuple[float, str | None]:
        j = self.maps.column(event_id)
        k = self.maps.k_pickup if which == "pk" else self.maps.k_trip
        cls = self.maps.classes[:, j]
        best, arg = -math.inf, None
        for i, a in enumerate(self.asset_ids):
            if cls[i] == "self":
                continue
            if cls[i] != "certified":
                return float("nan"), None
            if k[i, j] > best:
                best, arg = float(k[i, j]), a
        return (best if arg is not None else 0.0), arg


def build_engine(case: Any) -> tuple[EventResponseEngine, float]:
    ss = case.system
    selectors = [OutputSelector.bus_voltage(ss, b.collector_bus) for b in case.blocks]
    selectors += [OutputSelector.bus_voltage(ss, b.hv_bus) for b in case.blocks]
    t0 = time.perf_counter()
    engine = EventResponseEngine(
        ss, selectors, horizon_s=C.T_WINDOW_S, dt_s=C.DT_S, envelope_tolerance_pu=C.ENVELOPE_TOL_PU
    )
    return engine, time.perf_counter() - t0


def run_screen(
    case: Any,
    *,
    events: Sequence[Any] | None = None,
    controls: Sequence[Any] | None = None,
    log: Any = None,
) -> ScreenResult:
    engine, t_engine = build_engine(case)
    t0 = time.perf_counter()
    _ = engine.pre_mode
    t_pre = time.perf_counter() - t0
    evs = list(case.events if events is None else events)
    ctrls = list(case.controls if controls is None else controls)
    responses: dict[str, EventResponse] = {}
    failed: dict[str, str] = {}
    ev_rt: dict[str, float] = {}
    for ev in evs:
        t = time.perf_counter()
        try:
            responses[ev.event_id] = engine.event_response(ev)
        except Exception as exc:
            failed[ev.event_id] = f"{type(exc).__name__}: {exc}"
        ev_rt[ev.event_id] = time.perf_counter() - t
        if log:
            log(f"  screen {ev.event_id}: {ev_rt[ev.event_id]:.3f} s" + (" FAILED" if ev.event_id in failed else ""))
    cresp: dict[str, EventResponse] = {}
    cfailed: dict[str, str] = {}
    c_rt: dict[str, float] = {}
    for c in ctrls:
        t = time.perf_counter()
        try:
            cresp[c.control_id] = engine.control_response(c)
        except Exception as exc:
            cfailed[c.control_id] = f"{type(exc).__name__}: {exc}"
        c_rt[c.control_id] = time.perf_counter() - t
    blocks = case.blocks
    assets = tuple(b.asset_id for b in blocks)
    direct = output_models_from_blocks(blocks, engine.output_ids, engine.z0, kind="direct")
    recon = output_models_from_blocks(blocks, engine.output_ids, engine.z0, kind="reconstructed")
    for b, m in zip(blocks, direct):
        if abs(m.z0_pu - float(b.z0_pu)) > 1e-8:
            raise RuntimeError(f"engine z0 of {b.asset_id} ({m.z0_pu}) differs from the design z0 ({b.z0_pu})")
    order = tuple(ev.event_id for ev in evs)
    t = time.perf_counter()
    maps = assemble_window_maps(
        [responses[e] for e in order if e in responses],
        asset_ids=assets,
        output_index={m.asset_id: m.output_index for m in direct},
        margins_pu=[m.nominal_margin_pu for m in direct],
        dwell_s=[float(b.dwell_s) for b in blocks],
        horizon_s=C.T_WINDOW_S,
        failed_events=failed,
        event_order=order,
    )
    t_maps = time.perf_counter() - t
    amap = {b.asset_id: b.trip_event_id for b in blocks}
    missing = [e for e in amap.values() if e not in responses]
    if missing:
        raise RuntimeError(f"trip events without a response (cascade impossible): {missing}")
    return ScreenResult(
        slug=case.slug,
        variant=case.variant,
        engine=engine,
        asset_ids=assets,
        output_ids=tuple(engine.output_ids),
        responses=responses,
        failed_events=failed,
        control_responses=cresp,
        failed_controls=cfailed,
        maps=maps,
        models_direct=tuple(direct),
        models_reconstructed=tuple(recon),
        amap=amap,
        event_order=order,
        event_runtime_s=ev_rt,
        control_runtime_s=c_rt,
        timing_s={
            "engine_baseline_s": t_engine,
            "pre_mode_reduction_s": t_pre,
            "events_total_s": float(sum(ev_rt.values())),
            "controls_total_s": float(sum(c_rt.values())),
            "window_maps_s": t_maps,
        },
        relays_enabled=bool(case.relays_enabled),
    )


@dataclass
class SeedPrediction:
    seed_id: str
    excluded_assets: tuple[str, ...]
    classes: dict[str, dict[str, Any]]
    fixed_point: tuple[str, ...]
    predicted_trips: tuple[str, ...]
    first_secondary: str | None
    first_secondary_time_s: float | None
    max_k_pk_nonself: float
    max_k_tr_nonself: float
    argmax_asset: str | None
    relays_enabled: bool
    runtime_s: float = 0.0
    trip_checks: dict[str, dict[str, Any]] = field(default_factory=dict)
    refined_classes: dict[str, dict[str, Any]] = field(default_factory=dict)
    refined_fixed_point: tuple[str, ...] = ()
    refined_predicted_trips: tuple[str, ...] = ()
    refined_first_secondary: str | None = None
    refined_first_secondary_time_s: float | None = None
    refinement_iterations: int = 0
    refinement_changed: bool = False
    refinement_note: str = ""

    def refined(self) -> "SeedPrediction":
        from dataclasses import replace as _replace

        return _replace(self, classes=self.refined_classes, fixed_point=self.refined_fixed_point,
                        predicted_trips=self.refined_predicted_trips,
                        first_secondary=self.refined_first_secondary,
                        first_secondary_time_s=self.refined_first_secondary_time_s)

    @property
    def n_layers(self) -> int:
        layers = [c["layer"] for c in self.classes.values() if c.get("layer") is not None]
        return max(layers) if layers else 0

    @property
    def certified_no_secondary(self) -> bool:
        return all(c["asset_class"] == "certified" for c in self.classes.values())

    @property
    def verdict(self) -> str:
        cls = {c["asset_class"] for c in self.classes.values()}
        if self.predicted_trips:
            return "trip_predicted"
        if "pickup_predicted" in cls:
            return "pickup_predicted"
        if "risky" in cls:
            return "risky"
        if "data_limited" in cls:
            return "data_limited"
        return "certified_safe"


def _matrix_of(maps: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if hasattr(maps, "k_bar_pk"):
        return maps.k_bar_pk, maps.k_bar_tr, maps.classes
    return maps.k_pickup, maps.k_trip, maps.classes


def _column_max(maps: Any, event_id: str) -> tuple[float, float, str | None]:
    kpk, ktr, cls = _matrix_of(maps)
    j = list(maps.event_ids).index(event_id)
    best_pk, best_tr, arg = -math.inf, -math.inf, None
    for i, a in enumerate(maps.asset_ids):
        if cls[i, j] != "certified":
            continue
        v = float(kpk[i, j])
        if math.isfinite(v) and v > best_pk:
            best_pk, arg = v, a
        w = float(ktr[i, j])
        if math.isfinite(w) and w > best_tr:
            best_tr = w
    if arg is None:
        return float("nan"), float("nan"), None
    return best_pk, best_tr, arg


def _summarize(scr: Any) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    checks = {c.asset_id: c.to_dict() for c in scr.trip_checks}
    classes: dict[str, dict[str, Any]] = {}
    for c in scr.classification.classes:
        chk = checks.get(c.asset_id)
        confirmed = bool(c.asset_class == "pickup_predicted" and chk is not None and chk["trip_possible"])
        classes[c.asset_id] = {
            "asset_class": c.asset_class,
            "score": c.score,
            "layer": c.layer,
            "reason": c.reason,
            "trip_check": chk is not None,
            "trip_possible": None if chk is None else bool(chk["trip_possible"]),
            "trip_excursion_upper_pu": None if chk is None else chk["trip_excursion_upper_pu"],
            "combined_pickup_excursion_pu": None if chk is None else chk["pickup_excursion_pu"],
            "check_margin_pu": None if chk is None else chk["margin_pu"],
            "sources": None if chk is None else list(chk["source_event_ids"]),
            "predicted_trip": confirmed,
            "predicted_time_s": (chk["trip_time_estimate_s"] if confirmed else None),
        }
    return classes, checks


def predict_seed(
    maps: Any,
    seed_id: str,
    *,
    responses: Sequence[Any],
    amap: Mapping[str, str],
    out_idx: Mapping[str, int] | None = None,
    epsilon: float = C.EPSILON,
    relays_enabled: bool = True,
    max_refinements: int = 50,
) -> SeedPrediction:
    t0 = time.perf_counter()
    kwargs = dict(epsilon=epsilon, responses=responses)
    if out_idx is not None and not hasattr(maps, "k_bar_pk"):
        kwargs["output_index"] = dict(out_idx)
    raw = screen_seed(maps, seed_id, asset_event_map=dict(amap), **kwargs)
    raw_classes, _ = _summarize(raw)
    used = dict(amap)
    scr = raw
    classes = raw_classes
    it = 0
    note = ""
    removed: set[str] = set()
    while True:
        pickups = [a for a, c in classes.items() if c["asset_class"] == "pickup_predicted"]
        unconfirmed_sources = [a for a in pickups if not classes[a]["predicted_trip"] and a in used]
        readd = [a for a in removed if classes.get(a, {}).get("predicted_trip") and a not in used]
        if not unconfirmed_sources and not readd:
            break
        if it >= max_refinements:
            note = "refinement_not_converged"
            break
        it += 1
        for a in unconfirmed_sources:
            used.pop(a, None)
            removed.add(a)
        for a in readd:
            used[a] = amap[a]
        scr = screen_seed(maps, seed_id, asset_event_map=used, **kwargs)
        classes, _ = _summarize(scr)
    excluded = tuple(raw.classification.excluded_assets)
    if not relays_enabled:
        for cl in (classes, raw_classes):
            for c in cl.values():
                c["predicted_trip"] = False
                c["predicted_time_s"] = None
        note = (note + ";" if note else "") + "relays_disabled_no_trip_possible"
    trips = tuple(a for a, c in classes.items() if c["predicted_trip"])
    raw_trips = tuple(a for a, c in raw_classes.items() if c["predicted_trip"])
    first, first_t = _first_secondary(raw_classes, raw_trips)
    r_first, r_first_t = _first_secondary(classes, trips)
    kpk, ktr, arg = _column_max(maps, seed_id)
    return SeedPrediction(
        seed_id=seed_id,
        excluded_assets=excluded,
        classes=raw_classes,
        fixed_point=tuple(raw.classification.fixed_point),
        predicted_trips=raw_trips,
        first_secondary=first,
        first_secondary_time_s=first_t,
        max_k_pk_nonself=kpk,
        max_k_tr_nonself=ktr,
        argmax_asset=arg,
        relays_enabled=relays_enabled,
        runtime_s=time.perf_counter() - t0,
        trip_checks={c.asset_id: c.to_dict() for c in raw.trip_checks},
        refined_classes=classes,
        refined_fixed_point=tuple(scr.classification.fixed_point),
        refined_predicted_trips=trips,
        refined_first_secondary=r_first,
        refined_first_secondary_time_s=r_first_t,
        refinement_iterations=it,
        refinement_changed=set(trips) != set(raw_trips),
        refinement_note=note,
    )


def _first_secondary(classes: Mapping[str, Mapping[str, Any]], trips: Sequence[str]) -> tuple[str | None, float | None]:
    if not trips:
        return None, None
    lowest = min(classes[a]["layer"] or 0 for a in trips)
    cand = [a for a in trips if (classes[a]["layer"] or 0) == lowest]
    timed = [(classes[a]["predicted_time_s"], a) for a in cand if classes[a]["predicted_time_s"] is not None]
    if timed:
        t, a = min(timed)
        return a, t
    return min(cand), None


def predict_all(scr: ScreenResult, maps: Any | None = None, *, seeds: Sequence[str] | None = None) -> dict[str, SeedPrediction]:
    maps = scr.maps if maps is None else maps
    out: dict[str, SeedPrediction] = {}
    resp = scr.response_list
    for e in (seeds or scr.event_order):
        if e not in scr.responses:
            continue
        out[e] = predict_seed(
            maps, e, responses=resp, amap=scr.amap, out_idx=scr.out_idx, relays_enabled=scr.relays_enabled
        )
    return out


def peak_time(env: Any, horizon_s: float) -> float:
    t = env.fine_times_s
    m = t <= horizon_s + 1e-12
    vals = np.maximum(env.fine_values[m], env.fine_left_values[m])
    return float(t[m][int(np.argmax(vals))])


def designated_seed(preds: Mapping[str, SeedPrediction]) -> str | None:
    best = None
    for order, (sid, p) in enumerate(preds.items()):
        k = p.max_k_pk_nonself if math.isfinite(p.max_k_pk_nonself) else -math.inf
        key = (len(p.fixed_point), k, -order)
        if best is None or key > best[0]:
            best = (key, sid)
    return None if best is None else best[1]
