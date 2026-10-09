"""Nonlinear time-domain simulation of an event with definite-time overvoltage relays at the plants."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import math
import time
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .data_model import ProfileKind
from .event_response import EventResponseError, solve_frozen_algebraic
from .events import (
    ControlSpec,
    EventSpec,
    Mutation,
    ProtectedBlock,
    apply_mutations,
    profile_value,
    read_base_values,
)


class TDSRunnerError(RuntimeError):
    pass


@dataclass
class AssetTDSRecord:
    asset_id: str
    collector_bus: Any
    pickup_pu: float
    dwell_s: float
    active: bool
    inactive_reason: str = ""
    v0_pu: float = float("nan")
    first_pickup_s: float | None = None
    trip_s: float | None = None
    would_trip_s: float | None = None
    max_v_window_pu: float = -math.inf
    t_max_window_s: float | None = None
    max_v_after_pu: float = -math.inf
    tripped_by: str = ""

    def to_dict(self, t_seed: float | None = None) -> dict[str, Any]:
        def rel(value: float | None) -> float | None:
            return None if value is None or t_seed is None else float(value - t_seed)

        return {
            "asset_id": self.asset_id,
            "collector_bus": str(self.collector_bus),
            "pickup_pu": self.pickup_pu,
            "dwell_s": self.dwell_s,
            "active": self.active,
            "inactive_reason": self.inactive_reason,
            "v0_pu": self.v0_pu,
            "first_pickup_s": self.first_pickup_s,
            "trip_s": self.trip_s,
            "would_trip_s": self.would_trip_s,
            "first_pickup_rel_s": rel(self.first_pickup_s),
            "trip_rel_s": rel(self.trip_s),
            "max_v_window_pu": None if not math.isfinite(self.max_v_window_pu) else self.max_v_window_pu,
            "t_max_window_s": self.t_max_window_s,
            "max_v_after_pu": None if not math.isfinite(self.max_v_after_pu) else self.max_v_after_pu,
            "tripped_by": self.tripped_by,
        }


@dataclass
class TDSRunResult:
    seed_id: str
    converged: bool
    status: str
    err_msg: str
    final_time_s: float
    t_seed_s: float
    window_s: float
    t_end_s: float
    runtime_s: float
    assets: tuple[AssetTDSRecord, ...]
    trip_order: tuple[str, ...]
    pre_seed_max_dv_pu: float
    relays_enabled: bool
    traces: Mapping[str, np.ndarray] | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def asset(self, asset_id: str) -> AssetTDSRecord:
        for rec in self.assets:
            if rec.asset_id == asset_id:
                return rec
        raise KeyError(asset_id)

    def tripped_within_window(self) -> tuple[str, ...]:
        end = self.t_seed_s + self.window_s + 1e-9
        return tuple(a for a in self.trip_order if (self.asset(a).trip_s or math.inf) <= end)

    def to_dict(self) -> dict[str, Any]:
        return {
            "seed_id": self.seed_id,
            "converged": self.converged,
            "status": self.status,
            "err_msg": self.err_msg,
            "final_time_s": self.final_time_s,
            "t_seed_s": self.t_seed_s,
            "window_s": self.window_s,
            "t_end_s": self.t_end_s,
            "runtime_s": self.runtime_s,
            "trip_order": list(self.trip_order),
            "tripped_within_window": list(self.tripped_within_window()),
            "pre_seed_max_dv_pu": self.pre_seed_max_dv_pu,
            "relays_enabled": self.relays_enabled,
            "assets": [rec.to_dict(self.t_seed_s) for rec in self.assets],
            "metadata": dict(self.metadata),
        }


def _position(model: Any, idx: Any) -> int:
    values = list(model.idx.v)
    for cand in (idx, str(idx)):
        if cand in values:
            return values.index(cand)
    as_str = [str(v) for v in values]
    if str(idx) in as_str:
        return as_str.index(str(idx))
    raise TDSRunnerError(f"{model.class_name} has no device {idx!r}")


def _bus_v_address(system: Any, bus: Any) -> int:
    return int(system.Bus.v.a[_position(system.Bus, bus)])


class _Relay:
    def __init__(self, record: AssetTDSRecord, address: int, trip_mutations: tuple[Mutation, ...]):
        self.rec = record
        self.address = address
        self.trip_mutations = trip_mutations
        self.pickup_start: float | None = None
        self.tripped = False


def run_protection_tds(
    system_factory: Callable[[], Any],
    seed: EventSpec | None,
    blocks: Sequence[ProtectedBlock],
    *,
    relays_enabled: bool = True,
    pickup_override: float | Mapping[str, float] | None = None,
    dwell_override: float | Mapping[str, float] | None = None,
    controls: Sequence[tuple[ControlSpec, float]] = (),
    trip_events: Mapping[str, EventSpec] | None = None,
    t_seed: float = 0.5,
    window_s: float = 1.0,
    t_end: float | None = None,
    tstep: float | None = 0.002,
    tol: float | None = None,
    record_traces: bool = False,
    trace_addresses: Mapping[str, int] | None = None,
    steady_tolerance_pu: float = 1e-4,
    newton_max_iter: int = 30,
    callpert_hook: Callable[[Callable[[float, Any], None], float, Any], None] | None = None,
    dropout_ratio: float = 1.0,
) -> TDSRunResult:
    wall0 = time.perf_counter()
    if not (0.5 < float(dropout_ratio) <= 1.0):
        raise TDSRunnerError("dropout_ratio must be in (0.5, 1]")
    if t_seed <= 0.0:
        raise TDSRunnerError("t_seed must be positive (steady pre-run required)")
    t_end_value = float(t_seed + 1.5 * window_s if t_end is None else t_end)
    if t_end_value <= t_seed:
        raise TDSRunnerError("t_end must exceed t_seed")
    system = system_factory()
    tds = system.TDS
    if not getattr(tds, "initialized", False):
        raise TDSRunnerError("system_factory must return a TDS-initialized system")
    if float(system.dae.t) != 0.0:
        raise TDSRunnerError("system_factory must return a fresh system (dae.t == 0)")
    tds.config.no_tqdm = 1
    if tstep is not None:
        tds.config.tstep = float(tstep)
    if tol is not None:
        tds.config.tol = float(tol)

    trip_events = dict(trip_events or {})
    seed_id = seed.event_id if seed is not None else "no_event"
    seed_disconnects = set(seed.disconnects_assets) if seed is not None else set()
    seed_u_targets = set()
    if seed is not None:
        for mut in seed.mutations:
            if mut.attr == "u" and float(mut.value) == 0.0:
                seed_u_targets.add((mut.model, str(mut.device_idx)))

    relays: list[_Relay] = []
    for block in blocks:
        pickup = _override(pickup_override, block.asset_id, block.pickup_pu)
        dwell = _override(dwell_override, block.asset_id, block.dwell_s)
        if block.trip_event_id and block.trip_event_id in trip_events:
            trip_muts = tuple(trip_events[block.trip_event_id].mutations)
        else:
            trip_muts = (Mutation("PVD1", block.pvd1_idx, "u", 0.0),)
        disconnected = block.asset_id in seed_disconnects or any(
            (m.model, str(m.device_idx)) in seed_u_targets for m in trip_muts
        )
        addr = _bus_v_address(system, block.collector_bus)
        rec = AssetTDSRecord(
            asset_id=block.asset_id,
            collector_bus=block.collector_bus,
            pickup_pu=float(pickup),
            dwell_s=float(dwell),
            active=not disconnected,
            inactive_reason="disconnected_by_seed" if disconnected else "",
            v0_pu=float(system.dae.y[addr]),
            tripped_by="seed" if disconnected else "",
        )
        relays.append(_Relay(rec, addr, trip_muts))

    seed_base = read_base_values(system, seed.mutations) if seed is not None else ()
    control_bases = [read_base_values(system, spec.mutations) for spec, _ in controls]
    seed_kind = ProfileKind(seed.profile.kind) if seed is not None else ProfileKind.STEP
    v_init = np.array([float(system.dae.y[r.address]) for r in relays])
    extra = dict(trace_addresses or {})

    trace_t: list[float] = []
    trace_v: list[np.ndarray] = []
    trace_x: list[np.ndarray] = []
    state = {
        "last_acc": -math.inf,
        "phase": 1,
        "pre_max_dv": 0.0,
        "trip_count": 0,
    }
    trip_order: list[str] = []
    window_end = t_seed + window_s

    def record(t_acc: float) -> None:
        y = system.dae.y
        v = np.array([float(y[r.address]) for r in relays])
        if state["phase"] == 1:
            if v.size:
                state["pre_max_dv"] = max(state["pre_max_dv"], float(np.max(np.abs(v - v_init))))
        if record_traces:
            trace_t.append(float(t_acc))
            trace_v.append(v)
            trace_x.append(np.array([float(y[a]) for a in extra.values()]))
        for relay, volt in zip(relays, v):
            rec = relay.rec
            if state["phase"] == 2 and t_acc >= t_seed - 1e-12:
                if t_acc <= window_end + 1e-9:
                    if volt > rec.max_v_window_pu:
                        rec.max_v_window_pu = float(volt)
                        rec.t_max_window_s = float(t_acc)
                else:
                    rec.max_v_after_pu = max(rec.max_v_after_pu, float(volt))
            if not rec.active or relay.tripped:
                continue
            held = relay.pickup_start is not None and volt >= dropout_ratio * rec.pickup_pu
            if volt >= rec.pickup_pu or held:
                if relay.pickup_start is None:
                    relay.pickup_start = float(t_acc)
                if rec.first_pickup_s is None:
                    rec.first_pickup_s = float(t_acc)
                if t_acc - relay.pickup_start >= rec.dwell_s - 1e-9:
                    if rec.would_trip_s is None:
                        rec.would_trip_s = float(t_acc)
                    if relays_enabled:
                        trip(relay, t_acc)
            else:
                relay.pickup_start = None

    def trip(relay: _Relay, t_acc: float) -> None:
        base = read_base_values(system, relay.trip_mutations)
        apply_mutations(system, relay.trip_mutations, base, 1.0)
        relay.tripped = True
        relay.rec.trip_s = float(t_acc)
        relay.rec.tripped_by = "relay"
        trip_order.append(relay.rec.asset_id)
        state["trip_count"] += 1
        targets = {(m.model, str(m.device_idx)) for m in relay.trip_mutations if m.attr == "u"}
        for other in relays:
            if other is relay or other.tripped or not other.rec.active:
                continue
            if any((m.model, str(m.device_idx)) in targets for m in other.trip_mutations):
                other.rec.active = False
                other.rec.inactive_reason = f"disconnected_by_trip_of_{relay.rec.asset_id}"
        solve_frozen_algebraic(system, max_iter=newton_max_iter)
        system.TDS.custom_event = True

    seed_param_keys = (
        {(m.model, str(m.device_idx), m.attr) for m in seed.mutations if m.attr != "u"} if seed is not None else set()
    )
    control_keys = {(m.model, str(m.device_idx), m.attr) for spec, _ in controls for m in spec.mutations}
    shared_keys = seed_param_keys & control_keys

    def apply_profiles(t: float) -> None:
        if t < t_seed - 1e-12:
            return
        rel = t - t_seed
        parts: list[tuple[Sequence[Mutation], Sequence[float], float]] = []
        if seed is not None and seed.kind == "param" and (seed_kind != ProfileKind.STEP or shared_keys):
            parts.append((seed.mutations, seed_base, profile_value(seed.profile, rel)))
        elif seed is not None and seed.kind == "switch" and shared_keys:
            nonu = [(m, b) for m, b in zip(seed.mutations, seed_base) if m.attr != "u"]
            parts.append(([m for m, _ in nonu], [b for _, b in nonu], 1.0))
        for (spec, alpha), base in zip(controls, control_bases):
            parts.append((spec.mutations, base, float(alpha) * profile_value(spec.envelope, rel)))
        _apply_superposed(system, parts)

    def callpert(t: float, _system: Any) -> None:
        t = float(t)
        t_acc = t - float(system.TDS.h)
        if t_acc >= -1e-12 and t_acc > state["last_acc"] + 1e-12:
            state["last_acc"] = t_acc
            record(t_acc)
        if state["phase"] == 2:
            apply_profiles(t)

    if callpert_hook is None:
        tds.callpert = callpert
    else:
        tds.callpert = lambda t, sys_: callpert_hook(callpert, t, sys_)
    err_msg = ""
    status = "ok"
    converged = True
    tds.config.tf = float(t_seed)
    ok1 = bool(tds.run(no_summary=True))
    t_now = float(system.dae.t)
    if not ok1 or abs(t_now - t_seed) > 1e-9:
        converged = False
        status = "failed"
        err_msg = f"pre-run failed at t={t_now:.4f}: {getattr(tds, 'err_msg', '')}"
    else:
        record(t_now)
        state["last_acc"] = t_now
        state["phase"] = 2
        jumped = False
        parts0: list[tuple[Sequence[Mutation], Sequence[float], float]] = []
        if seed is not None:
            v0 = profile_value(seed.profile, 0.0)
            if seed.kind == "switch":
                parts0.append((seed.mutations, seed_base, 1.0))
                jumped = True
            else:
                parts0.append((seed.mutations, seed_base, v0))
                jumped = v0 != 0.0
        for (spec, alpha), base in zip(controls, control_bases):
            e0 = float(alpha) * profile_value(spec.envelope, 0.0)
            parts0.append((spec.mutations, base, e0))
            jumped = jumped or e0 != 0.0
        _apply_superposed(system, parts0)
        try:
            if jumped:
                solve_frozen_algebraic(system, max_iter=newton_max_iter)
                system.TDS.custom_event = True
            record(t_now)
        except EventResponseError as exc:
            converged = False
            status = "failed"
            err_msg = f"post-event algebraic solve failed: {exc}"
        if converged:
            tds.config.tf = float(t_end_value)
            ok2 = bool(tds.run(no_summary=True))
            t_now = float(system.dae.t)
            if t_now > state["last_acc"] + 1e-12:
                state["last_acc"] = t_now
                record(t_now)
            if not ok2 or abs(t_now - t_end_value) > 1e-9:
                converged = False
                status = "early_stop" if t_now > t_seed else "failed"
                err_msg = f"TDS stopped at t={t_now:.4f}: {getattr(tds, 'err_msg', '')}"
    if converged and state["pre_max_dv"] > steady_tolerance_pu:
        err_msg = (err_msg + "; " if err_msg else "") + (
            f"pre-run not steady: max |dV| = {state['pre_max_dv']:.3e} pu"
        )

    traces = None
    if record_traces:
        traces = {
            "t": np.asarray(trace_t, dtype=float),
            "v_collector": np.asarray(trace_v, dtype=float).reshape(len(trace_t), len(relays)),
            "asset_ids": np.asarray([r.rec.asset_id for r in relays], dtype=object),
        }
        if extra:
            traces["extra"] = np.asarray(trace_x, dtype=float).reshape(len(trace_t), len(extra))
            traces["extra_ids"] = np.asarray(list(extra.keys()), dtype=object)
    return TDSRunResult(
        seed_id=seed_id,
        converged=converged,
        status=status,
        err_msg=err_msg,
        final_time_s=float(system.dae.t),
        t_seed_s=float(t_seed),
        window_s=float(window_s),
        t_end_s=float(t_end_value),
        runtime_s=time.perf_counter() - wall0,
        assets=tuple(r.rec for r in relays),
        trip_order=tuple(trip_order),
        pre_seed_max_dv_pu=float(state["pre_max_dv"]),
        relays_enabled=bool(relays_enabled),
        traces=traces,
        metadata={
            "tstep": float(tds.config.tstep),
            "tol": float(tds.config.tol),
            "controls": [(spec.control_id, float(alpha)) for spec, alpha in controls],
        },
    )


def _apply_superposed(system: Any, parts: Sequence[tuple[Sequence[Mutation], Sequence[float], float]]) -> None:
    totals: dict[tuple[str, str, str], list[Any]] = {}
    for mutations, bases, fraction in parts:
        for mut, base in zip(mutations, bases, strict=True):
            if mut.attr == "u":
                apply_mutations(system, (mut,), (base,), fraction)
                continue
            key = (mut.model, str(mut.device_idx), mut.attr)
            if key not in totals:
                totals[key] = [mut, float(base), 0.0]
            elif abs(float(totals[key][1]) - float(base)) > 1e-12 * max(1.0, abs(float(base))):
                raise TDSRunnerError(f"inconsistent base values for {key}")
            totals[key][2] += float(fraction) * (float(mut.value) - float(base))
    for mut, base, delta in totals.values():
        apply_mutations(system, (replace(mut, value=base + delta),), (base,), 1.0)


def _override(value: float | Mapping[str, float] | None, asset_id: str, default: float) -> float:
    if value is None:
        return float(default)
    if isinstance(value, Mapping):
        return float(value.get(asset_id, default))
    return float(value)
