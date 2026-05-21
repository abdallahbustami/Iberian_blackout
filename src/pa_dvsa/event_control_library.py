"""Event and control channel library for protection-aware screening.

The functions in this module convert ANDES operating-point data and package
schemas into concrete disturbance/control channels. The sign convention is:

* positive ``p_injection_mw`` or ``q_injection_mvar`` means net injection into
  the network,
* a trip or shedding event changes injection by ``after - before``, and
* ``voltage_raising_q_mvar`` is the positive part of the reactive injection
  change, including loss of reactive absorption.

This preserves the underlying signed physics while giving the cascade screen
the voltage-raising component used in the paper.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence

from .andes_adapter import AndesAdapterError, AndesCase
from .data_model import (
    ActionDirection,
    AssessmentStatus,
    AssetRef,
    CandidateControl,
    CandidateEvent,
    ControlKind,
    DataModelError,
    EventKind,
    FixedPowerFactorSpec,
    Interval,
    LocationSpec,
    ProfileKind,
    ResponseEnvelope,
    TimeProfile,
    Unit,
)


class EventControlError(RuntimeError):
    """Raised when an event or control channel cannot be built."""


@dataclass(frozen=True, slots=True)
class AssetOperatingPoint:
    """Signed operating point of one connected or connectable asset."""

    asset: AssetRef
    connected: bool
    status: float | None
    bus_id: str | None
    p_injection_mw: float
    q_injection_mvar: float
    q_absorption_mvar: float
    source_model: str
    source_device_id: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "asset": self.asset.to_dict(),
            "connected": self.connected,
            "status": self.status,
            "bus_id": self.bus_id,
            "p_injection_mw": self.p_injection_mw,
            "q_injection_mvar": self.q_injection_mvar,
            "q_absorption_mvar": self.q_absorption_mvar,
            "source_model": self.source_model,
            "source_device_id": self.source_device_id,
        }


@dataclass(frozen=True, slots=True)
class EventChannel:
    """Computed disturbance channel for one candidate event."""

    event: CandidateEvent
    kind: EventKind
    delta_p_injection_mw: float
    delta_q_injection_mvar: float
    voltage_raising_q_mvar: float
    lost_p_injection_mw: float
    lost_q_injection_mvar: float
    lost_reactive_absorption_mvar: float
    affected_operating_points: tuple[AssetOperatingPoint, ...]
    mode_update_required: bool
    recompute_operating_point: bool
    active_within_horizon: bool
    status: AssessmentStatus
    reason: str
    metadata: Mapping[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "event": self.event.to_dict(),
            "kind": self.kind.value,
            "delta_p_injection_mw": self.delta_p_injection_mw,
            "delta_q_injection_mvar": self.delta_q_injection_mvar,
            "voltage_raising_q_mvar": self.voltage_raising_q_mvar,
            "lost_p_injection_mw": self.lost_p_injection_mw,
            "lost_q_injection_mvar": self.lost_q_injection_mvar,
            "lost_reactive_absorption_mvar": self.lost_reactive_absorption_mvar,
            "affected_operating_points": [
                item.to_dict() for item in self.affected_operating_points
            ],
            "mode_update_required": self.mode_update_required,
            "recompute_operating_point": self.recompute_operating_point,
            "active_within_horizon": self.active_within_horizon,
            "status": self.status.value,
            "reason": self.reason,
            "metadata": dict(self.metadata or {}),
        }


@dataclass(frozen=True, slots=True)
class ControlChannel:
    """Control channel with horizon-aware availability."""

    control: CandidateControl
    available_for_horizon: bool
    delay_s_upper: float
    horizon_s: float
    lower_bound_scale: float
    status: AssessmentStatus
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "control": self.control.to_dict(),
            "available_for_horizon": self.available_for_horizon,
            "delay_s_upper": self.delay_s_upper,
            "horizon_s": self.horizon_s,
            "lower_bound_scale": self.lower_bound_scale,
            "status": self.status.value,
            "reason": self.reason,
        }


def _base_mva(case: AndesCase) -> float:
    return float(case.base_mva or 100.0)


def _values(model: Any, attr: str) -> list[Any]:
    if not hasattr(model, attr):
        return []
    obj = getattr(model, attr)
    value = getattr(obj, "v", [])
    if isinstance(value, list):
        return value
    try:
        return list(value)
    except TypeError:
        return [value]


def _scalar_at(model: Any, attr: str, position: int, default: float = 0.0) -> float:
    values = _values(model, attr)
    if position >= len(values):
        return default
    try:
        return float(values[position])
    except Exception:
        return default


def _text_at(model: Any, attr: str, position: int) -> str | None:
    values = _values(model, attr)
    if position >= len(values):
        return None
    value = values[position]
    return None if value is None else str(value)


def _device_position(case: AndesCase, model_name: str, device_id: str | int | float) -> int:
    model = case.get_model(model_name)
    requested = str(device_id)
    for position, idx in enumerate(_values(model, "idx")):
        if idx == device_id or str(idx) == requested:
            return position
    raise EventControlError(f"{model_name} device {device_id!r} not found")


def _linked_static_generator(case: AndesCase, model_name: str, position: int) -> tuple[str, int] | None:
    model = case.get_model(model_name)
    gen_id = _text_at(model, "gen", position)
    if gen_id is None:
        return None
    for static_model in ("PV", "Slack"):
        try:
            static_position = _device_position(case, static_model, gen_id)
        except Exception:
            continue
        return static_model, static_position
    return None


def _actual_generator_power(case: AndesCase, model_name: str, position: int) -> tuple[float, float]:
    base = _base_mva(case)
    model = case.get_model(model_name)
    p_pu = _scalar_at(model, "p", position, _scalar_at(model, "p0", position))
    q_pu = _scalar_at(model, "q", position, _scalar_at(model, "q0", position))
    return p_pu * base, q_pu * base


def _load_power(case: AndesCase, model_name: str, position: int) -> tuple[float, float]:
    base = _base_mva(case)
    model = case.get_model(model_name)
    p_load = _scalar_at(model, "p0", position) * base
    q_load = _scalar_at(model, "q0", position) * base
    return -p_load, -q_load


def _shunt_power(case: AndesCase, model_name: str, position: int, *, include_status: bool) -> tuple[float, float]:
    base = _base_mva(case)
    model = case.get_model(model_name)
    status = _scalar_at(model, "u", position, 1.0) if include_status else 1.0
    voltage = _scalar_at(model, "v", position, 1.0)
    g = _scalar_at(model, "g", position, 0.0)
    b = _scalar_at(model, "b", position, 0.0)
    return status * g * voltage * voltage * base, status * b * voltage * voltage * base


def asset_operating_point(
    case: AndesCase,
    asset: AssetRef,
    *,
    include_status: bool = True,
) -> AssetOperatingPoint:
    """Return signed P/Q injection and reactive absorption for one asset."""

    if not asset.model_family:
        raise EventControlError(f"Asset {asset.asset_id!r} has no model_family")
    model_name = asset.model_family
    try:
        position = _device_position(case, model_name, asset.asset_id)
    except AndesAdapterError as exc:
        raise EventControlError(str(exc)) from exc

    model = case.get_model(model_name)
    source_model = model_name
    source_position = position
    linked = None
    group = str(getattr(model, "group", "") or "")
    if group == "SynGen":
        linked = _linked_static_generator(case, model_name, position)
        if linked is not None:
            source_model, source_position = linked
            model = case.get_model(source_model)

    source_device_id = str(_values(model, "idx")[source_position])
    status = _scalar_at(model, "u", source_position, 1.0)
    connected = bool(status > 0.5) if include_status else True

    group = str(getattr(model, "group", "") or "")
    if group == "StaticGen" or source_model in {"PV", "Slack"}:
        p_injection, q_injection = _actual_generator_power(case, source_model, source_position)
    elif group == "StaticLoad" or source_model == "PQ":
        p_injection, q_injection = _load_power(case, source_model, source_position)
    elif group == "StaticShunt" or source_model == "Shunt":
        p_injection, q_injection = _shunt_power(
            case, source_model, source_position, include_status=include_status
        )
    else:
        p_injection = _scalar_at(model, "p", source_position, _scalar_at(model, "p0", source_position))
        q_injection = _scalar_at(model, "q", source_position, _scalar_at(model, "q0", source_position))
        p_injection *= _base_mva(case)
        q_injection *= _base_mva(case)

    if not connected:
        p_injection = 0.0
        q_injection = 0.0

    bus_id = _text_at(model, "bus", source_position)
    q_absorption = max(-q_injection, 0.0)
    return AssetOperatingPoint(
        asset=asset,
        connected=connected,
        status=status,
        bus_id=bus_id,
        p_injection_mw=p_injection,
        q_injection_mvar=q_injection,
        q_absorption_mvar=q_absorption,
        source_model=source_model,
        source_device_id=source_device_id,
    )


def _exact_interval(value: float, unit: Unit | str) -> Interval:
    return Interval.exact(float(value), unit)


def build_trip_event(
    case: AndesCase,
    event_id: str,
    name: str,
    assets: Sequence[AssetRef],
    *,
    kind: EventKind = EventKind.PROTECTION_TRIP,
    profile: TimeProfile | None = None,
    source: str | None = None,
) -> EventChannel:
    """Build a trip event that removes connected asset injections."""

    if kind not in {
        EventKind.GENERATOR_TRIP,
        EventKind.COLLECTOR_TRIP,
        EventKind.PROTECTION_TRIP,
        EventKind.DER_BLOCK,
        EventKind.LOAD_PUMP_SHEDDING,
    }:
        raise EventControlError(f"{kind.value} is not a trip-like event kind")
    operating_points = tuple(asset_operating_point(case, asset) for asset in assets)
    connected = tuple(item for item in operating_points if item.connected)
    if not connected:
        raise EventControlError(f"Trip event {event_id!r} has no connected affected assets")

    before_p = sum(item.p_injection_mw for item in connected)
    before_q = sum(item.q_injection_mvar for item in connected)
    delta_p = -before_p
    delta_q = -before_q
    lost_absorption = sum(item.q_absorption_mvar for item in connected)
    voltage_raising = max(delta_q, 0.0)

    event = CandidateEvent(
        event_id=event_id,
        kind=kind,
        name=name,
        affected_assets=tuple(assets),
        profile=profile or TimeProfile(ProfileKind.STEP),
        voltage_raising_q_mvar=_exact_interval(voltage_raising, Unit.MVAR),
        delta_p_mw=_exact_interval(delta_p, Unit.MW),
        delta_q_mvar=_exact_interval(delta_q, Unit.MVAR),
        source=source,
        metadata={
            "before_p_injection_mw": before_p,
            "before_q_injection_mvar": before_q,
            "connected_assets": [item.asset.asset_id for item in connected],
            "skipped_assets": [item.asset.asset_id for item in operating_points if not item.connected],
            "sign_convention": "positive Q is network injection; voltage_raising=max(delta_q,0)",
        },
    )
    return EventChannel(
        event=event,
        kind=kind,
        delta_p_injection_mw=delta_p,
        delta_q_injection_mvar=delta_q,
        voltage_raising_q_mvar=voltage_raising,
        lost_p_injection_mw=max(before_p, 0.0),
        lost_q_injection_mvar=before_q,
        lost_reactive_absorption_mvar=lost_absorption,
        affected_operating_points=operating_points,
        mode_update_required=True,
        recompute_operating_point=True,
        active_within_horizon=True,
        status=AssessmentStatus.CERTIFIED,
        reason="connected_assets_removed",
        metadata={"event_equation": "Delta injection = - pre-event injection"},
    )


def fixed_pf_delta_q_mvar(delta_p_mw: float, fixed_pf: FixedPowerFactorSpec) -> float:
    """Return ``Delta Q = sigma * tan(arccos(|pf|)) * Delta P`` in Mvar."""

    pf = fixed_pf.pf_abs.midpoint
    if not 0 < pf <= 1:
        raise EventControlError("fixed power factor magnitude must lie in (0, 1]")
    kappa = math.tan(math.acos(pf))
    return float(fixed_pf.reactive_sign) * kappa * float(delta_p_mw)


def build_fixed_pf_ramp_event(
    event_id: str,
    name: str,
    delta_p_mw: Interval,
    fixed_pf: FixedPowerFactorSpec,
    *,
    profile: TimeProfile | None = None,
    location: LocationSpec | None = None,
    source: str | None = None,
) -> EventChannel:
    """Build a fixed-PF ramp disturbance using the paper's equation."""

    delta_q_nominal = fixed_pf_delta_q_mvar(delta_p_mw.midpoint, fixed_pf)
    q_candidates = [
        fixed_pf_delta_q_mvar(delta_p, fixed_pf)
        for delta_p in (delta_p_mw.lower, delta_p_mw.upper)
    ]
    q_lower, q_upper = min(q_candidates), max(q_candidates)
    voltage_raising = max(delta_q_nominal, 0.0)

    event = CandidateEvent(
        event_id=event_id,
        kind=EventKind.FIXED_PF_RAMP,
        name=name,
        location=location,
        profile=profile or TimeProfile(ProfileKind.RAMP),
        voltage_raising_q_mvar=Interval(
            max(q_lower, 0.0),
            max(q_upper, 0.0),
            Unit.MVAR,
            nominal=voltage_raising,
        ),
        delta_p_mw=delta_p_mw,
        delta_q_mvar=Interval(q_lower, q_upper, Unit.MVAR, nominal=delta_q_nominal),
        fixed_pf=fixed_pf,
        source=source,
        metadata={
            "equation": "Q = sigma * tan(arccos(|pf|)) * P",
            "kappa_nominal": fixed_pf.nominal_kappa,
            "sign_convention": "positive Delta Q is increased network injection",
        },
    )
    return EventChannel(
        event=event,
        kind=EventKind.FIXED_PF_RAMP,
        delta_p_injection_mw=delta_p_mw.midpoint,
        delta_q_injection_mvar=delta_q_nominal,
        voltage_raising_q_mvar=voltage_raising,
        lost_p_injection_mw=max(-delta_p_mw.midpoint, 0.0),
        lost_q_injection_mvar=-delta_q_nominal,
        lost_reactive_absorption_mvar=max(delta_q_nominal, 0.0)
        if fixed_pf.reactive_sign < 0
        else 0.0,
        affected_operating_points=(),
        mode_update_required=False,
        recompute_operating_point=False,
        active_within_horizon=True,
        status=AssessmentStatus.CERTIFIED,
        reason="fixed_pf_ramp_profile",
        metadata={"fixed_pf_delta_q_mvar": delta_q_nominal},
    )


def build_shunt_switch_event(
    case: AndesCase,
    event_id: str,
    name: str,
    shunt: AssetRef,
    *,
    connect: bool,
    actuation_delay_s: Interval,
    horizon_s: float,
    source: str | None = None,
) -> EventChannel:
    """Build a shunt switch event with horizon-aware mode update metadata."""

    if not shunt.model_family:
        shunt = AssetRef(shunt.asset_id, model_family="Shunt", bus_id=shunt.bus_id)
    op_with_status = asset_operating_point(case, shunt, include_status=True)
    op_nominal = asset_operating_point(case, shunt, include_status=False)
    before_q = op_with_status.q_injection_mvar
    after_q = op_nominal.q_injection_mvar if connect else 0.0
    delta_q = after_q - before_q
    before_p = op_with_status.p_injection_mw
    after_p = op_nominal.p_injection_mw if connect else 0.0
    delta_p = after_p - before_p
    active = actuation_delay_s.lower <= horizon_s
    mode_update = active
    voltage_raising = max(delta_q, 0.0) if active else 0.0

    profile = TimeProfile(ProfileKind.STEP, start_s=actuation_delay_s.midpoint)
    event = CandidateEvent(
        event_id=event_id,
        kind=EventKind.SHUNT_SWITCH,
        name=name,
        affected_assets=(shunt,),
        profile=profile,
        voltage_raising_q_mvar=_exact_interval(voltage_raising, Unit.MVAR),
        delta_p_mw=_exact_interval(delta_p if active else 0.0, Unit.MW),
        delta_q_mvar=_exact_interval(delta_q if active else 0.0, Unit.MVAR),
        source=source,
        metadata={
            "connect": connect,
            "actuation_delay_s": actuation_delay_s.to_dict(),
            "horizon_s": horizon_s,
            "active_within_horizon": active,
            "requires_mode_update": mode_update,
            "representation": "disturbance plus mode/topology update when active in window",
        },
    )
    return EventChannel(
        event=event,
        kind=EventKind.SHUNT_SWITCH,
        delta_p_injection_mw=delta_p if active else 0.0,
        delta_q_injection_mvar=delta_q if active else 0.0,
        voltage_raising_q_mvar=voltage_raising,
        lost_p_injection_mw=max(before_p - after_p, 0.0) if active else 0.0,
        lost_q_injection_mvar=before_q - after_q if active else 0.0,
        lost_reactive_absorption_mvar=max(op_with_status.q_absorption_mvar, 0.0)
        if active and not connect
        else 0.0,
        affected_operating_points=(op_with_status,),
        mode_update_required=mode_update,
        recompute_operating_point=mode_update,
        active_within_horizon=active,
        status=AssessmentStatus.CERTIFIED if active else AssessmentStatus.NOT_EVALUATED,
        reason="shunt_switch_inside_horizon" if active else "shunt_switch_slower_than_horizon",
        metadata={"before_q_injection_mvar": before_q, "after_q_injection_mvar": after_q},
    )


def build_line_topology_event(
    event_id: str,
    name: str,
    line: AssetRef,
    *,
    energize: bool,
    source: str | None = None,
) -> EventChannel:
    """Represent line energization/meshing as a mode update requiring recomputation."""

    event = CandidateEvent(
        event_id=event_id,
        kind=EventKind.LINE_TOPOLOGY_ACTION,
        name=name,
        affected_assets=(line,),
        profile=TimeProfile(ProfileKind.STEP),
        voltage_raising_q_mvar=_exact_interval(0.0, Unit.MVAR),
        delta_p_mw=_exact_interval(0.0, Unit.MW),
        delta_q_mvar=_exact_interval(0.0, Unit.MVAR),
        source=source,
        metadata={
            "energize": energize,
            "primary_representation": "new_mode_recompute_operating_point_and_jacobian",
            "small_signal_topology_sensitivity": "deferred",
        },
    )
    return EventChannel(
        event=event,
        kind=EventKind.LINE_TOPOLOGY_ACTION,
        delta_p_injection_mw=0.0,
        delta_q_injection_mvar=0.0,
        voltage_raising_q_mvar=0.0,
        lost_p_injection_mw=0.0,
        lost_q_injection_mvar=0.0,
        lost_reactive_absorption_mvar=0.0,
        affected_operating_points=(),
        mode_update_required=True,
        recompute_operating_point=True,
        active_within_horizon=True,
        status=AssessmentStatus.CERTIFIED,
        reason="topology_action_requires_new_mode",
        metadata={"line_id": line.asset_id},
    )


def build_load_or_der_disconnection_event(
    case: AndesCase,
    event_id: str,
    name: str,
    assets: Sequence[AssetRef],
    *,
    kind: EventKind = EventKind.LOAD_PUMP_SHEDDING,
    profile: TimeProfile | None = None,
    source: str | None = None,
) -> EventChannel:
    """Build load, pump, or DER disconnection as active/reactive disturbance profiles."""

    if kind not in {EventKind.LOAD_PUMP_SHEDDING, EventKind.DER_BLOCK}:
        raise EventControlError(f"{kind.value} is not a load/DER disconnection kind")
    return build_trip_event(
        case,
        event_id,
        name,
        assets,
        kind=kind,
        profile=profile or TimeProfile(ProfileKind.STEP),
        source=source,
    )


def control_channel_for_horizon(control: CandidateControl, horizon_s: float) -> ControlChannel:
    """Return whether a control channel can act within a relay-relevant horizon."""

    horizon = float(horizon_s)
    if not math.isfinite(horizon) or horizon < 0:
        raise EventControlError("horizon_s must be finite and nonnegative")
    delay_upper = control.response.delay_s.upper
    lower_bound = control.response.lower_bound_scale.lower
    available = bool(control.available and delay_upper <= horizon)
    if not control.available:
        status = AssessmentStatus.NOT_EVALUATED
        reason = "control_marked_unavailable"
    elif available:
        status = AssessmentStatus.CERTIFIED if control.response.certified else AssessmentStatus.EMPIRICAL
        reason = "control_available_within_horizon"
    else:
        status = AssessmentStatus.NOT_EVALUATED
        reason = "control_slower_than_horizon"
    return ControlChannel(
        control=control,
        available_for_horizon=available,
        delay_s_upper=delay_upper,
        horizon_s=horizon,
        lower_bound_scale=lower_bound,
        status=status,
        reason=reason,
    )


def build_reactive_control(
    control_id: str,
    name: str,
    kind: ControlKind,
    direction: ActionDirection,
    *,
    response: ResponseEnvelope,
    alpha_limits: Interval,
    magnitude_mvar: Interval | None = None,
    location: LocationSpec | None = None,
    source: str | None = None,
    cost: float = 1.0,
) -> CandidateControl:
    """Convenience constructor for fast reactive controls."""

    return CandidateControl(
        control_id=control_id,
        kind=kind,
        name=name,
        direction=direction,
        location=location,
        response=response,
        alpha_limits=alpha_limits,
        magnitude_mvar=magnitude_mvar,
        cost=cost,
        source=source,
    )
