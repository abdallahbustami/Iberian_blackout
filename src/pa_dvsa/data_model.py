"""Core data model for protection-aware dynamic voltage security assessment.

The classes in this module are intentionally independent of ANDES.  They
describe the paper-level objects: protected relay-side voltages, candidate
events, candidate controls, discrete modes, uncertainty envelopes, and result
containers used by both the sparse screen and nonlinear validation workflow.
"""

from __future__ import annotations

from dataclasses import fields, is_dataclass, dataclass
from enum import Enum
import math
from types import MappingProxyType
from typing import Any, Mapping, Sequence


class DataModelError(ValueError):
    """Raised when a data-model object violates the schema."""


class _StrEnum(str, Enum):
    """String enum base compatible with Python 3.10."""

    def __str__(self) -> str:
        return self.value


class Unit(_StrEnum):
    NONE = ""
    PU = "pu"
    SECOND = "s"
    MW = "MW"
    MVAR = "Mvar"
    KV = "kV"


class MeasurementSide(_StrEnum):
    TRANSMISSION_BUS = "transmission_bus"
    COLLECTOR_BUS = "collector_bus"
    TRANSFORMER_LOW_SIDE = "transformer_low_side"
    INTERNAL_PLANT_BUS = "internal_plant_bus"
    DISTRIBUTION_INTERFACE = "distribution_interface"
    RECONSTRUCTED = "reconstructed"


class ProtectedVoltageKind(_StrEnum):
    DIRECT_BUS_VOLTAGE = "direct_bus_voltage"
    FIXED_TAP_RECONSTRUCTION = "fixed_tap_reconstruction"
    CUSTOM_OUTPUT = "custom_output"


class DelayKind(_StrEnum):
    ZERO_DELAY = "zero_delay"
    DEFINITE_TIME = "definite_time"
    DWELL = "dwell"
    CUSTOM = "custom"


class ProfileKind(_StrEnum):
    STEP = "step"
    RAMP = "ramp"
    SAMPLED = "sampled"
    PIECEWISE_CONSTANT = "piecewise_constant"
    CUSTOM = "custom"


class EventKind(_StrEnum):
    GENERATOR_TRIP = "generator_trip"
    COLLECTOR_TRIP = "collector_trip"
    PROTECTION_TRIP = "protection_trip"
    FIXED_PF_RAMP = "fixed_pf_ramp"
    SHUNT_SWITCH = "shunt_switch"
    LINE_TOPOLOGY_ACTION = "line_topology_action"
    EXCHANGE_REDUCTION = "exchange_reduction"
    LOAD_PUMP_SHEDDING = "load_pump_shedding"
    DER_BLOCK = "der_block"
    HVDC_MODE_CHANGE = "hvdc_mode_change"


class ControlKind(_StrEnum):
    AVR_SETPOINT = "avr_setpoint"
    GENERATOR_VOLTAGE_CONTROL = "generator_voltage_control"
    IBR_VOLTAGE_SUPPORT = "ibr_voltage_support"
    STATCOM_SVC = "statcom_svc"
    HVDC_REACTIVE_COMMAND = "hvdc_reactive_command"
    SHUNT_ACTION = "shunt_action"
    RAMP_SMOOTHING = "ramp_smoothing"
    TOPOLOGY_BLOCK = "topology_block"


class ControlMode(_StrEnum):
    VOLTAGE = "voltage"
    REACTIVE_POWER = "reactive_power"
    FIXED_POWER_FACTOR = "fixed_power_factor"
    FIXED_ACTIVE_POWER = "fixed_active_power"
    DROOP = "droop"
    AC_VOLTAGE_EMULATION = "ac_voltage_emulation"
    CURRENT_LIMIT = "current_limit"
    UNAVAILABLE = "unavailable"
    UNKNOWN = "unknown"


class ActionDirection(_StrEnum):
    ABSORB_REACTIVE = "absorb_reactive"
    INJECT_REACTIVE = "inject_reactive"
    LOWER_VOLTAGE = "lower_voltage"
    RAISE_VOLTAGE = "raise_voltage"
    MODE_CHANGE = "mode_change"
    TOPOLOGY_CHANGE = "topology_change"
    LIMIT_RAMP = "limit_ramp"


class AssetStatus(_StrEnum):
    IN_SERVICE = "in_service"
    OUT_OF_SERVICE = "out_of_service"
    TRIPPED = "tripped"
    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"
    UNKNOWN = "unknown"


class LimiterStatus(_StrEnum):
    INACTIVE = "inactive"
    ACTIVE = "active"
    SATURATED = "saturated"
    BLOCKED = "blocked"
    UNKNOWN = "unknown"


class ProtectionStatus(_StrEnum):
    ARMED = "armed"
    PICKED_UP = "picked_up"
    TRIPPED = "tripped"
    BLOCKED = "blocked"
    UNKNOWN = "unknown"


class AssessmentStatus(_StrEnum):
    CERTIFIED = "certified"
    EMPIRICAL = "empirical"
    DATA_LIMITED = "data_limited"
    FAILED = "failed"
    NOT_EVALUATED = "not_evaluated"


Matrix = tuple[tuple[float, ...], ...]


def _finite(value: float, field_name: str) -> None:
    if not isinstance(value, int | float) or not math.isfinite(float(value)):
        raise DataModelError(f"{field_name} must be a finite number")


def _nonempty(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise DataModelError(f"{field_name} must be a non-empty string")


def _coerce_enum(value: Any, enum_cls: type[Enum], field_name: str) -> Enum:
    if isinstance(value, enum_cls):
        return value
    try:
        return enum_cls(value)
    except Exception as exc:
        allowed = ", ".join(item.value for item in enum_cls)
        raise DataModelError(f"{field_name} must be one of: {allowed}") from exc


def _as_tuple(value: Sequence[Any] | None) -> tuple[Any, ...]:
    if value is None:
        return ()
    return tuple(value)


def _freeze_metadata(value: Mapping[str, Any] | None) -> Mapping[str, Any]:
    return MappingProxyType(dict(value or {}))


def _require_no_duplicate(keys: Sequence[str], field_name: str) -> None:
    seen: set[str] = set()
    for key in keys:
        if key in seen:
            raise DataModelError(f"{field_name} contains duplicate key {key!r}")
        seen.add(key)


def _matrix(value: Sequence[Sequence[float]], field_name: str) -> Matrix:
    rows: list[tuple[float, ...]] = []
    width: int | None = None
    for row_index, row in enumerate(value):
        converted = tuple(float(item) for item in row)
        for col_index, item in enumerate(converted):
            _finite(item, f"{field_name}[{row_index}][{col_index}]")
        if width is None:
            width = len(converted)
        elif len(converted) != width:
            raise DataModelError(f"{field_name} must be rectangular")
        rows.append(converted)
    return tuple(rows)


def _check_shape(matrix: Matrix, rows: int, cols: int, field_name: str) -> None:
    if len(matrix) != rows:
        raise DataModelError(f"{field_name} must have {rows} rows")
    for index, row in enumerate(matrix):
        if len(row) != cols:
            raise DataModelError(f"{field_name} row {index} must have {cols} columns")


def _require_nonnegative_matrix(matrix: Matrix, field_name: str) -> None:
    for row_index, row in enumerate(matrix):
        for col_index, value in enumerate(row):
            if value < 0:
                raise DataModelError(
                    f"{field_name}[{row_index}][{col_index}] must be nonnegative"
                )


def _to_dict(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return {field.name: _to_dict(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, Mapping):
        return {str(key): _to_dict(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_to_dict(item) for item in value]
    return value


class Serializable:
    """Mixin for JSON-friendly dictionaries."""

    def to_dict(self) -> dict[str, Any]:
        return _to_dict(self)


@dataclass(frozen=True, slots=True)
class Interval(Serializable):
    """Closed interval for uncertain or exact scalar quantities."""

    lower: float
    upper: float
    unit: Unit | str = Unit.NONE
    nominal: float | None = None

    def __post_init__(self) -> None:
        lower = float(self.lower)
        upper = float(self.upper)
        _finite(lower, "Interval.lower")
        _finite(upper, "Interval.upper")
        if lower > upper:
            raise DataModelError("Interval.lower must be <= Interval.upper")
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)
        if self.nominal is not None:
            nominal = float(self.nominal)
            _finite(nominal, "Interval.nominal")
            if not lower <= nominal <= upper:
                raise DataModelError("Interval.nominal must lie inside [lower, upper]")
            object.__setattr__(self, "nominal", nominal)

    @classmethod
    def exact(cls, value: float, unit: Unit | str = Unit.NONE) -> "Interval":
        return cls(value, value, unit=unit, nominal=value)

    @property
    def midpoint(self) -> float:
        if self.nominal is not None:
            return self.nominal
        return 0.5 * (self.lower + self.upper)

    @property
    def width(self) -> float:
        return self.upper - self.lower

    def require_nonnegative(self, field_name: str) -> None:
        if self.lower < 0:
            raise DataModelError(f"{field_name} must be nonnegative")

    def require_positive(self, field_name: str) -> None:
        if self.lower <= 0:
            raise DataModelError(f"{field_name} must have a positive lower bound")


@dataclass(frozen=True, slots=True)
class LocationSpec(Serializable):
    """Electrical location independent of a specific simulator."""

    bus_id: str | None = None
    area: str | None = None
    zone: str | None = None
    base_kv: float | None = None
    model_family: str | None = None
    metadata: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.base_kv is not None:
            base_kv = float(self.base_kv)
            _finite(base_kv, "LocationSpec.base_kv")
            if base_kv <= 0:
                raise DataModelError("LocationSpec.base_kv must be positive")
            object.__setattr__(self, "base_kv", base_kv)
        object.__setattr__(self, "metadata", _freeze_metadata(self.metadata))


@dataclass(frozen=True, slots=True)
class AssetRef(Serializable):
    """Reference to a network, dynamic, protection, or control asset."""

    asset_id: str
    model_family: str | None = None
    component_type: str | None = None
    bus_id: str | None = None
    description: str | None = None
    metadata: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        _nonempty(self.asset_id, "AssetRef.asset_id")
        object.__setattr__(self, "metadata", _freeze_metadata(self.metadata))


@dataclass(frozen=True, slots=True)
class TimeProfile(Serializable):
    """Time profile for events and controls."""

    kind: ProfileKind
    start_s: float = 0.0
    duration_s: float | None = None
    final_value: float | None = None
    samples: tuple[tuple[float, float], ...] = ()
    unit: Unit | str = Unit.NONE
    description: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", _coerce_enum(self.kind, ProfileKind, "TimeProfile.kind"))
        start_s = float(self.start_s)
        _finite(start_s, "TimeProfile.start_s")
        if start_s < 0:
            raise DataModelError("TimeProfile.start_s must be nonnegative")
        object.__setattr__(self, "start_s", start_s)

        if self.duration_s is not None:
            duration_s = float(self.duration_s)
            _finite(duration_s, "TimeProfile.duration_s")
            if duration_s < 0:
                raise DataModelError("TimeProfile.duration_s must be nonnegative")
            object.__setattr__(self, "duration_s", duration_s)

        samples = tuple((float(t), float(v)) for t, v in self.samples)
        last_time = -math.inf
        for index, (time_s, value) in enumerate(samples):
            _finite(time_s, f"TimeProfile.samples[{index}].time")
            _finite(value, f"TimeProfile.samples[{index}].value")
            if time_s < 0:
                raise DataModelError("TimeProfile sample times must be nonnegative")
            if time_s <= last_time:
                raise DataModelError("TimeProfile sample times must be strictly increasing")
            last_time = time_s
        object.__setattr__(self, "samples", samples)

        if self.kind == ProfileKind.SAMPLED and not samples:
            raise DataModelError("sampled TimeProfile requires samples")
        if self.final_value is not None:
            final_value = float(self.final_value)
            _finite(final_value, "TimeProfile.final_value")
            object.__setattr__(self, "final_value", final_value)


@dataclass(frozen=True, slots=True)
class ProtectionTiming(Serializable):
    """Relay timing and dwell logic."""

    kind: DelayKind
    delay_s: Interval = Interval.exact(0.0, Unit.SECOND)
    dwell_time_s: Interval = Interval.exact(0.0, Unit.SECOND)
    reset_time_s: Interval | None = None
    description: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "kind", _coerce_enum(self.kind, DelayKind, "ProtectionTiming.kind")
        )
        self.delay_s.require_nonnegative("ProtectionTiming.delay_s")
        self.dwell_time_s.require_nonnegative("ProtectionTiming.dwell_time_s")
        if self.reset_time_s is not None:
            self.reset_time_s.require_nonnegative("ProtectionTiming.reset_time_s")
        if self.kind == DelayKind.ZERO_DELAY:
            if self.delay_s.upper != 0 or self.dwell_time_s.upper != 0:
                raise DataModelError("zero-delay protection must have zero delay and dwell time")
        if self.kind == DelayKind.DEFINITE_TIME:
            self.delay_s.require_positive("ProtectionTiming.delay_s")
        if self.kind == DelayKind.DWELL:
            self.dwell_time_s.require_positive("ProtectionTiming.dwell_time_s")


@dataclass(frozen=True, slots=True)
class ProtectedVoltageSpec(Serializable):
    """Relay-side voltage output z_i."""

    output_id: str
    side: MeasurementSide
    kind: ProtectedVoltageKind
    protected_bus_id: str | None = None
    transmission_bus_id: str | None = None
    tap_ratio: Interval | None = None
    reconstruction_error_pu: Interval = Interval.exact(0.0, Unit.PU)
    expression: str | None = None
    metadata: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        _nonempty(self.output_id, "ProtectedVoltageSpec.output_id")
        object.__setattr__(
            self,
            "side",
            _coerce_enum(self.side, MeasurementSide, "ProtectedVoltageSpec.side"),
        )
        object.__setattr__(
            self,
            "kind",
            _coerce_enum(self.kind, ProtectedVoltageKind, "ProtectedVoltageSpec.kind"),
        )
        if self.kind == ProtectedVoltageKind.DIRECT_BUS_VOLTAGE:
            if not self.protected_bus_id:
                raise DataModelError("direct protected voltage requires protected_bus_id")
        if self.kind == ProtectedVoltageKind.FIXED_TAP_RECONSTRUCTION:
            if not self.transmission_bus_id:
                raise DataModelError("fixed-tap reconstruction requires transmission_bus_id")
            if self.tap_ratio is None:
                raise DataModelError("fixed-tap reconstruction requires tap_ratio")
            self.tap_ratio.require_positive("ProtectedVoltageSpec.tap_ratio")
        if self.kind == ProtectedVoltageKind.CUSTOM_OUTPUT and not self.expression:
            raise DataModelError("custom protected voltage requires expression")
        object.__setattr__(self, "metadata", _freeze_metadata(self.metadata))


@dataclass(frozen=True, slots=True)
class DisconnectedAsset(Serializable):
    """Asset disconnected by a protection operation."""

    asset: AssetRef
    p_injection_mw: Interval | None = None
    q_absorption_mvar: Interval | None = None
    q_injection_mvar: Interval | None = None
    trip_delay_s: Interval = Interval.exact(0.0, Unit.SECOND)

    def __post_init__(self) -> None:
        self.trip_delay_s.require_nonnegative("DisconnectedAsset.trip_delay_s")
        if self.q_absorption_mvar is not None:
            self.q_absorption_mvar.require_nonnegative("DisconnectedAsset.q_absorption_mvar")


@dataclass(frozen=True, slots=True)
class ProtectedAsset(Serializable):
    """Protected relay-side asset used by the cascade screen."""

    asset_id: str
    name: str
    protected_voltage: ProtectedVoltageSpec
    threshold_pu: Interval
    timing: ProtectionTiming
    disconnected_assets: tuple[DisconnectedAsset, ...]
    p_injection_mw: Interval | None = None
    q_absorption_mvar: Interval | None = None
    tags: tuple[str, ...] = ()
    metadata: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        _nonempty(self.asset_id, "ProtectedAsset.asset_id")
        _nonempty(self.name, "ProtectedAsset.name")
        self.threshold_pu.require_positive("ProtectedAsset.threshold_pu")
        disconnected_assets = _as_tuple(self.disconnected_assets)
        if not disconnected_assets:
            raise DataModelError("ProtectedAsset requires at least one disconnected asset")
        object.__setattr__(self, "disconnected_assets", disconnected_assets)
        if self.q_absorption_mvar is not None:
            self.q_absorption_mvar.require_nonnegative("ProtectedAsset.q_absorption_mvar")
        object.__setattr__(self, "tags", tuple(self.tags))
        object.__setattr__(self, "metadata", _freeze_metadata(self.metadata))


@dataclass(frozen=True, slots=True)
class FixedPowerFactorSpec(Serializable):
    """Fixed power-factor coupling Q = sigma*kappa*P."""

    pf_abs: Interval
    reactive_sign: int
    description: str | None = None

    def __post_init__(self) -> None:
        if self.pf_abs.lower <= 0 or self.pf_abs.upper > 1:
            raise DataModelError("FixedPowerFactorSpec.pf_abs must lie in (0, 1]")
        if self.reactive_sign not in (-1, 1):
            raise DataModelError("FixedPowerFactorSpec.reactive_sign must be -1 or +1")

    @property
    def nominal_kappa(self) -> float:
        pf = self.pf_abs.midpoint
        return math.tan(math.acos(pf))


@dataclass(frozen=True, slots=True)
class CandidateEvent(Serializable):
    """Candidate disturbance or discrete event d_j."""

    event_id: str
    kind: EventKind
    name: str
    location: LocationSpec | None = None
    affected_assets: tuple[AssetRef, ...] = ()
    profile: TimeProfile = TimeProfile(ProfileKind.STEP)
    voltage_raising_q_mvar: Interval | None = None
    delta_p_mw: Interval | None = None
    delta_q_mvar: Interval | None = None
    fixed_pf: FixedPowerFactorSpec | None = None
    source: str | None = None
    metadata: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        _nonempty(self.event_id, "CandidateEvent.event_id")
        _nonempty(self.name, "CandidateEvent.name")
        object.__setattr__(self, "kind", _coerce_enum(self.kind, EventKind, "CandidateEvent.kind"))
        affected_assets = _as_tuple(self.affected_assets)
        object.__setattr__(self, "affected_assets", affected_assets)
        if self.voltage_raising_q_mvar is not None:
            self.voltage_raising_q_mvar.require_nonnegative(
                "CandidateEvent.voltage_raising_q_mvar"
            )
        trip_kinds = {
            EventKind.GENERATOR_TRIP,
            EventKind.COLLECTOR_TRIP,
            EventKind.PROTECTION_TRIP,
            EventKind.DER_BLOCK,
        }
        if self.kind in trip_kinds and not affected_assets:
            raise DataModelError(f"{self.kind.value} requires affected_assets")
        if self.kind == EventKind.FIXED_PF_RAMP and self.fixed_pf is None:
            raise DataModelError("fixed-PF ramp event requires fixed_pf")
        object.__setattr__(self, "metadata", _freeze_metadata(self.metadata))


@dataclass(frozen=True, slots=True)
class ResponseEnvelope(Serializable):
    """Dynamic response envelope for a control channel."""

    delay_s: Interval = Interval.exact(0.0, Unit.SECOND)
    rise_time_s: Interval | None = None
    settling_time_s: Interval | None = None
    gain_scale: Interval = Interval.exact(1.0)
    lower_bound_scale: Interval = Interval.exact(1.0)
    certified: bool = False
    source: str | None = None

    def __post_init__(self) -> None:
        self.delay_s.require_nonnegative("ResponseEnvelope.delay_s")
        self.gain_scale.require_nonnegative("ResponseEnvelope.gain_scale")
        self.lower_bound_scale.require_nonnegative("ResponseEnvelope.lower_bound_scale")
        if self.rise_time_s is not None:
            self.rise_time_s.require_nonnegative("ResponseEnvelope.rise_time_s")
        if self.settling_time_s is not None:
            self.settling_time_s.require_nonnegative("ResponseEnvelope.settling_time_s")


@dataclass(frozen=True, slots=True)
class CandidateControl(Serializable):
    """Candidate preventive action u_k."""

    control_id: str
    kind: ControlKind
    name: str
    direction: ActionDirection
    location: LocationSpec | None = None
    controlled_assets: tuple[AssetRef, ...] = ()
    mode: ControlMode = ControlMode.UNKNOWN
    response: ResponseEnvelope = ResponseEnvelope()
    alpha_limits: Interval = Interval(0.0, 1.0, nominal=0.0)
    magnitude_mvar: Interval | None = None
    cost: float = 1.0
    available: bool = True
    source: str | None = None
    metadata: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        _nonempty(self.control_id, "CandidateControl.control_id")
        _nonempty(self.name, "CandidateControl.name")
        object.__setattr__(
            self, "kind", _coerce_enum(self.kind, ControlKind, "CandidateControl.kind")
        )
        object.__setattr__(
            self,
            "direction",
            _coerce_enum(self.direction, ActionDirection, "CandidateControl.direction"),
        )
        object.__setattr__(
            self, "mode", _coerce_enum(self.mode, ControlMode, "CandidateControl.mode")
        )
        self.alpha_limits.require_nonnegative("CandidateControl.alpha_limits")
        if self.magnitude_mvar is not None:
            self.magnitude_mvar.require_nonnegative("CandidateControl.magnitude_mvar")
        cost = float(self.cost)
        _finite(cost, "CandidateControl.cost")
        if cost < 0:
            raise DataModelError("CandidateControl.cost must be nonnegative")
        object.__setattr__(self, "cost", cost)
        object.__setattr__(self, "controlled_assets", _as_tuple(self.controlled_assets))
        object.__setattr__(self, "metadata", _freeze_metadata(self.metadata))


@dataclass(frozen=True, slots=True)
class AssetStatusSpec(Serializable):
    asset: AssetRef
    status: AssetStatus

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "status", _coerce_enum(self.status, AssetStatus, "AssetStatusSpec.status")
        )


@dataclass(frozen=True, slots=True)
class ShuntStatusSpec(Serializable):
    shunt_id: str
    status: AssetStatus
    b_pu: float | None = None

    def __post_init__(self) -> None:
        _nonempty(self.shunt_id, "ShuntStatusSpec.shunt_id")
        object.__setattr__(
            self, "status", _coerce_enum(self.status, AssetStatus, "ShuntStatusSpec.status")
        )
        if self.b_pu is not None:
            b_pu = float(self.b_pu)
            _finite(b_pu, "ShuntStatusSpec.b_pu")
            object.__setattr__(self, "b_pu", b_pu)


@dataclass(frozen=True, slots=True)
class TapStatusSpec(Serializable):
    transformer_id: str
    ratio: Interval
    fixed_for_window: bool = True

    def __post_init__(self) -> None:
        _nonempty(self.transformer_id, "TapStatusSpec.transformer_id")
        self.ratio.require_positive("TapStatusSpec.ratio")


@dataclass(frozen=True, slots=True)
class ControllerModeState(Serializable):
    controller_id: str
    mode: ControlMode
    enabled: bool = True
    response: ResponseEnvelope | None = None

    def __post_init__(self) -> None:
        _nonempty(self.controller_id, "ControllerModeState.controller_id")
        object.__setattr__(
            self,
            "mode",
            _coerce_enum(self.mode, ControlMode, "ControllerModeState.mode"),
        )


@dataclass(frozen=True, slots=True)
class LimiterStateSpec(Serializable):
    limiter_id: str
    status: LimiterStatus
    controller_id: str | None = None

    def __post_init__(self) -> None:
        _nonempty(self.limiter_id, "LimiterStateSpec.limiter_id")
        object.__setattr__(
            self,
            "status",
            _coerce_enum(self.status, LimiterStatus, "LimiterStateSpec.status"),
        )


@dataclass(frozen=True, slots=True)
class ProtectionStateSpec(Serializable):
    protected_asset_id: str
    status: ProtectionStatus
    pickup_timer_s: float = 0.0

    def __post_init__(self) -> None:
        _nonempty(self.protected_asset_id, "ProtectionStateSpec.protected_asset_id")
        object.__setattr__(
            self,
            "status",
            _coerce_enum(self.status, ProtectionStatus, "ProtectionStateSpec.status"),
        )
        pickup_timer_s = float(self.pickup_timer_s)
        _finite(pickup_timer_s, "ProtectionStateSpec.pickup_timer_s")
        if pickup_timer_s < 0:
            raise DataModelError("ProtectionStateSpec.pickup_timer_s must be nonnegative")
        object.__setattr__(self, "pickup_timer_s", pickup_timer_s)


@dataclass(frozen=True, slots=True)
class ModeState(Serializable):
    """Discrete mode m for a mode-wise DAE computation."""

    mode_id: str
    topology_id: str
    connected_assets: tuple[AssetStatusSpec, ...] = ()
    shunts: tuple[ShuntStatusSpec, ...] = ()
    taps: tuple[TapStatusSpec, ...] = ()
    controller_modes: tuple[ControllerModeState, ...] = ()
    limiter_states: tuple[LimiterStateSpec, ...] = ()
    protection_states: tuple[ProtectionStateSpec, ...] = ()
    metadata: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        _nonempty(self.mode_id, "ModeState.mode_id")
        _nonempty(self.topology_id, "ModeState.topology_id")
        object.__setattr__(self, "connected_assets", _as_tuple(self.connected_assets))
        object.__setattr__(self, "shunts", _as_tuple(self.shunts))
        object.__setattr__(self, "taps", _as_tuple(self.taps))
        object.__setattr__(self, "controller_modes", _as_tuple(self.controller_modes))
        object.__setattr__(self, "limiter_states", _as_tuple(self.limiter_states))
        object.__setattr__(self, "protection_states", _as_tuple(self.protection_states))
        _require_no_duplicate(
            [item.asset.asset_id for item in self.connected_assets], "ModeState.connected_assets"
        )
        _require_no_duplicate([item.shunt_id for item in self.shunts], "ModeState.shunts")
        _require_no_duplicate([item.transformer_id for item in self.taps], "ModeState.taps")
        _require_no_duplicate(
            [item.controller_id for item in self.controller_modes],
            "ModeState.controller_modes",
        )
        _require_no_duplicate(
            [item.limiter_id for item in self.limiter_states], "ModeState.limiter_states"
        )
        _require_no_duplicate(
            [item.protected_asset_id for item in self.protection_states],
            "ModeState.protection_states",
        )
        object.__setattr__(self, "metadata", _freeze_metadata(self.metadata))


@dataclass(frozen=True, slots=True)
class ParameterBound(Serializable):
    """One uncertain scalar bound."""

    target_id: str
    parameter: str
    interval: Interval
    source: str | None = None

    def __post_init__(self) -> None:
        _nonempty(self.target_id, "ParameterBound.target_id")
        _nonempty(self.parameter, "ParameterBound.parameter")


@dataclass(frozen=True, slots=True)
class ControllerResponseUncertainty(Serializable):
    control_id: str
    delay_s: Interval | None = None
    rise_time_s: Interval | None = None
    gain_scale: Interval | None = None
    lower_bound_scale: Interval | None = None
    source: str | None = None

    def __post_init__(self) -> None:
        _nonempty(self.control_id, "ControllerResponseUncertainty.control_id")
        for name in ("delay_s", "rise_time_s", "gain_scale", "lower_bound_scale"):
            interval = getattr(self, name)
            if interval is not None:
                interval.require_nonnegative(f"ControllerResponseUncertainty.{name}")


@dataclass(frozen=True, slots=True)
class ComplianceEnvelope(Serializable):
    asset_id: str
    compliance_fraction: Interval
    reactive_response_scale: Interval
    source: str | None = None

    def __post_init__(self) -> None:
        _nonempty(self.asset_id, "ComplianceEnvelope.asset_id")
        if self.compliance_fraction.lower < 0 or self.compliance_fraction.upper > 1:
            raise DataModelError("ComplianceEnvelope.compliance_fraction must lie in [0, 1]")
        self.reactive_response_scale.require_nonnegative(
            "ComplianceEnvelope.reactive_response_scale"
        )


@dataclass(frozen=True, slots=True)
class UncertaintySet(Serializable):
    """Uncertainty set Theta used by the robust screen."""

    set_id: str
    thresholds_pu: tuple[ParameterBound, ...] = ()
    tap_ratios: tuple[ParameterBound, ...] = ()
    delays_s: tuple[ParameterBound, ...] = ()
    q_absorption_mvar: tuple[ParameterBound, ...] = ()
    fixed_pf_values: tuple[ParameterBound, ...] = ()
    shunt_status: tuple[ParameterBound, ...] = ()
    controller_responses: tuple[ControllerResponseUncertainty, ...] = ()
    compliance_envelopes: tuple[ComplianceEnvelope, ...] = ()
    reconstruction_errors_pu: tuple[ParameterBound, ...] = ()
    model_errors_pu: tuple[ParameterBound, ...] = ()
    metadata: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        _nonempty(self.set_id, "UncertaintySet.set_id")
        for field_name in (
            "thresholds_pu",
            "tap_ratios",
            "delays_s",
            "q_absorption_mvar",
            "fixed_pf_values",
            "shunt_status",
            "controller_responses",
            "compliance_envelopes",
            "reconstruction_errors_pu",
            "model_errors_pu",
        ):
            object.__setattr__(self, field_name, _as_tuple(getattr(self, field_name)))
        object.__setattr__(self, "metadata", _freeze_metadata(self.metadata))


@dataclass(frozen=True, slots=True)
class ChannelAssessment(Serializable):
    protected_asset_id: str
    channel_id: str
    status: AssessmentStatus
    bound_type: str
    notes: str | None = None

    def __post_init__(self) -> None:
        _nonempty(self.protected_asset_id, "ChannelAssessment.protected_asset_id")
        _nonempty(self.channel_id, "ChannelAssessment.channel_id")
        object.__setattr__(
            self,
            "status",
            _coerce_enum(self.status, AssessmentStatus, "ChannelAssessment.status"),
        )
        _nonempty(self.bound_type, "ChannelAssessment.bound_type")


@dataclass(frozen=True, slots=True)
class TimeGridControlAuthority(Serializable):
    time_s: float
    lower_bound_matrix: Matrix

    def __post_init__(self) -> None:
        time_s = float(self.time_s)
        _finite(time_s, "TimeGridControlAuthority.time_s")
        if time_s < 0:
            raise DataModelError("TimeGridControlAuthority.time_s must be nonnegative")
        object.__setattr__(self, "time_s", time_s)
        object.__setattr__(
            self, "lower_bound_matrix", _matrix(self.lower_bound_matrix, "lower_bound_matrix")
        )


@dataclass(frozen=True, slots=True)
class WindowMapResult(Serializable):
    """Finite-window disturbance and control maps."""

    mode_id: str
    protected_asset_ids: tuple[str, ...]
    event_ids: tuple[str, ...]
    control_ids: tuple[str, ...]
    horizons_s: tuple[float, ...]
    k_pickup_upper: Matrix
    k_trip_upper: Matrix
    control_authority_lower: tuple[TimeGridControlAuthority, ...] = ()
    channel_assessments: tuple[ChannelAssessment, ...] = ()
    data_limited_assets: tuple[str, ...] = ()
    status: AssessmentStatus = AssessmentStatus.NOT_EVALUATED
    metadata: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        _nonempty(self.mode_id, "WindowMapResult.mode_id")
        object.__setattr__(
            self, "status", _coerce_enum(self.status, AssessmentStatus, "WindowMapResult.status")
        )
        protected_asset_ids = tuple(self.protected_asset_ids)
        event_ids = tuple(self.event_ids)
        control_ids = tuple(self.control_ids)
        horizons_s = tuple(float(item) for item in self.horizons_s)
        for index, horizon in enumerate(horizons_s):
            _finite(horizon, f"WindowMapResult.horizons_s[{index}]")
            if horizon < 0:
                raise DataModelError("WindowMapResult horizons must be nonnegative")
        if len(horizons_s) != len(protected_asset_ids):
            raise DataModelError("WindowMapResult.horizons_s must match protected_asset_ids")
        k_pickup = _matrix(self.k_pickup_upper, "WindowMapResult.k_pickup_upper")
        k_trip = _matrix(self.k_trip_upper, "WindowMapResult.k_trip_upper")
        _check_shape(k_pickup, len(protected_asset_ids), len(event_ids), "k_pickup_upper")
        _check_shape(k_trip, len(protected_asset_ids), len(event_ids), "k_trip_upper")
        _require_nonnegative_matrix(k_pickup, "k_pickup_upper")
        _require_nonnegative_matrix(k_trip, "k_trip_upper")
        for authority in self.control_authority_lower:
            _check_shape(
                authority.lower_bound_matrix,
                len(protected_asset_ids),
                len(control_ids),
                "control_authority_lower.lower_bound_matrix",
            )
        object.__setattr__(self, "protected_asset_ids", protected_asset_ids)
        object.__setattr__(self, "event_ids", event_ids)
        object.__setattr__(self, "control_ids", control_ids)
        object.__setattr__(self, "horizons_s", horizons_s)
        object.__setattr__(self, "k_pickup_upper", k_pickup)
        object.__setattr__(self, "k_trip_upper", k_trip)
        object.__setattr__(self, "control_authority_lower", _as_tuple(self.control_authority_lower))
        object.__setattr__(self, "channel_assessments", _as_tuple(self.channel_assessments))
        object.__setattr__(self, "data_limited_assets", tuple(self.data_limited_assets))
        object.__setattr__(self, "metadata", _freeze_metadata(self.metadata))


@dataclass(frozen=True, slots=True)
class CascadeLayer(Serializable):
    layer_index: int
    newly_picked_up: tuple[str, ...]
    cumulative_set: tuple[str, ...]
    max_margin_exceedance: float = 0.0

    def __post_init__(self) -> None:
        if self.layer_index < 0:
            raise DataModelError("CascadeLayer.layer_index must be nonnegative")
        exceedance = float(self.max_margin_exceedance)
        _finite(exceedance, "CascadeLayer.max_margin_exceedance")
        object.__setattr__(self, "max_margin_exceedance", exceedance)
        object.__setattr__(self, "newly_picked_up", tuple(self.newly_picked_up))
        object.__setattr__(self, "cumulative_set", tuple(self.cumulative_set))


@dataclass(frozen=True, slots=True)
class CascadeResult(Serializable):
    """Result of iterating the threshold cascade map."""

    mode_id: str
    seed_ids: tuple[str, ...]
    layers: tuple[CascadeLayer, ...]
    fixed_point: tuple[str, ...]
    no_secondary_certified: bool
    status: AssessmentStatus
    data_limited_assets: tuple[str, ...] = ()
    notes: str | None = None

    def __post_init__(self) -> None:
        _nonempty(self.mode_id, "CascadeResult.mode_id")
        object.__setattr__(
            self, "status", _coerce_enum(self.status, AssessmentStatus, "CascadeResult.status")
        )
        object.__setattr__(self, "seed_ids", tuple(self.seed_ids))
        object.__setattr__(self, "layers", _as_tuple(self.layers))
        object.__setattr__(self, "fixed_point", tuple(self.fixed_point))
        object.__setattr__(self, "data_limited_assets", tuple(self.data_limited_assets))


@dataclass(frozen=True, slots=True)
class ControlSelection(Serializable):
    control_id: str
    alpha: float
    magnitude_mvar: float | None = None
    saturated: bool = False
    cost_contribution: float | None = None

    def __post_init__(self) -> None:
        _nonempty(self.control_id, "ControlSelection.control_id")
        alpha = float(self.alpha)
        _finite(alpha, "ControlSelection.alpha")
        if alpha < 0:
            raise DataModelError("ControlSelection.alpha must be nonnegative")
        object.__setattr__(self, "alpha", alpha)
        if self.magnitude_mvar is not None:
            magnitude = float(self.magnitude_mvar)
            _finite(magnitude, "ControlSelection.magnitude_mvar")
            object.__setattr__(self, "magnitude_mvar", magnitude)
        if self.cost_contribution is not None:
            cost = float(self.cost_contribution)
            _finite(cost, "ControlSelection.cost_contribution")
            object.__setattr__(self, "cost_contribution", cost)


@dataclass(frozen=True, slots=True)
class BindingConstraint(Serializable):
    protected_asset_id: str
    time_s: float
    lhs: float
    rhs: float
    slack: float

    def __post_init__(self) -> None:
        _nonempty(self.protected_asset_id, "BindingConstraint.protected_asset_id")
        for name in ("time_s", "lhs", "rhs", "slack"):
            value = float(getattr(self, name))
            _finite(value, f"BindingConstraint.{name}")
            object.__setattr__(self, name, value)
        if self.time_s < 0:
            raise DataModelError("BindingConstraint.time_s must be nonnegative")


@dataclass(frozen=True, slots=True)
class MitigationResult(Serializable):
    """Result of the mitigation LP/QP."""

    mode_id: str
    seed_ids: tuple[str, ...]
    feasible: bool
    objective_value: float
    slack_eta: float
    selections: tuple[ControlSelection, ...]
    binding_constraints: tuple[BindingConstraint, ...] = ()
    solver_status: str = "not_run"
    status: AssessmentStatus = AssessmentStatus.NOT_EVALUATED
    unavailable_controls: tuple[str, ...] = ()
    ineffective_controls: tuple[str, ...] = ()
    required_mvar_total: float | None = None
    notes: str | None = None
    metadata: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        _nonempty(self.mode_id, "MitigationResult.mode_id")
        object.__setattr__(
            self,
            "status",
            _coerce_enum(self.status, AssessmentStatus, "MitigationResult.status"),
        )
        for name in ("objective_value", "slack_eta"):
            value = float(getattr(self, name))
            _finite(value, f"MitigationResult.{name}")
            object.__setattr__(self, name, value)
        if self.slack_eta < 0:
            raise DataModelError("MitigationResult.slack_eta must be nonnegative")
        if self.required_mvar_total is not None:
            required = float(self.required_mvar_total)
            _finite(required, "MitigationResult.required_mvar_total")
            if required < 0:
                raise DataModelError("MitigationResult.required_mvar_total must be nonnegative")
            object.__setattr__(self, "required_mvar_total", required)
        object.__setattr__(self, "seed_ids", tuple(self.seed_ids))
        object.__setattr__(self, "selections", _as_tuple(self.selections))
        object.__setattr__(self, "binding_constraints", _as_tuple(self.binding_constraints))
        object.__setattr__(self, "unavailable_controls", tuple(self.unavailable_controls))
        object.__setattr__(self, "ineffective_controls", tuple(self.ineffective_controls))
        object.__setattr__(self, "metadata", _freeze_metadata(self.metadata))


@dataclass(frozen=True, slots=True)
class TripRecord(Serializable):
    asset_id: str
    time_s: float
    reason: str
    source: str

    def __post_init__(self) -> None:
        _nonempty(self.asset_id, "TripRecord.asset_id")
        _nonempty(self.reason, "TripRecord.reason")
        _nonempty(self.source, "TripRecord.source")
        time_s = float(self.time_s)
        _finite(time_s, "TripRecord.time_s")
        if time_s < 0:
            raise DataModelError("TripRecord.time_s must be nonnegative")
        object.__setattr__(self, "time_s", time_s)


@dataclass(frozen=True, slots=True)
class VoltageExcursion(Serializable):
    protected_asset_id: str
    max_voltage_pu: float
    threshold_pu: float
    time_s: float

    def __post_init__(self) -> None:
        _nonempty(self.protected_asset_id, "VoltageExcursion.protected_asset_id")
        for name in ("max_voltage_pu", "threshold_pu", "time_s"):
            value = float(getattr(self, name))
            _finite(value, f"VoltageExcursion.{name}")
            object.__setattr__(self, name, value)
        if self.time_s < 0:
            raise DataModelError("VoltageExcursion.time_s must be nonnegative")


@dataclass(frozen=True, slots=True)
class ValidationResult(Serializable):
    """Comparison between screen prediction and nonlinear validation."""

    scenario_id: str
    mode_id: str
    seed_ids: tuple[str, ...]
    predicted_trips: tuple[TripRecord, ...]
    simulated_trips: tuple[TripRecord, ...]
    false_positive_trips: tuple[str, ...]
    false_negative_trips: tuple[str, ...]
    max_excursions: tuple[VoltageExcursion, ...] = ()
    status: AssessmentStatus = AssessmentStatus.NOT_EVALUATED
    notes: str | None = None

    def __post_init__(self) -> None:
        _nonempty(self.scenario_id, "ValidationResult.scenario_id")
        _nonempty(self.mode_id, "ValidationResult.mode_id")
        object.__setattr__(
            self,
            "status",
            _coerce_enum(self.status, AssessmentStatus, "ValidationResult.status"),
        )
        object.__setattr__(self, "seed_ids", tuple(self.seed_ids))
        object.__setattr__(self, "predicted_trips", _as_tuple(self.predicted_trips))
        object.__setattr__(self, "simulated_trips", _as_tuple(self.simulated_trips))
        object.__setattr__(self, "false_positive_trips", tuple(self.false_positive_trips))
        object.__setattr__(self, "false_negative_trips", tuple(self.false_negative_trips))
        object.__setattr__(self, "max_excursions", _as_tuple(self.max_excursions))
