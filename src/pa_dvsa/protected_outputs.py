"""Relay voltages and margins: measured, computed from the transmission voltage and the tap, or computed from the
measured plant power, with their uncertainty intervals.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .andes_adapter import AndesAdapterError, AndesCase, VariableAddress
from .data_model import (
    AssessmentStatus,
    DataModelError,
    Interval,
    ProtectedAsset,
    ProtectedVoltageKind,
    Unit,
)


class ProtectedOutputError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class ScalarEnvelope:
    lower: float
    upper: float
    nominal: float
    unit: Unit | str = Unit.PU

    def __post_init__(self) -> None:
        lower = float(self.lower)
        upper = float(self.upper)
        nominal = float(self.nominal)
        if lower > upper:
            raise ProtectedOutputError("ScalarEnvelope.lower must be <= upper")
        if not lower <= nominal <= upper:
            raise ProtectedOutputError("ScalarEnvelope.nominal must lie inside [lower, upper]")
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)
        object.__setattr__(self, "nominal", nominal)

    @classmethod
    def exact(cls, value: float, unit: Unit | str = Unit.PU) -> "ScalarEnvelope":
        return cls(value, value, value, unit)

    @classmethod
    def from_interval(cls, interval: Interval) -> "ScalarEnvelope":
        return cls(interval.lower, interval.upper, interval.midpoint, interval.unit)

    @property
    def width(self) -> float:
        return self.upper - self.lower

    def add(self, other: "ScalarEnvelope") -> "ScalarEnvelope":
        return ScalarEnvelope(
            self.lower + other.lower,
            self.upper + other.upper,
            self.nominal + other.nominal,
            self.unit,
        )

    def positive_divide(self, denominator: "ScalarEnvelope") -> "ScalarEnvelope":
        if denominator.lower <= 0:
            raise ProtectedOutputError("positive division requires denominator lower bound > 0")
        return ScalarEnvelope(
            self.lower / denominator.upper,
            self.upper / denominator.lower,
            self.nominal / denominator.nominal,
            self.unit,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "lower": self.lower,
            "upper": self.upper,
            "nominal": self.nominal,
            "unit": str(self.unit),
        }


@dataclass(frozen=True, slots=True)
class ProtectedOutputEvaluation:
    asset_id: str
    output_id: str
    kind: ProtectedVoltageKind
    z_pu: ScalarEnvelope
    threshold_pu: ScalarEnvelope
    margin_pu: ScalarEnvelope
    status: AssessmentStatus
    worst_case_margin_pu: float
    is_certifiable: bool
    source_bus_id: str | None
    source_address: VariableAddress | None
    reason: str
    metadata: Mapping[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "asset_id": self.asset_id,
            "output_id": self.output_id,
            "kind": self.kind.value,
            "z_pu": self.z_pu.to_dict(),
            "threshold_pu": self.threshold_pu.to_dict(),
            "margin_pu": self.margin_pu.to_dict(),
            "status": self.status.value,
            "worst_case_margin_pu": self.worst_case_margin_pu,
            "is_certifiable": self.is_certifiable,
            "source_bus_id": self.source_bus_id,
            "source_address": None
            if self.source_address is None
            else {
                "model": self.source_address.model,
                "variable": self.source_address.variable,
                "device_id": self.source_address.device_id,
                "address": self.source_address.address,
                "domain": self.source_address.domain,
            },
            "reason": self.reason,
            "metadata": dict(self.metadata or {}),
        }


def _dae_y_value(case: AndesCase, address: int) -> float:
    system = case.require_system()
    try:
        return float(system.dae.y[address])
    except Exception as exc:
        raise ProtectedOutputError(f"Unable to read DAE algebraic y[{address}]") from exc


def bus_voltage_envelope(case: AndesCase, bus_id: str | int | float) -> tuple[ScalarEnvelope, VariableAddress]:
    case.setup()
    try:
        address = case.variable_address("Bus", "v", bus_id)
    except AndesAdapterError as exc:
        raise ProtectedOutputError(str(exc)) from exc
    value = _dae_y_value(case, address.address)
    return ScalarEnvelope.exact(value, Unit.PU), address


def direct_protected_voltage(
    case: AndesCase,
    asset: ProtectedAsset,
) -> tuple[ScalarEnvelope, str, VariableAddress]:
    spec = asset.protected_voltage
    if not spec.protected_bus_id:
        raise ProtectedOutputError(f"{asset.asset_id} direct output has no protected_bus_id")
    voltage, address = bus_voltage_envelope(case, spec.protected_bus_id)
    epsilon = ScalarEnvelope.from_interval(spec.reconstruction_error_pu)
    return voltage.add(epsilon), str(spec.protected_bus_id), address


def reconstructed_fixed_tap_voltage(
    case: AndesCase,
    asset: ProtectedAsset,
) -> tuple[ScalarEnvelope, str, VariableAddress]:
    spec = asset.protected_voltage
    if not spec.transmission_bus_id:
        raise ProtectedOutputError(
            f"{asset.asset_id} fixed-tap reconstruction has no transmission_bus_id"
        )
    if spec.tap_ratio is None:
        raise ProtectedOutputError(f"{asset.asset_id} fixed-tap reconstruction has no tap ratio")
    try:
        spec.tap_ratio.require_positive("tap_ratio")
    except DataModelError as exc:
        raise ProtectedOutputError(str(exc)) from exc
    voltage, address = bus_voltage_envelope(case, spec.transmission_bus_id)
    tap = ScalarEnvelope.from_interval(spec.tap_ratio)
    epsilon = ScalarEnvelope.from_interval(spec.reconstruction_error_pu)
    return voltage.positive_divide(tap).add(epsilon), str(spec.transmission_bus_id), address


def evaluate_protected_output(
    case: AndesCase,
    asset: ProtectedAsset,
) -> ProtectedOutputEvaluation:
    spec = asset.protected_voltage
    kind = ProtectedVoltageKind(spec.kind)
    if kind == ProtectedVoltageKind.DIRECT_BUS_VOLTAGE:
        z_pu, source_bus_id, address = direct_protected_voltage(case, asset)
    elif kind == ProtectedVoltageKind.FIXED_TAP_RECONSTRUCTION:
        z_pu, source_bus_id, address = reconstructed_fixed_tap_voltage(case, asset)
    else:
        raise ProtectedOutputError(
            f"{asset.asset_id} uses custom output {spec.expression!r}; "
            "custom output evaluators are not implemented in the protected-output layer"
        )

    threshold = ScalarEnvelope.from_interval(asset.threshold_pu)
    margin = ScalarEnvelope(
        threshold.lower - z_pu.upper,
        threshold.upper - z_pu.lower,
        threshold.nominal - z_pu.nominal,
        Unit.PU,
    )
    worst_case_margin = margin.lower
    is_certifiable = worst_case_margin > 0.0
    status = AssessmentStatus.CERTIFIED if is_certifiable else AssessmentStatus.DATA_LIMITED
    reason = "positive_worst_case_margin" if is_certifiable else "nonpositive_worst_case_margin"
    return ProtectedOutputEvaluation(
        asset_id=asset.asset_id,
        output_id=spec.output_id,
        kind=kind,
        z_pu=z_pu,
        threshold_pu=threshold,
        margin_pu=margin,
        status=status,
        worst_case_margin_pu=worst_case_margin,
        is_certifiable=is_certifiable,
        source_bus_id=source_bus_id,
        source_address=address,
        reason=reason,
        metadata={"threshold_equation": "h_i = V_trip - z_i^0"},
    )


def evaluate_protected_outputs(
    case: AndesCase,
    assets: list[ProtectedAsset] | tuple[ProtectedAsset, ...],
) -> tuple[ProtectedOutputEvaluation, ...]:
    return tuple(evaluate_protected_output(case, asset) for asset in assets)


import math as _math
from typing import Sequence as _Sequence


@dataclass(frozen=True, slots=True)
class ProtectedOutputModel:
    asset_id: str
    output_index: int
    z0_pu: float
    kind: str = "direct"
    pickup_pu: float = 1.10
    dwell_s: float = 0.05
    tap_nominal: float = 1.0

    def __post_init__(self) -> None:
        if self.kind not in {"direct", "reconstructed"}:
            raise ProtectedOutputError("ProtectedOutputModel.kind must be 'direct' or 'reconstructed'")
        for name in ("z0_pu", "pickup_pu", "dwell_s", "tap_nominal"):
            if not _math.isfinite(float(getattr(self, name))):
                raise ProtectedOutputError(f"ProtectedOutputModel.{name} must be finite")
        if self.kind == "reconstructed" and not float(self.tap_nominal) > 0.0:
            raise ProtectedOutputError("reconstructed outputs need a positive tap ratio")
        if float(self.dwell_s) < 0.0:
            raise ProtectedOutputError("dwell_s must be nonnegative")

    @property
    def nominal_z0(self) -> float:
        if self.kind == "direct":
            return float(self.z0_pu)
        return float(self.z0_pu) / float(self.tap_nominal)

    @property
    def nominal_margin_pu(self) -> float:
        return float(self.pickup_pu) - self.nominal_z0

    @property
    def response_scale(self) -> float:
        return 1.0 if self.kind == "direct" else 1.0 / float(self.tap_nominal)

    def to_dict(self) -> dict[str, Any]:
        return {
            "asset_id": self.asset_id,
            "output_index": self.output_index,
            "z0_pu": self.z0_pu,
            "kind": self.kind,
            "pickup_pu": self.pickup_pu,
            "dwell_s": self.dwell_s,
            "tap_nominal": self.tap_nominal,
            "nominal_margin_pu": self.nominal_margin_pu,
        }


def output_models_from_blocks(
    blocks: _Sequence[Any],
    output_ids: _Sequence[str],
    z0_values: _Sequence[float],
    *,
    kind: str = "direct",
    collector_output_id: Any = None,
    hv_output_id: Any = None,
) -> tuple[ProtectedOutputModel, ...]:
    ids = [str(item) for item in output_ids]
    z0 = [float(item) for item in z0_values]
    col = collector_output_id or (lambda b: f"V_{b.collector_bus}")
    hv = hv_output_id or (lambda b: f"V_{b.hv_bus}")
    models = []
    for block in blocks:
        key = str(col(block) if kind == "direct" else hv(block))
        if key not in ids:
            raise ProtectedOutputError(f"engine has no output {key!r} for asset {block.asset_id}")
        k = ids.index(key)
        models.append(
            ProtectedOutputModel(
                asset_id=str(block.asset_id),
                output_index=k,
                z0_pu=z0[k],
                kind=kind,
                pickup_pu=float(block.pickup_pu),
                dwell_s=float(block.dwell_s),
                tap_nominal=float(block.tap) if kind == "reconstructed" else 1.0,
            )
        )
    return tuple(models)
