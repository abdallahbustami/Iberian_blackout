"""Nonlinear validation utilities for protection-aware DVSA.

The nonlinear validation path is split into simulator-agnostic comparison
logic and a thin ANDES callback adapter. The screen-vs-simulation comparison
can be tested from recorded traces, while the callback can be attached to
``system.TDS.callpert`` for full time-domain validation.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Callable, Mapping, Sequence

import numpy as np
from scipy import sparse
from scipy.sparse import linalg as spla

from .andes_adapter import AndesCase
from .data_model import (
    AssessmentStatus,
    CascadeResult,
    TripRecord,
    ValidationResult,
    VoltageExcursion,
)


class NonlinearValidationError(RuntimeError):
    """Raised when a nonlinear validation problem is malformed."""


ResidualCallback = Callable[[np.ndarray], np.ndarray]
JacobianCallback = Callable[[np.ndarray], Any]
ModeUpdateCallback = Callable[["ProtectionRelaySpec", float, Any], None]


@dataclass(frozen=True, slots=True)
class FrozenAlgebraicValidationResult:
    """Result of a frozen-topology algebraic nonlinear solve.

    ``delta_y`` is the solved algebraic displacement from the supplied
    operating point.  ``output_delta_y`` is the subset selected by
    ``output_addresses`` and is the quantity normally compared against the
    linear screen prediction.
    """

    converged: bool
    iterations: int
    initial_residual_norm: float
    final_residual_norm: float
    delta_y: tuple[float, ...]
    output_addresses: tuple[int, ...]
    output_delta_y: tuple[float, ...]
    status: AssessmentStatus
    metadata: Mapping[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "converged": self.converged,
            "iterations": self.iterations,
            "initial_residual_norm": self.initial_residual_norm,
            "final_residual_norm": self.final_residual_norm,
            "delta_y": list(self.delta_y),
            "output_addresses": list(self.output_addresses),
            "output_delta_y": list(self.output_delta_y),
            "status": self.status.value,
            "metadata": dict(self.metadata or {}),
        }


@dataclass(frozen=True, slots=True)
class RelayTripTarget:
    """ANDES device status update applied when a relay trips."""

    model: str
    device_id: str
    status: float = 0.0

    def __post_init__(self) -> None:
        if not self.model.strip():
            raise NonlinearValidationError("trip target model must be non-empty")
        if not str(self.device_id).strip():
            raise NonlinearValidationError("trip target device_id must be non-empty")
        status = float(self.status)
        if not math.isfinite(status):
            raise NonlinearValidationError("trip target status must be finite")
        object.__setattr__(self, "device_id", str(self.device_id))
        object.__setattr__(self, "status", status)


@dataclass(frozen=True, slots=True)
class ProtectionRelaySpec:
    """Relay-side overvoltage protection used by replay and TDS callback paths."""

    asset_id: str
    threshold_pu: float
    dwell_time_s: float = 0.0
    source_bus_id: str | None = None
    source_address: int | None = None
    trip_targets: tuple[RelayTripTarget, ...] = ()
    reset_rate_s_per_s: float | None = None
    enabled_time_s: float = 0.0
    source: str = "nonlinear"

    def __post_init__(self) -> None:
        if not self.asset_id.strip():
            raise NonlinearValidationError("relay asset_id must be non-empty")
        threshold = float(self.threshold_pu)
        dwell = float(self.dwell_time_s)
        enabled = float(self.enabled_time_s)
        if not math.isfinite(threshold):
            raise NonlinearValidationError("relay threshold must be finite")
        if not math.isfinite(dwell) or dwell < 0.0:
            raise NonlinearValidationError("relay dwell_time_s must be finite and nonnegative")
        if not math.isfinite(enabled) or enabled < 0.0:
            raise NonlinearValidationError("relay enabled_time_s must be finite and nonnegative")
        if self.reset_rate_s_per_s is not None:
            reset = float(self.reset_rate_s_per_s)
            if not math.isfinite(reset) or reset < 0.0:
                raise NonlinearValidationError("relay reset_rate_s_per_s must be finite and nonnegative")
            object.__setattr__(self, "reset_rate_s_per_s", reset)
        if self.source_bus_id is None and self.source_address is None:
            raise NonlinearValidationError("relay must define source_bus_id or source_address")
        if self.source_address is not None and int(self.source_address) < 0:
            raise NonlinearValidationError("relay source_address must be nonnegative")
        object.__setattr__(self, "threshold_pu", threshold)
        object.__setattr__(self, "dwell_time_s", dwell)
        object.__setattr__(self, "enabled_time_s", enabled)
        object.__setattr__(self, "source_address", None if self.source_address is None else int(self.source_address))
        object.__setattr__(
            self,
            "source_bus_id",
            None if self.source_bus_id is None else str(self.source_bus_id),
        )
        object.__setattr__(self, "trip_targets", tuple(self.trip_targets))
        object.__setattr__(self, "source", self.source or "nonlinear")


@dataclass(frozen=True, slots=True)
class ProtectionTrace:
    """One protected relay voltage trace sampled from a nonlinear run."""

    relay: ProtectionRelaySpec
    times_s: tuple[float, ...]
    voltages_pu: tuple[float, ...]

    def __post_init__(self) -> None:
        times = tuple(float(item) for item in self.times_s)
        voltages = tuple(float(item) for item in self.voltages_pu)
        if len(times) != len(voltages):
            raise NonlinearValidationError("trace times and voltages must have the same length")
        if not times:
            raise NonlinearValidationError("trace must contain at least one sample")
        if any(not math.isfinite(item) or item < 0.0 for item in times):
            raise NonlinearValidationError("trace times must be finite and nonnegative")
        if any(b < a for a, b in zip(times, times[1:])):
            raise NonlinearValidationError("trace times must be nondecreasing")
        if any(not math.isfinite(item) for item in voltages):
            raise NonlinearValidationError("trace voltages must be finite")
        object.__setattr__(self, "times_s", times)
        object.__setattr__(self, "voltages_pu", voltages)


@dataclass(frozen=True, slots=True)
class ProtectionReplayResult:
    """Replay output from sampled nonlinear protection traces."""

    pickup_records: tuple[TripRecord, ...]
    trip_records: tuple[TripRecord, ...]
    max_excursions: tuple[VoltageExcursion, ...]

    @property
    def first_pickup(self) -> TripRecord | None:
        return min(self.pickup_records, key=lambda item: item.time_s, default=None)

    @property
    def first_trip(self) -> TripRecord | None:
        return min(self.trip_records, key=lambda item: item.time_s, default=None)

    def to_dict(self) -> dict[str, Any]:
        return {
            "pickup_records": [item.to_dict() for item in self.pickup_records],
            "trip_records": [item.to_dict() for item in self.trip_records],
            "max_excursions": [item.to_dict() for item in self.max_excursions],
            "first_pickup": self.first_pickup.to_dict() if self.first_pickup else None,
            "first_trip": self.first_trip.to_dict() if self.first_trip else None,
        }


@dataclass(frozen=True, slots=True)
class NonlinearValidationReport:
    """Detailed screen-vs-nonlinear comparison report."""

    validation_result: ValidationResult
    first_predicted_trip: TripRecord | None
    first_simulated_pickup: TripRecord | None
    first_simulated_trip: TripRecord | None
    unsafe_false_negative_trips: tuple[str, ...]
    uncertain_assets: tuple[str, ...]
    data_limited_assets: tuple[str, ...]
    robust_success: bool
    metadata: Mapping[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "validation_result": self.validation_result.to_dict(),
            "first_predicted_trip": (
                self.first_predicted_trip.to_dict() if self.first_predicted_trip else None
            ),
            "first_simulated_pickup": (
                self.first_simulated_pickup.to_dict() if self.first_simulated_pickup else None
            ),
            "first_simulated_trip": (
                self.first_simulated_trip.to_dict() if self.first_simulated_trip else None
            ),
            "unsafe_false_negative_trips": list(self.unsafe_false_negative_trips),
            "uncertain_assets": list(self.uncertain_assets),
            "data_limited_assets": list(self.data_limited_assets),
            "robust_success": self.robust_success,
            "metadata": dict(self.metadata or {}),
        }


@dataclass(slots=True)
class _RelayRuntime:
    timer_s: float = 0.0
    picked_up: bool = False
    pickup_time_s: float | None = None
    tripped: bool = False
    trip_time_s: float | None = None
    max_voltage_pu: float = -math.inf
    max_voltage_time_s: float = 0.0


def frozen_algebraic_nonlinear_validation(
    *,
    case: AndesCase | None = None,
    disturbance_g: Sequence[float] | np.ndarray | None = None,
    residual_callback: ResidualCallback | None = None,
    jacobian: Any | None = None,
    jacobian_update: JacobianCallback | None = None,
    y0: Sequence[float] | np.ndarray | None = None,
    output_addresses: Sequence[int] = (),
    max_iter: int = 12,
    tolerance: float = 1.0e-9,
    damping: float = 1.0,
    restore_case: bool = True,
) -> FrozenAlgebraicValidationResult:
    """Solve a frozen-topology nonlinear algebraic validation problem.

    The equation solved is

    ``residual_callback(y) + disturbance_g = 0``.

    If a residual callback is not supplied but an ``AndesCase`` is, the
    function updates ANDES model variables, re-evaluates algebraic residuals
    through ``g_update``/``fg_to_dae``, and refreshes ``G_y`` through
    ``j_update``.  If neither a callback nor a case is supplied, a linear
    residual fallback can still be formed from the supplied Jacobian and is
    marked ``DATA_LIMITED``.
    """

    if max_iter < 1:
        raise NonlinearValidationError("max_iter must be positive")
    if tolerance <= 0.0 or not math.isfinite(tolerance):
        raise NonlinearValidationError("tolerance must be finite and positive")
    if damping <= 0.0 or not math.isfinite(damping):
        raise NonlinearValidationError("damping must be finite and positive")

    metadata: dict[str, Any] = {}
    true_nonlinear_residual = residual_callback is not None
    restore_payload: tuple[Any, np.ndarray, np.ndarray | None] | None = None

    if case is not None:
        case.setup()
        system = case.require_system()
        x_frozen = None
        if hasattr(system, "dae") and hasattr(system.dae, "x"):
            x_frozen = np.asarray(_andes_array_values(system.dae.x), dtype=float)
        original_y = np.asarray(_andes_array_values(getattr(system.dae, "y", [])), dtype=float)
        if y0 is None:
            y0 = original_y.copy()
        if restore_case:
            restore_payload = (system, original_y.copy(), None if x_frozen is None else x_frozen.copy())
        if jacobian is None:
            jacobian = case.jacobians().gy
            metadata["jacobian_source"] = "andes_dae_gy"
        if residual_callback is None:
            models = _andes_validation_models(system)
            residual_callback = lambda y: _andes_algebraic_residual(system, models, x_frozen, y)
            metadata["residual_source"] = "andes_g_update"
            true_nonlinear_residual = True
            if jacobian_update is None:
                jacobian_update = lambda y: _andes_algebraic_jacobian(case, system, models, x_frozen, y)
                metadata["jacobian_update_source"] = "andes_j_update"

    if y0 is None:
        raise NonlinearValidationError("y0 is required unless an ANDES case is supplied")

    y_initial = _as_vector(y0, "y0")
    n = y_initial.size
    if n == 0:
        raise NonlinearValidationError("algebraic state y0 must be non-empty")

    if disturbance_g is None:
        disturbance = np.zeros(n, dtype=float)
    else:
        disturbance = _as_vector(disturbance_g, "disturbance_g")
        if disturbance.size != n:
            raise NonlinearValidationError("disturbance_g length must match y0")

    if jacobian is None and jacobian_update is None:
        raise NonlinearValidationError("jacobian or jacobian_update is required")

    if residual_callback is None:
        if jacobian is None:
            raise NonlinearValidationError("linear fallback residual requires a jacobian")
        j0 = _as_sparse_or_dense(jacobian, "jacobian")
        residual_callback = lambda y: _matvec(j0, y - y_initial)
        metadata["residual_source"] = "frozen_linear_jacobian_fallback"
    else:
        metadata.setdefault("residual_source", "user_nonlinear_callback")

    output_idx = tuple(int(item) for item in output_addresses)
    if any(item < 0 or item >= n for item in output_idx):
        raise NonlinearValidationError("output address out of range")

    y = y_initial.copy()
    initial_norm = math.inf
    final_norm = math.inf
    converged = False
    iterations = 0

    try:
        for iteration in range(max_iter + 1):
            residual = _as_vector(residual_callback(y), "residual") + disturbance
            norm = float(np.linalg.norm(residual, ord=np.inf))
            if iteration == 0:
                initial_norm = norm
            final_norm = norm
            if norm <= tolerance:
                converged = True
                iterations = iteration
                break
            if iteration == max_iter:
                iterations = iteration
                break

            matrix = jacobian_update(y) if jacobian_update is not None else jacobian
            if matrix is None:
                raise NonlinearValidationError("jacobian_update returned None")
            step = _solve_linear(matrix, -residual)
            if not np.all(np.isfinite(step)):
                raise NonlinearValidationError("Newton correction contains non-finite values")
            y = y + damping * step
            if not np.all(np.isfinite(y)):
                raise NonlinearValidationError("Newton iterate contains non-finite values")
    finally:
        if restore_payload is not None:
            _restore_andes_xy(*restore_payload)

    delta_y = y - y_initial
    output_delta = delta_y[np.asarray(output_idx, dtype=int)] if output_idx else np.array([])
    if converged and true_nonlinear_residual:
        status = AssessmentStatus.CERTIFIED
    elif converged:
        status = AssessmentStatus.DATA_LIMITED
    else:
        status = AssessmentStatus.FAILED

    return FrozenAlgebraicValidationResult(
        converged=converged,
        iterations=iterations,
        initial_residual_norm=initial_norm,
        final_residual_norm=final_norm,
        delta_y=tuple(float(item) for item in delta_y),
        output_addresses=output_idx,
        output_delta_y=tuple(float(item) for item in output_delta),
        status=status,
        metadata=metadata,
    )


class TDSProtectionCallback:
    """ANDES TDS callback implementing pickup timers, dwell logic, and trips."""

    def __init__(
        self,
        relays: Sequence[ProtectionRelaySpec],
        *,
        mode_update_callback: ModeUpdateCallback | None = None,
    ) -> None:
        relay_tuple = tuple(relays)
        if not relay_tuple:
            raise NonlinearValidationError("at least one relay is required")
        duplicated = _duplicates(item.asset_id for item in relay_tuple)
        if duplicated:
            raise NonlinearValidationError(f"duplicated relay asset ids: {duplicated}")
        self.relays = relay_tuple
        self.mode_update_callback = mode_update_callback
        self._states = {relay.asset_id: _RelayRuntime() for relay in self.relays}
        self._last_time_s: float | None = None
        self._pickup_records: list[TripRecord] = []
        self._trip_records: list[TripRecord] = []

    def __call__(self, t: float, system: Any) -> None:
        time_s = float(t)
        if not math.isfinite(time_s) or time_s < 0.0:
            raise NonlinearValidationError("TDS callback time must be finite and nonnegative")
        dt = 0.0 if self._last_time_s is None else max(0.0, time_s - self._last_time_s)
        self._last_time_s = time_s

        for relay in self.relays:
            voltage = _read_relay_voltage(system, relay)
            pickup, trip = _advance_relay_state(self._states[relay.asset_id], relay, time_s, dt, voltage)
            if pickup is not None:
                self._pickup_records.append(pickup)
            if trip is not None:
                self._apply_trip(system, relay)
                if self.mode_update_callback is not None:
                    self.mode_update_callback(relay, time_s, system)
                self._trip_records.append(trip)

    @property
    def pickup_records(self) -> tuple[TripRecord, ...]:
        return tuple(self._pickup_records)

    @property
    def trip_records(self) -> tuple[TripRecord, ...]:
        return tuple(self._trip_records)

    @property
    def max_excursions(self) -> tuple[VoltageExcursion, ...]:
        records: list[VoltageExcursion] = []
        for relay in self.relays:
            state = self._states[relay.asset_id]
            if not math.isfinite(state.max_voltage_pu):
                continue
            records.append(
                VoltageExcursion(
                    protected_asset_id=relay.asset_id,
                    max_voltage_pu=state.max_voltage_pu,
                    threshold_pu=relay.threshold_pu,
                    time_s=state.max_voltage_time_s,
                )
            )
        return tuple(records)

    def replay_result(self) -> ProtectionReplayResult:
        return ProtectionReplayResult(
            pickup_records=self.pickup_records,
            trip_records=self.trip_records,
            max_excursions=self.max_excursions,
        )

    def _apply_trip(self, system: Any, relay: ProtectionRelaySpec) -> None:
        for target in relay.trip_targets:
            if hasattr(system, "set_status"):
                system.set_status(target.model, target.device_id, target.status)
                continue
            model = _system_model(system, target.model)
            position = _find_device_position(model, target.device_id)
            status = getattr(model, "u", None)
            values = getattr(status, "v", None)
            if values is None:
                raise NonlinearValidationError(f"{target.model} has no writable u.v status")
            values[position] = target.status


def replay_protection_traces(traces: Sequence[ProtectionTrace]) -> ProtectionReplayResult:
    """Replay relay pickup and dwell logic from sampled nonlinear voltage traces."""

    trace_tuple = tuple(traces)
    if not trace_tuple:
        raise NonlinearValidationError("at least one trace is required")
    pickup_records: list[TripRecord] = []
    trip_records: list[TripRecord] = []
    excursions: list[VoltageExcursion] = []

    for trace in trace_tuple:
        state = _RelayRuntime()
        last_time: float | None = None
        for time_s, voltage in zip(trace.times_s, trace.voltages_pu):
            dt = 0.0 if last_time is None else max(0.0, time_s - last_time)
            last_time = time_s
            pickup, trip = _advance_relay_state(state, trace.relay, time_s, dt, voltage)
            if pickup is not None:
                pickup_records.append(pickup)
            if trip is not None:
                trip_records.append(trip)
        excursions.append(
            VoltageExcursion(
                protected_asset_id=trace.relay.asset_id,
                max_voltage_pu=state.max_voltage_pu,
                threshold_pu=trace.relay.threshold_pu,
                time_s=state.max_voltage_time_s,
            )
        )

    return ProtectionReplayResult(
        pickup_records=tuple(sorted(pickup_records, key=lambda item: (item.time_s, item.asset_id))),
        trip_records=tuple(sorted(trip_records, key=lambda item: (item.time_s, item.asset_id))),
        max_excursions=tuple(excursions),
    )


def compare_screen_to_nonlinear(
    *,
    scenario_id: str,
    mode_id: str,
    seed_ids: Sequence[str] = (),
    predicted: CascadeResult | ValidationResult | ProtectionReplayResult | Sequence[TripRecord | str],
    simulated: ProtectionReplayResult | ValidationResult | Sequence[TripRecord | str],
    simulated_pickups: Sequence[TripRecord] = (),
    max_excursions: Sequence[VoltageExcursion] = (),
    uncertain_assets: Sequence[str] = (),
    data_limited_assets: Sequence[str] = (),
    robust: bool = False,
    metadata: Mapping[str, Any] | None = None,
) -> NonlinearValidationReport:
    """Compare screen predictions against nonlinear protection outcomes.

    For robust screens, the pass/fail criterion is no unsafe false negatives.
    Simulated trips flagged as uncertain or data-limited are retained in
    ``false_negative_trips`` for traceability but excluded from
    ``unsafe_false_negative_trips``.
    """

    predicted_records = _records_from_prediction(predicted, default_source="screen")
    simulated_records = _records_from_prediction(simulated, default_source="nonlinear")
    pickup_records = tuple(simulated_pickups)
    excursion_records = tuple(max_excursions)
    if isinstance(simulated, ProtectionReplayResult):
        pickup_records = simulated.pickup_records
        if not excursion_records:
            excursion_records = simulated.max_excursions
    elif isinstance(simulated, ValidationResult) and not excursion_records:
        excursion_records = simulated.max_excursions

    predicted_set = {item.asset_id for item in predicted_records}
    simulated_set = {item.asset_id for item in simulated_records}
    false_positive = tuple(sorted(predicted_set - simulated_set))
    false_negative = tuple(sorted(simulated_set - predicted_set))

    uncertain = tuple(sorted({str(item) for item in uncertain_assets}))
    data_limited = tuple(sorted({str(item) for item in data_limited_assets}))
    acceptable = set(uncertain) | set(data_limited)
    unsafe_false_negative = tuple(sorted(set(false_negative) - acceptable))

    robust_success = not unsafe_false_negative if robust else not false_negative
    if robust:
        if unsafe_false_negative:
            status = AssessmentStatus.FAILED
        elif set(false_negative) & acceptable or data_limited:
            status = AssessmentStatus.DATA_LIMITED
        else:
            status = AssessmentStatus.CERTIFIED
    else:
        if false_negative:
            status = AssessmentStatus.FAILED
        elif false_positive:
            status = AssessmentStatus.EMPIRICAL
        else:
            status = AssessmentStatus.CERTIFIED

    notes = (
        "robust criterion satisfied: no unsafe false negatives"
        if robust_success
        else "unsafe false negatives observed"
    )
    validation = ValidationResult(
        scenario_id=scenario_id,
        mode_id=mode_id,
        seed_ids=tuple(seed_ids),
        predicted_trips=tuple(sorted(predicted_records, key=lambda item: (item.time_s, item.asset_id))),
        simulated_trips=tuple(sorted(simulated_records, key=lambda item: (item.time_s, item.asset_id))),
        false_positive_trips=false_positive,
        false_negative_trips=false_negative,
        max_excursions=excursion_records,
        status=status,
        notes=notes,
    )
    return NonlinearValidationReport(
        validation_result=validation,
        first_predicted_trip=min(predicted_records, key=lambda item: item.time_s, default=None),
        first_simulated_pickup=min(pickup_records, key=lambda item: item.time_s, default=None),
        first_simulated_trip=min(simulated_records, key=lambda item: item.time_s, default=None),
        unsafe_false_negative_trips=unsafe_false_negative,
        uncertain_assets=uncertain,
        data_limited_assets=data_limited,
        robust_success=robust_success,
        metadata=dict(metadata or {}),
    )


def attach_tds_protection_callback(
    case: AndesCase,
    relays: Sequence[ProtectionRelaySpec],
    *,
    mode_update_callback: ModeUpdateCallback | None = None,
) -> TDSProtectionCallback:
    """Attach a protection callback to ``case.system.TDS.callpert``."""

    system = case.require_system()
    callback = TDSProtectionCallback(relays, mode_update_callback=mode_update_callback)
    if not hasattr(system, "TDS"):
        raise NonlinearValidationError("ANDES system has no TDS object")
    system.TDS.callpert = callback
    return callback


def _advance_relay_state(
    state: _RelayRuntime,
    relay: ProtectionRelaySpec,
    time_s: float,
    dt: float,
    voltage_pu: float,
) -> tuple[TripRecord | None, TripRecord | None]:
    voltage = float(voltage_pu)
    if not math.isfinite(voltage):
        raise NonlinearValidationError("relay voltage must be finite")
    if voltage > state.max_voltage_pu:
        state.max_voltage_pu = voltage
        state.max_voltage_time_s = time_s

    if state.tripped or time_s < relay.enabled_time_s:
        return None, None

    pickup: TripRecord | None = None
    trip: TripRecord | None = None
    if voltage >= relay.threshold_pu:
        if not state.picked_up:
            state.picked_up = True
            state.pickup_time_s = time_s
            pickup = TripRecord(relay.asset_id, time_s, "pickup", relay.source)
        state.timer_s += max(0.0, dt)
        if relay.dwell_time_s == 0.0 or state.timer_s >= relay.dwell_time_s:
            state.tripped = True
            state.trip_time_s = time_s
            trip = TripRecord(relay.asset_id, time_s, "dwell_trip", relay.source)
    else:
        if relay.reset_rate_s_per_s is None:
            state.timer_s = 0.0
            state.picked_up = False
        else:
            state.timer_s = max(0.0, state.timer_s - relay.reset_rate_s_per_s * max(0.0, dt))
            if state.timer_s == 0.0:
                state.picked_up = False

    return pickup, trip


def _records_from_prediction(
    value: CascadeResult | ValidationResult | ProtectionReplayResult | Sequence[TripRecord | str],
    *,
    default_source: str,
) -> tuple[TripRecord, ...]:
    if isinstance(value, CascadeResult):
        return tuple(
            TripRecord(asset_id=item, time_s=float(index), reason="cascade_fixed_point", source=default_source)
            for index, item in enumerate(value.fixed_point)
        )
    if isinstance(value, ValidationResult):
        return value.simulated_trips if default_source == "nonlinear" else value.predicted_trips
    if isinstance(value, ProtectionReplayResult):
        return value.trip_records
    records: list[TripRecord] = []
    for index, item in enumerate(value):
        if isinstance(item, TripRecord):
            records.append(item)
        else:
            records.append(TripRecord(str(item), float(index), "listed_trip", default_source))
    return tuple(records)


def _as_vector(value: Sequence[float] | np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float).reshape(-1)
    if not np.all(np.isfinite(array)):
        raise NonlinearValidationError(f"{name} contains non-finite values")
    return array


def _as_sparse_or_dense(matrix: Any, name: str) -> Any:
    if sparse.issparse(matrix):
        return matrix.tocsc()
    array = np.asarray(matrix, dtype=float)
    if array.ndim != 2 or array.shape[0] != array.shape[1]:
        raise NonlinearValidationError(f"{name} must be a square matrix")
    if not np.all(np.isfinite(array)):
        raise NonlinearValidationError(f"{name} contains non-finite values")
    return array


def _matvec(matrix: Any, vector: np.ndarray) -> np.ndarray:
    result = matrix @ vector
    return np.asarray(result, dtype=float).reshape(-1)


def _solve_linear(matrix: Any, rhs: np.ndarray) -> np.ndarray:
    if sparse.issparse(matrix):
        csc = matrix.tocsc()
        if csc.shape[0] != csc.shape[1] or csc.shape[0] != rhs.size:
            raise NonlinearValidationError("sparse Jacobian shape does not match residual")
        return np.asarray(spla.spsolve(csc, rhs), dtype=float).reshape(-1)
    array = np.asarray(matrix, dtype=float)
    if array.ndim != 2 or array.shape[0] != array.shape[1] or array.shape[0] != rhs.size:
        raise NonlinearValidationError("Jacobian shape does not match residual")
    return np.linalg.solve(array, rhs)


def _andes_array_values(value: Any) -> list[float]:
    raw = getattr(value, "v", value)
    if raw is None:
        return []
    if isinstance(raw, np.ndarray):
        return raw.astype(float).reshape(-1).tolist()
    try:
        return [float(item) for item in raw]
    except TypeError:
        return [float(raw)]


def _andes_validation_models(system: Any) -> Mapping[str, Any]:
    exist = getattr(system, "exist", None)
    for name in ("pflow_tds", "tds", "pflow"):
        models = getattr(exist, name, None) if exist is not None else None
        if models:
            return models
    models = getattr(system, "models", None)
    if isinstance(models, Mapping) and models:
        return models
    raise NonlinearValidationError("ANDES system has no model set for residual evaluation")


def _set_andes_xy(
    system: Any,
    x_frozen: np.ndarray | None,
    y: np.ndarray,
) -> None:
    dae = getattr(system, "dae", None)
    if dae is None:
        raise NonlinearValidationError("ANDES system has no dae object")
    if x_frozen is not None and hasattr(dae, "x"):
        dae.x[:] = x_frozen
    dae.y[:] = y
    if hasattr(system, "vars_to_models"):
        system.vars_to_models()


def _andes_algebraic_residual(
    system: Any,
    models: Mapping[str, Any],
    x_frozen: np.ndarray | None,
    y: np.ndarray,
) -> np.ndarray:
    _set_andes_xy(system, x_frozen, y)
    dae = system.dae
    dae.clear_fg()
    if hasattr(system, "s_update_var"):
        system.s_update_var(models=models)
    if hasattr(system, "l_update_var"):
        system.l_update_var(models=models, niter=0, err=0.0)
    if hasattr(system, "f_update"):
        system.f_update(models=models)
    if hasattr(system, "l_update_eq"):
        system.l_update_eq(models=models, init=False, niter=0)
    system.g_update(models=models)
    system.fg_to_dae()
    return np.asarray(dae.g, dtype=float).reshape(-1).copy()


def _andes_algebraic_jacobian(
    case: AndesCase,
    system: Any,
    models: Mapping[str, Any],
    x_frozen: np.ndarray | None,
    y: np.ndarray,
) -> sparse.csc_matrix:
    _set_andes_xy(system, x_frozen, y)
    system.j_update(models=models, info="phase12 frozen algebraic validation")
    return case.jacobians().gy


def _restore_andes_xy(
    system: Any,
    y: np.ndarray,
    x: np.ndarray | None,
) -> None:
    _set_andes_xy(system, x, y)


def _read_relay_voltage(system: Any, relay: ProtectionRelaySpec) -> float:
    if relay.source_address is not None:
        dae = getattr(system, "dae", None)
        if dae is None:
            raise NonlinearValidationError("relay source_address requires system.dae")
        values = _andes_array_values(getattr(dae, "y", []))
        if relay.source_address >= len(values):
            raise NonlinearValidationError("relay source_address is outside system.dae.y")
        return float(values[relay.source_address])

    if relay.source_bus_id is None:
        raise NonlinearValidationError("relay has no readable source")
    bus = _system_model(system, "Bus")
    position = _find_device_position(bus, relay.source_bus_id)
    voltage = getattr(bus, "v", None)
    values = getattr(voltage, "v", None)
    if values is None:
        raise NonlinearValidationError("Bus.v.v is unavailable")
    return float(values[position])


def _system_model(system: Any, model_name: str) -> Any:
    models = getattr(system, "models", None)
    if isinstance(models, Mapping) and model_name in models:
        return models[model_name]
    if hasattr(system, model_name):
        return getattr(system, model_name)
    raise NonlinearValidationError(f"system model {model_name!r} is unavailable")


def _find_device_position(model: Any, device_id: str) -> int:
    idx = getattr(model, "idx", None)
    values = getattr(idx, "v", None)
    if values is None:
        raise NonlinearValidationError("model has no idx.v device list")
    requested = str(device_id)
    for position, value in enumerate(values):
        if value == device_id or str(value) == requested:
            return position
    raise NonlinearValidationError(f"device {device_id!r} not found")


def _duplicates(values: Sequence[str] | Any) -> tuple[str, ...]:
    seen: set[str] = set()
    duplicated: set[str] = set()
    for value in values:
        text = str(value)
        if text in seen:
            duplicated.add(text)
        seen.add(text)
    return tuple(sorted(duplicated))
