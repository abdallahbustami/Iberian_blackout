from __future__ import annotations

from types import SimpleNamespace

import numpy as np
from scipy import sparse

from pa_dvsa.data_model import AssessmentStatus, TripRecord
from pa_dvsa.nonlinear_validation import (
    ProtectionRelaySpec,
    ProtectionTrace,
    RelayTripTarget,
    TDSProtectionCallback,
    compare_screen_to_nonlinear,
    frozen_algebraic_nonlinear_validation,
    replay_protection_traces,
)


def test_frozen_algebraic_newton_uses_residual_and_jacobian_updates():
    result = frozen_algebraic_nonlinear_validation(
        y0=[0.5],
        residual_callback=lambda y: np.array([y[0] ** 2 - 1.0]),
        jacobian_update=lambda y: sparse.csc_matrix([[2.0 * y[0]]]),
        output_addresses=[0],
        tolerance=1.0e-11,
        max_iter=10,
    )

    assert result.converged
    assert result.status == AssessmentStatus.CERTIFIED
    assert result.iterations > 0
    assert abs(result.output_delta_y[0] - 0.5) < 1.0e-8
    assert result.final_residual_norm < 1.0e-10


def test_frozen_algebraic_linear_fallback_is_data_limited():
    result = frozen_algebraic_nonlinear_validation(
        y0=[1.0, 2.0],
        disturbance_g=[-2.0, 4.0],
        jacobian=sparse.csc_matrix([[2.0, 0.0], [0.0, 4.0]]),
        output_addresses=[0, 1],
    )

    assert result.converged
    assert result.status == AssessmentStatus.DATA_LIMITED
    assert result.output_delta_y == (1.0, -1.0)


def test_replay_protection_trace_records_pickup_trip_and_maximum():
    relay = ProtectionRelaySpec(
        asset_id="plant_A",
        threshold_pu=1.10,
        dwell_time_s=0.2,
        source_bus_id="1",
    )
    trace = ProtectionTrace(
        relay=relay,
        times_s=(0.0, 0.1, 0.2, 0.3),
        voltages_pu=(1.00, 1.12, 1.13, 1.11),
    )

    result = replay_protection_traces([trace])

    assert result.first_pickup is not None
    assert result.first_pickup.asset_id == "plant_A"
    assert result.first_pickup.time_s == 0.1
    assert result.first_trip is not None
    assert result.first_trip.time_s == 0.2
    assert result.max_excursions[0].max_voltage_pu == 1.13
    assert result.max_excursions[0].time_s == 0.2


def test_replay_protection_trace_resets_dwell_timer_on_dropout():
    relay = ProtectionRelaySpec(
        asset_id="plant_A",
        threshold_pu=1.10,
        dwell_time_s=0.2,
        source_bus_id="1",
    )
    trace = ProtectionTrace(
        relay=relay,
        times_s=(0.0, 0.1, 0.2, 0.3, 0.4),
        voltages_pu=(1.00, 1.12, 1.00, 1.13, 1.14),
    )

    result = replay_protection_traces([trace])

    assert result.first_trip is not None
    assert result.first_trip.time_s == 0.4


def test_tds_callback_trips_targets_and_calls_mode_update():
    system = _toy_system(bus_voltage=1.0)
    mode_updates: list[tuple[str, float]] = []
    relay = ProtectionRelaySpec(
        asset_id="load_relay",
        threshold_pu=1.10,
        dwell_time_s=0.1,
        source_bus_id="1",
        trip_targets=(RelayTripTarget("PQ", "L1", 0.0),),
    )
    callback = TDSProtectionCallback(
        [relay],
        mode_update_callback=lambda spec, t, _system: mode_updates.append((spec.asset_id, t)),
    )

    callback(0.0, system)
    system.Bus.v.v[0] = 1.12
    callback(0.1, system)

    assert callback.trip_records[0].asset_id == "load_relay"
    assert callback.trip_records[0].time_s == 0.1
    assert system.PQ.u.v[0] == 0.0
    assert mode_updates == [("load_relay", 0.1)]


def test_tds_callback_prefers_andes_set_status_when_available():
    system = _toy_system(bus_voltage=1.12)
    calls: list[tuple[str, str, float]] = []

    def set_status(model: str, device_id: str, value: float) -> None:
        calls.append((model, device_id, value))
        position = system.PQ.idx.v.index(device_id)
        system.PQ.u.v[position] = value

    system.set_status = set_status
    relay = ProtectionRelaySpec(
        asset_id="load_relay",
        threshold_pu=1.10,
        dwell_time_s=0.0,
        source_bus_id="1",
        trip_targets=(RelayTripTarget("PQ", "L1", 0.0),),
    )
    callback = TDSProtectionCallback([relay])

    callback(0.0, system)

    assert calls == [("PQ", "L1", 0.0)]
    assert system.PQ.u.v[0] == 0.0


def test_compare_robust_screen_accepts_data_limited_false_negative():
    simulated = [
        TripRecord("A", 0.2, "dwell_trip", "nonlinear"),
        TripRecord("B", 0.4, "dwell_trip", "nonlinear"),
    ]

    report = compare_screen_to_nonlinear(
        scenario_id="s1",
        mode_id="m1",
        seed_ids=("seed",),
        predicted=["A"],
        simulated=simulated,
        data_limited_assets=("B",),
        robust=True,
    )

    assert report.robust_success
    assert report.unsafe_false_negative_trips == ()
    assert report.validation_result.false_negative_trips == ("B",)
    assert report.validation_result.status == AssessmentStatus.DATA_LIMITED


def test_compare_robust_screen_fails_unsafe_false_negative():
    simulated = [
        TripRecord("A", 0.2, "dwell_trip", "nonlinear"),
        TripRecord("B", 0.4, "dwell_trip", "nonlinear"),
    ]

    report = compare_screen_to_nonlinear(
        scenario_id="s1",
        mode_id="m1",
        predicted=["A"],
        simulated=simulated,
        robust=True,
    )

    assert not report.robust_success
    assert report.unsafe_false_negative_trips == ("B",)
    assert report.validation_result.status == AssessmentStatus.FAILED


def _toy_system(bus_voltage: float):
    bus = SimpleNamespace(
        idx=SimpleNamespace(v=["1"]),
        v=SimpleNamespace(v=[bus_voltage]),
    )
    pq = SimpleNamespace(
        idx=SimpleNamespace(v=["L1"]),
        u=SimpleNamespace(v=[1.0]),
    )
    return SimpleNamespace(Bus=bus, PQ=pq, models={"Bus": bus, "PQ": pq})
