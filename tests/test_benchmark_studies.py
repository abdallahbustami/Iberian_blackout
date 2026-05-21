from __future__ import annotations

from pa_dvsa.benchmark_studies import (
    apply_ablation,
    build_causal_ablation_suite,
    build_ieee39_official_report_replica_config,
    build_large_system_screening_plan,
    build_prediction_study,
    legacy_blackout_scenario_kwargs,
    measure_execution,
    ranking_quality,
)
from pa_dvsa.data_model import AssessmentStatus, WindowMapResult
from pa_dvsa.nonlinear_validation import (
    ProtectionRelaySpec,
    ProtectionTrace,
    replay_protection_traces,
)


def test_ieee39_replica_config_preserves_blackout_constants():
    config = build_ieee39_official_report_replica_config()

    assert config.target.case_alias == "ieee39"
    assert len(config.collectors) == 5
    assert tuple(item.transmission_bus for item in config.collectors) == (21, 22, 16, 19, 3)
    assert tuple(item.tap_ratio for item in config.collectors) == (0.992, 0.995, 0.989, 0.999, 0.999)
    assert tuple(item.q_absorption_mvar for item in config.collectors) == (78.0, 72.0, 66.0, 60.0, 90.0)
    assert [item.action_id for item in config.operator_actions][:2] == ["OA1_mesh_1", "OA1_mesh_2"]

    kwargs = legacy_blackout_scenario_kwargs(config)
    assert kwargs["collector_trans_buses"] == (21, 22, 16, 19, 3)
    assert kwargs["collector_dwell_s"] == 0.01
    assert kwargs["collector_dwell_offsets"] == (0.05, 0.25, 0.20, 0.01, 0.10)
    assert kwargs["enable_hvdc_mode_change"] is True


def test_causal_ablation_suite_has_required_experiments_and_applies_overrides():
    base = build_ieee39_official_report_replica_config()
    suite = build_causal_ablation_suite(base)
    ids = {item.ablation_id for item in suite}

    assert {
        "no_protection",
        "delayed_protection",
        "fixed_pf_vs_voltage_mode",
        "preserved_q_absorption",
        "shunt_statcom_hvdc_support",
        "uel_active",
        "uel_inactive",
        "load_shedding_without_mvar_replacement",
        "load_shedding_with_mvar_replacement",
    }.issubset(ids)

    no_protection = next(item for item in suite if item.ablation_id == "no_protection")
    derived = apply_ablation(base, no_protection)
    assert derived.study_id.endswith("__no_protection")
    assert derived.feature_flags["protection"] is False


def test_prediction_study_includes_k_matrices_layers_and_nonlinear_overlay():
    window = WindowMapResult(
        mode_id="mode",
        protected_asset_ids=("A", "B"),
        event_ids=("A", "B"),
        control_ids=(),
        horizons_s=(1.0, 1.0),
        k_pickup_upper=((0.0, 0.0), (1.2, 0.0)),
        k_trip_upper=((0.0, 0.0), (1.1, 0.0)),
        status=AssessmentStatus.CERTIFIED,
    )
    relay = ProtectionRelaySpec("B", 1.1, dwell_time_s=0.1, source_bus_id="2")
    replay = replay_protection_traces(
        [ProtectionTrace(relay, (0.0, 0.1, 0.2), (1.0, 1.12, 1.13))]
    )

    result = build_prediction_study(
        window,
        study_id="prediction",
        seed_ids=("A",),
        nonlinear=replay,
        robust_validation=True,
    )

    assert result.k_pk == ((0.0, 0.0), (1.2, 0.0))
    assert result.cascade_result.fixed_point == ("A", "B")
    row_b = next(item for item in result.overlay if item.asset_id == "B")
    assert row_b.predicted_layer == 1
    assert row_b.simulated_trip_time_s == 0.1
    assert not row_b.false_negative


def test_large_system_plan_marks_availability_and_selection():
    plan = build_large_system_screening_plan(
        availability={
            "npcc_full": True,
            "npcc": True,
            "activsg2000_stable": False,
            "gbnetwork": True,
        }
    )

    statuses = {item.target.case_alias: item for item in plan.targets}
    assert statuses["npcc_full"].selected
    assert statuses["gbnetwork"].selected
    assert not statuses["activsg2000_stable"].selected


def test_ranking_quality_and_scalability_metrics():
    quality = ranking_quality(
        {"A": 0.9, "B": 0.1, "C": 0.8},
        observed_order=("C", "A"),
        k=2,
    )

    assert quality.predicted_top_k == ("A", "C")
    assert quality.hits == ("A", "C")
    assert quality.precision_at_k == 1.0
    assert quality.recall_at_k == 1.0

    result, metrics = measure_execution(
        lambda: "ok",
        study_id="s",
        case_id="case",
        sparse_solve_count=3,
        selected_channels=4,
        protected_asset_count=2,
        event_count=2,
        control_count=1,
        top_k_quality=quality,
    )
    assert result == "ok"
    assert metrics.status == AssessmentStatus.CERTIFIED
    assert metrics.sparse_solve_count == 3
    assert metrics.memory_peak_mb is not None
