"""Benchmark-study orchestration for the protection-aware DVSA workflow.

Benchmark assumptions live in explicit configuration objects.  This layer wires
the numerical screen into replica studies, prediction overlays, ablations,
large-system screening plans, and scalability summaries.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import json
import math
from pathlib import Path
import time
import tracemalloc
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .andes_adapter import AndesCaseSpec, available_benchmarks
from .cascade_certificate import CascadeCertificateResult, compute_cascade_certificate
from .data_model import AssessmentStatus, CascadeResult, Matrix, ValidationResult, WindowMapResult
from .finite_window import FiniteWindowMapsResult
from .nonlinear_validation import (
    NonlinearValidationReport,
    ProtectionReplayResult,
    compare_screen_to_nonlinear,
)
from .robust_screening import RobustScreenResult


class BenchmarkStudyError(RuntimeError):
    """Raised when a benchmark study cannot be formed consistently."""


@dataclass(frozen=True, slots=True)
class BenchmarkTarget:
    """One ANDES benchmark target and its initialization policy."""

    case_id: str
    case_alias: str
    description: str
    category: str
    setup: bool = True
    run_pflow: bool = True
    init_tds: bool = False
    required: bool = True
    reliability: str = "required"
    notes: str | None = None

    def __post_init__(self) -> None:
        _nonempty(self.case_id, "case_id")
        _nonempty(self.case_alias, "case_alias")
        _nonempty(self.description, "description")
        _nonempty(self.category, "category")

    def to_case_spec(self) -> AndesCaseSpec:
        return AndesCaseSpec(
            self.case_alias,
            setup=self.setup,
            run_pflow=self.run_pflow,
            init_tds=self.init_tds,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "case_alias": self.case_alias,
            "description": self.description,
            "category": self.category,
            "setup": self.setup,
            "run_pflow": self.run_pflow,
            "init_tds": self.init_tds,
            "required": self.required,
            "reliability": self.reliability,
            "notes": self.notes,
        }


@dataclass(frozen=True, slots=True)
class ReplicaCollectorConfig:
    """IEEE-39 collector-side protected asset used by the replica study."""

    asset_id: str
    transmission_bus: int
    collector_bus: int
    tap_ratio: float
    base_kv: float
    threshold_pu: float
    dwell_time_s: float
    q_absorption_mvar: float
    source_name: str
    shunt_name: str

    def __post_init__(self) -> None:
        _nonempty(self.asset_id, "asset_id")
        _positive(self.tap_ratio, "tap_ratio")
        _positive(self.base_kv, "base_kv")
        _positive(self.threshold_pu, "threshold_pu")
        _nonnegative(self.dwell_time_s, "dwell_time_s")
        _nonnegative(self.q_absorption_mvar, "q_absorption_mvar")
        if self.transmission_bus <= 0 or self.collector_bus <= 0:
            raise BenchmarkStudyError("collector bus IDs must be positive")

    def to_dict(self) -> dict[str, Any]:
        return {
            "asset_id": self.asset_id,
            "transmission_bus": self.transmission_bus,
            "collector_bus": self.collector_bus,
            "tap_ratio": self.tap_ratio,
            "base_kv": self.base_kv,
            "threshold_pu": self.threshold_pu,
            "dwell_time_s": self.dwell_time_s,
            "q_absorption_mvar": self.q_absorption_mvar,
            "source_name": self.source_name,
            "shunt_name": self.shunt_name,
        }


@dataclass(frozen=True, slots=True)
class OperatorActionConfig:
    """Official-report-constrained operator/action schedule entry."""

    action_id: str
    time_s: float
    kind: str
    description: str
    enabled: bool = True
    expected_voltage_raising_mvar: float = 0.0
    category: str = "operator_action"

    def __post_init__(self) -> None:
        _nonempty(self.action_id, "action_id")
        _nonnegative(self.time_s, "time_s")
        _nonempty(self.kind, "kind")
        _nonempty(self.description, "description")
        if not math.isfinite(float(self.expected_voltage_raising_mvar)):
            raise BenchmarkStudyError("expected_voltage_raising_mvar must be finite")

    def to_dict(self) -> dict[str, Any]:
        return {
            "action_id": self.action_id,
            "time_s": self.time_s,
            "kind": self.kind,
            "description": self.description,
            "enabled": self.enabled,
            "expected_voltage_raising_mvar": self.expected_voltage_raising_mvar,
            "category": self.category,
        }


@dataclass(frozen=True, slots=True)
class BenchmarkStudyConfig:
    """Complete benchmark-study configuration."""

    study_id: str
    target: BenchmarkTarget
    collectors: tuple[ReplicaCollectorConfig, ...] = ()
    operator_actions: tuple[OperatorActionConfig, ...] = ()
    scenario_parameters: Mapping[str, Any] | None = None
    feature_flags: Mapping[str, bool] | None = None
    protected_asset_ids: tuple[str, ...] = ()
    seed_ids: tuple[str, ...] = ()
    notes: str | None = None

    def __post_init__(self) -> None:
        _nonempty(self.study_id, "study_id")
        object.__setattr__(self, "collectors", tuple(self.collectors))
        object.__setattr__(
            self,
            "operator_actions",
            tuple(sorted(self.operator_actions, key=lambda item: (item.time_s, item.action_id))),
        )
        protected = self.protected_asset_ids or tuple(item.asset_id for item in self.collectors)
        seeds = self.seed_ids or protected
        object.__setattr__(self, "protected_asset_ids", tuple(protected))
        object.__setattr__(self, "seed_ids", tuple(seeds))
        object.__setattr__(self, "scenario_parameters", dict(self.scenario_parameters or {}))
        object.__setattr__(self, "feature_flags", dict(self.feature_flags or {}))

    def to_dict(self) -> dict[str, Any]:
        return {
            "study_id": self.study_id,
            "target": self.target.to_dict(),
            "collectors": [item.to_dict() for item in self.collectors],
            "operator_actions": [item.to_dict() for item in self.operator_actions],
            "scenario_parameters": dict(self.scenario_parameters or {}),
            "feature_flags": dict(self.feature_flags or {}),
            "protected_asset_ids": list(self.protected_asset_ids),
            "seed_ids": list(self.seed_ids),
            "notes": self.notes,
        }


@dataclass(frozen=True, slots=True)
class CausalAblation:
    """One causal ablation to run against a benchmark configuration."""

    ablation_id: str
    description: str
    parameter_overrides: Mapping[str, Any] | None = None
    feature_overrides: Mapping[str, bool] | None = None
    expected_comparison: str = "compare against baseline prediction and nonlinear trip overlay"

    def __post_init__(self) -> None:
        _nonempty(self.ablation_id, "ablation_id")
        _nonempty(self.description, "description")
        object.__setattr__(self, "parameter_overrides", dict(self.parameter_overrides or {}))
        object.__setattr__(self, "feature_overrides", dict(self.feature_overrides or {}))

    def to_dict(self) -> dict[str, Any]:
        return {
            "ablation_id": self.ablation_id,
            "description": self.description,
            "parameter_overrides": dict(self.parameter_overrides or {}),
            "feature_overrides": dict(self.feature_overrides or {}),
            "expected_comparison": self.expected_comparison,
        }


@dataclass(frozen=True, slots=True)
class PredictionOverlayRecord:
    """One asset row in the prediction/nonlinear overlay."""

    asset_id: str
    predicted_layer: int | None
    predicted_trip_order: int | None
    simulated_pickup_time_s: float | None
    simulated_trip_time_s: float | None
    max_voltage_pu: float | None
    false_positive: bool
    false_negative: bool
    data_limited: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "asset_id": self.asset_id,
            "predicted_layer": self.predicted_layer,
            "predicted_trip_order": self.predicted_trip_order,
            "simulated_pickup_time_s": self.simulated_pickup_time_s,
            "simulated_trip_time_s": self.simulated_trip_time_s,
            "max_voltage_pu": self.max_voltage_pu,
            "false_positive": self.false_positive,
            "false_negative": self.false_negative,
            "data_limited": self.data_limited,
        }


@dataclass(frozen=True, slots=True)
class PredictionStudyResult:
    """K-matrices, cascade layers, and nonlinear trip-sequence overlay."""

    study_id: str
    mode_id: str
    protected_asset_ids: tuple[str, ...]
    event_ids: tuple[str, ...]
    k_pk: Matrix
    k_tr: Matrix
    cascade_result: CascadeResult
    validation_report: NonlinearValidationReport | None
    overlay: tuple[PredictionOverlayRecord, ...]
    status: AssessmentStatus
    metadata: Mapping[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "study_id": self.study_id,
            "mode_id": self.mode_id,
            "protected_asset_ids": list(self.protected_asset_ids),
            "event_ids": list(self.event_ids),
            "k_pk": [list(row) for row in self.k_pk],
            "k_tr": [list(row) for row in self.k_tr],
            "cascade_result": self.cascade_result.to_dict(),
            "validation_report": None
            if self.validation_report is None
            else self.validation_report.to_dict(),
            "overlay": [item.to_dict() for item in self.overlay],
            "status": self.status.value,
            "metadata": dict(self.metadata or {}),
        }


@dataclass(frozen=True, slots=True)
class LargeSystemTargetStatus:
    """Availability and initialization policy for one large benchmark."""

    target: BenchmarkTarget
    available: bool
    selected: bool
    reason: str
    max_top_k: int = 25

    def to_dict(self) -> dict[str, Any]:
        return {
            "target": self.target.to_dict(),
            "available": self.available,
            "selected": self.selected,
            "reason": self.reason,
            "max_top_k": self.max_top_k,
        }


@dataclass(frozen=True, slots=True)
class LargeSystemScreeningPlan:
    """Plan for NPCC, ACTIVSg2000, and GBnetwork large-system screening."""

    plan_id: str
    targets: tuple[LargeSystemTargetStatus, ...]
    notes: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "targets": [item.to_dict() for item in self.targets],
            "notes": self.notes,
        }


@dataclass(frozen=True, slots=True)
class TopKRankingQuality:
    """Observed quality of a screening ranking against nonlinear trip order."""

    k: int
    precision_at_k: float
    recall_at_k: float
    hits: tuple[str, ...]
    predicted_top_k: tuple[str, ...]
    observed_top_k: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "k": self.k,
            "precision_at_k": self.precision_at_k,
            "recall_at_k": self.recall_at_k,
            "hits": list(self.hits),
            "predicted_top_k": list(self.predicted_top_k),
            "observed_top_k": list(self.observed_top_k),
        }


@dataclass(frozen=True, slots=True)
class ScalabilityMetrics:
    """Runtime, sparse-solve, channel, memory, and ranking-quality metrics."""

    study_id: str
    case_id: str
    runtime_s: float
    sparse_solve_count: int
    selected_channels: int
    protected_asset_count: int
    event_count: int
    control_count: int
    memory_peak_mb: float | None = None
    top_k_quality: TopKRankingQuality | None = None
    status: AssessmentStatus = AssessmentStatus.NOT_EVALUATED
    metadata: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        _nonempty(self.study_id, "study_id")
        _nonempty(self.case_id, "case_id")
        _nonnegative(self.runtime_s, "runtime_s")
        if self.sparse_solve_count < 0 or self.selected_channels < 0:
            raise BenchmarkStudyError("solve/channel counts must be nonnegative")
        if min(self.protected_asset_count, self.event_count, self.control_count) < 0:
            raise BenchmarkStudyError("asset/event/control counts must be nonnegative")
        if self.memory_peak_mb is not None:
            _nonnegative(self.memory_peak_mb, "memory_peak_mb")
        object.__setattr__(self, "metadata", dict(self.metadata or {}))

    def to_dict(self) -> dict[str, Any]:
        return {
            "study_id": self.study_id,
            "case_id": self.case_id,
            "runtime_s": self.runtime_s,
            "sparse_solve_count": self.sparse_solve_count,
            "selected_channels": self.selected_channels,
            "protected_asset_count": self.protected_asset_count,
            "event_count": self.event_count,
            "control_count": self.control_count,
            "memory_peak_mb": self.memory_peak_mb,
            "top_k_quality": None if self.top_k_quality is None else self.top_k_quality.to_dict(),
            "status": self.status.value,
            "metadata": dict(self.metadata or {}),
        }


def build_ieee39_official_report_replica_config() -> BenchmarkStudyConfig:
    """Return the IEEE-39 official-report-constrained replica configuration."""

    target = BenchmarkTarget(
        case_id="ieee39_official_report_replica",
        case_alias="ieee39",
        description="IEEE-39 official-report-constrained Iberian blackout replica",
        category="replica",
        setup=True,
        run_pflow=True,
        init_tds=True,
        required=True,
        reliability="required",
    )
    trans_buses = (21, 22, 16, 19, 3)
    taps = (0.992, 0.995, 0.989, 0.999, 0.999)
    thresholds = (1.08, 1.12, 1.12, 1.12, 1.13)
    q_abs = (78.0, 72.0, 66.0, 60.0, 90.0)
    dwell_offsets = (0.05, 0.25, 0.20, 0.01, 0.10)
    base_dwell = 0.01
    collectors = tuple(
        ReplicaCollectorConfig(
            asset_id=f"C{i + 1}",
            transmission_bus=trans_buses[i],
            collector_bus=40 + i,
            tap_ratio=taps[i],
            base_kv=138.0,
            threshold_pu=thresholds[i],
            dwell_time_s=base_dwell + dwell_offsets[i],
            q_absorption_mvar=q_abs[i],
            source_name=f"IBR_{i + 1}",
            shunt_name=f"COLL_SHUNT_{i + 1}",
        )
        for i in range(5)
    )
    actions = (
        OperatorActionConfig("OA1_mesh_1", 4.5, "line_energization", "Parallel line energized 1/4", True, 35.0),
        OperatorActionConfig("OA1_mesh_2", 5.0, "line_energization", "Parallel line energized 2/4", True, 35.0),
        OperatorActionConfig("OA2_export_reduction", 5.8, "exchange_reduction", "Exports reduced", True, 10.0),
        OperatorActionConfig("OA3_reactor_1", 6.5, "shunt_reactor_open", "Shunt reactor opened 1/2", True, 130.0),
        OperatorActionConfig("OA1_mesh_3", 7.0, "line_energization", "Parallel line energized 3/4", True, 35.0),
        OperatorActionConfig("OA4_hvdc_mode", 8.5, "hvdc_mode_change", "HVDC switched to fixed-power mode", True, 0.0),
        OperatorActionConfig("OA1_mesh_4", 9.0, "line_energization", "Parallel line energized 4/4", True, 35.0),
        OperatorActionConfig("OA3_reactor_2", 10.0, "shunt_reactor_open", "Shunt reactor opened 2/2", True, 130.0),
    )
    return BenchmarkStudyConfig(
        study_id="ieee39_official_report_replica",
        target=target,
        collectors=collectors,
        operator_actions=actions,
        scenario_parameters={
            "sim_tf_s": 18.0,
            "tstep_s": 0.01,
            "nominal_frequency_hz": 50.0,
            "load_scale": 0.8,
            "ehv_voltage_target_pu": 1.0,
            "collector_enable_time_s": 0.0,
            "collector_base_kv": 138.0,
            "collector_base_dwell_s": base_dwell,
            "collector_dwell_offsets_s": dwell_offsets,
            "pv_unit_count": 5,
            "pv_total_p_pu": 4.0,
            "hvdc_bus": 28,
            "hvdc_export_pu": 0.95,
            "hvdc_fixed_pu": 1.20,
            "ufls_trigger_hz": 49.4,
            "collapse_freq_hz": 48.3,
            "collapse_voltage_pu": 0.65,
        },
        feature_flags={
            "protection": True,
            "meshing": True,
            "reactor_switching": True,
            "hvdc_mode_change": True,
            "export_reduction": True,
            "ufls": True,
            "collector_q_absorption_trip": True,
        },
        notes="Collector-side protection study with fixed taps and report-style OA/AA categories.",
    )


def build_causal_ablation_suite(
    base: BenchmarkStudyConfig | None = None,
) -> tuple[CausalAblation, ...]:
    """Return the causal ablations required for benchmark studies."""

    _ = base
    return (
        CausalAblation(
            "no_protection",
            "Disable collector relay trips while preserving operator actions.",
            feature_overrides={"protection": False},
        ),
        CausalAblation(
            "delayed_protection",
            "Increase relay dwell times to test whether sequence timing depends on fast protection.",
            parameter_overrides={"collector_dwell_multiplier": 10.0},
        ),
        CausalAblation(
            "fixed_pf_vs_voltage_mode",
            "Compare fixed-power-factor IBR behavior against voltage-control behavior.",
            parameter_overrides={"ibr_mode": "fixed_power_factor", "comparison_mode": "voltage_control"},
        ),
        CausalAblation(
            "preserved_q_absorption",
            "Trip active sources while preserving collector reactive absorption.",
            feature_overrides={"collector_q_absorption_trip": False},
        ),
        CausalAblation(
            "shunt_statcom_hvdc_support",
            "Enable fast shunt/STATCOM/HVDC voltage support envelopes before relay dwell expires.",
            feature_overrides={"fast_shunt_support": True, "statcom_support": True, "hvdc_reactive_support": True},
        ),
        CausalAblation(
            "uel_active",
            "Treat under-excitation limiters as active and reduce available absorption.",
            feature_overrides={"uel_active": True},
        ),
        CausalAblation(
            "uel_inactive",
            "Treat under-excitation limiters as inactive for the same operating point.",
            feature_overrides={"uel_active": False},
        ),
        CausalAblation(
            "load_shedding_without_mvar_replacement",
            "Shed load without replacing the lost reactive demand.",
            feature_overrides={"load_shedding": True, "mvar_replacement": False},
        ),
        CausalAblation(
            "load_shedding_with_mvar_replacement",
            "Shed load while adding MVAr replacement so active relief is separated from reactive balance.",
            feature_overrides={"load_shedding": True, "mvar_replacement": True},
        ),
    )


def apply_ablation(
    config: BenchmarkStudyConfig,
    ablation: CausalAblation,
) -> BenchmarkStudyConfig:
    """Return a derived benchmark config with ablation overrides applied."""

    parameters = {**dict(config.scenario_parameters or {}), **dict(ablation.parameter_overrides or {})}
    flags = {**dict(config.feature_flags or {}), **dict(ablation.feature_overrides or {})}
    return replace(
        config,
        study_id=f"{config.study_id}__{ablation.ablation_id}",
        scenario_parameters=parameters,
        feature_flags=flags,
        notes=f"{config.notes or ''} Ablation: {ablation.description}".strip(),
    )


def build_prediction_study(
    window_source: WindowMapResult | FiniteWindowMapsResult | RobustScreenResult,
    *,
    study_id: str,
    seed_ids: Sequence[str] = (),
    cascade: CascadeResult | CascadeCertificateResult | None = None,
    nonlinear: ProtectionReplayResult | ValidationResult | NonlinearValidationReport | None = None,
    robust_validation: bool = True,
    metadata: Mapping[str, Any] | None = None,
) -> PredictionStudyResult:
    """Build a prediction study with ``K_pk``, ``K_tr``, layers, and overlay."""

    window = _unwrap_window(window_source)
    if cascade is None:
        cascade_certificate = compute_cascade_certificate(window, seed_ids=seed_ids or window.event_ids)
        cascade_result = cascade_certificate.cascade_result
        if cascade_result is None:
            raise BenchmarkStudyError("cascade certificate returned no seed result")
    elif isinstance(cascade, CascadeCertificateResult):
        cascade_result = cascade.cascade_result
        if cascade_result is None:
            raise BenchmarkStudyError("cascade certificate has no cascade_result")
    else:
        cascade_result = cascade

    report: NonlinearValidationReport | None
    if isinstance(nonlinear, NonlinearValidationReport):
        report = nonlinear
    elif nonlinear is None:
        report = None
    else:
        report = compare_screen_to_nonlinear(
            scenario_id=study_id,
            mode_id=window.mode_id,
            seed_ids=cascade_result.seed_ids,
            predicted=cascade_result,
            simulated=nonlinear,
            robust=robust_validation,
        )

    overlay = _prediction_overlay(window, cascade_result, report, nonlinear)
    status = _prediction_status(window, cascade_result, report)
    return PredictionStudyResult(
        study_id=study_id,
        mode_id=window.mode_id,
        protected_asset_ids=tuple(window.protected_asset_ids),
        event_ids=tuple(window.event_ids),
        k_pk=tuple(tuple(float(item) for item in row) for row in window.k_pickup_upper),
        k_tr=tuple(tuple(float(item) for item in row) for row in window.k_trip_upper),
        cascade_result=cascade_result,
        validation_report=report,
        overlay=overlay,
        status=status,
        metadata=dict(metadata or {}),
    )


def build_large_system_screening_plan(
    *,
    availability: Mapping[str, bool] | None = None,
    project_root: str | Path = ".",
) -> LargeSystemScreeningPlan:
    """Plan large-system screening for NPCC, ACTIVSg2000, and GBnetwork."""

    targets = _large_system_targets()
    if availability is None:
        try:
            resolved = available_benchmarks(project_root)
            availability_map = {key: value.exists for key, value in resolved.items()}
        except Exception:
            availability_map = {}
    else:
        availability_map = {str(key): bool(value) for key, value in availability.items()}

    statuses: list[LargeSystemTargetStatus] = []
    for target in targets:
        available = bool(availability_map.get(target.case_alias, False))
        selected = available and target.reliability != "opportunistic_only"
        if available:
            reason = "available; include in screening run"
        elif target.required:
            reason = "required benchmark is not available or failed resolution"
        else:
            reason = "optional benchmark skipped unless initialization is reliable"
        statuses.append(
            LargeSystemTargetStatus(
                target=target,
                available=available,
                selected=selected,
                reason=reason,
                max_top_k=50 if target.case_alias.startswith("activsg") else 25,
            )
        )
    return LargeSystemScreeningPlan(
        plan_id="large_system_screening",
        targets=tuple(statuses),
        notes="Run NPCC and GBnetwork as reliability checks; run ACTIVSg2000 when initialization is stable.",
    )


def ranking_quality(
    predicted_scores: Mapping[str, float] | Sequence[float],
    observed_order: Sequence[str],
    *,
    item_ids: Sequence[str] | None = None,
    k: int = 10,
) -> TopKRankingQuality:
    """Compute precision/recall@k for top-k screen ranking quality."""

    if k <= 0:
        raise BenchmarkStudyError("k must be positive")
    if isinstance(predicted_scores, Mapping):
        scored = [(str(item), float(score)) for item, score in predicted_scores.items()]
    else:
        scores = [float(item) for item in predicted_scores]
        ids = tuple(item_ids or tuple(f"item_{idx}" for idx in range(len(scores))))
        if len(ids) != len(scores):
            raise BenchmarkStudyError("item_ids length must match predicted_scores")
        scored = list(zip(ids, scores))
    if any(not math.isfinite(score) for _, score in scored):
        raise BenchmarkStudyError("predicted scores must be finite")
    predicted_top = tuple(item for item, _ in sorted(scored, key=lambda pair: (-pair[1], pair[0]))[:k])
    observed_top = tuple(str(item) for item in observed_order[:k])
    observed_set = set(observed_top)
    hits = tuple(item for item in predicted_top if item in observed_set)
    precision = len(hits) / max(len(predicted_top), 1)
    recall = len(hits) / max(len(observed_top), 1)
    return TopKRankingQuality(
        k=k,
        precision_at_k=float(precision),
        recall_at_k=float(recall),
        hits=hits,
        predicted_top_k=predicted_top,
        observed_top_k=observed_top,
    )


def measure_execution(
    function: Callable[[], Any],
    *,
    study_id: str,
    case_id: str,
    sparse_solve_count: int = 0,
    selected_channels: int = 0,
    protected_asset_count: int = 0,
    event_count: int = 0,
    control_count: int = 0,
    top_k_quality: TopKRankingQuality | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> tuple[Any, ScalabilityMetrics]:
    """Run ``function`` and return its result with scalability metrics."""

    tracemalloc.start()
    start = time.perf_counter()
    status = AssessmentStatus.CERTIFIED
    try:
        result = function()
    except Exception:
        status = AssessmentStatus.FAILED
        raise
    finally:
        runtime = time.perf_counter() - start
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
    metrics = ScalabilityMetrics(
        study_id=study_id,
        case_id=case_id,
        runtime_s=float(runtime),
        sparse_solve_count=int(sparse_solve_count),
        selected_channels=int(selected_channels),
        protected_asset_count=int(protected_asset_count),
        event_count=int(event_count),
        control_count=int(control_count),
        memory_peak_mb=float(peak) / 1_000_000.0,
        top_k_quality=top_k_quality,
        status=status,
        metadata=dict(metadata or {}),
    )
    return result, metrics


def write_benchmark_config(config: BenchmarkStudyConfig, path: str | Path) -> Path:
    """Write a benchmark configuration JSON file."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(config.to_dict(), indent=2), encoding="utf-8")
    return output


def legacy_blackout_scenario_kwargs(config: BenchmarkStudyConfig) -> dict[str, Any]:
    """Return kwargs compatible with ``blackout.ScenarioConfig`` where possible."""

    if not config.collectors:
        raise BenchmarkStudyError("legacy blackout kwargs require collector configs")
    params = dict(config.scenario_parameters or {})
    flags = dict(config.feature_flags or {})
    return {
        "sim_tf": params.get("sim_tf_s", 18.0),
        "tstep": params.get("tstep_s", 0.01),
        "fnom": params.get("nominal_frequency_hz", 50.0),
        "load_scale": params.get("load_scale", 0.8),
        "ehv_voltage_target": params.get("ehv_voltage_target_pu", 1.0),
        "collector_threshold_pu": max(item.threshold_pu for item in config.collectors),
        "collector_dwell_s": params.get("collector_base_dwell_s", min(item.dwell_time_s for item in config.collectors)),
        "collector_dwell_offsets": tuple(params.get("collector_dwell_offsets_s", ())),
        "collector_base_kv": params.get("collector_base_kv", config.collectors[0].base_kv),
        "collector_taps": tuple(item.tap_ratio for item in config.collectors),
        "collector_q_mvar": tuple(item.q_absorption_mvar for item in config.collectors),
        "collector_trans_buses": tuple(item.transmission_bus for item in config.collectors),
        "collector_thresholds": tuple(item.threshold_pu for item in config.collectors),
        "enable_meshing": flags.get("meshing", True),
        "enable_reactor_switch": flags.get("reactor_switching", True),
        "enable_hvdc_mode_change": flags.get("hvdc_mode_change", True),
        "enable_export_reduction": flags.get("export_reduction", True),
        "enable_ufls": flags.get("ufls", True),
    }


def _prediction_overlay(
    window: WindowMapResult,
    cascade: CascadeResult,
    report: NonlinearValidationReport | None,
    nonlinear: ProtectionReplayResult | ValidationResult | NonlinearValidationReport | None,
) -> tuple[PredictionOverlayRecord, ...]:
    predicted_layer: dict[str, int] = {}
    for layer in cascade.layers:
        for asset in layer.cumulative_set:
            predicted_layer.setdefault(asset, layer.layer_index)
    predicted_order = {asset: idx for idx, asset in enumerate(cascade.fixed_point)}

    pickup_times: dict[str, float] = {}
    trip_times: dict[str, float] = {}
    max_voltages: dict[str, float] = {}
    false_positive: set[str] = set()
    false_negative: set[str] = set()
    data_limited: set[str] = set(cascade.data_limited_assets)

    if report is not None:
        validation = report.validation_result
        false_positive.update(validation.false_positive_trips)
        false_negative.update(validation.false_negative_trips)
        data_limited.update(report.data_limited_assets)
        for item in validation.simulated_trips:
            trip_times[item.asset_id] = item.time_s
        if report.first_simulated_pickup is not None:
            pickup_times[report.first_simulated_pickup.asset_id] = report.first_simulated_pickup.time_s
        for item in validation.max_excursions:
            max_voltages[item.protected_asset_id] = item.max_voltage_pu

    replay = nonlinear if isinstance(nonlinear, ProtectionReplayResult) else None
    if isinstance(nonlinear, NonlinearValidationReport):
        data_limited.update(nonlinear.data_limited_assets)
    if replay is not None:
        for item in replay.pickup_records:
            pickup_times.setdefault(item.asset_id, item.time_s)
        for item in replay.trip_records:
            trip_times.setdefault(item.asset_id, item.time_s)
        for item in replay.max_excursions:
            max_voltages[item.protected_asset_id] = item.max_voltage_pu

    assets = tuple(dict.fromkeys(tuple(window.protected_asset_ids) + tuple(cascade.fixed_point) + tuple(trip_times)))
    return tuple(
        PredictionOverlayRecord(
            asset_id=asset,
            predicted_layer=predicted_layer.get(asset),
            predicted_trip_order=predicted_order.get(asset),
            simulated_pickup_time_s=pickup_times.get(asset),
            simulated_trip_time_s=trip_times.get(asset),
            max_voltage_pu=max_voltages.get(asset),
            false_positive=asset in false_positive,
            false_negative=asset in false_negative,
            data_limited=asset in data_limited,
        )
        for asset in assets
    )


def _prediction_status(
    window: WindowMapResult,
    cascade: CascadeResult,
    report: NonlinearValidationReport | None,
) -> AssessmentStatus:
    if report is not None:
        return report.validation_result.status
    if window.status == AssessmentStatus.FAILED or cascade.status == AssessmentStatus.FAILED:
        return AssessmentStatus.FAILED
    if window.status == AssessmentStatus.DATA_LIMITED or cascade.status == AssessmentStatus.DATA_LIMITED:
        return AssessmentStatus.DATA_LIMITED
    return AssessmentStatus.CERTIFIED


def _unwrap_window(
    source: WindowMapResult | FiniteWindowMapsResult | RobustScreenResult,
) -> WindowMapResult:
    if isinstance(source, WindowMapResult):
        return source
    if isinstance(source, FiniteWindowMapsResult):
        return source.window_result
    if isinstance(source, RobustScreenResult):
        return source.robust_window_result
    raise BenchmarkStudyError(f"unsupported window source {type(source).__name__}")


def _large_system_targets() -> tuple[BenchmarkTarget, ...]:
    return (
        BenchmarkTarget("npcc_full", "npcc_full", "NPCC RAW/DYR dynamic benchmark", "large_system", init_tds=True),
        BenchmarkTarget("npcc_static", "npcc", "NPCC static screening fallback", "large_system", init_tds=False, required=False),
        BenchmarkTarget("activsg2000", "activsg2000_stable", "ACTIVSg2000 trimmed stable RAW/DYR benchmark", "large_system", init_tds=True, required=False),
        BenchmarkTarget("gbnetwork", "gbnetwork", "GBnetwork benchmark", "large_system", init_tds=False, required=False),
    )


def _nonempty(value: str, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise BenchmarkStudyError(f"{name} must be non-empty")


def _positive(value: float, name: str) -> None:
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise BenchmarkStudyError(f"{name} must be finite and positive")


def _nonnegative(value: float, name: str) -> None:
    number = float(value)
    if not math.isfinite(number) or number < 0.0:
        raise BenchmarkStudyError(f"{name} must be finite and nonnegative")
