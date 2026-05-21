"""ACTIVSg2000 surrogate replication of the Iberian overvoltage cascade.

This script builds a final-report-constrained academic surrogate on the
``activsg2000_stable`` ANDES case.  It does not claim geographic equivalence to
Iberia.  The point is narrower and closer to the paper: on a large dynamic
benchmark, hidden collector-side overvoltage protection can trip plant blocks
after voltage-control actions remove reactive absorption, and each trip removes
more absorption.  The default calibration follows the final report's mechanism:
a Granada-like first protected transformer/collector trip while the upstream
transmission voltage is elevated but not yet catastrophic, followed by larger
Badajoz-like plant blocks and additional loss of reactive absorption.

Default run:

    PYTHONPATH=src python replication.py --plot

Outputs are written to ``results/activsg2000_iberian_replication`` and, when
``--plot`` is supplied, a figure is also copied into
``LaTeX/figures/generated/fig_replication_activsg2000.*``.
"""

from __future__ import annotations

import argparse
from collections import defaultdict, deque
from contextlib import redirect_stderr, redirect_stdout
import csv
from dataclasses import asdict, dataclass, field, replace
import io
import json
from pathlib import Path
import shutil
import sys
from typing import Any

import numpy as np

from pa_dvsa.andes_adapter import AndesCase, AndesCaseSpec


@dataclass(frozen=True, slots=True)
class OverVoltageRelayStage:
    name: str
    pickup_pu: float
    dwell_s: float
    reset_ratio: float = 0.95


@dataclass(frozen=True, slots=True)
class CollectorSpec:
    name: str
    transmission_bus: int
    tap: float
    threshold_pu: float
    dwell_s: float
    active_power_mw: float
    q_absorption_mvar: float
    fixed_pf_q_fraction: float = 0.35
    base_kv: float = 115.0
    fast_threshold_pu: float | None = None
    fast_dwell_s: float | None = None
    report_cluster: str = "E5 multi-site"
    report_time_s: float = 15.5
    min_trip_time_s: float = 0.0
    relay_stages: tuple[OverVoltageRelayStage, ...] = ()
    q_absorption_estimated: bool = False
    visible_trace: bool = False


@dataclass(slots=True)
class CollectorRuntime:
    spec: CollectorSpec
    collector_bus: int
    gsu_line_id: str
    pq_id: str
    main_q_shunt_id: str
    fixed_pf_q_shunt_id: str
    timer_s: float = 0.0
    fast_timer_s: float = 0.0
    tripped: bool = False
    trip_time_s: float | None = None
    trip_voltage_pu: float | None = None
    trip_stage_name: str | None = None
    last_voltage_pu: float = float("nan")
    last_transmission_voltage_pu: float = float("nan")
    max_voltage_pu: float = -float("inf")
    max_voltage_time_s: float = 0.0
    stage_timers_s: dict[str, float] = field(default_factory=dict)
    stage_picked_up: dict[str, bool] = field(default_factory=dict)

    @property
    def main_q_absorption_mvar(self) -> float:
        return self.spec.q_absorption_mvar * (1.0 - self.spec.fixed_pf_q_fraction)

    @property
    def fixed_pf_q_absorption_mvar(self) -> float:
        return self.spec.q_absorption_mvar * self.spec.fixed_pf_q_fraction


@dataclass(frozen=True, slots=True)
class OperatorAction:
    code: str
    time_s: float
    kind: str
    description: str


@dataclass(frozen=True, slots=True)
class ReactorSwitchAction:
    time_s: float
    bus: int
    q_mvar: float
    final_status: int
    description: str


@dataclass(frozen=True, slots=True)
class BlackoutDisturbanceAction:
    time_s: float
    bus: int
    p_mw: float
    q_mvar: float
    require_trip_count: int
    description: str


@dataclass(frozen=True, slots=True)
class HVDCSurrogateSpec:
    name: str = "IBERIAN_HVDC_SURROGATE"
    local_bus: int = 7406
    remote_bus: int = 7058
    transfer_mw: float = 550.0
    reactive_support_mvar: float = 420.0
    min_block_time_s: float = 21.0
    block_voltage_pu: float = 0.88
    block_over_voltage_pu: float = 1.135
    block_frequency_hz: float = 47.5
    dwell_s: float = 0.08
    require_collector_trips: int = 6
    require_france_ac_separation: bool = True


@dataclass(slots=True)
class HVDCSurrogateRuntime:
    spec: HVDCSurrogateSpec
    local_pq_id: str
    remote_pq_id: str
    timer_s: float = 0.0
    blocked: bool = False
    block_time_s: float | None = None


@dataclass(frozen=True, slots=True)
class OutOfStepRelaySpec:
    min_trip_time_s: float = 19.0
    dwell_s: float = 0.08
    angle_threshold_deg: float = 0.35
    require_collector_trips: int = 6
    require_morocco_trip: bool = True
    boundary_line_count: int = 8


@dataclass(slots=True)
class OutOfStepRelayRuntime:
    spec: OutOfStepRelaySpec
    line_ids: list[str]
    timer_s: float = 0.0
    tripped: bool = False
    trip_time_s: float | None = None


@dataclass(frozen=True, slots=True)
class UnderFrequencyTieRelaySpec:
    min_trip_time_s: float = 18.0
    dwell_s: float = 0.08
    under_frequency_hz: float = 49.5
    angle_threshold_deg: float = 0.60
    require_collector_trips: int = 6
    require_generator_trip: bool = True
    boundary_line_count: int = 8


@dataclass(slots=True)
class UnderFrequencyTieRelayRuntime:
    spec: UnderFrequencyTieRelaySpec
    line_ids: list[str]
    timer_s: float = 0.0
    tripped: bool = False
    trip_time_s: float | None = None


@dataclass(frozen=True, slots=True)
class GeneratorProtectionRelaySpec:
    min_trip_time_s: float = 16.35
    dwell_s: float = 0.08
    under_voltage_pu: float = 0.93
    under_frequency_hz: float = 49.8
    angle_threshold_deg: float = 0.45
    require_collector_trips: int = 6
    target_mva: float = 8000.0
    max_genrou: int = 16


@dataclass(slots=True)
class GeneratorProtectionRelayRuntime:
    spec: GeneratorProtectionRelaySpec
    genrou_ids: list[str]
    pv_ids: list[str]
    timer_s: float = 0.0
    tripped: bool = False
    trip_time_s: float | None = None
    tripped_mva: float = 0.0


@dataclass(frozen=True, slots=True)
class UFLSUVLSStageSpec:
    name: str
    min_time_s: float
    load_fraction: float
    under_voltage_pu: float = 0.90
    under_frequency_hz: float = 49.0
    dwell_s: float = 0.08
    max_shed_mw: float = 1200.0


@dataclass(slots=True)
class UFLSUVLSStageRuntime:
    spec: UFLSUVLSStageSpec
    load_ids: list[str]
    timer_s: float = 0.0
    armed: bool = False
    applied: bool = False
    applied_time_s: float | None = None
    shed_mw: float = 0.0


@dataclass(frozen=True, slots=True)
class BlackoutCriteriaSpec:
    voltage_pu: float = 0.75
    voltage_dwell_s: float = 0.08
    frequency_hz: float = 47.5
    frequency_dwell_s: float = 0.08
    angle_threshold_deg: float = 2.5
    angle_dwell_s: float = 1.20


@dataclass(frozen=True, slots=True)
class IslandBlackoutCriteriaSpec:
    min_generation_mw: float = 250.0
    min_reference_generation_mw: float = 50.0
    voltage_pu: float = 0.86
    voltage_dwell_s: float = 0.20
    frequency_hz: float = 48.5
    frequency_dwell_s: float = 0.20
    angle_threshold_deg: float = 10.0
    angle_dwell_s: float = 0.35
    imbalance_fraction: float = 0.55
    imbalance_dwell_s: float = 0.35
    min_time_after_hvdc_block_s: float = 0.40


@dataclass(slots=True)
class BlackoutCriteriaRuntime:
    spec: BlackoutCriteriaSpec
    voltage_timer_s: float = 0.0
    frequency_timer_s: float = 0.0
    angle_timer_s: float = 0.0
    declared: bool = False
    declared_time_s: float | None = None
    reason: str | None = None


@dataclass(slots=True)
class IslandBlackoutRuntime:
    spec: IslandBlackoutCriteriaSpec
    voltage_timers_s: dict[str, float] = field(default_factory=dict)
    frequency_timers_s: dict[str, float] = field(default_factory=dict)
    angle_timers_s: dict[str, float] = field(default_factory=dict)
    imbalance_timers_s: dict[str, float] = field(default_factory=dict)
    no_reference_timers_s: dict[str, float] = field(default_factory=dict)
    deenergized_islands: set[str] = field(default_factory=set)


@dataclass(frozen=True, slots=True)
class IslandSnapshot:
    island_id: str
    buses: frozenset[int]
    line_ids: tuple[str, ...]
    affected_bus_count: int
    online_genrou_ids: tuple[str, ...]
    online_pv_ids: tuple[str, ...]
    online_load_ids: tuple[str, ...]
    online_collector_names: tuple[str, ...]
    online_shunt_ids: tuple[str, ...]
    generation_mw: float
    load_mw: float
    q_load_mvar: float
    q_absorption_mvar: float
    vmin_pu: float
    vmax_pu: float
    frequency_hz: float
    angle_spread_deg: float
    has_voltage_reference: bool
    imbalance_fraction: float
    deenergized: bool = False


@dataclass(slots=True)
class AffectedAreaSelection:
    seed_buses: list[int]
    radius: int
    affected_buses: list[int]
    boundary_line_ids: list[str]
    affected_genrou_ids: list[str]
    protected_genrou_ids: list[str]
    affected_pv_ids: list[str]
    selected_load_ids: list[str]
    genrou_mva_by_id: dict[str, float]
    load_mw_by_id: dict[str, float]
    boundary_lines: list[dict[str, float | int | str]]


@dataclass(slots=True)
class EventRecord:
    time_s: float
    code: str
    category: str
    description: str
    lost_p_mw: float = 0.0
    lost_q_absorption_mvar: float = 0.0
    net_load_increase_mw: float = 0.0
    q_demand_increase_mvar: float = 0.0
    island_id: str | None = None
    trigger: str | None = None
    device_model: str | None = None
    device_id: str | None = None
    bus_id: int | None = None
    success: bool | None = None


def report_relay_stages(
    pickup_pu: float = 1.095,
    dwell_s: float = 0.25,
    reset_ratio: float = 0.95,
) -> tuple[OverVoltageRelayStage, ...]:
    """Hidden collector-side overvoltage relay with pickup/reset hysteresis.

    The paper replica uses this as a plant/evacuation-side relay, not a
    400-kV transmission alarm.  Event timing must therefore come from the
    collector voltage crossing this pickup plus dwell, not from external event
    scheduling.
    """

    return (
        OverVoltageRelayStage("ov", float(pickup_pu), float(dwell_s), reset_ratio),
    )


def _report_q_estimate(active_power_mw: float) -> float:
    """Transparent fallback for report rows where only MW is robustly usable."""

    return 0.22 * float(active_power_mw)


def build_report_collector_specs(
    *,
    count: int = 16,
    tap_scale: float = 0.95,
    q_absorption_scale: float = 1.0,
    reset_ratio: float = 0.95,
) -> tuple[CollectorSpec, ...]:
    """Report-event collector library compressed onto the ACTIVSg2000 corridor.

    The first three rows use the final-report MW/MVAr values directly.  The
    smaller event-5 rows use proportional MVAr estimates when the table gives
    reliable MW but not a separately auditable MVAr absorption number.
    """

    rows: list[dict[str, Any]] = [
        {
            "name": "Granada355",
            "bus": 7406,
            "tap": 1.012,
            "threshold": 1.1105,
            "mw": 355.0,
            "q": 165.0,
            "cluster": "E3 Granada",
            "time": 10.0,
            "visible": True,
            "dwell": 0.45,
        },
        {
            "name": "Badajoz582",
            "bus": 7058,
            "tap": 1.026,
            "threshold": 1.1035,
            "mw": 582.0,
            "q": 165.0,
            "cluster": "E4 Badajoz",
            "time": 14.55,
            "visible": True,
            "dwell": 0.20,
        },
        {
            "name": "Badajoz145",
            "bus": 7042,
            "tap": 1.010,
            "threshold": 1.0965,
            "mw": 145.0,
            "q": 37.0,
            "cluster": "E4 Badajoz",
            "time": 14.95,
            "visible": True,
            "dwell": 0.20,
        },
        {
            "name": "Segovia23",
            "bus": 7018,
            "tap": 1.002,
            "threshold": 1.0960,
            "mw": 23.0,
            "q": _report_q_estimate(23.0),
            "cluster": "E5 multi-site",
            "time": 15.18,
            "dwell": 0.12,
            "estimated": True,
        },
        {
            "name": "Seville550",
            "bus": 7029,
            "tap": 0.947,
            "threshold": 1.0960,
            "mw": 550.0,
            "q": 195.0,
            "cluster": "E5 multi-site",
            "time": 15.28,
            "visible": True,
            "dwell": 0.12,
        },
        {
            "name": "Segovia94",
            "bus": 7407,
            "tap": 1.020,
            "threshold": 1.0960,
            "mw": 94.0,
            "q": _report_q_estimate(94.0),
            "cluster": "E5 multi-site",
            "time": 15.38,
            "dwell": 0.12,
            "estimated": True,
        },
        {
            "name": "Badajoz118",
            "bus": 0,
            "tap": 0.967,
            "threshold": 1.1045,
            "mw": 118.0,
            "q": _report_q_estimate(118.0),
            "cluster": "E5 multi-site",
            "time": 15.46,
            "dwell": 0.12,
            "estimated": True,
        },
        {
            "name": "Huelva34",
            "bus": 0,
            "tap": 0.968,
            "threshold": 1.1045,
            "mw": 34.0,
            "q": _report_q_estimate(34.0),
            "cluster": "E5 multi-site",
            "time": 15.54,
            "dwell": 0.12,
            "estimated": True,
        },
        {
            "name": "Caceres38",
            "bus": 0,
            "tap": 0.954,
            "threshold": 1.1050,
            "mw": 38.0,
            "q": _report_q_estimate(38.0),
            "cluster": "E5 multi-site",
            "time": 15.62,
            "dwell": 0.12,
            "estimated": True,
        },
        {
            "name": "Badajoz72",
            "bus": 0,
            "tap": 1.020,
            "threshold": 1.0960,
            "mw": 72.0,
            "q": _report_q_estimate(72.0),
            "cluster": "E5 multi-site",
            "time": 15.70,
            "dwell": 0.12,
            "estimated": True,
        },
        {
            "name": "Badajoz16",
            "bus": 0,
            "tap": 0.936,
            "threshold": 1.1010,
            "mw": 16.0,
            "q": _report_q_estimate(16.0),
            "cluster": "E5 multi-site",
            "time": 15.78,
            "dwell": 0.16,
            "estimated": True,
        },
        {
            "name": "Caceres41",
            "bus": 0,
            "tap": 0.946,
            "threshold": 1.1000,
            "mw": 41.0,
            "q": _report_q_estimate(41.0),
            "cluster": "E5 multi-site",
            "time": 15.86,
            "dwell": 0.16,
            "estimated": True,
        },
        {
            "name": "Cadiz26",
            "bus": 0,
            "tap": 0.959,
            "threshold": 1.1050,
            "mw": 26.0,
            "q": _report_q_estimate(26.0),
            "cluster": "E5 multi-site",
            "time": 15.94,
            "dwell": 0.16,
            "estimated": True,
        },
        {
            "name": "Cadiz128",
            "bus": 0,
            "tap": 0.930,
            "threshold": 1.0800,
            "mw": 128.0,
            "q": _report_q_estimate(128.0),
            "cluster": "E5 multi-site",
            "time": 16.02,
            "dwell": 0.18,
            "estimated": True,
        },
        {
            "name": "Malaga154",
            "bus": 0,
            "tap": 0.957,
            "threshold": 1.1060,
            "mw": 154.0,
            "q": _report_q_estimate(154.0),
            "cluster": "E5 multi-site",
            "time": 16.10,
            "dwell": 0.16,
            "estimated": True,
        },
        {
            "name": "Cuenca530",
            "bus": 0,
            "tap": 0.951,
            "threshold": 1.1050,
            "mw": 530.0,
            "q": _report_q_estimate(530.0),
            "cluster": "E5 multi-site",
            "time": 16.18,
            "visible": True,
            "dwell": 0.18,
            "estimated": True,
        },
        {
            "name": "ReserveEast220",
            "bus": 0,
            "mw": 220.0,
            "q": _report_q_estimate(220.0),
            "cluster": "E5 reserve",
            "time": 16.28,
            "dwell": 1.80,
            "estimated": True,
        },
        {
            "name": "ReserveSouth180",
            "bus": 0,
            "mw": 180.0,
            "q": _report_q_estimate(180.0),
            "cluster": "E5 reserve",
            "time": 16.38,
            "dwell": 1.90,
            "estimated": True,
        },
        {
            "name": "ReserveWest160",
            "bus": 0,
            "mw": 160.0,
            "q": _report_q_estimate(160.0),
            "cluster": "E5 reserve",
            "time": 16.48,
            "dwell": 2.00,
            "estimated": True,
        },
        {
            "name": "ReserveNorth140",
            "bus": 0,
            "mw": 140.0,
            "q": _report_q_estimate(140.0),
            "cluster": "E5 reserve",
            "time": 16.58,
            "dwell": 2.10,
            "estimated": True,
        },
    ]

    specs: list[CollectorSpec] = []
    for row in rows[: int(count)]:
        specs.append(
            CollectorSpec(
                str(row["name"]),
                int(row["bus"]),
                float(row.get("tap", tap_scale)),
                float(row.get("threshold", 1.095)),
                float(row["dwell"]),
                float(row["mw"]),
                float(row["q"]) * float(q_absorption_scale),
                fixed_pf_q_fraction=0.35,
                fast_threshold_pu=None,
                fast_dwell_s=0.0,
                report_cluster=str(row["cluster"]),
                report_time_s=float(row["time"]),
                min_trip_time_s=0.0,
                relay_stages=report_relay_stages(
                    float(row.get("threshold", 1.095)),
                    float(row["dwell"]),
                    reset_ratio,
                ),
                q_absorption_estimated=bool(row.get("estimated", False)),
                visible_trace=bool(row.get("visible", False)),
            )
        )
    return tuple(specs)


@dataclass(slots=True)
class ReplicationConfig:
    case_id: str = "activsg2000_stable"
    tf_s: float = 30.0
    tstep_s: float = 0.01
    pflow_tol: float = 1e-4
    tds_tol: float = 1e-4
    # Report-event protected evacuation assets.  The first six are pinned to
    # the high-voltage ACTIVSg2000 corridor; remaining entries are assigned
    # deterministically to high-generation affected-area buses during build.
    collector_specs: tuple[CollectorSpec, ...] = field(
        default_factory=build_report_collector_specs
    )
    meshing_source_lines: tuple[str, ...] = (
        "Line_1767",
        "Line_1723",
        "Line_1678",
        "Line_1679",
    )
    # Operator actions are intentionally all before the first relay trip in the
    # default calibration.  They compress the report's long voltage-control
    # period into a 30 s academic surrogate.  The report-aligned default keeps
    # all manual/topology/HVDC actions before the first protection pickup; the
    # later cascade is driven by automatic protection and island viability.
    meshing_times_s: tuple[float, ...] = (0.6, 1.0, 1.4, 1.8)
    meshing_equivalent_strengths: tuple[float, ...] = (1.0, 1.0, 0.05, 0.05)
    reactor_actions: tuple[ReactorSwitchAction, ...] = (
        ReactorSwitchAction(
            2.2,
            7406,
            8.0,
            1,
            "Shunt reactor connected for local voltage control.",
        ),
        ReactorSwitchAction(
            2.6,
            7058,
            8.0,
            0,
            "Shunt reactor opened after operator voltage-control review.",
        ),
        ReactorSwitchAction(
            3.0,
            7042,
            8.0,
            1,
            "Shunt reactor connected in the high-voltage corridor.",
        ),
        ReactorSwitchAction(
            3.4,
            7406,
            8.0,
            0,
            "Shunt reactor opened as the operating point is rebalanced.",
        ),
        ReactorSwitchAction(
            3.8,
            7018,
            10.0,
            1,
            "Additional shunt reactor connected before the final voltage-reference action.",
        ),
        ReactorSwitchAction(
            4.2,
            7058,
            10.0,
            0,
            "Shunt reactor opened shortly before the protection cascade.",
        ),
    )
    hvdc_support_bus: int = 7406
    hvdc_q_absorption_mvar: float = 130.0
    hvdc_reference_time_s: float = 4.0
    hvdc_disable_time_s: float = 1e9
    allow_equivalent_blackout: bool = False
    blackout_actions: tuple[BlackoutDisturbanceAction, ...] = ()
    hvdc_surrogate: HVDCSurrogateSpec = field(default_factory=HVDCSurrogateSpec)
    out_of_step_relay: OutOfStepRelaySpec = field(default_factory=OutOfStepRelaySpec)
    generator_protection: GeneratorProtectionRelaySpec = field(
        default_factory=GeneratorProtectionRelaySpec
    )
    morocco_ac_relay: UnderFrequencyTieRelaySpec = field(
        default_factory=UnderFrequencyTieRelaySpec
    )
    defense_stages: tuple[UFLSUVLSStageSpec, ...] = (
        UFLSUVLSStageSpec("UFLS1", 17.0, 0.18, under_voltage_pu=0.97, max_shed_mw=900.0),
        UFLSUVLSStageSpec("UFLS2", 17.5, 0.22, under_voltage_pu=0.95, max_shed_mw=1200.0),
    )
    blackout_criteria: BlackoutCriteriaSpec = field(default_factory=BlackoutCriteriaSpec)
    island_blackout_criteria: IslandBlackoutCriteriaSpec = field(
        default_factory=IslandBlackoutCriteriaSpec
    )
    area_seed_buses: tuple[int, ...] = (7406, 7058, 7042, 7018, 7029, 7407)
    area_bfs_radius: int = 5
    area_min_genrou: int = 40
    area_min_loads: int = 20
    area_min_boundary_lines: int = 20
    area_max_radius: int = 8
    # For the paper-facing replica, place hidden collector/evacuation assets on
    # the high-voltage corridor so the transmission-side panel is a single
    # comparable voltage class, analogous to the 400 kV traces in the report.
    collector_upstream_min_kv: float = 500.0
    collector_initial_margin_pu: float = 0.085
    collector_anchor_initial_margin_pu: float = 0.080
    collector_granada_initial_margin_pu: float = 0.055
    collector_badajoz_initial_margin_pu: float = 0.095
    nominal_frequency_hz: float = 50.0
    enable_post_cascade_relays: bool = True
    # The report links fixed-power-factor renewable behavior and net-export
    # changes to a progressive loss of reactive absorption before the first
    # trip.  The default applies that ramp to the fixed-PF component of the
    # modeled collector blocks; a separate background-shunt mode is available
    # for sensitivity checks but is not used in the paper-ready calibration.
    export_reduction_mode: str = "collector_fixed_pf"
    background_fixed_pf_bus: int = 7406
    background_fixed_pf_q_absorption_mvar: float = 0.0
    export_reduction_time_s: float = 4.4
    export_reduction_ramp_s: float = 11.5
    relay_timer_decay: float = 0.30
    collapse_voltage_pu: float = 0.75


@dataclass(slots=True)
class ReplicationArtifacts:
    output_dir: Path
    summary_path: Path
    events_path: Path
    collector_trace_path: Path
    system_trace_path: Path
    config_path: Path
    island_trace_path: Path | None = None
    deenergized_assets_path: Path | None = None
    blackout_sequence_path: Path | None = None
    figure_paths: list[Path] = field(default_factory=list)


def _diagnostic_equivalent_blackout_actions() -> tuple[BlackoutDisturbanceAction, ...]:
    return (
        BlackoutDisturbanceAction(
            14.85,
            7406,
            1800.0,
            2800.0,
            3,
            "Diagnostic equivalent imbalance step; not used in the paper scenario.",
        ),
        BlackoutDisturbanceAction(
            15.45,
            7058,
            2000.0,
            3200.0,
            3,
            "Diagnostic equivalent imbalance step; not used in the paper scenario.",
        ),
        BlackoutDisturbanceAction(
            16.05,
            7042,
            1800.0,
            2800.0,
            3,
            "Diagnostic equivalent imbalance step; not used in the paper scenario.",
        ),
        BlackoutDisturbanceAction(
            16.65,
            7018,
            1600.0,
            2400.0,
            3,
            "Diagnostic equivalent imbalance step; not used in the paper scenario.",
        ),
    )


def _disable_predefined_events(system: Any) -> None:
    for model_name in ("Toggle", "Alter", "Fault"):
        if not hasattr(system, model_name):
            continue
        table = getattr(system, model_name)
        for i in range(getattr(table, "n", 0)):
            table.u.v[i] = 0


def _find_position(model: Any, device_id: str | int) -> int:
    for i, value in enumerate(list(model.idx.v)):
        if str(value) == str(device_id):
            return i
    raise ValueError(f"{device_id!r} is not an index in {model.class_name}")


def _set_status(system: Any, model_name: str, device_id: str | int, status: int) -> None:
    if hasattr(system, "set_status"):
        try:
            system.set_status(model_name, device_id, status)
            return
        except Exception:
            # Fall back to direct status only if the public updater rejects the
            # synthetic device id.  This is rare for added PQ/Shunt devices.
            pass
    model = getattr(system, model_name)
    model.u.v[_find_position(model, device_id)] = status


def _status(system: Any, model_name: str, device_id: str | int) -> float:
    model = getattr(system, model_name)
    return float(model.u.v[_find_position(model, device_id)])


def _safe_status(system: Any, model_name: str, device_id: str | int) -> float:
    try:
        return _status(system, model_name, device_id)
    except Exception:
        return 0.0


def _wrap_angle_diff(angle_a: float, angle_b: float) -> float:
    return float(np.angle(np.exp(1j * (float(angle_a) - float(angle_b)))))


def update_overvoltage_relay_timers(
    collector: CollectorRuntime,
    voltage_pu: float,
    dt: float,
    decay: float,
) -> str | None:
    """Advance two-stage OV relay pickup/reset logic and return a trip stage."""

    stages = collector.spec.relay_stages
    if not stages:
        stages = (
            OverVoltageRelayStage(
                "legacy",
                collector.spec.threshold_pu,
                collector.spec.dwell_s,
                reset_ratio=1.0,
            ),
        )
    for stage in stages:
        timer = float(collector.stage_timers_s.get(stage.name, 0.0))
        picked_up = bool(collector.stage_picked_up.get(stage.name, False))
        reset_level = stage.pickup_pu * stage.reset_ratio
        if voltage_pu >= stage.pickup_pu:
            picked_up = True
            timer += dt
        elif picked_up and voltage_pu > reset_level:
            timer += dt
        else:
            picked_up = False
            timer = max(0.0, timer - decay * dt)
        collector.stage_timers_s[stage.name] = timer
        collector.stage_picked_up[stage.name] = picked_up
        if timer >= stage.dwell_s:
            return stage.name
    return None


def _active_physical_bus_voltage(system: Any, physical_bus_count: int) -> np.ndarray:
    return np.asarray(system.Bus.v.v[:physical_bus_count], dtype=float)


def _estimate_frequency_hz(system: Any, nominal_hz: float) -> float:
    if not hasattr(system, "GENROU") or getattr(system.GENROU, "n", 0) == 0:
        return float(nominal_hz)
    omega = np.asarray(system.GENROU.omega.v, dtype=float)
    if omega.size == 0:
        return float(nominal_hz)
    status = np.asarray(system.GENROU.u.v, dtype=float)
    if hasattr(system.GENROU, "M"):
        weights = np.asarray(system.GENROU.M.v, dtype=float)
    elif hasattr(system.GENROU, "Sn"):
        weights = np.asarray(system.GENROU.Sn.v, dtype=float)
    else:
        weights = np.ones_like(omega)
    mask = np.isfinite(omega) & (status > 0.5) & np.isfinite(weights) & (weights > 0.0)
    if not np.any(mask):
        return float(nominal_hz)
    return float(nominal_hz * np.average(omega[mask], weights=weights[mask]))


def _coherent_angle_separation_rad(
    system: Any,
    affected_buses: set[int],
) -> float:
    if not hasattr(system, "GENROU") or getattr(system.GENROU, "n", 0) == 0:
        return 0.0
    gen = system.GENROU
    delta = np.asarray(gen.delta.v, dtype=float)
    status = np.asarray(gen.u.v, dtype=float)
    if hasattr(gen, "M"):
        weights = np.asarray(gen.M.v, dtype=float)
    elif hasattr(gen, "Sn"):
        weights = np.asarray(gen.Sn.v, dtype=float)
    else:
        weights = np.ones_like(delta)
    affected_delta: list[float] = []
    affected_weights: list[float] = []
    rest_delta: list[float] = []
    rest_weights: list[float] = []
    for i in range(getattr(gen, "n", 0)):
        if status[i] <= 0.5 or not np.isfinite(delta[i]):
            continue
        w = float(weights[i]) if np.isfinite(weights[i]) and weights[i] > 0.0 else 1.0
        if int(gen.bus.v[i]) in affected_buses:
            affected_delta.append(float(delta[i]))
            affected_weights.append(w)
        else:
            rest_delta.append(float(delta[i]))
            rest_weights.append(w)
    if not affected_delta or not rest_delta:
        return 0.0
    a = float(np.angle(np.average(np.exp(1j * np.asarray(affected_delta)), weights=affected_weights)))
    b = float(np.angle(np.average(np.exp(1j * np.asarray(rest_delta)), weights=rest_weights)))
    return abs(_wrap_angle_diff(a, b))


def _model_n(system: Any, model_name: str) -> int:
    return int(getattr(getattr(system, model_name, None), "n", 0))


def _model_status_at(model: Any, index: int) -> float:
    if hasattr(model, "u") and hasattr(model.u, "v"):
        return float(model.u.v[index])
    return 1.0


def _base_mva(system: Any) -> float:
    return float(getattr(getattr(system, "config", None), "mva", 100.0))


def _bus_positions(system: Any) -> dict[int, int]:
    return {int(bus): i for i, bus in enumerate(system.Bus.idx.v)}


def _line_graph_components(
    system: Any,
    *,
    ignored_line_ids: set[str] | None = None,
) -> tuple[list[frozenset[int]], dict[int, str], dict[str, tuple[int, int]]]:
    ignored = ignored_line_ids or set()
    bus_ids = [int(bus) for bus in system.Bus.idx.v]
    adjacency: dict[int, list[int]] = defaultdict(list)
    line_endpoints: dict[str, tuple[int, int]] = {}
    for i in range(_model_n(system, "Line")):
        line_id = str(system.Line.idx.v[i])
        b1 = int(system.Line.bus1.v[i])
        b2 = int(system.Line.bus2.v[i])
        line_endpoints[line_id] = (b1, b2)
        if line_id in ignored or _model_status_at(system.Line, i) <= 0.5:
            continue
        adjacency[b1].append(b2)
        adjacency[b2].append(b1)

    components: list[frozenset[int]] = []
    bus_to_island: dict[int, str] = {}
    seen: set[int] = set()
    for bus in bus_ids:
        if bus in seen:
            continue
        queue: deque[int] = deque([bus])
        seen.add(bus)
        component: set[int] = set()
        while queue:
            current = queue.popleft()
            component.add(current)
            for nxt in adjacency.get(current, []):
                if nxt in seen:
                    continue
                seen.add(nxt)
                queue.append(nxt)
        components.append(frozenset(component))
    components.sort(key=lambda item: (-len(item), min(item) if item else 0))
    for idx, component in enumerate(components, start=1):
        island_id = f"I{idx:03d}"
        for bus in component:
            bus_to_island[bus] = island_id
    return components, bus_to_island, line_endpoints


def _generator_output_mw(system: Any, position: int) -> float:
    if hasattr(system.GENROU, "Pe"):
        pe = float(system.GENROU.Pe.v[position])
        if np.isfinite(pe) and pe > 0.0:
            return pe * _base_mva(system)
    if hasattr(system.GENROU, "Sn"):
        return max(0.0, float(system.GENROU.Sn.v[position]) * 0.75)
    return 0.0


def compute_island_snapshots(
    system: Any,
    *,
    collectors: list[CollectorRuntime],
    affected_buses: set[int],
    ignored_line_ids: set[str] | None = None,
    deenergized_islands: set[str] | None = None,
    nominal_frequency_hz: float = 50.0,
) -> list[IslandSnapshot]:
    components, bus_to_island, line_endpoints = _line_graph_components(
        system,
        ignored_line_ids=ignored_line_ids,
    )
    base = _base_mva(system)
    bus_pos = _bus_positions(system)
    deenergized = deenergized_islands or set()
    lines_by_island: dict[str, list[str]] = defaultdict(list)
    for line_id, (b1, b2) in line_endpoints.items():
        island_id = bus_to_island.get(b1)
        if island_id and island_id == bus_to_island.get(b2):
            lines_by_island[island_id].append(line_id)

    snapshots: list[IslandSnapshot] = []
    for idx, component in enumerate(components, start=1):
        island_id = f"I{idx:03d}"
        buses = set(component)
        voltages = np.asarray(
            [float(system.Bus.v.v[bus_pos[bus]]) for bus in buses if bus in bus_pos],
            dtype=float,
        )
        vmin = float(np.nanmin(voltages)) if voltages.size else float("nan")
        vmax = float(np.nanmax(voltages)) if voltages.size else float("nan")

        online_genrou: list[str] = []
        online_pv: list[str] = []
        online_loads: list[str] = []
        online_shunts: list[str] = []
        online_collectors: list[str] = []
        generation_mw = 0.0
        reference_mw = 0.0
        load_mw = 0.0
        q_load_mvar = 0.0
        q_absorption_mvar = 0.0
        omega_values: list[float] = []
        omega_weights: list[float] = []
        angle_values: list[float] = []

        for i in range(_model_n(system, "GENROU")):
            bus = int(system.GENROU.bus.v[i])
            if bus not in buses or _model_status_at(system.GENROU, i) <= 0.5:
                continue
            gen_id = str(system.GENROU.idx.v[i])
            online_genrou.append(gen_id)
            gen_mw = _generator_output_mw(system, i)
            generation_mw += gen_mw
            reference_mw += gen_mw
            weight = float(system.GENROU.M.v[i]) if hasattr(system.GENROU, "M") else 1.0
            if not np.isfinite(weight) or weight <= 0.0:
                weight = 1.0
            if hasattr(system.GENROU, "omega"):
                omega = float(system.GENROU.omega.v[i])
                if np.isfinite(omega):
                    omega_values.append(omega)
                    omega_weights.append(weight)
            if hasattr(system.GENROU, "delta"):
                delta = float(system.GENROU.delta.v[i])
                if np.isfinite(delta):
                    angle_values.append(delta)

        for i in range(_model_n(system, "PV")):
            bus = int(system.PV.bus.v[i])
            if bus not in buses or _model_status_at(system.PV, i) <= 0.5:
                continue
            online_pv.append(str(system.PV.idx.v[i]))
            generation_mw += max(0.0, float(system.PV.p0.v[i]) * base)

        for i in range(_model_n(system, "PQ")):
            bus = int(system.PQ.bus.v[i])
            if bus not in buses or _model_status_at(system.PQ, i) <= 0.5:
                continue
            p_mw = float(system.PQ.p0.v[i]) * base if hasattr(system.PQ, "p0") else 0.0
            q_mvar = float(system.PQ.q0.v[i]) * base if hasattr(system.PQ, "q0") else 0.0
            if p_mw >= 0.0:
                online_loads.append(str(system.PQ.idx.v[i]))
                load_mw += p_mw
                q_load_mvar += max(0.0, q_mvar)
            else:
                generation_mw += -p_mw

        for i in range(_model_n(system, "Shunt")):
            bus = int(system.Shunt.bus.v[i])
            if bus not in buses or _model_status_at(system.Shunt, i) <= 0.5:
                continue
            online_shunts.append(str(system.Shunt.idx.v[i]))
            b = float(system.Shunt.b.v[i]) if hasattr(system.Shunt, "b") else 0.0
            if b < 0.0:
                q_absorption_mvar += -b * base

        for collector in collectors:
            if collector.collector_bus in buses and not collector.tripped:
                online_collectors.append(collector.spec.name)

        if omega_values:
            frequency_hz = float(
                nominal_frequency_hz
                * np.average(np.asarray(omega_values), weights=np.asarray(omega_weights))
            )
        else:
            frequency_hz = float("nan")
        if len(angle_values) >= 2:
            angles = np.asarray(angle_values, dtype=float)
            mean_angle = float(np.angle(np.average(np.exp(1j * angles))))
            spread = max(abs(_wrap_angle_diff(angle, mean_angle)) for angle in angles)
            angle_spread_deg = float(np.rad2deg(spread))
        else:
            angle_spread_deg = 0.0
        imbalance = abs(load_mw - generation_mw) / max(load_mw, generation_mw, 1.0)
        snapshots.append(
            IslandSnapshot(
                island_id=island_id,
                buses=frozenset(buses),
                line_ids=tuple(sorted(lines_by_island.get(island_id, []))),
                affected_bus_count=len(buses & affected_buses),
                online_genrou_ids=tuple(sorted(online_genrou)),
                online_pv_ids=tuple(sorted(online_pv)),
                online_load_ids=tuple(sorted(online_loads)),
                online_collector_names=tuple(sorted(online_collectors)),
                online_shunt_ids=tuple(sorted(online_shunts)),
                generation_mw=float(generation_mw),
                load_mw=float(load_mw),
                q_load_mvar=float(q_load_mvar),
                q_absorption_mvar=float(q_absorption_mvar),
                vmin_pu=vmin,
                vmax_pu=vmax,
                frequency_hz=frequency_hz,
                angle_spread_deg=angle_spread_deg,
                has_voltage_reference=reference_mw >= 1.0,
                imbalance_fraction=float(imbalance),
                deenergized=island_id in deenergized,
            )
        )
    return snapshots


def select_affected_area(system: Any, cfg: ReplicationConfig) -> AffectedAreaSelection:
    seeds = [int(bus) for bus in cfg.area_seed_buses]
    adjacency: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for i in range(system.Line.n):
        if float(system.Line.u.v[i]) <= 0.5:
            continue
        b1 = int(system.Line.bus1.v[i])
        b2 = int(system.Line.bus2.v[i])
        adjacency[b1].append((b2, i))
        adjacency[b2].append((b1, i))

    selected_radius = int(cfg.area_bfs_radius)
    affected_buses: set[int] = set(seeds)
    boundary_indices: list[int] = []
    affected_genrou: list[tuple[str, int, float]] = []
    affected_loads: list[tuple[str, int, float]] = []
    affected_pvs: list[tuple[str, int, float]] = []
    base_mva = float(system.config.mva)

    for radius in range(int(cfg.area_bfs_radius), int(cfg.area_max_radius) + 1):
        seen = set(seeds)
        queue: deque[tuple[int, int]] = deque((bus, 0) for bus in seeds)
        while queue:
            bus, depth = queue.popleft()
            if depth >= radius:
                continue
            for next_bus, _line_pos in adjacency.get(bus, []):
                if next_bus in seen:
                    continue
                seen.add(next_bus)
                queue.append((next_bus, depth + 1))

        boundary_indices = []
        for i in range(system.Line.n):
            b1 = int(system.Line.bus1.v[i])
            b2 = int(system.Line.bus2.v[i])
            if (b1 in seen) != (b2 in seen):
                boundary_indices.append(i)

        affected_genrou = []
        for i in range(getattr(system.GENROU, "n", 0)):
            if int(system.GENROU.bus.v[i]) in seen:
                mva = float(system.GENROU.Sn.v[i]) if hasattr(system.GENROU, "Sn") else 0.0
                affected_genrou.append((str(system.GENROU.idx.v[i]), int(system.GENROU.bus.v[i]), mva))

        affected_loads = []
        for i in range(getattr(system.PQ, "n", 0)):
            if int(system.PQ.bus.v[i]) in seen and float(system.PQ.p0.v[i]) > 0.0:
                affected_loads.append(
                    (
                        str(system.PQ.idx.v[i]),
                        int(system.PQ.bus.v[i]),
                        float(system.PQ.p0.v[i]) * base_mva,
                    )
                )

        affected_pvs = []
        for i in range(getattr(system.PV, "n", 0)):
            if int(system.PV.bus.v[i]) in seen:
                affected_pvs.append(
                    (
                        str(system.PV.idx.v[i]),
                        int(system.PV.bus.v[i]),
                        float(system.PV.p0.v[i]) * base_mva,
                    )
                )

        selected_radius = radius
        affected_buses = seen
        if (
            len(boundary_indices) >= cfg.area_min_boundary_lines
            and len(affected_genrou) >= cfg.area_min_genrou
            and len(affected_loads) >= cfg.area_min_loads
        ):
            break

    boundary_infos: list[dict[str, float | int | str]] = []
    for i in boundary_indices:
        boundary_infos.append(
            {
                "idx": str(system.Line.idx.v[i]),
                "bus1": int(system.Line.bus1.v[i]),
                "bus2": int(system.Line.bus2.v[i]),
                "x_abs": abs(float(system.Line.x.v[i])),
                "r_abs": abs(float(system.Line.r.v[i])),
            }
        )
    boundary_infos.sort(key=lambda item: (float(item["x_abs"]), str(item["idx"])))
    boundary_line_ids = [str(item["idx"]) for item in boundary_infos]

    affected_genrou.sort(key=lambda item: (-item[2], item[0]))
    protected_genrou: list[str] = []
    protected_mva = 0.0
    for gen_id, _bus, mva in affected_genrou:
        if len(protected_genrou) >= cfg.generator_protection.max_genrou:
            break
        protected_genrou.append(gen_id)
        protected_mva += mva
        if protected_mva >= cfg.generator_protection.target_mva:
            break

    protected_buses = {
        bus for gen_id, bus, _mva in affected_genrou if gen_id in set(protected_genrou)
    }
    affected_pv_ids = [
        pv_id for pv_id, bus, _mw in sorted(affected_pvs, key=lambda item: (-item[2], item[0]))
        if bus in protected_buses
    ]
    affected_loads.sort(key=lambda item: (-item[2], item[0]))
    selected_load_ids = [load_id for load_id, _bus, _mw in affected_loads]

    return AffectedAreaSelection(
        seed_buses=seeds,
        radius=selected_radius,
        affected_buses=sorted(affected_buses),
        boundary_line_ids=boundary_line_ids,
        affected_genrou_ids=[item[0] for item in affected_genrou],
        protected_genrou_ids=protected_genrou,
        affected_pv_ids=affected_pv_ids,
        selected_load_ids=selected_load_ids,
        genrou_mva_by_id={gen_id: mva for gen_id, _bus, mva in affected_genrou},
        load_mw_by_id={load_id: mw for load_id, _bus, mw in affected_loads},
        boundary_lines=boundary_infos,
    )


def resolve_collector_specs(
    system: Any,
    cfg: ReplicationConfig,
    area_selection: AffectedAreaSelection,
) -> tuple[CollectorSpec, ...]:
    """Assign report collectors to deterministic high-voltage generation buses."""

    affected = set(area_selection.affected_buses)
    bus_vn = {
        int(system.Bus.idx.v[i]): float(system.Bus.Vn.v[i])
        for i in range(getattr(system.Bus, "n", 0))
    }
    bus_position = _bus_positions(system)

    def initial_bus_voltage(bus: int) -> float:
        pos = bus_position.get(int(bus))
        if pos is None:
            return float("nan")
        if hasattr(system.Bus, "v0") and hasattr(system.Bus.v0, "v"):
            value = float(system.Bus.v0.v[pos])
            if np.isfinite(value) and value > 0.0:
                return value
        if hasattr(system.Bus, "v") and hasattr(system.Bus.v, "v"):
            return float(system.Bus.v.v[pos])
        return 1.0

    def with_initial_margin(spec: CollectorSpec, bus: int) -> CollectorSpec:
        if spec.report_cluster == "E3 Granada":
            margin = float(getattr(cfg, "collector_granada_initial_margin_pu", 0.0) or 0.0)
        elif spec.report_cluster == "E4 Badajoz":
            margin = float(getattr(cfg, "collector_badajoz_initial_margin_pu", 0.0) or 0.0)
        elif spec.report_cluster in {"E3 Granada", "E4 Badajoz"}:
            margin = float(getattr(cfg, "collector_anchor_initial_margin_pu", 0.0) or 0.0)
        else:
            margin = float(getattr(cfg, "collector_initial_margin_pu", 0.0) or 0.0)
        if margin <= 0.0:
            return spec
        upstream_v0 = initial_bus_voltage(bus)
        if not np.isfinite(upstream_v0) or upstream_v0 <= 0.0:
            return spec
        current_estimate = upstream_v0 / max(float(spec.tap), 1e-6)
        target = max(0.92, float(spec.threshold_pu) - margin)
        if spec.report_cluster not in {"E3 Granada", "E4 Badajoz"} and current_estimate <= target:
            return spec
        calibrated_tap = upstream_v0 / target
        if spec.report_cluster in {"E3 Granada", "E4 Badajoz"}:
            calibrated_tap = float(min(1.10, max(0.90, calibrated_tap)))
        else:
            calibrated_tap = float(min(1.10, max(float(spec.tap), calibrated_tap)))
        return replace(spec, tap=calibrated_tap)

    min_upstream_kv = float(getattr(cfg, "collector_upstream_min_kv", 0.0) or 0.0)
    used = {
        int(spec.transmission_bus)
        for spec in cfg.collector_specs
        if spec.transmission_bus > 0
        and bus_vn.get(int(spec.transmission_bus), 0.0) >= min_upstream_kv
    }
    bus_scores: dict[int, float] = defaultdict(float)
    for i in range(getattr(system.GENROU, "n", 0)):
        bus = int(system.GENROU.bus.v[i])
        if bus in affected and bus_vn.get(bus, 0.0) >= min_upstream_kv:
            bus_scores[bus] += float(system.GENROU.Sn.v[i]) if hasattr(system.GENROU, "Sn") else 0.0
    base_mva = float(system.config.mva)
    for i in range(getattr(system.PV, "n", 0)):
        bus = int(system.PV.bus.v[i])
        if bus in affected and bus_vn.get(bus, 0.0) >= min_upstream_kv:
            bus_scores[bus] += abs(float(system.PV.p0.v[i])) * base_mva

    adjacency: dict[int, list[int]] = defaultdict(list)
    for i in range(system.Line.n):
        if float(system.Line.u.v[i]) <= 0.5:
            continue
        b1 = int(system.Line.bus1.v[i])
        b2 = int(system.Line.bus2.v[i])
        adjacency[b1].append(b2)
        adjacency[b2].append(b1)
    for bus in affected:
        if bus_vn.get(int(bus), 0.0) >= min_upstream_kv:
            bus_scores[int(bus)] += 5.0 * len(adjacency.get(int(bus), []))

    def distance_to_used(candidate: int) -> int:
        if not used:
            return 99
        if candidate in used:
            return 0
        seen = {candidate}
        queue: deque[tuple[int, int]] = deque([(candidate, 0)])
        while queue:
            bus, distance = queue.popleft()
            if distance >= 8:
                continue
            for nxt in adjacency.get(bus, []):
                if nxt in seen:
                    continue
                if nxt in used:
                    return distance + 1
                seen.add(nxt)
                queue.append((nxt, distance + 1))
        return 99

    candidates = sorted(
        bus_scores,
        key=lambda bus: (-bus_scores[bus], -distance_to_used(bus), bus),
    )
    candidate_cursor = 0
    resolved: list[CollectorSpec] = []
    fallback = [
        int(bus)
        for bus in cfg.area_seed_buses
        if bus_vn.get(int(bus), 0.0) >= min_upstream_kv
    ] or list(cfg.area_seed_buses)
    for i, spec in enumerate(cfg.collector_specs):
        pinned_bus = int(spec.transmission_bus)
        if pinned_bus > 0 and bus_vn.get(pinned_bus, 0.0) >= min_upstream_kv:
            resolved.append(with_initial_margin(spec, pinned_bus))
            continue
        chosen: int | None = None
        while candidate_cursor < len(candidates):
            candidate = int(candidates[candidate_cursor])
            candidate_cursor += 1
            if candidate not in used:
                chosen = candidate
                break
        if chosen is None:
            chosen = int(fallback[i % len(fallback)])
        used.add(chosen)
        resolved.append(with_initial_margin(replace(spec, transmission_bus=chosen), chosen))
    return tuple(resolved)


class QuietAndes:
    def __init__(self, quiet: bool) -> None:
        self.quiet = quiet
        self._stdout_cm: Any = None
        self._stderr_cm: Any = None

    def __enter__(self) -> None:
        if not self.quiet:
            return None
        self._stdout = io.StringIO()
        self._stderr = io.StringIO()
        self._stdout_cm = redirect_stdout(self._stdout)
        self._stderr_cm = redirect_stderr(self._stderr)
        self._stdout_cm.__enter__()
        self._stderr_cm.__enter__()
        return None

    def __exit__(self, exc_type, exc, tb) -> bool:
        if self.quiet:
            self._stderr_cm.__exit__(exc_type, exc, tb)
            self._stdout_cm.__exit__(exc_type, exc, tb)
        return False


class IberianReplicationMonitor:
    def __init__(
        self,
        system: Any,
        cfg: ReplicationConfig,
        collectors: list[CollectorRuntime],
        mesh_line_ids: list[str],
        reactor_action_ids: list[str],
        area_selection: AffectedAreaSelection,
        hvdc: HVDCSurrogateRuntime,
        morocco_ac: UnderFrequencyTieRelayRuntime,
        out_of_step: OutOfStepRelayRuntime,
        generator_protection: GeneratorProtectionRelayRuntime,
        defense_stages: list[UFLSUVLSStageRuntime],
        blackout_criteria: BlackoutCriteriaRuntime,
        island_blackout: IslandBlackoutRuntime,
        blackout_load_ids: list[str],
        background_fixed_pf_shunt_id: str | None,
        physical_bus_count: int,
    ) -> None:
        self.system = system
        self.cfg = cfg
        self.collectors = collectors
        self.mesh_line_ids = mesh_line_ids
        self.reactor_action_ids = reactor_action_ids
        self.area_selection = area_selection
        self.affected_bus_set = set(area_selection.affected_buses)
        self.hvdc = hvdc
        self.morocco_ac = morocco_ac
        self.out_of_step = out_of_step
        self.generator_protection = generator_protection
        self.defense_stages = defense_stages
        self.blackout_criteria = blackout_criteria
        self.island_blackout = island_blackout
        self.blackout_load_ids = blackout_load_ids
        self.background_fixed_pf_shunt_id = background_fixed_pf_shunt_id
        self.physical_bus_count = physical_bus_count
        self.previous_time_s: float | None = None
        self.mesh_index = 0
        self.reactor_index = 0
        self.export_reduction_started = False
        self.export_reduction_done = False
        self.export_reduction_progress = 0.0
        if cfg.export_reduction_mode == "collector_fixed_pf":
            self.export_reduction_total_mvar = sum(
                collector.fixed_pf_q_absorption_mvar for collector in collectors
            )
        elif cfg.export_reduction_mode == "background_shunt":
            self.export_reduction_total_mvar = float(
                cfg.background_fixed_pf_q_absorption_mvar
            )
        else:
            raise ValueError(
                "export_reduction_mode must be 'collector_fixed_pf' or 'background_shunt'"
            )
        self.hvdc_reference_recorded = False
        self.blackout_action_index = 0
        self.events: list[EventRecord] = []
        self.collector_trace: list[dict[str, float | int | str]] = []
        self.system_trace: list[dict[str, float | int | str]] = []
        self.island_trace: list[dict[str, float | int | str]] = []
        self.deenergized_assets: list[dict[str, float | int | str | None]] = []
        self.blackout_sequence: list[dict[str, float | int | str | None]] = []
        self.last_islands: list[IslandSnapshot] = []
        self.cumulative_lost_p_mw = 0.0
        self.cumulative_lost_q_absorption_mvar = 0.0
        self.cumulative_blackout_load_mw = 0.0
        self.cumulative_blackout_q_demand_mvar = 0.0

    def __call__(self, t: float, system: Any) -> None:
        time_s = float(t)
        dt = 0.0 if self.previous_time_s is None else max(0.0, time_s - self.previous_time_s)
        self.previous_time_s = time_s

        if self.blackout_criteria.declared:
            self._record_system_trace(time_s, system)
            return

        self._apply_scheduled_actions(time_s)
        self._monitor_collectors(time_s, dt, system)
        self._apply_protection_chain(time_s, dt, system)
        self._record_system_trace(time_s, system)

    def _record_event(
        self,
        time_s: float,
        code: str,
        category: str,
        description: str,
        *,
        lost_p_mw: float = 0.0,
        lost_q_absorption_mvar: float = 0.0,
        net_load_increase_mw: float = 0.0,
        q_demand_increase_mvar: float = 0.0,
        island_id: str | None = None,
        trigger: str | None = None,
        device_model: str | None = None,
        device_id: str | None = None,
        bus_id: int | None = None,
        success: bool | None = None,
    ) -> None:
        self.cumulative_lost_p_mw += float(lost_p_mw)
        self.cumulative_lost_q_absorption_mvar += float(lost_q_absorption_mvar)
        self.cumulative_blackout_load_mw += float(net_load_increase_mw)
        self.cumulative_blackout_q_demand_mvar += float(q_demand_increase_mvar)
        event = EventRecord(
            float(time_s),
            code,
            category,
            description,
            float(lost_p_mw),
            float(lost_q_absorption_mvar),
            float(net_load_increase_mw),
            float(q_demand_increase_mvar),
            island_id,
            trigger,
            device_model,
            device_id,
            bus_id,
            success,
        )
        self.events.append(event)
        if category in {
            "protection_trip",
            "generator_trip",
            "defense_action",
            "morocco_ac_trip",
            "out_of_step_trip",
            "hvdc_block",
            "island_blackout_declared",
            "system_blackout_declared",
            "blackout_declared",
        }:
            self.blackout_sequence.append(asdict(event))

    def _apply_scheduled_actions(self, time_s: float) -> None:
        while (
            self.mesh_index < len(self.mesh_line_ids)
            and self.mesh_index < len(self.cfg.meshing_times_s)
            and time_s >= self.cfg.meshing_times_s[self.mesh_index]
        ):
            line_id = self.mesh_line_ids[self.mesh_index]
            _set_status(self.system, "Line", line_id, 1)
            self._record_event(
                self.cfg.meshing_times_s[self.mesh_index],
                "OA1",
                "operator_action",
                f"Additional 500-kV corridor branch energized ({line_id}).",
            )
            self.mesh_index += 1

        self._apply_fixed_pf_export_ramp(time_s)

        while (
            self.reactor_index < len(self.reactor_action_ids)
            and self.reactor_index < len(self.cfg.reactor_actions)
            and time_s >= self.cfg.reactor_actions[self.reactor_index].time_s
        ):
            action = self.cfg.reactor_actions[self.reactor_index]
            shunt_id = self.reactor_action_ids[self.reactor_index]
            _set_status(self.system, "Shunt", shunt_id, int(action.final_status))
            lost_q = action.q_mvar if int(action.final_status) == 0 else -action.q_mvar
            self._record_event(
                action.time_s,
                "OA3",
                "operator_action",
                f"{action.description} ({shunt_id}).",
                lost_q_absorption_mvar=lost_q,
            )
            self.reactor_index += 1

        if (
            not self.hvdc_reference_recorded
            and time_s >= self.cfg.hvdc_reference_time_s
        ):
            self.hvdc_reference_recorded = True
            _set_status(self.system, "PQ", self.hvdc.local_pq_id, 1)
            _set_status(self.system, "PQ", self.hvdc.remote_pq_id, 1)
            self._record_event(
                self.cfg.hvdc_reference_time_s,
                "OA4",
                "operator_action",
                (
                    "HVDC voltage-reference/setpoint adjustment before the "
                    "collector-side protection cascade."
                ),
            )

        if self.cfg.allow_equivalent_blackout:
            self._apply_blackout_disturbances(time_s)

    def _apply_blackout_disturbances(self, time_s: float) -> None:
        while (
            self.blackout_action_index < len(self.blackout_load_ids)
            and self.blackout_action_index < len(self.cfg.blackout_actions)
        ):
            action = self.cfg.blackout_actions[self.blackout_action_index]
            if time_s < action.time_s:
                return
            trip_count = sum(collector.tripped for collector in self.collectors)
            if trip_count < action.require_trip_count:
                return
            load_id = self.blackout_load_ids[self.blackout_action_index]
            _set_status(self.system, "PQ", load_id, 1)
            self._record_event(
                action.time_s,
                f"AD{self.blackout_action_index + 2}",
                "automatic_blackout",
                f"{action.description} ({load_id}).",
                net_load_increase_mw=action.p_mw,
                q_demand_increase_mvar=action.q_mvar,
            )
            self.blackout_action_index += 1

    def _apply_protection_chain(self, time_s: float, dt: float, system: Any) -> None:
        if dt <= 0.0 or self.blackout_criteria.declared:
            return
        physical_v = _active_physical_bus_voltage(system, self.physical_bus_count)
        vmin = float(np.nanmin(physical_v))
        freq_hz = _estimate_frequency_hz(system, self.cfg.nominal_frequency_hz)
        angle_sep = _coherent_angle_separation_rad(system, self.affected_bus_set)

        if not self.cfg.enable_post_cascade_relays:
            return
        self._apply_generator_protection(time_s, dt, system, vmin, freq_hz, angle_sep)
        self._apply_defense_stages(time_s, dt, system, vmin, freq_hz)
        self._apply_morocco_ac_trip(time_s, dt, system, freq_hz, angle_sep)
        self._apply_out_of_step_trip(time_s, dt, system, angle_sep)
        self._apply_hvdc_block(time_s, dt, system, physical_v, freq_hz)
        self._evaluate_island_blackout(time_s, dt, system)
        self._apply_blackout_criteria(time_s, dt, vmin, freq_hz, angle_sep)

    def _collector_trip_count(self) -> int:
        return int(sum(collector.tripped for collector in self.collectors))

    def _apply_hvdc_block(
        self,
        time_s: float,
        dt: float,
        system: Any,
        physical_v: np.ndarray,
        freq_hz: float,
    ) -> None:
        hvdc = self.hvdc
        spec = hvdc.spec
        if hvdc.blocked:
            return
        if self._collector_trip_count() < spec.require_collector_trips:
            hvdc.timer_s = 0.0
            return
        if spec.require_france_ac_separation and not self.out_of_step.tripped:
            hvdc.timer_s = 0.0
            return
        if time_s < spec.min_block_time_s:
            return
        bus_pos = {int(bus): i for i, bus in enumerate(system.Bus.idx.v)}
        local_v = float(system.Bus.v.v[bus_pos.get(spec.local_bus, 0)])
        islands = self._current_islands(system)
        local_island = next(
            (island for island in islands if spec.local_bus in island.buses),
            None,
        )
        local_frequency = (
            local_island.frequency_hz
            if local_island is not None and np.isfinite(local_island.frequency_hz)
            else freq_hz
        )
        local_vmin = (
            local_island.vmin_pu
            if local_island is not None and np.isfinite(local_island.vmin_pu)
            else float(np.nanmin(physical_v))
        )
        condition = (
            local_v <= spec.block_voltage_pu
            or local_v >= spec.block_over_voltage_pu
            or local_frequency <= spec.block_frequency_hz
            or local_vmin <= spec.block_voltage_pu
            or self.out_of_step.tripped
        )
        hvdc.timer_s = hvdc.timer_s + dt if condition else max(0.0, hvdc.timer_s - dt)
        if hvdc.timer_s < spec.dwell_s:
            return
        _set_status(system, "PQ", hvdc.local_pq_id, 0)
        _set_status(system, "PQ", hvdc.remote_pq_id, 0)
        hvdc.blocked = True
        hvdc.block_time_s = time_s
        self._record_event(
            time_s,
            "HVDC",
            "hvdc_block",
            (
                f"{spec.name} blocks on converter voltage/frequency logic "
                f"(Vlocal={local_v:.3f} pu, f={local_frequency:.2f} Hz)."
            ),
            q_demand_increase_mvar=spec.reactive_support_mvar,
            island_id=local_island.island_id if local_island else None,
            trigger="hvdc_terminal_voltage_frequency",
        )
        self._evaluate_island_blackout(time_s, dt, system)

    def _apply_out_of_step_trip(
        self,
        time_s: float,
        dt: float,
        system: Any,
        angle_sep: float,
    ) -> None:
        relay = self.out_of_step
        spec = relay.spec
        if relay.tripped:
            return
        if self._collector_trip_count() < spec.require_collector_trips:
            relay.timer_s = 0.0
            return
        if spec.require_morocco_trip and not self.morocco_ac.tripped:
            relay.timer_s = 0.0
            return
        if time_s < spec.min_trip_time_s:
            return
        threshold = np.deg2rad(spec.angle_threshold_deg)
        condition = angle_sep >= threshold
        relay.timer_s = relay.timer_s + dt if condition else max(0.0, relay.timer_s - dt)
        if relay.timer_s < spec.dwell_s:
            return
        tripped_lines: list[str] = []
        for line_id in relay.line_ids:
            if _safe_status(system, "Line", line_id) > 0.5:
                _set_status(system, "Line", line_id, 0)
                tripped_lines.append(str(line_id))
        relay.tripped = True
        relay.trip_time_s = time_s
        islands = self._current_islands(system)
        affected_island = self._affected_island(islands)
        self._record_event(
            time_s,
            "FR-AC",
            "out_of_step_trip",
            (
                "France-like AC loss-of-synchronism relay trips selected "
                "boundary tie-lines "
                f"(angle separation={np.rad2deg(angle_sep):.2f} deg, "
                f"lines={', '.join(tripped_lines[:5])}"
                f"{'...' if len(tripped_lines) > 5 else ''})."
            ),
            island_id=affected_island.island_id if affected_island else None,
            trigger="loss_of_synchronism_ac_separation",
        )

    def _apply_morocco_ac_trip(
        self,
        time_s: float,
        dt: float,
        system: Any,
        freq_hz: float,
        angle_sep: float,
    ) -> None:
        relay = self.morocco_ac
        spec = relay.spec
        if relay.tripped:
            return
        if self._collector_trip_count() < spec.require_collector_trips:
            relay.timer_s = 0.0
            return
        if spec.require_generator_trip and not self.generator_protection.tripped:
            relay.timer_s = 0.0
            return
        if time_s < spec.min_trip_time_s:
            return
        condition = (
            freq_hz <= spec.under_frequency_hz
            or angle_sep >= np.deg2rad(spec.angle_threshold_deg)
            or self.generator_protection.tripped
        )
        relay.timer_s = relay.timer_s + dt if condition else max(0.0, relay.timer_s - dt)
        if relay.timer_s < spec.dwell_s:
            return
        tripped_lines: list[str] = []
        for line_id in relay.line_ids:
            if _safe_status(system, "Line", line_id) > 0.5:
                _set_status(system, "Line", line_id, 0)
                tripped_lines.append(str(line_id))
        relay.tripped = True
        relay.trip_time_s = time_s
        islands = self._current_islands(system)
        affected_island = self._affected_island(islands)
        self._record_event(
            time_s,
            "MA-AC",
            "morocco_ac_trip",
            (
                "Morocco-like underfrequency AC interconnection relay trips "
                f"selected boundary tie-lines (f={freq_hz:.2f} Hz, "
                f"angle separation={np.rad2deg(angle_sep):.2f} deg, "
                f"lines={', '.join(tripped_lines[:5])}"
                f"{'...' if len(tripped_lines) > 5 else ''})."
            ),
            island_id=affected_island.island_id if affected_island else None,
            trigger="underfrequency_ac_interconnection_trip",
        )

    def _apply_generator_protection(
        self,
        time_s: float,
        dt: float,
        system: Any,
        vmin: float,
        freq_hz: float,
        angle_sep: float,
    ) -> None:
        relay = self.generator_protection
        spec = relay.spec
        if relay.tripped:
            return
        if self._collector_trip_count() < spec.require_collector_trips:
            relay.timer_s = 0.0
            return
        if time_s < spec.min_trip_time_s:
            return
        condition = (
            vmin <= spec.under_voltage_pu
            or freq_hz <= spec.under_frequency_hz
            or angle_sep >= np.deg2rad(spec.angle_threshold_deg)
            or self._collector_trip_count() >= spec.require_collector_trips
        )
        relay.timer_s = relay.timer_s + dt if condition else max(0.0, relay.timer_s - dt)
        if relay.timer_s < spec.dwell_s:
            return
        self._current_islands(system)
        tripped_genrou: list[str] = []
        lost_generation_mw = 0.0
        for gen_id in relay.genrou_ids:
            if _safe_status(system, "GENROU", gen_id) > 0.5:
                pos = _find_position(system.GENROU, gen_id)
                bus_id = int(system.GENROU.bus.v[pos])
                pe_pu = float(system.GENROU.Pe.v[pos]) if hasattr(system.GENROU, "Pe") else float("nan")
                pe_mw = pe_pu * float(system.config.mva) if np.isfinite(pe_pu) else 0.0
                if pe_mw <= 0.0:
                    pe_mw = self.area_selection.genrou_mva_by_id.get(str(gen_id), 0.0)
                _set_status(system, "GENROU", gen_id, 0)
                tripped_genrou.append(str(gen_id))
                lost_generation_mw += pe_mw
                relay.tripped_mva += self.area_selection.genrou_mva_by_id.get(str(gen_id), 0.0)
                self._append_deenergized_asset(
                    time_s,
                    self._island_id_for_bus(bus_id),
                    "GENROU",
                    str(gen_id),
                    bus_id,
                    "generator_uv_uf_loss_of_synchronism_protection",
                    lost_mw=pe_mw,
                )
        tripped_pv: list[str] = []
        for pv_id in relay.pv_ids:
            if _safe_status(system, "PV", pv_id) > 0.5:
                pos = _find_position(system.PV, pv_id)
                bus_id = int(system.PV.bus.v[pos])
                lost_pv_mw = max(0.0, float(system.PV.p0.v[pos]) * float(system.config.mva))
                _set_status(system, "PV", pv_id, 0)
                tripped_pv.append(str(pv_id))
                self._append_deenergized_asset(
                    time_s,
                    self._island_id_for_bus(bus_id),
                    "PV",
                    str(pv_id),
                    bus_id,
                    "plant_overvoltage_uv_uf_protection",
                    lost_mw=lost_pv_mw,
                )
        relay.tripped = True
        relay.trip_time_s = time_s
        self._record_event(
            time_s,
            "GEN",
            "generator_trip",
            (
                "Generator protection trips affected-area machines after "
                "the report-like collector cascade has removed generation and "
                f"reactive absorption (Vmin={vmin:.3f} pu, f={freq_hz:.2f} Hz, "
                f"angle={np.rad2deg(angle_sep):.2f} deg, "
                f"GENROU={len(tripped_genrou)}, PV={len(tripped_pv)})."
            ),
            lost_p_mw=lost_generation_mw,
            trigger="generator_uv_uf_loss_of_synchronism",
        )

    def _apply_defense_stages(
        self,
        time_s: float,
        dt: float,
        system: Any,
        vmin: float,
        freq_hz: float,
    ) -> None:
        if not self.generator_protection.tripped:
            return
        for stage in self.defense_stages:
            if stage.applied or time_s < stage.spec.min_time_s:
                continue
            condition = (
                vmin <= stage.spec.under_voltage_pu
                or freq_hz <= stage.spec.under_frequency_hz
                or self.generator_protection.tripped
            )
            stage.timer_s = stage.timer_s + dt if condition else max(0.0, stage.timer_s - dt)
            if stage.timer_s < stage.spec.dwell_s:
                continue
            islands = self._current_islands(system)
            affected_island = self._affected_island(islands)
            affected_buses = set(affected_island.buses) if affected_island else set()
            target_mw = min(
                stage.spec.max_shed_mw,
                stage.spec.load_fraction
                * sum(self.area_selection.load_mw_by_id.get(load_id, 0.0) for load_id in stage.load_ids),
            )
            shed_mw = 0.0
            shed_ids: list[str] = []
            for load_id in stage.load_ids:
                if shed_mw >= target_mw:
                    break
                if _safe_status(system, "PQ", load_id) <= 0.5:
                    continue
                pos = _find_position(system.PQ, load_id)
                bus_id = int(system.PQ.bus.v[pos])
                if affected_buses and bus_id not in affected_buses:
                    continue
                _set_status(system, "PQ", load_id, 0)
                shed_ids.append(str(load_id))
                shed_mw += self.area_selection.load_mw_by_id.get(str(load_id), 0.0)
                self._append_deenergized_asset(
                    time_s,
                    affected_island.island_id if affected_island else "",
                    "PQ",
                    str(load_id),
                    bus_id,
                    f"{stage.spec.name}_defense_load_shed",
                    lost_mw=self.area_selection.load_mw_by_id.get(str(load_id), 0.0),
                )
            stage.applied = True
            stage.applied_time_s = time_s
            stage.shed_mw = shed_mw
            success = bool(
                (np.isfinite(vmin) and vmin > stage.spec.under_voltage_pu)
                and freq_hz > stage.spec.under_frequency_hz
            )
            self._record_event(
                time_s,
                stage.spec.name,
                "defense_action",
                (
                    "UFLS/UVLS defense stage sheds affected-area demand but "
                    f"voltage/frequency remain in emergency range "
                    f"(shed={shed_mw:.1f} MW, loads={len(shed_ids)})."
                ),
                net_load_increase_mw=-shed_mw,
                island_id=affected_island.island_id if affected_island else None,
                trigger="ufls_uvls",
                success=success,
            )

    def _apply_blackout_criteria(
        self,
        time_s: float,
        dt: float,
        vmin: float,
        freq_hz: float,
        angle_sep: float,
    ) -> None:
        blackout = self.blackout_criteria
        spec = blackout.spec
        if blackout.declared:
            return
        if self.island_blackout.deenergized_islands:
            return
        if not self.hvdc.blocked:
            blackout.voltage_timer_s = 0.0
            blackout.frequency_timer_s = 0.0
            blackout.angle_timer_s = 0.0
            return
        blackout.voltage_timer_s = (
            blackout.voltage_timer_s + dt
            if vmin <= spec.voltage_pu
            else max(0.0, blackout.voltage_timer_s - dt)
        )
        blackout.frequency_timer_s = (
            blackout.frequency_timer_s + dt
            if freq_hz <= spec.frequency_hz
            else max(0.0, blackout.frequency_timer_s - dt)
        )
        angle_condition = (
            self.out_of_step.tripped
            and self.hvdc.blocked
            and angle_sep >= np.deg2rad(spec.angle_threshold_deg)
        )
        blackout.angle_timer_s = (
            blackout.angle_timer_s + dt
            if angle_condition
            else max(0.0, blackout.angle_timer_s - dt)
        )
        reason: str | None = None
        if blackout.voltage_timer_s >= spec.voltage_dwell_s:
            reason = f"physical-network voltage below {spec.voltage_pu:.2f} pu"
        elif blackout.frequency_timer_s >= spec.frequency_dwell_s:
            reason = f"frequency below {spec.frequency_hz:.1f} Hz"
        elif blackout.angle_timer_s >= spec.angle_dwell_s:
            reason = (
                f"coherent-area angle separation above {spec.angle_threshold_deg:.1f} deg"
            )
        if reason is None:
            return
        blackout.declared = True
        blackout.declared_time_s = time_s
        blackout.reason = reason
        self._record_event(
            time_s,
            "BO",
            "system_blackout_declared",
            f"Blackout endpoint declared by criteria: {reason}.",
            trigger="global_blackout_criteria",
        )

    def _ignored_lines_for_island_accounting(self) -> set[str]:
        if not self.out_of_step.tripped:
            return set()
        return set(self.area_selection.boundary_line_ids)

    def _current_islands(self, system: Any) -> list[IslandSnapshot]:
        snapshots = compute_island_snapshots(
            system,
            collectors=self.collectors,
            affected_buses=self.affected_bus_set,
            ignored_line_ids=self._ignored_lines_for_island_accounting(),
            deenergized_islands=self.island_blackout.deenergized_islands,
            nominal_frequency_hz=self.cfg.nominal_frequency_hz,
        )
        self.last_islands = snapshots
        return snapshots

    def _affected_island(self, islands: list[IslandSnapshot]) -> IslandSnapshot | None:
        affected = [item for item in islands if item.affected_bus_count > 0]
        if not affected:
            return None
        affected.sort(
            key=lambda item: (
                -item.affected_bus_count,
                -len(item.online_collector_names),
                -item.load_mw,
                item.island_id,
            )
        )
        return affected[0]

    def _evaluate_island_blackout(
        self,
        time_s: float,
        dt: float,
        system: Any,
    ) -> None:
        if self.blackout_criteria.declared or not self.cfg.enable_post_cascade_relays:
            return
        if not self.hvdc.blocked or self.hvdc.block_time_s is None:
            return
        if (
            time_s - float(self.hvdc.block_time_s)
            < self.cfg.island_blackout_criteria.min_time_after_hvdc_block_s
        ):
            return
        islands = self._current_islands(system)
        island = self._affected_island(islands)
        if island is None or island.island_id in self.island_blackout.deenergized_islands:
            return
        spec = self.island_blackout.spec
        island_id = island.island_id
        no_reference = (
            (not island.has_voltage_reference)
            or island.generation_mw < spec.min_generation_mw
        )
        low_voltage = np.isfinite(island.vmin_pu) and island.vmin_pu <= spec.voltage_pu
        low_frequency = (
            np.isfinite(island.frequency_hz)
            and island.frequency_hz <= spec.frequency_hz
        )
        high_angle = island.angle_spread_deg >= spec.angle_threshold_deg
        all_defense_applied = all(stage.applied for stage in self.defense_stages)
        imbalance = (
            all_defense_applied
            and island.imbalance_fraction >= spec.imbalance_fraction
        )
        runtime = self.island_blackout
        runtime.no_reference_timers_s[island_id] = (
            runtime.no_reference_timers_s.get(island_id, 0.0) + dt
            if no_reference
            else 0.0
        )
        runtime.voltage_timers_s[island_id] = (
            runtime.voltage_timers_s.get(island_id, 0.0) + dt if low_voltage else 0.0
        )
        runtime.frequency_timers_s[island_id] = (
            runtime.frequency_timers_s.get(island_id, 0.0) + dt if low_frequency else 0.0
        )
        runtime.angle_timers_s[island_id] = (
            runtime.angle_timers_s.get(island_id, 0.0) + dt if high_angle else 0.0
        )
        runtime.imbalance_timers_s[island_id] = (
            runtime.imbalance_timers_s.get(island_id, 0.0) + dt if imbalance else 0.0
        )
        reason: str | None = None
        trigger: str | None = None
        if runtime.no_reference_timers_s[island_id] >= spec.voltage_dwell_s:
            reason = "affected island has no viable voltage/frequency reference"
            trigger = "no_reference"
        elif runtime.voltage_timers_s[island_id] >= spec.voltage_dwell_s:
            reason = f"affected island voltage below {spec.voltage_pu:.2f} pu"
            trigger = "island_undervoltage"
        elif runtime.frequency_timers_s[island_id] >= spec.frequency_dwell_s:
            reason = f"affected island frequency below {spec.frequency_hz:.1f} Hz"
            trigger = "island_underfrequency"
        elif runtime.angle_timers_s[island_id] >= spec.angle_dwell_s:
            reason = (
                "affected island loss-of-synchronism angle spread "
                f"{island.angle_spread_deg:.1f} deg"
            )
            trigger = "island_angle_spread"
        elif runtime.imbalance_timers_s[island_id] >= spec.imbalance_dwell_s:
            reason = (
                "affected island load-generation imbalance remains too large "
                "after UFLS/UVLS"
            )
            trigger = "post_defense_imbalance"
        if reason is None or trigger is None:
            return
        self._deenergize_island(time_s, system, island, reason, trigger)

    def _deenergize_island(
        self,
        time_s: float,
        system: Any,
        island: IslandSnapshot,
        reason: str,
        trigger: str,
    ) -> None:
        island_id = island.island_id
        buses = set(island.buses)
        self.island_blackout.deenergized_islands.add(island_id)
        self._record_event(
            time_s,
            f"{island_id}-BO",
            "island_blackout_declared",
            (
                f"{island_id} de-energized: {reason} "
                f"(buses={len(buses)}, load={island.load_mw:.1f} MW, "
                f"generation={island.generation_mw:.1f} MW)."
            ),
            island_id=island_id,
            trigger=trigger,
        )
        self._trip_assets_in_island(time_s, system, island, reason)
        self.blackout_criteria.declared = True
        self.blackout_criteria.declared_time_s = time_s
        self.blackout_criteria.reason = reason
        self._record_event(
            time_s,
            "BO",
            "system_blackout_declared",
            (
                "System blackout endpoint declared after explicit island "
                f"de-energization ({island_id}): {reason}."
            ),
            island_id=island_id,
            trigger=trigger,
        )

    def _append_deenergized_asset(
        self,
        time_s: float,
        island_id: str,
        model: str,
        device_id: str,
        bus_id: int | None,
        reason: str,
        *,
        lost_mw: float = 0.0,
        q_mvar: float = 0.0,
    ) -> None:
        self.deenergized_assets.append(
            {
                "time_s": float(time_s),
                "island_id": island_id,
                "model": model,
                "device_id": str(device_id),
                "bus_id": bus_id,
                "reason": reason,
                "lost_mw": float(lost_mw),
                "q_mvar": float(q_mvar),
            }
        )

    def _island_id_for_bus(self, bus_id: int) -> str:
        islands = self.last_islands or self._current_islands(self.system)
        for island in islands:
            if int(bus_id) in island.buses:
                return island.island_id
        return ""

    def _trip_assets_in_island(
        self,
        time_s: float,
        system: Any,
        island: IslandSnapshot,
        reason: str,
    ) -> None:
        buses = set(island.buses)
        island_id = island.island_id
        base = _base_mva(system)
        for model_name in ("GENROU", "PV", "PQ", "Shunt", "Line"):
            if not hasattr(system, model_name):
                continue
            model = getattr(system, model_name)
            for i in range(getattr(model, "n", 0)):
                if _model_status_at(model, i) <= 0.5:
                    continue
                if model_name == "Line":
                    bus1 = int(model.bus1.v[i])
                    bus2 = int(model.bus2.v[i])
                    in_island = bus1 in buses or bus2 in buses
                    bus_id = bus1 if bus1 in buses else bus2 if bus2 in buses else None
                else:
                    bus_id = int(model.bus.v[i])
                    in_island = bus_id in buses
                if not in_island:
                    continue
                device_id = str(model.idx.v[i])
                lost_mw = 0.0
                q_mvar = 0.0
                if model_name == "GENROU":
                    lost_mw = _generator_output_mw(system, i)
                elif model_name == "PV" and hasattr(model, "p0"):
                    lost_mw = max(0.0, float(model.p0.v[i]) * base)
                elif model_name == "PQ":
                    p_mw = float(model.p0.v[i]) * base if hasattr(model, "p0") else 0.0
                    q_mvar = float(model.q0.v[i]) * base if hasattr(model, "q0") else 0.0
                    lost_mw = abs(p_mw)
                elif model_name == "Shunt" and hasattr(model, "b"):
                    q_mvar = float(model.b.v[i]) * base
                _set_status(system, model_name, device_id, 0)
                self._append_deenergized_asset(
                    time_s,
                    island_id,
                    model_name,
                    device_id,
                    bus_id,
                    reason,
                    lost_mw=lost_mw,
                    q_mvar=q_mvar,
                )
        for collector in self.collectors:
            if collector.collector_bus not in buses or collector.tripped:
                continue
            collector.tripped = True
            collector.trip_time_s = time_s
            collector.trip_stage_name = "island_deenergization"
            self._append_deenergized_asset(
                time_s,
                island_id,
                "Collector",
                collector.spec.name,
                collector.collector_bus,
                reason,
                lost_mw=collector.spec.active_power_mw,
                q_mvar=collector.spec.q_absorption_mvar,
            )

    def _monitor_collectors(self, time_s: float, dt: float, system: Any) -> None:
        bus_pos = {int(bus): i for i, bus in enumerate(system.Bus.idx.v)}
        row: dict[str, float | int | str] = {"time_s": time_s}

        for collector in self.collectors:
            coll_pos = bus_pos[collector.collector_bus]
            trans_pos = bus_pos[collector.spec.transmission_bus]
            voltage = float(system.Bus.v.v[coll_pos])
            transmission_voltage = float(system.Bus.v.v[trans_pos])
            collector.last_voltage_pu = voltage
            collector.last_transmission_voltage_pu = transmission_voltage
            if (not collector.tripped) and voltage > collector.max_voltage_pu:
                collector.max_voltage_pu = voltage
                collector.max_voltage_time_s = time_s

            row[f"{collector.spec.name}_voltage_pu_raw"] = voltage
            row[f"{collector.spec.name}_voltage_pu"] = "" if collector.tripped else voltage
            row[f"{collector.spec.name}_transmission_voltage_pu"] = transmission_voltage
            row[f"{collector.spec.name}_threshold_pu"] = collector.spec.threshold_pu
            row[f"{collector.spec.name}_online"] = 0 if collector.tripped else 1

            if collector.tripped or dt <= 0.0:
                continue
            trip_stage = update_overvoltage_relay_timers(
                collector,
                voltage,
                dt,
                self.cfg.relay_timer_decay,
            )
            collector.timer_s = max(collector.stage_timers_s.values(), default=0.0)
            collector.fast_timer_s = collector.stage_timers_s.get("high", 0.0)
            if trip_stage is not None and time_s >= collector.spec.min_trip_time_s:
                self._trip_collector(collector, time_s, trip_stage)

        self.collector_trace.append(row)

    def _apply_fixed_pf_export_ramp(self, time_s: float) -> None:
        start = self.cfg.export_reduction_time_s
        duration = max(float(self.cfg.export_reduction_ramp_s), 1e-9)
        if (
            time_s < start
            or self.export_reduction_done
            or self.export_reduction_total_mvar <= 0.0
        ):
            return

        progress = min(1.0, max(0.0, (time_s - start) / duration))
        if not self.export_reduction_started:
            self.export_reduction_started = True
            self._record_event(
                start,
                "OA2",
                "operator_action",
                (
                    "Export reduction and fixed-power-factor plant behavior begin "
                    "a ramped loss of reactive absorption."
                ),
            )

        if progress <= self.export_reduction_progress:
            return

        shunt_model = self.system.Shunt
        base_mva = float(self.system.config.mva)
        remaining_fraction = max(0.0, 1.0 - progress)
        if self.cfg.export_reduction_mode == "collector_fixed_pf":
            for collector in self.collectors:
                pos = _find_position(shunt_model, collector.fixed_pf_q_shunt_id)
                shunt_model.b.v[pos] = -(
                    collector.fixed_pf_q_absorption_mvar * remaining_fraction / base_mva
                )
                if (
                    progress >= 1.0
                    and _status(self.system, "Shunt", collector.fixed_pf_q_shunt_id) > 0.5
                ):
                    _set_status(self.system, "Shunt", collector.fixed_pf_q_shunt_id, 0)
        else:
            if self.background_fixed_pf_shunt_id is None:
                return
            pos = _find_position(shunt_model, self.background_fixed_pf_shunt_id)
            shunt_model.b.v[pos] = -(
                self.export_reduction_total_mvar * remaining_fraction / base_mva
            )
            if (
                progress >= 1.0
                and _status(self.system, "Shunt", self.background_fixed_pf_shunt_id) > 0.5
            ):
                _set_status(self.system, "Shunt", self.background_fixed_pf_shunt_id, 0)

        incremental_lost_q = self.export_reduction_total_mvar * (
            progress - self.export_reduction_progress
        )
        self.cumulative_lost_q_absorption_mvar += float(incremental_lost_q)
        self.export_reduction_progress = progress
        self.export_reduction_done = progress >= 1.0

    def _trip_collector(
        self,
        collector: CollectorRuntime,
        time_s: float,
        trip_stage_name: str | None = None,
    ) -> None:
        collector.tripped = True
        collector.trip_time_s = time_s
        collector.trip_voltage_pu = collector.last_voltage_pu
        collector.trip_stage_name = trip_stage_name
        _set_status(self.system, "PQ", collector.pq_id, 0)

        lost_q = 0.0
        if _status(self.system, "Shunt", collector.main_q_shunt_id) > 0.5:
            _set_status(self.system, "Shunt", collector.main_q_shunt_id, 0)
            lost_q += collector.main_q_absorption_mvar
        if _status(self.system, "Shunt", collector.fixed_pf_q_shunt_id) > 0.5:
            shunt = self.system.Shunt
            pos = _find_position(shunt, collector.fixed_pf_q_shunt_id)
            _set_status(self.system, "Shunt", collector.fixed_pf_q_shunt_id, 0)
            lost_q += max(0.0, -float(shunt.b.v[pos]) * float(self.system.config.mva))

        self._record_event(
            time_s,
            f"AA_{collector.spec.name}",
            "protection_trip",
            (
                f"{collector.spec.name} trips by collector-side overvoltage "
                f"(V={collector.last_voltage_pu:.3f} pu, "
                f"threshold={collector.spec.threshold_pu:.3f} pu, "
                f"stage={trip_stage_name or 'legacy'}, "
                f"cluster={collector.spec.report_cluster})."
            ),
            lost_p_mw=collector.spec.active_power_mw,
            lost_q_absorption_mvar=lost_q,
        )

    def _record_system_trace(self, time_s: float, system: Any) -> None:
        physical_v = np.asarray(system.Bus.v.v[: self.physical_bus_count], dtype=float)
        physical_v_finite = physical_v[np.isfinite(physical_v)]
        if physical_v_finite.size:
            physical_percentiles = np.nanpercentile(
                physical_v_finite,
                [1.0, 10.0, 50.0, 90.0, 99.0],
            )
        else:
            physical_percentiles = np.array([np.nan, np.nan, np.nan, np.nan, np.nan])
        frequency_hz = _estimate_frequency_hz(system, self.cfg.nominal_frequency_hz)
        angle_sep = _coherent_angle_separation_rad(system, self.affected_bus_set)
        islands = self._current_islands(system)
        affected_island = self._affected_island(islands)
        for island in islands:
            self.island_trace.append(
                {
                    "time_s": float(time_s),
                    "island_id": island.island_id,
                    "bus_count": len(island.buses),
                    "affected_bus_count": island.affected_bus_count,
                    "line_count": len(island.line_ids),
                    "online_genrou_count": len(island.online_genrou_ids),
                    "online_pv_count": len(island.online_pv_ids),
                    "online_load_count": len(island.online_load_ids),
                    "online_collector_count": len(island.online_collector_names),
                    "generation_mw": island.generation_mw,
                    "load_mw": island.load_mw,
                    "q_load_mvar": island.q_load_mvar,
                    "q_absorption_mvar": island.q_absorption_mvar,
                    "vmin_pu": float("nan") if island.deenergized else island.vmin_pu,
                    "vmax_pu": float("nan") if island.deenergized else island.vmax_pu,
                    "frequency_hz": float("nan")
                    if island.deenergized
                    else island.frequency_hz,
                    "angle_spread_deg": float("nan")
                    if island.deenergized
                    else island.angle_spread_deg,
                    "has_voltage_reference": int(island.has_voltage_reference),
                    "imbalance_fraction": island.imbalance_fraction,
                    "deenergized": int(island.deenergized),
                }
            )
        affected_generation = affected_island.generation_mw if affected_island else float("nan")
        affected_load = affected_island.load_mw if affected_island else float("nan")
        affected_vmin = affected_island.vmin_pu if affected_island else float("nan")
        affected_frequency = affected_island.frequency_hz if affected_island else float("nan")
        self.system_trace.append(
            {
                "time_s": time_s,
                "vmax_physical_pu": float(np.nanmax(physical_v)),
                "vmin_physical_pu": float(np.nanmin(physical_v)),
                "v01_physical_pu": float(physical_percentiles[0]),
                "v10_physical_pu": float(physical_percentiles[1]),
                "v50_physical_pu": float(physical_percentiles[2]),
                "v90_physical_pu": float(physical_percentiles[3]),
                "v99_physical_pu": float(physical_percentiles[4]),
                "frequency_hz": float(frequency_hz),
                "angle_separation_deg": float(np.rad2deg(angle_sep)),
                "affected_island_id": affected_island.island_id if affected_island else "",
                "affected_island_generation_mw": float(affected_generation),
                "affected_island_load_mw": float(affected_load),
                "affected_island_vmin_pu": float(affected_vmin),
                "affected_island_frequency_hz": float(affected_frequency),
                "deenergized_island_count": int(len(self.island_blackout.deenergized_islands)),
                "online_collectors": int(sum(not item.tripped for item in self.collectors)),
                "hvdc_blocked": int(self.hvdc.blocked),
                "morocco_ac_tripped": int(self.morocco_ac.tripped),
                "boundary_lines_tripped": int(
                    self.morocco_ac.tripped or self.out_of_step.tripped
                ),
                "france_ac_tripped": int(self.out_of_step.tripped),
                "generator_protection_tripped": int(self.generator_protection.tripped),
                "defense_stages_applied": int(sum(stage.applied for stage in self.defense_stages)),
                "blackout_declared": int(self.blackout_criteria.declared),
                "cumulative_lost_p_mw": float(self.cumulative_lost_p_mw),
                "cumulative_lost_q_absorption_mvar": float(
                    self.cumulative_lost_q_absorption_mvar
                ),
                "cumulative_blackout_load_mw": float(self.cumulative_blackout_load_mw),
                "cumulative_blackout_q_demand_mvar": float(
                    self.cumulative_blackout_q_demand_mvar
                ),
            }
        )


def build_replication_system(
    cfg: ReplicationConfig,
) -> tuple[
    Any,
    list[CollectorRuntime],
    list[str],
    list[str],
    AffectedAreaSelection,
    HVDCSurrogateRuntime,
    UnderFrequencyTieRelayRuntime,
    OutOfStepRelayRuntime,
    GeneratorProtectionRelayRuntime,
    list[UFLSUVLSStageRuntime],
    BlackoutCriteriaRuntime,
    IslandBlackoutRuntime,
    list[str],
    str | None,
    int,
]:
    case = AndesCase.load(
        AndesCaseSpec(cfg.case_id, setup=False, run_pflow=False, init_tds=False)
    )
    system = case.system
    _disable_predefined_events(system)

    base_mva = float(system.config.mva)
    physical_bus_count = int(system.Bus.n)
    area_selection = select_affected_area(system, cfg)
    collector_specs = resolve_collector_specs(system, cfg, area_selection)

    collectors: list[CollectorRuntime] = []
    next_bus = int(max(system.Bus.idx.v)) + 1
    for i, spec in enumerate(collector_specs):
        collector_bus = next_bus + i
        system.add(
            "Bus",
            {
                "idx": collector_bus,
                "name": f"{spec.name}_collector",
                "Vn": spec.base_kv,
                "v0": 1.0,
                "a0": 0.0,
            },
        )
        gsu_line_id = system.add(
            "Line",
            {
                "bus1": spec.transmission_bus,
                "bus2": collector_bus,
                "r": 0.001,
                "x": 0.08,
                "b": 0.0,
                "tap": spec.tap,
                "u": 1,
                "name": f"GSU_{spec.name}",
            },
        )
        pq_id = system.add(
            "PQ",
            {
                "bus": collector_bus,
                "p0": -spec.active_power_mw / base_mva,
                "q0": 0.0,
                "u": 1,
                "name": f"IBR_{spec.name}",
            },
        )
        main_q = spec.q_absorption_mvar * (1.0 - spec.fixed_pf_q_fraction)
        fixed_q = spec.q_absorption_mvar * spec.fixed_pf_q_fraction
        main_q_shunt_id = system.add(
            "Shunt",
            {
                "bus": collector_bus,
                "g": 0.0,
                "b": -main_q / base_mva,
                "u": 1,
                "name": f"{spec.name}_QABS_MAIN",
            },
        )
        fixed_pf_q_shunt_id = system.add(
            "Shunt",
            {
                "bus": collector_bus,
                "g": 0.0,
                "b": -fixed_q / base_mva,
                "u": 1,
                "name": f"{spec.name}_QABS_FIXEDPF",
            },
        )
        collectors.append(
            CollectorRuntime(
                spec=spec,
                collector_bus=collector_bus,
                gsu_line_id=str(gsu_line_id),
                pq_id=str(pq_id),
                main_q_shunt_id=str(main_q_shunt_id),
                fixed_pf_q_shunt_id=str(fixed_pf_q_shunt_id),
            )
        )

    mesh_line_ids: list[str] = []
    for i, source_id in enumerate(cfg.meshing_source_lines):
        line = system.Line
        pos = _find_position(line, source_id)
        strength = (
            float(cfg.meshing_equivalent_strengths[i])
            if i < len(cfg.meshing_equivalent_strengths)
            else 1.0
        )
        strength = max(strength, 1e-6)
        mesh_line_ids.append(
            str(
                system.add(
                    "Line",
                    {
                        "bus1": int(line.bus1.v[pos]),
                        "bus2": int(line.bus2.v[pos]),
                        "r": float(line.r.v[pos]) / strength,
                        "x": float(line.x.v[pos]) / strength,
                        "b": float(line.b.v[pos]) * strength,
                        "tap": float(line.tap.v[pos]),
                        "u": 0,
                        "name": f"MESH_{source_id}",
                    },
                )
            )
        )

    reactor_ids: list[str] = []
    for i, action in enumerate(cfg.reactor_actions, start=1):
        initial_status = 0 if int(action.final_status) == 1 else 1
        reactor_ids.append(
            str(
                system.add(
                    "Shunt",
                    {
                        "bus": action.bus,
                        "g": 0.0,
                        "b": -action.q_mvar / base_mva,
                        "u": initial_status,
                        "name": f"REPORT_REACTOR_{i}",
                    },
                )
            )
        )

    hvdc_spec = cfg.hvdc_surrogate
    hvdc_local_pq_id = str(
        system.add(
            "PQ",
            {
                "bus": hvdc_spec.local_bus,
                "p0": hvdc_spec.transfer_mw / base_mva,
                "q0": -hvdc_spec.reactive_support_mvar / base_mva,
                "u": 0,
                "name": f"{hvdc_spec.name}_LOCAL",
            },
        )
    )
    hvdc_remote_pq_id = str(
        system.add(
            "PQ",
            {
                "bus": hvdc_spec.remote_bus,
                "p0": -hvdc_spec.transfer_mw / base_mva,
                "q0": 0.0,
                "u": 0,
                "name": f"{hvdc_spec.name}_REMOTE",
            },
        )
    )
    hvdc_runtime = HVDCSurrogateRuntime(
        spec=hvdc_spec,
        local_pq_id=hvdc_local_pq_id,
        remote_pq_id=hvdc_remote_pq_id,
    )

    morocco_runtime = UnderFrequencyTieRelayRuntime(
        spec=cfg.morocco_ac_relay,
        line_ids=area_selection.boundary_line_ids[
            : cfg.morocco_ac_relay.boundary_line_count
        ],
    )
    out_of_step_runtime = OutOfStepRelayRuntime(
        spec=cfg.out_of_step_relay,
        line_ids=area_selection.boundary_line_ids[
            cfg.morocco_ac_relay.boundary_line_count : cfg.morocco_ac_relay.boundary_line_count
            + cfg.out_of_step_relay.boundary_line_count
        ],
    )
    generator_runtime = GeneratorProtectionRelayRuntime(
        spec=cfg.generator_protection,
        genrou_ids=list(area_selection.protected_genrou_ids),
        pv_ids=list(area_selection.affected_pv_ids),
    )
    defense_runtimes = [
        UFLSUVLSStageRuntime(spec=stage, load_ids=list(area_selection.selected_load_ids))
        for stage in cfg.defense_stages
    ]
    blackout_runtime = BlackoutCriteriaRuntime(spec=cfg.blackout_criteria)
    island_blackout_runtime = IslandBlackoutRuntime(spec=cfg.island_blackout_criteria)

    blackout_load_ids: list[str] = []
    if cfg.allow_equivalent_blackout:
        for i, action in enumerate(cfg.blackout_actions, start=1):
            blackout_load_ids.append(
                str(
                    system.add(
                        "PQ",
                        {
                            "bus": action.bus,
                            "p0": action.p_mw / base_mva,
                            "q0": action.q_mvar / base_mva,
                            "u": 0,
                            "name": f"BLACKOUT_IMBALANCE_{i}",
                        },
                    )
                )
            )

    background_fixed_pf_shunt_id: str | None = None
    if cfg.background_fixed_pf_q_absorption_mvar > 0.0:
        background_fixed_pf_shunt_id = str(
            system.add(
                "Shunt",
                {
                    "bus": cfg.background_fixed_pf_bus,
                    "g": 0.0,
                    "b": -cfg.background_fixed_pf_q_absorption_mvar / base_mva,
                    "u": 1,
                    "name": "BACKGROUND_FIXEDPF_Q_ABS",
                },
            )
        )
    return (
        system,
        collectors,
        mesh_line_ids,
        reactor_ids,
        area_selection,
        hvdc_runtime,
        morocco_runtime,
        out_of_step_runtime,
        generator_runtime,
        defense_runtimes,
        blackout_runtime,
        island_blackout_runtime,
        blackout_load_ids,
        background_fixed_pf_shunt_id,
        physical_bus_count,
    )


def run_replication(
    cfg: ReplicationConfig,
    *,
    out_dir: Path,
    quiet_andes: bool = True,
) -> tuple[ReplicationArtifacts, dict[str, Any], IberianReplicationMonitor]:
    out_dir.mkdir(parents=True, exist_ok=True)
    (
        system,
        collectors,
        mesh_ids,
        reactor_ids,
        area_selection,
        hvdc,
        morocco_ac,
        out_of_step,
        generator_protection,
        defense_stages,
        blackout_criteria,
        island_blackout,
        blackout_load_ids,
        background_fixed_pf_id,
        physical_bus_count,
    ) = build_replication_system(cfg)

    with QuietAndes(quiet_andes):
        system.setup()
        system.PFlow.config.max_iter = 100
        system.PFlow.config.tol = cfg.pflow_tol
        pflow_ok = bool(system.PFlow.run())
        if not pflow_ok:
            system.Bus.v0.v[:] = 1.0
            system.Bus.a0.v[:] = 0.0
            pflow_ok = bool(system.PFlow.run())
    if not pflow_ok:
        raise RuntimeError("ACTIVSg2000 replication power flow did not converge.")

    monitor = IberianReplicationMonitor(
        system,
        cfg,
        collectors,
        mesh_ids,
        reactor_ids,
        area_selection,
        hvdc,
        morocco_ac,
        out_of_step,
        generator_protection,
        defense_stages,
        blackout_criteria,
        island_blackout,
        blackout_load_ids,
        background_fixed_pf_id,
        physical_bus_count,
    )

    system.TDS.config.tf = cfg.tf_s
    system.TDS.config.tstep = cfg.tstep_s
    system.TDS.config.tol = cfg.tds_tol
    system.TDS.config.max_iter = 25
    system.TDS.config.criteria = 0
    system.TDS.config.verbose = 0
    system.TDS.callpert = monitor

    with QuietAndes(quiet_andes):
        system.TDS.init()
        tds_ok = bool(system.TDS.run())
    if not tds_ok and not monitor.blackout_criteria.declared:
        failure_time = float(system.dae.t)
        monitor.blackout_criteria.declared = True
        monitor.blackout_criteria.declared_time_s = failure_time
        monitor.blackout_criteria.reason = "ANDES TDS stopped after explicit protection actions"
        monitor._record_event(
            failure_time,
            "BO",
            "system_blackout_declared",
            (
                "Blackout endpoint declared because the dynamic simulation "
                "did not maintain a valid continuation after the explicit "
                "protection/control chain."
            ),
            trigger="tds_non_continuation",
        )

    summary = _build_summary(cfg, system, collectors, monitor, pflow_ok, tds_ok)
    artifacts = _write_outputs(out_dir, cfg, collectors, monitor, summary)
    return artifacts, summary, monitor


def _build_summary(
    cfg: ReplicationConfig,
    system: Any,
    collectors: list[CollectorRuntime],
    monitor: IberianReplicationMonitor,
    pflow_ok: bool,
    tds_ok: bool,
) -> dict[str, Any]:
    trip_events = [event for event in monitor.events if event.category == "protection_trip"]
    hvdc_events = [event for event in monitor.events if event.category == "hvdc_block"]
    morocco_events = [event for event in monitor.events if event.category == "morocco_ac_trip"]
    tie_events = [event for event in monitor.events if event.category == "out_of_step_trip"]
    gen_events = [event for event in monitor.events if event.category == "generator_trip"]
    defense_events = [event for event in monitor.events if event.category == "defense_action"]
    island_blackout_events = [
        event for event in monitor.events if event.category == "island_blackout_declared"
    ]
    system_blackout_events = [
        event for event in monitor.events if event.category == "system_blackout_declared"
    ]
    legacy_declared_events = [
        event for event in monitor.events if event.category == "blackout_declared"
    ]
    declared_events = system_blackout_events or legacy_declared_events
    system_trace = monitor.system_trace
    min_physical_voltage = min((row["vmin_physical_pu"] for row in system_trace), default=float("nan"))
    max_physical_voltage = max((row["vmax_physical_pu"] for row in system_trace), default=float("nan"))
    blackout_time = min((event.time_s for event in declared_events), default=None)
    if blackout_time is None:
        valid_trace = system_trace
    else:
        valid_trace = [
            row for row in system_trace if float(row["time_s"]) <= float(blackout_time)
        ]
    min_voltage_before_blackout = min(
        (row["vmin_physical_pu"] for row in valid_trace),
        default=float("nan"),
    )
    crash_detected = bool(
        np.isfinite(min_physical_voltage)
        and min_physical_voltage <= cfg.blackout_criteria.voltage_pu
    )
    blackout_events = [event for event in monitor.events if event.category == "automatic_blackout"]
    first_trip = min((event.time_s for event in trip_events), default=None)
    last_trip = max((event.time_s for event in trip_events), default=None)
    return {
        "case_id": cfg.case_id,
        "pflow_ok": pflow_ok,
        "tds_ok": tds_ok,
        "final_time_s": float(system.dae.t),
        "physical_bus_count": int(getattr(system.Bus, "n", 0)) - len(collectors),
        "total_bus_count_with_collectors": int(getattr(system.Bus, "n", 0)),
        "collector_count": len(collectors),
        "collector_trips": len(trip_events),
        "first_trip_time_s": first_trip,
        "last_trip_time_s": last_trip,
        "first_hvdc_block_time_s": min((event.time_s for event in hvdc_events), default=None),
        "first_morocco_ac_trip_time_s": min((event.time_s for event in morocco_events), default=None),
        "first_france_ac_separation_time_s": min((event.time_s for event in tie_events), default=None),
        "first_tie_trip_time_s": min((event.time_s for event in tie_events), default=None),
        "first_generator_protection_trip_time_s": min((event.time_s for event in gen_events), default=None),
        "first_defense_action_time_s": min((event.time_s for event in defense_events), default=None),
        "blackout_time_s": blackout_time,
        "blackout_reason": monitor.blackout_criteria.reason,
        "final_valid_time_s": blackout_time if blackout_time is not None else float(system.dae.t),
        "total_lost_p_mw": float(monitor.cumulative_lost_p_mw),
        "total_lost_q_absorption_mvar": float(monitor.cumulative_lost_q_absorption_mvar),
        "automatic_blackout_actions": len(blackout_events),
        "island_blackout_events": len(island_blackout_events),
        "system_blackout_events": len(system_blackout_events),
        "deenergized_island_count": len(monitor.island_blackout.deenergized_islands),
        "deenergized_asset_count": len(monitor.deenergized_assets),
        "equivalent_blackout_enabled": bool(cfg.allow_equivalent_blackout),
        "post_cascade_relays_enabled": bool(cfg.enable_post_cascade_relays),
        "hvdc_blocks": len(hvdc_events),
        "morocco_ac_trip_events": len(morocco_events),
        "tie_line_separation_events": len(tie_events),
        "generator_protection_events": len(gen_events),
        "defense_actions": len(defense_events),
        "total_blackout_load_increase_mw": float(monitor.cumulative_blackout_load_mw),
        "total_blackout_q_demand_increase_mvar": float(
            monitor.cumulative_blackout_q_demand_mvar
        ),
        "min_physical_voltage_pu": float(min_physical_voltage),
        "min_physical_voltage_before_blackout_pu": float(min_voltage_before_blackout),
        "max_physical_voltage_pu": float(max_physical_voltage),
        "voltage_crash_detected": crash_detected,
        "blackout_detected": bool(declared_events or crash_detected or not tds_ok),
        "affected_area": {
            "radius": monitor.area_selection.radius,
            "seed_buses": monitor.area_selection.seed_buses,
            "affected_bus_count": len(monitor.area_selection.affected_buses),
            "boundary_line_count": len(monitor.area_selection.boundary_line_ids),
            "selected_morocco_ac_lines": monitor.morocco_ac.line_ids,
            "selected_france_ac_lines": monitor.out_of_step.line_ids,
            "selected_boundary_lines": monitor.morocco_ac.line_ids + monitor.out_of_step.line_ids,
            "affected_genrou_count": len(monitor.area_selection.affected_genrou_ids),
            "protected_genrou_ids": monitor.generator_protection.genrou_ids,
            "affected_pv_ids": monitor.generator_protection.pv_ids,
            "selected_load_count": len(monitor.area_selection.selected_load_ids),
            "deenergized_islands": sorted(monitor.island_blackout.deenergized_islands),
        },
        "collector_maxima": {
            collector.spec.name: {
                "max_voltage_pu_before_trip": float(collector.max_voltage_pu),
                "time_s": float(collector.max_voltage_time_s),
                "threshold_pu": collector.spec.threshold_pu,
                "report_cluster": collector.spec.report_cluster,
                "report_time_s": collector.spec.report_time_s,
                "trip_stage": collector.trip_stage_name,
                "trip_time_s": collector.trip_time_s,
                "trip_voltage_pu": collector.trip_voltage_pu,
            }
            for collector in collectors
        },
        "interpretation": (
            "The run reproduces a report-constrained mechanism surrogate on a "
            "large ACTIVSg2000 dynamic backbone: hidden collector-side "
            "overvoltage trips remove generation and reactive absorption; after "
            "the collector cascade propagates, generator protection, failed "
            "defense action, Morocco-like AC underfrequency separation, "
            "France-like out-of-step separation, HVDC blocking, and affected "
            "island de-energization are evaluated explicitly. The paper figure "
            "trims traces at the operational blackout endpoint instead of "
            "plotting a misleading post-blackout continuation."
        ),
    }


def _write_outputs(
    out_dir: Path,
    cfg: ReplicationConfig,
    collectors: list[CollectorRuntime],
    monitor: IberianReplicationMonitor,
    summary: dict[str, Any],
) -> ReplicationArtifacts:
    summary_path = out_dir / "summary.json"
    events_path = out_dir / "events.json"
    collector_trace_path = out_dir / "collector_traces.csv"
    system_trace_path = out_dir / "system_traces.csv"
    island_trace_path = out_dir / "island_traces.csv"
    deenergized_assets_path = out_dir / "deenergized_assets.csv"
    blackout_sequence_path = out_dir / "blackout_sequence.json"
    config_path = out_dir / "case_config.json"

    config_payload = asdict(cfg)
    config_payload["collector_runtime"] = [
        {
            "name": item.spec.name,
            "transmission_bus": item.spec.transmission_bus,
            "collector_bus": item.collector_bus,
            "tap": item.spec.tap,
            "base_kv": item.spec.base_kv,
            "threshold_pu": item.spec.threshold_pu,
            "gsu_line_id": item.gsu_line_id,
            "pq_id": item.pq_id,
            "main_q_shunt_id": item.main_q_shunt_id,
            "fixed_pf_q_shunt_id": item.fixed_pf_q_shunt_id,
        }
        for item in collectors
    ]
    config_payload["background_fixed_pf_shunt_id"] = monitor.background_fixed_pf_shunt_id
    config_payload["blackout_load_ids"] = monitor.blackout_load_ids
    config_payload["area_selection"] = asdict(monitor.area_selection)
    config_payload["hvdc_surrogate_runtime"] = asdict(monitor.hvdc)
    config_payload["morocco_ac_runtime"] = asdict(monitor.morocco_ac)
    config_payload["out_of_step_runtime"] = asdict(monitor.out_of_step)
    config_payload["generator_protection_runtime"] = asdict(monitor.generator_protection)
    config_payload["defense_stage_runtime"] = [asdict(stage) for stage in monitor.defense_stages]
    config_payload["blackout_criteria_runtime"] = asdict(monitor.blackout_criteria)
    island_runtime_payload = asdict(monitor.island_blackout)
    island_runtime_payload["deenergized_islands"] = sorted(
        monitor.island_blackout.deenergized_islands
    )
    config_payload["island_blackout_runtime"] = island_runtime_payload

    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    events_path.write_text(
        json.dumps([asdict(event) for event in monitor.events], indent=2) + "\n"
    )
    blackout_sequence_path.write_text(json.dumps(monitor.blackout_sequence, indent=2) + "\n")
    config_path.write_text(json.dumps(config_payload, indent=2) + "\n")
    _write_csv(collector_trace_path, monitor.collector_trace)
    _write_csv(system_trace_path, monitor.system_trace)
    _write_csv(island_trace_path, monitor.island_trace)
    _write_csv(deenergized_assets_path, monitor.deenergized_assets)

    return ReplicationArtifacts(
        output_dir=out_dir,
        summary_path=summary_path,
        events_path=events_path,
        collector_trace_path=collector_trace_path,
        system_trace_path=system_trace_path,
        config_path=config_path,
        island_trace_path=island_trace_path,
        deenergized_assets_path=deenergized_assets_path,
        blackout_sequence_path=blackout_sequence_path,
    )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def make_replication_plot(
    artifacts: ReplicationArtifacts,
    cfg: ReplicationConfig,
    monitor: IberianReplicationMonitor,
    *,
    latex_fig_dir: Path | None = None,
    formats: tuple[str, ...] = ("pdf", "png", "svg"),
) -> list[Path]:
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 8.0,
            "axes.labelsize": 9.0,
            "xtick.labelsize": 8.0,
            "ytick.labelsize": 8.0,
            "legend.fontsize": 6.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "path",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    def _series(source_rows: list[dict[str, Any]], key: str, mask: np.ndarray) -> np.ndarray:
        values: list[float] = []
        for row in source_rows:
            value = row.get(key, float("nan"))
            if value == "" or value is None:
                values.append(float("nan"))
            else:
                values.append(float(value))
        return np.asarray(values, dtype=float)[mask]

    def _column_nan_percentile(values: np.ndarray, percentile: float) -> np.ndarray:
        out = np.full(values.shape[1], np.nan, dtype=float)
        for col in range(values.shape[1]):
            finite = values[:, col][np.isfinite(values[:, col])]
            if finite.size:
                out[col] = float(np.percentile(finite, percentile))
        return out

    collector_list = list(monitor.collectors)
    rows = monitor.collector_trace
    sys_rows = monitor.system_trace
    if not rows:
        raise RuntimeError("No collector trace was recorded; cannot plot replication.")
    if not sys_rows:
        raise RuntimeError("No system trace was recorded; cannot plot replication.")

    times = np.array([float(row["time_s"]) for row in rows])
    sys_times = np.array([float(row["time_s"]) for row in sys_rows])
    lost_p = np.array([float(row["cumulative_lost_p_mw"]) for row in sys_rows])
    lost_q = np.array([float(row["cumulative_lost_q_absorption_mvar"]) for row in sys_rows])
    physical_vmin = np.array([float(row["vmin_physical_pu"]) for row in sys_rows])
    physical_vmax = np.array([float(row["vmax_physical_pu"]) for row in sys_rows])
    physical_v01 = np.array(
        [float(row.get("v01_physical_pu", row["vmin_physical_pu"])) for row in sys_rows]
    )
    physical_v10 = np.array(
        [float(row.get("v10_physical_pu", row["vmin_physical_pu"])) for row in sys_rows]
    )
    physical_v50 = np.array(
        [
            float(
                row.get(
                    "v50_physical_pu",
                    0.5 * (float(row["vmin_physical_pu"]) + float(row["vmax_physical_pu"])),
                )
            )
            for row in sys_rows
        ]
    )
    physical_v90 = np.array(
        [float(row.get("v90_physical_pu", row["vmax_physical_pu"])) for row in sys_rows]
    )
    physical_v99 = np.array(
        [float(row.get("v99_physical_pu", row["vmax_physical_pu"])) for row in sys_rows]
    )
    affected_vmin = np.array(
        [float(row.get("affected_island_vmin_pu", np.nan)) for row in sys_rows],
        dtype=float,
    )
    affected_frequency = np.array(
        [float(row.get("affected_island_frequency_hz", np.nan)) for row in sys_rows],
        dtype=float,
    )
    affected_generation = np.array(
        [float(row.get("affected_island_generation_mw", np.nan)) for row in sys_rows],
        dtype=float,
    )
    affected_load = np.array(
        [float(row.get("affected_island_load_mw", np.nan)) for row in sys_rows],
        dtype=float,
    )
    blackout_time = monitor.blackout_criteria.declared_time_s
    plot_end = float(blackout_time) if blackout_time is not None else min(float(cfg.tf_s), 30.0)
    x_right = min(float(cfg.tf_s), max(24.0, plot_end + 1.6))
    time_mask = times <= plot_end + 1e-9
    sys_mask = sys_times <= plot_end + 1e-9
    times_plot = times[time_mask]
    sys_times_plot = sys_times[sys_mask]
    lost_p_plot = lost_p[sys_mask]
    lost_q_plot = lost_q[sys_mask]
    physical_vmin_plot = physical_vmin[sys_mask]
    physical_vmax_plot = physical_vmax[sys_mask]
    physical_v01_plot = physical_v01[sys_mask]
    physical_v10_plot = physical_v10[sys_mask]
    physical_v50_plot = physical_v50[sys_mask]
    physical_v90_plot = physical_v90[sys_mask]
    physical_v99_plot = physical_v99[sys_mask]
    affected_vmin_plot = affected_vmin[sys_mask]
    affected_frequency_plot = affected_frequency[sys_mask]
    affected_generation_plot = affected_generation[sys_mask]
    affected_load_plot = affected_load[sys_mask]

    palette = ["#486AA8", "#5C9E91", "#D49A4A", "#B75D69", "#7966A8", "#6D7F8F"]
    action_color = "#D88B45"
    trip_color = "#B64E5A"
    blackout_color = "#2F3437"
    threshold_color = "#555B62"

    fig, (ax_v, ax_trans, ax_island, ax_balance, ax_evt) = plt.subplots(
        5,
        1,
        figsize=(7.05, 7.15),
        sharex=True,
        gridspec_kw={
            "height_ratios": [1.60, 1.05, 1.05, 1.12, 0.40],
            "hspace": 0.08,
        },
        constrained_layout=False,
    )

    collector_voltage_stack: list[np.ndarray] = []
    transmission_voltage_by_collector: dict[str, np.ndarray] = {}
    visible_collectors = [collector for collector in collector_list if collector.spec.visible_trace]
    if not visible_collectors:
        visible_collectors = collector_list[: min(6, len(collector_list))]
    visible_names = {collector.spec.name for collector in visible_collectors}

    for i, collector in enumerate(collector_list):
        name = collector.spec.name
        voltage = np.array(
            [
                float(row[f"{name}_voltage_pu_raw"])
                if row[f"{name}_voltage_pu_raw"] != ""
                else np.nan
                for row in rows
            ],
            dtype=float,
        )[time_mask]
        collector_voltage_stack.append(voltage)
        transmission_voltage_by_collector[name] = np.array(
            [float(row[f"{name}_transmission_voltage_pu"]) for row in rows],
            dtype=float,
        )[time_mask]
    collector_array = np.vstack(collector_voltage_stack)
    collector_low = _column_nan_percentile(collector_array, 10)
    collector_high = _column_nan_percentile(collector_array, 90)
    ax_v.fill_between(
        times_plot,
        collector_low,
        collector_high,
        color="#D5A6AE",
        alpha=0.22,
        lw=0,
        label="Collector-bus voltage band",
        zorder=2,
    )
    for i, collector in enumerate(visible_collectors):
        name = collector.spec.name
        voltage = np.array(
            [
                float(row[f"{name}_voltage_pu_raw"])
                if row[f"{name}_voltage_pu_raw"] != ""
                else np.nan
                for row in rows
            ],
            dtype=float,
        )[time_mask]
        color = palette[i % len(palette)]
        trace_label = "Named protected assets" if i == 0 else "_nolegend_"
        ax_v.plot(times_plot, voltage, lw=1.75, color=color, label=trace_label, zorder=3)
        if collector.trip_time_s is not None and collector.trip_time_s <= plot_end + 1e-9:
            ax_v.plot(
                collector.trip_time_s,
                collector.trip_voltage_pu,
                marker="o",
                ms=4.2,
                mec="white",
                mew=0.6,
                color=color,
                zorder=5,
            )

    thresholds = sorted({round(item.spec.threshold_pu, 6) for item in collector_list})
    if len(thresholds) == 1:
        ax_v.axhline(
            thresholds[0],
            color=threshold_color,
            ls=(0, (4, 3)),
            lw=1.3,
            label="Protection pickup",
            zorder=2.5,
        )
    else:
        ax_v.axhspan(
            min(thresholds),
            max(thresholds),
            facecolor="#EFE7CB",
            edgecolor=threshold_color,
            linewidth=0.7,
            alpha=0.44,
            label="Protection pickup band",
            zorder=1.8,
        )
        ax_v.axhline(
            min(thresholds),
            color=threshold_color,
            lw=0.85,
            ls=(0, (3, 2)),
            alpha=0.75,
            zorder=2.6,
        )
        ax_v.axhline(
            max(thresholds),
            color=threshold_color,
            lw=0.85,
            ls=(0, (3, 2)),
            alpha=0.75,
            zorder=2.6,
        )

    trip_times = [event.time_s for event in monitor.events if event.category == "protection_trip"]
    if trip_times:
        ax_v.axvspan(min(trip_times), min(max(trip_times), plot_end), color="#F2D6DB", alpha=0.42, lw=0)

    ax_trans.fill_between(
        sys_times_plot,
        physical_vmin_plot,
        physical_vmax_plot,
        color="#87919A",
        alpha=0.16,
        lw=0,
        label="All buses min--max",
        zorder=0,
    )
    ax_trans.fill_between(
        sys_times_plot,
        physical_v01_plot,
        physical_v99_plot,
        color="#87919A",
        alpha=0.16,
        lw=0,
        label="1--99 percentile",
        zorder=1,
    )
    ax_trans.fill_between(
        sys_times_plot,
        physical_v10_plot,
        physical_v90_plot,
        color="#87919A",
        alpha=0.30,
        lw=0,
        label="10--90 percentile",
        zorder=2,
    )
    ax_trans.plot(
        sys_times_plot,
        physical_v50_plot,
        color="#46505A",
        lw=1.35,
        label="Median",
        zorder=3,
    )
    ax_trans.plot(
        sys_times_plot,
        physical_vmax_plot,
        color="#6D7780",
        lw=0.95,
        ls=(0, (4, 3)),
        alpha=0.80,
        label="Min/max",
        zorder=4,
    )
    ax_trans.plot(
        sys_times_plot,
        physical_vmin_plot,
        color="#3F464B",
        lw=0.95,
        ls=(0, (1.2, 2.0)),
        alpha=0.85,
        zorder=4,
    )
    for i, collector in enumerate(visible_collectors):
        name = collector.spec.name
        upstream_voltage = transmission_voltage_by_collector[name]
        trace_label = "Selected upstream buses" if i == 0 else "_nolegend_"
        ax_trans.plot(
            times_plot,
            upstream_voltage,
            color=palette[i % len(palette)],
            lw=0.95,
            alpha=0.70,
            label=trace_label,
            zorder=5,
        )

    if blackout_time is not None:
        for ax in (ax_v, ax_trans, ax_island, ax_balance, ax_evt):
            ax.axvspan(
                float(blackout_time),
                x_right,
                color="#2F3437",
                alpha=0.10 if ax is ax_v else 0.08,
                lw=0,
                zorder=-1,
            )
        ax_v.text(
            min(x_right - 0.08, float(blackout_time) + 0.18),
            0.985,
            "de-energized / no valid continuation",
            transform=ax_v.get_xaxis_transform(),
            ha="left",
            va="top",
            fontsize=6.8,
            color=blackout_color,
        )

    system_chain = {
        "hvdc_block",
        "morocco_ac_trip",
        "out_of_step_trip",
        "generator_trip",
        "defense_action",
        "island_blackout_declared",
        "system_blackout_declared",
        "blackout_declared",
    }
    for event in monitor.events:
        if event.time_s > plot_end + 1e-9 and event.category not in system_chain:
            continue
        if event.category == "operator_action":
            ax_v.axvline(event.time_s, color=action_color, lw=0.9, ls=(0, (3, 3)), alpha=0.72)
            ax_trans.axvline(event.time_s, color=action_color, lw=0.65, ls=(0, (3, 3)), alpha=0.35)
            ax_island.axvline(event.time_s, color=action_color, lw=0.65, ls=(0, (3, 3)), alpha=0.35)
            ax_balance.axvline(event.time_s, color=action_color, lw=0.65, ls=(0, (3, 3)), alpha=0.35)
        elif event.category == "protection_trip":
            ax_v.axvline(event.time_s, color=trip_color, lw=0.8, alpha=0.35)
            ax_trans.axvline(event.time_s, color=trip_color, lw=0.5, alpha=0.14)
            ax_island.axvline(event.time_s, color=trip_color, lw=0.5, alpha=0.14)
            ax_balance.axvline(event.time_s, color=trip_color, lw=0.5, alpha=0.14)
        elif event.category in system_chain:
            ax_v.axvline(event.time_s, color=blackout_color, lw=0.95, alpha=0.34)
            ax_trans.axvline(event.time_s, color=blackout_color, lw=0.7, alpha=0.20)
            ax_island.axvline(event.time_s, color=blackout_color, lw=0.7, alpha=0.20)
            ax_balance.axvline(event.time_s, color=blackout_color, lw=0.7, alpha=0.20)

    ax_v.set_ylabel("Collector\nvoltage (pu)", labelpad=5)
    ax_v.yaxis.set_label_coords(-0.075, 0.5)
    all_collector_v = np.concatenate(
        [item[np.isfinite(item)] for item in collector_voltage_stack if np.isfinite(item).any()]
    )
    plot_min = float(
        min(
            np.nanmin(all_collector_v),
            min(thresholds),
        )
    )
    plot_max = float(
        max(
            np.nanmax(all_collector_v),
            max(thresholds),
        )
    )
    ax_v.set_ylim(max(0.65, plot_min - 0.035), min(1.34, max(plot_max + 0.040, max(thresholds) + 0.035)))
    ax_v.set_xlim(0.0, x_right)
    ax_v.grid(axis="y", color="#E7E1DA", lw=0.7)
    ax_v.legend(
        loc="lower left",
        bbox_to_anchor=(0.0, 1.005, 1.0, 0.18),
        mode="expand",
        borderaxespad=0.0,
        ncol=3,
        frameon=False,
        fontsize=6.0,
        columnspacing=0.9,
        handlelength=1.6,
    )

    trans_min = float(np.nanmin(physical_vmin_plot))
    trans_max = float(np.nanmax(physical_vmax_plot))
    ax_trans.set_ylabel("Transmission\nvoltage (pu)", labelpad=5)
    ax_trans.yaxis.set_label_coords(-0.075, 0.5)
    ax_trans.set_ylim(max(0.65, trans_min - 0.035), min(1.34, trans_max + 0.040))
    ax_trans.grid(axis="y", color="#E7E1DA", lw=0.7)
    ax_trans.legend(
        loc="lower left",
        bbox_to_anchor=(0.0, 1.000, 1.0, 0.18),
        mode="expand",
        borderaxespad=0.0,
        ncol=5,
        frameon=False,
        fontsize=5.7,
        columnspacing=0.7,
        handlelength=1.4,
    )

    ax_island.plot(
        sys_times_plot,
        affected_vmin_plot,
        color="#486AA8",
        lw=1.55,
        label="Affected-island $V_{\\min}$",
    )
    ax_island.plot(
        sys_times_plot,
        physical_vmin_plot,
        color="#6D7F8F",
        lw=1.05,
        ls=(0, (3, 2)),
        alpha=0.85,
        label="System $V_{\\min}$",
    )
    ax_island.axhline(
        cfg.island_blackout_criteria.voltage_pu,
        color="#486AA8",
        lw=0.85,
        ls=(0, (2.4, 2.4)),
        alpha=0.65,
    )
    ax_freq = ax_island.twinx()
    ax_freq.plot(
        sys_times_plot,
        affected_frequency_plot,
        color="#B75D69",
        lw=1.25,
        label="Affected-island frequency",
    )
    ax_freq.axhline(
        cfg.island_blackout_criteria.frequency_hz,
        color="#B75D69",
        lw=0.85,
        ls=(0, (2.4, 2.4)),
        alpha=0.65,
    )
    ax_island.set_ylabel("Island voltage (pu)", labelpad=5)
    ax_island.yaxis.set_label_coords(-0.075, 0.5)
    finite_v = affected_vmin_plot[np.isfinite(affected_vmin_plot)]
    if finite_v.size:
        ax_island.set_ylim(max(0.65, min(float(np.nanmin(finite_v)) - 0.035, 0.84)), 1.18)
    ax_freq.set_ylabel("Frequency (Hz)", color="#B75D69", labelpad=5)
    ax_freq.tick_params(axis="y", colors="#B75D69")
    finite_f = affected_frequency_plot[np.isfinite(affected_frequency_plot)]
    if finite_f.size:
        ax_freq.set_ylim(
            min(47.0, float(np.nanmin(finite_f)) - 0.25),
            max(50.4, float(np.nanmax(finite_f)) + 0.12),
        )
    ax_island.grid(axis="y", color="#E7E1DA", lw=0.65)
    island_handles, island_labels = ax_island.get_legend_handles_labels()
    freq_handles, freq_labels = ax_freq.get_legend_handles_labels()
    ax_island.legend(
        island_handles + freq_handles,
        island_labels + freq_labels,
        loc="upper left",
        ncol=3,
        frameon=False,
        borderaxespad=0.0,
        fontsize=5.9,
        columnspacing=0.8,
        handlelength=1.4,
    )

    ax_balance.step(
        sys_times_plot,
        lost_q_plot,
        where="post",
        color="#4F7F73",
        lw=1.55,
        label="Lost MVAr absorption",
    )
    ax_balance.step(
        sys_times_plot,
        lost_p_plot,
        where="post",
        color="#B75D69",
        lw=1.35,
        label="Tripped MW",
    )
    ax_balance.set_ylabel("Cumulative loss", labelpad=5)
    ax_balance.yaxis.set_label_coords(-0.075, 0.5)
    ax_balance.grid(axis="y", color="#E7E1DA", lw=0.65)
    max_loss = max(
        2200.0,
        float(np.nanmax(lost_p_plot)) * 1.06 if lost_p_plot.size else 2200.0,
        float(np.nanmax(lost_q_plot)) * 1.06 if lost_q_plot.size else 2200.0,
    )
    ax_balance.set_ylim(-75, max_loss)
    ax_power = ax_balance.twinx()
    ax_power.plot(
        sys_times_plot,
        affected_generation_plot,
        color="#5D6870",
        lw=1.15,
        ls=(0, (5, 2.5)),
        label="Online generation",
    )
    ax_power.plot(
        sys_times_plot,
        affected_load_plot,
        color="#8B7D73",
        lw=1.15,
        ls=(0, (1.2, 2.0)),
        label="Online load",
    )
    finite_power = np.concatenate(
        [
            affected_generation_plot[np.isfinite(affected_generation_plot)],
            affected_load_plot[np.isfinite(affected_load_plot)],
        ]
    )
    if finite_power.size:
        ax_power.set_ylim(0.0, max(1000.0, float(np.nanmax(finite_power)) * 1.08))
    ax_power.set_ylabel("Affected island MW", color="#5D6870", labelpad=5)
    ax_power.tick_params(axis="y", colors="#5D6870")
    balance_handles, balance_labels = ax_balance.get_legend_handles_labels()
    power_handles, power_labels = ax_power.get_legend_handles_labels()
    ax_balance.legend(
        balance_handles + power_handles,
        balance_labels + power_labels,
        loc="upper left",
        frameon=False,
        ncol=2,
        fontsize=5.9,
        columnspacing=0.8,
        handlelength=1.35,
    )

    ax_evt.axhline(0.0, color="#C9C2BA", lw=0.8)
    for event in monitor.events:
        if event.time_s > x_right + 1e-9:
            continue
        if event.category == "operator_action":
            ax_evt.plot(event.time_s, 0.0, marker="v", ms=4.2, color=action_color, clip_on=False)
        elif event.category == "protection_trip":
            ax_evt.plot(event.time_s, 0.0, marker="|", ms=9.0, mew=1.3, color=trip_color)
        elif event.category in {"hvdc_block", "morocco_ac_trip", "out_of_step_trip", "generator_trip", "defense_action"}:
            ax_evt.plot(
                event.time_s,
                0.0,
                marker="D",
                ms=3.6,
                color=blackout_color,
                mec="white",
                mew=0.35,
                clip_on=False,
            )
        elif event.category in {"island_blackout_declared", "system_blackout_declared", "blackout_declared"}:
            ax_evt.plot(
                event.time_s,
                0.0,
                marker="X",
                ms=5.0,
                color=blackout_color,
                clip_on=False,
            )

    cluster_times: dict[str, list[float]] = defaultdict(list)
    for collector in collector_list:
        if collector.trip_time_s is not None:
            cluster_times[collector.spec.report_cluster].append(float(collector.trip_time_s))

    def _first_event_time(category: str) -> float | None:
        values = [event.time_s for event in monitor.events if event.category == category]
        return min(values) if values else None

    event_label_rows: list[tuple[float, str, str, int]] = []
    operator_times = [
        event.time_s for event in monitor.events if event.category == "operator_action"
    ]
    if operator_times:
        event_label_rows.append((float(np.median(operator_times)), "OA1-OA4", action_color, 8))
    label_by_cluster = {"E3": "E3 Granada", "E4": "E4 Badajoz", "E5": "E5 multi-site"}
    for cluster in ("E3", "E4", "E5"):
        if cluster in cluster_times:
            event_label_rows.append(
                (
                    float(np.median(cluster_times[cluster])),
                    label_by_cluster[cluster],
                    trip_color,
                    15 if cluster == "E4" else 8,
                )
            )
    label_by_category = [
        ("generator_trip", "GEN"),
        ("defense_action", "UFLS"),
        ("morocco_ac_trip", "MA AC"),
        ("out_of_step_trip", "FR AC"),
        ("hvdc_block", "HVDC"),
        ("island_blackout_declared", "BO"),
    ]
    for category, label in label_by_category:
        time_value = _first_event_time(category)
        if time_value is not None:
            event_label_rows.append(
                (
                    float(time_value),
                    label,
                    blackout_color,
                    14 if len(event_label_rows) % 2 else 7,
                )
            )

    for x_pos, label, color, offset in event_label_rows:
        if x_pos <= x_right + 1e-9:
            ax_evt.annotate(
                label,
                xy=(x_pos, 0.0),
                xytext=(0, offset),
                textcoords="offset points",
                ha="center",
                va="bottom",
                color=color,
                fontsize=5.8,
            )
    ax_evt.text(
        0.01,
        0.08,
        "operator actions, relay trips, defense, and island de-energization",
        transform=ax_evt.transAxes,
        ha="left",
        va="bottom",
        fontsize=7.2,
        color="#6E6862",
    )
    ax_evt.set_ylim(-0.5, 0.55)
    ax_evt.set_yticks([])
    ax_evt.tick_params(axis="x", which="both", length=3.0, labelbottom=True, pad=2)
    for name, spine in ax_evt.spines.items():
        spine.set_visible(name == "bottom")
    ax_evt.spines["bottom"].set_color("#C9C2BA")
    ax_evt.set_xlabel("Time (s)", labelpad=4)

    fig.align_ylabels([ax_v, ax_trans, ax_island, ax_balance])
    fig.subplots_adjust(left=0.155, right=0.875, top=0.965, bottom=0.105, hspace=0.11)

    written: list[Path] = []
    for fmt in formats:
        path = artifacts.output_dir / f"fig_replication_activsg2000.{fmt}"
        save_kwargs: dict[str, Any] = {"bbox_inches": "tight", "pad_inches": 0.04}
        if fmt == "png":
            save_kwargs["dpi"] = 600
        fig.savefig(path, **save_kwargs)
        written.append(path)

    if latex_fig_dir is not None:
        latex_fig_dir.mkdir(parents=True, exist_ok=True)
        for path in list(written):
            target = latex_fig_dir / path.name
            shutil.copy2(path, target)
            written.append(target)

    plt.close(fig)
    artifacts.figure_paths = written
    return written


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/activsg2000_iberian_replication"),
        help="Directory for JSON/CSV/figure outputs.",
    )
    parser.add_argument("--tf", type=float, default=30.0, help="TDS horizon in seconds.")
    parser.add_argument("--tstep", type=float, default=0.01, help="TDS step in seconds.")
    parser.add_argument("--plot", action="store_true", help="Generate replication figure.")
    parser.add_argument(
        "--latex-fig-dir",
        type=Path,
        default=Path("LaTeX/figures/generated"),
        help="Optional mirror directory for paper figures.",
    )
    parser.add_argument(
        "--formats",
        default="pdf,png,svg",
        help="Comma-separated figure formats when --plot is set.",
    )
    parser.add_argument(
        "--verbose-andes",
        action="store_true",
        help="Do not suppress ANDES init/run diagnostic tables.",
    )
    parser.add_argument(
        "--allow-equivalent-blackout",
        action="store_true",
        help=(
            "Enable the diagnostic PQ-imbalance blackout equivalent. "
            "This is off by default and should not be used for the paper figure."
        ),
    )
    parser.add_argument(
        "--disable-post-cascade-relays",
        action="store_true",
        help="Negative-control mode: leave collector trips enabled but disable HVDC/tie/generator/defense relays.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = ReplicationConfig(
        tf_s=float(args.tf),
        tstep_s=float(args.tstep),
        enable_post_cascade_relays=not args.disable_post_cascade_relays,
    )
    if args.allow_equivalent_blackout:
        cfg = replace(
            cfg,
            allow_equivalent_blackout=True,
            blackout_actions=_diagnostic_equivalent_blackout_actions(),
        )
    artifacts, summary, monitor = run_replication(
        cfg,
        out_dir=args.out,
        quiet_andes=not args.verbose_andes,
    )
    if args.plot:
        formats = tuple(item.strip().lower() for item in args.formats.split(",") if item.strip())
        make_replication_plot(
            artifacts,
            cfg,
            monitor,
            latex_fig_dir=args.latex_fig_dir,
            formats=formats,
        )
    return _print_summary(artifacts, summary, monitor)


def _print_summary(
    artifacts: ReplicationArtifacts,
    summary: dict[str, Any],
    monitor: IberianReplicationMonitor,
) -> int:
    print(f"Output directory: {artifacts.output_dir}")
    print(
        "Trips: "
        f"{summary['collector_trips']} collector relays, "
        f"first={summary['first_trip_time_s']}, last={summary['last_trip_time_s']}"
    )
    print(
        "Lost assets: "
        f"{summary['total_lost_p_mw']:.1f} MW, "
        f"{summary['total_lost_q_absorption_mvar']:.1f} MVAr absorption"
    )
    print(
        "Post-cascade chain: "
        f"Morocco AC trips={summary.get('morocco_ac_trip_events', 0)}, "
        f"France AC separations={summary['tie_line_separation_events']}, "
        f"HVDC blocks={summary['hvdc_blocks']}, "
        f"generator protection={summary['generator_protection_events']}, "
        f"defense actions={summary['defense_actions']}"
    )
    if summary.get("equivalent_blackout_enabled"):
        print(
            "Diagnostic equivalent blackout enabled: "
            f"{summary['automatic_blackout_actions']} PQ imbalance actions"
        )
    print(
        "Island blackout engine: "
        f"de-energized islands={summary.get('deenergized_island_count', 0)}, "
        f"recorded assets={summary.get('deenergized_asset_count', 0)}, "
        f"island blackout events={summary.get('island_blackout_events', 0)}"
    )
    print(
        "Physical-network voltage range during run: "
        f"{summary['min_physical_voltage_pu']:.3f} to "
        f"{summary['max_physical_voltage_pu']:.3f} pu"
    )
    if summary.get("blackout_detected"):
        print(
            "Blackout flag: "
            f"time={summary.get('blackout_time_s')}, reason={summary.get('blackout_reason')}"
        )
    else:
        print("Blackout flag: not triggered.")
    print(f"Events: {artifacts.events_path}")
    print(f"Collector traces: {artifacts.collector_trace_path}")
    if artifacts.island_trace_path is not None:
        print(f"Island traces: {artifacts.island_trace_path}")
    if artifacts.deenergized_assets_path is not None:
        print(f"De-energized assets: {artifacts.deenergized_assets_path}")
    if artifacts.blackout_sequence_path is not None:
        print(f"Blackout sequence: {artifacts.blackout_sequence_path}")
    for path in artifacts.figure_paths:
        print(f"Figure: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
