from __future__ import annotations

import json
from pathlib import Path
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from pa_dvsa.replication.activsg2000_engine import (
    CollectorRuntime,
    compute_island_snapshots,
    OverVoltageRelayStage,
    ReplicationConfig,
    resolve_collector_specs,
    select_affected_area,
    update_overvoltage_relay_timers,
)


class _Values:
    def __init__(self, values):
        self.v = np.asarray(values)


def _model(**fields):
    ns = SimpleNamespace()
    size = None
    for name, values in fields.items():
        setattr(ns, name, _Values(values))
        size = len(values)
    ns.n = int(size or 0)
    return ns


def test_default_replication_config_has_no_equivalent_blackout_path():
    cfg = ReplicationConfig()

    assert cfg.allow_equivalent_blackout is False
    assert cfg.blackout_actions == ()
    assert len(cfg.collector_specs) == 16
    assert all(spec.min_trip_time_s == 0.0 for spec in cfg.collector_specs)


def test_two_stage_overvoltage_relay_hysteresis_and_timing():
    spec = ReplicationConfig().collector_specs[0]
    spec = replace(
        spec,
        threshold_pu=1.10,
        dwell_s=1.0,
        relay_stages=(OverVoltageRelayStage("low", 1.10, 1.0, 0.95),),
    )
    relay = CollectorRuntime(
        spec=spec,
        collector_bus=1,
        gsu_line_id="L",
        pq_id="P",
        main_q_shunt_id="S1",
        fixed_pf_q_shunt_id="S2",
    )

    assert update_overvoltage_relay_timers(relay, 1.11, 0.4, 0.3) is None
    assert update_overvoltage_relay_timers(relay, 1.06, 0.4, 0.3) is None
    assert relay.stage_picked_up["low"] is True
    assert update_overvoltage_relay_timers(relay, 1.04, 0.4, 0.3) is None
    assert relay.stage_picked_up["low"] is False


def test_affected_area_selector_is_deterministic_on_synthetic_grid():
    system = SimpleNamespace(
        config=SimpleNamespace(mva=100.0),
        Line=_model(
            idx=["L12", "L23", "L34", "L35", "L45"],
            bus1=[1, 2, 3, 3, 4],
            bus2=[2, 3, 4, 5, 5],
            u=[1, 1, 1, 1, 1],
            r=[0.01, 0.02, 0.03, 0.01, 0.04],
            x=[0.20, 0.10, 0.05, 0.07, 0.04],
        ),
        GENROU=_model(
            idx=["G2", "G3", "G4"],
            bus=[2, 3, 4],
            Sn=[50.0, 200.0, 150.0],
        ),
        PQ=_model(
            idx=["P2", "P3", "P5"],
            bus=[2, 3, 5],
            p0=[1.0, 2.0, 3.0],
        ),
        PV=_model(
            idx=["V2", "V3", "V4"],
            bus=[2, 3, 4],
            p0=[0.5, 1.0, 1.5],
        ),
    )
    cfg = ReplicationConfig(
        area_seed_buses=(1,),
        area_bfs_radius=2,
        area_max_radius=3,
        area_min_genrou=2,
        area_min_loads=2,
        area_min_boundary_lines=1,
    )

    first = select_affected_area(system, cfg)
    second = select_affected_area(system, cfg)

    assert first.affected_buses == second.affected_buses
    assert first.boundary_line_ids == second.boundary_line_ids
    assert first.protected_genrou_ids == ["G3", "G2"]
    assert first.selected_load_ids == ["P3", "P2"]


def test_unpinned_report_collectors_are_assigned_to_high_voltage_buses():
    system = SimpleNamespace(
        config=SimpleNamespace(mva=100.0),
        Bus=_model(idx=[1, 2, 3, 4, 5], Vn=[500.0, 13.8, 230.0, 13.8, 500.0]),
        Line=_model(
            idx=["L12", "L23", "L34", "L45"],
            bus1=[1, 2, 3, 4],
            bus2=[2, 3, 4, 5],
            u=[1, 1, 1, 1],
            r=[0.01, 0.02, 0.03, 0.04],
            x=[0.20, 0.10, 0.05, 0.04],
        ),
        GENROU=_model(idx=["G2", "G4"], bus=[2, 4], Sn=[300.0, 250.0]),
        PQ=_model(idx=["P3"], bus=[3], p0=[1.0]),
        PV=_model(idx=[], bus=[], p0=[]),
    )
    cfg = ReplicationConfig(
        area_seed_buses=(1,),
        area_bfs_radius=4,
        area_max_radius=4,
        area_min_genrou=1,
        area_min_loads=0,
        area_min_boundary_lines=0,
        collector_specs=(
            ReplicationConfig().collector_specs[0],
            ReplicationConfig().collector_specs[6],
        ),
    )

    area = select_affected_area(system, cfg)
    resolved = resolve_collector_specs(system, cfg, area)

    assert resolved[1].transmission_bus in {3, 5}


def test_island_snapshot_accounting_is_deterministic():
    system = SimpleNamespace(
        config=SimpleNamespace(mva=100.0),
        Bus=_model(idx=[1, 2, 3, 4], v=[1.02, 1.01, 0.88, 0.91]),
        Line=_model(
            idx=["L12", "L23", "L34"],
            bus1=[1, 2, 3],
            bus2=[2, 3, 4],
            u=[1, 1, 1],
        ),
        GENROU=_model(
            idx=["G1", "G4"],
            bus=[1, 4],
            u=[1, 1],
            Sn=[100.0, 120.0],
            Pe=[0.5, 0.7],
            omega=[1.0, 0.98],
            delta=[0.0, 0.2],
            M=[4.0, 6.0],
        ),
        PV=_model(idx=["PV3"], bus=[3], u=[1], p0=[0.2]),
        PQ=_model(idx=["L3"], bus=[3], u=[1], p0=[0.3], q0=[0.1]),
        Shunt=_model(idx=["SH3"], bus=[3], u=[1], b=[-0.05]),
    )

    islands = compute_island_snapshots(
        system,
        collectors=[],
        affected_buses={3, 4},
        ignored_line_ids={"L23"},
        nominal_frequency_hz=50.0,
    )

    assert [sorted(island.buses) for island in islands] == [[1, 2], [3, 4]]
    affected = islands[1]
    assert affected.island_id == "I002"
    assert affected.affected_bus_count == 2
    assert affected.online_genrou_ids == ("G4",)
    assert affected.online_pv_ids == ("PV3",)
    assert affected.online_load_ids == ("L3",)
    assert affected.generation_mw == pytest.approx(90.0)
    assert affected.load_mw == pytest.approx(30.0)
    assert affected.q_absorption_mvar == pytest.approx(5.0)
    assert affected.vmin_pu == pytest.approx(0.88)
    assert affected.frequency_hz == pytest.approx(49.0)
    assert affected.has_voltage_reference is True


def test_generated_activsg2000_replication_chronology():
    events_path = Path("results/activsg2000_iberian_replication/events.json")
    summary_path = Path("results/activsg2000_iberian_replication/summary.json")
    island_path = Path("results/activsg2000_iberian_replication/island_traces.csv")
    deenergized_path = Path("results/activsg2000_iberian_replication/deenergized_assets.csv")
    sequence_path = Path("results/activsg2000_iberian_replication/blackout_sequence.json")
    if not events_path.exists() or not summary_path.exists():
        pytest.skip("ACTIVSg2000 replication artifacts have not been generated.")

    events = json.loads(events_path.read_text())
    summary = json.loads(summary_path.read_text())
    if "deenergized_island_count" not in summary:
        pytest.skip("ACTIVSg2000 replication artifacts predate the island blackout engine.")
    first_trip = summary["first_trip_time_s"]
    last_operator_action = max(
        event["time_s"] for event in events if event["category"] == "operator_action"
    )

    assert summary["equivalent_blackout_enabled"] is False
    assert summary["automatic_blackout_actions"] == 0
    assert summary["blackout_detected"] is True
    assert summary["collector_count"] >= 16
    assert summary["collector_trips"] >= 6
    assert summary["hvdc_blocks"] == 1
    assert summary["morocco_ac_trip_events"] == 1
    assert summary["tie_line_separation_events"] == 1
    assert summary["generator_protection_events"] == 1
    assert summary["defense_actions"] >= 1
    assert summary["deenergized_island_count"] >= 1
    assert summary["island_blackout_events"] >= 1
    assert summary["deenergized_asset_count"] > 0
    assert island_path.exists() and island_path.stat().st_size > 0
    assert deenergized_path.exists() and deenergized_path.stat().st_size > 0
    assert sequence_path.exists() and sequence_path.stat().st_size > 0
    assert all(
        event["time_s"] < first_trip
        for event in events
        if event["category"] == "operator_action"
    )
    assert first_trip > last_operator_action + 1.0
    categories = [event["category"] for event in events]
    assert categories.index("generator_trip") > categories.index("protection_trip")
    assert categories.index("defense_action") > categories.index("generator_trip")
    assert categories.index("morocco_ac_trip") > categories.index("defense_action")
    assert categories.index("out_of_step_trip") > categories.index("morocco_ac_trip")
    assert categories.index("hvdc_block") > categories.index("out_of_step_trip")
    assert categories.index("island_blackout_declared") > categories.index("hvdc_block")
    assert categories.index("system_blackout_declared") > categories.index("island_blackout_declared")
    assert categories[-1] == "system_blackout_declared"
