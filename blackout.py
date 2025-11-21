# blackout.py
# -*- coding: utf-8 -*-
"""
Iberian cascade replication - IEEE 39 bus power system

* Collector buses live behind fixed OLTC taps (0.90–0.94) so transmission and
  collector voltages decouple as described in the ENTSO-E factual report.
* Protection monitors the 138-kV collectors directly and trips when they exceed
  1.10 pu, even if the transmission buses remain within their own limits.
* Each collector owns a shunt reactor. Tripping the collector removes that absorption
  and the connected PV block, letting the physics raise voltages for the next trip.
"""

from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import random
import numpy as np
import andes


# --------------------------------------------------------------------------- #
# Utilities
# --------------------------------------------------------------------------- #
def case_path() -> str:
    """Return the IEEE-39 case bundled with ANDES."""
    base = os.path.join(os.path.dirname(andes.__file__), "cases", "ieee39")
    for candidate in ("ieee39_full.xlsx", "ieee39.xlsx"):
        path = os.path.join(base, candidate)
        if os.path.exists(path):
            return path
    raise FileNotFoundError("IEEE-39 case not found in ANDES installation.")


def disable_predefined_events(system: andes.System) -> None:
    for model_name in ("Toggle", "Alter", "Fault"):
        if hasattr(system, model_name):
            table = getattr(system, model_name)
            for i in range(getattr(table, "n", 0)):
                table.u.v[i] = 0


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
@dataclass
class CollectorMeta:
    """Metadata for a collector group protected on the 138-kV side."""
    name: str
    trans_bus: int
    coll_bus: int
    tap: float
    base_kv: float
    threshold: float
    delay: float
    pq_idx: int
    gen_ids: List[str]
    q_absorption: float
    shunt_id: Optional[str] = None
    timer: float = 0.0
    tripped: bool = False
    tripped_time: float = -1.0


@dataclass
class ScenarioConfig:
    # Simulation horizon
    sim_tf: float = 18.0
    tstep: float = 0.01
    fnom: float = 50.0

    # Loading + voltage
    load_scale: float = 0.8
    ehv_voltage_target: float = 1

    # Collector protection (138-kV base)
    collector_threshold_pu: float = 1.13
    collector_dwell_s: float = 0.01
    collector_dwell_offsets: Tuple[float, ...] = (0.05, 0.25, 0.20, 0.01, 0.10)
    collector_enable_time: float = 0.0
    collector_base_kv: float = 138.0
    collector_taps: Tuple[float, ...] = (0.992, 0.995, 0.989, 0.999, 0.999)
    collector_q_mvar: Tuple[float, ...] = (78.0, 72.0, 66.0, 60.0, 90.0)
    collector_trans_buses: Tuple[int, ...] = (21, 22, 16, 19, 3)
    collector_thresholds: Tuple[float, ...] = (1.08, 1.12, 1.12, 1.12, 1.13)

    # Renewable blocks feeding the collectors
    pv_unit_count: int = 5
    pv_total_p_pu: float = 4.0

    # Operator actions (seconds) - spread out more gradually
    meshing_times: Tuple[float, ...] = (4.5, 5.0, 7.0, 9.0)
    reactor_times: Tuple[float, ...] = (6.5, 10.0)
    hvdc_time: float = 8.5
    export_reduction_time: float = 5.8

    # HVDC surrogate (PQ injection)
    hvdc_bus: int = 28
    hvdc_export_pu: float = 0.95
    hvdc_fixed_pu: float = 1.20
    hvdc_droop_coeff: float = 0.30
    hvdc_ramp_rate: float = 0.03
    hvdc_min_pu: float = 0.10
    hvdc_max_pu: float = 1.40

    # Network reinforcements
    meshed_lines: int = 4
    reactor_count: int = 2
    reactor_mvar: float = 130.0

    # Export reduction (modeled as lighter PQ load)
    export_reduction_factor: float = 0.98

    # Undervoltage/frequency collapse guards
    ufls_trigger_hz: float = 49.4
    ufls_active_fraction: float = 0.92
    ufls_reactive_fraction: float = 0.85
    collapse_freq: float = 48.3
    collapse_voltage_pu: float = 0.65
    collapse_min_trips: int = 3
    collapse_watch_buses: Tuple[int, ...] = ()

    # Feature toggles
    enable_meshing: bool = True
    enable_reactor_switch: bool = True
    enable_hvdc_mode_change: bool = True
    enable_export_reduction: bool = True
    enable_ufls: bool = True

    # Compatibility knobs expected by plotting scripts
    enable_distribution_q: bool = False
    dq_percent: float = 0.0
    dq_mvar: float = 0.0
    hvdc_set_change: float = 0.0
    enable_dist_q: bool = False
    dist_q_percent: float = 0.0
    dist_q_mvar: float = 0.0
    hvdc_set_mult: float = 1.0
    threshold_hv_pu: float = 1.15
    threshold_lv_pu: float = 1.13
    transmission_gate_pu: float = 1.10
    transmission_lag_tau: float = 0.35
    mesh_mvar_loss: float = 35.0
    export_reduction_mvar: float = 10.0
    hvdc_mode_mvar: float = 0.0

    # Monitoring
    observed_transmission_buses: Tuple[int, ...] = (
        3,
        4,
        16,
        19,
        20,
        21,
        22,
        33,
        35,
        38,
    )
    start_time: datetime = field(
        default_factory=lambda: datetime(2025, 4, 28, 12, 32, 40)
    )


@dataclass
class EventRecord:
    time: float
    code: str
    description: str
    category: str

    def as_dict(self, base: datetime) -> Dict[str, str]:
        stamp = base + timedelta(seconds=self.time)
        return {
            "time_s": f"{self.time:.3f}",
            "time_local": stamp.strftime("%H:%M:%S"),
            "code": self.code,
            "category": self.category,
            "description": self.description,
        }


# --------------------------------------------------------------------------- #
# HVDC controller
# --------------------------------------------------------------------------- #
class HVDCController:
    """Simple PQ surrogate with droop mode and fixed export mode."""

    def __init__(self, scenario: "CascadeScenario") -> None:
        self.scenario = scenario
        self.cfg = scenario.cfg
        self.mode: str = "emulation"
        self.pq_idx: Optional[int] = None
        self._pq_pos: Optional[int] = None
        self.current_p: float = self.cfg.hvdc_export_pu

    def install(self) -> None:
        pq_idx = self.scenario.ss.add(
            "PQ",
            {
                "bus": self.cfg.hvdc_bus,
                "p0": max(self.current_p, 0.0),
                "q0": 0.0,
                "name": "HVDC_Link",
                "u": 1,
            },
        )
        self.pq_idx = pq_idx
        pq_list = list(self.scenario.ss.PQ.idx.v)
        self._pq_pos = pq_list.index(pq_idx)

    def _target_power(self, freq_hz: float) -> float:
        if self.mode == "fixed":
            return float(
                np.clip(
                    self.cfg.hvdc_fixed_pu, self.cfg.hvdc_min_pu, self.cfg.hvdc_max_pu
                )
            )
        deviation = max(self.cfg.fnom - freq_hz, 0.0)
        target = self.cfg.hvdc_export_pu - self.cfg.hvdc_droop_coeff * deviation
        return float(np.clip(target, self.cfg.hvdc_min_pu, self.cfg.hvdc_max_pu))

    def set_mode(self, mode: str) -> None:
        if mode not in ("emulation", "fixed"):
            raise ValueError(f"Unsupported HVDC mode '{mode}'.")
        self.mode = mode

    def step(self, freq_hz: float, dt: float) -> None:
        if self._pq_pos is None:
            return
        target = self._target_power(freq_hz)
        if dt > 0.0:
            ramp = np.clip(
                target - self.current_p,
                -self.cfg.hvdc_ramp_rate * dt,
                self.cfg.hvdc_ramp_rate * dt,
            )
            self.current_p += ramp
        else:
            self.current_p = target
        self.scenario.ss.PQ.p0.v[self._pq_pos] = self.current_p


# --------------------------------------------------------------------------- #
# Monitor
# --------------------------------------------------------------------------- #
class CascadeMonitor:
    """Execute operator actions and enforce collector protection."""

    def __init__(self, scenario: "CascadeScenario") -> None:
        self.scenario = scenario
        self.cfg = scenario.cfg
        self.collectors = scenario.collectors
        self.prev_t: Optional[float] = None

        # Action scheduling
        self.mesh_idx: int = 0
        self.reactor_idx: int = 0
        self.mesh_schedule: Tuple[float, ...] = tuple(
            self.cfg.meshing_times[: len(self.scenario.mesh_line_ids)]
        )
        self.reactor_schedule: Tuple[float, ...] = tuple(
            self.cfg.reactor_times[: len(self.scenario.reactor_ids)]
        )

        self.activated_events: Dict[str, bool] = {
            "hvdc": False,
            "export": False,
            "ufls": False,
            "collapse": False,
        }
        self.trip_count: int = 0
        self.collapse_time: Optional[float] = None
        self.ufls_time: Optional[float] = None
        self.cascade_started: bool = False

    def _set_status(self, model: str, idx: str, status: int) -> None:
        table = getattr(self.scenario.ss, model)
        ids = list(table.idx.v)
        if idx in ids:
            pos = ids.index(idx)
            table.u.v[pos] = status

    def _apply_next_mesh(self, t: float) -> None:
        if self.mesh_idx >= len(self.scenario.mesh_line_ids):
            return
        mesh_id = self.scenario.mesh_line_ids[self.mesh_idx]
        self._set_status("Line", mesh_id, 1)

        stage = self.mesh_idx + 1
        total = len(self.scenario.mesh_line_ids)
        self.scenario.record_event(
            t, "OA1", f"Parallel line energized ({stage}/{total}): {mesh_id}", "OA"
        )
        self.scenario.record_reactive_loss(
            t, self.cfg.mesh_mvar_loss, f"Mesh {mesh_id}", "OA"
        )
        self.mesh_idx += 1

    def _open_next_reactor(self, t: float) -> None:
        if self.reactor_idx >= len(self.scenario.reactor_ids):
            return
        shunt_id = self.scenario.reactor_ids[self.reactor_idx]
        self._set_status("Shunt", shunt_id, 0)

        stage = self.reactor_idx + 1
        total = len(self.scenario.reactor_ids)
        self.scenario.record_event(
            t, "OA3", f"Shunt reactor opened ({stage}/{total}): {shunt_id}", "OA"
        )
        self.scenario.record_reactive_loss(
            t, self.cfg.reactor_mvar, f"Reactor {shunt_id} opened", "OA"
        )
        self.reactor_idx += 1

    def _switch_hvdc(self, t: float) -> None:
        if self.activated_events["hvdc"]:
            return
        self.activated_events["hvdc"] = True
        self.scenario.hvdc.set_mode("fixed")
        self.scenario.record_event(t, "OA4", "HVDC switched to fixed-power mode", "OA")
        self.scenario.record_reactive_loss(
            t, self.cfg.hvdc_mode_mvar, "HVDC mode change", "OA"
        )

    def _reduce_exports(self, t: float) -> None:
        if self.activated_events["export"]:
            return
        self.activated_events["export"] = True

        for i in range(self.scenario.ss.PQ.n):
            if i in self.scenario.non_load_pq_indices:
                continue
            p0 = float(self.scenario.ss.PQ.p0.v[i])
            if p0 > 0.0:
                self.scenario.ss.PQ.p0.v[i] *= self.cfg.export_reduction_factor
                self.scenario.ss.PQ.q0.v[i] *= self.cfg.export_reduction_factor

        self.scenario.record_event(t, "OA2", "Exports to France/Portugal reduced", "OA")
        self.scenario.record_reactive_loss(
            t,
            self.cfg.export_reduction_mvar,
            "Export reduction",
            "OA",
        )

    def _trigger_ufls(self, t: float) -> None:
        if self.activated_events["ufls"]:
            return
        self.activated_events["ufls"] = True
        self.ufls_time = t
        self.scenario.apply_ufls(t)
        self.scenario.record_event(
            t, "UFLS", "Under-frequency load shedding armed", "AA"
        )

    def _trigger_collapse(self, t: float) -> None:
        if self.activated_events["collapse"]:
            return
        self.activated_events["collapse"] = True
        self.collapse_time = t
        self.scenario.record_event(t, "COLLAPSE", "Voltage collapse detected", "AA")

    def _trip_collector(
        self, meta: CollectorMeta, t: float, v_coll: float, v_trans: float
    ) -> None:
        meta.tripped = True
        meta.timer = 0.0
        meta.tripped_time = t

        # Disconnect PV block
        pq_list = list(self.scenario.ss.PQ.idx.v)
        if meta.pq_idx in pq_list:
            pos = pq_list.index(meta.pq_idx)
            self.scenario.ss.PQ.u.v[pos] = 0

        # Disconnect linked synchronous machines if any
        for gen_id in meta.gen_ids:
            self._set_status("GENROU", gen_id, 0)

        # Remove collector absorption
        if meta.shunt_id is not None:
            self._set_status("Shunt", meta.shunt_id, 0)

        self.trip_count += 1
        self.scenario.record_event(
            t,
            f"AA_{meta.name}",
            f"{meta.name} trips (Vcoll={v_coll:.3f} pu, Vtrans={v_trans:.3f} pu)",
            "AA",
        )
        self.scenario.record_reactive_loss(
            t, meta.q_absorption, f"{meta.name} trip", "AA"
        )

    def _monitor_collectors(self, t: float, system: andes.System, dt: float) -> None:
        bus_v = system.Bus.v.v
        for meta in self.collectors.values():
            pos_coll = self.scenario.bus_pos.get(meta.coll_bus)
            pos_trans = self.scenario.bus_pos.get(meta.trans_bus)
            if pos_coll is None or pos_trans is None:
                continue

            v_coll = float(bus_v[pos_coll])
            v_trans = float(bus_v[pos_trans])

            self.scenario.collector_trace.setdefault(meta.name, []).append((t, v_coll))
            self.scenario.trans_voltage_trace.setdefault(meta.name, []).append(
                (t, v_trans)
            )
            scada_list = self.scenario.scada_trace.setdefault(meta.name, [])
            if scada_list:
                prev_filtered = scada_list[-1][1]
            else:
                prev_filtered = v_trans
            if dt > 0.0:
                alpha = np.clip(dt / max(self.cfg.transmission_lag_tau, 1e-3), 0.0, 1.0)
                filtered = prev_filtered + alpha * (v_trans - prev_filtered)
            else:
                filtered = v_trans
            scada_list.append((t, filtered))

            if meta.tripped or t < self.cfg.collector_enable_time or dt <= 0:
                continue

            gate_ok = self.cascade_started or (
                filtered < self.cfg.transmission_gate_pu
            )

            if gate_ok and v_coll >= meta.threshold:
                meta.timer += dt
            else:
                meta.timer = max(0.0, meta.timer - 0.4 * dt)

            if meta.timer >= meta.delay:
                self._trip_collector(meta, t, v_coll, v_trans)

    def __call__(self, t: float, system: andes.System) -> None:
        if self.prev_t is None:
            self.prev_t = float(t)
            self.scenario.record_sync_state(system, self.prev_t)
            self._monitor_collectors(self.prev_t, system, 0.0)
            return

        t = float(t)
        dt = max(t - self.prev_t, 0.0)
        self.prev_t = t

        # Operator actions
        if self.cfg.enable_meshing:
            while (
                self.mesh_idx < len(self.mesh_schedule)
                and t >= self.mesh_schedule[self.mesh_idx]
            ):
                self._apply_next_mesh(self.mesh_schedule[self.mesh_idx])
        if self.cfg.enable_reactor_switch:
            while (
                self.reactor_idx < len(self.reactor_schedule)
                and t >= self.reactor_schedule[self.reactor_idx]
            ):
                self._open_next_reactor(self.reactor_schedule[self.reactor_idx])
        if self.cfg.enable_export_reduction and t >= self.cfg.export_reduction_time:
            self._reduce_exports(self.cfg.export_reduction_time)
        if self.cfg.enable_hvdc_mode_change and t >= self.cfg.hvdc_time:
            self._switch_hvdc(self.cfg.hvdc_time)

        # HVDC + UFLS
        freq_now = self.scenario.estimate_frequency(system, t)
        if self.cfg.enable_hvdc_mode_change:
            self.scenario.hvdc.step(freq_now, dt)
        if self.cfg.enable_ufls and freq_now <= self.cfg.ufls_trigger_hz:
            self._trigger_ufls(t)

        # Collector protection
        self._monitor_collectors(t, system, dt)

        # Collapse detection
        if not self.activated_events["collapse"]:
            min_v = 2.0
            for bus in self.scenario.collapse_watch_buses:
                pos = self.scenario.bus_pos.get(bus)
                if pos is not None:
                    min_v = min(min_v, float(system.Bus.v.v[pos]))
            if self.trip_count >= self.cfg.collapse_min_trips:
                if (
                    min_v <= self.cfg.collapse_voltage_pu
                    or freq_now <= self.cfg.collapse_freq
                ):
                    self._trigger_collapse(t)

        self.scenario.record_sync_state(system, t)


# --------------------------------------------------------------------------- #
# Cascade scenario
# --------------------------------------------------------------------------- #
class CascadeScenario:
    def __init__(self, cfg: ScenarioConfig | None = None) -> None:
        self.cfg = cfg or ScenarioConfig()

        case_file = case_path()
        print(f"Loading IEEE-39 case: {case_file}")
        self.ss = andes.load(
            case_file, setup=False, no_output=True, default_config=True
        )
        disable_predefined_events(self.ss)

        self.timeline: List[EventRecord] = []
        self.collectors: Dict[str, CollectorMeta] = {}
        self.mesh_line_ids: List[str] = []
        self.reactor_ids: List[str] = []
        self.dist_shunt_ids: List[str] = []
        self.non_load_pq_indices: set[int] = set()
        self.reactive_log: List[Tuple[float, float, str, str]] = []

        self.bus_pos: Dict[int, int] = {}
        self.ehv_buses: List[int] = []
        self.hv_buses: List[int] = []
        self.collapse_watch_buses: List[int] = []

        self.source_ids: List[int] = []
        self.source_bus_map: List[int] = []

        self.collector_trace: Dict[str, List[Tuple[float, float]]] = {}
        self.trans_voltage_trace: Dict[str, List[Tuple[float, float]]] = {}
        self.scada_trace: Dict[str, List[Tuple[float, float]]] = {}
        self.bus_traces: Dict[str, List[Tuple[float, float]]] = {}
        self.freq_history: List[Tuple[float, float]] = []
        self.sync_history: List[Tuple[float, float, float]] = []

        self.monitor: Optional[CascadeMonitor] = None
        self.hvdc = HVDCController(self)

    def build(self) -> None:
        print("\nBuilding cascade scenario...")
        self._scale_loads()
        self._set_voltage_targets()
        self._add_pq_fleet()
        self._identify_buses()
        self._configure_parallel_lines()
        if self.cfg.enable_reactor_switch:
            self._configure_reactors()
        self._attach_collectors()
        self._inject_distribution_q()

        if abs(self.cfg.hvdc_set_change) > 1e-9 or not np.isclose(
            self.cfg.hvdc_set_mult, 1.0
        ):
            new_set = (
                self.cfg.hvdc_export_pu * self.cfg.hvdc_set_mult
            ) + self.cfg.hvdc_set_change
            self.cfg.hvdc_export_pu = float(
                np.clip(new_set, self.cfg.hvdc_min_pu, self.cfg.hvdc_max_pu)
            )

        if not self.cfg.collapse_watch_buses:
            self.collapse_watch_buses = self.ehv_buses[:6]
        else:
            self.collapse_watch_buses = list(self.cfg.collapse_watch_buses)

        if self.cfg.enable_hvdc_mode_change:
            self.hvdc.install()

        self.monitor = CascadeMonitor(self)

        print(f"  Collectors: {len(self.collectors)}")
        print(f"  Parallel lines: {len(self.mesh_line_ids)}")
        print(f"  Reactors: {len(self.reactor_ids)}")

    def record_reactive_loss(self, time: float, amount: float, label: str, category: str) -> None:
        if amount == 0.0:
            return
        self.reactive_log.append((float(time), float(amount), label, category))

    def _scale_loads(self) -> None:
        if not hasattr(self.ss, "PQ"):
            return
        factor = float(self.cfg.load_scale)
        if np.isclose(factor, 1.0):
            return
        for i in range(self.ss.PQ.n):
            p0 = float(self.ss.PQ.p0.v[i])
            if p0 > 0.0:
                self.ss.PQ.p0.v[i] = p0 * factor
                self.ss.PQ.q0.v[i] = float(self.ss.PQ.q0.v[i]) * factor

    def _set_voltage_targets(self) -> None:
        bus_base_map = {
            int(idx): float(base)
            for idx, base in zip(self.ss.Bus.idx.v, self.ss.Bus.Vn.v)
        }
        target = float(self.cfg.ehv_voltage_target)
        if hasattr(self.ss, "PV"):
            for i in range(self.ss.PV.n):
                bus = int(self.ss.PV.bus.v[i])
                if bus_base_map.get(bus, 0.0) >= 300.0:
                    self.ss.PV.v0.v[i] = target
                    if hasattr(self.ss.PV, "vref"):
                        self.ss.PV.vref.v[i] = target
        for i, bus in enumerate(self.ss.Bus.idx.v):
            if bus_base_map.get(int(bus), 0.0) >= 300.0:
                self.ss.Bus.v0.v[i] = target

    def _identify_buses(self) -> None:
        basekv = np.array(self.ss.Bus.Vn.v, dtype=float)
        bus_idx = np.array(self.ss.Bus.idx.v, dtype=int)
        self.ehv_buses = bus_idx[basekv >= 300.0].tolist()
        self.hv_buses = bus_idx[(basekv >= 100.0) & (basekv < 300.0)].tolist()

    def _configure_parallel_lines(self) -> None:
        if not hasattr(self.ss, "Line"):
            return
        line_data = []
        for i in range(self.ss.Line.n):
            bus1 = int(self.ss.Line.bus1.v[i])
            bus2 = int(self.ss.Line.bus2.v[i])
            if bus1 in self.ehv_buses and bus2 in self.ehv_buses:
                x = abs(float(self.ss.Line.x.v[i]))
                idx = self.ss.Line.idx.v[i]
                line_data.append((x, idx, i))
        line_data.sort(reverse=True)
        count = min(self.cfg.meshed_lines, len(line_data))
        for j in range(count):
            _, base_idx, pos = line_data[j]
            params = {
                "bus1": int(self.ss.Line.bus1.v[pos]),
                "bus2": int(self.ss.Line.bus2.v[pos]),
                "r": float(self.ss.Line.r.v[pos]),
                "x": float(self.ss.Line.x.v[pos]),
                "b": float(self.ss.Line.b.v[pos]),
                "u": 0,
                "name": f"Mesh_{base_idx}",
            }
            new_idx = self.ss.add("Line", params)
            self.mesh_line_ids.append(new_idx)

    def _configure_reactors(self) -> None:
        if not self.ehv_buses:
            return
        base_mva = float(self.ss.config.mva)
        b_pu = -self.cfg.reactor_mvar / base_mva
        count = min(self.cfg.reactor_count, len(self.ehv_buses))
        if count == 0:
            return
        step = max(len(self.ehv_buses) // count, 1)
        selected = self.ehv_buses[::step][:count]
        for bus in selected:
            shunt_idx = self.ss.add(
                "Shunt",
                {
                    "bus": bus,
                    "g": 0.0,
                    "b": b_pu,
                    "u": 1,
                    "name": f"Reactor_{bus}",
                },
            )
            self.reactor_ids.append(shunt_idx)

    def _inject_distribution_q(self) -> None:
        """Optional constant MVAr injection for calibration/figures."""
        enabled = self.cfg.enable_distribution_q or self.cfg.enable_dist_q
        if not enabled:
            return
        target_buses = self.hv_buses or self.ehv_buses
        if not target_buses:
            return
        total_mvar = self.cfg.dq_mvar + self.cfg.dist_q_mvar
        if abs(total_mvar) < 1e-9:
            factor = self.cfg.dq_percent + self.cfg.dist_q_percent
            if abs(factor) > 1e-9:
                total_mvar = factor * max(
                    sum(abs(val) for val in self.cfg.collector_q_mvar), 1.0
                )
        if abs(total_mvar) < 1e-6:
            return
        base_mva = float(self.ss.config.mva)
        b_each = (total_mvar / base_mva) / len(target_buses)
        for bus in target_buses:
            shunt_idx = self.ss.add(
                "Shunt",
                {
                    "bus": bus,
                    "g": 0.0,
                    "b": b_each,
                    "u": 1,
                    "name": f"DISTQ_{bus}",
                },
            )
            self.dist_shunt_ids.append(shunt_idx)

    def _add_pq_fleet(self) -> None:
        if not hasattr(self.ss, "PQ"):
            return
        target_buses = list(self.cfg.collector_trans_buses)
        if target_buses:
            count = min(len(target_buses), self.cfg.pv_unit_count)
            bus_sequence = target_buses[:count]
        else:
            pq_pairs = [
                (int(self.ss.PQ.bus.v[i]), abs(float(self.ss.PQ.p0.v[i])))
                for i in range(self.ss.PQ.n)
            ]
            pq_pairs.sort(key=lambda item: item[1], reverse=True)
            count = min(self.cfg.pv_unit_count, len(pq_pairs))
            bus_sequence = [bus for bus, _ in pq_pairs[:count]]
        if count == 0:
            return
        per_unit = self.cfg.pv_total_p_pu / count
        for idx, bus in enumerate(bus_sequence):
            pq_idx = self.ss.add(
                "PQ",
                {
                    "bus": bus,
                    "p0": -per_unit,
                    "q0": -0.01,
                    "name": f"IBR_{idx+1}",
                    "u": 1,
                },
            )
            self.source_ids.append(pq_idx)
            self.source_bus_map.append(bus)
            pq_list = list(self.ss.PQ.idx.v)
            pos = pq_list.index(pq_idx)
            self.non_load_pq_indices.add(pos)

    def _attach_collectors(self) -> None:
        taps = self.cfg.collector_taps
        q_list = self.cfg.collector_q_mvar
        thresholds = self.cfg.collector_thresholds
        dwell_offsets = self.cfg.collector_dwell_offsets
        count = min(len(self.source_ids), len(taps), len(q_list))
        if count == 0:
            return
        base_mva = float(self.ss.config.mva)
        next_bus = int(max(self.ss.Bus.idx.v)) + 1
        for i in range(count):
            source_idx = self.source_ids[i]
            trans_bus = self.source_bus_map[i]
            coll_bus = next_bus + i

            self.ss.add(
                "Bus",
                {
                    "idx": coll_bus,
                    "name": f"Collector_{i+1}",
                    "v": 1.0,
                    "basekv": self.cfg.collector_base_kv,
                },
            )

            self.ss.add(
                "Line",
                {
                    "bus1": trans_bus,
                    "bus2": coll_bus,
                    "r": 0.001,
                    "x": 0.08,
                    "b": 0.0,
                    "tap": taps[i],
                    "u": 1,
                    "name": f"GSU_{i+1}",
                },
            )

            pq_list = list(self.ss.PQ.idx.v)
            pos = pq_list.index(source_idx)
            self.ss.PQ.bus.v[pos] = coll_bus

            b_pu = -q_list[i] / base_mva
            shunt_idx = self.ss.add(
                "Shunt",
                {
                    "bus": coll_bus,
                    "g": 0.0,
                    "b": b_pu,
                    "u": 1,
                    "name": f"COLL_SHUNT_{i+1}",
                },
            )

            meta = CollectorMeta(
                name=f"C{i+1}",
                trans_bus=int(trans_bus),
                coll_bus=int(coll_bus),
                tap=float(taps[i]),
                base_kv=self.cfg.collector_base_kv,
                threshold=(
                    float(thresholds[i])
                    if i < len(thresholds)
                    else self.cfg.collector_threshold_pu
                ),
                delay=self.cfg.collector_dwell_s
                + (
                    dwell_offsets[i]
                    if dwell_offsets and i < len(dwell_offsets)
                    else 0.0
                ),
                pq_idx=source_idx,
                gen_ids=[],
                q_absorption=float(q_list[i]),
                shunt_id=shunt_idx,
            )
            self.collectors[meta.name] = meta
            self.collector_trace[meta.name] = []
            self.trans_voltage_trace[meta.name] = []
            self.scada_trace[meta.name] = []

    def record_event(
        self, time: float, code: str, description: str, category: str
    ) -> None:
        self.timeline.append(EventRecord(float(time), code, description, category))

    def apply_ufls(self, time: float) -> None:
        if not hasattr(self.ss, "PQ"):
            return
        for i in range(self.ss.PQ.n):
            if i in self.non_load_pq_indices:
                continue
            p0 = float(self.ss.PQ.p0.v[i])
            if p0 > 0.0:
                self.ss.PQ.p0.v[i] *= self.cfg.ufls_active_fraction
                self.ss.PQ.q0.v[i] *= self.cfg.ufls_reactive_fraction

    def estimate_frequency(self, system: andes.System, t: float) -> float:
        freq = self.cfg.fnom
        if hasattr(system, "GENROU") and system.GENROU.n > 0:
            omegas = np.array(system.GENROU.omega.v, dtype=float)
            status = np.array(system.GENROU.u.v, dtype=float)
            inertias = np.array(system.GENROU.M.v, dtype=float) * np.array(
                system.GENROU.Sn.v, dtype=float
            )
            mask = (status > 0.5) & (omegas > 0.1)
            if np.any(mask):
                weighted = np.dot(inertias[mask], omegas[mask])
                freq = float(weighted / np.sum(inertias[mask])) * self.cfg.fnom
        self.freq_history.append((t, freq))
        return max(freq, 0.0)

    def record_sync_state(self, system: andes.System, t: float) -> None:
        if hasattr(system, "GENROU"):
            status = np.array(system.GENROU.u.v, dtype=float)
            rating = np.array(system.GENROU.Sn.v, dtype=float)
            self.sync_history.append(
                (t, float(status.sum()), float(np.sum(status * rating)))
            )
        else:
            self.sync_history.append((t, 0.0, 0.0))

        for label in self.bus_traces:
            bus = int(label.split("_")[1])
            pos = self.bus_pos.get(bus)
            if pos is not None:
                v = float(system.Bus.v.v[pos])
                self.bus_traces[label].append((t, v))

    def run(self) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        print("\nRunning power flow...")
        self.ss.setup()
        self.ss.PFlow.config.max_iter = 100
        self.ss.PFlow.config.tol = 1e-4
        ok = self.ss.PFlow.run()
        if not ok:
            self.ss.Bus.v0.v[:] = 1.0
            self.ss.Bus.a0.v[:] = 0.0
            ok = self.ss.PFlow.run()
        if not ok:
            raise RuntimeError("Power flow failed for base case.")

        self.bus_pos = {int(bus): i for i, bus in enumerate(self.ss.Bus.idx.v)}
        for bus in self.cfg.observed_transmission_buses:
            if bus in self.bus_pos:
                self.bus_traces[f"bus_{bus}"] = []

        if self.cfg.enable_hvdc_mode_change and self.hvdc.pq_idx:
            pq_list = list(self.ss.PQ.idx.v)
            self.non_load_pq_indices.add(pq_list.index(self.hvdc.pq_idx))

        self.record_sync_state(self.ss, float(self.ss.dae.t))
        for meta in self.collectors.values():
            pos_coll = self.bus_pos.get(meta.coll_bus)
            pos_trans = self.bus_pos.get(meta.trans_bus)
            if pos_coll is not None:
                self.collector_trace[meta.name].append(
                    (0.0, float(self.ss.Bus.v.v[pos_coll]))
                )
            if pos_trans is not None:
                self.trans_voltage_trace[meta.name].append(
                    (0.0, float(self.ss.Bus.v.v[pos_trans]))
                )

        print("\nRunning time-domain simulation...")
        self.ss.TDS.config.tf = self.cfg.sim_tf
        self.ss.TDS.config.tstep = self.cfg.tstep
        self.ss.TDS.config.tol = 1e-4
        self.ss.TDS.config.max_iter = 25
        self.ss.TDS.config.criteria = 0
        self.ss.TDS.config.verbose = 0
        self.ss.TDS.callpert = self.monitor

        self.ss.TDS.init()
        self.ss.TDS.run()

        print("Processing results...")
        time = np.array(self.ss.dae.ts.t)
        y = np.array(self.ss.dae.ts.y)
        v_idx = self.ss.Bus.v.a
        voltages = y[:, v_idx]
        vmax = np.max(voltages, axis=1)
        vmin = np.min(voltages, axis=1)
        vavg = np.mean(voltages, axis=1)

        freq_times = [pt[0] for pt in self.freq_history] or [0.0]
        freq_vals = [pt[1] for pt in self.freq_history] or [self.cfg.fnom]
        freq = np.interp(time, freq_times, freq_vals)

        sync_count = np.zeros_like(time)
        sync_mw = np.zeros_like(time)
        if self.sync_history:
            hist_t = [pt[0] for pt in self.sync_history]
            hist_count = [pt[1] for pt in self.sync_history]
            hist_mw = [pt[2] for pt in self.sync_history]
            sync_count = np.interp(time, hist_t, hist_count)
            sync_mw = np.interp(time, hist_t, hist_mw)

        bus_traces = {}
        bus_basekv = {}
        basekv_array = np.array(self.ss.Bus.Vn.v, dtype=float)
        for label, trace in self.bus_traces.items():
            if not trace:
                continue
            t_samples = [pt[0] for pt in trace]
            v_samples = [pt[1] for pt in trace]
            bus_traces[label] = np.interp(time, t_samples, v_samples)
            bus_num = int(label.split("_")[1])
            pos = self.bus_pos.get(bus_num)
            if pos is not None:
                bus_basekv[label] = float(basekv_array[pos])

        collector_series: Dict[str, np.ndarray] = {}
        trans_series: Dict[str, np.ndarray] = {}
        scada_series: Dict[str, np.ndarray] = {}
        for name, trace in self.collector_trace.items():
            if trace:
                t_samples = [pt[0] for pt in trace]
                v_samples = [pt[1] for pt in trace]
                collector_series[name] = np.interp(time, t_samples, v_samples)
            else:
                collector_series[name] = np.full_like(time, 1.0)
        for name, trace in self.trans_voltage_trace.items():
            if trace:
                t_samples = [pt[0] for pt in trace]
                v_samples = [pt[1] for pt in trace]
                trans_series[name] = np.interp(time, t_samples, v_samples)
            else:
                trans_series[name] = np.full_like(time, 1.0)
        for name, trace in self.scada_trace.items():
            if trace:
                t_samples = [pt[0] for pt in trace]
                v_samples = [pt[1] for pt in trace]
                scada_series[name] = np.interp(time, t_samples, v_samples)
            else:
                scada_series[name] = np.full_like(time, 1.0)

        self.metrics = {
            "time": time,
            "V_max": vmax,
            "V_min": vmin,
            "V_avg": vavg,
            "freq": freq,
            "sync_online_count": sync_count,
            "sync_online_mw": sync_mw,
            "bus_voltages": bus_traces,
            "bus_basekv": bus_basekv,
            "collector_voltages": collector_series,
            "trans_voltages": trans_series,
            "collector_scada": scada_series,
            "reactive_events": list(self.reactive_log),
        }
        return time, self.metrics

    def save_results(self) -> None:
        if not hasattr(self, "metrics"):
            raise RuntimeError("Run the simulation before saving results.")

        metrics = self.metrics
        events_sorted = sorted(self.timeline, key=lambda item: item.time)
        time = metrics["time"]

        bus_keys = sorted(
            metrics["bus_voltages"].keys(), key=lambda x: int(x.split("_")[1])
        )
        with open("iberian_cascade_bus_voltages.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["time_s"] + bus_keys)
            for i in range(len(time)):
                row = [f"{time[i]:.6f}"]
                row.extend(f"{metrics['bus_voltages'][key][i]:.6f}" for key in bus_keys)
                writer.writerow(row)

        collector_keys = sorted(metrics["collector_voltages"].keys())
        with open("iberian_cascade_collector_voltages.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["time_s"] + collector_keys)
            for i in range(len(time)):
                row = [f"{time[i]:.6f}"]
                row.extend(
                    f"{metrics['collector_voltages'][key][i]:.6f}"
                    for key in collector_keys
                )
                writer.writerow(row)
        scada_keys = sorted(metrics.get("collector_scada", {}).keys())
        if scada_keys:
            with open("iberian_cascade_scada_voltages.csv", "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["time_s"] + scada_keys)
                for i in range(len(time)):
                    row = [f"{time[i]:.6f}"]
                    row.extend(
                        f"{metrics['collector_scada'][key][i]:.6f}"
                        for key in scada_keys
                    )
                    writer.writerow(row)

        base_time = self.cfg.start_time
        collector_meta = {
            name: {
                "name": name,
                "trans_bus": meta.trans_bus,
                "collector_bus": meta.coll_bus,
                "tap": meta.tap,
                "offset": 0.0,
                "threshold_pu": meta.threshold,
                "collector_base_kv": meta.base_kv,
                "q_absorption_mvar": meta.q_absorption,
            }
            for name, meta in self.collectors.items()
        }

        collector_thresholds = {
            name: float(info["threshold_pu"]) for name, info in collector_meta.items()
        }

        meta = {
            "events": [event.as_dict(base_time) for event in events_sorted],
            "protection_threshold_pu": self.cfg.collector_threshold_pu,
            "transmission_threshold_pu": float(
                getattr(self.cfg, "threshold_hv_pu", self.cfg.collector_threshold_pu)
            ),
            "lower_voltage_threshold_pu": float(
                getattr(self.cfg, "threshold_lv_pu", self.cfg.collector_threshold_pu)
            ),
            "start_time_local": base_time.strftime("%Y-%m-%d %H:%M:%S"),
            "bus_basekv": {k: float(v) for k, v in metrics["bus_basekv"].items()},
            "collector_basekv": {k: self.collectors[k].base_kv for k in collector_keys},
            "collector_mapping": [collector_meta[name] for name in collector_keys],
            "collector_thresholds_pu": collector_thresholds,
            "reactive_events": [
                {
                    "time": float(entry[0]),
                    "mvar": float(entry[1]),
                    "label": entry[2],
                    "category": entry[3],
                }
                for entry in self.reactive_log
            ],
        }
        with open("iberian_cascade_events.json", "w") as f:
            json.dump(meta, f, indent=2)

        print("\n=== OUTPUT FILES ===")
        print("✓ iberian_cascade_bus_voltages.csv")
        print("✓ iberian_cascade_collector_voltages.csv")
        if scada_keys:
            print("✓ iberian_cascade_scada_voltages.csv")
        print("✓ iberian_cascade_events.json")


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def main() -> None:
    print("=" * 72)
    print("Physics-Based Iberian Cascade with Collector Protection")
    print("=" * 72)
    scenario = CascadeScenario()
    scenario.build()
    scenario.run()
    scenario.save_results()
    print("\nSimulation complete.")


if __name__ == "__main__":
    main()

CascadeConfig = ScenarioConfig
