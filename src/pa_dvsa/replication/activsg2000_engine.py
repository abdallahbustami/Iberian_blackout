"""Simulation engine of the 2000-bus system: builds the ACTIVSg2000 case in ANDES with protected plants, relays,
underfrequency load shedding and out-of-step separation, runs it, and writes its traces and events.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import heapq
import io
import json
import math
import sys
import time
from collections import defaultdict
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from pa_dvsa.andes_adapter import AndesCase, AndesCaseSpec
from pa_dvsa.andes_regularization import regularize_degenerate_blocks

NOMINAL_FREQ_EQ_HZ = 50.0
DEFAULT_OUT_ROOT = Path(__file__).resolve().parents[3] / "results" / "revised" / "activsg2000"


@dataclass(frozen=True, slots=True)
class DefiniteTimeStage:
    name: str
    quantity: str
    direction: str
    pickup: float
    delay_s: float
    dropout_ratio: float = 0.98
    basis: str = ""

    def __post_init__(self) -> None:
        if self.quantity not in {"voltage", "frequency", "angle"}:
            raise ValueError(f"unknown quantity {self.quantity!r}")
        if self.direction not in {"over", "under"}:
            raise ValueError(f"unknown direction {self.direction!r}")
        if not (0.5 < self.dropout_ratio <= 1.0):
            raise ValueError("dropout_ratio must be in (0.5, 1]")
        if self.delay_s < 0:
            raise ValueError("delay must be nonnegative")


FAST_OV_STAGE = DefiniteTimeStage(
    "OVF", "voltage", "over", 1.10, 0.05, 0.98,
    "fast definite-time plant OV (1.10 pu, 50 ms) as in the benchmark studies",
)


def plant_ov_stages(mode: str, ov1_delay_s: float = 1.0) -> tuple[DefiniteTimeStage, ...]:
    if mode == "fast":
        return (FAST_OV_STAGE,)
    if mode != "standard":
        raise ValueError(f"unknown plant OV mode {mode!r}")
    return tuple(st for st in _plant_stages(ov_enabled=True, ov1_delay_s=ov1_delay_s) if st.name in {"OV1", "OV2"})


def _plant_stages(*, ov_enabled: bool, ov1_delay_s: float, ov_mode: str = "standard") -> tuple[DefiniteTimeStage, ...]:
    stages = []
    if ov_enabled and ov_mode == "fast":
        stages.append(FAST_OV_STAGE)
    elif ov_enabled:
        stages += [
            DefiniteTimeStage(
                "OV1", "voltage", "over", 1.10, float(ov1_delay_s), 0.98,
                "NERC PRC-024-3 HVRT boundary (>=1.10 pu may trip after 1.0 s); IEEE 2800-2022 continuous limit 1.10 pu",
            ),
            DefiniteTimeStage(
                "OV2", "voltage", "over", 1.20, 0.16, 0.98,
                "IEEE 1547-2018 OV2 default (1.20 pu, 0.16 s); PRC-024-3 allows instantaneous trip at 1.20 pu",
            ),
        ]
    stages += [
        DefiniteTimeStage("UV1", "voltage", "under", 0.75, 2.0, 0.98, "NERC PRC-024-3 LVRT boundary point (0.75 pu, 2.0 s)"),
        DefiniteTimeStage("UV2", "voltage", "under", 0.45, 0.15, 0.98, "NERC PRC-024-3 LVRT boundary point (0.45 pu, 0.15 s)"),
        DefiniteTimeStage(
            "UF", "frequency", "under", 0.95, 0.20, 0.999,
            "ENTSO-E RfG: units may disconnect below 47.5 Hz (0.95 pu); 0.2 s relay+breaker",
        ),
        DefiniteTimeStage(
            "OF", "frequency", "over", 1.03, 0.20, 0.999,
            "ENTSO-E RfG: units may disconnect above 51.5 Hz (1.03 pu); 0.2 s relay+breaker",
        ),
    ]
    return tuple(stages)


def _generator_stages() -> tuple[DefiniteTimeStage, ...]:
    return (
        DefiniteTimeStage("OV1", "voltage", "over", 1.10, 1.0, 0.98, "NERC PRC-024-3 HVRT boundary (1.10 pu, 1.0 s)"),
        DefiniteTimeStage("OV2", "voltage", "over", 1.20, 0.16, 0.98, "1.20 pu, 0.16 s (as plants)"),
        DefiniteTimeStage("UV1", "voltage", "under", 0.75, 2.0, 0.98, "NERC PRC-024-3 LVRT boundary point (0.75 pu, 2.0 s)"),
        DefiniteTimeStage("UV2", "voltage", "under", 0.45, 0.15, 0.98, "NERC PRC-024-3 LVRT boundary point (0.45 pu, 0.15 s)"),
        DefiniteTimeStage("UF", "frequency", "under", 0.95, 0.20, 0.999, "ENTSO-E RfG 47.5 Hz (0.95 pu)"),
        DefiniteTimeStage("OF", "frequency", "over", 1.03, 0.20, 0.999, "ENTSO-E RfG 51.5 Hz (1.03 pu)"),
    )


@dataclass(frozen=True, slots=True)
class BlockSpec:
    name: str
    p_mw: float
    q_report_mvar: float | None
    report_cluster: str
    source: str


REPORT_BLOCKS: tuple[BlockSpec, ...] = (
    BlockSpec("Granada355", 355.0, 165.0, "E3", "ENTSO-E final report p.14 (355 MW, 165 MVAr absorbed)"),
    BlockSpec("Badajoz582", 582.0, 165.0, "E4", "report event chronology (E4a)"),
    BlockSpec("Badajoz145", 145.0, 37.0, "E4", "report event chronology (E4b)"),
    BlockSpec("Segovia23", 23.0, None, "E5", "report table MW; PF assumption"),
    BlockSpec("Seville550", 550.0, None, "E5", "report table MW; PF assumption"),
    BlockSpec("Segovia94", 94.0, None, "E5", "report table MW; PF assumption"),
    BlockSpec("Badajoz118", 118.0, None, "E5", "report table MW; PF assumption"),
    BlockSpec("Huelva34", 34.0, None, "E5", "report table MW; PF assumption"),
    BlockSpec("Caceres38", 38.0, None, "E5", "report table MW; PF assumption"),
    BlockSpec("Badajoz72", 72.0, None, "E5", "report table MW; PF assumption"),
    BlockSpec("Badajoz16", 16.0, None, "E5", "report table MW; PF assumption"),
    BlockSpec("Caceres41", 41.0, None, "E5", "report table MW; PF assumption"),
    BlockSpec("Cadiz26", 26.0, None, "E5", "report table MW; PF assumption"),
    BlockSpec("Cadiz128", 128.0, None, "E5", "report table MW; PF assumption"),
    BlockSpec("Malaga154", 154.0, None, "E5", "report table MW; PF assumption"),
    BlockSpec("Cuenca530", 530.0, None, "E5", "report table MW; PF assumption"),
)


@dataclass(frozen=True, slots=True)
class PreCascadeAction:
    code: str
    kind: str
    start_s: float
    duration_s: float
    size: float
    unit: str
    description: str
    justification: str


def default_pre_cascade_actions() -> tuple[PreCascadeAction, ...]:
    return (
        PreCascadeAction(
            "OA1a", "line_energization", 1.0, 0.0, 1.0, "circuit",
            "Energize a second circuit in parallel with the 500 kV line of the renewable zone with the largest charging susceptance.",
            "Report: TSOs coupled (meshed) EHV lines before 12:32, which adds line-charging MVAr and raises voltage.",
        ),
        PreCascadeAction(
            "OA1b", "line_energization", 3.0, 0.0, 1.0, "circuit",
            "Energize a second circuit in parallel with the zone 500 kV line with the second-largest charging susceptance.",
            "As OA1a.",
        ),
        PreCascadeAction(
            "OA2", "interchange_reduction", 5.0, 10.0, 1000.0, "MW",
            "Reduce the affected area's export by 1000 MW through governor set points (area governor units down pro rata; the voltage-regulated governor units of the directly tied areas up, pro rata to dispatch).",
            "Report: exports to France were reduced (HVDC set-point change); lower EHV loading reduces I^2X absorption and raises voltage. No HVDC exists in ACTIVSg2000, so the exchange change is modeled as a governor redispatch.",
        ),
        PreCascadeAction(
            "OA3", "fixed_pf_res_reduction", 5.0, 10.0, 208.0, "MW",
            "Reduce the output of all protected plants by 208 MW in total (pro rata) at constant power factor (P and Q scaled together).",
            "Report event E2: 208 MW RES operating-point change; at fixed power factor the reactive absorption falls with P.",
        ),
    )


@dataclass(frozen=True, slots=True)
class ReplicaConfig:
    case_id: str = "activsg2000_stable"
    variant: str = "baseline"
    mode: str = "physics"
    tf_s: float = 60.0
    tstep_s: float = 0.01
    tds_tol: float = 1e-4
    tds_max_iter: int = 25
    pflow_tol: float = 1e-8
    affected_area: int = 7
    load_scale: float = 0.63
    blocks: tuple[BlockSpec, ...] = REPORT_BLOCKS
    fixed_pf: float = 0.95
    zone_size: int = 16
    zone_min_kv: float = 230.0
    gsu_x_on_rating: float = 0.12
    gsu_rating_factor: float = 1.15
    gsu_x_over_r: float = 40.0
    collector_kv: float = 34.5
    tap_rule: str = "margin"
    tap_fixed: float = 0.95
    tap_margin_pu: float = 0.01
    tap_override: float | None = None
    pvd1_ialim: float = 1.3
    plant_ov_enabled: bool = True
    plant_ov1_delay_s: float = 1.0
    generator_protection_enabled: bool = True
    pole_slip_deg: float = 180.0
    ufls_stages: tuple[tuple[float, float], ...] = (
        (0.980, 0.075), (0.976, 0.075), (0.972, 0.075), (0.968, 0.075), (0.964, 0.075), (0.960, 0.075),
    )
    ufls_delay_s: float = 0.15
    separation_enabled: bool = True
    separation_angle_deg: float = 120.0
    separation_delay_s: float = 0.05
    collapse_voltage_pu: float = 0.5
    collapse_f_low_pu: float = 0.95
    collapse_f_high_pu: float = 1.05
    collapse_dwell_s: float = 0.5
    pre_cascade_actions: tuple[PreCascadeAction, ...] = field(default_factory=default_pre_cascade_actions)
    pre_cascade_enabled: bool = True
    preserved_absorption: bool = False
    voltage_mode_res: bool = False
    volt_var_slope: float = 7.33
    candidate_capacitors_mvar: tuple[float, ...] = (50.0, 100.0)
    record_every: int = 2
    forced_separation_at_s: float | None = None
    commitment_c: float | None = None
    dispatch_rule: str = "must_run"
    must_run_fraction: float = 0.20
    plant_ov_mode: str = "standard"
    shadow_plant_ov: bool = False
    oa2_counterpart: str = "exciter_units"
    tie_impedance_scale: float = 1.0
    export_reduction_mw: float = 0.0
    export_counterpart: str = "rest_generation"
    exciter_completion: str = "none"
    completion_sexs: tuple[tuple[str, float], ...] = ()
    initiating_block: str | None = None
    initiating_time_s: float = 15.0
    freeze_other_islands_after_separation: bool = True
    load_model: str = "z"
    initial_mesh_closed: int = 0
    extra_plant_stages: tuple[tuple[str, DefiniteTimeStage], ...] = ()
    area_q_control_tc_s: float = 0.0
    q_control_gain: float = 8.0
    area_ehv_shunt_mvar: float = 0.0
    area_uel_tc_s: float = 0.0
    embedded_pv_share: float = 0.0
    embedded_pv_tranches: tuple[float, ...] = (0.01, 0.02, 0.03, 0.045, 0.06)
    embedded_pv_ov: tuple[tuple[float, float], ...] = ((1.10, 1.5), (1.148, 0.2))
    plant_relay_side: str = "collector"
    block_tap_factor: tuple[tuple[str, float], ...] = ()
    area_pss_completion: bool = False

    @property
    def tanphi(self) -> float:
        return math.tan(math.acos(float(self.fixed_pf)))

    def plant_stages(self) -> tuple[DefiniteTimeStage, ...]:
        return _plant_stages(ov_enabled=self.plant_ov_enabled, ov1_delay_s=self.plant_ov1_delay_s,
                             ov_mode=self.plant_ov_mode)

    def generator_stages(self) -> tuple[DefiniteTimeStage, ...]:
        return _generator_stages()


VARIANTS: dict[str, str] = {
    "baseline": "Physics mode: report-like pre-cascade actions, uniform standard protection, everything after that emergent.",
    "no_pre_cascade_actions": "Control run: no operator actions (OA1-OA3); checks that the operating point is steady.",
    "no_ov_protection": "Plant over-voltage relays disabled (OV1/OV2 removed); all other protection unchanged.",
    "delayed_protection": "Plant OV1 delay 2.0 s (IEEE 1547-2018 default OV1 clearing time) instead of 1.0 s.",
    "preserved_absorption": "A plant trip removes its MW, but a collector reactor switched in at the trip keeps the pre-trip MVAr absorption.",
    "voltage_mode_res": "Protected plants in volt-var mode (PVD1 droop dq/dv = 7.33 pu/pu on rating above the operating voltage), same initial operating point.",
}


def variant_config(variant: str = "baseline", **overrides: Any) -> ReplicaConfig:
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}; choose from {sorted(VARIANTS)}")
    cfg = ReplicaConfig(variant=variant)
    if variant == "no_pre_cascade_actions":
        cfg = replace(cfg, pre_cascade_enabled=False)
    elif variant == "no_ov_protection":
        cfg = replace(cfg, plant_ov_enabled=False)
    elif variant == "delayed_protection":
        cfg = replace(cfg, plant_ov1_delay_s=2.0)
    elif variant == "preserved_absorption":
        cfg = replace(cfg, preserved_absorption=True)
    elif variant == "voltage_mode_res":
        cfg = replace(cfg, voltage_mode_res=True)
    if overrides:
        cfg = replace(cfg, **overrides)
    return cfg


class QuietAndes:
    def __init__(self, quiet: bool) -> None:
        self.quiet = quiet

    def __enter__(self) -> None:
        if self.quiet:
            import logging

            self._buf_out, self._buf_err = io.StringIO(), io.StringIO()
            self._cm_out, self._cm_err = redirect_stdout(self._buf_out), redirect_stderr(self._buf_err)
            self._cm_out.__enter__()
            self._cm_err.__enter__()
            self._logger = logging.getLogger("andes")
            self._level = self._logger.level
            self._logger.setLevel(logging.ERROR + 10)
        return None

    def __exit__(self, *exc: Any) -> bool:
        if self.quiet:
            self._logger.setLevel(self._level)
            self._cm_err.__exit__(*exc)
            self._cm_out.__exit__(*exc)
        return False


def _pos(model: Any, idx: Any) -> int:
    values = list(model.idx.v)
    try:
        return values.index(idx)
    except ValueError:
        pass
    s = str(idx)
    for i, v in enumerate(values):
        if str(v) == s:
            return i
    raise KeyError(f"{model.class_name} has no device {idx!r}")


def _n(system: Any, name: str) -> int:
    return int(getattr(getattr(system, name, None), "n", 0) or 0)


def _dijkstra(adj: dict[int, list[tuple[int, float]]], sources: Iterable[int]) -> dict[int, float]:
    dist: dict[int, float] = {}
    heap: list[tuple[float, int]] = []
    for s in sources:
        dist[int(s)] = 0.0
        heap.append((0.0, int(s)))
    heapq.heapify(heap)
    while heap:
        d, b = heapq.heappop(heap)
        if d > dist.get(b, math.inf):
            continue
        for nb, w in adj.get(b, ()):
            nd = d + w
            if nd < dist.get(nb, math.inf) - 1e-15:
                dist[nb] = nd
                heapq.heappush(heap, (nd, nb))
    return dist


def _components(buses: Sequence[int], edges: Iterable[tuple[int, int]]) -> list[set[int]]:
    parent = {int(b): int(b) for b in buses}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in edges:
        ra, rb = find(int(a)), find(int(b))
        if ra != rb:
            parent[ra] = rb
    groups: dict[int, set[int]] = defaultdict(set)
    for b in parent:
        groups[find(b)].add(b)
    return sorted(groups.values(), key=lambda s: (-len(s), min(s)))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set, frozenset)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        f = float(obj)
        return f if math.isfinite(f) else None
    if isinstance(obj, np.ndarray):
        return _jsonable(obj.tolist())
    if isinstance(obj, Path):
        return str(obj)
    return obj


IEEEST_F2_FAST_POLES = {"A3": 2.0e-3, "A4": 1.0e-6}


def regularize_improper_ieeest(system: Any) -> list[dict[str, Any]]:
    if getattr(system, "is_setup", False):
        raise RuntimeError("regularize_improper_ieeest must be called before System.setup()")
    changes: list[dict[str, Any]] = []
    model = getattr(system, "IEEEST", None)
    if model is None or not model.n:
        return changes
    for i in range(model.n):
        a3, a4 = float(model.A3.v[i]), float(model.A4.v[i])
        a5, a6 = float(model.A5.v[i]), float(model.A6.v[i])
        if a3 == 0.0 and a4 == 0.0 and (a5 != 0.0 or a6 != 0.0):
            model.A3.v[i] = IEEEST_F2_FAST_POLES["A3"]
            model.A4.v[i] = IEEEST_F2_FAST_POLES["A4"]
            changes.append(
                {
                    "model": "IEEEST",
                    "idx": str(model.idx.v[i]),
                    "reason": "improper F2 (A3=A4=0, A5/A6 != 0): output frozen by diag_eps in TDS; fast double pole added",
                    "old": {"A3": a3, "A4": a4},
                    "new": dict(IEEEST_F2_FAST_POLES),
                }
            )
    return changes


def _disable_predefined_events(system: Any) -> None:
    for name in ("Toggle", "Alter", "Fault"):
        model = getattr(system, name, None)
        for i in range(int(getattr(model, "n", 0) or 0)):
            model.u.v[i] = 0


LOAD_P_COMPONENTS = ("Ppf", "Ipeq", "Req")
EXCITER_MODELS = ("ESDC1A", "IEEET1", "SEXS", "ESST3A", "EXST1", "ESAC1A", "EXDC2", "ESST4B", "AC8B", "IEEEX1")
GOVERNOR_MODELS = ("TGOV1", "TGOV1N", "IEEEG1", "HYGOV", "GAST")
PSS_MODELS = ("IEEEST", "ST2CUT", "PSS2A")

GENERIC_SEXS = {"TATB": 0.4, "TB": 5.0, "K": 20.0, "TE": 1.0, "EMIN": -99.0, "EMAX": 99.0}
SCRX_SEXS = {"TATB": 0.1, "TB": 10.01, "K": 100.0, "TE": 0.05, "EMIN": -4.0, "EMAX": 5.0}


LOAD_MODELS = {
    "z": {"p2p": 0.0, "p2i": 0.0, "p2z": 1.0, "q2q": 0.0, "q2i": 0.0, "q2z": 1.0},
    "iz": {"p2p": 0.0, "p2i": 1.0, "p2z": 0.0, "q2q": 0.0, "q2i": 0.0, "q2z": 1.0},
    "es": {"p2p": 0.0, "p2i": 0.8, "p2z": 0.2, "q2q": 0.0, "q2i": 0.0, "q2z": 1.0},
}


def _load_system(cfg: ReplicaConfig, quiet: bool) -> Any:
    if cfg.load_model not in LOAD_MODELS:
        raise ValueError(f"unknown load_model {cfg.load_model!r}")
    spec = AndesCaseSpec(
        cfg.case_id,
        setup=False,
        run_pflow=False,
        init_tds=False,
        load_kwargs={"config_option": ["PV.pv2pq=1", "PFlow.max_iter=100"]},
    )
    with QuietAndes(quiet):
        case = AndesCase.load(spec)
    for key, value in LOAD_MODELS[cfg.load_model].items():
        setattr(case.system.PQ.config, key, value)
    return case.system


@dataclass(slots=True)
class AffectedArea:
    area_id: int
    buses: set[int]
    fragments_moved_out: list[int]
    cut_branch_ids: list[str]
    cut_branch_positions: list[int]
    rest_buses: set[int]


def select_affected_area(system: Any, area_id: int) -> AffectedArea:
    buses = [int(b) for b in system.Bus.idx.v]
    area = {int(b): int(a) for b, a in zip(system.Bus.idx.v, system.Bus.area.v)}
    line = system.Line
    online = [i for i in range(line.n) if float(line.u.v[i]) > 0.5]
    in_area = {b for b in buses if area[b] == int(area_id)}
    internal = [
        (int(line.bus1.v[i]), int(line.bus2.v[i]))
        for i in online
        if int(line.bus1.v[i]) in in_area and int(line.bus2.v[i]) in in_area
    ]
    comps = _components(sorted(in_area), internal)
    main = set(comps[0])
    moved = sorted(set().union(*comps[1:])) if len(comps) > 1 else []
    cut = [i for i in online if (int(line.bus1.v[i]) in main) != (int(line.bus2.v[i]) in main)]
    cut_set = set(cut)
    edges = [(int(line.bus1.v[i]), int(line.bus2.v[i])) for i in online if i not in cut_set]
    gcomps = _components(buses, edges)
    if len(gcomps) != 2:
        raise RuntimeError(
            f"opening the {len(cut)} boundary branches of area {area_id} gives {len(gcomps)} components "
            f"(sizes {[len(c) for c in gcomps][:6]}); expected exactly two"
        )
    rest = set(buses) - main
    return AffectedArea(
        area_id=int(area_id),
        buses=main,
        fragments_moved_out=moved,
        cut_branch_ids=[str(line.idx.v[i]) for i in cut],
        cut_branch_positions=list(cut),
        rest_buses=rest,
    )


@dataclass(slots=True)
class BlockRuntime:
    spec: BlockSpec
    hv_bus: int
    collector_bus: int
    gsu_id: str
    pv_id: str
    pvd1_id: str
    p_sched_mw: float
    q_sched_mvar: float
    q_source: str
    sn_mva: float
    x_sys: float
    r_sys: float
    tap: float = float("nan")
    zth_pu: float = float("nan")
    preserve_shunt_id: str | None = None
    z0_pu: float = float("nan")
    hv_v0_pu: float = float("nan")


@dataclass(slots=True)
class ReplicaSystem:
    system: Any
    cfg: ReplicaConfig
    armed: bool
    area: AffectedArea
    blocks: list[BlockRuntime]
    zone_buses: list[int]
    zone_center: int
    gnet_pvd1_ids: list[str]
    mesh_line_ids: list[str]
    mesh_source_lines: list[str]
    candidate_line_ids: list[str]
    candidate_cap_ids: list[str]
    physical_bus_count: int
    disabled_exciters: list[str]
    regularization: list[dict[str, Any]]
    redispatch_area_gov: list[tuple[str, float]]
    redispatch_rest_gov: list[tuple[str, float]]
    base_notes: dict[str, Any]
    tap: float
    pflow_ok: bool = False
    tds_initialized: bool = False
    embedded_pv: list[dict[str, Any]] = field(default_factory=list)


def _genrou_links(system: Any) -> dict[str, dict[str, list[str]]]:
    links: dict[str, dict[str, list[str]]] = defaultdict(lambda: {"exciter": [], "governor": [], "pss": []})
    exc_by_idx: dict[str, str] = {}
    for kind, names in (("exciter", EXCITER_MODELS), ("governor", GOVERNOR_MODELS)):
        for name in names:
            model = getattr(system, name, None)
            for i in range(int(getattr(model, "n", 0) or 0)):
                syn = str(model.syn.v[i])
                links[syn][kind].append(f"{name}:{model.idx.v[i]}")
                if kind == "exciter":
                    exc_by_idx[str(model.idx.v[i])] = syn
    for name in PSS_MODELS:
        model = getattr(system, name, None)
        for i in range(int(getattr(model, "n", 0) or 0)):
            avr = str(model.avr.v[i])
            syn = exc_by_idx.get(avr)
            if syn is not None:
                links[syn]["pss"].append(f"{name}:{model.idx.v[i]}")
    return links


def _set_u(system: Any, ref: str, value: float) -> None:
    model_name, idx = ref.split(":", 1)
    model = getattr(system, model_name)
    model.u.v[_pos(model, idx)] = float(value)


def line_series_admittance(system: Any, line_idx: Any) -> complex:
    p = _pos(system.Line, line_idx)
    return 1.0 / complex(float(system.Line.r.v[p]) + 1e-8, float(system.Line.x.v[p]) + 1e-8)


def set_line_status(system: Any, line_idx: Any, status: float) -> dict[str, float]:
    line = system.Line
    p = _pos(line, line_idx)
    y = line_series_admittance(system, line_idx) * float(status)
    before = {"ghk": float(line.ghk.v[p]), "bhk": float(line.bhk.v[p])}
    line.u.v[p] = float(status)
    line.ghk.v[p] = y.real
    line.bhk.v[p] = y.imag
    if hasattr(line, "yhk"):
        line.yhk.v[p] = y
    return {"ghk_before": before["ghk"], "bhk_before": before["bhk"], "ghk": y.real, "bhk": y.imag}


def trip_machine_mutations(system: Any, gen_idx: Any) -> list[tuple[str, Any, str, float]]:
    from pa_dvsa.events import generator_trip_mutations

    return [(m.model, m.device_idx, m.attr, float(m.value))
            for m in generator_trip_mutations(system, gen_idx, include_effective_status=True)]


def _apply_attr(system: Any, model_name: str, idx: Any, attr: str, value: float) -> None:
    model = getattr(system, model_name)
    getattr(model, attr).v[_pos(model, idx)] = float(value)


def _zone_and_assignment(
    system: Any, cfg: ReplicaConfig, area: AffectedArea
) -> tuple[int, list[int], dict[str, tuple[int, float]], list[str], dict[str, Any]]:
    bus_ids = [int(b) for b in system.Bus.idx.v]
    bpos = {b: i for i, b in enumerate(bus_ids)}
    vn = {int(b): float(v) for b, v in zip(system.Bus.idx.v, system.Bus.Vn.v)}
    line = system.Line
    adj: dict[int, list[tuple[int, float]]] = defaultdict(list)
    degree: dict[int, set[int]] = defaultdict(set)
    for i in range(line.n):
        if float(line.u.v[i]) <= 0.5:
            continue
        a, b, x = int(line.bus1.v[i]), int(line.bus2.v[i]), abs(float(line.x.v[i]))
        adj[a].append((b, x))
        adj[b].append((a, x))
        degree[a].add(b)
        degree[b].add(a)
    gen_buses = {int(b) for b in system.PV.bus.v} | {int(b) for b in system.Slack.bus.v}
    genrou_gen = {str(g) for g in system.GENROU.gen.v}
    links = _genrou_links(system)
    avr_buses = {
        int(system.GENROU.bus.v[i])
        for i in range(system.GENROU.n)
        if float(system.GENROU.u.v[i]) > 0.5 and links[str(system.GENROU.idx.v[i])]["exciter"]
    }
    static_pv_buses = {
        int(system.PV.bus.v[i])
        for i in range(system.PV.n)
        if float(system.PV.u.v[i]) > 0.5 and str(system.PV.idx.v[i]) not in genrou_gen
        and int(system.PV.bus.v[i]) not in area.buses
    }
    candidates = sorted(
        b for b in area.buses if vn[b] >= cfg.zone_min_kv and b not in gen_buses and len(degree[b]) >= 2
    )
    d_src = _dijkstra(adj, avr_buses | static_pv_buses)
    center = max(candidates, key=lambda b: (round(d_src.get(b, math.inf), 12), -b))
    d_c = _dijkstra(adj, [center])
    zone = sorted(candidates, key=lambda b: (round(d_c.get(b, math.inf), 12), b))[: cfg.zone_size]
    if len(zone) < len(cfg.blocks):
        raise RuntimeError("not enough candidate buses for the protected blocks")

    from scipy import sparse
    from scipy.sparse.linalg import splu

    n = len(bus_ids)
    rows: list[int] = []
    cols: list[int] = []
    vals: list[complex] = []
    for i in range(line.n):
        if float(line.u.v[i]) <= 0.5:
            continue
        a, b = bpos[int(line.bus1.v[i])], bpos[int(line.bus2.v[i])]
        y = 1.0 / complex(float(line.r.v[i]), float(line.x.v[i]))
        t = float(line.tap.v[i]) or 1.0
        bc = 1j * float(line.b.v[i]) / 2.0
        rows += [a, a, b, b]
        cols += [a, b, a, b]
        vals += [(y + bc) / t**2, -y / t, -y / t, y + bc]
    g = system.GENROU
    for i in range(g.n):
        if float(g.u.v[i]) <= 0.5:
            continue
        a = bpos[int(g.bus.v[i])]
        x = float(g.xd2.v[i]) * 100.0 / max(float(g.Sn.v[i]), 1e-6)
        rows.append(a)
        cols.append(a)
        vals.append(1.0 / (1j * x))
    ybus = sparse.csc_matrix((vals, (rows, cols)), shape=(n, n))
    lu = splu(ybus)
    zth: dict[int, float] = {}
    for b in zone:
        e = np.zeros(n, dtype=complex)
        e[bpos[b]] = 1.0
        zth[b] = float(abs(lu.solve(e)[bpos[b]]))
    strong_first = sorted(zone, key=lambda b: (zth[b], b))
    big_first = sorted(cfg.blocks, key=lambda s: (-s.p_mw, s.name))
    assignment = {spec.name: (bus, zth[bus]) for spec, bus in zip(big_first, strong_first)}

    zone_set = set(zone)
    zone_lines = [
        i
        for i in range(line.n)
        if float(line.u.v[i]) > 0.5
        and float(line.trans.v[i]) < 0.5
        and int(line.bus1.v[i]) in zone_set
        and int(line.bus2.v[i]) in zone_set
        and vn[int(line.bus1.v[i])] >= 500.0
        and vn[int(line.bus2.v[i])] >= 500.0
    ]
    zone_lines.sort(key=lambda i: (-float(line.b.v[i]), str(line.idx.v[i])))
    ordered_lines = [str(line.idx.v[i]) for i in zone_lines]
    notes = {
        "candidate_count": len(candidates),
        "zone_center": center,
        "zone_center_distance_to_regulation_pu_x": d_src.get(center),
        "zone_distance_from_center_pu_x": {b: d_c.get(b) for b in zone},
        "zone_zth_pu": zth,
        "zone_500kv_lines_by_charging": ordered_lines,
    }
    return center, zone, assignment, ordered_lines, notes


def build_replica_system(
    cfg: ReplicaConfig,
    *,
    armed: bool = False,
    quiet: bool = True,
    disabled_exciters: Sequence[str] | None = None,
    init_tds: bool = True,
) -> ReplicaSystem:
    if disabled_exciters is None:
        disabled_exciters = detect_nonequilibrium_exciters(cfg, quiet=quiet)
    tap_value = cfg.tap_override
    if tap_value is None:
        tap_value = solve_uniform_tap(cfg, quiet=quiet, disabled_exciters=disabled_exciters)
    rs = _assemble(cfg, armed=armed, quiet=quiet, disabled_exciters=disabled_exciters, tap=float(tap_value))
    _setup_and_pflow(rs, quiet=quiet)
    if cfg.voltage_mode_res:
        _configure_volt_var(rs)
    if init_tds:
        _init_tds(rs, quiet=quiet)
    return rs


_CACHE: dict[str, Any] = {}


def _cache_key(cfg: ReplicaConfig, what: str) -> str:
    relevant = {
        "case_id": cfg.case_id,
        "area": cfg.affected_area,
        "load_scale": cfg.load_scale,
        "blocks": [asdict(b) for b in cfg.blocks],
        "fixed_pf": cfg.fixed_pf,
        "zone": (cfg.zone_size, cfg.zone_min_kv),
        "gsu": (cfg.gsu_x_on_rating, cfg.gsu_rating_factor, cfg.gsu_x_over_r),
        "tap": (cfg.tap_rule, cfg.tap_fixed, cfg.tap_margin_pu, cfg.tap_override),
        "commitment": (cfg.commitment_c, cfg.dispatch_rule, cfg.must_run_fraction),
        "iberian": (cfg.tie_impedance_scale, cfg.export_reduction_mw, cfg.export_counterpart, cfg.exciter_completion,
                    cfg.completion_sexs, cfg.load_model),
        **({"initial_mesh_closed": cfg.initial_mesh_closed} if cfg.initial_mesh_closed else {}),
        **({"area_ehv_shunt_mvar": cfg.area_ehv_shunt_mvar} if cfg.area_ehv_shunt_mvar else {}),
        **({"embedded_pv_share": cfg.embedded_pv_share} if cfg.embedded_pv_share else {}),
        **({"block_tap_factor": list(cfg.block_tap_factor)} if cfg.block_tap_factor else {}),
        **({"area_pss_completion": True} if cfg.area_pss_completion else {}),
        "what": what,
    }
    return hashlib.sha1(json.dumps(relevant, sort_keys=True, default=str).encode()).hexdigest()


def detect_nonequilibrium_exciters(cfg: ReplicaConfig, *, quiet: bool = True, tol: float = 1e-4) -> list[str]:
    key = _cache_key(cfg, "exciters")
    if key in _CACHE:
        return list(_CACHE[key])
    probe_cfg = replace(cfg, tap_override=cfg.tap_fixed if cfg.tap_override is None else cfg.tap_override)
    rs = _assemble(probe_cfg, armed=False, quiet=quiet, disabled_exciters=(), tap=float(probe_cfg.tap_override))
    _setup_and_pflow(rs, quiet=quiet)
    _init_tds(rs, quiet=quiet)
    ss = rs.system
    ss.TDS.fg_update(ss.exist.pflow_tds)
    f = np.asarray(ss.dae.f, dtype=float)
    bad: list[str] = []
    for name in EXCITER_MODELS:
        model = getattr(ss, name, None)
        if model is None or not model.n:
            continue
        for k in range(model.n):
            worst = 0.0
            for state in model.states.values():
                worst = max(worst, abs(float(f[state.a[k]])))
            if worst > tol and float(model.u.v[k]) > 0.5:
                bad.append(f"{name}:{model.idx.v[k]}")
    _CACHE[key] = list(bad)
    return bad


def solve_uniform_tap(
    cfg: ReplicaConfig, *, quiet: bool = True, disabled_exciters: Sequence[str] = ()
) -> float:
    if cfg.tap_rule == "fixed":
        return float(cfg.tap_fixed)
    key = _cache_key(cfg, "tap")
    if key in _CACHE:
        return float(_CACHE[key])
    target = 1.10 - float(cfg.tap_margin_pu)

    def excess(tap: float) -> float:
        rs = _assemble(replace(cfg, tap_override=tap), armed=False, quiet=quiet,
                       disabled_exciters=disabled_exciters, tap=float(tap))
        _setup_and_pflow(rs, quiet=quiet)
        return float(max(b.z0_pu for b in rs.blocks)) - target

    grid = (0.86, 0.90, 0.94, 0.98, 1.02, 1.06)
    vals: list[tuple[float, float]] = []
    for t in grid:
        vals.append((t, excess(t)))
        if len(vals) >= 2 and vals[-2][1] > 0.0 >= vals[-1][1]:
            break
    pair = next(((a, fa, b, fb) for (a, fa), (b, fb) in zip(vals, vals[1:]) if fa > 0.0 >= fb), None)
    if pair is None:
        raise RuntimeError(f"margin target {target:.4f} not bracketed on the tap grid: "
                           f"{[(t, round(f, 4)) for t, f in vals]}")
    a, fa, b, fb = pair
    side = 0
    for _ in range(40):
        c = b - fb * (b - a) / (fb - fa)
        fc = excess(c)
        if abs(fc) < 5e-5:
            tap = round(float(c), 5)
            _CACHE[key] = tap
            return tap
        if fc > 0.0:
            a, fa = c, fc
            if side == -1:
                fb *= 0.5
            side = -1
        else:
            b, fb = c, fc
            if side == 1:
                fa *= 0.5
            side = 1
    raise RuntimeError("GSU tap regula falsi did not converge")


def area_commitment_summary(ss: Any, area: AffectedArea) -> dict[str, Any]:
    g = ss.GENROU
    links = _genrou_links(ss)
    on = [i for i in range(g.n) if float(g.u.v[i]) > 0.5 and int(g.bus.v[i]) in area.buses]
    return {
        "area_online_sync_units": len(on),
        "area_online_sync_mva": float(sum(float(g.Sn.v[i]) for i in on)),
        "area_online_units_with_exciter": int(sum(1 for i in on if links[str(g.idx.v[i])]["exciter"])),
        "area_online_units_with_governor": int(sum(1 for i in on if links[str(g.idx.v[i])]["governor"])),
    }


def decommit_area_units(ss: Any, area: AffectedArea, c_ratio: float, dispatch_mw: float) -> dict[str, Any]:
    g = ss.GENROU
    links = _genrou_links(ss)
    slack_gens = {str(v) for v in ss.Slack.idx.v}
    units = [i for i in range(g.n) if float(g.u.v[i]) > 0.5 and int(g.bus.v[i]) in area.buses]
    cap = float(sum(float(g.Sn.v[i]) for i in units))
    before = {"units": len(units), "mva": cap}
    order = sorted((i for i in units if str(g.gen.v[i]) not in slack_gens),
                   key=lambda i: (float(g.Sn.v[i]), int(g.bus.v[i]), str(g.idx.v[i])))
    limit = float(c_ratio) * float(dispatch_mw)
    removed: list[dict[str, Any]] = []
    for i in order:
        if cap <= limit + 1e-9:
            break
        gid = str(g.idx.v[i])
        g.u.v[i] = 0.0
        for ref in links[gid]["exciter"] + links[gid]["governor"] + links[gid]["pss"]:
            _set_u(ss, ref, 0.0)
        gen = g.gen.v[i]
        others = [k for k in range(g.n) if k != i and str(g.gen.v[k]) == str(gen) and float(g.u.v[k]) > 0.5]
        if not others:
            ss.PV.u.v[_pos(ss.PV, gen)] = 0.0
        cap -= float(g.Sn.v[i])
        removed.append({"genrou": gid, "bus": int(g.bus.v[i]), "sn_mva": float(g.Sn.v[i]),
                        "had_exciter": bool(links[gid]["exciter"])})
    if cap > limit + 1e-9:
        raise RuntimeError(f"cannot reach capacity <= {limit:.0f} MVA by decommitment")
    return {
        "rule": "ascending MVA rating, then bus id, then idx; slack unit kept; stop at capacity <= C x dispatch",
        "c_ratio": float(c_ratio), "dispatch_mw_for_rule": float(dispatch_mw), "capacity_limit_mva": limit,
        "before": before, "decommitted": removed, "decommitted_count": len(removed),
        "decommitted_mva": float(sum(r["sn_mva"] for r in removed)),
    }


def water_fill_capped(p0: np.ndarray, floor: np.ndarray, cap: np.ndarray, target: float) -> tuple[float, np.ndarray]:
    p0 = np.asarray(p0, dtype=float)
    floor = np.minimum(np.asarray(floor, dtype=float), np.asarray(cap, dtype=float))
    cap = np.asarray(cap, dtype=float)

    def total(k: float) -> float:
        return float(np.sum(np.clip(k * p0, floor, cap)))

    if total(0.0) > target + 1e-9:
        raise RuntimeError("must-run output alone exceeds the dispatch target")
    k_hi = float(np.max(cap / np.maximum(p0, 1e-12))) * 1.01
    if total(k_hi) < target - 1e-9:
        raise RuntimeError("online capability cannot supply the dispatch target")
    lo, hi = 0.0, k_hi
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if total(mid) < target:
            lo = mid
        else:
            hi = mid
    k = 0.5 * (lo + hi)
    return k, np.clip(k * p0, floor, cap)


def must_run_dispatch(ss: Any, area: AffectedArea, target_pu: float, must_run_fraction: float) -> dict[str, Any]:
    g = ss.GENROU
    base = float(ss.config.mva)
    links = _genrou_links(ss)
    rating: dict[str, float] = defaultdict(float)
    gov_min: dict[str, float] = defaultdict(float)
    for i in range(g.n):
        if float(g.u.v[i]) <= 0.5:
            continue
        key = str(g.gen.v[i])
        rating[key] += float(g.Sn.v[i]) / base
        for ref in links[str(g.idx.v[i])]["governor"]:
            name, idx = ref.split(":", 1)
            model = getattr(ss, name)
            k = _pos(model, idx)
            if float(model.u.v[k]) > 0.5 and hasattr(model, "VMIN"):
                gov_min[key] += max(float(model.VMIN.v[k]), 0.0) * float(g.Sn.v[i]) / base
    rows = [i for i in range(ss.PV.n)
            if int(ss.PV.bus.v[i]) in area.buses and str(ss.PV.idx.v[i]) in rating
            and float(ss.PV.u.v[i]) > 0.5 and float(ss.PV.p0.v[i]) > 0.0]
    p0 = np.asarray([float(ss.PV.p0.v[i]) for i in rows])
    cap = np.asarray([min(float(ss.PV.pmax.v[i]), rating[str(ss.PV.idx.v[i])]) for i in rows])
    cap = np.maximum(cap, 1e-6)
    floor = np.asarray([max(must_run_fraction * c, gov_min.get(str(ss.PV.idx.v[i]), 0.0))
                        for i, c in zip(rows, cap)])
    k, p_new = water_fill_capped(p0, floor, cap, float(target_pu))
    for i, pn in zip(rows, p_new):
        ss.PV.p0.v[i] = float(pn)
    return {
        "rule": "must_run", "k": k, "units": len(rows), "target_mw": float(target_pu) * base,
        "units_at_must_run": int(np.sum(np.isclose(p_new, np.minimum(floor, cap)) & (k * p0 < floor))),
        "units_at_capability": int(np.sum(np.isclose(p_new, cap) & (k * p0 > cap))),
        "must_run_fraction": must_run_fraction,
    }


def complete_exciters(ss: Any, mode: str, custom: dict[str, float] | None = None) -> dict[str, Any]:
    if mode == "none":
        return {"mode": "none", "added": 0}
    if mode not in {"all", "all_scrx", "all_custom"}:
        raise ValueError(f"unknown exciter_completion {mode!r}")
    if mode == "all_custom":
        if not custom or set(custom) != set(GENERIC_SEXS):
            raise ValueError(f"all_custom needs completion_sexs with keys {sorted(GENERIC_SEXS)}")
        params = dict(custom)
    else:
        params = GENERIC_SEXS if mode == "all" else SCRX_SEXS
    reparameterized = 0
    if mode in {"all_scrx", "all_custom"}:
        sexs = getattr(ss, "SEXS", None)
        for k in range(int(getattr(sexs, "n", 0) or 0)):
            if all(abs(float(getattr(sexs, p).v[k]) - v) < 1e-9 for p, v in GENERIC_SEXS.items()):
                for p, v in params.items():
                    getattr(sexs, p).v[k] = v
                reparameterized += 1
    links = _genrou_links(ss)
    added = []
    for i in range(ss.GENROU.n):
        gid = str(ss.GENROU.idx.v[i])
        if links[gid]["exciter"]:
            continue
        ss.add("SEXS", {"idx": f"SEXSC_{gid}", "name": f"SEXSC_{gid}", "syn": gid, "u": 1, **params})
        added.append(gid)
    return {"mode": mode, "added": len(added), "reparameterized_converted_sexs": reparameterized,
            "parameters": dict(params),
            "basis": {"all": "generic SEXS set ANDES uses for the case's converted exciter records",
                      "all_scrx": "the case's SCRX static-exciter record (TE 0 -> 0.05 s)",
                      "all_custom": "AVR-parameter sensitivity"}[mode]}


def scale_cut_set_impedance(ss: Any, area: AffectedArea, k: float) -> int:
    if float(k) == 1.0:
        return 0
    if not float(k) > 0.0:
        raise ValueError("tie_impedance_scale must be positive")
    line = ss.Line
    for p in area.cut_branch_positions:
        line.x.v[p] = float(line.x.v[p]) * float(k)
        line.r.v[p] = float(line.r.v[p]) * float(k)
    return len(area.cut_branch_positions)


def raise_rest_generation(ss: Any, area: AffectedArea, delta_pu: float) -> dict[str, Any]:
    links = _genrou_links(ss)
    base = float(ss.config.mva)
    rating: dict[str, float] = defaultdict(float)
    for i in range(ss.GENROU.n):
        if float(ss.GENROU.u.v[i]) <= 0.5 or int(ss.GENROU.bus.v[i]) in area.buses:
            continue
        lk = links[str(ss.GENROU.idx.v[i])]
        if lk["exciter"] and lk["governor"]:
            rating[str(ss.GENROU.gen.v[i])] += float(ss.GENROU.Sn.v[i]) / base
    rows = [i for i in range(ss.PV.n) if str(ss.PV.idx.v[i]) in rating and int(ss.PV.bus.v[i]) not in area.buses
            and float(ss.PV.u.v[i]) > 0.5 and float(ss.PV.p0.v[i]) > 0.0]
    p0 = np.asarray([float(ss.PV.p0.v[i]) for i in rows])
    cap = np.maximum(np.asarray([min(float(ss.PV.pmax.v[i]), rating[str(ss.PV.idx.v[i])]) for i in rows]), p0)
    k, p_new = water_fill_capped(p0, p0, cap, float(p0.sum() + delta_pu))
    for i, pn in zip(rows, p_new):
        ss.PV.p0.v[i] = float(pn)
    return {"rule": "rest-of-interconnection governor units with an exciter, pro rata, capped at capability",
            "delta_mw": float(delta_pu) * base, "units": len(rows), "k": float(k),
            "units_at_capability": int(np.sum(np.isclose(p_new, cap) & (cap > p0)))}


def reduce_rest_load(ss: Any, area: AffectedArea, delta_pu: float) -> dict[str, Any]:
    rows = [i for i in range(ss.PQ.n) if int(ss.PQ.bus.v[i]) not in area.buses and float(ss.PQ.u.v[i]) > 0.5]
    total = float(sum(float(ss.PQ.p0.v[i]) for i in rows))
    factor = 1.0 - float(delta_pu) / total
    if not 0.5 < factor < 1.0:
        raise RuntimeError(f"implausible rest-load scaling factor {factor:.3f}")
    for i in rows:
        ss.PQ.p0.v[i] = float(ss.PQ.p0.v[i]) * factor
        ss.PQ.q0.v[i] = float(ss.PQ.q0.v[i]) * factor
    return {"rule": "rest-of-interconnection load scaled down pro rata (constant PF)",
            "delta_mw": float(delta_pu) * float(ss.config.mva), "factor": factor, "loads": len(rows)}


def _assemble(
    cfg: ReplicaConfig,
    *,
    armed: bool,
    quiet: bool,
    disabled_exciters: Sequence[str],
    tap: float,
) -> ReplicaSystem:
    ss = _load_system(cfg, quiet)
    _disable_predefined_events(ss)
    completion = complete_exciters(ss, cfg.exciter_completion, dict(cfg.completion_sexs) or None)
    regularization = [dict(item, source="pa_dvsa.andes_regularization") for item in regularize_degenerate_blocks(ss)]
    regularization += regularize_improper_ieeest(ss)
    for ref in disabled_exciters:
        _set_u(ss, ref, 0.0)
        regularization.append({"model": ref.split(":")[0], "idx": ref.split(":")[1],
                               "reason": "exciter not in equilibrium at TDS init (regulator limits); disabled -> constant field"})
    physical_bus_count = int(ss.Bus.n)
    base_mva = float(ss.config.mva)
    area = select_affected_area(ss, cfg.affected_area)
    bus_ids = [int(b) for b in ss.Bus.idx.v]
    vn = {int(b): float(v) for b, v in zip(ss.Bus.idx.v, ss.Bus.Vn.v)}
    ties_scaled = scale_cut_set_impedance(ss, area, cfg.tie_impedance_scale)

    center, zone, assignment, zone_lines, zone_notes = _zone_and_assignment(ss, cfg, area)

    load_mw_before = 0.0
    for i in range(ss.PQ.n):
        if int(ss.PQ.bus.v[i]) in area.buses:
            load_mw_before += float(ss.PQ.p0.v[i]) * base_mva
            ss.PQ.p0.v[i] *= cfg.load_scale
            ss.PQ.q0.v[i] *= cfg.load_scale
    load_reduction_mw = load_mw_before * (1.0 - cfg.load_scale)

    genrou_gen = {str(gg) for gg in ss.GENROU.gen.v}
    area_sync_pv = [
        i for i in range(ss.PV.n)
        if int(ss.PV.bus.v[i]) in area.buses and str(ss.PV.idx.v[i]) in genrou_gen
        and float(ss.PV.u.v[i]) > 0.5 and float(ss.PV.p0.v[i]) > 0.0
    ]
    sync_mw = sum(float(ss.PV.p0.v[i]) for i in area_sync_pv) * base_mva
    block_mw = sum(b.p_mw for b in cfg.blocks)
    sync_factor = 1.0 - (block_mw + load_reduction_mw) / sync_mw
    if not (0.2 < sync_factor < 1.0):
        raise RuntimeError(f"implausible synchronous re-dispatch factor {sync_factor:.3f}")
    target_pv_mw = sync_mw * sync_factor
    target_pv_mw -= float(cfg.export_reduction_mw)
    commitment = {"commitment_c": cfg.commitment_c, "dispatch_rule": cfg.dispatch_rule}
    if cfg.commitment_c is not None:
        if cfg.dispatch_rule != "must_run":
            raise ValueError("a commitment level requires dispatch_rule='must_run'")
        commitment.update(decommit_area_units(ss, area, float(cfg.commitment_c),
                                              target_pv_mw + float(np.sum(ss.Slack.p0.v)) * base_mva))
    if cfg.dispatch_rule == "common_factor":
        for i in area_sync_pv:
            ss.PV.p0.v[i] *= sync_factor
        commitment["dispatch"] = {"rule": "common_factor", "k": sync_factor}
    elif cfg.dispatch_rule == "must_run":
        commitment["dispatch"] = must_run_dispatch(ss, area, target_pv_mw / base_mva, cfg.must_run_fraction)
    else:
        raise ValueError(f"unknown dispatch rule {cfg.dispatch_rule!r}")
    if cfg.export_reduction_mw > 0.0:
        if cfg.export_counterpart == "rest_generation":
            commitment["export_reduction"] = raise_rest_generation(ss, area, float(cfg.export_reduction_mw) / base_mva)
        elif cfg.export_counterpart == "rest_load":
            commitment["export_reduction"] = reduce_rest_load(ss, area, float(cfg.export_reduction_mw) / base_mva)
        else:
            raise ValueError(f"unknown export_counterpart {cfg.export_counterpart!r}")
    commitment.update(area_commitment_summary(ss, area))

    pss_added: list[str] = []
    if cfg.area_pss_completion:
        links0 = _genrou_links(ss)
        st = ss.IEEEST
        template = None
        for k in range(st.n):
            syn = str(st.syn.v[k]) if hasattr(st, "syn") else ""
            gpos = _pos(ss.GENROU, syn) if syn in list(ss.GENROU.idx.v) else None
            if gpos is not None and int(ss.GENROU.bus.v[gpos]) in area.buses and float(st.u.v[k]) > 0.5:
                template = {p_: getattr(st, p_).v[k] for p_ in ("MODE", "A1", "A2", "A3", "A4", "A5", "A6", "T1", "T2",
                                                                 "T3", "T4", "T5", "T6", "KS", "LSMAX", "LSMIN", "VCU", "VCL")}
                break
        if template is None:
            raise RuntimeError("no IEEEST record in the affected area to copy")
        disabled_set = set(disabled_exciters)
        for i in range(ss.GENROU.n):
            gid = str(ss.GENROU.idx.v[i])
            if float(ss.GENROU.u.v[i]) <= 0.5 or int(ss.GENROU.bus.v[i]) not in area.buses or links0[gid]["pss"]:
                continue
            exc = [r for r in links0[gid]["exciter"] if r not in disabled_set]
            if not exc:
                continue
            idx = f"PSSC_{gid}"
            ss.add("IEEEST", {"idx": idx, "name": idx, "avr": exc[0].split(":", 1)[1], **template})
            pss_added.append(idx)
    commitment["pss_completion"] = pss_added

    gnet_ids: list[str] = []
    for i in range(ss.PV.n):
        pid = ss.PV.idx.v[i]
        if int(ss.PV.bus.v[i]) not in area.buses or str(pid) in genrou_gen or float(ss.PV.u.v[i]) <= 0.5:
            continue
        sn = float(ss.PV.Sn.v[i]) if float(ss.PV.Sn.v[i]) > 0 else 100.0
        idx = f"GNET_{pid}"
        ss.add("PVD1", _pvd1_params(idx, int(ss.PV.bus.v[i]), pid, sn, cfg))
        gnet_ids.append(idx)

    embedded: list[dict[str, Any]] = []
    if cfg.embedded_pv_share > 0.0:
        for i in range(ss.PQ.n):
            bus = int(ss.PQ.bus.v[i])
            p_load = float(ss.PQ.p0.v[i])
            if bus not in area.buses or float(ss.PQ.u.v[i]) <= 0.5 or p_load <= 0.0:
                continue
            p_dg = float(cfg.embedded_pv_share) * p_load
            ss.PQ.p0.v[i] = p_load + p_dg
            pv_id = f"DGPV_{ss.PQ.idx.v[i]}"
            ss.add("PV", {"idx": pv_id, "bus": bus, "p0": p_dg, "q0": 0.0, "v0": 1.0, "qmax": 0.0, "qmin": 0.0,
                          "pmax": 1.2 * p_dg, "pmin": 0.0, "Sn": max(1.2 * p_dg * base_mva, 1.0), "Vn": vn[bus],
                          "name": pv_id})
            dg_id = f"DG_{ss.PQ.idx.v[i]}"
            ss.add("PVD1", _pvd1_params(dg_id, bus, pv_id, max(1.2 * p_dg * base_mva, 1.0), cfg))
            embedded.append({"pvd1_id": dg_id, "pv_id": pv_id, "bus": bus, "p_mw": p_dg * base_mva,
                             "load_idx": str(ss.PQ.idx.v[i])})

    next_bus = max(bus_ids) + 1
    blocks: list[BlockRuntime] = []
    res_fraction = 0.0
    if armed and cfg.pre_cascade_enabled:
        for act in cfg.pre_cascade_actions:
            if act.kind == "fixed_pf_res_reduction":
                res_fraction += act.size / block_mw
    for k, spec in enumerate(cfg.blocks):
        hv, zth = assignment[spec.name]
        p = spec.p_mw
        if spec.q_report_mvar is not None:
            q = -abs(spec.q_report_mvar)
            q_source = "report"
        else:
            q = -cfg.tanphi * p
            q_source = f"PF {cfg.fixed_pf:.2f} absorbing (assumption)"
        scale = 1.0 - res_fraction
        sn = cfg.gsu_rating_factor * p
        x = cfg.gsu_x_on_rating * base_mva / sn
        r = x / cfg.gsu_x_over_r
        cb = next_bus + k
        btap = float(tap) * dict(cfg.block_tap_factor).get(spec.name, 1.0)
        hp = bus_ids.index(hv)
        ss.add("Bus", {"idx": cb, "name": f"{spec.name}_PLANT", "Vn": cfg.collector_kv,
                       "v0": float(ss.Bus.v0.v[hp]) / btap, "a0": float(ss.Bus.a0.v[hp]),
                       "area": cfg.affected_area})
        gsu_id = f"GSU_{spec.name}"
        ss.add("Line", {"idx": gsu_id, "bus1": hv, "bus2": cb, "r": r, "x": x, "b": 0.0, "tap": btap, "phi": 0.0,
                        "trans": 1, "Vn1": vn[hv], "Vn2": cfg.collector_kv, "name": gsu_id})
        pv_id = f"PVPH_{spec.name}"
        qs = q * scale / base_mva
        ss.add("PV", {"idx": pv_id, "bus": cb, "p0": p * scale / base_mva, "q0": qs, "v0": 1.05,
                      "qmax": qs, "qmin": qs, "pmax": 1.2 * p / base_mva, "pmin": 0.0,
                      "Sn": sn, "Vn": cfg.collector_kv, "name": pv_id})
        pvd1_id = f"PVD1_{spec.name}"
        ss.add("PVD1", _pvd1_params(pvd1_id, cb, pv_id, sn, cfg))
        br = BlockRuntime(spec=spec, hv_bus=hv, collector_bus=cb, gsu_id=gsu_id, pv_id=pv_id, pvd1_id=pvd1_id,
                          p_sched_mw=p, q_sched_mvar=q, q_source=q_source, sn_mva=sn, x_sys=x, r_sys=r, tap=btap,
                          zth_pu=zth)
        if cfg.preserved_absorption:
            sid = f"QKEEP_{spec.name}"
            ss.add("Shunt", {"idx": sid, "bus": cb, "g": 0.0, "b": 0.0, "u": 0, "Vn": cfg.collector_kv, "name": sid})
            br.preserve_shunt_id = sid
        blocks.append(br)

    line = ss.Line
    mesh_src = zone_lines[:2]
    cand_src = zone_lines[2:4]
    mesh_ids: list[str] = []
    cand_ids: list[str] = []
    n_mesh_closed = (
        sum(1 for a in cfg.pre_cascade_actions if a.kind == "line_energization")
        if armed and cfg.pre_cascade_enabled else 0
    )
    n_mesh_closed = max(n_mesh_closed, int(cfg.initial_mesh_closed))
    for group, src_list, closed in ((mesh_ids, mesh_src, n_mesh_closed), (cand_ids, cand_src, 0)):
        for j, src in enumerate(src_list):
            status = 1 if j < closed else 0
            p = _pos(line, src)
            idx = f"PAR_{src}"
            ss.add("Line", {"idx": idx, "bus1": int(line.bus1.v[p]), "bus2": int(line.bus2.v[p]),
                            "r": float(line.r.v[p]), "x": float(line.x.v[p]), "b": float(line.b.v[p]),
                            "tap": float(line.tap.v[p]) or 1.0, "phi": 0.0, "trans": 0, "u": status,
                            "Vn1": vn[int(line.bus1.v[p])], "Vn2": vn[int(line.bus2.v[p])], "name": idx})
            group.append(idx)
    ehv_shunts: dict[int, float] = {}
    if cfg.area_ehv_shunt_mvar > 0.0:
        weight: dict[int, float] = defaultdict(float)
        for p in range(line.n):
            if float(line.u.v[p]) <= 0.5 or int(line.trans.v[p]):
                continue
            for b in (int(line.bus1.v[p]), int(line.bus2.v[p])):
                if b in area.buses and vn[b] >= 200.0:
                    weight[b] += 0.5 * abs(float(line.b.v[p]))
        total_w = sum(weight.values())
        for b, w in sorted(weight.items()):
            q = float(cfg.area_ehv_shunt_mvar) * w / total_w
            ss.add("Shunt", {"idx": f"EHVCHG_{b}", "bus": b, "g": 0.0, "b": q / base_mva, "u": 1, "Vn": vn[b],
                             "name": f"EHVCHG_{b}"})
            ehv_shunts[b] = q
    cap_ids: list[str] = []
    for j, mvar in enumerate(cfg.candidate_capacitors_mvar):
        bus = zone[j % len(zone)]
        idx = f"CAPCAND_{j + 1}_{bus}"
        ss.add("Shunt", {"idx": idx, "bus": bus, "g": 0.0, "b": mvar / base_mva, "u": 0, "Vn": vn[bus], "name": idx})
        cap_ids.append(idx)

    links = _genrou_links(ss)
    area_of = {int(b): int(a) for b, a in zip(ss.Bus.idx.v, ss.Bus.area.v)}
    neighbour_areas = sorted({
        area_of[int(b)]
        for p in area.cut_branch_positions
        for b in (line.bus1.v[p], line.bus2.v[p])
        if int(b) not in area.buses
    })
    area_gov: list[tuple[str, float]] = []
    rest_gov: list[tuple[str, float]] = []
    pv_p = {str(ss.PV.idx.v[i]): float(ss.PV.p0.v[i]) for i in range(ss.PV.n)}
    for i in range(ss.GENROU.n):
        if float(ss.GENROU.u.v[i]) <= 0.5:
            continue
        gov = [ref for ref in links[str(ss.GENROU.idx.v[i])]["governor"] if ref.startswith("TGOV1:")]
        if not gov:
            continue
        p_gen = pv_p.get(str(ss.GENROU.gen.v[i]), 0.0)
        if p_gen <= 0.0:
            continue
        bus = int(ss.GENROU.bus.v[i])
        if bus in area.buses:
            area_gov.append((gov[0], p_gen))
        elif area_of[bus] in neighbour_areas and (
            cfg.oa2_counterpart != "exciter_units" or links[str(ss.GENROU.idx.v[i])]["exciter"]
        ):
            rest_gov.append((gov[0], p_gen))
    interchange_mw = sum(a.size for a in cfg.pre_cascade_actions if a.kind == "interchange_reduction") if cfg.pre_cascade_enabled else 0.0
    a_tot = sum(p for _, p in area_gov)
    r_tot = sum(p for _, p in rest_gov)
    area_share = [(ref, interchange_mw / base_mva * p / a_tot) for ref, p in area_gov]
    rest_share = [(ref, interchange_mw / base_mva * p / r_tot) for ref, p in rest_gov]
    if armed and interchange_mw > 0:
        gen_of_gov = {}
        for name in GOVERNOR_MODELS:
            model = getattr(ss, name, None)
            for k in range(int(getattr(model, "n", 0) or 0)):
                gen_of_gov[f"{name}:{model.idx.v[k]}"] = str(model.syn.v[k])
        gen_pv = {str(ss.GENROU.idx.v[i]): str(ss.GENROU.gen.v[i]) for i in range(ss.GENROU.n)}
        for sign, share in ((-1.0, area_share), (+1.0, rest_share)):
            for ref, dp in share:
                pv = gen_pv[gen_of_gov[ref]]
                ss.PV.p0.v[_pos(ss.PV, pv)] += sign * dp

    notes = {
        "load_scale": cfg.load_scale,
        "area_load_mw_before_scaling": load_mw_before,
        "area_load_reduction_mw": load_reduction_mw,
        "area_sync_dispatch_mw_before": sync_mw,
        "sync_dispatch_factor": sync_factor,
        "block_total_mw": block_mw,
        "armed_res_reduction_fraction": res_fraction,
        "interchange_reduction_mw": interchange_mw if armed else 0.0,
        "interchange_counterpart_areas": neighbour_areas,
        "interchange_area_governors": len(area_gov),
        "interchange_counterpart_governors": len(rest_gov),
        "commitment": commitment,
        "exciter_completion": completion,
        "tie_impedance_scale": float(cfg.tie_impedance_scale),
        "ties_scaled": ties_scaled,
        "export_reduction_mw": float(cfg.export_reduction_mw),
        **({"area_ehv_shunt_mvar": float(cfg.area_ehv_shunt_mvar), "area_ehv_shunt_buses": len(ehv_shunts)}
           if ehv_shunts else {}),
        **({"embedded_pv_units": len(embedded), "embedded_pv_mw": sum(d["p_mw"] for d in embedded)}
           if embedded else {}),
        **zone_notes,
    }
    return ReplicaSystem(
        system=ss, cfg=cfg, armed=armed, area=area, blocks=blocks, zone_buses=zone, zone_center=center,
        gnet_pvd1_ids=gnet_ids, mesh_line_ids=mesh_ids, mesh_source_lines=list(mesh_src),
        candidate_line_ids=cand_ids, candidate_cap_ids=cap_ids, physical_bus_count=physical_bus_count,
        disabled_exciters=list(disabled_exciters), regularization=regularization,
        redispatch_area_gov=area_share, redispatch_rest_gov=rest_share, base_notes=notes, tap=float(tap),
        embedded_pv=embedded,
    )


def _pvd1_params(idx: str, bus: int, gen: Any, sn: float, cfg: ReplicaConfig) -> dict[str, Any]:
    return {
        "idx": idx, "name": idx, "bus": bus, "gen": gen, "Sn": float(sn), "fn": 60.0,
        "ialim": float(cfg.pvd1_ialim), "pqflag": 0, "qmx": 0.6, "qmn": -0.6,
        "dqdv": 1.0, "v0": 0.01, "v1": 5.0,
        "vt0": 0.01, "vt1": 0.02, "vt2": 5.0, "vt3": 6.0,
        "ft0": 1.0, "ft1": 1.1, "ft2": 999.0, "ft3": 1000.0, "recflag": 0,
    }


def _setup_and_pflow(rs: ReplicaSystem, *, quiet: bool) -> None:
    from pa_dvsa.replication.robust_pflow import complementarity_pflow

    ss = rs.system
    with QuietAndes(quiet):
        ss.setup()
        ss.PFlow.config.tol = rs.cfg.pflow_tol
        ok, info = complementarity_pflow(ss, tol=rs.cfg.pflow_tol)
    rs.base_notes["pflow"] = info
    if not ok:
        raise RuntimeError(f"replica power flow did not converge: {info.get('reason')}")
    v_phys = np.asarray(ss.Bus.v.v, dtype=float)[: rs.physical_bus_count]
    if v_phys.max() > 1.25 or v_phys.min() < 0.7:
        raise RuntimeError(f"abnormal power-flow solution (bus voltages in [{v_phys.min():.3f}, {v_phys.max():.3f}] pu)")
    rs.pflow_ok = True
    v = np.asarray(ss.Bus.v.v, dtype=float)
    for b in rs.blocks:
        b.z0_pu = float(v[_pos(ss.Bus, b.collector_bus)])
        b.hv_v0_pu = float(v[_pos(ss.Bus, b.hv_bus)])
        q = float(ss.PV.q.v[_pos(ss.PV, b.pv_id)]) * float(ss.config.mva)
        scale = 1.0 - float(rs.base_notes.get("armed_res_reduction_fraction", 0.0))
        if abs(q - b.q_sched_mvar * scale) > 0.5:
            raise RuntimeError(f"{b.spec.name}: power flow Q {q:.2f} MVAr differs from schedule {b.q_sched_mvar * scale:.2f}")


def _configure_volt_var(rs: ReplicaSystem) -> None:
    ss = rs.system
    pvd = ss.PVD1
    base = float(ss.config.mva)
    for b in rs.blocks:
        k = _pos(pvd, b.pvd1_id)
        pvd.v1.v[k] = b.z0_pu + 1e-5
        pvd.dqdv.v[k] = rs.cfg.volt_var_slope * b.sn_mva / base
        pvd.qmn.v[k] = 5.0 * b.sn_mva / base


def _init_tds(rs: ReplicaSystem, *, quiet: bool) -> None:
    ss = rs.system
    cfg = rs.cfg
    ss.TDS.config.tf = cfg.tf_s
    ss.TDS.config.tstep = cfg.tstep_s
    ss.TDS.config.tol = cfg.tds_tol
    ss.TDS.config.max_iter = cfg.tds_max_iter
    ss.TDS.config.criteria = 0
    ss.TDS.config.no_tqdm = 1
    ss.TDS.config.save_every = 0
    if hasattr(ss.PV, "qlim"):
        ss.PV.qlim.enable = False
    with QuietAndes(quiet):
        ss.TDS.init()
    rs.tds_initialized = True
    rs.base_notes["init_algebraic_settle"] = _settle_algebraics(ss, rs.disabled_exciters)
    value, name = max_abs_dxdt(rs)
    rs.base_notes["init_max_abs_dxdt"] = value
    rs.base_notes["init_max_abs_dxdt_state"] = name


def out_of_service_state_mask(ss: Any) -> np.ndarray:
    n = int(ss.dae.n)
    mask = np.zeros(n, dtype=bool)
    for model in ss.exist.pflow_tds.values():
        if not getattr(model, "n", 0) or not hasattr(model, "u"):
            continue
        off = np.flatnonzero(np.asarray(model.u.v, dtype=float) <= 0.5)
        for var in model.states.values():
            for k in off:
                mask[int(var.a[k])] = True
    return mask


def max_abs_dxdt(rs: ReplicaSystem, *, include_out_of_service: bool = False) -> tuple[float, str]:
    ss = rs.system
    ss.TDS.fg_update(ss.exist.pflow_tds)
    f = np.abs(np.asarray(ss.dae.f, dtype=float))
    if not f.size:
        return 0.0, ""
    keep = np.ones(f.size, dtype=bool) if include_out_of_service else ~out_of_service_state_mask(ss)
    worst = int(np.argmax(np.where(keep, f, 0.0)))
    return float(f[worst]), ss.dae.x_name[worst]


def _settle_algebraics(ss: Any, disabled: Sequence[str], *, tol: float = 1e-10, max_iter: int = 8) -> dict[str, Any]:
    from scipy import sparse
    from scipy.sparse.linalg import spsolve

    dae = ss.dae
    idx: list[int] = []
    for ref in disabled:
        name, dev = ref.split(":", 1)
        model = getattr(ss, name)
        k = _pos(model, dev)
        for var in model.algebs.values():
            idx.append(int(var.a[k]))
    ss.TDS.fg_update(ss.exist.pflow_tds)
    g = np.asarray(dae.g, dtype=float)
    g0 = float(np.max(np.abs(g))) if g.size else 0.0
    worst = dae.y_name[int(np.argmax(np.abs(g)))] if g.size else ""
    it = 0
    idx_arr = np.asarray(sorted(set(idx)), dtype=int)
    while idx_arr.size and it < max_iter and float(np.max(np.abs(g[idx_arr]))) > tol:
        ss.j_update(models=ss.exist.pflow_tds)
        gy = dae.gy
        mat = sparse.csc_matrix((np.asarray(gy.V).ravel(), (np.asarray(gy.I).ravel(), np.asarray(gy.J).ravel())),
                                shape=(dae.m, dae.m))
        sub = mat[idx_arr][:, idx_arr]
        dy = spsolve(sub.tocsc(), g[idx_arr])
        y = np.asarray(dae.y, dtype=float)
        y[idx_arr] -= np.atleast_1d(dy)
        dae.y[:] = y
        ss.vars_to_models()
        ss.TDS.fg_update(ss.exist.pflow_tds)
        g = np.asarray(dae.g, dtype=float)
        it += 1
    return {"max_abs_g_before": g0, "variable": worst, "iterations": it, "variables_solved": int(idx_arr.size),
            "max_abs_g_after": float(np.max(np.abs(g))) if g.size else 0.0}


@dataclass(slots=True)
class RelayState:
    picked: np.ndarray
    timer: np.ndarray


def _new_relay_state(n: int) -> RelayState:
    return RelayState(np.zeros(n, dtype=bool), np.zeros(n, dtype=float))


def advance_definite_time(state: RelayState, stage: DefiniteTimeStage, values: np.ndarray, dt: float,
                          active: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(values)
    if stage.direction == "over":
        pick = finite & (values >= stage.pickup)
        hold = finite & (values >= stage.pickup * stage.dropout_ratio)
    else:
        pick = finite & (values <= stage.pickup)
        hold = finite & (values <= stage.pickup / stage.dropout_ratio)
    new_picked = active & (pick | (state.picked & hold))
    state.timer = np.where(new_picked, np.where(state.picked, state.timer + dt, 0.0), 0.0)
    state.picked = new_picked
    return new_picked & (state.timer >= stage.delay_s - 1e-9)


@dataclass(slots=True)
class EventRecord:
    time_s: float
    code: str
    category: str
    description: str
    trigger: str
    device: str = ""
    bus: int | None = None
    measured: dict[str, float] = field(default_factory=dict)
    setting: dict[str, float] = field(default_factory=dict)
    lost_p_mw: float = 0.0
    lost_q_absorption_mvar: float = 0.0
    shed_mw: float = 0.0
    island: str = ""


def accepted_sample(last_t_acc: float | None, t_target: float, h: float) -> tuple[bool, float, float]:
    t_acc = round(float(t_target) - float(h), 9)
    if last_t_acc is None:
        return True, t_acc, 0.0
    if t_acc > last_t_acc + 1e-9:
        return True, t_acc, t_acc - last_t_acc
    return False, float(last_t_acc), 0.0


class PhysicsMonitor:
    def __init__(self, rs: ReplicaSystem) -> None:
        self.rs = rs
        ss = rs.system
        cfg = rs.cfg
        self.cfg = cfg
        self.ss = ss
        self.base = float(ss.config.mva)
        self.bus_ids = np.asarray([int(b) for b in ss.Bus.idx.v])
        self.bpos = {int(b): i for i, b in enumerate(self.bus_ids)}
        self.nphys = rs.physical_bus_count
        self.area_mask = np.zeros(ss.Bus.n, dtype=bool)
        for b in rs.area.buses:
            self.area_mask[self.bpos[b]] = True
        for blk in rs.blocks:
            self.area_mask[self.bpos[blk.collector_bus]] = True
        self.events: list[EventRecord] = []
        self.collector_rows: list[dict[str, Any]] = []
        self.system_rows: list[dict[str, Any]] = []
        self.last_t_acc: float | None = None
        self.step_count = 0
        self.rejected_calls = 0
        self.stopped_reason: str | None = None
        self.stalled_reason: str | None = None
        self.stall_guard = False
        self._small_steps = 0
        self.collapse_declared: dict[str, float] = {}
        self._initiated = False
        self.jump_solves = 0
        self.jump_failures: list[dict[str, Any]] = []
        self.retired = False
        pvd = ss.PVD1
        self.plant_ids = [b.pvd1_id for b in rs.blocks] + list(rs.gnet_pvd1_ids)
        self.plant_pos = np.asarray([_pos(pvd, i) for i in self.plant_ids], dtype=int)
        self.plant_is_block = np.asarray([True] * len(rs.blocks) + [False] * len(rs.gnet_pvd1_ids))
        self.plant_bus_pos = np.asarray([self.bpos[int(pvd.bus.v[p])] for p in self.plant_pos], dtype=int)
        if cfg.plant_relay_side not in {"collector", "hv"}:
            raise ValueError(f"unknown plant_relay_side {cfg.plant_relay_side!r}")
        self.plant_meas_pos = self.plant_bus_pos.copy()
        if cfg.plant_relay_side == "hv":
            for k, blk in enumerate(rs.blocks):
                self.plant_meas_pos[k] = self.bpos[int(blk.hv_bus)]
        bf_idx = list(ss.BusFreq.idx.v) if _n(ss, "BusFreq") else []
        self.plant_bf_pos = np.asarray(
            [bf_idx.index(pvd.busfreq.v[p]) if pvd.busfreq.v[p] in bf_idx else -1 for p in self.plant_pos], dtype=int
        )
        self.plant_online = np.ones(len(self.plant_ids), dtype=bool)
        self.plant_stages = cfg.plant_stages()
        self.plant_relays = {st.name: _new_relay_state(len(self.plant_ids)) for st in self.plant_stages}
        block_names = [b.spec.name for b in rs.blocks]
        self.extra_stages: list[tuple[DefiniteTimeStage, np.ndarray, RelayState]] = []
        for name, st in cfg.extra_plant_stages:
            if name not in block_names:
                raise ValueError(f"extra plant stage for unknown block {name!r}")
            mask = np.zeros(len(self.plant_ids), dtype=bool)
            mask[block_names.index(name)] = True
            self.extra_stages.append((st, mask, _new_relay_state(len(self.plant_ids))))
        self.plant_trip_time = np.full(len(self.plant_ids), np.nan)
        self.plant_max_v = np.full(len(self.plant_ids), -np.inf)
        self.plant_max_v_t = np.full(len(self.plant_ids), np.nan)
        self.plant_first_pickup = np.full(len(self.plant_ids), np.nan)
        self.shadow: dict[str, dict[str, Any]] = {}
        if cfg.shadow_plant_ov:
            for mode in ("standard", "fast"):
                stages = plant_ov_stages(mode, cfg.plant_ov1_delay_s)
                self.shadow[mode] = {
                    "stages": stages,
                    "state": {st.name: _new_relay_state(len(self.plant_ids)) for st in stages},
                    "operate_time": np.full(len(self.plant_ids), np.nan),
                    "stage": [""] * len(self.plant_ids),
                }
        self.block_pref0 = np.asarray([float(pvd.pref0.v[p]) for p in self.plant_pos[: len(rs.blocks)]])
        self.block_qref0 = np.asarray([float(pvd.qref0.v[p]) for p in self.plant_pos[: len(rs.blocks)]])
        g = ss.GENROU
        self.gen_ids = [str(v) for v in g.idx.v]
        self.gen_online = np.asarray(g.u.v, dtype=float) > 0.5
        self.gen_bus_pos = np.asarray([self.bpos[int(b)] for b in g.bus.v], dtype=int)
        self.gen_M = np.asarray(g.M.v, dtype=float)
        links = _genrou_links(ss)
        self.gen_links = [links[i]["exciter"] + links[i]["governor"] + links[i]["pss"] for i in self.gen_ids]
        self.gen_stages = cfg.generator_stages() if cfg.generator_protection_enabled else ()
        self.gen_relays = {st.name: _new_relay_state(len(self.gen_ids)) for st in self.gen_stages}
        self.gen_in_area = self.area_mask[self.gen_bus_pos]
        self.qctl: list[tuple[Any, int, int, float, float]] = []
        if cfg.area_q_control_tc_s > 0.0:
            for i in np.flatnonzero(self.gen_online & self.gen_in_area):
                for ref in links[self.gen_ids[i]]["exciter"]:
                    name, idx = ref.split(":", 1)
                    model = getattr(ss, name)
                    k = _pos(model, idx)
                    if float(model.u.v[k]) > 0.5:
                        self.qctl.append((model, k, int(i), float(g.Qe.v[i]), float(g.Sn.v[i]) / self.base))
        self.uel: list[tuple[Any, int, int, float, float, float]] = []
        if cfg.area_uel_tc_s > 0.0:
            qmin_of = {}
            for mname in ("PV", "Slack"):
                m = getattr(ss, mname)
                for k in range(m.n):
                    qmin_of[str(m.idx.v[k])] = float(m.qmin.v[k])
            for i in np.flatnonzero(self.gen_online & self.gen_in_area):
                qmin = qmin_of.get(str(g.gen.v[i]))
                if qmin is None:
                    continue
                for ref in links[self.gen_ids[i]]["exciter"]:
                    name, idx = ref.split(":", 1)
                    model = getattr(ss, name)
                    k = _pos(model, idx)
                    if float(model.u.v[k]) > 0.5:
                        self.uel.append((model, k, int(i), qmin, float(g.Sn.v[i]) / self.base, float(model.vref0.v[k])))
        self.uel_offset = np.zeros(len(self.uel))
        pq = ss.PQ
        self.load_pos = np.asarray([i for i in range(pq.n) if float(pq.u.v[i]) > 0.5 and float(pq.Req.v[i]) > 0.0], dtype=int)
        self.load_bus_pos = np.asarray([self.bpos[int(pq.bus.v[i])] for i in self.load_pos], dtype=int)
        self.load_req0 = np.asarray(pq.Req.v, dtype=float)[self.load_pos].copy()
        self.load_xeq0 = np.asarray(pq.Xeq.v, dtype=float)[self.load_pos].copy()
        self.load_comp0 = {name: np.asarray(getattr(pq, name).v, dtype=float)[self.load_pos].copy()
                           for name in ("Ppf", "Qpf", "Ipeq", "Iqeq", "Req", "Xeq") if hasattr(pq, name)}
        self.load_mix = (float(pq.config.p2p), float(pq.config.p2i), float(pq.config.p2z))
        self.load_shed_fraction = np.zeros(len(self.load_pos))
        self.load_ramp_factor = np.ones(len(self.load_pos))
        self.load_in_area = self.area_mask[self.load_bus_pos] if len(self.load_pos) else np.zeros(0, dtype=bool)
        v0 = np.asarray(ss.Bus.v.v, dtype=float)
        self.load_p0_mw = np.asarray([self._load_p(int(p), float(v0[b])) for p, b in zip(self.load_pos, self.load_bus_pos)])
        self.ufls_state = [_new_relay_state(len(self.load_pos)) for _ in cfg.ufls_stages]
        self.ufls_done = [np.zeros(len(self.load_pos), dtype=bool) for _ in cfg.ufls_stages]
        self.cut_pos = list(rs.area.cut_branch_positions)
        self.separated = False
        self.sep_state = _new_relay_state(1)
        self._refresh_islands()
        self.coi_sep0: float | None = None
        self.collapse_timers: dict[str, float] = defaultdict(float)
        self.actions = list(cfg.pre_cascade_actions) if (cfg.pre_cascade_enabled and not rs.armed) else []
        self.mesh_done = [False] * len(rs.mesh_line_ids)
        self.gov_pref0: dict[str, float] = {}
        for ref, _ in rs.redispatch_area_gov + rs.redispatch_rest_gov:
            name, idx = ref.split(":", 1)
            model = getattr(ss, name)
            self.gov_pref0[ref] = float(model.pref0.v[_pos(model, idx)])
        pv = ss.PV
        self.spv_pos = np.asarray([i for i in range(pv.n) if float(pv.u.v[i]) > 0.5], dtype=int)
        self.spv_qmax = np.asarray(pv.qmax.v, dtype=float)[self.spv_pos] * self.base
        self.spv_qmin = np.asarray(pv.qmin.v, dtype=float)[self.spv_pos] * self.base
        self.spv_max_exceed = 0.0
        self.spv_worst = ""
        flows0 = cut_branch_flows_mva(rs)
        self.export0_mw = float(np.sum(flows0.real))
        self.cut_abs_p0_mw = float(np.sum(np.abs(flows0.real)))
        self.cum_lost_p = 0.0
        self.cum_lost_q_abs = 0.0
        self.cum_shed = 0.0
        self.cum_gen_trip_mw = 0.0
        self.dg = list(rs.embedded_pv)
        self.dg_lost_mw = 0.0
        if self.dg:
            v0 = np.asarray(ss.Bus.v.v, dtype=float)
            ntr = len(cfg.embedded_pv_tranches)
            self.dg_pos = np.asarray([_pos(pvd, d["pvd1_id"]) for d in self.dg], dtype=int)
            self.dg_bus_pos = np.asarray([self.bpos[int(d["bus"])] for d in self.dg], dtype=int)
            self.dg_p0_mw = np.asarray([d["p_mw"] for d in self.dg])
            self.dg_pref0 = np.asarray(pvd.pref0.v, dtype=float)[self.dg_pos].copy()
            self.dg_qref0 = np.asarray(pvd.qref0.v, dtype=float)[self.dg_pos].copy()
            u1 = float(cfg.embedded_pv_ov[0][0])
            vb0 = v0[self.dg_bus_pos]
            self.dg_ratio = np.stack([u1 / (vb0 + m) for m in cfg.embedded_pv_tranches], axis=1)
            self.dg_on = np.ones((len(self.dg), ntr), dtype=bool)
            self.dg_stages = [DefiniteTimeStage(f"DGOV{j + 1}", "voltage", "over", float(u), float(t), 0.98,
                                                "embedded PV inverter protection at LV (final report Sec. 4.3.3.2)")
                              for j, (u, t) in enumerate(cfg.embedded_pv_ov)]
            self.dg_relays = [_new_relay_state(self.dg_on.size) for _ in self.dg_stages]

    def _refresh_islands(self) -> None:
        ss = self.ss
        line = ss.Line
        u = np.asarray(line.u.v, dtype=float) > 0.5
        a1 = [self.bpos[int(b)] for b in line.bus1.v]
        a2 = [self.bpos[int(b)] for b in line.bus2.v]
        edges = [(a1[i], a2[i]) for i in range(line.n) if u[i]]
        comps = _components(list(range(ss.Bus.n)), edges)
        self.island_of_bus = np.zeros(ss.Bus.n, dtype=int)
        for k, comp in enumerate(comps):
            for b in comp:
                self.island_of_bus[b] = k
        self.n_islands = len(comps)
        self.island_sizes = [len(c) for c in comps]
        self.gen_rel0 = None
        self.gen_rel_unwrapped = None
        area_idx = np.flatnonzero(self.area_mask)
        vals, counts = np.unique(self.island_of_bus[area_idx], return_counts=True)
        self.affected_island = int(vals[np.argmax(counts)])

    def _coi(self, mask: np.ndarray, x: np.ndarray, *, angle: bool = False) -> float:
        w = self.gen_M[mask]
        if w.sum() <= 0:
            return float("nan")
        if angle:
            return float(np.angle(np.sum(w * np.exp(1j * x[mask])) / w.sum()))
        return float(np.sum(w * x[mask]) / w.sum())

    def __call__(self, t_target: float, system: Any) -> None:
        ss = self.ss
        if self.stall_guard and self.stalled_reason is None:
            self._small_steps = self._small_steps + 1 if float(ss.TDS.h) < 1e-4 else 0
            if self._small_steps >= 300:
                self.stalled_reason = f"step size < 1e-4 s for 300 attempts at t={float(t_target):.4f} s"
                ss.TDS.config.tf = float(t_target)
        is_new, t_acc, dt = accepted_sample(self.last_t_acc, float(t_target), float(ss.TDS.h))
        if is_new:
            self.last_t_acc = t_acc
            self.step_count += 1
            self._process_sample(t_acc, dt)
        else:
            self.rejected_calls += 1
        self._apply_ramps(float(t_target))

    def _apply_ramps(self, t: float) -> None:
        ss = self.ss
        for act in self.actions:
            if act.kind not in {"interchange_reduction", "fixed_pf_res_reduction", "area_net_load_ramp"}:
                continue
            frac = 0.0 if t < act.start_s else min(1.0, (t - act.start_s) / max(act.duration_s, 1e-9))
            if act.kind == "area_net_load_ramp":
                if not len(self.load_pos):
                    continue
                area_mw = float(self.load_p0_mw[self.load_in_area].sum())
                rest_mw = float(self.load_p0_mw[~self.load_in_area].sum())
                fac = np.where(self.load_in_area, 1.0 + frac * act.size / area_mw, 1.0 - frac * act.size / rest_mw)
                if np.array_equal(fac, self.load_ramp_factor):
                    continue
                self.load_ramp_factor = fac
                pq = ss.PQ
                keep = 1.0 - self.load_shed_fraction
                for name, vals0 in self.load_comp0.items():
                    if name in LOAD_P_COMPONENTS:
                        getattr(pq, name).v[self.load_pos] = vals0 * fac * keep
                continue
            if act.kind == "interchange_reduction":
                for sign, share in ((-1.0, self.rs.redispatch_area_gov), (+1.0, self.rs.redispatch_rest_gov)):
                    for ref, dp in share:
                        name, idx = ref.split(":", 1)
                        model = getattr(ss, name)
                        model.pref0.v[_pos(model, idx)] = self.gov_pref0[ref] + sign * frac * dp
            else:
                total = sum(b.p_sched_mw for b in self.rs.blocks)
                red = 1.0 - frac * act.size / total
                pvd = ss.PVD1
                for k, _b in enumerate(self.rs.blocks):
                    p = self.plant_pos[k]
                    pvd.pref0.v[p] = self.block_pref0[k] * red
                    pvd.qref0.v[p] = self.block_qref0[k] * red

    def _event(self, rec: EventRecord) -> None:
        self.events.append(rec)

    def _jump(self, t: float, what: str) -> None:
        from pa_dvsa.replication.robust_pflow import damped_frozen_algebraic

        ok, it, res = damped_frozen_algebraic(self.ss)
        self.jump_solves += 1
        if not ok:
            self.jump_failures.append({"t": float(t), "event": what, "iterations": int(it), "max_residual": float(res)})

    def _device_buses(self, model: Any) -> list[int | None]:
        ss = self.ss
        gen_bus = {str(i): int(b) for i, b in zip(ss.GENROU.idx.v, ss.GENROU.bus.v)}
        if "bus" in model.params:
            return [int(b) for b in model.bus.v]
        if "syn" in model.params:
            return [gen_bus.get(str(v)) for v in model.syn.v]
        if "avr" in model.params:
            exc_syn: dict[str, str] = {}
            for name in EXCITER_MODELS:
                m = getattr(ss, name, None)
                for k in range(int(getattr(m, "n", 0) or 0)):
                    exc_syn[str(m.idx.v[k])] = str(m.syn.v[k])
            return [gen_bus.get(exc_syn.get(str(v), ""), None) for v in model.avr.v]
        return [None] * int(model.n)

    def _freeze_other_islands(self, t: float) -> None:
        ss = self.ss
        dae = ss.dae
        tf = np.array(dae.Tf, dtype=float)
        frozen = 0
        unmapped: set[str] = set()
        for name, model in ss.models.items():
            if not int(getattr(model, "n", 0) or 0) or not model.states:
                continue
            buses = self._device_buses(model)
            for k, b in enumerate(buses):
                if b is None:
                    unmapped.add(name)
                    continue
                if int(self.island_of_bus[self.bpos[int(b)]]) == self.affected_island:
                    continue
                for var in model.states.values():
                    a = int(var.a[k])
                    if tf[a] > 0.0:
                        tf[a] = 1e12
                        frozen += 1
        dae.Tf[:] = tf
        from kvxopt import spdiag

        ss.TDS.Teye = spdiag(dae.Tf.tolist())
        other = self.island_of_bus != self.affected_island
        line = ss.Line
        opened = 0
        for p in range(line.n):
            if float(line.u.v[p]) > 0.5 and other[self.bpos[int(line.bus1.v[p])]] and other[self.bpos[int(line.bus2.v[p])]]:
                set_line_status(ss, line.idx.v[p], 0.0)
                opened += 1
        off = 0
        for name in ("PQ", "Shunt", "PV"):
            model = getattr(ss, name, None)
            for k in range(int(getattr(model, "n", 0) or 0)):
                if float(model.u.v[k]) > 0.5 and other[self.bpos[int(model.bus.v[k])]]:
                    model.u.v[k] = 0.0
                    off += 1
        for i in np.flatnonzero(self.gen_online & other[self.gen_bus_pos]):
            ss.GENROU.u.v[i] = 0.0
            for ref in self.gen_links[i]:
                _set_u(ss, ref, 0.0)
            for model_name, idx, attr, mut_value in trip_machine_mutations(ss, self.gen_ids[i]):
                _apply_attr(ss, model_name, idx, attr, mut_value)
            self.gen_online[i] = False
            off += 1
        if len(self.load_pos):
            load_other = other[self.load_bus_pos]
            for done in self.ufls_done:
                done[load_other] = True
        with QuietAndes(True):
            ss.connectivity(info=False)
        self._refresh_islands()
        self.retired = True
        ss.TDS.custom_event = True
        self._event(EventRecord(
            t, "RETIRE_OTHER_ISLANDS", "numerical",
            "After separation the other island is not simulated: its dynamic states are frozen, its internal "
            "branches opened and its devices switched off (the islands are electrically independent, so the "
            "affected island's trajectory is unchanged)", "separation",
            measured={"states_frozen": float(frozen), "branches_opened": float(opened), "devices_off": float(off)},
            setting={"unmapped_models": float(len(unmapped))},
        ))
        self._jump(t, "retire other islands")

    def _initiating_contingency(self, t: float) -> None:
        cfg = self.cfg
        if cfg.initiating_block is None or self._initiated or t < cfg.initiating_time_s - 1e-9:
            return
        self._initiated = True
        names = [b.spec.name for b in self.rs.blocks]
        if cfg.initiating_block not in names:
            raise ValueError(f"initiating block {cfg.initiating_block!r} is not a protected block")
        k = names.index(cfg.initiating_block)
        ss = self.ss
        v = float(ss.Bus.v.v[self.plant_bus_pos[k]])
        p_idx = self.plant_pos[k]
        p = float(ss.PVD1.Ipout_y.v[p_idx]) * v * self.base
        q = float(ss.PVD1.Iqout_y.v[p_idx]) * v * self.base
        self._trip_plant(k, "INIT", t, v, float("nan"), p, q)

    def _process_sample(self, t: float, dt: float) -> None:
        ss = self.ss
        cfg = self.cfg
        self._initiating_contingency(t)
        if self.qctl and dt > 0.0:
            qe = np.asarray(ss.GENROU.Qe.v, dtype=float)
            for model, k, i, q0, sn in self.qctl:
                if self.gen_online[i]:
                    model.vref0.v[k] += dt / cfg.area_q_control_tc_s * (q0 - qe[i]) / (sn * cfg.q_control_gain)
        if self.uel and dt > 0.0:
            qe = np.asarray(ss.GENROU.Qe.v, dtype=float)
            for j, (model, k, i, qmin, sn, vref_init) in enumerate(self.uel):
                if not self.gen_online[i]:
                    continue
                d = dt / cfg.area_uel_tc_s * (qmin - qe[i]) / (sn * cfg.q_control_gain)
                new = max(0.0, self.uel_offset[j] + d)
                model.vref0.v[k] += new - self.uel_offset[j]
                self.uel_offset[j] = new
        v = np.asarray(ss.Bus.v.v, dtype=float)
        omega = np.asarray(ss.GENROU.omega.v, dtype=float)
        delta = np.asarray(ss.GENROU.delta.v, dtype=float)
        pvd = ss.PVD1
        plant_v = v[self.plant_meas_pos]
        bf_f = np.asarray(ss.BusFreq.f.v, dtype=float) if _n(ss, "BusFreq") else np.array([])
        plant_f = np.where(self.plant_bf_pos >= 0, bf_f[np.clip(self.plant_bf_pos, 0, None)] if bf_f.size else np.nan, np.nan)
        ip = np.asarray(pvd.Ipout_y.v, dtype=float)[self.plant_pos]
        iq = np.asarray(pvd.Iqout_y.v, dtype=float)[self.plant_pos]
        plant_p = ip * plant_v * self.base
        plant_q = iq * plant_v * self.base
        gen_on = self.gen_online.copy()

        for k, act in enumerate([a for a in self.actions if a.kind == "line_energization"]):
            if k < len(self.rs.mesh_line_ids) and not self.mesh_done[k] and t >= act.start_s - 1e-9:
                lid = self.rs.mesh_line_ids[k]
                sw = set_line_status(ss, lid, 1.0)
                ss.TDS.custom_event = True
                self.mesh_done[k] = True
                self._refresh_islands()
                self._jump(t, f"energize {lid}")
                lp = _pos(ss.Line, lid)
                self._event(EventRecord(t, act.code, "operator_action", act.description, "scheduled",
                                        device=lid, bus=int(ss.Line.bus1.v[lp]),
                                        measured={"b_pu": float(ss.Line.b.v[lp]),
                                                  "charging_mvar_at_1pu": float(ss.Line.b.v[lp]) * self.base,
                                                  "ghk_pu": sw["ghk"], "bhk_pu": sw["bhk"]}))
        for act in self.actions:
            if act.kind in {"interchange_reduction", "fixed_pf_res_reduction", "area_net_load_ramp"}:
                key = f"_started_{act.code}"
                if t >= act.start_s - 1e-9 and not getattr(self, key, False):
                    setattr(self, key, True)
                    self._event(EventRecord(t, act.code, "operator_action", act.description, "scheduled",
                                            setting={"size": act.size, "duration_s": act.duration_s}))

        isl = self.island_of_bus
        gen_isl = isl[self.gen_bus_pos]
        island_f: dict[int, float] = {}
        for k in set(gen_isl[gen_on].tolist()):
            island_f[k] = self._coi(gen_on & (gen_isl == k), omega)

        trips: list[tuple[int, str]] = []
        for st in self.plant_stages:
            vals = plant_v if st.quantity == "voltage" else plant_f
            op = advance_definite_time(self.plant_relays[st.name], st, vals, dt, self.plant_online)
            for k in np.flatnonzero(op):
                trips.append((int(k), st.name))
        for st, mask, state in self.extra_stages:
            vals = plant_v if st.quantity == "voltage" else plant_f
            op = advance_definite_time(state, st, vals, dt, self.plant_online & mask)
            for k in np.flatnonzero(op):
                trips.append((int(k), st.name))
        ov1 = self.plant_relays.get("OV1") or self.plant_relays.get("OVF")
        if ov1 is not None:
            newly = ov1.picked & np.isnan(self.plant_first_pickup)
            self.plant_first_pickup[newly] = t
        for sh in self.shadow.values():
            active = self.plant_online & np.isnan(sh["operate_time"])
            for st in sh["stages"]:
                op = advance_definite_time(sh["state"][st.name], st, plant_v, dt, active)
                for k in np.flatnonzero(op & np.isnan(sh["operate_time"])):
                    sh["operate_time"][k] = t
                    sh["stage"][k] = st.name
        upd = self.plant_online & (plant_v > self.plant_max_v)
        self.plant_max_v[upd] = plant_v[upd]
        self.plant_max_v_t[upd] = t
        done: set[int] = set()
        for k, stage in trips:
            if k in done or not self.plant_online[k]:
                continue
            done.add(k)
            state = self.plant_relays.get(stage) or next(s for st, m, s in self.extra_stages if st.name == stage and m[k])
            t_pick = t - float(state.timer[k])
            self._trip_plant(k, stage, t, plant_v[k], plant_f[k], plant_p[k], plant_q[k], t_pick)

        if self.dg and np.any(self.dg_on):
            lv = (v[self.dg_bus_pos][:, None] * self.dg_ratio).ravel()
            tripped = np.zeros(self.dg_on.size, dtype=bool)
            for st, state in zip(self.dg_stages, self.dg_relays):
                tripped |= advance_definite_time(state, st, lv, dt, self.dg_on.ravel())
            if np.any(tripped):
                on = self.dg_on.ravel()
                on[tripped] = False
                self.dg_on = on.reshape(self.dg_on.shape)
                frac = self.dg_on.mean(axis=1)
                pvd.pref0.v[self.dg_pos] = self.dg_pref0 * frac
                pvd.qref0.v[self.dg_pos] = self.dg_qref0 * frac
                lost = float(np.sum(self.dg_p0_mw * (1.0 - frac))) - self.dg_lost_mw
                self.dg_lost_mw += lost
                self.cum_lost_p += lost
                ss.TDS.custom_event = True
                self._event(EventRecord(
                    t, "EMBEDDED_PV_TRIP", "embedded_pv_trip",
                    f"{int(tripped.sum())} embedded PV tranches disconnected by LV overvoltage protection", "embedded_pv_ov",
                    measured={"tranches": float(tripped.sum()), "lost_mw": lost, "v_bus_max_pu": float(v[self.dg_bus_pos].max())},
                    lost_p_mw=lost,
                ))

        if self.gen_stages:
            gen_v = v[self.gen_bus_pos]
            gtrips: list[tuple[int, str, float]] = []
            for st in self.gen_stages:
                vals = gen_v if st.quantity == "voltage" else omega
                op = advance_definite_time(self.gen_relays[st.name], st, vals, dt, self.gen_online)
                for i in np.flatnonzero(op):
                    gtrips.append((int(i), st.name, float(vals[i])))
            rel = np.full(len(self.gen_ids), np.nan)
            for k in set(gen_isl[gen_on].tolist()):
                m = gen_on & (gen_isl == k)
                ref = self._coi(m, delta, angle=True)
                rel[m] = np.angle(np.exp(1j * (delta[m] - ref)))
            if self.gen_rel0 is None:
                self.gen_rel0 = rel.copy()
                self.gen_rel_unwrapped = rel.copy()
            else:
                prev = self.gen_rel_unwrapped
                step = np.angle(np.exp(1j * (rel - np.angle(np.exp(1j * prev)))))
                self.gen_rel_unwrapped = np.where(np.isfinite(rel) & np.isfinite(prev), prev + step, prev)
                slip = np.abs(self.gen_rel_unwrapped - self.gen_rel0) >= math.radians(cfg.pole_slip_deg)
                for i in np.flatnonzero(slip & self.gen_online):
                    gtrips.append((int(i), "POLE_SLIP", float(math.degrees(self.gen_rel_unwrapped[i] - self.gen_rel0[i]))))
            gdone: set[int] = set()
            for i, stage, val in gtrips:
                if i in gdone or not self.gen_online[i]:
                    continue
                gdone.add(i)
                self._trip_generator(i, stage, t, val)

        if len(self.load_pos) and cfg.ufls_stages:
            load_isl = isl[self.load_bus_pos]
            lf = np.asarray([island_f.get(int(k), np.nan) for k in load_isl])
            for s_idx, (fpu, frac) in enumerate(cfg.ufls_stages):
                stage = DefiniteTimeStage(f"UFLS{s_idx + 1}", "frequency", "under", fpu, cfg.ufls_delay_s, 0.999)
                active = ~self.ufls_done[s_idx]
                op = advance_definite_time(self.ufls_state[s_idx], stage, lf, dt, active)
                if np.any(op):
                    self._shed(s_idx, frac, op, t, lf)

        if cfg.separation_enabled and not self.separated:
            a_mask = gen_on & self.gen_in_area
            r_mask = gen_on & ~self.gen_in_area
            sep = float("nan")
            if a_mask.any() and r_mask.any():
                sep = self._coi(a_mask, delta, angle=True) - self._coi(r_mask, delta, angle=True)
                sep = float(np.angle(np.exp(1j * sep)))
                if self.coi_sep0 is None:
                    self.coi_sep0 = sep
                    self._sep_unwrapped = sep
                else:
                    self._sep_unwrapped += float(np.angle(np.exp(1j * (sep - np.angle(np.exp(1j * self._sep_unwrapped))))))
                change = abs(self._sep_unwrapped - self.coi_sep0)
                stage = DefiniteTimeStage("OOS", "angle", "over", math.radians(cfg.separation_angle_deg),
                                          cfg.separation_delay_s, 0.999)
                if advance_definite_time(self.sep_state, stage, np.array([change]), dt, np.array([True]))[0]:
                    self._separate(t, math.degrees(change))

        if cfg.forced_separation_at_s is not None and not self.separated and t >= cfg.forced_separation_at_s - 1e-9:
            self._separate(t, float("nan"), forced=True)

        self._check_collapse(t, dt, v, island_f)

        if self.spv_pos.size:
            q = np.asarray(ss.PV.q.v, dtype=float)[self.spv_pos] * self.base
            exc = np.maximum(q - self.spv_qmax, self.spv_qmin - q)
            j = int(np.argmax(exc))
            if exc[j] > self.spv_max_exceed:
                self.spv_max_exceed = float(exc[j])
                self.spv_worst = str(ss.PV.idx.v[self.spv_pos[j]])

        if self.step_count % max(self.cfg.record_every, 1) == 0 or dt == 0.0:
            self._record(t, v, plant_v, plant_p, plant_q, island_f, omega)

    def _trip_plant(self, k: int, stage: str, t: float, v: float, f: float, p: float, q: float,
                    t_pick: float = float("nan")) -> None:
        ss = self.ss
        pvd = ss.PVD1
        p_idx = self.plant_pos[k]
        if float(pvd.u.v[p_idx]) <= 0.5:
            self.plant_online[k] = False
            return
        pvd.u.v[p_idx] = 0.0
        ss.TDS.custom_event = True
        self.plant_online[k] = False
        self.plant_trip_time[k] = t
        name = self.plant_ids[k]
        lost_q = max(0.0, -q)
        kept = 0.0
        if self.plant_is_block[k]:
            blk = self.rs.blocks[k]
            if blk.preserve_shunt_id is not None and q < 0 and v > 0:
                sh = ss.Shunt
                sp = _pos(sh, blk.preserve_shunt_id)
                sh.b.v[sp] = q / self.base / (v * v)
                sh.u.v[sp] = 1.0
                kept = lost_q
            name = blk.spec.name
        setting = {st.name: st.pickup for st in self.plant_stages if st.name == stage}
        st = next((s for s in [*self.plant_stages, *(e[0] for e in self.extra_stages if e[1][k])] if s.name == stage), None)
        if st is not None:
            setting = {"pickup": st.pickup, "delay_s": st.delay_s}
        self.cum_lost_p += p
        self.cum_lost_q_abs += lost_q - kept
        initiating = stage == "INIT"
        self._event(EventRecord(
            t, f"TRIP_{name}", "plant_trip" if self.plant_is_block[k] else "plant_trip_other",
            (f"{name} disconnected: initiating contingency (documented first event)" if initiating
             else f"{name} tripped by plant {stage} relay"),
            "initiating_contingency" if initiating else f"plant_{stage}", device=self.plant_ids[k],
            bus=int(self.bus_ids[self.plant_bus_pos[k]]),
            measured={"v_pu": v, "f_pu": f, "p_mw": p, "q_mvar": q, "pickup_start_s": t_pick}, setting=setting,
            lost_p_mw=p, lost_q_absorption_mvar=lost_q - kept,
            island=str(self.island_of_bus[self.plant_bus_pos[k]]),
        ))
        self._jump(t, f"trip {name}")

    def _trip_generator(self, i: int, stage: str, t: float, value: float) -> None:
        ss = self.ss
        g = ss.GENROU
        if float(g.u.v[i]) <= 0.5:
            self.gen_online[i] = False
            return
        pe = float(g.Pe.v[i]) * self.base
        qe = float(g.Qe.v[i]) * self.base
        g.u.v[i] = 0.0
        for ref in self.gen_links[i]:
            _set_u(ss, ref, 0.0)
        for model_name, idx, attr, mut_value in trip_machine_mutations(ss, self.gen_ids[i]):
            _apply_attr(ss, model_name, idx, attr, mut_value)
        ss.TDS.custom_event = True
        self.gen_online[i] = False
        self.cum_gen_trip_mw += pe
        self.cum_lost_p += pe
        if qe < 0:
            self.cum_lost_q_abs += -qe
        st = next((s for s in self.gen_stages if s.name == stage), None)
        setting = {"pickup": st.pickup, "delay_s": st.delay_s} if st else {"pole_slip_deg": self.cfg.pole_slip_deg}
        self._event(EventRecord(
            t, f"GTRIP_{self.gen_ids[i]}", "generator_trip", f"{self.gen_ids[i]} tripped by generator {stage}",
            f"generator_{stage}", device=self.gen_ids[i], bus=int(self.bus_ids[self.gen_bus_pos[i]]),
            measured={"value": value, "p_mw": pe, "q_mvar": qe}, setting=setting, lost_p_mw=pe,
            lost_q_absorption_mvar=max(0.0, -qe), island=str(self.island_of_bus[self.gen_bus_pos[i]]),
        ))
        if self._last_machine_check(i, t):
            return
        self._jump(t, f"trip {self.gen_ids[i]}")

    def _shed(self, s_idx: int, frac: float, mask: np.ndarray, t: float, lf: np.ndarray) -> None:
        ss = self.ss
        pq = ss.PQ
        v = np.asarray(ss.Bus.v.v, dtype=float)
        shed = 0.0
        islands: set[int] = set()
        for j in np.flatnonzero(mask):
            self.ufls_done[s_idx][j] = True
            self.load_shed_fraction[j] = min(1.0, self.load_shed_fraction[j] + frac)
            p = self.load_pos[j]
            vb = v[self.load_bus_pos[j]]
            before = self._load_p(p, vb)
            for name, vals0 in self.load_comp0.items():
                ramp = self.load_ramp_factor[j] if name in LOAD_P_COMPONENTS else 1.0
                getattr(pq, name).v[p] = vals0[j] * ramp * (1.0 - self.load_shed_fraction[j])
            after = self._load_p(p, vb)
            shed += before - after
            islands.add(int(self.island_of_bus[self.load_bus_pos[j]]))
        self.cum_shed += shed
        fpu, _ = self.cfg.ufls_stages[s_idx]
        self._event(EventRecord(
            t, f"UFLS{s_idx + 1}", "ufls", f"UFLS stage {s_idx + 1} sheds {frac:.1%} of load in island(s) {sorted(islands)}",
            "island_coi_underfrequency", measured={"f_pu_min": float(np.nanmin(lf[mask])), "loads": int(mask.sum())},
            setting={"pickup_pu": fpu, "delay_s": self.cfg.ufls_delay_s}, shed_mw=shed,
            island=",".join(str(k) for k in sorted(islands)),
        ))
        self._jump(t, f"UFLS{s_idx + 1}")

    def _load_p(self, p: int, vb: float) -> float:
        pq = self.ss.PQ
        p2p, p2i, p2z = self.load_mix
        return float(p2p * float(pq.Ppf.v[p]) + p2i * float(pq.Ipeq.v[p]) * vb
                     + p2z * float(pq.Req.v[p]) * vb * vb) * self.base

    def _separate(self, t: float, change_deg: float, *, forced: bool = False) -> None:
        ss = self.ss
        for p in self.cut_pos:
            set_line_status(ss, ss.Line.idx.v[p], 0.0)
        ss.TDS.custom_event = True
        self.separated = True
        self._refresh_islands()
        n_fail = len(self.jump_failures)
        self._jump(t, "separation")
        jump_ok = len(self.jump_failures) == n_fail
        self._event(EventRecord(
            t, "SEPARATION", "forced_separation_check" if forced else "out_of_step_separation",
            (f"Numerical islanding check: cut set ({len(self.cut_pos)} branches) opened by script" if forced else
             f"Out-of-step relay opens the complete cut set ({len(self.cut_pos)} branches) of the affected area"),
            "scripted_check" if forced else "coi_angle_change",
            measured={"angle_change_deg": change_deg, "islands_after": self.n_islands,
                      "jump_solve_converged": float(jump_ok)},
            setting={"threshold_deg": self.cfg.separation_angle_deg, "delay_s": self.cfg.separation_delay_s},
        ))
        if self.cfg.freeze_other_islands_after_separation and not forced:
            self._freeze_other_islands(t)

    def _check_collapse(self, t: float, dt: float, v: np.ndarray, island_f: dict[int, float]) -> None:
        cfg = self.cfg
        isl = self.island_of_bus
        for k in range(self.n_islands):
            if self.island_sizes[k] < 2:
                continue
            key = str(k)
            if key in self.collapse_declared:
                continue
            mask = isl == k
            vmed = float(np.median(v[mask]))
            f = island_f.get(k, float("nan"))
            bad_v = vmed < cfg.collapse_voltage_pu
            bad_f = np.isfinite(f) and (f < cfg.collapse_f_low_pu or f > cfg.collapse_f_high_pu)
            no_sync = not np.isfinite(f)
            if bad_v or bad_f or no_sync:
                self.collapse_timers[key] += dt
            else:
                self.collapse_timers[key] = 0.0
            if self.collapse_timers[key] >= cfg.collapse_dwell_s:
                reason = "median voltage < %.2f pu" % cfg.collapse_voltage_pu if bad_v else (
                    "no online synchronous machine" if no_sync else "COI frequency outside [%.3f, %.3f] pu" % (
                        cfg.collapse_f_low_pu, cfg.collapse_f_high_pu))
                self._declare_collapse(k, t, reason, vmed, f)

    def _declare_collapse(self, k: int, t: float, reason: str, vmed: float, f: float) -> None:
        cfg = self.cfg
        key = str(k)
        self.collapse_declared[key] = t
        affected = k == self.affected_island
        self._event(EventRecord(
            t, f"COLLAPSE_I{k}", "island_collapse",
            f"Island {k} ({self.island_sizes[k]} buses{', affected area' if affected else ''}) meets collapse criterion: {reason}",
            "island_collapse_criterion", measured={"median_v_pu": vmed, "coi_f_pu": f},
            setting={"v_pu": cfg.collapse_voltage_pu, "f_low_pu": cfg.collapse_f_low_pu,
                     "f_high_pu": cfg.collapse_f_high_pu, "dwell_s": cfg.collapse_dwell_s},
            island=key,
        ))
        if affected or self.island_sizes[k] > 0.5 * self.ss.Bus.n:
            self.stopped_reason = f"collapse of island {k} at t={t:.3f} s"
            self.ss.TDS.config.tf = min(float(self.ss.TDS.config.tf), t + 2 * float(self.ss.TDS.h) + 1e-6)

    def _last_machine_check(self, i: int, t: float) -> bool:
        k = int(self.island_of_bus[self.gen_bus_pos[i]])
        if str(k) in self.collapse_declared or self.island_sizes[k] < 2:
            return False
        if np.any(self.gen_online & (self.island_of_bus[self.gen_bus_pos] == k)):
            return False
        v = np.asarray(self.ss.Bus.v.v, dtype=float)
        vmed = float(np.median(v[self.island_of_bus == k]))
        self._declare_collapse(k, t, "no online synchronous machine (last machine tripped)", vmed, float("nan"))
        return True

    def _record(self, t: float, v: np.ndarray, plant_v: np.ndarray, plant_p: np.ndarray, plant_q: np.ndarray,
                island_f: dict[int, float], omega: np.ndarray) -> None:
        row: dict[str, Any] = {"time_s": t}
        for k, blk in enumerate(self.rs.blocks):
            name = blk.spec.name
            on = bool(self.plant_online[k])
            row[f"{name}_v_pu"] = float(plant_v[k]) if on else float("nan")
            row[f"{name}_v_raw_pu"] = float(plant_v[k])
            row[f"{name}_hv_v_pu"] = float(v[self.bpos[blk.hv_bus]])
            row[f"{name}_p_mw"] = float(plant_p[k]) if on else 0.0
            row[f"{name}_q_mvar"] = float(plant_q[k]) if on else 0.0
            row[f"{name}_online"] = int(on)
            ov1 = self.plant_relays.get("OV1")
            row[f"{name}_ov1_timer_s"] = float(ov1.timer[k]) if ov1 is not None else 0.0
        self.collector_rows.append(row)
        phys = v[: self.nphys]
        area_phys = v[: self.nphys][self.area_mask[: self.nphys]]
        pct = np.percentile(phys, [1, 10, 50, 90, 99])
        aff = self.affected_island
        f_aff = island_f.get(aff, float("nan"))
        others = [f for k, f in island_f.items() if k != aff]
        srow = {
            "time_s": t,
            "vmin_pu": float(phys.min()), "vmax_pu": float(phys.max()),
            "v01_pu": float(pct[0]), "v10_pu": float(pct[1]), "v50_pu": float(pct[2]), "v90_pu": float(pct[3]), "v99_pu": float(pct[4]),
            "area_vmin_pu": float(area_phys.min()), "area_vmax_pu": float(area_phys.max()),
            "area_v50_pu": float(np.median(area_phys)),
            "zone_hv_vmax_pu": float(max(v[self.bpos[b]] for b in self.rs.zone_buses)),
            "gen_terminal_vmax_pu": float(np.max(v[self.gen_bus_pos[self.gen_online]])) if self.gen_online.any() else float("nan"),
            "collector_vmax_online_pu": float(np.nanmax(np.where(self.plant_online[: len(self.rs.blocks)], plant_v[: len(self.rs.blocks)], np.nan))) if self.plant_online[: len(self.rs.blocks)].any() else float("nan"),
            "affected_island_f_pu": f_aff,
            "affected_island_f_eq_hz": NOMINAL_FREQ_EQ_HZ * f_aff if np.isfinite(f_aff) else float("nan"),
            "rest_f_pu": float(others[0]) if len(others) == 1 else (float(np.nanmean(others)) if others else f_aff),
            "system_coi_f_pu": self._coi(self.gen_online, omega),
            "coi_angle_change_deg": float(math.degrees(abs(self._sep_unwrapped - self.coi_sep0))) if self.coi_sep0 is not None else 0.0,
            "n_islands": int(self.n_islands),
            "separated": int(self.separated),
            "blocks_online": int(self.plant_online[: len(self.rs.blocks)].sum()),
            "gens_online": int(self.gen_online.sum()),
            "cum_lost_p_mw": self.cum_lost_p,
            "cum_lost_q_absorption_mvar": self.cum_lost_q_abs,
            "cum_ufls_shed_mw": self.cum_shed,
            "cum_generator_trip_mw": self.cum_gen_trip_mw,
            "cum_embedded_pv_lost_mw": self.dg_lost_mw,
        }
        self.system_rows.append(srow)


@dataclass(slots=True)
class RunResult:
    cfg: ReplicaConfig
    rs: ReplicaSystem
    monitor: PhysicsMonitor
    tds_ok: bool
    final_time_s: float
    wall_time_s: float
    summary: dict[str, Any]
    out_dir: Path | None = None


def run_replica(cfg: ReplicaConfig, *, out_dir: Path | None = None, quiet: bool = True,
                armed: bool = False, stall_guard: bool = False) -> RunResult:
    t0 = time.time()
    rs = build_replica_system(cfg, armed=armed, quiet=quiet)
    build_s = time.time() - t0
    monitor = PhysicsMonitor(rs)
    monitor.stall_guard = stall_guard
    ss = rs.system
    ss.TDS.callpert = monitor
    with QuietAndes(quiet):
        tds_ok = bool(ss.TDS.run(no_summary=True))
    final_t = float(ss.dae.t)
    wall = time.time() - t0
    if not tds_ok and monitor.stopped_reason is None:
        monitor._event(EventRecord(
            round(final_t, 6), "TDS_STOP", "tds_nonconvergence",
            "ANDES TDS failed to converge (step size reduced to zero); the run is inconclusive beyond this time. "
            "This is NOT declared a blackout.", "numerical",
        ))
    summary = summarize(cfg, rs, monitor, tds_ok=tds_ok, final_time_s=final_t, wall_s=wall, build_s=build_s)
    summary["event_jump_solves"] = {
        "count": int(monitor.jump_solves), "failures": list(monitor.jump_failures),
        "note": "algebraic variables re-solved with states frozen after every switching event (exact event semantics)",
    }
    summary["static_pv_q_limit_exceedance"] = {
        "checked": int(len(monitor.spv_pos)),
        "max_exceedance_mvar": float(monitor.spv_max_exceed),
        "device": monitor.spv_worst,
        "note": "static generators keep their power-flow PV/PQ status during TDS; a regulating static PV has unlimited Q",
    }
    result = RunResult(cfg, rs, monitor, tds_ok, final_t, wall, summary, out_dir)
    if out_dir is not None:
        write_outputs(result, out_dir)
    return result


def neighbour_rise(mon: "PhysicsMonitor", trips: Sequence[EventRecord], *, settle_s: float = 0.5,
                   window_s: float = 2.0) -> list[dict[str, Any]]:
    return neighbour_rise_from_traces(
        mon.collector_rows, [b.spec.name for b in mon.rs.blocks],
        [{"time_s": e.time_s, "code": e.code, "category": e.category,
          "lost_q_absorption_mvar": e.lost_q_absorption_mvar, "lost_p_mw": e.lost_p_mw} for e in mon.events],
        settle_s=settle_s, window_s=window_s)


def neighbour_rise_from_traces(rows: Sequence[dict[str, Any]], names: Sequence[str], events: Sequence[dict[str, Any]],
                               *, settle_s: float = 0.5, window_s: float = 2.0) -> list[dict[str, Any]]:
    if not rows:
        return []
    t = np.asarray([float(r["time_s"]) for r in rows], dtype=float)
    trips = [e for e in events if e["category"] == "plant_trip"]
    stops = sorted(float(e["time_s"]) for e in events if e["category"] in {"plant_trip", "plant_trip_other", "generator_trip"})
    out = []
    for e in trips:
        tk = float(e["time_s"])
        name = str(e["code"]).replace("TRIP_", "")
        i0 = int(np.searchsorted(t, tk + 1e-9) - 1)
        if i0 < 0:
            continue
        nxt = min((x for x in stops if x > tk + 1e-9), default=float("inf"))
        t_end = min(tk + window_s, nxt - 1e-9)
        t_set = min(tk + settle_s, nxt - 1e-9)
        after = rows[min(i0 + 1, len(rows) - 1)]
        online_after = [n for n in names if n != name and float(after.get(f"{n}_online", 0) or 0) == 1.0]
        per = {}
        for n in online_after:
            v = np.asarray([float(r[f"{n}_v_raw_pu"]) for r in rows], dtype=float)
            v0 = v[i0]
            m = (t > tk) & (t <= t_end)
            vs = float(v[m][-1] if (np.any(m) and t_set < tk + settle_s) else np.interp(t_set, t, v)) - v0
            vp = float(np.max(v[m]) - v0) if np.any(m) else float("nan")
            per[n] = (vs, vp)
        if not per:
            continue
        sb = max(per, key=lambda n: per[n][0])
        pb = max(per, key=lambda n: per[n][1])
        out.append({
            "trip": name, "t": tk, "lost_q_absorption_mvar": float(e.get("lost_q_absorption_mvar", 0.0)),
            "lost_p_mw": float(e.get("lost_p_mw", 0.0)),
            "max_dv_settle_pu": per[sb][0], "max_dv_settle_block": sb,
            "max_dv_peak_pu": per[pb][1], "max_dv_peak_block": pb,
            "median_dv_settle_pu": float(np.median([a for a, _ in per.values()])),
            "window_end_s": t_end, "truncated_by_next_trip": bool(nxt < tk + window_s),
        })
    return out


def trip_causes(events: Sequence[EventRecord], cfg: ReplicaConfig) -> dict[str, str]:
    out: dict[str, str] = {}
    actions = [a for a in cfg.pre_cascade_actions] if cfg.pre_cascade_enabled else []
    trips = [e for e in events if e.category in {"plant_trip", "plant_trip_other", "generator_trip"}]
    for e in events:
        if e.category != "plant_trip":
            continue
        if e.trigger == "initiating_contingency":
            out[e.code] = "initiating contingency"
            continue
        tp = float(e.measured.get("pickup_start_s", e.time_s))
        prev = [x for x in trips if x.time_s <= tp + 1e-9 and x is not e]
        ramps = [a for a in actions if a.duration_s > 0 and a.start_s <= tp <= a.start_s + a.duration_s + 1e-9]
        discrete = [a for a in actions if a.duration_s == 0 and a.start_s <= tp + 1e-9]
        last_action_t = max([a.start_s + a.duration_s for a in actions if a.start_s <= tp + 1e-9], default=-1.0)
        if prev and prev[-1].time_s >= last_action_t:
            out[e.code] = f"after trip {prev[-1].code.replace('TRIP_', '').replace('GTRIP_', '')} (t={prev[-1].time_s:.2f} s)"
        elif ramps:
            out[e.code] = "during ramp " + "+".join(a.code for a in ramps)
        elif discrete and tp - max(a.start_s for a in discrete) < 1.0:
            a = max(discrete, key=lambda a: a.start_s)
            out[e.code] = f"after operator action {a.code} (t={a.start_s:.2f} s)"
        else:
            out[e.code] = "slow post-action dynamics (no preceding trip)"
    return out


def summarize(cfg: ReplicaConfig, rs: ReplicaSystem, mon: PhysicsMonitor, *, tds_ok: bool, final_time_s: float,
              wall_s: float, build_s: float) -> dict[str, Any]:
    ev = mon.events
    block_trips = [e for e in ev if e.category == "plant_trip"]
    other_plant = [e for e in ev if e.category == "plant_trip_other"]
    gen_trips = [e for e in ev if e.category == "generator_trip"]
    ufls = [e for e in ev if e.category == "ufls"]
    sep = [e for e in ev if e.category == "out_of_step_separation"]
    collapse = [e for e in ev if e.category == "island_collapse"]
    sys_rows = mon.system_rows

    def col_max(key: str) -> float:
        vals = [r[key] for r in sys_rows if r.get(key) is not None and np.isfinite(r[key])]
        return float(max(vals)) if vals else float("nan")

    def col_min(key: str) -> float:
        vals = [r[key] for r in sys_rows if r.get(key) is not None and np.isfinite(r[key])]
        return float(min(vals)) if vals else float("nan")

    per_block = {}
    for k, blk in enumerate(rs.blocks):
        trip = next((e for e in block_trips if e.device == blk.pvd1_id), None)
        per_block[blk.spec.name] = {
            "hv_bus": blk.hv_bus, "collector_bus": blk.collector_bus, "p_mw": blk.p_sched_mw, "q_mvar": blk.q_sched_mvar,
            "q_source": blk.q_source, "tap": blk.tap, "z0_pu": blk.z0_pu, "hv_v0_pu": blk.hv_v0_pu,
            "margin_to_ov1_pu": 1.10 - blk.z0_pu,
            "max_v_before_trip_pu": float(mon.plant_max_v[k]) if np.isfinite(mon.plant_max_v[k]) else None,
            "max_v_time_s": float(mon.plant_max_v_t[k]) if np.isfinite(mon.plant_max_v_t[k]) else None,
            "first_ov1_pickup_s": float(mon.plant_first_pickup[k]) if np.isfinite(mon.plant_first_pickup[k]) else None,
            "trip_time_s": trip.time_s if trip else None, "trip_stage": trip.trigger if trip else None,
            "trip_v_pu": trip.measured.get("v_pu") if trip else None,
        }
        for mode, sh in mon.shadow.items():
            t_sh = sh["operate_time"][k]
            per_block[blk.spec.name][f"shadow_{mode}_operate_s"] = float(t_sh) if np.isfinite(t_sh) else None
            per_block[blk.spec.name][f"shadow_{mode}_stage"] = sh["stage"][k] or None
    op_actions = [e for e in ev if e.category == "operator_action"]
    causes = trip_causes(ev, cfg)
    affected_collapse = [e for e in collapse if e.island == str(mon.affected_island)]
    summary = {
        "mode": cfg.mode,
        "variant": cfg.variant,
        "variant_description": VARIANTS.get(cfg.variant, ""),
        "tds_ok": tds_ok,
        "final_time_s": final_time_s,
        "horizon_s": cfg.tf_s,
        "stopped_reason": mon.stopped_reason,
        "wall_time_s": wall_s,
        "build_time_s": build_s,
        "accepted_steps": mon.step_count,
        "rejected_step_calls": mon.rejected_calls,
        "gsu_tap": rs.tap,
        "area_net_export_t0_mw": mon.export0_mw,
        "operator_actions": [{"t": e.time_s, "code": e.code, "description": e.description} for e in op_actions],
        "block_trips": len(block_trips),
        "block_trip_sequence": [{"t": e.time_s, "block": e.code.replace("TRIP_", ""), "trigger": e.trigger,
                                 "v_pu": e.measured.get("v_pu"), "p_mw": e.lost_p_mw,
                                 "lost_q_absorption_mvar": e.lost_q_absorption_mvar,
                                 "pickup_start_s": e.measured.get("pickup_start_s"),
                                 "cause": causes.get(e.code, "")} for e in block_trips],
        "trips_initiated_after_a_previous_trip": int(sum(1 for c in causes.values() if c.startswith("after trip"))),
        "neighbour_rise_per_trip": neighbour_rise(mon, block_trips),
        "other_plant_trips": len(other_plant),
        "generator_trips": len(gen_trips),
        "generator_trip_triggers": sorted({e.trigger for e in gen_trips}),
        "ufls_stages_operated": len(ufls),
        "ufls_shed_mw": float(mon.cum_shed),
        "separation": bool(sep),
        "separation_time_s": sep[0].time_s if sep else None,
        "island_collapses": [{"t": e.time_s, "island": e.island, "description": e.description} for e in collapse],
        "affected_area_collapse": bool(affected_collapse),
        "blackout_emerged": bool(affected_collapse),
        "lost_block_p_mw": float(sum(e.lost_p_mw for e in block_trips)),
        "lost_block_q_absorption_mvar": float(sum(e.lost_q_absorption_mvar for e in block_trips)),
        "lost_total_p_mw": float(mon.cum_lost_p),
        "lost_total_q_absorption_mvar": float(mon.cum_lost_q_abs),
        "max_collector_v_pu": float(np.nanmax([r["collector_vmax_online_pu"] for r in sys_rows])) if sys_rows else float("nan"),
        "max_zone_hv_v_pu": col_max("zone_hv_vmax_pu"),
        "max_area_v_pu": col_max("area_vmax_pu"),
        "max_system_v_pu": col_max("vmax_pu"),
        "max_gen_terminal_v_pu": col_max("gen_terminal_vmax_pu"),
        "control_flatness": {k: (float(np.nanmax([r[k] for r in sys_rows]) - np.nanmin([r[k] for r in sys_rows]))
                                 if sys_rows else None)
                             for k in ("vmax_pu", "vmin_pu", "collector_vmax_online_pu", "zone_hv_vmax_pu",
                                       "affected_island_f_pu")},
        "commitment": {k: v for k, v in rs.base_notes.get("commitment", {}).items() if k != "decommitted"},
        "plant_ov_mode": cfg.plant_ov_mode if cfg.plant_ov_enabled else "off",
        "min_system_v_pu": col_min("vmin_pu"),
        "min_area_v_pu": col_min("area_vmin_pu"),
        "affected_f_eq_hz_min": NOMINAL_FREQ_EQ_HZ * col_min("affected_island_f_pu"),
        "affected_f_eq_hz_max": NOMINAL_FREQ_EQ_HZ * col_max("affected_island_f_pu"),
        "max_coi_angle_change_deg": col_max("coi_angle_change_deg"),
        "per_block": per_block,
        "initial_margin_min_pu": float(min(1.10 - b.z0_pu for b in rs.blocks)),
        "initial_collector_v_pu": {b.spec.name: b.z0_pu for b in rs.blocks},
    }
    summary["interpretation"] = interpret(summary)
    return summary


def interpret(s: dict[str, Any]) -> str:
    parts = []
    n = s["block_trips"]
    if n == 0:
        parts.append("No protected plant tripped: the plant-side voltages stayed below the OV1 pickup (or the relays were disabled).")
    else:
        seq = s["block_trip_sequence"]
        parts.append(f"{n} of 16 protected plants tripped between t={seq[0]['t']:.2f} s and t={seq[-1]['t']:.2f} s "
                     f"({s['lost_block_p_mw']:.0f} MW, {s['lost_block_q_absorption_mvar']:.0f} MVAr absorption lost).")
    parts.append(f"Generator trips: {s['generator_trips']}; UFLS stages: {s['ufls_stages_operated']}; "
                 f"separation: {'yes' if s['separation'] else 'no'}; affected-area collapse: {'yes' if s['affected_area_collapse'] else 'no'}.")
    parts.append(f"Affected-island frequency stayed within {s['affected_f_eq_hz_min']:.3f}-{s['affected_f_eq_hz_max']:.3f} Hz (50-Hz equivalent); "
                 f"system voltage range {s['min_system_v_pu']:.3f}-{s['max_system_v_pu']:.3f} pu.")
    if not s["tds_ok"] and not s["stopped_reason"]:
        parts.append(f"TDS did not converge beyond t={s['final_time_s']:.3f} s (inconclusive, not a blackout).")
    return " ".join(parts)


def case_design(rs: ReplicaSystem) -> dict[str, Any]:
    cfg = rs.cfg
    ss = rs.system
    return {
        "config": _jsonable(asdict(cfg)),
        "plant_protection": [asdict(s) for s in cfg.plant_stages()],
        "generator_protection": [asdict(s) for s in cfg.generator_stages()] if cfg.generator_protection_enabled else [],
        "ufls": {"stages_pu_fraction": cfg.ufls_stages, "delay_s": cfg.ufls_delay_s,
                 "basis": "ENTSO-E NC ER (CE): 49.0->48.0 Hz, 6 steps x 7.5% (45% total), no intentional delay (~150 ms operating time)"},
        "separation": {"threshold_deg": cfg.separation_angle_deg, "delay_s": cfg.separation_delay_s,
                       "measured": "COI rotor-angle difference affected area vs rest, change from pre-disturbance value",
                       "cut_set_branches": rs.area.cut_branch_ids},
        "collapse": {"median_v_pu": cfg.collapse_voltage_pu, "f_low_pu": cfg.collapse_f_low_pu,
                     "f_high_pu": cfg.collapse_f_high_pu, "dwell_s": cfg.collapse_dwell_s},
        "affected_area": {"area_id": rs.area.area_id, "bus_count": len(rs.area.buses),
                          "fragments_moved_out": rs.area.fragments_moved_out,
                          "cut_set_size": len(rs.area.cut_branch_ids)},
        "zone": {"center": rs.zone_center, "buses": rs.zone_buses},
        "blocks": [
            {"name": b.spec.name, "cluster": b.spec.report_cluster, "p_mw": b.p_sched_mw, "q_mvar": b.q_sched_mvar,
             "q_source": b.q_source, "p_source": b.spec.source, "hv_bus": b.hv_bus,
             "hv_kv": float(ss.Bus.Vn.v[_pos(ss.Bus, b.hv_bus)]), "collector_bus": b.collector_bus,
             "gsu": b.gsu_id, "tap": b.tap, "x_pu_sys": b.x_sys, "r_pu_sys": b.r_sys, "sn_mva": b.sn_mva,
             "zth_pu": b.zth_pu, "z0_pu": b.z0_pu, "hv_v0_pu": b.hv_v0_pu, "pvd1": b.pvd1_id}
            for b in rs.blocks
        ],
        "gnet_pvd1": rs.gnet_pvd1_ids,
        "meshing_lines": {"energized_circuits": rs.mesh_line_ids, "parallel_to": rs.mesh_source_lines},
        "candidate_lines": rs.candidate_line_ids,
        "candidate_capacitors": rs.candidate_cap_ids,
        "regularization_and_corrections": rs.regularization,
        "gsu_tap": rs.tap,
        "notes": _jsonable(rs.base_notes),
        "pre_cascade_actions": [asdict(a) for a in cfg.pre_cascade_actions] if cfg.pre_cascade_enabled else [],
    }


def write_outputs(result: RunResult, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    mon = result.monitor
    (out_dir / "summary.json").write_text(json.dumps(_jsonable(result.summary), indent=2) + "\n", encoding="utf-8")
    (out_dir / "events.json").write_text(json.dumps(_jsonable([asdict(e) for e in mon.events]), indent=2) + "\n",
                                         encoding="utf-8")
    (out_dir / "case_design.json").write_text(json.dumps(_jsonable(case_design(result.rs)), indent=2) + "\n",
                                              encoding="utf-8")
    _write_csv(out_dir / "collector_traces.csv", mon.collector_rows)
    _write_csv(out_dir / "system_traces.csv", mon.system_rows)


def cut_branch_flows_mva(rs: ReplicaSystem) -> np.ndarray:
    ss = rs.system
    line = ss.Line
    v = np.asarray(ss.Bus.v.v, dtype=float)
    a = np.asarray(ss.Bus.a.v, dtype=float)
    out = []
    for p in rs.area.cut_branch_positions:
        b1, b2 = int(line.bus1.v[p]), int(line.bus2.v[p])
        i1, i2 = _pos(ss.Bus, b1), _pos(ss.Bus, b2)
        v1 = v[i1] * np.exp(1j * a[i1])
        v2 = v[i2] * np.exp(1j * a[i2])
        y = 1.0 / complex(float(line.r.v[p]), float(line.x.v[p]))
        t = (float(line.tap.v[p]) or 1.0) * np.exp(1j * float(line.phi.v[p]))
        bc = 1j * float(line.b.v[p]) / 2.0
        i_from = (v1 / t - v2) * y / np.conj(t) + v1 * bc / abs(t) ** 2
        i_to = (v2 - v1 / t) * y + v2 * bc
        out.append(v1 * np.conj(i_from) if b1 in rs.area.buses else v2 * np.conj(i_to))
    return np.asarray(out, dtype=complex) * float(ss.config.mva)


def area_net_export_mw(rs: ReplicaSystem) -> float:
    return float(np.sum(cut_branch_flows_mva(rs).real))


def islanding_check(cfg: ReplicaConfig | None = None, *, t_open_s: float = 1.0, tf_s: float = 10.0,
                    quiet: bool = True, out_path: Path | None = None) -> dict[str, Any]:
    base = cfg or ReplicaConfig()
    probe = build_replica_system(replace(base, pre_cascade_enabled=False), armed=False, quiet=quiet, init_tds=False)
    export = area_net_export_mw(probe)
    out: dict[str, Any] = {"area_net_export_mw_base": export}
    out["A_base_export"] = _islanding_case(base, t_open_s=t_open_s, tf_s=tf_s, quiet=quiet, balance_mw=0.0)
    out["B_balanced"] = _islanding_case(base, t_open_s=t_open_s, tf_s=tf_s, quiet=quiet, balance_mw=export)
    out["C_single_machine"] = small_island_check(base, quiet=quiet)
    out["andes_can_integrate_an_island"] = bool(out["C_single_machine"]["tds_ok"])
    out["affected_area_separation_integrable"] = bool(out["A_base_export"]["andes_handles_island"]
                                                      and out["B_balanced"]["andes_handles_island"])
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(_jsonable(out), indent=2) + "\n", encoding="utf-8")
    return out


def small_island_check(base: ReplicaConfig | None = None, *, t_open_s: float = 1.0, tf_s: float = 4.0,
                       quiet: bool = True) -> dict[str, Any]:
    base = base or ReplicaConfig()
    cfg = replace(base, variant="small_island_check", tf_s=tf_s, pre_cascade_enabled=False, plant_ov_enabled=False,
                  generator_protection_enabled=False, ufls_stages=(), separation_enabled=False)
    rs = build_replica_system(cfg, quiet=quiet)
    ss = rs.system
    line = ss.Line
    g = ss.GENROU
    adj: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for k in range(line.n):
        if float(line.u.v[k]) > 0.5:
            a, b = int(line.bus1.v[k]), int(line.bus2.v[k])
            adj[a].append((b, k))
            adj[b].append((a, k))
    gen_bus = {int(g.bus.v[i]) for i in range(g.n) if float(g.u.v[i]) > 0.5}
    found: tuple[int, set[int]] | None = None
    for i in range(line.n):
        if float(line.u.v[i]) <= 0.5:
            continue
        ends = (int(line.bus1.v[i]), int(line.bus2.v[i]))
        if any(e in rs.area.buses for e in ends):
            continue
        for side in ends:
            seen, stack, big = {side}, [side], False
            while stack and not big:
                c = stack.pop()
                for o, k in adj[c]:
                    if k == i or o in seen:
                        continue
                    seen.add(o)
                    stack.append(o)
                    if len(seen) > 15:
                        big = True
                        break
            if not big and seen & gen_bus:
                found = (i, seen)
                break
        if found:
            break
    if found is None:
        return {"found": False}
    branch, island = found
    mon = PhysicsMonitor(rs)
    state = {"done": False}

    def cb(t: float, system: Any) -> None:
        mon(t, system)
        if not state["done"] and mon.last_t_acc is not None and mon.last_t_acc >= t_open_s - 1e-9:
            set_line_status(system, system.Line.idx.v[branch], 0.0)
            system.TDS.custom_event = True
            state["done"] = True
            mon._refresh_islands()

    ss.TDS.callpert = cb
    with QuietAndes(quiet):
        ok = bool(ss.TDS.run(no_summary=True))
    machines = [k for k in range(g.n) if int(g.bus.v[k]) in island and float(g.u.v[k]) > 0.5]
    return {
        "found": True, "branch_opened": str(line.idx.v[branch]), "island_buses": sorted(island),
        "machines": [str(g.idx.v[k]) for k in machines], "tds_ok": ok, "final_time_s": float(ss.dae.t),
        "islands_after": mon.n_islands, "island_machine_speed_pu_end": [float(g.omega.v[k]) for k in machines],
        "note": "machine rejects its load with protection disabled; the overspeed is physical (no overspeed relay modeled)",
    }


def _islanding_case(base: ReplicaConfig, *, t_open_s: float, tf_s: float, quiet: bool,
                    balance_mw: float) -> dict[str, Any]:
    actions: tuple[PreCascadeAction, ...] = ()
    if balance_mw > 0:
        actions = (PreCascadeAction("BAL", "interchange_reduction", 0.0, 0.0, float(balance_mw), "MW",
                                    "steady-state re-dispatch of the area export (islanding check only)", "check"),)
    cfg = replace(base, variant="islanding_check", tf_s=tf_s, pre_cascade_actions=actions,
                  pre_cascade_enabled=bool(actions), plant_ov_enabled=False,
                  generator_protection_enabled=False, ufls_stages=(), separation_enabled=False,
                  forced_separation_at_s=t_open_s)
    res = run_replica(cfg, out_dir=None, quiet=quiet, armed=bool(actions), stall_guard=True)
    mon = res.monitor
    rows = mon.system_rows
    after = [r for r in rows if r["time_s"] >= t_open_s + 0.5]
    f_aff = [r["affected_island_f_pu"] for r in after]
    f_rest = [r["rest_f_pu"] for r in after]
    out = {
        "t_open_s": t_open_s,
        "tf_s": tf_s,
        "tds_ok": res.tds_ok,
        "final_time_s": res.final_time_s,
        "islands_after_opening": mon.n_islands,
        "island_sizes": mon.island_sizes,
        "cut_set_size": len(mon.cut_pos),
        "affected_island_f_pu_range": [float(np.nanmin(f_aff)), float(np.nanmax(f_aff))] if f_aff else None,
        "rest_f_pu_range": [float(np.nanmin(f_rest)), float(np.nanmax(f_rest))] if f_rest else None,
        "frequencies_diverge": bool(after and abs(np.nanmean(f_aff) - np.nanmean(f_rest)) > 1e-4),
        "area_export_at_t0_mw": res.monitor.export0_mw,
        "cut_set_sum_abs_p_at_t0_mw": res.monitor.cut_abs_p0_mw,
        "min_bus_voltage_after_opening_pu": float(min((r["vmin_pu"] for r in after), default=float("nan"))),
        "balance_redispatch_mw": balance_mw,
        "rejected_step_calls": res.monitor.rejected_calls,
        "stalled": res.monitor.stalled_reason,
    }
    out["andes_handles_island"] = bool(res.tds_ok and out["islands_after_opening"] == 2 and not res.monitor.stalled_reason)
    return out


@dataclass(slots=True)
class ArmedReplica:
    rs: ReplicaSystem
    system: Any
    protected_blocks: list[Any]
    events: list[Any]
    metadata: dict[str, Any]


def build_armed_replica(cfg: ReplicaConfig | None = None, *, method: str = "powerflow", settle_s: float = 45.0,
                        quiet: bool = True) -> ArmedReplica:
    from .screening_library import protected_blocks_from_replica, replica_event_library

    cfg = cfg or ReplicaConfig()
    residual = 0.0
    extra_meta: dict[str, Any] = {}
    if method == "dynamic_equilibrium":
        rs, extra_meta = _dynamic_equilibrium_state(cfg, settle_s=settle_s, quiet=quiet)
        residual = float(extra_meta["polish"]["max_abs_f"])
    elif method == "powerflow":
        rs = build_replica_system(cfg, armed=True, quiet=quiet)
        residual = float(rs.base_notes.get("init_max_abs_dxdt", 0.0))
    elif method == "dynamic":
        dyn_cfg = replace(cfg, plant_ov_enabled=False, tf_s=float(settle_s))
        rs = build_replica_system(dyn_cfg, armed=False, quiet=quiet)
        mon = PhysicsMonitor(rs)
        ss = rs.system
        ss.TDS.callpert = mon
        with QuietAndes(quiet):
            ok = bool(ss.TDS.run(no_summary=True))
        ss.TDS.callpert = None
        if not ok:
            raise RuntimeError("dynamic arming run did not reach settle_s")
        if mon.events and any(e.category not in {"operator_action"} for e in mon.events):
            raise RuntimeError(f"protection operated during the dynamic arming run: {[e.code for e in mon.events]}")
        v = np.asarray(ss.Bus.v.v, dtype=float)
        for b in rs.blocks:
            b.z0_pu = float(v[_pos(ss.Bus, b.collector_bus)])
            b.hv_v0_pu = float(v[_pos(ss.Bus, b.hv_bus)])
        residual, _ = max_abs_dxdt(rs)
        rs.armed = True
    else:
        raise ValueError("method must be 'powerflow', 'dynamic' or 'dynamic_equilibrium'")
    blocks = protected_blocks_from_replica(rs)
    events = replica_event_library(rs)
    meta = {
        "armed": True,
        "method": method,
        "time_s": float(rs.system.dae.t),
        "max_abs_dxdt": residual,
        "gsu_tap": rs.tap,
        "max_collector_v_pu": max(b.z0_pu for b in rs.blocks),
        "min_margin_to_ov1_pu": min(1.10 - b.z0_pu for b in rs.blocks),
        "families": sorted({e.family for e in events}),
        "omitted_families": {"shunt_reactor_disconnection": "no shunt reactor exists in the affected area of ACTIVSg2000"},
        "case_design": case_design(rs),
        **extra_meta,
    }
    return ArmedReplica(rs=rs, system=rs.system, protected_blocks=blocks, events=events, metadata=meta)


def agc_governor_mask(ss: Any) -> np.ndarray:
    tg = ss.TGOV1
    links = _genrou_links(ss)
    gpos = {str(v): i for i, v in enumerate(ss.GENROU.idx.v)}
    ue = np.asarray(tg.ue.v, dtype=float) if hasattr(tg, "ue") else np.asarray(tg.u.v, dtype=float)
    mask = np.zeros(tg.n, dtype=bool)
    for k in range(tg.n):
        syn = str(tg.syn.v[k])
        gi = gpos.get(syn)
        if gi is None or ue[k] <= 0.5 or float(ss.GENROU.u.v[gi]) <= 0.5:
            continue
        exc_on = False
        for ref in links[syn]["exciter"]:
            name, idx = ref.split(":", 1)
            model = getattr(ss, name)
            exc_on = exc_on or float(model.u.v[_pos(model, idx)]) > 0.5
        mask[k] = exc_on
    return mask


def agc_reset(ss: Any) -> dict[str, Any]:
    tg = getattr(ss, "TGOV1", None)
    if tg is None or not tg.n:
        return {"governors": 0}
    agc = agc_governor_mask(ss)
    wd = np.asarray(tg.wd.v, dtype=float)
    r = np.asarray(tg.R.v, dtype=float)
    shift = np.where(agc, wd / r, 0.0)
    tg.pref0.v[:] = np.asarray(tg.pref0.v, dtype=float) - shift
    return {"governors": int(np.sum(agc)), "max_abs_pref_shift_pu": float(np.max(np.abs(shift))),
            "sum_pref_shift_mw": float(-np.sum(shift)) * float(ss.config.mva),
            "participants": "in-service TGOV1 of machines with an in-service exciter"}


_DEBUG_POLISH = False


def polish_equilibrium(ss: Any, *, h0: float = 0.1, h_max: float = 1e6, tol: float = 1e-10,
                       max_iter: int = 80, secondary_control: bool = True, inner_iters: int = 1) -> dict[str, Any]:
    from scipy import sparse
    from scipy.sparse.linalg import spsolve

    dae = ss.dae
    n, m = int(dae.n), int(dae.m)
    tf = np.asarray(dae.Tf, dtype=float).ravel()

    def csc(mat: Any, shape: tuple[int, int]) -> Any:
        return sparse.csc_matrix((np.asarray(mat.V).ravel(), (np.asarray(mat.I).ravel(), np.asarray(mat.J).ravel())),
                                 shape=shape)

    tg = getattr(ss, "TGOV1", None)
    border = bool(secondary_control and tg is not None and tg.n)
    if border:
        r = np.asarray(tg.R.v, dtype=float)
        on = agc_governor_mask(ss)
        s_gain = float(np.sum(1.0 / r[on]))
        col = sparse.csc_matrix((np.full(int(on.sum()), 1.0 / s_gain), (np.asarray(tg.pref.a)[on], np.zeros(int(on.sum()), dtype=int))),
                                shape=(m, 1))
        g_sl = next((i for i in range(ss.GENROU.n) if str(ss.GENROU.gen.v[i]) in {str(v) for v in ss.Slack.idx.v}
                     and float(ss.GENROU.u.v[i]) > 0.5), None)
        if g_sl is None:
            g_sl = int(np.argmax(np.where(np.asarray(ss.GENROU.u.v) > 0.5, np.asarray(ss.GENROU.Sn.v), -1)))
        w_addr = int(ss.GENROU.omega.a[g_sl])
        k_agc = s_gain / 10.0
    lam_total = 0.0

    frozen_x: list[int] = []
    frozen_y: list[int] = []
    for model in ss.exist.pflow_tds.values():
        if not getattr(model, "n", 0) or not hasattr(model, "u"):
            continue
        off = np.flatnonzero(np.asarray(model.u.v, dtype=float) <= 0.5)
        if not off.size:
            continue
        for var in model.states.values():
            frozen_x.extend(int(var.a[k]) for k in off)
        for var in model.algebs.values():
            frozen_y.extend(int(var.a[k]) for k in off)
    frozen_x_arr = np.asarray(sorted(set(frozen_x)), dtype=int)
    frozen_y_arr = np.asarray(sorted(set(frozen_y)), dtype=int)
    keep_f = np.ones(n, dtype=bool)
    keep_f[frozen_x_arr] = False
    keep_g = np.ones(m, dtype=bool)
    keep_g[frozen_y_arr] = False
    frozen_all = np.concatenate([frozen_x_arr, n + frozen_y_arr]).astype(int)

    def freeze_rows(mat: Any) -> Any:
        if not frozen_all.size:
            return mat
        mat = mat.tolil()
        for i in frozen_all:
            mat.rows[i] = [int(i)]
            mat.data[i] = [1.0]
        return mat.tocsc()

    def residual() -> tuple[np.ndarray, np.ndarray]:
        ss.TDS.fg_update(ss.exist.pflow_tds)
        f = np.asarray(dae.f, dtype=float).copy()
        for item in ss.antiwindups:
            for key, _, eqval in item.x_set:
                np.put(f, key, eqval)
        return f, np.asarray(dae.g, dtype=float).copy()

    def measure(f: np.ndarray, g: np.ndarray) -> float:
        agc = abs(float(dae.x[w_addr]) - 1.0) if border else 0.0
        return max(float(np.max(np.abs(f[keep_f]))) if keep_f.any() else 0.0,
                   float(np.max(np.abs(g[keep_g]))) if keep_g.any() else 0.0, agc)

    f, g = residual()
    res = measure(f, g)
    history = [res]
    h = float(h0)
    it = 0
    while it < max_iter and res > tol:
        ss.j_update(models=ss.exist.pflow_tds)
        blocks = [[sparse.diags(tf) - h * csc(dae.fx, (n, n)), -h * csc(dae.fy, (n, m))],
                  [csc(dae.gx, (m, n)), csc(dae.gy, (m, m))]]
        if border:
            top = sparse.hstack([sparse.bmat(blocks), sparse.vstack([sparse.csc_matrix((n, 1)), col])])
            row = sparse.csc_matrix(([h * k_agc, 1.0], ([0, 0], [w_addr, n + m])), shape=(1, n + m + 1))
            a = sparse.vstack([top, row]).tocsc()
            rhs = np.concatenate([h * f, -g, [-h * k_agc * (float(dae.x[w_addr]) - 1.0)]])
        else:
            a = sparse.bmat(blocks, format="csc")
            rhs = np.concatenate([h * f, -g])
        a = freeze_rows(a)
        rhs[frozen_all] = 0.0
        x_old = np.asarray(dae.x, dtype=float).copy()
        y_old = np.asarray(dae.y, dtype=float).copy()
        pref_old = np.asarray(tg.pref0.v, dtype=float).copy() if border else None
        lam_step = 0.0
        for inner in range(inner_iters):
            d = spsolve(a, rhs)
            dae.x[:] = np.asarray(dae.x, dtype=float) + d[:n]
            dae.y[:] = np.asarray(dae.y, dtype=float) + d[n:n + m]
            if border:
                dl = float(d[n + m])
                lam_step += dl
                tg.pref0.v[:] = np.asarray(tg.pref0.v, dtype=float) + np.where(on, dl / (s_gain * r), 0.0)
            ss.vars_to_models()
            if inner == inner_iters - 1:
                break
            f_i, g_i = residual()
            qx = tf * (np.asarray(dae.x, dtype=float) - x_old) - h * f_i
            qx[frozen_x_arr] = 0.0
            g_i = g_i.copy()
            g_i[frozen_y_arr] = 0.0
            ql = [lam_step + h * k_agc * (float(dae.x[w_addr]) - 1.0)] if border else []
            if max(float(np.max(np.abs(qx))), float(np.max(np.abs(g_i)))) < 1e-10 * max(1.0, h):
                break
            ss.j_update(models=ss.exist.pflow_tds)
            blocks = [[sparse.diags(tf) - h * csc(dae.fx, (n, n)), -h * csc(dae.fy, (n, m))],
                      [csc(dae.gx, (m, n)), csc(dae.gy, (m, m))]]
            if border:
                top = sparse.hstack([sparse.bmat(blocks), sparse.vstack([sparse.csc_matrix((n, 1)), col])])
                row = sparse.csc_matrix(([h * k_agc, 1.0], ([0, 0], [w_addr, n + m])), shape=(1, n + m + 1))
                a = freeze_rows(sparse.vstack([top, row]).tocsc())
                rhs = np.concatenate([-qx, -g_i, [-ql[0]]])
            else:
                a = freeze_rows(sparse.bmat(blocks, format="csc"))
                rhs = np.concatenate([-qx, -g_i])
            rhs[frozen_all] = 0.0
        dlam = lam_step
        f_new, g_new = residual()
        res_new = measure(f_new, g_new)
        if _DEBUG_POLISH:
            jf = int(np.argmax(np.abs(np.where(keep_f, f_new, 0.0))))
            print(f"polish it={it} h={h:.3g} res={res:.3e} -> {res_new:.3e} top_f={dae.x_name[jf]} {f_new[jf]:.3e} dmax={np.max(np.abs(d)):.3e}")
        if not np.isfinite(res_new) or res_new > 10.0 * res + 1e-6:
            dae.x[:] = x_old
            dae.y[:] = y_old
            if border:
                tg.pref0.v[:] = pref_old
            ss.vars_to_models()
            f, g = residual()
            h = max(h / 10.0, 1e-4)
        else:
            if border:
                lam_total += dlam
            f, g, res = f_new, g_new, res_new
            h = min(h * 4.0, h_max)
            history.append(res)
        it += 1
    return {"iterations": it, "max_abs_f": float(np.max(np.abs(f[keep_f]))) if keep_f.any() else 0.0,
            "max_abs_g": float(np.max(np.abs(g[keep_g]))) if keep_g.any() else 0.0, "residual_history": history[-12:],
            "final_h": h, "converged": bool(res <= tol), "frozen_variables": int(frozen_all.size),
            "note": "residuals over variables of in-service devices (out-of-service devices are decoupled)",
            "secondary_control_mw": lam_total * float(ss.config.mva) if border else None}


def _dynamic_equilibrium_state(cfg: ReplicaConfig, *, settle_s: float, quiet: bool,
                               ) -> tuple[ReplicaSystem, dict[str, Any]]:
    dyn_cfg = replace(cfg, plant_ov_enabled=False, tf_s=float(settle_s))
    rs = build_replica_system(dyn_cfg, armed=False, quiet=quiet)
    mon = PhysicsMonitor(rs)
    ss = rs.system
    ss.TDS.callpert = mon
    with QuietAndes(quiet):
        ok = bool(ss.TDS.run(no_summary=True))
    ss.TDS.callpert = None
    if not ok:
        raise RuntimeError("dynamic arming run did not reach settle_s")
    other = [e.code for e in mon.events if e.category != "operator_action"]
    if other:
        raise RuntimeError(f"protection operated during the dynamic arming run: {other}")
    v = np.asarray(ss.Bus.v.v, dtype=float)
    plateau = {b.spec.name: float(v[_pos(ss.Bus, b.collector_bus)]) for b in rs.blocks}
    omega = np.asarray(ss.GENROU.omega.v, dtype=float)[np.asarray(ss.GENROU.u.v, dtype=float) > 0.5]
    dxdt_plateau, dxdt_plateau_state = max_abs_dxdt(rs)
    agc = agc_reset(ss)
    cont = polish_equilibrium(ss, h0=1.0, h_max=20.0, tol=1e-7, max_iter=250, inner_iters=10)
    polish = polish_equilibrium(ss, h0=100.0, h_max=1e6, tol=1e-10, max_iter=60, inner_iters=1)
    if not polish["converged"]:
        raise RuntimeError(f"equilibrium polish did not converge: {polish}")
    v = np.asarray(ss.Bus.v.v, dtype=float)
    for b in rs.blocks:
        b.z0_pu = float(v[_pos(ss.Bus, b.collector_bus)])
        b.hv_v0_pu = float(v[_pos(ss.Bus, b.hv_bus)])
    omega_eq = np.asarray(ss.GENROU.omega.v, dtype=float)[np.asarray(ss.GENROU.u.v, dtype=float) > 0.5]
    dv = {b.spec.name: b.z0_pu - plateau[b.spec.name] for b in rs.blocks}
    rs.armed = True
    meta = {
        "settle_s": float(settle_s),
        "plateau_collector_v_pu": plateau,
        "plateau_max_abs_dxdt": dxdt_plateau,
        "plateau_max_abs_dxdt_state": dxdt_plateau_state,
        "continuation": cont,
        "plateau_omega_range_pu": [float(omega.min()), float(omega.max())],
        "agc_reset": agc,
        "polish": polish,
        "equilibrium_omega_range_pu": [float(omega_eq.min()), float(omega_eq.max())],
        "plateau_max_abs_dv_pu": float(max(abs(x) for x in dv.values())),
        "plateau_dv_pu": dv,
    }
    return rs, meta


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--variant", default="baseline", choices=sorted(VARIANTS))
    p.add_argument("--out", type=Path, default=None, help="output directory (default results/revised/activsg2000/<variant>)")
    p.add_argument("--tf", type=float, default=None)
    p.add_argument("--tstep", type=float, default=None)
    p.add_argument("--verbose-andes", action="store_true")
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    overrides: dict[str, Any] = {}
    if args.tf is not None:
        overrides["tf_s"] = float(args.tf)
    if args.tstep is not None:
        overrides["tstep_s"] = float(args.tstep)
    cfg = variant_config(args.variant, **overrides)
    out = args.out or (DEFAULT_OUT_ROOT / args.variant)
    res = run_replica(cfg, out_dir=out, quiet=not args.verbose_andes)
    print(f"[{cfg.variant}] {res.summary['interpretation']}")
    print(f"outputs: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
