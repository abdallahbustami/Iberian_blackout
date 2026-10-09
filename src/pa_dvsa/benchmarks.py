"""The four benchmark systems (Kundur, IEEE 39, NPCC, GBnetwork) with protected renewable plants, their event
library and their controls, built deterministically from the ANDES cases.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, fields
import copy
import io
import json
import logging
import math
from pathlib import Path
import sys
import time
from typing import Any, Callable, Iterator, Mapping, Sequence
import warnings

import numpy as np
from scipy import sparse
from scipy.sparse import csgraph
from scipy.sparse import linalg as spla

from .andes_regularization import regularize_degenerate_blocks
from .data_model import ProfileKind, TimeProfile
from .events import ControlSpec, EventSpec, Mutation, ProtectedBlock, generator_trip_mutations, lag_envelope

__all__ = [
    "BENCHMARK_SLUGS",
    "VARIANTS",
    "EVENT_FAMILIES",
    "CONTROL_CLASSES",
    "SystemSpec",
    "SYSTEMS",
    "HiddenOutputData",
    "BenchmarkCase",
    "BenchmarkError",
    "ScenarioInfeasibleError",
    "build_benchmark",
    "write_design",
    "device_counts",
    "static_validation",
    "controllers_at_limit",
    "augmented_algebraic_matrix",
    "Scenario",
    "SCENARIOS",
    "scenario_of",
]


BENCHMARK_SLUGS = ("kundur", "ieee39", "npcc", "gbnetwork")
VARIANTS = ("baseline", "voltage_mode_res", "preserved_absorption", "delayed_protection", "no_ov_protection",
            "stock_load")
EVENT_FAMILIES = (
    "plant_trip",
    "fixed_pf_ramp",
    "load_disconnection",
    "generator_trip",
    "generation_redispatch",
    "shunt_reactor_disconnection",
    "capacitor_switching",
    "line_energization",
)
CONTROL_CLASSES = ("statcom", "plant_q_command", "avr_vref", "fast_reactor", "manual_reactor")

RES_SHARE_OF_LOAD = 0.50
BLOCK_P_SPREAD = 0.30
PLANT_PF = 0.95
KAPPA_Q_OVER_P = -math.tan(math.acos(PLANT_PF))
PLANT_SN_OVER_P = 1.15
GSU_X_DEVICE_PU = 0.12
GSU_X_OVER_R = 40.0
COLLECTOR_KV = 34.5
Z0_RANGE_PU = (1.060, 1.092)
TAP_RANGE = (0.90, 1.10)
Q_DESIGN_REL_TOL = 1.0e-2
RELAY_PICKUP_PU = 1.10
RELAY_DWELL_S = 0.05
RELAY_DWELL_DELAYED_S = 0.20
HV_MIN_KV = 200.0
SCR_MIN = 3.0
PLANT_SPACING_PU = 0.005
STRONG_QUANTILE = 0.5
MAX_PLACEMENT_REPAIRS = 12
N_CLUSTERS = {"kundur": 2, "ieee39": 3, "npcc": 4, "gbnetwork": 4}
MUST_RUN_FRACTION = 0.20
PMAX_ABSENT_PU = 99.0
GOV_MIN_MARGIN_PU = 1.0e-3
UEL_SYSTEMS = ("gbnetwork",)
UEL_Q_OVER_SN = -0.30
CONDENSER_P_MAX_MW = 1.0
UEL_Q_TOL_PU = 1.0e-6
RATING_CORRECTION_SYSTEMS = ("gbnetwork",)
RATING_CORRECTION_PF = 0.9
UEL_NONBINDING_PU = 1.0e6


@dataclass(frozen=True, slots=True)
class Scenario:
    name: str
    load_scale: float
    res_share: float
    tap_range: tuple[float, float]
    dispatch_rule: str
    must_run_fraction: float | None
    description: str


SCENARIOS = {
    "light_load": Scenario(
        "light_load", 0.60, 0.50, (0.90, 1.10), "proportional_must_run", MUST_RUN_FRACTION,
        "light load (every PQ load at 60 % of the stock P and Q), RES = 50 % of the scaled load, all synchronous "
        "units online with one common dispatch factor and a must-run minimum, stock voltage set points, taps "
        "within +/-10 %"),
    "stock_load": Scenario(
        "stock_load", 1.00, 0.35, (0.85, 1.15), "pv_only_slack_keeps_base", None,
        "previous design kept for reference: stock load, RES = 35 % of the load, PV set points scaled by one "
        "factor (slack keeps its base output), taps within [0.85, 1.15]"),
}


def scenario_of(variant: str) -> Scenario:
    return SCENARIOS["stock_load" if variant == "stock_load" else "light_load"]

PVD1_IALIM = 1.3
PVD1_QLIM_DEVICE = 0.6
PVD1_FIXED_PF = {"dqdv": -1.0, "v0": 0.01, "v1": 5.0}
PVD1_TRIP_DISABLED = {
    "vt0": 0.005, "vt1": 0.01, "vt2": 5.0, "vt3": 6.0,
    "ft0": 0.05, "ft1": 0.1, "ft2": 999.0, "ft3": 1000.0,
    "recflag": 0,
}
VOLTAGE_MODE_BREAKPOINT_OFFSET_PU = 1.0e-4
VOLTAGE_MODE_SAT_DV_PU = 0.05

FIXED_PF_RAMP_FRACTION = 0.70
FIXED_PF_RAMP_DURATION_S = 0.5
N_GENERATOR_TRIPS = 6
N_REDISPATCH_PAIRS = 3
REDISPATCH_FRACTION = 0.20
REDISPATCH_RAMP_S = 1.0
REACTOR_MVAR_RANGE = (50.0, 150.0)
CAPACITOR_MVAR_RANGE = (40.0, 100.0)
N_CAPACITORS = 2
N_PARALLEL_LINES = 2

STATCOM_DELAY_S, STATCOM_TAU_S = 0.020, 0.030
PLANT_Q_DELAY_S = 0.100
PLANT_Q_CAPABILITY_PF = 0.90
AVR_DELAY_S, AVR_DVREF_PU = 0.050, -0.03
N_AVR_CONTROLS = 6
FAST_REACTOR_MVAR, FAST_REACTOR_DELAY_S = 50.0, 0.150
MANUAL_REACTOR_MVAR, MANUAL_REACTOR_DELAY_S = 100.0, 30.0
ENVELOPE_HORIZON_S = 2.0

TDS_DEFAULTS = {"tf": 2.0, "tstep": 0.005, "no_tqdm": 1, "criteria": 0}

_GOLDEN = (math.sqrt(5.0) - 1.0) / 2.0
_SILVER = math.sqrt(2.0) - 1.0
_REPO_ROOT = Path(__file__).resolve().parents[2]


class BenchmarkError(RuntimeError):
    pass


class ScenarioInfeasibleError(BenchmarkError):
    def __init__(self, message: str, details: Mapping[str, Any]):
        super().__init__(message)
        self.details = dict(details)


@dataclass(frozen=True, slots=True)
class SystemSpec:
    slug: str
    label: str
    case: str
    addfile: str | None
    n_blocks: int
    n_clusters: int
    statcom_mvar: float


SYSTEMS: dict[str, SystemSpec] = {
    "kundur": SystemSpec("kundur", "Kundur two-area", "kundur/kundur_full.xlsx", None, 4, N_CLUSTERS["kundur"], 50.0),
    "ieee39": SystemSpec("ieee39", "IEEE 39-bus", "ieee39/ieee39_full.xlsx", None, 12, N_CLUSTERS["ieee39"], 100.0),
    "npcc": SystemSpec("npcc", "NPCC 140-bus", "npcc/npcc.raw", "npcc/npcc_full.dyr", 16, N_CLUSTERS["npcc"], 100.0),
    "gbnetwork": SystemSpec("gbnetwork", "GB network", "GBnetwork/GBnetwork.xlsx", None, 20, N_CLUSTERS["gbnetwork"], 100.0),
}


@dataclass(frozen=True, slots=True)
class HiddenOutputData:
    asset_id: str
    collector_bus: Any
    hv_bus: Any
    collector_v_address: int
    hv_v_address: int
    hv_a_address: int
    tap: float
    gsu_r_pu_sysbase: float
    gsu_x_pu_sysbase: float
    gsu_z_abs_pu_sysbase: float
    v_hv_op_pu: float
    z0_pu: float
    drop_op_pu: float
    drop_bound_op_pu: float
    current_limit_pu_sysbase: float
    drop_bound_ilim_pu: float


@dataclass(slots=True)
class BenchmarkCase:
    slug: str
    variant: str
    system: Any
    factory: Callable[..., Any]
    blocks: tuple[ProtectedBlock, ...]
    events: tuple[EventSpec, ...]
    controls: tuple[ControlSpec, ...]
    design: dict[str, Any]
    protected_output_addresses: tuple[int, ...]
    hidden_output_data: tuple[HiddenOutputData, ...]
    build_runtime_s: float = float("nan")

    def event(self, event_id: str) -> EventSpec:
        for item in self.events:
            if item.event_id == event_id:
                return item
        raise KeyError(event_id)

    def events_by_family(self, family: str) -> tuple[EventSpec, ...]:
        return tuple(item for item in self.events if item.family == family)

    def control(self, control_id: str) -> ControlSpec:
        for item in self.controls:
            if item.control_id == control_id:
                return item
        raise KeyError(control_id)

    @property
    def relays_enabled(self) -> bool:
        return not bool(self.design["variant"]["relays_disabled"])


@contextmanager
def _quiet() -> Iterator[None]:
    logger = logging.getLogger("andes")
    old_level = logger.level
    logger.setLevel(logging.ERROR)
    old_out, old_err = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = io.StringIO(), io.StringIO()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            yield
    finally:
        sys.stdout, sys.stderr = old_out, old_err
        logger.setLevel(old_level)


def _pos(model: Any, idx: Any) -> int:
    values = list(model.idx.v)
    if idx in values:
        return values.index(idx)
    as_str = [str(v) for v in values]
    if str(idx) in as_str:
        return as_str.index(str(idx))
    raise BenchmarkError(f"{model.class_name} has no device {idx!r}")


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        v = float(value)
        return v if math.isfinite(v) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if value is None or isinstance(value, (str, int)):
        return value
    return str(value)


def _to_csc(matrix: Any) -> sparse.csc_matrix:
    if sparse.issparse(matrix):
        return matrix.tocsc()
    size = getattr(matrix, "size", None)
    if size is not None and hasattr(matrix, "I") and hasattr(matrix, "J") and hasattr(matrix, "V"):
        rows, cols = int(size[0]), int(size[1])
        data = np.asarray(matrix.V, dtype=float).reshape(-1)
        row = np.asarray(matrix.I, dtype=int).reshape(-1)
        col = np.asarray(matrix.J, dtype=int).reshape(-1)
        return sparse.coo_matrix((data, (row, col)), shape=(rows, cols)).tocsc()
    return sparse.csc_matrix(np.asarray(matrix, dtype=float))


def _low_discrepancy_ranks(n: int, alpha: float) -> list[int]:
    keys = [((k + 1) * alpha) % 1.0 for k in range(n)]
    order = sorted(range(n), key=lambda k: (keys[k], k))
    ranks = [0] * n
    for r, k in enumerate(order):
        ranks[k] = r
    return ranks


def _midpoint_sizes(n: int, lo: float, hi: float) -> list[float]:
    return [lo + (hi - lo) * (j + 0.5) / n for j in range(n)]


def _load_raw(spec: SystemSpec, *, pv2pq: bool = False) -> Any:
    import andes

    kwargs: dict[str, Any] = {"setup": False, "no_output": True, "default_config": True}
    if pv2pq:
        kwargs["config_option"] = ["PV.pv2pq=1"]
    if spec.addfile:
        kwargs["addfile"] = andes.get_case(spec.addfile)
    with _quiet():
        ss = andes.load(andes.get_case(spec.case), **kwargs)
    if ss is None:
        raise BenchmarkError(f"ANDES failed to load {spec.case}")
    return ss


def _correct_ratings(ss: Any, slug: str | None) -> list[dict[str, Any]]:
    if slug not in RATING_CORRECTION_SYSTEMS:
        return []
    sb = float(ss.config.mva)
    pmax: dict[str, float] = {}
    p0: dict[str, float] = {}
    for mname in ("PV", "Slack"):
        m = getattr(ss, mname, None)
        if m is None:
            continue
        for i in range(m.n):
            pmax[str(m.idx.v[i])] = float(m.pmax.v[i])
            p0[str(m.idx.v[i])] = float(m.p0.v[i])
    changes: list[dict[str, Any]] = []
    for mname in ("GENCLS", "GENROU"):
        model = getattr(ss, mname, None)
        if model is None or model.n == 0:
            continue
        for i in range(model.n):
            gen = str(model.gen.v[i])
            pm = pmax.get(gen)
            if pm is None or pm >= PMAX_ABSENT_PU:
                continue
            share = float(model.gammap.v[i]) if hasattr(model, "gammap") else 1.0
            pmax_mw = share * pm * sb
            sn_old = float(model.Sn.v[i])
            if sn_old < pmax_mw - 1e-9:
                sn_new = pmax_mw / RATING_CORRECTION_PF
                model.Sn.v[i] = sn_new
                changes.append({"model": mname, "idx": model.idx.v[i], "gen": model.gen.v[i], "sn_old_mva": sn_old,
                                "sn_new_mva": sn_new, "pmax_mw": pmax_mw, "stock_p_mw": share * p0.get(gen, 0.0) * sb,
                                "condenser": abs(share * p0.get(gen, 0.0) * sb) < CONDENSER_P_MAX_MW})
    return changes


def _prepare_raw(ss: Any, load_scale: float = 1.0, slug: str | None = None) -> dict[str, Any]:
    changes = regularize_degenerate_blocks(ss)
    ratings = _correct_ratings(ss, slug)
    if load_scale != 1.0:
        for i in range(ss.PQ.n):
            ss.PQ.p0.v[i] = float(ss.PQ.p0.v[i]) * load_scale
            ss.PQ.q0.v[i] = float(ss.PQ.q0.v[i]) * load_scale
    disabled: list[dict[str, Any]] = []
    for name in ("Toggle", "Fault", "Alter"):
        model = getattr(ss, name, None)
        if model is None or model.n == 0:
            continue
        for i in range(model.n):
            if float(model.u.v[i]) != 0.0:
                disabled.append({"model": name, "idx": str(model.idx.v[i])})
            model.u.v[i] = 0
    return {"regularization": changes, "disabled_timed_events": disabled, "load_scale": load_scale,
            "rating_corrections": ratings}


def _is_transformer(line: Any, i: int) -> bool:
    vin = lambda name: float(getattr(line, name).v[i])
    return bool(
        float(line.trans.v[i]) != 0.0
        or abs(vin("Vn1") - vin("Vn2")) > 1e-9
        or abs(vin("tap") - 1.0) > 1e-12
        or abs(vin("phi")) > 1e-12
    )


def _machine_inventory(ss: Any) -> dict[str, Any]:
    machines: list[dict[str, Any]] = []
    for mname, model in ss.groups["SynGen"].models.items():
        if model.n == 0:
            continue
        for i in range(model.n):
            machines.append(
                {
                    "model": mname,
                    "idx": model.idx.v[i],
                    "bus": model.bus.v[i],
                    "gen": model.gen.v[i],
                    "sn_mva": float(model.Sn.v[i]),
                    "u": float(model.u.v[i]),
                    "gammap": float(model.gammap.v[i]) if hasattr(model, "gammap") else 1.0,
                    "exciters": [],
                    "governors": [],
                    "pss": [],
                }
            )
    by_idx = {str(m["idx"]): m for m in machines}
    exc_owner: dict[str, dict[str, Any]] = {}
    for ename, model in ss.groups["Exciter"].models.items():
        for i in range(model.n):
            owner = by_idx.get(str(model.syn.v[i]))
            if owner is not None:
                owner["exciters"].append({"model": ename, "idx": model.idx.v[i], "u": float(model.u.v[i])})
                exc_owner[str(model.idx.v[i])] = owner
    for gname, model in ss.groups["TurbineGov"].models.items():
        for i in range(model.n):
            owner = by_idx.get(str(model.syn.v[i]))
            if owner is not None:
                vmin = float(model.VMIN.v[i]) if hasattr(model, "VMIN") else None
                owner["governors"].append({"model": gname, "idx": model.idx.v[i], "u": float(model.u.v[i]),
                                           "vmin_sys_pu": vmin})
    for pname, model in ss.groups["PSS"].models.items():
        for i in range(model.n):
            owner = exc_owner.get(str(model.avr.v[i]))
            if owner is not None:
                owner["pss"].append({"model": pname, "idx": model.idx.v[i], "u": float(model.u.v[i])})
    return {"machines": machines}


def _static_gen_p(ss: Any) -> dict[str, float]:
    out: dict[str, float] = {}
    for i in range(ss.PV.n):
        out[str(ss.PV.idx.v[i])] = float(ss.PV.p0.v[i]) * float(ss.PV.u.v[i])
    for i in range(ss.Slack.n):
        out[str(ss.Slack.idx.v[i])] = float(ss.Slack.p.v[i]) * float(ss.Slack.u.v[i])
    return out


def _base_analysis(spec: SystemSpec) -> dict[str, Any]:
    ss = _load_raw(spec)
    prep = _prepare_raw(ss, 1.0, spec.slug)
    with _quiet():
        ss.setup()
        converged = bool(ss.PFlow.run())
    if not converged:
        raise BenchmarkError(f"{spec.slug}: base-case power flow did not converge")
    bus = ss.Bus
    buses = []
    for i in range(bus.n):
        buses.append(
            {
                "idx": bus.idx.v[i],
                "vn_kv": float(bus.Vn.v[i]),
                "area": bus.area.v[i],
                "zone": bus.zone.v[i] if hasattr(bus, "zone") else None,
                "u": float(bus.u.v[i]),
                "v": float(bus.v.v[i]),
                "a": float(bus.a.v[i]),
            }
        )
    line = ss.Line
    lines = []
    for i in range(line.n):
        lines.append(
            {
                "idx": line.idx.v[i],
                "bus1": line.bus1.v[i],
                "bus2": line.bus2.v[i],
                "u": float(line.u.v[i]),
                "x_sys": float(line.x.v[i]),
                "b_sys": float(line.b.v[i]),
                "is_transformer": _is_transformer(line, i),
                "input": {
                    name: _jsonable(getattr(line, name).vin[i] if getattr(line, name).vin is not None else getattr(line, name).v[i])
                    for name in ("Sn", "fn", "Vn1", "Vn2", "r", "x", "b", "g", "b1", "g1", "b2", "g2", "trans", "tap", "phi")
                },
            }
        )
    loads = []
    for i in range(ss.PQ.n):
        loads.append(
            {
                "idx": ss.PQ.idx.v[i],
                "bus": ss.PQ.bus.v[i],
                "p0": float(ss.PQ.p0.v[i]),
                "q0": float(ss.PQ.q0.v[i]),
                "u": float(ss.PQ.u.v[i]),
            }
        )
    static_gen_buses = {str(ss.PV.bus.v[i]) for i in range(ss.PV.n)} | {str(ss.Slack.bus.v[i]) for i in range(ss.Slack.n)}
    inv = _machine_inventory(ss)
    gen_p = _static_gen_p(ss)
    for m in inv["machines"]:
        m["p_base_pu"] = m["gammap"] * gen_p.get(str(m["gen"]), 0.0)
    syn_buses = {str(m["bus"]) for m in inv["machines"]}
    slack = {
        "idx": ss.Slack.idx.v[0],
        "bus": ss.Slack.bus.v[0],
        "u": float(ss.Slack.u.v[0]),
        "p_pu": float(ss.Slack.p.v[0]),
        "q_pu": float(ss.Slack.q.v[0]),
        "pmax": float(ss.Slack.pmax.v[0]),
        "pmin": float(ss.Slack.pmin.v[0]),
        "qmax": float(ss.Slack.qmax.v[0]),
        "qmin": float(ss.Slack.qmin.v[0]),
    }
    pv = [
        {"idx": ss.PV.idx.v[i], "bus": ss.PV.bus.v[i], "p0": float(ss.PV.p0.v[i]), "u": float(ss.PV.u.v[i]),
         "pmax": float(ss.PV.pmax.v[i]), "v0": float(ss.PV.v0.v[i])}
        for i in range(ss.PV.n)
    ]
    shunts = [
        {"idx": ss.Shunt.idx.v[i], "bus": ss.Shunt.bus.v[i], "b_sys": float(ss.Shunt.b.v[i]), "u": float(ss.Shunt.u.v[i])}
        for i in range(ss.Shunt.n)
    ]
    return {
        "prep": prep,
        "x_thevenin": _short_circuit_reactance(ss, [b["idx"] for b in buses if b["vn_kv"] >= HV_MIN_KV]),
        "mva": float(ss.config.mva),
        "freq": float(ss.config.freq),
        "buses": buses,
        "lines": lines,
        "loads": loads,
        "pv": pv,
        "slack": slack,
        "shunts": shunts,
        "gen_buses": sorted(static_gen_buses | syn_buses),
        "machines": inv["machines"],
        "device_counts": device_counts(ss),
        "v_range": [float(np.min(bus.v.v)), float(np.max(bus.v.v))],
    }


def _short_circuit_reactance(ss: Any, buses: Sequence[Any]) -> dict[str, float]:
    pos = {str(b): i for i, b in enumerate(ss.Bus.idx.v)}
    n = ss.Bus.n
    rows, cols, vals = [], [], []
    line = ss.Line
    for i in range(line.n):
        if float(line.u.v[i]) == 0.0:
            continue
        a, b = pos[str(line.bus1.v[i])], pos[str(line.bus2.v[i])]
        if a == b:
            continue
        w = 1.0 / max(abs(float(line.x.v[i])), 1.0e-4)
        rows += [a, b, a, b]
        cols += [b, a, a, b]
        vals += [-w, -w, w, w]
    diag = np.zeros(n)
    for mname, model in ss.groups["SynGen"].models.items():
        for i in range(model.n):
            if float(model.u.v[i]) == 0.0:
                continue
            x = float(model.xd2.v[i]) if hasattr(model, "xd2") else float(model.xd1.v[i])
            diag[pos[str(model.bus.v[i])]] += 1.0 / max(abs(x), 1.0e-4)
    ybus = (sparse.coo_matrix((vals, (rows, cols)), shape=(n, n)) + sparse.diags(diag)).tocsc()
    adj = (ybus != 0).astype(float)
    _, labels = csgraph.connected_components(adj, directed=False)
    grounded = {labels[i] for i in np.flatnonzero(diag > 0)}
    out: dict[str, float] = {}
    keep = np.flatnonzero(np.isin(labels, list(grounded)))
    kpos = {int(p): k for k, p in enumerate(keep)}
    lu = spla.splu(ybus[keep][:, keep].tocsc())
    for b in buses:
        p = pos[str(b)]
        if p not in kpos:
            out[str(b)] = float("inf")
            continue
        rhs = np.zeros(len(keep))
        rhs[kpos[p]] = 1.0
        out[str(b)] = float(lu.solve(rhs)[kpos[p]])
    return out


class _ElectricalDistance:
    def __init__(self, base: Mapping[str, Any]) -> None:
        self.bus_ids = [str(b["idx"]) for b in base["buses"]]
        self.pos = {b: i for i, b in enumerate(self.bus_ids)}
        n = len(self.bus_ids)
        rows, cols, vals = [], [], []
        for ln in base["lines"]:
            if ln["u"] == 0.0:
                continue
            i, j = self.pos[str(ln["bus1"])], self.pos[str(ln["bus2"])]
            if i == j:
                continue
            w = 1.0 / max(abs(ln["x_sys"]), 1.0e-4)
            rows += [i, j, i, j]
            cols += [j, i, i, j]
            vals += [-w, -w, w, w]
        lap = sparse.coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsc()
        self.laplacian = lap
        adj = (lap != 0).astype(float)
        _, labels = csgraph.connected_components(adj, directed=False)
        self.ref = self.pos[str(base["slack"]["bus"])]
        self.in_component = labels == labels[self.ref]
        keep = np.flatnonzero(self.in_component & (np.arange(n) != self.ref))
        self.keep_pos = {int(p): k for k, p in enumerate(keep)}
        self.lu = spla.splu(lap[keep][:, keep].tocsc())
        self._cache: dict[int, np.ndarray] = {}
        self.n_keep = len(keep)

    def _column(self, p: int) -> np.ndarray:
        if p not in self._cache:
            col = np.zeros(self.n_keep)
            if p != self.ref:
                rhs = np.zeros(self.n_keep)
                rhs[self.keep_pos[p]] = 1.0
                col = self.lu.solve(rhs)
            self._cache[p] = col
        return self._cache[p]

    def _z(self, p: int, q: int) -> float:
        if p == self.ref or q == self.ref:
            return 0.0
        return float(self._column(p)[self.keep_pos[q]])

    def to_ref(self, bus: Any) -> float:
        p = self.pos[str(bus)]
        return self._z(p, p)

    def between(self, a: Any, b: Any) -> float:
        p, q = self.pos[str(a)], self.pos[str(b)]
        return self._z(p, p) + self._z(q, q) - 2.0 * self._z(p, q)


def _spectral_halves(dist: _ElectricalDistance) -> dict[str, str]:
    comp = np.flatnonzero(dist.in_component)
    lap = dist.laplacian[comp][:, comp].tocsc()
    shift = 1e-6 * float(np.max(np.abs(lap.diagonal())))
    vals, vecs = spla.eigsh(lap, k=2, sigma=-shift, which="LM")
    fiedler = vecs[:, int(np.argsort(vals)[1])]
    ref = int(np.flatnonzero(comp == dist.ref)[0])
    if fiedler[ref] > 0:
        fiedler = -fiedler
    med = float(np.median(fiedler))
    return {dist.bus_ids[int(p)]: ("A" if fiedler[k] <= med else "B") for k, p in enumerate(comp)}


def _candidate_buses(base: Mapping[str, Any], dist: _ElectricalDistance, p_max_pu: float) -> tuple[list[Any], list[dict]]:
    gen_buses = set(base["gen_buses"])
    out, excluded = [], []
    for b in base["buses"]:
        key = str(b["idx"])
        if b["u"] == 0.0 or b["vn_kv"] < HV_MIN_KV or key in gen_buses:
            continue
        if not dist.in_component[dist.pos[key]]:
            excluded.append({"bus": b["idx"], "reason": "not in the slack island"})
            continue
        xth = base["x_thevenin"].get(key, float("inf"))
        scr = (1.0 / xth) / p_max_pu if xth > 0 else float("inf")
        if scr < SCR_MIN:
            excluded.append({"bus": b["idx"], "reason": f"SCR {scr:.2f} < {SCR_MIN:g}", "s_sc_mva": base["mva"] / xth})
            continue
        out.append(b["idx"])
    return out, excluded


def _place_blocks(spec: SystemSpec, candidates: Sequence[Any], dist: _ElectricalDistance,
                  x_thevenin: Mapping[str, float]) -> list[dict[str, Any]]:
    n, c = spec.n_blocks, spec.n_clusters
    sizes = [n // c + (1 if k < n % c else 0) for k in range(c)]
    if len(candidates) < n:
        raise BenchmarkError(f"{spec.slug}: only {len(candidates)} candidate HV buses for {n} blocks")
    order = {str(b): k for k, b in enumerate(candidates)}
    ssc = {str(b): 1.0 / x_thevenin[str(b)] for b in candidates}
    s_cut = float(np.quantile(list(ssc.values()), STRONG_QUANTILE))
    strong = [b for b in candidates if ssc[str(b)] >= s_cut]
    hubs = [max(strong, key=lambda b: (round(ssc[str(b)], 9), -order[str(b)]))]
    while len(hubs) < c:
        rest = [b for b in strong if b not in hubs]
        hubs.append(max(rest, key=lambda b: (round(min(dist.between(b, h) for h in hubs), 12), -order[str(b)])))
    members: list[list[Any]] = [[h] for h in hubs]
    placed: list[Any] = list(hubs)

    def _next(k: int) -> Any | None:
        for pool in (strong, candidates):
            ranked = sorted((b for b in pool if b not in placed),
                            key=lambda b: (round(dist.between(hubs[k], b), 12), order[str(b)]))
            for b in ranked:
                if all(dist.between(b, q) >= PLANT_SPACING_PU for q in placed):
                    return b
        return None

    while any(len(m) < sizes[k] for k, m in enumerate(members)):
        progressed = False
        for k in range(c):
            if len(members[k]) >= sizes[k]:
                continue
            b = _next(k)
            if b is None:
                continue
            members[k].append(b)
            placed.append(b)
            progressed = True
        if not progressed:
            raise BenchmarkError(f"{spec.slug}: not enough spaced candidate buses for {n} blocks")
    placement = []
    for k, mem in enumerate(members):
        for j, b in enumerate(mem):
            placement.append(
                {
                    "cluster": k,
                    "rank_in_cluster": j,
                    "hv_bus": b,
                    "seed_bus": hubs[k],
                    "in_strong_set": b in strong,
                    "d_to_seed_pu": dist.between(hubs[k], b),
                    "d_to_slack_pu": dist.to_ref(b),
                }
            )
    return placement


def _new_bus_ids(base: Mapping[str, Any], n: int) -> list[Any]:
    existing = [b["idx"] for b in base["buses"]]
    if all(isinstance(v, (int, np.integer)) for v in existing):
        start = 900001
        ids = [start + k for k in range(n)]
    else:
        ids = [f"COL{k + 1:02d}" for k in range(n)]
    if set(map(str, ids)) & set(map(str, existing)):
        raise BenchmarkError("collector bus idx collides with an existing bus")
    return ids


def _dispatch_units(base: Mapping[str, Any]) -> list[dict[str, Any]]:
    rating: dict[str, float] = {}
    gov_min: dict[str, float] = {}
    for m in base["machines"]:
        if m["u"] == 0.0:
            continue
        key = str(m["gen"])
        rating[key] = rating.get(key, 0.0) + m["sn_mva"] / base["mva"]
        for g in m["governors"]:
            if g["u"] != 0.0 and g.get("vmin_sys_pu") is not None:
                gov_min[key] = gov_min.get(key, 0.0) + max(float(g["vmin_sys_pu"]), 0.0)
    units = []
    rows = [dict(g, kind="PV") for g in base["pv"] if g["u"] != 0.0]
    sl = base["slack"]
    rows.append({"idx": sl["idx"], "bus": sl["bus"], "p0": sl["p_pu"], "pmax": sl["pmax"], "u": sl["u"],
                 "kind": "Slack"})
    for r in rows:
        key = str(r["idx"])
        p0 = float(r["p0"])
        cands = []
        if r["pmax"] < PMAX_ABSENT_PU:
            cands.append(float(r["pmax"]))
        rt = rating.get(key)
        if rt is not None and rt > 0.0 and rt >= p0 - 1e-9:
            cands.append(rt)
        cap = max(min(cands), p0) if cands else p0
        if p0 > 0.0:
            gmin = gov_min.get(key)
            gfloor = gmin + GOV_MIN_MARGIN_PU if gmin is not None and gmin > 0.0 else 0.0
            floor = min(p0, max(MUST_RUN_FRACTION * cap, gfloor))
            binding = "governor VMIN" if gfloor > MUST_RUN_FRACTION * cap else "20 % capability"
        else:
            floor, binding = p0, "fixed (stock output <= 0)"
        units.append({"idx": r["idx"], "kind": r["kind"], "bus": r["bus"], "p0_pu": p0, "pmax_pu": r["pmax"],
                      "rating_pu": rt, "capability_pu": cap, "governor_min_pu": gov_min.get(key),
                      "must_run_pu": floor, "must_run_basis": binding})
    return units


def _condenser_uel(spec: SystemSpec, base: Mapping[str, Any], scenario: Scenario) -> dict[str, Any]:
    if scenario.name != "light_load" or spec.slug not in UEL_SYSTEMS:
        return {"applied": False, "devices": {}}
    pv = {str(g["idx"]): g for g in base["pv"] if g["u"] != 0.0}
    sn: dict[str, float] = {}
    machines: dict[str, list] = {}
    for m in base["machines"]:
        if m["u"] == 0.0 or m["model"] != "GENCLS":
            continue
        key = str(m["gen"])
        if key in pv and abs(pv[key]["p0"]) * base["mva"] < CONDENSER_P_MAX_MW:
            sn[key] = sn.get(key, 0.0) + m["sn_mva"]
            machines.setdefault(key, []).append(m["idx"])
    devices = {}
    for key, s_mva in sn.items():
        devices[key] = {"pv": pv[key]["idx"], "bus": pv[key]["bus"], "machines": machines[key], "sn_mva": s_mva,
                        "qlim_pu": UEL_Q_OVER_SN * s_mva / base["mva"], "v0_orig": pv[key]["v0"]}
    return {"applied": True, "q_over_sn": UEL_Q_OVER_SN, "p_max_mw": CONDENSER_P_MAX_MW, "devices": devices,
            "rule": "zero-output synchronous condensers (GENCLS, |stock P| < 1 MW): Q >= -0.30 Sn (MVA rating, "
                    "ANDES default 100 MVA where the data give none), enforced per device in the power flow: a "
                    "condenser below its limit is switched from voltage control to fixed Q = limit (ANDES PV->PQ "
                    "switching with qmin = limit on the condensers only; every other PV limit, all data "
                    "placeholders, is set non-binding at +/-1e6 pu)"}


def _water_fill(units: Sequence[Mapping[str, Any]], target: float) -> tuple[float, list[float], bool]:
    def total(k: float) -> list[float]:
        return [u["p0_pu"] if u["p0_pu"] <= 0.0 else max(k * u["p0_pu"], u["must_run_pu"]) for u in units]

    if sum(total(0.0)) >= target:
        return 0.0, total(0.0), False
    if sum(total(1.0)) <= target:
        return 1.0, total(1.0), True
    lo, hi = 0.0, 1.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if sum(total(mid)) < target:
            lo = mid
        else:
            hi = mid
    k = 0.5 * (lo + hi)
    return k, total(k), True


def _dispatch(base: Mapping[str, Any], scenario: Scenario, load_p: float, p_res: float,
              loss_pu: float) -> dict[str, Any]:
    pv_on = [g for g in base["pv"] if g["u"] != 0.0]
    pv_total = sum(g["p0"] for g in pv_on)
    if scenario.dispatch_rule == "pv_only_slack_keeps_base":
        k = (pv_total - p_res) / pv_total
        if not (0.0 < k < 1.0):
            raise BenchmarkError(f"infeasible dispatch scale {k}")
        return {"rule": scenario.dispatch_rule, "k": k, "pv_p0": {str(g["idx"]): g["p0"] * k for g in pv_on},
                "slack_scheduled_pu": base["slack"]["p_pu"], "slack_must_run_pu": 0.0, "feasible": True,
                "target_sync_pu": pv_total * k + base["slack"]["p_pu"], "loss_estimate_pu": loss_pu,
                "units": []}
    units = _dispatch_units(base)
    target = load_p - p_res + loss_pu
    k, p_new, feasible = _water_fill(units, target)
    rows = []
    for u, pn in zip(units, p_new):
        rows.append(dict(u, p_new_pu=pn, at_must_run=bool(u["p0_pu"] > 0 and pn <= u["must_run_pu"] + 1e-12
                                                       and u["must_run_pu"] > k * u["p0_pu"])))
    slack_row = next(r for r in rows if r["kind"] == "Slack")
    return {"rule": scenario.dispatch_rule, "k": k, "feasible": feasible, "target_sync_pu": target,
            "loss_estimate_pu": loss_pu,
            "pv_p0": {str(r["idx"]): r["p_new_pu"] for r in rows if r["kind"] == "PV"},
            "slack_scheduled_pu": slack_row["p_new_pu"], "slack_must_run_pu": slack_row["must_run_pu"],
            "must_run_total_pu": sum(r["must_run_pu"] for r in rows), "units": rows}


def _make_plan(spec: SystemSpec, base: Mapping[str, Any], banned: Mapping[str, str] | None = None,
               scenario: Scenario | None = None) -> dict[str, Any]:
    sb = base["mva"]
    bus_by_id = {str(b["idx"]): b for b in base["buses"]}
    dist = _ElectricalDistance(base)
    n = spec.n_blocks
    scenario = scenario or SCENARIOS["light_load"]
    load_stock = sum(ld["p0"] for ld in base["loads"] if ld["u"] != 0.0)
    load_p = scenario.load_scale * load_stock
    load_p_gross = scenario.load_scale * sum(max(ld["p0"], 0.0) for ld in base["loads"] if ld["u"] != 0.0)
    p_total = scenario.res_share * load_p
    p_mean = p_total / n
    p_max = p_mean * (1.0 + BLOCK_P_SPREAD)
    candidates, excluded = _candidate_buses(base, dist, p_max)
    banned = dict(banned or {})
    if banned:
        excluded += [{"bus": b, "reason": banned[str(b)]} for b in candidates if str(b) in banned]
        candidates = [b for b in candidates if str(b) not in banned]
    placement = _place_blocks(spec, candidates, dist, base["x_thevenin"])
    s_levels = np.linspace(-1.0, 1.0, n) if n > 1 else np.zeros(1)
    t_levels = np.linspace(0.0, 1.0, n) if n > 1 else np.zeros(1)
    p_rank = _low_discrepancy_ranks(n, _GOLDEN)
    z_rank = _low_discrepancy_ranks(n, _SILVER)

    pv_total = sum(g["p0"] for g in base["pv"] if g["u"] != 0.0)
    stock_losses = pv_total + base["slack"]["p_pu"] - load_stock
    dispatch = _dispatch(base, scenario, load_p, p_total, stock_losses)
    uel = _condenser_uel(spec, base, scenario)

    col_ids = _new_bus_ids(base, n)
    blocks = []
    for k, place in enumerate(placement):
        hv = bus_by_id[str(place["hv_bus"])]
        p_pu = p_mean * (1.0 + BLOCK_P_SPREAD * float(s_levels[p_rank[k]]))
        sn = PLANT_SN_OVER_P * p_pu * sb
        x_sys = GSU_X_DEVICE_PU * sb / sn
        r_sys = x_sys / GSU_X_OVER_R
        q_pu = KAPPA_Q_OVER_P * p_pu
        z0 = Z0_RANGE_PU[0] + (Z0_RANGE_PU[1] - Z0_RANGE_PU[0]) * float(t_levels[z_rank[k]])
        vp = z0 + (x_sys * abs(q_pu) - r_sys * p_pu) / z0
        tap0 = hv["v"] / vp
        aid = f"RES{k + 1:02d}"
        xth = base["x_thevenin"][str(place["hv_bus"])]
        blocks.append(
            {
                "asset_id": aid,
                "s_sc_mva": sb / xth,
                "scr": (1.0 / xth) / p_pu,
                "cluster": place["cluster"],
                "rank_in_cluster": place["rank_in_cluster"],
                "seed_bus": place["seed_bus"],
                "d_to_seed_pu": place["d_to_seed_pu"],
                "d_to_slack_pu": place["d_to_slack_pu"],
                "hv_bus": place["hv_bus"],
                "hv_vn_kv": hv["vn_kv"],
                "hv_area": hv["area"],
                "hv_zone": hv["zone"],
                "hv_v_base_pf": hv["v"],
                "hv_a_base_pf": hv["a"],
                "collector_bus": col_ids[k],
                "gsu_idx": f"GSU_{aid}",
                "pv_idx": f"PV_{aid}",
                "pvd1_idx": aid,
                "p_pu": p_pu,
                "p_mw": p_pu * sb,
                "q_target_pu": q_pu,
                "q_target_mvar": q_pu * sb,
                "sn_mva": sn,
                "gsu_x_device_pu": GSU_X_DEVICE_PU,
                "gsu_r_device_pu": GSU_X_DEVICE_PU / GSU_X_OVER_R,
                "gsu_x_pu_sysbase": x_sys,
                "gsu_r_pu_sysbase": r_sys,
                "z0_pu": z0,
                "tap_initial_guess": tap0,
                "tap": tap0,
            }
        )

    by_cluster: dict[int, list[dict[str, Any]]] = {}
    for blk in blocks:
        by_cluster.setdefault(blk["cluster"], []).append(blk)
    n_c = spec.n_clusters
    n_quarter = max(1, n // 4)

    def _pick(j: int, from_end: bool, avoid: set[str]) -> dict[str, Any]:
        members = by_cluster[j % n_c]
        seq = list(reversed(members)) if from_end else list(members)
        start = (j // n_c) % len(seq)
        seq = seq[start:] + seq[:start]
        for blk in seq:
            if str(blk["hv_bus"]) not in avoid:
                return blk
        return seq[0]

    reactor_sizes = _midpoint_sizes(n_quarter, *REACTOR_MVAR_RANGE)
    reactors, used = [], set()
    for j in range(n_quarter):
        blk = _pick(j, False, used)
        used.add(str(blk["hv_bus"]))
        reactors.append(
            {
                "idx": f"SHR{j + 1:02d}",
                "bus": blk["hv_bus"],
                "vn_kv": blk["hv_vn_kv"],
                "mvar": reactor_sizes[j],
                "b_sys": -reactor_sizes[j] / sb,
                "near_block": blk["asset_id"],
                "cluster": blk["cluster"],
            }
        )
    reactor_buses = {str(r["bus"]) for r in reactors}
    cap_sizes = _midpoint_sizes(N_CAPACITORS, *CAPACITOR_MVAR_RANGE)
    capacitors, used_c = [], set()
    for j in range(N_CAPACITORS):
        blk = _pick(j, False, reactor_buses | used_c)
        used_c.add(str(blk["hv_bus"]))
        capacitors.append(
            {
                "idx": f"SHC{j + 1:02d}",
                "bus": blk["hv_bus"],
                "vn_kv": blk["hv_vn_kv"],
                "mvar": cap_sizes[j],
                "b_sys": cap_sizes[j] / sb,
                "near_block": blk["asset_id"],
                "cluster": blk["cluster"],
            }
        )
    statcoms, used_s = [], set()
    for j in range(n_quarter):
        blk = _pick(j, True, used_s)
        used_s.add(str(blk["hv_bus"]))
        statcoms.append(
            {
                "idx": f"STATCOM{j + 1:02d}",
                "bus": blk["hv_bus"],
                "vn_kv": blk["hv_vn_kv"],
                "mvar": spec.statcom_mvar,
                "near_block": blk["asset_id"],
                "cluster": blk["cluster"],
            }
        )
    fast = [
        {"idx": f"FASTR{j + 1:02d}", "bus": r["bus"], "vn_kv": r["vn_kv"], "mvar": FAST_REACTOR_MVAR, "cluster": r["cluster"]}
        for j, r in enumerate(reactors)
    ]
    manual = [
        {"idx": f"MANR{j + 1:02d}", "bus": r["bus"], "vn_kv": r["vn_kv"], "mvar": MANUAL_REACTOR_MVAR, "cluster": r["cluster"]}
        for j, r in enumerate(reactors)
    ]

    ehv = []
    for ln in base["lines"]:
        if ln["u"] == 0.0 or ln["is_transformer"] or ln["b_sys"] <= 0.0:
            continue
        if min(bus_by_id[str(ln["bus1"])]["vn_kv"], bus_by_id[str(ln["bus2"])]["vn_kv"]) < HV_MIN_KV:
            continue
        ehv.append(ln)
    line_order = {str(ln["idx"]): k for k, ln in enumerate(base["lines"])}
    ehv.sort(key=lambda ln: (-round(ln["b_sys"], 12), line_order[str(ln["idx"])]))
    parallel, corridors = [], set()
    for ln in ehv:
        corridor = frozenset((str(ln["bus1"]), str(ln["bus2"])))
        if corridor in corridors:
            continue
        corridors.add(corridor)
        parallel.append(
            {
                "idx": f"LPAR{len(parallel) + 1:02d}",
                "parallel_to": ln["idx"],
                "bus1": ln["bus1"],
                "bus2": ln["bus2"],
                "b_sys": ln["b_sys"],
                "x_sys": ln["x_sys"],
                "charging_mvar_at_1pu": ln["b_sys"] * sb,
                "input": dict(ln["input"]),
            }
        )
        if len(parallel) == N_PARALLEL_LINES:
            break

    return {
        "slug": spec.slug,
        "scenario": {"name": scenario.name, "load_scale": scenario.load_scale, "res_share": scenario.res_share,
                     "tap_range": list(scenario.tap_range), "dispatch_rule": scenario.dispatch_rule,
                     "must_run_fraction": scenario.must_run_fraction, "description": scenario.description},
        "mva": sb,
        "freq": base["freq"],
        "load_p_stock_pu": load_stock,
        "load_p_pu": load_p,
        "load_p_gross_pu": load_p_gross,
        "res_p_total_pu": p_total,
        "pv_p0_total_pu": pv_total,
        "stock_losses_pu": stock_losses,
        "dispatch": dispatch,
        "uel": uel,
        "dispatch_scale": dispatch["k"],
        "scaled_pv": [g["idx"] for g in base["pv"] if g["u"] != 0.0],
        "candidates": list(candidates),
        "excluded_candidates": excluded,
        "p_max_block_pu": p_max,
        "cluster_seeds": sorted({(blk["cluster"], str(blk["seed_bus"])) for blk in blocks}),
        "blocks": blocks,
        "reactors": reactors,
        "capacitors": capacitors,
        "parallel_lines": parallel,
        "statcoms": statcoms,
        "fast_reactors": fast,
        "manual_reactors": manual,
        "collector_reactors": [],
        "pvd1": {},
    }


def _pvd1_params(plan: Mapping[str, Any], blk: Mapping[str, Any], variant: str) -> dict[str, Any]:
    params: dict[str, Any] = {
        "idx": blk["pvd1_idx"],
        "name": blk["pvd1_idx"],
        "bus": blk["collector_bus"],
        "gen": blk["pv_idx"],
        "Sn": blk["sn_mva"],
        "fn": plan["freq"],
        "pqflag": 0,
        "ialim": PVD1_IALIM,
        "qmx": PVD1_QLIM_DEVICE,
        "qmn": -PVD1_QLIM_DEVICE,
        "tip": 0.02,
        "tiq": 0.02,
        "xc": 0.0,
        "ddn": 0.0,
        "gammap": 1.0,
        "gammaq": 1.0,
        **PVD1_TRIP_DISABLED,
        **PVD1_FIXED_PF,
    }
    if variant == "voltage_mode_res":
        q_dev = blk["q_pf_pu"] * plan["mva"] / blk["sn_mva"]
        w, d = VOLTAGE_MODE_SAT_DV_PU, VOLTAGE_MODE_BREAKPOINT_OFFSET_PU
        params.update(
            {
                "dqdv": -q_dev / (2.0 * w + d),
                "qmn": q_dev * w / (2.0 * w + d),
                "gammaq": 2.0 * w / (2.0 * w + d),
                "qmx": 0.0,
                "v0": 0.01,
                "v1": blk["z0_pu"] - d,
            }
        )
    return params


def _build_system(plan: Mapping[str, Any], variant: str, *, init_tds: bool = True,
                  tds_config: Mapping[str, Any] | None = None, plants_as_pq: bool = False) -> Any:
    spec = SYSTEMS[plan["slug"]]
    uel = plan.get("uel") or {}
    use_uel = bool(uel.get("applied")) and not plan.get("_uel_disabled", False)
    ss = _load_raw(spec, pv2pq=use_uel)
    _prepare_raw(ss, plan["scenario"]["load_scale"], plan["slug"])
    sb = plan["mva"]
    freq = plan["freq"]
    new_p0 = {str(k): v for k, v in plan["dispatch"]["pv_p0"].items()}
    for i in range(ss.PV.n):
        key = str(ss.PV.idx.v[i])
        if key in new_p0:
            ss.PV.p0.v[i] = float(new_p0[key])

    with _quiet():
        for blk in plan["blocks"]:
            ss.add("Bus", {
                "idx": blk["collector_bus"], "name": f"COL_{blk['asset_id']}", "Vn": COLLECTOR_KV,
                "v0": blk["z0_pu"], "a0": blk["hv_a_base_pf"], "area": blk["hv_area"], "zone": blk["hv_zone"],
            })
            ss.add("Line", {
                "idx": blk["gsu_idx"], "name": blk["gsu_idx"], "bus1": blk["hv_bus"], "bus2": blk["collector_bus"],
                "Sn": blk["sn_mva"], "fn": freq, "Vn1": blk["hv_vn_kv"], "Vn2": COLLECTOR_KV,
                "r": blk["gsu_r_device_pu"], "x": blk["gsu_x_device_pu"], "b": 0.0,
                "tap": blk["tap"], "phi": 0.0, "trans": 1,
            })
            if plants_as_pq:
                ss.add("PQ", {
                    "idx": f"INJ_{blk['asset_id']}", "name": f"INJ_{blk['asset_id']}", "bus": blk["collector_bus"],
                    "Vn": COLLECTOR_KV, "p0": -blk["p_pu"], "q0": -blk["q_target_pu"], "vmin": 0.5, "vmax": 1.5,
                })
                continue
            ss.add("PV", {
                "idx": blk["pv_idx"], "name": blk["pv_idx"], "bus": blk["collector_bus"], "Sn": blk["sn_mva"],
                "Vn": COLLECTOR_KV, "p0": blk["p_pu"], "q0": blk["q_target_pu"], "pmax": blk["p_pu"], "pmin": 0.0,
                "qmax": blk["sn_mva"] / sb, "qmin": -blk["sn_mva"] / sb, "v0": blk["z0_pu"],
            })
            ss.add("PVD1", _pvd1_params(plan, blk, variant))
        for r in plan["reactors"]:
            ss.add("Shunt", {"idx": r["idx"], "name": r["idx"], "bus": r["bus"], "Sn": sb, "Vn": r["vn_kv"],
                             "fn": freq, "g": 0.0, "b": r["b_sys"], "u": 1})
        for c in plan["capacitors"]:
            ss.add("Shunt", {"idx": c["idx"], "name": c["idx"], "bus": c["bus"], "Sn": sb, "Vn": c["vn_kv"],
                             "fn": freq, "g": 0.0, "b": c["b_sys"], "u": 0})
        for group in ("statcoms", "fast_reactors", "manual_reactors"):
            for c in plan[group]:
                ss.add("Shunt", {"idx": c["idx"], "name": c["idx"], "bus": c["bus"], "Sn": sb, "Vn": c["vn_kv"],
                                 "fn": freq, "g": 0.0, "b": 0.0, "u": 1})
        for c in plan["collector_reactors"]:
            ss.add("Shunt", {"idx": c["idx"], "name": c["idx"], "bus": c["bus"], "Sn": sb, "Vn": COLLECTOR_KV,
                             "fn": freq, "g": 0.0, "b": c["b_sys"], "u": 0})
        for ln in plan["parallel_lines"]:
            params = {name: ln["input"][name] for name in ln["input"]}
            params.update({"idx": ln["idx"], "name": ln["idx"], "bus1": ln["bus1"], "bus2": ln["bus2"], "u": 0})
            ss.add("Line", params)
        if use_uel:
            dev = uel["devices"]
            for i in range(ss.PV.n):
                key = str(ss.PV.idx.v[i])
                ss.PV.qmax.v[i] = UEL_NONBINDING_PU
                ss.PV.qmin.v[i] = float(dev[key]["qlim_pu"]) if key in dev else -UEL_NONBINDING_PU
        ss.setup()
        converged = bool(ss.PFlow.run())
    if not converged:
        raise BenchmarkError(f"{plan['slug']}: power flow of the augmented system did not converge")
    if init_tds:
        cfg = dict(TDS_DEFAULTS)
        cfg.update(dict(tds_config or {}))
        for key, value in cfg.items():
            setattr(ss.TDS.config, key, value)
        with _quiet():
            ss.TDS.init()
        if not bool(getattr(ss.TDS, "initialized", False)):
            raise BenchmarkError(f"{plan['slug']}: TDS initialization failed")
        if variant == "voltage_mode_res":
            _consistent_pvd1_ipmax(ss)
    return ss


def _consistent_pvd1_ipmax(ss: Any) -> None:
    model = ss.PVD1
    addr = np.asarray(model.Ipmax.a, dtype=int)
    for _ in range(3):
        g = np.asarray(ss.dae.g, dtype=float)[addr]
        if np.max(np.abs(g)) < 1e-13:
            break
        ss.dae.y[addr] += g
        ss.vars_to_models()
        ss.TDS.fg_update(ss.exist.pflow_tds)
    with _quiet():
        ss.TDS.test_ok = bool(ss.TDS.test_init())
    residual = float(np.max(np.abs(ss.dae.fg)))
    if residual > 1e-8:
        raise BenchmarkError(f"voltage_mode_res: inconsistent PVD1 initialization (max |f,g| = {residual:.2e})")


def _solve_taps(plan: dict[str, Any]) -> dict[str, Any]:
    ss = _build_system(plan, "baseline", init_tds=False, plants_as_pq=True)
    blocks = plan["blocks"]
    line_pos = np.array([_pos(ss.Line, blk["gsu_idx"]) for blk in blocks])
    col_pos = np.array([_pos(ss.Bus, blk["collector_bus"]) for blk in blocks])
    z0 = np.array([blk["z0_pu"] for blk in blocks])
    lo, hi = plan["scenario"]["tap_range"]
    taps = np.clip(np.array([blk["tap_initial_guess"] for blk in blocks], dtype=float), lo, hi)
    dispatch = plan["dispatch"]
    correct = dispatch["rule"] == "proportional_must_run"
    units = dispatch.get("units", [])
    pv_pos = {str(k): _pos(ss.PV, k) for k in dispatch["pv_p0"]}
    target = float(dispatch["target_sync_pu"])
    k_fac = float(dispatch["k"])
    history: list[dict[str, Any]] = []
    converged = False
    v_col = np.full(len(blocks), np.nan)
    slack_dev = 0.0
    for it in range(150):
        ss.Line.tap.v[line_pos] = taps
        with _quiet():
            ok = bool(ss.PFlow.run())
        if not ok:
            raise BenchmarkError(f"{plan['slug']}: power flow with fixed-PF plant injections did not converge "
                                 f"(iteration {it}); the operating point is infeasible")
        v_col = np.asarray(ss.Bus.v.v, dtype=float)[col_pos]
        new = np.clip(taps * v_col / z0, lo, hi)
        at_limit = (new <= lo) | (new >= hi)
        err = np.where(at_limit, np.abs(new - taps), np.abs(v_col - z0))
        slack_p = float(ss.Slack.p.v[0])
        slack_dev = slack_p - float(dispatch["slack_scheduled_pu"]) if correct else 0.0
        history.append({"iteration": it, "max_abs_v_col_error_pu": float(np.max(np.abs(v_col - z0)[~at_limit]))
                        if np.any(~at_limit) else 0.0, "n_at_limit": int(at_limit.sum()),
                        "slack_p_pu": slack_p, "slack_deviation_pu": slack_dev})
        if correct and not dispatch["feasible"]:
            break
        if np.max(err) < 1e-10 and abs(slack_dev) < 1e-9:
            converged = True
            break
        taps = new
        if correct and abs(slack_dev) >= 1e-9:
            target += slack_dev
            k_fac, p_new, feasible = _water_fill(units, target)
            dispatch["feasible"] = feasible
            for u, pn in zip(units, p_new):
                if u["kind"] == "PV":
                    dispatch["pv_p0"][str(u["idx"])] = pn
                    ss.PV.p0.v[pv_pos[str(u["idx"])]] = pn
                else:
                    dispatch["slack_scheduled_pu"] = pn
            for u, pn in zip(units, p_new):
                u["p_new_pu"] = pn
                u["at_must_run"] = bool(u["p0_pu"] > 0 and pn <= u["must_run_pu"] + 1e-12
                                        and u["must_run_pu"] > k_fac * u["p0_pu"])
            dispatch["k"] = k_fac
            dispatch["target_sync_pu"] = target
    dispatch["loss_corrected"] = correct
    dispatch["slack_pf_pu"] = float(ss.Slack.p.v[0])
    if correct and not dispatch["feasible"]:
        return {"infeasible_dispatch": True, "dispatch": dispatch, "history": history}
    if not converged:
        raise BenchmarkError(f"{plan['slug']}: tap iteration did not converge ({history[-1]})")
    at_limit = (taps <= lo + 1e-12) | (taps >= hi - 1e-12)
    return {
        "infeasible_dispatch": False,
        "dispatch": dispatch,
        "taps": taps.tolist(),
        "z0_op_pu": v_col.tolist(),
        "tap_at_limit": at_limit.tolist(),
        "v_hv_pu": np.asarray(ss.Bus.v.v, dtype=float)[[_pos(ss.Bus, b["hv_bus"]) for b in blocks]].tolist(),
        "iterations": len(history),
        "history": history,
        "converged": converged,
    }


_STEP = TimeProfile(ProfileKind.STEP, description="step at the event time")


def _ramp(duration: float) -> TimeProfile:
    return TimeProfile(ProfileKind.RAMP, start_s=0.0, duration_s=duration, final_value=1.0,
                       description=f"linear ramp over {duration:g} s")


def _machine_rank_key(m: Mapping[str, Any]) -> tuple:
    return (-m["sn_mva"], -abs(m.get("p_op_pu", m["p_base_pu"])), str(m["idx"]))


def _machine_op_table(ss: Any, base: Mapping[str, Any]) -> list[dict[str, Any]]:
    inv = _machine_inventory(ss)["machines"]
    base_p = {str(m["idx"]): m["p_base_pu"] for m in base["machines"]}
    bus_area = {str(ss.Bus.idx.v[i]): ss.Bus.area.v[i] for i in range(ss.Bus.n)}
    for m in inv:
        model = getattr(ss, m["model"])
        p = _pos(model, m["idx"])
        m["p_base_pu"] = base_p.get(str(m["idx"]), float("nan"))
        m["p_op_pu"] = float(model.tm0.v[p])
        m["q_op_pu"] = float(model.Qe.v[p]) if hasattr(model, "Qe") else float("nan")
        m["area"] = bus_area.get(str(m["bus"]))
        m["slack_linked"] = str(m["gen"]) in {str(v) for v in ss.Slack.idx.v}
    return inv


def _plant_events(ss: Any, plan: Mapping[str, Any], variant: str) -> list[EventSpec]:
    sb = plan["mva"]
    col_reactor = {c["block"]: c for c in plan["collector_reactors"]}
    events: list[EventSpec] = []
    for blk in plan["blocks"]:
        aid = blk["asset_id"]
        muts = [Mutation("PVD1", blk["pvd1_idx"], "u", 0.0)]
        meta: dict[str, Any] = {"asset_id": aid, "bus": blk["collector_bus"], "hv_bus": blk["hv_bus"],
                                "p_mw": blk["p_mw"], "q_mvar": blk["q_pf_pu"] * sb, "cluster": blk["cluster"]}
        desc = f"trip of plant {aid} ({blk['p_mw']:.1f} MW, {blk['q_pf_pu'] * sb:.1f} Mvar)"
        if variant == "preserved_absorption":
            cr = col_reactor[aid]
            muts.append(Mutation("Shunt", cr["idx"], "u", 1.0))
            meta["collector_reactor"] = cr["idx"]
            meta["collector_reactor_mvar_at_z0"] = -cr["b_sys"] * blk["z0_pu"] ** 2 * sb
            desc += f"; collector reactor {cr['idx']} switched in (preserves absorption)"
        events.append(EventSpec(f"plant_trip:{aid}", "plant_trip", "switch", tuple(muts), _STEP, desc,
                                disconnects_assets=(aid,), metadata=meta))
    for blk in plan["blocks"]:
        aid = blk["asset_id"]
        p = _pos(ss.PVD1, blk["pvd1_idx"])
        pref0 = float(ss.PVD1.pref0.v[p])
        qref0 = float(ss.PVD1.qref0.v[p])
        q_plant = float(ss.PVD1.Qsum.v[p])
        dq = (FIXED_PF_RAMP_FRACTION - 1.0) * q_plant
        muts = (
            Mutation("PVD1", blk["pvd1_idx"], "pref0", FIXED_PF_RAMP_FRACTION * pref0),
            Mutation("PVD1", blk["pvd1_idx"], "qref0", qref0 + dq),
        )
        meta = {"asset_id": aid, "bus": blk["collector_bus"], "dp_mw": (FIXED_PF_RAMP_FRACTION - 1) * pref0 * sb,
                "dq_mvar": dq * sb, "cluster": blk["cluster"]}
        events.append(EventSpec(
            f"fixed_pf_ramp:{aid}", "fixed_pf_ramp", "param", muts, _ramp(FIXED_PF_RAMP_DURATION_S),
            f"fixed-PF output ramp of {aid} to {FIXED_PF_RAMP_FRACTION:.0%} over {FIXED_PF_RAMP_DURATION_S:g} s",
            metadata=meta))
    return events


def _load_events(ss: Any, plan: Mapping[str, Any], base: Mapping[str, Any]) -> list[EventSpec]:
    sb = plan["mva"]
    n_loads = plan["n_blocks"] // 2
    order = {str(ld["idx"]): k for k, ld in enumerate(base["loads"])}
    loads = [ld for ld in base["loads"] if ld["u"] != 0.0]
    loads.sort(key=lambda ld: (-abs(ld["q0"]), order[str(ld["idx"])]))
    scale = float(plan["scenario"]["load_scale"])
    loads = [dict(ld, p0=ld["p0"] * scale, q0=ld["q0"] * scale) for ld in loads]
    events = []
    for ld in loads[:n_loads]:
        meta = {"load": ld["idx"], "bus": ld["bus"], "p_mw": ld["p0"] * sb, "q_mvar": ld["q0"] * sb,
                "negative_q": bool(ld["q0"] < 0.0)}
        flag = " [net capacitive load: q<0]" if ld["q0"] < 0 else ""
        events.append(EventSpec(
            f"load_disconnection:{ld['idx']}", "load_disconnection", "switch",
            (Mutation("PQ", ld["idx"], "u", 0.0),), _STEP,
            f"disconnection of load {ld['idx']} at bus {ld['bus']} ({ld['p0'] * sb:.1f} MW, {ld['q0'] * sb:+.1f} Mvar){flag}",
            metadata=meta))
    return events


def _generator_events(ss: Any, machines: Sequence[Mapping[str, Any]],
                      plan: Mapping[str, Any]) -> tuple[list[EventSpec], list[dict]]:
    sb = plan["mva"]
    eligible = [m for m in machines if m["u"] != 0.0 and not m["slack_linked"]]
    eligible.sort(key=_machine_rank_key)
    chosen = eligible[:N_GENERATOR_TRIPS]
    events = []
    for m in chosen:
        muts = generator_trip_mutations(ss, m["idx"], include_effective_status=True)
        meta = {"machine": m["idx"], "model": m["model"], "bus": m["bus"], "sn_mva": m["sn_mva"],
                "p_mw": m["p_op_pu"] * sb, "q_mvar": m["q_op_pu"] * sb,
                "linked": sorted({f"{mu.model}:{mu.device_idx}" for mu in muts[1:]})}
        events.append(EventSpec(
            f"generator_trip:{m['idx']}", "generator_trip", "switch", muts, _STEP,
            f"trip of {m['model']} {m['idx']} at bus {m['bus']} ({m['sn_mva']:.0f} MVA, {m['p_op_pu'] * sb:.1f} MW)"
            f" with {len(meta['linked'])} linked controller(s)",
            metadata=meta))
    return events, chosen


def _redispatch_events(ss: Any, machines: Sequence[Mapping[str, Any]], plan: Mapping[str, Any],
                       dist: _ElectricalDistance) -> tuple[list[EventSpec], dict[str, Any]]:
    sb = plan["mva"]
    in_service = [m for m in machines if m["u"] != 0.0]
    governed = [m for m in in_service if any(g["u"] != 0.0 for g in m["governors"])]
    if governed:
        pool, channel = governed, "governor pref0"
    else:
        pool = [m for m in in_service if not m["governors"] and m["p_op_pu"] > 0.0]
        channel = "classical-machine mechanical power tm0 (no governors in the case)"
    bus_area = {str(ss.Bus.idx.v[i]): ss.Bus.area.v[i] for i in range(ss.Bus.n)}
    areas_with_pool = sorted({str(bus_area[str(m["bus"])]) for m in pool})
    info: dict[str, Any] = {"channel": channel}
    if len(areas_with_pool) >= 2:
        info["partition"] = "case areas"
        area_of = {str(b): str(a) for b, a in bus_area.items()}
    else:
        area_of = _spectral_halves(dist)
        info["partition"] = "spectral bisection (median split of the Fiedler vector of the 1/|x| Laplacian)"
    export: dict[str, float] = {}
    for m in in_service:
        a = area_of.get(str(m["bus"]))
        if a is not None:
            export[a] = export.get(a, 0.0) + m["p_op_pu"]
    hv_of_collector = {str(b["collector_bus"]): str(b["hv_bus"]) for b in plan["blocks"]}
    for i in range(ss.PVD1.n):
        col = str(ss.PVD1.bus.v[i])
        a = area_of.get(col, area_of.get(hv_of_collector.get(col, col)))
        if a is not None:
            export[a] = export.get(a, 0.0) + float(ss.PVD1.pref0.v[i])
    for i in range(ss.PQ.n):
        a = area_of.get(str(ss.PQ.bus.v[i]))
        if a is not None and float(ss.PQ.u.v[i]) != 0.0:
            export[a] = export.get(a, 0.0) - float(ss.PQ.p0.v[i])
    info["net_export_mw"] = {a: v * sb for a, v in sorted(export.items())}
    exporters = {a for a, v in export.items() if v > 0.0}
    importers = {a for a, v in export.items() if v <= 0.0}
    key = lambda m: (-m["p_op_pu"], -m["sn_mva"], str(m["idx"]))
    down = sorted([m for m in pool if area_of.get(str(m["bus"])) in exporters], key=key)
    up = sorted([m for m in pool if area_of.get(str(m["bus"])) in importers], key=key)
    units = {str(u["idx"]): u for u in plan["dispatch"].get("units", [])}

    def _range(m: Mapping[str, Any]) -> tuple[float, float]:
        u = units.get(str(m["gen"]))
        share = float(m.get("gammap", 1.0))
        lo = share * float(u["must_run_pu"]) if u else 0.0
        hi = share * float(u["capability_pu"]) if u else math.inf
        for g in m["governors"]:
            if g["u"] != 0.0:
                gmodel = getattr(ss, g["model"])
                gp = _pos(gmodel, g["idx"])
                if hasattr(gmodel, "VMIN"):
                    lo = max(lo, float(gmodel.VMIN.v[gp]))
                if hasattr(gmodel, "VMAX"):
                    hi = min(hi, float(gmodel.VMAX.v[gp]))
                break
        return lo, hi

    pairs, used = [], set()
    for md in down:
        if len(pairs) == N_REDISPATCH_PAIRS:
            break
        lo_d, _ = _range(md)
        for mu in up:
            if str(mu["idx"]) in used:
                continue
            _, hi_u = _range(mu)
            dp_try = REDISPATCH_FRACTION * min(md["p_op_pu"], mu["p_op_pu"])
            if md["p_op_pu"] - dp_try >= lo_d + 1e-9 and mu["p_op_pu"] + dp_try <= hi_u - 1e-9:
                pairs.append((md, mu))
                used.add(str(mu["idx"]))
                break
    n_pairs = len(pairs)
    info["n_pairs"] = n_pairs
    info["pairing_rule"] = ("down units in exporting, up units in importing (pseudo-)areas, ranked by P; a pair is "
                            "used only if the down unit stays above max(must-run, governor VMIN) and the up unit "
                            "below min(governor VMAX, capability) after the 20 % exchange")
    info["n_down_candidates"] = len(down)
    info["n_down_with_headroom"] = sum(
        1 for m in down if m["p_op_pu"] * (1.0 - REDISPATCH_FRACTION) >= _range(m)[0] + 1e-9)
    events = []
    for md, mu in pairs:
        dp = REDISPATCH_FRACTION * min(md["p_op_pu"], mu["p_op_pu"])
        muts = []
        limits = {}
        for m, sign in ((md, -1.0), (mu, +1.0)):
            if governed:
                gov = next(g for g in m["governors"] if g["u"] != 0.0)
                gmodel = getattr(ss, gov["model"])
                gp = _pos(gmodel, gov["idx"])
                base_v = float(gmodel.pref0.v[gp])
                muts.append(Mutation(gov["model"], gov["idx"], "pref0", base_v + sign * dp))
                if hasattr(gmodel, "VMAX"):
                    limits[str(m["idx"])] = {"vmax_pu": float(gmodel.VMAX.v[gp]), "vmin_pu": float(gmodel.VMIN.v[gp])}
            else:
                mmodel = getattr(ss, m["model"])
                mp = _pos(mmodel, m["idx"])
                base_v = float(mmodel.tm0.v[mp])
                muts.append(Mutation(m["model"], m["idx"], "tm0", base_v + sign * dp))
        new_up = mu["p_op_pu"] + dp
        within = True
        if str(mu["idx"]) in limits:
            within = new_up <= limits[str(mu["idx"])]["vmax_pu"] + 1e-12
        meta = {"down_machine": md["idx"], "up_machine": mu["idx"], "down_area": area_of.get(str(md["bus"])),
                "up_area": area_of.get(str(mu["bus"])), "dp_mw": dp * sb,
                "down_p_mw": md["p_op_pu"] * sb, "up_p_mw": mu["p_op_pu"] * sb,
                "up_within_governor_limit": within, "governor_limits": limits, "channel": channel}
        events.append(EventSpec(
            f"generation_redispatch:{md['idx']}->{mu['idx']}", "generation_redispatch", "param", tuple(muts),
            _ramp(REDISPATCH_RAMP_S),
            f"exchange reduction: {md['idx']} -{dp * sb:.1f} MW, {mu['idx']} +{dp * sb:.1f} MW over {REDISPATCH_RAMP_S:g} s",
            metadata=meta))
    return events, info


def _device_events(ss: Any, plan: Mapping[str, Any]) -> list[EventSpec]:
    events = []
    for r in plan["reactors"]:
        events.append(EventSpec(
            f"shunt_reactor_disconnection:{r['idx']}", "shunt_reactor_disconnection", "switch",
            (Mutation("Shunt", r["idx"], "u", 0.0),), _STEP,
            f"disconnection of {r['mvar']:.1f} Mvar reactor {r['idx']} at bus {r['bus']}",
            metadata={"bus": r["bus"], "q_mvar_at_1pu": -r["mvar"], "near_block": r["near_block"]}))
    for c in plan["capacitors"]:
        events.append(EventSpec(
            f"capacitor_switching:{c['idx']}", "capacitor_switching", "switch",
            (Mutation("Shunt", c["idx"], "u", 1.0),), _STEP,
            f"energization of {c['mvar']:.1f} Mvar capacitor {c['idx']} at bus {c['bus']}",
            metadata={"bus": c["bus"], "q_mvar_at_1pu": c["mvar"], "near_block": c["near_block"]}))
    for ln in plan["parallel_lines"]:
        p = _pos(ss.Line, ln["idx"])
        yhk = 1.0 / complex(float(ss.Line.r.v[p]) + 1e-8, float(ss.Line.x.v[p]) + 1e-8)
        events.append(EventSpec(
            f"line_energization:{ln['idx']}", "line_energization", "switch",
            (Mutation("Line", ln["idx"], "u", 1.0), Mutation("Line", ln["idx"], "ghk", yhk.real),
             Mutation("Line", ln["idx"], "bhk", yhk.imag)), _STEP,
            f"energization of circuit {ln['idx']} parallel to {ln['parallel_to']} ({ln['bus1']}-{ln['bus2']},"
            f" charging {ln['charging_mvar_at_1pu']:.1f} Mvar at 1 pu)",
            metadata={"bus1": ln["bus1"], "bus2": ln["bus2"], "parallel_to": ln["parallel_to"],
                      "charging_mvar_at_1pu": ln["charging_mvar_at_1pu"]}))
    return events


def _controls(ss: Any, plan: Mapping[str, Any], machines: Sequence[Mapping[str, Any]]) -> tuple[list[ControlSpec], dict]:
    sb = plan["mva"]
    controls: list[ControlSpec] = []
    notes: dict[str, Any] = {}
    env_statcom = lag_envelope(STATCOM_DELAY_S, STATCOM_TAU_S, horizon_s=ENVELOPE_HORIZON_S)
    for c in plan["statcoms"]:
        controls.append(ControlSpec(
            f"statcom:{c['idx']}", "statcom", (Mutation("Shunt", c["idx"], "b", -c["mvar"] / sb),), env_statcom,
            alpha_max=1.0, mvar_per_unit=c["mvar"],
            description=f"{c['mvar']:.0f} Mvar STATCOM (absorbing) at bus {c['bus']} as controlled susceptance",
            metadata={"bus": c["bus"], "direction": "absorb", "model": "controlled susceptance b in [-Q/Sb, 0]",
                      "delay_s": STATCOM_DELAY_S, "tau_s": STATCOM_TAU_S}))
    env_plant = lag_envelope(PLANT_Q_DELAY_S, 0.0, horizon_s=ENVELOPE_HORIZON_S)
    cap_factor = math.sqrt(1.0 - PLANT_Q_CAPABILITY_PF**2)
    for blk in plan["blocks"]:
        p = _pos(ss.PVD1, blk["pvd1_idx"])
        qref0 = float(ss.PVD1.qref0.v[p])
        headroom = blk["sn_mva"] / sb * cap_factor - abs(float(blk["q_pf_pu"]))
        controls.append(ControlSpec(
            f"plant_q_command:{blk['asset_id']}", "plant_q_command",
            (Mutation("PVD1", blk["pvd1_idx"], "qref0", qref0 - headroom),), env_plant,
            alpha_max=1.0, mvar_per_unit=headroom * sb,
            description=f"additional absorption command of {blk['asset_id']} up to PF {PLANT_Q_CAPABILITY_PF} at S_n",
            metadata={"asset_id": blk["asset_id"], "bus": blk["collector_bus"], "headroom_mvar": headroom * sb,
                      "delay_s": PLANT_Q_DELAY_S}))
    env_avr = lag_envelope(AVR_DELAY_S, 0.0, horizon_s=ENVELOPE_HORIZON_S)
    with_exc = [m for m in machines if m["u"] != 0.0 and any(e["u"] != 0.0 for e in m["exciters"])]
    with_exc.sort(key=_machine_rank_key)
    for m in with_exc[:N_AVR_CONTROLS]:
        exc = next(e for e in m["exciters"] if e["u"] != 0.0)
        emodel = getattr(ss, exc["model"])
        ep = _pos(emodel, exc["idx"])
        vref0 = float(emodel.vref0.v[ep])
        controls.append(ControlSpec(
            f"avr_vref:{exc['idx']}", "avr_vref", (Mutation(exc["model"], exc["idx"], "vref0", vref0 + AVR_DVREF_PU),),
            env_avr, alpha_max=1.0, mvar_per_unit=None,
            description=f"AVR set-point reduction {AVR_DVREF_PU:+.2f} pu on {exc['model']} {exc['idx']} ({m['idx']})",
            metadata={"machine": m["idx"], "bus": m["bus"], "sn_mva": m["sn_mva"], "delay_s": AVR_DELAY_S,
                      "slack_linked": m["slack_linked"]}))
    if not with_exc:
        notes["avr_vref"] = "not available: the case has no exciter models (classical machines, constant EMF)"
    env_fast = lag_envelope(FAST_REACTOR_DELAY_S, 0.0, horizon_s=ENVELOPE_HORIZON_S)
    for c in plan["fast_reactors"]:
        controls.append(ControlSpec(
            f"fast_reactor:{c['idx']}", "fast_reactor", (Mutation("Shunt", c["idx"], "b", -c["mvar"] / sb),),
            env_fast, alpha_max=1.0, mvar_per_unit=c["mvar"],
            description=f"{c['mvar']:.0f} Mvar breaker-switched reactor at bus {c['bus']} ({FAST_REACTOR_DELAY_S * 1e3:.0f} ms)",
            metadata={"bus": c["bus"], "delay_s": FAST_REACTOR_DELAY_S}))
    env_manual = lag_envelope(MANUAL_REACTOR_DELAY_S, 0.0, horizon_s=MANUAL_REACTOR_DELAY_S + 1.0, dt_s=1e-2)
    for c in plan["manual_reactors"]:
        controls.append(ControlSpec(
            f"manual_reactor:{c['idx']}", "manual_reactor", (Mutation("Shunt", c["idx"], "b", -c["mvar"] / sb),),
            env_manual, alpha_max=1.0, mvar_per_unit=c["mvar"],
            description=f"{c['mvar']:.0f} Mvar operator-switched reactor at bus {c['bus']} ({MANUAL_REACTOR_DELAY_S:g} s)",
            metadata={"bus": c["bus"], "delay_s": MANUAL_REACTOR_DELAY_S}))
    return controls, notes


def device_counts(ss: Any) -> dict[str, Any]:
    def n_of(group: str) -> tuple[int, int, dict[str, int]]:
        total, active, by_model = 0, 0, {}
        grp = ss.groups.get(group)
        if grp is None:
            return 0, 0, {}
        for name, model in grp.models.items():
            if model.n:
                by_model[name] = int(model.n)
                total += int(model.n)
                active += int(np.sum(np.asarray(model.u.v, dtype=float) != 0.0))
        return total, active, by_model

    line = ss.Line
    is_tf = np.array([_is_transformer(line, i) for i in range(line.n)], dtype=bool)
    u_line = np.asarray(line.u.v, dtype=float) != 0.0
    counts: dict[str, Any] = {
        "buses": int(ss.Bus.n),
        "lines": int(np.sum(~is_tf)),
        "lines_in_service": int(np.sum(~is_tf & u_line)),
        "transformers": int(np.sum(is_tf)),
        "transformers_in_service": int(np.sum(is_tf & u_line)),
    }
    for key, group in (("machines", "SynGen"), ("exciters", "Exciter"), ("governors", "TurbineGov"),
                       ("pss", "PSS"), ("loads", "StaticLoad"), ("shunts", "StaticShunt"),
                       ("static_gens", "StaticGen"), ("der", "DG"), ("freq_measurement", "FreqMeasurement")):
        total, active, by_model = n_of(group)
        counts[key] = total
        counts[f"{key}_in_service"] = active
        counts[f"{key}_models"] = by_model
    counts["pvd1"] = int(ss.PVD1.n) if hasattr(ss, "PVD1") else 0
    return counts


def controllers_at_limit(ss: Any) -> list[dict[str, Any]]:
    from andes.core.discrete import AntiWindup

    out: list[dict[str, Any]] = []
    for grp in ("Exciter", "TurbineGov", "DG"):
        group = ss.groups.get(grp)
        if group is None:
            continue
        for name, model in group.models.items():
            if model.n == 0:
                continue
            for key, disc in model.discrete.items():
                if grp == "DG" and key not in ("Ipcmd_lim", "Iqcmd_lim"):
                    continue
                if grp != "DG" and not isinstance(disc, AntiWindup):
                    continue
                zu = np.asarray(getattr(disc, "zu", np.zeros(model.n)), dtype=float).reshape(-1)
                zl = np.asarray(getattr(disc, "zl", np.zeros(model.n)), dtype=float).reshape(-1)
                for i in range(model.n):
                    if (zu.size > i and zu[i] > 0) or (zl.size > i and zl[i] > 0):
                        dev = {"group": grp, "model": name, "idx": model.idx.v[i], "limiter": key,
                               "side": "upper" if zu.size > i and zu[i] > 0 else "lower"}
                        if hasattr(model, "syn"):
                            dev["machine"] = model.syn.v[i]
                        out.append(dev)
    return out


def _uel_reference(plan: Mapping[str, Any]) -> dict[str, Any]:
    ref_plan = dict(plan)
    ref_plan["_uel_disabled"] = True
    ref_plan["blocks"] = [dict(b, tap=b["tap"]) for b in plan["blocks"]]
    try:
        ss0 = _build_system(ref_plan, "baseline", init_tds=False, plants_as_pq=True)
        ss1 = _build_system(plan, "baseline", init_tds=False, plants_as_pq=True)
    except BenchmarkError as exc:
        return {"error": str(exc)}
    q = {str(k): float(ss0.PV.q.v[_pos(ss0.PV, d["pv"])]) for k, d in plan["uel"]["devices"].items()}
    v0 = np.asarray(ss0.Bus.v.v, dtype=float)
    v1 = np.asarray(ss1.Bus.v.v, dtype=float)
    vn = np.asarray(ss1.Bus.Vn.v, dtype=float)
    return {"q": q, "max_voltage_rise_pu": float(np.max(v1 - v0)), "max_voltage_drop_pu": float(np.min(v1 - v0)),
            "hv_v_max_without_uel_pu": float(np.max(v0[vn >= HV_MIN_KV])),
            "bus_v_max_without_uel_pu": float(np.max(v0))}


def _solve_scenario(spec: SystemSpec, base: Mapping[str, Any], scenario: Scenario,
                    slug: str) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    banned: dict[str, str] = {}
    repairs: list[dict[str, Any]] = []
    dist0 = _ElectricalDistance(base)
    bus_of_machine = {str(m["idx"]): m["bus"] for m in base["machines"]}
    for attempt in range(MAX_PLACEMENT_REPAIRS + 1):
        plan = _make_plan(spec, base, banned, scenario)
        plan["n_blocks"] = spec.n_blocks
        tap_result = _solve_taps(plan)
        if tap_result["infeasible_dispatch"]:
            d = tap_result["dispatch"]
            sb_ = plan["mva"]
            units = [u for u in d["units"] if u["p0_pu"] > 0.0]
            n_gov = sum(u["must_run_basis"] == "governor VMIN" for u in units)
            losses = (sum(d["pv_p0"].values()) + d["slack_pf_pu"] + plan["res_p_total_pu"]) - plan["load_p_pu"]
            needed = plan["load_p_pu"] - plan["res_p_total_pu"] + losses
            res_max = 1.0 - (d["must_run_total_pu"] - losses) / plan["load_p_pu"]
            details = {
                "scenario": scenario.name, "must_run_total_mw": d["must_run_total_pu"] * sb_,
                "sync_needed_mw": needed * sb_, "scaled_load_mw": plan["load_p_pu"] * sb_,
                "res_share": scenario.res_share,
                "res_mw": plan["res_p_total_pu"] * sb_, "losses_mw": losses * sb_,
                "n_units_online": len(units), "n_units_governor_minimum": n_gov,
                "slack_at_all_must_run_mw": d["slack_pf_pu"] * sb_, "slack_must_run_mw": d["slack_must_run_pu"] * sb_,
                "max_feasible_res_share": res_max,
                "largest_must_run_units": sorted(
                    [{"unit": u["idx"], "must_run_mw": u["must_run_pu"] * sb_, "basis": u["must_run_basis"],
                      "stock_mw": u["p0_pu"] * sb_} for u in units], key=lambda r: -r["must_run_mw"])[:8],
            }
            raise ScenarioInfeasibleError(
                f"{slug}: {scenario.name} scenario infeasible at RES share {scenario.res_share:.2f} -- the must-run output of the {len(units)} online "
                f"synchronous units ({details['must_run_total_mw']:.0f} MW; {n_gov} of them at their governor minimum "
                f"VMIN) exceeds the synchronous generation the scenario needs ({details['sync_needed_mw']:.0f} MW = "
                f"scaled load {details['scaled_load_mw']:.0f} MW - RES {details['res_mw']:.0f} MW + losses "
                f"{details['losses_mw']:.0f} MW). With every unit at its must-run the slack would produce "
                f"{details['slack_at_all_must_run_mw']:.0f} MW (its own must-run is {details['slack_must_run_mw']:.0f} MW). "
                f"Largest RES share compatible with the must-run: {res_max:.3f}.", details)
        for blk, tap, z0op, lim in zip(plan["blocks"], tap_result["taps"], tap_result["z0_op_pu"],
                                       tap_result["tap_at_limit"]):
            blk["tap"] = float(tap)
            blk["z0_design_pu"] = float(blk["z0_pu"])
            blk["tap_at_limit"] = bool(lim)
            blk["z0_pu"] = float(z0op) if lim else float(blk["z0_pu"])
            blk["q_pf_pu"] = float(blk["q_target_pu"])
        probe = _build_system(plan, "baseline")
        binding = controllers_at_limit(probe)
        if not binding:
            break
        governors = [dev for dev in binding if dev["group"] == "TurbineGov"]
        if governors:
            raise BenchmarkError(f"{slug}: governor limits bind at the {scenario.name} operating point "
                                 f"(dispatch outside the units' output range): {governors}")
        if attempt == MAX_PLACEMENT_REPAIRS:
            raise BenchmarkError(f"{slug}: controllers still at limits after {attempt} placement repairs: {binding}")
        for dev in binding:
            anchor = bus_of_machine.get(str(dev.get("machine")))
            if anchor is None:
                anchor = next(b["hv_bus"] for b in plan["blocks"] if str(b["pvd1_idx"]) == str(dev["idx"]))
            nearest = min(plan["blocks"], key=lambda b: (round(dist0.between(anchor, b["hv_bus"]), 12),
                                                         str(b["hv_bus"])))
            reason = (f"hosting repair {attempt + 1}: {dev['model']} {dev['idx']} {dev['limiter']} at {dev['side']} "
                      f"limit (machine {dev.get('machine')}, bus {anchor})")
            banned[str(nearest["hv_bus"])] = reason
            repairs.append({"attempt": attempt + 1, "binding": dev, "removed_hv_bus": nearest["hv_bus"],
                            "removed_block": nearest["asset_id"]})
    return plan, tap_result, repairs


def _solve_with_res_cap(spec: SystemSpec, base: Mapping[str, Any], scenario: Scenario,
                        slug: str) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    import dataclasses

    requested = float(scenario.res_share)
    log: list[dict[str, Any]] = []
    first_fail: dict[str, Any] | None = None
    feasible: dict[float, tuple] = {}
    infeasible: set[float] = set()

    def attempt(share: float) -> tuple | None:
        share = round(share, 2)
        sc = dataclasses.replace(scenario, res_share=share)
        try:
            out = _solve_scenario(spec, base, sc, slug)
        except ScenarioInfeasibleError as exc:
            d = exc.details
            log.append({"res_share": share, "feasible": False, "must_run_mw": d["must_run_total_mw"],
                        "required_mw": d["sync_needed_mw"], "losses_mw": d["losses_mw"],
                        "balance_share": d["max_feasible_res_share"]})
            infeasible.add(share)
            nonlocal first_fail
            first_fail = first_fail or d
            return None
        log.append({"res_share": share, "feasible": True})
        feasible[share] = out
        return out

    share = requested
    out = attempt(share)
    while out is None:
        est = math.floor(log[-1]["balance_share"] * 100.0 + 1e-9) / 100.0
        share = round(min(est, share - 0.01), 2)
        if share < 0.01:
            d = first_fail or {}
            raise ScenarioInfeasibleError(
                f"{slug}: no RES share >= 1 % is compatible with the must-run of the committed units", d)
        out = attempt(share)
    if share < requested - 1e-12:
        while round(share + 0.01, 2) <= requested + 1e-12 and round(share + 0.01, 2) not in infeasible:
            up = attempt(share + 0.01)
            if up is None:
                break
            share, out = round(share + 0.01, 2), up
    plan, tap_result, repairs = out
    capped = share < requested - 1e-12
    reason = "requested share feasible"
    if capped and first_fail is not None:
        reason = (f"capped by must-run: {first_fail['must_run_total_mw']:.0f} MW committed minimum vs "
                  f"{first_fail['sync_needed_mw']:.0f} MW required at the requested share {requested:.2f} "
                  f"(scaled load {first_fail['scaled_load_mw']:.0f} MW - RES {first_fail['res_mw']:.0f} MW + losses "
                  f"{first_fail['losses_mw']:.0f} MW); {first_fail['n_units_governor_minimum']} of "
                  f"{first_fail['n_units_online']} units on the governor-minimum basis")
    plan["scenario"].update({"res_share": share, "res_share_requested": requested, "res_share_capped": capped,
                             "res_share_reason": reason, "res_share_search": log})
    if capped and first_fail is not None:
        plan["scenario"]["res_share_cap_detail"] = first_fail
    return plan, tap_result, repairs


def build_benchmark(slug: str, *, variant: str = "baseline") -> BenchmarkCase:
    if slug not in SYSTEMS:
        raise BenchmarkError(f"unknown benchmark {slug!r}; expected one of {BENCHMARK_SLUGS}")
    if variant not in VARIANTS:
        raise BenchmarkError(f"unknown variant {variant!r}; expected one of {VARIANTS}")
    spec = SYSTEMS[slug]
    t_start = time.perf_counter()
    base = _base_analysis(spec)
    t_base = time.perf_counter()
    scenario = scenario_of(variant)
    if scenario.dispatch_rule == "proportional_must_run":
        plan, tap_result, repairs = _solve_with_res_cap(spec, base, scenario, slug)
    else:
        plan, tap_result, repairs = _solve_scenario(spec, base, scenario, slug)
    t_taps = time.perf_counter()
    plan["placement_repairs"] = repairs
    if (plan.get("uel") or {}).get("applied"):
        plan["_uel_reference"] = _uel_reference(plan)
    if variant == "preserved_absorption":
        plan["collector_reactors"] = [
            {"idx": f"COLR_{blk['asset_id']}", "block": blk["asset_id"], "bus": blk["collector_bus"],
             "b_sys": blk["q_pf_pu"] / blk["z0_pu"] ** 2}
            for blk in plan["blocks"]
        ]
    frozen_plan = copy.deepcopy(plan)

    def factory(tds_config: Mapping[str, Any] | None = None) -> Any:
        return _build_system(frozen_plan, variant, tds_config=tds_config)

    ss = factory()
    t_final = time.perf_counter()

    dwell = RELAY_DWELL_DELAYED_S if variant == "delayed_protection" else RELAY_DWELL_S
    relays_disabled = variant == "no_ov_protection"
    sb = plan["mva"]
    blocks: list[ProtectedBlock] = []
    hidden: list[HiddenOutputData] = []
    addresses: list[int] = []
    block_rows = []
    for blk in plan["blocks"]:
        cpos = _pos(ss.Bus, blk["collector_bus"])
        hpos = _pos(ss.Bus, blk["hv_bus"])
        ppos = _pos(ss.PVD1, blk["pvd1_idx"])
        lpos = _pos(ss.Line, blk["gsu_idx"])
        v_col = float(ss.dae.y[ss.Bus.v.a[cpos]])
        v_hv = float(ss.dae.y[ss.Bus.v.a[hpos]])
        pref0 = float(ss.PVD1.pref0.v[ppos])
        qref0 = float(ss.PVD1.Qsum.v[ppos])
        r_sys = float(ss.Line.r.v[lpos])
        x_sys = float(ss.Line.x.v[lpos])
        tap = float(ss.Line.tap.v[lpos])
        zabs = math.hypot(r_sys, x_sys)
        s0 = math.hypot(pref0, qref0)
        ilim = float(ss.PVD1.ialim.v[ppos])
        addresses.append(int(ss.Bus.v.a[cpos]))
        hidden.append(HiddenOutputData(
            asset_id=blk["asset_id"], collector_bus=blk["collector_bus"], hv_bus=blk["hv_bus"],
            collector_v_address=int(ss.Bus.v.a[cpos]), hv_v_address=int(ss.Bus.v.a[hpos]),
            hv_a_address=int(ss.Bus.a.a[hpos]), tap=tap, gsu_r_pu_sysbase=r_sys, gsu_x_pu_sysbase=x_sys,
            gsu_z_abs_pu_sysbase=zabs, v_hv_op_pu=v_hv, z0_pu=v_col, drop_op_pu=v_col - v_hv / tap,
            drop_bound_op_pu=zabs * s0 / v_col, current_limit_pu_sysbase=ilim, drop_bound_ilim_pu=zabs * ilim,
        ))
        meta = {
            "cluster": blk["cluster"], "hv_vn_kv": blk["hv_vn_kv"], "collector_kv": COLLECTOR_KV,
            "pv_placeholder": blk["pv_idx"], "relays_disabled": relays_disabled, "variant": variant,
            "margin_pu": RELAY_PICKUP_PU - v_col, "q_target_mvar": blk["q_target_mvar"],
        }
        if variant == "preserved_absorption":
            meta["collector_reactor"] = f"COLR_{blk['asset_id']}"
        blocks.append(ProtectedBlock(
            asset_id=blk["asset_id"], collector_bus=blk["collector_bus"], hv_bus=blk["hv_bus"],
            pvd1_idx=blk["pvd1_idx"], gsu_line_idx=blk["gsu_idx"], tap=tap, gsu_x_pu_sysbase=x_sys,
            p_mw=pref0 * sb, q_mvar=qref0 * sb, sn_mva=blk["sn_mva"], pickup_pu=RELAY_PICKUP_PU, dwell_s=dwell,
            z0_pu=v_col, trip_event_id=f"plant_trip:{blk['asset_id']}", metadata=meta,
        ))
        block_rows.append({
            **{k: blk[k] for k in ("asset_id", "cluster", "rank_in_cluster", "seed_bus", "hv_bus", "hv_vn_kv", "hv_area",
                                   "collector_bus", "gsu_idx", "pv_idx", "pvd1_idx", "p_mw", "sn_mva", "q_target_mvar",
                                   "gsu_x_device_pu", "gsu_r_device_pu", "gsu_x_pu_sysbase", "gsu_r_pu_sysbase",
                                   "d_to_seed_pu", "d_to_slack_pu", "tap_initial_guess", "s_sc_mva", "scr")},
            "tap": tap,
            "tap_within_range": scenario.tap_range[0] - 1e-12 <= tap <= scenario.tap_range[1] + 1e-12,
            "z0_design_pu": blk["z0_design_pu"],
            "z0_op_pu": v_col,
            "tap_at_limit": blk["tap_at_limit"],
            "margin_pu": RELAY_PICKUP_PU - v_col,
            "v_hv_op_pu": v_hv,
            "p_op_mw": pref0 * sb,
            "q_op_mvar": qref0 * sb,
            "q_rel_error": abs(qref0 - blk["q_target_pu"]) / abs(blk["q_target_pu"]),
            "pf_op": pref0 / s0,
            "drop_op_pu": v_col - v_hv / tap,
            "drop_bound_op_pu": zabs * s0 / v_col,
            "drop_bound_ilim_pu": zabs * ilim,
        })

    machines = _machine_op_table(ss, base)
    dist = _ElectricalDistance(base)
    events: list[EventSpec] = []
    events += _plant_events(ss, plan, variant)
    events += _load_events(ss, plan, base)
    gen_events, tripped = _generator_events(ss, machines, plan)
    events += gen_events
    redispatch, redispatch_info = _redispatch_events(ss, machines, plan, dist)
    events += redispatch
    events += _device_events(ss, plan)
    controls, control_notes = _controls(ss, plan, machines)

    runtime = time.perf_counter() - t_start
    design = _design_dict(spec, variant, base, plan, tap_result, ss, block_rows, hidden, events, controls,
                          machines, tripped, redispatch_info, control_notes, dwell, relays_disabled,
                          {"base_analysis_s": t_base - t_start, "tap_solve_s": t_taps - t_base,
                           "final_build_s": t_final - t_taps, "total_s": runtime})
    return BenchmarkCase(
        slug=slug, variant=variant, system=ss, factory=factory, blocks=tuple(blocks), events=tuple(events),
        controls=tuple(controls), design=design, protected_output_addresses=tuple(addresses),
        hidden_output_data=tuple(hidden), build_runtime_s=runtime,
    )


def _mutation_rows(ss: Any, mutations: Sequence[Mutation]) -> list[dict[str, Any]]:
    rows = []
    for m in mutations:
        model = getattr(ss, m.model)
        base = float(getattr(model, m.attr).v[_pos(model, m.device_idx)])
        rows.append({"model": m.model, "idx": m.device_idx, "attr": m.attr, "base": base, "value": m.value})
    return rows


def _profile_row(profile: TimeProfile) -> dict[str, Any]:
    row = {"kind": str(profile.kind), "start_s": profile.start_s, "duration_s": profile.duration_s,
           "description": profile.description}
    if profile.samples:
        row["n_samples"] = len(profile.samples)
        row["horizon_s"] = profile.samples[-1][0]
        row["final_value"] = profile.samples[-1][1]
    return row


def _dispatch_summary(plan: Mapping[str, Any], ss: Any) -> dict[str, Any]:
    sb = plan["mva"]
    d = plan["dispatch"]
    units = d.get("units", [])
    out = {
        "rule": d["rule"],
        "load_stock_mw": plan["load_p_stock_pu"] * sb,
        "load_scale": plan["scenario"]["load_scale"],
        "load_p_mw": plan["load_p_pu"] * sb,
        "load_p_gross_mw": plan["load_p_gross_pu"] * sb,
        "res_p_total_mw": plan["res_p_total_pu"] * sb,
        "res_share_of_load": plan["res_p_total_pu"] / plan["load_p_pu"],
        "dispatch_scale": d["k"],
        "target_sync_mw": d["target_sync_pu"] * sb,
        "stock_losses_mw": plan["stock_losses_pu"] * sb,
        "sync_pv_p0_total_stock_mw": plan["pv_p0_total_pu"] * sb,
        "sync_pv_p0_total_new_mw": sum(d["pv_p0"].values()) * sb,
        "slack_scheduled_mw": d["slack_scheduled_pu"] * sb,
        "slack_pf_mw": float(ss.Slack.p.v[0]) * sb,
        "loss_corrected": bool(d.get("loss_corrected", False)),
        "feasible": bool(d.get("feasible", True)),
    }
    if units:
        out["n_units"] = len(units)
        out["n_at_must_run"] = sum(bool(u["at_must_run"]) for u in units)
        out["n_must_run_governor_basis"] = sum(u["must_run_basis"] == "governor VMIN" for u in units)
        out["n_fixed_nonpositive"] = sum(u["p0_pu"] <= 0.0 for u in units)
        out["must_run_total_mw"] = d["must_run_total_pu"] * sb
        out["units"] = [{k: (v * sb if (k.endswith("_pu") and isinstance(v, float)) else v) for k, v in u.items()}
                        for u in units]
        out["units_unit_note"] = "all *_pu fields of 'units' are converted to MW (system base 100 MVA)"
        out["rule_text"] = ("one common factor k for every synchronous unit (PV and slack): P_i = max(k P_i,stock, "
                            "must-run_i); must-run_i = max(20 % of capability_i, governor VMIN_i), capped at the stock "
                            "output; capability = min(PV.pmax if < 99 pu, MVA rating if >= stock output); units with "
                            "stock output <= 0 unchanged; k solved so that the slack produces exactly its own share "
                            "(loss-corrected on the final power flow); voltage set points unchanged, all units online")
    else:
        out["rule_text"] = ("all in-service synchronous PV p0 multiplied by (sum p0 - P_RES)/sum p0 so the renewable "
                            "in-feed displaces synchronous generation and the slack keeps (approximately) its base output")
    return out


def _uel_summary(plan: Mapping[str, Any], ss: Any) -> dict[str, Any]:
    uel = plan.get("uel") or {}
    if not uel.get("applied"):
        return {"applied": False}
    sb = plan["mva"]
    ref = plan.get("_uel_reference") or {}
    rows = []
    bus_v = {str(b): float(v) for b, v in zip(ss.Bus.idx.v, ss.Bus.v.v)}
    for key, d in uel["devices"].items():
        q = float(ss.PV.q.v[_pos(ss.PV, d["pv"])])
        at_limit = abs(q - d["qlim_pu"]) < UEL_Q_TOL_PU
        v = bus_v[str(d["bus"])]
        rows.append({"pv": d["pv"], "bus": d["bus"], "machines": d["machines"], "sn_mva": d["sn_mva"],
                     "q_limit_mvar": d["qlim_pu"] * sb, "q_without_uel_mvar": ref.get("q", {}).get(key, float("nan")) * sb,
                     "q_final_mvar": q * sb, "v0_setpoint_pu": d["v0_orig"], "v_bus_pu": v,
                     "fixed_q_at_limit": bool(at_limit), "violation_mvar": max(0.0, (d["qlim_pu"] - q) * sb),
                     "spurious_conversion": bool(at_limit and v < d["v0_orig"] - 1e-6)})
    return {"applied": True, "rule": uel["rule"], "q_over_sn": uel["q_over_sn"], "p_max_mw": uel["p_max_mw"],
            "n_condensers": len(rows), "n_at_limit": sum(r["fixed_q_at_limit"] for r in rows),
            "n_below_limit_without_uel": sum(r["q_without_uel_mvar"] < r["q_limit_mvar"] - 1e-4 for r in rows),
            "max_violation_mvar": max((r["violation_mvar"] for r in rows), default=0.0),
            "n_spurious_conversions": sum(r["spurious_conversion"] for r in rows),
            "reference_without_uel": {k: v for k, v in ref.items() if k != "q"},
            "devices": rows}


def _internal_angles(ss: Any) -> dict[str, Any]:
    rows = []
    bpos = {str(b): i for i, b in enumerate(ss.Bus.idx.v)}
    for name, model in ss.groups["SynGen"].models.items():
        for i in range(model.n):
            if float(model.u.v[i]) == 0.0:
                continue
            ang = math.degrees(float(model.delta.v[i]) - float(ss.Bus.a.v[bpos[str(model.bus.v[i])]]))
            rows.append((abs(ang), ang, name, model.idx.v[i]))
    if not rows:
        return {}
    a = np.array([r[0] for r in rows])
    top = sorted(rows, key=lambda r: -r[0])[:5]
    return {"internal_angle_max_deg": float(a.max()), "internal_angle_p50_deg": float(np.median(a)),
            "internal_angle_p90_deg": float(np.quantile(a, 0.9)), "internal_angle_p99_deg": float(np.quantile(a, 0.99)),
            "n_internal_angle_over_60_deg": int(np.sum(a > 60.0)), "n_internal_angle_over_90_deg": int(np.sum(a > 90.0)),
            "internal_angle_top": [{"model": r[2], "idx": r[3], "angle_deg": r[1]} for r in top]}


def _low_q_machines(ss: Any, sb: float) -> list[dict[str, Any]]:
    out = []
    for name, model in ss.groups["SynGen"].models.items():
        for i in range(model.n):
            if float(model.u.v[i]) == 0.0:
                continue
            sn = float(model.Sn.v[i])
            q = float(model.Qe.v[i]) * sb
            if q < -0.5 * sn:
                out.append({"model": name, "idx": model.idx.v[i], "bus": model.bus.v[i], "sn_mva": sn,
                            "sn_is_default_100": abs(sn - 100.0) < 1e-9, "p_mw": float(model.tm0.v[i]) * sb,
                            "q_mvar": q, "q_over_sn": q / sn, "vf_pu": float(model.vf.v[i])})
    return out


def _voltage_profile(ss: Any, plan: Mapping[str, Any], machines: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    sb = plan["mva"]
    col = {str(b["collector_bus"]) for b in plan["blocks"]}
    v = np.asarray(ss.Bus.v.v, dtype=float)
    vn = np.asarray(ss.Bus.Vn.v, dtype=float)
    ids = [str(b) for b in ss.Bus.idx.v]
    net = np.array([i not in col for i in ids])
    hv = net & (vn >= HV_MIN_KV)
    on = [m for m in machines if m["u"] != 0.0]
    q = np.array([m["q_op_pu"] for m in on]) if on else np.zeros(0)
    rated = [m for m in on if abs(m["sn_mva"] - 100.0) > 1e-9 and m["sn_mva"] / sb >= abs(m.get("p_base_pu", 0.0)) - 1e-9]
    qsn = np.array([m["q_op_pu"] * sb / m["sn_mva"] for m in rated]) if rated else np.zeros(0)
    return {
        "bus_v_min_pu": float(v[net].min()), "bus_v_max_pu": float(v[net].max()),
        "bus_v_min_bus": ids[int(np.flatnonzero(net)[np.argmin(v[net])])],
        "bus_v_max_bus": ids[int(np.flatnonzero(net)[np.argmax(v[net])])],
        "hv_v_min_pu": float(v[hv].min()) if hv.any() else None,
        "hv_v_max_pu": float(v[hv].max()) if hv.any() else None,
        "hv_v_mean_pu": float(v[hv].mean()) if hv.any() else None,
        "n_bus_over_1p10": int(np.sum(net & (v > 1.10))),
        "n_hv_bus_over_1p10": int(np.sum(hv & (v > 1.10))),
        "kv_levels_over_1p10": sorted({float(x) for x in vn[net & (v > 1.10)]}),
        "block_hv_v_pu": {str(b["asset_id"]): float(v[ids.index(str(b["hv_bus"]))]) for b in plan["blocks"]},
        "collector_v_pu": {str(b["asset_id"]): float(v[ids.index(str(b["collector_bus"]))]) for b in plan["blocks"]},
        "n_machines": len(on),
        "n_machines_absorbing_q": int(np.sum(q < 0.0)),
        "machine_q_total_mvar": float(q.sum() * sb),
        "machine_q_min_mvar": float(q.min() * sb) if q.size else None,
        "machine_q_max_mvar": float(q.max() * sb) if q.size else None,
        "machine_q_over_sn_min": float(qsn.min()) if qsn.size else None,
        "machine_q_over_sn_max": float(qsn.max()) if qsn.size else None,
        "n_machines_rated": len(rated),
        "machines_q_below_minus_half_sn": _low_q_machines(ss, sb),
        **_internal_angles(ss),
        "min_machine_vf_pu": float(min(float(v) for m in ss.groups["SynGen"].models.values() if m.n
                                       for v, u in zip(m.vf.v, m.u.v) if float(u) != 0.0)),
        "machines_absorbing": [str(m["idx"]) for m in on if m["q_op_pu"] < 0.0],
    }


def _design_dict(spec, variant, base, plan, tap_result, ss, block_rows, hidden, events, controls, machines,
                 tripped, redispatch_info, control_notes, dwell, relays_disabled, runtime) -> dict[str, Any]:
    sb = plan["mva"]
    slack = {
        "idx": ss.Slack.idx.v[0], "bus": ss.Slack.bus.v[0],
        "p_mw": float(ss.Slack.p.v[0]) * sb, "q_mvar": float(ss.Slack.q.v[0]) * sb,
        "pmax_mw": float(ss.Slack.pmax.v[0]) * sb, "pmin_mw": float(ss.Slack.pmin.v[0]) * sb,
        "qmax_mvar": float(ss.Slack.qmax.v[0]) * sb, "qmin_mvar": float(ss.Slack.qmin.v[0]) * sb,
        "base_case_p_mw": base["slack"]["p_pu"] * sb, "base_case_q_mvar": base["slack"]["q_pu"] * sb,
    }
    slack["p_within_limits"] = slack["pmin_mw"] - 1e-6 <= slack["p_mw"] <= slack["pmax_mw"] + 1e-6
    slack["q_within_limits"] = slack["qmin_mvar"] - 1e-6 <= slack["q_mvar"] <= slack["qmax_mvar"] + 1e-6
    variant_info = {
        "name": variant,
        "relays_disabled": relays_disabled,
        "relay_pickup_pu": RELAY_PICKUP_PU,
        "relay_dwell_s": dwell,
        "plant_control": "voltage droop (over-voltage branch)" if variant == "voltage_mode_res" else "fixed power factor",
        "plant_trip_preserves_absorption": variant == "preserved_absorption",
        "description": {
            "baseline": "light-load high-voltage scenario, fixed-PF plants, relay 1.10 pu / 50 ms",
            "stock_load": "previous stock-load design (reference only): stock load, RES 35 %, taps [0.85, 1.15]",
            "voltage_mode_res": "PVD1 proportional over-voltage volt-var droop active at the operating point (no "
                                "dead band): v1 = z0 - 1e-4 pu, dqdv = -Q0/(2W+d) (W = 0.05), qmn = qref0/2, "
                                "qref0 = Q0 + dqdv d (gammaq); initial Q and operating point unchanged",
            "preserved_absorption": "plant_trip also switches in a collector reactor b = Q0/z0^2 (absorbs Q0 at z0)",
            "delayed_protection": "relay dwell 0.20 s",
            "no_ov_protection": "relays disabled (blocks flagged relays_disabled)",
        }[variant],
    }
    pvd1_rows = []
    for blk in plan["blocks"]:
        p = _pos(ss.PVD1, blk["pvd1_idx"])
        row = {"asset_id": blk["asset_id"]}
        for name in ("Sn", "ialim", "qmx", "qmn", "dqdv", "v0", "v1", "vt0", "vt1", "vt2", "vt3", "ft0", "ft1", "ft2",
                     "ft3", "recflag", "pqflag", "tip", "tiq", "fn"):
            par = getattr(ss.PVD1, name)
            row[name + "_input"] = float(par.vin[p]) if getattr(par, "vin", None) is not None else float(par.v[p])
            row[name + "_sysbase"] = float(par.v[p])
        row["Vqu"] = float(ss.PVD1.Vqu.v[p])
        row["Vql"] = float(ss.PVD1.Vql.v[p])
        row["pref0"] = float(ss.PVD1.pref0.v[p])
        row["qref0"] = float(ss.PVD1.qref0.v[p])
        pvd1_rows.append(row)
    ev_rows = [
        {"event_id": e.event_id, "family": e.family, "kind": e.kind, "description": e.description,
         "profile": _profile_row(e.profile), "mutations": _mutation_rows(ss, e.mutations),
         "disconnects_assets": list(e.disconnects_assets), "metadata": dict(e.metadata)}
        for e in events
    ]
    ctl_rows = [
        {"control_id": c.control_id, "control_class": c.control_class, "description": c.description,
         "alpha_max": c.alpha_max, "mvar_per_unit": c.mvar_per_unit, "envelope": _profile_row(c.envelope),
         "mutations": _mutation_rows(ss, c.mutations), "metadata": dict(c.metadata)}
        for c in controls
    ]
    fam_counts = {f: sum(1 for e in events if e.family == f) for f in EVENT_FAMILIES}
    ctl_counts = {c: sum(1 for k in controls if k.control_class == c) for c in CONTROL_CLASSES}
    design = {
        "schema": "pa_dvsa.benchmarks/case_design/v1",
        "slug": spec.slug,
        "label": spec.label,
        "case": spec.case,
        "addfile": spec.addfile,
        "variant": variant_info,
        "system_mva": sb,
        "system_freq_hz": plan["freq"],
        "preparation": base["prep"],
        "constants": {
            "res_share_of_load": plan["scenario"]["res_share"], "load_scale": plan["scenario"]["load_scale"],
            "block_p_spread": BLOCK_P_SPREAD, "plant_pf": PLANT_PF, "must_run_fraction": MUST_RUN_FRACTION,
            "kappa_q_over_p": KAPPA_Q_OVER_P, "plant_sn_over_p": PLANT_SN_OVER_P, "gsu_x_device_pu": GSU_X_DEVICE_PU,
            "gsu_x_over_r": GSU_X_OVER_R, "collector_kv": COLLECTOR_KV, "z0_range_pu": list(Z0_RANGE_PU),
            "tap_range": list(plan["scenario"]["tap_range"]), "q_design_rel_tol": Q_DESIGN_REL_TOL,
            "scr_min": SCR_MIN, "plant_spacing_pu": PLANT_SPACING_PU, "strong_quantile": STRONG_QUANTILE,
            "relay_pickup_pu": RELAY_PICKUP_PU, "hv_min_kv": HV_MIN_KV, "n_clusters": spec.n_clusters,
            "pvd1_ialim": PVD1_IALIM, "pvd1_qlim_device": PVD1_QLIM_DEVICE, "pvd1_fixed_pf": PVD1_FIXED_PF,
            "pvd1_trip_disabled": PVD1_TRIP_DISABLED, "voltage_mode_breakpoint_offset_pu": VOLTAGE_MODE_BREAKPOINT_OFFSET_PU,
            "voltage_mode_saturation_dv_pu": VOLTAGE_MODE_SAT_DV_PU, "p_permutation_alpha": _GOLDEN,
            "z0_permutation_alpha": _SILVER, "tds_defaults": TDS_DEFAULTS,
        },
        "scenario": plan["scenario"],
        "uel": _uel_summary(plan, ss),
        "dispatch": _dispatch_summary(plan, ss),
        "voltage_profile": _voltage_profile(ss, plan, machines),
        "slack": slack,
        "base_case": {"v_range_pu": base["v_range"], "device_counts": base["device_counts"]},
        "operating_point": {
            "v_min_pu": float(np.min(ss.Bus.v.v)), "v_max_pu": float(np.max(ss.Bus.v.v)),
            "v_max_bus": ss.Bus.idx.v[int(np.argmax(ss.Bus.v.v))],
            "v_max_non_collector_pu": float(max(ss.Bus.v.v[i] for i in range(ss.Bus.n)
                                                if str(ss.Bus.idx.v[i]) not in {str(b["collector_bus"]) for b in plan["blocks"]})),
            "pflow_converged": bool(ss.PFlow.converged),
            "tds_initialized": bool(ss.TDS.initialized),
            "tds_init_test_ok": bool(getattr(ss.TDS, "test_ok", False)),
            "max_abs_fg_at_init": float(np.max(np.abs(ss.dae.fg))) if ss.dae.fg.size else 0.0,
            "controllers_at_limit": controllers_at_limit(ss),
        },
        "placement": {
            "rule": "candidates: in-service buses with Vn >= 200 kV, no static or synchronous generator, in the "
                    "slack's island, short-circuit ratio S_sc/P_max >= 3 (S_sc: reactance network, machines through "
                    "x''d; P_max = largest block). Strong set S: S_sc >= median over candidates. Hubs: strongest bus "
                    "of S, then farthest-point (max-min effective reactance) sampling within S. Members: clusters take "
                    "turns adding the nearest unassigned bus to their hub, first from S then from all candidates, "
                    "skipping buses within 0.005 pu effective reactance of a placed plant (one plant per substation). "
                    "Ties by bus order.",
            "strong_quantile": STRONG_QUANTILE,
            "plant_spacing_pu": PLANT_SPACING_PU,
            "repairs": plan.get("placement_repairs", []),
            "n_candidates": len(plan["candidates"]),
            "scr_min": SCR_MIN,
            "p_max_block_mw": plan["p_max_block_pu"] * sb,
            "excluded": plan["excluded_candidates"],
            "candidates": plan["candidates"],
            "cluster_seeds": [{"cluster": c, "seed_bus": b} for c, b in plan["cluster_seeds"]],
        },
        "tap_solve": {k: tap_result[k] for k in ("iterations", "converged", "history", "tap_at_limit")},
        "blocks": block_rows,
        "hidden_outputs": [
            {f.name: getattr(h, f.name) for f in fields(HiddenOutputData)}
            for h in hidden
        ],
        "pvd1_parameters": pvd1_rows,
        "added_devices": {
            "shunt_reactors_in_service": plan["reactors"],
            "capacitors_out_of_service": plan["capacitors"],
            "parallel_circuits_out_of_service": plan["parallel_lines"],
            "statcom_shunts_b0": plan["statcoms"],
            "fast_reactor_shunts_b0": plan["fast_reactors"],
            "manual_reactor_shunts_b0": plan["manual_reactors"],
            "collector_reactors_out_of_service": plan["collector_reactors"],
        },
        "machines": [
            {k: m[k] for k in ("model", "idx", "bus", "gen", "sn_mva", "u", "p_base_pu", "p_op_pu", "q_op_pu", "area",
                               "slack_linked")}
            | {"exciters": [e["idx"] for e in m["exciters"]], "governors": [g["idx"] for g in m["governors"]],
               "pss": [p["idx"] for p in m["pss"]]}
            for m in machines
        ],
        "generator_trip_rule": "largest in-service synchronous machines by MVA rating (ties: |P|, idx) not linked to "
                               "the slack; trip sets u=0 on the machine and every linked exciter/governor/PSS",
        "generator_trip_machines": [m["idx"] for m in tripped],
        "load_disconnection_rule": "largest |q0| in-service PQ loads (n_blocks/2), signed; q<0 loads flagged",
        "redispatch": redispatch_info,
        "control_notes": control_notes,
        "event_family_counts": fam_counts,
        "control_class_counts": ctl_counts,
        "events": ev_rows,
        "controls": ctl_rows,
        "device_counts": {"augmented": device_counts(ss), "base_case": base["device_counts"]},
        "protected_output_addresses": [h.collector_v_address for h in hidden],
        "runtime_s": runtime,
    }
    return _jsonable(design)


def write_design(case: BenchmarkCase, root: str | Path | None = None) -> Path:
    base_dir = Path(root) if root is not None else _REPO_ROOT / "results" / "revised"
    out_dir = base_dir / case.slug
    out_dir.mkdir(parents=True, exist_ok=True)
    name = "case_design.json" if case.variant == "baseline" else f"case_design_{case.variant}.json"
    path = out_dir / name
    path.write_text(json.dumps(case.design, indent=2, sort_keys=False), encoding="utf-8")
    return path


def augmented_algebraic_matrix(ss: Any) -> tuple[sparse.csc_matrix, dict[str, int]]:
    fx, fy, gx, gy = (_to_csc(getattr(ss.dae, name)) for name in ("fx", "fy", "gx", "gy"))
    tf = np.asarray(ss.dae.Tf, dtype=float).reshape(-1)
    zero = np.flatnonzero(tf == 0.0)
    if zero.size:
        aug = sparse.bmat([[fx[zero][:, zero], fy[zero]], [gx[:, zero], gy]]).tocsc()
    else:
        aug = gy
    return aug, {"n_states": int(fx.shape[0]), "n_algebraic": int(gy.shape[0]), "n_zero_tf_states": int(zero.size)}


def static_validation(case: BenchmarkCase, *, dense_rank_limit: int = 3000) -> dict[str, Any]:
    from .andes_regularization import augmented_algebraic_rank

    ss = case.system
    out: dict[str, Any] = {"slug": case.slug, "variant": case.variant}
    rows = case.design["blocks"]
    out["pflow_converged"] = bool(ss.PFlow.converged)
    out["tds_init_test_ok"] = bool(getattr(ss.TDS, "test_ok", False))
    out["max_abs_fg_at_init"] = float(np.max(np.abs(ss.dae.fg)))
    out["taps"] = [r["tap"] for r in rows]
    lo, hi = case.design["scenario"]["tap_range"]
    out["taps_within_range"] = all(lo - 1e-12 <= r["tap"] <= hi + 1e-12 for r in rows)
    out["max_q_rel_error"] = max(r["q_rel_error"] for r in rows)
    out["q_within_1pct"] = out["max_q_rel_error"] < Q_DESIGN_REL_TOL
    free = [r for r in rows if not r["tap_at_limit"]]
    out["n_taps_at_limit"] = len(rows) - len(free)
    out["max_z0_error_pu"] = max((abs(r["z0_op_pu"] - r["z0_design_pu"]) for r in free), default=0.0)
    out["z0_shortfall_at_limit_pu"] = [r["z0_design_pu"] - r["z0_op_pu"] for r in rows if r["tap_at_limit"]]
    out["margins_pu"] = [r["margin_pu"] for r in rows]
    aug, dims = augmented_algebraic_matrix(ss)
    out.update(dims)
    size = aug.shape[0]
    out["augmented_size"] = int(size)
    off = aug - sparse.diags(aug.diagonal())
    off.eliminate_zeros()
    row_nnz = np.diff(off.tocsr().indptr)
    col_nnz = np.diff(off.tocsc().indptr)
    decoupled = (row_nnz == 0) & (col_nnz == 0)
    keep = np.flatnonzero(~decoupled)
    out["n_decoupled"] = int(decoupled.sum())
    names = list(getattr(ss.dae, "y_name", []) or [])
    n_z = dims["n_zero_tf_states"]
    out["decoupled_examples"] = [names[i - n_z] if n_z <= i < n_z + len(names) else int(i)
                                 for i in np.flatnonzero(decoupled)[:8]]
    coupled = aug[keep][:, keep].tocsc()
    out["coupled_size"] = int(coupled.shape[0])
    if size <= dense_rank_limit:
        fx, fy, gx, gy = (_to_csc(getattr(ss.dae, name)) for name in ("fx", "fy", "gx", "gy"))
        n, rank = augmented_algebraic_rank(fx, fy, gx, gy, ss.dae.Tf)
        sv = np.linalg.svd(coupled.toarray(), compute_uv=False)
        out["rank_method"] = "dense"
        out["augmented_rank"] = int(rank)
        out["full_rank"] = bool(rank == n)
        out["coupled_sigma_min"] = float(sv[-1])
        out["coupled_sigma_max"] = float(sv[0])
        out["coupled_cond_2"] = float(sv[0] / sv[-1]) if sv[-1] > 0 else float("inf")
    else:
        out["rank_method"] = "sparse_lu"
        aug = coupled
        try:
            lu = spla.splu(aug.tocsc())
            udiag = np.abs(lu.U.diagonal())
            inv_op = spla.LinearOperator(aug.shape, matvec=lu.solve, rmatvec=lambda v: lu.solve(v, trans="T"))
            norm1 = float(spla.norm(aug, 1))
            inv_norm1 = float(spla.onenormest(inv_op))
            out["lu_success"] = True
            out["u_diag_min"] = float(udiag.min())
            out["u_diag_max"] = float(udiag.max())
            out["cond_1_estimate"] = norm1 * inv_norm1
            out["full_rank"] = bool(np.all(np.isfinite(udiag)) and udiag.min() > 0.0
                                    and np.isfinite(out["cond_1_estimate"]) and out["cond_1_estimate"] < 1e15)
        except RuntimeError as exc:
            out["lu_success"] = False
            out["lu_error"] = str(exc)
            out["full_rank"] = False
    try:
        from .andes_regularization import algebraic_regularity_report

        probe = case.factory()
        with _quiet():
            rep = algebraic_regularity_report(probe)
        out["core_regularity_ok"] = bool(rep.get("ok"))
        out["core_regularity"] = {k: rep.get(k) for k in ("size", "condition_estimate", "removed_decoupled", "message")}
    except Exception as exc:
        out["core_regularity_ok"] = None
        out["core_regularity"] = {"message": f"not available: {exc!r}"}
    return _jsonable(out)
