"""Screening of the 2000-bus system and its validation against time-domain simulation: armed states, event
library, sparse engine, predictions and scoring.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field, replace
import hashlib
import json
import math
import os
import pickle
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
from scipy import sparse

from pa_dvsa.event_response import (
    EventResponse,
    EventResponseEngine,
    ModeReduction,
    OutputSelector,
    solve_frozen_algebraic,
)
from pa_dvsa.events import (
    EventSpec,
    Mutation,
    ProtectedBlock,
    apply_mutations,
    generator_trip_mutations,
    read_base_values,
    restore_mutations,
)
from pa_dvsa.finite_window import ModeDynamics, RigorousWindowMaps, assemble_window_maps, propagate_channel
from pa_dvsa.krylov_response import (
    KrylovModeDynamics,
    KrylovNotConverged,
    near_zero_eigenvalues_descriptor,
    propagate_channel_krylov,
    schur_products,
)
from pa_dvsa.linearization import (
    AlgebraicFactor,
    LinearDAEModel,
    ReducedLinearModel,
    ReductionDiagnostics,
)

from .activsg2000_engine import DEFAULT_OUT_ROOT, QuietAndes, ReplicaConfig, _pos

OUT = DEFAULT_OUT_ROOT / "screen"
SETTLE_S = 40.0
ENGINE_HORIZON_S = 2.0
DT_S = 0.005
EPSILON = 0.05
T_SEED_S = 0.5
TDS_TSTEP_S = 0.002
TDS_TOL = 1e-6
STABILITY_TOL = 1e-6
LEVELS = ("stock", "C2.0", "C1.5")
MARGINS = (0.01, 0.02)


@dataclass(frozen=True)
class RelaySetting:
    name: str
    pickup_pu: float
    dwell_s: float
    horizon_s: float
    description: str

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "pickup_pu": self.pickup_pu, "dwell_s": self.dwell_s,
                "horizon_s": self.horizon_s, "description": self.description}


RELAYS: dict[str, RelaySetting] = {
    "fast": RelaySetting("fast", 1.10, 0.05, 1.0, "benchmark fast relay: 1.10 pu, definite time 50 ms (primary)"),
    "ov1": RelaySetting("ov1", 1.10, 1.0, 2.0, "standard plant OV1: 1.10 pu / 1.0 s (NERC PRC-024-3)"),
    "ov2": RelaySetting("ov2", 1.20, 0.16, 1.0, "plant OV2: 1.20 pu / 0.16 s (IEEE 1547-2018)"),
}


def slug_of(level: str, margin: float) -> str:
    return f"{level}_m{margin:g}".replace(".", "p")


_DELAY_MEMORY_ATTRS = ("t", "_v_mem", "rewind", "_last_t", "_last_v")


def _is_delay(obj: Any) -> bool:
    try:
        from andes.core.discrete import Delay, Sampling
    except Exception:
        return False
    return isinstance(obj, (Delay, Sampling))


def _holders(ss: Any):
    seen: set[int] = set()
    for mname, model in ss.models.items():
        groups = [getattr(model, "params", {}), getattr(model, "discrete", {})]
        for key in ("services", "services_var", "services_var_seq", "services_var_nonseq", "services_post",
                    "services_subs", "services_icheck", "services_ref", "services_fnd", "services_ext",
                    "services_ops"):
            d = getattr(model, key, None)
            if d is not None:
                groups.append(d)
        for d in groups:
            for name, obj in d.items():
                if id(obj) in seen or not hasattr(obj, "__dict__"):
                    continue
                seen.add(id(obj))
                yield mname, name, obj


def snapshot_arrays(ss: Any) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for mname, name, obj in _holders(ss):
        delay = _is_delay(obj)
        for attr, val in vars(obj).items():
            if delay and attr in _DELAY_MEMORY_ATTRS:
                continue
            if isinstance(val, np.ndarray) and val.dtype.kind in "fcib":
                out[f"{mname}.{name}.{attr}"] = val.copy()
    out["dae.x"] = np.array(ss.dae.x, dtype=float).copy()
    out["dae.y"] = np.array(ss.dae.y, dtype=float).copy()
    return out


def restore_arrays(ss: Any, arrays: Mapping[str, np.ndarray]) -> int:
    written = 0
    present: set[str] = set()
    for mname, name, obj in _holders(ss):
        delay = _is_delay(obj)
        for attr, val in vars(obj).items():
            if delay and attr in _DELAY_MEMORY_ATTRS:
                continue
            key = f"{mname}.{name}.{attr}"
            if key not in arrays or not isinstance(val, np.ndarray):
                continue
            src = arrays[key]
            if val.shape != src.shape or val.dtype != src.dtype:
                raise RuntimeError(f"snapshot array {key} has shape/dtype {src.shape}/{src.dtype}, "
                                   f"system has {val.shape}/{val.dtype}")
            val[...] = src
            present.add(key)
            written += 1
    missing = [k for k in arrays if k not in present and not k.startswith("dae.")]
    if missing:
        raise RuntimeError(f"snapshot arrays missing in the target system: {missing[:10]}")
    ss.dae.x[:] = arrays["dae.x"]
    ss.dae.y[:] = arrays["dae.y"]
    ss.vars_to_models()
    return written


def arrays_digest(arrays: Mapping[str, np.ndarray]) -> str:
    h = hashlib.sha1()
    for k in sorted(arrays):
        v = np.ascontiguousarray(arrays[k])
        h.update(k.encode())
        h.update(str(v.dtype).encode())
        h.update(v.tobytes())
    return h.hexdigest()


def arrays_diff(a: Mapping[str, np.ndarray], b: Mapping[str, np.ndarray]) -> dict[str, float | str]:
    out: dict[str, float | str] = {}
    for k in sorted(set(a) | set(b)):
        if k not in a or k not in b:
            out[k] = "missing"
            continue
        x, y = np.asarray(a[k]), np.asarray(b[k])
        if x.shape != y.shape:
            out[k] = f"shape {x.shape} vs {y.shape}"
        elif not np.array_equal(x, y, equal_nan=(x.dtype.kind in "fc")):
            out[k] = float(np.max(np.abs(x.astype(complex) - y.astype(complex))))
    return out


@dataclass
class ArmedCase:
    level: str
    margin: float
    cfg: ReplicaConfig
    tap: float
    disabled_exciters: list[str]
    blocks: list[ProtectedBlock]
    events: list[EventSpec]
    library_corrections: list[dict[str, Any]]
    arrays: dict[str, np.ndarray]
    fingerprint: str
    metadata: dict[str, Any]
    build_s: float
    extra_events: list[EventSpec] = field(default_factory=list)

    @property
    def slug(self) -> str:
        return slug_of(self.level, self.margin)

    def event(self, event_id: str) -> EventSpec:
        for e in list(self.events) + list(self.extra_events):
            if e.event_id == event_id:
                return e
        raise KeyError(event_id)

    def no_margin_assets(self, pickup_pu: float = 1.10) -> list[str]:
        return [b.asset_id for b in self.blocks if not (pickup_pu - float(b.z0_pu) > 0.0)]


class SnapshotFactory:
    def __init__(self, case: ArmedCase) -> None:
        self.case = case
        self.last_build_s = float("nan")

    def __call__(self) -> Any:
        from .activsg2000_engine import build_replica_system

        t0 = time.perf_counter()
        c = self.case
        cfg = replace(c.cfg, plant_ov_enabled=False, tf_s=SETTLE_S, tap_override=float(c.tap))
        rs = build_replica_system(cfg, armed=False, quiet=True, disabled_exciters=list(c.disabled_exciters))
        ss = rs.system
        restore_arrays(ss, c.arrays)
        if float(ss.dae.t) != 0.0 or not ss.TDS.initialized:
            raise RuntimeError("factory system is not a fresh TDS-initialized system")
        self.last_build_s = time.perf_counter() - t0
        return ss


def _series_admittance(ss: Any, line_idx: Any) -> complex:
    p = _pos(ss.Line, line_idx)
    return 1.0 / complex(float(ss.Line.r.v[p]) + 1e-8, float(ss.Line.x.v[p]) + 1e-8)


def corrected_event_library(ss: Any, events: Sequence[EventSpec]) -> tuple[list[EventSpec], list[dict[str, Any]]]:
    out: list[EventSpec] = []
    notes: list[dict[str, Any]] = []
    for ev in events:
        if ev.family == "generator_trip":
            gid = next(m.device_idx for m in ev.mutations if m.model in ("GENROU", "GENCLS") and m.attr == "u")
            muts = generator_trip_mutations(ss, gid)
            old_dev = {(m.model, str(m.device_idx)) for m in ev.mutations}
            new_dev = {(m.model, str(m.device_idx)) for m in muts}
            if old_dev != new_dev:
                raise RuntimeError(f"{ev.event_id}: device sets differ ({sorted(old_dev)} vs {sorted(new_dev)})")
            notes.append({"event_id": ev.event_id, "correction": "controllers ue = 0 (events.generator_trip_mutations)",
                          "old": [[m.model, str(m.device_idx), m.attr, m.value] for m in ev.mutations],
                          "new": [[m.model, str(m.device_idx), m.attr, m.value] for m in muts]})
            out.append(replace(ev, mutations=tuple(muts), metadata={**dict(ev.metadata), "library_correction": "ue"}))
        elif ev.family == "line_energization":
            lid = next(m.device_idx for m in ev.mutations if m.model == "Line" and m.attr == "u")
            y = _series_admittance(ss, lid)
            muts = (Mutation("Line", lid, "u", 1.0), Mutation("Line", lid, "ghk", y.real), Mutation("Line", lid, "bhk", y.imag))
            p = _pos(ss.Line, lid)
            notes.append({"event_id": ev.event_id, "correction": "series admittance ghk/bhk set with u (ANDES yhk is a ConstService)",
                          "ghk_before": float(ss.Line.ghk.v[p]), "bhk_before": float(ss.Line.bhk.v[p]),
                          "ghk": y.real, "bhk": y.imag})
            out.append(replace(ev, mutations=muts, metadata={**dict(ev.metadata), "library_correction": "ghk_bhk"}))
        else:
            out.append(ev)
    return out, notes


def base_state_trip_event(ss: Any, blocks: Sequence[ProtectedBlock], assets: Sequence[str]) -> EventSpec | None:
    if not assets:
        return None
    by = {b.asset_id: b for b in blocks}
    muts = tuple(Mutation("PVD1", by[a].pvd1_idx, "u", 0.0) for a in assets)
    from pa_dvsa.data_model import ProfileKind, TimeProfile

    return EventSpec("base_state_trips:" + "+".join(assets), "base_state_trips", "switch", muts,
                     TimeProfile(ProfileKind.STEP),
                     description="joint trip of the assets that already sit at/above the relay pickup in the armed state",
                     disconnects_assets=tuple(assets))


def oa1_series_closure_mutations(ss: Any, mesh_line_ids: Sequence[str]) -> tuple[Mutation, ...]:
    muts: list[Mutation] = []
    for lid in mesh_line_ids:
        p = _pos(ss.Line, lid)
        if float(ss.Line.u.v[p]) > 0.5 and float(ss.Line.ghk.v[p]) == 0.0 and float(ss.Line.bhk.v[p]) == 0.0:
            y = _series_admittance(ss, lid)
            muts += [Mutation("Line", lid, "ghk", y.real), Mutation("Line", lid, "bhk", y.imag)]
    return tuple(muts)


def build_armed_case(level: str, margin: float, *, settle_s: float = SETTLE_S, verify: bool = True,
                     log: Callable[[str], None] | None = None) -> tuple[ArmedCase, Any, dict[str, Any]]:
    from .commitment_study import level_config

    return build_case_from_config(level_config(level, margin), level, margin, settle_s=settle_s, verify=verify, log=log)


def build_case_from_config(cfg: ReplicaConfig, level: str, margin: float, *, settle_s: float = SETTLE_S,
                           verify: bool = True, log: Callable[[str], None] | None = None,
                           operator_actions: bool = False) -> tuple[ArmedCase, Any, dict[str, Any]]:
    from .activsg2000_engine import build_armed_replica

    t0 = time.perf_counter()
    armed = build_armed_replica(cfg, method="dynamic_equilibrium", settle_s=settle_s, quiet=True)
    build_s = time.perf_counter() - t0
    ss = armed.system
    events, notes = corrected_event_library(ss, armed.events)
    if operator_actions:
        events = list(events) + operator_action_events(ss, list(armed.rs.mesh_line_ids))
    arrays = snapshot_arrays(ss)
    md = {k: v for k, v in armed.metadata.items() if k != "case_design"}
    md["mesh_line_ids"] = list(armed.rs.mesh_line_ids)
    md["candidate_line_ids"] = list(armed.rs.candidate_line_ids)
    md["candidate_cap_ids"] = list(armed.rs.candidate_cap_ids)
    md["arming_time_s"] = float(ss.dae.t)
    case = ArmedCase(level=level, margin=float(margin), cfg=cfg, tap=float(armed.rs.tap),
                     disabled_exciters=list(armed.rs.disabled_exciters), blocks=list(armed.protected_blocks),
                     events=events, library_corrections=notes, arrays=arrays, fingerprint=arrays_digest(arrays),
                     metadata=md, build_s=build_s)
    extra = base_state_trip_event(ss, case.blocks, case.no_margin_assets())
    if extra is not None:
        case.extra_events.append(extra)
    ver: dict[str, Any] = {}
    if verify:
        ver = verify_factory(case, ss)
        if log:
            log(f"[{case.slug}] factory verification: {json.dumps(ver)[:300]}")
    return case, ss, ver


def operator_action_events(ss: Any, mesh_line_ids: Sequence[str]) -> list[EventSpec]:
    from pa_dvsa.data_model import ProfileKind, TimeProfile

    out: list[EventSpec] = []
    all_muts: list[Mutation] = []
    for lid in mesh_line_ids:
        p = _pos(ss.Line, lid)
        if float(ss.Line.u.v[p]) > 0.5:
            continue
        y = _series_admittance(ss, lid)
        muts = (Mutation("Line", lid, "u", 1.0), Mutation("Line", lid, "ghk", y.real), Mutation("Line", lid, "bhk", y.imag))
        all_muts += list(muts)
        out.append(EventSpec(f"operator_action:{lid}", "line_energization", "switch", muts, TimeProfile(ProfileKind.STEP),
                             description=f"Operator action OA1: energize {lid}"))
    if len(out) > 1:
        out.append(EventSpec("operator_action:both", "line_energization", "switch", tuple(all_muts),
                             TimeProfile(ProfileKind.STEP), description="Operator actions OA1a and OA1b together"))
    return out


def output_reduction_event(ss: Any, blocks: Sequence[ProtectedBlock], mw: float, *, duration_s: float = 0.5,
                           event_id: str = "documented_output_reduction") -> EventSpec:
    from pa_dvsa.data_model import ProfileKind, TimeProfile

    pvd = ss.PVD1
    base = float(ss.config.mva)
    p0 = {b.asset_id: float(pvd.pref0.v[_pos(pvd, b.pvd1_idx)]) for b in blocks}
    total_mw = sum(p0.values()) * base
    f = 1.0 - float(mw) / total_mw
    muts: list[Mutation] = []
    for b in blocks:
        k = _pos(pvd, b.pvd1_idx)
        muts += [Mutation("PVD1", b.pvd1_idx, "pref0", f * float(pvd.pref0.v[k])),
                 Mutation("PVD1", b.pvd1_idx, "qref0", f * float(pvd.qref0.v[k]))]
    return EventSpec(event_id, "fixed_pf_ramp", "param", tuple(muts),
                     TimeProfile(ProfileKind.RAMP, duration_s=float(duration_s)),
                     description=f"Output of all protected plants reduced by {mw:g} MW ({100 * (1 - f):.1f} %) over "
                                 f"{duration_s:g} s at fixed power factor",
                     metadata={"mw": float(mw), "factor": f})


def build_sequence_case(kwargs: Mapping[str, Any], *, reduction_mw: float = 708.0, settle_s: float = SETTLE_S,
                        verify: bool = True, log: Callable[[str], None] | None = None
                        ) -> tuple[ArmedCase, Any, dict[str, Any]]:
    from .iberian_sequence import sequence_config

    cfg, record = sequence_config(**dict(kwargs))
    cfg = replace(cfg, pre_cascade_enabled=False, variant="sequence_screen")
    level = f"sequence_k{float(kwargs.get('tie_scale', 15.0)):g}"
    case, ss, ver = build_case_from_config(cfg, level, cfg.tap_margin_pu, settle_s=settle_s, verify=verify, log=log)
    case.events = list(case.events) + [output_reduction_event(ss, case.blocks, reduction_mw)]
    case.metadata["sequence_record"] = record
    return case, ss, ver


def _residuals(ss: Any) -> tuple[np.ndarray, np.ndarray]:
    ss.TDS.fg_update(ss.exist.pflow_tds)
    return np.array(ss.dae.f, dtype=float).copy(), np.array(ss.dae.g, dtype=float).copy()


def verify_factory(case: ArmedCase, armed_system: Any) -> dict[str, Any]:
    fac = SnapshotFactory(case)
    ss2 = fac()
    snap2 = snapshot_arrays(ss2)
    diff = arrays_diff(snap2, case.arrays)
    f1, g1 = _residuals(armed_system)
    f2, g2 = _residuals(ss2)
    snap_after = snapshot_arrays(ss2)
    snap_armed_after = snapshot_arrays(armed_system)
    diff_after = arrays_diff(snap_after, snap_armed_after)
    return {
        "factory_build_s": fac.last_build_s,
        "n_arrays": len(case.arrays),
        "fingerprint": case.fingerprint,
        "factory_fingerprint": arrays_digest(snap2),
        "arrays_identical": not diff,
        "arrays_differing": {k: v for k, v in list(diff.items())[:20]},
        "arrays_identical_after_residual_evaluation": not diff_after,
        "arrays_differing_after_residual_evaluation": {k: v for k, v in list(diff_after.items())[:20]},
        "max_abs_df": float(np.max(np.abs(f1 - f2))) if f1.size else 0.0,
        "max_abs_dg": float(np.max(np.abs(g1 - g2))) if g1.size else 0.0,
        "residuals_bitwise_identical": bool(np.array_equal(f1, f2) and np.array_equal(g1, g2)),
    }


def save_case(case: ArmedCase, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "wb") as fh:
        pickle.dump(case, fh, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)


def load_case(path: Path) -> ArmedCase:
    with open(path, "rb") as fh:
        case = pickle.load(fh)
    if arrays_digest(case.arrays) != case.fingerprint:
        raise RuntimeError(f"cached armed case {path} is corrupt (fingerprint mismatch)")
    return case


def reset_tds_for_reuse(ss: Any) -> None:
    tds = ss.TDS
    tds.deltat = tds.deltatmin = tds.deltatmax = 0
    tds.h = 0
    tds.last_pc = 0.0
    tds.converged = False
    tds.last_converged = False
    tds.busted = False
    tds.niter = 0
    tds._switch_idx = 0
    tds._last_switch_t = -999
    tds.custom_event = False
    tds.mis = [1, 1]
    tds.err_msg = ""
    tds.callpert = None
    ss.dae.t = np.array(0.0)
    for _, _, obj in _holders(ss):
        if _is_delay(obj):
            for attr in ("t", "_last_t"):
                if isinstance(getattr(obj, attr, None), np.ndarray):
                    setattr(obj, attr, np.zeros_like(getattr(obj, attr)))
            obj.rewind = False


def output_reachable_states(ar: sparse.spmatrix, cr: sparse.spmatrix) -> np.ndarray:
    a = sparse.csr_matrix(ar)
    a.eliminate_zeros()
    c = sparse.csc_matrix(cr)
    c.eliminate_zeros()
    n = a.shape[0]
    keep = np.zeros(n, dtype=bool)
    keep[np.unique(c.nonzero()[1])] = True
    frontier = np.flatnonzero(keep)
    while frontier.size:
        cols = np.unique(a[frontier].nonzero()[1])
        new = cols[~keep[cols]]
        keep[new] = True
        frontier = new
    return np.flatnonzero(keep)


def output_reachable_states_dense(ar: np.ndarray, cr: np.ndarray) -> np.ndarray:
    nz = np.asarray(ar) != 0.0
    keep = np.zeros(nz.shape[0], dtype=bool)
    keep[np.any(np.asarray(cr) != 0.0, axis=0)] = True
    frontier = np.flatnonzero(keep)
    while frontier.size:
        cols = np.flatnonzero(np.any(nz[frontier], axis=0))
        new = cols[~keep[cols]]
        keep[new] = True
        frontier = new
    return np.flatnonzero(keep)


def _stability(max_real: float | None, tol: float = STABILITY_TOL) -> str:
    if max_real is None or not math.isfinite(max_real):
        return "not_evaluated"
    if max_real > tol:
        return "unstable"
    if max_real >= -tol:
        return "marginal"
    return "stable"


@dataclass
class PrunedModeReduction(ModeReduction):
    keep: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=int))
    n_full: int = 0
    max_real_removed: float | None = None
    top_eigenvalues: tuple[complex, ...] = ()
    eig_method: str = "dense"
    eig_radius: float | None = None

    def reduce_forcing(self, df: np.ndarray, dg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        b, fd = ModeReduction.reduce_forcing(self, df, dg)
        return b[self.keep], fd

    def extended_diagnostics(self) -> dict[str, Any]:
        d = self.diagnostics().to_dict()
        full = [v for v in (d["max_real_eigenvalue"], self.max_real_removed) if v is not None]
        d.update({
            "n_states_full": int(self.n_full),
            "n_states_output_reachable": int(self.keep.size),
            "max_real_eigenvalue_removed_block": self.max_real_removed,
            "max_real_eigenvalue_full": max(full) if full else None,
            "unstable_flag_full": bool(full and max(full) > STABILITY_TOL),
            "top_eigenvalues": [[float(z.real), float(z.imag)] for z in self.top_eigenvalues],
            "eig_method": self.eig_method,
            "eig_check_radius": self.eig_radius,
            "timing_s": dict(self.timing_s),
        })
        return d


class _ChannelColumn:
    __slots__ = ("vec", "channel")

    def __init__(self, vec: np.ndarray, channel: np.ndarray) -> None:
        self.vec = vec
        self.channel = channel

    def __matmul__(self, other: Any) -> np.ndarray:
        v = np.asarray(other, dtype=float).reshape(-1)
        if v is not self.channel and not np.array_equal(v, self.channel):
            raise RuntimeError("channel-bound propagation block applied to a different input vector")
        return self.vec


class FastModeDynamics(ModeDynamics):
    def __init__(self, a: Any, *, dt_atomic: float = 2.5e-4, need_ramp_blocks: bool = True,
                 blocks_method: str = "exact") -> None:
        if blocks_method not in ("reference", "exact", "balanced"):
            raise ValueError("blocks_method must be 'reference', 'exact' or 'balanced'")
        super().__init__(a, dt_atomic=dt_atomic, need_ramp_blocks=need_ramp_blocks)
        self.blocks_method = blocks_method
        self._channel: np.ndarray | None = None
        self._vec_atomic: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        self._vec_dt: dict[float, tuple[np.ndarray, np.ndarray]] = {}
        self.block_timing_s = 0.0
        self.matrix_products = 0

    def bind_channel(self, b: np.ndarray) -> None:
        self._channel = np.array(b, dtype=float).reshape(-1).copy()
        self._vec_atomic.clear()
        self._vec_dt.clear()

    def blocks(self, dt: float) -> tuple[np.ndarray, Any, np.ndarray | None]:
        t0 = time.perf_counter()
        try:
            if self._channel is None or self.need_ramp_blocks or self.blocks_method == "reference":
                return super().blocks(dt)
            dt = float(dt)
            if dt <= 0.0:
                raise RuntimeError("propagation step must be positive")
            k = dt / self.dt_atomic
            k_int = int(round(k))
            if k_int >= 1 and abs(k - k_int) <= 1e-7 * max(k, 1.0):
                e, g = self._atomic_multiple_vec(k_int)
            else:
                key = round(dt, 12)
                if key not in self._vec_dt:
                    self._vec_dt[key] = self._step_vec(dt)
                e, g = self._vec_dt[key]
            return e, _ChannelColumn(g, self._channel), None
        finally:
            self.block_timing_s += time.perf_counter() - t0

    def _atomic_multiple_vec(self, k: int) -> tuple[np.ndarray, np.ndarray]:
        if k in self._vec_atomic:
            return self._vec_atomic[k]
        if k == 1:
            self._vec_atomic[1] = self._step_vec(self.dt_atomic)
            return self._vec_atomic[1]
        half = k // 2
        eh, gh = self._atomic_multiple_vec(half)
        e2, g2 = self._compose_vec((eh, gh), (eh, gh))
        if k % 2:
            e1, g1 = self._atomic_multiple_vec(1)
            e2, g2 = self._compose_vec((e2, g2), (e1, g1))
        self._vec_atomic[k] = (e2, g2)
        return self._vec_atomic[k]

    def _compose_vec(self, first: tuple[np.ndarray, np.ndarray], second: tuple[np.ndarray, np.ndarray]
                     ) -> tuple[np.ndarray, np.ndarray]:
        ea, ga = first
        eb, gb = second
        self.matrix_products += 1
        return eb @ ea, eb @ ga + gb

    def _step_vec(self, dt: float) -> tuple[np.ndarray, np.ndarray]:
        return self._balanced_vec(dt) if self.blocks_method == "balanced" else self._scaling_squaring_vec(dt)

    def _scaling_squaring_vec(self, dt: float) -> tuple[np.ndarray, np.ndarray]:
        n = self.n
        b = self._channel
        eye = np.eye(n)
        norm = float(np.abs(self.a).sum(axis=0).max()) * dt
        squarings = max(0, int(math.ceil(math.log2(norm / 0.5)))) if norm > 0.5 else 0
        h = dt / (2.0**squarings)
        ah = self.a * h
        e = eye.copy()
        g = h * b
        power = eye.copy()
        fact = 1.0
        for k in range(1, 30):
            power = power @ ah
            self.matrix_products += 1
            fact *= k
            e += power / fact
            g = g + (h / (fact * (k + 1))) * (power @ b)
            if np.abs(power).max() / fact < 1e-18:
                break
        for _ in range(squarings):
            g = e @ g + g
            e = e @ e
            self.matrix_products += 1
        return e, g

    def _balanced_vec(self, dt: float) -> tuple[np.ndarray, np.ndarray]:
        from scipy.linalg import expm

        n = self.n
        d = self.scale
        aug = np.zeros((n + 1, n + 1))
        aug[:n, :n] = self.a * (1.0 / d)[:, None] * d[None, :] * dt
        aug[:n, n] = (self._channel / d) * dt
        ea = expm(aug)
        self.matrix_products += 8
        e = ea[:n, :n] * d[:, None] * (1.0 / d)[None, :]
        g = ea[:n, n] * d
        return np.ascontiguousarray(e), np.ascontiguousarray(g)

    def _scaling_squaring(self, dt: float) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        if self.blocks_method != "balanced":
            return super()._scaling_squaring(dt)
        d = self.scale
        orig = self.a
        try:
            self.a = np.ascontiguousarray(orig * (1.0 / d)[:, None] * d[None, :])
            e, g1, g2 = super()._scaling_squaring(dt)
        finally:
            self.a = orig
        unscale = lambda m: None if m is None else np.ascontiguousarray(m * d[:, None] * (1.0 / d)[None, :])
        return unscale(e), unscale(g1), unscale(g2)


def near_zero_eigenvalues(a: np.ndarray, *, k: int = 12, sigma: float = 1e-3, tol: float = 1e-10
                          ) -> tuple[np.ndarray, float, int]:
    from scipy.sparse import linalg as spla

    keep = np.ones(a.shape[0], dtype=bool)
    while True:
        sub = a[np.ix_(keep, keep)]
        zero_rows = np.flatnonzero(~np.any(sub != 0.0, axis=1))
        if not zero_rows.size:
            break
        idx = np.flatnonzero(keep)[zero_rows]
        keep[idx] = False
    sub = a[np.ix_(keep, keep)]
    ndef = int(np.count_nonzero(~keep))
    m = sub.shape[0]
    if m <= k + 2:
        vals = np.linalg.eigvals(sub)
    else:
        vals = spla.eigs(sub, k=k, sigma=sigma, which="LM", return_eigenvectors=False, tol=tol, maxiter=5000)
    radius = float(np.max(np.abs(vals - sigma))) if vals.size else 0.0
    if ndef:
        vals = np.concatenate((vals, np.zeros(ndef)))
    return np.asarray(vals), radius, ndef


class PrunedEventResponseEngine(EventResponseEngine):
    def __init__(self, *args: Any, propagation: str = "krylov", reduce_solver: str = "klu",
                 eig_solver: str = "descriptor", blocks_method: str = "exact", post_eig: str = "near_zero",
                 reduce_columns: str = "nonzero", **kwargs: Any) -> None:
        if propagation not in ("krylov", "dense"):
            raise ValueError("propagation must be 'krylov' or 'dense'")
        if reduce_solver not in ("klu", "superlu"):
            raise ValueError("reduce_solver must be 'klu' or 'superlu'")
        if eig_solver not in ("descriptor", "dense"):
            raise ValueError("eig_solver must be 'descriptor' or 'dense'")
        if post_eig not in ("dense", "near_zero", "none"):
            raise ValueError("post_eig must be 'dense', 'near_zero' or 'none'")
        if reduce_columns not in ("all", "nonzero"):
            raise ValueError("reduce_columns must be 'all' or 'nonzero'")
        self.propagation = propagation
        self.reduce_solver = reduce_solver
        self.eig_solver = eig_solver
        self.blocks_method = blocks_method
        self.post_eig = post_eig
        self.reduce_columns = reduce_columns
        self._propagation_info: dict[str, Any] = {}
        super().__init__(*args, **kwargs)

    def _propagate_channel(self, mode: ModeReduction, b: np.ndarray, f: np.ndarray, profile: Any,
                           channel_id: str) -> Any:
        dyn = mode.dynamics
        self._propagation_info = {}
        if not isinstance(dyn, KrylovModeDynamics):
            return super()._propagate_channel(mode, b, f, profile, channel_id)
        try:
            resp, info = propagate_channel_krylov(dyn, mode.cr, b, f, profile, self.times_s, channel_id=channel_id)
            self._propagation_info = {"propagation": "krylov", **info}
            return resp
        except KrylovNotConverged as exc:
            self._propagation_info = {"propagation": "dense_fallback", "krylov_failure": str(exc)}
            dense = ModeDynamics(dyn.a, dt_atomic=self.dt_atomic_s, need_ramp_blocks=True)
            return propagate_channel(dense, mode.cr, b, f, profile, self.times_s, channel_id=channel_id,
                                     keep_states=True)

    def _finish(self, **kwargs: Any) -> EventResponse:
        mode = kwargs.get("mode")
        dyn = getattr(mode, "dynamics", None)
        if kwargs.get("kind") == "switch" and isinstance(dyn, FastModeDynamics):
            dyn.bind_channel(kwargs["b"])
        resp = super()._finish(**kwargs)
        if isinstance(dyn, FastModeDynamics):
            resp.timing_s = {**dict(resp.timing_s), "blocks": dyn.block_timing_s, "matrix_products": dyn.matrix_products}
            dyn.block_timing_s = 0.0
            dyn.matrix_products = 0
        if self._propagation_info:
            resp.metadata = {**dict(resp.metadata), **self._propagation_info}
        if isinstance(dyn, KrylovModeDynamics):
            resp.metadata = {**dict(resp.metadata), "log_norm_certificate_delta": dyn.mu_delta}
        return resp

    def _build_mode(self, jac: Any, mode_id: str, *, ramp_blocks: bool) -> ModeReduction:
        t0 = time.perf_counter()
        model = LinearDAEModel.from_andes_jacobians(jac, cx=self._cx, cy=self._cy, mode_id=mode_id)
        factor = AlgebraicFactor(model.gy, names=model.y_names, max_condition=self.max_condition)
        t1 = time.perf_counter()
        n = model.n_states
        if n and self.reduce_solver == "klu":
            py, pc = schur_products(factor, model.gx, model.ay, model.cy)
            ar_d = model.ax.toarray() - py
            cr_d = model.cx.toarray() - pc
            del py, pc
            keep = output_reachable_states_dense(ar_d, cr_d)
            drop = np.setdiff1d(np.arange(n), keep)
            if drop.size and np.any(ar_d[np.ix_(keep, drop)] != 0.0):
                raise RuntimeError(f"{mode_id}: output-reachability reduction is not exact (A_kd has entries)")
            if drop.size and np.any(cr_d[:, drop] != 0.0):
                raise RuntimeError(f"{mode_id}: removed states enter an output")
            akk = np.ascontiguousarray(ar_d[np.ix_(keep, keep)])
            crk = np.ascontiguousarray(cr_d[:, keep])
            add_dense = ar_d[np.ix_(drop, drop)] if drop.size and (mode_id == "pre_event" or self.post_eig == "dense") else None
            del ar_d, cr_d
        else:
            if not n:
                x_sol = np.zeros((model.n_algebraic, 0))
            elif self.reduce_columns == "nonzero":
                gx = sparse.csc_matrix(model.gx)
                gx.eliminate_zeros()
                cols = np.flatnonzero(np.diff(gx.indptr) > 0)
                x_sol = np.zeros((model.n_algebraic, n))
                if cols.size:
                    x_sol[:, cols] = factor.solve(gx[:, cols].toarray())
            else:
                x_sol = factor.solve(model.gx.toarray())
            if not np.all(np.isfinite(x_sol)):
                raise RuntimeError(f"{mode_id}: G_y^-1 G_x is not finite")
            ar_full = sparse.csc_matrix(np.asarray(model.ax - model.ay @ x_sol))
            cr_full = sparse.csc_matrix(np.asarray(model.cx - model.cy @ x_sol))
            del x_sol
            keep = output_reachable_states(ar_full, cr_full)
            drop = np.setdiff1d(np.arange(n), keep)
            akd = ar_full[keep][:, drop]
            akd.eliminate_zeros()
            if akd.nnz:
                raise RuntimeError(f"{mode_id}: output-reachability reduction is not exact (A_kd has {akd.nnz} entries)")
            cd = cr_full[:, drop]
            cd.eliminate_zeros()
            if cd.nnz:
                raise RuntimeError(f"{mode_id}: removed states enter an output")
            akk = ar_full[keep][:, keep].toarray()
            crk = cr_full[:, keep].toarray()
            add_dense = ar_full[drop][:, drop].toarray() if drop.size else None
        t2 = time.perf_counter()
        warnings: list[str] = list(self.warnings)
        eig_method = "dense" if (mode_id == "pre_event" or self.post_eig == "dense") else self.post_eig
        eig_radius = None
        try:
            if eig_method == "dense":
                ev = np.linalg.eigvals(akk) if keep.size else np.zeros(0)
            elif eig_method == "near_zero" and not keep.size:
                ev, eig_radius = np.zeros(0), 0.0
            elif eig_method == "near_zero" and self.eig_solver == "descriptor":
                ev, eig_radius, _ = near_zero_eigenvalues_descriptor(model, keep, akk)
            elif eig_method == "near_zero":
                ev, eig_radius, _ = near_zero_eigenvalues(akk)
            else:
                ev = np.zeros(0)
            max_real = float(np.max(ev.real)) if ev.size else None
            top = tuple(ev[np.argsort(-ev.real)[:6]]) if ev.size else ()
        except Exception as exc:
            warnings.append(f"stability_eigenvalue_check_failed: {exc}")
            max_real, top = None, ()
        max_removed = None
        if drop.size and eig_method == "dense" and add_dense is not None:
            evd = np.linalg.eigvals(add_dense)
            max_removed = float(np.max(evd.real))
        t3 = time.perf_counter()
        stability = _stability(max_real)
        if stability == "unstable":
            warnings.append(f"reduced_state_matrix_unstable_max_real={max_real:.6g}")
        elif stability == "marginal":
            warnings.append(f"reduced_state_matrix_marginal_max_real={max_real:.6g}")
        if max_removed is not None and max_removed > STABILITY_TOL:
            warnings.append(f"removed_unobservable_block_unstable_max_real={max_removed:.6g}")
        diag = ReductionDiagnostics(
            mode_id=mode_id, n_states=int(keep.size), n_algebraic=int(model.n_algebraic),
            n_outputs=int(model.n_outputs), n_controls=0, n_disturbances=0, algebraic_only=(keep.size == 0),
            gy_factorized=True, gy_min_abs_u_diag=factor.min_abs_pivot, max_real_eigenvalue=max_real,
            stability=stability, warnings=tuple(warnings), gy_condition_estimate=factor.condition_estimate,
            removed_algebraic=tuple(model.removed_algebraic),
        )
        nk = int(keep.size)
        reduced = ReducedLinearModel(
            ar=sparse.csc_matrix(akk), br=sparse.csc_matrix((nk, 0)), dr=sparse.csc_matrix((nk, 0)),
            cr=sparse.csc_matrix(crk), fu=sparse.csc_matrix((crk.shape[0], 0)), fd=sparse.csc_matrix((crk.shape[0], 0)),
            diagnostics=diag,
        )
        if self.propagation == "krylov":
            dyn = KrylovModeDynamics(akk, model, keep, dt_atomic=self.dt_atomic_s)
            extra = {"balance_only": dyn.timing_s["balance"], "log_norm": dyn.timing_s["log_norm"]}
        else:
            dyn = FastModeDynamics(akk, dt_atomic=self.dt_atomic_s, need_ramp_blocks=ramp_blocks,
                                   blocks_method=self.blocks_method)
            extra = {}
        t4 = time.perf_counter()
        return PrunedModeReduction(
            mode_id=mode_id, model=model, factor=factor, reduced=reduced, dynamics=dyn, cr=crk,
            timing_s={"factor": t1 - t0, "reduce": t2 - t1, "eig": t3 - t2, "balance": t4 - t3, **extra},
            warnings=tuple(warnings), keep=keep, n_full=int(n), max_real_removed=max_removed, top_eigenvalues=top,
            eig_method=eig_method, eig_radius=eig_radius,
        )


def collector_selectors(ss: Any, blocks: Sequence[ProtectedBlock]) -> list[OutputSelector]:
    return [OutputSelector.bus_voltage(ss, b.collector_bus, b.asset_id) for b in blocks]


ENGINE_OPTIONS = {"propagation": "krylov", "reduce_solver": "klu", "eig_solver": "descriptor",
                  "blocks_method": "exact", "post_eig": "near_zero", "reduce_columns": "nonzero"}
DENSE_ENGINE_OPTIONS = {"propagation": "dense", "reduce_solver": "superlu", "eig_solver": "dense",
                        "blocks_method": "exact", "post_eig": "near_zero", "reduce_columns": "nonzero"}
REFERENCE_ENGINE_OPTIONS = {"propagation": "dense", "reduce_solver": "superlu", "eig_solver": "dense",
                            "blocks_method": "reference", "post_eig": "dense", "reduce_columns": "all"}


def build_engine(ss: Any, blocks: Sequence[ProtectedBlock], *, keep_channels: bool = False,
                 **options: Any) -> PrunedEventResponseEngine:
    opts = {**ENGINE_OPTIONS, **options}
    return PrunedEventResponseEngine(ss, collector_selectors(ss, blocks), horizon_s=ENGINE_HORIZON_S, dt_s=DT_S,
                                     keep_channels=keep_channels, **opts)


def restoration_errors(engine: EventResponseEngine) -> dict[str, float]:
    ss = engine.system
    dae = ss.dae
    errs = {
        "x": float(np.max(np.abs(np.asarray(dae.x) - engine.x0))) if engine.x0.size else 0.0,
        "y": float(np.max(np.abs(np.asarray(dae.y) - engine.y0))) if engine.y0.size else 0.0,
        "f": float(np.max(np.abs(np.asarray(dae.f) - engine.f0))) if engine.f0.size else 0.0,
        "g": float(np.max(np.abs(np.asarray(dae.g) - engine.g0))) if engine.g0.size else 0.0,
    }
    jac = engine._jacobians()
    for name in ("fx", "fy", "gx", "gy"):
        diff = getattr(jac, name) - getattr(engine._jac0, name)
        errs[name] = float(np.max(np.abs(diff.data))) if diff.nnz else 0.0
    return errs


def in_service_masks(ss: Any) -> tuple[np.ndarray, np.ndarray]:
    xm = np.ones(int(ss.dae.n), dtype=bool)
    ym = np.ones(int(ss.dae.m), dtype=bool)
    for model in ss.exist.pflow_tds.values():
        if not getattr(model, "n", 0) or not hasattr(model, "u") or model.class_name in ("Bus",):
            continue
        u = np.asarray(model.u.v, dtype=float)
        ue = np.asarray(model.ue.v, dtype=float) if hasattr(model, "ue") else u
        off = np.flatnonzero((u <= 0.5) | (ue <= 0.5))
        if not off.size:
            continue
        for var in model.states.values():
            xm[np.asarray(var.a)[off]] = False
        for var in model.algebs.values():
            ym[np.asarray(var.a)[off]] = False
    return xm, ym


def residual_split(ss: Any) -> dict[str, Any]:
    f, g = _residuals(ss)
    xm, ym = in_service_masks(ss)
    out = {}
    for label, vec, mask, names in (("f", f, xm, ss.dae.x_name), ("g", g, ym, ss.dae.y_name)):
        k_all = int(np.argmax(np.abs(vec))) if vec.size else 0
        v_in = np.where(mask, np.abs(vec), 0.0)
        k_in = int(np.argmax(v_in)) if vec.size else 0
        out[f"max_abs_{label}_all"] = float(np.abs(vec[k_all])) if vec.size else 0.0
        out[f"argmax_{label}_all"] = names[k_all] if vec.size else ""
        out[f"max_abs_{label}_in_service"] = float(v_in[k_in]) if vec.size else 0.0
        out[f"argmax_{label}_in_service"] = names[k_in] if vec.size else ""
    return out


def _exciter_vout_addresses(ss: Any) -> dict[str, int]:
    out: dict[str, int] = {}
    for name in ("ESDC1A", "IEEET1", "SEXS"):
        model = getattr(ss, name, None)
        if model is None or not model.n or not hasattr(model, "vout"):
            continue
        ue = np.asarray(model.ue.v, dtype=float) if hasattr(model, "ue") else np.asarray(model.u.v, dtype=float)
        for k in range(model.n):
            if ue[k] > 0.5:
                out[f"{name}:{model.idx.v[k]}"] = int(model.vout.a[k])
    return out


def continuation_jump(ss: Any, mutations: Sequence[Mutation], *, steps: int = 10, max_iter: int = 30) -> dict[str, Any]:
    base = read_base_values(ss, mutations)
    its = []
    for k in range(1, steps + 1):
        s = k / steps
        for m, b in zip(mutations, base):
            model = getattr(ss, m.model)
            arr = getattr(model, m.attr).v
            arr[_pos(model, m.device_idx)] = float(b) + s * (float(m.value) - float(b))
        it, res = solve_frozen_algebraic(ss, max_iter=max_iter)
        its.append(it)
    return {"steps": steps, "newton_iterations": its, "final_residual": res}


def jump_branch_check(engine: EventResponseEngine, ev: EventSpec, factory: Callable[[], Any], response: EventResponse,
                      *, probe_s: float = 1e-6, t20_s: float = 0.02, tstep_s: float = 1e-3, tol: float = 1e-9
                      ) -> dict[str, Any]:
    ss = engine.system
    out: dict[str, Any] = {"event_id": ev.event_id, "family": ev.family}
    xm, ym = in_service_masks(ss)
    vout_all = _exciter_vout_addresses(ss)
    mutated = {f"{m.model}:{m.device_idx}" for m in ev.mutations}
    vout = {k: a for k, a in vout_all.items() if k not in mutated}
    vaddr = np.array(list(vout.values()), dtype=int)
    vaddr_all = np.array(list(vout_all.values()), dtype=int)
    zaddr = np.array([sel.address for sel in engine.selectors], dtype=int)
    base = read_base_values(ss, ev.mutations)
    y0 = engine.y0.copy()
    try:
        apply_mutations(ss, ev.mutations, base, 1.0)
        it, res = solve_frozen_algebraic(ss, zero_tf_mask=engine._zero_tf)
        y_direct = np.array(ss.dae.y, dtype=float).copy()
        x_direct = np.array(ss.dae.x, dtype=float).copy()
    finally:
        engine._restore(ev.mutations, base)
    try:
        cont = continuation_jump(ss, ev.mutations)
        y_cont = np.array(ss.dae.y, dtype=float).copy()
    except Exception as exc:
        cont = {"error": f"{type(exc).__name__}: {exc}"}
        y_cont = np.full_like(y_direct, np.nan)
    finally:
        engine._restore(ev.mutations, base)
    ym_post2 = ym
    try:
        apply_mutations(ss, ev.mutations, base, 1.0)
        _, ym_post2 = in_service_masks(ss)
    finally:
        restore_mutations(ss, ev.mutations, base)
        ss.vars_to_models()
    ymask = ym & ym_post2
    out.update({"direct_newton_iterations": it, "direct_residual": res,
                "direct_max_abs_dx": float(np.max(np.abs(x_direct - engine.x0))),
                "continuation": cont,
                "max_abs_dy_direct_vs_continuation": float(np.nanmax(np.abs(y_direct - y_cont)[ymask])),
                "max_abs_dvout_enabled_exciters_at_jump": float(np.max(np.abs(y_direct[vaddr] - y0[vaddr]))) if vaddr.size else 0.0,
                "max_abs_dvout_incl_switched_off_exciters": float(np.max(np.abs(y_direct[vaddr_all] - y0[vaddr_all]))) if vaddr_all.size else 0.0,
                "argmax_dvout_enabled_exciters": (list(vout)[int(np.argmax(np.abs(y_direct[vaddr] - y0[vaddr])))] if vaddr.size else ""),
                "max_abs_dy_in_service_at_jump": float(np.max(np.abs(y_direct - y0)[ymask]))})
    t0 = time.perf_counter()
    try:
        ss2 = factory()
        tds = ss2.TDS
        tds.config.tol = tol
        tds.config.no_tqdm = 1
        b2 = read_base_values(ss2, ev.mutations)
        apply_mutations(ss2, ev.mutations, b2, 1.0)
        tds.custom_event = True
        tds.config.tstep = probe_s
        tds.config.tf = probe_s
        with QuietAndes(True):
            ok1 = bool(tds.run(no_summary=True))
        y_nat = np.array(ss2.dae.y, dtype=float).copy()
        tds.config.tstep = tstep_s
        tds.config.tf = t20_s
        with QuietAndes(True):
            ok2 = bool(tds.run(no_summary=True))
        y_nat20 = np.array(ss2.dae.y, dtype=float).copy()
        k20 = int(np.argmin(np.abs(response.times_s - t20_s)))
        dz_screen20 = response.values_pu[k20]
        dz_nat20 = y_nat20[zaddr] - y0[zaddr]
        dz_nat0 = y_nat[zaddr] - y0[zaddr]
        out.update({
            "native_ok": ok1 and ok2,
            "native_time_s": float(ss2.dae.t),
            "max_abs_dy_direct_vs_native": float(np.max(np.abs(y_direct - y_nat)[ymask])),
            "argmax_dy_direct_vs_native": ss.dae.y_name[int(np.argmax(np.where(ymask, np.abs(y_direct - y_nat), 0)))],
            "max_abs_dvout_direct_vs_native": float(np.max(np.abs(y_direct[vaddr_all] - y_nat[vaddr_all]))) if vaddr_all.size else 0.0,
            "jump_screen_max_abs_mpu": float(np.max(np.abs(response.jump_pu))) * 1e3,
            "jump_screen_vs_native_max_err_mpu": float(np.max(np.abs(response.jump_pu - dz_nat0))) * 1e3,
            "dz20_screen_max_abs_mpu": float(np.max(np.abs(dz_screen20))) * 1e3,
            "dz20_native_max_abs_mpu": float(np.max(np.abs(dz_nat20))) * 1e3,
            "dz20_screen_vs_native_max_err_mpu": float(np.max(np.abs(dz_screen20 - dz_nat20))) * 1e3,
            "dz20_worst_asset": engine.output_ids[int(np.argmax(np.abs(dz_screen20 - dz_nat20)))],
            "native_wall_s": time.perf_counter() - t0,
        })
    except Exception as exc:
        out.update({"native_ok": False, "native_error": f"{type(exc).__name__}: {exc}"})
    return out


@dataclass
class ScreenStore:
    slug: str
    asset_ids: tuple[str, ...]
    z0: np.ndarray
    event_ids: tuple[str, ...]
    families: dict[str, str]
    responses: dict[str, EventResponse]
    failed: dict[str, str]
    event_runtime_s: dict[str, float]
    mode_diagnostics: dict[str, dict[str, Any]]
    restoration: dict[str, dict[str, float]]
    engine_info: dict[str, Any]
    timing_s: dict[str, float]


def run_screen(case: ArmedCase, factory: SnapshotFactory, *, log: Callable[[str], None] | None = None,
               event_ids: Sequence[str] | None = None) -> tuple[ScreenStore, PrunedEventResponseEngine]:
    t_fac = time.perf_counter()
    ss = factory()
    t_fac = time.perf_counter() - t_fac
    t0 = time.perf_counter()
    engine = build_engine(ss, case.blocks)
    t_engine = time.perf_counter() - t0
    for b, z in zip(case.blocks, engine.z0):
        if abs(float(z) - float(b.z0_pu)) > 1e-10:
            raise RuntimeError(f"engine z0 of {b.asset_id} ({z}) differs from the armed z0 ({b.z0_pu})")
    t0 = time.perf_counter()
    pre = engine.pre_mode
    t_pre = time.perf_counter() - t0
    modes = {"pre_event": pre.extended_diagnostics()}
    if log:
        d = modes["pre_event"]
        log(f"[{case.slug}] pre-mode n={d['n_states_full']}->{d['n_states_output_reachable']} max Re={d['max_real_eigenvalue']:.3e} "
            f"cond={d['gy_condition_estimate']:.3e} mu={d['log_norm_mu']:.1f} ({t_pre:.1f} s)")
    evs = [e for e in list(case.events) + list(case.extra_events) if event_ids is None or e.event_id in event_ids]
    responses: dict[str, EventResponse] = {}
    failed: dict[str, str] = {}
    rt: dict[str, float] = {}
    restoration: dict[str, dict[str, float]] = {}
    orig_build = engine._build_mode
    captured: dict[str, dict[str, Any]] = {}

    def capturing(jac: Any, mode_id: str, *, ramp_blocks: bool) -> ModeReduction:
        m = orig_build(jac, mode_id, ramp_blocks=ramp_blocks)
        captured[mode_id] = m.extended_diagnostics()
        return m

    engine._build_mode = capturing
    for ev in evs:
        t = time.perf_counter()
        try:
            responses[ev.event_id] = engine.event_response(ev)
        except Exception as exc:
            failed[ev.event_id] = f"{type(exc).__name__}: {exc}"
        rt[ev.event_id] = time.perf_counter() - t
        restoration[ev.event_id] = restoration_errors(engine)
        if ev.kind == "switch" and f"post_{ev.event_id}" in captured:
            modes[f"post_{ev.event_id}"] = captured[f"post_{ev.event_id}"]
        if log:
            r = responses.get(ev.event_id)
            pk = max(e.peak_upper for e in r.envelopes) * 1e3 if r is not None else float("nan")
            log(f"[{case.slug}]   {ev.event_id}: {rt[ev.event_id]:.1f} s, max peak {pk:.2f} mpu"
                + (f" FAILED {failed[ev.event_id]}" if ev.event_id in failed else ""))
    engine._build_mode = orig_build
    store = ScreenStore(
        slug=case.slug, asset_ids=tuple(b.asset_id for b in case.blocks), z0=np.asarray(engine.z0, dtype=float),
        event_ids=tuple(e.event_id for e in evs), families={e.event_id: e.family for e in evs}, responses=responses,
        failed=failed, event_runtime_s=rt, mode_diagnostics=modes, restoration=restoration,
        engine_info={"n_x": int(ss.dae.n), "n_y": int(ss.dae.m), "warnings": list(engine.warnings),
                     "pegged_states": list(engine.pegged_states), "baseline_residual": list(engine.baseline_residual),
                     "residual_split": residual_split(ss), "horizon_s": ENGINE_HORIZON_S, "dt_s": DT_S},
        timing_s={"factory_build_s": t_fac, "engine_baseline_s": t_engine, "pre_mode_s": t_pre,
                  "events_total_s": float(sum(rt.values()))},
    )
    return store, engine


def window_maps(store: ScreenStore, case: ArmedCase, relay: RelaySetting, *, library_only: bool = True) -> RigorousWindowMaps:
    order = [e for e in store.event_ids if (not library_only or not e.startswith("base_state_trips"))]
    resp = [store.responses[e] for e in order if e in store.responses]
    failed = {e: store.failed[e] for e in order if e in store.failed}
    margins = [relay.pickup_pu - float(z) for z in store.z0]
    return assemble_window_maps(resp, asset_ids=store.asset_ids, output_index={a: i for i, a in enumerate(store.asset_ids)},
                                margins_pu=margins, dwell_s=[relay.dwell_s] * len(store.asset_ids),
                                horizon_s=relay.horizon_s, failed_events=failed, event_order=order,
                                metadata={"relay": relay.name})


def predictions(store: ScreenStore, case: ArmedCase, relay: RelaySetting, maps: RigorousWindowMaps,
                *, extra_seeds: Sequence[str] = ()) -> dict[str, Any]:
    from pa_dvsa.revised_studies.screen import predict_seed

    amap = {b.asset_id: b.trip_event_id for b in case.blocks}
    out_idx = {a: i for i, a in enumerate(store.asset_ids)}
    resp = [store.responses[e] for e in maps.event_ids if e in store.responses]
    preds: dict[str, Any] = {}
    for sid in maps.event_ids:
        if sid not in store.responses:
            continue
        preds[sid] = predict_seed(maps, sid, responses=resp, amap=amap, out_idx=out_idx, epsilon=EPSILON,
                                  relays_enabled=True)
    for sid in extra_seeds:
        if sid not in store.responses:
            continue
        m2 = assemble_window_maps(resp + [store.responses[sid]], asset_ids=store.asset_ids, output_index=out_idx,
                                  margins_pu=list(maps.margins_pu), dwell_s=list(maps.dwell_s),
                                  horizon_s=relay.horizon_s, event_order=list(maps.event_ids) + [sid])
        preds[sid] = predict_seed(m2, sid, responses=resp + [store.responses[sid]], amap=amap, out_idx=out_idx,
                                  epsilon=EPSILON, relays_enabled=True)
    return preds


def peak_bound(env: Any, horizon_s: float) -> float:
    mask_pts = env.fine_times_s <= horizon_s + 1e-12
    mask_int = env.fine_times_s[1:] <= horizon_s + 1e-12
    vals = np.concatenate((env.fine_values[mask_pts], env.fine_left_values[mask_pts]))
    up = float(np.max(env.fine_upper[mask_int])) if (env.fine_upper is not None and np.any(mask_int)) else -math.inf
    return max(float(np.max(vals)), up)


def peak_time(env: Any, horizon_s: float) -> float:
    t = env.fine_times_s
    m = t <= horizon_s + 1e-12
    vals = np.maximum(env.fine_values[m], env.fine_left_values[m])
    return float(t[m][int(np.argmax(vals))])


def screen_k_rows(store: ScreenStore, relay: RelaySetting, maps: RigorousWindowMaps) -> list[dict[str, Any]]:
    rows = []
    for j, e in enumerate(maps.event_ids):
        r = store.responses.get(e)
        for i, a in enumerate(maps.asset_ids):
            cls = str(maps.classes[i, j])
            env = r.envelopes[i] if r is not None else None
            rows.append({
                "slug": store.slug, "relay": relay.name, "event_id": e, "family": store.families.get(e, ""),
                "asset_id": a, "channel_class": cls, "is_self": cls == "self", "z0_pu": float(store.z0[i]),
                "margin_pu": float(maps.margins_pu[i]), "no_margin": not (float(maps.margins_pu[i]) > 0.0),
                "k_pk": float(maps.k_pickup[i, j]), "k_tr": float(maps.k_trip[i, j]),
                "pickup_excursion_pu": float(maps.pickup_excursion_pu[i, j]),
                "trip_excursion_pu": float(maps.trip_excursion_pu[i, j]),
                "sampled_peak_pu": float(maps.sampled_peak_pu[i, j]), "padding_pu": float(maps.padding_pu[i, j]),
                "tolerance_met": bool(maps.tolerance_met[i, j]),
                "peak_time_s": peak_time(env, relay.horizon_s) if env is not None else float("nan"),
                "jump_pu": float(r.jump_pu[i]) if r is not None else float("nan"),
                "refined_intervals": int(getattr(env, "refined_intervals", -1)) if env is not None else -1,
            })
    return rows


def nonself_max(maps: RigorousWindowMaps, event_id: str, *, which: str = "pk", include_no_margin: bool = False
                ) -> tuple[float, str | None]:
    j = list(maps.event_ids).index(event_id)
    k = maps.k_pickup if which == "pk" else maps.k_trip
    best, arg = -math.inf, None
    for i, a in enumerate(maps.asset_ids):
        cls = str(maps.classes[i, j])
        if cls != "certified":
            continue
        if float(k[i, j]) > best:
            best, arg = float(k[i, j]), a
    return (best if arg is not None else float("nan")), arg


def no_margin_of(maps: RigorousWindowMaps) -> list[str]:
    return [a for a, h in zip(maps.asset_ids, maps.margins_pu) if not float(h) > 0.0]


def verdict_excluding(p: Any, exclude: Sequence[str]) -> str:
    ex = set(exclude)
    cls = {c["asset_class"] for a, c in p.classes.items() if a not in ex}
    if any(a not in ex for a in p.predicted_trips):
        return "trip_predicted"
    for k in ("pickup_predicted", "risky", "data_limited"):
        if k in cls:
            return k
    return "certified_safe"


def screen_event_rows(store: ScreenStore, case: ArmedCase, relay: RelaySetting, maps: RigorousWindowMaps,
                      preds: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for e in store.event_ids:
        r = store.responses.get(e)
        in_maps = e in maps.event_ids
        kpk, arg = nonself_max(maps, e, which="pk") if in_maps else (float("nan"), None)
        ktr, _ = nonself_max(maps, e, which="tr") if in_maps else (float("nan"), None)
        n_ge1 = n_ge = 0
        if in_maps:
            j = list(maps.event_ids).index(e)
            cert = [i for i in range(len(maps.asset_ids)) if str(maps.classes[i, j]) == "certified"]
            n_ge1 = sum(1 for i in cert if maps.k_pickup[i, j] >= 1.0)
            n_ge = sum(1 for i in cert if maps.k_pickup[i, j] >= 1.0 - EPSILON)
        diag = r.diagnostics.to_dict() if r is not None else {}
        mode = store.mode_diagnostics.get(f"post_{e}" if (r is not None and r.kind == "switch") else "pre_event", {})
        p = preds.get(e)
        tm = dict(r.timing_s) if r is not None else {}
        ev = case.event(e)
        rows.append({
            "slug": store.slug, "relay": relay.name, "event_id": e, "family": store.families.get(e, ""),
            "kind": ev.kind, "status": "ok" if r is not None else "failed", "failure_reason": store.failed.get(e, ""),
            "disconnects_assets": ";".join(ev.disconnects_assets), "n_mutations": len(ev.mutations),
            "max_k_pk_nonself": kpk, "argmax_asset": arg or "", "max_k_tr_nonself": ktr,
            "n_assets_k_pk_ge1": n_ge1, "n_assets_k_pk_ge_1meps": n_ge,
            "verdict": p.verdict if p is not None else "",
            "verdict_excl_no_margin": verdict_excluding(p, no_margin_of(maps)) if p is not None else "",
            "predicted_trips": ";".join(p.predicted_trips) if p else "",
            "fixed_point": ";".join(p.fixed_point) if p else "",
            "runtime_s": store.event_runtime_s.get(e, float("nan")),
            "t_jump_newton_s": tm.get("jump_newton", 0.0), "t_post_mode_s": tm.get("post_mode", 0.0),
            "t_residual_s": tm.get("residual_and_restore", 0.0), "t_propagate_s": tm.get("propagate", 0.0),
            "t_envelopes_s": tm.get("envelopes", 0.0),
            "newton_iterations": r.newton_iterations if r is not None else "",
            "newton_residual": r.newton_residual if r is not None else "",
            "mode_id": diag.get("mode_id", ""), "n_states_full": mode.get("n_states_full", ""),
            "n_states": diag.get("n_states", ""), "n_algebraic": diag.get("n_algebraic", ""),
            "max_real_eig": diag.get("max_real_eigenvalue", ""),
            "max_real_eig_removed_block": mode.get("max_real_eigenvalue_removed_block", ""),
            "unstable_flag": diag.get("unstable_flag", ""), "stability": diag.get("stability", ""),
            "gy_condition": diag.get("gy_condition_estimate", ""), "log_norm_mu": diag.get("log_norm_mu", ""),
            "restoration_max_err": max(store.restoration.get(e, {"x": float("nan")}).values()),
            "description": ev.description,
        })
    return rows


def prediction_rows(store: ScreenStore, relay: RelaySetting, preds: Mapping[str, Any], no_margin: Sequence[str] = ()
                    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    arows, srows = [], []
    for sid, p in preds.items():
        cnt = Counter(c["asset_class"] for c in p.classes.values())
        srows.append({
            "slug": store.slug, "relay": relay.name, "seed_id": sid, "family": store.families.get(sid, ""),
            "verdict": p.verdict, "verdict_excl_no_margin": verdict_excluding(p, no_margin),
            "no_margin_assets": ";".join(no_margin),
            "max_k_pk_nonself": p.max_k_pk_nonself, "argmax_asset": p.argmax_asset or "",
            "max_k_tr_nonself": p.max_k_tr_nonself, "excluded_assets": ";".join(p.excluded_assets),
            "fixed_point_size": len(p.fixed_point), "fixed_point": ";".join(p.fixed_point),
            "predicted_trip_size": len(p.predicted_trips), "predicted_trips": ";".join(p.predicted_trips),
            "refined_predicted_trips": ";".join(getattr(p, "refined_predicted_trips", ()) or ()),
            "refined_fixed_point": ";".join(getattr(p, "refined_fixed_point", ()) or ()),
            "refinement_iterations": p.refinement_iterations,
            "refinement_changed": p.refinement_changed, "refinement_note": p.refinement_note, "n_layers": p.n_layers,
            "first_secondary": p.first_secondary or "", "first_secondary_time_s": p.first_secondary_time_s,
            **{f"n_{k}": cnt.get(k, 0) for k in ("pickup_predicted", "risky", "certified", "data_limited")},
            "runtime_s": p.runtime_s,
        })
        for a, c in p.classes.items():
            arows.append({"slug": store.slug, "relay": relay.name, "seed_id": sid, "family": store.families.get(sid, ""),
                          "asset_id": a, **{k: (";".join(v) if isinstance(v, list) else v) for k, v in c.items()}})
    return arows, srows


_CASE_CACHE: dict[str, ArmedCase] = {}


def _get_case(path: str) -> ArmedCase:
    if path not in _CASE_CACHE:
        _CASE_CACHE.clear()
        _CASE_CACHE[path] = load_case(Path(path))
    return _CASE_CACHE[path]


def tds_job(job: Mapping[str, Any]) -> dict[str, Any]:
    import traceback

    from pa_dvsa.tds_runner import run_protection_tds

    t_job = time.perf_counter()
    out: dict[str, Any] = {"job_id": job["job_id"], "slug": job["slug"], "relay": job.get("relay"), "kind": job.get("kind")}
    try:
        case = _get_case(job["case_path"])
        if case.fingerprint != job["fingerprint"]:
            raise RuntimeError("armed-case fingerprint mismatch between the main process and the worker")
        fac = SnapshotFactory(case)
        check = {"s": None}

        def factory() -> Any:
            ss = fac()
            if job.get("verify_fingerprint", True):
                check["s"] = arrays_digest(snapshot_arrays(ss)) == case.fingerprint
            return ss

        seed = case.event(job["seed_id"]) if job.get("seed_id") else None
        relay = RELAYS[job["relay"]]
        blocked = set(job.get("blocked_assets", ()))
        pickup = {b.asset_id: (float("inf") if b.asset_id in blocked else relay.pickup_pu) for b in case.blocks}
        dwell = {b.asset_id: relay.dwell_s for b in case.blocks}
        trip_events = {e.event_id: e for e in case.events}
        with QuietAndes(True):
            res = run_protection_tds(
                factory, seed, case.blocks, relays_enabled=bool(job.get("relays_enabled", True)),
                pickup_override=pickup, dwell_override=dwell, trip_events=trip_events,
                t_seed=float(job.get("t_seed", T_SEED_S)), window_s=float(job.get("window_s", relay.horizon_s)),
                t_end=job.get("t_end"), tstep=float(job.get("tstep", TDS_TSTEP_S)), tol=float(job.get("tol", TDS_TOL)),
                record_traces=bool(job.get("record_traces", True)),
                dropout_ratio=float(job.get("dropout_ratio", 1.0)),
            )
        d = res.to_dict()
        out.update(d)
        out["factory_fingerprint_ok"] = check["s"]
        out["build_s"] = fac.last_build_s
        out["sim_s"] = float(res.runtime_s - fac.last_build_s)
        if res.traces is not None:
            out["trace_t"] = np.asarray(res.traces["t"], dtype=float)
            out["trace_v"] = np.asarray(res.traces["v_collector"], dtype=float)
            out["trace_asset_ids"] = [str(a) for a in res.traces["asset_ids"]]
        out["blocked_assets"] = sorted(blocked)
        out["error"] = None
    except Exception as exc:
        out.update({"status": "job_error", "converged": False, "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc()})
    out["job_wall_s"] = time.perf_counter() - t_job
    out["job"] = dict(job)
    return out


def tds_assets_for_scoring(res: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    return {a["asset_id"]: {"trip_rel_s": a.get("trip_rel_s"), "first_pickup_rel_s": a.get("first_pickup_rel_s"),
                            "max_v_window_pu": a.get("max_v_window_pu"), "max_v_after_pu": a.get("max_v_after_pu")}
            for a in res.get("assets", [])}


def score_validation(store: ScreenStore, case: ArmedCase, relay: RelaySetting, preds: Mapping[str, Any],
                     tds: Mapping[str, Mapping[str, Any]], *, no_margin: Sequence[str], prediction: str = "paper"
                     ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    from pa_dvsa.revised_studies.scoring import aggregate, score_seed

    rows_all: list[dict[str, Any]] = []
    seeds: list[dict[str, Any]] = []
    nm = set(no_margin)
    for sid, p in preds.items():
        res = tds.get(sid)
        if res is None:
            continue
        ok = bool(res.get("converged")) and res.get("status") == "ok"
        rows, summ = score_seed(seed_id=sid, family=store.families.get(sid, ""), self_assets=p.excluded_assets,
                                screen=p.classes, tds_assets=tds_assets_for_scoring(res), tds_ok=ok,
                                window_s=relay.horizon_s)
        for r in rows:
            r.update({"slug": store.slug, "relay": relay.name, "prediction": prediction,
                      "no_margin_blocked": r["asset_id"] in nm, "tds_status": res.get("status")})
        pickups = [a["asset_id"] for a in res.get("assets", []) if a.get("first_pickup_rel_s") is not None
                   and a.get("first_pickup_rel_s") <= relay.horizon_s + 1e-9 and a["asset_id"] not in p.excluded_assets]
        late = [a["asset_id"] for a in res.get("assets", []) if a.get("trip_rel_s") is not None and a["trip_rel_s"] > relay.horizon_s + 1e-9]
        summ.update({
            "slug": store.slug, "relay": relay.name, "prediction": prediction, "tds_status": res.get("status"), "tds_err_msg": res.get("err_msg", ""),
            "tds_runtime_s": res.get("job_wall_s"), "tds_build_s": res.get("build_s"), "tds_sim_s": res.get("sim_s"),
            "tds_trip_order": ";".join(res.get("trip_order", [])), "tds_pre_seed_max_dv_pu": res.get("pre_seed_max_dv_pu"),
            "tds_pickups_in_window": ";".join(pickups), "late_trips": ";".join(late),
            "factory_fingerprint_ok": res.get("factory_fingerprint_ok"),
            "max_k_pk_nonself": p.max_k_pk_nonself, "max_k_tr_nonself": p.max_k_tr_nonself,
            "screen_verdict": p.verdict, "screen_verdict_excl_no_margin": verdict_excluding(p, sorted(nm)),
            "certified_no_secondary_excl_no_margin": all(c["asset_class"] == "certified" for a, c in p.classes.items() if a not in nm),
            "seed_unsafe_FN_excl_no_margin": bool(ok and len(summ["tds_set"]) > 0
                                                  and all(c["asset_class"] == "certified" for a, c in p.classes.items() if a not in nm)),
            "screen_runtime_s": store.event_runtime_s.get(sid),
            "cascade_runtime_s": p.runtime_s, "blocked_assets": ";".join(sorted(nm)),
        })
        rows_all.extend(rows)
        seeds.append(summ)
    fams = sorted({s["family"] for s in seeds})
    grp = {"slug": store.slug, "relay": relay.name, "prediction": prediction}
    agg = [aggregate([s for s in seeds if s["family"] == f], group={**grp, "family": f}) for f in fams]
    agg.append(aggregate(seeds, group={**grp, "family": "ALL"}))
    return rows_all, seeds, agg


def conservatism_rows(store: ScreenStore, relay: RelaySetting, maps: RigorousWindowMaps,
                      tds: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    col = {e: j for j, e in enumerate(maps.event_ids)}
    for sid, res in tds.items():
        if sid not in col or res.get("status") != "ok" or res.get("trip_order"):
            continue
        j = col[sid]
        by = {a["asset_id"]: a for a in res.get("assets", [])}
        for i, a in enumerate(maps.asset_ids):
            if str(maps.classes[i, j]) == "self" or not by.get(a, {}).get("active", False):
                continue
            vmax = by[a].get("max_v_window_pu")
            if vmax is None:
                continue
            rise = float(vmax) - float(store.z0[i])
            bound = float(maps.pickup_excursion_pu[i, j])
            samp = float(maps.sampled_peak_pu[i, j])
            rows.append({"slug": store.slug, "relay": relay.name, "seed_id": sid, "family": store.families.get(sid, ""),
                         "asset_id": a, "margin_pu": float(maps.margins_pu[i]),
                         "screen_bound_mpu": bound * 1e3, "screen_sampled_mpu": samp * 1e3, "tds_max_rise_mpu": rise * 1e3,
                         "bound_minus_tds_mpu": (bound - rise) * 1e3,
                         "k_pk": float(maps.k_pickup[i, j]),
                         "k_tds": (max(rise, 0.0) / float(maps.margins_pu[i])) if float(maps.margins_pu[i]) > 0 else float("nan")})
    return rows


def conservatism_summary(rows: Sequence[Mapping[str, Any]], *, min_rise_mpu: float = 1.0) -> list[dict[str, Any]]:
    out = []
    keys = sorted({(r["slug"], r["relay"], r["family"]) for r in rows}) + sorted({(r["slug"], r["relay"], "ALL") for r in rows})
    for slug, relay, fam in keys:
        rr = [r for r in rows if r["slug"] == slug and r["relay"] == relay and (fam == "ALL" or r["family"] == fam)]
        d = np.array([r["bound_minus_tds_mpu"] for r in rr])
        big = [r for r in rr if r["tds_max_rise_mpu"] >= min_rise_mpu]
        ratio = np.array([r["screen_bound_mpu"] / r["tds_max_rise_mpu"] for r in big]) if big else np.zeros(0)
        out.append({"slug": slug, "relay": relay, "family": fam, "n_pairs": len(rr),
                    "n_bound_below_tds": int(np.sum(d < 0)), "min_bound_minus_tds_mpu": float(d.min()) if d.size else float("nan"),
                    "median_bound_minus_tds_mpu": float(np.median(d)) if d.size else float("nan"),
                    "max_bound_minus_tds_mpu": float(d.max()) if d.size else float("nan"),
                    "n_pairs_rise_ge_1mpu": len(big),
                    "median_bound_over_tds": float(np.median(ratio)) if ratio.size else float("nan"),
                    "p90_bound_over_tds": float(np.percentile(ratio, 90)) if ratio.size else float("nan"),
                    "min_bound_over_tds": float(ratio.min()) if ratio.size else float("nan")})
    return out


def accuracy_rows(store: ScreenStore, res: Mapping[str, Any], event_id: str) -> list[dict[str, Any]]:
    r = store.responses[event_id]
    t = np.asarray(res["trace_t"], dtype=float)
    v = np.asarray(res["trace_v"], dtype=float)
    ts = float(res["t_seed_s"])
    k_pre = int(np.flatnonzero(np.isclose(t, ts, atol=1e-12))[0])
    v_pre = v[k_pre]
    post_t = t[k_pre + 1:] - ts
    post_v = v[k_pre + 1:] - v_pre
    grid = r.times_s
    rows = []
    t_end = float(post_t[-1])
    for i, a in enumerate(store.asset_ids):
        if a in (res.get("inactive_assets") or []):
            continue
        tds_i = np.interp(grid[grid <= t_end + 1e-12], post_t, post_v[:, i])
        scr_i = r.values_pu[: tds_i.size, i]
        g = grid[: tds_i.size]
        err = np.abs(scr_i - tds_i)

        def emax(tmax: float) -> float:
            m = g <= tmax + 1e-12
            return float(np.max(err[m])) * 1e3 if np.any(m) else float("nan")

        k20 = int(np.argmin(np.abs(g - 0.02)))
        m1 = g <= 1.0 + 1e-12
        rows.append({
            "slug": store.slug, "event_id": event_id, "family": store.families.get(event_id, ""), "asset_id": a,
            "jump_screen_mpu": float(r.jump_pu[i]) * 1e3, "jump_tds_mpu": float(post_v[0, i]) * 1e3,
            "dz20_screen_mpu": float(scr_i[k20]) * 1e3, "dz20_tds_mpu": float(tds_i[k20]) * 1e3,
            "err_0_0p3_mpu": emax(0.3), "err_0_1_mpu": emax(1.0), "err_0_2_mpu": emax(2.0),
            "max_dz_0_1_screen_mpu": float(np.max(scr_i[m1])) * 1e3, "max_dz_0_1_tds_mpu": float(np.max(tds_i[m1])) * 1e3,
            "bound_0_1_mpu": peak_bound(r.envelopes[i], 1.0) * 1e3,
            "min_dz_0_1_screen_mpu": float(np.min(scr_i[m1])) * 1e3, "min_dz_0_1_tds_mpu": float(np.min(tds_i[m1])) * 1e3,
        })
    return rows


def jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return jsonable(obj.tolist())
    if isinstance(obj, (np.floating,)):
        obj = float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, complex):
        return [obj.real, obj.imag]
    if isinstance(obj, float) and not math.isfinite(obj):
        return None if math.isnan(obj) else ("inf" if obj > 0 else "-inf")
    if hasattr(obj, "to_dict"):
        return jsonable(obj.to_dict())
    return obj


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(obj), indent=2) + "\n", encoding="utf-8")


def _cell(v: Any) -> Any:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (list, tuple)):
        return ";".join(str(x) for x in v)
    if isinstance(v, float) and math.isnan(v):
        return "nan"
    if v is None:
        return ""
    return v


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: _cell(r.get(k)) for k in keys})


def fig9_family_rows(event_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    out = []
    keys = sorted({(r["slug"], r["relay"], r["family"]) for r in event_rows if not str(r["family"]).startswith("base_state")})
    for slug, relay, fam in keys:
        rr = [r for r in event_rows if (r["slug"], r["relay"], r["family"]) == (slug, relay, fam)]
        ks = [float(r["max_k_pk_nonself"]) for r in rr if isinstance(r["max_k_pk_nonself"], float) and math.isfinite(r["max_k_pk_nonself"])]
        worst = max(ks) if ks else float("nan")
        arg = next((r for r in rr if ks and r["max_k_pk_nonself"] == worst), None)
        out.append({"slug": slug, "relay": relay, "family": fam, "n_events": len(rr), "n_failed": sum(1 for r in rr if r["status"] != "ok"),
                    "worst_k_pk_nonself": worst, "worst_event": arg["event_id"] if arg else "",
                    "worst_asset": arg["argmax_asset"] if arg else "",
                    "n_events_k_pk_ge1": sum(1 for k in ks if k >= 1.0),
                    "n_events_k_pk_ge_1meps": sum(1 for k in ks if k >= 1.0 - EPSILON),
                    "median_k_pk_nonself": float(np.median(ks)) if ks else float("nan")})
    return out


__all__ = [
    "ArmedCase", "PrunedEventResponseEngine", "RELAYS", "RelaySetting", "ScreenStore", "SnapshotFactory",
    "accuracy_rows", "arrays_diff", "arrays_digest", "build_armed_case", "build_engine", "corrected_event_library",
    "fig9_family_rows", "jump_branch_check", "load_case", "output_reachable_states", "predictions", "restore_arrays",
    "run_screen", "save_case", "score_validation", "screen_event_rows", "screen_k_rows", "slug_of", "snapshot_arrays",
    "tds_job", "verify_factory", "window_maps",
]
