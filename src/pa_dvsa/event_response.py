"""Relay voltage response of every event and control: the exact algebraic jump of a switching event followed by
the linear response of the post-event mode, or the linear response of the pre-event mode to a parameter change.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import time
from typing import Any, Mapping, Sequence

import numpy as np
from scipy import sparse
from scipy.sparse import linalg as spla

from .andes_adapter import JacobianBlocks, system_jacobians
from .data_model import ProfileKind, TimeProfile
from .events import (
    ControlSpec,
    EventSpec,
    EventSpecError,
    Mutation,
    apply_mutations,
    read_base_values,
    restore_mutations,
)
from .finite_window import (
    ChannelResponse,
    ModeDynamics,
    RigorousEnvelope,
    propagate_channel,
    rigorous_envelope,
)
from .linearization import (
    DEFAULT_MAX_CONDITION,
    AlgebraicFactor,
    LinearDAEModel,
    ReducedLinearModel,
    reduce_linear_dae,
)


class EventResponseError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class OutputSelector:
    output_id: str
    address: int
    domain: str = "y"
    weight: float = 1.0

    def __post_init__(self) -> None:
        if self.domain not in {"x", "y"}:
            raise EventResponseError("OutputSelector.domain must be 'x' or 'y'")
        if int(self.address) < 0:
            raise EventResponseError("OutputSelector.address must be nonnegative")
        if not math.isfinite(float(self.weight)):
            raise EventResponseError("OutputSelector.weight must be finite")

    @classmethod
    def bus_voltage(
        cls, system: Any, bus_idx: Any, output_id: str | None = None, weight: float = 1.0
    ) -> "OutputSelector":
        idx = list(system.Bus.idx.v)
        pos = None
        for cand in (bus_idx, str(bus_idx)):
            if cand in idx:
                pos = idx.index(cand)
                break
        if pos is None:
            as_str = [str(item) for item in idx]
            if str(bus_idx) not in as_str:
                raise EventResponseError(f"bus {bus_idx!r} not found")
            pos = as_str.index(str(bus_idx))
        return cls(
            output_id=output_id or f"V_{bus_idx}",
            address=int(system.Bus.v.a[pos]),
            domain="y",
            weight=float(weight),
        )


@dataclass(frozen=True, slots=True)
class ResponseDiagnostics:
    mode_id: str
    n_states: int
    n_algebraic: int
    max_real_eigenvalue: float | None
    stability: str
    gy_condition_estimate: float | None
    log_norm_mu: float
    removed_algebraic: int
    warnings: tuple[str, ...] = ()

    @property
    def unstable_flag(self) -> bool:
        return self.max_real_eigenvalue is not None and self.max_real_eigenvalue > 1e-6

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode_id": self.mode_id,
            "n_states": self.n_states,
            "n_algebraic": self.n_algebraic,
            "max_real_eigenvalue": self.max_real_eigenvalue,
            "stability": self.stability,
            "gy_condition_estimate": self.gy_condition_estimate,
            "log_norm_mu": self.log_norm_mu,
            "removed_algebraic": self.removed_algebraic,
            "unstable_flag": self.unstable_flag,
            "warnings": list(self.warnings),
        }


@dataclass
class EventResponse:
    event_id: str
    family: str
    kind: str
    output_ids: tuple[str, ...]
    times_s: np.ndarray
    values_pu: np.ndarray
    jump_pu: np.ndarray
    profile: TimeProfile
    diagnostics: ResponseDiagnostics
    envelopes: tuple[RigorousEnvelope, ...]
    newton_iterations: int = 0
    newton_residual: float = 0.0
    disconnects_assets: tuple[str, ...] = ()
    timing_s: Mapping[str, float] = field(default_factory=dict)
    channel: ChannelResponse | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def n_outputs(self) -> int:
        return len(self.output_ids)

    def peak_upper(self, output: int) -> float:
        return float(self.envelopes[output].peak_upper)

    def to_dict(self, *, include_waveforms: bool = False) -> dict[str, Any]:
        out: dict[str, Any] = {
            "event_id": self.event_id,
            "family": self.family,
            "kind": self.kind,
            "output_ids": list(self.output_ids),
            "jump_pu": [float(v) for v in self.jump_pu],
            "peak_upper_pu": [float(e.peak_upper) for e in self.envelopes],
            "sampled_max_pu": [float(e.sampled_max) for e in self.envelopes],
            "diagnostics": self.diagnostics.to_dict(),
            "newton_iterations": self.newton_iterations,
            "newton_residual": self.newton_residual,
            "disconnects_assets": list(self.disconnects_assets),
            "timing_s": dict(self.timing_s),
            "metadata": dict(self.metadata),
        }
        if include_waveforms:
            out["times_s"] = [float(t) for t in self.times_s]
            out["values_pu"] = [[float(v) for v in row] for row in self.values_pu]
        return out


@dataclass
class ModeReduction:
    mode_id: str
    model: LinearDAEModel
    factor: AlgebraicFactor
    reduced: ReducedLinearModel
    dynamics: ModeDynamics
    cr: np.ndarray
    timing_s: Mapping[str, float]
    warnings: tuple[str, ...] = ()

    def reduce_forcing(self, df: np.ndarray, dg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.model.index_map is None:
            raise EventResponseError("mode model has no ANDES index map")
        dx, dy = self.model.index_map.map_residual(df, dg)
        if dy.size:
            sol = self.factor.solve(dy)
            b = dx - np.asarray(self.model.ay @ sol).reshape(-1)
            fd = -np.asarray(self.model.cy @ sol).reshape(-1)
        else:
            b = dx
            fd = np.zeros(self.model.n_outputs)
        return np.asarray(b, dtype=float), np.asarray(fd, dtype=float)

    def diagnostics(self) -> ResponseDiagnostics:
        d = self.reduced.diagnostics
        return ResponseDiagnostics(
            mode_id=self.mode_id,
            n_states=int(d.n_states),
            n_algebraic=int(d.n_algebraic),
            max_real_eigenvalue=d.max_real_eigenvalue,
            stability=d.stability,
            gy_condition_estimate=d.gy_condition_estimate,
            log_norm_mu=float(self.dynamics.mu),
            removed_algebraic=len(d.removed_algebraic),
            warnings=tuple(d.warnings) + tuple(self.warnings),
        )


class EventResponseEngine:
    def __init__(
        self,
        system: Any,
        outputs: Sequence[OutputSelector | int],
        *,
        output_ids: Sequence[str] | None = None,
        times_s: Sequence[float] | None = None,
        horizon_s: float = 1.0,
        dt_s: float = 0.005,
        newton_step_tol: float = 1e-10,
        newton_residual_tol: float = 1e-8,
        newton_max_iter: int = 30,
        restore_tol: float = 1e-9,
        dt_atomic_s: float = 2.5e-4,
        max_condition: float = DEFAULT_MAX_CONDITION,
        envelope_tolerance_pu: float = 1e-4,
        min_interval_s: float = 2.5e-4,
        keep_channels: bool = False,
        eigen_dense_limit: int = 2000,
        baseline_residual_tol: float = 1e-6,
        jacobian_corrections: bool = True,
    ) -> None:
        self.system = system
        tds = getattr(system, "TDS", None)
        if tds is None or not getattr(tds, "initialized", False):
            raise EventResponseError("system must be TDS-initialized (TDS.init()) before screening")
        if float(system.dae.t) != 0.0:
            raise EventResponseError("TDS has already been run; use a freshly initialized system")
        self.selectors = self._selectors(outputs, output_ids)
        self.output_ids = tuple(sel.output_id for sel in self.selectors)
        if len(set(self.output_ids)) != len(self.output_ids):
            raise EventResponseError("output IDs must be unique")
        if times_s is None:
            count = int(round(horizon_s / dt_s))
            if not math.isclose(count * dt_s, horizon_s, rel_tol=0, abs_tol=1e-12):
                raise EventResponseError("horizon_s must be a multiple of dt_s")
            grid = np.round(np.arange(count + 1) * dt_s, 12)
        else:
            grid = np.round(np.asarray(times_s, dtype=float), 12)
        if grid.size < 2 or grid[0] != 0.0 or np.any(np.diff(grid) <= 0.0):
            raise EventResponseError("times_s must start at 0 and be strictly increasing")
        self.times_s = grid
        self.horizon_s = float(grid[-1])
        self.newton_step_tol = float(newton_step_tol)
        self.newton_residual_tol = float(newton_residual_tol)
        self.newton_max_iter = int(newton_max_iter)
        self.restore_tol = float(restore_tol)
        self.dt_atomic_s = float(dt_atomic_s)
        self.max_condition = float(max_condition)
        self.envelope_tolerance_pu = float(envelope_tolerance_pu)
        self.min_interval_s = float(min_interval_s)
        self.keep_channels = bool(keep_channels)
        self.eigen_dense_limit = int(eigen_dense_limit)
        self.jacobian_corrections = bool(jacobian_corrections)

        dae = system.dae
        self._n = int(dae.n)
        self._m = int(dae.m)
        self._tf = np.asarray(dae.Tf, dtype=float).reshape(-1)
        self._zero_tf = self._tf == 0.0
        for sel in self.selectors:
            limit = self._m if sel.domain == "y" else self._n
            if sel.address >= limit:
                raise EventResponseError(f"output {sel.output_id} address {sel.address} out of range")
        self._cx, self._cy = self._output_matrices()
        self._custom_event0 = bool(getattr(tds, "custom_event", False))

        t0 = time.perf_counter()
        self.x0 = np.array(dae.x, dtype=float).copy()
        self.y0 = np.array(dae.y, dtype=float).copy()
        self._evaluate(jacobian=True)
        self.f0 = np.array(dae.f, dtype=float).copy()
        self.g0 = np.array(dae.g, dtype=float).copy()
        self._jac0 = self._jacobians()
        self.z0 = self._z(self.x0, self.y0)
        self.baseline_residual = (
            float(np.max(np.abs(self.f0))) if self.f0.size else 0.0,
            float(np.max(np.abs(self.g0))) if self.g0.size else 0.0,
        )
        self.pegged_states = self._pegged_states()
        self.warnings: list[str] = []
        if max(self.baseline_residual) > float(baseline_residual_tol):
            self.warnings.append(
                "operating_point_not_equilibrium: max|f0|=%.3e, max|g0|=%.3e (responses are relative "
                "to the baseline residual)" % self.baseline_residual
            )
        if self.pegged_states:
            self.warnings.append(
                ("antiwindup_pegged_states_frozen_in_linearization:" if self.jacobian_corrections
                 else "antiwindup_pegged_states_at_operating_point(not_frozen):")
                + f"{len(self.pegged_states)}:" + ",".join(self.pegged_states[:10])
            )
        self._pre_mode: ModeReduction | None = None
        self.timing_s: dict[str, float] = {"baseline": time.perf_counter() - t0}

    def _selectors(
        self, outputs: Sequence[OutputSelector | int], output_ids: Sequence[str] | None
    ) -> tuple[OutputSelector, ...]:
        sels: list[OutputSelector] = []
        ids = list(output_ids) if output_ids is not None else None
        if ids is not None and len(ids) != len(outputs):
            raise EventResponseError("output_ids length must match outputs")
        for k, item in enumerate(outputs):
            if isinstance(item, OutputSelector):
                sel = item
                if ids is not None:
                    sel = OutputSelector(str(ids[k]), sel.address, sel.domain, sel.weight)
            else:
                sel = OutputSelector(str(ids[k]) if ids is not None else f"y{int(item)}", int(item))
            sels.append(sel)
        if not sels:
            raise EventResponseError("at least one output is required")
        return tuple(sels)

    def _output_matrices(self) -> tuple[sparse.csc_matrix, sparse.csc_matrix]:
        p = len(self.selectors)
        cx = sparse.lil_matrix((p, self._n))
        cy = sparse.lil_matrix((p, self._m))
        for i, sel in enumerate(self.selectors):
            (cx if sel.domain == "x" else cy)[i, sel.address] = float(sel.weight)
        return cx.tocsc(), cy.tocsc()

    def _z(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return np.array(
            [
                float(sel.weight) * float((x if sel.domain == "x" else y)[sel.address])
                for sel in self.selectors
            ]
        )

    def _pegged_states(self) -> tuple[str, ...]:
        names = list(getattr(self.system.dae, "x_name", []) or [])
        pegged: list[str] = []
        for item in getattr(self.system, "antiwindups", []) or []:
            for key, _, _ in getattr(item, "x_set", []) or []:
                for addr in np.atleast_1d(key):
                    a = int(addr)
                    pegged.append(names[a] if a < len(names) else f"x[{a}]")
        return tuple(sorted(set(pegged)))

    def _evaluate(self, *, jacobian: bool) -> None:
        ss = self.system
        ss.TDS.fg_update(ss.exist.pflow_tds)
        if jacobian:
            ss.j_update(models=ss.exist.pflow_tds)

    def _jacobians(self) -> JacobianBlocks:
        return system_jacobians(self.system, update=False, corrected=self.jacobian_corrections)

    def _set_xy(self, x: np.ndarray, y: np.ndarray) -> None:
        dae = self.system.dae
        dae.x[:] = x
        dae.y[:] = y
        self.system.vars_to_models()

    def _restore(self, mutations: Sequence[Mutation], base: Sequence[float]) -> None:
        restore_mutations(self.system, mutations, base)
        self._set_xy(self.x0, self.y0)
        self._evaluate(jacobian=True)
        self.system.TDS.custom_event = self._custom_event0
        self.assert_restored()

    def assert_restored(self) -> None:
        dae = self.system.dae
        tol = self.restore_tol
        errs = {
            "x": float(np.max(np.abs(np.asarray(dae.x) - self.x0))) if self._n else 0.0,
            "y": float(np.max(np.abs(np.asarray(dae.y) - self.y0))) if self._m else 0.0,
            "f": float(np.max(np.abs(np.asarray(dae.f) - self.f0))) if self._n else 0.0,
            "g": float(np.max(np.abs(np.asarray(dae.g) - self.g0))) if self._m else 0.0,
        }
        jac = self._jacobians()
        for name in ("fx", "fy", "gx", "gy"):
            diff = getattr(jac, name) - getattr(self._jac0, name)
            errs[name] = float(np.max(np.abs(diff.data))) if diff.nnz else 0.0
        bad = {k: v for k, v in errs.items() if not (v <= tol)}
        if bad:
            raise EventResponseError(
                "system restoration failed (baseline not recovered): "
                + ", ".join(f"|d{k}|={v:.3e}" for k, v in bad.items())
            )

    def _solve_jump(self) -> tuple[int, float]:
        return solve_frozen_algebraic(
            self.system,
            zero_tf_mask=self._zero_tf,
            corrected=self.jacobian_corrections,
            max_iter=self.newton_max_iter,
            residual_tol=self.newton_residual_tol,
            step_tol=self.newton_step_tol,
        )

    def _build_mode(self, jac: JacobianBlocks, mode_id: str, *, ramp_blocks: bool) -> ModeReduction:
        t0 = time.perf_counter()
        model = LinearDAEModel.from_andes_jacobians(jac, cx=self._cx, cy=self._cy, mode_id=mode_id)
        factor = AlgebraicFactor(model.gy, names=model.y_names, max_condition=self.max_condition)
        t1 = time.perf_counter()
        reduced = reduce_linear_dae(model, factor=factor, eigen_dense_limit=self.eigen_dense_limit)
        t2 = time.perf_counter()
        dyn = ModeDynamics(reduced.ar, dt_atomic=self.dt_atomic_s, need_ramp_blocks=ramp_blocks)
        t3 = time.perf_counter()
        warnings: list[str] = list(self.warnings)
        return ModeReduction(
            mode_id=mode_id,
            model=model,
            factor=factor,
            reduced=reduced,
            dynamics=dyn,
            cr=reduced.cr.toarray(),
            timing_s={"factor": t1 - t0, "reduce": t2 - t1, "balance": t3 - t2},
            warnings=tuple(warnings),
        )

    @property
    def pre_mode(self) -> ModeReduction:
        if self._pre_mode is None:
            t0 = time.perf_counter()
            self._pre_mode = self._build_mode(self._jac0, "pre_event", ramp_blocks=True)
            self.timing_s["pre_mode"] = time.perf_counter() - t0
        return self._pre_mode

    def _propagate_channel(
        self, mode: ModeReduction, b: np.ndarray, f: np.ndarray, profile: TimeProfile, channel_id: str
    ) -> ChannelResponse:
        return propagate_channel(
            mode.dynamics, mode.cr, b, f, profile, self.times_s, channel_id=channel_id, keep_states=True
        )

    def _finish(
        self,
        *,
        event_id: str,
        family: str,
        kind: str,
        mode: ModeReduction,
        b: np.ndarray,
        f: np.ndarray,
        jump: np.ndarray,
        profile: TimeProfile,
        timing: dict[str, float],
        newton_iterations: int = 0,
        newton_residual: float = 0.0,
        disconnects: Sequence[str] = (),
        metadata: Mapping[str, Any] | None = None,
        release_mode: bool = False,
    ) -> EventResponse:
        t0 = time.perf_counter()
        channel = self._propagate_channel(mode, b, f, profile, event_id)
        t1 = time.perf_counter()
        envelopes = tuple(
            rigorous_envelope(
                channel,
                i,
                self.times_s,
                tolerance_pu=self.envelope_tolerance_pu,
                min_interval_s=self.min_interval_s,
            )
            for i in range(len(self.selectors))
        )
        t2 = time.perf_counter()
        pos = np.searchsorted(channel.times_s, self.times_s - 1e-12)
        values = channel.phi_right[pos, :].copy()
        if not self.keep_channels:
            channel.release()
            channel.mode = None
        timing = dict(timing)
        timing.update({"propagate": t1 - t0, "envelopes": t2 - t1})
        timing["total"] = float(
            sum(v for k, v in timing.items() if k != "total" and not k.startswith("post_mode_"))
        )
        return EventResponse(
            event_id=event_id,
            family=family,
            kind=kind,
            output_ids=self.output_ids,
            times_s=self.times_s.copy(),
            values_pu=values,
            jump_pu=np.asarray(jump, dtype=float),
            profile=profile,
            diagnostics=mode.diagnostics(),
            envelopes=envelopes,
            newton_iterations=int(newton_iterations),
            newton_residual=float(newton_residual),
            disconnects_assets=tuple(disconnects),
            timing_s=timing,
            channel=channel if self.keep_channels else None,
            metadata=dict(metadata or {}),
        )

    def event_response(self, spec: EventSpec) -> EventResponse:
        if spec.kind == "switch":
            return self._switch_response(spec)
        return self._param_response(
            event_id=spec.event_id,
            family=spec.family,
            kind="param",
            mutations=spec.mutations,
            profile=spec.profile,
            disconnects=spec.disconnects_assets,
            metadata=spec.metadata,
        )

    def control_response(self, spec: ControlSpec) -> EventResponse:
        return self._param_response(
            event_id=spec.control_id,
            family=spec.control_class,
            kind="control",
            mutations=spec.mutations,
            profile=spec.envelope,
            disconnects=(),
            metadata={"mvar_per_unit": spec.mvar_per_unit, "alpha_max": spec.alpha_max, **dict(spec.metadata)},
        )

    def responses(self, specs: Sequence[EventSpec | ControlSpec]) -> list[EventResponse]:
        out: list[EventResponse] = []
        for spec in specs:
            if isinstance(spec, ControlSpec):
                out.append(self.control_response(spec))
            else:
                out.append(self.event_response(spec))
        return out

    def _param_response(
        self,
        *,
        event_id: str,
        family: str,
        kind: str,
        mutations: Sequence[Mutation],
        profile: TimeProfile,
        disconnects: Sequence[str],
        metadata: Mapping[str, Any] | None,
    ) -> EventResponse:
        if any(m.attr == "u" for m in mutations):
            raise EventSpecError(f"{event_id}: status mutations must use the switch path")
        mode = self.pre_mode
        t0 = time.perf_counter()
        base = read_base_values(self.system, mutations)
        try:
            apply_mutations(self.system, mutations, base, 1.0)
            self._evaluate(jacobian=False)
            df = np.array(self.system.dae.f, dtype=float) - self.f0
            dg = np.array(self.system.dae.g, dtype=float) - self.g0
        finally:
            self._restore(mutations, base)
        t1 = time.perf_counter()
        b, fd = mode.reduce_forcing(df, dg)
        t2 = time.perf_counter()
        meta = {
            "residual_norm_f": float(np.linalg.norm(df)),
            "residual_norm_g": float(np.linalg.norm(dg)),
            **dict(metadata or {}),
        }
        return self._finish(
            event_id=event_id,
            family=family,
            kind=kind,
            mode=mode,
            b=b,
            f=fd,
            jump=np.zeros(len(self.selectors)),
            profile=profile,
            timing={"residual_and_restore": t1 - t0, "reduce_forcing": t2 - t1},
            disconnects=disconnects,
            metadata=meta,
        )

    def _switch_response(self, spec: EventSpec) -> EventResponse:
        if ProfileKind(spec.profile.kind) != ProfileKind.STEP or float(spec.profile.start_s) != 0.0:
            raise EventSpecError(f"{spec.event_id}: switch events must be a STEP at t = 0")
        final = 1.0 if spec.profile.final_value is None else float(spec.profile.final_value)
        if final != 1.0:
            raise EventSpecError(f"{spec.event_id}: switch events must have unit final value")
        t0 = time.perf_counter()
        base = read_base_values(self.system, spec.mutations)
        try:
            apply_mutations(self.system, spec.mutations, base, 1.0)
            iterations, residual = self._solve_jump()
            t1 = time.perf_counter()
            dae = self.system.dae
            x_plus = np.array(dae.x, dtype=float)
            y_plus = np.array(dae.y, dtype=float)
            df = np.array(dae.f, dtype=float) - self.f0
            dg = np.array(dae.g, dtype=float) - self.g0
            jac_post = self._jacobians()
            jump = self._z(x_plus, y_plus) - self.z0
            mode = self._build_mode(jac_post, f"post_{spec.event_id}", ramp_blocks=False)
            t2 = time.perf_counter()
        finally:
            self._restore(spec.mutations, base)
        t3 = time.perf_counter()
        b, fd = mode.reduce_forcing(df, dg)
        f_eff = fd + jump
        meta = {
            "post_event_residual_f_dyn": float(np.max(np.abs(df[~self._zero_tf]))) if self._n else 0.0,
            **dict(spec.metadata),
        }
        response = self._finish(
            event_id=spec.event_id,
            family=spec.family,
            kind="switch",
            mode=mode,
            b=b,
            f=f_eff,
            jump=jump,
            profile=TimeProfile(ProfileKind.STEP),
            timing={
                "jump_newton": t1 - t0,
                "post_mode": t2 - t1,
                "restore": t3 - t2,
                **{f"post_mode_{k}": v for k, v in mode.timing_s.items()},
            },
            newton_iterations=iterations,
            newton_residual=residual,
            disconnects=spec.disconnects_assets,
            metadata=meta,
        )
        return response


def solve_frozen_algebraic(
    system: Any,
    *,
    zero_tf_mask: np.ndarray | None = None,
    max_iter: int = 30,
    residual_tol: float = 1e-8,
    step_tol: float = 1e-10,
    corrected: bool = True,
) -> tuple[int, float]:
    dae = system.dae
    z = (
        np.asarray(dae.Tf, dtype=float).reshape(-1) == 0.0
        if zero_tf_mask is None
        else np.asarray(zero_tf_mask, dtype=bool)
    )
    nz = int(np.count_nonzero(z))
    last_step = math.inf
    rmax = math.inf
    for it in range(int(max_iter) + 1):
        system.TDS.fg_update(system.exist.pflow_tds)
        system.j_update(models=system.exist.pflow_tds)
        res = np.concatenate((np.asarray(dae.f)[z], np.asarray(dae.g)))
        if not np.all(np.isfinite(res)):
            raise EventResponseError("non-finite residual during algebraic jump solve")
        rmax = float(np.max(np.abs(res))) if res.size else 0.0
        if rmax <= residual_tol and (last_step <= step_tol or rmax <= 1e-13):
            return it, rmax
        if it == int(max_iter):
            break
        jac = system_jacobians(system, update=False, corrected=corrected)
        fx, fy, gx, gy = jac.fx, jac.fy, jac.gx, jac.gy
        if nz:
            aug = sparse.bmat([[fx[z][:, z], fy[z]], [gx[:, z], gy]], format="csc")
        else:
            aug = gy.tocsc()
        try:
            step = spla.splu(aug).solve(-res)
        except RuntimeError as exc:
            raise EventResponseError(f"singular post-event algebraic Jacobian: {exc}") from exc
        if not np.all(np.isfinite(step)):
            raise EventResponseError("non-finite Newton step in algebraic jump solve")
        x = np.array(dae.x, dtype=float)
        y = np.array(dae.y, dtype=float)
        x[z] += step[:nz]
        y += step[nz:]
        dae.x[:] = x
        dae.y[:] = y
        system.vars_to_models()
        last_step = float(np.max(np.abs(step))) if step.size else 0.0
    raise EventResponseError(
        f"algebraic jump Newton did not converge in {max_iter} iterations "
        f"(max residual {rmax:.3e}, last step {last_step:.3e})"
    )
