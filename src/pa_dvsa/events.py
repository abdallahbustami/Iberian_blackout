"""Event, control and protected-plant specifications shared by the screening and the time-domain simulations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from .data_model import ProfileKind, TimeProfile

_FORBIDDEN_TDS_ATTRS = {("PQ", "p0"), ("PQ", "q0"), ("PV", "p0"), ("PV", "q0"), ("PV", "u"), ("Slack", "u")}


class EventSpecError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class Mutation:
    model: str
    device_idx: Any
    attr: str
    value: float

    def __post_init__(self) -> None:
        if (self.model, self.attr) in _FORBIDDEN_TDS_ATTRS:
            raise EventSpecError(
                f"{self.model}.{self.attr} is ignored by ANDES after TDS init; use a supported attribute"
            )
        if not np.isfinite(float(self.value)):
            raise EventSpecError("Mutation.value must be finite")


@dataclass(frozen=True, slots=True)
class EventSpec:
    event_id: str
    family: str
    kind: str
    mutations: tuple[Mutation, ...]
    profile: TimeProfile = field(default_factory=lambda: TimeProfile(ProfileKind.STEP))
    description: str = ""
    disconnects_assets: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.kind not in {"switch", "param"}:
            raise EventSpecError("EventSpec.kind must be 'switch' or 'param'")
        if not self.mutations:
            raise EventSpecError("EventSpec needs at least one mutation")
        has_u = any(m.attr == "u" for m in self.mutations)
        if has_u and self.kind != "switch":
            raise EventSpecError(f"{self.event_id}: status mutations require kind='switch'")
        if not has_u and self.kind != "param":
            raise EventSpecError(f"{self.event_id}: parameter-only events require kind='param'")
        if has_u and ProfileKind(self.profile.kind) != ProfileKind.STEP:
            raise EventSpecError(f"{self.event_id}: switch events must use a STEP profile")


@dataclass(frozen=True, slots=True)
class ControlSpec:
    control_id: str
    control_class: str
    mutations: tuple[Mutation, ...]
    envelope: TimeProfile
    alpha_max: float = 1.0
    mvar_per_unit: float | None = None
    cost: float = 1.0
    description: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if any(m.attr == "u" for m in self.mutations):
            raise EventSpecError("controls must be continuous (parameter) actions")
        if not (self.alpha_max >= 0.0):
            raise EventSpecError("alpha_max must be nonnegative")


@dataclass(frozen=True, slots=True)
class ProtectedBlock:
    asset_id: str
    collector_bus: Any
    hv_bus: Any
    pvd1_idx: Any
    gsu_line_idx: Any
    tap: float
    gsu_x_pu_sysbase: float
    p_mw: float
    q_mvar: float
    sn_mva: float
    pickup_pu: float = 1.10
    dwell_s: float = 0.05
    z0_pu: float = float("nan")
    trip_event_id: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)


def lag_envelope(delay_s: float, tau_s: float, *, horizon_s: float = 2.0, dt_s: float = 1e-3) -> TimeProfile:
    if delay_s < 0 or tau_s < 0:
        raise EventSpecError("delay and tau must be nonnegative")
    times = np.arange(0.0, horizon_s + 0.5 * dt_s, dt_s)
    if tau_s == 0.0:
        values = (times >= delay_s).astype(float)
    else:
        values = np.where(times >= delay_s, 1.0 - np.exp(-(times - delay_s) / tau_s), 0.0)
    return TimeProfile(
        ProfileKind.SAMPLED,
        samples=tuple((float(t), float(v)) for t, v in zip(times, values)),
        description=f"delay {delay_s:g} s, first-order tau {tau_s:g} s",
    )


def profile_value(profile: TimeProfile, t_rel_s: float) -> float:
    kind = ProfileKind(profile.kind)
    start = float(profile.start_s)
    final = 1.0 if profile.final_value is None else float(profile.final_value)
    if kind == ProfileKind.STEP:
        return final if t_rel_s >= start - 1e-12 else 0.0
    if kind == ProfileKind.RAMP:
        if t_rel_s < start:
            return 0.0
        dur = float(profile.duration_s or 0.0)
        return final if dur <= 0 else final * min(max((t_rel_s - start) / dur, 0.0), 1.0)
    if kind in {ProfileKind.SAMPLED, ProfileKind.PIECEWISE_CONSTANT}:
        samples = profile.samples
        if not samples:
            return 0.0
        ts = np.array([s[0] for s in samples])
        vs = np.array([s[1] for s in samples])
        if kind == ProfileKind.SAMPLED:
            return float(np.interp(t_rel_s, ts, vs, left=vs[0], right=vs[-1]))
        k = int(np.searchsorted(ts, t_rel_s, side="right") - 1)
        return float(vs[max(k, 0)])
    raise EventSpecError(f"unsupported profile kind {kind}")


def _position(model: Any, device_idx: Any) -> int:
    idx = list(model.idx.v)
    for candidate in (device_idx, str(device_idx)):
        if candidate in idx:
            return idx.index(candidate)
    as_str = [str(v) for v in idx]
    if str(device_idx) in as_str:
        return as_str.index(str(device_idx))
    raise EventSpecError(f"{model.class_name} has no device {device_idx!r}")


def _array(system: Any, mutation: Mutation) -> tuple[np.ndarray, int]:
    model = getattr(system, mutation.model, None)
    if model is None:
        raise EventSpecError(f"system has no model {mutation.model}")
    holder = getattr(model, mutation.attr, None)
    if holder is None or not hasattr(holder, "v"):
        raise EventSpecError(f"{mutation.model} has no attribute {mutation.attr}")
    return holder.v, _position(model, mutation.device_idx)


def read_base_values(system: Any, mutations: Sequence[Mutation]) -> tuple[float, ...]:
    return tuple(float(_array(system, m)[0][_array(system, m)[1]]) for m in mutations)


def apply_mutations(
    system: Any,
    mutations: Sequence[Mutation],
    base_values: Sequence[float],
    fraction: float,
) -> None:
    for mutation, base in zip(mutations, base_values, strict=True):
        arr, pos = _array(system, mutation)
        if mutation.attr == "u":
            arr[pos] = float(mutation.value) if fraction >= 0.5 else float(base)
        else:
            arr[pos] = float(base) + float(fraction) * (float(mutation.value) - float(base))


def restore_mutations(system: Any, mutations: Sequence[Mutation], base_values: Sequence[float]) -> None:
    apply_mutations(system, mutations, base_values, 0.0)


def generator_trip_mutations(
    system: Any,
    syn_idx: Any,
    *,
    include_effective_status: bool = True,
) -> tuple[Mutation, ...]:
    syn_model = None
    for name in ("GENROU", "GENCLS", "GENTPJ"):
        model = getattr(system, name, None)
        if model is not None and model.n and str(syn_idx) in [str(v) for v in model.idx.v]:
            syn_model = name
            break
    if syn_model is None:
        raise EventSpecError(f"synchronous machine {syn_idx!r} not found")
    muts: list[Mutation] = [Mutation(syn_model, syn_idx, "u", 0.0)]
    exciters: list[str] = []
    groups = getattr(system, "groups", {})

    def linked(group: str, ref_attr: str, refs: set[str]) -> list[tuple[str, Any]]:
        out: list[tuple[str, Any]] = []
        grp = groups.get(group) if hasattr(groups, "get") else None
        if grp is None:
            return out
        for mname, model in grp.models.items():
            if not model.n or not hasattr(model, ref_attr):
                continue
            for idx, ref in zip(model.idx.v, getattr(model, ref_attr).v):
                if str(ref) in refs:
                    out.append((mname, idx))
        return out

    for mname, idx in linked("Exciter", "syn", {str(syn_idx)}):
        muts.append(Mutation(mname, idx, "u", 0.0))
        exciters.append(str(idx))
        if include_effective_status and hasattr(getattr(system, mname), "ue"):
            muts.append(Mutation(mname, idx, "ue", 0.0))
    for mname, idx in linked("TurbineGov", "syn", {str(syn_idx)}):
        muts.append(Mutation(mname, idx, "u", 0.0))
        if include_effective_status and hasattr(getattr(system, mname), "ue"):
            muts.append(Mutation(mname, idx, "ue", 0.0))
    for mname, idx in linked("PSS", "avr", set(exciters)):
        muts.append(Mutation(mname, idx, "u", 0.0))
        if include_effective_status and hasattr(getattr(system, mname), "ue"):
            muts.append(Mutation(mname, idx, "ue", 0.0))
    return tuple(muts)
