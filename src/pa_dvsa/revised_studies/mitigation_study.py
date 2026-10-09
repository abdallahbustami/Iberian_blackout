"""Mitigation LPs of the benchmark studies, with all controls and with one control class at a time."""

from __future__ import annotations

import time
from typing import Any, Mapping, Sequence

import numpy as np

from ..events import profile_value, read_base_values
from ..mitigation import solve_mitigation_from_responses
from . import config as C


def unavailable_controls(case: Any, seed: Any) -> dict[str, str]:
    off = {(m.model, str(m.device_idx)) for m in seed.mutations if m.attr == "u" and float(m.value) == 0.0}
    disconnected = set(seed.disconnects_assets)
    out: dict[str, str] = {}
    for c in case.controls:
        if c.control_class == "plant_q_command" and str(c.metadata.get("asset_id")) in disconnected:
            out[c.control_id] = f"plant {c.metadata.get('asset_id')} disconnected by the seed"
            continue
        for m in c.mutations:
            if (m.model, str(m.device_idx)) in off:
                out[c.control_id] = f"{m.model} {m.device_idx} switched off by the seed"
                break
    return out


def solve_lp(
    case: Any,
    seed_response: Any,
    control_responses: Mapping[str, Any],
    output_models: Sequence[Any],
    *,
    restriction: str,
    unavailable: Mapping[str, str],
    uncertainty: Any = None,
    controls: Sequence[Any] | None = None,
) -> dict[str, Any]:
    allowed = C.LP_RESTRICTIONS[restriction]
    specs = list(case.controls if controls is None else controls)
    use = [c for c in specs if (allowed is None or c.control_class in allowed)
           and c.control_id not in unavailable and c.control_id in control_responses]
    rec: dict[str, Any] = {
        "restriction": restriction,
        "n_controls_considered": len(use),
        "controls_excluded_unavailable": sorted(
            c.control_id for c in specs if (allowed is None or c.control_class in allowed) and c.control_id in unavailable
        ),
    }
    if not use:
        rec.update({"status": "no_controls_available", "feasible": False, "eta": float("nan"),
                    "objective": float("nan"), "alphas": {}, "lp_runtime_s": 0.0, "binding": [],
                    "solver_status": "not_run", "data_limited_assets": [], "excluded_assets": []})
        return rec
    t0 = time.perf_counter()
    lp = solve_mitigation_from_responses(
        [seed_response],
        [control_responses[c.control_id] for c in use],
        output_models,
        controls=use,
        uncertainty=uncertainty,
        epsilon=C.EPSILON,
    )
    rt = time.perf_counter() - t0
    res = lp.mitigation_result
    alphas = {c.control_id: 0.0 for c in use}
    for sel in res.selections:
        alphas[sel.control_id] = float(sel.alpha)
    resid = np.asarray(lp.residual_erosion, dtype=float)
    meta = dict(lp.metadata or {})
    binding = [(b.protected_asset_id, float(b.time_s)) for b in res.binding_constraints]
    excluded = set(meta.get("excluded_assets", [])) | set(meta.get("data_limited_assets", []))
    n_constrained = sum(1 for m in output_models if m.asset_id not in excluded)
    status = "solved" if str(res.solver_status).startswith("optimal") else "solver_failed"
    if status == "solved" and n_constrained == 0:
        status = "no_certifiable_asset"
    rec.update({
        "status": status,
        "n_constrained_assets": n_constrained,
        "feasible": bool(res.feasible) and n_constrained > 0,
        "eta": float(res.slack_eta),
        "objective": float(res.objective_value),
        "solver_status": str(res.solver_status),
        "alphas": alphas,
        "lp_runtime_s": rt,
        "binding": binding,
        "residual_max": float(np.max(resid)) if resid.size else float("nan"),
        "data_limited_assets": list(meta.get("data_limited_assets", [])),
        "excluded_assets": list(meta.get("excluded_assets", [])),
        "ineffective_controls": list(res.ineffective_controls),
        "lp_details": lp,
    })
    return rec


def mvar_by_class(case: Any, alphas: Mapping[str, float]) -> dict[str, float]:
    out: dict[str, float] = {}
    for c in case.controls:
        if c.control_id not in alphas:
            continue
        a = float(alphas[c.control_id])
        if c.mvar_per_unit is None:
            out[c.control_class] = float("nan")
            continue
        out[c.control_class] = out.get(c.control_class, 0.0) + a * float(c.mvar_per_unit)
    return out


def avr_dvref(case: Any, alphas: Mapping[str, float]) -> float:
    tot = 0.0
    for c in case.controls:
        if c.control_class == "avr_vref" and c.control_id in alphas:
            m = c.mutations[0]
            base = read_base_values(case.system, [m])[0]
            tot += float(alphas[c.control_id]) * (float(m.value) - base)
    return tot


def available_mvar(case: Any, cls: str, unavailable: Mapping[str, str]) -> float:
    tot, has = 0.0, False
    for c in case.controls:
        if c.control_class != cls or c.control_id in unavailable:
            continue
        if c.mvar_per_unit is None:
            return float("nan")
        tot += float(c.mvar_per_unit) * float(c.alpha_max)
        has = True
    return tot if has else 0.0


def deliverable_mvar(case: Any, cls: str, unavailable: Mapping[str, str], t_rel_s: float) -> float:
    tot, has = 0.0, False
    for c in case.controls:
        if c.control_class != cls or c.control_id in unavailable:
            continue
        if c.mvar_per_unit is None:
            return float("nan")
        tot += float(c.mvar_per_unit) * float(c.alpha_max) * profile_value(c.envelope, float(t_rel_s))
        has = True
    return tot if has else 0.0


def applied_controls(alphas: Mapping[str, float]) -> list[tuple[str, float]]:
    return [(cid, float(a)) for cid, a in sorted(alphas.items()) if float(a) > C.LP_ALPHA_TOL]
