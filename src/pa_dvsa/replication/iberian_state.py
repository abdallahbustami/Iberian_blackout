"""Renewable share, synchronous commitment, export, exciters and interconnection of the stressed state, applied to
the base state, and the inter-area modes of a state.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from pa_dvsa.replication import activsg2000_engine as E

IBERIAN_EXPORT_MW = 870.0
FLEET_SIZE_PATTERN_MW = (120.0, 180.0, 250.0, 320.0, 400.0)
INITIATING_BLOCK = "Granada355"


def area_load_mw(cfg: E.ReplicaConfig | None = None, *, quiet: bool = True) -> float:
    base = cfg or E.ReplicaConfig()
    rs = E._assemble(replace(base, tap_override=base.tap_fixed), armed=False, quiet=quiet,
                     disabled_exciters=(), tap=float(base.tap_fixed))
    return float(rs.base_notes["area_load_mw_before_scaling"]) * float(base.load_scale)


def fleet_blocks(res_share: float, area_generation_mw: float) -> tuple[E.BlockSpec, ...]:
    blocks = list(E.REPORT_BLOCKS)
    target = float(res_share) * float(area_generation_mw)
    total = sum(b.p_mw for b in blocks)
    k = 0
    while total + FLEET_SIZE_PATTERN_MW[k % len(FLEET_SIZE_PATTERN_MW)] <= target + 1e-6:
        p = FLEET_SIZE_PATTERN_MW[k % len(FLEET_SIZE_PATTERN_MW)]
        blocks.append(E.BlockSpec(f"RES{k + 1:02d}", p, None, "fleet", "generic RES fleet block (fixed size pattern)"))
        total += p
        k += 1
    return tuple(blocks)


def net_export_mw(cfg: E.ReplicaConfig, *, quiet: bool = True) -> float:
    rs = E._assemble(replace(cfg, tap_override=cfg.tap_fixed), armed=False, quiet=quiet,
                     disabled_exciters=(), tap=float(cfg.tap_fixed))
    E._setup_and_pflow(rs, quiet=quiet)
    return float(E.area_net_export_mw(rs))


def solve_export_reduction(cfg: E.ReplicaConfig, target_mw: float, *, tol_mw: float = 20.0,
                           quiet: bool = True) -> tuple[float, float]:
    d = 0.0
    export = net_export_mw(replace(cfg, export_reduction_mw=0.0), quiet=quiet)
    d = max(export - float(target_mw), 0.0)
    for _ in range(6):
        export = net_export_mw(replace(cfg, export_reduction_mw=d), quiet=quiet)
        if abs(export - float(target_mw)) <= tol_mw:
            return round(d, 1), export
        d = max(d + (export - float(target_mw)), 0.0)
    raise RuntimeError(f"export target {target_mw} MW not reached (last {export:.0f} MW)")


def iberian_state_config(
    *,
    tie_scale: float = 1.0,
    res_share: float = 0.45,
    commitment_c: float = 1.5,
    export_target_mw: float = IBERIAN_EXPORT_MW,
    exciter_completion: str = "all",
    export_counterpart: str = "rest_load",
    initiating_block: str | None = INITIATING_BLOCK,
    initiating_time_s: float = 15.0,
    quiet: bool = True,
    **overrides: Any,
) -> tuple[E.ReplicaConfig, dict[str, Any]]:
    base = E.ReplicaConfig()
    load = area_load_mw(base, quiet=quiet)
    blocks = fleet_blocks(res_share, load + float(export_target_mw))
    actions = tuple(a for a in E.default_pre_cascade_actions() if a.kind != "interchange_reduction")
    cfg = replace(
        base,
        variant=overrides.pop("variant", f"iberian_state:k{tie_scale:g}"),
        blocks=blocks, zone_size=len(blocks), commitment_c=float(commitment_c), dispatch_rule="must_run",
        tie_impedance_scale=float(tie_scale), exciter_completion=exciter_completion,
        export_counterpart=export_counterpart,
        pre_cascade_actions=actions, initiating_block=initiating_block, initiating_time_s=float(initiating_time_s),
        **overrides,
    )
    d, _ = solve_export_reduction(replace(cfg, tie_impedance_scale=1.0), export_target_mw, quiet=quiet)
    cfg = replace(cfg, export_reduction_mw=d)
    export = net_export_mw(cfg, quiet=quiet)
    for _ in range(4):
        if abs(export - float(export_target_mw)) <= 20.0:
            break
        d = round(max(d + (export - float(export_target_mw)), 0.0), 1)
        cfg = replace(cfg, export_reduction_mw=d)
        export = net_export_mw(cfg, quiet=quiet)
    if abs(export - float(export_target_mw)) > 20.0:
        raise RuntimeError(f"export {export:.0f} MW at tie scale {tie_scale} differs from the target {export_target_mw} MW")
    record = {
        "area_load_mw": load, "res_share": res_share, "fleet_blocks": len(blocks),
        "fleet_mw": sum(b.p_mw for b in blocks), "commitment_c": commitment_c,
        "export_target_mw": export_target_mw, "export_reduction_mw": d, "export_base_case_mw": export,
        "tie_scale": tie_scale, "exciter_completion": exciter_completion, "export_counterpart": export_counterpart,
        "pre_cascade_actions": [a.code for a in actions], "initiating_block": initiating_block,
        "initiating_time_s": initiating_time_s,
    }
    return cfg, record


def interarea_modes(rs: E.ReplicaSystem, sigmas_hz: tuple[float, ...] = (0.05, 0.12, 0.2, 0.3, 0.5, 0.7),
                    k: int = 40) -> list[dict[str, float]]:
    import numpy as np
    from scipy import sparse
    from scipy.sparse.linalg import LinearOperator, eigs, splu

    from pa_dvsa.andes_adapter import system_jacobians

    ss = rs.system
    jb = system_jacobians(ss, update=True)
    n, m = jb.n_states, jb.n_algebraic
    J = sparse.bmat([[jb.fx, jb.fy], [jb.gx, jb.gy]], format="csc").astype(complex)
    Em = sparse.diags(np.concatenate([np.asarray(jb.tf, dtype=float), np.zeros(m)]), format="csc").astype(complex)
    g = ss.GENROU
    on = np.asarray(g.u.v, dtype=float) > 0.5
    w_addr = np.asarray(g.omega.a, dtype=int)
    M = np.asarray(g.M.v, dtype=float)
    in_area = np.array([int(b) in rs.area.buses for b in g.bus.v])
    ma, mr = on & in_area, on & ~in_area
    oos = np.asarray(E.out_of_service_state_mask(ss), dtype=bool)
    out: dict[tuple[float, float], dict[str, float]] = {}
    for s in sigmas_hz:
        sig = 1j * 2.0 * np.pi * s
        lu = splu((J - sig * Em).tocsc())
        op = LinearOperator((n + m, n + m), matvec=lambda v: lu.solve(Em @ v), dtype=complex)
        mu, vec = eigs(op, k=k, which="LM", tol=1e-8, maxiter=5000)
        for j in range(len(mu)):
            if abs(mu[j]) < 1e-12:
                continue
            lam = sig + 1.0 / mu[j]
            if lam.imag <= 0 or not (0.03 < lam.imag / (2 * np.pi) < 2.0):
                continue
            v = vec[:, j]
            if oos.any() and np.max(np.abs(v[:n][oos])) > 0.5 * np.max(np.abs(v[:n])):
                continue
            wv = v[w_addr]
            if np.max(np.abs(wv[on])) < 1e-9 * np.max(np.abs(v)):
                continue
            ca = np.sum(M[ma] * wv[ma]) / M[ma].sum()
            cr = np.sum(M[mr] * wv[mr]) / M[mr].sum()
            rms = float(np.sqrt(np.mean(np.abs(wv[on]) ** 2)))
            out[(round(lam.real, 4), round(lam.imag, 4))] = {
                "f_hz": float(lam.imag / (2 * np.pi)), "zeta": float(-lam.real / abs(lam)), "sigma": float(lam.real),
                "interarea_score": float(abs(ca - cr) / max(rms, 1e-30)),
                "phase_deg": float(np.degrees(np.angle(ca / cr))) if abs(cr) > 0 else float("nan"),
            }
    return sorted(out.values(), key=lambda d: -d["interarea_score"])


def dominant_interarea_mode(modes: list[dict[str, float]]) -> dict[str, float] | None:
    cands = [m for m in modes if abs(m["phase_deg"]) > 120.0]
    return max(cands, key=lambda m: m["interarea_score"]) if cands else None
