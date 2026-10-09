"""Power flow with generator reactive limits enforced by PV/PQ complementarity."""

from __future__ import annotations

from typing import Any

import numpy as np

__all__ = ["damped_pflow", "complementarity_pflow", "damped_frozen_algebraic"]


def damped_pflow(ss: Any, *, tol: float = 1e-8, max_iter: int = 60, min_alpha: float = 1.0 / 1024) -> bool:
    from kvxopt import sparse as kv_sparse

    pf = ss.PFlow
    pf.init()
    dae = ss.dae
    n = int(dae.n)
    pf.niter = 0
    pf.mis = [1.0]
    pf.converged = False

    def resid() -> np.ndarray:
        pf.fg_update()
        return np.concatenate([np.asarray(dae.f, dtype=float), np.asarray(dae.g, dtype=float)])

    r = resid()
    for it in range(max_iter):
        pf.niter = it
        mis = float(np.max(np.abs(r))) if r.size else 0.0
        pf.mis.append(mis)
        if not np.isfinite(mis):
            return False
        if mis < tol:
            pf.converged = True
            pf.x_sol = dae.x.copy()
            pf.y_sol = dae.y.copy()
            ss.exit_code = 0
            return True
        ss.j_update(pf.models)
        pf.solver.worker.new_A = True
        pf.res[:n] = -dae.f[:]
        pf.res[n:] = -dae.g[:]
        pf.A = kv_sparse([[dae.fx, dae.gx], [dae.fy, dae.gy]])
        inc = np.ravel(np.array(pf.solver.solve(pf.A, pf.res)))
        x0, y0 = dae.x.copy(), dae.y.copy()
        n0 = float(np.linalg.norm(r))
        alpha = 1.0
        while True:
            dae.x[:] = x0 + alpha * inc[:n]
            dae.y[:] = y0 + alpha * inc[n:]
            ss.vars_to_models()
            r_new = resid()
            n1 = float(np.linalg.norm(r_new))
            if np.isfinite(n1) and (n1 <= (1.0 - 1e-4 * alpha) * n0 or alpha <= min_alpha):
                break
            alpha *= 0.5
        r = r_new
    ss.exit_code = 1
    return False


def complementarity_pflow(
    ss: Any,
    *,
    tol: float = 1e-8,
    tol_q: float = 1e-4,
    tol_v: float = 1e-4,
    max_outer: int = 60,
    freeze_after: int = 40,
    max_new_pq: int = 10,
) -> tuple[bool, dict[str, Any]]:
    pv = ss.PV
    lim = pv.qlim
    lim.enable = False
    n = int(pv.n)
    qmin = np.asarray(pv.qmin.v, dtype=float)
    qmax = np.asarray(pv.qmax.v, dtype=float)
    on = np.asarray(pv.u.v, dtype=float) > 0.5
    fixed = np.isclose(qmin, qmax)
    zl = np.zeros(n)
    zu = np.where(fixed, 1.0, 0.0)

    def solve(zl_: np.ndarray, zu_: np.ndarray, v0: np.ndarray, a0: np.ndarray) -> bool:
        ss.Bus.v0.v[:] = v0
        ss.Bus.a0.v[:] = a0
        lim.zl[:] = zl_
        lim.zu[:] = zu_
        lim.zi[:] = 1.0 - np.maximum(zl_, zu_)
        return damped_pflow(ss, tol=tol)

    v_last = np.array(ss.Bus.v0.v, dtype=float)
    a_last = np.array(ss.Bus.a0.v, dtype=float)
    if not solve(zl, zu, v_last, a_last):
        return False, {"reason": "Newton power flow failed with every unit in PV mode", "outer": 0}
    v_last, a_last = np.array(ss.Bus.v.v, dtype=float), np.array(ss.Bus.a.v, dtype=float)
    switches = 0
    for it in range(max_outer):
        q = np.asarray(pv.q.v, dtype=float)
        v = np.asarray(pv.v.v, dtype=float)
        vset = np.asarray(pv.v0.v, dtype=float)
        is_pv = on & (zl + zu == 0)
        up = is_pv & (q > qmax + tol_q)
        lo = is_pv & (q < qmin - tol_q)
        frozen = it >= freeze_after
        wrong_u = on & ~fixed & (zu > 0) & (v > vset + tol_v)
        wrong_l = on & ~fixed & (zl > 0) & (v < vset - tol_v)
        back = (wrong_u | wrong_l) & (not frozen)
        if not (up.any() or lo.any() or back.any()):
            return True, {
                "outer": it, "switch_operations": switches, "n_pq": int(np.sum((zl + zu > 0) & ~fixed & on)),
                "back_switching_frozen": bool(frozen),
                "complementarity_residual_units": int(np.sum(wrong_u | wrong_l)),
            }
        viol = np.where(up, q - qmax, 0.0) + np.where(lo, qmin - q, 0.0)
        new = [int(j) for j in np.argsort(-viol)[:max_new_pq] if viol[j] > 0]
        cand = [("back", int(j)) for j in np.flatnonzero(back)] + [("new", j) for j in new]
        m = len(cand)
        while m >= 1:
            zl_t, zu_t = zl.copy(), zu.copy()
            for kind, j in cand[:m]:
                if kind == "back":
                    zl_t[j] = 0.0
                    zu_t[j] = 0.0
                elif up[j]:
                    zu_t[j] = 1.0
                else:
                    zl_t[j] = 1.0
            if solve(zl_t, zu_t, v_last, a_last):
                zl, zu = zl_t, zu_t
                v_last, a_last = np.array(ss.Bus.v.v, dtype=float), np.array(ss.Bus.a.v, dtype=float)
                switches += m
                break
            m //= 2
        if m == 0:
            kind, j = cand[0]
            solve(zl, zu, v_last, a_last)
            return False, {
                "reason": "no power-flow solution after switching a single unit (reactive infeasibility)",
                "outer": it, "failing_unit": str(pv.idx.v[j]), "failing_bus": int(pv.bus.v[j]), "failing_switch": kind,
            }
    return False, {"reason": f"no complementarity solution within {max_outer} passes", "outer": max_outer}


def damped_frozen_algebraic(
    ss: Any, *, max_iter: int = 50, tol: float = 1e-8, min_alpha: float = 1.0 / 1024, corrected: bool = True,
) -> tuple[bool, int, float]:
    from scipy import sparse
    from scipy.sparse import linalg as spla

    from pa_dvsa.andes_adapter import system_jacobians

    dae = ss.dae
    z = np.asarray(dae.Tf, dtype=float).reshape(-1) == 0.0
    nz = int(np.count_nonzero(z))

    def resid() -> np.ndarray:
        ss.TDS.fg_update(ss.exist.pflow_tds)
        return np.concatenate((np.asarray(dae.f, dtype=float)[z], np.asarray(dae.g, dtype=float)))

    r = resid()
    rmax = float(np.max(np.abs(r))) if r.size else 0.0
    for it in range(max_iter):
        rmax = float(np.max(np.abs(r))) if r.size else 0.0
        if not np.isfinite(rmax):
            return False, it, rmax
        if rmax <= tol:
            return True, it, rmax
        ss.j_update(models=ss.exist.pflow_tds)
        jac = system_jacobians(ss, update=False, corrected=corrected)
        if nz:
            aug = sparse.bmat([[jac.fx[z][:, z], jac.fy[z]], [jac.gx[:, z], jac.gy]], format="csc")
        else:
            aug = jac.gy.tocsc()
        try:
            step = spla.splu(aug).solve(-r)
        except RuntimeError:
            return False, it, rmax
        if not np.all(np.isfinite(step)):
            return False, it, rmax
        x0 = np.array(dae.x, dtype=float)
        y0 = np.array(dae.y, dtype=float)
        n0 = float(np.linalg.norm(r))
        alpha = 1.0
        while True:
            x = x0.copy()
            x[z] += alpha * step[:nz]
            dae.x[:] = x
            dae.y[:] = y0 + alpha * step[nz:]
            ss.vars_to_models()
            r_new = resid()
            n1 = float(np.linalg.norm(r_new))
            if np.isfinite(n1) and (n1 <= (1.0 - 1e-4 * alpha) * n0 or alpha <= min_alpha):
                break
            alpha *= 0.5
        r = r_new
    rmax = float(np.max(np.abs(r))) if r.size else 0.0
    return bool(rmax <= tol), max_iter, rmax
