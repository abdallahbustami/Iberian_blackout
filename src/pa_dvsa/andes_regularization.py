"""Replaces the zero-time-constant stabilizer filters of ANDES cases by equivalent 1 ms filters, so that the DAE
has index one in both the screening and the time-domain simulation.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import sparse

IEEEST_FAST_FILTER = {"A1": 2.0e-3, "A2": 1.0e-6, "A3": 2.0e-3, "A4": 1.0e-6, "A5": 2.0e-3, "A6": 1.0e-6}


LAG2_FAST = {"T1": 2.0e-3, "T2": 1.0e-6}
LEADLAG2_PASS = {"T1": 2.0e-3, "T2": 1.0e-6, "T3": 2.0e-3, "T4": 1.0e-6}


def regularize_degenerate_blocks(system: Any, *, generic: bool = True) -> list[dict[str, Any]]:
    if getattr(system, "is_setup", False):
        raise RuntimeError("regularize_degenerate_blocks must be called before System.setup()")
    changes: list[dict[str, Any]] = []
    model = getattr(system, "IEEEST", None)
    if model is not None and model.n:
        names = list(IEEEST_FAST_FILTER)
        for i in range(model.n):
            values = [float(getattr(model, name).v[i]) for name in names]
            if all(v == 0.0 for v in values):
                for name, new in IEEEST_FAST_FILTER.items():
                    getattr(model, name).v[i] = new
                changes.append(
                    {
                        "model": "IEEEST",
                        "idx": str(model.idx.v[i]),
                        "reason": "all-zero A1..A6 filter (index-2 DAE); replaced by 1 ms pass-through",
                        "new_values": dict(IEEEST_FAST_FILTER),
                    }
                )
    if generic:
        changes.extend(_regularize_generic_second_order(system))
    return changes


def _param_values(model: Any, param: Any) -> list[float] | None:
    name = getattr(param, "name", None)
    if name is None or name not in getattr(model, "num_params", {}):
        return None
    if model.num_params[name] is not param:
        return None
    try:
        return [float(v) for v in param.v]
    except Exception:
        return None


def _regularize_generic_second_order(system: Any) -> list[dict[str, Any]]:
    changes: list[dict[str, Any]] = []
    for mname, model in system.models.items():
        if not getattr(model, "n", 0):
            continue
        for bname, block in getattr(model, "blocks", {}).items():
            cls = block.__class__.__name__
            if cls == "Lag2ndOrd":
                keys = ("T1", "T2")
                target = LAG2_FAST
            elif cls == "LeadLag2ndOrd":
                keys = ("T1", "T2", "T3", "T4")
                target = LEADLAG2_PASS
            else:
                continue
            params = [getattr(block, k, None) for k in keys]
            values = [_param_values(model, prm) for prm in params]
            for i in range(model.n):
                try:
                    t1 = float(params[0].v[i]) if params[0] is not None else float("nan")
                    t2 = float(params[1].v[i]) if params[1] is not None else float("nan")
                except Exception:
                    continue
                if not (t1 == 0.0 and t2 == 0.0):
                    continue
                record = {
                    "model": mname,
                    "idx": str(model.idx.v[i]),
                    "block": bname,
                    "block_class": cls,
                }
                if any(v is None for v in values):
                    changes.append({**record, "action": "not_regularized",
                                    "reason": "time constants are not plain model parameters"})
                    continue
                rest = [values[k][i] for k in range(2, len(keys))]
                if cls == "LeadLag2ndOrd" and any(v != 0.0 for v in rest):
                    changes.append({**record, "action": "not_regularized",
                                    "reason": "T1=T2=0 with nonzero T3/T4 (improper lead-lag)"})
                    continue
                for k, prm in zip(keys, params):
                    prm.v[i] = target[k]
                changes.append({**record, "action": "regularized",
                                "reason": f"{cls} with zero time constants (index-2 DAE); 1 ms regularization",
                                "new_values": {getattr(prm, 'name', k): target[k] for k, prm in zip(keys, params)}})
    return changes


def algebraic_regularity_report(system: Any, *, max_condition: float | None = None) -> dict[str, Any]:
    from .andes_adapter import system_jacobians
    from .linearization import DEFAULT_MAX_CONDITION, AlgebraicFactor, LinearDAEModel, LinearizationError

    jac = system_jacobians(system, update=True)
    cy = sparse.csc_matrix((1, jac.n_algebraic))
    cx = sparse.csc_matrix((1, jac.n_states))
    try:
        model = LinearDAEModel.from_andes_jacobians(jac, cx=cx, cy=cy)
        factor = AlgebraicFactor(
            model.gy, names=model.y_names,
            max_condition=DEFAULT_MAX_CONDITION if max_condition is None else float(max_condition),
        )
    except LinearizationError as exc:
        return {"ok": False, "size": None, "condition_estimate": None, "message": str(exc)}
    return {
        "ok": True,
        "size": int(model.gy.shape[0]),
        "condition_estimate": float(factor.condition_estimate),
        "removed_decoupled": len(model.removed_algebraic),
        "message": "augmented algebraic Jacobian is regular",
    }


def augmented_algebraic_rank(fx: Any, fy: Any, gx: Any, gy: Any, tf: np.ndarray) -> tuple[int, int]:
    tf = np.asarray(tf, dtype=float).reshape(-1)
    zero = tf == 0.0
    fx = sparse.csc_matrix(fx)
    fy = sparse.csc_matrix(fy)
    gx = sparse.csc_matrix(gx)
    gy = sparse.csc_matrix(gy)
    if np.any(zero):
        aug = sparse.bmat([[fx[zero][:, zero], fy[zero]], [gx[:, zero], gy]]).toarray()
    else:
        aug = gy.toarray()
    return aug.shape[0], int(np.linalg.matrix_rank(aug))
