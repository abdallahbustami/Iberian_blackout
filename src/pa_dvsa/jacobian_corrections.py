"""Derivatives that ANDES omits from its Jacobians (PVD1 voltage droop and current limits), added to the
linear model.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

Triplets = dict[str, tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]]


def _empty() -> Triplets:
    return {name: ([], [], []) for name in ("fx", "fy", "gx", "gy")}


def _add(trip: Triplets, block: str, rows: np.ndarray, cols: np.ndarray, vals: np.ndarray) -> None:
    rows = np.asarray(rows, dtype=int).reshape(-1)
    cols = np.asarray(cols, dtype=int).reshape(-1)
    vals = np.asarray(vals, dtype=float).reshape(-1)
    keep = vals != 0.0
    if np.any(keep):
        trip[block][0].append(rows[keep])
        trip[block][1].append(cols[keep])
        trip[block][2].append(vals[keep])


def _pvd1(system: Any, trip: Triplets) -> None:
    model = getattr(system, "PVD1", None)
    if model is None or not getattr(model, "n", 0):
        return
    dae = system.dae
    x = np.asarray(dae.x, dtype=float)
    y = np.asarray(dae.y, dtype=float)
    u = np.asarray(model.u.v, dtype=float)

    dqdv = np.asarray(model.dqdv.v, dtype=float)
    zi = np.asarray(model.VQ1.zi, dtype=float) + np.asarray(model.VQ2.zi, dtype=float)
    coef = -u * dqdv * zi
    active = np.flatnonzero(coef != 0.0)
    if active.size:
        xc = np.asarray(model.xc.v, dtype=float)[active]
        a_addr = np.asarray(model.a.a, dtype=int)[active]
        v_addr = np.asarray(model.v.a, dtype=int)[active]
        ip_addr = np.asarray(model.Ipout_y.a, dtype=int)[active]
        iq_addr = np.asarray(model.Iqout_y.a, dtype=int)[active]
        row = np.asarray(model.Qdrp.a, dtype=int)[active]
        v = y[v_addr]
        ang = y[a_addr]
        ip = x[ip_addr]
        iq = x[iq_addr]
        re = v * np.cos(ang) - xc * iq
        im = v * np.sin(ang) + xc * ip
        w = np.sqrt(re * re + im * im)
        if np.any(w <= 0.0):
            raise FloatingPointError("PVD1 Vcomp = 0: droop derivative undefined")
        c = coef[active]
        _add(trip, "gy", row, v_addr, c * (re * np.cos(ang) + im * np.sin(ang)) / w)
        _add(trip, "gy", row, a_addr, c * v * (-re * np.sin(ang) + im * np.cos(ang)) / w)
        _add(trip, "gx", row, ip_addr, c * xc * im / w)
        _add(trip, "gx", row, iq_addr, c * (-xc * re) / w)

    s0 = np.asarray(model.SWPQ.s0, dtype=float)
    s1 = np.asarray(model.SWPQ.s1, dtype=float)
    for limit_var, sq_service, cmd_var, switch in (
        ("Ipmax", "Ipmaxsq", "Iqcmd_y", s0),
        ("Iqmax", "Iqmaxsq", "Ipcmd_y", s1),
    ):
        sq = np.asarray(getattr(model, sq_service).v, dtype=float)
        cmd_addr = np.asarray(getattr(model, cmd_var).a, dtype=int)
        cmd = y[cmd_addr]
        ok = (switch != 0.0) & (sq > 0.0)
        if not np.any(ok):
            continue
        idx = np.flatnonzero(ok)
        row = np.asarray(getattr(model, limit_var).a, dtype=int)[idx]
        _add(trip, "gy", row, cmd_addr[idx], -switch[idx] * cmd[idx] / np.sqrt(sq[idx]))


CORRECTIONS: dict[str, Callable[[Any, Triplets], None]] = {"PVD1": _pvd1}


def correction_triplets(system: Any) -> Triplets:
    trip = _empty()
    for name, func in CORRECTIONS.items():
        model = getattr(system, name, None)
        if model is not None and getattr(model, "n", 0):
            func(system, trip)
    return trip


def pegged_state_addresses(system: Any) -> np.ndarray:
    addr: list[int] = []
    for item in getattr(system, "antiwindups", []) or []:
        for key, _, _ in getattr(item, "x_set", []) or []:
            addr.extend(int(a) for a in np.atleast_1d(key))
    return np.unique(np.asarray(addr, dtype=int))


def apply_corrections(system: Any, blocks: dict[str, Any], *, freeze_pegged: bool = True) -> dict[str, Any]:
    from scipy import sparse

    trip = correction_triplets(system)
    out = dict(blocks)
    for name, (rows, cols, vals) in trip.items():
        if not rows:
            continue
        mat = blocks[name]
        add = sparse.coo_matrix(
            (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=mat.shape
        )
        out[name] = (mat + add).tocsc()
    if freeze_pegged:
        pegged = pegged_state_addresses(system)
        if pegged.size:
            n = out["fx"].shape[0]
            keep = np.ones(n)
            keep[pegged[(pegged >= 0) & (pegged < n)]] = 0.0
            mask = sparse.diags(keep, format="csc")
            out["fx"] = (mask @ out["fx"]).tocsc()
            out["fy"] = (mask @ out["fy"]).tocsc()
    return out
