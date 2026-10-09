"""Finite-difference check of every column of the ANDES DAE Jacobian at the operating point."""

from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .andes_adapter import system_jacobians


class JacobianAuditError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class AuditEntry:
    block: str
    row: int
    col: int
    row_label: tuple[str, str, str]
    col_label: tuple[str, str, str]
    analytic: float
    fd: float
    fd_plus: float
    fd_minus: float
    column_scale: float
    tolerance: float
    classification: str
    cause: str

    @property
    def diff(self) -> float:
        return self.fd - self.analytic

    def to_dict(self) -> dict[str, Any]:
        return {
            "block": self.block,
            "row": self.row,
            "col": self.col,
            "row_label": list(self.row_label),
            "col_label": list(self.col_label),
            "analytic": self.analytic,
            "fd": self.fd,
            "fd_plus": self.fd_plus,
            "fd_minus": self.fd_minus,
            "diff": self.diff,
            "column_scale": self.column_scale,
            "tolerance": self.tolerance,
            "classification": self.classification,
            "cause": self.cause,
        }


@dataclass
class AuditResult:
    label: str
    corrected: bool
    n_states: int
    n_algebraic: int
    columns_checked: int
    columns_total: int
    full: bool
    entries: list[AuditEntry]
    runtime_s: float
    models_covered: tuple[str, ...]
    pairs_covered: int
    rel_step: float
    abs_tol: float
    rel_tol: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def groups(self) -> list[dict[str, Any]]:
        out: dict[tuple, dict[str, Any]] = {}
        for e in self.entries:
            key = (e.block, f"{e.row_label[0]}.{e.row_label[1]}", f"{e.col_label[0]}.{e.col_label[1]}", e.classification)
            g = out.setdefault(
                key,
                {
                    "block": e.block,
                    "equation": key[1],
                    "variable": key[2],
                    "classification": e.classification,
                    "count": 0,
                    "max_abs_diff": 0.0,
                    "example": None,
                    "cause": e.cause,
                },
            )
            g["count"] += 1
            if abs(e.diff) >= g["max_abs_diff"]:
                g["max_abs_diff"] = abs(e.diff)
                g["example"] = {
                    "row_device": e.row_label[2],
                    "col_device": e.col_label[2],
                    "analytic": e.analytic,
                    "fd": e.fd,
                    "fd_plus": e.fd_plus,
                    "fd_minus": e.fd_minus,
                    "column_scale": e.column_scale,
                }
        return sorted(out.values(), key=lambda g: (-g["max_abs_diff"], g["equation"]))

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "corrected": self.corrected,
            "n_states": self.n_states,
            "n_algebraic": self.n_algebraic,
            "columns_checked": self.columns_checked,
            "columns_total": self.columns_total,
            "full": self.full,
            "runtime_s": self.runtime_s,
            "models_covered": list(self.models_covered),
            "pairs_covered": self.pairs_covered,
            "rel_step": self.rel_step,
            "abs_tol": self.abs_tol,
            "rel_tol": self.rel_tol,
            "n_mismatches": len(self.entries),
            "groups": self.groups(),
            "metadata": dict(self.metadata),
        }


def _variables(model: Any) -> list[tuple[str, Any, str]]:
    out = []
    for container in ("states", "algebs", "states_ext", "algebs_ext"):
        for name, var in getattr(model, container, {}).items():
            domain = "x" if container.startswith("states") else "y"
            out.append((name, var, domain))
    return out


def address_labels(system: Any) -> tuple[list[tuple[str, str, str]], list[tuple[str, str, str]]]:
    xl: list[tuple[str, str, str]] = [("?", "?", "?")] * int(system.dae.n)
    yl: list[tuple[str, str, str]] = [("?", "?", "?")] * int(system.dae.m)
    for mname, model in system.models.items():
        if not getattr(model, "n", 0):
            continue
        idx = list(model.idx.v)
        for container, labels in (("states", xl), ("algebs", yl)):
            for vname, var in getattr(model, container, {}).items():
                for i, a in enumerate(np.asarray(var.a, dtype=int).reshape(-1)):
                    if 0 <= a < len(labels):
                        labels[a] = (mname, vname, str(idx[i]) if i < len(idx) else "?")
    return xl, yl


def _discrete_snapshot(system: Any) -> dict[str, np.ndarray]:
    snap: dict[str, np.ndarray] = {}
    for mname, model in system.models.items():
        if not getattr(model, "n", 0):
            continue
        for dname, disc in getattr(model, "discrete", {}).items():
            try:
                vals = disc.get_values()
                names = disc.get_names()
            except Exception:
                continue
            for flag, val in zip(names, vals):
                arr = np.asarray(val, dtype=float).reshape(-1)
                if arr.size:
                    snap[f"{mname}.{flag}"] = arr.copy()
    return snap


def _flags_changed(base: Mapping[str, np.ndarray], other: Mapping[str, np.ndarray]) -> list[str]:
    changed = []
    for key, arr in base.items():
        o = other.get(key)
        if o is None or o.shape != arr.shape or np.any(o != arr):
            changed.append(key)
    return changed


def _pegged(system: Any) -> set[int]:
    out: set[int] = set()
    for item in getattr(system, "antiwindups", []) or []:
        for key, _, _ in getattr(item, "x_set", []) or []:
            out.update(int(a) for a in np.atleast_1d(key))
    return out


def column_subset(system: Any, devices_per_model: int = 2) -> tuple[list[tuple[str, int]], int, set[str]]:
    cols: dict[tuple[str, int], None] = {}
    pairs = 0
    models: set[str] = set()
    for mname, model in system.models.items():
        n = int(getattr(model, "n", 0))
        if not n:
            continue
        u = np.asarray(getattr(getattr(model, "u", None), "v", np.ones(n)), dtype=float).reshape(-1)
        pos = [i for i in range(n) if i < u.size and u[i] != 0.0][:devices_per_model]
        if len(pos) < devices_per_model:
            pos += [i for i in range(n) if i not in pos][: devices_per_model - len(pos)]
        variables = _variables(model)
        if variables:
            models.add(mname)
        for vname, var, domain in variables:
            addr = np.asarray(var.a, dtype=int).reshape(-1)
            if not addr.size:
                continue
            pairs += 1
            for i in pos:
                if i < addr.size:
                    cols[(domain, int(addr[i]))] = None
    return list(cols), pairs, models


def audit_jacobian(
    system: Any,
    *,
    label: str = "",
    columns: str | Sequence[tuple[str, int]] = "auto",
    devices_per_model: int = 2,
    full_limit: int = 3000,
    rel_step: float = 1e-6,
    abs_tol: float = 1e-6,
    rel_tol: float = 1e-4,
    corrected: bool = True,
    restore_tol: float = 1e-9,
    tolerance_basis: str = "column",
) -> AuditResult:
    if tolerance_basis not in {"column", "entry"}:
        raise JacobianAuditError("tolerance_basis must be 'column' or 'entry'")

    t0 = time.perf_counter()
    dae = system.dae
    n, m = int(dae.n), int(dae.m)
    tds = system.TDS
    if not getattr(tds, "initialized", False):
        raise JacobianAuditError("system must be TDS-initialized")
    custom0 = bool(getattr(tds, "custom_event", False))
    x0 = np.array(dae.x, dtype=float)
    y0 = np.array(dae.y, dtype=float)
    jac = system_jacobians(system, update=True, corrected=corrected)
    f0 = np.array(dae.f, dtype=float)
    g0 = np.array(dae.g, dtype=float)
    F0 = np.concatenate((f0, g0))
    flags0 = _discrete_snapshot(system)
    pegged = _pegged(system)
    xl, yl = address_labels(system)
    models_cov: set[str] = set()
    pairs = 0
    if isinstance(columns, str):
        mode = columns
        if mode == "auto":
            mode = "all" if n + m <= full_limit else "subset"
        if mode == "all":
            cols = [("x", k) for k in range(n)] + [("y", k) for k in range(m)]
            _, pairs, models_cov = column_subset(system, devices_per_model)
        elif mode == "subset":
            cols, pairs, models_cov = column_subset(system, devices_per_model)
        else:
            raise JacobianAuditError(f"unknown column mode {columns!r}")
        full = mode == "all"
    else:
        cols = [(str(d), int(a)) for d, a in columns]
        full = False

    fxt, fyt, gxt, gyt = (jac.fx.tocsc(), jac.fy.tocsc(), jac.gx.tocsc(), jac.gy.tocsc())
    services = _service_sources(system)
    entries: list[AuditEntry] = []

    def evaluate() -> np.ndarray:
        system.vars_to_models()
        tds.fg_update(system.exist.pflow_tds)
        return np.concatenate((np.array(dae.f, dtype=float), np.array(dae.g, dtype=float)))

    try:
        for domain, k in cols:
            arr = dae.x if domain == "x" else dae.y
            val = float(arr[k])
            h = rel_step * max(1.0, abs(val))
            arr[k] = val + h
            Fp = evaluate()
            flags_p = _discrete_snapshot(system)
            arr[k] = val - h
            Fm = evaluate()
            flags_m = _discrete_snapshot(system)
            arr[k] = val
            system.vars_to_models()
            fd = (Fp - Fm) / (2.0 * h)
            dplus = (Fp - F0) / h
            dminus = (F0 - Fm) / h
            if domain == "x":
                ana = np.concatenate((fxt[:, k].toarray().ravel(), gxt[:, k].toarray().ravel()))
            else:
                ana = np.concatenate((fyt[:, k].toarray().ravel(), gyt[:, k].toarray().ravel()))
            scale = float(max(np.max(np.abs(ana)) if ana.size else 0.0, np.max(np.abs(fd)) if fd.size else 0.0))
            if tolerance_basis == "column":
                tol_vec = np.full(fd.shape, max(abs_tol, rel_tol * scale))
            else:
                tol_vec = np.maximum(abs_tol, rel_tol * np.maximum(np.abs(ana), np.abs(fd)))
            bad = np.flatnonzero(np.abs(fd - ana) > tol_vec)
            if not bad.size:
                continue
            changed = sorted(set(_flags_changed(flags0, flags_p)) | set(_flags_changed(flags0, flags_m)))
            col_label = (xl[k] if domain == "x" else yl[k])
            for r in bad:
                tol = float(tol_vec[r])
                is_f = r < n
                block = ("f" if is_f else "g") + domain
                row = int(r if is_f else r - n)
                row_label = xl[row] if is_f else yl[row]
                smooth = abs(dplus[r] - dminus[r]) <= max(tol, 1e-3 * max(abs(dplus[r]), abs(dminus[r])))
                if is_f and row in pegged:
                    cls, cause = "pegged_state", "anti-windup pegged state (f reset in TDS)"
                elif changed or not smooth:
                    cls = "kink"
                    related = [c for c in changed if c.split(".")[0] in {row_label[0], col_label[0]}]
                    cause = "discrete flags changed: " + ", ".join((related or changed)[:6]) if changed else (
                        "non-smooth residual (one-sided differences disagree)"
                    )
                else:
                    cls = "missing_derivative"
                    cause = _missing_cause(services, row_label, col_label)
                entries.append(
                    AuditEntry(
                        block=block,
                        row=row,
                        col=int(k),
                        row_label=row_label,
                        col_label=col_label,
                        analytic=float(ana[r]),
                        fd=float(fd[r]),
                        fd_plus=float(dplus[r]),
                        fd_minus=float(dminus[r]),
                        column_scale=scale,
                        tolerance=tol,
                        classification=cls,
                        cause=cause,
                    )
                )
    finally:
        dae.x[:] = x0
        dae.y[:] = y0
        system.vars_to_models()
        tds.fg_update(system.exist.pflow_tds)
        system.j_update(models=system.exist.pflow_tds)
        tds.custom_event = custom0
    err = max(
        float(np.max(np.abs(np.asarray(dae.f) - f0))) if n else 0.0,
        float(np.max(np.abs(np.asarray(dae.g) - g0))) if m else 0.0,
    )
    jac1 = system_jacobians(system, update=False, corrected=corrected)
    for name in ("fx", "fy", "gx", "gy"):
        d = getattr(jac1, name) - getattr(jac, name)
        if d.nnz:
            err = max(err, float(np.max(np.abs(d.data))))
    if not err <= restore_tol:
        raise JacobianAuditError(f"restoration after the audit failed (max deviation {err:.3e})")
    return AuditResult(
        label=label,
        corrected=corrected,
        n_states=n,
        n_algebraic=m,
        columns_checked=len(cols),
        columns_total=n + m,
        full=full,
        entries=entries,
        runtime_s=time.perf_counter() - t0,
        models_covered=tuple(sorted(models_cov)),
        pairs_covered=int(pairs),
        rel_step=rel_step,
        abs_tol=abs_tol,
        rel_tol=rel_tol,
        metadata={"tolerance_basis": tolerance_basis, "pegged_states": len(pegged),
                  "baseline_max_f": float(np.max(np.abs(f0))) if n else 0.0,
                  "baseline_max_g": float(np.max(np.abs(g0))) if m else 0.0},
    )


def _service_sources(system: Any) -> dict[str, list[tuple[str, str]]]:
    out: dict[str, list[tuple[str, str]]] = {}
    for mname, model in system.models.items():
        if not getattr(model, "n", 0):
            continue
        for sname, svc in getattr(model, "services_var", {}).items():
            out.setdefault(mname, []).append((sname, str(getattr(svc, "v_str", "") or "")))
    return out


def _missing_cause(
    services: Mapping[str, list[tuple[str, str]]],
    row_label: tuple[str, str, str],
    col_label: tuple[str, str, str],
) -> str:
    cands = []
    for model in {row_label[0], col_label[0]}:
        for sname, expr in services.get(model, []):
            if col_label[1] in expr or (col_label[0] == "Bus" and any(t in expr for t in ("v", "a"))):
                cands.append(f"{model}.{sname} = {expr}")
    if cands:
        return "chain rule through VarService: " + " | ".join(cands[:3])
    return "unexplained (no VarService of the row/column model mentions the variable)"


def summarize(results: Iterable[AuditResult]) -> list[dict[str, Any]]:
    rows = []
    for res in results:
        groups = res.groups()
        rows.append(
            {
                "label": res.label,
                "corrected": res.corrected,
                "n": res.n_states,
                "m": res.n_algebraic,
                "columns": f"{res.columns_checked}/{res.columns_total}" + (" (all)" if res.full else " (subset)"),
                "mismatches": len(res.entries),
                "missing_derivative": sum(1 for e in res.entries if e.classification == "missing_derivative"),
                "kink": sum(1 for e in res.entries if e.classification == "kink"),
                "pegged_state": sum(1 for e in res.entries if e.classification == "pegged_state"),
                "groups": groups,
                "runtime_s": res.runtime_s,
            }
        )
    return rows


CORRECTED_SERVICES = {("PVD1", "Vcomp"), ("PVD1", "Ipmaxsq"), ("PVD1", "Iqmaxsq")}


def varservice_inventory(system: Any) -> list[dict[str, Any]]:
    import re

    out: list[dict[str, Any]] = []
    for mname, model in system.models.items():
        if not getattr(model, "n", 0):
            continue
        svc = getattr(model, "services_var", {})
        if not svc:
            continue
        eqs: dict[str, str] = {}
        for container in ("states", "algebs", "states_ext", "algebs_ext"):
            for vname, var in getattr(model, container, {}).items():
                e = getattr(var, "e_str", None)
                if e:
                    eqs[vname] = str(e)
        for sname, service in svc.items():
            pat = re.compile(r"\b%s\b" % re.escape(sname))
            users = sorted(v for v, e in eqs.items() if pat.search(e))
            disc = []
            for dname, d in getattr(model, "discrete", {}).items():
                for attr in ("u", "lower", "upper", "center"):
                    if getattr(d, attr, None) is service:
                        disc.append(f"{dname}.{attr}")
            out.append(
                {
                    "model": mname,
                    "service": sname,
                    "expression": str(getattr(service, "v_str", "") or ""),
                    "used_in_equations": users,
                    "used_by_discrete": disc,
                    "corrected": (mname, sname) in CORRECTED_SERVICES,
                }
            )
    return out


def _build_target(target: str) -> tuple[str, Any]:
    if target == "replica":
        from .replication.activsg2000_engine import build_armed_replica
        from .replication.commitment_study import level_config

        armed = build_armed_replica(level_config("stock", 0.01), method="dynamic_equilibrium", settle_s=40)
        return "replica/stock_m0.01/dynamic_equilibrium", armed.system
    from .benchmarks import build_benchmark

    slug, _, variant = target.partition(":")
    case = build_benchmark(slug, variant=variant or "baseline")
    return f"{slug}/{variant or 'baseline'}", case.system


def main(argv: Sequence[str] | None = None) -> None:
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Finite-difference audit of ANDES Jacobians")
    parser.add_argument("targets", nargs="+")
    parser.add_argument("--columns", default="all", choices=("all", "subset", "auto"))
    parser.add_argument("--json", default=None)
    args = parser.parse_args(argv)
    report: list[dict[str, Any]] = []
    for target in args.targets:
        t0 = time.perf_counter()
        label, system = _build_target(target)
        item: dict[str, Any] = {
            "label": label,
            "build_s": time.perf_counter() - t0,
            "models": {k: int(m.n) for k, m in system.models.items() if getattr(m, "n", 0)},
            "varservices": varservice_inventory(system),
        }
        for key, corrected, basis in (
            ("raw_column", False, "column"),
            ("corrected_column", True, "column"),
            ("corrected_entry", True, "entry"),
        ):
            res = audit_jacobian(system, label=label, corrected=corrected, columns=args.columns, tolerance_basis=basis)
            item[key] = res.to_dict()
            print(f"{label} {key}: {res.columns_checked}/{res.columns_total} columns, "
                  f"{len(res.entries)} mismatches ({res.runtime_s:.1f} s)", flush=True)
        report.append(item)
        if args.json:
            with open(args.json, "w") as fh:
                json.dump(report, fh, indent=1, default=float)


if __name__ == "__main__":
    main()
