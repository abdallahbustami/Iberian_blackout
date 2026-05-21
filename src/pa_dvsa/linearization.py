r"""Mode-wise linear DAE reduction.

This module implements the reduction in ``LaTeX/root.tex``:

.. math::

   A_r = A_x - A_y G_y^{-1} G_x,\quad
   B_r = B_x - A_y G_y^{-1} B_y,\quad
   D_r = D_x - A_y G_y^{-1} D_y,

   C_r = C_x - C_y G_y^{-1} G_x,\quad
   F_u = -C_y G_y^{-1} B_y,\quad
   F_d = -C_y G_y^{-1} D_y.

No dense inverse of ``G_y`` is formed.  A sparse LU factorization of ``G_y`` is
computed once and reused for the right-hand-side blocks.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np
from scipy import sparse
from scipy.sparse import linalg as spla

from .andes_adapter import JacobianBlocks


class LinearizationError(RuntimeError):
    """Raised when a mode-wise linearization cannot be reduced."""


@dataclass(frozen=True, slots=True)
class LinearDAEModel:
    """Linearized mode-wise DAE blocks.

    Shapes follow the paper notation:

    * ``x``: dynamic states, length ``n``.
    * ``y``: algebraic variables/equations, length ``m``.
    * ``z``: protected outputs, length ``p``.
    * ``u``: control channels, length ``r``.
    * ``d``: disturbance channels, length ``q``.
    """

    ax: sparse.csc_matrix
    ay: sparse.csc_matrix
    gx: sparse.csc_matrix
    gy: sparse.csc_matrix
    cx: sparse.csc_matrix
    cy: sparse.csc_matrix
    bx: sparse.csc_matrix
    by: sparse.csc_matrix
    dx: sparse.csc_matrix
    dy: sparse.csc_matrix
    mode_id: str = "mode"

    @classmethod
    def from_blocks(
        cls,
        *,
        ax: Any,
        ay: Any,
        gx: Any,
        gy: Any,
        cx: Any,
        cy: Any,
        bx: Any | None = None,
        by: Any | None = None,
        dx: Any | None = None,
        dy: Any | None = None,
        mode_id: str = "mode",
    ) -> "LinearDAEModel":
        """Create and validate a linear DAE from matrix-like blocks."""

        ax_c = _as_csc(ax, "A_x")
        ay_c = _as_csc(ay, "A_y")
        gx_c = _as_csc(gx, "G_x")
        gy_c = _as_csc(gy, "G_y")
        cx_c = _as_csc(cx, "C_x")
        cy_c = _as_csc(cy, "C_y")

        n = ax_c.shape[0]
        m = gy_c.shape[0]
        p = cx_c.shape[0]

        bx_c, by_c = _paired_channel_blocks(
            left=bx,
            right=by,
            left_rows=n,
            right_rows=m,
            left_name="B_x",
            right_name="B_y",
        )
        dx_c, dy_c = _paired_channel_blocks(
            left=dx,
            right=dy,
            left_rows=n,
            right_rows=m,
            left_name="D_x",
            right_name="D_y",
        )

        model = cls(
            ax=ax_c,
            ay=ay_c,
            gx=gx_c,
            gy=gy_c,
            cx=cx_c,
            cy=cy_c,
            bx=bx_c,
            by=by_c,
            dx=dx_c,
            dy=dy_c,
            mode_id=mode_id,
        )
        model.validate()
        return model

    @classmethod
    def from_andes_jacobians(
        cls,
        jacobians: JacobianBlocks,
        *,
        cx: Any,
        cy: Any,
        bx: Any | None = None,
        by: Any | None = None,
        dx: Any | None = None,
        dy: Any | None = None,
        mode_id: str = "andes_mode",
        fold_zero_time_constants: bool = True,
        zero_time_constant_tol: float = 0.0,
        cleanup_dead_algebraic: bool = True,
        dead_algebraic_tol: float = 1e-10,
    ) -> "LinearDAEModel":
        """Build from ANDES mass-matrix Jacobians plus selected inputs/outputs.

        ANDES stores dynamics as ``Tf * xdot = f(x, y)``.  This adapter
        converts those residual Jacobians to the standard form used in the
        paper.  States with zero ``Tf`` are algebraic constraints and are
        folded into the algebraic block by default.
        """

        fx = _as_csc(jacobians.fx, "f_x")
        fy = _as_csc(jacobians.fy, "f_y")
        gx = _as_csc(jacobians.gx, "g_x")
        gy = _as_csc(jacobians.gy, "g_y")
        cx_c = _as_csc(cx, "C_x")
        cy_c = _as_csc(cy, "C_y")

        n = fx.shape[0]
        m = gy.shape[0]
        p = cx_c.shape[0]
        _require_shape(fx, (n, n), "f_x")
        _require_shape(fy, (n, m), "f_y")
        _require_shape(gx, (m, n), "g_x")
        _require_shape(gy, (m, m), "g_y")
        _require_shape(cx_c, (p, n), "C_x")
        _require_shape(cy_c, (p, m), "C_y")
        bx_c, by_c = _paired_channel_blocks(
            left=bx,
            right=by,
            left_rows=n,
            right_rows=m,
            left_name="B_x",
            right_name="B_y",
        )
        dx_c, dy_c = _paired_channel_blocks(
            left=dx,
            right=dy,
            left_rows=n,
            right_rows=m,
            left_name="D_x",
            right_name="D_y",
        )
        _require_shape(bx_c, (n, bx_c.shape[1]), "B_x")
        _require_shape(by_c, (m, bx_c.shape[1]), "B_y")
        _require_shape(dx_c, (n, dx_c.shape[1]), "D_x")
        _require_shape(dy_c, (m, dx_c.shape[1]), "D_y")

        tf = np.asarray(jacobians.tf, dtype=float).reshape(-1)
        if n == 0:
            if tf.size not in {0, n}:
                raise LinearizationError(f"Tf has length {tf.size}, expected {n}")
            return cls.from_blocks(
                ax=fx,
                ay=fy,
                gx=gx,
                gy=gy,
                cx=cx_c,
                cy=cy_c,
                bx=bx_c,
                by=by_c,
                dx=dx_c,
                dy=dy_c,
                mode_id=mode_id,
            )

        if tf.size != n:
            raise LinearizationError(f"Tf has length {tf.size}, expected {n}")
        if not np.all(np.isfinite(tf)):
            raise LinearizationError("Tf contains non-finite entries")
        if zero_time_constant_tol < 0.0:
            raise LinearizationError("zero_time_constant_tol must be nonnegative")
        if dead_algebraic_tol < 0.0:
            raise LinearizationError("dead_algebraic_tol must be nonnegative")

        zero_mask = np.abs(tf) <= zero_time_constant_tol
        if np.any(zero_mask) and not fold_zero_time_constants:
            raise LinearizationError("ANDES Jacobians contain zero-Tf states")

        if np.any(zero_mask):
            fx, fy, gx, gy, bx_c, by_c, dx_c, dy_c, cx_c, cy_c, tf = _fold_zero_tf_states(
                fx=fx,
                fy=fy,
                gx=gx,
                gy=gy,
                bx=bx_c,
                by=by_c,
                dx=dx_c,
                dy=dy_c,
                cx=cx_c,
                cy=cy_c,
                tf=tf,
                zero_mask=zero_mask,
            )

        if tf.size and np.any(tf <= 0.0):
            raise LinearizationError("nonzero ANDES time constants must be positive")
        if tf.size:
            inv_tf = sparse.diags(1.0 / tf, format="csc")
            fx = _to_csc(inv_tf @ fx, "A_x")
            fy = _to_csc(inv_tf @ fy, "A_y")
            bx_c = _to_csc(inv_tf @ bx_c, "B_x")
            dx_c = _to_csc(inv_tf @ dx_c, "D_x")

        if cleanup_dead_algebraic:
            fx, fy, gx, gy, bx_c, by_c, dx_c, dy_c, cx_c, cy_c = (
                _cleanup_dead_algebraic_constraints(
                    ax=fx,
                    ay=fy,
                    gx=gx,
                    gy=gy,
                    bx=bx_c,
                    by=by_c,
                    dx=dx_c,
                    dy=dy_c,
                    cx=cx_c,
                    cy=cy_c,
                    tol=dead_algebraic_tol,
                )
            )

        return cls.from_blocks(
            ax=fx,
            ay=fy,
            gx=gx,
            gy=gy,
            cx=cx_c,
            cy=cy_c,
            bx=bx_c,
            by=by_c,
            dx=dx_c,
            dy=dy_c,
            mode_id=mode_id,
        )

    @property
    def n_states(self) -> int:
        return self.ax.shape[0]

    @property
    def n_algebraic(self) -> int:
        return self.gy.shape[0]

    @property
    def n_outputs(self) -> int:
        return self.cx.shape[0]

    @property
    def n_controls(self) -> int:
        return self.bx.shape[1]

    @property
    def n_disturbances(self) -> int:
        return self.dx.shape[1]

    @property
    def algebraic_only(self) -> bool:
        return self.n_states == 0

    def validate(self) -> None:
        """Check dimensions and finite values."""

        n = self.n_states
        m = self.n_algebraic
        p = self.n_outputs
        r = self.n_controls
        q = self.n_disturbances

        _require_shape(self.ax, (n, n), "A_x")
        _require_shape(self.ay, (n, m), "A_y")
        _require_shape(self.gx, (m, n), "G_x")
        _require_shape(self.gy, (m, m), "G_y")
        _require_shape(self.cx, (p, n), "C_x")
        _require_shape(self.cy, (p, m), "C_y")
        _require_shape(self.bx, (n, r), "B_x")
        _require_shape(self.by, (m, r), "B_y")
        _require_shape(self.dx, (n, q), "D_x")
        _require_shape(self.dy, (m, q), "D_y")

        for name, matrix in (
            ("A_x", self.ax),
            ("A_y", self.ay),
            ("G_x", self.gx),
            ("G_y", self.gy),
            ("C_x", self.cx),
            ("C_y", self.cy),
            ("B_x", self.bx),
            ("B_y", self.by),
            ("D_x", self.dx),
            ("D_y", self.dy),
        ):
            _require_finite_sparse(matrix, name)


@dataclass(frozen=True, slots=True)
class ReductionDiagnostics:
    """Consistency and stability diagnostics for one reduced model."""

    mode_id: str
    n_states: int
    n_algebraic: int
    n_outputs: int
    n_controls: int
    n_disturbances: int
    algebraic_only: bool
    gy_factorized: bool
    gy_min_abs_u_diag: float | None
    max_real_eigenvalue: float | None
    stability: str
    warnings: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ReducedLinearModel:
    """Reduced ODE/output model from the paper."""

    ar: sparse.csc_matrix
    br: sparse.csc_matrix
    dr: sparse.csc_matrix
    cr: sparse.csc_matrix
    fu: sparse.csc_matrix
    fd: sparse.csc_matrix
    diagnostics: ReductionDiagnostics

    @property
    def algebraic_only(self) -> bool:
        return self.diagnostics.algebraic_only


def selector_matrix(addresses: list[int] | tuple[int, ...], width: int) -> sparse.csc_matrix:
    """Build a sparse row selector matrix for algebraic or state addresses."""

    rows = len(addresses)
    if width < 0:
        raise LinearizationError("selector width must be nonnegative")
    row_idx: list[int] = []
    col_idx: list[int] = []
    data: list[float] = []
    for row, address in enumerate(addresses):
        idx = int(address)
        if idx < 0 or idx >= width:
            raise LinearizationError(f"selector address {idx} is outside width {width}")
        row_idx.append(row)
        col_idx.append(idx)
        data.append(1.0)
    return sparse.coo_matrix((data, (row_idx, col_idx)), shape=(rows, width)).tocsc()


def reduce_linear_dae(
    model: LinearDAEModel,
    *,
    stability_tol: float = 1e-8,
    eigen_dense_limit: int = 256,
) -> ReducedLinearModel:
    """Reduce an index-1 linear DAE using sparse solves against ``G_y``."""

    model.validate()
    n = model.n_states
    m = model.n_algebraic
    p = model.n_outputs
    r = model.n_controls
    q = model.n_disturbances
    warnings: list[str] = []

    if m == 0:
        ar = model.ax.copy()
        br = model.bx.copy()
        dr = model.dx.copy()
        cr = model.cx.copy()
        fu = sparse.csc_matrix((p, r))
        fd = sparse.csc_matrix((p, q))
        gy_factorized = False
        gy_min_abs_u_diag = None
    else:
        lu = _factor_gy(model.gy)
        gy_factorized = True
        u_diag = np.asarray(lu.U.diagonal(), dtype=float)
        gy_min_abs_u_diag = float(np.min(np.abs(u_diag))) if u_diag.size else None
        if gy_min_abs_u_diag is not None and gy_min_abs_u_diag < 1e-10:
            warnings.append("G_y LU factor has a very small U diagonal entry")

        gy_inv_gx = _solve_block(lu, model.gx, "G_x")
        gy_inv_by = _solve_block(lu, model.by, "B_y")
        gy_inv_dy = _solve_block(lu, model.dy, "D_y")

        ar = _to_csc(model.ax - _as_product(model.ay, gy_inv_gx), "A_r")
        br = _to_csc(model.bx - _as_product(model.ay, gy_inv_by), "B_r")
        dr = _to_csc(model.dx - _as_product(model.ay, gy_inv_dy), "D_r")
        cr = _to_csc(model.cx - _as_product(model.cy, gy_inv_gx), "C_r")
        fu = _to_csc(-_as_product(model.cy, gy_inv_by), "F_u")
        fd = _to_csc(-_as_product(model.cy, gy_inv_dy), "F_d")

    for name, matrix in (
        ("A_r", ar),
        ("B_r", br),
        ("D_r", dr),
        ("C_r", cr),
        ("F_u", fu),
        ("F_d", fd),
    ):
        _require_finite_sparse(matrix, name)

    max_real, stability = _stability_diagnostic(ar, stability_tol, eigen_dense_limit, warnings)

    diagnostics = ReductionDiagnostics(
        mode_id=model.mode_id,
        n_states=n,
        n_algebraic=m,
        n_outputs=p,
        n_controls=r,
        n_disturbances=q,
        algebraic_only=(n == 0),
        gy_factorized=gy_factorized,
        gy_min_abs_u_diag=gy_min_abs_u_diag,
        max_real_eigenvalue=max_real,
        stability=stability,
        warnings=tuple(warnings),
    )
    return ReducedLinearModel(ar=ar, br=br, dr=dr, cr=cr, fu=fu, fd=fd, diagnostics=diagnostics)


def _as_csc(value: Any, name: str) -> sparse.csc_matrix:
    try:
        matrix = sparse.csc_matrix(value, dtype=float)
    except Exception as exc:
        raise LinearizationError(f"{name} cannot be converted to a sparse matrix") from exc
    if len(matrix.shape) != 2:
        raise LinearizationError(f"{name} must be two-dimensional")
    _require_finite_sparse(matrix, name)
    return matrix


def _paired_channel_blocks(
    *,
    left: Any | None,
    right: Any | None,
    left_rows: int,
    right_rows: int,
    left_name: str,
    right_name: str,
) -> tuple[sparse.csc_matrix, sparse.csc_matrix]:
    left_c = _as_csc(left, left_name) if left is not None else None
    right_c = _as_csc(right, right_name) if right is not None else None

    if left_c is not None and right_c is not None and left_c.shape[1] != right_c.shape[1]:
        raise LinearizationError(
            f"{left_name} and {right_name} must have the same number of columns; "
            f"got {left_c.shape[1]} and {right_c.shape[1]}"
        )

    width = 0
    if left_c is not None:
        width = left_c.shape[1]
    if right_c is not None:
        width = right_c.shape[1]

    if left_c is None:
        left_c = sparse.csc_matrix((left_rows, width))
    if right_c is None:
        right_c = sparse.csc_matrix((right_rows, width))
    return left_c, right_c


def _fold_zero_tf_states(
    *,
    fx: sparse.csc_matrix,
    fy: sparse.csc_matrix,
    gx: sparse.csc_matrix,
    gy: sparse.csc_matrix,
    bx: sparse.csc_matrix,
    by: sparse.csc_matrix,
    dx: sparse.csc_matrix,
    dy: sparse.csc_matrix,
    cx: sparse.csc_matrix,
    cy: sparse.csc_matrix,
    tf: np.ndarray,
    zero_mask: np.ndarray,
) -> tuple[
    sparse.csc_matrix,
    sparse.csc_matrix,
    sparse.csc_matrix,
    sparse.csc_matrix,
    sparse.csc_matrix,
    sparse.csc_matrix,
    sparse.csc_matrix,
    sparse.csc_matrix,
    sparse.csc_matrix,
    sparse.csc_matrix,
    np.ndarray,
]:
    dynamic_idx = np.flatnonzero(~zero_mask)
    zero_idx = np.flatnonzero(zero_mask)

    fx_new = _to_csc(fx[dynamic_idx, :][:, dynamic_idx], "f_x")
    fy_new = _to_csc(
        sparse.hstack((fx[dynamic_idx, :][:, zero_idx], fy[dynamic_idx, :]), format="csc"),
        "f_y",
    )

    gx_new = _to_csc(
        sparse.vstack((fx[zero_idx, :][:, dynamic_idx], gx[:, dynamic_idx]), format="csc"),
        "g_x",
    )
    gy_new = _to_csc(
        sparse.vstack(
            (
                sparse.hstack((fx[zero_idx, :][:, zero_idx], fy[zero_idx, :]), format="csc"),
                sparse.hstack((gx[:, zero_idx], gy), format="csc"),
            ),
            format="csc",
        ),
        "g_y",
    )

    bx_new = _to_csc(bx[dynamic_idx, :], "B_x")
    by_new = _to_csc(sparse.vstack((bx[zero_idx, :], by), format="csc"), "B_y")
    dx_new = _to_csc(dx[dynamic_idx, :], "D_x")
    dy_new = _to_csc(sparse.vstack((dx[zero_idx, :], dy), format="csc"), "D_y")

    cx_new = _to_csc(cx[:, dynamic_idx], "C_x")
    cy_new = _to_csc(sparse.hstack((cx[:, zero_idx], cy), format="csc"), "C_y")
    tf_new = tf[dynamic_idx]
    return fx_new, fy_new, gx_new, gy_new, bx_new, by_new, dx_new, dy_new, cx_new, cy_new, tf_new


def _cleanup_dead_algebraic_constraints(
    *,
    ax: sparse.csc_matrix,
    ay: sparse.csc_matrix,
    gx: sparse.csc_matrix,
    gy: sparse.csc_matrix,
    bx: sparse.csc_matrix,
    by: sparse.csc_matrix,
    dx: sparse.csc_matrix,
    dy: sparse.csc_matrix,
    cx: sparse.csc_matrix,
    cy: sparse.csc_matrix,
    tol: float,
) -> tuple[
    sparse.csc_matrix,
    sparse.csc_matrix,
    sparse.csc_matrix,
    sparse.csc_matrix,
    sparse.csc_matrix,
    sparse.csc_matrix,
    sparse.csc_matrix,
    sparse.csc_matrix,
    sparse.csc_matrix,
    sparse.csc_matrix,
]:
    dead_rows = np.flatnonzero(_sparse_row_norms(gy) < tol)
    if dead_rows.size == 0:
        gy = _regularize_dead_algebraic_columns(gy, tol)
        return ax, ay, gx, gy, bx, by, dx, dy, cx, cy

    gx_dead_norms = _sparse_row_norms(gx[dead_rows, :])
    by_dead_norms = _sparse_row_norms(by[dead_rows, :]) if by.shape[1] else np.zeros(dead_rows.size)
    dy_dead_norms = _sparse_row_norms(dy[dead_rows, :]) if dy.shape[1] else np.zeros(dead_rows.size)

    if np.any((by_dead_norms >= tol) | (dy_dead_norms >= tol)):
        raise LinearizationError(
            "dead algebraic state constraints with direct input/disturbance terms are unsupported"
        )

    constraint_rows = dead_rows[gx_dead_norms >= tol]
    alive_mask = np.ones(gy.shape[0], dtype=bool)
    alive_mask[dead_rows] = False
    alive_idx = np.flatnonzero(alive_mask)

    ay_alive = _to_csc(ay[:, alive_idx], "A_y")
    gx_alive = _to_csc(gx[alive_idx, :], "G_x")
    gy_alive = _to_csc(gy[alive_idx, :][:, alive_idx], "G_y")
    by_alive = _to_csc(by[alive_idx, :], "B_y")
    dy_alive = _to_csc(dy[alive_idx, :], "D_y")
    cy_alive = _to_csc(cy[:, alive_idx], "C_y")

    if constraint_rows.size:
        substitution = _state_constraint_substitution(gx[constraint_rows, :], tol)
        free = substitution.free_state_indices
        transform = substitution.full_from_free

        ax = _to_csc(ax[free, :] @ transform, "A_x")
        ay_alive = _to_csc(ay_alive[free, :], "A_y")
        bx = _to_csc(bx[free, :], "B_x")
        dx = _to_csc(dx[free, :], "D_x")
        gx_alive = _to_csc(gx_alive @ transform, "G_x")
        cx = _to_csc(cx @ transform, "C_x")

    gy_alive = _regularize_dead_algebraic_columns(gy_alive, tol)
    return ax, ay_alive, gx_alive, gy_alive, bx, by_alive, dx, dy_alive, cx, cy_alive


@dataclass(frozen=True, slots=True)
class _StateConstraintSubstitution:
    free_state_indices: np.ndarray
    full_from_free: np.ndarray


def _state_constraint_substitution(
    constraints: sparse.csc_matrix,
    tol: float,
) -> _StateConstraintSubstitution:
    c_dense = constraints.toarray()
    n_constraints, n_states = c_dense.shape
    if n_constraints > n_states:
        raise LinearizationError("more independent state constraints than states")

    dependent: list[int] = []
    used: set[int] = set()
    for row in range(n_constraints):
        row_abs = np.abs(c_dense[row, :])
        for pivot in np.argsort(-row_abs):
            pivot_int = int(pivot)
            if pivot_int not in used and row_abs[pivot_int] > tol:
                dependent.append(pivot_int)
                used.add(pivot_int)
                break
        else:
            raise LinearizationError("state constraint row has no usable pivot")

    dependent_idx = np.asarray(dependent, dtype=int)
    free_mask = np.ones(n_states, dtype=bool)
    free_mask[dependent_idx] = False
    free_idx = np.flatnonzero(free_mask)

    c_dep = c_dense[:, dependent_idx]
    c_free = c_dense[:, free_idx]
    try:
        dependent_from_free = -np.linalg.solve(c_dep, c_free)
    except np.linalg.LinAlgError as exc:
        raise LinearizationError("state constraint pivot block is singular") from exc

    transform = np.zeros((n_states, free_idx.size), dtype=float)
    transform[free_idx, np.arange(free_idx.size)] = 1.0
    transform[dependent_idx, :] = dependent_from_free
    return _StateConstraintSubstitution(free_state_indices=free_idx, full_from_free=transform)


def _regularize_dead_algebraic_columns(gy: sparse.csc_matrix, tol: float) -> sparse.csc_matrix:
    dead_cols = np.flatnonzero(_sparse_col_norms(gy) < tol)
    if dead_cols.size == 0:
        return gy
    gy_lil = gy.tolil(copy=True)
    for col in dead_cols:
        gy_lil[int(col), int(col)] = 1.0
    return _to_csc(gy_lil, "G_y")


def _sparse_row_norms(matrix: sparse.spmatrix) -> np.ndarray:
    if matrix.shape[0] == 0:
        return np.zeros(0, dtype=float)
    return np.sqrt(np.asarray(matrix.multiply(matrix).sum(axis=1)).reshape(-1))


def _sparse_col_norms(matrix: sparse.spmatrix) -> np.ndarray:
    if matrix.shape[1] == 0:
        return np.zeros(0, dtype=float)
    return np.sqrt(np.asarray(matrix.multiply(matrix).sum(axis=0)).reshape(-1))


def _to_csc(value: Any, name: str) -> sparse.csc_matrix:
    matrix = value if sparse.issparse(value) else sparse.csc_matrix(value)
    matrix = matrix.tocsc()
    _require_finite_sparse(matrix, name)
    return matrix


def _require_shape(matrix: sparse.csc_matrix, expected: tuple[int, int], name: str) -> None:
    if matrix.shape != expected:
        raise LinearizationError(f"{name} has shape {matrix.shape}, expected {expected}")


def _require_finite_sparse(matrix: sparse.spmatrix, name: str) -> None:
    data = matrix.data
    if data.size and not np.all(np.isfinite(data)):
        raise LinearizationError(f"{name} contains non-finite entries")


def _factor_gy(gy: sparse.csc_matrix) -> spla.SuperLU:
    if gy.shape[0] != gy.shape[1]:
        raise LinearizationError("G_y must be square")
    if gy.shape[0] == 0:
        raise LinearizationError("Cannot factor empty G_y")
    try:
        return spla.splu(gy.tocsc())
    except (RuntimeError, ValueError) as exc:
        raise LinearizationError("G_y is singular or numerically unsuited for sparse LU") from exc


def _solve_block(lu: spla.SuperLU, rhs: sparse.csc_matrix, name: str) -> np.ndarray:
    if rhs.shape[1] == 0:
        return np.zeros((rhs.shape[0], 0), dtype=float)
    dense_rhs = rhs.toarray()
    if not np.all(np.isfinite(dense_rhs)):
        raise LinearizationError(f"{name} right-hand side contains non-finite values")
    try:
        solution = lu.solve(dense_rhs)
    except RuntimeError as exc:
        raise LinearizationError(f"Sparse solve failed for {name}") from exc
    if not np.all(np.isfinite(solution)):
        raise LinearizationError(f"Sparse solve for {name} produced non-finite values")
    return np.asarray(solution, dtype=float)


def _as_product(left: sparse.csc_matrix, right: np.ndarray) -> sparse.csc_matrix:
    if right.shape[1] == 0:
        return sparse.csc_matrix((left.shape[0], 0))
    return sparse.csc_matrix(left @ right)


def _stability_diagnostic(
    ar: sparse.csc_matrix,
    tol: float,
    dense_limit: int,
    warnings: list[str],
) -> tuple[float | None, str]:
    n = ar.shape[0]
    if n == 0:
        return None, "algebraic_only"

    try:
        if n <= dense_limit:
            eigvals = np.linalg.eigvals(ar.toarray())
            max_real = float(np.max(np.real(eigvals)))
        else:
            eigvals = spla.eigs(ar, k=1, which="LR", return_eigenvectors=False)
            max_real = float(np.max(np.real(eigvals)))
    except Exception as exc:  # pragma: no cover - depends on sparse eigensolver behavior.
        warnings.append(f"stability_eigenvalue_check_failed: {exc}")
        return None, "not_evaluated"

    if not math.isfinite(max_real):
        warnings.append("stability_eigenvalue_check_nonfinite")
        return None, "not_evaluated"
    if max_real > tol:
        warnings.append(f"reduced_state_matrix_unstable_max_real={max_real:.6g}")
        return max_real, "unstable"
    if max_real >= -tol:
        warnings.append(f"reduced_state_matrix_marginal_max_real={max_real:.6g}")
        return max_real, "marginal"
    return max_real, "stable"
