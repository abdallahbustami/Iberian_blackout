"""Linear model of a mode: elimination of the algebraic variables of the DAE with a sparse LU factorization."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Sequence

import numpy as np
from scipy import sparse
from scipy.sparse import linalg as spla

from .andes_adapter import JacobianBlocks


class LinearizationError(RuntimeError):
    pass


DEFAULT_MAX_CONDITION = 1.0e11


@dataclass(frozen=True, slots=True)
class AndesIndexMap:
    n_raw: int
    m_raw: int
    dynamic_idx: np.ndarray
    zero_idx: np.ndarray
    kept_aug_idx: np.ndarray
    removed_aug_idx: np.ndarray
    removed_zero_diagonal: np.ndarray
    inv_tf: np.ndarray

    def map_residual(self, df: Any, dg: Any) -> tuple[np.ndarray, np.ndarray]:
        df = np.asarray(df, dtype=float).reshape(-1)
        dg = np.asarray(dg, dtype=float).reshape(-1)
        if df.size != self.n_raw or dg.size != self.m_raw:
            raise LinearizationError(
                f"residual sizes ({df.size}, {dg.size}) do not match ({self.n_raw}, {self.m_raw})"
            )
        if not (np.all(np.isfinite(df)) and np.all(np.isfinite(dg))):
            raise LinearizationError("residual difference contains non-finite values")
        dx = self.inv_tf * df[self.dynamic_idx]
        aug = np.concatenate((df[self.zero_idx], dg))
        if self.removed_aug_idx.size:
            bad = self.removed_aug_idx[self.removed_zero_diagonal & (aug[self.removed_aug_idx] != 0.0)]
            if bad.size:
                raise LinearizationError(
                    "forcing enters an empty (decoupled) equation at augmented indices "
                    + ", ".join(str(int(item)) for item in bad[:10])
                )
        return dx, aug[self.kept_aug_idx]


@dataclass(frozen=True, slots=True)
class LinearDAEModel:
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
    x_names: tuple[str, ...] | None = None
    y_names: tuple[str, ...] | None = None
    removed_algebraic: tuple[str, ...] = ()
    index_map: "AndesIndexMap | None" = None

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
        x_names: Sequence[str] | None = None,
        y_names: Sequence[str] | None = None,
        removed_algebraic: Sequence[str] = (),
    ) -> "LinearDAEModel":
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
            x_names=None if x_names is None else tuple(str(item) for item in x_names),
            y_names=None if y_names is None else tuple(str(item) for item in y_names),
            removed_algebraic=tuple(str(item) for item in removed_algebraic),
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
        dead_algebraic_tol: float = 0.0,
    ) -> "LinearDAEModel":
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

        x_names = _names_or_none(getattr(jacobians, "x_names", None), n, "x_names")
        y_names = _names_or_none(getattr(jacobians, "y_names", None), m, "y_names")

        tf = np.asarray(jacobians.tf, dtype=float).reshape(-1)
        if n == 0 and tf.size not in {0, n}:
            raise LinearizationError(f"Tf has length {tf.size}, expected {n}")
        if n > 0 and tf.size != n:
            raise LinearizationError(f"Tf has length {tf.size}, expected {n}")
        if not np.all(np.isfinite(tf)):
            raise LinearizationError("Tf contains non-finite entries")
        if zero_time_constant_tol < 0.0:
            raise LinearizationError("zero_time_constant_tol must be nonnegative")
        if dead_algebraic_tol < 0.0:
            raise LinearizationError("dead_algebraic_tol must be nonnegative")

        zero_mask = np.abs(tf) <= zero_time_constant_tol if n else np.zeros(0, dtype=bool)
        if np.any(zero_mask) and not fold_zero_time_constants:
            raise LinearizationError("ANDES Jacobians contain zero-Tf states")
        zero_idx = np.flatnonzero(zero_mask)
        dyn_idx = np.flatnonzero(~zero_mask)

        if np.any(zero_mask):
            if y_names is not None or x_names is not None:
                xn = x_names or tuple(f"x[{i}]" for i in range(n))
                yn = y_names or tuple(f"y[{i}]" for i in range(m))
                y_names = tuple(f"{xn[i]} (Tf=0 state)" for i in zero_idx) + tuple(yn)
                x_names = tuple(xn[i] for i in dyn_idx)
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
        inv_tf_vec = 1.0 / tf if tf.size else np.zeros(0)
        if tf.size:
            inv_tf = sparse.diags(inv_tf_vec, format="csc")
            fx = _to_csc(inv_tf @ fx, "A_x")
            fy = _to_csc(inv_tf @ fy, "A_y")
            bx_c = _to_csc(inv_tf @ bx_c, "B_x")
            dx_c = _to_csc(inv_tf @ dx_c, "D_x")

        m_aug = gy.shape[0]
        kept_idx = np.arange(m_aug)
        removed: tuple[str, ...] = ()
        removed_zero_diag = np.zeros(0, dtype=bool)
        if cleanup_dead_algebraic:
            fy, gx, gy_new, by_c, dy_c, cy_c, y_names_new, removed, kept_idx = (
                _remove_decoupled_algebraic(
                    ay=fy,
                    gx=gx,
                    gy=gy,
                    by=by_c,
                    dy=dy_c,
                    cy=cy_c,
                    names=y_names,
                    tol=dead_algebraic_tol,
                )
            )
            removed_idx = np.setdiff1d(np.arange(m_aug), kept_idx)
            removed_zero_diag = np.abs(gy.diagonal()[removed_idx]) <= dead_algebraic_tol
            gy, y_names = gy_new, y_names_new

        index_map = AndesIndexMap(
            n_raw=n,
            m_raw=m,
            dynamic_idx=dyn_idx,
            zero_idx=zero_idx,
            kept_aug_idx=np.asarray(kept_idx, dtype=int),
            removed_aug_idx=np.setdiff1d(np.arange(m_aug), kept_idx),
            removed_zero_diagonal=np.asarray(removed_zero_diag, dtype=bool),
            inv_tf=np.asarray(inv_tf_vec, dtype=float),
        )

        model = cls.from_blocks(
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
            x_names=x_names,
            y_names=y_names,
            removed_algebraic=removed,
        )
        object.__setattr__(model, "index_map", index_map)
        return model

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
    gy_condition_estimate: float | None = None
    removed_algebraic: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ReducedLinearModel:
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
    stability_tol: float = 1e-6,
    eigen_dense_limit: int = 2000,
    max_condition: float = DEFAULT_MAX_CONDITION,
    factor: "AlgebraicFactor | None" = None,
) -> ReducedLinearModel:
    model.validate()
    n = model.n_states
    m = model.n_algebraic
    p = model.n_outputs
    r = model.n_controls
    q = model.n_disturbances
    warnings: list[str] = []
    gy_condition: float | None = None

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
        lu = factor if factor is not None else AlgebraicFactor(
            model.gy, names=model.y_names, max_condition=max_condition
        )
        if lu.shape != model.gy.shape:
            raise LinearizationError("supplied G_y factor has the wrong shape")
        gy_factorized = True
        gy_min_abs_u_diag = lu.min_abs_pivot
        gy_condition = lu.condition_estimate

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
        gy_condition_estimate=gy_condition,
        removed_algebraic=tuple(model.removed_algebraic),
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


def _solve_block(lu: "AlgebraicFactor", rhs: sparse.csc_matrix, name: str) -> np.ndarray:
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
    except Exception as exc:
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


def _names_or_none(names: Sequence[str] | None, count: int, label: str) -> tuple[str, ...] | None:
    if names is None:
        return None
    result = tuple(str(item) for item in names)
    if len(result) != count:
        raise LinearizationError(f"{label} has length {len(result)}, expected {count}")
    return result


def _offdiag_abs_sums(matrix: sparse.csc_matrix) -> tuple[np.ndarray, np.ndarray]:
    coo = matrix.tocoo()
    off = coo.row != coo.col
    vals = np.abs(coo.data[off])
    rows = np.bincount(coo.row[off], weights=vals, minlength=matrix.shape[0])
    cols = np.bincount(coo.col[off], weights=vals, minlength=matrix.shape[1])
    return rows, cols


def _abs_row_sums(matrix: sparse.spmatrix) -> np.ndarray:
    if matrix.shape[0] == 0:
        return np.zeros(0)
    if matrix.shape[1] == 0 or matrix.nnz == 0:
        return np.zeros(matrix.shape[0])
    return np.asarray(abs(matrix).sum(axis=1)).reshape(-1)


def _abs_col_sums(matrix: sparse.spmatrix) -> np.ndarray:
    if matrix.shape[1] == 0:
        return np.zeros(0)
    if matrix.shape[0] == 0 or matrix.nnz == 0:
        return np.zeros(matrix.shape[1])
    return np.asarray(abs(matrix).sum(axis=0)).reshape(-1)


def decoupled_algebraic_mask(
    *,
    ay: sparse.csc_matrix,
    gx: sparse.csc_matrix,
    gy: sparse.csc_matrix,
    tol: float = 0.0,
) -> np.ndarray:
    gy_rows, gy_cols = _offdiag_abs_sums(gy)
    row = gy_rows + _abs_row_sums(gx)
    col = gy_cols + _abs_col_sums(ay)
    return (row <= tol) & (col <= tol)


def _remove_decoupled_algebraic(
    *,
    ay: sparse.csc_matrix,
    gx: sparse.csc_matrix,
    gy: sparse.csc_matrix,
    by: sparse.csc_matrix,
    dy: sparse.csc_matrix,
    cy: sparse.csc_matrix,
    names: tuple[str, ...] | None,
    tol: float,
) -> tuple[
    sparse.csc_matrix,
    sparse.csc_matrix,
    sparse.csc_matrix,
    sparse.csc_matrix,
    sparse.csc_matrix,
    sparse.csc_matrix,
    tuple[str, ...] | None,
    tuple[str, ...],
    np.ndarray,
]:
    if tol < 0.0:
        raise LinearizationError("dead_algebraic_tol must be nonnegative")
    m = gy.shape[0]
    if m == 0:
        return ay, gx, gy, by, dy, cy, names, (), np.arange(0)
    decoupled = decoupled_algebraic_mask(ay=ay, gx=gx, gy=gy, tol=tol)
    if not np.any(decoupled):
        return ay, gx, gy, by, dy, cy, names, (), np.arange(m)
    label = names or tuple(f"y[{i}]" for i in range(m))
    diag = np.abs(gy.diagonal())
    forcing = _abs_row_sums(by) + _abs_row_sums(dy)
    output_use = _abs_col_sums(cy)
    keep = np.ones(m, dtype=bool)
    bad: list[str] = []
    for k in np.flatnonzero(decoupled):
        is_output = output_use[k] > 0.0
        has_diag = diag[k] > tol
        if is_output:
            if has_diag:
                continue
            bad.append(f"{label[k]} (selected output with an empty equation)")
            continue
        if not has_diag and forcing[k] > 0.0:
            bad.append(f"{label[k]} (forcing enters an empty equation 0 = d)")
            continue
        keep[k] = False
    if bad:
        raise LinearizationError(
            "G_y is singular or ill posed for decoupled algebraic variables: " + "; ".join(bad[:20])
        )
    keep_idx = np.flatnonzero(keep)
    removed = tuple(label[k] for k in np.flatnonzero(~keep))
    new_names = None if names is None else tuple(names[k] for k in keep_idx)
    return (
        _to_csc(ay[:, keep_idx], "A_y"),
        _to_csc(gx[keep_idx, :], "G_x"),
        _to_csc(gy[keep_idx, :][:, keep_idx], "G_y"),
        _to_csc(by[keep_idx, :], "B_y"),
        _to_csc(dy[keep_idx, :], "D_y"),
        _to_csc(cy[:, keep_idx], "C_y"),
        new_names,
        removed,
        keep_idx,
    )


def _pow2(values: np.ndarray) -> np.ndarray:
    out = np.ones_like(values, dtype=float)
    good = np.isfinite(values) & (values > 0.0)
    out[good] = np.exp2(np.round(np.log2(values[good])))
    return out


class AlgebraicFactor:
    def __init__(
        self,
        gy: Any,
        *,
        names: Sequence[str] | None = None,
        max_condition: float = DEFAULT_MAX_CONDITION,
        name_limit: int = 20,
    ) -> None:
        g = _as_csc(gy, "G_y")
        if g.shape[0] != g.shape[1]:
            raise LinearizationError("G_y must be square")
        m = g.shape[0]
        if m == 0:
            raise LinearizationError("Cannot factor empty G_y")
        self.shape = g.shape
        self.names = tuple(str(item) for item in names) if names is not None else tuple(
            f"y[{i}]" for i in range(m)
        )
        if len(self.names) != m:
            raise LinearizationError("G_y names length does not match G_y")
        self.name_limit = int(name_limit)

        empty_rows = np.flatnonzero(_abs_row_sums(g) == 0.0)
        empty_cols = np.flatnonzero(_abs_col_sums(g) == 0.0)
        if empty_rows.size or empty_cols.size:
            parts = []
            if empty_rows.size:
                parts.append("empty equations: " + ", ".join(self.names[i] for i in empty_rows[: self.name_limit]))
            if empty_cols.size:
                parts.append("variables absent from all equations: " + ", ".join(self.names[i] for i in empty_cols[: self.name_limit]))
            raise LinearizationError("G_y is singular (structurally): " + "; ".join(parts))

        row_scale = np.ones(m)
        col_scale = np.ones(m)
        work = g.copy().tocsc()
        for _ in range(3):
            r = _pow2(1.0 / np.asarray(abs(work).max(axis=1).toarray()).reshape(-1))
            work = _to_csc(sparse.diags(r) @ work, "G_y")
            row_scale *= r
            c = _pow2(1.0 / np.asarray(abs(work).max(axis=0).toarray()).reshape(-1))
            work = _to_csc(work @ sparse.diags(c), "G_y")
            col_scale *= c
        self.row_scale = row_scale
        self.col_scale = col_scale
        self._scaled = work
        try:
            self._lu = spla.splu(work)
        except (RuntimeError, ValueError) as exc:
            raise LinearizationError(
                "G_y is singular (exactly singular LU after equilibration); "
                + self._null_space_report(None)
            ) from exc
        u_diag = np.abs(np.asarray(self._lu.U.diagonal(), dtype=float))
        self.min_abs_pivot = float(np.min(u_diag)) if u_diag.size else None
        if not np.all(np.isfinite(u_diag)) or (u_diag.size and float(np.min(u_diag)) == 0.0):
            raise LinearizationError("G_y is singular (zero pivot); " + self._null_space_report(self._lu))
        norm1 = float(_abs_col_sums(work).max())
        try:
            inv_norm = float(
                spla.onenormest(
                    spla.LinearOperator(
                        work.shape,
                        matvec=lambda v: self._lu.solve(np.asarray(v, dtype=float).reshape(-1)),
                        rmatvec=lambda v: self._lu.solve(np.asarray(v, dtype=float).reshape(-1), trans="T"),
                        dtype=float,
                    )
                )
            )
        except Exception as exc:
            raise LinearizationError(f"G_y condition estimate failed: {exc}") from exc
        self.condition_estimate = norm1 * inv_norm
        if not math.isfinite(self.condition_estimate) or self.condition_estimate > float(max_condition):
            raise LinearizationError(
                f"G_y is singular or rank-deficient (equilibrated cond1 ~ {self.condition_estimate:.3e} "
                f"> {float(max_condition):.1e}); " + self._null_space_report(self._lu)
            )

    def solve(self, rhs: np.ndarray, trans: str = "N") -> np.ndarray:
        b = np.asarray(rhs, dtype=float)
        one_d = b.ndim == 1
        b2 = b.reshape(b.shape[0], -1)
        if trans == "N":
            sol = self.col_scale[:, None] * self._lu.solve(self.row_scale[:, None] * b2)
        else:
            sol = self.row_scale[:, None] * self._lu.solve(self.col_scale[:, None] * b2, trans="T")
        return sol.reshape(-1) if one_d else sol

    def _null_space_report(self, lu: Any) -> str:
        m = self.shape[0]
        try:
            if m <= 2500:
                dense = self._scaled.toarray()
                _, sv, vt = np.linalg.svd(dense)
                tol = max(sv[0] * m * np.finfo(float).eps, 1e-12 * sv[0])
                null = np.flatnonzero(sv <= max(tol, sv[0] / max(DEFAULT_MAX_CONDITION, 1.0)))
                if null.size == 0:
                    null = np.array([sv.size - 1])
                weight = np.sum(vt[null] ** 2, axis=0)
                rank_def = int(null.size)
            else:
                shift = 1e-10 * float(_abs_col_sums(self._scaled).max())
                shifted = spla.splu(_to_csc(self._scaled + shift * sparse.eye(m, format="csc"), "G_y"))
                rng = np.random.default_rng(0)
                v = rng.standard_normal((m, 4))
                for _ in range(6):
                    v = shifted.solve(v)
                    v /= np.linalg.norm(v, axis=0, keepdims=True)
                weight = np.sum(v**2, axis=1)
                rank_def = -1
        except Exception as exc:
            return f"null-space analysis unavailable ({exc})"
        order = np.argsort(-weight)
        top = [self.names[i] for i in order[: self.name_limit] if weight[i] > 1e-3 * weight[order[0]]]
        prefix = f"rank deficiency {rank_def}; " if rank_def >= 0 else ""
        return prefix + "variables in the near-null space: " + ", ".join(top)
