"""Sparse and Krylov linear algebra for the event responses of the 2000-bus system: sparse LU reduction,
Lanczos log-norm with a Cholesky certificate, and shift-and-invert Krylov propagation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import time
from typing import Any, Sequence

import numpy as np
import scipy.linalg as sl
from scipy import sparse
from scipy.sparse import linalg as spla

from .data_model import TimeProfile
from .finite_window import ChannelResponse, FiniteWindowError, channel_grid, piecewise_linear_input
from .linearization import AlgebraicFactor, LinearDAEModel

KRYLOV_GAMMA = 0.02
KRYLOV_M_MIN = 30
KRYLOV_M_STEP = 10
KRYLOV_M_MAX = 200
KRYLOV_TOL_STATE_REL = 1e-11
KRYLOV_TOL_OUTPUT_ABS = 1e-12


class KrylovNotConverged(RuntimeError):
    pass


def schur_products(factor: AlgebraicFactor, gx: Any, ay: Any, cy: Any, *, chunk: int = 512
                   ) -> tuple[np.ndarray, np.ndarray]:
    from kvxopt import matrix, spmatrix
    import kvxopt.klu as klu

    g = factor._scaled.tocoo()
    gk = spmatrix(g.data.tolist(), g.row.tolist(), g.col.tolist(), g.shape)
    fs = klu.symbolic(gk)
    fn = klu.numeric(gk, fs)
    gx = sparse.csc_matrix(gx)
    gx.eliminate_zeros()
    n = gx.shape[1]
    cols = np.flatnonzero(np.diff(gx.indptr) > 0)
    gxs = sparse.csc_matrix(sparse.diags(factor.row_scale) @ gx)
    ays = sparse.csr_matrix(sparse.csr_matrix(ay) @ sparse.diags(factor.col_scale))
    cys = sparse.csr_matrix(sparse.csr_matrix(cy) @ sparse.diags(factor.col_scale))
    py = np.zeros((ays.shape[0], n))
    pc = np.zeros((cys.shape[0], n))
    for k in range(0, cols.size, int(chunk)):
        cc = cols[k : k + int(chunk)]
        b = matrix(gxs[:, cc].toarray(order="F"))
        klu.solve(gk, fs, fn, b)
        xc = np.asarray(b)
        if not np.all(np.isfinite(xc)):
            raise FiniteWindowError("G_y^-1 G_x is not finite")
        py[:, cc] = ays @ xc
        pc[:, cc] = cys @ xc
    return py, pc


def certified_lambda_max(s: np.ndarray, *, rel_delta: float = 1e-9) -> tuple[float, float]:
    n = s.shape[0]
    if n <= 50:
        return float(np.linalg.eigvalsh(s).max()), 0.0
    delta = float(rel_delta) * max(1.0, float(np.abs(s).sum(axis=0).max()))
    theta = float(spla.eigsh(s, k=1, which="LA", tol=1e-12, return_eigenvectors=False, maxiter=20000)[0])
    eye = np.eye(n)
    for _ in range(60):
        try:
            sl.cholesky((theta + delta) * eye - s, lower=True, check_finite=False)
            return theta + delta, delta
        except sl.LinAlgError:
            delta *= 4.0
    return float(np.linalg.eigvalsh(s).max()), 0.0


class DescriptorSolver:
    def __init__(self, model: LinearDAEModel, act: Sequence[int], alpha: float, beta: float) -> None:
        n, m = model.n_states, model.n_algebraic
        ax = sparse.csr_matrix(model.ax)
        ay = sparse.csr_matrix(model.ay)
        mask = np.zeros(n, dtype=bool)
        mask[np.asarray(act, dtype=int)] = True
        rows = sparse.diags(mask.astype(float))
        top_left = rows @ (float(alpha) * sparse.eye(n) - float(beta) * ax) + sparse.diags((~mask).astype(float))
        top_right = rows @ (-float(beta) * ay)
        j = sparse.bmat([[top_left, top_right], [sparse.csr_matrix(model.gx), sparse.csr_matrix(model.gy)]],
                        format="csc")
        self.lu = spla.splu(j)
        self.n, self.m = n, m
        self.act = np.asarray(act, dtype=int)

    def solve(self, r: np.ndarray) -> np.ndarray:
        rhs = np.zeros(self.n + self.m)
        rhs[self.act] = np.asarray(r, dtype=float).reshape(-1)
        return self.lu.solve(rhs)[: self.n][self.act]


class KrylovModeDynamics:
    def __init__(self, akk: np.ndarray, model: LinearDAEModel, keep: Sequence[int], *,
                 gamma: float = KRYLOV_GAMMA, dt_atomic: float = 2.5e-4) -> None:
        self.a = np.ascontiguousarray(np.asarray(akk, dtype=float))
        self.n = int(self.a.shape[0])
        self.dt_atomic = float(dt_atomic)
        self.need_ramp_blocks = True
        self.gamma = float(gamma)
        self.model = model
        self.keep = np.asarray(keep, dtype=int)
        t0 = time.perf_counter()
        scale = np.ones(self.n)
        if self.n:
            _, (sc, _) = sl.matrix_balance(self.a, permute=False, separate=True)
            sc = np.asarray(sc, dtype=float)
            if np.all(np.isfinite(sc)) and np.all(sc > 0.0):
                scale = sc
        self.scale = scale
        t1 = time.perf_counter()
        self.mu, self.mu_delta = 0.0, 0.0
        if self.n:
            ahat = self.a * (1.0 / scale)[:, None] * scale[None, :]
            self.mu, self.mu_delta = certified_lambda_max(0.5 * (ahat + ahat.T))
        self.timing_s = {"balance": t1 - t0, "log_norm": time.perf_counter() - t1}
        self._si: DescriptorSolver | None = None

    @property
    def mu_plus(self) -> float:
        return max(self.mu, 0.0)

    def growth(self, dt: float) -> float:
        return math.exp(self.mu_plus * float(dt))

    def scaled_norm(self, vectors: np.ndarray) -> np.ndarray:
        v = np.asarray(vectors, dtype=float)
        if v.ndim == 1:
            return np.asarray(np.linalg.norm(v / self.scale))
        return np.linalg.norm(v / self.scale[:, None], axis=0)

    def output_scaled_norms(self, c: np.ndarray) -> np.ndarray:
        c = np.atleast_2d(np.asarray(c, dtype=float))
        if c.shape[1] != self.n:
            raise FiniteWindowError(f"C has {c.shape[1]} columns, expected {self.n}")
        return np.linalg.norm(c * self.scale[None, :], axis=1) if self.n else np.zeros(c.shape[0])

    def si_solver(self) -> DescriptorSolver:
        if self._si is None:
            t0 = time.perf_counter()
            self._si = DescriptorSolver(self.model, self.keep, 1.0, self.gamma)
            self.timing_s["si_factor"] = time.perf_counter() - t0
        return self._si

    def blocks(self, dt: float) -> Any:
        raise FiniteWindowError("the Krylov path has no dense propagation blocks")


@dataclass
class KrylovSegment:
    t0: float
    t1: float
    v: np.ndarray
    ht: np.ndarray
    y0: np.ndarray
    m: int
    est_state_rel: float
    est_output_abs: float
    _expm: dict = field(default_factory=dict, repr=False)

    def coeffs(self, taus: np.ndarray) -> np.ndarray:
        taus = np.asarray(taus, dtype=float)
        out = np.zeros((taus.size, self.m))
        y = self.y0.copy()
        prev = 0.0
        for k in np.argsort(taus, kind="stable"):
            d = round(float(taus[k]) - prev, 13)
            if d > 0.0:
                if d not in self._expm:
                    self._expm[d] = sl.expm(d * self.ht)
                y = self._expm[d] @ y
                prev = float(taus[k])
            out[k] = y
        return out

    def states(self, taus: np.ndarray, n: int) -> np.ndarray:
        return self.coeffs(taus) @ self.v[:n].T


def krylov_segment(mode: KrylovModeDynamics, b: np.ndarray, c: np.ndarray, w0: np.ndarray, taus: np.ndarray, *,
                   m_min: int = KRYLOV_M_MIN, m_step: int = KRYLOV_M_STEP, m_max: int = KRYLOV_M_MAX,
                   tol_state_rel: float = KRYLOV_TOL_STATE_REL, tol_output_abs: float = KRYLOV_TOL_OUTPUT_ABS
                   ) -> KrylovSegment:
    n = mode.n
    gamma = mode.gamma
    solver = mode.si_solver()

    def op(v: np.ndarray) -> np.ndarray:
        zs = v[n + 1]
        zu = v[n] + gamma * zs
        zx = solver.solve(v[:n] + gamma * b * zu)
        return np.concatenate((zx, [zu, zs]))

    beta = float(np.linalg.norm(w0))
    basis = np.zeros((n + 2, m_max + 1))
    hess = np.zeros((m_max + 1, m_max))
    basis[:, 0] = w0 / beta
    dsc = mode.scale
    prev: np.ndarray | None = None
    m = 0
    while m < m_max:
        w = op(basis[:, m])
        for _ in range(2):
            coef = basis[:, : m + 1].T @ w
            w -= basis[:, : m + 1] @ coef
            hess[: m + 1, m] += coef
        hess[m + 1, m] = np.linalg.norm(w)
        m += 1
        breakdown = hess[m, m - 1] <= 1e-14 * max(1.0, float(np.abs(hess[:m, :m]).max()))
        if not breakdown:
            basis[:, m] = w / hess[m, m - 1]
        if not (breakdown or (m >= m_min and (m - m_min) % m_step == 0)):
            continue
        ht = (np.eye(m) - np.linalg.inv(hess[:m, :m])) / gamma
        y0 = np.zeros(m)
        y0[0] = beta
        seg = KrylovSegment(0.0, 0.0, basis[:, :m].copy(), ht, y0, m, math.inf, math.inf)
        xs = seg.states(taus, n)
        if breakdown:
            seg.est_state_rel, seg.est_output_abs = 0.0, 0.0
            return seg
        if prev is not None:
            ref = max(float(np.max(np.linalg.norm(xs / dsc, axis=1))), 1e-300)
            d_state = float(np.max(np.linalg.norm((xs - prev) / dsc, axis=1))) / ref
            d_out = float(np.max(np.abs((xs - prev) @ c.T))) if c.size else 0.0
            if d_state <= tol_state_rel and d_out <= tol_output_abs:
                seg.est_state_rel, seg.est_output_abs = d_state, d_out
                return seg
        prev = xs
    raise KrylovNotConverged(f"shift-and-invert Krylov did not converge in {m_max} steps")


class KrylovChannelResponse(ChannelResponse):
    segments: list = None

    def refine(self, interval: int, parts: int) -> ChannelResponse:
        key = (int(interval), int(parts))
        if key in self.refine_cache:
            return self.refine_cache[key]
        if self.mode is None or self.states is None or self.c is None or self.b is None:
            raise FiniteWindowError("channel response was released; refinement impossible")
        t0 = float(self.times_s[interval])
        t1 = float(self.times_s[interval + 1])
        sub = np.linspace(t0, t1, int(parts) + 1)
        u0 = float(self.u_right[interval])
        sig = float(self.sigma[interval])
        u_right = u0 + sig * (sub - t0)
        u_left = u_right.copy()
        u_left[0] = float(self.u_left[interval])
        u_right[-1] = float(self.u_right[interval + 1])
        u_left[-1] = u0 + sig * (t1 - t0)
        seg = _segment_for(self.segments, t0, t1)
        states = seg.states(sub - seg.t0, self.mode.n)
        states[0] = self.states[interval]
        out = channel_from_states(self.mode, self.c, self.b, self.f, sub, u_left, u_right, np.full(int(parts), sig),
                                  states, channel_id=f"{self.channel_id}[{interval}]", segments=[seg],
                                  c_norms=self.c_norms, b_norm=self.b_norm)
        self.refine_cache[key] = out
        return out

    def release(self) -> None:
        super().release()
        self.segments = []


def _segment_for(segments: Sequence[KrylovSegment], t0: float, t1: float) -> KrylovSegment:
    for s in segments:
        if s.t0 - 1e-12 <= t0 and t1 <= s.t1 + 1e-12:
            return s
    raise FiniteWindowError(f"no Krylov segment covers [{t0}, {t1}]")


def channel_from_states(mode: KrylovModeDynamics, c: np.ndarray, b: np.ndarray, f: np.ndarray, grid: np.ndarray,
                        u_left: np.ndarray, u_right: np.ndarray, sigma: np.ndarray, states: np.ndarray, *,
                        channel_id: str, segments: list, c_norms: np.ndarray | None = None,
                        b_norm: float | None = None) -> KrylovChannelResponse:
    xt = states.T
    slope = np.zeros(grid.size)
    slope[: sigma.size] = sigma
    xd = mode.a @ xt + np.outer(b, u_right)
    xdd = mode.a @ xd + np.outer(b, slope)
    cx = (c @ xt).T
    phi_r = cx + np.outer(u_right, f)
    if not np.all(np.isfinite(phi_r)):
        raise FiniteWindowError(f"{channel_id} response is not finite")
    resp = KrylovChannelResponse(
        channel_id=channel_id, times_s=np.asarray(grid, dtype=float), u_left=np.asarray(u_left, dtype=float),
        u_right=np.asarray(u_right, dtype=float), sigma=np.asarray(sigma, dtype=float),
        phi_left=cx + np.outer(u_left, f), phi_right=phi_r, cxdot=(c @ xd).T,
        xdot_norm=np.asarray(mode.scaled_norm(xd)), xddot_norm=np.asarray(mode.scaled_norm(xdd)),
        c_norms=mode.output_scaled_norms(c) if c_norms is None else c_norms, f=f,
        b_norm=float(mode.scaled_norm(b)) if b_norm is None else float(b_norm), mu=mode.mu, states=states,
        mode=mode, c=c, b=b,
    )
    resp.segments = segments
    return resp


def propagate_channel_krylov(mode: KrylovModeDynamics, c: Any, b: Any, f: Any, profile: TimeProfile,
                             base_grid: Sequence[float], *, channel_id: str = "channel"
                             ) -> tuple[KrylovChannelResponse, dict[str, Any]]:
    grid = channel_grid(base_grid, profile)
    u_left, u_right, sigma = piecewise_linear_input(profile, grid)
    c = np.atleast_2d(np.asarray(c.toarray() if sparse.issparse(c) else c, dtype=float))
    b = np.asarray(b, dtype=float).reshape(-1)
    f = np.asarray(f, dtype=float).reshape(-1)
    n = mode.n
    if c.shape[1] != n or b.size != n or f.size != c.shape[0]:
        raise FiniteWindowError("channel dimensions do not match the mode")
    tol_s = 1e-9 * max(1.0, float(np.max(np.abs(sigma))) if sigma.size else 1.0)
    tol_u = 1e-12 * max(1.0, float(np.max(np.abs(u_right))))
    starts = [0]
    for l in range(1, grid.size - 1):
        if abs(sigma[l] - sigma[l - 1]) > tol_s or abs(u_left[l] - u_right[l]) > tol_u:
            starts.append(l)
    starts.append(grid.size - 1)
    states = np.zeros((grid.size, n))
    segments: list[KrylovSegment] = []
    x = np.zeros(n)
    for s0, s1 in zip(starts[:-1], starts[1:]):
        t0 = float(grid[s0])
        taus = grid[s0 : s1 + 1] - t0
        w0 = np.concatenate((x, [u_right[s0], sigma[s0]]))
        if not np.any(w0):
            seg = KrylovSegment(t0, float(grid[s1]), np.zeros((n + 2, 1)), np.zeros((1, 1)), np.zeros(1), 1, 0.0, 0.0)
        else:
            seg = krylov_segment(mode, b, c, w0, taus)
            seg.t0, seg.t1 = t0, float(grid[s1])
        segx = seg.states(taus, n)
        states[s0 : s1 + 1] = segx
        x = segx[-1].copy()
        segments.append(seg)
    resp = channel_from_states(mode, c, b, f, grid, u_left, u_right, sigma, states, channel_id=channel_id,
                               segments=segments)
    info = {"krylov_segments": len(segments), "krylov_m_max": max(s.m for s in segments),
            "krylov_est_state_rel": max(s.est_state_rel for s in segments),
            "krylov_est_output_abs_pu": max(s.est_output_abs for s in segments)}
    return resp, info


def near_zero_eigenvalues_descriptor(model: LinearDAEModel, keep: Sequence[int], akk: np.ndarray, *, k: int = 12,
                                     sigma: float = 1e-3, tol: float = 1e-10) -> tuple[np.ndarray, float, int]:
    act = np.ones(akk.shape[0], dtype=bool)
    while True:
        sub = akk[np.ix_(act, act)]
        zero_rows = np.flatnonzero(~np.any(sub != 0.0, axis=1))
        if not zero_rows.size:
            break
        act[np.flatnonzero(act)[zero_rows]] = False
    sub = akk[np.ix_(act, act)]
    ndef = int(np.count_nonzero(~act))
    if sub.shape[0] <= k + 2:
        vals = np.linalg.eigvals(sub)
    else:
        solver = DescriptorSolver(model, np.asarray(keep, dtype=int)[act], -float(sigma), -1.0)
        opinv = spla.LinearOperator(sub.shape, matvec=lambda v: solver.solve(np.asarray(v, dtype=float).reshape(-1)),
                                    dtype=float)
        vals = spla.eigs(sub, k=k, sigma=sigma, which="LM", return_eigenvectors=False, tol=tol, maxiter=5000,
                         OPinv=opinv)
    radius = float(np.max(np.abs(vals - sigma))) if vals.size else 0.0
    if ndef:
        vals = np.concatenate((vals, np.zeros(ndef)))
    return np.asarray(vals), radius, ndef
