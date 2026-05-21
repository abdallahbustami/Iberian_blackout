r"""Zero-delay algebraic overvoltage screen.

This module implements the zero-delay limit from ``LaTeX/root.tex``:

.. math::

   K_{ij}^{m,0^+} =
   \frac{[e_i^\top F_d^m \bar d_j]_+}{h_i^m}.

It also keeps a direct algebraic shortcut for explicit collector/fixed-tap
studies:

.. math::

   K_{ij} = \frac{[S_{ij} Q_j^{abs} / n_i]_+}{h_i}.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence

import numpy as np
from scipy import sparse
from scipy.sparse import linalg as spla

from .data_model import (
    AssessmentStatus,
    ChannelAssessment,
    Matrix,
    WindowMapResult,
)
from .linearization import ReducedLinearModel
from .protected_outputs import ProtectedOutputEvaluation


class ZeroDelayScreenError(RuntimeError):
    """Raised when the zero-delay screen cannot be built consistently."""


@dataclass(frozen=True, slots=True)
class ZeroDelayScreenResult:
    """Detailed zero-delay screen plus the normalized window-map result."""

    window_result: WindowMapResult
    instant_delta_z_pu: Matrix
    margins_pu: tuple[float, ...]
    sensitivity_pu_per_pu_q: Matrix | None = None
    one_step_safe: Mapping[str, bool] | None = None
    max_offdiag_per_event: Mapping[str, float] | None = None
    spectral_radius: float | None = None
    metadata: Mapping[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "window_result": self.window_result.to_dict(),
            "instant_delta_z_pu": [list(row) for row in self.instant_delta_z_pu],
            "margins_pu": list(self.margins_pu),
            "sensitivity_pu_per_pu_q": None
            if self.sensitivity_pu_per_pu_q is None
            else [list(row) for row in self.sensitivity_pu_per_pu_q],
            "one_step_safe": dict(self.one_step_safe or {}),
            "max_offdiag_per_event": dict(self.max_offdiag_per_event or {}),
            "spectral_radius": self.spectral_radius,
            "metadata": dict(self.metadata or {}),
        }


def compute_zero_delay_screen(
    reduced: ReducedLinearModel,
    protected_outputs: Sequence[ProtectedOutputEvaluation],
    *,
    event_ids: Sequence[str] | None = None,
    disturbance_matrix: Any | None = None,
    mode_id: str | None = None,
    strict_margins: bool = False,
    metadata: Mapping[str, Any] | None = None,
) -> ZeroDelayScreenResult:
    """Compute ``K^{0+}`` directly from a reduced model's ``F_d`` block.

    Parameters
    ----------
    reduced:
        Reduced linear model. ``reduced.fd`` maps disturbance coordinates to
        instantaneous protected-output changes.
    protected_outputs:
        Protected-output evaluations. Their worst-case positive margins are
        the denominators ``h_i``.
    event_ids:
        IDs for disturbance columns. If omitted, the disturbance coordinates are
        named ``d_0``, ``d_1``, ...
    disturbance_matrix:
        Optional matrix whose columns are event vectors ``\bar d_j``. If omitted,
        the identity matrix is used, meaning each disturbance coordinate is one
        candidate event.
    strict_margins:
        If true, any nonpositive/uncertified margin raises. Otherwise the row is
        marked data-limited and kept out of certification.
    """

    fd = _as_2d_array(reduced.fd, "F_d")
    n_outputs, n_disturbances = fd.shape
    _require_protected_count(protected_outputs, n_outputs)

    dbar = (
        np.eye(n_disturbances, dtype=float)
        if disturbance_matrix is None
        else _as_2d_array(disturbance_matrix, "disturbance_matrix")
    )
    if dbar.shape[0] != n_disturbances:
        raise ZeroDelayScreenError(
            f"disturbance_matrix has {dbar.shape[0]} rows, expected {n_disturbances}"
        )

    ids = _event_ids(event_ids, dbar.shape[1])
    margins, data_limited_assets, assessments = _margin_data(
        protected_outputs, ids, strict_margins=strict_margins
    )

    delta_z = fd @ dbar
    k = _normalize_by_margins(delta_z, margins)
    status = _overall_status(data_limited_assets)
    window = _window_result(
        mode_id=mode_id or reduced.diagnostics.mode_id,
        protected_asset_ids=[item.asset_id for item in protected_outputs],
        event_ids=ids,
        k=k,
        assessments=assessments,
        data_limited_assets=data_limited_assets,
        status=status,
        metadata={
            "formula": "K_ij^{0+} = [e_i^T F_d dbar_j]_+ / h_i",
            "source": "reduced_F_d",
            **dict(metadata or {}),
        },
    )
    safe, offdiag = _one_step_safety(k, ids, data_limited=bool(data_limited_assets))
    return ZeroDelayScreenResult(
        window_result=window,
        instant_delta_z_pu=_matrix_tuple(delta_z),
        margins_pu=tuple(float(item) for item in margins),
        one_step_safe=safe,
        max_offdiag_per_event=offdiag,
        spectral_radius=_spectral_radius(k),
        metadata={"source": "reduced_F_d", **dict(metadata or {})},
    )


def algebraic_sensitivity_from_gy(
    gy: Any,
    *,
    output_addresses: Sequence[int],
    injection_equation_addresses: Sequence[int],
    sign: float | None = None,
    clamp_negative: bool = True,
    negative_tol: float = 1e-10,
) -> np.ndarray:
    """Return selected algebraic sensitivities using sparse solves on ``G_y``.

    This implements the robust fixed-tap algebraic screen: solve
    ``G_y s_j = e_j`` for selected reactive-balance columns, read
    selected voltage rows, normalize the sign from the diagonal when not given,
    and optionally clamp negative numerical artifacts to zero.
    """

    gy_c = _as_sparse_square(gy, "G_y")
    rows = _checked_addresses(output_addresses, gy_c.shape[0], "output_addresses")
    cols = _checked_addresses(
        injection_equation_addresses, gy_c.shape[0], "injection_equation_addresses"
    )
    if negative_tol < 0.0:
        raise ZeroDelayScreenError("negative_tol must be nonnegative")

    rhs = np.zeros((gy_c.shape[0], len(cols)), dtype=float)
    for col_index, address in enumerate(cols):
        rhs[address, col_index] = 1.0

    try:
        solved = spla.splu(gy_c).solve(rhs)
    except (RuntimeError, ValueError) as exc:
        raise ZeroDelayScreenError("G_y is singular or unsuitable for algebraic sensitivity") from exc
    if not np.all(np.isfinite(solved)):
        raise ZeroDelayScreenError("algebraic sensitivity solve produced non-finite values")

    raw = np.asarray(solved[rows, :], dtype=float)
    physical_sign = _infer_sensitivity_sign(raw) if sign is None else float(sign)
    if physical_sign not in (-1.0, 1.0):
        raise ZeroDelayScreenError("sign must be +1 or -1")
    sensitivity = physical_sign * raw

    if clamp_negative:
        bad = sensitivity < -abs(negative_tol)
        if np.any(bad):
            sensitivity = sensitivity.copy()
            sensitivity[bad] = 0.0
        sensitivity = np.maximum(sensitivity, 0.0)
    _require_finite_array(sensitivity, "sensitivity")
    return sensitivity


def compute_fixed_tap_zero_delay_screen(
    sensitivity_pu_per_pu_q: Any,
    protected_outputs: Sequence[ProtectedOutputEvaluation],
    *,
    q_absorption_mvar: Sequence[float],
    tap_ratios: Sequence[float],
    event_ids: Sequence[str] | None = None,
    base_mva: float = 100.0,
    mode_id: str = "zero_delay_fixed_tap",
    strict_margins: bool = False,
    metadata: Mapping[str, Any] | None = None,
) -> ZeroDelayScreenResult:
    """Compute the fixed-tap shortcut ``[S_ij Q_j^abs / n_i]_+ / h_i``."""

    sensitivity = _as_2d_array(sensitivity_pu_per_pu_q, "sensitivity_pu_per_pu_q")
    n_outputs, n_events = sensitivity.shape
    _require_protected_count(protected_outputs, n_outputs)
    ids = _event_ids(event_ids, n_events)

    base = float(base_mva)
    if not math.isfinite(base) or base <= 0.0:
        raise ZeroDelayScreenError("base_mva must be finite and positive")

    q_abs = _as_vector(q_absorption_mvar, n_events, "q_absorption_mvar")
    if np.any(q_abs < 0.0):
        raise ZeroDelayScreenError("q_absorption_mvar must be nonnegative")
    taps = _as_vector(tap_ratios, n_outputs, "tap_ratios")
    if np.any(taps <= 0.0):
        raise ZeroDelayScreenError("tap_ratios must be positive")

    margins, data_limited_assets, assessments = _margin_data(
        protected_outputs, ids, strict_margins=strict_margins
    )
    q_abs_pu = q_abs / base
    delta_z = sensitivity * q_abs_pu[None, :] / taps[:, None]
    k = _normalize_by_margins(delta_z, margins)
    status = _overall_status(data_limited_assets)
    window = _window_result(
        mode_id=mode_id,
        protected_asset_ids=[item.asset_id for item in protected_outputs],
        event_ids=ids,
        k=k,
        assessments=assessments,
        data_limited_assets=data_limited_assets,
        status=status,
        metadata={
            "formula": "K_ij = [S_ij Q_j^abs / n_i]_+ / h_i",
            "source": "fixed_tap_algebraic_shortcut",
            "base_mva": base,
            **dict(metadata or {}),
        },
    )
    safe, offdiag = _one_step_safety(k, ids, data_limited=bool(data_limited_assets))
    return ZeroDelayScreenResult(
        window_result=window,
        instant_delta_z_pu=_matrix_tuple(delta_z),
        margins_pu=tuple(float(item) for item in margins),
        sensitivity_pu_per_pu_q=_matrix_tuple(sensitivity),
        one_step_safe=safe,
        max_offdiag_per_event=offdiag,
        spectral_radius=_spectral_radius(k),
        metadata={
            "source": "fixed_tap_algebraic_shortcut",
            "base_mva": base,
            "q_absorption_pu": [float(item) for item in q_abs_pu],
            **dict(metadata or {}),
        },
    )


def compute_fixed_tap_screen_from_gy(
    gy: Any,
    protected_outputs: Sequence[ProtectedOutputEvaluation],
    *,
    output_addresses: Sequence[int],
    injection_equation_addresses: Sequence[int],
    q_absorption_mvar: Sequence[float],
    tap_ratios: Sequence[float],
    event_ids: Sequence[str] | None = None,
    base_mva: float = 100.0,
    mode_id: str = "zero_delay_fixed_tap",
    strict_margins: bool = False,
    sign: float | None = None,
    clamp_negative: bool = True,
    metadata: Mapping[str, Any] | None = None,
) -> ZeroDelayScreenResult:
    """Build the direct fixed-tap algebraic screen from ``G_y``."""

    sensitivity = algebraic_sensitivity_from_gy(
        gy,
        output_addresses=output_addresses,
        injection_equation_addresses=injection_equation_addresses,
        sign=sign,
        clamp_negative=clamp_negative,
    )
    return compute_fixed_tap_zero_delay_screen(
        sensitivity,
        protected_outputs,
        q_absorption_mvar=q_absorption_mvar,
        tap_ratios=tap_ratios,
        event_ids=event_ids,
        base_mva=base_mva,
        mode_id=mode_id,
        strict_margins=strict_margins,
        metadata=metadata,
    )


def _require_protected_count(
    protected_outputs: Sequence[ProtectedOutputEvaluation],
    expected: int,
) -> None:
    if len(protected_outputs) != expected:
        raise ZeroDelayScreenError(
            f"expected {expected} protected outputs, got {len(protected_outputs)}"
        )


def _margin_data(
    protected_outputs: Sequence[ProtectedOutputEvaluation],
    event_ids: Sequence[str],
    *,
    strict_margins: bool,
) -> tuple[np.ndarray, tuple[str, ...], tuple[ChannelAssessment, ...]]:
    margins = np.array([item.worst_case_margin_pu for item in protected_outputs], dtype=float)
    _require_finite_array(margins, "margins")
    bad_assets = tuple(
        item.asset_id
        for item, margin in zip(protected_outputs, margins)
        if (not item.is_certifiable) or margin <= 0.0
    )
    if bad_assets and strict_margins:
        raise ZeroDelayScreenError(
            "zero-delay screen requires positive certified margins; "
            f"data-limited assets: {', '.join(bad_assets)}"
        )

    assessments: list[ChannelAssessment] = []
    bad_set = set(bad_assets)
    for output in protected_outputs:
        status = AssessmentStatus.DATA_LIMITED if output.asset_id in bad_set else AssessmentStatus.CERTIFIED
        notes = None if output.asset_id not in bad_set else "nonpositive_or_uncertified_margin"
        for event_id in event_ids:
            assessments.append(
                ChannelAssessment(
                    protected_asset_id=output.asset_id,
                    channel_id=event_id,
                    status=status,
                    bound_type="zero_delay_pickup",
                    notes=notes,
                )
            )
    return margins, bad_assets, tuple(assessments)


def _normalize_by_margins(delta_z: np.ndarray, margins: np.ndarray) -> np.ndarray:
    _require_finite_array(delta_z, "instant_delta_z")
    if delta_z.shape[0] != margins.size:
        raise ZeroDelayScreenError(
            f"instant_delta_z has {delta_z.shape[0]} rows, expected {margins.size}"
        )
    k = np.zeros_like(delta_z, dtype=float)
    good = margins > 0.0
    if np.any(good):
        k[good, :] = np.maximum(delta_z[good, :], 0.0) / margins[good, None]
    _require_finite_array(k, "K_zero_delay")
    return k


def _window_result(
    *,
    mode_id: str,
    protected_asset_ids: Sequence[str],
    event_ids: Sequence[str],
    k: np.ndarray,
    assessments: Sequence[ChannelAssessment],
    data_limited_assets: Sequence[str],
    status: AssessmentStatus,
    metadata: Mapping[str, Any],
) -> WindowMapResult:
    return WindowMapResult(
        mode_id=mode_id,
        protected_asset_ids=tuple(protected_asset_ids),
        event_ids=tuple(event_ids),
        control_ids=(),
        horizons_s=tuple(0.0 for _ in protected_asset_ids),
        k_pickup_upper=_matrix_tuple(k),
        k_trip_upper=_matrix_tuple(k),
        channel_assessments=tuple(assessments),
        data_limited_assets=tuple(data_limited_assets),
        status=status,
        metadata=metadata,
    )


def _overall_status(data_limited_assets: Sequence[str]) -> AssessmentStatus:
    return AssessmentStatus.DATA_LIMITED if data_limited_assets else AssessmentStatus.CERTIFIED


def _one_step_safety(
    k: np.ndarray,
    event_ids: Sequence[str],
    *,
    data_limited: bool = False,
) -> tuple[dict[str, bool], dict[str, float]]:
    safe: dict[str, bool] = {}
    max_offdiag: dict[str, float] = {}
    for j, event_id in enumerate(event_ids):
        column = np.asarray(k[:, j], dtype=float).copy()
        if column.size == len(event_ids):
            column[j] = 0.0
        max_value = float(np.max(column)) if column.size else 0.0
        max_offdiag[event_id] = max_value
        safe[event_id] = bool(max_value < 1.0 and not data_limited)
    return safe, max_offdiag


def _spectral_radius(k: np.ndarray) -> float | None:
    if k.shape[0] != k.shape[1] or k.size == 0:
        return None
    eigvals = np.linalg.eigvals(k)
    return float(np.max(np.abs(eigvals))) if eigvals.size else 0.0


def _infer_sensitivity_sign(raw: np.ndarray) -> float:
    diag_len = min(raw.shape)
    if diag_len == 0:
        return 1.0
    diagonal = np.diag(raw[:diag_len, :diag_len])
    median = float(np.median(diagonal)) if diagonal.size else 0.0
    return 1.0 if median >= 0.0 else -1.0


def _event_ids(event_ids: Sequence[str] | None, expected: int) -> tuple[str, ...]:
    ids = tuple(str(item) for item in event_ids) if event_ids is not None else tuple(
        f"d_{index}" for index in range(expected)
    )
    if len(ids) != expected:
        raise ZeroDelayScreenError(f"expected {expected} event IDs, got {len(ids)}")
    if any(not item.strip() for item in ids):
        raise ZeroDelayScreenError("event IDs must be non-empty")
    if len(set(ids)) != len(ids):
        raise ZeroDelayScreenError("event IDs must be unique")
    return ids


def _checked_addresses(addresses: Sequence[int], width: int, name: str) -> tuple[int, ...]:
    checked: list[int] = []
    for address in addresses:
        value = int(address)
        if value < 0 or value >= width:
            raise ZeroDelayScreenError(f"{name} address {value} is outside width {width}")
        checked.append(value)
    return tuple(checked)


def _as_vector(values: Sequence[float], expected: int, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float).reshape(-1)
    if array.size != expected:
        raise ZeroDelayScreenError(f"{name} has length {array.size}, expected {expected}")
    _require_finite_array(array, name)
    return array


def _as_2d_array(value: Any, name: str) -> np.ndarray:
    if sparse.issparse(value):
        array = value.toarray()
    else:
        array = np.asarray(value, dtype=float)
    if array.ndim != 2:
        raise ZeroDelayScreenError(f"{name} must be two-dimensional")
    _require_finite_array(array, name)
    return np.asarray(array, dtype=float)


def _as_sparse_square(value: Any, name: str) -> sparse.csc_matrix:
    try:
        matrix = sparse.csc_matrix(value, dtype=float)
    except Exception as exc:
        raise ZeroDelayScreenError(f"{name} cannot be converted to a sparse matrix") from exc
    if matrix.shape[0] != matrix.shape[1]:
        raise ZeroDelayScreenError(f"{name} must be square")
    if matrix.shape[0] == 0:
        raise ZeroDelayScreenError(f"{name} must be nonempty")
    data = matrix.data
    if data.size and not np.all(np.isfinite(data)):
        raise ZeroDelayScreenError(f"{name} contains non-finite entries")
    return matrix.tocsc()


def _require_finite_array(array: np.ndarray, name: str) -> None:
    if not np.all(np.isfinite(array)):
        raise ZeroDelayScreenError(f"{name} contains non-finite entries")


def _matrix_tuple(array: np.ndarray) -> Matrix:
    return tuple(tuple(float(item) for item in row) for row in np.asarray(array, dtype=float))
