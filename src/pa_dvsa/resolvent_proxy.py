r"""Resolvent proxy for fast channel ranking.

The proxy from ``LaTeX/root.tex`` is

.. math::

   \widehat G_\nu(\tau) = F_\nu + C_r(1/\tau I - A_r)^{-1}B_\nu.

It is a speedup/ranking path by default.  It is used for disturbance safety
only as ``alpha_star * proxy`` when the corresponding positive channel is
verified monotone.  It is used as control authority only when it underestimates
the sampled beneficial response.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence

import numpy as np
from scipy import sparse
from scipy.sparse import linalg as spla

from .finite_window import FiniteWindowMapsResult, ProfileResponse
from .linearization import ReducedLinearModel
from .protected_outputs import ProtectedOutputEvaluation


ALPHA_STAR = 1.298425607525


class ResolventProxyError(RuntimeError):
    """Raised when the resolvent proxy cannot be evaluated consistently."""


@dataclass(frozen=True, slots=True)
class ResolventProxyResult:
    """Proxy values, conservative eligibility masks, and monotone statistics."""

    mode_id: str
    tau_s: float
    channel_type: str
    protected_asset_ids: tuple[str, ...]
    channel_ids: tuple[str, ...]
    proxy_pu: np.ndarray
    ranking_pu: np.ndarray
    monotone_mask: np.ndarray
    conservative_mask: np.ndarray
    monotone_fraction: float
    disturbance_upper_pu: np.ndarray | None = None
    disturbance_k_upper: np.ndarray | None = None
    control_authority_lower: np.ndarray | None = None
    channel_classes: Mapping[str, str] | None = None
    metadata: Mapping[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode_id": self.mode_id,
            "tau_s": self.tau_s,
            "channel_type": self.channel_type,
            "protected_asset_ids": list(self.protected_asset_ids),
            "channel_ids": list(self.channel_ids),
            "proxy_pu": _array_to_nested(self.proxy_pu),
            "ranking_pu": _array_to_nested(self.ranking_pu),
            "monotone_mask": self.monotone_mask.astype(bool).tolist(),
            "conservative_mask": self.conservative_mask.astype(bool).tolist(),
            "monotone_fraction": self.monotone_fraction,
            "disturbance_upper_pu": None
            if self.disturbance_upper_pu is None
            else _array_to_nested(self.disturbance_upper_pu, nonfinite_as_none=True),
            "disturbance_k_upper": None
            if self.disturbance_k_upper is None
            else _array_to_nested(self.disturbance_k_upper, nonfinite_as_none=True),
            "control_authority_lower": None
            if self.control_authority_lower is None
            else _array_to_nested(self.control_authority_lower),
            "channel_classes": dict(self.channel_classes or {}),
            "metadata": dict(self.metadata or {}),
        }


def alpha_star() -> float:
    """Return the monotone-channel proxy inflation constant."""

    return ALPHA_STAR


def ghat(
    reduced: ReducedLinearModel,
    tau_s: float,
    *,
    channel_type: str = "disturbance",
    input_matrix: Any | None = None,
) -> np.ndarray:
    """Evaluate ``Ghat(tau)`` for disturbances or controls."""

    tau = _positive_tau(tau_s)
    if channel_type == "disturbance":
        b = reduced.dr.tocsc()
        f = reduced.fd.tocsc()
    elif channel_type == "control":
        b = reduced.br.tocsc()
        f = reduced.fu.tocsc()
    else:
        raise ResolventProxyError("channel_type must be 'disturbance' or 'control'")

    a = reduced.ar.tocsc()
    c = reduced.cr.tocsc()
    n_states = a.shape[0]

    if b.shape[1] == 0:
        transfer = f.toarray()
    elif n_states == 0:
        transfer = f.toarray()
    else:
        system = (1.0 / tau) * sparse.eye(n_states, format="csc") - a
        try:
            solved = spla.splu(system).solve(b.toarray())
        except (RuntimeError, ValueError) as exc:
            raise ResolventProxyError("resolvent solve failed") from exc
        transfer = f.toarray() + c @ solved

    transfer = np.asarray(transfer, dtype=float)
    _require_finite(transfer, "Ghat")
    if input_matrix is None:
        return transfer
    inputs = _as_2d_array(input_matrix, "input_matrix")
    if inputs.shape[0] != transfer.shape[1]:
        raise ResolventProxyError(f"input_matrix has {inputs.shape[0]} rows, expected {transfer.shape[1]}")
    result = transfer @ inputs
    _require_finite(result, "Ghat input projection")
    return np.asarray(result, dtype=float)


def assess_disturbance_proxy(
    reduced: ReducedLinearModel,
    protected_outputs: Sequence[ProtectedOutputEvaluation],
    tau_s: float,
    *,
    disturbance_matrix: Any | None = None,
    event_ids: Sequence[str] | None = None,
    finite_response: ProfileResponse | None = None,
    finite_window: FiniteWindowMapsResult | None = None,
    monotone_tolerance: float = 1e-9,
    underestimate_tolerance: float = 1e-8,
    metadata: Mapping[str, Any] | None = None,
) -> ResolventProxyResult:
    """Assess disturbance proxy values and safe upper-bound eligibility."""

    tau = _positive_tau(tau_s)
    q = reduced.fd.shape[1]
    inputs = np.eye(q, dtype=float) if disturbance_matrix is None else _as_2d_array(
        disturbance_matrix, "disturbance_matrix"
    )
    ids = _channel_ids(event_ids, inputs.shape[1], "d")
    proxy = ghat(reduced, tau, channel_type="disturbance", input_matrix=inputs)
    _check_output_count(proxy, protected_outputs)

    response = finite_response or (None if finite_window is None else finite_window.disturbance_response)
    monotone_mask, sample_peak = _monotone_and_sample_peak(
        response,
        tau_s=tau,
        expected_shape=proxy.shape,
        beneficial=False,
        tolerance=monotone_tolerance,
    )
    positive_proxy = np.maximum(proxy, 0.0)
    alpha_upper = ALPHA_STAR * positive_proxy
    sample_consistent = sample_peak <= alpha_upper + abs(underestimate_tolerance)
    proxy_safe = monotone_mask & sample_consistent
    certified_margin = _certified_margin_mask(protected_outputs)
    proxy_safe &= certified_margin[:, None]

    upper = np.full_like(proxy, np.nan, dtype=float)
    classes = _initial_classes(protected_outputs, ids, "ranking_only")
    if finite_window is not None:
        finite_upper = _finite_window_upper(finite_window, proxy.shape)
        finite_classes = finite_window.channel_classes or {}
        for i, asset in enumerate(protected_outputs):
            for j, event_id in enumerate(ids):
                key = f"{asset.asset_id}:{event_id}"
                if certified_margin[i] and finite_classes.get(key) in {"certified", "monotone_proxy"}:
                    upper[i, j] = finite_upper[i, j]
                    classes[key] = "finite_window_upper"

    proxy_used = proxy_safe & ~np.isfinite(upper)
    upper[proxy_used] = alpha_upper[proxy_used]
    for i, asset in enumerate(protected_outputs):
        for j, event_id in enumerate(ids):
            key = f"{asset.asset_id}:{event_id}"
            if proxy_used[i, j]:
                classes[key] = "disturbance_alpha_star_proxy"
    _mark_data_limited(classes, protected_outputs, ids)

    margins = _margins(protected_outputs)
    k_upper = _normalize_optional(upper, margins)
    monotone_fraction = _fraction(monotone_mask)
    return ResolventProxyResult(
        mode_id=reduced.diagnostics.mode_id,
        tau_s=tau,
        channel_type="disturbance",
        protected_asset_ids=tuple(item.asset_id for item in protected_outputs),
        channel_ids=ids,
        proxy_pu=proxy,
        ranking_pu=positive_proxy,
        monotone_mask=monotone_mask,
        conservative_mask=proxy_safe,
        monotone_fraction=monotone_fraction,
        disturbance_upper_pu=upper,
        disturbance_k_upper=k_upper,
        channel_classes=classes,
        metadata={
            "alpha_star": ALPHA_STAR,
            "proxy_use": "ranking_only_unless_monotone_alpha_star_bound",
            **dict(metadata or {}),
        },
    )


def assess_control_proxy(
    reduced: ReducedLinearModel,
    protected_outputs: Sequence[ProtectedOutputEvaluation],
    tau_s: float,
    *,
    control_matrix: Any | None = None,
    control_ids: Sequence[str] | None = None,
    finite_response: ProfileResponse | None = None,
    finite_window: FiniteWindowMapsResult | None = None,
    monotone_tolerance: float = 1e-9,
    underestimate_tolerance: float = 1e-8,
    metadata: Mapping[str, Any] | None = None,
) -> ResolventProxyResult:
    """Assess control proxy values as conservative lower-bound authority."""

    tau = _positive_tau(tau_s)
    r = reduced.fu.shape[1]
    inputs = np.eye(r, dtype=float) if control_matrix is None else _as_2d_array(
        control_matrix, "control_matrix"
    )
    ids = _channel_ids(control_ids, inputs.shape[1], "u")
    proxy = ghat(reduced, tau, channel_type="control", input_matrix=inputs)
    _check_output_count(proxy, protected_outputs)

    response = finite_response or (None if finite_window is None else finite_window.control_response)
    monotone_mask, sampled_benefit = _monotone_and_sample_peak(
        response,
        tau_s=tau,
        expected_shape=proxy.shape,
        beneficial=True,
        tolerance=monotone_tolerance,
    )
    beneficial_proxy = np.maximum(-proxy, 0.0)
    underestimates = beneficial_proxy <= sampled_benefit + abs(underestimate_tolerance)
    conservative = monotone_mask & underestimates & (beneficial_proxy > 0.0)
    certified_margin = _certified_margin_mask(protected_outputs)
    conservative &= certified_margin[:, None]

    margins = _margins(protected_outputs)
    authority = np.zeros_like(proxy, dtype=float)
    good = certified_margin & (margins > 0.0)
    if np.any(good):
        authority[good, :] = beneficial_proxy[good, :] / margins[good, None]
    authority[~conservative] = 0.0

    classes = _initial_classes(protected_outputs, ids, "ranking_only")
    for i, asset in enumerate(protected_outputs):
        for j, control_id in enumerate(ids):
            key = f"{asset.asset_id}:{control_id}"
            if conservative[i, j]:
                classes[key] = "control_proxy_lower_bound"
    _mark_data_limited(classes, protected_outputs, ids)

    return ResolventProxyResult(
        mode_id=reduced.diagnostics.mode_id,
        tau_s=tau,
        channel_type="control",
        protected_asset_ids=tuple(item.asset_id for item in protected_outputs),
        channel_ids=ids,
        proxy_pu=proxy,
        ranking_pu=beneficial_proxy,
        monotone_mask=monotone_mask,
        conservative_mask=conservative,
        monotone_fraction=_fraction(monotone_mask),
        control_authority_lower=authority,
        channel_classes=classes,
        metadata={
            "proxy_use": "control_lower_bound_only_when_beneficial_proxy_underestimates",
            **dict(metadata or {}),
        },
    )


def _monotone_and_sample_peak(
    response: ProfileResponse | None,
    *,
    tau_s: float,
    expected_shape: tuple[int, int],
    beneficial: bool,
    tolerance: float,
) -> tuple[np.ndarray, np.ndarray]:
    mask = np.zeros(expected_shape, dtype=bool)
    peak = np.zeros(expected_shape, dtype=float)
    if response is None:
        return mask, peak
    values = np.asarray(response.values_pu, dtype=float)
    if values.shape[1:] != expected_shape:
        raise ResolventProxyError(
            f"response shape {values.shape[1:]} does not match proxy shape {expected_shape}"
        )
    times = np.asarray(response.times_s, dtype=float)
    if times.ndim != 1 or times.shape[0] != values.shape[0]:
        raise ResolventProxyError("response times must align with response samples")
    _require_finite(times, "response times")
    if np.any(np.diff(times) < -1e-12):
        raise ResolventProxyError("response times must be nondecreasing")
    time_mask = times <= tau_s + 1e-12
    if not np.any(time_mask):
        raise ResolventProxyError("finite response has no samples in proxy window")
    observed = values[time_mask]
    signal = -observed if beneficial else observed
    peak = np.max(np.maximum(signal, 0.0), axis=0)
    nonnegative = np.all(signal >= -abs(tolerance), axis=0)
    monotone = np.all(np.diff(signal, axis=0) >= -abs(tolerance), axis=0)
    mask = nonnegative & monotone
    return mask, peak


def _finite_window_upper(finite_window: FiniteWindowMapsResult, shape: tuple[int, int]) -> np.ndarray:
    upper = np.asarray(finite_window.pickup_excursion_pu, dtype=float)
    if upper.shape != shape:
        raise ResolventProxyError(f"finite-window upper shape {upper.shape} does not match {shape}")
    return upper


def _normalize_optional(values: np.ndarray, margins: np.ndarray) -> np.ndarray:
    normalized = np.full_like(values, np.nan, dtype=float)
    good = np.isfinite(values) & (margins[:, None] > 0.0)
    normalized[good] = np.maximum(values[good], 0.0) / np.broadcast_to(
        margins[:, None], values.shape
    )[good]
    return normalized


def _margins(protected_outputs: Sequence[ProtectedOutputEvaluation]) -> np.ndarray:
    margins = np.asarray([item.worst_case_margin_pu for item in protected_outputs], dtype=float)
    _require_finite(margins, "margins")
    return margins


def _certified_margin_mask(protected_outputs: Sequence[ProtectedOutputEvaluation]) -> np.ndarray:
    return np.asarray(
        [
            item.is_certifiable
            and math.isfinite(float(item.worst_case_margin_pu))
            and float(item.worst_case_margin_pu) > 0.0
            for item in protected_outputs
        ],
        dtype=bool,
    )


def _mark_data_limited(
    classes: dict[str, str],
    protected_outputs: Sequence[ProtectedOutputEvaluation],
    channel_ids: Sequence[str],
) -> None:
    for asset in protected_outputs:
        if not (
            asset.is_certifiable
            and math.isfinite(float(asset.worst_case_margin_pu))
            and float(asset.worst_case_margin_pu) > 0.0
        ):
            for channel_id in channel_ids:
                classes[f"{asset.asset_id}:{channel_id}"] = "data_limited"


def _check_output_count(proxy: np.ndarray, protected_outputs: Sequence[ProtectedOutputEvaluation]) -> None:
    if proxy.shape[0] != len(protected_outputs):
        raise ResolventProxyError(
            f"proxy has {proxy.shape[0]} output rows, got {len(protected_outputs)} protected outputs"
        )


def _positive_tau(tau_s: float) -> float:
    tau = float(tau_s)
    if not math.isfinite(tau) or tau <= 0.0:
        raise ResolventProxyError("tau_s must be finite and positive")
    return tau


def _as_2d_array(value: Any, name: str) -> np.ndarray:
    array = value.toarray() if sparse.issparse(value) else np.asarray(value, dtype=float)
    if array.ndim != 2:
        raise ResolventProxyError(f"{name} must be two-dimensional")
    _require_finite(array, name)
    return np.asarray(array, dtype=float)


def _channel_ids(ids: Sequence[str] | None, count: int, prefix: str) -> tuple[str, ...]:
    result = tuple(str(item) for item in ids) if ids is not None else tuple(
        f"{prefix}_{index}" for index in range(count)
    )
    if len(result) != count:
        raise ResolventProxyError(f"channel IDs has length {len(result)}, expected {count}")
    if any(not item.strip() for item in result):
        raise ResolventProxyError("channel IDs must be non-empty")
    if len(set(result)) != len(result):
        raise ResolventProxyError("channel IDs must be unique")
    return result


def _initial_classes(
    protected_outputs: Sequence[ProtectedOutputEvaluation],
    channel_ids: Sequence[str],
    default: str,
) -> dict[str, str]:
    return {
        f"{asset.asset_id}:{channel_id}": default
        for asset in protected_outputs
        for channel_id in channel_ids
    }


def _fraction(mask: np.ndarray) -> float:
    return 0.0 if mask.size == 0 else float(np.count_nonzero(mask) / mask.size)


def _require_finite(array: np.ndarray, name: str) -> None:
    if not np.all(np.isfinite(array)):
        raise ResolventProxyError(f"{name} contains non-finite values")


def _array_to_nested(array: np.ndarray, *, nonfinite_as_none: bool = False) -> list[list[float | None]]:
    rows: list[list[float | None]] = []
    for row in np.asarray(array, dtype=float):
        out_row: list[float | None] = []
        for value in row:
            if nonfinite_as_none and not math.isfinite(float(value)):
                out_row.append(None)
            else:
                out_row.append(float(value))
        rows.append(out_row)
    return rows
