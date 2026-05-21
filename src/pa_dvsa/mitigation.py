r"""Mitigation LP for protection-aware voltage security.

This module implements the LP form of ``LaTeX/root.tex`` equation
``eq:mitigation_qp``:

.. math::

   \min_{\alpha,\eta} c^\top\alpha+\rho\eta

subject to time-grid constraints

.. math::

   \bar r_i^{0,m}+\bar m_i^S(t_\ell)
   -\sum_k \underline H_{ik}^m(t_\ell)\alpha_k
   \le 1-\epsilon+\eta.

The implementation intentionally requires time-resolved
``TimeGridControlAuthority`` matrices.  Scalar control rankings are useful for
sorting controls but are not accepted as mitigation certificate constraints.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.optimize import linprog

from .data_model import (
    AssessmentStatus,
    BindingConstraint,
    ControlSelection,
    Matrix,
    MitigationResult,
    TimeGridControlAuthority,
    WindowMapResult,
)
from .finite_window import FiniteWindowMapsResult, ProfileResponse
from .protected_outputs import ProtectedOutputEvaluation
from .robust_screening import RobustScreenResult


class MitigationError(RuntimeError):
    """Raised when the mitigation LP cannot be formed consistently."""


@dataclass(frozen=True, slots=True)
class DisturbanceMarginProfile:
    """Time-resolved normalized disturbance erosion ``m_i^S(t_l)``."""

    times_s: tuple[float, ...]
    upper_erosion: Matrix

    def __post_init__(self) -> None:
        times = tuple(float(item) for item in self.times_s)
        if any(not math.isfinite(item) or item < 0.0 for item in times):
            raise MitigationError("disturbance profile times must be finite and nonnegative")
        if any(b < a for a, b in zip(times, times[1:])):
            raise MitigationError("disturbance profile times must be nondecreasing")
        rows = tuple(tuple(float(item) for item in row) for row in self.upper_erosion)
        if len(rows) != len(times):
            raise MitigationError("disturbance profile must have one row per time")
        if any(any(not math.isfinite(item) or item < 0.0 for item in row) for row in rows):
            raise MitigationError("disturbance profile erosion values must be finite and nonnegative")
        width = len(rows[0]) if rows else 0
        if any(len(row) != width for row in rows):
            raise MitigationError("disturbance profile matrix must be rectangular")
        object.__setattr__(self, "times_s", times)
        object.__setattr__(self, "upper_erosion", rows)

    def to_dict(self) -> dict[str, Any]:
        return {
            "times_s": list(self.times_s),
            "upper_erosion": [list(row) for row in self.upper_erosion],
        }


@dataclass(frozen=True, slots=True)
class MitigationLPDetails:
    """Detailed LP artifacts plus the public ``MitigationResult``."""

    mitigation_result: MitigationResult
    protected_asset_ids: tuple[str, ...]
    control_ids: tuple[str, ...]
    time_grid_s: tuple[float, ...]
    disturbance_erosion: Matrix
    control_authority: tuple[Matrix, ...]
    unavailable_controls: tuple[str, ...]
    ineffective_controls: tuple[str, ...]
    residual_erosion: Matrix
    metadata: Mapping[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "mitigation_result": self.mitigation_result.to_dict(),
            "protected_asset_ids": list(self.protected_asset_ids),
            "control_ids": list(self.control_ids),
            "time_grid_s": list(self.time_grid_s),
            "disturbance_erosion": [list(row) for row in self.disturbance_erosion],
            "control_authority": [[list(row) for row in item] for item in self.control_authority],
            "unavailable_controls": list(self.unavailable_controls),
            "ineffective_controls": list(self.ineffective_controls),
            "residual_erosion": [list(row) for row in self.residual_erosion],
            "metadata": dict(self.metadata or {}),
        }


def solve_mitigation_lp(
    window_result: WindowMapResult | FiniteWindowMapsResult | RobustScreenResult,
    *,
    seed_ids: Sequence[str] = (),
    seed_assets: Sequence[str] = (),
    disturbance_profile: DisturbanceMarginProfile | None = None,
    disturbance_response: ProfileResponse | None = None,
    disturbance_event_ids: Sequence[str] = (),
    protected_outputs: Sequence[ProtectedOutputEvaluation] | None = None,
    margins_pu: Sequence[float] | None = None,
    base_erosion: Sequence[float] | None = None,
    r_bar: Sequence[float] | None = None,
    alpha_upper: Mapping[str, float] | Sequence[float] | None = None,
    control_costs: Mapping[str, float] | Sequence[float] | None = None,
    control_mvar: Mapping[str, float] | Sequence[float] | None = None,
    unavailable_controls: Sequence[str] = (),
    asset_event_map: Mapping[str, str] | None = None,
    epsilon: float = 0.0,
    slack_penalty: float = 1.0e6,
    selection_tolerance: float = 1.0e-7,
    binding_tolerance: float = 1.0e-7,
    authority_tolerance: float = 1.0e-12,
    metadata: Mapping[str, Any] | None = None,
) -> MitigationLPDetails:
    """Solve the mitigation LP using time-resolved ``H(t_l)``."""

    window, finite_response, robust_r = _unwrap_window(window_result)
    response = disturbance_response or finite_response
    protected = tuple(window.protected_asset_ids)
    controls = tuple(window.control_ids)
    if not controls:
        raise MitigationError("mitigation requires at least one control channel")
    if not window.control_authority_lower:
        raise MitigationError(
            "mitigation constraints require time-resolved control_authority_lower, not scalar rankings"
        )

    epsilon_value = _epsilon(epsilon)
    rho = _positive(slack_penalty, "slack_penalty")
    authority_times, authority = _authority_stack(
        window.control_authority_lower,
        len(protected),
        len(controls),
    )
    r = _base_vector(
        r_bar if r_bar is not None else robust_r if robust_r is not None else base_erosion,
        len(protected),
        "r_bar/base_erosion",
    )
    excluded_assets, seed_event_columns = _excluded_seed_assets(
        protected,
        window.event_ids,
        seed_assets,
        seed_ids,
        asset_event_map,
    )
    disturbance = _disturbance_at_authority_times(
        authority_times=authority_times,
        protected_asset_ids=protected,
        window=window,
        profile=disturbance_profile,
        response=response,
        event_ids=disturbance_event_ids or seed_event_columns,
        margins=_margins(protected, protected_outputs, margins_pu),
    )
    upper = _factor_vector(alpha_upper, controls, len(controls), default=1.0, name="alpha_upper")
    costs = _factor_vector(control_costs, controls, len(controls), default=1.0, name="control_costs")
    mvar = _optional_factor_vector(control_mvar, controls, len(controls), name="control_mvar")
    unavailable = set(str(item) for item in unavailable_controls)
    for index, control_id in enumerate(controls):
        if control_id in unavailable:
            upper[index] = 0.0
    if np.any(upper < 0.0):
        raise MitigationError("alpha_upper must be nonnegative")
    if np.any(costs < 0.0):
        raise MitigationError("control_costs must be nonnegative")

    constrained_rows = [
        index for index, asset in enumerate(protected) if asset not in excluded_assets
    ]
    unavailable_tuple = tuple(control for control, bound in zip(controls, upper) if bound <= selection_tolerance)
    ineffective_tuple = _ineffective_controls(
        controls,
        authority,
        constrained_rows,
        upper,
        tolerance=authority_tolerance,
    )

    c = np.concatenate((costs, [rho]))
    bounds = [(0.0, float(item)) for item in upper] + [(0.0, None)]
    a_rows: list[np.ndarray] = []
    b_rows: list[float] = []
    row_meta: list[tuple[int, int]] = []
    for time_index in range(authority.shape[0]):
        for asset_index in constrained_rows:
            row = np.zeros(len(controls) + 1, dtype=float)
            row[: len(controls)] = -authority[time_index, asset_index, :]
            row[-1] = -1.0
            rhs = 1.0 - epsilon_value - r[asset_index] - disturbance[time_index, asset_index]
            a_rows.append(row)
            b_rows.append(float(rhs))
            row_meta.append((time_index, asset_index))

    if a_rows:
        a_ub = np.vstack(a_rows)
        b_ub = np.asarray(b_rows, dtype=float)
    else:
        a_ub = None
        b_ub = None

    lp = linprog(c, A_ub=a_ub, b_ub=b_ub, bounds=bounds, method="highs")
    if lp.success and lp.x is not None:
        alpha = np.asarray(lp.x[: len(controls)], dtype=float)
        eta = max(float(lp.x[-1]), 0.0)
        objective = float(lp.fun)
        solver_status = f"optimal:{lp.message}"
    else:
        alpha = np.zeros(len(controls), dtype=float)
        eta = math.inf
        objective = 0.0
        solver_status = f"failed:{lp.message}"

    residual = _residual_erosion(r, disturbance, authority, alpha)
    selections = _selections(
        controls,
        alpha,
        upper,
        costs,
        mvar,
        selection_tolerance=selection_tolerance,
    )
    total_mvar = None if mvar is None else float(np.sum(alpha * mvar))
    binding = _binding_constraints(
        protected,
        authority_times,
        residual,
        constrained_rows,
        rhs=1.0 - epsilon_value,
        eta=0.0 if not math.isfinite(eta) else eta,
        tolerance=binding_tolerance,
    )
    mitigation_feasible = bool(lp.success and eta <= binding_tolerance)
    status = AssessmentStatus.CERTIFIED if mitigation_feasible else AssessmentStatus.FAILED
    result = MitigationResult(
        mode_id=window.mode_id,
        seed_ids=tuple(seed_ids),
        feasible=mitigation_feasible,
        objective_value=objective,
        slack_eta=0.0 if not math.isfinite(eta) else eta,
        selections=selections,
        binding_constraints=binding,
        solver_status=solver_status,
        status=status,
        unavailable_controls=unavailable_tuple,
        ineffective_controls=ineffective_tuple,
        required_mvar_total=total_mvar,
        notes=(
            "LP uses time-resolved H(t_l); scalar control ranking R is not used. "
            f"Seed event columns: {', '.join(seed_event_columns) if seed_event_columns else 'none'}."
        ),
        metadata={
            "epsilon": epsilon_value,
            "slack_penalty": rho,
            "constraint_count": len(row_meta),
            **dict(metadata or {}),
        },
    )
    return MitigationLPDetails(
        mitigation_result=result,
        protected_asset_ids=protected,
        control_ids=controls,
        time_grid_s=tuple(float(item) for item in authority_times),
        disturbance_erosion=_matrix_tuple(disturbance),
        control_authority=tuple(_matrix_tuple(authority[index]) for index in range(authority.shape[0])),
        unavailable_controls=unavailable_tuple,
        ineffective_controls=ineffective_tuple,
        residual_erosion=_matrix_tuple(residual),
        metadata={
            "lp_variables": [*controls, "eta"],
            "equation": "eq:mitigation_qp",
            **dict(metadata or {}),
        },
    )


def _unwrap_window(
    source: WindowMapResult | FiniteWindowMapsResult | RobustScreenResult,
) -> tuple[WindowMapResult, ProfileResponse | None, np.ndarray | None]:
    if isinstance(source, FiniteWindowMapsResult):
        return source.window_result, source.disturbance_response, None
    if isinstance(source, RobustScreenResult):
        return source.robust_window_result, None, np.asarray(source.r_bar, dtype=float)
    return source, None, None


def _authority_stack(
    authorities: Sequence[TimeGridControlAuthority],
    n_assets: int,
    n_controls: int,
) -> tuple[np.ndarray, np.ndarray]:
    ordered = sorted(authorities, key=lambda item: item.time_s)
    times = np.asarray([item.time_s for item in ordered], dtype=float)
    if times.size == 0:
        raise MitigationError("control authority time grid is empty")
    matrices = []
    for authority in ordered:
        matrix = np.asarray(authority.lower_bound_matrix, dtype=float)
        if matrix.shape != (n_assets, n_controls):
            raise MitigationError(
                f"control authority at t={authority.time_s} has shape {matrix.shape}, "
                f"expected {(n_assets, n_controls)}"
            )
        if not np.all(np.isfinite(matrix)):
            raise MitigationError("control authority contains non-finite values")
        matrices.append(np.maximum(matrix, 0.0))
    return times, np.asarray(matrices, dtype=float)


def _disturbance_at_authority_times(
    *,
    authority_times: np.ndarray,
    protected_asset_ids: Sequence[str],
    window: WindowMapResult,
    profile: DisturbanceMarginProfile | None,
    response: ProfileResponse | None,
    event_ids: Sequence[str],
    margins: np.ndarray | None,
) -> np.ndarray:
    p = len(protected_asset_ids)
    if profile is not None:
        values = np.asarray(profile.upper_erosion, dtype=float)
        if values.shape[1] != p:
            raise MitigationError("disturbance profile width must match protected assets")
        return _interp_profile(np.asarray(profile.times_s, dtype=float), values, authority_times)
    if response is None:
        return np.zeros((authority_times.size, p), dtype=float)
    if margins is None:
        raise MitigationError("margins are required to normalize disturbance_response")
    response_values = np.asarray(response.values_pu, dtype=float)
    if response_values.ndim != 3 or response_values.shape[1] != p:
        raise MitigationError("disturbance_response shape does not match protected assets")
    response_index = {event: index for index, event in enumerate(response.channel_ids)}
    requested = tuple(str(item) for item in event_ids)
    missing = [event for event in requested if event not in response_index]
    if missing:
        raise MitigationError("disturbance_response missing event IDs: " + ", ".join(missing))
    if not requested:
        return np.zeros((authority_times.size, p), dtype=float)
    columns = [response_index[event] for event in requested]
    combined_pu = np.sum(response_values[:, :, columns], axis=2)
    erosion = np.zeros_like(combined_pu, dtype=float)
    good = margins > 0.0
    if np.any(good):
        erosion[:, good] = np.maximum(combined_pu[:, good], 0.0) / margins[good][None, :]
    return _interp_profile(np.asarray(response.times_s, dtype=float), erosion, authority_times)


def _interp_profile(times: np.ndarray, values: np.ndarray, target_times: np.ndarray) -> np.ndarray:
    if times.ndim != 1 or values.shape[0] != times.size:
        raise MitigationError("profile times must align with profile values")
    if np.any(np.diff(times) < -1e-12):
        raise MitigationError("profile times must be nondecreasing")
    if not np.all(np.isfinite(times)) or not np.all(np.isfinite(values)):
        raise MitigationError("profile contains non-finite values")
    result = np.zeros((target_times.size, values.shape[1]), dtype=float)
    for col in range(values.shape[1]):
        result[:, col] = np.interp(target_times, times, values[:, col], left=values[0, col], right=values[-1, col])
    return np.maximum(result, 0.0)


def _excluded_seed_assets(
    protected_asset_ids: Sequence[str],
    event_ids: Sequence[str],
    seed_assets: Sequence[str],
    seed_ids: Sequence[str],
    asset_event_map: Mapping[str, str] | None,
) -> tuple[set[str], tuple[str, ...]]:
    protected = set(protected_asset_ids)
    events = set(event_ids)
    supplied = {str(k): str(v) for k, v in dict(asset_event_map or {}).items()}
    inferred: dict[str, str] = {}
    for asset in protected_asset_ids:
        if asset in supplied:
            inferred[asset] = supplied[asset]
        elif asset in events:
            inferred[asset] = asset
        elif f"trip_{asset}" in events:
            inferred[asset] = f"trip_{asset}"
    excluded = set()
    seed_event_columns: list[str] = []
    for asset in seed_assets:
        asset_s = str(asset)
        if asset_s not in protected:
            raise MitigationError(f"unknown seed asset {asset_s!r}")
        excluded.add(asset_s)
        if asset_s in inferred:
            seed_event_columns.append(inferred[asset_s])
    inverse = {event: asset for asset, event in inferred.items()}
    for seed in seed_ids:
        seed_s = str(seed)
        if seed_s in protected:
            excluded.add(seed_s)
            if seed_s in inferred:
                seed_event_columns.append(inferred[seed_s])
        elif seed_s in events:
            seed_event_columns.append(seed_s)
            if seed_s in inverse:
                excluded.add(inverse[seed_s])
        else:
            raise MitigationError(f"unknown seed ID {seed_s!r}")
    return excluded, tuple(dict.fromkeys(seed_event_columns))


def _residual_erosion(
    r: np.ndarray,
    disturbance: np.ndarray,
    authority: np.ndarray,
    alpha: np.ndarray,
) -> np.ndarray:
    reduction = np.einsum("tik,k->ti", authority, alpha)
    return r[None, :] + disturbance - reduction


def _binding_constraints(
    protected_asset_ids: Sequence[str],
    times: np.ndarray,
    residual: np.ndarray,
    constrained_rows: Sequence[int],
    *,
    rhs: float,
    eta: float,
    tolerance: float,
) -> tuple[BindingConstraint, ...]:
    constraints: list[BindingConstraint] = []
    for time_index, time in enumerate(times):
        for asset_index in constrained_rows:
            lhs = float(residual[time_index, asset_index])
            slack = max(lhs - rhs, 0.0)
            if lhs >= rhs + eta - abs(tolerance) or slack > abs(tolerance):
                constraints.append(
                    BindingConstraint(
                        protected_asset_ids[asset_index],
                        float(time),
                        lhs,
                        rhs,
                        slack,
                    )
                )
    return tuple(constraints)


def _selections(
    control_ids: Sequence[str],
    alpha: np.ndarray,
    upper: np.ndarray,
    costs: np.ndarray,
    mvar: np.ndarray | None,
    *,
    selection_tolerance: float,
) -> tuple[ControlSelection, ...]:
    selections: list[ControlSelection] = []
    for index, control_id in enumerate(control_ids):
        value = float(alpha[index])
        if value <= selection_tolerance:
            continue
        selections.append(
            ControlSelection(
                control_id,
                value,
                magnitude_mvar=None if mvar is None else float(value * mvar[index]),
                saturated=bool(value >= upper[index] - selection_tolerance),
                cost_contribution=float(costs[index] * value),
            )
        )
    return tuple(selections)


def _ineffective_controls(
    control_ids: Sequence[str],
    authority: np.ndarray,
    constrained_rows: Sequence[int],
    upper: np.ndarray,
    *,
    tolerance: float,
) -> tuple[str, ...]:
    ineffective: list[str] = []
    rows = list(constrained_rows)
    if not rows:
        return ()
    for index, control_id in enumerate(control_ids):
        if upper[index] <= 0.0:
            continue
        max_authority = float(np.max(authority[:, rows, index])) if authority.size else 0.0
        if max_authority <= tolerance:
            ineffective.append(control_id)
    return tuple(ineffective)


def _margins(
    protected_asset_ids: Sequence[str],
    protected_outputs: Sequence[ProtectedOutputEvaluation] | None,
    margins_pu: Sequence[float] | None,
) -> np.ndarray | None:
    if margins_pu is not None:
        margins = np.asarray(margins_pu, dtype=float).reshape(-1)
    elif protected_outputs is not None:
        by_asset = {item.asset_id: item for item in protected_outputs}
        if set(by_asset) != set(protected_asset_ids):
            raise MitigationError("protected_outputs asset IDs must match protected assets")
        margins = np.asarray(
            [by_asset[asset].worst_case_margin_pu for asset in protected_asset_ids], dtype=float
        )
    else:
        return None
    if margins.size != len(protected_asset_ids):
        raise MitigationError(
            f"margins_pu has length {margins.size}, expected {len(protected_asset_ids)}"
        )
    if not np.all(np.isfinite(margins)):
        raise MitigationError("margins must be finite")
    if np.any(margins <= 0.0):
        raise MitigationError("margins must be positive for mitigation profiles")
    return margins


def _base_vector(values: Sequence[float] | np.ndarray | None, count: int, name: str) -> np.ndarray:
    if values is None:
        return np.zeros(count, dtype=float)
    array = np.asarray(values, dtype=float).reshape(-1)
    if array.size != count:
        raise MitigationError(f"{name} has length {array.size}, expected {count}")
    if not np.all(np.isfinite(array)):
        raise MitigationError(f"{name} must be finite")
    if np.any(array < 0.0):
        raise MitigationError(f"{name} must be nonnegative")
    return array


def _factor_vector(
    values: Mapping[str, float] | Sequence[float] | None,
    ids: Sequence[str],
    count: int,
    *,
    default: float,
    name: str,
) -> np.ndarray:
    if values is None:
        return np.full(count, float(default), dtype=float)
    if isinstance(values, Mapping):
        result = np.full(count, float(default), dtype=float)
        index = {item: pos for pos, item in enumerate(ids)}
        for key, value in values.items():
            key_s = str(key)
            if key_s not in index:
                raise MitigationError(f"{name} references unknown control {key_s!r}")
            result[index[key_s]] = float(value)
    else:
        result = np.asarray(values, dtype=float).reshape(-1)
        if result.size != count:
            raise MitigationError(f"{name} has length {result.size}, expected {count}")
    if not np.all(np.isfinite(result)):
        raise MitigationError(f"{name} must be finite")
    return result


def _optional_factor_vector(
    values: Mapping[str, float] | Sequence[float] | None,
    ids: Sequence[str],
    count: int,
    *,
    name: str,
) -> np.ndarray | None:
    if values is None:
        return None
    result = _factor_vector(values, ids, count, default=0.0, name=name)
    if np.any(result < 0.0):
        raise MitigationError(f"{name} must be nonnegative")
    return result


def _epsilon(value: float) -> float:
    epsilon = float(value)
    if not math.isfinite(epsilon) or epsilon < 0.0 or epsilon >= 1.0:
        raise MitigationError("epsilon must be finite and in [0, 1)")
    return epsilon


def _positive(value: float, name: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise MitigationError(f"{name} must be finite and positive")
    return number


def _matrix_tuple(array: np.ndarray) -> Matrix:
    return tuple(tuple(float(item) for item in row) for row in np.asarray(array, dtype=float))
