r"""Finite-window voltage response maps.

This module implements the response equations from ``LaTeX/root.tex``:

* ``step_response_d`` for
  ``F_d d + C_r int_0^t exp(A_r(t-s)) D_r d ds``.
* ``profile_response`` for step, ramp, and sampled profiles using the
  convolution form.
* ``scalar_voltage_response`` for one protected-output/channel waveform.

The propagation uses sparse augmented systems with
``scipy.sparse.linalg.expm_multiply``.  Disturbance excursions are padded by a
sampled Lipschitz bound; if the padding is not small enough after refinement,
the channel is explicitly marked empirical rather than certified.
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
    ProfileKind,
    TimeGridControlAuthority,
    TimeProfile,
    WindowMapResult,
)
from .linearization import ReducedLinearModel
from .protected_outputs import ProtectedOutputEvaluation


class FiniteWindowError(RuntimeError):
    """Raised when finite-window maps cannot be computed consistently."""


@dataclass(frozen=True, slots=True)
class ProfileResponse:
    """Time-domain protected-voltage response for selected channels."""

    times_s: tuple[float, ...]
    values_pu: np.ndarray
    derivatives_pu_per_s: np.ndarray
    channel_ids: tuple[str, ...]
    profile_kinds: tuple[str, ...]
    channel_type: str


@dataclass(frozen=True, slots=True)
class FiniteWindowMapsResult:
    """Detailed finite-window maps plus the normalized screen result."""

    window_result: WindowMapResult
    time_grid_s: tuple[float, ...]
    pickup_excursion_pu: Matrix
    trip_excursion_pu: Matrix
    disturbance_padding_pu: Matrix
    disturbance_response: ProfileResponse | None = None
    control_response: ProfileResponse | None = None
    control_authority_rank: Matrix | None = None
    channel_classes: Mapping[str, str] | None = None
    metadata: Mapping[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "window_result": self.window_result.to_dict(),
            "time_grid_s": list(self.time_grid_s),
            "pickup_excursion_pu": [list(row) for row in self.pickup_excursion_pu],
            "trip_excursion_pu": [list(row) for row in self.trip_excursion_pu],
            "disturbance_padding_pu": [list(row) for row in self.disturbance_padding_pu],
            "control_authority_rank": None
            if self.control_authority_rank is None
            else [list(row) for row in self.control_authority_rank],
            "channel_classes": dict(self.channel_classes or {}),
            "metadata": dict(self.metadata or {}),
        }


def step_response_d(
    reduced: ReducedLinearModel,
    times_s: Sequence[float],
    *,
    disturbance_matrix: Any | None = None,
    event_ids: Sequence[str] | None = None,
) -> ProfileResponse:
    """Compute the disturbance step response in equation ``step_response_d``."""

    q = reduced.fd.shape[1]
    vectors = np.eye(q, dtype=float) if disturbance_matrix is None else _as_2d_array(
        disturbance_matrix, "disturbance_matrix"
    )
    ids = _channel_ids(event_ids, vectors.shape[1], prefix="d")
    profiles = tuple(TimeProfile(ProfileKind.STEP) for _ in ids)
    return profile_response(
        reduced,
        times_s,
        input_matrix=vectors,
        profiles=profiles,
        channel_ids=ids,
        channel_type="disturbance",
    )


def profile_response(
    reduced: ReducedLinearModel,
    times_s: Sequence[float],
    *,
    input_matrix: Any,
    profiles: Sequence[TimeProfile] | None = None,
    channel_ids: Sequence[str] | None = None,
    channel_type: str = "disturbance",
) -> ProfileResponse:
    """Compute equation ``profile_response`` for disturbances or controls."""

    times = _time_array(times_s)
    if channel_type not in {"disturbance", "control"}:
        raise FiniteWindowError("channel_type must be 'disturbance' or 'control'")

    b_mat = reduced.dr if channel_type == "disturbance" else reduced.br
    f_mat = reduced.fd if channel_type == "disturbance" else reduced.fu
    input_dim = f_mat.shape[1]
    vectors = _as_2d_array(input_matrix, "input_matrix")
    if vectors.shape[0] != input_dim:
        raise FiniteWindowError(f"input_matrix has {vectors.shape[0]} rows, expected {input_dim}")
    n_channels = vectors.shape[1]
    ids = _channel_ids(channel_ids, n_channels, prefix="d" if channel_type == "disturbance" else "u")
    profs = _profiles(profiles, n_channels)

    a = reduced.ar.tocsc()
    b = b_mat.tocsc()
    c = reduced.cr.tocsc()
    f = f_mat.tocsc()
    n_states = a.shape[0]
    n_outputs = c.shape[0]

    scalars = _profile_values(profs, times)
    interval_slopes = _profile_interval_slopes(profs, times, scalars)
    u_values = _input_values(vectors, scalars)

    values = np.zeros((times.size, n_outputs, n_channels), dtype=float)
    derivatives = np.zeros_like(values)

    if n_states == 0:
        for idx in range(times.size):
            values[idx] = np.asarray(f @ u_values[idx], dtype=float)
            slope = _input_from_scalars(vectors, _right_slope(interval_slopes, idx, n_channels))
            derivatives[idx] = np.asarray(f @ slope, dtype=float)
        return ProfileResponse(
            times_s=tuple(float(item) for item in times),
            values_pu=values,
            derivatives_pu_per_s=derivatives,
            channel_ids=ids,
            profile_kinds=tuple(str(profile.kind.value) for profile in profs),
            channel_type=channel_type,
        )

    x = np.zeros((n_states, n_channels), dtype=float)
    values[0] = np.asarray(c @ x + f @ u_values[0], dtype=float)
    derivatives[0] = np.asarray(
        c @ (a @ x + b @ u_values[0])
        + f @ _input_from_scalars(vectors, _right_slope(interval_slopes, 0, n_channels)),
        dtype=float,
    )

    for interval in range(times.size - 1):
        dt = float(times[interval + 1] - times[interval])
        if dt <= 0.0:
            raise FiniteWindowError("time grid must be strictly increasing")
        slope = _input_from_scalars(vectors, interval_slopes[interval])
        x = _propagate_piecewise_linear(a, b, x, u_values[interval], slope, dt)
        values[interval + 1] = np.asarray(c @ x + f @ u_values[interval + 1], dtype=float)
        right_slope = _input_from_scalars(
            vectors, _right_slope(interval_slopes, interval + 1, n_channels)
        )
        derivatives[interval + 1] = np.asarray(
            c @ (a @ x + b @ u_values[interval + 1]) + f @ right_slope,
            dtype=float,
        )

    _require_finite(values, "profile response values")
    _require_finite(derivatives, "profile response derivatives")
    return ProfileResponse(
        times_s=tuple(float(item) for item in times),
        values_pu=values,
        derivatives_pu_per_s=derivatives,
        channel_ids=ids,
        profile_kinds=tuple(str(profile.kind.value) for profile in profs),
        channel_type=channel_type,
    )


def scalar_voltage_response(
    response: ProfileResponse,
    *,
    output_index: int,
    channel_index: int,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Return ``p_ij(t) = e_i^T Delta z_j(t)`` for one channel."""

    if output_index < 0 or output_index >= response.values_pu.shape[1]:
        raise FiniteWindowError("output_index is outside response output dimension")
    if channel_index < 0 or channel_index >= response.values_pu.shape[2]:
        raise FiniteWindowError("channel_index is outside response channel dimension")
    values = response.values_pu[:, output_index, channel_index]
    return response.times_s, tuple(float(item) for item in values)


def compute_finite_window_maps(
    reduced: ReducedLinearModel,
    protected_outputs: Sequence[ProtectedOutputEvaluation],
    *,
    event_ids: Sequence[str] | None = None,
    disturbance_matrix: Any | None = None,
    event_profiles: Sequence[TimeProfile] | None = None,
    control_ids: Sequence[str] | None = None,
    control_matrix: Any | None = None,
    control_profiles: Sequence[TimeProfile] | None = None,
    horizons_s: Sequence[float] | None = None,
    dwell_times_s: Sequence[float] | None = None,
    time_grid_s: Sequence[float] | None = None,
    samples_per_window: int = 41,
    max_refinements: int = 4,
    padding_tolerance_pu: float = 1e-4,
    derivative_safety_factor: float = 1.05,
    monotone_tolerance: float = 1e-9,
    strict_margins: bool = False,
    metadata: Mapping[str, Any] | None = None,
) -> FiniteWindowMapsResult:
    """Compute upper-bounded disturbance maps and lower-bounded control maps."""

    margins, data_limited_assets = _margins(protected_outputs, strict=strict_margins)
    data_limited_set = set(data_limited_assets)
    data_limited_rows = np.array(
        [item.asset_id in data_limited_set for item in protected_outputs],
        dtype=bool,
    )
    if samples_per_window < 2:
        raise FiniteWindowError("samples_per_window must be at least 2")
    if max_refinements < 0:
        raise FiniteWindowError("max_refinements must be nonnegative")
    if padding_tolerance_pu < 0.0:
        raise FiniteWindowError("padding_tolerance_pu must be nonnegative")
    if derivative_safety_factor < 0.0:
        raise FiniteWindowError("derivative_safety_factor must be nonnegative")
    p = len(protected_outputs)
    horizons = _horizons(horizons_s, p)
    dwell = _dwell_times(dwell_times_s, p)

    q = reduced.fd.shape[1]
    disturbance_vectors = (
        np.eye(q, dtype=float)
        if disturbance_matrix is None
        else _as_2d_array(disturbance_matrix, "disturbance_matrix")
    )
    if disturbance_vectors.shape[0] != q:
        raise FiniteWindowError(
            f"disturbance_matrix has {disturbance_vectors.shape[0]} rows, expected {q}"
        )
    events = _channel_ids(event_ids, disturbance_vectors.shape[1], prefix="d")
    event_profs = _profiles(event_profiles, len(events))

    r = reduced.fu.shape[1]
    if control_matrix is None:
        control_vectors = np.eye(r, dtype=float)
    else:
        control_vectors = _as_2d_array(control_matrix, "control_matrix")
    if control_vectors.shape[0] != r:
        raise FiniteWindowError(f"control_matrix has {control_vectors.shape[0]} rows, expected {r}")
    controls = _channel_ids(control_ids, control_vectors.shape[1], prefix="u")
    control_profs = _profiles(control_profiles, len(controls))

    grid = _initial_grid(
        horizons,
        samples_per_window=samples_per_window,
        explicit_grid=time_grid_s,
        profiles=tuple(event_profs) + tuple(control_profs),
    )

    disturbance_response: ProfileResponse | None = None
    pickup = np.zeros((p, len(events)), dtype=float)
    trip = np.zeros_like(pickup)
    padding = np.zeros_like(pickup)
    classes = np.full((p, len(events)), "failed", dtype=object)

    for refinement in range(max_refinements + 1):
        disturbance_response = profile_response(
            reduced,
            grid,
            input_matrix=disturbance_vectors,
            profiles=event_profs,
            channel_ids=events,
            channel_type="disturbance",
        )
        pickup, trip, padding, classes = _disturbance_excursions(
            disturbance_response,
            horizons=horizons,
            dwell_times=dwell,
            margins=margins,
            derivative_safety_factor=derivative_safety_factor,
            padding_tolerance_pu=padding_tolerance_pu,
            monotone_tolerance=monotone_tolerance,
            data_limited_rows=data_limited_rows,
        )
        if not _needs_refinement(classes):
            break
        if refinement < max_refinements:
            grid = _refine_grid(grid)

    k_pickup = _normalize(pickup, margins)
    k_trip = _normalize(np.maximum(trip, 0.0), margins)

    control_response: ProfileResponse | None = None
    authorities: tuple[TimeGridControlAuthority, ...] = ()
    control_rank: np.ndarray | None = None
    if len(controls):
        control_response = profile_response(
            reduced,
            grid,
            input_matrix=control_vectors,
            profiles=control_profs,
            channel_ids=controls,
            channel_type="control",
        )
        h_lower = _control_authority(control_response, margins)
        authorities = tuple(
            TimeGridControlAuthority(float(time), _matrix_tuple(h_lower[index]))
            for index, time in enumerate(control_response.times_s)
        )
        control_rank = np.maximum(h_lower, 0.0).max(axis=0)

    assessments = _channel_assessments(
        protected_outputs=protected_outputs,
        event_ids=events,
        classes=classes,
        data_limited_assets=data_limited_assets,
    )
    status = _overall_status(classes, data_limited_assets)
    window = WindowMapResult(
        mode_id=reduced.diagnostics.mode_id,
        protected_asset_ids=tuple(item.asset_id for item in protected_outputs),
        event_ids=events,
        control_ids=controls,
        horizons_s=tuple(float(item) for item in horizons),
        k_pickup_upper=_matrix_tuple(k_pickup),
        k_trip_upper=_matrix_tuple(k_trip),
        control_authority_lower=authorities,
        channel_assessments=assessments,
        data_limited_assets=data_limited_assets,
        status=status,
        metadata={
            "formula": "finite_window_maps",
            "padding_tolerance_pu": padding_tolerance_pu,
            "derivative_safety_factor": derivative_safety_factor,
            **dict(metadata or {}),
        },
    )
    return FiniteWindowMapsResult(
        window_result=window,
        time_grid_s=tuple(float(item) for item in grid),
        pickup_excursion_pu=_matrix_tuple(pickup),
        trip_excursion_pu=_matrix_tuple(trip),
        disturbance_padding_pu=_matrix_tuple(padding),
        disturbance_response=disturbance_response,
        control_response=control_response,
        control_authority_rank=None if control_rank is None else _matrix_tuple(control_rank),
        channel_classes=_class_map(protected_outputs, events, classes),
        metadata={"refined_time_points": len(grid), **dict(metadata or {})},
    )


def _propagate_piecewise_linear(
    a: sparse.csc_matrix,
    b: sparse.csc_matrix,
    x: np.ndarray,
    u: np.ndarray,
    slope: np.ndarray,
    dt: float,
) -> np.ndarray:
    n = a.shape[0]
    q = b.shape[1]
    zero_nq = sparse.csc_matrix((n, q))
    zero_qq = sparse.csc_matrix((q, q))
    eye_q = sparse.eye(q, format="csc")
    block = sparse.bmat(
        (
            (a, b, zero_nq),
            (None, zero_qq, eye_q),
            (None, None, zero_qq),
        ),
        format="csc",
    )
    y0 = np.vstack((x, u, slope))
    propagated = spla.expm_multiply(block * dt, y0)
    return np.asarray(propagated[:n, :], dtype=float)


def _disturbance_excursions(
    response: ProfileResponse,
    *,
    horizons: np.ndarray,
    dwell_times: np.ndarray,
    margins: np.ndarray,
    derivative_safety_factor: float,
    padding_tolerance_pu: float,
    monotone_tolerance: float,
    data_limited_rows: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    times = np.asarray(response.times_s, dtype=float)
    p = response.values_pu.shape[1]
    q = response.values_pu.shape[2]
    pickup = np.zeros((p, q), dtype=float)
    trip = np.zeros((p, q), dtype=float)
    padding = np.zeros((p, q), dtype=float)
    classes = np.empty((p, q), dtype=object)

    for i in range(p):
        mask = times <= horizons[i] + 1e-12
        if not np.any(mask):
            raise FiniteWindowError("assessment grid does not include horizon samples")
        dt_max = _max_spacing(times[mask])
        for j in range(q):
            values = response.values_pu[mask, i, j]
            deriv = response.derivatives_pu_per_s[mask, i, j]
            if margins[i] <= 0.0 or bool(data_limited_rows[i]):
                pickup[i, j] = 0.0
                trip[i, j] = 0.0
                padding[i, j] = 0.0
                classes[i, j] = "failed"
                continue
            if not np.all(np.isfinite(values)) or not np.all(np.isfinite(deriv)):
                classes[i, j] = "failed"
                continue

            monotone = bool(np.all(np.diff(values) >= -monotone_tolerance))
            l_bound = float(np.max(np.abs(deriv))) * float(derivative_safety_factor)
            pad = 0.5 * l_bound * dt_max
            padding[i, j] = pad
            sample_pk = float(np.max(np.maximum(values, 0.0)))
            if monotone:
                pickup[i, j] = sample_pk
                trip[i, j] = _dwell_excursion_sampled(
                    times[mask], values, horizons[i], dwell_times[i], 0.0, sample_pk
                )
                classes[i, j] = "monotone_proxy"
            else:
                pickup[i, j] = sample_pk + pad
                trip[i, j] = _dwell_excursion_sampled(
                    times[mask], values, horizons[i], dwell_times[i], pad, pickup[i, j]
                )
                classes[i, j] = "certified" if pad <= padding_tolerance_pu else "empirical_sampled"
    return pickup, trip, padding, classes


def _dwell_excursion_sampled(
    times: np.ndarray,
    values: np.ndarray,
    horizon: float,
    dwell: float,
    padding: float,
    pickup_upper: float,
) -> float:
    if dwell <= 0.0:
        return pickup_upper
    if horizon + 1e-12 < dwell:
        return 0.0
    best = -math.inf
    latest_start = horizon - dwell
    for start in times[times <= latest_start + 1e-12]:
        mask = (times >= start - 1e-12) & (times <= start + dwell + 1e-12)
        if np.any(mask):
            best = max(best, float(np.min(values[mask])) + padding)
    if not math.isfinite(best):
        return 0.0
    return min(best, pickup_upper)


def _control_authority(response: ProfileResponse, margins: np.ndarray) -> np.ndarray:
    values = response.values_pu
    h = margins.copy()
    good = h > 0.0
    authority = np.zeros_like(values, dtype=float)
    if np.any(good):
        authority[:, good, :] = -values[:, good, :] / h[good][None, :, None]
    return authority


def _initial_grid(
    horizons: np.ndarray,
    *,
    samples_per_window: int,
    explicit_grid: Sequence[float] | None,
    profiles: Sequence[TimeProfile],
) -> np.ndarray:
    max_horizon = float(np.max(horizons)) if horizons.size else 0.0
    if explicit_grid is not None:
        base = _time_array(explicit_grid)
    else:
        count = max(int(samples_per_window), 2)
        base = np.linspace(0.0, max_horizon, count)
    points = set(float(item) for item in base if 0.0 <= float(item) <= max_horizon + 1e-12)
    points.add(0.0)
    points.add(max_horizon)
    for horizon in horizons:
        points.add(float(horizon))
    for profile in profiles:
        for point in _profile_breakpoints(profile):
            if 0.0 <= point <= max_horizon + 1e-12:
                points.add(float(point))
    return np.array(sorted(points), dtype=float)


def _refine_grid(grid: np.ndarray) -> np.ndarray:
    mids = 0.5 * (grid[:-1] + grid[1:])
    return np.array(sorted(set(float(item) for item in np.concatenate((grid, mids)))), dtype=float)


def _profile_breakpoints(profile: TimeProfile) -> tuple[float, ...]:
    kind = ProfileKind(profile.kind)
    points = [float(profile.start_s)]
    if kind == ProfileKind.RAMP and profile.duration_s is not None:
        points.append(float(profile.start_s + profile.duration_s))
    if kind in {ProfileKind.SAMPLED, ProfileKind.PIECEWISE_CONSTANT}:
        points.extend(float(t) for t, _ in profile.samples)
    return tuple(points)


def _profile_values(profiles: Sequence[TimeProfile], times: np.ndarray) -> np.ndarray:
    values = np.zeros((len(profiles), times.size), dtype=float)
    for index, profile in enumerate(profiles):
        values[index, :] = [_profile_value(profile, float(time)) for time in times]
    return values


def _profile_interval_slopes(
    profiles: Sequence[TimeProfile],
    times: np.ndarray,
    values: np.ndarray,
) -> np.ndarray:
    if times.size < 2:
        return np.zeros((0, len(profiles)), dtype=float)
    slopes = np.zeros((times.size - 1, len(profiles)), dtype=float)
    for interval in range(times.size - 1):
        dt = float(times[interval + 1] - times[interval])
        if dt <= 0.0:
            raise FiniteWindowError("time grid must be strictly increasing")
        for channel, profile in enumerate(profiles):
            kind = ProfileKind(profile.kind)
            if kind in {ProfileKind.STEP, ProfileKind.PIECEWISE_CONSTANT}:
                slopes[interval, channel] = 0.0
            else:
                slopes[interval, channel] = (values[channel, interval + 1] - values[channel, interval]) / dt
    return slopes


def _profile_value(profile: TimeProfile, time_s: float) -> float:
    kind = ProfileKind(profile.kind)
    start = float(profile.start_s)
    final = 1.0 if profile.final_value is None else float(profile.final_value)
    if kind == ProfileKind.STEP:
        return final if time_s >= start - 1e-12 else 0.0
    if kind == ProfileKind.RAMP:
        if time_s < start:
            return 0.0
        duration = 0.0 if profile.duration_s is None else float(profile.duration_s)
        if duration <= 0.0:
            return final
        return final * min(max((time_s - start) / duration, 0.0), 1.0)
    if kind == ProfileKind.SAMPLED:
        return _sampled_value(profile.samples, time_s, linear=True)
    if kind == ProfileKind.PIECEWISE_CONSTANT:
        return _sampled_value(profile.samples, time_s, linear=False)
    raise FiniteWindowError(f"unsupported profile kind {kind.value!r}")


def _sampled_value(samples: Sequence[tuple[float, float]], time_s: float, *, linear: bool) -> float:
    if not samples:
        return 0.0
    if time_s <= samples[0][0]:
        return float(samples[0][1])
    for (t0, v0), (t1, v1) in zip(samples[:-1], samples[1:]):
        if time_s <= t1 + 1e-12:
            if not linear:
                return float(v0)
            alpha = (time_s - t0) / (t1 - t0)
            return float(v0 + alpha * (v1 - v0))
    return float(samples[-1][1])


def _input_values(vectors: np.ndarray, scalars: np.ndarray) -> list[np.ndarray]:
    return [_input_from_scalars(vectors, scalars[:, idx]) for idx in range(scalars.shape[1])]


def _input_from_scalars(vectors: np.ndarray, scalars: np.ndarray) -> np.ndarray:
    return vectors * np.asarray(scalars, dtype=float).reshape(1, -1)


def _right_slope(interval_slopes: np.ndarray, index: int, n_channels: int) -> np.ndarray:
    if interval_slopes.size == 0:
        return np.zeros(n_channels, dtype=float)
    if index >= interval_slopes.shape[0]:
        return interval_slopes[-1, :]
    return interval_slopes[index, :]


def _margins(
    protected_outputs: Sequence[ProtectedOutputEvaluation],
    *,
    strict: bool,
) -> tuple[np.ndarray, tuple[str, ...]]:
    margins = np.asarray([item.worst_case_margin_pu for item in protected_outputs], dtype=float)
    _require_finite(margins, "margins")
    bad = tuple(
        item.asset_id
        for item, margin in zip(protected_outputs, margins)
        if (not item.is_certifiable) or margin <= 0.0
    )
    if bad and strict:
        raise FiniteWindowError(f"positive certified margins are required: {', '.join(bad)}")
    return margins, bad


def _horizons(values: Sequence[float] | None, count: int) -> np.ndarray:
    horizons = np.ones(count, dtype=float) if values is None else np.asarray(values, dtype=float).reshape(-1)
    if horizons.size != count:
        raise FiniteWindowError(f"horizons_s has length {horizons.size}, expected {count}")
    _require_finite(horizons, "horizons_s")
    if np.any(horizons < 0.0):
        raise FiniteWindowError("horizons_s must be nonnegative")
    return horizons


def _dwell_times(values: Sequence[float] | None, count: int) -> np.ndarray:
    dwell = np.zeros(count, dtype=float) if values is None else np.asarray(values, dtype=float).reshape(-1)
    if dwell.size != count:
        raise FiniteWindowError(f"dwell_times_s has length {dwell.size}, expected {count}")
    _require_finite(dwell, "dwell_times_s")
    if np.any(dwell < 0.0):
        raise FiniteWindowError("dwell_times_s must be nonnegative")
    return dwell


def _profiles(profiles: Sequence[TimeProfile] | None, count: int) -> tuple[TimeProfile, ...]:
    if profiles is None:
        return tuple(TimeProfile(ProfileKind.STEP) for _ in range(count))
    result = tuple(profiles)
    if len(result) != count:
        raise FiniteWindowError(f"profiles has length {len(result)}, expected {count}")
    return result


def _channel_ids(ids: Sequence[str] | None, count: int, *, prefix: str) -> tuple[str, ...]:
    result = tuple(str(item) for item in ids) if ids is not None else tuple(
        f"{prefix}_{index}" for index in range(count)
    )
    if len(result) != count:
        raise FiniteWindowError(f"channel IDs has length {len(result)}, expected {count}")
    if any(not item.strip() for item in result):
        raise FiniteWindowError("channel IDs must be non-empty")
    if len(set(result)) != len(result):
        raise FiniteWindowError("channel IDs must be unique")
    return result


def _time_array(times_s: Sequence[float]) -> np.ndarray:
    times = np.asarray(times_s, dtype=float).reshape(-1)
    if times.size == 0:
        raise FiniteWindowError("time grid must be nonempty")
    _require_finite(times, "time grid")
    if np.any(times < 0.0):
        raise FiniteWindowError("time grid must be nonnegative")
    if np.any(np.diff(times) <= 0.0):
        raise FiniteWindowError("time grid must be strictly increasing")
    if abs(float(times[0])) > 1e-12:
        raise FiniteWindowError("time grid must start at zero")
    return times


def _max_spacing(times: np.ndarray) -> float:
    if times.size < 2:
        return 0.0
    return float(np.max(np.diff(times)))


def _normalize(excursion: np.ndarray, margins: np.ndarray) -> np.ndarray:
    k = np.zeros_like(excursion, dtype=float)
    good = margins > 0.0
    if np.any(good):
        k[good, :] = np.maximum(excursion[good, :], 0.0) / margins[good, None]
    return k


def _needs_refinement(classes: np.ndarray) -> bool:
    return bool(np.any(classes == "empirical_sampled"))


def _channel_assessments(
    *,
    protected_outputs: Sequence[ProtectedOutputEvaluation],
    event_ids: Sequence[str],
    classes: np.ndarray,
    data_limited_assets: Sequence[str],
) -> tuple[ChannelAssessment, ...]:
    bad_assets = set(data_limited_assets)
    assessments: list[ChannelAssessment] = []
    for i, output in enumerate(protected_outputs):
        for j, event_id in enumerate(event_ids):
            cls = str(classes[i, j])
            if output.asset_id in bad_assets:
                status = AssessmentStatus.DATA_LIMITED
                notes = "nonpositive_or_uncertified_margin"
            elif cls in {"certified", "monotone_proxy"}:
                status = AssessmentStatus.CERTIFIED
                notes = None
            elif cls == "empirical_sampled":
                status = AssessmentStatus.EMPIRICAL
                notes = "lipschitz_padding_tolerance_not_met"
            else:
                status = AssessmentStatus.FAILED
                notes = "finite_window_channel_failed"
            assessments.append(
                ChannelAssessment(
                    protected_asset_id=output.asset_id,
                    channel_id=event_id,
                    status=status,
                    bound_type=cls,
                    notes=notes,
                )
            )
    return tuple(assessments)


def _overall_status(classes: np.ndarray, data_limited_assets: Sequence[str]) -> AssessmentStatus:
    if data_limited_assets:
        return AssessmentStatus.DATA_LIMITED
    if np.any(classes == "failed"):
        return AssessmentStatus.FAILED
    if np.any(classes == "empirical_sampled"):
        return AssessmentStatus.EMPIRICAL
    return AssessmentStatus.CERTIFIED


def _class_map(
    protected_outputs: Sequence[ProtectedOutputEvaluation],
    event_ids: Sequence[str],
    classes: np.ndarray,
) -> dict[str, str]:
    return {
        f"{output.asset_id}:{event_id}": str(classes[i, j])
        for i, output in enumerate(protected_outputs)
        for j, event_id in enumerate(event_ids)
    }


def _as_2d_array(value: Any, name: str) -> np.ndarray:
    array = value.toarray() if sparse.issparse(value) else np.asarray(value, dtype=float)
    if array.ndim != 2:
        raise FiniteWindowError(f"{name} must be two-dimensional")
    _require_finite(array, name)
    return np.asarray(array, dtype=float)


def _require_finite(array: np.ndarray, name: str) -> None:
    if not np.all(np.isfinite(array)):
        raise FiniteWindowError(f"{name} contains non-finite values")


def _matrix_tuple(array: np.ndarray) -> Matrix:
    return tuple(tuple(float(value) for value in row) for row in np.asarray(array, dtype=float))
