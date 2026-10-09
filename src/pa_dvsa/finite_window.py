"""Window responses of the linear model and the upper bound on their maximum: exact piecewise-linear propagation,
logarithmic-norm interval bounds, and the pickup and trip excursions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
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
    pass


@dataclass(frozen=True, slots=True)
class ProfileResponse:
    times_s: tuple[float, ...]
    values_pu: np.ndarray
    derivatives_pu_per_s: np.ndarray
    channel_ids: tuple[str, ...]
    profile_kinds: tuple[str, ...]
    channel_type: str


@dataclass(frozen=True, slots=True)
class FiniteWindowMapsResult:
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
    bound_method: str = "rigorous",
    min_interval_s: float = 2.5e-4,
) -> FiniteWindowMapsResult:
    if bound_method not in {"rigorous", "empirical"}:
        raise FiniteWindowError("bound_method must be 'rigorous' or 'empirical'")

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

    if bound_method == "rigorous":
        disturbance_response, pickup, trip, padding, classes = _rigorous_disturbance_maps(
            reduced,
            grid,
            vectors=disturbance_vectors,
            profiles=event_profs,
            event_ids=events,
            horizons=horizons,
            dwell_times=dwell,
            margins=margins,
            data_limited_rows=data_limited_rows,
            tolerance=padding_tolerance_pu,
            min_interval_s=min_interval_s,
        )
        max_refinements = -1

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
            "bound_method": bound_method,
            "padding_tolerance_pu": padding_tolerance_pu,
            "derivative_safety_factor": derivative_safety_factor if bound_method == "empirical" else None,
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


def _rigorous_disturbance_maps(
    reduced: ReducedLinearModel,
    grid: np.ndarray,
    *,
    vectors: np.ndarray,
    profiles: Sequence[TimeProfile],
    event_ids: Sequence[str],
    horizons: np.ndarray,
    dwell_times: np.ndarray,
    margins: np.ndarray,
    data_limited_rows: np.ndarray,
    tolerance: float,
    min_interval_s: float,
) -> tuple[ProfileResponse, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mode = ModeDynamics(reduced.ar, dt_atomic=min_interval_s)
    c = reduced.cr.toarray()
    p = c.shape[0]
    q = vectors.shape[1]
    pickup = np.zeros((p, q))
    trip = np.zeros((p, q))
    padding = np.zeros((p, q))
    classes = np.full((p, q), "failed", dtype=object)
    values = np.zeros((grid.size, p, q))
    derivs = np.zeros_like(values)
    dr = reduced.dr.toarray() if reduced.dr.shape[1] else np.zeros((mode.n, 0))
    fd = reduced.fd.toarray() if reduced.fd.shape[1] else np.zeros((p, 0))
    for j in range(q):
        b = dr @ vectors[:, j] if mode.n else np.zeros(0)
        f = fd @ vectors[:, j]
        resp = propagate_channel(mode, c, b, f, profiles[j], grid, channel_id=str(event_ids[j]))
        pos = np.searchsorted(resp.times_s, grid - 1e-12)
        values[:, :, j] = resp.phi_right[pos, :]
        derivs[:, :, j] = resp.cxdot[pos, :] + np.outer(
            np.concatenate((resp.sigma, [0.0]))[pos], resp.f
        )
        for i in range(p):
            if margins[i] <= 0.0 or bool(data_limited_rows[i]):
                classes[i, j] = "failed"
                continue
            env = rigorous_envelope(
                resp,
                i,
                grid,
                horizon_s=float(horizons[i]),
                tolerance_pu=tolerance,
                min_interval_s=min_interval_s,
            )
            pickup[i, j] = max(env.peak_upper, 0.0)
            m_tr = trip_excursion_upper(env, horizon_s=float(horizons[i]), dwell_s=float(dwell_times[i]))
            trip[i, j] = min(m_tr, pickup[i, j]) if math.isfinite(m_tr) else 0.0
            padding[i, j] = env.peak_upper - env.sampled_max
            classes[i, j] = "certified" if math.isfinite(env.peak_upper) else "failed"
        resp.release()
    response = ProfileResponse(
        times_s=tuple(float(t) for t in grid),
        values_pu=values,
        derivatives_pu_per_s=derivs,
        channel_ids=tuple(str(e) for e in event_ids),
        profile_kinds=tuple(str(ProfileKind(pr.kind).value) for pr in profiles),
        channel_type="disturbance",
    )
    return response, pickup, trip, padding, classes


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
                classes[i, j] = "empirical_monotone"
            else:
                pickup[i, j] = sample_pk + pad
                trip[i, j] = _dwell_excursion_sampled(
                    times[mask], values, horizons[i], dwell_times[i], pad, pickup[i, j]
                )
                classes[i, j] = (
                    "empirical_sampled" if pad <= padding_tolerance_pu else "empirical_sampled_unrefined"
                )
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
    return bool(np.any(classes == "empirical_sampled_unrefined"))


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
            elif cls == "certified":
                status = AssessmentStatus.CERTIFIED
                notes = "rigorous_lognorm_bound"
            elif cls.startswith("empirical"):
                status = AssessmentStatus.EMPIRICAL
                notes = "empirical_heuristic_padding_not_a_certificate"
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
    if any(str(c).startswith("empirical") for c in classes.ravel()):
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


import scipy.linalg as _sla

_ATOMIC_ROUND = 1e-12


class ModeDynamics:
    def __init__(
        self,
        a: Any,
        *,
        dt_atomic: float = 2.5e-4,
        need_ramp_blocks: bool = True,
        balance: bool = True,
    ) -> None:
        a_dense = a.toarray() if sparse.issparse(a) else np.asarray(a, dtype=float)
        if a_dense.ndim != 2 or a_dense.shape[0] != a_dense.shape[1]:
            raise FiniteWindowError("A must be square")
        _require_finite(a_dense, "A")
        self.a = np.ascontiguousarray(a_dense, dtype=float)
        self.n = self.a.shape[0]
        self.dt_atomic = float(dt_atomic)
        if not self.dt_atomic > 0.0:
            raise FiniteWindowError("dt_atomic must be positive")
        self.need_ramp_blocks = bool(need_ramp_blocks)
        if self.n and balance:
            _, (scale, _) = _sla.matrix_balance(self.a, permute=False, separate=True)
            scale = np.asarray(scale, dtype=float)
            if not np.all(np.isfinite(scale)) or np.any(scale <= 0.0):
                scale = np.ones(self.n)
        else:
            scale = np.ones(self.n)
        self.scale = scale
        if self.n:
            ahat = self.a * (1.0 / scale)[:, None] * scale[None, :]
            self.mu = float(np.linalg.eigvalsh(0.5 * (ahat + ahat.T)).max())
        else:
            self.mu = 0.0
        self._atomic_cache: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray | None]] = {}
        self._dt_cache: dict[float, tuple[np.ndarray, np.ndarray, np.ndarray | None]] = {}

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
        if self.n == 0:
            return np.zeros(c.shape[0])
        return np.linalg.norm(c * self.scale[None, :], axis=1)

    def blocks(self, dt: float) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        dt = float(dt)
        if dt <= 0.0:
            raise FiniteWindowError("propagation step must be positive")
        k = dt / self.dt_atomic
        k_int = int(round(k))
        if k_int >= 1 and abs(k - k_int) <= 1e-7 * max(k, 1.0):
            return self._atomic_multiple(k_int)
        key = round(dt, 12)
        if key not in self._dt_cache:
            self._dt_cache[key] = self._scaling_squaring(dt)
        return self._dt_cache[key]

    def _atomic_multiple(self, k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        if k in self._atomic_cache:
            return self._atomic_cache[k]
        if k == 1:
            self._atomic_cache[1] = self._scaling_squaring(self.dt_atomic)
            return self._atomic_cache[1]
        half = k // 2
        eh, g1h, g2h = self._atomic_multiple(half)
        e2, g12, g22 = self._compose((eh, g1h, g2h), half * self.dt_atomic, (eh, g1h, g2h))
        if k % 2:
            e1, g11, g21 = self._atomic_multiple(1)
            e2, g12, g22 = self._compose((e2, g12, g22), 2 * half * self.dt_atomic, (e1, g11, g21))
        self._atomic_cache[k] = (e2, g12, g22)
        return self._atomic_cache[k]

    def _compose(
        self,
        first: tuple[np.ndarray, np.ndarray, np.ndarray | None],
        first_dt: float,
        second: tuple[np.ndarray, np.ndarray, np.ndarray | None],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        ea, g1a, g2a = first
        eb, g1b, g2b = second
        e = eb @ ea
        g1 = eb @ g1a + g1b
        g2 = None
        if self.need_ramp_blocks and g2a is not None and g2b is not None:
            g2 = eb @ g2a + first_dt * g1b + g2b
        return e, g1, g2

    def _scaling_squaring(self, dt: float) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        n = self.n
        eye = np.eye(n)
        if n == 0:
            z = np.zeros((0, 0))
            return z, z, (z if self.need_ramp_blocks else None)
        norm = float(np.abs(self.a).sum(axis=0).max()) * dt
        squarings = max(0, int(math.ceil(math.log2(norm / 0.5)))) if norm > 0.5 else 0
        h = dt / (2.0**squarings)
        ah = self.a * h
        e = eye.copy()
        g1 = h * eye.copy()
        g2 = (0.5 * h * h) * eye.copy() if self.need_ramp_blocks else None
        power = eye.copy()
        fact = 1.0
        for k in range(1, 30):
            power = power @ ah
            fact *= k
            e += power / fact
            g1 += (h / (fact * (k + 1))) * power
            if g2 is not None:
                g2 += (h * h / (fact * (k + 1) * (k + 2))) * power
            if np.abs(power).max() / fact < 1e-18:
                break
        step = h
        for _ in range(squarings):
            e_new = e @ e
            g1_new = e @ g1 + g1
            if g2 is not None:
                g2 = e @ g2 + step * g1 + g2
            e, g1 = e_new, g1_new
            step *= 2.0
        return e, g1, g2


def profile_breakpoints(profile: TimeProfile, horizon_s: float) -> tuple[float, ...]:
    pts = [p for p in _profile_breakpoints(profile) if 0.0 < p <= horizon_s + 1e-12]
    return tuple(sorted(set(round(p, 12) for p in pts)))


def channel_grid(base_grid: Sequence[float], profile: TimeProfile) -> np.ndarray:
    base = _time_array(base_grid)
    extra = profile_breakpoints(profile, float(base[-1]))
    merged = np.unique(np.concatenate((np.round(base, 12), np.asarray(extra, dtype=float))))
    return merged


def piecewise_linear_input(profile: TimeProfile, grid: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    times = np.asarray(grid, dtype=float)
    right = profile_values_vectorized(profile, times)
    sigma = np.zeros(max(times.size - 1, 0), dtype=float)
    left = np.zeros_like(right)
    if times.size > 1:
        dt = np.diff(times)
        mid = profile_values_vectorized(profile, times[:-1] + 0.5 * dt)
        sigma = (mid - right[:-1]) / (0.5 * dt)
        left[1:] = right[:-1] + sigma * dt
    left[0] = 0.0
    kind = ProfileKind(profile.kind)
    if kind in {ProfileKind.STEP, ProfileKind.PIECEWISE_CONSTANT}:
        sigma[:] = 0.0
        left[1:] = right[:-1]
        left[0] = 0.0
    return left, right, sigma


def profile_values_vectorized(profile: TimeProfile, times: np.ndarray) -> np.ndarray:
    t = np.asarray(times, dtype=float)
    kind = ProfileKind(profile.kind)
    start = float(profile.start_s)
    final = 1.0 if profile.final_value is None else float(profile.final_value)
    if kind == ProfileKind.STEP:
        return np.where(t >= start - 1e-12, final, 0.0)
    if kind == ProfileKind.RAMP:
        dur = float(profile.duration_s or 0.0)
        if dur <= 0.0:
            return np.where(t < start, 0.0, final)
        return np.where(t < start, 0.0, final * np.clip((t - start) / dur, 0.0, 1.0))
    if kind in {ProfileKind.SAMPLED, ProfileKind.PIECEWISE_CONSTANT}:
        if not profile.samples:
            return np.zeros_like(t)
        ts = np.array([sample[0] for sample in profile.samples], dtype=float)
        vs = np.array([sample[1] for sample in profile.samples], dtype=float)
        if kind == ProfileKind.SAMPLED:
            return np.interp(t, ts, vs, left=vs[0], right=vs[-1])
        k = np.searchsorted(ts, t, side="right") - 1
        return vs[np.maximum(k, 0)]
    raise FiniteWindowError(f"unsupported profile kind {kind.value!r}")


@dataclass
class ChannelResponse:
    channel_id: str
    times_s: np.ndarray
    u_left: np.ndarray
    u_right: np.ndarray
    sigma: np.ndarray
    phi_left: np.ndarray
    phi_right: np.ndarray
    cxdot: np.ndarray
    xdot_norm: np.ndarray
    xddot_norm: np.ndarray
    c_norms: np.ndarray
    f: np.ndarray
    b_norm: float
    mu: float
    states: np.ndarray | None
    mode: ModeDynamics | None = None
    c: np.ndarray | None = None
    b: np.ndarray | None = None
    refine_cache: dict = field(default_factory=dict, repr=False)

    @property
    def n_outputs(self) -> int:
        return int(self.phi_right.shape[1])

    def derivative_bounds(self, output: int) -> tuple[np.ndarray, np.ndarray]:
        dt = np.diff(self.times_s)
        if dt.size == 0:
            return np.zeros(0), np.zeros(0)
        growth = np.exp(max(self.mu, 0.0) * dt)
        cn = float(self.c_norms[output])
        sig = np.abs(self.sigma)
        feed = abs(float(self.f[output])) * sig
        l1 = cn * growth * (self.xdot_norm[:-1] + sig * dt * self.b_norm) + feed
        m = cn * growth * self.xddot_norm[:-1]
        l2 = np.abs(self.cxdot[:-1, output]) + dt * m + feed
        return np.minimum(l1, l2), m

    def interval_bounds(self, output: int) -> tuple[np.ndarray, np.ndarray]:
        dt = np.diff(self.times_s)
        a = self.phi_right[:-1, output]
        b = self.phi_left[1:, output]
        lb, mb = self.derivative_bounds(output)
        upper = np.minimum(0.5 * (a + b + lb * dt), np.maximum(a, b) + mb * dt * dt / 8.0)
        lower = np.maximum(0.5 * (a + b - lb * dt), np.minimum(a, b) - mb * dt * dt / 8.0)
        return np.maximum(upper, np.maximum(a, b)), np.minimum(lower, np.minimum(a, b))

    def refine(self, interval: int, parts: int) -> "ChannelResponse":
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
        sigma = np.full(int(parts), sig)
        x0 = self.states[interval]
        seg = _propagate(
            self.mode,
            self.c,
            self.b,
            self.f,
            sub,
            u_left,
            u_right,
            sigma,
            x0=x0,
            channel_id=f"{self.channel_id}[{interval}]",
            keep_states=True,
            c_norms=self.c_norms,
            b_norm=self.b_norm,
        )
        self.refine_cache[key] = seg
        return seg

    def release(self) -> None:
        self.states = None
        self.refine_cache = {}


def propagate_channel(
    mode: ModeDynamics,
    c: Any,
    b: Any,
    f: Any,
    profile: TimeProfile,
    base_grid: Sequence[float],
    *,
    channel_id: str = "channel",
    keep_states: bool = True,
) -> ChannelResponse:
    grid = channel_grid(base_grid, profile)
    u_left, u_right, sigma = piecewise_linear_input(profile, grid)
    c_arr = np.atleast_2d(np.asarray(c.toarray() if sparse.issparse(c) else c, dtype=float))
    b_arr = np.asarray(b.toarray() if sparse.issparse(b) else b, dtype=float).reshape(-1)
    f_arr = np.asarray(f.toarray() if sparse.issparse(f) else f, dtype=float).reshape(-1)
    if mode.n == 0 and c_arr.size == 0:
        c_arr = np.zeros((f_arr.size, 0))
    if c_arr.shape[1] != mode.n:
        raise FiniteWindowError(f"C has {c_arr.shape[1]} columns, expected {mode.n}")
    if b_arr.size != mode.n:
        raise FiniteWindowError(f"b has length {b_arr.size}, expected {mode.n}")
    if f_arr.size != c_arr.shape[0]:
        raise FiniteWindowError("f length must equal the number of outputs")
    return _propagate(
        mode,
        c_arr,
        b_arr,
        f_arr,
        grid,
        u_left,
        u_right,
        sigma,
        x0=np.zeros(mode.n),
        channel_id=channel_id,
        keep_states=keep_states,
    )


def _propagate(
    mode: ModeDynamics,
    c: np.ndarray,
    b: np.ndarray,
    f: np.ndarray,
    grid: np.ndarray,
    u_left: np.ndarray,
    u_right: np.ndarray,
    sigma: np.ndarray,
    *,
    x0: np.ndarray,
    channel_id: str,
    keep_states: bool,
    c_norms: np.ndarray | None = None,
    b_norm: float | None = None,
) -> ChannelResponse:
    n = mode.n
    npts = grid.size
    p = c.shape[0]
    states = np.zeros((npts, n)) if keep_states else None
    phi_l = np.zeros((npts, p))
    phi_r = np.zeros((npts, p))
    cxd = np.zeros((npts, p))
    nxd = np.zeros(npts)
    nxdd = np.zeros(npts)
    x = np.asarray(x0, dtype=float).copy()
    a = mode.a
    ramp = bool(np.any(sigma != 0.0))
    if ramp and not mode.need_ramp_blocks:
        raise FiniteWindowError("mode was built without ramp blocks but the input has slopes")
    gb_cache: dict[float, tuple[np.ndarray, np.ndarray, np.ndarray | None]] = {}
    for l in range(npts):
        if l > 0:
            dt = float(grid[l] - grid[l - 1])
            key = round(dt, 13)
            if key not in gb_cache:
                e, g1, g2 = mode.blocks(dt)
                gb_cache[key] = (e, g1 @ b, (g2 @ b) if g2 is not None else None)
            e, g1b, g2b = gb_cache[key]
            x = e @ x + g1b * u_right[l - 1]
            if sigma[l - 1] != 0.0:
                x = x + g2b * sigma[l - 1]
        if states is not None:
            states[l] = x
        cx = c @ x if n else np.zeros(p)
        phi_l[l] = cx + f * u_left[l]
        phi_r[l] = cx + f * u_right[l]
        xd = a @ x + b * u_right[l] if n else np.zeros(0)
        slope = sigma[l] if l < sigma.size else 0.0
        xdd = a @ xd + b * slope if n else np.zeros(0)
        cxd[l] = c @ xd if n else 0.0
        nxd[l] = float(mode.scaled_norm(xd)) if n else 0.0
        nxdd[l] = float(mode.scaled_norm(xdd)) if n else 0.0
    _require_finite(phi_r, f"{channel_id} response")
    return ChannelResponse(
        channel_id=channel_id,
        times_s=np.asarray(grid, dtype=float),
        u_left=np.asarray(u_left, dtype=float),
        u_right=np.asarray(u_right, dtype=float),
        sigma=np.asarray(sigma, dtype=float),
        phi_left=phi_l,
        phi_right=phi_r,
        cxdot=cxd,
        xdot_norm=nxd,
        xddot_norm=nxdd,
        c_norms=mode.output_scaled_norms(c) if c_norms is None else c_norms,
        f=f,
        b_norm=float(mode.scaled_norm(b)) if b_norm is None else float(b_norm),
        mu=mode.mu,
        states=states,
        mode=mode,
        c=c,
        b=b,
    )


@dataclass(frozen=True, slots=True)
class RigorousEnvelope:
    base_times_s: np.ndarray
    values: np.ndarray
    upper: np.ndarray
    lower: np.ndarray
    peak_upper: float
    trough_lower: float
    sampled_max: float
    sampled_min: float
    refined_intervals: int
    min_interval_s: float
    fine_times_s: np.ndarray
    fine_values: np.ndarray
    fine_left_values: np.ndarray
    fine_upper: np.ndarray | None = None
    fine_lower: np.ndarray | None = None

    @property
    def peak_padding(self) -> float:
        return float(self.peak_upper - self.sampled_max)

    @property
    def trough_padding(self) -> float:
        return float(self.sampled_min - self.trough_lower)


def _divisor_near(k: int, target: int = 4) -> int:
    if k <= target:
        return max(k, 2)
    divs = [d for d in range(2, k + 1) if k % d == 0]
    return min(divs, key=lambda d: (abs(d - target), d))


def _interval_extrema(
    resp: ChannelResponse,
    output: int,
    *,
    tol: float,
    min_dt: float,
    upper: bool,
    budget: list[int],
) -> tuple[np.ndarray, float, int]:
    up, lo = resp.interval_bounds(output)
    bnd = up if upper else lo
    a = resp.phi_right[:-1, output]
    b = resp.phi_left[1:, output]
    samp = np.maximum(a, b) if upper else np.minimum(a, b)
    refined = 0
    dt = np.diff(resp.times_s)
    for l in range(bnd.size):
        pad = (bnd[l] - samp[l]) if upper else (samp[l] - bnd[l])
        if pad <= tol or dt[l] <= min_dt * (1.0 + 1e-9) or budget[0] <= 0:
            continue
        mode = resp.mode
        k = dt[l] / mode.dt_atomic if mode is not None else 0.0
        if mode is not None and abs(k - round(k)) <= 1e-7 * max(k, 1.0) and round(k) >= 2:
            parts = _divisor_near(int(round(k)))
        else:
            parts = 2
            if dt[l] / 2.0 < min_dt * (1.0 - 1e-9):
                continue
        budget[0] -= 1
        seg = resp.refine(l, parts)
        seg_b, seg_s, seg_r = _interval_extrema(seg, output, tol=tol, min_dt=min_dt, upper=upper, budget=budget)
        refined += 1 + seg_r
        pts = np.concatenate((seg.phi_right[:-1, output], seg.phi_left[1:, output]))
        if upper:
            bnd[l] = max(float(np.max(seg_b)), float(np.max(pts)))
            samp[l] = max(samp[l], float(np.max(pts)))
        else:
            bnd[l] = min(float(np.min(seg_b)), float(np.min(pts)))
            samp[l] = min(samp[l], float(np.min(pts)))
    best = float(np.max(samp)) if upper else float(np.min(samp))
    return bnd, best if bnd.size else 0.0, refined


def rigorous_envelope(
    resp: ChannelResponse,
    output: int,
    base_grid: Sequence[float],
    *,
    horizon_s: float | None = None,
    tolerance_pu: float = 1e-4,
    min_interval_s: float = 2.5e-4,
    max_refinements: int = 2000,
    refine_lower: bool = False,
) -> RigorousEnvelope:
    base = np.asarray(base_grid, dtype=float)
    horizon = float(base[-1] if horizon_s is None else horizon_s)
    budget = [int(max_refinements)]
    up, samp_up, r1 = _interval_extrema(resp, output, tol=tolerance_pu, min_dt=min_interval_s, upper=True, budget=budget)
    lo_budget = budget if refine_lower else [0]
    lo, samp_lo, r2 = _interval_extrema(
        resp, output, tol=tolerance_pu, min_dt=min_interval_s, upper=False, budget=lo_budget
    )
    times = resp.times_s
    nb = base.size - 1
    bu = np.full(nb, -np.inf)
    bl = np.full(nb, np.inf)
    idx = np.searchsorted(base, times[:-1] + 1e-12, side="right") - 1
    idx = np.clip(idx, 0, max(nb - 1, 0))
    for l in range(times.size - 1):
        k = int(idx[l])
        bu[k] = max(bu[k], up[l])
        bl[k] = min(bl[k], lo[l])
    in_h = times <= horizon + 1e-12
    int_h = times[1:] <= horizon + 1e-12
    point_vals = np.concatenate((resp.phi_left[in_h, output], resp.phi_right[in_h, output]))
    sampled_max = float(np.max(point_vals))
    sampled_min = float(np.min(point_vals))
    peak = max(sampled_max, float(np.max(up[int_h])) if np.any(int_h) else -np.inf)
    trough = min(sampled_min, float(np.min(lo[int_h])) if np.any(int_h) else np.inf)
    base_pos = np.searchsorted(times, base - 1e-12)
    base_pos = np.clip(base_pos, 0, times.size - 1)
    return RigorousEnvelope(
        base_times_s=base,
        values=resp.phi_right[base_pos, output].copy(),
        upper=bu,
        lower=bl,
        peak_upper=float(peak),
        trough_lower=float(trough),
        sampled_max=sampled_max,
        sampled_min=sampled_min,
        refined_intervals=int(r1 + r2),
        min_interval_s=float(min_interval_s),
        fine_times_s=times.copy(),
        fine_values=resp.phi_right[:, output].copy(),
        fine_left_values=resp.phi_left[:, output].copy(),
        fine_upper=np.asarray(up, dtype=float).copy(),
        fine_lower=np.asarray(lo, dtype=float).copy(),
    )


def trip_excursion_upper(envelope: RigorousEnvelope, *, horizon_s: float, dwell_s: float) -> float:
    return dwell_excursion_upper(
        envelope.fine_times_s,
        envelope.fine_values,
        envelope.fine_left_values,
        envelope.fine_upper,
        horizon_s=horizon_s,
        dwell_s=dwell_s,
    )


@dataclass(frozen=True)
class RigorousWindowMaps:
    asset_ids: tuple[str, ...]
    event_ids: tuple[str, ...]
    margins_pu: np.ndarray
    horizon_s: float
    dwell_s: np.ndarray
    pickup_excursion_pu: np.ndarray
    trip_excursion_pu: np.ndarray
    sampled_peak_pu: np.ndarray
    padding_pu: np.ndarray
    k_pickup: np.ndarray
    k_trip: np.ndarray
    classes: np.ndarray
    tolerance_met: np.ndarray
    reasons: Mapping[str, str]
    padding_tolerance_pu: float
    metadata: Mapping[str, Any] | None = None

    @property
    def self_mask(self) -> np.ndarray:
        return self.classes == "self"

    def column(self, event_id: str) -> int:
        return self.event_ids.index(event_id)

    def row(self, asset_id: str) -> int:
        return self.asset_ids.index(asset_id)

    def to_window_result(self, *, mode_id: str = "pre_event", control_ids: Sequence[str] = ()) -> WindowMapResult:
        data_limited = tuple(
            a for i, a in enumerate(self.asset_ids) if not (self.margins_pu[i] > 0.0)
        )
        assessments: list[ChannelAssessment] = []
        for i, asset in enumerate(self.asset_ids):
            for j, event in enumerate(self.event_ids):
                cls = str(self.classes[i, j])
                status = {
                    "certified": AssessmentStatus.CERTIFIED,
                    "self": AssessmentStatus.NOT_EVALUATED,
                    "data_limited": AssessmentStatus.DATA_LIMITED,
                }.get(cls, AssessmentStatus.FAILED)
                assessments.append(
                    ChannelAssessment(
                        protected_asset_id=asset,
                        channel_id=event,
                        status=status,
                        bound_type="rigorous_lognorm" if cls == "certified" else cls,
                        notes=self.reasons.get(f"{asset}:{event}"),
                    )
                )
        overall = AssessmentStatus.CERTIFIED
        if data_limited:
            overall = AssessmentStatus.DATA_LIMITED
        elif np.any(self.classes == "failed"):
            overall = AssessmentStatus.FAILED
        return WindowMapResult(
            mode_id=mode_id,
            protected_asset_ids=self.asset_ids,
            event_ids=self.event_ids,
            control_ids=tuple(control_ids),
            horizons_s=tuple(float(self.horizon_s) for _ in self.asset_ids),
            k_pickup_upper=_matrix_tuple(self.k_pickup),
            k_trip_upper=_matrix_tuple(self.k_trip),
            control_authority_lower=(),
            channel_assessments=tuple(assessments),
            data_limited_assets=data_limited,
            status=overall,
            metadata={
                "formula": "rigorous_finite_window_maps",
                "padding_tolerance_pu": self.padding_tolerance_pu,
                **dict(self.metadata or {}),
            },
        )

    def to_dict(self) -> dict[str, Any]:
        def mat(a: np.ndarray) -> list[list[Any]]:
            return [[(float(v) if isinstance(v, (float, np.floating)) and math.isfinite(float(v)) else (None if isinstance(v, (float, np.floating)) else str(v))) for v in row] for row in a]

        return {
            "asset_ids": list(self.asset_ids),
            "event_ids": list(self.event_ids),
            "margins_pu": [float(v) for v in self.margins_pu],
            "horizon_s": self.horizon_s,
            "dwell_s": [float(v) for v in self.dwell_s],
            "pickup_excursion_pu": mat(self.pickup_excursion_pu),
            "trip_excursion_pu": mat(self.trip_excursion_pu),
            "sampled_peak_pu": mat(self.sampled_peak_pu),
            "padding_pu": mat(self.padding_pu),
            "k_pickup": mat(self.k_pickup),
            "k_trip": mat(self.k_trip),
            "classes": [[str(v) for v in row] for row in self.classes],
            "tolerance_met": [[bool(v) for v in row] for row in self.tolerance_met],
            "reasons": dict(self.reasons),
            "padding_tolerance_pu": self.padding_tolerance_pu,
            "metadata": dict(self.metadata or {}),
        }


def assemble_window_maps(
    responses: Sequence[Any],
    *,
    asset_ids: Sequence[str],
    output_index: Mapping[str, int] | Sequence[int] | None = None,
    margins_pu: Sequence[float],
    dwell_s: float | Sequence[float] = 0.05,
    horizon_s: float | None = None,
    failed_events: Mapping[str, str] | None = None,
    event_order: Sequence[str] | None = None,
    padding_tolerance_pu: float = 1e-4,
    metadata: Mapping[str, Any] | None = None,
) -> RigorousWindowMaps:
    assets = tuple(str(a) for a in asset_ids)
    p = len(assets)
    by_id = {str(r.event_id): r for r in responses}
    failed = {str(k): str(v) for k, v in dict(failed_events or {}).items()}
    if event_order is None:
        events = tuple(by_id) + tuple(e for e in failed if e not in by_id)
    else:
        events = tuple(str(e) for e in event_order)
        missing = [e for e in events if e not in by_id and e not in failed]
        if missing:
            raise FiniteWindowError("events without response or failure reason: " + ", ".join(missing))
    q = len(events)
    if output_index is None:
        out_idx = list(range(p))
    elif isinstance(output_index, Mapping):
        out_idx = [int(output_index[a]) for a in assets]
    else:
        out_idx = [int(v) for v in output_index]
    if len(out_idx) != p:
        raise FiniteWindowError("output_index length must match asset_ids")
    h = np.asarray(margins_pu, dtype=float).reshape(-1)
    if h.size != p:
        raise FiniteWindowError("margins_pu length must match asset_ids")
    dwell = np.broadcast_to(np.asarray(dwell_s, dtype=float), (p,)).astype(float).copy()
    if np.any(~np.isfinite(dwell)) or np.any(dwell < 0.0):
        raise FiniteWindowError("dwell_s must be finite and nonnegative")

    pk = np.zeros((p, q))
    tr = np.zeros((p, q))
    samp = np.zeros((p, q))
    pad = np.zeros((p, q))
    kpk = np.zeros((p, q))
    ktr = np.zeros((p, q))
    classes = np.full((p, q), "failed", dtype=object)
    tol_met = np.zeros((p, q), dtype=bool)
    reasons: dict[str, str] = {}
    horizon_used: float | None = None
    for j, event in enumerate(events):
        resp = by_id.get(event)
        if resp is None:
            for a in assets:
                reasons[f"{a}:{event}"] = failed.get(event, "response_unavailable")
            pk[:, j] = np.nan
            tr[:, j] = np.nan
            samp[:, j] = np.nan
            pad[:, j] = np.nan
            continue
        disconnected = set(getattr(resp, "disconnects_assets", ()) or ())
        T = float(resp.times_s[-1] if horizon_s is None else horizon_s)
        horizon_used = T if horizon_used is None else horizon_used
        for i, asset in enumerate(assets):
            env = resp.envelopes[out_idx[i]]
            peak = float(env.peak_upper)
            if horizon_s is not None and horizon_s < float(env.fine_times_s[-1]) - 1e-12:
                mask_pts = env.fine_times_s <= T + 1e-12
                mask_int = env.fine_times_s[1:] <= T + 1e-12
                vals = np.concatenate((env.fine_values[mask_pts], env.fine_left_values[mask_pts]))
                peak = max(float(np.max(vals)), float(np.max(env.fine_upper[mask_int])) if np.any(mask_int) else -math.inf)
                sampled = float(np.max(vals))
            else:
                sampled = float(env.sampled_max)
            m_tr = trip_excursion_upper(env, horizon_s=T, dwell_s=float(dwell[i]))
            pk[i, j] = peak
            tr[i, j] = m_tr
            samp[i, j] = sampled
            pad[i, j] = peak - sampled
            tol_met[i, j] = pad[i, j] <= padding_tolerance_pu + 1e-15
            if asset in disconnected:
                classes[i, j] = "self"
                reasons[f"{asset}:{event}"] = "asset_disconnected_by_event"
                continue
            if not (h[i] > 0.0) or not math.isfinite(float(h[i])):
                classes[i, j] = "data_limited"
                reasons[f"{asset}:{event}"] = "nonpositive_margin"
                continue
            if not (math.isfinite(peak) and (math.isfinite(m_tr) or m_tr == -math.inf)):
                classes[i, j] = "failed"
                reasons[f"{asset}:{event}"] = "nonfinite_bound"
                continue
            classes[i, j] = "certified"
            kpk[i, j] = max(peak, 0.0) / h[i]
            ktr[i, j] = (max(m_tr, 0.0) if math.isfinite(m_tr) else 0.0) / h[i]
            if not tol_met[i, j]:
                reasons[f"{asset}:{event}"] = f"rigorous_bound_padding_{pad[i, j]:.2e}_exceeds_tolerance"
    return RigorousWindowMaps(
        asset_ids=assets,
        event_ids=events,
        margins_pu=h.copy(),
        horizon_s=float(horizon_used if horizon_used is not None else (horizon_s or 0.0)),
        dwell_s=dwell,
        pickup_excursion_pu=pk,
        trip_excursion_pu=tr,
        sampled_peak_pu=samp,
        padding_pu=pad,
        k_pickup=kpk,
        k_trip=ktr,
        classes=classes,
        tolerance_met=tol_met,
        reasons=reasons,
        padding_tolerance_pu=float(padding_tolerance_pu),
        metadata={"bound": "balanced_lognorm_interval_bound", **dict(metadata or {})},
    )


def dwell_excursion_upper(
    times_s: np.ndarray,
    values_right: np.ndarray,
    values_left: np.ndarray | None,
    interval_upper: np.ndarray | None,
    *,
    horizon_s: float,
    dwell_s: float,
) -> float:
    t = np.asarray(times_s, dtype=float)
    vr = np.asarray(values_right, dtype=float)
    vl = vr if values_left is None else np.asarray(values_left, dtype=float)
    dwell = float(dwell_s)
    horizon = float(horizon_s)
    if dwell <= 0.0:
        pts = np.maximum(vr[t <= horizon + 1e-12], vl[t <= horizon + 1e-12])
        best = float(np.max(pts)) if pts.size else -math.inf
        if interval_upper is not None and interval_upper.size:
            mask = t[1:] <= horizon + 1e-12
            if np.any(mask):
                best = max(best, float(np.max(interval_upper[mask])))
        return best
    if horizon + 1e-12 < dwell:
        return -math.inf
    vmin = np.minimum(vr, vl)
    latest = horizon - dwell
    crit = np.concatenate((t, t - dwell, [0.0, latest]))
    crit = np.unique(np.clip(crit, 0.0, latest))
    cand = np.unique(np.concatenate((crit, 0.5 * (crit[:-1] + crit[1:]))))
    best = -math.inf
    for tau in cand:
        lo_i = int(np.searchsorted(t, tau - 1e-12, side="left"))
        hi_i = int(np.searchsorted(t, tau + dwell + 1e-12, side="right"))
        if hi_i > lo_i:
            vals = vmin[lo_i:hi_i].copy()
            if abs(t[lo_i] - tau) <= 1e-12:
                vals[0] = vr[lo_i]
            cand_val = float(np.min(vals))
        else:
            if interval_upper is None:
                raise FiniteWindowError("dwell window contains no grid point; interval bounds required")
            k = int(np.clip(np.searchsorted(t, tau, side="right") - 1, 0, interval_upper.size - 1))
            cand_val = float(interval_upper[k])
        best = max(best, cand_val)
    return best
