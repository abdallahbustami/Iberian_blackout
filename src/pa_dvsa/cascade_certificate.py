r"""Threshold-cascade certificates.

This module implements the additive pickup map from ``LaTeX/root.tex``:

.. math::

   \Phi_m(S)=S\cup\{i\in\mathcal C\setminus S:
   \bar r_i^{0,m}+\sum_{j\in S}\bar K_{ij}^{m,\mathrm{pk}}\ge 1\}.

The map uses the pickup matrix, not the trip matrix.  For delayed relays, the
predicted pickup set can be checked against combined finite-window waveforms so
that dwell logic is applied to the aggregate voltage trace rather than to a sum
of individual dwell metrics.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence

import numpy as np

from .data_model import AssessmentStatus, CascadeLayer, CascadeResult, WindowMapResult
from .finite_window import FiniteWindowMapsResult, ProfileResponse
from .protected_outputs import ProtectedOutputEvaluation


class CascadeCertificateError(RuntimeError):
    """Raised when a cascade certificate cannot be computed consistently."""


@dataclass(frozen=True, slots=True)
class DelayedTripConfirmation:
    """Layer-wise delayed-trip check from a combined disturbance waveform."""

    seed_id: str
    asset_id: str
    layer_index: int
    source_event_ids: tuple[str, ...]
    pickup_erosion: float
    trip_erosion: float
    pickup_excursion_pu: float
    trip_excursion_pu: float
    margin_pu: float | None
    dwell_time_s: float
    horizon_s: float
    confirmed: bool
    status: AssessmentStatus
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "seed_id": self.seed_id,
            "asset_id": self.asset_id,
            "layer_index": self.layer_index,
            "source_event_ids": list(self.source_event_ids),
            "pickup_erosion": self.pickup_erosion,
            "trip_erosion": self.trip_erosion,
            "pickup_excursion_pu": self.pickup_excursion_pu,
            "trip_excursion_pu": self.trip_excursion_pu,
            "margin_pu": self.margin_pu,
            "dwell_time_s": self.dwell_time_s,
            "horizon_s": self.horizon_s,
            "confirmed": self.confirmed,
            "status": self.status.value,
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class SeedRanking:
    """Operator-facing seed ranking entry."""

    seed_id: str
    rank: int
    first_secondary_pickup: tuple[str, ...]
    final_fixed_point: tuple[str, ...]
    first_secondary_exceedance: float
    worst_exceedance: float
    final_fixed_point_size: int
    delayed_trip_confirmed: tuple[str, ...]
    data_limited_assets: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "seed_id": self.seed_id,
            "rank": self.rank,
            "first_secondary_pickup": list(self.first_secondary_pickup),
            "final_fixed_point": list(self.final_fixed_point),
            "first_secondary_exceedance": self.first_secondary_exceedance,
            "worst_exceedance": self.worst_exceedance,
            "final_fixed_point_size": self.final_fixed_point_size,
            "delayed_trip_confirmed": list(self.delayed_trip_confirmed),
            "data_limited_assets": list(self.data_limited_assets),
        }


@dataclass(frozen=True, slots=True)
class CascadeCertificateResult:
    """Cascade results, delayed-trip checks, and ranked seed summary."""

    mode_id: str
    seed_results: tuple[CascadeResult, ...]
    seed_rankings: tuple[SeedRanking, ...]
    delayed_trip_confirmations: tuple[DelayedTripConfirmation, ...]
    asset_event_map: Mapping[str, str]
    base_erosion: tuple[float, ...]
    data_limited_assets: tuple[str, ...]
    status: AssessmentStatus
    metadata: Mapping[str, Any] | None = None

    @property
    def cascade_result(self) -> CascadeResult | None:
        """Return the highest-ranked seed result, if any."""

        if not self.seed_rankings:
            return None
        top_seed = self.seed_rankings[0].seed_id
        for result in self.seed_results:
            if result.seed_ids == (top_seed,):
                return result
        return self.seed_results[0] if self.seed_results else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode_id": self.mode_id,
            "seed_results": [item.to_dict() for item in self.seed_results],
            "seed_rankings": [item.to_dict() for item in self.seed_rankings],
            "delayed_trip_confirmations": [
                item.to_dict() for item in self.delayed_trip_confirmations
            ],
            "asset_event_map": dict(self.asset_event_map),
            "base_erosion": list(self.base_erosion),
            "data_limited_assets": list(self.data_limited_assets),
            "status": self.status.value,
            "metadata": dict(self.metadata or {}),
        }


def phi_m(
    window_result: WindowMapResult,
    active_assets: Sequence[str],
    *,
    base_erosion: Sequence[float] | None = None,
    asset_event_map: Mapping[str, str] | None = None,
    threshold: float = 1.0,
) -> tuple[str, ...]:
    """Evaluate the additive threshold map ``Phi_m(S)``.

    ``active_assets`` is the set ``S`` of already-tripped protected elements.
    The contribution of each active element is read from the event column given
    by ``asset_event_map``.  If no mapping is supplied, identity event IDs and
    ``trip_<asset_id>`` event IDs are inferred when present.
    """

    context = _window_context(window_result, base_erosion, asset_event_map, threshold=threshold)
    active = _asset_set(active_assets, context.protected_index)
    missing = sorted(asset for asset in active if asset not in context.asset_event_map)
    if missing:
        raise CascadeCertificateError(
            "missing event columns for active protected assets: " + ", ".join(missing)
        )
    scores = _additive_scores(context, active)
    next_assets = set(active)
    for asset in context.protected_asset_ids:
        if asset not in next_assets and scores[context.protected_index[asset]] >= context.threshold:
            next_assets.add(asset)
    return _ordered_assets(context.protected_asset_ids, next_assets)


def compute_cascade_certificate(
    window_result: WindowMapResult | FiniteWindowMapsResult,
    *,
    seed_ids: Sequence[str] | None = None,
    base_erosion: Sequence[float] | None = None,
    asset_event_map: Mapping[str, str] | None = None,
    finite_response: ProfileResponse | None = None,
    protected_outputs: Sequence[ProtectedOutputEvaluation] | None = None,
    margins_pu: Sequence[float] | None = None,
    dwell_times_s: Sequence[float] | None = None,
    threshold: float = 1.0,
    metadata: Mapping[str, Any] | None = None,
) -> CascadeCertificateResult:
    """Compute cascade fixed points and rank seed events.

    The pickup prediction uses ``WindowMapResult.k_pickup_upper``.  If
    ``finite_response`` or a ``FiniteWindowMapsResult`` disturbance response is
    available, delayed trips are checked using combined source waveforms and the
    supplied relay dwell times.
    """

    finite_window = window_result if isinstance(window_result, FiniteWindowMapsResult) else None
    window = finite_window.window_result if finite_window is not None else window_result
    response = finite_response or (None if finite_window is None else finite_window.disturbance_response)
    context = _window_context(window, base_erosion, asset_event_map, threshold=threshold)
    seeds = _seed_ids(seed_ids, context.event_ids)
    margins = _margins(context.protected_asset_ids, protected_outputs, margins_pu)
    dwell = _dwell_times(dwell_times_s, len(context.protected_asset_ids))

    seed_results: list[CascadeResult] = []
    ranking_inputs: list[_RankingInput] = []
    confirmations: list[DelayedTripConfirmation] = []
    data_limited_global = set(window.data_limited_assets)

    for seed_id in seeds:
        seed_data_limited: set[str] = set(window.data_limited_assets)
        initial, initial_scores, exogenous_sources = _initial_assets_for_seed(seed_id, context)
        layers, first_scores, worst_exceedance, missing_map_assets = _iterate_layers(
            context,
            initial,
            initial_scores=initial_scores,
            exogenous_source_events=exogenous_sources,
        )
        seed_data_limited.update(missing_map_assets)

        seed_asset = _seed_asset(seed_id, context)
        first_secondary, first_secondary_exceedance = _first_secondary(
            layers,
            first_scores,
            seed_asset=seed_asset,
            protected_index=context.protected_index,
        )
        if response is not None:
            seed_confirmations = _delayed_confirmations(
                seed_id=seed_id,
                layers=layers,
                seed_asset=seed_asset,
                exogenous_source_events=exogenous_sources,
                context=context,
                response=response,
                margins=margins,
                dwell_times=dwell,
            )
            confirmations.extend(seed_confirmations)
            for item in seed_confirmations:
                if item.status == AssessmentStatus.DATA_LIMITED:
                    seed_data_limited.add(item.asset_id)
        elif np.any(dwell > 0.0):
            seed_data_limited.update(
                _assets_needing_delayed_confirmation(layers, seed_asset=seed_asset)
            )

        fixed_point = layers[-1].cumulative_set if layers else ()
        secondary_exists = bool(first_secondary)
        status = AssessmentStatus.DATA_LIMITED if seed_data_limited else AssessmentStatus.CERTIFIED
        result = CascadeResult(
            mode_id=context.mode_id,
            seed_ids=(seed_id,),
            layers=tuple(layers),
            fixed_point=fixed_point,
            no_secondary_certified=(not secondary_exists and not seed_data_limited),
            status=status,
            data_limited_assets=tuple(sorted(seed_data_limited)),
            notes=(
                "pickup prediction uses k_pickup_upper; delayed trips checked from combined "
                "waveforms when available"
            ),
        )
        seed_results.append(result)
        delayed_confirmed = tuple(
            item.asset_id
            for item in confirmations
            if item.seed_id == seed_id and item.confirmed
        )
        ranking_inputs.append(
            _RankingInput(
                seed_id=seed_id,
                first_secondary_pickup=first_secondary,
                final_fixed_point=fixed_point,
                first_secondary_exceedance=first_secondary_exceedance,
                worst_exceedance=worst_exceedance,
                delayed_trip_confirmed=delayed_confirmed,
                data_limited_assets=tuple(sorted(seed_data_limited)),
            )
        )
        data_limited_global.update(seed_data_limited)

    rankings = _rank_seeds(ranking_inputs)
    status = AssessmentStatus.DATA_LIMITED if data_limited_global else AssessmentStatus.CERTIFIED
    return CascadeCertificateResult(
        mode_id=context.mode_id,
        seed_results=tuple(seed_results),
        seed_rankings=rankings,
        delayed_trip_confirmations=tuple(confirmations),
        asset_event_map=context.asset_event_map,
        base_erosion=tuple(float(item) for item in context.base_erosion),
        data_limited_assets=tuple(sorted(data_limited_global)),
        status=status,
        metadata={
            "formula": "Phi_m(S)=S union {i not in S: r0_i + sum_{j in S} K_pk_ij >= 1}",
            "pickup_matrix": "k_pickup_upper",
            "delayed_trip_confirmation": "combined_waveform_dwell"
            if response is not None
            else "not_available",
            **dict(metadata or {}),
        },
    )


@dataclass(frozen=True, slots=True)
class _WindowContext:
    mode_id: str
    protected_asset_ids: tuple[str, ...]
    event_ids: tuple[str, ...]
    horizons_s: tuple[float, ...]
    protected_index: Mapping[str, int]
    event_index: Mapping[str, int]
    k_pickup: np.ndarray
    base_erosion: np.ndarray
    asset_event_map: Mapping[str, str]
    threshold: float


@dataclass(frozen=True, slots=True)
class _RankingInput:
    seed_id: str
    first_secondary_pickup: tuple[str, ...]
    final_fixed_point: tuple[str, ...]
    first_secondary_exceedance: float
    worst_exceedance: float
    delayed_trip_confirmed: tuple[str, ...]
    data_limited_assets: tuple[str, ...]


def _window_context(
    window: WindowMapResult,
    base_erosion: Sequence[float] | None,
    asset_event_map: Mapping[str, str] | None,
    *,
    threshold: float,
) -> _WindowContext:
    threshold_value = float(threshold)
    if not math.isfinite(threshold_value) or threshold_value <= 0.0:
        raise CascadeCertificateError("threshold must be finite and positive")

    protected = tuple(window.protected_asset_ids)
    events = tuple(window.event_ids)
    horizons = tuple(float(item) for item in window.horizons_s)
    if len(horizons) != len(protected):
        raise CascadeCertificateError("horizons_s must match protected assets")
    protected_index = {asset: index for index, asset in enumerate(protected)}
    event_index = {event: index for index, event in enumerate(events)}
    k_pickup = np.asarray(window.k_pickup_upper, dtype=float)
    if k_pickup.shape != (len(protected), len(events)):
        raise CascadeCertificateError("k_pickup_upper shape does not match protected/event IDs")
    if not np.all(np.isfinite(k_pickup)) or np.any(k_pickup < 0.0):
        raise CascadeCertificateError("k_pickup_upper must be finite and nonnegative")
    r0 = _base_erosion(base_erosion, len(protected))
    mapping = _asset_event_map(protected, event_index, asset_event_map)
    return _WindowContext(
        mode_id=window.mode_id,
        protected_asset_ids=protected,
        event_ids=events,
        horizons_s=horizons,
        protected_index=protected_index,
        event_index=event_index,
        k_pickup=k_pickup,
        base_erosion=r0,
        asset_event_map=mapping,
        threshold=threshold_value,
    )


def _base_erosion(base_erosion: Sequence[float] | None, count: int) -> np.ndarray:
    if base_erosion is None:
        return np.zeros(count, dtype=float)
    array = np.asarray(base_erosion, dtype=float).reshape(-1)
    if array.size != count:
        raise CascadeCertificateError(f"base_erosion has length {array.size}, expected {count}")
    if not np.all(np.isfinite(array)) or np.any(array < 0.0):
        raise CascadeCertificateError("base_erosion must be finite and nonnegative")
    return array


def _asset_event_map(
    protected_asset_ids: Sequence[str],
    event_index: Mapping[str, int],
    supplied: Mapping[str, str] | None,
) -> dict[str, str]:
    mapping: dict[str, str] = {}
    supplied_map = {str(key): str(value) for key, value in dict(supplied or {}).items()}
    for asset in protected_asset_ids:
        if asset in supplied_map:
            event_id = supplied_map[asset]
        elif asset in event_index:
            event_id = asset
        elif f"trip_{asset}" in event_index:
            event_id = f"trip_{asset}"
        else:
            continue
        if event_id not in event_index:
            raise CascadeCertificateError(
                f"asset_event_map for {asset!r} references unknown event {event_id!r}"
            )
        mapping[asset] = event_id
    return mapping


def _asset_set(active_assets: Sequence[str], protected_index: Mapping[str, int]) -> set[str]:
    active = {str(item) for item in active_assets}
    unknown = sorted(item for item in active if item not in protected_index)
    if unknown:
        raise CascadeCertificateError("unknown protected assets: " + ", ".join(unknown))
    return active


def _additive_scores(context: _WindowContext, active: set[str]) -> np.ndarray:
    scores = context.base_erosion.copy()
    for asset in active:
        event_id = context.asset_event_map.get(asset)
        if event_id is None:
            continue
        scores += context.k_pickup[:, context.event_index[event_id]]
    return scores


def _ordered_assets(protected_asset_ids: Sequence[str], assets: set[str]) -> tuple[str, ...]:
    return tuple(asset for asset in protected_asset_ids if asset in assets)


def _seed_ids(seed_ids: Sequence[str] | None, event_ids: Sequence[str]) -> tuple[str, ...]:
    seeds = tuple(str(item) for item in seed_ids) if seed_ids is not None else tuple(event_ids)
    if any(not item.strip() for item in seeds):
        raise CascadeCertificateError("seed IDs must be non-empty")
    if len(set(seeds)) != len(seeds):
        raise CascadeCertificateError("seed IDs must be unique")
    return seeds


def _seed_asset(seed_id: str, context: _WindowContext) -> str | None:
    for asset, event_id in context.asset_event_map.items():
        if event_id == seed_id:
            return asset
    if seed_id in context.protected_index:
        return seed_id
    return None


def _initial_assets_for_seed(
    seed_id: str,
    context: _WindowContext,
) -> tuple[set[str], dict[str, float], tuple[str, ...]]:
    seed_asset = _seed_asset(seed_id, context)
    if seed_asset is not None:
        return {seed_asset}, {}, ()
    if seed_id not in context.event_index:
        raise CascadeCertificateError(f"seed {seed_id!r} is not an event or protected asset")
    scores = context.base_erosion + context.k_pickup[:, context.event_index[seed_id]]
    initial = {
        asset
        for asset in context.protected_asset_ids
        if scores[context.protected_index[asset]] >= context.threshold
    }
    return initial, {asset: float(scores[context.protected_index[asset]]) for asset in initial}, (
        seed_id,
    )


def _iterate_layers(
    context: _WindowContext,
    initial_assets: set[str],
    *,
    initial_scores: Mapping[str, float],
    exogenous_source_events: Sequence[str],
) -> tuple[list[CascadeLayer], np.ndarray, float, tuple[str, ...]]:
    current = set(initial_assets)
    layers: list[CascadeLayer] = [
        CascadeLayer(
            0,
            _ordered_assets(context.protected_asset_ids, current),
            _ordered_assets(context.protected_asset_ids, current),
            _max_exceedance(initial_scores.values(), context.threshold),
        )
    ]
    if exogenous_source_events:
        first_scores = context.base_erosion.copy()
        for event_id in exogenous_source_events:
            first_scores += context.k_pickup[:, context.event_index[event_id]]
    else:
        first_scores = _scores_with_exogenous(context, current, exogenous_source_events)
    worst_exceedance = layers[0].max_margin_exceedance
    missing_assets: set[str] = set()

    for layer_index in range(1, len(context.protected_asset_ids) + 1):
        missing_assets.update(asset for asset in current if asset not in context.asset_event_map)
        scores = _scores_with_exogenous(context, current, exogenous_source_events)
        next_assets = set(current)
        for asset in context.protected_asset_ids:
            if asset not in next_assets and scores[context.protected_index[asset]] >= context.threshold:
                next_assets.add(asset)
        new_assets = next_assets - current
        if not new_assets:
            break
        layer_exceedance = _max_exceedance(
            (scores[context.protected_index[asset]] for asset in new_assets),
            context.threshold,
        )
        worst_exceedance = max(worst_exceedance, layer_exceedance)
        layers.append(
            CascadeLayer(
                layer_index,
                _ordered_assets(context.protected_asset_ids, new_assets),
                _ordered_assets(context.protected_asset_ids, next_assets),
                layer_exceedance,
            )
        )
        current = next_assets
    return layers, first_scores, worst_exceedance, tuple(sorted(missing_assets))


def _scores_with_exogenous(
    context: _WindowContext,
    active: set[str],
    exogenous_source_events: Sequence[str],
) -> np.ndarray:
    scores = _additive_scores(context, active)
    for event_id in exogenous_source_events:
        scores += context.k_pickup[:, context.event_index[event_id]]
    return scores


def _max_exceedance(values: Sequence[float] | Any, threshold: float) -> float:
    array = np.asarray(list(values), dtype=float)
    if array.size == 0:
        return 0.0
    return float(np.max(array - threshold))


def _first_secondary(
    layers: Sequence[CascadeLayer],
    first_scores: np.ndarray,
    *,
    seed_asset: str | None,
    protected_index: Mapping[str, int],
) -> tuple[tuple[str, ...], float]:
    if seed_asset is None:
        first = layers[0].newly_picked_up if layers else ()
    elif len(layers) > 1:
        first = layers[1].newly_picked_up
    else:
        first = ()
    if first:
        exceedance = max(first_scores[protected_index[asset]] - 1.0 for asset in first)
    else:
        excluded = {seed_asset} if seed_asset is not None else set()
        candidates = [
            first_scores[index] - 1.0
            for asset, index in protected_index.items()
            if asset not in excluded
        ]
        exceedance = max(candidates) if candidates else 0.0
    return tuple(first), float(exceedance)


def _assets_needing_delayed_confirmation(
    layers: Sequence[CascadeLayer],
    *,
    seed_asset: str | None,
) -> tuple[str, ...]:
    assets: list[str] = []
    for layer in layers:
        if layer.layer_index == 0 and seed_asset is not None:
            continue
        assets.extend(layer.newly_picked_up)
    return tuple(dict.fromkeys(assets))


def _margins(
    protected_asset_ids: Sequence[str],
    protected_outputs: Sequence[ProtectedOutputEvaluation] | None,
    margins_pu: Sequence[float] | None,
) -> np.ndarray | None:
    if margins_pu is not None:
        margins = np.asarray(margins_pu, dtype=float).reshape(-1)
    elif protected_outputs is not None:
        if len(protected_outputs) != len(protected_asset_ids):
            raise CascadeCertificateError("protected_outputs length must match protected assets")
        by_asset = {item.asset_id: item for item in protected_outputs}
        if set(by_asset) != set(protected_asset_ids):
            raise CascadeCertificateError("protected_outputs asset IDs must match window result")
        margins = np.asarray(
            [by_asset[asset].worst_case_margin_pu for asset in protected_asset_ids], dtype=float
        )
    else:
        return None
    if margins.size != len(protected_asset_ids):
        raise CascadeCertificateError(
            f"margins_pu has length {margins.size}, expected {len(protected_asset_ids)}"
        )
    if not np.all(np.isfinite(margins)):
        raise CascadeCertificateError("margins_pu must be finite")
    return margins


def _dwell_times(dwell_times_s: Sequence[float] | None, count: int) -> np.ndarray:
    if dwell_times_s is None:
        return np.zeros(count, dtype=float)
    dwell = np.asarray(dwell_times_s, dtype=float).reshape(-1)
    if dwell.size != count:
        raise CascadeCertificateError(f"dwell_times_s has length {dwell.size}, expected {count}")
    if not np.all(np.isfinite(dwell)) or np.any(dwell < 0.0):
        raise CascadeCertificateError("dwell_times_s must be finite and nonnegative")
    return dwell


def _delayed_confirmations(
    *,
    seed_id: str,
    layers: Sequence[CascadeLayer],
    seed_asset: str | None,
    exogenous_source_events: Sequence[str],
    context: _WindowContext,
    response: ProfileResponse,
    margins: np.ndarray | None,
    dwell_times: np.ndarray,
) -> list[DelayedTripConfirmation]:
    values = np.asarray(response.values_pu, dtype=float)
    times = np.asarray(response.times_s, dtype=float)
    if values.ndim != 3:
        raise CascadeCertificateError("response values must have shape time x output x channel")
    if times.ndim != 1 or times.size != values.shape[0]:
        raise CascadeCertificateError("response times must align with response values")
    if not np.all(np.isfinite(values)) or not np.all(np.isfinite(times)):
        raise CascadeCertificateError("response contains non-finite values")
    if np.any(np.diff(times) < -1e-12):
        raise CascadeCertificateError("response times must be nondecreasing")
    if values.shape[1] != len(context.protected_asset_ids):
        raise CascadeCertificateError("response output count does not match protected assets")
    if values.shape[2] != len(response.channel_ids):
        raise CascadeCertificateError("response channel IDs do not match response values")
    if len(set(response.channel_ids)) != len(response.channel_ids):
        raise CascadeCertificateError("response channel IDs must be unique")
    response_event_index = {event_id: index for index, event_id in enumerate(response.channel_ids)}
    confirmations: list[DelayedTripConfirmation] = []
    previous_cumulative: set[str] = set()

    for layer in layers:
        if layer.layer_index == 0 and seed_asset is not None:
            previous_cumulative = set(layer.cumulative_set)
            continue
        source_events = _source_events(previous_cumulative, exogenous_source_events, context)
        for asset in layer.newly_picked_up:
            confirmations.append(
                _confirm_asset_trip(
                    seed_id=seed_id,
                    asset_id=asset,
                    layer_index=layer.layer_index,
                    source_event_ids=source_events,
                    context=context,
                    response_values=values,
                    response_times=times,
                    response_event_index=response_event_index,
                    margins=margins,
                    dwell_times=dwell_times,
                )
            )
        previous_cumulative = set(layer.cumulative_set)

    return confirmations


def _source_events(
    previous_cumulative: set[str],
    exogenous_source_events: Sequence[str],
    context: _WindowContext,
) -> tuple[str, ...]:
    events = list(exogenous_source_events)
    for asset in context.protected_asset_ids:
        if asset in previous_cumulative and asset in context.asset_event_map:
            events.append(context.asset_event_map[asset])
    return tuple(dict.fromkeys(events))


def _confirm_asset_trip(
    *,
    seed_id: str,
    asset_id: str,
    layer_index: int,
    source_event_ids: Sequence[str],
    context: _WindowContext,
    response_values: np.ndarray,
    response_times: np.ndarray,
    response_event_index: Mapping[str, int],
    margins: np.ndarray | None,
    dwell_times: np.ndarray,
) -> DelayedTripConfirmation:
    asset_idx = context.protected_index[asset_id]
    horizon = float(0.0)
    margin_value: float | None = None
    if margins is not None:
        margin_value = float(margins[asset_idx])
    unavailable = [event_id for event_id in source_event_ids if event_id not in response_event_index]
    if margins is None:
        return _unconfirmed(
            seed_id,
            asset_id,
            layer_index,
            source_event_ids,
            None,
            float(dwell_times[asset_idx]),
            horizon,
            "margin_unavailable",
        )
    if margin_value is None or margin_value <= 0.0:
        return _unconfirmed(
            seed_id,
            asset_id,
            layer_index,
            source_event_ids,
            margin_value,
            float(dwell_times[asset_idx]),
            horizon,
            "nonpositive_margin",
        )
    if unavailable:
        return _unconfirmed(
            seed_id,
            asset_id,
            layer_index,
            source_event_ids,
            margin_value,
            float(dwell_times[asset_idx]),
            horizon,
            "response_missing_source_event:" + ",".join(unavailable),
        )
    if not source_event_ids:
        return _unconfirmed(
            seed_id,
            asset_id,
            layer_index,
            source_event_ids,
            margin_value,
            float(dwell_times[asset_idx]),
            horizon,
            "source_event_unavailable",
        )

    horizon = _horizon_for_asset(context, asset_idx)
    mask = response_times <= horizon + 1e-12
    if not np.any(mask):
        return _unconfirmed(
            seed_id,
            asset_id,
            layer_index,
            source_event_ids,
            margin_value,
            float(dwell_times[asset_idx]),
            horizon,
            "no_response_samples_in_horizon",
        )
    cols = [response_event_index[event_id] for event_id in source_event_ids]
    combined = np.sum(response_values[mask, asset_idx, :][:, cols], axis=1)
    times = response_times[mask]
    pickup_excursion = float(np.max(np.maximum(combined, 0.0))) if combined.size else 0.0
    trip_excursion = _dwell_excursion_sampled(
        times,
        combined,
        horizon=horizon,
        dwell=float(dwell_times[asset_idx]),
        pickup_excursion=pickup_excursion,
    )
    pickup_erosion = pickup_excursion / margin_value
    trip_erosion = trip_excursion / margin_value
    confirmed = bool(trip_erosion >= 1.0 - 1e-12)
    return DelayedTripConfirmation(
        seed_id=seed_id,
        asset_id=asset_id,
        layer_index=layer_index,
        source_event_ids=tuple(source_event_ids),
        pickup_erosion=float(pickup_erosion),
        trip_erosion=float(trip_erosion),
        pickup_excursion_pu=pickup_excursion,
        trip_excursion_pu=trip_excursion,
        margin_pu=margin_value,
        dwell_time_s=float(dwell_times[asset_idx]),
        horizon_s=horizon,
        confirmed=confirmed,
        status=AssessmentStatus.CERTIFIED,
        reason="combined_waveform_dwell",
    )


def _horizon_for_asset(context: _WindowContext, asset_idx: int) -> float:
    return float(context.horizons_s[asset_idx])


def _unconfirmed(
    seed_id: str,
    asset_id: str,
    layer_index: int,
    source_event_ids: Sequence[str],
    margin_pu: float | None,
    dwell_time_s: float,
    horizon_s: float,
    reason: str,
) -> DelayedTripConfirmation:
    return DelayedTripConfirmation(
        seed_id=seed_id,
        asset_id=asset_id,
        layer_index=layer_index,
        source_event_ids=tuple(source_event_ids),
        pickup_erosion=0.0,
        trip_erosion=0.0,
        pickup_excursion_pu=0.0,
        trip_excursion_pu=0.0,
        margin_pu=margin_pu,
        dwell_time_s=dwell_time_s,
        horizon_s=horizon_s,
        confirmed=False,
        status=AssessmentStatus.DATA_LIMITED,
        reason=reason,
    )


def _dwell_excursion_sampled(
    times: np.ndarray,
    values: np.ndarray,
    *,
    horizon: float,
    dwell: float,
    pickup_excursion: float,
) -> float:
    if dwell <= 0.0:
        return pickup_excursion
    if horizon + 1e-12 < dwell:
        return 0.0
    latest_start = horizon - dwell
    best = -math.inf
    for start in times[times <= latest_start + 1e-12]:
        mask = (times >= start - 1e-12) & (times <= start + dwell + 1e-12)
        if np.any(mask):
            best = max(best, float(np.min(values[mask])))
    if not math.isfinite(best):
        return 0.0
    return min(max(best, 0.0), pickup_excursion)


def _rank_seeds(inputs: Sequence[_RankingInput]) -> tuple[SeedRanking, ...]:
    ordered = sorted(
        inputs,
        key=lambda item: (
            -len(item.first_secondary_pickup),
            -len(item.final_fixed_point),
            -item.worst_exceedance,
            item.seed_id,
        ),
    )
    return tuple(
        SeedRanking(
            seed_id=item.seed_id,
            rank=index + 1,
            first_secondary_pickup=item.first_secondary_pickup,
            final_fixed_point=item.final_fixed_point,
            first_secondary_exceedance=float(item.first_secondary_exceedance),
            worst_exceedance=float(item.worst_exceedance),
            final_fixed_point_size=len(item.final_fixed_point),
            delayed_trip_confirmed=item.delayed_trip_confirmed,
            data_limited_assets=item.data_limited_assets,
        )
        for index, item in enumerate(ordered)
    )
