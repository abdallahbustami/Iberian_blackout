r"""Robust finite-window screening under bounded uncertainty.

This module implements the robust quantities from ``LaTeX/root.tex``:

.. math::

   \bar K_{ij}^{m,\mathrm{pk}}=\sup_{\theta\in\Theta}K_{ij}^{m,\mathrm{pk}}(\theta),
   \quad
   \bar K_{ij}^{m,\mathrm{tr}}=\sup_{\theta\in\Theta}K_{ij}^{m,\mathrm{tr}}(\theta),
   \quad
   \bar r_i^{0,m}=\sup_{\theta\in\Theta}r_i^{0,m}(\theta).

The certificate uses bounded or worst-case quantities.  Monte Carlo sampling is
kept as supporting evidence and never relaxes the robust no-pickup decision.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence

import numpy as np

from .data_model import (
    AssessmentStatus,
    CandidateEvent,
    ChannelAssessment,
    Matrix,
    TimeGridControlAuthority,
    UncertaintySet,
    WindowMapResult,
)
from .finite_window import FiniteWindowMapsResult
from .protected_outputs import ProtectedOutputEvaluation


class RobustScreeningError(RuntimeError):
    """Raised when robust screening inputs are inconsistent."""


@dataclass(frozen=True, slots=True)
class RobustScenario:
    """One deterministic uncertainty scenario used in a scenario/interval sweep."""

    scenario_id: str
    margin_pu: Mapping[str, float] | Sequence[float] | None = None
    event_scale: Mapping[str, float] | Sequence[float] | None = None
    output_scale: Mapping[str, float] | Sequence[float] | None = None
    base_voltage_additive_pu: Mapping[str, float] | Sequence[float] | None = None
    k_additive_pu: Mapping[str, float] | Sequence[float] | None = None
    trip_uses_pickup_assets: tuple[str, ...] = ()
    control_scale: Mapping[str, float] | Sequence[float] | None = None
    control_delay_s: Mapping[str, float] | Sequence[float] | None = None
    data_limited_assets: tuple[str, ...] = ()
    metadata: Mapping[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "margin_pu": _serializable_factor(self.margin_pu),
            "event_scale": _serializable_factor(self.event_scale),
            "output_scale": _serializable_factor(self.output_scale),
            "base_voltage_additive_pu": _serializable_factor(self.base_voltage_additive_pu),
            "k_additive_pu": _serializable_factor(self.k_additive_pu),
            "trip_uses_pickup_assets": list(self.trip_uses_pickup_assets),
            "control_scale": _serializable_factor(self.control_scale),
            "control_delay_s": _serializable_factor(self.control_delay_s),
            "data_limited_assets": list(self.data_limited_assets),
            "metadata": dict(self.metadata or {}),
        }


@dataclass(frozen=True, slots=True)
class MonteCarloSummary:
    """Supporting random-sweep evidence.  Not used for certification."""

    sample_count: int
    max_k_pk_sampled: Matrix
    max_k_tr_sampled: Matrix
    max_r_sampled: tuple[float, ...]
    no_pickup_fraction: float
    rng_seed: int | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_count": self.sample_count,
            "max_k_pk_sampled": [list(row) for row in self.max_k_pk_sampled],
            "max_k_tr_sampled": [list(row) for row in self.max_k_tr_sampled],
            "max_r_sampled": list(self.max_r_sampled),
            "no_pickup_fraction": self.no_pickup_fraction,
            "rng_seed": self.rng_seed,
            "certification_note": "supporting evidence only; robust certificate uses worst-case bounds",
        }


@dataclass(frozen=True, slots=True)
class RobustScreenResult:
    """Worst-case robust matrices and the no-pickup certificate."""

    mode_id: str
    protected_asset_ids: tuple[str, ...]
    event_ids: tuple[str, ...]
    control_ids: tuple[str, ...]
    k_bar_pk: np.ndarray
    k_bar_tr: np.ndarray
    r_bar: np.ndarray
    margin_lower_pu: np.ndarray
    robust_no_pickup_certified: bool
    epsilon: float
    seed_assets: tuple[str, ...]
    seed_event_ids: tuple[str, ...]
    violating_assets: tuple[str, ...]
    data_limited_assets: tuple[str, ...]
    robust_window_result: WindowMapResult
    scenarios: tuple[RobustScenario, ...]
    monte_carlo: MonteCarloSummary | None = None
    control_authority_lower: tuple[TimeGridControlAuthority, ...] = ()
    status: AssessmentStatus = AssessmentStatus.NOT_EVALUATED
    metadata: Mapping[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode_id": self.mode_id,
            "protected_asset_ids": list(self.protected_asset_ids),
            "event_ids": list(self.event_ids),
            "control_ids": list(self.control_ids),
            "k_bar_pk": _array_to_nested(self.k_bar_pk, nonfinite_as_none=True),
            "k_bar_tr": _array_to_nested(self.k_bar_tr, nonfinite_as_none=True),
            "r_bar": _array_to_list(self.r_bar, nonfinite_as_none=True),
            "margin_lower_pu": _array_to_list(self.margin_lower_pu, nonfinite_as_none=True),
            "robust_no_pickup_certified": self.robust_no_pickup_certified,
            "epsilon": self.epsilon,
            "seed_assets": list(self.seed_assets),
            "seed_event_ids": list(self.seed_event_ids),
            "violating_assets": list(self.violating_assets),
            "data_limited_assets": list(self.data_limited_assets),
            "robust_window_result": self.robust_window_result.to_dict(),
            "scenarios": [item.to_dict() for item in self.scenarios],
            "monte_carlo": None if self.monte_carlo is None else self.monte_carlo.to_dict(),
            "control_authority_lower": [item.to_dict() for item in self.control_authority_lower],
            "status": self.status.value,
            "metadata": dict(self.metadata or {}),
        }


def compute_robust_screen(
    window_result: WindowMapResult | FiniteWindowMapsResult,
    *,
    protected_outputs: Sequence[ProtectedOutputEvaluation] | None = None,
    margins_pu: Sequence[float] | None = None,
    uncertainty_set: UncertaintySet | None = None,
    scenarios: Sequence[RobustScenario] | None = None,
    candidate_events: Sequence[CandidateEvent] | None = None,
    base_erosion: Sequence[float] | None = None,
    seed_assets: Sequence[str] = (),
    seed_event_ids: Sequence[str] = (),
    asset_event_map: Mapping[str, str] | None = None,
    epsilon: float = 0.0,
    monte_carlo_samples: int = 0,
    rng_seed: int | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> RobustScreenResult:
    """Compute robust ``K_bar_pk``, ``K_bar_tr``, ``r_bar``, and certificate.

    The robust no-pickup certificate checks
    ``r_bar_i + sum_{j in S} K_bar_pk[i,j] <= 1 - epsilon`` for every
    protected asset outside the seed set.  Any asset whose uncertainty envelope
    destroys positive margin is marked data-limited and prevents a safe
    certificate.
    """

    finite_window = window_result if isinstance(window_result, FiniteWindowMapsResult) else None
    window = finite_window.window_result if finite_window is not None else window_result
    epsilon_value = _epsilon(epsilon)
    protected = tuple(window.protected_asset_ids)
    events = tuple(window.event_ids)
    controls = tuple(window.control_ids)
    p = len(protected)
    q = len(events)

    k_pk_nom = _matrix(window.k_pickup_upper, (p, q), "k_pickup_upper")
    k_tr_nom = _matrix(window.k_trip_upper, (p, q), "k_trip_upper")
    margins_nom, protected_data_limited = _nominal_margins(
        protected, protected_outputs, margins_pu
    )
    r0 = _base_erosion(base_erosion, p)
    event_nominals = _event_nominal_values(events, candidate_events)
    generated = build_uncertainty_scenarios(
        protected_asset_ids=protected,
        event_ids=events,
        control_ids=controls,
        nominal_margins_pu=margins_nom,
        uncertainty_set=uncertainty_set,
        protected_outputs=protected_outputs,
        candidate_events=candidate_events,
        event_nominal_values=event_nominals,
    )
    scenario_list = tuple(scenarios or ()) + generated
    if not scenario_list:
        scenario_list = (RobustScenario("nominal"),)

    robust = _supremum_over_scenarios(
        scenarios=scenario_list,
        protected_asset_ids=protected,
        event_ids=events,
        control_ids=controls,
        nominal_margins=margins_nom,
        k_pk_nom=k_pk_nom,
        k_tr_nom=k_tr_nom,
        base_erosion=r0,
    )
    data_limited = set(protected_data_limited)
    data_limited.update(robust.data_limited_assets)
    robust.k_bar_pk[[asset in data_limited for asset in protected], :] = np.nan
    robust.k_bar_tr[[asset in data_limited for asset in protected], :] = np.nan
    robust.r_bar[[asset in data_limited for asset in protected]] = np.nan

    seed_assets_checked = _checked_ids(seed_assets, protected, "seed_assets")
    seed_events_checked = _checked_ids(seed_event_ids, events, "seed_event_ids")
    mapped_seed_events, excluded_assets = _seed_event_columns(
        protected,
        events,
        seed_assets_checked,
        seed_events_checked,
        asset_event_map,
    )
    scores = _robust_scores(robust.k_bar_pk, robust.r_bar, events, mapped_seed_events)
    violating = tuple(
        asset
        for index, asset in enumerate(protected)
        if asset not in excluded_assets
        and asset not in data_limited
        and math.isfinite(float(scores[index]))
        and float(scores[index]) > 1.0 - epsilon_value + 1e-12
    )
    certified = not data_limited and not violating
    status = AssessmentStatus.DATA_LIMITED if data_limited else (
        AssessmentStatus.CERTIFIED if certified else AssessmentStatus.FAILED
    )

    control_lower = _robust_control_authority(
        window, robust.control_scale_lower, robust.control_delay_upper_s
    )
    robust_window = _robust_window_result(
        window,
        robust.k_bar_pk,
        robust.k_bar_tr,
        data_limited_assets=tuple(sorted(data_limited)),
        status=status,
        control_authority_lower=control_lower,
    )
    monte_carlo = None
    if monte_carlo_samples:
        monte_carlo = _monte_carlo_summary(
            sample_count=monte_carlo_samples,
            rng_seed=rng_seed,
            protected_asset_ids=protected,
            event_ids=events,
            nominal_margins=margins_nom,
            k_pk_nom=k_pk_nom,
            k_tr_nom=k_tr_nom,
            base_erosion=r0,
            worst_margin=robust.margin_lower,
            worst_event_scale=robust.event_scale_upper,
            worst_output_scale=robust.output_scale_upper,
            worst_k_additive=robust.k_additive_upper,
            seed_event_ids=mapped_seed_events,
            excluded_assets=excluded_assets,
            epsilon=epsilon_value,
        )

    return RobustScreenResult(
        mode_id=window.mode_id,
        protected_asset_ids=protected,
        event_ids=events,
        control_ids=controls,
        k_bar_pk=robust.k_bar_pk,
        k_bar_tr=robust.k_bar_tr,
        r_bar=robust.r_bar,
        margin_lower_pu=robust.margin_lower,
        robust_no_pickup_certified=certified,
        epsilon=epsilon_value,
        seed_assets=seed_assets_checked,
        seed_event_ids=seed_events_checked,
        violating_assets=violating,
        data_limited_assets=tuple(sorted(data_limited)),
        robust_window_result=robust_window,
        scenarios=scenario_list,
        monte_carlo=monte_carlo,
        control_authority_lower=control_lower,
        status=status,
        metadata={
            "formula": "robust_K_and_r_supremum",
            "certificate": "r_bar_i + sum K_bar_pk_ij <= 1 - epsilon",
            "scenario_count": len(scenario_list),
            "monte_carlo_role": "supporting_evidence_only",
            **dict(metadata or {}),
        },
    )


def build_uncertainty_scenarios(
    *,
    protected_asset_ids: Sequence[str],
    event_ids: Sequence[str],
    control_ids: Sequence[str] = (),
    nominal_margins_pu: Sequence[float] | None = None,
    uncertainty_set: UncertaintySet | None = None,
    protected_outputs: Sequence[ProtectedOutputEvaluation] | None = None,
    candidate_events: Sequence[CandidateEvent] | None = None,
    event_nominal_values: Mapping[str, float] | None = None,
) -> tuple[RobustScenario, ...]:
    """Convert an ``UncertaintySet`` into conservative interval-sweep scenarios."""

    if uncertainty_set is None:
        return ()
    protected = tuple(protected_asset_ids)
    events = tuple(event_ids)
    controls = tuple(control_ids)
    margins = None if nominal_margins_pu is None else np.asarray(nominal_margins_pu, dtype=float)
    event_nominals = dict(event_nominal_values or {})

    margin_loss = {asset: 0.0 for asset in protected}
    output_scale = {asset: 1.0 for asset in protected}
    k_additive = {asset: 0.0 for asset in protected}
    event_scale = {event: 1.0 for event in events}
    trip_uses_pickup: set[str] = set()
    control_scale = {control: 1.0 for control in controls}
    control_delay = {control: 0.0 for control in controls}
    data_limited: set[str] = set()
    scenario_metadata: dict[str, Any] = {"uncertainty_set": uncertainty_set.set_id}
    individual: list[RobustScenario] = []

    for bound in uncertainty_set.thresholds_pu:
        for asset in _matching_assets(bound.target_id, protected, protected_outputs):
            current_threshold = _protected_threshold_lower(asset, protected_outputs)
            if current_threshold is not None:
                margin_loss[asset] += max(current_threshold - bound.interval.lower, 0.0)
            if margins is not None and margins[_asset_index(asset, protected)] - margin_loss[asset] <= 0.0:
                data_limited.add(asset)
            individual.append(
                RobustScenario(
                    f"threshold:{bound.target_id}:{bound.parameter}",
                    margin_pu=_single_margin(asset, protected, margins, margin_loss[asset]),
                    base_voltage_additive_pu=None
                    if margins is not None
                    else {asset: margin_loss[asset]},
                    metadata={"category": "thresholds_pu", "bound": bound.to_dict()},
                )
            )

    for bound in uncertainty_set.reconstruction_errors_pu:
        for asset in _matching_assets(bound.target_id, protected, protected_outputs):
            loss = max(float(bound.interval.upper), 0.0)
            margin_loss[asset] += loss
            k_additive[asset] += loss
            if margins is not None and margins[_asset_index(asset, protected)] - margin_loss[asset] <= 0.0:
                data_limited.add(asset)
            individual.append(
                RobustScenario(
                    f"reconstruction:{bound.target_id}:{bound.parameter}",
                    margin_pu=_single_margin(asset, protected, margins, margin_loss[asset]),
                    base_voltage_additive_pu=None if margins is not None else {asset: loss},
                    k_additive_pu={asset: loss},
                    metadata={"category": "reconstruction_errors_pu", "bound": bound.to_dict()},
                )
            )

    for bound in uncertainty_set.tap_ratios:
        scale = _tap_scale(bound.interval.lower, bound.interval.nominal, bound.interval.upper)
        for asset in _matching_assets(bound.target_id, protected, protected_outputs, loose=True):
            output_scale[asset] = max(output_scale[asset], scale)
            individual.append(
                RobustScenario(
                    f"tap:{bound.target_id}:{bound.parameter}",
                    output_scale={asset: scale},
                    metadata={"category": "tap_ratios", "bound": bound.to_dict()},
                )
            )

    for bound in uncertainty_set.q_absorption_mvar:
        scale = _quantity_scale(bound.interval.upper, _event_nominal(bound.target_id, event_nominals))
        for event in _matching_events(bound.target_id, events, candidate_events):
            event_scale[event] = max(event_scale[event], scale)
            individual.append(
                RobustScenario(
                    f"q_absorption:{bound.target_id}:{bound.parameter}",
                    event_scale={event: scale},
                    metadata={"category": "q_absorption_mvar", "bound": bound.to_dict()},
                )
            )

    for bound in uncertainty_set.fixed_pf_values:
        scale = _fixed_pf_scale(bound.target_id, bound.parameter, bound.interval, candidate_events)
        for event in _matching_events(bound.target_id, events, candidate_events):
            event_scale[event] = max(event_scale[event], scale)
            individual.append(
                RobustScenario(
                    f"fixed_pf:{bound.target_id}:{bound.parameter}",
                    event_scale={event: scale},
                    metadata={"category": "fixed_pf_values", "bound": bound.to_dict()},
                )
            )

    for bound in uncertainty_set.shunt_status:
        scale = _quantity_scale(bound.interval.upper, _event_nominal(bound.target_id, event_nominals))
        for event in _matching_events(bound.target_id, events, candidate_events):
            event_scale[event] = max(event_scale[event], scale)
            individual.append(
                RobustScenario(
                    f"shunt_status:{bound.target_id}:{bound.parameter}",
                    event_scale={event: scale},
                    metadata={"category": "shunt_status", "bound": bound.to_dict()},
                )
            )

    for bound in uncertainty_set.delays_s:
        for asset in _matching_assets(bound.target_id, protected, protected_outputs):
            if bound.interval.lower <= 0.0:
                trip_uses_pickup.add(asset)
            individual.append(
                RobustScenario(
                    f"delay:{bound.target_id}:{bound.parameter}",
                    trip_uses_pickup_assets=(asset,) if bound.interval.lower <= 0.0 else (),
                    metadata={"category": "delays_s", "bound": bound.to_dict()},
                )
            )

    for bound in uncertainty_set.model_errors_pu:
        for asset in _matching_assets(bound.target_id, protected, protected_outputs, loose=True):
            err = max(float(bound.interval.upper), 0.0)
            margin_loss[asset] += err
            k_additive[asset] += err
            if margins is not None and margins[_asset_index(asset, protected)] - margin_loss[asset] <= 0.0:
                data_limited.add(asset)
            individual.append(
                RobustScenario(
                    f"model_error:{bound.target_id}:{bound.parameter}",
                    margin_pu=_single_margin(asset, protected, margins, margin_loss[asset]),
                    base_voltage_additive_pu=None if margins is not None else {asset: err},
                    k_additive_pu={asset: err},
                    metadata={"category": "model_errors_pu", "bound": bound.to_dict()},
                )
            )

    for response in uncertainty_set.controller_responses:
        if response.control_id in control_scale:
            scale = 1.0
            if response.lower_bound_scale is not None:
                scale = min(scale, max(float(response.lower_bound_scale.lower), 0.0))
            if response.gain_scale is not None:
                scale = min(scale, max(float(response.gain_scale.lower), 0.0))
            control_scale[response.control_id] = min(control_scale[response.control_id], scale)
            if response.delay_s is not None:
                control_delay[response.control_id] = max(
                    control_delay[response.control_id], float(response.delay_s.upper)
                )
            individual.append(
                RobustScenario(
                    f"controller:{response.control_id}",
                    control_scale={response.control_id: scale},
                    control_delay_s={response.control_id: control_delay[response.control_id]},
                    metadata={"category": "controller_responses", "bound": response.to_dict()},
                )
            )

    for envelope in uncertainty_set.compliance_envelopes:
        scale = min(
            max(float(envelope.compliance_fraction.lower), 0.0),
            max(float(envelope.reactive_response_scale.lower), 0.0),
        )
        for control in controls:
            if control == envelope.asset_id or control.startswith(envelope.asset_id):
                control_scale[control] = min(control_scale[control], scale)
                individual.append(
                    RobustScenario(
                        f"compliance:{envelope.asset_id}",
                        control_scale={control: scale},
                        metadata={"category": "compliance_envelopes", "bound": envelope.to_dict()},
                    )
                )

    worst_margin = None
    if margins is not None:
        worst_margin = {
            asset: float(margins[index] - margin_loss[asset])
            for index, asset in enumerate(protected)
        }
    worst = RobustScenario(
        f"{uncertainty_set.set_id}:combined_worst_case",
        margin_pu=worst_margin,
        event_scale=event_scale,
        output_scale=output_scale,
        k_additive_pu=k_additive,
        trip_uses_pickup_assets=tuple(sorted(trip_uses_pickup)),
        control_scale=control_scale,
        control_delay_s=control_delay,
        data_limited_assets=tuple(sorted(data_limited)),
        metadata={
            **scenario_metadata,
            "categories": {
                "thresholds": len(uncertainty_set.thresholds_pu),
                "taps": len(uncertainty_set.tap_ratios),
                "delays": len(uncertainty_set.delays_s),
                "q_absorption": len(uncertainty_set.q_absorption_mvar),
                "fixed_pf_values": len(uncertainty_set.fixed_pf_values),
                "shunt_status": len(uncertainty_set.shunt_status),
                "controller_responses": len(uncertainty_set.controller_responses),
                "compliance_envelopes": len(uncertainty_set.compliance_envelopes),
                "reconstruction_errors": len(uncertainty_set.reconstruction_errors_pu),
                "model_errors": len(uncertainty_set.model_errors_pu),
            },
        },
    )
    return (worst, *individual)


@dataclass(slots=True)
class _RobustArrays:
    k_bar_pk: np.ndarray
    k_bar_tr: np.ndarray
    r_bar: np.ndarray
    margin_lower: np.ndarray
    event_scale_upper: np.ndarray
    output_scale_upper: np.ndarray
    k_additive_upper: np.ndarray
    control_scale_lower: np.ndarray
    control_delay_upper_s: np.ndarray
    data_limited_assets: tuple[str, ...]


def _supremum_over_scenarios(
    *,
    scenarios: Sequence[RobustScenario],
    protected_asset_ids: Sequence[str],
    event_ids: Sequence[str],
    control_ids: Sequence[str],
    nominal_margins: np.ndarray,
    k_pk_nom: np.ndarray,
    k_tr_nom: np.ndarray,
    base_erosion: np.ndarray,
) -> _RobustArrays:
    p = len(protected_asset_ids)
    q = len(event_ids)
    k_bar_pk = np.zeros((p, q), dtype=float)
    k_bar_tr = np.zeros((p, q), dtype=float)
    r_bar = np.zeros(p, dtype=float)
    margin_lower = nominal_margins.copy()
    event_scale_upper = np.ones(q, dtype=float)
    output_scale_upper = np.ones(p, dtype=float)
    k_additive_upper = np.zeros(p, dtype=float)
    control_scale_lower = np.ones(len(control_ids), dtype=float)
    control_delay_upper = np.zeros(len(control_ids), dtype=float)
    data_limited: set[str] = set()

    for scenario in scenarios:
        margins = _factor_vector(
            scenario.margin_pu,
            protected_asset_ids,
            p,
            default=nominal_margins,
            name=f"{scenario.scenario_id}.margin_pu",
        )
        event_scale = _factor_vector(
            scenario.event_scale,
            event_ids,
            q,
            default=np.ones(q),
            name=f"{scenario.scenario_id}.event_scale",
        )
        output_scale = _factor_vector(
            scenario.output_scale,
            protected_asset_ids,
            p,
            default=np.ones(p),
            name=f"{scenario.scenario_id}.output_scale",
        )
        base_add = _factor_vector(
            scenario.base_voltage_additive_pu,
            protected_asset_ids,
            p,
            default=np.zeros(p),
            name=f"{scenario.scenario_id}.base_voltage_additive_pu",
        )
        k_add = _factor_vector(
            scenario.k_additive_pu,
            protected_asset_ids,
            p,
            default=np.zeros(p),
            name=f"{scenario.scenario_id}.k_additive_pu",
        )
        control_scale = _factor_vector(
            scenario.control_scale,
            control_ids,
            len(control_ids),
            default=np.ones(len(control_ids)),
            name=f"{scenario.scenario_id}.control_scale",
        )
        control_delay = _factor_vector(
            scenario.control_delay_s,
            control_ids,
            len(control_ids),
            default=np.zeros(len(control_ids)),
            name=f"{scenario.scenario_id}.control_delay_s",
        )
        bad_rows = (~np.isfinite(margins)) | (margins <= 0.0)
        for index, asset in enumerate(protected_asset_ids):
            if bad_rows[index]:
                data_limited.add(asset)
        data_limited.update(str(item) for item in scenario.data_limited_assets)

        valid = ~bad_rows
        scenario_pk = np.full((p, q), np.nan, dtype=float)
        scenario_tr = np.full((p, q), np.nan, dtype=float)
        scenario_r = np.full(p, np.nan, dtype=float)
        if np.any(valid):
            scale = (
                nominal_margins[valid, None]
                / margins[valid, None]
                * output_scale[valid, None]
                * event_scale[None, :]
            )
            scenario_pk[valid, :] = k_pk_nom[valid, :] * scale + k_add[valid, None] / margins[
                valid, None
            ]
            trip_base = k_tr_nom.copy()
            for asset in scenario.trip_uses_pickup_assets:
                if asset in protected_asset_ids:
                    row = list(protected_asset_ids).index(asset)
                    trip_base[row, :] = np.maximum(trip_base[row, :], k_pk_nom[row, :])
            scenario_tr[valid, :] = trip_base[valid, :] * scale + k_add[valid, None] / margins[
                valid, None
            ]
            scenario_r[valid] = (
                base_erosion[valid] * nominal_margins[valid]
                + np.maximum(nominal_margins[valid] - margins[valid], 0.0)
                + base_add[valid]
            ) / margins[valid]

        k_bar_pk = np.fmax(k_bar_pk, np.nan_to_num(scenario_pk, nan=0.0))
        k_bar_tr = np.fmax(k_bar_tr, np.nan_to_num(scenario_tr, nan=0.0))
        r_bar = np.fmax(r_bar, np.nan_to_num(scenario_r, nan=0.0))
        margin_lower = np.minimum(margin_lower, np.where(bad_rows, np.nan, margins))
        event_scale_upper = np.maximum(event_scale_upper, event_scale)
        output_scale_upper = np.maximum(output_scale_upper, output_scale)
        k_additive_upper = np.maximum(k_additive_upper, k_add)
        if control_scale_lower.size:
            control_scale_lower = np.minimum(control_scale_lower, control_scale)
            control_delay_upper = np.maximum(control_delay_upper, control_delay)

    return _RobustArrays(
        k_bar_pk=k_bar_pk,
        k_bar_tr=k_bar_tr,
        r_bar=r_bar,
        margin_lower=margin_lower,
        event_scale_upper=event_scale_upper,
        output_scale_upper=output_scale_upper,
        k_additive_upper=k_additive_upper,
        control_scale_lower=control_scale_lower,
        control_delay_upper_s=control_delay_upper,
        data_limited_assets=tuple(sorted(data_limited)),
    )


def _robust_window_result(
    window: WindowMapResult,
    k_bar_pk: np.ndarray,
    k_bar_tr: np.ndarray,
    *,
    data_limited_assets: Sequence[str],
    status: AssessmentStatus,
    control_authority_lower: Sequence[TimeGridControlAuthority],
) -> WindowMapResult:
    protected = tuple(window.protected_asset_ids)
    events = tuple(window.event_ids)
    data_limited = set(data_limited_assets)
    assessments: list[ChannelAssessment] = []
    for asset in protected:
        for event in events:
            assessments.append(
                ChannelAssessment(
                    asset,
                    event,
                    AssessmentStatus.DATA_LIMITED
                    if asset in data_limited
                    else AssessmentStatus.CERTIFIED,
                    "robust_worst_case_bound",
                )
            )
    return WindowMapResult(
        mode_id=window.mode_id,
        protected_asset_ids=protected,
        event_ids=events,
        control_ids=tuple(window.control_ids),
        horizons_s=tuple(window.horizons_s),
        k_pickup_upper=_finite_matrix_for_window(k_bar_pk),
        k_trip_upper=_finite_matrix_for_window(k_bar_tr),
        control_authority_lower=tuple(control_authority_lower),
        channel_assessments=tuple(assessments),
        data_limited_assets=tuple(sorted(data_limited)),
        status=status,
        metadata={**dict(window.metadata or {}), "robust": True},
    )


def _robust_control_authority(
    window: WindowMapResult,
    control_scale_lower: np.ndarray,
    control_delay_upper_s: np.ndarray,
) -> tuple[TimeGridControlAuthority, ...]:
    if not window.control_authority_lower:
        return ()
    if control_scale_lower.size != len(window.control_ids):
        return tuple(window.control_authority_lower)
    result: list[TimeGridControlAuthority] = []
    for authority in window.control_authority_lower:
        matrix = np.asarray(authority.lower_bound_matrix, dtype=float)
        scaled = matrix * control_scale_lower[None, :]
        if control_delay_upper_s.size == len(window.control_ids):
            available = float(authority.time_s) >= control_delay_upper_s - 1e-12
            scaled = scaled * available[None, :]
        result.append(TimeGridControlAuthority(authority.time_s, _matrix_tuple(scaled)))
    return tuple(result)


def _robust_scores(
    k_bar_pk: np.ndarray,
    r_bar: np.ndarray,
    event_ids: Sequence[str],
    seed_event_ids: Sequence[str],
) -> np.ndarray:
    scores = r_bar.copy()
    event_index = {event: index for index, event in enumerate(event_ids)}
    for event_id in seed_event_ids:
        scores = scores + k_bar_pk[:, event_index[event_id]]
    return scores


def _seed_event_columns(
    protected_asset_ids: Sequence[str],
    event_ids: Sequence[str],
    seed_assets: Sequence[str],
    seed_event_ids: Sequence[str],
    asset_event_map: Mapping[str, str] | None,
) -> tuple[tuple[str, ...], set[str]]:
    event_set = set(event_ids)
    supplied = {str(key): str(value) for key, value in dict(asset_event_map or {}).items()}
    inferred: dict[str, str] = {}
    for asset in protected_asset_ids:
        if asset in supplied:
            inferred[asset] = supplied[asset]
        elif asset in event_set:
            inferred[asset] = asset
        elif f"trip_{asset}" in event_set:
            inferred[asset] = f"trip_{asset}"
    columns: list[str] = list(seed_event_ids)
    excluded = set(seed_assets)
    for asset in seed_assets:
        event_id = inferred.get(asset)
        if event_id is None:
            raise RobustScreeningError(f"no event column for seed asset {asset!r}")
        if event_id not in event_set:
            raise RobustScreeningError(f"asset_event_map references unknown event {event_id!r}")
        columns.append(event_id)
    inverse = {event: asset for asset, event in inferred.items()}
    for event in seed_event_ids:
        if event in inverse:
            excluded.add(inverse[event])
    return tuple(dict.fromkeys(columns)), excluded


def _monte_carlo_summary(
    *,
    sample_count: int,
    rng_seed: int | None,
    protected_asset_ids: Sequence[str],
    event_ids: Sequence[str],
    nominal_margins: np.ndarray,
    k_pk_nom: np.ndarray,
    k_tr_nom: np.ndarray,
    base_erosion: np.ndarray,
    worst_margin: np.ndarray,
    worst_event_scale: np.ndarray,
    worst_output_scale: np.ndarray,
    worst_k_additive: np.ndarray,
    seed_event_ids: Sequence[str],
    excluded_assets: set[str],
    epsilon: float,
) -> MonteCarloSummary:
    count = int(sample_count)
    if count <= 0:
        raise RobustScreeningError("monte_carlo_samples must be positive when requested")
    rng = np.random.default_rng(rng_seed)
    p = len(protected_asset_ids)
    q = len(event_ids)
    max_pk = np.zeros((p, q), dtype=float)
    max_tr = np.zeros((p, q), dtype=float)
    max_r = np.zeros(p, dtype=float)
    no_pickup_count = 0
    event_index = {event: index for index, event in enumerate(event_ids)}

    for _ in range(count):
        margin = nominal_margins - rng.random(p) * np.maximum(nominal_margins - worst_margin, 0.0)
        event_scale = 1.0 + rng.random(q) * np.maximum(worst_event_scale - 1.0, 0.0)
        output_scale = 1.0 + rng.random(p) * np.maximum(worst_output_scale - 1.0, 0.0)
        k_add = rng.random(p) * np.maximum(worst_k_additive, 0.0)
        valid = margin > 0.0
        pk = np.zeros((p, q), dtype=float)
        tr = np.zeros((p, q), dtype=float)
        r = np.zeros(p, dtype=float)
        if np.any(valid):
            scale = (
                nominal_margins[valid, None]
                / margin[valid, None]
                * output_scale[valid, None]
                * event_scale[None, :]
            )
            pk[valid, :] = k_pk_nom[valid, :] * scale + k_add[valid, None] / margin[
                valid, None
            ]
            tr[valid, :] = k_tr_nom[valid, :] * scale + k_add[valid, None] / margin[
                valid, None
            ]
            r[valid] = (
                base_erosion[valid] * nominal_margins[valid]
                + np.maximum(nominal_margins[valid] - margin[valid], 0.0)
            ) / margin[valid]
        max_pk = np.maximum(max_pk, pk)
        max_tr = np.maximum(max_tr, tr)
        max_r = np.maximum(max_r, r)
        scores = r.copy()
        for event in seed_event_ids:
            scores += pk[:, event_index[event]]
        safe = all(
            asset in excluded_assets or scores[index] <= 1.0 - epsilon + 1e-12
            for index, asset in enumerate(protected_asset_ids)
        )
        no_pickup_count += int(safe)

    return MonteCarloSummary(
        sample_count=count,
        max_k_pk_sampled=_matrix_tuple(max_pk),
        max_k_tr_sampled=_matrix_tuple(max_tr),
        max_r_sampled=tuple(float(item) for item in max_r),
        no_pickup_fraction=float(no_pickup_count / count),
        rng_seed=rng_seed,
    )


def _nominal_margins(
    protected_asset_ids: Sequence[str],
    protected_outputs: Sequence[ProtectedOutputEvaluation] | None,
    margins_pu: Sequence[float] | None,
) -> tuple[np.ndarray, tuple[str, ...]]:
    if margins_pu is not None:
        margins = np.asarray(margins_pu, dtype=float).reshape(-1)
        data_limited: list[str] = []
    elif protected_outputs is not None:
        by_asset = {item.asset_id: item for item in protected_outputs}
        if set(by_asset) != set(protected_asset_ids):
            raise RobustScreeningError("protected_outputs asset IDs must match protected assets")
        margins = np.asarray(
            [by_asset[asset].worst_case_margin_pu for asset in protected_asset_ids], dtype=float
        )
        data_limited = [
            asset
            for asset in protected_asset_ids
            if not by_asset[asset].is_certifiable or by_asset[asset].worst_case_margin_pu <= 0.0
        ]
    else:
        margins = np.ones(len(protected_asset_ids), dtype=float)
        data_limited = []
    if margins.size != len(protected_asset_ids):
        raise RobustScreeningError(
            f"margins_pu has length {margins.size}, expected {len(protected_asset_ids)}"
        )
    if not np.all(np.isfinite(margins)):
        raise RobustScreeningError("margins must be finite")
    for index, asset in enumerate(protected_asset_ids):
        if margins[index] <= 0.0:
            data_limited.append(asset)
    return margins, tuple(sorted(set(data_limited)))


def _matrix(value: Sequence[Sequence[float]], shape: tuple[int, int], name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.shape != shape:
        raise RobustScreeningError(f"{name} has shape {array.shape}, expected {shape}")
    if not np.all(np.isfinite(array)) or np.any(array < 0.0):
        raise RobustScreeningError(f"{name} must be finite and nonnegative")
    return array


def _base_erosion(base_erosion: Sequence[float] | None, count: int) -> np.ndarray:
    if base_erosion is None:
        return np.zeros(count, dtype=float)
    array = np.asarray(base_erosion, dtype=float).reshape(-1)
    if array.size != count:
        raise RobustScreeningError(f"base_erosion has length {array.size}, expected {count}")
    if not np.all(np.isfinite(array)) or np.any(array < 0.0):
        raise RobustScreeningError("base_erosion must be finite and nonnegative")
    return array


def _event_nominal_values(
    event_ids: Sequence[str],
    candidate_events: Sequence[CandidateEvent] | None,
) -> dict[str, float]:
    values = {event: 1.0 for event in event_ids}
    for event in candidate_events or ():
        if event.event_id not in values:
            continue
        if event.voltage_raising_q_mvar is not None:
            values[event.event_id] = max(float(event.voltage_raising_q_mvar.midpoint), 1e-12)
        elif event.delta_q_mvar is not None:
            values[event.event_id] = max(abs(float(event.delta_q_mvar.midpoint)), 1e-12)
        elif event.delta_p_mw is not None:
            values[event.event_id] = max(abs(float(event.delta_p_mw.midpoint)), 1e-12)
    return values


def _event_nominal(target_id: str, event_nominals: Mapping[str, float]) -> float:
    return float(event_nominals.get(target_id, 1.0))


def _quantity_scale(upper: float, nominal: float) -> float:
    upper_value = float(upper)
    nominal_value = float(nominal)
    if not math.isfinite(upper_value) or upper_value < 0.0:
        raise RobustScreeningError("uncertain quantity upper bound must be finite and nonnegative")
    if not math.isfinite(nominal_value) or nominal_value <= 0.0:
        return 1.0 if upper_value == 0.0 else math.inf
    return max(upper_value / nominal_value, 1.0)


def _fixed_pf_scale(
    target_id: str,
    parameter: str,
    interval: Any,
    candidate_events: Sequence[CandidateEvent] | None,
) -> float:
    parameter_l = str(parameter).lower()
    nominal_kappa: float | None = None
    for event in candidate_events or ():
        if event.event_id == str(target_id) and event.fixed_pf is not None:
            nominal_kappa = event.fixed_pf.nominal_kappa
            break
    if "kappa" in parameter_l:
        reference = nominal_kappa if nominal_kappa is not None else float(interval.midpoint)
        return _quantity_scale(float(interval.upper), reference)

    pf_worst = float(interval.lower)
    if pf_worst <= 0.0 or pf_worst > 1.0:
        raise RobustScreeningError("fixed-PF magnitude bounds must lie in (0, 1]")
    kappa_worst = math.tan(math.acos(pf_worst))
    reference = nominal_kappa
    if reference is None:
        pf_nominal = float(interval.nominal) if interval.nominal is not None else float(interval.midpoint)
        if pf_nominal <= 0.0 or pf_nominal > 1.0:
            raise RobustScreeningError("fixed-PF nominal value must lie in (0, 1]")
        reference = math.tan(math.acos(pf_nominal))
    if reference <= 0.0:
        return 1.0 if kappa_worst == 0.0 else math.inf
    return max(kappa_worst / reference, 1.0)


def _tap_scale(lower: float, nominal: float | None, upper: float) -> float:
    lower_value = float(lower)
    upper_value = float(upper)
    if lower_value <= 0.0:
        return math.inf
    reference = float(nominal) if nominal is not None else upper_value
    if reference <= 0.0:
        return math.inf
    return max(reference / lower_value, 1.0)


def _factor_vector(
    value: Mapping[str, float] | Sequence[float] | None,
    ids: Sequence[str],
    count: int,
    *,
    default: np.ndarray,
    name: str,
) -> np.ndarray:
    if count == 0:
        return np.zeros(0, dtype=float)
    if value is None:
        return np.asarray(default, dtype=float).copy()
    if isinstance(value, Mapping):
        result = np.asarray(default, dtype=float).copy()
        id_index = {item: index for index, item in enumerate(ids)}
        for key, item in value.items():
            key_s = str(key)
            if key_s not in id_index:
                continue
            result[id_index[key_s]] = float(item)
    else:
        result = np.asarray(value, dtype=float).reshape(-1)
        if result.size != count:
            raise RobustScreeningError(f"{name} has length {result.size}, expected {count}")
    if not np.all(np.isfinite(result)):
        raise RobustScreeningError(f"{name} must be finite")
    return result


def _matching_assets(
    target_id: str,
    protected_asset_ids: Sequence[str],
    protected_outputs: Sequence[ProtectedOutputEvaluation] | None,
    *,
    loose: bool = False,
) -> tuple[str, ...]:
    target = str(target_id)
    direct = [asset for asset in protected_asset_ids if asset == target]
    if direct:
        return tuple(direct)
    if protected_outputs is not None:
        matches = [
            item.asset_id
            for item in protected_outputs
            if item.output_id == target or item.source_bus_id == target
        ]
        if matches:
            return tuple(matches)
    if loose:
        loose_matches = [asset for asset in protected_asset_ids if target in asset or asset in target]
        return tuple(loose_matches)
    return ()


def _matching_events(
    target_id: str,
    event_ids: Sequence[str],
    candidate_events: Sequence[CandidateEvent] | None,
) -> tuple[str, ...]:
    target = str(target_id)
    direct = [event for event in event_ids if event == target]
    if direct:
        return tuple(direct)
    matches: list[str] = []
    for event in candidate_events or ():
        if event.event_id not in event_ids:
            continue
        if any(asset.asset_id == target for asset in event.affected_assets):
            matches.append(event.event_id)
    return tuple(matches)


def _protected_threshold_lower(
    asset_id: str,
    protected_outputs: Sequence[ProtectedOutputEvaluation] | None,
) -> float | None:
    for item in protected_outputs or ():
        if item.asset_id == asset_id:
            return float(item.threshold_pu.lower)
    return None


def _asset_index(asset_id: str, protected_asset_ids: Sequence[str]) -> int:
    return tuple(protected_asset_ids).index(asset_id)


def _single_margin(
    asset_id: str,
    protected_asset_ids: Sequence[str],
    margins: np.ndarray | None,
    cumulative_loss: float,
) -> dict[str, float] | None:
    if margins is None:
        return None
    index = _asset_index(asset_id, protected_asset_ids)
    return {asset_id: float(margins[index] - cumulative_loss)}


def _checked_ids(ids: Sequence[str], allowed: Sequence[str], name: str) -> tuple[str, ...]:
    allowed_set = set(allowed)
    result = tuple(str(item) for item in ids)
    unknown = [item for item in result if item not in allowed_set]
    if unknown:
        raise RobustScreeningError(f"{name} contains unknown IDs: {', '.join(unknown)}")
    if len(set(result)) != len(result):
        raise RobustScreeningError(f"{name} contains duplicates")
    return result


def _epsilon(value: float) -> float:
    epsilon = float(value)
    if not math.isfinite(epsilon) or epsilon < 0.0 or epsilon >= 1.0:
        raise RobustScreeningError("epsilon must be finite and in [0, 1)")
    return epsilon


def _finite_matrix_for_window(array: np.ndarray) -> Matrix:
    finite = np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)
    finite = np.maximum(finite, 0.0)
    return _matrix_tuple(finite)


def _matrix_tuple(array: np.ndarray) -> Matrix:
    return tuple(tuple(float(item) for item in row) for row in np.asarray(array, dtype=float))


def _array_to_nested(array: np.ndarray, *, nonfinite_as_none: bool = False) -> list[list[float | None]]:
    rows: list[list[float | None]] = []
    for row in np.asarray(array, dtype=float):
        rows.append(_array_to_list(row, nonfinite_as_none=nonfinite_as_none))
    return rows


def _array_to_list(array: np.ndarray, *, nonfinite_as_none: bool = False) -> list[float | None]:
    result: list[float | None] = []
    for value in np.asarray(array, dtype=float).reshape(-1):
        if nonfinite_as_none and not math.isfinite(float(value)):
            result.append(None)
        else:
            result.append(float(value))
    return result


def _serializable_factor(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, Mapping):
        return {str(key): float(item) for key, item in value.items()}
    return [float(item) for item in value]
