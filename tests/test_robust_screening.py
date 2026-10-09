"""Tests of the robust screening."""

from __future__ import annotations

import math
import unittest

from pa_dvsa.data_model import (
    AssessmentStatus,
    AssetRef,
    CandidateEvent,
    ControllerResponseUncertainty,
    EventKind,
    FixedPowerFactorSpec,
    Interval,
    ParameterBound,
    ProtectedVoltageKind,
    TimeGridControlAuthority,
    Unit,
    UncertaintySet,
    WindowMapResult,
)
from pa_dvsa.protected_outputs import ProtectedOutputEvaluation, ScalarEnvelope
from pa_dvsa.robust_screening import RobustScenario, compute_robust_screen


def _window(
    *,
    protected: tuple[str, ...] = ("A", "B"),
    events: tuple[str, ...] = ("A", "B"),
    k_pk: tuple[tuple[float, ...], ...] = ((0.0, 0.0), (0.4, 0.0)),
    k_tr: tuple[tuple[float, ...], ...] | None = None,
    controls: tuple[str, ...] = (),
) -> WindowMapResult:
    return WindowMapResult(
        mode_id="m0",
        protected_asset_ids=protected,
        event_ids=events,
        control_ids=controls,
        horizons_s=tuple(1.0 for _ in protected),
        k_pickup_upper=k_pk,
        k_trip_upper=k_tr if k_tr is not None else k_pk,
        control_authority_lower=(
            (TimeGridControlAuthority(0.5, ((1.0,), (1.0,))),)
            if controls
            else ()
        ),
        status=AssessmentStatus.CERTIFIED,
    )


def _protected(asset_id: str, margin: float = 0.2) -> ProtectedOutputEvaluation:
    return ProtectedOutputEvaluation(
        asset_id=asset_id,
        output_id=f"z_{asset_id}",
        kind=ProtectedVoltageKind.DIRECT_BUS_VOLTAGE,
        z_pu=ScalarEnvelope.exact(1.0, Unit.PU),
        threshold_pu=ScalarEnvelope.exact(1.0 + margin, Unit.PU),
        margin_pu=ScalarEnvelope.exact(margin, Unit.PU),
        status=AssessmentStatus.CERTIFIED if margin > 0 else AssessmentStatus.DATA_LIMITED,
        worst_case_margin_pu=margin,
        is_certifiable=margin > 0,
        source_bus_id=asset_id,
        source_address=None,
        reason="test",
    )


class RobustScreeningTests(unittest.TestCase):
    def test_threshold_uncertainty_rescales_k_and_r_bar(self) -> None:
        uncertainty = UncertaintySet(
            "theta",
            thresholds_pu=(
                ParameterBound("B", "V_trip", Interval(1.18, 1.20, Unit.PU, nominal=1.20)),
            ),
        )

        result = compute_robust_screen(
            _window(),
            protected_outputs=(_protected("A"), _protected("B")),
            uncertainty_set=uncertainty,
            seed_assets=("A",),
        )

        self.assertAlmostEqual(result.margin_lower_pu[1], 0.18)
        self.assertAlmostEqual(result.k_bar_pk[1, 0], 0.4 * 0.2 / 0.18)
        self.assertAlmostEqual(result.r_bar[1], (0.2 - 0.18) / 0.18)
        self.assertTrue(result.robust_no_pickup_certified)

    def test_robust_no_pickup_certificate_respects_epsilon(self) -> None:
        safe = compute_robust_screen(
            _window(k_pk=((0.0, 0.0), (0.79, 0.0))),
            margins_pu=(1.0, 1.0),
            seed_assets=("A",),
            base_erosion=(0.0, 0.1),
            epsilon=0.1,
        )
        unsafe = compute_robust_screen(
            _window(k_pk=((0.0, 0.0), (0.81, 0.0))),
            margins_pu=(1.0, 1.0),
            seed_assets=("A",),
            base_erosion=(0.0, 0.1),
            epsilon=0.1,
        )

        self.assertTrue(safe.robust_no_pickup_certified)
        self.assertFalse(unsafe.robust_no_pickup_certified)
        self.assertEqual(unsafe.violating_assets, ("B",))
        self.assertEqual(unsafe.status, AssessmentStatus.FAILED)

    def test_nonpositive_uncertain_margin_is_data_limited_not_safe(self) -> None:
        uncertainty = UncertaintySet(
            "theta_bad",
            thresholds_pu=(
                ParameterBound("B", "V_trip", Interval(0.95, 1.20, Unit.PU, nominal=1.20)),
            ),
        )

        result = compute_robust_screen(
            _window(),
            protected_outputs=(_protected("A"), _protected("B")),
            uncertainty_set=uncertainty,
            seed_assets=("A",),
        )

        self.assertFalse(result.robust_no_pickup_certified)
        self.assertEqual(result.status, AssessmentStatus.DATA_LIMITED)
        self.assertEqual(result.data_limited_assets, ("B",))
        self.assertTrue(math.isnan(result.k_bar_pk[1, 0]))

    def test_q_absorption_and_fixed_pf_intervals_scale_event_columns(self) -> None:
        event = CandidateEvent(
            event_id="A",
            kind=EventKind.GENERATOR_TRIP,
            name="Trip A",
            affected_assets=(AssetRef("G1"),),
            voltage_raising_q_mvar=Interval.exact(100.0, Unit.MVAR),
            fixed_pf=FixedPowerFactorSpec(Interval.exact(0.95), reactive_sign=1),
        )
        uncertainty = UncertaintySet(
            "theta_event",
            q_absorption_mvar=(
                ParameterBound("A", "Q_abs", Interval(100.0, 150.0, Unit.MVAR, nominal=100.0)),
            ),
            fixed_pf_values=(
                ParameterBound("A", "pf_abs", Interval(0.90, 0.95, Unit.NONE, nominal=0.95)),
            ),
        )

        result = compute_robust_screen(
            _window(k_pk=((0.0, 0.0), (0.2, 0.0))),
            protected_outputs=(_protected("A"), _protected("B")),
            uncertainty_set=uncertainty,
            candidate_events=(event,),
            seed_assets=("A",),
        )

        self.assertGreater(result.k_bar_pk[1, 0], 0.2 * 1.49)
        self.assertIn("fixed_pf_values", result.scenarios[0].metadata["categories"])

    def test_delay_uncertainty_uses_pickup_bound_for_trip_bound(self) -> None:
        uncertainty = UncertaintySet(
            "theta_delay",
            delays_s=(ParameterBound("B", "delay", Interval(0.0, 0.5, Unit.SECOND)),),
        )

        result = compute_robust_screen(
            _window(k_pk=((0.0, 0.0), (0.9, 0.0)), k_tr=((0.0, 0.0), (0.1, 0.0))),
            protected_outputs=(_protected("A"), _protected("B")),
            uncertainty_set=uncertainty,
            seed_assets=("A",),
        )

        self.assertAlmostEqual(result.k_bar_tr[1, 0], 0.9)

    def test_tap_and_shunt_status_uncertainties_scale_rows_and_columns(self) -> None:
        uncertainty = UncertaintySet(
            "theta_tap_shunt",
            tap_ratios=(ParameterBound("B", "n", Interval(0.9, 1.0, Unit.PU, nominal=1.0)),),
            shunt_status=(
                ParameterBound("A", "status_scale", Interval(1.0, 2.0, Unit.NONE, nominal=1.0)),
            ),
        )

        result = compute_robust_screen(
            _window(k_pk=((0.0, 0.0), (0.3, 0.0))),
            protected_outputs=(_protected("A"), _protected("B")),
            uncertainty_set=uncertainty,
            seed_assets=("A",),
        )

        self.assertAlmostEqual(result.k_bar_pk[1, 0], 0.3 * (1.0 / 0.9) * 2.0)

    def test_controller_timing_and_scale_lower_control_authority(self) -> None:
        uncertainty = UncertaintySet(
            "theta_control",
            controller_responses=(
                ControllerResponseUncertainty(
                    "C1",
                    delay_s=Interval(0.7, 0.8, Unit.SECOND),
                    lower_bound_scale=Interval(0.5, 1.0),
                ),
            ),
        )

        result = compute_robust_screen(
            _window(controls=("C1",)),
            protected_outputs=(_protected("A"), _protected("B")),
            uncertainty_set=uncertainty,
        )

        self.assertEqual(result.control_authority_lower[0].lower_bound_matrix[0][0], 0.0)

    def test_monte_carlo_is_supporting_evidence_only(self) -> None:
        result = compute_robust_screen(
            _window(k_pk=((0.0, 0.0), (0.81, 0.0))),
            margins_pu=(1.0, 1.0),
            seed_assets=("A",),
            base_erosion=(0.0, 0.1),
            epsilon=0.1,
            monte_carlo_samples=5,
            rng_seed=7,
        )

        self.assertFalse(result.robust_no_pickup_certified)
        self.assertIsNotNone(result.monte_carlo)
        self.assertEqual(result.monte_carlo.sample_count, 5)

    def test_explicit_scenario_sweep_is_used_for_worst_case(self) -> None:
        scenario = RobustScenario("stress", event_scale={"A": 2.0})

        result = compute_robust_screen(
            _window(k_pk=((0.0, 0.0), (0.3, 0.0))),
            margins_pu=(1.0, 1.0),
            scenarios=(scenario,),
            seed_assets=("A",),
        )

        self.assertAlmostEqual(result.k_bar_pk[1, 0], 0.6)


class ControlLowerAuthorityVectorizationTests(unittest.TestCase):
    def test_vectorized_equals_reference(self) -> None:
        from types import SimpleNamespace

        import numpy as np

        from pa_dvsa.robust_screening import (
            UncertaintyIntervals,
            _control_lower_authority_reference,
            control_lower_authority,
        )

        rng = np.random.default_rng(3)
        times = np.round(np.arange(0.0, 1.0 + 1e-9, 0.005), 12)
        nb = times.size - 1
        assets = [
            SimpleNamespace(data_limited=False, response_scale_lower=0.95, response_scale_upper=1.05, h_lower_pu=0.02),
            SimpleNamespace(data_limited=True, response_scale_lower=1.0, response_scale_upper=1.0, h_lower_pu=-0.01),
            SimpleNamespace(data_limited=False, response_scale_lower=1.0, response_scale_upper=1.0, h_lower_pu=0.008),
        ]
        bounds = SimpleNamespace(
            asset_ids=("A", "B", "C"), assets=assets, uncertainty=None,
            output_models=[SimpleNamespace(output_index=k) for k in (2, 0, 1)],
        )
        responses = []
        for c in range(4):
            envs = [SimpleNamespace(upper=rng.normal(0.0, 0.01, nb) - 0.005 * (c % 2)) for _ in range(3)]
            responses.append(SimpleNamespace(event_id=f"ctrl{c}", times_s=times, envelopes=envs))
        for unc in (
            None,
            UncertaintyIntervals(control_gain_scale={"*": (0.8, 1.1)}, control_delay_s={"ctrl1": (0.0, 0.0123), "ctrl2": (0.01, 0.04)}),
            UncertaintyIntervals(control_delay_s={"*": (0.0, 1.5)}),
        ):
            new = control_lower_authority(bounds, responses, uncertainty=unc)
            ref = _control_lower_authority_reference(bounds, responses, uncertainty=unc)
            np.testing.assert_array_equal(np.isnan(new), np.isnan(ref))
            np.testing.assert_allclose(new, ref, rtol=0.0, atol=0.0, equal_nan=True)
            self.assertTrue(np.all(np.isnan(new[:, 1, :])))


if __name__ == "__main__":
    unittest.main()
