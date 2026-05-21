from __future__ import annotations

import unittest

import numpy as np

from pa_dvsa.cascade_certificate import compute_cascade_certificate, phi_m
from pa_dvsa.data_model import (
    AssessmentStatus,
    ProtectedVoltageKind,
    Unit,
    WindowMapResult,
)
from pa_dvsa.finite_window import ProfileResponse
from pa_dvsa.protected_outputs import ProtectedOutputEvaluation, ScalarEnvelope


def _window(
    *,
    protected: tuple[str, ...] = ("A", "B", "C"),
    events: tuple[str, ...] = ("A", "B", "C"),
    k_pickup: tuple[tuple[float, ...], ...] = (
        (0.0, 0.0, 0.0),
        (1.1, 0.0, 0.0),
        (0.0, 1.0, 0.0),
    ),
    k_trip: tuple[tuple[float, ...], ...] | None = None,
    data_limited: tuple[str, ...] = (),
) -> WindowMapResult:
    return WindowMapResult(
        mode_id="m0",
        protected_asset_ids=protected,
        event_ids=events,
        control_ids=(),
        horizons_s=tuple(1.0 for _ in protected),
        k_pickup_upper=k_pickup,
        k_trip_upper=k_trip
        if k_trip is not None
        else tuple(tuple(0.0 for _ in events) for _ in protected),
        data_limited_assets=data_limited,
        status=AssessmentStatus.DATA_LIMITED if data_limited else AssessmentStatus.CERTIFIED,
    )


def _protected(asset_id: str, margin: float = 0.5) -> ProtectedOutputEvaluation:
    return ProtectedOutputEvaluation(
        asset_id=asset_id,
        output_id=f"z_{asset_id}",
        kind=ProtectedVoltageKind.DIRECT_BUS_VOLTAGE,
        z_pu=ScalarEnvelope.exact(1.0, Unit.PU),
        threshold_pu=ScalarEnvelope.exact(1.0 + margin, Unit.PU),
        margin_pu=ScalarEnvelope.exact(margin, Unit.PU),
        status=AssessmentStatus.CERTIFIED,
        worst_case_margin_pu=margin,
        is_certifiable=True,
        source_bus_id=asset_id,
        source_address=None,
        reason="test",
    )


class CascadeCertificateTests(unittest.TestCase):
    def test_phi_m_implements_additive_threshold_map(self) -> None:
        window = _window()

        first = phi_m(window, ("A",))
        second = phi_m(window, first)

        self.assertEqual(first, ("A", "B"))
        self.assertEqual(second, ("A", "B", "C"))

    def test_phi_m_includes_base_erosion_term(self) -> None:
        window = _window(
            protected=("A", "B"),
            events=("A", "B"),
            k_pickup=((0.0, 0.0), (0.25, 0.0)),
        )

        self.assertEqual(phi_m(window, ("A",), base_erosion=(0.0, 0.74)), ("A",))
        self.assertEqual(phi_m(window, ("A",), base_erosion=(0.0, 0.75)), ("A", "B"))

    def test_iterates_to_least_fixed_point_and_stores_layers(self) -> None:
        result = compute_cascade_certificate(_window(), seed_ids=("A",))
        cascade = result.seed_results[0]

        self.assertEqual(cascade.fixed_point, ("A", "B", "C"))
        self.assertEqual(cascade.layers[0].newly_picked_up, ("A",))
        self.assertEqual(cascade.layers[1].newly_picked_up, ("B",))
        self.assertEqual(cascade.layers[2].newly_picked_up, ("C",))
        self.assertFalse(cascade.no_secondary_certified)
        self.assertEqual(result.seed_rankings[0].first_secondary_pickup, ("B",))
        self.assertEqual(result.seed_rankings[0].final_fixed_point_size, 3)

    def test_pickup_prediction_uses_k_pk_not_k_trip(self) -> None:
        window = _window(
            protected=("A", "B"),
            events=("A", "B"),
            k_pickup=((0.0, 0.0), (1.1, 0.0)),
            k_trip=((0.0, 0.0), (0.0, 0.0)),
        )

        result = compute_cascade_certificate(window, seed_ids=("A",))

        self.assertEqual(result.seed_results[0].fixed_point, ("A", "B"))
        self.assertEqual(result.seed_results[0].layers[1].newly_picked_up, ("B",))

    def test_exogenous_seed_event_forms_initial_pickup_layer(self) -> None:
        window = _window(
            protected=("A", "B"),
            events=("E", "A", "B"),
            k_pickup=((1.2, 0.0, 0.0), (0.0, 1.1, 0.0)),
        )

        result = compute_cascade_certificate(
            window,
            seed_ids=("E",),
            asset_event_map={"A": "A", "B": "B"},
        )

        self.assertEqual(result.seed_results[0].layers[0].newly_picked_up, ("A",))
        self.assertEqual(result.seed_results[0].layers[1].newly_picked_up, ("B",))
        self.assertEqual(result.seed_rankings[0].first_secondary_pickup, ("A",))

    def test_combined_waveform_confirms_delayed_trip_with_dwell_logic(self) -> None:
        window = _window(
            protected=("A", "B"),
            events=("A", "B"),
            k_pickup=((0.0, 0.0), (1.1, 0.0)),
        )
        response = ProfileResponse(
            times_s=(0.0, 0.5, 1.0),
            values_pu=np.asarray(
                [
                    [[0.0, 0.0], [0.6, 0.0]],
                    [[0.0, 0.0], [0.6, 0.0]],
                    [[0.0, 0.0], [0.6, 0.0]],
                ],
                dtype=float,
            ),
            derivatives_pu_per_s=np.zeros((3, 2, 2)),
            channel_ids=("A", "B"),
            profile_kinds=("step", "step"),
            channel_type="disturbance",
        )

        result = compute_cascade_certificate(
            window,
            seed_ids=("A",),
            finite_response=response,
            protected_outputs=(_protected("A"), _protected("B")),
            dwell_times_s=(0.0, 0.5),
        )

        confirmations = result.delayed_trip_confirmations
        self.assertEqual(len(confirmations), 1)
        self.assertEqual(confirmations[0].asset_id, "B")
        self.assertTrue(confirmations[0].confirmed)
        self.assertAlmostEqual(confirmations[0].trip_erosion, 1.2)
        self.assertEqual(result.seed_rankings[0].delayed_trip_confirmed, ("B",))

    def test_data_limited_assets_are_reported_in_seed_ranking(self) -> None:
        window = _window(data_limited=("C",))

        result = compute_cascade_certificate(window, seed_ids=("A",))

        self.assertEqual(result.status, AssessmentStatus.DATA_LIMITED)
        self.assertEqual(result.seed_rankings[0].data_limited_assets, ("C",))
        self.assertEqual(result.seed_results[0].status, AssessmentStatus.DATA_LIMITED)

    def test_positive_dwell_without_combined_waveform_is_data_limited(self) -> None:
        window = _window(
            protected=("A", "B"),
            events=("A", "B"),
            k_pickup=((0.0, 0.0), (1.1, 0.0)),
        )

        result = compute_cascade_certificate(window, seed_ids=("A",), dwell_times_s=(0.0, 0.5))

        self.assertEqual(result.status, AssessmentStatus.DATA_LIMITED)
        self.assertEqual(result.seed_rankings[0].data_limited_assets, ("B",))


if __name__ == "__main__":
    unittest.main()
