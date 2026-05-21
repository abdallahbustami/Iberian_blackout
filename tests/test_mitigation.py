from __future__ import annotations

import unittest

import numpy as np

from pa_dvsa.data_model import (
    AssessmentStatus,
    ProtectedVoltageKind,
    TimeGridControlAuthority,
    Unit,
    WindowMapResult,
)
from pa_dvsa.finite_window import ProfileResponse
from pa_dvsa.mitigation import DisturbanceMarginProfile, MitigationError, solve_mitigation_lp
from pa_dvsa.protected_outputs import ProtectedOutputEvaluation, ScalarEnvelope


def _window(
    *,
    controls: tuple[str, ...] = ("C1",),
    authority: tuple[tuple[float, ...], ...] = ((0.0,), (0.5,)),
    time_s: float = 0.5,
) -> WindowMapResult:
    return WindowMapResult(
        mode_id="m0",
        protected_asset_ids=("A", "B"),
        event_ids=("A", "B"),
        control_ids=controls,
        horizons_s=(1.0, 1.0),
        k_pickup_upper=((0.0, 0.0), (0.0, 0.0)),
        k_trip_upper=((0.0, 0.0), (0.0, 0.0)),
        control_authority_lower=(TimeGridControlAuthority(time_s, authority),),
        status=AssessmentStatus.CERTIFIED,
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


class MitigationLPTests(unittest.TestCase):
    def test_lp_selects_minimum_action_and_required_mvar(self) -> None:
        profile = DisturbanceMarginProfile((0.5,), ((0.0, 1.2),))

        details = solve_mitigation_lp(
            _window(),
            seed_ids=("A",),
            disturbance_profile=profile,
            control_mvar={"C1": 100.0},
        )
        result = details.mitigation_result

        self.assertTrue(result.feasible)
        self.assertAlmostEqual(result.slack_eta, 0.0)
        self.assertEqual(result.status, AssessmentStatus.CERTIFIED)
        self.assertEqual(len(result.selections), 1)
        self.assertAlmostEqual(result.selections[0].alpha, 0.4)
        self.assertAlmostEqual(result.selections[0].magnitude_mvar, 40.0)
        self.assertAlmostEqual(result.required_mvar_total, 40.0)
        self.assertEqual(result.binding_constraints[0].protected_asset_id, "B")

    def test_lp_reports_residual_slack_when_authority_is_insufficient(self) -> None:
        profile = DisturbanceMarginProfile((0.5,), ((0.0, 1.5),))

        details = solve_mitigation_lp(
            _window(authority=((0.0,), (0.2,))),
            seed_ids=("A",),
            disturbance_profile=profile,
            alpha_upper={"C1": 1.0},
        )
        result = details.mitigation_result

        self.assertFalse(result.feasible)
        self.assertEqual(result.status, AssessmentStatus.FAILED)
        self.assertAlmostEqual(result.selections[0].alpha, 1.0)
        self.assertTrue(result.selections[0].saturated)
        self.assertAlmostEqual(result.slack_eta, 0.3)

    def test_requires_time_resolved_control_authority(self) -> None:
        window = WindowMapResult(
            mode_id="m0",
            protected_asset_ids=("A",),
            event_ids=("A",),
            control_ids=("C1",),
            horizons_s=(1.0,),
            k_pickup_upper=((0.0,),),
            k_trip_upper=((0.0,),),
            status=AssessmentStatus.CERTIFIED,
        )

        with self.assertRaises(MitigationError):
            solve_mitigation_lp(window, disturbance_profile=DisturbanceMarginProfile((0.0,), ((0.0,),)))

    def test_disturbance_response_is_normalized_by_margins(self) -> None:
        response = ProfileResponse(
            times_s=(0.0, 0.5),
            values_pu=np.asarray([[[0.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.6, 0.0]]]),
            derivatives_pu_per_s=np.zeros((2, 2, 2)),
            channel_ids=("A", "B"),
            profile_kinds=("step", "step"),
            channel_type="disturbance",
        )

        details = solve_mitigation_lp(
            _window(),
            seed_ids=("A",),
            disturbance_response=response,
            protected_outputs=(_protected("A"), _protected("B")),
        )

        self.assertAlmostEqual(details.disturbance_erosion[0][1], 1.2)
        self.assertAlmostEqual(details.mitigation_result.selections[0].alpha, 0.4)

    def test_seed_assets_infer_disturbance_response_event_columns(self) -> None:
        response = ProfileResponse(
            times_s=(0.0, 0.5),
            values_pu=np.asarray([[[0.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.6, 0.0]]]),
            derivatives_pu_per_s=np.zeros((2, 2, 2)),
            channel_ids=("A", "B"),
            profile_kinds=("step", "step"),
            channel_type="disturbance",
        )

        details = solve_mitigation_lp(
            _window(),
            seed_assets=("A",),
            disturbance_response=response,
            protected_outputs=(_protected("A"), _protected("B")),
        )

        self.assertAlmostEqual(details.disturbance_erosion[0][1], 1.2)

    def test_unavailable_and_ineffective_controls_are_reported(self) -> None:
        profile = DisturbanceMarginProfile((0.5,), ((0.0, 1.2),))
        window = _window(
            controls=("C1", "C2", "C3"),
            authority=((0.0, 0.0, 0.0), (0.5, 0.0, 0.2)),
        )

        details = solve_mitigation_lp(
            window,
            seed_ids=("A",),
            disturbance_profile=profile,
            unavailable_controls=("C3",),
        )
        result = details.mitigation_result

        self.assertEqual(result.unavailable_controls, ("C3",))
        self.assertEqual(result.ineffective_controls, ("C2",))
        self.assertEqual(details.unavailable_controls, ("C3",))
        self.assertEqual(details.ineffective_controls, ("C2",))


if __name__ == "__main__":
    unittest.main()
