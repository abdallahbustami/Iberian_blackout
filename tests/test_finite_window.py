from __future__ import annotations

import math
import unittest

from scipy import sparse

from pa_dvsa.data_model import AssessmentStatus, ProfileKind, ProtectedVoltageKind, TimeProfile, Unit
from pa_dvsa.finite_window import (
    compute_finite_window_maps,
    profile_response,
    scalar_voltage_response,
    step_response_d,
)
from pa_dvsa.linearization import ReducedLinearModel, ReductionDiagnostics
from pa_dvsa.protected_outputs import ProtectedOutputEvaluation, ScalarEnvelope


def _protected(asset_id: str, margin: float, *, certifiable: bool = True) -> ProtectedOutputEvaluation:
    return ProtectedOutputEvaluation(
        asset_id=asset_id,
        output_id=f"z_{asset_id}",
        kind=ProtectedVoltageKind.DIRECT_BUS_VOLTAGE,
        z_pu=ScalarEnvelope.exact(1.0, Unit.PU),
        threshold_pu=ScalarEnvelope.exact(1.0 + margin, Unit.PU),
        margin_pu=ScalarEnvelope.exact(margin, Unit.PU),
        status=AssessmentStatus.CERTIFIED if certifiable else AssessmentStatus.DATA_LIMITED,
        worst_case_margin_pu=margin,
        is_certifiable=certifiable,
        source_bus_id=asset_id,
        source_address=None,
        reason="test_margin",
    )


def _reduced(
    *,
    ar: list[list[float]],
    dr: list[list[float]],
    cr: list[list[float]],
    fd: list[list[float]],
    br: list[list[float]] | None = None,
    fu: list[list[float]] | None = None,
) -> ReducedLinearModel:
    n = len(ar)
    p = len(cr)
    q = len(fd[0]) if fd else 0
    r = len(fu[0]) if fu else 0
    ar_m = sparse.csc_matrix((0, 0)) if n == 0 else sparse.csc_matrix(ar)
    dr_m = sparse.csc_matrix((0, q)) if n == 0 else sparse.csc_matrix(dr)
    cr_m = sparse.csc_matrix((p, 0)) if n == 0 else sparse.csc_matrix(cr)
    if n == 0:
        br_m = sparse.csc_matrix((0, r))
    else:
        br_m = sparse.csc_matrix(br if br is not None else [[0.0] * r for _ in range(n)])
    return ReducedLinearModel(
        ar=ar_m,
        br=br_m,
        dr=dr_m,
        cr=cr_m,
        fu=sparse.csc_matrix(fu if fu is not None else [[0.0] * r for _ in range(p)]),
        fd=sparse.csc_matrix(fd),
        diagnostics=ReductionDiagnostics(
            mode_id="finite_toy",
            n_states=n,
            n_algebraic=1,
            n_outputs=p,
            n_controls=r,
            n_disturbances=q,
            algebraic_only=(n == 0),
            gy_factorized=True,
            gy_min_abs_u_diag=1.0,
            max_real_eigenvalue=None,
            stability="stable" if n else "algebraic_only",
            warnings=(),
        ),
    )


class FiniteWindowTests(unittest.TestCase):
    def test_step_response_d_matches_analytic_formula(self) -> None:
        reduced = _reduced(
            ar=[[-2.0]],
            dr=[[1.0]],
            cr=[[3.0]],
            fd=[[0.5]],
        )

        response = step_response_d(reduced, [0.0, 0.5, 1.0])
        values = response.values_pu[:, 0, 0]

        for time, value in zip(response.times_s, values):
            expected = 0.5 + 3.0 * (1.0 - math.exp(-2.0 * time)) / 2.0
            self.assertAlmostEqual(value, expected, places=9)

    def test_ramp_profile_response_matches_integrator_solution(self) -> None:
        reduced = _reduced(
            ar=[[0.0]],
            dr=[[1.0]],
            cr=[[1.0]],
            fd=[[0.0]],
        )
        profile = TimeProfile(ProfileKind.RAMP, duration_s=2.0)

        response = profile_response(
            reduced,
            [0.0, 1.0, 2.0, 3.0],
            input_matrix=[[1.0]],
            profiles=(profile,),
            channel_type="disturbance",
        )

        self.assertAlmostEqual(response.values_pu[0, 0, 0], 0.0)
        self.assertAlmostEqual(response.values_pu[1, 0, 0], 0.25)
        self.assertAlmostEqual(response.values_pu[2, 0, 0], 1.0)
        self.assertAlmostEqual(response.values_pu[3, 0, 0], 2.0)

    def test_sampled_profile_response_is_piecewise_linear(self) -> None:
        reduced = _reduced(
            ar=[[0.0]],
            dr=[[1.0]],
            cr=[[1.0]],
            fd=[[0.0]],
        )
        profile = TimeProfile(
            ProfileKind.SAMPLED,
            samples=((0.0, 0.0), (1.0, 2.0), (2.0, 0.0)),
        )

        response = profile_response(
            reduced,
            [0.0, 1.0, 2.0],
            input_matrix=[[1.0]],
            profiles=(profile,),
            channel_type="disturbance",
        )

        self.assertAlmostEqual(response.values_pu[0, 0, 0], 0.0)
        self.assertAlmostEqual(response.values_pu[1, 0, 0], 1.0)
        self.assertAlmostEqual(response.values_pu[2, 0, 0], 2.0)

        times, scalar = scalar_voltage_response(response, output_index=0, channel_index=0)
        self.assertEqual(times, (0.0, 1.0, 2.0))
        self.assertEqual(scalar, (0.0, 1.0, 2.0))

    def test_finite_window_maps_compute_pickup_trip_and_monotone_class(self) -> None:
        reduced = _reduced(
            ar=[[-1.0]],
            dr=[[1.0]],
            cr=[[1.0]],
            fd=[[0.0]],
        )

        result = compute_finite_window_maps(
            reduced,
            [_protected("p1", 0.5)],
            event_ids=("e1",),
            horizons_s=(1.0,),
            dwell_times_s=(0.0,),
            time_grid_s=(0.0, 0.5, 1.0),
        )

        expected_peak = 1.0 - math.exp(-1.0)
        self.assertAlmostEqual(result.pickup_excursion_pu[0][0], expected_peak, places=9)
        self.assertAlmostEqual(result.trip_excursion_pu[0][0], expected_peak, places=9)
        self.assertAlmostEqual(result.window_result.k_pickup_upper[0][0], expected_peak / 0.5)
        self.assertEqual(result.channel_classes["p1:e1"], "monotone_proxy")
        self.assertEqual(result.window_result.status, AssessmentStatus.CERTIFIED)

    def test_dwell_trip_excursion_uses_moving_minimum(self) -> None:
        reduced = _reduced(
            ar=[],
            dr=[],
            cr=[[]],
            fd=[[1.0]],
        )
        pulse = TimeProfile(
            ProfileKind.SAMPLED,
            samples=((0.0, 0.0), (0.5, 1.0), (1.0, 1.0), (1.5, 0.0)),
        )

        result = compute_finite_window_maps(
            reduced,
            [_protected("p1", 0.5)],
            event_ids=("pulse",),
            event_profiles=(pulse,),
            horizons_s=(1.5,),
            dwell_times_s=(0.5,),
            time_grid_s=(0.0, 0.5, 1.0, 1.5),
            derivative_safety_factor=0.0,
            padding_tolerance_pu=10.0,
        )

        self.assertAlmostEqual(result.pickup_excursion_pu[0][0], 1.0)
        self.assertAlmostEqual(result.trip_excursion_pu[0][0], 1.0)
        self.assertEqual(result.channel_classes["p1:pulse"], "certified")

    def test_empirical_sampled_class_when_padding_tolerance_not_met(self) -> None:
        reduced = _reduced(
            ar=[],
            dr=[],
            cr=[[]],
            fd=[[1.0]],
        )
        pulse = TimeProfile(
            ProfileKind.SAMPLED,
            samples=((0.0, 0.0), (0.5, 1.0), (1.0, 0.0)),
        )

        result = compute_finite_window_maps(
            reduced,
            [_protected("p1", 0.5)],
            event_ids=("pulse",),
            event_profiles=(pulse,),
            horizons_s=(1.0,),
            time_grid_s=(0.0, 0.5, 1.0),
            max_refinements=0,
            padding_tolerance_pu=1e-12,
        )

        self.assertEqual(result.channel_classes["p1:pulse"], "empirical_sampled")
        self.assertEqual(result.window_result.status, AssessmentStatus.EMPIRICAL)
        self.assertGreater(result.disturbance_padding_pu[0][0], 0.0)

    def test_control_authority_is_time_resolved_lower_bound(self) -> None:
        reduced = _reduced(
            ar=[],
            dr=[],
            cr=[[]],
            fd=[[0.0]],
            br=[],
            fu=[[-0.2]],
        )

        result = compute_finite_window_maps(
            reduced,
            [_protected("p1", 0.1)],
            event_ids=("e1",),
            control_ids=("c1",),
            horizons_s=(0.2,),
            time_grid_s=(0.0, 0.2),
        )

        self.assertEqual(result.window_result.control_ids, ("c1",))
        self.assertAlmostEqual(
            result.window_result.control_authority_lower[0].lower_bound_matrix[0][0],
            2.0,
        )
        self.assertAlmostEqual(result.control_authority_rank[0][0], 2.0)

    def test_data_limited_margin_marks_failed_channel(self) -> None:
        reduced = _reduced(
            ar=[],
            dr=[],
            cr=[[]],
            fd=[[0.1]],
        )

        result = compute_finite_window_maps(
            reduced,
            [_protected("p1", 0.0, certifiable=False)],
            event_ids=("e1",),
            horizons_s=(0.1,),
            time_grid_s=(0.0, 0.1),
        )

        self.assertEqual(result.channel_classes["p1:e1"], "failed")
        self.assertEqual(result.window_result.status, AssessmentStatus.DATA_LIMITED)
        self.assertEqual(result.window_result.data_limited_assets, ("p1",))


if __name__ == "__main__":
    unittest.main()
