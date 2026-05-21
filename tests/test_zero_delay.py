from __future__ import annotations

import unittest

from scipy import sparse

from pa_dvsa.data_model import AssessmentStatus, ProtectedVoltageKind, Unit
from pa_dvsa.linearization import ReducedLinearModel, ReductionDiagnostics
from pa_dvsa.protected_outputs import ProtectedOutputEvaluation, ScalarEnvelope
from pa_dvsa.zero_delay import (
    ZeroDelayScreenError,
    algebraic_sensitivity_from_gy,
    compute_fixed_tap_screen_from_gy,
    compute_fixed_tap_zero_delay_screen,
    compute_zero_delay_screen,
)


def _protected(asset_id: str, margin: float, *, certifiable: bool = True) -> ProtectedOutputEvaluation:
    z = ScalarEnvelope.exact(1.0, Unit.PU)
    threshold = ScalarEnvelope.exact(1.0 + margin, Unit.PU)
    margin_env = ScalarEnvelope.exact(margin, Unit.PU)
    return ProtectedOutputEvaluation(
        asset_id=asset_id,
        output_id=f"z_{asset_id}",
        kind=ProtectedVoltageKind.DIRECT_BUS_VOLTAGE,
        z_pu=z,
        threshold_pu=threshold,
        margin_pu=margin_env,
        status=AssessmentStatus.CERTIFIED if certifiable else AssessmentStatus.DATA_LIMITED,
        worst_case_margin_pu=margin,
        is_certifiable=certifiable,
        source_bus_id=asset_id,
        source_address=None,
        reason="test_margin",
    )


def _reduced_with_fd(fd: list[list[float]]) -> ReducedLinearModel:
    p = len(fd)
    q = len(fd[0]) if fd else 0
    return ReducedLinearModel(
        ar=sparse.csc_matrix((0, 0)),
        br=sparse.csc_matrix((0, 0)),
        dr=sparse.csc_matrix((0, q)),
        cr=sparse.csc_matrix((p, 0)),
        fu=sparse.csc_matrix((p, 0)),
        fd=sparse.csc_matrix(fd),
        diagnostics=ReductionDiagnostics(
            mode_id="toy_mode",
            n_states=0,
            n_algebraic=1,
            n_outputs=p,
            n_controls=0,
            n_disturbances=q,
            algebraic_only=True,
            gy_factorized=True,
            gy_min_abs_u_diag=1.0,
            max_real_eigenvalue=None,
            stability="algebraic_only",
            warnings=(),
        ),
    )


class ZeroDelayScreenTests(unittest.TestCase):
    def test_k_zero_plus_from_fd_and_margins(self) -> None:
        reduced = _reduced_with_fd([[0.02, -0.01], [0.04, 0.005]])
        result = compute_zero_delay_screen(
            reduced,
            [_protected("p1", 0.1), _protected("p2", 0.02)],
            event_ids=("e1", "e2"),
        )

        k = result.window_result.k_pickup_upper
        self.assertAlmostEqual(k[0][0], 0.2)
        self.assertAlmostEqual(k[0][1], 0.0)
        self.assertAlmostEqual(k[1][0], 2.0)
        self.assertAlmostEqual(k[1][1], 0.25)
        self.assertEqual(result.window_result.horizons_s, (0.0, 0.0))
        self.assertFalse(result.one_step_safe["e1"])
        self.assertTrue(result.one_step_safe["e2"])

    def test_k_zero_plus_supports_event_disturbance_vectors(self) -> None:
        reduced = _reduced_with_fd([[0.1, 0.2]])
        result = compute_zero_delay_screen(
            reduced,
            [_protected("p1", 0.4)],
            event_ids=("aggregate",),
            disturbance_matrix=[[2.0], [3.0]],
        )

        self.assertAlmostEqual(result.instant_delta_z_pu[0][0], 0.8)
        self.assertAlmostEqual(result.window_result.k_pickup_upper[0][0], 2.0)

    def test_data_limited_margin_is_not_certified(self) -> None:
        reduced = _reduced_with_fd([[0.02]])
        result = compute_zero_delay_screen(
            reduced,
            [_protected("p1", 0.0, certifiable=False)],
            event_ids=("e1",),
        )

        self.assertEqual(result.window_result.status, AssessmentStatus.DATA_LIMITED)
        self.assertEqual(result.window_result.data_limited_assets, ("p1",))
        self.assertEqual(result.window_result.k_pickup_upper[0][0], 0.0)
        self.assertFalse(result.one_step_safe["e1"])

        with self.assertRaisesRegex(ZeroDelayScreenError, "positive certified margins"):
            compute_zero_delay_screen(
                reduced,
                [_protected("p1", 0.0, certifiable=False)],
                event_ids=("e1",),
                strict_margins=True,
            )

    def test_fixed_tap_formula_matches_old_algebraic_loop(self) -> None:
        sensitivity = [[0.10, 0.02], [0.03, 0.05]]
        q_abs_mvar = [50.0, 100.0]
        taps = [1.0, 2.0]
        margins = [0.05, 0.10]
        result = compute_fixed_tap_zero_delay_screen(
            sensitivity,
            [_protected("p1", margins[0]), _protected("p2", margins[1])],
            q_absorption_mvar=q_abs_mvar,
            tap_ratios=taps,
            event_ids=("c1_trip", "c2_trip"),
            base_mva=100.0,
        )

        old_k = []
        for i in range(2):
            row = []
            for j in range(2):
                dz = sensitivity[i][j] * (q_abs_mvar[j] / 100.0) / taps[i]
                row.append(max(dz, 0.0) / margins[i])
            old_k.append(row)

        self.assertEqual(result.window_result.k_pickup_upper, tuple(tuple(row) for row in old_k))
        self.assertAlmostEqual(result.window_result.k_pickup_upper[0][0], 1.0)
        self.assertAlmostEqual(result.window_result.k_pickup_upper[0][1], 0.4)
        self.assertAlmostEqual(result.window_result.k_pickup_upper[1][0], 0.075)
        self.assertAlmostEqual(result.window_result.k_pickup_upper[1][1], 0.25)

    def test_algebraic_sensitivity_from_gy_sign_normalizes_and_clamps(self) -> None:
        gy = sparse.csc_matrix([[-10.0, 0.0], [0.0, -20.0]])
        sensitivity = algebraic_sensitivity_from_gy(
            gy,
            output_addresses=(0, 1),
            injection_equation_addresses=(0, 1),
        )

        self.assertAlmostEqual(sensitivity[0, 0], 0.1)
        self.assertAlmostEqual(sensitivity[1, 1], 0.05)
        self.assertAlmostEqual(sensitivity[0, 1], 0.0)
        self.assertAlmostEqual(sensitivity[1, 0], 0.0)

    def test_fixed_tap_screen_from_gy_matches_shortcut(self) -> None:
        gy = sparse.csc_matrix([[10.0, 0.0], [0.0, 20.0]])
        result = compute_fixed_tap_screen_from_gy(
            gy,
            [_protected("p1", 0.05), _protected("p2", 0.10)],
            output_addresses=(0, 1),
            injection_equation_addresses=(0, 1),
            q_absorption_mvar=(50.0, 100.0),
            tap_ratios=(1.0, 2.0),
            event_ids=("c1_trip", "c2_trip"),
            base_mva=100.0,
        )

        self.assertAlmostEqual(result.window_result.k_pickup_upper[0][0], 1.0)
        self.assertAlmostEqual(result.window_result.k_pickup_upper[1][1], 0.25)


if __name__ == "__main__":
    unittest.main()
