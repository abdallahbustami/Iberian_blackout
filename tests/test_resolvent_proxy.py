from __future__ import annotations

import math
import unittest

import numpy as np
from scipy import sparse

from pa_dvsa.data_model import AssessmentStatus, ProtectedVoltageKind, Unit
from pa_dvsa.finite_window import ProfileResponse, step_response_d
from pa_dvsa.linearization import ReducedLinearModel, ReductionDiagnostics
from pa_dvsa.protected_outputs import ProtectedOutputEvaluation, ScalarEnvelope
from pa_dvsa.resolvent_proxy import (
    alpha_star,
    assess_control_proxy,
    assess_disturbance_proxy,
    ghat,
)


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
    p = len(cr) if cr else len(fd)
    q = len(fd[0]) if fd else 0
    r = len(fu[0]) if fu else 0
    return ReducedLinearModel(
        ar=sparse.csc_matrix((0, 0)) if n == 0 else sparse.csc_matrix(ar),
        br=sparse.csc_matrix((0, r))
        if n == 0
        else sparse.csc_matrix(br if br is not None else [[0.0] * r for _ in range(n)]),
        dr=sparse.csc_matrix((0, q)) if n == 0 else sparse.csc_matrix(dr),
        cr=sparse.csc_matrix((p, 0)) if n == 0 else sparse.csc_matrix(cr),
        fu=sparse.csc_matrix(fu if fu is not None else [[0.0] * r for _ in range(p)]),
        fd=sparse.csc_matrix(fd),
        diagnostics=ReductionDiagnostics(
            mode_id="proxy_toy",
            n_states=n,
            n_algebraic=1,
            n_outputs=p,
            n_controls=r,
            n_disturbances=q,
            algebraic_only=(n == 0),
            gy_factorized=True,
            gy_min_abs_u_diag=1.0,
            max_real_eigenvalue=-1.0 if n else None,
            stability="stable" if n else "algebraic_only",
            warnings=(),
        ),
    )


class ResolventProxyTests(unittest.TestCase):
    def test_ghat_matches_analytic_resolvent(self) -> None:
        reduced = _reduced(
            ar=[[-2.0]],
            dr=[[1.0]],
            cr=[[3.0]],
            fd=[[0.5]],
        )

        proxy = ghat(reduced, 0.5, channel_type="disturbance")

        self.assertEqual(proxy.shape, (1, 1))
        self.assertAlmostEqual(proxy[0, 0], 0.5 + 3.0 / 4.0, places=12)
        self.assertGreater(alpha_star(), 1.29)
        self.assertLess(alpha_star(), 1.30)

    def test_disturbance_proxy_alpha_star_only_for_monotone_channel(self) -> None:
        reduced = _reduced(
            ar=[[-2.0]],
            dr=[[1.0]],
            cr=[[3.0]],
            fd=[[0.5]],
        )
        response = step_response_d(reduced, [0.0, 0.25, 0.5])

        result = assess_disturbance_proxy(
            reduced,
            [_protected("p1", 0.5)],
            0.5,
            event_ids=("e1",),
            finite_response=response,
        )

        expected_proxy = 0.5 + 3.0 / 4.0
        self.assertTrue(result.monotone_mask[0, 0])
        self.assertTrue(result.conservative_mask[0, 0])
        self.assertEqual(result.monotone_fraction, 1.0)
        self.assertAlmostEqual(result.proxy_pu[0, 0], expected_proxy)
        self.assertAlmostEqual(result.disturbance_upper_pu[0, 0], alpha_star() * expected_proxy)
        self.assertAlmostEqual(result.disturbance_k_upper[0, 0], alpha_star() * expected_proxy / 0.5)
        self.assertEqual(result.channel_classes["p1:e1"], "disturbance_alpha_star_proxy")

    def test_disturbance_proxy_is_ranking_only_without_monotone_verification(self) -> None:
        reduced = _reduced(
            ar=[[-2.0]],
            dr=[[1.0]],
            cr=[[3.0]],
            fd=[[0.5]],
        )

        result = assess_disturbance_proxy(
            reduced,
            [_protected("p1", 0.5)],
            0.5,
            event_ids=("e1",),
        )

        self.assertFalse(result.monotone_mask[0, 0])
        self.assertFalse(result.conservative_mask[0, 0])
        self.assertTrue(math.isnan(result.disturbance_upper_pu[0, 0]))
        self.assertGreater(result.ranking_pu[0, 0], 0.0)
        self.assertEqual(result.channel_classes["p1:e1"], "ranking_only")

    def test_nonmonotone_disturbance_cannot_use_proxy_for_safety(self) -> None:
        reduced = _reduced(
            ar=[[-1.0]],
            dr=[[1.0]],
            cr=[[-1.0]],
            fd=[[1.0]],
        )
        response = step_response_d(reduced, [0.0, 0.5, 1.0])

        result = assess_disturbance_proxy(
            reduced,
            [_protected("p1", 0.5)],
            1.0,
            event_ids=("e1",),
            finite_response=response,
        )

        self.assertFalse(result.monotone_mask[0, 0])
        self.assertFalse(result.conservative_mask[0, 0])
        self.assertTrue(math.isnan(result.disturbance_upper_pu[0, 0]))
        self.assertEqual(result.channel_classes["p1:e1"], "ranking_only")

    def test_disturbance_proxy_rejected_when_alpha_bound_misses_sampled_peak(self) -> None:
        reduced = _reduced(
            ar=[[-2.0]],
            dr=[[1.0]],
            cr=[[3.0]],
            fd=[[0.5]],
        )
        response = ProfileResponse(
            times_s=(0.0, 0.5),
            values_pu=np.asarray([[[0.0]], [[10.0]]], dtype=float),
            derivatives_pu_per_s=np.zeros((2, 1, 1)),
            channel_ids=("e1",),
            profile_kinds=("sampled",),
            channel_type="disturbance",
        )

        result = assess_disturbance_proxy(
            reduced,
            [_protected("p1", 0.5)],
            0.5,
            event_ids=("e1",),
            finite_response=response,
        )

        self.assertTrue(result.monotone_mask[0, 0])
        self.assertFalse(result.conservative_mask[0, 0])
        self.assertTrue(math.isnan(result.disturbance_upper_pu[0, 0]))
        self.assertEqual(result.channel_classes["p1:e1"], "ranking_only")

    def test_control_proxy_lower_bound_when_beneficial_monotone_underestimate(self) -> None:
        reduced = _reduced(
            ar=[[-2.0]],
            dr=[[]],
            cr=[[-3.0]],
            fd=[[]],
            br=[[1.0]],
            fu=[[-0.5]],
        )
        response = ProfileResponse(
            times_s=(0.0, 0.25, 0.5),
            values_pu=np.asarray([[[-0.5]], [[-1.0902]], [[-1.4482]]], dtype=float),
            derivatives_pu_per_s=np.zeros((3, 1, 1)),
            channel_ids=("u1",),
            profile_kinds=("step",),
            channel_type="control",
        )

        result = assess_control_proxy(
            reduced,
            [_protected("p1", 0.5)],
            0.5,
            control_ids=("u1",),
            finite_response=response,
        )

        expected_benefit = 0.5 + 3.0 / 4.0
        self.assertTrue(result.monotone_mask[0, 0])
        self.assertTrue(result.conservative_mask[0, 0])
        self.assertAlmostEqual(result.proxy_pu[0, 0], -expected_benefit)
        self.assertAlmostEqual(result.control_authority_lower[0, 0], expected_benefit / 0.5)
        self.assertEqual(result.channel_classes["p1:u1"], "control_proxy_lower_bound")

    def test_control_proxy_rejected_when_it_overestimates_sampled_benefit(self) -> None:
        reduced = _reduced(
            ar=[[-2.0]],
            dr=[[]],
            cr=[[-3.0]],
            fd=[[]],
            br=[[1.0]],
            fu=[[-0.5]],
        )
        response = ProfileResponse(
            times_s=(0.0, 0.5),
            values_pu=np.asarray([[[0.0]], [[-0.1]]], dtype=float),
            derivatives_pu_per_s=np.zeros((2, 1, 1)),
            channel_ids=("u1",),
            profile_kinds=("sampled"),
            channel_type="control",
        )

        result = assess_control_proxy(
            reduced,
            [_protected("p1", 0.5)],
            0.5,
            control_ids=("u1",),
            finite_response=response,
        )

        self.assertTrue(result.monotone_mask[0, 0])
        self.assertFalse(result.conservative_mask[0, 0])
        self.assertEqual(result.control_authority_lower[0, 0], 0.0)
        self.assertEqual(result.channel_classes["p1:u1"], "ranking_only")

    def test_data_limited_margin_blocks_conservative_proxy(self) -> None:
        reduced = _reduced(
            ar=[[-2.0]],
            dr=[[1.0]],
            cr=[[3.0]],
            fd=[[0.5]],
        )
        response = step_response_d(reduced, [0.0, 0.25, 0.5])

        result = assess_disturbance_proxy(
            reduced,
            [_protected("p1", 0.5, certifiable=False)],
            0.5,
            event_ids=("e1",),
            finite_response=response,
        )

        self.assertTrue(result.monotone_mask[0, 0])
        self.assertFalse(result.conservative_mask[0, 0])
        self.assertTrue(math.isnan(result.disturbance_upper_pu[0, 0]))
        self.assertEqual(result.channel_classes["p1:e1"], "data_limited")


if __name__ == "__main__":
    unittest.main()
