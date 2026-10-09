"""Tests of the exact propagation and of the upper bound on the window maximum."""

from __future__ import annotations

import math
import unittest

import numpy as np
from scipy import sparse

from pa_dvsa.data_model import ProfileKind, TimeProfile
from pa_dvsa.events import lag_envelope
from pa_dvsa.finite_window import (
    ModeDynamics,
    assemble_window_maps,
    dwell_excursion_upper,
    profile_response,
    propagate_channel,
    rigorous_envelope,
)
from pa_dvsa.linearization import ReducedLinearModel, ReductionDiagnostics

GRID = np.round(np.arange(0.0, 1.0 + 1e-9, 0.005), 12)
DENSE = np.round(np.arange(0.0, 1.0 + 1e-9, 1e-4), 12)
PROFILES = (
    TimeProfile(ProfileKind.STEP),
    TimeProfile(ProfileKind.RAMP, duration_s=0.5),
    lag_envelope(0.02, 0.03, horizon_s=1.5),
    TimeProfile(ProfileKind.STEP, start_s=0.1),
)


def _random_stiff_system(seed: int, n: int = 25, p: int = 3):
    rng = np.random.default_rng(seed)
    a = -np.diag(rng.uniform(0.5, 800.0, n)) + 8.0 * rng.standard_normal((n, n))
    shift = np.linalg.eigvals(a).real.max() + 0.3
    a -= shift * np.eye(n)
    a[0, 1], a[1, 0] = 12.0, -12.0
    b = rng.standard_normal(n)
    c = rng.standard_normal((p, n))
    f = 0.1 * rng.standard_normal(p)
    return a, b, c, f


def _dense_extrema(mode, c, b, f, profile):
    rd = propagate_channel(mode, c, b, f, profile, DENSE, keep_states=False)
    return rd


class PropagationTests(unittest.TestCase):
    def test_exact_propagation_matches_existing_profile_response(self) -> None:
        a, b, c, f = _random_stiff_system(3)
        n, p = a.shape[0], c.shape[0]
        diag = ReductionDiagnostics("toy", n, 1, p, 0, 1, False, True, 1.0, None, "stable", ())
        reduced = ReducedLinearModel(
            ar=sparse.csc_matrix(a),
            br=sparse.csc_matrix((n, 0)),
            dr=sparse.csc_matrix(b.reshape(-1, 1)),
            cr=sparse.csc_matrix(c),
            fu=sparse.csc_matrix((p, 0)),
            fd=sparse.csc_matrix(f.reshape(-1, 1)),
            diagnostics=diag,
        )
        mode = ModeDynamics(a)
        for profile in PROFILES:
            with self.subTest(kind=profile.kind.value, start=profile.start_s):
                resp = propagate_channel(mode, c, b, f, profile, GRID)
                ref = profile_response(
                    reduced, resp.times_s, input_matrix=[[1.0]], profiles=(profile,), channel_type="disturbance"
                )
                np.testing.assert_allclose(resp.phi_right, ref.values_pu[:, :, 0], atol=1e-11, rtol=1e-9)

    def test_step_response_of_scalar_system_is_analytic(self) -> None:
        mode = ModeDynamics(np.array([[-2.0]]))
        resp = propagate_channel(mode, [[3.0]], [1.0], [0.5], TimeProfile(ProfileKind.STEP), GRID)
        expected = 0.5 + 1.5 * (1.0 - np.exp(-2.0 * resp.times_s))
        np.testing.assert_allclose(resp.phi_right[:, 0], expected, atol=1e-13)
        self.assertEqual(resp.phi_left[0, 0], 0.0)


class RigorousBoundTests(unittest.TestCase):
    def test_bounds_dominate_a_dense_grid_for_random_stiff_channels(self) -> None:
        ratios = []
        for seed in range(4):
            a, b, c, f = _random_stiff_system(seed)
            mode = ModeDynamics(a)
            for profile in PROFILES:
                resp = propagate_channel(mode, c, b, f, profile, GRID)
                dense = _dense_extrema(mode, c, b, f, profile)
                for i in range(c.shape[0]):
                    env = rigorous_envelope(resp, i, GRID, refine_lower=True)
                    dmax = max(dense.phi_right[:, i].max(), dense.phi_left[:, i].max())
                    dmin = min(dense.phi_right[:, i].min(), dense.phi_left[:, i].min())
                    self.assertGreaterEqual(env.peak_upper, dmax - 1e-12)
                    self.assertLessEqual(env.trough_lower, dmin + 1e-12)
                    idx = np.searchsorted(GRID, dense.times_s[:-1] + 1e-12, side="right") - 1
                    for l in range(GRID.size - 1):
                        sel = idx == l
                        if np.any(sel):
                            self.assertGreaterEqual(env.upper[l], dense.phi_right[:-1][sel, i].max() - 1e-12)
                            self.assertLessEqual(env.lower[l], dense.phi_right[:-1][sel, i].min() + 1e-12)
                    scale = max(abs(dmax), abs(dmin), 1e-9)
                    ratios.append((env.peak_upper - dmax) / scale)
        self.assertLess(float(np.median(ratios)), 1e-2)

    def test_dwell_bound_dominates_dense_brute_force(self) -> None:
        a, b, c, f = _random_stiff_system(11)
        mode = ModeDynamics(a)
        for profile in PROFILES[:3]:
            resp = propagate_channel(mode, c, b, f, profile, GRID)
            dense = _dense_extrema(mode, c, b, f, profile)
            for i in range(c.shape[0]):
                env = rigorous_envelope(resp, i, GRID)
                bound = dwell_excursion_upper(
                    env.fine_times_s, env.fine_values, env.fine_left_values, env.fine_upper,
                    horizon_s=1.0, dwell_s=0.05,
                )
                v = dense.phi_right[:, i]
                w = int(round(0.05 / 1e-4))
                brute = max(float(np.min(v[k : k + w + 1])) for k in range(v.size - w))
                self.assertGreaterEqual(bound, brute - 1e-12)
                self.assertLessEqual(bound, env.peak_upper + 1e-12)

    def test_dwell_excursion_definition_cases(self) -> None:
        t = np.array([0.0, 0.5, 1.0, 1.5])
        v = np.array([0.0, 1.0, 1.0, 0.0])
        self.assertAlmostEqual(dwell_excursion_upper(t, v, None, None, horizon_s=1.5, dwell_s=0.5), 1.0)
        self.assertEqual(dwell_excursion_upper(t, v, None, None, horizon_s=0.4, dwell_s=0.5), -math.inf)
        self.assertAlmostEqual(dwell_excursion_upper(t, v, None, None, horizon_s=1.5, dwell_s=0.0), 1.0)

    def test_assemble_window_maps_excludes_self_terms_and_marks_data_limited(self) -> None:
        mode = ModeDynamics(np.array([[-5.0]]))
        chans = []

        class _Resp:
            pass

        for eid, gain, disc in (("trip_A", 1.0, ("A",)), ("trip_B", 0.5, ("B",))):
            ch = propagate_channel(mode, [[1.0], [2.0], [0.5]], [gain], [0.0, 0.0, 0.0], TimeProfile(ProfileKind.STEP), GRID)
            r = _Resp()
            r.event_id = eid
            r.times_s = GRID
            r.disconnects_assets = disc
            r.envelopes = tuple(rigorous_envelope(ch, i, GRID) for i in range(3))
            chans.append(r)
        maps = assemble_window_maps(
            chans, asset_ids=("A", "B", "C"), margins_pu=(0.1, 0.2, -0.01), dwell_s=0.05
        )
        self.assertEqual(maps.classes[0, 0], "self")
        self.assertEqual(maps.classes[1, 1], "self")
        self.assertEqual(maps.k_pickup[0, 0], 0.0)
        self.assertEqual(maps.classes[2, 0], "data_limited")
        self.assertEqual(maps.classes[1, 0], "certified")
        peak = 2.0 * (1.0 - math.exp(-5.0)) / 5.0
        self.assertGreaterEqual(maps.pickup_excursion_pu[1, 0], peak - 1e-12)
        self.assertLessEqual(maps.pickup_excursion_pu[1, 0], peak + 1e-4)
        self.assertAlmostEqual(maps.k_pickup[1, 0], maps.pickup_excursion_pu[1, 0] / 0.2)
        self.assertLessEqual(maps.k_trip[1, 0], maps.k_pickup[1, 0] + 1e-15)
        wr = maps.to_window_result()
        self.assertEqual(wr.data_limited_assets, ("C",))


if __name__ == "__main__":
    unittest.main()
