"""Window maps, plant classes, robust bounds and the mitigation LP on the IEEE 39 test system."""

from __future__ import annotations

import unittest

import numpy as np

from tests import _core_system as CS
from pa_dvsa.cascade_certificate import screen_seed
from pa_dvsa.data_model import ProfileKind, TimeProfile
from pa_dvsa.event_response import EventResponseEngine, OutputSelector
from pa_dvsa.events import EventSpec, Mutation, lag_envelope
from pa_dvsa.finite_window import assemble_window_maps, propagate_channel, rigorous_envelope
from pa_dvsa.mitigation import solve_mitigation_from_responses
from pa_dvsa.protected_outputs import output_models_from_blocks
from pa_dvsa.robust_screening import (
    UncertaintyIntervals,
    classify_seed,
    control_lower_authority,
    disturbance_upper_erosion,
    robust_window_bounds,
)

DENSE = np.round(np.arange(0.0, 1.0 + 1e-9, 1e-4), 12)


class CoreScreenPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.ss = CS.fresh_system()
        cls.blocks = CS.protected_blocks(cls.ss)
        buses = [b.collector_bus for b in cls.blocks] + [b.hv_bus for b in cls.blocks]
        cls.engine = EventResponseEngine(
            cls.ss, [OutputSelector.bus_voltage(cls.ss, b) for b in buses], keep_channels=True
        )
        ev = CS.event_library(cls.ss)
        trips = [
            EventSpec(f"plant_trip_{b.asset_id}", "plant_trip", "switch",
                      (Mutation("PVD1", b.pvd1_idx, "u", 0.0),), disconnects_assets=(b.asset_id,))
            for b in cls.blocks
        ]
        cls.seed = ev["load_disconnection"]
        cls.responses = cls.engine.responses(trips + [cls.seed, ev["fixed_pf_ramp"]])
        cls.control = CS.statcom_control()
        cls.control_resp = cls.engine.control_response(cls.control)
        cls.models = output_models_from_blocks(cls.blocks, cls.engine.output_ids, cls.engine.z0)
        cls.amap = {b.asset_id: f"plant_trip_{b.asset_id}" for b in cls.blocks}

    def test_rigorous_bound_dominates_dense_grid_on_real_channels(self) -> None:
        ratios = []
        rng = np.random.default_rng(5)
        pm = self.engine.pre_mode
        channels = [(r.channel, r.profile) for r in self.responses[:4]]
        for k in range(3):
            b = 0.05 * rng.standard_normal(pm.dynamics.n)
            c = 0.05 * rng.standard_normal((2, pm.dynamics.n))
            prof = (TimeProfile(ProfileKind.STEP), TimeProfile(ProfileKind.RAMP, duration_s=0.3),
                    lag_envelope(0.02, 0.03, horizon_s=1.5))[k]
            channels.append((propagate_channel(pm.dynamics, c, b, 0.01 * rng.standard_normal(2), prof,
                                               self.engine.times_s), prof))
        for ch, prof in channels:
            dense = propagate_channel(ch.mode, ch.c, ch.b, ch.f, prof, DENSE, keep_states=False)
            for i in range(ch.n_outputs):
                env = rigorous_envelope(ch, i, self.engine.times_s)
                dmax = max(dense.phi_right[:, i].max(), dense.phi_left[:, i].max())
                self.assertGreaterEqual(env.peak_upper, dmax - 1e-12)
                scale = max(abs(dmax), float(np.abs(dense.phi_right[:, i]).max()), 1e-9)
                ratios.append((env.peak_upper - dmax) / scale)
        self.assertLess(float(np.median(ratios)), 1e-2)

    def test_maps_classes_and_self_terms(self) -> None:
        maps = assemble_window_maps(
            self.responses,
            asset_ids=[m.asset_id for m in self.models],
            output_index={m.asset_id: m.output_index for m in self.models},
            margins_pu=[m.nominal_margin_pu for m in self.models],
            dwell_s=0.05,
        )
        for k, b in enumerate(self.blocks):
            j = maps.column(f"plant_trip_{b.asset_id}")
            self.assertEqual(maps.classes[k, j], "self")
            self.assertEqual(maps.k_pickup[k, j], 0.0)
        self.assertTrue(np.all(maps.k_pickup >= 0.0))
        self.assertTrue(np.all(maps.k_trip <= maps.k_pickup + 1e-12))
        seed = screen_seed(maps, self.seed.event_id, asset_event_map=self.amap, epsilon=0.05,
                           responses=self.responses)
        classes = {c.asset_id: c.asset_class for c in seed.classification.classes}
        self.assertEqual(set(classes), {"B1", "B2", "B3"})
        for c in seed.classification.classes:
            if c.asset_class == "pickup_predicted":
                self.assertGreaterEqual(c.score, 1.0)
            elif c.asset_class == "risky":
                self.assertTrue(0.95 <= c.score < 1.0)
            elif c.asset_class == "certified":
                self.assertLess(c.score, 0.95)
        self.assertEqual(classes["B1"], "pickup_predicted")
        own = screen_seed(maps, "plant_trip_B1", asset_event_map=self.amap)
        self.assertIn("B1", own.classification.excluded_assets)
        self.assertNotIn("B1", [c.asset_id for c in own.classification.classes])

    def test_robust_bounds_are_monotone_in_the_uncertainty_set(self) -> None:
        nominal = robust_window_bounds(self.responses, self.models)
        unc = UncertaintyIntervals(
            relay_threshold_pu={"*": (1.095, 1.105)},
            reconstruction_error_pu={"*": (0.0, 0.002)},
            relay_dwell_s={"*": (0.04, 0.06)},
            q_absorption_scale={"*": (0.9, 1.1)},
            control_gain_scale={"*": (0.9, 1.0)},
            control_delay_s={"*": (0.0, 0.02)},
        )
        robust = robust_window_bounds(self.responses, self.models, unc)
        self.assertTrue(np.all(robust.h_lower < nominal.h_lower))
        np.testing.assert_allclose(robust.h_lower, nominal.h_lower - 0.007, atol=1e-12)
        finite = np.isfinite(nominal.k_bar_pk)
        self.assertTrue(np.all(robust.k_bar_pk[finite] >= nominal.k_bar_pk[finite] - 1e-15))
        self.assertTrue(np.all(robust.k_bar_tr[finite] >= nominal.k_bar_tr[finite] - 1e-15))
        h_nom = control_lower_authority(nominal, [self.control_resp]) * nominal.h_lower[None, :, None]
        h_rob = control_lower_authority(robust, [self.control_resp], uncertainty=unc) * robust.h_lower[None, :, None]
        self.assertTrue(np.all(h_rob <= h_nom + 1e-15))
        self.assertTrue(np.any(h_rob < h_nom - 1e-6))
        m_nom = disturbance_upper_erosion(nominal, [self.responses[3]])
        m_rob = disturbance_upper_erosion(robust, [self.responses[3]])
        self.assertTrue(np.all(m_rob >= m_nom - 1e-15))
        big = robust_window_bounds(self.responses, self.models,
                                   UncertaintyIntervals(reconstruction_error_pu={"*": (0.0, 0.05)}))
        self.assertEqual(set(big.data_limited_assets), {"B1", "B2", "B3"})
        cls = classify_seed(big.k_bar_pk, asset_ids=big.asset_ids, event_ids=big.event_ids,
                            seed_event_id=self.seed.event_id, r_bar=big.r_bar, h_lower=big.h_lower,
                            channel_classes=big.classes)
        self.assertEqual({c.asset_class for c in cls.classes}, {"data_limited"})

    def test_reconstructed_outputs_scale_with_the_tap_bound(self) -> None:
        rec = output_models_from_blocks(self.blocks, self.engine.output_ids, self.engine.z0,
                                        kind="reconstructed")
        taps = {b.asset_id: (b.tap * 0.99, b.tap * 1.01) for b in self.blocks}
        bounds = robust_window_bounds(self.responses, rec, UncertaintyIntervals(tap_ratio=taps))
        for m, a in zip(rec, bounds.assets):
            self.assertAlmostEqual(a.response_scale_upper, 1.0 / (m.tap_nominal * 0.99))
            self.assertAlmostEqual(a.h_lower_pu, m.pickup_pu - m.z0_pu / (m.tap_nominal * 0.99))

    def test_mitigation_lp_from_rigorous_envelopes(self) -> None:
        seed = self.responses[3]
        lp = solve_mitigation_from_responses(
            [seed], [self.control_resp], self.models, controls=[self.control], epsilon=0.05,
        )
        res = lp.mitigation_result
        self.assertTrue(lp.metadata["continuous_time"])
        for sel in res.selections:
            self.assertLessEqual(sel.alpha, self.control.alpha_max + 1e-9)
            self.assertAlmostEqual(sel.magnitude_mvar, sel.alpha * self.control.mvar_per_unit)
        resid = np.asarray(lp.residual_erosion)
        if res.feasible:
            self.assertLessEqual(float(np.max(resid)), 0.95 + 1e-7)
        grid = solve_mitigation_from_responses(
            [seed], [self.control_resp], self.models, controls=[self.control], epsilon=0.05,
            continuous_time=False,
        )
        self.assertGreaterEqual(res.slack_eta, grid.mitigation_result.slack_eta - 1e-9)
        self.assertGreaterEqual(res.objective_value, grid.mitigation_result.objective_value - 1e-6)


if __name__ == "__main__":
    unittest.main()
