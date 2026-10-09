"""Tests of the event response engine on the IEEE 39 test system."""

from __future__ import annotations

import unittest

import numpy as np

from tests import _core_system as CS
from pa_dvsa.data_model import ProfileKind, TimeProfile
from pa_dvsa.event_response import (
    EventResponseEngine,
    EventResponseError,
    OutputSelector,
)
from pa_dvsa.events import EventSpec, Mutation


class EventResponseEngineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.ss = CS.fresh_system()
        cls.outputs = CS.output_buses()
        cls.engine = EventResponseEngine(
            cls.ss, [OutputSelector.bus_voltage(cls.ss, b, n) for n, b in cls.outputs]
        )
        cls.events = CS.event_library(cls.ss)

    def test_pre_mode_is_stable_after_regularization(self) -> None:
        diag = self.engine.pre_mode.diagnostics()
        self.assertLessEqual(diag.max_real_eigenvalue, 1e-6)
        self.assertFalse(diag.unstable_flag)
        self.assertGreater(diag.n_states, 100)
        self.assertLess(self.engine.baseline_residual[1], 1e-9)

    def test_every_family_restores_the_system_exactly(self) -> None:
        dae = self.ss.dae
        x0 = np.array(dae.x, dtype=float)
        y0 = np.array(dae.y, dtype=float)
        specs = list(self.events.values()) + [CS.statcom_control()]
        for spec in specs:
            with self.subTest(spec=getattr(spec, "event_id", getattr(spec, "control_id", ""))):
                if hasattr(spec, "control_id"):
                    resp = self.engine.control_response(spec)
                else:
                    resp = self.engine.event_response(spec)
                self.engine.assert_restored()
                np.testing.assert_array_equal(np.asarray(dae.x), x0)
                np.testing.assert_array_equal(np.asarray(dae.y), y0)
                self.assertTrue(np.all(np.isfinite(resp.values_pu)))
                self.assertEqual(resp.values_pu.shape, (self.engine.times_s.size, len(self.outputs)))
                for env in resp.envelopes:
                    self.assertGreaterEqual(env.peak_upper, env.sampled_max - 1e-15)
        pos = list(self.ss.PVD1.idx.v).index("PD_B1")
        self.assertEqual(float(self.ss.PVD1.u.v[pos]), 1.0)

    def test_restoration_check_detects_a_leftover_mutation(self) -> None:
        pos = list(self.ss.Shunt.idx.v).index("SHR1")
        self.ss.Shunt.b.v[pos] += 0.01
        try:
            self.ss.TDS.fg_update(self.ss.exist.pflow_tds)
            self.ss.j_update(models=self.ss.exist.pflow_tds)
            with self.assertRaisesRegex(EventResponseError, "restoration failed"):
                self.engine.assert_restored()
        finally:
            self.ss.Shunt.b.v[pos] -= 0.01
            self.ss.TDS.fg_update(self.ss.exist.pflow_tds)
            self.ss.j_update(models=self.ss.exist.pflow_tds)
        self.engine.assert_restored()

    def test_switch_jump_is_exact_algebraic_solution(self) -> None:
        resp = self.engine.event_response(self.events["plant_trip"])
        self.assertGreater(resp.newton_iterations, 0)
        self.assertLess(resp.newton_residual, 1e-8)
        np.testing.assert_allclose(resp.values_pu[0], resp.jump_pu, atol=1e-9)
        self.assertEqual(int(np.argmax(resp.jump_pu)), 0)
        self.assertEqual(resp.disconnects_assets, ("B1",))

    def test_param_path_is_linear_in_amplitude(self) -> None:
        full = EventSpec("stc_full", "test", "param", (Mutation("Shunt", "STC1", "b", -1.0),))
        half = EventSpec("stc_half", "test", "param", (Mutation("Shunt", "STC1", "b", -0.5),))
        r1 = self.engine.event_response(full)
        r2 = self.engine.event_response(half)
        np.testing.assert_allclose(r2.values_pu, 0.5 * r1.values_pu, rtol=1e-9, atol=1e-12)

    def test_control_response_follows_its_envelope_delay(self) -> None:
        resp = self.engine.control_response(CS.statcom_control())
        before = resp.times_s < 0.02 - 1e-12
        self.assertLess(float(np.max(np.abs(resp.values_pu[before]))), 1e-12)
        self.assertLess(float(resp.values_pu[-1, 3]), -5e-3)

    def test_ramp_event_starts_at_zero_and_is_continuous(self) -> None:
        resp = self.engine.event_response(self.events["fixed_pf_ramp"])
        np.testing.assert_allclose(resp.values_pu[0], 0.0, atol=1e-12)
        steps = np.abs(np.diff(resp.values_pu[:, 0]))
        self.assertLess(float(np.max(steps)), 2e-3)

    def test_newton_failure_raises_and_restores(self) -> None:
        engine = EventResponseEngine(
            self.ss,
            [OutputSelector.bus_voltage(self.ss, 16)],
            newton_max_iter=1,
            newton_residual_tol=1e-14,
            newton_step_tol=0.0,
        )
        with self.assertRaisesRegex(EventResponseError, "did not converge"):
            engine.event_response(self.events["generator_trip"])
        engine.assert_restored()
        self.engine.assert_restored()

    def test_switch_profiles_must_be_unit_steps_at_zero(self) -> None:
        bad = EventSpec(
            "late_trip", "plant_trip", "switch", (Mutation("PVD1", "PD_B2", "u", 0.0),),
            profile=TimeProfile(ProfileKind.STEP, start_s=0.1),
        )
        with self.assertRaises(Exception):
            self.engine.event_response(bad)
        self.engine.assert_restored()


if __name__ == "__main__":
    unittest.main()
