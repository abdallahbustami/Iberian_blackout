"""Tests of the time-domain simulation with relays on the IEEE 39 test system."""

from __future__ import annotations

import unittest

import numpy as np

from tests import _core_system as CS
from pa_dvsa.tds_runner import run_protection_tds


def _addr(ss, bus) -> int:
    return int(ss.Bus.v.a[list(ss.Bus.idx.v).index(bus)])


class TDSRunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.ss = CS.fresh_system()
        cls.blocks = CS.protected_blocks(cls.ss)
        cls.events = CS.event_library(cls.ss)
        cls.taps = CS.solved_taps()
        cls.trace = {
            "COL_B1": _addr(cls.ss, CS.collector_bus(0)),
            "HV16": _addr(cls.ss, 16),
            "COL_B2": _addr(cls.ss, CS.collector_bus(1)),
            "HV21": _addr(cls.ss, 21),
        }

    def _run(self, seed, **kwargs):
        return run_protection_tds(
            CS.fresh_system,
            seed,
            self.blocks,
            t_seed=0.5,
            window_s=1.0,
            record_traces=True,
            trace_addresses=self.trace,
            **kwargs,
        )

    def test_no_event_run_is_steady_and_has_no_pickups(self) -> None:
        res = self._run(None, t_end=1.0)
        self.assertTrue(res.converged, res.err_msg)
        self.assertLess(res.pre_seed_max_dv_pu, 1e-6)
        for rec in res.assets:
            self.assertIsNone(rec.first_pickup_s)
            self.assertIsNone(rec.trip_s)
            self.assertLess(abs(rec.max_v_window_pu - rec.v0_pu), 1e-6)
        ex = res.traces["extra"]
        drop = ex[-1, 0] - ex[-1, 1] / self.taps[0]
        self.assertGreater(abs(drop), 1e-3)

    def test_plant_trip_removes_injection_and_disables_own_relay(self) -> None:
        res = self._run(self.events["plant_trip"], relays_enabled=False, t_end=1.2)
        self.assertTrue(res.converged, res.err_msg)
        b1 = res.asset("B1")
        self.assertFalse(b1.active)
        self.assertEqual(b1.inactive_reason, "disconnected_by_seed")
        self.assertIsNone(b1.trip_s)
        t = res.traces["t"]
        ex = res.traces["extra"]
        post = t > 0.5 + 1e-9
        np.testing.assert_allclose(ex[post, 0], ex[post, 1] / self.taps[0], atol=1e-7)
        self.assertGreater(float(np.max(ex[post, 0])), float(ex[0, 0]) + 0.02)

    def test_relay_trips_after_dwell_and_removes_injection(self) -> None:
        base = self._run(self.events["plant_trip"], relays_enabled=False, t_end=1.2)
        v_b2 = base.traces["v_collector"][:, 1]
        pickup = float(v_b2[0] + 0.5 * (np.max(v_b2) - v_b2[0]))
        res = self._run(
            self.events["plant_trip"],
            relays_enabled=True,
            pickup_override={"B2": pickup},
            t_end=1.2,
        )
        self.assertTrue(res.converged, res.err_msg)
        b2 = res.asset("B2")
        self.assertIsNotNone(b2.first_pickup_s)
        self.assertIsNotNone(b2.trip_s)
        t = res.traces["t"]
        v = res.traces["v_collector"][:, 1]
        first = float(t[np.flatnonzero(v >= pickup)[0]])
        self.assertAlmostEqual(b2.first_pickup_s, first, places=9)
        self.assertGreaterEqual(b2.trip_s - b2.first_pickup_s, b2.dwell_s - 1e-9)
        self.assertLessEqual(b2.trip_s - b2.first_pickup_s, b2.dwell_s + 0.002 + 1e-9)
        mask = (t >= b2.first_pickup_s - 1e-12) & (t <= b2.trip_s + 1e-12)
        self.assertTrue(np.all(v[mask][:-1] >= pickup) or np.all(v[mask] >= pickup - 1e-12))
        self.assertEqual(res.trip_order[0], "B2")
        ex = res.traces["extra"]
        after = t > b2.trip_s + 1e-9
        np.testing.assert_allclose(ex[after, 2], ex[after, 3] / self.taps[1], atol=1e-7)

    def test_relay_timer_resets_below_pickup(self) -> None:
        base = self._run(self.events["plant_trip"], relays_enabled=False, t_end=1.2)
        v_b2 = base.traces["v_collector"][:, 1]
        res = self._run(
            self.events["plant_trip"],
            relays_enabled=True,
            pickup_override={"B2": float(v_b2[0] + 1e-4)},
            dwell_override={"B2": 5.0},
            t_end=1.2,
        )
        self.assertIsNotNone(res.asset("B2").first_pickup_s)
        self.assertIsNone(res.asset("B2").trip_s)

    def test_relay_with_dropout_keeps_timing_between_dropout_and_pickup(self) -> None:
        base = self._run(self.events["plant_trip"], relays_enabled=False, t_end=1.2)
        t = base.traces["t"]
        v = base.traces["v_collector"][:, 1]
        pickup = float(np.max(v) - 1e-4)
        k = int(np.flatnonzero(v >= pickup)[0])
        dwell = 0.2
        window = (t >= t[k]) & (t <= t[k] + dwell + 0.01)
        self.assertTrue(np.any(v[window] < pickup))
        self.assertTrue(np.all(v[window] >= 0.98 * pickup))
        kw = dict(relays_enabled=True, pickup_override={"B2": pickup}, dwell_override={"B2": dwell}, t_end=1.2)
        plain = self._run(self.events["plant_trip"], **kw)
        held = self._run(self.events["plant_trip"], dropout_ratio=0.98, **kw)
        self.assertIsNone(plain.asset("B2").trip_s)
        b2 = held.asset("B2")
        self.assertIsNotNone(b2.trip_s)
        self.assertAlmostEqual(b2.first_pickup_s, float(t[k]), places=9)
        self.assertGreaterEqual(b2.trip_s - b2.first_pickup_s, dwell - 1e-9)

    def test_callbacks_on_retried_steps_are_idempotent(self) -> None:
        def hook(callback, t, system):
            callback(t, system)
            callback(t - 0.5 * float(system.TDS.h), system)
            callback(t, system)

        kwargs = dict(relays_enabled=True, pickup_override={"B1": 1.09}, t_end=1.2)
        ref = self._run(self.events["fixed_pf_ramp"], **kwargs)
        dup = self._run(self.events["fixed_pf_ramp"], callpert_hook=hook, **kwargs)
        self.assertTrue(ref.converged and dup.converged)
        self.assertIsNotNone(ref.asset("B1").trip_s)
        for a, b in zip(ref.assets, dup.assets):
            self.assertEqual(a.first_pickup_s, b.first_pickup_s)
            self.assertEqual(a.trip_s, b.trip_s)
            self.assertAlmostEqual(a.max_v_window_pu, b.max_v_window_pu, places=12)
        np.testing.assert_allclose(ref.traces["extra"], dup.traces["extra"], atol=1e-12)

    def test_control_envelope_is_applied_with_alpha(self) -> None:
        spec = CS.statcom_control()
        res = self._run(None, controls=[(spec, 0.5)], t_end=1.0)
        self.assertTrue(res.converged, res.err_msg)
        t = res.traces["t"]
        hv16 = res.traces["extra"][:, 1]
        k0 = np.searchsorted(t, 0.5 - 1e-9)
        before_delay = (t > 0.5) & (t < 0.5 + 0.02 - 1e-9)
        self.assertLess(float(np.max(np.abs(hv16[before_delay] - hv16[k0]))), 1e-6)
        self.assertLess(float(hv16[-1] - hv16[k0]), -2e-3)

    def test_seed_and_control_on_the_same_parameter_superpose(self) -> None:
        from pa_dvsa.data_model import ProfileKind, TimeProfile
        from pa_dvsa.events import ControlSpec, Mutation, read_base_values

        seed = self.events["fixed_pf_ramp"]
        q0, p0 = read_base_values(self.ss, [Mutation("PVD1", "PD_B1", "qref0", 0.0), Mutation("PVD1", "PD_B1", "pref0", 0.0)])
        ctrl = ControlSpec(
            "plant_q_B1", "plant_q_command", (Mutation("PVD1", "PD_B1", "qref0", q0 - 0.2 * abs(q0)),),
            envelope=TimeProfile(ProfileKind.STEP, start_s=0.1), alpha_max=1.0,
        )
        holder = {}

        def factory():
            holder["ss"] = CS.fresh_system()
            return holder["ss"]

        res = run_protection_tds(factory, seed, self.blocks, relays_enabled=False, controls=[(ctrl, 0.5)],
                                 t_seed=0.5, window_s=1.0, t_end=1.2)
        self.assertTrue(res.converged, res.err_msg)
        ss = holder["ss"]
        pos = list(ss.PVD1.idx.v).index("PD_B1")
        self.assertAlmostEqual(float(ss.PVD1.qref0.v[pos]), 0.7 * q0 + 0.5 * (-0.2 * abs(q0)), places=12)
        self.assertAlmostEqual(float(ss.PVD1.pref0.v[pos]), 0.7 * p0, places=12)


if __name__ == "__main__":
    unittest.main()
