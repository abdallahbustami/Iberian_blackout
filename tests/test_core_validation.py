"""Screening responses against nonlinear time-domain simulation for every event type (slow)."""

from __future__ import annotations

import math
import unittest

import pytest

from tests import _core_validation as V

TOL = {
    "plant_trip": (0.6, 2.2),
    "generator_trip": (4.5, 10.0),
    "load_disconnection": (0.3, 0.4),
    "shunt_reactor_disconnection": (0.1, 0.3),
    "capacitor_switching": (0.1, 0.2),
    "line_energization": (0.12, 0.2),
    "fixed_pf_ramp": (0.6, 2.2),
    "governor_ramp": (0.05, 0.5),
    "exciter_step": (0.05, 0.05),
    "statcom": (0.4, 0.4),
}


@pytest.mark.slow
class CoreValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.data = V.run_validation()
        cls.rows = {row["family"]: row for row in cls.data["rows"]}

    def test_all_families_present_and_tds_converged(self) -> None:
        self.assertEqual(set(self.rows), set(V.FAMILIES))
        self.assertLess(self.data["no_event_max_abs_mpu"], 1e-3)
        for fam, row in self.rows.items():
            self.assertTrue(row["tds_converged"], f"{fam}: {row['tds_err_msg']}")
            self.assertFalse(row["diagnostics"]["unstable_flag"], fam)

    def test_every_family_changes_tds_voltages(self) -> None:
        for fam, row in self.rows.items():
            with self.subTest(family=fam):
                biggest = max(o["tds_minus_noevent_max_abs_mpu"] for o in row["outputs"])
                self.assertGreater(biggest, 0.5)

    def test_initial_response_sign_and_magnitude(self) -> None:
        for fam, row in self.rows.items():
            for o in row["outputs"]:
                with self.subTest(family=fam, output=o["output"]):
                    self.assertAlmostEqual(o["jump_screen_mpu"], o["dz0_tds_mpu"], delta=1e-3)
                    s, t = o["dz20ms_screen_mpu"], o["dz20ms_tds_mpu"]
                    if abs(t) > 0.5:
                        self.assertEqual(math.copysign(1.0, s), math.copysign(1.0, t))
                        self.assertLess(abs(s - t), 0.02 * abs(t) + 0.05)

    def test_waveform_errors_within_documented_tolerances(self) -> None:
        for fam, row in self.rows.items():
            tol03, tol1 = TOL[fam]
            with self.subTest(family=fam):
                self.assertLessEqual(max(o["max_abs_err_0_0p3_mpu"] for o in row["outputs"]), tol03)
                self.assertLessEqual(max(o["max_abs_err_0_1_mpu"] for o in row["outputs"]), tol1)


if __name__ == "__main__":
    unittest.main()
