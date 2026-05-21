from __future__ import annotations

from pathlib import Path
import unittest

from pa_dvsa.andes_adapter import (
    AndesAdapterError,
    AndesCase,
    AndesCaseSpec,
    available_benchmarks,
    resolve_case,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class AndesAdapterResolutionTests(unittest.TestCase):
    def test_resolves_stock_benchmarks(self) -> None:
        ieee39 = resolve_case("ieee39", PROJECT_ROOT)
        self.assertTrue(ieee39.exists)
        self.assertEqual(ieee39.source, "andes_stock")
        self.assertTrue(str(ieee39.case_path).endswith("ieee39_full.xlsx"))

        npcc_full = resolve_case("npcc_full", PROJECT_ROOT)
        self.assertTrue(npcc_full.exists)
        self.assertTrue(str(npcc_full.addfile_path).endswith("npcc_full.dyr"))

    def test_resolves_known_local_aliases_without_requiring_loadability(self) -> None:
        benchmarks = available_benchmarks(PROJECT_ROOT)
        self.assertIn("activsg2000_stable", benchmarks)
        self.assertTrue(benchmarks["activsg2000_stable"].exists)
        self.assertEqual(benchmarks["activsg2000_stable"].source, "local")


class AndesAdapterLoadTests(unittest.TestCase):
    def test_loads_static_case_and_exposes_bus_addresses(self) -> None:
        case = AndesCase.load(
            AndesCaseSpec("ieee14", setup=True),
            project_root=PROJECT_ROOT,
        )
        self.assertEqual(case.summary()["n_buses"], 14)
        self.assertEqual(case.bus_angle_address(1), 0)
        self.assertEqual(case.bus_voltage_address(1), 14)
        self.assertEqual(len(case.bus_voltage_addresses()), 14)

        jac = case.jacobians()
        self.assertEqual(jac.n_states, 0)
        self.assertEqual(jac.n_algebraic, 34)
        self.assertEqual(jac.gy.shape, (34, 34))

    def test_loads_dynamic_ieee39_and_exposes_jacobians_and_families(self) -> None:
        case = AndesCase.load(
            AndesCaseSpec("ieee39", setup=True, run_pflow=True, init_tds=True),
            project_root=PROJECT_ROOT,
        )
        summary = case.summary()
        self.assertEqual(summary["n_buses"], 39)
        self.assertGreater(summary["dae_n"], 0)
        self.assertGreater(summary["dae_m"], 0)

        jac = case.jacobians()
        self.assertEqual(jac.fx.shape, (summary["dae_n"], summary["dae_n"]))
        self.assertEqual(jac.gy.shape, (summary["dae_m"], summary["dae_m"]))
        self.assertEqual(len(jac.tf), summary["dae_n"])

        registry = case.registry()
        self.assertIn("GENROU", registry.families["synchronous_machines"])
        self.assertIn("IEEEX1", registry.families["exciters"])
        self.assertIn("IEEEST", registry.families["pss"])
        self.assertIn("TGOV1N", registry.families["governors"])

        gen_record = case.model_record("GENROU")
        self.assertEqual(gen_record.n, 10)
        self.assertEqual(gen_record.devices[0].status, 1.0)
        self.assertEqual(gen_record.devices[0].owners["bus"], "30")

        delta = case.variable_address("GENROU", "delta", "GENROU_1")
        self.assertEqual(delta.domain, "state")
        self.assertEqual(delta.address, 0)
        self.assertEqual(case.dynamic_state_address("GENROU", "delta", "GENROU_1"), 0)

        v = case.variable_address("Bus", "v", 30)
        self.assertEqual(v.domain, "algebraic")
        self.assertGreaterEqual(v.address, 39)
        self.assertEqual(case.algebraic_equation_address("Bus", "v", 30), v.address)

        with self.assertRaises(AndesAdapterError):
            case.dynamic_state_address("Bus", "v", 30)

    def test_unknown_model_and_device_errors_are_explicit(self) -> None:
        case = AndesCase.load(AndesCaseSpec("ieee14", setup=True), project_root=PROJECT_ROOT)
        with self.assertRaises(AndesAdapterError):
            case.get_model("DOES_NOT_EXIST")
        with self.assertRaises(AndesAdapterError):
            case.variable_address("Bus", "v", "missing_bus")


if __name__ == "__main__":
    unittest.main()
