from __future__ import annotations

from pathlib import Path
import unittest

from pa_dvsa.andes_adapter import AndesCase, AndesCaseSpec
from pa_dvsa.data_model import (
    AssessmentStatus,
    AssetRef,
    DelayKind,
    DisconnectedAsset,
    Interval,
    MeasurementSide,
    ProtectedAsset,
    ProtectedVoltageKind,
    ProtectedVoltageSpec,
    ProtectionTiming,
    Unit,
)
from pa_dvsa.protected_outputs import (
    ProtectedOutputError,
    evaluate_protected_output,
    evaluate_protected_outputs,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _case() -> AndesCase:
    return AndesCase.load(
        AndesCaseSpec("ieee14", setup=True, run_pflow=True),
        project_root=PROJECT_ROOT,
    )


def _disconnected(asset_id: str = "G1") -> tuple[DisconnectedAsset, ...]:
    return (
        DisconnectedAsset(
            AssetRef(asset_id, model_family="PV", bus_id="1"),
            q_absorption_mvar=Interval.exact(10.0, Unit.MVAR),
        ),
    )


def _direct_asset(threshold: Interval = Interval.exact(1.10, Unit.PU)) -> ProtectedAsset:
    return ProtectedAsset(
        asset_id="P_bus_1",
        name="Bus 1 direct overvoltage relay",
        protected_voltage=ProtectedVoltageSpec(
            output_id="z_bus_1",
            side=MeasurementSide.COLLECTOR_BUS,
            kind=ProtectedVoltageKind.DIRECT_BUS_VOLTAGE,
            protected_bus_id="1",
            reconstruction_error_pu=Interval.exact(0.0, Unit.PU),
        ),
        threshold_pu=threshold,
        timing=ProtectionTiming(DelayKind.ZERO_DELAY),
        disconnected_assets=_disconnected(),
    )


def _fixed_tap_asset(threshold: Interval = Interval.exact(1.10, Unit.PU)) -> ProtectedAsset:
    return ProtectedAsset(
        asset_id="P_tap_1",
        name="Bus 1 fixed tap reconstruction relay",
        protected_voltage=ProtectedVoltageSpec(
            output_id="z_tap_1",
            side=MeasurementSide.RECONSTRUCTED,
            kind=ProtectedVoltageKind.FIXED_TAP_RECONSTRUCTION,
            transmission_bus_id="1",
            protected_bus_id="collector_1",
            tap_ratio=Interval(0.98, 1.00, Unit.PU, nominal=0.99),
            reconstruction_error_pu=Interval(-0.002, 0.003, Unit.PU, nominal=0.001),
        ),
        threshold_pu=threshold,
        timing=ProtectionTiming(DelayKind.ZERO_DELAY),
        disconnected_assets=_disconnected("G_tap"),
    )


class ProtectedOutputTests(unittest.TestCase):
    def test_direct_protected_output_uses_existing_andes_bus(self) -> None:
        result = evaluate_protected_output(_case(), _direct_asset())

        self.assertEqual(result.status, AssessmentStatus.CERTIFIED)
        self.assertTrue(result.is_certifiable)
        self.assertEqual(result.output_id, "z_bus_1")
        self.assertEqual(result.source_bus_id, "1")
        self.assertEqual(result.source_address.address, 14)
        self.assertAlmostEqual(result.z_pu.nominal, 1.03, places=5)
        self.assertGreater(result.worst_case_margin_pu, 0.06)

    def test_fixed_tap_reconstruction_uses_worst_case_tap_and_error(self) -> None:
        result = evaluate_protected_output(_case(), _fixed_tap_asset())

        self.assertEqual(result.status, AssessmentStatus.CERTIFIED)
        self.assertEqual(result.source_bus_id, "1")
        self.assertAlmostEqual(result.z_pu.nominal, 1.03 / 0.99 + 0.001, places=5)
        expected_upper = 1.03 / 0.98 + 0.003
        self.assertAlmostEqual(result.z_pu.upper, expected_upper, places=5)
        self.assertAlmostEqual(result.worst_case_margin_pu, 1.10 - expected_upper, places=5)

    def test_nonpositive_worst_case_margin_is_data_limited_not_safe(self) -> None:
        result = evaluate_protected_output(
            _case(),
            _fixed_tap_asset(threshold=Interval(1.04, 1.08, Unit.PU, nominal=1.06)),
        )

        self.assertEqual(result.status, AssessmentStatus.DATA_LIMITED)
        self.assertFalse(result.is_certifiable)
        self.assertLessEqual(result.worst_case_margin_pu, 0.0)
        self.assertEqual(result.reason, "nonpositive_worst_case_margin")

    def test_collection_evaluation_preserves_asset_order(self) -> None:
        results = evaluate_protected_outputs(_case(), (_direct_asset(), _fixed_tap_asset()))

        self.assertEqual([item.asset_id for item in results], ["P_bus_1", "P_tap_1"])
        self.assertTrue(all(item.is_certifiable for item in results))

    def test_missing_bus_and_custom_output_raise_explicit_errors(self) -> None:
        bad_direct = ProtectedAsset(
            asset_id="P_missing",
            name="Missing bus relay",
            protected_voltage=ProtectedVoltageSpec(
                output_id="z_missing",
                side=MeasurementSide.COLLECTOR_BUS,
                kind=ProtectedVoltageKind.DIRECT_BUS_VOLTAGE,
                protected_bus_id="missing_bus",
            ),
            threshold_pu=Interval.exact(1.10, Unit.PU),
            timing=ProtectionTiming(DelayKind.ZERO_DELAY),
            disconnected_assets=_disconnected(),
        )
        with self.assertRaises(ProtectedOutputError):
            evaluate_protected_output(_case(), bad_direct)

        custom = ProtectedAsset(
            asset_id="P_custom",
            name="Custom output relay",
            protected_voltage=ProtectedVoltageSpec(
                output_id="z_custom",
                side=MeasurementSide.RECONSTRUCTED,
                kind=ProtectedVoltageKind.CUSTOM_OUTPUT,
                expression="custom(y)",
            ),
            threshold_pu=Interval.exact(1.10, Unit.PU),
            timing=ProtectionTiming(DelayKind.ZERO_DELAY),
            disconnected_assets=_disconnected(),
        )
        with self.assertRaises(ProtectedOutputError):
            evaluate_protected_output(_case(), custom)


if __name__ == "__main__":
    unittest.main()

