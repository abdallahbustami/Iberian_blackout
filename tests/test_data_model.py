from __future__ import annotations

from pathlib import Path
import sys
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pa_dvsa.data_model import (
    ActionDirection,
    AssessmentStatus,
    AssetRef,
    AssetStatus,
    AssetStatusSpec,
    CandidateControl,
    CandidateEvent,
    CascadeLayer,
    CascadeResult,
    ChannelAssessment,
    ComplianceEnvelope,
    ControlKind,
    ControlMode,
    ControllerModeState,
    ControllerResponseUncertainty,
    DataModelError,
    DelayKind,
    DisconnectedAsset,
    EventKind,
    FixedPowerFactorSpec,
    Interval,
    LimiterStateSpec,
    LimiterStatus,
    LocationSpec,
    MitigationResult,
    ModeState,
    ParameterBound,
    ProfileKind,
    ProtectedAsset,
    ProtectedVoltageKind,
    ProtectedVoltageSpec,
    ProtectionStateSpec,
    ProtectionStatus,
    ProtectionTiming,
    ResponseEnvelope,
    ShuntStatusSpec,
    TapStatusSpec,
    TimeGridControlAuthority,
    TimeProfile,
    Unit,
    UncertaintySet,
    ValidationResult,
    VoltageExcursion,
    WindowMapResult,
)


def protected_asset() -> ProtectedAsset:
    generator = AssetRef("G_Granada_like", model_family="GENROU", bus_id="collector_1")
    disconnected = DisconnectedAsset(
        generator,
        p_injection_mw=Interval.exact(355.0, Unit.MW),
        q_absorption_mvar=Interval(150.0, 180.0, Unit.MVAR, nominal=165.0),
    )
    voltage = ProtectedVoltageSpec(
        output_id="z_collector_1",
        side="collector_bus",
        kind=ProtectedVoltageKind.FIXED_TAP_RECONSTRUCTION,
        transmission_bus_id="400kV_1",
        protected_bus_id="collector_1",
        tap_ratio=Interval(0.98, 1.0, Unit.PU, nominal=0.99),
        reconstruction_error_pu=Interval(-0.005, 0.005, Unit.PU, nominal=0.001),
    )
    return ProtectedAsset(
        asset_id="P_collector_1",
        name="Collector 1 overvoltage relay",
        protected_voltage=voltage,
        threshold_pu=Interval.exact(1.10, Unit.PU),
        timing=ProtectionTiming(
            DelayKind.DWELL,
            dwell_time_s=Interval.exact(0.15, Unit.SECOND),
        ),
        disconnected_assets=(disconnected,),
        q_absorption_mvar=Interval(150.0, 180.0, Unit.MVAR, nominal=165.0),
        tags=("ieee39_replica", "collector"),
    )


class CoreDataModelTests(unittest.TestCase):
    def test_protected_asset_serializes_relay_side_voltage(self) -> None:
        asset = protected_asset()
        payload = asset.to_dict()

        self.assertEqual(payload["protected_voltage"]["output_id"], "z_collector_1")
        self.assertEqual(payload["protected_voltage"]["side"], "collector_bus")
        self.assertEqual(payload["protected_voltage"]["kind"], "fixed_tap_reconstruction")
        self.assertEqual(payload["timing"]["kind"], "dwell")
        self.assertEqual(payload["disconnected_assets"][0]["q_absorption_mvar"]["nominal"], 165.0)

    def test_interval_and_timing_validation(self) -> None:
        with self.assertRaises(DataModelError):
            Interval(2.0, 1.0)
        with self.assertRaises(DataModelError):
            ProtectionTiming(DelayKind.ZERO_DELAY, delay_s=Interval.exact(0.1, Unit.SECOND))
        with self.assertRaises(DataModelError):
            ProtectedVoltageSpec(
                output_id="bad",
                side="collector_bus",
                kind=ProtectedVoltageKind.FIXED_TAP_RECONSTRUCTION,
                transmission_bus_id="bus",
            )
        with self.assertRaises(DataModelError):
            CandidateEvent(
                event_id="bad_enum",
                kind="not_a_real_event",
                name="Bad enum",
            )

    def test_candidate_event_and_control_validation(self) -> None:
        fp = FixedPowerFactorSpec(Interval.exact(0.98), reactive_sign=-1)
        event = CandidateEvent(
            event_id="fixed_pf_ramp_1",
            kind=EventKind.FIXED_PF_RAMP,
            name="Fixed-PF renewable ramp",
            fixed_pf=fp,
            delta_p_mw=Interval(-500.0, -450.0, Unit.MW, nominal=-500.0),
            profile=TimeProfile(ProfileKind.RAMP, duration_s=48.0),
        )
        self.assertGreater(event.fixed_pf.nominal_kappa, 0.2)

        with self.assertRaises(DataModelError):
            CandidateEvent(
                event_id="bad",
                kind=EventKind.FIXED_PF_RAMP,
                name="Missing fixed PF",
            )

        control = CandidateControl(
            control_id="statcom_1_absorb",
            kind=ControlKind.STATCOM_SVC,
            name="STATCOM absorption command",
            direction=ActionDirection.ABSORB_REACTIVE,
            mode=ControlMode.VOLTAGE,
            response=ResponseEnvelope(
                delay_s=Interval(0.02, 0.05, Unit.SECOND, nominal=0.03),
                rise_time_s=Interval(0.05, 0.15, Unit.SECOND, nominal=0.10),
                lower_bound_scale=Interval(0.7, 1.0, nominal=0.8),
                certified=True,
            ),
            alpha_limits=Interval(0.0, 70.0, Unit.MVAR, nominal=0.0),
            magnitude_mvar=Interval.exact(70.0, Unit.MVAR),
        )
        self.assertTrue(control.available)

    def test_mode_state_detects_duplicate_ids(self) -> None:
        gen = AssetRef("G1", model_family="GENROU")
        mode = ModeState(
            mode_id="m0",
            topology_id="base",
            connected_assets=(AssetStatusSpec(gen, AssetStatus.IN_SERVICE),),
            shunts=(ShuntStatusSpec("R1", AssetStatus.IN_SERVICE, b_pu=-0.1),),
            taps=(TapStatusSpec("T1", Interval.exact(0.99, Unit.PU)),),
            controller_modes=(ControllerModeState("AVR1", ControlMode.VOLTAGE),),
            limiter_states=(LimiterStateSpec("UEL1", LimiterStatus.INACTIVE),),
            protection_states=(ProtectionStateSpec("P1", ProtectionStatus.ARMED),),
        )
        self.assertEqual(mode.mode_id, "m0")

        with self.assertRaises(DataModelError):
            ModeState(
                mode_id="bad",
                topology_id="base",
                controller_modes=(
                    ControllerModeState("AVR1", ControlMode.VOLTAGE),
                    ControllerModeState("AVR1", ControlMode.REACTIVE_POWER),
                ),
            )

    def test_uncertainty_set_has_required_channels(self) -> None:
        uncertainty = UncertaintySet(
            set_id="theta_1",
            thresholds_pu=(ParameterBound("P1", "V_trip", Interval(1.08, 1.12, Unit.PU)),),
            tap_ratios=(ParameterBound("T1", "n", Interval(0.98, 1.00, Unit.PU)),),
            delays_s=(ParameterBound("P1", "delay", Interval(0.1, 0.2, Unit.SECOND)),),
            q_absorption_mvar=(
                ParameterBound("G1", "Q_abs", Interval(150.0, 180.0, Unit.MVAR)),
            ),
            controller_responses=(
                ControllerResponseUncertainty(
                    "statcom_1_absorb",
                    delay_s=Interval(0.02, 0.05, Unit.SECOND),
                    lower_bound_scale=Interval(0.7, 1.0),
                ),
            ),
            compliance_envelopes=(
                ComplianceEnvelope("G_PO74", Interval(0.5, 0.75), Interval(0.5, 1.0)),
            ),
            reconstruction_errors_pu=(
                ParameterBound("z_collector_1", "epsilon", Interval(0.0, 0.005, Unit.PU)),
            ),
        )
        self.assertEqual(uncertainty.to_dict()["set_id"], "theta_1")

    def test_result_objects_validate_shapes(self) -> None:
        result = WindowMapResult(
            mode_id="m0",
            protected_asset_ids=("P1", "P2"),
            event_ids=("E1",),
            control_ids=("C1", "C2"),
            horizons_s=(0.2, 0.3),
            k_pickup_upper=((0.8,), (1.2,)),
            k_trip_upper=((0.6,), (1.0,)),
            control_authority_lower=(
                TimeGridControlAuthority(0.1, ((0.2, 0.0), (0.1, 0.3))),
            ),
            channel_assessments=(
                ChannelAssessment("P1", "E1", AssessmentStatus.CERTIFIED, "lipschitz_upper"),
            ),
            status=AssessmentStatus.CERTIFIED,
        )
        self.assertEqual(result.k_pickup_upper[1][0], 1.2)

        with self.assertRaises(DataModelError):
            WindowMapResult(
                mode_id="bad",
                protected_asset_ids=("P1", "P2"),
                event_ids=("E1",),
                control_ids=("C1",),
                horizons_s=(0.2, 0.3),
                k_pickup_upper=((0.8, 0.1), (1.2, 0.2)),
                k_trip_upper=((0.6,), (1.0,)),
            )
        with self.assertRaises(DataModelError):
            WindowMapResult(
                mode_id="bad_negative",
                protected_asset_ids=("P1",),
                event_ids=("E1",),
                control_ids=(),
                horizons_s=(0.2,),
                k_pickup_upper=((-0.1,),),
                k_trip_upper=((0.0,),),
            )

    def test_cascade_mitigation_validation_results_serialize(self) -> None:
        cascade = CascadeResult(
            mode_id="m0",
            seed_ids=("E1",),
            layers=(CascadeLayer(0, ("P1",), ("P1",)),),
            fixed_point=("P1",),
            no_secondary_certified=True,
            status=AssessmentStatus.CERTIFIED,
        )
        mitigation = MitigationResult(
            mode_id="m0",
            seed_ids=("E1",),
            feasible=True,
            objective_value=42.0,
            slack_eta=0.0,
            selections=(),
            status=AssessmentStatus.CERTIFIED,
        )
        validation = ValidationResult(
            scenario_id="case_b_seed_e1",
            mode_id="m0",
            seed_ids=("E1",),
            predicted_trips=(),
            simulated_trips=(),
            false_positive_trips=(),
            false_negative_trips=(),
            max_excursions=(VoltageExcursion("P1", 1.08, 1.10, 0.12),),
            status=AssessmentStatus.CERTIFIED,
        )
        self.assertEqual(cascade.to_dict()["fixed_point"], ["P1"])
        self.assertEqual(mitigation.to_dict()["slack_eta"], 0.0)
        self.assertEqual(validation.to_dict()["max_excursions"][0]["max_voltage_pu"], 1.08)


if __name__ == "__main__":
    unittest.main()
