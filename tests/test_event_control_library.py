from __future__ import annotations

from pathlib import Path
import unittest

from pa_dvsa.andes_adapter import AndesCase, AndesCaseSpec
from pa_dvsa.data_model import (
    ActionDirection,
    AssessmentStatus,
    AssetRef,
    ControlKind,
    EventKind,
    FixedPowerFactorSpec,
    Interval,
    ProfileKind,
    ResponseEnvelope,
    TimeProfile,
    Unit,
)
from pa_dvsa.event_control_library import (
    asset_operating_point,
    build_fixed_pf_ramp_event,
    build_line_topology_event,
    build_load_or_der_disconnection_event,
    build_reactive_control,
    build_shunt_switch_event,
    build_trip_event,
    control_channel_for_horizon,
    fixed_pf_delta_q_mvar,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _case() -> AndesCase:
    return AndesCase.load(
        AndesCaseSpec("ieee14", setup=True, run_pflow=True),
        project_root=PROJECT_ROOT,
    )


def _dynamic_case() -> AndesCase:
    return AndesCase.load(
        AndesCaseSpec("ieee39", setup=True, run_pflow=True),
        project_root=PROJECT_ROOT,
    )


class EventControlLibraryTests(unittest.TestCase):
    def test_asset_operating_point_uses_signed_injection_convention(self) -> None:
        case = _case()
        pv = asset_operating_point(case, AssetRef("2", model_family="PV"))
        slack = asset_operating_point(case, AssetRef("1", model_family="Slack"))
        load = asset_operating_point(case, AssetRef("PQ_1", model_family="PQ"))
        shunt = asset_operating_point(case, AssetRef("Shunt_1", model_family="Shunt"))

        self.assertGreater(pv.p_injection_mw, 0.0)
        self.assertGreater(pv.q_injection_mvar, 0.0)
        self.assertEqual(pv.q_absorption_mvar, 0.0)

        self.assertGreater(slack.p_injection_mw, 0.0)
        self.assertLess(slack.q_injection_mvar, 0.0)
        self.assertGreater(slack.q_absorption_mvar, 0.0)

        self.assertLess(load.p_injection_mw, 0.0)
        self.assertLess(load.q_injection_mvar, 0.0)
        self.assertGreater(load.q_absorption_mvar, 0.0)

        self.assertGreater(shunt.q_injection_mvar, 0.0)

    def test_trip_event_computes_lost_absorption_and_voltage_raising_q(self) -> None:
        channel = build_trip_event(
            _case(),
            "trip_slack_1",
            "Trip absorbing slack source",
            (AssetRef("1", model_family="Slack"),),
            kind=EventKind.GENERATOR_TRIP,
        )

        self.assertGreater(channel.lost_p_injection_mw, 80.0)
        self.assertGreater(channel.lost_reactive_absorption_mvar, 20.0)
        self.assertAlmostEqual(
            channel.voltage_raising_q_mvar,
            channel.lost_reactive_absorption_mvar,
            places=6,
        )
        self.assertLess(channel.delta_p_injection_mw, 0.0)
        self.assertGreater(channel.delta_q_injection_mvar, 0.0)
        self.assertTrue(channel.mode_update_required)
        self.assertTrue(channel.recompute_operating_point)

    def test_trip_event_can_follow_dynamic_generator_ownership(self) -> None:
        channel = build_trip_event(
            _dynamic_case(),
            "trip_genrou_1",
            "Trip dynamic synchronous generator",
            (AssetRef("GENROU_1", model_family="GENROU"),),
            kind=EventKind.GENERATOR_TRIP,
        )

        self.assertIn(channel.affected_operating_points[0].source_model, {"PV", "Slack"})
        self.assertNotEqual(channel.affected_operating_points[0].source_model, "GENROU")
        self.assertGreater(abs(channel.delta_p_injection_mw), 1.0)
        self.assertTrue(channel.mode_update_required)

    def test_trip_of_q_injecting_generator_is_not_voltage_raising_by_q_channel(self) -> None:
        channel = build_trip_event(
            _case(),
            "trip_pv_2",
            "Trip Q-injecting PV source",
            (AssetRef("2", model_family="PV"),),
            kind=EventKind.GENERATOR_TRIP,
        )

        self.assertGreater(channel.lost_p_injection_mw, 0.0)
        self.assertLess(channel.delta_q_injection_mvar, 0.0)
        self.assertEqual(channel.voltage_raising_q_mvar, 0.0)
        self.assertEqual(channel.lost_reactive_absorption_mvar, 0.0)

    def test_fixed_pf_ramp_sign_convention(self) -> None:
        absorbing = FixedPowerFactorSpec(Interval.exact(0.98), reactive_sign=-1)
        injecting = FixedPowerFactorSpec(Interval.exact(0.98), reactive_sign=1)

        absorbing_down = fixed_pf_delta_q_mvar(-500.0, absorbing)
        injecting_down = fixed_pf_delta_q_mvar(-500.0, injecting)
        self.assertGreater(absorbing_down, 100.0)
        self.assertLess(injecting_down, -100.0)

        channel = build_fixed_pf_ramp_event(
            "fixed_pf_down",
            "Absorbing fixed-PF source ramps down",
            Interval.exact(-500.0, Unit.MW),
            absorbing,
            profile=TimeProfile(ProfileKind.RAMP, duration_s=48.0),
        )
        self.assertGreater(channel.voltage_raising_q_mvar, 100.0)
        self.assertGreater(channel.event.voltage_raising_q_mvar.nominal, 100.0)
        self.assertIn("Q = sigma", channel.event.metadata["equation"])

        lowering = build_fixed_pf_ramp_event(
            "fixed_pf_injecting_down",
            "Injecting fixed-PF source ramps down",
            Interval.exact(-500.0, Unit.MW),
            injecting,
        )
        self.assertLess(lowering.delta_q_injection_mvar, 0.0)
        self.assertEqual(lowering.voltage_raising_q_mvar, 0.0)

    def test_shunt_switch_is_horizon_aware_mode_update(self) -> None:
        fast = build_shunt_switch_event(
            _case(),
            "disconnect_capacitor",
            "Disconnect capacitor shunt",
            AssetRef("Shunt_1", model_family="Shunt"),
            connect=False,
            actuation_delay_s=Interval.exact(0.05, Unit.SECOND),
            horizon_s=0.2,
        )
        self.assertTrue(fast.active_within_horizon)
        self.assertTrue(fast.mode_update_required)
        self.assertTrue(fast.recompute_operating_point)
        self.assertLess(fast.delta_q_injection_mvar, 0.0)
        self.assertEqual(fast.voltage_raising_q_mvar, 0.0)

        slow = build_shunt_switch_event(
            _case(),
            "slow_disconnect_capacitor",
            "Slow shunt action",
            AssetRef("Shunt_1", model_family="Shunt"),
            connect=False,
            actuation_delay_s=Interval.exact(10.0, Unit.SECOND),
            horizon_s=0.2,
        )
        self.assertFalse(slow.active_within_horizon)
        self.assertFalse(slow.mode_update_required)
        self.assertEqual(slow.delta_q_injection_mvar, 0.0)
        self.assertEqual(slow.reason, "shunt_switch_slower_than_horizon")

    def test_line_topology_action_requires_new_mode(self) -> None:
        channel = build_line_topology_event(
            "mesh_line_1",
            "Energize parallel line",
            AssetRef("Line_1", model_family="Line"),
            energize=True,
        )

        self.assertTrue(channel.mode_update_required)
        self.assertTrue(channel.recompute_operating_point)
        self.assertEqual(channel.voltage_raising_q_mvar, 0.0)
        self.assertEqual(
            channel.event.metadata["primary_representation"],
            "new_mode_recompute_operating_point_and_jacobian",
        )

    def test_load_shedding_disturbance_is_active_reactive_profile(self) -> None:
        channel = build_load_or_der_disconnection_event(
            _case(),
            "shed_pq_1",
            "Shed one PQ load block",
            (AssetRef("PQ_1", model_family="PQ"),),
            kind=EventKind.LOAD_PUMP_SHEDDING,
            profile=TimeProfile(ProfileKind.STEP, start_s=0.1),
        )

        self.assertGreater(channel.delta_p_injection_mw, 0.0)
        self.assertGreater(channel.delta_q_injection_mvar, 0.0)
        self.assertGreater(channel.voltage_raising_q_mvar, 0.0)
        self.assertEqual(channel.event.profile.start_s, 0.1)

    def test_controls_are_excluded_when_slower_than_relay_horizon(self) -> None:
        fast_control = build_reactive_control(
            "statcom_fast",
            "Fast STATCOM absorption",
            ControlKind.STATCOM_SVC,
            ActionDirection.ABSORB_REACTIVE,
            response=ResponseEnvelope(
                delay_s=Interval(0.02, 0.05, Unit.SECOND),
                lower_bound_scale=Interval(0.7, 1.0, nominal=0.8),
                certified=True,
            ),
            alpha_limits=Interval(0.0, 70.0, Unit.MVAR, nominal=0.0),
            magnitude_mvar=Interval.exact(70.0, Unit.MVAR),
        )
        slow_control = build_reactive_control(
            "manual_reactor",
            "Manual reactor switching",
            ControlKind.SHUNT_ACTION,
            ActionDirection.ABSORB_REACTIVE,
            response=ResponseEnvelope(
                delay_s=Interval(5.0, 10.0, Unit.SECOND),
                lower_bound_scale=Interval.exact(1.0),
                certified=True,
            ),
            alpha_limits=Interval(0.0, 100.0, Unit.MVAR, nominal=0.0),
            magnitude_mvar=Interval.exact(100.0, Unit.MVAR),
        )

        fast_channel = control_channel_for_horizon(fast_control, horizon_s=0.2)
        slow_channel = control_channel_for_horizon(slow_control, horizon_s=0.2)

        self.assertTrue(fast_channel.available_for_horizon)
        self.assertEqual(fast_channel.status, AssessmentStatus.CERTIFIED)
        self.assertFalse(slow_channel.available_for_horizon)
        self.assertEqual(slow_channel.reason, "control_slower_than_horizon")


if __name__ == "__main__":
    unittest.main()
