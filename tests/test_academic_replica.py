"""Integration tests of the base state of the 2000-bus system."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

from pa_dvsa.replication.academic_replica import (
    REPORT_ANCHORS,
    SENSITIVITY_CASES,
    VARIANT_ORDER,
    build_academic_replica_config,
)
from pa_dvsa.replication.activsg2000_engine import (
    ReplicaConfig,
    _components,
    _pos,
    build_armed_replica,
    build_replica_system,
    islanding_check,
    run_replica,
    variant_config,
)

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "revised" / "activsg2000"
RUN_SLOW = os.environ.get("PA_DVSA_SLOW", "") not in ("", "0")


def test_report_anchors_include_required_iberian_events():
    codes = {a.code for a in REPORT_ANCHORS}
    assert {"E3_Granada", "E4a_Badajoz", "E4b_Badajoz", "E5_multi_site", "France_AC", "blackout"} <= codes


def test_academic_config_is_the_physics_variant():
    cfg = build_academic_replica_config()
    assert cfg.mode == "physics" and cfg.variant == "baseline"
    assert cfg == variant_config("baseline")
    assert set(VARIANT_ORDER) <= set(__import__("pa_dvsa.replication.activsg2000_engine", fromlist=["VARIANTS"]).VARIANTS)
    assert all(set(v) <= {"tap_margin_pu", "fixed_pf", "tap_rule", "tap_fixed", "preserved_absorption"}
               for v in SENSITIVITY_CASES.values())


@pytest.fixture(scope="module")
def base_rs():
    return build_replica_system(ReplicaConfig(), armed=False, quiet=True)


def test_blocks_are_trippable_pvd1_plants_off_regulated_buses(base_rs):
    rs = base_rs
    ss = rs.system
    gen_buses = {int(b) for b in ss.PV.bus.v} | {int(b) for b in ss.Slack.bus.v}
    block_pv = {b.pv_id for b in rs.blocks}
    other_gen_buses = {int(ss.PV.bus.v[i]) for i in range(ss.PV.n) if str(ss.PV.idx.v[i]) not in block_pv}
    hv = [b.hv_bus for b in rs.blocks]
    assert len(set(hv)) == 16
    assert not set(hv) & (other_gen_buses | {int(b) for b in ss.Slack.bus.v})
    assert gen_buses
    for b in rs.blocks:
        k = _pos(ss.PVD1, b.pvd1_id)
        assert int(ss.PVD1.bus.v[k]) == b.collector_bus and float(ss.PVD1.recflag.v[k]) == 0.0
        assert float(ss.PV.u.v[_pos(ss.PV, b.pv_id)]) == 0.0


def test_uniform_tap_rule_and_scheduled_q(base_rs):
    rs = base_rs
    ss = rs.system
    taps = {round(float(ss.Line.tap.v[_pos(ss.Line, b.gsu_id)]), 9) for b in rs.blocks}
    assert len(taps) == 1
    assert max(b.z0_pu for b in rs.blocks) == pytest.approx(1.10 - rs.cfg.tap_margin_pu, abs=5e-4)
    base = float(ss.config.mva)
    for b in rs.blocks:
        k = _pos(ss.PVD1, b.pvd1_id)
        assert float(ss.PVD1.qref0.v[k]) * base == pytest.approx(b.q_sched_mvar, abs=0.5)
        assert float(ss.PVD1.pref0.v[k]) * base == pytest.approx(b.p_sched_mw, abs=0.5)
        if b.spec.q_report_mvar is None:
            assert -b.q_sched_mvar / b.p_sched_mw == pytest.approx(np.tan(np.arccos(rs.cfg.fixed_pf)))
        else:
            assert -b.q_sched_mvar == pytest.approx(b.spec.q_report_mvar)


def test_operating_point_is_an_equilibrium(base_rs):
    assert base_rs.base_notes["init_max_abs_dxdt"] < 1e-8


def test_cut_set_forms_exactly_two_electrical_islands(base_rs):
    rs = base_rs
    ss = rs.system
    line = ss.Line
    cut = set(rs.area.cut_branch_positions)
    edges = [(int(line.bus1.v[i]), int(line.bus2.v[i])) for i in range(line.n)
             if float(line.u.v[i]) > 0.5 and i not in cut]
    comps = _components([int(b) for b in ss.Bus.idx.v], edges)
    assert len(comps) == 2
    area_side = next(c for c in comps if rs.blocks[0].collector_bus in c)
    assert rs.area.buses <= area_side


def test_improper_pss_filters_regularized(base_rs):
    st = base_rs.system.IEEEST
    bad = [i for i in range(st.n) if st.A3.v[i] == 0 and st.A4.v[i] == 0 and (st.A5.v[i] != 0 or st.A6.v[i] != 0)]
    assert not bad


def test_control_run_is_flat_and_event_free():
    res = run_replica(variant_config("no_pre_cascade_actions", tf_s=1.0), quiet=True)
    assert res.tds_ok
    v0 = res.monitor.system_rows[0]
    v1 = res.monitor.system_rows[-1]
    for key in ("vmax_pu", "vmin_pu", "collector_vmax_online_pu", "zone_hv_vmax_pu"):
        assert abs(v1[key] - v0[key]) < 1e-6
    assert res.monitor.events == []


def test_line_energization_raises_zone_voltage_and_is_logged():
    res = run_replica(variant_config("baseline", tf_s=2.0), quiet=True)
    rows = res.monitor.system_rows
    before = next(r for r in rows if r["time_s"] >= 0.9)["zone_hv_vmax_pu"]
    after = rows[-1]["zone_hv_vmax_pu"]
    assert after > before + 1e-3
    assert [e.code for e in res.monitor.events] == ["OA1a"]


def test_line_switch_in_sets_the_series_admittance(base_rs):
    from pa_dvsa.replication.activsg2000_engine import set_line_status

    ss = base_rs.system
    line = ss.Line
    for lid, src in zip(base_rs.mesh_line_ids, base_rs.mesh_source_lines):
        p, q = _pos(line, lid), _pos(line, src)
        saved = (float(line.u.v[p]), float(line.ghk.v[p]), float(line.bhk.v[p]))
        assert saved == (0.0, 0.0, 0.0)
        try:
            set_line_status(ss, lid, 1.0)
            for attr in ("u", "ghk", "bhk", "gh", "bh", "gk", "bk", "itap", "itap2"):
                assert float(getattr(line, attr).v[p]) == pytest.approx(float(getattr(line, attr).v[q]), rel=1e-12, abs=1e-15)
            set_line_status(ss, lid, 0.0)
            assert (float(line.u.v[p]), float(line.ghk.v[p]), float(line.bhk.v[p])) == (0.0, 0.0, 0.0)
        finally:
            line.u.v[p], line.ghk.v[p], line.bhk.v[p] = saved


@pytest.mark.slow
@pytest.mark.skipif(not RUN_SLOW, reason="set PA_DVSA_SLOW=1 (TDS + equilibrium polish, ~2 min)")
def test_switched_in_oa1_circuits_reach_the_state_with_the_circuits_closed_in_the_case_data():
    from dataclasses import replace as dc_replace

    from pa_dvsa.replication.activsg2000_engine import (PhysicsMonitor, default_pre_cascade_actions,
                                                         polish_equilibrium, set_line_status)

    oa1 = tuple(a for a in default_pre_cascade_actions() if a.kind == "line_energization")
    cfg = dc_replace(ReplicaConfig(), pre_cascade_actions=oa1, plant_ov_enabled=False, tf_s=12.0,
                     variant="test_oa1_switch_in")
    rs = build_replica_system(cfg, armed=True, quiet=True)
    ss = rs.system
    line = ss.Line
    closed = {lid: (float(line.ghk.v[_pos(line, lid)]), float(line.bhk.v[_pos(line, lid)])) for lid in rs.mesh_line_ids}
    assert all(g != 0.0 and b != 0.0 for g, b in closed.values())
    v_ref = np.asarray(ss.Bus.v.v, dtype=float).copy()
    rs.armed = False
    mon = PhysicsMonitor(rs)
    state = {"opened": False}

    def cb(t, system):
        if not state["opened"]:
            for lid in rs.mesh_line_ids:
                set_line_status(system, lid, 0.0)
            system.TDS.custom_event = True
            state["opened"] = True
        mon(t, system)

    ss.TDS.callpert = cb
    assert ss.TDS.run(no_summary=True)
    assert [e.code for e in mon.events] == ["OA1a", "OA1b"]
    for lid, (g, b) in closed.items():
        assert float(line.ghk.v[_pos(line, lid)]) == pytest.approx(g, rel=1e-12)
        assert float(line.bhk.v[_pos(line, lid)]) == pytest.approx(b, rel=1e-12)
    pol = polish_equilibrium(ss, h0=1.0, h_max=1e6, tol=1e-10, max_iter=120, inner_iters=5)
    assert pol["converged"]
    assert abs(pol["secondary_control_mw"]) < 1.0
    assert np.max(np.abs(np.asarray(ss.Bus.v.v, dtype=float) - v_ref)) < 1e-6


def test_armed_replica_for_screening():
    armed = build_armed_replica(quiet=True)
    assert len(armed.protected_blocks) == 16
    assert armed.system.TDS.initialized
    fams = {e.family for e in armed.events}
    assert {"plant_trip", "fixed_pf_ramp", "load_disconnection", "generator_trip", "generation_redispatch",
            "line_energization", "capacitor_switching"} <= fams
    for e in armed.events:
        if e.family == "fixed_pf_ramp":
            p, q = e.mutations
            assert p.attr == "pref0" and q.attr == "qref0"
            assert p.value / e.metadata["pref0"] == pytest.approx(0.7)
            assert q.value / e.metadata["qref0"] == pytest.approx(0.7)
        if e.family == "plant_trip":
            assert e.kind == "switch" and e.disconnects_assets
        if e.family == "line_energization":
            lid = e.mutations[0].device_idx
            y = 1.0 / complex(float(armed.system.Line.r.v[_pos(armed.system.Line, lid)]) + 1e-8,
                              float(armed.system.Line.x.v[_pos(armed.system.Line, lid)]) + 1e-8)
            assert {(m.attr, m.value) for m in e.mutations} == {("u", 1.0), ("ghk", y.real), ("bhk", y.imag)}
        if e.family == "generator_trip":
            attrs = {(m.model, m.attr) for m in e.mutations if m.model != "GENROU"}
            assert {a for _, a in attrs} == {"u", "ue"}
            assert {mdl for mdl, a in attrs if a == "u"} == {mdl for mdl, a in attrs if a == "ue"}
    assert armed.metadata["min_margin_to_ov1_pu"] < ReplicaConfig().tap_margin_pu
    for pb in armed.protected_blocks:
        assert pb.pickup_pu == 1.10 and pb.dwell_s == 1.0 and pb.q_mvar < 0


@pytest.mark.slow
@pytest.mark.skipif(not RUN_SLOW, reason="set PA_DVSA_SLOW=1 (forced islanding runs take several minutes)")
def test_forced_islanding_check():
    out = islanding_check(tf_s=4.0, quiet=True)
    assert out["A_base_export"]["islands_after_opening"] == 2
    assert out["B_balanced"]["islands_after_opening"] == 2
    assert out["C_single_machine"]["tds_ok"] and out["C_single_machine"]["islands_after"] == 2


def test_small_island_capability():
    from pa_dvsa.replication.activsg2000_engine import small_island_check

    out = small_island_check(tf_s=2.0, quiet=True)
    assert out["found"] and out["tds_ok"] and out["islands_after"] == 2


def _load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.mark.skipif(not (RESULTS / "baseline" / "summary.json").exists(), reason="revised replica not generated")
def test_generated_artifacts_only_contain_measured_triggers():
    allowed_prefix = ("plant_", "generator_", "island_coi_underfrequency", "coi_angle_change",
                      "island_collapse_criterion", "scheduled", "numerical")
    for variant in VARIANT_ORDER:
        d = RESULTS / variant
        if not (d / "summary.json").exists():
            continue
        summary = _load(d / "summary.json")
        events = _load(d / "events.json")
        design = _load(d / "case_design.json")
        stages = {s["name"]: s for s in design["plant_protection"]}
        for e in events:
            assert e["trigger"].startswith(allowed_prefix), e
            if e["category"] == "plant_trip":
                st = stages[e["trigger"].replace("plant_", "")]
                v = e["measured"]["v_pu"]
                if st["direction"] == "over" and st["quantity"] == "voltage":
                    assert v >= st["pickup"] * st["dropout_ratio"] - 1e-9
        trips = [e for e in events if e["category"] == "plant_trip"]
        assert summary["block_trips"] == len(trips)
        times = [e["time_s"] for e in events]
        assert times == sorted(times)
        if variant == "no_ov_protection":
            assert not [e for e in trips if e["trigger"].startswith("plant_OV")]
        if variant == "no_pre_cascade_actions":
            assert events == []
        assert summary["blackout_emerged"] == any(e["category"] == "island_collapse" for e in events)


@pytest.mark.skipif(not (RESULTS / "variants_summary.json").exists(), reason="revised replica not generated")
def test_variant_summary_is_complete():
    rows = _load(RESULTS / "variants_summary.json")
    names = {r["name"] for r in rows}
    assert set(VARIANT_ORDER) <= names


def test_commitment_level_c15_is_an_equilibrium_with_documented_decommitment():
    cfg = ReplicaConfig(commitment_c=1.5, dispatch_rule="must_run")
    rs = build_replica_system(cfg, quiet=True)
    cm = rs.base_notes["commitment"]
    assert cm["area_online_sync_mva"] <= 1.5 * cm["dispatch_mw_for_rule"] + 1e-6
    sizes = [r["sn_mva"] for r in cm["decommitted"]]
    assert sizes == sorted(sizes)
    assert rs.base_notes["init_max_abs_dxdt"] < 1e-8
    assert abs(__import__("pa_dvsa.replication.activsg2000_engine", fromlist=["x"]).area_net_export_mw(rs)
               - 4251.0) < 10.0
    assert max(b.z0_pu for b in rs.blocks) == pytest.approx(1.10 - cfg.tap_margin_pu, abs=5e-4)


@pytest.mark.slow
@pytest.mark.skipif(not RUN_SLOW, reason="set PA_DVSA_SLOW=1 (dynamic arming + equilibrium polish, ~2-4 min)")
def test_dynamic_equilibrium_armed_state_matches_plateau():
    from pa_dvsa.replication.activsg2000_engine import max_abs_dxdt

    armed = build_armed_replica(method="dynamic_equilibrium", settle_s=40.0, quiet=True)
    md = armed.metadata
    assert md["polish"]["converged"] and md["polish"]["max_abs_f"] < 1e-9 and md["polish"]["max_abs_g"] < 1e-9
    assert max_abs_dxdt(armed.rs)[0] < 1e-9
    assert md["plateau_max_abs_dv_pu"] < 1e-3
    assert md["equilibrium_omega_range_pu"][0] == pytest.approx(1.0, abs=1e-9)
