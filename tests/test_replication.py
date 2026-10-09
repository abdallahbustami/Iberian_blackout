"""Unit tests of the 2000-bus simulation engine (no ANDES run)."""

from __future__ import annotations

import dataclasses
from types import SimpleNamespace

import numpy as np
import pytest

from pa_dvsa.replication.activsg2000_engine import (
    REPORT_BLOCKS,
    VARIANTS,
    BlockSpec,
    DefiniteTimeStage,
    ReplicaConfig,
    _components,
    _new_relay_state,
    accepted_sample,
    advance_definite_time,
    regularize_improper_ieeest,
    select_affected_area,
    variant_config,
)


class _V:
    def __init__(self, values):
        self.v = np.asarray(values)


def _model(**fields):
    ns = SimpleNamespace()
    n = None
    for name, values in fields.items():
        setattr(ns, name, _V(values))
        n = len(values)
    ns.n = int(n or 0)
    return ns


def _run(stage, values, dt=0.1):
    st = _new_relay_state(1)
    ops = []
    for v in values:
        ops.append(bool(advance_definite_time(st, stage, np.array([v]), dt, np.array([True]))[0]))
    return ops, st


def test_definite_time_over_element_times_from_pickup_and_holds_above_dropout():
    stage = DefiniteTimeStage("OV1", "voltage", "over", 1.10, 0.3, 0.98)
    ops, _ = _run(stage, [1.09, 1.10, 1.09, 1.095, 1.10])
    assert ops == [False, False, False, False, True]


def test_definite_time_over_element_resets_below_dropout():
    stage = DefiniteTimeStage("OV1", "voltage", "over", 1.10, 0.3, 0.98)
    ops, _ = _run(stage, [1.11, 1.11, 1.07, 1.11, 1.11, 1.11])
    assert not any(ops)
    ops, st = _run(stage, [1.11, 1.11, 1.07, 1.11, 1.11, 1.11, 1.11])
    assert ops[-1] is True
    assert st.timer[0] == pytest.approx(0.3)


def test_definite_time_under_frequency_element():
    stage = DefiniteTimeStage("UF", "frequency", "under", 0.95, 0.2, 0.999)
    ops, _ = _run(stage, [0.96, 0.949, 0.949, 0.949, 0.96])
    assert ops == [False, False, False, True, False]


def test_inactive_elements_never_time():
    stage = DefiniteTimeStage("OV1", "voltage", "over", 1.10, 0.0, 0.98)
    st = _new_relay_state(2)
    op = advance_definite_time(st, stage, np.array([1.2, 1.2]), 0.1, np.array([True, False]))
    assert op.tolist() == [True, False]


def test_stage_validation():
    with pytest.raises(ValueError):
        DefiniteTimeStage("x", "current", "over", 1.0, 0.1)
    with pytest.raises(ValueError):
        DefiniteTimeStage("x", "voltage", "sideways", 1.0, 0.1)


def test_accepted_sample_ignores_rejected_step_retries():
    last = None
    seen = []
    for t_target, h in [(0.01, 0.01), (0.02, 0.01), (0.019, 0.009), (0.028, 0.009)]:
        is_new, t_acc, dt = accepted_sample(last, t_target, h)
        if is_new:
            last = t_acc
            seen.append((t_acc, round(dt, 9)))
    assert seen == [(0.0, 0.0), (0.01, 0.01), (0.019, 0.009)]


def test_accepted_sample_never_goes_backwards():
    is_new, t_acc, dt = accepted_sample(1.0, 1.005, 0.01)
    assert is_new is False and t_acc == 1.0 and dt == 0.0


def test_report_blocks_use_report_mvar_only_where_given_and_no_inflation():
    assert len(REPORT_BLOCKS) == 16
    assert sum(b.p_mw for b in REPORT_BLOCKS) == pytest.approx(2906.0)
    reported = {b.name: b.q_report_mvar for b in REPORT_BLOCKS if b.q_report_mvar is not None}
    assert reported == {"Granada355": 165.0, "Badajoz582": 165.0, "Badajoz145": 37.0}
    names = {f.name for f in dataclasses.fields(ReplicaConfig)}
    assert not any("q_scale" in n or "absorption_scale" in n for n in names)
    block_fields = {f.name for f in dataclasses.fields(BlockSpec)}
    assert not block_fields & {"pickup_pu", "dwell_s", "tap", "threshold_pu", "report_time_s", "min_trip_time_s"}


def test_no_gating_fields_in_physics_config():
    names = {f.name for f in dataclasses.fields(ReplicaConfig)}
    forbidden = ("require_", "min_trip_time", "min_time", "min_block_time", "after_hvdc", "hvdc")
    assert not [n for n in names if any(tok in n for tok in forbidden)]


def test_uniform_standard_plant_protection():
    stages = {s.name: s for s in ReplicaConfig().plant_stages()}
    assert (stages["OV1"].pickup, stages["OV1"].delay_s) == (1.10, 1.0)
    assert (stages["OV2"].pickup, stages["OV2"].delay_s) == (1.20, 0.16)
    assert stages["UF"].pickup == pytest.approx(0.95) and stages["OF"].pickup == pytest.approx(1.03)
    assert all(s.basis for s in stages.values())


def test_variants_implement_what_their_names_say():
    assert set(VARIANTS) >= {"baseline", "no_ov_protection", "delayed_protection", "preserved_absorption",
                             "voltage_mode_res", "no_pre_cascade_actions"}
    base = variant_config("baseline")

    def names(c):
        return {s.name for s in c.plant_stages()}

    assert {"OV1", "OV2"} <= names(base)
    no_ov = variant_config("no_ov_protection")
    assert not ({"OV1", "OV2"} & names(no_ov)) and {"UV1", "UF", "OF"} <= names(no_ov)
    delayed = {s.name: s for s in variant_config("delayed_protection").plant_stages()}
    assert delayed["OV1"].delay_s == 2.0 and delayed["OV1"].pickup == 1.10 and delayed["OV2"].delay_s == 0.16
    assert variant_config("preserved_absorption").preserved_absorption is True
    vm = variant_config("voltage_mode_res")
    assert vm.voltage_mode_res is True and vm.volt_var_slope > 0
    assert variant_config("no_pre_cascade_actions").pre_cascade_enabled is False
    for name in VARIANTS:
        cfg = variant_config(name)
        for field in ("load_scale", "fixed_pf", "tap_rule", "tap_margin_pu", "affected_area", "blocks",
                      "generator_protection_enabled", "ufls_stages", "separation_angle_deg"):
            assert getattr(cfg, field) == getattr(base, field)


def test_pre_cascade_actions_are_physical_and_documented():
    acts = {a.code: a for a in ReplicaConfig().pre_cascade_actions}
    assert acts["OA3"].kind == "fixed_pf_res_reduction" and acts["OA3"].size == 208.0
    assert {a.kind for a in acts.values()} == {"line_energization", "interchange_reduction", "fixed_pf_res_reduction"}
    assert all(a.description and a.justification for a in acts.values())


def test_regularize_improper_ieeest_only_touches_improper_f2():
    st = SimpleNamespace(
        n=3, idx=_V(["P1", "P2", "P3"]),
        A1=_V([1.0, 1.0, 0.0]), A2=_V([0.01, 0.01, 0.0]),
        A3=_V([0.0, 0.2, 0.0]), A4=_V([0.0, 0.01, 0.0]),
        A5=_V([1.013, 0.3, 0.0]), A6=_V([0.113, 0.02, 0.0]),
    )
    system = SimpleNamespace(is_setup=False, IEEEST=st)
    changes = regularize_improper_ieeest(system)
    assert [c["idx"] for c in changes] == ["P1"]
    assert st.A3.v[0] == pytest.approx(2e-3) and st.A4.v[0] == pytest.approx(1e-6)
    assert st.A3.v[1] == 0.2 and st.A3.v[2] == 0.0
    with pytest.raises(RuntimeError):
        regularize_improper_ieeest(SimpleNamespace(is_setup=True, IEEEST=st))


def test_components_union_find():
    comps = _components([1, 2, 3, 4, 5], [(1, 2), (2, 3), (4, 5)])
    assert [sorted(c) for c in comps] == [[1, 2, 3], [4, 5]]


def test_affected_area_cut_set_is_complete_and_fragments_stay_outside():
    system = SimpleNamespace(
        Bus=_model(idx=[1, 2, 3, 4, 5, 6], area=[1, 1, 1, 2, 2, 1]),
        Line=_model(idx=["a", "b", "c", "d", "e", "f"], bus1=[1, 2, 3, 3, 4, 5], bus2=[2, 3, 1, 4, 5, 6],
                    u=[1, 1, 1, 1, 1, 1]),
    )
    area = select_affected_area(system, 1)
    assert area.buses == {1, 2, 3}
    assert area.fragments_moved_out == [6]
    assert area.cut_branch_ids == ["d"]
    assert area.rest_buses == {4, 5, 6}


def test_affected_area_rejects_incomplete_cut():
    system = SimpleNamespace(
        Bus=_model(idx=[1, 2, 3, 4, 5], area=[1, 1, 2, 2, 3]),
        Line=_model(idx=["a", "b", "c", "d"], bus1=[1, 2, 3, 1], bus2=[2, 3, 4, 5], u=[1, 1, 1, 1]),
    )
    with pytest.raises(RuntimeError):
        select_affected_area(system, 1)


def test_water_fill_capped_respects_floor_cap_and_target():
    from pa_dvsa.replication.activsg2000_engine import water_fill_capped

    p0 = np.array([1.0, 2.0, 4.0])
    floor = np.array([0.5, 0.2, 0.2])
    cap = np.array([1.2, 2.5, 4.4])
    k, p = water_fill_capped(p0, floor, cap, 6.0)
    assert p.sum() == pytest.approx(6.0, abs=1e-9)
    assert np.all(p >= floor - 1e-12) and np.all(p <= cap + 1e-12)
    k, p = water_fill_capped(p0, floor, cap, 7.9)
    assert p[2] == pytest.approx(4.4) and p.sum() == pytest.approx(7.9, abs=1e-9) and k > 1.0
    with pytest.raises(RuntimeError):
        water_fill_capped(p0, floor, cap, 9.0)
    with pytest.raises(RuntimeError):
        water_fill_capped(p0, floor, cap, 0.5)


def test_decommitment_order_and_stop_rule():
    from pa_dvsa.replication.activsg2000_engine import AffectedArea, decommit_area_units

    system = SimpleNamespace(
        GENROU=_model(idx=["G1", "G2", "G3", "G4", "G5"], gen=["P1", "P2", "P3", "P4", "SL"],
                      bus=[1, 2, 3, 4, 5], u=[1.0, 1.0, 1.0, 1.0, 1.0], Sn=[50.0, 50.0, 300.0, 100.0, 10.0]),
        PV=_model(idx=["P1", "P2", "P3", "P4"], u=[1.0, 1.0, 1.0, 1.0]),
        Slack=_model(idx=["SL"]),
    )
    area = AffectedArea(1, {1, 2, 3, 4, 5}, [], [], [], set())
    info = decommit_area_units(system, area, 2.0, 150.0)
    removed = [r["genrou"] for r in info["decommitted"]]
    assert removed[:3] == ["G1", "G2", "G4"]
    assert "G5" not in removed
    online = sum(sn for sn, u in zip(system.GENROU.Sn.v, system.GENROU.u.v) if u > 0.5)
    assert online <= 300.0
    assert all(system.PV.u.v[["P1", "P2", "P3", "P4"].index(r["genrou"].replace("G", "P"))] == 0.0
               for r in info["decommitted"])


def test_plant_ov_modes():
    from pa_dvsa.replication.activsg2000_engine import plant_ov_stages

    fast = plant_ov_stages("fast")
    assert [(s.pickup, s.delay_s) for s in fast] == [(1.10, 0.05)]
    std = plant_ov_stages("standard")
    assert [(s.name, s.pickup, s.delay_s) for s in std] == [("OV1", 1.10, 1.0), ("OV2", 1.20, 0.16)]
    names = {s.name for s in ReplicaConfig(plant_ov_mode="fast").plant_stages()}
    assert "OVF" in names and "OV1" not in names and {"UV1", "UF", "OF"} <= names
    with pytest.raises(ValueError):
        plant_ov_stages("slow")


def test_trip_classification_from_counterfactuals():
    from pa_dvsa.replication.commitment_study import classify

    base = {"block_trip_sequence": [
        {"block": "A", "t": 2.0, "v_pu": 1.1, "p_mw": 100.0, "lost_q_absorption_mvar": 30.0},
        {"block": "B", "t": 3.0, "v_pu": 1.1, "p_mw": 50.0, "lost_q_absorption_mvar": 10.0},
        {"block": "C", "t": 4.0, "v_pu": 1.1, "p_mw": 20.0, "lost_q_absorption_mvar": 5.0}]}
    cf = {"block_trip_sequence": [{"block": "A", "t": 2.0}, {"block": "B", "t": 3.4}]}
    npr = {"per_block": {"A": {"shadow_standard_operate_s": 2.0}, "B": {"shadow_standard_operate_s": None},
                         "C": {"shadow_standard_operate_s": None}}}
    cls = {c["block"]: c["class"] for c in classify(base, cf, npr, "standard")}
    assert cls == {"A": "action-driven", "B": "propagated (MW loss)", "C": "propagated (absorption loss)"}


def test_iberian_options_default_to_the_committed_replica():
    cfg = ReplicaConfig()
    assert cfg.tie_impedance_scale == 1.0
    assert cfg.export_reduction_mw == 0.0
    assert cfg.exciter_completion == "none"
    assert cfg.initiating_block is None


def test_sequence_options_default_to_the_committed_replica():
    cfg = ReplicaConfig()
    assert cfg.initial_mesh_closed == 0
    assert cfg.extra_plant_stages == ()
    assert cfg.area_q_control_tc_s == 0.0
    assert cfg.area_ehv_shunt_mvar == 0.0
    assert cfg.area_uel_tc_s == 0.0


def test_cut_set_scaling_changes_only_series_impedance_of_the_cut_set():
    from pa_dvsa.replication.activsg2000_engine import AffectedArea, scale_cut_set_impedance

    line = SimpleNamespace(x=_V([0.1, 0.2, 0.3]), r=_V([0.01, 0.02, 0.03]), b=_V([0.5, 0.6, 0.7]))
    line.x.v = list(line.x.v)
    line.r.v = list(line.r.v)
    system = SimpleNamespace(Line=line)
    area = AffectedArea(1, {1}, [], ["b"], [1], set())
    assert scale_cut_set_impedance(system, area, 10.0) == 1
    assert line.x.v == pytest.approx([0.1, 2.0, 0.3]) and line.r.v == pytest.approx([0.01, 0.2, 0.03])
    assert list(line.b.v) == pytest.approx([0.5, 0.6, 0.7])
    assert scale_cut_set_impedance(system, area, 1.0) == 0
    with pytest.raises(ValueError):
        scale_cut_set_impedance(system, area, 0.0)


def test_exciter_completion_uses_the_case_generic_sexs_set():
    from pa_dvsa.replication.activsg2000_engine import GENERIC_SEXS, complete_exciters

    added = []
    system = SimpleNamespace(
        GENROU=_model(idx=["G1", "G2"], gen=["P1", "P2"], bus=[1, 2], u=[1.0, 0.0]),
        SEXS=_model(idx=["E1"], syn=["G1"], u=[1.0]),
        add=lambda model, params: added.append((model, params)),
    )
    assert complete_exciters(system, "none")["added"] == 0 and not added
    info = complete_exciters(system, "all")
    assert info["added"] == 1
    model, params = added[0]
    assert model == "SEXS" and params["syn"] == "G2"
    assert {k: params[k] for k in GENERIC_SEXS} == GENERIC_SEXS
    with pytest.raises(ValueError):
        complete_exciters(system, "some")


def test_scrx_completion_reparameterizes_andes_default_sexs_and_completes_the_rest():
    from pa_dvsa.replication.activsg2000_engine import GENERIC_SEXS, SCRX_SEXS, complete_exciters

    added = []
    sexs = SimpleNamespace(n=2, idx=_V(["E1", "E2"]), syn=_V(["G1", "G2"]), u=_V([1.0, 1.0]),
                           **{p: SimpleNamespace(v=[GENERIC_SEXS[p], GENERIC_SEXS[p] * 2.0]) for p in GENERIC_SEXS})
    system = SimpleNamespace(
        GENROU=_model(idx=["G1", "G2", "G3"], gen=["P1", "P2", "P3"], bus=[1, 2, 3], u=[1.0, 1.0, 1.0]),
        SEXS=sexs,
        add=lambda model, params: added.append((model, params)),
    )
    info = complete_exciters(system, "all_scrx")
    assert info["reparameterized_converted_sexs"] == 1
    assert {p: sexs.__dict__[p].v[0] for p in SCRX_SEXS} == SCRX_SEXS
    assert sexs.K.v[1] == GENERIC_SEXS["K"] * 2.0
    assert info["added"] == 1 and added[0][1]["syn"] == "G3"
    assert {k: added[0][1][k] for k in SCRX_SEXS} == SCRX_SEXS


def test_load_models_are_valid_zip_mixes():
    from pa_dvsa.replication.activsg2000_engine import LOAD_MODELS

    for mix in LOAD_MODELS.values():
        assert mix["p2p"] + mix["p2i"] + mix["p2z"] == pytest.approx(1.0)
        assert mix["q2q"] + mix["q2i"] + mix["q2z"] == pytest.approx(1.0)
    assert LOAD_MODELS["z"]["p2z"] == 1.0
