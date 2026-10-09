"""Stressed state of the 2000-bus system (reactive surplus, slow reactive control, voltage-dependent load,
embedded PV) and the output reduction of the final minute.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from pa_dvsa.replication import activsg2000_engine as E
from pa_dvsa.replication.iberian_state import iberian_state_config

GRANADA_OVI = E.DefiniteTimeStage(
    "OVI", "voltage", "over", 1.10, 0.0, 0.98,
    "Granada transformer OV protection, 220 kV side: 110 % Un, t = 0 s (factual report p. 108)",
)


def _ov(name: str, pickup: float, delay: float, src: str) -> E.DefiniteTimeStage:
    return E.DefiniteTimeStage(name, "voltage", "over", pickup, delay, 0.98, src)


DOCUMENTED_STAGES: tuple[tuple[str, E.DefiniteTimeStage], ...] = (
    ("Granada355", GRANADA_OVI),
    ("Badajoz145", _ov("OVD1", 1.089, 1.5, "Badajoz 145 MW: 1.089 pu, 1.5 s (p. 135)")),
    ("Badajoz145", _ov("OVD2", 1.1385, 0.2, "Badajoz 145 MW: 1.1385 pu, 0.2 s (p. 135)")),
    ("Seville550", _ov("OVD1", 1.09, 1.0, "Seville: 1.09 pu, 1 s (p. 141)")),
    ("Segovia94", _ov("OVD1", 1.10, 0.5, "Segovia 94 MW: 1.10 pu, 0.5 s (p. 142)")),
    ("Caceres38", _ov("OVD1", 1.10, 1.5, "Caceres 37.5 MW: 1.10 pu, 1.5 s (p. 143)")),
    ("Malaga154", _ov("OVD1", 1.0989, 1.0, "Malaga 154 MW: 1.0989 pu, 1 s (p. 149)")),
    ("Cuenca530", _ov("OVD2", 1.11, 1.2, "Cuenca 530 MW: 1.11 pu, 1.2 s (p. 151)")),
)


def sequence_actions(*, res_reduction_mw: float = 500.0, res_duration_s: float = 48.0,
                     net_load_mw: float = 525.0, net_load_duration_s: float = 57.0,
                     area_makeup_mw: float = 0.0) -> tuple[E.PreCascadeAction, ...]:
    acts = []
    if area_makeup_mw:
        acts.append(E.PreCascadeAction(
            "MAKEUP", "interchange_reduction", 0.0, float(res_duration_s), -float(area_makeup_mw), "MW",
            "Area governor set points up and the rest's down by the same MW (export held).", "diagnostic"))
    if res_reduction_mw > 0:
        acts.append(E.PreCascadeAction(
            "RES500", "fixed_pf_res_reduction", 0.0, float(res_duration_s), float(res_reduction_mw), "MW",
            "Output of the protected plants reduced pro rata at fixed power factor (P and Q together).",
            "Final report p. 13: large RES at fixed PF -500 MW (12:32:00-12:32:48) and distributed RES -208 MW "
            "(12:32:00-12:32:57)."))
    if net_load_mw > 0:
        acts.append(E.PreCascadeAction(
            "NETLOAD", "area_net_load_ramp", 0.0, float(net_load_duration_s), float(net_load_mw), "MW",
            "Area load up pro rata, rest-of-interconnection load down by the same MW (the area's export falls).",
            "Factual report p. 108: net load +317 MW and distributed RES -208 MW between 12:32:00 and 12:32:57."))
    return tuple(acts)


def sequence_config(
    *,
    tie_scale: float = 10.0,
    voltage_control: str = "avr",
    q_control_tc_s: float = 10.0,
    tap_margin_pu: float = 0.04,
    granada_instantaneous: bool = True,
    documented_settings: bool = False,
    ehv_surplus_mvar: float = 0.0,
    uel_tc_s: float = 0.0,
    actions: dict[str, Any] | None = None,
    tf_s: float = 100.0,
    quiet: bool = True,
    **overrides: Any,
) -> tuple[E.ReplicaConfig, dict[str, Any]]:
    if voltage_control not in {"avr", "q_control", "constant_field"}:
        raise ValueError(f"unknown voltage_control {voltage_control!r}")
    cfg, rec = iberian_state_config(tie_scale=tie_scale, initiating_block=None, quiet=quiet,
                                    tap_margin_pu=float(tap_margin_pu), initial_mesh_closed=2, tf_s=float(tf_s),
                                    area_ehv_shunt_mvar=float(ehv_surplus_mvar), area_uel_tc_s=float(uel_tc_s),
                                    variant=overrides.pop("variant", f"iberian_sequence:k{tie_scale:g}"), **overrides)
    acts = sequence_actions(**(actions or {}))
    cfg = replace(
        cfg,
        pre_cascade_actions=acts,
        extra_plant_stages=(DOCUMENTED_STAGES if documented_settings else
                            (("Granada355", GRANADA_OVI),) if granada_instantaneous else ()),
        area_q_control_tc_s=float(q_control_tc_s) if voltage_control == "q_control" else 0.0,
    )
    rec.update({"voltage_control": voltage_control, "q_control_tc_s": q_control_tc_s if voltage_control == "q_control" else None,
                "tap_margin_pu": tap_margin_pu, "initial_mesh_closed": 2, "granada_instantaneous": granada_instantaneous,
                "documented_settings": documented_settings,
                "actions": [a.code for a in acts], "tf_s": tf_s, "ehv_surplus_mvar": ehv_surplus_mvar,
                "uel_tc_s": uel_tc_s})
    return cfg, rec


