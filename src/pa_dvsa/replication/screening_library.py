"""Protected plants and event library of the 2000-bus system for the screening."""

from __future__ import annotations

from dataclasses import replace
import numpy as np

from pa_dvsa.data_model import ProfileKind, TimeProfile
from pa_dvsa.events import EventSpec, Mutation, ProtectedBlock, generator_trip_mutations

from .activsg2000_engine import ReplicaSystem, _genrou_links, _pos, line_series_admittance


def protected_blocks_from_replica(rs: ReplicaSystem) -> list[ProtectedBlock]:
    ss = rs.system
    base = float(ss.config.mva)
    relay_cfg = replace(rs.cfg, plant_ov_enabled=True)
    stages = {s.name: s for s in relay_cfg.plant_stages()}
    ov1 = stages.get("OV1")
    out: list[ProtectedBlock] = []
    v = np.asarray(ss.Bus.v.v, dtype=float)
    for b in rs.blocks:
        k = _pos(ss.PVD1, b.pvd1_id)
        vb = v[_pos(ss.Bus, b.collector_bus)]
        p_mw = float(ss.PVD1.Ipout_y.v[k]) * vb * base
        q_mvar = float(ss.PVD1.Iqout_y.v[k]) * vb * base
        out.append(
            ProtectedBlock(
                asset_id=b.spec.name,
                collector_bus=b.collector_bus,
                hv_bus=b.hv_bus,
                pvd1_idx=b.pvd1_id,
                gsu_line_idx=b.gsu_id,
                tap=float(b.tap),
                gsu_x_pu_sysbase=float(b.x_sys),
                p_mw=p_mw,
                q_mvar=q_mvar,
                sn_mva=float(b.sn_mva),
                pickup_pu=float(ov1.pickup) if ov1 else float("inf"),
                dwell_s=float(ov1.delay_s) if ov1 else float("inf"),
                z0_pu=float(b.z0_pu),
                trip_event_id=f"plant_trip:{b.spec.name}",
                metadata={
                    "report_cluster": b.spec.report_cluster,
                    "q_source": b.q_source,
                    "gsu_r_pu_sysbase": float(b.r_sys),
                    "hv_v0_pu": float(b.hv_v0_pu),
                    "zth_pu": float(b.zth_pu),
                    "relay_stages": [
                        {"name": s.name, "direction": s.direction, "quantity": s.quantity, "pickup": s.pickup,
                         "delay_s": s.delay_s, "dropout_ratio": s.dropout_ratio}
                        for s in relay_cfg.plant_stages()
                    ],
                },
            )
        )
    return out


def _step() -> TimeProfile:
    return TimeProfile(ProfileKind.STEP)


def _ramp(duration_s: float) -> TimeProfile:
    return TimeProfile(ProfileKind.RAMP, duration_s=float(duration_s))


def replica_event_library(rs: ReplicaSystem, *, n_loads: int = 8, n_generators: int = 6,
                          n_redispatch_pairs: int = 3) -> list[EventSpec]:
    ss = rs.system
    base = float(ss.config.mva)
    area = rs.area.buses
    events: list[EventSpec] = []
    pvd = ss.PVD1

    for b in rs.blocks:
        events.append(EventSpec(
            f"plant_trip:{b.spec.name}", "plant_trip", "switch", (Mutation("PVD1", b.pvd1_id, "u", 0.0),), _step(),
            description=f"Trip of protected plant {b.spec.name} (PVD1.u 1->0)", disconnects_assets=(b.spec.name,),
            metadata={"hv_bus": b.hv_bus},
        ))
    for b in rs.blocks:
        k = _pos(pvd, b.pvd1_id)
        p0, q0 = float(pvd.pref0.v[k]), float(pvd.qref0.v[k])
        events.append(EventSpec(
            f"fixed_pf_ramp:{b.spec.name}", "fixed_pf_ramp", "param",
            (Mutation("PVD1", b.pvd1_id, "pref0", 0.7 * p0), Mutation("PVD1", b.pvd1_id, "qref0", 0.7 * q0)),
            _ramp(0.5), description=f"{b.spec.name} output to 70% over 0.5 s at fixed PF (P and Q together)",
            metadata={"pref0": p0, "qref0": q0},
        ))

    pq = ss.PQ
    loads = [i for i in range(pq.n) if int(pq.bus.v[i]) in area and float(pq.u.v[i]) > 0.5]
    loads.sort(key=lambda i: (-abs(float(pq.Qpf.v[i]) if hasattr(pq, "Qpf") else float(pq.q0.v[i])), str(pq.idx.v[i])))
    for i in loads[:n_loads]:
        idx = pq.idx.v[i]
        events.append(EventSpec(
            f"load_disconnection:{idx}", "load_disconnection", "switch", (Mutation("PQ", idx, "u", 0.0),), _step(),
            description=f"Disconnection of load {idx} at bus {int(pq.bus.v[i])}",
            metadata={"p_mw": float(pq.Ppf.v[i]) * base, "q_mvar": float(pq.Qpf.v[i]) * base},
        ))

    g = ss.GENROU
    slack_gens = {str(v) for v in ss.Slack.idx.v}
    links = _genrou_links(ss)
    gens = [i for i in range(g.n) if float(g.u.v[i]) > 0.5 and int(g.bus.v[i]) in area and str(g.gen.v[i]) not in slack_gens]
    gens.sort(key=lambda i: (-float(g.Pe.v[i]), str(g.idx.v[i])))
    for i in gens[:n_generators]:
        gid = str(g.idx.v[i])
        muts = list(generator_trip_mutations(ss, gid, include_effective_status=True))
        expected = {("GENROU", gid)} | {tuple(ref.split(":", 1)) for ref in
                                         links[gid]["exciter"] + links[gid]["governor"] + links[gid]["pss"]}
        if {(m.model, str(m.device_idx)) for m in muts} != expected:
            raise RuntimeError(f"generator_trip:{gid}: linked devices differ from the replica links")
        events.append(EventSpec(
            f"generator_trip:{gid}", "generator_trip", "switch", tuple(muts), _step(),
            description=f"Trip of {gid} (+ exciter/governor/PSS) at bus {int(g.bus.v[i])}",
            metadata={"p_mw": float(g.Pe.v[i]) * base, "q_mvar": float(g.Qe.v[i]) * base},
        ))

    tg = ss.TGOV1
    gpos = {str(v): i for i, v in enumerate(g.idx.v)}

    def gov_rows(inside: bool) -> list[int]:
        rows = []
        for k in range(tg.n):
            gi = gpos.get(str(tg.syn.v[k]))
            if gi is None or float(g.u.v[gi]) <= 0.5 or float(tg.u.v[k]) <= 0.5:
                continue
            if (int(g.bus.v[gi]) in area) == inside and float(tg.pref0.v[k]) > 0:
                rows.append(k)
        rows.sort(key=lambda k: (-float(tg.pref0.v[k]), str(tg.idx.v[k])))
        return rows

    inside, outside = gov_rows(True), gov_rows(False)
    for a, r in list(zip(inside, outside))[:n_redispatch_pairs]:
        dp = 0.2 * float(tg.pref0.v[a])
        events.append(EventSpec(
            f"generation_redispatch:{tg.idx.v[a]}->{tg.idx.v[r]}", "generation_redispatch", "param",
            (Mutation("TGOV1", tg.idx.v[a], "pref0", float(tg.pref0.v[a]) - dp),
             Mutation("TGOV1", tg.idx.v[r], "pref0", float(tg.pref0.v[r]) + dp)),
            _ramp(1.0), description=f"Exchange reduction {dp * base:.0f} MW: {tg.idx.v[a]} (area) down, {tg.idx.v[r]} (rest) up",
            metadata={"delta_mw": dp * base},
        ))

    for lid in rs.candidate_line_ids:
        y = line_series_admittance(ss, lid)
        events.append(EventSpec(
            f"line_energization:{lid}", "line_energization", "switch",
            (Mutation("Line", lid, "u", 1.0), Mutation("Line", lid, "ghk", y.real), Mutation("Line", lid, "bhk", y.imag)),
            _step(),
            description=f"Energization of parallel circuit {lid}",
            metadata={"charging_mvar_1pu": float(ss.Line.b.v[_pos(ss.Line, lid)]) * base, "ghk_pu": y.real, "bhk_pu": y.imag},
        ))
    for sid in rs.candidate_cap_ids:
        k = _pos(ss.Shunt, sid)
        events.append(EventSpec(
            f"capacitor_switching:{sid}", "capacitor_switching", "switch", (Mutation("Shunt", sid, "u", 1.0),), _step(),
            description=f"Energization of capacitor {sid} ({float(ss.Shunt.b.v[k]) * base:.0f} MVAr at 1 pu)",
            metadata={"mvar_1pu": float(ss.Shunt.b.v[k]) * base, "bus": int(ss.Shunt.bus.v[k])},
        ))
    return events


__all__ = ["protected_blocks_from_replica", "replica_event_library"]
