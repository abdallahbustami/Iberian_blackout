"""Reactive power balance of the area at the initial operating point of the stressed state, with and without
the added susceptance (results/revised/iberian_sequence/initial_balance.json).

    python scripts/report_iberian_sequence_balance.py
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from pa_dvsa.replication import activsg2000_engine as E
from pa_dvsa.replication.iberian_sequence import sequence_config
from run_iberian_sequence import FINAL_BASE

OUT = ROOT / "results" / "revised" / "iberian_sequence" / "initial_balance.json"


def balance(kw: dict) -> dict:
    cfg, _ = sequence_config(**kw)
    dis = E.detect_nonequilibrium_exciters(cfg)
    tap = E.solve_uniform_tap(cfg, disabled_exciters=dis)
    rs = E.build_replica_system(replace(cfg, tap_override=tap, pre_cascade_enabled=False), quiet=True,
                                disabled_exciters=dis, init_tds=False)
    ss = rs.system
    base = float(ss.config.mva)
    v = np.asarray(ss.Bus.v.v, float)
    pos = {int(b): k for k, b in enumerate(ss.Bus.idx.v)}
    area = rs.area.buses
    plants_q = sum(float(ss.PV.q.v[E._pos(ss.PV, b.pv_id)]) for b in rs.blocks) * base
    plants_p = sum(float(ss.PV.p0.v[E._pos(ss.PV, b.pv_id)]) for b in rs.blocks) * base
    hv = [float(v[pos[int(b.hv_bus)]]) for b in rs.blocks]
    surplus = sum(float(ss.Shunt.b.v[i]) * v[pos[int(ss.Shunt.bus.v[i])]] ** 2 * base
                  for i in range(ss.Shunt.n) if str(ss.Shunt.idx.v[i]).startswith("EHVCHG_"))
    g = ss.GENROU
    pv_ids = list(ss.PV.idx.v)
    units = []
    for i in range(g.n):
        if float(g.u.v[i]) <= 0.5 or int(g.bus.v[i]) not in area:
            continue
        gen = g.gen.v[i]
        m = ss.PV if gen in pv_ids else ss.Slack
        k = E._pos(m, gen)
        q, qmin = float(m.q.v[k]) * base, float(m.qmin.v[k]) * base
        units.append({"genrou": str(g.idx.v[i]), "sn_mva": float(g.Sn.v[i]), "p_mw": float(m.p0.v[k]) * base,
                      "q_mvar": q, "qmin_mvar": qmin, "at_qmin": bool(q <= qmin + 1.0)})
    return {
        "kwargs": kw, "gsu_tap": tap,
        "plants": {"n": len(rs.blocks), "p_mw": plants_p, "q_mvar": plants_q},
        "plant_hv_v_pu": {"median": float(np.median(hv)), "min": float(min(hv)), "max": float(max(hv))},
        "added_surplus_mvar_at_operating_voltage": surplus,
        "sync_units": {"n": len(units), "net_q_mvar": sum(u["q_mvar"] for u in units),
                       "absorbing": sum(u["q_mvar"] < 0 for u in units), "at_qmin": sum(u["at_qmin"] for u in units),
                       "units": units},
        "embedded_pv": {"n_buses": len(rs.embedded_pv), "p_mw": sum(d["p_mw"] for d in rs.embedded_pv)},
        "area_net_export_mw": E.area_net_export_mw(rs),
    }


def main(argv: list[str] | None = None) -> int:
    argparse.ArgumentParser(description=__doc__).parse_args(argv)
    out = {"final_main": balance(dict(FINAL_BASE)),
           "final_no_surplus": balance({**FINAL_BASE, "ehv_surplus_mvar": 0.0})}
    OUT.write_text(json.dumps(out, indent=2, default=float))
    for k, r in out.items():
        print(k, f"plants {r['plants']['q_mvar']:.0f} MVAr, units net {r['sync_units']['net_q_mvar']:.0f} MVAr "
              f"({r['sync_units']['at_qmin']}/{r['sync_units']['n']} at Qmin), surplus "
              f"{r['added_surplus_mvar_at_operating_voltage']:.0f} MVAr, plant HV median {r['plant_hv_v_pu']['median']:.4f}, "
              f"DG {r['embedded_pv']['p_mw']:.0f} MW at {r['embedded_pv']['n_buses']} loads, export {r['area_net_export_mw']:.0f} MW")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
