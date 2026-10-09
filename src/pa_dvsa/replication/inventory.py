"""Component inventory of the 2000-bus system, counted from the RAW and DYR files and from the ANDES model.

    python -m pa_dvsa.replication.inventory
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from .activsg2000_engine import (
    EXCITER_MODELS,
    GOVERNOR_MODELS,
    PSS_MODELS,
    ReplicaConfig,
    ReplicaSystem,
    _genrou_links,
    _jsonable,
    build_replica_system,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = REPO_ROOT / "data" / "activsg2000_stable"

DYR_SUBSTITUTIONS = {
    "GENROU": ("GENROU", "native"),
    "GENSAL": ("GENROU", "salient-pole machine mapped to round-rotor GENROU with xq'=xd', xq''=xd'', Tq0'=Td0'"),
    "ESDC1A": ("ESDC1A", "native (limiter references repaired by pa_dvsa.andes_adapter)"),
    "IEEET1": ("IEEET1", "native"),
    "ESAC6A": ("SEXS", "not implemented in ANDES; mapped to SEXS with ALL parameters at ANDES defaults (TA/TB=0.4, TB=5 s, K=20, TE=1 s, EMIN/EMAX=-99/99)"),
    "EXPIC1": ("SEXS", "not implemented in ANDES; mapped to SEXS with default parameters (K=20, TE=1 s, EMIN/EMAX=-99/99)"),
    "SCRX": ("SEXS", "not implemented in ANDES; mapped to SEXS with default parameters"),
    "GGOV1": ("TGOV1", "not implemented in ANDES; mapped to TGOV1 keeping only R (VMAX=1.2, VMIN=0, T1=0.1 s, T2=0.2 s, T3=10 s, Dt=0 defaults)"),
    "IEEEST": ("IEEEST", "native; F2 regularized where improper"),
}


def _raw_sections(path: Path) -> dict[str, int]:
    lines = path.read_text(errors="ignore").splitlines()
    counts: dict[str, int] = {}
    current = "BUS"
    n = 0
    for line in lines[3:]:
        stripped = line.strip()
        if stripped.startswith("0 /") or stripped.startswith("0/") or re.match(r"^0\s*/", stripped):
            counts[current] = n
            m = re.search(r"BEGIN (.+?) DATA", stripped.upper())
            current = m.group(1).strip() if m else "END"
            n = 0
            continue
        if stripped.startswith("Q"):
            break
        n += 1
    counts[current] = n
    if "TRANSFORMER" in counts:
        counts["TRANSFORMER_RECORD_LINES"] = counts["TRANSFORMER"]
        counts["TRANSFORMER"] = counts["TRANSFORMER"] // 4
    return counts


def _dyr_models(path: Path) -> Counter:
    text = path.read_text(errors="ignore")
    return Counter(re.findall(r"^\s*\d+\s+'([A-Z0-9_]+)'", text, flags=re.MULTILINE))


def build_inventory(rs: ReplicaSystem | None = None, *, cfg: ReplicaConfig | None = None) -> dict[str, Any]:
    cfg = cfg or ReplicaConfig()
    if rs is None:
        rs = build_replica_system(cfg, armed=False, quiet=True)
    ss = rs.system
    base = float(ss.config.mva)
    nphys = rs.physical_bus_count
    raw = _raw_sections(DATA_DIR / "ACTIVSg2000.RAW")
    dyr = _dyr_models(DATA_DIR / "ACTIVSg2000_dynamics.dyr")
    vn = np.asarray(ss.Bus.Vn.v, dtype=float)[:nphys]
    area = rs.area.buses

    line = ss.Line
    added_lines = {b.gsu_id for b in rs.blocks} | set(rs.mesh_line_ids) | set(rs.candidate_line_ids)
    orig = [i for i in range(line.n) if str(line.idx.v[i]) not in added_lines]
    trans = [i for i in orig if float(line.trans.v[i]) > 0.5]
    lines_ = [i for i in orig if float(line.trans.v[i]) <= 0.5]

    g = ss.GENROU
    links = _genrou_links(ss)
    online = [i for i in range(g.n) if float(g.u.v[i]) > 0.5]
    with_exc = [i for i in online if links[str(g.idx.v[i])]["exciter"]]
    with_gov = [i for i in online if links[str(g.idx.v[i])]["governor"]]
    with_pss = [i for i in online if links[str(g.idx.v[i])]["pss"]]
    in_area = [i for i in online if int(g.bus.v[i]) in area]

    exc_counts = {m: int(getattr(ss, m).n) for m in EXCITER_MODELS if getattr(getattr(ss, m, None), "n", 0)}
    gov_counts = {m: int(getattr(ss, m).n) for m in GOVERNOR_MODELS if getattr(getattr(ss, m, None), "n", 0)}
    pss_counts = {m: int(getattr(ss, m).n) for m in PSS_MODELS if getattr(getattr(ss, m, None), "n", 0)}

    pq = ss.PQ
    load_p = float(np.sum(np.asarray(pq.p0.v, dtype=float))) * base
    load_q = float(np.sum(np.asarray(pq.q0.v, dtype=float))) * base
    area_load = [i for i in range(pq.n) if int(pq.bus.v[i]) in area]
    cfg_pq = ss.PQ.config.as_dict()

    sh = ss.Shunt
    added_sh = set(rs.candidate_cap_ids) | {b.preserve_shunt_id for b in rs.blocks if b.preserve_shunt_id}
    sh_orig = [i for i in range(sh.n) if str(sh.idx.v[i]) not in added_sh]
    b_arr = np.asarray(sh.b.v, dtype=float)
    caps = [i for i in sh_orig if b_arr[i] > 0]
    reac = [i for i in sh_orig if b_arr[i] < 0]

    pv = ss.PV
    genrou_gen = {str(x) for x in g.gen.v}
    block_pv = {b.pv_id for b in rs.blocks}
    static = [i for i in range(pv.n) if str(pv.idx.v[i]) not in genrou_gen and str(pv.idx.v[i]) not in block_pv]
    static_in_area = [i for i in static if int(pv.bus.v[i]) in area]
    static_outside = [i for i in static if int(pv.bus.v[i]) not in area]

    inv = {
        "source_files": {"raw": str((DATA_DIR / "ACTIVSg2000.RAW").relative_to(REPO_ROOT)), "dyr": str((DATA_DIR / "ACTIVSg2000_dynamics.dyr").relative_to(REPO_ROOT))},
        "raw_record_counts": raw,
        "dyr_model_counts": dict(dyr),
        "dyr_to_andes": {k: {"count": dyr.get(k, 0), "andes_model": v[0], "note": v[1]} for k, v in DYR_SUBSTITUTIONS.items() if dyr.get(k, 0)},
        "buses": {"total": nphys, "by_kV": {f"{k:g}": int(v) for k, v in sorted(Counter(vn.tolist()).items())},
                  "areas": int(len(set(int(a) for a in ss.Bus.area.v[:nphys])))},
        "branches": {"lines": len(lines_), "transformers": len(trans), "total": len(orig),
                     "in_service": int(sum(1 for i in orig if float(line.u.v[i]) > 0.5))},
        "synchronous_machines": {
            "GENROU_records_in_andes": int(g.n), "online": len(online), "offline_in_data": int(g.n - len(online)),
            "online_with_exciter": len(with_exc), "online_without_exciter_constant_field": len(online) - len(with_exc),
            "online_with_governor": len(with_gov), "online_without_governor": len(online) - len(with_gov),
            "online_with_pss": len(with_pss),
            "online_capacity_mva": float(sum(float(g.Sn.v[i]) for i in online)),
            "online_in_affected_area": len(in_area),
            "online_in_affected_area_with_exciter": len([i for i in in_area if links[str(g.idx.v[i])]["exciter"]]),
        },
        "exciters_andes": exc_counts,
        "exciters_disabled_nonequilibrium": list(rs.disabled_exciters),
        "governors_andes": gov_counts,
        "pss_andes": pss_counts,
        "pss_improper_F2_regularized_replica_local": int(sum(
            1 for r in rs.regularization if r.get("model") == "IEEEST" and r.get("source") != "pa_dvsa.andes_regularization")),
        "shared_regularization_records": [
            {k: r.get(k) for k in ("model", "block", "action", "reason")}
            for r in rs.regularization if r.get("source") == "pa_dvsa.andes_regularization"
        ][:3],
        "shared_regularization_record_count": int(sum(1 for r in rs.regularization if r.get("source") == "pa_dvsa.andes_regularization")),
        "loads": {"count": int(pq.n),
                  "p_mw_in_data": load_p + float(rs.base_notes.get("area_load_reduction_mw", 0.0)),
                  "p_mw_after_affected_area_light_load_scaling": load_p, "q_mvar_after_scaling": load_q,
                  "tds_model": "constant impedance (ANDES PQ pq2z=1: p2z=%g, q2z=%g)" % (cfg_pq.get("p2z", 1), cfg_pq.get("q2z", 1)),
                  "affected_area_count": len(area_load),
                  "affected_area_p_mw_after_light_load_scaling": float(sum(float(pq.p0.v[i]) for i in area_load)) * base},
        "shunts": {"count": len(sh_orig), "capacitors": len(caps), "reactors": len(reac),
                   "capacitor_mvar_at_1pu": float(b_arr[caps].sum()) * base if caps else 0.0,
                   "reactor_mvar_at_1pu": float(-b_arr[reac].sum()) * base if reac else 0.0,
                   "note": "RAW fixed shunts (%d) + switched shunts (%d) are both fixed Shunt devices in ANDES (no switched-shunt control)"
                   % (raw.get("FIXED SHUNT", 0), raw.get("SWITCHED SHUNT", 0))},
        "static_generators_without_dynamic_model": {
            "count": len(static), "p_mw": float(sum(float(pv.p0.v[i]) for i in static)) * base,
            "in_affected_area": len(static_in_area), "converted_to_PVD1_in_affected_area": len(rs.gnet_pvd1_ids),
            "outside_affected_area": len(static_outside),
            "outside_tds_behaviour": "ideal voltage source (PV equations stay active in TDS; PV/PQ status frozen at the power-flow result)",
        },
        "hvdc": {"two_terminal": raw.get("TWO-TERMINAL DC", 0), "vsc": raw.get("VOLTAGE SOURCE CONVERTER", 0),
                 "multi_terminal": raw.get("MULTI-TERMINAL DC", 0), "facts": raw.get("FACTS CONTROL DEVICE", 0),
                 "note": "ACTIVSg2000 contains no HVDC; the replica has no HVDC model or HVDC-blocking stage"},
        "affected_area": {
            "acitvsg_area": rs.area.area_id, "name": str(ss.Area.name.v[list(ss.Area.idx.v).index(rs.area.area_id)]) if ss.Area.n else "",
            "buses": len(rs.area.buses), "fragments_left_outside": rs.area.fragments_moved_out,
            "cut_set_branches": len(rs.area.cut_branch_ids),
        },
        "renewable_blocks": [
            {"name": b.spec.name, "hv_bus": b.hv_bus, "hv_kV": float(ss.Bus.Vn.v[list(ss.Bus.idx.v).index(b.hv_bus)]),
             "p_mw": b.p_sched_mw, "q_mvar": b.q_sched_mvar, "q_source": b.q_source, "sn_mva": b.sn_mva,
             "gsu_tap": b.tap, "gsu_x_pu_sys": b.x_sys, "z0_pu": b.z0_pu}
            for b in rs.blocks
        ],
        "added_devices": {
            "collector_buses": len(rs.blocks), "gsu_transformers": len(rs.blocks), "pv_placeholders": len(rs.blocks),
            "pvd1_blocks": len(rs.blocks), "pvd1_gnet": len(rs.gnet_pvd1_ids),
            "busfreq_auto": int(getattr(ss.BusFreq, "n", 0)),
            "meshing_circuits_u0_in_base": len(rs.mesh_line_ids), "candidate_circuits_u0": len(rs.candidate_line_ids),
            "candidate_capacitors_u0": len(rs.candidate_cap_ids),
        },
        "tds_frequency_base_hz": float(ss.config.freq),
    }
    return _jsonable(inv)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=REPO_ROOT / "results" / "revised" / "activsg2000" / "inventory.json")
    args = p.parse_args(argv)
    inv = build_inventory()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(inv, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(inv, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
