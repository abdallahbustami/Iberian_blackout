"""Tables of the stressed-state runs: final_table.{json,md} (Table 3), final_main/timeline_facts.json and
tables.md in results/revised/iberian_sequence/.

    python scripts/report_iberian_sequence.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "revised" / "iberian_sequence"


def drivers(acts: dict) -> str:
    res = acts.get("res_reduction_mw", 500.0)
    load = acts.get("net_load_mw", 525.0)
    out = [f"RES -{res:g} MW" if res else "", f"load +{load:g} MW" if load else ""]
    if acts.get("area_makeup_mw"):
        out.append(f"area makeup {acts['area_makeup_mw']:g} MW")
    return ", ".join(x for x in out if x) or "none"


def row(d: Path) -> dict | None:
    st = json.loads((d / "study.json").read_text())
    if "error" in st or not (d / "system_traces.csv").exists():
        return None
    kw, rec = st["kwargs"], st.get("record", {})
    s = pd.read_csv(d / "system_traces.csv")
    c = pd.read_csv(d / "collector_traces.csv")
    hv = [x for x in c.columns if x.endswith("_hv_v_pu")]
    c["hv_med"] = c[hv].median(axis=1)
    ev = json.loads((d / "events.json").read_text())
    trips = [e for e in ev if e["category"] == "plant_trip"]
    sep = [e["time_s"] for e in ev if e["category"] == "out_of_step_separation"]
    col = [e["time_s"] for e in ev if e["category"] == "island_collapse"]
    t1 = trips[0]["time_s"] if trips else None
    t_end_rise = min(57.0, t1) if t1 is not None else 57.0
    v0 = float(c.hv_med.iloc[0])
    v1 = float(c.hv_med.iloc[(c.time_s - t_end_rise).abs().argmin()])
    lost4 = None
    if t1 is not None:
        lost4 = float(s.cum_lost_p_mw.iloc[(s.time_s - (t1 + 4.0)).abs().argmin()])
    return {
        "study": d.name, "k": kw.get("tie_scale"), "voltage control": kw.get("voltage_control"),
        "surplus (MVAr)": kw.get("ehv_surplus_mvar", 0.0), "UEL": "yes" if kw.get("uel_tc_s") else "no",
        "margin (pu)": rec.get("tap_margin_pu"), "relays": kw.get("plant_ov_mode", "standard") +
        (" + documented" if kw.get("documented_settings") else ""),
        "RES share": kw.get("res_share", 0.45), "PF": kw.get("fixed_pf", 0.95), "load": kw.get("load_model", "z"),
        "embedded PV": kw.get("embedded_pv_share", 0.0), "Granada tap": "yes" if kw.get("block_tap_factor") else "no",
        "drivers": drivers(kw.get("actions") or {}),
        "V plants t=0 (pu)": round(v0, 3), "rise to first trip / 57 s (pu)": round(v1 - v0, 4),
        "first trip (s)": None if t1 is None else round(t1, 2),
        "first plant": trips[0]["code"].replace("TRIP_", "") if trips else "-",
        "observe only": "yes" if kw.get("plant_ov_enabled") is False else "no",
        "lost in 4 s (MW)": None if lost4 is None else round(lost4),
        "separation after first trip (s)": round(sep[0] - t1, 2) if sep and t1 is not None else "-",
        "collapse after first trip (s)": round(col[0] - t1, 2) if col and t1 is not None else "-",
        "end": ("collapse" if col else ("TDS stops in collapse (all plants and >= 9 units tripped)" if (not st.get("tds_ok") and sep)
                else ("survives" if st.get("tds_ok") else "TDS stops"))),
    }


def final_row(d: Path) -> dict:
    st = json.loads((d / "study.json").read_text())
    s = pd.read_csv(d / "system_traces.csv")
    ev = json.loads((d / "events.json").read_text())
    trips = [e for e in ev if e["category"] == "plant_trip"]
    sep = next((e["time_s"] for e in ev if e["category"] == "out_of_step_separation"), None)
    col = next((e["time_s"] for e in ev if e["category"] == "island_collapse"), None)
    t_cut = sep if sep is not None else float(s.time_s.max())
    before = [e for e in trips if e["time_s"] <= t_cut + 1e-9]
    pre = s[s.time_s <= t_cut + 1e-9]
    dg = float(pre.cum_embedded_pv_lost_mw.iloc[-1]) if "cum_embedded_pv_lost_mw" in pre else 0.0
    gens = [e for e in ev if e["category"] == "generator_trip"]
    ufls = [e for e in ev if e["category"] == "ufls"]
    t1 = trips[0]["time_s"] if trips else None
    lost25 = (float(s.cum_lost_p_mw.iloc[(s.time_s - (t1 + 2.5)).abs().argmin()]) if t1 is not None else None)
    c = pd.read_csv(d / "collector_traces.csv")
    names = [x[:-len("_hv_v_pu")] for x in c.columns if x.endswith("_hv_v_pu")]
    rv = c[[f"{n}_v_pu" for n in names]].where(c[[f"{n}_online" for n in names]].to_numpy() > 0.5)
    relay_vmax = float(rv[c.time_s <= (t1 if t1 is not None else c.time_s.max()) + 1e-9].max().max())
    hv_med = c[[f"{n}_hv_v_pu" for n in names]].median(axis=1)
    t_r = t1 if t1 is not None else float(c.time_s.max())
    hv_rise = float(hv_med[c.time_s < t_r - 1e-6].iloc[-1] - hv_med.iloc[0])
    if col is not None:
        end = "collapse"
    elif not st.get("tds_ok") and sep is not None:
        end = "TDS stops after separation"
    elif sep is not None:
        end = "island survives"
    else:
        end = "no separation"
    return {
        "study": d.name,
        "first_trip_s": None if t1 is None else round(t1, 3),
        "first_plant": trips[0]["code"].replace("TRIP_", "") if trips else None,
        "plants_before_separation": len(before),
        "plant_mw_before_separation": round(sum(float(e["lost_p_mw"]) for e in before)),
        "embedded_pv_lost_before_separation_mw": round(dg),
        "lost_2p5s_after_first_trip_mw": None if lost25 is None else round(lost25),
        "separation_s": None if sep is None else round(sep, 3),
        "separation_after_first_trip_s": None if (sep is None or t1 is None) else round(sep - t1, 3),
        "relay_v_max_before_first_trip_pu": round(relay_vmax, 4),
        "hv_median_rise_before_first_trip_pu": round(hv_rise, 4),
        "hv_median_rise_at_48s_pu": None if (t1 is not None and t1 < 48.0) else
        round(float(hv_med.iloc[(c.time_s - 48.0).abs().argmin()] - hv_med.iloc[0]), 4),
        "plants_total": len(trips), "plant_mw_total": round(float(st.get("lost_block_p_mw") or 0.0)),
        "generator_trips": len(gens), "ufls_stages": len(ufls), "ufls_shed_mw": round(float(st.get("ufls_shed_mw") or 0.0)),
        "final_time_s": round(float(s.time_s.max()), 3), "tds_ok": bool(st.get("tds_ok")), "end": end,
        "f_min_hz": round(float(s.affected_island_f_eq_hz.min()), 2),
    }


def final_table() -> None:
    rows = [final_row(d) for d in sorted(OUT.glob("final_*")) if (d / "study.json").exists()
            and (d / "system_traces.csv").exists()]
    (OUT / "final_table.json").write_text(json.dumps(rows, indent=2))
    df = pd.DataFrame(rows)
    lines = ["# Final sequence-replica set (time 0 = 12:32:00); plants and embedded PV counted up to separation", "",
             "| " + " | ".join(df.columns) + " |", "|" + "---|" * len(df.columns)]
    lines += ["| " + " | ".join("" if v is None or (isinstance(v, float) and np.isnan(v)) else str(v) for v in r) + " |"
              for r in df.itertuples(index=False)]
    (OUT / "final_table.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main_facts(label: str = "final_main") -> dict:
    d = OUT / label
    s = pd.read_csv(d / "system_traces.csv")
    c = pd.read_csv(d / "collector_traces.csv")
    ev = json.loads((d / "events.json").read_text())
    names = [x[:-len("_hv_v_pu")] for x in c.columns if x.endswith("_hv_v_pu")]
    hv = c[[f"{n}_hv_v_pu" for n in names]]
    on = c[[f"{n}_online" for n in names]].to_numpy() > 0.5
    hv_on = hv.where(on)
    c["hv_med"] = hv.median(axis=1)
    trips = [e for e in ev if e["category"] == "plant_trip"]
    t1 = trips[0]["time_s"]
    t_sep = next(e["time_s"] for e in ev if e["category"] == "out_of_step_separation")
    burst = [e for e in trips if e["time_s"] < t1 + 2.0]
    t_burst = burst[-1]["time_s"]

    def at(df, col, t):
        return float(df[col].iloc[(df.time_s - t).abs().argmin()])

    pre = s[s.time_s <= t_sep + 1e-9]
    swing = pre[(pre.time_s > t_burst) & (pre.time_s < t_sep - 0.5)]
    i_pk = swing.coi_angle_change_deg.idxmax()
    cpre = c[(c.time_s > t_burst) & (c.time_s < t_sep)]
    post = [e for e in trips if e["time_s"] > t_sep]
    gens = [e for e in ev if e["category"] == "generator_trip"]
    ufls = [e for e in ev if e["category"] == "ufls"]
    hv_after_on = hv_on[c.time_s > t_sep]
    col_v = c[[f"{n}_v_pu" for n in names]].where(on)
    facts = {
        "hv_median_t0_pu": float(c.hv_med.iloc[0]),
        "first_trip": {"plant": trips[0]["code"].replace("TRIP_", ""), "t_s": t1,
                       "pickup_start_s": trips[0]["measured"]["pickup_start_s"],
                       "lost_q_absorption_mvar": trips[0]["lost_q_absorption_mvar"]},
        "hv_median_at_pickup_start_pu": at(c, "hv_med", trips[0]["measured"]["pickup_start_s"]),
        "hv_median_at_first_trip_pu": at(c, "hv_med", t1),
        "burst": {"n_plants": len(burst), "mw": sum(e["lost_p_mw"] for e in burst), "last_s": t_burst,
                  "plants": [e["code"].replace("TRIP_", "") for e in burst]},
        "hv_online_range_after_burst_pu": [float(hv_on.iloc[(c.time_s - (t_burst + 0.1)).abs().argmin()].min()),
                                           float(hv_on.iloc[(c.time_s - (t_burst + 0.1)).abs().argmin()].max())],
        "embedded_pv_lost_2s_after_first_trip_mw": at(s, "cum_embedded_pv_lost_mw", t1 + 2.0),
        "first_swing_peak": {"t_s": float(swing.time_s.loc[i_pk]), "angle_deg": float(swing.coi_angle_change_deg.loc[i_pk])},
        "hv_median_min_before_separation_pu": float(cpre.hv_med.min()),
        "hv_median_min_first_swing_pu": float(c.hv_med[(c.time_s > float(swing.time_s.loc[i_pk]) - 0.5)
                                                      & (c.time_s < float(swing.time_s.loc[i_pk]) + 0.5)].min()),
        "frequency_at_separation_hz": at(s, "affected_island_f_eq_hz", t_sep),
        "trips_before_separation": [{"plant": e["code"].replace("TRIP_", ""), "t_s": e["time_s"], "trigger": e["trigger"],
                                     "mw": e["lost_p_mw"]} for e in trips if e["time_s"] <= t_sep],
        "separation_s": t_sep, "angle_at_separation_deg": at(s, "coi_angle_change_deg", t_sep),
        "ufls": {"n": len(ufls), "first_s": ufls[0]["time_s"] if ufls else None, "last_s": ufls[-1]["time_s"] if ufls else None,
                 "shed_mw": float(s.cum_ufls_shed_mw.max())},
        "plants_after_separation": {"n": len(post), "triggers": sorted({e["trigger"] for e in post})},
        "relay_side_v_max_after_separation_pu": float(col_v[c.time_s > t_sep].max().max()),
        "hv_max_online_after_separation_pu": float(hv_after_on.max().max()),
        "generator_trips": {"n": len(gens), "by_trigger": {k: sum(e["trigger"] == k for e in gens)
                                                            for k in sorted({e["trigger"] for e in gens})},
                            "first_s": gens[0]["time_s"] if gens else None,
                            "terminal_v_pu": [e["measured"].get("value") for e in gens if e["trigger"].endswith("OV2")]},
        "units_online_at_end": int(s.gens_online.iloc[-1]),
        "f_min_hz": float(s.affected_island_f_eq_hz.min()),
        "final_time_s": float(s.time_s.max()),
    }
    (d / "timeline_facts.json").write_text(json.dumps(facts, indent=2))
    return facts


def main(argv: list[str] | None = None) -> int:
    argparse.ArgumentParser(description=__doc__).parse_args(argv)
    runs = [d for d in sorted(OUT.glob("final_*")) if (d / "study.json").exists()]
    missing = [d.name for d in runs if not ((d / "system_traces.csv").exists() and (d / "collector_traces.csv").exists())]
    if not runs or missing:
        print("The traces of these runs are missing (the stored results keep only those of final_main):",
              ", ".join(missing) or "all", "\nRun scripts/run_iberian_sequence.py first. Nothing was written.")
        return 1
    final_table()
    main_facts()
    rows = [r for d in sorted(OUT.iterdir()) if d.is_dir() and (d / "study.json").exists() for r in [row(d)] if r]
    df = pd.DataFrame(rows)
    lines = ["# Sequence-replica studies (time 0 = 12:32:00; 28 April: Granada 57.2 s, Badajoz 582 MW 76.46 s, "
             "loss of synchronism 79.62 s, AC separation 81.54 s, i.e. +5.1 s after Badajoz)", ""]
    cols = list(df.columns)
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("|" + "---|" * len(cols))
    for _, r in df.iterrows():
        lines.append("| " + " | ".join("" if (isinstance(v, float) and np.isnan(v)) else str(v) for v in r) + " |")
    (OUT / "tables.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
