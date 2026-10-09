"""Screening responses against nonlinear time-domain simulation for every event type (helper and script).

    python tests/_core_validation.py --json out.json --markdown out.md
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tests import _core_system as CS
from pa_dvsa.event_response import EventResponseEngine, OutputSelector
from pa_dvsa.tds_runner import run_protection_tds

T_SEED = 0.5
TDS_TSTEP = 0.002
TDS_TOL = 1e-8
FAMILIES = (
    "plant_trip",
    "generator_trip",
    "load_disconnection",
    "shunt_reactor_disconnection",
    "capacitor_switching",
    "line_energization",
    "fixed_pf_ramp",
    "governor_ramp",
    "exciter_step",
    "statcom",
)


def _tds_response(res: Any, times: np.ndarray) -> np.ndarray:
    t = res.traces["t"]
    x = res.traces["extra"]
    k0 = int(np.searchsorted(t, T_SEED - 1e-9))
    base = x[k0]
    post = t > T_SEED + 1e-9
    tp = np.concatenate(([0.0], t[post] - T_SEED))
    xp = np.vstack((x[k0 + 1], x[post])) - base
    return np.stack([np.interp(times, tp, xp[:, j]) for j in range(x.shape[1])], axis=1)


def _extreme(values: np.ndarray) -> float:
    k = int(np.argmax(np.abs(values)))
    return float(values[k])


def run_validation(families: tuple[str, ...] = FAMILIES) -> dict[str, Any]:
    ss = CS.fresh_system(tstep=TDS_TSTEP)
    outs = CS.output_buses()
    ib = list(ss.Bus.idx.v)
    addr = {name: int(ss.Bus.v.a[ib.index(bus)]) for name, bus in outs}
    engine = EventResponseEngine(ss, [OutputSelector.bus_voltage(ss, b, n) for n, b in outs])
    t0 = time.perf_counter()
    pre = engine.pre_mode
    pre_time = time.perf_counter() - t0
    events = CS.event_library(ss)
    blocks = CS.protected_blocks(ss)
    names = [n for n, _ in outs]
    rows: list[dict[str, Any]] = []
    noevent = run_protection_tds(
        lambda: CS.fresh_system(tstep=TDS_TSTEP), None, blocks, relays_enabled=False,
        t_seed=T_SEED, t_end=T_SEED + 1.0, tstep=TDS_TSTEP, tol=TDS_TOL,
        record_traces=True, trace_addresses=addr,
    )
    no_event_dz = _tds_response(noevent, engine.times_s)
    for fam in families:
        is_control = fam == "statcom"
        spec = CS.statcom_control() if is_control else events[fam]
        t1 = time.perf_counter()
        resp = engine.control_response(spec) if is_control else engine.event_response(spec)
        screen_s = time.perf_counter() - t1
        tds = run_protection_tds(
            lambda: CS.fresh_system(tstep=TDS_TSTEP),
            None if is_control else spec,
            blocks,
            relays_enabled=False,
            controls=[(spec, 1.0)] if is_control else (),
            t_seed=T_SEED,
            t_end=T_SEED + 1.0,
            tstep=TDS_TSTEP,
            tol=TDS_TOL,
            record_traces=True,
            trace_addresses=addr,
        )
        dz_tds = _tds_response(tds, engine.times_s)
        err = resp.values_pu - dz_tds
        m03 = engine.times_s <= 0.3 + 1e-9
        k20 = int(np.searchsorted(engine.times_s, 0.02 - 1e-9))
        per_output = []
        for j, name in enumerate(names):
            ext_s = _extreme(resp.values_pu[:, j])
            ext_t = _extreme(dz_tds[:, j])
            wmax_s = float(np.max(resp.values_pu[:, j]))
            wmax_t = float(np.max(dz_tds[:, j]))
            per_output.append(
                {
                    "output": name,
                    "jump_screen_mpu": 1e3 * float(resp.jump_pu[j]),
                    "dz0_tds_mpu": 1e3 * float(dz_tds[0, j]),
                    "dz20ms_screen_mpu": 1e3 * float(resp.values_pu[k20, j]),
                    "dz20ms_tds_mpu": 1e3 * float(dz_tds[k20, j]),
                    "max_abs_err_0_0p3_mpu": 1e3 * float(np.max(np.abs(err[m03, j]))),
                    "max_abs_err_0_1_mpu": 1e3 * float(np.max(np.abs(err[:, j]))),
                    "extreme_screen_mpu": 1e3 * ext_s,
                    "extreme_tds_mpu": 1e3 * ext_t,
                    "extreme_rel_err": (ext_s - ext_t) / abs(ext_t) if abs(ext_t) > 1e-6 else float("nan"),
                    "window_max_screen_mpu": 1e3 * wmax_s,
                    "window_max_tds_mpu": 1e3 * wmax_t,
                    "window_max_rel_err": (wmax_s - wmax_t) / abs(wmax_t) if abs(wmax_t) > 1e-3 else float("nan"),
                    "rigorous_peak_bound_mpu": 1e3 * float(resp.envelopes[j].peak_upper),
                    "bound_covers_tds_max": bool(resp.envelopes[j].peak_upper >= wmax_t - 1e-12),
                    "tds_minus_noevent_max_abs_mpu": 1e3 * float(np.max(np.abs(dz_tds[:, j] - no_event_dz[:, j]))),
                }
            )
        rows.append(
            {
                "family": fam,
                "event_id": resp.event_id,
                "kind": resp.kind,
                "description": getattr(spec, "description", ""),
                "screen_seconds": screen_s,
                "tds_seconds": tds.runtime_s,
                "tds_converged": tds.converged,
                "tds_err_msg": tds.err_msg,
                "newton_iterations": resp.newton_iterations,
                "diagnostics": resp.diagnostics.to_dict(),
                "outputs": per_output,
            }
        )
    return {
        "system": "IEEE 39 + 3 PVD1 plant blocks (tests/_core_system.py), regularized IEEEST, Toggle disabled",
        "tds": {"tstep_s": TDS_TSTEP, "tol": TDS_TOL, "t_seed_s": T_SEED, "window_s": 1.0},
        "screen_grid_s": 0.005,
        "pre_mode": pre.diagnostics().to_dict(),
        "pre_mode_seconds": pre_time,
        "no_event_max_abs_mpu": 1e3 * float(np.max(np.abs(no_event_dz))),
        "rows": rows,
    }


def _fmt(v: float, nd: int = 2) -> str:
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return "n/a"
    return f"{v:.{nd}f}"


def markdown_table(data: dict[str, Any]) -> str:
    lines = [
        "| event | kind | group | max abs err [0,0.3] s (mpu) | max abs err [0,1] s (mpu) | "
        "largest window extreme screen / TDS (mpu) | rel. err at that output | "
        "max rel. err over outputs (extreme >= 1 mpu) | screen (s) | TDS (s) |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    groups = (
        ("collectors", lambda o: o["output"].startswith("COL")),
        ("HV buses", lambda o: o["output"].startswith("HV")),
    )
    for row in data["rows"]:
        for group, pred in groups:
            outs = [o for o in row["outputs"] if pred(o)]
            e03 = max(o["max_abs_err_0_0p3_mpu"] for o in outs)
            e1 = max(o["max_abs_err_0_1_mpu"] for o in outs)
            worst = max(outs, key=lambda o: abs(o["extreme_tds_mpu"]))
            rel = [o["extreme_rel_err"] for o in outs if abs(o["extreme_tds_mpu"]) >= 1.0 and math.isfinite(o["extreme_rel_err"])]
            rel_w = worst["extreme_rel_err"]
            rel_w_s = _fmt(100 * rel_w, 1) + " %" if math.isfinite(rel_w) and abs(worst["extreme_tds_mpu"]) >= 1.0 else "n/a"
            rel_s = _fmt(100 * max(rel, key=abs), 1) + " %" if rel else "n/a (< 1 mpu)"
            lines.append(
                f"| {row['family']} | {row['kind']} | {group} | {_fmt(e03, 3)} | {_fmt(e1, 3)} | "
                f"{_fmt(worst['extreme_screen_mpu'])} / {_fmt(worst['extreme_tds_mpu'])} ({worst['output']}) | "
                f"{rel_w_s} | {rel_s} | {_fmt(row['screen_seconds'], 3)} | {_fmt(row['tds_seconds'], 1)} |"
            )
    return "\n".join(lines)


def collector_table(data: dict[str, Any]) -> str:
    lines = [
        "| event | output | 0+ jump screen / TDS (mpu) | dz(20 ms) screen / TDS (mpu) | err [0,0.3] s | err [0,1] s | "
        "window max screen / TDS / rigorous bound (mpu) | window-max rel. err |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for row in data["rows"]:
        for o in row["outputs"]:
            if not o["output"].startswith("COL"):
                continue
            rel = o["window_max_rel_err"]
            rel_s = f"{100 * rel:+.1f} %" if math.isfinite(rel) and o["window_max_tds_mpu"] >= 1.0 else "n/a (max < 1 mpu)"
            lines.append(
                f"| {row['family']} | {o['output']} | {o['jump_screen_mpu']:+.3f} / {o['dz0_tds_mpu']:+.3f} | "
                f"{o['dz20ms_screen_mpu']:+.3f} / {o['dz20ms_tds_mpu']:+.3f} | {o['max_abs_err_0_0p3_mpu']:.3f} | "
                f"{o['max_abs_err_0_1_mpu']:.3f} | {o['window_max_screen_mpu']:+.2f} / {o['window_max_tds_mpu']:+.2f} / "
                f"{o['rigorous_peak_bound_mpu']:+.2f} | {rel_s} |"
            )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path, default=None)
    parser.add_argument("--markdown", type=Path, default=None)
    args = parser.parse_args()
    data = run_validation()
    table = markdown_table(data) + "\n\n" + collector_table(data)
    print(table)
    if args.json:
        args.json.write_text(json.dumps(data, indent=2, default=float))
    if args.markdown:
        args.markdown.write_text(table + "\n")


if __name__ == "__main__":
    main()
