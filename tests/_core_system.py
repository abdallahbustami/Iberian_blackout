"""IEEE 39 test system with three protected plants and the devices of every event family (helper, not a test)."""

from __future__ import annotations

from functools import lru_cache
import math
from typing import Any

import numpy as np

import andes

from pa_dvsa.andes_regularization import regularize_degenerate_blocks
from pa_dvsa.data_model import ProfileKind, TimeProfile
from pa_dvsa.events import (
    ControlSpec,
    EventSpec,
    Mutation,
    ProtectedBlock,
    generator_trip_mutations,
    lag_envelope,
)

andes.config_logger(stream_level=40)

CASE = "ieee39/ieee39_full.xlsx"
KAPPA = -math.tan(math.acos(0.95))
BLOCKS = (
    ("B1", 16, 2.0, 1.080),
    ("B2", 21, 1.6, 1.070),
    ("B3", 4, 2.4, 1.085),
)
HV_KV = 345.0
COL_KV = 34.5
SHUNT_REACTOR = ("SHR1", 15, -0.8)
CAPACITOR = ("CAP1", 21, 0.6)
STATCOM = ("STC1", 16)
PARALLEL_LINE = ("PAR1", 16, 17, 0.0007, 0.0089, 0.1342)


def collector_bus(k: int) -> int:
    return 1001 + k


def _gsu_x_sysbase(p_pu: float) -> float:
    sn_mva = 1.15 * p_pu * 100.0
    return 0.12 * 100.0 / sn_mva


def _load(taps: tuple[float, ...]) -> Any:
    ss = andes.load(andes.get_case(CASE), setup=False, no_output=True, default_config=True)
    for i in range(ss.Toggle.n):
        ss.Toggle.u.v[i] = 0
    total_block_p = sum(p for _, _, p, _ in BLOCKS)
    pv_total = float(sum(ss.PV.p0.v))
    factor = (pv_total - total_block_p) / pv_total
    for i in range(ss.PV.n):
        ss.PV.p0.v[i] = float(ss.PV.p0.v[i]) * factor
    for k, ((asset, hv, p, z0), tap) in enumerate(zip(BLOCKS, taps)):
        col = collector_bus(k)
        sn = 1.15 * p * 100.0
        x = _gsu_x_sysbase(p)
        ss.add("Bus", dict(idx=col, name=f"COL_{asset}", Vn=COL_KV, v0=z0, a0=0.0))
        ss.add(
            "Line",
            dict(idx=f"GSU_{asset}", bus1=hv, bus2=col, r=x / 40.0, x=x, b=0.0, tap=tap, phi=0.0,
                 trans=1, Vn1=HV_KV, Vn2=COL_KV, Sn=100.0),
        )
        ss.add("PV", dict(idx=f"PVB_{asset}", bus=col, p0=p, v0=z0, qmax=5.0, qmin=-5.0, Sn=sn, Vn=COL_KV))
        ss.add(
            "PVD1",
            dict(idx=f"PD_{asset}", bus=col, gen=f"PVB_{asset}", Sn=sn, fn=60.0, ialim=1.3,
                 vt0=0.0, vt1=0.01, vt2=5.0, vt3=6.0, ft0=0.0, ft1=0.1, ft2=999.0, ft3=1000.0,
                 pqflag=0, qmx=0.6, qmn=-0.6, dqdv=0.0, v0=0.0, v1=5.0, recflag=0),
        )
    name, bus, b = SHUNT_REACTOR
    ss.add("Shunt", dict(idx=name, bus=bus, Sn=100.0, Vn=HV_KV, b=b, u=1))
    name, bus, b = CAPACITOR
    ss.add("Shunt", dict(idx=name, bus=bus, Sn=100.0, Vn=HV_KV, b=b, u=0))
    name, bus = STATCOM
    ss.add("Shunt", dict(idx=name, bus=bus, Sn=100.0, Vn=HV_KV, b=0.0, u=1))
    name, b1, b2, r, x, bb = PARALLEL_LINE
    ss.add("Line", dict(idx=name, bus1=b1, bus2=b2, r=r, x=x, b=bb, u=0, Vn1=HV_KV, Vn2=HV_KV))
    regularize_degenerate_blocks(ss)
    return ss


def _pflow(taps: tuple[float, ...]) -> Any:
    ss = _load(taps)
    ss.setup()
    if not ss.PFlow.run():
        raise RuntimeError("power flow failed for the core test system")
    return ss


def _block_q(ss: Any) -> np.ndarray:
    idx = list(ss.PV.idx.v)
    return np.array([float(ss.PV.q.v[idx.index(f"PVB_{a}")]) for a, *_ in BLOCKS])


@lru_cache(maxsize=1)
def solved_taps() -> tuple[float, ...]:
    target = np.array([KAPPA * p for _, _, p, _ in BLOCKS])
    t0 = np.full(len(BLOCKS), 0.97)
    t1 = np.full(len(BLOCKS), 0.95)
    q0 = _block_q(_pflow(tuple(t0))) - target
    q1 = _block_q(_pflow(tuple(t1))) - target
    for _ in range(12):
        denom = np.where(np.abs(q1 - q0) > 1e-12, q1 - q0, 1e-12)
        t2 = np.clip(t1 - q1 * (t1 - t0) / denom, 0.85, 1.15)
        t0, q0 = t1, q1
        t1 = t2
        q1 = _block_q(_pflow(tuple(t1))) - target
        if np.all(np.abs(q1) <= 0.01 * np.abs(target)):
            break
    if not np.all(np.abs(q1) <= 0.01 * np.abs(target)):
        raise RuntimeError(f"tap solve did not converge: residual {q1}")
    return tuple(float(t) for t in t1)


def fresh_system(*, tstep: float = 0.002) -> Any:
    ss = _pflow(solved_taps())
    ss.TDS.config.tstep = tstep
    ss.TDS.config.no_tqdm = 1
    ss.TDS.init()
    return ss


def protected_blocks(ss: Any) -> tuple[ProtectedBlock, ...]:
    taps = solved_taps()
    out = []
    pv_idx = list(ss.PV.idx.v)
    for k, ((asset, hv, p, z0), tap) in enumerate(zip(BLOCKS, taps)):
        q = float(ss.PV.q.v[pv_idx.index(f"PVB_{asset}")]) * 100.0
        out.append(
            ProtectedBlock(
                asset_id=asset,
                collector_bus=collector_bus(k),
                hv_bus=hv,
                pvd1_idx=f"PD_{asset}",
                gsu_line_idx=f"GSU_{asset}",
                tap=tap,
                gsu_x_pu_sysbase=_gsu_x_sysbase(p),
                p_mw=p * 100.0,
                q_mvar=q,
                sn_mva=1.15 * p * 100.0,
                z0_pu=z0,
                trip_event_id=f"plant_trip_{asset}",
            )
        )
    return tuple(out)


def _device_value(ss: Any, model: str, idx: Any, attr: str) -> float:
    mod = getattr(ss, model)
    pos = [str(v) for v in mod.idx.v].index(str(idx))
    return float(getattr(mod, attr).v[pos])


def generator_devices(ss: Any, syn: str) -> tuple[tuple[str, str], ...]:
    devs = [("GENROU", syn)]
    exc = [str(i) for i, s in zip(ss.IEEEX1.idx.v, ss.IEEEX1.syn.v) if str(s) == syn]
    devs += [("IEEEX1", e) for e in exc]
    devs += [("TGOV1N", str(i)) for i, s in zip(ss.TGOV1N.idx.v, ss.TGOV1N.syn.v) if str(s) == syn]
    devs += [("IEEEST", str(i)) for i, a in zip(ss.IEEEST.idx.v, ss.IEEEST.avr.v) if str(a) in exc]
    return tuple(devs)


def event_library(ss: Any) -> dict[str, EventSpec]:
    ev: dict[str, EventSpec] = {}
    ev["plant_trip"] = EventSpec(
        "plant_trip_B1", "plant_trip", "switch", (Mutation("PVD1", "PD_B1", "u", 0.0),),
        disconnects_assets=("B1",), description="PVD1 plant trip of block B1",
    )
    ev["generator_trip"] = EventSpec(
        "generator_trip_G4", "generator_trip", "switch",
        generator_trip_mutations(ss, "GENROU_4"),
        description="GENROU_4 with exciter, governor and PSS (u and ue)",
    )
    ev["load_disconnection"] = EventSpec(
        "load_disconnection_PQ_6", "load_disconnection", "switch", (Mutation("PQ", "PQ_6", "u", 0.0),),
        description="PQ_6 at bus 15 (320 MW, 153 Mvar)",
    )
    ev["shunt_reactor_disconnection"] = EventSpec(
        "shunt_reactor_disconnection_SHR1", "shunt_reactor_disconnection", "switch",
        (Mutation("Shunt", "SHR1", "u", 0.0),), description="80 Mvar reactor at bus 15",
    )
    ev["capacitor_switching"] = EventSpec(
        "capacitor_switching_CAP1", "capacitor_switching", "switch",
        (Mutation("Shunt", "CAP1", "u", 1.0),), description="60 Mvar capacitor at bus 21",
    )
    _, _, _, par_r, par_x, _ = PARALLEL_LINE
    yhk = 1.0 / complex(par_r + 1e-8, par_x + 1e-8)
    ev["line_energization"] = EventSpec(
        "line_energization_PAR1", "line_energization", "switch",
        (
            Mutation("Line", "PAR1", "u", 1.0),
            Mutation("Line", "PAR1", "ghk", yhk.real),
            Mutation("Line", "PAR1", "bhk", yhk.imag),
        ),
        description="parallel circuit 16-17",
    )
    p0 = _device_value(ss, "PVD1", "PD_B1", "pref0")
    q0 = _device_value(ss, "PVD1", "PD_B1", "qref0")
    ev["fixed_pf_ramp"] = EventSpec(
        "fixed_pf_ramp_B1", "fixed_pf_ramp", "param",
        (Mutation("PVD1", "PD_B1", "pref0", 0.7 * p0), Mutation("PVD1", "PD_B1", "qref0", 0.7 * q0)),
        profile=TimeProfile(ProfileKind.RAMP, duration_s=0.5),
        description="B1 pref0 and qref0 to 70% over 0.5 s (fixed PF kept)",
    )
    gov = generator_devices(ss, "GENROU_4")
    tg = [i for m, i in gov if m == "TGOV1N"][0]
    pg = _device_value(ss, "TGOV1N", tg, "pref0")
    ev["governor_ramp"] = EventSpec(
        "governor_ramp_G4", "generation_redispatch", "param",
        (Mutation("TGOV1N", tg, "pref0", 0.8 * pg),),
        profile=TimeProfile(ProfileKind.RAMP, duration_s=1.0),
        description="TGOV1N pref0 of GENROU_4 -20% over 1 s",
    )
    ex = [i for m, i in gov if m == "IEEEX1"][0]
    vr = _device_value(ss, "IEEEX1", ex, "vref0")
    ev["exciter_step"] = EventSpec(
        "exciter_step_G4", "avr_vref_step", "param", (Mutation("IEEEX1", ex, "vref0", vr - 0.03),),
        profile=TimeProfile(ProfileKind.STEP), description="IEEEX1 vref0 -0.03 pu step",
    )
    return ev


def statcom_control(*, b_pu: float = -1.0) -> ControlSpec:
    return ControlSpec(
        "statcom_STC1", "statcom", (Mutation("Shunt", STATCOM[0], "b", b_pu),),
        envelope=lag_envelope(0.02, 0.03, horizon_s=3.0),
        alpha_max=1.0, mvar_per_unit=abs(b_pu) * 100.0,
        description="controlled shunt at bus 16, 100 Mvar absorbing",
    )


def output_buses() -> tuple[tuple[str, int], ...]:
    cols = tuple((f"COL_{a}", collector_bus(k)) for k, (a, *_rest) in enumerate(BLOCKS))
    hv = (("HV16", 16), ("HV21", 21), ("HV4", 4), ("HV15", 15), ("HV24", 24))
    return cols + hv
