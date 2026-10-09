"""Levels of information on the relay voltage and the uncertainty sweeps of the benchmark studies."""

from __future__ import annotations

from collections import Counter
from dataclasses import replace
import itertools
import math
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.stats import kendalltau

from ..cascade_certificate import screen_seed
from ..events import lag_envelope
from ..protected_outputs import ProtectedOutputModel
from ..robust_screening import UncertaintyIntervals, robust_window_bounds
from . import config as C
from .screen import ScreenResult

REGIMES = ("telemetry", "measured_pq", "known_tap", "bounded_tap", "transmission_only")
SENSITIVITY_REGIMES = ("known_tap_symmetric", "measured_pq_z5", "measured_pq_z20", "measured_pq_bounded_tap")
PQ_REGIMES = {
    "measured_pq": (C.GSU_Z_REL, 0.0),
    "measured_pq_z5": (C.GSU_Z_REL_SENSITIVITY[0], 0.0),
    "measured_pq_z20": (C.GSU_Z_REL_SENSITIVITY[1], 0.0),
    "measured_pq_bounded_tap": (C.GSU_Z_REL, C.TAP_BAND_REL),
}


def gsu_drop_interval(hidden: Any, block: Any) -> dict[str, Any]:
    r = float(hidden.gsu_r_pu_sysbase)
    x = float(hidden.gsu_x_pu_sysbase)
    kappa = float(block.q_mvar) / float(block.p_mw) if float(block.p_mw) != 0.0 else 0.0
    ilim = float(hidden.drop_bound_ilim_pu)
    absorbing = (r + kappa * x) <= 0.0 and float(block.p_mw) > 0.0
    hi = 0.0 if absorbing else ilim
    return {
        "drop_lower_pu": -ilim,
        "drop_upper_pu": hi,
        "drop_op_pu": float(hidden.drop_op_pu),
        "drop_bound_op_pu": float(hidden.drop_bound_op_pu),
        "drop_bound_ilim_pu": ilim,
        "r_plus_kappa_x": r + kappa * x,
        "upper_rule": "sign (r P + x Q <= 0)" if absorbing else "|Z| I_lim",
    }


def gsu_reconstruct(v_hv: float, p_hv: float, q_hv: float, r: float, x: float, tap: float) -> float:
    vi = float(v_hv) / float(tap)
    return math.hypot(vi + (r * p_hv + x * q_hv) / vi, (x * p_hv - r * q_hv) / vi)


def plant_hv_power(hidden: Any, block: Any) -> tuple[float, float]:
    pc, qc = float(block.p_mw) / C.BASE_MVA, float(block.q_mvar) / C.BASE_MVA
    i2 = (pc * pc + qc * qc) / float(hidden.z0_pu) ** 2
    return pc - float(hidden.gsu_r_pu_sysbase) * i2, qc - float(hidden.gsu_x_pu_sysbase) * i2


def pq_error_interval(hidden: Any, block: Any, z_rel: float, tap_band: float = 0.0) -> dict[str, Any]:
    r, x, n0 = float(hidden.gsu_r_pu_sysbase), float(hidden.gsu_x_pu_sysbase), float(hidden.tap)
    v_hv, z0 = float(hidden.v_hv_op_pu), float(hidden.z0_pu)
    p_hv, q_hv = plant_hv_power(hidden, block)
    z_nom = gsu_reconstruct(v_hv, p_hv, q_hv, r, x, n0)
    if abs(z_nom - z0) > 1e-6:
        raise ValueError(f"{hidden.asset_id}: GSU reconstruction {z_nom:.8f} != power-flow collector voltage {z0:.8f}")
    taps = (-1, 1) if tap_band > 0.0 else (0,)
    vals = [gsu_reconstruct(v_hv + sv * C.EPS_MEAS_PU, p_hv * (1 + sp * C.EPS_PQ_REL), q_hv * (1 + sq * C.EPS_PQ_REL),
                            r * (1 + sz * z_rel), x * (1 + sz * z_rel), n0 * (1 + sn * tap_band))
            for sv, sp, sq, sz, sn in itertools.product((-1, 1), (-1, 1), (-1, 1), (-1, 1), taps)]
    return {"e_lo": min(vals) - z0, "e_hi": max(vals) - z0, "p_hv_pu": p_hv, "q_hv_pu": q_hv,
            "z_reconstructed_pu": z_nom, "gsu_z_rel": z_rel, "tap_band_rel": tap_band}


def apply_impedance_reserve(scr: ScreenResult, case: Any, bounds: Any, z_rel: float) -> Any:
    hid = {h.asset_id: h for h in case.hidden_output_data}
    rec = {m.asset_id: m.output_index for m in scr.models_reconstructed}
    col = {m.asset_id: m.output_index for m in scr.models_direct}
    for i, a in enumerate(bounds.asset_ids):
        n0 = float(hid[a].tap)
        for j, e in enumerate(bounds.event_ids):
            if bounds.classes[i, j] != "certified":
                continue
            v = scr.responses[e].values_pu
            dd = float(np.max(np.abs(v[:, col[a]] - v[:, rec[a]] / n0)))
            add = z_rel * dd / float(bounds.assets[i].h_lower_pu)
            bounds.k_bar_pk[i, j] += add
            bounds.k_bar_tr[i, j] += add
    return bounds


def _recon_models(scr: ScreenResult, taps: Mapping[str, float]) -> tuple[ProtectedOutputModel, ...]:
    return tuple(replace(m, tap_nominal=float(taps[m.asset_id])) for m in scr.models_reconstructed)


def regime_setup(scr: ScreenResult, case: Any, regime: str) -> tuple[tuple[ProtectedOutputModel, ...], UncertaintyIntervals, dict[str, dict[str, Any]]]:
    hid = {h.asset_id: h for h in case.hidden_output_data}
    blk = {b.asset_id: b for b in case.blocks}
    info: dict[str, dict[str, Any]] = {}
    em = C.EPS_MEAS_PU
    if regime == "telemetry":
        theta = UncertaintyIntervals(reconstruction_error_pu={"*": (-em, em)}, set_id="telemetry")
        for a in scr.asset_ids:
            info[a] = {"kind": "direct", "tap_lo": float("nan"), "tap_hi": float("nan"), "e_lo": -em, "e_hi": em}
        return scr.models_direct, theta, info
    if regime in PQ_REGIMES:
        z_rel, band = PQ_REGIMES[regime]
        err: dict[str, tuple[float, float]] = {}
        for a in scr.asset_ids:
            d = pq_error_interval(hid[a], blk[a], z_rel, band)
            err[a] = (d["e_lo"], d["e_hi"])
            info[a] = {"kind": "measured_pq", "tap_lo": float(hid[a].tap) * (1 - band), "tap_hi": float(hid[a].tap) * (1 + band),
                       **d}
        theta = UncertaintyIntervals(reconstruction_error_pu=err, set_id=regime)
        return scr.models_direct, theta, info
    taps0 = {a: float(hid[a].tap) for a in scr.asset_ids}
    models = _recon_models(scr, taps0)
    tap_iv: dict[str, tuple[float, float]] = {}
    err_iv: dict[str, tuple[float, float]] = {}
    for a in scr.asset_ids:
        d = gsu_drop_interval(hid[a], blk[a])
        n0 = taps0[a]
        if regime == "known_tap":
            tap_iv[a] = (n0, n0)
            err_iv[a] = (d["drop_lower_pu"] - em, d["drop_upper_pu"] + em)
        elif regime == "bounded_tap":
            tap_iv[a] = (n0 * (1.0 - C.TAP_BAND_REL), n0 * (1.0 + C.TAP_BAND_REL))
            err_iv[a] = (d["drop_lower_pu"] - em, d["drop_upper_pu"] + em)
        elif regime == "transmission_only":
            tap_iv[a] = tuple(C.TRANSMISSION_ONLY_TAP)
            err_iv[a] = tuple(C.TRANSMISSION_ONLY_ERR_PU)
        elif regime == "known_tap_symmetric":
            tap_iv[a] = (n0, n0)
            err_iv[a] = (-d["drop_bound_op_pu"] - em, d["drop_bound_op_pu"] + em)
        else:
            raise ValueError(f"unknown regime {regime!r}")
        info[a] = {"kind": "reconstructed", "tap_lo": tap_iv[a][0], "tap_hi": tap_iv[a][1],
                   "e_lo": err_iv[a][0], "e_hi": err_iv[a][1], **d}
    theta = UncertaintyIntervals(tap_ratio=tap_iv, reconstruction_error_pu=err_iv, set_id=regime)
    return models, theta, info


def bounds_for(scr: ScreenResult, models: Sequence[ProtectedOutputModel], theta: UncertaintyIntervals) -> Any:
    return robust_window_bounds(
        scr.response_list, models, theta, event_order=[e for e in scr.event_order if e in scr.responses],
        horizon_s=C.T_WINDOW_S,
    )


def regime_bounds(scr: ScreenResult, case: Any, regime: str) -> tuple[Any, tuple[ProtectedOutputModel, ...], UncertaintyIntervals, dict[str, dict[str, Any]]]:
    models, theta, info = regime_setup(scr, case, regime)
    b = bounds_for(scr, models, theta)
    if regime in PQ_REGIMES:
        apply_impedance_reserve(scr, case, b, PQ_REGIMES[regime][0])
    return b, models, theta, info


def uncertainty_bounds(scr: ScreenResult, case: Any, models: Sequence[ProtectedOutputModel], theta: UncertaintyIntervals) -> Any:
    b = bounds_for(scr, models, theta)
    if str(theta.set_id).startswith("measured_pq"):
        apply_impedance_reserve(scr, case, b, C.GSU_Z_REL)
    return b


def asset_rows(scr: ScreenResult, bounds: Any, info: Mapping[str, Mapping[str, Any]], regime: str) -> list[dict[str, Any]]:
    rows = []
    for i, (a, ab) in enumerate(zip(bounds.asset_ids, bounds.assets)):
        best, arg = -math.inf, None
        for j, e in enumerate(bounds.event_ids):
            if bounds.classes[i, j] != "certified":
                continue
            v = float(bounds.k_bar_pk[i, j])
            if math.isfinite(v) and v > best:
                best, arg = v, e
        if ab.data_limited:
            cls = "data_limited"
            best = float("nan")
        elif best >= 1.0:
            cls = "pickup_predicted"
        elif best >= 1.0 - C.EPSILON:
            cls = "risky"
        else:
            cls = "certified"
        rows.append({
            "regime": regime, "asset_id": a, **{k: info[a].get(k) for k in ("kind", "tap_lo", "tap_hi", "e_lo", "e_hi")},
            "h_nominal_pu": ab.h_nominal_pu, "h_lower_pu": ab.h_lower_pu, "z0_upper_pu": ab.z0_upper_pu,
            "response_scale_upper": ab.response_scale_upper, "data_limited": ab.data_limited,
            "max_kbar_pk_nonself": best if arg is not None or ab.data_limited else 0.0,
            "argmax_event": arg, "asset_class": cls,
        })
    return rows


def seed_verdicts(scr: ScreenResult, bounds: Any) -> dict[str, dict[str, Any]]:
    out = {}
    for e in scr.event_order:
        if e not in scr.responses:
            continue
        cls = screen_seed(bounds, e, asset_event_map=scr.amap, epsilon=C.EPSILON, responses=None).classification
        cnt = Counter(c.asset_class for c in cls.classes)
        out[e] = {
            "n_pickup_predicted": cnt.get("pickup_predicted", 0), "n_risky": cnt.get("risky", 0),
            "n_certified": cnt.get("certified", 0), "n_data_limited": cnt.get("data_limited", 0),
            "fixed_point_size": len(cls.fixed_point),
            "certified_no_secondary": all(c.asset_class == "certified" for c in cls.classes),
        }
    return out


def seed_scores(bounds: Any, seeds: Sequence[str]) -> dict[str, tuple[float, float]]:
    out = {}
    for e in seeds:
        j = list(bounds.event_ids).index(e)
        best_pk, best_tr = -math.inf, -math.inf
        for i in range(len(bounds.asset_ids)):
            if bounds.classes[i, j] != "certified":
                continue
            v, w = float(bounds.k_bar_pk[i, j]), float(bounds.k_bar_tr[i, j])
            if math.isfinite(v):
                best_pk = max(best_pk, v)
            if math.isfinite(w):
                best_tr = max(best_tr, w)
        out[e] = (best_pk if best_pk > -math.inf else float("nan"), best_tr if best_tr > -math.inf else float("nan"))
    return out


def ranking(scores: Mapping[str, tuple[float, float]], order: Sequence[str]) -> list[str]:
    pos = {e: k for k, e in enumerate(order)}
    fin = [e for e in order if math.isfinite(scores[e][0])]
    return sorted(fin, key=lambda e: (-scores[e][0], -(scores[e][1] if math.isfinite(scores[e][1]) else -math.inf), pos[e]))


def rank_compare(nominal: Mapping[str, tuple[float, float]], other: Mapping[str, tuple[float, float]], order: Sequence[str], top: int = 5) -> dict[str, Any]:
    common = [e for e in order if math.isfinite(nominal[e][0]) and math.isfinite(other[e][0])]
    if len(common) >= 2:
        tau = kendalltau([nominal[e][0] for e in common], [other[e][0] for e in common]).statistic
    else:
        tau = float("nan")
    rn, ro = ranking(nominal, order)[:top], ranking(other, order)[:top]
    k = min(top, len(rn))
    overlap = (len(set(rn) & set(ro)) / k) if k else float("nan")
    return {"kendall_tau": float(tau) if tau is not None else float("nan"), "top5_overlap": overlap,
            "n_ranked": len(common), "top5_nominal": rn, "top5_level": ro}


def scaled_controls(case: Any, factor: float) -> list[Any]:
    out = []
    for c in case.controls:
        delay = float(c.metadata.get("delay_s", 0.0))
        tau = float(c.metadata.get("tau_s", 0.0))
        samples = c.envelope.samples
        if not samples:
            raise ValueError(f"{c.control_id}: envelope is not sampled; cannot rescale timing")
        dt = float(samples[1][0] - samples[0][0]) if len(samples) > 1 else 1e-3
        horizon = float(samples[-1][0]) + delay * (factor - 1.0)
        env = lag_envelope(delay * factor, tau * factor, horizon_s=horizon, dt_s=dt)
        out.append(replace(c, envelope=env))
    return out


def uncertainty_theta(case: Any, scr: ScreenResult, cls: str, level: float) -> tuple[tuple[ProtectedOutputModel, ...], UncertaintyIntervals, str]:
    em = C.EPS_MEAS_PU
    l = float(level)
    if cls == "relay_threshold":
        w = C.UNC_RELAY_THRESHOLD_PU * l
        th = UncertaintyIntervals(reconstruction_error_pu={"*": (-em, em)},
                                  relay_threshold_pu={a: (b.pickup_pu - w, b.pickup_pu + w) for a, b in zip(scr.asset_ids, case.blocks)},
                                  set_id=f"{cls}@{l}")
        return scr.models_direct, th, f"V_trip +/- {w:.4f} pu"
    if cls == "tap_ratio":
        th = UncertaintyIntervals(reconstruction_error_pu={"*": (-em, em)},
                                  tap_ratio={b.asset_id: (b.tap * (1 - C.UNC_TAP_REL * l), b.tap * (1 + C.UNC_TAP_REL * l)) for b in case.blocks},
                                  set_id=f"{cls}@{l}")
        return scr.models_direct, th, f"n0 (1 +/- {C.UNC_TAP_REL * l:.4f}) (direct output: no effect)"
    if cls == "tap_ratio_reconstructed":
        hid = {h.asset_id: h for h in case.hidden_output_data}
        blk = {bb.asset_id: bb for bb in case.blocks}
        band = C.UNC_TAP_REL * l
        err = {}
        for a in scr.asset_ids:
            d = pq_error_interval(hid[a], blk[a], C.GSU_Z_REL, band)
            err[a] = (d["e_lo"], d["e_hi"])
        th = UncertaintyIntervals(reconstruction_error_pu=err, set_id=f"measured_pq:{cls}@{l}")
        return scr.models_direct, th, f"measured-P,Q regime, n0 (1 +/- {band:.4f})"
    if cls == "reconstruction_error":
        w = em + C.UNC_RECONSTRUCTION_PU * l
        th = UncertaintyIntervals(reconstruction_error_pu={"*": (-w, w)}, set_id=f"{cls}@{l}")
        return scr.models_direct, th, f"e +/- {w:.4f} pu"
    if cls == "q_absorption":
        s = C.UNC_Q_ABSORPTION * l
        trips = {b.trip_event_id: (1.0 - s, 1.0 + s) for b in case.blocks}
        th = UncertaintyIntervals(reconstruction_error_pu={"*": (-em, em)}, q_absorption_scale=trips, set_id=f"{cls}@{l}")
        return scr.models_direct, th, f"plant-trip responses x [{1 - s:.3f}, {1 + s:.3f}]"
    if cls == "controller_timing":
        th = UncertaintyIntervals(reconstruction_error_pu={"*": (-em, em)}, set_id=f"{cls}@{l}")
        return scr.models_direct, th, f"control delays and lags x {1 + l:.2f}"
    raise ValueError(cls)


UNCERTAINTY_CLASSES = ("relay_threshold", "tap_ratio", "tap_ratio_reconstructed", "reconstruction_error",
                       "q_absorption", "controller_timing")


def robust_margin_metrics(bounds: Any) -> dict[str, Any]:
    vals = []
    for i, ab in enumerate(bounds.assets):
        if ab.data_limited:
            continue
        row = [float(bounds.k_bar_pk[i, j]) for j in range(len(bounds.event_ids)) if bounds.classes[i, j] == "certified"]
        row = [v for v in row if math.isfinite(v)]
        vals.append(1.0 - (max(row) if row else 0.0))
    h = np.array([ab.h_lower_pu for ab in bounds.assets], dtype=float)
    return {
        "min_robust_margin": float(min(vals)) if vals else float("nan"),
        "min_h_lower_pu": float(np.min(h)) if h.size else float("nan"),
        "median_h_lower_pu": float(np.median(h)) if h.size else float("nan"),
        "n_assets_data_limited": int(sum(1 for ab in bounds.assets if ab.data_limited)),
    }
