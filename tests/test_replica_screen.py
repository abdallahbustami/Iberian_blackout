"""Tests of the screening of the 2000-bus system (the slow test builds the armed state, about 2 min)."""

from __future__ import annotations

import math
import os
from types import SimpleNamespace

import numpy as np
import pytest
from scipy import sparse
from scipy.linalg import expm

from pa_dvsa.replication import replica_screen as R

RUN_SLOW = os.environ.get("PA_DVSA_SLOW") == "1"


def test_output_reachable_states_is_exact_on_a_synthetic_system() -> None:
    rng = np.random.default_rng(3)
    n = 9
    a = np.zeros((n, n))
    a[:4, :4] = rng.normal(size=(4, 4)) - 3 * np.eye(4)
    a[4:6, :4] = rng.normal(size=(2, 4))
    a[4:6, 4:6] = -np.eye(2)
    a[6:, 6:] = rng.normal(size=(3, 3))
    c = np.zeros((2, n))
    c[0, 1] = 1.0
    c[1, 3] = 0.5
    keep = R.output_reachable_states(sparse.csc_matrix(a), sparse.csc_matrix(c))
    assert list(keep) == [0, 1, 2, 3]
    b = rng.normal(size=n)
    for t in (0.1, 0.7):
        full = c @ _step(a, b, t)
        red = c[:, keep] @ _step(a[np.ix_(keep, keep)], b[keep], t)
        np.testing.assert_allclose(full, red, atol=1e-12)


def _step(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    n = a.shape[0]
    aug = np.zeros((n + 1, n + 1))
    aug[:n, :n] = a
    aug[:n, n] = b
    return expm(aug * t)[:n, n]


def test_output_reachable_states_follows_chains() -> None:
    a = sparse.csc_matrix(np.array([[-1.0, 1.0, 0.0, 0.0], [0.0, -1.0, 1.0, 0.0], [0.0, 0.0, -1.0, 0.0], [0.0, 0.0, 1.0, -1.0]]))
    c = sparse.csc_matrix(np.array([[1.0, 0.0, 0.0, 0.0]]))
    assert list(R.output_reachable_states(a, c)) == [0, 1, 2]


@pytest.fixture(scope="module")
def core():
    from tests import _core_system as CS

    ss = CS.fresh_system()
    return SimpleNamespace(CS=CS, ss=ss, events=CS.event_library(ss), outputs=CS.output_buses())


def test_pruned_engine_reproduces_the_core_engine(core) -> None:
    from pa_dvsa.event_response import EventResponseEngine, OutputSelector

    sels = [OutputSelector.bus_voltage(core.ss, b, n) for n, b in core.outputs]
    ref = EventResponseEngine(core.ss, sels)
    ours = R.PrunedEventResponseEngine(core.ss, sels)
    d_ref = ref.pre_mode.diagnostics()
    d_ours = ours.pre_mode.extended_diagnostics()
    assert d_ours["n_states_output_reachable"] <= d_ours["n_states_full"] == d_ref.n_states
    assert d_ours["max_real_eigenvalue_full"] == pytest.approx(d_ref.max_real_eigenvalue, abs=1e-8)
    for fam in ("plant_trip", "generator_trip", "load_disconnection", "line_energization", "fixed_pf_ramp",
                "capacitor_switching"):
        ev = core.events[fam]
        r_ref = ref.event_response(ev)
        r_ours = ours.event_response(ev)
        ours.assert_restored()
        np.testing.assert_allclose(r_ours.values_pu, r_ref.values_pu, atol=1e-9, err_msg=fam)
        np.testing.assert_allclose(r_ours.jump_pu, r_ref.jump_pu, atol=1e-12, err_msg=fam)
        for e1, e2 in zip(r_ours.envelopes, r_ref.envelopes):
            assert e1.peak_upper >= e1.sampled_max - 1e-15
            assert e1.sampled_max == pytest.approx(e2.sampled_max, abs=1e-9)
        errs = R.restoration_errors(ours)
        assert max(errs.values()) <= 1e-9


def test_snapshot_restore_reproduces_state_and_tds(core) -> None:
    from pa_dvsa.tds_runner import run_protection_tds

    CS = core.CS
    src = CS.fresh_system()
    pos = list(src.PVD1.idx.v).index("PD_B1")
    src.PVD1.pref0.v[pos] *= 0.97
    src.TDS.config.tf = 0.2
    src.TDS.run(no_summary=True)
    snap = R.snapshot_arrays(src)
    f_src, g_src = R._residuals(src)
    dst = CS.fresh_system()
    assert R.arrays_diff(R.snapshot_arrays(dst), snap)
    R.restore_arrays(dst, snap)
    assert not R.arrays_diff(R.snapshot_arrays(dst), snap)
    f_dst, g_dst = R._residuals(dst)
    np.testing.assert_array_equal(f_dst, f_src)
    np.testing.assert_array_equal(g_dst, g_src)
    assert float(dst.dae.t) == 0.0 and dst.TDS.initialized

    blocks = CS.protected_blocks(dst)

    def factory():
        ss = CS.fresh_system()
        R.restore_arrays(ss, snap)
        return ss

    runs = [run_protection_tds(factory, core.events["load_disconnection"], blocks, relays_enabled=False, t_seed=0.05,
                               window_s=0.1, record_traces=True) for _ in range(2)]
    assert all(r.status == "ok" for r in runs)
    np.testing.assert_array_equal(runs[0].traces["v_collector"], runs[1].traces["v_collector"])
    with pytest.raises(RuntimeError):
        R.restore_arrays(dst, {**snap, "PVD1.pref0.v": np.zeros(snap["PVD1.pref0.v"].size + 1)})


def test_event_library_corrections(core) -> None:
    from pa_dvsa.events import EventSpec, Mutation

    ss = core.ss
    gen = EventSpec("generator_trip:GENROU_4", "generator_trip", "switch",
                    tuple(Mutation(m, i, "u", 0.0) for m, i in core.CS.generator_devices(ss, "GENROU_4")))
    line = EventSpec("line_energization:PAR1", "line_energization", "switch", (Mutation("Line", "PAR1", "u", 1.0),))
    other = core.events["plant_trip"]
    out, notes = R.corrected_event_library(ss, [gen, line, other])
    assert [e.event_id for e in out] == [gen.event_id, line.event_id, other.event_id]
    attrs = {(m.model, m.attr) for m in out[0].mutations}
    assert ("IEEEX1", "ue") in attrs and ("TGOV1N", "ue") in attrs and ("GENROU", "u") in attrs
    muts = {m.attr: m.value for m in out[1].mutations}
    p = list(ss.Line.idx.v).index("PAR1")
    y = 1.0 / complex(float(ss.Line.r.v[p]) + 1e-8, float(ss.Line.x.v[p]) + 1e-8)
    assert muts == {"u": 1.0, "ghk": pytest.approx(y.real), "bhk": pytest.approx(y.imag)}
    assert float(ss.Line.ghk.v[p]) == 0.0
    assert out[2] is other
    assert len(notes) == 2


def _pred(sid: str, classes: dict, excluded=()):
    return SimpleNamespace(seed_id=sid, classes=classes, excluded_assets=tuple(excluded), max_k_pk_nonself=0.5,
                           max_k_tr_nonself=0.4, verdict="x", runtime_s=0.01, predicted_trips=())


def _cls(c: str, trip: bool = False, score: float = 0.5):
    return {"asset_class": c, "predicted_trip": trip, "score": score, "layer": 1 if trip else None,
            "predicted_time_s": 0.1 if trip else None}


def _tds(trips: dict, status="ok"):
    assets = [{"asset_id": a, "trip_rel_s": t, "first_pickup_rel_s": (None if t is None else t - 0.05),
               "max_v_window_pu": 1.1, "max_v_after_pu": 1.1, "trip_s": None if t is None else 0.5 + t}
              for a, t in trips.items()]
    return {"converged": status == "ok", "status": status, "assets": assets, "trip_order": [a for a, t in trips.items() if t],
            "t_seed_s": 0.5, "job_wall_s": 10.0, "build_s": 2.0, "sim_s": 8.0, "pre_seed_max_dv_pu": 0.0}


def test_score_validation_outcomes_and_no_margin_flag() -> None:
    store = SimpleNamespace(slug="s", families={"e1": "plant_trip", "e2": "load_disconnection", "e3": "x"},
                            event_runtime_s={"e1": 1.0, "e2": 1.0, "e3": 1.0})
    relay = R.RELAYS["fast"]
    preds = {
        "e1": _pred("e1", {"B": _cls("pickup_predicted", True, 1.3), "C": _cls("certified"), "D": _cls("data_limited")},
                    excluded=("A",)),
        "e2": _pred("e2", {"A": _cls("certified"), "B": _cls("risky", False, 0.97), "C": _cls("certified"),
                           "D": _cls("data_limited")}),
        "e3": _pred("e3", {"A": _cls("certified"), "B": _cls("certified"), "C": _cls("certified"), "D": _cls("data_limited")}),
    }
    tds = {"e1": _tds({"A": None, "B": 0.2, "C": 1.4, "D": None}), "e2": _tds({"A": None, "B": None, "C": 0.6, "D": None}),
           "e3": _tds({"A": None, "B": None, "C": None, "D": None}, status="failed")}
    rows, seeds, agg = R.score_validation(store, None, relay, preds, tds, no_margin=["D"])
    out = {(r["seed_id"], r["asset_id"]): r["outcome"] for r in rows}
    assert out[("e1", "B")] == "TP" and out[("e1", "C")] == "late_trip" and out[("e1", "D")] == "data_limited_clear"
    assert ("e1", "A") not in out
    assert out[("e2", "C")] == "unsafe_FN" and out[("e2", "B")] == "risky_clear" and out[("e2", "A")] == "TN"
    assert all(v == "tds_inconclusive" for (s, _), v in out.items() if s == "e3")
    assert all(r["no_margin_blocked"] == (r["asset_id"] == "D") for r in rows)
    s = {x["seed_id"]: x for x in seeds}
    assert s["e1"]["jaccard"] == 1.0 and s["e1"]["dangerous"] and s["e2"]["dangerous"]
    assert s["e2"]["jaccard"] == 0.0 and not s["e2"]["predicted_dangerous"]
    assert s["e1"]["late_trips"] == "C"
    tot = next(a for a in agg if a["family"] == "ALL")
    assert tot["TP"] == 1 and tot["unsafe_FN"] == 1 and tot["late_trip"] == 1 and tot["n_seeds_inconclusive"] == 1


def test_fig9_family_rows() -> None:
    rows = [
        {"slug": "s", "relay": "fast", "family": "plant_trip", "event_id": "p1", "max_k_pk_nonself": 1.2, "argmax_asset": "B", "status": "ok"},
        {"slug": "s", "relay": "fast", "family": "plant_trip", "event_id": "p2", "max_k_pk_nonself": 0.3, "argmax_asset": "C", "status": "ok"},
        {"slug": "s", "relay": "fast", "family": "plant_trip", "event_id": "p3", "max_k_pk_nonself": 0.96, "argmax_asset": "C", "status": "ok"},
        {"slug": "s", "relay": "fast", "family": "line_energization", "event_id": "l1", "max_k_pk_nonself": float("nan"), "argmax_asset": "", "status": "failed"},
        {"slug": "s", "relay": "fast", "family": "base_state_trips", "event_id": "b", "max_k_pk_nonself": 9.0, "argmax_asset": "", "status": "ok"},
    ]
    fam = {r["family"]: r for r in R.fig9_family_rows(rows)}
    assert set(fam) == {"plant_trip", "line_energization"}
    assert fam["plant_trip"]["worst_k_pk_nonself"] == 1.2 and fam["plant_trip"]["worst_event"] == "p1"
    assert fam["plant_trip"]["n_events_k_pk_ge1"] == 1 and fam["plant_trip"]["n_events_k_pk_ge_1meps"] == 2
    assert math.isnan(fam["line_energization"]["worst_k_pk_nonself"]) and fam["line_energization"]["n_failed"] == 1


def test_relay_settings_and_slugs() -> None:
    assert R.RELAYS["fast"].pickup_pu == 1.10 and R.RELAYS["fast"].dwell_s == 0.05 and R.RELAYS["fast"].horizon_s == 1.0
    assert R.RELAYS["ov1"].dwell_s == 1.0 and R.RELAYS["ov1"].horizon_s == 2.0 <= R.ENGINE_HORIZON_S
    assert R.RELAYS["ov2"].pickup_pu == 1.20 and R.RELAYS["ov2"].dwell_s == 0.16
    assert R.slug_of("C1.5", 0.02) == "C1p5_m0p02" and R.slug_of("stock", 0.01) == "stock_m0p01"


@pytest.mark.slow
@pytest.mark.skipif(not RUN_SLOW, reason="set PA_DVSA_SLOW=1 (builds the armed ACTIVSg2000 replica, ~3-4 min)")
def test_replica_plant_trip_jump_matches_native_tds_and_20ms_response() -> None:
    case, armed_ss, ver = R.build_armed_case("stock", 0.02)
    assert ver["arrays_identical"] and ver["residuals_bitwise_identical"]
    fac = R.SnapshotFactory(case)
    ev = case.event("plant_trip:Badajoz582")
    store, engine = R.run_screen(case, fac, event_ids=[ev.event_id])
    pre = store.mode_diagnostics["pre_event"]
    assert pre["max_real_eigenvalue_full"] <= 1e-6
    assert max(store.restoration[ev.event_id].values()) <= 1e-9
    chk = R.jump_branch_check(engine, ev, fac, store.responses[ev.event_id])
    assert chk["native_ok"]
    assert chk["max_abs_dy_direct_vs_continuation"] < 1e-8
    assert chk["max_abs_dy_direct_vs_native"] < 1e-4
    assert chk["jump_screen_vs_native_max_err_mpu"] < 0.05
    assert chk["dz20_screen_vs_native_max_err_mpu"] < 0.5
