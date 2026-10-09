"""Tests of the PV/PQ complementarity power flow."""

from __future__ import annotations

import numpy as np
import pytest

andes = pytest.importorskip("andes")

from pa_dvsa.replication.robust_pflow import complementarity_pflow, damped_pflow


def _ieee14(qmax_scale: float = 1.0, qmin_override: dict[int, float] | None = None):
    ss = andes.load(andes.get_case("ieee14/ieee14.raw"), setup=False, no_output=True, default_config=True)
    for i in range(ss.PV.n):
        ss.PV.qmax.v[i] = float(ss.PV.qmax.v[i]) * qmax_scale
    for i, q in (qmin_override or {}).items():
        ss.PV.qmin.v[i] = q
    ss.setup()
    ss.PFlow.config.tol = 1e-10
    return ss


def _check_complementarity(ss, tol_v=1e-4, tol_q=1e-4):
    pv = ss.PV
    lim = pv.qlim
    q = np.asarray(pv.q.v)
    v = np.asarray(pv.v.v)
    vset = np.asarray(pv.v0.v)
    qmin = np.asarray(pv.qmin.v)
    qmax = np.asarray(pv.qmax.v)
    zi, zl, zu = np.asarray(lim.zi), np.asarray(lim.zl), np.asarray(lim.zu)
    for k in range(pv.n):
        if zi[k] > 0.5:
            assert abs(v[k] - vset[k]) < 1e-8
            assert qmin[k] - tol_q <= q[k] <= qmax[k] + tol_q
        elif zu[k] > 0.5:
            assert abs(q[k] - qmax[k]) < 1e-8 and v[k] <= vset[k] + tol_v
        else:
            assert zl[k] > 0.5
            assert abs(q[k] - qmin[k]) < 1e-8 and v[k] >= vset[k] - tol_v


def test_damped_newton_matches_andes_newton_without_limits():
    ss = _ieee14()
    ss.PV.qlim.enable = False
    ss.PV.qlim.zi[:] = 1.0
    ss.PV.qlim.zl[:] = 0.0
    ss.PV.qlim.zu[:] = 0.0
    assert damped_pflow(ss, tol=1e-10)
    v_damped = np.array(ss.Bus.v.v)
    ref = _ieee14()
    ref.PV.qlim.enable = False
    assert ref.PFlow.run()
    assert np.max(np.abs(v_damped - np.asarray(ref.Bus.v.v))) < 1e-8


def test_complementarity_solution_with_binding_limits():
    ss = _ieee14(qmax_scale=0.3)
    ok, info = complementarity_pflow(ss, tol=1e-10)
    assert ok, info
    assert info["n_pq"] >= 1
    assert info["complementarity_residual_units"] == 0
    _check_complementarity(ss)


def test_complementarity_is_deterministic():
    a = _ieee14(qmax_scale=0.3)
    b = _ieee14(qmax_scale=0.3)
    assert complementarity_pflow(a, tol=1e-10)[0] and complementarity_pflow(b, tol=1e-10)[0]
    assert np.array_equal(np.asarray(a.Bus.v.v), np.asarray(b.Bus.v.v))


def test_back_switching_releases_a_unit_whose_voltage_is_on_the_wrong_side():
    ss = _ieee14(qmax_scale=0.5, qmin_override={0: -0.05})
    ok, info = complementarity_pflow(ss, tol=1e-10, max_new_pq=5)
    assert ok, info
    _check_complementarity(ss)
