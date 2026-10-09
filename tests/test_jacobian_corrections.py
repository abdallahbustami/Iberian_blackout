"""Tests of the Jacobian corrections and of the finite-difference check."""

from __future__ import annotations

import unittest

import numpy as np

from pa_dvsa.benchmarks import build_benchmark
from pa_dvsa.data_model import ProfileKind, TimeProfile
from pa_dvsa.event_response import EventResponseEngine, OutputSelector
from pa_dvsa.events import EventSpec, Mutation
from pa_dvsa.jacobian_audit import audit_jacobian
from pa_dvsa.tds_runner import run_protection_tds


def _pairs(result) -> set[tuple[str, str]]:
    return {(g["equation"], g["variable"]) for g in result.groups()}


class PeggedStateFreezeTests(unittest.TestCase):
    def test_pegged_antiwindup_rows_are_zeroed(self) -> None:
        from types import SimpleNamespace

        from scipy import sparse

        from pa_dvsa.jacobian_corrections import apply_corrections

        fx = sparse.csc_matrix(np.array([[-1.0, 2.0, 0.0], [0.5, -3.0, 1.0], [0.0, 1.0, -2.0]]))
        fy = sparse.csc_matrix(np.array([[1.0], [4.0], [0.0]]))
        gx = sparse.csc_matrix(np.array([[1.0, 1.0, 1.0]]))
        gy = sparse.csc_matrix(np.array([[-1.0]]))
        aw = SimpleNamespace(x_set=[(np.array([1]), np.array([0.7]), 0)])
        fake = SimpleNamespace(antiwindups=[aw])
        out = apply_corrections(fake, {"fx": fx, "fy": fy, "gx": gx, "gy": gy})
        np.testing.assert_array_equal(out["fx"].toarray()[1], 0.0)
        np.testing.assert_array_equal(out["fy"].toarray()[1], 0.0)
        np.testing.assert_array_equal(out["fx"].toarray()[[0, 2]], fx.toarray()[[0, 2]])
        np.testing.assert_array_equal(out["gx"].toarray(), gx.toarray())
        kept = apply_corrections(fake, {"fx": fx, "fy": fy, "gx": gx, "gy": gy}, freeze_pegged=False)
        np.testing.assert_array_equal(kept["fx"].toarray(), fx.toarray())


class JacobianCorrectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.case = build_benchmark("kundur", variant="voltage_mode_res")

    def test_droop_branch_is_active_at_the_operating_point(self) -> None:
        pvd = self.case.system.PVD1
        self.assertTrue(np.all(np.asarray(pvd.VQ2.zi) == 1.0))
        self.assertTrue(np.all(np.asarray(pvd.dqdv.v) > 0.0))

    def test_raw_jacobian_misses_varservice_terms_and_correction_fixes_them(self) -> None:
        ss = self.case.system
        raw = audit_jacobian(ss, label="kundur/vmr raw", corrected=False, columns="all")
        self.assertTrue(raw.full)
        self.assertEqual(
            _pairs(raw), {("PVD1.Qdrp", "Bus.v"), ("PVD1.Ipmax", "PVD1.Iqcmd_y")}
        )
        self.assertTrue(all(e.classification == "missing_derivative" for e in raw.entries))
        for e in raw.entries:
            if e.row_label[1] == "Qdrp":
                pos = list(ss.PVD1.idx.v).index(e.row_label[2]) if e.row_label[2] in list(ss.PVD1.idx.v) else None
                if pos is not None:
                    self.assertAlmostEqual(e.fd, -float(ss.PVD1.dqdv.v[pos]), places=5)
        for basis in ("column", "entry"):
            fixed = audit_jacobian(ss, label="kundur/vmr", corrected=True, columns="all", tolerance_basis=basis)
            self.assertEqual(len(fixed.entries), 0, fixed.groups())

    def test_nonzero_xc_droop_derivative_includes_angle_and_currents(self) -> None:
        ss = self.case.factory()
        pvd = ss.PVD1
        pvd.xc.v[:] = 0.01
        pvd.v1.v[:] = np.asarray(pvd.v1.v) - 0.02
        ss.TDS.fg_update(ss.exist.pflow_tds)
        ss.j_update(models=ss.exist.pflow_tds)
        self.assertTrue(np.all(np.asarray(pvd.VQ2.zi) == 1.0))
        cols = []
        for i in range(pvd.n):
            cols += [("y", int(pvd.v.a[i])), ("y", int(pvd.a.a[i])),
                     ("x", int(pvd.Ipout_y.a[i])), ("x", int(pvd.Iqout_y.a[i]))]
        raw = audit_jacobian(ss, corrected=False, columns=cols, tolerance_basis="entry")
        missing = {(g["equation"], g["variable"]) for g in raw.groups()}
        for var in ("Bus.v", "Bus.a", "PVD1.Ipout_y", "PVD1.Iqout_y"):
            self.assertIn(("PVD1.Qdrp", var), missing)
        fixed = audit_jacobian(ss, corrected=True, columns=cols, tolerance_basis="entry")
        self.assertEqual(len(fixed.entries), 0, fixed.groups())

    def test_voltage_mode_screen_matches_tds_only_with_the_correction(self) -> None:
        case = self.case
        ss = case.system
        blk = case.blocks[0]
        pos = list(ss.PVD1.idx.v).index(blk.pvd1_idx)
        spec = EventSpec(
            "pf_1pct", "fixed_pf_ramp", "param",
            (Mutation("PVD1", blk.pvd1_idx, "pref0", 0.99 * float(ss.PVD1.pref0.v[pos])),
             Mutation("PVD1", blk.pvd1_idx, "qref0", 0.99 * float(ss.PVD1.qref0.v[pos]))),
            profile=TimeProfile(ProfileKind.RAMP, duration_s=0.5),
        )
        outs = [OutputSelector.bus_voltage(ss, blk.collector_bus)]
        cor = EventResponseEngine(ss, outs).event_response(spec)
        raw = EventResponseEngine(ss, outs, jacobian_corrections=False).event_response(spec)
        addr = {"col": int(ss.Bus.v.a[list(ss.Bus.idx.v).index(blk.collector_bus)])}
        tds = run_protection_tds(case.factory, spec, case.blocks, relays_enabled=False, t_seed=0.5,
                                 t_end=1.5, tstep=0.002, tol=1e-8, record_traces=True, trace_addresses=addr)
        self.assertTrue(tds.converged, tds.err_msg)
        t, x = tds.traces["t"], tds.traces["extra"][:, 0]
        k0 = int(np.searchsorted(t, 0.5 - 1e-9))
        post = t > 0.5 + 1e-9
        tp = np.concatenate(([0.0], t[post] - 0.5))
        xp = np.concatenate(([x[k0 + 1]], x[post])) - x[k0]
        dz = np.interp(cor.times_s, tp, xp)
        err_cor = float(np.max(np.abs(cor.values_pu[:, 0] - dz)))
        err_raw = float(np.max(np.abs(raw.values_pu[:, 0] - dz)))
        peak = float(np.max(dz))
        self.assertGreater(peak, 1e-4)
        self.assertLess(err_cor, 0.01 * peak)
        self.assertGreater(err_raw, 10.0 * err_cor)


if __name__ == "__main__":
    unittest.main()
