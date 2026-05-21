from __future__ import annotations

from pathlib import Path
import unittest

import numpy as np
from scipy import sparse

from pa_dvsa.andes_adapter import AndesCase, AndesCaseSpec, JacobianBlocks
from pa_dvsa.linearization import (
    LinearDAEModel,
    LinearizationError,
    reduce_linear_dae,
    selector_matrix,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _scalar(matrix: sparse.spmatrix) -> float:
    return float(matrix.toarray()[0, 0])


class ModeWiseLinearizationTests(unittest.TestCase):
    def test_reduced_matrices_match_analytic_toy_dae(self) -> None:
        model = LinearDAEModel.from_blocks(
            ax=[[2.0]],
            ay=[[3.0]],
            gx=[[5.0]],
            gy=[[7.0]],
            bx=[[11.0]],
            by=[[13.0]],
            dx=[[17.0]],
            dy=[[19.0]],
            cx=[[23.0]],
            cy=[[29.0]],
            mode_id="analytic_toy",
        )

        reduced = reduce_linear_dae(model)

        self.assertAlmostEqual(_scalar(reduced.ar), -1.0 / 7.0)
        self.assertAlmostEqual(_scalar(reduced.br), 38.0 / 7.0)
        self.assertAlmostEqual(_scalar(reduced.dr), 62.0 / 7.0)
        self.assertAlmostEqual(_scalar(reduced.cr), 16.0 / 7.0)
        self.assertAlmostEqual(_scalar(reduced.fu), -377.0 / 7.0)
        self.assertAlmostEqual(_scalar(reduced.fd), -551.0 / 7.0)
        self.assertTrue(reduced.diagnostics.gy_factorized)
        self.assertEqual(reduced.diagnostics.stability, "stable")
        self.assertEqual(reduced.diagnostics.n_controls, 1)
        self.assertEqual(reduced.diagnostics.n_disturbances, 1)

    def test_algebraic_only_path_allows_channels_entering_only_through_g(self) -> None:
        model = LinearDAEModel.from_blocks(
            ax=sparse.csc_matrix((0, 0)),
            ay=sparse.csc_matrix((0, 1)),
            gx=sparse.csc_matrix((1, 0)),
            gy=[[2.0]],
            by=[[4.0]],
            dy=[[5.0]],
            cx=sparse.csc_matrix((1, 0)),
            cy=[[3.0]],
            mode_id="algebraic_only",
        )

        reduced = reduce_linear_dae(model)

        self.assertEqual(reduced.ar.shape, (0, 0))
        self.assertEqual(reduced.br.shape, (0, 1))
        self.assertEqual(reduced.dr.shape, (0, 1))
        self.assertEqual(reduced.cr.shape, (1, 0))
        self.assertAlmostEqual(_scalar(reduced.fu), -6.0)
        self.assertAlmostEqual(_scalar(reduced.fd), -7.5)
        self.assertTrue(reduced.algebraic_only)
        self.assertEqual(reduced.diagnostics.stability, "algebraic_only")

    def test_andes_mass_matrix_scaling_and_zero_tf_folding(self) -> None:
        jac = JacobianBlocks(
            fx=sparse.csc_matrix([[4.0, 6.0], [14.0, 16.0]]),
            fy=sparse.csc_matrix([[8.0], [18.0]]),
            gx=sparse.csc_matrix([[24.0, 26.0]]),
            gy=sparse.csc_matrix([[28.0]]),
            tf=np.array([2.0, 0.0]),
        )
        model = LinearDAEModel.from_andes_jacobians(
            jac,
            bx=[[10.0], [20.0]],
            by=[[30.0]],
            dx=[[12.0], [22.0]],
            dy=[[32.0]],
            cx=[[34.0, 36.0]],
            cy=[[38.0]],
            mode_id="andes_mass_toy",
        )

        reduced = reduce_linear_dae(model)

        self.assertEqual(model.n_states, 1)
        self.assertEqual(model.n_algebraic, 2)
        self.assertAlmostEqual(_scalar(reduced.ar), 0.0)
        self.assertAlmostEqual(_scalar(reduced.br), 0.0)
        self.assertAlmostEqual(_scalar(reduced.dr), 0.0)
        self.assertAlmostEqual(_scalar(reduced.cr), 0.0)
        self.assertAlmostEqual(_scalar(reduced.fu), -40.0)
        self.assertAlmostEqual(_scalar(reduced.fd), -42.0)

    def test_singular_gy_raises_explicit_error(self) -> None:
        model = LinearDAEModel.from_blocks(
            ax=[[0.0]],
            ay=[[1.0]],
            gx=[[1.0]],
            gy=[[0.0]],
            cx=[[1.0]],
            cy=[[0.0]],
        )

        with self.assertRaisesRegex(LinearizationError, "G_y is singular"):
            reduce_linear_dae(model)

    def test_channel_column_mismatch_is_rejected(self) -> None:
        with self.assertRaisesRegex(LinearizationError, "same number of columns"):
            LinearDAEModel.from_blocks(
                ax=[[0.0]],
                ay=[[0.0]],
                gx=[[0.0]],
                gy=[[1.0]],
                bx=[[1.0, 2.0]],
                by=[[3.0]],
                cx=[[1.0]],
                cy=[[0.0]],
            )

    def test_unstable_reduced_matrix_emits_warning(self) -> None:
        model = LinearDAEModel.from_blocks(
            ax=[[2.0]],
            ay=[[0.0]],
            gx=[[0.0]],
            gy=[[1.0]],
            cx=[[1.0]],
            cy=[[0.0]],
        )

        reduced = reduce_linear_dae(model)

        self.assertEqual(reduced.diagnostics.stability, "unstable")
        self.assertAlmostEqual(reduced.diagnostics.max_real_eigenvalue or 0.0, 2.0)
        self.assertTrue(
            any("reduced_state_matrix_unstable" in item for item in reduced.diagnostics.warnings)
        )

    def test_andes_static_case_uses_algebraic_only_reduction(self) -> None:
        case = AndesCase.load(
            AndesCaseSpec("ieee14", setup=True, run_pflow=True),
            project_root=PROJECT_ROOT,
        )
        jac = case.jacobians()
        cy = selector_matrix([case.bus_voltage_address(1)], jac.n_algebraic)
        cx = sparse.csc_matrix((1, jac.n_states))
        model = LinearDAEModel.from_andes_jacobians(jac, cx=cx, cy=cy, mode_id="ieee14")

        reduced = reduce_linear_dae(model)

        self.assertTrue(reduced.algebraic_only)
        self.assertEqual(reduced.diagnostics.n_algebraic, 34)
        self.assertEqual(reduced.diagnostics.stability, "algebraic_only")
        self.assertEqual(reduced.fu.shape, (1, 0))
        self.assertEqual(reduced.fd.shape, (1, 0))

    def test_andes_dynamic_case_folds_zero_tf_constraints_and_reduces(self) -> None:
        case = AndesCase.load(
            AndesCaseSpec("ieee39", setup=True, run_pflow=True, init_tds=True),
            project_root=PROJECT_ROOT,
        )
        jac = case.jacobians()
        cy = selector_matrix([case.bus_voltage_address(30)], jac.n_algebraic)
        cx = sparse.csc_matrix((1, jac.n_states))

        model = LinearDAEModel.from_andes_jacobians(jac, cx=cx, cy=cy, mode_id="ieee39")
        reduced = reduce_linear_dae(model)

        self.assertLess(model.n_states, jac.n_states)
        self.assertGreater(model.n_algebraic, jac.n_algebraic)
        self.assertEqual(reduced.ar.shape, (model.n_states, model.n_states))
        self.assertEqual(reduced.cr.shape, (1, model.n_states))
        self.assertTrue(reduced.diagnostics.gy_factorized)


if __name__ == "__main__":
    unittest.main()
