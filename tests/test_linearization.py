"""Tests of the linear model of a mode."""

from __future__ import annotations

from pathlib import Path
import unittest

import numpy as np
from scipy import sparse

from pa_dvsa.andes_adapter import AndesCase, AndesCaseSpec, JacobianBlocks
from pa_dvsa.andes_regularization import regularize_degenerate_blocks
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

    def test_andes_dynamic_case_raw_ieee39_raises_named_linearization_error(self) -> None:
        case = AndesCase.load(
            AndesCaseSpec("ieee39", setup=True, run_pflow=True, init_tds=True),
            project_root=PROJECT_ROOT,
        )
        jac = case.jacobians(update=True)
        cy = selector_matrix([case.bus_voltage_address(30)], jac.n_algebraic)
        cx = sparse.csc_matrix((1, jac.n_states))

        with self.assertRaisesRegex(LinearizationError, "G_y is singular.*IEEEST"):
            model = LinearDAEModel.from_andes_jacobians(jac, cx=cx, cy=cy, mode_id="ieee39")
            reduce_linear_dae(model)

    def test_andes_dynamic_case_regularized_ieee39_folds_and_is_stable(self) -> None:
        case = AndesCase.load(
            AndesCaseSpec("ieee39", setup=False),
            project_root=PROJECT_ROOT,
        )
        changes = regularize_degenerate_blocks(case.require_system())
        self.assertEqual(len(changes), 10)
        case.init_tds()
        jac = case.jacobians(update=True)
        cy = selector_matrix([case.bus_voltage_address(30)], jac.n_algebraic)
        cx = sparse.csc_matrix((1, jac.n_states))

        model = LinearDAEModel.from_andes_jacobians(jac, cx=cx, cy=cy, mode_id="ieee39")
        reduced = reduce_linear_dae(model)

        self.assertLess(model.n_states, jac.n_states)
        self.assertEqual(reduced.ar.shape, (model.n_states, model.n_states))
        self.assertEqual(reduced.cr.shape, (1, model.n_states))
        self.assertTrue(reduced.diagnostics.gy_factorized)
        self.assertTrue(reduced.diagnostics.removed_algebraic)
        self.assertTrue(
            all(" PV " in name or " Slack " in name for name in reduced.diagnostics.removed_algebraic)
        )
        self.assertIsNotNone(reduced.diagnostics.max_real_eigenvalue)
        self.assertLessEqual(reduced.diagnostics.max_real_eigenvalue, 1e-6)
        self.assertLess(reduced.diagnostics.gy_condition_estimate, 1e8)

    def test_generic_second_order_regularization_matches_ieeest_rule(self) -> None:
        from pa_dvsa.andes_regularization import (
            IEEEST_FAST_FILTER,
            _regularize_generic_second_order,
            algebraic_regularity_report,
        )

        case = AndesCase.load(AndesCaseSpec("ieee39", setup=False), project_root=PROJECT_ROOT)
        ss = case.require_system()
        changes = _regularize_generic_second_order(ss)
        self.assertEqual(len(changes), 20)
        self.assertTrue(all(c["action"] == "regularized" for c in changes))
        for name, value in IEEEST_FAST_FILTER.items():
            self.assertTrue(all(float(v) == value for v in getattr(ss.IEEEST, name).v))
        case.init_tds()
        self.assertTrue(algebraic_regularity_report(ss)["ok"])

    def test_exactly_decoupled_variable_is_removed(self) -> None:
        jac = JacobianBlocks(
            fx=sparse.csc_matrix([[-1.0]]),
            fy=sparse.csc_matrix([[1.0, 0.0]]),
            gx=sparse.csc_matrix([[1.0], [0.0]]),
            gy=sparse.csc_matrix([[-2.0, 0.0], [0.0, 1e-8]]),
            tf=np.array([1.0]),
            x_names=("x0",),
            y_names=("y0", "y1"),
        )
        model = LinearDAEModel.from_andes_jacobians(
            jac, cx=[[0.0]], cy=[[1.0, 0.0]], dy=[[1.0], [0.0]]
        )
        self.assertEqual(model.n_algebraic, 1)
        self.assertEqual(model.removed_algebraic, ("y1",))
        reduced = reduce_linear_dae(model)
        self.assertAlmostEqual(_scalar(reduced.ar), -0.5)
        self.assertAlmostEqual(_scalar(reduced.dr), 0.5)
        self.assertAlmostEqual(_scalar(reduced.fd), 0.5)

    def test_coupled_dead_algebraic_row_raises_instead_of_substitution(self) -> None:
        jac = JacobianBlocks(
            fx=sparse.csc_matrix([[-1.0, 0.0], [0.0, -1.0]]),
            fy=sparse.csc_matrix([[1.0, 1.0], [0.0, 0.0]]),
            gx=sparse.csc_matrix([[1.0, 0.0], [1.0, -1.0]]),
            gy=sparse.csc_matrix([[-2.0, 0.0], [0.0, 0.0]]),
            tf=np.array([1.0, 1.0]),
            y_names=("y0", "y_dead"),
        )
        model = LinearDAEModel.from_andes_jacobians(jac, cx=[[0.0, 0.0]], cy=[[1.0, 0.0]])
        with self.assertRaisesRegex(LinearizationError, "y_dead"):
            reduce_linear_dae(model)

    def test_numerically_rank_deficient_gy_raises(self) -> None:
        eps = 1e-14
        model = LinearDAEModel.from_blocks(
            ax=[[-1.0]],
            ay=[[1.0, 1.0]],
            gx=[[1.0], [1.0]],
            gy=[[1.0, 1.0], [1.0, 1.0 + eps]],
            cx=[[0.0]],
            cy=[[1.0, 0.0]],
            y_names=("ya", "yb"),
        )
        with self.assertRaisesRegex(LinearizationError, "rank-deficient|singular"):
            reduce_linear_dae(model)


if __name__ == "__main__":
    unittest.main()
