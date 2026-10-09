"""Sparse and Krylov kernels against the dense path."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import sparse

from pa_dvsa import krylov_response as KR
from pa_dvsa.data_model import ProfileKind, TimeProfile
from pa_dvsa.finite_window import ModeDynamics, propagate_channel, rigorous_envelope, trip_excursion_upper
from pa_dvsa.linearization import AlgebraicFactor, LinearDAEModel
from pa_dvsa.replication import replica_screen as R


def _model(seed: int = 3, n: int = 40, m: int = 30, p: int = 4) -> LinearDAEModel:
    rng = np.random.default_rng(seed)
    ax = sparse.random(n, n, density=0.08, random_state=rng).toarray()
    ax -= np.diag(np.concatenate((rng.uniform(1.0, 50.0, n - 10), rng.uniform(500.0, 2000.0, 10))))
    ay = sparse.random(n, m, density=0.08, random_state=rng).toarray()
    gx = sparse.random(m, n, density=0.08, random_state=rng).toarray()
    gy = sparse.random(m, m, density=0.1, random_state=rng).toarray() + 5.0 * np.eye(m)
    ax[: n - 5, n - 5 :] = 0.0
    gx[:, n - 5 :] = 0.0
    cx = np.zeros((p, n))
    cy = np.zeros((p, m))
    for i in range(p):
        cy[i, i] = 1.0
    return LinearDAEModel.from_blocks(ax=ax, ay=ay, gx=gx, gy=gy, cx=cx, cy=cy)


def _reduced(model: LinearDAEModel) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.linalg.solve(model.gy.toarray(), model.gx.toarray())
    ar = model.ax.toarray() - model.ay.toarray() @ x
    cr = model.cx.toarray() - model.cy.toarray() @ x
    keep = R.output_reachable_states(sparse.csr_matrix(ar), sparse.csc_matrix(cr))
    return ar, cr, keep


def test_certified_lambda_max_bounds_the_spectrum() -> None:
    rng = np.random.default_rng(0)
    a = rng.standard_normal((300, 300))
    s = 0.5 * (a + a.T)
    lam = float(np.linalg.eigvalsh(s).max())
    bound, delta = KR.certified_lambda_max(s)
    assert lam <= bound <= lam + 1e-6
    assert delta > 0.0


def test_schur_products_match_the_dense_formula() -> None:
    model = _model()
    factor = AlgebraicFactor(model.gy)
    py, pc = KR.schur_products(factor, model.gx, model.ay, model.cy, chunk=7)
    x = np.linalg.solve(model.gy.toarray(), model.gx.toarray())
    np.testing.assert_allclose(py, model.ay.toarray() @ x, atol=1e-12)
    np.testing.assert_allclose(pc, model.cy.toarray() @ x, atol=1e-12)


@pytest.mark.parametrize("profile", [TimeProfile(ProfileKind.STEP),
                                     TimeProfile(ProfileKind.RAMP, start_s=0.0, duration_s=0.5),
                                     TimeProfile(ProfileKind.RAMP, start_s=0.1, duration_s=0.37)])
def test_krylov_channel_matches_dense_propagation_and_bounds(profile: TimeProfile) -> None:
    model = _model()
    ar, cr, keep = _reduced(model)
    assert not np.isin(np.arange(model.n_states - 5, model.n_states), keep).any()
    akk, crk = ar[np.ix_(keep, keep)], cr[:, keep]
    rng = np.random.default_rng(1)
    b = rng.standard_normal(keep.size)
    f = rng.standard_normal(crk.shape[0]) * 1e-3
    grid = np.round(np.arange(201) * 0.005, 12)
    dense = ModeDynamics(akk, dt_atomic=2.5e-4)
    kryl = KR.KrylovModeDynamics(akk, model, keep, dt_atomic=2.5e-4)
    assert dense.mu <= kryl.mu <= dense.mu + 1e-6 * max(1.0, abs(dense.mu))
    np.testing.assert_array_equal(kryl.scale, dense.scale)
    ref = propagate_channel(dense, crk, b, f, profile, grid, channel_id="ref")
    new, info = KR.propagate_channel_krylov(kryl, crk, b, f, profile, grid, channel_id="new")
    scale = max(1.0, float(np.abs(ref.phi_right).max()))
    np.testing.assert_allclose(new.phi_right, ref.phi_right, atol=1e-10 * scale)
    np.testing.assert_allclose(new.phi_left, ref.phi_left, atol=1e-10 * scale)
    assert info["krylov_segments"] == (1 if profile.kind == ProfileKind.STEP else (2 if profile.start_s == 0.0 else 3))
    sub_ref, sub_new = ref.refine(7, 4), new.refine(7, 4)
    np.testing.assert_allclose(sub_new.phi_right, sub_ref.phi_right, atol=1e-10 * scale)
    for i in range(crk.shape[0]):
        e_ref = rigorous_envelope(ref, i, grid, tolerance_pu=1e-6)
        e_new = rigorous_envelope(new, i, grid, tolerance_pu=1e-6)
        assert e_new.peak_upper == pytest.approx(e_ref.peak_upper, abs=1e-8 * scale)
        assert e_new.peak_upper >= e_new.sampled_max
        t_ref = trip_excursion_upper(e_ref, horizon_s=1.0, dwell_s=0.05)
        t_new = trip_excursion_upper(e_new, horizon_s=1.0, dwell_s=0.05)
        assert t_new == pytest.approx(t_ref, abs=1e-8 * scale)


def test_descriptor_near_zero_eigenvalues_match_the_dense_check() -> None:
    model = _model(n=60, m=40)
    ar, _, keep = _reduced(model)
    akk = ar[np.ix_(keep, keep)]
    ref, r_ref, nd_ref = R.near_zero_eigenvalues(akk, k=6)
    new, r_new, nd_new = KR.near_zero_eigenvalues_descriptor(model, keep, akk, k=6)
    assert nd_new == nd_ref
    np.testing.assert_allclose(np.sort_complex(new), np.sort_complex(ref), atol=1e-8)
    assert r_new == pytest.approx(r_ref, rel=1e-8)
