"""Mathematical tests of the adjoint-consistency metrics using a known linear operator."""

from __future__ import annotations

import numpy as np
import pytest

from pat_fno.evaluation.adjoint_consistency import (
    adjoint_metrics,
    explicit_matrix,
    matrix_adjoint_comparison,
)


def _problem(seed: int = 0, n_in: int = 6, n_out: int = 9):
    rng = np.random.default_rng(seed)
    m = rng.standard_normal((n_out, n_in))
    return m, rng.standard_normal(n_in), rng.standard_normal(n_out)


def test_true_transpose_gives_machine_precision_agreement() -> None:
    m, x, y = _problem()
    r = adjoint_metrics(m @ x, y, x, m.T @ y)
    assert r["abs_diff"] < 1e-12
    assert r["marta"] < 1e-24
    assert r["sym_rel"] < 1e-12
    assert r["ratio"] == pytest.approx(1.0)


def test_scaled_adjoint_exposes_constant_ratio() -> None:
    m, x, y = _problem(1)
    r = adjoint_metrics(m @ x, y, x, 2.0 * (m.T @ y))
    assert r["ratio"] == pytest.approx(0.5)
    # <Ax,y> = L, <x,By> = 2L  ->  sym_rel = |L-2L| / (3|L|) = 1/3 exactly.
    assert r["sym_rel"] == pytest.approx(1.0 / 3.0, rel=1e-6)


def test_wrong_operator_is_detected() -> None:
    m, x, y = _problem(2)
    wrong = np.random.default_rng(5).standard_normal(m.T.shape)
    assert adjoint_metrics(m @ x, y, x, wrong @ y)["sym_rel"] > 1e-2


def test_weights_restore_adjointness() -> None:
    """With <u,v>_Y = wy*sum(uv) and A* := (wx/wy) A^T the weighted identity holds exactly."""
    m, x, y = _problem(3)
    wx, wy = 0.3, 1.7
    r = adjoint_metrics(m @ x, y, x, (wy / wx) * (m.T @ y), wx=wx, wy=wy)
    assert r["abs_diff"] < 1e-12
    # ... and fails if the weights are ignored for the same operator.
    assert adjoint_metrics(m @ x, y, x, (wy / wx) * (m.T @ y))["sym_rel"] > 1e-3


def test_marta_metric_is_squared_discrepancy_over_norm() -> None:
    m, x, y = _problem(4)
    bty = 1.1 * (m.T @ y)
    r = adjoint_metrics(m @ x, y, x, bty)
    assert r["marta"] == pytest.approx(r["abs_diff"] ** 2 / np.sum(y**2))


def test_orthogonal_inner_products_behaviour_near_zero() -> None:
    """sym_rel is dominated by eps when both inner products vanish; cs_rel stays finite."""
    ax, y = np.array([1.0, 0.0]), np.array([0.0, 1.0])  # <Ax,y> = 0
    x, bty = np.array([1.0, 0.0]), np.array([0.0, 1.0])  # <x,By> = 0
    r = adjoint_metrics(ax, y, x, bty, eps=1e-12)
    assert r["sym_rel"] == 0.0 and r["abs_diff"] == 0.0
    assert np.isnan(r["ratio"])  # 0/0 is reported as undefined, not as a number
    r2 = adjoint_metrics(ax, y, x, bty + np.array([1e-9, 0.0]), eps=1e-12)
    assert r2["sym_rel"] > 0.4  # 1e-9 error vs eps=1e-12 denominator: looks "large" -> unstable
    assert r2["cs_rel"] < 1e-8  # Cauchy-Schwarz normalisation shows the error is tiny


def test_explicit_matrix_recovers_operator_and_transpose_comparison() -> None:
    m, _, _ = _problem(6)
    a_mat = explicit_matrix(lambda v: (m @ v.ravel()).reshape(3, 3), (2, 3))
    assert a_mat.shape == (9, 6)
    np.testing.assert_allclose(a_mat, m, atol=1e-6)  # unit vectors stored as float32
    b_mat = explicit_matrix(lambda v: (m.T @ v.ravel()).reshape(2, 3), (3, 3))
    res = matrix_adjoint_comparison(a_mat, b_mat)
    assert res["rel_fro_raw"] < 1e-6 and res["scale_s"] == pytest.approx(1.0, abs=1e-6)
    res2 = matrix_adjoint_comparison(a_mat, 4.0 * b_mat)
    assert res2["scale_s"] == pytest.approx(0.25, rel=1e-5)
    assert res2["rel_fro_after_scale"] < 1e-6 and res2["rel_fro_raw"] > 0.5
