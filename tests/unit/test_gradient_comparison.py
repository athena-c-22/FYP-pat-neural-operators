"""Mathematical-correctness tests for the gradient-comparison helpers (known operators)."""

from __future__ import annotations

import numpy as np
import torch

from pat_fno.evaluation.gradient_comparison import (
    directional_fd,
    fd_check,
    gradient_metrics,
    quadratic_step_decrease,
)


def _problem(seed=0, n=7, m=11):
    g = torch.Generator().manual_seed(seed)
    a = torch.randn(m, n, generator=g, dtype=torch.float64)
    b = torch.randn(m, dtype=torch.float64, generator=g)
    x = torch.randn(n, dtype=torch.float64, generator=g)
    return a, b, x


def test_linear_gradient_equals_transpose_formula():
    a, b, x = _problem()
    xr = x.clone().requires_grad_(True)
    (g_ad,) = torch.autograd.grad(0.5 * ((a @ xr - b) ** 2).sum(), xr)
    g_ref = a.T @ (a @ x - b)
    assert torch.allclose(g_ad, g_ref, atol=1e-12)
    m = gradient_metrics(g_ad.numpy(), g_ref.numpy())
    assert m["rel_l2"] < 1e-12 and abs(m["cos"] - 1) < 1e-12 and abs(m["scale"] - 1) < 1e-12


def test_gradient_metrics_scale_and_wrong_direction():
    rng = np.random.default_rng(1)
    r = rng.standard_normal(50)
    m = gradient_metrics(3.0 * r, r)
    assert abs(m["scale"] - 3.0) < 1e-12 and abs(m["cos"] - 1) < 1e-12
    assert abs(m["rel_l2"] - 2.0) < 1e-12  # ||3r - r|| / ||r||
    assert m["rel_l2_fit"] < 1e-12  # right shape, wrong scale
    m = gradient_metrics(-r, r)
    assert abs(m["cos"] + 1) < 1e-12 and m["scale"] < 0
    dirs = rng.standard_normal((200, 50))
    assert gradient_metrics(-r, r, dirs)["sign_agree"] == 0.0
    assert gradient_metrics(r, r, dirs)["sign_agree"] == 1.0


def test_fd_check_quadratic_exact_and_nonlinear_has_sweet_spot():
    a, b, x = _problem(2)
    dirs = [v / v.norm() for v in torch.randn(3, 7, dtype=torch.float64)]

    # Quadratic: central differences are exact up to round-off, for any h.
    res = fd_check(lambda z: 0.5 * ((a @ z - b) ** 2).sum(), x, dirs, steps=(1e-1, 1e-3))
    assert res["rel_err"].max() < 1e-8

    # Nonlinear: error is O(h^2) then round-off; the best step must reach ~1e-8 or better,
    # and the largest step must be visibly worse.
    def f(z):
        return torch.tanh(a @ z - b).pow(2).sum()

    res = fd_check(f, x, dirs, steps=(1e-1, 1e-3, 1e-5))
    assert res["rel_err"].min(axis=1).max() < 1e-6
    assert (res["rel_err"][:, 0] > res["rel_err"][:, 1]).all()


def test_fd_detects_wrong_gradient():
    a, b, x = _problem(3)
    fn = lambda z: 0.5 * ((a @ z - b) ** 2).sum()  # noqa: E731
    v = torch.randn(7, dtype=torch.float64)
    wrong = (a.T @ (a @ x - b) * 1.1 * v).sum()  # a 10 % wrong gradient
    fd = directional_fd(fn, x, v, 1e-3)
    assert abs(float(wrong) - fd) / abs(fd) > 0.05


def test_quadratic_step_decrease_matches_brute_force():
    a, b, x = _problem(4)
    r = (a @ x - b).numpy()
    g = (a.T @ (a @ x - b)).numpy()
    t_star, dec = quadratic_step_decrease(r, a.numpy() @ g)
    j = lambda t: 0.5 * np.sum((r - t * (a.numpy() @ g)) ** 2)  # noqa: E731
    ts = np.linspace(-1, 1, 20001) * 2 * abs(t_star)
    assert abs(j(0.0) - j(t_star) - dec) < 1e-9
    assert j(t_star) <= min(j(t) for t in ts[::50]) + 1e-9
    assert t_star > 0  # the true gradient is a descent direction
    t_neg, _ = quadratic_step_decrease(r, -a.numpy() @ g)
    assert t_neg < 0  # the opposite direction is an ascent direction


def test_fno_preserves_float64_and_float32_dtype():
    from pat_fno.models.fno import TinyFNO2d

    torch.manual_seed(0)
    m32 = TinyFNO2d(4, 4, 8, 2).eval()
    x = torch.randn(1, 1, 16, 20)
    y32 = m32(x)
    assert y32.dtype == torch.float32
    m64 = TinyFNO2d(4, 4, 8, 2).eval()
    m64.load_state_dict(m32.state_dict())
    for p in m64.parameters():
        p.data = p.data.to(torch.complex128 if p.is_complex() else torch.float64)
    y64 = m64(x.double())
    assert y64.dtype == torch.float64  # was silently float32 when the buffer was hard-coded cfloat
    assert torch.allclose(y64.float(), y32, atol=1e-4)
