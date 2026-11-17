"""Tests for the learned-adjoint wrapper and the dot-product loss (known linear operator)."""

from __future__ import annotations

import torch

from pat_fno.models.learned_adjoint import LearnedAdjoint, adjoint_dot_loss


def _linear_problem(b=6, nx=12, ny=15, seed=0):
    g = torch.Generator().manual_seed(seed)
    a = torch.randn(ny, nx, generator=g, dtype=torch.float64)  # A: R^nx -> R^ny
    x = torch.randn(b, nx, generator=g, dtype=torch.float64)
    return a, x, x @ a.T  # y_i = A x_i


def test_dot_loss_zero_for_exact_transpose():
    a, x, y = _linear_problem()
    assert adjoint_dot_loss(x, y, y @ a).item() < 1e-28  # B y = A^T y


def test_dot_loss_detects_scaled_and_wrong_adjoint():
    a, x, y = _linear_problem()
    loss_scaled = adjoint_dot_loss(x, y, 1.1 * (y @ a)).item()
    assert loss_scaled > 1e-6
    g = torch.Generator().manual_seed(5)
    wrong = y @ torch.randn_like(a, generator=g)
    assert adjoint_dot_loss(x, y, wrong).item() > loss_scaled


def test_dot_loss_joint_scale_invariant_and_differentiable():
    a, x, y = _linear_problem()
    bt = (1.1 * (y @ a)).requires_grad_(True)
    l1 = adjoint_dot_loss(x, y, bt)
    l2 = adjoint_dot_loss(7.0 * x, 7.0 * y, 7.0 * bt)  # y = A x and B linear
    assert abs(l1.item() - l2.item()) < 1e-9 * max(1.0, l1.item())
    l1.backward()
    assert bt.grad is not None and bt.grad.abs().sum() > 0


def test_learned_adjoint_shapes_stats_and_gradients():
    torch.manual_seed(0)
    m = LearnedAdjoint(modes=4, width=8, layers=2)
    y = torch.randn(3, 64, 91)
    out = m(y)
    assert out.shape == (3, 64, 64)
    m.set_stats(0.5, 2.0, -1.0, 3.0)
    base = LearnedAdjoint(modes=4, width=8, layers=2)
    base.load_state_dict({k: v for k, v in m.state_dict().items()}, strict=True)
    # statistics are buffers: they must be saved in the state dict
    assert {"y_mean", "y_std", "x_mean", "x_std"} <= set(m.state_dict())
    assert torch.allclose(base.x_std, torch.tensor(3.0))
    m(y).pow(2).mean().backward()
    assert all(p.grad is not None for p in m.parameters())
