"""Diagnostics for comparing data-fidelity gradients of two forward models.

For a forward model ``G`` and data ``b`` the data-fidelity functional is

    J_G(x) = 1/2 ||G(x) - b||^2        (plain Euclidean norm)

and its gradient is ``grad J_G(x) = G'(x)^T (G(x) - b)``. For a linear operator ``A`` this is the
familiar reference ``A^*(A x - b)``. This module provides

* ``gradient_metrics``: how close a gradient ``g`` is to a reference ``g_ref`` (relative error,
  cosine similarity, best scalar, sign agreement of directional derivatives);
* ``directional_fd``: central finite-difference directional derivative of a scalar torch function;
* ``fd_check``: compare an autograd gradient with finite differences over several step sizes;
* ``quadratic_step_decrease``: the decrease of ``1/2 ||A(x - t g) - b||^2`` obtained by an exact
  line search along ``-g`` for a *linear* ``A`` (a practical "is -g a useful descent direction").

None of this is specific to PAT; the PAT-specific wiring lives in
``scripts/evaluation/gradient_comparison.py``.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import torch

__all__ = [
    "gradient_metrics",
    "directional_fd",
    "fd_check",
    "quadratic_step_decrease",
]


def gradient_metrics(
    g: np.ndarray,
    g_ref: np.ndarray,
    directions: np.ndarray | None = None,
    eps: float = 1e-30,
) -> dict[str, float]:
    """Compare a gradient ``g`` with a reference gradient ``g_ref`` (same shape).

    Args:
        g: Gradient under test.
        g_ref: Reference gradient.
        directions: Optional array ``(K, *shape)`` of directions ``v``; if given, the fraction of
            directions for which ``<g, v>`` and ``<g_ref, v>`` have the same sign is reported.
        eps: Guard against division by zero.

    Returns:
        ``rel_l2`` = ||g - g_ref|| / ||g_ref||; ``cos`` = cosine similarity; ``scale`` = best
        scalar ``alpha`` minimising ||g - alpha g_ref||; ``rel_l2_fit`` = the residual after
        that scalar relative to ||g||; ``norm_ratio`` = ||g|| / ||g_ref||; optionally
        ``sign_agree``.
    """
    g = np.asarray(g, np.float64).ravel()
    r = np.asarray(g_ref, np.float64).ravel()
    ng, nr = np.linalg.norm(g), np.linalg.norm(r)
    dot = float(g @ r)
    alpha = dot / (nr**2 + eps)
    out = {
        "rel_l2": float(np.linalg.norm(g - r) / (nr + eps)),
        "cos": dot / (ng * nr + eps),
        "scale": alpha,
        "rel_l2_fit": float(np.linalg.norm(g - alpha * r) / (ng + eps)),
        "norm_ratio": float(ng / (nr + eps)),
    }
    if directions is not None:
        v = np.asarray(directions, np.float64).reshape(len(directions), -1)
        out["sign_agree"] = float(np.mean(np.sign(v @ g) == np.sign(v @ r)))
    return out


def directional_fd(
    fn: Callable[[torch.Tensor], torch.Tensor], x: torch.Tensor, v: torch.Tensor, h: float
) -> float:
    """Central difference ``(f(x + h v) - f(x - h v)) / (2 h)`` of a scalar function."""
    with torch.no_grad():
        return float((fn(x + h * v) - fn(x - h * v)) / (2.0 * h))


def fd_check(
    fn: Callable[[torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    directions: list[torch.Tensor],
    steps: tuple[float, ...] = (1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6),
) -> dict[str, np.ndarray]:
    """Compare the autograd directional derivative ``<grad f, v>`` with central differences.

    Use float64 tensors; in float32 the truncation/round-off trade-off leaves very little
    usable range of ``h``.

    Returns:
        ``ad`` (K,), ``fd`` (K, len(steps)), ``rel_err`` (K, len(steps)) with
        ``|fd - ad| / |ad|``, and ``steps``.
    """
    xr = x.detach().clone().requires_grad_(True)
    (grad,) = torch.autograd.grad(fn(xr), xr)
    ad = np.array([float((grad * v).sum()) for v in directions])
    fd = np.array([[directional_fd(fn, x.detach(), v, h) for h in steps] for v in directions])
    rel = np.abs(fd - ad[:, None]) / (np.abs(ad[:, None]) + 1e-300)
    return {"ad": ad, "fd": fd, "rel_err": rel, "steps": np.array(steps)}


def quadratic_step_decrease(residual: np.ndarray, a_g: np.ndarray) -> tuple[float, float]:
    """Exact line search of ``J(t) = 1/2 ||r - t A g||^2`` for a linear ``A``.

    Args:
        residual: ``r = A x - b``.
        a_g: ``A g`` for the candidate direction ``g`` (the step is ``x - t g``).

    Returns:
        ``(t_star, decrease)`` with ``t_star = <r, A g> / ||A g||^2`` and
        ``decrease = J(0) - J(t_star) = <r, A g>^2 / (2 ||A g||^2)``. A negative ``t_star``
        means ``-g`` is an ascent direction (the optimal move is along ``+g``).
    """
    r = np.asarray(residual, np.float64).ravel()
    ag = np.asarray(a_g, np.float64).ravel()
    denom = float(ag @ ag)
    if denom == 0.0:
        return 0.0, 0.0
    num = float(r @ ag)
    return num / denom, num**2 / (2.0 * denom)
