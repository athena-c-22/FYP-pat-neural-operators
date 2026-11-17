"""Adjoint-consistency diagnostics for linear operators ``A: X -> Y``.

A linear operator ``A`` and a candidate ``B`` are *adjoint* if, for all ``x`` in X and ``y`` in Y,

    <A x, y>_Y = <x, B y>_X .

Here the inner products may carry quadrature weights, ``<u, v>_Y = wy * sum(u * v)`` and
``<x, z>_X = wx * sum(x * z)``. Only the ratio ``wy / wx`` matters for the identity, and it is
exposed explicitly because whether the discretisation needs such weights is part of what is being
investigated, not something to assume.

Metrics (``adjoint_metrics``), with ``d = <A x, y> - <x, B y>``:

* ``abs_diff``   = |d|. Scale dependent (changes if A is rescaled).
* ``marta``      = |d|^2 / ||y||^2, the expression from the supervisor meeting notes. The
  notes do not fix the norm or the weights, so the plain Euclidean ``||y||^2`` is used and
  flagged as an interpretation. Squares the discrepancy, so it is tiny for small errors and
  is not dimensionless.
* ``sym_rel``    = |d| / (|<Ax,y>| + |<x,By>| + eps). Bounded in [0, 1] when the two inner
  products have the same sign (0 is perfect agreement, 1 is "one of them is zero"); with
  opposite signs it can reach 1 as well. Near zero denominators it is dominated by ``eps``.
* ``cs_rel``     = |d| / (||A x|| ||y|| + eps). Normalised by the Cauchy-Schwarz bound, so it
  is stable when both inner products are small by cancellation. Not part of the brief; added
  because ``sym_rel`` is unstable for nearly orthogonal pairs.
* ``ratio``      = <A x, y> / <x, B y>. Exposes a pure scale error (ratio constant across samples).
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

__all__ = [
    "adjoint_metrics",
    "explicit_matrix",
    "matrix_adjoint_comparison",
    "inner",
]


def inner(u: np.ndarray, v: np.ndarray, weight: float = 1.0) -> float:
    """Weighted real inner product ``weight * sum(u * v)`` in float64."""
    return float(weight * np.vdot(np.asarray(u, np.float64), np.asarray(v, np.float64)).real)


def adjoint_metrics(
    ax: np.ndarray,
    y: np.ndarray,
    x: np.ndarray,
    bty: np.ndarray,
    *,
    wx: float = 1.0,
    wy: float = 1.0,
    eps: float = 1e-12,
) -> dict[str, float]:
    """Compare ``<Ax, y>_Y`` with ``<x, B y>_X`` for one pair ``(x, y)``.

    Args:
        ax: ``A x`` (data space).
        y: Data-space vector.
        x: Image-space vector.
        bty: ``B y`` (image space), the candidate adjoint applied to ``y``.
        wx: Quadrature weight of the image-space inner product.
        wy: Quadrature weight of the data-space inner product.
        eps: Regulariser for the symmetric / Cauchy-Schwarz denominators.

    Returns:
        Dictionary with both inner products and the metrics listed in the module docstring.
    """
    lhs = inner(ax, y, wy)
    rhs = inner(x, bty, wx)
    diff = lhs - rhs
    ny2 = wy * float(np.sum(np.asarray(y, np.float64) ** 2))
    nax = float(np.sqrt(wy * np.sum(np.asarray(ax, np.float64) ** 2)))
    return {
        "lhs": lhs,
        "rhs": rhs,
        "abs_diff": abs(diff),
        "marta": diff**2 / max(ny2, eps),
        "sym_rel": abs(diff) / (abs(lhs) + abs(rhs) + eps),
        "cs_rel": abs(diff) / (nax * np.sqrt(max(ny2, 0.0)) + eps),
        "ratio": lhs / rhs if abs(rhs) > eps else float("nan"),
    }


def explicit_matrix(
    op: Callable[[np.ndarray], np.ndarray],
    in_shape: tuple[int, ...],
    *,
    columns: Callable[[list[np.ndarray]], list[np.ndarray]] | None = None,
) -> np.ndarray:
    """Assemble the matrix of a linear operator by applying it to unit vectors.

    Args:
        op: Linear map on arrays of shape ``in_shape``.
        in_shape: Shape of the operator's input.
        columns: Optional batch evaluator (e.g. a process pool map) taking a list of
            unit-vector inputs and returning the list of outputs; defaults to a serial
            loop over ``op``.

    Returns:
        Matrix of shape ``(prod(out_shape), prod(in_shape))`` (C-order flattening).
    """
    n = int(np.prod(in_shape))
    units = []
    for j in range(n):
        e = np.zeros(n, dtype=np.float32)
        e[j] = 1.0
        units.append(e.reshape(in_shape))
    outs = columns(units) if columns is not None else [op(u) for u in units]
    return np.stack([np.asarray(o, np.float64).ravel() for o in outs], axis=1)


def matrix_adjoint_comparison(a_mat: np.ndarray, b_mat: np.ndarray) -> dict[str, float]:
    """Compare the transpose of ``a_mat`` with a candidate adjoint matrix ``b_mat``.

    Reports the relative Frobenius error with no rescaling, the best scalar ``s`` such that
    ``A^T ~ s B`` (least squares) and the relative error after that rescaling. A constant
    ``s != 1`` with a small post-fit error means "adjoint up to a scale" (a weighting/units
    issue); a large post-fit error means the two operators genuinely differ in shape.
    """
    at = a_mat.T
    s = float(np.sum(at * b_mat) / np.sum(b_mat * b_mat))
    na = np.linalg.norm(at)
    return {
        "rel_fro_raw": float(np.linalg.norm(at - b_mat) / na),
        "scale_s": s,
        "rel_fro_after_scale": float(np.linalg.norm(at - s * b_mat) / na),
        "norm_At": float(na),
        "norm_B": float(np.linalg.norm(b_mat)),
        "cos_sim": float(np.sum(at * b_mat) / (na * np.linalg.norm(b_mat))),
    }
