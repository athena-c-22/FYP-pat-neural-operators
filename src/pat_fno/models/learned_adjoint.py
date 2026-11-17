"""Learned adjoint ``G_phi: y -> x`` and an adjoint-consistency (dot-product) training loss.

``G_phi`` maps sensor data ``y`` (``Ny x Nt``, default 64x91) to an image (``Nx x Ny``,
64x64). It reuses :class:`TinyFNO2d`: the data are bilinearly resized to the image grid (the
inverse of the resize used by the forward surrogate), z-score normalised, passed through the
FNO and de-normalised. Normalisation statistics are buffers so they travel with checkpoints.

Dot-product loss. For a forward operator ``A`` and training images ``x_i`` with data
``y_i = A x_i`` the adjoint identity ``<A x_i, y_j> = <x_i, A* y_j>`` gives, for every pair,

    <y_i, y_j>  =  <x_i, G_phi(y_j)>.

``adjoint_dot_loss`` penalises the squared Cauchy-Schwarz-normalised discrepancy (the stable
``cs_rel`` of ``pat_fno.evaluation.adjoint_consistency``) averaged over all ``i, j`` pairs. It
needs no extra k-Wave calls beyond the pairs ``(x_i, y_i)`` already generated, and it
constrains ``G_phi`` towards the transpose of ``A`` itself rather than towards an approximate
reference adjoint.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from pat_fno.models.fno import TinyFNO2d

__all__ = ["LearnedAdjoint", "adjoint_dot_loss"]


class LearnedAdjoint(nn.Module):
    """FNO-based learned adjoint with built-in resizing and normalisation.

    Args:
        modes, width, layers: :class:`TinyFNO2d` hyper-parameters.
        image_size: Output image shape ``(Nx, Ny)``.
    """

    def __init__(
        self,
        modes: int = 8,
        width: int = 16,
        layers: int = 3,
        image_size: tuple[int, int] = (64, 64),
    ) -> None:
        """Build the FNO and the (identity-initialised) normalisation buffers."""
        super().__init__()
        self.image_size = tuple(image_size)
        self.fno = TinyFNO2d(modes, modes, width, layers)
        for name in ("y_mean", "y_std", "x_mean", "x_std"):
            self.register_buffer(name, torch.tensor(0.0 if name.endswith("mean") else 1.0))

    def set_stats(self, y_mean: float, y_std: float, x_mean: float, x_std: float) -> None:
        """Set the data/image z-score statistics (computed on the training split only)."""
        for name, value in (
            ("y_mean", y_mean),
            ("y_std", y_std),
            ("x_mean", x_mean),
            ("x_std", x_std),
        ):
            getattr(self, name).fill_(float(value))

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        """Map data ``(B, Ny, Nt)`` to images ``(B, Nx, Ny)`` in physical units."""
        if y.ndim != 3:
            raise ValueError("LearnedAdjoint expects data of shape (batch, Ny, Nt).")
        z = F.interpolate(y[:, None], size=self.image_size, mode="bilinear", align_corners=False)
        z = (z - self.y_mean) / self.y_std
        return self.fno(z)[:, 0] * self.x_std + self.x_mean


def adjoint_dot_loss(
    x: torch.Tensor, y: torch.Tensor, bty: torch.Tensor, eps: float = 1e-12
) -> torch.Tensor:
    """Mean squared normalised dot-product discrepancy over all ``(i, j)`` pairs.

    Args:
        x: Images ``(B, ...)``.
        y: Data ``(B, ...)`` with ``y_i = A x_i`` (``A x_i`` itself, so the identity's left
            side is ``<y_i, y_j>``).
        bty: Candidate adjoint outputs ``B y_j`` with the shape of ``x``.
        eps: Guard for the denominator.

    Returns:
        ``mean_ij ( (<y_i, y_j> - <x_i, B y_j>) / (||y_i|| ||y_j|| + eps) )^2``. It is zero iff
        the identity holds for all pairs; it is invariant to scaling ``x``, ``y`` and ``bty``
        together (``y = A x``, ``B`` linear), but not to rescaling ``bty`` alone: a pure scale
        error of the adjoint is penalised.
    """
    xf, yf, bf = x.flatten(1), y.flatten(1), bty.flatten(1)
    lhs = yf @ yf.T
    rhs = xf @ bf.T
    ny = yf.norm(dim=1)
    return (((lhs - rhs) / (ny[:, None] * ny[None, :] + eps)) ** 2).mean()
