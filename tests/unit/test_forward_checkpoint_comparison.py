import numpy as np
import torch

from pat_fno.models.fno import TinyFNO2d
from scripts.evaluation.forward_checkpoint_comparison import positive_step, surrogate


def test_ascent_is_rejected():
    alpha, after, dot = positive_step(np.array([2.0]), np.array([-1.0]))
    assert alpha == 0
    assert after == 2
    assert dot < 0


def test_positive_search_and_small_step():
    r, ag = np.array([2.0, 1.0]), np.array([1.0, 0.0])
    alpha, after, dot = positive_step(r, ag)
    assert alpha == 2
    assert after == 0.5
    assert dot == 2
    alpha, after, _ = positive_step(r, ag, 0.1)
    assert alpha == 0.2
    assert np.isclose(after, 2.12)


def test_zero_direction():
    assert positive_step(np.ones(2), np.zeros(2)) == (0.0, 1.0, 0.0)


def test_batched_surrogate_preserves_single_image_and_gradient():
    torch.manual_seed(20260728)
    model = TinyFNO2d(2, 2, 2, 1)
    checkpoint = {
        "arguments": {
            "modes": 2,
            "width": 2,
            "layers": 1,
            "scenario": "fno_only",
            "condition": "periodic_theta89",
            "dataset": "mnist_large_v1",
        },
        "model": model.state_dict(),
        "normalization": {"p0_mean": 0.1, "p0_std": 0.3, "data_mean": 0.2, "data_std": 0.25},
    }
    predict = surrogate(checkpoint)
    images = torch.randn(2, 64, 64, dtype=torch.float64, requires_grad=True)
    batch = predict(images)
    singles = torch.stack([predict(x) for x in images])
    torch.testing.assert_close(batch, singles, rtol=1e-12, atol=1e-12)
    assert batch.shape == (2, 64, 91)
    batch_gradient = torch.autograd.grad(batch.square().sum(), images)[0]
    single_gradient = torch.autograd.grad(singles.square().sum(), images)[0]
    torch.testing.assert_close(batch_gradient, single_gradient, rtol=1e-12, atol=1e-12)
