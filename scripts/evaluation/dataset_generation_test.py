"""Minimal k-Wave data-generation smoke test on bundled initial-pressure images.

Calls the same ``generate_sample`` routine as ``scripts/data/generate_mnist_dataset.py``
on a few bundled p0 images (no MNIST download, no dataset written) and compares the
fresh k-Wave output with the stored reference measurements.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pat_fno.data.mnist import build_setting, generate_sample, rel_l2

SAMPLES = Path("examples/sample_data/periodic_large_test_samples.npz")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=2)
    parser.add_argument("--output", type=Path, default=Path("outputs/baseline_evaluation"))
    args = parser.parse_args()
    data = np.load(SAMPLES)
    setting = build_setting("periodic_theta89")
    n = args.count
    fig, axes = plt.subplots(n, 3, figsize=(13, 3.2 * n), constrained_layout=True, squeeze=False)
    for i in range(n):
        p0 = data["initial_pressure"][i]
        start = time.perf_counter()
        raw, scaled, kwave = generate_sample(p0, setting)
        seconds = time.perf_counter() - start
        print(
            f"sample {i}: shapes p0={p0.shape} kwave={kwave.shape} "
            f"finite={np.isfinite(kwave).all()} range=[{kwave.min():.3f},{kwave.max():.3f}] "
            f"time={seconds:.2f}s\n"
            f"  fresh k-Wave vs stored target: max|diff|="
            f"{np.abs(kwave - data['kwave_target'][i]).max():.3e}, "
            f"rel L2={rel_l2(kwave, data['kwave_target'][i]):.3e}\n"
            f"  fresh c*Fourier vs stored: max|diff|="
            f"{np.abs(scaled - data['fourier_prediction'][i]).max():.3e}"
        )
        extent = (0, setting.t_array[-1] * 1e6, setting.Ny * setting.dy * 1e3, 0)
        v = float(np.abs(kwave).max())
        im = axes[i, 0].imshow(p0, cmap="gray", extent=(0, 6.4, 6.4, 0))
        axes[i, 0].set(title="Initial pressure p0", xlabel="x [mm]", ylabel="y [mm]")
        fig.colorbar(im, ax=axes[i, 0], label="p0 [a.u.]")
        for ax, arr, title in (
            (axes[i, 1], kwave, f"Generated k-Wave data ({seconds:.1f} s, CPU)"),
            (axes[i, 2], scaled, "c x Fourier forward"),
        ):
            im = ax.imshow(arr, cmap="RdBu_r", vmin=-v, vmax=v, extent=extent, aspect="auto")
            ax.set(title=title, xlabel="time [µs]", ylabel="sensor position y [mm]")
        fig.colorbar(im, ax=axes[i, 1:], label="pressure [a.u.]")
    fig.savefig(args.output / "dataset_generation_test.png", dpi=150)


if __name__ == "__main__":
    main()
