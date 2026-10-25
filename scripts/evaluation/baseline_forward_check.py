"""Evaluate the bundled FNO-only checkpoint on the bundled k-Wave samples.

Reuses the loaders and inference path of the reproducibility check and writes a
per-sample metrics table and a reference/prediction/error figure. Metrics are
computed in physical units (same scale as the k-Wave target); normalised-space
MSE, the training-loss scale, is reported alongside for reference.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from pat_fno.config import load_experiment_config
from pat_fno.data.mnist import build_setting, centered_corr, rel_l2
from scripts.smoke.run_reproducibility_check import (
    _fno_predictions,
    _load_samples,
    _repository_path,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="smoke/reproducibility.yaml")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/baseline_evaluation"))
    args = parser.parse_args()
    torch.manual_seed(20260728)

    config = load_experiment_config(args.config)
    samples = _load_samples(_repository_path(config["sample_data"]), config["sample_count"])
    checkpoint_path = _repository_path(config["checkpoint"])
    norm = torch.load(checkpoint_path, map_location="cpu", weights_only=False)["normalization"]
    setting = build_setting(config["condition"])

    p0, target = samples["initial_pressure"], samples["kwave_target"]
    fno = _fno_predictions(p0, checkpoint_path)
    fourier = samples["fourier_prediction"]

    rows = []
    for i in range(len(p0)):
        error = fno[i] - target[i]
        nerr = error / norm["data_std"]
        rows.append(
            {
                "sample": i,
                "source_index": int(samples["source_index"][i]),
                "label": int(samples["label"][i]),
                "fno_rel_l2": rel_l2(fno[i], target[i]),
                "fno_mse": float(np.mean(error**2)),
                "fno_mae": float(np.mean(np.abs(error))),
                "fno_normalised_mse": float(np.mean(nerr**2)),
                "fno_centered_corr": centered_corr(fno[i], target[i]),
                "fourier_rel_l2": rel_l2(fourier[i], target[i]),
                "fourier_mse": float(np.mean((fourier[i] - target[i]) ** 2)),
                "fourier_mae": float(np.mean(np.abs(fourier[i] - target[i]))),
            }
        )
    keys = [k for k in rows[0] if k not in ("sample", "source_index", "label")]
    mean_row = {"sample": "mean", "source_index": "", "label": ""}
    mean_row.update({k: float(np.mean([r[k] for r in rows])) for k in keys})
    # Global (pooled) relative L2 over the whole subset, for comparison with the per-sample mean.
    mean_row["fno_rel_l2_pooled"] = float(np.linalg.norm(fno - target) / np.linalg.norm(target))

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0]) + ["fno_rel_l2_pooled"]
    with (out / "metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows + [mean_row])

    # Figure: p0 | reference | prediction | |error|, one row per sample.
    t_us = setting.t_array[-1] * 1e6
    y_mm = setting.Ny * setting.dy * 1e3
    extent = (0, t_us, y_mm, 0)
    vmax = float(max(np.abs(target).max(), np.abs(fno).max()))
    emax = float(np.abs(fno - target).max())
    n = len(p0)
    fig, axes = plt.subplots(n, 4, figsize=(15, 3.1 * n), constrained_layout=True)
    for i in range(n):
        a = axes[i]
        im0 = a[0].imshow(
            p0[i],
            cmap="gray",
            extent=(0, setting.Nx * setting.dx * 1e3, setting.Ny * setting.dy * 1e3, 0),
        )
        a[0].set(title=f"p0 (MNIST '{samples['label'][i]}', idx {samples['source_index'][i]})")
        a[0].set_xlabel("x [mm]")
        a[0].set_ylabel("y [mm]")
        kw = dict(cmap="RdBu_r", vmin=-vmax, vmax=vmax, extent=extent, aspect="auto")
        imr = a[1].imshow(target[i], **kw)
        imp = a[2].imshow(fno[i], **kw)
        ime = a[3].imshow(np.abs(fno[i] - target[i]), cmap="magma", vmin=0, vmax=emax, extent=extent, aspect="auto")
        a[1].set_title("Reference (k-Wave)")
        a[2].set_title(f"FNO-only prediction (rel L2 {rows[i]['fno_rel_l2']:.3f})")
        a[3].set_title("|prediction - reference|")
        for ax in a[1:]:
            ax.set_xlabel("time [µs]")
            ax.set_ylabel("sensor position y [mm] (sensor index 0-63)")
        fig.colorbar(im0, ax=a[0], label="p0 [a.u.]")
        fig.colorbar(imr, ax=a[1:3], label="pressure [a.u.] (shared scale)")
        fig.colorbar(ime, ax=a[3], label="|error| [a.u.]")
    fig.suptitle(
        "FNO-only, periodic 89° — sensor measurements are a (64 sensors × 91 time steps) array; "
        "physical scale, p0 peak = 1",
        fontsize=11,
    )
    fig.savefig(out / "sensor_comparison.png", dpi=150)
    for r in rows + [mean_row]:
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})


if __name__ == "__main__":
    main()
