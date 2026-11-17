"""Stage 1: numerically validate the c-scaling between Fourier and k-Wave operators.

Marta Betcke's upstream MATLAB commit 5eb43c6 states (for the example script):

    data = c * FourierForward(p0)          # Fourier forward is multiplied by c
    adj  = FourierAdjoint(data / c)        # Fourier adjoint / inverse undo it
    inv  = FourierInverse(data / c)

and reports agreement with the k-Wave forward and adjoint. This script tests the
claim in the Python port, per acquisition condition and test image, using
*best-fit scalars* so that a pure scaling error is visible as a factor, separately
from shape (discretisation / limited-angle / boundary) error.

For two arrays U (reference) and V (candidate) we report
    alpha   = <U, V> / <V, V>              (least-squares scale: U ~ alpha V)
    rel_raw = ||U - V|| / ||U||            (error with NO rescaling)
    rel_fit = ||U - alpha V|| / ||U||      (error after the optimal rescaling)
alpha ~ 1 and rel_raw ~ rel_fit  => scaling is consistent.

Experiments (all with the same ``setting`` object used for dataset generation):
  E1  forward : U = kwave_forward(p0),          V = numpy_forward(p0)  [raw, no c]
  E2  adjoint : U = kwave_adjoint(K),           V = numpy_adjoint(K/c), and V' = numpy_adjoint(K)
                with K = kwave_forward(p0)  (same data fed to both adjoints)
  E3  inverse : p0 vs numpy_inverse(K/c) and numpy_inverse(K)
  E4  python  : torch ``fpat_forward_2d`` vs NumPy ``numpy_forward_2d`` (same convention)

Nothing is written outside ``--output-dir``. Seeded; no network access.
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from pat_fno.config.mnist_pat import CONDITIONS, SOUND_SPEED
from pat_fno.data.mnist import build_setting
from pat_fno.operators.fourier import fpat_forward_2d
from pat_fno.operators.fourier.numpy_reference import (
    numpy_adjoint_2d,
    numpy_forward_2d,
    numpy_inverse_2d,
)
from pat_fno.operators.kwave import kwave_adjoint_2d, kwave_forward_2d

SEED = 20260728
ROOT = Path(__file__).resolve().parents[2]
SAMPLES = ROOT / "examples" / "sample_data" / "periodic_large_test_samples.npz"


def phantoms(n: int, rng: np.random.Generator) -> dict[str, np.ndarray]:
    """Synthetic p0 images (peak 1) on the 64x64 grid, plus the bundled MNIST ones."""
    xx, yy = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    out: dict[str, np.ndarray] = {}
    out["gauss_blob"] = np.exp(-(((xx - 24) ** 2 + (yy - 30) ** 2) / (2 * 4.0**2)))
    out["disc"] = (((xx - 32) ** 2 + (yy - 32) ** 2) < 10**2).astype(float)
    out["two_gauss"] = np.exp(-(((xx - 20) ** 2 + (yy - 20) ** 2) / 18)) + 0.6 * np.exp(
        -(((xx - 42) ** 2 + (yy - 38) ** 2) / 32)
    )
    noise = np.fft.fft2(rng.standard_normal((n, n)))
    kk = np.hypot(*np.meshgrid(np.fft.fftfreq(n), np.fft.fftfreq(n), indexing="ij"))
    smooth = np.real(np.fft.ifft2(noise * np.exp(-((kk / 0.08) ** 2))))
    smooth = (smooth - smooth.min()) / (smooth.max() - smooth.min())
    out["smooth_random"] = smooth
    # y-independent bump: a 1-D wave problem with a closed-form sensor signal (see E0).
    out["plane_bump"] = np.exp(-((xx - 18) ** 2) / (2 * 6.0**2))
    if SAMPLES.exists():
        data = np.load(SAMPLES)
        for i, img in enumerate(data["initial_pressure"]):
            out[f"mnist_{int(data['source_index'][i])}"] = img.astype(float)
    return {k: v.astype(np.float32) / max(float(v.max()), 1e-12) for k, v in out.items()}


def compare(ref: np.ndarray, cand: np.ndarray) -> dict[str, float]:
    ref, cand = np.asarray(ref, float), np.asarray(cand, float)
    alpha = float(np.vdot(ref, cand) / np.vdot(cand, cand))  # U ~ alpha V  with <U,V>/<V,V>
    nref = np.linalg.norm(ref)
    return {
        "alpha": alpha,
        "rel_raw": float(np.linalg.norm(ref - cand) / nref),
        "rel_fit": float(np.linalg.norm(ref - alpha * cand) / nref),
        "norm_ref": float(nref),
        "norm_cand": float(np.linalg.norm(cand)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/operator_scaling"))
    parser.add_argument("--conditions", nargs="+", default=list(CONDITIONS))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    images = phantoms(64, rng)
    c = SOUND_SPEED

    rows: list[dict] = []
    keep: dict = {}
    t0 = time.time()
    for cond in args.conditions:
        setting = build_setting(cond)
        # build_setting only sets the forward method; MATLAB example uses 'cubic' for all three.
        setting.computation.interpolationMethodA = "cubic"
        setting.computation.interpolationMethodI = "cubic"
        for name, p0 in images.items():
            p0d = p0.astype(np.float64)
            K = kwave_forward_2d(p0, setting)  # (Ny, Nt) reference
            F = numpy_forward_2d(p0, setting)  # raw Fourier, no c
            Ft = fpat_forward_2d(
                torch.from_numpy(p0d),
                setting.computation.theta_max,
                c,
                setting.Nt,
                (setting.dx, setting.dy),
                setting.dt,
            ).numpy()
            Akw = kwave_adjoint_2d(K, setting)
            Aff_scaled = numpy_adjoint_2d(K / c, setting)
            Aff_raw = numpy_adjoint_2d(K, setting)
            Inv_scaled = numpy_inverse_2d(K / c, setting)
            Inv_raw = numpy_inverse_2d(K, setting)
            exps = {
                "E1_fwd_kw_vs_Fraw": compare(K, F),
                "E1_fwd_kw_vs_cF": compare(K, c * F),
                "E2_adj_kw_vs_Fadj(K/c)": compare(Akw, Aff_scaled),
                "E2_adj_kw_vs_Fadj(K)": compare(Akw, Aff_raw),
                "E3_inv_p0_vs_Finv(K/c)": compare(p0d, Inv_scaled),
                "E3_inv_p0_vs_Finv(K)": compare(p0d, Inv_raw),
                "E4_torch_vs_numpy_fwd": compare(F, Ft),
            }
            if name == "plane_bump":
                # E0: analytic truth. Mirror symmetry + d'Alembert => sensor (row Nx of the mirrored
                # grid = p0 row 0) sees 0.5*[p0m(+ct)+p0m(-ct)] = (p0[n] + p0[n-1])/2 at t = n dt,
                # because c dt = dx (CFL=1). Independent of c, so a pure test of absolute scale.
                nn = 48  # stay clear of boundary / wrap-around arrivals
                col = p0d[:, 0]
                truth = 0.5 * (col[:nn] + np.concatenate(([col[0]], col[: nn - 1])))
                truth = np.tile(truth, (setting.Ny, 1))
                exps["E0_analytic_vs_kwave"] = compare(truth, K[:, :nn])
                exps["E0_analytic_vs_cF"] = compare(truth, c * F[:, :nn])
                exps["E0_analytic_vs_Fraw"] = compare(truth, F[:, :nn])
            for exp, m in exps.items():
                rows.append({"condition": cond, "image": name, "experiment": exp, **m})
            keep[(cond, name)] = (p0, K, F, Akw, Aff_scaled)
            print(
                f"[{time.time() - t0:6.1f}s] {cond:22s} {name:18s} "
                f"alpha(kw/Fraw)={exps['E1_fwd_kw_vs_Fraw']['alpha']:9.2f} "
                f"alpha(adj kw/F(K/c))={exps['E2_adj_kw_vs_Fadj(K/c)']['alpha']:.3f}",
                flush=True,
            )

    # E5: bundled periodic-89 MNIST samples (stored k-Wave target and stored c*Fourier prediction;
    # no new k-Wave runs). Does an amplitude factor explain the ~0.94 "Fourier baseline" error?
    if SAMPLES.exists():
        d = np.load(SAMPLES)
        for i in range(len(d["kwave_target"])):
            name = f"mnist_{int(d['source_index'][i])}"
            Kb, Fb = d["kwave_target"][i], d["fourier_prediction"][i]
            for exp, cand in (
                ("E5_bundled_kw_vs_cF", Fb),
                ("E5_bundled_kw_vs_cF/sqrt2", Fb / np.sqrt(2)),
            ):
                rows.append(
                    {
                        "condition": "periodic_theta89",
                        "image": name,
                        "experiment": exp,
                        **compare(Kb, cand),
                    }
                )

    fields = [
        "condition",
        "image",
        "experiment",
        "alpha",
        "rel_raw",
        "rel_fit",
        "norm_ref",
        "norm_cand",
    ]
    with (args.output_dir / "results.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    # Summary (median/min/max of alpha and rel errors) per condition x experiment.
    summary = []
    for cond in args.conditions:
        for exp in dict.fromkeys(r["experiment"] for r in rows):
            sel = [r for r in rows if r["condition"] == cond and r["experiment"] == exp]
            if not sel:
                continue
            s = {"condition": cond, "experiment": exp, "n": len(sel)}
            for key in ("alpha", "rel_raw", "rel_fit"):
                v = np.array([r[key] for r in sel])
                s[f"{key}_median"], s[f"{key}_min"], s[f"{key}_max"] = (
                    float(np.median(v)),
                    float(v.min()),
                    float(v.max()),
                )
            summary.append(s)
    meta = {
        "seed": SEED,
        "sound_speed": c,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "images": list(images),
        "runtime_s": time.time() - t0,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps({"meta": meta, "summary": summary}, indent=2)
    )
    for s in summary:
        print(
            f"{s['condition']:22s} {s['experiment']:26s} alpha med {s['alpha_median']:10.4g} "
            f"[{s['alpha_min']:.4g},{s['alpha_max']:.4g}]  rel_raw med {s['rel_raw_median']:.3g}  "
            f"rel_fit med {s['rel_fit_median']:.3g}"
        )

    # Figure: one smooth phantom per condition: K vs cF (central-sensor trace) and adjoint images.
    fig, axes = plt.subplots(
        len(args.conditions),
        3,
        figsize=(14, 3.6 * len(args.conditions)),
        constrained_layout=True,
        squeeze=False,
    )
    for r, cond in enumerate(args.conditions):
        p0, K, F, Akw, Aff = keep[(cond, "two_gauss")]
        j = K.shape[0] // 2
        axes[r, 0].plot(K[j], label="k-Wave forward")
        axes[r, 0].plot(c * F[j], "--", label="c x Fourier forward")
        axes[r, 0].plot(F[j], ":", label="Fourier forward (no c)")
        axes[r, 0].set(title=f"{cond}: sensor {j}", xlabel="time step", ylabel="pressure [a.u.]")
        axes[r, 0].legend(fontsize=7)
        v = float(max(np.abs(Akw).max(), np.abs(Aff).max()))
        for k, (img, t) in enumerate(
            ((Akw, "k-Wave adjoint of K"), (Aff, "Fourier adjoint of K/c"))
        ):
            im = axes[r, 1 + k].imshow(img, cmap="RdBu_r", vmin=-v, vmax=v)
            axes[r, 1 + k].set_title(t)
            fig.colorbar(im, ax=axes[r, 1 + k])
    fig.savefig(args.output_dir / "scaling_validation.png", dpi=130)


if __name__ == "__main__":
    main()
