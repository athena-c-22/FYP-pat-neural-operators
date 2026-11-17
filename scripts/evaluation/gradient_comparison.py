"""Stage 3: compare the data-fidelity gradient through the learned FNO with the k-Wave reference.

Functional (plain Euclidean norm, no quadrature weights; see the Stage 2 report):

    J_G(x) = 1/2 ||G(x) - b||^2   with G = pre-processing -> TinyFNO2d -> de-normalisation,
    J_A(x) = 1/2 ||A x - b||^2    with A = k-Wave forward.

Gradients at a test point ``x`` (``r_true = A x - b``, ``r_sur = G(x) - b``, ``J = G'(x)``):

    g_ref = A*  r_true     reference (k-Wave adjoint; approximate adjoint, see Stage 2)
    g_ad  = J^T r_sur      autograd through the FNO (what reconstruction with G would use)
    g_res = A*  r_sur      only the residual is the surrogate's
    g_jac = J^T r_true     only the operator (Jacobian) is the surrogate's

Part 1 (no k-Wave): finite-difference check of autograd through the float64 FNO.
Part 2 (k-Wave): the four gradients, metrics against ``g_ref``, directional derivatives, a
finite-difference check of the reference itself, and an exact line search along -g.

Test points per bundled sample (periodic 89 degree checkpoint): ``truth`` (b = A x, so
g_ref = 0 and g_ad is pure surrogate bias), ``half`` (0.5 x), ``smooth_pert`` (x + smooth field)
and ``noisy`` (x + white noise).

Outputs (``--output-dir``, default ``outputs/gradient_comparison``): fd_check.csv, results.csv,
summary.json, fd_check.png, gradient_images.png, gradient_metrics.png, meta.json.
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from pat_fno.evaluation.gradient_comparison import (
    fd_check,
    gradient_metrics,
    quadratic_step_decrease,
)
from pat_fno.models.fno import TinyFNO2d
from scripts.evaluation.adjoint_consistency import apply_adjoint, apply_forward, make_setting

SEED = 20260728
ROOT = Path(__file__).resolve().parents[2]
SAMPLES = ROOT / "examples" / "sample_data" / "periodic_large_test_samples.npz"
CHECKPOINT = ROOT / "examples" / "checkpoints" / "fno_only_periodic_large.pt"
CONDITION = "periodic_theta89"
POINTS = ("truth", "half", "smooth_pert", "noisy")
STEPS = (1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6)


# --------------------------------------------------------------------------- model wrapper
def load_surrogate(dtype=torch.float64):
    """Return ``G(x: (64,64) tensor) -> (64,91) tensor`` using the bundled checkpoint."""
    ck = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    a = ck["arguments"]
    model = TinyFNO2d(a["modes"], a["modes"], a["width"], a["layers"])
    model.load_state_dict(ck["model"])
    model.eval()
    cdtype = torch.complex128 if dtype == torch.float64 else torch.complex64
    for p in model.parameters():  # module.to(float64) would DROP the imaginary parts
        p.data = p.data.to(cdtype if p.is_complex() else dtype)
        p.requires_grad_(False)
    n = ck["normalization"]

    def g(x: torch.Tensor) -> torch.Tensor:
        z = F.interpolate(x[None, None], size=(64, 91), mode="bilinear", align_corners=False)
        z = (z - n["p0_mean"]) / n["p0_std"]
        return (model(z)[0, 0] * n["data_std"] + n["data_mean"]).to(x.dtype)

    return g


def smooth_field(shape, rng, width=0.08):
    noise = np.fft.fft2(rng.standard_normal(shape))
    kk = np.hypot(*np.meshgrid(np.fft.fftfreq(shape[0]), np.fft.fftfreq(shape[1]), indexing="ij"))
    f = np.real(np.fft.ifft2(noise * np.exp(-((kk / width) ** 2))))
    return f / np.abs(f).max()


def make_point(kind: str, p0: np.ndarray, rng) -> np.ndarray:
    scale = float(np.std(p0))
    if kind == "truth":
        return p0.copy()
    if kind == "half":
        return 0.5 * p0
    if kind == "smooth_pert":
        return p0 + 0.5 * scale * smooth_field(p0.shape, rng)
    if kind == "noisy":
        return p0 + 0.1 * scale * rng.standard_normal(p0.shape)
    raise KeyError(kind)


def unit_directions(shape, rng):
    """Two smooth and two white unit-norm directions."""
    vs = [smooth_field(shape, rng), smooth_field(shape, rng, 0.15)]
    vs += [rng.standard_normal(shape), rng.standard_normal(shape)]
    return [v / np.linalg.norm(v) for v in vs]


# --------------------------------------------------------------------------- part 1
def part1(samples, out_dir):
    g64 = load_surrogate(torch.float64)
    rows = []
    for si, p0 in enumerate(samples["initial_pressure"]):
        b = torch.from_numpy(samples["kwave_target"][si].astype(np.float64))
        rng = np.random.default_rng(SEED + 1000 + si)
        for kind in POINTS:
            x = torch.from_numpy(make_point(kind, p0.astype(np.float64), rng))
            dirs = [torch.from_numpy(v) for v in unit_directions(p0.shape, rng)]
            res = fd_check(lambda z, b=b: 0.5 * ((g64(z) - b) ** 2).sum(), x, dirs, STEPS)
            for k in range(len(dirs)):
                for j, h in enumerate(STEPS):
                    rows.append(
                        {
                            "sample": si,
                            "point": kind,
                            "direction": ["smooth", "smooth_wide", "white", "white"][k] + str(k),
                            "h": h,
                            "ad": res["ad"][k],
                            "fd": res["fd"][k, j],
                            "rel_err": res["rel_err"][k, j],
                        }
                    )
    _write_csv(out_dir / "fd_check.csv", rows)
    return rows


# --------------------------------------------------------------------------- part 2 (worker)
def job(args):
    si, kind, seed = args
    torch.set_num_threads(1)
    d = np.load(SAMPLES)
    p0 = d["initial_pressure"][si].astype(np.float64)
    b = d["kwave_target"][si].astype(np.float64)
    rng = np.random.default_rng(seed)
    x = make_point(kind, p0, rng)
    dirs = unit_directions(p0.shape, rng)
    s = make_setting(CONDITION)
    g64, g32 = load_surrogate(torch.float64), load_surrogate(torch.float32)

    def kf(z):
        return apply_forward("kwave", z, s).astype(np.float64)

    def ka(y):
        return apply_adjoint("kwave", y, s).astype(np.float64)

    def vjp(g, xx, y, dtype):
        xt = torch.from_numpy(xx.astype(np.float64)).to(dtype).requires_grad_(True)
        out = g(xt)
        (grad,) = torch.autograd.grad(out, xt, grad_outputs=torch.from_numpy(y).to(dtype))
        return grad.numpy().astype(np.float64), out.detach().numpy().astype(np.float64)

    ax = kf(x)
    r_true = ax - b
    gx = g64(torch.from_numpy(x)).numpy()
    r_sur = gx - b
    g_ref = ka(r_true)
    g_res = ka(r_sur)
    g_jac, _ = vjp(g64, x, r_true, torch.float64)
    g_ad, _ = vjp(g64, x, r_sur, torch.float64)
    g_ad32, _ = vjp(g32, x, r_sur, torch.float32)

    # autograd of the scalar functional must equal the explicit J^T r_sur
    xt = torch.from_numpy(x).requires_grad_(True)
    (g_scalar,) = torch.autograd.grad(0.5 * ((g64(xt) - torch.from_numpy(b)) ** 2).sum(), xt)
    consistency = float(np.linalg.norm(g_scalar.numpy() - g_ad) / (np.linalg.norm(g_ad) + 1e-300))

    row = {
        "sample": si,
        "point": kind,
        "J_A": 0.5 * float(np.sum(r_true**2)),
        "J_G": 0.5 * float(np.sum(r_sur**2)),
        "target_vs_kwave_fwd_of_p0": float(
            np.linalg.norm(kf(p0) - b) / np.linalg.norm(b) if kind == "truth" else np.nan
        ),
        "norm_g_ref": float(np.linalg.norm(g_ref)),
        "norm_g_ad": float(np.linalg.norm(g_ad)),
        "norm_g_res": float(np.linalg.norm(g_res)),
        "norm_g_jac": float(np.linalg.norm(g_jac)),
        "ad_scalar_vs_vjp": consistency,
        "ad_f32_vs_f64": float(np.linalg.norm(g_ad32 - g_ad) / np.linalg.norm(g_ad)),
    }
    vs = np.stack(dirs)
    for name, g in (("ad", g_ad), ("res", g_res), ("jac", g_jac)):
        if row["norm_g_ref"] > 1e-8 * row["norm_g_ad"]:
            m = gradient_metrics(g, g_ref, vs)
        else:  # reference gradient is (numerically) zero: relative metrics are undefined
            m = dict.fromkeys(("rel_l2", "cos", "scale", "rel_l2_fit", "sign_agree"), np.nan)
            m["norm_ratio"] = np.nan
        row.update({f"{name}_{k}": v for k, v in m.items()})

    # directional derivatives: AD, reference, FD of J_G (float64), FD of J_A (k-Wave, h = 1)
    def jg(z):
        return 0.5 * ((g64(z) - torch.from_numpy(b)) ** 2).sum()

    for k, v in enumerate(dirs):
        row[f"dd{k}_ad"] = float(np.sum(g_ad * v))
        row[f"dd{k}_ref"] = float(np.sum(g_ref * v))
        row[f"dd{k}_fd_JG"] = float(
            (jg(torch.from_numpy(x + 1e-4 * v)) - jg(torch.from_numpy(x - 1e-4 * v))) / 2e-4
        )
    for k in (0, 2):  # one smooth, one white direction (2 forwards each)
        h = 1.0
        jp, jm = (
            0.5 * np.sum((kf(x + h * dirs[k]) - b) ** 2),
            0.5 * np.sum((kf(x - h * dirs[k]) - b) ** 2),
        )
        row[f"dd{k}_fd_JA"] = float((jp - jm) / (2 * h))

    # exact line search of J_A along -g (A linear): needs A g
    for name, g in (("ad", g_ad), ("ref", g_ref)):
        if np.linalg.norm(g) > 1e-8 * max(1.0, np.linalg.norm(g_ad)):
            t, dec = quadratic_step_decrease(r_true, kf(g))
        else:
            t, dec = np.nan, np.nan
        row[f"t_star_{name}"], row[f"decrease_{name}"] = t, dec
    row["decrease_frac"] = (
        row["decrease_ad"] / row["decrease_ref"]
        if np.isfinite(row["decrease_ref"]) and row["decrease_ref"] > 0
        else np.nan
    )
    arrays = {
        "x": x,
        "g_ref": g_ref,
        "g_ad": g_ad,
        "g_res": g_res,
        "g_jac": g_jac,
        "r_true": r_true,
        "r_sur": r_sur,
    }
    return row, arrays


def part2(samples, out_dir, workers):
    jobs = [
        (si, kind, SEED + 100 * si + ki)
        for si in range(len(samples["initial_pressure"]))
        for ki, kind in enumerate(POINTS)
    ]
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        out = list(ex.map(job, jobs))
    print(f"[part2] {len(jobs)} jobs in {time.time() - t0:.0f}s")
    rows = [o[0] for o in out]
    _write_csv(out_dir / "results.csv", rows)
    np.savez_compressed(
        out_dir / "gradients.npz",
        **{
            f"s{r['sample']}_{r['point']}__{k}": v
            for r, (_, a) in zip(rows, out, strict=True)
            for k, v in a.items()
        },
    )
    return rows


# --------------------------------------------------------------------------- output helpers
def _write_csv(path, rows):
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def _nanstat(vals):
    v = np.array([x for x in vals if np.isfinite(x)], float)
    if v.size == 0:
        return None
    return {
        "median": float(np.median(v)),
        "min": float(v.min()),
        "max": float(v.max()),
        "n": int(v.size),
    }


def summarise(fd_rows, rows):
    summ = {"fd_check": {}, "gradients": {}}
    for h in STEPS:
        summ["fd_check"][f"{h:g}"] = _nanstat([r["rel_err"] for r in fd_rows if r["h"] == h])
    for kind in POINTS:
        sel = [r for r in rows if r["point"] == kind]
        summ["gradients"][kind] = {
            k: _nanstat([r[k] for r in sel])
            for k in (
                "norm_g_ref",
                "norm_g_ad",
                "ad_rel_l2",
                "ad_cos",
                "ad_scale",
                "ad_rel_l2_fit",
                "res_rel_l2",
                "res_cos",
                "jac_rel_l2",
                "jac_cos",
                "ad_sign_agree",
                "decrease_frac",
                "t_star_ad",
                "t_star_ref",
                "ad_f32_vs_f64",
                "ad_scalar_vs_vjp",
            )
        }
    for k in (0, 2):
        errs = [
            abs(r[f"dd{k}_fd_JA"] - r[f"dd{k}_ref"]) / (abs(r[f"dd{k}_fd_JA"]) + 1e-12)
            for r in rows
            if r["norm_g_ref"] > 1e-8 * r["norm_g_ad"]
        ]
        summ["gradients"][f"ref_vs_fd_JA_dir{k}"] = _nanstat(errs)
    return summ


def fig_fd(fd_rows, path):
    fig, ax = plt.subplots(figsize=(6, 4))
    hs = np.array(STEPS)
    for kind, col in zip(POINTS, ("C0", "C1", "C2", "C3"), strict=True):
        errs = np.array(
            [[r["rel_err"] for r in fd_rows if r["point"] == kind and r["h"] == h] for h in STEPS]
        )
        ax.loglog(hs, np.median(errs, axis=1), "-o", color=col, label=f"{kind} (median)")
        ax.fill_between(hs, errs.min(axis=1), errs.max(axis=1), color=col, alpha=0.12)
    ax.set_xlabel("finite-difference step h")
    ax.set_ylabel("|FD - AD| / |AD|")
    ax.set_title("Autograd through float64 FNO vs central differences")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_images(path, npz, sample=0, point="half"):
    keys = ("g_ref", "g_ad", "g_res", "g_jac")
    titles = (
        "g_ref = A*(Ax-b)",
        "g_ad = J^T(G(x)-b)",
        "g_res = A*(G(x)-b)",
        "g_jac = J^T(Ax-b)",
    )
    arrs = [npz[f"s{sample}_{point}__{k}"] for k in keys]
    vmax = max(np.abs(a).max() for a in arrs)
    fig, axs = plt.subplots(2, 4, figsize=(13, 6.2))
    for i, (a, t) in enumerate(zip(arrs, titles, strict=True)):
        im = axs[0, i].imshow(a, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        axs[0, i].set_title(t, fontsize=9)
        if i:
            axs[1, i].imshow(a - arrs[0], cmap="RdBu_r", vmin=-vmax, vmax=vmax)
            axs[1, i].set_title("minus g_ref", fontsize=9)
        else:
            axs[1, i].imshow(npz[f"s{sample}_{point}__x"], cmap="gray")
            axs[1, i].set_title("test point x", fontsize=9)
    for a in axs.ravel():
        a.axis("off")
    fig.colorbar(im, ax=axs, shrink=0.6, label="gradient (shared colour scale)")
    fig.suptitle(f"sample {sample}, point '{point}', periodic 89 deg")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_metrics(rows, path):
    kinds = [p for p in POINTS if p != "truth"]
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    w = 0.25
    for j, (name, lab) in enumerate((("ad", "g_ad"), ("res", "g_res"), ("jac", "g_jac"))):
        for i, kind in enumerate(kinds):
            vals = [r[f"{name}_cos"] for r in rows if r["point"] == kind]
            rel = [r[f"{name}_rel_l2"] for r in rows if r["point"] == kind]
            axs[0].scatter(
                [i + (j - 1) * w] * len(vals), vals, color=f"C{j}", label=lab if i == 0 else None
            )
            axs[1].scatter(
                [i + (j - 1) * w] * len(rel), rel, color=f"C{j}", label=lab if i == 0 else None
            )
    for a, t in zip(
        axs, ("cosine similarity with g_ref", "relative L2 error vs g_ref"), strict=True
    ):
        a.set_xticks(range(len(kinds)), kinds)
        a.set_title(t)
        a.grid(alpha=0.3)
    axs[1].axhline(1.0, color="k", lw=0.6, ls="--")
    axs[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--output-dir", type=Path, default=Path("outputs/gradient_comparison"))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--skip-kwave", action="store_true", help="only run the FD check (Part 1)")
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(SEED)
    samples = dict(np.load(SAMPLES))

    fd_rows = part1(samples, args.output_dir)
    fig_fd(fd_rows, args.output_dir / "fd_check.png")
    print("[part1] median rel err by h:")
    for h in STEPS:
        e = [r["rel_err"] for r in fd_rows if r["h"] == h]
        print(f"   h={h:g}: median {np.median(e):.3g}  max {np.max(e):.3g}")
    rows = []
    if not args.skip_kwave:
        rows = part2(samples, args.output_dir, args.workers)
        npz = np.load(args.output_dir / "gradients.npz")
        fig_images(args.output_dir / "gradient_images.png", npz, 0, "half")
        fig_metrics(rows, args.output_dir / "gradient_metrics.png")
    summ = summarise(fd_rows, rows) if rows else {"fd_check": summarise(fd_rows, [])["fd_check"]}
    (args.output_dir / "summary.json").write_text(json.dumps(summ, indent=2))
    (args.output_dir / "meta.json").write_text(
        json.dumps(
            {
                "seed": SEED,
                "python": platform.python_version(),
                "torch": torch.__version__,
                "numpy": np.__version__,
                "condition": CONDITION,
                "checkpoint": str(CHECKPOINT.relative_to(ROOT)),
                "samples": str(SAMPLES.relative_to(ROOT)),
                "points": POINTS,
                "fd_steps": STEPS,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
