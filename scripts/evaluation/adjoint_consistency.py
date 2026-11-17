"""Stage 2: adjoint-consistency evaluation of the PAT forward / adjoint operator pairs.

Part A (dot-product test, 64x64 grid): for several (x, y) pairs and both acquisition
conditions, compute <A x, y> and <x, B y> for each candidate pair (A, B) and the metrics of
``pat_fno.evaluation.adjoint_consistency``.

Part B (explicit matrices, 12x12 grid): assemble A and B column by column from unit vectors
and compare A^T with B directly (best scalar, residual shape error). 12x12 is the smallest
grid that works with k-Wave-python 0.5.0rc1 (mirrored grids with fewer than 256 points crash
inside k-Wave's index-type casting).

Outputs (``--output-dir``, default ``outputs/adjoint_consistency``): results.csv (one row per
pair and (x, y)), summary.csv, explicit_matrix.json, explicit_matrices.npz,
adjoint_scatter.png, explicit_matrix.png, meta.json.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from pat_fno.config.mnist_pat import CONDITIONS, DX, MEDIUM_DENSITY, SOUND_SPEED
from pat_fno.data.mnist import build_setting
from pat_fno.evaluation.adjoint_consistency import (
    adjoint_metrics,
    explicit_matrix,
    matrix_adjoint_comparison,
)
from pat_fno.operators.fourier import fpat_adjoint_2d, fpat_forward_2d
from pat_fno.operators.fourier.numpy_reference import numpy_adjoint_2d, numpy_forward_2d
from pat_fno.operators.kwave import kwave_adjoint_2d, kwave_forward_2d

SEED = 20260728
C = SOUND_SPEED
ROOT = Path(__file__).resolve().parents[2]
SAMPLES = ROOT / "examples" / "sample_data" / "periodic_large_test_samples.npz"
PAIRS = (
    "kwave",
    "fourier_np_raw",
    "fourier_np_c",
    "fpat_torch_raw",
    "fpat_autograd",
    "mixed_kwfwd_Fadj",
)


def make_setting(condition: str, n: int | None = None) -> SimpleNamespace:
    """Setting for ``condition``; ``n`` overrides the (square) grid size (default 64)."""
    s = build_setting(condition)
    if n is not None:
        dt = DX / C
        nt = int(math.ceil(math.hypot(n * DX, n * DX) / (C * dt)))
        s.Nx = s.Ny = n
        s.Nt = nt
        s.t_array = np.arange(nt) * dt
    s.computation.interpolationMethodA = "cubic"
    s.computation.interpolationMethodI = "cubic"
    return s


# --------------------------------------------------------------------------- operators
def _theta(s):
    return float(s.computation.theta_max)


def apply_forward(name: str, x: np.ndarray, s) -> np.ndarray:
    if name == "kwave":
        return kwave_forward_2d(x.astype(np.float32), s)
    if name in ("fourier_np_raw", "fourier_raw"):
        return numpy_forward_2d(x, s)
    if name == "fourier_np_c":
        return C * numpy_forward_2d(x, s)
    if name == "fpat":
        return fpat_forward_2d(
            torch.from_numpy(x.astype(np.float64)), _theta(s), C, s.Nt, (s.dx, s.dy), s.dt
        ).numpy()
    raise KeyError(name)


def apply_adjoint(name: str, y: np.ndarray, s, x_for_vjp: np.ndarray | None = None) -> np.ndarray:
    if name == "kwave":
        return kwave_adjoint_2d(y.astype(np.float32), s)
    if name == "fourier_np_raw":
        return numpy_adjoint_2d(y, s)
    if name == "fourier_np_c":
        return numpy_adjoint_2d(y / C, s)
    if name == "fpat":
        return fpat_adjoint_2d(
            torch.from_numpy(y.astype(np.float64)), _theta(s), C, (s.Nx, s.Ny), (s.dx, s.dy), s.dt
        ).numpy()
    if name == "fpat_autograd":  # exact transpose of the torch forward at (any) x (it is linear)
        x = torch.zeros(s.Nx, s.Ny, dtype=torch.float64, requires_grad=True)
        out = fpat_forward_2d(x, _theta(s), C, s.Nt, (s.dx, s.dy), s.dt)
        (g,) = torch.autograd.grad(out, x, grad_outputs=torch.from_numpy(y.astype(np.float64)))
        return g.numpy()
    raise KeyError(name)


PAIR_OPS = {
    "kwave": ("kwave", "kwave"),
    "fourier_np_raw": ("fourier_np_raw", "fourier_np_raw"),
    "fourier_np_c": ("fourier_np_c", "fourier_np_c"),
    "fpat_torch_raw": ("fpat", "fpat"),
    "fpat_autograd": ("fpat", "fpat_autograd"),
    "mixed_kwfwd_Fadj": ("kwave", "fourier_np_c"),
}


# --------------------------------------------------------------------------- test inputs
def smooth_field(shape, rng, width=0.08):
    noise = np.fft.fft2(rng.standard_normal(shape))
    kk = np.hypot(*np.meshgrid(np.fft.fftfreq(shape[0]), np.fft.fftfreq(shape[1]), indexing="ij"))
    f = np.real(np.fft.ifft2(noise * np.exp(-((kk / width) ** 2))))
    return f / np.abs(f).max()


def build_pairs(s, rng):
    """List of (kind, x_name, x, y_name, y) test pairs. White noise is N(0,1) before any scaling."""
    n = s.Nx
    xx, yy = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    phantoms = {
        "gauss_blob": np.exp(-(((xx - 24) ** 2 + (yy - 30) ** 2) / 32.0)),
        "disc": (((xx - 32) ** 2 + (yy - 32) ** 2) < 100).astype(float),
        "two_gauss": np.exp(-(((xx - 20) ** 2 + (yy - 20) ** 2) / 18))
        + 0.6 * np.exp(-(((xx - 42) ** 2 + (yy - 38) ** 2) / 32)),
    }
    dshape = (s.Ny, s.Nt)
    out = []
    for i in range(5):
        out.append(
            (
                "random_white",
                f"white_s{i}",
                rng.standard_normal((n, n)),
                f"white_s{i}",
                rng.standard_normal(dshape),
            )
        )
    for i in range(3):
        out.append(
            (
                "random_smooth_x",
                f"smooth_s{i}",
                smooth_field((n, n), rng),
                f"white_s{i}",
                rng.standard_normal(dshape),
            )
        )
    for k, v in phantoms.items():
        out.append(("phantom", k, v, f"white_{k}", rng.standard_normal(dshape)))
    if SAMPLES.exists():
        d = np.load(SAMPLES)
        m = len(d["initial_pressure"])
        for i in range(m):
            nm = f"mnist_{int(d['source_index'][i])}"
            out.append(
                (
                    "mnist_x_white_y",
                    nm,
                    d["initial_pressure"][i].astype(float),
                    f"white_{nm}",
                    rng.standard_normal(dshape),
                )
            )
            j = (i + 1) % m  # realistic y: k-Wave data of a *different* sample, never of x itself
            out.append(
                (
                    "mnist_x_kwave_y",
                    nm,
                    d["initial_pressure"][i].astype(float),
                    f"kwave_data_of_{int(d['source_index'][j])}",
                    d["kwave_target"][j].astype(float),
                )
            )
    return out


# --------------------------------------------------------------------------- Part A
def part_a(conditions, out_dir, rng):
    rows = []
    t0 = time.time()
    for cond in conditions:
        s = make_setting(cond)
        wx, wy = s.dx * s.dy, s.dy * s.dt  # candidate physical quadrature weights (rectangle rule)
        fcache, acache = {}, {}
        for kind, xn, x, yn, y in build_pairs(s, rng):
            for pair in PAIRS:
                fop, aop = PAIR_OPS[pair]
                fk, ak = (fop, xn, id(x)), (aop, yn, id(y))
                if fk not in fcache:
                    fcache[fk] = apply_forward(fop, x, s)
                if ak not in acache:
                    acache[ak] = apply_adjoint(aop, y, s)
                ax, bty = fcache[fk], acache[ak]
                m = adjoint_metrics(ax, y, x, bty)
                mp = adjoint_metrics(ax, y, x, bty, wx=wx, wy=wy)
                rows.append(
                    {
                        "condition": cond,
                        "pair": pair,
                        "input_kind": kind,
                        "x": xn,
                        "y": yn,
                        **m,
                        "sym_rel_phys": mp["sym_rel"],
                        "ratio_phys": mp["ratio"],
                        "norm_x": float(np.linalg.norm(x)),
                        "norm_y": float(np.linalg.norm(y)),
                        "norm_Ax": float(np.linalg.norm(ax)),
                        "norm_Bty": float(np.linalg.norm(bty)),
                    }
                )
            print(f"[{time.time() - t0:6.1f}s] {cond} {kind:16s} {xn:16s} {yn}", flush=True)
    return rows


def summarise(rows):
    out = []
    for cond in dict.fromkeys(r["condition"] for r in rows):
        for pair in PAIRS:
            for group, pred in (
                ("all", lambda r: True),
                ("random_white_only", lambda r: r["input_kind"] == "random_white"),
            ):
                sel = [r for r in rows if r["condition"] == cond and r["pair"] == pair and pred(r)]
                if not sel:
                    continue
                rec = {"condition": cond, "pair": pair, "group": group, "n": len(sel)}
                for key in ("sym_rel", "cs_rel", "marta", "ratio", "sym_rel_phys"):
                    v = np.array([r[key] for r in sel], float)
                    rec[f"{key}_median"], rec[f"{key}_min"], rec[f"{key}_max"] = (
                        float(np.nanmedian(v)),
                        float(np.nanmin(v)),
                        float(np.nanmax(v)),
                    )
                out.append(rec)
    return out


# --------------------------------------------------------------------------- Part B
def _kw_column(args):
    kind, arr, cond, n = args
    s = make_setting(cond, n)
    return (kwave_forward_2d(arr, s) if kind == "fwd" else kwave_adjoint_2d(arr, s)).astype(
        np.float64
    )


def part_b(conditions, n, workers):
    results, mats = {}, {}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for cond in conditions:
            s = make_setting(cond, n)
            shape_x, shape_y = (s.Nx, s.Ny), (s.Ny, s.Nt)
            t0 = time.time()
            a_kw = explicit_matrix(
                None,
                shape_x,
                columns=lambda us, cond=cond: list(
                    pool.map(_kw_column, [("fwd", u, cond, n) for u in us], chunksize=4)
                ),
            )
            b_kw = explicit_matrix(
                None,
                shape_y,
                columns=lambda us, cond=cond: list(
                    pool.map(_kw_column, [("adj", u, cond, n) for u in us], chunksize=4)
                ),
            )
            print(
                f"[B {cond}] k-Wave matrices {a_kw.shape} {b_kw.shape} in {time.time() - t0:.0f}s",
                flush=True,
            )
            a_np = explicit_matrix(
                lambda v, s=s: apply_forward("fourier_np_raw", v.astype(float), s), shape_x
            )
            b_np = explicit_matrix(
                lambda v, s=s: apply_adjoint("fourier_np_raw", v.astype(float), s), shape_y
            )
            a_fp = explicit_matrix(lambda v, s=s: apply_forward("fpat", v, s), shape_x)
            b_fp = explicit_matrix(lambda v, s=s: apply_adjoint("fpat", v, s), shape_y)
            b_ag = explicit_matrix(lambda v, s=s: apply_adjoint("fpat_autograd", v, s), shape_y)
            mats[cond] = dict(
                a_kw=a_kw, b_kw=b_kw, a_np=a_np, b_np=b_np, a_fp=a_fp, b_fp=b_fp, b_ag=b_ag
            )
            comp = {
                "kwave: A^T vs B": matrix_adjoint_comparison(a_kw, b_kw),
                "fourier_np_raw: A^T vs B": matrix_adjoint_comparison(a_np, b_np),
                "fourier_np_c: (cA)^T vs B/c": matrix_adjoint_comparison(C * a_np, b_np / C),
                "fpat_torch_raw: A^T vs B": matrix_adjoint_comparison(a_fp, b_fp),
                "fpat_autograd (control): A^T vs VJP": matrix_adjoint_comparison(a_fp, b_ag),
                "mixed: kwave A^T vs Fourier B/c": matrix_adjoint_comparison(a_kw, b_np / C),
                "forward agreement: kwave A vs c*Fourier A": matrix_adjoint_comparison(
                    a_kw.T, C * a_np
                ),
                "forward agreement: numpy vs torch Fourier": matrix_adjoint_comparison(
                    a_np.T, a_fp
                ),
            }
            results[cond] = {
                "grid": n,
                "Nt": s.Nt,
                "n_cols_forward": a_kw.shape[1],
                "n_cols_adjoint": b_kw.shape[1],
                "spectral_norm_A_kwave": float(np.linalg.norm(a_kw, 2)),
                "spectral_norm_B_kwave": float(np.linalg.norm(b_kw, 2)),
                "comparisons": comp,
            }
    return results, mats


# --------------------------------------------------------------------------- figures
def fig_scatter(rows, conditions, path):
    pairs = [p for p in PAIRS]
    fig, axes = plt.subplots(
        len(conditions),
        len(pairs),
        figsize=(3.2 * len(pairs), 3.3 * len(conditions)),
        constrained_layout=True,
        squeeze=False,
    )
    for r, cond in enumerate(conditions):
        for c_, pair in enumerate(pairs):
            ax = axes[r, c_]
            sel = [q for q in rows if q["condition"] == cond and q["pair"] == pair]
            lhs = np.array([q["lhs"] for q in sel])
            rhs = np.array([q["rhs"] for q in sel])
            kinds = [q["input_kind"] for q in sel]
            for k in dict.fromkeys(kinds):
                idx = [i for i, kk in enumerate(kinds) if kk == k]
                ax.scatter(rhs[idx], lhs[idx], s=18, label=k)
            lim = np.abs(np.concatenate([lhs, rhs])).max() * 1.1
            ax.plot([-lim, lim], [-lim, lim], "k--", lw=0.8)
            ax.set_xscale("symlog", linthresh=lim * 1e-3)
            ax.set_yscale("symlog", linthresh=lim * 1e-3)
            ax.set(title=f"{pair}\n{cond}", xlabel="<x, B y>", ylabel="<A x, y>")
            if r == 0 and c_ == 0:
                ax.legend(fontsize=6)
    fig.suptitle("Dot-product test (dashed = perfect adjoint). Symmetric-log axes.")
    fig.savefig(path, dpi=120)
    plt.close(fig)


def fig_matrix(mats, conditions, path):
    fig, axes = plt.subplots(
        len(conditions), 2, figsize=(9, 4 * len(conditions)), constrained_layout=True, squeeze=False
    )
    for r, cond in enumerate(conditions):
        m = mats[cond]
        for c_, (a, b, t) in enumerate(
            ((m["a_kw"], m["b_kw"], "k-Wave"), (m["a_np"], m["b_np"], "Fourier (NumPy, raw)"))
        ):
            at = a.T
            ax = axes[r, c_]
            ax.scatter(b.ravel(), at.ravel(), s=2, alpha=0.4)
            lim = max(np.abs(at).max(), np.abs(b).max())
            ax.plot([-lim, lim], [-lim, lim], "k--", lw=0.8, label="A^T = B")
            ax.set(title=f"{t}, {cond}", xlabel="entries of B", ylabel="entries of A^T")
            ax.legend(fontsize=7)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--output-dir", type=Path, default=Path("outputs/adjoint_consistency"))
    ap.add_argument("--conditions", nargs="+", default=list(CONDITIONS))
    ap.add_argument("--matrix-grid", type=int, default=12)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--skip-matrix", action="store_true")
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    t0 = time.time()
    rows = part_a(args.conditions, args.output_dir, rng)
    with (args.output_dir / "results.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    summary = summarise(rows)
    with (args.output_dir / "summary.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(summary[0]))
        w.writeheader()
        w.writerows(summary)
    fig_scatter(rows, args.conditions, args.output_dir / "adjoint_scatter.png")
    for s_ in summary:
        if s_["group"] == "all":
            print(
                f"{s_['condition']:22s} {s_['pair']:18s} sym_rel med {s_['sym_rel_median']:.3g} "
                f"[{s_['sym_rel_min']:.3g},{s_['sym_rel_max']:.3g}] "
                f"ratio med {s_['ratio_median']:.4g} "
                f"[{s_['ratio_min']:.4g},{s_['ratio_max']:.4g}]"
            )

    if not args.skip_matrix:
        res, mats = part_b(args.conditions, args.matrix_grid, args.workers)
        (args.output_dir / "explicit_matrix.json").write_text(json.dumps(res, indent=2))
        np.savez_compressed(
            args.output_dir / "explicit_matrices.npz",
            **{f"{c}__{k}": v for c, d in mats.items() for k, v in d.items()},
        )
        fig_matrix(mats, args.conditions, args.output_dir / "explicit_matrix.png")
        for cond, r in res.items():
            for k, v in r["comparisons"].items():
                print(
                    f"[B {cond}] {k:45s} raw {v['rel_fro_raw']:.3g}  s={v['scale_s']:.4g}  "
                    f"after {v['rel_fro_after_scale']:.3g}  cos {v['cos_sim']:.4f}"
                )

    meta = {
        "seed": SEED,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "sound_speed": C,
        "medium_density": MEDIUM_DENSITY,
        "runtime_s": time.time() - t0,
        "matrix_grid": args.matrix_grid,
        "n_rows": len(rows),
    }
    (args.output_dir / "meta.json").write_text(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
