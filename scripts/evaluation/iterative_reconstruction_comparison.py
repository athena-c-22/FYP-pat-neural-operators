"""Zero- and backprojection-initialised iterative PAT reconstruction comparison.

Runs gradient-descent reconstruction with an exact physical (k-Wave) line search,
using three search directions:

    reference : B(A x - b)                      approximate k-Wave adjoint B
    F0, F5    : J_F(x)^T (F(x) - b)             autograd through the frozen forward FNO
    fourier_c : A_F^*(A x - b)                  analytical Fourier adjoint (c convention)
    G_<run>   : G_phi(A x - b)                  learned adjoint (--learned checkpoints)

The residual A x - b is always computed with k-Wave. Every method follows its own
trajectory. At every iterate of every trajectory, all directions are computed so that
the cosine alignment with the reference can be logged per iteration. The rows of the
reference trajectory therefore give the "common trajectory" comparison, and the other
rows give the independent ones. Wall-clock time of each direction is logged as t_<name>.

Run from the repository root:

    $env:PYTHONPATH = "$PWD\\src;$PWD"
    python -m scripts.evaluation.iterative_reconstruction_comparison `
        --checkpoint-archive <same value used for heldout_gradient_comparison.py>
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch

from scripts.evaluation.adjoint_consistency import apply_adjoint, apply_forward, make_setting
from scripts.evaluation.forward_checkpoint_comparison import MEMBERS, read_checkpoint, surrogate
from scripts.evaluation.learned_adjoint_eval import load_model, predict

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SAMPLES = PROJECT_ROOT / "examples" / "sample_data" / "periodic_large_test_samples.npz"
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "iterative_reconstruction"
DEFAULT_LEARNED = [PROJECT_ROOT / "outputs" / "learned_adjoint" / "runs_4k_e300" / "lam0_seed0.pt"]
CONDITION = "periodic_theta89"
IMAGE_SHAPE = (64, 64)
DATA_SHAPE = (64, 91)


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint-archive", type=Path,
                        help="Same checkpoint archive argument as heldout_gradient_comparison.py "
                             "(required unless --no-fno).")
    parser.add_argument("--samples", type=Path, default=DEFAULT_SAMPLES)
    parser.add_argument("--sample-index", type=int, nargs="+", default=[0])
    parser.add_argument("--learned", type=Path, nargs="*", default=DEFAULT_LEARNED,
                        help="Learned-adjoint checkpoints (train_learned_adjoint.py); none to skip.")
    parser.add_argument("--no-fourier", action="store_true", help="Skip the fourier_c direction.")
    parser.add_argument("--no-fno", action="store_true", help="Skip the F0/F5 Jacobian directions.")
    parser.add_argument("--fallback", action="store_true",
                        help="Also run '<G>+fallback' trajectories: step along G, and take one "
                             "k-Wave adjoint step whenever G is not a descent direction.")
    parser.add_argument("--trajectories", nargs="+",
                        help="Only run these trajectories (directions are still all logged).")
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--init", choices=("zero", "backprojection", "both"), default="both")
    parser.add_argument("--use-stored-b", action="store_true",
                        help="Use the stored k-Wave data instead of recomputing b = A(x_true).")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_sample(path: Path, index: int, setting, use_stored_b: bool):
    """Load x_true and b for one sample from the debug .npz file."""
    with np.load(path) as data:
        x_true = data["initial_pressure"][index].astype(float)
        b_stored = data["kwave_target"][index].astype(float)
        label = int(data["label"][index])
        source_index = int(data["source_index"][index])
    if x_true.shape != IMAGE_SHAPE or b_stored.shape != DATA_SHAPE:
        raise ValueError(f"Unexpected shapes {x_true.shape}, {b_stored.shape}")

    b_fresh = forward(x_true, setting)
    rel_diff = np.linalg.norm(b_fresh - b_stored) / np.linalg.norm(b_stored)
    print(f"Sample {index}: label={label}, source_index={source_index}")
    print(f"  ||x_true||={np.linalg.norm(x_true):.4f}  ||b_stored||={np.linalg.norm(b_stored):.4f}")
    print(f"  relative difference fresh vs stored b: {rel_diff:.3e}")
    b = b_stored if use_stored_b else b_fresh
    return x_true, b, {"label": label, "source_index": source_index, "b_rel_diff": rel_diff}


# ---------------------------------------------------------------------------
# Operators and directions
# ---------------------------------------------------------------------------

def forward(x: np.ndarray, setting) -> np.ndarray:
    return np.asarray(apply_forward("kwave", x, setting), dtype=float)


def adjoint(y: np.ndarray, setting) -> np.ndarray:
    return np.asarray(apply_adjoint("kwave", y, setting), dtype=float)


def physical_loss(residual: np.ndarray) -> float:
    return float(0.5 * np.sum(residual ** 2))


def relative_error(x: np.ndarray, x_true: np.ndarray) -> float:
    return float(np.linalg.norm(x - x_true) / np.linalg.norm(x_true))


def cosine(u: np.ndarray, v: np.ndarray) -> float:
    nu, nv = np.linalg.norm(u), np.linalg.norm(v)
    if nu == 0.0 or nv == 0.0:
        return float("nan")
    return float(np.sum(u * v) / (nu * nv))


def surrogate_gradient(fno, x: np.ndarray, b: np.ndarray):
    """Gradient of 0.5||F(x) - b||^2 w.r.t. the image, through the frozen FNO.

    Returns (gradient, surrogate loss). Mirrors the autograd call in
    heldout_gradient_comparison.py; model weights are not updated.
    """
    xt = torch.tensor(x, requires_grad=True)
    y = fno(xt)
    surrogate_residual = y.detach().cpu().numpy().astype(float) - b
    grad = torch.autograd.grad(y, xt, torch.as_tensor(surrogate_residual, dtype=y.dtype))[0]
    return grad.detach().cpu().numpy().astype(float), physical_loss(surrogate_residual)


def all_directions(x, residual, b, setting, fnos, residual_ops):
    """All directions at the same iterate x, with the wall-clock time of each.

    fnos: name -> frozen forward FNO (direction J_F^T (F(x) - b)).
    residual_ops: name -> callable applied to the k-Wave residual (learned / Fourier adjoint).
    """
    timings = {}
    start = time.perf_counter()
    directions = {"reference": adjoint(residual, setting)}
    timings["reference"] = time.perf_counter() - start
    surrogate_losses = {}
    for name, fno in fnos.items():
        start = time.perf_counter()
        directions[name], surrogate_losses[name] = surrogate_gradient(fno, x, b)
        timings[name] = time.perf_counter() - start
    for name, op in residual_ops.items():
        start = time.perf_counter()
        directions[name] = op(residual)
        timings[name] = time.perf_counter() - start
    return directions, surrogate_losses, timings


def line_search(residual: np.ndarray, a_g: np.ndarray):
    """Exact nonnegative minimiser of 0.5||r - alpha A g||^2 for linear A.

    Returns (alpha, status). status is 'ok', 'zero_step' (direction not a descent
    direction for the physical loss) or 'zero_denominator'.
    """
    denominator = float(np.sum(a_g ** 2))
    if denominator == 0.0:
        return 0.0, "zero_denominator"
    alpha = float(np.sum(residual * a_g)) / denominator
    if alpha <= 0.0:
        return 0.0, "zero_step"
    return alpha, "ok"


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------

def initial_image(kind: str, b: np.ndarray, setting) -> tuple[np.ndarray, dict]:
    """Zero image, or the line-search-scaled backprojection alpha * B(b).

    The scaled backprojection equals one reference step from the zero image,
    because the reference direction at x = 0 is B(-b).
    """
    if kind == "zero":
        return np.zeros(IMAGE_SHAPE), {"init_scale": 0.0}
    backprojection = adjoint(b, setting)
    scale, status = line_search(b, forward(backprojection, setting))
    print(f"  backprojection scale = {scale:.6e} ({status})")
    return scale * backprojection, {"init_scale": scale}


# ---------------------------------------------------------------------------
# Reconstruction loop
# ---------------------------------------------------------------------------

def run_trajectory(method, x0, x_true, b, setting, fnos, residual_ops, iterations, init_kind):
    """Gradient descent following `method`'s direction; logs all directions' alignment."""
    x = x0.copy()
    others = [*fnos, *residual_ops]
    base = method.removesuffix("+fallback")
    # Cumulative k-Wave calls an efficient implementation would need: one forward per step
    # for the line search (the next residual is r - alpha A g by linearity), one adjoint per
    # reference step, plus one forward and one adjoint per fallback step.
    n_fwd = n_adj = n_fallback = 0
    rows, images = [], [x.copy()]
    for k in range(iterations + 1):
        start = time.perf_counter()
        residual = forward(x, setting) - b
        t_forward = time.perf_counter() - start
        directions, surrogate_losses, timings = all_directions(
            x, residual, b, setting, fnos, residual_ops)
        row = {
            "init": init_kind,
            "trajectory": method,
            "iteration": k,
            "physical_loss": physical_loss(residual),
            "image_rel_l2": relative_error(x, x_true),
            "t_forward": t_forward,
        }
        for name in others:
            row[f"cos_{name}_vs_reference"] = cosine(directions[name], directions["reference"])
        for name in fnos:
            row[f"surrogate_loss_{name}"] = surrogate_losses[name]
        row.update({f"t_{name}": t for name, t in timings.items()})

        row.update(n_kwave_forward=n_fwd, n_kwave_adjoint=n_adj, n_fallback=n_fallback)
        if k < iterations:  # last row only records the final state
            g = directions[base]
            alpha, status = line_search(residual, forward(g, setting))
            n_fwd += 1
            n_adj += base == "reference"
            if method.endswith("+fallback") and status != "ok":
                g = directions["reference"]
                alpha, status = line_search(residual, forward(g, setting))
                n_fwd, n_adj, n_fallback = n_fwd + 1, n_adj + 1, n_fallback + 1
                status = f"fallback_{status}"
            row.update(alpha=alpha, step_status=status)
            if status not in ("ok", "fallback_ok"):
                print(f"    [{init_kind}/{method}] iteration {k}: {status}, x unchanged")
            x = x - alpha * g
            images.append(x.copy())
        else:
            row.update(alpha=float("nan"), step_status="final")
        row["seconds"] = time.perf_counter() - start
        rows.append(row)
        print(f"    [{init_kind}/{method}] k={k:2d} loss={row['physical_loss']:.4e} "
              f"img_err={row['image_rel_l2']:.4f} "
              + " ".join(f"cos_{n}={row[f'cos_{n}_vs_reference']:.3f}" for n in others),
              flush=True)
    return rows, np.stack(images)


# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------

def write_rows(path: Path, rows: list[dict]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def plot_curves(path: Path, rows: list[dict], methods, other_names) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available; skipping plot")
        return
    inits = sorted({r["init"] for r in rows})
    fig, axes = plt.subplots(len(inits), 3, figsize=(13, 3.6 * len(inits)), squeeze=False)
    for i, init in enumerate(inits):
        for method in methods:
            sel = [r for r in rows if r["init"] == init and r["trajectory"] == method]
            ks = [r["iteration"] for r in sel]
            axes[i, 0].semilogy(ks, [r["physical_loss"] for r in sel], marker="o", label=method)
            axes[i, 1].plot(ks, [r["image_rel_l2"] for r in sel], marker="o", label=method)
        ref = [r for r in rows if r["init"] == init and r["trajectory"] == "reference"]
        for name in other_names:
            axes[i, 2].plot([r["iteration"] for r in ref],
                            [r[f"cos_{name}_vs_reference"] for r in ref], marker="o",
                            label=f"{name} (on reference trajectory)")
        axes[i, 0].set_title(f"{init} init: physical loss")
        axes[i, 1].set_title(f"{init} init: relative L2 image error")
        axes[i, 2].set_title(f"{init} init: cosine vs reference")
        axes[i, 2].set_ylim(-1, 1)
        for ax in axes[i]:
            ax.set_xlabel("iteration")
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    setting = make_setting(CONDITION)

    if not args.no_fno and args.checkpoint_archive is None:
        raise SystemExit("--checkpoint-archive is required unless --no-fno is given")
    print("Iterative reconstruction comparison")
    fnos = {} if args.no_fno else {
        name: surrogate(read_checkpoint(args.checkpoint_archive, name)[0]) for name in MEMBERS}
    residual_ops = {}
    if not args.no_fourier:
        residual_ops["fourier_c"] = lambda r: np.asarray(
            apply_adjoint("fourier_np_c", r, setting), dtype=float)
    for path in args.learned:
        model, _ = load_model(path)
        name = f"G_{path.parent.name}_{path.stem}" if len(args.learned) > 1 else "G"
        residual_ops[name] = lambda r, m=model: predict(m, r[None])[0]
    methods = ("reference", *fnos, *residual_ops)
    print(f"Directions: {', '.join(methods)}")
    if args.fallback:
        learned = [n for n in residual_ops if n.startswith("G")]
        methods = (*methods, *(f"{n}+fallback" for n in learned))
    if args.trajectories:
        unknown = set(args.trajectories) - set(methods)
        if unknown:
            raise SystemExit(f"unknown trajectories {sorted(unknown)}; choose from {methods}")
        methods = tuple(m for m in methods if m in args.trajectories)
    print(f"Trajectories: {', '.join(methods)}")

    inits = ("zero", "backprojection") if args.init == "both" else (args.init,)
    for sample_index in args.sample_index:
        x_true, b, meta = load_sample(args.samples, sample_index, setting, args.use_stored_b)
        rows, images = [], {}
        for init_kind in inits:
            print(f"\nSample {sample_index}, initialisation: {init_kind}")
            x0, init_meta = initial_image(init_kind, b, setting)
            for method in methods:
                trajectory_rows, trajectory_images = run_trajectory(
                    method, x0, x_true, b, setting, fnos, residual_ops, args.iterations, init_kind)
                for row in trajectory_rows:
                    row.update(sample_index=sample_index, **meta, **init_meta)
                rows.extend(trajectory_rows)
                images[f"{init_kind}__{method}"] = trajectory_images

        stem = f"sample{sample_index}_iters{args.iterations}"
        write_rows(args.output_dir / f"{stem}_metrics.csv", rows)
        np.savez_compressed(args.output_dir / f"{stem}_iterates.npz", x_true=x_true, b=b, **images)
        plot_curves(args.output_dir / f"{stem}_curves.png", rows, methods,
                    [*fnos, *residual_ops])
        print(f"\nSaved metrics, iterates and plot for sample {sample_index} to {args.output_dir}")


if __name__ == "__main__":
    main()