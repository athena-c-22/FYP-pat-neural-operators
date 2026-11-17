"""Evaluate learned adjoints against the k-Wave reference adjoint and the Fourier adjoint.

(a) Accuracy on the held-out synthetic test split: relative L2 and cosine against ``z = A* y``.
(b) Dot-product test on the same split (needs no extra k-Wave): for pairs ``i != j`` compare
    ``<y_i, y_j> = <A x_i, y_j>`` with ``<x_i, B y_j>`` for ``B`` in {learned, k-Wave A*,
    Fourier adjoint with the c convention}, using the Stage 2 metrics.
(c) Gradient test at the 16 Stage 3 points (bundled MNIST samples = out-of-distribution for a
    model trained on procedural phantoms): ``g_L = G(r_true)`` against the Stage 3 reference
    ``g_ref = A*(A x - b)``, and (seed 0 of each variant only, ``--linesearch``) the exact
    line-search decrease of ``J_A`` along ``-g_L``.

Needs ``outputs/learned_adjoint/runs/*.pt`` (train_learned_adjoint.py) and
``outputs/gradient_comparison/{gradients.npz,results.csv}`` (Stage 3).
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from pat_fno.config.mnist_pat import SOUND_SPEED
from pat_fno.evaluation.adjoint_consistency import adjoint_metrics
from pat_fno.evaluation.gradient_comparison import gradient_metrics, quadratic_step_decrease
from pat_fno.models.learned_adjoint import LearnedAdjoint
from pat_fno.operators.fourier.numpy_reference import numpy_adjoint_2d
from pat_fno.operators.kwave import kwave_forward_2d
from scripts.evaluation.adjoint_consistency import make_setting
from scripts.train.train_learned_adjoint import load_data

C = SOUND_SPEED
POINTS = ("truth", "half", "smooth_pert", "noisy")


def load_model(path: Path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    a = ck["arguments"]
    m = LearnedAdjoint(a["modes"], a["width"], a["layers"])
    m.load_state_dict(ck["model"])
    return m.eval(), ck


def predict(model, y: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        return model(torch.from_numpy(np.asarray(y, np.float32))).numpy().astype(np.float64)


def rel_cos(pred, ref):
    p, r = pred.reshape(len(pred), -1), ref.reshape(len(ref), -1)
    rel = np.linalg.norm(p - r, axis=1) / np.linalg.norm(r, axis=1)
    cos = (p * r).sum(1) / (np.linalg.norm(p, axis=1) * np.linalg.norm(r, axis=1) + 1e-30)
    return rel, cos


def dot_test(x, y, bty_all) -> dict[str, float]:
    """Medians of the Stage 2 metrics over all ordered pairs i != j."""
    rows = [
        adjoint_metrics(y[i], y[j], x[i], bty_all[j])
        for i in range(len(x))
        for j in range(len(x))
        if i != j
    ]
    return {
        k: float(np.median([r[k] for r in rows])) for k in ("sym_rel", "cs_rel", "ratio", "marta")
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--runs", type=Path, default=Path("outputs/learned_adjoint/runs"))
    ap.add_argument("--data", type=Path, default=Path("outputs/learned_adjoint/data"))
    ap.add_argument("--stage3", type=Path, default=Path("outputs/gradient_comparison"))
    ap.add_argument("--out", type=Path, default=Path("outputs/learned_adjoint"))
    ap.add_argument("--linesearch", action="store_true", help="k-Wave forwards (~5 s each)")
    args = ap.parse_args()

    d = load_data(args.data)
    s = make_setting("periodic_theta89")
    ckpts = sorted(args.runs.glob("*.pt"))
    models = {p.stem: load_model(p) for p in ckpts}
    test = np.array(next(iter(models.values()))[1]["split"]["test"])
    x, y, z = d["x"][test], d["y"][test], d["z"][test]
    rows: list[dict] = []

    # ---------------------------------------------------------------- (a) + (b)
    f_pred = np.stack([numpy_adjoint_2d(yi.astype(np.float64) / C, s) for yi in y])
    alpha = float((f_pred * z).sum() / (f_pred * f_pred).sum())
    preds = {"kwave_reference": z.astype(np.float64), "fourier_c": f_pred}
    preds.update({n: predict(m, y) for n, (m, _) in models.items()})
    for name, p in preds.items():
        rel, cos = rel_cos(p, z)
        row = {
            "part": "test_split",
            "model": name,
            "rel_l2_mean": rel.mean(),
            "cos_mean": cos.mean(),
        }
        row.update({f"dot_{k}": v for k, v in dot_test(x, y, p).items()})
        if name == "fourier_c":
            relf, _ = rel_cos(alpha * p, z)
            row.update({"best_scalar": alpha, "rel_l2_after_scalar": relf.mean()})
        rows.append(row)
    zero_rel = 1.0  # predicting zero

    # ---------------------------------------------------------------- (c)
    g3 = np.load(args.stage3 / "gradients.npz")
    s3 = {
        (int(r["sample"]), r["point"]): r
        for r in csv.DictReader((args.stage3 / "results.csv").open())
    }
    keys = [(si, p) for si in range(4) for p in POINTS]
    r_true = np.stack([g3[f"s{si}_{p}__r_true"] for si, p in keys])
    g_ref = np.stack([g3[f"s{si}_{p}__g_ref"] for si, p in keys])
    cands = {"fourier_c": np.stack([numpy_adjoint_2d(r / C, s) for r in r_true])}
    cands.update({n: predict(m, r_true) for n, (m, _) in models.items()})
    zero_in = {
        n: float(np.linalg.norm(predict(m, np.zeros((1, 64, 91)))[0]))
        for n, (m, _) in models.items()
    }
    for name, g in cands.items():
        for point in POINTS:
            sel = [i for i, (_, p) in enumerate(keys) if p == point]
            if point == "truth":
                rows.append(
                    {
                        "part": "stage3_grad",
                        "model": name,
                        "point": point,
                        "norm_g_mean": np.mean([np.linalg.norm(g[i]) for i in sel]),
                    }
                )
                continue
            ms = [gradient_metrics(g[i], g_ref[i]) for i in sel]
            rows.append(
                {
                    "part": "stage3_grad",
                    "model": name,
                    "point": point,
                    "cos_mean": np.mean([m["cos"] for m in ms]),
                    "rel_l2_mean": np.mean([m["rel_l2"] for m in ms]),
                    "cos_min": np.min([m["cos"] for m in ms]),
                    "cos_max": np.max([m["cos"] for m in ms]),
                }
            )
    for point in POINTS[1:]:  # Stage 3 reference numbers (autograd through the forward FNO)
        sel = [r for (si, p), r in s3.items() if p == point]
        rows.append(
            {
                "part": "stage3_grad",
                "model": "stage3_g_ad_forward_FNO",
                "point": point,
                "cos_mean": np.mean([float(r["ad_cos"]) for r in sel]),
                "rel_l2_mean": np.mean([float(r["ad_rel_l2"]) for r in sel]),
                "cos_min": min(float(r["ad_cos"]) for r in sel),
                "cos_max": max(float(r["ad_cos"]) for r in sel),
            }
        )
    rows.append(
        {
            "part": "stage3_grad",
            "model": "stage3_g_ad_forward_FNO",
            "point": "truth",
            "norm_g_mean": np.mean(
                [float(r["norm_g_ad"]) for (si, p), r in s3.items() if p == "truth"]
            ),
        }
    )

    if args.linesearch:
        todo = {"fourier_c": cands["fourier_c"]}
        todo.update({n: cands[n] for n in cands if n.endswith("seed0")})
        for name, g in todo.items():
            fracs = []
            for i, (si, p) in enumerate(keys):
                if p == "truth":
                    continue
                t, dec = quadratic_step_decrease(
                    r_true[i], kwave_forward_2d(g[i].astype(np.float32), s)
                )
                ref = float(s3[(si, p)]["decrease_ref"])
                fracs.append((p, t, dec / ref))
            for point in POINTS[1:]:
                f = [(t, fr) for pp, t, fr in fracs if pp == point]
                rows.append(
                    {
                        "part": "stage3_linesearch",
                        "model": name,
                        "point": point,
                        "decrease_frac_mean": np.mean([fr for _, fr in f]),
                        "n_ascent": int(sum(t < 0 for t, _ in f)),
                    }
                )
                print(rows[-1], flush=True)

    # ---------------------------------------------------------------- outputs
    fields = sorted(
        {k for r in rows for k in r}, key=lambda k: (k not in ("part", "model", "point"), k)
    )
    with (args.out / "eval_results.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    summary = {
        "zero_predictor_rel_l2": zero_rel,
        "norm_G_of_zero_input": zero_in,
        "n_test": len(test),
    }
    (args.out / "eval_summary.json").write_text(json.dumps(summary, indent=2))
    fig_curves(args.runs, args.out / "training_curves.png")
    fig_example(x, z, preds, args.out / "test_example.png")
    fig_grad(rows, args.out / "gradient_cosine.png")
    print("done; wrote", args.out / "eval_results.csv")


def fig_curves(runs: Path, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    for f in sorted(runs.glob("*.json")):
        h = json.loads(f.read_text())["history"]
        lam = f.stem.split("_")[0]
        ax.semilogy(
            [e["val_rel_l2"] for e in h],
            color={"lam0": "C0", "lam1": "C1", "lam10": "C2"}.get(lam, "k"),
            alpha=0.6,
            label=lam if f.stem.endswith("seed0") else None,
        )
    ax.set_xlabel("epoch")
    ax.set_ylabel("validation rel. L2 vs A*y")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_example(x, z, preds, path: Path) -> None:
    names = ["kwave_reference", "fourier_c"] + [n for n in preds if n.endswith("seed0")]
    fig, axs = plt.subplots(2, len(names) + 1, figsize=(2.6 * (len(names) + 1), 5.4))
    i = 0
    vmax = np.abs(z[i]).max()
    axs[0, 0].imshow(x[i], cmap="gray")
    axs[0, 0].set_title("phantom x", fontsize=8)
    axs[1, 0].axis("off")
    for j, n in enumerate(names, 1):
        axs[0, j].imshow(preds[n][i], cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        axs[0, j].set_title(n, fontsize=8)
        axs[1, j].imshow(preds[n][i] - z[i], cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        axs[1, j].set_title("minus k-Wave A*y", fontsize=8)
    for a in axs.ravel():
        a.set_xticks([])
        a.set_yticks([])
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_grad(rows, path: Path) -> None:
    g = [r for r in rows if r["part"] == "stage3_grad" and r.get("point") in POINTS[1:]]
    models = sorted({r["model"] for r in g})
    fig, ax = plt.subplots(figsize=(8, 4))
    w = 0.8 / len(models)
    for k, m in enumerate(models):
        v = [
            next(r["cos_mean"] for r in g if r["model"] == m and r["point"] == p)
            for p in POINTS[1:]
        ]
        ax.bar(np.arange(3) + k * w, v, w, label=m)
    ax.set_xticks(np.arange(3) + 0.4 - w / 2, POINTS[1:])
    ax.set_ylabel("mean cosine with g_ref (Stage 3 points)")
    ax.legend(fontsize=6)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
