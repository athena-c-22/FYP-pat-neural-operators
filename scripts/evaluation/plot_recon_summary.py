"""Report figures and tables for the learned-adjoint reconstruction study.

Reads
    --recon   per-sample *_metrics.csv from iterative_reconstruction_comparison.py
    --timing  timing.csv from time_reconstruction.py (optional)
    runs/, runs_1k/, runs_4k_e300/ training logs (data-scaling figure)

Writes to --out
    fig_error_vs_iteration.png   median image error (IQR band) per method, one panel per start
    fig_error_vs_time.png        median image error against measured wall-clock time
    fig_cosine_paths.png         cos(G r, A* r) on the reference path vs on G's own path
    fig_data_scaling.png         clean test error vs number of training pairs
    summary_table.csv            per method/start: error at k = 5, 10, 20, fallbacks, adjoint calls

Colours: fixed categorical order (validated palette, light mode). Two slots are below 3:1
contrast on white, so every series also has a direct label and its own marker.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]  # slots 1-4, fixed order
MARKERS = ["o", "s", "^", "D"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
STARTS = {"zero": "Zero start", "backprojection": "Backprojection start"}


def pretty(name: str) -> str:
    if name == "reference":
        return "k-Wave adjoint (reference)"
    m = re.search(r"G_runs_resid_([A-Z])_", name) or re.match(r"([A-Z])(?:\+|$)", name)
    label = f"Learned G ({m.group(1)})" if m else name
    return label + (" + fallback" if name.endswith("+fallback") else "")


def style(ax, xlabel, ylabel):
    ax.set_xlabel(xlabel, color=INK2)
    ax.set_ylabel(ylabel, color=INK2)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK2)
    ax.tick_params(colors=INK2)


def label_end(ax, x, y, text, color_index):
    """Mark a series end; the text is placed later by place_labels (collision-free)."""
    ax.plot([x], [y], marker=MARKERS[color_index], color=SERIES[color_index], ms=6,
            markeredgecolor="white", markeredgewidth=1.2, zorder=5)
    ax._end_labels = getattr(ax, "_end_labels", []) + [(x, y, text)]


def place_labels(ax, min_gap_frac=0.045):
    """Write end labels, nudged apart vertically so that none overlap."""
    labels = sorted(getattr(ax, "_end_labels", []), key=lambda v: v[1])
    lo, hi = ax.get_ylim()
    gap, placed = min_gap_frac * (hi - lo), []
    shift = matplotlib.transforms.ScaledTranslation(6 / 72, 0, ax.figure.dpi_scale_trans)
    for x, y, text in labels:
        ty = max(y, placed[-1] + gap) if placed else y
        placed.append(ty)
        ax.text(x, ty, text, va="center", fontsize=8, color=INK, transform=ax.transData + shift)


def load_recon(folder: Path):
    rows = []
    for f in sorted(folder.glob("sample*_metrics.csv")):
        rows += list(csv.DictReader(f.open()))
    return rows


def curves(rows, init, method, key="image_rel_l2"):
    """Array (n_samples, n_iterations) of `key` along `method`'s trajectory."""
    samples = sorted({int(r["sample_index"]) for r in rows})
    out = []
    for s in samples:
        sel = sorted((r for r in rows if r["init"] == init and r["trajectory"] == method
                      and int(r["sample_index"]) == s), key=lambda r: int(r["iteration"]))
        if sel:
            out.append([float(r[key]) for r in sel])
    return np.array(out)


def fig_error_vs_iteration(rows, methods, path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, (init, title) in zip(axes, STARTS.items()):
        for i, m in enumerate(methods):
            c = curves(rows, init, m)
            k = np.arange(c.shape[1])
            med, lo, hi = np.median(c, 0), np.percentile(c, 25, 0), np.percentile(c, 75, 0)
            ax.fill_between(k, lo, hi, color=SERIES[i], alpha=0.12, linewidth=0)
            ax.plot(k, med, color=SERIES[i], lw=2, marker=MARKERS[i], ms=4, markevery=5,
                    label=pretty(m))
            label_end(ax, k[-1], med[-1], f"{med[-1]:.3f}", i)
        style(ax, "Iteration", "Relative L2 image error")
        ax.set_title(f"{title} (median, IQR band, n = {c.shape[0]})", color=INK, fontsize=10)
        ax.set_xlim(0, k[-1] + 3)
    axes[0].legend(frameon=False, fontsize=8, loc="upper right")
    for ax in axes:
        place_labels(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def fig_error_vs_time(timing, path):
    methods = list(dict.fromkeys(r["method"] for r in timing))
    if len(methods) > len(SERIES):  # drop the extra plain-G series, keep reference first
        plain = [m for m in methods[1:] if not m.endswith("+fallback")]
        methods = [m for m in methods if m not in plain[1:]][:len(SERIES)]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, (init, title) in zip(axes, STARTS.items()):
        for i, m in enumerate(methods):
            sel = [r for r in timing if r["init"] == init and r["method"] == m]
            if not sel:
                continue
            its = sorted({int(r["iteration"]) for r in sel})
            t = [np.median([float(r["seconds"]) for r in sel if int(r["iteration"]) == k]) for k in its]
            e = [np.median([float(r["image_rel_l2"]) for r in sel if int(r["iteration"]) == k]) for k in its]
            ax.plot(t, e, color=SERIES[i], lw=2, marker=MARKERS[i], ms=4, markevery=5, label=pretty(m))
            label_end(ax, t[-1], e[-1], f"{e[-1]:.3f} @ {t[-1]:.0f}s", i)
        style(ax, "Measured wall-clock time (s)", "Relative L2 image error")
        n = len({r["sample"] for r in timing})
        ax.set_title(f"{title} (median over {n} images)", color=INK, fontsize=10)
    axes[0].legend(frameon=False, fontsize=8, loc="upper right")
    for ax in axes:
        ax.set_xlim(right=ax.get_xlim()[1] * 1.12)
        place_labels(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def fig_cosine_paths(rows, learned, path):
    col = f"cos_{learned}_vs_reference"
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0), sharey=True)
    for ax, (init, title) in zip(axes, STARTS.items()):
        for i, (traj, lab) in enumerate([("reference", "on the reference path"),
                                         (learned, "on G's own path")]):
            c = curves(rows, init, traj, col)
            k = np.arange(c.shape[1])
            med = np.median(c, 0)
            ax.fill_between(k, np.percentile(c, 25, 0), np.percentile(c, 75, 0),
                            color=SERIES[i], alpha=0.12, linewidth=0)
            ax.plot(k, med, color=SERIES[i], lw=2, marker=MARKERS[i], ms=4, markevery=5,
                    label=f"cos(G r, A* r) {lab}")
            label_end(ax, k[-1], med[-1], f"{med[-1]:.2f}", i)
        style(ax, "Iteration", "Cosine with the k-Wave adjoint direction")
        ax.set_ylim(-0.1, 1.05)
        ax.set_xlim(0, k[-1] + 3)
        ax.set_title(f"{title}: {pretty(learned)}", color=INK, fontsize=10)
    axes[0].legend(frameon=False, fontsize=8, loc="center right", bbox_to_anchor=(1.0, 0.42))
    for ax in axes:
        place_labels(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def fig_data_scaling(root: Path, path):
    points = []
    for folder in ("runs", "runs_1k", "runs_4k_e300"):
        logs = [json.loads(f.read_text()) for f in sorted((root / folder).glob("lam0_seed*.json"))]
        if logs:
            points.append((logs[0]["n"]["train"], [l["test_rel_l2"] for l in logs],
                           logs[0]["history"][-1]["epoch"] + 1))
    fig, ax = plt.subplots(figsize=(5.5, 4))
    n = np.array([p[0] for p in points])
    mean = np.array([np.mean(p[1]) for p in points])
    sd = np.array([np.std(p[1], ddof=1) for p in points])
    ax.errorbar(n, mean, yerr=sd, color=SERIES[0], lw=2, marker="o", ms=6, capsize=3,
                markeredgecolor="white")
    for x, y, (_, _, ep) in zip(n, mean, points):
        ax.annotate(f"{y:.3f}\n({ep} epochs)", (x, y), xytext=(8, 4), textcoords="offset points",
                    fontsize=8, color=INK)
    ax.set_xscale("log")
    ax.set_xticks(n, [str(v) for v in n])
    ax.minorticks_off()
    ax.set_xlim(n[0] / 1.6, n[-1] * 2.2)
    ax.set_ylim(0, max(mean) * 1.2)
    style(ax, "Training pairs (log scale)", "Test relative L2 vs k-Wave adjoint")
    ax.set_title("Clean-data accuracy vs training set size (mean ± SD, 3 seeds)",
                 color=INK, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def summary_table(rows, methods, path):
    out = []
    for init in STARTS:
        for m in methods:
            c = curves(rows, init, m)
            adj = curves(rows, init, m, "n_kwave_adjoint") if "n_kwave_adjoint" in rows[0] else None
            fb = curves(rows, init, m, "n_fallback") if "n_fallback" in rows[0] else None
            row = {"init": init, "method": pretty(m), "n_images": c.shape[0]}
            for k in (5, 10, 20):
                row[f"median_err_k{k}"] = round(float(np.median(c[:, k])), 4)
                row[f"iqr_err_k{k}"] = f"{np.percentile(c[:, k], 25):.3f}-{np.percentile(c[:, k], 75):.3f}"
            if adj is not None:
                row["mean_adjoint_calls_20"] = round(float(adj[:, -1].mean()), 2)
                row["mean_fallbacks_20"] = round(float(fb[:, -1].mean()), 2)
            out.append(row)
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--recon", type=Path, default=Path("outputs/learned_adjoint/recon_test20"))
    ap.add_argument("--timing", type=Path, default=Path("outputs/learned_adjoint/timing/timing.csv"))
    ap.add_argument("--runs-root", type=Path, default=Path("outputs/learned_adjoint"))
    ap.add_argument("--out", type=Path, default=Path("outputs/learned_adjoint/report_figures"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    rows = load_recon(args.recon)
    methods = list(dict.fromkeys(r["trajectory"] for r in rows))[:4]
    plain = [m for m in methods if m.startswith("G") and not m.endswith("+fallback")]
    fig_error_vs_iteration(rows, methods, args.out / "fig_error_vs_iteration.png")
    if plain:
        fig_cosine_paths(rows, plain[0], args.out / "fig_cosine_paths.png")
    fig_data_scaling(args.runs_root, args.out / "fig_data_scaling.png")
    if args.timing.exists():
        fig_error_vs_time(list(csv.DictReader(args.timing.open())), args.out / "fig_error_vs_time.png")
    for r in summary_table(rows, methods, args.out / "summary_table.csv"):
        print(r)
    print("wrote figures to", args.out)


if __name__ == "__main__":
    main()
