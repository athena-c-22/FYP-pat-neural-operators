"""Time and evaluate stratified held-out gradients, with optional small descent probes."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from pat_fno.evaluation.gradient_comparison import gradient_metrics
from scripts.evaluation.adjoint_consistency import apply_adjoint, apply_forward, make_setting
from scripts.evaluation.forward_checkpoint_comparison import (
    MEMBERS,
    SEED,
    positive_step,
    read_checkpoint,
    surrogate,
    write_csv,
)
from scripts.evaluation.gradient_comparison import POINTS, make_point

KINDS = ("half", "smooth_pert", "noisy")


def select_images(root):
    """Select two examples per digit, excluding the original four test positions."""
    candidates = []
    position = 0
    for path in sorted(root.glob("test_*.h5")):
        with h5py.File(path) as f:
            for offset, (label, index) in enumerate(
                zip(f["label"][:], f["source_index"][:], strict=True)
            ):
                if position + offset >= 4:
                    candidates.append(
                        {
                            "test_position": position + offset,
                            "source_index": int(index),
                            "label": int(label),
                            "shard": path.name,
                            "shard_offset": offset,
                        }
                    )
            position += len(f["p0"])
    rng = np.random.default_rng(SEED)
    selected = []
    for label in range(10):
        pool = [r for r in candidates if r["label"] == label]
        selected.extend(pool[i] for i in rng.choice(len(pool), 2, replace=False))
    return selected


def gradient_job(task):
    """Use six fresh k-Wave calls per image and no finite-difference repetitions."""
    root, archive, item = task
    torch.set_num_threads(1)
    started = time.monotonic()
    with h5py.File(root / item["shard"]) as f:
        p0 = f["p0"][item["shard_offset"]].astype(float)
        b = f["kwave_forward"][item["shard_offset"]].astype(float)
    s = make_setting("periodic_theta89")
    maps = {name: surrogate(read_checkpoint(archive, name)[0]) for name in MEMBERS}
    rows, arrays = [], {"p0": p0, "b": b}
    for kind in KINDS:
        rng = np.random.default_rng(SEED + 100 * item["test_position"] + POINTS.index(kind))
        x = make_point(kind, p0, rng)
        residual = apply_forward("kwave", x, s).astype(float) - b
        ref = apply_adjoint("kwave", residual, s).astype(float)
        arrays.update({f"{kind}_x": x, f"{kind}_r": residual, f"{kind}_reference": ref})
        for name, g in maps.items():
            xt = torch.tensor(x, requires_grad=True)
            y = g(xt)
            rs = y.detach().numpy() - b
            ad = torch.autograd.grad(y, xt, torch.tensor(rs), retain_graph=True)[0].numpy()
            jac = torch.autograd.grad(y, xt, torch.tensor(residual))[0].numpy()
            arrays[f"{kind}_{name}"] = ad
            arrays[f"{kind}_{name}_same_residual"] = jac
            row = {
                **item,
                "point": kind,
                "model": name,
                "J_A": 0.5 * float(np.sum(residual**2)),
                "norm_ref": float(np.linalg.norm(ref)),
                "norm_ad": float(np.linalg.norm(ad)),
            }
            row.update({f"ad_{k}": v for k, v in gradient_metrics(ad, ref).items()})
            row.update({f"same_residual_{k}": v for k, v in gradient_metrics(jac, ref).items()})
            rows.append(row)
    return item, rows, arrays, time.monotonic() - started


def step_job(task):
    """Probe one full nonnegative line-minimum step using saved gradients."""
    directory, item = task
    torch.set_num_threads(1)
    s = make_setting("periodic_theta89")
    rows = []
    with np.load(directory / f"test{item['test_position']}.npz") as data:
        for kind in KINDS:
            x, r, b = data[f"{kind}_x"], data[f"{kind}_r"], data["b"]
            j0 = 0.5 * float(np.sum(r**2))
            for name in ("reference", *MEMBERS):
                grad = data[f"{kind}_{name}"]
                ag = apply_forward("kwave", grad, s).astype(float)
                alpha, predicted, dot = positive_step(r, ag)
                after = (
                    0.5
                    * float(
                        np.sum((apply_forward("kwave", x - alpha * grad, s).astype(float) - b) ** 2)
                    )
                    if alpha
                    else j0
                )
                rows.append(
                    {
                        **item,
                        "point": kind,
                        "model": name,
                        "initial_objective": j0,
                        "after_objective": after,
                        "alpha": alpha,
                        "linear_prediction": predicted,
                        "relative_reduction": (j0 - after) / j0,
                        "is_descent": dot > 0,
                        "improved": after < j0,
                        "worsened": after > j0,
                    }
                )
    return rows


def summarise(rows, out):
    """Aggregate within point and model, preserving per-image rows separately."""
    result = []
    for kind in KINDS:
        for name in MEMBERS:
            rr = [r for r in rows if r["point"] == kind and r["model"] == name]
            for metric in (
                "ad_cos",
                "ad_rel_l2",
                "ad_norm_ratio",
                "same_residual_cos",
                "same_residual_rel_l2",
                "same_residual_norm_ratio",
            ):
                v = np.array([float(r[metric]) for r in rr])
                result.append(
                    {
                        "point": kind,
                        "model": name,
                        "metric": metric,
                        "n": len(v),
                        "mean": v.mean(),
                        "sample_sd": v.std(ddof=1),
                        "min": v.min(),
                        "max": v.max(),
                    }
                )
    write_csv(out / "summary.csv", result)


def plot_results(out):
    """Plot paired gradient metrics and save paired improvement counts."""
    with (out / "gradients.csv").open() as f:
        rows = list(csv.DictReader(f))
    fig, axes = plt.subplots(3, 3, figsize=(12, 10), constrained_layout=True)
    paired = []
    for i, kind in enumerate(KINDS):
        for j, metric in enumerate(("ad_cos", "ad_rel_l2", "ad_norm_ratio")):
            first = {
                int(r["test_position"]): float(r[metric])
                for r in rows
                if r["point"] == kind and r["model"] == "F0"
            }
            second = {
                int(r["test_position"]): float(r[metric])
                for r in rows
                if r["point"] == kind and r["model"] == "F5"
            }
            a, b = np.array(list(first.values())), np.array([second[k] for k in first])
            ax = axes[i, j]
            for va, vb in zip(a, b, strict=True):
                ax.plot([0, 1], [va, vb], "o-", alpha=0.45, markersize=3)
            ax.set_xticks([0, 1], ["F0", "F5"])
            ax.set_title(f"{kind}: {metric}")
            ax.axhline(0 if metric == "ad_cos" else 1, color="gray", linestyle="--")
            improved = b > a if metric == "ad_cos" else b < a
            if metric == "ad_norm_ratio":
                improved = np.abs(b - 1) < np.abs(a - 1)
            paired.append(
                {
                    "point": kind,
                    "metric": metric,
                    "n": len(a),
                    "mean_F5_minus_F0": float(np.mean(b - a)),
                    "F5_improved_count": int(improved.sum()),
                    "improvement_definition": "closer to one"
                    if metric == "ad_norm_ratio"
                    else "higher"
                    if metric == "ad_cos"
                    else "lower",
                }
            )
    fig.savefig(out / "gradient_metrics.png", dpi=150)
    plt.close(fig)
    write_csv(out / "paired_gradients.csv", paired)


def summarise_steps(out):
    """Aggregate the smaller actual-objective descent experiment separately."""
    with (out / "steps.csv").open() as f:
        rows = list(csv.DictReader(f))
    summary = []
    for kind in KINDS:
        for name in ("reference", *MEMBERS):
            selected = [r for r in rows if r["point"] == kind and r["model"] == name]
            values = np.array([float(r["relative_reduction"]) for r in selected])
            summary.append(
                {
                    "point": kind,
                    "model": name,
                    "n": len(values),
                    "mean_initial_objective": np.mean(
                        [float(r["initial_objective"]) for r in selected]
                    ),
                    "mean_after_objective": np.mean(
                        [float(r["after_objective"]) for r in selected]
                    ),
                    "mean_relative_reduction": values.mean(),
                    "sample_sd_relative_reduction": values.std(ddof=1),
                    "min_relative_reduction": values.min(),
                    "max_relative_reduction": values.max(),
                    "descent_count": sum(r["is_descent"] == "True" for r in selected),
                    "improved_count": sum(r["improved"] == "True" for r in selected),
                    "worsened_count": sum(r["worsened"] == "True" for r in selected),
                }
            )
    write_csv(out / "steps_summary.csv", summary)


def main():
    """Run three timed pilot images or all twenty, reusing pilot results when supplied."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset-dir", type=Path, required=True)
    ap.add_argument("--checkpoint-archive", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--reuse-pilot", type=Path)
    ap.add_argument("--descent-images", type=int, choices=(0, 3), default=0)
    ap.add_argument("--workers", type=int, default=3)
    args = ap.parse_args()
    selected = select_images(args.dataset_dir)
    pilot = [selected[i] for i in (0, 2, 4)]
    evaluate = pilot if args.pilot else selected
    provenance = {}
    for name in MEMBERS:
        ck, raw = read_checkpoint(args.checkpoint_archive, name)
        surrogate(ck)
        provenance[name] = hashlib.sha256(raw).hexdigest()
    rows, cached = [], set()
    if args.reuse_pilot:
        oldmeta = json.loads((args.reuse_pilot / "meta.json").read_text())
        if oldmeta["checkpoint_sha256"] != provenance or oldmeta["selection_20"] != selected:
            raise ValueError("Pilot checkpoint or selection mismatch")
        with (args.reuse_pilot / "gradients.csv").open() as f:
            rows = list(csv.DictReader(f))
        cached = {int(r["test_position"]) for r in rows}
        if cached != {r["test_position"] for r in pilot}:
            raise ValueError("Incomplete pilot")
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=False)
    (out / "selection.json").write_text(json.dumps(evaluate, indent=2))
    if cached:
        for position in cached:
            (out / f"test{position}.npz").write_bytes(
                (args.reuse_pilot / f"test{position}.npz").read_bytes()
            )
    started = time.monotonic()
    timings = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [
            pool.submit(gradient_job, (args.dataset_dir, args.checkpoint_archive, item))
            for item in evaluate
            if item["test_position"] not in cached
        ]
        for future in as_completed(futures):
            item, rr, arrays, elapsed = future.result()
            rows.extend(rr)
            timings.append(elapsed)
            np.savez_compressed(out / f"test{item['test_position']}.npz", **arrays)
            write_csv(out / "gradients.csv", rows)
            print(
                f"Gradient image {item['test_position']} digit {item['label']}: {elapsed:.1f}s; "
                f"wall {time.monotonic() - started:.1f}s",
                flush=True,
            )
    gradient_wall = time.monotonic() - started
    summarise(rows, out)
    plot_results(out)
    meta = {
        "selection_20": selected,
        "evaluated_selection": evaluate,
        "checkpoint_sha256": provenance,
        "seed": SEED,
        "workers": args.workers,
        "dataset_dir": str(args.dataset_dir.resolve()),
        "pilot_reused": str(args.reuse_pilot),
        "cached_images": len(cached),
        "new_image_seconds": timings,
        "gradient_wall_s": gradient_wall,
        "estimated_20_image_gradient_wall_s": gradient_wall * 20 / len(evaluate)
        if args.pilot
        else None,
        "reference": "k-Wave forward / approximate adjoint; physical Euclidean objective",
        "finite_difference_checks_repeated": False,
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    if args.descent_images:
        steps = []
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for rr in pool.map(step_job, [(out, item) for item in pilot]):
                steps.extend(rr)
                write_csv(out / "steps.csv", steps)
        meta["descent_image_positions"] = [r["test_position"] for r in pilot]
        summarise_steps(out)
    meta["total_runtime_s"] = time.monotonic() - started
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(
        json.dumps(
            {
                "evaluated_images": len(evaluate),
                "cached_images": len(cached),
                "gradient_wall_s": gradient_wall,
                "estimated_20_image_gradient_wall_s": meta["estimated_20_image_gradient_wall_s"],
                "total_runtime_s": meta["total_runtime_s"],
                "metadata": str(out / "meta.json"),
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
