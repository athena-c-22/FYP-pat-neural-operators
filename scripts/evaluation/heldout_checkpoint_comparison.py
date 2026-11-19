"""Inspect original held-out shards and stream a paired forward-checkpoint evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import time
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from pat_fno.data.mnist import centered_corr, rel_l2
from scripts.evaluation.forward_checkpoint_comparison import (
    MEMBERS,
    SAMPLES,
    SEED,
    read_checkpoint,
    surrogate,
    write_csv,
)


def inspect_dataset(root, checkpoints):
    """Validate all shard headers, deterministic source order and bundled provenance."""
    manifest = json.loads((root / "manifest.json").read_text())
    expected = {
        "dataset": "mnist_large_v1",
        "condition": "periodic_theta89",
        "boundary": "periodic",
        "grid": 64,
        "Nt": 91,
        "complete": True,
    }
    for k, v in expected.items():
        if manifest.get(k) != v:
            raise ValueError(f"Manifest {k}: {manifest.get(k)!r}, expected {v!r}")
    for ck in checkpoints.values():
        surrogate(ck)  # strict architecture/state compatibility
        setting = ck["setting"]
        for k, mk in (("Nx", "grid"), ("Ny", "grid"), ("Nt", "Nt"), ("CFL", "cfl"), ("dt", "dt")):
            if not np.isclose(setting[k], manifest[mk], rtol=1e-12, atol=0):
                raise ValueError(f"Checkpoint setting differs: {k}")
        if setting["kwaveBoundary"] != manifest["boundary"]:
            raise ValueError("Checkpoint boundary differs")
        if not np.isclose(np.rad2deg(setting["computation"].theta_max), manifest["theta_deg"]):
            raise ValueError("Checkpoint angle differs")
    if checkpoints["F0"]["normalization"] != checkpoints["F5"]["normalization"]:
        raise ValueError("Checkpoint normalisations differ")
    rng = np.random.default_rng(SEED)
    train_order = rng.permutation(60000)
    test_order = rng.permutation(10000)
    schema = {
        "p0": (64, 64),
        "kwave_forward": (64, 91),
        "data_fft": (64, 91),
        "fourier_raw": (64, 91),
        "label": (),
        "source_index": (),
    }
    info = {
        "manifest": manifest,
        "manifest_sha256": hashlib.sha256((root / "manifest.json").read_bytes()).hexdigest(),
        "splits": {},
    }
    for split in ("test", "validation"):
        paths = sorted(root.glob(f"{split}_*.h5"))
        position, records, indices = 0, [], []
        for path in paths:
            match = re.fullmatch(rf"{split}_(\d+)_(\d+)\.h5", path.name)
            if match is None or int(match[1]) != position:
                raise ValueError(f"Missing/overlapping shard: {path}")
            with h5py.File(path) as f:
                attrs = dict(f.attrs)
                n = len(f["p0"])
                if int(match[2]) - position != n:
                    raise ValueError(f"Filename/sample-count mismatch: {path}")
                for k, shape in schema.items():
                    if f[k].shape != (n, *shape):
                        raise ValueError(f"Invalid shape {path}/{k}: {f[k].shape}")
                    expected_dtype = "int64" if k in ("label", "source_index") else "float32"
                    if str(f[k].dtype) != expected_dtype:
                        raise ValueError(f"Unexpected dtype {path}/{k}")
                for k, v in {
                    "split": split,
                    "condition": "periodic_theta89",
                    "boundary": "periodic",
                    "complete": True,
                    "Nt": 91,
                }.items():
                    if attrs.get(k) != v:
                        raise ValueError(f"Invalid attribute {path}/{k}")
                for ck in checkpoints.values():
                    if not np.isclose(attrs["sound_speed"], ck["setting"]["soundSpeed"]):
                        raise ValueError("Sound speed mismatch")
                if not np.isclose(attrs["dt"], manifest["dt"], rtol=1e-12, atol=0):
                    raise ValueError("Time spacing mismatch")
                if attrs["theta_deg"] != manifest["theta_deg"]:
                    raise ValueError("Angle mismatch")
                labels = f["label"][:]
                if not np.all((labels >= 0) & (labels <= 9)):
                    raise ValueError("Invalid MNIST labels")
                indices.extend(f["source_index"][:].tolist())
                records.append(
                    {
                        "file": path.name,
                        "count": n,
                        "bytes": path.stat().st_size,
                        "attributes": attrs,
                        "schema": {
                            k: {
                                "shape": list(f[k].shape),
                                "dtype": str(f[k].dtype),
                                "compression": f[k].compression,
                            }
                            for k in f
                        },
                    }
                )
                position += n
        if position != manifest["splits"][split] or len(set(indices)) != position:
            raise ValueError(f"Wrong split count or duplicate source indices: {split}")
        order = test_order if split == "test" else train_order[50000:55000]
        if not np.array_equal(indices, order):
            raise ValueError(f"Source order does not match deterministic original split: {split}")
        info["splits"][split] = {
            "count": position,
            "shards": records,
            "source_order_matches_seed": True,
            "unique_source_indices": True,
        }
    with np.load(SAMPLES) as bundled, h5py.File(sorted(root.glob("test_*.h5"))[0]) as f:
        for key, bkey in (
            ("p0", "initial_pressure"),
            ("kwave_forward", "kwave_target"),
            ("source_index", "source_index"),
            ("label", "label"),
            ("data_fft", "fourier_prediction"),
        ):
            if not np.array_equal(f[key][:4], bundled[bkey]):
                raise ValueError(f"Bundled provenance mismatch: {key}")
    info["bundled_first_four_exact_match"] = True
    info["scope"] = (
        "Header/index validation of both splits; "
        "payload finiteness checked for evaluated test samples"
    )
    return info


def summaries(rows, out):
    """Write overall/digit variability and paired differences, with direction explicit."""
    summary, paired = [], []
    metrics = ("rel_l2", "mse", "normalised_mse", "correlation")
    for label in ["all", *range(10)]:
        selected = [r for r in rows if label == "all" or r["label"] == label]
        for model in MEMBERS:
            rr = [r for r in selected if r["model"] == model]
            if not rr:
                continue
            for metric in metrics:
                v = np.array([r[metric] for r in rr])
                summary.append(
                    {
                        "label": label,
                        "model": model,
                        "metric": metric,
                        "n": len(v),
                        "mean": v.mean(),
                        "sample_sd": v.std(ddof=1) if len(v) > 1 else 0,
                        "median": np.median(v),
                        "min": v.min(),
                        "max": v.max(),
                    }
                )
        first = {r["test_position"]: r for r in selected if r["model"] == "F0"}
        second = {r["test_position"]: r for r in selected if r["model"] == "F5"}
        for metric in metrics:
            delta = np.array([second[i][metric] - r[metric] for i, r in first.items()])
            if not len(delta):
                continue
            se = delta.std(ddof=1) / np.sqrt(len(delta)) if len(delta) > 1 else 0
            improved = delta > 0 if metric == "correlation" else delta < 0
            paired.append(
                {
                    "label": label,
                    "metric": metric,
                    "n": len(delta),
                    "mean_F5_minus_F0": delta.mean(),
                    "sample_sd_delta": delta.std(ddof=1),
                    "mean_delta_normal_95_low": delta.mean() - 1.96 * se,
                    "mean_delta_normal_95_high": delta.mean() + 1.96 * se,
                    "F5_improved_count": int(improved.sum()),
                    "F5_worsened_count": int((~improved & (delta != 0)).sum()),
                }
            )
    write_csv(out / "summary.csv", summary)
    write_csv(out / "paired.csv", paired)
    return summary


def main():
    """Inspect, then evaluate a fixed prefix of the original randomised test order."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset-dir", type=Path, required=True)
    ap.add_argument("--checkpoint-archive", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=16)
    args = ap.parse_args()
    if args.limit < 1 or args.limit > 10000 or args.batch_size < 1:
        raise ValueError("Require 1 <= limit <= 10000 and positive batch size")
    torch.set_num_threads(1)
    torch.manual_seed(SEED)
    models, provenance = {}, {}
    for name in MEMBERS:
        ck, raw = read_checkpoint(args.checkpoint_archive, name)
        models[name] = ck
        provenance[name] = {
            "sha256": hashlib.sha256(raw).hexdigest(),
            "member": MEMBERS[name],
            "arguments": ck["arguments"],
            "normalization": ck["normalization"],
        }
    inspected = inspect_dataset(args.dataset_dir, models)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    out = args.output_dir
    (out / "inspection.json").write_text(json.dumps(inspected, indent=2, default=str))
    predict = {name: surrogate(ck) for name, ck in models.items()}
    rows, position, checks = [], 0, {}
    start = time.monotonic()
    for path in sorted(args.dataset_dir.glob("test_*.h5")):
        with h5py.File(path) as f:
            count = min(len(f["p0"]), args.limit - position)
            for offset in range(0, count, args.batch_size):
                stop = min(offset + args.batch_size, count)
                x, b = (
                    f["p0"][offset:stop].astype(float),
                    f["kwave_forward"][offset:stop].astype(float),
                )
                if not np.isfinite(x).all() or not np.isfinite(b).all():
                    raise ValueError("Nonfinite image/measurement payload")
                with torch.no_grad():
                    predictions = {name: g(torch.tensor(x)).numpy() for name, g in predict.items()}
                    if position == 0 and offset == 0:
                        for name, g in predict.items():
                            single = g(torch.tensor(x[0])).numpy()
                            difference = float(
                                np.linalg.norm(single - predictions[name][0])
                                / np.linalg.norm(single)
                            )
                            if difference > 1e-10:
                                raise ValueError("Batch versus single inference mismatch")
                            checks[name] = difference
                for name, pred in predictions.items():
                    if not np.isfinite(pred).all():
                        raise ValueError("Nonfinite prediction")
                    for j, (p, target) in enumerate(zip(pred, b, strict=True)):
                        mse = float(np.mean((p - target) ** 2))
                        rows.append(
                            {
                                "test_position": position + offset + j,
                                "source_index": int(f["source_index"][offset + j]),
                                "label": int(f["label"][offset + j]),
                                "shard": path.name,
                                "shard_offset": offset + j,
                                "model": name,
                                "rel_l2": rel_l2(p, target),
                                "mse": mse,
                                "normalised_mse": mse
                                / models[name]["normalization"]["data_std"] ** 2,
                                "correlation": centered_corr(p, target),
                            }
                        )
        position += count
        write_csv(out / "metrics.csv", rows)
        print(f"Evaluated {position}/{args.limit}; {time.monotonic() - start:.1f}s", flush=True)
        if position == args.limit:
            break
    summary = summaries(rows, out)
    meta = {
        "dataset_dir": str(args.dataset_dir.resolve()),
        "limit": args.limit,
        "selection": "first N in original seed-randomised test order; includes bundled first four",
        "checkpoint_archive": str(args.checkpoint_archive.resolve()),
        "provenance": provenance,
        "dtype": "float64/complex128; stored inputs float32",
        "seed": SEED,
        "batch_size": args.batch_size,
        "batch_single_relative_discrepancy": checks,
        "runtime_s": time.monotonic() - start,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "reference_gradients_evaluated": False,
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    vals = {name: np.array([r["rel_l2"] for r in rows if r["model"] == name]) for name in MEMBERS}
    for name, v in vals.items():
        axes[0].hist(v, bins=50, alpha=0.5, label=name)
    axes[0].set(xlabel="Per-sample relative L2", ylabel="Count")
    axes[0].legend()
    axes[1].scatter(vals["F0"], vals["F5"], s=3, alpha=0.3)
    maximum = max(v.max() for v in vals.values())
    axes[1].plot([0, maximum], [0, maximum], "k--")
    axes[1].set(
        xlabel="F0 relative L2", ylabel="F5 relative L2", title="Below diagonal: F5 improves"
    )
    fig.savefig(out / "forward_distribution.png", dpi=150)
    plt.close(fig)
    for row in summary:
        if row["label"] == "all":
            print(row, flush=True)


if __name__ == "__main__":
    main()
