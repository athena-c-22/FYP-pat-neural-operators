"""Independently re-evaluate the thesis forward-operator results on held-out test data.

Two parts, both writing only under ``outputs/forward_reproduction``:

``principal``
    Thesis Table 1: scaled Fourier, FNO-only, Fourier-to-FNO and FNO-to-Fourier on the
    medium (1,000) and large (10,000) test splits under both acquisition conditions.
``sample_efficiency``
    Thesis Table 2 / Figure 5: every sample-efficiency checkpoint under a checkpoint
    root, evaluated on the large test split and aggregated over seeds.

Predictions are produced with the training script's own ``_model_prediction`` and
scored with ``pat_fno.data.mnist.batch_metrics``-equivalent per-sample metrics, so the
numbers are computed exactly as in the original code. Test data are never used for any
selection here; checkpoints are loaded frozen.

Run from the repository root, for example::

    python -m scripts.evaluation.reproduce_forward_results --part principal
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import time
from pathlib import Path

import h5py
import numpy as np
import torch

from pat_fno.data.mnist import CONDITIONS, build_setting, centered_corr, rel_l2
from pat_fno.models import TinyFNO2d
from scripts.train import train_forward_operator as trainer

REPO = Path(__file__).resolve().parents[2]
OUTPUTS = REPO / "outputs"
SCENARIOS = ("fno_only", "fourier_to_fno", "fno_to_fourier")
LABELS = {
    "fourier": "Scaled Fourier",
    "fno_only": "FNO-only",
    "fourier_to_fno": "Fourier-to-FNO",
    "fno_to_fourier": "FNO-to-Fourier",
}
DATASETS = {"medium": "mnist_medium_v1", "large": "mnist_large_v1"}

# Thesis-reported values (results/tables/*.csv, full precision) used for comparison.
THESIS_TABLE = REPO / "results" / "tables" / "table2_forward_accuracy.csv"
THESIS_SAMPLE_EFFICIENCY = REPO / "results" / "tables" / "table3_sample_efficiency.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--part", choices=("principal", "sample_efficiency", "all"), default="all")
    parser.add_argument("--data-root", type=Path, default=OUTPUTS / "datasets")
    parser.add_argument("--checkpoint-root", type=Path, default=OUTPUTS / "checkpoints")
    parser.add_argument(
        "--sample-efficiency-root",
        type=Path,
        default=None,
        help="Root containing <condition>/<scenario>/n<N>_seed<S>/best.pt. Defaults to "
        "the author's checkpoints.",
    )
    parser.add_argument("--tag", default="author", help="Label for a sample-efficiency run set.")
    parser.add_argument("--output", type=Path, default=OUTPUTS / "forward_reproduction")
    parser.add_argument("--scale", choices=("medium", "large", "all"), default="all")
    parser.add_argument("--condition", choices=("all", *CONDITIONS), default="all")
    parser.add_argument("--limit", type=int, default=None, help="Debug: first N test samples.")
    parser.add_argument("--batch-size", type=int, default=250)
    parser.add_argument("--threads", type=int, default=None)
    return parser.parse_args()


# ----------------------------------------------------------------------------- data


def load_test(data_root: Path, dataset: str, condition: str, limit: int | None) -> dict:
    """Load the test split in on-disk shard order (the order used by the thesis)."""
    paths = sorted((data_root / dataset / condition).glob("test_*.h5"))
    if not paths:
        raise FileNotFoundError(f"No test shards under {data_root / dataset / condition}")
    names = ("p0", "data_fft", "kwave_forward", "label", "source_index")
    chunks = {name: [] for name in names}
    for path in paths:
        with h5py.File(path, "r") as handle:
            if not handle.attrs.get("complete", False):
                raise RuntimeError(f"Incomplete shard: {path}")
            for name in names:
                chunks[name].append(handle[name][...])
    arrays = {name: np.concatenate(values) for name, values in chunks.items()}
    if limit is not None:
        arrays = {name: values[:limit] for name, values in arrays.items()}
    return arrays


# ------------------------------------------------------------------------ inference


def load_checkpoint(path: Path) -> tuple[TinyFNO2d, dict]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    arguments = checkpoint["arguments"]
    model = TinyFNO2d(arguments["modes"], arguments["modes"], arguments["width"], arguments["layers"])
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, checkpoint


@torch.no_grad()
def predict(model: TinyFNO2d, checkpoint: dict, arrays: dict, condition: str, batch: int) -> np.ndarray:
    """Physical-unit predictions using the training script's own forward path."""
    scenario = checkpoint["arguments"]["scenario"]
    norm = checkpoint["normalization"]
    # Use the forward path the checkpoint was trained with (scalar vs batched Fourier).
    trainer._configure_fourier_forward(bool(checkpoint["arguments"].get("batched_fourier", False)))
    dataset = trainer.ForwardDataset(
        arrays, scenario, norm["p0_mean"], norm["p0_std"], norm["data_mean"], norm["data_std"]
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch, shuffle=False, num_workers=0)
    setting = build_setting(condition)
    outputs = []
    for source, _target, p0 in loader:
        normalised = trainer._model_prediction(
            model, scenario, source, p0, setting, norm["data_mean"], norm["data_std"]
        )
        outputs.append((normalised * norm["data_std"] + norm["data_mean"]).numpy())
    return np.concatenate(outputs)


def per_sample_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, np.ndarray]:
    """Per-sample rel-L2, centred correlation and MSE (same definitions as the thesis)."""
    return {
        "rel_l2": np.array([rel_l2(p, t) for p, t in zip(prediction, target, strict=True)]),
        "correlation": np.array(
            [centered_corr(p, t) for p, t in zip(prediction, target, strict=True)]
        ),
        "mse": np.mean((prediction.astype(np.float64) - target) ** 2, axis=(1, 2)),
    }


# ----------------------------------------------------------------------- utilities


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_thesis_table1() -> dict[tuple[str, str, str], dict]:
    if not THESIS_TABLE.exists():
        return {}
    rename = {
        "c × Fourier baseline": "Scaled Fourier",
        "Fourier → FNO": "Fourier-to-FNO",
        "FNO → Fourier": "FNO-to-Fourier",
    }
    with THESIS_TABLE.open(encoding="utf-8") as handle:
        return {
            (row["dataset_scale"], row["condition"], rename.get(row["method"], row["method"])): row
            for row in csv.DictReader(handle)
        }


def scales(selected: str) -> list[str]:
    return list(DATASETS) if selected == "all" else [selected]


def condition_list(selected: str) -> list[str]:
    return list(CONDITIONS) if selected == "all" else [selected]


# ------------------------------------------------------------------- principal part


def run_principal(args: argparse.Namespace) -> None:
    output = args.output / "principal"
    thesis = read_thesis_table1()
    summary_rows, per_sample_rows, provenance = [], [], {}
    for scale in scales(args.scale):
        dataset = DATASETS[scale]
        for condition in condition_list(args.condition):
            arrays = load_test(args.data_root, dataset, condition, args.limit)
            target = arrays["kwave_forward"].astype(np.float64)
            n = len(target)
            print(f"[{scale}/{condition}] {n} test samples")
            predictions = {"fourier": arrays["data_fft"]}
            for scenario in SCENARIOS:
                path = (
                    args.checkpoint_root
                    / "principal/results/mnist_medium"
                    / dataset
                    / condition
                    / scenario
                    / "best.pt"
                )
                model, checkpoint = load_checkpoint(path)
                start = time.perf_counter()
                predictions[scenario] = predict(model, checkpoint, arrays, condition, args.batch_size)
                print(f"  {scenario}: {time.perf_counter() - start:.1f}s")
                provenance[f"{scale}/{condition}/{scenario}"] = {
                    "checkpoint": str(path.relative_to(args.checkpoint_root)),
                    "arguments": checkpoint["arguments"],
                    "normalization": checkpoint["normalization"],
                }
            for method, prediction in predictions.items():
                metrics = per_sample_metrics(prediction.astype(np.float64), target)
                reference = thesis.get((scale, condition, LABELS[method]), {})
                row = {
                    "scale": scale,
                    "condition": condition,
                    "method": LABELS[method],
                    "n": n,
                    "rel_l2_mean": metrics["rel_l2"].mean(),
                    "rel_l2_sd": metrics["rel_l2"].std(ddof=1),
                    "rel_l2_median": float(np.median(metrics["rel_l2"])),
                    "rel_l2_p95": float(np.percentile(metrics["rel_l2"], 95)),
                    "correlation_mean": metrics["correlation"].mean(),
                    "mse_mean": metrics["mse"].mean(),
                    "thesis_rel_l2": float(reference["relative_l2_mean"]) if reference else None,
                    "thesis_correlation": (
                        float(reference["centered_correlation_mean"]) if reference else None
                    ),
                    "thesis_mse": float(reference["mse_mean"]) if reference else None,
                }
                if reference:
                    row["abs_diff_rel_l2"] = abs(row["rel_l2_mean"] - row["thesis_rel_l2"])
                summary_rows.append(row)
                for index in range(n):
                    per_sample_rows.append(
                        {
                            "scale": scale,
                            "condition": condition,
                            "method": LABELS[method],
                            "test_position": index,
                            "source_index": int(arrays["source_index"][index]),
                            "label": int(arrays["label"][index]),
                            "rel_l2": metrics["rel_l2"][index],
                            "correlation": metrics["correlation"][index],
                            "mse": metrics["mse"][index],
                        }
                    )
                print(
                    f"  {LABELS[method]:>15}: rel-L2 {row['rel_l2_mean']:.4f}"
                    + (f" (thesis {row['thesis_rel_l2']:.4f})" if reference else "")
                )
    write_csv(output / "table1_forward_accuracy.csv", summary_rows)
    write_csv(output / "per_sample_metrics.csv", per_sample_rows)
    (output / "meta.json").write_text(
        json.dumps({**environment(args), "provenance": provenance}, indent=2), encoding="utf-8"
    )
    print(f"Saved {output}")


# ----------------------------------------------------------- sample-efficiency part


def run_sample_efficiency(args: argparse.Namespace) -> None:
    root = args.sample_efficiency_root or (
        args.checkpoint_root / "sample_efficiency/results/sample_efficiency/mnist_large_v1"
    )
    output = args.output / f"sample_efficiency_{args.tag}"
    run_rows = []
    for condition in condition_list(args.condition):
        arrays = load_test(args.data_root, "mnist_large_v1", condition, args.limit)
        target = arrays["kwave_forward"].astype(np.float64)
        for scenario in SCENARIOS:
            for path in sorted((root / condition / scenario).glob("n*_seed*/best.pt")):
                model, checkpoint = load_checkpoint(path)
                arguments = checkpoint["arguments"]
                start = time.perf_counter()
                prediction = predict(model, checkpoint, arrays, condition, args.batch_size)
                metrics = per_sample_metrics(prediction.astype(np.float64), target)
                run_rows.append(
                    {
                        "condition": condition,
                        "model": LABELS[scenario],
                        "train_samples": int(arguments["train_samples"]),
                        "seed": int(arguments["seed"]),
                        "epochs": int(arguments["epochs"]),
                        "n_test": len(target),
                        "rel_l2_mean": metrics["rel_l2"].mean(),
                        "correlation_mean": metrics["correlation"].mean(),
                        "mse_mean": metrics["mse"].mean(),
                        "checkpoint": str(path),
                    }
                )
                print(
                    f"{condition} {scenario} n={arguments['train_samples']} "
                    f"seed={arguments['seed']}: {run_rows[-1]['rel_l2_mean']:.4f} "
                    f"({time.perf_counter() - start:.1f}s)"
                )
                write_csv(output / "runs.csv", run_rows)  # incremental, survives interruption
    write_csv(output / "table2_sample_efficiency.csv", aggregate_sample_efficiency(run_rows))
    (output / "meta.json").write_text(
        json.dumps({**environment(args), "checkpoint_root": str(root)}, indent=2), encoding="utf-8"
    )
    print(f"Saved {output}")


def aggregate_sample_efficiency(run_rows: list[dict]) -> list[dict]:
    thesis = {}
    if THESIS_SAMPLE_EFFICIENCY.exists():
        with THESIS_SAMPLE_EFFICIENCY.open(encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                thesis[(row["condition"], row["model"], int(row["train_samples"]))] = row
    groups: dict[tuple, list[float]] = {}
    for row in run_rows:
        groups.setdefault((row["condition"], row["model"], row["train_samples"]), []).append(
            row["rel_l2_mean"]
        )
    table = []
    for (condition, model, n), values in sorted(groups.items()):
        reference = thesis.get((condition, model, n))
        table.append(
            {
                "condition": condition,
                "model": model,
                "train_samples": n,
                "runs": len(values),
                "rel_l2_mean": float(np.mean(values)),
                "rel_l2_sd_across_seeds": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                "thesis_rel_l2_mean": float(reference["rel_l2_mean"]) if reference else None,
                "thesis_rel_l2_sd": (
                    float(reference["rel_l2_std_across_seeds"]) if reference else None
                ),
            }
        )
    return table


def environment(args: argparse.Namespace) -> dict:
    return {
        "command": vars(args) | {k: str(v) for k, v in vars(args).items() if isinstance(v, Path)},
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "cpu": platform.processor(),
        "threads": torch.get_num_threads(),
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
    }


def main() -> None:
    args = parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    if args.part in ("principal", "all"):
        run_principal(args)
    if args.part in ("sample_efficiency", "all"):
        run_sample_efficiency(args)


if __name__ == "__main__":
    main()
