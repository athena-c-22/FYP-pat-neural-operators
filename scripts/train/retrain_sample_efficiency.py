"""Retrain the thesis sample-efficiency grid with the original training code.

Reuses ``scripts.train.train_forward_operator`` (dataset class, normalisation, seeding,
AdamW + cosine schedule, best-validation checkpointing) with two memory/time-driven
differences, neither of which changes what is trained:

* Only the selected training subset is read from the HDF5 shards. The subset indices
  are computed exactly as ``_select_training_subset`` does on the full 50,000-sample
  split (``default_rng(seed).permutation(50000)[:N]``, sorted), so subsets match.
* No test-split predictions are made here; test evaluation is done separately by
  ``scripts.evaluation.reproduce_forward_results`` for both these and the author's
  checkpoints, so both sets are scored identically.

Configuration follows ``training_logs/configs/sample_efficiency.yaml``. Runs that
already have ``best.pt`` + ``history.json`` are skipped, so the grid can be resumed.

    python -m scripts.train.retrain_sample_efficiency --condition all --scenario all
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader

from pat_fno.data.mnist import CONDITIONS, build_setting, save_json
from pat_fno.models import TinyFNO2d
from scripts.train import train_forward_operator as trainer

REPO = Path(__file__).resolve().parents[2]
OUTPUTS = REPO / "outputs"
DATASET = "mnist_large_v1"
TRAIN_SIZES = (1000, 5000, 10000, 25000, 50000)
SEEDS = (20260728, 20260729, 20260730)
CONFIG = {"epochs": 5, "batch_size": 128, "learning_rate": 2e-3, "modes": 8, "width": 16, "layers": 3}
FIELDS = ("p0", "data_fft", "kwave_forward", "label", "source_index")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition", choices=("all", *CONDITIONS), default="all")
    parser.add_argument("--scenario", choices=("all", *trainer.SCENARIOS), default="all")
    parser.add_argument("--sizes", type=int, nargs="+", default=list(TRAIN_SIZES))
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--epochs", type=int, default=CONFIG["epochs"])
    parser.add_argument("--max-batches", type=int, default=None, help="Debug/timing only.")
    parser.add_argument("--data-root", type=Path, default=OUTPUTS / "datasets")
    parser.add_argument(
        "--output-root", type=Path, default=OUTPUTS / "sample_efficiency_retrain" / DATASET
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--threads", type=int, default=None)
    return parser.parse_args()


def shard_paths(data_root: Path, condition: str, split: str) -> list[Path]:
    return sorted((data_root / DATASET / condition).glob(f"{split}_*.h5"))


def shard_lengths(paths: list[Path]) -> list[int]:
    lengths = []
    for path in paths:
        with h5py.File(path, "r") as handle:
            if not handle.attrs.get("complete", False):
                raise RuntimeError(f"Incomplete shard: {path}")
            lengths.append(handle["p0"].shape[0])
    return lengths


def read_rows(paths: list[Path], lengths: list[int], indices: np.ndarray | None) -> dict:
    """Read the given global row indices (sorted) or every row from a split's shards."""
    offsets = np.concatenate([[0], np.cumsum(lengths)])
    chunks = {name: [] for name in FIELDS}
    for shard, path in enumerate(paths):
        if indices is None:
            local = slice(None)
        else:
            mask = (indices >= offsets[shard]) & (indices < offsets[shard + 1])
            if not mask.any():
                continue
            local = indices[mask] - offsets[shard]
        with h5py.File(path, "r") as handle:
            for name in FIELDS:
                chunks[name].append(handle[name][local])
    return {name: np.concatenate(values) for name, values in chunks.items()}


def subset_indices(available: int, count: int, seed: int) -> np.ndarray:
    """Identical to ``trainer._select_training_subset`` on the full split."""
    return np.sort(np.random.default_rng(seed).permutation(available)[:count])


def train_one(args, condition, scenario, count, seed, train_paths, train_lengths, validation):
    run_root = args.output_root / condition / scenario / f"n{count}_seed{seed}"
    if (run_root / "best.pt").exists() and (run_root / "history.json").exists():
        print(f"skip existing {run_root}")
        return
    run_root.mkdir(parents=True, exist_ok=True)
    trainer._configure_fourier_forward(False)  # sample-efficiency runs used the scalar path
    trainer._seed(seed)
    train = read_rows(train_paths, train_lengths, subset_indices(sum(train_lengths), count, seed))
    normalization = trainer._normalization(train)
    statistics = (
        normalization["p0_mean"],
        normalization["p0_std"],
        normalization["data_mean"],
        normalization["data_std"],
    )
    train_set = trainer.ForwardDataset(train, scenario, *statistics)
    validation_set = trainer.ForwardDataset(validation, scenario, *statistics)
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_set, batch_size=CONFIG["batch_size"], shuffle=True, num_workers=0, generator=generator
    )
    validation_loader = DataLoader(
        validation_set, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=0
    )
    if args.max_batches:
        train_loader = [batch for _, batch in zip(range(args.max_batches), train_loader)]
        validation_loader = [batch for _, batch in zip(range(args.max_batches), validation_loader)]
    device = trainer._device(args.device)
    model = TinyFNO2d(CONFIG["modes"], CONFIG["modes"], CONFIG["width"], CONFIG["layers"]).to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=CONFIG["learning_rate"], weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=args.epochs)
    arguments = {
        "dataset": DATASET,
        "scale": "large",
        "condition": condition,
        "scenario": scenario,
        "batched_fourier": False,
        "epochs": args.epochs,
        "batch_size": CONFIG["batch_size"],
        "learning_rate": CONFIG["learning_rate"],
        "modes": CONFIG["modes"],
        "width": CONFIG["width"],
        "layers": CONFIG["layers"],
        "device": device.type,
        "seed": seed,
        "train_samples": count,
        "output_root": str(run_root),
        "retrained_by": "scripts.train.retrain_sample_efficiency",
    }
    run_args = argparse.Namespace(epochs=args.epochs)
    setting = build_setting(condition)
    start = time.perf_counter()
    history, best = trainer._train_model(
        model,
        train_loader,
        validation_loader,
        optimiser,
        scheduler,
        run_args,
        setting,
        normalization,
        device,
        run_root,
        arguments,
    )
    elapsed = time.perf_counter() - start
    save_json(
        run_root / "history.json",
        {
            "arguments": arguments,
            "history": history,
            "best_validation_normalized_mse": best,
            "train_seconds": elapsed,
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": device.type,
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "threads": torch.get_num_threads(),
        },
    )
    print(f"done {condition} {scenario} n={count} seed={seed}: best val {best:.5f} ({elapsed:.0f}s)")


def main() -> None:
    args = parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    conditions = list(CONDITIONS) if args.condition == "all" else [args.condition]
    scenarios = list(trainer.SCENARIOS) if args.scenario == "all" else [args.scenario]
    for condition in conditions:
        train_paths = shard_paths(args.data_root, condition, "train")
        train_lengths = shard_lengths(train_paths)
        validation_paths = shard_paths(args.data_root, condition, "validation")
        validation = read_rows(validation_paths, shard_lengths(validation_paths), None)
        print(
            f"{condition}: {sum(train_lengths)} train available, "
            f"{len(validation['p0'])} validation"
        )
        for scenario in scenarios:
            for count in args.sizes:
                for seed in args.seeds:
                    train_one(
                        args, condition, scenario, count, seed, train_paths, train_lengths, validation
                    )
    (args.output_root / "run_info.json").write_text(
        json.dumps({"config": CONFIG, "sizes": args.sizes, "seeds": args.seeds}, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
