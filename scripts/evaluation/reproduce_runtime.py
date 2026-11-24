"""Re-run the thesis forward-runtime table (Table 3) on this machine's CPU.

Thin wrapper around ``scripts.evaluation.benchmark_forward_operators`` that keeps the
original timing code and only redirects data, checkpoint and output locations to
``outputs/``. Modes follow ``training_logs/configs/runtime.yaml``:

* ``cpu_batch1``  - batch 1, 20 warm-up calls, 200 repeats, k-Wave included (as thesis).
* ``cpu_batch64`` - batch 64, 10 warm-up calls, 50 repeats, k-Wave included. Not in the
  thesis (which ran batch 64 on an RTX 4090); added as the CPU throughput analogue.

The RTX 4090 rows cannot be reproduced without a GPU and are not produced here.

    python -m scripts.evaluation.reproduce_runtime
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch

from pat_fno.data import mnist
from pat_fno.data.mnist import CONDITIONS
from scripts.evaluation import benchmark_forward_operators as bench

REPO = Path(__file__).resolve().parents[2]
OUTPUTS = REPO / "outputs"
MODES = {
    "cpu_batch1": {"batch_size": 1, "warmup": 20, "repeats": 200, "include_kwave": True},
    # k-Wave is a per-sample loop in the benchmark, so batching does not change its
    # per-sample time; batch-64 speed-ups are computed against the batch-1 k-Wave time.
    "cpu_batch64": {"batch_size": 64, "warmup": 10, "repeats": 50, "include_kwave": False},
}
THESIS_RUNTIME = REPO / "results" / "tables" / "table4_runtime.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modes", nargs="+", choices=tuple(MODES), default=list(MODES))
    parser.add_argument("--condition", choices=("all", *CONDITIONS), default="all")
    parser.add_argument("--dataset", default="mnist_large_v1")
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--output", type=Path, default=OUTPUTS / "runtime_reproduction")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    # benchmark_condition reads ROOT / "datasets" / <dataset>; point it at outputs/.
    bench.ROOT = OUTPUTS
    mnist.ROOT = OUTPUTS
    checkpoint_root = OUTPUTS / "checkpoints/principal/results/mnist_medium" / args.dataset
    device = torch.device("cpu")
    conditions = list(CONDITIONS) if args.condition == "all" else [args.condition]
    combined = []
    for mode in args.modes:
        settings = MODES[mode]
        output_root = args.output / mode
        output_root.mkdir(parents=True, exist_ok=True)
        bench_args = argparse.Namespace(
            dataset=args.dataset,
            split="test",
            device="cpu",
            batch_size=settings["batch_size"],
            max_samples=settings["batch_size"],
            warmup=settings["warmup"],
            repeats=settings["repeats"],
            include_kwave=settings["include_kwave"],
            overwrite=True,
        )
        for condition in conditions:
            print(f"[{mode}] {condition}")
            bench.benchmark_condition(bench_args, condition, device, checkpoint_root, output_root)
            metadata = json.loads((output_root / f"{condition}_cpu.json").read_text())
            for row in metadata["rows"]:
                combined.append({"execution_mode": mode, "batch_size": settings["batch_size"], **row})
    kwave_ms = {
        row["condition"]: row["mean_ms_per_sample"]
        for row in combined
        if row["method"] == "k-Wave" and row["execution_mode"] == "cpu_batch1"
    }
    for row in combined:
        if row["speedup_vs_kwave"] is None and row["condition"] in kwave_ms:
            row["speedup_vs_kwave"] = kwave_ms[row["condition"]] / row["mean_ms_per_sample"]
            row["speedup_reference"] = "cpu_batch1 k-Wave"
        else:
            row.setdefault("speedup_reference", "same-mode k-Wave")
    thesis = {}
    if THESIS_RUNTIME.exists():
        with THESIS_RUNTIME.open(encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if row["execution_mode"] == "cpu_batch1":
                    thesis[(row["condition"], row["method"])] = row
    for row in combined:
        reference = thesis.get((row["condition"], row["method"])) if row["execution_mode"] == "cpu_batch1" else None
        row["thesis_mean_ms_per_sample"] = float(reference["mean_ms_per_sample"]) if reference else None
        row["thesis_speedup_vs_kwave"] = float(reference["speedup_vs_kwave"]) if reference else None
    with (args.output / "table3_runtime_cpu.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(combined[0]))
        writer.writeheader()
        writer.writerows(combined)
    print(f"Saved {args.output / 'table3_runtime_cpu.csv'}")


if __name__ == "__main__":
    main()
