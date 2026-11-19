import csv

import h5py
import numpy as np

from scripts.evaluation.heldout_checkpoint_comparison import summaries
from scripts.evaluation.heldout_gradient_comparison import select_images


def test_selection_is_reproducible_stratified_and_excludes_bundled(tmp_path):
    with h5py.File(tmp_path / "test_00000_00030.h5", "w") as f:
        f["p0"] = np.zeros((30, 1))
        f["label"] = np.tile(np.arange(10), 3)
        f["source_index"] = np.arange(30)
    chosen = select_images(tmp_path)
    assert chosen == select_images(tmp_path)
    assert len({r["source_index"] for r in chosen}) == 20
    assert all(r["test_position"] >= 4 for r in chosen)
    assert [sum(r["label"] == digit for r in chosen) for digit in range(10)] == [2] * 10


def test_paired_summary_preserves_sample_pairing_and_improvement_sign(tmp_path):
    rows = []
    for position in range(2):
        for model, error, corr in (("F0", 0.4, 0.8), ("F5", 0.2, 0.95)):
            rows.append(
                {
                    "test_position": position,
                    "label": 0,
                    "model": model,
                    "rel_l2": error,
                    "mse": error,
                    "normalised_mse": error,
                    "correlation": corr,
                }
            )
    summaries(rows, tmp_path)
    with (tmp_path / "paired.csv").open() as f:
        paired = [r for r in csv.DictReader(f) if r["label"] == "all"]
    assert all(int(r["F5_improved_count"]) == 2 for r in paired)
    assert float(next(r for r in paired if r["metric"] == "rel_l2")["mean_F5_minus_F0"]) < 0
    assert float(next(r for r in paired if r["metric"] == "correlation")["mean_F5_minus_F0"]) > 0
