"""Build an .npz of MNIST test images for iterative_reconstruction_comparison.py.

Draws ``--n`` images (seeded) from the stored ``mnist_large_v1`` test split, excluding the
images already in ``examples/sample_data/periodic_large_test_samples.npz`` (matched on
``source_index``), and writes them with the keys the reconstruction script reads:
``initial_pressure`` (p0), ``kwave_target`` (stored k-Wave forward data), ``label``,
``source_index`` and ``test_index`` (position in the test split).

Example::

    python -m scripts.data.make_recon_test_set --n 20 \
        --out outputs/learned_adjoint/recon_test20.npz
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATASET = ROOT / "outputs" / "datasets" / "mnist_large_v1" / "periodic_theta89"
EXISTING = ROOT / "examples" / "sample_data" / "periodic_large_test_samples.npz"
SEED = 20261124


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--dataset", type=Path, default=DATASET)
    ap.add_argument("--out", type=Path, default=ROOT / "outputs" / "learned_adjoint" / "recon_test20.npz")
    args = ap.parse_args()

    shards = sorted(args.dataset.glob("test_*.h5"))
    source = np.concatenate([h5py.File(f, "r")["source_index"][:] for f in shards])
    with np.load(EXISTING) as d:
        exclude = set(int(v) for v in d["source_index"])
    candidates = np.flatnonzero(~np.isin(source, list(exclude)))
    chosen = np.sort(np.random.default_rng(args.seed).choice(candidates, args.n, replace=False))

    cols = {k: [] for k in ("initial_pressure", "kwave_target", "label", "source_index")}
    sizes = np.cumsum([0] + [len(h5py.File(f, "r")["source_index"]) for f in shards])
    for i in chosen:
        s = int(np.searchsorted(sizes, i, side="right") - 1)
        with h5py.File(shards[s], "r") as h:
            j = int(i - sizes[s])
            cols["initial_pressure"].append(h["p0"][j])
            cols["kwave_target"].append(h["kwave_forward"][j])
            cols["label"].append(h["label"][j])
            cols["source_index"].append(h["source_index"][j])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, test_index=chosen, **{k: np.stack(v) for k, v in cols.items()})
    print(f"wrote {args.n} images to {args.out}; labels {np.stack(cols['label']).tolist()}")


if __name__ == "__main__":
    main()
