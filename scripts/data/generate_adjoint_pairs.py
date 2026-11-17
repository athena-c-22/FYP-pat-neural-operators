"""Generate small synthetic (x, y = A x, z = A* y) triples with k-Wave for learned-adjoint tests.

``x``: 64x64 procedural phantom (no external dataset, no patient data); ``y``: k-Wave forward
data (64x91); ``z``: k-Wave adjoint of ``y`` (64x64). Phantom kinds mimic what a reconstruction
loop feeds an adjoint: digit-like ``strokes``, ``blobs``, and signed ``diff`` images (a
difference of two phantoms, like ``p_true - p_current``; the data of such an image is a
residual ``A x - b`` because A is linear).

Cost is dominated by k-Wave (measured ~6.7 s per pair on the development machine, with no
speed-up from more worker processes), so keep ``--n`` small. Output is written in shards of
``--shard`` pairs so an interrupted run can be resumed (existing shards are never overwritten).

Example::

    python -m scripts.data.generate_adjoint_pairs --n 200 --out outputs/learned_adjoint/data
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from pat_fno.data.mnist import build_setting
from pat_fno.operators.kwave import kwave_adjoint_2d, kwave_forward_2d

SEED = 20260728
N = 64
KINDS = ("strokes", "blobs", "diff")


def _blur(img: np.ndarray, sigma: float) -> np.ndarray:
    k = np.hypot(*np.meshgrid(np.fft.fftfreq(N), np.fft.fftfreq(N), indexing="ij"))
    return np.real(np.fft.ifft2(np.fft.fft2(img) * np.exp(-2 * (np.pi * sigma * k) ** 2)))


def strokes(rng: np.random.Generator) -> np.ndarray:
    """1-3 random polylines of width ~1.5-3 px inside the central 40x40 (MNIST-like)."""
    img = np.zeros((N, N))
    for _ in range(rng.integers(1, 4)):
        pts = rng.uniform(12, 52, size=(rng.integers(2, 5), 2))
        for a, b in zip(pts[:-1], pts[1:], strict=True):
            for t in np.linspace(0, 1, 80):
                i, j = np.rint(a + t * (b - a)).astype(int)
                img[i, j] = 1.0
    img = _blur(img, rng.uniform(1.0, 1.8))
    return img / img.max()


def blobs(rng: np.random.Generator) -> np.ndarray:
    """3-8 Gaussian blobs of random width and amplitude, away from the border."""
    yy, xx = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    img = np.zeros((N, N))
    for _ in range(rng.integers(3, 9)):
        c = rng.uniform(14, 50, size=2)
        w = rng.uniform(1.5, 5.0)
        img += rng.uniform(0.3, 1.0) * np.exp(-((yy - c[0]) ** 2 + (xx - c[1]) ** 2) / (2 * w**2))
    return img / img.max()


def make_phantom(kind: str, rng: np.random.Generator) -> np.ndarray:
    if kind == "strokes":
        return strokes(rng)
    if kind == "blobs":
        return blobs(rng)
    if kind == "diff":
        a = strokes(rng) if rng.random() < 0.5 else blobs(rng)
        b = strokes(rng) if rng.random() < 0.5 else blobs(rng)
        return a - rng.uniform(0.3, 1.0) * b
    raise KeyError(kind)


def _one(args):
    kind, seed = args
    x = make_phantom(kind, np.random.default_rng(seed)).astype(np.float32)
    s = build_setting("periodic_theta89")
    y = kwave_forward_2d(x, s).astype(np.float32)
    z = kwave_adjoint_2d(y, s).astype(np.float32)
    return x, y, z


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--shard", type=int, default=20)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--out", type=Path, default=Path("outputs/learned_adjoint/data"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    kinds = [KINDS[i] for i in rng.integers(0, len(KINDS), size=args.n)]
    seeds = rng.integers(0, 2**31 - 1, size=args.n)
    t0 = time.time()
    for s0 in range(0, args.n, args.shard):
        path = args.out / f"shard_{s0:05d}.npz"
        if path.exists():
            print(f"skip existing {path.name}", flush=True)
            continue
        idx = range(s0, min(s0 + args.shard, args.n))
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            res = list(ex.map(_one, [(kinds[i], int(seeds[i])) for i in idx]))
        np.savez_compressed(
            path,
            x=np.stack([r[0] for r in res]),
            y=np.stack([r[1] for r in res]),
            z=np.stack([r[2] for r in res]),
            kind=np.array([kinds[i] for i in idx]),
            index=np.array(list(idx)),
        )
        print(f"{path.name} done, {time.time() - t0:.0f}s elapsed", flush=True)
    (args.out / "meta.json").write_text(
        json.dumps(
            {
                "seed": SEED,
                "n": args.n,
                "kinds": KINDS,
                "condition": "periodic_theta89",
                "python": platform.python_version(),
                "note": "z = k-Wave adjoint of y = k-Wave forward of x (approximate adjoint)",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
