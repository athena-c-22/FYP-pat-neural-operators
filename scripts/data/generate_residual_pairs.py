"""Harvest residual training pairs (r_k, A* r_k) from gradient-descent trajectories.

The learned adjoint G trained on clean data y = A x fails on reconstruction residuals
r_k = A x_k - b (its cosine with A* r_k falls from ~0.98 at k = 0 to ~0.05 by k = 10).
This script collects exactly those inputs. For each new procedural phantom x_true
(same generator as generate_adjoint_pairs.py, different seed) with b = A x_true it runs
gradient descent from x = 0 with an exact line search and stores, at every iterate,

    y = r_k = A x_k - b,   z = A* r_k   (k-Wave adjoint),   x = x_k - x_true

(r_k = A (x_k - x_true) because A is linear, so x is the image whose data is y).

Policies (--policies):
    reference : step along A* r_k (the target is already the step direction)
    G         : step along G(r_k) for the --learned checkpoint; if that is not a descent
                direction for the k-Wave loss (line-search step <= 0), fall back to A* r_k.
                This gives the residuals G itself produces (DAgger).

Cost: per pair one k-Wave adjoint (the target) and one k-Wave forward (the line search,
A d); the next residual is r - alpha * A d by linearity, so no further forward is needed.
A G fallback costs one extra forward. Shards hold whole phantoms (all iterations, all
policies), so a split by phantom (``group``) keeps trajectories out of val/test leakage.

Example::

    python -m scripts.data.generate_residual_pairs --n-phantoms 100 --iterations 12 \
        --out outputs/learned_adjoint/data_resid
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np

from scripts.data.generate_adjoint_pairs import KINDS, make_phantom
from scripts.evaluation.adjoint_consistency import apply_adjoint, apply_forward, make_setting
from scripts.evaluation.learned_adjoint_eval import load_model, predict

SEED = 20260730
CONDITION = "periodic_theta89"


def forward(x, s):
    return np.asarray(apply_forward("kwave", x, s), dtype=np.float64)


def adjoint(y, s):
    return np.asarray(apply_adjoint("kwave", y, s), dtype=np.float64)


def trajectory(x_true, b, s, policy, model, iterations):
    """Run GD from zero; return per-iterate (error image, residual, A* residual, used_fallback)."""
    x = np.zeros_like(x_true)
    r = -b.copy()
    out = []
    for _ in range(iterations):
        g_ref = adjoint(r, s)
        d, fallback = g_ref, False
        if policy == "G":
            d = predict(model, r[None].astype(np.float32))[0]
        a_d = forward(d, s)
        alpha = float((r * a_d).sum() / (a_d * a_d).sum())
        if policy == "G" and alpha <= 0.0:
            d, fallback = g_ref, True
            a_d = forward(d, s)
            alpha = float((r * a_d).sum() / (a_d * a_d).sum())
        out.append((x - x_true, r.copy(), g_ref, fallback))
        alpha = max(alpha, 0.0)
        x = x - alpha * d
        r = r - alpha * a_d
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--n-phantoms", type=int, default=100)
    ap.add_argument("--iterations", type=int, default=12)
    ap.add_argument("--policies", nargs="+", default=["reference", "G"], choices=["reference", "G"])
    ap.add_argument("--learned", type=Path,
                    default=Path("outputs/learned_adjoint/runs_4k_e300/lam0_seed0.pt"))
    ap.add_argument("--shard", type=int, default=5, help="phantoms per shard")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--out", type=Path, default=Path("outputs/learned_adjoint/data_resid"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    s = make_setting(CONDITION)
    model = load_model(args.learned)[0] if "G" in args.policies else None
    rng = np.random.default_rng(args.seed)
    kinds = [KINDS[i] for i in rng.integers(0, len(KINDS), size=args.n_phantoms)]
    seeds = rng.integers(0, 2**31 - 1, size=args.n_phantoms)
    per_phantom = args.iterations * len(args.policies)

    t0 = time.time()
    for p0 in range(0, args.n_phantoms, args.shard):
        path = args.out / f"shard_{p0 * per_phantom:05d}.npz"
        if path.exists():
            print(f"skip existing {path.name}", flush=True)
            continue
        cols = {k: [] for k in ("x", "y", "z", "kind", "group", "iteration", "policy", "fallback")}
        for p in range(p0, min(p0 + args.shard, args.n_phantoms)):
            x_true = make_phantom(kinds[p], np.random.default_rng(int(seeds[p]))).astype(np.float64)
            b = forward(x_true, s)
            for policy in args.policies:
                for k, (e, r, g, fb) in enumerate(trajectory(x_true, b, s, policy, model,
                                                             args.iterations)):
                    for key, v in zip(("x", "y", "z", "kind", "group", "iteration", "policy",
                                       "fallback"), (e, r, g, kinds[p], p, k, policy, fb)):
                        cols[key].append(v)
        n0 = p0 * per_phantom
        np.savez_compressed(
            path,
            x=np.stack(cols["x"]).astype(np.float32),
            y=np.stack(cols["y"]).astype(np.float32),
            z=np.stack(cols["z"]).astype(np.float32),
            kind=np.array(cols["kind"]),
            index=np.arange(n0, n0 + len(cols["x"])),
            group=np.array(cols["group"]),
            iteration=np.array(cols["iteration"]),
            policy=np.array(cols["policy"]),
            fallback=np.array(cols["fallback"]),
        )
        print(f"{path.name} done ({p + 1}/{args.n_phantoms} phantoms, "
              f"G fallbacks {int(np.sum(cols['fallback']))}), {time.time() - t0:.0f}s elapsed",
              flush=True)
    (args.out / "meta.json").write_text(json.dumps({
        "seed": args.seed, "n_phantoms": args.n_phantoms, "iterations": args.iterations,
        "policies": args.policies, "learned": str(args.learned), "kinds": KINDS,
        "condition": CONDITION, "python": platform.python_version(),
        "note": "y = A x_k - b residual on a zero-init GD trajectory, z = k-Wave adjoint of y, "
                "x = x_k - x_true",
    }, indent=2))


if __name__ == "__main__":
    main()
