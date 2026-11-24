"""Measured wall-clock time of lean gradient-descent reconstruction loops.

iterative_reconstruction_comparison.py computes every direction at every iterate (for the
cosine logging), so its run time is not the cost of any single method. This script runs each
method on its own, with nothing extra, and times it end to end:

    reference      d = A* r                                   1 forward + 1 adjoint per step
    G              d = G(r)                                   1 forward + G per step
    G+fallback     d = G(r); if not a descent direction,      as G, plus 1 forward + 1
                   d = A* r for that step                     adjoint per fallback

All use the exact k-Wave line search alpha = <r, A d> / ||A d||^2 and update the residual by
linearity (r <- r - alpha A d), so every step costs exactly one k-Wave forward for A d. The
backprojection start costs one adjoint and one forward, and it is included in the time.

Output: one CSV row per (sample, init, method, iteration) with the cumulative wall-clock
seconds and the image error, plus a summary of the time to reach the reference's error at
iterations 5, 10 and 20.

Example::

    python -m scripts.evaluation.time_reconstruction --samples outputs/learned_adjoint/recon_test20.npz \
        --sample-index 0 1 2 3 4 --learned A=outputs/learned_adjoint/runs_resid/A_ws_m8_lam0_seed0.pt
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch

from scripts.evaluation.adjoint_consistency import apply_adjoint, apply_forward, make_setting
from scripts.evaluation.learned_adjoint_eval import load_model, predict

CONDITION = "periodic_theta89"
TARGET_ITERATIONS = (5, 10, 20)


def forward(x, s):
    return np.asarray(apply_forward("kwave", x, s), dtype=np.float64)


def adjoint(y, s):
    return np.asarray(apply_adjoint("kwave", y, s), dtype=np.float64)


def run(method, model, x_true, b, s, init, iterations):
    """Return a list of (iteration, cumulative seconds, image rel. error, fallbacks)."""
    rel = lambda x: float(np.linalg.norm(x - x_true) / np.linalg.norm(x_true))  # noqa: E731
    start = time.perf_counter()
    x, r = np.zeros_like(x_true), -b.copy()
    if init == "backprojection":  # one reference step from zero, as in the comparison script
        d = adjoint(r, s)
        a_d = forward(d, s)
        alpha = max(float((r * a_d).sum() / (a_d * a_d).sum()), 0.0)
        x, r = x - alpha * d, r - alpha * a_d
    out, fallbacks = [(0, time.perf_counter() - start, rel(x), 0)], 0
    for k in range(1, iterations + 1):
        if method == "reference":
            d = adjoint(r, s)
        else:
            d = predict(model, r[None].astype(np.float32))[0]
        a_d = forward(d, s)
        alpha = float((r * a_d).sum() / (a_d * a_d).sum())
        if method.endswith("+fallback") and alpha <= 0.0:
            fallbacks += 1
            d = adjoint(r, s)
            a_d = forward(d, s)
            alpha = float((r * a_d).sum() / (a_d * a_d).sum())
        alpha = max(alpha, 0.0)
        x, r = x - alpha * d, r - alpha * a_d
        out.append((k, time.perf_counter() - start, rel(x), fallbacks))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--samples", type=Path, default=Path("outputs/learned_adjoint/recon_test20.npz"))
    ap.add_argument("--sample-index", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--learned", nargs="+", required=True, help="NAME=checkpoint.pt")
    ap.add_argument("--iterations", type=int, default=20)
    ap.add_argument("--init", nargs="+", default=["zero", "backprojection"])
    ap.add_argument("--out", type=Path, default=Path("outputs/learned_adjoint/timing"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)  # G is tiny; avoid competing with k-Wave for cores

    s = make_setting(CONDITION)
    models = {}
    for item in args.learned:
        name, path = item.split("=", 1)
        models[name] = load_model(Path(path))[0]
    methods = [("reference", None)]
    for name, m in models.items():
        methods += [(name, m), (f"{name}+fallback", m)]

    with np.load(args.samples) as d:
        xs, bs = d["initial_pressure"].astype(np.float64), d["kwave_target"].astype(np.float64)
    forward(xs[0], s)  # warm-up call (first k-Wave call pays one-off start-up costs)

    rows = []
    for i in args.sample_index:
        for init in args.init:
            for name, model in methods:
                res = run(name, model, xs[i], bs[i], s, init, args.iterations)
                rows += [dict(sample=i, init=init, method=name, iteration=k, seconds=t,
                              image_rel_l2=e, fallbacks=f) for k, t, e, f in res]
                print(f"sample {i} {init:14s} {name:12s} {res[-1][1]:6.1f}s  "
                      f"err {res[-1][2]:.3f}  fallbacks {res[-1][3]}", flush=True)
    with (args.out / "timing.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    summary = {}
    for init in args.init:
        for name, _ in methods:
            entry = {}
            for kk in (k for k in TARGET_ITERATIONS if k <= args.iterations):
                times = []
                for i in args.sample_index:
                    get = lambda m: [r for r in rows if r["sample"] == i and r["init"] == init  # noqa: E731
                                     and r["method"] == m]
                    target = get("reference")[kk]["image_rel_l2"]
                    hit = [r["seconds"] for r in get(name) if r["image_rel_l2"] <= target + 1e-12]
                    times.append(hit[0] if hit else None)
                ok = [t for t in times if t is not None]
                entry[f"ref_k{kk}"] = {"reached": len(ok), "of": len(times),
                                       "median_seconds": float(np.median(ok)) if ok else None}
            last = [r for r in rows if r["init"] == init and r["method"] == name
                    and r["iteration"] == args.iterations]
            entry["median_seconds_total"] = float(np.median([r["seconds"] for r in last]))
            entry["median_final_err"] = float(np.median([r["image_rel_l2"] for r in last]))
            summary[f"{init}/{name}"] = entry
    (args.out / "timing_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
