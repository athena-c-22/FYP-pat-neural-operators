"""Train a learned adjoint ``G_phi: y -> x`` from (x, y = A x, z = A* y) triples.

Loss = MSE(G(y), z) in normalised units  +  lambda * adjoint_dot_loss(x, y, G(y)).

``lambda = 0`` is plain supervised regression onto the reference adjoint ``z``; ``lambda > 0``
adds the adjoint-consistency term, which uses only the pairs ``(x_i, y_i)`` and pulls ``G``
towards the transpose of ``A`` instead of towards the (approximate) reference adjoint.

Split: a seeded permutation (``SEED_SPLIT``, fixed) -> 70 % train / 15 % val / 15 % test, done
separately for each ``--data`` folder. Folders whose shards carry a ``group`` key (residual
pairs from generate_residual_pairs.py, one group per phantom trajectory) are split by group,
so no trajectory is shared between train, val and test. Test error is reported per folder.

``--loss rel`` uses the per-sample relative squared error ||G(y) - z||^2 / ||z||^2, so that
small late-iteration residuals count as much as large clean data. ``--init-from`` warm-starts
from an earlier checkpoint (its normalisation statistics are kept).

Normalisation statistics come from the training split only. The best-validation checkpoint
is saved; an existing checkpoint is never overwritten.

Small CPU run (about a minute)::

    python -m scripts.train.train_learned_adjoint --data outputs/learned_adjoint/data \
        --out outputs/learned_adjoint/runs --dot-weight 1 --seed 0

The same script scales to a large dataset on a GPU (``--device cuda``, larger ``--epochs``,
``--width``, ``--batch-size``); nothing here is GPU specific.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch

from pat_fno.models.learned_adjoint import LearnedAdjoint, adjoint_dot_loss

SEED_SPLIT = 20260728


def load_data(path: Path) -> dict[str, np.ndarray]:
    """Concatenate all ``shard_*.npz`` files (or one ``.npz``) in index order."""
    files = [path] if path.suffix == ".npz" else sorted(path.glob("shard_*.npz"))
    if not files:
        raise FileNotFoundError(f"no shards in {path}")
    parts = [np.load(f) for f in files]
    return {k: np.concatenate([p[k] for p in parts]) for k in ("x", "y", "z", "kind", "index")}


def split_indices(n: int, seed: int = SEED_SPLIT) -> dict[str, np.ndarray]:
    perm = np.random.default_rng(seed).permutation(n)
    a, b = int(0.7 * n), int(0.85 * n)
    return {"train": perm[:a], "val": perm[a:b], "test": perm[b:]}


def split_groups(group: np.ndarray, seed: int = SEED_SPLIT) -> dict[str, np.ndarray]:
    """70/15/15 split of whole groups (phantom trajectories); returns pair indices."""
    g = split_indices(len(np.unique(group)), seed)
    ids = np.unique(group)
    return {k: np.flatnonzero(np.isin(group, ids[v])) for k, v in g.items()}


def load_parts(paths: list[Path]):
    """Load several data folders; split each one separately and concatenate.

    Returns (data, split, split_parts) where split holds indices into the concatenated
    arrays and split_parts[name] holds them per folder (also into the concatenation).
    """
    datas, split, parts, offset = [], {"train": [], "val": [], "test": []}, {}, 0
    for path in paths:
        files = [path] if path.suffix == ".npz" else sorted(path.glob("shard_*.npz"))
        if not files:
            raise FileNotFoundError(f"no shards in {path}")
        shards = [np.load(f) for f in files]
        d = {k: np.concatenate([s[k] for s in shards]) for k in ("x", "y", "z", "kind")}
        sp = split_groups(np.concatenate([s["group"] for s in shards])) if "group" in shards[0]             else split_indices(len(d["x"]))
        parts[str(path)] = {k: v + offset for k, v in sp.items()}
        for k in split:
            split[k].append(v := sp[k] + offset)
        datas.append(d)
        offset += len(d["x"])
    data = {k: np.concatenate([d[k] for d in datas]) for k in ("x", "y", "z", "kind")}
    return data, {k: np.concatenate(v) for k, v in split.items()}, parts


def evaluate(model, d, idx, dev) -> float:
    """Relative L2 error of ``G(y)`` against ``z`` (mean over samples)."""
    model.eval()
    with torch.no_grad():
        pred = model(torch.from_numpy(d["y"][idx]).to(dev)).cpu().numpy()
    z = d["z"][idx]
    num = np.linalg.norm((pred - z).reshape(len(idx), -1), axis=1)
    den = np.linalg.norm(z.reshape(len(idx), -1), axis=1)
    return float(np.mean(num / den))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--data", type=Path, nargs="+", default=[Path("outputs/learned_adjoint/data")])
    ap.add_argument("--out", type=Path, default=Path("outputs/learned_adjoint/runs"))
    ap.add_argument("--dot-weight", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--modes", type=int, default=8)
    ap.add_argument("--width", type=int, default=16)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--loss", choices=("mse", "rel"), default="mse")
    ap.add_argument("--init-from", type=Path, help="warm-start checkpoint (keeps its stats)")
    ap.add_argument("--tag", default="", help="prefix for the run name")
    args = ap.parse_args()

    name = f"{args.tag}lam{args.dot_weight:g}_seed{args.seed}"
    args.out.mkdir(parents=True, exist_ok=True)
    ckpt_path, log_path = args.out / f"{name}.pt", args.out / f"{name}.json"
    if ckpt_path.exists() or log_path.exists():
        raise FileExistsError(f"{name} already exists in {args.out}; refusing to overwrite")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dev = torch.device(args.device)
    d, sp, sp_parts = load_parts(args.data)
    tr = sp["train"]
    stats = (
        float(d["y"][tr].mean()),
        float(d["y"][tr].std()),
        float(d["z"][tr].mean()),
        float(d["z"][tr].std()),
    )
    model = LearnedAdjoint(args.modes, args.width, args.layers).to(dev)
    model.set_stats(*stats)
    if args.init_from is not None:
        init = torch.load(args.init_from, map_location="cpu", weights_only=False)
        model.load_state_dict(init["model"])  # includes the normalisation buffers
        stats = tuple(init["stats"])
        print(f"warm start from {args.init_from} (best epoch {init['best_epoch']})", flush=True)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)

    x_t, y_t, z_t = (torch.from_numpy(d[k]).to(dev) for k in ("x", "y", "z"))
    gen = torch.Generator().manual_seed(args.seed)
    hist, best, t0 = [], (np.inf, -1), time.time()
    for ep in range(args.epochs):
        model.train()
        order = tr[torch.randperm(len(tr), generator=gen).numpy()]
        tot_sup = tot_dot = 0.0
        for s in range(0, len(order) - args.batch_size + 1, args.batch_size):
            b = torch.from_numpy(order[s : s + args.batch_size]).to(dev)
            pred = model(y_t[b])
            if args.loss == "rel":
                err = (pred - z_t[b]).flatten(1).pow(2).sum(1)
                sup = (err / z_t[b].flatten(1).pow(2).sum(1).clamp_min(1e-12)).mean()
            else:
                sup = (((pred - z_t[b]) / model.x_std) ** 2).mean()
            dot = adjoint_dot_loss(x_t[b], y_t[b], pred)
            loss = sup + args.dot_weight * dot
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot_sup += float(sup.detach())
            tot_dot += float(dot.detach())
        sched.step()
        nb = max(1, len(order) // args.batch_size)
        val = evaluate(model, d, sp["val"], dev)
        hist.append(
            {"epoch": ep, "train_sup": tot_sup / nb, "train_dot": tot_dot / nb, "val_rel_l2": val}
        )
        if ep % 10 == 0 or ep == args.epochs - 1:
            print(f"epoch {ep + 1}/{args.epochs}  train {tot_sup / nb:.4f}  val {val:.4f}  {time.time() - t0:.0f}s", flush=True)
        if val < best[0]:
            best = (val, ep)
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    torch.save(
        {
            "model": best_state,
            "arguments": vars(args) | {"data": [str(p) for p in args.data], "out": str(args.out),
                                       "init_from": str(args.init_from)},
            "stats": stats,
            "split": {k: v.tolist() for k, v in sp.items()},
            "split_parts": {n: {k: v.tolist() for k, v in s.items()} for n, s in sp_parts.items()},
            "best_epoch": best[1],
        },
        ckpt_path,
    )
    log = {
        "name": name,
        "n_params": n_params,
        "n": {k: len(v) for k, v in sp.items()},
        "best_epoch": best[1],
        "best_val_rel_l2": best[0],
        "test_rel_l2": evaluate(model, d, sp["test"], dev),
        "test_rel_l2_by_data": {n: evaluate(model, d, s["test"], dev) for n, s in sp_parts.items()},
        "seconds": time.time() - t0,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "history": hist,
    }
    log_path.write_text(json.dumps(log, indent=1))
    print(
        f"{name}: params {n_params}, best epoch {best[1]}, val {best[0]:.4f}, "
        f"test {log['test_rel_l2']:.4f}, {log['seconds']:.0f}s"
    )
    for n, v in log["test_rel_l2_by_data"].items():
        print(f"  test {n}: {v:.4f}")


if __name__ == "__main__":
    main()
