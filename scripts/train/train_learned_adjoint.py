"""Train a learned adjoint ``G_phi: y -> x`` from (x, y = A x, z = A* y) triples.

Loss = MSE(G(y), z) in normalised units  +  lambda * adjoint_dot_loss(x, y, G(y)).

``lambda = 0`` is plain supervised regression onto the reference adjoint ``z``; ``lambda > 0``
adds the adjoint-consistency term, which uses only the pairs ``(x_i, y_i)`` and pulls ``G``
towards the transpose of ``A`` instead of towards the (approximate) reference adjoint.

Split: a seeded permutation (``--seed-split``, fixed) -> 70 % train / 15 % val / 15 % test.
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
    ap.add_argument("--data", type=Path, default=Path("outputs/learned_adjoint/data"))
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
    args = ap.parse_args()

    name = f"lam{args.dot_weight:g}_seed{args.seed}"
    args.out.mkdir(parents=True, exist_ok=True)
    ckpt_path, log_path = args.out / f"{name}.pt", args.out / f"{name}.json"
    if ckpt_path.exists() or log_path.exists():
        raise FileExistsError(f"{name} already exists in {args.out}; refusing to overwrite")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dev = torch.device(args.device)
    d = load_data(args.data)
    sp = split_indices(len(d["x"]))
    tr = sp["train"]
    stats = (
        float(d["y"][tr].mean()),
        float(d["y"][tr].std()),
        float(d["z"][tr].mean()),
        float(d["z"][tr].std()),
    )
    model = LearnedAdjoint(args.modes, args.width, args.layers).to(dev)
    model.set_stats(*stats)
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
        if val < best[0]:
            best = (val, ep)
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    torch.save(
        {
            "model": best_state,
            "arguments": vars(args) | {"data": str(args.data), "out": str(args.out)},
            "stats": stats,
            "split": {k: v.tolist() for k, v in sp.items()},
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


if __name__ == "__main__":
    main()
