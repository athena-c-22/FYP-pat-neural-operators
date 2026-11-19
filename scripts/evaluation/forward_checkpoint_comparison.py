"""Compare the two handover FNO checkpoints without training or extracting weights.

Run with --checkpoint-archive PAT_checkpoints.zip --logs-archive PAT_logs_and_configs.zip.
Each run requires a new output directory. All metrics use unmasked physical measurements.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import platform
import time
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from pat_fno.data.mnist import centered_corr, rel_l2
from pat_fno.evaluation.gradient_comparison import fd_check, gradient_metrics
from pat_fno.models.fno import TinyFNO2d
from scripts.evaluation.adjoint_consistency import apply_adjoint, apply_forward, make_setting
from scripts.evaluation.gradient_comparison import POINTS, SEED, make_point, unit_directions
from scripts.smoke.run_reproducibility_check import _fno_predictions

ROOT = Path(__file__).resolve().parents[2]
SAMPLES = ROOT / "examples/sample_data/periodic_large_test_samples.npz"
MEMBERS = {
    "F0": "checkpoints/principal/results/mnist_medium/mnist_large_v1/"
    "periodic_theta89/fno_only/best.pt",
    "F5": "checkpoints/sample_efficiency/results/sample_efficiency/mnist_large_v1/"
    "periodic_theta89/fno_only/n50000_seed20260728/best.pt",
}


def read_checkpoint(archive, name):
    """Load an original checkpoint directly from the supplied archive."""
    with zipfile.ZipFile(archive) as z:
        raw = z.read(MEMBERS[name])
    return torch.load(io.BytesIO(raw), map_location="cpu", weights_only=False), raw


def surrogate(ck):
    """Build the physical-space map, preserving complex weights in float64."""
    a, n = ck["arguments"], ck["normalization"]
    if (a["scenario"], a["condition"], a["dataset"]) != (
        "fno_only",
        "periodic_theta89",
        "mnist_large_v1",
    ):
        raise ValueError("Incompatible checkpoint provenance")
    model = TinyFNO2d(a["modes"], a["modes"], a["width"], a["layers"])
    model.load_state_dict(ck["model"], strict=True)
    model.eval()
    for p in model.parameters():
        p.data = p.data.to(torch.complex128 if p.is_complex() else torch.float64)
        p.requires_grad_(False)

    def predict(x):
        batched = x.ndim == 3
        source = x[:, None] if batched else x[None, None]
        z = F.interpolate(source, size=(64, 91), mode="bilinear", align_corners=False)
        result = model((z - n["p0_mean"]) / n["p0_std"])[:, 0]
        result = result * n["data_std"] + n["data_mean"]
        return result if batched else result[0]

    return predict


def positive_step(residual, ag, fraction=1.0):
    """Constrain a quadratic line search to steps along minus the candidate gradient."""
    dot, den = float(np.vdot(residual, ag)), float(np.vdot(ag, ag))
    alpha = fraction * max(0.0, dot / den) if den > 0 else 0.0
    return alpha, 0.5 * float(np.sum((residual - alpha * ag) ** 2)), dot


def write_csv(path, rows):
    """Write a homogeneous metrics table."""
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def aggregate_results(out):
    """Save descriptive means, sample SDs and ranges without pooling unlike test points."""
    decomposition = []
    for path in sorted(out.glob("sample*_*.npz")):
        sample, point = path.stem.split("_", 1)
        with np.load(path) as arrays:
            ref = arrays["g_ref"]
            for model in MEMBERS:
                jacobian_error = arrays[f"{model}_g_jac"] - ref
                residual_error = arrays[f"{model}_g_ad"] - arrays[f"{model}_g_jac"]
                decomposition.append(
                    {
                        "sample": int(sample.removeprefix("sample")),
                        "point": point,
                        "model": model,
                        "jacobian_component_norm": float(np.linalg.norm(jacobian_error)),
                        "residual_component_norm": float(np.linalg.norm(residual_error)),
                        "total_error_norm": float(np.linalg.norm(arrays[f"{model}_g_ad"] - ref)),
                        "component_inner_product": float(np.vdot(jacobian_error, residual_error)),
                    }
                )
    write_csv(out / "decomposition.csv", decomposition)
    summary = []
    specifications = {
        "forward.csv": (("model",), ("rel_l2", "mse", "normalised_mse", "correlation")),
        "gradients.csv": (
            ("model", "point"),
            (
                "norm_ad",
                "norm_ref",
                "ad_cos",
                "ad_rel_l2",
                "ad_norm_ratio",
                "jac_same_residual_cos",
                "jac_same_residual_rel_l2",
                "res_only_cos",
            ),
        ),
        "steps.csv": (
            ("model", "point", "fraction_of_line_minimum"),
            ("initial_objective", "after_objective", "relative_reduction"),
        ),
    }
    for filename, (groups, metrics) in specifications.items():
        with (out / filename).open() as f:
            rows = list(csv.DictReader(f))
        for group in sorted({tuple(r[k] for k in groups) for r in rows}):
            selected = [r for r in rows if tuple(r[k] for k in groups) == group]
            for metric in metrics:
                values = np.array([float(r[metric]) for r in selected])
                values = values[np.isfinite(values)]
                if not len(values):
                    continue
                summary.append(
                    {
                        "table": filename,
                        "group": "/".join(group),
                        "metric": metric,
                        "n": len(values),
                        "mean": float(values.mean()),
                        "sample_sd": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                        "min": float(values.min()),
                        "max": float(values.max()),
                    }
                )
    write_csv(out / "summary.csv", summary)


def worker(task):
    """Evaluate both models at exactly the same point using fresh k-Wave calls."""
    archive, si, kind = task
    torch.set_num_threads(1)
    with np.load(SAMPLES) as data:
        p0 = data["initial_pressure"][si].astype(float)
        b = data["kwave_target"][si].astype(float)
    rng = np.random.default_rng(SEED + 100 * si + POINTS.index(kind))
    x = make_point(kind, p0, rng)
    s = make_setting("periodic_theta89")

    def kf(z):
        return apply_forward("kwave", z, s).astype(float)

    def ka(z):
        return apply_adjoint("kwave", z, s).astype(float)

    r = kf(x) - b
    ref = ka(r)
    j0 = 0.5 * float(np.sum(r**2))
    arrays = {"x": x, "r_true": r, "g_ref": ref}
    gradients = {"reference": ref}
    rows, fdrows, steps = [], [], []
    for name in MEMBERS:
        predict = surrogate(read_checkpoint(archive, name)[0])
        xt = torch.tensor(x, requires_grad=True)
        y = predict(xt)
        rs = y.detach().numpy() - b
        gad = torch.autograd.grad(y, xt, torch.tensor(rs), retain_graph=True)[0].numpy()
        gjac = torch.autograd.grad(y, xt, torch.tensor(r))[0].numpy()
        gres = ka(rs)
        gradients[name] = gad
        arrays.update(
            {
                f"{name}_{k}": v
                for k, v in {
                    "prediction": y.detach().numpy(),
                    "g_ad": gad,
                    "g_jac": gjac,
                    "g_res": gres,
                    "r_sur": rs,
                }.items()
            }
        )
        row = {
            "sample": si,
            "point": kind,
            "model": name,
            "J_A": j0,
            "J_F": 0.5 * float(np.sum(rs**2)),
            "norm_ref": float(np.linalg.norm(ref)),
            "norm_ad": float(np.linalg.norm(gad)),
            "residual_rel_error": float(np.linalg.norm(rs - r) / (np.linalg.norm(r) + 1e-30)),
        }
        for key, g in (("ad", gad), ("jac_same_residual", gjac), ("res_only", gres)):
            metrics = gradient_metrics(g, ref)
            if kind == "truth":
                metrics = dict.fromkeys(metrics, float("nan"))
            row.update({f"{key}_{k}": v for k, v in metrics.items()})
        rows.append(row)
        # Independent scalar-objective finite differences include resize and normalisation.
        dirs = [torch.tensor(v) for v in unit_directions(x.shape, np.random.default_rng(SEED + si))]
        check = fd_check(
            lambda z, predict=predict: 0.5 * ((predict(z) - torch.tensor(b)) ** 2).sum(),
            torch.tensor(x),
            dirs,
            (1e-3, 1e-4, 1e-5),
        )
        for di in range(len(dirs)):
            for hi, h in enumerate(check["steps"]):
                fdrows.append(
                    {
                        "sample": si,
                        "point": kind,
                        "model": name,
                        "direction": di,
                        "h": h,
                        "ad": check["ad"][di],
                        "fd": check["fd"][di, hi],
                        "rel_error": check["rel_err"][di, hi],
                    }
                )
    for name, g in gradients.items():
        ag = kf(g)
        for fraction in (0.1, 1.0):
            alpha, predicted, dot = positive_step(r, ag, fraction)
            # A real forward call checks linearity/float32 effects after the proposed update.
            after = 0.5 * float(np.sum((kf(x - alpha * g) - b) ** 2)) if alpha else j0
            steps.append(
                {
                    "sample": si,
                    "point": kind,
                    "model": name,
                    "fraction_of_line_minimum": fraction,
                    "alpha": alpha,
                    "initial_objective": j0,
                    "after_objective": after,
                    "linear_prediction": predicted,
                    "relative_reduction": (j0 - after) / j0 if kind != "truth" else np.nan,
                    "improved": after < j0,
                    "worsened": after > j0,
                    "reference_directional_derivative": -dot,
                    "is_descent": dot > 0,
                }
            )
    return si, kind, rows, fdrows, steps, arrays


def forward_comparison(models, data, out):
    """Measure and plot the paired four-sample forward comparison."""
    rows, predictions = [], {}
    for name, ck in models.items():
        predict = surrogate(ck)
        predictions[name] = np.stack(
            [
                predict(torch.tensor(x.astype(float))).detach().numpy()
                for x in data["initial_pressure"]
            ]
        )
        for si, (p, b) in enumerate(zip(predictions[name], data["kwave_target"], strict=True)):
            error = p - b
            rows.append(
                {
                    "model": name,
                    "sample": si,
                    "source_index": int(data["source_index"][si]),
                    "rel_l2": rel_l2(p, b),
                    "mse": float(np.mean(error**2)),
                    "normalised_mse": float(
                        np.mean(error**2) / ck["normalization"]["data_std"] ** 2
                    ),
                    "correlation": centered_corr(p, b),
                }
            )
    baseline = _fno_predictions(
        data["initial_pressure"], ROOT / "examples/checkpoints/fno_only_periodic_large.pt"
    )
    discrepancy = float(np.linalg.norm(predictions["F0"] - baseline) / np.linalg.norm(baseline))
    write_csv(out / "forward.csv", rows)
    np.savez_compressed(out / "forward.npz", target=data["kwave_target"], **predictions)
    vmax = max(
        np.max(np.abs(data["kwave_target"])), *(np.max(np.abs(p)) for p in predictions.values())
    )
    emax = max(np.max(np.abs(p - data["kwave_target"])) for p in predictions.values())
    fig, axes = plt.subplots(4, 5, figsize=(17, 11), constrained_layout=True)
    for si, axrow in enumerate(axes):
        images = [
            data["kwave_target"][si],
            predictions["F0"][si],
            predictions["F5"][si],
            np.abs(predictions["F0"][si] - data["kwave_target"][si]),
            np.abs(predictions["F5"][si] - data["kwave_target"][si]),
        ]
        for j, (ax, img) in enumerate(zip(axrow, images, strict=True)):
            im = ax.imshow(
                img,
                aspect="auto",
                cmap="RdBu_r" if j < 3 else "magma",
                vmin=-vmax if j < 3 else 0,
                vmax=vmax if j < 3 else emax,
            )
            ax.set(
                title=["k-Wave target", "F0", "F5", "F0 absolute error", "F5 absolute error"][j],
                xlabel="Time index",
                ylabel=f"Sample {si}: sensor index",
            )
            fig.colorbar(im, ax=ax)
    fig.savefig(out / "forward.png", dpi=130)
    plt.close(fig)
    return rows, discrepancy


def main():
    """Run the paired comparison into a new, isolated output directory."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint-archive", type=Path, required=True)
    ap.add_argument("--logs-archive", type=Path, required=True)
    ap.add_argument(
        "--output-dir", type=Path, default=Path("outputs/forward_checkpoint_comparison")
    )
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(SEED)
    models, provenance = {}, {}
    with zipfile.ZipFile(args.checkpoint_archive) as z:
        for name, member in MEMBERS.items():
            ck, raw = read_checkpoint(args.checkpoint_archive, name)
            surrogate(ck)  # Strict compatibility check before creating outputs.
            models[name] = ck
            provenance[name] = {
                "member": member,
                "sha256": hashlib.sha256(raw).hexdigest(),
                "arguments": ck["arguments"],
                "normalization": ck["normalization"],
                "setting": ck["setting"],
                "historical_metrics": json.loads(z.read(member.replace("best.pt", "metrics.json"))),
                "history": json.loads(z.read(member.replace("best.pt", "history.json"))),
            }
        index = z.read("checkpoints/CHECKPOINT_INDEX.csv").decode()
    bundled_hash = hashlib.sha256(
        (ROOT / "examples/checkpoints/fno_only_periodic_large.pt").read_bytes()
    ).hexdigest()
    if bundled_hash != provenance["F0"]["sha256"]:
        raise ValueError("Archive F0 does not match bundled F0")
    if models["F0"]["normalization"] != models["F5"]["normalization"]:
        raise ValueError("Normalisations differ; specify a common metric denominator")
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=False)
    with zipfile.ZipFile(args.logs_archive) as z:
        for name in ("mnist_pat.yaml", "forward_operators.yaml", "sample_efficiency.yaml"):
            (out / name).write_bytes(z.read("training_logs/configs/" + name))
    (out / "CHECKPOINT_INDEX.csv").write_text(index)
    data = dict(np.load(SAMPLES))
    start = time.monotonic()
    forward, baseline_difference = forward_comparison(models, data, out)
    meta = {
        "seed": SEED,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "checkpoint_archive": str(args.checkpoint_archive.resolve()),
        "logs_archive": str(args.logs_archive.resolve()),
        "provenance": provenance,
        "samples_sha256": hashlib.sha256(SAMPLES.read_bytes()).hexdigest(),
        "source_indices": data["source_index"].tolist(),
        "F0_float64_vs_existing_float32_rel_l2": baseline_difference,
        "reference": "fresh k-Wave forward and approximate k-Wave adjoint; Euclidean sums",
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2, default=str))
    print("Forward means:", flush=True)
    for name in MEMBERS:
        print(
            name,
            {
                k: float(np.mean([r[k] for r in forward if r["model"] == name]))
                for k in ("rel_l2", "mse", "normalised_mse", "correlation")
            },
            flush=True,
        )
    tasks = [(args.checkpoint_archive, si, kind) for si in range(4) for kind in POINTS]
    rows, fdrows, steps = [], [], []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(worker, t) for t in tasks]
        for future in as_completed(futures):
            si, kind, rr, ff, ss, arrays = future.result()
            rows.extend(rr)
            fdrows.extend(ff)
            steps.extend(ss)
            np.savez_compressed(out / f"sample{si}_{kind}.npz", **arrays)
            write_csv(
                out / "gradients.csv",
                sorted(rows, key=lambda r: (r["sample"], r["point"], r["model"])),
            )
            write_csv(out / "finite_differences.csv", fdrows)
            write_csv(out / "steps.csv", steps)
            print(f"Completed sample {si}, {kind}; {time.monotonic() - start:.0f}s", flush=True)
    fig, axes = plt.subplots(4, 5, figsize=(15, 12), constrained_layout=True)
    for axrow, kind in zip(axes, POINTS, strict=True):
        d = np.load(out / f"sample0_{kind}.npz")
        keys = ["g_ref", "F0_g_ad", "F5_g_ad", "F0_g_jac", "F5_g_jac"]
        vmax = max(np.abs(d[k]).max() for k in keys)
        for ax, k in zip(axrow, keys, strict=True):
            im = ax.imshow(d[k], cmap="RdBu_r", vmin=-vmax, vmax=vmax)
            ax.set_title(f"{kind}: {k}")
            fig.colorbar(im, ax=ax)
    fig.savefig(out / "gradients.png", dpi=130)
    plt.close(fig)
    aggregate_results(out)
    meta["runtime_s"] = time.monotonic() - start
    (out / "meta.json").write_text(json.dumps(meta, indent=2, default=str))


if __name__ == "__main__":
    main()
