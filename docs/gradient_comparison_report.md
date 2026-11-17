# Stage 3 — Reconstruction-gradient comparison

Date: 2026-11-16. Everything below is **measured** unless marked *hypothesis*. All data are the 4 bundled synthetic MNIST samples; no patient data.

## 1. Question and why it matters

A learned forward operator `G ≈ A` (here the FNO) is useful in reconstruction only if it can be *differentiated*: gradient descent on

    J_G(x) = ½‖G(x) − b‖²      ∇J_G(x) = G′(x)ᵀ (G(x) − b)        (autograd)

must point where the true problem points:

    J_A(x) = ½‖A x − b‖²       ∇J_A(x) = A*(A x − b)               (k-Wave forward / adjoint)

Matching *outputs* (forward error, Stage "baseline": rel. L2 0.39) does not imply matching *gradients*: a network can fit `A x` well yet have a Jacobian unrelated to `A`. Here `A` = k-Wave and `A*` = the k-Wave adjoint, which Stage 2 showed is an approximate adjoint (accurate to ≈1 % on smooth inputs, poor on white noise).

Norms are plain Euclidean (no `dx·dy`, `dy·dt` weights), as justified in Stage 2.

## 2. Setup (code map)

| Item | Where |
|---|---|
| Metrics (`gradient_metrics`, `directional_fd`, `fd_check`, `quadratic_step_decrease`) | `src/pat_fno/evaluation/gradient_comparison.py` |
| Tests on known operators (6) | `tests/unit/test_gradient_comparison.py` |
| Experiment | `scripts/evaluation/gradient_comparison.py` (`python -m scripts.evaluation.gradient_comparison`, ~35 min CPU with 8 workers; `--skip-kwave` runs the FD part in <1 min) |
| Outputs | `outputs/gradient_comparison/` (`results.csv`, `fd_check.csv`, `summary.json`, `gradients.npz`, 3 PNG, `meta.json`, `run.log`) |

* Surrogate `G`: `examples/checkpoints/fno_only_periodic_large.pt` (TinyFNO2d, 1 epoch, periodic 89°). `G(x)` = bilinear resize 64×64→64×91, z-score, FNO, un-z-score. Condition `periodic_theta89` only (the checkpoint's condition).
* Data `b` = the stored k-Wave `kwave_target` of each sample (noise-free). Fresh k-Wave forward of `p0` reproduces it to 3·10⁻⁵ relative (float32 noise), so `J_A(truth) ≈ 10⁻⁷`.
* Seed 20260728 (per-job seeds `SEED + 100·sample + point`). Python 3.10.20, torch 2.11.0+cpu, numpy 2.2.6, k-Wave-python 0.5.0rc1.
* Test points per sample: `truth` (`x = p0`), `half` (`0.5·p0`), `smooth_pert` (`p0` + 0.5·std·smooth field), `noisy` (`p0` + 0.1·std·white noise). 16 points.

Four gradients per point (`r_true = Ax − b`, `r_sur = G(x) − b`, `J = G′(x)`):

| Name | Formula | Isolates |
|---|---|---|
| `g_ref` | `A* r_true` | reference |
| `g_ad` | `Jᵀ r_sur` | what reconstruction with `G` actually uses |
| `g_res` | `A* r_sur` | only the *residual* comes from the surrogate |
| `g_jac` | `Jᵀ r_true` | only the *operator* (Jacobian) comes from the surrogate |

## 3. A code issue found and fixed (flag for review)

`SpectralConv2d.forward` allocated its output buffer with `dtype=torch.cfloat`, so a float64 model silently computed in float32 (and `module.to(float64)` additionally *drops the imaginary part* of complex weights). Finite-difference checks need float64. I changed that one line in `src/pat_fno/models/fno.py` to `dtype=transformed.dtype`. For float32 inputs `transformed.dtype` is `complex64 == cfloat`, so float32 behaviour is unchanged (a unit test checks both dtypes). The experiment casts complex weights manually. **This is the only existing source file modified in Stage 3.**

## 4. Part 1 — is autograd through the FNO correct? (yes)

Directional derivative `⟨∇J_G, v⟩` from autograd vs central differences `(J(x+hv) − J(x−hv))/2h`, float64, 16 points × 4 unit directions (2 smooth, 2 white):

| h | 1e-1 | 1e-2 | 1e-3 | 1e-4 | 1e-5 | 1e-6 |
|---|---|---|---|---|---|---|
| median rel. error | 1.0e-5 | 1.0e-7 | 1.0e-9 | 1.9e-10 | 1.7e-9 | 2.5e-8 |

Error falls ∝ h² (truncation), bottoming near 2·10⁻¹⁰, then rises (round-off ∝ 1/h): the textbook shape (`fd_check.png`). Additional checks: autograd of the scalar `J_G` equals the explicit `Jᵀ r_sur` exactly (diff 0); float32 gradient differs from float64 by ≈10⁻⁶ relative. **Conclusion: `g_ad` is the exact gradient of `J_G`; any disagreement with `g_ref` is a property of the surrogate, not of autograd.**

## 5. Part 2 — comparison with the k-Wave reference

Per-point values (`ad_*` = `g_ad` vs `g_ref`); range over the 4 samples:

| Point | ‖g_ref‖ | ‖g_ad‖ | cos(g_ad, g_ref) | rel. L2 | best scale | decrease of exact line search, AD / ref |
|---|---|---|---|---|---|---|
| truth | ≈1e-4 (zero) | 3.5–6.3 | undefined | undefined | – | – |
| half | 9.4–17.5 | 9.4–21.2 | 0.65–0.73 | 0.76–0.90 | 0.66–0.89 | 0.54–0.63 |
| smooth_pert | 3.0–5.6 | 3.6–7.4 | 0.14–0.34 | 1.05–1.72 | 0.22–0.41 | 0.03–0.17 |
| noisy | 1.1–1.6 | 3.5–6.4 | −0.03–+0.01 | 2.4–4.1 | ≈0 | ≈10⁻⁴ (0.004–0.13 %) |

Decomposition of the loss of agreement (cosine with `g_ref`):

| Point | `g_ad` (both wrong) | `g_res` (only residual wrong) | `g_jac` (only Jacobian wrong) |
|---|---|---|---|
| half | 0.65–0.73 | 0.93–0.98 | 0.66–0.74 |
| smooth_pert | 0.14–0.34 | −0.03–0.19 | 0.25–0.62 |
| noisy | ≈0 | −0.07–0.04 | 0.07–0.16 |

Figure `gradient_images.png` (sample 0, `half`) shows `g_ref` tracing the digit, `g_ad` a blurred, ringing, digit-shaped but stripe-contaminated field, and `g_jac` the same stripes as `g_ad`.

### Findings

1. **Truth is not a stationary point of `J_G`.** At `x = p0` the true gradient is 0 but `‖g_ad‖ = 3.5–6.3`, comparable to the reference gradient magnitude at `half` (9–17). Descent on `J_G` is pulled toward a point displaced from the truth by the surrogate's forward error (rel. 0.39). This is a *bias*, independent of autograd correctness.
2. **The same bias dominates at `noisy`.** There `‖g_ad‖` (3.5–6.4) is ≈3× `‖g_ref‖` (1.1–1.6) and the cosine is ≈0: the gradient is mostly the truth-point bias, not information about the perturbation.
3. **At `half` (a smooth, in-distribution-like scaling) the direction is partly right:** cosine ≈0.7, and a line search along `−g_ad` achieves ≈55–60 % of the decrease available along `−g_ref`. The loss is shared between the residual (`g_res` cos ≈0.95 — the residual mismatch alone costs little here) and the Jacobian (`g_jac` cos ≈0.7 — the Jacobian is the weaker part).
4. **Jacobian quality degrades away from the training manifold:** `g_jac` cosine 0.7 → 0.25–0.6 → 0.1 from `half` to `smooth_pert` to `noisy`. *Hypothesis:* the FNO was fit on outputs only, so nothing constrains `G′(x)` off the data manifold. Not tested here.
5. **Descent holds only weakly:** the exact line-search step along `−g_ad` is positive (a descent direction for `J_A`) at 10 of the 12 non-truth points; at 2 of the 4 `noisy` points it is negative, i.e. `−g_ad` is an *ascent* direction for the true objective. Where it descends, the achieved decrease collapses to 3–17 % (`smooth_pert`) and ≈0.004–0.13 % (`noisy`). Sign agreement of directional derivatives over only 4 directions is 0.25–1 (too few directions for more than "no better than chance at `noisy`").
6. **The reference itself is accurate only on smooth directions.** FD of `J_A` through k-Wave vs `⟨g_ref, v⟩`: the smooth direction agrees to 0.2–6 % at all 12 non-truth points (e.g. 2.043 vs 2.048; 0.415 vs 0.413); the white direction disagrees by 2–45 % and at one point (sample 3, `noisy`) has the wrong sign (+0.030 vs −0.006). This repeats Stage 2 and bounds how much we can conclude from low cosines against `g_ref` at `noisy`; findings 1–2 do not depend on it, because they rest on `‖g_ad‖` at the truth point where `A*` is applied to a near-zero residual.

## 6. How to read this

* **cos** says whether the direction is useful; **rel. L2** also includes magnitude (relative error > 1 means worse than predicting zero); **best scale** isolates a pure step-size error; **line-search decrease ratio** translates direction quality into progress on the *true* objective.
* "Truth" rows are deliberately reported as bias, not as errors relative to a zero reference.

## 7. Limitations

* One checkpoint (1 epoch, weak: forward rel. L2 0.39), 4 samples, 4 test points each, periodic condition only. Gradient quality of a well-trained model is **not** established by this.
* Test points are synthetic displacements of the truth, not iterates of an actual reconstruction run.
* Reference `A*` is approximate (Stage 2). `A*` was never the exact transpose of the k-Wave forward.
* 4 random directions per point for sign agreement; 2 for the k-Wave FD check (FD at `truth` is below float32 noise and uninformative).
* No gradient clipping, no noise on `b`, no regularisation.
* Part 2 took 35 min. A benchmark run afterwards (Stage 4 preparation) measured ≈6.7 s per k-Wave forward+adjoint pair with **no speed-up from parallel worker processes** (1 worker 6.7 s/pair, 4 workers 7.0, 8 workers 6.6; the backend is single-threaded per call). My earlier explanation ("oversubscribed cores") was an unverified guess and is withdrawn; the actual cause of the flat throughput (*hypothesis*: shared disk I/O or memory bandwidth) is not isolated. Practical consequence: k-Wave cost is ≈ calls × 3–5 s regardless of `--workers`.

## 8. Decisions / questions for Marta

1. Is a gradient-error criterion (cosine / line-search decrease) the right thing to optimise when training `G`, i.e. should training include a Jacobian-matching or adjoint-consistency term (Stage 4)?
2. Is an *approximate* k-Wave adjoint an acceptable gradient reference, or should the reference be the exact autograd transpose of a differentiable solver?
3. Should reconstruction use `G` only for the Jacobian (`g_jac`-like: true residual from k-Wave, cheap Jacobian from `G`) or for both? Measured: the residual from `G` costs little at `half` but hurts when the surrogate bias matters (truth/noisy).

## 9. Recommendation

Do not use the bundled checkpoint for gradient-based reconstruction claims. For Stage 4, the gradient diagnostics here (`gradient_metrics`, line-search ratio) are ready to evaluate a learned adjoint; a learned adjoint `G_φ(y)` can be compared directly against `g_ref` at the `half`/`smooth_pert` points.
