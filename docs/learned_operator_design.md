# Stage 4 — Learned adjoint, learned inverse, joint training

Date: 2026-11-16. Status of each part: **4A learned adjoint — designed, implemented, small CPU experiment run (§5). 4B learned inverse — design only. 4C joint training — design only (nothing launched).** Measured results are in §5; everything else is design or hypothesis.

## 1. Why we need an adjoint at all

Reconstruction by gradient methods minimises `J(x) = ½‖F(x) − b‖²`, whose gradient needs `Fᵀ`/`A*` applied to a data residual. Three ways to get it:

| Route | Gradient | Status in this repo |
|---|---|---|
| k-Wave adjoint `A*` | `A*(Ax − b)` | exists; ≈1 % adjoint on smooth data, poor on rough data (Stage 2); ≈2.5 s per call |
| Autograd through learned forward `F_θ` | `F′ᵀ(F(x) − b)` | exact gradient of the *surrogate* objective, but the surrogate's Jacobian is poor and the surrogate has bias (Stage 3: cos 0.65–0.73 → ≈0, truth not stationary) |
| Fourier adjoint of `Ax − b` (c convention) | `A*_F((Ax − b)/c)` | exists; not an adjoint pair (Stage 2) yet cos 0.93 with `g_ref` (§5) |
| **Learned adjoint `G_φ`** | `G_φ(F_θ(x) − b)` or `G_φ(Ax − b)` | **this stage** |

A learned adjoint is a *cheap (one FNO pass), differentiable-free* surrogate for `A*`. Its attraction after Stage 3: it can be trained directly on the object we need (the map residual → gradient), instead of hoping that the Jacobian of a forward surrogate happens to approximate `A*`.

## 2. 4A — Learned adjoint (primary)

**Model.** `G_φ: y ∈ R^{64×91} → x ∈ R^{64×64}` (`src/pat_fno/models/learned_adjoint.py`): bilinear resize 64×91→64×64, z-score, `TinyFNO2d` (modes 8, width 16, 3 layers = 100,337 parameters, the same as the forward surrogate), de-normalise.

**Training data.** Triples `(x_i, y_i = A x_i, z_i = A* y_i)` from k-Wave (periodic 89°). Because `A` and `A*` are linear, data for signed images `x` (differences of phantoms) are exactly what the optimiser feeds the adjoint (`A x − b = A(x − p0)`), so the training distribution includes signed phantoms.

**Losses** (`scripts/train/train_learned_adjoint.py`):

* *Supervised:* `L_sup = ‖G_φ(y) − z‖² / σ_z²` — regress onto the k-Wave adjoint. Ceiling: `G_φ` can at best become `A*`, which is itself only an approximate adjoint of `A`.
* *Adjoint consistency (dot-product):* for training pairs, `⟨A x_i, y_j⟩ = ⟨y_i, y_j⟩` must equal `⟨x_i, G_φ(y_j)⟩`; `L_dot = mean_ij ((⟨y_i,y_j⟩ − ⟨x_i,G_φ y_j⟩) / (‖y_i‖‖y_j‖))²` (the Cauchy-Schwarz-normalised `cs_rel` of Stage 2, squared; `adjoint_dot_loss`). It needs **no extra k-Wave calls** and pulls `G_φ` toward the transpose of `A` rather than toward `A*`. Marta's un-normalised `|⟨Ax,y⟩−⟨x,A*y⟩|²/‖y‖²` is *not* used for training because it depends on the scale of the data; it remains available as the `marta` metric.
* Total: `L = L_sup + λ L_dot`. Variants: `λ ∈ {0, 1, 10}`.

**Why both losses, mathematically.** `L_sup` fixes `G_φ` on the range of the sampled `y_i` toward `A*y_i`. `L_dot` only constrains the *component of `G_φ(y_j)` along the span of `{x_i}`* (it sees `G_φ(y_j)` only through inner products with training images), so it cannot by itself determine `G_φ`; it is a regulariser, not a replacement. *Hypothesis:* with few samples `L_dot` mostly trades fit to `z` for adjointness on the training span.

**Evaluation (reuses Stages 2/3):** rel. L2/cosine against `A*y`; dot-product metrics (`sym_rel`, `cs_rel`, ratio, `marta`) on held-out pairs; gradient cosine and exact line-search decrease at the 16 Stage 3 points, directly comparable with the Stage 3 autograd-through-forward numbers.

## 3. 4B — Learned inverse (optional; design only)

`H_ψ: y → x`, trained with `L_inv = ‖H_ψ(A x) − x‖²`. Points to settle before any experiment:

* The problem is ill-posed for limited angle (89° periodic is mild; 45° PML is severe: the Fourier inverse already reaches only rel. L2 ≈0.94 on the bundled samples), so a regression network learns a *prior*, not an inverse, and will hallucinate stroke-like structure (it is trained on MNIST-like data).
* Marta's inverse-consistency expression and the normalised variant `‖A⁻¹(Ax) − x‖² / ‖A‖₂²` are **different quantities and I have not assumed which is intended**. `‖A‖₂` would have to be estimated by power iteration, which needs `A*`; with the approximate k-Wave adjoint this is itself approximate (`‖A‖₂² ≈ λ_max(A*A)` only if `A*` is accurate).
* Cycle consistency `‖A H_ψ(y) − y‖²` is cheap only with the forward surrogate `F_θ` (one pass) or k-Wave (≈5 s).
* Recommendation: do 4A first; revisit 4B only if Marta wants direct reconstruction as a comparison baseline.

## 4. 4C — Joint training design (not run)

Networks `F_θ` (forward), `G_φ` (adjoint), optionally `H_ψ` (inverse). Candidate loss

    L = L_fwd(θ)  +  λ_sup L_sup(φ)  +  λ_dot L_dot(θ, φ)  +  λ_vjp L_vjp(θ, φ)

* `L_fwd = ‖F_θ(x) − A x‖²` (existing).
* `L_dot(θ, φ) = E_{x,y} (⟨F_θ x, y⟩ − ⟨x, G_φ y⟩)² / (‖F_θ x‖‖y‖)²` with `x` any images and `y` any data-shaped arrays — **no labels needed**, so it can use unlimited unlabeled `x` (and random or realistic `y`). Danger: `F_θ = G_φ = 0` minimises it, hence it must be anchored by `L_fwd`/`L_sup`; and a *consistent but wrong* pair (adjoint of the wrong operator) also minimises it — it enforces mutual consistency, not correctness. *Hypothesis*, not tested.
* `L_vjp(θ, φ) = E ‖F_θ′(x)ᵀ y − G_φ(y)‖²` ties the adjoint to the *Jacobian-transpose* of the forward network. Targets exactly the Stage 3 weakness (Jacobian quality), but needs double backpropagation (≈3× the cost per step).
* Schedule: pre-train `F_θ` and `G_φ` separately (supervised), then add `L_dot` and `L_vjp` with small `λ`, monitoring Stage 2/3 metrics on held-out data rather than the training loss.

**Compute (estimates, not measurements except where noted).** k-Wave labels: ≈6.7 s per (forward, adjoint) pair measured, not reduced by parallel workers → 10⁴ pairs ≈ 19 h CPU; this, not network training, is the bottleneck. Network training of `F_θ`/`G_φ` at this size is minutes per epoch on CPU for hundreds of samples; for the full MNIST scale (the existing forward-training setup, 1 epoch) a GPU is appropriate. *Prepared, not launched:* the same `train_learned_adjoint.py` runs on a GPU machine with `--device cuda --data <dir with shard_*.npz>`. No GPU job has been started.

**Decisions for Marta.** (1) Is the target the k-Wave `A*` (approximate) or the exact transpose of `A`? `L_dot` implements the latter. (2) Is joint training worth the k-Wave label cost, or should the adjoint be trained from the forward surrogate's Jacobian (`L_vjp`) alone? (3) Is a learned inverse a required comparison?

## 5. Small CPU experiment

**Setup (measured, reproducible).** Data: 200 k-Wave triples from `scripts/data/generate_adjoint_pairs.py` (seed 20260728; strokes/blobs/signed-difference phantoms; ≈8 s per pair, 24 min total), split 140 train / 30 val / 30 test (seeded permutation, indices stored in each checkpoint). Training: `scripts/train/train_learned_adjoint.py`, Adam lr 2e-3 with cosine decay, batch 16, ≤150 epochs, best-validation checkpoint; 3 variants (λ = 0, 1, 10) × 3 seeds (0, 1, 2), ≈3 min each on CPU. Evaluation: `scripts/evaluation/learned_adjoint_eval.py --linesearch`. Outputs in `outputs/learned_adjoint/` (`runs/`, `eval_results.csv`, `eval_summary.json`, `training_curves.png`, `test_example.png`, `gradient_cosine.png`, `odd_symmetrised_check.{py,txt}`). Environment: Python 3.10.20, torch 2.11.0+cpu, numpy 2.2.6, k-Wave-python 0.5.0rc1.

**(a) Accuracy against the k-Wave adjoint on the held-out synthetic test split (30 pairs).**

| Model | rel. L2 to `A*y` | cosine | notes |
|---|---|---|---|
| predict 0 | 1.00 | – | |
| Fourier adjoint (c convention) | 0.437 (0.347 after best scalar 1.40) | 0.937 | existing code, no training |
| learned, λ=0 (3 seeds) | 0.740 / 0.759 / 0.746 | 0.67 | best epoch 35–45 of 150: overfits |
| learned, λ=1 | 0.734 / 0.758 / 0.747 | 0.67 | |
| learned, λ=10 | 0.752 / 0.772 / 0.786 | 0.65 | best epoch 62–98 |

The learned adjoints are **worse than the untrained Fourier adjoint** and only moderately better than predicting zero. The seed-to-seed spread (≈0.02) is as large as the differences between λ values, so **no benefit of `L_dot` is demonstrated** (λ=10 is, if anything, slightly worse).

**(b) Dot-product test on the same 30 test images/data (870 ordered pairs, medians; Stage 2 metrics).**

| `B` | `sym_rel` | `cs_rel` | ratio `⟨Ax,y⟩/⟨x,By⟩` | `marta` |
|---|---|---|---|---|
| k-Wave `A*` (reference) | 0.0024 | 0.0009 | 1.002 | 8e-5 |
| Fourier adjoint (c conv.) | 0.229 | 0.066 | 1.49 | 0.46 |
| learned (9 models) | 0.19–0.22 | 0.065–0.072 | 0.93–0.99 | 0.40–0.51 |

Adjoint-consistency of the learned adjoints is **indistinguishable from the Fourier adjoint** (`cs_rel` ≈0.07) and ≈70× worse than the k-Wave reference in `cs_rel`; the learned ratio is closer to 1 (≈0.95 vs 1.49), i.e. their scale is better, their shape is not.

**(c) Gradient at the 16 Stage 3 points (bundled MNIST samples: out-of-distribution for a model trained on procedural phantoms).** Mean cosine with `g_ref = A*(Ax−b)`; line-search column = decrease of `J_A` along `−g` divided by that along `−g_ref` (seed 0 of each variant; Fourier and Stage 3 shown for context).

| Direction | cos half | cos smooth_pert | cos noisy | line-search half / smooth / noisy | `‖g‖` at truth (ideal 0) |
|---|---|---|---|---|---|
| Fourier adjoint of `Ax−b` (c conv.) | **0.93** | **0.90** | **0.63** | **0.99 / 1.01 / 1.07** | 4e-4 |
| Stage 3: autograd through forward FNO | 0.69 | 0.26 | −0.01 | ≈0.55–0.63 / 0.03–0.17 / ≈1e-4 | 4.7 |
| learned λ=0 (seed mean) | 0.53 | 0.18 | 0.02 | 0.28 / 0.07 / 0.002 | 2.4–3.4 |
| learned λ=1 | 0.54 | 0.21 | 0.02 | 0.35 / 0.11 / 0.0007 | 2.1–3.3 |
| learned λ=10 | 0.54 | 0.19 | 0.03 | 0.33 / 0.08 / 0.003 | 2.9–3.1 |
| learned, odd-symmetrised (no retraining) | 0.74–0.75 | 0.29–0.30 | 0.06–0.07 | not run | 0 by construction |

For the learned models `−g` is an ascent direction at 1 of 4 `noisy` points (every seed-0 model); never for the Fourier adjoint.

### What these results show and do not show

1. **Negative result for learned adjoints at this data scale.** With 140 training pairs a 100k-parameter FNO overfits within ≈40 epochs and generalises worse than an existing analytic operator. Nothing here says a learned adjoint cannot work with thousands of pairs; it says 200 pairs are far from enough and that λ did not rescue it.
2. **The Fourier adjoint is a much better descent direction than Stage 2 suggested** (*measured*): cos 0.93 and ≈100 % of the reference line-search decrease, even though it is not an adjoint pair in the dot-product sense (scale ×1.4–1.5 and a 0.35 shape residual after scaling, Stage 2). Reconciling the two: a descent method with a line search (or any step-size rule) is insensitive to scale, so a direction with cosine 0.93 is nearly as useful as the reference; a fixed-step method is not. The line-search ratio slightly above 1 (1.01, 1.07) is possible because `A*` is only an approximate adjoint, so the exact `A⊤(Ax−b)` direction is not the k-Wave one; *hypothesis*, not tested.
3. **Zero-input bias.** The learned adjoint maps a zero residual to a non-zero image (`‖G(0)‖` = 2.1–3.4), the same pathology as Stage 3's autograd gradient at the truth. Removing the even part of `G` (a linear map is odd) after training raised the cosine at `half` from 0.54 to 0.74–0.75 with no retraining (*measured*; ad hoc script saved as `odd_symmetrised_check.py`). It is only a diagnostic; *design implication (hypothesis)*: build the adjoint without biases / with exact oddness so it is linear by construction.
4. The dot-product loss did not improve the dot-product metrics (`cs_rel` 0.065–0.072 for all λ). Plausible reason (*hypothesis*): `L_dot` constrains `G` only along the span of 140 training images (§2).

### Limitations of this experiment

Procedural phantoms ≠ MNIST; 12 points × 4 samples for (c); line-search only for seed 0; 3 seeds with no significance test; hyper-parameters (lr, width, epochs, early stopping on a 30-sample validation set) not tuned; the reference `A*` is approximate; Fourier numbers use the c-convention exactly as in the repository's reconstruction scripts.


## 6. Limitations

* The dataset is procedural phantoms (strokes, blobs, differences), not MNIST; its first-order statistics (mean 0.05 vs 0.13) differ from the MNIST training set of the forward model.
* 200 k-Wave pairs, split 140/30/30; one condition (periodic 89°); the reference `A*` is approximate (Stage 2).
* Three seeds per variant; seed-to-seed spread is reported, not hidden, and no significance test is run.
