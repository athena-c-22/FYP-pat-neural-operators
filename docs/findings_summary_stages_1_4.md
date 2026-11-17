# Operator scaling, adjoint consistency, gradients and learned adjoints: findings from Stages 1–4

Prepared 2026-11-17 for the supervisor meeting. Everything here is **measured** unless marked *hypothesis*. Details, tables and caveats are in the stage documents listed in §7. All data are synthetic or the 4 bundled MNIST samples; no patient data.

## 1. Summary (five sentences)

1. **Scaling (Stage 1):** k-Wave's absolute amplitude is physically exact (1-D analytic test, error 2.4·10⁻⁶), and the `c` factor from commit `5eb43c6` is already in the Python code; `c·Fourier` is ≈1.48× too large in the same test (≈√2, *hypothesis*), and no dataset or checkpoint needs regenerating.
2. **Adjointness (Stage 2):** the k-Wave forward/adjoint pair is an adjoint to about 0.1–2 % on smooth, realistic data (plain Euclidean inner products) but is **not the exact transpose** (explicit 12×12 matrices differ by 0.6); the Fourier pair is not an adjoint even up to a scalar.
3. **Gradients (Stage 3):** autograd through the learned forward FNO is a correct gradient of its own objective (finite differences agree to ≈10⁻¹⁰), but that gradient is a poor stand-in for the k-Wave gradient: the true image is not a stationary point (gradient norm 3.5–6.3 where it should be 0) and the direction agreement falls from cosine ≈0.7 to ≈0 as the test point moves away from the truth.
4. **Learned adjoint (Stage 4):** a small FNO adjoint trained on 140 k-Wave pairs overfits and is **worse than the existing Fourier adjoint**, which gives ≈100 % of the reference line-search decrease despite not being an adjoint pair.
5. The open questions that block further progress are about the **definition of the reference adjoint** (approximate k-Wave vs exact transpose) and the intended norm/weights, listed in §5.

## 2. Findings by stage

### Stage 1 — Does the `c` scaling (commit 5eb43c6) hold?

| Question | Result |
|---|---|
| What the commit changes | Only a MATLAB example: Fourier forward ×`c`, adjoint/inverse get `data/c`. |
| Is it in Python? | Yes, since earlier commits (callers use `c·numpy_forward_2d`, `numpy_inverse_2d(measurement/c)`); operator functions themselves contain no `c`. |
| Is `c` needed? | Yes, dimensionally: without it adjoint/inverse errors are ≈10³ (best-fit scalar ≈9·10⁻⁴ for the adjoint, ≈800 for the forward). |
| Does it make Fourier equal k-Wave? | No. 1-D analytic truth: k-Wave α = 1.000 (rel. error 2.4·10⁻⁶); `c·Fourier` α = 0.674 (≈1/√2; 0.7075 when Nt = Nx). Full 2-D images also show a shape error of ≈0.5 independent of scale (continuous homogeneous model, interpolation, different boundaries). |
| Regenerate data or checkpoints? | **No.** The k-Wave targets are verified in physical units; the FNO-only checkpoint trains on k-Wave data. Fourier-to-FNO models (not in the repo) might be affected by a convention change. |

*Not established:* that the √2 is the MATLAB "keep norm" mirror factor (present in the MATLAB source too); the 2-D amplitude ratios are contaminated by shape error.

### Stage 2 — Which forward/adjoint pairs are actually adjoint?

Dot-product test `⟨Ax,y⟩ = ⟨x,A*y⟩` (64×64, both conditions), metrics: both inner products, absolute discrepancy, Marta's `|⟨Ax,y⟩−⟨x,A*y⟩|²/‖y‖²`, a symmetric relative discrepancy, a Cauchy-Schwarz-normalised one and the ratio.

| Pair | Result |
|---|---|
| Control (autograd of torch Fourier forward) | exact, ≈10⁻¹⁶ — the test harness is correct |
| **k-Wave forward/adjoint** | ratio ≈1.00; `sym_rel` 0.2–2 % on smooth/realistic inputs; **≈100 % error on white noise**; not the exact transpose (12×12 matrix error 0.62, still 0.58 after the best scalar 1.37). Physical weights `dx·dy`, `dy·dt` make it ≈1500× worse: the k-Wave scalings already contain the quadrature factors. |
| Fourier (raw) | ratio ≈10⁻⁶ |
| Fourier with the `c` convention | ratio O(1) but not an adjoint even up to a scalar (shape residual 0.29–0.52) |
| Mixed (k-Wave forward, Fourier adjoint) | poor |

Why several metrics: for nearly orthogonal random inputs a 1 %-of-norm error becomes a ≈100 % error of the inner product, so `sym_rel` and the Cauchy-Schwarz metric disagree; Marta's squared metric depends on the data scale and its norm is not fixed by the notes (Euclidean assumed). *Not isolated:* the source of the k-Wave mismatch (candidate causes in the report, none tested).

### Stage 3 — Reconstruction gradient: autograd through the learned forward vs the k-Wave reference

Setup: bundled FNO checkpoint (1 epoch, periodic 89°), 4 samples × 4 test points; `g_ref = A*(Ax−b)` (k-Wave), `g_ad = J^T(G(x)−b)` (autograd).

* **Autograd is correct:** float64 central differences agree with autograd, error ∝h², best ≈2·10⁻¹⁰ (a code issue found on the way is in §6).
* **Bias:** at the true image `g_ref = 0` but `‖g_ad‖ = 3.5–6.3` (reference gradient norms at the other points: 1–17). Descent on the surrogate objective is pulled away from the truth.
* **Direction:** cosine with `g_ref` 0.65–0.73 at `half` (0.5·image), 0.14–0.34 at a smooth perturbation, ≈0 at a noisy perturbation; the exact line search along `−g_ad` removes 54–63 %, 3–17 % and 0.004–0.13 % of the reference decrease; at 2 of 4 noisy points `−g_ad` is an **ascent** direction for the true objective.
* **Where it fails:** replacing only the surrogate's residual keeps cosine ≈0.95 at `half`; replacing only its Jacobian drops it to ≈0.7 — the Jacobian is the weaker part.
* *Caveat on the reference:* in finite-difference checks `A*` agrees on smooth directions (0.2–6 %) but is off by 2–45 % (once wrong sign) on a white direction.

### Stage 4 — Learned adjoint (done), learned inverse and joint training (design only)

* 200 k-Wave triples of procedural phantoms (≈8 s per pair), split 140/30/30; 3 loss variants (supervised, plus adjoint-consistency weight 1 and 10) × 3 seeds, ≈3 min CPU each.

| Direction | Test rel. L2 to `A*y` | Gradient cosine at Stage 3 points (half / smooth / noisy) | Line-search decrease vs reference (half / smooth / noisy) |
|---|---|---|---|
| Fourier adjoint of the residual (existing, `c` convention) | 0.44 | **0.93 / 0.90 / 0.63** | **0.99 / 1.01 / 1.07** |
| Autograd through forward FNO (Stage 3) | – | 0.69 / 0.26 / −0.01 | ≈0.55–0.63 / 0.03–0.17 / ≈10⁻⁴ |
| Learned adjoint (9 models) | 0.73–0.79 | ≈0.54 / 0.2 / 0.02 | 0.28–0.35 / 0.07–0.11 / ≈10⁻³ (seed 0 only) |

* The adjoint-consistency loss gave **no benefit** (differences between weights lie within the seed spread; dot-product `cs_rel` 0.065–0.072 for all).
* The learned adjoint has the same **zero-input bias** as the Stage 3 gradient (output norm 2.1–3.4 for zero input); removing the even part after training raises the cosine at `half` from 0.54 to 0.74 (ad hoc diagnostic).
* **Reconciling with Stage 2:** the Fourier adjoint is not an adjoint pair (scale off by ≈1.4–1.5, shape residual ≈0.35 after scaling) yet is a very good *descent direction* (cosine 0.93) — a line search removes the scale error; a fixed step size would not. Line-search ratios slightly above 1 are possible because the k-Wave `A*` is itself approximate (*hypothesis*).
* Learned inverse and joint training: designed, not run (`docs/learned_operator_design.md` §3–4). The inverse-consistency normalisation `‖A⁻¹(Ax)−x‖²/‖A‖₂²` has **not** been assumed; it is kept separate from Marta's original expression.

## 3. What can and cannot be concluded

| Supported by measurements | Not supported / untested |
|---|---|
| k-Wave amplitude is physical; `c` is necessary and present | That the √2 is the only amplitude discrepancy of Fourier vs k-Wave in 2-D |
| k-Wave pair ≈ adjoint on smooth data, not exact transpose | The cause of the k-Wave mismatch |
| Autograd through the FNO is exact; the surrogate gradient is biased and its Jacobian poor (for this checkpoint) | That a well-trained forward model has a good Jacobian (only one 1-epoch checkpoint tested) |
| A 100k-parameter FNO adjoint trained on 140 pairs does not beat the Fourier adjoint | That learned adjoints cannot work at larger scale or with a linear-by-construction architecture |

## 4. Reliability of the evidence

* One checkpoint, 4 samples, periodic condition only for Stages 3–4; PML/45° is only covered by Stages 1–2.
* Stage 3 test points are synthetic displacements of the truth, not iterates of a real reconstruction.
* The reference adjoint is approximate (Stage 2), which bounds every cosine in Stages 3–4 for rough directions.
* Stage 4 uses procedural phantoms, 3 seeds, no significance tests, untuned hyper-parameters.
* Explicit-matrix tests are only 12×12 because k-Wave-python 0.5.0rc1 crashes on mirrored grids below 256 points (`uint8` cast).

## 5. Decisions needed from Prof. Betcke

1. **Reference adjoint:** is an approximate k-Wave adjoint acceptable, or is the exact discrete transpose required (e.g. differentiable k-Wave / JAX)?
2. **Norm and weights** in `|⟨Ax,y⟩−⟨x,A*y⟩|²/‖y‖²`, and which inner products define adjointness (Euclidean was assumed; physical weights worsen the k-Wave pair).
3. **√2 factor:** is it meant to stay in the physical Fourier forward, and what did "in agreement" mean in the commit message?
4. **Inverse-consistency normalisation:** is `‖A⁻¹(Ax)−x‖²/‖A‖₂²` intended, or the original expression? (Estimating `‖A‖₂` needs an adjoint, so it inherits question 1.)
5. **Forward training objective:** should forward training include a gradient/Jacobian or adjoint-consistency term, given Stage 3?
6. **Learned adjoint:** is it worth its k-Wave label cost (≈7–8 s per pair, no gain from parallel workers) given the Fourier adjoint's performance, and is a learned inverse a required comparison?

## 6. Repository state and code changes

* **Nothing is committed or pushed.** All new files from Stages 1–4 are untracked: `docs/`, new scripts under `scripts/{data,evaluation,train}/`, `src/pat_fno/evaluation/`, `src/pat_fno/models/learned_adjoint.py`, `tests/unit/` and `outputs/`.
* **One existing source file changed:** `src/pat_fno/models/fno.py`, one line (`dtype=torch.cfloat` → `transformed.dtype` in `SpectralConv2d`). `torch.cfloat` was silently forcing float32 in a float64 model; for float32 behaviour is identical (unit test). I did not re-run the bundled-checkpoint baseline, to avoid overwriting `outputs/baseline_evaluation`.
* No operator code was modified; no checkpoint or result was overwritten.
* **Tests:** 121 unit tests pass (104 existing + 17 new).
* **CI note:** the committed `scripts/evaluation/baseline_forward_check.py` has two Ruff findings (F841 unused `imp`, E501) and is not formatter-clean, so `ruff check .` / `ruff format --check .` would fail on it. I left it untouched.
* **Correction:** early on I attributed the 35-minute Stage 3 run to core oversubscription. A benchmark showed k-Wave throughput is ≈7–8 s per forward+adjoint pair regardless of the number of worker processes; the cause is not isolated.

## 7. Documents and reproduction

| Stage | Document | Command | Run time (CPU) |
|---|---|---|---|
| 1 | `docs/operator_scaling_analysis.md` | `python -m scripts.evaluation.operator_scaling_validation` | ≈3.5 min |
| 2 | `docs/adjoint_consistency_report.md` | `python -m scripts.evaluation.adjoint_consistency` | ≈11 min |
| 3 | `docs/gradient_comparison_report.md` | `python -m scripts.evaluation.gradient_comparison` (`--skip-kwave` for the FD part only) | ≈35 min (<1 min) |
| 4 | `docs/learned_operator_design.md` | `scripts.data.generate_adjoint_pairs`, `scripts.train.train_learned_adjoint`, `scripts.evaluation.learned_adjoint_eval --linesearch` | ≈24 min data + ≈3 min per model + ≈5 min eval |

Seed 20260728 throughout; environment: conda `pat_fno`, Python 3.10.20, torch 2.11.0+cpu, numpy 2.2.6, k-Wave-python 0.5.0rc1, 12 CPU cores, no GPU. Outputs under `outputs/{operator_scaling,adjoint_consistency,gradient_comparison,learned_adjoint}/`; chronological record in `docs/research_log.md`.

## 8. Suggested next steps (for discussion, none started)

1. Settle questions 1–4 above; they determine what the "reference" is.
2. Isolate the source of the k-Wave forward/adjoint mismatch (scaling, k-space correction, source sampling) and, if feasible, obtain an exact transpose.
3. Re-run the Stage 3 gradient comparison on a properly trained forward model (needs GPU time and your approval) to see whether the Jacobian problem is specific to the 1-epoch checkpoint.
4. Use the Fourier adjoint with a line search as the baseline reconstruction direction; test a bias-free (odd-by-construction) learned adjoint only if a larger dataset is approved.
