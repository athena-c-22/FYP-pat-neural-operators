# Original F0 versus Mengjie's five-epoch forward FNO

Date: 2026-11-17. Research question: **does improved forward prediction accuracy lead to better reconstruction gradients?** No models were trained or fine-tuned.

Stages A–D are complete on the four bundled samples. F5 improves forward accuracy and gradients at half-amplitude/smooth points, but retains biased, weak gradients near truth with white-noise perturbations. The original held-out dataset is now available; the [Stage E extension report](forward_checkpoint_comparison_heldout.md) records the 100-sample gate, full 10000-image forward comparison and separate stratified gradient experiment. The four-sample results below remain under `outputs/forward_checkpoint_comparison_full/`.

## Checkpoint and repository inspection (Stage A)

The initial branch was `experiment/forward-checkpoint-comparison` and the working tree was clean. No `AGENTS.md` was found in the repository search. Inspected the existing FNO, baseline inference, scaling validation, adjoint-consistency and gradient-comparison implementations and their reports before adding the comparison runner. Earlier reports are historical evidence; the measurements below are from this comparison unless explicitly labelled historical.

Located original archives:

- `C:\Users\athen\Downloads\PAT_checkpoints.zip`
- `C:\Users\athen\Downloads\PAT_logs_and_configs.zip` (also a duplicate named `PAT_logs_and_configs (1).zip`).

The checkpoint index is inside the checkpoint ZIP at `checkpoints/CHECKPOINT_INDEX.csv`; configuration and source README are inside the logs ZIP. Checkpoints are loaded directly from ZIP entries; originals are never extracted over repository files or modified.

| Property | F0 | F5 |
|---|---|---|
| Archive member | `checkpoints/principal/results/mnist_medium/mnist_large_v1/periodic_theta89/fno_only/best.pt` | `checkpoints/sample_efficiency/results/sample_efficiency/mnist_large_v1/periodic_theta89/fno_only/n50000_seed20260728/best.pt` |
| Epochs / batch size | 1 / 256 | 5 / 128 |
| SHA-256 | `0ec25bf641adaeb1339bde9c5d87e6213263748436cb663d378f6d8844faf8a3` | `73a66a3ef37bb401723c494c86bbd15c8a8f3b60bccb35e9b4a7519510d69537` |

Both are `mnist_large_v1`, `periodic_theta89`, `fno_only`, 50,000 training samples, 8 modes, width 16, 3 spectral layers, AdamW, learning rate 0.002, weight decay 0.00001, seed 20260728. These values agree across checkpoint arguments (where present), index and original YAMLs. F5 records training device `cuda`; F0 records `auto`. Each checkpoint has `model`, `arguments`, `normalization`, `setting`; both state dictionaries load strictly into the existing `TinyFNO2d`. F0 has exactly the same file hash as `examples/checkpoints/fno_only_periodic_large.pt`.

The physical settings agree: 64 by 64 image grid, 0.1 mm spacing, sound speed 1500 m/s, density 1000 kg/m^3, CFL 1, dt = 6.6666667e-8 s, 91 time samples, periodic boundary. The condition's 89-degree mask belongs to the Fourier model; these experiments use the k-Wave forward without an extra angular mask.

## Methodology and scaling

The new runner is `scripts/evaluation/forward_checkpoint_comparison.py`. It reuses existing model, prediction metrics, test-point generation, finite-difference checks, gradient metrics and k-Wave wrappers. Existing scripts and model code are unchanged.

Input images `(64,64)` are bilinearly resized to `(64,91)` with `align_corners=False`, converted to z-scores, passed through the FNO, then converted back to physical pressure. Both checkpoints have identical statistics:

```
p0_mean = 0.13131004571914673
p0_std  = 0.2903306484222412
data_mean = 0.12845729291439056
data_std  = 0.2594354748725891
```

No extra sound-speed factor, amplitude fit, per-image normalisation, quadrature weights or observation mask is applied. Objectives use plain Euclidean sums in physical units. Autograd differentiates the entire map, including resize, normalisation and de-normalisation. Evaluation is CPU float64 with complex128 spectral weights preserved; k-Wave runs in float32. Compared with the existing float32 F0 inference path, predictions differ by only 3.43e-7 relative L2.

The four bundled test samples have original MNIST source indices 2253, 7962, 9832 and 6563. Their test provenance is documented in `examples/sample_data/README.md`. It could not be independently checked during the original run; the subsequent held-out extension verified an exact match against the first four original test-shard entries. Sample NPZ SHA-256: `adfdca98c5d48b29dc6b0a7c8f3d0390a9c66a57fc03225d69103e66a0953f73`.

Forward metrics are computed per sample, then averaged: relative L2 `||prediction-target||/||target||`, physical MSE, normalised MSE `MSE/data_std^2` (the repository's training-space definition, **not** error divided by target energy), and centred Pearson correlation. Variability across four samples is descriptive, not a population confidence interval.

## Forward prediction results (Stage B)

New measurements on the same four samples:

| Model | Mean relative L2 | Mean MSE | Mean normalised MSE | Mean correlation |
|---|---:|---:|---:|---:|
| F0 | 0.394457 | 0.0132401 | 0.196712 | 0.887594 |
| F5 | 0.221510 | 0.00418841 | 0.0622286 | 0.965948 |

F5 reduces mean relative L2 by 43.8% and MSE by 68.4%; all four samples improve. This establishes output accuracy improvement on this subset, not derivative accuracy.

Across-sample standard deviations (sample SD, n=4): relative L2 0.13086 / 0.07992, MSE 0.009581 / 0.003152, normalised MSE 0.14235 / 0.04683, correlation 0.06787 / 0.02139 (F0 / F5). These reflect digit-to-digit spread, not uncertainty from repeated training.

For comparison, these are **historical metrics** read from each checkpoint's original `metrics.json`, not rerun on the full original test set:

| Model | Reported test relative L2 | MSE | Normalised MSE | Correlation |
|---|---:|---:|---:|---:|
| F0 | 0.378684 | 0.0116689 | 0.173370 | 0.900921 |
| F5 | 0.215497 | 0.00366618 | 0.0544698 | 0.968670 |

Original best validation normalised MSE: F0 0.174211; F5 0.0559119. The saved histories contain one and five epochs respectively. This is a comparison of two existing training runs; epoch count and batch size both differ, so improvement cannot be attributed solely to epochs.

![Targets, predictions and errors with shared scales](../outputs/forward_checkpoint_comparison_full/forward.png)

## Gradient comparison design (Stage C)

At four points per image (`truth`, `half`, `smooth_pert`, `noisy`), both models use the same x, observations b and fresh reference calculation. Point seeds are `20260728 + 100*sample + point_index`, matching the existing Stage 3 worker. The points are truth, half the truth, truth plus a smooth field with amplitude 0.5 times image standard deviation, and truth plus white noise with standard deviation 0.1 times image standard deviation.

Let `r = Ax-b`, `r_F = F(x)-b`, and `B` denote the existing k-Wave approximate adjoint. We calculate:

| Quantity | Formula | Purpose |
|---|---|---|
| Reference | `B r` | Common reference direction |
| AD gradient | `J_F(x)^T r_F` | Exact gradient of the surrogate objective |
| Same-residual Jacobian diagnostic | `J_F(x)^T r` | Removes forward residual mismatch |
| Residual-only diagnostic | `B r_F` | Removes learned Jacobian mismatch |

The AD gradient and reference direction belong to **different objectives**, `0.5||F(x)-b||^2` and `0.5||Ax-b||^2`. Comparing them asks whether optimising the surrogate helps the reference problem. The same-residual diagnostic isolates the Jacobian-transpose discrepancy; it is not mislabelled as the gradient of the surrogate objective.

There is also an exact additive decomposition relative to B:

```
g_AD - B r = (J_F^T - B) r + J_F^T (F(x) - Ax)
```

`decomposition.csv` records the norms of these Jacobian and forward-residual contributions and their inner product, derived from the saved arrays. They are generally not orthogonal, so their norms cannot simply be added to obtain the total error. The residual-only diagnostic `B r_F` is a complementary substitution, rather than one term of that identity.

Cosine, relative L2 and norm ratio are computed against `B r`. At truth, relative gradient metrics are marked undefined because the reference residual is numerically zero; gradient norm is the meaningful bias diagnostic. Float64 central finite differences independently verify autograd of the surrogate scalar objective using two smooth and two white directions and h = 1e-3, 1e-4, 1e-5.

**Reference limitation:** the earlier adjoint tests show B is not the exact discrete transpose of A, particularly for grid-scale directions. This task does not repair that operator. Therefore cosine/error results refer to an approximate reference. Descent is checked separately using A itself.

### Newly measured gradients

Values below are means ± sample SD across four images. Relative L2 and norm ratio are dimensionless; all comparisons use the same B r.

| Point | Model | Cosine | Relative gradient L2 | Gradient norm ratio |
|---|---|---:|---:|---:|
| half | F0 | 0.689 ± 0.043 | 0.832 ± 0.058 | 1.095 ± 0.096 |
| half | F5 | 0.817 ± 0.024 | 0.621 ± 0.032 | 1.049 ± 0.030 |
| smooth perturbation | F0 | 0.262 ± 0.093 | 1.315 ± 0.286 | 1.126 ± 0.330 |
| smooth perturbation | F5 | 0.567 ± 0.069 | 0.887 ± 0.082 | 0.867 ± 0.198 |
| noisy | F0 | -0.008 ± 0.018 | 3.354 ± 0.713 | 3.186 ± 0.756 |
| noisy | F5 | 0.053 ± 0.048 | 1.645 ± 0.446 | 1.326 ± 0.519 |

At truth, mean gradient norm drops from **4.723 ± 1.435** (F0) to **1.910 ± 0.886** (F5), but remains large compared with the approximate reference norm **6.91e-5**. F5 reduces this bias by 59.6%; it does not make the true image a stationary point of its surrogate objective.

Fresh k-Wave at truth differs from stored measurements by **2.50e-5 to 4.55e-5 relative L2**, with mean reference objective 3.22e-7. This is slightly larger than the ~3e-6 stated in the historical gradient report; that numerical reproduction claim is not reproduced here. Its cause is not isolated. It is far below either model's forward error and does not warrant a rescaling. Relative truth-gradient and step-reduction metrics remain undefined because they would divide by this numerical floor.

Finite-difference checks support correctness of both surrogate gradients. At h=1e-4, median relative directional-derivative errors are **1.83e-10 (F0)** and **3.51e-10 (F5)**, with maxima 1.35e-8 and 3.59e-8. The full finite-difference table has 384 rows. All 16 F0 test points and AD gradient arrays match the previous Stage 3 NPZ exactly (maximum gradient difference zero); the new k-Wave directions and steps were nevertheless computed afresh.

### Residual versus Jacobian diagnostics

Mean cosine against B r:

| Point | F0 AD | F5 AD | F0 same-residual J^T r | F5 same-residual J^T r | F0 residual-only B r_F | F5 residual-only B r_F |
|---|---:|---:|---:|---:|---:|---:|
| half | 0.689 | 0.817 | 0.698 | 0.818 | 0.953 | 0.981 |
| smooth perturbation | 0.262 | 0.567 | 0.381 | 0.574 | 0.120 | 0.323 |
| noisy | -0.008 | 0.053 | 0.120 | 0.237 | -0.012 | 0.015 |

Same-residual relative gradient L2 improves from **0.806 to 0.615** at half, **0.955 to 0.845** at smooth perturbations and **0.998 to 0.973** at noisy points. F5's Jacobian is better against the approximate adjoint on these probes, but still almost completely misses the noisy reference direction in relative-error terms.

The additive error decomposition gives these mean component norms (norm of each sample's component, then averaged):

| Point | F0 Jacobian component | F5 Jacobian component | F0 forward-residual component | F5 forward-residual component |
|---|---:|---:|---:|---:|
| half | 11.054 | 8.437 | 2.240 | 1.048 |
| smooth perturbation | 4.560 | 4.020 | 4.877 | 3.011 |
| noisy | 1.471 | 1.434 | 4.760 | 1.920 |

At half amplitude, Jacobian mismatch dominates both models' error budgets. At noisy points, improved predictions reduce the residual-induced bias strongly, while the same-residual Jacobian error barely shrinks in norm. At truth the residual-induced component is essentially the entire gradient bias. These results distinguish two mechanisms; identical normalisation and the F0 baseline check argue against a preprocessing/scaling discrepancy as their explanation.

![Reference, AD and same-residual gradients for sample 0; shared scale within each row](../outputs/forward_checkpoint_comparison_full/gradients.png)

## Reconstruction-step design (Stage D)

For each candidate g, calculate `Ag` and the actual directional derivative of the linear reference objective along `-g`, `-<r,Ag>`. A negative derivative means descent, regardless of the approximate adjoint's cosine. Use the same nonnegative line search for all directions:

```
alpha_star = max(0, <r,Ag> / ||Ag||^2)
x_new = x - fraction * alpha_star * g
fraction in {0.1, 1.0}
```

The 0.1 fraction is a small step, while 1.0 tests the full line minimum. Every accepted update is evaluated by a fresh k-Wave forward call; the quadratic prediction is also recorded. Ascent directions receive a zero step. A positive cosine alone does not count as useful reconstruction performance. The existing `quadratic_step_decrease` helper permits negative steps, so it is deliberately not used to claim descent here. No positivity projection or regularisation is applied; these are direction diagnostics, not complete reconstruction algorithms. A multi-iteration reconstruction is not part of this run.

### Newly measured reconstruction-step results

Full nonnegative line minimum (fraction=1). Objectives are averaged over four samples; percentage reduction is the mean of the four per-sample relative reductions, not the percentage change of the two means.

| Point | Direction | Mean initial objective | Mean objective after | Mean reduction | Descent directions / 4 | Improved / unchanged / worsened |
|---|---|---:|---:|---:|---:|---|
| half | reference | 62.5588 | 7.06597 | 87.812% | 4 | 4 / 0 / 0 |
| half | F0 | 62.5588 | 29.1376 | 51.604% | 4 | 4 / 0 / 0 |
| half | F5 | 62.5588 | 19.0472 | 68.291% | 4 | 4 / 0 / 0 |
| smooth perturbation | reference | 6.19664 | 0.981412 | 84.106% | 4 | 4 / 0 / 0 |
| smooth perturbation | F0 | 6.19664 | 5.67046 | 8.814% | 4 | 4 / 0 / 0 |
| smooth perturbation | F5 | 6.19664 | 4.04167 | 35.939% | 4 | 4 / 0 / 0 |
| noisy | reference | 1.24539 | 0.676869 | 45.663% | 4 | 4 / 0 / 0 |
| noisy | F0 | 1.24539 | 1.24533 | 0.00431% | 2 | 2 / 2 / 0 |
| noisy | F5 | 1.24539 | 1.24234 | 0.22507% | 3 | 3 / 1 / 0 |

The small-step test (10% of each direction's line minimum) gives:

| Point | Reference mean reduction | F0 mean reduction | F5 mean reduction |
|---|---:|---:|---:|
| half | 16.684% | 9.805% | 12.975% |
| smooth perturbation | 15.980% | 1.675% | 6.828% |
| noisy | 8.676% | -0.00072% | 0.04191% |

For the noisy small F0 steps, one image improves, one worsens by **0.00434%**, and two directions are rejected. F5 improves three and rejects one. The largest difference between an actually evaluated objective and the quadratic prediction is **7.89e-5 of the initial objective** away from truth. The tiny noisy F0 reduction is within this discrepancy scale, so its practical benefit is not reliable. Finite-precision solver/casting effects are a possible explanation, not an isolated cause. The CSV preserves the actual worsening; no backtracking was used to hide it.

Truth-point steps are stored for transparency but excluded from useful-descent counts and percentage reductions: several tiny accepted steps, including reference steps, worsen an objective already near its numerical floor. They do not establish meaningful reconstruction improvement or deterioration. At the three nontrivial point types, F5's full steps improve 11/12 and reject 1/12, versus 10/12 and 2/12 for F0. Better acceptance counts alone should not obscure that F5's noisy-point objective reductions are tiny.

The run produced 32 model/point gradient records, 96 step records and 384 finite-difference records in **735.6 seconds** on CPU. Each of the 48 direction tests records both a small and a full step. The original Stage 3 arrays were used only for an after-the-fact reproducibility comparison, not as cached inputs to the new experiment.

## Interpretation and recommended FYP experiments

**On these four examples, improved forward prediction accompanies better reconstruction gradients, but does not guarantee adequate gradients.** F5 improves all images' forward metrics, all half/smooth AD cosines, the same-residual Jacobian probes, and the truth-point bias. Its real reference-objective decreases are substantially better at half amplitude and smooth perturbations. The comparison supports testing training maturity as one factor in derivative quality.

However, F5 retains a substantial truth-point bias and nearly orthogonal noisy-point gradients. Even the oracle reference-objective line search extracts only 0.225% mean decrease there, compared with 45.663% along the reference direction. Output accuracy remains insufficient evidence for robust inverse-problem optimisation. This comparison establishes an association for two training runs, not a causal law relating forward error and Jacobian error.

For learned adjoints, use F5 as a stronger frozen forward baseline before assessing whether an independently learned adjoint helps. The results do not establish that a learned adjoint will outperform autograd, or that F5's transpose is physically adjoint-consistent. If training an adjoint later, supervise physically relevant residuals and test descent/Jacobian consistency independently of forward metrics. Obtain a better-defined discrete reference transpose before interpreting rough-direction cosine ceilings as exclusively learned-model failure.

Recommended sequence:

1. **Completed in the held-out extension:** locate the original shards, run the full paired forward comparison and a separate stratified gradient subset, reporting spread and paired differences. See the [extension report](forward_checkpoint_comparison_heldout.md).
2. Run short matched reconstructions from common zero/half-image starts, with the same reference-objective acceptance rule, tracking both objective and image error. The current line search uses k-Wave to find the best step along each candidate direction; it does not measure reconstruction acceleration.
3. Test whether F5's reduced bias persists with measurement noise and partial sensors, keeping masking inside both objectives and their derivatives.
4. Resolve the approximate-adjoint versus exact-transpose issue, then consider residual/Jacobian supervision or bias-free linear architectures as separately approved training experiments. This task provides no new training evidence for those proposals.

## Larger evaluation and limitations (Stage E)

The original four-sample run initially lacked the larger dataset. The user then downloaded Mengjie's `PAT_dataset_large_periodic_validation_test.zip` and supplied the extracted local directory. No replacement data were generated. All 40 test and 20 validation shard headers and source-index permutations were checked, and the first four test entries exactly match the bundled arrays.

The [held-out extension report](forward_checkpoint_comparison_heldout.md) contains the detailed schema audit, compatibility evidence, 100-image correctness gate, newly measured full 10000-image forward results, and a separate 20-image gradient experiment selected with two images per digit and excluding these original four. All larger-experiment outputs use new directories. F5 improves all four forward metrics on all 10000 original test images; mean relative L2 is 0.378684 (F0) versus 0.215494 (F5).

The tables above remain the original four-image measurements. Their limitations are unchanged: one geometry, two checkpoints, an approximate reference adjoint and synthetic perturbations rather than reconstruction trajectories. Broader forward accuracy does not establish gradient quality across the entire test split; consult the separate gradient subset results in the extension report. No clinical, noisy-observation or partial-sensor claims follow.

## Reproduction and output safety

Use the existing conda environment `pat_fno`, Python 3.10.20, torch 2.11.0+cpu, NumPy 2.2.6 and k-Wave-python 0.5.0rc1. From the repository root in PowerShell:

```powershell
& C:\Users\athen\anaconda3_\envs\pat_fno\python.exe -m scripts.evaluation.forward_checkpoint_comparison `
  --checkpoint-archive C:\Users\athen\Downloads\PAT_checkpoints.zip `
  --logs-archive C:\Users\athen\Downloads\PAT_logs_and_configs.zip `
  --output-dir outputs/forward_checkpoint_comparison_full --workers 4
```

Choose a fresh output directory when rerunning; the runner refuses an existing directory. Generated files under `outputs/` are ignored by Git. `meta.json` saves checkpoint provenance, histories, historical metrics, sample hash, software versions, seed and timing. `forward.csv`, `gradients.csv`, `finite_differences.csv` and `steps.csv` contain per-sample numerical results; `summary.csv` contains 126 aggregate metric rows with sample SD and ranges; `decomposition.csv` contains the 32 gradient-error decompositions. NPZs retain predictions and gradient arrays; PNGs show shared-scale sensor and gradient comparisons. Original YAMLs and index are copied as provenance only.

The first sandboxed attempt completed forward evaluation under `outputs/forward_checkpoint_comparison/`, then failed because k-Wave calls chmod on its installed solver outside the writable workspace. The approved rerun uses the separate `_full` directory. Earlier experiment outputs, checkpoints and training code are untouched. Nothing was committed or pushed.

Focused validation:

```powershell
& C:\Users\athen\anaconda3_\envs\pat_fno\python.exe -m pytest tests/unit/test_forward_checkpoint_comparison.py tests/unit/test_gradient_comparison.py tests/unit/test_fno.py -q -p no:cacheprovider
& C:\Users\athen\anaconda3_\envs\pat_fno\python.exe -m ruff check scripts/evaluation/forward_checkpoint_comparison.py tests/unit/test_forward_checkpoint_comparison.py
```

The focused tests passed (12). The complete unit suite passed **124 tests**, using `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, and `--basetemp outputs/comparison_tests_20261117 -p no:cacheprovider`. Its first sandboxed attempt had 96 passes and 28 temporary-directory setup errors; the approved rerun resolved those permission errors. Ruff check and format checks pass for both new Python files. Three added tests verify rejection of ascent directions, positive full/small steps, and a zero direction. No existing tests were changed.
