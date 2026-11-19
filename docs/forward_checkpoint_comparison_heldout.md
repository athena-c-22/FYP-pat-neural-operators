# F0 versus five-epoch FNO: original held-out dataset extension

Date: 2026-11-17. This extends the [four-sample comparison](forward_checkpoint_comparison.md), using the original dataset supplied by the user. F0 and F5 retain the checkpoint identities and configurations documented there. No training or data generation was performed. All results here are newly computed unless explicitly labelled historical.

F5 improves forward metrics on all 10000 test images. On a separate digit-stratified subset of 20 images, it improves gradient cosine on every half-amplitude/smooth probe and 18/20 noisy probes, but noisy gradients remain nearly orthogonal to the approximate reference. Better prediction accuracy accompanies better gradients here; it remains insufficient to guarantee useful reconstruction updates.

## Dataset inspection and compatibility

Actual dataset directory:

```
C:\Users\athen\projects\fyp\FYP-matlab-kwave\outputs\datasets\PAT_dataset_large_periodic_validation_test\mnist_large_v1\periodic_theta89
```

It contains `manifest.json`, 40 `test_*.h5` shards and 20 `validation_*.h5` shards. Every shard has 250 samples. Filenames cover contiguous ranges 0–10000 and 0–5000 respectively, with no gaps or overlaps. Training shards are not included in this validation/test handover; the manifest still records the original training count of 50000.

Manifest: `mnist_large_v1`, `periodic_theta89`, periodic boundary, 89 degrees, 64-pixel grid, CFL 1, Nt 91, dt 6.666666666666667e-8 s, complete=true. Split counts: train 50000, validation 5000, test 10000.

Every shard's inspected schema:

| Dataset | Shape | dtype |
|---|---|---|
| `p0` | (250,64,64) | float32 |
| `kwave_forward` | (250,64,91) | float32 |
| `data_fft` | (250,64,91) | float32 |
| `fourier_raw` | (250,64,91) | float32 |
| `label` | (250,) | int64 |
| `source_index` | (250,) | int64 |

Each shard records complete=true, split, condition, boundary, theta_deg=89, sound_speed=1500, Nt=91 and the same dt. These fields, shapes and dtypes were checked across all 60 shards. `inspection.json` preserves each shard's metadata, compression, size and schema. All evaluated p0/target arrays and predictions were checked for finiteness. Validation payloads were not evaluated; only their headers, labels and source indices were inspected.

Both checkpoint state dictionaries load strictly. Dataset grid, CFL, sensor/time shape, angle, sound speed, boundary and dt agree with their saved settings; the checkpoints have identical normalisation statistics. Grid spacing and density are not separately recorded in these shard headers or manifest; the source configuration documents them, and the exact bundled-data match below supplies additional provenance evidence. Do not interpret missing metadata as an independent verification of those physical parameters on every sample.

Source indices are unique within each split. All 10000 test indices exactly match the deterministic test permutation produced by the original loader with seed 20260728; the validation indices exactly match its separate 5000-image slice of the MNIST training permutation. Test and validation source indices refer to different original MNIST splits, so equality of their integer values would not indicate image leakage.

The first four test records exactly reproduce the bundled **p0, k-Wave targets, Fourier predictions, labels and source indices**, element for element. Thus their provenance is now independently checked against the original shards. The larger evaluation includes those four; it does not treat them as additional independent samples.

## Evaluation method and 100-sample gate

`scripts/evaluation/heldout_checkpoint_comparison.py` streams the original test shards in batches of 16. It reuses the physical-space surrogate from the four-sample runner, extended to accept batches. Both models see identical images and stored, unmasked k-Wave observations. Resize is bilinear with align_corners=false; input/output z-score conversion uses the original training statistics; no amplitude fitting, per-image rescaling, extra sound-speed factor or quadrature weights are applied.

Inference uses float64/complex128 as in the earlier comparison. Metrics are per-image relative L2, physical MSE, normalised MSE = MSE/data_std^2, and centred correlation, then averaged. Predictions are streamed rather than retained for all 10000 samples. The metadata record seed, dtype, checkpoint hashes, manifest hash, selection and runtime. Model and training code remain unchanged; the single-image wrapper behaviour is covered by a new batching/gradient equivalence test.

The gate used the first 100 records in the **already randomly permuted** original test order, without selecting by errors or labels. All four metrics improved for all 100 images. Batch versus single predictions differed by only 4.55e-16 (F0) and 1.32e-15 (F5), relative L2. Against the old four-sample CSV, MSE/correlation agree to double precision; relative L2 differs by at most 3.54e-8 because the old metric used a float32 target norm and the streaming metric computes it in float64. This harmless precision difference is recorded in `pilot_verification.json`.

| 100-sample gate | F0 | F5 |
|---|---:|---:|
| Mean relative L2 ± sample SD | 0.385817 ± 0.077962 | 0.217184 ± 0.052077 |
| Mean physical MSE | 0.0117050 | 0.00357202 |
| Mean normalised MSE | 0.173905 | 0.0530707 |
| Mean correlation | 0.898248 | 0.968477 |

Only after these checks passed was the full split evaluated. All 200 pilot model/image records agree with the full-run records within 1e-12 relative tolerance. The gate overlaps the full split and is not a second independent experiment.

## Full 10000-image forward comparison

| Newly measured metric | F0 | F5 |
|---|---:|---:|
| Mean relative L2 ± sample SD | 0.378684 ± 0.082382 | 0.215494 ± 0.054342 |
| Median relative L2 | 0.372106 | 0.205214 |
| Mean physical MSE ± sample SD | 0.01166895 ± 0.00540685 | 0.00366611 ± 0.00160807 |
| Mean normalised MSE ± sample SD | 0.17336961 ± 0.08033143 | 0.05446869 ± 0.02389159 |
| Mean correlation ± sample SD | 0.900921 ± 0.045199 | 0.968671 ± 0.016502 |

F5 improves **all four forward metrics on every one of the 10000 test images**. The mean paired relative-L2 difference F5−F0 is -0.163190, with sample SD 0.039600. `paired.csv` also provides approximate normal intervals for mean paired differences under an i.i.d. image assumption. The test-split means themselves are fully observed; these intervals do not describe training variability or provide guarantees for other imaging distributions.

Mean relative L2 by digit:

| Digit | n | F0 | F5 |
|---|---:|---:|---:|
| 0 | 980 | 0.333188 | 0.188411 |
| 1 | 1135 | 0.346243 | 0.208519 |
| 2 | 1032 | 0.401632 | 0.217296 |
| 3 | 1010 | 0.396360 | 0.220555 |
| 4 | 982 | 0.398325 | 0.221081 |
| 5 | 892 | 0.431299 | 0.238647 |
| 6 | 958 | 0.354433 | 0.204195 |
| 7 | 1028 | 0.375444 | 0.216071 |
| 8 | 974 | 0.390118 | 0.228194 |
| 9 | 1009 | 0.367860 | 0.214715 |

These results closely reproduce the historical reported forward metrics (F0 relative L2 0.37868410; F5 0.21549711), with small non-bitwise numerical differences. Historical and new metrics are kept distinct. The precise cause of the small differences from the original evaluator was not isolated; they do not change the comparison.

The full forward run took 567.4 seconds. This wall time includes I/O and overlapped the gradient pilot, so it is not a controlled inference-speed benchmark.

![Full-split error distribution and paired per-image errors](../outputs/forward_checkpoint_comparison_heldout10000/forward_distribution.png)

## Efficient held-out gradient protocol

The user approved a separate, timed gradient experiment after seeing the larger-evaluation cost distinction. `scripts/evaluation/heldout_gradient_comparison.py` selects **two images per digit**, using NumPy's default_rng with seed 20260728, without replacement and excluding original test positions 0–3. Selection is independent of either model's errors. All 20 source indices and shard offsets are stored in `selection.json` and the metrics CSV. The three-image pilot uses the first selected example for digits 0, 1 and 2: test positions 6425, 9401 and 9430 (source indices 3613, 5036 and 7986).

At each selected image, evaluate half-amplitude, smooth-perturbation and noisy points using the existing point generator. Perturbation seeds are `20260728 + 100*test_position + point_index`, where point indices retain the original ordering truth/half/smooth/noisy. Observations are the original stored k-Wave measurements. Both models share the exact same x, b, fresh Ax and reference B(Ax-b).

Each image uses only six k-Wave calls: one forward and one approximate adjoint at each of the three points. Compute AD gradients `J_F^T(F(x)-b)` and same-residual diagnostics `J_F^T(Ax-b)` for both checkpoints. Report cosine, relative gradient L2 and norm ratio, plus same-residual diagnostics. Expensive residual-only adjoint substitutions and previously validated finite-difference checks are omitted. The gradient comparison still involves two different objectives, as explicitly defined in the original report.

The pilot completed in **82.25 seconds** with three workers; per-worker image times were 70.23–70.64 seconds. Extrapolation to 20 images was **548 seconds (9.1 minutes)** for gradients. Before the full gradient launch, the user was informed of a conservative **12–15 minute** estimate including objective tests on three images, below their 30-minute reporting threshold. The full run reuses the three pilot images rather than recomputing them, verifying checkpoint hashes and selection consistency before reuse.

Objective-descent tests are limited to the three pilot images. For each point and reference/F0/F5 direction, use one full nonnegative quadratic line-minimum step, then evaluate its actual objective with a fresh k-Wave call. Ascent directions receive zero steps. There is no repeated small-step sweep, no full 20-image reconstruction-step protocol and no multi-iteration reconstruction. This is a deliberately small diagnostic, not a representative reconstruction benchmark.

The 20-image design weights each digit equally, unlike the natural label frequencies of the full test split. Its variation reflects selected images and perturbations, not independent training runs. The k-Wave adjoint remains an approximate discrete transpose, particularly for noisy directions; actual descent measurements are therefore reported separately from cosine.

### Newly measured 20-image gradient results

Means ± sample SD across the 20 selected images:

| Point | Model | Gradient cosine | Relative gradient L2 | Gradient norm ratio |
|---|---|---:|---:|---:|
| half | F0 | 0.690 ± 0.051 | 0.843 ± 0.067 | 1.122 ± 0.070 |
| half | F5 | 0.808 ± 0.040 | 0.643 ± 0.061 | 1.073 ± 0.032 |
| smooth perturbation | F0 | 0.304 ± 0.062 | 1.367 ± 0.222 | 1.264 ± 0.286 |
| smooth perturbation | F5 | 0.581 ± 0.069 | 0.867 ± 0.081 | 0.872 ± 0.106 |
| noisy | F0 | 0.006 ± 0.029 | 3.161 ± 0.725 | 2.997 ± 0.761 |
| noisy | F5 | 0.068 ± 0.036 | 1.535 ± 0.242 | 1.221 ± 0.284 |

Paired counts: F5 improves relative gradient L2 on **20/20** images for all three point types. Cosine improves on **20/20 half**, **20/20 smooth**, and **18/20 noisy** probes. Norm ratios move closer to one on 18/20, 13/20 and 20/20 respectively. A norm ratio nearer one does not establish directional accuracy: the noisy F5 norm is substantially improved, yet its mean cosine is only 0.068.

Same-residual diagnostics compare `J_F^T(Ax-b)` against exactly the same reference B(Ax-b), removing differences in model measurement residuals:

| Point | F0 same-residual cosine | F5 same-residual cosine | F0 same-residual relative L2 | F5 same-residual relative L2 |
|---|---:|---:|---:|---:|
| half | 0.703 | 0.812 | 0.806 | 0.632 |
| smooth perturbation | 0.384 | 0.612 | 0.956 | 0.808 |
| noisy | 0.108 | 0.223 | 1.000 | 0.977 |

F5 improves both the surrogate-objective gradients and these residual-controlled Jacobian probes. However, even with the true residual supplied, its noisy-point gradient relative error remains about 0.98 and its norm ratio is only 0.286. This indicates an unresolved Jacobian mismatch, alongside the residual-error contribution seen when using the model's own predictions. The common preprocessing, equal normalisation and reproduced bundled data argue against an inconsistent scaling convention as the explanation.

There are 120 gradient rows (20 images × 3 points × 2 checkpoints). The additional 17 images took **386.0 seconds**, after reusing the pilot's 18 gradient rows. Total gradient acquisition wall time including the separately measured pilot is approximately **468.3 seconds (7.8 minutes)**. No finite-difference checks or truth-point probes were repeated on this subset.

![Paired gradient cosine, relative error and norm ratio on the 20-image subset](../outputs/forward_checkpoint_comparison_gradient20/gradient_metrics.png)

### Three-image objective-descent diagnostic

These are **new, separate** measurements on the three timed pilot images, using stored gradient arrays and fresh forward solves. The full nonnegative line-minimum step was applied once per candidate, without a small-step sweep. Percentages are means of individual relative reductions; initial/after objectives are separately averaged.

| Point | Direction | Mean initial objective | Mean objective after | Mean reduction | Descent / improved / rejected out of 3 |
|---|---|---:|---:|---:|---|
| half | reference | 58.6777 | 7.46411 | 87.309% | 3 / 3 / 0 |
| half | F0 | 58.6777 | 27.3584 | 53.321% | 3 / 3 / 0 |
| half | F5 | 58.6777 | 17.2971 | 69.779% | 3 / 3 / 0 |
| smooth perturbation | reference | 4.61164 | 0.811328 | 81.945% | 3 / 3 / 0 |
| smooth perturbation | F0 | 4.61164 | 4.23824 | 7.626% | 3 / 3 / 0 |
| smooth perturbation | F5 | 4.61164 | 2.89954 | 40.484% | 3 / 3 / 0 |
| noisy | reference | 1.22182 | 0.657979 | 45.973% | 3 / 3 / 0 |
| noisy | F0 | 1.22182 | 1.22177 | 0.00426% | 2 / 2 / 1 |
| noisy | F5 | 1.22182 | 1.21585 | 0.49201% | 3 / 3 / 0 |

No accepted step worsened the measured objective. The rejected F0 noisy direction received a zero step. F5's noisy-point reductions range from 0.167% to 0.830%, while the reference achieves 44.87–47.33%. The maximum discrepancy between the linear quadratic prediction and the actually evaluated objective is 7.94e-5 of the initial objective. Tiny F0 noisy improvements are on that numerical-discrepancy scale and should not be treated as reliable practical progress.

The 27 individual step rows are in `steps.csv`; `steps_summary.csv` records means, sample SDs, ranges and acceptance counts. Do not infer actual descent rates for all 20 images from these three images, or from the other 17 images' cosine signs. For context, noisy-point cosine against the approximate reference is nonpositive on 7/20 F0 and 1/20 F5 probes; that is an alignment diagnostic, not an exact reference-objective descent test.

The full 20-image run, reusing pilot gradients and including these objective tests, took **580.0 seconds**. Adding the separately timed pilot gives approximately **662.2 seconds (11.0 minutes)** for the expanded gradient/step work. This stayed below the communicated 12–15 minute conservative estimate and the user's 30-minute threshold.

## Interpretation, limits and next experiments

The earlier four-sample finding survives this extension. F5's forward improvement is consistent over the entire original test split, and its half/smooth gradient improvements occur on every one of the 20 independently selected images. Its same-residual diagnostics also improve, so the difference is not explained solely by a better predicted measurement residual. This supports using F5 as the frozen forward baseline for subsequent learned-adjoint work.

Nevertheless, noisy-point gradients remain poorly aligned and give small true-objective reductions even with an oracle line search. Relative output accuracy alone is not an adequate criterion for reconstruction usefulness. An adjoint/Jacobian experiment remains necessary; the data do not establish that a separately learned adjoint would outperform F5's autograd gradient.

Limitations:

- Full-split evidence concerns forward predictions only. Gradients use 20 balanced images, and descent uses only three images/digits; neither is a full reconstruction experiment.
- B remains an approximate reference adjoint. Rough-direction disagreement combines learned Jacobian error with reference-discretisation limitations. Same-residual probes control residual mismatch but do not repair B.
- These are synthetic image perturbations and noise-free observations, not reconstruction trajectories or noisy measurements. No positivity projection, regularisation, partial sensors or clinical data were tested.
- Two existing checkpoints differ in both training epochs and batch size. The comparison is paired on images, not replicated across training seeds; it does not isolate a causal effect of training duration.
- Header/source-index checks and exact bundled matching establish strong provenance and compatibility evidence. Full-test targets were not regenerated with k-Wave, and validation forward accuracy was not evaluated.

Next, run a short matched iterative reconstruction study from common initialisations on a small, fixed held-out subset. Track reference objective and image error separately, with a shared acceptance rule and F5/F0/reference baselines. Resolve the approximate-adjoint issue before interpreting noisy-direction errors as exclusively learned-model failure. Any future learned-adjoint training should use realistic residuals and be assessed with both same-residual probes and actual objective decreases; no training was authorised or performed here.

## Outputs and reproduction

All output directories are new and ignored by Git:

- `outputs/forward_checkpoint_comparison_heldout100/`: pilot metrics, inspection, paired summaries, plot and verification.
- `outputs/forward_checkpoint_comparison_heldout10000/`: all 20000 model/image metric records, overall/digit summaries, paired differences, inspection, metadata and plot.
- `outputs/forward_checkpoint_comparison_gradient_pilot3/`: timing pilot and its saved gradient arrays.
- `outputs/forward_checkpoint_comparison_gradient20/`: separate stratified experiment, including reused pilot results and three-image descent diagnostics.

All runs completed. Previous experiment outputs and checkpoint files were preserved; no commit or push was made.

Run from the repository root, using the existing `pat_fno` environment. In PowerShell:

```powershell
$datasetPath = 'outputs/datasets/PAT_dataset_large_periodic_validation_test/mnist_large_v1/periodic_theta89'
$archivePath = 'C:\Users\athen\Downloads\PAT_checkpoints.zip'
$pythonPath = 'C:\Users\athen\anaconda3_\envs\pat_fno\python.exe'

& $pythonPath -m scripts.evaluation.heldout_checkpoint_comparison --dataset-dir $datasetPath --checkpoint-archive $archivePath --output-dir outputs/forward_checkpoint_comparison_heldout100 --limit 100
# Inspect the gate's inspection.json, metrics and numerical checks before continuing.
& $pythonPath -m scripts.evaluation.heldout_checkpoint_comparison --dataset-dir $datasetPath --checkpoint-archive $archivePath --output-dir outputs/forward_checkpoint_comparison_heldout10000 --limit 10000

& $pythonPath -m scripts.evaluation.heldout_gradient_comparison --dataset-dir $datasetPath --checkpoint-archive $archivePath --output-dir outputs/forward_checkpoint_comparison_gradient_pilot3 --pilot --workers 3
# Read the measured pilot timing and estimate before launching the larger run.
& $pythonPath -m scripts.evaluation.heldout_gradient_comparison --dataset-dir $datasetPath --checkpoint-archive $archivePath --output-dir outputs/forward_checkpoint_comparison_gradient20 --reuse-pilot outputs/forward_checkpoint_comparison_gradient_pilot3 --descent-images 3 --workers 3
```

Choose fresh output-directory names when repeating these commands. The runners refuse existing output directories. No earlier four-sample outputs or checkpoints are overwritten. k-Wave commands required execution outside the sandbox because its wrapper prepares the installed solver executable; those launches were approved.

Validation: 15 focused tests pass, covering batched prediction/gradient equivalence, stratified deterministic selection/exclusion, paired metric signs, previous gradient helpers and the existing FNO tests. Original shard inspection, exact bundled matching, complete split count/uniqueness, finite payload checks and pilot/full agreement were also checked against the real data. No tests download or regenerate datasets.
