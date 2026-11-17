# Operator scaling analysis (Stage 1)

Date: 2026-11-16 · Branch: `experiment/forward-training` · Status: **findings measured; one discrepancy needs supervisor input**

## 1. Question

Marta pointed to upstream commit
[`5eb43c6`](https://github.com/meng20250923-tech/FYP-matlab-kwave/commit/5eb43c636da025fdb3bec810779cbc34cbfe8f6c)
("verified against scaled Fourier operators (FFT-Fwd \*c, FFT-Adj/Inv \*1/c). This scaling is in agreement.").
Do the Python operators use the same convention, and is that convention numerically consistent with k-Wave?

## 2. What the commit actually changes

Only **one file**: `CoronaeNet/FourierOperator2D/example_LimitedAnlge_Fourier_Operatros.m` (the *example script*;
no operator `.m` file is modified; `+31/-6` lines). Changes:

| | Before | After |
|---|---|---|
| Fourier forward | `data = kSpaceForwardMirrorFFT2D(p0,setting)` | `data = c*kSpaceForwardMirrorFFT2D(p0,setting)` |
| Fourier adjoint | `kSpaceAdjointMirrorFFT2D(data,setting)` | `kSpaceAdjointMirrorFFT2D(1/c*data,setting)` |
| Fourier inverse | `kSpaceInverseMirrorFFT2D(data,setting)` | `kSpaceInverseMirrorFFT2D(1/c*data,setting)` |
| k-Wave | (absent) | adds `kSpaceForwardKWave2D`, `kSpaceAdjointKWave2D`, and a "mixed" call feeding the c-scaled FFT data into the k-Wave adjoint (*without* undoing the scaling) |

So the "correction" is a **convention applied by the caller**: the Fourier operators themselves still return data in units of
`1/c` relative to the quantity that the k-Wave forward returns. The inline comment explains it as the
(ω, k) non-normalised vs normalised frequency (ω vs ω/c): the forward weight `w/(c²·kx) = 1/c` is the Jacobian of `ω = c·kx`.
It affects **both** forward and adjoint/inverse, in opposite directions (×c, ÷c). The k-Wave operators are not changed.
(Limitation: I could read the patch only; the commit message states the check but the numerical evidence for it is not in the repo.)

## 3. Four different "scalings" (do not conflate)

| Kind | Where | Value |
|---|---|---|
| Physical operator scaling | k-Wave forward returns pressure `p(sensor,t)`; for 1-D data `p = p0(ct)` | amplitude 1 |
| Frequency-variable (ω vs ω/c) | Fourier forward weight `w/(c²kx)` → `1/c` | **the `c` in commit 5eb43c6** |
| FFT normalisation | `unitary_fft`, `sqrt(N_src/N_dst)` after interpolation, `*sqrt(numel)` after inverse FFT | norm-preserving conventions |
| Mirror normalisation | forward `√2·frec[...]`; adjoint/inverse `data_mirror/√2` ("rescale to keep norm") | **not mentioned by the commit** |
| Discretisation weights | k-Wave adjoint source `ρ c dx/(4 dt)`, output `1/(ρc²)` | physical/numerical, belongs to Stage 2 |
| NN data normalisation | z-score of p0 and k-Wave target (`p0_mean/std`, `data_mean/std` in checkpoint) | a training convenience; **not** a physical scale |

## 4. Python status

* The `c` factor **is already in Python**, and has been since the first pipeline commit (`8f80cb0`, "Add reproducible MNIST PAT dataset pipeline"):
  `generate_sample` stores `data_fft = c * numpy_forward_2d(p0)` ([mnist.py:116](../src/pat_fno/data/mnist.py#L116));
  `train_forward_operator.py`, `run_reconstruction.py` multiply `fpat_forward_2d*` by `c`; the Fourier inverse is always called with `measurement / c`
  ([run_reconstruction.py:140](../scripts/reconstruction/run_reconstruction.py#L140), [run_reproducibility_check.py:121](../scripts/smoke/run_reproducibility_check.py#L121)).
* The operator functions themselves (`numpy_forward_2d`, `fpat_forward_2d`, adjoint, inverse) are faithful translations of the MATLAB files and, like MATLAB, **do not** contain the `c` factor. The convention lives in the callers.
* `numpy_adjoint_2d` / `fpat_adjoint_2d` are **not called by any script**, only by shape/finite tests. The `1/c` adjoint convention is therefore currently untested in Python.
* `build_setting()` defines only `interpolationMethodF`; calling the Fourier adjoint/inverse requires setting `interpolationMethodA/I` (my script does this; no change to the library).
* The k-Wave learned targets (`kwave_forward`) are k-Wave output directly. FNO-only is trained on (p0 → k-Wave).

## 5. Numerical validation

Script: `python -m scripts.evaluation.operator_scaling_validation` (≈3.5 min on CPU; seed 20260728; outputs
`outputs/operator_scaling/{results.csv,summary.json,scaling_validation.png}`).
Both conditions (`periodic_theta89`, `pml_outside_theta45`), 9 images (4 synthetic phantoms + y-independent bump + 4 bundled MNIST p0).
Environment: Python 3.10.20, torch 2.11.0+cpu, numpy 2.2.6, k-Wave-python 0.5.0rc1.
For reference `U` and candidate `V`: `alpha = <U,V>/<V,V>` (best scalar), `rel_raw = ‖U−V‖/‖U‖`, `rel_fit = ‖U−αV‖/‖U‖`.

### E0 — analytic ground truth (cleanest test)
A p0 that is constant in y is a 1-D wave problem. With the mirror line and CFL = 1, the sensor sees exactly `(p0[n] + p0[n−1])/2`.

| | periodic 89° | PML 45° |
|---|---|---|
| k-Wave vs analytic: α, rel error | **1.000, 2.4e−6** | 1.021, 0.44 (PML absorbs; boundary effect, not scale) |
| Analytic vs `c·Fourier`: α | **0.674** | **0.674** |
| …shape error after best scalar | 0.077 | 0.076 |

→ **k-Wave's absolute scale is exactly physical.** `c·Fourier` is ≈ 1.48× too large (≈ √2 to within the Nt-dependent interpolation effect).
A scan of `Nx, Nt, σ` (diagnostic, not saved as a script): with `Nt = Nx` the best-fit factor is **0.7075 ≈ 1/√2**; it drifts between 0.65 and 0.76 as `Nt/Nx` varies and barely depends on smoothness (σ = 3…10).

### E1–E3 — full 2-D images (medians; ranges in `summary.json`)

| Experiment | periodic α (range) | PML α | Reading |
|---|---|---|---|
| E1 `K` vs `Fraw` | 814 (639–1074) | 781 | order of c = 1500, but not 1500 |
| E1 `K` vs `c·F` | 0.54 (0.43–0.72) | 0.52 | k-Wave ≈ 0.5–0.7 × `c·F`; rel_fit 0.52 → large *shape* error remains |
| E2 `adj_kW(K)` vs `adj_F(K/c)` | **1.41** (1.35–1.50) | 1.24 | `1/c` makes magnitudes comparable (without it α≈9e−4, rel err ≈ 1000) |
| E2 `adj_kW(K)` vs `adj_F(K)` | 9.4e−4 | 8.2e−4 | confirms `1/c` is needed *dimensionally* |
| E3 `p0` vs `inv_F(K/c)` | 1.22 | 1.32 | inverse is also O(1) only with `1/c` |
| E4 torch `fpat_forward_2d` vs NumPy `numpy_forward_2d` | 1.06, rel 0.10 | 1.03, 0.085 | same convention; 8–10 % difference = Catmull-Rom vs `griddata` cubic |
| E5 bundled MNIST: `K` vs `c·F` raw rel L2 | 0.876 | – | stored Fourier baseline ≈ 0.94 |
| E5 same, `c·F/√2` | 0.58 | – | amplitude explains part; shape error 0.54 remains |

### Interpretation
1. **The `c` factor is necessary and Python already uses it.** Dropping it gives errors of ≈ 10³ for adjoint/inverse and ≈ 1 for forward.
2. **The `c` factor alone does not make Fourier agree with k-Wave in amplitude.** In the one case where truth is known (E0), `c·F` is √2-ish too large. The `√2` is the MATLAB "rescale to keep norm" mirror factor (forward `√2`, adjoint/inverse `1/√2`), which is present in both the MATLAB and Python sources. In 2-D the adjoint comparison shows the opposite-direction ratio ≈ √2 (1.41, periodic) which is *consistent* with a single √2 convention difference, but the 2-D numbers are contaminated by real shape error (rel_fit 0.35–0.8), so this is a **hypothesis, not a result**.
3. **Large shape differences** (rel_fit ≈ 0.5) exist independently of scale: the Fourier operator is a continuous homogeneous-medium model, with cubic interpolation in k-space, non-causal pre-arrivals (visible in the figure) and a different boundary treatment. This is the known reason the Fourier baseline is poor (0.94).
4. The comparison is made on a 64-px grid with CFL = 1, i.e. coarse; Marta's MATLAB example may use different `p0`/grid, so "in agreement" there may have been judged visually or with different settings. I cannot tell from the commit.

## 6. Do existing data / checkpoints need regenerating?

| Artefact | Affected? | Reason |
|---|---|---|
| k-Wave targets (`kwave_forward`), `examples/sample_data` | **No** | Physical units, verified analytically (E0) |
| FNO-only checkpoint `fno_only_periodic_large.pt` and its metrics | **No** | p0 → k-Wave; z-score stats come from k-Wave data. Unaffected by Fourier convention |
| `data_fft` / Fourier baseline numbers (0.936 etc.) | Interpretation only | Already `c·F`; a √2 amplitude mismatch inflates it, shape error dominates (E5) |
| Fourier-to-FNO / FNO-to-Fourier models (not in repo) | **Possibly** | Take/produce `c·F`; an FNO can absorb a constant amplitude factor, so not invalid, but changing the convention would change them. Decide after Marta's answer |
| Reconstruction using `inv_F(measurement/c)` | consistent | `1/c` is applied, same convention as commit |
| Neural-net z-score normalisation | **Unrelated** | Must not be treated as physical scale (see Stage 3: gradients must be computed in physical units) |

**Conclusion: no data or checkpoint needs regeneration now. No operator code was modified**, because the evidence shows a convention question (√2),
not a clear bug, and the MATLAB source has the same factor.

## 7. Decisions / questions for Marta
1. The commit's `c` factor is in Python already. But does she consider `c·F` vs k-Wave agreement exact? In my 1-D analytic test `c·F ≈ √2·p0(ct)`. Is the `√2` ("keep norm" for the mirrored domain) intended to remain in the *physical* forward, or only in the unitary pair (F, inverse)?
2. Which pair is the reference adjoint for the project: k-Wave adjoint (with `ρ c dx/(4dt)` and `1/(ρc²)` weights) or the Fourier adjoint? (needed for Stage 2/3)
3. Which exact discrete inner products define `A*`: plain Euclidean sums, or with `dx·dy` and `dt` weights?

## 8. Limitations
* I read the diff of the commit but not the MATLAB run output; I could not run MATLAB.
* Single grid / single `c`; α scan over `Nx, Nt` used a diagnostic snippet that is not stored as a script.
* E0 holds for ky = 0 only; 2-D amplitude relationships are confounded by shape error.
* No automated test was added in Stage 1, to avoid encoding the √2 question as "expected behaviour" before Marta answers.
