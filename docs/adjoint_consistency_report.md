# Adjoint consistency report (Stage 2)

Date: 2026-11-16 · Branch `experiment/forward-training` · All numbers below are measured unless marked *hypothesis*.

## 1. Problem and why it matters
A learned adjoint `G ≈ A*` and any gradient `A*(Ax − b)` are only meaningful if we have a trustworthy reference pair
`(A, A*)` satisfying `<Ax, y>_Y = <x, A*y>_X`. Before training anything, we test which existing implementations are such a pair.

## 2. Operators tested (all take `x: (Nx,Ny)`, `y: (Ny,Nt)`; 2-D, homogeneous c = 1500 m/s, dx = dy = 0.1 mm, dt = dx/c, Nt = 91 at 64×64)

| Pair name | A | B (candidate A\*) | Notes |
|---|---|---|---|
| `kwave` | `kwave_forward_2d` | `kwave_adjoint_2d` | mirrored 128×64 grid, sensor on mirror line; periodic (no PML) or PML outside. Adjoint = time-reversed pressure source scaled by `ρ c dx/(4 dt)`, output `/(ρ c²)` |
| `fourier_np_raw` | `numpy_forward_2d` | `numpy_adjoint_2d` | MATLAB translation; limited-angle mask (89° / 45°), cubic `griddata` |
| `fourier_np_c` | `c·numpy_forward_2d` | `numpy_adjoint_2d(y/c)` | the convention of upstream commit 5eb43c6 |
| `fpat_torch_raw` | `fpat_forward_2d` | `fpat_adjoint_2d` | torch version (Catmull-Rom) |
| `fpat_autograd` | `fpat_forward_2d` | autograd VJP of it | **control**: exact transpose by construction |
| `mixed_kwfwd_Fadj` | `kwave_forward_2d` | `numpy_adjoint_2d(y/c)` | cross pair (as in the MATLAB "mixed" example) |

Boundary: periodic or PML-outside (see `build_setting`); acquisition: θ_max = 89° (periodic) / 45° (PML) only for Fourier operators; k-Wave's sensor is the same flat line of 64 points in both conditions.
**Inner-product weights.** Tested two choices: plain Euclidean sums (`wx = wy = 1`) and rectangle-rule physical weights `wx = dx·dy`, `wy = dy·dt`. Only the ratio `wy/wx` matters; it is exposed in `adjoint_metrics`.

## 3. Metrics, with strengths and limits (`src/pat_fno/evaluation/adjoint_consistency.py`)
Let `d = <Ax,y> − <x,By>`.

| Metric | Formula | Good for | Weakness |
|---|---|---|---|
| absolute | `|d|` | simple | depends on the scale of x, y, A |
| Marta's | `|d|² / ‖y‖²` | the expression in the notes | not dimensionless; **the notes do not define the norm or weights, so plain Euclidean ‖y‖² is my interpretation**; squaring hides small errors, and it is blind to a pure scale error when `<Ax,y>` is small |
| symmetric relative | `|d| / (|<Ax,y>|+|<x,By>|+ε)` | bounded, symmetric, shows 1 when one side is 0 | unstable near zero: if both inner products are ≈ 0 by cancellation (random x, y are nearly orthogonal) a tiny error looks "large"; ε then dominates (unit test shows this) |
| Cauchy-Schwarz | `|d| / (‖Ax‖‖y‖)` | stable under cancellation; reads as "error relative to the largest possible inner product" | extra metric (not in the brief); can look good even when the dot product itself is wrong in relative terms |
| ratio | `<Ax,y>/<x,By>` | reveals a constant scale error | undefined at 0, sign flips make it meaningless for noisy pairs |

## 4. Experiments (reproduce: `python -m scripts.evaluation.adjoint_consistency`, ≈11 min CPU; seed 20260728; Python 3.10.20, torch 2.11.0+cpu, numpy 2.2.6, k-Wave-python 0.5.0rc1)
Outputs in `outputs/adjoint_consistency/`: `results.csv` (228 rows), `summary.csv`, `explicit_matrix.json`, `explicit_matrices.npz`, `adjoint_scatter.png`, `explicit_matrix.png`, `grid_size_sweep.txt`, `meta.json`.

**A. Dot-product test, 64×64, both conditions, 19 (x, y) pairs each:** 5 white-noise pairs; 3 smooth random x with white y; 3 phantoms (blob, disc, two blobs) with white y;
4 bundled MNIST p0 with white y; 4 bundled MNIST p0 with *realistic* y = k-Wave data of a *different* bundled sample.
No training/test leakage issue arises (no learning); the 4 MNIST samples are the only bundled ones.

**B. Explicit matrices, 12×12 grid** (smallest that k-Wave-python 0.5.0rc1 accepts: grids with < 256 mirrored points crash in its `uint8` index cast).
`A` is 204×144 (Ny·Nt × Nx·Ny), `B` 144×204, built from unit vectors; compared A^T with B.

**C. Grid-size sweep** with smooth inputs, N = 12…64 (`grid_size_sweep.txt`) — added after A and B seemed to disagree. Hypothesis-driven, one blob configuration only.

## 5. Results

### 5.1 Control
`fpat_autograd`: sym_rel median 4e-16 (periodic), 2e-16 (PML); explicit matrix mismatch 3e-16. The harness and metrics are correct.

### 5.2 k-Wave pair (Euclidean inner products)
| Input type | sym_rel median (max) | ratio `<Ax,y>/<x,By>` range | cs_rel median |
|---|---|---|---|
| smooth x, white y (periodic / PML) | 0.009 (0.021) / 0.004 (0.014) | 0.995–1.044 / 1.003–1.027 | 3e-4 / 1e-4 |
| phantoms | 0.007 (0.054) / 0.011 (0.026) | 0.986–1.115 / 0.98–1.05 | 1e-4 |
| MNIST x, white y | 0.024 / 0.024 | 0.87–1.07 / 0.96–1.21 | 5e-4 |
| **MNIST x, realistic y** | **0.0018 (0.003) / 0.0026 (0.018)** | **0.995–1.004 / 1.002–1.036** | 1.3e-3 / 1.0e-3 |
| white x, white y | 0.32 (1.0) / 0.47 (0.80) | −0.36…2.25 / 1.3…8.9 | 0.006 / 0.008 |

* For smooth, physically plausible inputs the k-Wave pair satisfies the *Euclidean* dot-product test to ≈ 0.1–2 % with ratio ≈ 1.00.
* With physical weights `wx = dx·dy, wy = dy·dt` the ratio becomes ≈ 6.8e-4 = 1/c × 1.02, i.e. far worse. So the `ρ c dx/(4dt)` / `1/(ρc²)` scalings **already contain the quadrature factors**; adding more weights breaks the identity. Use plain sums for this pair.
* For white-noise x and y the test fails badly (ratios from −0.36 to 8.9). Note that sym_rel and cs_rel disagree in magnitude: random x, y are nearly orthogonal (`|<Ax,y>|/(‖Ax‖‖y‖) ≈ 0.01`), so a 1 %-of-norm error is a ~100 % error of the inner product. Both statements are true; which matters depends on the use (gradients of smooth images vs. adversarial rough directions).
* Grid sweep (smooth inputs, realistic y): ratio 1.030 → 1.000 and sym_rel 1.5e-2 → 2e-4 as N goes 12 → 64 (periodic; PML similar). Caveat: the blobs are also smoother in pixels at larger N, so grid size and smoothness are confounded.
* Explicit 12×12 matrices: `A^T` vs `B` has relative Frobenius error 0.62 (periodic) / 0.61 (PML); after the best scalar (s = 1.37 / 1.38) still 0.58 / 0.57; cosine similarity 0.82. Per-singular-mode ratio `<v_i, B u_i>/σ_i` ranges from ≈ 0 (strongest mode) to 0.85 in the top 8 modes, and only 7 / 2 of 144 modes lie within 10 % of 1. The scale s = 1.37 on this grid contrasts with ≈ 1.00 for smooth inputs: the explicit matrix is dominated by grid-scale (rough) components.
* **Conclusion:** `kwave_adjoint_2d` is **not the exact discrete transpose** of `kwave_forward_2d`. It is a consistent discretisation of the continuous adjoint that is accurate for smooth/band-limited data and degrades for grid-scale content. *Hypothesis (not isolated):* the mismatch arises from the first-order time-reversal source scaling, staggered-grid and k-space correction differences, mirrored-grid summation, and time-sampling in the source; I did not test which one.

### 5.3 Fourier pairs
* `fourier_np_raw` and `fpat_torch_raw`: ratio ≈ 1e-6 (7e-7 – 1.2e-6 median), sym_rel = 1. Explicit-matrix best scalar s = 8.4e-7 / 7.8e-7 (≈ 1.9/c²). They are far from adjoint in raw form.
* With the `c`-convention (`fourier_np_c`): ratio of order 1 (median 1.2–2.9, range −9…59), sym_rel 0.13–0.63 median; realistic y: cs_rel 0.12–0.25. Explicit matrices: after the best scalar (1.9 / 1.7) the shape residual is 0.52 / 0.29 (cosine 0.85 / 0.96). So the `c` convention fixes **magnitude** (10⁶ → O(1)) but the Fourier pair is **not an adjoint pair, even up to a scalar**.
* This agrees with the library docstring in `differentiable.py` ("the constant-weight reference adjoint … must not be interpreted as that numerical transpose"); the adjoint uses a constant factor `c`, the forward a mode-dependent weight `w/(c²kx)`. The numbers confirm it: autograd of `fpat_forward_2d` is the true transpose, `fpat_adjoint_2d` is not.
* numpy vs torch Fourier forward matrices differ by 7–8 % after scaling (interpolation).
* Mixed pair (k-Wave forward, Fourier adjoint): poor (sym_rel median 0.13–1.0; matrix cosine 0.57 / 0.32). Not suitable as a reference.
* k-Wave forward vs `c·Fourier` forward matrices at 12×12: cosine 0.61 / 0.35, in line with the large shape error found in Stage 1.

## 6. Interpretation for the project
1. **Reference pair for later stages:** k-Wave forward/adjoint, with Euclidean inner products, is the only existing numerical pair that is adjoint to a useful accuracy, **but only on smooth/realistic inputs**. Report it as "approximate adjoint, error ≈ 10⁻³–10⁻² (sym_rel) on realistic data", never as exact.
2. **Exact transpose:** autograd through the torch Fourier forward is exact, yet that forward is a poor model of k-Wave (Stage 1). For an exact reference of the *Fourier* model use autograd, not `fpat_adjoint_2d`.
3. **Learned adjoint design:** because the k-Wave adjoint is not the exact transpose, a learned `G_φ` trained on its outputs inherits this error; "adjoint consistency of `G_φ` vs the physical forward" cannot be better than the reference's own consistency (≈ 10⁻³ on smooth data, much worse on rough y). Random white y are a poor training distribution for this reason.
4. Metrics: use `sym_rel` and `cs_rel` together; report the ratio to expose scale errors; never rely on Marta's squared metric alone because its normalisation is not fixed by the notes.

## 7. Limitations and unresolved questions
* Explicit matrices only on 12×12 (k-Wave constraint) and a single set of smooth test blobs for the grid sweep; behaviour at 64×64 for rough modes is not matrix-verified.
* The A/B discrepancy sources inside k-Wave are not isolated (hypotheses only).
* White-noise inputs are unphysical; results on them are diagnostic only.
* The 4 MNIST samples are the only bundled ones; only 3 hand-made phantoms and 8 random smooth/white seeds per condition.
* Marta's metric uses an unspecified norm; Euclidean assumed.
* k-Wave runs in single precision (`DataCast: single`), so floors of ~1e-6 relative are expected; observed errors are well above this.
* Which acquisition geometry the Fourier operators should be paired with is untested: k-Wave has no angle mask, Fourier does (89° / 45°).

## 8. Questions for Marta
1. Is an *approximate* k-Wave adjoint (accurate on smooth data, not the exact transpose) acceptable as the reference for training `G_φ`, or should the exact transpose (e.g. an explicit/auto-differentiated discrete k-Wave) be required? (differentiable k-Wave / JAX was on her list.)
2. Confirm the norm in the error `|<Ax,y>−<x,A*y>|²/‖y‖²`: Euclidean or weighted?
3. Should the adjoint be judged with Euclidean inner products? (k-Wave's scalings imply yes; the Fourier pair would need `wy/wx ≈ c²/1.9`, which has no clear physical meaning.)
