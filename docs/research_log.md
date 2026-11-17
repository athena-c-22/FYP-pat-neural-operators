# Research log

## 2026-11-16 — Repository audit and Stage 1 (operator scaling)

**Objective.** Audit the repo, then check whether Marta's scaling commit (`5eb43c6`, upstream) is consistent with the Python code and k-Wave.

**Audit (read-only).** Branch `experiment/forward-training`; only untracked item was `outputs/`. 104 unit tests pass (`pat_fno` env, ~50 s).
MATLAB sources are only on `origin/matlab-kwave`. Python has: k-Wave forward/adjoint/inverse (`src/pat_fno/operators/kwave`), Fourier forward/inverse/adjoint (NumPy + differentiable torch),
`TinyFNO2d`, forward-only training, and reconstruction scripts. There is no learned adjoint/inverse, no adjointness test, no gradient check. One checkpoint (FNO-only, periodic 89°, 1 epoch); only 4 sample test images; the full MNIST/k-Wave datasets are not in the repo.

**Decisions.**
* No operator source modified (no clear evidence of a bug).
* New script placed in `scripts/evaluation/`; outputs in untracked `outputs/operator_scaling/`.

**Commands.**
```
conda activate pat_fno
python -m scripts.evaluation.operator_scaling_validation
```
Seed 20260728. Python 3.10.20, torch 2.11.0+cpu, numpy 2.2.6, k-Wave-python 0.5.0rc1.

**Results (measured).** See `docs/operator_scaling_analysis.md`.
* Commit changes only an MATLAB example script: Fourier fwd ×c; adjoint/inverse get data/c.
* Python already follows that convention (since `8f80cb0`).
* 1-D analytic test: k-Wave exact (rel err 2.4e−6, periodic). `c·Fourier` too large by ≈ √2 (α = 0.674; 0.7075 when Nt = Nx).
* 2-D: adjoint ratio k-Wave/Fourier(K/c) ≈ 1.41 (periodic), 1.24 (PML) — hypothesis: same √2 mirror-normalisation factor; confounded by shape error.
* Large shape error (rel_fit ≈ 0.5) between Fourier and k-Wave regardless of scale.

**Limitations / open questions.** See §7–8 of the analysis: intent of the √2 factor; reference adjoint; inner-product weights.
Bug note: `build_setting` lacks `interpolationMethodA/I` (workaround in script).

**Next.** Await Marta's/the user's approval, then Stage 2 (adjoint consistency tests).

## 2026-11-16 — Stage 2 (adjoint consistency)

**Objective.** Find which existing forward/adjoint pairs are adjoint; build reusable metrics and tests.

**Implementation.** `src/pat_fno/evaluation/adjoint_consistency.py` (metrics, explicit matrix, matrix comparison); `tests/unit/test_adjoint_consistency.py` (7 tests, known linear operator); `scripts/evaluation/adjoint_consistency.py` (experiment). No operator code modified.

**Commands.**
```
python -m pytest tests/unit/test_adjoint_consistency.py -q
python -m scripts.evaluation.adjoint_consistency        # ~11 min CPU, outputs/adjoint_consistency/
```
Seed 20260728; torch 2.11.0+cpu; k-Wave-python 0.5.0rc1. A grid-size sweep (12–64) was run ad hoc and its output saved as `grid_size_sweep.txt` (snippet not stored as a script).

**Results (measured).** See `docs/adjoint_consistency_report.md`.
* Control (autograd of torch Fourier forward) exact: ~1e-16.
* k-Wave pair: ratio ≈ 1.00 and sym_rel 0.2–2 % on smooth/realistic inputs (Euclidean weights; physical weights make it worse); fails on white noise; 12×12 explicit matrix: A^T vs B error 0.6 even after scaling => not the exact transpose.
* Fourier pairs: raw ratio ~1e-6; with the c-convention O(1) but not adjoint even up to scale (residual 0.3–0.5).

**Problems encountered.** k-Wave-python crashes on mirrored grids < 256 points (uint8 cast) => matrix grid 12×12. First run of Part B crashed on a transposition bug in my own comparison calls; fixed and rerun.

**Next.** Await approval for Stage 3 (gradient comparison). Decisions needed from Marta: see report §8.

## 2026-11-16 — Stage 3 (gradient comparison)

**Objective.** Compare autograd gradient through the FNO with the k-Wave reference `A*(Ax-b)`; verify autograd with finite differences.

**Implementation.** `src/pat_fno/evaluation/gradient_comparison.py`; `tests/unit/test_gradient_comparison.py` (6 tests); `scripts/evaluation/gradient_comparison.py`. **One existing file changed:** `src/pat_fno/models/fno.py` (`dtype=torch.cfloat` -> `transformed.dtype` in `SpectralConv2d`; identical for float32, enables float64). Unit suite: 117 passed.

**Commands.**
```
python -m scripts.evaluation.gradient_comparison --workers 8   # ~35 min CPU (Part 2); --skip-kwave ~1 min
```
Seed 20260728; checkpoint `examples/checkpoints/fno_only_periodic_large.pt`; 4 bundled samples x 4 test points.

**Results (measured).** See `docs/gradient_comparison_report.md`.
* Autograd matches float64 central differences (O(h^2), best rel. error ~2e-10).
* At truth, g_ref = 0 but ||g_ad|| = 3.5-6.3: surrogate bias.
* cos(g_ad, g_ref): 0.65-0.73 (half), 0.14-0.34 (smooth_pert), ~0 (noisy). Jacobian is the weaker part.
* Line-search step along -g_ad was an ascent direction at 2 of 4 noisy points.

**Problems.** float64 FNO impossible without the one-line dtype fix; `.to(float64)` drops complex weight imaginary parts. Part 2 slow (35 min); no progress output (should add). CORRECTION (later benchmark): k-Wave throughput ≈6.7 s per fwd+adj pair and does NOT scale with worker processes (1/4/8 workers: 6.7/7.0/6.6 s per pair); the 'oversubscription' explanation given earlier was a guess and is wrong/unverified.

**Next.** Await approval for Stage 4 (learned adjoint / inverse / joint-training design).

## 2026-11-16 — Stage 4 (learned adjoint; learned inverse and joint training design only)

**Objective.** Design learned adjoint / inverse / joint training; test a small learned adjoint on CPU.

**Implementation.** `src/pat_fno/models/learned_adjoint.py` (+ `tests/unit/test_learned_adjoint.py`, 4 tests), `scripts/data/generate_adjoint_pairs.py`, `scripts/train/train_learned_adjoint.py`, `scripts/evaluation/learned_adjoint_eval.py`; design in `docs/learned_operator_design.md`.

**Commands.**
```
python -m scripts.data.generate_adjoint_pairs --n 200 --out outputs/learned_adjoint/data   # ~24 min CPU
# for lam in 0 1 10, seed in 0 1 2:
python -m scripts.train.train_learned_adjoint --dot-weight $lam --seed $seed               # ~3 min each
python -m scripts.evaluation.learned_adjoint_eval --linesearch
PYTHONPATH=. python outputs/learned_adjoint/odd_symmetrised_check.py
```
Seed 20260728 (data/split), model seeds 0-2. 200 k-Wave pairs, 140/30/30.

**Results (measured).** See section 5 of the design doc.
* Learned adjoints (100k params) overfit and are worse than the existing Fourier adjoint: test rel. L2 0.73-0.79 vs 0.44; gradient cos at Stage 3 points 0.54/0.2/0.02 vs 0.93/0.90/0.63 (Fourier).
* The Fourier adjoint of the residual gives ~100% of the reference line-search decrease (despite not being an adjoint pair in the Stage 2 sense).
* lambda (dot-product loss) shows no benefit within seed spread.
* Learned G(0) != 0 (norm 2.1-3.4); odd-symmetrising raises cos at `half` 0.54 -> 0.74 without retraining.

**Problems / corrections.** k-Wave throughput measured at ~6.7-8 s per fwd+adj pair with no speed-up from workers (corrected the earlier oversubscription claim). A dot-loss test exposed a wrong docstring claim (loss is invariant to joint scaling of x, y, B y, not y alone); fixed. Training runs took ~3 min each, longer than the 1 min I guessed.

**Next.** Stage 4 complete as scoped; supervisor decisions in design doc section 4.
