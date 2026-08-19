# PDEBench 2D diffusion-reaction probe (2026-08-19) — NEGATIVE, model-class limited

Ran the narrow claim (PDE-form kernels vs generic kernels, everything else held
fixed) on real PDEBench `2D_diff-react_NA_NA.h5`.  **The gate fails, but not on
the kernel comparison** -- the functional *CP* model cannot represent the field
at all, so the kernel comparison is not yet meaningful.

## Evidence

Fully observed CP fit (no missing data, free non-periodic factors), one sample,
`[t=32, x=16, y=16]`:

| CP rank | relative error |
|---:|---:|
| 5  | 0.581 |
| 10 | 0.321 |
| 20 | 0.047 |

Multilinear (Tucker) rank at 95% energy: **[2, 11, 11]**.

Trained model, 30% observations, rank 5, 1500 steps:

| | value |
|---|---|
| observed-fit NRMSE | 0.62 |
| held-out NRMSE | 1.31 |
| learned noise std | 0.500 (clamped at its upper bound) |

Held-out NRMSE above 1.0 means it is worse than predicting the mean, which
`docs/SHARED_PROTOCOL.md` section 6 rejects outright.  Sweeping rank in
{3, 5, 10} and observation ratio in {0.10, 0.30} left every cell in 1.29--1.45.

## Diagnosis

Two-dimensional Turing patterns are isotropic blobs, not `x` tensor `y`
separable, so they are not low CP-rank; but their *multilinear* rank is small.
CP is the wrong host model for real 2D fields.  Our synthetic POC only worked
because its data was generated from a rank-2 CP.

This is a limitation of the host model, not of the operator-derived prior, and
it is the concrete reason to move the host from CP to a small-core Tucker --
which is also what FunBaT uses, so the drop-in story aligns.

Reproduce: `experiments/run_pdebench_kernel_comparison.py`.
