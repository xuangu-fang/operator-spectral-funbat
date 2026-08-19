# Operator-Spectral FunBaT

Bayesian functional tensor learning with operator-induced, mode-adaptive Gaussian-process kernels.

This repository is Track 3 of the [Physics-Informed Tensor Learning Hub](https://github.com/xuangu-fang/Geo-Aware-Tensor). It contains two deliberately separated levels:

1. **Kernel dictionary sanity:** a global nonnegative mixture of Matérn/resolvent, heat, geodesic, and Euclidean kernels learned with a finite-feature ELBO.
2. **Advanced POC:** derive a joint physical spectrum from a PDE/operator, separate it into positive mode-wise spectra, and let different functional tensor modes/ranks adapt different GP kernels.

The dictionary is a tested optimization and identifiability scaffold, not the final novelty claim.

## The question this project asks

Not *do you know the solution*, but **how much do you know about the operator that governs
the field**?  That admits a ladder, and the level that matters in practice is not the top one:

| Level | What is known | Bank construction |
|---|---|---|
| K2 | PDE form **and** coefficients | separate a single `S_op(theta*)` |
| **K1** | **PDE form only, coefficients unknown** | draw `theta_1..theta_M` from a declared range, separate each, pool the atoms |
| K0 | physics class only | as above, wider prior |
| K-1 | nothing | generic dictionary |

The claim K1 licenses: *we do not need you to know the coefficients, only which equation it
is.*  All results below are at **K2** and validate the mechanism; the K1 experiment is
designed but not yet run (see `docs/PAPER_TECHNICAL_REPORT_ZH.md` section 10).

## Current POC result (level K2)

The submission confirmation uses a bank-size-independent, collapsed spectral
mixture posterior, five untouched seeds (201--205), 2% observations and a fixed
400-step budget.  On the mathematically matched anisotropic-diffusion case,
operator per-mode/rank kernels reach `0.118±0.058` NRMSE versus
`0.157±0.099` for generic per-mode/rank kernels (5/5 paired wins), while matching
the oracle mean.  Induced-spectrum cosine improves from `0.926` to `0.977`.

The strict audit identifies a second, narrower contribution.  A free generic
dictionary is not a support guarantee; reserving a fixed 25% generic spectral
floor repairs a deleted-support prior from `0.615` to `0.130` NRMSE on
anisotropic diffusion (5/5 wins), with a matched-prior cost of about `0.012`.
This is a robustness--specificity tradeoff, not automatic kernel discovery.
Full signed-grid separation still exposes tilted advection coupling (rank-4
error about `0.18`, versus `0.0043` for anisotropic diffusion); advection is a
limitation, not the main example. Historical expanded-feature results remain
separate in `results/advanced_poc_r1_r5/`.

![submission confirmation](results/submission_confirmation/submission_confirmation_nrmse.png)

## Target formulation

The method has exactly two components.

**M1 — operator knowledge to a valid kernel bank.** For `L u = w`, construct
`S_op(omega; theta) = |L_hat(omega; theta)|^-2 S_w(omega)` and approximate its even/magnitude
component as a nonnegative sum of separable spectra. Each one-dimensional nonnegative
spectrum is a valid PSD stationary kernel, so nonnegative CP -> parameter pooling -> routing
softmax preserves PSD throughout. The knowledge level only decides *which* `theta` feed this
chain; the chain itself is unchanged.

**M2 — mixture under a guaranteed support floor.** A convex combination cannot create
frequency support that no atom has, so a fixed, non-closable fraction of generic spectral
mass is reserved. It is insurance for descending the ladder.

Everything else (CP likelihood, finite-Fourier ELBO+SGD, the collapsed parameterisation,
routing granularity) is model machinery or a fairness control, not a claim. The current
real-feature implementation does not represent full signed cross-mode phase, which is why the
headline family is anisotropic reaction-diffusion rather than advection.

## Repository map

- `src/geoaware/domain_kernels.py`: geometry/operator kernel features.
- `src/geoaware/variational_domain_gp.py`: explicit finite-feature variational GP.
- `experiments/track3_*`: migrated dictionary and residual experiments.
- `results/`: migrated kernel-dictionary evidence.
- `docs/PAPER_TECHNICAL_REPORT_ZH.md`: paper-level Chinese Introduction, positioning, Method and confirmation report (current submission baseline; GitHub-renderable math).
- `paper/`: LaTeX project for the paper itself (`cd paper && make`); notation macros in `paper/macros.tex` mirror the Chinese report's symbol table.
- `docs/TECHNICAL_REPORT.md`: historical expanded POC and earlier domain-kernel baseline.
- `docs/SUBMISSION_PLAN_ZH.md`: two-week submission plan — prior-not-model positioning, the pre-declared dataset screen (including rejected datasets), the four main experiments, and three kill gates.
- `docs/DATASETS_AND_RESOURCES.md`: local/shared data, official PDE resources,
  operator-spectrum audit fields, external-data baselines, and priority gates.
- `docs/ITERATIONS.md`: advanced-POC research diary.
- `docs/SHARED_PROTOCOL.md`: hub-level evaluation discipline.

Matched/operator-friendly data is labeled as mechanism sanity. The next publication gate requires a PDE solution dataset not sampled directly from the same finite atom family.
