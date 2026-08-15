# Operator-Spectral FunBaT

Bayesian functional tensor learning with operator-induced, mode-adaptive Gaussian-process kernels.

This repository is Track 3 of the [Physics-Informed Tensor Learning Hub](https://github.com/xuangu-fang/Geo-Aware-Tensor). It contains two deliberately separated levels:

1. **Kernel dictionary sanity:** a global nonnegative mixture of Matérn/resolvent, heat, geodesic, and Euclidean kernels learned with a finite-feature ELBO.
2. **Advanced POC:** derive a joint physical spectrum from a PDE/operator, separate it into positive mode-wise spectra, and let different functional tensor modes/ranks adapt different GP kernels.

The dictionary is a tested optimization and identifiability scaffold, not the final novelty claim.

## Current POC result

The complete operator-spectrum chain is now implemented with a finite Fourier
variational posterior and Gaussian ELBO. On a planted 24³ tensor, per-mode/rank
routing improves held-out NRMSE over one global dictionary at 1% observations
(`0.482` vs `0.593`) and matches the oracle by 5% (`0.0325` vs `0.0326`). Atom
top-1 recovery remains poor, so the project does **not** claim kernel discovery.

Rank-4 nonnegative separation reconstructs diffusion/advection/wave joint spectra
with relative errors `0.0028 / 0.0325 / 0.1079`. A wrong spectral-support prior
fails (`0.631` NRMSE), while adding generic dictionary atoms restores performance
(`0.068`). The recommended method is therefore an operator-derived kernel bank
with a generic misspecification escape route.

![routing phase](results/advanced_poc_r1_r5/r1_r2_routing_phase.png)

## Target formulation

For `L u = w`, construct `S_phys(omega) = |L_hat(omega)|^-2 S_w(omega)` and approximate it as a nonnegative sum of separable spectra. The inverse Fourier factors define valid one-dimensional GP kernels for the functional Tucker factors. ELBO+SGD jointly learns the variational posterior, tensor parameters, and constrained mode/rank routing weights.

## Repository map

- `src/geoaware/domain_kernels.py`: geometry/operator kernel features.
- `src/geoaware/variational_domain_gp.py`: explicit finite-feature variational GP.
- `experiments/track3_*`: migrated dictionary and residual experiments.
- `results/`: migrated kernel-dictionary evidence.
- `docs/TECHNICAL_REPORT.md`: complete technical baseline.
- `docs/ITERATIONS.md`: advanced-POC research diary.
- `docs/SHARED_PROTOCOL.md`: hub-level evaluation discipline.

Matched/operator-friendly data must always be labeled as mechanism sanity. A publishable claim requires mode-assignment recovery, kernel-swap controls, and a mismatched PDE layer.
