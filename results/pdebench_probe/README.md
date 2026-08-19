# PDEBench 2D diffusion-reaction — REJECTED (2026-08-19)

Rejected on a **structural** criterion, not on "we tried and the numbers were
bad".  Turing patterns are intrinsically high-rank along each spatial axis:

| resolution | spatial rank95 | fraction of modes |
|---|---:|---:|
| full 128 x 128 | 81 / 128 | 63% |
| block-averaged to 16 | 10 / 16 | 62% |
| strided to 16 | 11 / 16 | 69% |

The fraction is invariant to resolution and to whether one strides or
block-averages, so this is not an aliasing artefact.  A field needing 63% of its
modes for 95% of its energy has an almost flat spectrum along each axis, and a
flat spectrum cannot be interpolated from a random subset by *any* low-rank
model.  Measured consequence: held-out NRMSE stayed in 1.08--1.45 for every
combination of host model (CP, Tucker), basis (periodic, Neumann cosine),
frequency budget (7--16 features) and observation ratio (10%, 30%).

## What was learned and kept

Two real fixes came out of this rejection and are retained:

1. **Host model CP -> Tucker.**  Real 2D fields are not `x` tensor `y`
   separable, so CP cannot represent them at any rank that sparse observations
   can identify, while their multilinear rank is small.  See
   `ModeAdaptiveVariationalTucker`.
2. **Periodic Fourier -> boundary-matched eigenbasis.**  Real initial-value data
   is not periodic in time: on this data the first-to-last-frame jump is 5.6x a
   typical adjacent step.  The eigenbasis of a stationary operator depends on
   the boundary condition; for no-flux boundaries it is the cosine basis.
   Switching the basis moved held-out NRMSE from 1.30 to 1.08 on an otherwise
   identical run, which is why it is now the default rather than a variant.

Neither fix rescues this dataset, because its problem is the rank of the field
rather than the model or the basis.

Reproduce: `experiments/run_pdebench_kernel_comparison.py`.
