# Operator-Spectral Priors for Sparse Field Reconstruction

**In many physical settings the sensors cannot be placed where the answer is
wanted.** They mount on a wall, a duct, whatever surface is reachable, and the
interior is never observed. Reconstruction is then extrapolation rather than
interpolation, and a smoothness prior — which says only that distant points
correlate weakly — has nothing left to say about it.

This repository derives per-mode GP kernels from the **form** of the governing
equation (not its coefficients, not its solution) and drops them into a
functional tensor decomposition in place of its generic kernels. Model,
capacity, optimiser and step budget are held fixed; the only variable is where
the spectra come from.

---

## What this actually shows

### 1. Where sensors are scattered, physics is unnecessary. Where sensors are confined, tuning is impossible.

The paper's quantity is the **tuning cost**: what it costs to choose a kernel
using only data a practitioner can collect (a split of the sensor readings)
rather than against the held-out region (an oracle nobody can run).

| sensor layout | tuning cost, reaction–diffusion | diffusion-dominated | advection–diffusion |
|---|---|---|---|
| anywhere in the room | **+0.0000** | **+0.0008** | **+0.0031** |
| all four walls | +0.0519 | +0.0320 | +0.0046 |
| band inside the walls | +0.1219 | +0.0090 | +0.1154 |
| **one wall only** | **+0.2281** | **+0.3411** | **+0.1996** |
| one corner patch | +0.0726 | +0.0581 | +0.1501 |

The mechanism is visible in the hyper-parameter itself, not only in the error:
at one wall, validation on the sensors picks length scales of 0.5–0.8 while the
held-out region wants 1.6–2.4. Every point you can hold out lies inside the
same strip, so validation scores **interpolation** while deployment demands
**extrapolation**. In the 3-D room the same split picks 0.12 where the answer
wants 3.5 — an order of magnitude.

### 2. The same physics is worth ten times more as a prior than as a residual penalty

Both arms are told the equation's form with coefficients wrong by 50%. Neither
is told where the leaks are.

| sensor layout | ours (prior) | PINN* (residual penalty) | same network, physics off |
|---|---|---|---|
| anywhere in the room | 0.0547 | **0.0522** | 0.1174 |
| one wall only | **0.5387** | 0.8008 | 0.8229 |

At one wall the residual is worth 0.022 and on two of three seeds the oracle
sweep chose a residual weight of **zero**. A residual constrains the function at
collocation points, and the homogeneous equation has many solutions; a spectral
prior constrains which functions are *a priori* plausible, which is what is left
once the data has stopped speaking.

### 3. A declared range for the coefficients is nearly as good as knowing them

| what is known | one wall |
|---|---|
| the true coefficients | 0.5450 |
| **only that they lie in ×[1/3, 3]** | **0.5468** |
| only that they lie in ×[1/10, 10] | 0.5783 |
| no physics (generic, same atom count) | 0.7408 |

This is where the construction differs *in kind* from the alternatives: a PINN
must pick one operator to penalise and an AutoIP-style GP must pick one to
condition on. A pooled bank commits only to a range and lets the mixture weights
infer within it.

---

## What this does **not** show

Reported here because it is load-bearing for reading everything above.

- **We do not beat a well-tuned kernel; we match it.** Against a Matérn whose
  length scale is chosen on the held-out region, the one-wall case is a tie
  (0.5448 against 0.5449). An earlier draft claimed +17.2% and was wrong: the
  baseline's length-scale grid did not contain its own optimum. **Retracted.**
- **In 2-D, simply fixing a sensible constant nearly matches us.** A practitioner
  who never tunes and commits to ℓ = 1.6 is at most 0.0135 behind on any layout.
- **The failure mode is not graceful.** Across every experiment here, the only
  arm that ever scores worse than predicting the mean is ours (1.03 at a corner
  patch, 1.27 at the other wall); no baseline exceeds 0.95. A long-length-scale
  Matérn shrinks toward the mean and stops; our scale is fixed by the equation,
  so where the observations stop constraining the field it keeps extrapolating.
  **This needs a fallback before deployment and does not have one.**
- **A standard physics-informed GP is *faster* than us at this scale** (1.7–8.0 s
  against 20–25 s). Do not write that it "does not scale" at 2-D sizes.

---

## Repository map

| path | what it is |
|---|---|
| `docs/GETTING_STARTED_ZH.md` | **run something first** — from an empty machine to the main table in about twenty minutes |
| `docs/HANDOVER_TECHNICAL_ZH.md` | full derivation with algorithm boxes, related work, baselines, experiment settings, dataset provenance |
| `configs/*.yaml` | every setting worth varying; a new study is a YAML file, not an edit |
| `docs/FUTURE_BRANCHES_ZH.md` | parked directions and open puzzles |
| `src/geoaware/operator_spectral_funbat.py` | spectrum construction, nonnegative CP, Tucker host |
| `experiments/forced_pde_solver.py` | field generation, 2-D and 3-D |
| `experiments/run_leak_sensors.py` | main table, layout definitions, shared fit routine |
| `experiments/make_*.py` | every table and figure, generated from recorded JSON |
| `tools/check_github_math.py` | formula-rendering check; run after editing any document |
| `paper/` | LaTeX source, `make` to build |

## Reproducing the main table

No dataset download is required — the fields are solved locally in about a
second per seed.

```bash
python experiments/run_leak_sensors.py --config base --tag leak_main3tier
python experiments/make_paper_tables.py     # writes paper/sections/table_*.tex
python experiments/make_leak_figures.py     # writes results/leak/figure_*.png
```

A different field is a config, not a code change:

```bash
python experiments/run_leak_sensors.py --config advection_diffusion --tag advection
python experiments/run_leak_sensors.py --config base --set evaluation.ratio=0.02
```

Unknown keys are refused rather than ignored, and the resolved config is written
into every summary, so a number can always be traced back to what produced it.

A GPU is used automatically when available (`--device`), and makes the sweeps
roughly an order of magnitude cheaper.

## Settings that every table shares

```python
FIELD  = grid 64x64, D = (0.02, 0.006), r = 0.04, three leaks,
         dt = 0.6, 200 burn-in steps, 64 recorded frames, background noise 0.02
NOMINAL = D = (0.03, 0.012), r = 0.06        # deliberately wrong by 50%
BINS   = (12, 12, 12)   RANKS = (8, 5, 5)   observed 1%   noise std 0.05
Adam, lr 0.02, 1000 steps, 3-sample ELBO
```

Within a seed every arm shares the field, the mask and the noise, so every
comparison is paired. NRMSE is normalised by the held-out standard deviation,
so **1.0 is exactly what predicting the mean scores**.

## House rules that earned themselves

1. **Check that every hyper-parameter grid brackets its own optimum.** The one
   retraction in this project came from a grid that did not. Tuning a baseline
   *up* is an intervention a reader can see; tuning it *too little* produces a
   number that looks like a measurement.
2. **Write the mechanism prediction and its refutation condition into the script
   before running it.** Four mechanism predictions here have been refuted by
   their own controls, and each was easy to re-narrate afterwards.
3. **Generate tables and figures from recorded JSON.** Hand transcription is how
   a retracted number survives a revision.
4. **Run the feasibility screens before comparing methods**, not after.
