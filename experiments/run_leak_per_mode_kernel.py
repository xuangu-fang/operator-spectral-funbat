#!/usr/bin/env python3
"""Is our advantage the operator, or merely one length scale per axis?

The Matern arm in every other table shares a single length scale across all
three modes, while our construction hands each mode its own spectrum.  That is a
real asymmetry in our favour, and it makes the remaining gap ambiguous: we may be
winning because the equation tells us the shape of each axis's spectrum, or
merely because it tells us the axes differ at all.

A practitioner needs no data to suspect the second.  "Time varies slowly, space
varies fast" is a judgment, not a measurement, so fixing three different
constants is as deployable as fixing one.

Three arms, all needing no tuning data except where starred:

  ours                 spectra from the operator, coefficients wrong by 50%.
  shared constant      one length scale for every mode, the value that was best
                       across layouts in the earlier sweep.
  per-mode constant*   a different constant per mode, the combination chosen on
                       the held-out region.  Starred because searching the grid
                       is an oracle; a practitioner would guess instead, so this
                       is an upper bound on what guessing per mode could buy.

If the per-mode oracle closes the gap to us, the operator's contribution is the
scale of each axis rather than the shape of its spectrum, and the paper should
say that plainly.
"""
from __future__ import annotations
import argparse, itertools, json, sys
from pathlib import Path
import numpy as np, torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "experiments"))
from geoaware.operator_spectral_funbat import (  # noqa: E402
    normalize_spectrum_cosine, real_cosine_basis)
from forced_pde_solver import solve_multi_leak  # noqa: E402
import run_leak_sensors as base  # noqa: E402

# Three values rather than four.  The full four-value grid is 64 combinations
# per seed per layout, which is several GPU-hours for a control question that a
# coarser grid answers just as well: whether letting the axes differ at all
# closes the gap, not which exact triple is optimal.
PER_MODE_GRID = (0.32, 1.0, 2.4)
SHARED = 1.6


def mixed_spectra(bins, scales):
    """One length scale per mode rather than one for all of them."""
    out = []
    for b, scale in zip(bins, scales):
        k = torch.arange(b, dtype=torch.float32)
        s = (1 + (scale * k).square()).pow(-2.0)
        out.append(normalize_spectrum_cosine((s / s.sum())[None]))
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--layouts", nargs="+", default=["random", "one_wall_strip"])
    p.add_argument("--ratio", type=float, default=0.01)
    p.add_argument("--noise-std", type=float, default=0.05)
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--lr", type=float, default=0.02)
    p.add_argument("--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="an idle GPU makes these sweeps roughly an order of magnitude cheaper")
    p.add_argument("--output", type=Path, default=ROOT / "results" / "leak")
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    device = torch.device(a.device)

    shape = tuple(solve_multi_leak(seed=0, **base.FIELD).field.shape)
    budget = int(round(a.ratio * int(np.prod(shape))))
    ours = base.operator_spectra(shape, base.FIELD["dt"], base.BINS)
    bases = tuple(real_cosine_basis(torch.arange(s, dtype=torch.float64) / s, b).float()
                  for s, b in zip(shape, base.BINS))
    combinations = list(itertools.product(PER_MODE_GRID, repeat=3))
    print(f"  {len(combinations)} per-mode combinations per seed per layout", flush=True)

    records = []
    for layout in a.layouts:
        rows: dict[str, list] = {}
        chosen: list = []
        for seed in a.seeds:
            field = solve_multi_leak(seed=seed, **base.FIELD).field.to(device)
            observed, test = base.sensor_mask(shape, layout, budget, seed, device)
            g = torch.Generator(device=device).manual_seed(seed + 991)
            targets = field[tuple(observed.T)] + a.noise_std * torch.randn(
                len(observed), generator=g, device=device)
            truth = field[tuple(test.T)]

            def fit(spectra):
                return base.fit_gp(field, observed, targets, test, truth, spectra, bases,
                                   steps=a.steps, seed=seed, device=device, lr=a.lr)

            rows.setdefault("ours_pde", []).append(fit(ours))
            rows.setdefault("shared_constant", []).append(
                fit(mixed_spectra(base.BINS, (SHARED,) * 3)))
            scores = {c: fit(mixed_spectra(base.BINS, c)) for c in combinations}
            best = min(scores, key=scores.get)
            rows.setdefault("per_mode_oracle", []).append(scores[best])
            chosen.append(list(best))

        cell = {"layout": layout, "observed": budget, "chosen_per_mode": chosen}
        for key, values in rows.items():
            values = np.array(values)
            cell[key] = {"mean": float(values.mean()), "std": float(values.std()),
                         "values": values.tolist()}
        cell["per_mode_beats_shared"] = (cell["shared_constant"]["mean"]
                                         - cell["per_mode_oracle"]["mean"])
        cell["ours_vs_per_mode_oracle"] = (cell["per_mode_oracle"]["mean"]
                                           - cell["ours_pde"]["mean"])
        records.append(cell)
        print(f"  {layout:16s} ours {cell['ours_pde']['mean']:.4f}   "
              f"shared l={SHARED} {cell['shared_constant']['mean']:.4f}   "
              f"per-mode oracle* {cell['per_mode_oracle']['mean']:.4f}   "
              f"per-mode buys {cell['per_mode_beats_shared']:+.4f}   "
              f"ours vs per-mode oracle {cell['ours_vs_per_mode_oracle']:+.4f}", flush=True)
        print(f"  {'':16s} chose {chosen}", flush=True)

    (a.output / "per_mode_kernel_summary.json").write_text(json.dumps(
        {"grid": PER_MODE_GRID, "shared": SHARED, "seeds": a.seeds,
         "records": records}, indent=2))


if __name__ == "__main__":
    main()
