#!/usr/bin/env python3
"""Does the Matern baseline's length-scale grid bracket its own optimum?

The main table gives the Matern arm an oracle advantage: its length scale is
chosen per seed and per layout by the held-out error itself.  That is only a
real advantage if the grid contains the best value.  If the optimum sits on an
end point, the arm is grid-limited and any margin over it is partly an artefact
of where we stopped scanning.  This sweeps a wide grid and reports, per layout,
which value wins and how much the coarse grid costs.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np, torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "experiments"))
from geoaware.operator_spectral_funbat import (  # noqa: E402
    normalize_spectrum_cosine, real_cosine_basis)
from forced_pde_solver import solve_multi_leak  # noqa: E402
from run_leak_sensors import (  # noqa: E402
    FIELD, BINS, LENGTH_SCALES, fit_gp, matern_spectra, sensor_mask)

WIDE = (0.02, 0.04, 0.06, 0.09, 0.12, 0.18, 0.24, 0.32, 0.45, 0.6, 0.8, 1.1, 1.6, 2.4)

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    p.add_argument("--layouts", nargs="+",
                   default=["one_wall_strip", "near_wall", "corner_block", "random"])
    p.add_argument("--ratio", type=float, default=0.01)
    p.add_argument("--noise-std", type=float, default=0.05)
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--lr", type=float, default=0.02)
    p.add_argument("--output", type=Path, default=ROOT / "results" / "leak")
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu")

    shape = tuple(solve_multi_leak(seed=0, **FIELD).field.shape)
    budget = int(round(a.ratio * int(np.prod(shape))))
    bases = tuple(real_cosine_basis(torch.arange(s, dtype=torch.float64) / s, b).float()
                  for s, b in zip(shape, BINS))

    out = []
    for layout in a.layouts:
        curve = {ls: [] for ls in WIDE}
        for seed in a.seeds:
            field = solve_multi_leak(seed=seed, **FIELD).field.to(device)
            observed, test = sensor_mask(shape, layout, budget, seed, device)
            g = torch.Generator(device=device).manual_seed(seed + 991)
            targets = field[tuple(observed.T)] + a.noise_std * torch.randn(
                len(observed), generator=g, device=device)
            truth = field[tuple(test.T)]
            for ls in WIDE:
                curve[ls].append(fit_gp(field, observed, targets, test, truth,
                                        matern_spectra(BINS, ls), bases, steps=a.steps,
                                        seed=seed, device=device, lr=a.lr))
        means = {ls: float(np.mean(v)) for ls, v in curve.items()}
        wide_best_ls = min(means, key=means.get)
        coarse_best = min(means[ls] for ls in LENGTH_SCALES)
        rec = {"layout": layout, "means": means, "wide_best_ls": wide_best_ls,
               "wide_best": means[wide_best_ls], "coarse_best": coarse_best,
               "coarse_grid_cost": coarse_best - means[wide_best_ls],
               "optimum_on_grid_edge": wide_best_ls in (WIDE[0], WIDE[-1])}
        out.append(rec)
        print(f"  {layout:16s} wide-best ls={wide_best_ls:<5} {means[wide_best_ls]:.4f}   "
              f"coarse-grid {coarse_best:.4f}   the coarse grid costs "
              f"{rec['coarse_grid_cost']:+.4f}"
              f"{'   OPTIMUM ON EDGE' if rec['optimum_on_grid_edge'] else ''}", flush=True)
    (a.output / "baseline_lengthscale_audit.json").write_text(json.dumps(
        {"wide_grid": WIDE, "coarse_grid": LENGTH_SCALES, "seeds": a.seeds,
         "records": out}, indent=2))

if __name__ == "__main__":
    main()
