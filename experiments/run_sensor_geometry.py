#!/usr/bin/env python3
"""Sensors where you can actually put them: boundaries and small blocks.

Random masks are the statistically easiest case and they are not what the
motivating applications look like.  In gas-leak detection, combustion-chamber
modelling or structural monitoring, sensors attach to accessible surfaces --
walls, edges, one instrumented patch -- so the observed region is a small
contiguous part of the domain and the rest must be reconstructed by
*extrapolation*, not interpolation.

That is the regime where a smoothness prior has nothing to offer: far from any
sensor it reverts to the mean.  A PDE prior does have something to offer, and
for a dissipative operator the reason is not soft.  Boundary data determines the
interior; that is what a boundary-value problem *is*.  A generic stationary
kernel cannot express that relation at any length scale.

Three sensor layouts, all at matched observation counts so the comparison is
about *where* the sensors are, not how many:

  random    uniform over the tensor, the usual easy case
  boundary  a ring of spatial cells at the domain edge, all times
  block     one contiguous interior patch, all times
"""

from __future__ import annotations

import argparse, json, sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "experiments"))

from geoaware.operator_spectral_funbat import real_cosine_basis  # noqa: E402
from forced_pde_solver import solve_forced_spectral, solve_localized_source  # noqa: E402
from neural_functional_tucker import fit_neural_tucker  # noqa: E402
from run_highres_baselines import (  # noqa: E402
    TRUE_SETTING, fixed_spectrum, operator_bank, train,
)

LENGTH_SCALES = (0.12, 0.32, 0.8)


def sensor_mask(shape, layout: str, budget: int, seed: int, device):
    """Return observed and held-out index sets for a sensor layout.

    Every layout is subsampled to the same `budget`, so a layout cannot win by
    seeing more data.
    """
    nt, nx, ny = shape
    generator = torch.Generator(device=device).manual_seed(seed + 7717)
    grid = torch.stack(torch.meshgrid(
        *[torch.arange(s, device=device) for s in shape], indexing="ij"), -1).reshape(-1, 3)
    if layout == "random":
        region = torch.ones(len(grid), dtype=torch.bool, device=device)
    elif layout == "boundary":
        # Widen the ring until it can hold the budget, then subsample it.
        x, y = grid[:, 1], grid[:, 2]
        width = 1
        while True:
            region = ((x < width) | (x >= nx - width) |
                      (y < width) | (y >= ny - width))
            if int(region.sum()) >= budget or width >= min(nx, ny) // 2:
                break
            width += 1
    elif layout == "block":
        side = 1
        while True:
            lo_x, lo_y = (nx - side) // 2, (ny - side) // 2
            region = ((grid[:, 1] >= lo_x) & (grid[:, 1] < lo_x + side) &
                      (grid[:, 2] >= lo_y) & (grid[:, 2] < lo_y + side))
            if int(region.sum()) >= budget or side >= min(nx, ny):
                break
            side += 1
    else:
        raise ValueError(f"unknown layout {layout}")
    candidates = torch.nonzero(region, as_tuple=False).squeeze(-1)
    order = torch.randperm(len(candidates), generator=generator, device=device)
    observed_rows = candidates[order[:budget]]
    keep = torch.zeros(len(grid), dtype=torch.bool, device=device)
    keep[observed_rows] = True
    return grid[keep], grid[~keep]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--layouts", nargs="+", default=["random", "boundary", "block"])
    parser.add_argument("--ratio", type=float, default=0.01)
    parser.add_argument("--noise-std", type=float, default=0.05)
    parser.add_argument("--bins", type=int, nargs=3, default=[16, 12, 12])
    parser.add_argument("--ranks", type=int, nargs=3, default=[12, 6, 6])
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--neural-steps", type=int, default=1500)
    parser.add_argument("--lr", type=float, default=0.02)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--field", default="localized_source",
                        choices=["localized_source", "stochastic"],
                        help="a stochastically forced field's interior is not "
                             "determined by its boundary, so wall sensors cannot "
                             "work there for any method; a localized source is the "
                             "case the motivating applications actually have")
    parser.add_argument("--tag", default="sensor_geometry")
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "highres")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    ranks, bins = tuple(args.ranks), tuple(args.bins)

    def simulate(seed):
        if args.field == "localized_source":
            return solve_localized_source(seed=seed, grid=TRUE_SETTING["grid"],
                                          diffusivity=TRUE_SETTING["diffusivity"],
                                          reaction=TRUE_SETTING["reaction"],
                                          dt=TRUE_SETTING["dt"],
                                          record_steps=TRUE_SETTING["record_steps"])
        return solve_forced_spectral(seed=seed, **TRUE_SETTING)

    probe = simulate(0)
    shape = tuple(probe.field.shape)
    budget = int(round(args.ratio * int(np.prod(shape))))
    ours = operator_bank(bins, 4, shape, TRUE_SETTING["dt"])
    bases = tuple(real_cosine_basis(torch.arange(s, dtype=torch.float64) / s, b).float()
                  for s, b in zip(shape, bins))

    records = []
    for layout in args.layouts:
        rows: dict[str, list[float]] = {}
        for seed in args.seeds:
            solved = simulate(seed)
            field = solved.field.to(device)
            observed, test = sensor_mask(shape, layout, budget, seed, device)
            generator = torch.Generator(device=device).manual_seed(seed + 991)
            targets = field[tuple(observed.T)] + args.noise_std * torch.randn(
                len(observed), generator=generator, device=device)
            truth = field[tuple(test.T)]

            def fit(spectra):
                return train(field, observed, targets, test, truth, spectra, bases,
                             ranks=ranks, steps=args.steps, seed=seed,
                             device=device, lr=args.lr)[0]

            rows.setdefault("ours", []).append(fit(ours))
            rows.setdefault("matern_oracle", []).append(min(
                fit([fixed_spectrum("matern32", b, ls) for b in bins])
                for ls in LENGTH_SCALES))
            rows.setdefault("neural_tucker", []).append(fit_neural_tucker(
                shape, observed, targets, test, truth, ranks=ranks,
                steps=args.neural_steps, seed=seed, device=device))
        cell = {"layout": layout, "observed": budget,
                "ratio": budget / int(np.prod(shape))}
        for name, values in rows.items():
            values = np.array(values)
            cell[name] = {"mean": float(values.mean()), "std": float(values.std()),
                          "values": values.tolist()}
        best_other = min(cell["matern_oracle"]["mean"], cell["neural_tucker"]["mean"])
        cell["margin_vs_best_other"] = best_other - cell["ours"]["mean"]
        cell["relative_percent"] = 100 * cell["margin_vs_best_other"] / best_other
        records.append(cell)
        print(f"  {layout:9s} n={budget}  ours {cell['ours']['mean']:.4f}  "
              f"matern* {cell['matern_oracle']['mean']:.4f}  "
              f"neural {cell['neural_tucker']['mean']:.4f}  "
              f"margin {cell['margin_vs_best_other']:+.4f} "
              f"({cell['relative_percent']:+.1f}%)", flush=True)

    (args.output / f"{args.tag}_summary.json").write_text(json.dumps(
        {"note": "all layouts share the same observation budget; the matern arm is "
                 "oracle-scanned over a small length-scale grid",
         "config": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
         "records": records}, indent=2))


if __name__ == "__main__":
    main()
