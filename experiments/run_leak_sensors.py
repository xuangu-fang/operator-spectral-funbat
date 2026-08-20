#!/usr/bin/env python3
"""Main experiment: sensors you can actually place, and what physics buys you.

Setting.  A room with several gas leaks.  The field obeys
``d_t u = Dx u_xx + Dy u_yy - r u`` with zero flux at the walls, and away from
the leaks it is source-free, so the boundary genuinely constrains the interior.
Sensors cannot be scattered uniformly through the room: they attach to walls, to
a strip along one wall, or to one instrumented patch.  Reconstruction is then
extrapolation into unobserved regions, which is where a smoothness prior has
nothing left to say and a PDE prior does.

Everything except the sensor layout is held fixed: one field family, one rank,
one step budget, one set of baselines.  The table is layout x method.
"""

from __future__ import annotations

import argparse, json, sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "experiments"))

from geoaware.operator_spectral_funbat import (  # noqa: E402
    ModeAdaptiveVariationalTucker, extended_generic_dictionary, nonnegative_cp_spectrum,
    normalize_spectrum_cosine, real_cosine_basis,
)
from forced_pde_solver import solve_multi_leak  # noqa: E402
from neural_functional_tucker import fit_neural_tucker  # noqa: E402

FIELD = dict(grid=(64, 64), diffusivity=(0.02, 0.006), reaction=0.04,
             sources=((0.30, 0.65, 0.09, 15.0), (0.70, 0.35, 0.07, 25.0),
                      (0.55, 0.75, 0.06, 40.0)),
             dt=0.6, burn_in=200, record_steps=64, background_noise=0.02)
# What the method is told: the equation's form with nominal coefficients that
# are deliberately not the generating ones.
NOMINAL = dict(diffusivity=(0.03, 0.012), reaction=0.06)
LENGTH_SCALES = (0.12, 0.32, 0.8)
BINS = (12, 12, 12)
RANKS = (8, 5, 5)


def neumann_eigenvalues(size: int) -> torch.Tensor:
    k = torch.arange(size, dtype=torch.float64)
    return (2 - 2 * torch.cos(np.pi * k / size)) * size ** 2


def operator_spectra(shape, dt: float, bins, atoms: int = 4):
    """Per-mode spectra from the discrete operator with nominal coefficients."""
    dx, dy = NOMINAL["diffusivity"]
    lam_x = neumann_eigenvalues(shape[1])[:bins[1]]
    lam_y = neumann_eigenvalues(shape[2])[:bins[2]]
    omega = np.pi * torch.arange(bins[0], dtype=torch.float64) / (shape[0] * dt)
    elliptic = NOMINAL["reaction"] + dx * lam_x[:, None] + dy * lam_y[None, :]
    joint = (1.0 / (omega[:, None, None].square() + elliptic[None].square())
             .clamp_min(1e-12)).float()
    separated = nonnegative_cp_spectrum(joint, rank=atoms, steps=1200, seed=17)
    return [normalize_spectrum_cosine(f) for f in separated.factors]


def matern_spectra(bins, length_scale: float):
    out = []
    for b in bins:
        k = torch.arange(b, dtype=torch.float32)
        s = (1 + (length_scale * k).square()).pow(-2.0)
        out.append(normalize_spectrum_cosine((s / s.sum())[None]))
    return out


def sensor_mask(shape, layout: str, budget: int, seed: int, device):
    """Observed and held-out indices for one sensor layout, at a fixed budget."""
    nt, nx, ny = shape
    generator = torch.Generator(device=device).manual_seed(seed + 7717)
    grid = torch.stack(torch.meshgrid(
        *[torch.arange(s, device=device) for s in shape], indexing="ij"), -1).reshape(-1, 3)
    x, y = grid[:, 1], grid[:, 2]
    if layout == "random":
        region = torch.ones(len(grid), dtype=torch.bool, device=device)
    elif layout == "wall_ring":
        region = (x < 2) | (x >= nx - 2) | (y < 2) | (y >= ny - 2)
    elif layout == "near_wall":
        # A band set slightly in from the wall, as instruments usually are.
        inner, outer = 3, 8
        depth = torch.minimum(torch.minimum(x, nx - 1 - x), torch.minimum(y, ny - 1 - y))
        region = (depth >= inner) & (depth < outer)
    elif layout == "one_wall_strip":
        region = x < 5
    elif layout == "corner_block":
        region = (x < 20) & (y < 20)
    else:
        raise ValueError(f"unknown layout {layout}")
    candidates = torch.nonzero(region, as_tuple=False).squeeze(-1)
    if len(candidates) < budget:
        raise ValueError(f"layout {layout} holds {len(candidates)} < budget {budget}")
    order = torch.randperm(len(candidates), generator=generator, device=device)
    keep = torch.zeros(len(grid), dtype=torch.bool, device=device)
    keep[candidates[order[:budget]]] = True
    return grid[keep], grid[~keep]


def fit_gp(field, observed, targets, test, truth, spectra, bases, *, steps, seed, device, lr):
    torch.manual_seed(seed + 10_000)
    model = ModeAdaptiveVariationalTucker(
        tuple(torch.arange(s, device=device) / s for s in field.shape),
        [s.to(device) for s in spectra], ranks=RANKS, routing="global", noise_std=0.08,
        basis=("operator", "operator", "operator"),
        eigenbasis=tuple(b.to(device) for b in bases)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss, _ = model.negative_elbo(observed, targets, total_count=len(targets), samples=3)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        optimizer.step()
    with torch.no_grad():
        prediction = model.posterior_mean(test)
        return float(torch.sqrt(torch.mean((prediction - truth).square()))
                     / truth.std().clamp_min(1e-8))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--layouts", nargs="+",
                        default=["random", "wall_ring", "near_wall",
                                 "one_wall_strip", "corner_block"])
    parser.add_argument("--ratio", type=float, default=0.01)
    parser.add_argument("--noise-std", type=float, default=0.05)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--neural-steps", type=int, default=1500)
    parser.add_argument("--lr", type=float, default=0.02)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--tag", default="leak_sensors")
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "leak")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    probe = solve_multi_leak(seed=0, **FIELD)
    shape = tuple(probe.field.shape)
    budget = int(round(args.ratio * int(np.prod(shape))))
    ours = operator_spectra(shape, FIELD["dt"], BINS)
    dictionary = [normalize_spectrum_cosine(extended_generic_dictionary(4, b - 1)[1])
                  for b in BINS]
    bases = tuple(real_cosine_basis(torch.arange(s, dtype=torch.float64) / s, b).float()
                  for s, b in zip(shape, BINS))

    records = []
    for layout in args.layouts:
        rows: dict[str, list[float]] = {}
        for seed in args.seeds:
            solved = solve_multi_leak(seed=seed, **FIELD)
            field = solved.field.to(device)
            observed, test = sensor_mask(shape, layout, budget, seed, device)
            generator = torch.Generator(device=device).manual_seed(seed + 991)
            targets = field[tuple(observed.T)] + args.noise_std * torch.randn(
                len(observed), generator=generator, device=device)
            truth = field[tuple(test.T)]

            def gp(spectra):
                return fit_gp(field, observed, targets, test, truth, spectra, bases,
                              steps=args.steps, seed=seed, device=device, lr=args.lr)

            rows.setdefault("ours_pde", []).append(gp(ours))
            rows.setdefault("matern", []).append(min(
                gp(matern_spectra(BINS, ls)) for ls in LENGTH_SCALES))
            rows.setdefault("spectral_mixture", []).append(gp(dictionary))
            rows.setdefault("neural_tucker", []).append(fit_neural_tucker(
                shape, observed, targets, test, truth, ranks=RANKS,
                steps=args.neural_steps, seed=seed, device=device))
        cell = {"layout": layout, "observed": budget}
        for name, values in rows.items():
            values = np.array(values)
            cell[name] = {"mean": float(values.mean()), "std": float(values.std()),
                          "values": values.tolist()}
        best_other = min(cell[k]["mean"] for k in
                         ("matern", "spectral_mixture", "neural_tucker"))
        cell["margin_vs_best_baseline"] = best_other - cell["ours_pde"]["mean"]
        cell["relative_percent"] = 100 * cell["margin_vs_best_baseline"] / best_other
        records.append(cell)
        print(f"  {layout:16s} ours {cell['ours_pde']['mean']:.4f}  "
              f"matern {cell['matern']['mean']:.4f}  "
              f"mixture {cell['spectral_mixture']['mean']:.4f}  "
              f"neural {cell['neural_tucker']['mean']:.4f}  "
              f"margin {cell['margin_vs_best_baseline']:+.4f} "
              f"({cell['relative_percent']:+.1f}%)", flush=True)

    (args.output / f"{args.tag}_summary.json").write_text(json.dumps(
        {"field": FIELD, "nominal_prior": NOMINAL, "bins": BINS, "ranks": RANKS,
         "config": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
         "records": records}, indent=2))


if __name__ == "__main__":
    main()
