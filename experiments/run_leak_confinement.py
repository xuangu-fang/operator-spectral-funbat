#!/usr/bin/env python3
"""The main figure: how much does physics buy as sensors get more confined?

One knob, one curve.  Sensors are drawn from a strip of the room whose width
shrinks from the whole domain down to a thin band against one wall.  At full
width the layout is a uniform random mask and reconstruction is interpolation;
as the strip narrows, more and more of the room lies beyond any sensor and
reconstruction becomes extrapolation.

The prediction is that the margin over a generic kernel is near zero at full
width -- random interpolation does not need physics -- and grows as the strip
narrows.  A flat curve would say the setting does not need physics either, and
a curve that turns over would locate where extrapolation becomes hopeless for
everyone.

The observation budget is held fixed at every width, so the curve is about
*where* the sensors are, never how many.
"""

from __future__ import annotations

import argparse, json, sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "experiments"))

from geoaware.operator_spectral_funbat import (  # noqa: E402
    extended_generic_dictionary, normalize_spectrum_cosine, real_cosine_basis,
)
from forced_pde_solver import solve_multi_leak  # noqa: E402
from neural_functional_tucker import fit_neural_tucker  # noqa: E402
from run_leak_sensors import (  # noqa: E402
    BINS, FIELD, LENGTH_SCALES, RANKS, fit_gp, matern_spectra, operator_spectra,
)


def strip_mask(shape, width: int, budget: int, seed: int, device):
    """Sensors confined to the first `width` columns; budget held fixed."""
    generator = torch.Generator(device=device).manual_seed(seed + 7717)
    grid = torch.stack(torch.meshgrid(
        *[torch.arange(s, device=device) for s in shape], indexing="ij"), -1).reshape(-1, 3)
    region = grid[:, 1] < width
    candidates = torch.nonzero(region, as_tuple=False).squeeze(-1)
    if len(candidates) < budget:
        raise ValueError(f"width {width} holds {len(candidates)} < budget {budget}")
    order = torch.randperm(len(candidates), generator=generator, device=device)
    keep = torch.zeros(len(grid), dtype=torch.bool, device=device)
    keep[candidates[order[:budget]]] = True
    return grid[keep], grid[~keep]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--widths", type=int, nargs="+", default=[64, 48, 32, 20, 12, 8, 5, 3])
    parser.add_argument("--ratio", type=float, default=0.01)
    parser.add_argument("--noise-std", type=float, default=0.05)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--neural-steps", type=int, default=1500)
    parser.add_argument("--lr", type=float, default=0.02)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--tag", default="confinement")
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
    for width in args.widths:
        rows: dict[str, list[float]] = {}
        for seed in args.seeds:
            solved = solve_multi_leak(seed=seed, **FIELD)
            field = solved.field.to(device)
            observed, test = strip_mask(shape, width, budget, seed, device)
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
        cell = {"strip_width": width, "fraction_of_room": width / shape[1],
                "observed": budget}
        for name, values in rows.items():
            values = np.array(values)
            cell[name] = {"mean": float(values.mean()), "std": float(values.std()),
                          "values": values.tolist()}
        best = min(cell[k]["mean"] for k in ("matern", "spectral_mixture", "neural_tucker"))
        cell["margin_vs_best_baseline"] = best - cell["ours_pde"]["mean"]
        cell["relative_percent"] = 100 * cell["margin_vs_best_baseline"] / best
        records.append(cell)
        print(f"  width {width:3d} ({100*width/shape[1]:5.1f}% of room)  "
              f"ours {cell['ours_pde']['mean']:.4f}  matern {cell['matern']['mean']:.4f}  "
              f"mixture {cell['spectral_mixture']['mean']:.4f}  "
              f"neural {cell['neural_tucker']['mean']:.4f}  "
              f"margin {cell['margin_vs_best_baseline']:+.4f} "
              f"({cell['relative_percent']:+.1f}%)", flush=True)

    (args.output / f"{args.tag}_summary.json").write_text(json.dumps(
        {"field": FIELD, "bins": BINS, "ranks": RANKS,
         "prediction": "margin near zero at full width and growing as the strip "
                       "narrows; flat would mean the setting does not need physics",
         "config": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
         "records": records}, indent=2))


if __name__ == "__main__":
    main()
