#!/usr/bin/env python3
"""Round four: does the advantage survive coefficient error, sparsity and noise?

The claim is that knowing the equation's *form* is enough, so the prior is built
from nominal coefficients that are already wrong by 50%.  This sweeps how wrong
they may be, and checks the two axes a prior is supposed to help on -- fewer
observations and more noise -- on the single-wall layout where the effect is
largest.

Three separate one-dimensional sweeps rather than a grid, so each curve answers
one question and the whole thing stays readable.
"""

from __future__ import annotations

import argparse, json, sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "experiments"))

from geoaware.operator_spectral_funbat import (  # noqa: E402
    extended_generic_dictionary, nonnegative_cp_spectrum, normalize_spectrum_cosine,
    real_cosine_basis,
)
from forced_pde_solver import solve_multi_leak  # noqa: E402
from neural_functional_tucker import fit_neural_tucker  # noqa: E402
from run_leak_sensors import (  # noqa: E402
    BINS, FIELD, LENGTH_SCALES, RANKS, fit_gp, matern_spectra, neumann_eigenvalues,
    sensor_mask,
)


def operator_spectra_scaled(shape, dt, bins, factor: float, atoms: int = 4):
    """Prior from the equation's form with the diffusivity scaled by `factor`.

    `factor = 1` reproduces the nominal coefficients used everywhere else, which
    are themselves already wrong relative to the generating ones.
    """
    dx, dy = 0.03 * factor, 0.012 * factor
    lam_x = neumann_eigenvalues(shape[1])[:bins[1]]
    lam_y = neumann_eigenvalues(shape[2])[:bins[2]]
    omega = np.pi * torch.arange(bins[0], dtype=torch.float64) / (shape[0] * dt)
    elliptic = 0.06 + dx * lam_x[:, None] + dy * lam_y[None, :]
    joint = (1.0 / (omega[:, None, None].square() + elliptic[None].square())
             .clamp_min(1e-12)).float()
    joint[0, 0, 0] = 0.0
    separated = nonnegative_cp_spectrum(joint, rank=atoms, steps=1200, seed=17)
    return [normalize_spectrum_cosine(f) for f in separated.factors]


def evaluate(sweep, values, *, seeds, layout, ratio, noise, steps, neural_steps,
             lr, device, shape, bases, dictionary, budget_of):
    records = []
    for value in values:
        ratio_here = value if sweep == "ratio" else ratio
        noise_here = value if sweep == "noise" else noise
        factor = value if sweep == "coefficient" else 1.0
        ours = operator_spectra_scaled(shape, FIELD["dt"], BINS, factor)
        budget = budget_of(ratio_here)
        rows: dict[str, list[float]] = {}
        for seed in seeds:
            solved = solve_multi_leak(seed=seed, **FIELD)
            field = solved.field.to(device)
            observed, test = sensor_mask(shape, layout, budget, seed, device)
            generator = torch.Generator(device=device).manual_seed(seed + 991)
            targets = field[tuple(observed.T)] + noise_here * torch.randn(
                len(observed), generator=generator, device=device)
            truth = field[tuple(test.T)]

            def gp(spectra):
                return fit_gp(field, observed, targets, test, truth, spectra, bases,
                              steps=steps, seed=seed, device=device, lr=lr)

            rows.setdefault("ours_pde", []).append(gp(ours))
            rows.setdefault("matern", []).append(min(
                gp(matern_spectra(BINS, ls)) for ls in LENGTH_SCALES))
            rows.setdefault("spectral_mixture", []).append(gp(dictionary))
            rows.setdefault("neural_tucker", []).append(fit_neural_tucker(
                shape, observed, targets, test, truth, ranks=RANKS,
                steps=neural_steps, seed=seed, device=device))
        cell = {sweep: value, "observed": budget}
        for name, series in rows.items():
            series = np.array(series)
            cell[name] = {"mean": float(series.mean()), "std": float(series.std()),
                          "values": series.tolist()}
        best = min(cell[k]["mean"] for k in ("matern", "spectral_mixture", "neural_tucker"))
        cell["margin_vs_best_baseline"] = best - cell["ours_pde"]["mean"]
        cell["relative_percent"] = 100 * cell["margin_vs_best_baseline"] / best
        records.append(cell)
        print(f"  {sweep}={value:<8g} n={budget:5d}  ours {cell['ours_pde']['mean']:.4f}  "
              f"matern {cell['matern']['mean']:.4f}  "
              f"neural {cell['neural_tucker']['mean']:.4f}  "
              f"margin {cell['margin_vs_best_baseline']:+.4f} "
              f"({cell['relative_percent']:+.1f}%)", flush=True)
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--layout", default="one_wall_strip")
    parser.add_argument("--ratio", type=float, default=0.01)
    parser.add_argument("--noise-std", type=float, default=0.05)
    parser.add_argument("--ratios", type=float, nargs="+",
                        default=[0.003, 0.005, 0.01, 0.02, 0.05])
    parser.add_argument("--noises", type=float, nargs="+", default=[0.02, 0.05, 0.15, 0.4])
    parser.add_argument("--factors", type=float, nargs="+",
                        default=[0.1, 0.3, 1.0, 3.0, 10.0])
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--neural-steps", type=int, default=1500)
    parser.add_argument("--lr", type=float, default=0.02)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--tag", default="robustness")
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "leak")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    probe = solve_multi_leak(seed=0, **FIELD)
    shape = tuple(probe.field.shape)
    total = int(np.prod(shape))
    bases = tuple(real_cosine_basis(torch.arange(s, dtype=torch.float64) / s, b).float()
                  for s, b in zip(shape, BINS))
    dictionary = [normalize_spectrum_cosine(extended_generic_dictionary(4, b - 1)[1])
                  for b in BINS]
    common = dict(seeds=args.seeds, layout=args.layout, steps=args.steps,
                  neural_steps=args.neural_steps, lr=args.lr, device=device,
                  shape=shape, bases=bases, dictionary=dictionary,
                  budget_of=lambda r: int(round(r * total)))

    summary = {"layout": args.layout, "field": FIELD, "sweeps": {}}
    print("sweep 1: observation ratio")
    summary["sweeps"]["ratio"] = evaluate(
        "ratio", args.ratios, ratio=args.ratio, noise=args.noise_std, **common)
    print("sweep 2: observation noise")
    summary["sweeps"]["noise"] = evaluate(
        "noise", args.noises, ratio=args.ratio, noise=args.noise_std, **common)
    print("sweep 3: how wrong the nominal coefficients may be")
    summary["sweeps"]["coefficient"] = evaluate(
        "coefficient", args.factors, ratio=args.ratio, noise=args.noise_std, **common)
    (args.output / f"{args.tag}_summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
