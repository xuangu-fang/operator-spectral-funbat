#!/usr/bin/env python3
"""Does the PDE prior start to pay off as data gets scarcer or noisier?

On the anisotropic-diffusion field all three methods tie at 1% observed and low
noise -- the field's covariance is close to Matern, so the operator has little
to add.  The natural question is whether that changes in the regime a prior is
supposed to be for.  If the margin stays flat as observations fall and noise
rises, the honest conclusion is that this field simply does not need physics.
"""

from __future__ import annotations

import argparse, json, sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "experiments"))

from geoaware.operator_spectral_funbat import real_cosine_basis  # noqa: E402
from forced_pde_solver import solve_forced_spectral  # noqa: E402
from neural_functional_tucker import fit_neural_tucker  # noqa: E402
from run_highres_baselines import (  # noqa: E402
    TRUE_SETTING, fixed_spectrum, make_task, operator_bank, train,
)

LENGTH_SCALES = (0.12, 0.32, 0.8)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--ratios", type=float, nargs="+",
                        default=[0.002, 0.005, 0.01, 0.02])
    parser.add_argument("--noises", type=float, nargs="+", default=[0.05, 0.3])
    parser.add_argument("--bins", type=int, nargs=3, default=[16, 12, 12])
    parser.add_argument("--ranks", type=int, nargs=3, default=[12, 6, 6])
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--neural-steps", type=int, default=1500)
    parser.add_argument("--lr", type=float, default=0.02)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--tag", default="sparsity_noise")
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "highres")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    ranks, bins = tuple(args.ranks), tuple(args.bins)

    probe = solve_forced_spectral(seed=0, **TRUE_SETTING)
    ours = operator_bank(bins, 4, probe.field.shape, TRUE_SETTING["dt"])
    bases = tuple(real_cosine_basis(torch.arange(s, dtype=torch.float64) / s, b).float()
                  for s, b in zip(probe.field.shape, bins))

    records = []
    for noise in args.noises:
        for ratio in args.ratios:
            rows: dict[str, list[float]] = {}
            for seed in args.seeds:
                solved = solve_forced_spectral(seed=seed, **TRUE_SETTING)
                field, obs, tgt, test, truth, _, _ = make_task(
                    solved.field, ratio, seed, noise, device)

                def fit(spectra):
                    return train(field, obs, tgt, test, truth, spectra, bases,
                                 ranks=ranks, steps=args.steps, seed=seed,
                                 device=device, lr=args.lr)[0]

                rows.setdefault("ours", []).append(fit(ours))
                # The Matern arm is given the best of a small length-scale grid
                # judged on the test set, i.e. an advantage it would not have.
                rows.setdefault("matern_oracle", []).append(min(
                    fit([fixed_spectrum("matern32", b, ls) for b in bins])
                    for ls in LENGTH_SCALES))
                rows.setdefault("neural_tucker", []).append(fit_neural_tucker(
                    solved.field.shape, obs, tgt, test, truth, ranks=ranks,
                    steps=args.neural_steps, seed=seed, device=device))
            cell = {"ratio": ratio, "noise": noise,
                    "observed": int(round(ratio * int(np.prod(probe.field.shape))))}
            for name, values in rows.items():
                values = np.array(values)
                cell[name] = {"mean": float(values.mean()), "std": float(values.std())}
            best_other = min(cell["matern_oracle"]["mean"], cell["neural_tucker"]["mean"])
            cell["margin_vs_best_other"] = best_other - cell["ours"]["mean"]
            cell["relative_percent"] = 100 * cell["margin_vs_best_other"] / best_other
            records.append(cell)
            print(f"  noise {noise:4.2f} ratio {ratio:6.4f} (n={cell['observed']:5d})  "
                  f"ours {cell['ours']['mean']:.4f}  matern* {cell['matern_oracle']['mean']:.4f}  "
                  f"neural {cell['neural_tucker']['mean']:.4f}  "
                  f"margin {cell['margin_vs_best_other']:+.4f} "
                  f"({cell['relative_percent']:+.1f}%)", flush=True)

    (args.output / f"{args.tag}_summary.json").write_text(json.dumps(
        {"note": "matern arm is oracle-scanned over a small length-scale grid",
         "config": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
         "records": records}, indent=2))


if __name__ == "__main__":
    main()
