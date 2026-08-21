#!/usr/bin/env python3
"""Neural completions as the strong baseline, given every advantage we can give.

The honest threat to this paper is not a stationary kernel with a slightly
better length scale.  It is a model with enough capacity to learn the structure
our prior supplies.  So the neural arms here are handled more generously than
our own:

  * each gets a small architecture sweep, and the reported number is the best
    configuration scored on the *held-out region itself* -- an oracle, marked
    with a star, that no practitioner could run;
  * each gets several times our optimisation budget, mini-batched, with a cosine
    schedule;
  * our arm gets what it always gets: spectra read off the equation with
    coefficients wrong by 50%, no tuning data, one configuration.

If a neural completion closes the gap under those terms, the prior is not
earning its place and the paper should say so.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np, torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "experiments"))
from geoaware.operator_spectral_funbat import real_cosine_basis  # noqa: E402
from forced_pde_solver import solve_multi_leak  # noqa: E402
from neural_baselines import CoSTCo, FourierMLP, fit_neural  # noqa: E402
from neural_functional_tucker import fit_neural_tucker  # noqa: E402
import run_leak_sensors as base  # noqa: E402

COSTCO = [dict(rank=16, channels=32), dict(rank=32, channels=64), dict(rank=8, channels=128)]
FOURIER = [dict(bands=6, width=256, depth=4), dict(bands=10, width=256, depth=4),
           dict(bands=8, width=512, depth=5)]
RATES = (3e-4, 1e-3)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--layouts", nargs="+",
                   default=["random", "one_wall_strip", "corner_block"])
    p.add_argument("--ratio", type=float, default=0.01)
    p.add_argument("--noise-std", type=float, default=0.05)
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--neural-steps", type=int, default=6000)
    p.add_argument("--lr", type=float, default=0.02)
    p.add_argument("--tag", default="neural_strong")
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
    print(f"  tensor {shape}, {budget} observed, neural budget {a.neural_steps} steps",
          flush=True)

    records = []
    for layout in a.layouts:
        rows: dict[str, list] = {}
        picked: dict[str, list] = {}
        for seed in a.seeds:
            field = solve_multi_leak(seed=seed, **base.FIELD).field.to(device)
            observed, test = base.sensor_mask(shape, layout, budget, seed, device)
            g = torch.Generator(device=device).manual_seed(seed + 991)
            targets = field[tuple(observed.T)] + a.noise_std * torch.randn(
                len(observed), generator=g, device=device)
            truth = field[tuple(test.T)]

            rows.setdefault("ours_pde", []).append(
                base.fit_gp(field, observed, targets, test, truth, ours, bases,
                            steps=a.steps, seed=seed, device=device, lr=a.lr))

            for name, configs, builder in (
                ("costco", COSTCO, lambda c: CoSTCo(shape, **c)),
                ("fourier_mlp", FOURIER, lambda c: FourierMLP(shape, **c)),
            ):
                scores = {}
                for config in configs:
                    for rate in RATES:
                        scores[(tuple(sorted(config.items())), rate)] = fit_neural(
                            builder(config), observed, targets, test, truth,
                            steps=a.neural_steps, lr=rate, seed=seed, device=device)
                best = min(scores, key=scores.get)
                rows.setdefault(f"{name}_oracle", []).append(scores[best])
                picked.setdefault(name, []).append({"config": dict(best[0]), "lr": best[1]})

            rows.setdefault("lrtfr_siren", []).append(fit_neural_tucker(
                shape, observed, targets, test, truth, ranks=base.RANKS,
                steps=a.neural_steps, seed=seed, device=device))

        cell = {"layout": layout, "observed": budget, "chosen_configs": picked}
        for key, values in rows.items():
            values = np.array(values)
            cell[key] = {"mean": float(values.mean()), "std": float(values.std()),
                         "values": values.tolist()}
        neural_best = min(cell[k]["mean"] for k in
                          ("costco_oracle", "fourier_mlp_oracle", "lrtfr_siren"))
        cell["ours_vs_best_neural"] = neural_best - cell["ours_pde"]["mean"]
        records.append(cell)
        print(f"  {layout:16s} ours {cell['ours_pde']['mean']:.4f}   "
              f"CoSTCo* {cell['costco_oracle']['mean']:.4f}   "
              f"FourierMLP* {cell['fourier_mlp_oracle']['mean']:.4f}   "
              f"LRTFR/SIREN {cell['lrtfr_siren']['mean']:.4f}   "
              f"ours vs best neural {cell['ours_vs_best_neural']:+.4f}", flush=True)
        for name, choices in picked.items():
            print(f"  {'':16s} {name} chose {choices}", flush=True)

    (a.output / f"{a.tag}_summary.json").write_text(json.dumps(
        {"seeds": a.seeds, "neural_steps": a.neural_steps, "records": records}, indent=2))


if __name__ == "__main__":
    main()
