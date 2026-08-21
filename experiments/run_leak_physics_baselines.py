#!/usr/bin/env python3
"""Two ways to spend the same physics, compared head to head.

Both arms are told the equation's form with coefficients wrong by 50%.  Ours
turns that into a prior over per-mode spectra.  The PINN arm turns it into a
penalty on the PDE residual of a coordinate network.  Nothing else differs in
what they know, so the comparison is about how the knowledge is used.

The PINN is treated generously, as the strong baselines are throughout: its
residual weight and learning rate are chosen on the held-out region itself,
which is an oracle no practitioner could run, and it gets several times our
optimisation budget.  A residual weight of zero is included in the sweep, so the
arm can decline the physics if it does not help -- that keeps the comparison
about the physics rather than about the architecture.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np, torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "experiments"))
from geoaware.operator_spectral_funbat import real_cosine_basis  # noqa: E402
from forced_pde_solver import solve_multi_leak  # noqa: E402
from pinn_baseline import fit_pinn  # noqa: E402
import run_leak_sensors as base  # noqa: E402

WEIGHTS = (0.0, 1e-4, 1e-3, 1e-2, 1e-1)
RATES = (3e-4, 1e-3)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--layouts", nargs="+",
                   default=["random", "one_wall_strip", "corner_block"])
    p.add_argument("--ratio", type=float, default=0.01)
    p.add_argument("--noise-std", type=float, default=0.05)
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--pinn-steps", type=int, default=4000)
    p.add_argument("--lr", type=float, default=0.02)
    p.add_argument("--output", type=Path, default=ROOT / "results" / "leak")
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu")

    shape = tuple(solve_multi_leak(seed=0, **base.FIELD).field.shape)
    budget = int(round(a.ratio * int(np.prod(shape))))
    ours = base.operator_spectra(shape, base.FIELD["dt"], base.BINS)
    bases = tuple(real_cosine_basis(torch.arange(s, dtype=torch.float64) / s, b).float()
                  for s, b in zip(shape, base.BINS))

    records = []
    for layout in a.layouts:
        rows: dict[str, list] = {}
        picked: list = []
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

            scores = {}
            for weight in WEIGHTS:
                for rate in RATES:
                    scores[(weight, rate)] = fit_pinn(
                        shape, observed, targets, test, truth,
                        diffusivity=base.NOMINAL["diffusivity"],
                        reaction=base.NOMINAL["reaction"], dt=base.FIELD["dt"],
                        residual_weight=weight, steps=a.pinn_steps, lr=rate,
                        seed=seed, device=device)
            best = min(scores, key=scores.get)
            rows.setdefault("pinn_oracle", []).append(scores[best])
            # What the same network scores with the physics switched off, so the
            # residual's own contribution is visible rather than inferred.
            rows.setdefault("network_no_physics", []).append(
                min(scores[(0.0, rate)] for rate in RATES))
            picked.append({"residual_weight": best[0], "lr": best[1]})

        cell = {"layout": layout, "observed": budget, "chosen": picked}
        for key, values in rows.items():
            values = np.array(values)
            cell[key] = {"mean": float(values.mean()), "std": float(values.std()),
                         "values": values.tolist()}
        cell["residual_buys"] = (cell["network_no_physics"]["mean"]
                                 - cell["pinn_oracle"]["mean"])
        cell["ours_vs_pinn"] = cell["pinn_oracle"]["mean"] - cell["ours_pde"]["mean"]
        records.append(cell)
        print(f"  {layout:16s} ours {cell['ours_pde']['mean']:.4f}   "
              f"PINN* {cell['pinn_oracle']['mean']:.4f}   "
              f"same net, no physics {cell['network_no_physics']['mean']:.4f}   "
              f"the residual buys {cell['residual_buys']:+.4f}   "
              f"ours vs PINN {cell['ours_vs_pinn']:+.4f}", flush=True)
        print(f"  {'':16s} chose {picked}", flush=True)

    (a.output / "physics_baselines_summary.json").write_text(json.dumps(
        {"weights": WEIGHTS, "rates": RATES, "seeds": a.seeds,
         "pinn_steps": a.pinn_steps, "records": records}, indent=2))


if __name__ == "__main__":
    main()
