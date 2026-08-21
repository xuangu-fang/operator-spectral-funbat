#!/usr/bin/env python3
"""Can a kernel be tuned at all when the sensors are confined?

The main table gives the Matern arm its length scale by minimising held-out
error -- an oracle.  A wide-grid audit showed that oracle is stronger than we
first measured, so the honest question is no longer "do we beat a tuned kernel"
but "can the kernel be tuned by anyone standing at the wall".

It cannot, and the reason is structural rather than statistical.  Tuning needs
validation data that resembles the prediction target.  With sensors on one wall
every point you can hold out is also on that wall, so validation measures
interpolation within the strip while deployment requires extrapolation across
the room.  The two ask for different length scales, and the validation split
cannot see that.

Three tiers, same grid, same everything else:
  deployable   -- length scale chosen on a held-out split of the *observed*
                  sensor readings, which is all a practitioner has;
  oracle       -- length scale chosen on the true held-out region (not
                  deployable, reported as an upper bound);
  ours         -- no tuning data at all, spectra read off the equation with
                  coefficients wrong by 50%.

The quantity of interest is the gap between the deployable and oracle tiers.  A
large gap means the setting punishes tuning; a small gap means our advantage
over the deployable tier is uninteresting because anyone could have tuned their
way to it.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np, torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "experiments"))
from geoaware.operator_spectral_funbat import real_cosine_basis  # noqa: E402
from forced_pde_solver import solve_multi_leak  # noqa: E402
import run_leak_sensors as base  # noqa: E402

GRID = (0.06, 0.12, 0.24, 0.5, 0.8, 1.6, 2.4, 4.0, 8.0, 16.0)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--layouts", nargs="+",
                   default=["one_wall_strip", "corner_block", "wall_ring", "random"])
    p.add_argument("--ratio", type=float, default=0.01)
    p.add_argument("--noise-std", type=float, default=0.05)
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--lr", type=float, default=0.02)
    p.add_argument("--val-fraction", type=float, default=0.25)
    p.add_argument("--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="an idle GPU makes these sweeps roughly an order of magnitude cheaper")
    p.add_argument("--output", type=Path, default=ROOT / "results" / "leak")
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    device = torch.device(a.device)

    shape = tuple(solve_multi_leak(seed=0, **base.FIELD).field.shape)
    budget = int(round(a.ratio * int(np.prod(shape))))
    bases = tuple(real_cosine_basis(torch.arange(s, dtype=torch.float64) / s, b).float()
                  for s, b in zip(shape, base.BINS))
    ours = base.operator_spectra(shape, base.FIELD["dt"], base.BINS)

    records = []
    for layout in a.layouts:
        rows = {"ours_pde": [], "matern_deployable": [], "matern_oracle": []}
        picked = {"deployable": [], "oracle": []}
        for seed in a.seeds:
            field = solve_multi_leak(seed=seed, **base.FIELD).field.to(device)
            observed, test = base.sensor_mask(shape, layout, budget, seed, device)
            g = torch.Generator(device=device).manual_seed(seed + 991)
            targets = field[tuple(observed.T)] + a.noise_std * torch.randn(
                len(observed), generator=g, device=device)
            truth = field[tuple(test.T)]

            # A practitioner's split: hold out part of what the sensors recorded.
            perm = torch.randperm(len(observed), generator=torch.Generator(
                device=device).manual_seed(seed + 4242), device=device)
            cut = int(len(observed) * (1 - a.val_fraction))
            tr_idx, va_idx = perm[:cut], perm[cut:]

            def fit(spectra, obs, tgt, ev, ev_truth):
                return base.fit_gp(field, obs, tgt, ev, ev_truth, spectra, bases,
                                   steps=a.steps, seed=seed, device=device, lr=a.lr)

            val_scores = {ls: fit(base.matern_spectra(base.BINS, ls),
                                  observed[tr_idx], targets[tr_idx],
                                  observed[va_idx], targets[va_idx]) for ls in GRID}
            chosen = min(val_scores, key=val_scores.get)
            picked["deployable"].append(chosen)
            rows["matern_deployable"].append(
                fit(base.matern_spectra(base.BINS, chosen), observed, targets, test, truth))

            test_scores = {ls: fit(base.matern_spectra(base.BINS, ls),
                                   observed, targets, test, truth) for ls in GRID}
            best = min(test_scores, key=test_scores.get)
            picked["oracle"].append(best)
            rows["matern_oracle"].append(test_scores[best])

            rows["ours_pde"].append(fit(ours, observed, targets, test, truth))

        cell = {"layout": layout, "chosen_length_scales": picked,
                "oracle_on_grid_edge": any(b in (GRID[0], GRID[-1]) for b in picked["oracle"])}
        for k, v in rows.items():
            v = np.array(v)
            cell[k] = {"mean": float(v.mean()), "std": float(v.std()), "values": v.tolist()}
        cell["tuning_gap"] = cell["matern_deployable"]["mean"] - cell["matern_oracle"]["mean"]
        cell["ours_vs_deployable"] = cell["matern_deployable"]["mean"] - cell["ours_pde"]["mean"]
        cell["ours_vs_oracle"] = cell["matern_oracle"]["mean"] - cell["ours_pde"]["mean"]
        records.append(cell)
        print(f"  {layout:16s} ours {cell['ours_pde']['mean']:.4f}   "
              f"matern deployable {cell['matern_deployable']['mean']:.4f}   "
              f"matern oracle {cell['matern_oracle']['mean']:.4f}", flush=True)
        print(f"      validation picked {picked['deployable']}, "
              f"the test set wanted {picked['oracle']}"
              f"{'   ORACLE ON GRID EDGE' if cell['oracle_on_grid_edge'] else ''}", flush=True)
        print(f"      cost of having to tune on sensor data: {cell['tuning_gap']:+.4f}   "
              f"ours vs deployable {cell['ours_vs_deployable']:+.4f}   "
              f"ours vs oracle {cell['ours_vs_oracle']:+.4f}", flush=True)

    (a.output / "tunability_summary.json").write_text(json.dumps(
        {"grid": GRID, "seeds": a.seeds, "val_fraction": a.val_fraction,
         "records": records}, indent=2))


if __name__ == "__main__":
    main()
