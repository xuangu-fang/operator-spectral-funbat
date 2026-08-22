#!/usr/bin/env python3
"""Why does the margin dip negative at moderate confinement?

The eight-seed confinement curve is not monotone.  At a strip covering 31% of
the room our arm loses to Mat\'ern by 5.9%, while at 7.8% it wins by 17%.  The
loss is larger with eight seeds than it was with three, so it is not noise and
needs an explanation that can be wrong.

Pre-registered hypothesis: the dip is coefficient error, not structure.  At
moderate confinement only short-range extrapolation is required, and an
oracle-tuned Mat\'ern is nearly optimal for that; our prior is built from nominal
coefficients that are wrong by 50%, and at short range that mis-scaling costs
more than per-axis structure buys.  At extreme confinement the structure matters
more than the scale, so the margin returns.

Prediction: giving our arm the true coefficients turns the width-20 margin
positive.  If the margin stays negative with true coefficients, the hypothesis
is wrong and the dip comes from something else -- record that outcome as it
falls.
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


def strip_mask(shape, width, budget, seed, device):
    nt, nx, ny = shape
    grid = torch.stack(torch.meshgrid(
        *[torch.arange(s, device=device) for s in shape], indexing="ij"), -1).reshape(-1, 3)
    candidates = torch.nonzero(grid[:, 1] < width, as_tuple=False).squeeze(-1)
    g = torch.Generator(device=device).manual_seed(seed + 7717)
    order = torch.randperm(len(candidates), generator=g, device=device)
    keep = torch.zeros(len(grid), dtype=torch.bool, device=device)
    keep[candidates[order[:budget]]] = True
    return grid[keep], grid[~keep]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5, 6, 7])
    p.add_argument("--widths", type=int, nargs="+", default=[20, 5])
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
    bases = tuple(real_cosine_basis(torch.arange(s, dtype=torch.float64) / s, b).float()
                  for s, b in zip(shape, base.BINS))

    banks = {}
    banks["nominal (wrong by 50%)"] = base.operator_spectra(
        shape, base.FIELD["dt"], base.BINS)
    banks["true coefficients"] = base.operator_spectra(
        shape, base.FIELD["dt"], base.BINS,
        nominal={"diffusivity": base.FIELD["diffusivity"],
                 "reaction": base.FIELD["reaction"]})

    records = []
    for width in a.widths:
        rows = {k: [] for k in list(banks) + ["matern"]}
        for seed in a.seeds:
            field = solve_multi_leak(seed=seed, **base.FIELD).field.to(device)
            observed, test = strip_mask(shape, width, budget, seed, device)
            g = torch.Generator(device=device).manual_seed(seed + 991)
            targets = field[tuple(observed.T)] + a.noise_std * torch.randn(
                len(observed), generator=g, device=device)
            truth = field[tuple(test.T)]
            fit = lambda sp: base.fit_gp(field, observed, targets, test, truth, sp, bases,
                                         steps=a.steps, seed=seed, device=device, lr=a.lr)
            for name, bank in banks.items():
                rows[name].append(fit(bank))
            rows["matern"].append(min(fit(base.matern_spectra(base.BINS, ls))
                                      for ls in base.LENGTH_SCALES))
        cell = {"strip_width": width, "fraction_of_room": width / shape[1]}
        for k, v in rows.items():
            v = np.array(v)
            cell[k] = {"mean": float(v.mean()), "std": float(v.std()), "values": v.tolist()}
        for k in banks:
            paired = np.array(rows["matern"]) - np.array(rows[k])
            cell[f"margin[{k}]"] = float(paired.mean())
            cell[f"wins[{k}]"] = int((paired > 0).sum())
        records.append(cell)
        print(f"  width {width:3d} ({100*width/shape[1]:.1f}% of room)  "
              f"matern {cell['matern']['mean']:.4f}", flush=True)
        for k in banks:
            print(f"      {k:24s} {cell[k]['mean']:.4f}   margin "
                  f"{cell[f'margin[{k}]']:+.4f}   wins {cell[f'wins[{k}]']}/{len(a.seeds)}",
                  flush=True)

    dip = next((r for r in records if r["strip_width"] == 20), None)
    if dip is not None:
        verdict = {"hypothesis": "the dip at moderate confinement is coefficient error",
                   "margin_nominal": dip["margin[nominal (wrong by 50%)]"],
                   "margin_true": dip["margin[true coefficients]"]}
        verdict["hypothesis_holds"] = verdict["margin_true"] > 0
        print(f"\n  width 20 margin: nominal {verdict['margin_nominal']:+.4f}, "
              f"true coefficients {verdict['margin_true']:+.4f}  ->  "
              f"hypothesis holds: {verdict['hypothesis_holds']}")
    else:
        verdict = None
    (a.output / "dip_diagnosis_summary.json").write_text(json.dumps(
        {"seeds": a.seeds, "records": records, "verdict": verdict}, indent=2))


if __name__ == "__main__":
    main()
