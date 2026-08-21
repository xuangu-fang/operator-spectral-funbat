#!/usr/bin/env python3
"""Mechanism test: does the margin follow the axis, as anisotropy predicts?

The claim behind the main table is specific.  A per-mode spectrum is a statement
about how the field decays along one axis, and a strip of sensors on one wall
forces extrapolation along one axis, so the two line up.  A weaker reading --
"physics is generically useful" -- makes no prediction about *which* wall.

The field is anisotropic: Dx = 0.02 against Dy = 0.006, so x is the quickly
diffusing, smoother axis and y is the slowly diffusing, rougher one.  A generic
isotropic kernel must pick a single length scale, so it is wrong by more along y
than along x.  If our gain comes from knowing the per-axis decay, the margin at
the y wall should exceed the margin at the x wall.  If the two are equal, the
mechanism claim is wrong and only the weaker reading survives.

To make the prediction falsifiable rather than post hoc, the isotropic control
is run in the same script: with Dx = Dy the two walls must agree.
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

WALLS = ("one_wall_strip", "one_wall_strip_y")


def run_condition(name, field_kwargs, nominal, seeds, a, device):
    saved = dict(base.NOMINAL)
    base.NOMINAL.update(nominal)
    shape = tuple(solve_multi_leak(seed=0, **field_kwargs).field.shape)
    budget = int(round(a.ratio * int(np.prod(shape))))
    ours = base.operator_spectra(shape, field_kwargs["dt"], base.BINS)
    bases = tuple(real_cosine_basis(torch.arange(s, dtype=torch.float64) / s, b).float()
                  for s, b in zip(shape, base.BINS))
    out = []
    for layout in WALLS:
        rows = {"ours_pde": [], "matern": []}
        for seed in seeds:
            field = solve_multi_leak(seed=seed, **field_kwargs).field.to(device)
            observed, test = base.sensor_mask(shape, layout, budget, seed, device)
            g = torch.Generator(device=device).manual_seed(seed + 991)
            targets = field[tuple(observed.T)] + a.noise_std * torch.randn(
                len(observed), generator=g, device=device)
            truth = field[tuple(test.T)]
            fit = lambda sp: base.fit_gp(field, observed, targets, test, truth, sp, bases,
                                         steps=a.steps, seed=seed, device=device, lr=a.lr)
            rows["ours_pde"].append(fit(ours))
            rows["matern"].append(min(fit(base.matern_spectra(base.BINS, ls))
                                      for ls in base.LENGTH_SCALES))
        cell = {"condition": name, "layout": layout}
        for k, v in rows.items():
            v = np.array(v)
            cell[k] = {"mean": float(v.mean()), "std": float(v.std()), "values": v.tolist()}
        # Paired, because both arms see the same field, mask and noise per seed.
        paired = np.array(rows["matern"]) - np.array(rows["ours_pde"])
        cell["paired_margin"] = float(paired.mean())
        cell["paired_wins"] = int((paired > 0).sum())
        cell["relative_percent"] = 100 * cell["paired_margin"] / cell["matern"]["mean"]
        out.append(cell)
        print(f"  [{name:11s}] {layout:19s} ours {cell['ours_pde']['mean']:.4f}  "
              f"matern {cell['matern']['mean']:.4f}  margin {cell['paired_margin']:+.4f} "
              f"({cell['relative_percent']:+.1f}%)  wins {cell['paired_wins']}/{len(seeds)}",
              flush=True)
    base.NOMINAL.clear(); base.NOMINAL.update(saved)
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5, 6, 7])
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

    anis = dict(base.FIELD)
    iso = dict(base.FIELD); iso["diffusivity"] = (0.013, 0.013)
    records = []
    records += run_condition("anisotropic", anis, dict(base.NOMINAL), a.seeds, a, device)
    records += run_condition("isotropic", iso, {"diffusivity": (0.02, 0.02),
                                                "reaction": 0.06}, a.seeds, a, device)

    by = {(r["condition"], r["layout"]): r for r in records}
    verdict = {
        "anisotropic_gap": (by[("anisotropic", "one_wall_strip_y")]["relative_percent"]
                            - by[("anisotropic", "one_wall_strip")]["relative_percent"]),
        "isotropic_gap": (by[("isotropic", "one_wall_strip_y")]["relative_percent"]
                          - by[("isotropic", "one_wall_strip")]["relative_percent"]),
    }
    verdict["prediction_holds"] = (verdict["anisotropic_gap"] > 0
                                   and abs(verdict["isotropic_gap"])
                                   < abs(verdict["anisotropic_gap"]))
    print("\n  y-wall minus x-wall margin:  anisotropic "
          f"{verdict['anisotropic_gap']:+.1f} pts,  isotropic control "
          f"{verdict['isotropic_gap']:+.1f} pts")
    print(f"  prediction holds: {verdict['prediction_holds']}")
    (a.output / "anisotropy_axis_summary.json").write_text(json.dumps(
        {"seeds": a.seeds, "records": records, "verdict": verdict}, indent=2))


if __name__ == "__main__":
    main()
