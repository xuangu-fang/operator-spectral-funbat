#!/usr/bin/env python3
"""Does the tuning-cost result hold for operators other than the one we tuned on?

Every number in the main table comes from one field family: anisotropic
reaction-diffusion with three leaks.  A claim about sensor geometry should not
depend on which dissipative operator drives the field, so this repeats the
three-tier comparison across operator families on the two layouts that matter --
the one where tuning works (scattered sensors) and the one where it does not
(a single instrumented wall).

Families, and why each is here:

  reaction-diffusion   the main table's field, for continuity.
  diffusion-dominated  the reaction term reduced tenfold, so decay comes from
                       diffusion rather than from absorption.  Tests whether the
                       result needs the absorbing term.
  advection-diffusion  a mean wind, Peclet about five, so transport dominates.
                       This one is expected to be hard for us: the construction
                       uses an axis-wise even magnitude spectrum, and advection
                       tilts the spectrum off centre.  Our own signed-spectrum
                       audit lists tilted transport as a representational limit,
                       so this is a test of a documented boundary rather than a
                       hopeful extra row.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np, torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "experiments"))
from geoaware.operator_spectral_funbat import (  # noqa: E402
    nonnegative_cp_spectrum, normalize_spectrum_cosine, real_cosine_basis)
from forced_pde_solver import solve_multi_leak  # noqa: E402
import run_leak_sensors as base  # noqa: E402

SOURCES = ((0.30, 0.65, 0.09, 15.0), (0.70, 0.35, 0.07, 25.0), (0.55, 0.75, 0.06, 40.0))

FAMILIES = {
    "reaction-diffusion": dict(
        field=dict(grid=(64, 64), diffusivity=(0.02, 0.006), reaction=0.04,
                   sources=SOURCES, dt=0.6, burn_in=200, record_steps=64,
                   background_noise=0.02),
        nominal=dict(diffusivity=(0.03, 0.012), reaction=0.06, drift=(0.0, 0.0))),
    "diffusion-dominated": dict(
        field=dict(grid=(64, 64), diffusivity=(0.02, 0.006), reaction=0.004,
                   sources=SOURCES, dt=0.6, burn_in=200, record_steps=64,
                   background_noise=0.02),
        nominal=dict(diffusivity=(0.03, 0.012), reaction=0.006, drift=(0.0, 0.0))),
    "advection-diffusion": dict(
        field=dict(grid=(64, 64), diffusivity=(0.02, 0.006), reaction=0.04,
                   sources=SOURCES, dt=0.15, burn_in=800, record_steps=64,
                   background_noise=0.02, drift=(0.09, 0.0)),
        nominal=dict(diffusivity=(0.03, 0.012), reaction=0.06, drift=(0.13, 0.0))),
}


def operator_spectra(shape, dt, bins, nominal, atoms=4):
    """Per-mode spectra from the nominal operator, advection included.

    With a mean wind the symbol is ``i(omega + v.k) + r + D|k|^2``, so the
    response is no longer centred on ``omega = 0``.  The cosine basis carries
    only ``k >= 0`` and cannot represent the sign of ``k``, so we average the two
    branches, which is the honest version of what the construction can express.
    """
    dx, dy = nominal["diffusivity"]
    lam_x = base.neumann_eigenvalues(shape[1])[:bins[1]]
    lam_y = base.neumann_eigenvalues(shape[2])[:bins[2]]
    omega = np.pi * torch.arange(bins[0], dtype=torch.float64) / (shape[0] * dt)
    elliptic = nominal["reaction"] + dx * lam_x[:, None] + dy * lam_y[None, :]
    vx, vy = nominal.get("drift", (0.0, 0.0))
    shift = vx * lam_x.sqrt()[:, None] + vy * lam_y.sqrt()[None, :]
    denominator = elliptic.square()[None]
    plus = 1.0 / (denominator + (omega[:, None, None] + shift[None]).square()).clamp_min(1e-12)
    minus = 1.0 / (denominator + (omega[:, None, None] - shift[None]).square()).clamp_min(1e-12)
    joint = (0.5 * (plus + minus)).float()
    mask = torch.ones_like(joint)
    mask[0, 0, 0] = 0.0
    separated = nonnegative_cp_spectrum(joint, rank=atoms, steps=1200, seed=17, mask=mask)
    return [normalize_spectrum_cosine(f) for f in separated.factors]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--layouts", nargs="+", default=["random", "one_wall_strip"])
    p.add_argument("--families", nargs="+", default=list(FAMILIES))
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

    records = []
    for family in a.families:
        spec = FAMILIES[family]
        shape = tuple(solve_multi_leak(seed=0, **spec["field"]).field.shape)
        budget = int(round(a.ratio * int(np.prod(shape))))
        bases = tuple(real_cosine_basis(torch.arange(s, dtype=torch.float64) / s, b).float()
                      for s, b in zip(shape, base.BINS))
        ours = operator_spectra(shape, spec["field"]["dt"], base.BINS, spec["nominal"])

        for layout in a.layouts:
            rows: dict[str, list] = {}
            for seed in a.seeds:
                field = solve_multi_leak(seed=seed, **spec["field"]).field.to(device)
                observed, test = base.sensor_mask(shape, layout, budget, seed, device)
                g = torch.Generator(device=device).manual_seed(seed + 991)
                targets = field[tuple(observed.T)] + a.noise_std * torch.randn(
                    len(observed), generator=g, device=device)
                truth = field[tuple(test.T)]

                def fit(spectra, obs=observed, tgt=targets, ev=test, ev_truth=truth):
                    return base.fit_gp(field, obs, tgt, ev, ev_truth, spectra, bases,
                                       steps=a.steps, seed=seed, device=device, lr=a.lr)

                rows.setdefault("ours_pde", []).append(fit(ours))

                oracle = {ls: fit(base.matern_spectra(base.BINS, ls))
                          for ls in base.LENGTH_SCALES}
                rows.setdefault("matern_oracle", []).append(min(oracle.values()))

                split = torch.randperm(len(observed), generator=torch.Generator(
                    device=device).manual_seed(seed + 4242), device=device)
                cut = int(len(observed) * 0.75)
                inner, held = split[:cut], split[cut:]
                validation = {ls: fit(base.matern_spectra(base.BINS, ls),
                                      observed[inner], targets[inner],
                                      observed[held], targets[held])
                              for ls in base.LENGTH_SCALES}
                chosen = min(validation, key=validation.get)
                rows.setdefault("matern_deployable", []).append(
                    fit(base.matern_spectra(base.BINS, chosen)))
                rows.setdefault("_chosen", []).append(chosen)

            cell = {"family": family, "layout": layout, "observed": budget,
                    "chosen_length_scales": rows.pop("_chosen")}
            for key, values in rows.items():
                values = np.array(values)
                cell[key] = {"mean": float(values.mean()), "std": float(values.std()),
                             "values": values.tolist()}
            cell["tuning_cost"] = (cell["matern_deployable"]["mean"]
                                   - cell["matern_oracle"]["mean"])
            cell["gap_to_oracle"] = cell["matern_oracle"]["mean"] - cell["ours_pde"]["mean"]
            records.append(cell)
            print(f"  {family:20s} {layout:16s} ours {cell['ours_pde']['mean']:.4f}  "
                  f"deployable {cell['matern_deployable']['mean']:.4f}  "
                  f"oracle {cell['matern_oracle']['mean']:.4f}  "
                  f"tuning cost {cell['tuning_cost']:+.4f}  "
                  f"ours vs oracle {cell['gap_to_oracle']:+.4f}", flush=True)

    (a.output / "operator_families_summary.json").write_text(json.dumps(
        {"families": {k: {kk: str(vv) for kk, vv in v.items()} for k, v in FAMILIES.items()},
         "seeds": a.seeds, "records": records}, indent=2))


if __name__ == "__main__":
    main()
