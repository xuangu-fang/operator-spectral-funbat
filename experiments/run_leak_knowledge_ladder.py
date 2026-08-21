#!/usr/bin/env python3
"""How much of the equation do you actually need?

The main table uses a single nominal coefficient vector that is wrong by 50%.
By the taxonomy this project declared in advance that is neither K2 nor K1: it
is K2's machinery applied to a wrong point estimate.  Proper K1 says only that
the coefficients lie in a declared range, samples the range, separates each
sample and pools the atoms, leaving the routing weights to do soft parameter
inference from the data.

That distinction is the one that separates this construction from the two
obvious alternatives.  A PINN penalises the residual of one operator, and an
AutoIP-style GP conditions on one operator; both have to commit to a single
theta before seeing the data.  A pooled bank does not -- it spans the reachable
spectra of the whole family and lets the mixture weights choose.  If K1 lands
near K2, the method needs strictly less knowledge than either alternative.

Arms, all sharing field, mask, noise, host, ranks, optimiser and step budget:

  K2            atoms from the true coefficients.
  K1 point      one nominal guess, wrong by 50%.  This is the main table.
  K1 bank       M samples from theta* x logU[1/3, 3], separated and pooled.
  K0 bank       the same from the wider theta* x logU[1/10, 10].
  K-1 generic   a generic dictionary with the same number of atoms as K1.

The generic arm matches the pooled bank's atom count on purpose: the claim is
that a physically reachable set beats an unconstrained one of equal size, not
that more atoms help.  The collapsed parameterisation keeps the variational
coefficient count independent of bank size, so the comparison is not confounded
by parameters.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np, torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "experiments"))
from geoaware.operator_spectral_funbat import (  # noqa: E402
    extended_generic_dictionary, nonnegative_cp_spectrum, normalize_spectrum_cosine,
    real_cosine_basis)
from forced_pde_solver import solve_multi_leak  # noqa: E402
import run_leak_sensors as base  # noqa: E402

SAMPLES = 8          # M, declared in advance
ATOMS_PER_SAMPLE = 2  # Q1, so a pooled bank holds 16 atoms
ATOMS_K2 = 4


def joint_spectrum(shape, dt, bins, diffusivity, reaction):
    lam_x = base.neumann_eigenvalues(shape[1])[:bins[1]]
    lam_y = base.neumann_eigenvalues(shape[2])[:bins[2]]
    omega = np.pi * torch.arange(bins[0], dtype=torch.float64) / (shape[0] * dt)
    elliptic = reaction + diffusivity[0] * lam_x[:, None] + diffusivity[1] * lam_y[None, :]
    return (1.0 / (omega[:, None, None].square()
                   + elliptic[None].square()).clamp_min(1e-12)).float()


def separate(joint, atoms, seed):
    mask = torch.ones_like(joint)
    mask[0, 0, 0] = 0.0
    return nonnegative_cp_spectrum(joint, rank=atoms, steps=1200, seed=seed, mask=mask)


def pooled_bank(shape, dt, bins, truth, width, *, samples=SAMPLES,
                atoms=ATOMS_PER_SAMPLE, seed=0):
    """Sample the declared range log-uniformly, separate each, pool the atoms."""
    rng = np.random.default_rng(seed)
    banks = [[] for _ in bins]
    for index in range(samples):
        factors = np.exp(rng.uniform(-np.log(width), np.log(width), size=3))
        diffusivity = (truth["diffusivity"][0] * factors[0],
                       truth["diffusivity"][1] * factors[1])
        reaction = truth["reaction"] * factors[2]
        joint = joint_spectrum(shape, dt, bins, diffusivity, reaction)
        for mode, factor in enumerate(separate(joint, atoms, 17 + index).factors):
            banks[mode].append(factor)
    return [normalize_spectrum_cosine(torch.cat(b, 0)) for b in banks]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--layouts", nargs="+", default=["random", "one_wall_strip"])
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
    truth = {"diffusivity": base.FIELD["diffusivity"], "reaction": base.FIELD["reaction"]}

    k2 = [normalize_spectrum_cosine(f) for f in separate(
        joint_spectrum(shape, base.FIELD["dt"], base.BINS,
                       truth["diffusivity"], truth["reaction"]), ATOMS_K2, 17).factors]
    k1_point = base.operator_spectra(shape, base.FIELD["dt"], base.BINS)
    k1_bank = pooled_bank(shape, base.FIELD["dt"], base.BINS, truth, 3.0)
    k0_bank = pooled_bank(shape, base.FIELD["dt"], base.BINS, truth, 10.0)
    pooled_atoms = k1_bank[0].shape[0]
    generic = [normalize_spectrum_cosine(
        extended_generic_dictionary(pooled_atoms, b - 1)[1]) for b in base.BINS]
    print(f"  bank sizes: K2 {k2[0].shape[0]}, K1 point {k1_point[0].shape[0]}, "
          f"K1 bank {pooled_atoms}, K0 bank {k0_bank[0].shape[0]}, "
          f"generic {generic[0].shape[0]}", flush=True)

    ARMS = [("K2 true coefficients", k2), ("K1 point (main table)", k1_point),
            ("K1 bank x[1/3,3]", k1_bank), ("K0 bank x[1/10,10]", k0_bank),
            ("K-1 generic, matched atoms", generic)]

    records = []
    for layout in a.layouts:
        rows: dict[str, list] = {}
        for seed in a.seeds:
            field = solve_multi_leak(seed=seed, **base.FIELD).field.to(device)
            observed, test = base.sensor_mask(shape, layout, budget, seed, device)
            g = torch.Generator(device=device).manual_seed(seed + 991)
            targets = field[tuple(observed.T)] + a.noise_std * torch.randn(
                len(observed), generator=g, device=device)
            truth_values = field[tuple(test.T)]
            for name, spectra in ARMS:
                rows.setdefault(name, []).append(
                    base.fit_gp(field, observed, targets, test, truth_values, spectra,
                                bases, steps=a.steps, seed=seed, device=device, lr=a.lr))
        cell = {"layout": layout, "observed": budget, "pooled_atoms": pooled_atoms}
        for name, values in rows.items():
            values = np.array(values)
            cell[name] = {"mean": float(values.mean()), "std": float(values.std()),
                          "values": values.tolist()}
        cell["k1_bank_minus_k2"] = cell["K1 bank x[1/3,3]"]["mean"] - cell["K2 true coefficients"]["mean"]
        cell["k1_bank_minus_generic"] = (cell["K-1 generic, matched atoms"]["mean"]
                                         - cell["K1 bank x[1/3,3]"]["mean"])
        records.append(cell)
        print(f"  {layout}", flush=True)
        for name, _ in ARMS:
            print(f"    {name:30s} {cell[name]['mean']:.4f}", flush=True)
        print(f"    K1 bank costs {cell['k1_bank_minus_k2']:+.4f} against knowing "
              f"the coefficients; beats a matched generic bank by "
              f"{cell['k1_bank_minus_generic']:+.4f}", flush=True)

    (a.output / "knowledge_ladder_leak_summary.json").write_text(json.dumps(
        {"samples": SAMPLES, "atoms_per_sample": ATOMS_PER_SAMPLE, "seeds": a.seeds,
         "records": records}, indent=2))


if __name__ == "__main__":
    main()
