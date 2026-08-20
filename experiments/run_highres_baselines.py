#!/usr/bin/env python3
"""High-resolution 2-D field, with the baseline tiers separated by what data
their hyper-parameters were chosen on.

Earlier comparisons used a single baseline -- a spectral-mixture dictionary with
learned mixture weights -- and that choice quietly weakened the claim in two
ways.  A spectral mixture is dense in the space of stationary kernels, so it can
represent an operator-derived spectrum given enough data, and any advantage over
it can only ever be sample efficiency rather than expressiveness.  Worse, the
dictionary's atoms were hand-picked by us, and one of them is oscillatory, which
handed the baseline the answer in the band-pass experiment.

FunBaT, the reference method here, does not learn its length scales by ELBO
either: they are fixed in its config and chosen by scanning.  So the honest
default baseline is a single named kernel whose length scale is scanned, and the
question that matters is *which data the scan used*:

  tier 1a  fixed kernel, length scale scanned on a validation split   deployable
  tier 1b  fixed kernel, length scale scanned on the test set         oracle
  tier 2   length scale learned by the same ELBO                      deployable
  tier 3   spectral-mixture dictionary, learned mixture weights       deployable
  ours     spectrum derived from the PDE form                         needs no tuning data

The gap between 1a and 1b is itself the quantity of interest: it is the price of
tuning a kernel, and it is the price our arm does not pay.
"""

from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "experiments"))

from geoaware.operator_spectral_funbat import (  # noqa: E402
    LearnableStationarySpectrum, ModeAdaptiveVariationalTucker,
    extended_generic_dictionary, nonnegative_cp_spectrum, normalize_spectrum_cosine,
    operator_joint_spectrum, real_cosine_basis,
)
from forced_pde_solver import solve_forced_spectral  # noqa: E402

TRUE_SETTING = dict(operator="anisotropic_diffusion", grid=(64, 64),
                    diffusivity=(0.02, 0.006), reaction=0.8, forcing_scale=16,
                    dt=0.1, burn_in=400, record_steps=64)
# The prior is built from the *discrete* operator's own eigenvalues rather than
# from hand-written index-space coefficients.  Writing them by hand silently
# couples the prior to the grid and the timestep: the same numbers that were
# right at 32^3 were four times too steep at 64^3, over-smoothing the prior.
# Nominal physical coefficients are deliberately wrong by a factor, which is the
# "we know the equation, not the medium" case.
NOMINAL = dict(diffusivity=(0.03, 0.01), reaction=1.0, source_length=16.0)
LENGTH_SCALES = (0.05, 0.08, 0.12, 0.2, 0.32, 0.5, 0.8, 1.3, 2.0)


def fixed_spectrum(family: str, bins: int, length_scale: float) -> torch.Tensor:
    module = LearnableStationarySpectrum(family, bins, initial_length_scale=length_scale)
    with torch.no_grad():
        return module().clone()


def neumann_eigenvalues(size: int) -> torch.Tensor:
    """Eigenvalues of the zero-flux second difference on ``size`` points of [0,1]."""
    k = torch.arange(size, dtype=torch.float64)
    return (2 - 2 * torch.cos(np.pi * k / size)) * size ** 2


def operator_bank(bins: tuple[int, int, int], atoms: int, shape, dt: float):
    """Per-mode spectra from the discrete operator, with nominal coefficients.

    ``S = S_w / (omega_t^2 + (r + Dx lam_x + Dy lam_y)^2)``, where ``lam`` are the
    grid's own Neumann eigenvalues and ``omega_t = pi k / (steps * dt)`` is the
    angular frequency the cosine time basis index actually represents.
    """
    dx, dy = NOMINAL["diffusivity"]
    lam_x = neumann_eigenvalues(shape[1])[:bins[1]]
    lam_y = neumann_eigenvalues(shape[2])[:bins[2]]
    window = shape[0] * dt
    omega = np.pi * torch.arange(bins[0], dtype=torch.float64) / window
    elliptic = NOMINAL["reaction"] + dx * lam_x[:, None] + dy * lam_y[None, :]
    response = omega[:, None, None].square() + elliptic[None].square()
    length = NOMINAL["source_length"]
    source = (torch.exp(-0.5 * (lam_x[:, None] + lam_y[None, :]) / length ** 2)[None]
              * torch.ones_like(omega)[:, None, None])
    joint = (source / response.clamp_min(1e-12)).float()
    separated = nonnegative_cp_spectrum(joint, rank=atoms, steps=1600, seed=17)
    return [normalize_spectrum_cosine(f) for f in separated.factors]


def make_task(field, ratio, seed, noise_std, device, validation_fraction=0.0):
    field = field.to(device)
    grid = torch.stack(torch.meshgrid(
        *[torch.arange(s, device=device) for s in field.shape], indexing="ij"), -1).reshape(-1, 3)
    generator = torch.Generator(device=device).manual_seed(seed + 7717)
    order = torch.randperm(len(grid), generator=generator, device=device)
    count = round(ratio * len(grid))
    observed, test = grid[order[:count]], grid[order[count:]]
    targets = field[tuple(observed.T)] + noise_std * torch.randn(
        count, generator=generator, device=device)
    if validation_fraction <= 0:
        return field, observed, targets, test, field[tuple(test.T)], None, None
    # The validation split is carved out of the *observed* entries, so tuning on
    # it costs training data exactly as it would in practice.
    split = int(round((1 - validation_fraction) * count))
    return (field, observed[:split], targets[:split], test, field[tuple(test.T)],
            observed[split:], targets[split:])


def train(field, observed, targets, evaluate_at, truth, spectra, bases, *,
          ranks, steps, seed, device, lr):
    torch.manual_seed(seed + 10_000)
    model = ModeAdaptiveVariationalTucker(
        tuple(torch.arange(s, device=device) / s for s in field.shape),
        [s.to(device) if torch.is_tensor(s) else s for s in spectra],
        ranks=ranks, routing="global", noise_std=0.08,
        basis=("operator", "operator", "operator"),
        eigenbasis=tuple(b.to(device) for b in bases)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    started = time.time()
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss, _ = model.negative_elbo(observed, targets, total_count=len(targets), samples=3)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        optimizer.step()
    with torch.no_grad():
        prediction = model.posterior_mean(evaluate_at)
        error = float(torch.sqrt(torch.mean((prediction - truth).square()))
                      / truth.std().clamp_min(1e-8))
    return error, time.time() - started


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--ratios", type=float, nargs="+", default=[0.005, 0.01, 0.02])
    parser.add_argument("--bins", type=int, nargs=3, default=[16, 12, 12])
    parser.add_argument("--ranks", type=int, nargs=3, default=[12, 6, 6])
    parser.add_argument("--atoms", type=int, default=4)
    parser.add_argument("--family", default="matern32")
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--lr", type=float, default=0.02)
    parser.add_argument("--noise-std", type=float, default=0.05)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--tag", default="highres")
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "highres")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device); ranks = tuple(args.ranks); bins = tuple(args.bins)

    probe = solve_forced_spectral(seed=0, **TRUE_SETTING)
    ours = operator_bank(bins, args.atoms, probe.field.shape, TRUE_SETTING["dt"])
    dictionary = [normalize_spectrum_cosine(extended_generic_dictionary(4, b - 1)[1])
                  for b in bins]

    records = []
    for ratio in args.ratios:
        rows: dict[str, list[float]] = {}
        chosen = {"validation": [], "oracle": []}
        for seed in args.seeds:
            solved = solve_forced_spectral(seed=seed, **TRUE_SETTING)
            bases = tuple(real_cosine_basis(
                torch.arange(s, dtype=torch.float64) / s, b).float()
                for s, b in zip(solved.field.shape, bins))
            field, obs, tgt, test, truth, val_i, val_y = make_task(
                solved.field, ratio, seed, args.noise_std, device,
                validation_fraction=args.validation_fraction)
            full_field, full_obs, full_tgt, _, _, _, _ = make_task(
                solved.field, ratio, seed, args.noise_std, device)

            def fit(spectra, observed, targets, at, target_values):
                return train(field, observed, targets, at, target_values, spectra, bases,
                             ranks=ranks, steps=args.steps, seed=seed, device=device,
                             lr=args.lr)[0]

            # tier 1a: scan on validation, then refit on all observed entries
            scores = {ls: fit([fixed_spectrum(args.family, b, ls) for b in bins],
                              obs, tgt, val_i, val_y) for ls in LENGTH_SCALES}
            best_validation = min(scores, key=scores.get)
            chosen["validation"].append(best_validation)
            rows.setdefault("tier1a_scan_on_validation", []).append(
                fit([fixed_spectrum(args.family, b, best_validation) for b in bins],
                    full_obs, full_tgt, test, truth))
            # tier 1b: scan directly on the test set -- an oracle, not deployable
            oracle = {ls: fit([fixed_spectrum(args.family, b, ls) for b in bins],
                              full_obs, full_tgt, test, truth) for ls in LENGTH_SCALES}
            best_oracle = min(oracle, key=oracle.get)
            chosen["oracle"].append(best_oracle)
            rows.setdefault("tier1b_scan_on_test_ORACLE", []).append(oracle[best_oracle])
            # tier 2: length scale learned by the ELBO
            rows.setdefault("tier2_elbo_learned", []).append(
                fit([LearnableStationarySpectrum(args.family, b) for b in bins],
                    full_obs, full_tgt, test, truth))
            # tier 3: spectral-mixture dictionary
            rows.setdefault("tier3_spectral_mixture", []).append(
                fit(dictionary, full_obs, full_tgt, test, truth))
            # ours
            rows.setdefault("ours_pde_form", []).append(
                fit(ours, full_obs, full_tgt, test, truth))

        cell = {"ratio": ratio, "observed": int(round(ratio * 262144)),
                "chosen_length_scales": chosen}
        for name, values in rows.items():
            values = np.array(values)
            cell[name] = {"mean": float(values.mean()), "std": float(values.std()),
                          "values": values.tolist()}
        reference = np.array(rows["ours_pde_form"])
        for name in rows:
            cell[name]["wins_against_ours"] = int((np.array(rows[name]) < reference).sum())
        records.append(cell)
        print(f"  ratio {ratio:6.4f} (n={cell['observed']})  " + "  ".join(
            f"{k.split('_')[0]}={cell[k]['mean']:.4f}" for k in rows), flush=True)

    (args.output / f"{args.tag}_summary.json").write_text(json.dumps(
        {"true_setting": TRUE_SETTING, "nominal_prior": NOMINAL,
         "length_scale_grid": list(LENGTH_SCALES),
         "config": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
         "records": records}, indent=2))


if __name__ == "__main__":
    main()
