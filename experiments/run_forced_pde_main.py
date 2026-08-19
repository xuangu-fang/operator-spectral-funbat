#!/usr/bin/env python3
"""Main experiment: do PDE-form kernels beat generic kernels on solver data?

The claim is narrow on purpose.  Model, capacity, optimizer, budget, mask, noise
and seeds are all held fixed; the only thing that varies is where the per-mode
GP spectra come from.

Ground truth comes from `forced_pde_solver`, an independent finite-difference
integration of a stochastically forced linear PDE run to statistical steady
state.  The field therefore carries the operator's true, fully non-separable
joint spectrum, while the method only ever sees a rank-Q nonnegative separation
of it built from *nominal* coefficients.  Nothing is sampled from the model's
own prior, which is what made the earlier synthetic results self-fulfilling.
"""

from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "experiments"))

from geoaware.operator_spectral_funbat import (  # noqa: E402
    ModeAdaptiveVariationalTucker, extended_generic_dictionary, nonnegative_cp_spectrum,
    normalize_spectrum_cosine, operator_joint_spectrum,
)
from forced_pde_solver import solve_forced  # noqa: E402

# Physical setting.  Strong anisotropy is deliberate: it is what makes per-mode
# kernels carry information that a single shared kernel cannot.
TRUE_SETTING = dict(
    operator="anisotropic_diffusion", grid=(32, 32),
    diffusivity=(0.02, 0.006), reaction=0.8, forcing_scale=8,
    record_every=40, record_steps=32, burn_in=3000, dt=1.5e-3,
)
# What the method is told: the *form*, with nominal coefficients that are not
# the generating ones.  Coefficient sensitivity is a separate ablation.
NOMINAL = dict(source_scale=0.05, diffusivity=(1.0, 0.3), rate=-1.0, damping=0.3)


def operator_bank(max_frequency: int, atoms: int, nominal: dict) -> tuple[torch.Tensor, float]:
    frequency = torch.arange(max_frequency + 1, dtype=torch.float32)
    joint = operator_joint_spectrum(
        "reaction_diffusion", frequency, source_scale=nominal["source_scale"],
        reaction_diffusivity=nominal["diffusivity"], reaction_rate=nominal["rate"],
        reaction_damping=nominal["damping"],
    ).permute(2, 0, 1)                       # (wx,wy,wt) -> (t,x,y)
    separated = nonnegative_cp_spectrum(joint, rank=atoms, steps=1600, seed=17)
    return normalize_spectrum_cosine(torch.stack(separated.factors)), separated.relative_error


def make_task(field, ratio, seed, noise_std, device):
    field = field.to(device)
    grid = torch.stack(torch.meshgrid(
        *[torch.arange(s, device=device) for s in field.shape], indexing="ij"), -1).reshape(-1, 3)
    generator = torch.Generator(device=device).manual_seed(seed + 7717)
    order = torch.randperm(len(grid), generator=generator, device=device)
    count = round(ratio * len(grid))
    observed, test = grid[order[:count]], grid[order[count:]]
    targets = field[tuple(observed.T)] + noise_std * torch.randn(
        count, generator=generator, device=device)
    return field, observed, targets, test, field[tuple(test.T)]


def nrmse(prediction, truth):
    return float(torch.sqrt(torch.mean((prediction - truth).square()))
                 / truth.std().clamp_min(1e-8))


def train(field, observed, targets, test, truth, spectra, *, ranks, steps, seed, device, lr,
          routing="global"):
    """Default routing is global.

    The ablation showed that per-mode/rank routing over-fits at 1% observations
    for *both* banks, and hurts the generic dictionary roughly twice as much as
    the operator bank (0.042 versus 0.023).  Comparing both arms at
    per-mode/rank therefore flatters the operator bank by handicapping the
    baseline, so each arm is now run at the setting that is best for it, which
    is global for both.
    """
    torch.manual_seed(seed + 10_000)
    model = ModeAdaptiveVariationalTucker(
        tuple(torch.arange(s, device=device) / s for s in field.shape),
        spectra.to(device), ranks=ranks, routing=routing,
        noise_std=0.08, basis=("cosine", "cosine", "cosine"),
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    started = time.time()
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss, _ = model.negative_elbo(observed, targets, total_count=len(targets), samples=3)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        optimizer.step()
    with torch.no_grad():
        return {
            "test_nrmse": nrmse(model.posterior_mean(test), truth),
            "observed_fit_nrmse": nrmse(model.posterior_mean(observed), targets),
            "trainable_parameter_count": sum(
                p.numel() for p in model.parameters() if p.requires_grad),
            "seconds": time.time() - started,
        }


def trivial_baselines(field, observed, targets, test, truth):
    """Global mean and nearest observed neighbour in index space."""
    out = {"global_mean": nrmse(torch.full_like(truth, float(targets.mean())), truth)}
    obs = observed.float(); tst = test.float()
    chunks = []
    for start in range(0, len(tst), 4096):
        block = tst[start:start + 4096]
        distance = torch.cdist(block, obs)
        chunks.append(targets[distance.argmin(1)])
    out["nearest_neighbour"] = nrmse(torch.cat(chunks), truth)
    return out


def tucker_ceiling(field, ranks):
    T = field.double().cpu(); factors = []
    for mode in range(3):
        unfold = T.movedim(mode, 0).reshape(T.shape[mode], -1)
        u, _, _ = torch.linalg.svd(unfold, full_matrices=False)
        factors.append(u[:, :ranks[mode]])
    core = T.clone()
    for mode in range(3):
        core = torch.tensordot(core, factors[mode], dims=([0], [0]))
    rec = core
    for mode in range(3):
        rec = torch.tensordot(rec, factors[mode].T, dims=([0], [0]))
    return float((rec - T).norm() / T.norm())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--ratios", type=float, nargs="+", default=[0.01, 0.02, 0.05])
    parser.add_argument("--ranks", type=int, nargs=3, default=[8, 5, 5])
    parser.add_argument("--atoms", type=int, default=4)
    parser.add_argument("--max-frequency", type=int, default=8)
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--lr", type=float, default=0.02)
    parser.add_argument("--noise-std", type=float, default=0.05)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--routing", default="global",
                        choices=["global", "per_mode", "per_mode_rank"])
    parser.add_argument("--tag", default="main")
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "forced_pde")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    ranks = tuple(args.ranks)

    ops, separation_error = operator_bank(args.max_frequency, args.atoms, NOMINAL)
    _, generic = extended_generic_dictionary(args.atoms, args.max_frequency)
    generic = normalize_spectrum_cosine(generic)[None].expand(3, -1, -1).clone()
    print(f"operator separation relative error {separation_error:.4f}", flush=True)

    records = []
    for seed in args.seeds:
        solved = solve_forced(seed=seed, **TRUE_SETTING)
        ceiling = tucker_ceiling(solved.field, ranks)
        for ratio in args.ratios:
            field, observed, targets, test, truth = make_task(
                solved.field, ratio, seed, args.noise_std, device)
            row = {"seed": seed, "ratio": ratio, "tucker_ceiling": ceiling,
                   "observed": int(len(targets))}
            row.update(trivial_baselines(field, observed, targets, test, truth))
            for name, bank in (("operator", ops), ("generic", generic)):
                row[name] = train(field, observed, targets, test, truth, bank,
                                  ranks=ranks, steps=args.steps, seed=seed,
                                  device=device, lr=args.lr, routing=args.routing)
            row["margin"] = row["generic"]["test_nrmse"] - row["operator"]["test_nrmse"]
            records.append(row)
            print(f"  seed {seed} ratio {ratio:5.3f} n={row['observed']:5d} "
                  f"ceil={ceiling:.3f}  operator {row['operator']['test_nrmse']:.4f} "
                  f"generic {row['generic']['test_nrmse']:.4f} "
                  f"nn {row['nearest_neighbour']:.4f}  margin {row['margin']:+.4f}", flush=True)

    summary = {"true_setting": TRUE_SETTING, "nominal_prior": NOMINAL,
               "config": {k: (str(v) if isinstance(v, Path) else v)
                          for k, v in vars(args).items()},
               "operator_separation_relative_error": separation_error,
               "records": records}
    for ratio in args.ratios:
        cells = [r for r in records if r["ratio"] == ratio]
        wins = sum(1 for r in cells if r["margin"] > 0)
        summary[f"ratio_{ratio}"] = {
            "operator_mean": float(np.mean([r["operator"]["test_nrmse"] for r in cells])),
            "generic_mean": float(np.mean([r["generic"]["test_nrmse"] for r in cells])),
            "nearest_neighbour_mean": float(np.mean([r["nearest_neighbour"] for r in cells])),
            "paired_wins": f"{wins}/{len(cells)}",
        }
        print(f"ratio {ratio}: {summary[f'ratio_{ratio}']}")
    (args.output / f"{args.tag}_summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
