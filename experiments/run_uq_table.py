#!/usr/bin/env python3
"""Predictive uncertainty: does the derived prior also calibrate better?

Most competitors here give a point prediction only.  The Bayesian host gives a
posterior predictive, so this reports what it is actually worth -- coverage of a
nominal 95% interval, Monte-Carlo predictive NLL, and mean interval width --
against the same generic dictionary, at the routing each arm prefers.

Everything is measured after the fixed training budget on held-out entries with
independently drawn noise.  This is a finite-feature, mean-field variational
posterior predictive, not an exact GP posterior, and is labelled as such.
"""

from __future__ import annotations

import argparse, json, math, sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "experiments"))

from geoaware.operator_spectral_funbat import (  # noqa: E402
    ModeAdaptiveVariationalTucker, extended_generic_dictionary, normalize_spectrum_cosine,
)
from forced_pde_solver import solve_forced  # noqa: E402
from run_forced_pde_main import TRUE_SETTING, make_task, nrmse, operator_bank  # noqa: E402

UQ_SAMPLES = 64


def fit_and_score(field, observed, targets, test, truth, spectra, *, ranks, steps,
                  seed, device, lr, noise_std, routing="global"):
    torch.manual_seed(seed + 10_000)
    model = ModeAdaptiveVariationalTucker(
        tuple(torch.arange(s, device=device) / s for s in field.shape),
        spectra.to(device), ranks=ranks, routing=routing, noise_std=0.08,
        basis=("cosine", "cosine", "cosine")).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss, _ = model.negative_elbo(observed, targets, total_count=len(targets), samples=3)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        optimizer.step()

    with torch.no_grad():
        generator = torch.Generator(device=device).manual_seed(seed + 30_000)
        # Score against independently re-noised held-out values, so calibration
        # is measured against observations rather than against the clean field.
        noisy = truth + noise_std * torch.randn(
            truth.shape, generator=generator, device=device)
        latent = model.posterior_predictive_samples(
            test, samples=UQ_SAMPLES, generator=generator, include_noise=False)
        sigma = model.noise_std
        log_prob = -0.5 * (math.log(2 * math.pi) + 2 * torch.log(sigma)
                           + (noisy[None] - latent).square() / sigma.square())
        nll = -(torch.logsumexp(log_prob, dim=0) - math.log(UQ_SAMPLES)).mean()
        predictive = latent + sigma * torch.randn(
            latent.shape, generator=generator, device=device)
        lower = torch.quantile(predictive, 0.025, dim=0)
        upper = torch.quantile(predictive, 0.975, dim=0)
        return {
            "test_nrmse": nrmse(model.posterior_mean(test), truth),
            "coverage_95": float(((noisy >= lower) & (noisy <= upper)).float().mean()),
            "predictive_nll": float(nll),
            "interval_width": float((upper - lower).mean()),
            "learned_noise_std": float(sigma),
        }


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
    parser.add_argument("--uq-points", type=int, default=4096)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "forced_pde")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device); ranks = tuple(args.ranks)

    from run_forced_pde_main import NOMINAL
    ops, _ = operator_bank(args.max_frequency, args.atoms, NOMINAL)
    _, generic = extended_generic_dictionary(args.atoms, args.max_frequency)
    generic = normalize_spectrum_cosine(generic)[None].expand(3, -1, -1).clone()

    summary = {"uq_samples": UQ_SAMPLES,
               "note": "finite-feature mean-field variational posterior predictive, "
                       "not an exact GP posterior; targets are independently re-noised "
                       "held-out values",
               "ratios": {}}
    for ratio in args.ratios:
        rows = {"operator": [], "generic": []}
        for seed in args.seeds:
            solved = solve_forced(seed=seed, **TRUE_SETTING)
            field, observed, targets, test, truth = make_task(
                solved.field, ratio, seed, args.noise_std, device)
            keep = test[: args.uq_points]; keep_truth = truth[: args.uq_points]
            for name, bank in (("operator", ops), ("generic", generic)):
                rows[name].append(fit_and_score(
                    field, observed, targets, keep, keep_truth, bank, ranks=ranks,
                    steps=args.steps, seed=seed, device=device, lr=args.lr,
                    noise_std=args.noise_std))
            print(f"  ratio {ratio:5.3f} seed {seed}  " + "  ".join(
                f"{n}: nrmse={rows[n][-1]['test_nrmse']:.3f} "
                f"cov={rows[n][-1]['coverage_95']:.3f} nll={rows[n][-1]['predictive_nll']:+.3f}"
                for n in rows), flush=True)
        block = {}
        for name, cells in rows.items():
            block[name] = {k: {"mean": float(np.mean([c[k] for c in cells])),
                               "std": float(np.std([c[k] for c in cells]))}
                           for k in cells[0]}
        summary["ratios"][str(ratio)] = block
    (args.output / "uq_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n{'ratio':>7s}{'arm':>10s}{'NRMSE':>9s}{'cov95':>9s}{'NLL':>9s}{'width':>9s}")
    for ratio in args.ratios:
        for name in ("operator", "generic"):
            b = summary["ratios"][str(ratio)][name]
            print(f"{ratio:7.3f}{name:>10s}{b['test_nrmse']['mean']:9.4f}"
                  f"{b['coverage_95']['mean']:9.3f}{b['predictive_nll']['mean']:+9.3f}"
                  f"{b['interval_width']['mean']:9.3f}")


if __name__ == "__main__":
    main()
