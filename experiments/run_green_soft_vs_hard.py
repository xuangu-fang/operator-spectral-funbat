#!/usr/bin/env python3
"""Soft spectral weighting versus hard spectral truncation.

The sibling repository's iteration 2 recorded a problem this method is shaped to
solve.  Truncating the operator's eigenbasis at K modes is a bias--variance
knob: raising K from 5 to 12 lowers the oracle projection residual
(0.165 -> 0.070 -> 0.025) yet makes recovery at 2% observations *worse*
(0.293 -> 0.273 -> 0.331), because every extra mode is another factor to
estimate from the same scarce data.  Choosing K well needs exactly the held-out
data you do not have.

A GP prior removes the choice.  Keep every mode, so approximation bias is zero,
and let the operator say how much to trust each one through a decaying
spectrum.  Truncation becomes the special case of a spectrum that is flat up to
K and exactly zero after it.

Arms, all sharing field, mask, noise, host, ranks, optimiser and budget:

  soft, all modes     full eigenbasis, operator-derived decaying spectrum
  hard K = 5/8/12     eigenbasis truncated at K, flat spectrum over it
  soft, truncated     the same decaying spectrum but also truncated, to
                      separate "keep every mode" from "weight them"
"""

from __future__ import annotations

import argparse, json, sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "experiments"))

from geoaware.operator_spectral_funbat import normalize_spectrum_cosine  # noqa: E402
from green_response_data import green_response_tensor, neumann_diffusion_operator  # noqa: E402
from run_green_response_main import make_task, train  # noqa: E402


def bases_and_spectrum(green, modes: int, reaction: float, decay: float,
                       flat: bool, device):
    """Reference-operator bases over `modes` eigenmodes, with a chosen spectrum."""
    grid = green.field.shape[1]
    reference, _ = neumann_diffusion_operator(grid, 0.0)
    values, vectors = torch.linalg.eigh(reference)
    rates = (values / values[1].clamp_min(1e-12))[:modes]
    time_decay = torch.exp(-green.time.double()[:, None] * (reaction + rates[None, :]))
    time_basis = torch.linalg.qr(time_decay, mode="reduced").Q.float()
    spatial = vectors[:, :modes].float()
    bases = tuple(b.to(device) for b in (time_basis, spatial, spatial.clone()))
    if flat:
        weight = torch.ones(modes, dtype=torch.float64)
    else:
        weight = (1 + rates).pow(-2 * decay)
    weight = (weight / weight.sum()).float()
    spectra = normalize_spectrum_cosine(
        weight[None, None, :].expand(3, 1, -1).clone()).to(device)
    return bases, spectra


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--ratios", type=float, nargs="+", default=[0.02, 0.05, 0.10])
    parser.add_argument("--masks", nargs="+", default=["random", "source_fibers"])
    parser.add_argument("--cutoffs", type=int, nargs="+", default=[5, 8, 12])
    parser.add_argument("--contrast", type=float, default=1.0)
    parser.add_argument("--reaction", type=float, default=0.15)
    parser.add_argument("--decay", type=float, default=0.18)
    parser.add_argument("--ranks", type=int, nargs=3, default=[4, 5, 5])
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--lr", type=float, default=0.02)
    parser.add_argument("--noise-std", type=float, default=0.05)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--tag", default="soft_vs_hard")
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "green_response")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device); ranks = tuple(args.ranks)

    green = green_response_tensor(contrast=args.contrast, learner_modes=8)
    # "All modes" is capped by the shortest mode: the time axis can only carry
    # as many orthogonal decay profiles as it has samples.  Truth uses 14
    # modes, so this cap still leaves no truncation bias with respect to the
    # field's actual mode content.
    full = min(green.field.shape)

    arms = {}
    arms["soft_all_modes"] = bases_and_spectrum(
        green, full, args.reaction, args.decay, flat=False, device=device)
    for cutoff in args.cutoffs:
        arms[f"hard_cutoff_{cutoff}"] = bases_and_spectrum(
            green, cutoff, args.reaction, args.decay, flat=True, device=device)
        arms[f"soft_cutoff_{cutoff}"] = bases_and_spectrum(
            green, cutoff, args.reaction, args.decay, flat=False, device=device)

    records = []
    for mask in args.masks:
        for ratio in args.ratios:
            cell = {"mask": mask, "ratio": ratio}
            for name, (bases, spectra) in arms.items():
                values = []
                for seed in args.seeds:
                    field, observed, targets, test, truth = make_task(
                        green.field, ratio, seed, args.noise_std, device, mask=mask)
                    values.append(train(field, observed, targets, test, truth,
                                        spectra, bases, ranks=ranks, steps=args.steps,
                                        seed=seed, device=device, lr=args.lr)["test_nrmse"])
                cell[name] = {"mean": float(np.mean(values)),
                              "std": float(np.std(values)), "values": values}
            reference = np.array(cell["soft_all_modes"]["values"])
            for name in arms:
                cell[name]["wins_against_soft_all"] = int(
                    (np.array(cell[name]["values"]) < reference).sum())
            records.append(cell)
            print(f"  {mask:14s} ratio {ratio:5.3f}  " + "  ".join(
                f"{k}={cell[k]['mean']:.4f}" for k in arms), flush=True)

    (args.output / f"{args.tag}_summary.json").write_text(json.dumps(
        {"metadata": green.metadata, "full_modes": full,
         "config": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
         "records": records}, indent=2))


if __name__ == "__main__":
    main()
