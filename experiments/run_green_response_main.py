#!/usr/bin/env python3
"""Our method on the Green-response tensor.

Here the operator supplies *both* the basis and the spectrum for all three
modes, so the arms are designed to separate those two contributions:

  ours              operator eigenbasis + operator spectrum
  operator basis    operator eigenbasis + flat spectrum   -> what the spectrum adds
  generic           cosine basis + generic dictionary     -> no physics at all
  wrong medium      eigenbasis of a *different* a(x)      -> is it the right operator?

Masks include whole missing source fibres, because "we did not fire that
source" is what sparsity actually looks like in a Green-response measurement,
whereas a uniform random mask is the statistically easiest case.
"""

from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "experiments"))

from geoaware.operator_spectral_funbat import (  # noqa: E402
    ModeAdaptiveVariationalTucker, extended_generic_dictionary, normalize_spectrum_cosine,
)
from green_response_data import (  # noqa: E402
    green_response_tensor, learner_bases, neumann_diffusion_operator, operator_spectra,
)


def nrmse(prediction, truth):
    return float(torch.sqrt(torch.mean((prediction - truth).square()))
                 / truth.std().clamp_min(1e-8))


def make_task(field, ratio, seed, noise_std, device, mask="random"):
    """Random entries, or whole missing source fibres."""
    field = field.to(device)
    nt, nr, ns = field.shape
    grid = torch.stack(torch.meshgrid(
        *[torch.arange(s, device=device) for s in field.shape], indexing="ij"), -1).reshape(-1, 3)
    generator = torch.Generator(device=device).manual_seed(seed + 7717)
    if mask == "random":
        order = torch.randperm(len(grid), generator=generator, device=device)
        count = round(ratio * len(grid))
        observed, test = grid[order[:count]], grid[order[count:]]
    elif mask == "source_fibers":
        # Keep a subset of sources; within them keep all times and receivers.
        keep = max(1, round(ratio * ns))
        chosen = torch.randperm(ns, generator=generator, device=device)[:keep]
        is_kept = torch.zeros(ns, dtype=torch.bool, device=device)
        is_kept[chosen] = True
        selector = is_kept[grid[:, 2]]
        observed, test = grid[selector], grid[~selector]
    else:
        raise ValueError(f"unknown mask {mask}")
    targets = field[tuple(observed.T)] + noise_std * torch.randn(
        len(observed), generator=generator, device=device)
    return field, observed, targets, test, field[tuple(test.T)]


def train(field, observed, targets, test, truth, spectra, bases, *, ranks, steps,
          seed, device, lr, routing="global"):
    torch.manual_seed(seed + 10_000)
    model = ModeAdaptiveVariationalTucker(
        tuple(torch.arange(s, device=device) / s for s in field.shape),
        spectra.to(device), ranks=ranks, routing=routing, noise_std=0.08,
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
        return {"test_nrmse": nrmse(model.posterior_mean(test), truth),
                "seconds": time.time() - started,
                "parameters": sum(p.numel() for p in model.parameters() if p.requires_grad)}


def cosine_bases(shape, modes, device):
    """Basis with no operator knowledge: plain cosines on the index grid."""
    out = []
    for size in shape:
        x = torch.arange(size, dtype=torch.float64) / size
        k = torch.arange(modes, dtype=torch.float64)
        basis = torch.cos(np.pi * x[:, None] * k[None]) * np.sqrt(2.0)
        basis[:, 0] = 1.0
        out.append(torch.linalg.qr(basis, mode="reduced").Q.float().to(device))
    return tuple(out)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--ratios", type=float, nargs="+", default=[0.02, 0.05, 0.10])
    parser.add_argument("--masks", nargs="+", default=["random", "source_fibers"])
    parser.add_argument("--contrast", type=float, default=1.0)
    parser.add_argument("--learner-modes", type=int, default=8)
    parser.add_argument("--ranks", type=int, nargs=3, default=[4, 5, 5])
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--lr", type=float, default=0.02)
    parser.add_argument("--noise-std", type=float, default=0.05)
    parser.add_argument("--wrong-contrast", type=float, default=2.2)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--tag", default="green")
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "green_response")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device); ranks = tuple(args.ranks)
    modes = args.learner_modes

    green = green_response_tensor(contrast=args.contrast, learner_modes=modes)
    bases = tuple(b.to(device) for b in learner_bases(green))
    spectra = operator_spectra(green).to(device)
    flat = normalize_spectrum_cosine(torch.ones(3, 1, modes)).to(device)
    _, generic_raw = extended_generic_dictionary(4, modes - 1)
    generic = normalize_spectrum_cosine(generic_raw)[None].expand(3, -1, -1).clone().to(device)
    cosine = cosine_bases(green.field.shape, modes, device)

    # Wrong medium: eigenbasis of a different a(x), same equation form.
    wrong_op, _ = neumann_diffusion_operator(green.field.shape[1], args.wrong_contrast)
    wrong_values, wrong_vectors = torch.linalg.eigh(wrong_op)
    wrong_rates = (wrong_values / wrong_values[1].clamp_min(1e-12))[:modes]
    wrong_decay = torch.exp(-green.time.double()[:, None] * (0.15 + wrong_rates[None, :]))
    wrong_bases = (torch.linalg.qr(wrong_decay, mode="reduced").Q.float().to(device),
                   wrong_vectors[:, :modes].float().to(device),
                   wrong_vectors[:, :modes].float().to(device).clone())

    arms = {
        "ours_operator_basis_and_spectrum": (spectra[:, None, :], bases),
        "operator_basis_flat_spectrum": (flat, bases),
        "generic_basis_and_dictionary": (generic, cosine),
        "wrong_medium_operator": (spectra[:, None, :], wrong_bases),
    }

    records = []
    for mask in args.masks:
        for ratio in args.ratios:
            rows = {k: [] for k in arms}
            for seed in args.seeds:
                field, observed, targets, test, truth = make_task(
                    green.field, ratio, seed, args.noise_std, device, mask=mask)
                for name, (bank, basis) in arms.items():
                    rows[name].append(train(field, observed, targets, test, truth,
                                            bank, basis, ranks=ranks, steps=args.steps,
                                            seed=seed, device=device, lr=args.lr))
            cell = {"mask": mask, "ratio": ratio,
                    "observed": int(round(ratio * green.field.numel()))
                    if mask == "random" else None}
            for name, out in rows.items():
                values = np.array([o["test_nrmse"] for o in out])
                cell[name] = {"mean": float(values.mean()), "std": float(values.std()),
                              "values": values.tolist()}
            base = np.array(rows["ours_operator_basis_and_spectrum"])
            ours = np.array([o["test_nrmse"] for o in rows["ours_operator_basis_and_spectrum"]])
            for name in arms:
                other = np.array([o["test_nrmse"] for o in rows[name]])
                cell[name]["wins_against_ours"] = int((other < ours).sum())
            records.append(cell)
            print(f"  {mask:14s} ratio {ratio:5.3f}  " + "  ".join(
                f"{k.split('_')[0]}={cell[k]['mean']:.4f}" for k in arms), flush=True)

    summary = {"metadata": green.metadata,
               "oracle_projection_residual_note":
                   "the learner's reference basis cannot represent the true field exactly; "
                   "that irreducible floor is reported in the data module",
               "config": {k: (str(v) if isinstance(v, Path) else v)
                          for k, v in vars(args).items()},
               "records": records}
    (args.output / f"{args.tag}_summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
