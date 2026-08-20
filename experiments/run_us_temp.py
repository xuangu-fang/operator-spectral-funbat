#!/usr/bin/env python3
"""Our kernels on FunBaT's US-TEMP benchmark, on its own official folds.

The arms differ only in where the two *spatial* modes' bases and spectra come
from.  The time mode is generic in every arm, because 267 annual means are
trend plus interannual variability rather than diffusive relaxation, and
inventing a temporal operator here would be exactly the over-reach this work
avoids elsewhere.

  ours      Laplace-Beltrami eigenbasis for latitude and longitude,
            with the diffusive weight (1 + lambda)^-1 as the spectrum
  basis     the same eigenbasis with a flat spectrum
  generic   cosine bases and a generic spectral dictionary on all three modes
"""

from __future__ import annotations

import argparse, json, sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "experiments"))

from geoaware.operator_spectral_funbat import (  # noqa: E402
    ModeAdaptiveVariationalTucker, extended_generic_dictionary, normalize_spectrum_cosine,
)
from us_temp_data import load_fold, spatial_operator_basis, spatial_operator_spectrum  # noqa: E402


def cosine_basis(size: int, modes: int) -> torch.Tensor:
    x = torch.arange(size, dtype=torch.float64) / size
    k = torch.arange(modes, dtype=torch.float64)
    basis = np.sqrt(2.0) * torch.cos(np.pi * x[:, None] * k[None])
    basis[:, 0] = 1.0
    return torch.linalg.qr(basis, mode="reduced").Q.float()


def run(fold, bases, spectra, *, ranks, steps, seed, device, lr):
    spectra = ([s.to(device) for s in spectra] if isinstance(spectra, list)
               else spectra.to(device))
    torch.manual_seed(seed + 10_000)
    coordinates = tuple(torch.arange(s, device=device) / s for s in fold.dims)
    model = ModeAdaptiveVariationalTucker(
        coordinates, spectra, ranks=ranks, routing="global", noise_std=0.1,
        basis=("operator", "operator", "operator"),
        eigenbasis=tuple(b.to(device) for b in bases)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    train_index = fold.train_index.to(device)
    train_value = fold.train_value.to(device)
    test_index = fold.test_index.to(device)
    test_value = fold.test_value.to(device)
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss, _ = model.negative_elbo(train_index, train_value,
                                      total_count=len(train_value), samples=3)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        optimizer.step()
    with torch.no_grad():
        prediction = model.posterior_mean(test_index)
        rmse = float(torch.sqrt(torch.mean((prediction - test_value).square())))
        # Report in the raw units the benchmark uses as well as normalised.
        return {"test_rmse_normalised": rmse,
                "test_rmse_raw": rmse * fold.scale,
                "test_nrmse": rmse / float(test_value.std()),
                "parameters": sum(p.numel() for p in model.parameters()
                                  if p.requires_grad)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--modes", type=int, nargs=3, default=[8, 8, 8],
                        help="feature budget per mode; the time mode has 267 points "
                             "and should not be capped by the 15-point latitude mode")
    parser.add_argument("--ranks", type=int, nargs=3, default=[3, 5, 5])
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--lr", type=float, default=0.02)
    parser.add_argument("--decay", type=float, default=1.0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--tag", default="us_temp")
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "us_temp")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device); ranks = tuple(args.ranks); modes = list(args.modes)

    records = []
    for fold_index in args.folds:
        fold = load_fold(fold_index)
        budget = [min(m, s) for m, s in zip(modes, fold.dims)]
        latitude, longitude = spatial_operator_basis(fold, max(budget[0], budget[1]))
        operator_bases = (latitude[:, :budget[0]], longitude[:, :budget[1]],
                          cosine_basis(fold.dims[2], budget[2]))
        cosine_bases_all = tuple(cosine_basis(s, b) for s, b in zip(fold.dims, budget))
        wl, wo = spatial_operator_spectrum(fold, max(budget[0], budget[1]),
                                           decay=args.decay)
        def renorm(v):
            return (v / v.sum()).float()
        generic_per_mode = [normalize_spectrum_cosine(
            extended_generic_dictionary(4, b - 1)[1]) for b in budget]
        operator_spectra = [renorm(wl[:budget[0]])[None, :],
                            renorm(wo[:budget[1]])[None, :],
                            generic_per_mode[2][:1]]
        flat_per_mode = [normalize_spectrum_cosine(torch.ones(1, b)) for b in budget]

        cell = {"fold": fold_index, "train": len(fold.train_value),
                "test": len(fold.test_value), "raw_std": fold.scale}
        for name, (bases, spectra) in {
            "ours": (operator_bases, operator_spectra),
            "operator_basis_flat_spectrum": (operator_bases, flat_per_mode),
            "generic": (cosine_bases_all, generic_per_mode),
        }.items():
            cell[name] = run(fold, bases, spectra, ranks=ranks, steps=args.steps,
                             seed=fold_index, device=device, lr=args.lr)
        records.append(cell)
        print(f"  fold {fold_index}  " + "  ".join(
            f"{k}: rmse_raw={cell[k]['test_rmse_raw']:.4f}"
            for k in ("ours", "operator_basis_flat_spectrum", "generic")), flush=True)

    summary = {"dataset": "FunBaT US-TEMP, [latitude, longitude, year], 15x95x267, "
                          "official five-fold split",
               "note": "time mode uses a generic kernel in every arm; only the two "
                       "spatial modes receive operator-derived bases and spectra",
               "config": {k: (str(v) if isinstance(v, Path) else v)
                          for k, v in vars(args).items()},
               "records": records}
    for name in ("ours", "operator_basis_flat_spectrum", "generic"):
        values = np.array([r[name]["test_rmse_raw"] for r in records])
        summary[name] = {"rmse_raw_mean": float(values.mean()),
                         "rmse_raw_std": float(values.std())}
        print(f"{name:32s} raw RMSE {values.mean():.4f} +- {values.std():.4f}")
    (args.output / f"{args.tag}_summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
