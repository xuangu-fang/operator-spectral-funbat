#!/usr/bin/env python3
"""Round five: can you find the leak, not just lower the error?

Reconstruction error is the metric the tensor literature uses, but it is not the
question a gas-leak operator asks.  They ask where the leak is.  This measures
that directly: reconstruct the field from wall sensors, take the time-averaged
field, and report how far its peak is from the true source.

The metric is in room-widths, so it is interpretable without reference to any
model.  A method that reduces NRMSE but still puts the leak against the wrong
wall has not solved the user's problem; one that localises to a few percent of
the room has, even if its NRMSE is unremarkable.

Only the strongest leak is scored, since with several sources at once the
correspondence between predicted and true peaks is ambiguous and scoring it
would require a matching rule that could be tuned.
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
    real_cosine_basis,
)
from forced_pde_solver import solve_multi_leak  # noqa: E402
from neural_functional_tucker import NeuralFunctionalTucker  # noqa: E402
from run_leak_sensors import (  # noqa: E402
    BINS, FIELD, LENGTH_SCALES, RANKS, matern_spectra, operator_spectra, sensor_mask,
)

SINGLE_SOURCE = dict(FIELD, sources=((0.30, 0.65, 0.09, 15.0),))


def peak_location(field: torch.Tensor) -> tuple[float, float]:
    """Time-averaged field's argmax, in [0,1]^2."""
    mean_map = field.mean(0)
    flat = int(mean_map.argmax())
    nx, ny = mean_map.shape
    return ((flat // ny + 0.5) / nx, (flat % ny + 0.5) / ny)


def reconstruct_gp(field, observed, targets, spectra, bases, *, steps, seed, device, lr):
    torch.manual_seed(seed + 10_000)
    model = ModeAdaptiveVariationalTucker(
        tuple(torch.arange(s, device=device) / s for s in field.shape),
        [s.to(device) for s in spectra], ranks=RANKS, routing="global", noise_std=0.08,
        basis=("operator", "operator", "operator"),
        eigenbasis=tuple(b.to(device) for b in bases)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss, _ = model.negative_elbo(observed, targets, total_count=len(targets), samples=3)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        optimizer.step()
    grid = torch.stack(torch.meshgrid(
        *[torch.arange(s, device=device) for s in field.shape], indexing="ij"),
        -1).reshape(-1, 3)
    with torch.no_grad():
        return model.posterior_mean(grid).reshape(field.shape)


def reconstruct_neural(shape, observed, targets, *, steps, seed, device, lr=1e-3):
    model = NeuralFunctionalTucker(RANKS, seed=seed).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    sizes = torch.tensor(shape, device=device, dtype=torch.float32)
    columns = lambda idx: [(idx.float() / sizes)[:, m:m + 1] for m in range(3)]
    train_columns = columns(observed)
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss = (model(train_columns) - targets).square().mean()
        loss.backward()
        optimizer.step()
    grid = torch.stack(torch.meshgrid(
        *[torch.arange(s, device=device) for s in shape], indexing="ij"),
        -1).reshape(-1, 3)
    with torch.no_grad():
        return model(columns(grid)).reshape(shape)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--layouts", nargs="+",
                        default=["one_wall_strip", "wall_ring", "corner_block"])
    parser.add_argument("--ratio", type=float, default=0.01)
    parser.add_argument("--noise-std", type=float, default=0.05)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--neural-steps", type=int, default=1500)
    parser.add_argument("--lr", type=float, default=0.02)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--tag", default="localization")
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "leak")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    probe = solve_multi_leak(seed=0, **SINGLE_SOURCE)
    shape = tuple(probe.field.shape)
    truth_xy = SINGLE_SOURCE["sources"][0][:2]
    budget = int(round(args.ratio * int(np.prod(shape))))
    ours = operator_spectra(shape, FIELD["dt"], BINS)
    dictionary = [normalize_spectrum_cosine(extended_generic_dictionary(4, b - 1)[1])
                  for b in BINS]
    bases = tuple(real_cosine_basis(torch.arange(s, dtype=torch.float64) / s, b).float()
                  for s, b in zip(shape, BINS))

    records = []
    for layout in args.layouts:
        rows: dict[str, list[float]] = {}
        for seed in args.seeds:
            solved = solve_multi_leak(seed=seed, **SINGLE_SOURCE)
            field = solved.field.to(device)
            observed, _ = sensor_mask(shape, layout, budget, seed, device)
            generator = torch.Generator(device=device).manual_seed(seed + 991)
            targets = field[tuple(observed.T)] + args.noise_std * torch.randn(
                len(observed), generator=generator, device=device)

            def distance(reconstruction):
                px, py = peak_location(reconstruction.cpu())
                return float(np.hypot(px - truth_xy[0], py - truth_xy[1]))

            rows.setdefault("ours_pde", []).append(distance(reconstruct_gp(
                field, observed, targets, ours, bases, steps=args.steps,
                seed=seed, device=device, lr=args.lr)))
            best = min(((distance(reconstruct_gp(
                field, observed, targets, matern_spectra(BINS, ls), bases,
                steps=args.steps, seed=seed, device=device, lr=args.lr)), ls)
                for ls in LENGTH_SCALES), key=lambda pair: pair[0])
            rows.setdefault("matern", []).append(best[0])
            rows.setdefault("spectral_mixture", []).append(distance(reconstruct_gp(
                field, observed, targets, dictionary, bases, steps=args.steps,
                seed=seed, device=device, lr=args.lr)))
            rows.setdefault("neural_tucker", []).append(distance(reconstruct_neural(
                shape, observed, targets, steps=args.neural_steps,
                seed=seed, device=device)))
        cell = {"layout": layout, "true_source": truth_xy, "observed": budget}
        for name, values in rows.items():
            values = np.array(values)
            cell[name] = {"mean_distance": float(values.mean()),
                          "std": float(values.std()), "values": values.tolist()}
        records.append(cell)
        print(f"  {layout:16s} localisation error in room widths:  " + "  ".join(
            f"{k.split('_')[0]}={cell[k]['mean_distance']:.3f}" for k in rows), flush=True)

    (args.output / f"{args.tag}_summary.json").write_text(json.dumps(
        {"note": "distance from the reconstructed time-averaged peak to the true "
                 "source, in room widths; single source so the correspondence is "
                 "unambiguous",
         "field": SINGLE_SOURCE,
         "config": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
         "records": records}, indent=2))


if __name__ == "__main__":
    main()
