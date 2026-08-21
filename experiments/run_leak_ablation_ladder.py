#!/usr/bin/env python3
"""Separating what is a published baseline from what is our own ablation.

The generic-kernel arm reported elsewhere in this work is not an off-the-shelf
method.  It runs inside *our* host: the boundary-matched cosine eigenbasis, the
Tucker rather than CP factorisation, the collapsed mixture parameterisation and
the per-mode feature budgets.  Every one of those is a design choice this paper
argues for, and all of them are handed to the generic arm for free.  Calling it
a baseline overstates the competition and understates the method; it is an
ablation, and only isolates the contribution of the spectra.

This script walks down the stack one choice at a time, so the two questions can
be answered separately: how much does the whole construction beat what is
published, and how much of that is the spectra specifically?

  A  ours                     operator spectra, cosine eigenbasis, Tucker host
  B  generic spectra          fixed length scale, cosine eigenbasis, Tucker
  C  generic + periodic       fixed length scale, periodic Fourier basis, Tucker
  D  operator + periodic      operator spectra, periodic Fourier basis, Tucker
  E  generic + periodic + CP  closest arrangement to a published functional
                              tensor model: generic kernels, a periodic basis and
                              a CP rather than Tucker factorisation

A to B is the ablation; A to E is the comparison against what exists.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np, torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "experiments"))
from geoaware.operator_spectral_funbat import (  # noqa: E402
    ModeAdaptiveVariationalCP, ModeAdaptiveVariationalTucker, normalize_spectrum,
    real_cosine_basis)
from forced_pde_solver import solve_multi_leak  # noqa: E402
import run_leak_sensors as base  # noqa: E402

FIXED = 1.6          # the best single constant found in the fixed-kernel sweep
CP_RANK = 12


def fit(shape, observed, targets, test, truth, spectra, *, basis, bases, host,
        steps, seed, device, lr):
    torch.manual_seed(seed + 10_000)
    coordinates = tuple(torch.arange(s, device=device) / s for s in shape)
    if host == "tucker":
        model = ModeAdaptiveVariationalTucker(
            coordinates, [s.to(device) for s in spectra], ranks=base.RANKS,
            routing="global", noise_std=0.08, basis=(basis,) * 3,
            eigenbasis=tuple(b.to(device) for b in bases) if basis == "operator" else None)
    else:
        stacked = torch.stack([s.to(device) for s in spectra])
        model = ModeAdaptiveVariationalCP(
            coordinates, stacked, rank=CP_RANK, routing="global", noise_std=0.08,
            mixture_parameterization="collapsed")
    model = model.to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(steps):
        optimiser.zero_grad(set_to_none=True)
        loss, _ = model.negative_elbo(observed, targets, total_count=len(targets), samples=3)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        optimiser.step()
    with torch.no_grad():
        prediction = model.posterior_mean(test)
        return float(torch.sqrt(torch.mean((prediction - truth).square()))
                     / truth.std().clamp_min(1e-8))


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
    operator = base.operator_spectra(shape, base.FIELD["dt"], base.BINS)
    generic_cos = base.matern_spectra(base.BINS, FIXED)
    # The periodic arms need spectra normalised for a two-sided Fourier support.
    def generic_fourier():
        out = []
        for b in base.BINS:
            k = torch.arange(b, dtype=torch.float32)
            s = (1 + (FIXED * k).square()).pow(-2.0)
            out.append(normalize_spectrum((s / s.sum())[None]))
        return out
    def operator_fourier():
        return [normalize_spectrum(s) for s in operator]

    ARMS = [
        ("A ours (operator, cosine, Tucker)", lambda: operator, "operator", "tucker"),
        ("B generic spectra (cosine, Tucker)", lambda: generic_cos, "operator", "tucker"),
        ("C generic + periodic basis", generic_fourier, "fourier", "tucker"),
        ("D operator + periodic basis", operator_fourier, "fourier", "tucker"),
        ("E generic + periodic + CP host", generic_fourier, "fourier", "cp"),
    ]

    records = []
    for layout in a.layouts:
        rows: dict[str, list] = {}
        for seed in a.seeds:
            field = solve_multi_leak(seed=seed, **base.FIELD).field.to(device)
            observed, test = base.sensor_mask(shape, layout, budget, seed, device)
            g = torch.Generator(device=device).manual_seed(seed + 991)
            targets = field[tuple(observed.T)] + a.noise_std * torch.randn(
                len(observed), generator=g, device=device)
            truth = field[tuple(test.T)]
            for name, spectra, basis, host in ARMS:
                try:
                    score = fit(shape, observed, targets, test, truth, spectra(),
                                basis=basis, bases=bases, host=host, steps=a.steps,
                                seed=seed, device=device, lr=a.lr)
                except Exception as error:                      # noqa: BLE001
                    print(f"    arm {name} failed: {error}", flush=True)
                    score = float("nan")
                rows.setdefault(name, []).append(score)
        cell = {"layout": layout, "observed": budget}
        for name, values in rows.items():
            values = np.array(values)
            cell[name] = {"mean": float(np.nanmean(values)),
                          "std": float(np.nanstd(values)), "values": values.tolist()}
        records.append(cell)
        print(f"  {layout}", flush=True)
        for name, _, _, _ in ARMS:
            print(f"    {name:38s} {cell[name]['mean']:.4f}", flush=True)
    (a.output / "ablation_ladder_summary.json").write_text(json.dumps(
        {"fixed_length_scale": FIXED, "cp_rank": CP_RANK, "seeds": a.seeds,
         "records": records}, indent=2))


if __name__ == "__main__":
    main()
