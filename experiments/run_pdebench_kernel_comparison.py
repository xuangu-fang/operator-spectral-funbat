#!/usr/bin/env python3
"""G1/G2 probe: do PDE-form kernels beat generic kernels on real PDEBench data?

The claim under test is deliberately narrow: with the model, capacity,
optimizer and budget held fixed, does deriving the per-mode spectra from the
governing equation's *form* beat a generic spectral dictionary?

Design note that must survive into the paper.  Real 2D Turing patterns are not
low CP-rank -- they are isotropic blobs, not x-tensor-y separable -- so at full
resolution no kernel can help because the *model* cannot represent the field at
all.  We therefore subsample to a resolution where a fully observed CP of the
chosen rank represents the field to within a pre-checked tolerance, which
isolates the kernel comparison from model misspecification.  That tolerance is
reported, not hidden.
"""

from __future__ import annotations

import argparse, json, sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "experiments"))

from geoaware.operator_spectral_funbat import (  # noqa: E402
    ModeAdaptiveVariationalCP, extended_generic_dictionary, nonnegative_cp_spectrum,
    normalize_spectrum, operator_joint_spectrum,
)
from pdebench_data import load_field, make_task, nrmse  # noqa: E402

# PDEBench 2D_diff-react is FitzHugh-Nagumo: d_t u = Du grad^2 u + Ru(u,v).
# Only the FORM is used here.  Coefficients are nominal; the coefficient
# sensitivity study is a separate ablation.
NOMINAL = {"diffusivity": (1.0, 1.0), "reaction_rate": 6.0, "damping": 0.4,
           "source_scale": 0.06}


def operator_atoms(max_frequency: int, atoms: int, nominal: dict) -> tuple[torch.Tensor, float]:
    """Per-mode spectra for a [t, x, y] tensor from the reaction-diffusion form."""
    frequency = torch.arange(max_frequency + 1, dtype=torch.float32)
    joint = operator_joint_spectrum(
        "reaction_diffusion", frequency,
        source_scale=nominal["source_scale"],
        reaction_diffusivity=nominal["diffusivity"],
        reaction_rate=nominal["reaction_rate"],
        reaction_damping=nominal["damping"],
    )                                   # axes (wx, wy, wt)
    joint = joint.permute(2, 0, 1)      # -> (wt, wx, wy), matching the tensor
    separated = nonnegative_cp_spectrum(joint, rank=atoms, steps=1600, seed=17)
    return normalize_spectrum(torch.stack(separated.factors)), separated.relative_error


def train(task, spectra, *, rank, steps, seed, device, floor=None, lr=0.02):
    torch.manual_seed(seed + 10_000)
    model = ModeAdaptiveVariationalCP(
        tuple(torch.arange(s, device=device) / s for s in task.field.shape),
        spectra.to(device), rank=rank, routing="per_mode_rank",
        noise_std=0.08, mixture_parameterization="collapsed", routing_floor=floor,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss, _ = model.negative_elbo(
            task.observed_indices, task.observed_targets,
            total_count=len(task.observed_targets), samples=3)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite ELBO at step {step + 1}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        optimizer.step()
    with torch.no_grad():
        prediction = model.posterior_mean(task.test_indices)
        error = nrmse(prediction, task.test_targets)
        params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"test_nrmse": error, "trainable_parameter_count": params}


def cp_reference(field: torch.Tensor, rank: int, iterations: int = 90) -> float:
    """Fully observed CP error: how much of the gap is the model, not the kernel."""
    T = field.double().cpu()
    g = torch.Generator().manual_seed(0)
    F = [torch.randn(s, rank, generator=g, dtype=torch.float64) for s in T.shape]
    for _ in range(iterations):
        for m in range(3):
            o = [i for i in range(3) if i != m]
            kr = (F[o[0]][:, None, :] * F[o[1]][None, :, :]).reshape(-1, rank)
            unf = T.movedim(m, 0).reshape(T.shape[m], -1)
            gram = (F[o[0]].T @ F[o[0]]) * (F[o[1]].T @ F[o[1]])
            F[m] = torch.linalg.lstsq(gram, (unf @ kr).T).solution.T
    return float((torch.einsum("ir,jr,kr->ijk", *F) - T).norm() / T.norm())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--ratios", type=float, nargs="+", default=[0.05, 0.10, 0.20])
    parser.add_argument("--rank", type=int, default=10)
    parser.add_argument("--atoms", type=int, default=4)
    parser.add_argument("--max-frequency", type=int, default=6)
    parser.add_argument("--time-points", type=int, default=32)
    parser.add_argument("--spatial-stride", type=int, default=8)
    parser.add_argument("--component", type=int, default=1)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "pdebench_probe")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    ops, separation_error = operator_atoms(args.max_frequency, args.atoms, NOMINAL)
    _, generic = extended_generic_dictionary(args.atoms, args.max_frequency)
    generic = generic[None].expand(3, -1, -1).clone()
    print(f"operator separation relative error: {separation_error:.4f}", flush=True)

    records = []
    for sample in args.samples:
        field = load_field(sample, component=args.component,
                           time_points=args.time_points, spatial_stride=args.spatial_stride)
        ceiling = cp_reference(field, args.rank)
        print(f"\nsample {sample}  shape {tuple(field.shape)}  "
              f"fully-observed CP{args.rank} error {ceiling:.4f}", flush=True)
        for ratio in args.ratios:
            task = make_task(field, ratio=ratio, seed=sample, device=device)
            row = {"sample": sample, "ratio": ratio, "cp_ceiling": ceiling,
                   "observed": int(len(task.observed_targets))}
            for name, bank in (("operator", ops), ("generic", generic)):
                row[name] = train(task, bank, rank=args.rank, steps=args.steps,
                                  seed=sample, device=device)
            row["margin"] = row["generic"]["test_nrmse"] - row["operator"]["test_nrmse"]
            records.append(row)
            print(f"  ratio {ratio:4.2f}  n={row['observed']:5d}  "
                  f"operator {row['operator']['test_nrmse']:.4f}  "
                  f"generic {row['generic']['test_nrmse']:.4f}  "
                  f"margin {row['margin']:+.4f}", flush=True)
    summary = {"config": vars(args) | {"output": str(args.output)},
               "nominal_operator": NOMINAL,
               "operator_separation_relative_error": separation_error,
               "records": records}
    wins = sum(1 for r in records if r["margin"] > 0)
    summary["operator_wins"] = f"{wins}/{len(records)}"
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"\noperator beats generic in {wins}/{len(records)} (sample, ratio) cells")


if __name__ == "__main__":
    main()
