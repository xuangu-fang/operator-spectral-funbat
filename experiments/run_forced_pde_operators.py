#!/usr/bin/env python3
"""Does the mechanism hold across operators, and where does it stop?

Three regimes on the same harness:

  anisotropic_diffusion  strongly anisotropic, even symbol   -- the main case
  isotropic_diffusion    same operator, Dx = Dy              -- if per-mode
                         kernels help only through anisotropy, the margin here
                         should shrink toward zero
  advection_diffusion    tilted transport, symbol not axis-wise even -- the
                         known representational limit; a smaller or absent
                         margin here is a prediction, not a surprise

Each regime is given a prior derived from its own equation's form.  Reporting
the regime where the margin disappears is the point, not a caveat.
"""

from __future__ import annotations

import argparse, json, sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "experiments"))

from geoaware.operator_spectral_funbat import (  # noqa: E402
    extended_generic_dictionary, nonnegative_cp_spectrum, normalize_spectrum_cosine,
    operator_joint_spectrum,
)
from forced_pde_solver import solve_forced  # noqa: E402
from run_forced_pde_main import (  # noqa: E402
    NOMINAL, TRUE_SETTING, make_task, nrmse, train, tucker_ceiling,
)

REGIMES = {
    "anisotropic_diffusion": {
        "solver": dict(TRUE_SETTING),
        "prior": dict(family="reaction_diffusion", diffusivity=(1.0, 0.3), rate=-1.0),
    },
    "isotropic_diffusion": {
        "solver": dict(TRUE_SETTING, diffusivity=(0.012, 0.012)),
        "prior": dict(family="reaction_diffusion", diffusivity=(1.0, 1.0), rate=-1.0),
    },
    "advection_diffusion": {
        "solver": dict(TRUE_SETTING, operator="advection_diffusion",
                       diffusivity=(0.012, 0.012), velocity=(0.35, -0.2)),
        "prior": dict(family="advection"),
    },
}


def build_prior(spec: dict, max_frequency: int, atoms: int) -> torch.Tensor:
    frequency = torch.arange(max_frequency + 1, dtype=torch.float32)
    if spec["family"] == "reaction_diffusion":
        joint = operator_joint_spectrum(
            "reaction_diffusion", frequency, source_scale=NOMINAL["source_scale"],
            reaction_diffusivity=spec["diffusivity"], reaction_rate=spec["rate"],
            reaction_damping=NOMINAL["damping"])
    else:
        joint = operator_joint_spectrum(
            "advection", frequency, source_scale=NOMINAL["source_scale"],
            advection_diffusivity=(0.4, 0.4), advection_velocity=(0.9, -0.55),
            advection_reaction=0.6)
    separated = nonnegative_cp_spectrum(joint.permute(2, 0, 1), rank=atoms, steps=1600, seed=17)
    return normalize_spectrum_cosine(torch.stack(separated.factors))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--ratio", type=float, default=0.01)
    parser.add_argument("--ranks", type=int, nargs=3, default=[8, 5, 5])
    parser.add_argument("--atoms", type=int, default=4)
    parser.add_argument("--max-frequency", type=int, default=8)
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--lr", type=float, default=0.02)
    parser.add_argument("--noise-std", type=float, default=0.05)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "forced_pde")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device); ranks = tuple(args.ranks)

    _, generic = extended_generic_dictionary(args.atoms, args.max_frequency)
    generic = normalize_spectrum_cosine(generic)[None].expand(3, -1, -1).clone()

    summary = {"ratio": args.ratio, "regimes": {}}
    for name, spec in REGIMES.items():
        prior = build_prior(spec["prior"], args.max_frequency, args.atoms)
        rows = []
        for seed in args.seeds:
            solved = solve_forced(seed=seed, **spec["solver"])
            ceiling = tucker_ceiling(solved.field, ranks)
            field, observed, targets, test, truth = make_task(
                solved.field, args.ratio, seed, args.noise_std, device)
            cell = {"seed": seed, "tucker_ceiling": ceiling}
            for arm, bank in (("operator", prior), ("generic", generic)):
                # Global routing for both arms: per-mode/rank over-fits at 1%
                # and hurts the generic dictionary about twice as much.
                cell[arm] = train(field, observed, targets, test, truth, bank,
                                  ranks=ranks, steps=args.steps, seed=seed,
                                  device=device, lr=args.lr,
                                  routing="global")["test_nrmse"]
            cell["margin"] = cell["generic"] - cell["operator"]
            rows.append(cell)
            print(f"  {name:22s} seed {seed} ceil={ceiling:.3f} "
                  f"operator {cell['operator']:.4f} generic {cell['generic']:.4f} "
                  f"margin {cell['margin']:+.4f}", flush=True)
        margins = np.array([r["margin"] for r in rows])
        summary["regimes"][name] = {
            "operator_mean": float(np.mean([r["operator"] for r in rows])),
            "generic_mean": float(np.mean([r["generic"] for r in rows])),
            "margin_mean": float(margins.mean()), "margin_std": float(margins.std()),
            "wins": f"{int((margins > 0).sum())}/{len(margins)}",
            "tucker_ceiling_mean": float(np.mean([r["tucker_ceiling"] for r in rows])),
            "rows": rows,
        }
    (args.output / "operators_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n{'regime':24s}{'operator':>10s}{'generic':>10s}{'margin':>10s}{'wins':>7s}")
    for name, block in summary["regimes"].items():
        print(f"{name:24s}{block['operator_mean']:10.4f}{block['generic_mean']:10.4f}"
              f"{block['margin_mean']:+10.4f}{block['wins']:>7s}")


if __name__ == "__main__":
    main()
