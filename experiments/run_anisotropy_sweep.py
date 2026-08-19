#!/usr/bin/env python3
"""Does the *shape* of the operator spectrum matter, or only its smoothness?

The ablation exposed a confound: at low anisotropy, a generic dictionary with
global routing ties the operator prior, and the apparent margin came mostly from
the generic dictionary over-fitting when given free per-mode/rank routing.  Any
smooth low-pass prior looked equally good.

The mechanism actually claimed is stronger and sharper than "be smooth": an
anisotropic operator implies *different spectra along different axes*, which a
dictionary shared across modes cannot express no matter how it is routed.  This
sweep varies the anisotropy ratio Dx/Dy and measures whether the margin grows
with it.  A flat curve would refute the mechanism; a growing one isolates it
from smoothness.

Both routings are reported for both banks, so the confound stays visible.
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
    NOMINAL, TRUE_SETTING, make_task, nrmse, tucker_ceiling,
)
from run_forced_pde_ablation import train  # noqa: E402

BASE_DIFFUSIVITY = 0.012   # geometric mean held fixed while the ratio varies


def prior_for_ratio(ratio: float, max_frequency: int, atoms: int) -> torch.Tensor:
    """Prior from the *form* with the anisotropy ratio known but scale nominal."""
    joint = operator_joint_spectrum(
        "reaction_diffusion", torch.arange(max_frequency + 1, dtype=torch.float32),
        source_scale=NOMINAL["source_scale"],
        reaction_diffusivity=(float(np.sqrt(ratio)), float(1 / np.sqrt(ratio))),
        reaction_rate=NOMINAL["rate"], reaction_damping=NOMINAL["damping"])
    separated = nonnegative_cp_spectrum(joint.permute(2, 0, 1), rank=atoms, steps=1600, seed=17)
    return normalize_spectrum_cosine(torch.stack(separated.factors))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--ratios", type=float, nargs="+", default=[1.0, 3.0, 10.0, 30.0])
    parser.add_argument("--observation-ratio", type=float, default=0.01)
    parser.add_argument("--ranks", type=int, nargs=3, default=[8, 5, 5])
    parser.add_argument("--atoms", type=int, default=4)
    parser.add_argument("--max-frequency", type=int, default=8)
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--lr", type=float, default=0.02)
    parser.add_argument("--noise-std", type=float, default=0.05)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--tag", default="anisotropy")
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "forced_pde")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device); ranks = tuple(args.ranks)

    _, generic_raw = extended_generic_dictionary(args.atoms, args.max_frequency)
    generic = normalize_spectrum_cosine(generic_raw)[None].expand(3, -1, -1).clone()

    summary = {"observation_ratio": args.observation_ratio, "base_diffusivity": BASE_DIFFUSIVITY,
               "seeds": args.seeds, "anisotropy": {}}
    for ratio in args.ratios:
        dx = BASE_DIFFUSIVITY * float(np.sqrt(ratio))
        dy = BASE_DIFFUSIVITY / float(np.sqrt(ratio))
        prior = prior_for_ratio(ratio, args.max_frequency, args.atoms)
        arms = {"operator_per_mode_rank": (prior, "per_mode_rank"),
                "operator_global": (prior, "global"),
                "generic_per_mode_rank": (generic, "per_mode_rank"),
                "generic_global": (generic, "global")}
        rows = {name: [] for name in arms}
        ceilings = []
        for seed in args.seeds:
            solved = solve_forced(seed=seed, **dict(TRUE_SETTING, diffusivity=(dx, dy)))
            ceilings.append(tucker_ceiling(solved.field, ranks))
            field, observed, targets, test, truth = make_task(
                solved.field, args.observation_ratio, seed, args.noise_std, device)
            for name, (bank, routing) in arms.items():
                rows[name].append(train(
                    field, observed, targets, test, truth, bank, ranks=ranks,
                    steps=args.steps, seed=seed, device=device, lr=args.lr,
                    routing=routing)["test_nrmse"])
        block = {name: {"mean": float(np.mean(v)), "std": float(np.std(v)), "values": v}
                 for name, v in rows.items()}
        best_operator = min(("operator_per_mode_rank", "operator_global"),
                            key=lambda n: block[n]["mean"])
        best_generic = min(("generic_per_mode_rank", "generic_global"),
                           key=lambda n: block[n]["mean"])
        paired = np.array(block[best_operator]["values"]) - np.array(block[best_generic]["values"])
        block["best_operator_arm"] = best_operator
        block["best_generic_arm"] = best_generic
        # Compare each side at *its own* best routing, so the margin cannot be
        # an artefact of handicapping the baseline with a bad routing choice.
        block["margin_best_vs_best"] = float(-paired.mean())
        block["wins_best_vs_best"] = int((paired < 0).sum())
        block["tucker_ceiling_mean"] = float(np.mean(ceilings))
        block["diffusivity"] = (dx, dy)
        summary["anisotropy"][str(ratio)] = block
        print(f"ratio {ratio:5.1f}  D=({dx:.4f},{dy:.4f}) ceil={np.mean(ceilings):.3f}  "
              f"op/pmr {block['operator_per_mode_rank']['mean']:.4f}  "
              f"op/glob {block['operator_global']['mean']:.4f}  "
              f"gen/pmr {block['generic_per_mode_rank']['mean']:.4f}  "
              f"gen/glob {block['generic_global']['mean']:.4f}  "
              f"best-vs-best {block['margin_best_vs_best']:+.4f} "
              f"({block['wins_best_vs_best']}/{len(args.seeds)})", flush=True)
    (args.output / f"{args.tag}_summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
