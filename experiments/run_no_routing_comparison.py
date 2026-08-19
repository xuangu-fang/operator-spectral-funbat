#!/usr/bin/env python3
"""The decisive comparison: a derived kernel versus the best tuned kernel.

Earlier ablations left a confound.  The operator prior beat a generic
*dictionary with learned routing*, but a generic dictionary under global routing
tied it, which means part of the margin was the dictionary over-fitting its own
routing rather than the operator spectrum being right.

This removes routing from the question entirely.  The method arm uses a single
fixed spectrum per mode, read straight off the operator with no learnable kernel
parameters at all.  The baseline arm is a one-parameter smooth spectral family
whose length scale is chosen *per mode by held-out error* -- an oracle no
deployable method has.  The comparison then reads:

  how close does the PDE form get you, for free, to a kernel that was tuned on
  the answer?

and, against the same family tuned on the *observed* data instead, whether
tuning from 1% of entries is even reliable.
"""

from __future__ import annotations

import argparse, itertools, json, sys
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
from run_forced_pde_main import NOMINAL, TRUE_SETTING, make_task, nrmse, tucker_ceiling  # noqa: E402
from run_forced_pde_ablation import train  # noqa: E402

LENGTH_SCALES = (0.15, 0.25, 0.4, 0.6, 0.9, 1.4, 2.0)


def squared_exponential(max_frequency: int, length_scale: float) -> torch.Tensor:
    k = torch.arange(max_frequency + 1, dtype=torch.float32)
    return normalize_spectrum_cosine(torch.exp(-0.5 * (length_scale * k).square())[None])[0]


def operator_marginal(max_frequency: int) -> torch.Tensor:
    """One fixed spectrum per mode: marginalise the joint spectrum, no routing."""
    joint = operator_joint_spectrum(
        "reaction_diffusion", torch.arange(max_frequency + 1, dtype=torch.float32),
        source_scale=NOMINAL["source_scale"], reaction_diffusivity=NOMINAL["diffusivity"],
        reaction_rate=NOMINAL["rate"], reaction_damping=NOMINAL["damping"]).permute(2, 0, 1)
    return normalize_spectrum_cosine(torch.stack(
        [joint.sum(dim=(1, 2)), joint.sum(dim=(0, 2)), joint.sum(dim=(0, 1))]))


def operator_separated(max_frequency: int, atoms: int) -> torch.Tensor:
    joint = operator_joint_spectrum(
        "reaction_diffusion", torch.arange(max_frequency + 1, dtype=torch.float32),
        source_scale=NOMINAL["source_scale"], reaction_diffusivity=NOMINAL["diffusivity"],
        reaction_rate=NOMINAL["rate"], reaction_damping=NOMINAL["damping"]).permute(2, 0, 1)
    separated = nonnegative_cp_spectrum(joint, rank=atoms, steps=1600, seed=17)
    return normalize_spectrum_cosine(torch.stack(separated.factors))


def single(spectrum_per_mode: torch.Tensor) -> torch.Tensor:
    """Wrap [3, freq] as a one-atom bank so routing is a no-op."""
    return spectrum_per_mode[:, None, :].clone()


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
    parser.add_argument("--tag", default="no_routing")
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "forced_pde")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device); ranks = tuple(args.ranks)

    marginal = operator_marginal(args.max_frequency)
    separated = operator_separated(args.max_frequency, args.atoms)
    _, generic_raw = extended_generic_dictionary(args.atoms, args.max_frequency)
    generic = normalize_spectrum_cosine(generic_raw)[None].expand(3, -1, -1).clone()
    se = {ls: squared_exponential(args.max_frequency, ls) for ls in LENGTH_SCALES}

    summary = {"length_scales": list(LENGTH_SCALES), "ratios": {}}
    for ratio in args.ratios:
        rows: dict[str, list[float]] = {}
        tuned_choices = []
        for seed in args.seeds:
            solved = solve_forced(seed=seed, **TRUE_SETTING)
            field, observed, targets, test, truth = make_task(
                solved.field, ratio, seed, args.noise_std, device)

            def run(bank, routing="per_mode_rank"):
                return train(field, observed, targets, test, truth, bank, ranks=ranks,
                             steps=args.steps, seed=seed, device=device, lr=args.lr,
                             routing=routing)["test_nrmse"]

            rows.setdefault("operator_marginal_fixed", []).append(run(single(marginal)))
            rows.setdefault("operator_separated_routed", []).append(run(separated))
            rows.setdefault("generic_dictionary_routed", []).append(run(generic))
            rows.setdefault("generic_dictionary_global", []).append(run(generic, "global"))
            # Oracle-tuned isotropic SE kernel: one length scale shared by all modes.
            shared = {ls: run(single(torch.stack([se[ls]] * 3))) for ls in LENGTH_SCALES}
            best_shared = min(shared, key=shared.get)
            rows.setdefault("se_oracle_shared", []).append(shared[best_shared])
            # Oracle-tuned per-mode SE kernel: a separate length scale per mode,
            # chosen greedily on held-out error.  The strongest tuned baseline.
            choice = [best_shared] * 3
            for mode in range(3):
                scored = {}
                for ls in LENGTH_SCALES:
                    trial = list(choice); trial[mode] = ls
                    scored[ls] = run(single(torch.stack([se[l] for l in trial])))
                choice[mode] = min(scored, key=scored.get)
            best_per_mode = run(single(torch.stack([se[l] for l in choice])))
            rows.setdefault("se_oracle_per_mode", []).append(best_per_mode)
            tuned_choices.append({"shared": best_shared, "per_mode": list(choice)})
            print(f"  seed {seed} ratio {ratio:5.3f}  " + "  ".join(
                f"{k}={v[-1]:.4f}" for k, v in rows.items()) +
                f"  tuned_ls={choice}", flush=True)
        block = {k: {"mean": float(np.mean(v)), "std": float(np.std(v)), "values": v}
                 for k, v in rows.items()}
        block["oracle_length_scales"] = tuned_choices
        reference = np.array(rows["operator_marginal_fixed"])
        for name in rows:
            block[name]["wins_against_operator_marginal"] = int(
                (np.array(rows[name]) < reference).sum())
        summary["ratios"][str(ratio)] = block
        print(f"ratio {ratio}: " + "  ".join(
            f"{k}={block[k]['mean']:.4f}" for k in rows), flush=True)
    (args.output / f"{args.tag}_summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
