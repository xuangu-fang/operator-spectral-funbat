#!/usr/bin/env python3
"""Where does the gain come from?  Ablations on the forced-PDE main setting.

Every arm shares the field, mask, noise, host model, ranks, optimizer and step
budget with the main experiment; only the spectral bank or the routing changes.

Arms and the question each answers:

  oracle            exact operator spectrum, no nonnegative separation
                    -> how much does the rank-Q projection cost?
  operator          the method                            -> the claim
  operator_global   one shared kernel for all modes/ranks -> is mode-adaptivity
                    doing work, or would a single physics kernel do?
  wrong_advection   advection symbol as the prior         -> must it be the
  wrong_wave        damped-wave symbol as the prior          *right* operator?
  coefficient_*     the right form, coefficients off by a factor
                    -> is only the form needed?
  generic           generic dictionary, matched atom count -> the baseline
  robust            operator + generic with a fixed support floor
"""

from __future__ import annotations

import argparse, json, sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "experiments"))

from geoaware.operator_spectral_funbat import (  # noqa: E402
    ModeAdaptiveVariationalTucker, extended_generic_dictionary, nonnegative_cp_spectrum,
    normalize_spectrum_cosine, operator_joint_spectrum,
)
from forced_pde_solver import solve_forced  # noqa: E402
from run_forced_pde_main import (  # noqa: E402
    NOMINAL, TRUE_SETTING, make_task, nrmse, operator_bank, tucker_ceiling,
)

ESCAPE_FLOOR = 0.25


def separate(joint: torch.Tensor, atoms: int) -> torch.Tensor:
    result = nonnegative_cp_spectrum(joint.permute(2, 0, 1), rank=atoms, steps=1600, seed=17)
    return normalize_spectrum_cosine(torch.stack(result.factors))


def build_banks(max_frequency: int, atoms: int) -> tuple[dict, dict]:
    frequency = torch.arange(max_frequency + 1, dtype=torch.float32)
    banks, meta = {}, {}

    def rd(diffusivity, rate, damping=NOMINAL["damping"]):
        return operator_joint_spectrum(
            "reaction_diffusion", frequency, source_scale=NOMINAL["source_scale"],
            reaction_diffusivity=diffusivity, reaction_rate=rate, reaction_damping=damping)

    joint = rd(NOMINAL["diffusivity"], NOMINAL["rate"])
    banks["operator"] = separate(joint, atoms)
    # Oracle: the exact joint spectrum, marginalised per mode without any
    # nonnegative low-rank projection.  Upper bound on what separation costs.
    exact = joint.permute(2, 0, 1)
    banks["oracle_marginal"] = normalize_spectrum_cosine(torch.stack([
        exact.sum(dim=(1, 2)), exact.sum(dim=(0, 2)), exact.sum(dim=(0, 1))
    ]))[:, None, :].expand(3, atoms, -1).clone()

    # The right form with wrong coefficients: is only the form needed?
    for factor in (0.1, 0.3, 3.0, 10.0):
        scaled = tuple(value * factor for value in NOMINAL["diffusivity"])
        banks[f"coefficient_x{factor:g}"] = separate(rd(scaled, NOMINAL["rate"]), atoms)
    # Isotropic prior: does the anisotropy in the symbol matter?
    banks["isotropic_prior"] = separate(rd((1.0, 1.0), NOMINAL["rate"]), atoms)

    # Wrong operator families, same atom count and same feature budget.
    banks["wrong_advection"] = separate(operator_joint_spectrum(
        "advection", frequency, source_scale=NOMINAL["source_scale"],
        advection_diffusivity=(0.4, 0.12), advection_velocity=(0.9, -0.55),
        advection_reaction=0.6), atoms)
    banks["wrong_wave"] = separate(operator_joint_spectrum(
        "wave", frequency, source_scale=NOMINAL["source_scale"]), atoms)

    _, generic = extended_generic_dictionary(atoms, max_frequency)
    banks["generic"] = normalize_spectrum_cosine(generic)[None].expand(3, -1, -1).clone()
    _, generic_big = extended_generic_dictionary(2 * atoms, max_frequency)
    banks["generic_double"] = normalize_spectrum_cosine(generic_big)[None].expand(3, -1, -1).clone()
    banks["robust"] = torch.cat((banks["operator"], banks["generic"]), dim=1)
    meta["atom_counts"] = {name: int(bank.shape[1]) for name, bank in banks.items()}
    return banks, meta


def train(field, observed, targets, test, truth, spectra, *, ranks, steps, seed,
          device, lr, routing="per_mode_rank", floor=None):
    torch.manual_seed(seed + 10_000)
    model = ModeAdaptiveVariationalTucker(
        tuple(torch.arange(s, device=device) / s for s in field.shape),
        spectra.to(device), ranks=ranks, routing=routing, noise_std=0.08,
        basis=("cosine", "cosine", "cosine"), routing_floor=floor).to(device)
    if floor is not None and isinstance(model.routing_logits, torch.nn.ParameterList):
        with torch.no_grad():
            for logits in model.routing_logits:
                logits[:, floor > 0] = -2.0
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss, _ = model.negative_elbo(observed, targets, total_count=len(targets), samples=3)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        optimizer.step()
    with torch.no_grad():
        return {"test_nrmse": nrmse(model.posterior_mean(test), truth),
                "trainable_parameter_count": sum(
                    p.numel() for p in model.parameters() if p.requires_grad)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--ratio", type=float, default=0.02)
    parser.add_argument("--ranks", type=int, nargs=3, default=[8, 5, 5])
    parser.add_argument("--atoms", type=int, default=4)
    parser.add_argument("--max-frequency", type=int, default=8)
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--lr", type=float, default=0.02)
    parser.add_argument("--noise-std", type=float, default=0.05)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--tag", default="ablation")
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "forced_pde")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device); ranks = tuple(args.ranks)

    banks, meta = build_banks(args.max_frequency, args.atoms)
    floor = torch.cat((torch.zeros(args.atoms),
                       torch.full((args.atoms,), ESCAPE_FLOOR / args.atoms)))
    arms = {name: (bank, "per_mode_rank", None) for name, bank in banks.items()}
    arms["robust"] = (banks["robust"], "per_mode_rank", floor)
    arms["operator_global"] = (banks["operator"], "global", None)
    arms["generic_global"] = (banks["generic"], "global", None)

    results = {name: [] for name in arms}
    for seed in args.seeds:
        solved = solve_forced(seed=seed, **TRUE_SETTING)
        field, observed, targets, test, truth = make_task(
            solved.field, args.ratio, seed, args.noise_std, device)
        for name, (bank, routing, arm_floor) in arms.items():
            out = train(field, observed, targets, test, truth, bank, ranks=ranks,
                        steps=args.steps, seed=seed, device=device, lr=args.lr,
                        routing=routing, floor=arm_floor)
            results[name].append(out)
            print(f"  seed {seed} {name:22s} {out['test_nrmse']:.4f}", flush=True)

    summary = {"ratio": args.ratio, "seeds": args.seeds, "meta": meta, "arms": {}}
    reference = np.array([r["test_nrmse"] for r in results["operator"]])
    for name, rows in results.items():
        values = np.array([r["test_nrmse"] for r in rows])
        summary["arms"][name] = {
            "mean": float(values.mean()), "std": float(values.std()),
            "values": values.tolist(),
            "wins_against_operator": int((values < reference).sum()),
            "parameters": rows[0]["trainable_parameter_count"],
        }
    (args.output / f"{args.tag}_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n{'arm':24s}{'mean':>9s}{'std':>9s}{'atoms':>7s}{'params':>8s}")
    for name in sorted(summary["arms"], key=lambda n: summary["arms"][n]["mean"]):
        arm = summary["arms"][name]
        print(f"{name:24s}{arm['mean']:9.4f}{arm['std']:9.4f}"
              f"{meta['atom_counts'].get(name, '-'):>7}{arm['parameters']:8d}")


if __name__ == "__main__":
    main()
