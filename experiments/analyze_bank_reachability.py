#!/usr/bin/env python3
"""A-priori diagnostic: can a bank reach the truth spectrum at all?

The knowledge-ladder development sweep ordered banks by prediction error, but
prediction error needs training data.  This computes, without any training, how
well each candidate bank can approximate the ground-truth per-mode/rank spectra
under the same nonnegative convex-combination constraint the router obeys.

If this a-priori number predicts the trained ranking, then "is my operator prior
close enough to be useful?" is answerable before spending any observations,
which is the practically useful form of the applicability question.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

from geoaware.operator_spectral_funbat import (  # noqa: E402
    extended_generic_dictionary, generic_spectral_dictionary,
)
from run_submission_confirmation import MAX_FREQUENCY, SEPARATION_RANK  # noqa: E402
from run_knowledge_ladder_development import (  # noqa: E402
    THETA_STAR, TRUE_CONFIG, WRONG_THETA_STAR, advection_config,
    build_pooled_bank, diffusion_config, separate, FAMILY_SAMPLING_SEED, POOLED_ATOMS,
)
from run_knowledge_ladder_width_sweep import WAVE_THETA_STAR, wave_config  # noqa: E402

TRUTH_ROUTE = [[0, 1], [1, 2], [2, 3]]  # frozen planted route, [mode][rank] -> atom


def best_convex_approximation(target: np.ndarray, atoms: np.ndarray,
                              steps: int = 4000, learning_rate: float = 0.5) -> float:
    """Relative L2 of the closest convex combination of ``atoms`` to ``target``.

    Exponentiated-gradient descent on the simplex; deterministic start.
    """
    weights = np.full(len(atoms), 1.0 / len(atoms))
    scale = np.linalg.norm(target)
    for _ in range(steps):
        residual = weights @ atoms - target
        gradient = atoms @ residual
        weights = weights * np.exp(-learning_rate * gradient / max(scale, 1e-12))
        weights /= weights.sum()
    return float(np.linalg.norm(weights @ atoms - target) / max(scale, 1e-12))


def prior_concentration(bank: torch.Tensor, truth_atoms: torch.Tensor,
                        samples: int = 4000, seed: int = 5) -> float:
    """Expected relative L2 to the truth under the router's own prior.

    Reachability alone turned out not to predict performance: a bank can reach
    the truth and still be a bad prior if most of what it can express is wrong.
    This draws routing weights from the uniform simplex --- the distribution the
    untrained router starts from --- and measures the *typical* distance rather
    than the best case.
    """
    rng = np.random.default_rng(seed)
    errors = []
    for mode, row in enumerate(TRUTH_ROUTE):
        atoms = bank[mode].cpu().numpy().astype(np.float64)
        weights = rng.dirichlet(np.ones(len(atoms)), size=samples)
        mixtures = weights @ atoms
        for atom_index in row:
            target = truth_atoms[mode, atom_index].cpu().numpy().astype(np.float64)
            distance = np.linalg.norm(mixtures - target[None], axis=1)
            errors.append(float((distance / max(np.linalg.norm(target), 1e-12)).mean()))
    return float(np.mean(errors))


def reachability(bank: torch.Tensor, truth_atoms: torch.Tensor) -> float:
    """Mean relative L2 over the frozen (mode, rank) truth spectra."""
    errors = []
    for mode, row in enumerate(TRUTH_ROUTE):
        atoms = bank[mode].cpu().numpy().astype(np.float64)
        for atom_index in row:
            target = truth_atoms[mode, atom_index].cpu().numpy().astype(np.float64)
            errors.append(best_convex_approximation(target, atoms))
    return float(np.mean(errors))


def main() -> None:
    truth_atoms, _ = separate(TRUE_CONFIG, SEPARATION_RANK)
    banks = {"k2": truth_atoms}
    for width in (1.5, 3.0, 5.0, 10.0, 20.0):
        banks[f"pooled_w{width}"], _ = build_pooled_bank(
            THETA_STAR, width, diffusion_config, FAMILY_SAMPLING_SEED)
    banks["k0_seed18"], _ = build_pooled_bank(
        THETA_STAR, 10.0, diffusion_config, FAMILY_SAMPLING_SEED + 1)
    banks["wrong_advection"], _ = build_pooled_bank(
        WRONG_THETA_STAR, 3.0, advection_config, FAMILY_SAMPLING_SEED)
    banks["wrong_wave"], _ = build_pooled_bank(
        WAVE_THETA_STAR, 3.0, wave_config, FAMILY_SAMPLING_SEED)
    _, generic4 = generic_spectral_dictionary(MAX_FREQUENCY)
    _, generic_matched = extended_generic_dictionary(POOLED_ATOMS, MAX_FREQUENCY)
    banks["generic4"] = generic4[None].expand(3, -1, -1).clone()
    banks["generic_matched"] = generic_matched[None].expand(3, -1, -1).clone()

    scores = {name: reachability(bank, truth_atoms) for name, bank in banks.items()}
    concentration = {
        name: prior_concentration(bank, truth_atoms) for name, bank in banks.items()
    }
    output = ROOT / "results" / "knowledge_ladder_development" / "bank_reachability.json"
    output.write_text(json.dumps({
        "definition": "mean relative L2 of the best convex combination of bank atoms "
                      "approximating each frozen (mode, rank) truth spectrum; no training",
        "truth_route": TRUTH_ROUTE,
        "scores": scores,
        "prior_concentration_definition":
            "expected relative L2 to the truth spectrum for routing weights drawn "
            "from the uniform simplex; measures how typical, not how possible, a "
            "good fit is",
        "prior_concentration": concentration,
    }, indent=2), encoding="utf-8")
    print(f"{'bank':20s}{'reachability':>14s}{'prior concentration':>22s}")
    for name in sorted(concentration, key=lambda n: concentration[n]):
        print(f"{name:20s}{scores[name]:14.5f}{concentration[name]:22.5f}")


if __name__ == "__main__":
    main()
