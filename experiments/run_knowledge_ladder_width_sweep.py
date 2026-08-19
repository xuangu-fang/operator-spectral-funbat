#!/usr/bin/env python3
"""Development-only follow-up to the knowledge-ladder sweep.

The first sweep produced two results that this script is designed to pin down.

1.  A *single* plausible coefficient guess did as well as a six-draw pooled
    family, so the pooling machinery may be unnecessary.  Here both variants are
    run across a range of prior widths to find where either one degrades.
2.  At the K0 width the pooled operator bank lost to an atom-count-matched
    generic bank, i.e. the pre-declared degeneracy boundary was reached.  The
    width sweep turns that binary failure into a measured boundary.

It also adds a second wrong-family control.  Advection was too close to
diffusion in even magnitude spectrum to be discriminative; the damped wave has a
dispersion surface and should be much further away.  Wave parameters were
exposed for this purpose with defaults that reproduce the frozen literals.

Development seeds only.  Nothing here may enter a main table, and the widths
swept here must be re-frozen before any confirmation run.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

from geoaware.operator_spectral_funbat import (  # noqa: E402
    all_grid_indices, extended_generic_dictionary, generic_spectral_dictionary,
    sample_planted_tensor,
)
from run_submission_confirmation import (  # noqa: E402
    DEVELOPMENT_SEEDS, GRID_SIZE, MAX_FREQUENCY, RATIO, SEPARATION_RANK, STEPS,
    one_hot_routing, train_model,
)
from run_knowledge_ladder_development import (  # noqa: E402
    ATOMS_PER_DRAW, DRAWS, FAMILY_SAMPLING_SEED, POOLED_ATOMS, THETA_STAR,
    TRUE_CONFIG, WRONG_THETA_STAR, advection_config, bank_diversity,
    build_pooled_bank, diffusion_config, latin_hypercube_log, separate,
)

WIDTHS = (1.5, 3.0, 5.0, 10.0, 20.0)
WAVE_THETA_STAR = np.array([1.35, 0.65, 0.45, 0.18])  # (c_x, c_y, gamma_0, gamma_1)


def wave_config(theta: np.ndarray) -> dict[str, Any]:
    return {
        "operator": "wave",
        "source_scale": TRUE_CONFIG["source_scale"],
        "wave_coefficients": (float(theta[0]), float(theta[1])),
        "wave_damping": (float(theta[2]), float(theta[3])),
    }


def build_banks(device: torch.device) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    banks: dict[str, torch.Tensor] = {}
    meta: dict[str, Any] = {"widths": list(WIDTHS)}
    k2_atoms, k2_error = separate(TRUE_CONFIG, SEPARATION_RANK)
    banks["k2"] = k2_atoms
    meta["k2"] = {"separation_relative_error": k2_error}

    for width in WIDTHS:
        pooled, pooled_meta = build_pooled_bank(
            THETA_STAR, width, diffusion_config, FAMILY_SAMPLING_SEED,
        )
        banks[f"pooled_w{width}"] = pooled
        meta[f"pooled_w{width}"] = pooled_meta
        # A single draw from the same prior, at the frozen K2 atom count.
        theta = latin_hypercube_log(THETA_STAR, width, DRAWS, FAMILY_SAMPLING_SEED)[0]
        single, single_error = separate(diffusion_config(theta), SEPARATION_RANK)
        banks[f"single_w{width}"] = single
        meta[f"single_w{width}"] = {
            "sampled_theta": theta.tolist(), "separation_relative_error": single_error,
        }

    advection_bank, advection_meta = build_pooled_bank(
        WRONG_THETA_STAR, 3.0, advection_config, FAMILY_SAMPLING_SEED,
    )
    wave_bank, wave_meta = build_pooled_bank(
        WAVE_THETA_STAR, 3.0, wave_config, FAMILY_SAMPLING_SEED,
    )
    banks["wrong_advection"] = advection_bank
    banks["wrong_wave"] = wave_bank
    meta["wrong_advection"] = advection_meta
    meta["wrong_wave"] = wave_meta

    _, generic4 = generic_spectral_dictionary(MAX_FREQUENCY)
    _, generic_matched = extended_generic_dictionary(POOLED_ATOMS, MAX_FREQUENCY)
    banks["generic4"] = generic4[None].expand(3, -1, -1).clone()
    banks["generic_matched"] = generic_matched[None].expand(3, -1, -1).clone()

    banks = {name: value.to(device) for name, value in banks.items()}
    meta["diversity"] = {name: bank_diversity(value.cpu()) for name, value in banks.items()}
    return banks, meta


def run_seed(banks: dict[str, torch.Tensor], seed: int, device: torch.device,
             steps: int) -> dict[str, float]:
    k2_atoms = banks["k2"]
    coordinates = tuple(torch.arange(GRID_SIZE, device=device) / GRID_SIZE for _ in range(3))
    truth_route = one_hot_routing([[0, 1], [1, 2], [2, 3]], SEPARATION_RANK, device)
    field = sample_planted_tensor(coordinates, k2_atoms, truth_route, seed=seed + 401)
    all_indices = all_grid_indices(tuple(field.shape), device)
    generator = torch.Generator(device=device).manual_seed(seed + 402)
    order = torch.randperm(len(all_indices), generator=generator, device=device)
    observed_count = round(RATIO * len(all_indices))
    observed_indices, test_indices = all_indices[order[:observed_count]], all_indices[order[observed_count:]]
    observed_targets = field[tuple(observed_indices.T)] + 0.05 * torch.randn(
        observed_count, generator=generator, device=device)
    test_noisy_targets = field[tuple(test_indices.T)] + 0.05 * torch.randn(
        len(test_indices), generator=generator, device=device)

    out = {}
    for name, bank in banks.items():
        result = train_model(
            coordinates=coordinates, candidate_atoms=bank, field=field,
            observed_indices=observed_indices, observed_targets=observed_targets,
            test_indices=test_indices, test_noisy_targets=test_noisy_targets,
            truth_route=truth_route, truth_atoms=k2_atoms,
            routing="per_mode_rank", fixed_routing=None, routing_floor=None,
            seed=seed, steps=steps,
        )
        out[name] = result["test_nrmse"]
        print(f"  seed={seed} {name:20s} nrmse={result['test_nrmse']:.4f}", flush=True)
    return out


def render_figure(summary: dict[str, Any], output: Path) -> None:
    figure, axis = plt.subplots(figsize=(7.6, 4.4))
    widths = list(WIDTHS)
    for prefix, label, style in (("pooled_w", f"pooled family ({POOLED_ATOMS} atoms)", "-o"),
                                 ("single_w", f"single guess ({SEPARATION_RANK} atoms)", "-s")):
        means = [summary["methods"][f"{prefix}{w}"]["mean"] for w in widths]
        axis.plot(widths, means, style, label=label)
    for name, label, colour in (("k2", "K2 (true coefficients)", "tab:green"),
                                ("generic_matched", "generic, atom-matched", "tab:red"),
                                ("wrong_advection", "wrong family: advection", "tab:orange"),
                                ("wrong_wave", "wrong family: wave", "tab:purple")):
        axis.axhline(summary["methods"][name]["mean"], ls="--", lw=1.2,
                     color=colour, label=label)
    axis.set_xscale("log")
    axis.set_xlabel("coefficient prior width (multiplicative, log scale)")
    axis.set_ylabel("mean held-out NRMSE")
    axis.set_title("Where does operator knowledge stop helping?  DEVELOPMENT seeds 101-105")
    axis.legend(fontsize=7.5)
    axis.grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(output / "knowledge_ladder_width_sweep.png", dpi=150)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--steps", type=int, default=STEPS)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "results" / "knowledge_ladder_development")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    banks, meta = build_banks(device)
    records = [run_seed(banks, seed, device, args.steps) for seed in DEVELOPMENT_SEEDS]
    methods = {}
    for name in records[0]:
        values = np.array([record[name] for record in records])
        methods[name] = {"mean": float(values.mean()), "std": float(values.std()),
                         "values": values.tolist(),
                         "atom_count": int(banks[name].shape[1])}
    summary = {
        "status": "DEVELOPMENT ONLY - widths swept here must be re-frozen before confirmation",
        "protocol": {"seeds": list(DEVELOPMENT_SEEDS), "steps": args.steps,
                     "widths": list(WIDTHS), "draws": DRAWS,
                     "atoms_per_draw": ATOMS_PER_DRAW},
        "banks": meta, "methods": methods,
    }
    (args.output / "width_sweep_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    render_figure(summary, args.output)
    for name in sorted(methods):
        print(f"{name:22s}{methods[name]['mean']:9.4f} +- {methods[name]['std']:.4f}")


if __name__ == "__main__":
    main()
