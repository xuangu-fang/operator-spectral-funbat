#!/usr/bin/env python3
"""Development-only sweep for the operator-knowledge ladder (K2/K1/K0/K-1).

Seeds 101--105 were already exposed by earlier audits and are development
seeds; confirmation will require untouched seeds 301--305 because 201--205 are
now published.  Nothing here may be promoted to a main table.

The question is whether the K1 level --- "we know it is anisotropic diffusion
but not the diffusion tensor" --- retains most of the K2 advantage.  The two
controls that can kill the claim are run alongside, not afterwards:

  * an atom-count-matched *generic* bank (does a pooled operator bank simply
    win on bank size?);
  * an atom-count-matched *wrong-family* pooled bank (does the correct operator
    family do any work at all?).

Every declared quantity below is frozen before the sweep and must not be
retuned against the resulting numbers.
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
    all_grid_indices,
    extended_generic_dictionary,
    generic_spectral_dictionary,
    nonnegative_cp_spectrum,
    normalize_spectrum,
    operator_joint_spectrum,
    sample_planted_tensor,
)
from run_submission_confirmation import (  # noqa: E402
    DEVELOPMENT_SEEDS,
    ESCAPE_FLOOR,
    GRID_SIZE,
    MAX_FREQUENCY,
    RATIO,
    SEPARATION_RANK,
    STEPS,
    TENSOR_RANK,
    one_hot_routing,
    train_model,
)

# ---------------------------------------------------------------------------
# Pre-declared protocol.  Frozen before the sweep.
# ---------------------------------------------------------------------------

# Headline family: anisotropic reaction-diffusion.  Identical to the frozen
# confirmation case so that the K2 column reproduces a known reference point.
TRUE_CONFIG: dict[str, Any] = {
    "operator": "diffusion",
    "source_scale": 0.085,
    "reaction": 0.55,
    "diffusion_coefficients": (0.12, 1.8, 0.35),
}
# theta = (reaction, D_x, D_y, D_t).  The forcing spectrum ``source_scale`` is
# assumed known; forcing misspecification is a separate axis, not this one.
THETA_KEYS = ("reaction", "D_x", "D_y", "D_t")
THETA_STAR = np.array([0.55, 0.12, 1.8, 0.35])

WIDTH_K1 = 3.0    # each component multiplied by LogUniform[1/3, 3]
WIDTH_K0 = 10.0   # each component multiplied by LogUniform[1/10, 10]
DRAWS = 6         # M
ATOMS_PER_DRAW = 2  # Q1  ->  pooled bank has DRAWS * ATOMS_PER_DRAW atoms
POOLED_ATOMS = DRAWS * ATOMS_PER_DRAW
FAMILY_SAMPLING_SEED = 17
SEPARATION_STEPS = 1600

# Wrong family: advection-diffusion, i.e. the operator a practitioner would
# plausibly confuse with diffusion.  Deliberately not the damped wave, whose
# spectrum is grossly different and would make the control trivially easy.
WRONG_FAMILY_BASE: dict[str, Any] = {
    "operator": "advection",
    "source_scale": 0.085,
    "advection_diffusivity": (0.18, 0.252),
    "advection_velocity": (0.9, -0.55),
    "advection_reaction": 0.6,
}
WRONG_THETA_STAR = np.array([0.6, 0.18, 0.252, 0.9, -0.55])


def latin_hypercube_log(centre: np.ndarray, width: float, count: int, seed: int) -> np.ndarray:
    """Latin-hypercube draws multiplying each component by LogUniform[1/w, w].

    Signs are preserved so that a negative velocity component stays negative;
    only the magnitude is perturbed.
    """
    if width <= 1.0 or count < 1:
        raise ValueError("width must exceed 1 and count must be positive")
    rng = np.random.default_rng(seed)
    dimension = len(centre)
    unit = np.empty((count, dimension))
    for axis in range(dimension):
        unit[:, axis] = (rng.permutation(count) + rng.random(count)) / count
    log_width = np.log(width)
    factor = np.exp(-log_width + unit * 2 * log_width)
    return centre[None] * factor


def diffusion_config(theta: np.ndarray) -> dict[str, Any]:
    return {
        "operator": "diffusion",
        "source_scale": TRUE_CONFIG["source_scale"],
        "reaction": float(theta[0]),
        "diffusion_coefficients": (float(theta[1]), float(theta[2]), float(theta[3])),
    }


def advection_config(theta: np.ndarray) -> dict[str, Any]:
    return {
        "operator": "advection",
        "source_scale": WRONG_FAMILY_BASE["source_scale"],
        "advection_reaction": float(theta[0]),
        "advection_diffusivity": (float(theta[1]), float(theta[2])),
        "advection_velocity": (float(theta[3]), float(theta[4])),
    }


def separate(config: dict[str, Any], rank: int) -> tuple[torch.Tensor, float]:
    frequency = torch.arange(MAX_FREQUENCY + 1, dtype=torch.float32)
    joint = operator_joint_spectrum(**config, frequencies=frequency)
    result = nonnegative_cp_spectrum(
        joint, rank=rank, steps=SEPARATION_STEPS, seed=FAMILY_SAMPLING_SEED,
    )
    return normalize_spectrum(torch.stack(result.factors)), result.relative_error


def build_pooled_bank(
    centre: np.ndarray, width: float, to_config, seed: int,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Pool nonnegative separations over a parameter family (levels K1/K0)."""
    thetas = latin_hypercube_log(centre, width, DRAWS, seed)
    banks, errors = [], []
    for theta in thetas:
        atoms, error = separate(to_config(theta), ATOMS_PER_DRAW)
        banks.append(atoms)
        errors.append(error)
    pooled = torch.cat(banks, dim=1)
    return pooled, {
        "width": width,
        "draws": DRAWS,
        "atoms_per_draw": ATOMS_PER_DRAW,
        "atom_count": int(pooled.shape[1]),
        "sampled_theta": thetas.tolist(),
        "separation_relative_error_per_draw": errors,
    }


def bank_diversity(atoms: torch.Tensor) -> dict[str, float]:
    """Mean/max off-diagonal cosine between atoms, averaged over modes.

    A pooled operator bank is supposed to span a *restricted*, physically
    reachable set.  If its atoms were as mutually diverse as a generic
    dictionary of the same size, the "physically constrained subset" reading
    would not hold, so this is reported next to every bank.
    """
    means, maxima = [], []
    for mode in range(atoms.shape[0]):
        normalized = atoms[mode] / atoms[mode].norm(dim=-1, keepdim=True).clamp_min(1e-12)
        gram = normalized @ normalized.T
        count = gram.shape[0]
        if count < 2:
            means.append(1.0)
            maxima.append(1.0)
            continue
        mask = ~torch.eye(count, dtype=torch.bool)
        means.append(float(gram[mask].mean()))
        maxima.append(float(gram[mask].max()))
    return {
        "mean_pairwise_cosine": float(np.mean(means)),
        "max_pairwise_cosine": float(np.mean(maxima)),
    }


def generic_floor_vector(operator_atoms: int, generic_atoms: int, device) -> torch.Tensor:
    return torch.cat((
        torch.zeros(operator_atoms, device=device),
        torch.full((generic_atoms,), ESCAPE_FLOOR / generic_atoms, device=device),
    ))


def build_banks(device: torch.device) -> tuple[dict[str, Any], dict[str, Any]]:
    """Construct every bank once; they do not depend on the data seed."""
    k2_atoms, k2_error = separate(TRUE_CONFIG, SEPARATION_RANK)

    # K1-single: one wrong parameter guess, no pooling.  Isolates whether the
    # benefit comes from pooling or merely from an operator-shaped prior.
    single_theta = latin_hypercube_log(THETA_STAR, WIDTH_K1, DRAWS, FAMILY_SAMPLING_SEED)[0]
    k1_single_atoms, k1_single_error = separate(diffusion_config(single_theta), SEPARATION_RANK)

    k1_atoms, k1_meta = build_pooled_bank(
        THETA_STAR, WIDTH_K1, diffusion_config, FAMILY_SAMPLING_SEED,
    )
    k0_atoms, k0_meta = build_pooled_bank(
        THETA_STAR, WIDTH_K0, diffusion_config, FAMILY_SAMPLING_SEED + 1,
    )
    wrong_atoms, wrong_meta = build_pooled_bank(
        WRONG_THETA_STAR, WIDTH_K1, advection_config, FAMILY_SAMPLING_SEED,
    )

    _, generic4_cpu = generic_spectral_dictionary(MAX_FREQUENCY)
    generic4 = generic4_cpu[None].expand(3, -1, -1).clone()
    _, generic_matched_cpu = extended_generic_dictionary(POOLED_ATOMS, MAX_FREQUENCY)
    generic_matched = generic_matched_cpu[None].expand(3, -1, -1).clone()

    banks = {
        "k2": k2_atoms, "k1_single": k1_single_atoms, "k1": k1_atoms, "k0": k0_atoms,
        "wrong_family": wrong_atoms, "generic4": generic4, "generic_matched": generic_matched,
    }
    banks = {name: value.to(device) for name, value in banks.items()}
    meta = {
        "theta_star": {key: float(value) for key, value in zip(THETA_KEYS, THETA_STAR)},
        "k2": {"atom_count": SEPARATION_RANK, "separation_relative_error": k2_error},
        "k1_single": {
            "atom_count": SEPARATION_RANK,
            "sampled_theta": single_theta.tolist(),
            "separation_relative_error": k1_single_error,
        },
        "k1": k1_meta, "k0": k0_meta, "wrong_family": wrong_meta,
        "generic4": {"atom_count": 4}, "generic_matched": {"atom_count": POOLED_ATOMS},
        "diversity": {name: bank_diversity(value.cpu()) for name, value in banks.items()},
    }
    return banks, meta


def run_seed(banks: dict[str, torch.Tensor], seed: int, device: torch.device,
             steps: int) -> dict[str, Any]:
    """One development seed; every method sees the identical field and mask."""
    k2_atoms = banks["k2"]
    _, generic4_cpu = generic_spectral_dictionary(MAX_FREQUENCY)
    generic4 = generic4_cpu[None].expand(3, -1, -1).clone().to(device)

    coordinates = tuple(torch.arange(GRID_SIZE, device=device) / GRID_SIZE for _ in range(3))
    truth_route = one_hot_routing([[0, 1], [1, 2], [2, 3]], SEPARATION_RANK, device)
    # The truth is planted from the K2 atoms, so K1/K0 must *approximate* the
    # generating spectra from neighbouring parameter settings.  This is the
    # point of the experiment, not a leak.
    field = sample_planted_tensor(coordinates, k2_atoms, truth_route, seed=seed + 401)
    all_indices = all_grid_indices(tuple(field.shape), device)
    data_generator = torch.Generator(device=device).manual_seed(seed + 402)
    order = torch.randperm(len(all_indices), generator=data_generator, device=device)
    observed_count = round(RATIO * len(all_indices))
    observed_indices = all_indices[order[:observed_count]]
    test_indices = all_indices[order[observed_count:]]
    observed_targets = field[tuple(observed_indices.T)] + 0.05 * torch.randn(
        observed_count, generator=data_generator, device=device,
    )
    test_noisy_targets = field[tuple(test_indices.T)] + 0.05 * torch.randn(
        len(test_indices), generator=data_generator, device=device,
    )

    def with_generic(operator_bank: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        combined = torch.cat((operator_bank, generic4), dim=1)
        floor = generic_floor_vector(operator_bank.shape[1], generic4.shape[1], device)
        return combined, floor

    methods: dict[str, tuple[torch.Tensor, str, torch.Tensor | None, torch.Tensor | None]] = {}
    for level in ("k2", "k1_single", "k1", "k0", "wrong_family"):
        methods[level] = (banks[level], "per_mode_rank", None, None)
        combined, floor = with_generic(banks[level])
        methods[f"{level}_floor"] = (combined, "per_mode_rank", None, floor)
    methods["km1_generic4"] = (banks["generic4"], "per_mode_rank", None, None)
    methods["km1_generic_matched"] = (banks["generic_matched"], "per_mode_rank", None, None)
    methods["oracle_route"] = (k2_atoms, "fixed", truth_route, None)

    results = {}
    for name, (candidate, routing, fixed, floor) in methods.items():
        results[name] = train_model(
            coordinates=coordinates,
            candidate_atoms=candidate,
            field=field,
            observed_indices=observed_indices,
            observed_targets=observed_targets,
            test_indices=test_indices,
            test_noisy_targets=test_noisy_targets,
            truth_route=truth_route,
            truth_atoms=k2_atoms,
            routing=routing,
            fixed_routing=fixed,
            routing_floor=floor,
            seed=seed,
            steps=steps,
        )
        results[name]["bank_atom_count"] = int(candidate.shape[1])
        print(f"  seed={seed} {name:24s} nrmse={results[name]['test_nrmse']:.4f}", flush=True)
    return {
        "seed": seed,
        "observed_count": observed_count,
        "grid_shape": list(field.shape),
        "models": results,
    }


def paired(records: list[dict[str, Any]], better: str, worse: str) -> dict[str, Any]:
    left = np.array([r["models"][better]["test_nrmse"] for r in records])
    right = np.array([r["models"][worse]["test_nrmse"] for r in records])
    return {
        "wins": int((left < right).sum()),
        "total": len(records),
        "mean_difference": float((left - right).mean()),
    }


def aggregate(records: list[dict[str, Any]], meta: dict[str, Any]) -> dict[str, Any]:
    names = list(records[0]["models"])
    methods = {}
    for name in names:
        values = np.array([r["models"][name]["test_nrmse"] for r in records])
        cosines = np.array([r["models"][name]["induced_spectrum"]["cosine"] for r in records])
        methods[name] = {
            "test_nrmse": {
                "mean": float(values.mean()), "std": float(values.std()),
                "values": values.tolist(),
            },
            "induced_spectrum_cosine_mean": float(cosines.mean()),
            "bank_atom_count": records[0]["models"][name]["bank_atom_count"],
            "trainable_parameter_count": records[0]["models"][name]["trainable_parameter_count"],
            "variational_coefficient_parameter_count":
                records[0]["models"][name]["variational_coefficient_parameter_count"],
        }

    mean = {name: methods[name]["test_nrmse"]["mean"] for name in names}
    gain_k2 = mean["km1_generic4"] - mean["k2"]
    gain_k1 = mean["km1_generic4"] - mean["k1"]
    predictions = {
        "P1_monotone_k2_k1_k0_km1": {
            "sequence": [mean["k2"], mean["k1"], mean["k0"], mean["km1_generic4"]],
            "monotone_nondecreasing": bool(
                mean["k2"] <= mean["k1"] <= mean["k0"] <= mean["km1_generic4"]
            ),
        },
        "P2_k1_retains_half_of_k2_gain": {
            "k2_gain_over_km1": gain_k2, "k1_gain_over_km1": gain_k1,
            "retained_fraction": float(gain_k1 / gain_k2) if gain_k2 > 0 else None,
            "passes": bool(gain_k2 > 0 and gain_k1 >= 0.5 * gain_k2),
        },
        "P3_floor_benefit_grows_as_level_drops": {
            level: mean[f"{level}_floor"] - mean[level]
            for level in ("k2", "k1", "k0")
        },
        "P4_k0_still_beats_km1": paired(records, "k0", "km1_generic4"),
        "P5_k1_beats_atom_matched_generic": paired(records, "k1", "km1_generic_matched"),
        "P6_k1_beats_wrong_family": paired(records, "k1", "wrong_family"),
    }
    return {
        "status": "DEVELOPMENT ONLY - seeds 101-105 are exposed; never promote to a main table",
        "protocol": {
            "seeds": list(DEVELOPMENT_SEEDS), "steps": STEPS, "observation_ratio": RATIO,
            "grid": GRID_SIZE, "tensor_rank": TENSOR_RANK, "max_frequency": MAX_FREQUENCY,
            "theta_keys": list(THETA_KEYS), "width_k1": WIDTH_K1, "width_k0": WIDTH_K0,
            "draws": DRAWS, "atoms_per_draw": ATOMS_PER_DRAW,
            "generic_escape_floor_total": ESCAPE_FLOOR,
            "forcing_spectrum": "source_scale assumed known at every level",
        },
        "banks": meta,
        "methods": methods,
        "predictions": predictions,
    }


def render_figure(summary: dict[str, Any], output: Path) -> None:
    order = ["oracle_route", "k2", "k1", "k1_single", "k0",
             "km1_generic4", "km1_generic_matched", "wrong_family"]
    figure, axis = plt.subplots(figsize=(9.5, 4.6))
    for position, name in enumerate(order):
        values = summary["methods"][name]["test_nrmse"]["values"]
        axis.scatter([position] * len(values), values, alpha=0.55, s=26, color="tab:blue")
        axis.scatter([position], [summary["methods"][name]["test_nrmse"]["mean"]],
                     marker="_", s=520, color="tab:red", zorder=3)
    axis.set_xticks(range(len(order)))
    axis.set_xticklabels(
        [f"{n}\n({summary['methods'][n]['bank_atom_count']} atoms)" for n in order],
        fontsize=8,
    )
    axis.set_ylabel("held-out NRMSE")
    axis.set_title("Knowledge ladder, DEVELOPMENT seeds 101-105 (2% observations)")
    axis.grid(axis="y", alpha=0.3)
    figure.tight_layout()
    figure.savefig(output / "knowledge_ladder_development.png", dpi=150)
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
    print(json.dumps(meta["diversity"], indent=2), flush=True)
    records = []
    for seed in DEVELOPMENT_SEEDS:
        record = run_seed(banks, seed, device, args.steps)
        records.append(record)
        (args.output / f"seed{seed}.json").write_text(
            json.dumps(record, indent=2), encoding="utf-8",
        )
    summary = aggregate(records, meta)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    render_figure(summary, args.output)
    print(json.dumps(summary["predictions"], indent=2))


if __name__ == "__main__":
    main()
