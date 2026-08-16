#!/usr/bin/env python3
"""Frozen five-seed confirmation for the operator-spectral submission gate.

The script deliberately does not tune on held-out entries.  Every method within
a (case, seed) pair receives the same field, observation mask, training noise,
held-out noisy targets, rank, Fourier support, optimizer and 400-step budget.
The canonical collapsed spectral-mixture parameterization keeps the GP
coefficient count independent of the number of atoms in a bank.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from geoaware.operator_spectral_funbat import (  # noqa: E402
    ModeAdaptiveVariationalCP,
    all_grid_indices,
    generic_spectral_dictionary,
    nonnegative_cp_spectrum,
    normalize_spectrum,
    operator_joint_spectrum,
    sample_planted_tensor,
)


# 101--105 were consumed by the numerical/fairness audit and are development
# seeds.  The final frozen confirmation uses untouched 201--205.
DEVELOPMENT_SEEDS = (101, 102, 103, 104, 105)
SEEDS = (201, 202, 203, 204, 205)
STEPS = 400
RATIO = 0.02
GRID_SIZE = 24
MAX_FREQUENCY = 6
SEPARATION_RANK = 4
TENSOR_RANK = 2
UQ_POINTS = 1024
UQ_SAMPLES = 64
ESCAPE_FLOOR = 0.25  # frozen only after the separate 101--105 development sweep

OPERATOR_CASES: dict[str, dict[str, Any]] = {
    "reference_advection": {
        "operator": "advection",
        "source_scale": 0.12,
        "advection_diffusivity": (0.18, 0.252),
        "advection_velocity": (0.9, -0.55),
        "advection_reaction": 0.6,
    },
    "shifted_advection": {
        "operator": "advection",
        "source_scale": 0.075,
        "advection_diffusivity": (0.08, 0.38),
        "advection_velocity": (1.35, -0.2),
        "advection_reaction": 0.42,
    },
    "anisotropic_diffusion": {
        "operator": "diffusion",
        "source_scale": 0.085,
        "reaction": 0.55,
        "diffusion_coefficients": (0.12, 1.8, 0.35),
    },
}


def one_hot_routing(indices: list[list[int]], families: int, device: torch.device) -> torch.Tensor:
    route = torch.zeros(3, len(indices[0]), families, device=device)
    for mode, row in enumerate(indices):
        for rank, family in enumerate(row):
            route[mode, rank, family] = 1
    return route


def build_operator_atoms(config: dict[str, Any]) -> tuple[torch.Tensor, dict[str, Any]]:
    frequency = torch.arange(MAX_FREQUENCY + 1, dtype=torch.float32)
    joint = operator_joint_spectrum(**config, frequencies=frequency)
    separations = {
        rank: nonnegative_cp_spectrum(joint, rank=rank, steps=1600, seed=17)
        for rank in range(1, 7)
    }
    separated = separations[SEPARATION_RANK]
    atoms = normalize_spectrum(torch.stack(separated.factors))
    return atoms, {
        "config": config,
        "separation_rank": SEPARATION_RANK,
        "relative_error": separated.relative_error,
        "relative_error_by_rank": {
            str(rank): result.relative_error for rank, result in separations.items()
        },
        "component_weights": separated.weights.tolist(),
    }


def spectrum_metrics(
    learned_weights: torch.Tensor,
    candidate_atoms: torch.Tensor,
    truth_route: torch.Tensor,
    truth_atoms: torch.Tensor,
) -> dict[str, float]:
    learned = torch.einsum("drq,dqk->drk", learned_weights, candidate_atoms)
    truth = torch.einsum("drq,dqk->drk", truth_route, truth_atoms)
    cosine = torch.nn.functional.cosine_similarity(learned, truth, dim=-1)
    relative_l2 = torch.linalg.vector_norm(learned - truth, dim=-1) / torch.linalg.vector_norm(
        truth, dim=-1,
    ).clamp_min(1e-12)
    return {
        "cosine": float(cosine.mean()),
        "relative_l2": float(relative_l2.mean()),
    }


def train_model(
    *,
    coordinates: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    candidate_atoms: torch.Tensor,
    field: torch.Tensor,
    observed_indices: torch.Tensor,
    observed_targets: torch.Tensor,
    test_indices: torch.Tensor,
    test_noisy_targets: torch.Tensor,
    truth_route: torch.Tensor,
    truth_atoms: torch.Tensor,
    routing: str,
    fixed_routing: torch.Tensor | None,
    routing_floor: torch.Tensor | None,
    seed: int,
    steps: int,
) -> dict[str, Any]:
    # Reset before every method.  With collapsed coefficients all methods now
    # have the same coefficient/core initialization and MC-ELBO random stream.
    torch.manual_seed(seed + 10_000)
    model = ModeAdaptiveVariationalCP(
        coordinates,
        candidate_atoms,
        rank=TENSOR_RANK,
        routing=routing,
        fixed_routing=fixed_routing,
        noise_std=0.08,
        mixture_parameterization="collapsed",
        routing_floor=routing_floor,
    ).to(field.device)
    # A robust bank is operator-centred at initialization.  Generic atoms keep
    # their fixed support floor and can gain mass, but eight-way uniform routing
    # was visibly seed-unstable in the development audit.  This bias is frozen
    # before final seeds and is not applied to operator/generic-only baselines.
    if routing_floor is not None and model.routing_logits is not None:
        with torch.no_grad():
            model.routing_logits[..., routing_floor > 0] = -2.0
    optimizer = torch.optim.Adam(model.parameters(), lr=0.025)
    loss_trace = []
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss, diagnostics = model.negative_elbo(
            observed_indices,
            observed_targets,
            total_count=len(observed_targets),
            samples=3,
        )
        if not torch.isfinite(loss):
            raise FloatingPointError(
                f"non-finite ELBO at step {step + 1}; routing={routing}"
            )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        optimizer.step()
        if step in (0, 99, 199, 299, steps - 1):
            loss_trace.append({
                "step": step + 1,
                "negative_elbo_per_observation": float(loss.detach()),
                "kl": float(diagnostics["kl"]),
                "noise_std": float(diagnostics["noise_std"]),
            })

    with torch.no_grad():
        prediction = model.posterior_mean(test_indices)
        truth = field[tuple(test_indices.T)]
        test_nrmse = torch.sqrt(torch.mean((prediction - truth).square())) / truth.std().clamp_min(1e-8)
        observed_prediction = model.posterior_mean(observed_indices)
        observed_truth = field[tuple(observed_indices.T)]
        observed_nrmse = torch.sqrt(torch.mean((observed_prediction - observed_truth).square())) / observed_truth.std().clamp_min(1e-8)

        # UQ is evaluated once after the fixed training budget on an untouched,
        # fixed subset with independently generated observation noise.
        uq_indices = test_indices[:UQ_POINTS]
        uq_targets = test_noisy_targets[:UQ_POINTS]
        uq_generator = torch.Generator(device=field.device).manual_seed(seed + 30_000)
        latent_samples = model.posterior_predictive_samples(
            uq_indices, samples=UQ_SAMPLES, generator=uq_generator, include_noise=False,
        )
        conditional_log_prob = -0.5 * (
            math.log(2 * math.pi)
            + 2 * torch.log(model.noise_std)
            + (uq_targets[None] - latent_samples).square() / model.noise_std.square()
        )
        predictive_nll = -(torch.logsumexp(conditional_log_prob, dim=0) - math.log(UQ_SAMPLES)).mean()
        predictive_samples = latent_samples + model.noise_std * torch.randn(
            latent_samples.shape,
            generator=uq_generator,
            device=field.device,
            dtype=field.dtype,
        )
        lower = torch.quantile(predictive_samples, 0.025, dim=0)
        upper = torch.quantile(predictive_samples, 0.975, dim=0)
        coverage = ((uq_targets >= lower) & (uq_targets <= upper)).float().mean()
        interval_width = (upper - lower).mean()
        exact_mean, exact_latent_variance = model.posterior_moments(uq_indices)
        moment_error = (exact_mean - latent_samples.mean(0)).abs().mean()
        parameter_count = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        coefficient_count = model.variational_mean.numel() + model.raw_variational_std.numel()

    return {
        "test_nrmse": float(test_nrmse),
        "observed_clean_nrmse": float(observed_nrmse),
        "induced_spectrum": spectrum_metrics(
            model.routing_weights().detach(), candidate_atoms, truth_route, truth_atoms,
        ),
        "uq": {
            "predictive_coverage_95": float(coverage),
            "predictive_nll_mc": float(predictive_nll),
            "mean_interval_width_95": float(interval_width),
            "points": UQ_POINTS,
            "posterior_samples": UQ_SAMPLES,
            "target": "independently noised held-out observations",
            "exact_vs_mc_latent_mean_mae": float(moment_error),
            "mean_exact_latent_variance": float(exact_latent_variance.mean()),
        },
        "noise_std": float(model.noise_std.detach()),
        "routing_weights": model.routing_weights().detach().cpu().tolist(),
        "trainable_parameter_count": parameter_count,
        "variational_coefficient_parameter_count": coefficient_count,
        "loss_trace": loss_trace,
    }


def run_case(
    case_name: str,
    config: dict[str, Any],
    atoms_cpu: torch.Tensor,
    separation: dict[str, Any],
    seed: int,
    device: torch.device,
    steps: int,
    escape_floor: float = ESCAPE_FLOOR,
    only_methods: set[str] | None = None,
) -> dict[str, Any]:
    atoms = atoms_cpu.to(device)
    _, generic_cpu = generic_spectral_dictionary(MAX_FREQUENCY)
    generic = generic_cpu[None].expand(3, -1, -1).clone().to(device)
    robust = torch.cat((atoms, generic), dim=1)
    wrong_support = atoms.clone()
    wrong_support[..., 2:] = 0
    wrong_support = normalize_spectrum(wrong_support)
    wrong_support_robust = torch.cat((wrong_support, generic), dim=1)

    coordinates = tuple(torch.arange(GRID_SIZE, device=device) / GRID_SIZE for _ in range(3))
    truth_route = one_hot_routing([[0, 1], [1, 2], [2, 3]], SEPARATION_RANK, device)
    field = sample_planted_tensor(coordinates, atoms, truth_route, seed=seed + 401)
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

    oracle_robust = torch.cat(
        (truth_route, torch.zeros(3, TENSOR_RANK, generic.shape[1], device=device)), dim=-1,
    )
    # The robust method reserves 25% total prior mass for the four generic
    # atoms.  A merely optional generic bank collapsed back to the misspecified
    # operator in pilot optimization and therefore did not provide a real
    # support guarantee.  The floor is fixed before fresh-seed confirmation.
    if not 0 <= escape_floor < 1:
        raise ValueError("escape_floor must lie in [0,1)")
    generic_escape_floor = torch.cat((
        torch.zeros(SEPARATION_RANK, device=device),
        torch.full((generic.shape[1],), escape_floor / generic.shape[1], device=device),
    ))
    methods: dict[str, tuple[torch.Tensor, str, torch.Tensor | None, torch.Tensor | None]] = {
        "operator_global": (atoms, "global", None, None),
        "operator_per_mode_rank": (atoms, "per_mode_rank", None, None),
        "generic_global": (generic, "global", None, None),
        "generic_per_mode_rank": (generic, "per_mode_rank", None, None),
        "robust_global": (robust, "global", None, generic_escape_floor),
        "robust_per_mode_rank": (robust, "per_mode_rank", None, generic_escape_floor),
        "oracle_operator_route": (robust, "fixed", oracle_robust, None),
        "wrong_support_operator": (wrong_support, "per_mode_rank", None, None),
        "wrong_support_robust": (wrong_support_robust, "per_mode_rank", None, generic_escape_floor),
    }
    results = {}
    for method_name, (candidate, routing, fixed, floor) in methods.items():
        if only_methods is not None and method_name not in only_methods:
            continue
        results[method_name] = train_model(
            coordinates=coordinates,
            candidate_atoms=candidate,
            field=field,
            observed_indices=observed_indices,
            observed_targets=observed_targets,
            test_indices=test_indices,
            test_noisy_targets=test_noisy_targets,
            truth_route=truth_route,
            truth_atoms=atoms,
            routing=routing,
            fixed_routing=fixed,
            routing_floor=floor,
            seed=seed,
            steps=steps,
        )
    return {
        "case": case_name,
        "operator_config": config,
        "operator_separation": separation,
        "seed": seed,
        "observation_ratio": RATIO,
        "observed_count": observed_count,
        "grid_shape": list(field.shape),
        "training_noise_std": 0.05,
        "protocol": {
            "steps": steps,
            "tensor_rank": TENSOR_RANK,
            "frequency_support": list(range(MAX_FREQUENCY + 1)),
            "operator_atoms": SEPARATION_RANK,
            "generic_escape_atoms": 4,
            "generic_escape_floor_total": escape_floor,
            "robust_route_initialization": "operator logits 0; generic logits -2; fixed generic floor",
            "mixture_parameterization": "collapsed",
            "split": "fixed random observed mask; every other entry held out; no validation/early stopping",
            "test_use": "metrics only after all 400 updates; never used by optimizer or selection",
        },
        "models": results,
    }


def aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "frozen_protocol": {
            "seeds": list(SEEDS), "steps": STEPS, "observation_ratio": RATIO,
            "rank": TENSOR_RANK, "max_frequency": MAX_FREQUENCY,
            "operator_atoms": SEPARATION_RANK, "generic_atoms": 4,
        },
        "cases": {},
    }
    for case_name in OPERATOR_CASES:
        group = [record for record in records if record["case"] == case_name]
        case_summary: dict[str, Any] = {
            "operator_separation": group[0]["operator_separation"],
            "methods": {},
        }
        for method in group[0]["models"]:
            models = [record["models"][method] for record in group]
            case_summary["methods"][method] = {}
            paths = {
                "test_nrmse": lambda model: model["test_nrmse"],
                "spectrum_cosine": lambda model: model["induced_spectrum"]["cosine"],
                "spectrum_relative_l2": lambda model: model["induced_spectrum"]["relative_l2"],
                "coverage_95": lambda model: model["uq"]["predictive_coverage_95"],
                "predictive_nll": lambda model: model["uq"]["predictive_nll_mc"],
            }
            for metric, getter in paths.items():
                values = [float(getter(model)) for model in models]
                case_summary["methods"][method][metric] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "values": values,
                }
        comparisons = {
            "operator_per_mode_rank_vs_operator_global": ("operator_per_mode_rank", "operator_global"),
            "robust_per_mode_rank_vs_robust_global": ("robust_per_mode_rank", "robust_global"),
            "robust_vs_operator_matched": ("robust_per_mode_rank", "operator_per_mode_rank"),
            "operator_vs_generic_matched": ("operator_per_mode_rank", "generic_per_mode_rank"),
            "robust_escape_vs_wrong_support": ("wrong_support_robust", "wrong_support_operator"),
        }
        case_summary["paired_wins"] = {}
        for label, (left, right) in comparisons.items():
            left_values = [record["models"][left]["test_nrmse"] for record in group]
            right_values = [record["models"][right]["test_nrmse"] for record in group]
            differences = [a - b for a, b in zip(left_values, right_values)]
            case_summary["paired_wins"][label] = {
                "wins": int(sum(value < 0 for value in differences)),
                "total": len(differences),
                "mean_paired_nrmse_difference": float(np.mean(differences)),
            }
        summary["cases"][case_name] = case_summary
    return summary


def render_figure(summary: dict[str, Any], output: Path) -> None:
    selected = [
        "operator_global", "operator_per_mode_rank", "generic_per_mode_rank",
        "generic_global", "oracle_operator_route",
    ]
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharey=False)
    for axis, case_name in zip(axes, OPERATOR_CASES):
        methods = summary["cases"][case_name]["methods"]
        means = [methods[name]["test_nrmse"]["mean"] for name in selected]
        stds = [methods[name]["test_nrmse"]["std"] for name in selected]
        axis.bar(np.arange(len(selected)), means, yerr=stds, capsize=2.5)
        axis.set_title(case_name.replace("_", "\n"))
        axis.set_xticks(np.arange(len(selected)), [name.replace("_", "\n") for name in selected], rotation=45, ha="right", fontsize=7)
        axis.grid(axis="y", alpha=0.25)
    for axis in axes:
        axis.set_ylabel("held-out NRMSE")
    figure.suptitle("Proposed operator prior and primary baselines (2% observations)")
    figure.tight_layout()
    figure.savefig(output / "submission_confirmation_nrmse.png", dpi=190)
    plt.close(figure)

    negative = [
        "operator_per_mode_rank", "robust_per_mode_rank",
        "wrong_support_operator", "wrong_support_robust",
    ]
    figure, axes = plt.subplots(1, 3, figsize=(12.5, 4.4), sharey=True)
    for axis, case_name in zip(axes, OPERATOR_CASES):
        methods = summary["cases"][case_name]["methods"]
        means = [methods[name]["test_nrmse"]["mean"] for name in negative]
        stds = [methods[name]["test_nrmse"]["std"] for name in negative]
        axis.bar(np.arange(len(negative)), means, yerr=stds, capsize=2.5, color="#888888")
        axis.set_title(case_name.replace("_", "\n"))
        axis.set_xticks(np.arange(len(negative)), [name.replace("_", "\n") for name in negative], rotation=40, ha="right", fontsize=7)
        axis.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("held-out NRMSE")
    figure.suptitle("Robustness audit: a fixed generic floor repairs deleted support")
    figure.tight_layout()
    figure.savefig(output / "negative_support_audit.png", dpi=190)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(6.6, 4.2))
    for case_name in OPERATOR_CASES:
        curve = summary["cases"][case_name]["operator_separation"]["relative_error_by_rank"]
        ranks = np.asarray([int(rank) for rank in curve])
        errors = np.asarray([curve[str(rank)] for rank in ranks])
        axis.plot(ranks, errors, marker="o", label=case_name.replace("_", " "))
    axis.set(
        xlabel="nonnegative spectrum rank",
        ylabel="relative joint-spectrum error",
        title="Operator joint-spectrum separability",
    )
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(output / "operator_spectrum_separability.png", dpi=190)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--steps", type=int, default=STEPS)
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "submission_confirmation")
    args = parser.parse_args()
    if args.steps != STEPS:
        raise ValueError(f"submission confirmation is frozen at {STEPS} steps")
    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    prepared = {
        name: build_operator_atoms(config) for name, config in OPERATOR_CASES.items()
    }
    records = []
    for case_name, config in OPERATOR_CASES.items():
        atoms, separation = prepared[case_name]
        for seed in SEEDS:
            record = run_case(case_name, config, atoms, separation, seed, device, args.steps)
            records.append(record)
            path = args.output / f"{case_name}_seed{seed}.json"
            path.write_text(json.dumps(record, indent=2), encoding="utf-8")
            print(f"finished {case_name} seed={seed}", flush=True)
    summary = aggregate(records)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    render_figure(summary, args.output)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
