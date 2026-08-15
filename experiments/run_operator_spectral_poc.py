#!/usr/bin/env python3
"""Five-round POC for mode-adaptive operator-spectral FunBaT.

All model comparisons within a seed share the planted field, noise realization,
and observation mask.  Hyperparameters and step counts are fixed in advance;
the unobserved target entries are never used for model selection or stopping.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys

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


def _routing(indices: list[list[int]], families: int, device: torch.device) -> torch.Tensor:
    weights = torch.zeros(3, len(indices[0]), families, device=device)
    for mode, row in enumerate(indices):
        for rank, family in enumerate(row):
            weights[mode, rank, family] = 1
    return weights


def _operator_dictionary(operator: str, components: int, max_frequency: int) -> tuple[torch.Tensor, dict]:
    frequency = torch.arange(max_frequency + 1, dtype=torch.float32)
    joint = operator_joint_spectrum(operator, frequency)
    separated = nonnegative_cp_spectrum(joint, rank=components, steps=1200, seed=17)
    spectra = torch.stack(separated.factors)
    return normalize_spectrum(spectra), {
        "operator": operator,
        "separation_rank": components,
        "relative_spectrum_error": separated.relative_error,
        "component_weights": separated.weights.tolist(),
    }


def _train(
    *,
    coordinates: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    spectra: torch.Tensor,
    field: torch.Tensor,
    observed_indices: torch.Tensor,
    observed_targets: torch.Tensor,
    test_indices: torch.Tensor,
    routing: str,
    fixed_routing: torch.Tensor | None,
    rank: int,
    seed: int,
    steps: int,
) -> dict:
    torch.manual_seed(seed + 1000)
    model = ModeAdaptiveVariationalCP(
        coordinates, spectra, rank=rank, routing=routing,
        fixed_routing=fixed_routing, noise_std=0.08,
    ).to(field.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.025)
    loss_trace = []
    for step in range(steps):
        if routing == "hierarchical":
            # First quarter is a genuine global-dictionary warm start.  The
            # total optimization budget remains identical across methods.
            model.mode_deviation.requires_grad_(step >= steps // 4)
        optimizer.zero_grad(set_to_none=True)
        # Full observed batch: ratios are <=5%, so this is faster and removes
        # an avoidable source of routing noise while remaining ELBO+SGD.
        loss, diagnostics = model.negative_elbo(
            observed_indices, observed_targets,
            total_count=len(observed_targets), samples=3,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        optimizer.step()
        if step in {0, steps // 4, steps // 2, 3 * steps // 4, steps - 1}:
            loss_trace.append({
                "step": step + 1,
                "negative_elbo_per_observation": float(loss.detach()),
                "kl": float(diagnostics["kl"]),
                "noise_std": float(diagnostics["noise_std"]),
            })
    with torch.no_grad():
        prediction = model.posterior_mean(test_indices)
        truth = field[tuple(test_indices.T)]
        nrmse = torch.sqrt(torch.mean((prediction - truth).square())) / truth.std().clamp_min(1e-8)
        observed_prediction = model.posterior_mean(observed_indices)
        observed_truth = field[tuple(observed_indices.T)]
        observed_nrmse = torch.sqrt(torch.mean((observed_prediction - observed_truth).square())) / observed_truth.std().clamp_min(1e-8)
    return {
        "test_nrmse": float(nrmse),
        "observed_clean_nrmse": float(observed_nrmse),
        "routing_weights": model.routing_weights().detach().cpu().tolist(),
        "noise_std": float(model.noise_std.detach()),
        "loss_trace": loss_trace,
    }


def run_planted(
    seed: int, ratio: float, steps: int, device: torch.device,
    only_models: set[str] | None = None,
) -> dict:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    family_names, base = generic_spectral_dictionary(6)
    spectra = base[None].expand(3, -1, -1).clone().to(device)
    coordinate = tuple(torch.arange(24, device=device) / 24 for _ in range(3))
    oracle = _routing([[0, 1], [2, 0], [1, 2]], len(family_names), device)
    swapped = _routing([[2, 0], [1, 2], [0, 1]], len(family_names), device)
    field = sample_planted_tensor(coordinate, spectra, oracle, seed=seed + 31)
    all_indices = all_grid_indices(tuple(field.shape), device)
    generator = torch.Generator(device=device).manual_seed(seed + 51)
    order = torch.randperm(len(all_indices), generator=generator, device=device)
    observed_count = max(12, round(ratio * len(all_indices)))
    observed_indices = all_indices[order[:observed_count]]
    test_indices = all_indices[order[observed_count:]]
    noise = 0.05 * torch.randn(observed_count, generator=generator, device=device)
    observed_targets = field[tuple(observed_indices.T)] + noise

    configurations = {
        "global_dictionary": ("global", None),
        "per_mode_routing": ("per_mode", None),
        "per_mode_rank_routing": ("per_mode_rank", None),
        "hierarchical_mode_routing": ("hierarchical", None),
        "oracle_routing": ("fixed", oracle),
        "swapped_routing": ("fixed", swapped),
    }
    models = {}
    for name, (routing, fixed) in configurations.items():
        if only_models is not None and name not in only_models:
            continue
        models[name] = _train(
            coordinates=coordinate, spectra=spectra, field=field,
            observed_indices=observed_indices, observed_targets=observed_targets,
            test_indices=test_indices, routing=routing, fixed_routing=fixed,
            rank=2, seed=seed, steps=steps,
        )
    if "per_mode_rank_routing" in models:
        learned = torch.tensor(models["per_mode_rank_routing"]["routing_weights"])
        models["per_mode_rank_routing"]["route_top1_accuracy"] = float(
            (learned.argmax(-1).cpu() == oracle.cpu().argmax(-1)).float().mean()
        )
    return {
        "rounds": ["R1_planted_identifiability", "R2_routing_controls"],
        "seed": seed,
        "observation_ratio": ratio,
        "observed_count": observed_count,
        "grid_shape": list(field.shape),
        "noise_std": 0.05,
        "family_names": list(family_names),
        "oracle_routes": oracle.argmax(-1).cpu().tolist(),
        "protocol": "fixed 400-step budget; no validation/early stopping; all unobserved entries are test",
        "models": models,
    }


def run_operator_case(
    *,
    seed: int,
    truth_operator: str,
    prior_operator: str,
    ratio: float,
    steps: int,
    device: torch.device,
    only_models: set[str] | None = None,
) -> dict:
    operator_spectra, operator_meta = _operator_dictionary(prior_operator, 4, 6)
    truth_spectra, truth_meta = _operator_dictionary(truth_operator, 4, 6)
    _, generic = generic_spectral_dictionary(6)
    operator_spectra = operator_spectra.to(device)
    truth_spectra = truth_spectra.to(device)
    generic = generic.to(device)
    coordinate = tuple(torch.arange(24, device=device) / 24 for _ in range(3))
    truth_route = _routing([[0, 1], [1, 2], [2, 3]], 4, device)
    field = sample_planted_tensor(coordinate, truth_spectra, truth_route, seed=seed + 401)
    all_indices = all_grid_indices(tuple(field.shape), device)
    generator = torch.Generator(device=device).manual_seed(seed + 402)
    order = torch.randperm(len(all_indices), generator=generator, device=device)
    observed_count = max(12, round(ratio * len(all_indices)))
    observed_indices = all_indices[order[:observed_count]]
    test_indices = all_indices[order[observed_count:]]
    observed_targets = field[tuple(observed_indices.T)] + 0.05 * torch.randn(
        observed_count, generator=generator, device=device
    )

    hybrid = torch.cat((operator_spectra, generic[None].expand(3, -1, -1)), dim=1)
    wrong_support = operator_spectra.clone()
    wrong_support[..., 2:] = 0
    wrong_support = normalize_spectrum(wrong_support)
    wrong_support_hybrid = torch.cat(
        (wrong_support, generic[None].expand(3, -1, -1)), dim=1
    )
    model_specs = {
        "operator_per_mode_rank": (operator_spectra, "per_mode_rank"),
        "operator_global": (operator_spectra, "global"),
        "generic_per_mode_rank": (generic[None].expand(3, -1, -1), "per_mode_rank"),
        "hybrid_per_mode_rank": (hybrid, "per_mode_rank"),
        "hybrid_hierarchical": (hybrid, "hierarchical"),
        "hybrid_global": (hybrid, "global"),
        "wrong_support_operator": (wrong_support, "per_mode_rank"),
        "wrong_support_hybrid": (wrong_support_hybrid, "per_mode_rank"),
    }
    models = {}
    for name, (spectra, routing) in model_specs.items():
        if only_models is not None and name not in only_models:
            continue
        models[name] = _train(
            coordinates=coordinate, spectra=spectra, field=field,
            observed_indices=observed_indices, observed_targets=observed_targets,
            test_indices=test_indices, routing=routing, fixed_routing=None,
            rank=2, seed=seed + 13, steps=steps,
        )
    return {
        "round": "R4_operator_dictionary_bridge" if truth_operator == prior_operator else "R5_operator_mismatch",
        "seed": seed,
        "truth_operator": truth_operator,
        "prior_operator": prior_operator,
        "observation_ratio": ratio,
        "observed_count": observed_count,
        "truth_separation": truth_meta,
        "prior_separation": operator_meta,
        "models": models,
    }


def run_separation(output: Path) -> dict:
    frequency = torch.arange(7, dtype=torch.float32)
    operators = ("diffusion", "wave", "advection")
    errors = {}
    for operator in operators:
        joint = operator_joint_spectrum(operator, frequency)
        errors[operator] = {}
        for rank in range(1, 7):
            result = nonnegative_cp_spectrum(joint, rank=rank, steps=1200, seed=17)
            errors[operator][str(rank)] = result.relative_error

    figure, axis = plt.subplots(figsize=(6.4, 4.0))
    for operator in operators:
        axis.plot(range(1, 7), list(errors[operator].values()), marker="o", label=operator)
    axis.set(xlabel="nonnegative separation rank", ylabel="relative spectrum error",
             title="Operator joint-spectrum separability")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output / "r3_operator_spectrum_separation.png", dpi=180)
    plt.close(figure)
    return {"round": "R3_operator_joint_spectrum_separation", "frequency_bins": 7, "errors": errors}


def _summarize(records: list[dict], r3: dict) -> dict:
    planted = [record for record in records if "models" in record and "truth_operator" not in record]
    operator = [record for record in records if record.get("round") == "R4_operator_dictionary_bridge"]
    mismatch = [record for record in records if record.get("round") == "R5_operator_mismatch"]

    def aggregate(group: list[dict]) -> dict:
        keys = group[0]["models"] if group else []
        result = {}
        for key in keys:
            values = [record["models"][key]["test_nrmse"] for record in group]
            result[key] = {"mean": float(np.mean(values)), "std": float(np.std(values)), "values": values}
        return result

    planted_by_ratio = {}
    for ratio in sorted({record["observation_ratio"] for record in planted}):
        subset = [record for record in planted if record["observation_ratio"] == ratio]
        planted_by_ratio[str(ratio)] = aggregate(subset)
        planted_by_ratio[str(ratio)]["route_top1_accuracy"] = {
            "mean": float(np.mean([r["models"]["per_mode_rank_routing"]["route_top1_accuracy"] for r in subset]))
        }
    return {
        "protocol": {"seeds": [0, 1, 2], "steps": 400, "observation_ratios": [0.01, 0.02, 0.05]},
        "r1_r2_planted": planted_by_ratio,
        "r3_separation": r3,
        "r4_matched_operator_2pct": aggregate(operator),
        "r5_mismatched_operator_2pct": aggregate(mismatch),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "advanced_poc_r1_r5")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    r3 = run_separation(args.output)
    (args.output / "r3_separation.json").write_text(json.dumps(r3, indent=2), encoding="utf-8")

    records = []
    for seed in (0, 1, 2):
        for ratio in (0.01, 0.02, 0.05):
            record = run_planted(seed, ratio, args.steps, device)
            records.append(record)
            path = args.output / f"r1_r2_seed{seed}_ratio{int(100*ratio):02d}.json"
            path.write_text(json.dumps(record, indent=2), encoding="utf-8")
        for truth_operator, prior_operator in (("advection", "advection"), ("advection", "diffusion")):
            record = run_operator_case(
                seed=seed, truth_operator=truth_operator, prior_operator=prior_operator,
                ratio=0.02, steps=args.steps, device=device,
            )
            records.append(record)
            label = "matched" if truth_operator == prior_operator else "mismatch"
            (args.output / f"r4_r5_{label}_seed{seed}.json").write_text(
                json.dumps(record, indent=2), encoding="utf-8"
            )
    summary = _summarize(records, r3)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
