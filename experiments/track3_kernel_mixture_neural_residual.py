#!/usr/bin/env python3
"""Track 3 R4: neural tensor mean plus ELBO-selected geometry-GP residual."""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path

import numpy as np
import torch

from geoaware.functional_tucker import GeometryConditionedNeuralFunctionalCP
from geoaware.variational_domain_gp import (
    FiniteFeatureVariationalGP,
    NonnegativeKernelMixture,
)
from run_irregular_elliptic_paper_b import fixed_mask
from track3_geometry_kernel_dictionary import (
    FAMILIES,
    family_features,
    load_dictionary_case,
)
from track3_neural_mean_gp_residual import (
    batched_neural_mean,
    full_neural_mean,
    normalized_gaussian_nll,
)
from track3_variational_domain_gp import (
    all_indices,
    summarize_cases,
    validation_metrics,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path,
                        default=Path("data/irregular_boundary_elliptic"))
    parser.add_argument("--split", type=Path, default=Path(
        "experiments/dataset_splits/track3_kernel_dictionary.json"))
    parser.add_argument("--ratio", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--checkpoint-every", type=int, default=50)
    parser.add_argument("--modes", type=int, default=48)
    parser.add_argument("--output", type=Path, default=Path(
        "papers/four_tracks/results/track3_kernel_mixture_neural_residual_seed0.json"))
    args = parser.parse_args()
    if args.steps > 500:
        raise ValueError("early-stage protocol caps optimization at 500 steps")
    split = json.loads(args.split.read_text(encoding="utf-8"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_cases = [
        load_dictionary_case(args.data / f"{name}_r24.npz", args.modes)
        for name in split["train_geometries"]
    ]
    validation_cases = [
        load_dictionary_case(args.data / f"{name}_r32.npz", args.modes)
        for name in split["validation_geometries"]
    ]

    observed_case_ids = []
    observed_indices = []
    observed_by_family = {name: [] for name in FAMILIES}
    observed_targets = []
    observed_counts = {}
    for case_id, case in enumerate(train_cases):
        mask = fixed_mask(case["target"].shape, args.ratio, args.seed + case_id)
        indices = torch.from_numpy(np.argwhere(mask)).long()
        features = family_features(case, indices, device)
        observed_counts[case["name"]] = len(indices)
        observed_case_ids.append(torch.full((len(indices),), case_id))
        observed_indices.append(indices)
        for family in FAMILIES:
            observed_by_family[family].append(features[family])
        observed_targets.append(case["target"][mask].to(device))
    case_ids = torch.cat(observed_case_ids).long()
    indices = torch.cat(observed_indices).long()
    observed_by_family = {
        family: torch.cat(parts) for family, parts in observed_by_family.items()
    }
    physical_targets = torch.cat(observed_targets)
    center = physical_targets.mean()
    scale = physical_targets.std().clamp_min(1e-6)
    targets = (physical_targets - center) / scale

    configurations = (
        ("shared_neural_cp_mean_only", None),
        ("shared_neural_cp_plus_matern_gp", "matern_resolvent"),
        ("shared_neural_cp_plus_heat_gp", "heat_diffusion"),
        ("shared_neural_cp_plus_learned_kernel_gp", "mixture"),
    )
    rows = []
    for model_name, residual_kind in configurations:
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        generator = torch.Generator().manual_seed(args.seed)
        mean_model = GeometryConditionedNeuralFunctionalCP(
            rank=24, hidden=64, use_sdf=True).to(device)
        mixture = (NonnegativeKernelMixture(FAMILIES).to(device)
                   if residual_kind == "mixture" else None)
        if residual_kind is None:
            gp = None
            log_noise_std = torch.nn.Parameter(
                torch.tensor(math.log(0.15), device=device))
            groups = [
                {"params": mean_model.parameters(), "lr": 2e-3},
                {"params": [log_noise_std], "lr": 2e-2},
            ]
        else:
            dimension = (sum(value.shape[1] for value in observed_by_family.values())
                         if mixture else observed_by_family[residual_kind].shape[1])
            gp = FiniteFeatureVariationalGP(dimension, noise_std=0.15).to(device)
            groups = [
                {"params": mean_model.parameters(), "lr": 2e-3},
                {"params": gp.parameters(), "lr": 2e-2},
            ]
            if mixture:
                groups.append({"params": mixture.parameters(), "lr": 1e-2})
        optimizer = torch.optim.Adam(groups)

        def residual_features(selected: torch.Tensor | None = None) -> torch.Tensor:
            raw = observed_by_family if selected is None else {
                name: value[selected] for name, value in observed_by_family.items()
            }
            return mixture(raw) if mixture else raw[residual_kind]

        best_loss = float("inf")
        best_state = None
        trace = []
        for step in range(1, args.steps + 1):
            chosen = torch.randint(
                len(indices), (min(args.batch_size, len(indices)),),
                generator=generator)
            neural_mean = batched_neural_mean(
                mean_model, train_cases, case_ids[chosen], indices[chosen], device)
            target = targets[chosen.to(device)]
            if gp is None:
                noise_std = log_noise_std.clamp(math.log(1e-3), math.log(2)).exp()
                loss = normalized_gaussian_nll(neural_mean, target, noise_std)
            else:
                loss, _ = gp.negative_elbo(
                    residual_features(chosen.to(device)), target,
                    total_count=len(indices), mean_offset=neural_mean)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for group in optimizer.param_groups for p in group["params"]], 20.0)
            optimizer.step()
            if step % args.checkpoint_every == 0 or step == args.steps:
                mean_model.eval()
                with torch.no_grad():
                    full_mean = batched_neural_mean(
                        mean_model, train_cases, case_ids, indices, device)
                    if gp is None:
                        noise_std = log_noise_std.clamp(
                            math.log(1e-3), math.log(2)).exp()
                        full_loss = normalized_gaussian_nll(
                            full_mean, targets, noise_std)
                        kl = 0.0
                    else:
                        full_loss, diagnostics = gp.negative_elbo(
                            residual_features(), targets, total_count=len(indices),
                            mean_offset=full_mean)
                        noise_std = gp.noise_std
                        kl = float(diagnostics["kl"])
                score = float(full_loss)
                trace.append({
                    "step": step,
                    "negative_objective_per_observation": score,
                    "kl": kl,
                    "noise_std_normalized": float(noise_std.detach()),
                    "mixture_weights": mixture.weight_dict() if mixture else None,
                })
                if score < best_loss:
                    best_loss = score
                    best_state = {
                        "mean": copy.deepcopy(mean_model.state_dict()),
                        "gp": copy.deepcopy(gp.state_dict()) if gp else None,
                        "mixture": copy.deepcopy(mixture.state_dict()) if mixture else None,
                        "log_noise_std": float(log_noise_std.detach()) if gp is None else None,
                    }
                mean_model.train()
        if best_state is None:
            raise RuntimeError("no checkpoint evaluated")
        mean_model.load_state_dict(best_state["mean"])
        mean_model.eval()
        if gp:
            gp.load_state_dict(best_state["gp"])
            gp.eval()
        else:
            log_noise_std.data.fill_(best_state["log_noise_std"])
        if mixture:
            mixture.load_state_dict(best_state["mixture"])
            mixture.eval()

        case_metrics = []
        for case in validation_cases:
            full_indices = all_indices(tuple(case["target"].shape))
            mean = full_neural_mean(mean_model, case, device)
            if gp is None:
                noise = log_noise_std.clamp(math.log(1e-3), math.log(2)).exp()
                variance = torch.full_like(mean, float(noise.detach().square()))
            else:
                raw = family_features(case, full_indices, device)
                features = mixture(raw) if mixture else raw[residual_kind]
                residual, variance = gp.predict(features)
                mean = mean + residual
            case_metrics.append(validation_metrics(
                case, mean, variance, center, scale))
        row = {
            "model": model_name,
            "mean_architecture": "GeometryConditionedNeuralFunctionalCP_rank24_hidden64",
            "residual_kernel": residual_kind,
            "inference": ("joint_neural_mean_and_full_cov_q(u)_mini_batch_ELBO_SGD"
                          if gp else "Gaussian_neural_mean_maximum_likelihood_SGD"),
            "observed_entries": len(indices),
            "best_negative_objective_per_observation": best_loss,
            "learned_mixture_weights": mixture.weight_dict() if mixture else None,
            "optimization_trace": trace,
            "case_metrics": case_metrics,
            **summarize_cases(case_metrics),
        }
        rows.append(row)
        print(
            f"{model_name}: NRMSE={row['validation_nrmse']:.4f} "
            f"boundary={row['validation_boundary_nrmse']:.4f} "
            f"weights={row['learned_mixture_weights']}", flush=True)

    result = {
        "experiment_id": f"TRACK3-KERNEL-MIXTURE-NEURAL-RESIDUAL-SEED{args.seed}",
        "status": "VALIDATION_ONLY_JOINT_ELBO_R4",
        "protocol": {
            "train_geometries": split["train_geometries"],
            "validation_geometries": split["validation_geometries"],
            "test_geometries_read": [],
            "observation_ratio": args.ratio,
            "validation_target_used_for_checkpointing": False,
            "matched_initialization_masks_batches_steps_and_mean_architecture": True,
            "kernel_mixture": "simplex PSD sum, weights learned jointly by train-observed ELBO",
        },
        "config": vars(args),
        "observed_counts": observed_counts,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
