#!/usr/bin/env python3
"""Track 3 R3: shared neural mean plus variational domain-GP residual.

All learned models use the same geometry-conditioned neural CP mean.  The two
Bayesian variants differ only in the fixed intrinsic or Euclidean residual
feature map.  Mean and q(u) are jointly optimized by a mini-batch ELBO.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

from geoaware.functional_tucker import GeometryConditionedNeuralFunctionalCP
from geoaware.variational_domain_gp import FiniteFeatureVariationalGP
from run_irregular_elliptic_paper_b import fixed_mask
from track3_variational_domain_gp import (
    all_indices,
    case_features,
    load_case,
    summarize_cases,
    validation_metrics,
)


def batched_neural_mean(
    model: GeometryConditionedNeuralFunctionalCP,
    cases: list[dict],
    case_ids: torch.Tensor,
    indices: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Evaluate mixed-domain entries while retaining autograd order."""
    positions = []
    predictions = []
    for case_id in torch.unique(case_ids, sorted=True).tolist():
        selected = torch.nonzero(case_ids == case_id, as_tuple=False).squeeze(1)
        positions.append(selected.to(device))
        predictions.append(model.forward_case(
            cases[case_id], indices[selected].to(device)))
    joined_positions = torch.cat(positions)
    joined_predictions = torch.cat(predictions)
    return joined_predictions[torch.argsort(joined_positions)]


@torch.no_grad()
def full_neural_mean(model, case, device, chunk=65536):
    indices = all_indices(tuple(case["target"].shape))
    return torch.cat([
        model.forward_case(case, indices[start:start + chunk].to(device))
        for start in range(0, len(indices), chunk)
    ])


def normalized_gaussian_nll(
    prediction: torch.Tensor, target: torch.Tensor, noise_std: torch.Tensor
) -> torch.Tensor:
    variance = noise_std.square()
    return 0.5 * (
        math.log(2 * math.pi) + torch.log(variance)
        + (target - prediction).square() / variance
    ).mean()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path,
                        default=Path("data/irregular_boundary_elliptic"))
    parser.add_argument("--split", type=Path, default=Path(
        "experiments/dataset_splits/irregular_boundary_wave_smoke.json"))
    parser.add_argument("--ratio", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--checkpoint-every", type=int, default=25)
    parser.add_argument("--modes", type=int, default=48)
    parser.add_argument("--mean-learning-rate", type=float, default=2e-3)
    parser.add_argument("--gp-learning-rate", type=float, default=2e-2)
    parser.add_argument("--output", type=Path, default=Path(
        "papers/four_tracks/results/track3_neural_mean_gp_residual_seed0.json"))
    args = parser.parse_args()

    split = json.loads(args.split.read_text(encoding="utf-8"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    configurations = (
        ("shared_neural_mean_only", "intrinsic", False),
        ("shared_neural_mean_plus_intrinsic_gp", "intrinsic", True),
        ("shared_neural_mean_plus_euclidean_gp", "euclidean", True),
    )
    rows = []
    trivial_row = None

    for model_name, feature_kind, use_gp in configurations:
        train_cases = [
            load_case(args.data / f"{name}_r24.npz", args.modes, feature_kind)
            for name in split["train_geometries"]
        ]
        validation_cases = [
            load_case(args.data / f"{name}_r32.npz", args.modes, feature_kind)
            for name in split["validation_geometries"]
        ]
        observed_case_ids = []
        observed_indices = []
        observed_features = []
        observed_targets = []
        observed_counts = {}
        for case_id, case in enumerate(train_cases):
            mask = fixed_mask(case["target"].shape, args.ratio,
                              args.seed + case_id)
            indices = torch.from_numpy(np.argwhere(mask)).long()
            observed_counts[case["name"]] = len(indices)
            observed_case_ids.append(torch.full((len(indices),), case_id))
            observed_indices.append(indices)
            observed_features.append(case_features(case, indices, device))
            observed_targets.append(case["target"][mask].to(device))
        case_ids = torch.cat(observed_case_ids).long()
        indices = torch.cat(observed_indices).long()
        features = torch.cat(observed_features)
        physical_targets = torch.cat(observed_targets)
        center = physical_targets.mean()
        scale = physical_targets.std().clamp_min(1e-6)
        targets = (physical_targets - center) / scale

        if trivial_row is None:
            trivial_metrics = []
            for case in validation_cases:
                count = int(np.prod(case["target"].shape))
                mean = torch.zeros(count, device=device)
                variance = torch.ones(count, device=device)
                trivial_metrics.append(validation_metrics(
                    case, mean, variance, center, scale))
            trivial_row = {
                "model": "train_observed_global_mean",
                "inference": "no_fit_train_observations_only",
                "case_metrics": trivial_metrics,
                **summarize_cases(trivial_metrics),
            }

        # Reset both initialization and mini-batch order for every method.
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        generator = torch.Generator().manual_seed(args.seed)
        mean_model = GeometryConditionedNeuralFunctionalCP(
            rank=24, hidden=64, use_sdf=True).to(device)
        gp = (FiniteFeatureVariationalGP(features.shape[1], noise_std=0.15).to(device)
              if use_gp else None)
        if gp is None:
            log_noise_std = torch.nn.Parameter(
                torch.tensor(math.log(0.15), device=device))
            optimizer = torch.optim.Adam([
                {"params": mean_model.parameters(), "lr": args.mean_learning_rate},
                {"params": [log_noise_std], "lr": args.gp_learning_rate},
            ])
        else:
            optimizer = torch.optim.Adam([
                {"params": mean_model.parameters(), "lr": args.mean_learning_rate},
                {"params": gp.parameters(), "lr": args.gp_learning_rate},
            ])

        best_loss = float("inf")
        best_state = None
        trace = []
        started = time.perf_counter()
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
                diagnostics = {"kl": torch.tensor(0.0), "noise_std": noise_std.detach()}
            else:
                loss, diagnostics = gp.negative_elbo(
                    features[chosen.to(device)], target,
                    total_count=len(indices), mean_offset=neural_mean)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            parameters = list(mean_model.parameters())
            if gp is not None:
                parameters += list(gp.parameters())
            torch.nn.utils.clip_grad_norm_(parameters, 20.0)
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
                        full_loss, full_diagnostics = gp.negative_elbo(
                            features, targets, total_count=len(indices),
                            mean_offset=full_mean)
                        noise_std = gp.noise_std
                        kl = float(full_diagnostics["kl"])
                score = float(full_loss)
                trace.append({
                    "step": step,
                    "negative_objective_per_observation": score,
                    "kl": kl,
                    "noise_std_normalized": float(noise_std.detach()),
                })
                if score < best_loss:
                    best_loss = score
                    best_state = {
                        "mean": copy.deepcopy(mean_model.state_dict()),
                        "gp": copy.deepcopy(gp.state_dict()) if gp is not None else None,
                        "log_noise_std": (float(log_noise_std.detach())
                                          if gp is None else None),
                    }
                mean_model.train()
        if best_state is None:
            raise RuntimeError("no checkpoint evaluated")
        mean_model.load_state_dict(best_state["mean"])
        mean_model.eval()
        if gp is not None:
            gp.load_state_dict(best_state["gp"])
            gp.eval()
            final_noise = float(gp.noise_std.detach())
        else:
            log_noise_std.data.fill_(best_state["log_noise_std"])
            final_noise = float(log_noise_std.detach().exp())

        case_metrics = []
        for case in validation_cases:
            full_indices = all_indices(tuple(case["target"].shape))
            neural_mean = full_neural_mean(mean_model, case, device)
            if gp is None:
                predictive_mean = neural_mean
                predictive_variance = torch.full_like(neural_mean, final_noise ** 2)
            else:
                residual_mean, predictive_variance = gp.predict(
                    case_features(case, full_indices, device))
                predictive_mean = neural_mean + residual_mean
            case_metrics.append(validation_metrics(
                case, predictive_mean, predictive_variance, center, scale))
        row = {
            "model": model_name,
            "mean_architecture": "GeometryConditionedNeuralFunctionalCP_rank24_hidden64",
            "residual_kernel": feature_kind if use_gp else None,
            "inference": ("joint_neural_mean_and_full_cov_q(u)_mini_batch_ELBO_SGD"
                          if use_gp else "Gaussian_neural_mean_maximum_likelihood_SGD"),
            "observed_entries": len(indices),
            "finite_feature_dimension": features.shape[1] if use_gp else 0,
            "best_negative_objective_per_observation": best_loss,
            "learned_noise_std_normalized": final_noise,
            "case_metrics": case_metrics,
            "optimization_trace": trace,
            "elapsed_seconds": time.perf_counter() - started,
            **summarize_cases(case_metrics),
        }
        rows.append(row)
        print(
            f"{model_name}: NRMSE={row['validation_nrmse']:.4f}, "
            f"boundary={row['validation_boundary_nrmse']:.4f}, "
            f"coverage95={row['validation_coverage_95']:.3f}", flush=True)

    result = {
        "experiment_id": f"TRACK3-NEURAL-MEAN-GP-RESIDUAL-SEED{args.seed}",
        "status": "VALIDATION_ONLY_JOINT_ELBO_R3",
        "protocol": {
            "train_geometries": split["train_geometries"],
            "validation_geometries": split["validation_geometries"],
            "test_geometries_read": [],
            "train_resolution": 24,
            "validation_resolution": 32,
            "observation_ratio": args.ratio,
            "validation_target_used_for_checkpointing": False,
            "shared_mean_architecture_and_initialization": True,
            "matched_masks_batch_sequence_steps_and_learning_rates": True,
            "joint_residual_objective": "E_q Gaussian log likelihood - KL[q(u)||p(u)]",
        },
        "config": vars(args),
        "observed_counts": observed_counts,
        "trivial_baseline": trivial_row,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
