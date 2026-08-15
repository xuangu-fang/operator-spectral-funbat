#!/usr/bin/env python3
"""Track 3: explicit domain-kernel variational GP with mini-batch ELBO.

The intrinsic and Euclidean models differ only in their fixed kernel sections.
Both use the same tensor-product parameter features, full-covariance q(u),
Gaussian likelihood, optimizer budget, masks, and exact finite-GP control.
Validation fields are never used for checkpointing or hyperparameter fitting.
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

from geoaware.domain_kernels import (
    euclidean_rbf_kernel_sections,
    matern_domain_kernel_sections,
)
from geoaware.variational_domain_gp import (
    FiniteFeatureVariationalGP,
    exact_finite_gp_posterior,
    exact_finite_gp_predict,
    tensor_product_gp_features,
)
from run_irregular_elliptic_paper_b import case_payload, fixed_mask


def all_indices(shape: tuple[int, ...]) -> torch.Tensor:
    return torch.cartesian_prod(*[torch.arange(length) for length in shape])


def load_case(path: Path, modes: int, feature_kind: str) -> dict:
    case = case_payload(path, "correct", modes, seed=0)
    if feature_kind == "intrinsic":
        sections = matern_domain_kernel_sections(
            case["basis"], case["eigenvalues"], case["source_nodes"])
    elif feature_kind == "euclidean":
        sections = euclidean_rbf_kernel_sections(
            case["coords"], case["source_nodes"])
    else:
        raise ValueError(f"unknown feature kind: {feature_kind}")
    case["kernel_sections"] = sections
    return case


def case_features(case: dict, indices: torch.Tensor, device: torch.device) -> torch.Tensor:
    return tensor_product_gp_features(
        case["kernel_sections"].to(device),
        case["parameters"].to(device),
        indices.to(device),
    )


def validation_metrics(
    case: dict,
    mean: torch.Tensor,
    variance: torch.Tensor,
    center: torch.Tensor,
    scale: torch.Tensor,
) -> dict:
    shape = case["target"].shape
    target = case["target"].flatten().to(mean.device)
    physical_mean = mean * scale + center
    physical_variance = variance * scale.square()
    error = physical_mean - target
    boundary = case["boundary"][None, None, :].expand(shape).flatten().to(mean.device)
    target_std = target.std().clamp_min(1e-8)
    boundary_std = target[boundary].std().clamp_min(1e-8)
    standard_deviation = physical_variance.sqrt().clamp_min(1e-8)
    z = error.abs() / standard_deviation
    nll = 0.5 * (
        math.log(2 * math.pi) + torch.log(physical_variance.clamp_min(1e-12))
        + error.square() / physical_variance.clamp_min(1e-12)
    )
    error_np = error.abs().detach().cpu().numpy()
    std_np = standard_deviation.detach().cpu().numpy()
    std_spread = float(np.ptp(std_np))
    meaningful_spread = max(1e-8, 1e-5 * abs(float(np.mean(std_np))))
    if np.std(error_np) > 1e-12 and std_spread > meaningful_spread:
        uncertainty_error_correlation = float(np.corrcoef(error_np, std_np)[0, 1])
    else:
        uncertainty_error_correlation = 0.0
    return {
        "case": case["name"],
        "nrmse": float(error.square().mean().sqrt() / target_std),
        "boundary_nrmse": float(
            error[boundary].square().mean().sqrt() / boundary_std),
        "predictive_nll": float(nll.mean()),
        "coverage_90": float((z <= 1.6448536).float().mean()),
        "coverage_95": float((z <= 1.959964).float().mean()),
        "mean_predictive_std": float(standard_deviation.mean()),
        "uncertainty_abs_error_correlation": uncertainty_error_correlation,
    }


def summarize_cases(case_metrics: list[dict]) -> dict:
    keys = (
        "nrmse", "boundary_nrmse", "predictive_nll", "coverage_90",
        "coverage_95", "mean_predictive_std",
        "uncertainty_abs_error_correlation",
    )
    return {f"validation_{key}": float(np.mean([row[key] for row in case_metrics]))
            for key in keys}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path,
                        default=Path("data/irregular_boundary_elliptic"))
    parser.add_argument("--split", type=Path, default=Path(
        "experiments/dataset_splits/irregular_boundary_wave_smoke.json"))
    parser.add_argument("--ratio", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--checkpoint-every", type=int, default=25)
    parser.add_argument("--modes", type=int, default=48)
    parser.add_argument("--learning-rate", type=float, default=2e-2)
    parser.add_argument("--output", type=Path, default=Path(
        "papers/four_tracks/results/track3_variational_gp_seed0.json"))
    args = parser.parse_args()
    if not 0 < args.ratio <= 1:
        raise ValueError("ratio must be in (0,1]")

    split = json.loads(args.split.read_text(encoding="utf-8"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    rows = []
    for feature_kind in ("intrinsic", "euclidean"):
        # Reuse the identical stochastic mini-batch sequence in both controls.
        generator = torch.Generator().manual_seed(args.seed)
        train_cases = [
            load_case(args.data / f"{name}_r24.npz", args.modes, feature_kind)
            for name in split["train_geometries"]
        ]
        validation_cases = [
            load_case(args.data / f"{name}_r32.npz", args.modes, feature_kind)
            for name in split["validation_geometries"]
        ]
        observed_features = []
        observed_targets = []
        observed_counts = {}
        for case_index, case in enumerate(train_cases):
            mask = fixed_mask(case["target"].shape, args.ratio,
                              args.seed + case_index)
            indices = torch.from_numpy(np.argwhere(mask)).long()
            observed_counts[case["name"]] = len(indices)
            observed_features.append(case_features(case, indices, device))
            observed_targets.append(case["target"][mask].to(device))
        physical_targets = torch.cat(observed_targets)
        center = physical_targets.mean()
        scale = physical_targets.std().clamp_min(1e-6)
        targets = (physical_targets - center) / scale
        features = torch.cat(observed_features)

        model = FiniteFeatureVariationalGP(features.shape[1], noise_std=0.15).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
        best_loss = float("inf")
        best_state = None
        trace = []
        started = time.perf_counter()
        for step in range(1, args.steps + 1):
            chosen = torch.randint(
                len(features), (min(args.batch_size, len(features)),),
                generator=generator,
            ).to(device)
            loss, _ = model.negative_elbo(
                features[chosen], targets[chosen], total_count=len(features))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 20.0)
            optimizer.step()
            if step % args.checkpoint_every == 0 or step == args.steps:
                with torch.no_grad():
                    full_loss, diagnostics = model.negative_elbo(
                        features, targets, total_count=len(features))
                score = float(full_loss)
                trace.append({
                    "step": step,
                    "negative_elbo_per_observation": score,
                    "kl": float(diagnostics["kl"]),
                    "noise_std_normalized": float(diagnostics["noise_std"]),
                })
                if score < best_loss:
                    best_loss = score
                    best_state = copy.deepcopy(model.state_dict())
        if best_state is None:
            raise RuntimeError("no variational checkpoint evaluated")
        model.load_state_dict(best_state)
        model.eval()

        exact_mean, exact_covariance = exact_finite_gp_posterior(
            features, targets, model.noise_std)
        variational_metrics = []
        exact_metrics = []
        inference_comparison = []
        for case in validation_cases:
            indices = all_indices(tuple(case["target"].shape))
            phi = case_features(case, indices, device)
            variational_mean, variational_variance = model.predict(phi)
            exact_prediction, exact_variance = exact_finite_gp_predict(
                phi, exact_mean, exact_covariance, noise_std=model.noise_std)
            variational_metrics.append(validation_metrics(
                case, variational_mean, variational_variance, center, scale))
            exact_metrics.append(validation_metrics(
                case, exact_prediction, exact_variance, center, scale))
            inference_comparison.append({
                "case": case["name"],
                "variational_exact_mean_rmse_normalized": float(
                    (variational_mean - exact_prediction).square().mean().sqrt()),
                "variational_exact_variance_relative_l1": float(
                    (variational_variance - exact_variance).abs().mean()
                    / exact_variance.mean().clamp_min(1e-12)),
            })

        common = {
            "kernel": feature_kind,
            "finite_feature_dimension": features.shape[1],
            "observed_entries": len(features),
            "learned_noise_std_normalized": float(model.noise_std.detach()),
        }
        rows.append({
            "model": f"{feature_kind}_finite_feature_variational_gp",
            "inference": "full_covariance_q(u)_mini_batch_ELBO_SGD",
            **common,
            "best_negative_elbo_per_observation": best_loss,
            "case_metrics": variational_metrics,
            "inference_comparison": inference_comparison,
            "optimization_trace": trace,
            "elapsed_seconds": time.perf_counter() - started,
            **summarize_cases(variational_metrics),
        })
        rows.append({
            "model": f"{feature_kind}_exact_finite_gp_control",
            "inference": "closed_form_exact_posterior_same_kernel_and_noise",
            **common,
            "case_metrics": exact_metrics,
            **summarize_cases(exact_metrics),
        })
        print(
            f"{feature_kind}: variational NRMSE={rows[-2]['validation_nrmse']:.4f}, "
            f"exact={rows[-1]['validation_nrmse']:.4f}, "
            f"coverage95={rows[-2]['validation_coverage_95']:.3f}",
            flush=True,
        )

    result = {
        "experiment_id": f"TRACK3-VARIATIONAL-DOMAIN-GP-SEED{args.seed}",
        "status": "VALIDATION_ONLY_EXPLICIT_GP_ELBO_POC",
        "protocol": {
            "train_geometries": split["train_geometries"],
            "validation_geometries": split["validation_geometries"],
            "test_geometries_read": [],
            "train_resolution": 24,
            "validation_resolution": 32,
            "observation_ratio": args.ratio,
            "mask": "entry_random_train_only",
            "validation_target_used_for_checkpointing": False,
            "checkpoint_metric": "full_observed_training_negative_ELBO",
            "posterior": "q(u)=N(m,LL^T), full covariance",
            "prior": "p(u)=N(0,I), whitened finite spectral coefficients",
            "likelihood": "Gaussian with learned homoscedastic noise",
            "objective": "mini-batch SGD ELBO with N/B likelihood scaling",
        },
        "config": vars(args),
        "observed_counts": observed_counts,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
