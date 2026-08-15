#!/usr/bin/env python3
"""Track 3 R4: geometry-kernel dictionary selected by mini-batch ELBO.

Three data layers deliberately separate a method-friendly mechanism check from
an approximate operator match and the existing mismatched elliptic simulator.
No validation target participates in normalization, checkpointing, mixture
weight learning, or any optimizer update.  The frozen hole test is not read.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import time
import zlib

import numpy as np
import torch

from geoaware.domain_kernels import (
    euclidean_rbf_kernel_sections,
    geodesic_rbf_kernel_sections,
    heat_domain_kernel_sections,
    matern_domain_kernel_sections,
)
from geoaware.variational_domain_gp import (
    FiniteFeatureVariationalGP,
    NonnegativeKernelMixture,
    tensor_product_gp_features,
)
from run_irregular_elliptic_paper_b import case_payload, fixed_mask
from track3_variational_domain_gp import (
    all_indices,
    summarize_cases,
    validation_metrics,
)


FAMILIES = ("matern_resolvent", "heat_diffusion", "geodesic_rbf", "euclidean_rbf")
SCALES = (0.04, 0.12, 0.36, 1.08, 3.24)


def load_dictionary_case(path: Path, modes: int) -> dict:
    """Load geometry once and construct target-independent kernel sections."""
    case = case_payload(path, "correct", modes, seed=0)
    with np.load(path) as payload:
        edges = torch.from_numpy(payload["undirected_edges"].astype(np.int64))
    case["kernel_sections_by_family"] = {
        "matern_resolvent": matern_domain_kernel_sections(
            case["basis"], case["eigenvalues"], case["source_nodes"],
            scales=SCALES, smoothness=1.5),
        "heat_diffusion": heat_domain_kernel_sections(
            case["basis"], case["eigenvalues"], case["source_nodes"],
            diffusion_times=SCALES),
        "geodesic_rbf": geodesic_rbf_kernel_sections(
            case["coords"], case["source_nodes"], edges,
            lengthscales=(0.08, 0.16, 0.32, 0.64, 1.28)),
        "euclidean_rbf": euclidean_rbf_kernel_sections(
            case["coords"], case["source_nodes"],
            lengthscales=(0.08, 0.16, 0.32, 0.64, 1.28)),
    }
    return case


def family_features(
    case: dict, indices: torch.Tensor, device: torch.device
) -> dict[str, torch.Tensor]:
    return {
        name: tensor_product_gp_features(
            sections.to(device), case["parameters"].to(device), indices.to(device),
            parameter_centers=5, parameter_lengthscale=0.32)
        for name, sections in case["kernel_sections_by_family"].items()
    }


def perturbed_diffusion_sections(case: dict) -> torch.Tensor:
    """A nearby spectral operator absent from the fitted dictionary.

    Its response uses noninteger eigenvalue warping and midway diffusion times;
    it is close to, but not exactly one of, the heat or resolvent channels.
    """
    phi = case["basis"].float()
    lam = case["eigenvalues"].float().clamp_min(0)
    source_phi = phi[case["source_nodes"].long()]
    times = torch.tensor((0.065, 0.20, 0.62, 1.85, 5.4), dtype=phi.dtype)
    warped = lam + 0.08 * lam.pow(0.7)
    filters = torch.exp(-times[:, None] * warped[None])
    filters *= 1 + 0.06 * torch.cos(1.7 * torch.log1p(lam))[None]
    sections = torch.einsum("nk,sk,qk->snq", phi, source_phi, filters)
    sections /= max(1, phi.shape[1])
    rms = sections.square().mean((0, 1), keepdim=True).sqrt().clamp_min(1e-6)
    return sections / rms


def install_synthetic_targets(cases: list[dict], layer: str) -> None:
    """Install a fixed shared-function GP sample on every unseen geometry."""
    generator = torch.Generator().manual_seed(314159 if layer == "matched" else 271828)
    coefficient = torch.randn(25, generator=generator)
    secondary = torch.randn(25, generator=generator)
    for case in cases:
        indices = all_indices(tuple(case["target"].shape))
        if layer == "matched":
            sections = case["kernel_sections_by_family"]["heat_diffusion"]
            phi = tensor_product_gp_features(
                sections, case["parameters"], indices,
                parameter_centers=5, parameter_lengthscale=0.32)
            target = phi @ coefficient
            noise_scale = 0.03
        elif layer == "near_matched":
            primary = tensor_product_gp_features(
                perturbed_diffusion_sections(case), case["parameters"], indices,
                parameter_centers=5, parameter_lengthscale=0.32)
            resolvent = tensor_product_gp_features(
                case["kernel_sections_by_family"]["matern_resolvent"],
                case["parameters"], indices,
                parameter_centers=5, parameter_lengthscale=0.32)
            target = primary @ coefficient + 0.2 * torch.tanh(resolvent @ secondary)
            noise_scale = 0.05
        else:
            raise ValueError(f"unknown synthetic layer: {layer}")
        # Fixed observation noise is part of the generated dataset, not a seed-
        # dependent training augmentation.
        case_seed = zlib.crc32(f"{layer}:{case['name']}".encode("utf-8"))
        noise_generator = torch.Generator().manual_seed(case_seed)
        target = target + noise_scale * target.std().clamp_min(1e-6) * torch.randn(
            target.shape, generator=noise_generator)
        case["target"] = target.reshape(case["target"].shape)


def fit_one(
    model_name: str,
    train_cases: list[dict],
    validation_cases: list[dict],
    *,
    ratio: float,
    seed: int,
    steps: int,
    batch_size: int,
    checkpoint_every: int,
    device: torch.device,
) -> dict:
    observed_by_family = {name: [] for name in FAMILIES}
    observed_targets = []
    observed_counts = {}
    for case_id, case in enumerate(train_cases):
        mask = fixed_mask(case["target"].shape, ratio, seed + case_id)
        indices = torch.from_numpy(np.argwhere(mask)).long()
        observed_counts[case["name"]] = len(indices)
        features = family_features(case, indices, device)
        for family in FAMILIES:
            observed_by_family[family].append(features[family])
        observed_targets.append(case["target"][mask].to(device))
    observed_by_family = {
        family: torch.cat(parts) for family, parts in observed_by_family.items()
    }
    physical_targets = torch.cat(observed_targets)
    center = physical_targets.mean()
    scale = physical_targets.std().clamp_min(1e-6)
    targets = (physical_targets - center) / scale

    mixture = None
    if model_name == "learned_nonnegative_mixture":
        mixture = NonnegativeKernelMixture(FAMILIES).to(device)
        dimension = sum(value.shape[1] for value in observed_by_family.values())
    else:
        dimension = observed_by_family[model_name].shape[1]
    gp = FiniteFeatureVariationalGP(dimension, noise_std=0.15).to(device)
    parameter_groups = [{"params": gp.parameters(), "lr": 2e-2}]
    if mixture is not None:
        parameter_groups.append({"params": mixture.parameters(), "lr": 1e-2})
    optimizer = torch.optim.Adam(parameter_groups)
    generator = torch.Generator().manual_seed(seed)
    best_loss = float("inf")
    best_state = None
    trace = []
    started = time.perf_counter()

    def transform(selected: torch.Tensor | None = None) -> torch.Tensor:
        values = observed_by_family if selected is None else {
            name: features[selected] for name, features in observed_by_family.items()
        }
        return mixture(values) if mixture is not None else values[model_name]

    for step in range(1, steps + 1):
        chosen = torch.randint(
            len(targets), (min(batch_size, len(targets)),), generator=generator,
        ).to(device)
        features = transform(chosen)
        loss, _ = gp.negative_elbo(
            features, targets[chosen], total_count=len(targets))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        parameters = list(gp.parameters())
        if mixture is not None:
            parameters += list(mixture.parameters())
        torch.nn.utils.clip_grad_norm_(parameters, 20.0)
        optimizer.step()
        if step % checkpoint_every == 0 or step == steps:
            with torch.no_grad():
                full_loss, diagnostics = gp.negative_elbo(
                    transform(), targets, total_count=len(targets))
            score = float(full_loss)
            trace.append({
                "step": step,
                "negative_elbo_per_observation": score,
                "kl": float(diagnostics["kl"]),
                "noise_std_normalized": float(diagnostics["noise_std"]),
                "mixture_weights": mixture.weight_dict() if mixture else None,
            })
            if score < best_loss:
                best_loss = score
                best_state = {
                    "gp": copy.deepcopy(gp.state_dict()),
                    "mixture": copy.deepcopy(mixture.state_dict()) if mixture else None,
                }
    if best_state is None:
        raise RuntimeError("no checkpoint evaluated")
    gp.load_state_dict(best_state["gp"])
    gp.eval()
    if mixture is not None:
        mixture.load_state_dict(best_state["mixture"])
        mixture.eval()

    case_metrics = []
    for case in validation_cases:
        indices = all_indices(tuple(case["target"].shape))
        raw = family_features(case, indices, device)
        features = mixture(raw) if mixture is not None else raw[model_name]
        mean, variance = gp.predict(features)
        case_metrics.append(validation_metrics(
            case, mean, variance, center, scale))
    return {
        "model": model_name,
        "inference": "full_covariance_q(u)_mini_batch_ELBO_SGD",
        "prior": "standard_normal_whitened_coefficients",
        "finite_feature_dimension": dimension,
        "observed_entries": len(targets),
        "observed_counts": observed_counts,
        "best_negative_elbo_per_observation": best_loss,
        "learned_noise_std_normalized": float(gp.noise_std.detach()),
        "learned_mixture_weights": mixture.weight_dict() if mixture else None,
        "optimization_trace": trace,
        "case_metrics": case_metrics,
        "elapsed_seconds": time.perf_counter() - started,
        **summarize_cases(case_metrics),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path,
                        default=Path("data/irregular_boundary_elliptic"))
    parser.add_argument("--split", type=Path, default=Path(
        "experiments/dataset_splits/track3_kernel_dictionary.json"))
    parser.add_argument("--layers", nargs="+", default=(
        "matched", "near_matched", "elliptic"))
    parser.add_argument("--ratio", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--checkpoint-every", type=int, default=50)
    parser.add_argument("--modes", type=int, default=48)
    parser.add_argument("--output", type=Path, default=Path(
        "papers/four_tracks/results/track3_kernel_dictionary_seed0.json"))
    args = parser.parse_args()
    if args.steps > 500:
        raise ValueError("early-stage protocol caps optimization at 500 steps")
    split = json.loads(args.split.read_text(encoding="utf-8"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    layer_rows = {}
    for layer in args.layers:
        train_cases = [
            load_dictionary_case(args.data / f"{name}_r24.npz", args.modes)
            for name in split["train_geometries"]
        ]
        validation_cases = [
            load_dictionary_case(args.data / f"{name}_r32.npz", args.modes)
            for name in split["validation_geometries"]
        ]
        if layer in ("matched", "near_matched"):
            install_synthetic_targets(train_cases + validation_cases, layer)
        elif layer != "elliptic":
            raise ValueError(f"unknown layer: {layer}")
        rows = []
        for model_name in (*FAMILIES, "learned_nonnegative_mixture"):
            row = fit_one(
                model_name, train_cases, validation_cases, ratio=args.ratio,
                seed=args.seed, steps=args.steps, batch_size=args.batch_size,
                checkpoint_every=args.checkpoint_every, device=device)
            rows.append(row)
            weights = row["learned_mixture_weights"]
            print(
                f"{layer} {model_name}: NRMSE={row['validation_nrmse']:.4f}"
                + (f" weights={weights}" if weights else ""), flush=True)
        layer_rows[layer] = rows

    result = {
        "experiment_id": f"TRACK3-KERNEL-DICTIONARY-SEED{args.seed}",
        "status": "VALIDATION_ONLY_THREE_LAYER_KERNEL_AUDIT",
        "protocol": {
            "train_geometries": split["train_geometries"],
            "validation_geometries": split["validation_geometries"],
            "test_geometries_read": [],
            "observation_ratio": args.ratio,
            "optimization": "mini-batch ELBO+SGD, train-observed checkpoint only",
            "validation_target_used_for_checkpointing": False,
            "matched_layer": "shared heat finite-GP sample plus fixed 3% noise; method-friendly sanity only",
            "near_matched_layer": "perturbed diffusion response plus 0.2 nonlinear resolvent component and fixed 5% noise",
            "mismatched_layer": "existing screened elliptic PDE simulator",
            "kernel_mixture": "k=sum_q softmax(logit)_q k_q; logits learned jointly by ELBO",
        },
        "config": vars(args),
        "layers": layer_rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
