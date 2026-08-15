#!/usr/bin/env python3
"""Track-3-only audit of intrinsic versus Euclidean kernel-section inputs.

This experiment deliberately tests the mechanism of the current neural POC;
it does not call the models Gaussian processes.  The four configurations form
two parameter-matched pairs and checkpoint by the loss on *all* observed
entries, never by a single random minibatch.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import time

import numpy as np
import torch

from geoaware.domain_kernels import (
    euclidean_rbf_kernel_sections,
    matern_domain_kernel_sections,
)
from geoaware.functional_tucker import DomainKernelFunctionalTucker
from run_irregular_elliptic_paper_b import case_payload, fixed_mask


def all_indices(shape: tuple[int, ...]) -> torch.Tensor:
    return torch.cartesian_prod(*[torch.arange(length) for length in shape])


def load_case(path: Path, modes: int, feature_kind: str) -> dict:
    case = case_payload(path, "correct", modes, seed=0)
    if feature_kind == "intrinsic":
        features = matern_domain_kernel_sections(
            case["basis"], case["eigenvalues"], case["source_nodes"])
    elif feature_kind == "euclidean":
        features = euclidean_rbf_kernel_sections(
            case["coords"], case["source_nodes"])
    else:
        raise ValueError(f"unknown feature kind: {feature_kind}")
    case["domain_kernel_features"] = features
    return case


@torch.no_grad()
def observed_loss(model, batches, center, scale, device) -> float:
    total_squared_error = 0.0
    count = 0
    for case, indices, values in batches:
        prediction = model.forward_case(case, indices.to(device))
        target = ((values - center) / scale).to(device)
        total_squared_error += float((prediction - target).square().sum())
        count += len(values)
    return total_squared_error / max(1, count)


@torch.no_grad()
def predict_case(model, case, device, chunk=65536):
    indices = all_indices(tuple(case["target"].shape))
    output = []
    for start in range(0, len(indices), chunk):
        output.append(model.forward_case(
            case, indices[start:start + chunk].to(device)).cpu())
    return torch.cat(output).reshape(case["target"].shape)


def configurations():
    # Concatenating local features is an input-feature composite, not an
    # additive GP kernel.  Names state this distinction explicitly.
    return {
        "intrinsic_sections_only_tucker": ("intrinsic", False),
        "euclidean_rbf_sections_only_tucker": ("euclidean", False),
        "intrinsic_plus_local_inputs_tucker": ("intrinsic", True),
        "euclidean_rbf_plus_local_inputs_tucker": ("euclidean", True),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path,
                        default=Path("data/irregular_boundary_elliptic"))
    parser.add_argument("--split", type=Path, default=Path(
        "experiments/dataset_splits/irregular_boundary_wave_smoke.json"))
    parser.add_argument("--ratio", type=float, default=.01)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=900)
    parser.add_argument("--checkpoint-every", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--modes", type=int, default=48)
    parser.add_argument("--output", type=Path, default=Path(
        "papers/four_tracks/results/track3_kernel_input_ablation_seed0.json"))
    args = parser.parse_args()

    split = json.loads(args.split.read_text())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = []
    for name, (feature_kind, local_inputs) in configurations().items():
        train = [load_case(args.data / f"{geometry}_r24.npz", args.modes,
                           feature_kind)
                 for geometry in split["train_geometries"]]
        validation = [load_case(args.data / f"{geometry}_r32.npz", args.modes,
                                feature_kind)
                      for geometry in split["validation_geometries"]]
        batches = []
        for case_index, case in enumerate(train):
            mask = fixed_mask(case["target"].shape, args.ratio,
                              args.seed + case_index)
            batches.append((case, torch.from_numpy(np.argwhere(mask)).long(),
                            case["target"][mask]))
        observed = torch.cat([values for _, _, values in batches])
        center = observed.mean()
        scale = observed.std().clamp_min(1e-6)

        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        model = DomainKernelFunctionalTucker(
            kernel_channels=5, ranks=(6, 8, 12), hidden=64,
            composite_local_kernel=local_inputs).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3,
                                      weight_decay=2e-5)
        generator = np.random.default_rng(args.seed)
        best_loss = float("inf")
        best_state = None
        started = time.perf_counter()
        for step in range(1, args.steps + 1):
            case, indices, values = batches[
                int(generator.integers(len(batches)))]
            chosen = torch.from_numpy(generator.integers(
                len(indices), size=args.batch_size)).long()
            prediction = model.forward_case(case, indices[chosen].to(device))
            target = ((values[chosen] - center) / scale).to(device)
            loss = (prediction - target).square().mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step()
            if step % args.checkpoint_every == 0 or step == args.steps:
                model.eval()
                score = observed_loss(model, batches, center, scale, device)
                if score < best_loss:
                    best_loss = score
                    best_state = copy.deepcopy(model.state_dict())
                model.train()
        if best_state is None:
            raise RuntimeError("no checkpoint was evaluated")
        model.load_state_dict(best_state)
        model.eval()

        case_metrics = []
        for case in validation:
            prediction = predict_case(model, case, device) * scale + center
            target = case["target"]
            error = prediction - target
            boundary = case["boundary"][None, None, :].expand_as(target)
            case_metrics.append({
                "case": case["name"],
                "nrmse": float(error.square().mean().sqrt()
                               / target.std().clamp_min(1e-8)),
                "boundary_nrmse": float(error[boundary].square().mean().sqrt()
                                        / target[boundary].std().clamp_min(1e-8)),
            })
        row = {
            "model": name,
            "kernel_section_input": feature_kind,
            "uses_correct_sdf_and_local_coordinates": local_inputs,
            "parameters": sum(p.numel() for p in model.parameters()),
            "best_all_observed_loss": best_loss,
            "validation_macro_nrmse": float(np.mean(
                [metric["nrmse"] for metric in case_metrics])),
            "validation_boundary_nrmse": float(np.mean(
                [metric["boundary_nrmse"] for metric in case_metrics])),
            "case_metrics": case_metrics,
            "elapsed_seconds": time.perf_counter() - started,
        }
        rows.append(row)
        print(f"{name}: nrmse={row['validation_macro_nrmse']:.4f} "
              f"boundary={row['validation_boundary_nrmse']:.4f}", flush=True)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    result = {
        "experiment_id": f"TRACK3-KERNEL-INPUT-ABLATION-SEED{args.seed}",
        "status": "VALIDATION_ONLY_MECHANISM_AUDIT_NOT_GP_INFERENCE",
        "protocol": {
            "train_geometries": split["train_geometries"],
            "validation_geometries": split["validation_geometries"],
            "test_geometries_read": [],
            "train_resolution": 24,
            "validation_resolution": 32,
            "observation_ratio": args.ratio,
            "checkpoint_metric": "MSE over every observed training entry",
        },
        "config": vars(args),
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, default=str),
                           encoding="utf-8")


if __name__ == "__main__":
    main()
