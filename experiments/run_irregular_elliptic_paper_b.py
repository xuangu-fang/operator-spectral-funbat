#!/usr/bin/env python3
"""Minimal operator-spectral functional CP across unseen irregular domains."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn

from run_irregular_boundary_paper_a import graph_basis, rectangle_basis


def mlp(input_dim: int, output_dim: int, hidden: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(input_dim, hidden), nn.GELU(),
                         nn.Linear(hidden, hidden), nn.GELU(),
                         nn.Linear(hidden, output_dim))


def fixed_mask(shape, ratio, seed):
    generator = np.random.default_rng(seed)
    mask = np.zeros(int(np.prod(shape)), dtype=bool)
    mask[generator.choice(len(mask), max(1, int(round(ratio*len(mask)))), replace=False)] = True
    return mask.reshape(shape)


def geometry_descriptor(payload) -> torch.Tensor:
    coords = payload["coordinates"].astype(np.float32)
    boundary = payload["boundary_distance"].astype(np.float32)
    fluid_fraction = float(payload["fluid_mask"].mean())
    values = [fluid_fraction, float(coords[:, 0].mean()), float(coords[:, 1].mean()),
              float(coords[:, 0].std()), float(coords[:, 1].std()),
              float(np.quantile(boundary, .25)), float(np.quantile(boundary, .75))]
    return torch.tensor(values, dtype=torch.float32)


def case_payload(path: Path, representation: str, modes: int, seed: int):
    payload = np.load(path)
    if representation == "bbox":
        basis, eigenvalues = rectangle_basis(payload["coordinates"], modes)
    else:
        basis, eigenvalues = graph_basis(payload, modes)
        # Continuous empirical-L2 normalization makes the feature scale stable
        # between the 24 and 32 background meshes.
        basis = basis*math.sqrt(len(basis))
        if representation == "wrong":
            permutation = torch.randperm(len(basis), generator=torch.Generator().manual_seed(seed))
            basis = basis[permutation]
    source_nodes = torch.from_numpy(payload["source_nodes"].astype(np.int64))
    coords = torch.from_numpy(payload["coordinates"].astype(np.float32))
    boundary = torch.from_numpy(payload["boundary_distance"].astype(np.float32))
    boundary /= torch.quantile(boundary, .95).clamp_min(1e-8)
    source_xy = torch.from_numpy(payload["source_xy"].astype(np.float32))
    return {
        "name": path.stem,
        "target": torch.from_numpy(payload["field"].astype(np.float32)),
        "parameters": torch.from_numpy(payload["diffusivities"].astype(np.float32)),
        "basis": basis,
        "eigenvalues": eigenvalues,
        "source_nodes": source_nodes,
        "coords": coords,
        "boundary_distance": boundary,
        "source_xy": source_xy,
        "descriptor": geometry_descriptor(payload),
        "boundary": torch.from_numpy(payload["boundary_mask"].astype(bool)[
            tuple(payload["grid_indices"].T)]),
    }


class OperatorSpectralFunctionalCP(nn.Module):
    """Explicit CP with a sign-invariant operator-spectral spatial factor."""
    def __init__(self, rank=24, hidden=64):
        super().__init__(); self.rank = rank
        self.spectral_filter = mlp(2, rank, hidden)
        self.geometry_factor = mlp(7, rank, hidden)
        self.parameter_factor = mlp(2, rank, hidden)
        self.weight = nn.Parameter(torch.ones(rank)/math.sqrt(rank))

    def forward_case(self, case, indices):
        source, parameter, node = indices.T
        device = self.weight.device
        basis = case["basis"].to(device)
        eigenvalues = case["eigenvalues"].to(device)
        lam = torch.stack([torch.log1p(eigenvalues),
                           torch.sqrt(eigenvalues.clamp_min(0))/(1+torch.sqrt(eigenvalues.clamp_min(0)))], 1)
        transfer = self.spectral_filter(lam)
        # phi(x) phi(s) is invariant to arbitrary eigenvector sign flips.
        source_projection = basis[case["source_nodes"].to(device)]
        spatial_table = torch.einsum("nk,sk,kr->snr", basis, source_projection, transfer)/len(eigenvalues)
        spatial = spatial_table[source, node]
        descriptor = case["descriptor"].to(device)[None].expand(len(indices), -1)
        parameter_value = case["parameters"].to(device)[parameter]
        parameter_feature = torch.stack([torch.log(parameter_value), parameter_value], 1)
        return (self.geometry_factor(descriptor)*self.parameter_factor(parameter_feature)
                *spatial*self.weight).sum(1)


class SDFCoordinateFunctionalCP(nn.Module):
    """Coordinates/SDF-only CP with the same rank and separated contraction."""
    def __init__(self, rank=24, hidden=64):
        super().__init__(); self.rank = rank
        self.geometry_factor = mlp(7, rank, hidden)
        self.parameter_factor = mlp(2, rank, hidden)
        self.space_factor = mlp(7, rank, hidden)
        self.weight = nn.Parameter(torch.ones(rank)/math.sqrt(rank))

    def forward_case(self, case, indices):
        source, parameter, node = indices.T
        device = self.weight.device
        descriptor = case["descriptor"].to(device)[None].expand(len(indices), -1)
        parameter_value = case["parameters"].to(device)[parameter]
        parameter_feature = torch.stack([torch.log(parameter_value), parameter_value], 1)
        xy = case["coords"].to(device)[node]
        source_xy = case["source_xy"].to(device)[source]
        distance = torch.linalg.vector_norm(xy-source_xy, dim=1, keepdim=True)
        spatial_feature = torch.cat([
            xy, case["boundary_distance"].to(device)[node, None],
            source_xy, distance, torch.ones_like(distance)], 1)
        return (self.geometry_factor(descriptor)*self.parameter_factor(parameter_feature)
                *self.space_factor(spatial_feature)*self.weight).sum(1)


def evaluate(model, case, device, chunk=65536):
    shape = case["target"].shape
    indices = torch.cartesian_prod(*[torch.arange(n) for n in shape])
    predictions = []
    with torch.no_grad():
        for start in range(0, len(indices), chunk):
            predictions.append(model.forward_case(case, indices[start:start+chunk].to(device)).cpu())
    return torch.cat(predictions).reshape(shape)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/irregular_boundary_elliptic"))
    parser.add_argument("--split", type=Path,
                        default=Path("experiments/dataset_splits/irregular_boundary_wave_smoke.json"))
    parser.add_argument("--ratio", type=float, default=.01)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=700)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--modes", type=int, default=32)
    parser.add_argument("--output", type=Path,
                        default=Path("papers/longterm_results/irregular_elliptic_paper_b_1pct.json"))
    args = parser.parse_args()
    split = json.loads(args.split.read_text())
    model_specs = {
        "operator_spectral_cp": (OperatorSpectralFunctionalCP, "correct"),
        "wrong_boundary_spectral_cp": (OperatorSpectralFunctionalCP, "wrong"),
        "topology_erased_spectral_cp": (OperatorSpectralFunctionalCP, "bbox"),
        "sdf_coordinate_cp": (SDFCoordinateFunctionalCP, "correct"),
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = []
    for model_name, (factory, representation) in model_specs.items():
        train = [case_payload(args.data/f"{name}_r24.npz", representation,
                              args.modes, 3000+args.seed+index)
                 for index, name in enumerate(split["train_geometries"])]
        validation = [case_payload(args.data/f"{name}_r32.npz", representation,
                                   args.modes, 4000+args.seed+index)
                      for index, name in enumerate(split["validation_geometries"])]
        batches = []
        for case_index, case in enumerate(train):
            mask = fixed_mask(case["target"].shape, args.ratio, args.seed+case_index)
            batches.append((case, torch.from_numpy(np.argwhere(mask)).long(), case["target"][mask]))
        normalizer = torch.std(torch.cat([values for _, _, values in batches])).clamp_min(1e-6)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
        model = factory().to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-6)
        generator = np.random.default_rng(args.seed)
        started = time.perf_counter(); model.train()
        for _ in range(args.steps):
            case, indices, values = batches[int(generator.integers(len(batches)))]
            chosen = torch.from_numpy(generator.integers(len(indices), size=args.batch_size)).long()
            query = indices[chosen].to(device); target = values[chosen].to(device)
            loss = ((model.forward_case(case, query)-target)/normalizer.to(device)).square().mean()
            optimizer.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.); optimizer.step()
        model.eval(); case_metrics = []
        for case in validation:
            prediction = evaluate(model, case, device); target = case["target"]
            error = prediction-target
            boundary = case["boundary"][None, None, :].expand_as(target)
            case_metrics.append({
                "case": case["name"],
                "nrmse": float(error.square().mean().sqrt()/target.std().clamp_min(1e-8)),
                "boundary_nrmse": float(error[boundary].square().mean().sqrt()
                                        / target[boundary].std().clamp_min(1e-8)),
            })
        row = {
            "model": model_name,
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "validation_macro_nrmse": float(np.mean([x["nrmse"] for x in case_metrics])),
            "validation_boundary_nrmse": float(np.mean([x["boundary_nrmse"] for x in case_metrics])),
            "case_metrics": case_metrics,
            "elapsed_seconds": time.perf_counter()-started,
        }
        rows.append(row)
        print(f"{model_name}: validation={row['validation_macro_nrmse']:.4f} "
              f"boundary={row['validation_boundary_nrmse']:.4f}", flush=True)
        del model
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    result = {
        "experiment_id": f"B-IRREGULAR-ELLIPTIC-{100*args.ratio:.0f}BP-SEED{args.seed}",
        "status": "VALIDATION_GATE_TEST_UNREAD",
        "config": vars(args),
        "train_geometries": split["train_geometries"],
        "validation_geometries": split["validation_geometries"],
        "test_geometries_read": [],
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
