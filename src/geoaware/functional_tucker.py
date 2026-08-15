"""Minimal functional CP/Tucker models shared by research tracks 3 and 4."""

from __future__ import annotations

import math

import torch
from torch import nn


def _mlp(input_dim: int, output_dim: int, hidden: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden), nn.GELU(),
        nn.Linear(hidden, hidden), nn.GELU(),
        nn.Linear(hidden, output_dim),
    )


def sdf_query_features(case: dict, indices: torch.Tensor,
                       *, use_sdf: bool = True) -> torch.Tensor:
    """Build ``(x, boundary distance, source, Euclidean distance, 1)``.

    ``boundary_distance`` is currently stored only on active-domain nodes and
    is nonnegative.  The historical function/flag name is kept for API
    compatibility; this is not a full signed distance field on an ambient grid.
    """
    source, _, node = indices.T
    device = indices.device
    xy = case["coords"].to(device)[node]
    source_xy = case["source_xy"].to(device)[source]
    distance = torch.linalg.vector_norm(xy - source_xy, dim=1, keepdim=True)
    if use_sdf:
        sdf = case["boundary_distance"].to(device)[node, None]
    else:
        sdf = torch.zeros_like(distance)
    return torch.cat([xy, sdf, source_xy, distance, torch.ones_like(distance)], 1)


def scalar_parameter_features(case: dict, indices: torch.Tensor) -> torch.Tensor:
    """Use a positive scalar physical parameter and its logarithm."""
    _, parameter, _ = indices.T
    value = case["parameters"].to(indices.device)[parameter]
    return torch.stack([torch.log(value.clamp_min(1e-8)), value], 1)


class FunctionalTucker(nn.Module):
    """Three functional mode maps contracted through an explicit small core."""

    def __init__(self, spatial_dim: int, ranks=(6, 8, 16), hidden: int = 64,
                 core_init: str = "random"):
        super().__init__()
        self.ranks = tuple(int(rank) for rank in ranks)
        rg, rp, rs = self.ranks
        self.geometry_factor = _mlp(7, rg, hidden)
        self.parameter_factor = _mlp(2, rp, hidden)
        self.spatial_factor = _mlp(spatial_dim, rs, hidden)
        if core_init == "random":
            core = torch.randn(rg, rp, rs) / math.sqrt(rg * rp * rs)
        elif core_init == "cp_diagonal":
            core = torch.zeros(rg, rp, rs)
            diagonal_rank = min(rg, rp, rs)
            index = torch.arange(diagonal_rank)
            core[index, index, index] = 1 / math.sqrt(diagonal_rank)
        else:
            raise ValueError(f"unknown core initialization: {core_init}")
        self.core = nn.Parameter(core)

    def forward_features(self, geometry: torch.Tensor, parameter: torch.Tensor,
                         spatial: torch.Tensor) -> torch.Tensor:
        gf = self.geometry_factor(geometry)
        pf = self.parameter_factor(parameter)
        sf = self.spatial_factor(spatial)
        return torch.einsum("ng,np,ns,gps->n", gf, pf, sf, self.core)


class GeometryConditionedNeuralFunctionalTucker(FunctionalTucker):
    """Track 4: boundary-distance-conditioned factors and a Tucker core."""

    def __init__(self, ranks=(6, 8, 16), hidden: int = 64, use_sdf: bool = True,
                 core_init: str = "random"):
        super().__init__(spatial_dim=7, ranks=ranks, hidden=hidden,
                         core_init=core_init)
        self.use_sdf = bool(use_sdf)

    def forward_case(self, case: dict, indices: torch.Tensor) -> torch.Tensor:
        geometry = case["descriptor"].to(indices.device)[None].expand(len(indices), -1)
        parameter = scalar_parameter_features(case, indices)
        spatial = sdf_query_features(case, indices, use_sdf=self.use_sdf)
        return self.forward_features(geometry, parameter, spatial)


class GeometryConditionedNeuralFunctionalCP(nn.Module):
    """CP restriction of track 4, used as the method-matched baseline."""

    def __init__(self, rank: int = 24, hidden: int = 64, use_sdf: bool = True):
        super().__init__()
        self.use_sdf = bool(use_sdf)
        self.geometry_factor = _mlp(7, rank, hidden)
        self.parameter_factor = _mlp(2, rank, hidden)
        self.spatial_factor = _mlp(7, rank, hidden)
        self.weight = nn.Parameter(torch.ones(rank) / math.sqrt(rank))

    def forward_case(self, case: dict, indices: torch.Tensor) -> torch.Tensor:
        geometry = case["descriptor"].to(indices.device)[None].expand(len(indices), -1)
        parameter = scalar_parameter_features(case, indices)
        spatial = sdf_query_features(case, indices, use_sdf=self.use_sdf)
        return (self.geometry_factor(geometry) * self.parameter_factor(parameter)
                * self.spatial_factor(spatial) * self.weight).sum(1)


class DomainKernelFunctionalTucker(FunctionalTucker):
    """Track 3 POC: domain-kernel-section inputs and a Tucker core.

    The sections are inspired by a finite GP covariance, but the current MLP
    and AdamW fit have no explicit Gaussian prior or posterior.  This class is
    therefore a deterministic feature POC, not GP or GP-MAP inference.
    """

    def __init__(self, kernel_channels: int = 5, ranks=(6, 8, 12),
                 hidden: int = 64, composite_local_kernel: bool = True):
        # Concatenating local coordinates/boundary distance with kernel sections
        # is an input-feature composite, not an additive or product GP kernel.
        self.composite_local_kernel = bool(composite_local_kernel)
        spatial_dim = kernel_channels + (7 if composite_local_kernel else 0)
        super().__init__(spatial_dim=spatial_dim, ranks=ranks, hidden=hidden)

    def forward_case(self, case: dict, indices: torch.Tensor) -> torch.Tensor:
        source, _, node = indices.T
        geometry = case["descriptor"].to(indices.device)[None].expand(len(indices), -1)
        parameter = scalar_parameter_features(case, indices)
        spatial = case["domain_kernel_features"].to(indices.device)[source, node]
        if self.composite_local_kernel:
            spatial = torch.cat([spatial, sdf_query_features(case, indices)], 1)
        return self.forward_features(geometry, parameter, spatial)
