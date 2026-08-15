"""Reproducible low-observation and geometry-aware missingness protocols."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from .data import FieldDataset


@dataclass
class ObservationSplit:
    observed: torch.Tensor
    held_out: torch.Tensor
    eligible: torch.Tensor
    ratio_requested: float
    ratio_actual: float
    kind: str
    seed: int


def _ravel(indices: torch.Tensor, shape: tuple[int, ...]) -> torch.Tensor:
    stride, out = 1, torch.zeros(len(indices), dtype=torch.long)
    for d in range(len(shape) - 1, -1, -1):
        out += indices[:, d] * stride
        stride *= shape[d]
    return out


def make_observation_split(dataset: FieldDataset, ratio: float, kind: str = "random",
                           seed: int = 0) -> ObservationSplit:
    if not 0 < ratio < 1:
        raise ValueError("ratio must lie in (0, 1)")
    shape, ndim = dataset.shape, len(dataset.shape)
    idx = dataset.flat_indices()
    eligible = torch.ones(len(idx), dtype=torch.bool)
    g = torch.Generator().manual_seed(seed)

    if kind == "random":
        pass
    elif kind == "periodic_gap":
        periodic_dims = [i for i, p in enumerate(dataset.periodic) if p]
        d = periodic_dims[-1] if periodic_dims else ndim - 1
        x = idx[:, d].float() / shape[d]
        # Remove a connected 25% sector centered on the topological seam.
        eligible &= (x >= 0.125) & (x < 0.875)
    elif kind == "block":
        spatial = list(range(max(0, ndim - 2), ndim))
        in_block = torch.ones(len(idx), dtype=torch.bool)
        for d in spatial:
            x = idx[:, d].float() / max(shape[d] - 1, 1)
            in_block &= (x > 0.3) & (x < 0.7)
        eligible &= ~in_block
    elif kind == "sensor_tracks":
        if ndim < 2:
            raise ValueError("sensor_tracks needs at least two dimensions")
        spatial_shape = shape[1:]
        n_spatial = math.prod(spatial_shape)
        n_sensors = max(1, round(ratio * n_spatial))
        sensor_ids = torch.randperm(n_spatial, generator=g)[:n_sensors]
        spatial_idx = idx[:, 1:]
        eligible_ids = _ravel(spatial_idx, spatial_shape)
        observed = torch.isin(eligible_ids, sensor_ids)
        held_out = ~observed
        return ObservationSplit(observed, held_out, torch.ones_like(observed), ratio,
                                float(observed.float().mean()), kind, seed)
    else:
        raise ValueError(f"unknown mask kind: {kind}")

    candidates = torch.where(eligible)[0]
    n_obs = max(1, min(len(candidates), round(ratio * len(idx))))
    chosen = candidates[torch.randperm(len(candidates), generator=g)[:n_obs]]
    observed = torch.zeros(len(idx), dtype=torch.bool)
    observed[chosen] = True
    held_out = ~observed
    return ObservationSplit(observed, held_out, eligible, ratio,
                            float(observed.float().mean()), kind, seed)
