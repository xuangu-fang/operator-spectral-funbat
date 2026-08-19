#!/usr/bin/env python3
"""PDEBench 2D diffusion-reaction loader for sparse tensor completion.

The main experiment is deliberately the simplest thing that can test the claim:
one sample becomes a `[t, x, y]` tensor, a fixed random fraction of entries is
observed, and everything else is held out.  All three modes are PDE coordinates,
so every mode's kernel comes straight from the equation -- no "partial
knowledge" caveat is needed anywhere in the main story.

PDE (FitzHugh-Nagumo reaction-diffusion, PDEBench 2D_diff-react):

    d_t u = Du (d_xx + d_yy) u + Ru(u, v)
    d_t v = Dv (d_xx + d_yy) v + Rv(u, v)

with Du = 1e-3, Dv = 5e-3.  Only the *form* is used by the method; the
coefficient ablation lives elsewhere.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import torch

DEFAULT_PATH = Path(
    "/mnt/data/xuangu-fang/operator-spectral-funbat/pdebench/2D_diff-react_NA_NA.h5"
)
# PDEBench generator constants for this file; used only by the oracle-coefficient
# ablation, never by the headline method.
GENERATOR_DIFFUSIVITY = {"u": 1e-3, "v": 5e-3}


@dataclass(frozen=True)
class CompletionTask:
    """One fully specified sparse-completion instance."""
    field: torch.Tensor            # [t, x, y], standardized
    observed_indices: torch.Tensor  # [n, 3]
    observed_targets: torch.Tensor  # [n]
    test_indices: torch.Tensor      # [m, 3]
    test_targets: torch.Tensor      # [m] clean held-out values
    test_noisy_targets: torch.Tensor
    sample_id: int
    ratio: float
    noise_std: float


def list_samples(path: Path = DEFAULT_PATH) -> list[str]:
    with h5py.File(path, "r") as handle:
        return sorted(handle.keys())


def load_field(
    sample_index: int,
    *,
    path: Path = DEFAULT_PATH,
    component: int = 0,
    time_points: int = 32,
    spatial_stride: int = 4,
) -> torch.Tensor:
    """Return one standardized `[t, x, y]` tensor.

    Sub-sampling keeps the tensor small enough that every baseline can be run
    many times, and keeps the low-rank assumption honest; the strides are frozen
    before any method is compared.
    """
    keys = list_samples(path)
    if not 0 <= sample_index < len(keys):
        raise IndexError(f"sample {sample_index} outside 0..{len(keys) - 1}")
    with h5py.File(path, "r") as handle:
        group = handle[keys[sample_index]]
        data = np.asarray(group["data"])  # [t, x, y, component]
    if data.ndim != 4:
        raise ValueError(f"unexpected PDEBench layout {data.shape}")
    steps = data.shape[0]
    if time_points > steps:
        raise ValueError(f"time_points {time_points} exceeds {steps}")
    time_index = np.linspace(0, steps - 1, time_points).round().astype(int)
    field = data[time_index][:, ::spatial_stride, ::spatial_stride, component]
    tensor = torch.from_numpy(np.ascontiguousarray(field)).float()
    return (tensor - tensor.mean()) / tensor.std().clamp_min(1e-8)


def make_task(
    field: torch.Tensor,
    *,
    ratio: float,
    seed: int,
    noise_std: float = 0.05,
    sample_id: int = -1,
    device: torch.device | str = "cpu",
) -> CompletionTask:
    """Uniform random observation mask; every other entry is held out.

    The same (field, ratio, seed) must produce the identical mask and noise for
    every method, so the split is derived only from these three.
    """
    if not 0 < ratio < 1:
        raise ValueError("ratio must lie in (0,1)")
    field = field.to(device)
    shape = field.shape
    grid = torch.stack(torch.meshgrid(
        *[torch.arange(size, device=field.device) for size in shape], indexing="ij",
    ), dim=-1).reshape(-1, len(shape))
    generator = torch.Generator(device=field.device).manual_seed(seed + 402)
    order = torch.randperm(len(grid), generator=generator, device=field.device)
    count = round(ratio * len(grid))
    if count < 10:
        raise ValueError("observation ratio leaves too few entries")
    observed_indices, test_indices = grid[order[:count]], grid[order[count:]]
    observed_clean = field[tuple(observed_indices.T)]
    test_clean = field[tuple(test_indices.T)]
    observed_targets = observed_clean + noise_std * torch.randn(
        count, generator=generator, device=field.device)
    test_noisy = test_clean + noise_std * torch.randn(
        len(test_indices), generator=generator, device=field.device)
    return CompletionTask(
        field=field, observed_indices=observed_indices, observed_targets=observed_targets,
        test_indices=test_indices, test_targets=test_clean, test_noisy_targets=test_noisy,
        sample_id=sample_id, ratio=ratio, noise_std=noise_std,
    )


def nrmse(prediction: torch.Tensor, truth: torch.Tensor) -> float:
    return float(
        torch.sqrt(torch.mean((prediction - truth).square()))
        / truth.std().clamp_min(1e-8)
    )
