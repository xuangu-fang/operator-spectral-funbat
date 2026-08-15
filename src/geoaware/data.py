"""Synthetic and local public physical-field dataset adapters."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math

import numpy as np
import torch

from .bases import BasisSpec


DEFAULT_ACTIVE_MATTER = Path(
    "/home/ubuntu/project/yanjiu/data/active_matter_multi/benchmark_strict_r48.npz"
)
DEFAULT_REALPDE = Path(
    "/home/ubuntu/project/yanjiu/data/realpde_cylinder_fresh_locked/locked_r64.npz"
)


@dataclass
class FieldDataset:
    name: str
    values: torch.Tensor
    mode_names: tuple[str, ...]
    basis_specs: tuple[BasisSpec, ...]
    periodic: tuple[bool, ...]
    source: str
    description: str

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(self.values.shape)

    def flat_coordinates(self) -> torch.Tensor:
        axes = []
        for n, periodic in zip(self.shape, self.periodic):
            if periodic:
                axes.append(torch.arange(n, dtype=torch.float32) / n)
            else:
                axes.append(torch.linspace(0.0, 1.0, n))
        return torch.stack(torch.meshgrid(*axes, indexing="ij"), -1).reshape(-1, len(axes))

    def flat_indices(self) -> torch.Tensor:
        axes = [torch.arange(n) for n in self.shape]
        return torch.stack(torch.meshgrid(*axes, indexing="ij"), -1).reshape(-1, len(axes))


def synthetic_wave(shape: tuple[int, int, int] = (24, 32, 48)) -> FieldDataset:
    """Low-rank, noisy-looking wave field on interval x interval x circle.

    The range factor obeys a Neumann boundary condition and azimuth is exactly
    periodic.  A weak non-separable chirp prevents the task being a tautological
    rank-3 recovery problem.
    """
    nt, nr, na = shape
    t = torch.linspace(0, 1, nt)[:, None, None]
    r = torch.linspace(0, 1, nr)[None, :, None]
    a = (2 * math.pi * torch.arange(na) / na)[None, None, :]
    field = (
        0.95 * torch.cos(2 * math.pi * t) * torch.cos(math.pi * r) * (1 + 0.35 * torch.cos(a))
        + 0.55 * torch.sin(4 * math.pi * t + 0.15) * torch.cos(3 * math.pi * r) * torch.sin(3 * a)
        + 0.28 * torch.cos(6 * math.pi * t) * torch.cos(6 * math.pi * r) * torch.cos(7 * a)
        + 0.12 * torch.cos(2 * math.pi * (2.0 * t + 2.5 * r) + 2 * a)
    )
    return FieldDataset(
        "synthetic_wave", field.float(), ("time", "range", "azimuth"),
        (BasisSpec("periodic", 8, "time-circle"),
         BasisSpec("neumann", 10, "range-neumann"),
         BasisSpec("periodic", 10, "azimuth-circle")),
        (True, False, True), "generated:geoaware.data.synthetic_wave",
        "Physics-inspired low-rank radial/azimuthal wave with exact product geometry.",
    )


def synthetic_boundary(shape: tuple[int, int] = (64, 64)) -> FieldDataset:
    """Interval x circle sanity check with exact boundary/topology semantics."""
    nx, na = shape
    x = torch.linspace(0, 1, nx)[:, None]
    a = (2 * math.pi * torch.arange(na) / na)[None, :]
    field = (torch.sin(math.pi * x) * (1 + 0.45 * torch.cos(a))
             + 0.45 * torch.sin(3 * math.pi * x) * torch.sin(3 * a)
             + 0.18 * torch.sin(6 * math.pi * x) * torch.cos(7 * a))
    return FieldDataset(
        "synthetic_boundary", field.float(), ("bounded_x", "azimuth"),
        (BasisSpec("dirichlet", 8, "x-dirichlet"),
         BasisSpec("periodic", 10, "azimuth-circle")),
        (False, True), "generated:geoaware.data.synthetic_boundary",
        "Boundary/topology sanity check on a Dirichlet interval times a circle.",
    )


def load_active_matter(path: Path = DEFAULT_ACTIVE_MATTER, record: int = 0,
                       time_stride: int = 2, spatial_stride: int = 1) -> FieldDataset:
    if not path.exists():
        raise FileNotFoundError(f"Active Matter benchmark not found: {path}")
    d = np.load(path, allow_pickle=False)
    fields = d["test_fields"]
    if not 0 <= record < len(fields):
        raise IndexError(f"record {record} outside [0, {len(fields)})")
    values = torch.from_numpy(np.asarray(
        fields[record, ::time_stride, ::spatial_stride, ::spatial_stride], dtype=np.float32
    ).copy())
    return FieldDataset(
        "active_matter", values, ("time", "x", "y"),
        (BasisSpec("neumann", 10, "time-interval"),
         BasisSpec("periodic", 12, "x-periodic"),
         BasisSpec("periodic", 12, "y-periodic")),
        (False, True, True), str(path),
        "The Well Active Matter concentration trajectory; x/y are periodic.",
    )


def load_realpde(path: Path = DEFAULT_REALPDE, record: int = 0, channel: int = 0,
                 n_time: int = 48) -> FieldDataset:
    if not path.exists():
        raise FileNotFoundError(f"RealPDEBench benchmark not found: {path}")
    d = np.load(path, allow_pickle=False)
    raw = d["locked_fields"]
    if not 0 <= record < len(raw):
        raise IndexError(f"record {record} outside [0, {len(raw)})")
    tidx = np.linspace(0, raw.shape[1] - 1, n_time).round().astype(int)
    values = torch.from_numpy(np.asarray(raw[record, tidx, channel], dtype=np.float32).copy())
    return FieldDataset(
        "realpde_cylinder", values, ("time", "height", "streamwise"),
        (BasisSpec("periodic", 10, "vortex-phase"),
         BasisSpec("neumann", 12, "piv-height"),
         BasisSpec("neumann", 16, "piv-streamwise")),
        (True, False, False), str(path),
        "RealPDEBench cylinder-wake PIV velocity channel, temporally subsampled.",
    )


def load_dataset(name: str, **kwargs) -> FieldDataset:
    if name == "synthetic_wave":
        return synthetic_wave(kwargs.pop("shape", (24, 32, 48)))
    if name == "synthetic_boundary":
        return synthetic_boundary(kwargs.pop("shape", (64, 64)))
    if name == "active_matter":
        return load_active_matter(**kwargs)
    if name == "realpde_cylinder":
        return load_realpde(**kwargs)
    raise ValueError(f"unknown dataset: {name}")
