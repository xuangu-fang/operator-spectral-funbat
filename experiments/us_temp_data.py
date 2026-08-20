#!/usr/bin/env python3
"""US surface-temperature tensor with an operator basis for its spatial modes.

Real data, real sparsity and a published baseline: this is FunBaT's US-TEMP
benchmark, a [latitude, longitude, year] tensor of Berkeley Earth surface
temperature, 15 x 95 x 267, of which only 5.2% of entries exist at all.  It
ships official five-fold splits, so the comparison is on someone else's split
rather than one we chose.

The operator is not invented for the occasion.  Surface temperature diffuses on
a sphere, and in latitude/longitude coordinates the Laplace-Beltrami operator is

    lap T = (1/cos p) d_p (cos p d_p T) + (1/cos^2 p) d_ll T,

so the latitude mode carries a genuinely variable-coefficient operator -- the
cos(latitude) metric weight -- while longitude carries a plain second
difference.  Its eigenbasis is separable across the two spatial modes, which is
what this method needs.

The time mode is left to a generic kernel, and that is deliberate.  267 annual
means are trend plus interannual variability, not diffusive relaxation, so the
physics does not supply a defensible temporal kernel here.  Claiming one would
be the kind of over-reach the rest of this work is set up to avoid.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

DEFAULT_PATH = Path("/tmp/claude-1000/-home-ubuntu-project-operator-spectral-funbat/"
                    "d4172fb2-0143-45aa-9510-be797dcda49b/scratchpad/funbat/data/US_temp/"
                    "DISCT_15x95x267.npy")
CONTINUOUS_PATH = DEFAULT_PATH.with_name("CONTI_15x95x267.npy")


@dataclass(frozen=True)
class TemperatureFold:
    dims: tuple[int, int, int]
    train_index: torch.Tensor
    train_value: torch.Tensor
    test_index: torch.Tensor
    test_value: torch.Tensor
    latitude: torch.Tensor      # degrees, ascending
    longitude: torch.Tensor
    mean: float
    scale: float


def load_fold(fold: int = 0, path: Path = DEFAULT_PATH,
              continuous: Path = CONTINUOUS_PATH) -> TemperatureFold:
    raw = np.load(path, allow_pickle=True).item()
    block = raw["data"][fold]
    dims = tuple(int(d) for d in raw["ndims"])
    coordinates = np.load(continuous, allow_pickle=True).item()["NORMAL_2_RAW_dicts"]
    latitude = torch.tensor([v for _, v in sorted(coordinates[0].items())],
                            dtype=torch.float64)
    longitude = torch.tensor([v for _, v in sorted(coordinates[1].items())],
                             dtype=torch.float64)
    train_value = torch.as_tensor(np.asarray(block["tr_y"], dtype=np.float64))
    mean, scale = float(train_value.mean()), float(train_value.std())
    return TemperatureFold(
        dims=dims,
        train_index=torch.as_tensor(np.asarray(block["tr_ind"], dtype=np.int64)),
        train_value=((train_value - mean) / scale).float(),
        test_index=torch.as_tensor(np.asarray(block["te_ind"], dtype=np.int64)),
        test_value=((torch.as_tensor(np.asarray(block["te_y"], dtype=np.float64))
                     - mean) / scale).float(),
        latitude=latitude, longitude=longitude, mean=mean, scale=scale,
    )


def _neumann_weighted_operator(nodes: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Symmetric finite-volume ``-(1/w) d(w d/dx)`` on an irregular grid.

    Edge conductances use the midpoint weight and the actual node spacing, so an
    irregular latitude grid is handled without pretending it is uniform.  The
    result is positive semi-definite and preserves the constant mode, as a
    zero-flux operator must.
    """
    size = len(nodes)
    spacing = (nodes[1:] - nodes[:-1]).clamp_min(1e-9)
    midpoint_weight = 0.5 * (weight[1:] + weight[:-1])
    conductance = midpoint_weight / spacing
    operator = torch.zeros(size, size, dtype=torch.float64)
    edge = torch.arange(size - 1)
    operator[edge, edge] += conductance
    operator[edge + 1, edge + 1] += conductance
    operator[edge, edge + 1] -= conductance
    operator[edge + 1, edge] -= conductance
    # Divide rows by the cell weight to represent (1/w) d(w d/dx) rather than
    # d(w d/dx); symmetrise afterwards so eigh stays valid.
    cell = weight.clamp_min(1e-9)
    operator = operator / cell[:, None]
    return 0.5 * (operator + operator.T)


def spatial_operator_basis(fold: TemperatureFold, modes: int
                           ) -> tuple[torch.Tensor, torch.Tensor]:
    """Eigenvectors of the spherical Laplacian's latitude and longitude parts."""
    radians = torch.deg2rad(fold.latitude)
    cosine = torch.cos(radians).clamp_min(1e-6)
    latitude_operator = _neumann_weighted_operator(radians, cosine)
    longitude_operator = _neumann_weighted_operator(
        torch.deg2rad(fold.longitude), torch.ones_like(fold.longitude))
    _, latitude_vectors = torch.linalg.eigh(latitude_operator)
    _, longitude_vectors = torch.linalg.eigh(longitude_operator)
    return (latitude_vectors[:, :modes].float().contiguous(),
            longitude_vectors[:, :modes].float().contiguous())


def spatial_operator_spectrum(fold: TemperatureFold, modes: int, decay: float = 1.0
                              ) -> tuple[torch.Tensor, torch.Tensor]:
    """Prior weight per eigenmode, ``(1 + lambda_q)^-decay``.

    A diffusive field's energy falls off with the operator's eigenvalue; the
    exponent is a nominal choice, not fitted, in keeping with using only the
    equation's form.
    """
    radians = torch.deg2rad(fold.latitude)
    cosine = torch.cos(radians).clamp_min(1e-6)
    latitude_values, _ = torch.linalg.eigh(_neumann_weighted_operator(radians, cosine))
    longitude_values, _ = torch.linalg.eigh(_neumann_weighted_operator(
        torch.deg2rad(fold.longitude), torch.ones_like(fold.longitude)))

    def weight(values: torch.Tensor) -> torch.Tensor:
        rates = values[:modes] / values[1].clamp_min(1e-12)
        out = (1 + rates.clamp_min(0)).pow(-decay)
        return (out / out.sum()).float()

    return weight(latitude_values), weight(longitude_values)
