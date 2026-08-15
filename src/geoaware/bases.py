"""Analytic eigenbases for one-dimensional factors of product domains.

The basis values are L2-normalized in their continuous domains.  Eigenvalues
are rescaled by the first non-zero eigenvalue before being consumed by a prior
or regularizer, which keeps hyperparameters comparable across domain lengths.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch


@dataclass(frozen=True)
class BasisSpec:
    kind: str
    n_frequencies: int
    name: str = ""

    @property
    def size(self) -> int:
        if self.kind == "periodic":
            return 1 + 2 * self.n_frequencies
        if self.kind == "neumann":
            return 1 + self.n_frequencies
        if self.kind == "dirichlet":
            return self.n_frequencies
        if self.kind == "raw_fourier":
            return 2 * self.n_frequencies
        raise ValueError(f"unknown basis kind: {self.kind}")


def evaluate_basis(x: torch.Tensor, spec: BasisSpec) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(Phi(x), normalized_eigenvalues)`` for x in [0, 1]."""
    x = x.reshape(-1, 1)
    dtype, device = x.dtype, x.device
    if spec.kind == "periodic":
        k = torch.arange(1, spec.n_frequencies + 1, device=device, dtype=dtype)
        phase = 2.0 * math.pi * x * k
        phi = torch.cat(
            [torch.ones_like(x), math.sqrt(2.0) * torch.cos(phase),
             math.sqrt(2.0) * torch.sin(phase)], dim=-1
        )
        eig = torch.cat([torch.zeros(1, device=device, dtype=dtype), k.square(), k.square()])
    elif spec.kind == "neumann":
        k = torch.arange(1, spec.n_frequencies + 1, device=device, dtype=dtype)
        phi = torch.cat([torch.ones_like(x), math.sqrt(2.0) * torch.cos(math.pi * x * k)], -1)
        eig = torch.cat([torch.zeros(1, device=device, dtype=dtype), k.square()])
    elif spec.kind == "dirichlet":
        k = torch.arange(1, spec.n_frequencies + 1, device=device, dtype=dtype)
        phi = math.sqrt(2.0) * torch.sin(math.pi * x * k)
        eig = k.square()
    elif spec.kind == "raw_fourier":
        # Deliberately geometry-agnostic feature baseline: non-integer frequencies
        # do not enforce equality at x=0 and x=1.
        k = torch.linspace(0.5, spec.n_frequencies - 0.5, spec.n_frequencies,
                           device=device, dtype=dtype)
        phase = 2.0 * math.pi * x * k
        phi = torch.cat([torch.cos(phase), torch.sin(phase)], -1)
        eig = torch.cat([k.square(), k.square()])
    else:
        raise ValueError(f"unknown basis kind: {spec.kind}")
    return phi, eig


def basis_on_grid(size: int, spec: BasisSpec, *, device: torch.device | str = "cpu"):
    x = torch.linspace(0.0, 1.0, size, device=device)
    # Periodic grids must not duplicate the endpoint.
    if spec.kind == "periodic":
        x = torch.arange(size, device=device, dtype=torch.float32) / size
    return evaluate_basis(x, spec)
