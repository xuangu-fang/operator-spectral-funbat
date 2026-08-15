"""Method-matched functional baselines for operator-informed Tucker experiments.

These models deliberately keep the same separated CP/Tucker decoder as the
proposed method while replacing operator-spectral factors by small neural
functions.  Periodic modes receive a seam-preserving ``(sin, cos)`` encoding;
bounded modes receive the raw normalized coordinate.  They are deterministic
reconstruction baselines and must not be used for uncertainty comparisons.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import nn


class _ModeFactor(nn.Module):
    def __init__(self, rank: int, hidden: int, periodic: bool):
        super().__init__()
        self.periodic = bool(periodic)
        input_dim = 2 if self.periodic else 1
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, rank),
        )

    def forward(self, coordinate: torch.Tensor) -> torch.Tensor:
        x = coordinate.reshape(-1, 1)
        if self.periodic:
            x = torch.cat([torch.sin(2 * math.pi * x),
                           torch.cos(2 * math.pi * x)], dim=1)
        else:
            x = 2 * x - 1
        return self.net(x)


class NeuralFunctionalCP(nn.Module):
    """Continuous mode-wise neural CP with topology-matched input encoding."""

    def __init__(self, periodic: Sequence[bool], rank: int = 10,
                 hidden: int = 48):
        super().__init__()
        self.rank = int(rank)
        self.factors = nn.ModuleList(
            [_ModeFactor(self.rank, hidden, flag) for flag in periodic]
        )
        self.weight = nn.Parameter(torch.ones(self.rank) / math.sqrt(self.rank))

    def forward(self, coordinates: torch.Tensor) -> torch.Tensor:
        product = torch.ones(len(coordinates), self.rank,
                             device=coordinates.device)
        for mode, factor in enumerate(self.factors):
            product = product * factor(coordinates[:, mode])
        return (product * self.weight).sum(1)

    def regularization(self) -> torch.Tensor:
        return self.weight.square().mean()


class NeuralFunctionalTucker(nn.Module):
    """Continuous mode-wise neural Tucker with an explicit dense small core."""

    def __init__(self, periodic: Sequence[bool], ranks: Sequence[int] = (4, 5, 5),
                 hidden: int = 48):
        super().__init__()
        if len(periodic) != 3 or len(ranks) != 3:
            raise ValueError("current baseline expects an order-three tensor")
        self.ranks = tuple(int(rank) for rank in ranks)
        self.factors = nn.ModuleList([
            _ModeFactor(rank, hidden, flag)
            for rank, flag in zip(self.ranks, periodic)
        ])
        self.core = nn.Parameter(
            torch.randn(*self.ranks) / math.sqrt(math.prod(self.ranks))
        )

    def forward(self, coordinates: torch.Tensor) -> torch.Tensor:
        first, second, third = [
            factor(coordinates[:, mode])
            for mode, factor in enumerate(self.factors)
        ]
        return torch.einsum("na,nb,nc,abc->n", first, second, third, self.core)

    def regularization(self) -> torch.Tensor:
        return self.core.square().mean()
