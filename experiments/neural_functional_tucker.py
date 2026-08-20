#!/usr/bin/env python3
"""SIREN-style neural functional Tucker, the method-matched neural baseline.

Adapted from the Continuous Tensor Toolbox
(https://github.com/YisiLuo/Continuous-Tensor-Toolbox): each mode's factor is a
small sine-activation MLP of that mode's continuous coordinate, and the factors
are contracted with a dense core.  It is the natural neural counterpart of what
this work does with GP priors -- same functional Tucker structure, same fixed
rank, but the factor functions are learned by a network instead of drawn from a
kernel -- so it isolates whether an explicit prior beats network capacity at
this sparsity.

Only entries at observed indices are evaluated, so a 64^3 tensor never has to be
materialised during training.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn


class SineLayer(nn.Module):
    """SIREN layer; initialisation follows the reference implementation."""

    def __init__(self, in_features: int, out_features: int, *, is_first: bool = False,
                 omega: float = 30.0):
        super().__init__()
        self.omega = omega
        self.is_first = is_first
        self.linear = nn.Linear(in_features, out_features)
        with torch.no_grad():
            if is_first:
                bound = 1.0 / in_features
            else:
                bound = math.sqrt(6.0 / in_features) / omega
            self.linear.weight.uniform_(-bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.omega * self.linear(x))


class NeuralFunctionalTucker(nn.Module):
    def __init__(self, ranks: tuple[int, int, int], *, hidden: int = 64,
                 omega: float = 30.0, seed: int = 0):
        super().__init__()
        torch.manual_seed(seed)
        self.nets = nn.ModuleList([
            nn.Sequential(SineLayer(1, hidden, is_first=True, omega=omega),
                          SineLayer(hidden, hidden, omega=omega),
                          nn.Linear(hidden, rank))
            for rank in ranks
        ])
        self.core = nn.Parameter(torch.randn(*ranks) / math.sqrt(float(np.prod(ranks))))

    def forward(self, coordinates: list[torch.Tensor]) -> torch.Tensor:
        """`coordinates` holds one [entry, 1] column per mode; returns [entry]."""
        factors = [net(c) for net, c in zip(self.nets, coordinates)]
        partial = torch.einsum("na,abc->nbc", factors[0], self.core)
        partial = torch.einsum("nb,nbc->nc", factors[1], partial)
        return (factors[2] * partial).sum(-1)


def fit_neural_tucker(field_shape, observed, targets, evaluate_at, truth, *,
                      ranks, steps=3000, lr=1e-3, weight_decay=0.0, hidden=64,
                      omega=30.0, seed=0, device="cuda"):
    """Fit on observed entries, report normalised error at `evaluate_at`."""
    model = NeuralFunctionalTucker(ranks, hidden=hidden, omega=omega, seed=seed).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    sizes = torch.tensor(field_shape, device=device, dtype=torch.float32)

    def columns(index):
        scaled = index.float() / sizes
        return [scaled[:, m:m + 1] for m in range(3)]

    train_columns = columns(observed)
    test_columns = columns(evaluate_at)
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss = (model(train_columns) - targets).square().mean()
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        prediction = model(test_columns)
        return float(torch.sqrt(torch.mean((prediction - truth).square()))
                     / truth.std().clamp_min(1e-8))
