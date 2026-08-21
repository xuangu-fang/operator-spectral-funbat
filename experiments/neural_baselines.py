#!/usr/bin/env python3
"""Stronger neural baselines for sparse tensor completion.

The GP arms in this paper are deliberately small.  If an explicit physical prior
is worth anything, it has to survive against models with far more capacity that
learn their structure from the data, so this file collects the neural
completions that a reviewer would reach for.

  CoSTCo          Liu et al., KDD 2019.  Per-index embeddings, then convolutions
                  that mix first across the rank axis and then across modes, so
                  the model is not restricted to a multilinear (outer-product)
                  interaction the way CP and Tucker are.
  Fourier MLP     A coordinate network in the NeRF sense: positional encoding of
                  the continuous index followed by a plain MLP.  No low-rank
                  assumption at all, and the strongest generic fitter here.
  LRTFR / SIREN   In neural_functional_tucker.py: sine-activation factors with a
                  Tucker core, the Continuous-Tensor-Toolbox construction.

Every model here trains only on observed entries, so the full tensor is never
materialised.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn


class CoSTCo(nn.Module):
    """Convolutional sparse tensor completion.

    The point of the architecture is that it is *not* multilinear: after
    embedding each index it convolves across the rank axis and then across the
    modes, so it can represent interactions no CP or Tucker model of the same
    rank can.  That is exactly the capacity argument this paper has to answer.
    """

    def __init__(self, shape: tuple[int, ...], *, rank: int = 16, channels: int = 32):
        super().__init__()
        self.order = len(shape)
        self.embeddings = nn.ModuleList([nn.Embedding(size, rank) for size in shape])
        for embedding in self.embeddings:
            nn.init.normal_(embedding.weight, std=0.1)
        self.across_rank = nn.Conv2d(1, channels, kernel_size=(rank, 1))
        self.across_modes = nn.Conv2d(channels, channels, kernel_size=(1, self.order))
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(channels, channels),
                                  nn.ReLU(), nn.Linear(channels, 1))

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        stacked = torch.stack(
            [self.embeddings[mode](indices[:, mode]) for mode in range(self.order)], -1)
        hidden = torch.relu(self.across_rank(stacked.unsqueeze(1)))
        hidden = torch.relu(self.across_modes(hidden))
        return self.head(hidden).squeeze(-1)


class FourierMLP(nn.Module):
    """Coordinate network with positional encoding; no low-rank assumption.

    Each index is mapped to its position in $[0,1]$ and expanded into sines and
    cosines at geometrically spaced frequencies, then a plain MLP predicts the
    value.  This is the most expressive arm in the paper and the one with the
    least inductive bias, which is the comparison worth having: capacity against
    a prior.
    """

    def __init__(self, shape: tuple[int, ...], *, bands: int = 8, width: int = 256,
                 depth: int = 4):
        super().__init__()
        self.register_buffer("sizes", torch.tensor(shape, dtype=torch.float32))
        self.register_buffer("frequencies", 2.0 ** torch.arange(bands) * np.pi)
        features = len(shape) * (1 + 2 * bands)
        layers: list[nn.Module] = [nn.Linear(features, width), nn.GELU()]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), nn.GELU()]
        layers += [nn.Linear(width, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        position = indices.float() / self.sizes
        scaled = position[..., None] * self.frequencies
        encoded = torch.cat([position, scaled.sin().flatten(1), scaled.cos().flatten(1)], -1)
        return self.net(encoded).squeeze(-1)


def fit_neural(model: nn.Module, observed: torch.Tensor, targets: torch.Tensor,
               test: torch.Tensor, truth: torch.Tensor, *, steps: int, lr: float,
               seed: int, device: torch.device, batch: int = 4096,
               weight_decay: float = 0.0) -> float:
    """Train on observed entries, report held-out NRMSE.

    Mini-batched because the strong arms are given many more observations than
    parameters would suggest, and full-batch training wastes their advantage.
    """
    torch.manual_seed(seed + 20_000)
    model = model.to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=steps)
    generator = torch.Generator(device=device).manual_seed(seed + 31)
    count = len(observed)
    for _ in range(steps):
        optimiser.zero_grad(set_to_none=True)
        if count > batch:
            pick = torch.randint(count, (batch,), generator=generator, device=device)
            index, value = observed[pick], targets[pick]
        else:
            index, value = observed, targets
        loss = (model(index) - value).square().mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimiser.step(); schedule.step()
    model.eval()
    with torch.no_grad():
        prediction = torch.cat([model(test[i:i + 65536])
                                for i in range(0, len(test), 65536)])
        return float(torch.sqrt(torch.mean((prediction - truth).square()))
                     / truth.std().clamp_min(1e-8))
