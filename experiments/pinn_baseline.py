#!/usr/bin/env python3
"""The other way to use the same equation: a PDE residual penalty.

This is the baseline the paper most needs.  Our construction turns the operator
into a prior over spectra; the obvious alternative turns it into a penalty on
the residual, which is what a physics-informed network does.  Both are given
exactly the same knowledge -- the equation's form and nominal coefficients that
are wrong by 50% -- so the comparison isolates *how* the physics is used rather
than how much of it is known.

The residual is the natural analogue of our prior rather than a different
assumption.  We model the solution's spectrum as |L(w,k)|^-2 times a white
forcing spectrum, which is the statement "L u is small and unstructured".
Penalising ||L u||^2 at collocation points says the same thing in the primal.
Neither arm is told where the leaks are.

Derivatives come from autograd on a coordinate network, so the residual is
exact for the model rather than finite-differenced.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from neural_baselines import FourierMLP


def pde_residual(model: nn.Module, points: torch.Tensor, *, diffusivity, reaction,
                 time_span: float) -> torch.Tensor:
    """``u_t - Dx u_xx - Dy u_yy + r u`` at continuous points in the unit cube.

    ``points`` are normalised to [0,1]^3.  Space is already physical because the
    domain is the unit square; time is not, so its derivative is rescaled by the
    physical duration the 64 recorded frames span.
    """
    points = points.clone().requires_grad_(True)
    value = model.forward_continuous(points)
    ones = torch.ones_like(value)
    gradient = torch.autograd.grad(value, points, ones, create_graph=True)[0]
    u_t = gradient[:, 0] / time_span
    second = []
    for axis in (1, 2):
        component = gradient[:, axis]
        second.append(torch.autograd.grad(component, points, torch.ones_like(component),
                                          create_graph=True)[0][:, axis])
    return u_t - diffusivity[0] * second[0] - diffusivity[1] * second[1] + reaction * value


class ContinuousFourierMLP(FourierMLP):
    """A FourierMLP that also accepts continuous coordinates, for autograd."""

    def forward_continuous(self, position: torch.Tensor) -> torch.Tensor:
        scaled = position[..., None] * self.frequencies
        encoded = torch.cat([position, scaled.sin().flatten(1), scaled.cos().flatten(1)], -1)
        return self.net(encoded).squeeze(-1)

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        return self.forward_continuous(indices.float() / self.sizes)


def fit_pinn(shape, observed, targets, test, truth, *, diffusivity, reaction, dt,
             residual_weight: float, steps: int, lr: float, seed: int,
             device: torch.device, collocation: int = 2048, batch: int = 4096,
             bands: int = 8, width: int = 256, depth: int = 4) -> float:
    torch.manual_seed(seed + 20_000)
    model = ContinuousFourierMLP(shape, bands=bands, width=width, depth=depth).to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=steps)
    generator = torch.Generator(device=device).manual_seed(seed + 31)
    time_span = shape[0] * dt
    count = len(observed)
    for _ in range(steps):
        optimiser.zero_grad(set_to_none=True)
        if count > batch:
            pick = torch.randint(count, (batch,), generator=generator, device=device)
            index, value = observed[pick], targets[pick]
        else:
            index, value = observed, targets
        loss = (model(index) - value).square().mean()
        if residual_weight > 0:
            points = torch.rand(collocation, 3, generator=generator, device=device)
            residual = pde_residual(model, points, diffusivity=diffusivity,
                                    reaction=reaction, time_span=time_span)
            loss = loss + residual_weight * residual.square().mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimiser.step(); schedule.step()
    model.eval()
    with torch.no_grad():
        prediction = torch.cat([model(test[i:i + 65536]) for i in range(0, len(test), 65536)])
        return float(torch.sqrt(torch.mean((prediction - truth).square()))
                     / truth.std().clamp_min(1e-8))
