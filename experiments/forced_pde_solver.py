#!/usr/bin/env python3
"""Independent finite-difference solver for stochastically forced linear PDEs.

Why this exists.  Every earlier positive result used fields *sampled from the
model's own prior*, so the ground truth was a rank-4 nonnegative-CP field by
construction and the comparison was partly self-fulfilling.  Here the truth is a
genuine numerical solution: the field carries the operator's true, fully
non-separable joint spectrum, while the method only ever sees a rank-Q
nonnegative separation of it.  The gap between the two is real, which is exactly
what we want to test.

Setting: drive a linear operator with white-in-time noise and run to statistical
steady state, so the field is a stationary Gaussian process with spectral
density ``|L_hat|^-2 S_w`` -- the object the method claims to approximate.  No
part of the solver knows about atoms, routing or Fourier features.

Boundary conditions are no-flux (Neumann), matching the cosine eigenbasis, and
also matching how most real measurements of a bounded domain behave.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch


@dataclass(frozen=True)
class ForcedField:
    field: torch.Tensor          # [t, x, y], standardized
    operator: str
    parameters: dict
    grid: tuple[int, int]
    dt: float


def _laplacian_neumann(u: np.ndarray, dx: float, dy: float) -> tuple[np.ndarray, np.ndarray]:
    """Second differences with reflecting (no-flux) edges, per axis."""
    ux = np.empty_like(u)
    ux[1:-1] = (u[2:] - 2 * u[1:-1] + u[:-2]) / dx**2
    ux[0] = 2 * (u[1] - u[0]) / dx**2
    ux[-1] = 2 * (u[-2] - u[-1]) / dx**2
    uy = np.empty_like(u)
    uy[:, 1:-1] = (u[:, 2:] - 2 * u[:, 1:-1] + u[:, :-2]) / dy**2
    uy[:, 0] = 2 * (u[:, 1] - u[:, 0]) / dy**2
    uy[:, -1] = 2 * (u[:, -2] - u[:, -1]) / dy**2
    return ux, uy


def _smooth_noise(rng: np.random.Generator, shape: tuple[int, int], scale: int) -> np.ndarray:
    """Spatially correlated forcing: white noise block-averaged then bilinear-ish."""
    coarse = rng.standard_normal((max(shape[0] // scale, 1), max(shape[1] // scale, 1)))
    return np.kron(coarse, np.ones((scale, scale)))[: shape[0], : shape[1]]


def solve_forced(
    *,
    operator: Literal["anisotropic_diffusion", "advection_diffusion", "damped_wave"],
    grid: tuple[int, int] = (64, 64),
    diffusivity: tuple[float, float] = (0.004, 0.0012),
    reaction: float = 0.6,
    velocity: tuple[float, float] = (0.35, -0.2),
    wave_speed: float = 0.35,
    wave_damping: float = 0.9,
    forcing_scale: int = 4,
    forcing_std: float = 1.0,
    dt: float = 2.0e-3,
    burn_in: int = 4000,
    record_steps: int = 64,
    record_every: int = 30,
    seed: int = 0,
) -> ForcedField:
    """Euler-Maruyama integration to statistical steady state, then record."""
    rng = np.random.default_rng(seed)
    nx, ny = grid
    dx, dy = 1.0 / nx, 1.0 / ny
    u = np.zeros(grid)
    v = np.zeros(grid)  # velocity slot for the wave operator
    dxc, dyc = diffusivity
    noise_gain = forcing_std * np.sqrt(dt)

    def drift(state: np.ndarray) -> np.ndarray:
        ux, uy = _laplacian_neumann(state, dx, dy)
        if operator == "anisotropic_diffusion":
            return dxc * ux + dyc * uy - reaction * state
        if operator == "advection_diffusion":
            gx = np.empty_like(state); gy = np.empty_like(state)
            gx[1:-1] = (state[2:] - state[:-2]) / (2 * dx)
            gx[0] = (state[1] - state[0]) / dx; gx[-1] = (state[-1] - state[-2]) / dx
            gy[:, 1:-1] = (state[:, 2:] - state[:, :-2]) / (2 * dy)
            gy[:, 0] = (state[:, 1] - state[:, 0]) / dy
            gy[:, -1] = (state[:, -1] - state[:, -2]) / dy
            return dxc * ux + dyc * uy - velocity[0] * gx - velocity[1] * gy - reaction * state
        raise ValueError(f"unknown operator: {operator}")

    frames = []
    total = burn_in + record_steps * record_every
    for step in range(total):
        forcing = noise_gain * _smooth_noise(rng, grid, forcing_scale)
        if operator == "damped_wave":
            ux, uy = _laplacian_neumann(u, dx, dy)
            acceleration = wave_speed**2 * (ux + uy) - wave_damping * v
            v = v + dt * acceleration + forcing
            u = u + dt * v
        else:
            u = u + dt * drift(u) + forcing
        if not np.isfinite(u).all():
            raise FloatingPointError(f"solver diverged at step {step}; reduce dt")
        if step >= burn_in and (step - burn_in) % record_every == 0:
            frames.append(u.copy())
        if len(frames) == record_steps:
            break
    field = torch.from_numpy(np.stack(frames)).float()
    field = (field - field.mean()) / field.std().clamp_min(1e-8)
    return ForcedField(
        field=field, operator=operator, grid=grid, dt=dt,
        parameters={"diffusivity": diffusivity, "reaction": reaction,
                    "velocity": velocity, "forcing_scale": forcing_scale,
                    "record_every": record_every, "burn_in": burn_in},
    )


def block_average(field: torch.Tensor, factors: tuple[int, int, int]) -> torch.Tensor:
    """Anti-aliased downsampling; striding aliases structure into white noise."""
    shape = []
    for size, factor in zip(field.shape, factors):
        if size % factor:
            raise ValueError(f"size {size} not divisible by {factor}")
        shape += [size // factor, factor]
    return field.reshape(*shape).mean(dim=(1, 3, 5))
