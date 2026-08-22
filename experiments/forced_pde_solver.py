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
    """Spatially correlated forcing with a smooth Gaussian spectrum.

    An earlier version block-replicated coarse noise, which is piecewise
    constant and therefore has a sinc-shaped spectrum with exact zeros at
    multiples of ``n/scale`` plus harmonics.  No smooth spectral prior can model
    that, and it silently dominated the field's per-mode spectra -- the measured
    y-mode spectrum peaked at the forcing's harmonic rather than at the origin.
    Filtering white noise with a Gaussian instead gives ``S_w(k) ~ exp(-a k^2)``,
    which is the forcing model the operator construction actually assumes.
    """
    from scipy.ndimage import gaussian_filter
    white = rng.standard_normal(shape)
    smoothed = gaussian_filter(white, sigma=scale / 2.0, mode="reflect")
    return smoothed / max(smoothed.std(), 1e-12)


def solve_forced(
    *,
    operator: Literal["anisotropic_diffusion", "advection_diffusion", "damped_wave",
                      "banded_pattern"],
    grid: tuple[int, int] = (64, 64),
    diffusivity: tuple[float, float] = (0.004, 0.0012),
    reaction: float = 0.6,
    velocity: tuple[float, float] = (0.35, -0.2),
    band_wavenumber: float = 2.0,
    band_stiffness: float = 2.0e-4,
    band_offset: float = 0.25,
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
        if operator == "banded_pattern":
            # Linear Swift-Hohenberg: L = a + b (lap + k0^2)^2.  Its response is
            # minimised at |k| = k0, so the steady-state spectrum has a peak at
            # a *nonzero* wavenumber.  A generic smooth dictionary cannot
            # express a band-pass spectrum at all, which is what makes the
            # shape of the operator spectrum -- not merely its smoothness --
            # the thing being tested.  A small k0 keeps the pattern large-scale
            # and therefore still low multilinear rank.
            k0sq = (2 * np.pi * band_wavenumber) ** 2
            lap = ux + uy
            lx, ly = _laplacian_neumann(lap, dx, dy)
            biharmonic = lx + ly
            return -(band_offset * state
                     + band_stiffness * (biharmonic + 2 * k0sq * lap + k0sq**2 * state))
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


def narrowband_forcing_spectrum(
    steps: int, dt: float, centre: float, width: float,
) -> np.ndarray:
    """Band-pass temporal forcing power, a Gaussian bump at ``centre``.

    This is the separable band-pass case, and it is deliberately not the damped
    wave.  A wave's temporal resonance sits at ``c sqrt(lambda_q)``, so it moves
    with the spatial mode and the joint spectrum lies on a dispersion surface --
    coupled, and the worst case for an axis-separable approximation.  Driving a
    dissipative system with narrowband noise instead puts the same temporal band
    on every spatial mode, so the joint spectrum factorises by construction.

    Physically this is ordinary: ocean swell, machinery vibration, ambient
    seismic noise in a band, mains ripple.
    """
    if centre <= 0 or width <= 0:
        raise ValueError("centre and width must be positive")
    frequency = np.fft.rfftfreq(steps, d=dt) * 2 * np.pi
    return np.exp(-0.5 * ((frequency - centre) / width) ** 2)


def _dct_eigenvalues(n: int, spacing: float) -> np.ndarray:
    """Eigenvalues of the Neumann finite-difference Laplacian on a uniform grid.

    They are ``-(2 - 2 cos(pi k / n)) / h^2``, which differs from the continuum
    ``-(pi k)^2`` -- a genuine discretisation mismatch between the field and the
    continuum symbol the prior is built from.
    """
    k = np.arange(n)
    return -(2 - 2 * np.cos(np.pi * k / n)) / spacing**2


def solve_forced_spectral(
    *,
    operator: Literal["anisotropic_diffusion", "banded_pattern"],
    grid: tuple[int, int] = (32, 32),
    diffusivity: tuple[float, float] = (0.02, 0.006),
    reaction: float = 0.8,
    band_wavenumber: float = 2.0,
    band_stiffness: float = 2.0e-4,
    band_offset: float = 0.25,
    forcing_scale: int = 8,
    forcing_std: float = 1.0,
    dt: float = 0.06,
    burn_in: int = 200,
    record_steps: int = 32,
    seed: int = 0,
) -> ForcedField:
    """Exponential (exact-in-time) integration for DCT-diagonal operators.

    Explicit stepping is unusable for the fourth-order banded operator, whose
    stability limit scales like ``h^4``.  Because a Neumann finite-difference
    operator with constant coefficients is diagonalised exactly by the DCT, the
    linear part can instead be integrated exactly, which is unconditionally
    stable and lets the forcing correlation rather than the timestep set the
    physics.
    """
    from scipy.fft import dctn, idctn

    rng = np.random.default_rng(seed)
    nx, ny = grid
    lx = _dct_eigenvalues(nx, 1.0 / nx)[:, None]
    ly = _dct_eigenvalues(ny, 1.0 / ny)[None, :]
    if operator == "anisotropic_diffusion":
        multiplier = diffusivity[0] * lx + diffusivity[1] * ly - reaction
    elif operator == "banded_pattern":
        k0sq = (2 * np.pi * band_wavenumber) ** 2
        multiplier = -(band_offset + band_stiffness * (lx + ly + k0sq) ** 2)
    else:
        raise ValueError(f"{operator} is not DCT-diagonal; use solve_forced")
    if np.any(multiplier >= 0):
        raise ValueError("operator has a non-decaying mode; steady state does not exist")
    decay = np.exp(multiplier * dt)
    # Exact Ornstein-Uhlenbeck increment for each mode.
    increment = np.sqrt((1 - decay**2) / (-2 * multiplier))

    state = np.zeros(grid)
    frames = []
    for step in range(burn_in + record_steps):
        forcing = forcing_std * _smooth_noise(rng, grid, forcing_scale)
        forcing_hat = dctn(forcing, norm="ortho")
        state = idctn(dctn(state, norm="ortho") * decay + forcing_hat * increment,
                      norm="ortho")
        if not np.isfinite(state).all():
            raise FloatingPointError(f"spectral solver diverged at step {step}")
        if step >= burn_in:
            frames.append(state.copy())
    field = torch.from_numpy(np.stack(frames)).float()
    field = (field - field.mean()) / field.std().clamp_min(1e-8)
    return ForcedField(
        field=field, operator=operator, grid=grid, dt=dt,
        parameters={"diffusivity": diffusivity, "reaction": reaction,
                    "band_wavenumber": band_wavenumber, "band_stiffness": band_stiffness,
                    "band_offset": band_offset, "forcing_scale": forcing_scale,
                    "integrator": "exponential/DCT"},
    )


def solve_narrowband_forced(
    *,
    grid: tuple[int, int] = (32, 32),
    diffusivity: tuple[float, float] = (0.02, 0.006),
    reaction: float = 0.8,
    forcing_scale: int = 8,
    band_centre: float = 6.0,
    band_width: float = 1.2,
    dt: float = 0.05,
    burn_in: int = 256,
    record_steps: int = 32,
    seed: int = 0,
) -> ForcedField:
    """Dissipative field driven by temporally narrowband, spatially smooth noise.

    The forcing is white in space (after Gaussian smoothing) but band-limited in
    time, so the response's joint spectrum is (band-pass in omega) times
    (low-pass in k).  That is the one case where a generic monotone kernel is
    structurally unable to follow the truth while the axis-separable
    approximation the method relies on remains exact.
    """
    from scipy.fft import dctn, idctn

    rng = np.random.default_rng(seed)
    nx, ny = grid
    lx = _dct_eigenvalues(nx, 1.0 / nx)[:, None]
    ly = _dct_eigenvalues(ny, 1.0 / ny)[None, :]
    multiplier = diffusivity[0] * lx + diffusivity[1] * ly - reaction
    if np.any(multiplier >= 0):
        raise ValueError("operator has a non-decaying mode")

    total = burn_in + record_steps
    # Draw the whole forcing history at once so its temporal spectrum can be
    # shaped exactly, rather than approximated by an online filter.
    power = narrowband_forcing_spectrum(total, dt, band_centre, band_width)
    white = rng.standard_normal((total, nx, ny))
    smoothed = np.stack([_smooth_noise(rng, grid, forcing_scale) for _ in range(total)])
    spectrum = np.fft.rfft(smoothed, axis=0)
    forcing = np.fft.irfft(spectrum * np.sqrt(power)[:, None, None], n=total, axis=0)

    decay = np.exp(multiplier * dt)
    state = np.zeros(grid)
    frames = []
    for step in range(total):
        hat = dctn(state, norm="ortho") * decay + dctn(forcing[step], norm="ortho") * dt
        state = idctn(hat, norm="ortho")
        if not np.isfinite(state).all():
            raise FloatingPointError(f"narrowband solver diverged at step {step}")
        if step >= burn_in:
            frames.append(state.copy())
    field = torch.from_numpy(np.stack(frames)).float()
    field = (field - field.mean()) / field.std().clamp_min(1e-8)
    return ForcedField(
        field=field, operator="narrowband_forced", grid=grid, dt=dt,
        parameters={"diffusivity": diffusivity, "reaction": reaction,
                    "band_centre": band_centre, "band_width": band_width,
                    "forcing_scale": forcing_scale, "integrator": "exponential/DCT"},
    )


def solve_localized_source(
    *,
    grid: tuple[int, int] = (64, 64),
    diffusivity: tuple[float, float] = (0.02, 0.006),
    reaction: float = 0.8,
    source_centre: tuple[float, float] = (0.35, 0.6),
    source_width: float = 0.05,
    source_period: float = 6.0,
    dt: float = 0.1,
    burn_in: int = 200,
    record_steps: int = 64,
    background_noise: float = 0.02,
    forcing_scale: int = 16,
    seed: int = 0,
) -> ForcedField:
    """A leak: one localized, time-varying source; elsewhere the field is source-free.

    This is the case the boundary-sensor scenario actually needs.  When a field
    is stochastically forced everywhere, its interior is *not* determined by its
    boundary -- there is independent forcing inside that no boundary
    measurement can see -- so reconstructing the interior from wall sensors is
    information-theoretically impossible and every method fails together.

    With a localized source the rest of the domain satisfies ``L u = 0``, the
    boundary genuinely does determine the interior, and a prior that knows the
    operator has something real to contribute.  A small background noise is kept
    so the task is not noiseless inverse-crime.
    """
    from scipy.fft import dctn, idctn

    rng = np.random.default_rng(seed)
    nx, ny = grid
    lx = _dct_eigenvalues(nx, 1.0 / nx)[:, None]
    ly = _dct_eigenvalues(ny, 1.0 / ny)[None, :]
    multiplier = diffusivity[0] * lx + diffusivity[1] * ly - reaction
    if np.any(multiplier >= 0):
        raise ValueError("operator has a non-decaying mode")
    decay = np.exp(multiplier * dt)

    x = (np.arange(nx) + 0.5) / nx
    y = (np.arange(ny) + 0.5) / ny
    profile = np.exp(-0.5 * (((x[:, None] - source_centre[0]) ** 2
                              + (y[None, :] - source_centre[1]) ** 2)
                             / source_width ** 2))
    profile = profile / max(profile.max(), 1e-12)

    state = np.zeros(grid)
    frames = []
    for step in range(burn_in + record_steps):
        time = step * dt
        amplitude = 1.0 + 0.6 * np.sin(2 * np.pi * time / source_period)
        forcing = amplitude * profile
        if background_noise > 0:
            forcing = forcing + background_noise * _smooth_noise(rng, grid, forcing_scale)
        hat = dctn(state, norm="ortho") * decay + dctn(forcing, norm="ortho") * dt
        state = idctn(hat, norm="ortho")
        if not np.isfinite(state).all():
            raise FloatingPointError(f"localized-source solver diverged at step {step}")
        if step >= burn_in:
            frames.append(state.copy())
    field = torch.from_numpy(np.stack(frames)).float()
    field = (field - field.mean()) / field.std().clamp_min(1e-8)
    return ForcedField(
        field=field, operator="localized_source", grid=grid, dt=dt,
        parameters={"diffusivity": diffusivity, "reaction": reaction,
                    "source_centre": source_centre, "source_width": source_width,
                    "source_period": source_period,
                    "background_noise": background_noise,
                    "integrator": "exponential/DCT"},
    )


def solve_multi_leak(
    *,
    grid: tuple[int, int] = (64, 64),
    diffusivity: tuple[float, float] = (0.02, 0.006),
    reaction: float = 0.04,
    sources: tuple[tuple[float, float, float, float], ...] = (
        (0.30, 0.65, 0.09, 5.0),
        (0.70, 0.35, 0.07, 8.0),
        (0.55, 0.75, 0.06, 13.0),
    ),
    dt: float = 0.1,
    burn_in: int = 300,
    record_steps: int = 64,
    background_noise: float = 0.02,
    forcing_scale: int = 16,
    drift: tuple[float, float] = (0.0, 0.0),
    seed: int = 0,
) -> ForcedField:
    """Several leaks at different places, each with its own time signature.

    A single leak makes the field rank one -- a fixed spatial plume times a
    scalar in time -- so the task degenerates into recognising one pattern.
    Several leaks with different periods give a genuinely multi-component field
    whose plumes must be separated, which is what makes boundary reconstruction
    a real problem and what a prior that knows how each plume decays can help
    with.  Each source is ``(x, y, width, period)``.

    Away from the sources the field satisfies ``L u = 0`` up to a small
    background, so the boundary genuinely constrains the interior.
    """
    from scipy.fft import dctn, idctn

    rng = np.random.default_rng(seed)
    nx, ny = grid
    lx = _dct_eigenvalues(nx, 1.0 / nx)[:, None]
    ly = _dct_eigenvalues(ny, 1.0 / ny)[None, :]
    multiplier = diffusivity[0] * lx + diffusivity[1] * ly - reaction
    if np.any(multiplier >= 0):
        raise ValueError("operator has a non-decaying mode")
    decay = np.exp(multiplier * dt)

    x = (np.arange(nx) + 0.5) / nx
    y = (np.arange(ny) + 0.5) / ny
    profiles, periods = [], []
    for cx, cy, width, period in sources:
        blob = np.exp(-0.5 * (((x[:, None] - cx) ** 2 + (y[None, :] - cy) ** 2)
                              / width ** 2))
        profiles.append(blob / max(blob.max(), 1e-12))
        periods.append(period)

    state = np.zeros(grid)
    frames = []
    for step in range(burn_in + record_steps):
        time = step * dt
        forcing = np.zeros(grid)
        for blob, period in zip(profiles, periods):
            forcing = forcing + (1.0 + 0.8 * np.sin(2 * np.pi * time / period)) * blob
        if background_noise > 0:
            forcing = forcing + background_noise * _smooth_noise(rng, grid, forcing_scale)
        hat = dctn(state, norm="ortho") * decay + dctn(forcing, norm="ortho") * dt
        state = idctn(hat, norm="ortho")
        if drift[0] or drift[1]:
            # Advection is not diagonal in the cosine basis, so it is applied by
            # operator splitting: a first-order upwind step in real space after
            # the spectral diffusion-reaction step.  Walls are no-flux, so the
            # gradient is taken as zero across them.  The scheme is stable while
            # |v| dt / h < 1, which is checked below rather than assumed.
            for axis, speed in enumerate(drift):
                if not speed:
                    continue
                spacing = 1.0 / grid[axis]
                if abs(speed) * dt / spacing >= 1.0:
                    raise ValueError(
                        f"advection violates CFL on axis {axis}: "
                        f"|v| dt / h = {abs(speed) * dt / spacing:.2f} >= 1")
                forward = np.diff(state, axis=axis, append=np.take(
                    state, [-1], axis=axis)) / spacing
                backward = np.diff(state, axis=axis, prepend=np.take(
                    state, [0], axis=axis)) / spacing
                gradient = backward if speed > 0 else forward
                state = state - dt * speed * gradient
        if not np.isfinite(state).all():
            raise FloatingPointError(f"multi-leak solver diverged at step {step}")
        if step >= burn_in:
            frames.append(state.copy())
    field = torch.from_numpy(np.stack(frames)).float()
    field = (field - field.mean()) / field.std().clamp_min(1e-8)
    return ForcedField(
        field=field, operator="multi_leak", grid=grid, dt=dt,
        parameters={"diffusivity": diffusivity, "reaction": reaction,
                    "sources": sources, "background_noise": background_noise,
                    "integrator": "exponential/DCT"},
    )


def solve_multi_leak_3d(
    *,
    grid: tuple[int, int, int] = (32, 32, 32),
    diffusivity: tuple[float, float, float] = (0.02, 0.006, 0.012),
    reaction: float = 0.04,
    sources: tuple[tuple[float, float, float, float, float], ...] = (
        (0.30, 0.65, 0.40, 0.10, 15.0),
        (0.70, 0.35, 0.60, 0.09, 25.0),
        (0.55, 0.75, 0.30, 0.08, 40.0),
    ),
    dt: float = 0.6,
    burn_in: int = 200,
    record_steps: int = 32,
    background_noise: float = 0.02,
    forcing_scale: int = 8,
    drift: tuple[float, float, float] = (0.0, 0.0, 0.0),
    seed: int = 0,
) -> ForcedField:
    """The leak room in three spatial dimensions.

    This is the geometry the motivating applications actually have.  In two
    dimensions a wall-mounted array is a line and the unobserved region is a
    half-plane; in three it is a *face* and the unobserved region is a half
    volume, so the same fraction of observed cells buys a great deal less.  It
    is also where a per-mode construction has the most to gain or lose, since
    there are now three spatial spectra to get right instead of two.

    Each source is ``(x, y, z, width, period)``.  Boundaries are no-flux on all
    six faces, so the cosine eigenbasis carries over unchanged.
    """
    from scipy.fft import dctn, idctn

    rng = np.random.default_rng(seed)
    nx, ny, nz = grid
    lx = _dct_eigenvalues(nx, 1.0 / nx)[:, None, None]
    ly = _dct_eigenvalues(ny, 1.0 / ny)[None, :, None]
    lz = _dct_eigenvalues(nz, 1.0 / nz)[None, None, :]
    multiplier = (diffusivity[0] * lx + diffusivity[1] * ly
                  + diffusivity[2] * lz - reaction)
    if np.any(multiplier >= 0):
        raise ValueError("operator has a non-decaying mode")
    decay = np.exp(multiplier * dt)

    x = (np.arange(nx) + 0.5) / nx
    y = (np.arange(ny) + 0.5) / ny
    z = (np.arange(nz) + 0.5) / nz
    profiles, periods = [], []
    for cx, cy, cz, width, period in sources:
        squared = ((x[:, None, None] - cx) ** 2 + (y[None, :, None] - cy) ** 2
                   + (z[None, None, :] - cz) ** 2)
        blob = np.exp(-0.5 * squared / width ** 2)
        profiles.append(blob / max(blob.max(), 1e-12))
        periods.append(period)

    state = np.zeros(grid)
    frames = []
    for step in range(burn_in + record_steps):
        time = step * dt
        forcing = np.zeros(grid)
        for blob, period in zip(profiles, periods):
            forcing = forcing + (1.0 + 0.8 * np.sin(2 * np.pi * time / period)) * blob
        if background_noise > 0:
            forcing = forcing + background_noise * _smooth_noise(rng, grid, forcing_scale)
        hat = dctn(state, norm="ortho") * decay + dctn(forcing, norm="ortho") * dt
        state = idctn(hat, norm="ortho")
        if any(drift):
            # Same operator splitting as in two dimensions: a first-order upwind
            # step in real space after the spectral diffusion-reaction step, with
            # the CFL condition checked rather than assumed.
            for axis, speed in enumerate(drift):
                if not speed:
                    continue
                spacing = 1.0 / grid[axis]
                if abs(speed) * dt / spacing >= 1.0:
                    raise ValueError(
                        f"advection violates CFL on axis {axis}: "
                        f"|v| dt / h = {abs(speed) * dt / spacing:.2f} >= 1")
                forward = np.diff(state, axis=axis, append=np.take(
                    state, [-1], axis=axis)) / spacing
                backward = np.diff(state, axis=axis, prepend=np.take(
                    state, [0], axis=axis)) / spacing
                state = state - dt * speed * (backward if speed > 0 else forward)
        if not np.isfinite(state).all():
            raise FloatingPointError(f"3-D multi-leak solver diverged at step {step}")
        if step >= burn_in:
            frames.append(state.copy())
    field = torch.from_numpy(np.stack(frames)).float()
    field = (field - field.mean()) / field.std().clamp_min(1e-8)
    return ForcedField(
        field=field,
        operator="multi_leak_3d",
        parameters={"diffusivity": diffusivity, "reaction": reaction,
                    "sources": sources, "seed": seed},
        grid=grid,
        dt=dt,
    )


def solve_steady_state_2d(
    *,
    grid: tuple[int, int] = (256, 256),
    diffusivity: tuple[float, float] = (0.02, 0.006),
    reaction: float = 0.04,
    sources: tuple[tuple[float, float, float, float], ...] = (
        (0.30, 0.65, 0.09, 1.0),
        (0.70, 0.35, 0.07, 0.8),
        (0.55, 0.75, 0.06, 0.6),
    ),
    background_noise: float = 0.02,
    forcing_scale: int = 32,
    seed: int = 0,
) -> ForcedField:
    """A purely spatial field at steady state: no time axis at all.

    Every other field in this project is space plus time, so the tensor has a
    mode along which the operator is a first derivative and the field is nearly
    constant -- a mode that carries a great deal of the structure and is fully
    observed by any sensor layout.  Removing it is a genuinely different regime,
    not a smaller version of the same one: two modes instead of three, and no
    time axis to borrow information along.

    It is also where resolution can be pushed.  At steady state the solve is a
    single division in the cosine domain rather than an integration, so 256x256
    costs no more than 64x64 did with 64 frames, while making the same 1%
    observation budget cover far more of the domain.

    Solves ``(r - Dx d_xx - Dy d_yy) u = f`` with no-flux walls, which is
    diagonal in the DCT basis, so the answer is exact rather than integrated.
    Each source is ``(x, y, width, amplitude)``.
    """
    from scipy.fft import dctn, idctn

    rng = np.random.default_rng(seed)
    nx, ny = grid
    lx = -_dct_eigenvalues(nx, 1.0 / nx)[:, None]
    ly = -_dct_eigenvalues(ny, 1.0 / ny)[None, :]
    elliptic = reaction + diffusivity[0] * lx + diffusivity[1] * ly
    if np.any(elliptic <= 0):
        raise ValueError("elliptic operator is not positive definite")

    x = (np.arange(nx) + 0.5) / nx
    y = (np.arange(ny) + 0.5) / ny
    forcing = np.zeros(grid)
    for cx, cy, width, amplitude in sources:
        squared = (x[:, None] - cx) ** 2 + (y[None, :] - cy) ** 2
        blob = np.exp(-0.5 * squared / width ** 2)
        forcing = forcing + amplitude * blob / max(blob.max(), 1e-12)
    if background_noise > 0:
        forcing = forcing + background_noise * _smooth_noise(rng, grid, forcing_scale)

    field = idctn(dctn(forcing, norm="ortho") / elliptic, norm="ortho")
    if not np.isfinite(field).all():
        raise FloatingPointError("steady-state solve produced non-finite values")
    field = torch.from_numpy(field).float()
    field = (field - field.mean()) / field.std().clamp_min(1e-8)
    return ForcedField(
        field=field,
        operator="steady_state_2d",
        parameters={"diffusivity": diffusivity, "reaction": reaction,
                    "sources": sources, "seed": seed},
        grid=grid,
        dt=0.0,
    )
