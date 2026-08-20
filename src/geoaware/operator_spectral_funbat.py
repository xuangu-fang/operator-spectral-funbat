"""Operator-spectral Gaussian-process factors for continuous CP tensors.

The implementation is deliberately finite and auditable.  A nonnegative power
spectrum is converted to real Fourier features, so every routed kernel is PSD by
construction.  Mean-field Gaussian posteriors over the Fourier coefficients are
trained with a Monte-Carlo Gaussian ELBO.  This is the smallest implementation
that can test the mode/rank-specific kernel idea without hiding it in a neural
feature extractor.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def normalize_spectrum(spectrum: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Normalize a nonnegative one-sided spectrum to unit marginal variance."""
    if spectrum.ndim < 1 or spectrum.shape[-1] < 2:
        raise ValueError("spectrum must have at least two frequency bins")
    if torch.any(spectrum < 0) or not torch.isfinite(spectrum).all():
        raise ValueError("spectrum must be finite and nonnegative")
    # Frequencies k>0 contribute a cosine and sine feature.
    mass = spectrum[..., :1] + 2 * spectrum[..., 1:].sum(-1, keepdim=True)
    return spectrum / mass.clamp_min(eps)


def generic_spectral_dictionary(max_frequency: int = 6) -> tuple[tuple[str, ...], torch.Tensor]:
    """Return smooth, rough, oscillatory and broadband one-dimensional spectra."""
    if max_frequency < 2:
        raise ValueError("max_frequency must be >= 2")
    w = torch.arange(max_frequency + 1, dtype=torch.float32)
    spectra = torch.stack(
        [
            torch.exp(-0.70 * w.square()),
            (1 + 0.32 * w.square()).pow(-1.5),
            torch.exp(-0.5 * ((w - 4.0) / 0.75).square()) + 0.03,
            (1 + 0.08 * w.square()).reciprocal(),
        ]
    )
    return ("smooth", "matern", "oscillatory", "broadband"), normalize_spectrum(spectra)


def extended_generic_dictionary(
    count: int, max_frequency: int = 6,
) -> tuple[tuple[str, ...], torch.Tensor]:
    """Return ``count`` generic one-dimensional spectra.

    The first four entries are exactly ``generic_spectral_dictionary`` so that
    every previously frozen result stays reproducible.  Additional atoms extend
    the same three families (squared-exponential, Matern-like, oscillatory)
    with deterministically interleaved parameters.  This exists to build an
    atom-count-matched generic control for pooled operator banks: without it a
    larger operator bank could win on bank size rather than on physics.
    """
    if count < 1:
        raise ValueError("count must be positive")
    names, base = generic_spectral_dictionary(max_frequency)
    if count <= len(names):
        return names[:count], base[:count]
    w = torch.arange(max_frequency + 1, dtype=torch.float32)
    # Interleaved so that any prefix of the extras stays family-diverse.
    extras: list[tuple[str, torch.Tensor]] = []
    smooth_scales = (0.35, 1.40, 0.18)
    matern_scales = (0.12, 0.80, 0.05)
    oscillatory_centres = (2.0, 5.0, 3.0, 6.0)
    broadband_scales = (0.03, 0.20)
    for index in range(max(len(smooth_scales), len(matern_scales),
                           len(oscillatory_centres), len(broadband_scales))):
        if index < len(smooth_scales):
            scale = smooth_scales[index]
            extras.append((f"smooth_{scale}", torch.exp(-scale * w.square())))
        if index < len(matern_scales):
            scale = matern_scales[index]
            extras.append((f"matern_{scale}", (1 + scale * w.square()).pow(-1.5)))
        if index < len(oscillatory_centres):
            centre = oscillatory_centres[index]
            extras.append((
                f"oscillatory_{centre}",
                torch.exp(-0.5 * ((w - centre) / 0.75).square()) + 0.03,
            ))
        if index < len(broadband_scales):
            scale = broadband_scales[index]
            extras.append((f"broadband_{scale}", (1 + scale * w.square()).reciprocal()))
    needed = count - len(names)
    if needed > len(extras):
        raise ValueError(
            f"extended dictionary supports at most {len(names) + len(extras)} atoms"
        )
    extra_names = tuple(name for name, _ in extras[:needed])
    extra_spectra = torch.stack([spectrum for _, spectrum in extras[:needed]])
    return names + extra_names, normalize_spectrum(
        torch.cat((base, extra_spectra), dim=0)
    )


def fourier_features(coordinate: torch.Tensor, spectrum: torch.Tensor) -> torch.Tensor:
    """Map one-sided spectra to real features with ``K = Phi Phi^T``.

    Args:
        coordinate: ``[node]`` coordinates scaled to the periodic interval [0, 1).
        spectrum: ``[..., frequency]`` nonnegative normalized spectra.

    Returns:
        ``[..., node, 1 + 2 * (frequency - 1)]`` features.
    """
    if coordinate.ndim != 1:
        raise ValueError("coordinate must be one-dimensional")
    spectrum = normalize_spectrum(spectrum.float())
    leading = spectrum.shape[:-1]
    root = spectrum.sqrt()
    constant = root[..., :1].unsqueeze(-2).expand(*leading, len(coordinate), 1)
    scale = (2 * spectrum[..., 1:]).sqrt().unsqueeze(-2)
    frequency = torch.arange(1, spectrum.shape[-1], device=coordinate.device,
                             dtype=coordinate.dtype)
    phase = 2 * math.pi * coordinate[:, None] * frequency[None]
    cosine = torch.cos(phase).reshape(*(1 for _ in leading), len(coordinate), -1)
    sine = torch.sin(phase).reshape(*(1 for _ in leading), len(coordinate), -1)
    return torch.cat((constant, scale * cosine, scale * sine), dim=-1)


def real_fourier_basis(coordinate: torch.Tensor, frequency_bins: int) -> torch.Tensor:
    """Atom-independent ``[1,sqrt(2)cos,sqrt(2)sin]`` feature basis."""
    if coordinate.ndim != 1 or frequency_bins < 2:
        raise ValueError("coordinate/frequency_bins are invalid")
    frequency = torch.arange(1, frequency_bins, device=coordinate.device,
                             dtype=coordinate.dtype)
    phase = 2 * math.pi * coordinate[:, None] * frequency[None]
    return torch.cat((
        torch.ones(len(coordinate), 1, device=coordinate.device, dtype=coordinate.dtype),
        math.sqrt(2) * torch.cos(phase),
        math.sqrt(2) * torch.sin(phase),
    ), dim=-1)


class LearnableStationarySpectrum(nn.Module):
    """A standard stationary kernel whose length scale is learned by the ELBO.

    This is the baseline a practitioner actually uses, and the one FunBaT-style
    models fit: a single named kernel with its hyper-parameters estimated from
    the same data, rather than a dictionary of hand-picked shapes with learned
    mixture weights.

    The distinction matters for what can be claimed.  A spectral mixture is
    dense in the space of stationary kernels, so given enough data it can
    represent an operator-derived spectrum exactly, and any advantage over it
    can only ever be sample efficiency.  A single Matern or squared-exponential
    spectrum is monotone by construction and therefore cannot represent a
    band-pass spectrum at any sample size -- a qualitatively different kind of
    baseline, and the honest default.
    """

    FAMILIES = ("rbf", "matern12", "matern32", "matern52")

    def __init__(self, family: str, frequency_bins: int, *,
                 initial_length_scale: float = 0.3):
        super().__init__()
        if family not in self.FAMILIES:
            raise ValueError(f"family must be one of {self.FAMILIES}")
        if frequency_bins < 2 or initial_length_scale <= 0:
            raise ValueError("frequency_bins >= 2 and a positive length scale are required")
        self.family = family
        self.register_buffer("frequency", torch.arange(frequency_bins, dtype=torch.float32))
        self.raw_length_scale = nn.Parameter(
            torch.tensor(math.log(initial_length_scale), dtype=torch.float32))

    @property
    def length_scale(self) -> torch.Tensor:
        # Clamped only to keep the spectrum numerically representable on the
        # frequency grid; the range spans four orders of magnitude.
        return self.raw_length_scale.clamp(math.log(1e-2), math.log(1e2)).exp()

    def forward(self) -> torch.Tensor:
        scaled = (self.length_scale * self.frequency).square()
        if self.family == "rbf":
            spectrum = torch.exp(-0.5 * scaled)
        elif self.family == "matern12":
            spectrum = (1 + scaled).pow(-1.0)
        elif self.family == "matern32":
            spectrum = (1 + scaled).pow(-2.0)
        else:
            spectrum = (1 + scaled).pow(-3.0)
        return (spectrum / spectrum.sum().clamp_min(1e-12))[None]


def real_cosine_basis(coordinate: torch.Tensor, frequency_bins: int) -> torch.Tensor:
    """Neumann (no-flux) eigenbasis ``[1, sqrt(2) cos(pi k x)]``.

    The eigenbasis of a stationary operator depends on the boundary condition,
    not only on the symbol.  On a periodic domain it is the complex exponential
    pair, which :func:`real_fourier_basis` provides; on a no-flux domain the
    Laplacian eigenfunctions are cosines with eigenvalues ``(pi k)^2``.  Real
    initial-value data is typically *not* periodic along time -- forcing
    ``f(0) = f(1)`` there is a large, unphysical constraint -- so this basis is
    what makes the construction usable outside self-generated periodic data.

    The induced kernel is still PSD for any nonnegative spectrum, because it
    remains a nonnegatively weighted sum of outer products.  Unlike the
    periodic case the pointwise variance is not constant, which is correct: a
    bounded domain genuinely has a non-stationary covariance near its edges.
    """
    if coordinate.ndim != 1 or frequency_bins < 2:
        raise ValueError("coordinate/frequency_bins are invalid")
    frequency = torch.arange(1, frequency_bins, device=coordinate.device,
                             dtype=coordinate.dtype)
    phase = math.pi * coordinate[:, None] * frequency[None]
    return torch.cat((
        torch.ones(len(coordinate), 1, device=coordinate.device, dtype=coordinate.dtype),
        math.sqrt(2) * torch.cos(phase),
    ), dim=-1)


def normalize_spectrum_cosine(spectrum: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Normalize to unit *average* marginal variance in the cosine basis.

    Each cosine feature has mean square one over the domain, so the average
    pointwise variance is ``sum_k s(k)`` rather than the periodic
    ``s_0 + 2 sum_{k>0} s_k``.
    """
    if spectrum.ndim < 1 or spectrum.shape[-1] < 2:
        raise ValueError("spectrum must have at least two frequency bins")
    if torch.any(spectrum < 0) or not torch.isfinite(spectrum).all():
        raise ValueError("spectrum must be finite and nonnegative")
    return spectrum / spectrum.sum(-1, keepdim=True).clamp_min(eps)


def operator_joint_spectrum(
    operator: Literal["diffusion", "wave", "advection", "reaction_diffusion",
                      "banded_pattern", "narrowband_diffusion"],
    frequencies: torch.Tensor,
    *,
    source_scale: float = 0.12,
    reaction: float = 0.8,
    diffusion_coefficients: tuple[float, float, float] = (0.35, 1.1, 0.55),
    advection_diffusivity: tuple[float, float] = (0.18, 0.252),
    advection_velocity: tuple[float, float] = (0.9, -0.55),
    advection_reaction: float = 0.6,
    wave_coefficients: tuple[float, float] = (1.35, 0.65),
    wave_damping: tuple[float, float] = (0.45, 0.18),
    reaction_diffusivity: tuple[float, float] = (1.0, 1.0),
    reaction_rate: float = 0.0,
    reaction_damping: float = 0.15,
    band_wavenumber: float = 2.0,
    band_stiffness: float = 1.0,
    band_offset: float = 0.2,
    forcing_band_centre: float = 5.0,
    forcing_band_width: float = 1.2,
) -> torch.Tensor:
    """Construct ``|L_hat|^-2 S_w`` on a three-dimensional frequency grid.

    The axes are interpreted as ``omega_x, omega_y, omega_t``.  A small damping
    term makes the wave/advection responses finite; the construction is a
    periodic constant-coefficient mechanism sanity, not a boundary-aware PDE
    solver.
    """
    if frequencies.ndim != 1 or len(frequencies) < 3:
        raise ValueError("frequencies must be a nontrivial one-dimensional grid")
    wx, wy, wt = torch.meshgrid(frequencies, frequencies, frequencies, indexing="ij")
    source = torch.exp(-source_scale * (wx.square() + wy.square() + wt.square()))
    positive_parameters = (
        source_scale, reaction, *diffusion_coefficients,
        *advection_diffusivity, advection_reaction,
        *wave_coefficients, *wave_damping,
        *reaction_diffusivity, reaction_damping,
        band_stiffness, band_offset, forcing_band_width,
    )
    if any(value <= 0 for value in positive_parameters):
        raise ValueError("source, reaction and diffusivity parameters must be positive")
    if operator == "diffusion":
        dx, dy, dt = diffusion_coefficients
        response_sq = (reaction + dx * wx.square() + dy * wy.square() + dt * wt.square()).square()
    elif operator == "wave":
        # Defaults reproduce the original hard-coded literals exactly, so every
        # frozen result stays reproducible; the arguments exist so that a wave
        # *family* can be sampled for wrong-family controls.
        cx, cy = wave_coefficients
        gamma0, gamma1 = wave_damping
        dispersion = cx * (wx.square() + cy * wy.square()) - wt.square()
        response_sq = dispersion.square() + (gamma0 + gamma1 * wt.abs()).square()
    elif operator == "narrowband_diffusion":
        # A dissipative operator driven by temporally narrowband noise.  The
        # response is band-pass along time and low-pass along space, and --
        # unlike a wave -- the temporal band does not move with the spatial
        # mode, so the joint spectrum is close to axis-separable.  This is the
        # one structure a generic monotone kernel cannot follow while the
        # separable approximation the method needs still holds.
        dx, dy = reaction_diffusivity
        response_sq = (wt.square()
                       + (reaction + dx * wx.square() + dy * wy.square()).square())
        source = (torch.exp(-0.5 * ((wt - forcing_band_centre) / forcing_band_width).square())
                  * torch.exp(-source_scale * (wx.square() + wy.square())))
        spectrum = source / response_sq.clamp_min(1e-8)
        return spectrum / spectrum.sum().clamp_min(1e-12)
    elif operator == "banded_pattern":
        # Linear Swift-Hohenberg symbol: |L|^2 = wt^2 + (a + b(|k|^2 - k0^2)^2)^2.
        # Even in every axis, so the real cosine representation is exact, but
        # *not* axis-separable, so the nonnegative projection does real work.
        # Its solution spectrum peaks at |k| = k0 rather than at the origin,
        # which is the property a generic smooth dictionary cannot express: the
        # operator says *where* the energy sits, not merely that it decays.
        radial = wx.square() + wy.square() - band_wavenumber**2
        response_sq = wt.square() + (band_offset + band_stiffness * radial.square()).square()
    elif operator == "reaction_diffusion":
        # Parabolic symbol of  d_t u = D grad^2 u + a u  (+ higher-order terms
        # absorbed into the forcing).  Even in every axis, so the real
        # axis-wise Fourier representation is exact here rather than an
        # approximation.  With a > 0 the symbol dips near the Turing wavenumber
        # D k^2 = a, giving a band-pass spatial spectrum that a generic smooth
        # kernel cannot express -- the reason this family is the headline case.
        dx, dy = reaction_diffusivity
        elliptic = dx * wx.square() + dy * wy.square() - reaction_rate
        response_sq = wt.square() + elliptic.square() + reaction_damping**2
    elif operator == "advection":
        dx, dy = advection_diffusivity
        vx, vy = advection_velocity
        dissipative = advection_reaction + dx * wx.square() + dy * wy.square()
        transport = wt + vx * wx + vy * wy
        response_sq = dissipative.square() + transport.square()
    else:
        raise ValueError(f"unknown operator: {operator}")
    spectrum = source / response_sq.clamp_min(1e-8)
    return spectrum / spectrum.sum().clamp_min(1e-12)


@dataclass(frozen=True)
class SpectrumCP:
    weights: torch.Tensor
    factors: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    reconstruction: torch.Tensor
    relative_error: float


def nonnegative_cp_spectrum(
    spectrum: torch.Tensor,
    *,
    rank: int = 4,
    steps: int = 1000,
    seed: int = 0,
    eps: float = 1e-10,
) -> SpectrumCP:
    """Nonnegative CP decomposition by multiplicative Euclidean-loss updates.

    The output factors are normalized as one-sided spectra.  Their positive
    weights retain component scale.  This routine is intentionally small and
    deterministic so the projection from a joint operator spectrum is fully
    inspectable.
    """
    if spectrum.ndim != 3 or min(spectrum.shape) < 2:
        raise ValueError("spectrum must be a nontrivial order-three tensor")
    if rank < 1 or steps < 1 or torch.any(spectrum < 0):
        raise ValueError("rank/steps/spectrum are invalid")
    data = spectrum.detach().double().cpu().numpy()
    data = data / max(data.sum(), eps)
    rng = np.random.default_rng(seed)
    factors = [rng.random((size, rank)) + 0.2 for size in data.shape]
    weights = np.ones(rank)

    # Euclidean nonnegative CP multiplicative updates using matricized tensor
    # times Khatri-Rao products.  The order below matches NumPy C flattening.
    for _ in range(steps):
        for mode in range(3):
            others = [axis for axis in range(3) if axis != mode]
            unfolded = np.moveaxis(data, mode, 0).reshape(data.shape[mode], -1)
            kr = np.einsum("ir,jr->ijr", factors[others[0]], factors[others[1]]).reshape(-1, rank)
            numerator = unfolded @ kr
            gram = (factors[others[0]].T @ factors[others[0]]) * (
                factors[others[1]].T @ factors[others[1]]
            )
            denominator = factors[mode] @ gram + eps
            factors[mode] *= numerator / denominator
            factors[mode] = np.maximum(factors[mode], eps)
        # Move arbitrary component scales into weights to stabilize updates.
        weights = np.ones(rank)
        for mode in range(3):
            norms = np.linalg.norm(factors[mode], axis=0).clip(min=eps)
            factors[mode] /= norms
            weights *= norms

    reconstruction = np.einsum("r,ir,jr,kr->ijk", weights, *factors)
    relative_error = float(np.linalg.norm(data - reconstruction) / np.linalg.norm(data))
    torch_factors = tuple(
        torch.from_numpy(factor.T).float() for factor in factors
    )
    torch_weights = torch.from_numpy(weights).float()
    return SpectrumCP(
        weights=torch_weights,
        factors=torch_factors,  # type: ignore[arg-type]
        reconstruction=torch.from_numpy(reconstruction).float(),
        relative_error=relative_error,
    )


Routing = Literal["global", "per_mode", "per_mode_rank", "hierarchical", "fixed"]


class ModeAdaptiveVariationalCP(nn.Module):
    """Functional CP with routed finite-spectrum GP factors and an ELBO."""

    def __init__(
        self,
        coordinates: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        spectra: torch.Tensor,
        *,
        rank: int = 2,
        routing: Routing = "per_mode_rank",
        fixed_routing: torch.Tensor | None = None,
        noise_std: float = 0.08,
        mode_deviation_std: float = 0.35,
        mixture_parameterization: Literal["expanded", "collapsed"] = "expanded",
        routing_floor: torch.Tensor | None = None,
    ):
        super().__init__()
        if spectra.ndim != 3 or spectra.shape[0] != 3:
            raise ValueError("spectra must have shape [3,family,frequency]")
        if rank < 1 or routing not in {"global", "per_mode", "per_mode_rank", "hierarchical", "fixed"}:
            raise ValueError("invalid rank or routing")
        if mixture_parameterization not in {"expanded", "collapsed"}:
            raise ValueError("mixture_parameterization must be expanded or collapsed")
        if mode_deviation_std <= 0:
            raise ValueError("mode_deviation_std must be positive")
        self.rank = rank
        self.family_count = spectra.shape[1]
        self.routing = routing
        self.mixture_parameterization = mixture_parameterization
        self.mode_deviation_std = float(mode_deviation_std)
        features = [fourier_features(coordinates[d], spectra[d]) for d in range(3)]
        feature_size = features[0].shape[-1]
        if any(value.shape[-1] != feature_size for value in features):
            raise ValueError("all modes must use the same frequency budget")
        self.feature_size = feature_size
        self.register_buffer("spectra", normalize_spectrum(spectra.float()))
        if routing_floor is None:
            floor = torch.zeros(self.family_count, dtype=spectra.dtype, device=spectra.device)
        else:
            floor = routing_floor.to(device=spectra.device, dtype=spectra.dtype)
            if floor.shape != (self.family_count,) or torch.any(floor < 0) or floor.sum() >= 1:
                raise ValueError("routing_floor must be nonnegative [family] with sum < 1")
        self.register_buffer("routing_floor", floor)
        for mode, value in enumerate(features):
            self.register_buffer(f"features_{mode}", value)
            self.register_buffer(
                f"fourier_basis_{mode}",
                real_fourier_basis(coordinates[mode], spectra.shape[-1]),
            )

        # ``expanded`` preserves the first POC exactly: every atom owns an
        # independent coefficient vector.  ``collapsed`` uses the canonical
        # finite representation of a spectral-mixture GP: first form
        # S_mix=sum_q pi_q S_q, then use one coefficient vector.  The latter
        # makes the variational coefficient budget independent of bank size.
        shape = ((3, rank, self.family_count, feature_size)
                 if mixture_parameterization == "expanded"
                 else (3, rank, feature_size))
        self.variational_mean = nn.Parameter(0.12 * torch.randn(shape))
        self.raw_variational_std = nn.Parameter(torch.full(shape, -2.5))
        self.core = nn.Parameter(torch.ones(rank) / math.sqrt(rank))
        self.log_noise_std = nn.Parameter(torch.tensor(math.log(noise_std)))

        if routing == "global":
            self.routing_logits = nn.Parameter(torch.zeros(self.family_count))
        elif routing == "per_mode":
            self.routing_logits = nn.Parameter(torch.zeros(3, self.family_count))
        elif routing == "per_mode_rank":
            self.routing_logits = nn.Parameter(torch.zeros(3, rank, self.family_count))
        elif routing == "hierarchical":
            # A global dictionary weight is the statistically stable anchor.
            # Shrunk mode deviations can move away from it only when the ELBO
            # provides enough evidence.  Ranks share a route at this bridge
            # stage to avoid an unnecessary 1% identifiability burden.
            self.routing_logits = nn.Parameter(torch.zeros(self.family_count))
            self.mode_deviation = nn.Parameter(torch.zeros(3, self.family_count))
        else:
            if fixed_routing is None or fixed_routing.shape != (3, rank, self.family_count):
                raise ValueError("fixed routing requires [3,rank,family] weights")
            fixed = fixed_routing.float()
            if torch.any(fixed < 0) or torch.any(fixed.sum(-1) <= 0):
                raise ValueError("fixed routing must be nonnegative and nonempty")
            self.register_buffer("fixed_routing", fixed / fixed.sum(-1, keepdim=True))
            self.register_parameter("routing_logits", None)

    @property
    def noise_std(self) -> torch.Tensor:
        return self.log_noise_std.clamp(math.log(0.01), math.log(0.5)).exp()

    def routing_weights(self) -> torch.Tensor:
        if self.routing == "fixed":
            return self.fixed_routing
        def add_floor(value: torch.Tensor) -> torch.Tensor:
            return self.routing_floor + (1 - self.routing_floor.sum()) * value

        weights = add_floor(torch.softmax(self.routing_logits, dim=-1))
        if self.routing == "global":
            return weights.expand(3, self.rank, -1)
        if self.routing == "hierarchical":
            weights = add_floor(torch.softmax(
                self.routing_logits[None] + self.mode_deviation, dim=-1,
            ))
            return weights[:, None].expand(-1, self.rank, -1)
        if self.routing == "per_mode":
            return weights[:, None].expand(-1, self.rank, -1)
        return weights

    def variational_std(self) -> torch.Tensor:
        return F.softplus(self.raw_variational_std) + 1e-4

    def induced_spectra(self) -> torch.Tensor:
        """Return the routed one-sided prior spectrum for every mode/rank."""
        return torch.einsum("drq,dqk->drk", self.routing_weights(), self.spectra)

    def _collapsed_features(self, mode: int, node_index: torch.Tensor) -> torch.Tensor:
        """Canonical features of the routed mixture, shaped [entry,rank,feature]."""
        mixed = self.induced_spectra()[mode]
        # Exact zero support is allowed (and is required by the strict
        # mismatch control), but sqrt has an infinite derivative at zero.
        # Clamping only for the feature construction keeps the intended zero
        # covariance to numerical precision and prevents NaN routing gradients.
        root = mixed.clamp_min(1e-12).sqrt()
        amplitude = torch.cat((root[:, :1], root[:, 1:], root[:, 1:]), dim=-1)
        basis = getattr(self, f"fourier_basis_{mode}")
        return basis[node_index.long(), None, :] * amplitude[None]

    def kl_to_prior(self) -> torch.Tensor:
        mean, std = self.variational_mean, self.variational_std()
        kl = 0.5 * (mean.square() + std.square() - 1 - 2 * torch.log(std)).sum()
        if self.routing == "hierarchical":
            kl = kl + 0.5 * self.mode_deviation.square().sum() / self.mode_deviation_std**2
        return kl

    def _prediction_from_coefficients(self, indices: torch.Tensor, coefficients: torch.Tensor) -> torch.Tensor:
        if indices.ndim != 2 or indices.shape[1] != 3:
            raise ValueError("indices must have shape [entry,3]")
        factors = []
        for mode in range(3):
            if self.mixture_parameterization == "expanded":
                feature = getattr(self, f"features_{mode}")[:, indices[:, mode].long(), :]
                value = torch.einsum(
                    "qnf,rqf,rq->nr", feature, coefficients[mode],
                    self.routing_weights()[mode].sqrt(),
                )
            else:
                feature = self._collapsed_features(mode, indices[:, mode])
                value = torch.einsum("nrf,rf->nr", feature, coefficients[mode])
            factors.append(value)
        return torch.einsum("nr,nr,nr,r->n", *factors, self.core)

    def posterior_mean(self, indices: torch.Tensor) -> torch.Tensor:
        return self._prediction_from_coefficients(indices, self.variational_mean)

    def posterior_moments(self, indices: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Exact marginal latent mean/variance under the mean-field posterior."""
        factor_means, factor_variances = [], []
        std = self.variational_std()
        for mode in range(3):
            if self.mixture_parameterization == "expanded":
                feature = getattr(self, f"features_{mode}")[:, indices[:, mode].long(), :]
                weighted = feature[None] * self.routing_weights()[mode, :, :, None, None].sqrt()
                weighted = weighted.permute(0, 2, 1, 3)
                mean = torch.einsum("rnqf,rqf->nr", weighted, self.variational_mean[mode])
                variance = torch.einsum("rnqf,rqf->nr", weighted.square(), std[mode].square())
            else:
                feature = self._collapsed_features(mode, indices[:, mode])
                mean = torch.einsum("nrf,rf->nr", feature, self.variational_mean[mode])
                variance = torch.einsum("nrf,rf->nr", feature.square(), std[mode].square())
            factor_means.append(mean)
            factor_variances.append(variance)
        product_mean = factor_means[0] * factor_means[1] * factor_means[2]
        product_second = torch.ones_like(product_mean)
        for mean, variance in zip(factor_means, factor_variances):
            product_second = product_second * (mean.square() + variance)
        mean = torch.einsum("nr,r->n", product_mean, self.core)
        variance = torch.einsum(
            "nr,r->n", (product_second - product_mean.square()).clamp_min(0),
            self.core.square(),
        )
        return mean, variance.clamp_min(1e-12)

    def posterior_predictive_samples(
        self,
        indices: torch.Tensor,
        *,
        samples: int,
        generator: torch.Generator,
        include_noise: bool = True,
    ) -> torch.Tensor:
        """Monte-Carlo samples from q(f) or q(f)p(y|f)."""
        if samples < 1:
            raise ValueError("samples must be positive")
        std = self.variational_std()
        draws = []
        for _ in range(samples):
            epsilon = torch.randn(
                std.shape, generator=generator, device=std.device, dtype=std.dtype,
            )
            value = self._prediction_from_coefficients(
                indices, self.variational_mean + std * epsilon,
            )
            if include_noise:
                value = value + self.noise_std * torch.randn(
                    value.shape, generator=generator, device=value.device,
                    dtype=value.dtype,
                )
            draws.append(value)
        return torch.stack(draws)

    def negative_elbo(
        self,
        indices: torch.Tensor,
        targets: torch.Tensor,
        *,
        total_count: int,
        samples: int = 3,
        kl_weight: float = 1.0,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if len(indices) != len(targets) or total_count < len(targets) or samples < 1:
            raise ValueError("invalid ELBO batch")
        std = self.variational_std()
        predictions = []
        for _ in range(samples):
            coefficients = self.variational_mean + std * torch.randn_like(std)
            predictions.append(self._prediction_from_coefficients(indices, coefficients))
        prediction = torch.stack(predictions)
        noise_variance = self.noise_std.square()
        expected_log_likelihood = -0.5 * (
            math.log(2 * math.pi) + torch.log(noise_variance)
            + (targets[None] - prediction).square() / noise_variance
        ).mean(0).sum() * (total_count / len(targets))
        kl = self.kl_to_prior()
        loss = -(expected_log_likelihood - kl_weight * kl) / total_count
        return loss, {
            "expected_log_likelihood": expected_log_likelihood.detach(),
            "kl": kl.detach(),
            "noise_std": self.noise_std.detach(),
        }


def all_grid_indices(shape: tuple[int, int, int], device: torch.device | str = "cpu") -> torch.Tensor:
    axes = [torch.arange(size, device=device) for size in shape]
    return torch.cartesian_prod(*axes)


@torch.no_grad()
def sample_planted_tensor(
    coordinates: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    spectra: torch.Tensor,
    routing: torch.Tensor,
    *,
    seed: int,
    core: torch.Tensor | None = None,
) -> torch.Tensor:
    """Sample a tensor from the same routed finite GP prior for sanity checks."""
    generator = torch.Generator(device=coordinates[0].device).manual_seed(seed)
    rank = routing.shape[1]
    features = [fourier_features(coordinates[d], spectra[d]) for d in range(3)]
    feature_size = features[0].shape[-1]
    coefficients = torch.randn(
        3, rank, spectra.shape[1], feature_size,
        generator=generator, device=coordinates[0].device,
    )
    factors = []
    for mode in range(3):
        value = torch.einsum(
            "qnf,rqf,rq->nr",
            features[mode], coefficients[mode], routing[mode].sqrt(),
        )
        factors.append(value)
    if core is None:
        core = torch.linspace(1.0, 0.65, rank, device=coordinates[0].device)
    field = torch.einsum("ir,jr,kr,r->ijk", *factors, core)
    return (field - field.mean()) / field.std().clamp_min(1e-6)


class ModeAdaptiveVariationalTucker(nn.Module):
    """Functional Tucker with routed finite-spectrum GP factors and an ELBO.

    Same spectral machinery as :class:`ModeAdaptiveVariationalCP` --- the
    operator-derived per-mode kernels are unchanged --- but the diagonal CP
    weight is replaced by a small dense core.  This exists because real
    two-dimensional physical fields are generally *not* low CP-rank: a Turing
    pattern is a field of isotropic blobs, which is not ``x`` tensor ``y``
    separable, yet its multilinear rank is small.  With a CP host no kernel can
    help, because the model cannot represent the field at all.

    Per-mode ranks are independent on purpose.  A field whose multilinear rank
    is ``[2, 11, 11]`` needs a ``2*11*11 = 242`` core, whereas forcing an equal
    rank of 11 would need ``1331`` --- a difference that decides whether the
    model is identifiable at realistic observation counts.
    """

    def __init__(
        self,
        coordinates: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        spectra: torch.Tensor | list[torch.Tensor] | tuple[torch.Tensor, ...],
        *,
        ranks: tuple[int, int, int],
        routing: Literal["global", "per_mode", "per_mode_rank", "fixed"] = "per_mode_rank",
        fixed_routing: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
        noise_std: float = 0.08,
        routing_floor: torch.Tensor | None = None,
        basis: tuple[str, str, str] = ("fourier", "fourier", "fourier"),
        eigenbasis: tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]
        | None = None,
    ):
        super().__init__()
        # Spectra may be one stacked [3, family, frequency] tensor when every
        # mode uses the same frequency budget, or three separate [family,
        # frequency_d] tensors when they do not.  The latter matters on real
        # data: a shared budget forces every mode down to whatever the
        # shortest one can support, which on a 15 x 95 x 267 tensor caps the
        # 267-point time mode at 15 basis functions.
        learnable = [s for s in (spectra if isinstance(spectra, (list, tuple)) else [])
                     if isinstance(s, LearnableStationarySpectrum)]
        if learnable:
            if len(spectra) != 3 or len(learnable) != 3:
                raise ValueError("learnable spectra must be supplied for all three modes")
            self.learnable_spectra = nn.ModuleList(spectra)
            per_mode = [module() for module in spectra]
        elif isinstance(spectra, (list, tuple)):
            if len(spectra) != 3 or any(s.ndim != 2 for s in spectra):
                raise ValueError("per-mode spectra must be three [family,frequency] tensors")
            if len({s.shape[0] for s in spectra}) != 1:
                raise ValueError("all modes must offer the same number of atoms")
            self.learnable_spectra = None
            per_mode = [s.float() for s in spectra]
        else:
            self.learnable_spectra = None
            if spectra.ndim != 3 or spectra.shape[0] != 3:
                raise ValueError("spectra must have shape [3,family,frequency]")
            per_mode = [spectra[mode].float() for mode in range(3)]
        if len(ranks) != 3 or any(rank < 1 for rank in ranks):
            raise ValueError("ranks must be three positive integers")
        if routing not in {"global", "per_mode", "per_mode_rank", "fixed"}:
            raise ValueError(f"unknown routing: {routing}")
        self.ranks = tuple(int(rank) for rank in ranks)
        self.family_count = per_mode[0].shape[0]
        self.routing = routing
        if len(basis) != 3 or any(kind not in {"fourier", "cosine", "operator"} for kind in basis):
            raise ValueError("basis must be three entries from {fourier, cosine, operator}")
        if any(kind == "operator" for kind in basis) and eigenbasis is None:
            raise ValueError("an 'operator' basis requires the eigenvectors in `eigenbasis`")
        self.basis = tuple(basis)
        self.frequency_bins = tuple(s.shape[-1] for s in per_mode)
        # Each mode normalizes its spectrum for the basis it actually uses.
        for mode in range(3):
            normalizer = (normalize_spectrum if self.basis[mode] == "fourier"
                          else normalize_spectrum_cosine)
            self.register_buffer(f"spectra_{mode}", normalizer(per_mode[mode]))
        for mode, coordinate in enumerate(coordinates):
            frequency_bins = self.frequency_bins[mode]
            if self.basis[mode] == "operator":
                # The eigenfunctions of the operator itself.  For a
                # constant-coefficient Neumann Laplacian these *are* the
                # cosines, so `cosine` is the special case; a variable
                # coefficient, an irregular mesh or a barrier changes them, and
                # the construction is otherwise unchanged.  This is the exact
                # Mercer feature map of the induced kernel, not an
                # approximation to it.
                vectors = eigenbasis[mode]
                if vectors is None or vectors.ndim != 2:
                    raise ValueError(f"eigenbasis[{mode}] must be [node, mode]")
                if vectors.shape[0] != len(coordinate):
                    raise ValueError(f"eigenbasis[{mode}] has the wrong node count")
                if vectors.shape[1] != frequency_bins:
                    raise ValueError(
                        f"eigenbasis[{mode}] must supply {frequency_bins} modes")
                self.register_buffer(f"fourier_basis_{mode}", vectors.float().clone())
            else:
                builder = (real_fourier_basis if self.basis[mode] == "fourier"
                           else real_cosine_basis)
                self.register_buffer(f"fourier_basis_{mode}",
                                     builder(coordinate, frequency_bins))

        if routing_floor is None:
            floor = torch.zeros(self.family_count, dtype=torch.float32)
        else:
            floor = routing_floor.to(dtype=torch.float32)
            if floor.shape != (self.family_count,) or torch.any(floor < 0) or floor.sum() >= 1:
                raise ValueError("routing_floor must be nonnegative [family] with sum < 1")
        self.register_buffer("routing_floor", floor)

        # One coefficient vector per (mode, mode-rank); the count is independent
        # of the number of atoms in the bank, exactly as in the CP host, so
        # banks of different sizes stay comparable.
        self.feature_sizes = tuple(
            (1 + 2 * (self.frequency_bins[mode] - 1)) if self.basis[mode] == "fourier"
            else self.frequency_bins[mode] for mode in range(3)
        )
        self.variational_mean = nn.ParameterList([
            nn.Parameter(0.12 * torch.randn(rank, size))
            for rank, size in zip(self.ranks, self.feature_sizes)
        ])
        self.raw_variational_std = nn.ParameterList([
            nn.Parameter(torch.full((rank, size), -2.5))
            for rank, size in zip(self.ranks, self.feature_sizes)
        ])
        core = torch.randn(*self.ranks) / math.sqrt(float(np.prod(self.ranks)))
        self.core = nn.Parameter(core)
        self.log_noise_std = nn.Parameter(torch.tensor(math.log(noise_std)))

        if routing == "fixed":
            if fixed_routing is None or len(fixed_routing) != 3:
                raise ValueError("fixed routing requires three [rank,family] tensors")
            for mode, value in enumerate(fixed_routing):
                if value.shape != (self.ranks[mode], self.family_count):
                    raise ValueError("fixed routing shape mismatch")
                if torch.any(value < 0) or torch.any(value.sum(-1) <= 0):
                    raise ValueError("fixed routing must be nonnegative and nonempty")
                self.register_buffer(
                    f"fixed_routing_{mode}", value.float() / value.float().sum(-1, keepdim=True),
                )
            self.routing_logits = None
        elif routing == "global":
            self.routing_logits = nn.Parameter(torch.zeros(self.family_count))
        elif routing == "per_mode":
            self.routing_logits = nn.Parameter(torch.zeros(3, self.family_count))
        else:
            self.routing_logits = nn.ParameterList([
                nn.Parameter(torch.zeros(rank, self.family_count)) for rank in self.ranks
            ])

    @property
    def noise_std(self) -> torch.Tensor:
        return self.log_noise_std.clamp(math.log(0.01), math.log(0.5)).exp()

    def _add_floor(self, value: torch.Tensor) -> torch.Tensor:
        return self.routing_floor + (1 - self.routing_floor.sum()) * value

    def routing_weights(self) -> list[torch.Tensor]:
        """Per mode, a ``[rank, family]`` simplex of atom weights."""
        if self.routing == "fixed":
            return [getattr(self, f"fixed_routing_{mode}") for mode in range(3)]
        if self.routing == "global":
            weight = self._add_floor(torch.softmax(self.routing_logits, dim=-1))
            return [weight[None].expand(rank, -1) for rank in self.ranks]
        if self.routing == "per_mode":
            weight = self._add_floor(torch.softmax(self.routing_logits, dim=-1))
            return [weight[mode][None].expand(self.ranks[mode], -1) for mode in range(3)]
        return [
            self._add_floor(torch.softmax(self.routing_logits[mode], dim=-1))
            for mode in range(3)
        ]

    def induced_spectra(self) -> list[torch.Tensor]:
        weights = self.routing_weights()
        if self.learnable_spectra is not None:
            # Recomputed every call so the length scales receive gradients.
            return [weights[mode] @ self.learnable_spectra[mode]() for mode in range(3)]
        return [weights[mode] @ getattr(self, f"spectra_{mode}") for mode in range(3)]

    def variational_std(self) -> list[torch.Tensor]:
        return [F.softplus(raw) + 1e-4 for raw in self.raw_variational_std]

    def _collapsed_features(self, mode: int, node_index: torch.Tensor) -> torch.Tensor:
        """``[entry, rank, feature]`` features of the routed mixture spectrum."""
        mixed = self.induced_spectra()[mode]
        # Exact zero support must stay representable; sqrt has an infinite
        # derivative there, so clamp only while building features.
        root = mixed.clamp_min(1e-12).sqrt()
        if self.basis[mode] == "fourier":
            amplitude = torch.cat((root[:, :1], root[:, 1:], root[:, 1:]), dim=-1)
        else:
            amplitude = root
        basis = getattr(self, f"fourier_basis_{mode}")
        return basis[node_index.long(), None, :] * amplitude[None]

    def _factor_values(
        self, indices: torch.Tensor, coefficients: list[torch.Tensor],
    ) -> list[torch.Tensor]:
        if indices.ndim != 2 or indices.shape[1] != 3:
            raise ValueError("indices must have shape [entry,3]")
        return [
            (self._collapsed_features(mode, indices[:, mode]) * coefficients[mode][None]).sum(-1)
            for mode in range(3)
        ]

    def _contract(self, factors: list[torch.Tensor]) -> torch.Tensor:
        partial = torch.einsum("na,abc->nbc", factors[0], self.core)
        partial = torch.einsum("nb,nbc->nc", factors[1], partial)
        return (factors[2] * partial).sum(-1)

    def kl_to_prior(self) -> torch.Tensor:
        total = self.core.new_zeros(())
        for mean, std in zip(self.variational_mean, self.variational_std()):
            total = total + 0.5 * (mean.square() + std.square() - 1 - 2 * torch.log(std)).sum()
        return total

    def posterior_mean(self, indices: torch.Tensor) -> torch.Tensor:
        return self._contract(self._factor_values(indices, list(self.variational_mean)))

    def posterior_predictive_samples(
        self, indices: torch.Tensor, *, samples: int,
        generator: torch.Generator | None = None, include_noise: bool = False,
    ) -> torch.Tensor:
        stds = self.variational_std()
        draws = []
        for _ in range(samples):
            coefficients = [
                mean + std * torch.randn(
                    mean.shape, generator=generator, device=mean.device, dtype=mean.dtype)
                for mean, std in zip(self.variational_mean, stds)
            ]
            draws.append(self._contract(self._factor_values(indices, coefficients)))
        stacked = torch.stack(draws)
        if include_noise:
            stacked = stacked + self.noise_std * torch.randn(
                stacked.shape, generator=generator, device=stacked.device, dtype=stacked.dtype)
        return stacked

    def negative_elbo(
        self, indices: torch.Tensor, targets: torch.Tensor, *,
        total_count: int, samples: int = 3,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if len(indices) != len(targets) or total_count < len(targets):
            raise ValueError("indices/targets/total_count are inconsistent")
        stds = self.variational_std()
        noise = self.noise_std
        expected = self.core.new_zeros(())
        for _ in range(samples):
            coefficients = [
                mean + std * torch.randn_like(mean) for mean, std in zip(self.variational_mean, stds)
            ]
            prediction = self._contract(self._factor_values(indices, coefficients))
            expected = expected + (
                -0.5 * math.log(2 * math.pi) - torch.log(noise)
                - 0.5 * (targets - prediction).square() / noise.square()
            ).sum()
        expected = expected / samples * (total_count / len(targets))
        kl = self.kl_to_prior()
        loss = (kl - expected) / total_count
        if not torch.isfinite(loss):
            raise FloatingPointError("non-finite ELBO")
        return loss, {"kl": kl.detach(), "noise_std": noise.detach()}
