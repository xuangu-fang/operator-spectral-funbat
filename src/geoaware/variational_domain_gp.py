"""Finite-feature variational GPs for irregular-domain field regression.

This module intentionally keeps the Bayesian object explicit.  A fixed feature
map ``phi(x)`` defines the finite-rank kernel ``k(x, x') = phi(x)^T phi(x')``.
The whitened feature coefficients are inter-domain inducing variables
``u ~ N(0, I)`` and inference optimizes a full-covariance Gaussian ``q(u)`` by
the mini-batch Gaussian ELBO.  No neural network weights are called a GP.
"""

from __future__ import annotations

from collections.abc import Sequence
import math

import torch
from torch import nn
from torch.nn import functional as F


def parameter_rbf_features(
    parameters: torch.Tensor,
    *,
    centers: int = 7,
    lengthscale: float = 0.35,
) -> torch.Tensor:
    """Return target-independent RBF features of positive scalar parameters."""
    if parameters.ndim != 1 or len(parameters) < 2:
        raise ValueError("parameters must be a one-dimensional nontrivial grid")
    if centers < 2 or lengthscale <= 0:
        raise ValueError("centers must be >=2 and lengthscale must be positive")
    log_parameter = torch.log(parameters.float().clamp_min(1e-8))
    lo, hi = log_parameter.min(), log_parameter.max()
    coordinate = (log_parameter - lo) / (hi - lo).clamp_min(1e-8)
    locations = torch.linspace(0, 1, centers, device=parameters.device)
    features = torch.exp(
        -(coordinate[:, None] - locations[None]).square()
        / (2 * lengthscale ** 2)
    )
    rms = features.square().mean(0, keepdim=True).sqrt().clamp_min(1e-6)
    return features / rms


def tensor_product_gp_features(
    kernel_sections: torch.Tensor,
    parameters: torch.Tensor,
    indices: torch.Tensor,
    *,
    parameter_centers: int = 7,
    parameter_lengthscale: float = 0.35,
) -> torch.Tensor:
    """Build ``phi(s,a,x)=z_Omega(s,x) tensor-product psi(a)``.

    ``kernel_sections`` has shape ``[source, node, channel]``.  Intrinsic and
    Euclidean controls use this identical construction and dimensional budget.
    Scaling by the square root of the feature count keeps the prior marginal
    variance numerically stable.
    """
    if kernel_sections.ndim != 3:
        raise ValueError("kernel_sections must have shape [source,node,channel]")
    if indices.ndim != 2 or indices.shape[1] != 3:
        raise ValueError("indices must have shape [entry,3]")
    source, parameter, node = indices.long().T
    sections = kernel_sections[source, node]
    parameter_features = parameter_rbf_features(
        parameters,
        centers=parameter_centers,
        lengthscale=parameter_lengthscale,
    )[parameter]
    features = torch.einsum("nc,np->ncp", sections, parameter_features)
    features = features.flatten(1)
    return features / math.sqrt(features.shape[1])


class FiniteFeatureVariationalGP(nn.Module):
    """Full-covariance ``q(u)`` optimized with the mini-batch Gaussian ELBO."""

    def __init__(self, num_features: int, *, noise_std: float = 0.1):
        super().__init__()
        if num_features < 1 or noise_std <= 0:
            raise ValueError("num_features and noise_std must be positive")
        self.num_features = int(num_features)
        self.variational_mean = nn.Parameter(torch.zeros(num_features))
        raw = torch.zeros(num_features, num_features)
        initial_diagonal = math.log(math.expm1(1.0 - 1e-4))
        raw.diagonal().fill_(initial_diagonal)
        self.raw_cholesky = nn.Parameter(raw)
        self.log_noise_std = nn.Parameter(torch.tensor(math.log(noise_std)))

    @property
    def noise_std(self) -> torch.Tensor:
        return self.log_noise_std.clamp(math.log(1e-3), math.log(2.0)).exp()

    def covariance_cholesky(self) -> torch.Tensor:
        lower = torch.tril(self.raw_cholesky, diagonal=-1)
        diagonal = F.softplus(self.raw_cholesky.diagonal()) + 1e-4
        return lower + torch.diag(diagonal)

    def kl_to_prior(self) -> torch.Tensor:
        """KL[q(u)||N(0,I)] for a full-covariance Gaussian q."""
        cholesky = self.covariance_cholesky()
        trace = cholesky.square().sum()
        squared_mean = self.variational_mean.square().sum()
        log_determinant = 2 * torch.log(cholesky.diagonal()).sum()
        return 0.5 * (
            trace + squared_mean - self.num_features - log_determinant
        )

    def latent_moments(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if features.ndim != 2 or features.shape[1] != self.num_features:
            raise ValueError("feature dimension does not match GP")
        cholesky = self.covariance_cholesky()
        mean = features @ self.variational_mean
        variance = (features @ cholesky).square().sum(1).clamp_min(1e-12)
        return mean, variance


    def negative_elbo(
        self,
        features: torch.Tensor,
        targets: torch.Tensor,
        *,
        total_count: int,
        mean_offset: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Return negative mini-batch ELBO and detached diagnostics.

        The likelihood term is scaled by ``N / batch_size``; the KL is included
        exactly once, so uniformly sampled batches provide an unbiased ELBO
        estimator.
        """
        if len(features) != len(targets) or total_count < len(targets):
            raise ValueError("invalid batch or total_count")
        mean, latent_variance = self.latent_moments(features)
        if mean_offset is not None:
            if mean_offset.shape != mean.shape:
                raise ValueError("mean_offset must match the batch shape")
            mean = mean + mean_offset
        noise_variance = self.noise_std.square()
        expected_log_likelihood = -0.5 * (
            math.log(2 * math.pi)
            + torch.log(noise_variance)
            + ((targets - mean).square() + latent_variance) / noise_variance
        ).sum()
        expected_log_likelihood *= total_count / len(targets)
        kl = self.kl_to_prior()
        loss = -(expected_log_likelihood - kl) / total_count
        return loss, {
            "expected_log_likelihood": expected_log_likelihood.detach(),
            "kl": kl.detach(),
            "noise_std": self.noise_std.detach(),
        }

    @torch.no_grad()
    def predict(
        self, features: torch.Tensor, *, include_noise: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mean, variance = self.latent_moments(features)
        if include_noise:
            variance = variance + self.noise_std.square()
        return mean, variance


class NonnegativeKernelMixture(nn.Module):
    """Simplex-weighted sum of finite kernels with an auditable feature map.

    Given family features ``Phi_q``, this module returns the concatenation
    ``[sqrt(w_q) Phi_q]_q`` where ``w=softmax(logits)``.  With the standard
    Gaussian coefficient prior this is exactly the PSD kernel
    ``sum_q w_q Phi_q Phi_q^T``.  The logits may be optimized jointly with the
    variational posterior by the same ELBO; this is evidence-based kernel
    selection, not a discrete model-selection claim.
    """

    def __init__(self, family_names: Sequence[str]):
        super().__init__()
        if not family_names or len(set(family_names)) != len(family_names):
            raise ValueError("family_names must be nonempty and unique")
        self.family_names = tuple(family_names)
        self.logits = nn.Parameter(torch.zeros(len(self.family_names)))

    def weights(self) -> torch.Tensor:
        return torch.softmax(self.logits, dim=0)

    def forward(self, features: dict[str, torch.Tensor]) -> torch.Tensor:
        if tuple(features) != self.family_names:
            raise ValueError("feature dictionary order/names do not match mixture")
        rows = {len(value) for value in features.values()}
        if len(rows) != 1 or any(value.ndim != 2 for value in features.values()):
            raise ValueError("all family features must be 2D with equal row count")
        weights = self.weights()
        return torch.cat([
            features[name] * weights[index].sqrt()
            for index, name in enumerate(self.family_names)
        ], dim=1)

    def weight_dict(self) -> dict[str, float]:
        values = self.weights().detach().cpu().tolist()
        return dict(zip(self.family_names, values, strict=True))


@torch.no_grad()
def exact_finite_gp_posterior(
    features: torch.Tensor,
    targets: torch.Tensor,
    noise_std: float | torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Exact Gaussian posterior over the same whitened feature coefficients."""
    if features.ndim != 2 or len(features) != len(targets):
        raise ValueError("features/targets have incompatible shapes")
    noise = torch.as_tensor(noise_std, dtype=features.dtype, device=features.device)
    if noise.numel() != 1 or noise <= 0:
        raise ValueError("noise_std must be a positive scalar")
    dimension = features.shape[1]
    precision = torch.eye(dimension, device=features.device, dtype=features.dtype)
    precision = precision + features.T @ features / noise.square()
    cholesky = torch.linalg.cholesky(precision)
    rhs = features.T @ targets / noise.square()
    mean = torch.cholesky_solve(rhs[:, None], cholesky).squeeze(1)
    covariance = torch.cholesky_inverse(cholesky)
    return mean, covariance


@torch.no_grad()
def exact_finite_gp_predict(
    features: torch.Tensor,
    posterior_mean: torch.Tensor,
    posterior_covariance: torch.Tensor,
    *,
    noise_std: float | torch.Tensor = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Predictive mean and variance for an exact finite-feature GP posterior."""
    mean = features @ posterior_mean
    variance = torch.einsum(
        "ni,ij,nj->n", features, posterior_covariance, features
    )
    noise = torch.as_tensor(noise_std, dtype=features.dtype, device=features.device)
    return mean, (variance + noise.square()).clamp_min(1e-12)
