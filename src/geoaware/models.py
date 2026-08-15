"""Baselines and the two geometry-aware tensor proposals."""

from __future__ import annotations

import math
from typing import Sequence

import torch
from torch import nn
import torch.nn.functional as F

from .bases import BasisSpec, evaluate_basis


class FieldModel(nn.Module):
    is_bayesian = False

    def forward(self, coords: torch.Tensor, indices: torch.Tensor | None = None,
                sample: bool = True) -> torch.Tensor:
        raise NotImplementedError

    def regularization(self) -> torch.Tensor:
        return next(self.parameters()).new_zeros(())

    def kl_divergence(self) -> torch.Tensor:
        return next(self.parameters()).new_zeros(())


class SineLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int, omega: float,
                 first: bool = False):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.omega = omega
        with torch.no_grad():
            bound = 1 / in_features if first else math.sqrt(6 / in_features) / omega
            self.linear.weight.uniform_(-bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.omega * self.linear(x))


class SirenINR(FieldModel):
    """Strong coordinate-INR baseline using the published SIREN initialization."""

    def __init__(self, ndim: int, hidden: int = 128, depth: int = 3, omega: float = 20.0):
        super().__init__()
        layers: list[nn.Module] = [SineLayer(ndim, hidden, omega, first=True)]
        layers += [SineLayer(hidden, hidden, omega) for _ in range(depth - 1)]
        layers.append(nn.Linear(hidden, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, coords, indices=None, sample=True):
        return self.net(2 * coords - 1).squeeze(-1)


class FourierINR(FieldModel):
    """NeRF-style Fourier-feature MLP baseline with no boundary semantics."""

    def __init__(self, ndim: int, n_freq: int = 10, hidden: int = 128, depth: int = 3):
        super().__init__()
        self.register_buffer("freq", 2.0 ** torch.arange(n_freq).float())
        in_dim = ndim * (1 + 2 * n_freq)
        layers: list[nn.Module] = [nn.Linear(in_dim, hidden), nn.GELU()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), nn.GELU()]
        layers.append(nn.Linear(hidden, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, coords, indices=None, sample=True):
        phase = 2 * math.pi * coords[..., None] * self.freq
        enc = torch.cat([coords, phase.sin().flatten(1), phase.cos().flatten(1)], -1)
        return self.net(enc).squeeze(-1)


class DiscreteCP(FieldModel):
    """Classical CP completion baseline with discrete factor tables."""

    def __init__(self, shape: Sequence[int], rank: int):
        super().__init__()
        self.factors = nn.ParameterList([nn.Parameter(torch.randn(n, rank) * 0.1) for n in shape])
        self.weight = nn.Parameter(torch.ones(rank) / rank)

    def forward(self, coords, indices=None, sample=True):
        if indices is None:
            raise ValueError("DiscreteCP requires integer grid indices")
        prod = torch.ones(len(indices), len(self.weight), device=indices.device)
        for d, factor in enumerate(self.factors):
            prod = prod * factor[indices[:, d]]
        return (prod * self.weight).sum(-1)

    def regularization(self):
        return sum(p.square().mean() for p in self.factors) + self.weight.square().mean()


class ScalarFactorNet(nn.Module):
    def __init__(self, rank: int, hidden: int = 48):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(1, hidden), nn.Tanh(), nn.Linear(hidden, hidden),
                                 nn.Tanh(), nn.Linear(hidden, rank))

    def forward(self, x):
        return self.net(2 * x.reshape(-1, 1) - 1)


class NeuralCP(FieldModel):
    """Continuous low-rank tensor baseline with raw-coordinate neural factors."""

    def __init__(self, ndim: int, rank: int, hidden: int = 48):
        super().__init__()
        self.factors = nn.ModuleList([ScalarFactorNet(rank, hidden) for _ in range(ndim)])
        self.weight = nn.Parameter(torch.ones(rank) / rank)

    def forward(self, coords, indices=None, sample=True):
        prod = torch.ones(len(coords), len(self.weight), device=coords.device)
        for d, factor in enumerate(self.factors):
            prod = prod * factor(coords[:, d])
        return (prod * self.weight).sum(-1)

    def regularization(self):
        return self.weight.square().mean()


class SpectralCP(FieldModel):
    """Deterministic operator-spectral CP ablation."""

    def __init__(self, specs: Sequence[BasisSpec], rank: int, prior_power: float = 1.5,
                 wrong_geometry: bool = False):
        super().__init__()
        self.specs = tuple(
            BasisSpec("raw_fourier", s.n_frequencies, f"wrong-{s.name}")
            if wrong_geometry else s for s in specs
        )
        self.coeff = nn.ParameterList(
            [nn.Parameter(torch.randn(s.size, rank) / math.sqrt(s.size)) for s in self.specs]
        )
        self.weight = nn.Parameter(torch.ones(rank) / rank)
        self.prior_power = prior_power

    def factor_values(self, d: int, x: torch.Tensor):
        phi, _ = evaluate_basis(x, self.specs[d])
        # CP admits arbitrary reciprocal rescaling across modes.  Unit-norm
        # factor columns remove that ill-conditioned degree of freedom; the
        # component weight then carries the physical amplitude.
        coeff = self.coeff[d] / self.coeff[d].square().sum(0, keepdim=True).sqrt().clamp_min(1e-6)
        return phi @ coeff

    def forward(self, coords, indices=None, sample=True):
        prod = torch.ones(len(coords), len(self.weight), device=coords.device)
        for d in range(len(self.specs)):
            prod = prod * self.factor_values(d, coords[:, d])
        return (prod * self.weight).sum(-1)

    def regularization(self):
        reg = self.weight.square().mean()
        for s, w in zip(self.specs, self.coeff):
            dummy = w.new_zeros(1)
            _, eig = evaluate_basis(dummy, s)
            wn = w / w.square().sum(0, keepdim=True).sqrt().clamp_min(1e-6)
            reg = reg + ((1 + eig).pow(self.prior_power)[:, None] * wn.square()).mean()
        return reg


class GaussianParameter(nn.Module):
    def __init__(self, shape, init_scale: float = 0.05):
        super().__init__()
        self.mean = nn.Parameter(torch.randn(*shape) * init_scale)
        self.rho = nn.Parameter(torch.full(shape, -4.0))

    @property
    def std(self):
        return F.softplus(self.rho) + 1e-5

    def draw(self, sample: bool = True):
        return self.mean + self.std * torch.randn_like(self.mean) if sample else self.mean

    def kl(self, prior_precision: torch.Tensor):
        var = self.std.square()
        precision = torch.broadcast_to(prior_precision, self.mean.shape)
        return 0.5 * (-torch.log(precision) - torch.log(var) + precision *
                      (var + self.mean.square()) - 1).sum()


class BayesianSpectralCP(FieldModel):
    """Proposal 1: operator-spectral variational Bayesian functional CP.

    Each mode uses the eigenfunctions of its domain operator.  A learnable
    monotone Sobolev precision controls shrinkage by geometric frequency.
    """

    is_bayesian = True

    def __init__(self, specs: Sequence[BasisSpec], rank: int, init_power: float = 1.5):
        super().__init__()
        self.specs = tuple(specs)
        self.coeff = nn.ModuleList([GaussianParameter((s.size, rank), 0.08) for s in specs])
        self.component = GaussianParameter((rank,), 0.15)
        self.log_alpha = nn.Parameter(torch.full((len(specs),), -2.0))
        # A minimum Sobolev order is essential in the <5% regime.  An
        # unconstrained learned p collapsed toward zero in pilot runs and
        # recreated an unstructured coefficient prior.
        inv_softplus = math.log(math.exp(max(init_power - 1.5, 0.1)) - 1)
        self.raw_power = nn.Parameter(torch.full((len(specs),), inv_softplus))
        self.log_noise = nn.Parameter(torch.tensor(-2.5))

    @property
    def noise_std(self):
        return F.softplus(self.log_noise) + 1e-4

    @property
    def powers(self):
        return F.softplus(self.raw_power) + 1.5

    def _precision(self, d: int, eig: torch.Tensor):
        alpha = self.log_alpha[d].exp()
        return alpha * (1 + eig).pow(self.powers[d])

    def forward(self, coords, indices=None, sample=True):
        prod = torch.ones(len(coords), self.component.mean.numel(), device=coords.device)
        for d, (spec, q) in enumerate(zip(self.specs, self.coeff)):
            phi, _ = evaluate_basis(coords[:, d], spec)
            prod = prod * (phi @ q.draw(sample))
        return (prod * self.component.draw(sample)).sum(-1)

    def kl_divergence(self):
        kl = self.component.kl(torch.ones_like(self.component.mean))
        for d, (spec, q) in enumerate(zip(self.specs, self.coeff)):
            _, eig = evaluate_basis(q.mean.new_zeros(1), spec)
            kl = kl + q.kl(self._precision(d, eig)[:, None])
        # Weak hyperprior prevents the learned prior from escaping the KL.
        kl = kl + 0.5 * (self.log_alpha + 2.0).square().sum() + 0.5 * (self.powers - 2.0).square().sum()
        return kl

    def spectral_summary(self):
        out = []
        for d, (spec, q) in enumerate(zip(self.specs, self.coeff)):
            _, eig = evaluate_basis(q.mean.new_zeros(1), spec)
            out.append({
                "mode": spec.name,
                "alpha": float(self.log_alpha[d].exp().detach()),
                "power": float(self.powers[d].detach()),
                "eigenvalue": eig.detach().cpu().tolist(),
                "posterior_energy": (q.mean.square() + q.std.square()).sum(1).detach().cpu().tolist(),
                "posterior_uncertainty": q.std.square().sum(1).detach().cpu().tolist(),
            })
        return out


class BayesianSpectralTensor(FieldModel):
    """Stable proposal-1 POC: exact Bayesian core in a low-energy Tucker basis.

    The Cartesian product basis is truncated by *joint operator energy*, not by
    an axis-aligned box.  Conditional on that geometry-aware basis the model is
    Bayesian linear regression, so its posterior is exact and calibrated rather
    than a difficult mean-field approximation over multiplicative CP factors.
    """

    is_bayesian = True

    def __init__(self, specs: Sequence[BasisSpec], max_features: int = 384,
                 prior_power: float = 2.0):
        super().__init__()
        self.specs = tuple(specs)
        eigs = []
        for spec in specs:
            _, eig = evaluate_basis(torch.zeros(1), spec)
            eigs.append(eig)
        combos = torch.cartesian_prod(*[torch.arange(len(e)) for e in eigs])
        if combos.ndim == 1:
            combos = combos[:, None]
        joint = sum(eigs[d][combos[:, d]] for d in range(len(eigs)))
        keep = torch.argsort(joint)[:min(max_features, len(joint))]
        self.register_buffer("combos", combos[keep].long())
        self.register_buffer("joint_eigenvalue", joint[keep])
        self.prior_power = prior_power
        self.register_buffer("posterior_mean", torch.zeros(len(keep)))
        self.register_buffer("posterior_cholesky", torch.eye(len(keep)))
        self.register_buffer("fitted_noise", torch.tensor(0.1))
        self.register_buffer("fitted_alpha", torch.tensor(0.1))
        self.register_buffer("fitted_power", torch.tensor(float(prior_power)))

    @property
    def noise_std(self):
        return self.fitted_noise

    def design(self, coords: torch.Tensor):
        out = torch.ones(len(coords), len(self.combos), device=coords.device, dtype=coords.dtype)
        for d, spec in enumerate(self.specs):
            phi, _ = evaluate_basis(coords[:, d], spec)
            out = out * phi[:, self.combos[:, d]]
        return out

    @torch.no_grad()
    def fit_posterior(self, coords: torch.Tensor, y: torch.Tensor):
        # Evidence optimization and covariance factors are computed in float64;
        # sparse/underdetermined designs are otherwise fragile in float32.
        x = self.design(coords).double()
        y = y.double()
        xtx, xty = x.T @ x, x.T @ y
        n, m = x.shape
        eye = torch.eye(m, device=x.device, dtype=x.dtype)
        best = None
        # Empirical Bayes on a deliberately small, predeclared grid.
        for power in (0.5, 1.0, 1.5, 2.0):
            for alpha in (0.001, 0.003, 0.01, 0.03, 0.1, 0.3):
                precision = alpha * (1 + self.joint_eigenvalue.double()).pow(power)
                logdet_prior = torch.log(precision).sum()
                for noise in (0.03, 0.06, 0.1, 0.18):
                    variance = noise * noise
                    a = xtx / variance + torch.diag(precision) + 1e-6 * eye
                    chol, info = torch.linalg.cholesky_ex(a)
                    if int(info.max()) != 0:
                        continue
                    mean = torch.cholesky_solve((xty / variance)[:, None], chol).squeeze(1)
                    quad = y.square().sum() / variance - (xty * mean).sum() / variance
                    logdet = (n * math.log(variance) + 2 * torch.log(torch.diagonal(chol)).sum()
                              - logdet_prior)
                    nll = 0.5 * (quad + logdet + n * math.log(2 * math.pi))
                    if not torch.isfinite(nll) or not torch.isfinite(mean).all():
                        continue
                    if best is None or float(nll) < best[0]:
                        best = (float(nll), alpha, power, noise, mean.clone(), chol.clone())
        if best is None:
            raise RuntimeError("no positive-definite Bayesian spectral posterior")
        _, alpha, power, noise, mean, precision_chol = best
        # If posterior precision A = L L^T, then L^{-T} is a covariance
        # square-root.  Sampling through the triangular solve is substantially
        # more stable than explicitly forming A^{-1} below 1% observations.
        covariance_sqrt = torch.linalg.solve_triangular(
            precision_chol.T, eye, upper=True
        )
        self.posterior_mean.copy_(mean.float())
        self.posterior_cholesky.copy_(covariance_sqrt.float())
        self.fitted_alpha.fill_(alpha)
        self.fitted_power.fill_(power)
        self.fitted_noise.fill_(noise)

    def forward(self, coords, indices=None, sample=True):
        design = self.design(coords)
        weight = self.posterior_mean
        if sample:
            weight = weight + self.posterior_cholesky @ torch.randn_like(weight)
        return design @ weight

    def spectral_summary(self):
        energy = self.posterior_mean.square() + self.posterior_cholesky.square().sum(1)
        return [{"mode": "joint-product-operator", "alpha": float(self.fitted_alpha),
                 "power": float(self.fitted_power), "noise_std": float(self.fitted_noise),
                 "features": len(self.combos),
                 "joint_eigenvalue": self.joint_eigenvalue.detach().cpu().tolist(),
                 "posterior_energy": energy.detach().cpu().tolist()}]


class EigenFeatureFactor(nn.Module):
    def __init__(self, spec: BasisSpec, rank: int, hidden: int = 64):
        super().__init__()
        self.spec = spec
        self.linear = nn.Parameter(torch.randn(spec.size, rank) / math.sqrt(spec.size))
        self.adapter = nn.Sequential(nn.Linear(spec.size, hidden), nn.GELU(),
                                     nn.Linear(hidden, rank))
        self.residual_gate = nn.Parameter(torch.tensor(-4.0))

    def _raw(self, x):
        phi, _ = evaluate_basis(x, self.spec)
        return phi @ self.linear + torch.sigmoid(self.residual_gate) * self.adapter(phi)

    def _grid_values(self):
        n = max(32, 2 * self.spec.n_frequencies + 1)
        if self.spec.kind == "periodic":
            x = torch.arange(n, device=self.linear.device, dtype=self.linear.dtype) / n
        else:
            x = torch.linspace(0, 1, n, device=self.linear.device, dtype=self.linear.dtype)
        return self._raw(x)

    def forward(self, x):
        value = self._raw(x)
        rms = self._grid_values().square().mean(0, keepdim=True).sqrt().clamp_min(1e-4)
        return value / rms

    def operator_energy(self):
        values = self._grid_values()
        values = values / values.square().mean(0, keepdim=True).sqrt().clamp_min(1e-4)
        if self.spec.kind == "periodic":
            diff = torch.roll(values, -1, 0) - values
        else:
            diff = values[1:] - values[:-1]
        # Penalize the complete neural factor, not just its linear branch.
        return diff.square().mean() + 1e-3 * sum(
            p.square().mean() for p in self.adapter.parameters()
        )


class GeoNeuralTensor(FieldModel):
    """Proposal 2: eigenfeature neural factors plus operator-energy control."""

    def __init__(self, specs: Sequence[BasisSpec], rank: int, hidden: int = 64):
        super().__init__()
        self.specs = tuple(specs)
        self.factors = nn.ModuleList([EigenFeatureFactor(s, rank, hidden) for s in specs])
        self.weight = nn.Parameter(torch.ones(rank) / rank)

    def forward(self, coords, indices=None, sample=True):
        prod = torch.ones(len(coords), len(self.weight), device=coords.device)
        for d, factor in enumerate(self.factors):
            prod = prod * factor(coords[:, d])
        return (prod * self.weight).sum(-1)

    def regularization(self):
        return sum(f.operator_energy() for f in self.factors) + 0.1 * self.weight.square().mean()

    def spectral_summary(self):
        return [{"mode": f.spec.name,
                 "residual_gate": float(torch.sigmoid(f.residual_gate).detach()),
                 "linear_energy": f.linear.square().sum(1).detach().cpu().tolist()}
                for f in self.factors]


def build_model(name: str, shape: Sequence[int], specs: Sequence[BasisSpec], rank: int = 8,
                hidden: int = 64) -> FieldModel:
    if name == "cp":
        return DiscreteCP(shape, rank)
    if name == "inr":
        return SirenINR(len(shape), hidden=max(64, hidden), depth=3)
    if name == "fourier_inr":
        return FourierINR(len(shape), n_freq=8, hidden=max(64, hidden), depth=3)
    if name == "neural_cp":
        return NeuralCP(len(shape), rank, hidden)
    if name == "spectral_cp":
        return SpectralCP(specs, rank)
    if name == "wrong_spectral_cp":
        return SpectralCP(specs, rank, wrong_geometry=True)
    if name == "bayesian_spectral_cp":
        return BayesianSpectralCP(specs, rank)
    if name == "bayesian_spectral_tensor":
        return BayesianSpectralTensor(specs, max_features=max(128, 48 * rank))
    if name == "geo_nft":
        return GeoNeuralTensor(specs, rank, hidden)
    raise ValueError(f"unknown model: {name}")
