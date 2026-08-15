"""Classical Bayesian CP with operator-spectral priors on mode factors.

This module deliberately keeps the multilinear model explicit.  Product
features are never formed until the CP factors have reduced every mode to R
columns.  Inference is empirical Bayesian: mode-factor means are MAP estimates;
conditional on them, component amplitudes have an exact Gaussian ARD posterior.
An optional diagonal Gauss--Newton correction propagates factor uncertainty.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import torch
from torch import nn


@dataclass
class TensorBayesPrediction:
    mean: torch.Tensor
    std: torch.Tensor
    effective_rank: int | tuple[int, ...]
    component_precision: torch.Tensor
    component_energy: torch.Tensor
    factor_spectral_energy: list[torch.Tensor]
    factor_std: list[torch.Tensor] | None
    history: list[dict]
    metadata: dict


def _normalize_columns(values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    rms = values.square().mean(0, keepdim=True).sqrt().clamp_min(1e-6)
    return values / rms, rms


def _normalized_spectral_coefficients(basis: torch.Tensor,
                                      coefficients: torch.Tensor) -> torch.Tensor:
    """Coefficients of the unit-RMS factor represented in ``basis``.

    Forward factors are normalized to remove CP/Tucker scale ambiguity.  Their
    Sobolev penalty must use the same normalization; otherwise shrinking every
    coefficient changes the penalty while leaving the decoded tensor unchanged.
    """
    _, rms = _normalize_columns(basis.to(coefficients.device) @ coefficients)
    return coefficients / rms


class OperatorBayesianCP(nn.Module):
    """CP factorization with mode-wise operator bases and component ARD.

    ``basis[m]`` has shape ``n_m x k_m`` and maps spectral coefficients to a
    factor table. ``eigenvalues[m]`` defines the Sobolev precision.  Passing
    identity bases and zero eigenvalues recovers ordinary discrete Bayesian CP.
    """

    def __init__(self, basis: Sequence[torch.Tensor], eigenvalues: Sequence[torch.Tensor],
                 rank: int = 12, power: float = 1.5, ard: bool = True,
                 factor_laplace: bool = False, device: str = "cuda"):
        super().__init__()
        self.basis = [b.float().contiguous() for b in basis]
        self.eigenvalues = [e.float().contiguous() for e in eigenvalues]
        self.rank = rank; self.power = power; self.ard = ard
        self.factor_laplace = factor_laplace
        self.device_name = device if torch.cuda.is_available() else "cpu"
        self.coeff = nn.ParameterList([
            nn.Parameter(torch.randn(b.shape[1], rank) / math.sqrt(b.shape[1])) for b in basis
        ])
        self.amplitude = nn.Parameter(torch.ones(rank) / math.sqrt(rank))
        self._posterior = None

    def factor_tables(self) -> list[torch.Tensor]:
        out = []
        for b, u in zip(self.basis, self.coeff):
            values = b.to(u.device) @ u
            out.append(_normalize_columns(values)[0])
        return out

    @staticmethod
    def cp_design(indices: torch.Tensor, factors: Sequence[torch.Tensor]) -> torch.Tensor:
        z = torch.ones(len(indices), factors[0].shape[1], device=indices.device)
        for m, f in enumerate(factors): z *= f[indices[:, m]]
        return z

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        return self.cp_design(indices, self.factor_tables()) @ self.amplitude

    def factor_prior(self) -> torch.Tensor:
        total = self.amplitude.square().mean()
        for b, u, eig in zip(self.basis, self.coeff, self.eigenvalues):
            precision = (1 + eig.to(u.device)).pow(self.power)[:, None]
            normalized = _normalized_spectral_coefficients(b, u)
            total = total + (precision * normalized.square()).mean()
        return total

    @torch.no_grad()
    def initialize_from_tensor(self, tensor: torch.Tensor, iterations: int = 40):
        """CP-ALS initialization followed by projection into each mode basis."""
        if tensor.ndim != 3: raise ValueError("current ALS initializer expects an order-3 tensor")
        x=tensor.to(next(self.parameters()).device).float(); i,j,k=x.shape; r=self.rank
        gen=torch.Generator(device=x.device).manual_seed(91027)
        fac=[torch.randn(n,r,generator=gen,device=x.device) for n in (i,j,k)]
        eye=torch.eye(r,device=x.device)
        for _ in range(iterations):
            kr=torch.einsum("jr,kr->jkr",fac[1],fac[2]).reshape(j*k,r)
            fac[0]=(x.reshape(i,j*k)@kr)@torch.linalg.inv((fac[1].T@fac[1])*(fac[2].T@fac[2])+1e-4*eye)
            kr=torch.einsum("ir,kr->ikr",fac[0],fac[2]).reshape(i*k,r)
            fac[1]=(x.permute(1,0,2).reshape(j,i*k)@kr)@torch.linalg.inv((fac[0].T@fac[0])*(fac[2].T@fac[2])+1e-4*eye)
            kr=torch.einsum("ir,jr->ijr",fac[0],fac[1]).reshape(i*j,r)
            fac[2]=(x.permute(2,0,1).reshape(k,i*j)@kr)@torch.linalg.inv((fac[0].T@fac[0])*(fac[1].T@fac[1])+1e-4*eye)
        amp=torch.ones(r,device=x.device)
        for m,f in enumerate(fac):
            rms=f.square().mean(0).sqrt().clamp_min(1e-6); f=f/rms; amp*=rms
            b=self.basis[m].to(x.device)
            self.coeff[m].copy_(torch.linalg.lstsq(b,f).solution)
        self.amplitude.copy_(amp)

    def fit(self, indices_obs: torch.Tensor, y_obs: torch.Tensor, *, steps: int = 1800,
            lr: float = 3e-3, reg_weight: float = 2e-3, ard_cycles: int = 1,
            seed: int = 0, initial_tensor: torch.Tensor | None = None) -> "OperatorBayesianCP":
        torch.manual_seed(seed); device = torch.device(self.device_name)
        self.to(device)
        if initial_tensor is not None: self.initialize_from_tensor(initial_tensor)
        ix = indices_obs.to(device); y = y_obs.to(device)
        opt = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=1e-6)
        history=[]; best=(float("inf"),None)
        for step in range(steps):
            pred = self(ix); data_loss=(pred-y).square().mean()
            loss=data_loss+reg_weight*self.factor_prior()
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(),5.); opt.step()
            value=float(loss.detach())
            if value < best[0]: best=(value,{k:v.detach().cpu().clone() for k,v in self.state_dict().items()})
            if step % max(1,steps//10)==0 or step==steps-1:
                history.append({"step":step,"loss":value,"data_loss":float(data_loss.detach())})
        self.load_state_dict(best[1]); self.to(device)

        # Alternating exact core/amplitude inference and a short factor refit is
        # the smallest way for rank ARD to influence the learned CP factors.
        for cycle in range(max(1, ard_cycles)):
            self._fit_amplitude_posterior(ix, y)
            if cycle + 1 < ard_cycles:
                with torch.no_grad(): self.amplitude.copy_(self._posterior["mean"])
                opt=torch.optim.AdamW(self.coeff.parameters(),lr=lr*.35)
                for _ in range(max(100,steps//6)):
                    pred=self(ix); loss=(pred-y).square().mean()+reg_weight*self.factor_prior()
                    opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        self._fit_amplitude_posterior(ix,y)
        self._posterior["history"]=history
        self._posterior["best_observed_objective"]=best[0]
        if self.factor_laplace:
            self._posterior["factor_var"] = self._diagonal_factor_laplace(ix)
        return self

    @torch.no_grad()
    def _fit_amplitude_posterior(self, ix: torch.Tensor, y: torch.Tensor):
        z=self.cp_design(ix,self.factor_tables()).double(); yd=y.double()
        r=z.shape[1]; eye=torch.eye(r,device=z.device,dtype=z.dtype)
        alpha=torch.ones(r,device=z.device,dtype=z.dtype)
        beta=torch.tensor(25.,device=z.device,dtype=z.dtype)
        if self.ard:
            for _ in range(80):
                precision=beta*(z.T@z)+torch.diag(alpha)+1e-8*eye
                chol=torch.linalg.cholesky(precision)
                cov=torch.cholesky_inverse(chol)
                mean=beta*(cov@z.T@yd)
                gamma=(1-alpha*cov.diagonal()).clamp(0,1)
                alpha_new=((gamma+2e-3)/(mean.square()+2e-3)).clamp(1e-3,1e5)
                resid=(yd-z@mean).square().sum()
                beta_new=((len(yd)-gamma.sum()).clamp_min(1)/(resid+1e-3)).clamp(1e-2,1e5)
                if max(float((alpha_new-alpha).abs().max()),float((beta_new-beta).abs()))<1e-5: break
                alpha,beta=alpha_new,beta_new
        else:
            # Equal component precision is the no-rank-ARD ablation.
            alpha.fill_(1.)
        precision=beta*(z.T@z)+torch.diag(alpha)+1e-8*eye
        cov=torch.linalg.inv(precision); mean=beta*(cov@z.T@yd)
        # Analytic Bayesian-linear leverage gives LOO calibration without an
        # O(n_obs^2) kernel matrix.  This matters on public fields where even a
        # 1% mask contains thousands of values.
        fitted=z@mean
        predictive_var=((z@cov)*z).sum(1)
        leverage=(beta*predictive_var).clamp(max=.999)
        loo_resid=(yd-fitted)/(1-leverage).clamp_min(1e-4)
        loo_std=torch.sqrt(1/beta+predictive_var)
        absz=loo_resid.abs()/loo_std.clamp_min(1e-8)
        calibration=float(torch.quantile(absz,.95)/1.96)
        calibration=max(.5,min(4.,calibration))
        self._posterior={"mean":mean.float(),"cov":cov.float(),"alpha":alpha.float(),
                         "noise":float(beta.rsqrt()),"calibration":calibration}

    def _diagonal_factor_laplace(self, ix: torch.Tensor) -> list[torch.Tensor]:
        """Diagonal Gauss--Newton posterior for spectral factor coefficients."""
        factors=self.factor_tables(); amp=self._posterior["mean"].to(ix.device)
        beta=1/(self._posterior["noise"]**2)
        variances=[]
        for m,(u,b,eig) in enumerate(zip(self.coeff,self.basis,self.eigenvalues)):
            other=torch.ones(len(ix),self.rank,device=ix.device)
            for q,f in enumerate(factors):
                if q!=m: other*=f[ix[:,q]]
            phi=b.to(ix.device)[ix[:,m]]
            # Ignore the small derivative of whole-grid RMS normalization.
            fisher=beta*torch.einsum("nk,nr->kr",phi.square(),(other*amp).square())
            prior=(1+eig.to(ix.device)).pow(self.power)[:,None]
            variances.append((fisher+prior+1e-6).reciprocal().detach())
        return variances

    @torch.no_grad()
    def predict(self, all_indices: torch.Tensor, *, samples: int = 32,
                chunk_size: int = 8192) -> TensorBayesPrediction:
        if self._posterior is None: raise RuntimeError("fit first")
        device=next(self.parameters()).device; ix=all_indices.to(device)
        factors=self.factor_tables(); mean_w=self._posterior["mean"].to(device)
        cov=self._posterior["cov"].to(device); chunks=[]; vars_=[]
        for start in range(0,len(ix),chunk_size):
            z=self.cp_design(ix[start:start+chunk_size],factors)
            chunks.append((z@mean_w).cpu())
            vars_.append(((z@cov)*z).sum(1).clamp_min(0).cpu())
        mean=torch.cat(chunks); var=torch.cat(vars_)
        factor_std=None
        if self.factor_laplace:
            factor_var=self._posterior["factor_var"]
            factor_std=[v.sqrt().cpu() for v in factor_var]
            # Delta-method correction, preserving CP multilinearity.
            for start in range(0,len(ix),chunk_size):
                ids=ix[start:start+chunk_size]; correction=torch.zeros(len(ids),device=device)
                for m,(b,v) in enumerate(zip(self.basis,factor_var)):
                    other=torch.ones(len(ids),self.rank,device=device)
                    for q,f in enumerate(factors):
                        if q!=m: other*=f[ids[:,q]]
                    phi2=b.to(device)[ids[:,m]].square()
                    correction += ((phi2@v)*(other*mean_w).square()).sum(1)
                var[start:start+len(ids)] += correction.cpu()
        std=(var+self._posterior["noise"]**2).sqrt()*self._posterior["calibration"]
        alpha=self._posterior["alpha"].cpu(); energy=(mean_w.square()+cov.diagonal()).cpu()
        effective=int(((alpha<100.) & (energy>1e-4)).sum())
        spectral=[]
        for b,u,eig in zip(self.basis,self.coeff,self.eigenvalues):
            normalized=_normalized_spectral_coefficients(b,u).detach().cpu()
            spectral.append(((1+eig[:,None]).pow(self.power)*normalized.square()).sum(0))
        return TensorBayesPrediction(mean,std,effective,alpha,energy,spectral,factor_std,
                                     self._posterior["history"],
                                     {"rank_cap":self.rank,"ard":self.ard,"power":self.power,
                                      "noise":self._posterior["noise"],
                                      "calibration":self._posterior["calibration"],
                                      "best_observed_objective":
                                          self._posterior["best_observed_objective"]})


class OperatorBayesianTucker(nn.Module):
    """Small-core Tucker factorization with mode-wise operator priors.

    The nonlinear part of inference estimates one smooth factor table per mode.
    Conditional on those tables, the Tucker core has an exact Gaussian
    posterior.  This is deliberately a minimal extension of
    :class:`OperatorBayesianCP`: geometry still enters only through the mode
    operators, while the explicit core relaxes CP's super-diagonal constraint.
    """

    def __init__(self, basis: Sequence[torch.Tensor], eigenvalues: Sequence[torch.Tensor],
                 ranks: Sequence[int] = (4, 5, 5), power: float = 1.5,
                 device: str = "cuda"):
        super().__init__()
        if len(basis) != 3 or len(ranks) != 3:
            raise ValueError("current implementation expects an order-3 tensor")
        self.basis = [b.float().contiguous() for b in basis]
        self.eigenvalues = [e.float().contiguous() for e in eigenvalues]
        self.ranks = tuple(int(r) for r in ranks)
        self.power = power
        self.device_name = device if torch.cuda.is_available() else "cpu"
        self.coeff = nn.ParameterList([
            nn.Parameter(torch.randn(b.shape[1], r) / math.sqrt(b.shape[1]))
            for b, r in zip(basis, self.ranks)
        ])
        self.core = nn.Parameter(torch.randn(*self.ranks) / math.sqrt(math.prod(self.ranks)))
        self._posterior = None

    def factor_tables(self) -> list[torch.Tensor]:
        return [_normalize_columns(b.to(u.device) @ u)[0]
                for b, u in zip(self.basis, self.coeff)]

    @staticmethod
    def tucker_design(indices: torch.Tensor,
                      factors: Sequence[torch.Tensor]) -> torch.Tensor:
        selected = [f[indices[:, m]] for m, f in enumerate(factors)]
        return torch.einsum("na,nb,nc->nabc", *selected).flatten(1)

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        return self.tucker_design(indices, self.factor_tables()) @ self.core.flatten()

    def factor_prior(self) -> torch.Tensor:
        total = self.core.square().mean()
        for b, u, eig in zip(self.basis, self.coeff, self.eigenvalues):
            precision = (1 + eig.to(u.device)).pow(self.power)[:, None]
            normalized = _normalized_spectral_coefficients(b, u)
            total = total + (precision * normalized.square()).mean()
        return total

    @torch.no_grad()
    def initialize_from_tensor(self, tensor: torch.Tensor):
        """Operator-projected HOSVD initializer followed by a core solve."""
        if tensor.ndim != 3:
            raise ValueError("current HOSVD initializer expects an order-3 tensor")
        x = tensor.to(next(self.parameters()).device).float()
        targets = []
        for mode, rank in enumerate(self.ranks):
            unfold = x.movedim(mode, 0).reshape(x.shape[mode], -1)
            targets.append(torch.linalg.svd(unfold, full_matrices=False).U[:, :rank])
        for m, target in enumerate(targets):
            b = self.basis[m].to(x.device)
            self.coeff[m].copy_(torch.linalg.lstsq(b, target).solution)
        design = self.tucker_design(
            torch.cartesian_prod(*[torch.arange(n, device=x.device) for n in x.shape]),
            self.factor_tables())
        eye = torch.eye(design.shape[1], device=x.device)
        solution = torch.linalg.solve(design.T @ design + 1e-5 * eye,
                                      design.T @ x.flatten())
        self.core.copy_(solution.reshape(self.ranks))

    def fit(self, indices_obs: torch.Tensor, y_obs: torch.Tensor, *, steps: int = 1800,
            lr: float = 3e-3, reg_weight: float = 2e-3, seed: int = 0,
            initial_tensor: torch.Tensor | None = None) -> "OperatorBayesianTucker":
        torch.manual_seed(seed)
        device = torch.device(self.device_name)
        self.to(device)
        if initial_tensor is not None:
            self.initialize_from_tensor(initial_tensor)
        ix, y = indices_obs.to(device), y_obs.to(device)
        opt = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=1e-6)
        history = []
        best = (float("inf"), None)
        for step in range(steps):
            pred = self(ix)
            data_loss = (pred - y).square().mean()
            loss = data_loss + reg_weight * self.factor_prior()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 5.)
            opt.step()
            value = float(loss.detach())
            if value < best[0]:
                best = (value, {k: v.detach().cpu().clone()
                                for k, v in self.state_dict().items()})
            if step % max(1, steps // 10) == 0 or step == steps - 1:
                history.append({"step": step, "loss": value,
                                "data_loss": float(data_loss.detach())})
        self.load_state_dict(best[1])
        self.to(device)
        self._fit_core_posterior(ix, y)
        self._posterior["history"] = history
        self._posterior["best_observed_objective"] = best[0]
        return self

    @torch.no_grad()
    def _fit_core_posterior(self, ix: torch.Tensor, y: torch.Tensor):
        z = self.tucker_design(ix, self.factor_tables()).double()
        yd = y.double()
        p = z.shape[1]
        eye = torch.eye(p, device=z.device, dtype=z.dtype)
        alpha = torch.tensor(1., device=z.device, dtype=z.dtype)
        beta = torch.tensor(25., device=z.device, dtype=z.dtype)
        for _ in range(80):
            precision = beta * (z.T @ z) + alpha * eye
            cov = torch.linalg.inv(precision)
            mean = beta * (cov @ z.T @ yd)
            gamma = (p - alpha * cov.trace()).clamp(1e-3, p - 1e-3)
            alpha_new = (gamma / mean.square().sum().clamp_min(1e-8)).clamp(1e-4, 1e5)
            resid = (yd - z @ mean).square().sum()
            beta_new = ((len(yd) - gamma).clamp_min(1.) / resid.clamp_min(1e-8)).clamp(1e-3, 1e5)
            if max(float((alpha_new-alpha).abs()), float((beta_new-beta).abs())) < 1e-5:
                alpha, beta = alpha_new, beta_new
                break
            alpha, beta = alpha_new, beta_new
        precision = beta * (z.T @ z) + alpha * eye
        cov = torch.linalg.inv(precision)
        mean = beta * (cov @ z.T @ yd)
        # Analytic linear-model LOO residuals avoid an n_obs x n_obs inverse.
        fitted = z @ mean
        leverage = (beta * ((z @ cov) * z).sum(1)).clamp(max=.999)
        loo_resid = (yd - fitted) / (1 - leverage).clamp_min(1e-4)
        loo_std = torch.sqrt(1 / beta + ((z @ cov) * z).sum(1))
        calibration = float(torch.quantile(loo_resid.abs() / loo_std.clamp_min(1e-8), .95) / 1.96)
        calibration = max(.5, min(4., calibration))
        self._posterior = {"mean": mean.float(), "cov": cov.float(),
                           "alpha": float(alpha), "noise": float(beta.rsqrt()),
                           "calibration": calibration}

    @torch.no_grad()
    def predict(self, all_indices: torch.Tensor, *, chunk_size: int = 8192
                ) -> TensorBayesPrediction:
        if self._posterior is None:
            raise RuntimeError("fit first")
        device = next(self.parameters()).device
        ix = all_indices.to(device)
        factors = self.factor_tables()
        mean_core = self._posterior["mean"].to(device)
        cov = self._posterior["cov"].to(device)
        means, variances = [], []
        for start in range(0, len(ix), chunk_size):
            z = self.tucker_design(ix[start:start+chunk_size], factors)
            means.append((z @ mean_core).cpu())
            variances.append(((z @ cov) * z).sum(1).clamp_min(0).cpu())
        mean = torch.cat(means)
        var = torch.cat(variances)
        std = (var + self._posterior["noise"] ** 2).sqrt() * self._posterior["calibration"]
        spectral = [((1 + eig[:, None]).pow(self.power) *
                     _normalized_spectral_coefficients(b, u).detach().cpu().square()).sum(0)
                    for b, u, eig in zip(self.basis, self.coeff, self.eigenvalues)]
        core_energy = mean_core.cpu().square() + cov.diagonal().cpu()
        core_precision = torch.full_like(core_energy, self._posterior["alpha"])
        return TensorBayesPrediction(
            mean, std, self.ranks, core_precision, core_energy, spectral, None,
            self._posterior["history"],
            {"ranks": self.ranks, "power": self.power,
             "core_size": math.prod(self.ranks),
             "core_precision": self._posterior["alpha"],
             "noise": self._posterior["noise"],
             "calibration": self._posterior["calibration"],
             "best_observed_objective":
                 self._posterior["best_observed_objective"]})
