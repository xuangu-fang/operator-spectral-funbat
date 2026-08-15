"""Exact finite-feature Bayesian field inference and uncertainty diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import torch
from scipy.stats import spearmanr


@dataclass
class BayesianPrediction:
    mean: torch.Tensor
    raw_std: torch.Tensor
    calibrated_std: torch.Tensor
    conditional_std: torch.Tensor
    hyperparameters: dict


class ExactFeatureBayes:
    """Exact GP induced by operator features, solved in observation space.

    With ``w ~ N(0, diag(tau))`` and ``y = Phi w + eps``, observation-space
    inference costs O(n_obs^3 + n_all m n_obs).  This is particularly attractive
    at 0.1--0.5% observation rates, where n_obs is much smaller than m.
    """

    def __init__(self, features: torch.Tensor, eigenvalues: torch.Tensor,
                 selector: str = "loo", calibrate: bool = True,
                 device: str = "cuda"):
        self.features_cpu = features.float().contiguous()
        self.eigenvalues_cpu = eigenvalues.float().contiguous()
        self.selector = selector
        self.calibrate = calibrate
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self._state = None

    @staticmethod
    def _candidate_grid(anisotropic: bool):
        powers = ((.5, .5), (1., 1.), (1.5, 1.5), (2., 2.), (3., 3.),
                  (.5, 1.5), (.5, 3.), (1., 2.), (1.5, .5), (2., 1.), (3., .5)) \
                 if anisotropic else ((.5, .5), (1., 1.), (1.5, 1.5), (2., 2.), (3., 3.))
        for time_power, graph_power in powers:
            for amplitude in (0.3, 1.0, 3.0):
                for noise in (0.03, 0.07, 0.15, 0.30):
                    yield time_power, graph_power, amplitude, noise

    @torch.no_grad()
    def fit(self, observed: torch.Tensor, y: torch.Tensor) -> "ExactFeatureBayes":
        obs = observed.long().cpu()
        center = float(y.mean())
        scale = float(y.std().clamp_min(1e-6))
        yn = ((y - center) / scale).to(self.device, torch.float64)
        xo = self.features_cpu[obs].to(self.device, torch.float64)
        eig = self.eigenvalues_cpu.to(self.device, torch.float64)
        n = len(obs)
        eye = torch.eye(n, device=self.device, dtype=torch.float64)
        best = None
        anisotropic = eig.ndim == 2
        for time_power, graph_power, amplitude, noise in self._candidate_grid(anisotropic):
            tau = (amplitude * (1. + eig[:, 0]).pow(-time_power)
                   * (1. + eig[:, 1]).pow(-graph_power)) if anisotropic \
                  else amplitude * (1. + eig).pow(-time_power)
            z = xo * tau.sqrt()
            k = z @ z.T + (noise * noise + 1e-8) * eye
            chol, info = torch.linalg.cholesky_ex(k)
            if int(info.max()) != 0:
                continue
            alpha = torch.cholesky_solve(yn[:, None], chol).squeeze(1)
            if self.selector == "evidence":
                score = 0.5 * (yn @ alpha + 2 * torch.log(torch.diagonal(chol)).sum()
                               + n * math.log(2 * math.pi))
            elif self.selector == "loo":
                kinv_diag = torch.cholesky_inverse(chol).diagonal().clamp_min(1e-12)
                residual = alpha / kinv_diag
                variance = 1.0 / kinv_diag
                score = (0.5 * residual.square() / variance + 0.5 * torch.log(variance)
                         + 0.5 * math.log(2 * math.pi)).mean()
            else:
                raise ValueError(f"unknown selector: {self.selector}")
            if torch.isfinite(score) and (best is None or float(score) < best[0]):
                best = (float(score), time_power, graph_power, amplitude, noise, tau, chol, alpha)
        if best is None:
            raise RuntimeError("no valid exact Bayesian feature posterior")
        score, time_power, graph_power, amplitude, noise, tau, chol, alpha = best

        # Cross-fitted standardized residuals provide an observation-only
        # dispersion correction.  It is deliberately scalar: geometry remains
        # responsible for the spatial pattern of uncertainty.
        kinv_diag = torch.cholesky_inverse(chol).diagonal().clamp_min(1e-12)
        loo_residual = alpha / kinv_diag
        loo_std = (1.0 / kinv_diag).sqrt()
        abs_z = (loo_residual.abs() / loo_std).detach().cpu()
        q = float(torch.quantile(abs_z, 0.95)) if len(abs_z) >= 8 else 1.96
        calibration = max(0.5, min(4.0, q / 1.96)) if self.calibrate else 1.0
        # Conditional calibration: stratify by cross-fitted predictive scale.
        # All thresholds and multipliers use observations only.  Shrink noisy
        # bin estimates toward the global factor in the ultra-small-n regime.
        loo_std_cpu = loo_std.detach().cpu()
        n_bins = 3 if len(abs_z) >= 36 else (2 if len(abs_z) >= 16 else 1)
        thresholds = torch.quantile(
            loo_std_cpu, torch.linspace(0, 1, n_bins + 1, dtype=loo_std_cpu.dtype)[1:-1]
        )
        bin_scales = []
        for b in range(n_bins):
            lo = -float("inf") if b == 0 else float(thresholds[b - 1])
            hi = float("inf") if b == n_bins - 1 else float(thresholds[b])
            use = (loo_std_cpu > lo) & (loo_std_cpu <= hi)
            local = float(torch.quantile(abs_z[use], .95) / 1.96) if int(use.sum()) >= 4 else calibration
            weight = int(use.sum()) / (int(use.sum()) + 12.0)
            bin_scales.append(max(.5, min(4., weight * local + (1 - weight) * calibration)))
        self._state = dict(obs=obs, center=center, scale=scale, time_power=time_power,
                           graph_power=graph_power, amplitude=amplitude, noise=noise,
                           tau=tau, chol=chol,
                           alpha=alpha, selection_score=score, calibration=calibration,
                           calibration_thresholds=thresholds.tolist(),
                           conditional_scales=bin_scales, loo_abs_z=abs_z.tolist())
        return self

    @torch.no_grad()
    def predict(self, chunk_size: int = 8192) -> BayesianPrediction:
        if self._state is None:
            raise RuntimeError("fit must be called before predict")
        st = self._state
        xo = self.features_cpu[st["obs"]].to(self.device, torch.float64)
        tau = st["tau"]
        weighted_obs = xo * tau
        means, stds = [], []
        for start in range(0, len(self.features_cpu), chunk_size):
            x = self.features_cpu[start:start + chunk_size].to(self.device, torch.float64)
            kxo = x @ weighted_obs.T
            mean = kxo @ st["alpha"]
            solved = torch.cholesky_solve(kxo.T, st["chol"])
            prior_var = x.square() @ tau
            latent_var = (prior_var - (kxo * solved.T).sum(1)).clamp_min(1e-10)
            pred_var = latent_var + st["noise"] ** 2
            means.append(mean.float().cpu()); stds.append(pred_var.sqrt().float().cpu())
        mean = torch.cat(means) * st["scale"] + st["center"]
        raw_std = torch.cat(stds) * st["scale"]
        cal_std = raw_std * st["calibration"]
        cond_std = raw_std.clone()
        normalized_std = raw_std / st["scale"]
        bins = torch.bucketize(normalized_std, torch.tensor(st["calibration_thresholds"]))
        scales = torch.tensor(st["conditional_scales"])
        cond_std *= scales[bins]
        hp = {k: st[k] for k in ("time_power", "graph_power", "amplitude", "noise",
                                  "selection_score", "calibration", "conditional_scales")}
        hp["selector"] = self.selector
        return BayesianPrediction(mean, raw_std, cal_std, cond_std, hp)

    def posterior_variance(self) -> torch.Tensor:
        return self.predict().raw_std.square()

    @torch.no_grad()
    def integrated_variance_scores(self, candidate_ids: torch.Tensor,
                                   chunk_size: int = 4096) -> torch.Tensor:
        """Expected global latent-variance reduction from one noisily observed point.

        For candidate feature ``phi_c`` and posterior coefficient covariance S,
        the integrated reduction is
        ``phi_c^T S (Phi^T Phi / N) S phi_c / (phi_c^T S phi_c + sigma^2)``.
        This turns geometry-resolved uncertainty into a principled sensor score.
        """
        if self._state is None:
            raise RuntimeError("fit must be called before acquisition")
        st = self._state
        phi = self.features_cpu.to(self.device, torch.float64)
        xo = self.features_cpu[st["obs"]].to(self.device, torch.float64)
        tau = st["tau"]
        # S = D - D Phi_o^T K^-1 Phi_o D.
        d_xot = tau[:, None] * xo.T
        solved = torch.cholesky_solve(d_xot.T, st["chol"])
        s = torch.diag(tau) - d_xot @ solved
        out = []
        ids = candidate_ids.long().cpu()
        for start in range(0, len(ids), chunk_size):
            pc = self.features_cpu[ids[start:start + chunk_size]].to(self.device, torch.float64)
            q = pc @ s
            # The graph x temporal basis is orthonormal under the uniform grid
            # measure, hence Phi^T Phi / N = I up to discretization error.
            numerator = q.square().sum(1)
            denominator = (q * pc).sum(1).clamp_min(0) + st["noise"] ** 2
            out.append((numerator / denominator.clamp_min(1e-12)).float().cpu())
        return torch.cat(out)

    @torch.no_grad()
    def group_integrated_variance_scores(self, candidate_groups: torch.Tensor,
                                         chunk_size: int = 32) -> torch.Tensor:
        """Exact IV reduction when one action reveals a group of observations."""
        if self._state is None:
            raise RuntimeError("fit must be called before acquisition")
        st = self._state
        xo = self.features_cpu[st["obs"]].to(self.device, torch.float64)
        tau = st["tau"]
        d_xot = tau[:, None] * xo.T
        solved = torch.cholesky_solve(d_xot.T, st["chol"])
        s = torch.diag(tau) - d_xot @ solved
        groups = candidate_groups.long().cpu(); out = []
        group_n = groups.shape[1]
        eye = torch.eye(group_n, device=self.device, dtype=torch.float64)
        for start in range(0, len(groups), chunk_size):
            ids = groups[start:start + chunk_size]
            b = self.features_cpu[ids].to(self.device, torch.float64)
            q = b @ s
            conditional = q @ b.transpose(1, 2) + st["noise"] ** 2 * eye
            reduction = q @ q.transpose(1, 2)
            score = torch.linalg.solve(conditional, reduction).diagonal(dim1=1, dim2=2).sum(1)
            out.append(score.float().cpu())
        return torch.cat(out)


class ExactRBF:
    """Full Euclidean RBF GP baseline, with the same LOO/evidence protocol."""

    def __init__(self, coordinates: torch.Tensor, selector: str = "loo",
                 calibrate: bool = True, device: str = "cuda"):
        self.coords = coordinates.float()
        self.selector = selector; self.calibrate = calibrate
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self._state = None

    @torch.no_grad()
    def fit(self, observed: torch.Tensor, y: torch.Tensor):
        obs = observed.long().cpu(); xo = self.coords[obs].to(self.device, torch.float64)
        center, scale = float(y.mean()), float(y.std().clamp_min(1e-6))
        yn = ((y - center) / scale).to(self.device, torch.float64)
        n = len(obs); eye = torch.eye(n, dtype=torch.float64, device=self.device)
        best = None
        for lengthscale in (0.05, 0.10, 0.18, 0.30, 0.50):
            dist2 = torch.cdist(xo / lengthscale, xo / lengthscale).square()
            base = torch.exp(-0.5 * dist2)
            for amplitude in (0.3, 1.0, 3.0):
                for noise in (0.03, 0.07, 0.15, 0.30):
                    k = amplitude * base + (noise * noise + 1e-8) * eye
                    chol, info = torch.linalg.cholesky_ex(k)
                    if int(info.max()) != 0: continue
                    alpha = torch.cholesky_solve(yn[:, None], chol).squeeze(1)
                    if self.selector == "evidence":
                        score = 0.5 * (yn @ alpha + 2 * torch.log(torch.diagonal(chol)).sum()
                                       + n * math.log(2 * math.pi))
                    else:
                        kd = torch.cholesky_inverse(chol).diagonal().clamp_min(1e-12)
                        res, var = alpha / kd, 1 / kd
                        score = (0.5 * res.square() / var + 0.5 * torch.log(var)
                                 + 0.5 * math.log(2 * math.pi)).mean()
                    if torch.isfinite(score) and (best is None or float(score) < best[0]):
                        best = (float(score), lengthscale, amplitude, noise, chol, alpha)
        score, ls, amp, noise, chol, alpha = best
        kd = torch.cholesky_inverse(chol).diagonal().clamp_min(1e-12)
        z = (alpha / kd).abs() / (1 / kd).sqrt()
        q = float(torch.quantile(z, .95)) if n >= 8 else 1.96
        calibration = max(.5, min(4., q / 1.96)) if self.calibrate else 1.
        self._state = dict(obs=obs, xo=xo, center=center, scale=scale, lengthscale=ls,
                           amplitude=amp, noise=noise, chol=chol, alpha=alpha,
                           selection_score=score, calibration=calibration)
        return self

    @torch.no_grad()
    def predict(self, chunk_size: int = 8192) -> BayesianPrediction:
        st = self._state; means, stds = [], []
        for start in range(0, len(self.coords), chunk_size):
            x = self.coords[start:start + chunk_size].to(self.device, torch.float64)
            d2 = torch.cdist(x / st["lengthscale"], st["xo"] / st["lengthscale"]).square()
            kxo = st["amplitude"] * torch.exp(-.5 * d2)
            mean = kxo @ st["alpha"]
            sol = torch.cholesky_solve(kxo.T, st["chol"])
            var = (st["amplitude"] - (kxo * sol.T).sum(1)).clamp_min(1e-10) + st["noise"] ** 2
            means.append(mean.float().cpu()); stds.append(var.sqrt().float().cpu())
        mean = torch.cat(means) * st["scale"] + st["center"]
        raw = torch.cat(stds) * st["scale"]
        hp = {k: st[k] for k in ("lengthscale", "amplitude", "noise", "selection_score", "calibration")}
        hp["selector"] = self.selector
        return BayesianPrediction(mean, raw, raw * st["calibration"], raw * st["calibration"], hp)


def uncertainty_metrics(truth: torch.Tensor, pred: BayesianPrediction,
                        held_out: torch.Tensor) -> dict:
    y, mu = truth[held_out].float(), pred.mean[held_out].float()
    err = y - mu
    rmse = float(err.square().mean().sqrt())
    out = {"rmse": rmse, "nrmse": rmse / float(y.std().clamp_min(1e-8)),
           "mae": float(err.abs().mean())}
    for label, std in (("raw", pred.raw_std), ("cal", pred.calibrated_std),
                       ("conditional", pred.conditional_std)):
        s = std[held_out].clamp_min(1e-7)
        z = err.abs() / s
        coverages = {str(level): float((z <= torch.distributions.Normal(0, 1).icdf(
            torch.tensor((1 + level) / 2))).float().mean()) for level in (.5, .8, .95)}
        nll = float((.5 * (err / s).square() + torch.log(s) + .5 * math.log(2 * math.pi)).mean())
        ece = sum(abs(coverages[str(q)] - q) for q in (.5, .8, .95)) / 3
        out[f"{label}_nll"] = nll; out[f"{label}_coverage"] = coverages
        out[f"{label}_coverage_ece"] = ece; out[f"{label}_width95"] = float(3.92 * s.mean())
    u = pred.raw_std[held_out].numpy(); ae = err.abs().numpy()
    corr = spearmanr(u, ae).statistic
    out["uncertainty_error_spearman"] = float(corr) if np.isfinite(corr) else 0.0
    order = np.argsort(u)
    selective = {}
    for retain in (.5, .7, .9, 1.0):
        take = order[:max(1, round(retain * len(order)))]
        selective[str(retain)] = float(np.sqrt(np.mean(ae[take] ** 2)))
    out["selective_rmse"] = selective
    out["selective_gain_50"] = 1.0 - selective["0.5"] / max(selective["1.0"], 1e-12)
    return out
