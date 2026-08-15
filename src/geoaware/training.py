"""Fair masked-field training and evaluation harness."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import copy
import math
import random
import time

import numpy as np
import torch

from .data import FieldDataset
from .masks import ObservationSplit
from .models import BayesianSpectralCP, FieldModel


@dataclass
class TrainConfig:
    steps: int = 2500
    lr: float = 2e-3
    batch_size: int = 4096
    weight_decay: float = 1e-6
    reg_weight: float = 2e-4
    kl_weight: float = 1.0
    kl_warmup: int = 500
    grad_clip: float = 5.0
    eval_samples: int = 32
    seed: int = 0
    log_every: int = 250


@dataclass
class FitResult:
    metrics: dict
    prediction: torch.Tensor
    predictive_std: torch.Tensor | None
    history: list[dict]
    elapsed_seconds: float
    normalization: dict


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parameter_count(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _rankdata(x: torch.Tensor) -> torch.Tensor:
    order = torch.argsort(x)
    ranks = torch.empty_like(order, dtype=torch.float32)
    ranks[order] = torch.arange(len(x), device=x.device, dtype=torch.float32)
    return ranks


def _corr(x: torch.Tensor, y: torch.Tensor) -> float:
    x, y = x.float(), y.float()
    x = x - x.mean(); y = y - y.mean()
    den = torch.sqrt(x.square().sum() * y.square().sum()).clamp_min(1e-12)
    return float((x * y).sum() / den)


def compute_metrics(dataset: FieldDataset, split: ObservationSplit, truth: torch.Tensor,
                    pred: torch.Tensor, std: torch.Tensor | None) -> dict:
    mask = split.held_out
    y, p = truth[mask], pred[mask]
    err = p - y
    rmse = err.square().mean().sqrt()
    rel_l2 = torch.linalg.vector_norm(err) / torch.linalg.vector_norm(y).clamp_min(1e-12)
    metrics = {
        "rmse": float(rmse),
        "nrmse": float(rmse / y.std().clamp_min(1e-12)),
        "relative_l2": float(rel_l2),
        "mae": float(err.abs().mean()),
        "observed_rmse": float((pred[split.observed] - truth[split.observed]).square().mean().sqrt()),
    }
    full_pred = pred.reshape(dataset.shape)
    full_true = truth.reshape(dataset.shape)
    seam = {}
    for d, periodic in enumerate(dataset.periodic):
        if not periodic:
            continue
        first, last = full_pred.select(d, 0), full_pred.select(d, dataset.shape[d] - 1)
        tf, tl = full_true.select(d, 0), full_true.select(d, dataset.shape[d] - 1)
        seam[dataset.mode_names[d]] = {
            "prediction_jump": float((first - last).abs().mean()),
            "excess_jump": float(((first - last) - (tf - tl)).abs().mean()),
        }
    metrics["periodic_seam"] = seam
    if std is not None:
        s = std[mask].clamp_min(1e-6)
        z = err.abs() / s
        metrics.update({
            "gaussian_nll": float((0.5 * (err / s).square() + torch.log(s) +
                                   0.5 * math.log(2 * math.pi)).mean()),
            "coverage_95": float((z <= 1.96).float().mean()),
            "mean_predictive_std": float(s.mean()),
            "uncertainty_error_spearman": _corr(_rankdata(s), _rankdata(err.abs())),
        })
    return metrics


@torch.no_grad()
def predict_full(model: FieldModel, coords: torch.Tensor, indices: torch.Tensor,
                 *, mean: float, scale: float, samples: int, chunk_size: int = 262144):
    model.eval()
    if model.is_bayesian:
        mean_chunks = [model(coords[i:i + chunk_size], indices[i:i + chunk_size],
                             sample=False).cpu()
                       for i in range(0, len(coords), chunk_size)]
        posterior_mean = torch.cat(mean_chunks)
        draws = []
        for _ in range(samples):
            chunks = [model(coords[i:i + chunk_size], indices[i:i + chunk_size], sample=True).cpu()
                      for i in range(0, len(coords), chunk_size)]
            draws.append(torch.cat(chunks))
        draw = torch.stack(draws) * scale + mean
        # Independence of the mean-field factor posteriors gives an exact mean
        # equal to the product of factor means.  Using it avoids Monte-Carlo
        # bias from products of uncertain CP factors.
        pred = posterior_mean * scale + mean
        epistemic = (draw - pred).square().mean(0).sqrt()
        noise = float(model.noise_std.detach().cpu()) * scale
        return pred, torch.sqrt(epistemic.square() + noise * noise)
    chunks = [model(coords[i:i + chunk_size], indices[i:i + chunk_size], sample=False).cpu()
              for i in range(0, len(coords), chunk_size)]
    return torch.cat(chunks) * scale + mean, None


def fit_model(model: FieldModel, dataset: FieldDataset, split: ObservationSplit,
              cfg: TrainConfig, device: str = "cuda") -> FitResult:
    seed_everything(cfg.seed)
    started = time.perf_counter()
    device = torch.device(device if device != "cuda" or torch.cuda.is_available() else "cpu")
    coords_all = dataset.flat_coordinates()
    indices_all = dataset.flat_indices()
    truth = dataset.values.reshape(-1).float()
    obs_ids = torch.where(split.observed)[0]
    # Only observed values determine preprocessing statistics.
    center = float(truth[obs_ids].mean())
    scale = float(truth[obs_ids].std().clamp_min(1e-6))
    target = (truth - center) / scale
    coords_obs = coords_all[obs_ids].to(device)
    indices_obs = indices_all[obs_ids].to(device)
    y_obs = target[obs_ids].to(device)
    model = model.to(device)
    if hasattr(model, "fit_posterior"):
        model.fit_posterior(coords_obs, y_obs)
        history = [{"step": 0, "loss": float("nan"), "data_loss": float("nan"),
                    "complexity": float("nan"), "lr": 0.0}]
        pred, pred_std = predict_full(model, coords_all.to(device), indices_all.to(device),
                                      mean=center, scale=scale, samples=cfg.eval_samples)
        metrics = compute_metrics(dataset, split, truth, pred, pred_std)
        metrics["parameters"] = parameter_count(model)
        metrics["training_points"] = int(split.observed.sum())
        metrics["observation_ratio"] = split.ratio_actual
        metrics["final_train_loss"] = None
        metrics["spectral_summary"] = model.spectral_summary()
        return FitResult(metrics, pred.reshape(dataset.shape), pred_std.reshape(dataset.shape),
                         history, time.perf_counter() - started,
                         {"observed_mean": center, "observed_std": scale, "config": asdict(cfg)})
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg.steps, eta_min=cfg.lr * 0.05)
    generator = torch.Generator(device=device).manual_seed(cfg.seed + 9187)
    history: list[dict] = []
    best_loss, best_state = float("inf"), None

    for step in range(cfg.steps):
        if len(obs_ids) > cfg.batch_size:
            sel = torch.randint(len(obs_ids), (cfg.batch_size,), generator=generator, device=device)
            x, ix, y = coords_obs[sel], indices_obs[sel], y_obs[sel]
        else:
            x, ix, y = coords_obs, indices_obs, y_obs
        pred = model(x, ix, sample=True)
        if model.is_bayesian:
            noise = model.noise_std
            data_loss = (0.5 * ((pred - y) / noise).square() + torch.log(noise)).mean()
            beta = cfg.kl_weight * min(1.0, (step + 1) / max(cfg.kl_warmup, 1))
            complexity = beta * model.kl_divergence() / len(obs_ids)
        else:
            data_loss = (pred - y).square().mean()
            complexity = cfg.reg_weight * model.regularization()
        loss = data_loss + complexity
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step(); scheduler.step()
        value = float(loss.detach())
        if value < best_loss and step > cfg.kl_warmup:
            best_loss = value
            best_state = copy.deepcopy(model.state_dict())
        if step % cfg.log_every == 0 or step == cfg.steps - 1:
            history.append({"step": step, "loss": value,
                            "data_loss": float(data_loss.detach()),
                            "complexity": float(complexity.detach()),
                            "lr": scheduler.get_last_lr()[0]})
    if best_state is not None:
        model.load_state_dict(best_state)
    pred, pred_std = predict_full(model, coords_all.to(device), indices_all.to(device),
                                  mean=center, scale=scale, samples=cfg.eval_samples)
    metrics = compute_metrics(dataset, split, truth, pred, pred_std)
    metrics["parameters"] = parameter_count(model)
    metrics["training_points"] = int(split.observed.sum())
    metrics["observation_ratio"] = split.ratio_actual
    metrics["final_train_loss"] = history[-1]["loss"]
    if hasattr(model, "spectral_summary"):
        metrics["spectral_summary"] = model.spectral_summary()
    return FitResult(metrics, pred.reshape(dataset.shape),
                     None if pred_std is None else pred_std.reshape(dataset.shape),
                     history, time.perf_counter() - started,
                     {"observed_mean": center, "observed_std": scale, "config": asdict(cfg)})
