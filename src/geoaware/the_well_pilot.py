"""Leakage-safe loader and trivial sanity baselines for The Well pilot subset."""
from __future__ import annotations

from pathlib import Path

import numpy as np


def block_mean_256_to_64(array: np.ndarray) -> np.ndarray:
    """Anti-aliased reduction over the final two spatial axes."""
    if array.shape[-2:] != (256, 256):
        raise ValueError(f"expected 256x256 spatial axes, got {array.shape[-2:]}")
    reshaped = array.reshape(*array.shape[:-2], 64, 4, 64, 4)
    return reshaped.mean(axis=(-3, -1))


def load_the_well_case(path: Path) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Return model inputs and future-pressure targets as separate objects."""
    with np.load(path) as payload:
        pressure = payload["pressure"].astype(np.float32)
        inputs = {
            "density": payload["density"].astype(np.float32),
            "speed_of_sound": payload["speed_of_sound"].astype(np.float32),
            "initial_pressure": pressure[0].copy(),
            "x": payload["x"].astype(np.float32),
            "y": payload["y"].astype(np.float32),
            "query_times": payload["times"][1:].astype(np.float32),
        }
        targets = pressure[1:].copy()
    if targets.shape[0] != len(inputs["query_times"]):
        raise ValueError(f"time/target mismatch in {path}")
    return inputs, targets


def fixed_random_mask(shape: tuple[int, ...], ratio: float, seed: int) -> np.ndarray:
    if not 0 < ratio < 1:
        raise ValueError("ratio must lie strictly between zero and one")
    generator = np.random.default_rng(seed)
    count = max(1, round(ratio*np.prod(shape)))
    chosen = generator.choice(np.prod(shape), size=count, replace=False)
    mask = np.zeros(np.prod(shape), dtype=bool)
    mask[chosen] = True
    return mask.reshape(shape)


def nrmse_on_mask(target: np.ndarray, prediction: np.ndarray,
                  evaluation_mask: np.ndarray) -> float:
    residual = target[evaluation_mask]-prediction[evaluation_mask]
    denominator = np.sqrt(np.mean(target[evaluation_mask]**2)) + 1e-12
    return float(np.sqrt(np.mean(residual**2))/denominator)


def sanity_baselines(inputs: dict[str, np.ndarray], targets: np.ndarray,
                     observation_ratio: float = .01, seed: int = 0) -> dict[str, float]:
    observed = fixed_random_mask(targets.shape, observation_ratio, seed)
    held_out = ~observed
    zero = np.zeros_like(targets)
    persistence = np.broadcast_to(inputs["initial_pressure"], targets.shape)
    observed_mean = np.full_like(targets, float(targets[observed].mean()))
    return {
        "zero_nrmse": nrmse_on_mask(targets, zero, held_out),
        "persistence_nrmse": nrmse_on_mask(targets, persistence, held_out),
        "observed_mean_nrmse": nrmse_on_mask(targets, observed_mean, held_out),
        "realized_observation_ratio": float(observed.mean()),
    }
