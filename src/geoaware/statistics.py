"""Small, dependency-light statistics used by both paper campaigns."""

from __future__ import annotations

import itertools
import math

import numpy as np


def paired_seed_summary(
    proposed: dict[int, float], baseline: dict[int, float], *,
    bootstrap_samples: int = 20_000, seed: int = 20260812,
) -> dict:
    """Summarize paired lower-is-better seed aggregates without pseudo-replication.

    The paired permutation test flips the sign of each seed-level difference.
    For at most 16 seeds it enumerates every sign assignment exactly; otherwise
    it uses deterministic Monte Carlo.  The confidence interval resamples seed
    pairs, never individual tasks nested inside a seed.
    """
    common = sorted(set(proposed) & set(baseline))
    if len(common) < 2:
        raise ValueError("at least two paired seeds are required")
    a = np.asarray([proposed[s] for s in common], dtype=np.float64)
    b = np.asarray([baseline[s] for s in common], dtype=np.float64)
    if not (np.isfinite(a).all() and np.isfinite(b).all()):
        raise ValueError("metrics must be finite")
    difference = a - b
    observed = abs(float(difference.mean()))
    n = len(common)
    if n <= 16:
        permuted = np.fromiter(
            (abs(float(np.mean(difference * np.asarray(signs))))
             for signs in itertools.product((-1.0, 1.0), repeat=n)),
            dtype=np.float64,
        )
    else:
        rng = np.random.default_rng(seed)
        signs = rng.choice((-1.0, 1.0), size=(bootstrap_samples, n))
        permuted = np.abs((signs * difference[None]).mean(1))
    # Including equality makes the test conservative and exact.
    p_value = float(np.mean(permuted >= observed - 1e-15))
    rng = np.random.default_rng(seed + 1)
    sampled = rng.integers(0, n, size=(bootstrap_samples, n))
    boot_difference = difference[sampled].mean(1)
    relative = (b[sampled].mean(1) - a[sampled].mean(1)) / np.maximum(
        np.abs(b[sampled].mean(1)), 1e-12
    )
    return {
        "seeds": common,
        "n_seeds": n,
        "proposed_mean": float(a.mean()),
        "baseline_mean": float(b.mean()),
        "paired_difference": float(difference.mean()),
        "paired_difference_ci95": [float(x) for x in np.quantile(boot_difference, [.025, .975])],
        "relative_improvement": float((b.mean() - a.mean()) / max(abs(b.mean()), 1e-12)),
        "relative_improvement_ci95": [float(x) for x in np.quantile(relative, [.025, .975])],
        "two_sided_paired_permutation_p": p_value,
        "exact_permutation": n <= 16,
    }
