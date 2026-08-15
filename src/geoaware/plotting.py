"""Compact POC figures with enough information to audit sparse reconstruction."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from .data import FieldDataset
from .masks import ObservationSplit


def plot_reconstruction(dataset: FieldDataset, split: ObservationSplit, prediction: torch.Tensor,
                        predictive_std: torch.Tensor | None, path: Path, title: str):
    truth = dataset.values.detach().cpu()
    pred = prediction.detach().cpu()
    mask = split.observed.reshape(dataset.shape).cpu()
    while truth.ndim > 2:
        mid = truth.shape[0] // 2
        truth, pred, mask = truth[mid], pred[mid], mask[mid]
        if predictive_std is not None:
            predictive_std = predictive_std[mid]
    panels = [(truth, "truth"), (pred, "prediction"), ((pred - truth).abs(), "absolute error")]
    if predictive_std is not None:
        panels.append((predictive_std.detach().cpu(), "predictive std"))
    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 3.6), constrained_layout=True)
    for ax, (arr, label) in zip(np.atleast_1d(axes), panels):
        im = ax.imshow(arr.numpy(), origin="lower", aspect="auto", cmap="viridis")
        ax.set_title(label); fig.colorbar(im, ax=ax, fraction=0.046)
        if label == "truth":
            yy, xx = torch.where(mask)
            if len(xx) < 3000:
                ax.scatter(xx.numpy(), yy.numpy(), s=2, c="white", alpha=0.45)
    fig.suptitle(title)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_comparison(dataset: FieldDataset, rows: list[dict], path: Path):
    labels = [r["model"] for r in rows]
    vals = [r["metrics"]["relative_l2"] for r in rows]
    fig, ax = plt.subplots(figsize=(max(6, 1.2 * len(rows)), 4), constrained_layout=True)
    bars = ax.bar(labels, vals, color=["#4c78a8" if "geo" not in x and "bayesian" not in x
                                      else "#e45756" for x in labels])
    ax.bar_label(bars, fmt="%.3f", padding=3)
    ax.set_ylabel("held-out relative L2 (lower is better)")
    ax.set_title(f"{dataset.name}: {rows[0]['mask']} at {100*rows[0]['ratio']:.2g}% observations")
    ax.tick_params(axis="x", rotation=25)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)
