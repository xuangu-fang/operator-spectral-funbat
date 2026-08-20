#!/usr/bin/env python3
"""Figures for the sensor-placement story, read only from recorded summaries."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "leak"
METHODS = [("ours_pde", "PDE-form kernels (ours)", "tab:blue"),
           ("matern", "Matern", "tab:orange"),
           ("spectral_mixture", "spectral mixture", "tab:green"),
           ("neural_tucker", "neural functional Tucker", "tab:red")]


def load(name):
    path = RESULTS / name
    if not path.exists():
        print(f"skip: {name} not found")
        return None
    return json.loads(path.read_text())


def figure_layouts(tag: str = "leak_round2_summary.json") -> None:
    data = load(tag)
    if data is None:
        return
    records = data["records"]
    labels = [r["layout"] for r in records]
    positions = np.arange(len(labels))
    width = 0.2
    figure, axis = plt.subplots(figsize=(9.0, 4.3))
    for index, (key, label, colour) in enumerate(METHODS):
        means = [r[key]["mean"] for r in records]
        errors = [r[key]["std"] / np.sqrt(len(r[key]["values"])) for r in records]
        axis.bar(positions + index * width, means, width, yerr=errors, capsize=2,
                 label=label, color=colour)
    axis.axhline(1.0, color="black", lw=0.8, ls=":", label="predicting the mean")
    axis.set_xticks(positions + width * 1.5)
    axis.set_xticklabels(labels, fontsize=8)
    axis.set_ylabel("held-out NRMSE")
    axis.set_title("Sensors you can actually place (1% observed, 3 seeds)")
    axis.legend(fontsize=7.5); axis.grid(axis="y", alpha=0.3)
    figure.tight_layout()
    figure.savefig(RESULTS / "figure_layouts.png", dpi=160)
    plt.close(figure)
    print("wrote figure_layouts.png")


def figure_confinement(tag: str = "confinement_summary.json") -> None:
    data = load(tag)
    if data is None:
        return
    records = sorted(data["records"], key=lambda r: -r["strip_width"])
    fraction = [100 * r["fraction_of_room"] for r in records]
    figure, axes = plt.subplots(1, 2, figsize=(11.0, 4.3))
    for key, label, colour in METHODS:
        means = [r[key]["mean"] for r in records]
        errors = [r[key]["std"] / np.sqrt(len(r[key]["values"])) for r in records]
        axes[0].errorbar(fraction, means, yerr=errors, marker="o", capsize=3,
                         label=label, color=colour)
    axes[0].axhline(1.0, color="black", lw=0.8, ls=":")
    axes[0].set_xlabel("share of the room the sensors may occupy (%)")
    axes[0].set_ylabel("held-out NRMSE")
    axes[0].set_title("Reconstruction as sensors are confined")
    axes[0].invert_xaxis(); axes[0].legend(fontsize=7.5); axes[0].grid(alpha=0.3)

    margins = [100 * r["relative_percent"] / 100 for r in records]
    axes[1].plot(fraction, margins, marker="o", color="tab:blue", lw=2)
    axes[1].axhline(0, color="black", lw=0.8)
    axes[1].set_xlabel("share of the room the sensors may occupy (%)")
    axes[1].set_ylabel("improvement over the best baseline (%)")
    axes[1].set_title("Physics pays where extrapolation is forced")
    axes[1].invert_xaxis(); axes[1].grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(RESULTS / "figure_confinement.png", dpi=160)
    plt.close(figure)
    print("wrote figure_confinement.png")


if __name__ == "__main__":
    figure_layouts(); figure_confinement()
