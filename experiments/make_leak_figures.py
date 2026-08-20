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


PRETTY = {"random": "anywhere\nin the room", "wall_ring": "all four\nwalls",
          "near_wall": "band inside\nthe walls", "one_wall_strip": "one wall\nonly",
          "corner_block": "one corner\npatch"}


def layout_picture(layout: str, size: int = 64):
    """The sensor-eligible region in the x-y plane, as the reader would draw it."""
    x, y = np.meshgrid(np.arange(size), np.arange(size), indexing="ij")
    depth = np.minimum(np.minimum(x, size - 1 - x), np.minimum(y, size - 1 - y))
    return {"random": np.ones_like(x, dtype=bool),
            "wall_ring": depth < 2,
            "near_wall": (depth >= 3) & (depth < 8),
            "one_wall_strip": x < 5,
            "corner_block": (x < 20) & (y < 20)}[layout]


def figure_layouts(tag: str = "leak_round2_summary.json", seeds: int | None = None) -> None:
    """Two rows: what the sensor layout looks like, and what it costs each method.

    The picture row is the point of the figure.  A reader who sees that one
    layout is a thin strip against a wall understands immediately why the
    reconstruction there is extrapolation, which no arrangement of bars conveys.
    """
    data = load(tag)
    if data is None:
        return
    records = data["records"]
    labels = [r["layout"] for r in records]
    positions = np.arange(len(labels))
    width = 0.2
    count = seeds or len(records[0]["ours_pde"]["values"])

    figure = plt.figure(figsize=(9.6, 5.6))
    grid = figure.add_gridspec(2, len(records), height_ratios=[1.0, 2.6],
                               hspace=0.32, wspace=0.18)
    for column, record in enumerate(records):
        axis = figure.add_subplot(grid[0, column])
        axis.imshow(layout_picture(record["layout"]).T, origin="lower",
                    cmap="Blues", vmin=0, vmax=1.6)
        axis.set_xticks([]); axis.set_yticks([])
        axis.set_title(PRETTY.get(record["layout"], record["layout"]), fontsize=8.5)
        share = 100 * layout_picture(record["layout"]).mean()
        axis.set_xlabel(f"reachable: {share:.0f}% of the room", fontsize=7)

    bars = figure.add_subplot(grid[1, :])
    for index, (key, label, colour) in enumerate(METHODS):
        means = [r[key]["mean"] for r in records]
        errors = [r[key]["std"] / np.sqrt(len(r[key]["values"])) for r in records]
        bars.bar(positions + index * width, means, width, yerr=errors, capsize=2,
                 label=label, color=colour)
    bars.axhline(1.0, color="black", lw=0.9, ls=":")
    bars.text(len(records) - 0.55, 1.02, "predicting the mean", fontsize=7.5, va="bottom")
    bars.set_xticks(positions + width * 1.5)
    bars.set_xticklabels([PRETTY.get(l, l).replace("\n", " ") for l in labels], fontsize=8.5)
    bars.set_ylabel("held-out NRMSE")
    bars.set_title(f"Every layout observes the same 1% of the room "
                   f"({count} seeds); only the arrangement differs", fontsize=10)
    bars.legend(fontsize=7.5, ncol=4, loc="upper left"); bars.grid(axis="y", alpha=0.3)
    bars.set_ylim(0, 1.25)
    figure.savefig(RESULTS / "figure_layouts.png", dpi=160, bbox_inches="tight")
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
    import sys
    layout_tag = sys.argv[1] if len(sys.argv) > 1 else "leak_round2_summary.json"
    curve_tag = sys.argv[2] if len(sys.argv) > 2 else "confinement_summary.json"
    figure_layouts(layout_tag); figure_confinement(curve_tag)
