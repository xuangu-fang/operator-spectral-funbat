#!/usr/bin/env python3
"""The paper's headline figure: the argument in one row of panels.

Three panels, left to right, each answering one question a reader has:

  what changes    the sensor layouts, drawn, with the reach each one has.
  what it costs   the price of having to choose a kernel from data you can
                  collect, which is the paper's quantity, across layouts and
                  across two other operator families and a 3-D room.
  what is needed  the knowledge ladder, showing that a declared range for the
                  coefficients is nearly as good as knowing them.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
from make_leak_figures import layout_picture  # noqa: E402

RESULTS = ROOT / "results" / "leak"
SHOW = ["random", "wall_ring", "near_wall", "one_wall_strip", "corner_block"]
SHORT = {"random": "anywhere", "wall_ring": "four walls", "near_wall": "inner band",
         "one_wall_strip": "one wall", "corner_block": "one corner"}


def load(name):
    path = RESULTS / name
    return json.loads(path.read_text()) if path.exists() else None


def main() -> None:
    main_data = load("leak_main3tier_summary.json")
    ops = load("operator_families_summary.json")
    three = load("leak3d_summary.json")
    ladder = load("knowledge_ladder_leak_summary.json")
    if main_data is None:
        print("main summary missing"); return

    figure = plt.figure(figsize=(15.0, 4.6))
    grid = figure.add_gridspec(1, 3, width_ratios=[1.15, 1.35, 1.0], wspace=0.28)

    # ---- panel 1: the layouts, drawn ----
    left = grid[0, 0].subgridspec(2, 3, hspace=0.55, wspace=0.15)
    for index, layout in enumerate(SHOW):
        axis = figure.add_subplot(left[index // 3, index % 3])
        picture = layout_picture(layout)
        axis.imshow(picture.T, origin="lower", cmap="Blues", vmin=0, vmax=1.6)
        axis.set_xticks([]); axis.set_yticks([])
        axis.set_title(f"{SHORT[layout]}\n{100 * picture.mean():.0f}% reachable",
                       fontsize=8.5)
    note = figure.add_subplot(left[1, 2]); note.axis("off")
    note.text(0.0, 0.55, "every layout\nobserves the same\n1% of the room",
              fontsize=8.5, va="center")
    figure.text(0.055, 0.955, "a  where sensors may go", fontsize=11.5, weight="bold")

    # ---- panel 2: the cost of having to tune ----
    axis = figure.add_subplot(grid[0, 1])
    rows, labels, colours = [], [], []
    present = {r["layout"]: r for r in main_data["records"]}
    for layout in SHOW:
        if layout in present:
            rows.append(present[layout]["tuning_cost"]); labels.append(SHORT[layout])
            colours.append("tab:blue")
    if ops is not None:
        for record in ops["records"]:
            if record["layout"] != "one_wall_strip":
                continue
            rows.append(record["tuning_cost"])
            labels.append(record["family"].replace("-", "-\n") + "\none wall")
            colours.append("tab:purple")
    if three is not None:
        for record in three["records"]:
            if record["layout"] != "one_face":
                continue
            rows.append(record["tuning_cost"]); labels.append("3-D room\none face")
            colours.append("tab:red")
    positions = np.arange(len(rows))
    axis.barh(positions, rows, color=colours)
    axis.set_yticks(positions); axis.set_yticklabels(labels, fontsize=8.5)
    axis.invert_yaxis()
    axis.set_xlabel("NRMSE lost by choosing the kernel from data you can collect")
    axis.grid(axis="x", alpha=0.3)
    axis.axvline(0, color="black", lw=0.8)
    for position, value in zip(positions, rows):
        axis.text(value + 0.004, position, f"{value:+.3f}", va="center", fontsize=8)
    axis.set_xlim(min(0, min(rows) * 1.2), max(rows) * 1.28)
    figure.text(0.375, 0.955, "b  what it costs to have to tune", fontsize=11.5,
                weight="bold")

    # ---- panel 3: the knowledge ladder ----
    axis = figure.add_subplot(grid[0, 2])
    if ladder is not None:
        arms = [("K2 true coefficients", "knows $\\theta^\\star$"),
                ("K1 bank x[1/3,3]", "knows $\\theta^\\star\\in\\times[1/3,3]$"),
                ("K0 bank x[1/10,10]", "knows $\\theta^\\star\\in\\times[1/10,10]$"),
                ("K-1 generic, matched atoms", "no physics")]
        by_layout = {r["layout"]: r for r in ladder["records"]}
        for layout, colour, marker in (("one_wall_strip", "tab:blue", "o"),
                                       ("near_wall", "tab:cyan", "s"),
                                       ("random", "0.6", "^")):
            record = by_layout.get(layout)
            if record is None:
                continue
            values = [record[key]["mean"] for key, _ in arms]
            axis.plot(range(len(arms)), values, marker=marker, color=colour,
                      label=SHORT[layout], lw=2)
        axis.set_xticks(range(len(arms)))
        axis.set_xticklabels([label for _, label in arms], fontsize=8, rotation=18,
                             ha="right")
        axis.set_ylabel("held-out NRMSE")
        axis.legend(fontsize=8.5, title="sensors", title_fontsize=8.5)
        axis.grid(alpha=0.3)
    figure.text(0.70, 0.955, "c  how much of the equation is needed", fontsize=11.5,
                weight="bold")

    figure.savefig(RESULTS / "figure_headline_main.png", dpi=170, bbox_inches="tight")
    plt.close(figure)
    print("wrote figure_headline_main.png")


if __name__ == "__main__":
    main()
