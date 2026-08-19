#!/usr/bin/env python3
"""Figures for the forced-PDE experiments.

Reads only the summary JSONs, so figures never disagree with the recorded
numbers.  Missing inputs are skipped with a message rather than faked.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "forced_pde"


def load(name: str):
    path = RESULTS / name
    if not path.exists():
        print(f"skip: {path.name} not found")
        return None
    return json.loads(path.read_text())


def figure_main(tag: str = "main_fixedforcing_summary.json") -> None:
    data = load(tag)
    if data is None:
        return
    ratios = sorted({r["ratio"] for r in data["records"]})
    series = {"operator": [], "generic": [], "nearest_neighbour": []}
    errors = {k: [] for k in series}
    for ratio in ratios:
        cells = [r for r in data["records"] if r["ratio"] == ratio]
        for key in series:
            values = [c[key]["test_nrmse"] if isinstance(c[key], dict) else c[key]
                      for c in cells]
            series[key].append(np.mean(values))
            errors[key].append(np.std(values) / np.sqrt(len(values)))

    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.1))
    labels = {"operator": "PDE-form kernels (ours)", "generic": "generic dictionary",
              "nearest_neighbour": "nearest neighbour"}
    for key, colour in zip(series, ("tab:blue", "tab:orange", "tab:grey")):
        axes[0].errorbar([100 * r for r in ratios], series[key], yerr=errors[key],
                         marker="o", capsize=3, label=labels[key], color=colour)
    axes[0].set_xlabel("observed entries (%)")
    axes[0].set_ylabel("held-out NRMSE")
    axes[0].set_title("Sparse reconstruction")
    axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3)

    margins = [g - o for g, o in zip(series["generic"], series["operator"])]
    axes[1].bar([f"{100 * r:g}%" for r in ratios], margins, color="tab:blue", alpha=0.8)
    axes[1].axhline(0, color="black", lw=0.8)
    axes[1].set_ylabel("NRMSE reduction vs generic")
    axes[1].set_title("The prior earns its keep where data is scarce")
    axes[1].grid(axis="y", alpha=0.3)
    figure.tight_layout()
    figure.savefig(RESULTS / "figure_main.png", dpi=160)
    plt.close(figure)
    print("wrote figure_main.png")


def figure_baselines() -> None:
    data = load("baselines_summary.json")
    main = load("main_fixedforcing_summary.json")
    if data is None or main is None:
        return
    ratios = sorted(float(r) for r in data["aggregate"])
    keep = ["global_mean", "em_cp_rank5", "em_tucker", "rbf_best_oracle_lengthscale"]
    labels = {"global_mean": "global mean", "em_cp_rank5": "discrete CP (EM)",
              "em_tucker": "discrete Tucker (EM)",
              "rbf_best_oracle_lengthscale": "kernel ridge (oracle length scale)"}
    figure, axis = plt.subplots(figsize=(6.6, 4.2))
    for key in keep:
        axis.plot([100 * r for r in ratios],
                  [data["aggregate"][str(r)][key] for r in ratios],
                  marker="s", ls="--", lw=1.2, label=labels[key])
    ours, generic = [], []
    for ratio in ratios:
        cells = [r for r in main["records"] if abs(r["ratio"] - ratio) < 1e-9]
        ours.append(np.mean([c["operator"]["test_nrmse"] for c in cells]))
        generic.append(np.mean([c["generic"]["test_nrmse"] for c in cells]))
    axis.plot([100 * r for r in ratios], ours, marker="o", lw=2.2,
              color="tab:blue", label="PDE-form kernels (ours)")
    axis.plot([100 * r for r in ratios], generic, marker="o", lw=1.6,
              color="tab:orange", label="generic dictionary")
    axis.axhline(1.0, color="red", lw=0.8, ls=":", label="predicting the mean")
    axis.set_xlabel("observed entries (%)"); axis.set_ylabel("held-out NRMSE")
    axis.set_title("Baselines on the identical masks")
    axis.legend(fontsize=7.5); axis.grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(RESULTS / "figure_baselines.png", dpi=160)
    plt.close(figure)
    print("wrote figure_baselines.png")


def figure_no_routing() -> None:
    data = load("no_routing_summary.json")
    if data is None:
        return
    ratios = sorted(float(r) for r in data["ratios"])
    arms = ["operator_marginal_fixed", "operator_separated_routed",
            "se_oracle_per_mode", "se_oracle_shared",
            "generic_dictionary_global", "generic_dictionary_routed"]
    labels = {"operator_marginal_fixed": "PDE kernel, no tuning (ours)",
              "operator_separated_routed": "PDE bank + routing",
              "se_oracle_per_mode": "SE kernel, oracle-tuned per mode",
              "se_oracle_shared": "SE kernel, oracle-tuned shared",
              "generic_dictionary_global": "generic dictionary, global",
              "generic_dictionary_routed": "generic dictionary, routed"}
    figure, axis = plt.subplots(figsize=(7.4, 4.3))
    width = 0.13
    positions = np.arange(len(ratios))
    for index, arm in enumerate(arms):
        if arm not in data["ratios"][str(ratios[0])]:
            continue
        means = [data["ratios"][str(r)][arm]["mean"] for r in ratios]
        errs = [data["ratios"][str(r)][arm]["std"] / np.sqrt(
            len(data["ratios"][str(r)][arm]["values"])) for r in ratios]
        axis.bar(positions + index * width, means, width, yerr=errs, capsize=2,
                 label=labels[arm])
    axis.set_xticks(positions + width * (len(arms) - 1) / 2)
    axis.set_xticklabels([f"{100 * r:g}%" for r in ratios])
    axis.set_xlabel("observed entries"); axis.set_ylabel("held-out NRMSE")
    axis.set_title("A derived kernel versus a tuned one")
    axis.legend(fontsize=7); axis.grid(axis="y", alpha=0.3)
    figure.tight_layout()
    figure.savefig(RESULTS / "figure_no_routing.png", dpi=160)
    plt.close(figure)
    print("wrote figure_no_routing.png")




def figure_headline() -> None:
    """Per-seed paired scatter: the claim stands or falls on this one."""
    data = load("headline_summary.json")
    if data is None:
        return
    rows = []
    for name in ("main_fixedforcing_summary.json", "main_extraseeds_summary.json"):
        block = load(name)
        if block is None:
            continue
        rows += [r for r in block["records"] if abs(r["ratio"] - data["ratio"]) < 1e-9]
    operator = np.array([r["operator"]["test_nrmse"] for r in rows])
    generic = np.array([r["generic"]["test_nrmse"] for r in rows])
    figure, axes = plt.subplots(1, 2, figsize=(10.0, 4.2))
    limits = [min(operator.min(), generic.min()) - 0.02,
              max(operator.max(), generic.max()) + 0.02]
    axes[0].plot(limits, limits, color="grey", lw=0.9, ls="--")
    axes[0].scatter(generic, operator, s=42, color="tab:blue", zorder=3)
    axes[0].set_xlim(limits); axes[0].set_ylim(limits)
    axes[0].set_xlabel("generic dictionary, held-out NRMSE")
    axes[0].set_ylabel("PDE-form kernels, held-out NRMSE")
    axes[0].set_title(f"Paired, {data['seeds']} seeds at "
                      f"{100 * data['ratio']:g}% observed ({data['paired_wins']})")
    axes[0].grid(alpha=0.3)
    axes[0].text(0.04, 0.93, "below the line = ours wins", transform=axes[0].transAxes,
                 fontsize=8, color="tab:blue")

    margin = generic - operator
    axes[1].hist(margin, bins=10, color="tab:blue", alpha=0.85)
    axes[1].axvline(0, color="black", lw=1.0)
    axes[1].axvline(margin.mean(), color="tab:red", lw=1.4,
                    label=f"mean {margin.mean():+.4f}")
    axes[1].set_xlabel("paired NRMSE reduction")
    axes[1].set_ylabel("seeds")
    axes[1].set_title(f"sign test $p$ = {data['sign_test_p']:.1e}")
    axes[1].legend(fontsize=8); axes[1].grid(axis="y", alpha=0.3)
    figure.tight_layout()
    figure.savefig(RESULTS / "figure_headline.png", dpi=160)
    plt.close(figure)
    print("wrote figure_headline.png")


if __name__ == "__main__":
    tag = sys.argv[1] if len(sys.argv) > 1 else "main_fixedforcing_summary.json"
    figure_main(tag); figure_baselines(); figure_no_routing(); figure_headline()
