#!/usr/bin/env python3
"""Audit full signed-frequency separability versus the training octant.

The finite real factors in the POC use nonnegative frequency magnitudes.  This
audit exposes how much easier that projection is than separating the full
signed joint spectrum, especially for tilted advection spectra.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

from geoaware.operator_spectral_funbat import (  # noqa: E402
    nonnegative_cp_spectrum,
    operator_joint_spectrum,
)
from run_submission_confirmation import MAX_FREQUENCY, OPERATOR_CASES  # noqa: E402


def curve(frequencies: torch.Tensor, config: dict) -> dict[str, float]:
    joint = operator_joint_spectrum(**config, frequencies=frequencies)
    return {
        str(rank): nonnegative_cp_spectrum(
            joint, rank=rank, steps=1600, seed=17,
        ).relative_error
        for rank in range(1, 7)
    }


def main() -> None:
    output = ROOT / "results" / "signed_spectrum_audit"
    output.mkdir(parents=True, exist_ok=True)
    positive = torch.arange(MAX_FREQUENCY + 1, dtype=torch.float32)
    signed = torch.arange(-MAX_FREQUENCY, MAX_FREQUENCY + 1, dtype=torch.float32)
    summary = {
        "purpose": (
            "The model atoms use the nonnegative magnitude octant. Full signed "
            "spectra reveal cross-sign coupling that axis-wise real kernels omit."
        ),
        "cases": {},
    }
    for name, config in OPERATOR_CASES.items():
        summary["cases"][name] = {
            "positive_octant": curve(positive, config),
            "full_signed_grid": curve(signed, config),
        }
        print(f"finished {name}", flush=True)
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    figure, axes = plt.subplots(1, 3, figsize=(13.5, 4.0), sharey=True)
    for axis, (name, result) in zip(axes, summary["cases"].items()):
        ranks = list(range(1, 7))
        axis.plot(ranks, [result["positive_octant"][str(r)] for r in ranks], marker="o", label="positive octant")
        axis.plot(ranks, [result["full_signed_grid"][str(r)] for r in ranks], marker="s", label="full signed grid")
        axis.set_title(name.replace("_", "\n"))
        axis.set_xlabel("nonnegative CP rank")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("relative spectrum error")
    axes[-1].legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(output / "signed_vs_octant_separability.png", dpi=190)
    plt.close(figure)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
