#!/usr/bin/env python3
"""Add the spectral-support mismatch control omitted from the first R5 sweep."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))
from run_operator_spectral_poc import run_operator_case  # noqa: E402


def main() -> None:
    output = ROOT / "results" / "advanced_poc_r1_r5"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    selected = {"wrong_support_operator", "wrong_support_hybrid"}
    for seed in (0, 1, 2):
        for truth, prior, label in (
            ("advection", "advection", "matched"),
            ("advection", "diffusion", "mismatch"),
        ):
            addition = run_operator_case(
                seed=seed, truth_operator=truth, prior_operator=prior,
                ratio=0.02, steps=400, device=device, only_models=selected,
            )
            path = output / f"r4_r5_{label}_seed{seed}.json"
            record = json.loads(path.read_text(encoding="utf-8"))
            record["models"].update(addition["models"])
            path.write_text(json.dumps(record, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
