#!/usr/bin/env python3
"""Run only the shrunk global-to-mode bridge and merge it into R1--R5 JSON."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

from run_operator_spectral_poc import run_operator_case, run_planted  # noqa: E402


def main() -> None:
    output = ROOT / "results" / "advanced_poc_r1_r5"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for seed in (0, 1, 2):
        for ratio in (0.01, 0.02, 0.05):
            addition = run_planted(
                seed, ratio, 400, device, only_models={"hierarchical_mode_routing"}
            )
            path = output / f"r1_r2_seed{seed}_ratio{int(100*ratio):02d}.json"
            record = json.loads(path.read_text(encoding="utf-8"))
            record["models"].update(addition["models"])
            path.write_text(json.dumps(record, indent=2), encoding="utf-8")
        for truth, prior, label in (
            ("advection", "advection", "matched"),
            ("advection", "diffusion", "mismatch"),
        ):
            addition = run_operator_case(
                seed=seed, truth_operator=truth, prior_operator=prior,
                ratio=0.02, steps=400, device=device,
                only_models={"hybrid_hierarchical"},
            )
            path = output / f"r4_r5_{label}_seed{seed}.json"
            record = json.loads(path.read_text(encoding="utf-8"))
            record["models"].update(addition["models"])
            path.write_text(json.dumps(record, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
