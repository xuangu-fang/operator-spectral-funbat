#!/usr/bin/env python3
"""Development-only sweep for the fixed generic-support floor.

Seeds 101--105 were already exposed during the collapsed-parameterization audit
and must never be included in final confirmation.  A free/uniform robust route
already failed.  This script performs one minimal retry with a predeclared 25%
generic floor and operator-centred logits before confirmation on seeds 201--205.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

from run_submission_confirmation import (  # noqa: E402
    DEVELOPMENT_SEEDS,
    OPERATOR_CASES,
    STEPS,
    build_operator_atoms,
    run_case,
)


# One minimal retry after the free/uniform robust routing audit failed.  We do
# not search a large grid: 25% is the single predeclared support floor.
FLOORS = (0.25,)
METHODS = {
    "operator_per_mode_rank",
    "generic_per_mode_rank",
    "robust_per_mode_rank",
    "wrong_support_operator",
    "wrong_support_robust",
}


def main() -> None:
    output = ROOT / "results" / "escape_floor_operator_init_development"
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = OPERATOR_CASES["reference_advection"]
    atoms, separation = build_operator_atoms(config)
    records = []
    for floor in FLOORS:
        for seed in DEVELOPMENT_SEEDS:
            record = run_case(
                "reference_advection", config, atoms, separation, seed, device,
                STEPS, escape_floor=floor, only_methods=METHODS,
            )
            records.append(record)
            path = output / f"floor{int(100 * floor):02d}_seed{seed}.json"
            path.write_text(json.dumps(record, indent=2), encoding="utf-8")
            print(f"finished floor={floor:.2f} seed={seed}", flush=True)

    summary = {"development_seeds": list(DEVELOPMENT_SEEDS), "floors": {}}
    for floor in FLOORS:
        group = [record for record in records if record["protocol"]["generic_escape_floor_total"] == floor]
        methods = {}
        for method in sorted(METHODS):
            values = [record["models"][method]["test_nrmse"] for record in group]
            methods[method] = {
                "mean": float(np.mean(values)), "std": float(np.std(values)),
                "values": values,
            }
        matched_penalty = np.asarray(methods["robust_per_mode_rank"]["values"]) - np.asarray(
            methods["operator_per_mode_rank"]["values"]
        )
        wrong_gain = np.asarray(methods["wrong_support_operator"]["values"]) - np.asarray(
            methods["wrong_support_robust"]["values"]
        )
        summary["floors"][str(floor)] = {
            "methods": methods,
            "paired": {
                "robust_matched_penalty_mean": float(matched_penalty.mean()),
                "robust_wrong_support_gain_mean": float(wrong_gain.mean()),
                "wrong_support_escape_wins": int((wrong_gain > 0).sum()),
                "matched_robust_wins_vs_operator": int((matched_penalty < 0).sum()),
            },
        }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
