#!/usr/bin/env python3
"""Rerun only strict-support controls after fixing atom-dependent basis loss."""

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
    ESCAPE_FLOOR,
    OPERATOR_CASES,
    SEEDS,
    STEPS,
    aggregate,
    build_operator_atoms,
    render_figure,
    run_case,
)


SELECTED = {"wrong_support_operator", "wrong_support_robust"}


def update_record(
    path: Path,
    *,
    case_name: str,
    config: dict,
    atoms: torch.Tensor,
    separation: dict,
    seed: int,
    device: torch.device,
) -> dict:
    record = json.loads(path.read_text(encoding="utf-8"))
    before = {
        name: record["models"][name]["test_nrmse"] for name in SELECTED
    }
    replacement = run_case(
        case_name, config, atoms, separation, seed, device, STEPS,
        escape_floor=ESCAPE_FLOOR, only_methods=SELECTED,
    )
    record["models"].update(replacement["models"])
    record["strict_support_basis_fix"] = (
        "Rerun after replacing atom-derived Fourier basis with an analytic, "
        "atom-independent real basis; all other protocol fields frozen."
    )
    path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    return {
        "case": case_name,
        "seed": seed,
        "before_invalid": before,
        "after_fixed": {
            name: record["models"][name]["test_nrmse"] for name in SELECTED
        },
    }


def refresh_development_summary(output: Path) -> None:
    records = [
        json.loads((output / f"floor25_seed{seed}.json").read_text())
        for seed in DEVELOPMENT_SEEDS
    ]
    methods = {}
    for method in records[0]["models"]:
        values = [record["models"][method]["test_nrmse"] for record in records]
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
    summary = {
        "development_seeds": list(DEVELOPMENT_SEEDS),
        "floors": {
            str(ESCAPE_FLOOR): {
                "methods": methods,
                "paired": {
                    "robust_matched_penalty_mean": float(matched_penalty.mean()),
                    "robust_wrong_support_gain_mean": float(wrong_gain.mean()),
                    "wrong_support_escape_wins": int((wrong_gain > 0).sum()),
                    "matched_robust_wins_vs_operator": int((matched_penalty < 0).sum()),
                },
            }
        },
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    prepared = {
        name: build_operator_atoms(config) for name, config in OPERATOR_CASES.items()
    }
    audit = []

    development = ROOT / "results" / "escape_floor_operator_init_development"
    config = OPERATOR_CASES["reference_advection"]
    atoms, separation = prepared["reference_advection"]
    for seed in DEVELOPMENT_SEEDS:
        audit.append(update_record(
            development / f"floor25_seed{seed}.json",
            case_name="reference_advection", config=config, atoms=atoms,
            separation=separation, seed=seed, device=device,
        ))
        print(f"fixed development seed={seed}", flush=True)
    refresh_development_summary(development)

    confirmation = ROOT / "results" / "submission_confirmation"
    final_records = []
    for case_name, config in OPERATOR_CASES.items():
        atoms, separation = prepared[case_name]
        for seed in SEEDS:
            path = confirmation / f"{case_name}_seed{seed}.json"
            audit.append(update_record(
                path, case_name=case_name, config=config, atoms=atoms,
                separation=separation, seed=seed, device=device,
            ))
            final_records.append(json.loads(path.read_text()))
            print(f"fixed {case_name} seed={seed}", flush=True)
    summary = aggregate(final_records)
    (confirmation / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    render_figure(summary, confirmation)
    (ROOT / "results" / "strict_support_basis_fix_audit.json").write_text(
        json.dumps({"reason": __doc__, "records": audit}, indent=2), encoding="utf-8",
    )


if __name__ == "__main__":
    main()
