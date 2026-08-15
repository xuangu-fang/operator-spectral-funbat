#!/usr/bin/env python3
"""Aggregate raw R1--R5 JSON and render paper-facing POC figures."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

import sys


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "advanced_poc_r1_r5"
sys.path.insert(0, str(ROOT / "src"))

from geoaware.operator_spectral_funbat import generic_spectral_dictionary  # noqa: E402


def aggregate(records: list[dict]) -> dict:
    names = sorted(set.intersection(*(set(record["models"]) for record in records)))
    return {
        name: {
            "mean": float(np.mean(values := [r["models"][name]["test_nrmse"] for r in records])),
            "std": float(np.std(values)),
            "values": values,
        }
        for name in names
    }


def induced_spectrum_metrics(record: dict, model_name: str) -> tuple[float, float]:
    """Compare routed prior spectra, not unstable atom argmax labels."""
    _, spectra_t = generic_spectral_dictionary(6)
    spectra = spectra_t.numpy()
    weights = np.asarray(record["models"][model_name]["routing_weights"])
    oracle_index = np.asarray(record["oracle_routes"])
    oracle = spectra[oracle_index]
    induced = np.einsum("drq,qk->drk", weights, spectra)
    cosine = np.sum(induced * oracle, axis=-1) / (
        np.linalg.norm(induced, axis=-1) * np.linalg.norm(oracle, axis=-1) + 1e-12
    )
    relative_l2 = np.linalg.norm(induced - oracle, axis=-1) / (
        np.linalg.norm(oracle, axis=-1) + 1e-12
    )
    return float(cosine.mean()), float(relative_l2.mean())


def main() -> None:
    planted = {
        ratio: [
            json.loads((RESULTS / f"r1_r2_seed{seed}_ratio{ratio:02d}.json").read_text())
            for seed in (0, 1, 2)
        ]
        for ratio in (1, 2, 5)
    }
    matched = [json.loads((RESULTS / f"r4_r5_matched_seed{s}.json").read_text()) for s in (0, 1, 2)]
    mismatch = [json.loads((RESULTS / f"r4_r5_mismatch_seed{s}.json").read_text()) for s in (0, 1, 2)]
    r3 = json.loads((RESULTS / "r3_separation.json").read_text())
    summary = {
        "protocol": {
            "seeds": [0, 1, 2], "steps": 400, "grid": [24, 24, 24],
            "observation_ratios": [0.01, 0.02, 0.05],
            "split": "fixed train mask per seed; every non-observed entry is test; no validation or early stopping",
        },
        "r1_r2_planted": {str(r / 100): aggregate(planted[r]) for r in (1, 2, 5)},
        "r3_separation": r3,
        "r4_matched_operator_2pct": aggregate(matched),
        "r5_mismatched_operator_2pct": aggregate(mismatch),
    }
    for ratio, records in planted.items():
        ratio_summary = summary["r1_r2_planted"][str(ratio / 100)]
        ratio_summary["route_top1_accuracy"] = {
            "mean": float(np.mean([
                record["models"]["per_mode_rank_routing"]["route_top1_accuracy"]
                for record in records
            ]))
        }
        ratio_summary["induced_prior_spectrum_audit"] = {}
        for model_name in (
            "global_dictionary", "hierarchical_mode_routing", "per_mode_routing",
            "per_mode_rank_routing", "oracle_routing", "swapped_routing",
        ):
            values = [induced_spectrum_metrics(record, model_name) for record in records]
            ratio_summary["induced_prior_spectrum_audit"][model_name] = {
                "cosine_mean": float(np.mean([value[0] for value in values])),
                "relative_l2_mean": float(np.mean([value[1] for value in values])),
            }
    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    selected = [
        "global_dictionary", "hierarchical_mode_routing", "per_mode_routing",
        "per_mode_rank_routing", "oracle_routing", "swapped_routing",
    ]
    labels = {
        "global_dictionary": "global dictionary",
        "hierarchical_mode_routing": "global + shrunk mode",
        "per_mode_routing": "per-mode",
        "per_mode_rank_routing": "per-mode/rank",
        "oracle_routing": "oracle",
        "swapped_routing": "swapped control",
    }
    figure, axis = plt.subplots(figsize=(7.5, 4.5))
    x = np.array([1, 2, 5])
    for name in selected:
        mean = [summary["r1_r2_planted"][str(r / 100)][name]["mean"] for r in x]
        std = [summary["r1_r2_planted"][str(r / 100)][name]["std"] for r in x]
        axis.errorbar(x, mean, yerr=std, marker="o", capsize=3, label=labels[name])
    axis.set(xlabel="observation ratio (%)", ylabel="held-out NRMSE",
             title="Mode-kernel routing: planted sanity (3 seeds)")
    axis.set_xticks(x)
    axis.grid(alpha=0.25)
    axis.legend(ncol=2, fontsize=8)
    figure.tight_layout()
    figure.savefig(RESULTS / "r1_r2_routing_phase.png", dpi=180)
    plt.close(figure)

    names = ["operator_per_mode_rank", "generic_per_mode_rank", "hybrid_per_mode_rank",
             "hybrid_hierarchical", "operator_global", "hybrid_global"]
    x = np.arange(len(names))
    width = 0.36
    figure, axis = plt.subplots(figsize=(8.5, 4.4))
    for offset, key, label in ((-width / 2, "r4_matched_operator_2pct", "matched"),
                               (width / 2, "r5_mismatched_operator_2pct", "mismatched")):
        mean = [summary[key][name]["mean"] for name in names]
        std = [summary[key][name]["std"] for name in names]
        axis.bar(x + offset, mean, width, yerr=std, capsize=3, label=label)
    axis.set_ylabel("held-out NRMSE")
    axis.set_title("Operator dictionary bridge at 2% observations")
    axis.set_xticks(x, [name.replace("_", "\n") for name in names], fontsize=7)
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(RESULTS / "r4_r5_operator_bridge.png", dpi=180)
    plt.close(figure)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
