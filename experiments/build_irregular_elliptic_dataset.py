#!/usr/bin/env python3
"""Build and audit the method-matched irregular-domain elliptic tensor gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from geoaware.irregular_domain_solver import generate_irregular_elliptic_dataset


def audit(output: Path, manifest: dict) -> dict:
    finite, stds, residuals, node_counts = [], [], [], []
    for case in manifest["cases"]:
        payload = np.load(output/case["file"])
        field = payload["field"]
        finite.append(bool(np.isfinite(field).all()))
        stds.append(float(field.std()))
        residuals.append(float(case["simulation"]["max_relative_linear_residual"]))
        node_counts.append(len(payload["coordinates"]))
        if tuple(field.shape) != (4, 14, len(payload["coordinates"])):
            raise RuntimeError(f"unexpected tensor shape in {case['file']}: {field.shape}")
    result = {
        "n_cases": len(manifest["cases"]),
        "n_geometries": len({case["geometry"]["name"] for case in manifest["cases"]}),
        "resolutions": sorted({case["resolution"] for case in manifest["cases"]}),
        "tensor_semantics": manifest["tensor_semantics"],
        "all_finite": all(finite),
        "field_std_range": [min(stds), max(stds)],
        "node_count_range": [min(node_counts), max(node_counts)],
        "max_relative_linear_residual": max(residuals),
    }
    if not result["all_finite"] or min(stds) <= 1e-8 or max(residuals) > 1e-8:
        raise RuntimeError(f"elliptic audit failed: {result}")
    return result


def plot(output: Path, manifest: dict, destination: Path):
    selected = [case for case in manifest["cases"] if case["resolution"] == 32]
    fig, axes = plt.subplots(2, 3, figsize=(10, 6.4), constrained_layout=True)
    for axis, case in zip(axes.flat, selected):
        payload = np.load(output/case["file"])
        image = np.full(payload["fluid_mask"].shape, np.nan, dtype=np.float32)
        image[payload["fluid_mask"].astype(bool)] = payload["field"][2, 8]
        axis.imshow(image.T, origin="lower", cmap="viridis")
        axis.set_title(case["geometry"]["name"].replace("_", " "))
        axis.set_xticks([]); axis.set_yticks([])
    fig.suptitle("Smooth elliptic tensors on irregular physical domains")
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=180); plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("data/irregular_boundary_elliptic"))
    parser.add_argument("--summary", type=Path,
                        default=Path("papers/dataset_gates/irregular_boundary_elliptic_summary.json"))
    parser.add_argument("--figure", type=Path,
                        default=Path("papers/dataset_gates/irregular_boundary_elliptic.png"))
    args = parser.parse_args()
    manifest = generate_irregular_elliptic_dataset(args.output)
    result = audit(args.output, manifest); plot(args.output, manifest, args.figure)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
