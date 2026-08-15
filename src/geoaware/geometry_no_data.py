"""Frozen random-domain protocol for the geometry-NO tensor POC."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from scipy import ndimage
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import torch

from .geometry_no_tensor import ambient_geometry_bundle


SPLIT_COUNTS = {"train": 48, "id_validation": 8,
                "topology_ood_validation": 8, "test": 8}
SPLIT_OFFSETS = {"train": 1000, "id_validation": 2000,
                 "topology_ood_validation": 3000, "test": 4000}


def random_domain_spec(identifier: int, split: str) -> dict:
    """Create one deterministic irregular radial domain specification."""
    generator = np.random.default_rng(identifier)
    if split in {"train", "id_validation"}:
        holes = identifier % 2
    elif split == "topology_ood_validation":
        holes = 2
    elif split == "test":
        holes = 3 if identifier % 2 else 2
    else:
        raise ValueError(f"unknown split: {split}")
    spec = {
        "id": identifier,
        "split": split,
        "center": generator.uniform(-.07, .07, size=2).round(6).tolist(),
        "base_radius": float(generator.uniform(.72, .84)),
        "frequencies": [int(generator.integers(2, 5)),
                        int(generator.integers(5, 8))],
        "amplitudes": generator.uniform(.035, .105, size=2).round(6).tolist(),
        "phases": generator.uniform(0, 2*np.pi, size=2).round(6).tolist(),
        "holes": [],
    }
    base_angle = generator.uniform(0, 2*np.pi)
    for index in range(holes):
        angle = base_angle + index * np.pi + generator.uniform(-.25, .25)
        radius = generator.uniform(.17, .31)
        center = [radius*np.cos(angle), radius*np.sin(angle)]
        spec["holes"].append({
            "center": np.asarray(center).round(6).tolist(),
            "radii": generator.uniform(.075, .135, size=2).round(6).tolist(),
            "angle": float(generator.uniform(0, np.pi)),
        })
    return spec


def domain_mask(spec: dict, resolution: int) -> np.ndarray:
    axis = np.linspace(-1., 1., resolution)
    xx, yy = np.meshgrid(axis, axis, indexing="ij")
    cx, cy = spec["center"]
    dx, dy = xx-cx, yy-cy
    theta = np.arctan2(dy, dx)
    radius = np.sqrt(dx**2 + dy**2)
    limit = np.full_like(radius, spec["base_radius"])
    for frequency, amplitude, phase in zip(
            spec["frequencies"], spec["amplitudes"], spec["phases"]):
        limit += amplitude*np.cos(frequency*theta + phase)
    mask = radius <= limit
    for hole in spec["holes"]:
        hx, hy = hole["center"]
        rx, ry = hole["radii"]
        angle = hole["angle"]
        cosine, sine = np.cos(angle), np.sin(angle)
        rotated_x = cosine*(xx-hx) + sine*(yy-hy)
        rotated_y = -sine*(xx-hx) + cosine*(yy-hy)
        mask &= (rotated_x/rx)**2 + (rotated_y/ry)**2 >= 1.
    _, components = ndimage.label(mask)
    if components != 1:
        raise RuntimeError(f"generated domain {spec['id']} is disconnected")
    return mask


def solve_screened_fields(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Numerically solve a source/parameter family on one irregular domain."""
    resolution = len(mask)
    axis = np.linspace(-1., 1., resolution)
    xx, yy = np.meshgrid(axis, axis, indexing="ij")
    active = np.argwhere(mask)
    coordinates = np.stack([xx[mask], yy[mask]], 1)
    node_of = np.full(mask.shape, -1, dtype=np.int64)
    node_of[mask] = np.arange(len(active))
    rows, cols = [], []
    for node, (i, j) in enumerate(active):
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ni, nj = i+di, j+dj
            if 0 <= ni < resolution and 0 <= nj < resolution and mask[ni, nj]:
                rows.append(node)
                cols.append(int(node_of[ni, nj]))
    spacing = 2. / (resolution-1)
    adjacency = sp.coo_matrix((np.ones(len(rows)), (rows, cols)),
                              shape=(len(active), len(active))).tocsr()
    laplacian = (sp.diags(np.asarray(adjacency.sum(1)).ravel()) - adjacency) / spacing**2
    geometry = ambient_geometry_bundle(mask)
    signed = geometry[2, mask]
    reaction = .24 + .08*(1 + np.sin(2.3*coordinates[:, 0])
                          * np.cos(1.7*coordinates[:, 1]))
    reaction += .035*np.exp(-signed/.13)
    source_anchors = np.asarray([[-.52, -.32], [-.46, .38], [.46, -.08]])
    source_nodes = [int(np.argmin(((coordinates-anchor)**2).sum(1)))
                    for anchor in source_anchors]
    source_xy = coordinates[source_nodes]
    parameters = np.geomspace(.018, .24, 6)
    fields = np.empty((len(source_xy), len(parameters), len(active)), np.float32)
    for parameter_index, diffusivity in enumerate(parameters):
        matrix = sp.diags(reaction) + diffusivity*laplacian
        factor = spla.factorized(matrix.tocsc())
        phase = 2*np.pi*parameter_index/(len(parameters)-1)
        for source_index, source in enumerate(source_xy):
            distance2 = ((coordinates-source)**2).sum(1)
            forcing = np.exp(-distance2/(2*.105**2))
            forcing += .065*np.exp(-signed/.095)*np.cos(
                (source_index+2)*np.arctan2(coordinates[:, 1],
                                            coordinates[:, 0]) + phase)
            fields[source_index, parameter_index] = factor(forcing).astype(np.float32)
    return fields, source_xy.astype(np.float32), parameters.astype(np.float32)


def case_from_spec(spec: dict, resolution: int = 28) -> dict:
    mask = domain_mask(spec, resolution)
    geometry = ambient_geometry_bundle(mask)
    fields, source_xy, parameters = solve_screened_fields(mask)
    active = np.argwhere(mask).astype(np.int64)
    coordinates = geometry[5:7, mask].T.astype(np.float32)
    boundary = geometry[2, mask] <= (3.0 * 2/(resolution-1))
    return {
        "name": f"domain_{spec['id']}",
        "spec": spec,
        "geometry": torch.from_numpy(geometry),
        "mask": torch.from_numpy(mask),
        "active_indices": torch.from_numpy(active),
        "coordinates": torch.from_numpy(coordinates),
        "active_sdf": torch.from_numpy(geometry[2, mask].astype(np.float32)),
        "boundary": torch.from_numpy(boundary),
        "source_xy": torch.from_numpy(source_xy),
        "parameters": torch.from_numpy(parameters),
        "target": torch.from_numpy(fields),
    }


def protocol_manifest(resolution: int = 28) -> dict:
    splits = {}
    for split, count in SPLIT_COUNTS.items():
        offset = SPLIT_OFFSETS[split]
        splits[split] = [random_domain_spec(offset+index, split)
                         for index in range(count)]
    canonical = json.dumps(splits, sort_keys=True).encode()
    return {
        "schema_version": 1,
        "protocol": "TRACK4-GEOMETRY-NO-R1",
        "resolution": resolution,
        "observation_ratio": .01,
        "train_topology": "0/1 holes",
        "topology_ood_validation": "exactly 2 holes",
        "test_policy": "specifications frozen; fields are not materialized or read in R1",
        "spec_sha256": hashlib.sha256(canonical).hexdigest(),
        "splits": splits,
    }


def write_protocol_manifest(path: Path, resolution: int = 28) -> dict:
    manifest = protocol_manifest(resolution)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
