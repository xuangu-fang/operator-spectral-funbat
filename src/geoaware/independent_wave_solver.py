"""Independent finite-difference wave dataset generator.

This module intentionally does not import any learner, tensor factorization, or
feature-construction code.  It produces physical fields and geometry/operator
metadata on disk; downstream models only consume the resulting files.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


@dataclass(frozen=True)
class WaveGeometrySpec:
    name: str
    kind: str
    parameters: dict[str, float]


@dataclass
class WaveDomain:
    spec: WaveGeometrySpec
    resolution: int
    spacing: float
    coordinates: np.ndarray
    grid_indices: np.ndarray
    fluid_mask: np.ndarray
    signed_distance: np.ndarray
    material_speed: np.ndarray
    geometry_operator: sp.csr_matrix
    wave_operator: sp.csr_matrix
    undirected_edges: np.ndarray


def default_geometry_specs() -> list[WaveGeometrySpec]:
    """Eight topology/shape variants for the dataset gate."""
    return [
        WaveGeometrySpec("wall_left_door_low", "wall",
                         {"cx": -0.10, "width": .10, "door_y": -.28, "gap": .34}),
        WaveGeometrySpec("wall_center_door_mid", "wall",
                         {"cx": .02, "width": .12, "door_y": .02, "gap": .28}),
        WaveGeometrySpec("wall_right_door_high", "wall",
                         {"cx": .16, "width": .14, "door_y": .31, "gap": .38}),
        WaveGeometrySpec("circle_center", "circle",
                         {"cx": .05, "cy": .02, "radius": .23}),
        WaveGeometrySpec("circle_offset", "circle",
                         {"cx": .23, "cy": -.22, "radius": .18}),
        WaveGeometrySpec("ellipse_tilted", "ellipse",
                         {"cx": .08, "cy": .10, "rx": .31, "ry": .15, "angle": .58}),
        WaveGeometrySpec("double_vertical", "double",
                         {"cx1": .02, "cy1": -.30, "r1": .15,
                          "cx2": .10, "cy2": .29, "r2": .17}),
        WaveGeometrySpec("double_diagonal", "double",
                         {"cx1": -.02, "cy1": -.12, "r1": .17,
                          "cx2": .31, "cy2": .25, "r2": .14}),
    ]


def _rectangle_sdf(x: np.ndarray, y: np.ndarray, cx: float, cy: float,
                   half_width: float, half_height: float) -> np.ndarray:
    qx = np.abs(x - cx) - half_width
    qy = np.abs(y - cy) - half_height
    outside = np.sqrt(np.maximum(qx, 0) ** 2 + np.maximum(qy, 0) ** 2)
    return outside + np.minimum(np.maximum(qx, qy), 0)


def obstacle_signed_distance(x: np.ndarray, y: np.ndarray,
                             spec: WaveGeometrySpec) -> np.ndarray:
    """Positive outside the solid obstacle and negative inside."""
    p = spec.parameters
    if spec.kind == "circle":
        return np.sqrt((x-p["cx"])**2 + (y-p["cy"])**2) - p["radius"]
    if spec.kind == "ellipse":
        ca, sa = math.cos(p["angle"]), math.sin(p["angle"])
        xx = ca*(x-p["cx"]) + sa*(y-p["cy"])
        yy = -sa*(x-p["cx"]) + ca*(y-p["cy"])
        return (np.sqrt((xx/p["rx"])**2 + (yy/p["ry"])**2)-1)*min(p["rx"], p["ry"])
    if spec.kind == "double":
        d1 = np.sqrt((x-p["cx1"])**2 + (y-p["cy1"])**2) - p["r1"]
        d2 = np.sqrt((x-p["cx2"])**2 + (y-p["cy2"])**2) - p["r2"]
        return np.minimum(d1, d2)
    if spec.kind == "wall":
        lower_hi = p["door_y"] - p["gap"]/2
        upper_lo = p["door_y"] + p["gap"]/2
        low = _rectangle_sdf(x, y, p["cx"], (-1+lower_hi)/2,
                             p["width"]/2, (lower_hi+1)/2)
        high = _rectangle_sdf(x, y, p["cx"], (upper_lo+1)/2,
                              p["width"]/2, (1-upper_lo)/2)
        return np.minimum(low, high)
    raise ValueError(f"unknown geometry kind: {spec.kind}")


def _laplacian(n_nodes: int, edges: np.ndarray, weights: np.ndarray,
               spacing: float) -> sp.csr_matrix:
    a, b = edges[:, 0], edges[:, 1]
    rows = np.concatenate([a, b, a, b])
    cols = np.concatenate([a, b, b, a])
    vals = np.concatenate([weights, weights, -weights, -weights]) / spacing**2
    return sp.coo_matrix((vals, (rows, cols)), shape=(n_nodes, n_nodes)).tocsr()


def build_wave_domain(spec: WaveGeometrySpec, resolution: int) -> WaveDomain:
    if resolution < 10:
        raise ValueError("resolution must be at least 10")
    axis = np.linspace(-1., 1., resolution, dtype=np.float64)
    xx, yy = np.meshgrid(axis, axis, indexing="ij")
    sdf = obstacle_signed_distance(xx, yy, spec)
    fluid = sdf >= 0
    grid_indices = np.argwhere(fluid)
    node_of = np.full((resolution, resolution), -1, dtype=np.int64)
    node_of[fluid] = np.arange(len(grid_indices))
    edges = []
    for node, (i, j) in enumerate(grid_indices):
        for di, dj in ((1, 0), (0, 1)):
            ni, nj = i+di, j+dj
            if ni < resolution and nj < resolution and fluid[ni, nj]:
                edges.append((node, int(node_of[ni, nj])))
    edges = np.asarray(edges, dtype=np.int64)
    coords = np.stack([xx[fluid], yy[fluid]], axis=1)
    # A smooth, geometry-independent material law.  Geometry and material are
    # stored separately even though both influence the wave operator.
    speed = (.82 + .10*np.sin(2.1*coords[:, 0])*np.cos(1.7*coords[:, 1])
             + .07*np.tanh(5*sdf[fluid])).clip(.55, 1.05)
    geometry_operator = _laplacian(len(coords), edges, np.ones(len(edges)), axis[1]-axis[0])
    edge_speed = .5*(speed[edges[:, 0]] + speed[edges[:, 1]])
    wave_operator = _laplacian(len(coords), edges, edge_speed**2, axis[1]-axis[0])
    return WaveDomain(
        spec=spec, resolution=resolution, spacing=float(axis[1]-axis[0]),
        coordinates=coords.astype(np.float32), grid_indices=grid_indices.astype(np.int32),
        fluid_mask=fluid, signed_distance=sdf[fluid].astype(np.float32),
        material_speed=speed.astype(np.float32), geometry_operator=geometry_operator,
        wave_operator=wave_operator, undirected_edges=edges.astype(np.int32))


def simulate_damped_wave(domain: WaveDomain, source_xy: tuple[float, float],
                         record_times: np.ndarray, damping: float = .16,
                         central_frequency: float = 5., source_width: float = .075
                         ) -> tuple[np.ndarray, dict]:
    """Solve a reflected variable-coefficient wave equation by leapfrog.

    The semi-discrete equation is ``u_tt + damping*u_t + L_c*u = f(t)``.
    Missing graph edges at solid and outer boundaries implement zero-flux
    reflection.  Time step size is chosen from the largest eigenvalue rather
    than shared with any learner.
    """
    times = np.asarray(record_times, dtype=np.float64)
    if len(times) < 2 or np.any(np.diff(times) <= 0) or times[0] < 0:
        raise ValueError("record_times must be increasing and nonnegative")
    coords = domain.coordinates.astype(np.float64)
    src = np.asarray(source_xy, dtype=np.float64)
    source_node = int(np.argmin(np.sum((coords-src)**2, axis=1)))
    profile = np.exp(-np.sum((coords-coords[source_node])**2, axis=1)/(2*source_width**2))
    profile /= np.sqrt(np.mean(profile**2)) + 1e-12
    largest = float(spla.eigsh(domain.wave_operator, k=1, which="LM",
                               return_eigenvectors=False, tol=1e-4)[0])
    stable_dt = 1.75/math.sqrt(max(largest, 1e-8))
    dt = min(stable_dt, float(np.min(np.diff(times)))/4)
    n_steps = int(math.ceil(times[-1]/dt)) + 1
    dt = times[-1]/max(1, n_steps-1)
    previous = np.zeros(len(coords), dtype=np.float64)
    current = np.zeros_like(previous)
    fields = np.zeros((len(times), len(coords)), dtype=np.float32)
    next_record = 0
    pulse_center = .13
    for step in range(n_steps):
        time = step*dt
        while next_record < len(times) and time+dt/2 >= times[next_record]:
            fields[next_record] = current
            next_record += 1
        tau = math.pi*central_frequency*(time-pulse_center)
        ricker = (1-2*tau*tau)*math.exp(-tau*tau)
        forcing = 55.*ricker*profile
        numerator = (2*current - (1-damping*dt/2)*previous
                     - dt*dt*(domain.wave_operator@current) + dt*dt*forcing)
        following = numerator/(1+damping*dt/2)
        previous, current = current, following
    while next_record < len(times):
        fields[next_record] = current
        next_record += 1
    metadata = {
        "source_xy_requested": [float(source_xy[0]), float(source_xy[1])],
        "source_xy_discrete": coords[source_node].tolist(),
        "source_node": source_node,
        "damping": damping,
        "central_frequency": central_frequency,
        "source_width": source_width,
        "largest_wave_operator_eigenvalue": largest,
        "time_step": dt,
        "n_internal_steps": n_steps,
        "boundary_condition": "reflecting zero-flux on outer and obstacle boundaries",
    }
    return fields, metadata


def _sparse_payload(prefix: str, matrix: sp.csr_matrix) -> dict[str, np.ndarray]:
    return {
        f"{prefix}_data": matrix.data.astype(np.float32),
        f"{prefix}_indices": matrix.indices.astype(np.int32),
        f"{prefix}_indptr": matrix.indptr.astype(np.int32),
        f"{prefix}_shape": np.asarray(matrix.shape, dtype=np.int32),
    }


def save_wave_case(path: Path, domain: WaveDomain, source_index: int,
                   source_xy: tuple[float, float], record_times: np.ndarray) -> dict:
    fields, simulation = simulate_damped_wave(domain, source_xy, record_times)
    payload = {
        "coordinates": domain.coordinates,
        "grid_indices": domain.grid_indices,
        "fluid_mask": domain.fluid_mask.astype(np.uint8),
        "signed_distance": domain.signed_distance,
        "material_speed": domain.material_speed,
        "undirected_edges": domain.undirected_edges,
        "record_times": np.asarray(record_times, dtype=np.float32),
        "field": fields,
        "source_index": np.asarray(source_index, dtype=np.int32),
        "source_xy": np.asarray(source_xy, dtype=np.float32),
        **_sparse_payload("geometry_operator", domain.geometry_operator),
        **_sparse_payload("wave_operator", domain.wave_operator),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return {
        "file": path.name,
        "sha256": digest,
        "geometry": asdict(domain.spec),
        "resolution": domain.resolution,
        "n_nodes": len(domain.coordinates),
        "n_edges": len(domain.undirected_edges),
        "source_index": source_index,
        "simulation": simulation,
        "field_std": float(fields.std()),
        "field_abs_max": float(np.abs(fields).max()),
    }


def generate_wave_dataset(output: Path, specs: Iterable[WaveGeometrySpec] | None = None,
                          resolutions: Iterable[int] = (24, 32),
                          sources: Iterable[tuple[float, float]] = ((-.72, -.38), (-.72, .38)),
                          record_times: np.ndarray | None = None) -> dict:
    specs = list(default_geometry_specs() if specs is None else specs)
    resolutions = list(resolutions)
    sources = list(sources)
    # The source-to-obstacle travel time is roughly 0.8--1.0 in these units.
    # Recording to t=2.0 ensures that the target contains transmitted and
    # reflected waves rather than only the geometry-insensitive incident pulse.
    times = np.linspace(0., 2., 40) if record_times is None else np.asarray(record_times)
    output.mkdir(parents=True, exist_ok=True)
    cases = []
    for spec in specs:
        for resolution in resolutions:
            domain = build_wave_domain(spec, resolution)
            for source_index, source in enumerate(sources):
                name = f"{spec.name}_r{resolution}_s{source_index}.npz"
                cases.append(save_wave_case(output/name, domain, source_index, source, times))
    manifest = {
        "schema_version": 1,
        "generator": "geoaware.independent_wave_solver.generate_wave_dataset",
        "equation": "u_tt + damping*u_t + L_c*u = Ricker(t)*Gaussian(x-source)",
        "tensor_semantics": ["geometry", "source", "time", "irregular spatial node"],
        "resolutions": resolutions,
        "sources": [list(x) for x in sources],
        "record_times": times.tolist(),
        "n_geometries": len(specs),
        "n_cases": len(cases),
        "cases": cases,
    }
    (output/"manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest
