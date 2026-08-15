"""Geometry-neural-operator tensor models for label-scarce PDE surrogates.

The neural operator sees an ambient-grid description of the *domain*, not the
solution.  Its output is a geometry-conditioned spatial basis.  A CP head then
couples that basis to source and physical-parameter factors.  Consequently the
full geometry is available even when only a tiny subset of output labels is
observed.
"""

from __future__ import annotations

import math

import numpy as np
from scipy import ndimage
import torch
from torch import nn


def ambient_geometry_bundle(mask: np.ndarray) -> np.ndarray:
    """Return occupancy, smooth occupancy, true signed distance, normals, x/y.

    The signed distance is positive inside the physical domain and negative
    outside.  It is defined on the complete ambient grid, including holes.
    """
    if mask.ndim != 2 or mask.dtype != np.bool_:
        raise ValueError("mask must be a two-dimensional boolean array")
    h, w = mask.shape
    spacing = 2.0 / max(h - 1, w - 1)
    inside = ndimage.distance_transform_edt(mask) * spacing
    outside = ndimage.distance_transform_edt(~mask) * spacing
    signed = inside - outside
    smooth = ndimage.gaussian_filter(mask.astype(np.float32), sigma=1.25)
    normal_x, normal_y = np.gradient(signed, spacing, spacing)
    norm = np.sqrt(normal_x**2 + normal_y**2) + 1e-8
    normal_x, normal_y = normal_x / norm, normal_y / norm
    axis_x = np.linspace(-1.0, 1.0, h, dtype=np.float32)
    axis_y = np.linspace(-1.0, 1.0, w, dtype=np.float32)
    xx, yy = np.meshgrid(axis_x, axis_y, indexing="ij")
    return np.stack([
        mask.astype(np.float32), smooth.astype(np.float32),
        signed.astype(np.float32), normal_x.astype(np.float32),
        normal_y.astype(np.float32), xx, yy,
    ])


def boundary_token_bundle(mask: np.ndarray, geometry: np.ndarray | None = None,
                          max_tokens: int = 192) -> np.ndarray:
    """Sample a deterministic, typed quadrature cloud on every boundary.

    Each row contains ``(x, y, nx, ny, component_type, curvature, weight)``.
    ``component_type`` is -1 on the exterior boundary and +1 on hole
    boundaries.  The final entry is a per-component normalized quadrature
    weight, so adding a hole does not silently dilute the exterior integral.
    This routine uses only the domain mask and never reads solution values.
    """
    if mask.ndim != 2 or mask.dtype != np.bool_:
        raise ValueError("mask must be a two-dimensional boolean array")
    if max_tokens < 8:
        raise ValueError("max_tokens must be at least 8")
    geometry = ambient_geometry_bundle(mask) if geometry is None else geometry
    if geometry.shape[1:] != mask.shape:
        raise ValueError("geometry and mask resolutions do not match")

    outside_labels, _ = ndimage.label(~mask)
    exterior_label = int(outside_labels[0, 0])
    interface = mask & ~ndimage.binary_erosion(mask)
    coordinates = geometry[5:7]
    normal_x, normal_y = geometry[3], geometry[4]
    spacing = 2.0 / max(mask.shape[0]-1, mask.shape[1]-1)
    curvature = (np.gradient(normal_x, spacing, axis=0)
                 + np.gradient(normal_y, spacing, axis=1))

    # Assign each active interface pixel to the adjacent outside component.
    component = np.zeros_like(outside_labels)
    for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1),
                   (1, 1), (1, -1), (-1, 1), (-1, -1)):
        shifted = np.roll(np.roll(outside_labels, di, axis=0), dj, axis=1)
        choose = interface & (component == 0) & (shifted > 0)
        component[choose] = shifted[choose]
    labels = sorted(int(value) for value in np.unique(component[interface])
                    if value > 0)
    if not labels:
        raise ValueError("mask has no detectable boundary")

    # Give every component a minimum representation, then distribute the rest
    # approximately in proportion to its pixel perimeter.
    groups = [np.argwhere(interface & (component == label)) for label in labels]
    minimum = min(12, max_tokens // len(groups))
    remaining = max_tokens - minimum*len(groups)
    total = sum(len(group) for group in groups)
    budgets = [minimum + int(remaining*len(group)/total) for group in groups]
    for index in range(max_tokens-sum(budgets)):
        budgets[index % len(budgets)] += 1

    rows = []
    for label, pixels, budget in zip(labels, groups, budgets):
        if len(pixels) > budget:
            # Deterministic uniform sampling after ordering by polar angle
            # around this component.  This avoids a raster-scan directional
            # bias while keeping protocol repeats bitwise stable.
            center = pixels.mean(0)
            angle = np.arctan2(pixels[:, 1]-center[1], pixels[:, 0]-center[0])
            ordered = pixels[np.argsort(angle)]
            selection = np.linspace(0, len(ordered), budget,
                                    endpoint=False).astype(int)
            pixels = ordered[selection]
        kind = -1.0 if label == exterior_label else 1.0
        weight = 1.0/max(1, len(pixels))
        for i, j in pixels:
            rows.append([coordinates[0, i, j], coordinates[1, i, j],
                         normal_x[i, j], normal_y[i, j], kind,
                         np.clip(curvature[i, j]*spacing, -2., 2.), weight])
    return np.asarray(rows, dtype=np.float32)


def handcrafted_geometry_descriptor(mask: np.ndarray,
                                    geometry: np.ndarray) -> np.ndarray:
    """Seven low-order domain statistics for a method-matched gate baseline."""
    active_distance = geometry[2, mask]
    _, outside_components = ndimage.label(~mask)
    # The exterior is one outside component; all remaining components are holes.
    holes = max(0, outside_components-1)
    return np.asarray([
        mask.mean(), active_distance.mean(), active_distance.std(),
        *np.quantile(active_distance, [.25, .5, .75]), holes/3.,
    ], dtype=np.float32)


class SpectralConv2d(nn.Module):
    """A compact FNO spectral convolution with resolution-safe mode clipping."""

    def __init__(self, channels: int, modes: int):
        super().__init__()
        self.modes = int(modes)
        scale = 1.0 / math.sqrt(channels)
        self.weight = nn.Parameter(
            scale * torch.randn(channels, channels, self.modes, self.modes,
                                dtype=torch.cfloat))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = inputs.shape
        transformed = torch.fft.rfft2(inputs, norm="ortho")
        output = torch.zeros(batch, channels, height, width // 2 + 1,
                             dtype=torch.cfloat, device=inputs.device)
        mx = min(self.modes, height)
        my = min(self.modes, width // 2 + 1)
        output[:, :, :mx, :my] = torch.einsum(
            "bcxy,coxy->boxy", transformed[:, :, :mx, :my],
            self.weight[:, :, :mx, :my])
        return torch.fft.irfft2(output, s=(height, width), norm="ortho")


class GeometryFNOEncoder(nn.Module):
    """FNO-style encoder mapping domain metadata to spatial basis channels."""

    def __init__(self, output_channels: int, width: int = 24, modes: int = 8,
                 layers: int = 3, masked: bool = True,
                 geometry_inputs: str = "full"):
        super().__init__()
        if geometry_inputs not in {"full", "sdf_only"}:
            raise ValueError("geometry_inputs must be 'full' or 'sdf_only'")
        self.masked = bool(masked)
        self.geometry_inputs = geometry_inputs
        self.lift = nn.Conv2d(7, width, 1)
        self.spectral = nn.ModuleList(
            [SpectralConv2d(width, modes) for _ in range(layers)])
        self.local = nn.ModuleList([nn.Conv2d(width, width, 1)
                                    for _ in range(layers)])
        self.norms = nn.ModuleList([nn.GroupNorm(4, width)
                                    for _ in range(layers)])
        self.project = nn.Sequential(nn.Conv2d(width, width, 1), nn.GELU(),
                                     nn.Conv2d(width, output_channels, 1))

    def forward(self, geometry: torch.Tensor) -> torch.Tensor:
        if geometry.ndim == 3:
            geometry = geometry[None]
        occupancy = geometry[:, :1]
        if self.geometry_inputs == "sdf_only":
            selected = torch.zeros_like(geometry)
            selected[:, 2:3] = geometry[:, 2:3]
            geometry = selected
        hidden = self.lift(geometry)
        if self.masked:
            hidden = hidden * occupancy
        for spectral, local, norm in zip(self.spectral, self.local, self.norms):
            hidden = torch.nn.functional.gelu(norm(spectral(hidden) + local(hidden)))
            if self.masked:
                hidden = hidden * occupancy
        output = self.project(hidden)
        return output * occupancy if self.masked else output


def _factor_mlp(input_dim: int, rank: int, hidden: int = 32) -> nn.Sequential:
    return nn.Sequential(nn.Linear(input_dim, hidden), nn.GELU(),
                         nn.Linear(hidden, rank))


def _query_geometry_table(table: torch.Tensor, case: dict,
                          indices: torch.Tensor) -> torch.Tensor:
    node = indices[:, 2]
    grid = case["active_indices"].to(indices.device)[node]
    return table[0, :, grid[:, 0], grid[:, 1]].T


class GeometryNOFunctionalCP(nn.Module):
    """CP head over source, parameter, and an FNO-produced geometry basis."""

    def __init__(self, rank: int = 20, width: int = 24, modes: int = 8,
                 masked: bool = True, geometry_inputs: str = "full"):
        super().__init__()
        self.encoder = GeometryFNOEncoder(rank, width, modes, masked=masked,
                                          geometry_inputs=geometry_inputs)
        self.source_factor = _factor_mlp(2, rank)
        self.parameter_factor = _factor_mlp(2, rank)
        self.weight = nn.Parameter(torch.ones(rank) / math.sqrt(rank))

    def forward_case(self, case: dict, indices: torch.Tensor) -> torch.Tensor:
        source, parameter, _ = indices.T
        table = self.encoder(case["geometry"].to(indices.device))
        spatial = _query_geometry_table(table, case, indices)
        source_xy = case["source_xy"].to(indices.device)[source]
        value = case["parameters"].to(indices.device)[parameter]
        parameter_features = torch.stack([value, torch.log(value)], 1)
        return (self.source_factor(source_xy)
                * self.parameter_factor(parameter_features)
                * spatial * self.weight).sum(1)


class GeometryNODenseHead(nn.Module):
    """Same geometry encoder with a non-factorized pointwise regression head."""

    def __init__(self, latent: int = 20, width: int = 24, modes: int = 8,
                 masked: bool = True, geometry_inputs: str = "full"):
        super().__init__()
        self.encoder = GeometryFNOEncoder(latent, width, modes, masked=masked,
                                          geometry_inputs=geometry_inputs)
        self.head = nn.Sequential(nn.Linear(latent + 6, 48), nn.GELU(),
                                  nn.Linear(48, 48), nn.GELU(),
                                  nn.Linear(48, 1))

    def forward_case(self, case: dict, indices: torch.Tensor) -> torch.Tensor:
        source, parameter, node = indices.T
        table = self.encoder(case["geometry"].to(indices.device))
        latent = _query_geometry_table(table, case, indices)
        source_xy = case["source_xy"].to(indices.device)[source]
        value = case["parameters"].to(indices.device)[parameter]
        parameter_features = torch.stack([value, torch.log(value)], 1)
        xy = case["coordinates"].to(indices.device)[node]
        return self.head(torch.cat([latent, source_xy, parameter_features, xy], 1)).squeeze(1)


class CoordinateSDFFunctionalCP(nn.Module):
    """No-operator CP baseline using only pointwise coordinates and SDF."""

    def __init__(self, rank: int = 20, hidden: int = 48,
                 use_sdf: bool = True):
        super().__init__()
        self.use_sdf = bool(use_sdf)
        self.source_factor = _factor_mlp(2, rank, hidden)
        self.parameter_factor = _factor_mlp(2, rank, hidden)
        self.space_factor = _factor_mlp(6, rank, hidden)
        self.weight = nn.Parameter(torch.ones(rank) / math.sqrt(rank))

    def forward_case(self, case: dict, indices: torch.Tensor) -> torch.Tensor:
        source, parameter, node = indices.T
        source_xy = case["source_xy"].to(indices.device)[source]
        value = case["parameters"].to(indices.device)[parameter]
        parameter_features = torch.stack([value, torch.log(value)], 1)
        xy = case["coordinates"].to(indices.device)[node]
        sdf = case["active_sdf"].to(indices.device)[node, None]
        if not self.use_sdf:
            sdf = torch.zeros_like(sdf)
        distance = torch.linalg.vector_norm(xy - source_xy, dim=1, keepdim=True)
        spatial = torch.cat([xy, sdf, source_xy, distance], 1)
        return (self.source_factor(source_xy)
                * self.parameter_factor(parameter_features)
                * self.space_factor(spatial) * self.weight).sum(1)


class BoundaryOperatorFunctionalCP(nn.Module):
    """Low-capacity boundary-operator-conditioned functional CP.

    A shared query-to-boundary kernel constructs one correction per CP rank.
    This is a tensor bottleneck, not a general neural operator: the boundary
    branch can affect the field only through the rank-wise spatial factors
    shared across source and physical-parameter modes.

    ``operator`` selects the method and two causal controls. ``integral`` uses
    query-boundary displacement, normal alignment and boundary type;
    ``pooled`` removes query-boundary interaction while retaining a DeepSets
    summary; ``integral_outer_only`` drops all hole tokens.
    """

    def __init__(self, rank: int = 20, hidden: int = 48,
                 operator: str = "integral", initial_gate: float = .05,
                 use_sdf: bool = True):
        super().__init__()
        if operator not in {"integral", "pooled", "integral_outer_only",
                            "integral_wrong_type"}:
            raise ValueError(f"unknown boundary operator: {operator}")
        self.operator = operator
        self.use_sdf = bool(use_sdf)
        self.source_factor = _factor_mlp(2, rank, hidden)
        self.parameter_factor = _factor_mlp(2, rank, hidden)
        self.local_factor = _factor_mlp(6, rank, hidden)
        self.token_value = nn.Sequential(
            nn.Linear(6, hidden//2), nn.GELU(), nn.Linear(hidden//2, rank))
        # A compact learned Green-like radial kernel.  Positive length scales
        # make the inductive bias explicit and stable under sparse supervision.
        self.raw_lengthscale = nn.Parameter(
            torch.linspace(-2.5, -.2, rank))
        self.normal_gate = nn.Parameter(torch.zeros(rank))
        self.type_gate = nn.Parameter(torch.zeros(rank))
        self.integral_mix = nn.Sequential(
            nn.Linear(4, hidden//2), nn.GELU(), nn.Linear(hidden//2, rank))
        self.weight = nn.Parameter(torch.ones(rank) / math.sqrt(rank))
        self.operator_gate = nn.Parameter(torch.tensor(float(initial_gate)))

    def _integral_basis(self, xy: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        if self.operator == "integral_outer_only":
            tokens = tokens[tokens[:, 4] < 0]
        if self.operator == "integral_wrong_type":
            tokens = tokens.clone()
            tokens[:, 4] = -tokens[:, 4]
        token_features = tokens[:, :6]
        token_value = self.token_value(token_features)
        weights = tokens[:, 6]
        if self.operator == "pooled":
            pooled = (token_value * weights[:, None]).sum(0)
            return pooled[None].expand(len(xy), -1)

        delta = xy[:, None, :] - tokens[None, :, :2]
        distance = torch.linalg.vector_norm(delta, dim=2).clamp_min(1e-5)
        direction = delta/distance[:, :, None]
        alignment = (direction*tokens[None, :, 2:4]).sum(2)
        pair = torch.stack([distance, alignment, tokens[None, :, 4].expand_as(distance),
                            tokens[None, :, 5].expand_as(distance)], 2)
        learned_pair = self.integral_mix(pair)
        lengthscale = torch.nn.functional.softplus(self.raw_lengthscale) + .03
        radial = torch.exp(-distance[:, :, None]/lengthscale)
        orientation = (1. + torch.tanh(self.normal_gate)[None, None]*alignment[:, :, None])
        boundary_type = (1. + torch.tanh(self.type_gate)[None, None]
                         * tokens[None, :, 4:5])
        kernel = radial*orientation*boundary_type
        return ((token_value[None] + learned_pair)*kernel
                * weights[None, :, None]).sum(1)

    def forward_case(self, case: dict, indices: torch.Tensor) -> torch.Tensor:
        source, parameter, node = indices.T
        device = indices.device
        source_xy = case["source_xy"].to(device)[source]
        value = case["parameters"].to(device)[parameter]
        parameter_features = torch.stack([value, torch.log(value)], 1)
        xy = case["coordinates"].to(device)[node]
        sdf = case["active_sdf"].to(device)[node, None]
        if not self.use_sdf:
            sdf = torch.zeros_like(sdf)
        distance = torch.linalg.vector_norm(xy-source_xy, dim=1, keepdim=True)
        local = self.local_factor(torch.cat([xy, sdf, source_xy, distance], 1))

        unique_node, inverse = torch.unique(node, sorted=False,
                                            return_inverse=True)
        unique_xy = case["coordinates"].to(device)[unique_node]
        correction = self._integral_basis(
            unique_xy, case["boundary_tokens"].to(device))[inverse]
        spatial = local + self.operator_gate*correction
        return (self.source_factor(source_xy)
                * self.parameter_factor(parameter_features)
                * spatial*self.weight).sum(1)


class RankModulatedCoordinateCP(nn.Module):
    """Geometry-coordinate CP with one small domain-level rank gate.

    ``descriptor`` uses seven handcrafted statistics. ``boundary`` uses a
    permutation-invariant DeepSets embedding of typed boundary tokens.
    ``wrong_boundary`` has identical parameters but reads a cyclically
    mismatched boundary set supplied by the experiment protocol.
    """

    def __init__(self, rank: int = 20, hidden: int = 48,
                 conditioning: str = "boundary", set_width: int = 16,
                 initial_gate: float = .05):
        super().__init__()
        if conditioning not in {"descriptor", "boundary", "wrong_boundary"}:
            raise ValueError(f"unknown rank conditioning: {conditioning}")
        self.conditioning = conditioning
        self.source_factor = _factor_mlp(2, rank, hidden)
        self.parameter_factor = _factor_mlp(2, rank, hidden)
        self.space_factor = _factor_mlp(6, rank, hidden)
        if conditioning == "descriptor":
            self.gate_network = nn.Sequential(
                nn.Linear(7, set_width), nn.GELU(), nn.Linear(set_width, rank))
        else:
            self.token_encoder = nn.Sequential(
                nn.Linear(6, set_width), nn.GELU(),
                nn.Linear(set_width, set_width), nn.GELU())
            self.gate_network = nn.Linear(2*set_width, rank)
        self.modulation_scale = nn.Parameter(torch.tensor(float(initial_gate)))
        self.weight = nn.Parameter(torch.ones(rank) / math.sqrt(rank))

    def domain_gate(self, case: dict, device: torch.device) -> torch.Tensor:
        if self.conditioning == "descriptor":
            inputs = case["geometry_descriptor"].to(device)
        else:
            key = ("wrong_boundary_tokens" if self.conditioning == "wrong_boundary"
                   else "boundary_tokens")
            tokens = case[key].to(device)
            embedding = self.token_encoder(tokens[:, :6])
            weighted_sum = (embedding*tokens[:, 6:7]).sum(0)
            maximum = embedding.max(0).values
            inputs = torch.cat([weighted_sum, maximum])
        return 1. + self.modulation_scale*torch.tanh(self.gate_network(inputs))

    def forward_case(self, case: dict, indices: torch.Tensor) -> torch.Tensor:
        source, parameter, node = indices.T
        device = indices.device
        source_xy = case["source_xy"].to(device)[source]
        value = case["parameters"].to(device)[parameter]
        parameter_features = torch.stack([value, torch.log(value)], 1)
        xy = case["coordinates"].to(device)[node]
        sdf = case["active_sdf"].to(device)[node, None]
        distance = torch.linalg.vector_norm(xy-source_xy, dim=1, keepdim=True)
        spatial = self.space_factor(torch.cat([xy, sdf, source_xy, distance], 1))
        gate = self.domain_gate(case, device)
        return (self.source_factor(source_xy)
                * self.parameter_factor(parameter_features)
                * spatial*gate[None]*self.weight).sum(1)


class CoordinateSDFPlusGeometryNOCP(nn.Module):
    """Strong local CP mean plus a small geometry-NO CP residual.

    The local mean is exactly ``CoordinateSDFFunctionalCP``.  The residual gate
    starts near zero so adding the operator branch does not destroy the useful
    low-capacity inductive bias at the start of sparse-label optimization.
    """

    def __init__(self, rank: int = 20, hidden: int = 48, width: int = 24,
                 modes: int = 8, masked: bool = True,
                 initial_residual_gate: float = .01):
        super().__init__()
        self.mean = CoordinateSDFFunctionalCP(rank=rank, hidden=hidden)
        self.residual = GeometryNOFunctionalCP(
            rank=rank, width=width, modes=modes, masked=masked,
            geometry_inputs="full")
        self.residual_gate = nn.Parameter(torch.tensor(float(initial_residual_gate)))

    def forward_case(self, case: dict, indices: torch.Tensor) -> torch.Tensor:
        return (self.mean.forward_case(case, indices)
                + self.residual_gate * self.residual.forward_case(case, indices))
