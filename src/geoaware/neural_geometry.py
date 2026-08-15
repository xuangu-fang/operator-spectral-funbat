"""Graph-spectral neural fields for obstacle-domain geometry transfer.

This module is deliberately independent from the Bayesian POC.  It studies a
different question: can a neural field trained from very sparse sensors on a
family of domains transfer to a new obstacle and a new mesh?  The key object is
the sign-invariant source-to-point spectral feature

    z_k(x, s; G) = phi_k^G(x) phi_k^G(s),

which lets one learned spectral transfer function operate on many graphs.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import scipy.sparse.csgraph as csgraph
import torch
from torch import nn


@dataclass
class ObstacleDomain:
    name: str
    resolution: int
    coords: torch.Tensor
    grid_indices: torch.Tensor
    mask: torch.Tensor
    eigenvalues: torch.Tensor
    eigenvectors: torch.Tensor
    source_projection: torch.Tensor
    signed_distance: torch.Tensor
    descriptor: torch.Tensor
    edge_index: torch.Tensor | None = None
    degree: torch.Tensor | None = None

    @property
    def n_nodes(self) -> int:
        return len(self.coords)


@dataclass
class WaveTask:
    domain: ObstacleDomain
    time: float
    values: torch.Tensor
    observed: torch.Tensor
    noisy_values: torch.Tensor

    @property
    def name(self) -> str:
        return f"{self.domain.name}_t{self.time:.3f}"


def _obstacle_sdf(x: np.ndarray, y: np.ndarray, spec: dict) -> np.ndarray:
    """Positive outside an obstacle, negative inside (approximate for unions)."""
    kind = spec["kind"]
    if kind == "circle":
        cx, cy, r = spec["cx"], spec["cy"], spec["r"]
        return np.sqrt((x - cx) ** 2 + (y - cy) ** 2) - r
    if kind == "ellipse":
        cx, cy, rx, ry = spec["cx"], spec["cy"], spec["rx"], spec["ry"]
        a = spec.get("angle", 0.0)
        ca, sa = np.cos(a), np.sin(a)
        xx, yy = ca * (x - cx) + sa * (y - cy), -sa * (x - cx) + ca * (y - cy)
        q = np.sqrt((xx / rx) ** 2 + (yy / ry) ** 2)
        return (q - 1.0) * min(rx, ry)
    if kind == "double":
        d1 = np.sqrt((x - spec["cx1"]) ** 2 + (y - spec["cy1"]) ** 2) - spec["r1"]
        d2 = np.sqrt((x - spec["cx2"]) ** 2 + (y - spec["cy2"]) ** 2) - spec["r2"]
        return np.minimum(d1, d2)
    if kind == "wall":
        cx, width, door_y, gap = spec["cx"], spec["width"], spec["door_y"], spec["gap"]
        def rect_sdf(cy, hx, hy):
            qx, qy = np.abs(x - cx) - hx, np.abs(y - cy) - hy
            outside = np.sqrt(np.maximum(qx, 0) ** 2 + np.maximum(qy, 0) ** 2)
            return outside + np.minimum(np.maximum(qx, qy), 0)
        lower_hi, upper_lo = door_y - gap / 2, door_y + gap / 2
        dlow = rect_sdf((-1 + lower_hi) / 2, width / 2, (lower_hi + 1) / 2)
        dhigh = rect_sdf((upper_lo + 1) / 2, width / 2, (1 - upper_lo) / 2)
        return np.minimum(dlow, dhigh)
    raise ValueError(kind)


def _descriptor(spec: dict, fluid_fraction: float) -> np.ndarray:
    kind = spec["kind"]
    if kind == "circle":
        vals = [0.0, spec["cx"], spec["cy"], spec["r"], spec["r"], 0.0]
    elif kind == "ellipse":
        vals = [0.5, spec["cx"], spec["cy"], spec["rx"], spec["ry"], spec.get("angle", 0.0) / math.pi]
    elif kind == "double":
        vals = [1.0, (spec["cx1"] + spec["cx2"]) / 2, (spec["cy1"] + spec["cy2"]) / 2,
                spec["r1"], spec["r2"], math.hypot(spec["cx1"] - spec["cx2"], spec["cy1"] - spec["cy2"]) / 2]
    else:
        vals = [1.5, spec["cx"], spec["door_y"], spec["width"], spec["gap"], 0.0]
    return np.asarray(vals + [fluid_fraction], np.float32)


def build_obstacle_domain(spec: dict, resolution: int = 40, n_eigen: int = 96,
                          source=(-0.72, -0.38)) -> ObstacleDomain:
    """Create a 4-neighbour fluid graph and its scaled Neumann Laplacian."""
    axis = np.linspace(-1.0, 1.0, resolution, dtype=np.float64)
    xx, yy = np.meshgrid(axis, axis, indexing="ij")
    sdf = _obstacle_sdf(xx, yy, spec)
    fluid = sdf >= 0
    ij = np.argwhere(fluid)
    node_of = -np.ones((resolution, resolution), dtype=np.int64)
    node_of[fluid] = np.arange(len(ij))
    rows, cols = [], []
    for node, (i, j) in enumerate(ij):
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ni, nj = i + di, j + dj
            if 0 <= ni < resolution and 0 <= nj < resolution and fluid[ni, nj]:
                rows.append(node); cols.append(int(node_of[ni, nj]))
    adjacency = sp.coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(ij), len(ij))).tocsr()
    lap = sp.diags(np.asarray(adjacency.sum(1)).ravel()) - adjacency
    # Grid graph eigenvalues are O(h^2).  Scaling makes the spectrum converge
    # to the continuum Laplacian and is essential for cross-resolution tests.
    lap = lap * ((resolution - 1) / 2.0) ** 2
    k = min(n_eigen, len(ij) - 2)
    eigval, eigvec = spla.eigsh(lap, k=k, which="SM", tol=2e-5)
    order = np.argsort(eigval)
    eigval, eigvec = np.maximum(eigval[order], 0), eigvec[:, order]
    # Continuous L2 normalization under the empirical area measure.
    eigvec *= math.sqrt(len(ij))
    coords = np.stack([xx[fluid], yy[fluid]], 1).astype(np.float32)
    dsrc = np.sum((coords - np.asarray(source, np.float32)) ** 2, 1)
    source_profile = np.exp(-dsrc / (2 * 0.075 ** 2)).astype(np.float64)
    source_profile /= np.sqrt(np.mean(source_profile ** 2)) + 1e-12
    source_projection = eigvec.T @ source_profile / len(ij)
    desc = _descriptor(spec, float(fluid.mean()))
    return ObstacleDomain(
        name=spec.get("name", spec["kind"]), resolution=resolution,
        coords=torch.from_numpy(coords), grid_indices=torch.from_numpy(ij),
        mask=torch.from_numpy(fluid), eigenvalues=torch.from_numpy(eigval.astype(np.float32)),
        eigenvectors=torch.from_numpy(eigvec.astype(np.float32)),
        source_projection=torch.from_numpy(source_projection.astype(np.float32)),
        signed_distance=torch.from_numpy(sdf[fluid].astype(np.float32)),
        descriptor=torch.from_numpy(desc),
        edge_index=torch.from_numpy(np.stack(adjacency.nonzero()).astype(np.int64)),
        degree=torch.from_numpy(np.asarray(adjacency.sum(1)).ravel().astype(np.float32)),
    )


def wave_field(domain: ObstacleDomain, time: float, damping: float = 0.0015,
               speed: float = 1.0) -> torch.Tensor:
    """Truncated graph wave kernel applied to a smooth localized source."""
    lam = domain.eigenvalues
    transfer = torch.exp(-damping * lam) * torch.cos(speed * time * torch.sqrt(lam.clamp_min(0)))
    return domain.eigenvectors @ (domain.source_projection * transfer)


def heterogeneous_scattering_field(domain: ObstacleDomain, time: float, steps: int = 72) -> torch.Tensor:
    """Graph finite-difference wave with spatial speed and cubic mode mixing.

    Unlike ``wave_field``, this process is not diagonal in the unweighted graph
    eigenbasis exposed to the learner, so it serves as the non-isomorphic task.
    """
    src_nodes, dst_nodes = domain.edge_index
    degree = domain.degree.clamp_min(1)
    def lap_mv(u):
        neighbour_sum = torch.zeros_like(u)
        neighbour_sum.index_add_(0, src_nodes, u[dst_nodes])
        return u - neighbour_sum / degree
    xy = domain.coords
    speed2 = (0.55 + 0.35 * torch.sigmoid(8 * domain.signed_distance)
              + 0.16 * torch.sin(2.7 * xy[:, 0]) * torch.cos(2.1 * xy[:, 1])).clamp(.25, 1.2)
    src = torch.exp(-torch.sum((xy - torch.tensor([-.72, -.38])) ** 2, 1) / (2 * .075 ** 2))
    src = src / src.square().mean().sqrt().clamp_min(1e-8)
    u0, u1 = src, src.clone()
    dt = 0.32
    nsteps = max(2, int(round(steps * time / .40)))
    for _ in range(nsteps):
        accel = -speed2 * lap_mv(u1) - .012 * u1.pow(3)
        u2 = 1.992 * u1 - .992 * u0 + dt * dt * accel
        u0, u1 = u1, u2
    return u1


def elliptic_boundary_layer_field(domain: ObstacleDomain, time: float) -> torch.Tensor:
    """Independent variable-coefficient screened-Poisson boundary-layer solve."""
    src, dst = domain.edge_index.numpy()
    n = domain.n_nodes
    adjacency = sp.coo_matrix((np.ones(len(src)), (src, dst)), shape=(n, n)).tocsr()
    degree = np.asarray(adjacency.sum(1)).ravel()
    lap = sp.diags(degree) - adjacency
    xy, sdf = domain.coords.numpy(), domain.signed_distance.numpy()
    # Spatial reaction is not a function of the unweighted Laplacian, hence
    # this operator is not diagonal in the learner basis.
    reaction = 0.12 + 0.18 * (1 + np.sin(2.4 * xy[:, 0]) * np.cos(1.7 * xy[:, 1]))
    matrix = sp.diags(reaction) + (0.32 + .18 * time) * lap
    point = np.exp(-((xy[:, 0] + .72) ** 2 + (xy[:, 1] + .38) ** 2) / (2 * .055 ** 2))
    boundary = sdf < (2.5 / domain.resolution)
    angle = np.arctan2(xy[:, 1], xy[:, 0])
    is_wall = domain.descriptor[0].item() > 1.25
    boundary_drive = 0.0 if is_wall else boundary * (1.0 + .45 * np.sin((3 + round(4 * time)) * angle))
    forcing = 2.2 * point + boundary_drive
    solution = spla.spsolve(matrix.tocsc(), forcing)
    # A weak local nonlinearity produces harmonics while retaining PDE origin.
    solution = solution + .08 * np.tanh(2 * solution)
    return torch.from_numpy(solution.astype(np.float32))


def geodesic_wavepacket_field(domain: ObstacleDomain, time: float) -> torch.Tensor:
    """Eikonal/shortest-path wavepacket, independent of Laplacian eigenvectors."""
    src, dst = domain.edge_index.numpy()
    h = 2.0 / (domain.resolution - 1)
    graph = sp.coo_matrix((np.full(len(src), h), (src, dst)),
                          shape=(domain.n_nodes, domain.n_nodes)).tocsr()
    source_xy = np.asarray([-.72, -.38], np.float32)
    source_node = int(np.argmin(np.sum((domain.coords.numpy() - source_xy) ** 2, 1)))
    distance = csgraph.dijkstra(graph, directed=False, indices=source_node)
    phase_distance = distance - (1.15 + .65 * time)
    carrier = 27.0
    packet = np.exp(-(phase_distance / .22) ** 2) * np.cos(carrier * phase_distance)
    # A weaker second packet makes the target non-stationary and prevents a
    # single radial basis from being an exact simulator match.
    packet += .28 * np.exp(-((distance - (.55 + .35 * time)) / .13) ** 2) * np.sin(19 * distance)
    return torch.from_numpy(packet.astype(np.float32))


def geodesic_harmonic_field(domain: ObstacleDomain, time: float) -> torch.Tensor:
    """Rank-3 eikonal harmonics plus a weak off-model moving residual."""
    d = source_geodesic_distance(domain)
    e = domain.descriptor
    amplitudes = torch.stack([1.0 + .18 * e[1] - .10 * e[2],
                              .55 + .12 * e[4] + .08 * e[1],
                              .30 + .10 * e[6] - .05 * e[2]])
    bands = torch.tensor([7., 13., 19.]); decay = torch.tensor([.45, 1.05, 1.8])
    phase = torch.tensor([.2, -.55, .8]); tt = torch.tensor(time)
    dominant = sum(amplitudes[b] * torch.exp(-decay[b] * tt) *
                   torch.cos(bands[b] * d + phase[b]) for b in range(3))
    residual = .06 * torch.exp(-((d - (.85 + .22 * time)) / .20).square()) * \
        torch.cos(23. * (d - .22 * time))
    return dominant + residual


def geodesic_mixed_field(domain: ObstacleDomain, time: float,
                         mismatch: float) -> torch.Tensor:
    """Continuous harmonic-to-moving-envelope approximation-error axis."""
    if not 0 <= mismatch <= 1:
        raise ValueError("mismatch must lie in [0, 1]")
    harmonic = geodesic_harmonic_field(domain, time)
    packet = geodesic_wavepacket_field(domain, time)
    packet = packet * harmonic.std().clamp_min(1e-8) / packet.std().clamp_min(1e-8)
    return (1 - mismatch) * harmonic + mismatch * packet


def source_geodesic_distance(domain: ObstacleDomain) -> torch.Tensor:
    src, dst = domain.edge_index.numpy()
    h = 2.0 / (domain.resolution - 1)
    graph = sp.coo_matrix((np.full(len(src), h), (src, dst)),
                          shape=(domain.n_nodes, domain.n_nodes)).tocsr()
    source_xy = np.asarray([-.72, -.38], np.float32)
    source_node = int(np.argmin(np.sum((domain.coords.numpy() - source_xy) ** 2, 1)))
    return torch.from_numpy(csgraph.dijkstra(graph, directed=False, indices=source_node).astype(np.float32))


def make_tasks(domains: Iterable[ObstacleDomain], times: Iterable[float], ratio: float,
               seed: int, noise_std: float = 0.05, mask_kind: str = "random",
               target_kind: str = "heterogeneous", mismatch: float = 0.) -> list[WaveTask]:
    tasks = []
    for di, domain in enumerate(domains):
        for ti, time in enumerate(times):
            truth = (wave_field(domain, time) if target_kind == "matched" else
                     geodesic_harmonic_field(domain, time) if target_kind == "harmonic" else
                     geodesic_mixed_field(domain, time, mismatch) if target_kind == "mixed" else
                     geodesic_wavepacket_field(domain, time) if target_kind == "geodesic" else
                     elliptic_boundary_layer_field(domain, time) if target_kind == "elliptic"
                     else heterogeneous_scattering_field(domain, time))
            gen = torch.Generator().manual_seed(seed + 1009 * di + 53 * ti)
            nobs = max(4, int(round(ratio * domain.n_nodes)))
            if mask_kind == "random":
                ids = torch.randperm(domain.n_nodes, generator=gen)[:nobs]
            elif mask_kind == "upstream":
                candidates = torch.where(domain.coords[:, 0] < -0.05)[0]
                ids = candidates[torch.randperm(len(candidates), generator=gen)[:min(nobs, len(candidates))]]
            elif mask_kind == "boundary_stratified":
                near = torch.where(domain.signed_distance < 3.5 / domain.resolution)[0]
                nnear = min(len(near), nobs // 2)
                a = near[torch.randperm(len(near), generator=gen)[:nnear]]
                rest = torch.where(domain.signed_distance >= 3.5 / domain.resolution)[0]
                b = rest[torch.randperm(len(rest), generator=gen)[:nobs-nnear]]
                ids = torch.cat([a, b])
            else:
                raise ValueError(mask_kind)
            observed = torch.zeros(domain.n_nodes, dtype=torch.bool); observed[ids] = True
            sigma = noise_std * truth[ids].std().clamp_min(1e-6)
            noisy = truth.clone(); noisy[ids] += sigma * torch.randn(len(ids), generator=gen)
            tasks.append(WaveTask(domain, float(time), truth, observed, noisy))
    return tasks


def task_coordinate_features(task: WaveTask) -> torch.Tensor:
    n = task.domain.n_nodes
    time = torch.full((n, 1), task.time)
    desc = task.domain.descriptor[None].expand(n, -1)
    sdf = task.domain.signed_distance[:, None].clamp(max=0.6)
    radius = torch.linalg.vector_norm(task.domain.coords - torch.tensor([-0.72, -0.38]), dim=1, keepdim=True)
    return torch.cat([task.domain.coords, time, sdf, radius, desc], 1)


def rectangle_spectral_features(task: WaveTask, n_modes: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Obstacle-blind Neumann rectangle features, used as wrong-geometry control."""
    xy = (task.domain.coords + 1) / 2
    src = (torch.tensor([-0.72, -0.38]) + 1) / 2
    pairs = []
    side = int(math.ceil(math.sqrt(n_modes))) + 2
    for i in range(side):
        for j in range(side):
            pairs.append((i * i + j * j, i, j))
    pairs.sort()
    pairs = pairs[:n_modes]
    lam = torch.tensor([p[0] for p in pairs], dtype=torch.float32) * (math.pi / 2) ** 2
    cols = []
    for _, i, j in pairs:
        px = torch.cos(math.pi * i * xy[:, 0]); py = torch.cos(math.pi * j * xy[:, 1])
        ps = torch.cos(math.pi * i * src[0]) * torch.cos(math.pi * j * src[1])
        norm = (math.sqrt(2) if i else 1.0) * (math.sqrt(2) if j else 1.0)
        cols.append(px * py * ps * norm * norm)
    return torch.stack(cols, 1), lam


def rectangle_basis(task: WaveTask, n_modes: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Parameter-matched obstacle-blind rectangle eigenbasis."""
    xy = (task.domain.coords + 1) / 2
    pairs = []
    side = int(math.ceil(math.sqrt(n_modes))) + 2
    for i in range(side):
        for j in range(side): pairs.append((i * i + j * j, i, j))
    pairs.sort(); pairs = pairs[:n_modes]
    lam = torch.tensor([p[0] for p in pairs], dtype=torch.float32) * (math.pi / 2) ** 2
    cols = []
    for _, i, j in pairs:
        norm = (math.sqrt(2) if i else 1.0) * (math.sqrt(2) if j else 1.0)
        cols.append(norm * torch.cos(math.pi * i * xy[:, 0]) * torch.cos(math.pi * j * xy[:, 1]))
    return torch.stack(cols, 1), lam


class Sine(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, omega: float = 15.0):
        super().__init__(); self.linear = nn.Linear(in_dim, out_dim); self.omega = omega
        nn.init.uniform_(self.linear.weight, -1 / in_dim, 1 / in_dim)

    def forward(self, x):
        return torch.sin(self.omega * self.linear(x))


class CoordinateField(nn.Module):
    """Shared coordinate INR baseline (SIREN or random Fourier features)."""
    def __init__(self, in_dim: int, hidden: int = 128, kind: str = "siren", seed: int = 0):
        super().__init__(); self.kind = kind
        if kind == "rff":
            gen = torch.Generator().manual_seed(seed)
            self.register_buffer("projection", torch.randn(in_dim, hidden // 2, generator=gen) * 4.0)
            net_in = hidden
            self.net = nn.Sequential(nn.Linear(net_in, hidden), nn.GELU(), nn.Linear(hidden, hidden),
                                     nn.GELU(), nn.Linear(hidden, 1))
        elif kind == "siren":
            self.net = nn.Sequential(Sine(in_dim, hidden, 18.0), Sine(hidden, hidden, 18.0),
                                     nn.Linear(hidden, 1))
        else:
            raise ValueError(kind)

    def forward(self, x):
        if self.kind == "rff":
            phase = 2 * math.pi * x @ self.projection
            x = torch.cat([phase.sin(), phase.cos()], 1)
        return self.net(x).squeeze(-1)


class SharedNeuralCP(nn.Module):
    """Low-rank shared neural tensor baseline over x, y and task/domain context."""
    def __init__(self, context_dim: int, rank: int = 32, hidden: int = 64):
        super().__init__()
        def factor(dim):
            return nn.Sequential(nn.Linear(dim, hidden), nn.Tanh(), nn.Linear(hidden, rank))
        self.fx, self.fy, self.fc = factor(1), factor(1), factor(context_dim)
        self.weight = nn.Parameter(torch.ones(rank) / rank)

    def forward(self, x):
        return (self.fx(x[:, :1]) * self.fy(x[:, 1:2]) * self.fc(x[:, 2:]) * self.weight).sum(1)


class SpectralTransferNet(nn.Module):
    """A shared graph-spectral source-to-field adapter.

    ``phase_aligned`` exposes the dimensionless wave phase t*sqrt(lambda).
    This is the round-2 change targeting high-frequency spectral bias.
    """
    def __init__(self, hidden: int = 64, phase_aligned: bool = False):
        super().__init__(); self.phase_aligned = phase_aligned
        in_dim = 4 if phase_aligned else 2
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.GELU(), nn.Linear(hidden, hidden),
                                 nn.GELU(), nn.Linear(hidden, 1))
        self.high_gate = nn.Parameter(torch.tensor(-2.0))
        if phase_aligned:
            self.phase_net = nn.Sequential(Sine(4, hidden, 8.0), nn.Linear(hidden, 1))

    def transfer(self, eigenvalues: torch.Tensor, time: float | torch.Tensor):
        root = torch.sqrt(eigenvalues.clamp_min(0))
        loglam = torch.log1p(eigenvalues) / 8.0
        if isinstance(time, torch.Tensor):
            tt = time.to(root.device, root.dtype)
            while tt.ndim < root.ndim:
                tt = tt[..., None]
            tt = torch.broadcast_to(tt, root.shape)
        else:
            tt = torch.full_like(root, time)
        if self.phase_aligned:
            phase = root * tt
            feat = torch.stack([loglam, tt, torch.sin(phase), torch.cos(phase)], -1)
            base = self.net(feat).squeeze(-1)
            high = self.phase_net(feat).squeeze(-1)
            return base + torch.sigmoid(self.high_gate) * high
        return self.net(torch.stack([loglam, tt], -1)).squeeze(-1)

    def forward_task(self, task: WaveTask, node_ids: torch.Tensor | None = None,
                     wrong_geometry: bool = False):
        if wrong_geometry:
            z, lam = rectangle_spectral_features(task, len(task.domain.eigenvalues))
            z, lam = z.to(next(self.parameters()).device), lam.to(next(self.parameters()).device)
        else:
            phi = task.domain.eigenvectors.to(next(self.parameters()).device)
            src = task.domain.source_projection.to(phi.device)
            z, lam = phi * src[None], task.domain.eigenvalues.to(phi.device)
        if node_ids is not None:
            z = z[node_ids]
        return z @ self.transfer(lam, task.time)


class IntrinsicKernelField(nn.Module):
    """Local adapter over a multi-scale bank of intrinsic source kernels.

    Fixed heat and oscillatory kernels provide resolution-stable geometric
    coordinates; an INR learns spatially varying mode coupling on top.  The
    same network with rectangle kernels is the parameter-matched geometry
    ablation.
    """
    def __init__(self, coordinate_dim: int, hidden: int = 96):
        super().__init__()
        self.heat_scales = (0.0, .002, .006, .015, .04)
        self.wave_speeds = (.65, 1.0, 1.35)
        kernel_dim = len(self.heat_scales) + 2 * len(self.wave_speeds)
        self.net = nn.Sequential(nn.Linear(coordinate_dim + kernel_dim, hidden), nn.GELU(),
                                 nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 1))

    def feature_task(self, task: WaveTask, wrong_geometry: bool = False):
        device = next(self.parameters()).device
        if wrong_geometry:
            z, lam = rectangle_spectral_features(task, len(task.domain.eigenvalues))
        else:
            z = task.domain.eigenvectors * task.domain.source_projection[None]
            lam = task.domain.eigenvalues
        z, lam = z.to(device), lam.to(device)
        channels = [z @ torch.exp(-tau * lam) for tau in self.heat_scales]
        root = torch.sqrt(lam.clamp_min(0))
        for speed in self.wave_speeds:
            phase = speed * task.time * root
            envelope = torch.exp(-.0015 * lam)
            channels.extend([z @ (envelope * torch.cos(phase)), z @ (envelope * torch.sin(phase))])
        intrinsic = torch.stack(channels, 1)
        return torch.cat([task_coordinate_features(task).to(device), intrinsic], 1)

    def forward(self, features):
        return self.net(features).squeeze(-1)

    def forward_task(self, task: WaveTask, node_ids: torch.Tensor | None = None,
                     wrong_geometry: bool = False):
        features = self.feature_task(task, wrong_geometry)
        if node_ids is not None:
            features = features[node_ids]
        return self(features)


class GatedIntrinsicResidual(nn.Module):
    """RFF coordinate path plus a strongly bottlenecked intrinsic correction."""
    def __init__(self, coordinate_dim: int, hidden: int = 96, seed: int = 0):
        super().__init__()
        gen = torch.Generator().manual_seed(seed)
        self.register_buffer("projection", torch.randn(coordinate_dim, hidden // 2, generator=gen) * 4.0)
        self.base = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 1))
        # Eleven geometry kernels + their interaction with time. Keeping this
        # branch linear is intentional at 0.5% observations.
        self.correction = nn.Linear(22, 1, bias=False)
        self.gate = nn.Parameter(torch.tensor(-3.0))
        self.kernel_builder = IntrinsicKernelField(coordinate_dim, hidden)
        # Only use its deterministic feature routine, not its trainable decoder.
        for p in self.kernel_builder.net.parameters(): p.requires_grad_(False)

    def feature_task(self, task: WaveTask, wrong_geometry: bool = False):
        full = self.kernel_builder.feature_task(task, wrong_geometry)
        coordinate_dim = self.projection.shape[0]
        coord, kernels = full[:, :coordinate_dim], full[:, coordinate_dim:]
        phase = 2 * math.pi * coord @ self.projection
        base = torch.cat([phase.sin(), phase.cos()], 1)
        time = coord[:, 2:3]
        return torch.cat([base, kernels, kernels * time], 1)

    def forward(self, features):
        h = self.projection.shape[1] * 2
        return self.base(features[:, :h]).squeeze(-1) + torch.sigmoid(self.gate) * self.correction(features[:, h:]).squeeze(-1)

    def forward_task(self, task: WaveTask, node_ids: torch.Tensor | None = None,
                     wrong_geometry: bool = False):
        x = self.feature_task(task, wrong_geometry)
        return self(x if node_ids is None else x[node_ids])


class SensorConditionedRFF(nn.Module):
    """Shared RFF prior plus intrinsic heat-kernel sparse-context correction."""
    def __init__(self, coordinate_dim: int, hidden: int = 96, seed: int = 0,
                 diffusion: float = .008, ridge: float = .08, base_kind: str = "rff"):
        super().__init__()
        self.base = CoordinateField(coordinate_dim, hidden, base_kind, seed)
        self.diffusion, self.ridge = diffusion, ridge

    def forward(self, coordinate_features):
        return self.base(coordinate_features)

    def forward_task(self, task: WaveTask, node_ids: torch.Tensor | None = None,
                     wrong_geometry: bool = False, use_context: bool = False):
        device = next(self.parameters()).device
        coords = task_coordinate_features(task).to(device)
        base = self(coords)
        if use_context and int(task.observed.sum()) > 0:
            ids = torch.where(task.observed)[0].to(device)
            if wrong_geometry:
                phi, lam = rectangle_basis(task, len(task.domain.eigenvalues))
            else:
                phi, lam = task.domain.eigenvectors, task.domain.eigenvalues
            phi, lam = phi.to(device), lam.to(device)
            filt = torch.exp(-self.diffusion * lam)
            kqo = (phi * filt[None]) @ phi[ids].T / len(phi)
            koo = kqo[ids]
            alpha = self.ridge * torch.diagonal(koo).mean().clamp_min(1e-6)
            residual = task.noisy_values[task.observed].to(device) - base[ids]
            coeff = torch.linalg.solve(koo + alpha * torch.eye(len(ids), device=device), residual)
            base = base + kqo @ coeff
        return base if node_ids is None else base[node_ids]


class IntrinsicPhaseField(nn.Module):
    """Band-adaptive INR in intrinsic distance coordinates."""
    def __init__(self, context_dim: int, hidden: int = 96, use_bands: bool = True):
        super().__init__()
        self.use_bands = use_bands
        self.register_buffer("bands", torch.tensor([7., 13., 19., 27., 37.]))
        self.register_buffer("speeds", torch.tensor([.65, 1.0, 1.35]))
        # raw context plus sin/cos at 5 bands and 3 traveling phases per band
        phase_dim = 2 * len(self.bands) * (1 + len(self.speeds)) if use_bands else 0
        # Distance-only ablation gets one scalar; the phase model remains bitwise
        # compatible with the frozen confirmatory configuration.
        context_dim += 0 if use_bands else 1
        self.net = nn.Sequential(nn.Linear(context_dim + phase_dim, hidden), nn.GELU(),
                                 nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 1))

    def feature_task(self, task: WaveTask, wrong_geometry: bool = False):
        raw = task_coordinate_features(task)
        if wrong_geometry:
            distance = raw[:, 4]
        else:
            distance = source_geodesic_distance(task.domain)
        bands, speeds = self.bands.cpu(), self.speeds.cpu()
        if not self.use_bands:
            return torch.cat([raw, distance[:, None]], 1)
        phases = [distance[:, None] * bands[None]]
        for speed in speeds:
            phases.append((distance[:, None] - speed * task.time) * bands[None])
        phase = torch.cat(phases, 1)
        return torch.cat([raw, phase.sin(), phase.cos()], 1)

    def forward(self, features):
        return self.net(features).squeeze(-1)

    def forward_task(self, task: WaveTask, node_ids: torch.Tensor | None = None,
                     wrong_geometry: bool = False):
        x = self.feature_task(task, wrong_geometry).to(next(self.parameters()).device)
        return self(x if node_ids is None else x[node_ids])


def obstacle_boundary_mask(domain: ObstacleDomain, width: float = 0.12) -> torch.Tensor:
    return domain.signed_distance < width


def shadow_mask(domain: ObstacleDomain) -> torch.Tensor:
    # A fixed downstream diagnostic, not used for optimization.
    return (domain.coords[:, 0] > 0.05) & (domain.coords[:, 1].abs() < 0.45)
