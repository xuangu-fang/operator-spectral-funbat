"""Sign-invariant kernels on irregular domains.

The functions in this module turn a Laplacian eigenbasis into a small set of
Matérn-kernel sections ``k_Omega(x, source)``. A section is invariant to an
eigenvector sign flip and gives a mesh-independent, geometry-aware coordinate
for a query point. This is deliberately a finite-feature POC, not a claim of
full variational Gaussian-process inference.
"""

from __future__ import annotations

from collections.abc import Sequence
import heapq

import torch


def _rms_normalize_sections(sections: torch.Tensor) -> torch.Tensor:
    """Normalize channels without using any field/target values."""
    rms = sections.square().mean(dim=(0, 1), keepdim=True).sqrt().clamp_min(1e-6)
    return sections / rms


def matern_domain_kernel_sections(
    basis: torch.Tensor,
    eigenvalues: torch.Tensor,
    source_nodes: torch.Tensor,
    *,
    scales: Sequence[float] = (0.03, 0.1, 0.3, 1.0, 3.0),
    smoothness: float = 1.5,
) -> torch.Tensor:
    """Return normalized ``[source, node, scale]`` domain-kernel sections.

    For graph-Laplacian eigenpairs ``(phi_j, lambda_j)``, channel ``q`` is

    ``sum_j phi_j(x) phi_j(s) (1 + scale_q * lambda_j)^(-smoothness)``.

    This is the finite spectral form of a Matérn-like GP covariance on the
    domain. Normalizing each channel by its RMS makes features comparable
    across meshes and resolutions without changing their topology.
    """
    if basis.ndim != 2 or eigenvalues.ndim != 1:
        raise ValueError("basis must be [node, mode] and eigenvalues [mode]")
    if basis.shape[1] != len(eigenvalues):
        raise ValueError("basis/eigenvalue mode counts do not match")
    if not scales:
        raise ValueError("at least one kernel scale is required")

    phi = basis.float()
    lam = eigenvalues.float().clamp_min(0)
    source_phi = phi[source_nodes.long()]
    scale_tensor = torch.as_tensor(scales, dtype=phi.dtype, device=phi.device)
    filters = (1 + scale_tensor[:, None] * lam[None]).pow(-smoothness)
    sections = torch.einsum("nk,sk,qk->snq", phi, source_phi, filters)
    sections = sections / max(1, basis.shape[1])
    return _rms_normalize_sections(sections)


def heat_domain_kernel_sections(
    basis: torch.Tensor,
    eigenvalues: torch.Tensor,
    source_nodes: torch.Tensor,
    *,
    diffusion_times: Sequence[float] = (0.03, 0.1, 0.3, 1.0, 3.0),
) -> torch.Tensor:
    """Return heat-kernel sections ``exp(-t L_Omega)``.

    The construction is identical to :func:`matern_domain_kernel_sections`
    except for the spectral response.  It therefore isolates *kernel family*
    rather than feature budget, eigensolver, source placement, or downstream
    inference.  The graph Laplacian and its boundary condition define which
    paths diffusion can follow around walls and holes.
    """
    if basis.ndim != 2 or eigenvalues.ndim != 1:
        raise ValueError("basis must be [node, mode] and eigenvalues [mode]")
    if basis.shape[1] != len(eigenvalues):
        raise ValueError("basis/eigenvalue mode counts do not match")
    if not diffusion_times or min(diffusion_times) <= 0:
        raise ValueError("diffusion_times must be positive")

    phi = basis.float()
    lam = eigenvalues.float().clamp_min(0)
    source_phi = phi[source_nodes.long()]
    times = torch.as_tensor(diffusion_times, dtype=phi.dtype, device=phi.device)
    filters = torch.exp(-times[:, None] * lam[None])
    sections = torch.einsum("nk,sk,qk->snq", phi, source_phi, filters)
    sections = sections / max(1, basis.shape[1])
    return _rms_normalize_sections(sections)


def geodesic_rbf_kernel_sections(
    coordinates: torch.Tensor,
    source_nodes: torch.Tensor,
    undirected_edges: torch.Tensor,
    *,
    lengthscales: Sequence[float] = (0.08, 0.16, 0.32, 0.64, 1.28),
) -> torch.Tensor:
    """Return RBF sections based on shortest paths inside the domain graph.

    Edge lengths are measured in the ambient coordinates, while distances are
    accumulated only along valid mesh edges.  Consequently, two points across
    a thin wall can be close for the Euclidean control but far for this kernel.
    Dijkstra is intentionally used here because the POC meshes are small and
    it makes the geometry semantics auditable.
    """
    if coordinates.ndim != 2:
        raise ValueError("coordinates must be [node, coordinate]")
    if undirected_edges.ndim != 2 or undirected_edges.shape[1] != 2:
        raise ValueError("undirected_edges must be [edge,2]")
    if not lengthscales or min(lengthscales) <= 0:
        raise ValueError("lengthscales must be positive")

    coords = coordinates.detach().float().cpu()
    edges = undirected_edges.detach().long().cpu()
    node_count = len(coords)
    adjacency: list[list[tuple[int, float]]] = [[] for _ in range(node_count)]
    for left, right in edges.tolist():
        if not (0 <= left < node_count and 0 <= right < node_count):
            raise ValueError("edge endpoint outside coordinate table")
        weight = float(torch.linalg.vector_norm(coords[left] - coords[right]))
        adjacency[left].append((right, weight))
        adjacency[right].append((left, weight))

    all_distances = []
    for source in source_nodes.detach().long().cpu().tolist():
        if not 0 <= source < node_count:
            raise ValueError("source node outside coordinate table")
        distances = [float("inf")] * node_count
        distances[source] = 0.0
        queue = [(0.0, source)]
        while queue:
            distance, node = heapq.heappop(queue)
            if distance != distances[node]:
                continue
            for neighbour, weight in adjacency[node]:
                candidate = distance + weight
                if candidate < distances[neighbour]:
                    distances[neighbour] = candidate
                    heapq.heappush(queue, (candidate, neighbour))
        all_distances.append(distances)

    distance = torch.tensor(all_distances, dtype=coordinates.dtype,
                            device=coordinates.device)
    ell = torch.as_tensor(lengthscales, dtype=distance.dtype,
                          device=distance.device)
    sections = torch.exp(-distance[..., None].square() / (2 * ell.square()))
    return _rms_normalize_sections(sections)


def euclidean_rbf_kernel_sections(
    coordinates: torch.Tensor,
    source_nodes: torch.Tensor,
    *,
    lengthscales: Sequence[float] = (0.08, 0.16, 0.32, 0.64, 1.28),
) -> torch.Tensor:
    """Return normalized Euclidean RBF sections ``[source, node, scale]``.

    This is the method-matched geometry-agnostic control for
    :func:`matern_domain_kernel_sections`: both expose covariance sections
    centred at exactly the same source nodes and have the same channel count.
    Unlike the intrinsic sections, Euclidean distance can connect points that
    are close through a wall or across a hole.

    The returned sections are deterministic geometry features.  They become a
    GP covariance only when an inference method actually uses the corresponding
    kernel and Gaussian prior; passing them through an MLP is *not* GP inference.
    """
    if coordinates.ndim != 2:
        raise ValueError("coordinates must be [node, coordinate]")
    if not lengthscales:
        raise ValueError("at least one lengthscale is required")
    if min(lengthscales) <= 0:
        raise ValueError("lengthscales must be positive")

    coords = coordinates.float()
    source_coords = coords[source_nodes.long()]
    squared_distance = (source_coords[:, None] - coords[None]).square().sum(-1)
    ell = torch.as_tensor(lengthscales, dtype=coords.dtype, device=coords.device)
    sections = torch.exp(-squared_distance[..., None] / (2 * ell.square()))
    return _rms_normalize_sections(sections)
