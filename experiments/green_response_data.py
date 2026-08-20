#!/usr/bin/env python3
"""A Green-response tensor: one 1-D PDE, three modes, one operator.

This is the setting the method was really built for, and it is a real
measurement modality rather than a generic completion benchmark.  Fire an
impulse at source position `s`, record at receiver `r`, sample over time `t`:
the entries `u(t, r, s)` form a three-mode tensor.  Seismic gathers, thermal
tomography, ultrasound arrays and network-diffusion probing all produce exactly
this object, and in all of them you cannot fire every source, record at every
receiver, and sample every instant -- so sparse observation is the actual
problem, not a synthetic constraint.

The construction's usefulness is that a *single* one-dimensional operator
determines the structure of *all three* modes.  For

    d_t u + (L_a + kappa) u = 0,     L_a = -d/dx (a(x) d/dx),   zero flux,

the eigenpairs of `L_a` give

    u(t, r, s) = sum_q  exp(-t (kappa + lambda_q))  phi_q(r) phi_q(s) w_q,

so the time mode inherits the decay basis fixed by the eigenvalues, and the
receiver and source modes both inherit the eigenfunctions -- the latter two
being identical because a Green's function is symmetric in source and receiver.

Misspecification is physical rather than invented.  Ground truth uses a
variable coefficient `a(x)`; the learner is told only the *form* -- diffusion
with zero-flux boundaries -- and uses the constant-coefficient reference
operator.  Both the eigenvectors and the decay rates are therefore wrong, which
is exactly the "know the equation, not the medium" case.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch


@dataclass(frozen=True)
class GreenTensor:
    field: torch.Tensor                 # [t, receiver, source], standardized
    reference_eigenvalues: torch.Tensor  # what the learner may use
    reference_eigenvectors: torch.Tensor
    true_eigenvalues: torch.Tensor       # recorded for diagnostics only
    time: torch.Tensor
    metadata: dict


def neumann_diffusion_operator(
    size: int, contrast: float, phase: float = 0.37,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Finite-volume ``-d/dx(a(x) d/dx)`` with zero-flux boundaries.

    The symmetric edge-flux discretisation is positive semi-definite and keeps
    the constant Neumann mode exactly, so eigenvalue zero is preserved.
    ``contrast`` is a log-diffusivity amplitude: ``contrast = 0`` is the uniform
    medium the learner assumes.
    """
    if size < 4:
        raise ValueError("need at least four grid points")
    if contrast < 0:
        raise ValueError("contrast must be non-negative")
    x = torch.linspace(0.0, 1.0, size, dtype=torch.float64)
    midpoint = 0.5 * (x[:-1] + x[1:])
    diffusivity = torch.exp(contrast * (
        torch.cos(2 * math.pi * midpoint)
        + 0.35 * torch.sin(3 * math.pi * midpoint + phase)))
    conductance = diffusivity * (size - 1) ** 2
    operator = torch.zeros(size, size, dtype=torch.float64)
    edge = torch.arange(size - 1)
    operator[edge, edge] += conductance
    operator[edge + 1, edge + 1] += conductance
    operator[edge, edge + 1] -= conductance
    operator[edge + 1, edge] -= conductance
    return operator, diffusivity


def green_response_tensor(
    *,
    grid: int = 24,
    time_points: int = 18,
    contrast: float = 1.0,
    reaction: float = 0.15,
    truth_modes: int = 14,
    spectral_decay: float = 0.18,
    time_span: tuple[float, float] = (0.025, 0.55),
    learner_modes: int = 8,
) -> GreenTensor:
    """Build `u(t, receiver, source)` and the reference operator the learner sees.

    Time starts away from zero: the `t = 0` Green kernel is a delta, which would
    turn the task into completing an identity matrix rather than a physical
    field.
    """
    if not 2 <= learner_modes <= grid:
        raise ValueError("learner_modes must lie in [2, grid]")
    if not learner_modes <= truth_modes <= grid:
        raise ValueError("truth_modes must lie in [learner_modes, grid]")

    reference, _ = neumann_diffusion_operator(grid, 0.0)
    physical, diffusivity = neumann_diffusion_operator(grid, contrast)
    ref_values, ref_vectors = torch.linalg.eigh(reference)
    true_values, true_vectors = torch.linalg.eigh(physical)
    # Normalise both spectra by their own first non-zero eigenvalue so that the
    # comparison is about spectral *shape*, not about an overall time rescaling
    # the learner could absorb anyway.
    ref_rates = ref_values / ref_values[1].clamp_min(1e-12)
    true_rates = true_values / true_values[1].clamp_min(1e-12)

    time = torch.linspace(*time_span, time_points, dtype=torch.float64)
    decay = torch.exp(-time[:, None] * (reaction + true_rates[:truth_modes][None, :]))
    weight = (1 + true_rates[:truth_modes]).pow(-spectral_decay)
    field = torch.einsum("tq,rq,sq,q->trs", decay,
                         true_vectors[:, :truth_modes],
                         true_vectors[:, :truth_modes], weight)
    field = (field - field.mean()) / field.std().clamp_min(1e-12)

    return GreenTensor(
        field=field.float(),
        reference_eigenvalues=ref_rates[:learner_modes].float(),
        reference_eigenvectors=ref_vectors[:, :learner_modes].float(),
        true_eigenvalues=true_rates[:truth_modes].float(),
        time=time.float(),
        metadata={
            "pde": "d_t u + (-d/dx(a(x) d/dx) + kappa) u = 0",
            "boundary_condition": "homogeneous Neumann (zero flux)",
            "tensor_semantics": "time x receiver x source (Green response)",
            "log_diffusivity_contrast": float(contrast),
            "diffusivity_min": float(diffusivity.min()),
            "diffusivity_max": float(diffusivity.max()),
            "reaction": float(reaction),
            "truth_modes": int(truth_modes),
            "learner_modes": int(learner_modes),
            "learner_knows": "the equation form and the zero-flux boundary; "
                             "not a(x), so both eigenvectors and decay rates are wrong",
        },
    )


def learner_bases(green: GreenTensor, reaction: float = 0.15
                  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The three per-mode bases the learner derives from the reference operator.

    Receiver and source use the eigenvectors directly.  Time uses the decay
    family ``exp(-t (kappa + lambda_q))`` orthonormalised, which is what the
    operator predicts the temporal structure to be -- so all three bases come
    from the same one-dimensional operator and nothing is chosen by hand.
    """
    rates = green.reference_eigenvalues.double()
    decay = torch.exp(-green.time.double()[:, None] * (reaction + rates[None, :]))
    time_basis = torch.linalg.qr(decay, mode="reduced").Q.float()
    spatial = green.reference_eigenvectors
    return time_basis, spatial, spatial.clone()


def operator_spectra(green: GreenTensor, spectral_decay: float = 0.18
                     ) -> torch.Tensor:
    """Per-mode nonnegative spectra over those bases, read off the operator.

    The solution weights each eigenmode by ``(1 + lambda_q)^-p`` and the time
    mode by the same decay family, so the operator supplies the *relative*
    importance of the basis functions as well as the basis itself.
    """
    rates = green.reference_eigenvalues.double()
    weight = (1 + rates).pow(-2 * spectral_decay)      # power, hence 2p
    weight = (weight / weight.sum()).float()
    return torch.stack([weight, weight.clone(), weight.clone()])
