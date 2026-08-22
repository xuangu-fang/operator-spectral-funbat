#!/usr/bin/env python3
"""AutoIP-style physics-informed GP: the equation as virtual observations.

Long et al., ICML 2022, put a GP prior on the solution and add the differential
equation as extra observations at collocation points, using the fact that a
linear operator applied to a GP is again a GP with cross-covariances obtained by
differentiating the kernel.  It is the closest relative of this work: physics
enters as a prior rather than as a loss, exactly as it does for us.

The difference is what carries the prior.  AutoIP keeps a standard GP over the
whole field, so it factorises a dense matrix over observations plus collocation
points.  We put the physics into per-mode spectra of a low-rank functional
tensor, which never forms that matrix.  So this baseline is here to be measured
on two axes at once, accuracy and cost, and the cost is reported rather than
asserted.

The kernel is a product squared-exponential.  Its derivatives are written
analytically because a fourth-order mixed derivative by autograd over a dense
kernel block is not affordable at this size; ``verify_against_autograd`` checks
the algebra against autograd on a small block, and the runner calls it.
"""
from __future__ import annotations

import torch


def _factors(difference: torch.Tensor, length_scale: float):
    """1-D squared-exponential and the r-derivatives the operator needs.

    With ``k`` a function of ``r = z - z'`` only: ``d/dz = d/dr`` and
    ``d/dz' = -d/dr``, so every derivative the operator needs is a derivative in
    ``r`` with a sign that the caller applies.
    """
    scaled = difference / length_scale
    value = torch.exp(-0.5 * scaled.square())
    second = (scaled.square() - 1.0) / length_scale ** 2 * value
    fourth = (3.0 - 6.0 * scaled.square() + scaled.pow(4)) / length_scale ** 4 * value
    first = -difference / length_scale ** 2 * value
    return value, first, second, fourth


class PhysicsInformedGP:
    """GP over ``[t, x, y]`` with ``L u = 0`` imposed at collocation points."""

    def __init__(self, *, length_scales, diffusivity, reaction, time_span,
                 signal_std=1.0, noise_std=0.05, residual_std=0.05, jitter=1e-6):
        self.length_scales = length_scales
        self.diffusivity = diffusivity
        self.reaction = reaction
        self.time_span = time_span
        self.signal = signal_std ** 2
        self.noise = noise_std ** 2
        self.residual = residual_std ** 2
        self.jitter = jitter

    def _pieces(self, left, right):
        parts = []
        for axis in range(3):
            difference = left[:, axis, None] - right[None, :, axis]
            parts.append(_factors(difference, self.length_scales[axis]))
        return parts

    def k_uu(self, left, right):
        (t, _, _, _), (x, _, _, _), (y, _, _, _) = self._pieces(left, right)
        return self.signal * t * x * y

    def k_ur(self, left, right):
        """cov(u(left), Lu(right)); L acts on the second argument."""
        (t, t1, t2, _), (x, _, x2, _), (y, _, y2, _) = self._pieces(left, right)
        # d/dt' = -d/dr ; d^2/dx'^2 = d^2/dr^2
        term = (-t1 / self.time_span) * x * y
        term = term - self.diffusivity[0] * t * x2 * y
        term = term - self.diffusivity[1] * t * x * y2
        term = term + self.reaction * t * x * y
        return self.signal * term

    def k_rr(self, left, right):
        """cov(Lu(left), Lu(right)), both operators applied."""
        (t, t1, t2, t4), (x, x1, x2, x4), (y, y1, y2, y4) = self._pieces(left, right)
        dx, dy, r = self.diffusivity[0], self.diffusivity[1], self.reaction
        s = self.time_span
        # d/dt d/dt' = -d^2/dr^2 ; d^2/dx^2 d^2/dx'^2 = d^4/dr^4 ;
        # d/dt d^2/dx'^2 = (d/dr_t)(d^2/dr_x^2) ; d^2/dx^2 d/dt' = -(same)
        total = (-t2 / s ** 2) * x * y
        total = total + (-dx / s) * t1 * x2 * y + (-dx / s) * (-t1) * x2 * y
        total = total + (-dy / s) * t1 * x * y2 + (-dy / s) * (-t1) * x * y2
        total = total + (r / s) * t1 * x * y + (r / s) * (-t1) * x * y
        total = total + dx * dx * t * x4 * y
        total = total + dy * dy * t * x * y4
        total = total + 2 * dx * dy * t * x2 * y2
        total = total - r * dx * t * x2 * y - r * dx * t * x2 * y
        total = total - r * dy * t * x * y2 - r * dy * t * x * y2
        total = total + r * r * t * x * y
        return self.signal * total

    def fit_predict(self, observed, targets, collocation, test, *, chunk=20000):
        n, m = len(observed), len(collocation)
        # Everything is built where the observations already live; a matrix
        # assembled on the wrong device is the whole cost of this method.
        where = dict(dtype=observed.dtype, device=observed.device)
        collocation = collocation.to(observed.device)
        top = torch.cat([self.k_uu(observed, observed) + (self.noise + self.jitter)
                         * torch.eye(n, **where),
                         self.k_ur(observed, collocation)], 1)
        # The lower-left block is cov(Lu(collocation), u(observed)), which is the
        # transpose of cov(u(observed), Lu(collocation)) -- not of the same
        # function called with the arguments swapped, since L acts on the second
        # argument only.
        bottom = torch.cat([self.k_ur(observed, collocation).T,
                            self.k_rr(collocation, collocation)
                            + (self.residual + self.jitter)
                            * torch.eye(m, **where)], 1)
        joint = torch.cat([top, bottom], 0)
        values = torch.cat([targets, torch.zeros(m, dtype=targets.dtype,
                                                 device=targets.device)])
        weights = torch.linalg.solve(joint, values)
        out = []
        for start in range(0, len(test), chunk):
            block = test[start:start + chunk]
            cross = torch.cat([self.k_uu(block, observed),
                               self.k_ur(block, collocation)], 1)
            out.append(cross @ weights)
        return torch.cat(out)


def verify_against_autograd(seed: int = 0, tolerance: float = 1e-4) -> dict:
    """Check the analytic derivative kernels against autograd on a small block."""
    torch.manual_seed(seed)
    model = PhysicsInformedGP(length_scales=(0.3, 0.2, 0.25), diffusivity=(0.03, 0.012),
                              reaction=0.06, time_span=38.4)
    left = torch.rand(4, 3, dtype=torch.float64)
    right = torch.rand(5, 3, dtype=torch.float64)

    def scalar_k(a, b):
        parts = 1.0
        for axis in range(3):
            scaled = (a[axis] - b[axis]) / model.length_scales[axis]
            parts = parts * torch.exp(-0.5 * scaled ** 2)
        return model.signal * parts

    def operator(function, point, other, on_first):
        point = point.clone().requires_grad_(True)
        value = function(point, other) if on_first else function(other, point)
        grad = torch.autograd.grad(value, point, create_graph=True)[0]
        laplacian = []
        for axis in (1, 2):
            component = grad[axis]
            laplacian.append(torch.autograd.grad(component, point, create_graph=True)[0][axis])
        return (grad[0] / model.time_span - model.diffusivity[0] * laplacian[0]
                - model.diffusivity[1] * laplacian[1] + model.reaction * value)

    numeric_ur = torch.zeros(4, 5, dtype=torch.float64)
    numeric_rr = torch.zeros(4, 5, dtype=torch.float64)
    for i in range(4):
        for j in range(5):
            numeric_ur[i, j] = operator(scalar_k, right[j], left[i], on_first=False)
            inner = lambda a, b: operator(scalar_k, b, a, on_first=False)  # noqa: E731
            numeric_rr[i, j] = operator(inner, left[i], right[j], on_first=True)
    report = {
        "k_ur_max_error": float((model.k_ur(left, right) - numeric_ur).abs().max()),
        "k_rr_max_error": float((model.k_rr(left, right) - numeric_rr).abs().max()),
    }
    report["passes"] = (report["k_ur_max_error"] < tolerance
                        and report["k_rr_max_error"] < tolerance)
    return report


if __name__ == "__main__":
    print(verify_against_autograd())
