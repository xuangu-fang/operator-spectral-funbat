#!/usr/bin/env python3
"""What the reconstructions actually look like, which no table can show.

A reader looking at NRMSE 0.54 against 0.81 has no idea what either means.  This
renders the time-averaged field: the truth, what each arm recovers from sensors
confined to one wall, and where each one's error lives.  The sensor strip is
drawn on every panel, because the whole argument is about the region the sensors
cannot see.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "experiments"))
from geoaware.operator_spectral_funbat import (  # noqa: E402
    ModeAdaptiveVariationalTucker, real_cosine_basis)
from forced_pde_solver import solve_multi_leak  # noqa: E402
import run_leak_sensors as base  # noqa: E402

RESULTS = ROOT / "results" / "leak"


def reconstruct(field, observed, targets, spectra, bases, shape, *, steps, seed, device, lr):
    torch.manual_seed(seed + 10_000)
    model = ModeAdaptiveVariationalTucker(
        tuple(torch.arange(s, device=device) / s for s in shape),
        [s.to(device) for s in spectra], ranks=base.RANKS, routing="global",
        noise_std=0.08, basis=("operator",) * 3,
        eigenbasis=tuple(b.to(device) for b in bases)).to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(steps):
        optimiser.zero_grad(set_to_none=True)
        loss, _ = model.negative_elbo(observed, targets, total_count=len(targets), samples=3)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        optimiser.step()
    with torch.no_grad():
        grid = torch.stack(torch.meshgrid(
            *[torch.arange(s, device=device) for s in shape], indexing="ij"), -1).reshape(-1, 3)
        whole = torch.cat([model.posterior_mean(grid[i:i + 200_000])
                           for i in range(0, len(grid), 200_000)])
    return whole.reshape(shape).cpu()


def main(seed: int = 0, layout: str = "one_wall_strip", steps: int = 1000,
         device_name: str = "cuda" if torch.cuda.is_available() else "cpu") -> None:
    device = torch.device(device_name)
    solved = solve_multi_leak(seed=seed, **base.FIELD)
    field = solved.field.to(device)
    shape = tuple(field.shape)
    budget = int(round(0.01 * int(np.prod(shape))))
    bases = tuple(real_cosine_basis(torch.arange(s, dtype=torch.float64) / s, b).float()
                  for s, b in zip(shape, base.BINS))
    observed, test = base.sensor_mask(shape, layout, budget, seed, device)
    g = torch.Generator(device=device).manual_seed(seed + 991)
    targets = field[tuple(observed.T)] + 0.05 * torch.randn(len(observed), generator=g,
                                                            device=device)
    truth = field[tuple(test.T)]

    # The deployable Matern picks its length scale on a split of the readings.
    split = torch.randperm(len(observed), generator=torch.Generator(
        device=device).manual_seed(seed + 4242), device=device)
    cut = int(len(observed) * 0.75)
    validation = {ls: base.fit_gp(field, observed[split[:cut]], targets[split[:cut]],
                                  observed[split[cut:]], targets[split[cut:]],
                                  base.matern_spectra(base.BINS, ls), bases, steps=steps,
                                  seed=seed, device=device, lr=0.02)
                  for ls in base.LENGTH_SCALES}
    chosen = min(validation, key=validation.get)

    arms = [("operator prior (ours)", base.operator_spectra(shape, base.FIELD["dt"], base.BINS)),
            (rf"Mat\'ern tuned on sensors ($\ell$={chosen})",
             base.matern_spectra(base.BINS, chosen)),
            (r"Mat\'ern, fixed $\ell$=1.6", base.matern_spectra(base.BINS, 1.6))]
    arms = [(label.replace(r"\'", ""), spectra) for label, spectra in arms]

    truth_map = field.mean(0).cpu().numpy()
    panels = [("truth", truth_map, None)]
    for label, spectra in arms:
        whole = reconstruct(field, observed, targets, spectra, bases, shape,
                            steps=steps, seed=seed, device=device, lr=0.02)
        with torch.no_grad():
            model_error = float(torch.sqrt(torch.mean(
                (whole.to(device)[tuple(test.T)] - truth).square()))
                / truth.std().clamp_min(1e-8))
        panels.append((f"{label}\nNRMSE {model_error:.3f}", whole.mean(0).numpy(), None))

    strip = int(observed[:, 1].max().item()) + 1
    limit = float(np.abs(truth_map).max())
    figure, axes = plt.subplots(2, len(panels), figsize=(3.1 * len(panels), 6.4))
    for column, (title, values, _) in enumerate(panels):
        top = axes[0, column]
        top.imshow(values.T, origin="lower", cmap="RdBu_r", vmin=-limit, vmax=limit)
        top.set_title(title, fontsize=9.5)
        top.set_xticks([]); top.set_yticks([])
        top.add_patch(Rectangle((-0.5, -0.5), strip, values.shape[1],
                                facecolor="none", edgecolor="black", lw=1.8))
        bottom = axes[1, column]
        if column == 0:
            bottom.axis("off")
            bottom.text(0.5, 0.5, "sensors occupy the boxed strip;\n"
                        "everything else is extrapolated\n\n"
                        "bottom row: |error|, shared scale",
                        ha="center", va="center", fontsize=9.5)
            continue
        residual = np.abs(values - truth_map)
        image = bottom.imshow(residual.T, origin="lower", cmap="magma", vmin=0,
                              vmax=float(np.abs(panels[2][1] - truth_map).max()))
        bottom.set_xticks([]); bottom.set_yticks([])
        bottom.add_patch(Rectangle((-0.5, -0.5), strip, values.shape[1],
                                   facecolor="none", edgecolor="white", lw=1.8))
        bottom.set_title(f"mean |error| {residual.mean():.3f}", fontsize=9)
    figure.colorbar(image, ax=axes[1, 1:].tolist(), fraction=0.03, pad=0.01)
    figure.suptitle(f"Time-averaged field, sensors on one wall "
                    f"({budget} readings, 1% of the room), seed {seed}", fontsize=11)
    figure.savefig(RESULTS / "figure_reconstruction.png", dpi=170, bbox_inches="tight")
    plt.close(figure)
    print("wrote figure_reconstruction.png")


if __name__ == "__main__":
    main()
