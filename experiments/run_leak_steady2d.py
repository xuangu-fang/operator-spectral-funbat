#!/usr/bin/env python3
"""A purely spatial field, at four times the resolution.

Every other study here uses space plus time, which gives the tensor a mode that
every sensor layout observes in full and along which the field is nearly
constant.  That mode carries structure for free, and its presence makes it hard
to say whether the construction works because of the operator or because of the
time axis.  This removes it: two modes, 256x256, steady state.

It also raises the resolution, which the space-time studies could not afford.
The same 1% observation budget now covers 655 points of a 65536-cell domain, and
the unobserved region beyond a wall strip is four times deeper in grid units.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np, torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "experiments"))
from geoaware.config import add_config_arguments, load_config  # noqa: E402
from geoaware.operator_spectral_funbat import (  # noqa: E402
    ModeAdaptiveVariationalTucker, nonnegative_cp_spectrum, normalize_spectrum_cosine,
    real_cosine_basis)
from forced_pde_solver import solve_steady_state_2d  # noqa: E402
from run_leak_sensors import neumann_eigenvalues  # noqa: E402


def operator_spectra(shape, bins, nominal, atoms=4):
    """Per-mode spectra of the elliptic operator; there is no time factor here."""
    dx, dy = nominal["diffusivity"]
    lam_x = neumann_eigenvalues(shape[0])[:bins[0]]
    lam_y = neumann_eigenvalues(shape[1])[:bins[1]]
    elliptic = nominal["reaction"] + dx * lam_x[:, None] + dy * lam_y[None, :]
    joint = (1.0 / elliptic.square().clamp_min(1e-12)).float()
    mask = torch.ones_like(joint)
    mask[0, 0] = 0.0            # mean-centred data lacks exactly this joint mode
    separated = nonnegative_cp_spectrum(joint, rank=atoms, steps=1200, seed=17, mask=mask)
    return [normalize_spectrum_cosine(f) for f in separated.factors]


def matern_spectra(bins, length_scale):
    out = []
    for b in bins:
        k = torch.arange(b, dtype=torch.float32)
        s = (1 + (length_scale * k).square()).pow(-2.0)
        out.append(normalize_spectrum_cosine((s / s.sum())[None]))
    return out


def sensor_mask(shape, layout, budget, seed, device):
    nx, ny = shape
    generator = torch.Generator(device=device).manual_seed(seed + 7717)
    grid = torch.stack(torch.meshgrid(
        *[torch.arange(s, device=device) for s in shape], indexing="ij"), -1).reshape(-1, 2)
    x, y = grid[:, 0], grid[:, 1]
    depth = torch.minimum(torch.minimum(x, nx - 1 - x), torch.minimum(y, ny - 1 - y))
    # Fractions match the 64x64 layouts so the two resolutions are comparable.
    if layout == "random":
        region = torch.ones(len(grid), dtype=torch.bool, device=device)
    elif layout == "wall_ring":
        region = depth < nx // 32
    elif layout == "one_wall_strip":
        region = x < nx // 13
    elif layout == "corner_block":
        region = (x < nx // 3) & (y < ny // 3)
    else:
        raise ValueError(f"unknown layout {layout}")
    candidates = torch.nonzero(region, as_tuple=False).squeeze(-1)
    if len(candidates) < budget:
        raise ValueError(f"layout {layout} holds {len(candidates)} < budget {budget}")
    order = torch.randperm(len(candidates), generator=generator, device=device)
    keep = torch.zeros(len(grid), dtype=torch.bool, device=device)
    keep[candidates[order[:budget]]] = True
    return grid[keep], grid[~keep]


def fit(shape, observed, targets, test, truth, spectra, bases, ranks, *,
        steps, seed, device, lr):
    torch.manual_seed(seed + 10_000)
    model = ModeAdaptiveVariationalTucker(
        tuple(torch.arange(s, device=device) / s for s in shape),
        [s.to(device) for s in spectra], ranks=ranks, routing="global", noise_std=0.08,
        basis=("operator",) * len(shape),
        eigenbasis=tuple(b.to(device) for b in bases)).to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(steps):
        optimiser.zero_grad(set_to_none=True)
        loss, _ = model.negative_elbo(observed, targets, total_count=len(targets), samples=3)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        optimiser.step()
    with torch.no_grad():
        prediction = model.posterior_mean(test)
        return float(torch.sqrt(torch.mean((prediction - truth).square()))
                     / truth.std().clamp_min(1e-8))


def main() -> None:
    p = argparse.ArgumentParser()
    add_config_arguments(p)
    p.add_argument("--seeds", type=int, nargs="+", default=None)
    p.add_argument("--layouts", nargs="+", default=None)
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--lr", type=float, default=0.02)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--tag", default="steady2d")
    p.add_argument("--output", type=Path, default=ROOT / "results" / "leak")
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    device = torch.device(a.device)

    config = load_config(a.config, overrides=a.overrides)
    # A config meant for a space-time field produces a shape this script cannot
    # use, and the failure surfaces much later as a rank-length mismatch.
    if config.get("field.solver") != "steady_state_2d":
        raise SystemExit(
            f"config '{config.name}' has solver "
            f"'{config.get('field.solver')}'; this script needs a steady-state "
            "field.  Try --config steady2d.")
    settings = {k: v for k, v in config.field_kwargs().items()
                if k in {"grid", "diffusivity", "reaction", "sources", "background_noise"}}
    told = {k: v for k, v in config.nominal().items() if k != "drift"}
    bins = tuple(config.require("model")["bins"])
    ranks = tuple(config.require("model")["ranks"])
    length_scales = tuple(config.require("baselines")["length_scales"])
    seeds = a.seeds if a.seeds is not None else config.require("evaluation")["seeds"]
    layouts = a.layouts if a.layouts is not None else config.require("layouts")
    ratio = config.require("evaluation")["ratio"]
    noise_std = config.require("evaluation")["noise_std"]

    shape = tuple(solve_steady_state_2d(seed=0, **settings).field.shape)
    budget = int(round(ratio * int(np.prod(shape))))
    ours = operator_spectra(shape, bins, told)
    bases = tuple(real_cosine_basis(torch.arange(s, dtype=torch.float64) / s, b).float()
                  for s, b in zip(shape, bins))
    print(f"  config '{config.name}': {shape} = {int(np.prod(shape))} cells, "
          f"{budget} observed ({100 * ratio:.0f}%), {len(bins)} modes", flush=True)

    records = []
    for layout in layouts:
        rows: dict[str, list] = {}
        picked = []
        for seed in seeds:
            field = solve_steady_state_2d(seed=seed, **settings).field.to(device)
            observed, test = sensor_mask(shape, layout, budget, seed, device)
            g = torch.Generator(device=device).manual_seed(seed + 991)
            targets = field[tuple(observed.T)] + noise_std * torch.randn(
                len(observed), generator=g, device=device)
            truth = field[tuple(test.T)]

            def run(spectra, obs=observed, tgt=targets, ev=test, ev_truth=truth):
                return fit(shape, obs, tgt, ev, ev_truth, spectra, bases, ranks,
                           steps=a.steps, seed=seed, device=device, lr=a.lr)

            rows.setdefault("ours_pde", []).append(run(ours))
            rows.setdefault("matern_oracle", []).append(
                min(run(matern_spectra(bins, ls)) for ls in length_scales))

            split = torch.randperm(len(observed), generator=torch.Generator(
                device=device).manual_seed(seed + 4242), device=device)
            cut = int(len(observed) * 0.75)
            validation = {ls: run(matern_spectra(bins, ls), observed[split[:cut]],
                                  targets[split[:cut]], observed[split[cut:]],
                                  targets[split[cut:]]) for ls in length_scales}
            chosen = min(validation, key=validation.get)
            rows.setdefault("matern_deployable", []).append(
                run(matern_spectra(bins, chosen)))
            picked.append(chosen)

        cell = {"layout": layout, "observed": budget, "chosen_length_scales": picked}
        for key, values in rows.items():
            values = np.array(values)
            cell[key] = {"mean": float(values.mean()), "std": float(values.std()),
                         "values": values.tolist()}
        cell["tuning_cost"] = cell["matern_deployable"]["mean"] - cell["matern_oracle"]["mean"]
        cell["gap_to_oracle"] = cell["matern_oracle"]["mean"] - cell["ours_pde"]["mean"]
        records.append(cell)
        print(f"  {layout:16s} ours {cell['ours_pde']['mean']:.4f}   "
              f"deployable {cell['matern_deployable']['mean']:.4f}   "
              f"oracle {cell['matern_oracle']['mean']:.4f}   "
              f"validation picked {picked}", flush=True)

    (a.output / f"{a.tag}_summary.json").write_text(json.dumps(
        {"config": config.as_record(), "seeds": seeds, "records": records}, indent=2))


if __name__ == "__main__":
    main()
