#!/usr/bin/env python3
"""The leak room in three spatial dimensions, which is the real geometry.

In two dimensions a wall-mounted array is a line and the unobserved region is a
half-plane.  In three it is a *face*, and the unobserved region is a half
volume, so the same fraction of observed cells buys much less and the
extrapolation is longer in every direction at once.  This is the setting the
applications actually have, and it is also where a per-mode construction has
most to gain or lose: there are now three spatial spectra to get right.

The comparison is the same three tiers as the main table -- ours with no tuning
data, a Matern tuned on a split of the sensor readings, and a Matern tuned
against the held-out region -- so the numbers are read the same way.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np, torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "experiments"))
from geoaware.operator_spectral_funbat import (  # noqa: E402
    ModeAdaptiveVariationalTucker, nonnegative_cp_spectrum, normalize_spectrum_cosine,
    real_cosine_basis)
from forced_pde_solver import solve_multi_leak_3d  # noqa: E402
from geoaware.config import add_config_arguments, load_config  # noqa: E402
from run_leak_sensors import neumann_eigenvalues  # noqa: E402

FIELD = dict(grid=(32, 32, 32), diffusivity=(0.02, 0.006, 0.012), reaction=0.04,
             sources=((0.30, 0.65, 0.40, 0.10, 15.0), (0.70, 0.35, 0.60, 0.09, 25.0),
                      (0.55, 0.75, 0.30, 0.08, 40.0)),
             dt=0.6, burn_in=200, record_steps=32, background_noise=0.02)
NOMINAL = dict(diffusivity=(0.03, 0.012, 0.018), reaction=0.06)
LENGTH_SCALES = (0.12, 0.32, 0.8, 1.6, 2.4, 3.5)
BINS = (8, 8, 8, 8)
RANKS = (5, 4, 4, 4)


def operator_spectra(shape, dt, bins, atoms=4):
    dx, dy, dz = NOMINAL["diffusivity"]
    lam = [neumann_eigenvalues(shape[axis + 1])[:bins[axis + 1]] for axis in range(3)]
    omega = np.pi * torch.arange(bins[0], dtype=torch.float64) / (shape[0] * dt)
    elliptic = (NOMINAL["reaction"] + dx * lam[0][:, None, None]
                + dy * lam[1][None, :, None] + dz * lam[2][None, None, :])
    joint = (1.0 / (omega[:, None, None, None].square()
                    + elliptic[None].square()).clamp_min(1e-12)).float()
    mask = torch.ones_like(joint)
    mask[0, 0, 0, 0] = 0.0        # mean-centred data lacks exactly this joint mode
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
    """Sensor-eligible cells for a three-dimensional room."""
    generator = torch.Generator(device=device).manual_seed(seed + 7717)
    grid = torch.stack(torch.meshgrid(
        *[torch.arange(s, device=device) for s in shape], indexing="ij"), -1).reshape(-1, 4)
    x, y, z = grid[:, 1], grid[:, 2], grid[:, 3]
    nx, ny, nz = shape[1], shape[2], shape[3]
    depth = torch.minimum(torch.minimum(torch.minimum(x, nx - 1 - x),
                                        torch.minimum(y, ny - 1 - y)),
                          torch.minimum(z, nz - 1 - z))
    if layout == "random":
        region = torch.ones(len(grid), dtype=torch.bool, device=device)
    elif layout == "one_face":
        region = x < 3
    elif layout == "two_faces_opposite":
        region = (x < 3) | (x >= nx - 3)
    elif layout == "all_six_faces":
        region = depth < 1
    elif layout == "floor_only":
        region = z < 3
    elif layout == "corner_cube":
        region = (x < 10) & (y < 10) & (z < 10)
    else:
        raise ValueError(f"unknown layout {layout}")
    candidates = torch.nonzero(region, as_tuple=False).squeeze(-1)
    if len(candidates) < budget:
        raise ValueError(f"layout {layout} holds {len(candidates)} < budget {budget}")
    order = torch.randperm(len(candidates), generator=generator, device=device)
    keep = torch.zeros(len(grid), dtype=torch.bool, device=device)
    keep[candidates[order[:budget]]] = True
    return grid[keep], grid[~keep]


def fit_gp(shape, observed, targets, test, truth, spectra, bases, *, steps, seed, device, lr):
    torch.manual_seed(seed + 10_000)
    model = ModeAdaptiveVariationalTucker(
        tuple(torch.arange(s, device=device) / s for s in shape),
        [s.to(device) for s in spectra], ranks=RANKS, routing="global", noise_std=0.08,
        basis=("operator",) * 4,
        eigenbasis=tuple(b.to(device) for b in bases)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss, _ = model.negative_elbo(observed, targets, total_count=len(targets), samples=3)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        optimizer.step()
    with torch.no_grad():
        prediction = model.posterior_mean(test)
        return float(torch.sqrt(torch.mean((prediction - truth).square()))
                     / truth.std().clamp_min(1e-8))


def main() -> None:
    p = argparse.ArgumentParser()
    add_config_arguments(p)
    p.add_argument("--seeds", type=int, nargs="+", default=None)
    p.add_argument("--layouts", nargs="+",
                   default=["random", "all_six_faces", "two_faces_opposite",
                            "one_face", "corner_cube"])
    p.add_argument("--ratio", type=float, default=0.01)
    p.add_argument("--noise-std", type=float, default=0.05)
    p.add_argument("--steps", type=int, default=800)
    p.add_argument("--lr", type=float, default=0.02)
    p.add_argument("--max-test", type=int, default=60_000,
                   help="held-out cells are subsampled to keep evaluation affordable")
    p.add_argument("--tag", default="leak3d")
    p.add_argument("--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="an idle GPU makes these sweeps roughly an order of magnitude cheaper")
    p.add_argument("--output", type=Path, default=ROOT / "results" / "leak")
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    device = torch.device(a.device)

    config = load_config(a.config, overrides=a.overrides)
    settings = config.field_kwargs()
    NOMINAL.update({k: v for k, v in config.nominal().items()})
    seeds = a.seeds if a.seeds is not None else config.require("evaluation")["seeds"]
    print(f"  config '{config.name}'"
          + (f" with {a.overrides}" if a.overrides else ""), flush=True)

    shape = tuple(solve_multi_leak_3d(seed=0, **settings).field.shape)
    budget = int(round(a.ratio * int(np.prod(shape))))
    ours = operator_spectra(shape, settings["dt"], BINS)
    bases = tuple(real_cosine_basis(torch.arange(s, dtype=torch.float64) / s, b).float()
                  for s, b in zip(shape, BINS))
    print(f"  tensor {shape}, {int(np.prod(shape))} cells, {budget} observed", flush=True)

    records = []
    for layout in a.layouts:
        rows: dict[str, list] = {}
        for seed in seeds:
            field = solve_multi_leak_3d(seed=seed, **settings).field.to(device)
            observed, test = sensor_mask(shape, layout, budget, seed, device)
            if len(test) > a.max_test:
                pick = torch.randperm(len(test), generator=torch.Generator(
                    device=device).manual_seed(seed + 55), device=device)[:a.max_test]
                test = test[pick]
            g = torch.Generator(device=device).manual_seed(seed + 991)
            targets = field[tuple(observed.T)] + a.noise_std * torch.randn(
                len(observed), generator=g, device=device)
            truth = field[tuple(test.T)]

            def fit(spectra, obs=observed, tgt=targets, ev=test, ev_truth=truth):
                return fit_gp(shape, obs, tgt, ev, ev_truth, spectra, bases,
                              steps=a.steps, seed=seed, device=device, lr=a.lr)

            rows.setdefault("ours_pde", []).append(fit(ours))
            oracle = {ls: fit(matern_spectra(BINS, ls)) for ls in LENGTH_SCALES}
            # Keep every candidate, not only the winner.  A practitioner who
            # declines to tune and simply fixes a sensible length scale is a
            # competitor that also uses no tuning data, and the only way to
            # judge them is to see how one value fares across every layout.
            for ls, score in oracle.items():
                rows.setdefault(f"fixed_{ls}", []).append(score)
            best = min(oracle, key=oracle.get)
            rows.setdefault("matern_oracle", []).append(oracle[best])
            rows.setdefault("_oracle_ls", []).append(best)

            split = torch.randperm(len(observed), generator=torch.Generator(
                device=device).manual_seed(seed + 4242), device=device)
            cut = int(len(observed) * 0.75)
            inner, held = split[:cut], split[cut:]
            validation = {ls: fit(matern_spectra(BINS, ls), observed[inner], targets[inner],
                                  observed[held], targets[held]) for ls in LENGTH_SCALES}
            chosen = min(validation, key=validation.get)
            rows.setdefault("matern_deployable", []).append(fit(matern_spectra(BINS, chosen)))
            rows.setdefault("_chosen_ls", []).append(chosen)

        cell = {"layout": layout, "observed": budget,
                "chosen_length_scales": rows.pop("_chosen_ls"),
                "oracle_length_scales": rows.pop("_oracle_ls")}
        for key, values in rows.items():
            values = np.array(values)
            cell[key] = {"mean": float(values.mean()), "std": float(values.std()),
                         "values": values.tolist()}
        cell["tuning_cost"] = cell["matern_deployable"]["mean"] - cell["matern_oracle"]["mean"]
        cell["gap_to_oracle"] = cell["matern_oracle"]["mean"] - cell["ours_pde"]["mean"]
        records.append(cell)
        print(f"  {layout:20s} ours {cell['ours_pde']['mean']:.4f}  "
              f"deployable {cell['matern_deployable']['mean']:.4f}  "
              f"oracle {cell['matern_oracle']['mean']:.4f}  "
              f"tuning cost {cell['tuning_cost']:+.4f}  "
              f"ours vs oracle {cell['gap_to_oracle']:+.4f}", flush=True)
        print(f"  {'':20s} validation picked {cell['chosen_length_scales']}, "
              f"the answer wanted {cell['oracle_length_scales']}", flush=True)

    (a.output / f"{a.tag}_summary.json").write_text(json.dumps(
        {"field": {k: str(v) for k, v in FIELD.items()}, "nominal": NOMINAL,
         "bins": BINS, "ranks": RANKS, "seeds": seeds, "config": config.as_record(), "records": records}, indent=2))


if __name__ == "__main__":
    main()
