#!/usr/bin/env python3
"""Separable band-pass: the structure a monotone kernel cannot follow.

A generic stationary kernel's spectrum decays monotonically from zero
frequency.  When the truth instead has almost no energy at low frequency and
all of it in a band, the generic family is structurally wrong, not merely
mis-tuned -- and unlike anisotropy, no length scale repairs it.

The band must be *separable* to be useful here: a wave's temporal resonance
sits at c sqrt(lambda_q) and therefore moves with the spatial mode, putting the
joint spectrum on a dispersion surface.  Driving a dissipative operator with
temporally narrowband noise puts the same band on every spatial mode instead,
so the joint spectrum stays close to a product.  Measured before any training,
the rank-4 nonnegative separation error is 0.005--0.013 across band centres,
while the time marginal is numerically zero in its first three bins.

Physically this is ordinary rather than contrived: ocean swell, rotating
machinery, ambient seismic noise in a band, mains ripple.
"""

from __future__ import annotations

import argparse, json, sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "experiments"))

from geoaware.operator_spectral_funbat import (  # noqa: E402
    extended_generic_dictionary, nonnegative_cp_spectrum, normalize_spectrum_cosine,
    operator_joint_spectrum,
)
from forced_pde_solver import solve_narrowband_forced  # noqa: E402
from run_forced_pde_main import make_task, nrmse, tucker_ceiling  # noqa: E402
from run_forced_pde_ablation import train  # noqa: E402

TRUE_SETTING = dict(grid=(32, 32), diffusivity=(0.02, 0.006), reaction=0.8,
                    forcing_scale=8, band_width=1.2, dt=0.05,
                    burn_in=256, record_steps=32)
NOMINAL = dict(source_scale=0.05, diffusivity=(1.0, 0.3), reaction=0.8, band_width=1.4)


def index_band_centre(angular_centre: float, steps: int, dt: float) -> float:
    """Convert an angular forcing frequency to the cosine-basis index it excites.

    The window is ``T = steps * dt`` and the cosine basis has ``omega_k = pi k / T``,
    so the excited index is ``omega_0 T / pi``.  Getting this wrong would point
    the prior at the wrong band, which is exactly the failure the wrong-band arm
    is meant to measure.
    """
    return angular_centre * (steps * dt) / np.pi


def narrowband_bank(max_frequency: int, atoms: int, band_index: float, nominal: dict):
    joint = operator_joint_spectrum(
        "narrowband_diffusion", torch.arange(max_frequency + 1, dtype=torch.float32),
        source_scale=nominal["source_scale"], reaction_diffusivity=nominal["diffusivity"],
        reaction=nominal["reaction"], forcing_band_centre=band_index,
        forcing_band_width=nominal["band_width"]).permute(2, 0, 1)
    separated = nonnegative_cp_spectrum(joint, rank=atoms, steps=1600, seed=17)
    return normalize_spectrum_cosine(torch.stack(separated.factors)), separated.relative_error


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--ratios", type=float, nargs="+", default=[0.01, 0.02, 0.05])
    parser.add_argument("--band-centres", type=float, nargs="+", default=[6.0, 10.0])
    parser.add_argument("--ranks", type=int, nargs=3, default=[8, 5, 5])
    parser.add_argument("--atoms", type=int, default=4)
    parser.add_argument("--max-frequency", type=int, default=8)
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--lr", type=float, default=0.02)
    parser.add_argument("--noise-std", type=float, default=0.05)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--tag", default="narrowband")
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "narrowband")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device); ranks = tuple(args.ranks)

    _, generic_raw = extended_generic_dictionary(args.atoms, args.max_frequency)
    generic = normalize_spectrum_cosine(generic_raw)[None].expand(3, -1, -1).clone().to(device)

    records = []
    for centre in args.band_centres:
        index = index_band_centre(centre, TRUE_SETTING["record_steps"], TRUE_SETTING["dt"])
        ops, separation = narrowband_bank(args.max_frequency, args.atoms, index, NOMINAL)
        # Same family, wrong band: isolates "knowing where the band is" from
        # "knowing the equation".
        wrong, _ = narrowband_bank(args.max_frequency, args.atoms, 0.5, NOMINAL)
        ops, wrong = ops.to(device), wrong.to(device)
        for ratio in args.ratios:
            rows = {"operator": [], "generic": [], "wrong_band": []}
            ceilings = []
            for seed in args.seeds:
                solved = solve_narrowband_forced(seed=seed, band_centre=centre, **TRUE_SETTING)
                ceilings.append(tucker_ceiling(solved.field, ranks))
                field, observed, targets, test, truth = make_task(
                    solved.field, ratio, seed, args.noise_std, device)
                for name, bank in (("operator", ops), ("generic", generic),
                                   ("wrong_band", wrong)):
                    rows[name].append(train(field, observed, targets, test, truth, bank,
                                            ranks=ranks, steps=args.steps, seed=seed,
                                            device=device, lr=args.lr,
                                            routing="global")["test_nrmse"])
            cell = {"band_centre": centre, "excited_index": float(index), "ratio": ratio,
                    "separation_relative_error": separation,
                    "tucker_ceiling": float(np.mean(ceilings))}
            for name, values in rows.items():
                values = np.array(values)
                cell[name] = {"mean": float(values.mean()), "std": float(values.std()),
                              "values": values.tolist()}
            margin = np.array(rows["generic"]) - np.array(rows["operator"])
            cell["margin_mean"] = float(margin.mean())
            cell["relative_percent"] = float(100 * margin.mean() / np.mean(rows["generic"]))
            cell["wins"] = int((margin > 0).sum())
            records.append(cell)
            print(f"  band {centre:4.1f} (index {index:4.1f})  ratio {ratio:5.3f}  "
                  f"ceil={cell['tucker_ceiling']:.3f}  ours {cell['operator']['mean']:.4f}  "
                  f"generic {cell['generic']['mean']:.4f}  wrong-band "
                  f"{cell['wrong_band']['mean']:.4f}  margin {cell['margin_mean']:+.4f} "
                  f"({cell['relative_percent']:+.1f}%) {cell['wins']}/{len(args.seeds)}",
                  flush=True)

    (args.output / f"{args.tag}_summary.json").write_text(json.dumps(
        {"true_setting": TRUE_SETTING, "nominal_prior": NOMINAL,
         "config": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
         "records": records}, indent=2))


if __name__ == "__main__":
    main()
