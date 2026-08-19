#!/usr/bin/env python3
"""The sharpest form of the claim: the operator says *where* the energy sits.

The diffusion experiments showed a real but modest margin, and the ablation
showed why it was modest -- at low anisotropy any smooth low-pass prior does
about as well, so the operator's spectral *shape* was barely being tested.

A band-pass operator removes that escape route.  The linear Swift-Hohenberg
operator has its response minimised at a nonzero wavenumber, so the steady-state
field has a spectral peak away from the origin.  No monotone-decaying kernel can
express that, and a generic dictionary can only approximate it if it happens to
contain an oscillatory atom at the right frequency.  Meanwhile a small band
wavenumber keeps the pattern large-scale, so the field stays low multilinear
rank and the completion task stays feasible.

Both banks are compared at their own best routing, so the margin cannot come
from handicapping the baseline.
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
from forced_pde_solver import solve_forced_spectral  # noqa: E402
from run_forced_pde_main import make_task, nrmse, tucker_ceiling  # noqa: E402
from run_forced_pde_ablation import train  # noqa: E402

TRUE_SETTING = dict(operator="banded_pattern", grid=(32, 32), band_wavenumber=2.5,
                    band_stiffness=1.0e-3, band_offset=0.2, forcing_scale=8,
                    dt=1.0, burn_in=200, record_steps=32)
# The prior knows the form and the band, with a nominal stiffness and offset that
# are not the generating ones.
NOMINAL = dict(source_scale=0.02, band_wavenumber=2.5, band_stiffness=0.6, band_offset=0.12)


def banded_prior(max_frequency: int, atoms: int, nominal: dict) -> tuple[torch.Tensor, float]:
    joint = operator_joint_spectrum(
        "banded_pattern", torch.arange(max_frequency + 1, dtype=torch.float32),
        source_scale=nominal["source_scale"], band_wavenumber=nominal["band_wavenumber"],
        band_stiffness=nominal["band_stiffness"], band_offset=nominal["band_offset"],
    ).permute(2, 0, 1)
    separated = nonnegative_cp_spectrum(joint, rank=atoms, steps=1600, seed=17)
    return normalize_spectrum_cosine(torch.stack(separated.factors)), separated.relative_error


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--ratios", type=float, nargs="+", default=[0.01, 0.02, 0.05])
    parser.add_argument("--ranks", type=int, nargs=3, default=[8, 5, 5])
    parser.add_argument("--atoms", type=int, default=4)
    parser.add_argument("--max-frequency", type=int, default=8)
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--lr", type=float, default=0.02)
    parser.add_argument("--noise-std", type=float, default=0.05)
    parser.add_argument("--wrong-band", type=float, default=6.0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--tag", default="bandpass")
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "forced_pde")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device); ranks = tuple(args.ranks)

    prior, separation_error = banded_prior(args.max_frequency, args.atoms, NOMINAL)
    # Same family, wrong band: isolates "knowing where the energy is" from
    # "knowing the functional form".
    wrong_band, _ = banded_prior(args.max_frequency, args.atoms,
                                 dict(NOMINAL, band_wavenumber=args.wrong_band))
    _, generic_raw = extended_generic_dictionary(args.atoms, args.max_frequency)
    generic = normalize_spectrum_cosine(generic_raw)[None].expand(3, -1, -1).clone()
    _, generic_big_raw = extended_generic_dictionary(3 * args.atoms, args.max_frequency)
    generic_big = normalize_spectrum_cosine(generic_big_raw)[None].expand(3, -1, -1).clone()
    print(f"operator separation relative error {separation_error:.4f}", flush=True)

    arms = {"operator_pmr": (prior, "per_mode_rank"), "operator_global": (prior, "global"),
            "wrong_band_pmr": (wrong_band, "per_mode_rank"),
            "wrong_band_global": (wrong_band, "global"),
            "generic_pmr": (generic, "per_mode_rank"), "generic_global": (generic, "global"),
            "generic3x_pmr": (generic_big, "per_mode_rank"),
            "generic3x_global": (generic_big, "global")}

    summary = {"true_setting": TRUE_SETTING, "nominal_prior": NOMINAL,
               "operator_separation_relative_error": separation_error, "ratios": {}}
    for ratio in args.ratios:
        rows = {name: [] for name in arms}
        ceilings = []
        for seed in args.seeds:
            solved = solve_forced_spectral(seed=seed, **TRUE_SETTING)
            ceilings.append(tucker_ceiling(solved.field, ranks))
            field, observed, targets, test, truth = make_task(
                solved.field, ratio, seed, args.noise_std, device)
            for name, (bank, routing) in arms.items():
                rows[name].append(train(field, observed, targets, test, truth, bank,
                                        ranks=ranks, steps=args.steps, seed=seed,
                                        device=device, lr=args.lr, routing=routing)["test_nrmse"])
        block = {n: {"mean": float(np.mean(v)), "std": float(np.std(v)), "values": v}
                 for n, v in rows.items()}
        best_op = min(("operator_pmr", "operator_global"), key=lambda n: block[n]["mean"])
        best_gen = min([n for n in arms if n.startswith("generic")],
                       key=lambda n: block[n]["mean"])
        paired = np.array(block[best_op]["values"]) - np.array(block[best_gen]["values"])
        block.update({"best_operator_arm": best_op, "best_generic_arm": best_gen,
                      "margin_best_vs_best": float(-paired.mean()),
                      "wins_best_vs_best": int((paired < 0).sum()),
                      "tucker_ceiling_mean": float(np.mean(ceilings))})
        summary["ratios"][str(ratio)] = block
        print(f"ratio {ratio:5.3f} ceil={np.mean(ceilings):.3f}  " + "  ".join(
            f"{n}={block[n]['mean']:.4f}" for n in arms) +
            f"  BEST {best_op} vs {best_gen} margin {block['margin_best_vs_best']:+.4f} "
            f"({block['wins_best_vs_best']}/{len(args.seeds)})", flush=True)
    (args.output / f"{args.tag}_summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
