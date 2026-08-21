#!/usr/bin/env python3
"""The competitor our argument has to survive: a fixed kernel, chosen by judgment.

The paper argues that under a confined layout the kernel cannot be selected from
data, and that a construction needing no tuning data is therefore worth having.
The gap between those two sentences is a practitioner who does not tune *at all*
-- who reasons "this is extrapolation, so use a smooth prior", fixes a long
length scale by judgment, and collects no validation data either.  If one fixed
value is competitive everywhere, our contribution is nothing: the generic kernel
also needs no tuning data, and we chose a needlessly clumsy opponent.

So the test is not whether some fixed length scale beats us somewhere.  It is
whether *one* fixed value does well across every layout, since a practitioner
must commit to a value before seeing which layout they are in -- and if the value
has to change with the layout, choosing it is tuning by another name.

Both outcomes are reportable.  A single value that works everywhere refutes the
paper's claim.  A value that has to change with the geometry supports it.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np, torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "experiments"))
from geoaware.operator_spectral_funbat import real_cosine_basis  # noqa: E402
from forced_pde_solver import solve_multi_leak  # noqa: E402
import run_leak_sensors as base  # noqa: E402

CANDIDATES = (0.12, 0.32, 0.8, 1.6, 2.4, 3.5)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--layouts", nargs="+",
                   default=["random", "wall_ring", "near_wall", "one_wall_strip",
                            "corner_block"])
    p.add_argument("--ratio", type=float, default=0.01)
    p.add_argument("--noise-std", type=float, default=0.05)
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--lr", type=float, default=0.02)
    p.add_argument("--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="an idle GPU makes these sweeps roughly an order of magnitude cheaper")
    p.add_argument("--output", type=Path, default=ROOT / "results" / "leak")
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    device = torch.device(a.device)

    shape = tuple(solve_multi_leak(seed=0, **base.FIELD).field.shape)
    budget = int(round(a.ratio * int(np.prod(shape))))
    ours_spectra = base.operator_spectra(shape, base.FIELD["dt"], base.BINS)
    bases = tuple(real_cosine_basis(torch.arange(s, dtype=torch.float64) / s, b).float()
                  for s, b in zip(shape, base.BINS))

    table = {}
    for layout in a.layouts:
        rows = {f"fixed_{ls}": [] for ls in CANDIDATES}
        rows["ours_pde"] = []
        for seed in a.seeds:
            field = solve_multi_leak(seed=seed, **base.FIELD).field.to(device)
            observed, test = base.sensor_mask(shape, layout, budget, seed, device)
            g = torch.Generator(device=device).manual_seed(seed + 991)
            targets = field[tuple(observed.T)] + a.noise_std * torch.randn(
                len(observed), generator=g, device=device)
            truth = field[tuple(test.T)]

            def fit(spectra):
                return base.fit_gp(field, observed, targets, test, truth, spectra, bases,
                                   steps=a.steps, seed=seed, device=device, lr=a.lr)

            rows["ours_pde"].append(fit(ours_spectra))
            for ls in CANDIDATES:
                rows[f"fixed_{ls}"].append(fit(base.matern_spectra(base.BINS, ls)))
        table[layout] = {k: float(np.mean(v)) for k, v in rows.items()}
        best = min(CANDIDATES, key=lambda ls: table[layout][f"fixed_{ls}"])
        print(f"  {layout:16s} ours {table[layout]['ours_pde']:.4f}   "
              + "  ".join(f"l={ls}:{table[layout][f'fixed_{ls}']:.4f}" for ls in CANDIDATES)
              + f"   best fixed here: {best}", flush=True)

    # A practitioner commits to one value before knowing the layout.  Score each
    # candidate by its worst case and by its mean, both relative to ours.
    print("\n  one value, committed before seeing the layout:")
    verdict = {}
    for ls in CANDIDATES:
        losses = [table[l][f"fixed_{ls}"] - table[l]["ours_pde"] for l in a.layouts]
        verdict[ls] = {"worst_case_gap": float(max(losses)),
                       "mean_gap": float(np.mean(losses)),
                       "layouts_where_it_beats_ours":
                           [l for l, d in zip(a.layouts, losses) if d < 0]}
        print(f"    l={ls:<4}  worst-case gap vs ours {max(losses):+.4f}   "
              f"mean {np.mean(losses):+.4f}   beats us on "
              f"{len(verdict[ls]['layouts_where_it_beats_ours'])}/{len(a.layouts)} layouts")
    champion = min(CANDIDATES, key=lambda ls: verdict[ls]["worst_case_gap"])
    refuted = verdict[champion]["worst_case_gap"] <= 0
    print(f"\n  best single fixed value is l={champion}, worst case "
          f"{verdict[champion]['worst_case_gap']:+.4f} against ours")
    print(f"  claim refuted (one fixed kernel matches us everywhere): {refuted}")
    (a.output / "fixed_kernel_summary.json").write_text(json.dumps(
        {"candidates": CANDIDATES, "seeds": a.seeds, "per_layout": table,
         "commit_before_seeing_layout": {str(k): v for k, v in verdict.items()},
         "best_single_value": champion, "claim_refuted": refuted}, indent=2))


if __name__ == "__main__":
    main()
