#!/usr/bin/env python3
"""How many sources must be fired before the operator prior helps?

The first fibre-mask run was a negative result for the method and also had a
defect: parameterising the mask by a ratio quantises badly on 24 sources, so 2%
and 5% both rounded to a single fired source and produced identical numbers
under two different labels.  Sources fired is the meaningful quantity here.

The negative result itself looks real and has a mechanism.  With very few
sources fired, the source-mode factors must be reconstructed from a handful of
columns.  The operator's eigenfunctions are oscillatory and spatially
structured, so extrapolating them from two columns commits confidently to a
shape that may be wrong, while a smooth generic kernel hedges.  A sharper prior
is a liability exactly where the mode is too under-sampled to identify which
basis functions are active.

This sweep locates the crossover instead of reporting a single point.
"""

from __future__ import annotations

import argparse, json, sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "experiments"))

from geoaware.operator_spectral_funbat import (  # noqa: E402
    extended_generic_dictionary, normalize_spectrum_cosine,
)
from green_response_data import (  # noqa: E402
    green_response_tensor, learner_bases, operator_spectra,
)
from run_green_response_main import cosine_bases, make_task, train  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--sources", type=int, nargs="+", default=[2, 4, 6, 9, 12, 18])
    parser.add_argument("--contrast", type=float, default=1.0)
    parser.add_argument("--learner-modes", type=int, default=8)
    parser.add_argument("--ranks", type=int, nargs=3, default=[4, 5, 5])
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--lr", type=float, default=0.02)
    parser.add_argument("--noise-std", type=float, default=0.05)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--tag", default="fiber_sweep")
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "green_response")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device); ranks = tuple(args.ranks)
    modes = args.learner_modes

    green = green_response_tensor(contrast=args.contrast, learner_modes=modes)
    bases = tuple(b.to(device) for b in learner_bases(green))
    spectra = operator_spectra(green)[:, None, :].to(device)
    _, generic_raw = extended_generic_dictionary(4, modes - 1)
    generic = normalize_spectrum_cosine(generic_raw)[None].expand(3, -1, -1).clone().to(device)
    cosine = cosine_bases(green.field.shape, modes, device)
    total_sources = green.field.shape[2]

    records = []
    for kept in args.sources:
        ours, base, fractions = [], [], []
        for seed in args.seeds:
            field, observed, targets, test, truth = make_task(
                green.field, None, seed, args.noise_std, device,
                mask="source_fibers", sources_kept=kept)
            fractions.append(len(observed) / green.field.numel())
            ours.append(train(field, observed, targets, test, truth, spectra, bases,
                              ranks=ranks, steps=args.steps, seed=seed,
                              device=device, lr=args.lr)["test_nrmse"])
            base.append(train(field, observed, targets, test, truth, generic, cosine,
                              ranks=ranks, steps=args.steps, seed=seed,
                              device=device, lr=args.lr)["test_nrmse"])
        ours, base = np.array(ours), np.array(base)
        margin = base - ours
        records.append({
            "sources_fired": kept, "of_total": total_sources,
            "observed_fraction": float(np.mean(fractions)),
            "ours": {"mean": float(ours.mean()), "std": float(ours.std())},
            "generic": {"mean": float(base.mean()), "std": float(base.std())},
            "margin_mean": float(margin.mean()),
            "relative_percent": float(100 * margin.mean() / base.mean()),
            "wins": int((margin > 0).sum()), "seeds": len(ours),
        })
        r = records[-1]
        print(f"  {kept:2d}/{total_sources} sources fired ({100*r['observed_fraction']:4.1f}% "
              f"of entries)  ours {r['ours']['mean']:.4f}  generic {r['generic']['mean']:.4f}  "
              f"margin {r['margin_mean']:+.4f} ({r['relative_percent']:+.1f}%)  "
              f"{r['wins']}/{r['seeds']}", flush=True)

    (args.output / f"{args.tag}_summary.json").write_text(json.dumps(
        {"question": "how many sources must be fired before the operator prior helps",
         "config": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
         "records": records}, indent=2))


if __name__ == "__main__":
    main()
