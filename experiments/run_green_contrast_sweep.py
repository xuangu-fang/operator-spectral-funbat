#!/usr/bin/env python3
"""Does the margin grow with how non-stationary the medium is?

First-principles prediction.  A generic stationary kernel assumes translation
invariance.  The eigenfunctions of a variable-coefficient operator are not
translation invariant -- they stretch where the medium is soft and compress
where it is stiff -- so a stationary kernel is structurally wrong there and no
length scale fixes it.  The operator's own eigenbasis carries exactly that
information.

Therefore the margin should vanish at `contrast = 0`, where the medium is
uniform, the eigenfunctions degenerate to cosines and a generic kernel is
adequate; and it should grow with contrast.  A flat curve refutes the account.

This is the non-stationarity analogue of the isotropy negative control, and it
tests a channel a generic kernel cannot partially cover, unlike anisotropy.
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
    parser.add_argument("--contrasts", type=float, nargs="+",
                        default=[0.0, 0.25, 0.5, 1.0, 1.5, 2.0])
    parser.add_argument("--ratio", type=float, default=0.02)
    parser.add_argument("--mask", default="random")
    parser.add_argument("--learner-modes", type=int, default=8)
    parser.add_argument("--ranks", type=int, nargs=3, default=[4, 5, 5])
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--lr", type=float, default=0.02)
    parser.add_argument("--noise-std", type=float, default=0.05)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--tag", default="contrast_sweep")
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "green_response")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device); ranks = tuple(args.ranks)
    modes = args.learner_modes

    _, generic_raw = extended_generic_dictionary(4, modes - 1)
    generic = normalize_spectrum_cosine(generic_raw)[None].expand(3, -1, -1).clone().to(device)

    records = []
    for contrast in args.contrasts:
        green = green_response_tensor(contrast=contrast, learner_modes=modes)
        bases = tuple(b.to(device) for b in learner_bases(green))
        spectra = operator_spectra(green)[:, None, :].to(device)
        cosine = cosine_bases(green.field.shape, modes, device)
        # Three arms, so that "operator basis" and "operator spectrum" can be
        # separated.  The first sweep showed a 15% margin even at contrast 0,
        # where the medium is uniform and there is no non-stationarity to
        # exploit, so the advantage cannot be attributed to the operator
        # without this decomposition.
        ours, base, basis_only = [], [], []
        for seed in args.seeds:
            field, observed, targets, test, truth = make_task(
                green.field, args.ratio, seed, args.noise_std, device, mask=args.mask)
            ours.append(train(field, observed, targets, test, truth, spectra, bases,
                              ranks=ranks, steps=args.steps, seed=seed,
                              device=device, lr=args.lr)["test_nrmse"])
            base.append(train(field, observed, targets, test, truth, generic, cosine,
                              ranks=ranks, steps=args.steps, seed=seed,
                              device=device, lr=args.lr)["test_nrmse"])
            basis_only.append(train(field, observed, targets, test, truth, generic, bases,
                                    ranks=ranks, steps=args.steps, seed=seed,
                                    device=device, lr=args.lr)["test_nrmse"])
        ours, base, basis_only = np.array(ours), np.array(base), np.array(basis_only)
        margin = base - ours
        records.append({
            "contrast": contrast,
            "diffusivity_ratio": (green.metadata["diffusivity_max"]
                                  / green.metadata["diffusivity_min"]),
            "ours": {"mean": float(ours.mean()), "std": float(ours.std())},
            "generic": {"mean": float(base.mean()), "std": float(base.std())},
            "operator_basis_generic_spectrum": {"mean": float(basis_only.mean()),
                                                "std": float(basis_only.std())},
            "basis_contribution": float(base.mean() - basis_only.mean()),
            "spectrum_contribution": float(basis_only.mean() - ours.mean()),
            "margin_mean": float(margin.mean()), "margin_std": float(margin.std()),
            "relative_percent": float(100 * margin.mean() / base.mean()),
            "wins": int((margin > 0).sum()), "seeds": len(ours),
        })
        r = records[-1]
        print(f"  contrast {contrast:4.2f} (a_max/a_min {r['diffusivity_ratio']:6.1f})  "
              f"ours {r['ours']['mean']:.4f}  generic {r['generic']['mean']:.4f}  "
              f"basis-only {r['operator_basis_generic_spectrum']['mean']:.4f}  "
              f"margin {r['margin_mean']:+.4f} ({r['relative_percent']:+.1f}%)  "
              f"[basis {r['basis_contribution']:+.4f} | spectrum "
              f"{r['spectrum_contribution']:+.4f}]  {r['wins']}/{r['seeds']}", flush=True)

    (args.output / f"{args.tag}_summary.json").write_text(json.dumps(
        {"prediction": "margin vanishes at contrast 0 and grows with contrast; "
                       "a flat curve refutes the non-stationarity account",
         "config": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
         "records": records}, indent=2))


if __name__ == "__main__":
    main()
