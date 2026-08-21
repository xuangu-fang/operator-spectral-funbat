#!/usr/bin/env python3
"""Why one wall works and the other does not, and a test that could refute it.

The paper reported an unexplained asymmetry: the same strip of sensors on the
x wall ties the oracle and on the y wall loses by 0.42.  An isotropic control was
run and did not remove it, so anisotropy was ruled out and the cause was left
open.

The reconstruction figure shows what the arms actually recover from one wall: a
band that varies along one axis and is nearly constant along the other -- a
one-dimensional marginal profile, not a two-dimensional field.  That suggests the
mechanism.  A strip at the x wall spans the whole y axis, so the component of the
field that varies only with y is fully observed and needs no extrapolation at
all.  A strip at the y wall spans x instead, and so pins down the x profile.

Which wall is better therefore depends on which profile carries the field's
variance, and in this field the y profile carries 0.64 against 0.04 for x.  It
carries 0.39 against 0.14 even when the operator is made isotropic, which is why
the isotropic control failed to remove the asymmetry: the asymmetry was never
about the diffusivities, it is about where the sources sit.

The refutable prediction: swapping the diffusivities moves the dominant profile
to x (0.35 against 0.18), so the advantage must move to the y wall.  If it does
not, this explanation is wrong too.
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

CONDITIONS = {
    "as published (Dx=0.02, Dy=0.006)": (0.02, 0.006),
    "isotropic (Dx=Dy=0.013)": (0.013, 0.013),
    "swapped (Dx=0.006, Dy=0.02)": (0.006, 0.02),
}
WALLS = ("one_wall_strip", "one_wall_strip_y")


def profile_share(field):
    """Fraction of the field's variance carried by each axis-marginal profile."""
    centred = field - field.mean()
    total = float((centred ** 2).mean())
    return (float((centred.mean(dim=(0, 2)) ** 2).mean()) / total,
            float((centred.mean(dim=(0, 1)) ** 2).mean()) / total)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--ratio", type=float, default=0.01)
    p.add_argument("--noise-std", type=float, default=0.05)
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--lr", type=float, default=0.02)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--output", type=Path, default=ROOT / "results" / "leak")
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    device = torch.device(a.device)

    records = []
    for label, diffusivity in CONDITIONS.items():
        settings = dict(base.FIELD, diffusivity=diffusivity)
        shape = tuple(solve_multi_leak(seed=0, **settings).field.shape)
        budget = int(round(a.ratio * int(np.prod(shape))))
        bases = tuple(real_cosine_basis(torch.arange(s, dtype=torch.float64) / s, b).float()
                      for s, b in zip(shape, base.BINS))
        saved = dict(base.NOMINAL)
        base.NOMINAL.update(diffusivity=tuple(1.5 * d for d in diffusivity))
        ours = base.operator_spectra(shape, settings["dt"], base.BINS)
        base.NOMINAL.clear(); base.NOMINAL.update(saved)

        share_x, share_y = profile_share(solve_multi_leak(seed=0, **settings).field)
        entry = {"condition": label, "profile_share_x": share_x,
                 "profile_share_y": share_y, "walls": {}}
        print(f"  {label}: the x profile carries {share_x:.3f}, the y profile "
              f"{share_y:.3f}", flush=True)

        for wall in WALLS:
            scores, oracles = [], []
            for seed in a.seeds:
                field = solve_multi_leak(seed=seed, **settings).field.to(device)
                observed, test = base.sensor_mask(shape, wall, budget, seed, device)
                g = torch.Generator(device=device).manual_seed(seed + 991)
                targets = field[tuple(observed.T)] + a.noise_std * torch.randn(
                    len(observed), generator=g, device=device)
                truth = field[tuple(test.T)]
                fit = lambda sp: base.fit_gp(field, observed, targets, test, truth, sp,
                                             bases, steps=a.steps, seed=seed,
                                             device=device, lr=a.lr)
                scores.append(fit(ours))
                oracles.append(min(fit(base.matern_spectra(base.BINS, ls))
                                   for ls in base.LENGTH_SCALES))
            entry["walls"][wall] = {"ours": float(np.mean(scores)),
                                    "oracle": float(np.mean(oracles)),
                                    "ours_values": scores, "oracle_values": oracles}
            print(f"    {wall:18s} ours {np.mean(scores):.4f}   oracle "
                  f"{np.mean(oracles):.4f}", flush=True)

        # The prediction: the wall that spans the dominant profile's axis wins.
        spans_dominant = ("one_wall_strip" if share_y > share_x else "one_wall_strip_y")
        other = [w for w in WALLS if w != spans_dominant][0]
        entry["predicted_better_wall"] = spans_dominant
        entry["actually_better_wall"] = min(
            WALLS, key=lambda w: entry["walls"][w]["ours"])
        entry["prediction_holds"] = entry["predicted_better_wall"] == entry["actually_better_wall"]
        entry["gap"] = entry["walls"][other]["ours"] - entry["walls"][spans_dominant]["ours"]
        print(f"    predicted better wall: {spans_dominant}; actual: "
              f"{entry['actually_better_wall']}; holds: {entry['prediction_holds']} "
              f"(gap {entry['gap']:+.4f})", flush=True)
        records.append(entry)

    passed = sum(r["prediction_holds"] for r in records)
    print(f"\n  the profile-coverage account holds in {passed}/{len(records)} conditions",
          flush=True)
    (a.output / "profile_mechanism_summary.json").write_text(json.dumps(
        {"seeds": a.seeds, "records": records,
         "conditions_passed": passed}, indent=2))


if __name__ == "__main__":
    main()
