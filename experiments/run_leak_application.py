#!/usr/bin/env python3
"""Application study: find the leak from wall-mounted sensors.

Held-out NRMSE is the metric the tensor literature reports, but it is not what
anyone deploying this asks.  A gas-leak operator asks where the leak is, and
whether they should send someone.  This measures that directly.

Protocol.  The dominant source is placed at a different, randomly drawn position
for every seed, so a method cannot do well by learning one fixed answer; nothing
about the position is available to any arm.  Each arm reconstructs the field
from the sensors it is allowed, takes the time-averaged reconstruction, and its
answer is the location of the peak.

Three numbers are reported, in decreasing order of how much they hide:

  localisation error   distance from the true source, in room widths.  A
                       practitioner can read this without knowing the model:
                       0.05 means "within five percent of the room".
  success rate         the fraction of runs landing within a tenth of a room
                       width, which is roughly "the right corner of the room".
  held-out NRMSE       reported alongside so the two metrics can be compared,
                       since a method can lower NRMSE without ever finding the
                       leak, and that dissociation is the point of the study.
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

SUCCESS_RADIUS = 0.10


def field_for_seed(seed: int, *, drift=(0.0, 0.0), reaction=None):
    """One dominant leak at a random position, plus two weaker ones."""
    rng = np.random.default_rng(seed + 20250821)
    primary = (float(rng.uniform(0.25, 0.75)), float(rng.uniform(0.25, 0.75)))
    others = [(float(rng.uniform(0.15, 0.85)), float(rng.uniform(0.15, 0.85)))
              for _ in range(2)]
    # The dominant leak is made clearly dominant.  With three sources of similar
    # strength the peak of the time-averaged field is not reliably the one we
    # are scoring against, and the metric would be measuring which blob happened
    # to win rather than whether the method found the leak.
    sources = ((primary[0], primary[1], 0.13, 15.0),
               (others[0][0], others[0][1], 0.05, 25.0),
               (others[1][0], others[1][1], 0.045, 40.0))
    settings = dict(base.FIELD, sources=sources)
    if reaction is not None:
        settings["reaction"] = reaction
    if drift[0] or drift[1]:
        settings["drift"] = drift
    return solve_multi_leak(seed=seed, **settings), primary


def peak_location(values: torch.Tensor, shape) -> tuple[float, float]:
    """Where the time-averaged field is largest, in [0,1]^2."""
    mean_map = values.mean(0)
    flat = int(mean_map.argmax())
    return ((flat // shape[2] + 0.5) / shape[1], (flat % shape[2] + 0.5) / shape[2])


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="+", default=list(range(8)))
    p.add_argument("--layouts", nargs="+",
                   default=["one_wall_strip", "wall_ring", "random"])
    p.add_argument("--ratio", type=float, default=0.01)
    p.add_argument("--noise-std", type=float, default=0.05)
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--lr", type=float, default=0.02)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--tag", default="application")
    p.add_argument("--output", type=Path, default=ROOT / "results" / "leak")
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    device = torch.device(a.device)

    probe, _ = field_for_seed(0)
    shape = tuple(probe.field.shape)
    budget = int(round(a.ratio * int(np.prod(shape))))
    ours = base.operator_spectra(shape, base.FIELD["dt"], base.BINS)
    bases = tuple(real_cosine_basis(torch.arange(s, dtype=torch.float64) / s, b).float()
                  for s, b in zip(shape, base.BINS))
    print(f"  {len(a.seeds)} seeds, source position redrawn each seed, "
          f"{budget} sensors ({100 * a.ratio:.0f}% of the room)", flush=True)

    records = []
    for layout in a.layouts:
        rows: dict[str, dict[str, list]] = {}
        for seed in a.seeds:
            solved, truth_xy = field_for_seed(seed)
            field = solved.field.to(device)
            observed, test = base.sensor_mask(shape, layout, budget, seed, device)
            g = torch.Generator(device=device).manual_seed(seed + 991)
            targets = field[tuple(observed.T)] + a.noise_std * torch.randn(
                len(observed), generator=g, device=device)
            truth = field[tuple(test.T)]

            def reconstruct(spectra):
                """Full field from one arm, so its peak can be located."""
                torch.manual_seed(seed + 10_000)
                from geoaware.operator_spectral_funbat import ModeAdaptiveVariationalTucker
                model = ModeAdaptiveVariationalTucker(
                    tuple(torch.arange(s, device=device) / s for s in shape),
                    [s.to(device) for s in spectra], ranks=base.RANKS, routing="global",
                    noise_std=0.08, basis=("operator",) * 3,
                    eigenbasis=tuple(b.to(device) for b in bases)).to(device)
                optimiser = torch.optim.Adam(model.parameters(), lr=a.lr)
                for _ in range(a.steps):
                    optimiser.zero_grad(set_to_none=True)
                    loss, _ = model.negative_elbo(observed, targets,
                                                  total_count=len(targets), samples=3)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
                    optimiser.step()
                with torch.no_grad():
                    grid = torch.stack(torch.meshgrid(
                        *[torch.arange(s, device=device) for s in shape],
                        indexing="ij"), -1).reshape(-1, 3)
                    whole = torch.cat([model.posterior_mean(grid[i:i + 200_000])
                                       for i in range(0, len(grid), 200_000)])
                    whole = whole.reshape(shape)
                    error = float(torch.sqrt(torch.mean(
                        (model.posterior_mean(test) - truth).square()))
                        / truth.std().clamp_min(1e-8))
                return whole, error

            # The floor of the metric.  Diffusion is anisotropic, so even the
            # true field's peak sits a little away from the source; without this
            # row a reader cannot tell a good localisation from a lucky one.
            perfect = peak_location(field, shape)
            rows.setdefault("truth_itself", {"distance": [], "nrmse": []})
            rows["truth_itself"]["distance"].append(
                float(np.hypot(perfect[0] - truth_xy[0], perfect[1] - truth_xy[1])))
            rows["truth_itself"]["nrmse"].append(0.0)

            arms = {"ours_pde": ours}
            # Deployable Matern: length scale from a split of the sensor readings.
            split = torch.randperm(len(observed), generator=torch.Generator(
                device=device).manual_seed(seed + 4242), device=device)
            cut = int(len(observed) * 0.75)
            validation = {}
            for ls in base.LENGTH_SCALES:
                validation[ls] = base.fit_gp(
                    field, observed[split[:cut]], targets[split[:cut]],
                    observed[split[cut:]], targets[split[cut:]],
                    base.matern_spectra(base.BINS, ls), bases, steps=a.steps,
                    seed=seed, device=device, lr=a.lr)
            arms["matern_deployable"] = base.matern_spectra(
                base.BINS, min(validation, key=validation.get))
            arms["matern_fixed"] = base.matern_spectra(base.BINS, 1.6)

            for name, spectra in arms.items():
                whole, error = reconstruct(spectra)
                guess = peak_location(whole, shape)
                distance = float(np.hypot(guess[0] - truth_xy[0], guess[1] - truth_xy[1]))
                bucket = rows.setdefault(name, {"distance": [], "nrmse": []})
                bucket["distance"].append(distance)
                bucket["nrmse"].append(error)

        cell = {"layout": layout, "sensors": budget, "seeds": len(a.seeds)}
        for name, bucket in rows.items():
            distances = np.array(bucket["distance"])
            cell[name] = {
                "median_distance": float(np.median(distances)),
                "mean_distance": float(distances.mean()),
                "success_rate": float((distances < SUCCESS_RADIUS).mean()),
                "nrmse": float(np.mean(bucket["nrmse"])),
                "distances": distances.tolist()}
        records.append(cell)
        print(f"  {layout}", flush=True)
        for name in rows:
            c = cell[name]
            print(f"    {name:20s} median {c['median_distance']:.3f} room widths   "
                  f"within {SUCCESS_RADIUS:.0%}: {c['success_rate']:.0%}   "
                  f"NRMSE {c['nrmse']:.4f}", flush=True)

    (a.output / f"{a.tag}_summary.json").write_text(json.dumps(
        {"success_radius": SUCCESS_RADIUS, "seeds": a.seeds, "ratio": a.ratio,
         "records": records}, indent=2))


if __name__ == "__main__":
    main()
