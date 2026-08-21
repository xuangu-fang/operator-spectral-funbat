#!/usr/bin/env python3
"""AutoIP-style physics-informed GP against ours, on accuracy and on cost.

This is the closest relative of the method: physics enters as a prior rather
than as a loss.  It is also the one where the cost matters as much as the
error, because it keeps a standard GP over the field and factorises a dense
matrix over observations plus collocation points, while we never form one.  So
the run records wall-clock and peak memory alongside NRMSE, and sweeps the
observation count so the scaling is measured rather than argued.

The GP is given the same physics we are -- the operator's form with nominal
coefficients wrong by 50% -- and, as with every strong baseline here, its
hyper-parameters are chosen against the held-out region, which is an oracle.
"""
from __future__ import annotations
import argparse, json, sys, time, tracemalloc
from pathlib import Path
import numpy as np, torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "experiments"))
from geoaware.operator_spectral_funbat import real_cosine_basis  # noqa: E402
from forced_pde_solver import solve_multi_leak  # noqa: E402
from physics_informed_gp import PhysicsInformedGP, verify_against_autograd  # noqa: E402
import run_leak_sensors as base  # noqa: E402

SCALES = ((0.30, 0.20, 0.20), (0.60, 0.35, 0.35), (1.00, 0.60, 0.60))
RESIDUAL_STD = (0.02, 0.2)


def coordinates(indices, shape):
    return (indices.double() + 0.5) / torch.tensor(shape, dtype=torch.float64)


def run_once(shape, observed, targets, collocation, test, truth, *, scales,
             residual_std, dt):
    model = PhysicsInformedGP(length_scales=scales,
                              diffusivity=base.NOMINAL["diffusivity"],
                              reaction=base.NOMINAL["reaction"],
                              time_span=shape[0] * dt, residual_std=residual_std)
    started = time.time(); tracemalloc.start()
    prediction = model.fit_predict(coordinates(observed, shape), targets.double(),
                                   collocation, coordinates(test, shape), chunk=5000)
    peak = tracemalloc.get_traced_memory()[1]; tracemalloc.stop()
    error = float(torch.sqrt(torch.mean((prediction - truth.double()).square()))
                  / truth.double().std().clamp_min(1e-8))
    return error, time.time() - started, peak / 2 ** 20


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--layouts", nargs="+", default=["random", "one_wall_strip"])
    p.add_argument("--ratio", type=float, default=0.01)
    p.add_argument("--noise-std", type=float, default=0.05)
    p.add_argument("--collocation", type=int, default=1500)
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--lr", type=float, default=0.02)
    p.add_argument("--scaling-ratios", type=float, nargs="+",
                   default=[0.005, 0.01, 0.02, 0.04])
    p.add_argument("--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="an idle GPU makes these sweeps roughly an order of magnitude cheaper")
    p.add_argument("--output", type=Path, default=ROOT / "results" / "leak")
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    device = torch.device(a.device)

    check = verify_against_autograd()
    print(f"  kernel derivatives verified against autograd: {check}", flush=True)
    if not check["passes"]:
        raise SystemExit("physics-informed GP kernel derivatives are wrong")

    shape = tuple(solve_multi_leak(seed=0, **base.FIELD).field.shape)
    ours_spectra = base.operator_spectra(shape, base.FIELD["dt"], base.BINS)
    bases = tuple(real_cosine_basis(torch.arange(s, dtype=torch.float64) / s, b).float()
                  for s, b in zip(shape, base.BINS))

    records = []
    for layout in a.layouts:
        budget = int(round(a.ratio * int(np.prod(shape))))
        rows: dict[str, list] = {}
        costs: list = []
        for seed in a.seeds:
            field = solve_multi_leak(seed=seed, **base.FIELD).field.to(device)
            observed, test = base.sensor_mask(shape, layout, budget, seed, device)
            g = torch.Generator(device=device).manual_seed(seed + 991)
            targets = field[tuple(observed.T)] + a.noise_std * torch.randn(
                len(observed), generator=g, device=device)
            truth = field[tuple(test.T)]
            collocation = torch.rand(a.collocation, 3, generator=torch.Generator(
                device=device).manual_seed(seed + 77), dtype=torch.float64, device=device)

            rows.setdefault("ours_pde", []).append(
                base.fit_gp(field, observed, targets, test, truth, ours_spectra, bases,
                            steps=a.steps, seed=seed, device=device, lr=a.lr))

            scored = {}
            for scales in SCALES:
                for residual in RESIDUAL_STD:
                    error, seconds, megabytes = run_once(
                        shape, observed, targets, collocation, test, truth,
                        scales=scales, residual_std=residual, dt=base.FIELD["dt"])
                    scored[(scales, residual)] = (error, seconds, megabytes)
            best = min(scored, key=lambda key: scored[key][0])
            rows.setdefault("autoip_oracle", []).append(scored[best][0])
            costs.append({"seconds": scored[best][1], "peak_mb": scored[best][2],
                          "scales": list(best[0]), "residual_std": best[1]})

            # The same GP with the physics removed, so the equation's own
            # contribution is measured rather than inferred.
            empty = torch.zeros(0, 3, dtype=torch.float64, device=device)
            plain = min(run_once(shape, observed, targets, empty, test, truth,
                                 scales=scales, residual_std=residual,
                                 dt=base.FIELD["dt"])[0] for scales in SCALES
                        for residual in RESIDUAL_STD[:1])
            rows.setdefault("gp_no_physics", []).append(plain)

        cell = {"layout": layout, "observed": budget, "collocation": a.collocation,
                "cost": costs}
        for key, values in rows.items():
            values = np.array(values)
            cell[key] = {"mean": float(values.mean()), "std": float(values.std()),
                         "values": values.tolist()}
        cell["physics_buys"] = cell["gp_no_physics"]["mean"] - cell["autoip_oracle"]["mean"]
        cell["ours_vs_autoip"] = cell["autoip_oracle"]["mean"] - cell["ours_pde"]["mean"]
        records.append(cell)
        print(f"  {layout:16s} ours {cell['ours_pde']['mean']:.4f}   "
              f"AutoIP-style* {cell['autoip_oracle']['mean']:.4f}   "
              f"same GP, no physics {cell['gp_no_physics']['mean']:.4f}   "
              f"physics buys {cell['physics_buys']:+.4f}   "
              f"ours vs AutoIP {cell['ours_vs_autoip']:+.4f}", flush=True)
        print(f"  {'':16s} cost {np.mean([c['seconds'] for c in costs]):.1f}s, "
              f"{np.mean([c['peak_mb'] for c in costs]):.0f} MB peak", flush=True)

    # How the cost grows with the observation count, measured not argued.
    print("\n  cost against observation count (one wall, seed 0):", flush=True)
    scaling = []
    field = solve_multi_leak(seed=0, **base.FIELD).field.to(device)
    for ratio in a.scaling_ratios:
        budget = int(round(ratio * int(np.prod(shape))))
        try:
            observed, test = base.sensor_mask(shape, "one_wall_strip", budget, 0, device)
        except ValueError as error:
            print(f"    ratio {ratio}: {error}", flush=True); continue
        g = torch.Generator(device=device).manual_seed(991)
        targets = field[tuple(observed.T)] + a.noise_std * torch.randn(
            len(observed), generator=g, device=device)
        truth = field[tuple(test.T)]
        collocation = torch.rand(a.collocation, 3, dtype=torch.float64, device=device)
        error, seconds, megabytes = run_once(
            shape, observed, targets, collocation, test, truth,
            scales=SCALES[1], residual_std=RESIDUAL_STD[0], dt=base.FIELD["dt"])
        started = time.time()
        base.fit_gp(field, observed, targets, test, truth, ours_spectra, bases,
                    steps=a.steps, seed=0, device=device, lr=a.lr)
        ours_seconds = time.time() - started
        scaling.append({"ratio": ratio, "observed": budget, "autoip_seconds": seconds,
                        "autoip_peak_mb": megabytes, "ours_seconds": ours_seconds})
        print(f"    n={budget:6d}   AutoIP {seconds:7.1f}s {megabytes:7.0f} MB   "
              f"ours {ours_seconds:6.1f}s", flush=True)

    (a.output / "autoip_summary.json").write_text(json.dumps(
        {"verification": check, "scales": [list(s) for s in SCALES],
         "residual_std": RESIDUAL_STD, "seeds": a.seeds, "records": records,
         "scaling": scaling}, indent=2))


if __name__ == "__main__":
    main()
