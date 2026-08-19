#!/usr/bin/env python3
"""Non-GP baselines on the forced-PDE completion task.

These answer the questions a reviewer asks before believing any kernel story:
is the model learning anything at all, is the gain just local smoothing, and are
continuous function factors necessary rather than a discrete low-rank fit?

All baselines see exactly the same field, mask, noisy observations and held-out
set as the main experiment.
"""

from __future__ import annotations

import argparse, json, sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "experiments"))

from forced_pde_solver import solve_forced  # noqa: E402
from run_forced_pde_main import TRUE_SETTING, make_task, nrmse  # noqa: E402


def _dense(observed, targets, shape, device):
    values = torch.zeros(shape, device=device)
    mask = torch.zeros(shape, device=device)
    values[tuple(observed.T)] = targets
    mask[tuple(observed.T)] = 1.0
    return values, mask


def em_cp_completion(observed, targets, shape, *, rank, device, sweeps=60, inner=6):
    """CP-ALS inside an EM loop: impute, refit, repeat.  The standard
    discrete-low-rank competitor for entry-wise completion."""
    values, mask = _dense(observed, targets, shape, device)
    filled = values + (1 - mask) * targets.mean()
    generator = torch.Generator(device=device).manual_seed(0)
    factors = [torch.randn(s, rank, generator=generator, device=device) for s in shape]
    for _ in range(sweeps):
        for _ in range(inner):
            for mode in range(3):
                others = [i for i in range(3) if i != mode]
                khatri = (factors[others[0]][:, None, :]
                          * factors[others[1]][None, :, :]).reshape(-1, rank)
                unfold = filled.movedim(mode, 0).reshape(shape[mode], -1)
                gram = (factors[others[0]].T @ factors[others[0]]) * \
                       (factors[others[1]].T @ factors[others[1]])
                gram = gram + 1e-6 * torch.eye(rank, device=device)
                factors[mode] = torch.linalg.solve(gram, (unfold @ khatri).T).T
        reconstruction = torch.einsum("ir,jr,kr->ijk", *factors)
        filled = mask * values + (1 - mask) * reconstruction
    return torch.einsum("ir,jr,kr->ijk", *factors)


def em_tucker_completion(observed, targets, shape, *, ranks, device, sweeps=60):
    """HOSVD inside the same EM loop, at the multilinear ranks the GP host uses."""
    values, mask = _dense(observed, targets, shape, device)
    filled = values + (1 - mask) * targets.mean()
    reconstruction = filled
    for _ in range(sweeps):
        bases = []
        for mode in range(3):
            unfold = filled.movedim(mode, 0).reshape(shape[mode], -1)
            u, _, _ = torch.linalg.svd(unfold, full_matrices=False)
            bases.append(u[:, :ranks[mode]])
        core = filled.clone()
        for mode in range(3):
            core = torch.tensordot(core, bases[mode], dims=([0], [0]))
        reconstruction = core
        for mode in range(3):
            reconstruction = torch.tensordot(reconstruction, bases[mode].T, dims=([0], [0]))
        filled = mask * values + (1 - mask) * reconstruction
    return reconstruction


def rbf_interpolation(observed, targets, test, *, device, length_scale, ridge=1e-3):
    """Kernel ridge regression on the index grid: is the gain just smoothing?"""
    obs = observed.float(); tst = test.float()
    gram = torch.exp(-torch.cdist(obs, obs).square() / (2 * length_scale**2))
    weights = torch.linalg.solve(
        gram + ridge * torch.eye(len(obs), device=device), targets)
    out = []
    for start in range(0, len(tst), 4096):
        block = tst[start:start + 4096]
        cross = torch.exp(-torch.cdist(block, obs).square() / (2 * length_scale**2))
        out.append(cross @ weights)
    return torch.cat(out)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--ratios", type=float, nargs="+", default=[0.01, 0.02, 0.05])
    parser.add_argument("--ranks", type=int, nargs=3, default=[8, 5, 5])
    parser.add_argument("--cp-ranks", type=int, nargs="+", default=[5, 10, 20])
    parser.add_argument("--noise-std", type=float, default=0.05)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "forced_pde")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device); ranks = tuple(args.ranks)

    records = []
    for seed in args.seeds:
        solved = solve_forced(seed=seed, **TRUE_SETTING)
        for ratio in args.ratios:
            field, observed, targets, test, truth = make_task(
                solved.field, ratio, seed, args.noise_std, device)
            row = {"seed": seed, "ratio": ratio}
            row["global_mean"] = nrmse(torch.full_like(truth, float(targets.mean())), truth)
            for rank in args.cp_ranks:
                rec = em_cp_completion(observed, targets, field.shape, rank=rank, device=device)
                row[f"em_cp_rank{rank}"] = nrmse(rec[tuple(test.T)], truth)
            rec = em_tucker_completion(observed, targets, field.shape, ranks=ranks, device=device)
            row["em_tucker"] = nrmse(rec[tuple(test.T)], truth)
            best = None
            for length_scale in (1.0, 2.0, 3.0, 5.0, 8.0):
                value = nrmse(rbf_interpolation(
                    observed, targets, test, device=device, length_scale=length_scale), truth)
                best = value if best is None else min(best, value)
                row[f"rbf_ls{length_scale:g}"] = value
            # Oracle-tuned: the strongest possible version of "just smoothing".
            row["rbf_best_oracle_lengthscale"] = best
            records.append(row)
            print(f"  seed {seed} ratio {ratio:5.3f}  " + "  ".join(
                f"{k}={v:.3f}" for k, v in row.items()
                if k.startswith(("em_", "rbf_best", "global"))), flush=True)

    summary = {"records": records, "ranks": list(ranks), "aggregate": {}}
    keys = [k for k in records[0] if k not in ("seed", "ratio")]
    for ratio in args.ratios:
        cells = [r for r in records if r["ratio"] == ratio]
        summary["aggregate"][str(ratio)] = {
            k: float(np.mean([c[k] for c in cells])) for k in keys}
    (args.output / "baselines_summary.json").write_text(json.dumps(summary, indent=2))
    print("\nratio  " + "  ".join(f"{k}" for k in keys if not k.startswith("rbf_ls")))
    for ratio in args.ratios:
        agg = summary["aggregate"][str(ratio)]
        print(f"{ratio:5.3f}  " + "  ".join(
            f"{agg[k]:.4f}" for k in keys if not k.startswith("rbf_ls")))


if __name__ == "__main__":
    main()
