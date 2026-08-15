"""Command line runner for reproducible POC experiments."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path

import torch

from .data import load_dataset
from .masks import make_observation_split
from .models import build_model
from .plotting import plot_comparison, plot_reconstruction
from .training import TrainConfig, fit_model, seed_everything


DEFAULT_MODELS = "cp,inr,neural_cp,spectral_cp,bayesian_spectral_tensor,geo_nft"


def _jsonable(x):
    if isinstance(x, Path): return str(x)
    raise TypeError(type(x).__name__)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", choices=["synthetic_wave", "synthetic_boundary", "active_matter",
                                          "realpde_cylinder"],
                   default="synthetic_wave")
    p.add_argument("--models", default=DEFAULT_MODELS)
    p.add_argument("--ratios", default="0.01")
    p.add_argument("--masks", default="random")
    p.add_argument("--seeds", default="0")
    p.add_argument("--rank", type=int, default=8)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--steps", type=int, default=2500)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--batch-size", type=int, default=4096)
    p.add_argument("--reg-weight", type=float, default=2e-4)
    p.add_argument("--kl-weight", type=float, default=1.0)
    p.add_argument("--noise-std", type=float, default=0.05,
                   help="Gaussian noise as a fraction of observed-value std")
    p.add_argument("--record", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--output", type=Path, default=Path("runs/poc"))
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args(argv)
    if args.smoke:
        args.steps = min(args.steps, 30)
    dataset_kwargs = {"record": args.record} if args.dataset in {
        "active_matter", "realpde_cylinder"
    } else {}
    dataset = load_dataset(args.dataset, **dataset_kwargs)
    models = [x.strip() for x in args.models.split(",") if x.strip()]
    ratios = [float(x) for x in args.ratios.split(",")]
    masks = [x.strip() for x in args.masks.split(",") if x.strip()]
    seeds = [int(x) for x in args.seeds.split(",")]
    args.output.mkdir(parents=True, exist_ok=True)
    results = []

    for ratio in ratios:
        for mask_kind in masks:
            for seed in seeds:
                split = make_observation_split(dataset, ratio, mask_kind, seed)
                clean = dataset.values.clone()
                obs_scale = clean.reshape(-1)[split.observed].std()
                noise = torch.zeros_like(clean.reshape(-1))
                generator = torch.Generator().manual_seed(seed + 4401)
                noise[split.observed] = torch.randn(int(split.observed.sum()), generator=generator) * (
                    args.noise_std * obs_scale)
                dataset.values = (clean.reshape(-1) + noise).reshape(clean.shape)
                group = []
                for model_name in models:
                    seed_everything(seed)
                    model = build_model(model_name, dataset.shape, dataset.basis_specs,
                                        rank=args.rank, hidden=args.hidden)
                    cfg = TrainConfig(steps=args.steps, lr=args.lr, batch_size=args.batch_size,
                                      reg_weight=args.reg_weight, kl_weight=args.kl_weight,
                                      kl_warmup=max(10, args.steps // 5), seed=seed,
                                      eval_samples=8 if args.smoke else 32,
                                      log_every=max(1, args.steps // 10))
                    fit = fit_model(model, dataset, split, cfg, args.device)
                    tag = f"{dataset.name}_{mask_kind}_r{ratio:g}_s{seed}_{model_name}"
                    row = {"dataset": dataset.name, "source": dataset.source,
                           "shape": dataset.shape, "model": model_name, "mask": mask_kind,
                           "ratio": ratio, "seed": seed, "noise_std_fraction": args.noise_std,
                           "metrics": fit.metrics, "history": fit.history,
                           "elapsed_seconds": fit.elapsed_seconds,
                           "normalization": fit.normalization}
                    (args.output / f"{tag}.json").write_text(json.dumps(row, indent=2, default=_jsonable))
                    plot_reconstruction(dataset, split, fit.prediction, fit.predictive_std,
                                        args.output / "figures" / f"{tag}.png", tag)
                    results.append(row); group.append(row)
                    print(f"[{tag}] relL2={fit.metrics['relative_l2']:.4f} "
                          f"RMSE={fit.metrics['rmse']:.4g} time={fit.elapsed_seconds:.1f}s", flush=True)
                plot_comparison(dataset, group,
                                args.output / "figures" /
                                f"compare_{dataset.name}_{mask_kind}_r{ratio:g}_s{seed}.png")
                dataset.values = clean
    manifest = {"created_utc": datetime.now(timezone.utc).isoformat(),
                "dataset": {"name": dataset.name, "source": dataset.source,
                            "shape": dataset.shape, "description": dataset.description},
                "arguments": vars(args), "results": results}
    (args.output / "results.json").write_text(json.dumps(manifest, indent=2, default=_jsonable))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
