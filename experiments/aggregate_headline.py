#!/usr/bin/env python3
"""Pool every 1% run into the headline statistic, with a paired test.

Kept separate from the run scripts so the statistic is computed once, from the
recorded JSONs, and cannot drift from what the tables report.  The test is
paired because every arm sees the identical field, mask and noise within a seed.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.stats import binomtest, wilcoxon

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "forced_pde"
# Default to the fair comparison, in which each arm runs at the routing that is
# best for it.  The earlier per-mode/rank sources are kept for the record but
# are not the headline, because that setting handicaps the generic dictionary
# roughly twice as much as it handicaps the operator bank.
SOURCES = ["fair_global_summary.json"]
LEGACY_SOURCES = ["main_fixedforcing_summary.json", "main_extraseeds_summary.json"]


def main(ratio: float = 0.01, sources: list[str] | None = None,
         output: str = "headline_summary.json") -> None:
    rows, used = [], []
    for name in (sources or SOURCES):
        path = RESULTS / name
        if not path.exists():
            print(f"skip {name}: not found")
            continue
        used.append(name)
        rows += [r for r in json.loads(path.read_text())["records"]
                 if abs(r["ratio"] - ratio) < 1e-9]
    if not rows:
        print("no records"); return
    operator = np.array([r["operator"]["test_nrmse"] for r in rows])
    generic = np.array([r["generic"]["test_nrmse"] for r in rows])
    neighbour = np.array([r["nearest_neighbour"] for r in rows])
    ceiling = np.array([r["tucker_ceiling"] for r in rows])
    margin = generic - operator
    wins = int((margin > 0).sum())
    summary = {
        "sources": used, "ratio": ratio, "seeds": len(rows),
        "operator": {"mean": float(operator.mean()), "std": float(operator.std())},
        "generic": {"mean": float(generic.mean()), "std": float(generic.std())},
        "nearest_neighbour_mean": float(neighbour.mean()),
        "fully_observed_tucker_ceiling_mean": float(ceiling.mean()),
        "paired_margin": {"mean": float(margin.mean()), "std": float(margin.std())},
        "relative_improvement_percent": float(100 * margin.mean() / generic.mean()),
        "paired_wins": f"{wins}/{len(rows)}",
        "sign_test_p": float(binomtest(wins, len(rows), 0.5, alternative="greater").pvalue),
        "wilcoxon_p": float(wilcoxon(operator, generic, alternative="less").pvalue),
    }
    (RESULTS / output).write_text(json.dumps(summary, indent=2))
    for key, value in summary.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    print("fair comparison (both arms at global routing):")
    main()
    print("\nlegacy per-mode/rank comparison, kept for the record:")
    main(sources=LEGACY_SOURCES, output="headline_legacy_pmr_summary.json")
