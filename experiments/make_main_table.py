#!/usr/bin/env python3
"""Generate the paper's main table straight from the recorded summary.

Transcribing numbers by hand is how a retracted table survives a revision, so
the LaTeX is generated rather than typed.  Running this after any rerun is the
only supported way to update Table 1.
"""
from __future__ import annotations
import json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PRETTY = {"random": "anywhere in the room", "wall_ring": "all four walls",
          "near_wall": "band inside the walls", "one_wall_strip": "one wall only",
          "corner_block": "one corner patch"}
COVERAGE = {"random": 100, "wall_ring": 12, "near_wall": 26,
            "one_wall_strip": 8, "corner_block": 10}


def main() -> None:
    tag = sys.argv[1] if len(sys.argv) > 1 else "leak_main3tier"
    data = json.loads((ROOT / "results" / "leak" / f"{tag}_summary.json").read_text())
    seeds = len(data["records"][0]["ours_pde"]["values"])
    lines = [
        r"\begin{table}[t]", r"  \centering",
        r"  \caption{Held-out NRMSE by sensor layout, all at $1\%$ observed, "
        f"{seeds} seeds, identical fields, masks and noise across arms."
        r"  The two Mat\'ern columns differ only in what data chose the length"
        r" scale: \emph{deployable} scores candidates on a quarter of the sensor"
        r" readings, which is all a practitioner has, while \emph{oracle} scores"
        r" them on the true held-out region and is marked $\star$ because no one"
        r" can run it.  \emph{Tuning cost} is the gap between them -- what it"
        r" costs to have to choose the length scale from data you can actually"
        r" collect.  NRMSE is normalised by the held-out standard deviation, so"
        r" $1.0$ is what predicting the mean scores.}",
        r"  \label{tab:layouts}", r"  \small",
        r"  \begin{tabular}{lccccccr}", r"    \toprule",
        r"    Sensor layout & reach & ours & Mat\'ern & Mat\'ern$^\star$ & mixture"
        r" & neural CP & tuning \\",
        r"     & & (no tuning) & deployable & oracle & & & cost \\", r"    \midrule",
    ]
    order = ["random", "wall_ring", "near_wall", "one_wall_strip", "corner_block"]
    present = {r["layout"]: r for r in data["records"]}
    for layout in order:
        record = present.get(layout)
        if record is None:
            continue
        best_dep = min(record[k]["mean"] for k in
                       ("matern_deployable", "spectral_mixture", "neural_tucker"))
        cells = []
        for key in ("ours_pde", "matern_deployable", "matern_oracle",
                    "spectral_mixture", "neural_tucker"):
            value = f"{record[key]['mean']:.4f}"
            # Bold the best arm a practitioner could actually deploy.
            if key != "matern_oracle" and abs(record[key]["mean"] - min(
                    best_dep, record["ours_pde"]["mean"])) < 1e-12:
                value = rf"\textbf{{{value}}}"
            cells.append(value)
        lines.append(f"    {PRETTY[layout]} & ${COVERAGE[layout]}\\%$ & "
                     + " & ".join(cells)
                     + f" & ${record['tuning_cost']:+.4f}$ \\\\")
    lines += [r"    \bottomrule", r"  \end{tabular}", r"\end{table}"]
    out = ROOT / "paper" / "sections" / "table_layouts.tex"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {out} from {tag} ({len(present)} of {len(order)} layouts, {seeds} seeds)")
    if len(present) < len(order):
        print("  WARNING: table is incomplete; missing "
              + ", ".join(l for l in order if l not in present))


if __name__ == "__main__":
    main()
