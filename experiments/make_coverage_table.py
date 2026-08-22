#!/usr/bin/env python3
"""The scenario grid: physics x dimension x sensor layout, in one table.

Individual studies each answer one question, and a reader who sees only the main
table cannot tell whether the result is a property of one field or of the
setting.  This assembles every recorded cell into the grid the claim actually
needs to hold on, and marks the cells that were never run rather than leaving
them to be inferred from silence.
"""
from __future__ import annotations
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LEAK = ROOT / "results" / "leak"
OUT = ROOT / "paper" / "sections"

# (label, summary file, dimension label)
SOURCES = [
    ("reaction--diffusion", "operator_grid_summary.json", "2-D space $+$ time",
     "reaction-diffusion"),
    ("diffusion-dominated", "operator_grid_summary.json", "2-D space $+$ time",
     "diffusion-dominated"),
    ("advection--diffusion", "operator_grid_summary.json", "2-D space $+$ time",
     "advection-diffusion"),
    ("reaction--diffusion", "leak3d_fixed_summary.json", "3-D space $+$ time", None),
    ("diffusion-dominated", "leak3d_diffusion_summary.json", "3-D space $+$ time", None),
    ("advection--diffusion", "leak3d_advection_summary.json", "3-D space $+$ time", None),
]

LAYOUT_2D = ["random", "wall_ring", "near_wall", "one_wall_strip", "corner_block"]
LAYOUT_3D = ["random", "two_faces_opposite", "one_face", "corner_cube"]
PRETTY = {"random": "scattered", "wall_ring": "all walls", "near_wall": "inner band",
          "one_wall_strip": "one wall", "corner_block": "one corner",
          "two_faces_opposite": "two faces", "one_face": "one face",
          "corner_cube": "one corner"}


def load(name):
    path = LEAK / name
    return json.loads(path.read_text()) if path.exists() else None


def collect():
    grid = {}
    for label, filename, dimension, family in SOURCES:
        data = load(filename)
        if data is None:
            grid[(dimension, label)] = None
            continue
        cells = {}
        for record in data["records"]:
            if family is not None and record.get("family") != family:
                continue
            cells[record["layout"]] = record
        grid[(dimension, label)] = cells or None
    return grid


def main() -> None:
    grid = collect()
    lines = [r"\begin{table}[t]", r"  \centering",
             r"  \caption{\textbf{Scenario coverage: held-out NRMSE.}  Three operator "
             r"families, two dimensionalities, every sensor layout, $1\%$ observed.  "
             r"Each cell is \emph{ours} against the best arm a practitioner could "
             r"deploy, with the relative improvement.  Where sensors are scattered the "
             r"two are level; where they are confined the gap opens.  Cells marked "
             r"\emph{---} were not run.}",
             r"  \label{tab:coverage}", r"  \small",
             r"  \begin{tabular}{llccccc}", r"    \toprule",
             r"    & & scattered & \multicolumn{4}{c}{confined} \\",
             r"    \cmidrule(lr){3-3}\cmidrule(lr){4-7}"]

    for dimension, layouts in (("2-D space $+$ time", LAYOUT_2D),
                               ("3-D space $+$ time", LAYOUT_3D)):
        lines.append(r"    \midrule")
        header = " & ".join(PRETTY[l] for l in layouts)
        pad = " & " * (5 - len(layouts))
        lines.append(rf"    \multicolumn{{2}}{{l}}{{\emph{{{dimension}}}}} & "
                     + header + pad + r" \\")
        for label in ("reaction--diffusion", "diffusion-dominated",
                      "advection--diffusion"):
            cells = grid.get((dimension, label))
            ours_row, base_row = [], []
            for layout in layouts:
                record = (cells or {}).get(layout)
                if record is None:
                    ours_row.append("---"); base_row.append("")
                    continue
                ours = record["ours_pde"]["mean"]
                rival = min(record[k]["mean"] for k in
                            ("matern_deployable", "spectral_mixture", "neural_tucker")
                            if k in record)
                better = ours < rival
                ours_row.append(rf"\textbf{{{ours:.4f}}}" if better else f"{ours:.4f}")
                base_row.append(rf"\emph{{{rival:.4f}}}"
                                + (rf" \scriptsize$({100 * (rival - ours) / rival:+.0f}\%)$"
                                   if better else ""))
            while len(ours_row) < 5:
                ours_row.append(""); base_row.append("")
            lines.append(f"    & {label} & " + " & ".join(ours_row) + r" \\")
            lines.append(r"    & \scriptsize best deployable & "
                         + " & ".join(rf"\scriptsize {b}" if b else "" for b in base_row)
                         + r" \\")
            lines.append(r"    \addlinespace[2pt]")
    lines += [r"    \bottomrule", r"  \end{tabular}", r"\end{table}"]
    (OUT / "table_coverage.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    report = ["| 维度 | 算子 | 布局 | **ours** | 最佳 baseline | 相对改进 |",
              "|---|---|---|---|---|---|"]
    for (dimension, label), cells in grid.items():
        if cells is None:
            report.append(f"| {dimension} | {label} | — | **未跑** | | |")
            continue
        for layout in (LAYOUT_3D if "3-D" in dimension else LAYOUT_2D):
            record = cells.get(layout)
            if record is None:
                continue
            ours = record["ours_pde"]["mean"]
            rival = min(record[k]["mean"] for k in
                        ("matern_deployable", "spectral_mixture", "neural_tucker")
                        if k in record)
            report.append(
                f"| {dimension} | {label} | {PRETTY.get(layout, layout)} | "
                f"**{ours:.4f}** | {rival:.4f} | {100 * (rival - ours) / rival:+.1f}% |")
    (LEAK / "coverage_table.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    filled = sum(1 for c in grid.values() if c)
    print(f"wrote table_coverage.tex and coverage_table.md "
          f"({filled}/{len(grid)} scenario rows have data)")


if __name__ == "__main__":
    main()
