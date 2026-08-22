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
             r"  \caption{\textbf{Scenario coverage.}  The tuning cost -- what it "
             r"costs to choose a kernel from data a practitioner can collect rather "
             r"than from the held-out region -- across three operator families, two "
             r"dimensionalities and every sensor layout.  It is near zero wherever "
             r"the sensors are scattered and large wherever they are confined, in "
             r"every scenario, which is the claim this paper makes.  Cells marked "
             r"\emph{---} were not run.}",
             r"  \label{tab:coverage}", r"  \small",
             r"  \begin{tabular}{llccccc}", r"    \toprule",
             r"    & & \multicolumn{5}{c}{tuning cost, by sensor layout} \\",
             r"    \cmidrule(lr){3-7}",
             r"    dimension & operator & scattered & \multicolumn{3}{c}{confined}"
             r" & \\", r"    \midrule"]

    for dimension, layouts in (("2-D space $+$ time", LAYOUT_2D),
                               ("3-D space $+$ time", LAYOUT_3D)):
        header = " & ".join(PRETTY[l] for l in layouts)
        lines.append(rf"    \multicolumn{{2}}{{l}}{{\emph{{{dimension}}}}} & "
                     + header + (" & " if len(layouts) < 5 else "") + r" \\")
        for label in ("reaction--diffusion", "diffusion-dominated", "advection--diffusion"):
            cells = grid.get((dimension, label))
            values = []
            for layout in layouts:
                record = (cells or {}).get(layout)
                values.append("---" if record is None
                              else f"${record['tuning_cost']:+.4f}$")
            while len(values) < 5:
                values.append("")
            lines.append(f"    & {label} & " + " & ".join(values) + r" \\")
        lines.append(r"    \addlinespace")
    lines += [r"    \bottomrule", r"  \end{tabular}", r"\end{table}"]
    (OUT / "table_coverage.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # a plain-text version for the documentation
    report = ["| 维度 | 算子 | 布局 | 调参代价 | ours | oracle | ours−oracle |",
              "|---|---|---|---|---|---|---|"]
    for (dimension, label), cells in grid.items():
        if cells is None:
            report.append(f"| {dimension} | {label} | — | **未跑** | | | |")
            continue
        for layout, record in cells.items():
            report.append(
                f"| {dimension} | {label} | {PRETTY.get(layout, layout)} | "
                f"{record['tuning_cost']:+.4f} | {record['ours_pde']['mean']:.4f} | "
                f"{record['matern_oracle']['mean']:.4f} | "
                f"{record['gap_to_oracle']:+.4f} |")
    (LEAK / "coverage_table.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    filled = sum(1 for c in grid.values() if c)
    print(f"wrote table_coverage.tex and coverage_table.md "
          f"({filled}/{len(grid)} scenario rows have data)")


if __name__ == "__main__":
    main()
