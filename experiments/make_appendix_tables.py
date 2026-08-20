#!/usr/bin/env python3
"""Generate the appendix's per-seed tables from recorded summaries."""
from __future__ import annotations
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LEAK = ROOT / "results" / "leak"
PRETTY = {"random": "anywhere in the room", "wall_ring": "all four walls",
          "near_wall": "band inside the walls", "one_wall_strip": "one wall only",
          "corner_block": "one corner patch"}
ARMS = [("ours_pde", r"ours"), ("matern_deployable", r"Mat\'ern deployable"),
        ("matern_oracle", r"Mat\'ern oracle$^\star$"),
        ("spectral_mixture", "spectral mixture"), ("neural_tucker", "neural CP")]


def main() -> None:
    data = json.loads((LEAK / "leak_main3tier_summary.json").read_text())
    seeds = len(data["records"][0]["ours_pde"]["values"])
    lines = [r"\section{Full numerical tables}", r"\label{app:tables}", "",
             "Per-seed held-out NRMSE behind \\Cref{tab:layouts}.  All arms share "
             "the field, mask and noise within a seed, so every comparison is paired.",
             ""]
    for record in data["records"]:
        lines += [r"\begin{table}[h]", r"  \centering",
                  rf"  \caption{{{PRETTY.get(record['layout'], record['layout'])}, "
                  rf"$1\%$ observed, {record['observed']} points.  Validation chose "
                  rf"$\ell = {record['chosen_length_scales']}$; the held-out region "
                  rf"wanted $\ell = {record.get('oracle_length_scales', 'n/a')}$.}}",
                  r"  \small",
                  r"  \begin{tabular}{l" + "c" * seeds + r"c}", r"    \toprule",
                  r"    & " + " & ".join(f"seed {i}" for i in range(seeds)) + r" & mean \\",
                  r"    \midrule"]
        for key, label in ARMS:
            values = record[key]["values"]
            lines.append(f"    {label} & "
                         + " & ".join(f"{v:.4f}" for v in values)
                         + f" & {record[key]['mean']:.4f} " + r"\\")
        lines += [r"    \bottomrule", r"  \end{tabular}", r"\end{table}", ""]
    out = ROOT / "paper" / "sections" / "A1_full_tables.tex"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {out}: {len(data['records'])} layouts x {seeds} seeds")


if __name__ == "__main__":
    main()
