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


SWEEP_LABEL = {"ratio": "fraction of entries observed",
               "noise": "observation noise standard deviation",
               "coefficient": "nominal coefficients, as a factor of the truth"}


def robustness() -> None:
    """The three one-dimensional sweeps, against the oracle Matern tier."""
    path = LEAK / "robustness_summary.json"
    if not path.exists():
        print("skip robustness: run not finished"); return
    data = json.loads(path.read_text())
    lines = []
    for sweep, records in data["sweeps"].items():
        lines += [r"\begin{table}[h]", r"  \centering",
                  rf"  \caption{{Sweep over the {SWEEP_LABEL.get(sweep, sweep)}, "
                  r"one-wall layout.  The Mat\'ern column is the oracle tier, "
                  r"tuned against the held-out region.}",
                  rf"  \label{{tab:sweep-{sweep}}}", r"  \small",
                  r"  \begin{tabular}{lcccc}", r"    \toprule",
                  rf"    {SWEEP_LABEL.get(sweep, sweep)} & ours & Mat\'ern$^\star$ "
                  r"& neural CP & margin \\", r"    \midrule"]
        for record in records:
            value = record[sweep]
            margin = record.get("margin_vs_best_baseline", 0.0)
            lines.append(f"    ${value:g}$ & {record['ours_pde']['mean']:.4f} & "
                         f"{record['matern']['mean']:.4f} & "
                         f"{record['neural_tucker']['mean']:.4f} & "
                         f"${margin:+.4f}$ \\\\")
        lines += [r"    \bottomrule", r"  \end{tabular}", r"\end{table}", ""]
    out = ROOT / "paper" / "sections" / "table_robustness.tex"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {out}: {len(data['sweeps'])} sweeps")


def anisotropy() -> None:
    """The y-wall against x-wall mechanism test, with its isotropic control."""
    path = LEAK / "anisotropy_axis_summary.json"
    if not path.exists():
        print("skip anisotropy: run not finished"); return
    data = json.loads(path.read_text())
    verdict = data["verdict"]
    wall = {"one_wall_strip": "$x$ wall (extrapolate along the smooth axis)",
            "one_wall_strip_y": "$y$ wall (extrapolate along the rough axis)"}
    lines = [r"\begin{table}[h]", r"  \centering",
             r"  \caption{Which wall the sensors sit on, at $1\%$ observed.  The "
             r"prediction, registered before the run: the $y$ wall must show the "
             r"larger margin in the anisotropic field, and the isotropic control "
             r"must show no such gap.}",
             r"  \label{tab:anisotropy}", r"  \small",
             r"  \begin{tabular}{llccc}", r"    \toprule",
             r"    field & sensors on & ours & Mat\'ern$^\star$ & margin \\",
             r"    \midrule"]
    for record in data["records"]:
        lines.append(f"    {record['condition']} & {wall.get(record['layout'], record['layout'])} & "
                     f"{record['ours_pde']['mean']:.4f} & {record['matern']['mean']:.4f} & "
                     f"${record['paired_margin']:+.4f}$ ({record['relative_percent']:+.1f}\\%) \\\\")
    lines += [r"    \bottomrule", r"  \end{tabular}", r"\end{table}", "",
              rf"The $y$ wall minus $x$ wall margin is "
              rf"${verdict['anisotropic_gap']:+.1f}$ points in the anisotropic field "
              rf"and ${verdict['isotropic_gap']:+.1f}$ points in the isotropic control. "
              + ("The prediction holds: the advantage follows the axis, and it "
                 "vanishes when the axes are made equivalent."
                 if verdict["prediction_holds"] else
                 "\\textbf{The prediction fails.}  The margin does not follow the "
                 "axis in the way a per-axis spectral prior requires, so the "
                 "mechanism offered for the main table is not supported by this "
                 "test and should be read as a conjecture rather than an "
                 "explanation.")]
    out = ROOT / "paper" / "sections" / "table_anisotropy.tex"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {out}: prediction_holds={verdict['prediction_holds']}")


if __name__ == "__main__":
    main(); robustness(); anisotropy()
