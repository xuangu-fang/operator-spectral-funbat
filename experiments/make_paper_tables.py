#!/usr/bin/env python3
"""Generate every table in the paper from the recorded summaries.

Nothing here is typed by hand.  This table has been retracted once already, and
hand transcription is how a retracted number survives a revision.
"""
from __future__ import annotations
import json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LEAK = ROOT / "results" / "leak"
OUT = ROOT / "paper" / "sections"

PRETTY = {"random": "anywhere in the room", "wall_ring": "all four walls",
          "near_wall": "band inside the walls", "one_wall_strip": "one wall ($x$)",
          "corner_block": "one corner patch", "one_wall_strip_y": "one wall ($y$)",
          "two_walls_lr": "left and right walls", "two_walls_tb": "top and bottom walls",
          "two_walls_adjacent": "two adjacent walls", "four_corners": "four corner patches",
          "one_face": "one face", "two_faces_opposite": "two opposite faces",
          "corner_cube": "one corner cube", "all_six_faces": "all six faces",
          "floor_only": "the floor"}
REACH = {"random": 100, "wall_ring": 12, "near_wall": 26, "one_wall_strip": 8,
         "corner_block": 10, "one_wall_strip_y": 8, "two_walls_lr": 16,
         "two_walls_tb": 16, "two_walls_adjacent": 15, "four_corners": 10}


def load(name):
    path = LEAK / name
    return json.loads(path.read_text()) if path.exists() else None


def bold(value, is_best):
    return rf"\textbf{{{value:.4f}}}" if is_best else f"{value:.4f}"


def main_table():
    data = load("leak_main3tier_summary.json")
    if data is None:
        print("skip main table"); return
    seeds = len(data["records"][0]["ours_pde"]["values"])
    lines = [r"\begin{table}[t]", r"  \centering",
             r"  \caption{\textbf{Held-out NRMSE by sensor layout}, all at $1\%$"
             rf" observed, {seeds} seeds, identical fields, masks and noise across"
             r" arms.  The two Mat\'ern columns differ in one thing only: what data"
             r" chose the length scale.  \emph{Deployable} scores candidates on a"
             r" quarter of the sensor readings, which is all a practitioner has;"
             r" \emph{oracle}$^\star$ scores them on the true held-out region, which"
             r" nobody can do.  Their difference is the \emph{tuning cost} -- what it"
             r" costs to have to choose a kernel from data you can collect.  Our arm"
             r" uses no tuning data at all.  NRMSE is normalised by the held-out"
             r" standard deviation, so $1.0$ is exactly what predicting the mean"
             r" scores.  Bold marks the best arm a practitioner could deploy.}",
             r"  \label{tab:layouts}", r"  \small",
             r"  \begin{tabular}{lccccccr}", r"    \toprule",
             r"    & & \multicolumn{2}{c}{no tuning data} &"
             r" \multicolumn{2}{c}{tuned} & & \\",
             r"    \cmidrule(lr){3-4}\cmidrule(lr){5-6}",
             r"    Sensor layout & reach & ours & neural CP & Mat\'ern &"
             r" Mat\'ern$^\star$ & mixture & tuning \\",
             r"    & & & & deployable & oracle & & cost \\", r"    \midrule"]
    order = ["random", "wall_ring", "near_wall", "one_wall_strip", "corner_block"]
    present = {r["layout"]: r for r in data["records"]}
    for layout in order:
        record = present.get(layout)
        if record is None:
            continue
        deployable = {k: record[k]["mean"] for k in
                      ("ours_pde", "matern_deployable", "spectral_mixture", "neural_tucker")}
        best = min(deployable.values())
        cells = [bold(record[k]["mean"], abs(record[k]["mean"] - best) < 1e-12)
                 if k != "matern_oracle" else f"{record[k]['mean']:.4f}"
                 for k in ("ours_pde", "neural_tucker", "matern_deployable",
                           "matern_oracle", "spectral_mixture")]
        lines.append(f"    {PRETTY[layout]} & ${REACH[layout]}\\%$ & "
                     + " & ".join(cells)
                     + f" & ${record['tuning_cost']:+.4f}$ \\\\")
    lines += [r"    \bottomrule", r"  \end{tabular}", r"\end{table}"]
    (OUT / "table_layouts.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote table_layouts.tex ({len(present)}/{len(order)} layouts, {seeds} seeds)")


def replication_table():
    """One table for the claim's reach: other operators, and three dimensions."""
    ops = load("operator_families_summary.json")
    three = load("leak3d_summary.json")
    if ops is None and three is None:
        print("skip replication table"); return
    lines = [r"\begin{table}[t]", r"  \centering",
             r"  \caption{\textbf{The tuning cost tracks the sensor geometry, not"
             r" the operator or the dimension.}  Same three tiers as"
             r" \Cref{tab:layouts}.  Every row with scattered sensors pays almost"
             r" nothing for having to tune; every confined row pays heavily.  The"
             r" advection family is included because we expected it to fail: the"
             r" construction uses an axis-wise even magnitude spectrum and advection"
             r" tilts the spectrum off centre.  At Peclet ${\approx}5$ it does not"
             r" fail.}",
             r"  \label{tab:replication}", r"  \small",
             r"  \begin{tabular}{llcccr}", r"    \toprule",
             r"    Field & sensors & ours & Mat\'ern & Mat\'ern$^\star$ & tuning \\",
             r"    & & (no tuning) & deployable & oracle & cost \\", r"    \midrule"]
    if ops is not None:
        for record in ops["records"]:
            lines.append(
                f"    {record['family']} & {PRETTY.get(record['layout'], record['layout'])} & "
                f"{record['ours_pde']['mean']:.4f} & "
                f"{record['matern_deployable']['mean']:.4f} & "
                f"{record['matern_oracle']['mean']:.4f} & "
                f"${record['tuning_cost']:+.4f}$ \\\\")
    if three is not None:
        lines.append(r"    \midrule")
        seeds3 = len(three["records"][0]["ours_pde"]["values"])
        for index, record in enumerate(three["records"]):
            label = (rf"3-D room ($32^4$, {seeds3} seeds)" if index == 0 else "")
            lines.append(
                f"    {label} & {PRETTY.get(record['layout'], record['layout'])} & "
                f"{record['ours_pde']['mean']:.4f} & "
                f"{record['matern_deployable']['mean']:.4f} & "
                f"{record['matern_oracle']['mean']:.4f} & "
                f"${record['tuning_cost']:+.4f}$ \\\\")
    lines += [r"    \bottomrule", r"  \end{tabular}", r"\end{table}"]
    (OUT / "table_replication.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("wrote table_replication.tex")


def geometry_table():
    """Matched pairs: one factor changes per pair, the budget never does."""
    data = load("leak_geometry_summary.json")
    if data is None:
        print("skip geometry table"); return
    seeds = len(data["records"][0]["ours_pde"]["values"])
    present = {r["layout"]: r for r in data["records"]}
    pairs = [("one_wall_strip", "one_wall_strip_y", "which wall, same $8\\%$ reach"),
             ("two_walls_lr", "two_walls_tb", "which pair of walls, same $16\\%$"),
             ("corner_block", "four_corners", "one patch or four, same $400$ cells")]
    lines = [r"\begin{table}[t]", r"  \centering",
             rf"  \caption{{\textbf{{Matched geometries, {seeds} seeds.}}  Each pair"
             r" changes one factor and holds the observation budget fixed.  The first"
             r" pair is the paper's open problem: moving the same strip from one wall"
             r" to the other turns a tie with the oracle into a large loss, and an"
             r" isotropic control (\Cref{app:robustness}) shows this is not about the"
             r" axes' diffusivities.  Rows where our arm exceeds $1.0$ are worse than"
             r" predicting the mean.}",
             r"  \label{tab:geometry}", r"  \small",
             r"  \begin{tabular}{llcccr}", r"    \toprule",
             r"    Pair & layout & ours & Mat\'ern & Mat\'ern$^\star$ & ours vs \\",
             r"    & & (no tuning) & deployable & oracle & oracle \\", r"    \midrule"]
    for first, second, caption in pairs:
        for index, layout in enumerate((first, second)):
            record = present.get(layout)
            if record is None:
                continue
            gap = record["gap_to_oracle"]
            value = f"{record['ours_pde']['mean']:.4f}"
            if record["ours_pde"]["mean"] > 1.0:
                value = rf"\underline{{{value}}}"
            lines.append(
                f"    {caption if index == 0 else ''} & {PRETTY.get(layout, layout)} & "
                f"{value} & {record['matern_deployable']['mean']:.4f} & "
                f"{record['matern_oracle']['mean']:.4f} & ${gap:+.4f}$ \\\\")
        lines.append(r"    \addlinespace")
    record = present.get("two_walls_adjacent")
    if record is not None:
        lines.append(
            r"    two adjacent walls, for reference & two adjacent walls & "
            f"{record['ours_pde']['mean']:.4f} & "
            f"{record['matern_deployable']['mean']:.4f} & "
            f"{record['matern_oracle']['mean']:.4f} & "
            f"${record['gap_to_oracle']:+.4f}$ \\\\")
    lines += [r"    \bottomrule", r"  \end{tabular}", r"\end{table}"]
    (OUT / "table_geometry.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("wrote table_geometry.tex")


def baselines_table():
    """Everything that actually competes, in one place, with what each was given."""
    neural = load("neural_strong_summary.json")
    pinn = load("physics_baselines_summary.json")
    if neural is None and pinn is None:
        print("skip baselines table"); return
    order = ["random", "one_wall_strip", "corner_block"]
    by_layout = {}
    for source, keys in ((neural, ("costco_oracle", "fourier_mlp_oracle", "lrtfr_siren")),
                         (pinn, ("pinn_oracle", "network_no_physics"))):
        if source is None:
            continue
        for record in source["records"]:
            entry = by_layout.setdefault(record["layout"], {})
            entry["ours_pde"] = record["ours_pde"]["mean"]
            for key in keys:
                entry[key] = record[key]["mean"]
    columns = [("ours_pde", r"ours"), ("pinn_oracle", r"PINN$^\star$"),
               ("costco_oracle", r"CoSTCo$^\star$"),
               ("fourier_mlp_oracle", r"Fourier MLP$^\star$"),
               ("lrtfr_siren", "LRTFR"),
               ("network_no_physics", "same net, no physics")]
    lines = [r"\begin{table}[t]", r"  \centering",
             r"  \caption{\textbf{Against models with far more capacity, and against "
             r"the same physics spent differently.}  Held-out NRMSE at $1\%$ observed, "
             r"three seeds.  Starred arms choose their architecture, learning rate or "
             r"residual weight against the held-out region and get four to six times "
             r"our optimisation budget; ours uses one configuration, coefficients wrong "
             r"by $50\%$ and no tuning data.  The PINN is told exactly what we are told "
             r"and spends it on a residual penalty instead of a prior; the last column "
             r"is that same network with the physics switched off, so the residual's "
             r"own contribution is measured rather than inferred.}",
             r"  \label{tab:baselines-strong}", r"  \small",
             r"  \begin{tabular}{l" + "c" * len(columns) + r"}", r"    \toprule",
             r"    Sensor layout & " + " & ".join(label for _, label in columns) + r" \\",
             r"    \midrule"]
    for layout in order:
        entry = by_layout.get(layout)
        if entry is None:
            continue
        cells = []
        for key, _ in columns:
            value = entry.get(key)
            cells.append("---" if value is None else f"{value:.4f}")
        lines.append(f"    {PRETTY.get(layout, layout)} & " + " & ".join(cells) + r" \\")
    lines += [r"    \bottomrule", r"  \end{tabular}", r"\end{table}"]
    (OUT / "table_baselines.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("wrote table_baselines.tex")


def ladder_table():
    """How much of the equation is actually needed."""
    data = load("knowledge_ladder_leak_summary.json")
    if data is None:
        print("skip ladder table"); return
    arms = ["K2 true coefficients", "K1 point (main table)", "K1 bank x[1/3,3]",
            "K0 bank x[1/10,10]", "K-1 generic, matched atoms"]
    labels = [r"K2: true $\theta^\star$", r"K1 point: one guess, $1.5\times$ off",
              r"\textbf{K1 bank}: only $\theta^\star \in \times[1/3,3]$",
              r"K0 bank: only $\theta^\star \in \times[1/10,10]$",
              r"K$-$1: generic, same atom count"]
    lines = [r"\begin{table}[t]", r"  \centering",
             r"  \caption{\textbf{How much of the equation is needed.}  Held-out NRMSE "
             r"at $1\%$ observed, three seeds.  K1 knows only that the coefficients lie "
             r"in a declared range: it samples the range, separates each sample and pools "
             r"the atoms, leaving the mixture weights to infer the parameter from data.  "
             r"A PINN and an AutoIP-style GP cannot work this way -- both must commit to "
             r"one $\theta$ before seeing data.  The generic bank is given the same atom "
             r"count as the pooled banks, and the collapsed parameterisation keeps the "
             r"variational coefficient count independent of bank size, so neither "
             r"comparison is confounded by parameters.}",
             r"  \label{tab:ladder}", r"  \small",
             r"  \begin{tabular}{lccc}", r"    \toprule",
             r"    What is known & anywhere & band inside & one wall \\",
             r"    & in the room & the walls & only \\", r"    \midrule"]
    present = {r["layout"]: r for r in data["records"]}
    for arm, label in zip(arms, labels):
        cells = []
        for layout in ("random", "near_wall", "one_wall_strip"):
            record = present.get(layout)
            cells.append("---" if record is None else f"{record[arm]['mean']:.4f}")
        lines.append(f"    {label} & " + " & ".join(cells) + r" \\")
    lines += [r"    \bottomrule", r"  \end{tabular}", r"\end{table}"]
    (OUT / "table_ladder.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("wrote table_ladder.tex")


if __name__ == "__main__":
    main_table(); replication_table(); geometry_table()
    baselines_table(); ladder_table()
