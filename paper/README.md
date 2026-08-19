# `paper/` — LaTeX 论文工程

论文正文用 LaTeX 直接写在这里。中文技术报告仍留在 `docs/`（markdown，GitHub 可渲染公式），
两者的**记号必须一致**：LaTeX 侧的符号宏集中定义在 [`macros.tex`](macros.tex)，
markdown 侧的符号表在 `docs/PAPER_TECHNICAL_REPORT_ZH.md` 第 3 节。

## 构建

```bash
cd paper
make            # latexmk -pdf main.tex  ->  main.pdf
make watch      # 持续编译预览
make clean      # 清中间文件
```

把中文技术报告渲染成 PDF（用于本地检查公式）：

```bash
make report-zh  # -> docs/PAPER_TECHNICAL_REPORT_ZH.pdf
```

## 目录结构

```
paper/
├── main.tex            # 主文件；换会议模板只需改顶部 documentclass/style 块
├── macros.tex          # 共享记号宏（与中文报告符号表一一对应）
├── refs.bib            # 参考文献
├── figures/            # 论文专用图；结果图直接从 ../results/ 引用（见 \graphicspath）
└── sections/
    ├── 00_abstract.tex
    ├── 01_introduction.tex
    ├── 02_related_work.tex
    ├── 03_problem.tex
    ├── 04_method.tex
    ├── 05_experiments.tex
    ├── 06_limitations.tex
    ├── 07_conclusion.tex
    └── A1_full_tables.tex
```

## 换会议模板

`main.tex` 顶部注释列出了 NeurIPS / ICML / ICLR 的样式包引入方式。把官方 `.sty` 放进
`paper/`，替换 `\documentclass` 与 `\usepackage{...}` 块即可，其余 section 文件不用改。

## 纪律提醒

- 表格数值只能来自 `results/submission_confirmation/summary.json`（untouched seeds 201–205）
  与 `results/signed_spectrum_audit/summary.json`。
- **不得**把 `results/advanced_poc_r1_r5/`（expanded 参数化）的数值与主表合并。
- `\todo{}` / `\note{}` 是草稿标记，投稿前必须清空（`grep -rn 'todo{' sections/`）。
