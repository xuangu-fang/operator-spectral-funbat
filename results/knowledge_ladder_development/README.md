# Knowledge-ladder development sweep (2026-08-19)

**DEVELOPMENT SEEDS 101--105 ONLY.  Nothing here may enter a main table.**
Confirmation requires fresh seeds 301--305, because 201--205 are published.

| file | produced by |
|---|---|
| `seed1xx.json`, `summary.json`, `knowledge_ladder_development.png` | `experiments/run_knowledge_ladder_development.py` |
| `width_sweep_summary.json`, `knowledge_ladder_width_sweep.png` | `experiments/run_knowledge_ladder_width_sweep.py` |
| `bank_reachability.json` | `experiments/analyze_bank_reachability.py` |

Findings and the three refuted predictions are written up in
`docs/PAPER_TECHNICAL_REPORT_ZH.md` section 10.7 and `docs/ITERATIONS.md` R10.

Two caveats that are easy to lose:

1. `summary.json -> predictions -> P1 -> monotone_nondecreasing` reports `false`.
   The pre-registered text permits `K2 ~= K1`; the script implemented a strict
   `<=`.  The registered prediction passes; the implementation is stricter than
   what was registered.  Left as-is rather than retro-fitted.
2. The `k0` entry in `summary.json` and `pooled_w10.0` in
   `width_sweep_summary.json` are the *same* prior width with *different* Latin
   hypercube seeds, and differ by `0.0550` vs `0.0443`.  Bank construction is
   itself a random object; single-draw bank results must not be read as width
   effects.
