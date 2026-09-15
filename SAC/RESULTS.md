# SAC reproduction results — ours vs. original

_Generated 2026-09-15 20:01 by `util/make_results_md.py`. Re-run it after each eval to refresh._

Paper: *Autoencoding-Free Context Compression for LLMs via Contextual Semantic Anchors* (ICLR 2026). Base model Llama-3.2-1B, MRQA benchmark, 6 in-domain (ID) and 6 out-of-domain (OOD) subsets.

**How scores are computed.** Macro-average over the 6 subsets, ×100 — the convention the paper uses. (The `total_*` fields in the result JSONs are micro-averages and are not comparable to the paper.) F1 is the repo's ROUGE-based F1; EM is exact match.

## Summary

| Run | Ratio | ID F1 (ours / original) | ID EM (ours / original) | OOD F1 (ours / original) | OOD EM (ours / original) | Status |
|---|---|---|---|---|---|---|
| Released SAC checkpoint (eval only) | 15× | **54.98** / 54.95 (+0.03) | **39.67** / 39.67 (+0.00) | **39.25** / 39.26 (-0.02) | **26.17** / 26.02 (+0.15) | ✅ reproduced |
| Released SAC checkpoint (eval only) | 5× | — / 63.63 | — / 46.95 | — / 47.72 | — / 32.30 | ⏳ running |
| SAC trained by us | 15× | — / 54.95 | — / 39.67 | — / 39.26 | — / 26.02 | not started |
| SAC trained by us | 5× | — / 63.63 | — / 46.95 | — / 47.72 | — / 32.30 | not started |
| SAC trained by us | 51× | — / 46.37 | — / 33.08 | — / 32.24 | — / 21.44 | not started |
| Ablation: AE loss only | 5× | — / 56.55 | — / 40.34 | — / 42.08 | — / 27.98 | not started |
| Ablation: AE + LM loss | 5× | — / 62.04 | — / 45.80 | — / 47.26 | — / 32.25 | not started |

Original = the authors' released result files for the eval-only runs (identical to the paper's Table 1/2), and the paper's tables for runs we train ourselves. Pass bar: within 0.5 F1 for eval-only runs, within ~2 F1 for runs we train (training in bf16 on a different GPU count is not bit-reproducible).

## Per-subset: Released SAC checkpoint (eval only), 15×

**In-domain**

| Subset | F1 ours | F1 original | Δ | EM ours | EM original | Δ |
|---|---|---|---|---|---|---|
| SQuAD | 47.49 | 47.43 | +0.07 | 30.23 | 30.25 | -0.02 |
| NewsQA | 36.52 | 36.55 | -0.03 | 18.11 | 18.07 | +0.05 |
| TriviaQA-web | 61.27 | 61.13 | +0.14 | 52.20 | 52.19 | +0.01 |
| SearchQA | 68.97 | 68.97 | +0.00 | 56.73 | 56.76 | -0.04 |
| HotpotQA | 58.81 | 58.83 | -0.01 | 41.82 | 41.86 | -0.03 |
| NaturalQuestionsShort | 56.80 | 56.79 | +0.02 | 38.92 | 38.88 | +0.05 |
| **Average** | **54.98** | **54.95** | +0.03 | **39.67** | **39.67** | +0.00 |

**Out-of-domain**

| Subset | F1 ours | F1 original | Δ | EM ours | EM original | Δ |
|---|---|---|---|---|---|---|
| BioASQ | 41.17 | 41.31 | -0.15 | 28.86 | 28.66 | +0.20 |
| DROP | 36.65 | 36.72 | -0.07 | 27.54 | 27.48 | +0.07 |
| DuoRC.ParaphraseRC | 28.95 | 28.94 | +0.01 | 19.25 | 18.99 | +0.27 |
| RACE | 23.29 | 23.35 | -0.06 | 4.90 | 4.90 | +0.00 |
| RelationExtraction | 61.32 | 61.04 | +0.28 | 48.07 | 47.90 | +0.17 |
| TextbookQA | 44.11 | 44.21 | -0.10 | 28.41 | 28.21 | +0.20 |
| **Average** | **39.25** | **39.26** | -0.02 | **26.17** | **26.02** | +0.15 |

## Setup differences from the authors

| | Authors | Ours |
|---|---|---|
| GPU | 8× RTX 3090 (24 GB) | 1× RTX 2000 Ada (16 GB) |
| Base model source | `meta-llama/Llama-3.2-1B` | `unsloth/Llama-3.2-1B` @ `1d05b8ce9cd7` (ungated mirror; tokenizer verified identical on 201 cached examples) |
| Data caches | built by authors | the authors' released caches from `lx-Meteors/SAC` |
| Eval | greedy, batch 1 | same |
| Code change | — | none; only `MPLBACKEND=Agg` set so `plt.show()` doesn't block on a desktop |
