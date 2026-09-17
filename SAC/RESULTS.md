# SAC reproduction results — ours vs. original

_Generated 2026-09-17 11:09 by `util/make_results_md.py`. Re-run it after each eval to refresh._

Paper: *Autoencoding-Free Context Compression for LLMs via Contextual Semantic Anchors* (ICLR 2026). Base model Llama-3.2-1B, MRQA benchmark, 6 in-domain (ID) and 6 out-of-domain (OOD) subsets.

**How scores are computed.** Macro-average over the 6 subsets, ×100 — the convention the paper uses. (The `total_*` fields in the result JSONs are micro-averages and are not comparable to the paper.) F1 is the repo's ROUGE-based F1; EM is exact match.

## Summary

| Run | Ratio | ID F1 (ours / original) | ID EM (ours / original) | OOD F1 (ours / original) | OOD EM (ours / original) | Status |
|---|---|---|---|---|---|---|
| Released SAC checkpoint (eval only) | 15× | **54.98** / 54.95 (+0.03) | **39.67** / 39.67 (+0.00) | **39.25** / 39.26 (-0.02) | **26.17** / 26.02 (+0.15) | ✅ reproduced |
| Released SAC checkpoint (eval only) | 5× | **63.70** / 63.63 (+0.07) | **47.04** / 46.95 (+0.10) | **47.71** / 47.72 (-0.01) | **32.46** / 32.30 (+0.16) | ✅ reproduced |
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

## Per-subset: Released SAC checkpoint (eval only), 5×

**In-domain**

| Subset | F1 ours | F1 original | Δ | EM ours | EM original | Δ |
|---|---|---|---|---|---|---|
| SQuAD | 65.58 | 65.37 | +0.21 | 45.05 | 44.83 | +0.22 |
| NewsQA | 49.53 | 49.39 | +0.14 | 27.33 | 27.14 | +0.19 |
| TriviaQA-web | 65.15 | 65.06 | +0.09 | 56.03 | 55.93 | +0.10 |
| SearchQA | 70.00 | 69.99 | +0.01 | 58.13 | 58.06 | +0.08 |
| HotpotQA | 67.36 | 67.41 | -0.05 | 50.28 | 50.28 | +0.00 |
| NaturalQuestionsShort | 64.55 | 64.56 | -0.01 | 45.45 | 45.44 | +0.01 |
| **Average** | **63.70** | **63.63** | +0.07 | **47.04** | **46.95** | +0.10 |

**Out-of-domain**

| Subset | F1 ours | F1 original | Δ | EM ours | EM original | Δ |
|---|---|---|---|---|---|---|
| BioASQ | 44.47 | 44.66 | -0.19 | 31.52 | 31.45 | +0.07 |
| DROP | 41.79 | 41.55 | +0.24 | 31.14 | 30.87 | +0.27 |
| DuoRC.ParaphraseRC | 39.27 | 39.48 | -0.20 | 26.78 | 26.92 | -0.13 |
| RACE | 30.38 | 30.53 | -0.15 | 6.38 | 6.23 | +0.15 |
| RelationExtraction | 77.81 | 77.87 | -0.06 | 65.57 | 65.40 | +0.17 |
| TextbookQA | 52.53 | 52.24 | +0.29 | 33.40 | 32.93 | +0.47 |
| **Average** | **47.71** | **47.72** | -0.01 | **32.46** | **32.30** | +0.16 |

## Pilot: SAC × ACON (5k-step SFT, 1/10 eval)

Every arm starts from the authors' released 15× pretrain adapter and is paired at the training level — same seed, same data order, same initialisation — differing only in the loss term. `get_wsd_scheduler` is warmup(300) + constant with no decay phase, so a 5k-step run is exactly the first 5k steps of the 20k schedule. Absolute numbers therefore sit below the released 20k checkpoint; compare **within** this table only.

| Run | Role | ID F1 | OOD F1 | ΔID vs R0 (95% CI) | ΔOOD vs R0 (95% CI) |
|---|---|---|---|---|---|
| R0 · SAC SFT (matched control) | — | 51.37 | 38.62 | — | — |
| R2 · + FGD (failure-gated KL, λ=1) | proposed | 51.15 | 38.47 | -0.23 [-0.95, +0.47] | -0.14 [-2.04, +1.77] |
| R1 · + uniform KL (λ=ḡ, mass-matched) | ablation | 51.06 | 38.17 | -0.32 [-1.02, +0.44] | -0.45 [-2.11, +1.44] |
| R1a · + uniform KL (λ=1, KV-Distill style) | literature | — | — | not started | not started |
| R3 · + TGA + FGD (query-aware) | separate track | — | — | not started | not started |
| R3b · + TGA alone (query-aware) | separate track | — | — | not started | not started |

**\*** = 95% CI of the paired bootstrap (1,000 resamples, stratified by subset) excludes 0. The CI covers example variance only — all arms are single-seed (plan-v2 T1).

**Supporting ablation — R2 vs R1 (same KL mass, different examples):** ID +0.09 [-0.62, +0.80], OOD +0.30 [-1.71, +2.31]. This isolates *which* examples receive the KL from *how much* KL there is.

### G0 — is the failure gate real?

Gold-answer NLL on 2000 train examples: **student 0.75, teacher 2.42** — the zero-shot full-context teacher is much worse *on average*, so uniform KL should hurt. Gate rate at margin 0: **9.5%**.

But the gate fires where it should: on 2000 eval examples it selects **19.2%** of the examples the released student got completely wrong (F1=0) versus **1.5%** of those it got right. Mean F1 **0.180 when gated** vs **0.616 when not** (point-biserial -0.287).

### Realized gate rate during training

| Run | mean | first quarter | last quarter |
|---|---|---|---|
| R2 · + FGD (failure-gated KL, λ=1) | 16.1% | 22.6% | 13.3% |

A rate collapsing toward 0 means the gate annealed itself away and R2 degenerates into R0 (plan-v2 T11).

## Setup differences from the authors

| | Authors | Ours |
|---|---|---|
| GPU | 8× RTX 3090 (24 GB) | 1× RTX 2000 Ada (16 GB) |
| Base model source | `meta-llama/Llama-3.2-1B` | `unsloth/Llama-3.2-1B` @ `1d05b8ce9cd7` (ungated mirror; tokenizer verified identical on 201 cached examples) |
| Data caches | built by authors | the authors' released caches from `lx-Meteors/SAC` |
| Eval | greedy, batch 1 | same |
| Code change | — | none for the reproduction rows above (only `MPLBACKEND=Agg`). The pilot rows add FGD/TGA behind config keys that are absent by default — verified bit-identical to the pre-change code on 20 SFT examples. |
