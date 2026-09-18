# ACON on AppWorld — Full Evaluation Results

_Generated 2026-09-16 17:38 on `cse-sdpl-HP-Z4-G5-Workstation-Desktop-PC`._

## Setup

| | |
|---|---|
| Host | `cse-sdpl@100.79.174.22` (NVIDIA RTX 2000 Ada, 16 GB) |
| Serving | Ollama, OpenAI-compatible endpoint `http://localhost:11434/v1` |
| Agent model | `qwen2.5:14b` |
| Compressor models | `qwen2.5:7b` (primary), `llama3.1:8b` (comparison arm) |
| Benchmark | AppWorld |
| Split | `train_history_tiny`, full split (38 tasks) |
| Max iterations / task | 30 |
| Seed | 42 |
| Temperature | 0.0 |

## Limitations

State these in the report rather than letting a reader find them.

1. **Context window capped at 4096 tokens.** Ollama served both models with `num_ctx=4096`, the
   largest that fits with the 14b agent and 7b compressor co-resident in 16 GB of VRAM. Baseline
   agent inputs exceed this on long tasks, so those requests were truncated server-side. Part of
   any ACON gain is therefore ACON keeping context under the window, not compression helping in
   the abstract. This is the single biggest caveat on these numbers.
2. **Compression thresholds deviate from the paper.** The stock thresholds (4096 history / 1024
   observation) almost never fire on this split, whose history text runs 57-700 tokens. The
   primary arms use 512 / 256 so the optimizer actually engages; the `acon_hist_q7b_t4096` arm
   retains the paper default to quantify how often stock settings fire here.
3. **Single seed (42), single split.** No variance estimate across seeds. Differences of one or
   two tasks out of 38 should not be treated as significant.

## Headline results

| Arm | Split | Method | Compressor | Tasks done | Solved | Success rate | Avg reward | Avg iters | Agent input tokens | Wall clock |
|---|---|---|---|---|---|---|---|---|---|---|
| `base_none` | train_history_tiny | Baseline (no context optimization) | — | 38/38 | 12 | **31.6%** | 0.316 | 23.7 | 2,972,376 | 121m 52s |
| `acon_hist_q7b` | train_history_tiny | ACON history (threshold 512) | qwen2.5:7b | — | — | _not run_ | — | — | — | — |
| `acon_hist_q7b_t4096` | train_history_tiny | ACON history (threshold 4096, paper default) | qwen2.5:7b | — | — | _not run_ | — | — | — | — |
| `acon_obs_q7b` | train_history_tiny | ACON observation (threshold 256) | qwen2.5:7b | — | — | _not run_ | — | — | — | — |
| `acon_uni_q7b` | train_history_tiny | ACON unified (history + observation) | qwen2.5:7b | — | — | _not run_ | — | — | — | — |
| `acon_hist_l8b` | train_history_tiny | ACON history (threshold 512) | llama3.1:8b | — | — | _not run_ | — | — | — | — |

## Context-compression activity

Compression only fires when the accumulated context crosses the optimizer's token threshold (512 or 4096 for history, 256 for observations -- see arm names). Token counts for compressor traffic are approximate (~4 chars/token) since the compressor runs outside the agent's metered path.

| Arm | Compressor | Compression calls | Text in (approx tok) | Text out (approx tok) | Compression ratio |
|---|---|---|---|---|---|

## Token economics (agent-side)

| Arm | Requests | Input tokens | Output tokens | Total | Avg input/task |
|---|---|---|---|---|---|
| `base_none` | 902 | 2,972,376 | 190,214 | 3,162,590 | 78,220 |

All models are served locally through Ollama, so monetary cost is $0.00 for every arm.

## Termination reasons

| Arm | completed | max iterations | other |
|---|---|---|---|
| `base_none` | 38 | 0 | 0 |

## Per-task detail

### `base_none` — Baseline (no context optimization) (train_history_tiny)

| Task | Result | Reward | Iters | In tok | Out tok | Compress calls | Termination |
|---|---|---|---|---|---|---|---|
| `07b42fd_1` | PASS | 1.00 | 4 | 10,080 | 695 | 0 | task_completed |
| `07b42fd_2` | PASS | 1.00 | 4 | 9,544 | 293 | 0 | task_completed |
| `07b42fd_3` | PASS | 1.00 | 3 | 6,967 | 281 | 0 | task_completed |
| `229360a_1` | FAIL | 0.00 | 30 (capped) | 85,961 | 5,268 | 0 | task_completed |
| `229360a_3` | FAIL | 0.00 | 30 (capped) | 106,782 | 8,599 | 0 | task_completed |
| `22cc237_1` | FAIL | 0.00 | 30 (capped) | 99,613 | 1,221 | 0 | task_completed |
| `22cc237_2` | PASS | 1.00 | 19 | 55,126 | 1,819 | 0 | task_completed |
| `22cc237_3` | PASS | 1.00 | 14 | 41,228 | 1,178 | 0 | task_completed |
| `27e1026_1` | FAIL | 0.00 | 30 (capped) | 100,533 | 4,994 | 0 | task_completed |
| `27e1026_2` | PASS | 1.00 | 5 | 13,164 | 639 | 0 | task_completed |
| `287e338_2` | FAIL | 0.00 | 30 (capped) | 93,070 | 2,179 | 0 | task_completed |
| `34d9492_1` | FAIL | 0.00 | 30 (capped) | 102,655 | 2,244 | 0 | task_completed |
| `34d9492_2` | FAIL | 0.00 | 30 (capped) | 102,002 | 1,977 | 0 | task_completed |
| `34d9492_3` | FAIL | 0.00 | 30 (capped) | 86,668 | 16,907 | 0 | task_completed |
| `3c13f5a_1` | FAIL | 0.00 | 30 (capped) | 86,657 | 1,458 | 0 | task_completed |
| `3c13f5a_2` | PASS | 1.00 | 21 | 73,849 | 8,805 | 0 | task_completed |
| `3c13f5a_3` | FAIL | 0.00 | 30 (capped) | 97,552 | 3,802 | 0 | task_completed |
| `60d0b5b_1` | FAIL | 0.00 | 30 (capped) | 112,455 | 12,344 | 0 | task_completed |
| `60d0b5b_2` | FAIL | 0.00 | 30 (capped) | 108,251 | 3,056 | 0 | task_completed |
| `76f2c72_2` | FAIL | 0.00 | 30 (capped) | 93,897 | 3,407 | 0 | task_completed |
| `771d8fc_2` | FAIL | 0.00 | 30 (capped) | 114,053 | 15,801 | 0 | task_completed |
| `771d8fc_3` | FAIL | 0.00 | 30 (capped) | 106,412 | 11,941 | 0 | task_completed |
| `7d7fbf6_2` | FAIL | 0.00 | 30 (capped) | 92,864 | 4,194 | 0 | task_completed |
| `7d7fbf6_3` | FAIL | 0.00 | 30 (capped) | 90,880 | 3,289 | 0 | task_completed |
| `82e2fac_1` | FAIL | 0.00 | 30 (capped) | 109,944 | 768 | 0 | task_completed |
| `82e2fac_2` | FAIL | 0.00 | 30 (capped) | 104,336 | 10,160 | 0 | task_completed |
| `82e2fac_3` | FAIL | 0.00 | 30 (capped) | 98,254 | 4,279 | 0 | task_completed |
| `b7a9ee9_2` | PASS | 1.00 | 7 | 20,295 | 637 | 0 | task_completed |
| `b7a9ee9_3` | FAIL | 0.00 | 30 (capped) | 88,395 | 5,489 | 0 | task_completed |
| `ccb4494_1` | PASS | 1.00 | 4 | 9,717 | 438 | 0 | task_completed |
| `ce359b5_3` | PASS | 1.00 | 14 | 45,909 | 7,235 | 0 | task_completed |
| `d0b1f43_2` | FAIL | 0.00 | 30 (capped) | 107,886 | 9,425 | 0 | task_completed |
| `e3d6c94_1` | FAIL | 0.00 | 30 (capped) | 101,186 | 4,409 | 0 | task_completed |
| `e3d6c94_3` | PASS | 1.00 | 22 | 74,345 | 7,983 | 0 | task_completed |
| `e7a10f8_1` | FAIL | 0.00 | 30 (capped) | 99,269 | 6,412 | 0 | task_completed |
| `e7a10f8_2` | FAIL | 0.00 | 30 (capped) | 103,785 | 8,225 | 0 | task_completed |
| `e7a10f8_3` | FAIL | 0.00 | 30 (capped) | 106,542 | 8,158 | 0 | task_completed |
| `e85d92a_1` | PASS | 1.00 | 5 | 12,250 | 205 | 0 | task_completed |

## Reproducing

```bash
cd /home/cse-sdpl/acon-main/experiments/appworld
export PYTHONPATH=/home/cse-sdpl/acon-main/src

# one arm, e.g. ACON history optimizer with the qwen2.5:7b compressor
/home/cse-sdpl/acon-main/.venv/bin/python run_all.py \
    --split train_history_tiny \
    --model_name qwen2.5:14b \
    --tag acon_hist_q7b \
    --co_config_path configs/context_opt/ollama_qwen2.5_7b_history.yaml \
    --max_iter 30

# the whole matrix
./run_acon_full_test.sh

# regenerate this file
/home/cse-sdpl/acon-main/.venv/bin/python aggregate_results.py
```

