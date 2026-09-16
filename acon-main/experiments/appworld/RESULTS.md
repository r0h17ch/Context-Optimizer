# ACON on AppWorld — Full Evaluation Results

_Generated 2026-09-16 14:25 on `cse-sdpl-HP-Z4-G5-Workstation-Desktop-PC`._

## Setup

| | |
|---|---|
| Host | `cse-sdpl@100.79.174.22` (NVIDIA RTX 2000 Ada, 16 GB) |
| Serving | Ollama, OpenAI-compatible endpoint `http://localhost:11434/v1` |
| Agent model | `qwen2.5:14b` |
| Compressor model | `qwen2.5:7b` |
| Benchmark | AppWorld |
| Split | `train_history_tiny`, fixed 8-task subset (identical across all arms) |
| Max iterations / task | 12 |
| Seed | 42 |
| Temperature | 0.0 |

## Read this before the numbers

Three constraints materially limit what this run can show. They are properties of how the run
was configured and of the host, not of ACON.

1. **Sample size is 8 tasks.** A one-task difference is a 12.5 percentage-point swing. None of
   the success-rate differences below are statistically meaningful. Token counts, which are
   aggregated over thousands of requests, are the trustworthy signal here; success rates are not.
2. **`max_iter` was 12.** Three of the eight tasks were separately measured needing 14, 19 and 21
   iterations to succeed, so they cannot be solved by any arm under this cap. They are marked
   `capped` in the per-task tables. This depresses every arm equally but compresses the range in
   which ACON could demonstrate a gain.
3. **Ollama served both models with a 4096-token context window.** Baseline agent inputs reach
   ~100k cumulative tokens on long tasks, so individual requests exceeding 4096 tokens were
   silently truncated by the server. A gap between ACON and baseline therefore partly measures
   ACON avoiding that truncation, rather than compression helping on its own terms. This is not a
   clean context-length experiment.

A conclusive run needs the full 38-task split at `max_iter` 30 with a context window at or above
the model maximum. That is roughly the 11-hour configuration in `run_acon_full_test.sh`.

## Headline results

| Arm | Split | Method | Compressor | Tasks | Solved | Success rate | Avg reward | Avg iters | Agent input tokens | Wall clock |
|---|---|---|---|---|---|---|---|---|---|---|
| `fast_base` | train_history_tiny | Baseline (no context optimization) | — | 8 | 1 | **12.5%** | 0.125 | 11.0 | 269,935 | 15m 53s |
| `fast_hist_q7b` | train_history_tiny | ACON history optimizer | qwen2.5:7b | 8 | 2 | **25.0%** | 0.250 | 10.8 | 232,749 | 14m 42s |
| `fast_obs_q7b` | train_history_tiny | ACON observation optimizer | qwen2.5:7b | 8 | 1 | **12.5%** | 0.125 | 11.0 | 261,422 | 14m 33s |

## Context-compression activity

Compression only fires when the accumulated context crosses the optimizer's token threshold (512 for history, 256 for observations). Token counts for compressor traffic are approximate (~4 chars/token) since the compressor runs outside the agent's metered path.

| Arm | Compressor | Compression calls | Text in (approx tok) | Text out (approx tok) | Compression ratio |
|---|---|---|---|---|---|
| `fast_hist_q7b` | qwen2.5:7b | 31 | 34,320 | 5,117 | 0.149 |
| `fast_obs_q7b` | qwen2.5:7b | 6 | 7,792 | 972 | 0.125 |

## Token economics (agent-side)

| Arm | Requests | Input tokens | Output tokens | Total | Avg input/task |
|---|---|---|---|---|---|
| `fast_base` | 88 | 269,935 | 18,045 | 287,980 | 33,742 |
| `fast_hist_q7b` | 86 | 232,749 | 15,300 | 248,049 | 29,094 |
| `fast_obs_q7b` | 88 | 261,422 | 16,591 | 278,013 | 32,678 |

All models are served locally through Ollama, so monetary cost is $0.00 for every arm.

## Termination reasons

| Arm | completed | max iterations | other |
|---|---|---|---|
| `fast_base` | 8 | 0 | 0 |
| `fast_hist_q7b` | 8 | 0 | 0 |
| `fast_obs_q7b` | 8 | 0 | 0 |

## Per-task detail

### `fast_base` — Baseline (no context optimization) (train_history_tiny)

| Task | Result | Reward | Iters | In tok | Out tok | Compress calls | Termination |
|---|---|---|---|---|---|---|---|
| `07b42fd_1` | PASS | 1.00 | 4 | 10,195 | 805 | 0 | task_completed |
| `229360a_1` | FAIL | 0.00 | 12 (capped) | 40,847 | 8,329 | 0 | task_completed |
| `22cc237_1` | FAIL | 0.00 | 12 (capped) | 41,541 | 612 | 0 | task_completed |
| `22cc237_2` | FAIL | 0.00 | 12 (capped) | 35,166 | 765 | 0 | task_completed |
| `22cc237_3` | FAIL | 0.00 | 12 (capped) | 35,211 | 720 | 0 | task_completed |
| `27e1026_2` | FAIL | 0.00 | 12 (capped) | 36,349 | 3,878 | 0 | task_completed |
| `287e338_2` | FAIL | 0.00 | 12 (capped) | 33,956 | 1,902 | 0 | task_completed |
| `34d9492_1` | FAIL | 0.00 | 12 (capped) | 36,670 | 1,034 | 0 | task_completed |

### `fast_hist_q7b` — ACON history optimizer, compressor `qwen2.5:7b` (train_history_tiny)

| Task | Result | Reward | Iters | In tok | Out tok | Compress calls | Termination |
|---|---|---|---|---|---|---|---|
| `07b42fd_1` | PASS | 1.00 | 4 | 10,081 | 725 | 0 | task_completed |
| `229360a_1` | FAIL | 0.00 | 12 (capped) | 34,274 | 5,442 | 10 | task_completed |
| `22cc237_1` | FAIL | 0.00 | 12 (capped) | 33,243 | 733 | 3 | task_completed |
| `22cc237_2` | FAIL | 0.00 | 12 (capped) | 32,608 | 624 | 2 | task_completed |
| `22cc237_3` | PASS | 1.00 | 10 | 26,291 | 755 | 1 | task_completed |
| `27e1026_2` | FAIL | 0.00 | 12 (capped) | 33,199 | 4,926 | 9 | task_completed |
| `287e338_2` | FAIL | 0.00 | 12 (capped) | 30,635 | 729 | 2 | task_completed |
| `34d9492_1` | FAIL | 0.00 | 12 (capped) | 32,418 | 1,366 | 4 | task_completed |

### `fast_obs_q7b` — ACON observation optimizer, compressor `qwen2.5:7b` (train_history_tiny)

| Task | Result | Reward | Iters | In tok | Out tok | Compress calls | Termination |
|---|---|---|---|---|---|---|---|
| `07b42fd_1` | PASS | 1.00 | 4 | 10,080 | 724 | 0 | task_completed |
| `229360a_1` | FAIL | 0.00 | 12 (capped) | 40,050 | 6,219 | 0 | task_completed |
| `22cc237_1` | FAIL | 0.00 | 12 (capped) | 36,261 | 1,105 | 3 | task_completed |
| `22cc237_2` | FAIL | 0.00 | 12 (capped) | 36,260 | 705 | 2 | task_completed |
| `22cc237_3` | FAIL | 0.00 | 12 (capped) | 35,247 | 899 | 0 | task_completed |
| `27e1026_2` | FAIL | 0.00 | 12 (capped) | 34,480 | 2,804 | 0 | task_completed |
| `287e338_2` | FAIL | 0.00 | 12 (capped) | 33,956 | 1,902 | 0 | task_completed |
| `34d9492_1` | FAIL | 0.00 | 12 (capped) | 35,088 | 2,233 | 1 | task_completed |

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

