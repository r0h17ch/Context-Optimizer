#!/usr/bin/env python3
"""Aggregate ACON AppWorld experiment outputs into a single RESULTS.md."""
import json, os, glob, datetime
from pathlib import Path

OUT = Path("outputs")

ARMS = [
    ("fast_base",     "train_history_tiny", "Baseline (no context optimization)",   None),
    ("fast_hist_q7b", "train_history_tiny", "ACON history optimizer",               "qwen2.5:7b"),
    ("fast_obs_q7b",  "train_history_tiny", "ACON observation optimizer",           "qwen2.5:7b"),
]

def approx_tokens(s):
    return max(1, len(s) // 4) if s else 0

def load_arm(tag, split):
    root = OUT / f"qwen2.5_14b_{tag}" / split
    if not root.exists():
        return None
    summ_path = root / "experiment_summary.json"
    summ = json.load(open(summ_path)) if summ_path.exists() else None

    tasks = []
    for td in sorted(root.glob("task_*")):
        rp = td / "results.json"
        if not rp.exists():
            continue
        r = json.load(open(rp))
        tu = r.get("token_usage") or {}
        rec = {
            "task_id": r.get("task_id", td.name[5:]),
            "success": bool(r.get("success", False)),
            "reward": r.get("final_reward", 0.0),
            "iters": r.get("iterations", 0),
            "reason": r.get("termination_reason", "unknown"),
            "in_tok": tu.get("total_input_tokens", 0),
            "out_tok": tu.get("total_output_tokens", 0),
            "reqs": tu.get("total_requests", 0),
        }
        # compression stats from optimizer dumps
        comp_in = comp_out = comp_calls = 0
        for name in ("history_optimizer_history.json", "obs_optimizer_history.json"):
            p = td / name
            if not p.exists():
                continue
            try:
                entries = json.load(open(p))
            except Exception:
                continue
            for e in entries:
                # entries are [system_message, prompt, response, prompt_args]
                if isinstance(e, (list, tuple)) and len(e) >= 3:
                    comp_in += approx_tokens(e[1])
                    comp_out += approx_tokens(e[2])
                    comp_calls += 1
        rec.update(comp_in=comp_in, comp_out=comp_out, comp_calls=comp_calls)
        tasks.append(rec)
    return {"tag": tag, "split": split, "summary": summ, "tasks": tasks}

def agg(a):
    t = a["tasks"]
    n = len(t)
    if n == 0:
        return None
    ok = sum(x["success"] for x in t)
    ci, co = sum(x["comp_in"] for x in t), sum(x["comp_out"] for x in t)
    return {
        "n": n,
        "ok": ok,
        "sr": 100.0 * ok / n,
        "avg_reward": sum(x["reward"] for x in t) / n,
        "avg_iters": sum(x["iters"] for x in t) / n,
        "in_tok": sum(x["in_tok"] for x in t),
        "out_tok": sum(x["out_tok"] for x in t),
        "reqs": sum(x["reqs"] for x in t),
        "avg_in_tok": sum(x["in_tok"] for x in t) / n,
        "comp_calls": sum(x["comp_calls"] for x in t),
        "comp_in": ci,
        "comp_out": co,
        "comp_ratio": (co / ci) if ci else None,
        "time_s": (a["summary"] or {}).get("total_time_seconds"),
    }

rows = []
for tag, split, desc, comp in ARMS:
    a = load_arm(tag, split)
    if a is None:
        rows.append((tag, split, desc, comp, None, None))
        continue
    rows.append((tag, split, desc, comp, a, agg(a)))

L = []
W = L.append
W("# ACON on AppWorld — Full Evaluation Results")
W("")
W(f"_Generated {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')} on `{os.uname().nodename}`._")
W("")
W("## Setup")
W("")
W("| | |")
W("|---|---|")
W("| Host | `cse-sdpl@100.79.174.22` (NVIDIA RTX 2000 Ada, 16 GB) |")
W("| Serving | Ollama, OpenAI-compatible endpoint `http://localhost:11434/v1` |")
W("| Agent model | `qwen2.5:14b` |")
W("| Compressor model | `qwen2.5:7b` |")
W("| Benchmark | AppWorld |")
W("| Split | `train_history_tiny`, fixed 8-task subset (identical across all arms) |")
W("| Max iterations / task | 12 |")
W("| Seed | 42 |")
W("| Temperature | 0.0 |")
W("")
W("## Read this before the numbers")
W("")
W("Three constraints materially limit what this run can show. They are properties of how the run")
W("was configured and of the host, not of ACON.")
W("")
W("1. **Sample size is 8 tasks.** A one-task difference is a 12.5 percentage-point swing. None of")
W("   the success-rate differences below are statistically meaningful. Token counts, which are")
W("   aggregated over thousands of requests, are the trustworthy signal here; success rates are not.")
W("2. **`max_iter` was 12.** Three of the eight tasks were separately measured needing 14, 19 and 21")
W("   iterations to succeed, so they cannot be solved by any arm under this cap. They are marked")
W("   `capped` in the per-task tables. This depresses every arm equally but compresses the range in")
W("   which ACON could demonstrate a gain.")
W("3. **Ollama served both models with a 4096-token context window.** Baseline agent inputs reach")
W("   ~100k cumulative tokens on long tasks, so individual requests exceeding 4096 tokens were")
W("   silently truncated by the server. A gap between ACON and baseline therefore partly measures")
W("   ACON avoiding that truncation, rather than compression helping on its own terms. This is not a")
W("   clean context-length experiment.")
W("")
W("A conclusive run needs the full 38-task split at `max_iter` 30 with a context window at or above")
W("the model maximum. That is roughly the 11-hour configuration in `run_acon_full_test.sh`.")
W("")
W("## Headline results")
W("")
W("| Arm | Split | Method | Compressor | Tasks | Solved | Success rate | Avg reward | Avg iters | Agent input tokens | Wall clock |")
W("|---|---|---|---|---|---|---|---|---|---|---|")
for tag, split, desc, comp, a, g in rows:
    if g is None:
        W(f"| `{tag}` | {split} | {desc} | {comp or '—'} | — | — | _not run_ | — | — | — | — |")
        continue
    ts = g["time_s"]
    tstr = f"{int(ts//60)}m {int(ts%60)}s" if ts else "—"
    W(f"| `{tag}` | {split} | {desc} | {comp or '—'} | {g['n']} | {g['ok']} | **{g['sr']:.1f}%** | "
      f"{g['avg_reward']:.3f} | {g['avg_iters']:.1f} | {g['in_tok']:,} | {tstr} |")
W("")

W("## Context-compression activity")
W("")
W("Compression only fires when the accumulated context crosses the optimizer's token threshold "
  "(512 for history, 256 for observations). Token counts for compressor traffic are approximate "
  "(~4 chars/token) since the compressor runs outside the agent's metered path.")
W("")
W("| Arm | Compressor | Compression calls | Text in (approx tok) | Text out (approx tok) | Compression ratio |")
W("|---|---|---|---|---|---|")
for tag, split, desc, comp, a, g in rows:
    if g is None or comp is None:
        continue
    cr = f"{g['comp_ratio']:.3f}" if g["comp_ratio"] is not None else "—"
    W(f"| `{tag}` | {comp} | {g['comp_calls']} | {g['comp_in']:,} | {g['comp_out']:,} | {cr} |")
W("")

W("## Token economics (agent-side)")
W("")
W("| Arm | Requests | Input tokens | Output tokens | Total | Avg input/task |")
W("|---|---|---|---|---|---|")
for tag, split, desc, comp, a, g in rows:
    if g is None:
        continue
    W(f"| `{tag}` | {g['reqs']:,} | {g['in_tok']:,} | {g['out_tok']:,} | "
      f"{g['in_tok']+g['out_tok']:,} | {g['avg_in_tok']:,.0f} |")
W("")
W("All models are served locally through Ollama, so monetary cost is $0.00 for every arm.")
W("")

W("## Termination reasons")
W("")
W("| Arm | " + " | ".join(["completed", "max iterations", "other"]) + " |")
W("|---|---|---|---|")
for tag, split, desc, comp, a, g in rows:
    if g is None:
        continue
    done = sum(1 for x in a["tasks"] if x["reason"] == "task_completed")
    maxi = sum(1 for x in a["tasks"] if "max" in str(x["reason"]).lower())
    other = len(a["tasks"]) - done - maxi
    W(f"| `{tag}` | {done} | {maxi} | {other} |")
W("")

W("## Per-task detail")
W("")
for tag, split, desc, comp, a, g in rows:
    if g is None:
        continue
    W(f"### `{tag}` — {desc}" + (f", compressor `{comp}`" if comp else "") + f" ({split})")
    W("")
    W("| Task | Result | Reward | Iters | In tok | Out tok | Compress calls | Termination |")
    W("|---|---|---|---|---|---|---|---|")
    for x in a["tasks"]:
        cap = " (capped)" if x['iters'] >= 12 else ""
        W(f"| `{x['task_id']}` | {'PASS' if x['success'] else 'FAIL'} | {x['reward']:.2f} | {x['iters']}{cap} | "
          f"{x['in_tok']:,} | {x['out_tok']:,} | {x['comp_calls']} | {x['reason']} |")
    W("")

W("## Reproducing")
W("")
W("```bash")
W("cd /home/cse-sdpl/acon-main/experiments/appworld")
W("export PYTHONPATH=/home/cse-sdpl/acon-main/src")
W("")
W("# one arm, e.g. ACON history optimizer with the qwen2.5:7b compressor")
W("/home/cse-sdpl/acon-main/.venv/bin/python run_all.py \\")
W("    --split train_history_tiny \\")
W("    --model_name qwen2.5:14b \\")
W("    --tag acon_hist_q7b \\")
W("    --co_config_path configs/context_opt/ollama_qwen2.5_7b_history.yaml \\")
W("    --max_iter 30")
W("")
W("# the whole matrix")
W("./run_acon_full_test.sh")
W("")
W("# regenerate this file")
W("/home/cse-sdpl/acon-main/.venv/bin/python aggregate_results.py")
W("```")
W("")

Path("RESULTS.md").write_text("\n".join(L) + "\n")
print("Wrote RESULTS.md")
for tag, split, desc, comp, a, g in rows:
    status = "MISSING" if g is None else f"n={g['n']:3d} solved={g['ok']:3d} sr={g['sr']:5.1f}%"
    print(f"  {tag:16s} {status}")
