"""Dev-split comparison: naive (shipped Prompt D.5) vs the paper's optimized UTCO guidelines
(Prompts D.7 / D.10), and history threshold 512 vs the paper's 4096.

Reads results.json for every arm and the per-request logs from both machines
(run_logs/*.requests.jsonl and run_logs/m1|m2/*.requests.jsonl).

    python analyze_dev.py            # from experiments/appworld
"""
import glob
import json
import math
import random
import statistics as st
from collections import defaultdict
from pathlib import Path

AGENT = "qwen2.5-14b-c16k"
SPLIT = "dev"
CAP = 30
# (label, tag, guideline, history threshold)
ARMS = [
    ("none",              "base_none_c16k",             "-",     "-"),
    ("hist naive@512",    "acon_hist_c16k",             "naive", "512"),
    ("hist paper@512",    "acon_hist_paper_c16k",       "paper", "512"),
    ("hist naive@4096",   "acon_hist_t4096_c16k",       "naive", "4096"),
    ("hist paper@4096",   "acon_hist_paper_t4096_c16k", "paper", "4096"),
    ("obs naive",         "acon_obs_c16k",              "naive", "-"),
    ("obs paper",         "acon_obs_paper_c16k",        "paper", "-"),
    ("both naive",        "acon_uni_c16k",              "naive", "512"),
    ("both paper",        "acon_uni_paper_c16k",        "paper", "512"),
    ("both paper@4096",   "acon_uni_paper_t4096_c16k",  "paper", "4096"),
    ("hist paper@512 r2", "acon_hist_paper_rep2_c16k",  "paper", "512"),
    ("hist paper+ledger", "acon_hist_paper_ledger_c16k", "paper", "512"),
    ("ledger errors only", "acon_hist_ledger_err_c16k",  "paper", "512"),
    ("ledger values only", "acon_hist_ledger_val_c16k",  "paper", "512"),
    ("hist keepfail",     "acon_hist_keepfail_c16k",    "keepfail", "512"),
    ("hist keepfail2",    "acon_hist_keepfail2_c16k",   "keepfail2", "512"),
]


def load_results(tag):
    out = {}
    for f in glob.glob(f"outputs/{AGENT}_{tag}/{SPLIT}/task_*/results.json"):
        d = json.load(open(f))
        out[d["task_id"]] = {"success": bool(d["success"]), "steps": d["iterations"]}
    return out


def load_requests(tag):
    """Requests per task from every log that mentions this tag, keeping only the last
    attempt of a task (segments before a >1h gap are earlier, interrupted attempts)."""
    paths = set(glob.glob(f"run_logs/{tag}*.requests.jsonl")) | set(glob.glob(f"run_logs/m*/{tag}*.requests.jsonl"))
    by_task = defaultdict(list)
    for p in paths:
        for line in open(p):
            x = json.loads(line)
            by_task[x["task"]].append(x)
    out = {}
    for t, reqs in by_task.items():
        reqs.sort(key=lambda x: x["ts"])
        cut = 0
        for i in range(1, len(reqs)):
            if reqs[i]["ts"] - reqs[i - 1]["ts"] > 3600:
                cut = i
        out[t] = reqs[cut:]
    return out


def mcnemar(a, b, tasks):
    wins = sum(b[t] and not a[t] for t in tasks)
    losses = sum(a[t] and not b[t] for t in tasks)
    m = wins + losses
    p = min(1.0, 2 * sum(math.comb(m, i) for i in range(min(wins, losses) + 1)) / 2 ** m) if m else 1.0
    return wins, losses, p


def main():
    R = {lab: load_results(tag) for lab, tag, _, _ in ARMS}
    tasks = sorted(set.intersection(*[set(v) for v in R.values()]))
    n = len(tasks)
    S = {lab: {t: R[lab][t]["success"] for t in tasks} for lab in R}
    print(f"AppWorld {SPLIT}: {n} tasks common to all {len(ARMS)} arms\n")
    print(f"{'arm':19} {'solved':>7} {'rate':>6} {'steps':>5} {'cap':>4} {'peak':>6} {'in tok/task':>11} {'compr':>6} {'LLM s/task':>10}")
    for lab, tag, _, _ in ARMS:
        rq = load_requests(tag)
        agent = {t: [x for x in rq.get(t, []) if "14b" in x["model"]] for t in tasks}
        comp = [x for t in tasks for x in rq.get(t, []) if "7b" in x["model"]]
        have = [t for t in tasks if agent[t]]
        peak = st.mean(max(x["prompt_tokens"] or 0 for x in agent[t]) for t in have) if have else float("nan")
        tin = sum(x["prompt_tokens"] or 0 for t in have for x in agent[t]) / max(len(have), 1)
        secs = sum(x["seconds"] for t in have for x in agent[t] + [y for y in rq.get(t, []) if "7b" in y["model"]]) / max(len(have), 1)
        s = sum(S[lab].values())
        steps = st.mean(R[lab][t]["steps"] for t in tasks)
        cap = sum(R[lab][t]["steps"] >= CAP for t in tasks)
        print(f"{lab:19} {s:>3}/{n} {100*s/n:5.1f}% {steps:5.1f} {cap:4} {peak:6.0f} {tin:11.0f} {len(comp):6} {secs:10.1f}"
              + ("" if len(have) == n else f"   (token stats on {len(have)} tasks)"))

    print("\nPaired McNemar (wins / losses of the first arm vs the second, exact two-sided p):")
    pairs = [("hist naive@512", "none"), ("hist paper@512", "none"), ("hist naive@4096", "none"),
             ("hist paper@4096", "none"), ("hist paper@512", "hist naive@512"),
             ("hist paper@4096", "hist naive@4096"), ("hist paper@4096", "hist paper@512"),
             ("obs paper", "obs naive"), ("both paper", "both naive"), ("both paper", "none"),
             ("both paper@4096", "none"), ("both paper@4096", "hist paper@4096"),
             ("hist paper@512 r2", "hist paper@512"), ("hist paper+ledger", "hist paper@512"),
             ("hist paper+ledger", "hist paper@512 r2"), ("hist paper+ledger", "none"),
             ("hist paper+ledger", "hist paper@4096"),
             ("ledger errors only", "hist paper@512"), ("ledger errors only", "hist paper@512 r2"),
             ("ledger values only", "hist paper@512"), ("ledger values only", "hist paper@512 r2"),
             ("ledger errors only", "ledger values only"),
             ("hist keepfail", "hist paper@512"), ("hist keepfail", "hist paper@512 r2"),
             ("hist keepfail2", "hist paper@512"), ("hist keepfail2", "hist paper@512 r2"),
             ("hist keepfail2", "hist keepfail"), ("hist keepfail2", "none"),
             ("hist keepfail2", "hist naive@512"), ("hist keepfail2", "hist paper@4096")]
    for a, b in pairs:
        w, l, p = mcnemar(S[b], S[a], tasks)
        print(f"  {a:19} vs {b:18}: +{w:2} / -{l:2}   p = {p:.4f}")

    r = {lab: 100 * sum(S[lab].values()) / n for lab in S}
    for g in ("naive", "paper"):
        d = {k: r[f"{k} {g}" if k != "hist" else f"hist {g}@512"] - r["none"] for k in ("hist", "obs", "both")}
        print(f"\nInteraction ({g} guideline, @512): I = d(both) - d(hist) - d(obs) = "
              f"{d['both']:+.1f} - ({d['hist']:+.1f}) - ({d['obs']:+.1f}) = {d['both'] - d['hist'] - d['obs']:+.1f} pts")


if __name__ == "__main__":
    main()
