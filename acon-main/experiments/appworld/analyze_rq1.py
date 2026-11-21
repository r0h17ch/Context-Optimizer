"""RQ1 analysis: does ACON's combined-compression drop reproduce with open models?

Reads the four c16k arms (none / history / observation / both) on AppWorld train + dev,
and reports success, paired McNemar tests, the interaction term
    I = d(both) - d(hist) - d(obs),   d(x) = rate(x) - rate(none)
with a paired bootstrap CI over tasks, plus token, peak-context, compression and
latency figures from the per-request logs (ACON_REQ_LOG).

    python analyze_rq1.py            # from experiments/appworld
"""
import glob
import json
import math
import random
import statistics as st
from collections import defaultdict
from pathlib import Path

AGENT = "qwen2.5-14b-c16k"
ARMS = [("none", "base_none_c16k"), ("hist", "acon_hist_c16k"),
        ("obs", "acon_obs_c16k"), ("both", "acon_uni_c16k")]
SPLITS = ["train", "dev"]
NUM_CTX = 16384
PAPER_I = -9.5  # ACON Table 14, gpt-4.1, UTCO


def load_results(tag):
    out = {}
    for split in SPLITS:
        for f in glob.glob(f"outputs/{AGENT}_{tag}/{split}/task_*/results.json"):
            d = json.load(open(f))
            out[d["task_id"]] = {"split": split, "success": bool(d["success"]),
                                 "steps": d["iterations"]}
    return out


def load_requests(tag):
    """Per-task requests from both logs, keeping only the last attempt of a task
    (a task interrupted by a machine shutdown was rerun; drop segments before a >1h gap)."""
    by_task = defaultdict(list)
    for path in (f"run_logs/{tag}.requests.jsonl", f"run_logs/{tag}_full.requests.jsonl"):
        if Path(path).exists():
            for line in open(path):
                x = json.loads(line)
                by_task[x["task"]].append(x)
    out = {}
    for task, reqs in by_task.items():
        reqs.sort(key=lambda x: x["ts"])
        cut = 0
        for i in range(1, len(reqs)):
            if reqs[i]["ts"] - reqs[i - 1]["ts"] > 3600:
                cut = i
        out[task] = reqs[cut:]
    return out


def mcnemar(a, b, tasks):
    wins = sum(b[t] and not a[t] for t in tasks)
    losses = sum(a[t] and not b[t] for t in tasks)
    m = wins + losses
    p = min(1.0, 2 * sum(math.comb(m, i) for i in range(min(wins, losses) + 1)) / 2 ** m) if m else 1.0
    return wins, losses, p


def interaction(S, tasks):
    r = {k: 100 * sum(S[k][t] for t in tasks) / len(tasks) for k in S}
    d = {k: r[k] - r["none"] for k in r}
    return d["both"] - d["hist"] - d["obs"], d


def bootstrap_ci(S, tasks, fn, n=10000, seed=0):
    rng = random.Random(seed)
    vals = sorted(fn(S, [rng.choice(tasks) for _ in tasks]) for _ in range(n))
    return vals[int(0.025 * n)], vals[int(0.975 * n)]


def main():
    R = {k: load_results(tag) for k, tag in ARMS}
    tasks = sorted(set.intersection(*[set(v) for v in R.values()]))
    S = {k: {t: R[k][t]["success"] for t in tasks} for k in R}
    n = len(tasks)
    print(f"AppWorld train+dev, {n} tasks common to all four arms\n")

    print(f"{'arm':5} {'solved':>8} {'rate':>6} {'train':>6} {'dev':>6} {'steps':>6} "
          f"{'peak':>6} {'in tok/task':>11} {'compr':>6} {'trunc':>5} {'LLM s/task':>10}")
    for k, tag in ARMS:
        rq = load_requests(tag)
        agent = {t: [x for x in rq.get(t, []) if "14b" in x["model"]] for t in tasks}
        comp = [x for t in tasks for x in rq.get(t, []) if "7b" in x["model"]]
        allreq = [x for t in tasks for x in agent[t]] + comp
        peak = st.mean(max([x["prompt_tokens"] or 0 for x in agent[t]] or [0]) for t in tasks)
        tin = sum(x["prompt_tokens"] or 0 for t in tasks for x in agent[t]) / n
        trunc = sum((x["prompt_tokens"] or 0) >= NUM_CTX - 400 for x in allreq)
        secs = sum(x["seconds"] for x in allreq) / n
        s = sum(S[k].values())
        by_split = {sp: [t for t in tasks if R[k][t]["split"] == sp] for sp in SPLITS}
        rate_sp = {sp: 100 * sum(S[k][t] for t in ts) / len(ts) for sp, ts in by_split.items()}
        steps = st.mean(R[k][t]["steps"] for t in tasks)
        print(f"{k:5} {s:>4}/{n} {100*s/n:5.1f}% {rate_sp['train']:5.1f}% {rate_sp['dev']:5.1f}% "
              f"{steps:6.1f} {peak:6.0f} {tin:11.0f} {len(comp):6} {trunc:5} {secs:10.1f}")

    I, d = interaction(S, tasks)
    lo, hi = bootstrap_ci(S, tasks, lambda S, ts: interaction(S, ts)[0])
    print(f"\nChange vs none (points): hist {d['hist']:+.1f}  obs {d['obs']:+.1f}  both {d['both']:+.1f}")
    print(f"Interaction I = {I:+.1f} points, 95% paired-bootstrap CI [{lo:+.1f}, {hi:+.1f}]"
          f"   (ACON paper: {PAPER_I:+.1f})")
    for k in ("hist", "obs", "both"):
        lo, hi = bootstrap_ci(S, tasks, lambda S, ts, k=k: interaction(S, ts)[1][k])
        print(f"  d({k}) 95% CI [{lo:+.1f}, {hi:+.1f}]")

    print("\nPaired McNemar (wins / losses of the second arm, exact two-sided p):")
    for a, b in [("none", "hist"), ("none", "obs"), ("none", "both"),
                 ("hist", "both"), ("obs", "both")]:
        w, l, p = mcnemar(S[a], S[b], tasks)
        print(f"  {b:4} vs {a:4}: +{w:2} / -{l:2}   p = {p:.3f}")


if __name__ == "__main__":
    main()
