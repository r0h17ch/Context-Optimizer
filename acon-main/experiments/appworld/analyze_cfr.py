"""Critical-Fact Retention (CFR) of history compression on AppWorld.

For each history compression of a task, the facts at stake are those present in the
history being compressed that the *uncompressed reference run* of the same task used in
its code at any point (so the yardstick does not depend on the arm being measured):

  - literals: string literals (>= 3 chars) and integers (|n| >= 10) in the reference code,
    e.g. emails, titles, ids, page sizes;
  - variables: names the reference code reads, when the compressed history assigns them.

A fact is retained if it appears verbatim in what the agent sees after the compression:
the <HISTORY_SUMMARY> block written into the next session's prompt, which includes the
state ledger when that is enabled. CFR = retained / at stake, pooled over compressions.

    python analyze_cfr.py            # from experiments/appworld
"""
import ast
import glob
import json
import re
import statistics as st
from collections import defaultdict

AGENT = "qwen2.5-14b-c16k"
SPLIT = "dev"
REFERENCE = "base_none_c16k"
ARMS = [
    ("hist naive@512", "acon_hist_c16k"),
    ("hist naive@4096", "acon_hist_t4096_c16k"),
    ("hist paper@512", "acon_hist_paper_c16k"),
    ("hist paper@512 r2", "acon_hist_paper_rep2_c16k"),
    ("hist paper@4096", "acon_hist_paper_t4096_c16k"),
    ("hist paper+ledger", "acon_hist_paper_ledger_c16k"),
    ("both naive", "acon_uni_c16k"),
    ("both paper", "acon_uni_paper_c16k"),
    ("both paper@4096", "acon_uni_paper_t4096_c16k"),
]


def sessions(path):
    h = json.load(open(path))
    return h if h and isinstance(h[0], list) else [h]


def code_blocks(sess_list):
    """Agent code of every assistant turn (all sessions)."""
    return [m["content"] for s in sess_list for m in s if m.get("role") == "assistant"]


def reference_facts(code_list):
    lits, names = set(), set()
    for code in code_list:
        try:
            tree = ast.parse(code)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant):
                v = node.value
                if isinstance(v, str) and len(v.strip()) >= 3 and "\n" not in v:
                    lits.add(v)
                elif isinstance(v, int) and not isinstance(v, bool) and abs(v) >= 10:
                    lits.add(str(v))
            elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                names.add(node.id)
    names -= {"apis", "print", "len", "range", "str", "int", "list", "dict", "sorted",
              "sum", "min", "max", "enumerate", "set", "True", "False", "None"}
    return lits, names


def assigned_names(history_text):
    """Names assigned in the ASSISTANT code blocks of a compressed history text."""
    out = set()
    for block in re.split(r"\n?(?:USER|ASSISTANT):\n", history_text):
        try:
            tree = ast.parse(block)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                out.add(node.id)
    return out


def shown_summary(session):
    """What the agent sees from the compression: the HISTORY_SUMMARY block of the
    session's first user message (the few-shot prompt before it is excluded)."""
    if len(session) < 2:
        return None
    content = session[1].get("content", "")
    if "<HISTORY_SUMMARY>" not in content:
        return None
    return content.split("<HISTORY_SUMMARY>", 1)[1]


def task_cfr(tag, task, ref_lits, ref_names):
    d = f"outputs/{AGENT}_{tag}/{SPLIT}/task_{task}/"
    try:
        hist = json.load(open(d + "history_optimizer_history.json"))
        sess = sessions(d + "llm_history.json")
    except (FileNotFoundError, json.JSONDecodeError):
        return None
    stake = defaultdict(int)
    kept = defaultdict(int)
    for k, entry in enumerate(hist):
        x = entry[3].get("history", "")
        if k + 1 >= len(sess):
            break
        summary = shown_summary(sess[k + 1])
        if summary is None:
            continue
        for lit in ref_lits:
            if lit in x:
                stake["literal"] += 1
                kept["literal"] += lit in summary
        for name in ref_names & assigned_names(x):
            stake["variable"] += 1
            kept["variable"] += bool(re.search(rf"\b{re.escape(name)}\b", summary))
    return stake, kept, len(hist)


def main():
    ref = {}
    for f in glob.glob(f"outputs/{AGENT}_{REFERENCE}/{SPLIT}/task_*/llm_history.json"):
        task = f.split("task_")[1].split("/")[0]
        ref[task] = reference_facts(code_blocks(sessions(f)))

    print(f"CFR on AppWorld {SPLIT}; facts defined by the uncompressed run ({len(ref)} tasks)\n")
    print(f"{'arm':18} {'compr':>6} {'facts':>6} {'CFR':>6} {'literals':>9} {'variables':>10} "
          f"{'CFR|success':>12} {'CFR|fail':>9}")
    pooled = []
    for label, tag in ARMS:
        tot_s, tot_k = defaultdict(int), defaultdict(int)
        n_compr, per_task = 0, []
        for task, (lits, names) in ref.items():
            r = task_cfr(tag, task, lits, names)
            if r is None:
                continue
            stake, kept, nc = r
            n_compr += nc
            for c in stake:
                tot_s[c] += stake[c]
                tot_k[c] += kept[c]
            s = sum(stake.values())
            if s:
                res = json.load(open(f"outputs/{AGENT}_{tag}/{SPLIT}/task_{task}/results.json"))
                per_task.append((sum(kept.values()) / s, bool(res["success"])))
        s_all, k_all = sum(tot_s.values()), sum(tot_k.values())
        frac = lambda c: f"{100 * tot_k[c] / tot_s[c]:.0f}%" if tot_s[c] else "-"
        ok = [c for c, y in per_task if y]
        bad = [c for c, y in per_task if not y]
        mean = lambda v: f"{100 * st.mean(v):.0f}%" if v else "-"
        print(f"{label:18} {n_compr:6} {s_all:6} {100 * k_all / max(s_all, 1):5.0f}% {frac('literal'):>9} "
              f"{frac('variable'):>10} {mean(ok):>12} {mean(bad):>9}")
        pooled += per_task

    # Does per-task CFR predict success? (point-biserial correlation over all arms)
    if len(pooled) > 2:
        xs = [c for c, _ in pooled]
        ys = [1.0 if y else 0.0 for _, y in pooled]
        mx, my = st.mean(xs), st.mean(ys)
        cov = sum((x - mx) * (y - my) for x, y in pooled_pairs(xs, ys)) / len(xs)
        r = cov / (st.pstdev(xs) * st.pstdev(ys)) if st.pstdev(xs) and st.pstdev(ys) else float("nan")
        print(f"\nPer-task CFR vs success, all arms pooled: r = {r:+.2f} (n = {len(pooled)} task-runs with compression)")


def pooled_pairs(xs, ys):
    return zip(xs, ys)


if __name__ == "__main__":
    main()
