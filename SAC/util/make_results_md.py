"""Regenerate RESULTS.md: our reproduced scores vs. the authors' / paper numbers.

Usage (from the SAC folder):  python util/make_results_md.py
"""
import json, os
from datetime import datetime

SAC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ID = ["SQuAD", "NewsQA", "TriviaQA-web", "SearchQA", "HotpotQA", "NaturalQuestionsShort"]
OOD = ["BioASQ", "DROP", "DuoRC.ParaphraseRC", "RACE", "RelationExtraction", "TextbookQA"]

# (work dir, label, ratio, paper ID F1/EM, paper OOD F1/EM, reference dir or None)
RUNS = [
    ("release_15x", "Released SAC checkpoint (eval only)", "15×", (54.95, 39.67), (39.26, 26.02), "reference"),
    ("release_5x", "Released SAC checkpoint (eval only)", "5×", (63.63, 46.95), (47.72, 32.30), "reference"),
    ("sac_15x_mine", "SAC trained by us", "15×", (54.95, 39.67), (39.26, 26.02), None),
    ("sac_5x_mine", "SAC trained by us", "5×", (63.63, 46.95), (47.72, 32.30), None),
    ("sac_51x_mine", "SAC trained by us", "51×", (46.37, 33.08), (32.24, 21.44), None),
    ("sac_ae_only", "Ablation: AE loss only", "5×", (56.55, 40.34), (42.08, 27.98), None),
    ("sac_ae_lm", "Ablation: AE + LM loss", "5×", (62.04, 45.80), (47.26, 32.25), None),
]


def load(path):
    return json.load(open(path)) if os.path.exists(path) else None


def subset_scores(d, subsets, f1, em):
    return {s: (100 * d[f"{s}_{f1}"], 100 * d[f"{s}_{em}"]) for s in subsets}


def macro(scores):
    v = list(scores.values())
    return sum(x[0] for x in v) / len(v), sum(x[1] for x in v) / len(v)


def run_scores(base):
    iid, ood = load(f"{base}/iid_subset_eval_results.json"), load(f"{base}/ood_subset_eval_results.json")
    if not (iid and ood):
        return None
    return (subset_scores(iid, ID, "rouge-f1", "exact_match"),
            subset_scores(ood, OOD, "f1", "em"))


def diff(a, b):
    return f"{a - b:+.2f}"


def main():
    out = []
    w = out.append
    w("# SAC reproduction results — ours vs. original\n")
    w(f"_Generated {datetime.now():%Y-%m-%d %H:%M} by `util/make_results_md.py`. Re-run it after each eval to refresh._\n")
    w("Paper: *Autoencoding-Free Context Compression for LLMs via Contextual Semantic Anchors* (ICLR 2026). "
      "Base model Llama-3.2-1B, MRQA benchmark, 6 in-domain (ID) and 6 out-of-domain (OOD) subsets.\n")
    w("**How scores are computed.** Macro-average over the 6 subsets, ×100 — the convention the paper uses. "
      "(The `total_*` fields in the result JSONs are micro-averages and are not comparable to the paper.) "
      "F1 is the repo's ROUGE-based F1; EM is exact match.\n")

    w("## Summary\n")
    w("| Run | Ratio | ID F1 (ours / original) | ID EM (ours / original) | OOD F1 (ours / original) | OOD EM (ours / original) | Status |")
    w("|---|---|---|---|---|---|---|")
    details = []
    for name, label, ratio, p_id, p_ood, ref in RUNS:
        base = f"{SAC}/experiment/{name}"
        ours = run_scores(f"{base}/output")
        # compare against the authors' released JSONs where we have them, else the paper table
        orig = run_scores(f"{base}/{ref}") if ref else None
        o_id, o_ood = (macro(orig[0]), macro(orig[1])) if orig else (p_id, p_ood)
        if ref and ours and orig and ours == orig:
            ours = None  # output/ still holds the authors' files, our eval hasn't overwritten them yet
        if ours:
            m_id, m_ood = macro(ours[0]), macro(ours[1])
            cell = lambda a, b: f"**{a:.2f}** / {b:.2f} ({diff(a, b)})"
            worst = max(abs(m_id[0] - o_id[0]), abs(m_ood[0] - o_ood[0]))
            status = "✅ reproduced" if worst <= (0.5 if ref else 2.0) else "⚠️ off"
            w(f"| {label} | {ratio} | {cell(m_id[0], o_id[0])} | {cell(m_id[1], o_id[1])} | "
              f"{cell(m_ood[0], o_ood[0])} | {cell(m_ood[1], o_ood[1])} | {status} |")
            if orig:
                details.append((f"{label}, {ratio}", ours, orig))
        else:
            running = os.path.exists(f"{base}/phase_a.log") and ref
            status = "⏳ running" if running else "not started"
            w(f"| {label} | {ratio} | — / {o_id[0]:.2f} | — / {o_id[1]:.2f} | — / {o_ood[0]:.2f} | — / {o_ood[1]:.2f} | {status} |")
    w("")
    w("Original = the authors' released result files for the eval-only runs (identical to the paper's Table 1/2), "
      "and the paper's tables for runs we train ourselves. Pass bar: within 0.5 F1 for eval-only runs, "
      "within ~2 F1 for runs we train (training in bf16 on a different GPU count is not bit-reproducible).\n")

    for title, ours, orig in details:
        w(f"## Per-subset: {title}\n")
        for kind, subsets, idx in (("In-domain", ID, 0), ("Out-of-domain", OOD, 1)):
            w(f"**{kind}**\n")
            w("| Subset | F1 ours | F1 original | Δ | EM ours | EM original | Δ |")
            w("|---|---|---|---|---|---|---|")
            for s in subsets:
                (f, e), (fo, eo) = ours[idx][s], orig[idx][s]
                w(f"| {s} | {f:.2f} | {fo:.2f} | {diff(f, fo)} | {e:.2f} | {eo:.2f} | {diff(e, eo)} |")
            (f, e), (fo, eo) = macro(ours[idx]), macro(orig[idx])
            w(f"| **Average** | **{f:.2f}** | **{fo:.2f}** | {diff(f, fo)} | **{e:.2f}** | **{eo:.2f}** | {diff(e, eo)} |")
            w("")

    w("## Setup differences from the authors\n")
    w("| | Authors | Ours |")
    w("|---|---|---|")
    w("| GPU | 8× RTX 3090 (24 GB) | 1× RTX 2000 Ada (16 GB) |")
    w("| Base model source | `meta-llama/Llama-3.2-1B` | `unsloth/Llama-3.2-1B` @ `1d05b8ce9cd7` (ungated mirror; tokenizer verified identical on 201 cached examples) |")
    w("| Data caches | built by authors | the authors' released caches from `lx-Meteors/SAC` |")
    w("| Eval | greedy, batch 1 | same |")
    w("| Code change | — | none; only `MPLBACKEND=Agg` set so `plt.show()` doesn't block on a desktop |")
    w("")
    open(f"{SAC}/RESULTS.md", "w").write("\n".join(out))
    print(f"wrote {SAC}/RESULTS.md")


if __name__ == "__main__":
    main()
