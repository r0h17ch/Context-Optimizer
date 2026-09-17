"""Paired bootstrap over macro-averaged MRQA scores (plan-v2 decision rules).

Compares two runs on the *same* examples and reports the 95% CI of the difference. The
macro structure of the paper's Table 1/2 is preserved: examples are resampled with
replacement *within* each subset, each subset mean is recomputed, and those means are then
macro-averaged -- the same aggregation `util/paper_scores.py` does.

Reads the compact `per_example_scores*.json` sidecar written by the evaluator. If a run
predates the sidecar (the released 15x run does), it is derived once from the big
`instruction_inference_results*.json` and cached.

    python util/paired_bootstrap.py --variant experiment/r2_fgd --baseline experiment/r0_sac \
        --suffix _stride10

Caveat this tool cannot fix: the CI covers *example* variance only. Both runs are single
seed, so it says nothing about run-to-run variance (plan-v2 T1). It is meaningful here
because the pilot runs are paired at the training level -- same seed, same data order, same
initialisation, differing only in the loss term.
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SAC_HOME = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from paper_scores import ID_SUBSETS, OOD_SUBSETS  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--variant", required=True, help="work_dir of the run under test")
    p.add_argument("--baseline", required=True, help="work_dir it is compared against")
    p.add_argument("--suffix", default="", help="artefact suffix, e.g. _stride10")
    p.add_argument("--variant_suffix", default=None, help="override suffix for the variant")
    p.add_argument("--baseline_suffix", default=None, help="override suffix for the baseline")
    p.add_argument("--metric", default="rouge-f1", choices=["rouge-f1", "exact_match", "bleu4"])
    p.add_argument("--resamples", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None)
    return p.parse_args()


def load_scores(work_dir, suffix):
    """Per-example scores for one run, preferring the compact sidecar."""
    work_dir = work_dir if os.path.isabs(work_dir) else os.path.join(SAC_HOME, work_dir)
    sidecar = os.path.join(work_dir, "output", f"per_example_scores{suffix}.json")
    if os.path.exists(sidecar):
        with open(sidecar) as f:
            return json.load(f)

    big = os.path.join(work_dir, "output", f"instruction_inference_results{suffix}.json")
    if not os.path.exists(big):
        raise FileNotFoundError(f"neither {sidecar} nor {big} exists")
    print(f"[bootstrap] no sidecar; deriving once from {big} "
          f"({os.path.getsize(big)/1e9:.2f} GB)...")
    with open(big) as f:
        data = json.load(f)
    scores = [{"i": i, "subset": ex["subset"], "rouge-f1": ex["rouge-f1"],
               "exact_match": ex["exact_match"], "bleu4": ex["bleu4"]}
              for i, ex in enumerate(data)]
    del data
    with open(sidecar, "w") as f:
        json.dump(scores, f)
    print(f"[bootstrap] cached {sidecar}")
    return scores


def macro(values_by_subset, subsets):
    present = [s for s in subsets if len(values_by_subset.get(s, ())) > 0]
    if not present:
        return float("nan")
    return 100.0 * float(np.mean([values_by_subset[s].mean() for s in present]))


def group(scores, metric, subsets):
    by = {}
    for s in subsets:
        by[s] = np.array([r[metric] for r in scores if r["subset"] == s], dtype=float)
    return by


def paired_bootstrap(var_by, base_by, subsets, resamples, rng):
    """Resample within subset, macro-average, take the difference."""
    present = [s for s in subsets if len(var_by.get(s, ())) > 0]
    deltas = np.empty(resamples)
    for b in range(resamples):
        v_means, b_means = [], []
        for s in present:
            n = len(var_by[s])
            idx = rng.integers(0, n, n)          # same indices for both runs -> paired
            v_means.append(var_by[s][idx].mean())
            b_means.append(base_by[s][idx].mean())
        deltas[b] = 100.0 * (np.mean(v_means) - np.mean(b_means))
    return deltas


def report(name, var_scores, base_scores, subsets, metric, resamples, rng):
    var_by = group(var_scores, metric, subsets)
    base_by = group(base_scores, metric, subsets)
    v, b = macro(var_by, subsets), macro(base_by, subsets)
    deltas = paired_bootstrap(var_by, base_by, subsets, resamples, rng)
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    # two-sided bootstrap p: how often the resampled difference crosses zero
    # (clamped -- with a degenerate all-zero delta both tails are 1.0)
    p = min(1.0, 2 * min((deltas <= 0).mean(), (deltas >= 0).mean()))
    wins = bool(lo > 0 or hi < 0)
    n = sum(len(var_by[s]) for s in subsets if s in var_by)
    print(f"  {name:5s} n={n:6d}  variant {v:6.2f}  baseline {b:6.2f}  "
          f"delta {v - b:+6.2f}  95% CI [{lo:+.2f}, {hi:+.2f}]  p={p:.3f}  "
          f"{'SIGNIFICANT' if wins else 'not significant'}")
    return {"n": n, "variant": v, "baseline": b, "delta": v - b,
            "ci_low": float(lo), "ci_high": float(hi), "p_two_sided": float(p),
            "significant": wins}


def main():
    args = parse_args()
    v_suffix = args.variant_suffix if args.variant_suffix is not None else args.suffix
    b_suffix = args.baseline_suffix if args.baseline_suffix is not None else args.suffix

    var_scores = load_scores(args.variant, v_suffix)
    base_scores = load_scores(args.baseline, b_suffix)

    if len(var_scores) != len(base_scores):
        raise SystemExit(f"unpaired: {len(var_scores)} vs {len(base_scores)} examples -- the two "
                         f"runs were not evaluated on the same slice")
    mismatch = sum(1 for a, b in zip(var_scores, base_scores) if a["subset"] != b["subset"])
    if mismatch:
        raise SystemExit(f"unpaired: {mismatch} examples disagree on subset -- index alignment broken")

    rng = np.random.default_rng(args.seed)
    print(f"variant : {args.variant}{v_suffix}")
    print(f"baseline: {args.baseline}{b_suffix}")
    print(f"metric  : {args.metric}   resamples: {args.resamples}   paired, stratified by subset\n")
    out = {"variant": args.variant, "baseline": args.baseline, "metric": args.metric,
           "resamples": args.resamples}
    out["ID"] = report("ID", var_scores, base_scores, ID_SUBSETS, args.metric, args.resamples, rng)
    out["OOD"] = report("OOD", var_scores, base_scores, OOD_SUBSETS, args.metric, args.resamples, rng)

    print("\n  (single seed: this CI covers example variance only -- see plan-v2 T1)")
    if args.out:
        with open(args.out, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
