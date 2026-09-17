"""G0 -- the FGD kill switch, plus the validity check plan-v2 T3/T4 asks for.

Two questions, answered before any GPU hours go into R1/R2:

1. **Operating point (train split).** How often does the frozen full-context teacher beat
   the compressed student on gold-answer NLL? That rate is the gate rate, and it is also
   the lambda used by the mass-matched uniform-KL control R1. Reported as a full
   gate-rate-vs-margin curve, not a single number, so the margin is chosen from evidence.

2. **Validity (eval split).** Does "teacher wins in NLL" actually coincide with "the
   student got the answer wrong"? The gate fires on NLL but the paper reports F1, so if
   the two are unrelated the gate is selecting noise. The released 15x run already holds
   per-example F1 for all 67,854 eval examples, so this costs one extra pass and no
   training.

Usage:
    python util/gate_diagnostic.py --work_dir experiment/release_15x \
        --n_train 2000 --n_eval 2000
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
from tqdm import tqdm

HERE = os.path.dirname(os.path.abspath(__file__))
SAC_HOME = os.path.dirname(HERE)
sys.path.insert(0, SAC_HOME)
sys.path.insert(0, os.path.join(SAC_HOME, "sft"))

from model.modeling import get_model, load_adapter  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--work_dir", default="experiment/release_15x",
                   help="run whose instruction_adapter.pt is the student")
    p.add_argument("--n_train", type=int, default=2000)
    p.add_argument("--n_eval", type=int, default=2000)
    p.add_argument("--out", default=None, help="default: <work_dir>/output/gate_diagnostic.json")
    return p.parse_args()


def tokenize_like_get_ids(example, tokenizer):
    """Rebuild one train-style example (context / question+answer / labels).

    Mirrors instruction_prepare_data.get_ids with split='train'. Needed because the cached
    *eval* tensors stop at the question -- they carry no gold answer, so no NLL can be
    computed from them.
    """
    answer = example["answers"][0]
    context = tokenizer(example["context"], add_special_tokens=False)["input_ids"]
    prompt = tokenizer(example["question"], add_special_tokens=False)["input_ids"]
    answer = tokenizer(answer, add_special_tokens=False)["input_ids"]

    context_ids = ([tokenizer.bos_token_id]
                   + tokenizer("### Context:\n", add_special_tokens=False)["input_ids"] + context)
    question_ids = (tokenizer("\n### Question:\n", add_special_tokens=False)["input_ids"] + prompt
                    + tokenizer("\n### Answer:\n", add_special_tokens=False)["input_ids"])
    answer_ids = answer + [tokenizer.eos_token_id]

    instruction_target = ([-100] * len(question_ids) + list(answer_ids))[1:]
    return {
        "input_ids": torch.LongTensor(context_ids),
        "lm_targets": torch.LongTensor(question_ids + answer_ids),
        "instruction_target": torch.LongTensor(instruction_target),
    }


def run_gate(model, examples, desc):
    rows = []
    for ex in tqdm(examples, desc=desc):
        inputs = {k: v.unsqueeze(0).to(model.device) for k, v in ex.items()}
        stats = model.gate_stats(inputs)
        if stats is not None:
            rows.append(stats)
    return rows


def margin_curve(margins_nats, thresholds):
    """Fraction of examples where nll_teacher < nll_student - m, for each m."""
    a = np.asarray(margins_nats)
    return {f"{t:g}": float((a > t).mean()) for t in thresholds}


def main():
    args = parse_args()
    work_dir = args.work_dir if os.path.isabs(args.work_dir) else os.path.join(SAC_HOME, args.work_dir)
    out_path = args.out or os.path.join(work_dir, "output", "gate_diagnostic.json")

    with open(os.path.join(work_dir, "output", "config.json")) as f:
        config = json.load(f)
    training_config = config["sft_training_config"]
    task_config = config["sft_task_config"]
    config["data_config"]["model_id"] = training_config["model_id"]

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(training_config["model_id"])

    model = get_model(training_config["model_id"], task_config, 0)
    model = load_adapter(model, save_path_and_name=os.path.join(work_dir, "output", "instruction_adapter.pt"))
    model.eval()

    report = {"work_dir": work_dir, "n_train_requested": args.n_train, "n_eval_requested": args.n_eval}
    thresholds = [0.0, 0.1, 0.25, 0.5, 1.0, 2.0]

    # ---------------- 1. operating point, train split ----------------
    os.chdir(os.path.join(SAC_HOME, "sft"))  # get_examples caches under ./output
    from instruction_prepare_data import get_examples
    train_examples, _ = get_examples(**config["data_config"])
    stride = max(1, len(train_examples) // args.n_train)
    sample = train_examples[::stride][:args.n_train]
    print(f"[G0] train: {len(sample)} examples (stride {stride} of {len(train_examples)})")
    rows = run_gate(model, sample, "train gate")

    nll_s = np.array([r["nll_student"] for r in rows])
    nll_t = np.array([r["nll_teacher"] for r in rows])
    margin = nll_s - nll_t  # > m  <=>  teacher beats student by more than m nats
    report["train"] = {
        "n": len(rows),
        "nll_student_mean": float(nll_s.mean()),
        "nll_teacher_mean": float(nll_t.mean()),
        "margin_mean": float(margin.mean()),
        "margin_percentiles": {str(q): float(np.percentile(margin, q)) for q in (5, 25, 50, 75, 95)},
        "gate_rate_by_margin": margin_curve(margin, thresholds),
    }
    print(json.dumps(report["train"], indent=2))

    # ---------------- 2. validity, eval split vs released per-example F1 ----------------
    results_path = os.path.join(work_dir, "output", "instruction_inference_results.json")
    if not os.path.exists(results_path):
        print(f"[G0] no {results_path}; skipping the F1 validity check")
        report["eval"] = None
    else:
        print(f"[G0] loading released per-example F1 ({os.path.getsize(results_path)/1e9:.2f} GB)...")
        with open(results_path) as f:
            released = json.load(f)
        stride = max(1, len(released) // args.n_eval)
        picked = released[::stride][:args.n_eval]
        del released
        print(f"[G0] eval: {len(picked)} examples (stride {stride})")

        tokenised, meta = [], []
        for ex in picked:
            try:
                tokenised.append(tokenize_like_get_ids(ex, tokenizer))
                meta.append({"subset": ex["subset"], "f1": ex["rouge-f1"], "em": ex["exact_match"]})
            except Exception as e:  # malformed example -- skip, don't abort the diagnostic
                print(f"[G0] skipping one eval example: {e}")

        rows = run_gate(model, tokenised, "eval gate")
        assert len(rows) == len(meta), "gate_stats dropped examples; index alignment broken"

        e_margin = np.array([r["nll_student"] - r["nll_teacher"] for r in rows])
        f1 = np.array([m["f1"] for m in meta])
        gate_at_0 = e_margin > 0.0

        wrong = f1 == 0.0
        right = f1 > 0.5
        report["eval"] = {
            "n": len(rows),
            "gate_rate_by_margin": margin_curve(e_margin, thresholds),
            "released_student_f1_mean": float(f1.mean()),
            "frac_f1_zero": float(wrong.mean()),
            # the check that matters: does the gate concentrate on the student's failures?
            "gate_rate_on_f1_zero": float(gate_at_0[wrong].mean()) if wrong.any() else None,
            "gate_rate_on_f1_gt_half": float(gate_at_0[right].mean()) if right.any() else None,
            "mean_f1_when_gated": float(f1[gate_at_0].mean()) if gate_at_0.any() else None,
            "mean_f1_when_not_gated": float(f1[~gate_at_0].mean()) if (~gate_at_0).any() else None,
            "pointbiserial_margin_vs_f1": float(np.corrcoef(e_margin, f1)[0, 1]),
        }
        print(json.dumps(report["eval"], indent=2))

    # ---------------- verdict ----------------
    rate = report["train"]["gate_rate_by_margin"]["0"]
    report["verdict"] = {
        "gate_rate_at_margin_0": rate,
        "kill_switch_triggered": bool(rate < 0.05),
        "suggested_lambda_for_R1_mass_matched": rate,
    }
    print("\n[G0] VERDICT:", json.dumps(report["verdict"], indent=2))
    if report["verdict"]["kill_switch_triggered"]:
        print("[G0] gate rate < 5%: FGD has no signal. Drop R1/R2; run R0 + R3 only.")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[G0] wrote {out_path}")


if __name__ == "__main__":
    main()
