#!/usr/bin/env bash
# Phase A: evaluate the authors' released checkpoints. Usage: ./run_phase_a.sh 15x
set -euo pipefail
RATIO="${1:?usage: run_phase_a.sh <15x|5x>}"
source "$HOME/Context-Optimizer/SAC/env.sh"
cd "$SAC_HOME"

W="../experiment/release_${RATIO}"
[ -d "experiment/release_${RATIO}" ] || { echo "no such work dir: experiment/release_${RATIO}"; exit 1; }

python patch_config.py "experiment/release_${RATIO}"

cd "$SAC_HOME/sft"
# Section 2.7: stale eval files make the evaluator re-score old output instead of generating.
rm -f "$W/output/instruction_eval_info_list_"*.json

python instruction_evaluator.py --work_dir "$W" --batch_size 1 2>&1 | tee "$W/eval.log"
python ../util/evaluate_iid.py --work_dir "$W"
python ../util/evaluate_ood.py --work_dir "$W"

cd "$SAC_HOME"
echo "=== reproduced ==="
python util/paper_scores.py --work_dir "experiment/release_${RATIO}"
echo "=== authors' reference ==="
python - "$RATIO" <<'PY'
import json, sys
r = sys.argv[1]
ID = ["SQuAD","NewsQA","TriviaQA-web","SearchQA","HotpotQA","NaturalQuestionsShort"]
OOD = ["BioASQ","DROP","DuoRC.ParaphraseRC","RACE","RelationExtraction","TextbookQA"]
b = f"experiment/release_{r}/reference"
d = json.load(open(f"{b}/iid_subset_eval_results.json"))
o = json.load(open(f"{b}/ood_subset_eval_results.json"))
m = lambda d, s, k: 100*sum(d[f"{x}_{k}"] for x in s)/len(s)
print(f"  ID  F1 {m(d,ID,'rouge-f1'):.2f}  EM {m(d,ID,'exact_match'):.2f}")
print(f"  OOD F1 {m(o,OOD,'f1'):.2f}  EM {m(o,OOD,'em'):.2f}")
PY
