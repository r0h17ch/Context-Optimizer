"""Verification gate 4 (plan-v2 Part 5): stride-10 scoring must reproduce the full-eval
macro scores to within ~1 point.

Re-scores the *existing* released 15x generations -- nothing is generated -- by slicing them
[::10] exactly as Evaluator.evaluate would, then running the normal scoring path.
"""
import json, os, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
SAC_HOME = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(SAC_HOME, "util"))
from paper_scores import macro, ID_SUBSETS, OOD_SUBSETS  # noqa: E402

WD = os.path.join(SAC_HOME, "experiment/release_15x")
OUT = os.path.join(WD, "output")
STRIDE = 10
PUBLISHED = (54.98, 39.25)

full = os.path.join(OUT, "instruction_eval_info_list_0.json")
strided = os.path.join(OUT, f"instruction_eval_info_list_0_stride{STRIDE}.json")

if not os.path.exists(strided):
    print(f"slicing {full} [::{STRIDE}] -> {strided}")
    with open(full) as f:
        gens = json.load(f)
    sliced = gens[::STRIDE]
    print(f"  {len(gens)} generations -> {len(sliced)}")
    with open(strided, "w") as f:
        json.dump(sliced, f, ensure_ascii=False)

# scoring path: generations already exist, so this only re-scores
env = dict(os.environ, MPLBACKEND="Agg")
for cmd in (
    [sys.executable, "instruction_evaluator.py", "--work_dir", "../experiment/release_15x",
     "--eval_stride", str(STRIDE)],
):
    print("+", " ".join(cmd))
    r = subprocess.run(cmd, cwd=os.path.join(SAC_HOME, "sft"), env=env,
                       capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-3000:]); print(r.stderr[-3000:]); sys.exit(1)
    print("  ", r.stdout.strip().splitlines()[-1])

for script in ("evaluate_iid.py", "evaluate_ood.py"):
    cmd = [sys.executable, os.path.join(SAC_HOME, "util", script),
           "--work_dir", WD, "--suffix", f"_stride{STRIDE}"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-2000:]); print(r.stderr[-2000:]); sys.exit(1)

iid = macro(f"{OUT}/iid_subset_eval_results_stride{STRIDE}.json", ID_SUBSETS, "rouge-f1", "exact_match")
ood = macro(f"{OUT}/ood_subset_eval_results_stride{STRIDE}.json", OOD_SUBSETS, "f1", "em")
print(f"\n  published full eval : ID F1 {PUBLISHED[0]:.2f}   OOD F1 {PUBLISHED[1]:.2f}")
print(f"  stride-{STRIDE} re-score  : ID F1 {iid[0]:.2f}   OOD F1 {ood[0]:.2f}")
d_id, d_ood = abs(iid[0] - PUBLISHED[0]), abs(ood[0] - PUBLISHED[1])
print(f"  delta               : ID {d_id:+.2f}      OOD {d_ood:+.2f}")
ok = d_id <= 1.5 and d_ood <= 1.5
print(f"\nVERIFICATION 4 (stride-{STRIDE} sanity): {'PASS' if ok else 'FAIL'}")
sys.exit(0 if ok else 1)
