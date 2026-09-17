"""Print the pilot comparison table from whatever has finished so far."""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
SAC = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(SAC, "util"))
from paper_scores import macro, ID_SUBSETS, OOD_SUBSETS

RUNS = ["r0_sac", "r2_fgd", "r1_uniform_matched", "r1a_uniform_lambda1", "r3_tga_fgd", "r3_tga"]
for tag, sfx in [("5000 steps", "_stride10"), ("2500 steps", "_stride10_step2500")]:
    print(f"=== {tag} ===")
    for r in RUNS:
        o = f"{SAC}/experiment/{r}/output"
        i = macro(f"{o}/iid_subset_eval_results{sfx}.json", ID_SUBSETS, "rouge-f1", "exact_match")
        d = macro(f"{o}/ood_subset_eval_results{sfx}.json", OOD_SUBSETS, "f1", "em")
        if i and d:
            print(f"  {r:22s} ID F1 {i[0]:6.2f}  EM {i[1]:6.2f}   |   OOD F1 {d[0]:6.2f}  EM {d[1]:6.2f}")
        else:
            print(f"  {r:22s} (not available)")
    print()
