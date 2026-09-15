"""Print paper-style (Table 1/2) macro-averaged scores for a work_dir.

The paper reports the macro-average over the 6 ID and 6 OOD MRQA subsets,
multiplied by 100 -- not the `total_*` micro-averages stored in the JSONs.
"""
import json, os, sys, argparse

ID_SUBSETS = ["SQuAD", "NewsQA", "TriviaQA-web", "SearchQA", "HotpotQA",
              "NaturalQuestionsShort"]
OOD_SUBSETS = ["BioASQ", "DROP", "DuoRC.ParaphraseRC", "RACE",
               "RelationExtraction", "TextbookQA"]


def macro(path, subsets, f1_key, em_key):
    if not os.path.exists(path):
        return None
    d = json.load(open(path))
    f1 = [d[f"{s}_{f1_key}"] for s in subsets]
    em = [d[f"{s}_{em_key}"] for s in subsets]
    return 100 * sum(f1) / len(f1), 100 * sum(em) / len(em)


def report(work_dir):
    out = os.path.join(work_dir, "output")
    iid = macro(f"{out}/iid_subset_eval_results.json", ID_SUBSETS,
                "rouge-f1", "exact_match")
    ood = macro(f"{out}/ood_subset_eval_results.json", OOD_SUBSETS,
                "f1", "em")
    print(work_dir)
    print(f"  ID  F1 {iid[0]:.2f}  EM {iid[1]:.2f}" if iid else "  ID  (missing)")
    print(f"  OOD F1 {ood[0]:.2f}  EM {ood[1]:.2f}" if ood else "  OOD (missing)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--work_dir", required=True)
    report(p.parse_args().work_dir)
