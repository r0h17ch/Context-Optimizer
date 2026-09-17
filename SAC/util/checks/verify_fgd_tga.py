"""Verification gates 2 and 3 (plan-v2 Part 5): teacher alignment, TGA no-leakage, and an
FGD forward smoke test."""
import copy, json, os, sys, torch, torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
SAC_HOME = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, SAC_HOME)
sys.path.insert(0, os.path.join(SAC_HOME, "sft"))
os.chdir(os.path.join(SAC_HOME, "sft"))

from transformers import AutoTokenizer
from model.modeling import get_model, load_adapter
from instruction_prepare_data import get_examples
from instruction_dataloader import get_dataset, derive_query_ids

N = 12
wd = os.path.join(SAC_HOME, "experiment/release_15x")
cfg = json.load(open(wd + "/output/config.json"))
tc, kc = cfg["sft_training_config"], cfg["sft_task_config"]
cfg["data_config"]["model_id"] = tc["model_id"]
tok = AutoTokenizer.from_pretrained(tc["model_id"])
tr, _ = get_examples(**cfg["data_config"])
failures = []


def batches(task_config, n=N, add_query_ids=False):
    ds = get_dataset(task_config["task_type"], tr[:n], 1, add_query_ids=add_query_ids)
    for inputs in torch.utils.data.DataLoader(ds, batch_size=None):
        yield {k: (v.to(0) if v is not None else None) for k, v in inputs.items()}


# ---------------------------------------------------------------- gate 2: alignment
# The teacher slice starts at C = len(input_ids). If that offset were wrong the gold-answer
# NLL would jump, so score the neighbouring offsets too and require C to win.
print("=" * 70)
print("GATE 2: teacher/student logit alignment")
model = get_model(tc["model_id"], kc, 0)
model = load_adapter(model, save_path_and_name=wd + "/output/instruction_adapter.pt")
model.eval()

nll_by_offset = {-1: [], 0: [], 1: []}
first_token_hits = 0
checked = 0
with torch.no_grad():
    for inputs in batches(kc):
        C = inputs["input_ids"].size(1)
        teacher_ids = torch.cat([inputs["input_ids"], inputs["lm_targets"][:, :-1]], dim=1)
        hidden = model.decoder.model(input_ids=teacher_ids).last_hidden_state
        target = inputs["instruction_target"].view(-1)
        mask = target != -100
        gold = target[mask]
        for off in nll_by_offset:
            sl = hidden[0, C + off:][: target.size(0)]
            if sl.size(0) != target.size(0):
                continue
            logits = model.decoder.lm_head(sl[mask]).float()
            nll_by_offset[off].append(F.cross_entropy(logits, gold).item())
        # first answer token: does the aligned teacher predict the gold token?
        logits0 = model.decoder.lm_head(hidden[0, C:][: target.size(0)][mask]).float()
        first_token_hits += int(logits0[0].argmax().item() == gold[0].item())
        checked += 1

means = {o: sum(v) / len(v) for o, v in nll_by_offset.items() if v}
print(f"  mean gold-answer NLL by teacher slice offset: "
      + ", ".join(f"{o:+d}: {m:.4f}" for o, m in sorted(means.items())))
print(f"  teacher predicts gold first answer token on {first_token_hits}/{checked} examples")
if means.get(0) is not None and means[0] == min(means.values()):
    print("  PASS: offset 0 (== len(input_ids)) has the lowest NLL -> alignment correct")
else:
    failures.append("teacher alignment: offset 0 is not the best slice")
    print("  FAIL: offset 0 is not the best slice")

# ---------------------------------------------------------------- gate 3: TGA leakage
print("=" * 70)
print("GATE 3: TGA -- query conditioning without leakage")
ans_marker = tok("\n### Answer:\n", add_special_tokens=False)["input_ids"]
bad_query = 0
for ex in tr[:200]:
    q = derive_query_ids(ex).tolist()
    if q[-len(ans_marker):] != ans_marker:
        bad_query += 1
    n_q = int((ex["instruction_target"] == -100).sum()) + 1
    if len(q) != n_q or ex["lm_targets"][:n_q].tolist() != q:
        bad_query += 1
if bad_query:
    failures.append(f"query_ids malformed on {bad_query}/200 examples")
    print(f"  FAIL: {bad_query}/200 query_ids malformed")
else:
    print("  PASS: 200/200 query_ids end in '### Answer:\\n' and carry no answer tokens")
    print(f"        sample tail: {tok.decode(derive_query_ids(tr[0])[-12:])!r}")

kc_tga = copy.deepcopy(kc)
kc_tga["encoder_query"] = True
model.task_config = kc_tga
anchors_plain, anchors_tga, kv_lens = [], [], []
with torch.no_grad():
    for inputs in batches(kc_tga, n=6, add_query_ids=True):
        model.task_config = kc
        ids_p, _, _, _, _, mem_p = model.compress({k: v for k, v in inputs.items() if k != "query_ids"})
        model.task_config = kc_tga
        ids_t, _, _, _, kv_t, mem_t = model.compress(inputs)
        anchors_plain.append(mem_p)
        anchors_tga.append(mem_t)
        kv_lens.append(kv_t[0][0].size(2))
ok_ratio = anchors_plain == anchors_tga
ok_kv = all(k == a for k, a in zip(kv_lens, anchors_tga))
print(f"  anchors without query: {anchors_plain}")
print(f"  anchors with query:    {anchors_tga}")
print(f"  decoder KV lengths:    {kv_lens}")
if ok_ratio:
    print("  PASS: anchor count unchanged by TGA -> compression ratio still 15x")
else:
    failures.append("TGA changed the anchor count")
    print("  FAIL: TGA changed the anchor count")
if ok_kv:
    print("  PASS: decoder KV length == anchor count -> question never reaches the decoder")
else:
    failures.append("TGA leaked query tokens into the decoder KV")
    print("  FAIL: decoder KV longer than anchor count -- query leaked")

# ---------------------------------------------------------------- FGD forward smoke
print("=" * 70)
print("FGD forward smoke (gated and ungated)")
for gate_mode, lam in [("failure", 1.0), ("none", 1.0)]:
    kc_d = copy.deepcopy(kc)
    kc_d.update({"distill_weight": lam, "distill_gate": gate_mode, "distill_margin": 0.0})
    model.task_config = kc_d
    torch.cuda.reset_peak_memory_stats()
    rows = []
    for inputs in batches(kc_d):
        out = model(inputs=inputs)
        rows.append(out["loss_info"])
        out["loss"].backward()
        model.zero_grad(set_to_none=True)
    kl = [r["kl"] for r in rows]
    ce = [r["lm_loss"] for r in rows]
    gates = [r["gate"] for r in rows]
    finite = all(k == k and abs(k) != float("inf") for k in kl)
    print(f"  gate={gate_mode:8s} mean CE {sum(ce)/len(ce):.4f}  mean KL {sum(kl)/len(kl):.4f}  "
          f"gate_rate {sum(gates)/len(gates):.2f}  kl/ce {(sum(kl)/len(kl))/(sum(ce)/len(ce)):.2f}")
    print(f"                   peak VRAM {torch.cuda.max_memory_allocated()/1024**3:.2f} GiB, "
          f"KL finite: {finite}")
    if not finite:
        failures.append(f"non-finite KL with gate={gate_mode}")
    if gate_mode == "failure" and not all(g in (0.0, 1.0) for g in gates):
        failures.append("gate is not binary")

print("=" * 70)
if failures:
    print("FAILURES:")
    for f in failures:
        print("  -", f)
    sys.exit(1)
print("ALL VERIFICATION GATES PASSED")
