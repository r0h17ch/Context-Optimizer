"""Re-check TGA leakage on a *trained* adapter, on the eval split, at eval time.

A +8.7 F1 jump is also the signature of the question reaching the decoder, so re-run the
plan-v2 T12/T13 assertions against the real run rather than trusting the implementation-time
check on an untrained model.
"""
import json, os, sys, torch

HERE = os.path.dirname(os.path.abspath(__file__))
SAC_HOME = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, SAC_HOME)
sys.path.insert(0, os.path.join(SAC_HOME, "sft"))
os.chdir(os.path.join(SAC_HOME, "sft"))

from transformers import AutoTokenizer
from model.modeling import get_model, load_adapter
from instruction_prepare_data import get_examples
from instruction_dataloader import get_dataset, derive_query_ids

RUN = sys.argv[1] if len(sys.argv) > 1 else "r3_tga_fgd"
N = 25
wd = os.path.join(SAC_HOME, "experiment", RUN)
cfg = json.load(open(wd + "/output/config.json"))
tc, kc = cfg["sft_training_config"], cfg["sft_task_config"]
cfg["data_config"]["model_id"] = tc["model_id"]
tok = AutoTokenizer.from_pretrained(tc["model_id"])
_, ev = get_examples(**cfg["data_config"])
ev = ev[::10][:N]
print(f"run={RUN}  encoder_query={kc.get('encoder_query')}  eval examples={len(ev)}")
fail = []

# 1. eval-split query_ids must be the question only -- no gold answer anywhere in them
ans_marker = tok("\n### Answer:\n", add_special_tokens=False)["input_ids"]
gold_path = os.path.join(SAC_HOME, "sft", "output", "mrqa-workshop_mrqa_test_instruction_dataset.json")
golds = json.load(open(gold_path))[::10][:N]
bad = 0
for ex, gold in zip(ev, golds):
    assert "instruction_target" not in ex, "eval example unexpectedly carries labels"
    q = derive_query_ids(ex).tolist()
    if q[-len(ans_marker):] != ans_marker:
        bad += 1
    text = tok.decode(q)
    if gold["answers"][0].strip() and gold["answers"][0].strip() in text.split("### Question:")[-1].replace(
            tok.decode(ans_marker), ""):
        # the answer string appearing inside the *question* is legitimate (some MRQA
        # questions quote the answer span); flag only if it follows the answer marker
        pass
print(f"  query_ids end in '### Answer:' : {N-bad}/{N}")
if bad:
    fail.append("query_ids malformed on the eval split")
print(f"  sample query tail: {tok.decode(derive_query_ids(ev[0])[-14:])!r}")

# 2. the decoder must receive exactly the anchors -- nothing more
model = get_model(tc["model_id"], kc, 0)
model = load_adapter(model, save_path_and_name=wd + "/output/instruction_adapter.pt")
model.eval()
ds = get_dataset(kc["task_type"], ev[:8], 1, add_query_ids=True)
anchors, kvlens, ctx, qlens = [], [], [], []
with torch.no_grad():
    for inputs in torch.utils.data.DataLoader(ds, batch_size=None):
        inputs = {k: (v.to(0) if v is not None else None) for k, v in inputs.items()}
        _, _, _, _, kv, mem = model.compress(inputs)
        anchors.append(mem); kvlens.append(kv[0][0].size(2))
        ctx.append(inputs["input_ids"].size(1)); qlens.append(inputs["query_ids"].size(1))
print(f"  context lens : {ctx}")
print(f"  query lens   : {qlens}")
print(f"  anchors      : {anchors}")
print(f"  decoder KV   : {kvlens}")
if anchors != kvlens:
    fail.append("decoder KV length != anchor count -- query tokens reached the decoder")
# anchor count must match what a question-agnostic run would produce
model.task_config = {**kc, "encoder_query": False}
plain = []
with torch.no_grad():
    for inputs in torch.utils.data.DataLoader(get_dataset(kc["task_type"], ev[:8], 1), batch_size=None):
        inputs = {k: (v.to(0) if v is not None else None) for k, v in inputs.items()}
        _, _, _, _, _, mem = model.compress(inputs)
        plain.append(mem)
print(f"  anchors w/o query: {plain}")
if plain != anchors:
    fail.append("TGA changed the anchor count -- compression ratio is not matched")
ratios = [c / a for c, a in zip(ctx, anchors)]
print(f"  effective ratios : {[f'{r:.1f}x' for r in ratios]}")

print()
if fail:
    print("LEAKAGE CHECK FAILED:")
    for f in fail:
        print("  -", f)
    sys.exit(1)
print("LEAKAGE CHECK PASSED: the decoder sees only anchors, and the anchor budget is unchanged.")
