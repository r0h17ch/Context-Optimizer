"""Verification gate 1 (plan-v2 Part 5): with the new config keys absent, the patched
model must reproduce the pre-change SFT loss bit-identically."""
import importlib.util, json, os, sys, torch

HERE = os.path.dirname(os.path.abspath(__file__))
SAC_HOME = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, SAC_HOME)
sys.path.insert(0, os.path.join(SAC_HOME, "sft"))
os.chdir(os.path.join(SAC_HOME, "sft"))

from instruction_prepare_data import get_examples
from instruction_dataloader import get_dataset

BASE = sys.argv[1] if len(sys.argv) > 1 else "/tmp/sac_base/modeling_base.py"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 20

wd = os.path.join(SAC_HOME, "experiment/release_15x")
cfg = json.load(open(wd + "/output/config.json"))
tc, kc = cfg["sft_training_config"], cfg["sft_task_config"]
cfg["data_config"]["model_id"] = tc["model_id"]
tr, _ = get_examples(**cfg["data_config"])


def losses(module_path, module_name):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    m = mod.get_model(tc["model_id"], kc, 0)
    m = mod.load_adapter(m, save_path_and_name=wd + "/output/instruction_adapter.pt")
    m.train()
    out = []
    ds = get_dataset(kc["task_type"], tr[:N], 1)
    for inputs in torch.utils.data.DataLoader(ds, batch_size=None):
        inputs = {k: (v.to(0) if v is not None else None) for k, v in inputs.items()}
        out.append(m(inputs=inputs)["loss"].item())
    del m
    torch.cuda.empty_cache()
    return out


new = losses(os.path.join(SAC_HOME, "model/modeling.py"), "mod_new")
base = losses(BASE, "mod_base")
same = all(x == y for x, y in zip(new, base))
print("new :", [f"{x:.6f}" for x in new[:6]])
print("base:", [f"{x:.6f}" for x in base[:6]])
verdict = "BIT-IDENTICAL PASS" if same else "FAIL"
print(f"\nVERIFICATION 1 (defaults unchanged, {N} examples): {verdict}")
if not same:
    print("max abs diff:", max(abs(x - y) for x, y in zip(new, base)))
    sys.exit(1)
