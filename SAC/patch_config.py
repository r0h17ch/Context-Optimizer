import json, sys, os, torch
work_dir = sys.argv[1]
root = os.environ["SAC_ROOT"]
n_gpu = torch.cuda.device_count()
assert 16 % max(n_gpu, 1) == 0, "use 1, 2, 4 or 8 GPUs (CUDA_VISIBLE_DEVICES)"
n_gpu = max(n_gpu, 1)

model = f"{root}/models/Llama-3.2-1B"
for path in [f"{work_dir}/config.json", f"{work_dir}/output/config.json"]:
    if not os.path.exists(path):
        continue
    cfg = json.load(open(path))
    for k in ["pretrain_training_config", "sft_training_config"]:
        cfg[k]["model_id"] = model
        cfg[k]["device_count"] = n_gpu
        cfg[k]["batch_size_per_device"] = 1
        cfg[k]["gradient_accumulation_steps"] = 16 // n_gpu
    cfg["data_config"]["model_id"] = model
    cfg["data_config"]["dataset_repo"] = f"{root}/data/SlimPajama-6B"
    cfg["data_config"]["instruction_dataset_repo"] = f"{root}/data/mrqa-workshop_mrqa"
    json.dump(cfg, open(path, "w"), indent=4)
    print("patched", path, "| gpus:", n_gpu)
