import logging
import wandb
from path_config import BASE_PATH
import sys
sys.path.append(BASE_PATH)
import random
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import optimizer
from torch.utils.data import DataLoader, Dataset, IterableDataset
import torch.multiprocessing as mp
import os
import time
import json
from tqdm import tqdm
from transformers.models.llama.modeling_llama import LlamaForCausalLM
import argparse
import torch.distributed as dist
from model.modeling import CompressLLM
from model.lora import LinearLoraLayer
from torch.optim.lr_scheduler import LinearLR
from torch.optim.lr_scheduler import ConstantLR
from torch.optim.lr_scheduler import SequentialLR

def get_wsd_scheduler(optimizer, training_steps):
    W = 300
    S = training_steps - W

    warmup_scheduler = LinearLR(optimizer, start_factor=1/W, total_iters=W)
    stable_scheduler = ConstantLR(optimizer, factor=1.0, total_iters=S)

    milestones = [W]
    wsd_scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, stable_scheduler], milestones=milestones)

    return wsd_scheduler


def setup(rank, world_size, port):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = port

    # Initialize the distributed environment.
    # SAC_DDP_BACKEND lets a machine fall back to "gloo": NCCL calls nvmlInit(), which
    # fails outright when the loaded kernel module and the userspace NVIDIA libraries
    # are different versions. gloo talks to CUDA tensors directly and does not.
    backend = os.environ.get("SAC_DDP_BACKEND", "nccl")
    dist.init_process_group(backend, rank=rank, world_size=world_size)




def calculate_gradient_norm(model):
    total_norm = 0.0
    for param in model.parameters():
        if param.grad is not None:  # 检查梯度是否存在
            param_norm = param.grad.data.norm(2)  # 计算每个参数梯度的L2范数
            total_norm += param_norm.item() ** 2  # 累加每个梯度范数的平方
    total_norm = total_norm ** 0.5  # 求平方根得到整体L2范数
    return total_norm


def count_parameters(model, config):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    embedding_params = sum(
        p.numel() for name, p in model.named_parameters() if ('lm_head' in name or 'emb' in name))
    non_embedding_params = total_params - embedding_params

    config["Total_parameters"] = total_params
    config["Trainable_parameters"] = trainable_params
    config["Embedding_parameters"] = embedding_params
    config["non-Embedding_parameters"] = non_embedding_params

    logging.info(f"Total parameters: {total_params}")
    logging.info(f"Trainable parameters: {trainable_params}")
    logging.info(f"Embedding parameters: {embedding_params}")
    logging.info(f"non-Embedding parameters: {non_embedding_params}")

    embedding_percentage = (embedding_params / total_params) * 100
    logging.info(f"Embedding parameters percentage: {embedding_percentage:.2f}%")

    trainable_percentage = (trainable_params / total_params) * 100
    logging.info(f"Trainable parameters percentage: {trainable_percentage:.2f}%")

    # trainable_params = [name for name, param in model.named_parameters() if param.requires_grad]
    # print("Trainable parameters:")
    # for name in trainable_params:
    #     print(name)



def training_step(ddp_model, inputs, rank, accumulation_steps):
    # inputs = {key:value.to(rank) for key,value in inputs.items()}
    inputs = {key:(value.to(rank) if value is not None else None) for key,value in inputs.items()}
    output = ddp_model(inputs=inputs)
    loss = output["loss"]
    loss /= accumulation_steps
    loss.backward()
    # 计算当前的梯度范数
    # grad_norm = calculate_gradient_norm(ddp_model)
    # output["loss_info"]["grad_norm"] = grad_norm
    return output["loss_info"]








# ---------------------------------------------------------------------------
# Resume-from-checkpoint support (added for local reproduction runs).
# The data order is deterministic (no shuffling, no dropout), so restoring the
# adapter, optimizer, scheduler and step counter and then skipping the examples
# already consumed continues the run exactly where it stopped.
# ---------------------------------------------------------------------------

RESUME_CHECKPOINT = "resume_checkpoint.pt"


def add_resume_args(parser):
    parser.add_argument('--resume', action='store_true',
                        help=f'continue from <work_dir>/output/{RESUME_CHECKPOINT}')
    return parser


def resume_checkpoint_path(work_dir):
    return os.path.join(work_dir, "output", RESUME_CHECKPOINT)


def check_resume_args(args):
    """Fail fast in the launcher so a fresh run never silently discards saved progress."""
    path = resume_checkpoint_path(args.work_dir)
    if os.path.exists(path) and not args.resume:
        raise SystemExit(f"{path} exists from an interrupted run. "
                         f"Re-run with --resume to continue it, or delete the file to start over.")
    if args.resume and not os.path.exists(path):
        raise SystemExit(f"--resume given but {path} does not exist.")


def _run_signature(training_config, task_config, world_size, training_steps):
    # anything that changes data sharding or the optimisation trajectory
    return {
        "model_id": os.path.basename(os.path.normpath(training_config["model_id"])),
        "world_size": world_size,
        "total_batch_size": training_config["total_batch_size"],
        "batch_size_per_device": training_config["batch_size_per_device"],
        "gradient_accumulation_steps": training_config["gradient_accumulation_steps"],
        "learning_rate": training_config["learning_rate"],
        "training_steps": training_steps,
        "task_config": task_config,
    }


def save_resume_checkpoint(work_dir, model, optimizer, scheduler, step_num, info_list,
                           training_config, task_config, world_size, training_steps):
    """Write atomically, so a crash or power cut mid-save leaves the previous checkpoint intact."""
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    state = {
        "adapter": {n: t for n, t in model.state_dict().items() if n in trainable},
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "step_num": step_num,
        "info_list": info_list,
        "signature": _run_signature(training_config, task_config, world_size, training_steps),
        "rng": {"torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state(),
                "python": random.getstate()},
    }
    path = resume_checkpoint_path(work_dir)
    tmp = path + ".tmp"
    torch.save(state, tmp)
    os.replace(tmp, path)
    logging.info(f"[RESUME] checkpoint saved at micro-step {step_num} -> {path}")


def load_resume_checkpoint(work_dir, model, optimizer, scheduler,
                           training_config, task_config, world_size, training_steps):
    """Restore state in place; returns (step_num, info_list)."""
    path = resume_checkpoint_path(work_dir)
    state = torch.load(path, map_location="cpu")
    expected = _run_signature(training_config, task_config, world_size, training_steps)
    if state["signature"] != expected:
        diff = {k: (state["signature"].get(k), v) for k, v in expected.items()
                if state["signature"].get(k) != v}
        raise SystemExit(f"{path} was saved with a different setup (saved, current): {diff}")

    missing = set(n for n, p in model.named_parameters() if p.requires_grad) - set(state["adapter"])
    if missing:
        raise SystemExit(f"{path} is missing trainable parameters, e.g. {sorted(missing)[:3]}")
    model.load_state_dict({k: v.to(model.device) for k, v in state["adapter"].items()}, strict=False)
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    torch.set_rng_state(state["rng"]["torch"])
    torch.cuda.set_rng_state(state["rng"]["cuda"])
    random.setstate(state["rng"]["python"])
    logging.info(f"[RESUME] restored micro-step {state['step_num']} from {path}")
    return state["step_num"], state["info_list"]


def remove_resume_checkpoint(work_dir):
    path = resume_checkpoint_path(work_dir)
    if os.path.exists(path):
        os.remove(path)
