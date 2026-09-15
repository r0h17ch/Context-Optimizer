# SAC reproduction — local setup notes

Machine-specific state and corrections to `SAC_reproduction_plan.md`.
Everything below is done unless marked TODO.

## Environment

The machine had no conda, no pip, and no `python3.12-venv`, and `sudo` needs a
password. Miniconda was installed into `$HOME/miniconda3` (no sudo required).

The `SAC` env was created from **conda-forge**, not Anaconda's default
channels — recent conda refuses `defaults` until its ToS is accepted, and that
ToS carries commercial-use restrictions.

```bash
source ~/Context-Optimizer/SAC/env.sh   # conda activate SAC + SAC_ROOT + NCCL_P2P_DISABLE
```

`requirements.txt` installed unmodified, including `torch==2.2.0+cu121`. The
GPU is an **RTX 2000 Ada (sm_89)**, which cu121 supports — the plan's Blackwell
workaround does not apply here.

## Assets (`$SAC_ROOT` = `~/sac_assets`, ~30 GB)

| Path | Size | Verified |
|---|---|---|
| `data/mrqa-workshop_mrqa` | 1.6 G | loads, train/test/validation splits |
| `data/SlimPajama-6B` | 14 G | loads, `train` split streams |
| `release` (`lx-Meteors/SAC`) | 14 G | complete |
| `models/Llama-3.2-1B` | — | **TODO** — gated, needs `huggingface-cli login` |

## Corrections to the plan

**§3.3 Option 1 works, and it is the right choice.** The release ships all
five caches, so no tokenization is needed. They are already copied in:

- `pretrain/output/` — `train_…510-2040len.pt` (320000 ex), `eval_…` (1000 ex)
- `sft/output/` — `mrqa-workshop_mrqa_{train,eval}_Llama-3.2-1B_320000samples_instruction.pt`
  (320000 / 67854 ex) and `mrqa-workshop_mrqa_test_instruction_dataset.json` (67854 gold)

The 4.5 G `mrqa-workshop_mrqa_train_instruction_dataset.json` was deliberately
**not** copied: it is only read when regenerating the train `.pt`, which we are
not doing. Note that `get_examples` keys off the *train* `.pt` existing — if
that file is ever deleted, both it and the eval `.pt` are rebuilt from scratch.

**§3.5 compares the wrong number.** The `total_*` keys in
`*_subset_eval_results.json` are micro-averages over all examples. The paper
reports the **macro-average over the 6 subsets, ×100**. For 15× the micro ID F1
is 57.97 but the paper's figure is 54.95. Use:

```bash
python util/paper_scores.py --work_dir experiment/release_15x
```

which reproduces all four published SAC numbers exactly:

| Run | ID F1 / EM | OOD F1 / EM |
|---|---|---|
| release_15x | 54.95 / 39.67 | 39.26 / 26.02 |
| release_5x  | 63.63 / 46.95 | 47.72 / 32.30 |

Subset counts confirm the eval set matches the authors': ID 58221 and OOD 9633
examples, equal to `iid_test_samples_num` / `ood_test_samples_num` in their
reference JSONs.

**A GPU-count assertion the plan misses.** `pre_evaluator.py:103` requires
`len(eval_examples) % device_count == 0`. The pretrain eval set is 1000, so 1,
2, 4, 5 or 8 GPUs are fine — but not 3 or 6.

## GPU driver — TODO

`nvidia-driver-550` is installed but its kernel modules exist only for
`6.14.0-27-generic`, while the machine boots `7.0.0-31-generic`. So `nvidia-smi`
fails and `torch.cuda.is_available()` is False.

Fix chosen: **reboot into 6.14.0-27-generic** (GRUB → Advanced options), which
needs no package install. The alternative, `linux-modules-nvidia-580-7.0.0-31-generic`
plus `nvidia-driver-580`, needs sudo and upgrades the driver.

Configs are patched for `device_count=1`, `batch_size_per_device=1`,
`gradient_accumulation_steps=16` (total batch 16, as the paper).

## Resume from here

1. Reboot into 6.14.0-27; confirm `nvidia-smi`.
2. `huggingface-cli login` after accepting the Llama 3.2 license.
3. `huggingface-cli download meta-llama/Llama-3.2-1B --local-dir $SAC_ROOT/models/Llama-3.2-1B`
4. Phase A eval — see plan §3.4, then `util/paper_scores.py`.

## Resume from checkpoint (local addition)

The original trainers restart from step 0 when interrupted (and SFT saves nothing
until the end). Both `pretrain/pre_trainer.py` and `sft/instruction_trainer.py`
now write `<work_dir>/output/resume_checkpoint.pt` every `checkpoint_step`
optimizer steps (default 500, roughly 100 MB). It holds the adapter, AdamW and LR
scheduler state, step counter and loss history, and is written atomically
(temp file + rename), so a power cut during a save leaves the previous one intact.

```bash
# interrupted? run the exact same command again with --resume
python pre_trainer.py --work_dir $W --port 14572 --resume
python instruction_trainer.py --work_dir $W --port 14527 --resume
```

- Without `--resume`, a trainer refuses to start if a checkpoint exists, so a
  fresh run can never silently throw away progress. Delete the file to start over.
- Resume refuses if the setup changed (GPU count, batch sizes, LR, sample count,
  task config) — those change data sharding or the optimisation trajectory.
- At most `checkpoint_step` steps are redone after a crash.
- The checkpoint is deleted once the stage finishes; the final `adapter.pt` /
  `instruction_adapter.pt` are written exactly as before.
- The training data order is deterministic (no shuffling or dropout), so a
  resumed run follows the same trajectory as an uninterrupted one.
