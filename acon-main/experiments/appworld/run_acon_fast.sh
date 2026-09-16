#!/usr/bin/env bash
# Fast ACON evaluation: 3 arms over an identical 8-task subset of train_history_tiny.
# Subset mixes short / medium / long tasks so compression has something to compress.
set -u
cd /home/cse-sdpl/acon-main/experiments/appworld
export PYTHONPATH=/home/cse-sdpl/acon-main/src
PY=/home/cse-sdpl/acon-main/.venv/bin/python
MAXITER=12
TASKS="07b42fd_1 27e1026_2 22cc237_3 22cc237_2 229360a_1 22cc237_1 34d9492_1 287e338_2"

run_arm () {
  local tag="$1"; local cfg="$2"
  local extra=""
  [ -n "$cfg" ] && extra="--co_config_path $cfg"
  echo "###### ARM $tag start=$(date -Is) cfg=${cfg:-NONE}"
  $PY run_all.py \
      --split train_history_tiny \
      --task_ids $TASKS \
      --model_name qwen2.5:14b \
      --tag "$tag" \
      --max_iter $MAXITER \
      $extra > "run_logs/$tag.log" 2>&1
  echo "###### ARM $tag DONE rc=$? end=$(date -Is)"
}

echo "===== ACON FAST TEST START $(date -Is) ====="
run_arm fast_base     ""
run_arm fast_hist_q7b configs/context_opt/ollama_qwen2.5_7b_history.yaml
run_arm fast_obs_q7b  configs/context_opt/ollama_qwen2.5_7b_obs.yaml
echo "===== ACON FAST TEST END $(date -Is) ====="
