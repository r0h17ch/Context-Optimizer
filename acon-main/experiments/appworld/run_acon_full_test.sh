#!/usr/bin/env bash
# Full ACON evaluation matrix on AppWorld.
# Agent model: qwen2.5:14b. Compressor: qwen2.5:7b (with a llama3.1:8b comparison arm).
set -u
cd /home/cse-sdpl/acon-main/experiments/appworld
export PYTHONPATH=/home/cse-sdpl/acon-main/src
PY=/home/cse-sdpl/acon-main/.venv/bin/python
LOGDIR=run_logs
MAXITER=30

run_arm () {
  local tag="$1"; local split="$2"; local model="$3"; local cfg="$4"
  echo "###### ARM $tag | split=$split | agent=$model | co_config=${cfg:-NONE} | start=$(date -Is)"
  local extra=""
  [ -n "$cfg" ] && extra="--co_config_path $cfg"
  $PY run_all.py \
      --split "$split" \
      --model_name "$model" \
      --tag "$tag" \
      --max_iter $MAXITER \
      $extra > "$LOGDIR/$tag.log" 2>&1
  echo "###### ARM $tag DONE rc=$? end=$(date -Is)"
}

echo "===== ACON FULL TEST START $(date -Is) ====="

# --- train_history_tiny (38 tasks) ---
run_arm base_none        train_history_tiny qwen2.5:14b ""
run_arm acon_hist_q7b    train_history_tiny qwen2.5:14b configs/context_opt/ollama_qwen2.5_7b_history.yaml
run_arm acon_hist_l8b    train_history_tiny qwen2.5:14b configs/context_opt/ollama_llama3.1_8b_history.yaml
run_arm acon_uni_q7b     train_history_tiny qwen2.5:14b configs/context_opt/ollama_qwen2.5_7b_unified.yaml

# --- train_obs_tiny (17 tasks) ---
run_arm base_none_obs    train_obs_tiny     qwen2.5:14b ""
run_arm acon_obs_q7b     train_obs_tiny     qwen2.5:14b configs/context_opt/ollama_qwen2.5_7b_obs.yaml

echo "===== ACON FULL TEST END $(date -Is) ====="
