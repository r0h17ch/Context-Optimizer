#!/usr/bin/env bash
# Full ACON evaluation on AppWorld, train_history_tiny (38 tasks), max_iter 30.
# Agent: qwen2.5:14b. Compressor: qwen2.5:7b (plus llama3.1:8b comparison arm).
#
# Each TASK runs in its own process with a 20 GB address-space cap. run_all.py leaks
# badly across tasks (a 38-task run OOM-killed the box at 59 GB RSS), so process-per-task
# bounds memory and confines a pathological task to itself instead of the whole arm.
set -u
cd /home/cse-sdpl/acon-main/experiments/appworld
export PYTHONPATH=/home/cse-sdpl/acon-main/src
PY=/home/cse-sdpl/acon-main/.venv/bin/python
MAXITER=30
MEMCAP_KB=20971520   # 20 GB virtual per task

TASKS=$(cat data/datasets/train_history_tiny.txt)

run_arm () {
  local tag="$1"; local cfg="$2"
  local extra=""
  [ -n "$cfg" ] && extra="--co_config_path $cfg"
  local t0=$(date +%s)
  echo "###### ARM $tag start=$(date -Is) cfg=${cfg:-NONE}"
  : > "run_logs/$tag.log"
  for tid in $TASKS; do
    if [ -f "outputs/qwen2.5_14b_$tag/train_history_tiny/task_$tid/results.json" ]; then
      echo "[skip done] $tid" >> "run_logs/$tag.log"; continue
    fi
    # stale/partial dir from a killed attempt would make run_all.py skip the task
    rm -rf "outputs/qwen2.5_14b_$tag/train_history_tiny/task_$tid"
    ( ulimit -v $MEMCAP_KB; \
      $PY run_all.py --split train_history_tiny --task_id "$tid" \
         --model_name qwen2.5:14b --tag "$tag" --max_iter $MAXITER $extra \
    ) >> "run_logs/$tag.log" 2>&1
    local rc=$?
    [ $rc -ne 0 ] && echo "[task $tid EXITED rc=$rc]" >> "run_logs/$tag.log"
    echo "[progress $tag] $tid rc=$rc $(date -Is)" >> run_logs/progress.log
  done
  local t1=$(date +%s)
  echo "{\"tag\":\"$tag\",\"seconds\":$((t1-t0))}" > "run_logs/$tag.time.json"
  echo "###### ARM $tag DONE end=$(date -Is) seconds=$((t1-t0))"
  $PY aggregate_results.py >> run_logs/aggregate.log 2>&1
}

echo "===== ACON FULL TEST START $(date -Is) ====="
run_arm base_none            ""
run_arm acon_hist_q7b        configs/context_opt/ollama_qwen2.5_7b_history.yaml
run_arm acon_hist_q7b_t4096  configs/context_opt/ollama_qwen2.5_7b_history_t4096.yaml
run_arm acon_obs_q7b         configs/context_opt/ollama_qwen2.5_7b_obs.yaml
run_arm acon_uni_q7b         configs/context_opt/ollama_qwen2.5_7b_unified.yaml
run_arm acon_hist_l8b        configs/context_opt/ollama_llama3.1_8b_history_t512.yaml
echo "===== ACON FULL TEST END $(date -Is) ====="
