#!/usr/bin/env bash
# RQ1: does ACON's combined-compression drop reproduce with open models?
# Four arms on the same 38 tasks (train_history_tiny), max_iter 30:
#   none / history only / observation only / history + observation.
# Agent qwen2.5:14b, compressor qwen2.5:7b, both served with num_ctx 16384.
#
# Why a new script: the Sep-2026 runs used Ollama's default num_ctx 4096, and 32 of 38
# baseline tasks had prompts above that, so they were silently truncated. Those numbers are
# superseded by these arms (tag suffix _c16k).
#
# Process-per-task with a 20 GB address-space cap (run_all.py leaks across tasks), and
# finished tasks are skipped, so the script can simply be re-launched after a crash/reboot.
set -u
cd /home/cse-sdpl/acon-main/experiments/appworld
export PYTHONPATH=/home/cse-sdpl/acon-main/src
PY=/home/cse-sdpl/acon-main/.venv/bin/python
MAXITER=30
MEMCAP_KB=20971520
AGENT=qwen2.5-14b-c16k
SPLIT=train_history_tiny
TASKS=$(cat data/datasets/$SPLIT.txt)
mkdir -p run_logs

# Ollama variants with a 16k window. FROM reuses the existing weights, so no extra disk.
make_variant () {
  local name="$1" base="$2"
  if ! ollama list | grep -q "^$name"; then
    printf 'FROM %s\nPARAMETER num_ctx 16384\n' "$base" > "/tmp/Modelfile.$name"
    ollama create "$name" -f "/tmp/Modelfile.$name"
  fi
}
make_variant qwen2.5-14b-c16k qwen2.5:14b
make_variant qwen2.5-7b-c16k  qwen2.5:7b

run_arm () {
  local tag="$1"; local cfg="$2"
  local extra=""
  [ -n "$cfg" ] && extra="--co_config_path $cfg"
  local out="outputs/${AGENT}_${tag}/$SPLIT"
  local t0=$(date +%s)
  echo "###### ARM $tag start=$(date -Is) cfg=${cfg:-NONE}"
  export ACON_REQ_LOG="$PWD/run_logs/$tag.requests.jsonl"
  for tid in $TASKS; do
    if [ -f "$out/task_$tid/results.json" ]; then
      echo "[skip done] $tid" >> "run_logs/$tag.log"; continue
    fi
    # a partial dir from a killed attempt would make run_all.py skip the task
    rm -rf "$out/task_$tid"
    ( export ACON_TASK_ID="$tid"; ulimit -v $MEMCAP_KB; \
      $PY run_all.py --split $SPLIT --task_id "$tid" \
         --model_name $AGENT --tag "$tag" --max_iter $MAXITER $extra \
    ) >> "run_logs/$tag.log" 2>&1
    local rc=$?
    [ $rc -ne 0 ] && echo "[task $tid EXITED rc=$rc]" >> "run_logs/$tag.log"
    echo "[progress $tag] $tid rc=$rc $(date -Is)" >> run_logs/progress_c16k.log
  done
  local t1=$(date +%s)
  echo "{\"tag\":\"$tag\",\"seconds\":$((t1-t0))}" >> "run_logs/$tag.time.json"
  echo "###### ARM $tag DONE end=$(date -Is) seconds=$((t1-t0))"
}

echo "===== RQ1 c16k START $(date -Is) ====="
run_arm base_none_c16k ""
run_arm acon_hist_c16k configs/context_opt/ollama_qwen2.5_7b_c16k_history.yaml
run_arm acon_obs_c16k  configs/context_opt/ollama_qwen2.5_7b_c16k_obs.yaml
run_arm acon_uni_c16k  configs/context_opt/ollama_qwen2.5_7b_c16k_unified.yaml
echo "===== RQ1 c16k END $(date -Is) ====="
