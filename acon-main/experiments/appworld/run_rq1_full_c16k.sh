#!/usr/bin/env bash
# RQ1 at full size: the same four arms as run_rq1_c16k.sh on AppWorld train (90) + dev (57),
# 147 tasks. The 39 train_history_tiny tasks (a subset of train) were already run with
# identical code and config on 2026-10-29; they are copied in once and skipped here.
# Same per-task process, memory cap and skip-if-done resume as run_rq1_c16k.sh.
set -u
cd /home/cse-sdpl/acon-main/experiments/appworld
export PYTHONPATH=/home/cse-sdpl/acon-main/src
PY=/home/cse-sdpl/acon-main/.venv/bin/python
MAXITER=30
MEMCAP_KB=${MEMCAP_KB:-20971520}   # override to retry a task that hit the cap
AGENT=qwen2.5-14b-c16k
SPLITS="train dev"
mkdir -p run_logs

run_arm () {
  local tag="$1"; local cfg="$2"
  local extra=""
  [ -n "$cfg" ] && extra="--co_config_path $cfg"
  local t0=$(date +%s)
  echo "###### ARM $tag start=$(date -Is) cfg=${cfg:-NONE}"
  export ACON_REQ_LOG="$PWD/run_logs/${tag}_full.requests.jsonl"
  for split in $SPLITS; do
    local out="outputs/${AGENT}_${tag}/$split"
    # reuse the finished train_history_tiny results (copied once, never overwritten)
    if [ "$split" = train ] && [ -d "outputs/${AGENT}_${tag}/train_history_tiny" ]; then
      mkdir -p "$out"
      for d in outputs/${AGENT}_${tag}/train_history_tiny/task_*; do
        [ -f "$d/results.json" ] && [ ! -e "$out/$(basename $d)" ] && cp -r "$d" "$out/"
      done
    fi
    for tid in $(grep . data/datasets/$split.txt); do
      if [ -f "$out/task_$tid/results.json" ]; then continue; fi
      rm -rf "$out/task_$tid"
      ( export ACON_TASK_ID="$tid"; ulimit -v $MEMCAP_KB; \
        $PY run_all.py --split $split --task_id "$tid" \
           --model_name $AGENT --tag "$tag" --max_iter $MAXITER $extra \
      ) >> "run_logs/${tag}_full.log" 2>&1
      local rc=$?
      [ $rc -ne 0 ] && echo "[task $tid EXITED rc=$rc]" >> "run_logs/${tag}_full.log"
      echo "[progress $tag] $split/$tid rc=$rc $(date -Is)" >> run_logs/progress_full_c16k.log
    done
  done
  echo "###### ARM $tag DONE end=$(date -Is) seconds=$(( $(date +%s) - t0 ))"
}

echo "===== RQ1 full (train+dev) c16k START $(date -Is) ====="
run_arm base_none_c16k ""
run_arm acon_hist_c16k configs/context_opt/ollama_qwen2.5_7b_c16k_history.yaml
run_arm acon_obs_c16k  configs/context_opt/ollama_qwen2.5_7b_c16k_obs.yaml
run_arm acon_uni_c16k  configs/context_opt/ollama_qwen2.5_7b_c16k_unified.yaml
echo "===== RQ1 full (train+dev) c16k END $(date -Is) ====="
