#!/usr/bin/env bash
# Run one or more arms on one AppWorld split with the c16k setup of run_rq1_c16k.sh.
#   bash run_arms_c16k.sh <split> <tag>=<co_config|NONE> [<tag>=<co_config|NONE> ...]
# e.g. bash run_arms_c16k.sh dev acon_hist_paper_c16k=configs/context_opt/ollama_qwen2.5_7b_c16k_history_paperutco.yaml
# Same per-task process, memory cap and skip-if-done resume as run_rq1_full_c16k.sh.
set -u
cd /home/cse-sdpl/acon-main/experiments/appworld
export PYTHONPATH=/home/cse-sdpl/acon-main/src
PY=/home/cse-sdpl/acon-main/.venv/bin/python
MAXITER=30
MEMCAP_KB=${MEMCAP_KB:-20971520}
AGENT=qwen2.5-14b-c16k
SPLIT=$1; shift
# TASKS_FILE: run only these task ids (one per line) but still write to outputs/.../$SPLIT,
# so two machines can each take part of a split.
TASKS_FILE=${TASKS_FILE:-data/datasets/$SPLIT.txt}
mkdir -p run_logs

for v in "qwen2.5-14b-c16k qwen2.5:14b" "qwen2.5-7b-c16k qwen2.5:7b"; do
  set -- $v "$@"; name=$1; base=$2; shift 2
  if ! ollama list | grep -q "^$name"; then
    printf 'FROM %s\nPARAMETER num_ctx 16384\n' "$base" > "$HOME/Modelfile.$name"
    ollama create "$name" -f "$HOME/Modelfile.$name"
  fi
done

for arm in "$@"; do
  tag=${arm%%=*}; cfg=${arm#*=}
  extra=""; [ "$cfg" != NONE ] && extra="--co_config_path $cfg"
  out="outputs/${AGENT}_${tag}/$SPLIT"
  t0=$(date +%s)
  echo "###### ARM $tag split=$SPLIT start=$(date -Is) cfg=$cfg"
  export ACON_REQ_LOG="$PWD/run_logs/${tag}_${SPLIT}.requests.jsonl"
  for tid in $(grep . "$TASKS_FILE"); do
    [ -f "$out/task_$tid/results.json" ] && continue
    rm -rf "$out/task_$tid"
    ( export ACON_TASK_ID="$tid"; ulimit -v $MEMCAP_KB; \
      $PY run_all.py --split $SPLIT --task_id "$tid" \
         --model_name $AGENT --tag "$tag" --max_iter $MAXITER $extra \
    ) >> "run_logs/${tag}_${SPLIT}.log" 2>&1
    rc=$?
    [ $rc -ne 0 ] && echo "[task $tid EXITED rc=$rc]" >> "run_logs/${tag}_${SPLIT}.log"
    echo "[progress $tag] $SPLIT/$tid rc=$rc $(date -Is)" >> run_logs/progress_arms_c16k.log
  done
  echo "###### ARM $tag DONE end=$(date -Is) seconds=$(( $(date +%s) - t0 ))"
done
