#!/usr/bin/env bash
# The plan-v2 pilot queue: train -> stride-10 eval -> score, one run at a time, unattended.
#
#   ./scripts/run_pilot.sh                 # default queue, in priority order
#   ./scripts/run_pilot.sh r0_sac r2_fgd   # just these
#
# Crash-tolerant by design: a stage that fails is recorded and the queue moves on, so one
# bad run costs one run and not the night. Re-running the script skips finished stages
# (training resumes from its checkpoint; evals reuse existing generations).
set -uo pipefail

source "$HOME/Context-Optimizer/SAC/env.sh"
cd "$SAC_HOME"

STRIDE="${STRIDE:-10}"
MID_STEP="${MID_STEP:-2500}"
STATUS="$SAC_HOME/experiment/pilot_status.tsv"
DEFAULT_QUEUE=(r0_sac r2_fgd r1_uniform_matched r3_tga_fgd r1a_uniform_lambda1)
QUEUE=("${@:-}")
[ -z "${QUEUE[0]:-}" ] && QUEUE=("${DEFAULT_QUEUE[@]}")

mark() { printf '%s\t%s\t%s\t%s\n' "$(date -Is)" "$1" "$2" "$3" >> "$STATUS"; }

echo "=== pilot queue: ${QUEUE[*]} ==="
echo "=== stride $STRIDE | mid-checkpoint step $MID_STEP | status -> $STATUS ==="
python scripts/make_pilot_configs.py --only "${QUEUE[@]}" || exit 1

for RUN in "${QUEUE[@]}"; do
  W="$SAC_HOME/experiment/$RUN"
  REL="../experiment/$RUN"
  echo ""
  echo "############################################################"
  echo "### $RUN  ($(date -Is))"
  echo "############################################################"
  [ -d "$W" ] || { echo "no such work dir: $W"; mark "$RUN" setup MISSING; continue; }

  # R1 only means anything if its KL mass matches R2's, and R2's realized gate rate is not
  # knowable until R2 has run (the pretrain-adapter student starts far weaker than the
  # converged one G0 measured). The queue order R0 -> R2 -> R1 makes this possible.
  if [ "$RUN" = "r1_uniform_matched" ] && [ -f "$SAC_HOME/experiment/r2_fgd/output/instruction_info.json" ] \
     && [ ! -f "$W/output/instruction_adapter.pt" ]; then
    echo "[configs] re-deriving R1's lambda from R2's realized gate rate"
    python scripts/make_pilot_configs.py --only r1_uniform_matched --lambda_from_run r2_fgd --force
  fi

  python patch_config.py "experiment/$RUN" || { mark "$RUN" patch_config FAIL; continue; }

  # ---- train (resume if a checkpoint is already there) ----
  if [ -f "$W/output/instruction_adapter.pt" ]; then
    echo "[skip] $RUN already trained"
    mark "$RUN" train SKIP
  else
    RESUME=""
    [ -f "$W/resume_checkpoint.pt" ] && RESUME="--resume" && echo "[resume] $RUN"
    ( cd "$SAC_HOME/sft" && python instruction_trainer.py --work_dir "$REL" $RESUME ) \
      2>&1 | tee -a "$W/train.log"
    if [ -f "$W/output/instruction_adapter.pt" ]; then mark "$RUN" train OK
    else mark "$RUN" train FAIL; echo "!!! $RUN training produced no adapter, skipping its evals"; continue; fi
  fi

  # ---- eval: final checkpoint, then the mid-training one (plan-v2 T1) ----
  for SPEC in "final:instruction_adapter.pt" "mid:instruction_adapter_step${MID_STEP}.pt"; do
    TAG="${SPEC%%:*}"; ADAPTER="${SPEC#*:}"
    [ -f "$W/output/$ADAPTER" ] || { echo "[skip] $RUN $TAG: no $ADAPTER"; continue; }

    SUFFIX="_stride${STRIDE}"
    EXTRA=()
    if [ "$TAG" = "mid" ]; then SUFFIX="${SUFFIX}_step${MID_STEP}"; EXTRA=(--adapter_name "$ADAPTER"); fi

    if [ -f "$W/output/per_example_scores${SUFFIX}.json" ]; then
      echo "[skip] $RUN $TAG already scored"; mark "$RUN" "eval_$TAG" SKIP; continue
    fi
    ( cd "$SAC_HOME/sft" && python instruction_evaluator.py --work_dir "$REL" \
        --batch_size 1 --eval_stride "$STRIDE" "${EXTRA[@]}" ) 2>&1 | tee -a "$W/eval_${TAG}.log"
    python util/evaluate_iid.py --work_dir "$W" --suffix "$SUFFIX" > /dev/null
    python util/evaluate_ood.py --work_dir "$W" --suffix "$SUFFIX" > /dev/null
    if [ -f "$W/output/per_example_scores${SUFFIX}.json" ]; then mark "$RUN" "eval_$TAG" OK
    else mark "$RUN" "eval_$TAG" FAIL; fi
  done
done

echo ""
echo "############################################################"
echo "### pilot queue finished $(date -Is)"
echo "############################################################"
column -t "$STATUS" 2>/dev/null || cat "$STATUS"

echo ""
echo "=== paired bootstrap vs R0 (primary claim: R2 > R0) ==="
for RUN in "${QUEUE[@]}"; do
  [ "$RUN" = "r0_sac" ] && continue
  [ -f "$SAC_HOME/experiment/$RUN/output/per_example_scores_stride${STRIDE}.json" ] || continue
  echo "--- $RUN vs r0_sac ---"
  python util/paired_bootstrap.py --variant "experiment/$RUN" --baseline experiment/r0_sac \
    --suffix "_stride${STRIDE}" \
    --out "$SAC_HOME/experiment/$RUN/output/bootstrap_vs_r0.json" 2>&1 | grep -E "ID |OOD |^variant|^baseline"
done

echo ""
echo "=== supporting ablation: R2 vs R1 (which examples, not how much KL) ==="
if [ -f "$SAC_HOME/experiment/r1_uniform_matched/output/per_example_scores_stride${STRIDE}.json" ]; then
  python util/paired_bootstrap.py --variant experiment/r2_fgd --baseline experiment/r1_uniform_matched \
    --suffix "_stride${STRIDE}" \
    --out "$SAC_HOME/experiment/r2_fgd/output/bootstrap_vs_r1.json" 2>&1 | grep -E "ID |OOD "
fi
