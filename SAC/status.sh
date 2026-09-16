#!/usr/bin/env bash
# One-shot status of any SAC run.  Usage: ./status.sh [work_dir_name]   (default sac_15x_mine)
W="${1:-sac_15x_mine}"
SAC="$HOME/Context-Optimizer/SAC"; D="$SAC/experiment/$W"
[ -d "$D" ] || { echo "no such run: $D"; exit 1; }

stage=""; ps -eo args | grep -q "pre_trainer\.p[y]" && stage="pretraining"
ps -eo args | grep -q "instruction_trainer\.p[y]" && stage="fine-tuning (SFT)"
ps -eo args | grep -q "instruction_evaluator\.p[y]" && stage="evaluating"
if [ -n "$stage" ]; then echo "STATUS   : running — $stage"
else echo "STATUS   : no training/eval process running"; fi

for f in pretrain sft eval; do
  L="$D/$f.log"; [ -f "$L" ] || continue
  line=$(tr '\r' '\n' < "$L" | grep -E "[0-9]+/[0-9]+ \[" | tail -1)
  [ -n "$line" ] && echo "$f: $line"
done

# latest loss from whichever info file is newest
info=$(ls -t "$D/output"/info.json "$D/output"/instruction_info.json 2>/dev/null | head -1)
[ -n "$info" ] && python3 - "$info" <<'PY'
import json,sys
d=json.load(open(sys.argv[1]))
if d:
    e=d[-1]; first=d[0]["training_loss"].get("lm_loss")
    print(f"LOSS     : {e['training_loss'].get('lm_loss'):.4f} at step {e['steps']:.0f}/{e['total_steps']} "
          f"(started {first:.4f}) | elapsed {e['run_time(hours)']:.1f} h")
PY

ck="$D/output/resume_checkpoint.pt"
[ -f "$ck" ] && echo "CHECKPT  : $(du -h "$ck" | cut -f1), saved $(date -r "$ck" '+%H:%M:%S')" \
             || echo "CHECKPT  : none (not yet written, or stage finished)"
"$HOME/miniconda3/envs/SAC/bin/python" -c "
import torch;f,t=torch.cuda.mem_get_info();print('GPU      : %.1f / %.1f GB used'%((t-f)/1e9,t/1e9))" 2>/dev/null | grep GPU
