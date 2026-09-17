#!/usr/bin/env bash
# Remote GPU workflow helper for the SAC x ACON pilot.
#
#   ./remote.sh push            code -> gpubox (dry-run with: push -n)
#   ./remote.sh pull            results (RESULTS.md, eval json, png) gpubox -> here
#   ./remote.sh run "<cmd>"     run a command in the SAC conda env on gpubox
#   ./remote.sh launch <name> "<cmd>"   detached run, logged to ~/runs/<name>.log
#   ./remote.sh watch <name>    tail a detached run's log
#   ./remote.sh jobs            what is running, GPU memory, disk
#   ./remote.sh sh              interactive shell
set -euo pipefail

HOST=gpubox
RDIR='~/Context-Optimizer'
LDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# never sync: git metadata, caches, big binaries, experiment outputs, datasets
EXCLUDES=(
  --exclude '.git' --exclude '__pycache__' --exclude '*.pyc'
  --exclude '*.pdf' --exclude '.DS_Store' --exclude 'runs/'
  --exclude 'SAC/experiment/*/output' --exclude 'SAC/experiment/*/cache'
  --exclude 'sac_assets' --exclude '*.pt' --exclude 'wandb/'
)

cmd=${1:-jobs}; shift || true

case "$cmd" in
  push)
    rsync -avzc --partial "${EXCLUDES[@]}" "$@" "$LDIR/" "$HOST:$RDIR/"
    ;;
  pull)
    # instruction_inference_results*.json writes the whole context back per example and
    # reaches 1.5 GB on a full eval -- everything the analysis needs is in the compact
    # per_example_scores sidecar and the subset/bootstrap summaries.
    rsync -avz --partial \
      --include '*/' \
      --exclude 'instruction_inference_results*.json' \
      --exclude 'instruction_eval_info_list_*.json' \
      --include 'RESULTS.md' --include '*.json' --include '*.png' --include '*.md' \
      --exclude '*' \
      "$HOST:$RDIR/SAC/" "$LDIR/SAC/"
    ;;
  run)
    ssh -t "$HOST" "source $RDIR/SAC/env.sh >/dev/null 2>&1; cd $RDIR/SAC && $*"
    ;;
  launch)
    # -f backgrounds ssh itself after auth; without it the channel stays open for the
    # lifetime of the remote job and the caller blocks on a multi-hour training run.
    name=$1; shift
    ssh "$HOST" "mkdir -p ~/runs && cat > ~/runs/$name.sh" <<EOSCRIPT
#!/usr/bin/env bash
source $RDIR/SAC/env.sh >/dev/null 2>&1
cd $RDIR/SAC || exit 1
$*
EOSCRIPT
    ssh -f "$HOST" "setsid nohup bash ~/runs/$name.sh > ~/runs/$name.log 2>&1 < /dev/null"
    echo "launched $name -> ~/runs/$name.log  (./remote.sh watch $name)"
    ;;
  watch)
    ssh -t "$HOST" "tail -f ~/runs/${1:?need a run name}.log"
    ;;
  jobs)
    ssh "$HOST" 'echo "=== python/torchrun ==="; ps -eo pid,etime,pcpu,rss,cmd --sort=-pcpu \
        | grep -E "python|torchrun" | grep -v grep | grep -v unattended | head
      echo; echo "=== GPU (nvidia-smi is broken: 550 module vs 580 userspace) ==="
      python3 -c "
import ctypes,sys
try:
    c=ctypes.CDLL(\"libcuda.so.1\"); c.cuInit(0)
    f=ctypes.c_size_t(); t=ctypes.c_size_t(); ctx=ctypes.c_void_p(); d=ctypes.c_int()
    c.cuDeviceGet(ctypes.byref(d),0); c.cuCtxCreate_v2(ctypes.byref(ctx),0,d)
    c.cuMemGetInfo_v2(ctypes.byref(f),ctypes.byref(t))
    print(\"VRAM used %.2f / %.2f GiB\"%((t.value-f.value)/1024**3, t.value/1024**3))
except Exception as e: print(\"probe failed:\",e)"
      echo; echo "=== logs ==="; ls -lt ~/runs 2>/dev/null | head
      echo; echo "=== disk ==="; df -h ~ | tail -1'
    ;;
  sh)  ssh -t "$HOST" "cd $RDIR/SAC && exec bash -l" ;;
  *)   sed -n '2,12p' "${BASH_SOURCE[0]}"; exit 1 ;;
esac
