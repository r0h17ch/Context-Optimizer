#!/usr/bin/env bash
# On-screen notice for other lab users while the RQ1 experiment runs.
# Refreshes once a minute and closes itself when the run is finished.
cd /home/cse-sdpl/acon-main/experiments/appworld
# Defaults are the 39-task run; the full run passes its own values.
# grep -c counts the last task even though the file has no trailing newline
PROGRESS=${PROGRESS:-run_logs/progress_c16k.log}
TOTAL=${TOTAL:-$((4 * $(grep -c . data/datasets/train_history_tiny.txt)))}
PATTERN=${PATTERN:-run_rq1_c16k[.]sh}
DURATION=${DURATION:-about a day}
while true; do
  clear
  done_n=$(grep -c "rc=" "$PROGRESS" 2>/dev/null || echo 0)
  last=$(tail -1 "$PROGRESS" 2>/dev/null)
  tput bold; tput setaf 1
  cat <<'EOF'

   ##########################################################################
   #                                                                        #
   #     PLEASE DO NOT SHUT DOWN, RESTART OR LOG OUT OF THIS MACHINE        #
   #                                                                        #
   ##########################################################################

EOF
  tput sgr0; tput bold
  echo "   A final-year project experiment is running on this machine:"
  tput sgr0
  echo "   LLM agents (qwen2.5 14B + 7B on the GPU) are being tested on the"
  echo "   AppWorld benchmark. A full run takes $DURATION. Shutting down"
  echo "   loses the progress and the GPU results."
  echo
  echo "   You can still use this computer normally for light work."
  echo "   Please avoid heavy GPU jobs until the run finishes."
  echo
  tput bold; echo "   Progress: $done_n / $TOTAL task runs finished"; tput sgr0
  echo "   Last update: ${last:-starting...}"
  echo "   Screen refreshed: $(date '+%d %b %Y, %H:%M')"
  echo
  if ! pgrep -f "$PATTERN" >/dev/null; then
    tput bold; tput setaf 2
    echo "   The experiment has FINISHED. It is now safe to shut down. Thank you!"
    tput sgr0
    break
  fi
  echo "   Thank you for your patience."
  sleep 60
done
read -r -p "   Press Enter to close this window." _
