#!/usr/bin/env bash
# Stops the full-test driver once the 4th arm (acon_obs_q7b) finishes, so the
# unified and llama3.1:8b arms never start. Scope cut to fit the report deadline.
cd /home/cse-sdpl/acon-main/experiments/appworld
while true; do
  if grep -q "ARM acon_obs_q7b DONE" run_logs/driver.log 2>/dev/null; then
    echo "[watchdog] 4th arm complete at $(date -Is); stopping driver."
    pkill -f "run_acon_full[_]test" 2>/dev/null
    sleep 2
    pkill -f "run_all[.]py" 2>/dev/null
    sleep 3
    /home/cse-sdpl/acon-main/.venv/bin/python aggregate_results.py >> run_logs/aggregate.log 2>&1
    echo "[watchdog] RESULTS.md regenerated. Done."
    exit 0
  fi
  pgrep -f "run_acon_full[_]test" >/dev/null || { echo "[watchdog] driver already gone at $(date -Is)"; exit 0; }
  sleep 60
done
