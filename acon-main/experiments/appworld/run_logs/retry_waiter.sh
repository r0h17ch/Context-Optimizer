#!/usr/bin/env bash
# Wait for the main driver, then re-invoke it: finished tasks are skipped, so only the
# task that hit the 20 GB cap (acon_obs dev/df61dc5_1) runs, now with a 40 GB cap.
cd /home/cse-sdpl/acon-main/experiments/appworld
while kill -0 157886 2>/dev/null; do sleep 5; done
echo "===== RETRY pass (MEMCAP 40 GB) $(date -Is) =====" >> run_logs/driver_full_c16k.log
MEMCAP_KB=41943040 bash run_rq1_full_c16k.sh >> run_logs/driver_full_c16k.log 2>&1
