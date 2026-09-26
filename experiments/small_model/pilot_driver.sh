#!/usr/bin/env bash
# The pilot's heavy work, one run_heavy hold at a time (compute lock, thread cap 6, resource watchdog), each hold at
# most 60 minutes. Each hold runs seg_next.sh, which does as much pending work as fits. Exit 3 (work left) queues
# another hold; any other failure stops the driver (reported, never retried in a loop). Log: runs/pilot/logs/driver.log.
set -u
ROOT=/c/Users/migue/OneDrive/Desktop/Personal/AILABS/Tez-pilot
BASH_EXE="C:/Program Files/Git/bin/bash.exe"      # not a bare `bash`: CreateProcess would find WSL's first
SM=$ROOT/experiments/small_model
BIN=$ROOT/runs/pilot/bin
LOG=$ROOT/runs/pilot/logs/driver.log
for i in $(seq 1 14); do
  cp "$SM/seg_next.sh" "$BIN/seg_next.sh.run"      # a snapshot: bash reads a script as it runs
  echo "$(date -Iseconds) queue: hold $i" >> "$LOG"
  python C:/temp/run_heavy.py --owner projects-0c-pilot --purpose "Tez small-model pilot: hold $i (schemas, teacher, training, evaluation, latency; resumable)" \
    --minutes 60 -- "$BASH_EXE" "$BIN/seg_next.sh.run" 55 >> "$LOG" 2>&1
  rc=$?
  echo "$(date -Iseconds) exit $rc: hold $i" >> "$LOG"
  [ $rc -eq 0 ] && { echo "$(date -Iseconds) DONE" >> "$LOG"; exit 0; }
  [ $rc -eq 3 ] && continue
  echo "$(date -Iseconds) STOP: hold $i failed with exit $rc" >> "$LOG"
  exit $rc
done
echo "$(date -Iseconds) STOP: work left after 14 holds" >> "$LOG"
exit 4
