#!/usr/bin/env bash
# The pilot's heavy steps, one at a time, each through C:/temp/run_heavy.py (compute lock, thread cap 6, resource
# watchdog), each at most 60 minutes. A step that ends with exit 3 (resumable work left) is queued again; any other
# failure stops the driver (reported, never retried in a loop). Progress: runs/pilot/logs/driver.log.
set -u
ROOT=/c/Users/migue/OneDrive/Desktop/Personal/AILABS/Tez-pilot
PY=$ROOT/.venv-pilot/Scripts/python.exe
BASH_EXE="C:/Program Files/Git/bin/bash.exe"      # not a bare `bash`: CreateProcess would find WSL's first
SM=$ROOT/experiments/small_model
LOG=$ROOT/runs/pilot/logs/driver.log
RH() {   # purpose minutes cmd...
  local purpose=$1 minutes=$2; shift 2
  echo "$(date -Iseconds) queue: $purpose" >> "$LOG"
  python C:/temp/run_heavy.py --owner projects-0c-pilot --purpose "Tez small-model pilot: $purpose" --minutes "$minutes" -- "$@" >> "$LOG" 2>&1
  local rc=$?
  echo "$(date -Iseconds) exit $rc: $purpose" >> "$LOG"
  return $rc
}
segments() {   # name max_segments purpose minutes cmd...   (repeat while exit 3)
  local name=$1 max=$2; shift 2
  for i in $(seq 1 "$max"); do
    RH "$@"; local rc=$?
    [ $rc -eq 0 ] && return 0
    [ $rc -eq 3 ] && continue
    echo "$(date -Iseconds) STOP: $name failed with exit $rc" >> "$LOG"; exit $rc
  done
  echo "$(date -Iseconds) STOP: $name not finished after $max segments" >> "$LOG"; exit 4
}
teacher_left() {
  $PY -c "
import glob,json,sys
done=sum(1 for p in glob.glob(r'$ROOT/runs/pilot/teacher/part-*.jsonl') for _ in open(p,encoding='utf-8'))
total=sum(1 for _ in open(r'$ROOT/runs/pilot/assign.jsonl',encoding='utf-8'))
print(total-done)"
}

[ -f "$ROOT/runs/pilot/assign.jsonl" ] || segments schemas 3 "schemas: pool filter, clusters, 12B writes schemas (llama-server :8851)" 60 "$BASH_EXE" "$SM/seg_schemas.sh" 55

for i in 1 2 3; do
  [ "$(teacher_left)" -le 0 ] && break
  RH "teacher: 12B labels decisions in process (resumable)" 60 "$PY" "$SM/teacher.py" --minutes 52 --n-ctx 4096 \
    || { echo "$(date -Iseconds) STOP: teacher failed" >> "$LOG"; exit 1; }
done
[ "$(teacher_left)" -le 0 ] || { echo "$(date -Iseconds) STOP: teacher unfinished after 3 segments" >> "$LOG"; exit 4; }

segments train_a 4 "train option A arms (mmBERT base 30k, 10k; small 30k)" 60 "$BASH_EXE" "$SM/seg_train_a.sh" 55
[ -f "$ROOT/results/pilot/train_c-qwen08-30k.json" ] || \
  segments train_c 2 "train option C (Qwen3.5-0.8B LoRA r=16)" 60 "$PY" "$SM/train_c.py" --arm c-qwen08-30k --minutes 50 --max-segments 2
segments eval 3 "evaluate arms and GLiClass on the held-out suites" 60 "$BASH_EXE" "$SM/seg_eval.sh" 55
RH "latency on the GPU (thermal state recorded)" 45 "$BASH_EXE" "$SM/seg_latency.sh" gpu
RH "latency on the CPU (6 and 4 threads)" 45 "$BASH_EXE" "$SM/seg_latency.sh" cpu
echo "$(date -Iseconds) DONE" >> "$LOG"
