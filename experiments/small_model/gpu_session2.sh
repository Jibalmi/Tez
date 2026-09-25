#!/usr/bin/env bash
# Pilot GPU sessions 2+ (days 2-4): finish the teacher if needed, train the four arms, evaluate them (and GLiClass),
# time them on the GPU. Each step is skipped when its output exists; the lock is released and queued for again
# whenever a hold passes ~100 minutes, so other sessions get the card between steps.
set -u
ROOT=/c/Users/migue/OneDrive/Desktop/Personal/AILABS/Tez-pilot
PY=$ROOT/.venv-pilot/Scripts/python.exe
LOCK="python C:/temp/gpu_lock.py"
OWNER=projects-0c-pilot
LOG=$ROOT/runs/pilot/logs
RES=$ROOT/results/pilot
cd "$ROOT/experiments/small_model" || exit 1
HELD=0
START=0
acquire() {
  $LOCK acquire --owner $OWNER --purpose "Tez small-model pilot: $1" --minutes 110 --wait >> "$LOG/gpu_session2.log" 2>&1
  HELD=1; START=$(date +%s)
  echo "acquired for: $1 $(date -Iseconds)" >> "$LOG/gpu_session2.log"
  $LOCK snapshot >> "$LOG/vram_session2.jsonl"
}
release() {
  if [ "$HELD" = 1 ]; then
    $LOCK release --owner $OWNER >> "$LOG/gpu_session2.log" 2>&1
    echo "released after $(( ($(date +%s) - START) / 60 )) min $(date -Iseconds)" >> "$LOG/gpu_session2.log"
    HELD=0
  fi
}
trap release EXIT
need() {   # $1 = minutes the next step may take; re-queue if the hold would pass ~110 minutes
  if [ "$HELD" = 1 ] && [ $(( ($(date +%s) - START) / 60 + $1 )) -gt 110 ]; then release; sleep 60; fi
  if [ "$HELD" = 0 ]; then acquire "$2"; fi
}
teacher_done() {
  $PY -c "
import json,sys,glob
n=sum(1 for p in glob.glob(r'$ROOT/runs/pilot/teacher/part-*.jsonl') for _ in open(p,encoding='utf-8'))
m=sum(1 for _ in open(r'$ROOT/runs/pilot/assign.jsonl',encoding='utf-8'))
sys.exit(0 if n>=m else 1)"
}

until [ -f "$ROOT/runs/pilot/assign.jsonl" ] && [ -f "$LOG/vram_session1_end.json" ]; do sleep 30; done

while ! teacher_done; do
  need 60 "finish teacher labels (12B in process)"
  $PY teacher.py --minutes 60 --n-ctx 4096 >> "$LOG/teacher.log" 2>&1
done

train() {   # arm, minutes, command...
  local arm=$1 mins=$2; shift 2
  if [ ! -f "$RES/train_$arm.json" ]; then
    need "$mins" "train $arm"
    "$@" >> "$LOG/train_$arm.log" 2>&1
  fi
}
train a-base-30k 35 $PY train_a.py --arm a-base-30k --backbone jhu-clsp/mmBERT-base --data 30k --minutes 30
train a-base-10k 15 $PY train_a.py --arm a-base-10k --backbone jhu-clsp/mmBERT-base --data 10k --minutes 12
train c-qwen08-30k 75 $PY train_c.py --arm c-qwen08-30k --minutes 70
train a-small-30k 25 $PY train_a.py --arm a-small-30k --backbone jhu-clsp/mmBERT-small --data 30k --lr 5e-5 --minutes 20

for arm in a-base-30k a-base-10k c-qwen08-30k a-small-30k gliclass-x-base; do
  if [ ! -f "$RES/eval_$arm.json" ]; then
    need 20 "evaluate $arm"
    $PY eval_arms.py --arm "$arm" >> "$LOG/eval_$arm.log" 2>&1
  fi
done
for arm in a-base-30k a-small-30k c-qwen08-30k; do
  if ! grep -q "\"${arm}_cuda\"" "$RES/latency.json" 2>/dev/null; then
    need 15 "GPU latency $arm"
    $PY latency.py --arm "$arm" --device cuda >> "$LOG/latency.log" 2>&1
  fi
done
echo "session2 done $(date -Iseconds)" >> "$LOG/gpu_session2.log"
