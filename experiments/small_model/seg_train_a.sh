#!/usr/bin/env bash
# One heavy segment: train the option-A arms still missing, in order, within MINUTES. Exit 0 when all are trained,
# 3 when the segment ended with work left (checkpointed), 1 on failure.
set -u
ROOT=/c/Users/migue/OneDrive/Desktop/Personal/AILABS/Tez-pilot
PY=$ROOT/.venv-pilot/Scripts/python.exe
LOG=$ROOT/runs/pilot/logs
MINUTES=${1:-55}
START=$(date +%s)
cd "$ROOT/experiments/small_model" || exit 1
left() { echo $(( MINUTES - ($(date +%s) - START) / 60 )); }
run_arm() {   # arm backbone data lr
  [ -f "$ROOT/results/pilot/train_$1.json" ] && return 0
  L=$(( $(left) - 5 ))
  [ "$L" -lt 8 ] && return 3
  nvidia-smi --query-gpu=temperature.gpu,clocks_throttle_reasons.sw_thermal_slowdown,memory.used --format=csv,noheader >> "$LOG/train_$1.log"
  $PY train_a.py --arm "$1" --backbone "$2" --data "$3" --lr "$4" --minutes "$L" >> "$LOG/train_$1.log" 2>&1
  return $?
}
run_arm a-base-30k jhu-clsp/mmBERT-base 30k 3e-5; rc=$?; [ $rc -ne 0 ] && exit $rc
run_arm a-base-10k jhu-clsp/mmBERT-base 10k 3e-5; rc=$?; [ $rc -ne 0 ] && exit $rc
run_arm a-small-30k jhu-clsp/mmBERT-small 30k 5e-5; rc=$?; [ $rc -ne 0 ] && exit $rc
exit 0
