#!/usr/bin/env bash
# One heavy segment: latency on §1b's protocol. $1 = gpu | cpu. GPU: each arm after a cool-down (the script waits
# for the GPU to fall below 65 C and records the thermal state). CPU: 6 threads (the machine cap) and 4 threads.
set -u
ROOT=/c/Users/migue/OneDrive/Desktop/Personal/AILABS/Tez-pilot
PY=$ROOT/.venv-pilot/Scripts/python.exe
LOG=$ROOT/runs/pilot/logs
cd "$ROOT/experiments/small_model" || exit 1
if [ "$1" = gpu ]; then
  for arm in a-base-30k a-small-30k c-qwen08-30k; do
    [ -f "$ROOT/results/pilot/train_$arm.json" ] || continue
    $PY latency.py --arm "$arm" --device cuda >> "$LOG/latency.log" 2>&1 || echo "latency $arm cuda failed" >> "$LOG/latency.log"
  done
else
  for arm in a-small-30k a-base-30k; do
    [ -f "$ROOT/results/pilot/train_$arm.json" ] || continue
    for t in 6 4; do
      $PY latency.py --arm "$arm" --device cpu --threads $t --warmup 3 --reps 20 --reps50 10 >> "$LOG/latency.log" 2>&1 || echo "latency $arm cpu$t failed" >> "$LOG/latency.log"
    done
  done
fi
