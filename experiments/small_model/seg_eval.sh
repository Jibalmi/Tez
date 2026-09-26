#!/usr/bin/env bash
# One heavy segment: evaluate every trained arm (and GLiClass-x-base) not yet evaluated, within MINUTES.
set -u
ROOT=/c/Users/migue/OneDrive/Desktop/Personal/AILABS/Tez-pilot
PY=$ROOT/.venv-pilot/Scripts/python.exe
LOG=$ROOT/runs/pilot/logs
MINUTES=${1:-55}
START=$(date +%s)
cd "$ROOT/experiments/small_model" || exit 1
left() { echo $(( MINUTES - ($(date +%s) - START) / 60 )); }
rc_all=0
for arm in a-base-30k a-base-10k a-small-30k gliclass-x-base c-qwen08-30k; do
  [ -f "$ROOT/results/pilot/eval_$arm.json" ] && continue
  case $arm in gliclass*) ;; *) [ -f "$ROOT/results/pilot/train_$arm.json" ] || continue ;; esac
  need=8; [ "$arm" = c-qwen08-30k ] && need=25
  if [ "$(left)" -lt "$need" ]; then rc_all=3; break; fi
  $PY eval_arms.py --arm "$arm" >> "$LOG/eval_$arm.log" 2>&1 || { echo "eval $arm failed" >> "$LOG/eval_$arm.log"; rc_all=1; }
done
exit $rc_all
