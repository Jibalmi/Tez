#!/usr/bin/env bash
# One heavy segment (inside C:/temp/run_heavy.py): do the pilot's pending work in order, as much as fits in MINUTES.
# The queue for the shared lock costs 30-90 minutes per hold, so a hold is not spent on a single step.
#   schemas -> teacher -> train A (base 30k, base 10k, small 30k) -> train C -> evaluate -> GPU latency -> CPU latency
# Exit 0: everything done. 3: work left (resumable). 1: a step failed (the driver stops; nothing is retried).
set -u
ROOT=/c/Users/migue/OneDrive/Desktop/Personal/AILABS/Tez-pilot
PY=$ROOT/.venv-pilot/Scripts/python.exe
BASH_EXE="C:/Program Files/Git/bin/bash.exe"
SM=$ROOT/experiments/small_model
BIN=$ROOT/runs/pilot/bin
LOG=$ROOT/runs/pilot/logs
RES=$ROOT/results/pilot
MINUTES=${1:-55}
START=$(date +%s)
cd "$SM" || exit 1
left() { echo $(( MINUTES - ($(date +%s) - START) / 60 )); }
say() { echo "$(date -Iseconds) seg_next: $*" >> "$LOG/driver.log"; }
teacher_left() {
  $PY -c "
import glob
done=sum(1 for p in glob.glob(r'$ROOT/runs/pilot/teacher/part-*.jsonl') for _ in open(p,encoding='utf-8'))
print(sum(1 for _ in open(r'$ROOT/runs/pilot/assign.jsonl',encoding='utf-8')) - done)"
}

# 1. schemas (own llama-server; exits 3 while clusters remain)
if [ ! -f "$ROOT/runs/pilot/assign.jsonl" ]; then
  [ "$(left)" -lt 12 ] && exit 3
  cp "$SM/seg_schemas.sh" "$BIN/seg_schemas.sh.run"
  say "schemas ($(left) min left)"
  "$BASH_EXE" "$BIN/seg_schemas.sh.run" "$(left)"; rc=$?
  [ $rc -eq 3 ] && exit 3
  [ $rc -ne 0 ] && { say "schemas failed ($rc)"; exit 1; }
fi

# 2. teacher (12B in process; stops starting states before the deadline)
while [ "$(teacher_left)" -gt 0 ]; do
  L=$(( $(left) - 4 ))
  [ "$L" -lt 8 ] && exit 3
  say "teacher ($L min)"
  $PY teacher.py --minutes "$L" --n-ctx 4096 >> "$LOG/teacher.log" 2>&1 || { say "teacher failed"; exit 1; }
done

# 3. option A arms
train_a() {   # arm backbone data lr
  [ -f "$RES/train_$1.json" ] && return 0
  local L=$(( $(left) - 5 ))
  [ "$L" -lt 10 ] && return 3
  say "train $1 ($L min)"
  $PY train_a.py --arm "$1" --backbone "$2" --data "$3" --lr "$4" --minutes "$L" >> "$LOG/train_$1.log" 2>&1
}
train_a a-base-30k jhu-clsp/mmBERT-base 30k 3e-5; rc=$?; [ $rc -eq 3 ] && exit 3; [ $rc -ne 0 ] && { say "train a-base-30k failed ($rc)"; exit 1; }
train_a a-base-10k jhu-clsp/mmBERT-base 10k 3e-5; rc=$?; [ $rc -eq 3 ] && exit 3; [ $rc -ne 0 ] && { say "train a-base-10k failed ($rc)"; exit 1; }
train_a a-small-30k jhu-clsp/mmBERT-small 30k 5e-5; rc=$?; [ $rc -eq 3 ] && exit 3; [ $rc -ne 0 ] && { say "train a-small-30k failed ($rc)"; exit 1; }

# 4. option C (at most two segments of training; then it finishes with the part of the epoch it saw)
if [ ! -f "$RES/train_c-qwen08-30k.json" ]; then
  L=$(( $(left) - 8 ))
  [ "$L" -lt 15 ] && exit 3
  say "train c-qwen08-30k ($L min)"
  $PY train_c.py --arm c-qwen08-30k --minutes "$L" --max-segments 2 >> "$LOG/train_c-qwen08-30k.log" 2>&1; rc=$?
  [ $rc -eq 3 ] && exit 3
  [ $rc -ne 0 ] && { say "train c failed ($rc)"; exit 1; }
fi

# 5. evaluation (each arm, and GLiClass-x-base)
for arm in a-base-30k a-base-10k a-small-30k gliclass-x-base c-qwen08-30k; do
  [ -f "$RES/eval_$arm.json" ] && continue
  need=10; [ "$arm" = c-qwen08-30k ] && need=30
  [ "$(left)" -lt "$need" ] && exit 3
  say "evaluate $arm ($(left) min left)"
  $PY eval_arms.py --arm "$arm" >> "$LOG/eval_$arm.log" 2>&1 || { say "evaluate $arm failed"; exit 1; }
done

# 6. GPU latency (each size starts below 65 C; the thermal state is recorded)
for arm in a-base-30k a-small-30k c-qwen08-30k; do
  grep -q "\"${arm}_cuda\"" "$RES/latency.json" 2>/dev/null && continue
  [ "$(left)" -lt 20 ] && exit 3
  say "latency $arm cuda ($(left) min left)"
  $PY latency.py --arm "$arm" --device cuda >> "$LOG/latency.log" 2>&1 || { say "latency $arm cuda failed"; exit 1; }
done

# 7. CPU latency (6 threads, the machine cap; and 4 threads, as Laya's CPU table)
for arm in a-small-30k a-base-30k; do
  for t in 6 4; do
    grep -q "\"${arm}_cpu${t}t\"" "$RES/latency.json" 2>/dev/null && continue
    [ "$(left)" -lt 12 ] && exit 3
    say "latency $arm cpu $t threads ($(left) min left)"
    $PY latency.py --arm "$arm" --device cpu --threads $t --warmup 3 --reps 20 --reps50 10 >> "$LOG/latency.log" 2>&1 \
      || { say "latency $arm cpu$t failed"; exit 1; }
  done
done
say "all steps done"
exit 0
