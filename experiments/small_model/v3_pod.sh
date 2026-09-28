#!/usr/bin/env bash
# The capacity round (v3): the Qwen control recipe (train_c.py defaults) on Qwen3.5-2B instead of 0.8B, trained on the
# same 155,835 v2 teacher decisions and evaluated on the frozen suites of the 30k run. Training is bounded to one
# 150-minute segment; an unfinished epoch is recorded as such by train_c.py.
#   bash experiments/small_model/v3_pod.sh setup | train | eval | package | all
set -euo pipefail
ROOT=/workspace/tez-pilot
cd "$ROOT"
DATA=$ROOT/runs/student-data-v2
ARM=c-qwen2b-v2
export HF_HOME=/workspace/hf-cache HF_HUB_DISABLE_IMPLICIT_TOKEN=1 HF_HUB_DISABLE_TELEMETRY=1
export OMP_NUM_THREADS=6 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=6 RUN_HEAVY_THREADS=6 PYTHONUTF8=1
export TEZ_PILOT_QWEN_BASE=Qwen/Qwen3.5-2B TEZ_PILOT_TRAIN_ONLY_PREFIX=v2-
export TEZ_PILOT_RUNS=$DATA TEZ_PILOT_RESULTS=$ROOT/results/pilot-v3
export TEZ_PILOT_EVAL_SUITES=$ROOT/runs/pilot/student-prep-20260927/eval-suites.json
venv() { source "$ROOT/runs/student-venv/bin/activate"; }

setup() {
  python3 - <<'EOF'
import hashlib, json, pathlib, sys
m = json.loads(pathlib.Path("/workspace/manifest.json").read_text())
bad = [n for n, h in m.items() if hashlib.sha256(pathlib.Path(n).read_bytes()).hexdigest() != h]
print(f"manifest: {len(m) - len(bad)}/{len(m)} files match"); sys.exit(1 if bad else 0)
EOF
  python3 -m venv --system-site-packages runs/student-venv; venv
  python -m pip install -q 'transformers==5.17.0' 'peft==0.21.0' sentencepiece scipy scikit-learn psutil requests PyYAML
  MAX_JOBS=8 python -m pip install -q --no-build-isolation 'fla-core==0.5.2' 'causal-conv1d==1.7.0' 'tilelang==0.1.14'
  python -m pip freeze > /workspace/environment.txt
  touch /workspace/setup.done
}
train() { venv; python experiments/small_model/train_c.py --arm $ARM --minutes 150 --max-segments 1; touch /workspace/train.done; }
evaluate() { venv; python experiments/small_model/eval_arms.py --arm $ARM; touch /workspace/eval.done; }
package() {
  tar -cf /workspace/v3-return.tar results/pilot-v3 "runs/student-data-v2/models/$ARM" /workspace/environment.txt 2>/dev/null || true
  sha256sum /workspace/v3-return.tar
}
all() {
  [ -f /workspace/setup.done ] || setup
  [ -f /workspace/train.done ] || train
  [ -f /workspace/eval.done ] || evaluate
  package
  touch /workspace/all.done
}
case "${1:-}" in
  setup) setup ;; train) train ;; eval) evaluate ;; package) package ;; all) all ;;
  *) echo "usage: $0 setup|train|eval|package|all" >&2; exit 2 ;;
esac
