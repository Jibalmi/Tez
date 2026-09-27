#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$ROOT"
python -m venv --system-site-packages runs/student-venv
source runs/student-venv/bin/activate
export HF_HOME=/workspace/hf-cache
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1 HF_HUB_DISABLE_TELEMETRY=1
export OMP_NUM_THREADS=6 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=6 RUN_HEAVY_THREADS=6
export TEZ_PILOT_RUNS="$ROOT/runs/student-data"
export TEZ_PILOT_RESULTS="$ROOT/results/pilot-student-h200-20260927"
export TEZ_PILOT_EVAL_SUITES="$ROOT/runs/student-input/eval-suites.json"
python -m pip install 'transformers==5.17.0' 'peft==0.21.0' 'gliclass==0.1.20' 'sentencepiece' 'scipy' 'scikit-learn' 'psutil' 'requests' 'PyYAML'
MAX_JOBS=4 python -m pip install --no-build-isolation 'fla-core==0.5.2' 'causal-conv1d==1.7.0' 'tilelang==0.1.14'
mkdir -p "$TEZ_PILOT_RESULTS"
python -m pip freeze > "$TEZ_PILOT_RESULTS/environment.txt"
nvidia-smi -q > "$TEZ_PILOT_RESULTS/gpu-before.txt"
python experiments/small_model/cloud_student.py run "$@"
