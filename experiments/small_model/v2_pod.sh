#!/usr/bin/env bash
# The data-scaling round (v2) on one RunPod GPU: the 12B writes schemas for the new pool's clusters, assigns questions,
# labels every decision in both option orders (two isolated workers, fresh state caches, as the H200 teacher run), then
# the Qwen control recipe (train_c.py defaults) trains on the original + new teacher data and is evaluated on the frozen
# suites of the 30k run. Every step is restartable; each writes a done-marker.
#   bash experiments/small_model/v2_pod.sh setup | schemas | teacher | data | train | eval | package
set -euo pipefail
ROOT=/workspace/tez-pilot
cd "$ROOT"
V2=$ROOT/runs/pilot-v2
V1=$ROOT/runs/pilot-cloud-h200-20260927          # the original exported teacher dataset (pool, assign, teacher/)
DATA=$ROOT/runs/student-data-v2
ARM=c-qwen08-v2
RT=$ROOT/runs/cloud-runtime
export HF_HOME=/workspace/hf-cache HF_HUB_DISABLE_IMPLICIT_TOKEN=1 HF_HUB_DISABLE_TELEMETRY=1
export OMP_NUM_THREADS=6 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=6 RUN_HEAVY_THREADS=6 PYTHONUTF8=1
export TEZ_PILOT_GGUF=$RT/models/047dae1d7894b9de8f08141e841544e007243290c02df8b39872991d1940c795.gguf
venv() { source "$ROOT/runs/student-venv/bin/activate"; }
libs() {
  export TEZ_PILOT_LLAMA_LIB; TEZ_PILOT_LLAMA_LIB=$(find "$RT/llama" -name libllama.so -print -quit)
  LIB_DIRS=$(find "$RT/llama" -type d -printf '%p:'); export LD_LIBRARY_PATH="${LIB_DIRS}${LD_LIBRARY_PATH:-}"
}

setup() {
  python3 - <<'EOF'
import hashlib, json, pathlib, sys
m = json.loads(pathlib.Path("/workspace/manifest.json").read_text())
bad = [n for n, h in m.items() if hashlib.sha256(pathlib.Path(n).read_bytes()).hexdigest() != h]
print(f"manifest: {len(m) - len(bad)}/{len(m)} files match"); sys.exit(1 if bad else 0)
EOF
  # the 12B GGUF and llama.cpp b11100 exactly as cloud_bootstrap.sh fetches and checks them
  sed -n '/^python - <<.PY.$/,/^PY$/p' experiments/small_model/cloud_bootstrap.sh | sed '1d;$d' > /workspace/fetch_runtime.py
  mkdir -p "$RT/models" "$RT/llama"
  python3 /workspace/fetch_runtime.py
  # the student environment exactly as cloud_student_bootstrap.sh builds it, plus the embedder for assignment
  python3 -m venv --system-site-packages runs/student-venv; venv
  python -m pip install -q 'transformers==5.17.0' 'peft==0.21.0' 'gliclass==0.1.20' sentencepiece scipy scikit-learn psutil \
    requests PyYAML sentence-transformers datasketch 'datasets==5.0.1' 'pyarrow==25.0.1' 'dill==0.4.1' \n    'multiprocess==0.70.19' 'xxhash==4.0.1'   # datasets: schemas.py assign loads the evaluation option sets
  MAX_JOBS=8 python -m pip install -q --no-build-isolation 'fla-core==0.5.2' 'causal-conv1d==1.7.0' 'tilelang==0.1.14'
  python -m pip freeze > /workspace/environment.txt
  touch /workspace/setup.done
}

schemas() {
  venv; libs
  SERVER=$(find "$RT/llama" -name llama-server -type f -print -quit)
  "$SERVER" -m "$TEZ_PILOT_GGUF" -ngl 99 -c 32768 -np 8 --port 8851 --host 127.0.0.1 > /workspace/llama-server.log 2>&1 &
  echo $! > /workspace/llama-server.pid
  for i in $(seq 1 120); do curl -sf http://127.0.0.1:8851/health >/dev/null && break; sleep 5; done
  TEZ_PILOT_RUNS=$V2 TEZ_PILOT_RESULTS=$ROOT/results/pilot-v2 python experiments/small_model/schemas.py write \
    --server http://127.0.0.1:8851 --parallel 8
  kill "$(cat /workspace/llama-server.pid)" || true
  TEZ_PILOT_RUNS=$V2 TEZ_PILOT_RESULTS=$ROOT/results/pilot-v2 python experiments/small_model/schemas.py assign --device cuda
  touch /workspace/schemas.done
}

teacher() {
  venv; libs
  for i in 0 1; do
    W=$ROOT/runs/pilot-v2-w$i; mkdir -p "$W"
    ln -sf "$V2/pool.jsonl" "$W/pool.jsonl"; ln -sf "$V2/assign.jsonl" "$W/assign.jsonl"
    TEZ_PILOT_RUNS=$W TEZ_PILOT_RESULTS=$W/results python experiments/small_model/teacher.py --fresh-state-cache \
      --num-shards 2 --shard-index $i --minutes 400 > /workspace/teacher-w$i.log 2>&1 &
  done
  wait
  touch /workspace/teacher.done
}

data() {
  venv
  python - <<'EOF'
import json, math, pathlib, shutil
root = pathlib.Path("/workspace/tez-pilot/runs")
v1, v2, out = root / "pilot-cloud-h200-20260927", root / "pilot-v2", root / "student-data-v2"
(out / "teacher").mkdir(parents=True, exist_ok=True)
rows = lambda p: [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
pool1, pool2 = rows(v1 / "pool.jsonl"), rows(v2 / "pool.jsonl")
as1, as2 = rows(v1 / "assign.jsonl"), rows(v2 / "assign.jsonl")
assert not {r["id"] for r in pool1} & {r["id"] for r in pool2}, "v1 and v2 state ids overlap"
reads = {}
for w in (0, 1):
    for p in sorted((root / f"pilot-v2-w{w}" / "teacher").glob("part-*.jsonl")):
        for r in rows(p):
            assert r["id"] not in reads, "duplicate teacher row"
            for q in r["questions"]:
                assert all(math.isfinite(x) for x in q["z_orig"] + q["z_rev"]), "non-finite logits"
            reads[r["id"]] = r
missing = {a["id"] for a in as2} - set(reads)
assert not missing, f"{len(missing)} assigned v2 states have no teacher reading"
(out / "pool.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in pool1 + pool2), encoding="utf-8")
(out / "assign.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in as1 + as2), encoding="utf-8")
for p in sorted((v1 / "teacher").glob("part-*.jsonl")):
    shutil.copy2(p, out / "teacher" / p.name)
ids = sorted(reads)
for k in range(0, len(ids), 500):
    (out / "teacher" / f"part-v2-{k // 500:04d}.jsonl").write_text(
        "".join(json.dumps(reads[i], ensure_ascii=False) + "\n" for i in ids[k:k + 500]), encoding="utf-8")
n2 = sum(len(reads[i]["questions"]) for i in ids)
print(json.dumps({"v1_states": len(pool1), "v2_states": len(reads), "v2_decisions": n2}))
EOF
  cp "$V1/manifest.json" "$DATA/manifest.json"
  touch /workspace/data.done
}

train() {
  venv
  TEZ_PILOT_RUNS=$DATA TEZ_PILOT_RESULTS=$ROOT/results/pilot-v2 TEZ_PILOT_TRAIN_ONLY_PREFIX=v2- \
    python experiments/small_model/train_c.py --arm $ARM --minutes 600 --max-segments 99
  touch /workspace/train.done
}

evaluate() {
  venv
  TEZ_PILOT_RUNS=$DATA TEZ_PILOT_RESULTS=$ROOT/results/pilot-v2 TEZ_PILOT_TRAIN_ONLY_PREFIX=v2- \
    TEZ_PILOT_EVAL_SUITES=$ROOT/runs/pilot/student-prep-20260927/eval-suites.json \
    python experiments/small_model/eval_arms.py --arm $ARM
  touch /workspace/eval.done
}

package() {
  tar -cf /workspace/v2-return.tar results/pilot-v2 runs/pilot-v2/schemas_raw.jsonl runs/pilot-v2/schemas.jsonl \
    runs/pilot-v2/assign.jsonl runs/pilot-v2/gold_questions.jsonl runs/student-data-v2/teacher \
    "runs/student-data-v2/models/$ARM" /workspace/*.log /workspace/environment.txt 2>/dev/null || true
  sha256sum /workspace/v2-return.tar
}

all() {   # every GPU stage in order, skipping finished ones, stopping at the first failure
  [ -f /workspace/schemas.done ] || schemas
  [ -f /workspace/teacher.done ] || teacher
  for w in 0 1; do grep -qE "Traceback" /workspace/teacher-w$w.log && { echo "teacher worker $w failed" >&2; exit 1; }; done
  [ -f /workspace/data.done ] || data
  [ -f /workspace/train.done ] || train
  [ -f /workspace/eval.done ] || evaluate
  package
  touch /workspace/all.done
}

case "${1:-}" in
  setup) setup ;; schemas) schemas ;; teacher) teacher ;; data) data ;; train) train ;; eval) evaluate ;; package) package ;;
  all) all ;;
  *) echo "usage: $0 setup|schemas|teacher|data|train|eval|package|all" >&2; exit 2 ;;
esac
