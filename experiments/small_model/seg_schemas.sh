#!/usr/bin/env bash
# One heavy segment (run under C:/temp/run_heavy.py, which holds the lock): the pool filter and clusters if missing,
# then the 12B writes schemas on its own llama-server (:8851) until MINUTES are nearly used; assign when all are
# written. Resumable: a rerun continues with the clusters not yet written. Always stops its server.
set -u
ROOT=/c/Users/migue/OneDrive/Desktop/Personal/AILABS/Tez-pilot
PY=$ROOT/.venv-pilot/Scripts/python.exe
GGUF=C:/Users/migue/.ollama/models/blobs/sha256-047dae1d7894b9de8f08141e841544e007243290c02df8b39872991d1940c795
LOG=$ROOT/runs/pilot/logs
PORT=8851
MINUTES=${1:-55}
START=$(date +%s)
cd "$ROOT/experiments/small_model" || exit 1
SERVER_PID=""
stop_server() {
  [ -n "$SERVER_PID" ] && kill "$SERVER_PID" 2>/dev/null
  # only our own server, found by its port (never another session's llama-server)
  powershell -NoProfile -Command "Get-CimInstance Win32_Process -Filter \"Name='llama-server.exe'\" | Where-Object { \$_.CommandLine -match '--port $PORT' } | ForEach-Object { Stop-Process -Id \$_.ProcessId -Force }" >/dev/null 2>&1
  SERVER_PID=""
}
trap stop_server EXIT
left() { echo $(( MINUTES - ($(date +%s) - START) / 60 )); }

if [ ! -f "$ROOT/runs/pilot/clusters.jsonl" ]; then
  $PY build_pool.py filter --device cuda > "$LOG/pool_filter.log" 2>&1 || { echo "filter failed"; exit 1; }
  $PY schemas.py cluster > "$LOG/cluster.log" 2>&1 || { echo "cluster failed"; exit 1; }
fi
python C:/temp/gpu_lock.py snapshot > "$LOG/vram_schemas_$(date +%H%M).json"

# One slot, 4,096 tokens: the production server's shape. 6 slots x 4,096 (2026-09-26 12:33) did not fit beside the
# 12.7 GB of weights and 1.4 GB of desktop VRAM: it spilled into shared memory (prompt 140 tok/s, decode 3-10 tok/s).
/c/temp/llamacpp/llama-server.exe -m "$GGUF" -ngl 99 -c 4096 -np 1 -fa on --port $PORT --host 127.0.0.1 \
  --no-webui -t "${RUN_HEAVY_THREADS:-6}" > "$LOG/llama_server_8851.log" 2>&1 &
SERVER_PID=$!
up=0
for i in $(seq 1 90); do
  if curl -s "http://127.0.0.1:$PORT/health" | grep -q '"ok"'; then up=1; break; fi
  kill -0 "$SERVER_PID" 2>/dev/null || break
  sleep 4
done
[ "$up" = 1 ] || { echo "llama-server did not start (see llama_server_8851.log)"; exit 1; }
echo "server up $(date -Iseconds)" >> "$LOG/schemas_write.log"
USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ')
echo "VRAM used after load: $USED MiB" >> "$LOG/schemas_write.log"
if [ "$USED" -gt 15600 ]; then echo "VRAM nearly full ($USED MiB): would spill, stopping" >> "$LOG/schemas_write.log"; exit 1; fi
nvidia-smi --query-gpu=temperature.gpu,clocks_throttle_reasons.sw_thermal_slowdown,memory.used --format=csv,noheader >> "$LOG/schemas_write.log"

# the day-1 decoding-rate check: single stream on 2 clusters, once
[ -f "$ROOT/results/pilot/schema_generation_timing_p1.json" ] || \
  $PY schemas.py write --server http://127.0.0.1:$PORT --parallel 1 --limit 2 >> "$LOG/schemas_write.log" 2>&1
$PY schemas.py write --server http://127.0.0.1:$PORT --parallel 1 --minutes $(( $(left) - 6 )) >> "$LOG/schemas_write.log" 2>&1
stop_server
nvidia-smi --query-gpu=temperature.gpu,clocks_throttle_reasons.sw_thermal_slowdown,memory.used --format=csv,noheader >> "$LOG/schemas_write.log"

# all clusters written? then validate, filter and assign (embeds the schema texts on the GPU)
n_c=$(wc -l < "$ROOT/runs/pilot/clusters.jsonl")
n_w=$($PY -c "import json;print(len({json.loads(l)['cid'] for l in open(r'$ROOT/runs/pilot/schemas_raw.jsonl',encoding='utf-8') if json.loads(l)['items']}))")
echo "clusters written: $n_w of $n_c" >> "$LOG/schemas_write.log"
if [ "$n_w" -ge "$n_c" ] && [ -s "$ROOT/runs/pilot/gold_questions.jsonl" ]; then
  $PY schemas.py assign --device cuda > "$LOG/schemas_assign.log" 2>&1 || { echo "assign failed"; exit 1; }
  exit 0
fi
exit 3
