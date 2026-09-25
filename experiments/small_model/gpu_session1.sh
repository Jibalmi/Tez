#!/usr/bin/env bash
# Pilot GPU session 1 (day 2): the 12B writes the schemas (own llama-server on :8851), then reads every decision in
# process. Queues on the cooperative GPU lock, holds it at most ~2 h, always stops its server and releases.
set -u
ROOT=/c/Users/migue/OneDrive/Desktop/Personal/AILABS/Tez-pilot
PY=$ROOT/.venv-pilot/Scripts/python.exe
LOCK="python C:/temp/gpu_lock.py"
OWNER=projects-0c-pilot
GGUF=C:/Users/migue/.ollama/models/blobs/sha256-047dae1d7894b9de8f08141e841544e007243290c02df8b39872991d1940c795
LOG=$ROOT/runs/pilot/logs
PORT=8851
cd "$ROOT/experiments/small_model" || exit 1

until [ -f "$ROOT/data/small_model/eval_hashes.json.gz" ] && [ -f "$ROOT/runs/pilot/eval_texts.jsonl" ]; do sleep 20; done
$LOCK acquire --owner $OWNER --purpose "Tez small-model pilot: 12B writes schemas (llama-server :8851) then labels 30k decisions in process" --minutes 120 --wait
START=$(date +%s)
echo "lock acquired $(date -Iseconds)" | tee -a "$LOG/gpu_session1.log"
$LOCK snapshot > "$LOG/vram_session1_start.json"
SERVER_PID=""
stop_server() {
  if [ -n "$SERVER_PID" ]; then kill "$SERVER_PID" 2>/dev/null; sleep 2; fi
  # only our own server: found by its port in the command line (never another session's llama-server)
  powershell -NoProfile -Command "Get-CimInstance Win32_Process -Filter \"Name='llama-server.exe'\" | Where-Object { \$_.CommandLine -match '--port $PORT' } | ForEach-Object { Stop-Process -Id \$_.ProcessId -Force }" >/dev/null 2>&1
  SERVER_PID=""
}
cleanup() {
  stop_server
  $LOCK release --owner $OWNER
  echo "released $(date -Iseconds) after $(( ($(date +%s) - START) / 60 )) min" | tee -a "$LOG/gpu_session1.log"
}
trap cleanup EXIT

start_server() {   # $1 slots, $2 context
  /c/temp/llamacpp/llama-server.exe -m "$GGUF" -ngl 99 -c "$2" -np "$1" -fa on -ctk q8_0 -ctv q8_0 --port $PORT \
     --host 127.0.0.1 --no-webui > "$LOG/llama_server_8851.log" 2>&1 &
  SERVER_PID=$!
  for i in $(seq 1 90); do
    if curl -s "http://127.0.0.1:$PORT/health" | grep -q '"ok"'; then echo "server up: $1 slots, ctx $2" | tee -a "$LOG/gpu_session1.log"; return 0; fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then break; fi
    sleep 4
  done
  echo "server failed: $1 slots, ctx $2" | tee -a "$LOG/gpu_session1.log"
  stop_server
  return 1
}

# --- day-1 leftovers that need the GPU for speed: embeddings of the evaluation and pool states, filter, clusters
if [ ! -f "$ROOT/runs/pilot/clusters.jsonl" ]; then
  [ -f "$ROOT/runs/pilot/eval_emb.npz" ] || $PY eval_hashes.py embed --device cuda >> "$LOG/eval_embed.log" 2>&1
  $PY build_pool.py filter --device cuda > "$LOG/pool_filter.log" 2>&1
  $PY schemas.py cluster > "$LOG/cluster.log" 2>&1
fi
[ -f "$ROOT/runs/pilot/clusters.jsonl" ] || { echo "no clusters: stopping" | tee -a "$LOG/gpu_session1.log"; exit 1; }

# --- schemas: a single-stream decoding measurement first (day 1's rate check), then the parallel slots
if [ ! -f "$ROOT/runs/pilot/schemas.jsonl" ]; then
  if start_server 4 16384 || start_server 3 12288 || start_server 2 8192; then
    SLOTS=$(grep "server up" "$LOG/gpu_session1.log" | tail -1 | sed 's/server up: \([0-9]*\) slots.*/\1/')
    $LOCK snapshot > "$LOG/vram_session1_server.json"
    $PY schemas.py write --server http://127.0.0.1:$PORT --parallel 1 --limit 2 >> "$LOG/schemas_write.log" 2>&1
    $PY schemas.py write --server http://127.0.0.1:$PORT --parallel "$SLOTS" >> "$LOG/schemas_write.log" 2>&1
    stop_server
    sleep 5
    $PY schemas.py assign --device cuda >> "$LOG/schemas_assign.log" 2>&1
  fi
fi

# --- teacher: in process, until the 2-hour hold is nearly used (resumable)
if [ -f "$ROOT/runs/pilot/assign.jsonl" ]; then
  USED=$(( ($(date +%s) - START) / 60 ))
  LEFT=$(( 112 - USED ))
  if [ "$LEFT" -gt 5 ]; then
    $PY teacher.py --minutes "$LEFT" --n-ctx 4096 >> "$LOG/teacher.log" 2>&1
  fi
fi
$LOCK snapshot > "$LOG/vram_session1_end.json"
