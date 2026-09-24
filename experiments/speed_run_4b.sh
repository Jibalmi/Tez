#!/usr/bin/env bash
# The Qwen3.5-4B part of the speed study (Exp 2 probes, Exp 3 voice with the 4B, Exp 1 with the 4B letters).
# Needs the GPU lock (python C:/temp/gpu_lock.py acquire --owner <you> ...) and no other model on the GPU; the
# production llama-server on :8091 must be stopped (restore it afterwards with experiments/speed_servers.ps1 start-prod).
# Every timed script stores before/after VRAM snapshots (gpu_guard) in its JSON.
#   bash experiments/speed_run_4b.sh
set -u
cd "$(dirname "$0")/.."
PS="powershell -ExecutionPolicy Bypass -File experiments/speed_servers.ps1"
L24="$(pwd -W)/tools/models/Qwen3.5-4B-Q8_0-L24.gguf"
L32="$(pwd -W)/tools/models/Qwen3.5-4B-Q8_0.gguf"
BASE="-ngl 99 -c 4096 -b 512 -np 1 --no-webui --embeddings --pooling last"
renew() { python C:/temp/gpu_lock.py acquire --owner projects-37-speedlab --purpose "Tez latency measurements" --minutes 60; }

renew
# ---- served 24-block model, one slot (caching left on: exact extension is safe, see qwen_cache_check_*.json)
$PS start -Port 8095 -Model "$L24" -Base "$BASE" -Extra "--cache-ram 0"
python experiments/speed_llama.py --model "$L24" --template qwen3 --url http://127.0.0.1:8095      # in-process vs server logits
python experiments/speed_voice.py --engine http --url http://127.0.0.1:8095 --template qwen3 --cache off --gap 0.25 --cool 85 --tag q4bL24_http_nocache
python experiments/speed_voice.py --engine http --url http://127.0.0.1:8095 --template qwen3 --readout probe --cache on --gap 0.25 --cool 85 --tag q4bL24_probe_http
python experiments/speed_probe_multiq.py extract --layout stateonly --url http://127.0.0.1:8095 --tag L24_stateonly
python experiments/speed_probe_multiq.py latency --engine http --layout stateonly --url http://127.0.0.1:8095 --cool 85 --tag L24_http
$PS stop -Port 8095
renew
# ---- served 24-block model, eight slots: the question-in-prompt probe batched across questions
$PS start -Port 8095 -Model "$L24" -Base "-ngl 99 -c 8192 -b 512 -np 8 -kvu --no-webui --embeddings --pooling last" -Extra "--cache-ram 0"
python experiments/speed_probe_multiq.py latency --engine http --layout qprompt --url http://127.0.0.1:8095 --cool 85 --tag L24_httpnp8
python experiments/speed_probe_multiq.py latency --engine http --layout qprompt --multi --url http://127.0.0.1:8095 --cool 85 --tag L24_httpnp8multi
$PS stop -Port 8095
renew
# ---- in-process
python experiments/speed_voice.py --engine inproc --model "$L24" --template qwen3 --gap 0.25 --cool 85 --tag q4bL24_inproc
python experiments/speed_voice.py --engine inproc --readout probe --model "$L24" --template qwen3 --gap 0.25 --cool 85 --tag q4bL24_probe
python experiments/speed_voice.py --engine inproc --model "$L32" --template qwen3 --gap 0.25 --cool 85 --tag q4b_inproc
renew
python experiments/speed_probe_multiq.py extract-inproc --layout sfq --model "$L24" --tag L24_sfq
python experiments/speed_probe_multiq.py latency --engine inproc --layout stateonly,sfq,qprompt --model "$L24" --cool 85 --tag L24_inproc
python experiments/speed_multiq_inproc.py --mode td --model "$L24" --template qwen3 --arms batch_statefirst,batch_today --n-batch 4096 --tag q4bL24_inproc
python experiments/speed_multiq_inproc.py --mode laya --unique --model "$L24" --template qwen3 --arms seq_today,seq_statefirst,batch_statefirst --n-batch 4096 --gap 0.3 --duty 1.0 --cool 85 --warmup 2 --reps 12 --reps50 4 --tag q4bL24_inproc_unique
python experiments/speed_probe_multiq.py accuracy --feats L24_stateonly,L24_sfq --with-reference
echo "4B chain done"
