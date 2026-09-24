#!/usr/bin/env bash
# Follow-up to speed_run_4b.sh: the in-process probe runs (they need embeddings switched on only for the decodes whose
# states are read -- libllama outputs every token of a decode while embeddings are on, see speed_llama.set_embeddings).
# Needs the GPU lock and no other model on the GPU.
#   bash experiments/speed_run_4b_probes.sh
set -u
cd "$(dirname "$0")/.."
L24="$(pwd -W)/tools/models/Qwen3.5-4B-Q8_0-L24.gguf"
L32="$(pwd -W)/tools/models/Qwen3.5-4B-Q8_0.gguf"
renew() { python C:/temp/gpu_lock.py acquire --owner projects-37-speedlab --purpose "Tez latency measurements" --minutes 60; }

renew
python experiments/speed_voice.py --engine inproc --readout probe --model "$L24" --template qwen3 --gap 0.25 --cool 85 --tag q4bL24_probe
python experiments/speed_voice.py --engine inproc --readout probe_tail --model "$L24" --template qwen3 --gap 0.25 --cool 85 --tag q4bL24_probetail
python experiments/speed_voice.py --engine inproc --readout probe_tail --model "$L32" --template qwen3 --gap 0.25 --cool 85 --tag q4b_probetail
renew
python experiments/speed_probe_multiq.py extract-inproc --layout stateonly --model "$L24" --tag L24_stateonly_inproc
python experiments/speed_probe_multiq.py extract-inproc --layout sfq --model "$L24" --tag L24_sfq
python experiments/speed_probe_multiq.py latency --engine inproc --layout stateonly,sfq,qprompt --model "$L24" --cool 85 --tag L24_inproc
python experiments/speed_probe_multiq.py accuracy --feats L24_stateonly,L24_stateonly_inproc,L24_sfq --with-reference
python experiments/speed_multiq_inproc.py --mode laya --unique --model "$L24" --template qwen3 --arms batch_today,batch_statefirst --n-batch 4096 --gap 0.3 --duty 1.0 --cool 85 --warmup 2 --reps 12 --reps50 4 --tag q4bL24_inproc_unique   # sequential arms need KV rollback, which hybrid (recurrent) memory cannot do
echo "4B probe chain done"
