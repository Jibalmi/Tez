#!/usr/bin/env bash
# Deferred-commit voice runs: the decision decodes a single sequence (a copy of the committed transcript + the new word +
# the closing part), and the new word is committed to the base sequence after the decision, in the gap before the next
# word (experiments/speed_voice.py --defer). Needs the GPU lock and a free GPU (the 12B does not fit next to a server).
#   bash experiments/speed_run_voice_defer.sh
set -u
cd "$(dirname "$0")/.."
L24="$(pwd -W)/tools/models/Qwen3.5-4B-Q8_0-L24.gguf"
GEMMA="C:/Users/migue/.ollama/models/blobs/sha256-047dae1d7894b9de8f08141e841544e007243290c02df8b39872991d1940c795"
python C:/temp/gpu_lock.py acquire --owner projects-37-speedlab --purpose "Tez latency measurements" --minutes 60
python experiments/speed_voice.py --engine inproc --readout probe_tail --defer --model "$L24" --template qwen3 --gap 0.25 --cool 85 --tag q4bL24_probetail_defer
python experiments/speed_voice.py --engine inproc --readout probe_end --defer --model "$L24" --template qwen3 --gap 0.25 --cool 85 --tag q4bL24_probeend_defer
python experiments/speed_voice.py --engine inproc --readout letters --defer --model "$GEMMA" --template gemma4 --gap 0.25 --cool 85 --tag 12b_inproc_defer
echo "voice defer chain done"
