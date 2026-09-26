#!/usr/bin/env bash
# Phase 1 of the Theioma voice design (docs/plans/theioma-voice-design.md, section 4e), in the order it was run.
# Run every step as one heavy job through the shared machine's wrapper (it takes the GPU lock, waits for free RAM and
# a quiet CPU, caps threads at 6 and runs below normal priority), one at a time with a pause between them:
#   python C:/temp/run_heavy.py --owner <you> --purpose "voice phase 1: data" --minutes 30 -- bash experiments/run_voice_phase1.sh data
# Every step writes results/voice/*.json with CPU load and VRAM snapshots.
#   bash experiments/run_voice_phase1.sh data      # Kokoro synthesis + word timings (GPU), whisper alignment of tez220
#   bash experiments/run_voice_phase1.sh asr       # streaming ASR spikes, ASR only, no pacing
#   bash experiments/run_voice_phase1.sh decide    # in-process decisions on a lag-free recogniser, real time
#   bash experiments/run_voice_phase1.sh probe     # the cut 4B: two-fold prefix probes (tez fit) and their replays
#   bash experiments/run_voice_phase1.sh e2e       # the recogniser and the session together, real time
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-C:/temp/voice-venv/Scripts/python}           # system Python 3.14 + sherpa-onnx (venv with site packages)
TTS_PY=${TTS_PY:-../Tez/.venv-tts/Scripts/python}      # Kokoro's PyTorch package (the TTS harness's environment)
LIB=C:/temp/llamacpp
BLOBS=C:/Users/migue/.ollama/models/blobs
Q8=$BLOBS/sha256-047dae1d7894b9de8f08141e841544e007243290c02df8b39872991d1940c795   # Gemma 4 12B Q8_0 (gemma4:12b-it-q8_0)
Q4=$BLOBS/sha256-1278394b693672ac2799eadc9a83fd98259a6a88a40acfb1dcaa6c6fc895a606   # Gemma 4 12B Q4_K_M (gemma4:12b)
Q4B=../Tez/tools/models/Qwen3.5-4B-Q8_0-L24.gguf
NEMO_EN=C:/temp/asr-models/nemotron-speech-streaming-en-0.6b.q8_0.gguf
NEMO_35=C:/temp/asr-models/nemotron-3.5-asr-streaming-0.6b.q8_0.gguf
FC80=C:/temp/asr-models/sherpa-onnx-nemo-streaming-fast-conformer-transducer-en-80ms-int8
EOU=C:/temp/asr-models/parakeet_realtime_eou_120m-v1/parakeet_realtime_eou_120m-v1.nemo
R="$PY experiments/voice_replay.py"

case "${1:-}" in
data)
  PYTHONIOENCODING=utf-8 $TTS_PY experiments/voice_tts.py --device cuda
  $PY experiments/voice_align.py --model large-v3 --device cuda --compute-type float16
  $PY experiments/voice_align.py --model large-v3 --device cuda --compute-type float16 --check data/voice/theioma
  ;;
asr)               # ASR only, audio pushed as fast as each recogniser takes it (no pacing): step compute, emission, WER
  for s in theioma tez220; do
    $R --set $s --asr nemocpp:$NEMO_EN --rc 0 --no-realtime --gap 0 --tag ${s}_asr_nemotron_en_80; sleep 60
    $R --set $s --asr nemocpp:$NEMO_EN --rc 1 --no-realtime --gap 0 --tag ${s}_asr_nemotron_en_160; sleep 60
  done
  $R --set theioma --asr nemocpp:$NEMO_35 --rc 0 --no-realtime --gap 0 --tag theioma_asr_nemotron35_80; sleep 60
  $R --set theioma --asr whisper:base.en --no-realtime --gap 0 --tag theioma_asr_whisper_base_200; sleep 60
  $R --set theioma --asr nemo:$EOU --rc 1 --no-realtime --gap 0 --tag theioma_asr_parakeet_eou_160 || true
  ;;
asr-cpu)           # the CPU recognisers, alone (never beside a GPU run)
  $R --set theioma --asr sherpa:$FC80 --asr-threads 4 --no-realtime --gap 0 --tag theioma_asr_sherpa_fc80_cpu4; sleep 60
  $R --set theioma --asr nemocpp:$NEMO_EN --asr-gpu -1 --rc 0 --no-realtime --gap 0 --limit 60 --tag theioma_asr_nemotron_en_80_cpu
  ;;
decide)            # the primary arm: Gemma 4 12B Q4_K_M in process (Theioma's shared 12B), ~40 min
  $R --set theioma --asr oracle --backend inproc:$Q4 --lib $LIB --parity 20 --tag theioma_oracle_12b_q4km; sleep 60
  $R --set theioma --asr oracle --backend inproc:$Q4 --lib $LIB --commit fold --tag theioma_oracle_12b_q4km_fold; sleep 60
  $R --set tez220 --asr oracle --backend inproc:$Q4 --lib $LIB --parity 20 --tag tez220_oracle_12b_q4km
  ;;
decide-q8)         # Q8_0 in process, ~30 min
  $R --set theioma --asr oracle --backend inproc:$Q8 --lib $LIB --tag theioma_oracle_12b_q8; sleep 60
  $R --set tez220 --asr oracle --backend inproc:$Q8 --lib $LIB --tag tez220_oracle_12b_q8
  ;;
decide-http)       # the Q4_K_M over HTTP from a llama-server of our own with :8797's flags, ~30 min
  $PY experiments/voice_server_arm.py --gguf $Q4 -- --set theioma --asr oracle --tag theioma_oracle_12b_q4km_http; sleep 60
  $PY experiments/voice_server_arm.py --gguf $Q4 -- --set tez220 --asr oracle --tag tez220_oracle_12b_q4km_http
  ;;
probe)             # the cut 4B: two-fold prefix-labelled probes (tez fit) and their held-out oracle replays, ~40 min
  for s in theioma tez220; do
    $PY experiments/voice_probe_labels.py --set $s --out C:/temp/voice-fit
    for f in 0 1; do
      d=C:/temp/voice-fit/$s/fold$f
      $PY experiments/voice_probe_fit.py --fold $d --model $Q4B --lib $LIB
      $R --set $s --asr oracle --backend inproc:$Q4B --template qwen3 --lib $LIB --schema $d/schema.yaml           --ids $d/ids.txt --tag ${s}_oracle_4b_probe_fold$f; sleep 30
    done
  done
  ;;
e2e)               # composed end to end: job 1's recorded Nemotron 80 ms emissions, each fed at its audio time + its
                   # step compute into the live 12B Q4_K_M session (the ASR itself is not loaded: less RAM), ~35 min
  $R --set theioma --asr replay:results/voice/replay_theioma_asr_nemotron_en_80_rows.jsonl --backend inproc:$Q4 --lib $LIB      --tag theioma_e2e_nemotron80_12b_q4km; sleep 60
  $R --set tez220 --asr replay:results/voice/replay_tez220_asr_nemotron_en_80_rows.jsonl --backend inproc:$Q4 --lib $LIB      --tag tez220_e2e_nemotron80_12b_q4km; sleep 30
  $R --set theioma --asr nemocpp:$NEMO_EN --rc 0 --no-realtime --gap 0 --limit 20 --tag theioma_asr_nemotron_en_80_vram
  ;;
e2e-live)          # the recogniser and the session loaded together, real time (needs a quiet machine: RAM), ~40 min
  $R --set theioma --asr nemocpp:$NEMO_EN --rc 0 --backend inproc:$Q4 --lib $LIB --tag theioma_e2elive_nemotron80_12b_q4km; sleep 60
  $R --set theioma --asr replay:results/voice/replay_theioma_asr_whisper_base_200_rows.jsonl --backend inproc:$Q4 --lib $LIB      --tag theioma_e2e_whisper200_12b_q4km; sleep 30
  $R --set theioma --asr nemo:$EOU --rc 1 --no-realtime --gap 0 --tag theioma_asr_parakeet_eou_160 || true
  ;;
*)
  echo "usage: $0 data|asr|asr-cpu|decide|decide-q8|decide-http|probe|e2e|e2e-live" >&2
  exit 2
  ;;
esac
