# Voice Phase 1: results

The fast path of Theioma's voice commands on the Tez side, measured against the gates of
`docs/plans/theioma-voice-design.md` §4e (Phase 0 and Phase 1). Branch `voice-fast`. Every number below comes from a
file in `results/voice/` (manifest: data hashes, models, settings, CPU load, VRAM and GPU thermal state before, during
and after), produced by the scripts named with it. Runs were made on the RTX 5080 Laptop (16 GB) through the shared
machine's `C:/temp/run_heavy.py` (one heavy job at a time, below-normal priority, 6 threads), and a timed arm started
only with the GPU below 70 °C and no thermal slowdown; a run that saw a slowdown is marked [throttled].

DRAFT: the GPU arms are filled in as they finish.

## 1. What was built

### `tez.stream`: a streaming decision session in the runtime

```python
from tez import Tez

tez = Tez(backend="inproc:gemma-4-12b-it-Q4_K_M.gguf", template="gemma4", schemas="data/voice/theioma/schema.yaml",
          inproc={"lib": "C:/temp/llamacpp", "n_ctx": 2048, "n_seq_max": 8, "n_outputs": 64, "n_threads": 6})
session = tez.stream(schema="theioma-voice", on_action=execute, on_undo=retract).load()   # prefix evaluated once

u = session.feed("open")                 # a complete word
u = session.update("open no")            # or a recogniser's text as it grows: the last word may still change
u.decision, u.confidence, u.committed, u.action, u.stable, u.timings["decide_ms"]
u = session.end()                        # deferred (R2) actions act, R3 becomes an approval request, none hands off
session.undo()                           # retract the last action
session.reset()                          # next utterance
```

- **One session, one choice question.** It answers a choice question of a schema after every word or recogniser
  step. The layout is question first: instructions and options, then the transcript as the state.
- **Three sequences per prompt in process.** With `InprocBackend`, `InprocStreamReader` keeps:
  - the constant prefix, evaluated once and never touched again;
  - the committed transcript;
  - a scratch copy, which takes the new words plus the answer tail for each decision.
- **Committing words.** By default (`commit="gap"`) the word is committed after the decision and the dispatch, in the
  gap before the next word. With `commit="fold"` it is folded into the next decision's decode, so each step is one
  forward pass.
- **No rollback.** Nothing is ever rolled back, so hybrid Qwen3.5 models work the same way. A revised transcript is
  rebuilt from the prefix.
- **Same answers as a request.** `Tez.decide_question` itself reads the answers from the stream's reads. Letters,
  probes, blends, `__none__`, fitted temperatures and `Tez.unfitted_temperature` therefore apply exactly as in a
  `/v1/systemone` request. Tests confirm the probabilities equal a whole-prompt `decide_question` to 1e-6 on the stub,
  word by word, for:
  - letters and probes;
  - an abstaining question;
  - temperature auto, off and a number;
  - hybrid models;
  - partial words;
  - folded commits.
- **Other backends.** llama-server over HTTP and `FakeBackend` also work. They read the whole prompt for every word,
  and llama-server's prompt cache reuses the prefix.
- **The early-commit policy is data.** It comes from `experiments/stream_policy.py`, now written as the schema's
  `actions:` block:
  - each option gets a class: R0 instant, R1 refinable (with the ancestor it `refines`), R2 deferred, or R3 gated;
  - each option also gets a stability;
  - `Policy` reproduces `stream_policy.executed_sequence` on random trajectories (tested).
- **Policy behaviour.**
  - `none` on a prefix keeps listening; `none` at the end sets `handoff`.
  - `undo()` retracts an action.
  - `Policy(retract=True)` retracts early actions that the final decision does not confirm.
  - `Policy(residual=True)` reads what follows an early action with a prior line.
  - `Policy.replay()` re-scores logged decisions with another threshold or temperature.

Files:

- `tez/stream.py`
- `tez/inproc.py` (`InprocStreamReader`, `n_outputs` and `n_threads` options, sequence reservation)
- `tez/schema.py` (`actions:`)
- `tez/engine.py` (`Tez.stream`)
- `tests/test_stream.py` (23 tests; the whole offline suite stays green)

### The replay harness: `experiments/voice_replay.py`

- **Replay.** A set is replayed word-aligned in real time:
  - audio is pushed as 10 ms frames, each at the moment its last sample would have been captured, on the
    `perf_counter` (QPC) clock;
  - a streaming recogniser runs in process, or in a worker process that paces the audio itself;
  - every step that changes the transcript becomes `session.update(text)`;
  - actions go to a no-op native executor, which makes one ctypes Win32 call and time-stamps it.
- **Oracle mode.** `--asr oracle` feeds each reference word at its forced end time, which isolates the decision and
  the dispatch.
- **What is logged.** Every emission and update, with timings and full distributions, goes to `*_rows.jsonl`.
- **Metrics.**
  - word end to decision, and to first action, p50/p95, for the decisive word and for every word;
  - the projected word end to action (decision plus the measured dispatch);
  - ASR step compute, busy time per chunk, emission delay and lag beyond the chunk, WER;
  - intent accuracy, strict and accepting then/alt;
  - harmful actions, out-of-scope false actions, first-action word;
  - parity of the stream against whole-prompt reads.
- **Tools.**
  - `--rescore` re-runs the policy on logged decisions (threshold sweep, retempering).
  - `--report` prints one line per run.

Supporting scripts:

- `experiments/voice_asr.py`: the recognisers behind one interface:
  - NeMo-Speech.cpp v0.1.0 through its C ABI (cache-aware Nemotron at 80/160 ms);
  - sherpa-onnx;
  - faster-whisper with re-decoding;
  - NeMo's own PyTorch cache-aware step.
- `experiments/voice_server_arm.py`: our own llama-server with the flags of Theioma's :8797, for the shared-server arm.
- `experiments/voice_probe_labels.py`, `voice_probe_fit.py`: the two-fold, prefix-labelled `tez fit` of the cut-4B
  probe.
- `experiments/run_voice_phase1.sh`: every step, in order.

## 2. Data (Phase 0)

| set | utterances | speech | labels | word timings |
|---|---:|---|---|---|
| `tez220` (`data/voice`) | 220 (198 actionable, 22 out of scope) | Windows SAPI, one voice (Tez's existing WAVs) | Tez's 16 desktop actions, human decisive word | CTC forced alignment |
| `theioma` (`data/voice/theioma`) | 269 (239 actionable, 30 out of scope, 25 compound, 15 ambiguous) | **synthetic**: Kokoro-82M v1.0 on CUDA, six voices in turn (af_heart, am_michael, bf_emma, bm_george, af_bella, am_adam), 16 kHz | 23 tools of Theioma's voice registry + none, decisive word, slots, then/alt | Kokoro's phoneme durations, ends cut to the speech energy |

**The Theioma command set.** The 23 options are tool names from `apps/web/src/lib/voice-tools` (read only), chosen to
cover every action class of the design:

- **R0:** opens, media keys, dashboard pages, window size, read-only questions.
- **R1:** play on YouTube/Spotify, web search, a project's section.
- **R2:** typing, messages to agents, reminders, delegation, spoken answers, volume level.
- **R3:** approvals, closing windows, killing processes, shell commands.

There are 269 hand-written utterances:

| style | count |
|---|---:|
| canonical | 76 |
| paraphrase | 70 |
| noisy | 53 |
| compound | 25 |
| ambiguous | 15 |
| out of scope | 30 |

Fourteen of the out-of-scope rows name an app, page or session ("youtube is down again"). Mean length is 6.0 words.
The decisive word is word 3.7 on average, and it is the last word in 26 % of the actionable commands.

**Why the word timings were replaced.** The harness measures latency from the end of the decisive word, so word ends
must be accurate to tens of milliseconds.

- **Kokoro's own timings** are the phoneme durations the audio was generated from. The last word's end includes the
  trailing pause, 245 ms past the end of the speech energy. Each end is therefore cut to the end of the speech energy
  inside the word's own window (−35 dB), which moves last words by 260 ms at the median and leaves connected speech
  unchanged (`results/voice/align_trim_theioma.json`).
- **faster-whisper large-v3 word timestamps** are cross-attention estimates, not an alignment. The first alignment of
  the SAPI set used them. Checked against the Kokoro timings, their word ends are 105 ms early at the median (mean
  −188 ms) and 528 ms off at the 90th percentile (`results/voice/align_check_large-v3.json`). That is as large as the
  latencies being measured, so they were replaced.
- **CTC forced alignment** (`experiments/voice_ctc_align.py`) replaced them. It uses wav2vec2-base-960h (20 ms
  frames) and a Viterbi pass of the reference characters through its posteriors.
  - Against the Kokoro timings, its word ends are 10 ms early at the median (MAE 32 ms, p10 −80, p90 +25).
  - On the last word they are 60 ms early (`results/voice/align_ctc_theioma_check.json`): CTC's last character peaks
    before the sound dies away.
  - The SAPI set now uses these ends; whisper's are kept as `end_whisper` (`results/voice/align_ctc_tez220.json`).
  - The residual bias makes latencies on `tez220` look slightly longer than they are (conservative). The `theioma` set
    has exact timings and is the primary set.

## 3. Phase 1 spikes

### (a) Streaming ASR

(pending: job 1)

### (b) The decision in process

(pending: jobs 2-4)

### (c) VRAM of the stack

(pending)

## 4. End to end

(pending: job 5)

## 5. Gates

(pending)

## 6. What Phase 2 needs

(pending)
