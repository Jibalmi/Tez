# Voice Phase 1: results

The fast path of Theioma's voice commands on the Tez side, measured against the gates of
`docs/plans/theioma-voice-design.md` §4e (Phase 0 and Phase 1). Branch `voice-fast`. Every number below comes from a
file in `results/voice/` (manifest: data hashes, models, settings, CPU load, VRAM and GPU thermal state before, during
and after), produced by the scripts named with it. Runs were made on the RTX 5080 Laptop (16 GB) through the shared
machine's `C:/temp/run_heavy.py` (one heavy job at a time, below-normal priority, 6 threads), and a timed arm started
only with the GPU below 70 °C and no thermal slowdown; a run that saw a slowdown is marked [throttled].

**Verdict in one paragraph.** The runtime piece is done and tested: `tez.stream` decides word by word over the
in-process backend with Tez's own answers and temperatures. On the replay sets, the decision itself is fast (12B
Q4_K_M in process: 29 ms p50 per word; the cut 4B probe: 21 ms). The ASR step is cheap (Nemotron Speech Streaming
at 80 ms: 4.7 ms per chunk), and the dispatch is free (a few microseconds). The 50 ms word-to-action target is **not**
met. The recogniser emits the decisive word about 210 ms after the word ends (mean 216 ms, against a 20 ms go/no-go),
and that lag, not compute, sets the latency. Composed end to end, an instant (R0) command's first action lands 173 ms
p50 after its decisive word, 30 % of them before it ends and 36 % within 50 ms. Accuracy holds at Q4 (0.921 on the
Theioma set, within half a point of Q8). Harmful early actions on the Theioma set are too many (6.7 % with a perfect
recogniser), mostly from overlapping tools in its schema. Section 5 gives the gate table; section 6 lists what would
fix each failure and what Phase 2 needs.

Runs made while other sessions loaded the machine are marked [loaded] (system CPU mean above 30 %), and runs that
saw a thermal slowdown are marked [throttled].

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

### (a) Streaming ASR on Windows

These arms measure the recogniser alone: the whole set, with audio pushed as fast as each recogniser takes it (the
GPU is cool at the start of each arm). Every recogniser listed runs on Windows without WSL:

- NVIDIA's NeMo-Speech.cpp v0.1.0 (ggml, CUDA), loaded through its stable C ABI with ctypes;
- NeMo 2.7.3's own PyTorch step;
- sherpa-onnx 1.13.8;
- faster-whisper.

The tables are `experiments/voice_phase1_tables.py` over `results/voice/replay_*_asr_*.json`.

The columns:

- **busy / chunk** is the compute per recogniser step, which Phase 1 gates at 8 ms p50.
- **complete** is the time after the end of the decisive word at which the transcript first holds that whole word.
- **prefix** is the same for a two-letter prefix of it ("spot" for "spotify"). The session already acts on
  prefixes.
- Both are in audio time (the audio position of the emitting step); adding the step's compute moves them by the busy
  time.

| recogniser (set) | chunk | WER | busy / chunk p50 (p95) ms | complete: mean / p50 / p95 ms | prefix: mean / p50 ms, share before the word ends | GPU °C start->max |
|---|---:|---:|---:|---:|---:|---|
| **Nemotron Speech Streaming En 0.6B Q8_0**, NeMo-Speech.cpp, right context 0 (theioma) | 80 ms | **0.080** | **4.68** (5.79) | **216 / 208 / 500** | 128 / 125, 31 % | 69->81 |
| same, right context 1 (theioma) | 160 ms | 0.079 | 5.53 (6.93) | 223 / 218 / 527 | 144 / 138, 28 % | 68->77 |
| Nemotron 3.5 ASR Streaming 0.6B Q8_0, right context 0 (theioma) | 80 ms | 0.099 | 4.49 (5.14) | 316 / 330 / 528 | 150 / 158, 19 % | 68->81 |
| Nemotron Speech Streaming En, 80 ms (tez220, SAPI) | 80 ms | 0.045 | 4.46 (5.36) | 243 / 230 / 450 | 186 / 170, 13 % | 69->79 |
| same, 160 ms (tez220) | 160 ms | 0.042 | 5.23 (6.71) | 247 / 250 / 470 | 188 / 170, 11 % | 69->76 |
| faster-whisper base.en fp16, re-decoding the buffer every 200 ms (theioma) [throttled] | 200 ms | 0.104 | 42.6 (65.7) | **−11 / −30 / 314** | −71 / −80, 64 % | 65->89 |

**Measured in this spike.**

- **Nemotron's compute is small.** Nemotron Speech Streaming En at 80 ms (NeMo-Speech.cpp) takes 4.7 ms per 80 ms
  chunk. That passes the 8 ms gate with room to spare. The recogniser process holds 1.12 GB of VRAM after warm-up
  (`replay_theioma_asr_nemotron_en_80_vram.json`).
- **Nemotron's emission lag is ten times the go/no-go.**
  - On the Kokoro set, the decisive word is complete in the transcript 216 ms after it ends (mean; p50 208 ms,
    p95 500 ms). Beyond the 80 ms chunk that holds the word's end, that is still 186 ms.
  - A prefix of it appears at 128 ms (mean), and 31 % of prefixes appear before the word ends.
  - The 160 ms setting and Nemotron 3.5 are no better.
  - On SAPI speech, the lag is about 245 ms. The CTC word ends it is measured against run 10–60 ms early, so this
    figure is somewhat high.
- **faster-whisper completes words earlier.** Re-decoding the whole buffer from the start, its decoder completes a
  word from partial audio: complete at −11 ms (mean), and 64 % of prefixes before the word ends. Its costs:
  - WER 0.104 against Nemotron's 0.080;
  - 43 ms of compute per 200 ms step, and it grows with the utterance;
  - partials that flicker ("oh", "loon", "stunned" before "open", "launch", "start");
  - this arm ran hot ([throttled]). Its paced re-run is the not-yet-run last job (section 4).
- **sherpa-onnx.** sherpa-onnx 1.13.8 runs the cache-aware Nemotron export
  (`sherpa-onnx-nemotron-speech-streaming-en-0.6b-2026-01-14`) and the 80 ms NeMo FastConformer transducer on
  Windows, but only on the CPU wheel. Light CPU checks, not recorded runs (so no numbers are quoted), were far
  slower per step than the GPU path and no earlier in emission, so no arm was run. The rules keep CPU recognisers out
  of the GPU phase, and no GPU sherpa-onnx wheel for Windows was available to try.
- **Parakeet realtime EOU 120M.** It runs through NeMo's PyTorch step, and on the CPU check it emits `<EOU>`
  (`experiments/voice_asr.py` `nemo:`). Its GPU arm is in the not-yet-run last job.

### (b) The decision in process

With a lag-free recogniser (`--asr oracle`: each reference word arrives at its forced end time, in real time), this
measures the Tez session alone. Decision is the `session.update` wall time per word; word end -> decision is the same
from the decisive word's end, on the `perf_counter` clock. Every arm here started below 70 °C and saw no slowdown.

| arm (set) | decide p50 (p95) ms | word end -> decision p50 (p95) | intent acc. strict / then-alt | harmful / actionable | out-of-scope false | model VRAM |
|---|---:|---:|---:|---:|---:|---:|
| **Gemma 4 12B Q4_K_M in process** (theioma) | **29.4** (37.5) | 31.3 (40.9) | **0.921** / 0.971 | 16 / 239 (12 early) | 6 / 30 | 8.05 GB |
| same, folded commits (theioma) | 30.4 (37.1) | 31.9 (41.6) | 0.916 / 0.971 | 16 / 239 | 6 / 30 | 8.05 GB |
| Gemma 4 12B Q4_K_M in process (tez220) | 28.7 (35.5) | 30.4 (35.8) | 0.874 / 0.980 | **1 / 198** | 5 / 22 | 8.05 GB |
| Gemma 4 12B Q8_0 in process (theioma) | 37.9 (42.5) | 39.9 (48.3) | 0.925 / 0.979 | 7 / 239 | 3 / 30 | 13.09 GB |
| Gemma 4 12B Q8_0 in process (tez220) | 37.9 (41.3) | 38.8 (41.1) | 0.879 / 0.980 | 1 / 198 | 4 / 22 | 13.09 GB |
| Gemma 4 12B Q4_K_M **over HTTP** (own llama-server, :8797's flags) (theioma) | 36.9 (45.4) | 39.0 (48.8) | 0.912 / 0.971 | 20 / 239 | 6 / 30 | 8.72 GB |
| same over HTTP (tez220) | 37.3 (45.2) | 38.2 (45.6) | 0.874 / 0.980 | 2 / 198 | 5 / 22 | 8.72 GB |
| Qwen3.5-4B cut to 24 blocks, prefix probe, 2 folds pooled (theioma) | 21.4 (25.2) | 22.4-23.3 p50 | 0.816 / 0.862 | 14 / 239 | 4 / 30 | 4.02 GB |
| same (tez220) | 20.4 (26.0) | 19.7-20.8 p50 | 0.879 / 0.889 | 6 / 198 | 3 / 22 | 4.02 GB |

With the perfect recogniser, the first action of an instant (R0) command lands this long after the end of its
decisive word:

| decision model | p50 | p95 | ≤ 50 ms | before the word ends |
|---|---:|---:|---:|---:|
| 12B Q4_K_M in process | **33.7 ms** | 497 ms | 67 % | 21 % |
| 12B Q8_0 | 45.3 ms | 751 ms | 58 % | 6 % |
| 12B Q4_K_M over HTTP | 40.4 ms | 509 ms | 66 % | — |

The p95 comes from R0 actions that wait for a second stable word by policy (media keys, window size) or that the
model decides only on a later word. These are the Theioma set's 100 R0 commands that acted, out of 239 actionable.

Notes:

- **Q4 costs no accuracy that shows.** Q4_K_M against Q8_0 on the same prompts: 0.921 against 0.925 (theioma) and
  0.874 against 0.879 (tez220), strict.
- **Tez's generic prompt loses a few points on `tez220`.** On Tez's own set, strict accuracy with the generic
  question-first prompt is 0.874–0.879. The speed study's 0.909 came from the voice-specific prompt of
  `run_voice_fast.py` ("Transcript: ..."). Accepting then/alt, it is 0.980.
- **Harmful counts include wrong final decisions.** They follow `stream_policy.py`, so they count final decisions
  that are wrong as well as wrong early actions. On the Theioma set, 12 of Q4's 16 are early actions. Most come from
  overlapping tools in the 23-tool schema:
  - `system.open_app` against `system.open_url`;
  - `media.key` against `system.volume_set`;
  - "play X" opening Spotify's ancestor before "on youtube" arrives;
  - a single app name ("chrome is frozen ...") opening the app at word 1.
- **Retuning the threshold does not help.** At temperature 1, a sweep of τ from 0.9 to 0.999 (`--rescore`) moves
  Q4's harmful count only from 16 to 11. Waiting for 2 stable words brings it to 11–13, and the first action p50
  rises to 171–193 ms.
- **Q8's harmful count is scored at its default temperature.** Q8 is read at Tez's default temperature for this
  model (2.66 for 24 options, `tez.temperature`); Q4 is not in the table, so it is read at T = 1. Retempered to the
  same scale, the two land in the same place: Q4 goes 16 -> 10 harmful at T = 2.66, and Q8 goes 7 -> 14 at T = 1.
- **The cut-4B probe is the fastest but the least accurate.** It decides in 20–21 ms p50, at 0.82–0.88 held-out. Each
  fold is fitted with `tez fit` on the other fold's 456–704 prefix-labelled rows; the fit's own held-out accuracy is
  0.905–0.939 (`results/voice/probe_fits/`). It is also slow to act, with a first action at 170–500 ms, because the
  prefix labels teach it to wait for the decisive word.
- **Parity.** The stream agrees with whole-prompt reads of the same model within total variation 0.011–0.018 (mean),
  and the argmax differs on about 1 in 100 updates (`manifest.parity`). The residual is batch-shape numerics of
  llama.cpp's CUDA kernels, not a bug: on the stub, the stream is exact to 1e-6. The Phase 2 gate of < 0.01 is not
  met.
- **The dispatch costs nothing measurable.** The no-op native executor (a ctypes Win32 call) takes 0.00–0.01 ms p95.
- **Folding commits does not help latency.** Folding the commit into the decision's decode leaves the decision as it
  was and saves only the separate ~6 ms commit decode; `gap` stays the default.

### (c) VRAM

Each component was measured on its own, with `gpu_lock.py snapshot` after loading:

| component | VRAM |
|---|---:|
| 12B Q4_K_M in process | 8.05 GB |
| 12B Q8_0 in process | 13.09 GB |
| 12B Q4_K_M in a llama-server with :8797's flags (`-c 4096 -np 2 --swa-full`) | 8.72 GB |
| cut 4B | 4.02 GB |
| Nemotron recogniser | 1.12 GB |
| Windows desktop apps (dwm, Code, Chrome, TradingView, explorer) | 1.26 GB |

- **Stack ASR + 12B Q4_K_M in process + desktop:** 10.4 GB, within the 14.5 GB gate (the sum of the measured
  parts; the co-loaded check is in the not-yet-run last job).
- **A second 12B copy does not fit.** An in-process Q4 next to Theioma's shared :8797 server would be
  8.05 + 8.72 + 1.12 + 1.26 = 19.2 GB.
- **Q8 in process** (13.09 + 1.12 + 1.26 = 15.5 GB) exceeds the gate on its own.

## 4. End to end

Composed replay (`--asr replay:ROWS`):

- The live run with the recogniser and the 12B loaded together was killed by `run_heavy`'s watchdog: free RAM fell to
  3.6 GB while Theioma's day jobs ran.
- So each recorded Nemotron 80 ms emission from job 1 was fed, in real time, at its audio time plus its recorded step
  compute, into the live 12B Q4_K_M session. This is exactly when a one-process loop would have produced it.
- Dispatch is measured, so the projected word end -> action is word end -> decision plus 0.00 ms.
- Both runs were [loaded]: system CPU mean 46 % (theioma) and 36 % (tez220), and the GPU rose to 82–84 °C (no
  slowdown recorded).

| set | WER | decide p50 (p95) | decisive word end -> decision p50 (p95) | R0 first action p50 (p95), ≤ 50 ms, before end | intent strict / then-alt | harmful | out-of-scope false |
|---|---:|---:|---:|---:|---:|---:|---:|
| theioma [loaded] | 0.080 | 33.5 (58.2) | **269 (585)** | **173 (775)**, 36 %, 30 % | 0.916 / 0.958 | 32 / 239 | 6 / 30 |
| tez220 [loaded] | 0.045 | 33.0 (46.2) | 276 (526) | 262 (792), 8 %, 3 % | 0.874 / 0.980 | 2 / 198 | 5 / 22 |

The time from the decisive word to its decision breaks down as the recogniser's emission (~210 ms), then its step
(~5 ms), then the decision (~33 ms). The emission is almost all of it. Where the decision can act on a prefix (long
app names on the Theioma set), 30 % of instant actions fire before the word ends. Where the decisive word is short
("up", "pause", "notes": `tez220`), nothing can come before Nemotron emits it. Recognition errors on partial words
double the Theioma set's harmful actions (16 -> 32 utterances). Of the 34 wrong actions, 29 were early or ancestor
actions (reversible opens and media keys: `system.open_app` 19, `system.open_url` 6, `media.key` 3) and 5 were wrong
final decisions.

**Not run yet (the last job).** The job queued behind Theioma's day jobs (seventh in the queue at 11:47), so it
was withdrawn before it started, as the shared-machine rules ask. It holds four arms, runnable in one ~45-minute slot
on a quiet machine with `bash experiments/run_voice_phase1.sh e2e-live`:
- faster-whisper re-decode as the partial source, composed end to end with the live 12B;
- the whisper ASR arm again, paced in real time on a cool GPU (the job 1 arm was [throttled]);
- the Parakeet realtime EOU 120M arm;
- the recogniser and the 12B co-loaded for 40 utterances, which measures the stack's VRAM and RAM directly.

The first of these decides whether re-decoding beats Nemotron's emission lag end to end.

## 5. Gates

Phase 1 gates of `theioma-voice-design.md` §4e, and the Phase 2 replay gates for reference:

| gate | target | measured | verdict |
|---|---|---|---|
| ASR step | ≤ 8 ms p50 | Nemotron 80 ms: 4.68 ms p50 (5.79 p95) per chunk | **pass** |
| emission lag known (go/no-go) | mean ≤ 20 ms for a 50 ms p50 | Nemotron 80 ms: decisive word complete +216 ms mean (p50 208), prefix +128 ms; 186 ms beyond the chunk | **known: no-go** |
| decision, fast model | ≤ 10 ms p50 | cut 4B probe 20.4–21.4 ms p50 | **fail** |
| decision, 12B Q4 | ≤ 20 ms p50 | 28.7–29.4 ms p50 in process; 36.9–37.3 over HTTP | **fail** |
| intent at Q4 | ≥ 0.90 on Tez's set | tez220 0.874 strict (0.980 then/alt); theioma 0.921 | **fail** on Tez's set (strict), pass on Theioma's |
| VRAM of the stack | ≤ 14.5 GB | 10.4 GB (ASR + 12B Q4 in process + desktop, summed from separate runs; the co-loaded run is in the last job) | **pass** (as the only 12B; a second copy next to :8797 does not fit) |
| Phase 2: replay L, R0/R1 | p50 ≤ 50, p95 ≤ 100 ms | R0 first action 173 ms p50, 775 p95 (composed, [loaded]); 33.7 ms p50 with a perfect recogniser | **fail** (the recogniser's lag) |
| Phase 2: harmful | ≤ 1/198 on Tez's set, ≤ 1 % on Theioma's | tez220 1/198 (oracle), 2/198 (e2e); theioma 6.7 % (oracle), 13 % (e2e) | pass on Tez's (oracle), **fail** on Theioma's |
| Phase 2: out-of-scope false actions | ≤ 2/22 | tez220 5/22 (Q4), 4/22 (Q8); theioma 6/30 | **fail** |
| Phase 2: in-process vs server parity | TV < 0.01 | 0.011–0.018 mean, 1 argmax in ~100 | **fail** (narrowly) |

## 6. What would fix each failure, and what Phase 2 needs

**The recogniser's emission lag, about 210 ms.** It is the whole gap between 33 ms and 173 ms. In order of cost:

1. **Act on prefixes, which already works.** `session.update` reads partial words, and 30 % of instant actions fire
   before their word ends. More aggressive prefix decisions trade against harmful actions.
2. **Re-decode (whisper) as the partial source.** In job 1 it completed words at the word's end (−11 ms mean), but
   with higher WER, 43 ms steps that grow with the utterance, and flicker. The composed end-to-end test is the
   not-yet-run last job.
3. **FastEmit fine-tuning of Nemotron on command audio.** NeMo's `fastemit_lambda` is published to cut the
   transducer's emission delay by ~200 ms. This needs NeMo training (Linux or WSL) and a few hours of command audio.
4. **CTC keyword spotting** on the streaming encoder, for a fixed vocabulary (app names, page names).

**The decision at 29 ms instead of ≤ 20 ms.** Each word decodes the new word plus the answer tail (about ten
tokens) through 48 layers. The levers:

- a shorter answer tail (a template change, which re-fingerprints fits);
- CUDA graphs in llama.cpp;
- the 4B, if its accuracy can be brought up: 12B teacher labels on many more prefixes, or the state-only layout with
  several probes.

The decision is not the binding constraint while the recogniser lags by 210 ms.

**Harmful actions on the Theioma set:**

- **Schema design first.** Merge or separate the overlapping tools (open app / open URL, media key / volume level,
  web search / answer). Fix the refinement graph so that "play X ... on youtube" does not open Spotify's ancestor.
- **The policy second.** Stability 2 for opens named by a single word, or `Policy(retract=True)` with undo on
  disagreement.
- **The 12B verifier after the fact.** The design's plan catches the rest.

**Accuracy on Tez's set.** Use the voice-specific prompt wording (a transcript line) as a Tez layout, or fit the
letters temperature and a probe blend with `tez fit` on labelled voice rows.

**Where the 12B lives.** This is the coordinator's question, answered by the numbers:

- An in-process copy in `tez-voice` saves 7.5 ms per word over HTTP (29.4 against 36.9 ms p50). It costs 8.05 GB of
  VRAM, and it cannot sit next to Theioma's shared :8797 Q4 server (19.2 GB).
- Two workable layouts:
  - **(A) `tez-voice` reads :8797 over HTTP.** `tez.stream` does this today through `LlamaCppBackend`: a whole prompt
    per word, the server's prompt cache reusing the prefix. It costs +7.5 ms per word and no extra VRAM. :8797 must
    keep `--swa-full` and one slot free for voice, and long generations on it must pause while the user speaks.
  - **(B) `tez-voice` hosts the only 12B in process**, and Theioma's converse route reaches it through another door.
    `tez serve` only decides and does not generate, so (B) needs a generation endpoint in `tez-voice`.
- Recommendation: (A) for Phase 2, since 7.5 ms is small beside the recogniser's 210 ms. Revisit (B) if the lag is
  fixed.

**What Phase 2 (`tez-voice`) needs:**

- **Capture.** WASAPI shared-mode capture in 10 ms packets, QPC-stamped, and Silero VAD on the CPU.
- **Recogniser.** The Nemotron recogniser runs through NeMo-Speech.cpp's C ABI in a worker process of its own: its
  `ggml`/`llama` DLLs share names with llama.cpp b11100's and cannot load into the same process. Alternatively,
  build llama.cpp and NeMo-Speech.cpp from one ggml.
- **The session.** `tez.stream` sessions (one per utterance, `update(text)` per recogniser step), with the schema's
  `actions:` block as the policy.
- **Native executors,** each obeying Theioma's voice audit log and kill switch:
  - `ShellExecuteExW`;
  - `SendInput` Unicode, with a foreground check before typing;
  - media keys;
  - Theioma's `POST /api/voice/tools/run`.
- **Feedback.** Earcons and cached Kokoro clips.
- **Events.** On :8818: transcript, decision, action, undo, end of utterance. `none` at the end goes to
  `/api/voice/converse`, and R3 opens an `ApprovalRequest`.
- **A live-microphone Phase 0 set.** Both sets here are synthetic (SAPI and Kokoro). At least 50 recorded commands
  are needed for the live gate.
- **The co-loaded stack.** RAM headroom must be measured on a quiet machine: the live co-loaded run was killed by the
  RAM watchdog under Theioma's day load.

**What the Theioma integration needs from its coordinator:**

- **The hosting choice.** (A) or (B) above. For (A), a slot reserved for voice on :8797, and a pause of long
  generations while voice mode is armed.
- **The generated schema.** Build it from `apps/web/src/lib/voice-tools/registry.ts`, with a class (R0–R3) and a
  refinement parent for every tool and a parity test, and resolve the overlapping tools named above.
- **Microphone ownership.** Arming `tez-voice` must disable the voice sidecar's continuous loop.
- **The :8818 event contract** that the daemon or widget consumes.
- **Executors.** Either the executors honour Theioma's audit log and kill switch, or the tool-run route is callable
  with `x-theioma-secret`.
