# Text-to-speech for the Theioma voice loop: candidates, tests and gates

Plan, 2026-09-25. It extends `docs/plans/theioma-voice-design.md` (§1.3, §1.7, §3.3, §4b, §4d-4e). The harness is
`experiments/tts_bench.py`, with its utterances in `data/voice/tts_utterances.json`, its environment in
`experiments/tts_requirements.txt` (`.venv-tts`), and its results in `results/tts/`.

Theioma (`AILABS/Jarvis`) was only read, never changed. The only GPU work was smoke tests under the GPU lock, next
to the production llama-server. **No number from this machine in this document is a timing yet.** §6 lists what the
smoke tests showed, labelled as smoke.

## 0. Summary

- **Theioma speaks through one path.** The converse route cuts a reply into clauses, the browser posts each clause
  to `/api/voice/speak`, and that calls the Kokoro sidecar's whole-clause `POST /synthesize`. The browser then fetches
  the WAV and plays it in an `<audio>` element.
  - The next clause is requested only after the current one **has finished playing**, so every clause boundary adds a
    full synthesis round trip of silence.
  - The sidecar runs the **fp16** Kokoro ONNX export (its `kokoro-v1_0.onnx` has the same SHA-256 as
    `kokoro-v1.0.fp16.onnx`), with onnxruntime's default cuDNN settings. Theioma measured it at 1,175 ms to first
    audio.
  - On the CUDA provider that export needs **241** host-device copies against **17** for the fp32 export; 241 is the
    count Theioma reported (§1).
  - There is no barge-in, no echo cancellation and no cached speech. No TTS work is in progress there (§1).
- **The candidate field is narrow once the constraints apply.** The constraints are ≤ ~0.6 GB of VRAM next to a 12B
  model and a streaming ASR, Windows, commercial use, and ≤ 150 ms to first audio.
  - **Kokoro-82M** is the reference and the likely winner, on GPU or CPU.
  - **Kyutai Pocket TTS** and **Supertonic 3** are the CPU challengers, which would leave the GPU entirely to the 12B
    and the ASR.
  - **Sopro V2 Turbo** is the one expressive GPU model that plausibly fits. It needs a reference voice clip and its
    own environment, so it goes to round 2.
  - Everything else fails on licence, memory or latency (§2).
- **What gets tested** (§3):
  - time to first audio for new text, repeated text, the first call after load and the first call after idle;
  - streaming stalls and real-time factor;
  - VRAM and spill, and CPU;
  - intelligibility (WER of Whisper large-v3 on the output) and naturalness (UTMOS22-strong);
  - barge-in stop latency;
  - playback latency of earcons and cached clips.
- **Gates** are tied to the voice plan's phases (§4): TTS never delays an action; earcons and clips ≤ 20 ms; live
  speech p50 ≤ 150 ms / p95 ≤ 300 ms; plain-text WER ≤ 3 %; ≤ 0.6 GB of VRAM in-process or ≤ 1.0 GB as a sidecar,
  with no spill.
- **Integration** (§5): earcons, clips and the single audio output stream live in `tez-voice`. Live synthesis
  stays a sidecar, the existing `:8796` slot rebuilt around the winner, that streams per clause, synthesises ahead of
  playback and can be cancelled.

## 1. What Theioma has today

Read-only audit of `J\` = `AILABS\Jarvis`: HEAD `33166bc51` on `jpex/consolidate-docs-corpus`, 2026-09-25. The
voice-related `git log --since=2026-09-10` has 38 commits. Only one touches TTS (`b8b927378`, below). No branch or
plan file carries TTS work.

| Piece | State | Where |
|---|---|---|
| Kokoro sidecar, `:8796`, in `SERVICES` | `kokoro-onnx==0.6.1`, `onnxruntime-gpu[cuda,cudnn]==1.29.0` (a CUDA 13 build) with the CUDA 13 `nvidia-*` wheels pinned, Python 3.13 venv. It loads `~/.theioma/kokoro/kokoro-v1_0.onnx`, **the fp16 export** (SHA-256 `c1610a85…`), and `voices.bin` (= `voices-v1.0.bin`). The session asks for `CUDAExecutionProvider` with **no provider options**, so onnxruntime's default cuDNN algorithm search (EXHAUSTIVE) applies. The default voice is `af_bella`; the widget pins `am_michael`. | `packages/kokoro-tts/src/server.py:126-186, 188-250`, `requirements.txt` |
| Sidecar API | `POST /synthesize` renders a whole clause, returns an `audioUrl` and keeps the WAV in a 64 MB LRU. `WS /stream` synthesises each `speak` message whole; **nothing in the app opens it**. There is no cancel. An `edge_tts` fallback is coded but not installed. | same, `:384-414, 420-490` |
| Measured by Theioma (2026-09-06, RTX 5080 Laptop, CUDA EP, warm) | `WS /stream` first audio **1,175 ms** (sidecar clock). One word: **683 ms**. `/synthesize` for 2.0 / 5.2 / 8.3 s of audio: 1,376 / 1,743 / 2,039 ms. First call after a restart: 3,255 ms. Load: 3.2 s. onnxruntime reported **241 Memcpy nodes**. Theioma's write-up blames espeak phonemisation and the copies. | `docs/design/evidence/2026-09-06-wave4-kokoro-gpu/README.md` |
| App path | The converse route's clause splitter emits `speak_clause` over SSE. The widget and modal each `POST /api/voice/speak` (Kokoro first, 10 s timeout, Piper fallback), then play `audioUrl` through `/api/voice/audio`. **Clause N+1 is requested only after clause N has finished playing.** | `apps/web/src/app/api/voice/converse/route.ts:1556-1619`; `api/voice/speak/route.ts:43-69`; `(widget)/voice-widget/page.tsx:825-871`; `components/voice-chat-modal.tsx:1161-1230` |
| Piper fallback | Runs in the voice sidecar: `piper-tts 1.4.2`, `en_GB-alan-medium`, onnxruntime 1.25 on the CPU, played on the host speakers. Since `b8b927378` (2026-09-25) `/api/voice/speak` returns 503 unless that sidecar is already running. | `packages/voice/sidecar/theioma_voice_sidecar.py:124, 390-408`; `lib/voice/sidecar-fallback.ts` |
| Chatterbox-Turbo package, `:8814` | Manifest and server only: no venv, not in `SERVICES`. Its weights are in the HF cache. | `packages/chatterbox-tts/` |
| Moshi | A stub (`/health` 503). | `packages/moshi-sidecar` |
| Barge-in, AEC | None. Every `voice-aec` adapter is a stub. The widget mutes the mic half-duplex while it plays. `/api/voice/speak/cancel` reaches only Piper. | `packages/voice-aec/src/barge.ts:24-31`; `api/voice/speak/cancel/route.ts` |
| Metric | `tts_ttft_ms` is the whole `/synthesize` request, not first audio. | `api/voice/speak/route.ts:95-103` |
| Stale claims | "50ms TTFT" (`api/voice/speak/route.ts:4-6`); "~50-200ms" (`voice-chat-modal.tsx:1165`). | |
| Related, not TTS | The Tez adoption work of 2026-09-24/25 (`docs/app/tez-adoption.md`, `packages/voice-fastpath`, V0-V2) is about voice-to-action. It uses the Kokoro sidecar only to synthesise its replay fixtures. The one uncommitted voice file, `lib/voice/speculative-eou.ts`, changes a model id. | |

What this points to:

1. **The 241 host copies come from the fp16 export. This is measured, not a hypothesis.** onnxruntime 1.30 inserts
   **17** Memcpy nodes for the CUDA provider with `kokoro-v1.0.onnx` (fp32), **241** with `kokoro-v1.0.fp16.onnx`
   (Theioma's count exactly) and **547** with the int8 export. The cuDNN search mode does not change these counts
   (`results/tts/kokoro-onnx_memcpy-probe-20260925.json`; a property of the graph, not a timing).
   - In the smoke tests the fp16 file was also 3-5x slower than fp32 to first audio on the same GPU (§6).
   - The price is memory: the fp32 file is 325 MB against 177 MB. After load the smoke processes held ~715 MB of
     dedicated VRAM with fp32 against ~460 MB with fp16, CUDA context included. That is ~0.25 GB more, and still
     inside the 1.0 GB sidecar budget of §4.
2. **The EXHAUSTIVE cuDNN search** is a hypothesis: it could re-run for every new input length in a dynamic-shape
   graph. Piper's own CUDA path sets `HEURISTIC` (`piper/voice.py`). The suite runs both; the smoke tests did not
   separate them.
3. **No synthesis ahead of playback**, and a browser in the middle. This is structural, not a model problem.

Phonemisation is not the problem. espeak-ng through `phonemizer` measured 0.4-0.9 ms per clause in the smoke runs
(`g2p_ms`).

## 2. Candidates, September 2026

The research was checked on the web on 2026-09-25; sources are in §8. [V] marks the vendor's claim.

[I] marks tts-bench's independent measurement. Its rig is Windows 11, a Ryzen 9 9950X3D and a desktop RTX 5090 running
upstream PyTorch, and it reports warm and cold time to first audio (TTFA). For non-streaming engines that is the time
to the whole short prompt. It also reports UTMOS and Whisper-large-v3 WER.

| Model | Licence (commercial?) | Size / memory | Streaming; TTFA as published | Windows / CUDA | Quality evidence | Languages | Runs as |
|---|---|---|---|---|---|---|---|
| **Kokoro-82M** v1.0 (2025-01-27) | Apache-2.0 weights; code Apache/MIT (yes). espeak-ng fallback is GPL: check if you redistribute. | 82M; 327 MB .pth; ONNX 325 / 177 (fp16) / 92 MB (int8); [I] 925 MB VRAM peak | no text-in; chunks per split. [I] CUDA: 67 ms warm / 895 ms cold; CPU: 532 / 609 ms. [V] Kokoro-FastAPI ~300 ms | yes: `kokoro` (PyTorch, Py < 3.13), `kokoro-onnx` (Py ≤ 3.13), sherpa-onnx | [I] UTMOS 4.18, WER 5.4 %; TTS Arena V2 1477, rank 30 (a retired entry) | 8 (en-US/GB, es, fr, it, pt-br, hi, ja, zh) | PyTorch, ONNX, sherpa-onnx, Kokoro-FastAPI |
| **Kyutai Pocket TTS** 3.3.0 (2026-09-24) | code MIT, weights CC-BY-4.0 (yes, with attribution). The cloning repo is gated; `pocket-tts-without-voice-cloning` is not. | 100M; 236 MB; [I] 1.95 GB RAM, no GPU needed | streams audio out, with a `stop` event. [I] Win CPU 119 ms warm / 142 ms cold, 4.4x real time. [V] ~200 ms | yes, pure Python on torch; CPU by default | [I] UTMOS 3.97, WER 4.5 % | en, fr, de, pt, it, es, nl | PyTorch (sherpa-onnx export exists) |
| **Supertonic 3** (2026-05) | OpenRAIL-M weights (yes, with use restrictions); MIT code | ~99M; ONNX int8; [I] 570 MB RAM | whole utterance. [I] Win CPU 741 ms warm, 9.9x | yes: ONNX Runtime, sherpa-onnx, `supertonic` SDK | [I] UTMOS 4.08, WER 5.4 % | 31 | ONNX, sherpa-onnx |
| **Kitten TTS** 0.8 (nano 15M, micro 40M, mini 80M; 2026-02) | Apache-2.0 (yes). PyPI `kittentts` 0.1.3 is a third-party upload; use the GitHub wheel or sherpa. | 25-80 MB | per chunk; none published for 0.8 | yes (ONNX) | [I] nano 0.1 only: UTMOS 3.67, WER 9.3 % | en (8 voices) | ONNX, sherpa-onnx |
| **Sopro V2 Turbo** (2026-08) | Apache-2.0 (yes) | 120M; [I] 0.86-1.09 GB VRAM | streams. [I] CUDA streaming 109 ms warm; default mode 400 ms | torch + torchaudio ≤ 2.11, so not the torch 2.14 venv; needs a reference voice clip | [I] UTMOS 4.10, WER 5.9 % | en, pt-PT, fr, de | PyTorch (`sopro` 2.2.0) |
| Piper 1.8.0 (2026-09-04) | **GPL-3.0** (`piper1-gpl`; the MIT repo was archived 2025-10-06); per-voice licences (lessac data is non-commercial) | 63 MB/voice; [I] 470 MB RAM | per sentence. [I] Win CPU 107 ms warm, 59x | yes (ONNX) | [I] UTMOS 3.88, WER 5.5 % | 35 | ONNX; control and fallback only |
| Chatterbox-Turbo (Dec 2025) | MIT, **watermark always on** | 350M; [I] 3.0 GB VRAM | [V] 75 ms; [I] 1.62 s warm (stock, not streaming); NVIDIA ACE GGML plugin streams, ≥ 2 GB | torch pin 2.6 (needs an override for sm_120) | [I] UTMOS 4.27, WER 7.4 %; Arena #29 | en | PyTorch; GGML (ACE) |
| StyleTTS 2 | MIT; inactive since 2024-03; disclosure terms | 148M; [I] 1.49 GB | no; [I] 265 ms warm | source only | [I] UTMOS 4.26, **WER 15.8 %** | en | PyTorch |
| XTTS-v2 | **CPML, non-commercial** | 750M; [I] 2.1 GB | [V] < 200 ms; [I] 1.87 s warm | `coqui-tts` fork | [I] UTMOS 3.94 | 17 | PyTorch |
| F5-TTS | **CC-BY-NC weights** | 336M | [I] 845 ms | yes | [I] WER 19.5 % (cloning) | en, zh | PyTorch |
| Orpheus 3B | Apache (gated; Llama base) | 2.36 GB as Q4 GGUF | [I] 3090: 361 ms, ~1x real time | vLLM is not on Windows; llama.cpp + SNAC | [I] UTMOS 4.00 | en | GGUF, vLLM |
| Sesame CSM-1B | Apache, gated | ~1.1B; [I] 3.5 GB | no; [I] 12.45 s | needs triton-windows | [I] UTMOS 4.15 | en | PyTorch |
| Dia 1.6B / Dia2 | Apache | [I] 6.3 GB / Dia2 1-2B | Dia2 streams text in; [I] Dia 22.8 s | torch nightly on RTX 50 | [I] Dia UTMOS 2.39 | en | PyTorch |
| Zonos v0.1 | Apache | 1.6B; 6 GB+ | [I] 10.4 s | no native Windows | - | 5 | PyTorch |
| Kyutai TTS 1.6B | CC-BY-4.0 | ~4 GB of weights | true text-in; [V] 220 ms | Windows unverified | - | en, fr | PyTorch, Rust |
| VibeVoice-Realtime 0.5B | MIT but "research only"; **audible disclaimer and watermark** | [I] 2.6 GB | [V] ~300 ms; [I] 3.77 s | container | [I] UTMOS 4.04 | en | PyTorch |
| Qwen3-TTS 0.6B / 1.7B | Apache | [I] 2.6-4.9 GB | [V] 97 ms (vLLM); streaming only in forks; [I] 1.60 s | flash-attn recommended | [V] strong WER | 10 | PyTorch, vLLM |
| NeuTTS Air / Nano | Apache / NeuTTS Open License (gated) | 748M / 229M; GGUF | [I] GGUF Q4: 258-471 ms | llama-cpp-python | [I] cloning WER 57-68 % | en | GGUF |
| Fish/OpenAudio S1-mini, S2; Spark-TTS; Llasa; Higgs v3; Voxtral-4B-TTS; IndexTTS-2.5; CosyVoice 3 | non-commercial or restricted licences, or Linux-only serving stacks, or ≥ 4 GB | | | | | | |
| Watch list: Inflect-Nano v2 (4M, Apache), Vaniq-Edge (8.9M, MIT), Chatterbox-Nano (110M, MIT, watermark) | single voice, single author, new | ≤ 0.3 GB | [I] Inflect 131 ms CUDA; Vaniq 250 ms CPU | ONNX / .pth | [I] UTMOS 4.35 / 4.23, WER 5.9 / 11.5 % | en | |

**Shortlist for round 1, and why:**

1. **Kokoro-82M** is the reference. It has the best quality per millisecond on independent data, 8 languages
   including pt-br and es, is Apache-2.0, and is what Theioma already ships and uses for the Phase 0 fixtures. Its
   open question is **the runtime**, not the model. Round 1 runs it five ways:
   - ONNX fp32, fp16 and int8;
   - ONNX fp16 as Theioma configures it;
   - PyTorch on CUDA;
   - PyTorch and sherpa-onnx int8 on the CPU.
2. **Pocket TTS** (CPU): about 120 ms warm with streaming and a native stop event, and no VRAM. If it passes the
   latency and quality gates on this laptop's CPU while the ASR runs, the GPU budget question disappears.
3. **Supertonic 3** (CPU, via sherpa-onnx): 31 languages and UTMOS 4.08 at no VRAM cost. Its whole-utterance first
   audio (~0.6-0.75 s) is likely too slow for live replies, but it may serve pre-rendering and long replies.
4. **Kitten 0.8** (nano, mini; CPU via sherpa-onnx): the smallest footprint. Quality is the question.
5. **Piper** as a control only (GPL, and the lessac voice licence).
6. **Round 2, if nothing in round 1 passes the quality gate:** Sopro V2 Turbo (its own venv with torch 2.11 +
   torchaudio 2.11, plus a licensed reference clip), then Chatterbox-Turbo through the GGML path.

## 3. The test

### 3.1 Matrix

| Axis | Values |
|---|---|
| Engine and runtime | the `suite` lists in `experiments/tts_bench.py`: `control` (a sine engine that measures the harness's and the OS's own noise), `gpu` (Kokoro ONNX fp32 and fp16 with HEURISTIC, fp16 with EXHAUSTIVE as in Theioma, Kokoro PyTorch), `cpu` (Kokoro ONNX int8, Kokoro PyTorch, sherpa Kokoro int8, Supertonic 3, Kitten mini and nano, Pocket fp32 and int8, Piper) |
| Chunking | `clause` for every engine by default (what Theioma's converse route emits). The winner is also run with `first-clause` and `sentence`. |
| GPU condition | **A**, idle: nothing decoding, and the production Q8 12B **stopped** (it leaves ~1 GB free, and the smoke tests spilled). **B**, co-resident: a Gemma 4 12B **Q4_K_M** llama-server loaded (the design's shared-12B target, ~9-9.5 GB) and idle. **C**, contended: B plus `--contend-url`, the 12B decoding throughout. **D**, full stack, once Phase 1 exists: the ASR and the fast model in `tez-voice`. |
| Voice | Kokoro `af_heart` for comparisons; Theioma's `af_bella` and `am_michael` for the winner |

### 3.2 Metrics (per call, monotonic clock `time.perf_counter_ns`)

| Metric | Definition | Reported |
|---|---|---|
| Time to first audio | from the call to the first audio chunk the engine returns; includes text chunking and G2P, excludes playback | p50 / p95 per phase: **cold** (first call after load, plus 5 fresh processes for its spread, with load time); **first** (each utterance once: new text and new lengths, as live replies are); **repeat** (reps 2-10, interleaved); **idle** (after 3 s and 15 s of silence, with the SM clock and P-state at the call) |
| Real-time factor | synthesis wall time ÷ audio duration | p50 / p95 on the long tier |
| Streaming chunk latency | gaps between chunks, and **stall_ms**: how long a 1x player that starts at the first chunk would have waited | max gap and stall p95 per tier; the stall must be 0 |
| VRAM | NVML device memory sampled every 50 ms: delta after load and peak delta. This process's dedicated and **shared** memory from the Windows GPU counters. Spill and eviction make the run invalid (below). `torch.cuda.max_memory_*` for PyTorch engines. | MB, against the budget in §4 |
| CPU | process CPU time ÷ wall time per call (cores); system CPU % per phase | p50 |
| Intelligibility | WER of faster-whisper **large-v3** (beam 5, CPU int8) on each `first`-phase utterance against its `spoken` form, with a small number normaliser | **plain** tiers (clip, short, medium, long, barge) and **stress** (numbers, times, acronyms) apart |
| Naturalness proxy | **UTMOS22-strong** (SpeechMOS v1.2.0 via `torch.hub`, 16 kHz), chosen because tts-bench and most TTS papers report it, so the numbers compare. It is a proxy: the top two also get a blind A/B listening pass of 10 utterances by the owner. | mean per tier |
| Barge-in | the long passage played at 1x with a 500 ms lookahead, cancelled 400 ms after its first audio. **stop_ms**: cancel → the engine's call has returned (the compute still in flight). **next_ttfa_ms**: cancel → first audio of the next utterance. | p50 / p95 over 5 trials |
| Playback tier | `playback`: one WASAPI shared stream, 10 ms blocks; trigger → callback that writes the clip, plus the callback's reported DAC time. `--mute` gives the same path in silence. | p50 / p95. A loopback-microphone check comes later. |
| Contention | with `--contend-url`, the 12B's tokens/s before and during | the % drop |

**Validity.** A result is `timing_valid` only if all of these hold:
- it is not `--smoke`;
- the system CPU was ≤ 25 % and GPU utilisation ≤ 10 % before the load;
- this process's shared GPU memory stayed ≤ 160 MB. Any CUDA process here shows 76-82 MB of shared memory with no
  pressure at all.
- **no other process was evicted**: this process's dedicated VRAM minus the growth of the device total is
  ≤ 200 MB.

The eviction check exists because under WDDM an oversubscribed card does not fail; it quietly pushes other
processes' memory to system RAM. In the first fp32 smoke run the TTS process held 1,241 MB while the device total
rose 258 MB, next to the production 12B.

Other GPU holders over 1 GB are recorded. Every result carries package versions, model SHA-256s, the GPU clocks and
power, the power scheme, whether the laptop is on AC power, the git commit, the utterance-file hash, and nvidia-smi
plus `gpu_lock.py snapshot` before and after.

### 3.3 Utterances (`data/voice/tts_utterances.json`, 41 texts + 4 earcons)

| Tier | n | Purpose |
|---|---|---|
| earcon | 4 | tones for acknowledge, listen, undo and error: playback path only, never synthesised |
| clip | 10 | fixed confirmations ("Opening Spotify.", "Done.", "Sorry, I didn't catch that."). They are pre-rendered in production; synthesising them live measures the fallback and the build cost. |
| short | 8 | a confirmation with a dynamic slot ("Opening Blender.", "Playing lofi beats on YouTube.") |
| medium | 8 | one status sentence (8-15 words) |
| long | 5 | a 2-4 sentence conversational reply (40-70 words) |
| stress | 6 | numbers, times, money, paths, acronyms, product names ("3:45 PM", "$1,250.40", "origin/main", "Theioma") |
| barge | 1 | a ~30 s passage for interruption |
| i18n | 2 | pt and es, only for engines that support them |

## 4. Gates, by phase of the voice plan

| Phase | TTS gate | Measured by |
|---|---|---|
| **0: harness and baseline** | The harness runs every shortlisted engine end to end (met for all but Sopro; §6). The Kokoro voices that synthesise the Phase 0 Theioma command set have plain WER ≤ 3 % under large-v3, so fixture errors do not leak into ASR measurements. | `run --score` on the fixture voices |
| **1: component spikes (GPU)** | A live-synthesis (T2) engine proceeds only if, in condition A (idle, 12B stopped): <ul><li>`first` TTFA p50 ≤ 150 ms and p95 ≤ 300 ms (CPU engines p50 ≤ 250 ms);</li><li>long-tier RTF ≤ 0.2 and stall p95 = 0;</li><li>idle(15 s) TTFA p95 ≤ 2x `first` p95;</li><li>plain WER ≤ 3 % and UTMOS ≥ Kokoro fp32 − 0.15;</li><li>barge stop_ms p95 ≤ 150 ms and next_ttfa p95 ≤ 300 ms.</li></ul> **Fit:** peak VRAM delta ≤ 0.6 GB in-process or ≤ 1.0 GB as a sidecar (own CUDA context), with no spill or eviction in condition B. **Contention (C):** TTFA p95 ≤ 400 ms and the 12B's tokens/s drop ≤ 15 %. | `suite --name control,gpu,cpu` per condition |
| **2: `tez.stream` + `tez.voice`** | Earcons and clips: trigger → DAC p95 ≤ 20 ms on wired output. Every confirmation in the schema's `say:` fields is pre-rendered, including slot values from the lexicons. **TTS never delays an action:** in the replay harness, action dispatch times with TTS on and off differ by ≤ 2 ms at p95. | `playback`; the Phase 2 replay harness |
| **3: Theioma integration** | For a converse reply, first audio ≤ 300 ms after its first clause exists. No gap > 50 ms between clauses (synthesis ahead of playback). Speech start → TTS silent ≤ 100 ms with a headset. `tts_ttft_ms` redefined as first audio. | Theioma's voice metrics, with the same definitions as here |
| **4: hardening** | The design's gate: TTS first audio ≤ 200 ms p95 live. Barge-in on speakers only with real echo cancellation. No TTS regression in CI replay. | live runs; CI |

**Decision rule.** Among engines that pass every Phase 1 gate, take the lowest `first` TTFA p95 in condition C.
- Ties go to the one using less VRAM.
- A CPU engine that passes wins over a GPU engine within 50 ms of it: it leaves the GPU to the 12B and the ASR, and
  GPU contention cannot touch it. This needs a condition-D check that it does not slow the ASR's CPU threads.
- If none passes the latency gate, Kokoro on the best runtime becomes the T2 engine with `first-clause` chunking, and
  Phase 4's 200 ms p95 becomes the target.

## 5. How the winner integrates

| Tier | Where | How |
|---|---|---|
| T0 earcons, T1 cached clips | **in-process in `tez-voice`** | PCM in RAM, written into one permanently open WASAPI output stream owned by `tez-voice`. Nothing is synthesised on the hot path. |
| T2 live synthesis | **sidecar**: the `:8796` slot, `packages/kokoro-tts` rebuilt around the winner (or `tez voice tts` for other hosts) | A local WebSocket. Text goes in by clause; PCM comes back per clause. A `cancel` message stops between clauses (or mid-stream for Pocket). The sidecar synthesises at least one clause ahead of playback. Audio returns to `tez-voice`'s output stream, not to a browser `<audio>`, so clips, earcons and speech share one device, and barge-in ducks and stops them in one place. |

Why a sidecar for T2:
- **Isolation.** A TTS crash or a 2 s first call must not stall the ASR loop.
- **The GIL.** PyTorch engines hold Python threads that `tez-voice`'s 10 ms loop cannot afford to share.
- **Restart independence.** It can be restarted on its own.

The cost is a CUDA context (~0.3-0.5 GB) and ~1 ms of local IPC. If Phase 1 shows the context does not fit the VRAM
budget, a GPU winner moves into `tez-voice` on a worker thread (onnxruntime releases the GIL during `run`). A CPU
winner stays a sidecar with 2-4 fixed threads, kept off the ASR's cores.

**Clip cache.** Clips are keyed by (engine, model hash, voice, speed, text) and built when the Tez schema is
generated, from every `say:` template expanded over its lexicon (every app name for "Opening {app}"). A missing key
falls back to T2 and logs the miss.

**Changes Theioma would need.** These are recommendations only; none was made here:
1. In the sidecar, set the CUDA provider options and choose the export by measurement (fp32 or int8, not fp16
   unless round 1 clears it).
2. Request clause N+1 while clause N plays, in the widget and the modal.
3. Make `tts_ttft_ms` measure first audio.
4. Remove the "50 ms" claims.
5. Move playback out of the browser once `tez-voice` owns audio.

## 6. Smoke tests, 2026-09-25 (functional only; not timings)

Every run below is marked `timing_valid: false` in its file. System CPU sat at 45-100 % from other sessions
throughout. The GPU runs were taken under the GPU lock with the production Q8 12B resident (~14.2 GB, leaving
~1.1 GB free), and at least one of them evicted other processes' VRAM. The numbers show that the adapters work and give
rough orders of magnitude, nothing more. Rows are `results/tts/*_smoke*.json`, 1-2 reps each.

| Engine config | Works | first-call TTFA p50 (clip / dynamic) | long RTF | Notes |
|---|---|---|---|---|
| sine (harness control) | yes | 11 ms / 11 ms | - | OS noise alone gave 100-240 ms outliers on this busy machine: the reason for the control run |
| kokoro-onnx fp32, CPU | yes | 2,399 / 3,127 ms | 1.36 | ~15 cores busy on a CPU already at 60 %; `g2p_ms` 0.4-0.9 |
| kokoro-onnx fp32, CUDA, HEURISTIC | yes | 283 / 493 ms | 0.44 | 17 Memcpy nodes. The process held 1,241 MB while the device total rose 258 MB, so ~1 GB of other processes was evicted (§3.2). Repeat calls were slower than first calls. |
| kokoro-onnx fp16, CUDA, HEURISTIC | yes | 1,315 / 1,362 ms | 0.76 | 241 Memcpy nodes; 728 MB dedicated |
| kokoro-onnx fp16, CUDA, EXHAUSTIVE (Theioma's config) | yes | 947 / 1,452 ms | 0.97 | 241 Memcpy nodes |
| kokoro-torch, CPU | yes | 1,272 / 1,440 ms | 0.44 | misaki pip-installs `en_core_web_sm` on first use |
| kokoro-torch, CUDA | yes | 288 / 978 ms | 0.41 | first call after load 8.5 s; `torch.cuda` max allocated 680 MB, reserved 762 MB; 1,016 MB process dedicated at the end |
| sherpa Kokoro int8, CPU (sentence chunks) | yes | 1,795 / 4,434 ms | 1.28 | callback convention: return 1 to continue, 0 to stop |
| sherpa Supertonic 3 int8, CPU | yes | 560 / 765 ms | 0.27 | emits one chunk per call, so it needs clause chunking |
| sherpa Kitten nano 0.8 int8, CPU | yes | 1,382 / 2,088 ms | 0.53 | |
| sherpa Kitten mini 0.8, CPU | yes | 1,448 / 2,971 ms | 0.82 | |
| Pocket TTS, CPU | yes | 108 / 149 ms | 0.41 | streams; stop event honoured (barge stop 7 ms) |
| Piper lessac-medium, CPU | yes | 237 / 215 ms | 0.14 | control |
| `playback` (earcons, muted), WASAPI shared, 10 ms blocks, Realtek speakers | yes | trigger → DAC estimate p50 14.7 / p95 19.1 ms (n = 40) | - | PortAudio reports 22 ms of stream latency, so the 20 ms gate needs the loopback check |

Two things the smoke runs already say, whatever the timing noise:
- **The fp16 Kokoro export is the wrong file for the CUDA provider.** It was 3-5x slower than fp32 to first audio,
  under both cuDNN search modes.
- **Pocket TTS is the CPU candidate to beat.** It gives first audio in about 110-150 ms with the CPU already busy.

Scoring (WER, UTMOS) was exercised on one run; see §7.

## 7. Running it

```
.venv-tts\Scripts\python experiments\tts_bench.py list
.venv-tts\Scripts\python experiments\tts_bench.py suite --name control,gpu,cpu --tag r1-A --print   # one command per engine
.venv-tts\Scripts\python experiments\tts_bench.py suite --name control,gpu,cpu --tag r1-A --lock none  # inside an idle window that holds the lock
.venv-tts\Scripts\python experiments\tts_bench.py run --engine kokoro-onnx --variant fp32 --device cuda --tag r1-C ^
    --reps 10 --cold-runs 5 --idle-gaps 3,15 --barge-trials 5 --contend-url http://127.0.0.1:8095 --lock acquire --score
.venv-tts\Scripts\python experiments\tts_bench.py score results\tts\<run>.json
.venv-tts\Scripts\python experiments\tts_bench.py playback --tag r1 --no-mute
```

Exit codes:
- 0: ok;
- 2: a dependency or model file is missing;
- 3: the GPU lock is held by someone else (`--lock check`);
- 4: the engine failed (partial rows are still written).

Timed round 1 belongs in an idle window. `experiments/run_when_idle.py` (another session's runner) already waits for
low CPU, a free lock, no Ollama model and no stray GPU holders, then restores the production server. A `tts` plan
there would:
1. stop the production server;
2. run `suite --name control,gpu,cpu --tag r1-A --lock none` (condition A);
3. start a Q4_K_M 12B on another port with `speed_servers.ps1 start`, then rerun `suite --name gpu,cpu --tag r1-B`
   (condition B) and again with `--contend-url` (condition C);
4. restore the production server.

That plan is not added here.

## 8. Sources (checked 2026-09-25)

- tts-bench speed and scores (Windows RTX 5090 and CPU rigs): https://5uck1ess.github.io/tts-bench/speed.html ,
  https://5uck1ess.github.io/tts-bench/scores.html , https://github.com/5uck1ess/tts-bench
- TTS Arena V2 leaderboard: https://tts-agi-tts-arena-v2.hf.space/leaderboard
- Kokoro: https://huggingface.co/hexgrad/Kokoro-82M ; kokoro-onnx 0.6.1 and model-files-v1.0:
  https://github.com/thewh1teagle/kokoro-onnx ; https://pypi.org/project/kokoro/ ; Kokoro-FastAPI:
  https://github.com/remsky/Kokoro-FastAPI
- Pocket TTS: https://github.com/kyutai-labs/pocket-tts , https://pypi.org/project/pocket-tts/ ,
  https://huggingface.co/kyutai/pocket-tts-without-voice-cloning
- Supertonic 3: https://huggingface.co/Supertone/supertonic-3 (README, LICENSE: OpenRAIL-M)
- Kitten TTS: https://github.com/KittenML/KittenTTS/releases (0.8.1), https://huggingface.co/KittenML
- Sopro: https://huggingface.co/samuel-vitorino/sopro-v2-turbo , https://pypi.org/project/sopro/ (2.2.0)
- Piper: https://github.com/OHF-Voice/piper1-gpl , https://pypi.org/project/piper-tts/ , https://github.com/rhasspy/piper
  (archived), https://huggingface.co/rhasspy/piper-voices
- sherpa-onnx 1.13.8 and its TTS models: https://github.com/k2-fsa/sherpa-onnx/releases/tag/tts-models ,
  https://k2-fsa.github.io/sherpa/onnx/cuda.html
- Chatterbox: https://github.com/resemble-ai/chatterbox , https://huggingface.co/ResembleAI/chatterbox-turbo , NVIDIA ACE
  plugin: https://docs.nvidia.com/ace-for-games/chatterbox-tts/programming-guide-tts-chatterbox.html
- XTTS-v2 licence: https://huggingface.co/coqui/XTTS-v2 ; F5-TTS: https://huggingface.co/SWivid/F5-TTS ; Orpheus:
  https://huggingface.co/canopylabs/orpheus-3b-0.1-ft ; CSM: https://huggingface.co/sesame/csm-1b ; Dia:
  https://github.com/nari-labs/dia , https://github.com/nari-labs/dia2 ; Zonos: https://github.com/Zyphra/Zonos ; Kyutai TTS:
  https://huggingface.co/kyutai/tts-1.6b-en_fr ; VibeVoice: https://huggingface.co/microsoft/VibeVoice-Realtime-0.5B ;
  Qwen3-TTS: https://github.com/QwenLM/Qwen3-TTS ; NeuTTS: https://github.com/neuphonic/neutts ; Inflect-Nano v2:
  https://huggingface.co/owensong/Inflect-Nano-v2 ; Vaniq-Edge: https://huggingface.co/Abiray/Vaniq-Edge
- onnxruntime 1.30.0 release notes (Windows SM120 build fixes) and issue 26177 (sm_120 PTX on 1.23):
  https://github.com/microsoft/onnxruntime/releases , https://github.com/microsoft/onnxruntime/issues/26177 ; PyTorch
  2.7 Blackwell wheels: https://pytorch.org/blog/pytorch-2-7/
- UTMOS22 via SpeechMOS: https://github.com/tarepan/SpeechMOS ; faster-whisper: https://github.com/SYSTRAN/faster-whisper
