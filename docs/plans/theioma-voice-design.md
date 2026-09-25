# Tez in Theioma: tool routing and a sub-50 ms voice loop

Audit and design, 2026-09-24. Read-only: nothing in Theioma (`AILABS/Jarvis`) or Tez (`AILABS/Tez`) was
modified, no GPU job was started, nothing was committed.

Contents: Summary · 1. Theioma today (services, request flow, voice pipeline, routing, computer use, runtimes,
measured latencies) · 2. What Tez already has for voice · 3. Local component options in 2026 (ASR, VAD and
endpointing, TTS, computer-use execution) · 4. Design (4a router integration, 4b the voice loop and its budget,
4c packaging, 4d risks, 4e phased plan) · 5. Sources

## Summary

**Theioma today.** Theioma is the `Jarvis` folder: a Next.js 16 command center (`:3000`) with 66 packages,
~22 Python sidecars and a 105-tool MCP channel. Its always-on voice path is not the one its CLAUDE.md
describes. A Python child of the Node voice daemon (`:8804`) listens with openWakeWord ("hey jarvis"), ends an
utterance after ~1.44 s of silence on an RMS gate, and transcribes the whole utterance with faster-whisper
`base.en`. Text then travels daemon -> browser -> `POST /api/voice/converse`, where regex shortcuts or `gemma4:12b`
tool calling through Ollama choose a tool. Tools run as PowerShell processes inside Next.js, and replies come back
as whole-clause Kokoro WAVs played by the browser. Silero VAD, Smart Turn and Parakeet streaming are declared but
not on this path. There is no echo cancellation and no barge-in. The MCP `computer_use_*` tools bypass the
approval gate.

No learned model makes any routing decision. Voice tools (174 registered) reach `gemma4:12b` through
keyword-selected packs. Agent spawns route through keyword tables, a risk floor and a 10 % random roll. The only
learned-router table, `RouterTrainingRow`, is written and never read. Two channel tools cannot work:
`task_orchestrate` gets a 405 on every call and `load_balance` reports success against a missing route.

**Today's latencies** were measured on synthetic speech on 2026-09-06; no live microphone and no computer-use
action has ever been timed:

- Full chain: best turn 8.2 s; median 11.9 s over 6 turns.
- Converse turn: median 8.1 s.
- A media command routed by the 12B over 14 tool schemas: ~21 s. The same command through a regex shortcut:
  ~1-2 s.
- `tool_execute`: p50 0.3 s, p95 1.25 s.
- Kokoro time to first audio: 1.2 s.

Nothing commits early, so an action starts roughly 2 s after the decisive word at best.

**Design.**

- **Tez as the router.** Tez becomes the typed decision layer behind Theioma's rules and in front of its
  generative models, through three doors:
  - in-process, for voice;
  - local HTTP `/v1/systemone`, for the converse turn, the cost-cascade's low-confidence cases and raise-only
    approval checks;
  - MCP, for agents.

  Tool lists become Tez `choice` questions. Beyond 26 options they become a category -> tool hierarchy on the
  hot path, and a tournament, retrieval shortlist or many-class probe for one-shot calls. Per-class conformal
  gates decide act or escalate. Irreversible actions always go through `ApprovalRequest`.
- **A new `tez-voice` process owns the voice hot path:**
  - WASAPI 10 ms capture;
  - cache-aware Nemotron Speech Streaming 0.6B at 80 ms;
  - an in-process Tez session that runs one forward pass per step (bring-up on Gemma 4 12B letters, then a
    cut 4B probe on the hot path with the 12B as verifier);
  - Tez's class-aware early-commit rules;
  - native executors.

  Earcons and cached clips give instant spoken feedback; Theioma receives events.
- **Per-word budget:** ~5 ms capture + ~10 ms step wait + 5-8 ms ASR + 4-8 ms decision + ~2 ms dispatch, about
  25-35 ms plus the recogniser's emission lag. The ~10 ms step wait assumes 20 ms provisional steps; plain
  80 ms steps add ~30 ms. A 50 ms p50 holds if the emission lag averages 20 ms or less, which Phase 1 measures
  as a go/no-go. A p95 under 100 ms is the honest tail target. In Tez's own data, 72 % of commands become
  decidable exactly on the decisive word, so this budget is what the user feels.

**What the 2026 research says (§3).**

| Component | Finding |
|---|---|
| ASR | A cache-aware streaming model (NVIDIA Nemotron Speech Streaming En 0.6B: 80 ms chunks, 8.4 % average WER) must replace the offline Parakeet-TDT. |
| VAD | Silero runs in microseconds on the CPU. Any end-of-turn decision still waits 200-600 ms, so commands must fire from ASR tokens. |
| TTS | Warm Kokoro-82M delivers first audio in 67 ms on a desktop RTX 5090 under Windows, against 1.2 s in Theioma's path. Only cached clips and earcons reliably come in under 20 ms. |
| Computer use | OS-API actions (`SendInput`, cached UI Automation) take milliseconds. Every screenshot -> vision-model step takes 0.6-4 s. |

**Plan.**

| Phase | Work |
|---|---|
| 0 | Word-aligned replay harness and baselines |
| 1 | ASR and in-process decision spikes |
| 2 | `tez.stream` + `tez.voice` |
| 3 | Theioma sidecar on :8818, generated schema, one shared 12B |
| 4 | Learning loop and hardening |

Each phase has numeric exit gates (§4e).

---

## 1. Theioma today

Theioma is `C:\Users\migue\OneDrive\Desktop\Personal\AILABS\Jarvis` (npm workspace `theioma`; HEAD `e3319ccb6`
on `jpex/consolidate-docs-corpus`, 2026-09-24). Below, `J\` stands for that folder. It is a TypeScript monorepo
(a Next.js 16 app and 66 packages) with ~22 Python sidecars, a Rust (Tauri 2) desktop launcher, Postgres
(Supabase) through Prisma 6 (185 models), and an in-process SSE event bus. `Jarvis_DIY` is an April 2026
predecessor snapshot and `theioma-dogfood` is empty; neither is on the live path.

### 1.1 Services

The authoritative list is the `SERVICES` array in `J\scripts\dev-multi.mjs:89-317` (23 services, 22 sidecars).
Rows relevant to voice, routing and computer use:

| Service | Package | Language | Port | Role |
|---|---|---|---|---|
| web | `J\apps\web` (`node server.mjs`) | TypeScript, Next 16.3.4 | 3000 | dashboard, 669 API routes, SSE bus (`lib\sse-clients.ts`), the voice tool registry (voice OS actions execute **inside this process**) |
| stream-mux | `J\packages\stream-mux` | TypeScript (node-pty) | 8791 | PTY mux wrapping each Claude Code session |
| channel MCP | `J\packages\channel` (`dist\index.js`, 9,254-line `src\index.ts`) | TypeScript | stdio; HTTP 127.0.0.1:8790 | 105 tools (`CHANNEL_TOOL_COUNT`); one process per Claude session; not in `SERVICES` |
| engineer-mcp | `J\packages\engineer-mcp` | TypeScript | stdio | 69 `jpex_*` tools, each with a voice twin |
| voice | `J\packages\voice` (`tsx src\daemon.ts`) + Python child `sidecar\theioma_voice_sidecar.py` | Node + Python 3.14 venv | 8804 (WS) | wake word, continuous STT, regex intent router, Piper greetings |
| parakeet-stt | `J\packages\parakeet-stt` | Python 3.13 venv, NeMo 3.0.0, torch 2.11+cu128 | 8795 | Parakeet-TDT-0.6B-v3; used only by the widget's tap-to-talk |
| kokoro-tts | `J\packages\kokoro-tts` | Python 3.13 venv, kokoro-onnx (CUDA EP) | 8796 | Kokoro-82M, 24 kHz |
| moshi | `J\packages\moshi-sidecar` | Python | 8799 | stub: `/health` 503 `mode:"stub"` |
| voiceprint | `J\packages\voiceprint-sidecar` | Python | 8798 | resemblyzer; enrolment only, `/identify` unused |
| desktop-driver | `J\packages\desktop-driver` | Python Flask (bare `python`, 3.14) | 8802 | pywin32 `PostMessage` + mss; binds 0.0.0.0, no auth |
| comfyui, genesis | `comfyui-sidecar`, `genesis-sidecar` | Python | 8800, 8816 | GPU users that compete with voice for VRAM |
| outside `SERVICES` | Ollama (operator-run), `embedder` (BGE-M3), reserved llama-server | - | 11434, 8794, 8797 | Ollama serves `gemma4:12b`; :8797 is reserved and unused |

Port **8818** is not claimed by any manifest. Generated counts live in `J\apps\web\src\app\docs\_generated\`.

### 1.2 How a request flows

- **Operator UI** (browser or Tauri webview) -> `:3000`. `server.mjs` proxies WebSockets at
  `/__cockpit-proxy/mux/` and `/__cockpit-proxy/voice` (`lib\cockpit-proxy.mjs:51-54`). Route handlers use Prisma,
  then `broadcast()` onto the SSE bus (`GET /api/events`).
- **Agent sessions**: `POST /api/sessions/pty` spawns Claude Code under `stream-mux` (`app\api\sessions\pty\route.ts`),
  wires the cost-cascade tier and writes the MCP config for `theioma-channel` and `engineer-mcp`.
- **Channel MCP**: Claude Code -> channel over stdio; channel -> web over HTTP with `x-theioma-secret`
  (`index.ts:103-127`); web -> channel by POST to 127.0.0.1:8790, surfaced as a channel notification.
- **Voice**: microphone -> Python sidecar -> Node daemon -> WS -> browser widget or modal ->
  `POST /api/voice/converse` (SSE) -> Ollama -> voice tool (PowerShell) -> `/api/voice/speak` -> Kokoro -> browser
  `<audio>`. Detailed in §1.3.

### 1.3 The voice pipeline, hop by hop

The live, always-on desktop path does **not** match CLAUDE.md's stack line ("Parakeet + Silero-VAD v5 + Smart
Turn v2 EoU + Kokoro streaming"):

| Hop | What actually runs | Where |
|---|---|---|
| Capture | PortAudio `sounddevice.InputStream`, 16 kHz mono, 80 ms blocks, in the Python sidecar (not the browser) | `J\packages\voice\sidecar\theioma_voice_sidecar.py:110-112, 553-559, 974-983` |
| Wake word | openWakeWord 0.6.0, pretrained `hey_jarvis_v0.1`, ONNX Runtime on CPU (TFLite missing), threshold 0.5; the 2 s wake buffer is re-checked with Whisper | `theioma_voice_sidecar.py:114-121, 446-570`; `J\packages\voice\src\daemon.ts:45-49, 510-518, 856` |
| VAD / endpoint | **RMS gate at -45 dBFS**, endpoint after 18 silent frames (~1,440 ms), 160 ms pre-roll. The Silero branch is dead: `from openwakeword.vad import SileroVad` fails because openWakeWord 0.6.0 only defines `class VAD`. Smart Turn v2 has no loader (the browser EoU detector was deleted in `cb1bffa65`). | `theioma_voice_sidecar.py:247-254, 940-946`; `...\.venv\Lib\site-packages\openwakeword\vad.py:54` |
| STT | **faster-whisper `base.en`** (CUDA fp16) on the whole utterance after the endpoint | `theioma_voice_sidecar.py:299-330, 1105-1126` |
| Sidecar -> daemon | stdio JSONL (wake audio as base64 WAV in JSON) | `J\packages\voice\src\sidecar-client.ts:213-273` |
| Intent | regex `classifyIntent`; a wake that carries a command ("Jarvis, open YouTube") is broadcast as `voice_route` and **no client handles it** | `J\packages\voice\src\intent-router.ts:152-186` |
| Daemon -> browser | WS JSON on :8804 via `/__cockpit-proxy/voice` | `J\apps\web\src\lib\voice-config.ts:72-84` |
| Browser -> turn | `POST /api/voice/converse`, response is SSE | `J\apps\web\src\app\(widget)\voice-widget\page.tsx:707`; `voice-chat-modal.tsx` |
| Decision | deterministic shortcuts, a canned "On it." before any model, then `runAgentLoop` (<= 3 steps) on Ollama `gemma4:12b` (see §1.4) | `J\apps\web\src\app\api\voice\converse\route.ts:2538-2545, 3219-3239` |
| Action | voice tool registry: every tool is `spawnSync("powershell", -File ...)` inside the Next.js process (blocks its event loop) | `J\apps\web\src\lib\voice-tools\registry.ts:128-181` |
| TTS | Kokoro `POST /synthesize` per clause (whole-clause WAV, no streaming), fetched and played by an `<audio>` element; Piper on the host speakers for greetings | `J\apps\web\src\app\api\voice\speak\route.ts:46-69`; `J\packages\kokoro-tts\src\server.py:384-414` |
| Echo, barge-in | **none**: every `packages\voice-aec` adapter is a stub, `BargeinDetector.detect` always returns false; half-duplex muting in the widget only | `J\packages\voice-aec\src\barge.ts:24-31` |

Two other paths exist: the widget's **tap-to-talk** (`getUserMedia` + `MediaRecorder` -> whole-utterance
`POST /api/voice/transcribe` -> Parakeet one-shot decode; the only live use of Parakeet), and a **mobile** path
on the browser's Web Speech API. Parakeet's `WS /stream` builds a `CacheAwareStreamingAudioBuffer` and never uses
it: every frame re-transcribes the whole accumulated buffer (`J\packages\parakeet-stt\src\server.py:446-460,
520-533`); nothing in the app opens that socket, Kokoro's `WS /stream` or Moshi's `/duplex`.

### 1.4 Tool calling and task routing today

**No learned model makes any routing decision in Theioma.** Every choice is a regex, a keyword table or a
hand-weighted sum; the one learned-router data tap, `RouterTrainingRow` (`J\apps\web\prisma\schema.prisma:3526`),
is written at `decideTierAuto` and never read.

| Decision | Made by | Where | Speed |
|---|---|---|---|
| Voice intent in the daemon (kill, brief, ack, activate, command, ...) | regex `classifyIntent` | `J\packages\voice\src\intent-router.ts:152-186` | microseconds |
| Voice turn: media, connect, launch-orchestrator, replies to agents, orchestrator bridge | regex shortcuts that fire tools directly | `converse\route.ts:2256-2524, 2676-2819` (matchers `:802-898`) | microseconds, then the tool |
| Which voice tools the model may see | keyword regexes over 28 intent groups (`partitionVoiceToolsByIntent`); fewer than 4 matches falls back to the 35 always-on tools | `J\apps\web\src\lib\voice-tools\essential-tools.ts:35-92, 130-449, 535-558` | microseconds |
| Which voice tool runs, with what arguments | **Ollama `gemma4:12b` native function calling** (`tools` array; the system prompt also lists the 35 always-on tools); `think:false`, `temperature` 0.4, `num_ctx` 16384, `num_predict` 512, up to 3 steps (`MAX_STEPS`, `:1345`), **no timeout**; tool calls also recovered from JSON in the text and from fenced blocks | `converse\route.ts:1097-1338` (`streamOllama`), `:1484-1913` (`runAgentLoop`); `lib\voice-theioma-tool-schema.ts:62-97` | measured ~0.9-3.7 s per step warm, 26.6 s cold; a media command ~21 s before its shortcut |
| Tool execution | `executeTool`: approval gate, cache, verify wrapper (only 2 of 174 tools have a verify rule) | `J\apps\web\src\lib\voice-tools\registry.ts:128-181`; `verify\verify-wrapper.ts:34-174` | `tool_execute_ms` p50 300 / p95 1,250 ms |
| Cost-cascade tier for text tasks (War Room chat only) | `classifyAuto`: hand-tuned per-tier keyword/tool/length weights; confidence < `DEFERRAL_THRESHOLD` (0.6) means "would defer" | `J\apps\web\src\lib\cost-cascade-classifier.ts:951-1006`; `cost-cascade-confidence.ts:108, 218` | measured p50 0.043 / p95 0.105 ms (`J\bench\reasoning\baseline-R1.json`, 52 synthetic tasks, 3.8 % deferred) |
| Model for a spawned Claude Code session | `routeModel`: `decideTier` (task-kind table + risk floor), keyword difficulty, 14-day error rate, 10 % random roll (epsilon 0.1), budget pressure, local health probe | `J\apps\web\src\lib\routing\model-router.ts:473-703`, called from `app\api\sessions\pty\route.ts:935-940` | not measured; the risk input is MEDIUM for normal roles, so **the local tier is unreachable for PTY spawns** |
| Which agent the delegation engine picks | UCB1 `selectBestAgent` | `J\apps\web\src\lib\bandit-scorer.ts:36-45`; `delegation-engine.ts:560-587` | not measured |
| The `cascade_tier {"tier":"cheap","threshold":1.5}` voice frame | a constant planner; the mid-stream escalation it describes is never invoked | `J\apps\web\src\lib\cascade-planner.ts:259-266`; `cost-cascade.ts:963-974` | cosmetic (drives the sphere's glow) |

Theioma has **174 registered voice tools** on Windows (69 of them the `jpex.*` twins of `engineer-mcp`), 35 of them
always on, 6 behind the approval gate, and **105 channel MCP tools**. The channel's orchestration tools route like
this (all in `J\packages\channel\src\index.ts`):

| Tool | What decides | What happens |
|---|---|---|
| `delegate_to_local` (`:5838-5877`) | the calling agent | `POST /api/orchestration/delegate-local` -> Ollama `gemma4:12b` `/api/chat`, fixed system line, `temperature` 0.3; no timeout at either hop; on failure `{delegated:false}` |
| `task_orchestrate` (`:4762-4810`) | nothing | posts to `/api/tasks/orchestrate`, **which does not exist** (only `api\tasks\[id]` and `active`): every call fails with 405 |
| `skill_execute` (`:5271-5287`) | the caller names the skill; substring lookup | types `/<skill> args` into the caller's own session; invents a "queued" reply if the backend is down |
| `task_auction` (`:6702-6746`) | a weighted sum of caller-supplied scores (0.35 fit + 0.45 tool relevance + 0.20 recent success) | returns a winner id |
| `spawn_agent` (`:4596-4681`) | the caller's `modelTier` (default sonnet) | bypasses `routeModel` entirely |
| `reasoning_cascade`, `swarm_haiku_fanout` | Jaccard agreement across 3 Sonnet calls; caller-supplied fan-out on Haiku | both return 500 without an Anthropic key |
| `load_balance` (`:5353-5380`) | nothing | posts to a route that does not exist and **always reports "rebalanced"** |

Nothing in the repository calls an external decision engine: `tez`, `onepass`, `/v1/decide` have no real
references. The one planned HTTP classifier is a stub: `J\apps\web\src\lib\security\injection-classifier.ts:51-72`
was meant to call a sidecar at `127.0.0.1:8797/classify` with a 50 ms timeout and currently always returns 0.

### 1.5 Computer use today

Three stacks that do not share gates (all paths under `J\`):

| Stack | Entry | Mechanism | Gate |
|---|---|---|---|
| A. Channel MCP `computer_use_capture/click/key/scroll/type` | `packages\channel\src\index.ts:3518-3622` (definitions), `:8389-8414` (handlers), `src\tools\computer_use\*.ts` | HTTP to `desktop-driver` :8802 (`localhost`, 5-10 s timeouts); Win32 `PostMessage` (`WM_LBUTTONDOWN`, `WM_CHAR`, `WM_KEYDOWN`, `WM_MOUSEWHEEL`), mss screenshots as base64 PNG | **none in practice**: the channel sends `approved: true` on every call (`click.ts:16`, `type.ts:14`); only hard blocklists (Win+L, `rm -rf`) apply |
| B. Web `POST /api/computer-use` | `apps\web\src\app\api\computer-use\route.ts:121-328` | same driver | risk-classified: medium/high create an `ApprovalRequest` (`gateKind:"COMPUTER_USE"`) |
| C. Voice tool registry | `apps\web\src\lib\voice-tools\registry.ts:33-58, 128-181` | PowerShell per call: `Start-Process` + fixed 800 ms sleep (`app.launch`), `cmd /c start` (`system.open_app`), `SendKeys` (`keyboard.type`), UIA `ValuePattern` (`uia.set_value`), CopyFromScreen; `computer.do` = screenshot -> vision model (qwen2.5vl:7b first) -> action, 550 ms per step, up to 15 steps | 4 tools need registry approval; audit log `~\.theioma\voice-audit.jsonl`; ESC-hold kill switch (its abort flag has no non-test reset) |

Driver defects worth knowing before any reuse: `hwnd` dropped on type, modifiers and letters silently skipped on
key combos (`['ctrl','c']` sends nothing but returns ok), screen coordinates passed as client coordinates,
`computer.do` scales clicks to the 1280-px downscaled capture. **No computer-use action has ever been
benchmarked** (`J\docs\audit\2026-09-07-launch-readiness.md:97-99`); the design spec claims ~150 ms PowerShell
start-up per call. Reaching the desktop by voice today means: 1.44 s of silence, Whisper, a browser hop, a turn,
often the LLM, then a PowerShell process.

### 1.6 Local model runtimes

| Runtime | State | Where |
|---|---|---|
| Ollama :11434 (0.30.6, operator-run, not in `SERVICES`) | `gemma4:12b` (7.38 GB incl. a 175 MB image projector: the voice model, `delegate_to_local`, the `local-sglang` tier), `gemma4:12b-it-q8_0` (Tez's copy), `gemma3:4b`, `llama3.2:3b`, `bge-m3`, `nomic-embed-text` | `C:\Users\migue\.ollama\models\manifests\registry.ollama.ai\library\`; discovery `J\apps\web\src\lib\cost-cascade-local-tier.ts:233-300` |
| `packages\llama-server` | manifest declares **port 8080** and Qwen3.6-35B-A3B (MTP, Q2_K_XL, WSL2, "147 tok/s"); not in `SERVICES`; CLAUDE.md reserves :8797 for it instead | `J\packages\llama-server\manifest.json` |
| `packages\embedder` :8794 | BGE-M3, real, not in `SERVICES` | `J\packages\embedder\src\main.py:61, 103-119` |
| `packages\model-router` | UCB1 router; its only importer is a dead route | `J\packages\model-router\src\index.ts` |
| `packages\speculator-loop` | EAGLE-3 head training: all four scripts are stubs | `J\packages\speculator-loop\README.md:81-87` |
| Speech models | faster-whisper `base.en` (voice sidecar), Parakeet-TDT-0.6B-v3 (NeMo), Kokoro-82M (ONNX), Piper `en_GB-alan-medium`, openWakeWord ONNX, resemblyzer | §1.3 |
| Vision | Ollama vision models by priority (qwen2.5vl:7b first), else Claude Sonnet | `J\apps\web\src\lib\voice-tools\screen\vision.ts:58-79` |

Measured local-model throughput is thin: `docs\local-llm\hardware-baseline.md` is an unfilled template; the
committed benchmarks (May 2026) cover `llama3.2:3b` (125.5 tok/s, warm TTFT 112 ms; 9/10 correct tool calls in a
10-prompt tool bench, median 950 ms) and `qwen2.5:14b` (33.1 tok/s; 5/10 tool calls, median 3,224 ms), but **no
benchmark of `gemma4:12b`**, the model that makes Theioma's voice tool choices. Tez's measurements (§2) are the
only numbers for Gemma 4 12B on this machine.

### 1.7 Latencies today (measured, with sources)

All Theioma numbers are on synthetic speech fed as WAV (no lane has measured a live microphone), 2026-09-06.

| Measure | Value | Source |
|---|---|---|
| Full chain through the app routes, best | 8,177 ms (CUDA torch) | `J\docs\audit\2026-09-06-voice-e2e.md` §8.1 (D3b) |
| Full chain, local DB rerun, n = 6 (min / median / max) | 5,969 / 11,894 / 15,983 ms | `J\docs\design\evidence\2026-09-06-wave4-e2e-voice-rerun\README.md` |
| Converse turn complete, same rerun | 2,654 / 8,102 / 11,277 ms | same |
| First SSE frame (a canned filler, not the model) | 531 / 638 / 1,078 ms; histogram p50 694 / p95 1,500 ms | same |
| `tool_execute_ms` | p50 300 / p95 1,250 ms (n = 10) | same (in-process histograms) |
| `full_turn_ms` | p50 5,000 ms; p95 at or above the 10 s top bucket (n = 24) | same |
| STT via `/api/voice/transcribe` (Parakeet one-shot) | 107 / 154 / 313 ms; `stt_finalize_ms` p50 138 / p95 288 | same |
| Parakeet `WS /stream` first partial / finalize | 1,787 / 1,257 ms (D3b); 731 / 173 ms (rerun) | D3b; rerun |
| Kokoro `WS /stream` TTFT on GPU | 1,175 ms; one-word floor 683 ms, attributed to 241 host-device copy nodes plus CPU espeak phonemisation | `J\docs\design\evidence\2026-09-06-wave4-kokoro-gpu\README.md` |
| Continuous-mode endpoint | ~1,440 ms of trailing silence by design, before Whisper runs | `theioma_voice_sidecar.py:944` |
| A media command through `gemma4:12b` (14 tool schemas, ~20.6k-char prompt) | ~21 s, versus ~1-2 s once a regex fast path was added | comment at `J\apps\web\src\app\api\voice\converse\route.ts:843-855` |
| Wake detection, live microphone to STT, browser playback start, any computer-use action | never measured | - |

Budgets exist only as targets: full turn p95 < 800 ms (`/voice/metrics` page), `llm_ttft` p95 < 200 ms, a "V1
gate" of 600 ms p95. `J\bench\voice\baseline.json` (612 ms full-turn p50) is synthetic (`"synthetic": true` in
every trace). The "~700 ms p95" and "Kokoro 50 ms TTFT" claims survive in `docs\ARCHITECTURE.md:188`,
`docs\OPERATORS-GUIDE.md:554`, `speak\route.ts:6` and `converse\route.ts:2529`; no measured path supports them.

**Word-to-action today:** there is no early commit anywhere. An action can start only after the whole utterance
ends, 1.44 s of silence passes, Whisper transcribes, the browser posts a turn, the turn is routed (a regex
shortcut, or seconds to tens of seconds through the 12B with tool schemas), and PowerShell starts. Measured from
the decisive word, the best case is roughly 2 s and the common case several seconds.

### 1.8 What this means for the design

1. The live path's slowest parts are structural, not tunable: a 1.44 s silence endpoint, whole-utterance
   Whisper, a browser round trip in the middle of the loop, and tool selection by a generative 12B over tool
   schemas. None of them can reach 50 ms; they have to leave the hot path.
2. Parakeet-TDT is an offline model, and Theioma's "streaming" endpoint re-decodes everything. Streaming needs a
   cache-aware model (§3.1), not a better wrapper.
3. The OS actions run as PowerShell processes inside Next.js, and the MCP computer-use path is ungated. The hot
   path needs native executors, and the gates need fixing regardless of this project.
4. There is no measurement of a live microphone or of any action's latency. Phase 0 has to create the harness
   before any number in this design can be claimed.
5. Tez would be Theioma's first calibrated decision component, and the sockets for one already exist: the
   deferral threshold in `cost-cascade-confidence.ts`, the write-only `RouterTrainingRow`, the injection-classifier
   stub, and the converse route's `withTimeout` slot between its shortcuts and its agent loop.

---

## 2. What Tez already has for voice

Everything below is in `C:\Users\migue\OneDrive\Desktop\Personal\AILABS\Tez`. The voice loop exists only as
**experiments**; the installable runtime (`tez/`) has no streaming or audio code yet (README roadmap item 5 lists
"a streaming/early-commit contract with action classes" as still open).

| File | What it does |
|---|---|
| `experiments/run_voice_fast.py` | Transcript-LAST prompt (instructions + 16 actions first, transcript last) so the constant part is a KV-cache prefix; scores every word prefix; one forward pass, letter logits over 16 options, nothing decoded. |
| `experiments/stream_policy.py` | Class-aware commit: `none` on a prefix never commits; OPEN fires on one partial at p >= 0.9; MEDIA / `play_liked_songs` need 2 consistent partials; slot-bearing PLAY/SEARCH open their app early and search at the end; `type_text` / `close_app` only at end of utterance; refinement graph (open_youtube -> play_youtube, open_notes -> type_text). |
| `experiments/residual_check.py` | Compound commands as two decisions: after an action fires, only the residual words are scored, with a `prior` line naming the action already taken. |
| `experiments/voice_actions.py` | The model only picks the intent; slots are regex over the transcript; executors are plain OS calls (`os.startfile`, `pynput` media keys and typing, `webbrowser`, `taskkill`). Dry run by default. |
| `experiments/asr_bench.py` | faster-whisper partials: re-decodes the whole growing buffer after every 200 ms chunk (no VAD, greedy). |
| `experiments/voice_demo.py` | Live loop (`--text` / `--wav` / `--mic`): `Decider` (cursor, residual, prior, class policy) calls llama-server over HTTP per partial; RMS gate for voice; 600 ms of silence ends the utterance. |
| `data/voice/intents.json`, `commands.jsonl`, `wav/` | 16 actions, 220 labelled commands in six styles with a human-annotated commit word, 220 SAPI-synthesised 16 kHz WAVs. |
| `tez/engine.py`, `tez/server.py`, `tez/backends.py`, `tez/prompt.py`, `tez/gate.py` | The runtime: `Tez.decide()`; `noul` / `choice` (2-255 options) / `score`; above 26 options a chunked tournament (chunks of 20 + "none", then a final); letters via llama-server `/completion` (top-200 log-probs) or a probe via `/embedding`; conformal act/escalate gate; FastAPI server on 127.0.0.1:8787 speaking TypeSafe's `/v1/systemone`. |

### Measured (Gemma 4 12B Q8_0, llama.cpp b11100, RTX 5080 Laptop)

Sources: `BENCHMARKS.md` §3, `docs/REPORT.md` §3.4 and §4.4, `results/voicefast_final_last_gemma4-12b-q8_0.summary.json`,
`results/asr_bench_base-en_gemma4-12b-q8_0.summary.json`.

| Quantity | Value |
|---|---|
| Intent accuracy, 220 commands (strict / accepting `then` and `alt`) | 0.905 / 0.982 |
| Out-of-scope -> `none` (recall / precision) | 1.000 / 0.917 |
| Uncached tokens per streamed word (with `--swa-full`) | 11 (the new word, the closing quote, the template tail) |
| Decision compute per word, p50 / p95 | 31.1 / 36.6 ms |
| HTTP round trip per word, p50 / p95 | 47.4 / 56.7 ms |
| After an action fires (33-token `prior` line) | ~55 ms compute / ~70 ms round trip |
| Without `--swa-full` (Gemma's sliding-window cache cannot roll back) | 394-397 tokens re-evaluated, 205 ms |
| Class-aware policy, 198 actionable utterances | 1 harmful action; final action consistent 0.985; first action at mean word 3.6; 2/22 out-of-scope utterances trigger a (cheap, reversible) open |
| Compound commands via residual + `prior` | 22/22 |
| ASR faster-whisper base.en fp16, 200 ms chunks, re-decode | partial p50 40.9 / p95 130.6 ms (35 ms at 1 s of audio, 62 ms at 4 s); WER 7.8 % on synthetic speech; ASR-vs-text intent agreement 97.3 % |
| small.en | 120 ms p50, WER 7.3 %: 3x slower for < 1 point |
| Live `--wav` "open notes and type hello world" (contended GPU) | partial "open note" at +493 ms, Notepad launched at +652 ms from the start of the audio, `type_text` at end of speech |
| Stated per-word budget | ~40 ASR + ~30 decide + ~15 HTTP + < 1 policy = ~85 ms |

Three things this budget does not include, and which the 50 ms design has to remove:

1. **Chunk quantisation.** Audio reaches ASR in 200 ms blocks (`sounddevice` blocksize = chunk), so a word that
   ends just after a block boundary waits up to 200 ms (mean ~100 ms) before any computation starts.
2. **Whisper is not a streaming model.** Re-decoding the growing buffer costs 35-130 ms and grows with the
   utterance, and partials flicker ("lofi" / "low-fee"), which is why slot-bearing actions need two stable partials.
3. **The 652 ms is measured from the start of the audio, not from the end of the decisive word.** It shows the
   action fires ~1.3 s before speech ends; it is not a word-to-action latency. No word-aligned measurement exists
   yet (Phase 0 below adds one). In that run, 159 ms passed between the partial's timestamp and the launch.

The costs that are already right: the prompt layout (constant first, transcript last), the `none`-means-keep-
listening rule, the action classes, the residual + `prior` trick, deterministic slots, and the measurement
discipline (manifests with data hashes and model paths).

---

## 3. Local component options in 2026 (research)

Hardware assumed: Windows 11, RTX 5080 Laptop GPU (Blackwell sm_120, 16 GB GDDR7, 256-bit, 896 GB/s, 80-150 W).
On this machine 15.6 of 16.3 GB of VRAM were in use during this audit (a llama-server and a Python measurement,
plus the usual Windows desktop apps that each hold some VRAM), so every component below is judged on memory
as well as speed.

### 3.1 Streaming ASR

The key distinction is **cache-aware** (true streaming: each step processes only the new audio and reuses
cached encoder state, so a transcript token is final once emitted) versus **buffered / re-decode** (the left
context is recomputed every step; latency and cost grow, partials can change).

| Model (date) | Size | Type | Latency setting | Accuracy | Licence | On an RTX 5080 Windows laptop |
|---|---|---|---|---|---|---|
| NVIDIA **Nemotron Speech Streaming En 0.6B** (HF 2026-03-13; checkpoint 2026-01-05) | 600M, 24-layer FastConformer + RNNT | cache-aware | `att_context_size` [70,0] = 80 ms chunk, [70,1] = 160 ms, [70,6] = 560 ms, [70,13] = 1.12 s | avg WER 8.43 % at 80 ms, 7.67 % at 160 ms, 6.93 % at 1.12 s (LibriSpeech clean 2.80 / 2.56 / 2.32 %); punctuation and capitals | NVIDIA Open Model License | Primary pick. Same NeMo stack Theioma already runs for Parakeet (officially Linux; Theioma runs NeMo on Windows with a pinned cu128 torch). Also exported to ONNX Runtime GenAI by Microsoft (Foundry Local; int4 0.67 GB, RTFx 7.2 on CPU at 560 ms). |
| NVIDIA **Nemotron 3.5 ASR Streaming 0.6B** (June 2026) | 600M | cache-aware | [56,0] = 80 ms ... [56,13] = 1.12 s | FLEURS-en WER 9.43 % at 80 ms, 8.88 % at 160 ms | OpenMDW-1.1 | Multilingual (40 locales) alternative; needs NeMo 26.06. |
| NVIDIA **parakeet_realtime_eou_120m-v1** | 120M, 17 layers | cache-aware | [70,1] (80-160 ms) | avg WER 9.30 %; emits an `<EOU>` token, EOU latency median 160 ms, p90 280 ms; no punctuation | NVIDIA Open Model License | Small ASR that is also an endpointer; English only. |
| NVIDIA Parakeet-TDT-0.6B-v3 (what Theioma runs) | 600M | **offline model**, buffered streaming only | - | 6.32 % batch; 9.22 % at 2.4 s buffered delay (Microsoft, Apr 2026) | CC-BY-4.0 | Theioma's `/stream` wrapper re-decodes the whole buffer with this offline model every frame, hence first partials of 0.7-1.8 s; no wrapper can make it cache-aware. |
| NVIDIA parakeet-unified-en-0.6b (2026-04-07) | 600M | buffered only ("cache-aware streaming not currently supported") | 0.08 s + 0.08 s = 160 ms | 5.91 % offline avg | NVIDIA Open Model License | Good offline/final pass; not for the per-word path. |
| **Moonshine v2** (arXiv 2602.12241, 2026-02-12) | Tiny 34M / Small 123M / Medium 245M | sliding-window streaming encoder, 80 ms lookahead | response latency on Apple M3: 50 / 148 / 258 ms | Open-ASR avg WER 12.01 / 7.84 / 6.65 % | MIT (`pip install moonshine-voice`, Windows supported) | CPU fallback when the GPU is saturated. |
| Kyutai STT (June 2025) | 1B (en/fr) / 2.6B (en) | streaming, fixed delay | 500 ms / 2.5 s | - | CC-BY-4.0 | Delay floor of 500 ms rules it out for the action path; its semantic VAD idea is worth copying. |
| Mistral Voxtral Mini 4B Realtime 2602 (2026-02) | 4B | natively streaming | 80-2400 ms (480 ms recommended) | FLEURS avg 12.60 % at 160 ms, 8.72 % at 480 ms | Apache-2.0 | Needs >= 16 GB GPU on its own; does not fit next to Tez. |
| faster-whisper base.en (Tez's choice) | 74M | re-decode | 200 ms chunks | 7.8 % (synthetic) | MIT | Measured 41 / 131 ms p50 / p95 per partial; flickers. Keep only as a fallback. |
| Whisper + SimulStreaming / WhisperLiveKit (AlignAtt, LocalAgreement) | - | policy over re-decode | 200-800 ms first word | - | MIT | Too slow for the action path. |
| Vosk (Kaldi), sherpa-onnx Zipformer / NeMo streaming exports | 20M-100M | true streaming, CPU | Zipformer chunk-16 = 320 ms; sherpa-onnx ships NeMo streaming FastConformer at 80 / 480 ms | lower than Nemotron | Apache-2.0 | Torch-free Windows path (ONNX Runtime); sherpa-onnx is how to run the NeMo models without NeMo if needed. |

NVIDIA's own numbers for the family (blog, 2026-01-05): median time from the end of the audio to the final
transcript 24 ms on an H100 in the 560 ms mode, independent of utterance length, and 560 concurrent streams per
H100 at 320 ms chunks. That is a data-centre GPU and a slower mode; the single-stream step cost on the laptop at
80 ms is not published anywhere, so the design's 5-8 ms is an estimate for Phase 1 to measure.

Two published techniques matter for the 50 ms target: **FastEmit** regularisation (arXiv 2010.11148; NeMo's RNNT
loss exposes `fastemit_lambda`, recommended 1e-4 to 1e-2) cuts transducer emission delay by ~200 ms in the paper and
can make tokens appear before the word ends; and **CTC-based word spotting** on the streaming encoder
(arXiv 2605.18222) for a fixed vocabulary such as app names.

Pick: **Nemotron Speech Streaming En 0.6B at the 80 ms setting** for the action path (fall back to 160 ms if
emission lag or WER is worse than expected), `parakeet_realtime_eou_120m-v1` as the small alternative that also
endpoints, Moonshine v2 Small on CPU as the fallback.

### 3.2 VAD and end of turn

Legend for this and the next two sections: [V] vendor or author claim, [I] independent measurement, [D] derived
estimate.

| Component (date) | Decides | Frame | Compute cost | Windows CPU | Licence |
|---|---|---|---|---|---|
| **Silero VAD v6.2.3** (2026-09-23) | speech probability per frame | 512 samples at 16 kHz = 32 ms | < 1 ms per chunk, single thread [V]; v5 ONNX 189 us per chunk on a Threadripper [V]; 63 us per frame in an independent Rust + ONNX Runtime harness [I] | yes (ONNX Runtime or TorchScript, ~2 MB) | MIT |
| TEN VAD v1.0 (2025-07-11) | speech per frame | 10 or 16 ms hops | RTF 0.015 on a Windows i7 [V], ~0.24 ms per hop [D] | yes (prebuilt Windows libraries) | Apache-2.0 **plus a clause against competing with Agora** |
| WebRTC VAD | speech per frame (GMM) | 10/20/30 ms | ~2 us per frame [I] | yes (`webrtcvad-wheels`) | BSD / MIT; far less accurate (Silero's wiki: ROC-AUC 0.73 vs 0.97 for Silero v6 [V]) |
| **Pipecat Smart Turn v3.2** (2026-01-07) | end of turn from the turn's audio (Whisper-Tiny encoder, ~8M params, 8 MB int8, 23 languages) | once per VAD pause | 9 ms int8 / 13 ms fp32 on an AWS c7a.2xlarge CPU for v3.1 [V]; up to 95 ms on a t3.medium for v3 [V] | yes (ONNX, int8) | BSD-2 |
| NVIDIA parakeet_realtime_eou_120m-v1 | ASR that emits `<EOU>` | 80-160 ms | EOU p50 160 / p90 280 / p95 320 ms after end of speech [V] | NeMo (Linux preferred); community ONNX ports | NVIDIA Open Model License, English |
| LiveKit turn detector v1 / v1-mini (2026-06) | audio + text end of turn | per turn | legacy text model 50-160 ms [V] | CPU | **licence limits it to LiveKit Agents** |
| TurnSense 1.1 (2026-05-22), Namo v1 | text-based end of turn | per turn | CPU p50 55 ms [V]; < 29 ms quantised [V] | CPU | Apache-2.0 |
| Kyutai semantic VAD | end-of-turn probability inside Kyutai STT | - | bound to the STT's 0.5 s delay | Rust server only | - |

The compute is microseconds to milliseconds; the **waiting** is not. Silero's `VADIterator` declares the end of
speech only after `min_silence_duration_ms` (default 100 ms) of quiet frames; Pipecat waits `stop_secs` (0.2 s)
before running Smart Turn; LiveKit's Silero default `min_silence_duration` is 0.55 s. Any end-of-turn decision
therefore lands 200-600 ms after the last word, so **commands must fire from ASR tokens, never from VAD or turn
detection**; VAD opens and closes the ASR stream and detects barge-in, and turn detection only closes deferred
actions and conversational turns.

Pick for this laptop: Silero v6.2 ONNX on the CPU (one intra-op thread, exact 512-sample frames; est. 0.1-0.3 ms
per frame on a laptop core), threshold 0.5 / 0.35, minimum speech 60-100 ms so one-word commands survive,
minimum silence 100-150 ms for segmentation only, 30-60 ms padding and a 300-500 ms pre-roll into the ASR; Smart
Turn v3.2 int8 on the CPU for deferred and dictation endings, with `stop_secs` ~0.2 s and a 3 s fallback.
Theioma already ships a Silero v6 ONNX file inside faster-whisper's assets, but its own Silero branch is dead
(§1.3).

### 3.3 Low-latency local TTS

"Win 5090" is the independent tts-bench rig (Windows 11, Ryzen 9 9950X3D, desktop RTX 5090, stock Python paths;
figures re-checked on the benchmark's page for this audit). For non-streaming models its time to first audio is
the time to synthesise the whole short prompt.

| Model (release) | First audio | Speed | Streams text in? | Windows | Licence |
|---|---|---|---|---|---|
| **Kokoro-82M** (v1.0, 2025-01-27) | Win 5090 GPU **67 ms warm**, 895 ms cold; Win CPU 532 ms warm [I]; Kokoro-FastAPI ~300 ms first chunk on GPU [V] | 104x real time on GPU, 14x on CPU [I] | no (sentence chunks) | yes | Apache-2.0 weights; kokoro-onnx MIT |
| **Piper** (piper-tts 1.8.0, 2026-09-04) | Win CPU **107 ms warm**, 160 ms cold [I] | 59x real time [I] | no (per sentence) | yes | **GPL-3.0** now (the MIT repo was archived 2025-10-06) |
| **Kyutai Pocket TTS** (100M, Jan 2026) | Win CPU **119 ms warm** [I]; ~200 ms [V] | 4.4x real time on CPU [I] | streams audio out | yes | code MIT, weights CC-BY-4.0 (gated) |
| Kyutai TTS 1.6B (July 2025) | 220 ms from the first text token [V] | - | **yes** (true text-in streaming) | undocumented | code MIT/Apache, weights CC-BY-4.0 |
| Sopro V2 Turbo (120M, 2026-08-27) | Win 5090 GPU 400 ms warm (default-mode table) [I] | 19x real time [I] | streams audio out | yes | Apache-2.0 |
| StyleTTS 2 | Win 5090 GPU 265 ms warm [I] | 32x [I] | no | yes | MIT |
| Chatterbox-Turbo (Dec 2025) | claims 75 ms [V]; 1.62 s warm on the stock Windows path [I] | 4.3x [I] | no | yes | MIT, watermark always on |
| VibeVoice-Realtime-0.5B (2025-12-03) | claims ~200-300 ms [V]; 3.77 s on the stock path [I] | 2.4x [I] | advertised, listed as TODO | container + flash-attn | MIT, research use only |
| Qwen3-TTS 0.6B / 1.7B (2026-01-22) | 97-101 ms with vLLM, torch.compile, CUDA graphs [V]; 1.60 s stock [I] | 3.8x [I] | not in the released package | yes | Apache-2.0 |
| Orpheus 3B, CSM-1B, Dia, CosyVoice 3, Fish S2, Voxtral-4B-TTS | 0.36-23 s on stock paths [I], or sub-200 ms only on Linux serving stacks [V] | around or below real time | varies | mostly Linux-first | several non-commercial or research-only |

Vendor claims and independent measurements differ by 10-50x for the LLM-codec models (Chatterbox-Turbo 75 ms
claimed vs 1.62 s measured; VibeVoice ~300 ms vs 3.77 s). Their fast paths are vLLM, SGLang or TensorRT-LLM
servers, which are Linux-first. Two Blackwell notes: PyTorch 2.7 with cu128 wheels is the first release that
supports sm_120, and onnxruntime-gpu 1.23 failed on an RTX 5090 with a PTX JIT error (microsoft/onnxruntime
issue 26177, 2025-09-26), which matters for Theioma's Kokoro-on-ONNX-CUDA path. The laptop RTX 5080 is rated
1,334 AI TOPS against the desktop RTX 5090's 3,352, so expect Kokoro-size models to run perhaps 1.5-2.5x slower
than the table [D].

**Earcons and cached phrases.** The Windows audio engine adds 1.3 ms; shared-mode buffers default to ~10 ms and
go down to 128 samples (2.66 ms at 48 kHz) with `IAudioClient3`. A pre-decoded clip written into an
already-open WASAPI stream is audible roughly 5-20 ms after the trigger on wired or USB output [D]; Bluetooth
A2DP adds ~100-200 ms, so it is unsuitable for confirmations. LiveKit's `BackgroundAudioPlayer` uses the same
pattern in production.

Pick for this laptop: tier 1, earcons and phrases pre-rendered with Kokoro and held as PCM in RAM, on a
permanently open WASAPI stream (<= 20 ms); tier 2, Kokoro-82M kept warm on the GPU (PyTorch 2.7+ cu128) and
synthesised per clause for dynamic replies (est. 70-170 ms on the laptop); CPU fallback Pocket TTS or Piper
(~110-120 ms first audio). Theioma's Kokoro sidecar, at 1,175 ms streaming TTFT and a 683 ms one-word floor, is
paying roughly 10-15x what the same model needs on a comparable GPU.

### 3.4 Fast local computer-use execution on Windows

| Mechanism | Typical latency | Notes |
|---|---|---|
| Win32 `SendInput` (batched `INPUT[]`) | effectively instantaneous; no official figure | the modern API (`keybd_event` is superseded); ~5,000 characters per send; **UIPI silently blocks input to elevated windows** |
| `pyautogui` / `pydirectinput` | **>= 100 ms per call by default** (`PAUSE = 0.1`) | set `PAUSE = 0` or call `SendInput` directly |
| `pynput` | no default delay | what Tez's experiments use |
| UI Automation queries | no authoritative per-call figure; each uncached property read is a cross-process COM call; full-screen element dumps take seconds to minutes; a UIA-based MCP server reports 0.2-0.5 s between actions [V] | resolve elements ahead of time with `CacheRequest`, refresh on UIA events, invoke on a worker thread with a timeout (a misbehaving provider can block `Invoke` until a modal dialog closes) |
| pywinauto, UIA backend | anecdotally ~6x slower than its win32 backend; some lookups take tens of seconds | setup and discovery only |
| `SetForegroundWindow` | fast when allowed | allowed only if the caller received the last input event or owns the foreground; otherwise the taskbar button flashes |
| App launch (`ShellExecute`, `os.startfile`) | Windows Terminal 757 ms cold vs 175 ms warm (issue tracker, 2024) | `os.startfile` returns once the launch is started; no published Notepad timing (Notepad 11.2605 and 11.2606 shipped launch-performance work in mid-2026); Windows 11's Low Latency Profile (KB5121003, build 26200.9168+) claims faster launches [secondary] |
| Screen capture | DXcam (Desktop Duplication) ~4-10 ms per frame; mss ~13-35 ms | |
| Screenshot -> parse | OmniParser V2 0.6 s per frame on an A100, 0.8 s on a 4090 [V] | parsing only; the model's decision comes on top |
| Screenshot -> VLM -> action | UI-TARS-2: 2.5-4.0 s per round [V]; OSWorld-Human: planning and reflection take 75-97 % of agent latency | Anthropic's computer-use docs suggest ~0.5 s settling between actions; OpenAI's guide now recommends code execution for its newest model |
| MCP on Windows / App Actions | structured invocation, no latency published; MCP docs marked prerelease | worth watching for app-level actions |

Achievable tiers [D]: **under 10 ms**: `SendInput` into a focused, responsive window; messages to a known window
handle; pattern calls on pre-resolved, cached UIA elements; starting an earcon; a DXGI frame grab. **10-200 ms**:
Smart Turn, warm Kokoro, Piper or Pocket first audio, mss, a warm app relaunch, any default-settings `pyautogui`
call. **0.2-1 s**: a cold app launch, UIA-based action loops, OmniParser, silence-based end of turn. **Seconds**:
every screenshot -> VLM -> action step.

Pick for this laptop: every recognised command maps to a pre-compiled OS action (ctypes `SendInput` batches,
`ShellExecuteExW`, window messages, cached UIA pattern calls on a worker thread), never to `pyautogui` defaults, a
PowerShell process or a vision step; target apps are pre-launched where possible and switched to rather than
started; the process runs at the same integrity level as its targets (UIPI); text longer than a phrase is pasted
through the clipboard with history exclusion; screenshots and vision stay a verification or fallback path. This is
the opposite of Theioma's voice executors today (PowerShell per call, `SendKeys`, `Start-Process` plus an 800 ms
sleep, a 550 ms-per-step vision loop).

### 3.5 Prior art for fixed-intent speech control

From Tez's own literature notes (`Tez\docs\research\RESEARCH-NOTES.md` §3), for calibration of expectations:

- For 10-100 fixed intents, 97-99 % accuracy with under 100 ms of decision time has been available since about
  2019: Picovoice Rhino (audio to intent, built-in endpointing, proprietary), Snips NLU (60 ms on a Raspberry
  Pi 3), keyword spotters such as BC-ResNet (12 ms on a Cortex-M7).
- Silence timeouts of 300-800 ms are often the single largest contributor to voice latency (Soniox's endpoint
  notes). A silence-gated loop on consumer hardware floors at about 350-500 ms; only early commit gets under
  200 ms.
- Early commit is established: streaming SLU spots over 30 % of intents before the utterance ends without
  accuracy loss; Google's streaming intended-query detector saves 600 ms (arXiv 2208.13322). Calibration is what
  makes it safe.

What Tez adds over that prior art is an action list you can edit without training, a calibrated early-commit
signal with a measured harm rate, and one decision primitive shared with the LLM fallback.

---

## 4. Design

### 4.0 Principles carried over from the Tez evidence

- **Rules first, Tez behind the rules, the big model behind Tez.** Tez's own report (`docs/REPORT.md` §6) says a
  30-100 ms model in front of a 2 ms rule is a regression; it goes behind the rule's deferral threshold.
- **Tez picks; code does.** Tez chooses among typed options and says how sure it is. Arguments come from
  deterministic extraction, then grammar-constrained generation, then the big model. Tez never writes arguments.
- **Tez may raise scrutiny, never lower it** (`REPORT.md` §4.6). Irreversible actions stay behind Theioma's
  approval gate whatever Tez's confidence.
- **Constant first, variable last.** Every prompt the voice loop evaluates has a cached prefix; only the new
  transcript tokens and the answer tail are computed per step.
- **Nothing is claimed without a word-aligned measurement.** Theioma's history (the "~700 ms p95" claim, the
  Moshi stub) and Tez's audit trail both argue for the same rule: every milestone below is a number from a replay
  harness with a manifest.

### 4a. Tez as Theioma's tool router and task router

#### Three ways in, one per caller

| Mode | Caller | Latency it adds | Use it for |
|---|---|---|---|
| **In-process Python** (`tez.stream.Session` inside `tez-voice`) | the voice hot path | none beyond the forward pass (no HTTP, no serialisation) | per-word decisions; the only mode that fits the 50 ms budget |
| **Local HTTP** (`tez serve` speaking `/v1/systemone`, called from `apps/web`) | Theioma's TypeScript: the converse turn, the cost-cascade, approvals, task routing | ~15 ms of HTTP + one forward pass (Tez measured 45 ms with a cached prefix, 139 ms per `tez serve` call and 178 ms for a fresh 142-token prompt on the 12B) | one decision per request, where the alternative is seconds of generation |
| **MCP tool** (`tez_decide`, `tez_route`: new tools on `theioma-channel` proxying to `tez serve`, or the standalone `tez mcp`) | Claude Code sessions and Theioma's agents | irrelevant next to an agent step | cheap typed questions inside agent work ("flaky or real failure?", "which of these 105 tools?") without spending model tokens |

The HTTP server and the voice process load the same schemas and write the same decision log, so a decision
means the same thing whichever door it came through.

#### Where it plugs in, in order of value

1. **The converse turn, between the regex shortcuts and the agent loop.** Today a turn that misses the
   shortcuts (`J\apps\web\src\app\api\voice\converse\route.ts:2256-2524, 2676-2819`) goes to `runAgentLoop`
   (`:3219-3239`, up to 3 Ollama steps), which re-reads a large system prompt with every candidate tool schema
   (the media case: 14 schemas, ~20.6k characters, ~21 s). Insert one Tez `choice` over the tools the turn
   could use, time-boxed with the route's existing `withTimeout` (`:661-666`):
   - **act** (calibrated, single-member conformal set, tool class allows it): run the tool directly through the
     registry's `executeTool`, with arguments from deterministic extraction; if the tool needs free-form arguments,
     make one grammar-constrained generation call that carries **only that tool's schema**;
   - **escalate**: call `runAgentLoop` as today, but with Tez's top-k shortlist instead of the whole pack, which
     shrinks the prompt the 12B has to prefill;
   - **`none`**: a conversational turn, unchanged.
2. **Tool-pack selection.** Voice tools reach the model through keyword-triggered packs
   (`J\apps\web\src\lib\voice-tools\essential-tools.ts:195-277, 535-558`). Keywords miss paraphrases (the
   desktop-control pack's keywords do not include "type", so "open Notepad and type hello" probably lands in the
   vision loop). A Tez category question picks packs by meaning, with the keywords kept as a floor.
3. **The cost-cascade, behind its own threshold.** `confidenceFromSignals` and `DEFERRAL_THRESHOLD = 0.6`
   (`J\apps\web\src\lib\cost-cascade-confidence.ts:108, 218`) already mark the decisions `classifyAuto` is unsure
   of (3.8 % of its synthetic bench). Only those go to Tez, as a `choice` over tiers with the task text as the
   state. When Tez acts, its choice and distribution go into the existing `RouterTrainingRow` (`routerScore` =
   Tez's calibrated probability, `features` = its distribution); when it escalates, the heuristic's choice stands
   (or the higher tier, which is the conservative direction). The 0.1 ms heuristic stays in front, as Tez's own
   report demands. Today only the War Room chat calls `classifyAuto`; spawns go through `routeModel`
   (`J\apps\web\src\lib\routing\model-router.ts:473-703`), which has no confidence value, so extending the same
   pattern there first needs `routeModel` to expose one (its keyword and error-rate signals are enough to compute
   it).
4. **Agent and task routing.** Spawn routing keys on a task kind derived from the agent's role by substring rules
   (`J\apps\web\src\app\api\sessions\pty\route.ts:917-929`) and on a risk level that is MEDIUM for every normal
   role. A Tez `choice` over the task kinds of `DEFAULT_KIND_TIER` replaces the substring rules; the tier table
   and the risk floor stay exactly as they are, because Tez never lowers risk (which also means Tez does not by
   itself make the local tier reachable for spawns; that is a separate policy decision). `delegate_to_local` gets
   a Tez `noul` "can the local model do this subtask well?" before it spends an unbounded 12B generation, and the
   delegation engine, `skill_execute` and `task_auction` get Tez shortlists (top 3-7, never one hard pick) over
   agents (171 templates), skills and bidders, which the existing logic then chooses from. `task_orchestrate`
   (always 405) and `load_balance` (fake success) must be fixed first: Tez cannot route through a door that does
   not exist.
5. **Approvals, raise-only.** A Tez `noul` "does this action need a human?" may turn a low-risk classification
   into an `ApprovalRequest`; it can never approve anything or lower a risk class. This also closes the MCP
   computer-use gap cheaply once the hard-coded `approved: true` is removed.
6. **The injection-classifier slot that is already there.** `J\apps\web\src\lib\security\injection-classifier.ts`
   was designed to call a local classifier on :8797 with a 50 ms timeout and to fail open, and returns 0 today. A
   Tez `noul` is that classifier: zero-shot Tez scored 0.759 on the prompt-injection set and 0.865 on jailbreak
   guardrails in the head-to-head (Tez `BENCHMARKS.md` §1, §1b). The 50 ms timeout rules out the 12B's uncached
   57-250 ms; it fits a cut-4B probe (which needs labels; Tez's task-probe recipe trains one from a public
   injection set), and fail-open stays the contract.

#### Turning tool lists into Tez questions

- **Source of truth:** Theioma's voice tool registry and the channel's generated tool list. A generator emits
  the Tez schema: option id = tool name, criterion = the tool's one-line description, plus the `actions:` block
  (class, slots, executor, spoken confirmation). A parity test fails CI when a tool lacks its Tez entry. A
  schema change changes the fingerprint, and Tez already falls back to zero-shot letters for stale probes.
- **Up to 26 options:** one `choice` question with the implicit `__none__` option (`tez.abstain`).
- **More than 26 options:**
  - *On the voice hot path:* a two-level hierarchy with fixed option blocks, so both levels stay cacheable: a
    category question (packs; <= 25 + none) and one member question per category (split any category above 25;
    the 69 J-PEX twins split by room). Each step evaluates the category question and the member questions of
    the top two categories in the same decode; the commit uses p(category) x p(tool | category) against the
    class threshold.
  - *One-shot (HTTP):* Tez's existing chunked tournament (chunks of 20 + "none", then a final) or a retrieval
    shortlist (Theioma already runs BGE-M3 on :8794) of 20, then letters. The in-process engine runs the
    tournament chunks as parallel sequences in one decode.
  - *Long term:* a many-class probe. A probe's cost does not grow with the number of tools; it needs labels,
    which the decision log and the 12B teacher supply.
- **Arguments:** deterministic first (regex, lexicons of app names, fuzzy match against Theioma's own lists of
  projects, agents and approvals), then grammar-constrained generation with the single chosen tool's JSON
  schema, then the big model. Tez never generates arguments.
- **Compound commands:** the residual + `prior` mechanism (22/22 in Tez's measurement).

#### Confidence gating and escalation

| Tez says | Instant / refinable action (R0, R1) | Deferred action (R2) | Gated action (R3) | One-shot routing |
|---|---|---|---|---|
| act (calibrated p >= the class's conformal cut) | fire; verifier checks after | fire at end of utterance if the 12B verifier agrees and slots validate | open an `ApprovalRequest` | take Tez's choice |
| escalate (conformal set of 2+) | keep listening; at end of utterance ask "A or B?" by voice | 12B letters with six-ordering permutation averaging; then the LLM with the shortlist | open an `ApprovalRequest` | LLM with Tez's shortlist (local 12B, then the cost-cascade's Claude tier) |
| `none` on a prefix | keep listening | keep listening | - | - |
| `none` at the end | hand to `/api/voice/converse` | same | same | conversational turn |

Cloud escalation needs `ANTHROPIC_API_KEY`, which none of Theioma's env files sets today (per the routing
survey), so in the current deployment escalation ends at the local 12B and then a clarifying question.
Thresholds are per action class and set by Tez's conformal gate (`tez fit` precomputes cuts for alpha 0.01 to
0.3); until a schema is fitted, only R0 acts, at p >= 0.9, exactly as in Tez's measured policy. Every decision,
undo and escalation is logged with the inputs hash, schema fingerprint, distribution, readout, gate decision,
action and outcome, into `RouterTrainingRow` (whose `taskKind` already has a `voice` value) and Tez's own
feedback file, so `tez fit` can learn from Theioma's traffic on the same schedule as Theioma's other monthly
jobs.

### 4b. The voice loop: word to action in under 50 ms

#### What is measured

**L = t_dispatch - t_end(w\*)**, where w\* is the decisive word (the human-annotated commit word, as in Tez's
`commands.jsonl`), t_end(w\*) is when the last sample of that word left the microphone, and t_dispatch is when the
executor issued the OS or Theioma call. Both are read from one clock (QueryPerformanceCounter): WASAPI stamps
every capture packet with a QPC time, so any sample index maps to a clock time; in replay the harness pushes
10 ms frames in real time and knows each word's end from forced alignment. L can be negative (the action fires
before the word is complete; Tez's live run acted on the partial "open note"). Measuring from the word leaving
the microphone is slightly stricter than "from the word's audio arriving": the difference is the ~5 ms capture
term. Reported per action class: p50, p95, harmful-action rate, false actions on out-of-scope speech, and, for
deferred actions, t_dispatch - t_end(last word).

**How often this budget is the one that binds.** Re-reading Tez's stored 12B trajectories
(`results/voicefast_final_last_gemma4-12b-q8_0_stream.jsonl`, 198 actionable commands with a commit word; the
first word prefix with a non-`none` answer at p >= 0.9):

| First confident decision | Share | Accuracy at that point |
|---|---|---|
| on the decisive word itself | 72.2 % | 0.986 |
| on an earlier word | 19.7 % | 0.667 (mostly refinable opens such as "youtube ..." -> `open_youtube`) |
| 1-5 words later | 6.6 % | 0.923 |
| never | 1.5 % | - |

So for about three commands in four, the latency from the end of the decisive word to the dispatch *is* the
user-visible latency, and it is set by recognition plus decision, not by the model's anticipation. The decisive
word is also the last word in 30 % of commands; early commit still pays there, because it removes the
end-of-utterance wait (600 ms of silence in Tez, ~1,440 ms in Theioma). On average the decisive word is word 2.9
of 5.3, so the action lands about 2.4 words before the user stops speaking.

Targets: **L p50 <= 50 ms and p95 <= 100 ms** for instant and refinable actions on the replay set; <= 60 ms p50
live (a real microphone adds device latency). A p95 under 50 ms is not a realistic target for a recogniser that
must hear a word before it can act on it: the 80 ms encoder frame alone spreads arrival over 0-80 ms.

#### The process layout

The hot path lives in **one native Python process**, `tez-voice`, which owns audio in, VAD, streaming ASR, the
Tez session, the commit policy, the dispatcher and earcon playback. Nothing on the hot path crosses HTTP, a
browser, Node or another CUDA context. Theioma becomes the control plane and the display: it receives events and
handles everything that is not a fast action.

```
 mic --WASAPI shared, 10 ms packets, QPC-stamped--> ring buffer (16 kHz mono, 300 ms pre-roll)
                                                         |
          +------------------ every 10 ms ---------------+------------------------------+
          v                                                                             v
   Silero VAD (CPU, 32 ms windows): speech start/stop, barge-in              every 80 ms chunk (optional
                                                                             20 ms provisional "peek" steps)
                                                                                        v
                                    cache-aware streaming ASR (Nemotron 0.6B, [70,0], same CUDA context)
                                                                                        | append-only token delta
                                                                                        v
                     Tez session: ONE llama_decode per step over all questions (tool, category, "done?")
                                                                                        | calibrated distributions
                                                                                        v
      commit policy: `none` = keep listening | action class | stability | refinement | residual + prior
          |                         |                                   |
          v                         v                                   v
   OS executor              Theioma executor                    earcon / cached clip -> audio out
   (ShellExecuteEx async,   (POST /api/voice/tools/run with     (no synthesis on the hot path)
   SendInput, media keys;   x-theioma-secret: keeps Theioma's
   writes Theioma's voice   audit log, approval contract and
   audit log, obeys its     kill switch; for Theioma-side
   kill switch)             actions, not OS input)
          |
          +--> verifier: Gemma 4 12B letters on the shared llama-server, async, after the fact
          |    (confirms instant actions, undo on disagreement; decides deferred actions at end of utterance)
          v
   events (WS) --> Theioma: transcript, decision, action, undo, utterance end; `none` at end of utterance
                   is handed to /api/voice/converse as a normal conversational turn
```

#### Per-word latency budget

Each row says what today's systems pay and what the design pays. "Est." marks an estimate to be replaced by a
Phase 1 measurement; everything else is measured, with its source.

| Stage | Theioma today | Tez today | Design target (p50) | How |
|---|---|---|---|---|
| Microphone to process | 80 ms PortAudio blocks in the voice sidecar | 200 ms `sounddevice` blocks | ~5 ms (half of a 10 ms packet) | WASAPI shared mode, 10 ms default; `IAudioClient3` allows 2.67 ms at 48 kHz |
| Waiting for the recogniser | the whole utterance plus ~1,440 ms of trailing silence (RMS endpoint) | 0-200 ms (mean ~100) | mean 40 ms at 80 ms chunks; ~10 ms with 20 ms provisional steps | cache-aware model steps every 80 ms; provisional steps run the encoder on the partial chunk without committing its cache |
| ASR compute | faster-whisper `base.en` on the whole utterance (not timed separately); the Parakeet route's `stt_finalize` p50 138 / p95 288 ms; Parakeet's `/stream` re-decodes everything (first partial 731-1,787 ms) | 41 ms p50 / 131 ms p95 per partial, grows with the buffer | est. 5-8 ms per step | only the new 80 ms frame through 24 layers + RNNT greedy; CUDA graphs |
| Token emission lag after the word ends | - | - (re-decode) | unknown; must be <= ~20 ms mean for a 50 ms p50 | decide on sub-word prefixes; FastEmit fine-tune; CTC keyword spotting as a fallback |
| Decision | daemon regex, then in the browser-posted turn: regex shortcuts or `gemma4:12b` tool calling via Ollama (a media command took ~21 s before its shortcut existed); turn complete median 8.1 s | 30 ms compute + 15 ms HTTP (12B Q8) | est. 4-8 ms (cut 4B, probe) or 15-20 ms (12B Q4, letters) | in-process session, one decode per step, logits only at the tail |
| Policy and slots | - | < 1 ms | < 1 ms | unchanged |
| Dispatch issued | PowerShell process per tool inside Next.js (claimed ~150 ms start-up; `app.launch` adds a fixed 800 ms sleep); `tool_execute_ms` p50 300 / p95 1,250 ms | `os.startfile` | <= 2 ms | `ShellExecuteExW` async, `SendInput`, no `pyautogui` default pause |
| **Word end to dispatch** | **~2 s at best, usually several seconds** (no early commit; best full turn 8.2 s) | ~85 ms compute + 0-200 ms wait | **~25-35 ms + emission lag** | |

Reading the last row honestly: with the fast model the computation totals roughly 5 + 10 + 7 + 6 + 2 = ~30 ms,
so **a 50 ms p50 holds only if the recogniser emits the decisive token within ~20 ms of the word's end on
average, or before it**. For long, distinctive words ("spotify", "notepad", "youtube") Tez already fires on
prefixes, which helps; for short final words ("pause", "next") it does not. In Tez's 198 actionable commands
the decisive word's median length is 5 characters and 61 % are 5 characters or shorter ("up", "down", "type",
"notes", "close", "pause"), so anticipation helps a minority and the emission lag sets the p50. The emission lag of the cache-aware
model on this command set is the first thing Phase 1 measures, because it decides whether the target is met by
this design or needs FastEmit fine-tuning or an acoustic keyword spotter.

What runs in parallel, and what does not:

- **Serial per step, in one CUDA context:** ASR step, then the Tez decode. Keeping them in one process avoids
  cross-process GPU time-slicing and a second CUDA context (each costs several hundred MB of VRAM).
- **Overlapped with the next step:** policy, dispatch, event emission and earcon playback run on CPU threads
  while the GPU processes the next chunk.
- **Asynchronous, off the hot path:** the 12B verifier (one forward pass per commit on the shared
  llama-server), slot filling by grammar-constrained generation for deferred actions, Theioma's conversational
  turn for `none` utterances, and all speech synthesis.
- **Paused while the user speaks:** long generations on the shared GPU (Theioma's converse LLM, image
  generation, training jobs). A 12B decode reads the whole model per token and would double the decision time if
  it ran concurrently; the user speaking is exactly when nothing else needs to generate.

#### Worked example: "open notes and type hello world"

Illustrative timings (estimates, not measurements), at a speaking rate where "notes" ends at 800 ms and
"world" at 2,050 ms:

| t (ms) | Designed loop | Theioma today |
|---|---|---|
| ~410 | "open" emitted; Tez says `none` at p ~ 1 (a one-word prefix is not a command), so it keeps listening | listening |
| ~735-815 | "open note" or "open notes" emitted; Tez: `open_notes` at p >= 0.9; R0 commits; `ShellExecuteExW("notepad.exe")` issued; earcon starts; L between -65 and +15 ms | listening |
| ~850 | the 12B verifier agrees (after the fact; on disagreement it would close the window it opened) | listening |
| 820-2,050 | residual "and type hello world" scored with `prior = open_notes`; `type_text` is R2, so it waits | listening |
| ~2,250-2,350 | end of utterance (250 ms of silence + a semantic signal); 12B confirms `type_text`; slot "hello world" by regex; text typed with `SendInput` Unicode events once Notepad's window is focused | trailing-silence counter running |
| ~3,490 and later | (finished at ~2,350) | 1,440 ms of silence ends the utterance at ~3,490; then Whisper; daemon -> browser -> converse; a regex shortcut or the 12B picks a tool; PowerShell `Start-Process` + an 800 ms sleep; typing needs a second tool call or the vision loop. Notepad appears seconds after the sentence ends |

Tez's own live run of this sentence (200 ms chunks, HTTP, contended GPU) launched Notepad at +652 ms from the
start of the audio.

#### Which decision model sits on the hot path

| Option | Readout | Per-step cost | Accuracy evidence | Needs labels | Notes |
|---|---|---|---|---|---|
| F1: Qwen3.5-4B cut to 24 of 32 blocks (`tools/models/Qwen3.5-4B-Q8_0-L24.gguf`, 3.53 GB) | probe on the tail state, blended with its letters | measured 58 ms uncached over HTTP on ~227-token prompts; est. 4-8 ms in-process with a cached prefix | probe 0.793 on typed-decisions (12B letters 0.705); English-trained MASSIVE probe 0.910 at 20 intents, 0.850 at 60 | yes (teacher labels from the 12B are enough to match the teacher) | hybrid (Gated DeltaNet) model: llama.cpp b11100 crashes on partial prefix reuse, so the session must copy state rather than roll back (see below) |
| F2: Gemma 4 12B Q4_K_M | zero-shot letters | est. 15-20 ms in-process (measured 30 ms at Q8) | 0.905 intents at Q8; Q4 vs Q8 on SemIf 0.918 vs 0.943 (p = 0.25); Q4 voice accuracy not measured | no | pure attention with a full sliding-window cache; editable action list with no training |
| F1s: the same cut 4B, **"state only" layout**: a short constant instruction + the transcript, no options and no answer tail; every question's probe reads the hidden state at the transcript's last token | probes only | est. 3-5 ms (only the 1-3 new transcript tokens are computed per step) | Tez's `probe_multiq.py`: state-only 0.748 vs 0.793 with per-question prompts on typed-decisions; untested on voice | yes | append-only, one sequence, no scratch copies: the simplest possible session, and it works for the hybrid model without any state copying |
| F3: small sentence encoder + linear head | head | est. 2-5 ms | none on this task | yes | only if F1 fails |

Recommendation: **bring the loop up on F2** (zero-shot, so the action list can be edited on day one), **move the
hot path to F1s or F1 once 12B teacher labels exist**, and keep the 12B as the verifier. The labels come for free:
Tez's streaming runs already score every word prefix of every utterance with the 12B (1,158 prefix decisions for
the 220 commands), which is exactly the training set a prefix-aware probe needs (prefixes before the commit word
labelled "keep listening", later ones with the intent). Every schema edit changes the prompt fingerprint; Tez
already marks a fitted probe `stale` and falls back to letters in that case, which here means the loop degrades
to F2 speed instead of acting on a probe trained for a different action list.

Where the 12B lives follows from this. **During bring-up (Phases 1-2)** F2 runs in-process in `tez-voice` with a
short full cache and is its own verifier; Theioma's Ollama copy is stopped while testing. **In steady state
(Phase 3 on)** the hot path is F1s/F1 in-process and the 12B exists once, in the shared llama-server, as verifier
and as Theioma's generative local tier (memory plan below).

#### The in-process streaming session (the core Tez change)

Per question q (tool choice, category, "is the command complete?"), the engine keeps two sequences in one
unified KV cache: **A_q** = the question's constant prefix + the transcript so far, and a scratch **T_q**. Each
step:

1. `seq_rm(T_q)`, then `seq_cp(A_q -> T_q)`: T_q now equals A_q (cheap for attention layers: cells gain a sequence id).
2. One `llama_decode` whose batch holds the new transcript tokens tagged {A_q, T_q} and the answer tail
   (closing quote + chat-template tail, ~10 tokens) tagged {T_q} only, for every q. llama.cpp allows one token to
   belong to several sequences when the KV cache is unified.
3. Read the letter logits (or the probe features) at each T_q's last tail token; nothing else requests logits.

With the "state only" layout (F1s) the scratch sequences disappear: one sequence holds the instruction and the
transcript, each step decodes only the new transcript tokens, and every question's probe reads the same final
hidden state.

A_q only ever grows, which fits a cache-aware transducer: its emitted tokens are final, so the transcript is
append-only (unlike Whisper re-decoding, which rewrites earlier words). If the recogniser does revise, A_q is
rebuilt from a frozen prefix-only sequence P_q plus the corrected transcript (~20-30 tokens, one decode). For the
hybrid Qwen3.5 model the recurrent layers cannot roll back, so step 1 must copy the recurrent state
(`seq_cp` for recurrent memory, or `llama_state_seq_get_data` / `set_data`); this must be verified on the pinned
build before F1 is used. After a commit the residual starts a new transcript segment with a `prior` line
(measured to matter: 20/22 compound commands without it, 22/22 with it).

Several questions in one decode cost little while the pass is memory-bound: the weights are read once per step.
For the 12B keep it to two or three questions per step; for the cut 4B, five or six fit comfortably.

#### Tez changes this needs

| # | Change | Why | Done when |
|---|---|---|---|
| T1 | **Persistent in-process engine** (`tez/engines/llamacpp_inproc`: ctypes over the pinned `llama.dll`, unified KV, multi-sequence batches, logits only at tail positions, embeddings at the last token) | removes the ~15 ms HTTP hop, lets one forward pass serve several questions, and keeps the model resident next to the ASR in one CUDA context | per-step decision <= 10 ms p50 (fast model); distributions match the server path (total variation < 0.01) |
| T2 | **Streaming session API** (`tez.stream.Session`: `push(delta)`, `end()`, `reset()`, `prior`, residual cursor, revision rebuild) | the voice loop's unit is a token delta, not a request | replays Tez's 220 commands through the session with the same decisions as `run_voice_fast.py` |
| T3 | **Smaller or cut model with a probe on transcript states** (`tez truncate` exists; add prefix-aware `tez fit` from 12B teacher labels; state-only layout) | 12B letters cost ~30 ms per step; a 24-block 4B probe reads the same decision for a fraction | probe >= the 12B teacher's accuracy on the 220 commands, per-step <= 5 ms |
| T4 | **Streaming contract in the schema** (`actions:` block: class, refines, stable, tau, slots, executor, confirmation) and the class-aware policy moved from `experiments/stream_policy.py` into `tez.stream` | the policy is what keeps harmful actions at 1/198; it must be data, not experiment code | policy unit tests reproduce `stream_policy.py`'s numbers from the stored trajectories |
| T5 | **Large option sets in one step** (category + member questions, or tournament chunks as parallel sequences; many-class probes) | Theioma's voice registry is far above 26 tools | per-step cost with 3 sequences <= 1.5x the single-question cost |
| T6 | **`tez.voice`**: WASAPI capture with timestamps, Silero VAD, cache-aware ASR adapters, endpointing, earcons, native executors, undo, WS events | the hot path has to live in one process | replay harness passes the Phase 2 gates |
| T7 | **Decision log + feedback** in a Theioma-compatible shape | Tez's roadmap item 4; Theioma already has the table (`RouterTrainingRow`) | every commit, undo and escalation is a row with inputs hash, fingerprint, distribution and outcome |
| T8 | **Safe server defaults for acting processes** (loopback, token, no CORS) | `tez serve` is open to every origin today | a test that a cross-origin request is refused |

#### Early-commit rules for Theioma's action classes

Generalises `stream_policy.py`. Each tool in the schema declares its class; the policy reads it.

| Class | Examples | Commits when | Safety net |
|---|---|---|---|
| R0 instant, reversible | open or focus an app, open a Theioma page or panel, media keys, volume, start or stop dictation mode | one step at calibrated p >= tau0 (start at 0.9) | 3 s voice or hotkey undo; the verifier undoes on disagreement |
| R1 refinable | play X on Y, search X, open project X | the ancestor (open the app or page) after two consistent steps; the slot action at end of utterance | as R0 for the ancestor |
| R2 deferred, stateful | type text, send an instruction to an agent, create a task, store a memory | end of utterance only, after the 12B verifier agrees and the slot validates | read-back for long text; `none` or disagreement goes to the conversational path |
| R3 gated | approve or deny, deploy, commit or push, delete, device writes, spending | never automatically | Tez may only open an `ApprovalRequest` carrying the transcript; approval stays Theioma's existing two-keystroke or explicit voice confirmation |

Plus the rules Tez measured: `none` on a prefix means keep listening, never "do nothing"; `none` at the end of the
utterance hands the transcript to the conversational route; after a commit, later words are a new decision with
a `prior`. Tez measured 2 false opens on 22 out-of-scope utterances ("youtube is down again" opens YouTube at
word 1); that is an endpointing problem no model removes, so R0 stays limited to cheap, reversible actions, the
microphone is armed (push-to-talk hotkey or the existing wake word package), and undo is always available.

#### End of utterance, for deferred actions

Tez waits for 600 ms of silence. The design combines three signals and fires on the first two that agree:
Silero silence >= 200-300 ms; a semantic end-of-turn signal (Smart Turn on the audio, or the `<EOU>` token if the
120M Parakeet EOU model is used, whose EOU latency is 160 ms median); and a Tez `noul` question "is the command
complete?" evaluated in the same decode as the tool choice. Target: deferred actions dispatched <= 300 ms after the
last word, against 600 ms plus decode today in Tez and a whole-utterance round trip in Theioma.

#### Speech back, in parallel

- **Instant feedback without synthesis:** an earcon, and per-action confirmation clips ("Opening Notepad")
  synthesised once with Kokoro when the schema is built and cached as audio. Playback of a cached clip through
  WASAPI starts within about one buffer period.
- **Dynamic replies** (conversational turns, results) use streaming Kokoro. Theioma's Kokoro path is not
  streaming today: `/synthesize` renders a whole clause, the browser fetches the WAV, then plays it. Its lane
  measured 1,175 ms TTFT on the GPU and a 683 ms floor for one word, attributed to 241 host-device copy nodes in
  the ONNX graph plus CPU espeak phonemisation. Fixing it means true chunked synthesis (first sentence fragment
  first), keeping the graph on the device, and caching phonemes for fixed phrases; §3 lists what the same model
  achieves elsewhere.
- **Barge-in:** VAD speech-start during playback ducks and then stops TTS. Theioma has no echo cancellation
  (every `voice-aec` adapter is a stub) and no working barge-in, so voice mode needs either a real AEC or a
  headset; earcons are short enough to be masked by timing.
- TTS never waits on, or delays, an action.

#### GPU memory plan (16 GB)

| Component | Process | VRAM |
|---|---|---|
| Nemotron Speech Streaming 0.6B fp16 + CUDA context | `tez-voice` | est. 1.5-2 GB (the 120M EOU model: < 0.5 GB) |
| Fast decision model F1/F1s (Q8, 24 of 32 blocks) + a short full KV cache | `tez-voice` | 3.53 GB + < 0.2 GB (20 blocks: 3.06 GB) |
| Gemma 4 12B Q4_K_M, **normal sliding-window cache**, converse-sized context | shared llama-server (Theioma's reserved :8797) | est. 9-9.5 GB with KV |
| Kokoro-82M | TTS sidecar | est. 0.3-0.5 GB, or 0 on CPU |
| Windows desktop apps | - | est. 0.5-1 GB |
| **Total** | | **est. 15-16 GB of 16.3 GB: at the limit** |

Two consequences. First, the 12B must exist **once**: today Theioma's local tier runs `gemma4:12b` in Ollama
and Tez runs a separate Q8 copy in llama-server (Ollama's model store holds both `gemma4:12b` and
`gemma4:12b-it-q8_0`); the Q8 (~12.5 GB of weights) cannot sit next to the voice path at all. Second, the shared
12B server should **not** use `--swa-full`: the verifier makes one fresh pass per commit (Tez measured ~110 ms for
a fresh 142-token prompt, fine off the hot path), while `--swa-full` would give every layer a full-length cache
at the converse route's 16k context (`num_ctx` 16384), several GB more. Only the small in-process model keeps a
full cache, over a few hundred tokens. If Phase 1 measures more than ~14.5 GB, the levers in order are: the
120M ASR, the 20-block or Q4 fast model, Kokoro on CPU, and a smaller converse context. Other GPU sidecars
(image generation, simulation, training) are paused while voice mode is armed.

Which server hosts the single 12B is a real choice. Theioma's converse client speaks Ollama's native
`/api/chat` (`streamOllama`), while Tez needs llama-server's full log-probabilities (Ollama caps `top_logprobs`
at 20 and its `/v1` endpoint drops them, per Tez's traps list). The clean route is llama-server for both, which
needs an OpenAI-compatible client in the converse route (the multimodal router already speaks
`/v1/chat/completions`). The fallback keeps Ollama as the host and lets the verifier read Ollama's top-20
log-probabilities: good enough to confirm an argmax over 20 or fewer options, not good enough for calibration
work.

### 4c. Packaging: a plug-and-play `tez` voice and router add-on

One Python distribution (`tez-decisions`, already MIT and pip-installable), grown with optional extras, so that
Theioma is one host among several and nothing in the add-on imports Theioma.

```
tez/                          existing runtime: engine, schemas, readouts, fit, gate, truncate, server, CLI
tez/stream/                   Session (append-only transcript, multi-question steps, prior + residual,
                              revision rebuild), action-class policy (from experiments/stream_policy.py),
                              decision log (JSONL; inputs hash, schema fingerprint, distribution, readout,
                              gate decision, action, outcome/undo)
tez/engines/llamacpp_inproc   ctypes binding to a pinned llama.dll (b11100 is already in Tez/tools/llamacpp):
                              unified KV, multi-sequence batches, logits only at tail positions, probe features
tez/voice/                    audio (WASAPI via sounddevice, QPC-stamped), VAD (Silero ONNX), ASR adapters
                              (NeMo cache-aware, sherpa-onnx, Moonshine CPU, faster-whisper fallback),
                              endpointing, earcon/clip player, the loop, WS event server, replay bench
tez/actions/                  executor registry: windows (ShellExecuteEx launch, window focus, SendInput
                              Unicode typing, media keys, clipboard paste), http, mcp (call a tool on any MCP
                              server), python callback; undo stack
tez/mcp/                      `tez mcp`: an MCP server exposing tez_decide (typed question) and tez_route
                              (shortlist + choice over a tool list)
extras: [voice] [voice-nemo] [voice-onnx] [actions-windows] [mcp] [fit] [truncate]
CLI:    tez voice serve --config voice.yaml    tez voice bench --wavs DIR --align align.jsonl
        tez schema from-mcp --server <cmd>     tez mcp --schemas DIR
```

One configuration file drives a host integration:

```yaml
# voice.yaml
audio:    {device: default, packet_ms: 10, sample_rate: 16000, arm: hotkey}   # or wakeword
asr:      {engine: nemo-cache-aware, model: nvidia/nemotron-speech-streaming-en-0.6b, chunk_ms: 80}
decide:
  fast:     {gguf: Qwen3.5-4B-Q8_0-L24.gguf, readout: auto}       # in-process
  verifier: {backend: http://127.0.0.1:8797, template: gemma4}   # shared llama-server
schemas:  [schemas/desktop.yaml, schemas/theioma.generated.yaml]
policy:   {tau: {R0: 0.9, R1: 0.9, R2: 0.95}, stable: {R1: 2}, undo_s: 3}
endpoint: {silence_ms: 250, semantic: tez-noul}
tts:      {earcons: sounds/, clips: cache/clips/, stream: {url: http://127.0.0.1:8796}}
events:   {ws: 127.0.0.1:8818, token_env: THEIOMA_INTERNAL_SECRET}
```

Each action in a schema carries what the loop needs beyond Tez's existing `choice` fields:

```yaml
questions:
  action:
    type: choice
    instructions: Which action is the user asking for?
    criteria:
      open_app:     Open or switch to a desktop application.
      theioma.show_approvals: Show the pending approvals.
actions:                       # new block, ignored by plain Tez clients
  open_app:              {class: R0, slots: {app: lexicon:apps}, run: windows.launch, say: "Opening {app}"}
  theioma.show_approvals: {class: R0, run: http:POST /api/voice/tools/run {tool: approvals.list}}
  type_text:             {class: R2, refines: open_app, slots: {text: regex:after(type|write)}, run: windows.type}
```

**For Theioma**, the add-on arrives as a normal sidecar following the repository's own conventions:
`packages/tez-voice/` with its own `.venv`, `scripts/run-venv.mjs`, `manifest.json` (`healthPath`), a
`SERVICES` entry in `scripts/dev-multi.mjs`, the sidecar registry and port tests updated, on port **8818**
(unclaimed per the port audit). A generator (`apps/web/scripts/gen-tez-schema.ts`) turns the voice tool
registry into `theioma.generated.yaml`, with a parity test in the style of `jpex-tool-name-parity.test.ts` so a
new voice tool cannot land without its Tez entry, and a small TypeScript client (`apps/web/src/lib/tez/`) calls
`/v1/systemone` for one-shot routing.

Coexistence with the current voice stack: only one process may own the microphone, so arming `tez-voice`
disables the voice sidecar's continuous loop (the wake loop can stay there until `tez-voice` takes it over).
The browser keeps connecting to `/__cockpit-proxy/voice` -> the daemon on :8804, and `tez-voice`'s events reach
it through the daemon in one of two ways: the daemon launches `tez-voice` in place of today's Python sidecar and
`tez-voice` speaks the same stdio JSONL protocol (`J\packages\voice\src\sidecar-client.ts`), which is the least
code; or the daemon subscribes to `tez-voice`'s WebSocket on :8818 and re-broadcasts, which keeps `tez-voice`
independent of the daemon, as the generic package needs. Either way the daemon's existing messages
(`voice_state`, `voice_continuous_state`, `voice_continuous_utterance`) keep the widget and modal working
unchanged, and `voice_decision`, `voice_action` and `voice_undo` let the UI show what was heard, decided and
done. Utterances that end in `none` still reach `/api/voice/converse` through the browser exactly as today, so
conversational turns, their TTS and their persistence do not move.

**For any other project:** `pip install "tez-decisions[voice,actions-windows]"`, write `voice.yaml` and a
schema, run `tez voice serve`, and receive events on a WebSocket or Python callbacks. `tez schema from-mcp` builds
a starting schema from any MCP server's tool list, and `tez mcp` lets any MCP host (Claude Code included) ask
typed questions without a voice loop.

Security defaults for anything that can act: bind 127.0.0.1, require a token, and no CORS. Today `tez serve`
answers every origin and grants Private Network Access preflights (`docs/API.md`), which is harmless for a
decision API and unacceptable for a process that launches programs and types text; the voice server must not
inherit that default.

### 4d. Risks

| # | Risk | Why it matters | Mitigation | Detected by |
|---|---|---|---|---|
| 1 | **Transducer emission lag** is unknown on this command set | The compute budget is ~30 ms; the 50 ms p50 only holds if the decisive token arrives within ~20 ms of the word's end on average | sub-word decisions; provisional 20 ms steps; FastEmit fine-tune (`fastemit_lambda`) on command audio; CTC word spotting for app names | Phase 1 forced-alignment measurement (go/no-go) |
| 2 | **VRAM** | 16.3 GB total; 15.6 GB was in use during this audit; Windows apps hold VRAM too | one 12B copy (Q4) shared by Theioma and Tez; ASR and the fast model in one CUDA context; pause image generation, simulation and training sidecars in voice mode; a VRAM guard that refuses to arm voice mode when headroom is short | `nvidia-smi` budget check at arm time |
| 3 | **Accuracy of the fast model and of Q4** on Theioma's larger, domain-specific tool list | Tez's 0.905 is 16 desktop intents on a Q8 12B | the 12B verifier on every commit; per-class conformal thresholds; probe trained on 12B teacher labels; a Theioma command set in Phase 0 | Phase 1 and 2 accuracy gates |
| 4 | **Hybrid-architecture state copy** (Qwen3.5 Gated DeltaNet) in llama.cpp | b11100 crashes on partial prefix reuse for this model | copy state instead of rolling back; if unsupported, a pure-attention small model that passes Tez's symbol-binding screen, or transformers in-process | Phase 1 spike |
| 5 | **NeMo on Windows** and dependency weight | NeMo is officially Linux; Theioma already fought a CPU-only torch wheel (D3 F-1) and a TLS-intercepting proxy (F-3) | keep the pinned cu128 torch and `truststore`; the torch-free path (sherpa-onnx or ONNX Runtime GenAI export of the same model, which Microsoft shipped in Foundry Local) as plan B | Phase 1 install on a clean venv |
| 6 | **False actions** on out-of-scope speech, and self-triggering from TTS | Tez: 2/22 out-of-scope utterances opened an app; without echo cancellation the recogniser hears the assistant | only R0 commits early; arming by hotkey or wake word; undo; `voice-aec` or a headset; earcons instead of speech for instant feedback | harmful and out-of-scope rates per build |
| 7 | **Theioma's current execution paths are slow and partly ungated** | voice tools spawn PowerShell per call inside Next (blocking its event loop; `app.launch` sleeps 800 ms); MCP `computer_use_*` hard-code `approved: true` (`packages/channel/src/tools/computer_use/click.ts:16`, `type.ts:14`); `desktop-driver` binds 0.0.0.0 with no auth (`src/main.py:127`); the kill switch's abort flag has no non-test reset | native executors in `tez-voice` that write Theioma's voice audit log and honour the same kill switch; R3 only via `ApprovalRequest`; fix the three gate defects independently of this project | code review + a gate test per executor |
| 8 | **Two sources of truth for tools** | Theioma's registry changes weekly (every jpex tool lands with a voice twin) | generated schema + parity test; schema fingerprint makes stale probes fall back to letters | CI |
| 9 | **Another voice path in an already crowded system** | the daemon regex router, the widget, converse fast paths and the agent loop overlap today | `tez-voice` behind a feature flag; retire the sidecar's Whisper STT and the daemon's regex router only after Phase 3 gates pass | Phase 3 exit review |
| 10 | **Claims outrunning measurements** | Theioma's own audits found a 33x gap between a documented and a measured latency | every milestone is a replay-harness number with a manifest (data hash, model path, settings); live-microphone numbers reported separately | manifests in `results/` |
| 11 | **Windows focus and integrity rules** | a background process may not be allowed to call `SetForegroundWindow`, so "open notes and type X" can type into the wrong window; `SendInput` into an elevated window is silently dropped (UIPI) | type only after a focus event from the launched window (or set text through a cached UIA `ValuePattern` on the target element); verify the foreground window before every `SendInput`; run `tez-voice` at the same integrity level as its targets; treat a focus failure as an escalation, not a retry loop | executor tests that launch, focus and type into Notepad and an elevated window |

### 4e. Phased plan with measurable milestones

GPU phases start only after the current measurements finish.

| Phase | Work | Exit criteria (all measured on the word-aligned replay sets) |
|---|---|---|
| **0. Harness and baseline** (2-3 days, CPU only) | Forced-align Tez's 220 WAVs; build a Theioma command set (~250 utterances over its voice tools: every action class, out-of-scope speech, compounds; 50+ recorded on the real microphone, the rest synthesised with several Kokoro voices); a replay harness that pushes 10 ms frames in real time and logs QPC-stamped spans; measure today's Theioma path on it. | A manifest with today's word-end-to-execution latency, intent accuracy and harmful rate for Theioma's current path and for Tez's experiment loop, on the same rows. |
| **1. Component spikes** (1 week, GPU) | Nemotron Speech Streaming 0.6B at 80 and 160 ms (step compute, emission lag vs alignment, WER, ASR-vs-text intent agreement); in-process session over `llama.dll` with Gemma 4 12B Q4_K_M and Q8_0 letters and the cut 4B probe (per-step latency, parity with the HTTP path, accuracy at Q4); VRAM of the combined stack. | ASR step <= 8 ms p50; emission lag known (**go** for the 50 ms target if mean <= 20 ms, otherwise FastEmit or keyword spotting moves into Phase 2); decision <= 10 ms p50 on the fast model, <= 20 ms on the 12B Q4; intent accuracy >= 0.90 on Tez's set at Q4; total VRAM <= 14.5 GB. |
| **2. `tez.stream` + `tez.voice`** (1-2 weeks) | Session API, action classes in the schema, native Windows executors, earcons and cached clips, WS events, endpointing (silence + semantic + Tez `noul`), async verifier, undo. | Replay L p50 <= 50 ms and p95 <= 100 ms for R0/R1; deferred actions p50 <= 300 ms after the last word; harmful <= 1/198 on Tez's set and <= 1 % on Theioma's; out-of-scope false actions <= 2/22 with undo; parity of in-process vs server distributions (total variation < 0.01). |
| **3. Theioma integration** (1-2 weeks) | `packages/tez-voice` on 8818 with the repository's sidecar conventions; generated schema + parity test; voice widget shows live transcript, decision and action from `tez-voice` events; `none` utterances go to `/api/voice/converse`; R3 goes to `ApprovalRequest`; one-shot routing client in converse; one shared 12B on llama-server at :8797 (the port CLAUDE.md reserves; `packages/llama-server/manifest.json` says 8080 and must be reconciled with the port-registry test), with Theioma's local tier pointed at it instead of Ollama. | Live-microphone L p50 <= 60 ms; converse tool selection <= 150 ms p95 whenever Tez acts (today a media command took ~21 s through the 12B with 14 tool schemas); escalation rate and accuracy against the LLM path on a labelled routing set; existing voice tests green. |
| **4. Learning loop and hardening** (ongoing) | Decision log into Theioma's router training rows; monthly `tez fit`; per-class conformal alphas; probe hot path; FastEmit fine-tune if Phase 1 said so; Kokoro TTFT fix; barge-in with echo cancellation; retire the duplicate voice paths. | Harmful <= 0.5 %; escalations <= 10 % of commands; TTS first audio <= 200 ms; no regression in L across releases (CI replay). |

---

## 5. Sources

Local (read for this audit): Theioma `CLAUDE.md`, `docs/audit/2026-09-06-voice-e2e.md` (lanes D3 and D3b),
`packages/VOICE_SIDECARS.md`, the files cited inline in §1; Tez `README.md`, `BENCHMARKS.md` §1, §3, §4b-§4c,
`docs/REPORT.md` §3.4, §4.4-§4.6, §6, `docs/API.md`, `tez/*.py`, `experiments/*.py` cited in §2, and
`results/*.summary.json`.

Streaming ASR and audio (web, dates as published):

- NVIDIA Nemotron Speech Streaming En 0.6B model card (HF release 2026-03-13; WER by chunk size):
  https://huggingface.co/nvidia/nemotron-speech-streaming-en-0.6b
- NVIDIA, "Scaling Real-Time Voice Agents with Cache-Aware Streaming ASR" (2026-01-05; 24 ms median time to
  final on H100, 560 concurrent streams at 320 ms): https://huggingface.co/blog/nvidia/nemotron-speech-asr-scaling-voice-agents
- NVIDIA Nemotron 3.5 ASR Streaming 0.6B model card (2026-06-04):
  https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b
- NVIDIA parakeet_realtime_eou_120m-v1 model card: https://huggingface.co/nvidia/parakeet_realtime_eou_120m-v1
- NVIDIA parakeet-unified-en-0.6b model card (2026-04-07): https://huggingface.co/nvidia/parakeet-unified-en-0.6b
- Banfic et al. (Microsoft CoreAI), "Pushing the Limits of On-Device Streaming ASR", arXiv 2604.14493 (April 2026):
  https://arxiv.org/html/2604.14493v1
- Moonshine v2, arXiv 2602.12241 (2026-02-12): https://arxiv.org/html/2602.12241v1 ; code and licence:
  https://github.com/moonshine-ai/moonshine
- Kyutai STT (June 2025): https://kyutai.org/stt/ , https://huggingface.co/kyutai/stt-1b-en_fr
- Mistral Voxtral Mini 4B Realtime 2602 (2026-02-04): https://huggingface.co/mistralai/Voxtral-Mini-4B-Realtime-2602 ,
  https://mistral.ai/news/voxtral-transcribe-2/
- WhisperLiveKit / SimulStreaming (2025): https://github.com/QuentinFuxa/WhisperLiveKit , https://github.com/ufal/SimulStreaming
- sherpa-onnx streaming models: https://k2-fsa.github.io/sherpa/onnx/pretrained_models/online-transducer/zipformer-transducer-models.html ,
  https://huggingface.co/csukuangfj/sherpa-onnx-nemo-streaming-fast-conformer-transducer-en-480ms-int8
- Vosk on latency (2020-11-27): https://alphacephei.com/nsh/2020/11/27/latency.html
- FastEmit, arXiv 2010.11148: https://arxiv.org/abs/2010.11148 ; NeMo cache-aware streaming config with
  `fastemit_lambda`: https://github.com/NVIDIA-NeMo/NeMo/blob/main/examples/asr/conf/fastconformer/hybrid_cache_aware_streaming/fastconformer_hybrid_transducer_ctc_bpe_streaming.yaml
- Contextual biasing for streaming ASR via CTC-based word spotting, arXiv 2605.18222 (May 2026): https://arxiv.org/pdf/2605.18222
- Microsoft, Low Latency Audio (WASAPI `IAudioClient3`, 10 ms default shared buffers):
  https://learn.microsoft.com/en-us/windows-hardware/drivers/audio/low-latency-audio
- RTX 5080 Laptop GPU specifications (16 GB GDDR7, 256-bit, 896 GB/s): https://videocardz.net/nvidia-geforce-rtx-5080-laptop-gpu
- llama.cpp batching, tokens in several sequences, unified KV: https://github.com/ggml-org/llama.cpp/discussions/4130 ,
  https://github.com/ggml-org/llama.cpp/discussions/17421

VAD and end of turn:

- Silero VAD (v6.2.3, 2026-09-23; releases; performance and quality wikis; `utils_vad.py` defaults):
  https://github.com/snakers4/silero-vad , https://github.com/snakers4/silero-vad/wiki/Performance-Metrics ,
  https://github.com/snakers4/silero-vad/wiki/Quality-Metrics
- TEN VAD (v1.0, 2025-07-11) and licence: https://github.com/TEN-framework/ten-vad
- WebRTC VAD wheels (2.0.14, 2024-09-05): https://pypi.org/project/webrtcvad-wheels/ ; independent VAD timings
  (wavekat-vad v0.1.17, 2026-08-30): https://github.com/wavekat/wavekat-vad
- Pipecat Smart Turn (v3.2, 2026-01-07; v3.1 2025-12-03; v3 2025-09-11): https://github.com/pipecat-ai/smart-turn ,
  https://www.daily.co/blog/improved-accuracy-in-smart-turn-v3-1/ ,
  https://www.daily.co/blog/smart-turn-v3-2-handling-noisy-environments-and-short-responses/
- Pipecat VAD and turn-detection defaults: https://docs.pipecat.ai/server/utilities/audio/silero-vad-analyzer
- LiveKit turn detector docs, blog (2026-06-17) and licence: https://docs.livekit.io/agents/build/turns/turn-detector/ ,
  https://livekit.com/blog/solving-end-of-turn-detection , https://huggingface.co/livekit/turn-detector/blob/main/LICENSE
- TurnSense 1.1 (2026-05-22): https://huggingface.co/brgroup/TurnSense ; Namo v1:
  https://huggingface.co/videosdk-live/Namo-Turn-Detector-v1-Multilingual

TTS:

- tts-bench speed page and repository (Windows RTX 5090 and CPU rigs; last commit 2026-09-22; figures re-checked
  2026-09-24): https://5uck1ess.github.io/tts-bench/speed.html , https://github.com/5uck1ess/tts-bench
- Kokoro-82M (v1.0, 2025-01-27): https://huggingface.co/hexgrad/Kokoro-82M ; Kokoro-FastAPI:
  https://github.com/remsky/Kokoro-FastAPI ; kokoro-onnx: https://github.com/thewh1teagle/kokoro-onnx
- Piper (MIT repo archived 2025-10-06; piper-tts 1.8.0, 2026-09-04, GPL-3.0): https://github.com/rhasspy/piper ,
  https://pypi.org/project/piper-tts/
- Kyutai Pocket TTS (Jan 2026): https://github.com/kyutai-labs/pocket-tts ; Kyutai TTS 1.6B:
  https://huggingface.co/kyutai/tts-1.6b-en_fr , https://github.com/kyutai-labs/delayed-streams-modeling
- Sopro V2 (2026-08-27): https://github.com/samuel-vitorino/sopro ; Chatterbox-Turbo:
  https://huggingface.co/ResembleAI/chatterbox-turbo ; VibeVoice-Realtime-0.5B (2025-12-03):
  https://github.com/microsoft/VibeVoice ; Qwen3-TTS (2026-01-22): https://github.com/QwenLM/Qwen3-TTS
- PyTorch 2.7 Blackwell support: https://pytorch.org/blog/pytorch-2-7/ ; onnxruntime-gpu on RTX 5090 (issue,
  2025-09-26): https://github.com/microsoft/onnxruntime/issues/26177 ; NVIDIA laptop and desktop specifications:
  https://www.nvidia.com/en-us/geforce/laptops/50-series/
- LiveKit background audio: https://docs.livekit.io/agents/multimodality/audio/background-audio/ ; Nielsen
  response-time limits: https://www.nngroup.com/articles/response-times-3-important-limits/

Computer-use execution:

- `SendInput` (updated 2025-07-01): https://learn.microsoft.com/en-us/windows/win32/api/winuser/nf-winuser-sendinput ;
  `SetForegroundWindow` (2025-10-06): https://learn.microsoft.com/en-us/windows/win32/api/winuser/nf-winuser-setforegroundwindow
- pyautogui `PAUSE` default: https://github.com/asweigart/pyautogui/blob/master/pyautogui/__init__.py ; pynput:
  https://pynput.readthedocs.io/en/latest/keyboard.html
- UI Automation caching (2025-07-14): https://learn.microsoft.com/en-us/windows/win32/winauto/uiauto-cachingforclients ;
  `IInvokeProvider::Invoke` blocking note: https://learn.microsoft.com/en-us/windows/win32/api/uiautomationcore/nf-uiautomationcore-iinvokeprovider-invoke ;
  pywinauto UIA-backend slowness: https://github.com/pywinauto/pywinauto/issues/842
- Windows Terminal launch timings (2024-08-01; 2026-02-12): https://github.com/microsoft/terminal/issues/17642 ,
  https://github.com/microsoft/terminal/issues/19859 ; Notepad release notes (2026-07-24):
  https://learn.microsoft.com/en-us/windows-insider/release-notes/apps/notepad ; Low Latency Profile (secondary,
  2026-08-12): https://www.windowslatest.com/2026/08/12/windows-11s-faster-app-launches-released-today-enable-it-using-these-steps/
- DXcam: https://github.com/ra1nty/DXcam ; OmniParser V2 (2025-02-12): https://huggingface.co/microsoft/OmniParser-v2.0 ;
  UI-TARS-2, arXiv 2509.02544: https://arxiv.org/abs/2509.02544 ; OSWorld-Human, arXiv 2506.16042 (v2 2026-05-18):
  https://arxiv.org/abs/2506.16042
- Anthropic computer-use tool docs: https://platform.claude.com/docs/en/docs/agents-and-tools/tool-use/computer-use-tool ;
  OpenAI computer-use guide: https://developers.openai.com/api/docs/guides/tools-computer-use ; MCP on Windows
  (updated 2026-06-04): https://learn.microsoft.com/en-us/windows/ai/mcp/overview

Prior art (from Tez's `docs/research/RESEARCH-NOTES.md` §3): Picovoice Rhino https://picovoice.ai/docs/faq/rhino/ ;
Snips NLU https://arxiv.org/abs/1805.10190 ; BC-ResNet https://arxiv.org/abs/2106.04140 ; Soniox on endpointing
https://soniox.com/wiki/endpoint-detection ; Google streaming intended-query detection https://arxiv.org/abs/2208.13322 .
