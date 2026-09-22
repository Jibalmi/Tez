# Tez

**An open, local "System One" decision layer.** Ask a typed question — which option, yes/no, what
level — and get a probability distribution over the answers from **one forward pass** of a local
model, with a calibrated confidence and an explicit abstain/escalate contract. No tokens are
generated. Think of it as an open, self-hosted alternative to TypeSafe AI's Jev, built on the
published technique rather than on Jev's outputs (which its terms forbid).

> Status: **research, experiments and a working voice→action loop in `experiments/`;
> no library API yet.** Read [`docs/REPORT.md`](docs/REPORT.md) first — it was audited before
> release and §3.7 lists what the first draft got wrong.

![Tez dashboard — every measurement on one page](docs/figures/tez_dashboard.png)

Full tables: [`BENCHMARKS.md`](BENCHMARKS.md). Individual figures: `docs/figures/`.

## What we measured

Frozen models, zero training, SemIf's exact prompt and public fixtures (144 three-option decisions),
RTX 5080 laptop. Mean-family balanced accuracy, classes keyed by option id, 95 % CI by source-group
bootstrap:

| Model | Authored | Perturbation | NLL (raw → best) | Reversal flips / 36 | p50 fresh |
|---|---:|---:|---:|---:|---:|
| **Gemma 4 12B Q8_0** | **0.943** (0.897–0.981) | **0.992** | 0.478 → **0.158** | **0** (≤ 9.6 %) | 110 ms |
| Gemma 4 12B Q4_K_M | 0.918 (0.871–0.958) | 0.981 | 0.549 → 0.184 | 1 | 100 ms |
| Qwen3.5-4B BF16 (SemIf's model, reproduced in-process) | 0.813 | — | 0.427 | — | 224 ms* |
| Qwen3.5-4B + 6-permutation average, one batch | **0.912** | — | — | — | 321 ms* |
| Gemma 3 4B Q4_K_M | 0.646 | 0.729 | 3.433 → 0.696 | 13 (36 %) | 40 ms |
| Llama 3.2 3B Q4_K_M | 0.358 (≈ chance) | 0.496 | 1.673 → 1.043 | 10 (28 %) | 25 ms |
| SemIf published: Qwen3.5-4B / EXL3 27B | 0.813 / 0.958 | 0.766 / — | — | — | — |

\* transformers, not llama.cpp. Best calibration recipe: **six-ordering permutation average +
out-of-fold temperature** (Q8: NLL 0.158, Brier 0.075, error-detection AUROC 0.906). Paired tests:
12B vs SemIf's 4B p < 10⁻⁴; 12B vs the 4B *with* permutation averaging p = 0.18 — roughly half the
"scale" gap is removable option-order bias. Q4 vs Q8: p = 0.25.

**Voice → action, built and measured** (220 spoken commands, 16 actions, Gemma 4 12B Q8_0):
intent accuracy **0.905** (0.982 accepting the second action of compound commands); out-of-scope
recall 1.00 / precision 0.92; **30 ms compute / 45 ms round-trip per streamed word** because the
action list is a cached prefix and only the transcript is re-evaluated (`--swa-full` is what makes
llama-server actually reuse it); class-aware early commit: **1 harmful action in 198 utterances**,
Notepad open at +652 ms into "open notes and type hello world" from audio; compound commands scored
as two decisions on the residual: 22/22. ASR (faster-whisper base.en): 7.8 % WER on synthetic
speech, 97 % intent agreement with the true text.

**Head-to-head with Laya** (the open encoder-based "System One" that claims to beat Jev), its three
checkpoints and Tez on byte-identical rows, same GPU, Laya's own datasets and protocol
(`experiments/bench_h2h.py`, figures in `docs/figures/`):

| task | **Tez (zero-shot)** | laya | laya-multilingual | laya-typed-decisions | Jev (published) |
|---|---:|---:|---:|---:|---:|
| typed-decisions, 2,000 decisions (zero-shot / 4 examples in the cached prefix) | **0.704 / 0.725** | 0.362 | 0.352 | 0.766 (fine-tuned on its train split) | 0.727 |
| … soft accuracy vs teacher distribution | **0.575 / 0.583** | 0.331 | 0.328 | 0.471 | 0.580 |
| Banking77 (77 options) | **0.713** | 0.395 | 0.357 | 0.388 | 0.870 |
| MASSIVE intent, macro over 11 languages | **0.885** | 0.354 | 0.524 | 0.345 | — |
| … Khmer | **0.790** | 0.000 | 0.210 | 0.050 | — |
| SST-5 / BoolQ / prompt-injections | **0.512 / 0.850 / 0.759** | 0.362 / 0.843 / 0.672 | 0.280 / 0.782 / 0.569 | 0.460 / 0.835 / 0.647 | — |
| AG News / XNLI-en (in Laya's training mix) | 0.880 / 0.730 | **0.932** / 0.900 | 0.948 / 0.860 | 0.932 / **0.920** | 0.910 / — |
| order flip at 20 options | **0.07** | 0.18 | 0.20 | 0.15 | 0.13 |
| ECE as shipped → per-task refit (mean) | 0.212 → 0.078 | 0.323 → 0.071 | 0.253 → 0.103 | 0.226 → 0.068 | 0.246 |
| ms per decision (p50, this GPU) | 57–250 | 25–50 | 23–35 | 36–52 | 236–276 |

Laya's published numbers reproduce in our harness (its base 0.362, fine-tuned 0.766, Khmer 0.000),
so the comparison is controlled. Zero-shot Tez wins six of nine English tasks and every language;
Laya wins where it was trained (AG News, NLI) and on single-question latency.

Findings worth knowing before you build anything like this:

- **Symbol-logit decisions are scale-gated.** A 3B model is at chance with 28 % order sensitivity
  and no calibration trick recovers it; a 4B sits at 0.646 with a third of decisions flipping under
  option reversal. Screen the backbone for symbol binding first.
- **Permutation averaging is the cheapest accuracy in the stack**: +10 points on Qwen3.5-4B, +7 on
  Gemma 3 4B, in one batched forward pass.
- **Zhao et al. contextual calibration is harmful here** (0.951 → 0.778, and averaging three
  placeholders does not rescue it): with the evidence blanked the model *correctly* answers
  "insufficient", so dividing that out deletes real answers. Never apply it to option sets that
  contain an abstain-style answer.
- **Prompt layout is a systems decision**: constant material first, variable last. Transcript-last
  cut per-word latency 7× *and* raised accuracy (0.868 → 0.905).
- **A confident `none` on a partial transcript means "keep listening", never "do nothing"** — the one
  rule that makes early commit work.

## Layout

```
docs/REPORT.md                   the write-up: mechanism, papers, experiments, audit, voice loop, roadmap
docs/research/RESEARCH-NOTES.md  the seven commissioned literature surveys, with links
experiments/run_direct.py        symbol-logit readout via llama-server, SemIf-compatible prompt
experiments/run_hf.py            in-process transformers readout: fresh / prefix reuse / batched permutations
experiments/metrics.py           balanced accuracy (by option id) + CIs, NLL/Brier/ECE, paired perturbation stability
experiments/calibrate.py         temperature (OOF), contextual calibration, permutation averaging, risk–coverage, AURC, AUROC
experiments/perm_analysis.py     all six orderings: 2/6-perm averaging, PriDe prior, Zhao 3-placeholder + damping
experiments/run_voice.py         voice intents with SemIf's payload (transcript first) — the slow control
experiments/run_voice_fast.py    transcript-LAST prompt with KV prefix reuse; full + word-by-word streaming
experiments/stream_analysis.py   commit policies on streaming trajectories (naive vs non-none rule)
experiments/stream_policy.py     class-aware commit/refine policy: harmful-action rate, time to first action
experiments/residual_check.py    compound commands as two decisions on the residual (+ prior)
experiments/voice_actions.py     deterministic slot extraction + OS executors (dry-run by default)
experiments/asr_bench.py         faster-whisper partial latency, WER, ASR-vs-text intent agreement
experiments/tts_wavs.py          Windows SAPI synthesis of the 220 commands to 16 kHz WAV
experiments/voice_demo.py        live loop: --text / --wav / --mic → ASR partials → decision → commit → action
experiments/bench_h2h.py         Laya's public benchmarks (AG News, Emotion, Banking77, SST-5, BoolQ, prompt-injections,
                                 MASSIVE ×11 languages, XNLI ×10, typed-decisions) — same rows through Tez and the Laya checkpoints
experiments/make_h2h_plots.py    the comparison figures (docs/figures/)
experiments/make_dashboard.py    the all-in-one dashboard (docs/figures/tez_dashboard.png)
experiments/run_h2h_all.ps1      detached runner for the whole head-to-head chain
experiments/bench_jevbench.py    JevBench public tiers (data/jevbench/) with the leaderboard's intelligence/speed formulas
experiments/conformal_td.py      Mondrian split-conformal act/escalate sets on typed-decisions
experiments/batch_calibration.py Batch Calibration (Zhou 2023) applied offline, out-of-fold
experiments/cascade_analysis.py  small→large cascade (negative result)
experiments/fewshot_td.py        k worked examples in the cached prefix (zero training)
experiments/think_td.py          "System 1.5": a seeded thinking budget before the readout
experiments/backend.py           llama-server / Ollama backends (TEZ_BACKEND, TEZ_TEMPLATE, TEZ_OLLAMA_MODEL)
docs/research/*-survey.md        commissioned literature surveys: speed, accuracy, models & landscape
BENCHMARKS.md                    every number, consolidated
experiments/probe_gemma.py       template / tokenizer / logit probe for a GGUF
data/semif/                      SemIf's MIT fixtures and core.py (prompt construction), vendored
data/voice/                      16 intents, 220 labelled commands (styles, slots, commit word), synthesised wavs
results/                         run outputs + manifests (data SHA-256, model path, settings, latency)
tools/                           llama.cpp release binaries and downloaded GGUFs (not committed)
```

## Reproduce

Needs a llama.cpp release build (b11100 `win-cuda-13.4` was used) and a Gemma 4 12B GGUF.

```
llama-server.exe -m <gguf> -ngl 99 -c 2048 -b 512 --port 8091 --host 127.0.0.1 -np 1 --no-webui --swa-full
curl http://127.0.0.1:8091/props        # assert model_path before every run — see REPORT §3.6

py experiments/run_direct.py --data data/semif/authored144.jsonl --out results/a.jsonl --template gemma4
py experiments/run_direct.py --data data/semif/authored144.jsonl --out results/a_rev.jsonl --reverse
py experiments/run_direct.py --data data/semif/authored144.jsonl --out results/a_cf.jsonl --content-free
py experiments/run_direct.py --data data/semif/perturbations108.jsonl --out results/p.jsonl
py experiments/metrics.py   --pred results/a.jsonl
py experiments/metrics.py   --pred results/p.jsonl --base results/a.jsonl
py experiments/calibrate.py --pred results/a.jsonl --cf results/a_cf.jsonl --rev results/a_rev.jsonl

py experiments/run_voice_fast.py --variant final --order last --out results/vf.jsonl
py experiments/stream_policy.py  --stream results/vf_stream.jsonl --full results/vf.jsonl
py experiments/voice_demo.py     --text "open notes and type hello world"     # dry run; --execute to act
```

Traps we hit (details in the report §3.6): prebuilt `llama-cpp-python` wheels use AVX-512; Ollama
caps `top_logprobs` at 20 and its `/v1` endpoint drops logprobs; Gemma 4 is a thinking model and
needs the empty thought channel in the prompt; bare and space-prefixed letters are different tokens;
`taskkill /IM` silently fails under Git Bash; **sliding-window models silently re-evaluate the whole
prompt unless `--swa-full` is set — check `timings.prompt_n`**; CTranslate2 needs the CUDA 12 DLLs
registered with `os.add_dll_directory` next to a CUDA 13 torch.

## Roadmap (short)

1. Backbone by MCSB screen (Gemma 4 12B passes; Q8_0 for calibration-grade probabilities; a 4B with
   batched permutation averaging is the budget option).
2. Six-ordering permutation averaging in one batch; per-schema temperature; Mondrian conformal →
   act / escalate / fall back.
3. Prompt layout as part of the schema (constant first, variable last, `prior` for multi-step);
   prefix reuse verified by `prompt_n`.
4. Decision log + shadow-eval loop; sample auto-acted cases to humans too.
5. Jev-compatible wire format plus `multi`, implicit `__none__`, `p_correct` with calibration id,
   and a streaming/early-commit contract with action classes.
6. Optional, last: LoRA on Gemma 4 12B *base*, proper scoring rule, gated on an OOD hold-out.

## Licence

MIT (see `LICENSE`). Vendored SemIf material is MIT (TheoLeeCJ). Gemma 4 is Apache-2.0.
faster-whisper/CTranslate2 and Whisper weights are MIT. Not affiliated with TypeSafe AI; nothing here
was trained on or derived from Jev outputs.
