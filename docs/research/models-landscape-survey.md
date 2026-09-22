# Survey: backbones, encoder alternatives and the open "System One" landscape (22 Sept 2026)

Commissioned survey; every HF repo returned 200 from the Hub API, every arXiv id resolved by title. Excluded as unverifiable: "Phi-5" (401 / not in Microsoft's listing), Qwen3.7 (no open weights). Llama 4 has nothing under 109B and a non-clean licence; DeepSeek nothing under 284B.

## 1. Decoder backbones that fit 16 GB (2026)

| Model | Type | Ctx | Licence | llama.cpp | MMLU-Pro / GPQA | Fits 16 GB | Verdict |
|---|---|---|---|---|---|---|---|
| **Gemma 4 12B** (`google/gemma-4-12B-it`) — current | dense 12B | 256K | Apache-2.0 | yes | 77.2 / 78.8 | Q8 ~12.5 GB | baseline |
| **Gemma 4 26B-A4B** (`google/gemma-4-26B-A4B-it`) | MoE 25B / 3.8B active | 256K | Apache-2.0 | yes; official `-qat-q4_0-gguf` | 82.6 / 82.3 | Q4 ≈ 15 GB (small ctx) | **top candidate**: +5.4 MMLU-Pro at ~4B active compute |
| **Qwen3.5-9B** (`Qwen/Qwen3.5-9B`) | dense hybrid (Gated DeltaNet) | 262K | Apache-2.0 | yes | 82.5 / 81.7 | Q8 ~10 GB | **top candidate** |
| Qwen3.5-4B | dense | 262K | Apache-2.0 | yes | 79.1 / 76.2 | yes | SemIf's backbone |
| **Qwen3.6-35B-A3B** | MoE 35B / 3B | 262K | Apache-2.0 | yes | 85.2 / 86.0 | Q4 ≈ 19–21 GB → expert offload | strong if offload latency acceptable |
| **Qwen3.6-27B** / **Qwen3.8-27B** | dense | 262K | Apache-2.0 | yes | 86.2 / 87.8; Qwen3.8 on JevBench frozen readout intel 84.7–85.8, calib 81–86 | Q3 only | best zero-shot decision intelligence of any open model; offline teacher |
| GLM-4.7-Flash (MIT) | MoE 30B / 3B | 128K | MIT | yes | – / 75.2 | Q4 ≈ 17–18 GB → offload | weaker than Qwen3.6-35B-A3B |
| Granite 4.2-8B / 30B | dense | 128K | Apache-2.0 | yes | 74.0 / 64.1; 77.6 / 66.4 | 8B yes | not better than Gemma 12B / Qwen 9B |
| Gemma 4 E4B | 8B total / 4.5B eff. | 128K | Apache-2.0 | yes | 69.4 / 58.6 | yes | cheap tier only |
| Gemma 4 31B | dense | 256K | Apache-2.0 | yes | 85.2 / 84.3 | Q4 ≈ 18 GB | out |
| MiniCPM5-2B | dense 2.5B | 128K | Apache-2.0 | yes | 34-bench avg 53.9 | yes | sub-4B → below the binding cliff |
| DiffusionGemma 26B-A4B | discrete-diffusion MoE | – | Apache-2.0 | **no** (vLLM PR only) | – | ~18 GB | see §4a |

Licence flags (NOT Apache/MIT-clean): Llama 4 Community; GLM-5.3 full; T5Gemma 2 and EmbeddingGemma are `license:gemma`, gated (unlike Gemma 4 proper).

## 2. Encoder / non-autoregressive alternatives

| Model | Params | Licence | Numbers |
|---|---|---|---|
| **Laya** (`convaiinnovations/laya`) | 421M / 322M | Apache-2.0 | typed-decisions 0.766, Brier 0.062; Banking77 0.425; JevBench v1.3 #33, score 54.4, intelligence 45.8 |
| **Von** (`wfzyx/von`) — ModernBERT-Large, CE+Brier, ~290k NLI examples | 395M | Apache-2.0 | own 49-task suite 72.0 % macro (GLiNER2 68.4, Laya 58.3); < 18 ms; not on JevBench |
| **openJev-verdict 2.0** (`heman10x/rlcd-modernbert-151m`) | 150M | Apache-2.0 | typed-decisions test 0.771, ECE 0.014 — but JevBench #36/37 (score ~38): overfit to that split |
| OpenDecision (ModernBERT-large NLI) | ~400M | Apache-2.0 | JevBench #35, 40.6 |
| **GLiNER2.5** (`fastino/gliner2.5-*`) | 74M–0.3B | Apache-2.0 | +24.75 XNLI over GLiNER2; GLiNER2-large 29.6 on JevBench |
| GLiClass (`knowledgator/gliclass-*`) | 0.3–0.4B | Apache-2.0 (paper CC-BY-NC-ND) | large-v3.0 avg 0.719 over 14 zero-shot sets; Banking77 0.557 |
| ModernBERT-Large-Instruct (2502.03793) | 395M | Apache-2.0 | 93 % of Llama3-1B MMLU |
| Ettin encoders (2507.11412) | 17M–1B | MIT | largest modern open encoder; no decision head |
| mmBERT (2509.06888) | 140M / 307M | MIT | XNLI 77.1 |
| Rerankers as deciders: zerank-2, Qwen3-Reranker-4B | – | – | JevBench 66.0 / 63.8 — beat every encoder-head project |

BTZSC (2603.11991, 22 datasets): rerankers lead (Qwen3-Reranker-8B macro-F1 0.72), GTE-large best accuracy/latency; 4–12B instruct LLMs ≤ 0.67; NLI cross-encoders plateau. On JevBench every encoder-head system collapses on the 220-item hard tier.

## 3. Open "System One" projects on JevBench v1.3.0 (21 Sept 2026, 534 decisions, 41 systems)

| Rank / score | Project | Method | Intel / Calib |
|---|---|---|---|
| #1 74.4 | Jev 1.13 (closed) | – | 85.7 / 82.7 |
| #2 73.1 | SemIf (Qwen3.5-4B frozen readout) | – | 79.0 / 72.6 |
| #3 73.0 | djev (DiffusionGemma 26B-A4B, pinned answer slots, read-only logprobs) | vLLM only, ~18 GB | 82.7 / 65.4 (thinking variant 80.8 / **92.7**) |
| #4 71.2 | Winnow-12B Q8 (private corpus; likely Gemma 4 12B Q8 GGUF) | /v1/systemone server | 82.0 / 72.0 — **our hardware class** |
| #5 70.3 | reflex 4B (Qwen3.5-4B + LoRA, proper scoring rule; shared-state prefill, branches) | MIT | 80.1 / 75.2 (reflex-27b frozen Qwen3.8: 85.8 / 86.2) |
| #6 68.6 | jqv (Qwen3-32B zero-shot; one temperature T = 3.02 fitted on 400 MMLU items) | | 79.3 / 79.0 |
| #8 / #23 | decider-35b-a3b / decider-2b (SFT+RL) | Apache-2.0 | 79.6 / 71.5; 61.2 / 46.6 |
| #10 66.6 | system-one-open (Gemma 4 E2B attention-LoRA) | | 69.5 / 56.7 |
| #12 / #20 | SimpleJev (frozen Qwen3.8-27B / Qwen3.6-35B-A3B) | | 84.7 / 81.1; 79.5 / 67.1 |
| #19–#38 | kev family (Qwen3.5 LoRA + pointer head on typed-decisions) | Apache-2.0 | ≤ 69 / ≤ 51 |
| #30 / #34 | Open-Jev 9B/2B (LoRA + scalar head) | | 71.2 / 63.3 |

Datasets: `LocalLLaMA/typed-decisions` (1,200 train / 400 test); JevBench public subset (72 items, 36 paraphrase pairs; github.com/fstandhartinger/jevbench); `Luni/laya-jev-benchmark`. Scoring: chance-corrected intelligence; speed 100 pts at 0.1 s, −20 per 10×; **the calibration axis is where every open decoder loses 10+ points to Jev.**

## 4. Different abstract approaches worth testing

a. **Diffusion-LM readout** (djev): all answer slots denoised in parallel; its thinking variant has the best calibration of any open system (92.7). Blocked here (no llama.cpp path, ~18 GB).
b. **Verifier / energy scoring**: LLM-as-a-Verifier (2607.05391) expectation over score-token logits for calibrated continuous scores in one pass; Distributional EBMs (2605.18871). Option-conditioned log-likelihood over a shared prefix KV removes symbol binding — the fix for the sub-4B cliff.
c. **Embedding shortlist pre-filter** (SetFit 2209.11055, FastFit 2404.12365): narrow > 20-option choices to top-k, decoder decides among k — the Banking77 regime.
d. **Decision heads on frozen hidden states**: Brier-loss linear probes (2512.22245), token/layer-selective probes (2601.13288); caveat: probes latch onto format (2606.02907).
e. **Hybrid encoder-gate + decoder-decide**: Von / GLiNER2.5 (≤ 20 ms) accept high-margin items; escalate the rest. Metric: work accepted at a fixed error budget.
f. **Distillation from an OPEN teacher** (reflex: proper-scoring-rule LoRA lifts a 4B from 79.0/72.6 to 80.1/75.2); soft labels from frozen Qwen3.8-27B — never Jev.
g. **MoE for cheap prefill**: readout is prefill-only, so 26B-A4B / 35B-A3B give ~30B-class intelligence at ~4B active FLOPs; VRAM, not compute, is the limit.

## 5. Ranked: what to test next on this laptop

1. **Gemma 4 26B-A4B QAT-Q4_0** as a drop-in backbone — SemIf bench + typed-decisions vs 12B Q8; check Q4 does not reopen order-swap errors.
2. **Qwen3.5-9B Q8** head-to-head; if it matches the 12B on symbol binding, bank the 3 GB.
3. **Brier-loss probe on the frozen backbone's answer-position hidden state** — 1,200 typed-decisions rows, 5-fold; vs temperature scaling and jqv's T ≈ 3 trick.
4. **Proper-scoring-rule LoRA distilled from an open 27B teacher** (reflex-style) on Gemma 12B / Qwen 9B; target calibration 72 → 80+.
5. **Encoder gate + embedding shortlist cascade** (Von / GLiNER2.5 + GTE/Qwen3-Embedding + Gemma) on the 72 public JevBench items and Banking77.
